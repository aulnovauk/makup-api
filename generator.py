"""
generator.py — U-Net Generator for Neural Makeup Transfer  v3
═══════════════════════════════════════════════════════════════
Production-grade. Key changes from v2:

  [P-01] OUTPUT SIZE FIX: Final ConvTranspose2d was doubling H×W
         (256→512). Output must match input resolution. Replaced with
         ReflectionPad + Conv2d (no stride) so output is always H×W.

  [P-02] SPECTRAL NORM on all encoder Conv layers for training
         stability — prevents gradient explosion without LR tuning.

  [P-03] SELF-ATTENTION at bottleneck for global face coherence.
         Applied after residual blocks at lowest spatial resolution.

  [P-04] CHANNEL CAP at 512 — each encoder stage is min(bf*n, 512)
         so the architecture matches BeautyGAN paper exactly.

  [P-05] GRADIENT CHECKPOINTING support via enable_gradient_checkpointing()
         for training at 512px+ without OOM.

  [P-06] DROPOUT now uses configurable dropout_rate (default 0.5)
         via standard nn.Dropout — correctly disabled at eval().

  [P-07] SKIP-CAT HELPER with bilinear resize fallback for odd dims.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

log = logging.getLogger("MakeupAI.Generator")


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def _make_skip_cat(dec: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
    """Concat decoder + encoder features. Resize dec if spatial dims differ."""
    if dec.shape[2:] != enc.shape[2:]:
        dec = F.interpolate(dec, size=enc.shape[2:], mode="bilinear", align_corners=False)
    return torch.cat([dec, enc], dim=1)


# ═══════════════════════════════════════════════════════════
#  BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════

class ConvINLeaky(nn.Module):
    """Stride-2 encoder block: SpectralNorm(Conv) → InstanceNorm → LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, norm: bool = True, spec_norm: bool = True) -> None:
        super().__init__()
        conv: nn.Module = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm)
        if spec_norm:
            conv = nn.utils.spectral_norm(conv)
        layers: list[nn.Module] = [conv]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpConvINReLU(nn.Module):
    """Stride-2 decoder block: ConvTranspose → InstanceNorm → ReLU → Dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """ReflectionPad → Conv → IN → ReLU → ReflectionPad → Conv → IN + residual."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SelfAttention(nn.Module):
    """
    SAGAN-style self-attention (Zhang et al., 2019).
    Applied at bottleneck for global spatial coherence.
    gamma (learned scalar) starts at 0 so attention is additive.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(ch, ch // 8, 1, bias=False)
        self.k = nn.Conv2d(ch, ch // 8, 1, bias=False)
        self.v = nn.Conv2d(ch, ch,       1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        q = self.q(x).view(B, -1, N).permute(0, 2, 1)   # [B, N, C/8]
        k = self.k(x).view(B, -1, N)                     # [B, C/8, N]
        v = self.v(x).view(B, -1, N)                     # [B, C, N]
        scale = (C // 8) ** 0.5
        attn  = torch.softmax(torch.bmm(q, k) / scale, dim=-1)  # [B, N, N]
        out   = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


# ═══════════════════════════════════════════════════════════
#  U-NET GENERATOR
# ═══════════════════════════════════════════════════════════

class UNetGenerator(nn.Module):
    """
    U-Net Generator for makeup transfer.

    Input:   [source(3) ‖ reference(3)] → [6, H, W]  in [-1, 1]
    Output:  [3, H, W]  in [-1, 1]  — SAME spatial size as input [P-01]

    Encoder channel flow (bf=64):
        6 → 64 → 128 → 256 → 512 → 512 → 512 → 512
    Bottleneck: n_residual ResBlocks + SelfAttention
    Decoder: symmetric with skip concatenation

    Args:
        in_channels:   6    (source + reference concatenated)
        out_channels:  3    (RGB)
        base_features: 64   (bf; stages capped at 512)
        n_residual:    6    (bottleneck residual blocks)
        dropout_rate:  0.5  (first 3 decoder blocks)
        use_attention: True (SelfAttention at bottleneck)
        use_spec_norm: True (SpectralNorm on encoder convs)
    """

    def __init__(
        self,
        in_channels:   int   = 6,
        out_channels:  int   = 3,
        base_features: int   = 64,
        n_residual:    int   = 6,
        dropout_rate:  float = 0.5,
        use_attention: bool  = True,
        use_spec_norm: bool  = True,
    ) -> None:
        super().__init__()
        bf  = base_features
        sn  = use_spec_norm
        dr  = dropout_rate

        def c(n: int) -> int:          # channel count capped at 512 [P-04]
            return min(bf * n, 512)

        self._use_grad_ckpt = False

        # ── Encoder ────────────────────────────────────────────────
        self.e1 = ConvINLeaky(in_channels, bf,    norm=False, spec_norm=sn)
        self.e2 = ConvINLeaky(bf,          c(2),  spec_norm=sn)
        self.e3 = ConvINLeaky(c(2),        c(4),  spec_norm=sn)
        self.e4 = ConvINLeaky(c(4),        c(8),  spec_norm=sn)
        self.e5 = ConvINLeaky(c(8),        c(8),  spec_norm=sn)
        self.e6 = ConvINLeaky(c(8),        c(8),  spec_norm=sn)
        self.e7 = ConvINLeaky(c(8),        c(8),  spec_norm=sn)

        # ── Bottleneck ─────────────────────────────────────────────
        bottleneck_layers: list[nn.Module] = [
            ResidualBlock(c(8)) for _ in range(n_residual)
        ]
        if use_attention:
            bottleneck_layers.append(SelfAttention(c(8)))
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # ── Decoder ────────────────────────────────────────────────
        self.d1 = UpConvINReLU(c(8),          c(8),  dropout_rate=dr)
        self.d2 = UpConvINReLU(c(8) + c(8),   c(8),  dropout_rate=dr)
        self.d3 = UpConvINReLU(c(8) + c(8),   c(8),  dropout_rate=dr)
        self.d4 = UpConvINReLU(c(8) + c(8),   c(8))
        self.d5 = UpConvINReLU(c(8) + c(8),   c(4))
        self.d6 = UpConvINReLU(c(4) + c(2),   c(2))
        self.d7 = UpConvINReLU(c(2) + bf,     bf)

        # ── Output head — same H×W as input [P-01] ─────────────────
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(bf, out_channels, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
        )

        self._init_weights()

    # ── Initialisation ─────────────────────────────────────────────
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                w = getattr(m, "weight_orig", m.weight)   # unwrap spectral norm
                nn.init.normal_(w, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Gradient checkpointing [P-05] ──────────────────────────────
    def enable_gradient_checkpointing(self) -> None:
        """Checkpoint the bottleneck to save memory at 512px training."""
        self._use_grad_ckpt = True
        log.info("UNetGenerator: gradient checkpointing ON")

    # ── Forward ────────────────────────────────────────────────────
    def forward(self, source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source:    [B, 3, H, W] in [-1, 1] — no-makeup face
            reference: [B, 3, H, W] in [-1, 1] — makeup reference face
        Returns:
            [B, 3, H, W] in [-1, 1] — source face with makeup applied
        """
        x = torch.cat([source, reference], dim=1)

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        if self._use_grad_ckpt and self.training:
            b = grad_checkpoint(self.bottleneck, e7, use_reentrant=False)
        else:
            b = self.bottleneck(e7)

        d1 = self.d1(b)
        d2 = self.d2(_make_skip_cat(d1, e6))
        d3 = self.d3(_make_skip_cat(d2, e5))
        d4 = self.d4(_make_skip_cat(d3, e4))
        d5 = self.d5(_make_skip_cat(d4, e3))
        d6 = self.d6(_make_skip_cat(d5, e2))
        d7 = self.d7(_make_skip_cat(d6, e1))

        return self.out_conv(d7)   # [B, 3, H, W] — same size as input

    # ── Utilities ──────────────────────────────────────────────────
    @property
    def n_parameters(self) -> dict[str, int]:
        total = trainable = 0
        for p in self.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        return {"total": total, "trainable": trainable}
