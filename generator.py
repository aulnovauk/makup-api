"""
generator.py — U-Net Generator for Neural Makeup Transfer  v2
═══════════════════════════════════════════════════════════════
FIXES vs v1:
  [BUG-01] forward() had a dangling conditional expression on the output
           layer that evaluated to a dead branch every single call — the
           condition `d7.shape[1] * 2 == self.out_conv[0].in_channels`
           is always False at runtime (d7 has bf channels, out_conv
           expects bf*2 because of the skip cat). The intended U-Net
           output skip was never applied, making the last decoder block
           useless. Fixed by removing the ternary and always applying
           the correct skip cat from e1.

  [BUG-02] out_conv head accepted bf*2 channels but d7 only produces
           bf channels — input channel mismatch. The correct pattern is
           cat([d7, e1_half]) so out_conv in_channels = bf + bf/2? No —
           standard U-Net: cat d7 with the MATCHING encoder skip e1
           which also has bf channels → bf+bf = bf*2. Fixed encoder
           so e1 outputs bf channels and out_conv takes bf*2.

  [BUG-03] `from typing import List` imported but never used.

  [BUG-04] _init_weights() accessed m.bias on InstanceNorm2d without
           checking m.bias is not None — when affine=False bias is None
           and this raises AttributeError. Guarded with `if m.bias`.

  [BUG-05] e7 (bottleneck input) has norm=False which means no
           InstanceNorm before the residual blocks. This destabilises
           the residual path. Added norm=True on e7 to match all other
           encoder blocks (standard BeautyGAN architecture).

  [BUG-06] count_parameters() was a public method with no docstring
           and could be called on eval() models. Made it a proper
           @property and added total/trainable split.
"""

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════
#  BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════

class ConvInstanceNorm(nn.Module):
    """Conv2d → InstanceNorm2d → Activation (encoder block)."""

    def __init__(
        self,
        in_ch:      int,
        out_ch:     int,
        kernel:     int  = 4,
        stride:     int  = 2,
        padding:    int  = 1,
        norm:       bool = True,
        activation: str  = "leaky",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=not norm)
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        if activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "tanh":
            layers.append(nn.Tanh())
        # activation == "none" → no activation appended
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpConvInstanceNorm(nn.Module):
    """ConvTranspose2d → InstanceNorm2d → ReLU (+ optional Dropout)."""

    def __init__(
        self,
        in_ch:   int,
        out_ch:  int,
        dropout: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual block at bottleneck.
    ReflectionPad avoids border artefacts.
    Two conv + norm + relu, residual addition on exit.
    """

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


# ═══════════════════════════════════════════════════════════
#  U-NET GENERATOR
# ═══════════════════════════════════════════════════════════

class UNetGenerator(nn.Module):
    """
    U-Net Generator for makeup transfer.

    Input:  [source (3) || reference (3)] → [6, H, W]
    Output: source face with reference makeup applied → [3, H, W]

    Channel flow (base_features = bf = 64):
      Encoder:  6→bf→bf2→bf4→bf8→bf8→bf8→bf8
      Bottleneck: bf8 residual blocks
      Decoder:  (bf8+bf8)→bf8 → … → (bf+bf)→3   [skip cats]
      Output head: ConvTranspose(bf*2 → 3) + Tanh

    Args:
        in_channels:   6   (source 3 + reference 3)
        out_channels:  3   (RGB output)
        base_features: 64  (doubled each encoder stage, capped at 512)
        n_residual:    6   residual blocks at bottleneck
    """

    def __init__(
        self,
        in_channels:   int = 6,
        out_channels:  int = 3,
        base_features: int = 64,
        n_residual:    int = 6,
    ) -> None:
        super().__init__()
        bf = base_features

        # ── Encoder ────────────────────────────────────────
        # e1: no norm on first layer (standard GAN practice)
        self.e1 = ConvInstanceNorm(in_channels, bf,     norm=False, activation="leaky")
        self.e2 = ConvInstanceNorm(bf,          bf * 2, activation="leaky")
        self.e3 = ConvInstanceNorm(bf * 2,      bf * 4, activation="leaky")
        self.e4 = ConvInstanceNorm(bf * 4,      bf * 8, activation="leaky")
        self.e5 = ConvInstanceNorm(bf * 8,      bf * 8, activation="leaky")
        self.e6 = ConvInstanceNorm(bf * 8,      bf * 8, activation="leaky")
        # [FIX-05] e7 uses norm=True to stabilise residual path input
        self.e7 = ConvInstanceNorm(bf * 8,      bf * 8, norm=True,  activation="leaky")

        # ── Bottleneck ─────────────────────────────────────
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(bf * 8) for _ in range(n_residual)]
        )

        # ── Decoder (U-Net skip connections) ───────────────
        # Each decoder block receives cat(d_prev, e_skip) → 2× channels
        self.d1 = UpConvInstanceNorm(bf * 8,     bf * 8, dropout=True)
        self.d2 = UpConvInstanceNorm(bf * 8 * 2, bf * 8, dropout=True)
        self.d3 = UpConvInstanceNorm(bf * 8 * 2, bf * 8, dropout=True)
        self.d4 = UpConvInstanceNorm(bf * 8 * 2, bf * 8)
        self.d5 = UpConvInstanceNorm(bf * 8 * 2, bf * 4)
        self.d6 = UpConvInstanceNorm(bf * 4 * 2, bf * 2)
        self.d7 = UpConvInstanceNorm(bf * 2 * 2, bf)

        # ── Output head ────────────────────────────────────
        # [FIX-01, FIX-02] cat([d7, e1]) gives bf + bf = bf*2 channels
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(bf * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise Conv weights ~ N(0, 0.02) — standard GAN practice."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                # [FIX-04] guard: affine=False → weight/bias are None
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        source:    torch.Tensor,   # [B, 3, H, W]  no-makeup face
        reference: torch.Tensor,   # [B, 3, H, W]  makeup reference
    ) -> torch.Tensor:             # [B, 3, H, W]  with makeup applied
        x = torch.cat([source, reference], dim=1)   # [B, 6, H, W]

        # Encoder — store all intermediate feature maps for skip connections
        e1 = self.e1(x)    # [B, bf,    H/2,  W/2]
        e2 = self.e2(e1)   # [B, bf*2,  H/4,  W/4]
        e3 = self.e3(e2)   # [B, bf*4,  H/8,  W/8]
        e4 = self.e4(e3)   # [B, bf*8,  H/16, W/16]
        e5 = self.e5(e4)   # [B, bf*8,  H/32, W/32]
        e6 = self.e6(e5)   # [B, bf*8,  H/64, W/64]
        e7 = self.e7(e6)   # [B, bf*8,  H/128,W/128]

        # Bottleneck — residual blocks preserve structure
        b  = self.bottleneck(e7)   # [B, bf*8, H/128, W/128]

        # Decoder — U-Net skip connections via torch.cat
        d1 = self.d1(b)                          # [B, bf*8, H/64,  W/64]
        d2 = self.d2(torch.cat([d1, e6], dim=1)) # [B, bf*8, H/32,  W/32]
        d3 = self.d3(torch.cat([d2, e5], dim=1)) # [B, bf*8, H/16,  W/16]
        d4 = self.d4(torch.cat([d3, e4], dim=1)) # [B, bf*8, H/8,   W/8]
        d5 = self.d5(torch.cat([d4, e3], dim=1)) # [B, bf*4, H/4,   W/4]
        d6 = self.d6(torch.cat([d5, e2], dim=1)) # [B, bf*2, H/2,   W/2]
        d7 = self.d7(torch.cat([d6, e1], dim=1)) # [B, bf,   H,     W]

        # [FIX-01, FIX-02] Always cat d7 with e1 to get bf*2 for out_conv
        # This gives the output head direct access to early edge features
        return self.out_conv(torch.cat([d7, e1], dim=1))  # [B, 3, H*2, W*2]
        # Note: final upsample doubles spatial dims back to input size

    @property
    def n_parameters(self) -> dict[str, int]:
        """Return total and trainable parameter counts.
        Iterates parameters() exactly once (not twice). [R2]
        """
        total, trainable = 0, 0
        for p in self.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        return {"total": total, "trainable": trainable}
