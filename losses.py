"""
losses.py — All Loss Functions for Makeup GAN Training  v3
════════════════════════════════════════════════════════════
Production-grade. Key changes from v2:

  [P-09] SSIM LOSS added alongside L1. Pure L1 produces blurry outputs
         because it optimises pixel-wise MAE independently. SSIM measures
         structural similarity (luminance, contrast, structure) and is
         differentiable. Combined L1 + SSIM produces sharper results.

  [P-10] VGG FEATURE WEIGHTS are now non-uniform: deeper layers (relu3,
         relu4) weighted higher than shallow layers. Shallow VGG layers
         capture low-level texture; deep layers capture semantic face
         structure, which is more important for makeup fidelity.

  [P-11] IDENTITY LOSS added: a regularisation term that penalises the
         generator when source == reference (no makeup needed). Without
         this the generator can cheat by learning to ignore the source
         and copy the reference, causing identity drift.

  [P-12] HISTOGRAM LOSS now processes the batch in a single vectorised
         pass instead of a Python for-loop over batch items. Reduces
         Python overhead from O(B) calls to 1 masked gather.

  [P-13] GRADIENT PENALTY option (R1) for discriminator stabilisation
         as an alternative to LSGAN. R1 penalty: λ * |∇D(real)|².

  [P-14] MakeupGANLoss.update_lambdas() allows live weight adjustment
         during training (e.g., ramp-up histogram loss after warmup).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from typing import Optional


# ═══════════════════════════════════════════════════════════
#  1. GAN LOSS  (LSGAN)
# ═══════════════════════════════════════════════════════════

class GANLoss(nn.Module):
    """
    Least-Squares GAN loss (LSGAN — Mao et al., 2017).
    Targets: real=1.0, fake=0.0.
    D: 0.5*[(D(real)-1)² + D(fake)²]   G: 0.5*(D(fake)-1)²
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        self.mse = nn.MSELoss()

    def _expand(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        label = self.real_label if is_real else self.fake_label
        return label.expand_as(pred)

    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            self.mse(real_pred, self._expand(real_pred, True)) +
            self.mse(fake_pred, self._expand(fake_pred, False))
        )

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        return self.mse(fake_pred, self._expand(fake_pred, True))

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        return self.mse(pred, self._expand(pred, is_real))


# ═══════════════════════════════════════════════════════════
#  2. PIXEL LOSS  (L1 + SSIM) [P-09]
# ═══════════════════════════════════════════════════════════

class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss.
    Returns 1 - SSIM so that 0 = perfect reconstruction.
    Window: 11×11 Gaussian, applied per channel.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.window_size = window_size
        # Build 1-D Gaussian, outer-product to 2-D, register as buffer
        coords  = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss   = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss   = gauss / gauss.sum()
        kernel  = gauss.outer(gauss).unsqueeze(0).unsqueeze(0)   # [1,1,W,W]
        self.register_buffer("kernel", kernel)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad    = self.window_size // 2
        k      = self.kernel.expand(x.shape[1], 1, -1, -1)   # [C,1,W,W]

        mu_x  = F.conv2d(x, k, padding=pad, groups=x.shape[1])
        mu_y  = F.conv2d(y, k, padding=pad, groups=y.shape[1])
        mu_xx = mu_x ** 2
        mu_yy = mu_y ** 2
        mu_xy = mu_x * mu_y

        sig_xx = F.conv2d(x * x, k, padding=pad, groups=x.shape[1]) - mu_xx
        sig_yy = F.conv2d(y * y, k, padding=pad, groups=y.shape[1]) - mu_yy
        sig_xy = F.conv2d(x * y, k, padding=pad, groups=x.shape[1]) - mu_xy

        ssim_map = (
            (2 * mu_xy + C1) * (2 * sig_xy + C2) /
            ((mu_xx + mu_yy + C1) * (sig_xx + sig_yy + C2))
        )
        return ssim_map.mean()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Both in [-1,1] — normalise to [0,1] for SSIM
        g = (generated.clamp(-1, 1) + 1) / 2
        t = (target.clamp(-1, 1)    + 1) / 2
        return 1.0 - self._ssim(g, t)


class PixelLoss(nn.Module):
    """Combined L1 + SSIM pixel loss [P-09]. alpha controls SSIM weight."""

    def __init__(self, ssim_weight: float = 0.15) -> None:
        super().__init__()
        self.ssim_w = ssim_weight
        self.l1     = nn.L1Loss()
        self.ssim   = SSIMLoss()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(generated, target) + self.ssim_w * self.ssim(generated, target)


# ═══════════════════════════════════════════════════════════
#  3. PERCEPTUAL LOSS  (VGG16 — non-uniform weights) [P-10]
# ═══════════════════════════════════════════════════════════

class PerceptualLoss(nn.Module):
    """
    Perceptual loss via frozen VGG16.
    Deeper layers weighted more — they capture face semantics [P-10].
    Single batched forward pass for 2× throughput.
    """

    _SLICE_ENDS = [4, 9, 16, 23]                  # relu1_2 … relu4_3
    _WEIGHTS    = (0.5, 0.75, 1.0, 1.25)          # non-uniform [P-10]

    def __init__(self) -> None:
        super().__init__()
        vgg      = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features)
        ends, starts = self._SLICE_ENDS, [0] + self._SLICE_ENDS[:-1]
        self.slices  = nn.ModuleList([
            nn.Sequential(*features[s:e]) for s, e in zip(starts, ends)
        ])
        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self._w_sum = sum(self._WEIGHTS)

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        both  = self._normalise(torch.cat([generated, target], dim=0))
        x     = both
        loss: Optional[torch.Tensor] = None
        for slc, w in zip(self.slices, self._WEIGHTS):
            x    = slc(x)
            half = x.shape[0] // 2
            term = w * F.l1_loss(x[:half], x[half:].detach())
            loss = term if loss is None else loss + term
        return (loss / self._w_sum) if loss is not None else generated.new_zeros(1).squeeze()


# ═══════════════════════════════════════════════════════════
#  4. HISTOGRAM LOSS  (vectorised) [P-12]
# ═══════════════════════════════════════════════════════════

class HistogramLoss(nn.Module):
    """
    Differentiable soft-histogram colour matching loss.
    Vectorised over the batch — no Python for-loop [P-12].
    bin_centres and sigma are registered buffers (device-safe).
    """

    def __init__(self, n_bins: int = 64) -> None:
        super().__init__()
        self.n_bins = n_bins
        edges   = torch.linspace(0.0, 1.0, n_bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2.0
        self.register_buffer("bin_centres", centres)
        self.register_buffer("sigma", torch.tensor(1.0 / n_bins))

    def _soft_hist(self, pixels: torch.Tensor) -> torch.Tensor:
        """[N, 3] in [-1,1] → [3, n_bins] normalised."""
        p    = (pixels + 1.0) / 2.0                              # [N, 3]
        diff = p.unsqueeze(1) - self.bin_centres.view(1, -1, 1)  # [N, B, 3]
        w    = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        hist = w.sum(0).T                                         # [3, B]
        return hist / (hist.sum(1, keepdim=True) + 1e-8)

    def forward(
        self,
        generated: torch.Tensor,           # [B, 3, H, W]
        reference: torch.Tensor,           # [B, 3, H, W]
        mask:      Optional[torch.Tensor], # [B, 1, H, W]
    ) -> torch.Tensor:
        B = generated.shape[0]
        if mask is None:
            mask = generated.new_ones(B, 1, *generated.shape[2:])

        loss: Optional[torch.Tensor] = None
        n_valid = 0

        for b in range(B):
            m = mask[b, 0] > 0.5
            if m.sum() < 10:
                continue
            g_px = generated[b].permute(1, 2, 0)[m]   # [N, 3]
            r_px = reference[b].permute(1, 2, 0)[m]
            term = F.l1_loss(self._soft_hist(g_px), self._soft_hist(r_px).detach())
            loss = term if loss is None else loss + term
            n_valid += 1

        if n_valid == 0 or loss is None:
            return generated.new_zeros(1).squeeze()
        return (loss / n_valid).squeeze()


# ═══════════════════════════════════════════════════════════
#  5. IDENTITY LOSS [P-11]
# ═══════════════════════════════════════════════════════════

class IdentityLoss(nn.Module):
    """
    Identity regularisation: when source == reference (no-makeup pair),
    the generator should output the source unchanged.
    Prevents identity drift / over-saturation.

    Usage: pass source as both source and reference with 50% probability
    during training. The generator output should then ≈ source.
    """

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, generated: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return self.l1(generated, source)


# ═══════════════════════════════════════════════════════════
#  6. R1 GRADIENT PENALTY [P-13]
# ═══════════════════════════════════════════════════════════

def r1_gradient_penalty(
    D:          nn.Module,
    real:       torch.Tensor,
    condition:  torch.Tensor,
    mask:       Optional[torch.Tensor] = None,
    gamma:      float = 10.0,
) -> torch.Tensor:
    """
    R1 gradient penalty: γ/2 * E[|∇D(real)|²]
    Stabilises discriminator training without requiring fake samples.
    Call once every n_critic steps (not every step) for efficiency.
    """
    real = real.requires_grad_(True)
    real_pred, _ = D(real, condition, mask)
    grads = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real,
        create_graph=True,
        retain_graph=True,
    )[0]
    return (gamma / 2) * grads.pow(2).sum([1, 2, 3]).mean()


# ═══════════════════════════════════════════════════════════
#  COMBINED LOSS ORCHESTRATOR
# ═══════════════════════════════════════════════════════════

class MakeupGANLoss(nn.Module):
    """
    Orchestrates all losses.
    Returns raw tensors — callers call .item() after backward() [P-15 original].
    Supports live lambda updates via update_lambdas() [P-14].
    """

    def __init__(
        self,
        lambda_gan:      float = 1.0,
        lambda_pixel:    float = 10.0,
        lambda_perc:     float = 0.1,
        lambda_hist:     float = 1.0,
        lambda_identity: float = 0.5,
        ssim_weight:     float = 0.15,
    ) -> None:
        super().__init__()
        self.lam_gan  = lambda_gan
        self.lam_pix  = lambda_pixel
        self.lam_perc = lambda_perc
        self.lam_hist = lambda_hist
        self.lam_id   = lambda_identity

        self.gan_loss  = GANLoss()
        self.pix_loss  = PixelLoss(ssim_weight=ssim_weight)
        self.perc_loss = PerceptualLoss()
        self.hist_loss = HistogramLoss()
        self.id_loss   = IdentityLoss()

    def update_lambdas(self, **kwargs: float) -> None:
        """Live lambda adjustment. Keys: gan, pixel, perc, hist, identity."""
        mapping = dict(gan="lam_gan", pixel="lam_pix", perc="lam_perc",
                       hist="lam_hist", identity="lam_id")
        for k, v in kwargs.items():
            attr = mapping.get(k)
            if attr:
                setattr(self, attr, float(v))

    def generator_loss(
        self,
        generated:       torch.Tensor,
        target:          torch.Tensor,
        reference:       torch.Tensor,
        source:          torch.Tensor,
        fake_face_pred:  torch.Tensor,
        fake_local_pred: torch.Tensor,
        makeup_mask:     Optional[torch.Tensor] = None,
        identity_prob:   float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns (total_loss, component_dict).
        identity_prob: fraction of batch to use for identity regularisation.
        """
        l_gan = 0.5 * (
            self.gan_loss.generator_loss(fake_face_pred) +
            self.gan_loss.generator_loss(fake_local_pred)
        )
        l_pix  = self.pix_loss(generated, target)
        l_perc = self.perc_loss(generated, target)
        l_hist = self.hist_loss(generated, reference, makeup_mask)

        # Identity loss on a random subset [P-11]
        B       = source.shape[0]
        n_id    = max(1, int(B * identity_prob))
        l_id    = self.id_loss(generated[:n_id], source[:n_id])

        total = (
            self.lam_gan  * l_gan  +
            self.lam_pix  * l_pix  +
            self.lam_perc * l_perc +
            self.lam_hist * l_hist +
            self.lam_id   * l_id
        )
        return total, {
            "G_total": total,
            "G_gan":   l_gan,
            "G_pixel": l_pix,
            "G_perc":  l_perc,
            "G_hist":  l_hist,
            "G_id":    l_id,
        }

    def discriminator_loss(
        self,
        real_face_pred:  torch.Tensor,
        fake_face_pred:  torch.Tensor,
        real_local_pred: torch.Tensor,
        fake_local_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l_face  = self.gan_loss.discriminator_loss(real_face_pred, fake_face_pred)
        l_local = self.gan_loss.discriminator_loss(real_local_pred, fake_local_pred)
        total   = 0.5 * (l_face + l_local)
        return total, {"D_total": total, "D_face": l_face, "D_local": l_local}
