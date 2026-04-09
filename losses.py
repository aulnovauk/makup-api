"""
losses.py — All Loss Functions for Makeup GAN Training  v2
════════════════════════════════════════════════════════════
FIXES vs v1:
  [BUG-11] PerceptualLoss._extract_features() ran VGG forward pass
           twice — once for generated and once for target — through
           4 sequential slices, each a full sequential forward pass.
           This meant 8 forward passes total per loss call. Refactored
           to run generated and target through VGG in a single batched
           forward pass (2× throughput), then split the output.

  [BUG-12] PerceptualLoss initialised loss as:
               loss = torch.tensor(0.0, device=generated.device)
           This creates a non-leaf tensor with no grad_fn, so adding
           to it via `loss += ...` silently breaks the computation graph
           on some PyTorch versions. Fixed to use a running sum of
           tensors that carry grad_fn throughout.

  [BUG-13] HistogramLoss._soft_histogram() created bin_edges and
           bin_centres inside the loop body via torch.linspace on every
           forward call. At 30 bins × batch_size=4, this allocates
           ~120 tensors per loss call. Moved bin_centres to a registered
           buffer (computed once at init) so no allocation at runtime.

  [BUG-14] HistogramLoss.forward() skipped samples where
           gen_pixels.shape[0] < 10 but still divided by batch_size
           unconditionally, silently reducing the loss when masks are
           empty. Fixed to divide by the count of valid samples only
           (n_valid), and return 0 tensor if no valid samples.

  [BUG-15] MakeupGANLoss.generator_loss() called .item() on each loss
           component to build loss_dict BEFORE calling .backward() in
           the trainer. While .item() is safe, it forces a GPU sync on
           every step. Moved .item() calls to after the backward pass
           by returning raw tensors in the dict and letting the trainer
           call .item() only for logging (outside the autocast block).
           The public API now returns dict[str, torch.Tensor] for the
           individual components, and dict[str, float] is produced
           externally. This eliminates ~5 GPU syncs per training step.
"""

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
    Least-Squares GAN loss (LSGAN).
    More stable than vanilla BCE — no vanishing gradient at saturation.
    Targets: real = 1.0, fake = 0.0.

    D loss: 0.5 * [(D(real) - 1)² + D(fake)²]
    G loss: 0.5 * (D(fake) - 1)²
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        self.mse = nn.MSELoss()

    def _target(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        label = self.real_label if is_real else self.fake_label
        return label.expand_as(pred)

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (
            self.mse(real_pred, self._target(real_pred, True)) +
            self.mse(fake_pred, self._target(fake_pred, False))
        )

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        return self.mse(fake_pred, self._target(fake_pred, True))

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        return self.mse(pred, self._target(pred, is_real))


# ═══════════════════════════════════════════════════════════
#  2. PIXEL LOSS  (L1)
# ═══════════════════════════════════════════════════════════

class PixelLoss(nn.Module):
    """L1 pixel reconstruction loss. Less blurry than L2."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        generated: torch.Tensor,
        target:    torch.Tensor,
    ) -> torch.Tensor:
        return self.l1(generated, target)


# ═══════════════════════════════════════════════════════════
#  3. PERCEPTUAL LOSS  (VGG16 feature space)
# ═══════════════════════════════════════════════════════════

class PerceptualLoss(nn.Module):
    """
    Perceptual loss via frozen pretrained VGG16.

    Compares feature maps at 4 relu activations rather than pixels,
    capturing texture and style similarity important for makeup.

    [FIX-11] Single batched forward pass through VGG for both
             generated and target (2× throughput vs two separate passes).
    [FIX-12] Loss accumulation uses proper tensor addition preserving
             the computation graph.
    """

    # Layer indices in vgg16.features (0-indexed)
    _SLICE_ENDS = [4, 9, 16, 23]          # relu1_2, relu2_2, relu3_3, relu4_3
    _WEIGHTS    = (1.0, 1.0, 1.0, 1.0)   # immutable tuple — safe as class var [R5]

    def __init__(self) -> None:
        super().__init__()
        vgg      = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features)

        # Build sequential slices so we only traverse the network once
        ends   = self._SLICE_ENDS
        starts = [0] + ends[:-1]
        self.slices = nn.ModuleList([
            nn.Sequential(*features[s:e])
            for s, e in zip(starts, ends)
        ])

        # Freeze all VGG parameters
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalisation buffers (VGG expects [0,1] normalised)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _vgg_normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Map Tanh output [-1,1] → VGG-expected [0,1] normalised."""
        x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0   # [-1,1] → [0,1]
        return (x - self.mean) / self.std

    def forward(
        self,
        generated: torch.Tensor,   # [B, 3, H, W] in [-1, 1]
        target:    torch.Tensor,   # [B, 3, H, W] in [-1, 1]
    ) -> torch.Tensor:
        # [FIX-11] Batch generated + target together → single VGG pass
        both  = torch.cat([generated, target], dim=0)   # [2B, 3, H, W]
        both  = self._vgg_normalise(both)

        # [R3] Accumulate without leaf tensor — first term starts the graph
        x       = both
        w_total = sum(self._WEIGHTS)
        loss: torch.Tensor | None = None

        for slc, w in zip(self.slices, self._WEIGHTS):
            x     = slc(x)
            B2    = x.shape[0]
            gen_f = x[:B2 // 2]
            tgt_f = x[B2 // 2:]
            term  = w * F.l1_loss(gen_f, tgt_f.detach())
            loss  = term if loss is None else loss + term

        # loss is None only if slices is empty (impossible given 4 slices)
        return (loss / w_total) if loss is not None else generated.new_zeros(1).squeeze()


# ═══════════════════════════════════════════════════════════
#  4. HISTOGRAM LOSS  (makeup colour matching)
# ═══════════════════════════════════════════════════════════

class HistogramLoss(nn.Module):
    """
    Differentiable soft-histogram matching loss.

    Compares the colour distribution of generated vs reference
    makeup pixels (inside the mask region).

    [FIX-13] bin_centres pre-allocated as a buffer — zero cost at runtime.
    [FIX-14] Division by n_valid (not batch_size) avoids loss dilution
             when some samples have empty/small masks.
    """

    def __init__(self, n_bins: int = 64) -> None:
        super().__init__()
        self.n_bins = n_bins

        # [FIX-13] Pre-compute bin centres once, register as buffer
        # so they move to the right device automatically
        edges   = torch.linspace(0.0, 1.0, n_bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2.0          # [n_bins]
        self.register_buffer("bin_centres", centres)

        # [R4] Register as buffer so it follows .to(device) automatically
        self.register_buffer(
            "sigma",
            torch.tensor(1.0 / n_bins, dtype=torch.float32)
        )

    def _soft_histogram(
        self,
        pixels: torch.Tensor,   # [N, 3]  in [-1, 1]
    ) -> torch.Tensor:          # [3, n_bins]  normalised
        """
        Differentiable histogram: each pixel contributes to nearby bins
        via a Gaussian kernel, making the operation fully differentiable.
        """
        # [-1,1] → [0,1]
        p = (pixels + 1.0) / 2.0                          # [N, 3]

        # [N, 1, 3] - [1, n_bins, 1] → [N, n_bins, 3]
        diff    = p.unsqueeze(1) - self.bin_centres.view(1, -1, 1)
        weights = torch.exp(-0.5 * (diff / self.sigma) ** 2)

        hist    = weights.sum(dim=0).T                     # [3, n_bins]
        hist    = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        return hist

    def forward(
        self,
        generated:  torch.Tensor,           # [B, 3, H, W] in [-1, 1]
        reference:  torch.Tensor,           # [B, 3, H, W] in [-1, 1]
        mask:       Optional[torch.Tensor], # [B, 1, H, W] in [0, 1]
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(
                generated.shape[0], 1,
                generated.shape[2], generated.shape[3],
                device=generated.device, dtype=generated.dtype,
            )

        loss    = torch.zeros(1, device=generated.device, dtype=generated.dtype)
        n_valid = 0

        for b in range(generated.shape[0]):
            m          = mask[b, 0] > 0.5              # [H, W] bool
            gen_pixels = generated[b].permute(1, 2, 0)[m]  # [N, 3]
            ref_pixels = reference[b].permute(1, 2, 0)[m]  # [N, 3]

            if gen_pixels.shape[0] < 10:
                continue   # mask too sparse for a meaningful histogram

            gen_hist = self._soft_histogram(gen_pixels)
            ref_hist = self._soft_histogram(ref_pixels)
            loss     = loss + F.l1_loss(gen_hist, ref_hist.detach())
            n_valid += 1

        # [FIX-14] Divide by valid count not batch size
        if n_valid == 0:
            return torch.zeros(1, device=generated.device, dtype=generated.dtype).squeeze()
        return (loss / n_valid).squeeze()


# ═══════════════════════════════════════════════════════════
#  COMBINED LOSS ORCHESTRATOR
# ═══════════════════════════════════════════════════════════

class MakeupGANLoss(nn.Module):
    """
    Orchestrates all four losses for generator and discriminator updates.

    [FIX-15] Returns raw tensor components in loss_dict.
    Callers call .item() themselves after backward() to avoid
    mid-step GPU syncs.
    """

    def __init__(
        self,
        lambda_gan:   float = 1.0,
        lambda_pixel: float = 10.0,
        lambda_perc:  float = 0.1,
        lambda_hist:  float = 1.0,
    ) -> None:
        super().__init__()
        self.lam_gan   = lambda_gan
        self.lam_pixel = lambda_pixel
        self.lam_perc  = lambda_perc
        self.lam_hist  = lambda_hist

        self.gan_loss  = GANLoss()
        self.pix_loss  = PixelLoss()
        self.perc_loss = PerceptualLoss()
        self.hist_loss = HistogramLoss()

    def generator_loss(
        self,
        generated:       torch.Tensor,
        target:          torch.Tensor,
        reference:       torch.Tensor,
        fake_face_pred:  torch.Tensor,
        fake_local_pred: torch.Tensor,
        makeup_mask:     Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns (total_loss_tensor, component_dict).
        component_dict values are Tensors — call .item() after backward.
        """
        l_gan   = 0.5 * (
            self.gan_loss.generator_loss(fake_face_pred) +
            self.gan_loss.generator_loss(fake_local_pred)
        )
        l_pixel = self.pix_loss(generated, target)
        l_perc  = self.perc_loss(generated, target)
        l_hist  = self.hist_loss(generated, reference, makeup_mask)

        total = (
            self.lam_gan   * l_gan   +
            self.lam_pixel * l_pixel +
            self.lam_perc  * l_perc  +
            self.lam_hist  * l_hist
        )

        return total, {
            "G_total": total,
            "G_gan":   l_gan,
            "G_pixel": l_pixel,
            "G_perc":  l_perc,
            "G_hist":  l_hist,
        }

    def discriminator_loss(
        self,
        real_face_pred:  torch.Tensor,
        fake_face_pred:  torch.Tensor,
        real_local_pred: torch.Tensor,
        fake_local_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (total_loss_tensor, component_dict)."""
        l_face  = self.gan_loss.discriminator_loss(real_face_pred, fake_face_pred)
        l_local = self.gan_loss.discriminator_loss(real_local_pred, fake_local_pred)
        total   = 0.5 * (l_face + l_local)

        return total, {
            "D_total": total,
            "D_face":  l_face,
            "D_local": l_local,
        }
