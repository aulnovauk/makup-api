"""
beautygan.py — Unified BeautyGAN Model Wrapper
════════════════════════════════════════════════
Single class that owns both Generator and Discriminator.
Provides a clean top-level API for:
  - Training step execution
  - Inference (no D needed at runtime)
  - Checkpoint save/load of the full model pair
  - Device management
  - Parameter summary

This is the object you hand to the Trainer, and the object
you load for inference. It decouples the training loop from
knowing about individual G/D internals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

import sys
from pathlib import Path as _Path
_pkg_root = str(_Path(__file__).resolve().parent.parent)
if _pkg_root not in sys.path:          # [R7/R16] guarded — idempotent
    sys.path.insert(0, _pkg_root)

from models.generator     import UNetGenerator
from models.discriminator import DualDiscriminator

log = logging.getLogger("MakeupAI.BeautyGAN")


class BeautyGAN(nn.Module):
    """
    Unified BeautyGAN model: Generator + Dual Discriminator.

    Usage — training:
        model = BeautyGAN()
        model.train()
        out = model.generate(source, reference)
        face_pred, local_pred = model.discriminate(out, reference, mask)

    Usage — inference only (no D weights loaded):
        model = BeautyGAN.load_for_inference("best_model.pt")
        result_bgr = model.apply_numpy(source_bgr, reference_bgr)

    Usage — full checkpoint save/load:
        model.save("checkpoint.pt", opt_G, opt_D, scaler, epoch, metrics)
        epoch, metrics = model.load("checkpoint.pt", opt_G, opt_D, scaler)
    """

    def __init__(
        self,
        in_channels:   int = 6,
        out_channels:  int = 3,
        base_features: int = 64,
        n_residual:    int = 6,
    ) -> None:
        super().__init__()
        self.G = UNetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=base_features,
            n_residual=n_residual,
        )
        self.D = DualDiscriminator()
        log.info(f"BeautyGAN initialised — {self.summary()}")

    # ── Forward (generator only — standard nn.Module convention) ───
    def forward(
        self,
        source:    torch.Tensor,   # [B, 3, H, W]
        reference: torch.Tensor,   # [B, 3, H, W]
    ) -> torch.Tensor:             # [B, 3, H, W]
        """Generate makeup output. Calls G only."""
        return self.G(source, reference)

    # ── Discriminator access ────────────────────────────────────────
    def discriminate(
        self,
        image:      torch.Tensor,
        condition:  torch.Tensor,
        mask:       Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run both discriminators. Returns (face_logits, local_logits)."""
        return self.D(image, condition, mask)

    # ── Convenience alias ───────────────────────────────────────────
    def generate(
        self,
        source:    torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return self.G(source, reference)

    # ── Numpy convenience (inference only) ─────────────────────────
    @torch.no_grad()
    def apply_numpy(
        self,
        source_bgr:    "np.ndarray",   # type: ignore[name-defined]
        reference_bgr: "np.ndarray",
        image_size:    int = 256,
    ) -> "np.ndarray":
        """
        End-to-end BGR numpy → makeup → BGR numpy.
        Handles pre/post processing internally.
        Device is inferred from model parameters.

        [R13] Explicitly sets eval() regardless of training state,
        restores original mode via try/finally so dropout/BN behave
        correctly even when called mid-training.
        """
        import numpy as np
        import cv2

        was_training = self.G.training
        self.G.eval()
        try:
            return self._apply_numpy_impl(source_bgr, reference_bgr, image_size)
        finally:
            if was_training:
                self.G.train()

    def _apply_numpy_impl(
        self,
        source_bgr:    "np.ndarray",
        reference_bgr: "np.ndarray",
        image_size:    int,
    ) -> "np.ndarray":
        """Internal implementation — called by apply_numpy after eval() is set."""
        import numpy as np
        import cv2

        device = next(self.G.parameters()).device

        def _pre(img: "np.ndarray") -> torch.Tensor:
            h0, w0 = img.shape[:2]
            itp = cv2.INTER_AREA if (h0 > image_size or w0 > image_size) else cv2.INTER_LINEAR
            r = cv2.resize(img, (image_size, image_size), interpolation=itp)
            r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            r = (r - 0.5) / 0.5
            return torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).to(device)

        def _post(t: torch.Tensor, h: int, w: int) -> "np.ndarray":
            img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = ((img.clip(-1, 1) + 1) / 2 * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            return img

        orig_h, orig_w = source_bgr.shape[:2]
        src = _pre(source_bgr)
        ref = _pre(reference_bgr)
        out = self.G(src, ref)
        return _post(out, orig_h, orig_w)

    # ── Parameter summary ───────────────────────────────────────────
    def summary(self) -> str:
        gp = self.G.n_parameters
        dp = sum(p.numel() for p in self.D.parameters() if p.requires_grad)
        return (
            f"G={gp['trainable']:,} params | "
            f"D={dp:,} params | "
            f"Total={gp['trainable']+dp:,}"
        )

    # ── Checkpoint save ─────────────────────────────────────────────
    def save(
        self,
        path:    Path,
        opt_G:   Optional[torch.optim.Optimizer]  = None,
        opt_D:   Optional[torch.optim.Optimizer]  = None,
        scaler:  Optional[torch.amp.GradScaler]   = None,
        epoch:   int                              = 0,
        metrics: Optional[dict]                   = None,
    ) -> None:
        """Save full training state to disk.

        [R14] Raises RuntimeError if called after load_for_inference()
        replaced self.D with nn.Identity — such a checkpoint would have
        empty D weights and silently break any resume-from-checkpoint.
        """
        if isinstance(self.D, nn.Identity):
            raise RuntimeError(
                "save() called on an inference-only BeautyGAN model "
                "(D was replaced with Identity by load_for_inference). "
                "Cannot save — D weights are gone. Load a full checkpoint "
                "or do not call save() after load_for_inference()."
            )
        payload: dict = {
            "epoch":    epoch,
            "G_state":  self.G.state_dict(),
            "D_state":  self.D.state_dict(),
            "metrics":  metrics or {},
        }
        if opt_G   is not None: payload["opt_G"]  = opt_G.state_dict()
        if opt_D   is not None: payload["opt_D"]  = opt_D.state_dict()
        if scaler  is not None: payload["scaler"] = scaler.state_dict()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        log.info(f"BeautyGAN saved → {path}  (epoch {epoch})")

    # ── Checkpoint load ─────────────────────────────────────────────
    def load(
        self,
        path:       Path,
        opt_G:      Optional[torch.optim.Optimizer] = None,
        opt_D:      Optional[torch.optim.Optimizer] = None,
        scaler:     Optional[torch.amp.GradScaler]  = None,
        device:     Optional[torch.device]           = None,
        strict:     bool                             = True,
    ) -> tuple[int, dict]:
        """
        Load checkpoint. Returns (epoch, metrics).
        Pass opt_G/opt_D/scaler to restore optimiser state (training).
        Omit them for inference-only loading.
        """
        map_dev = device or (next(self.parameters()).device if len(list(self.parameters())) else "cpu")
        ckpt    = torch.load(path, map_location=map_dev, weights_only=False)

        self.G.load_state_dict(ckpt["G_state"], strict=strict)
        if "D_state" in ckpt:
            self.D.load_state_dict(ckpt["D_state"], strict=strict)

        if opt_G  is not None and "opt_G"  in ckpt: opt_G.load_state_dict(ckpt["opt_G"])
        if opt_D  is not None and "opt_D"  in ckpt: opt_D.load_state_dict(ckpt["opt_D"])
        if scaler is not None and "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])

        epoch   = ckpt.get("epoch",   0)
        metrics = ckpt.get("metrics", {})
        log.info(f"BeautyGAN loaded ← {path}  (epoch {epoch})")
        return epoch, metrics

    # ── Inference-only factory ──────────────────────────────────────
    @classmethod
    def load_for_inference(
        cls,
        path:       Path,
        device:     str = "auto",
        image_size: int = 256,
    ) -> "BeautyGAN":
        """
        Load G weights only (no D). Sets eval mode automatically.
        Smaller memory footprint at inference time.
        """
        if device == "auto":
            dev = (
                torch.device("cuda")  if torch.cuda.is_available() else
                torch.device("mps")   if torch.backends.mps.is_available() else
                torch.device("cpu")
            )
        else:
            dev = torch.device(device)

        model = cls()
        model.load(path, device=dev)
        model.D = nn.Identity()   # release discriminator weights from memory
        model.eval()
        model.to(dev)
        log.info(f"BeautyGAN inference-only on {dev}")
        return model

    # ── Context managers for train/eval switching ───────────────────
    def generator_parameters(self):
        """Yield only generator parameters (for opt_G)."""
        return self.G.parameters()

    def discriminator_parameters(self):
        """Yield only discriminator parameters (for opt_D)."""
        return self.D.parameters()
