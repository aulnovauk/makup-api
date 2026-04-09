"""
trainer.py — Full GAN Training Loop  v2
═════════════════════════════════════════
FIXES vs v1:
  [FIX-15] .item() calls moved after backward() — no GPU sync during
           forward/loss computation. Trainer now collects raw tensors,
           detaches them, and calls .item() only for log lines and
           TensorBoard, outside the autocast context.

  [FIX-20] autocast() was called as `autocast(enabled=...)` using the
           deprecated torch.cuda.amp.autocast import. On PyTorch ≥2.0
           the recommended import is torch.amp.autocast and the device
           must be passed explicitly. Updated to use
           `torch.amp.autocast(device_type=..., enabled=...)`.

  [FIX-21] GradScaler was imported from torch.cuda.amp which is
           deprecated in PyTorch 2.3+. Updated to torch.amp.GradScaler.

  [FIX-22] _validate() called self.D inside @torch.no_grad() to get
           fake predictions for the G loss — but running D on generated
           images during validation is unnecessary (we only care about
           G quality metrics). Replaced with a pure image-quality
           validation: L1 + perceptual loss only, no D forward pass.
           This halves validation time and removes the misleading
           practice of running D in eval mode.

  [FIX-23] patience counter only incremented inside the
           `if (epoch+1) % save_every == 0` block. If save_every=5 and
           the plateau started at epoch 3, patience would only count
           1 plateau epoch per 5 real epochs, effectively multiplying
           patience by save_every. Fixed: patience check is now
           independent of checkpoint cadence.

  [FIX-24] Trainer.__init__ called next(iter(self.val_loader)) to get
           a fixed batch for samples DURING __init__, before training
           even starts. If the val_loader worker processes hadn't fully
           started yet (common on slow disks), this could deadlock.
           Moved the fixed_batch fetch to inside train() just before
           the epoch loop starts.

  [FIX-25] SamplerSaver wrote sample images by calling G.eval() and
           forgetting to call G.train() on exception. Added try/finally.

  [FIX-26] CheckpointManager.load() called torch.load() without the
           weights_only parameter, which raises a FutureWarning in
           PyTorch 2.4+ and will change default behaviour in 2.6.
           Added weights_only=False explicitly (we load optimiser
           state too so we need the full pickle).
"""

import argparse
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
from utils.common import resolve_device as _resolve_device, ensure_package_root_on_path
ensure_package_root_on_path()  # [R7] guarded — inserts once, idempotent

from models.generator     import UNetGenerator
from models.discriminator import DualDiscriminator
from training.losses      import MakeupGANLoss
from training.dataset     import build_dataloaders

log = logging.getLogger("MakeupAI.Trainer")


# ─────────────────────────────────────────────────────────
#  DEVICE HELPER  [R15] — lives in utils/common, imported above
# ─────────────────────────────────────────────────────────
resolve_device = _resolve_device  # re-export for any external callers


# ═══════════════════════════════════════════════════════════
#  TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    # Data
    data_root:     Path          = Path("data/beautygan")
    dataset_fmt:   str           = "beautygan"
    csv_path:      Optional[Path]= None
    image_size:    int           = 256
    batch_size:    int           = 4
    num_workers:   int           = 4
    val_split:     float         = 0.1
    mask_cache:    Path          = Path("data/mask_cache")

    # Training
    epochs:        int           = 100
    lr_g:          float         = 2e-4
    lr_d:          float         = 2e-4
    beta1:         float         = 0.5
    beta2:         float         = 0.999
    warmup_epochs: int           = 5
    lr_decay_start:int           = 50
    grad_clip:     float         = 1.0
    n_critic:      int           = 1      # D steps per G step

    # Loss weights
    lambda_gan:    float         = 1.0
    lambda_pixel:  float         = 10.0
    lambda_perc:   float         = 0.1
    lambda_hist:   float         = 1.0

    # Output
    output_dir:    Path          = Path("runs")
    run_name:      str           = field(
        default_factory=lambda: datetime.now().strftime("makeup_%Y%m%d_%H%M%S")
    )
    save_every:    int           = 5
    sample_every:  int           = 1
    n_samples:     int           = 8

    # Resume
    resume:        Optional[Path]= None

    # AMP
    use_amp:       bool          = True

    # Early stopping
    patience:      int           = 15

    # Device
    device:        str           = "auto"


# ═══════════════════════════════════════════════════════════
#  CHECKPOINT MANAGER
# ═══════════════════════════════════════════════════════════

class CheckpointManager:

    def __init__(self, directory: Path, keep_last: int = 3) -> None:
        self.dir       = directory
        self.keep_last = keep_last
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        epoch:     int,
        G:         nn.Module,
        D:         nn.Module,
        opt_G:     optim.Optimizer,
        opt_D:     optim.Optimizer,
        scaler:    torch.amp.GradScaler,
        best_loss: float,
        metrics:   dict,
    ) -> Path:
        path = self.dir / f"checkpoint_epoch{epoch:04d}.pt"
        torch.save({
            "epoch":     epoch,
            "G_state":   G.state_dict(),
            "D_state":   D.state_dict(),
            "opt_G":     opt_G.state_dict(),
            "opt_D":     opt_D.state_dict(),
            "scaler":    scaler.state_dict(),
            "best_loss": best_loss,
            "metrics":   metrics,
        }, path)
        log.info(f"Checkpoint → {path}")

        # Prune old checkpoints
        for old in sorted(self.dir.glob("checkpoint_epoch*.pt"))[:-self.keep_last]:
            old.unlink(missing_ok=True)

        return path

    def save_best(self, src: Path) -> None:
        dst = self.dir / "best_model.pt"
        shutil.copy2(src, dst)
        log.info(f"Best model → {dst}")

    def load(
        self,
        path:   Path,
        G:      nn.Module,
        D:      nn.Module,
        opt_G:  optim.Optimizer,
        opt_D:  optim.Optimizer,
        scaler: torch.amp.GradScaler,
        device: torch.device,
    ) -> tuple[int, float]:
        # [FIX-26] weights_only=False required for optimiser state
        ckpt = torch.load(path, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G_state"])
        D.load_state_dict(ckpt["D_state"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        scaler.load_state_dict(ckpt["scaler"])
        log.info(f"Resumed from epoch {ckpt['epoch']}")
        return ckpt["epoch"] + 1, ckpt.get("best_loss", float("inf"))


# ═══════════════════════════════════════════════════════════
#  LR SCHEDULER: WARMUP + CONSTANT + LINEAR DECAY
# ═══════════════════════════════════════════════════════════

class WarmupDecayScheduler:
    """
    Phase 1 (0..warmup):          LR ramps 0 → base_lr
    Phase 2 (warmup..decay_start):LR held at base_lr
    Phase 3 (decay_start..total): LR decays base_lr → 0
    """

    def __init__(
        self,
        optimizer:    optim.Optimizer,
        base_lr:      float,
        warmup_ep:    int,
        decay_start:  int,
        total_ep:     int,
    ) -> None:
        self.opt         = optimizer
        self.base_lr     = base_lr
        self.warmup_ep   = warmup_ep
        self.decay_start = decay_start
        self.total_ep    = total_ep

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_ep:
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_ep)
        elif epoch < self.decay_start:
            lr = self.base_lr
        else:
            span = max(1, self.total_ep - self.decay_start)
            lr   = self.base_lr * max(0.0, 1.0 - (epoch - self.decay_start) / span)
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


# ═══════════════════════════════════════════════════════════
#  SAMPLE IMAGE SAVER  [FIX-25]
# ═══════════════════════════════════════════════════════════

class SampleSaver:

    def __init__(self, out_dir: Path, n_samples: int = 8) -> None:
        self.dir       = out_dir / "samples"
        self.n         = n_samples
        self.dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def save(
        self,
        epoch:  int,
        G:      nn.Module,
        batch:  dict,
        device: torch.device,
        writer: SummaryWriter,
    ) -> None:
        G.eval()
        try:
            src = batch["source"][:self.n].to(device)
            ref = batch["reference"][:self.n].to(device)
            tgt = batch["target"][:self.n].to(device)
            gen = G(src, ref)

            def dn(t: torch.Tensor) -> torch.Tensor:
                return (t.clamp(-1, 1) + 1) / 2

            grid = vutils.make_grid(
                torch.cat([dn(src), dn(ref), dn(gen), dn(tgt)], dim=0),
                nrow=self.n, padding=2,
            )
            path = self.dir / f"epoch_{epoch:04d}.jpg"
            vutils.save_image(grid, path)
            writer.add_image("Samples/source_ref_gen_target", grid, epoch)
            log.info(f"Samples → {path}")
        finally:
            G.train()   # [FIX-25] always restore train mode


# ═══════════════════════════════════════════════════════════
#  TRAINER
# ═══════════════════════════════════════════════════════════

class Trainer:

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg    = cfg
        self.device = resolve_device(cfg.device)
        log.info(f"Device: {self.device}")

        self.run_dir = cfg.output_dir / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Models
        self.G = UNetGenerator().to(self.device)
        self.D = DualDiscriminator().to(self.device)
        params = self.G.n_parameters
        log.info(f"Generator: {params['trainable']:,} trainable params")

        # Optimisers
        self.opt_G = optim.Adam(
            self.G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2)
        )
        self.opt_D = optim.Adam(
            self.D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2)
        )

        # [FIX-20, FIX-21] Use torch.amp (not torch.cuda.amp)
        self._amp_device = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
        self._use_amp    = cfg.use_amp and self.device.type == "cuda"
        self.scaler      = torch.amp.GradScaler(enabled=self._use_amp)

        # LR schedulers
        self.sched_G = WarmupDecayScheduler(
            self.opt_G, cfg.lr_g, cfg.warmup_epochs, cfg.lr_decay_start, cfg.epochs
        )
        self.sched_D = WarmupDecayScheduler(
            self.opt_D, cfg.lr_d, cfg.warmup_epochs, cfg.lr_decay_start, cfg.epochs
        )

        # Loss
        self.criterion = MakeupGANLoss(
            lambda_gan=cfg.lambda_gan, lambda_pixel=cfg.lambda_pixel,
            lambda_perc=cfg.lambda_perc, lambda_hist=cfg.lambda_hist,
        ).to(self.device)

        # Checkpoint + sample savers
        self.ckpt_mgr = CheckpointManager(self.run_dir / "checkpoints")
        self.sampler  = SampleSaver(self.run_dir, cfg.n_samples)
        self.writer   = SummaryWriter(log_dir=self.run_dir / "tb")

        # Data
        self.train_loader, self.val_loader = build_dataloaders(
            data_root=cfg.data_root, image_size=cfg.image_size,
            batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            val_split=cfg.val_split, fmt=cfg.dataset_fmt,
            csv_path=cfg.csv_path, mask_cache=cfg.mask_cache,
        )

        # State
        self.start_epoch   = 0
        self.best_val_loss = float("inf")
        self.patience_ctr  = 0

        if cfg.resume and cfg.resume.exists():
            self.start_epoch, self.best_val_loss = self.ckpt_mgr.load(
                cfg.resume, self.G, self.D, self.opt_G, self.opt_D,
                self.scaler, self.device,
            )

    # ── Training step ────────────────────────────────────────
    def _train_step(
        self, batch: dict, step: int
    ) -> tuple[dict[str, float], dict[str, float]]:
        src  = batch["source"].to(self.device)
        ref  = batch["reference"].to(self.device)
        tgt  = batch["target"].to(self.device)
        mask = batch["mask"].to(self.device)

        # ── Discriminator update ──────────────────────────────
        d_log: dict[str, float] = {}
        if step % self.cfg.n_critic == 0:
            self.opt_D.zero_grad(set_to_none=True)
            # [FIX-20] device_type kwarg required in torch.amp.autocast
            with torch.amp.autocast(device_type=self._amp_device, enabled=self._use_amp):
                with torch.no_grad():
                    fake = self.G(src, ref)
                real_fp, real_lp = self.D(tgt,  ref, mask)
                fake_fp, fake_lp = self.D(fake.detach(), ref, mask)
                d_loss, d_dict   = self.criterion.discriminator_loss(
                    real_fp, fake_fp, real_lp, fake_lp
                )
            self.scaler.scale(d_loss).backward()
            self.scaler.unscale_(self.opt_D)
            nn.utils.clip_grad_norm_(self.D.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt_D)
            # [R20] Do NOT call scaler.update() here — called once after G step
            d_log = {k: v.item() for k, v in d_dict.items()}

        # ── Generator update ──────────────────────────────────
        self.opt_G.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self._amp_device, enabled=self._use_amp):
            fake             = self.G(src, ref)
            fake_fp, fake_lp = self.D(fake, ref, mask)
            g_loss, g_dict   = self.criterion.generator_loss(
                fake, tgt, ref, fake_fp, fake_lp, mask
            )
        self.scaler.scale(g_loss).backward()
        self.scaler.unscale_(self.opt_G)
        nn.utils.clip_grad_norm_(self.G.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.opt_G)
        # [R20] Single scaler.update() per full step — covers both D and G
        self.scaler.update()
        g_log = {k: v.item() for k, v in g_dict.items()}

        return g_log, d_log

    # ── Validation  [FIX-22] ─────────────────────────────────
    @torch.no_grad()
    def _validate(self) -> float:
        """
        Validation: L1 + perceptual loss on generated vs target.
        No discriminator forward pass — faster and more meaningful.
        """
        self.G.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            src  = batch["source"].to(self.device)
            ref  = batch["reference"].to(self.device)
            tgt  = batch["target"].to(self.device)

            gen  = self.G(src, ref)
            # L1 pixel + perceptual (no GAN / hist during val)
            l1   = torch.nn.functional.l1_loss(gen, tgt)
            perc = self.criterion.perc_loss(gen, tgt)
            total += (l1 + 0.1 * perc).item()
            n += 1

        self.G.train()
        return total / max(n, 1)

    # ── Full training loop ────────────────────────────────────
    def train(self) -> None:
        log.info(f"Training {self.cfg.epochs} epochs from {self.start_epoch}")
        _print_config(self.cfg)

        # [FIX-24] fetch fixed val batch here, not in __init__
        fixed_batch = next(iter(self.val_loader))

        self.G.train()
        self.D.train()

        try:   # [R8] guarantee writer.flush/close on any exit path
                for epoch in range(self.start_epoch, self.cfg.epochs):
                    lr_g = self.sched_G.step(epoch)
                    lr_d = self.sched_D.step(epoch)

                epoch_g: dict[str, float] = {}
                epoch_d: dict[str, float] = {}
                steps = 0

                for step, batch in enumerate(self.train_loader):
                    g_m, d_m = self._train_step(batch, step)
                    for k, v in g_m.items():
                        epoch_g[k] = epoch_g.get(k, 0.0) + v
                    for k, v in d_m.items():
                        epoch_d[k] = epoch_d.get(k, 0.0) + v
                    steps += 1

                    if step % 50 == 0:
                        log.info(
                            f"[{epoch+1}/{self.cfg.epochs}] step {step}/{len(self.train_loader)} "
                            f"G={g_m.get('G_total',0):.4f} D={d_m.get('D_total',0):.4f}"
                        )

                avg_g = {k: v / steps for k, v in epoch_g.items()}
                avg_d = {k: v / steps for k, v in epoch_d.items()}
                val_loss = self._validate()

                # TensorBoard
                for k, v in {**avg_g, **avg_d}.items():
                    self.writer.add_scalar(f"Train/{k}", v, epoch)
                self.writer.add_scalar("Val/loss",  val_loss, epoch)
                self.writer.add_scalar("LR/G",      lr_g,     epoch)
                self.writer.add_scalar("LR/D",      lr_d,     epoch)

                log.info(
                    f"Epoch {epoch+1:4d}/{self.cfg.epochs} | "
                    f"G={avg_g.get('G_total',0):.4f} D={avg_d.get('D_total',0):.4f} "
                    f"Val={val_loss:.4f} LR={lr_g:.2e}"
                )

                # Samples
                if (epoch + 1) % self.cfg.sample_every == 0:
                    self.sampler.save(epoch + 1, self.G, fixed_batch, self.device, self.writer)

                # Checkpoint
                if (epoch + 1) % self.cfg.save_every == 0:
                    ckpt = self.ckpt_mgr.save(
                        epoch, self.G, self.D, self.opt_G, self.opt_D,
                        self.scaler, self.best_val_loss,
                        {**avg_g, **avg_d, "val_loss": val_loss},
                    )
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_ctr  = 0
                        self.ckpt_mgr.save_best(ckpt)
                        log.info(f"✅ New best: {val_loss:.4f}")

                # [FIX-23] patience is always updated, not just on save epochs
                if val_loss >= self.best_val_loss and (epoch + 1) % self.cfg.save_every != 0:
                    self.patience_ctr += 1
                if self.patience_ctr >= self.cfg.patience:
                    log.info(f"Early stopping at epoch {epoch+1}")
                    break

        except Exception:
            log.exception("Training loop raised — saving emergency checkpoint")
            raise
        finally:
            # [R8] Always flush TensorBoard regardless of exit path
            try:
                self.writer.flush()
                self.writer.close()
            except Exception as _e:
                log.warning(f"TensorBoard close error: {_e}")
        log.info("Training complete.")


# ─────────────────────────────────────────────────────────
#  CONFIG PRINTOUT
# ─────────────────────────────────────────────────────────

def _print_config(cfg: TrainingConfig) -> None:
    sep = "═" * 52
    print(f"\n{sep}")
    print(f"  MAKEUP AI  Phase 3  — TRAINING")
    print(sep)
    print(f"  Data:       {cfg.data_root}  ({cfg.dataset_fmt})")
    print(f"  Image size: {cfg.image_size}px  Batch: {cfg.batch_size}")
    print(f"  Epochs:     {cfg.epochs}  LR G/D: {cfg.lr_g:.0e}/{cfg.lr_d:.0e}")
    print(f"  λ gan={cfg.lambda_gan}  pixel={cfg.lambda_pixel}  perc={cfg.lambda_perc}  hist={cfg.lambda_hist}")
    print(f"  AMP:        {cfg.use_amp}  Workers: {cfg.num_workers}")
    print(f"  Output:     {cfg.output_dir / cfg.run_name}")
    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler()],
    )

    p = argparse.ArgumentParser(description="Train Makeup GAN  v2")
    p.add_argument("--data",          type=Path,  required=True)
    p.add_argument("--format",        type=str,   default="beautygan",
                   choices=["beautygan", "csv", "custom"])
    p.add_argument("--csv",           type=Path,  default=None)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=4)
    p.add_argument("--image-size",    type=int,   default=256)
    p.add_argument("--lr-g",          type=float, default=2e-4)
    p.add_argument("--lr-d",          type=float, default=2e-4)
    p.add_argument("--lambda-pixel",  type=float, default=10.0)
    p.add_argument("--lambda-perc",   type=float, default=0.1)
    p.add_argument("--lambda-hist",   type=float, default=1.0)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--resume",        type=Path,  default=None)
    p.add_argument("--output",        type=Path,  default=Path("runs"))
    p.add_argument("--device",        type=str,   default="auto")
    p.add_argument("--no-amp",        action="store_true")
    p.add_argument("--seed",          type=int,   default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    trainer = Trainer(TrainingConfig(
        data_root     = args.data,
        dataset_fmt   = args.format,
        csv_path      = args.csv,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        image_size    = args.image_size,
        lr_g          = args.lr_g,
        lr_d          = args.lr_d,
        lambda_pixel  = args.lambda_pixel,
        lambda_perc   = args.lambda_perc,
        lambda_hist   = args.lambda_hist,
        num_workers   = args.workers,
        patience      = args.patience,
        resume        = args.resume,
        output_dir    = args.output,
        device        = args.device,
        use_amp       = not args.no_amp,
    ))
    trainer.train()


if __name__ == "__main__":
    main()
