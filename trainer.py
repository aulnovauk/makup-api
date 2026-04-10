"""
trainer.py — Full GAN Training Loop  v3
═════════════════════════════════════════
Production-grade. Key changes from v2:

  [P-15] SEED EVERYTHING: seeds torch, numpy, random, and CUDA for
         full reproducibility. Passed via --seed CLI argument.

  [P-16] CONFIG SAVED AS JSON at run start so every experiment is
         fully reproducible from its output directory alone.

  [P-17] STEP TIMER logged every N steps — shows samples/sec and
         estimated time remaining to end of epoch.

  [P-18] EMERGENCY CHECKPOINT: on exception, saves current G/D
         state before re-raising so training is never lost entirely.

  [P-19] IDENTITY LOSS RAMP-UP: lambda_identity starts at 0 and
         linearly ramps to target over warmup_epochs, then held constant.
         This prevents identity loss dominating early training.

  [P-20] VAL LOSS HISTORY tracked per epoch. Written to a CSV for
         plotting without needing TensorBoard.

  [P-21] HISTOGRAM LOSS RAMP-UP: lambda_hist starts at 0 and ramps
         up after warmup — histogram matching is meaningless before
         the generator learns basic face structure.

  [P-22] --mask-cache argument wired through to build_dataloaders.

  [P-23] GradScaler device string now derived correctly for both
         CUDA and CPU (CPU scaler is no-op but valid).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
from pathlib import Path as _Path
_pkg = str(_Path(__file__).resolve().parent)
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)

from common   import resolve_device, ensure_package_root_on_path
ensure_package_root_on_path()

from generator import UNetGenerator
from losses    import MakeupGANLoss
from dataset   import build_dataloaders

log = logging.getLogger("MakeupAI.Trainer")

# ── Try to import DualDiscriminator ──────────────────────────
try:
    from discriminator import DualDiscriminator
except ImportError:
    DualDiscriminator = None  # type: ignore[assignment,misc]
    log.warning("discriminator.py not found — D will be a placeholder")


# ═══════════════════════════════════════════════════════════
#  REPRODUCIBILITY [P-15]
# ═══════════════════════════════════════════════════════════

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    log.info(f"Seed: {seed}")


# ═══════════════════════════════════════════════════════════
#  TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    # Data
    data_root:      Path           = Path("data/beautygan")
    dataset_fmt:    str            = "beautygan"
    csv_path:       Optional[Path] = None
    image_size:     int            = 256
    batch_size:     int            = 4
    num_workers:    int            = 4
    val_split:      float          = 0.1
    mask_cache:     Path           = Path("data/mask_cache")

    # Training
    epochs:         int            = 100
    lr_g:           float          = 2e-4
    lr_d:           float          = 2e-4
    beta1:          float          = 0.5
    beta2:          float          = 0.999
    warmup_epochs:  int            = 5
    lr_decay_start: int            = 50
    grad_clip:      float          = 1.0
    n_critic:       int            = 1
    seed:           int            = 42

    # Loss weights
    lambda_gan:      float         = 1.0
    lambda_pixel:    float         = 10.0
    lambda_perc:     float         = 0.1
    lambda_hist:     float         = 1.0
    lambda_identity: float         = 0.5

    # Output
    output_dir:     Path           = Path("runs")
    run_name:       str            = field(
        default_factory=lambda: datetime.now().strftime("makeup_%Y%m%d_%H%M%S")
    )
    save_every:     int            = 5
    sample_every:   int            = 1
    n_samples:      int            = 8

    # Resume
    resume:         Optional[Path] = None

    # AMP
    use_amp:        bool           = True

    # Early stopping
    patience:       int            = 15

    # Device
    device:         str            = "auto"

    # Architecture
    use_attention:  bool           = True
    use_spec_norm:  bool           = True
    grad_ckpt:      bool           = False


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
        # Prune old checkpoints (keep best_model.pt always)
        candidates = sorted(self.dir.glob("checkpoint_epoch*.pt"))
        for old in candidates[:-self.keep_last]:
            old.unlink(missing_ok=True)
        return path

    def save_best(self, src: Path) -> None:
        dst = self.dir / "best_model.pt"
        shutil.copy2(src, dst)
        log.info(f"Best model → {dst}")

    def save_emergency(self, G: nn.Module, D: nn.Module, epoch: int) -> Path:
        """Save weights-only emergency snapshot on unexpected crash [P-18]."""
        path = self.dir / f"emergency_epoch{epoch:04d}.pt"
        torch.save({"epoch": epoch, "G_state": G.state_dict(), "D_state": D.state_dict()}, path)
        log.warning(f"Emergency checkpoint → {path}")
        return path

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
        ckpt = torch.load(path, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G_state"])
        D.load_state_dict(ckpt["D_state"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        scaler.load_state_dict(ckpt["scaler"])
        epoch = ckpt.get("epoch", 0)
        log.info(f"Resumed from epoch {epoch}")
        return epoch + 1, ckpt.get("best_loss", float("inf"))


# ═══════════════════════════════════════════════════════════
#  LR SCHEDULER: WARMUP + CONSTANT + LINEAR DECAY
# ═══════════════════════════════════════════════════════════

class WarmupDecayScheduler:
    """Warmup → constant → linear decay to 0."""

    def __init__(
        self,
        optimizer:   optim.Optimizer,
        base_lr:     float,
        warmup_ep:   int,
        decay_start: int,
        total_ep:    int,
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
#  SAMPLE IMAGE SAVER
# ═══════════════════════════════════════════════════════════

class SampleSaver:

    def __init__(self, out_dir: Path, n_samples: int = 8) -> None:
        self.dir = out_dir / "samples"
        self.n   = n_samples
        self.dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def save(self, epoch: int, G: nn.Module, batch: dict,
             device: torch.device, writer: SummaryWriter) -> None:
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
            writer.add_image("Samples/src_ref_gen_tgt", grid, epoch)
            log.info(f"Samples → {path}")
        finally:
            G.train()


# ═══════════════════════════════════════════════════════════
#  VAL LOSS CSV WRITER [P-20]
# ═══════════════════════════════════════════════════════════

class ValLossCSV:

    def __init__(self, path: Path) -> None:
        self.path = path
        self._wrote_header = False

    def write(self, epoch: int, metrics: dict[str, float]) -> None:
        write_header = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            fieldnames = ["epoch"] + sorted(metrics.keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow({"epoch": epoch, **metrics})


# ═══════════════════════════════════════════════════════════
#  TRAINER
# ═══════════════════════════════════════════════════════════

class Trainer:

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg    = cfg
        seed_everything(cfg.seed)
        self.device = resolve_device(cfg.device)
        log.info(f"Device: {self.device}")

        self.run_dir = cfg.output_dir / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config for reproducibility [P-16]
        cfg_path = self.run_dir / "config.json"
        cfg_dict = {k: str(v) if isinstance(v, Path) else v
                    for k, v in asdict(cfg).items()}
        cfg_path.write_text(json.dumps(cfg_dict, indent=2))
        log.info(f"Config → {cfg_path}")

        # Models
        self.G = UNetGenerator(
            use_attention=cfg.use_attention,
            use_spec_norm=cfg.use_spec_norm,
        ).to(self.device)

        if cfg.grad_ckpt:
            self.G.enable_gradient_checkpointing()

        if DualDiscriminator is not None:
            self.D = DualDiscriminator().to(self.device)
        else:
            # Fallback stub so training loop doesn't crash
            self.D = nn.Identity()  # type: ignore[assignment]
            log.warning("Using Identity discriminator — GAN loss will be 0")

        params = self.G.n_parameters
        log.info(f"Generator: {params['trainable']:,} trainable params")

        # Optimisers
        self.opt_G = optim.Adam(self.G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
        self.opt_D = optim.Adam(
            self.D.parameters() if not isinstance(self.D, nn.Identity) else [torch.zeros(1, requires_grad=True)],
            lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2)
        )

        # AMP [P-23]
        self._amp_device = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
        self._use_amp    = cfg.use_amp and self.device.type == "cuda"
        self.scaler      = torch.amp.GradScaler(device=self._amp_device, enabled=self._use_amp)

        # Schedulers
        self.sched_G = WarmupDecayScheduler(self.opt_G, cfg.lr_g, cfg.warmup_epochs, cfg.lr_decay_start, cfg.epochs)
        self.sched_D = WarmupDecayScheduler(self.opt_D, cfg.lr_d, cfg.warmup_epochs, cfg.lr_decay_start, cfg.epochs)

        # Loss
        self.criterion = MakeupGANLoss(
            lambda_gan=cfg.lambda_gan, lambda_pixel=cfg.lambda_pixel,
            lambda_perc=cfg.lambda_perc, lambda_hist=0.0,       # hist ramped up [P-21]
            lambda_identity=cfg.lambda_identity,
        ).to(self.device)
        self._target_hist = cfg.lambda_hist

        # Helpers
        self.ckpt_mgr = CheckpointManager(self.run_dir / "checkpoints")
        self.sampler  = SampleSaver(self.run_dir, cfg.n_samples)
        self.writer   = SummaryWriter(log_dir=self.run_dir / "tb")
        self.val_csv  = ValLossCSV(self.run_dir / "val_loss.csv")

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

    # ── Lambda ramp-up helpers [P-19, P-21] ──────────────────────
    def _update_ramps(self, epoch: int) -> None:
        wp = max(1, self.cfg.warmup_epochs)
        # Identity: ramp 0 → target over warmup
        lam_id = self.cfg.lambda_identity * min(1.0, epoch / wp)
        # Histogram: start ramping AFTER warmup (model needs basic structure first)
        lam_hist = self._target_hist * max(0.0, (epoch - wp) / max(1, wp))
        self.criterion.update_lambdas(identity=lam_id, hist=lam_hist)

    # ── Training step ─────────────────────────────────────────────
    def _train_step(self, batch: dict, step: int) -> tuple[dict[str, float], dict[str, float]]:
        src  = batch["source"].to(self.device)
        ref  = batch["reference"].to(self.device)
        tgt  = batch["target"].to(self.device)
        mask = batch["mask"].to(self.device)

        d_log: dict[str, float] = {}

        # ── Discriminator ────────────────────────────────────────
        if not isinstance(self.D, nn.Identity) and step % self.cfg.n_critic == 0:
            self.opt_D.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self._amp_device, enabled=self._use_amp):
                with torch.no_grad():
                    fake = self.G(src, ref)
                real_fp, real_lp = self.D(tgt,          ref, mask)
                fake_fp, fake_lp = self.D(fake.detach(), ref, mask)
                d_loss, d_dict   = self.criterion.discriminator_loss(
                    real_fp, fake_fp, real_lp, fake_lp
                )
            self.scaler.scale(d_loss).backward()
            self.scaler.unscale_(self.opt_D)
            nn.utils.clip_grad_norm_(self.D.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt_D)
            d_log = {k: v.item() for k, v in d_dict.items()}

        # ── Generator ────────────────────────────────────────────
        self.opt_G.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self._amp_device, enabled=self._use_amp):
            fake = self.G(src, ref)
            if isinstance(self.D, nn.Identity):
                # Stub: fake predictions are zeros
                B = fake.shape[0]
                fake_fp = fake.new_zeros(B, 1, 1, 1)
                fake_lp = fake.new_zeros(B, 1, 1, 1)
            else:
                fake_fp, fake_lp = self.D(fake, ref, mask)
            g_loss, g_dict = self.criterion.generator_loss(
                fake, tgt, ref, src, fake_fp, fake_lp, mask
            )
        self.scaler.scale(g_loss).backward()
        self.scaler.unscale_(self.opt_G)
        nn.utils.clip_grad_norm_(self.G.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.opt_G)
        self.scaler.update()   # single update per step
        g_log = {k: v.item() for k, v in g_dict.items()}

        return g_log, d_log

    # ── Validation ────────────────────────────────────────────────
    @torch.no_grad()
    def _validate(self) -> float:
        self.G.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            src = batch["source"].to(self.device)
            ref = batch["reference"].to(self.device)
            tgt = batch["target"].to(self.device)
            gen = self.G(src, ref)
            l1   = torch.nn.functional.l1_loss(gen, tgt)
            perc = self.criterion.perc_loss(gen, tgt)
            total += (l1 + 0.1 * perc).item()
            n += 1
        self.G.train()
        return total / max(n, 1)

    # ── Full training loop ────────────────────────────────────────
    def train(self) -> None:
        log.info(f"Training {self.cfg.epochs} epochs from epoch {self.start_epoch}")
        _print_config(self.cfg)

        fixed_batch = next(iter(self.val_loader))
        self.G.train()
        if not isinstance(self.D, nn.Identity):
            self.D.train()

        current_epoch = self.start_epoch  # track for emergency save

        try:
            for epoch in range(self.start_epoch, self.cfg.epochs):
                current_epoch = epoch
                lr_g = self.sched_G.step(epoch)
                lr_d = self.sched_D.step(epoch)
                self._update_ramps(epoch)    # [P-19, P-21]

                epoch_g: dict[str, float] = {}
                epoch_d: dict[str, float] = {}
                steps         = 0
                epoch_t0      = time.perf_counter()

                for step, batch in enumerate(self.train_loader):
                    step_t0        = time.perf_counter()
                    g_m, d_m       = self._train_step(batch, step)
                    step_ms        = (time.perf_counter() - step_t0) * 1000

                    for k, v in g_m.items():
                        epoch_g[k] = epoch_g.get(k, 0.0) + v
                    for k, v in d_m.items():
                        epoch_d[k] = epoch_d.get(k, 0.0) + v
                    steps += 1

                    if step % 50 == 0:
                        elapsed  = time.perf_counter() - epoch_t0
                        remain   = len(self.train_loader) - step
                        eta_s    = (elapsed / max(step, 1)) * remain
                        log.info(
                            f"[{epoch+1}/{self.cfg.epochs}] "
                            f"step {step}/{len(self.train_loader)}  "
                            f"G={g_m.get('G_total',0):.4f}  "
                            f"D={d_m.get('D_total',0):.4f}  "
                            f"{step_ms:.0f}ms/step  "
                            f"ETA {eta_s:.0f}s"
                        )

                avg_g    = {k: v / steps for k, v in epoch_g.items()}
                avg_d    = {k: v / steps for k, v in epoch_d.items()}
                val_loss = self._validate()
                epoch_s  = time.perf_counter() - epoch_t0

                # TensorBoard
                for k, v in {**avg_g, **avg_d}.items():
                    self.writer.add_scalar(f"Train/{k}", v, epoch)
                self.writer.add_scalar("Val/loss", val_loss, epoch)
                self.writer.add_scalar("LR/G",     lr_g,     epoch)
                self.writer.add_scalar("LR/D",     lr_d,     epoch)
                self.writer.add_scalar("Lambda/hist",     self.criterion.lam_hist,  epoch)
                self.writer.add_scalar("Lambda/identity", self.criterion.lam_id,    epoch)

                # CSV [P-20]
                self.val_csv.write(epoch + 1, {**avg_g, **avg_d, "val_loss": val_loss})

                log.info(
                    f"Epoch {epoch+1:4d}/{self.cfg.epochs} | "
                    f"G={avg_g.get('G_total',0):.4f}  "
                    f"D={avg_d.get('D_total',0):.4f}  "
                    f"Val={val_loss:.4f}  "
                    f"LR={lr_g:.1e}  "
                    f"Time={epoch_s:.0f}s"
                )

                if (epoch + 1) % self.cfg.sample_every == 0:
                    self.sampler.save(epoch + 1, self.G, fixed_batch, self.device, self.writer)

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
                        log.info(f"New best val loss: {val_loss:.4f}")

                if val_loss >= self.best_val_loss:
                    self.patience_ctr += 1
                if self.patience_ctr >= self.cfg.patience:
                    log.info(f"Early stopping at epoch {epoch+1}")
                    break

        except Exception:
            log.exception("Training crashed — saving emergency checkpoint")
            self.ckpt_mgr.save_emergency(self.G, self.D, current_epoch)  # [P-18]
            raise
        finally:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception as e:
                log.warning(f"TensorBoard close error: {e}")

        log.info("Training complete.")


# ─────────────────────────────────────────────────────────
#  CONFIG PRINTOUT
# ─────────────────────────────────────────────────────────

def _print_config(cfg: TrainingConfig) -> None:
    sep = "═" * 56
    print(f"\n{sep}")
    print(f"  Makeup AI — Training Run")
    print(sep)
    print(f"  Data:       {cfg.data_root}  ({cfg.dataset_fmt})")
    print(f"  Image size: {cfg.image_size}px  Batch: {cfg.batch_size}")
    print(f"  Epochs:     {cfg.epochs}  LR G/D: {cfg.lr_g:.0e}/{cfg.lr_d:.0e}")
    print(f"  λ gan={cfg.lambda_gan}  pixel={cfg.lambda_pixel}  "
          f"perc={cfg.lambda_perc}  hist={cfg.lambda_hist}  id={cfg.lambda_identity}")
    print(f"  AMP:        {cfg.use_amp}  Attn: {cfg.use_attention}  "
          f"SpecNorm: {cfg.use_spec_norm}  GradCkpt: {cfg.grad_ckpt}")
    print(f"  Seed:       {cfg.seed}")
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

    p = argparse.ArgumentParser(description="Makeup AI — Train GAN v3")
    p.add_argument("--data",           type=Path,  required=True)
    p.add_argument("--format",         type=str,   default="beautygan",
                   choices=["beautygan", "csv", "custom"])
    p.add_argument("--csv",            type=Path,  default=None)
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--image-size",     type=int,   default=256)
    p.add_argument("--lr-g",           type=float, default=2e-4)
    p.add_argument("--lr-d",           type=float, default=2e-4)
    p.add_argument("--lambda-pixel",   type=float, default=10.0)
    p.add_argument("--lambda-perc",    type=float, default=0.1)
    p.add_argument("--lambda-hist",    type=float, default=1.0)
    p.add_argument("--lambda-identity",type=float, default=0.5)
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--patience",       type=int,   default=15)
    p.add_argument("--resume",         type=Path,  default=None)
    p.add_argument("--output",         type=Path,  default=Path("runs"))
    p.add_argument("--mask-cache",     type=Path,  default=Path("data/mask_cache"))
    p.add_argument("--device",         type=str,   default="auto")
    p.add_argument("--no-amp",         action="store_true")
    p.add_argument("--no-attention",   action="store_true")
    p.add_argument("--no-spec-norm",   action="store_true")
    p.add_argument("--grad-ckpt",      action="store_true")
    p.add_argument("--seed",           type=int,   default=42)
    args = p.parse_args()

    trainer = Trainer(TrainingConfig(
        data_root       = args.data,
        dataset_fmt     = args.format,
        csv_path        = args.csv,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        image_size      = args.image_size,
        lr_g            = args.lr_g,
        lr_d            = args.lr_d,
        lambda_pixel    = args.lambda_pixel,
        lambda_perc     = args.lambda_perc,
        lambda_hist     = args.lambda_hist,
        lambda_identity = args.lambda_identity,
        num_workers     = args.workers,
        patience        = args.patience,
        resume          = args.resume,
        output_dir      = args.output,
        mask_cache      = args.mask_cache,
        device          = args.device,
        use_amp         = not args.no_amp,
        use_attention   = not args.no_attention,
        use_spec_norm   = not args.no_spec_norm,
        grad_ckpt       = args.grad_ckpt,
        seed            = args.seed,
    ))
    trainer.train()


if __name__ == "__main__":
    main()
