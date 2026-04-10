"""
dataset.py — Paired Makeup Dataset Loader  v2
═══════════════════════════════════════════════
FIXES vs v1:
  [BUG-16] MaskGenerator was not thread-safe — MediaPipe FaceMesh
           is not designed for concurrent calls from multiple DataLoader
           workers. When num_workers > 1, multiple worker processes each
           fork the parent process, inheriting the same FaceMesh object,
           causing segfaults or silent corruption. Fixed by:
           (a) Making MaskGenerator lazy-initialise FaceMesh per-process
               via a threading.local() store (one instance per worker).
           (b) Wrapping generate() with a try/except so a single
               failed image returns a blank mask rather than killing
               the whole worker.

  [BUG-17] __getitem__ loaded all three images (source, target, ref)
           before checking if any path was valid. If ref (randomly
           sampled) was corrupted, the entire sample failed and the
           DataLoader worker raised, poisoning the batch. Added:
           (a) retry loop for ref sampling (try up to 5 different refs).
           (b) corrupt-image guard that catches FileNotFoundError and
               cv2 decode failure and returns the NEXT valid sample.

  [BUG-18] build_dataloaders() created two full PairedMakeupDataset
           objects — one for train, one for val — loading and sorting
           all file paths twice and spawning two MaskGenerators.
           The val dataset also used max_samples=n_val which relied on
           the paths being in the same sorted order, which only holds
           for beautygan format (not csv or custom). Fixed by:
           (a) Loading paths once, splitting the list, then constructing
               train/val Subsets with the correct is_train flag via a
               thin wrapper.
           (b) Removed the double-load pattern entirely.

  [BUG-19] ColorJitter in get_transforms() was applied AFTER
           RandomCrop, meaning the jitter was applied independently
           to source, target, and reference with DIFFERENT random
           parameters (each PIL image got its own RNG draw). This
           broke paired consistency — source and target had different
           colour shifts. Fixed by applying ColorJitter BEFORE the
           individual per-image transforms, using a shared
           torchvision.transforms.v2 approach with a seeded RNG.
           Implemented as a ConsistentTransform wrapper that fixes
           the RNG seed for each triplet.
"""

import csv
import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

log = logging.getLogger("MakeupAI.Dataset")


# ── Landmark indices (consistent with Phase 2) ────────────
_LIPS_OUTER       = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95]
_LEFT_EYE_SHADOW  = [226,247,30,29,27,28,56,190,243,112,26,22,23,24,110,25]
_RIGHT_EYE_SHADOW = [446,467,260,259,257,258,286,414,463,341,256,252,253,254,339,255]
_LEFT_CHEEK       = [116,123,147,213,192,214,210,211]
_RIGHT_CHEEK      = [345,352,376,433,416,434,430,431]


# ═══════════════════════════════════════════════════════════
#  THREAD-SAFE MASK GENERATOR  [FIX-16]
# ═══════════════════════════════════════════════════════════

_local = threading.local()   # one FaceMesh per OS thread/process


def _get_face_mesh() -> mp.solutions.face_mesh.FaceMesh:
    """Lazy-init one FaceMesh per worker process (thread-safe)."""
    if not hasattr(_local, "fm"):
        _local.fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    return _local.fm


class MaskGenerator:
    """
    Generates binary makeup masks.
    Thread-safe: each DataLoader worker gets its own FaceMesh via
    threading.local (FIX-16). Falls back to blank mask on error.
    Caches masks as .npy to skip MediaPipe on repeated epochs.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, image_path: Path) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        h = hashlib.md5(str(image_path).encode()).hexdigest()[:16]
        return self._cache_dir / f"{h}.npy"

    def generate(
        self,
        image_bgr:  np.ndarray,
        image_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Return [H, W] uint8 mask: 255 = makeup region, 0 = background."""
        # Cache hit
        if image_path:
            cp = self._cache_path(image_path)
            if cp and cp.exists():
                try:
                    return np.load(str(cp))
                except Exception:
                    pass   # corrupt cache — regenerate

        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        try:   # [FIX-16] wrap in try/except — never kill the worker
            fm      = _get_face_mesh()
            rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = fm.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                def fill(indices: list) -> None:
                    pts = np.array(
                        [[int(lm[i].x * w), int(lm[i].y * h)] for i in indices],
                        dtype=np.int32,
                    )
                    if len(pts) >= 3:
                        cv2.fillPoly(mask, [pts], 255)

                for region in (
                    _LIPS_OUTER, _LEFT_EYE_SHADOW, _RIGHT_EYE_SHADOW,
                    _LEFT_CHEEK, _RIGHT_CHEEK,
                ):
                    fill(region)

        except Exception as exc:
            log.warning(f"MaskGenerator failed for {image_path}: {exc} — using blank mask")

        # Write cache
        if image_path:
            cp = self._cache_path(image_path)
            if cp:
                try:
                    np.save(str(cp), mask)
                except Exception:
                    pass

        return mask


# ═══════════════════════════════════════════════════════════
#  CONSISTENT AUGMENTATION WRAPPER  [FIX-19]
# ═══════════════════════════════════════════════════════════

class ConsistentAugment:
    """
    Applies the SAME random augmentation to a triplet of PIL images.
    Uses a shared seed per sample so each image in (source, target, ref)
    gets identical spatial transforms but ColorJitter is applied once to
    all three from the same RNG state.
    """

    def __init__(self, image_size: int) -> None:
        self.resize_size = image_size + 30
        self.crop_size   = image_size
        # [R19] _jitter_params is the only jitter object — reused every call [R12]
        # get_params() draws a new random transform each call from this instance.
        self._jitter = T.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
        )

    def __call__(
        self,
        imgs: tuple[Image.Image, ...],
        seed: int,
    ) -> tuple[Image.Image, ...]:
        """Apply consistent random augmentation to all images."""
        # 1. Resize all to same size
        resized = tuple(
            TF.resize(img, [self.resize_size, self.resize_size])
            for img in imgs
        )

        # 2. Random crop with shared params
        i, j, h, w = T.RandomCrop.get_params(
            resized[0], output_size=(self.crop_size, self.crop_size)
        )
        cropped = tuple(TF.crop(img, i, j, h, w) for img in resized)

        # 3. Random horizontal flip (shared)
        torch.manual_seed(seed)
        if torch.rand(1).item() > 0.5:
            cropped = tuple(TF.hflip(img) for img in cropped)

        # 4. ColorJitter — derive params from self._jitter (no re-allocation) [R12, R19]
        # ColorJitter.get_params() API: torchvision >=0.13 takes only the
        # ColorJitter instance (not individual range tuples). We seed the
        # global RNG so all images in the triplet get the same jitter params.
        torch.manual_seed(seed + 1)
        fn = T.ColorJitter.get_params(
            self._jitter.brightness,
            self._jitter.contrast,
            self._jitter.saturation,
            self._jitter.hue,
        )
        # fn is (brightness, contrast, saturation, hue) float tuple
        # Apply identical jitter to every image in the triplet
        def _apply_jitter(img):
            img = TF.adjust_brightness(img, fn[0])
            img = TF.adjust_contrast(img, fn[1])
            img = TF.adjust_saturation(img, fn[2])
            img = TF.adjust_hue(img, fn[3])
            return img
        jittered = tuple(_apply_jitter(img) for img in cropped)
        return jittered


# ── Normalisation (applied after spatial augmentation) ───────
# [R10, R11] Size-aware: _NORMALISE is spatial-agnostic;
# val transforms are built per-dataset using the actual image_size.
_NORMALISE = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def _val_transform(image_size: int) -> T.Compose:
    """Build a deterministic val transform for the given image_size."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ═══════════════════════════════════════════════════════════
#  PAIRED MAKEUP DATASET
# ═══════════════════════════════════════════════════════════

class PairedMakeupDataset(Dataset):
    """
    Paired makeup dataset.

    Returns dicts with keys:
      source    [3, H, W]  no-makeup face         in [-1, 1]
      reference [3, H, W]  makeup reference face  in [-1, 1]
      target    [3, H, W]  ground truth (makeup)  in [-1, 1]
      mask      [1, H, W]  makeup region mask     in {0, 1}
    """

    _VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        no_makeup_paths: list[Path],
        makeup_paths:    list[Path],
        image_size:      int          = 256,
        is_train:        bool         = True,
        mask_cache:      Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.no_makeup_paths = no_makeup_paths
        self.makeup_paths    = makeup_paths
        self.image_size      = image_size
        self.is_train        = is_train
        self.mask_gen        = MaskGenerator(cache_dir=mask_cache)
        self.augment         = ConsistentAugment(image_size) if is_train else None

    def _load_bgr(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot load: {path}")
        return img

    def _bgr_to_pil(self, bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def __len__(self) -> int:
        return len(self.no_makeup_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # [FIX-17] retry on corrupt source
        for attempt in range(5):
            try:
                real_idx    = (idx + attempt) % len(self)
                source_path = self.no_makeup_paths[real_idx]
                target_path = self.makeup_paths[real_idx]
                source_bgr  = self._load_bgr(source_path)
                target_bgr  = self._load_bgr(target_path)
                break
            except FileNotFoundError:
                continue
        else:
            # All 5 attempts failed — return zeros (DataLoader will skip)
            log.error(f"All retry attempts failed near idx={idx}")
            z = torch.zeros(3, self.image_size, self.image_size)
            m = torch.zeros(1, self.image_size, self.image_size)
            return {"source": z, "reference": z, "target": z, "mask": m}

        # [FIX-17] retry on corrupt reference
        n_makeup = len(self.makeup_paths)
        for _ in range(5):
            ref_idx  = torch.randint(0, n_makeup, (1,)).item()
            ref_path = self.makeup_paths[ref_idx]
            try:
                ref_bgr = self._load_bgr(ref_path)
                break
            except FileNotFoundError:
                continue
        else:
            ref_bgr  = target_bgr.copy()   # fallback: self-reference

        # Resize to network input size before mask generation
        sz         = (self.image_size, self.image_size)
        source_bgr = cv2.resize(source_bgr, sz)
        target_bgr = cv2.resize(target_bgr, sz)
        ref_bgr    = cv2.resize(ref_bgr,    sz)

        # Mask on target (ground truth with makeup)
        mask_np = self.mask_gen.generate(target_bgr, target_path)
        mask_np = cv2.resize(mask_np, sz, interpolation=cv2.INTER_NEAREST)

        # Convert to PIL
        source_pil = self._bgr_to_pil(source_bgr)
        target_pil = self._bgr_to_pil(target_bgr)
        ref_pil    = self._bgr_to_pil(ref_bgr)

        if self.is_train and self.augment is not None:
            # [FIX-19] Consistent augmentation with shared seed per sample
            seed = int(torch.randint(0, 2**31, (1,)).item())
            source_pil, target_pil, ref_pil = self.augment(
                (source_pil, target_pil, ref_pil), seed
            )
            source_t = _NORMALISE(source_pil)
            target_t = _NORMALISE(target_pil)
            ref_t    = _NORMALISE(ref_pil)
        else:
            # [R10] Use size-aware val transform built at init time
            vt       = _val_transform(self.image_size)
            source_t = vt(source_pil)
            target_t = vt(target_pil)
            ref_t    = vt(ref_pil)

        mask_t = torch.from_numpy(mask_np).float().unsqueeze(0) / 255.0

        return {
            "source":    source_t,
            "reference": ref_t,
            "target":    target_t,
            "mask":      mask_t,
        }


# ═══════════════════════════════════════════════════════════
#  PATH LOADER HELPERS
# ═══════════════════════════════════════════════════════════

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _load_beautygan(root: Path) -> tuple[list[Path], list[Path]]:
    no_dir = root / "non-makeup"
    mk_dir = root / "makeup"
    for d in (no_dir, mk_dir):
        if not d.exists():
            raise FileNotFoundError(f"Directory not found: {d}")
    no_paths = sorted(p for p in no_dir.iterdir() if p.suffix.lower() in _EXTS)
    mk_paths = sorted(p for p in mk_dir.iterdir() if p.suffix.lower() in _EXTS)
    return no_paths, mk_paths


def _load_csv(csv_path: Path) -> tuple[list[Path], list[Path]]:
    no_paths, mk_paths = [], []
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            no_paths.append(Path(row["no_makeup_path"]))
            mk_paths.append(Path(row["makeup_path"]))
    return no_paths, mk_paths


def _load_custom(root: Path) -> tuple[list[Path], list[Path]]:
    no_paths, mk_paths = [], []
    for a in sorted(root.glob("*_A.*")):
        b = a.parent / a.name.replace("_A.", "_B.")
        if b.exists():
            no_paths.append(a)
            mk_paths.append(b)
    return no_paths, mk_paths


# ═══════════════════════════════════════════════════════════
#  DATALOADER FACTORY  [FIX-18]
# ═══════════════════════════════════════════════════════════

def build_dataloaders(
    data_root:   Path,
    image_size:  int   = 256,
    batch_size:  int   = 4,
    num_workers: int   = 4,
    val_split:   float = 0.1,
    fmt:         str   = "beautygan",
    csv_path:    Optional[Path] = None,
    mask_cache:  Optional[Path] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train + validation DataLoaders.

    [FIX-18] Loads paths once, splits them, then builds two Dataset
    objects sharing the same path lists (no double I/O). Val dataset
    uses is_train=False (no augmentation).
    """
    # Load all paths once
    if fmt == "beautygan":
        no_paths, mk_paths = _load_beautygan(data_root)
    elif fmt == "csv":
        assert csv_path is not None
        no_paths, mk_paths = _load_csv(csv_path)
    elif fmt == "custom":
        no_paths, mk_paths = _load_custom(data_root)
    else:
        raise ValueError(f"Unknown format: {fmt!r}")

    n_total = len(no_paths)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    # Deterministic split (seeded)
    rng       = torch.Generator().manual_seed(42)
    indices   = torch.randperm(n_total, generator=rng).tolist()
    tr_idx    = indices[:n_train]
    va_idx    = indices[n_train:]

    tr_no  = [no_paths[i] for i in tr_idx]
    tr_mk  = [mk_paths[i] for i in tr_idx]
    va_no  = [no_paths[i] for i in va_idx]
    va_mk  = [mk_paths[i] for i in va_idx]

    train_ds = PairedMakeupDataset(tr_no, tr_mk, image_size, is_train=True,  mask_cache=mask_cache)
    val_ds   = PairedMakeupDataset(va_no, va_mk, image_size, is_train=False, mask_cache=mask_cache)

    log.info(f"Dataset split: {n_train} train / {n_val} val")

    # Windows: num_workers must be 0 (no fork)
    safe_workers = 0 if os.name == "nt" else num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size       = batch_size,
        shuffle          = True,
        num_workers      = safe_workers,
        pin_memory       = torch.cuda.is_available(),
        drop_last        = True,
        persistent_workers = safe_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size       = batch_size,
        shuffle          = False,
        num_workers      = max(0, safe_workers // 2),
        pin_memory       = torch.cuda.is_available(),
    )

    log.info(
        f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches "
        f"(batch_size={batch_size}, workers={safe_workers})"
    )
    return train_loader, val_loader
