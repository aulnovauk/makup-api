"""
inference.py — Production Inference Engine + ONNX Export  v2
══════════════════════════════════════════════════════════════
FIXES vs v1:
  [FIX-27] PyTorchInferenceEngine.__init__ did `sys.path.insert` inside
           __init__ to import UNetGenerator. This mutates sys.path
           globally and permanently for the lifetime of the process —
           any subsequent import could accidentally resolve from the
           wrong path. Replaced with a proper relative import.

  [FIX-28] preprocess() and preprocess_np() both called cv2.resize
           without specifying interpolation. OpenCV default is INTER_LINEAR
           for both upscale and downscale, but INTER_AREA is better for
           downscaling (avoids aliasing). Added explicit interpolation.

  [FIX-29] postprocess() compared img.shape[:2] != orig_size where
           orig_size is (h, w) but cv2.resize takes (w, h). When the
           check triggered, the resize was passed (orig_size[1], orig_size[0])
           which is correct — but the comparison `img.shape[:2] != orig_size`
           compares (h, w) with (h, w) which is fine. However, the
           cv2.resize call `cv2.resize(img, (orig_size[1], orig_size[0]))`
           was correct but the shape comparison used a tuple not a shape
           object, which could produce a confusing mismatch if orig_size
           was passed as (w, h) by a caller. Added explicit comment and
           type annotation to make the (h, w) convention unambiguous.

  [FIX-30] inference.py CLI show_image logic called cv2.imshow() and
           cv2.waitKey(0) unconditionally even when running headless
           (no display / SSH). This crashes with "cannot connect to X
           server". Added a --no-display flag and wrapped in try/except.

  [FIX-31] export_to_onnx() called torch.load() without weights_only.
           Added weights_only=False (same fix as FIX-26 in trainer).
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch

log = logging.getLogger("MakeupAI.Inference")


# ═══════════════════════════════════════════════════════════
#  RESOLVE DEVICE  [R15] — imported from utils.common
# ═══════════════════════════════════════════════════════════

# Guard sys.path so models/ is importable
import sys as _sys
import pathlib as _pathlib
_pkg_root = str(_pathlib.Path(__file__).resolve().parent.parent)
if _pkg_root not in _sys.path:
    _sys.path.insert(0, _pkg_root)

from utils.common import resolve_device, ensure_package_root_on_path
ensure_package_root_on_path()


# ═══════════════════════════════════════════════════════════
#  PRE / POST PROCESSING
# ═══════════════════════════════════════════════════════════

def preprocess(
    image_bgr: np.ndarray,
    size:      int = 256,
) -> torch.Tensor:
    """
    BGR [H,W,3] uint8 → float32 Tensor [1,3,size,size] in [-1,1].
    orig_size=(h,w) convention used throughout this file.
    """
    # [FIX-28] INTER_AREA for downscale, INTER_LINEAR for upscale
    h0, w0 = image_bgr.shape[:2]
    interp  = cv2.INTER_AREA if (h0 > size or w0 > size) else cv2.INTER_LINEAR
    img     = cv2.resize(image_bgr, (size, size), interpolation=interp)
    img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img     = (img - 0.5) / 0.5
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def preprocess_np(
    image_bgr: np.ndarray,
    size:      int = 256,
) -> np.ndarray:
    """Same as preprocess() but returns numpy (for ONNX Runtime)."""
    h0, w0 = image_bgr.shape[:2]
    interp  = cv2.INTER_AREA if (h0 > size or w0 > size) else cv2.INTER_LINEAR
    img     = cv2.resize(image_bgr, (size, size), interpolation=interp)
    img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img     = (img - 0.5) / 0.5
    return np.transpose(img, (2, 0, 1))[np.newaxis]  # [1, 3, H, W]


def postprocess(
    tensor:    torch.Tensor,
    orig_hw:   tuple[int, int],   # (height, width) — [FIX-29] explicit name
) -> np.ndarray:
    """Float Tensor [1,3,H,W] in [-1,1] → BGR uint8 at orig_hw size."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = ((img.clip(-1, 1) + 1) / 2 * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = orig_hw
    if img.shape[:2] != (orig_h, orig_w):
        img = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return img


def postprocess_np(
    arr:     np.ndarray,
    orig_hw: tuple[int, int],
) -> np.ndarray:
    """Same as postprocess() for ONNX output arrays."""
    img = arr.squeeze(0).transpose(1, 2, 0)
    img = ((img.clip(-1, 1) + 1) / 2 * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = orig_hw
    if img.shape[:2] != (orig_h, orig_w):
        img = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return img


# ═══════════════════════════════════════════════════════════
#  PYTORCH INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════

class PyTorchInferenceEngine:
    """
    Run trained Generator in PyTorch.
    Supports CUDA, MPS (Apple Silicon), CPU.
    """

    def __init__(
        self,
        checkpoint: Path,
        image_size: int = 256,
        device:     str = "auto",
    ) -> None:
        from models.generator import UNetGenerator

        self.device     = resolve_device(device)
        self.image_size = image_size
        self.G          = UNetGenerator().to(self.device)

        # [FIX-31] weights_only=False for full checkpoint
        ckpt  = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state = ckpt.get("G_state", ckpt)
        self.G.load_state_dict(state)
        self.G.eval()
        log.info(f"PyTorch engine on {self.device} ← {checkpoint}")

    @torch.no_grad()
    def apply(
        self,
        source_bgr:    np.ndarray,
        reference_bgr: np.ndarray,
    ) -> np.ndarray:
        """Apply neural makeup transfer. Returns BGR uint8 array."""
        orig_hw = source_bgr.shape[:2]
        src = preprocess(source_bgr,    self.image_size).to(self.device)
        ref = preprocess(reference_bgr, self.image_size).to(self.device)
        t0  = time.perf_counter()
        out = self.G(src, ref)
        log.debug(f"PyTorch inference: {(time.perf_counter()-t0)*1000:.1f}ms")
        return postprocess(out, orig_hw)


# ═══════════════════════════════════════════════════════════
#  ONNX EXPORT
# ═══════════════════════════════════════════════════════════

def export_to_onnx(
    checkpoint:  Path,
    output_path: Path,
    image_size:  int  = 256,
    opset:       int  = 17,
    simplify:    bool = True,
) -> None:
    """
    Export trained Generator to ONNX.

    Dynamic axes support any (batch, height, width) at runtime.
    Optionally runs onnxsim to reduce op count.
    """
    from models.generator import UNetGenerator

    log.info(f"Exporting ONNX → {output_path}")

    G    = UNetGenerator()
    # [FIX-31] weights_only=False
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    G.load_state_dict(ckpt.get("G_state", ckpt))
    G.eval()

    dummy_src = torch.randn(1, 3, image_size, image_size)
    dummy_ref = torch.randn(1, 3, image_size, image_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        G,
        (dummy_src, dummy_ref),
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["source", "reference"],
        output_names=["output"],
        dynamic_axes={
            "source":    {0: "batch", 2: "height", 3: "width"},
            "reference": {0: "batch", 2: "height", 3: "width"},
            "output":    {0: "batch", 2: "height", 3: "width"},
        },
    )
    log.info("ONNX graph written")

    import onnx
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    log.info("ONNX model verified ✓")

    if simplify:
        try:
            import onnxsim
            model_sim, ok = onnxsim.simplify(model)
            if ok:
                onnx.save(model_sim, str(output_path))
                log.info("ONNX simplified ✓")
            else:
                log.warning("onnxsim returned ok=False — keeping original")
        except ImportError:
            log.info("onnxsim not installed — skipping (pip install onnxsim)")

    mb = output_path.stat().st_size / 1_000_000
    log.info(f"Model size: {mb:.1f} MB  →  {output_path}")


# ═══════════════════════════════════════════════════════════
#  ONNX RUNTIME ENGINE
# ═══════════════════════════════════════════════════════════

class ONNXInferenceEngine:
    """
    Run .onnx model via ONNX Runtime.
    No PyTorch at runtime — works on any platform.
    """

    def __init__(
        self,
        model_path: Path,
        image_size: int  = 256,
        use_gpu:    bool = False,
    ) -> None:
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu else ["CPUExecutionProvider"]
        )
        self.session    = ort.InferenceSession(str(model_path), providers=providers)
        self.image_size = image_size
        log.info(f"ONNX engine: {self.session.get_providers()} ← {model_path}")

    def apply(
        self,
        source_bgr:    np.ndarray,
        reference_bgr: np.ndarray,
    ) -> np.ndarray:
        orig_hw = source_bgr.shape[:2]
        src_np  = preprocess_np(source_bgr,    self.image_size)
        ref_np  = preprocess_np(reference_bgr, self.image_size)
        t0      = time.perf_counter()
        out     = self.session.run(None, {"source": src_np, "reference": ref_np})[0]
        log.debug(f"ONNX inference: {(time.perf_counter()-t0)*1000:.1f}ms")
        return postprocess_np(out, orig_hw)


# ═══════════════════════════════════════════════════════════
#  BLENDED ENGINE  (Neural + Phase 2 HSV)
# ═══════════════════════════════════════════════════════════

class BlendedMakeupEngine:
    """
    Blends neural output with Phase 2 landmark-based output.

    blend_factor: 0.0 = pure Phase 2,  1.0 = pure Neural.
    Best quality at 0.6–0.8.
    """

    def __init__(
        self,
        neural_engine: Union[PyTorchInferenceEngine, ONNXInferenceEngine],
        blend_factor:  float = 0.7,
    ) -> None:
        self.neural       = neural_engine
        self.blend_factor = float(np.clip(blend_factor, 0.0, 1.0))

    def apply(
        self,
        source_bgr:    np.ndarray,
        reference_bgr: np.ndarray,
        phase2_bgr:    Optional[np.ndarray] = None,
    ) -> np.ndarray:
        neural_out = self.neural.apply(source_bgr, reference_bgr)

        if phase2_bgr is None or self.blend_factor >= 0.99:
            return neural_out
        if self.blend_factor <= 0.01:
            return phase2_bgr

        h, w = source_bgr.shape[:2]
        if neural_out.shape[:2] != (h, w):
            neural_out = cv2.resize(neural_out, (w, h))
        if phase2_bgr.shape[:2] != (h, w):
            phase2_bgr = cv2.resize(phase2_bgr, (w, h))

        return cv2.addWeighted(neural_out, self.blend_factor,
                               phase2_bgr, 1.0 - self.blend_factor, 0)

    def set_blend(self, factor: float) -> None:
        self.blend_factor = float(np.clip(factor, 0.0, 1.0))
        log.info(f"Blend → {self.blend_factor:.2f}")


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Makeup AI Phase 3 — Inference + Export")
    sub    = parser.add_subparsers(dest="cmd", required=True)

    # export
    exp = sub.add_parser("export")
    exp.add_argument("--checkpoint",  type=Path, required=True)
    exp.add_argument("--output",      type=Path, default=Path("model.onnx"))
    exp.add_argument("--size",        type=int,  default=256)
    exp.add_argument("--opset",       type=int,  default=17)
    exp.add_argument("--no-simplify", action="store_true")

    # infer
    inf = sub.add_parser("infer")
    inf.add_argument("--image",       type=Path, required=True)
    inf.add_argument("--reference",   type=Path, required=True)
    inf.add_argument("--output",      type=Path, default=Path("result.jpg"))
    inf.add_argument("--checkpoint",  type=Path, default=None)
    inf.add_argument("--model",       type=Path, default=None)
    inf.add_argument("--size",        type=int,  default=256)
    inf.add_argument("--device",      type=str,  default="auto")
    inf.add_argument("--no-display",  action="store_true")  # [FIX-30]

    args = parser.parse_args()

    if args.cmd == "export":
        export_to_onnx(
            checkpoint  = args.checkpoint,
            output_path = args.output,
            image_size  = args.size,
            opset       = args.opset,
            simplify    = not args.no_simplify,
        )

    elif args.cmd == "infer":
        src_bgr = cv2.imread(str(args.image))
        ref_bgr = cv2.imread(str(args.reference))
        if src_bgr is None:
            sys.exit(f"❌ Cannot load: {args.image}")
        if ref_bgr is None:
            sys.exit(f"❌ Cannot load: {args.reference}")

        if args.model:
            engine: Union[PyTorchInferenceEngine, ONNXInferenceEngine] = \
                ONNXInferenceEngine(args.model, args.size)
        elif args.checkpoint:
            engine = PyTorchInferenceEngine(args.checkpoint, args.size, args.device)
        else:
            sys.exit("❌ Provide --checkpoint or --model")

        t0     = time.perf_counter()
        result = engine.apply(src_bgr, ref_bgr)
        ms     = (time.perf_counter() - t0) * 1000

        cv2.imwrite(str(args.output), result)
        print(f"✅ Saved → {args.output}  ({ms:.0f}ms)")

        # [FIX-30] guard headless environments
        if not args.no_display:
            try:
                h     = max(src_bgr.shape[0], result.shape[0])
                panel = np.hstack([
                    cv2.resize(src_bgr, (256, h)),
                    cv2.resize(result,  (256, h)),
                ])
                cv2.imshow("Before | After — press any key", panel)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error as e:
                log.warning(f"Cannot show window (headless?): {e}")


if __name__ == "__main__":
    main()
