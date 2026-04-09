"""
utils/onnx_export.py — Standalone ONNX Export + Benchmark Utility
═══════════════════════════════════════════════════════════════════
Separate from inference.py — this file owns the full export pipeline:

  1. Load a trained BeautyGAN checkpoint
  2. Trace the generator with dummy inputs
  3. Export to ONNX with dynamic axes
  4. Verify the exported graph
  5. Optionally simplify with onnxsim
  6. Optionally quantise to INT8 (smaller, faster on CPU)
  7. Benchmark: compare PyTorch vs ONNX latency
  8. Print a full model report

Why separate from inference.py?
  inference.py is the runtime path — it imports onnxruntime and must
  be lean. This file imports onnx, onnxsim, and torch.onnx — heavy
  deps only needed at export time, not at inference time.

Usage:
    # Full export with all steps
    python utils/onnx_export.py --checkpoint runs/best_model.pt

    # Export + INT8 quantisation
    python utils/onnx_export.py --checkpoint runs/best_model.pt --quantize

    # Benchmark PyTorch vs ONNX
    python utils/onnx_export.py --checkpoint runs/best_model.pt --benchmark

    # Export at 512px resolution
    python utils/onnx_export.py --checkpoint runs/best_model.pt --size 512
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# [R16] Use shared guarded path-insertion helper — idempotent
from utils.common import ensure_package_root_on_path
ensure_package_root_on_path()

from models.generator import UNetGenerator

log = logging.getLogger("MakeupAI.ONNXExport")


# ═══════════════════════════════════════════════════════════
#  LOADER HELPER
# ═══════════════════════════════════════════════════════════

def _load_generator(checkpoint: Path, device: torch.device) -> UNetGenerator:
    """Load generator weights from a full checkpoint or weights-only file."""
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("G_state", ckpt)   # support both checkpoint formats
    G     = UNetGenerator()
    G.load_state_dict(state)
    G.eval()
    G.to(device)
    log.info(f"Generator loaded from {checkpoint}")
    params = G.n_parameters
    log.info(f"  Trainable params: {params['trainable']:,}")
    return G


# ═══════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════

def export(
    checkpoint:  Path,
    output_path: Path,
    image_size:  int  = 256,
    opset:       int  = 17,
    simplify:    bool = True,
    verbose:     bool = False,
) -> Path:
    """
    Export the generator to ONNX format.

    Args:
        checkpoint:  Path to .pt file
        output_path: Where to write the .onnx file
        image_size:  Spatial resolution for dummy inputs
        opset:       ONNX opset version (17 recommended for PyTorch 2.x)
        simplify:    Run onnx-simplifier to fold constants and prune ops
        verbose:     Print detailed ONNX graph info

    Returns:
        Path to the exported .onnx file
    """
    import onnx

    G = _load_generator(checkpoint, torch.device("cpu"))

    dummy_src = torch.randn(1, 3, image_size, image_size)
    dummy_ref = torch.randn(1, 3, image_size, image_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Exporting ONNX (opset {opset}, size {image_size}px) → {output_path}")

    with torch.no_grad():
        torch.onnx.export(
            G,
            args=(dummy_src, dummy_ref),
            f=str(output_path),
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
            verbose=verbose,
        )

    # Verify graph integrity
    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    log.info("ONNX graph verified ✓")

    initial_mb = output_path.stat().st_size / 1e6
    log.info(f"  Size before simplification: {initial_mb:.1f} MB")

    # Simplify — folds constants, removes redundant ops
    if simplify:
        try:
            import onnxsim
            model_sim, ok = onnxsim.simplify(model_proto)
            if ok:
                onnx.save(model_sim, str(output_path))
                final_mb = output_path.stat().st_size / 1e6
                saved    = initial_mb - final_mb
                log.info(f"  Simplified: {final_mb:.1f} MB  (saved {saved:.1f} MB) ✓")
            else:
                log.warning("  onnxsim returned ok=False — keeping original graph")
        except ImportError:
            log.info("  onnxsim not installed — skipping (pip install onnxsim)")

    final_mb = output_path.stat().st_size / 1e6
    log.info(f"Export complete: {output_path}  ({final_mb:.1f} MB)")
    return output_path


# ═══════════════════════════════════════════════════════════
#  INT8 QUANTISATION (CPU only)
# ═══════════════════════════════════════════════════════════

def quantize_int8(
    onnx_path:  Path,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Post-training static INT8 quantisation via ONNX Runtime.

    Produces a smaller, faster model for CPU inference.
    Typically 2–3× smaller and 1.5–2× faster than FP32 on CPU.
    Quality is nearly indistinguishable for makeup transfer.

    Args:
        onnx_path:   Path to the FP32 .onnx model
        output_path: Where to save the INT8 model (default: _int8.onnx)

    Returns:
        Path to the INT8 .onnx model
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if output_path is None:
        output_path = onnx_path.parent / (onnx_path.stem + "_int8.onnx")

    log.info(f"Quantising INT8: {onnx_path} → {output_path}")

    quantize_dynamic(
        model_input        = str(onnx_path),
        model_output       = str(output_path),
        weight_type        = QuantType.QUInt8,
        optimize_model     = True,
    )

    fp32_mb = onnx_path.stat().st_size / 1e6
    int8_mb = output_path.stat().st_size / 1e6
    log.info(
        f"INT8 quantisation complete: {int8_mb:.1f} MB  "
        f"(was {fp32_mb:.1f} MB, {100*(1-int8_mb/fp32_mb):.0f}% smaller)"
    )
    return output_path


# ═══════════════════════════════════════════════════════════
#  BENCHMARK
# ═══════════════════════════════════════════════════════════

def benchmark(
    checkpoint:  Path,
    onnx_path:   Path,
    image_size:  int = 256,
    n_runs:      int = 50,
    warmup:      int = 5,
) -> dict[str, float]:
    """
    Compare latency between PyTorch and ONNX Runtime inference.

    Args:
        checkpoint:  PyTorch .pt checkpoint
        onnx_path:   ONNX .onnx model
        image_size:  Input resolution
        n_runs:      Number of timed runs
        warmup:      Warmup runs (excluded from timing)

    Returns:
        Dict with pt_ms, onnx_ms, speedup keys
    """
    import onnxruntime as ort

    log.info(f"Benchmarking  {n_runs} runs at {image_size}px ...")

    # Prepare inputs
    src_t = torch.randn(1, 3, image_size, image_size)
    ref_t = torch.randn(1, 3, image_size, image_size)
    src_n = src_t.numpy()
    ref_n = ref_t.numpy()

    # ── PyTorch ───────────────────────────────────────────
    G = _load_generator(checkpoint, torch.device("cpu"))
    G.eval()

    with torch.no_grad():
        for _ in range(warmup):
            G(src_t, ref_t)

    pt_times: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            G(src_t, ref_t)
            pt_times.append((time.perf_counter() - t0) * 1000)

    pt_ms = float(np.median(pt_times))

    # ── ONNX Runtime ──────────────────────────────────────
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    for _ in range(warmup):
        session.run(None, {"source": src_n, "reference": ref_n})

    ort_times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {"source": src_n, "reference": ref_n})
        ort_times.append((time.perf_counter() - t0) * 1000)

    ort_ms  = float(np.median(ort_times))
    speedup = pt_ms / max(ort_ms, 0.001)

    results = {"pt_ms": pt_ms, "onnx_ms": ort_ms, "speedup": speedup}

    sep = "─" * 42
    print(f"\n{sep}")
    print(f"  Benchmark  ({n_runs} runs, {image_size}px, CPU)")
    print(sep)
    print(f"  PyTorch:      {pt_ms:7.1f} ms  (median)")
    print(f"  ONNX Runtime: {ort_ms:7.1f} ms  (median)")
    print(f"  Speedup:      {speedup:7.2f}×")
    print(sep)

    log.info(f"Benchmark: PT={pt_ms:.1f}ms  ONNX={ort_ms:.1f}ms  {speedup:.2f}×")
    return results


# ═══════════════════════════════════════════════════════════
#  MODEL REPORT
# ═══════════════════════════════════════════════════════════

def print_model_report(checkpoint: Path, onnx_path: Optional[Path] = None) -> None:
    """Print a comprehensive model summary."""
    import onnx

    G = _load_generator(checkpoint, torch.device("cpu"))
    params = G.n_parameters

    print("\n╔══════════════════════════════════════════╗")
    print("║      Makeup AI  Phase 3 — Model Report  ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Checkpoint:       {checkpoint}")
    print(f"  Generator params: {params['total']:,} total  /  {params['trainable']:,} trainable")

    # Estimate model size in memory
    size_mb = sum(
        p.numel() * p.element_size()
        for p in G.parameters()
    ) / 1e6
    print(f"  Memory (FP32):    {size_mb:.1f} MB")

    if onnx_path and onnx_path.exists():
        model_proto  = onnx.load(str(onnx_path))
        n_nodes      = len(model_proto.graph.node)
        n_inputs     = len(model_proto.graph.input)
        n_outputs    = len(model_proto.graph.output)
        onnx_mb      = onnx_path.stat().st_size / 1e6
        print(f"\n  ONNX model:       {onnx_path}")
        print(f"  ONNX size:        {onnx_mb:.1f} MB")
        print(f"  ONNX nodes:       {n_nodes}")
        print(f"  ONNX inputs:      {n_inputs}  (source, reference)")
        print(f"  ONNX outputs:     {n_outputs}  (makeup output)")

    print("\n  Input:   [B, 3, H, W]  source face (no makeup)  in [-1, 1]")
    print("           [B, 3, H, W]  reference face (makeup)  in [-1, 1]")
    print("  Output:  [B, 3, H, W]  source with makeup       in [-1, 1]")
    print()


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Makeup AI Phase 3 — ONNX Export + Benchmark"
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to trained .pt checkpoint")
    p.add_argument("--output",     type=Path, default=None,
                   help="Output .onnx path (default: same dir as checkpoint)")
    p.add_argument("--size",       type=int,  default=256,
                   help="Input image size in pixels (default: 256)")
    p.add_argument("--opset",      type=int,  default=17,
                   help="ONNX opset version (default: 17)")
    p.add_argument("--no-simplify",action="store_true",
                   help="Skip onnxsim simplification")
    p.add_argument("--quantize",   action="store_true",
                   help="Also export INT8 quantised model")
    p.add_argument("--benchmark",  action="store_true",
                   help="Benchmark PyTorch vs ONNX latency")
    p.add_argument("--report",     action="store_true",
                   help="Print model report")
    p.add_argument("--runs",       type=int,  default=50,
                   help="Number of benchmark runs (default: 50)")
    args = p.parse_args()

    # Resolve output path
    onnx_path = args.output or (
        args.checkpoint.parent / f"{args.checkpoint.stem}.onnx"
    )

    # Always export
    export(
        checkpoint  = args.checkpoint,
        output_path = onnx_path,
        image_size  = args.size,
        opset       = args.opset,
        simplify    = not args.no_simplify,
    )

    # Optional INT8 quantisation
    if args.quantize:
        quantize_int8(onnx_path)

    # Optional benchmark
    if args.benchmark:
        benchmark(
            checkpoint = args.checkpoint,
            onnx_path  = onnx_path,
            image_size = args.size,
            n_runs     = args.runs,
        )

    # Optional report
    if args.report:
        print_model_report(args.checkpoint, onnx_path)


if __name__ == "__main__":
    main()
