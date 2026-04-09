"""
utils/common.py — Shared Utilities
════════════════════════════════════
Cross-cutting helpers used by trainer, inference, and onnx_export.

Extracted to fix R15: resolve_device() was duplicated in trainer.py
and inference.py. Single source of truth lives here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """
    Resolve a device string to a torch.device.

    Args:
        spec: "auto" | "cuda" | "mps" | "cpu" | "cuda:0" etc.

    "auto" selects: CUDA > MPS (Apple Silicon) > CPU in that order.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def ensure_package_root_on_path() -> None:
    """
    Add the package root (two levels up from utils/) to sys.path
    exactly once, so sibling packages (models, training, api) are
    importable regardless of the working directory.

    Guards against repeated insertion (R7, R16).
    """
    pkg_root = str(Path(__file__).resolve().parent.parent)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
