"""
utils/histogram.py — Makeup Histogram Matching Utility
════════════════════════════════════════════════════════
Standalone utility for histogram-based colour transfer.
Used both during training (as part of HistogramLoss) and
at inference time to post-process GAN outputs for tighter
colour fidelity to the reference makeup.

Two modes:
  1. Soft (differentiable) — used in training loss
  2. Hard (exact CDF matching) — used in post-processing

Hard histogram matching is the classic Reinhard et al. technique
applied per makeup region independently.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────
#  LANDMARK INDICES (consistent with Phase 2 + Phase 3)
# ─────────────────────────────────────────────────────────
_LIPS_OUTER       = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95]
_LEFT_EYE_SHADOW  = [226,247,30,29,27,28,56,190,243,112,26,22,23,24,110,25]
_RIGHT_EYE_SHADOW = [446,467,260,259,257,258,286,414,463,341,256,252,253,254,339,255]
_LEFT_CHEEK       = [116,123,147,213,192,214,210,211]
_RIGHT_CHEEK      = [345,352,376,433,416,434,430,431]


# ─────────────────────────────────────────────────────────
#  MASK BUILDING
# ─────────────────────────────────────────────────────────

def build_makeup_mask(
    image_bgr: np.ndarray,
    regions:   list[list[int]],
    landmarks: list,              # MediaPipe landmark list
) -> np.ndarray:
    """
    Build a binary mask for a set of landmark regions.

    Args:
        image_bgr:  Source image (for shape)
        regions:    List of landmark index lists
        landmarks:  MediaPipe face_landmarks[0].landmark

    Returns:
        mask: [H, W] uint8, 255 = makeup region
    """
    h, w  = image_bgr.shape[:2]
    mask  = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        pts = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in region],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)
    return mask


# ─────────────────────────────────────────────────────────
#  HARD HISTOGRAM MATCHING  (CDF-based, not differentiable)
# ─────────────────────────────────────────────────────────

def match_histograms_region(
    source_bgr:    np.ndarray,
    reference_bgr: np.ndarray,
    mask:          np.ndarray,
    strength:      float = 1.0,
) -> np.ndarray:
    """
    Transfer the colour histogram of `reference_bgr` (inside `mask`)
    onto `source_bgr` (inside `mask`) using CDF matching.

    Applied per channel independently.
    Outside the mask, source pixels are untouched.

    Args:
        source_bgr:    Source image to modify
        reference_bgr: Reference image whose colours to match
        mask:          [H, W] uint8 binary mask (255 = region)
        strength:      Blend factor 0.0–1.0 (1.0 = full match)

    Returns:
        Modified source image with histogram-matched region
    """
    assert source_bgr.shape == reference_bgr.shape, \
        "source and reference must have the same shape"
    assert mask.shape == source_bgr.shape[:2], \
        "mask shape must match image H×W"

    result = source_bgr.copy().astype(np.float32)
    src_f  = source_bgr.astype(np.float32)
    ref_f  = reference_bgr.astype(np.float32)

    bool_mask = mask > 127

    for c in range(3):
        src_vals = src_f[bool_mask, c]
        ref_vals = ref_f[bool_mask, c]

        if src_vals.size < 5 or ref_vals.size < 5:
            continue   # too few pixels — skip channel

        # Build CDFs
        src_hist, bins = np.histogram(src_vals, bins=256, range=(0, 256), density=True)
        ref_hist, _    = np.histogram(ref_vals, bins=256, range=(0, 256), density=True)

        src_cdf = np.cumsum(src_hist)
        ref_cdf = np.cumsum(ref_hist)

        # Normalise to [0, 1]
        src_cdf = src_cdf / (src_cdf[-1] + 1e-8)
        ref_cdf = ref_cdf / (ref_cdf[-1] + 1e-8)

        # Build lookup table: for each src intensity, find matching ref intensity
        lut = np.zeros(256, dtype=np.float32)
        j   = 0
        for i in range(256):
            while j < 255 and ref_cdf[j] < src_cdf[i]:
                j += 1
            lut[i] = float(j)

        # Apply LUT to masked pixels only
        matched = np.interp(src_vals, np.arange(256), lut)

        # Blend original and matched by strength
        blended = src_vals * (1.0 - strength) + matched * strength

        result[bool_mask, c] = blended

    return np.clip(result, 0, 255).astype(np.uint8)


def match_makeup_colours(
    source_bgr:    np.ndarray,
    reference_bgr: np.ndarray,
    landmarks_src: list,
    landmarks_ref: list,
    strength:      float = 0.8,
) -> np.ndarray:
    """
    Match makeup colours on all regions (lips, eyes, blush) simultaneously.

    Args:
        source_bgr:    No-makeup face to apply matching to
        reference_bgr: Makeup reference face
        landmarks_src: MediaPipe landmarks for source
        landmarks_ref: MediaPipe landmarks for reference
        strength:      Histogram match strength 0.0–1.0

    Returns:
        source with makeup regions histogram-matched to reference
    """
    result = source_bgr.copy()
    regions = [
        _LIPS_OUTER,
        _LEFT_EYE_SHADOW,
        _RIGHT_EYE_SHADOW,
        _LEFT_CHEEK,
        _RIGHT_CHEEK,
    ]

    # Build masks for source (region to modify) and reference (colour source)
    mask_src = build_makeup_mask(source_bgr,    regions, landmarks_src)
    mask_ref = build_makeup_mask(reference_bgr, regions, landmarks_ref)

    # Match source region → reference region histogram
    result = match_histograms_region(result, reference_bgr, mask_src, strength)
    return result


# ─────────────────────────────────────────────────────────
#  POST-PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────

def postprocess_neural_output(
    generated_bgr:  np.ndarray,
    reference_bgr:  np.ndarray,
    source_bgr:     np.ndarray,
    hist_strength:  float = 0.5,
    blend_strength: float = 0.9,
    face_mesh=None,
) -> np.ndarray:
    """
    Post-process GAN output to improve colour accuracy.

    Pipeline:
      1. Histogram match generated → reference on makeup regions
      2. Blend with original source to reduce identity drift

    Args:
        generated_bgr:  Raw GAN output
        reference_bgr:  Makeup reference used for generation
        source_bgr:     Original no-makeup face
        hist_strength:  How strongly to histogram-match (0–1)
        blend_strength: How much of generated vs source (0–1)
        face_mesh:      Optional pre-created MediaPipe FaceMesh instance.
                        [R17] Pass a reusable instance to avoid the ~200ms
                        FaceMesh init overhead on every call.
                        If None, a temporary one is created and closed.

    Returns:
        Post-processed BGR image
    """
    import mediapipe as mp

    # [R17] Reuse caller-provided FaceMesh; only create+close if not provided
    _owns_fm = face_mesh is None
    if _owns_fm:
        mp_fm    = mp.solutions.face_mesh
        face_mesh = mp_fm.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
        )

    try:
        def _landmarks(bgr: np.ndarray):
            r = face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            return r.multi_face_landmarks[0].landmark if r.multi_face_landmarks else None

        lm_gen = _landmarks(generated_bgr)
        lm_ref = _landmarks(reference_bgr)

        result = generated_bgr.copy()

        # Step 1: histogram match makeup regions to reference
        if lm_gen is not None and lm_ref is not None and hist_strength > 0:
            result = match_makeup_colours(
                result, reference_bgr,
                lm_gen, lm_ref,
                strength=hist_strength,
            )

        # Step 2: blend back some of the source to preserve identity
        if blend_strength < 1.0 and source_bgr.shape == result.shape:
            h, w = source_bgr.shape[:2]
            result = cv2.resize(result, (w, h))
            result = cv2.addWeighted(
                result,     blend_strength,
                source_bgr, 1.0 - blend_strength,
                0,
            )

        return result

    finally:
        if _owns_fm:
            face_mesh.close()


# ─────────────────────────────────────────────────────────
#  COLOUR STATISTICS UTILS
# ─────────────────────────────────────────────────────────

def region_colour_stats(
    image_bgr: np.ndarray,
    mask:      np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute mean and std of BGR channels inside mask region.
    Useful for adaptive colour normalisation.
    """
    bool_mask = mask > 127
    if not bool_mask.any():
        return {"mean": np.zeros(3), "std": np.ones(3)}

    pixels = image_bgr[bool_mask].astype(np.float32)   # [N, 3]
    return {
        "mean": pixels.mean(axis=0),
        "std":  pixels.std(axis=0) + 1e-8,
        "n":    len(pixels),
    }


def colour_distance(
    stats_a: dict[str, np.ndarray],
    stats_b: dict[str, np.ndarray],
) -> float:
    """
    L2 distance between mean BGR colours of two regions.
    Used to score how well generated makeup matches reference.
    Lower = better match.
    """
    return float(np.linalg.norm(stats_a["mean"] - stats_b["mean"]))
