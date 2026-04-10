"""
lighting_estimator.py — Scene Lighting Estimation & Makeup Color Adaptation
═════════════════════════════════════════════════════════════════════════════
Gap Fix #2: Makeup color adapts to scene illumination.

THE PROBLEM:
  We apply the same saturated BGR color to lips/eyes regardless of
  ambient lighting. In a dim room, "red" lipstick (BGR 30,20,200)
  renders as a bright, glowing, physically-impossible saturated red.
  Under a warm incandescent bulb, the same red should look slightly
  orange-brown. Under fluorescent light, slightly cool. At night,
  dark burgundy. Banuba does this automatically using AI lighting
  estimation — we replicate it with classic image processing.

OUR APPROACH — THREE-STAGE PIPELINE:

  Stage 1: Grey-World Illuminant Estimation
    Compute the per-channel mean of the full frame (excluding
    highly saturated pixels which are scene objects, not neutral).
    The deviation of this mean from (128, 128, 128) tells us the
    scene color temperature. A yellow/warm mean → warm light.
    A blue mean → cold/fluorescent.

  Stage 2: Skin-region Brightness Analysis
    Sample brightness (V channel in HSV) from known skin-region
    landmarks. This tells us overall exposure: is the scene well-lit
    or is this a dim environment? Used to scale makeup opacity — in
    dim scenes makeup needs higher alpha to be visible.

  Stage 3: Makeup Color Adaptation
    For each makeup color:
    a. Convert BGR → LAB
    b. Shift L (lightness) down in dim scenes (makeup looks darker)
    c. Shift A/B channels toward the illuminant tint
       (warm light adds yellow, cool light adds blue)
    d. Convert back to BGR

This runs once per frame (not per landmark) and takes ~0.5ms.

TEMPORAL SMOOTHING:
    The illuminant estimate is EMA-smoothed across frames (alpha=0.1)
    to prevent flickering when the scene lighting is stable.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np


class LightingEstimator:
    """
    Estimates scene illumination from a webcam frame and returns
    adjusted makeup colors that appear physically correct.

    Usage:
        estimator = LightingEstimator()

        # In render loop, before applying makeup:
        scene = estimator.analyze(frame)

        # Then adapt each color:
        adapted_lip_color = estimator.adapt_color(lip_color, scene)
        adapted_eye_color = estimator.adapt_color(eye_color, scene)
    """

    # Landmarks used for brightness sampling (stable mid-cheek skin area)
    _SKIN_SAMPLE = [117, 118, 119, 120, 121, 346, 347, 348, 349, 350]

    def __init__(
        self,
        ema_alpha:       float = 0.08,   # smoothing across frames (lower = smoother)
        sat_threshold:   float = 0.60,   # exclude highly-saturated pixels from grey-world
        brightness_min:  float = 0.25,   # below this: "dark scene" → boost opacity
        brightness_max:  float = 0.80,   # above this: "bright scene"
    ) -> None:
        self._ema_alpha  = ema_alpha
        self._sat_thresh = sat_threshold
        self._br_min     = brightness_min
        self._br_max     = brightness_max

        # Smoothed illuminant state
        self._illum_b = 128.0   # smoothed blue channel mean
        self._illum_g = 128.0
        self._illum_r = 128.0
        self._brightness  = 0.5   # smoothed scene brightness [0,1]
        self._initialized = False

    # ── Main analysis ─────────────────────────────────────

    def analyze(
        self,
        frame_bgr:  np.ndarray,
        landmarks:  Optional[list] = None,
        frame_hw:   Optional[tuple] = None,
    ) -> dict:
        """
        Analyze scene lighting from the current frame.

        Args:
            frame_bgr:  Current BGR frame
            landmarks:  Optional MediaPipe landmark list (for skin sampling)
            frame_hw:   (h, w) if landmarks are provided

        Returns:
            scene dict with keys:
              'illuminant_shift': (db, dg, dr) LAB shift to apply to colors
              'brightness':       float [0,1] scene brightness
              'warmth':           float [-1,1] negative=cool, positive=warm
              'opacity_scale':    float [0.8, 1.3] multiply makeup alpha by this
              'lab_ab_shift':     (da, db) LAB chrominance offset for adaptation
        """
        h, w = frame_bgr.shape[:2]

        # ── Stage 1: Grey-world illuminant ───────────────
        # Only use low-saturation pixels (excludes colorful objects)
        hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat  = hsv[..., 1] / 255.0
        mask = sat < self._sat_thresh   # True = low-saturation (near-grey)

        if mask.sum() < 100:
            # Too little neutral content — use full frame
            mask = np.ones((h, w), dtype=bool)

        b_mean = float(frame_bgr[..., 0][mask].mean())
        g_mean = float(frame_bgr[..., 1][mask].mean())
        r_mean = float(frame_bgr[..., 2][mask].mean())

        # ── Stage 2: Skin brightness ─────────────────────
        if landmarks is not None and frame_hw is not None:
            fh, fw = frame_hw
            skin_v = []
            for idx in self._SKIN_SAMPLE:
                if idx < len(landmarks):
                    lm  = landmarks[idx]
                    px  = int(np.clip(lm.x * fw, 0, fw - 1))
                    py  = int(np.clip(lm.y * fh, 0, fh - 1))
                    skin_v.append(float(hsv[py, px, 2]) / 255.0)
            brightness_raw = float(np.median(skin_v)) if skin_v else float(hsv[..., 2].mean() / 255.0)
        else:
            brightness_raw = float(hsv[..., 2].mean() / 255.0)

        # ── EMA smoothing ─────────────────────────────────
        α = self._ema_alpha
        if not self._initialized:
            self._illum_b    = b_mean
            self._illum_g    = g_mean
            self._illum_r    = r_mean
            self._brightness = brightness_raw
            self._initialized = True
        else:
            self._illum_b    = α * b_mean    + (1 - α) * self._illum_b
            self._illum_g    = α * g_mean    + (1 - α) * self._illum_g
            self._illum_r    = α * r_mean    + (1 - α) * self._illum_r
            self._brightness = α * brightness_raw + (1 - α) * self._brightness

        # ── Compute adaptation parameters ─────────────────
        # Deviation from neutral grey (128,128,128)
        db = self._illum_b - 128.0   # positive = blue cast
        dg = self._illum_g - 128.0
        dr = self._illum_r - 128.0

        # Warmth: positive = warm (red/yellow dominant), negative = cool (blue)
        warmth = float(np.clip((dr - db) / 80.0, -1.0, 1.0))

        # LAB chrominance shift: move makeup color toward scene illuminant
        # A channel: green(-) ↔ red(+). B channel: blue(-) ↔ yellow(+)
        # Warm light adds yellow/red → positive B, positive A shift
        # Cool light adds blue → negative B shift
        lab_da = float(np.clip(dr * 0.12, -10, 10))    # red channel → A axis
        lab_db = float(np.clip((dr - db) * 0.08, -8, 8))  # warmth → B axis

        # Opacity scale: dim scene → boost alpha so makeup is visible
        br = self._brightness
        if br < self._br_min:
            opacity_scale = 1.25   # very dark → boost 25%
        elif br < 0.40:
            opacity_scale = 1.10
        elif br > self._br_max:
            opacity_scale = 0.90   # very bright → reduce slightly (overexposed)
        else:
            opacity_scale = 1.0

        return {
            "illuminant_bgr":  (self._illum_b, self._illum_g, self._illum_r),
            "illuminant_shift": (db, dg, dr),
            "brightness":       self._brightness,
            "warmth":           warmth,
            "opacity_scale":    opacity_scale,
            "lab_ab_shift":     (lab_da, lab_db),
        }

    # ── Color adaptation ──────────────────────────────────

    def adapt_color(
        self,
        bgr_color:    tuple,
        scene:        dict,
        strength:     float = 0.40,   # how strongly to adapt [0=none, 1=full]
    ) -> tuple:
        """
        Adapt a BGR makeup color to match the scene illumination.

        Args:
            bgr_color: Original BGR tuple (e.g. (30, 20, 200) for red lipstick)
            scene:     Output from analyze()
            strength:  How strongly to apply adaptation (0.4 = subtle, natural)

        Returns:
            Adapted BGR tuple
        """
        if strength <= 0:
            return bgr_color

        # Convert scalar color to 1×1 pixel for cv2 conversion
        px = np.array([[[bgr_color[0], bgr_color[1], bgr_color[2]]]], dtype=np.uint8)
        lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB).astype(np.float32)

        da, db = scene["lab_ab_shift"]
        brightness = scene["brightness"]

        # Brightness adaptation: darken makeup in dim scenes
        # (too-bright makeup in dark rooms looks obviously fake)
        l_shift = 0.0
        if brightness < 0.35:
            l_shift = -12.0 * (0.35 - brightness) / 0.35   # up to -12 L units

        lab[0, 0, 0] = np.clip(lab[0, 0, 0] + l_shift * strength, 0, 255)
        lab[0, 0, 1] = np.clip(lab[0, 0, 1] + da * strength, 0, 255)
        lab[0, 0, 2] = np.clip(lab[0, 0, 2] + db * strength, 0, 255)

        adapted = cv2.cvtColor(np.round(lab).clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        b, g, r = int(adapted[0, 0, 0]), int(adapted[0, 0, 1]), int(adapted[0, 0, 2])
        return (b, g, r)

    def adapt_alpha(self, base_alpha: float, scene: dict) -> float:
        """Scale opacity based on scene brightness."""
        return float(np.clip(base_alpha * scene["opacity_scale"], 0.05, 0.95))

    def reset(self) -> None:
        """Reset smoothing state (e.g. when camera restarts)."""
        self._initialized = False
