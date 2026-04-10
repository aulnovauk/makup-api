"""
app.py — Makeup AI Web Application  v3 (Production)
══════════════════════════════════════════════════════
ROOT CAUSE OF "keep processing" STALL:
  The old design was fully synchronous — the browser sent a frame,
  waited for the server to finish (150–400ms), received the result,
  THEN sent the next frame. This gated the entire live feed on
  server latency. At 720p it was completely blocked.

PRODUCTION FIX — DECOUPLED PIPELINE:

  OLD (broken):
    capture → POST → wait 300ms → render result → capture next frame
    Effective FPS = 1000 / server_latency = ~3 fps at 720p

  NEW (production):
    rAF loop  ──── draw last result immediately (60 fps display) ──▶
                                                                      Browser
    push loop ──── POST frame every 100ms (fire-and-forget) ────────▶
                                                                      Server
    SSE listener ◀── receive result whenever ready ──────────────────
    Ring buffer: if server is slow, newest frame replaces old one.
    Never queue up. Never block. Never stutter.

  Key properties:
  • rAF render loop runs at monitor Hz (60 fps) regardless of server
  • Browser never awaits server — POST is fire-and-forget (202 Accepted)
  • Only ONE frame is in-flight at a time — ring buffer of size 1
  • Results come back via SSE push — browser just listens
  • Adaptive resolution: auto-adjusts send size based on latency feedback
    640px if avg_ms < 80 → 480px if 80-200ms → 320px if >200ms
  • Thread-safe: RLock around MediaPipe per request

ROUTES:
  GET  /                  → UI
  GET  /health            → status + latency stats
  POST /api/stream/push   → push camera frame (fire-and-forget, 202)
  GET  /api/stream        → SSE stream of processed results
  POST /api/apply         → single-shot sync (photo mode + compat)
  POST /api/photo         → photo file upload
  GET  /api/presets       → colour presets
  POST /api/screenshot    → save screenshot
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# ── Logging (must be set up before any log.* calls) ───────
_ROOT = Path(__file__).resolve().parent
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_ROOT / "webapp.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("MakeupAI.Web")

# Ensure project root is on path BEFORE importing local modules
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Gap fixes — local modules (imported AFTER logging + sys.path are ready)
try:
    from landmark_smoother import MultiFaceSmoother
    from lighting_estimator import LightingEstimator
    _GAP_FIXES_AVAILABLE = True
except ImportError:
    _GAP_FIXES_AVAILABLE = False
    log.warning("landmark_smoother / lighting_estimator not found — gap fixes disabled")


# ═══════════════════════════════════════════════════════════
#  LANDMARK INDICES + PALETTES
# ═══════════════════════════════════════════════════════════

LIPSTICK_COLORS  = {"red_rose":(30,20,200),"pink_blush":(150,100,220),"coral_sunset":(80,100,240),"berry_wine":(60,20,130),"nude_beige":(130,150,190),"deep_plum":(80,30,100)}
EYESHADOW_COLORS = {"smoky_brown":(40,60,100),"rose_gold":(100,120,200),"ocean_blue":(180,100,50),"forest_green":(60,130,60),"purple_haze":(150,60,130)}
BLUSH_COLORS     = {"soft_rose":(140,130,230),"peach":(110,160,240),"berry":(100,80,180),"bronze":(80,110,170)}
ALL_PRESETS      = {
    "lipstick":  {k:{"bgr":list(v)} for k,v in LIPSTICK_COLORS.items()},
    "eyeshadow": {k:{"bgr":list(v)} for k,v in EYESHADOW_COLORS.items()},
    "blush":     {k:{"bgr":list(v)} for k,v in BLUSH_COLORS.items()},
}

FINISH_MATTE    = "matte"
FINISH_GLOSS    = "gloss"
FINISH_METALLIC = "metallic"
FINISH_SHIMMER  = "shimmer" 

# ── Lip regions ──────────────────────────────────────────
# Full outer boundary (canonical MediaPipe lip contour)
LIPS_OUTER = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
# Inner mouth opening (used to cut a hole so lip fill stays on lip, not beard gap)
LIPS_INNER_UPPER = [78,191,80,81,82,13,312,311,310,415,308]
LIPS_INNER_LOWER = [78,95,88,178,87,14,317,402,318,324,308]

# ── Eye shadow — TIGHT to the eyelid crease only ─────────
# These indices follow the upper eyelid fold precisely.
# Do NOT use indices 226,190,243 — they map to temple/outer corner
# and produce large polygons that fall onto the cheek.
LEFT_EYE_SHADOW  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
RIGHT_EYE_SHADOW = [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249]

# ── Extended shadow (includes socket/crease area) ─────────
# Used when opacity > 0.4 — deeper, more dramatic look
LEFT_EYE_SHADOW_DEEP  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,
                          110,24,23,22,26,112,243,190,56,28,27,29,30,247,226]
RIGHT_EYE_SHADOW_DEEP = [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249,
                          339,254,253,252,256,341,463,414,286,258,257,259,260,467,446]

# ── Eyebrows ──────────────────────────────────────────────
LEFT_EYEBROW  = [70,63,105,66,107,55,65,52,53,46]
RIGHT_EYEBROW = [300,293,334,296,336,285,295,282,283,276]

# ── Cheeks — verified not overlapping with eye shadow ─────
# These sit on the cheekbone, well below the eye region
LEFT_CHEEK  = [187,207,206,205,50,36,100,101,116,123,147,192,214,210]
RIGHT_CHEEK = [411,427,426,425,280,266,329,330,345,352,376,416,434,430]

# ── Other regions ─────────────────────────────────────────
NOSE_BRIDGE = [6,197,195,5,4]
FACE_OVAL   = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
SKIN_SAMPLE_L = [117,118,119,120,121]
SKIN_SAMPLE_R = [346,347,348,349,350]

# ── Cupid's bow highlight ─────────────────────────────────
CUPIDS_BOW = [0,37,39,40,185,61,146,91,181,84]


# ═══════════════════════════════════════════════════════════
#  MAKEUP ENGINE  (thread-safe)
# ═══════════════════════════════════════════════════════════

class WebMakeupEngine:
    """MediaPipe is not re-entrant — a threading.Lock gates all calls."""

    def __init__(self) -> None:
        import mediapipe as mp
        self._mp   = mp
        self._lock = threading.Lock()

        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_ROOT / "face_landmarker.task")),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker  = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        self._mask_buf    = None
        self._frame_shape = None

        # Gap Fix #1: One-Euro landmark smoothing
        if _GAP_FIXES_AVAILABLE:
            self._smoother  = MultiFaceSmoother(max_faces=2, n_landmarks=478,
                                                min_cutoff=1.0, beta=0.05)
            self._estimator = LightingEstimator(ema_alpha=0.08)
        else:
            self._smoother  = None
            self._estimator = None

        log.info("WebMakeupEngine ready ✓")

    def __del__(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass

    # ── internal helpers ──────────────────────────────────

    def _ensure_buf(self, frame: np.ndarray) -> None:
        if self._frame_shape != frame.shape:
            self._mask_buf    = np.zeros(frame.shape[:2], dtype=np.uint8)
            self._frame_shape = frame.shape

    @staticmethod
    def _to_pts(lm, indices, w, h):
        pts = np.empty((len(indices), 2), dtype=np.int32)
        for i, idx in enumerate(indices):
            pts[i, 0] = int(np.clip(lm[idx].x * w, 0, w - 1))
            pts[i, 1] = int(np.clip(lm[idx].y * h, 0, h - 1))
        return pts

    @staticmethod
    def _visible(lm, indices, thresh=0.3):
        valid = [lm[i].visibility for i in indices[:5]
                 if lm[i].visibility is not None and lm[i].visibility > 0]
        return True if not valid else (sum(v > thresh for v in valid) / len(valid)) >= 0.4

    def _region(self, frame, pts, color, alpha, blur_d=9, erode_px=0):
        """
        Blend a makeup color over a landmark region.
        
        Strategy — "Screen + Multiply" cosmetic blend:
          1. Fill polygon mask, optionally erode to prevent edge bleeding
          2. Gaussian blur the mask for soft edges
          3. Convert both frame and color to LAB:
             - Keep L (luminance) from original skin — preserves texture
             - Blend A and B channels toward the color — adds the tint
          4. Result: vivid color that follows skin contours naturally
        
        erode_px: shrink mask before blurring to prevent color leaking
                  outside the true anatomical boundary.
        """
        if len(pts) < 3:
            return frame
        self._mask_buf[:] = 0
        cv2.fillPoly(self._mask_buf, [pts], 255)
        
        # Erode to pull mask away from hard anatomical edges
        # Use a local variable — do NOT reassign self._mask_buf (shared buffer)
        if erode_px > 0:
            k_e   = max(3, erode_px * 2 + 1)   # kernel must be odd and ≥ erode_px
            k_el  = np.ones((k_e, k_e), np.uint8)
            eroded = cv2.erode(self._mask_buf, k_el, iterations=1)
        else:
            eroded = self._mask_buf
        
        # Soft gaussian feather (use eroded mask for tight edges)
        ksize = max(3, (blur_d // 2) * 2 + 1)
        soft  = cv2.GaussianBlur(
            eroded.astype(np.float32), (ksize, ksize), blur_d * 0.5
        )
        mx   = soft.max()
        soft = np.clip(soft / (mx + 1e-6), 0.0, 1.0)
        
        # LAB blend: keep skin luminance, shift chrominance toward color
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        color_bgr = np.array([[[color[0], color[1], color[2]]]], dtype=np.uint8)
        color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        out_lab = frame_lab.copy()
        # L: slight darkening toward color lightness (makeup has depth)
        out_lab[..., 0] = frame_lab[..., 0] * (1 - alpha * soft * 0.15) +                           color_lab[0, 0, 0] * (alpha * soft * 0.15)
        # A channel (green-red axis)
        out_lab[..., 1] = frame_lab[..., 1] * (1 - alpha * soft) +                           color_lab[0, 0, 1] * (alpha * soft)
        # B channel (blue-yellow axis)
        out_lab[..., 2] = frame_lab[..., 2] * (1 - alpha * soft) +                           color_lab[0, 0, 2] * (alpha * soft)
        
        result = cv2.cvtColor(
            np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
        )
        
        return result

    def _region_with_hole(self, frame, outer_pts, hole_pts, color, alpha, blur_d=5):
        """
        Fill outer polygon MINUS inner polygon (donut shape).
        Used for lips: fills the lip area but excludes the mouth opening
        and inter-lip gap, preventing color from falling on beard/chin.
        """
        if len(outer_pts) < 3:
            return frame
        self._mask_buf[:] = 0
        cv2.fillPoly(self._mask_buf, [outer_pts], 255)
        # Cut hole for mouth opening
        if len(hole_pts) >= 3:
            cv2.fillPoly(self._mask_buf, [hole_pts], 0)
        
        ksize = max(3, (blur_d // 2) * 2 + 1)
        soft  = cv2.GaussianBlur(self._mask_buf.astype(np.float32), (ksize, ksize), blur_d * 0.5)
        soft  = np.clip(soft / (soft.max() + 1e-6), 0.0, 1.0)
        
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        color_bgr = np.array([[[color[0], color[1], color[2]]]], dtype=np.uint8)
        color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        out_lab = frame_lab.copy()
        out_lab[..., 0] = frame_lab[..., 0] * (1 - alpha * soft * 0.2) + color_lab[0,0,0] * (alpha * soft * 0.2)
        out_lab[..., 1] = frame_lab[..., 1] * (1 - alpha * soft) + color_lab[0,0,1] * (alpha * soft)
        out_lab[..., 2] = frame_lab[..., 2] * (1 - alpha * soft) + color_lab[0,0,2] * (alpha * soft)
        
        result = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result

    def _gloss(self, frame, pts):
        return self._apply_finish(frame, pts, mode=FINISH_GLOSS)

    def _apply_finish(self, frame, pts, mode=None, hole_pts=None):
        if mode is None: mode = FINISH_GLOSS
        if len(pts) < 3: return frame
        self._mask_buf[:] = 0
        cv2.fillPoly(self._mask_buf, [pts], 255)
        # Cut inner hole (e.g. mouth opening for lip finish) to prevent
        # gloss/metallic/shimmer from bleeding into beard-gap area
        if hole_pts is not None and len(hole_pts) >= 3:
            cv2.fillPoly(self._mask_buf, [hole_pts], 0)
        if mode == FINISH_MATTE:
            region = self._mask_buf > 127
            if not region.any(): return frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[region, 1] = np.clip(hsv[region, 1] * 1.08, 0, 255)
            v_mean = float(hsv[region, 2].mean())
            hsv[region, 2] = hsv[region, 2] * 0.6 + v_mean * 0.4
            result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
            soft = cv2.GaussianBlur(self._mask_buf.astype(np.float32), (7,7), 3) / 255.0 * 0.7
            s3 = cv2.merge([soft, soft, soft])
            return np.clip(frame.astype(np.float32)*(1-s3) + result.astype(np.float32)*s3, 0, 255).astype(np.uint8)
        elif mode == FINISH_GLOSS:
            hi = cv2.erode(self._mask_buf, np.ones((5,5),np.uint8), iterations=3)
            hi = cv2.GaussianBlur(hi, (13,13), 7)
            hl = np.full_like(frame, (250, 250, 255), np.uint8)
            a  = hi.astype(np.float32) / 255 * 0.28
            a3 = cv2.merge([a, a, a])
            return np.clip(frame.astype(np.float32)*(1-a3) + hl.astype(np.float32)*a3, 0, 255).astype(np.uint8)
        elif mode == FINISH_METALLIC:
            h, w = frame.shape[:2]
            M = cv2.moments(self._mask_buf)
            if M["m00"] == 0: return frame
            cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
            # Place highlight upper-right of centroid (simulates overhead-right directional light)
            hx = int(np.clip(cx + w * 0.04, 0, w - 1))
            hy = int(np.clip(cy - h * 0.03, 0, h - 1))
            # Build highlight: place point, Gaussian spread, then mask to region
            # Order matters: mask FIRST, then check if any remains > 0
            hi_buf = np.zeros(frame.shape[:2], dtype=np.float32)
            # Check if highlight centre is inside the mask; if not, use centroid
            if self._mask_buf[hy, hx] == 0:
                hx, hy = cx, cy
            hi_buf[hy, hx] = 255.0
            hi_buf = cv2.GaussianBlur(hi_buf, (25, 25), 10)
            hi_buf[self._mask_buf == 0] = 0.0   # mask AFTER blur for soft falloff
            mx = hi_buf.max()
            if mx < 1.0:   # guard: no valid highlight after masking
                return frame
            hi_buf = hi_buf / mx
            hl = np.full_like(frame, (255, 255, 245), np.uint8)
            a  = hi_buf * 0.55
            a3 = cv2.merge([a, a, a])
            return np.clip(frame.astype(np.float32)*(1-a3) + hl.astype(np.float32)*a3, 0, 255).astype(np.uint8)
        elif mode == FINISH_SHIMMER:
            h, w = frame.shape[:2]
            # Seed changes ~8x per second so sparkle animates without full
            # randomness each frame (which would look like noise, not shimmer)
            frame_seed = int(time.perf_counter() * 8) % 1000
            rng    = np.random.RandomState(frame_seed)
            n_pts  = 150
            xs = rng.randint(0, w, n_pts); ys = rng.randint(0, h, n_pts)
            hi_buf = np.zeros(frame.shape[:2], dtype=np.float32)
            for px, py in zip(xs, ys):
                if self._mask_buf[py, px] > 0:
                    hi_buf[py, px] = float(rng.uniform(0.5, 1.0))
            hi_buf = cv2.GaussianBlur(hi_buf, (5, 5), 2)
            mx = hi_buf.max()
            if mx > 1e-6:
                hi_buf = hi_buf / mx * 0.40
            hl = np.full_like(frame, (255, 255, 230), np.uint8)
            a3 = cv2.merge([hi_buf, hi_buf, hi_buf])
            return np.clip(frame.astype(np.float32)*(1-a3) + hl.astype(np.float32)*a3, 0, 255).astype(np.uint8)
        return frame

    def _skin(self, frame, lm, w, h):
        px = [frame[int(np.clip(lm[i].y*h,0,h-1)), int(np.clip(lm[i].x*w,0,w-1))].tolist()
              for i in SKIN_SAMPLE_L + SKIN_SAMPLE_R]
        b, g, r = np.median(np.array(px, np.float32), axis=0)
        return (int(np.clip(b*1.02,0,255)), int(np.clip(g*1.02,0,255)), int(np.clip(r*1.02,0,255)))

    # ── public API ────────────────────────────────────────

    def render(self, frame: np.ndarray, params: dict, static: bool = False) -> np.ndarray:
        with self._lock:
            return self._render(frame, params)

    def _render(self, frame, params):
        """
        Apply all enabled makeup layers to frame using corrected
        landmark indices and LAB-space color blending.

        Gap fixes applied here:
          #1 — One-Euro landmark smoothing (eliminates jitter)
          #2 — Scene lighting estimation (adapts makeup colors)

        Layer order (bottom to top, matches real makeup application):
          Foundation → Contour → Blush → Eyeshadow → Eyebrow →
          Highlighter → Lipstick → Lip Liner
        """
        self._ensure_buf(frame)
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img  = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res  = self._landmarker.detect(img)
        out  = frame.copy()

        if not res.face_landmarks:
            # Reset smoother when face is lost — prevents stale state
            if self._smoother:
                self._smoother.reset()
            return out

        # Gap Fix #1: Smooth all detected face landmarks
        t = time.perf_counter()
        if self._smoother:
            face_lms = self._smoother.smooth_all(t, res.face_landmarks)
        else:
            face_lms = res.face_landmarks

        # Gap Fix #2: Estimate lighting once per frame from the primary face
        # Calling per-face would double-update the EMA for 2-face sessions
        if self._estimator and face_lms:
            scene = self._estimator.analyze(frame, face_lms[0], (h, w))
        else:
            scene = None

        for lm_idx, lm in enumerate(face_lms):
            p = params

            # ── 1. Foundation & skin smooth ───────────────────────────
            if p.get("foundation", {}).get("enabled"):
                fd = p["foundation"]
                a  = float(fd.get("opacity", 0.15))
                if self._visible(lm, FACE_OVAL):
                    fpts = self._to_pts(lm, FACE_OVAL, w, h)
                    skin = self._skin(frame, lm, w, h)
                    if fd.get("smooth"):
                        # Bilateral filter preserves edges while smoothing pores
                        sf  = cv2.bilateralFilter(out, 9, 55, 55)
                        mf  = np.zeros(out.shape[:2], np.uint8)
                        cv2.fillPoly(mf, [fpts], 255)
                        mf  = cv2.erode(mf, np.ones((7,7), np.uint8), iterations=3)
                        mf  = cv2.GaussianBlur(mf.astype(np.float32), (31,31), 14) / 255.0 * 0.6
                        mf3 = cv2.merge([mf, mf, mf])
                        out = np.clip(
                            out.astype(np.float32)*(1-mf3) + sf.astype(np.float32)*mf3,
                            0, 255
                        ).astype(np.uint8)
                    # Apply foundation tint (very subtle, erode=3 to stay inside face oval)
                    out = self._region(out, fpts, skin, a * 0.35, blur_d=21, erode_px=3)

            # ── 2. Contour ────────────────────────────────────────────
            if p.get("contour", {}).get("enabled"):
                a = float(p["contour"].get("opacity", 0.25))
                contour_color = (55, 75, 105)   # cool taupe in BGR
                contour_regions = [
                    # Left jaw sweep
                    [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152],
                    # Right jaw sweep
                    [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152],
                    # Left cheek hollow (below cheekbone)
                    [116, 123, 147, 213, 192, 214, 210],
                    # Right cheek hollow
                    [345, 352, 376, 433, 416, 434, 430],
                ]
                for region in contour_regions:
                    if self._visible(lm, region):
                        pts = self._to_pts(lm, region, w, h)
                        out = self._region(out, pts, contour_color, a * 0.55, blur_d=17, erode_px=2)

            # ── 3. Blush ──────────────────────────────────────────────
            if p.get("blush", {}).get("enabled"):
                bl    = p["blush"]
                color = tuple(int(c) for c in bl.get("color", [140,130,230]))
                a     = float(bl.get("opacity", 0.20))
                if scene and self._estimator:
                    color = self._estimator.adapt_color(color, scene, strength=0.30)
                    a     = self._estimator.adapt_alpha(a, scene)
                # LEFT_CHEEK / RIGHT_CHEEK now use verified non-overlapping indices
                for cheek_idx in (LEFT_CHEEK, RIGHT_CHEEK):
                    if self._visible(lm, cheek_idx):
                        pts = self._to_pts(lm, cheek_idx, w, h)
                        # erode_px=2 keeps blush from leaking toward eyes
                        out = self._region(out, pts, color, a, blur_d=19, erode_px=2)

            # ── 4. Eyeshadow ──────────────────────────────────────────
            if p.get("eyeshadow", {}).get("enabled"):
                es    = p["eyeshadow"]
                color = tuple(int(c) for c in es.get("color", [40,60,100]))
                a     = float(es.get("opacity", 0.30))
                raw_opacity = a   # preserve user intent for region selection
                if scene and self._estimator:
                    color = self._estimator.adapt_color(color, scene, strength=0.25)
                    a     = self._estimator.adapt_alpha(a, scene)
                # Choose deep indices based on RAW user opacity, not adapted value
                if raw_opacity > 0.40:
                    l_shadow, r_shadow = LEFT_EYE_SHADOW_DEEP, RIGHT_EYE_SHADOW_DEEP
                else:
                    l_shadow, r_shadow = LEFT_EYE_SHADOW, RIGHT_EYE_SHADOW
                for shadow_idx in (l_shadow, r_shadow):
                    if self._visible(lm, shadow_idx):
                        pts = self._to_pts(lm, shadow_idx, w, h)
                        # erode_px=1 critical: prevents shadow leaking below eye onto cheek
                        out = self._region(out, pts, color, a, blur_d=9, erode_px=1)

            # ── 5. Eyebrow fill ───────────────────────────────────────
            if p.get("eyebrow", {}).get("enabled"):
                ey    = p["eyebrow"]
                color = tuple(int(c) for c in ey.get("color", [30,45,80]))
                a     = float(ey.get("opacity", 0.45))
                for brow_idx in (LEFT_EYEBROW, RIGHT_EYEBROW):
                    if self._visible(lm, brow_idx):
                        pts = self._to_pts(lm, brow_idx, w, h)
                        # Tight blur (5) and erode (1) keeps brow color inside brow shape
                        out = self._region(out, pts, color, a, blur_d=5, erode_px=1)

            # ── 6. Highlighter ────────────────────────────────────────
            if p.get("highlighter", {}).get("enabled"):
                a  = float(p["highlighter"].get("opacity", 0.30))
                HL = (240, 245, 255)   # warm pearl white in BGR

                # Nose bridge highlight
                if self._visible(lm, NOSE_BRIDGE):
                    pts = self._to_pts(lm, NOSE_BRIDGE, w, h)
                    out = self._region(out, pts, HL, a * 0.5, blur_d=11)

                # Cheekbone highlight (upper cheek, above blush zone)
                hl_cheeks = (
                    [117,118,119,100,101,50,36,47,114,188,122,6],   # left
                    [346,347,348,329,330,280,266,277,343,412,351,6], # right
                )
                for hl_pts_idx in hl_cheeks:
                    if self._visible(lm, hl_pts_idx):
                        pts = self._to_pts(lm, hl_pts_idx, w, h)
                        out = self._region(out, pts, HL, a * 0.4, blur_d=15, erode_px=1)

                # Cupid's bow highlight (top center of upper lip)
                if self._visible(lm, CUPIDS_BOW):
                    pts = self._to_pts(lm, CUPIDS_BOW, w, h)
                    out = self._region(out, pts, HL, a * 0.25, blur_d=7)

            # ── 7. Lipstick ───────────────────────────────────────────
            if p.get("lipstick", {}).get("enabled"):
                ls    = p["lipstick"]
                color = tuple(int(c) for c in ls.get("color", [30,20,200]))
                a     = float(ls.get("opacity", 0.45))
                # Gap Fix #2: adapt color + alpha to scene lighting
                if scene and self._estimator:
                    color = self._estimator.adapt_color(color, scene, strength=0.35)
                    a     = self._estimator.adapt_alpha(a, scene)
                if self._visible(lm, LIPS_OUTER):
                    outer_pts = self._to_pts(lm, LIPS_OUTER, w, h)
                    # Build inner mouth opening mask (excludes beard gap)
                    # Deduplicate shared endpoints (78 and 308) at corners
                    inner_u   = self._to_pts(lm, LIPS_INNER_UPPER, w, h)  # 11 pts, starts+ends at 78/308
                    inner_l   = self._to_pts(lm, LIPS_INNER_LOWER, w, h)  # 11 pts, starts+ends at 78/308
                    # Skip first point of lower (duplicate of last of upper at 308)
                    # and reverse so polygon winds consistently
                    inner_pts = np.vstack([inner_u, inner_l[1:-1][::-1]])

                    # Apply lipstick with beard-gap exclusion
                    out = self._region_with_hole(out, outer_pts, inner_pts, color, a, blur_d=5)

                    # Apply finish mode
                    finish = ls.get("finish", FINISH_GLOSS)
                    if finish == FINISH_GLOSS:
                        lab_l = cv2.cvtColor(
                            np.array([[[color[0],color[1],color[2]]]], dtype=np.uint8),
                            cv2.COLOR_BGR2LAB
                        )[0,0,0]
                        if lab_l < 40:
                            finish = FINISH_MATTE
                    out = self._apply_finish(out, outer_pts, mode=finish, hole_pts=inner_pts)

            # ── 8. Lip liner ──────────────────────────────────────────
            if p.get("lip_liner", {}).get("enabled"):
                ll    = p["lip_liner"]
                color = tuple(int(c) for c in ll.get("color", [30,20,200]))
                a     = float(ll.get("opacity", 0.60))
                if self._visible(lm, LIPS_OUTER):
                    pts = self._to_pts(lm, LIPS_OUTER, w, h)
                    # Draw a 1px AA polyline into a float mask
                    liner_mask = np.zeros(out.shape[:2], np.float32)
                    cv2.polylines(liner_mask, [pts], True, 1.0, 2, cv2.LINE_AA)
                    liner_mask = cv2.GaussianBlur(liner_mask, (3,3), 1)
                    liner_color = np.full_like(out, color, np.uint8)
                    m3  = cv2.merge([liner_mask*a, liner_mask*a, liner_mask*a])
                    out = np.clip(
                        out.astype(np.float32)*(1-m3) + liner_color.astype(np.float32)*m3,
                        0, 255
                    ).astype(np.uint8)

        return out


# ═══════════════════════════════════════════════════════════
#  LATENCY TRACKER  (feeds adaptive resolution to client)
# ═══════════════════════════════════════════════════════════

class LatencyTracker:
    def __init__(self, window: int = 20) -> None:
        self._s: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, ms: float) -> None:
        with self._lock:
            self._s.append(ms)

    @property
    def avg_ms(self) -> float:
        with self._lock:
            return sum(self._s) / len(self._s) if self._s else 0.0

    @property
    def p95_ms(self) -> float:
        with self._lock:
            if not self._s: return 0.0
            s = sorted(self._s)
            return s[int(len(s) * 0.95)]

    def recommended_width(self) -> int:
        """Adaptive: scale down send resolution when server is slow."""
        avg = self.avg_ms
        if avg == 0 or avg < 80:   return 640
        if avg < 160:              return 480
        return 320


# ═══════════════════════════════════════════════════════════
#  STREAM WORKER  — decoupled SSE processor
# ═══════════════════════════════════════════════════════════

class StreamWorker:
    """
    Background thread that processes camera frames and broadcasts
    results to all SSE subscribers.

    Ring buffer (maxsize=1): if server is slow, the newest frame
    REPLACES the old one — frames are never queued, never stale.
    """

    def __init__(self, engine: WebMakeupEngine) -> None:
        self.engine      = engine
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._subs: list[queue.Queue] = []
        self._sub_lock   = threading.Lock()
        self._latency    = LatencyTracker()
        self._thread     = threading.Thread(target=self._run, daemon=True, name="StreamWorker")
        self._thread.start()
        log.info("StreamWorker started")

    def push(self, frame: np.ndarray, params: dict) -> None:
        """Non-blocking push — drops stale frame if queue is full."""
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait((frame, params))
        except queue.Full:
            pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=3)
        with self._sub_lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._sub_lock:
            try: self._subs.remove(q)
            except ValueError: pass

    def _broadcast(self, payload: str) -> None:
        with self._sub_lock:
            dead = []
            for q in self._subs:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass  # slow client — drop this frame, never block
                except Exception:
                    dead.append(q)
            for q in dead:
                self._subs.remove(q)

    def _run(self) -> None:
        while True:
            try:
                frame, params = self._q.get(timeout=1.0)
                t0     = time.perf_counter()
                result = self.engine.render(frame, params)
                ms     = (time.perf_counter() - t0) * 1000
                self._latency.record(ms)
                b64 = _encode_b64(result, quality=75)
                self._broadcast(json.dumps({
                    "result":     b64,
                    "ms":         round(ms, 1),
                    "avg_ms":     round(self._latency.avg_ms, 1),
                    "send_width": self._latency.recommended_width(),
                }))
            except queue.Empty:
                continue
            except Exception as e:
                log.exception(f"StreamWorker: {e}")
                time.sleep(0.05)

    @property
    def latency(self) -> LatencyTracker:
        return self._latency

    @property
    def n_subs(self) -> int:
        with self._sub_lock:
            return len(self._subs)


# ═══════════════════════════════════════════════════════════
#  CODEC HELPERS
# ═══════════════════════════════════════════════════════════

def _decode_b64(b64str: str) -> np.ndarray:
    if "," in b64str:
        b64str = b64str.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(b64str), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _encode_b64(img: np.ndarray, quality: int = 80) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


# ═══════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

try:
    ENGINE = WebMakeupEngine()
    WORKER = StreamWorker(ENGINE)
except Exception as e:
    log.critical(f"Engine init failed: {e}")
    ENGINE = None
    WORKER = None

SCREENSHOT_DIR = _ROOT / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    stats = {}
    if WORKER:
        stats = {
            "avg_ms":      round(WORKER.latency.avg_ms, 1),
            "p95_ms":      round(WORKER.latency.p95_ms, 1),
            "send_width":  WORKER.latency.recommended_width(),
            "subscribers": WORKER.n_subs,
        }
    return jsonify({"status": "ok", "engine": ENGINE is not None,
                    "time": datetime.now().isoformat(), "latency": stats})


@app.route("/api/presets")
def presets():
    return jsonify(ALL_PRESETS)


# ── SSE: frame push (fire-and-forget) ────────────────────────────────────────
@app.route("/api/stream/push", methods=["POST"])
def stream_push():
    """
    Receive a camera frame. Returns 202 immediately — never blocks.
    Processing happens in the background worker thread.
    """
    if not WORKER:
        return _error("Engine not ready", 503)
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return _error("Missing 'image'"), 400
    try:
        frame  = _decode_b64(data["image"])
        params = data.get("params", {})
        WORKER.push(frame, params)
        return jsonify({"status": "queued",
                        "send_width": WORKER.latency.recommended_width()}), 202
    except ValueError as e:
        return _error(str(e)), 400
    except Exception as e:
        log.exception("stream_push")
        return _error(str(e)), 500


# ── SSE: result stream ───────────────────────────────────────────────────────
@app.route("/api/stream")
def stream_sse():
    """
    Server-Sent Events. Browser connects once and receives results
    as they come off the worker — no polling.
    """
    if not WORKER:
        return _error("Engine not ready", 503)

    q = WORKER.subscribe()

    def generate() -> Generator[str, None, None]:
        yield ": connected\n\n"
        try:
            while True:
                try:
                    msg = q.get(timeout=15.0)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"   # keep connection alive
        except GeneratorExit:
            pass
        finally:
            WORKER.unsubscribe(q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",       # nginx: disable buffering
            "Connection":        "keep-alive",
        },
    )


# ── Single-shot sync apply (photo mode / backward compat) ────────────────────
@app.route("/api/apply", methods=["POST"])
def apply_makeup():
    if not ENGINE:
        return _error("Engine not initialised", 503)
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return _error("Missing 'image'"), 400
    try:
        frame  = _decode_b64(data["image"])
        params = data.get("params", {})
        t0     = time.perf_counter()
        result = ENGINE.render(frame, params)
        ms     = (time.perf_counter() - t0) * 1000
        return jsonify({"result": _encode_b64(result), "ms": round(ms, 1)})
    except ValueError as e:
        return _error(str(e)), 400
    except Exception as e:
        log.exception("apply_makeup")
        return _error(str(e)), 500


# ── Photo upload ──────────────────────────────────────────────────────────────
@app.route("/api/photo", methods=["POST"])
def apply_photo():
    if not ENGINE:
        return _error("Engine not initialised", 503)
    if "file" not in request.files:
        return _error("Missing 'file'"), 400
    file = request.files["file"]
    if not file.filename:
        return _error("Empty filename"), 400
    try:
        arr   = np.frombuffer(file.read(), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return _error("Could not decode image"), 400
        try:
            params = json.loads(request.form.get("params", "{}"))
        except json.JSONDecodeError:
            params = {}
        h, w = frame.shape[:2]
        if max(h, w) > 1920:
            s = 1920 / max(h, w)
            frame = cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        t0     = time.perf_counter()
        result = ENGINE.render(frame, params, static=True)
        ms     = (time.perf_counter() - t0) * 1000
        return jsonify({"result": _encode_b64(result, quality=92), "ms": round(ms, 1)})
    except Exception as e:
        log.exception("apply_photo")
        return _error(str(e)), 500


# ── Screenshot ────────────────────────────────────────────────────────────────
@app.route("/api/screenshot", methods=["POST"])
def save_screenshot():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return _error("Missing 'image'"), 400
    try:
        img  = _decode_b64(data["image"])
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SCREENSHOT_DIR / f"makeup_{ts}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return jsonify({"path": str(path), "filename": path.name})
    except Exception as e:
        log.exception("screenshot")
        return _error(str(e)), 500


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port",  type=int, default=5000)
    p.add_argument("--host",  type=str, default="0.0.0.0")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    log.info(f"Makeup AI v3 → {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug,
            use_reloader=False, threaded=True)
