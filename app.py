"""
app.py — Makeup AI Web Application  v2.0
══════════════════════════════════════════
Production Flask backend connecting Phase 2 MakeupEngine to the web UI.

Routes:
  GET  /                  → Serve the main UI
  GET  /health            → Health check + system info
  POST /api/apply         → Apply makeup to a base64 image frame
  POST /api/photo         → Apply makeup to an uploaded photo file
  GET  /api/presets       → Return all available color presets
  POST /api/screenshot    → Save a screenshot server-side

Architecture:
  - MakeupEngine (Phase 2) is initialised once at startup and reused
  - Each request gets a fresh frame copy — no shared state between requests
  - Base64 encoding used for webcam frames (browser → server → browser)
  - Multipart form data used for photo uploads
  - All errors return JSON with "error" key and appropriate HTTP status

Usage:
  pip install flask
  python app.py
  Open: http://localhost:5000
"""

from __future__ import annotations

import base64
import io
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

# ── Ensure Phase 2 engine is importable ──────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_ROOT / "webapp.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("MakeupAI.Web")


# ═══════════════════════════════════════════════════════════
#  MAKEUP STATE  (mirroring Phase 2 MakeupState)
# ═══════════════════════════════════════════════════════════

# Color palettes — BGR format (OpenCV)
LIPSTICK_COLORS = {
    "red_rose":     (30,  20,  200),
    "pink_blush":   (150, 100, 220),
    "coral_sunset": (80,  100, 240),
    "berry_wine":   (60,  20,  130),
    "nude_beige":   (130, 150, 190),
    "deep_plum":    (80,  30,  100),
}
EYESHADOW_COLORS = {
    "smoky_brown":  (40,  60,  100),
    "rose_gold":    (100, 120, 200),
    "ocean_blue":   (180, 100, 50 ),
    "forest_green": (60,  130, 60 ),
    "purple_haze":  (150, 60,  130),
}
BLUSH_COLORS = {
    "soft_rose":    (140, 130, 230),
    "peach":        (110, 160, 240),
    "berry":        (100,  80, 180),
    "bronze":       (80,  110, 170),
}

ALL_PRESETS = {
    "lipstick":  {k: {"bgr": list(v)} for k, v in LIPSTICK_COLORS.items()},
    "eyeshadow": {k: {"bgr": list(v)} for k, v in EYESHADOW_COLORS.items()},
    "blush":     {k: {"bgr": list(v)} for k, v in BLUSH_COLORS.items()},
}

# Landmark indices — Phase 2 canonical
LIPS_OUTER       = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95]
LEFT_EYE_SHADOW  = [226,247,30,29,27,28,56,190,243,112,26,22,23,24,110,25]
RIGHT_EYE_SHADOW = [446,467,260,259,257,258,286,414,463,341,256,252,253,254,339,255]
LEFT_EYEBROW     = [70,63,105,66,107,55,65,52,53,46]
RIGHT_EYEBROW    = [300,293,334,296,336,285,295,282,283,276]
LEFT_CHEEK       = [116,123,147,213,192,214,210,211]
RIGHT_CHEEK      = [345,352,376,433,416,434,430,431]
NOSE_BRIDGE      = [6,197,195,5,4]
LEFT_BROW_BONE   = [70,63,105,66,107,46,53,52,65,55]
RIGHT_BROW_BONE  = [300,293,334,296,336,276,283,282,295,285]
FACE_OVAL        = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
SKIN_SAMPLE_L    = [117,118,119,120,121]
SKIN_SAMPLE_R    = [346,347,348,349,350]


# ═══════════════════════════════════════════════════════════
#  MAKEUP ENGINE  (inline Phase 2 renderer — no import deps)
# ═══════════════════════════════════════════════════════════

class WebMakeupEngine:
    """
    Self-contained makeup rendering engine for the web app.
    Initialised once at startup, reused across all requests.
    Thread safety: MediaPipe FaceLandmarker is NOT thread-safe.
    Flask dev server is single-threaded by default (safe).
    For production (gunicorn): use --workers 1 or add a lock.
    """

    def __init__(self) -> None:
        try:
            import mediapipe as mp
            self._mp = mp
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            RunningMode = mp.tasks.vision.RunningMode

            model_path = str(_ROOT / "face_landmarker.task")

            self._landmarker = FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=RunningMode.IMAGE,
                    num_faces=2,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
            self._mask_buf    = None
            self._frame_shape = None
            log.info("WebMakeupEngine initialised ✓")
        except Exception as e:
            log.error(f"MediaPipe init failed: {e}")
            raise

    def __del__(self) -> None:
        try:
            if hasattr(self, "_landmarker"):
                self._landmarker.close()
        except Exception:
            pass

    def _ensure_buf(self, frame: np.ndarray) -> None:
        if self._frame_shape != frame.shape:
            self._mask_buf    = np.zeros(frame.shape[:2], dtype=np.uint8)
            self._frame_shape = frame.shape

    @staticmethod
    def _to_pts(lm: list, indices: list, w: int, h: int) -> np.ndarray:
        pts = np.empty((len(indices), 2), dtype=np.int32)
        for i, idx in enumerate(indices):
            pts[i, 0] = int(lm[idx].x * w)
            pts[i, 1] = int(lm[idx].y * h)
        return pts

    @staticmethod
    def _visible(lm: list, indices: list, thresh: float = 0.45) -> bool:
        for i in indices[:3]:
            vis = getattr(lm[i], 'visibility', None)
            if vis is not None and vis < thresh:
                return False
        return True

    def _apply_region(
        self,
        frame:  np.ndarray,
        pts:    np.ndarray,
        color:  tuple,
        alpha:  float,
        blur_d: int = 9,
    ) -> np.ndarray:
        if len(pts) < 3:
            return frame
        self._mask_buf[:] = 0
        cv2.fillPoly(self._mask_buf, [pts], 255)
        soft = cv2.bilateralFilter(self._mask_buf, d=blur_d,
                                   sigmaColor=75, sigmaSpace=75)
        color_layer = np.full_like(frame, color, dtype=np.uint8)
        fh = cv2.cvtColor(frame,       cv2.COLOR_BGR2HSV).astype(np.float32)
        ch = cv2.cvtColor(color_layer, cv2.COLOR_BGR2HSV).astype(np.float32)
        bh = fh.copy()
        bh[..., 0] = ch[..., 0]
        bh[..., 1] = ch[..., 1] * 0.85
        blended = cv2.cvtColor(
            np.clip(bh, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR
        )
        m  = soft.astype(np.float32) / 255.0
        m3 = cv2.merge([m, m, m])
        out = (frame.astype(np.float32)   * (1 - alpha * m3) +
               blended.astype(np.float32) * (alpha * m3))
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_gloss(self, frame: np.ndarray, pts: np.ndarray) -> np.ndarray:
        if len(pts) < 3:
            return frame
        self._mask_buf[:] = 0
        cv2.fillPoly(self._mask_buf, [pts], 255)
        hi  = cv2.erode(self._mask_buf, np.ones((5,5), np.uint8), iterations=4)
        hi  = cv2.GaussianBlur(hi, (11, 11), 6)
        hl  = np.full_like(frame, (245, 245, 255), dtype=np.uint8)
        a   = hi.astype(np.float32) / 255.0 * 0.22
        a3  = cv2.merge([a, a, a])
        out = (frame.astype(np.float32) * (1 - a3) +
               hl.astype(np.float32)    * a3)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _sample_skin(self, frame: np.ndarray, lm: list,
                     w: int, h: int) -> tuple:
        pixels = []
        for idx in SKIN_SAMPLE_L + SKIN_SAMPLE_R:
            x = int(np.clip(lm[idx].x * w, 0, w - 1))
            y = int(np.clip(lm[idx].y * h, 0, h - 1))
            pixels.append(frame[y, x].tolist())
        arr     = np.array(pixels, dtype=np.float32)
        b, g, r = np.median(arr, axis=0)
        return (int(np.clip(b * 1.02, 0, 255)),
                int(np.clip(g * 1.02, 0, 255)),
                int(np.clip(r * 1.02, 0, 255)))

    def render(self, frame: np.ndarray, params: dict,
               static: bool = False) -> np.ndarray:
        """
        Apply makeup to frame based on params dict from the frontend.
        params keys mirror the frontend state object.
        """
        self._ensure_buf(frame)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        results = self._landmarker.detect(mp_image)

        output = frame.copy()
        if not results.face_landmarks:
            return output

        for lm in results.face_landmarks:

            # ── Foundation + skin smooth ─────────────────
            if params.get("foundation", {}).get("enabled"):
                fd    = params["foundation"]
                alpha = float(fd.get("opacity", 0.15))
                if self._visible(lm, FACE_OVAL):
                    face_pts   = self._to_pts(lm, FACE_OVAL, w, h)
                    skin_color = self._sample_skin(frame, lm, w, h)
                    if fd.get("smooth"):
                        smooth_frame = cv2.bilateralFilter(output, 9, 55, 55)
                        mask_f = np.zeros(output.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask_f, [face_pts], 255)
                        mf = mask_f.astype(np.float32) / 255.0 * 0.65
                        mf3 = cv2.merge([mf, mf, mf])
                        output = np.clip(
                            output.astype(np.float32) * (1 - mf3) +
                            smooth_frame.astype(np.float32) * mf3,
                            0, 255
                        ).astype(np.uint8)
                    output = self._apply_region(
                        output, face_pts, skin_color, alpha * 0.5, blur_d=15
                    )

            # ── Contour ───────────────────────────────────
            if params.get("contour", {}).get("enabled"):
                alpha = float(params["contour"].get("opacity", 0.25))
                CONTOUR_COLOR = (60, 80, 110)
                for region in ([234,227,116,123,147,187,207,206,205],
                               [454,447,345,352,376,411,427,426,425],
                               [116,123,147,187,207,213,192],
                               [345,352,376,411,427,433,416]):
                    if self._visible(lm, region):
                        pts    = self._to_pts(lm, region, w, h)
                        output = self._apply_region(output, pts,
                                                    CONTOUR_COLOR, alpha, 13)

            # ── Eyebrow ───────────────────────────────────
            if params.get("eyebrow", {}).get("enabled"):
                ey    = params["eyebrow"]
                color = tuple(int(c) for c in ey.get("color", [30,45,80]))
                alpha = float(ey.get("opacity", 0.45))
                for brow in (LEFT_EYEBROW, RIGHT_EYEBROW):
                    if self._visible(lm, brow):
                        pts    = self._to_pts(lm, brow, w, h)
                        output = self._apply_region(output, pts, color, alpha, 5)

            # ── Eyeshadow ─────────────────────────────────
            if params.get("eyeshadow", {}).get("enabled"):
                es    = params["eyeshadow"]
                color = tuple(int(c) for c in es.get("color", [40,60,100]))
                alpha = float(es.get("opacity", 0.30))
                for eye_idx in (LEFT_EYE_SHADOW, RIGHT_EYE_SHADOW):
                    if self._visible(lm, eye_idx):
                        pts    = self._to_pts(lm, eye_idx, w, h)
                        output = self._apply_region(output, pts, color, alpha)

            # ── Highlighter ───────────────────────────────
            if params.get("highlighter", {}).get("enabled"):
                alpha = float(params["highlighter"].get("opacity", 0.30))
                HL    = (220, 230, 245)
                if self._visible(lm, NOSE_BRIDGE):
                    output = self._apply_region(
                        output, self._to_pts(lm, NOSE_BRIDGE, w, h), HL, alpha, 11
                    )
                for bb in (LEFT_BROW_BONE, RIGHT_BROW_BONE):
                    if self._visible(lm, bb):
                        output = self._apply_region(
                            output, self._to_pts(lm, bb, w, h), HL, alpha * 0.7, 11
                        )

            # ── Blush ─────────────────────────────────────
            if params.get("blush", {}).get("enabled"):
                bl    = params["blush"]
                color = tuple(int(c) for c in bl.get("color", [140,130,230]))
                alpha = float(bl.get("opacity", 0.20))
                for cheek in (LEFT_CHEEK, RIGHT_CHEEK):
                    if self._visible(lm, cheek):
                        pts    = self._to_pts(lm, cheek, w, h)
                        output = self._apply_region(output, pts, color, alpha, 13)

            # ── Lipstick ──────────────────────────────────
            if params.get("lipstick", {}).get("enabled"):
                ls    = params["lipstick"]
                color = tuple(int(c) for c in ls.get("color", [30,20,200]))
                alpha = float(ls.get("opacity", 0.45))
                if self._visible(lm, LIPS_OUTER):
                    lip_pts = self._to_pts(lm, LIPS_OUTER, w, h)
                    output  = self._apply_region(output, lip_pts, color, alpha)
                    output  = self._apply_gloss(output, lip_pts)

            # ── Lip liner ─────────────────────────────────
            if params.get("lip_liner", {}).get("enabled"):
                ll    = params["lip_liner"]
                color = tuple(int(c) for c in ll.get("color", [30,20,200]))
                alpha = float(ll.get("opacity", 0.60))
                if self._visible(lm, LIPS_OUTER):
                    pts   = self._to_pts(lm, LIPS_OUTER, w, h)
                    liner = output.copy()
                    cv2.polylines(liner, [pts], True, color, 1, cv2.LINE_AA)
                    mask  = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.polylines(mask, [pts], True, 255, 3)
                    mask  = cv2.GaussianBlur(mask, (3, 3), 1)
                    m     = mask.astype(np.float32) / 255.0 * alpha
                    m3    = cv2.merge([m, m, m])
                    output = np.clip(
                        output.astype(np.float32) * (1 - m3) +
                        liner.astype(np.float32)  * m3, 0, 255
                    ).astype(np.uint8)

        return output


# ═══════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload

# Initialise engine once
try:
    ENGINE = WebMakeupEngine()
except Exception as e:
    log.critical(f"Engine init failed: {e}")
    ENGINE = None

SCREENSHOT_DIR = _ROOT / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────

def _decode_b64_image(b64str: str) -> np.ndarray:
    """Decode base64 image string to BGR numpy array."""
    # Strip data URL prefix if present
    if "," in b64str:
        b64str = b64str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64str)
    arr       = np.frombuffer(img_bytes, dtype=np.uint8)
    img       = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image data")
    return img


def _encode_b64_image(img_bgr: np.ndarray, quality: int = 85) -> str:
    """Encode BGR numpy array to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _error(msg: str, code: int = 400) -> tuple[Response, int]:
    return jsonify({"error": msg}), code


# ─────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────

@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/health")
def health() -> Response:
    return jsonify({
        "status":  "ok",
        "engine":  ENGINE is not None,
        "python":  sys.version,
        "time":    datetime.now().isoformat(),
    })


@app.route("/api/presets")
def presets() -> Response:
    """Return all available color presets to the frontend."""
    return jsonify(ALL_PRESETS)


@app.route("/api/apply", methods=["POST"])
def apply_makeup() -> tuple[Response, int]:
    """
    Apply makeup to a webcam frame.

    Request JSON:
        image:  base64 encoded JPEG frame from webcam
        params: makeup parameter object (see frontend state)

    Response JSON:
        result: base64 encoded JPEG with makeup applied
        ms:     processing time in milliseconds
    """
    if ENGINE is None:
        return _error("Makeup engine not initialised", 503)

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return _error("Missing 'image' field")

    try:
        frame  = _decode_b64_image(data["image"])
        params = data.get("params", {})

        t0     = time.perf_counter()
        result = ENGINE.render(frame, params, static=False)
        ms     = (time.perf_counter() - t0) * 1000

        return jsonify({
            "result": _encode_b64_image(result),
            "ms":     round(ms, 1),
        })

    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.exception("apply_makeup error")
        return _error(f"Processing error: {e}", 500)


@app.route("/api/photo", methods=["POST"])
def apply_photo() -> tuple[Response, int]:
    """
    Apply makeup to an uploaded photo file.

    Request: multipart/form-data
        file:   image file (jpg/png)
        params: JSON string of makeup parameters

    Response JSON:
        result: base64 encoded JPEG with makeup applied
        ms:     processing time in milliseconds
    """
    if ENGINE is None:
        return _error("Makeup engine not initialised", 503)

    if "file" not in request.files:
        return _error("Missing 'file' field")

    file = request.files["file"]
    if file.filename == "":
        return _error("Empty filename")

    try:
        img_bytes = file.read()
        arr       = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return _error("Could not decode uploaded image")

        params_raw = request.form.get("params", "{}")
        try:
            params = json.loads(params_raw)
        except json.JSONDecodeError:
            params = {}

        # Resize very large photos for performance
        h, w = frame.shape[:2]
        if max(h, w) > 1920:
            scale = 1920 / max(h, w)
            frame = cv2.resize(frame,
                               (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        t0     = time.perf_counter()
        result = ENGINE.render(frame, params, static=True)
        ms     = (time.perf_counter() - t0) * 1000

        return jsonify({
            "result": _encode_b64_image(result, quality=92),
            "ms":     round(ms, 1),
        })

    except Exception as e:
        log.exception("apply_photo error")
        return _error(f"Processing error: {e}", 500)


@app.route("/api/screenshot", methods=["POST"])
def save_screenshot() -> tuple[Response, int]:
    """
    Save a screenshot server-side.

    Request JSON:
        image: base64 encoded image to save

    Response JSON:
        path: saved file path
        filename: just the filename
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return _error("Missing 'image' field")

    try:
        img  = _decode_b64_image(data["image"])
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SCREENSHOT_DIR / f"makeup_{ts}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        log.info(f"Screenshot saved → {path}")
        return jsonify({"path": str(path), "filename": path.name})
    except Exception as e:
        log.exception("screenshot error")
        return _error(f"Save error: {e}", 500)


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Makeup AI Web App")
    parser.add_argument("--port",  type=int,  default=5000)
    parser.add_argument("--host",  type=str,  default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    log.info(f"Starting Makeup AI Web App on {args.host}:{args.port}")
    # NOTE: use_reloader=False prevents MediaPipe from double-initialising
    app.run(host=args.host, port=args.port,
            debug=args.debug, use_reloader=False)
