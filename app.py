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

# ── Logging ───────────────────────────────────────────────
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

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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

LIPS_OUTER       = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95]
LEFT_EYE_SHADOW  = [226,247,30,29,27,28,56,190,243,112,26,22,23,24,110,25]
RIGHT_EYE_SHADOW = [446,467,260,259,257,258,286,414,463,341,256,252,253,254,339,255]
LEFT_EYEBROW     = [70,63,105,66,107,55,65,52,53,46]
RIGHT_EYEBROW    = [300,293,334,296,336,285,295,282,283,276]
LEFT_CHEEK       = [116,123,147,213,192,214,210,211]
RIGHT_CHEEK      = [345,352,376,433,416,434,430,431]
NOSE_BRIDGE      = [6,197,195,5,4]
FACE_OVAL        = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
SKIN_SAMPLE_L    = [117,118,119,120,121]
SKIN_SAMPLE_R    = [346,347,348,349,350]


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
            pts[i, 0] = int(lm[idx].x * w)
            pts[i, 1] = int(lm[idx].y * h)
        return pts

    @staticmethod
    def _visible(lm, indices, thresh=0.3):
        valid = [lm[i].visibility for i in indices[:5]
                 if lm[i].visibility is not None and lm[i].visibility > 0]
        return True if not valid else (sum(v > thresh for v in valid) / len(valid)) >= 0.4

    def _region(self, frame, pts, color, alpha, blur_d=9):
        if len(pts) < 3:
            return frame
        self._mask_buf[:] = 0
        cv2.fillPoly(self._mask_buf, [pts], 255)
        k    = max(3, (blur_d // 2) * 2 + 1)
        soft = np.clip(cv2.GaussianBlur(self._mask_buf.astype(np.float32), (k, k), blur_d * 0.6) / 255, 0, 1)
        fh   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        ch   = cv2.cvtColor(np.full_like(frame, color, np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        out  = fh.copy()
        out[..., 0] = fh[..., 0] * (1 - alpha * soft) + ch[..., 0] * (alpha * soft)
        out[..., 1] = fh[..., 1] * (1 - alpha * soft) + ch[..., 1] * 0.8 * (alpha * soft)
        out[..., 2] = np.clip(fh[..., 2] + (ch[..., 2] - 128) * 0.08 * alpha * soft, 0, 255)
        return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _gloss(self, frame, pts):
        if len(pts) < 3:
            return frame
        self._mask_buf[:] = 0
        cv2.fillPoly(self._mask_buf, [pts], 255)
        hi = cv2.GaussianBlur(cv2.erode(self._mask_buf, np.ones((5,5),np.uint8), iterations=4), (11,11), 6)
        hl = np.full_like(frame, (245, 245, 255), np.uint8)
        a  = hi.astype(np.float32) / 255 * 0.22
        a3 = cv2.merge([a, a, a])
        return np.clip(frame.astype(np.float32)*(1-a3) + hl.astype(np.float32)*a3, 0, 255).astype(np.uint8)

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
        self._ensure_buf(frame)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res   = self._landmarker.detect(img)
        out   = frame.copy()

        if not res.face_landmarks:
            return out

        for lm in res.face_landmarks:
            p = params   # shorthand

            # Foundation
            if p.get("foundation",{}).get("enabled"):
                fd = p["foundation"];  a = float(fd.get("opacity", 0.15))
                if self._visible(lm, FACE_OVAL):
                    fpts = self._to_pts(lm, FACE_OVAL, w, h)
                    skin = self._skin(frame, lm, w, h)
                    if fd.get("smooth"):
                        sf   = cv2.bilateralFilter(out, 11, 65, 65)
                        mf   = np.zeros(out.shape[:2], np.uint8)
                        cv2.fillPoly(mf, [fpts], 255)
                        mf   = cv2.erode(mf, np.ones((5,5),np.uint8), iterations=2)
                        mf   = cv2.GaussianBlur(mf.astype(np.float32),(25,25),12)/255*0.55
                        mf3  = cv2.merge([mf,mf,mf])
                        out  = np.clip(out.astype(np.float32)*(1-mf3)+sf.astype(np.float32)*mf3,0,255).astype(np.uint8)
                    out = self._region(out, fpts, skin, a*0.45, 19)

            # Contour
            if p.get("contour",{}).get("enabled"):
                a = float(p["contour"].get("opacity", 0.25))
                for r in ([234,93,132,58,172,136,150,149,176,148,152],
                          [454,323,361,288,397,365,379,378,400,377,152],
                          [116,123,147,213,192,214,210],[345,352,376,433,416,434,430]):
                    if self._visible(lm, r):
                        out = self._region(out, self._to_pts(lm,r,w,h), (45,65,95), a*0.7, 15)

            # Eyebrow
            if p.get("eyebrow",{}).get("enabled"):
                ey = p["eyebrow"];  color = tuple(int(c) for c in ey.get("color",[30,45,80]));  a = float(ey.get("opacity",0.45))
                for brow in (LEFT_EYEBROW, RIGHT_EYEBROW):
                    if self._visible(lm, brow):
                        out = self._region(out, self._to_pts(lm,brow,w,h), color, a, 5)

            # Eyeshadow
            if p.get("eyeshadow",{}).get("enabled"):
                es = p["eyeshadow"];  color = tuple(int(c) for c in es.get("color",[40,60,100]));  a = float(es.get("opacity",0.30))
                for ei in (LEFT_EYE_SHADOW, RIGHT_EYE_SHADOW):
                    if self._visible(lm, ei):
                        pts = self._to_pts(lm,ei,w,h)
                        out = self._region(out, pts, color, a, 11)
                        self._mask_buf[:] = 0
                        cv2.fillPoly(self._mask_buf, [pts], 255)
                        sm  = cv2.GaussianBlur(self._mask_buf.astype(np.float32),(21,21),10)/255*a*0.35
                        cl  = np.full_like(out, color, np.float32);  ff = out.astype(np.float32)
                        sc  = 255-(255-ff)*(255-cl)/255
                        out = np.clip(ff*(1-cv2.merge([sm,sm,sm]))+sc*cv2.merge([sm,sm,sm]),0,255).astype(np.uint8)

            # Highlighter
            if p.get("highlighter",{}).get("enabled"):
                a = float(p["highlighter"].get("opacity",0.30));  HL=(230,235,250)
                if self._visible(lm, NOSE_BRIDGE):
                    out = self._region(out, self._to_pts(lm,NOSE_BRIDGE,w,h), HL, a*0.6, 11)
                for cb in ([117,118,119,100,126,209,49,131],[346,347,348,329,355,429,279,360]):
                    if self._visible(lm, cb):
                        out = self._region(out, self._to_pts(lm,cb,w,h), HL, a*0.5, 13)

            # Blush
            if p.get("blush",{}).get("enabled"):
                bl = p["blush"];  color = tuple(int(c) for c in bl.get("color",[140,130,230]));  a = float(bl.get("opacity",0.20))
                for ck in (LEFT_CHEEK, RIGHT_CHEEK):
                    if self._visible(lm, ck):
                        out = self._region(out, self._to_pts(lm,ck,w,h), color, a, 13)

            # Lipstick
            if p.get("lipstick",{}).get("enabled"):
                ls = p["lipstick"];  color = tuple(int(c) for c in ls.get("color",[30,20,200]));  a = float(ls.get("opacity",0.45))
                if self._visible(lm, LIPS_OUTER):
                    lpts = self._to_pts(lm,LIPS_OUTER,w,h)
                    out  = self._region(out, lpts, color, a, 5)
                    if a < 0.75:
                        out = self._gloss(out, lpts)

            # Lip liner
            if p.get("lip_liner",{}).get("enabled"):
                ll = p["lip_liner"];  color = tuple(int(c) for c in ll.get("color",[30,20,200]));  a = float(ll.get("opacity",0.60))
                if self._visible(lm, LIPS_OUTER):
                    pts = self._to_pts(lm,LIPS_OUTER,w,h)
                    lm_ = np.zeros(out.shape[:2], np.float32)
                    cv2.polylines(lm_, [pts], True, 1.0, 2, cv2.LINE_AA)
                    lm_ = cv2.GaussianBlur(lm_,(3,3),1)
                    lc  = np.full_like(out, color, np.uint8)
                    m3  = cv2.merge([lm_*a, lm_*a, lm_*a])
                    out = np.clip(out.astype(np.float32)*(1-m3)+lc.astype(np.float32)*m3,0,255).astype(np.uint8)

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

@app.after_request
def add_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

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


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


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
