# Makeup AI

## Overview
A deep learning computer vision project for automated makeup transfer. Features a production web UI for real-time virtual makeup try-on using server-side face landmark detection and OpenCV rendering.

## Architecture v3 — Decoupled SSE Pipeline

### The Core Problem v2 Had (and v3 Fix)
v2 was synchronous: browser sent a frame, **waited** for server to process (150-400ms), then sent the next. This gated the entire live feed on server latency (~3fps at 720p).

v3 uses a **fully decoupled pipeline** with three independent loops:

```
┌─ rAF render loop (60fps) ──── always draws lastMakeupImg ──────┐
│                                                                  │
│  push loop (setInterval 100ms) ── fire-and-forget POST ──▶ srv │
│                                                                  │
│  SSE listener ◀────────── result whenever ready ────────────────┘
└──────────────────────────────────────────────────────────────────┘
```

### Tier 1: Web Application (Production)

**Frontend** (`templates/index.html`, 845 lines) — Single-file SPA with inline CSS/JS
- **Three independent loops** when camera is active:
  1. `renderLoop()` — rAF at 60fps, draws raw mirrored video + overlays `lastMakeupImg`
  2. `startPushLoop()` — `setInterval(100ms)`, fire-and-forget POST to `/api/stream/push`
  3. `startSSE()` — `EventSource('/api/stream')`, receives processed frames from server
- **Adaptive resolution**: `sendWidth` starts at 480px, server adjusts up/down (640/480/320) based on measured latency
- **Tab visibility handling**: push loop pauses when tab is hidden, resumes when visible
- **Before/After split**: clip-based canvas split with labeled halves
- **Photo mode**: uses debounced `reapplyPhoto()` (200ms) with `lastRawPhotoDataUrl` for parameter changes
- **Latency badge**: color-coded (green < 100ms, amber < 200ms, red > 200ms)
- **AbortSignal.timeout(3000)**: push requests never hang longer than 3 seconds

**Backend** (`app.py`, 630 lines) — Flask server with decoupled SSE architecture
- `WebMakeupEngine` — Thread-safe with `threading.Lock` around MediaPipe calls
- `StreamWorker` — Background daemon thread that:
  - Reads from a ring buffer (maxsize=1 queue) — newest frame replaces stale one
  - Processes frames via `engine.render()`
  - Broadcasts base64 results to all SSE subscribers
  - Tracks latency via `LatencyTracker` (20-sample rolling window, avg + p95)
- `LatencyTracker` — Recommends send resolution: 640px if avg<80ms, 480px if 80-160ms, 320px if >160ms
- Flask runs with `threaded=True` for concurrent SSE + push handling

### Tier 2: ML Training Pipeline (BeautyGAN Phase 3)
Full GAN-based makeup transfer training infrastructure (not used by web UI):
- `generator.py` — U-Net Generator
- `beautygan.py` — Dual Discriminator wrapper
- `losses.py` — Custom MakeupGANLoss (perceptual + GAN losses)
- `trainer.py` — Mixed-precision training with PyTorch AMP
- `inference.py` — Production inference engine + ONNX export
- `dataset.py` — BeautyGAN dataset pipeline
- `histogram.py` — Color/histogram matching utilities
- `common.py` — Shared device/path helpers

## Makeup Rendering Pipeline (WebMakeupEngine)

### Face Detection
- MediaPipe FaceLandmarker (tasks API v0.10.33+), `RunningMode.IMAGE`, up to 2 faces
- Model: `face_landmarker.task` (3.7MB)
- `_visible()` — Checks visibility of first 5 landmarks in region; returns True if ≥40% exceed threshold (0.3), or if no visibility data available (graceful fallback)

### Rendering Order (per face)
1. **Foundation** — Bilateral filter smoothing (d=11, sigma=65) with eroded+blurred mask (0.55 opacity), then skin-color overlay at 45% of user opacity
2. **Contour** — 4 jaw/cheek sub-regions, fixed BGR (45,65,95) at 70% of user opacity, GaussianBlur mask (d=15)
3. **Eyebrow Fill** — LEFT/RIGHT_EYEBROW (10 landmarks each), small blur (d=5) for sharp edges
4. **Eyeshadow** — LEFT/RIGHT_EYE_SHADOW (16 landmarks each), primary color + screen-blend shimmer layer (35% of alpha)
5. **Highlighter** — Nose bridge (60% alpha) + cheekbone highlights (50% alpha) using custom 8-landmark regions, BGR (230,235,250)
6. **Blush** — LEFT/RIGHT_CHEEK (8 landmarks each), GaussianBlur mask (d=13)
7. **Lipstick** — LIPS_OUTER (21 landmarks), small blur (d=5) for crisp edges, gloss only when opacity < 75%
8. **Lip Liner** — `cv2.polylines` with width=2, anti-aliased, GaussianBlur(3,3) for soft edges

### Color Blending (`_region()`)
- GaussianBlur mask instead of bilateral filter (faster, less artifacts)
- HSV space blending: interpolates Hue and Saturation (sat scaled to 80%), slightly shifts Value based on color brightness
- Formula: `H_out = H_face*(1-a*m) + H_color*(a*m)`, `S_out = S_face*(1-a*m) + S_color*0.8*(a*m)`, `V_out = V_face + (V_color-128)*0.08*a*m`

### Eyeshadow Screen Blend (new in v3)
After the base color region, applies a screen blend: `screen = 255 - (255-frame)*(255-color)/255`, mixed at 35% of eyeshadow alpha for shimmer effect

## API Routes

| Route | Method | Response | Purpose |
|-------|--------|----------|---------|
| `/` | GET | HTML | Serve UI |
| `/health` | GET | JSON | Status + latency stats (avg_ms, p95_ms, send_width, subscribers) |
| `/api/stream/push` | POST | 202 JSON | Push camera frame (fire-and-forget, non-blocking) |
| `/api/stream` | GET | SSE | Server-Sent Events stream of processed results |
| `/api/apply` | POST | JSON | Single-shot sync processing (backward compat) |
| `/api/photo` | POST | JSON | Photo file upload + process |
| `/api/presets` | GET | JSON | Color presets |
| `/api/screenshot` | POST | JSON | Save screenshot server-side + browser download |

## Frontend State

### State Object
```javascript
state = {
  lipstick:    { enabled, opacity, color:[B,G,R] },
  lip_liner:   { enabled, opacity, color:[B,G,R] },
  eyeshadow:   { enabled, opacity, color:[B,G,R] },
  eyebrow:     { enabled, opacity, color:[B,G,R] },
  blush:       { enabled, opacity, color:[B,G,R] },
  highlighter: { enabled, opacity },
  contour:     { enabled, opacity },
  foundation:  { enabled, opacity, smooth:bool },
}
```

### Color Presets
- Lipstick: 6 colors, Eyeshadow: 5 colors, Eyebrow: 4 colors, Blush: 4 colors
- BGR arrays (OpenCV) + hex strings (UI swatches)

### Preset Looks
- **Natural**: Subtle lipstick, light blush, eyebrow fill, foundation
- **Glam**: Full coverage, all layers high opacity
- **Smoky**: Dark eyeshadow + berry lips + strong contour
- **Fresh**: Coral tones, light eyeshadow, no contour

## Tech Stack
- **Python 3.12** with Flask (threaded mode)
- **OpenCV (opencv-python-headless 4.13)** — image processing, HSV blending
- **MediaPipe 0.10.33** — FaceLandmarker (tasks API)
- **NumPy 2.4** — array operations
- **System deps**: xorg.libxcb, xorg.libX11, libGL, libz, glib

## Key Files
| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | 630 | Flask server + WebMakeupEngine + StreamWorker + SSE |
| `templates/index.html` | 845 | Full SPA (HTML + CSS + JS inline) |
| `face_landmarker.task` | — | MediaPipe model (3.7MB) |
| `static/makeup.js` | 363 | Legacy (not used) |
| `static/style.css` | 214 | Legacy (not used) |

## Running
- Development: `python app.py` (port 5000, threaded=True)
- Production: `gunicorn --bind=0.0.0.0:5000 --workers=1 --threads=4 app:app`
