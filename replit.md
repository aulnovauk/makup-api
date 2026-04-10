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

**Frontend** (`templates/index.html`, 844 lines) — Single-file SPA with inline CSS/JS
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

**Backend** (`app.py`, 927 lines) — Flask server with decoupled SSE architecture
- `WebMakeupEngine` — Thread-safe with `threading.Lock` around MediaPipe calls
- `StreamWorker` — Background daemon thread that:
  - Reads from a ring buffer (maxsize=1 queue) — newest frame replaces stale one
  - Processes frames via `engine.render()`
  - Broadcasts base64 results to all SSE subscribers
  - Tracks latency via `LatencyTracker` (20-sample rolling window, avg + p95)
- `LatencyTracker` — Recommends send resolution: 640px if avg<80ms, 480px if 80-160ms, 320px if >160ms
- Flask runs with `threaded=True` for concurrent SSE + push handling

### Gap Fixes (Production Quality)
- **Gap Fix #1** (`landmark_smoother.py`) — One-Euro temporal filter on all 478 face landmarks. Eliminates 2-5px per-frame jitter. `MultiFaceSmoother` manages per-face smoother lifecycle.
- **Gap Fix #2** (`lighting_estimator.py`) — Grey-world illuminant estimation + skin brightness analysis. Adapts makeup colors to scene lighting (warm/cool tint, dim/bright opacity scaling). EMA-smoothed across frames (alpha=0.08).

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
- MediaPipe FaceLandmarker (tasks API), `RunningMode.IMAGE`, up to 2 faces
- Model: `face_landmarker.task` (3.7MB)
- `_visible()` — Checks visibility of first 5 landmarks in region; returns True if ≥40% exceed threshold (0.3), or if no visibility data available (graceful fallback)

### Color Blending (`_region()`)
- LAB-space blending: keeps L (luminance) from original skin, shifts A/B chrominance toward makeup color
- Gaussian blur mask with configurable `blur_d` and `erode_px` for soft edges
- `_region_with_hole()` — Donut-shaped fill for lips (excludes mouth opening)

### Finish Modes (`_apply_finish()`)
- **Matte** — Reduces V variance, slight saturation boost
- **Gloss** — Eroded + blurred white highlight at 28% opacity
- **Metallic** — Directional point-source highlight (upper-right of centroid)
- **Shimmer** — Animated sparkle points (150 random, seed cycles 8x/sec)

### Rendering Order (per face)
1. **Foundation** — Bilateral filter smoothing + skin-color tint (erode=3)
2. **Contour** — 4 jaw/cheek sub-regions, cool taupe BGR (55,75,105)
3. **Blush** — LEFT/RIGHT_CHEEK (14 landmarks each), blur_d=19, erode_px=2
4. **Eyeshadow** — Standard (16 landmarks) or deep (31 landmarks) based on opacity>0.40
5. **Eyebrow Fill** — 10 landmarks each, tight blur_d=5, erode_px=1
6. **Highlighter** — Nose bridge + cheekbone highlights + Cupid's bow, warm pearl white
7. **Lipstick** — LIPS_OUTER with inner mouth hole exclusion, finish modes, auto-matte for dark colors (LAB L<40)
8. **Lip Liner** — AA polylines with Gaussian blur, drawn on top

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
- **OpenCV (opencv-python-headless)** — image processing, LAB-space blending
- **MediaPipe** — FaceLandmarker (tasks API)
- **NumPy** — array operations
- **System deps**: xorg.libxcb, xorg.libX11, libGL, libz, glib

## Key Files
| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | 927 | Flask server + WebMakeupEngine + StreamWorker + SSE |
| `templates/index.html` | 844 | Full SPA (HTML + CSS + JS inline) |
| `face_landmarker.task` | — | MediaPipe model (3.7MB) |
| `landmark_smoother.py` | 236 | One-Euro temporal landmark smoothing |
| `lighting_estimator.py` | 246 | Scene lighting estimation + color adaptation |
| `beautygan.py` | — | GAN discriminator (Tier 2) |
| `trainer.py` | — | GAN training loop (Tier 2) |

## Running
- Development: `python app.py` (port 5000, threaded=True)
- Production: `gunicorn --bind=0.0.0.0:5000 --workers=1 --threads=4 app:app`
