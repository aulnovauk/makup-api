# Makeup AI

## Overview
A deep learning computer vision project for automated makeup transfer. Features a production web UI for real-time virtual makeup try-on using server-side face landmark detection and OpenCV rendering.

## Architecture — Two-Tier System

### Tier 1: Web Application (Production-Ready)
The live web app uses a **server-side rendering pipeline**:

1. **Frontend** (`templates/index.html`) — Single-file SPA with inline CSS/JS
   - Two modes: **Live Camera** and **Photo Upload**
   - Camera frames are downscaled to 480px, JPEG-encoded at 60% quality, and sent to `/api/apply` via POST
   - Photo uploads are sent as multipart form data to `/api/photo`
   - Results are returned as base64 JPEG and overlaid on the canvas
   - Video feed renders at full framerate; makeup overlay updates asynchronously (non-blocking)
   - Features: 4 preset looks (Natural/Glam/Smoky/Fresh), Before/After split view, Screenshot/Save

2. **Backend** (`app.py`) — Flask server with `WebMakeupEngine`
   - Engine initialized once at startup using MediaPipe FaceLandmarker (tasks API v0.10.33+)
   - Model file: `face_landmarker.task` (3.7MB, downloaded from Google's model hub)
   - `RunningMode.IMAGE` — each frame processed independently (no VIDEO tracking mode)
   - Supports up to 2 faces simultaneously
   - Single-threaded (Flask dev server) — MediaPipe is NOT thread-safe

### Tier 2: ML Training Pipeline (BeautyGAN Phase 3)
Full GAN-based makeup transfer training infrastructure (not used by web UI):
- `generator.py` — U-Net Generator
- `beautygan.py` — Dual Discriminator wrapper
- `losses.py` — Custom MakeupGANLoss (perceptual + GAN losses)
- `trainer.py` — Mixed-precision training with PyTorch AMP
- `inference.py` — Production inference engine + ONNX export
- `dataset.py` — BeautyGAN dataset pipeline (supports CSV, folder, and custom formats)
- `histogram.py` — Color/histogram matching utilities
- `common.py` — Shared device/path helpers

## Makeup Rendering Pipeline (app.py `WebMakeupEngine.render()`)

### Face Detection
- MediaPipe FaceLandmarker detects 478 landmarks per face
- Landmarks accessed as `lm[idx].x / .y` (normalized 0-1 coordinates)
- `_visible()` checks landmark visibility (gracefully handles Tasks API where visibility may be None)
- `_to_pts()` converts normalized landmarks to pixel coordinates

### Rendering Order (per face)
1. **Foundation** — Bilateral filter for skin smoothing + skin-color-matched overlay (FACE_OVAL region, 36 landmarks)
2. **Contour** — Dark shadow along jawline/cheekbone (4 sub-regions, fixed BGR color 60,80,110)
3. **Eyebrow Fill** — Color fill on eyebrow landmarks (LEFT/RIGHT_EYEBROW, 10 landmarks each)
4. **Eyeshadow** — Color fill on upper eyelid (LEFT/RIGHT_EYE_SHADOW, 16 landmarks each)
5. **Highlighter** — Light shimmer on nose bridge + brow bones (fixed BGR 220,230,245)
6. **Blush** — Cheek color (LEFT/RIGHT_CHEEK, 8 landmarks each)
7. **Lipstick** — Lip color + gloss effect (LIPS_OUTER, 21 landmarks)
8. **Lip Liner** — Thin outline around lips using cv2.polylines

### Color Blending Algorithm (`_apply_region()`)
- Creates a binary mask from landmark polygon (`cv2.fillPoly`)
- Softens mask edges with bilateral filter for natural blending
- Converts frame and color to HSV, replaces Hue and Saturation (keeps original Value/brightness)
- Alpha-blends using the softened mask: `output = frame * (1 - alpha*mask) + blended * (alpha*mask)`

### Gloss Effect (`_apply_gloss()`)
- Erodes the lip mask to create a smaller highlight area
- Gaussian blur for soft highlight edges
- Blends a light color (245,245,255) at 22% opacity

## Frontend State Architecture

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

### Color Presets (PRESETS object)
- Lipstick: 6 colors (Red Rose, Pink, Coral, Berry, Nude, Plum)
- Eyeshadow: 5 colors (Smoky Brown, Rose Gold, Blue, Green, Purple)
- Eyebrow: 4 colors (Dark Brown, Taupe, Black, Auburn)
- Blush: 4 colors (Soft Rose, Peach, Berry, Bronze)
- Colors stored as BGR arrays (matching OpenCV) + hex strings for UI swatches

### Preset Looks (LOOKS object)
Each look sets all 8 layers with specific enabled/opacity/color values:
- **Natural**: Subtle lipstick, light blush, eyebrow fill, foundation
- **Glam**: Full coverage — all layers at high opacity
- **Smoky**: Dark eyeshadow + berry lips + strong contour
- **Fresh**: Coral tones, light eyeshadow, no contour

## API Routes

| Route | Method | Purpose | Input | Output |
|-------|--------|---------|-------|--------|
| `/` | GET | Serve UI | — | HTML |
| `/health` | GET | Engine status check | — | `{status, engine, python, time}` |
| `/api/apply` | POST | Process webcam frame | JSON: `{image: base64, params: state}` | `{result: base64, ms}` |
| `/api/photo` | POST | Process uploaded photo | FormData: file + params JSON | `{result: base64, ms}` |
| `/api/presets` | GET | Color presets | — | `{lipstick, eyeshadow, blush}` |
| `/api/screenshot` | POST | Save screenshot | JSON: `{image: base64}` | `{path, filename}` |

## Camera Processing Loop (Optimized)
1. `requestAnimationFrame` loop draws raw video to canvas at full framerate (smooth video)
2. Last processed makeup result is overlaid on top (asynchronous, non-blocking)
3. When not already processing: downscale frame to 480px → encode JPEG at 60% quality → POST to server
4. Server response (base64 JPEG) loaded into `lastMakeupImg` for next overlay
5. FPS counter shows makeup processing rate (smoothed with 0.85/0.15 EMA)
6. "Processing" badge shown during server round-trip

## Tech Stack
- **Python 3.12** with Flask
- **OpenCV (opencv-python-headless 4.13)** — image processing, HSV blending
- **MediaPipe 0.10.33** — FaceLandmarker (tasks API)
- **NumPy 2.4** — array operations
- **System deps**: xorg.libxcb, xorg.libX11, libGL, libz, glib

## Key Files
| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | 578 | Flask server + WebMakeupEngine |
| `templates/index.html` | 1227 | Full SPA (HTML + CSS + JS inline) |
| `face_landmarker.task` | — | MediaPipe model (3.7MB) |
| `static/makeup.js` | 363 | Legacy client-side rendering (not used by current UI) |
| `static/style.css` | 214 | Legacy styles (not used by current UI) |

## Running
- Development: `python app.py` (port 5000)
- Production: `gunicorn --bind=0.0.0.0:5000 --workers=1 --reuse-port app:app`
