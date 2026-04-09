# Makeup AI

## Overview
A deep learning computer vision project for automated makeup transfer. Evolved through 3 phases, currently implementing a BeautyGAN architecture that transfers makeup from a reference image to a source face while preserving identity.

## Web Interface
Interactive web app with two modes:
- **Live Camera**: Real-time webcam makeup with server-side processing via MediaPipe FaceLandmarker + OpenCV
- **Photo Upload**: Upload a face photo, detect landmarks, apply makeup effects, download result

The frontend sends frames/photos to the Flask backend, which uses MediaPipe FaceLandmarker (tasks API) and OpenCV for face detection and makeup rendering, then returns the processed image.

### Controls
- Lipstick, Lip Liner, Eyeshadow, Eyebrow Fill, Blush, Highlighter, Contour, Foundation
- Each with enable toggle, color swatches, and opacity slider
- Preset looks: Natural, Glam, Smoky, Fresh
- Before/After comparison toggle

## Architecture (Server-Side Processing)
- **WebMakeupEngine**: Self-contained makeup rendering engine using MediaPipe FaceLandmarker (tasks API v0.10.33+) and OpenCV
- **FaceLandmarker model**: `face_landmarker.task` (downloaded from Google's model hub)
- **Rendering pipeline**: HSV color blending with bilateral filtering for natural look
- **Routes**: `/` (UI), `/health`, `/api/apply` (webcam frames), `/api/photo` (uploads), `/api/presets`, `/api/screenshot`

## Architecture (Phase 3 - ML Pipeline)
- **Generator**: U-Net Generator (`generator.py`)
- **Discriminator**: Dual Discriminator in `beautygan.py`
- **Loss**: Custom `MakeupGANLoss` with perceptual + GAN losses (`losses.py`)
- **Training**: Mixed-precision with PyTorch AMP (`trainer.py`)
- **Inference**: Production engine with ONNX export (`inference.py`)
- **Data**: BeautyGAN dataset pipeline (`dataset.py`)
- **Utils**: Shared device/path helpers (`common.py`), histogram matching (`histogram.py`)

## Tech Stack
- **Python 3.12** — runtime
- **Flask** — web server
- **Gunicorn** — production WSGI server
- **OpenCV (opencv-python-headless)** — image processing, color blending
- **MediaPipe 0.10.33+** — face landmark detection (tasks API with FaceLandmarker)
- **NumPy** — array operations
- **PyTorch** — deep learning & GAN training (backend)
- **ONNX** — model export for production

## System Dependencies
- `xorg.libxcb`, `xorg.libX11`, `libGL`, `libz`, `glib` — required by OpenCV/MediaPipe

## Running the App
- Development: `python app.py` (port 5000)
- Production: `gunicorn --bind=0.0.0.0:5000 --reuse-port app:app`

## Key Files
| File | Purpose |
|------|---------|
| `app.py` | Flask web server + WebMakeupEngine (server-side rendering) |
| `templates/index.html` | Main UI template (inline CSS/JS) |
| `face_landmarker.task` | MediaPipe face landmark model file |
| `beautygan.py` | Unified GAN wrapper |
| `generator.py` | U-Net Generator |
| `losses.py` | GAN + perceptual loss |
| `trainer.py` | Training loop |
| `dataset.py` | Data loading & preprocessing |
| `inference.py` | Inference engine + ONNX export |
| `histogram.py` | Color/histogram matching |
| `common.py` | Shared utilities |
| `onnx_export.py` | ONNX export script |
| `test_phase3.py` | Phase 3 test suite |
| `requirements.txt` | Python dependencies |
