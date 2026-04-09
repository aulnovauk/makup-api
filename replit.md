# Makeup AI

## Overview
A deep learning computer vision project for automated makeup transfer. Evolved through 3 phases, currently implementing a BeautyGAN architecture that transfers makeup from a reference image to a source face while preserving identity.

## Web Interface
Interactive web app with two modes:
- **Live Camera**: Real-time webcam makeup using browser MediaPipe Face Mesh JS + Canvas API
- **Photo Upload**: Upload a face photo, detect landmarks, apply makeup effects, download result

All face detection and makeup rendering happens client-side in the browser.

### Controls
- Lipstick, Eyeshadow, Blush: each with enable toggle, color swatches, and opacity slider
- Landmark indices match the Python codebase (`histogram.py`)

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
- **MediaPipe Face Mesh JS** — browser-side face landmark detection (468 points)
- **Canvas API** — client-side makeup rendering
- **PyTorch** — deep learning & GAN training (backend)
- **OpenCV** — image processing (backend)
- **MediaPipe Python** — face landmarks (backend)
- **ONNX** — model export for production

## Running the App
- Development: `python app.py` (port 5000)
- Production: `gunicorn --bind=0.0.0.0:5000 --reuse-port app:app`

## Key Files
| File | Purpose |
|------|---------|
| `app.py` | Flask web server |
| `templates/index.html` | Main UI template |
| `static/style.css` | Styling |
| `static/makeup.js` | Client-side face detection & makeup rendering |
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
