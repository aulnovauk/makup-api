# Makeup AI

## Overview
A deep learning computer vision project for automated makeup transfer. Evolved through 3 phases, currently implementing a BeautyGAN architecture that transfers makeup from a reference image to a source face while preserving identity.

## Architecture (Phase 3)
- **Generator**: U-Net Generator (`generator.py`)
- **Discriminator**: Dual Discriminator in `beautygan.py`
- **Loss**: Custom `MakeupGANLoss` with perceptual + GAN losses (`losses.py`)
- **Training**: Mixed-precision with PyTorch AMP (`trainer.py`)
- **Inference**: Production engine with ONNX export (`inference.py`)
- **Data**: BeautyGAN dataset pipeline (`dataset.py`)
- **Utils**: Shared device/path helpers (`common.py`), histogram matching (`histogram.py`)

## Web Interface
- `app.py`: Flask dashboard served on port 5000 — shows project info/overview

## Tech Stack
- **Python 3.12**
- **PyTorch** — deep learning & GAN training
- **OpenCV** — image processing
- **MediaPipe** — face landmark detection (468 points)
- **NumPy** — numerical operations
- **ONNX** — model export for production
- **TensorBoard** — training visualization
- **Flask** — web dashboard
- **Gunicorn** — production WSGI server

## Running the App
- Development: `python app.py` (port 5000)
- Production: `gunicorn --bind=0.0.0.0:5000 --reuse-port app:app`

## Key Files
| File | Purpose |
|------|---------|
| `app.py` | Flask web dashboard |
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

## Dependencies
Install with: `pip install -r requirements.txt`
Also required: `flask`, `gunicorn`, `torch`, `torchvision`
