"""
app.py — Makeup AI Web Dashboard
Serves the project overview and documentation on port 5000.
"""

from flask import Flask, render_template_string, jsonify
import sys
import os

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Makeup AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a0a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }

        header {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding: 1.5rem 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        header .logo {
            font-size: 2rem;
        }

        header h1 {
            font-size: 1.8rem;
            background: linear-gradient(90deg, #ff6b9d, #c44dff, #4dc9ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        header p {
            font-size: 0.9rem;
            color: #aaa;
            margin-top: 0.2rem;
        }

        .badge {
            display: inline-block;
            background: linear-gradient(90deg, #c44dff, #4dc9ff);
            color: white;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            margin-left: 0.5rem;
            vertical-align: middle;
        }

        main {
            max-width: 1100px;
            margin: 0 auto;
            padding: 2rem;
        }

        .hero {
            text-align: center;
            padding: 3rem 1rem 2rem;
        }

        .hero h2 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, #ff6b9d, #c44dff, #4dc9ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero p {
            font-size: 1.1rem;
            color: #bbb;
            max-width: 650px;
            margin: 0 auto;
            line-height: 1.7;
        }

        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-top: 3rem;
        }

        .card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 1.75rem;
            transition: transform 0.2s, border-color 0.2s;
        }

        .card:hover {
            transform: translateY(-3px);
            border-color: rgba(196, 77, 255, 0.4);
        }

        .card-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }

        .card h3 {
            font-size: 1.2rem;
            margin-bottom: 0.6rem;
            color: #fff;
        }

        .card p {
            font-size: 0.9rem;
            color: #aaa;
            line-height: 1.6;
        }

        .phases {
            margin-top: 3rem;
        }

        .phases h2 {
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            color: #fff;
        }

        .phase-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .phase-item {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            display: flex;
            align-items: flex-start;
            gap: 1.25rem;
        }

        .phase-num {
            font-size: 1.5rem;
            font-weight: 700;
            min-width: 2.5rem;
            height: 2.5rem;
            background: linear-gradient(135deg, #c44dff, #4dc9ff);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1rem;
        }

        .phase-content h4 {
            font-size: 1rem;
            color: #fff;
            margin-bottom: 0.3rem;
        }

        .phase-content p {
            font-size: 0.875rem;
            color: #999;
            line-height: 1.5;
        }

        .phase-item.active {
            border-color: rgba(196, 77, 255, 0.4);
            background: rgba(196, 77, 255, 0.07);
        }

        .phase-item.active .phase-num {
            box-shadow: 0 0 16px rgba(196, 77, 255, 0.5);
        }

        .stack {
            margin-top: 3rem;
        }

        .stack h2 {
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            color: #fff;
        }

        .stack-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 1rem;
        }

        .stack-item {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }

        .stack-item .si {
            font-size: 1.8rem;
            margin-bottom: 0.4rem;
        }

        .stack-item p {
            font-size: 0.85rem;
            color: #ccc;
        }

        .stack-item small {
            font-size: 0.75rem;
            color: #777;
        }

        footer {
            text-align: center;
            padding: 2rem;
            color: #555;
            font-size: 0.85rem;
            margin-top: 3rem;
            border-top: 1px solid rgba(255,255,255,0.05);
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">💄</div>
        <div>
            <h1>Makeup AI <span class="badge">Phase 3</span></h1>
            <p>Deep Learning Makeup Transfer with BeautyGAN</p>
        </div>
    </header>

    <main>
        <div class="hero">
            <h2>Automated Makeup Transfer</h2>
            <p>
                An advanced computer vision system that transfers makeup from a reference
                image to a source face using a GAN-based architecture — preserving identity
                while applying realistic makeup styles.
            </p>
        </div>

        <div class="cards">
            <div class="card">
                <div class="card-icon">🧠</div>
                <h3>BeautyGAN Architecture</h3>
                <p>U-Net Generator paired with a Dual Discriminator for global face and local feature discrimination. Mixed-precision training with PyTorch AMP.</p>
            </div>
            <div class="card">
                <div class="card-icon">🎯</div>
                <h3>Face Landmark Detection</h3>
                <p>MediaPipe-powered landmark detection with 468 facial points for precise lip, eye, and cheek region segmentation.</p>
            </div>
            <div class="card">
                <div class="card-icon">📦</div>
                <h3>ONNX Export</h3>
                <p>Production-ready inference with ONNX export support for cross-platform deployment and hardware acceleration.</p>
            </div>
            <div class="card">
                <div class="card-icon">📊</div>
                <h3>TensorBoard Logging</h3>
                <p>Full training visibility with loss tracking, validation metrics, and visual output logging via TensorBoard SummaryWriter.</p>
            </div>
        </div>

        <div class="phases">
            <h2>Project Roadmap</h2>
            <div class="phase-list">
                <div class="phase-item">
                    <div class="phase-num">1</div>
                    <div class="phase-content">
                        <h4>Phase 1 — Real-time Prototype ✅</h4>
                        <p>Live webcam makeup using polygon masking and color blending via MediaPipe + OpenCV. Lipstick, eyeshadow, and blush toggles with multiple color presets.</p>
                    </div>
                </div>
                <div class="phase-item">
                    <div class="phase-num">2</div>
                    <div class="phase-content">
                        <h4>Phase 2 — Improved Segmentation ✅</h4>
                        <p>Better eye region segmentation, eyeliner effects, foundation/skin smoothing, side-profile face handling, and multi-face support.</p>
                    </div>
                </div>
                <div class="phase-item active">
                    <div class="phase-num">3</div>
                    <div class="phase-content">
                        <h4>Phase 3 — GAN-based Makeup Transfer 🚀 Current</h4>
                        <p>Full BeautyGAN implementation with U-Net Generator, histogram matching, perceptual loss functions, and ONNX production export. Training pipeline with TensorBoard visualization.</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="stack">
            <h2>Tech Stack</h2>
            <div class="stack-grid">
                <div class="stack-item">
                    <div class="si">🐍</div>
                    <p>Python 3.12</p>
                    <small>Runtime</small>
                </div>
                <div class="stack-item">
                    <div class="si">🔥</div>
                    <p>PyTorch</p>
                    <small>Deep Learning</small>
                </div>
                <div class="stack-item">
                    <div class="si">👁️</div>
                    <p>OpenCV</p>
                    <small>Computer Vision</small>
                </div>
                <div class="stack-item">
                    <div class="si">🗺️</div>
                    <p>MediaPipe</p>
                    <small>Face Landmarks</small>
                </div>
                <div class="stack-item">
                    <div class="si">📐</div>
                    <p>NumPy</p>
                    <small>Numerics</small>
                </div>
                <div class="stack-item">
                    <div class="si">📦</div>
                    <p>ONNX</p>
                    <small>Model Export</small>
                </div>
            </div>
        </div>
    </main>

    <footer>
        Makeup AI &mdash; Built with MediaPipe (Google) &amp; PyTorch &mdash; Phase 3
    </footer>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "python": sys.version})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
