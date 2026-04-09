# Makeup AI — Phase 1
## Real-time Lipstick & Eyeshadow using MediaPipe + OpenCV

---

## ⚡ Quick Setup (5 minutes)

### Step 1 — Install Python
Download Python 3.10+ from https://python.org
Make sure to check "Add Python to PATH" during install.

### Step 2 — Install dependencies
Open terminal / command prompt in this folder and run:
```
pip install -r requirements.txt
```

### Step 3 — Run the app
```
python makeup_ai.py
```

Your webcam will open with real-time makeup applied! 💄

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| L   | Toggle lipstick ON/OFF |
| E   | Toggle eyeshadow ON/OFF |
| B   | Toggle blush ON/OFF |
| 1   | Lipstick → Red Rose |
| 2   | Lipstick → Pink Blush |
| 3   | Lipstick → Coral Sunset |
| 4   | Lipstick → Berry Wine |
| 5   | Lipstick → Nude Beige |
| 6   | Lipstick → Deep Plum |
| a   | Eyeshadow → Smoky Brown |
| b   | Eyeshadow → Rose Gold |
| c   | Eyeshadow → Ocean Blue |
| d   | Eyeshadow → Forest Green |
| e   | Eyeshadow → Purple Haze |
| +   | Increase opacity |
| -   | Decrease opacity |
| D   | Show face landmarks (debug) |
| Q   | Quit |

---

## 🛠️ How It Works

1. **MediaPipe** detects 468 landmark points on your face every frame
2. Specific points are selected for lips, eyes, cheeks
3. A polygon mask is created from those points
4. Color is blended into that mask region
5. Gaussian blur softens edges for natural look
6. Gloss highlight is added to lips

---

## ➕ Adding Your Own Colors

Open `makeup_ai.py` and find `LIPSTICK_COLORS`:

```python
LIPSTICK_COLORS = {
    "1": {"name": "Red Rose", "bgr": (30, 20, 200)},
    # Add yours:
    "7": {"name": "My Color",  "bgr": (B, G, R)},  # BGR format!
}
```

Note: OpenCV uses BGR, not RGB. So red is (0, 0, 255).

---

## 🚨 Troubleshooting

**Webcam not opening?**
- Make sure no other app (Zoom, Teams) is using your camera
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

**pip install fails?**
- Try: `pip install mediapipe --upgrade`
- On Mac M1/M2: `pip install mediapipe-silicon`

**Slow FPS?**
- Reduce resolution in code: change `1280` to `640`
- Close other heavy apps

---

## 📁 Project Structure

```
makeup_ai_phase1/
├── makeup_ai.py       ← Main app (run this)
├── requirements.txt   ← Dependencies
└── README.md          ← This file
```

---

## 🚀 What's Next — Phase 2

- Better eye region segmentation
- Eyeliner effect
- Foundation / skin smoothing
- Handle side-profile faces
- Multi-face support testing

---

Built with ❤️ using MediaPipe (Google) + OpenCV — both FREE & open source.
