const LIPS_UPPER_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291];
const LIPS_LOWER_OUTER = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61];
const LIPS_FULL = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];

const LEFT_UPPER_LID = [226, 247, 30, 29, 27, 28, 56, 190, 173, 157, 158, 159, 160, 161, 246];
const RIGHT_UPPER_LID = [446, 467, 260, 259, 257, 258, 286, 414, 398, 384, 385, 386, 387, 388, 466];

const LEFT_CHEEK = [116, 123, 147, 213, 192, 214, 210, 211];
const RIGHT_CHEEK = [345, 352, 376, 433, 416, 434, 430, 431];

let cameraRunning = false;
let faceMeshCamera = null;
let faceMeshUpload = null;
let uploadedImage = null;
let lastUploadLandmarks = null;
let tempCanvas = null;
let tempCtx = null;

function getSettings(prefix) {
    function getActiveColor(sectionIndex) {
        const sections = document.querySelectorAll(`#${prefix}-tab .control-section`);
        if (sections[sectionIndex]) {
            const active = sections[sectionIndex].querySelector('.color-swatch.active');
            if (active) return active.dataset.color.split(',').map(Number);
        }
        return [200, 30, 30];
    }

    const pref = prefix === 'upload' ? 'upload-' : '';
    return {
        lipEnabled: document.getElementById(pref + 'lip-toggle').checked,
        lipColor: getActiveColor(0),
        lipOpacity: parseInt(document.getElementById(pref + 'lip-opacity').value) / 100,
        eyeEnabled: document.getElementById(pref + 'eye-toggle').checked,
        eyeColor: getActiveColor(1),
        eyeOpacity: parseInt(document.getElementById(pref + 'eye-opacity').value) / 100,
        blushEnabled: document.getElementById(pref + 'blush-toggle').checked,
        blushColor: getActiveColor(2),
        blushOpacity: parseInt(document.getElementById(pref + 'blush-opacity').value) / 100,
    };
}

function smoothPolygon(ctx, landmarks, indices, w, h) {
    const pts = indices.map(i => ({ x: landmarks[i].x * w, y: landmarks[i].y * h }));
    if (pts.length < 3) return;

    ctx.beginPath();

    const startX = (pts[pts.length - 1].x + pts[0].x) / 2;
    const startY = (pts[pts.length - 1].y + pts[0].y) / 2;
    ctx.moveTo(startX, startY);

    for (let i = 0; i < pts.length; i++) {
        const curr = pts[i];
        const next = pts[(i + 1) % pts.length];
        const cpx = (curr.x + next.x) / 2;
        const cpy = (curr.y + next.y) / 2;
        ctx.quadraticCurveTo(curr.x, curr.y, cpx, cpy);
    }
    ctx.closePath();
}

function drawMakeupRegion(ctx, landmarks, indices, w, h, r, g, b, opacity) {
    ctx.save();
    smoothPolygon(ctx, landmarks, indices, w, h);
    ctx.clip();

    ctx.globalAlpha = opacity * 0.55;
    ctx.globalCompositeOperation = 'multiply';
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(0, 0, w, h);

    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = opacity * 0.3;
    ctx.fillStyle = `rgba(${r},${g},${b},1)`;
    ctx.fillRect(0, 0, w, h);

    ctx.restore();
}

function applyAllMakeup(ctx, landmarks, settings, w, h) {
    if (settings.lipEnabled) {
        const [r, g, b] = settings.lipColor;
        drawMakeupRegion(ctx, landmarks, LIPS_FULL, w, h, r, g, b, settings.lipOpacity);

        ctx.save();
        smoothPolygon(ctx, landmarks, LIPS_FULL, w, h);
        ctx.clip();
        ctx.globalAlpha = settings.lipOpacity * 0.15;
        const upperCenter = landmarks[0];
        const lowerCenter = landmarks[17];
        const glossX = upperCenter.x * w;
        const glossY = ((upperCenter.y + lowerCenter.y) / 2) * h;
        const glossR = Math.abs(landmarks[291].x - landmarks[61].x) * w * 0.2;
        const glossGrad = ctx.createRadialGradient(glossX, glossY, 0, glossX, glossY, glossR);
        glossGrad.addColorStop(0, 'rgba(255,255,255,0.6)');
        glossGrad.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = glossGrad;
        ctx.fillRect(0, 0, w, h);
        ctx.restore();
    }

    if (settings.eyeEnabled) {
        const [r, g, b] = settings.eyeColor;
        drawMakeupRegion(ctx, landmarks, LEFT_UPPER_LID, w, h, r, g, b, settings.eyeOpacity);
        drawMakeupRegion(ctx, landmarks, RIGHT_UPPER_LID, w, h, r, g, b, settings.eyeOpacity);
    }

    if (settings.blushEnabled) {
        const [r, g, b] = settings.blushColor;

        [LEFT_CHEEK, RIGHT_CHEEK].forEach(indices => {
            let sumX = 0, sumY = 0;
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            indices.forEach(i => {
                const px = landmarks[i].x * w;
                const py = landmarks[i].y * h;
                sumX += px;
                sumY += py;
                minX = Math.min(minX, px);
                maxX = Math.max(maxX, px);
                minY = Math.min(minY, py);
                maxY = Math.max(maxY, py);
            });

            const cx = sumX / indices.length;
            const regionH = maxY - minY;
            const cy = sumY / indices.length - regionH * 0.25;

            const radius = Math.max(maxX - minX, maxY - minY) * 0.45;

            ctx.save();
            ctx.globalAlpha = settings.blushOpacity;
            const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
            grad.addColorStop(0, `rgba(${r},${g},${b},0.4)`);
            grad.addColorStop(0.4, `rgba(${r},${g},${b},0.25)`);
            grad.addColorStop(0.7, `rgba(${r},${g},${b},0.08)`);
            grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
            ctx.fillStyle = grad;
            ctx.fillRect(cx - radius, cy - radius, radius * 2, radius * 2);
            ctx.restore();
        });
    }
}

function applyMakeup(canvas, landmarks, settings, sourceImage) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    if (sourceImage) {
        ctx.drawImage(sourceImage, 0, 0, w, h);
    }

    applyAllMakeup(ctx, landmarks, settings, w, h);
}

function initFaceMesh(onResults) {
    const fm = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${file}`
    });
    fm.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
    });
    fm.onResults(onResults);
    return fm;
}

async function startCamera() {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('camera-canvas');
    const status = document.getElementById('camera-status');
    const startBtn = document.getElementById('start-camera-btn');
    const stopBtn = document.getElementById('stop-camera-btn');

    status.textContent = 'Loading face detection model...';

    faceMeshCamera = initFaceMesh((results) => {
        const w = canvas.width = results.image.width || 1280;
        const h = canvas.height = results.image.height || 720;
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        ctx.save();
        ctx.translate(w, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(results.image, 0, 0, w, h);
        ctx.restore();

        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            const lm = results.multiFaceLandmarks[0].map(p => ({
                x: 1 - p.x,
                y: p.y,
                z: p.z
            }));
            const settings = getSettings('camera');
            applyAllMakeup(ctx, lm, settings, w, h);
            status.textContent = '';
        } else {
            status.textContent = 'No face detected - look at the camera';
        }
    });

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user',
                frameRate: { ideal: 30 }
            }
        });
        video.srcObject = stream;
        await video.play();

        status.textContent = 'Starting...';
        startBtn.style.display = 'none';
        stopBtn.style.display = '';
        cameraRunning = true;

        function tick() {
            if (!cameraRunning) return;
            faceMeshCamera.send({ image: video }).then(() => {
                if (cameraRunning) requestAnimationFrame(tick);
            });
        }
        requestAnimationFrame(tick);

    } catch (err) {
        status.textContent = 'Camera access denied or unavailable: ' + err.message;
    }
}

function stopCamera() {
    cameraRunning = false;
    const video = document.getElementById('webcam');
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
    const canvas = document.getElementById('camera-canvas');
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

    document.getElementById('start-camera-btn').style.display = '';
    document.getElementById('stop-camera-btn').style.display = 'none';
    document.getElementById('camera-status').textContent = 'Camera stopped';
}

async function processUploadedImage() {
    if (!uploadedImage) return;

    const canvas = document.getElementById('upload-canvas');
    const status = document.getElementById('upload-status');

    const maxDim = 800;
    let w = uploadedImage.naturalWidth || uploadedImage.width;
    let h = uploadedImage.naturalHeight || uploadedImage.height;
    if (w > maxDim || h > maxDim) {
        const scale = maxDim / Math.max(w, h);
        w = Math.round(w * scale);
        h = Math.round(h * scale);
    }
    canvas.width = w;
    canvas.height = h;

    status.textContent = 'Detecting face...';

    if (!faceMeshUpload) {
        faceMeshUpload = initFaceMesh((results) => {
            const c = document.getElementById('upload-canvas');
            const st = document.getElementById('upload-status');
            const cw = c.width;
            const ch = c.height;
            if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
                lastUploadLandmarks = results.multiFaceLandmarks[0];
                renderUploadResult();
                st.textContent = '';
                document.getElementById('download-btn').style.display = '';
            } else {
                const ctx = c.getContext('2d');
                ctx.clearRect(0, 0, cw, ch);
                if (uploadedImage) ctx.drawImage(uploadedImage, 0, 0, cw, ch);
                st.textContent = 'No face detected in this photo';
            }
        });
    }

    await faceMeshUpload.send({ image: uploadedImage });
}

function renderUploadResult() {
    if (!uploadedImage || !lastUploadLandmarks) return;
    const canvas = document.getElementById('upload-canvas');
    const settings = getSettings('upload');
    applyMakeup(canvas, lastUploadLandmarks, settings, uploadedImage);
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tab.dataset.tab + '-tab').classList.add('active');
            if (tab.dataset.tab !== 'camera' && cameraRunning) {
                stopCamera();
            }
        });
    });

    document.getElementById('start-camera-btn').addEventListener('click', startCamera);
    document.getElementById('stop-camera-btn').addEventListener('click', stopCamera);

    document.querySelectorAll('#camera-tab .control-section').forEach(section => {
        section.querySelectorAll('.color-swatch').forEach(swatch => {
            swatch.addEventListener('click', () => {
                section.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('active'));
                swatch.classList.add('active');
            });
        });
    });

    document.querySelectorAll('#upload-tab .control-section').forEach(section => {
        section.querySelectorAll('.color-swatch').forEach(swatch => {
            swatch.addEventListener('click', () => {
                section.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('active'));
                swatch.classList.add('active');
                renderUploadResult();
            });
        });
    });

    document.querySelectorAll('#upload-tab input[type="range"], #upload-tab input[type="checkbox"]').forEach(input => {
        input.addEventListener('input', () => {
            renderUploadResult();
        });
    });

    document.getElementById('photo-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const img = new Image();
        img.onload = () => {
            uploadedImage = img;
            lastUploadLandmarks = null;
            processUploadedImage();
        };
        img.src = URL.createObjectURL(file);
    });

    document.getElementById('download-btn').addEventListener('click', () => {
        const canvas = document.getElementById('upload-canvas');
        const link = document.createElement('a');
        link.download = 'makeup-result.png';
        link.href = canvas.toDataURL('image/png');
        link.click();
    });
});
