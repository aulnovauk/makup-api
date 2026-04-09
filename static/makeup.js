const LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95];
const LEFT_EYE_SHADOW = [226,247,30,29,27,28,56,190,243,112,26,22,23,24,110,25];
const RIGHT_EYE_SHADOW = [446,467,260,259,257,258,286,414,463,341,256,252,253,254,339,255];
const LEFT_CHEEK = [116,123,147,213,192,214,210,211];
const RIGHT_CHEEK = [345,352,376,433,416,434,430,431];

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

function drawPolygon(ctx, landmarks, indices, w, h) {
    ctx.beginPath();
    const first = landmarks[indices[0]];
    ctx.moveTo(first.x * w, first.y * h);
    for (let i = 1; i < indices.length; i++) {
        const pt = landmarks[indices[i]];
        ctx.lineTo(pt.x * w, pt.y * h);
    }
    ctx.closePath();
}

function smoothPolygon(ctx, landmarks, indices, w, h) {
    const pts = indices.map(i => ({ x: landmarks[i].x * w, y: landmarks[i].y * h }));
    if (pts.length < 3) return;

    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);

    for (let i = 0; i < pts.length; i++) {
        const curr = pts[i];
        const next = pts[(i + 1) % pts.length];
        const cpx = (curr.x + next.x) / 2;
        const cpy = (curr.y + next.y) / 2;
        ctx.quadraticCurveTo(curr.x, curr.y, cpx, cpy);
    }
    ctx.closePath();
}

function applyMakeup(canvas, landmarks, settings, sourceImage) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (sourceImage) {
        ctx.drawImage(sourceImage, 0, 0, w, h);
    }

    ctx.save();

    if (settings.lipEnabled) {
        const [r, g, b] = settings.lipColor;

        ctx.save();
        smoothPolygon(ctx, landmarks, LIPS_OUTER, w, h);
        ctx.clip();

        ctx.globalAlpha = settings.lipOpacity * 0.6;
        ctx.globalCompositeOperation = 'multiply';
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(0, 0, w, h);

        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = settings.lipOpacity * 0.35;
        ctx.fillStyle = `rgba(${r},${g},${b},1)`;
        ctx.fillRect(0, 0, w, h);

        ctx.globalAlpha = settings.lipOpacity * 0.2;
        const glossY = landmarks[13].y * h;
        const glossX = ((landmarks[0].x + landmarks[267].x) / 2) * w;
        const glossGrad = ctx.createRadialGradient(glossX, glossY, 0, glossX, glossY, 12);
        glossGrad.addColorStop(0, 'rgba(255,255,255,0.7)');
        glossGrad.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = glossGrad;
        ctx.fillRect(0, 0, w, h);

        ctx.restore();
    }

    if (settings.eyeEnabled) {
        const [r, g, b] = settings.eyeColor;

        [LEFT_EYE_SHADOW, RIGHT_EYE_SHADOW].forEach(indices => {
            ctx.save();
            smoothPolygon(ctx, landmarks, indices, w, h);
            ctx.clip();

            ctx.globalAlpha = settings.eyeOpacity * 0.5;
            ctx.globalCompositeOperation = 'multiply';
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(0, 0, w, h);

            ctx.globalCompositeOperation = 'source-over';
            ctx.globalAlpha = settings.eyeOpacity * 0.3;
            ctx.fillStyle = `rgba(${r},${g},${b},1)`;
            ctx.fillRect(0, 0, w, h);

            ctx.restore();
        });
    }

    if (settings.blushEnabled) {
        const [r, g, b] = settings.blushColor;

        [LEFT_CHEEK, RIGHT_CHEEK].forEach(indices => {
            const cx = indices.reduce((s, i) => s + landmarks[i].x, 0) / indices.length * w;
            const cy = indices.reduce((s, i) => s + landmarks[i].y, 0) / indices.length * h;

            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            indices.forEach(i => {
                minX = Math.min(minX, landmarks[i].x * w);
                maxX = Math.max(maxX, landmarks[i].x * w);
                minY = Math.min(minY, landmarks[i].y * h);
                maxY = Math.max(maxY, landmarks[i].y * h);
            });
            const radius = Math.max(maxX - minX, maxY - minY) * 0.8;

            ctx.save();
            ctx.globalAlpha = settings.blushOpacity;
            const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
            grad.addColorStop(0, `rgba(${r},${g},${b},0.45)`);
            grad.addColorStop(0.5, `rgba(${r},${g},${b},0.2)`);
            grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
            ctx.fillStyle = grad;
            ctx.fillRect(cx - radius, cy - radius, radius * 2, radius * 2);
            ctx.restore();
        });
    }

    ctx.restore();
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
        const w = canvas.width = results.image.width || 640;
        const h = canvas.height = results.image.height || 480;
        const ctx = canvas.getContext('2d');

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
            if (!tempCanvas) {
                tempCanvas = document.createElement('canvas');
                tempCtx = tempCanvas.getContext('2d');
            }
            tempCanvas.width = w;
            tempCanvas.height = h;
            tempCtx.drawImage(canvas, 0, 0);

            applyMakeupOverlay(canvas, lm, settings, tempCanvas);
            status.textContent = '';
        } else {
            status.textContent = 'No face detected - look at the camera';
        }
    });

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
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

function applyMakeupOverlay(canvas, landmarks, settings, baseImage) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(baseImage, 0, 0, w, h);

    ctx.save();

    if (settings.lipEnabled) {
        const [r, g, b] = settings.lipColor;

        ctx.save();
        smoothPolygon(ctx, landmarks, LIPS_OUTER, w, h);
        ctx.clip();

        ctx.globalAlpha = settings.lipOpacity * 0.6;
        ctx.globalCompositeOperation = 'multiply';
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(0, 0, w, h);

        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = settings.lipOpacity * 0.35;
        ctx.fillStyle = `rgba(${r},${g},${b},1)`;
        ctx.fillRect(0, 0, w, h);

        ctx.globalAlpha = settings.lipOpacity * 0.2;
        const glossY = landmarks[13].y * h;
        const glossX = ((landmarks[0].x + landmarks[267].x) / 2) * w;
        const glossGrad = ctx.createRadialGradient(glossX, glossY, 0, glossX, glossY, 12);
        glossGrad.addColorStop(0, 'rgba(255,255,255,0.7)');
        glossGrad.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = glossGrad;
        ctx.fillRect(0, 0, w, h);

        ctx.restore();
    }

    if (settings.eyeEnabled) {
        const [r, g, b] = settings.eyeColor;

        [LEFT_EYE_SHADOW, RIGHT_EYE_SHADOW].forEach(indices => {
            ctx.save();
            smoothPolygon(ctx, landmarks, indices, w, h);
            ctx.clip();

            ctx.globalAlpha = settings.eyeOpacity * 0.5;
            ctx.globalCompositeOperation = 'multiply';
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(0, 0, w, h);

            ctx.globalCompositeOperation = 'source-over';
            ctx.globalAlpha = settings.eyeOpacity * 0.3;
            ctx.fillStyle = `rgba(${r},${g},${b},1)`;
            ctx.fillRect(0, 0, w, h);

            ctx.restore();
        });
    }

    if (settings.blushEnabled) {
        const [r, g, b] = settings.blushColor;

        [LEFT_CHEEK, RIGHT_CHEEK].forEach(indices => {
            const cx = indices.reduce((s, i) => s + landmarks[i].x, 0) / indices.length * w;
            const cy = indices.reduce((s, i) => s + landmarks[i].y, 0) / indices.length * h;

            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            indices.forEach(i => {
                minX = Math.min(minX, landmarks[i].x * w);
                maxX = Math.max(maxX, landmarks[i].x * w);
                minY = Math.min(minY, landmarks[i].y * h);
                maxY = Math.max(maxY, landmarks[i].y * h);
            });
            const radius = Math.max(maxX - minX, maxY - minY) * 0.8;

            ctx.save();
            ctx.globalAlpha = settings.blushOpacity;
            const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
            grad.addColorStop(0, `rgba(${r},${g},${b},0.45)`);
            grad.addColorStop(0.5, `rgba(${r},${g},${b},0.2)`);
            grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
            ctx.fillStyle = grad;
            ctx.fillRect(cx - radius, cy - radius, radius * 2, radius * 2);
            ctx.restore();
        });
    }

    ctx.restore();
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
