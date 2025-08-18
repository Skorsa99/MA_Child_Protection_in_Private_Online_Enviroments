const videoEl   = document.getElementById('video-holder');
const resulTxttEl = document.getElementById('result-text');
const resultEl = document.getElementById('result-holder');
const fpsTxtEl = document.getElementById('fps-counter-text');
const camToggle = document.getElementById('cam-toggle');

let startTime = Date.now();

let currentStream = null;
let stopFrameLoop = null;   // <-- add this

async function startCamera(){
    stopCamera(); // ensure everything is clean

    try {
        const constraints = {
            video: { facingMode: "user", width: { ideal: 1280 }, height:{ ideal: 720 } },
            audio: false
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        currentStream = stream;
        videoEl.srcObject = stream;

        // start frame loop and keep the stop handle
        stopFrameLoop = startFrameLoop(
            document.getElementById('video-holder'),
            (imgData, canvas) => { /* optional callback */ }
        );

    } catch (err) {
        camToggle.checked = false;
        console.error(err);
        alert(
            (location.protocol !== 'https:' && location.hostname !== 'localhost')
            ? 'getUserMedia erfordert HTTPS oder localhost.'
            : 'Kamera konnte nicht gestartet werden. Prüfe Berechtigungen und Geräte.'
        );
    }
}

function stopCamera(){
    // stop the loop FIRST so it doesn't keep reading frames
    if (typeof stopFrameLoop === 'function') {
        stopFrameLoop();
        stopFrameLoop = null;
    }

    if (currentStream){
        currentStream.getTracks().forEach(t => t.stop());
        currentStream = null;
    }

    videoEl.srcObject = null;

    // ❌ Do NOT start a new frame loop here.
    // (Your old code called startFrameLoop() again — that was the bug.)
}

// Toggle steuert Kamera
camToggle.addEventListener('change', () => {
    if (camToggle.checked) startCamera(); else stopCamera();
});

// Optional: beim Laden NICHT automatisch starten – User soll togglen.
// Wenn du automatisch starten willst, einfach:
// window.addEventListener('load', () => { camToggle.checked = true; startCamera(); });

// (Optional) Nette Kleinigkeit: auf Größenänderungen reagieren
window.addEventListener('beforeunload', stopCamera);

function startFrameLoop(videoEl, callback) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = IMG_SIZE;
    canvas.height = IMG_SIZE;

    const processedDiv = document.getElementById('processed_video_holder');
    if (processedDiv){
        processedDiv.innerHTML = '';
        processedDiv.appendChild(canvas);
    }

    let running = true;
    let rafId = null;

    async function loop() {
        if (!running) return;

        // guard: skip if no frame available
        if (!videoEl.srcObject || videoEl.readyState < 2 || videoEl.videoWidth === 0) {
            rafId = requestAnimationFrame(loop);
            return;
        }

        ctx.drawImage(videoEl, 0, 0, IMG_SIZE, IMG_SIZE);

        let probs = null;
        try { probs = await classifyFrameFromCanvas(canvas); } catch {}

        if (probs) {
            // top class
            let maxIdx = 0, maxVal = -1;
            for (let i = 0; i < probs.length; i++) { if (probs[i] > maxVal) { maxVal = probs[i]; maxIdx = i; } }
            const topLabel = CLASSES[maxIdx] ?? ('class ' + maxIdx);

            // pass probs so updateBorder can log them
            updateBorder(topLabel, probs);
            // drawOverlay(ctx, probs);
        }

        if (typeof callback === 'function') {
            try { callback(ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE), canvas); } catch {}
        }

        rafId = requestAnimationFrame(loop);
    }

    loop();

    // return stopper
    return () => {
        running = false;
        if (rafId !== null) cancelAnimationFrame(rafId);
    };
}







// ------------------------------------------------------------------------------------------
// - Classification logic -------------------------------------------------------------------
// ------------------------------------------------------------------------------------------
const MODEL_PATH = '/models_V2/V_0_1/model.json';
const META_PATH  = '/models_V2/V_0_1/labels.json';
let model = undefined;
let META  = { classes: ['unsafe','safe','empty'], img_size: 224 };
let IMG_SIZE = 224;
let CLASSES  = META.classes;

async function loadModelAndMeta() {
    try {
        const res = await fetch(META_PATH);
        if (res.ok) {
            META = await res.json();
            if (Array.isArray(META.classes)) CLASSES = META.classes;
            if (typeof META.img_size === 'number') IMG_SIZE = META.img_size;
        }
    } catch (e) { console.warn('Failed to load labels.json, using defaults.', e); }
    model = await tf.loadLayersModel(MODEL_PATH);
    console.log('TFJS model loaded:', MODEL_PATH, 'img_size=', IMG_SIZE, 'classes=', CLASSES);
}

function updateBorder(topLabel, probs) {
    videoEl.classList.remove('video-holder-safe', 'video-holder-unsafe', 'video-holder-empty');
    resultEl.classList.remove('video-holder-safe', 'video-holder-unsafe', 'video-holder-empty');
    
    // Build probability string with fixed format: xx.xx%
    const probText = CLASSES.map((label, i) => {
        return `${label}: ${(probs[i] * 100).toFixed(2).padStart(5, ' ')}%`;
    }).join(' | ');

    // Console log
    // console.log(`${String(topLabel).toUpperCase()} — ${probText}`);

    // Update #result-text div
    if (resulTxttEl) {
        resulTxttEl.textContent = `${probText}`;
    }

    let excactElapsedTime = Date.now() - startTime;
    let fps = 1000 / excactElapsedTime;
    startTime = Date.now();
    fpsTxtEl.textContent = `FPS: ${fps.toFixed(2)}`;

    if (String(topLabel).toLowerCase() === 'empty') {
        videoEl.classList.add('video-holder-empty');
        resultEl.classList.add('video-holder-empty');
    } else if (String(topLabel).toLowerCase() === 'unsafe') {
        videoEl.classList.add('video-holder-unsafe');
        resultEl.classList.add('video-holder-unsafe');
    } else {
        videoEl.classList.add('video-holder-safe');
        resultEl.classList.add('video-holder-safe');
    }
}

function drawOverlay(ctx, probs) {
    if (!probs || !probs.length) return;
    const w = ctx.canvas.width, h = ctx.canvas.height;
    ctx.save();
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(0, 0, w, Math.max(30, 16 * CLASSES.length + 10));
    ctx.fillStyle = 'white';
    ctx.font = '12px sans-serif';
    let y = 16;
    let maxIdx = 0, maxVal = -1;
    for (let i = 0; i < probs.length; i++) { if (probs[i] > maxVal) { maxVal = probs[i]; maxIdx = i; } }
    for (let i = 0; i < probs.length; i++) {
        const p = probs[i];
        const label = (CLASSES[i] !== undefined) ? CLASSES[i] : ('class ' + i);
        const star = (i === maxIdx) ? '★ ' : '';
        ctx.fillText(`${star}${label}: ${(p*100).toFixed(1)}%`, 8, y);
        y += 16;
    }
    ctx.restore();
}

async function classifyFrameFromCanvas(canvas) {
    if (!model) return null;
    // IMPORTANT: Do not return a Promise from inside tf.tidy().
    // Create the tensor inside tidy, return a TENSOR, then await .data() outside.
    const probsTensor = tf.tidy(() => {
        const input = tf.browser.fromPixels(canvas)
            .resizeNearestNeighbor([IMG_SIZE, IMG_SIZE])
            .toFloat()
            .div(255)
            .expandDims(0);
        const logits = model.predict(input);
        const t = Array.isArray(logits) ? logits[0] : logits; // ensure Tensor
        // Apply softmax unconditionally; if the model already has softmax, this is idempotent for most cases.
        return tf.softmax(t);
    });
    const probsArr = await probsTensor.data();
    probsTensor.dispose();
    return Array.from(probsArr);
}

// Replace startFrameLoop to use dynamic IMG_SIZE and run inference
function startFrameLoop(videoEl, callback) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = IMG_SIZE;
    canvas.height = IMG_SIZE;
    canvas.style.maxHeight = '100%';

    processedDiv = document.getElementById('processed_video_holder');
    processedDiv.innerHTML = '';
    processedDiv.appendChild(canvas);

    let running = true;

    async function loop() {
        if (!running) return;
        ctx.drawImage(videoEl, 0, 0, IMG_SIZE, IMG_SIZE);
        let probs = null;
        try {
            probs = await classifyFrameFromCanvas(canvas);
        } catch (e) {
            // swallow inference errors for robustness
        }
        if (probs) {
            // Determine top class
            let maxIdx = 0, maxVal = -1;
            for (let i = 0; i < probs.length; i++) { if (probs[i] > maxVal) { maxVal = probs[i]; maxIdx = i; } }
            const topLabel = (CLASSES[maxIdx] !== undefined) ? CLASSES[maxIdx] : ('class ' + maxIdx);
            updateBorder(topLabel, probs);
            // drawOverlay(ctx, probs);
        }

        // If a callback was provided, pass raw ImageData too
        if (typeof callback === 'function') {
            const imgData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
            try { callback(imgData, canvas); } catch (e) {}
        }

        requestAnimationFrame(loop);
    }

    loop();
    return () => { running = false; };
}

// Kick off loading
loadModelAndMeta();