
const videoEl   = document.getElementById('video-holder');
const camToggle = document.getElementById('cam-toggle');
const processedDiv = document.getElementById('processed_video_holder');

// --------- Model & metadata paths (keep consistent with backend export) ---------
const MODEL_PATH = '/models/V_0_11/model.json';
const META_PATH  = '/models/V_0_11/metadata.json';

// Defaults in case metadata.json is missing
let META    = { classes: ['unsafe','safe','empty'], img_size: 224, framework: 'tfjs-layers', created_by: 'frontend' };
let IMG_SIZE = META.img_size;
let CLASSES  = META.classes;

let model = undefined;
let currentStream = null;
let stopLoop = null;

async function loadModelAndMeta() {
  try {
    const res = await fetch(META_PATH);
    if (res.ok) {
      META = await res.json();
      if (Array.isArray(META.classes)) CLASSES = META.classes;
      if (typeof META.img_size === 'number') IMG_SIZE = META.img_size;
    }
  } catch (e) {
    console.warn('Failed to load metadata.json, using defaults.', e);
  }
  model = await tf.loadLayersModel(MODEL_PATH);
  console.log('TFJS model loaded:', MODEL_PATH, 'img_size=', IMG_SIZE, 'classes=', CLASSES);
}

function updateBorder(topLabel) {
  videoEl.classList.remove('video-holder-safe','video-holder-unsafe','video-holder-empty');
  const l = String(topLabel).toLowerCase();
  if (l === 'empty') videoEl.classList.add('video-holder-empty');
  else if (l === 'unsafe') videoEl.classList.add('video-holder-unsafe');
  else videoEl.classList.add('video-holder-safe');
}

function drawOverlay(ctx, probs) {
  if (!probs || !probs.length) return;
  const w = ctx.canvas.width;
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

// Inference: NO extra /255, NO extra softmax (Keras model already has softmax)
async function classifyFrameFromCanvas(canvas) {
  if (!model) return null;
  const logits = tf.tidy(() => {
    const input = tf.browser.fromPixels(canvas)
      .resizeBilinear([IMG_SIZE, IMG_SIZE]) // match training (bilinear)
      .toFloat()                            // keep 0..255 (EfficientNet preprocess is inside the model)
      .expandDims(0);
    const out = model.predict(input);
    return Array.isArray(out) ? out[0] : out;
  });
  const arr = await logits.data();
  logits.dispose();
  return Array.from(arr);
}

function startFrameLoop(videoEl, callback) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;

  processedDiv.innerHTML = '';
  processedDiv.appendChild(canvas);

  let running = true;

  async function loop() {
    if (!running) return;

    // Draw current video frame into canvas
    ctx.drawImage(videoEl, 0, 0, IMG_SIZE, IMG_SIZE);

    try {
      const probs = await classifyFrameFromCanvas(canvas);
      if (probs && probs.length) {
        // Top-1
        let maxIdx = 0, maxVal = -1;
        for (let i = 0; i < probs.length; i++) if (probs[i] > maxVal) { maxVal = probs[i]; maxIdx = i; }
        const topLabel = (CLASSES[maxIdx] !== undefined) ? CLASSES[maxIdx] : ('class ' + maxIdx);
        updateBorder(topLabel);
        drawOverlay(ctx, probs);
      }
      if (typeof callback === 'function') {
        const imgData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
        try { callback(imgData, canvas); } catch (_) {}
      }
    } catch (_) {
      // ignore single-frame inference errors to keep loop alive
    }

    requestAnimationFrame(loop);
  }
  loop();

  // return stopper
  return () => { running = false; };
}

async function startCamera(){
  stopCamera(); // ensure clean state
  try {
    const constraints = {
      video: { facingMode: "user", width: { ideal: 1280 }, height:{ ideal: 720 } },
      audio: false
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    currentStream = stream;
    videoEl.srcObject = stream;
    stopLoop = startFrameLoop(videoEl, () => {});
  } catch (err) {
    camToggle.checked = false;
    console.error(err);
    alert((location.protocol !== 'https:' && location.hostname !== 'localhost')
      ? 'getUserMedia erfordert HTTPS oder localhost.'
      : 'Kamera konnte nicht gestartet werden. Prüfe Berechtigungen und Geräte.');
  }
}

function stopCamera(){
  if (stopLoop) { stopLoop(); stopLoop = null; }
  if (currentStream){
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
  videoEl.srcObject = null;
}

// Toggle steuert Kamera
camToggle.addEventListener('change', () => {
  if (camToggle.checked) startCamera(); else stopCamera();
});
window.addEventListener('beforeunload', stopCamera);

// Load model & metadata
loadModelAndMeta();
  