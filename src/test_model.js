// --------------------------------------------------------------------------------------------------
// - Video handling ---------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------

const videoEl   = document.getElementById('video-holder');
const camToggle = document.getElementById('cam-toggle');
const processed_holder = document.getElementById('processed-video-holder');
const canvas = document.getElementById('canvas');
const result_display = document.getElementById('result-text');
const result_display_holder = document.getElementById('result-holder');
const fpsCounterText = document.getElementById("fps-counter-text");
const imageInput = document.getElementById('image-input');
const canvasCtx = canvas.getContext('2d');
const IMG_SIZE = 256;
const INV_255 = 1 / 255;
const CLASS_STATES = ["video-holder-unsafe", "video-holder-safe", "video-holder-empty"];
const UI_UPDATE_MS = 150; // throttle UI/paint updates to avoid slowing raw throughput
const BACKEND_PREFERENCE = ["webgpu", "webgl", "cpu"]; // prefer WebGPU → WebGL → CPU fallback
const TESTSUIT_VERSION = "V_2";
const MODEL_TO_USE = 'V_4_6/model_tfjs';

document.getElementById('version-holder').innerHTML = "TestSuit_V: " + TESTSUIT_VERSION + " | Model_V: " + MODEL_TO_USE

async function tensorData(tensor) {
    // Use async reads on WebGPU to avoid sync stalls; sync reads elsewhere
    if (tf.getBackend() === "webgpu") {
        return await tensor.data();
    }
    return tensor.dataSync();
}

let backendReadyPromise = null;
async function ensureBackend() {
    if (backendReadyPromise) return backendReadyPromise;
    backendReadyPromise = (async () => {
        for (const name of BACKEND_PREFERENCE) {
            try {
                await tf.setBackend(name);
                break;
            } catch (err) {
                console.warn(`Failed to set backend ${name}, trying next`, err);
            }
        }
        await tf.ready();
        console.log("TF backend:", tf.getBackend());
    })();
    return backendReadyPromise;
}

let videoStreamRunning = false;

camToggle.addEventListener('change', async (event) => {
    if (event.target.checked) {
        try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoEl.srcObject = stream;
        videoEl.play();
        console.log("Camera started.");
        startVideoProcessing();
        } catch (err) {
        console.error("Error accessing camera:", err);
        }
    } else {
        const stream = videoEl.srcObject;
        if (stream) {
        stream.getTracks().forEach(track => track.stop());
        videoEl.srcObject = null;
        console.log("Camera stopped.");
        }
        stopVideoProcessing();
    }
});

async function startVideoProcessing() {
    if (!videoEl.srcObject || videoStreamRunning) return;
    videoStreamRunning = true;
    resetFpsCounters();

    classifyFromCameraLoop_V2(); // 🔥 startet parallele Klassifikation
}

function stopVideoProcessing() {
  videoStreamRunning = false;
}


// --------------------------------------------------------------------------------------------------
// - Classification ---------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------

// app.js
let model;
let labels;
let modelLoadingPromise = null;

async function loadModelAndLabels() {
    if (model && labels) return;
    if (modelLoadingPromise) {
        await modelLoadingPromise;
        return;
    }
    modelLoadingPromise = (async () => {
        await ensureBackend();
        model = await tf.loadLayersModel('../../models/' + MODEL_TO_USE + '/model.json');
        const labelsRes = await fetch('../../models/' + MODEL_TO_USE + '/labels.json');
        // model = await tf.loadLayersModel('../../models/V_4_6_2/model_tfjs_q8/model.json');
        // const labelsRes = await fetch('../../models/V_4_6_2/model_tfjs_q8/labels.json');
        labels = await labelsRes.json();
        console.log("Model and labels loaded.");
    })();
    await modelLoadingPromise;
}

function preprocessImage(imgElement) {
    // Draw and resize to IMG_SIZExIMG_SIZE
    canvasCtx.drawImage(imgElement, 0, 0, IMG_SIZE, IMG_SIZE);
    return tf.browser.fromPixels(canvas)
        .toFloat()
        .mul(INV_255)
        .expandDims(0); // shape: [1, IMG_SIZE, IMG_SIZE, 3]
}

function topIndex(probs) {
    let maxVal = probs[0];
    let maxIdx = 0;
    for (let i = 1; i < probs.length; i++) {
        if (probs[i] > maxVal) {
            maxVal = probs[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}

function applyBorderState(label) {
    processed_holder.classList.remove(...CLASS_STATES);
    result_display_holder.classList.remove(...CLASS_STATES);
    if (label === "unsafe") {
        processed_holder.classList.add("video-holder-unsafe");
        result_display_holder.classList.add("video-holder-unsafe");
    } else if (label === "empty") {
        processed_holder.classList.add("video-holder-empty");
        result_display_holder.classList.add("video-holder-empty");
    } else {
        processed_holder.classList.add("video-holder-safe");
        result_display_holder.classList.add("video-holder-safe");
    }
}

function formatProbs(probs) {
    const pairs = [];
    for (let i = 0; i < labels.length; i++) {
        pairs.push([labels[i], probs[i]]);
    }
    pairs.sort((a, b) => b[1] - a[1]);
    return pairs.map(([l, p]) => `${l}: ${(p * 100).toFixed(1)}%`).join(", ");
}

async function classifyImage(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = async () => {
            try {
                const prediction = tf.tidy(() => {
                    const inputTensor = preprocessImage(img);
                    return model.predict(inputTensor);
                });
                const probs = await tensorData(prediction);
                prediction.dispose?.();

                const bestIdx = topIndex(probs);

                applyBorderState(labels[bestIdx]);
                fpsCounterText.innerText = `... FPS`;
                fpsCounterText.style.color = "black";

                const formatted = formatProbs(probs);
                console.log("Predictions:", formatted);
                result_display.textContent = formatted;
                resolve(labels[bestIdx]);
            } catch (err) {
                reject(err);
            } finally {
                URL.revokeObjectURL(img.src);
            }
        };
        img.onerror = (err) => {
            URL.revokeObjectURL(img.src);
            reject(err);
        };
        img.src = URL.createObjectURL(file);
    });
}

function classifyFileForBatch(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = async () => {
            try {
                const topIdx = await tf.tidy(async () => {
                    const inputTensor = preprocessImage(img);
                    const prediction = model.predict(inputTensor);
                    const arg = prediction.argMax(-1);
                    const res = await tensorData(arg);
                    arg.dispose?.();
                    return res[0];
                });
                resolve(labels[topIdx]);
            } catch (err) {
                reject(err);
            } finally {
                URL.revokeObjectURL(img.src);
            }
        };
        img.onerror = (err) => {
            URL.revokeObjectURL(img.src);
            reject(err);
        };
        img.src = URL.createObjectURL(file);
    });
}

async function classifyFiles(files) {
    if (!files.length) return;
    if (!model || !labels) await loadModelAndLabels();

    const counts = Object.fromEntries(labels.map(l => [l, 0]));
    const start = performance.now();

    for (const file of files) {
        try {
            const label = await classifyFileForBatch(file);
            counts[label] = (counts[label] || 0) + 1;
        } catch (err) {
            console.error("Error classifying file:", file?.name || file, err);
        }
    }

    const elapsedMs = performance.now() - start;
    const fps = files.length ? (files.length / (elapsedMs / 1000)) : 0;
    const avgMs = files.length ? elapsedMs / files.length : 0;

    processed_holder.classList.remove(...CLASS_STATES);
    result_display_holder.classList.remove(...CLASS_STATES);

    result_display.textContent = labels.map(l => `${l}: ${counts[l] ?? 0}`).join("; ");

    const fpsText = files.length ? `~ ${fps.toFixed(1)} FPS (avg ${avgMs.toFixed(1)} ms/img)` : "... FPS";
    fpsCounterText.innerText = fpsText;
    fpsCounterText.style.color = "black";
}

imageInput.addEventListener('change', async (event) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;

    if (!model || !labels) await loadModelAndLabels();

    if (files.length === 1) {
        await classifyImage(files[0]);
        return;
    }

    await classifyFiles(files);
});

let lastTime = performance.now();
let frames = 0;

function resetFpsCounters() {
    frames = 0;
    lastTime = performance.now();
}

async function classifyFromCameraLoop() {
    if (!model || !labels) await loadModelAndLabels();

    resetFpsCounters();

    async function loop() {
        if (!videoStreamRunning) return;

        // Video wurde bereits im Canvas gezeichnet → keine drawImage mehr nötig
        const inputTensor = preprocessImage(canvas);
        const prediction = model.predict(inputTensor);
        const probs = await prediction.data();

        console.log("Loaded labels:", labels);

        console.log("Predictions:", labels
            .map((l, i) => [l, probs[i]])
            .sort((a, b) => b[1] - a[1])
            .map(([l, p]) => `${l}: ${(p * 100).toFixed(1)}%`)
            .join(", ")
        );

        result_display.textContent = labels.map((l, i) => [l, probs[i]]).sort((a, b) => b[1] - a[1]).map(([l, p]) => `${l}: ${(p * 100).toFixed(1)}%`).join(", ");

        processed_holder.classList.remove("video-holder-unsafe", "video-holder-safe", "video-holder-empty");
        result_display_holder.classList.remove("video-holder-unsafe", "video-holder-safe", "video-holder-empty");
        const topIdx = probs.indexOf(Math.max(...probs));
        if (labels[topIdx] == "unsafe") {
            processed_holder.classList.add("video-holder-unsafe");
            result_display_holder.classList.add("video-holder-unsafe");
        } else if (labels[topIdx] == "empty") {
            processed_holder.classList.add("video-holder-empty");
            result_display_holder.classList.add("video-holder-empty");
        } else {
            processed_holder.classList.add("video-holder-safe");
            result_display_holder.classList.add("video-holder-safe");
        }

        // FPS counter logic
        frames++;
        const now = performance.now();
        const elapsed = now - lastTime;
        if (elapsed >= 1000) {
            const fps = (frames * 1000) / elapsed;
            document.getElementById("fps-counter-text").innerText = `~ ${fps.toFixed(1)} FPS`;
            if (fps >= 30) {
                document.getElementById("fps-counter-text").style.color = "limegreen";
            } else if (fps >= 10) {
                document.getElementById("fps-counter-text").style.color = "orange";
            } else {
                document.getElementById("fps-counter-text").style.color = "red";
            }
            frames = 0;
            lastTime = now;
        }

        requestAnimationFrame(loop);
    }

    loop();
}

function preprocessFromVideo(videoEl) {
    // Preprocessing for IMG_SIZE×IMG_SIZE, need to be the same as training
    const t = tf.browser.fromPixels(videoEl);                   // [H,W,3], RGB
    const r = tf.image.resizeBilinear(t, [IMG_SIZE, IMG_SIZE]);           // alignCorners=false by default (matches TF default)
    const x = r.toFloat().mul(INV_255);                         // match [0, 1] normalization used during training
    const input = x.expandDims(0);                              // [1,IMG_SIZE,IMG_SIZE,3]
    return input;
}

async function classifyFromCameraLoop_V2() {
    if (!model || !labels) await loadModelAndLabels();

    let firstLogDone = false;

    resetFpsCounters();

    async function loop() {
        if (!videoStreamRunning) return;

        // 1) Visualize exactly what the model sees
        canvasCtx.drawImage(videoEl, 0, 0, IMG_SIZE, IMG_SIZE);

        // 2) Classify straight from the video element (not the canvas)
        if (videoEl.readyState >= 2) {
            const prediction = tf.tidy(() => {
                const input = preprocessFromVideo(videoEl);
                return model.predict(input);        // softmax from Keras layers model
            });
            const probs = await tensorData(prediction);
            prediction.dispose?.();

            const topIdx = topIndex(probs);

            // One-time sanity logs
            if (!firstLogDone) {
                console.log("--- Sanity Check on first frame ----------------------------------------------------------")
                console.log("Used Model: " + MODEL_TO_USE);
                console.log("model units:", model.layers.at(-1).units, "labels length:", labels.length);
                const sum = probs[0] + probs[1] + probs[2];
                console.log("sum(probs) ~", sum.toFixed(4), "top:", labels[topIdx], "p=", probs[topIdx].toFixed(3));
                firstLogDone = true;
                console.log("------------------------------------------------------------------------------------------")
            }

            // Update UI
            result_display.textContent = labels
                .map((l, i) => [l, probs[i]])
                .sort((a, b) => b[1] - a[1])
                .map(([l, p]) => `${l}: ${(p * 100).toFixed(1)}%`)
                .join(", ");

            applyBorderState(labels[topIdx]);

            // FPS counter logic
            frames++;
            const now = performance.now();
            const elapsed = now - lastTime;
            if (elapsed >= 1000) {
                const fps = (frames * 1000) / elapsed;
                fpsCounterText.innerText = `~ ${fps.toFixed(1)} FPS`;
                if (fps >= 30) {
                    fpsCounterText.style.color = "limegreen";
                } else if (fps >= 10) {
                    fpsCounterText.style.color = "orange";
                } else {
                    fpsCounterText.style.color = "red";
                }
                frames = 0;
                lastTime = now;
            }

        }

        setTimeout(loop, 0); // run as fast as the model/browser allow, not tied to display refresh
    }

    setTimeout(loop, 0);
}
