// --------------------------------------------------------------------------------------------------
// - Video handling ---------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------

const videoEl   = document.getElementById('video-holder');
const camToggle = document.getElementById('cam-toggle');
const processed_holder = document.getElementById('processed-video-holder');
const canvas = document.getElementById('canvas');
const result_display = document.getElementById('result-text');
const result_display_holder = document.getElementById('result-holder');

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

    const ctx = canvas.getContext('2d');

    const drawFrame = () => {
        if (!videoStreamRunning) return;

        // Draw current video frame into canvas (resized to 224x224)
        ctx.drawImage(videoEl, 0, 0, 224, 224);

        // You could also classify here if needed
        // const imgTensor = preprocessImage(canvas);

        requestAnimationFrame(drawFrame);
    };

    drawFrame();
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

async function loadModelAndLabels() {
    model = await tf.loadLayersModel('../../models/V_1_11/model_tfjs/model.json');
    const labelsRes = await fetch('../../models/V_1_11/model_tfjs/labels.json');
    labels = await labelsRes.json();
    console.log("Model and labels loaded.");
}

function preprocessImage(imgElement) {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    // Draw and resize to 224x224
    ctx.drawImage(imgElement, 0, 0, 224, 224);

    const imageData = ctx.getImageData(0, 0, 224, 224);
    let imgTensor = tf.browser.fromPixels(imageData)
        .toFloat()
        .div(127.5)
        .sub(1.0) // match [-1, 1] expected by MobileNetV3

    return imgTensor.expandDims(0); // shape: [1, 224, 224, 3]
}

async function classifyImage(file) {
    const img = new Image();
    img.onload = async () => {
        const inputTensor = preprocessImage(img);
        const prediction = model.predict(inputTensor);
        const probs = await prediction.data();

        const topIdx = probs.indexOf(Math.max(...probs));
        const label = labels[topIdx];
        
        processed_holder.classList.remove("video-holder-unsafe", "video-holder-safe", "video-holder-empty");
        result_display_holder.classList.remove("video-holder-unsafe", "video-holder-safe", "video-holder-empty");
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
        
        document.getElementById("fps-counter-text").innerText = `... FPS`;
        document.getElementById("fps-counter-text").style.color = "black";

        console.log("Predictions:", labels.map((l, i) => [l, probs[i]]).sort((a, b) => b[1] - a[1]).map(([l, p]) => `${l}: ${(p * 100).toFixed(1)}%`).join(", "));

        result_display.textContent = labels.map((l, i) => [l, probs[i]]).sort((a, b) => b[1] - a[1]).map(([l, p]) => `${l}: ${(p * 100).toFixed(1)}%`).join(", ");
    };
    img.src = URL.createObjectURL(file);
}

document.getElementById('image-input').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        if (!model || !labels) await loadModelAndLabels();
        classifyImage(file);
    }
});

let lastTime = performance.now();
let frames = 0;

async function classifyFromCameraLoop() {
    if (!model || !labels) await loadModelAndLabels();

    const ctx = canvas.getContext('2d');

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
            const fps = frames;
            document.getElementById("fps-counter-text").innerText = `~ ${fps} FPS`;
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
    // Preprocessing for 224×224, need to be the same as training
    const t = tf.browser.fromPixels(videoEl);                   // [H,W,3], RGB
    const r = tf.image.resizeBilinear(t, [224, 224]);           // alignCorners=false by default (matches TF default)
    const x = r.toFloat().div(127.5).sub(1.0);                  // MobileNetV2: [-1, 1]
    const input = x.expandDims(0);                              // [1,224,224,3]
    return input;
}

async function classifyFromCameraLoop_V2() {
    if (!model || !labels) await loadModelAndLabels();

    const ctx = canvas.getContext('2d');
    const targetMs = 1000 / 1000; // ~1000 FPS inference if we want to limit fps
    let lastInfer = 0;
    let firstLogDone = false;

    async function loop(ts) {
        if (!videoStreamRunning) return;

        // 1) Visualize exactly what the model sees
        ctx.drawImage(videoEl, 0, 0, 224, 224);

        // 2) Classify straight from the video element (not the canvas)
        if (videoEl.readyState >= 2 && (ts - lastInfer) >= targetMs) {
            const probs = await tf.tidy(() => {
                const input = preprocessFromVideo(videoEl);
                const pred = model.predict(input);        // softmax from Keras layers model
                return pred.dataSync();            // ✅ TypedArray, no Promise
            });

            const topIdx = probs.indexOf(Math.max(...probs));

            // One-time sanity logs
            if (!firstLogDone) {
                console.log("model units:", model.layers.at(-1).units, "labels length:", labels.length);
                const sum = probs[0] + probs[1] + probs[2];
                console.log("sum(probs) ~", sum.toFixed(4), "top:", labels[topIdx], "p=", probs[topIdx].toFixed(3));
                firstLogDone = true;
            }

            // Update UI
            result_display.textContent = labels
                .map((l, i) => [l, probs[i]])
                .sort((a, b) => b[1] - a[1])
                .map(([l, p]) => `${l}: ${(p * 100).toFixed(1)}%`)
                .join(", ");

            processed_holder.classList.remove("video-holder-unsafe", "video-holder-safe", "video-holder-empty");
            result_display_holder.classList.remove("video-holder-unsafe", "video-holder-safe", "video-holder-empty");

            if (labels[topIdx] === "unsafe") {
                processed_holder.classList.add("video-holder-unsafe");
                result_display_holder.classList.add("video-holder-unsafe");
            } else if (labels[topIdx] === "empty") {
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
                const fps = frames;
                document.getElementById("fps-counter-text").innerText = `~ ${fps} FPS`;
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

            lastInfer = ts;
        }

        requestAnimationFrame(loop);
    }

    requestAnimationFrame(loop);
}