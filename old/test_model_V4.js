
const videoEl   = document.getElementById('video-holder');
const camToggle = document.getElementById('cam-toggle');
let currentStream = null;
let model;

// Modell laden
(async () => {
    model = await tf.loadLayersModel('../../models/tests/V1/model.json'); // Pfad ggf. anpassen
    console.log("✅ Modell geladen");
})();

// Klassifikation eines einzelnen Frames
async function classifyImage(imgData, canvas) {
    if (!model) return;
    const labels = ['empty', 'safe', 'unsafe'];

    const inputTensor = tf.browser.fromPixels(imgData)
        .resizeBilinear([180, 180])   // nur falls nötig, ansonsten Canvas muss korrekt sein
        .expandDims(0)
        .toFloat()
        .div(255.0);

    const prediction = model.predict(inputTensor);
    const result = await prediction.data();
    const maxIndex = result.indexOf(Math.max(...result));

    const processedDiv = document.getElementById('processed_video_holder');
    processedDiv.classList.remove("video-holder-empty", "video-holder-safe", "video-holder-unsafe");
    if (maxIndex === 0) {
        processedDiv.classList.add("video-holder-empty");
        console.log("Frame classified as empty");
    } else if (maxIndex === 2) {
        processedDiv.classList.add("video-holder-unsafe");
        console.log("Frame classified as unsafe");
    } else {
        processedDiv.classList.add("video-holder-safe");
        console.log("Frame classified as safe");
    }

    inputTensor.dispose();
    prediction.dispose();
}

// Loop zum Verarbeiten der Videoframes
function startFrameLoop(videoEl, callback) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 180;
    canvas.height = 180;

    const processedDiv = document.getElementById('processed_video_holder');
    processedDiv.innerHTML = '';
    processedDiv.appendChild(canvas);

    let running = true;

    async function loop() {
        if (!running) return;

        ctx.drawImage(videoEl, 0, 0, 180, 180);
        const imgData = ctx.getImageData(0, 0, 180, 180);
        await callback(imgData, canvas);

        requestAnimationFrame(loop);
    }

    loop();

    return () => { running = false; };
}

// Kamera starten
async function startCamera(){
    stopCamera();

    try {
        const constraints = {
            video: {
                facingMode: "user",
                width: { ideal: 1280 },
                height:{ ideal: 720 }
            },
            audio: false
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        currentStream = stream;
        videoEl.srcObject = stream;

        const stopLoop = startFrameLoop(videoEl, classifyImage); // ✅ hier korrekt aufrufen

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

// Kamera stoppen
function stopCamera(){
    if (currentStream){
        currentStream.getTracks().forEach(t => t.stop());
        currentStream = null;
        videoEl.srcObject = null;
    }
}

// Toggle Event
camToggle.addEventListener('change', () => {
    if (camToggle.checked) startCamera();
    else stopCamera();
});

// Aufräumen beim Verlassen
window.addEventListener('beforeunload', stopCamera);
