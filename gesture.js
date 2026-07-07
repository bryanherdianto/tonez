// Element references
const canvas = document.getElementById("gesture-canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clear-btn");
const resultEl = document.getElementById("result");
const loaderWrap = document.getElementById("loader-wrap");
const mainApp = document.getElementById("mainAPP");

let isModelLoaded = false;
let model;
let CLASSES = [];
let isDrawing = false;
let hasDrawn = false;

const IMG_SIZE = 28;
const CONFIDENCE_THRESHOLD = 0.6;

// Offscreen canvas used to downsample the visible 280x280 drawing to 28x28
const offscreen = document.createElement("canvas");
offscreen.width = IMG_SIZE;
offscreen.height = IMG_SIZE;
const offscreenCtx = offscreen.getContext("2d");

function detectWebGLContext() {
  const testCanvas = document.createElement("canvas");
  const gl =
    testCanvas.getContext("webgl") ||
    testCanvas.getContext("experimental-webgl");

  if (gl && gl instanceof WebGLRenderingContext) {
    init();
  } else {
    showResult(
      "Your browser or device does not support WebGL, which is required to run this demo.",
      "error",
    );
  }
}

function resetCanvas() {
  // White strokes on black background, matching the synthetic training data
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "white";
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  hasDrawn = false;
}

function getPos(event) {
  const rect = canvas.getBoundingClientRect();
  const source = event.touches ? event.touches[0] : event;
  return {
    x: ((source.clientX - rect.left) / rect.width) * canvas.width,
    y: ((source.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function startStroke(event) {
  event.preventDefault();
  isDrawing = true;
  hasDrawn = true;
  const { x, y } = getPos(event);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function moveStroke(event) {
  if (!isDrawing) return;
  event.preventDefault();
  const { x, y } = getPos(event);
  ctx.lineTo(x, y);
  ctx.stroke();
}

function endStroke(event) {
  if (!isDrawing) return;
  event.preventDefault();
  isDrawing = false;
  predict();
}

function preprocessCanvas() {
  offscreenCtx.drawImage(
    canvas,
    0,
    0,
    canvas.width,
    canvas.height,
    0,
    0,
    IMG_SIZE,
    IMG_SIZE,
  );
  const { data } = offscreenCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);

  return tf.tidy(() => {
    const grayscale = new Float32Array(IMG_SIZE * IMG_SIZE);
    for (let i = 0; i < grayscale.length; i++) {
      grayscale[i] = data[i * 4] / 255; // red channel is enough for a grayscale drawing
    }
    return tf.tensor4d(grayscale, [1, IMG_SIZE, IMG_SIZE, 1]);
  });
}

function predict() {
  if (!isModelLoaded || !hasDrawn) return;

  const { label, confidence } = tf.tidy(() => {
    const inputTensor = preprocessCanvas();
    const prediction = model.predict(inputTensor);
    const probs = prediction.dataSync();
    const classIndex = prediction.argMax(1).dataSync()[0];
    return { label: CLASSES[classIndex], confidence: probs[classIndex] };
  });

  if (confidence >= CONFIDENCE_THRESHOLD) {
    showResult(
      `Detected: ${label} (${Math.round(confidence * 100)}% confidence)`,
      "detected",
    );
  } else {
    showResult(
      `Not sure — closest guess was ${label} (${Math.round(confidence * 100)}%)`,
      "warning",
    );
  }
}

function showResult(message, type) {
  resultEl.textContent = message;
  resultEl.className = "result" + (type ? " " + type : "");
}

function onClear() {
  resetCanvas();
  showResult("Draw a symbol...");
}

async function init() {
  try {
    // Relative paths so the demo works both locally (via a local server) and on GitHub Pages
    model = await tf.loadLayersModel("tfjs_model_gesture/model.json");

    const labelsJson = await fetch("notebooks/gesture_labels.json");
    const labels = await labelsJson.json();
    CLASSES = labels.classes;

    resetCanvas();
    isModelLoaded = true;
    loaderWrap.style.display = "none";
    mainApp.style.display = "block";

    console.log("Gesture model & metadata loaded successfully");
  } catch (err) {
    showResult(
      "Failed to load the model. Please refresh and try again.",
      "error",
    );
    console.error(err);
  }
}

canvas.addEventListener("mousedown", startStroke);
canvas.addEventListener("mousemove", moveStroke);
window.addEventListener("mouseup", endStroke);

canvas.addEventListener("touchstart", startStroke);
canvas.addEventListener("touchmove", moveStroke);
canvas.addEventListener("touchend", endStroke);

clearBtn.addEventListener("click", onClear);

detectWebGLContext();
