// Element references
const inputEl = document.getElementById("input");
const button = document.getElementById("submit-btn");
const resultEl = document.getElementById("result");
const loaderWrap = document.getElementById("loader-wrap");
const mainApp = document.getElementById("mainAPP");
const historyListEl = document.getElementById("history-list");
const exampleChips = document.querySelectorAll(".chip");
const meterEl = document.getElementById("meter");
const meterFillEl = document.getElementById("meter-fill");

let isModelLoaded = false;
let model;
let word2index;
const history = [];
const MAX_HISTORY = 8;

// Data preprocessing parameters
const maxlen = 20;
const padding = "post";
const truncating = "post";

function detectWebGLContext() {
  const canvas = document.createElement("canvas");
  const gl =
    canvas.getContext("webgl") || canvas.getContext("experimental-webgl");

  if (gl && gl instanceof WebGLRenderingContext) {
    init();
  } else {
    showResult(
      "Your browser or device does not support WebGL, which is required to run this demo.",
      "error",
    );
  }
}

function padSequence(
  sequences,
  maxLen,
  padding = "post",
  truncating = "post",
  padValue = 0,
) {
  return sequences.map((seq) => {
    if (seq.length > maxLen) {
      if (truncating === "pre") {
        seq.splice(0, seq.length - maxLen);
      } else {
        seq.splice(maxLen, seq.length - maxLen);
      }
    }

    if (seq.length < maxLen) {
      const pad = new Array(maxLen - seq.length).fill(padValue);
      seq = padding === "pre" ? pad.concat(seq) : seq.concat(pad);
    }

    return seq;
  });
}

function predict(inputText) {
  // Turn the review into a sequence of token indices
  const sequence = inputText.map((word) => {
    const indexed = word2index[word];
    return indexed === undefined ? 1 : indexed; // 1 = out-of-vocabulary token
  });

  const paddedSequence = padSequence([sequence], maxlen, padding, truncating);

  return tf.tidy(() => {
    const inputTensor = tf.tensor2d(paddedSequence, [1, maxlen]);
    const result = model.predict(inputTensor);
    return result.dataSync()[0];
  });
}

function showResult(message, type) {
  resultEl.textContent = message;
  resultEl.className = "result" + (type ? " " + type : "");
}

function showMeter(score, label) {
  meterFillEl.style.width = `${Math.round(score * 100)}%`;
  meterFillEl.className = "meter-fill " + label.toLowerCase();
  meterEl.hidden = false;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function renderHistory() {
  if (history.length === 0) {
    historyListEl.innerHTML =
      '<li class="history-empty">No reviews analyzed yet — try one above!</li>';
    return;
  }

  historyListEl.innerHTML = history
    .map(
      (entry) => `
        <li class="history-item">
          <div class="history-text">"${escapeHtml(entry.text)}"</div>
          <div class="history-meta">
            <span class="badge ${entry.label.toLowerCase()}">${entry.label}</span>
            <span class="history-score">${entry.score.toFixed(3)}</span>
            <span class="history-time">${entry.time}</span>
          </div>
        </li>
      `,
    )
    .join("");
}

function addHistoryEntry(text, score, label) {
  history.unshift({
    text,
    score,
    label,
    time: new Date().toLocaleTimeString(),
  });
  history.length = Math.min(history.length, MAX_HISTORY);
  renderHistory();
}

function onClick() {
  if (!isModelLoaded) {
    showResult("Model is still loading, please wait a moment.", "warning");
    return;
  }

  const review = inputEl.value.trim();
  if (review === "") {
    showResult("Review can't be empty.", "warning");
    inputEl.focus();
    return;
  }

  const inputText = review.toLowerCase().split(/\s+/);
  const score = predict(inputText);
  const label = score > 0.5 ? "Positive" : "Negative";

  showResult(
    `${label} review (score: ${score.toFixed(3)})`,
    label.toLowerCase(),
  );
  showMeter(score, label);
  addHistoryEntry(review, score, label);
}

async function init() {
  try {
    // Relative paths so the demo works both locally (via a local server) and on GitHub Pages
    model = await tf.loadLayersModel("tfjs_model_sentiment/model.json");

    const word_indexjson = await fetch("notebooks/word_index.json");
    word2index = await word_indexjson.json();

    isModelLoaded = true;
    loaderWrap.style.display = "none";
    mainApp.style.display = "block";

    console.log("Model & metadata loaded successfully");
  } catch (err) {
    showResult(
      "Failed to load the model. Please refresh and try again.",
      "error",
    );
    console.error(err);
  }
}

button.addEventListener("click", onClick);
inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    onClick();
  }
});

exampleChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    inputEl.value = chip.dataset.example;
    inputEl.focus();
    onClick();
  });
});

detectWebGLContext();
