const fileInput = document.getElementById("file-input");
const identifyBtn = document.getElementById("identify-btn");
const statusEl = document.getElementById("status");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultsList = document.getElementById("results-list");

let currentFile = null;
let currentImage = null;

function clearResults() {
  resultsList.innerHTML = "";
}

function renderStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#ff6b6b" : "#9ad4ff";
}

function resetCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawImageOnCanvas(img) {
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);
}

function drawDetections(detections) {
  if (!currentImage) return;
  drawImageOnCanvas(currentImage);

  clearResults();
  if (!Array.isArray(detections) || detections.length === 0) {
    renderStatus("No detections found.");
    return;
  }

  detections.forEach((det, index) => {
    const [x1, y1, x2, y2] = det.box ?? [];
    if ([x1, y1, x2, y2].some((value) => typeof value !== "number")) return;

    const width = x2 - x1;
    const height = y2 - y1;

    ctx.lineWidth = 3;
    ctx.strokeStyle = "#1fddff";
    ctx.strokeRect(x1, y1, width, height);

    const label = det.label ?? "unknown";
    const score = det.score != null ? det.score.toFixed(2) : "?";

    const text = `${label} (${score})`;
    ctx.font = "18px Arial";
    ctx.fillStyle = "rgba(31, 221, 255, 0.8)";
    const textWidth = ctx.measureText(text).width + 10;
    const textHeight = 22;
    const textX = x1;
    const textY = Math.max(y1 - textHeight, 0);

    ctx.fillRect(textX, textY, textWidth, textHeight);
    ctx.fillStyle = "#05121f";
    ctx.fillText(text, textX + 5, textY + textHeight - 6);

    const item = document.createElement("div");
    item.className = "result-item";
    item.textContent = `#${index + 1}: ${text}`;
    resultsList.appendChild(item);
  });

  renderStatus(`Found ${detections.length} detection(s).`);
}

async function handleIdentify() {
  if (!currentFile) {
    renderStatus("Please choose an image first.", true);
    return;
  }

  identifyBtn.disabled = true;
  renderStatus("Uploading…");

  try {
    const formData = new FormData();
    formData.append("image", currentFile);

    const response = await fetch("/identify", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || `HTTP ${response.status}`);
    }

    const detections = await response.json();
    drawDetections(detections);
  } catch (error) {
    console.error(error);
    renderStatus(`Error: ${error.message}`, true);
  } finally {
    identifyBtn.disabled = false;
  }
}

function loadSelectedFile(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    drawImageOnCanvas(img);
    renderStatus(`Loaded image (${img.width}×${img.height}).`);
    clearResults();
    URL.revokeObjectURL(url);
  };
  img.onerror = () => {
    renderStatus("Failed to load image.", true);
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files ?? [];
  if (!file) return;
  currentFile = file;
  loadSelectedFile(file);
});

identifyBtn.addEventListener("click", handleIdentify);

resetCanvas();
renderStatus("Choose an image to begin.");
