import cv2
import io
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


DETECTOR_WEIGHTS = Path("runs/detect/cat_head/weights/best.pt")
EMBEDDER_WEIGHTS = Path("models/embedder.pt")
GALLERY_VECTORS = Path("gallery/vectors.npy")
GALLERY_METADATA = Path("gallery/metadata.json")
STATIC_DIR = Path(__file__).parent / "static"

EMBED_DIM = 256
MARGIN_RATIO = 0.08
IMAGE_SIZE = 224
DEFAULT_THRESHOLD = 0.55

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ArcMarginProduct(nn.Module):
    """ArcFace-style cosine margin layer for compatibility with training checkpoint."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.eps = 1e-7

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings)
        weight = F.normalize(self.weight)
        cosine = F.linear(embeddings, weight).clamp(-1.0 + self.eps, 1.0 - self.eps)

        target_cosine = cosine.gather(1, labels.view(-1, 1))
        theta = torch.acos(target_cosine)
        target_logits = torch.cos(theta + self.m)

        logits = cosine.clone()
        logits.scatter_(1, labels.view(-1, 1), target_logits)
        logits *= self.s
        return logits


class EmbedderModel(nn.Module):
    """Encapsulates backbone and projection head used for embedding generation."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=False,
            num_classes=0,
        )
        backbone_dim = getattr(self.backbone, "num_features")
        self.projection = nn.Linear(backbone_dim, EMBED_DIM)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.margin_head = ArcMarginProduct(EMBED_DIM, num_classes)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        embeddings = F.normalize(self.projection(features), dim=1)
        return embeddings


def select_device() -> torch.device:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available() and mps_backend.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def expand_box(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
    margin_ratio: float = MARGIN_RATIO,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0, 0, width, height

    margin_x = w * margin_ratio / 2
    margin_y = h * margin_ratio / 2

    x1 = max(0, math.floor(x1 - margin_x))
    y1 = max(0, math.floor(y1 - margin_y))
    x2 = min(width, math.ceil(x2 + margin_x))
    y2 = min(height, math.ceil(y2 + margin_y))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE + 32),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_gallery(device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    if not GALLERY_VECTORS.exists() or not GALLERY_METADATA.exists():
        raise RuntimeError("Gallery not found. Ensure scripts/enroll_gallery.py has been executed.")

    vectors = np.load(GALLERY_VECTORS)
    with GALLERY_METADATA.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if len(metadata) != vectors.shape[0]:
        raise RuntimeError("Gallery metadata length does not match vectors array.")

    pet_ids = [entry["pet_id"] for entry in metadata]
    gallery = torch.from_numpy(vectors).to(device=device, dtype=torch.float32)
    gallery = F.normalize(gallery, dim=1)
    return gallery, pet_ids


def load_embedder(device: torch.device) -> EmbedderModel:
    if not EMBEDDER_WEIGHTS.exists():
        raise RuntimeError(f"Embedder weights not found at {EMBEDDER_WEIGHTS}.")
    checkpoint = torch.load(EMBEDDER_WEIGHTS, map_location=device)

    class_names = checkpoint.get("classes")
    if class_names is None:
        raise RuntimeError("Embedder checkpoint missing 'classes' key.")
    if checkpoint.get("emb_dim") != EMBED_DIM:
        raise RuntimeError(f"Expected embedding dim {EMBED_DIM}, found {checkpoint.get('emb_dim')}.")

    model = EmbedderModel(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_detector() -> YOLO:
    if not DETECTOR_WEIGHTS.exists():
        raise RuntimeError(f"YOLO weights not found at {DETECTOR_WEIGHTS}.")
    return YOLO(str(DETECTOR_WEIGHTS))


DEVICE = select_device()
TRANSFORM = build_transform()
DETECTOR = load_detector()
EMBEDDER = load_embedder(DEVICE)
GALLERY_VECTORS_TENSOR, GALLERY_LABELS = load_gallery(DEVICE)


app = FastAPI()

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Pet Identifier</title>
    <style>
      body { font-family: sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
      h1 { margin-bottom: 1rem; }
      form { margin-bottom: 2rem; }
      pre { background: #f4f4f4; padding: 12px; white-space: pre-wrap; word-wrap: break-word; border-radius: 4px; }
      #response { min-height: 120px; border: 1px solid #ccc; padding: 12px; border-radius: 4px; }
    </style>
  </head>
  <body>
    <h1>Pet Identifier</h1>
    <p>Upload an image to run detection and identification.</p>
    <form id="identify-form">
      <input type="file" id="image-input" name="image" accept="image/*" required />
      <label>
        Threshold:
        <input type="number" id="thresh-input" name="thresh" step="0.01" value="0.55" min="0" max="1" />
      </label>
      <button type="submit">Identify</button>
    </form>
    <h2>Response</h2>
    <div id="response">Submit an image to see results.</div>
    <script>
      const form = document.getElementById("identify-form");
      const responseEl = document.getElementById("response");
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const fileInput = document.getElementById("image-input");
        const threshInput = document.getElementById("thresh-input");

        if (!fileInput.files.length) {
          responseEl.textContent = "Please choose an image first.";
          return;
        }

        const formData = new FormData();
        formData.append("image", fileInput.files[0]);
        formData.append("thresh", threshInput.value || "0.55");

        responseEl.textContent = "Processing...";
        try {
          const res = await fetch("/identify", {
            method: "POST",
            body: formData
          });
          const text = await res.text();
          try {
            const json = JSON.parse(text);
            responseEl.textContent = JSON.stringify(json, null, 2);
          } catch (parseErr) {
            responseEl.textContent = text;
          }
        } catch (err) {
          responseEl.textContent = "Request failed: " + err;
        }
      });
    </script>
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTML_TEMPLATE


def prepare_image(image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            image_rgb = img.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    np_rgb = np.array(image_rgb)
    if np_rgb.ndim != 3 or np_rgb.shape[2] != 3:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    np_bgr = np_rgb[..., ::-1]
    return np_rgb, np_bgr


def process_detections(
    image_rgb: np.ndarray,
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    annotate: bool = False,
) -> Tuple[List[dict], np.ndarray | None]:
    detections: List[dict] = []
    height, width = image_bgr.shape[:2]
    annotated_image = image_bgr.copy() if annotate else None

    crops = []
    crop_meta = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = expand_box(box, width, height)
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_image = Image.fromarray(crop)
        tensor = TRANSFORM(crop_image)
        crops.append(tensor)
        crop_meta.append((x1, y1, x2, y2, float(score)))

    if not crops:
        return [], annotated_image

    with torch.no_grad():
        batch = torch.stack(crops).to(DEVICE)
        embeddings = EMBEDDER.encode(batch)
        embeddings = F.normalize(embeddings, dim=1)
        similarities = torch.matmul(embeddings, GALLERY_VECTORS_TENSOR.T)

    for idx, (x1, y1, x2, y2, score) in enumerate(crop_meta):
        sim_vector = similarities[idx]
        best_value, best_idx = torch.max(sim_vector, dim=0)
        best_value = float(best_value.item())
        best_label = GALLERY_LABELS[int(best_idx)]
        label = best_label if best_value >= threshold else "unknown"

        box_coords = [int(x1), int(y1), int(x2), int(y2)]
        detections.append(
            {
                "box": box_coords,
                "score": float(score),
                "label": label,
            }
        )

        if annotated_image is not None:
            x1_i, y1_i, x2_i, y2_i = box_coords
            text = f"{label} ({best_value:.2f})"
            cv2.rectangle(annotated_image, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
            cv2.putText(
                annotated_image,
                text,
                (x1_i, max(y1_i - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    return detections, annotated_image


def run_identification(
    image_rgb: np.ndarray,
    image_bgr: np.ndarray,
    threshold: float,
    annotate: bool = False,
) -> Tuple[List[dict], np.ndarray | None]:
    base_image = image_bgr.copy() if annotate else None
    results = DETECTOR(image_bgr, verbose=False)
    if not results:
        return [], base_image

    result = results[0]
    if result.boxes is None or result.boxes.xyxy is None or result.boxes.xyxy.shape[0] == 0:
        return [], base_image

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    detections, annotated = process_detections(
        image_rgb=image_rgb,
        image_bgr=image_bgr,
        boxes=boxes,
        scores=scores,
        threshold=threshold,
        annotate=annotate,
    )
    if annotate:
        return detections, annotated if annotated is not None else base_image
    return detections, None


@app.post("/identify")
async def identify(image: UploadFile = File(...), thresh: float = DEFAULT_THRESHOLD):
    if GALLERY_VECTORS_TENSOR.size(0) == 0:
        raise HTTPException(status_code=500, detail="Gallery is empty.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image_rgb, image_bgr = prepare_image(image_bytes)

    detections, _ = run_identification(image_rgb, image_bgr, thresh)
    return detections


@app.get("/overlay")
def overlay(p: str, thresh: float = DEFAULT_THRESHOLD):
    image_path = Path(p)
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found at {image_path}.")

    image_bytes = image_path.read_bytes()
    image_rgb, image_bgr = prepare_image(image_bytes)

    _, annotated = run_identification(image_rgb, image_bgr, thresh, annotate=True)
    if annotated is None:
        annotated = image_bgr

    success, buffer = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/ui")
def ui():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found.")
    return FileResponse(index_path)
