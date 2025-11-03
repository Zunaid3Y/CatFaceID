import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


DETECTOR_WEIGHTS = Path("runs/detect/cat_head/weights/best.pt")
EMBEDDER_WEIGHTS = Path("models/embedder.pt")
GALLERY_VECTORS = Path("gallery/vectors.npy")
GALLERY_METADATA = Path("gallery/metadata.json")

MARGIN_RATIO = 0.08
EMBED_DIM = 256
IMAGE_SIZE = 224

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
        raise SystemExit("Gallery not found. Please run scripts/enroll_gallery.py first.")

    vectors = np.load(GALLERY_VECTORS)
    with GALLERY_METADATA.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if len(metadata) != vectors.shape[0]:
        raise SystemExit("Gallery metadata length does not match vectors.")

    pet_ids = [entry["pet_id"] for entry in metadata]
    gallery = torch.from_numpy(vectors).to(device=device, dtype=torch.float32)
    gallery = F.normalize(gallery, dim=1)
    return gallery, pet_ids


def load_embedder(device: torch.device) -> EmbedderModel:
    if not EMBEDDER_WEIGHTS.exists():
        raise SystemExit(f"Embedder weights not found at {EMBEDDER_WEIGHTS}.")
    checkpoint = torch.load(EMBEDDER_WEIGHTS, map_location=device)
    class_names = checkpoint.get("classes")
    if class_names is None:
        raise SystemExit("Embedder checkpoint missing 'classes' key.")
    if checkpoint.get("emb_dim") != EMBED_DIM:
        raise SystemExit(f"Expected embedding dim {EMBED_DIM}, found {checkpoint.get('emb_dim')}.")

    model = EmbedderModel(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def process_frame(
    frame_bgr: np.ndarray,
    detector: YOLO,
    embedder: EmbedderModel,
    gallery_vectors: torch.Tensor,
    gallery_labels: List[str],
    threshold: float,
    device: torch.device,
    transform: transforms.Compose,
) -> Tuple[np.ndarray, List[dict]]:
    results = detector(frame_bgr, verbose=False)
    if not results:
        return frame_bgr, []

    result = results[0]
    if result.boxes is None or result.boxes.xyxy is None or result.boxes.xyxy.shape[0] == 0:
        return frame_bgr, []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    original_height, original_width = frame_bgr.shape[:2]
    crops = []
    crop_boxes = []

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = expand_box(box, original_width, original_height)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_image = Image.fromarray(crop_rgb)
        tensor = transform(crop_image)
        crops.append(tensor)
        crop_boxes.append((x1, y1, x2, y2, float(score)))

    if not crops:
        return frame_bgr, []

    with torch.no_grad():
        batch = torch.stack(crops).to(device)
        embeddings = embedder.encode(batch)
        embeddings = F.normalize(embeddings, dim=1)
        similarities = torch.matmul(embeddings, gallery_vectors.T)

    detections: List[dict] = []
    for idx, (x1, y1, x2, y2, score) in enumerate(crop_boxes):
        sim_vector = similarities[idx]
        best_value, best_idx = torch.max(sim_vector, dim=0)
        best_value = float(best_value.item())
        best_label = gallery_labels[int(best_idx)]
        final_label = best_label if best_value >= threshold else "unknown"

        detections.append(
            {
                "box": (x1, y1, x2, y2),
                "label": final_label,
                "similarity": best_value,
                "score": score,
            }
        )

    annotated = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        similarity = det["similarity"]
        text = f"{label} ({similarity:.2f})"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated, detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live pet head identification demo.")
    parser.add_argument("--thresh", type=float, default=0.55, help="Similarity threshold for gallery match.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not DETECTOR_WEIGHTS.exists():
        raise SystemExit(f"YOLO weights not found at {DETECTOR_WEIGHTS}.")

    device = select_device()
    detector = YOLO(str(DETECTOR_WEIGHTS))
    embedder = load_embedder(device)
    gallery_vectors, gallery_labels = load_gallery(device)
    if gallery_vectors.size(0) == 0:
        raise SystemExit("Gallery is empty.")

    transform = build_transform()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Failed to open webcam.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to read frame from webcam.")
                break

            start_time = time.perf_counter()
            annotated_frame, detections = process_frame(
                frame,
                detector,
                embedder,
                gallery_vectors,
                gallery_labels,
                args.thresh,
                device,
                transform,
            )
            elapsed = time.perf_counter() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Pet ID Demo", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
