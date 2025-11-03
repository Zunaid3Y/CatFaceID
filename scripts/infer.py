import argparse
import json
import math
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


def process_detections(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    model: EmbedderModel,
    gallery_vectors: torch.Tensor,
    gallery_labels: List[str],
    threshold: float,
    device: torch.device,
    transform: transforms.Compose,
) -> List[dict]:
    detections = []
    original_height, original_width = image_bgr.shape[:2]

    with torch.no_grad():
        crops = []
        crop_boxes = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = expand_box(box, original_width, original_height)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_image = Image.fromarray(crop_rgb)
            tensor = transform(crop_image)
            crops.append(tensor)
            crop_boxes.append((x1, y1, x2, y2, float(score)))

        if not crops:
            return []

        batch = torch.stack(crops).to(device)
        embeddings = model.encode(batch)
        embeddings = F.normalize(embeddings, dim=1)

        similarities = torch.matmul(embeddings, gallery_vectors.T)
        for idx, (x1, y1, x2, y2, score) in enumerate(crop_boxes):
            sim_vector = similarities[idx]
            best_value, best_idx = torch.max(sim_vector, dim=0)
            best_value = float(best_value.item())
            best_label = gallery_labels[int(best_idx)]
            if best_value < threshold:
                final_label = "unknown"
            else:
                final_label = best_label

            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "score": score,
                    "similarity": best_value,
                    "label": final_label,
                }
            )

    return detections


def draw_detections(image_bgr: np.ndarray, detections: List[dict]) -> np.ndarray:
    output = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        similarity = det["similarity"]
        score = det["score"]
        text = f"{label} ({similarity:.2f})"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            text,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer pet identities from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--thresh", type=float, default=0.55, help="Similarity threshold for gallery match.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise SystemExit(f"Image not found at {image_path}.")

    if not DETECTOR_WEIGHTS.exists():
        raise SystemExit(f"YOLO weights not found at {DETECTOR_WEIGHTS}.")

    device = select_device()

    model = YOLO(str(DETECTOR_WEIGHTS))
    embedder = load_embedder(device)
    gallery_vectors, gallery_labels = load_gallery(device)
    if gallery_vectors.size(0) == 0:
        raise SystemExit("Gallery is empty.")

    transform = build_transform()

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise SystemExit(f"Failed to load image at {image_path}.")

    results = model(str(image_path), verbose=False)
    if not results:
        print(json.dumps({"detections": []}))
        return

    result = results[0]
    if result.boxes is None or result.boxes.xyxy is None or result.boxes.xyxy.shape[0] == 0:
        print(json.dumps({"detections": []}))
        return

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    detections = process_detections(
        image_bgr=image_bgr,
        boxes=boxes,
        scores=scores,
        model=embedder,
        gallery_vectors=gallery_vectors,
        gallery_labels=gallery_labels,
        threshold=args.thresh,
        device=device,
        transform=transform,
    )

    print(json.dumps({"detections": detections}))
    if not detections:
        return

    annotated = draw_detections(image_bgr, detections)
    cv2.imshow("Pet Identification", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
