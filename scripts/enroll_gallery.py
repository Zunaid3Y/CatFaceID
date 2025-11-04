import os, sys
THIS_DIR = os.path.dirname(__file__)
PARENT   = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

try:
    from scripts.embedder_resnet import PetEmbedder, get_transforms
except ImportError:
    from embedder_resnet import PetEmbedder, get_transforms

from ultralytics import YOLO

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
def get_weights_path(cli_weights: str | None) -> Path:
    """Resolve YOLO detector weights with CLI/env/fallbacks.
    Order: CLI --weights → CATFACEID_WEIGHTS → runs/pet_head/best.pt → last.pt → yolov8n.pt
    """
    candidates = []
    if cli_weights:
        candidates.append(Path(cli_weights))
    env_w = os.getenv("CATFACEID_WEIGHTS")
    if env_w:
        candidates.append(Path(env_w))
    candidates += [
        Path("runs/pet_head/weights/best.pt"),
        Path("runs/pet_head/weights/last.pt"),
        Path("yolov8n.pt"),
    ]
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    raise RuntimeError(
        f"Detector weights not found. Tried: CLI={cli_weights!r}, ENV={env_w!r}, "
        "fallbacks=['runs/pet_head/weights/best.pt','runs/pet_head/weights/last.pt','yolov8n.pt']"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enroll gallery prototypes from face crops.")
    p.add_argument("--data", type=str, default=os.getenv("DATA_DIR", "data") + "/faceid", help="Input folder: data/faceid")
    p.add_argument("--gallery", type=str, default=os.getenv("GALLERY_DIR", "gallery"), help="Output gallery dir")
    p.add_argument("--weights", type=str, default=os.getenv("CATFACEID_WEIGHTS", "runs/pet_head/weights/best.pt"), help="YOLO detector weights")
    p.add_argument("--limit-per-pet", type=int, default=None, help="Cap number of images per pet")
    return p.parse_args()


def list_image_paths(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix in SUPPORTED_EXTS
    )

def main() -> None:
    args = parse_args()
    faceid_dir = Path(args.data)
    gallery_dir = Path(args.gallery)
    try:
        weights_path = get_weights_path(args.weights)
    except Exception as e:
        raise RuntimeError(str(e))
    limit = args.limit_per_pet

    if not faceid_dir.exists():
        raise FileNotFoundError(f"Face gallery directory not found: {faceid_dir}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    embedder = PetEmbedder().to(device)
    embedder.eval()
    transform = get_transforms()

    detector = YOLO(str(weights_path))
    detector.to("mps" if torch.backends.mps.is_available() else "cpu")

    pet_labels: List[str] = []
    pet_vectors: List[np.ndarray] = []

    total_used = 0
    total_images = 0
    total_skipped_no_det = 0

    for pet_dir in sorted(p for p in faceid_dir.iterdir() if p.is_dir()):
        image_paths = list_image_paths(pet_dir)
        if limit is not None:
            image_paths = image_paths[: max(0, int(limit))]
        if not image_paths:
            continue

        embeddings = []
        used = 0
        skipped_no_det = 0

        for img_path in image_paths:
            total_images += 1
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            # Detect best head box
            res = detector.predict(source=bgr, device=("mps" if torch.backends.mps.is_available() else "cpu"), imgsz=640, conf=0.25, max_det=1, verbose=False)
            if not res or res[0].boxes is None or len(res[0].boxes) == 0:
                skipped_no_det += 1
                continue
            x1, y1, x2, y2 = map(int, res[0].boxes.xyxy[0].tolist())
            h, w = bgr.shape[:2]
            x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w)); y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                skipped_no_det += 1
                continue
            crop = bgr[y1:y2, x1:x2]
            # Embed
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = embedder(tensor).squeeze(0)
            embeddings.append(emb)
            used += 1

        if not embeddings:
            total_skipped_no_det += skipped_no_det
            continue

        pet_tensor = torch.stack(embeddings, dim=0)
        proto = F.normalize(pet_tensor.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
        pet_labels.append(pet_dir.name)
        pet_vectors.append(proto.cpu().numpy().astype(np.float32))
        total_used += used

    if not pet_vectors:
        raise RuntimeError("No embeddings generated; ensure data/faceid/* has images with detectable heads.")

    gallery_dir.mkdir(parents=True, exist_ok=True)
    np.save(gallery_dir / "vectors.npy", np.stack(pet_vectors, axis=0))
    with open(gallery_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(pet_labels, f, indent=2)

    print(f"Enrolled pets: {len(pet_labels)} | Total images processed: {total_images} | Skipped (no detection): {total_skipped_no_det}")


if __name__ == "__main__":
    main()
