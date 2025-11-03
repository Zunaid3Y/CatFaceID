import os, sys
THIS_DIR = os.path.dirname(__file__)
PARENT   = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

try:
    from scripts.embedder_resnet import PetEmbedder, get_transforms
except ImportError:
    from embedder_resnet import PetEmbedder, get_transforms

try:
    from scripts.faiss_index import build_faiss, search_faiss
except ImportError:
    try:
        from faiss_index import build_faiss, search_faiss
    except ImportError:
        build_faiss = search_faiss = None

from scripts.landmark_align import try_align_from_cat
from scripts.quality import pass_quality

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
GALLERY_VECTORS = Path("gallery") / "vectors.npy"
GALLERY_LABELS = Path("gallery") / "labels.json"
YOLO_WEIGHTS = Path("runs/cat_head/weights/best.pt")
REJECT_THRESHOLD = 0.75  # try 0.70–0.80 and tune
TOPK = 5                 # top-k results to report

def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_gallery(device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    if not GALLERY_VECTORS.exists() or not GALLERY_LABELS.exists():
        raise FileNotFoundError("Gallery files not found. Run scripts/enroll_gallery.py first.")
    vectors = torch.from_numpy(np.load(GALLERY_VECTORS)).to(device=device, dtype=torch.float32)
    labels = json.loads(GALLERY_LABELS.read_text(encoding="utf-8"))
    if vectors.ndim != 2 or vectors.size(0) != len(labels):
        raise ValueError("Gallery vectors and labels are mismatched.")
    vectors = F.normalize(vectors, p=2, dim=1)
    return vectors, labels


def load_detector(device: torch.device) -> YOLO:
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLO weights not found at {YOLO_WEIGHTS}")
    model = YOLO(str(YOLO_WEIGHTS))
    model.to(str(device))
    return model


def best_detection(detector: YOLO, image_path: Path, device: torch.device) -> Optional[Tuple[int, int, int, int]]:
    results = detector.predict(
        source=str(image_path),
        device=str(device),
        imgsz=640,
        conf=0.25,
        max_det=5,
        verbose=False,
    )
    if not results:
        return None
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    idx = torch.argmax(result.boxes.conf)
    x1, y1, x2, y2 = result.boxes.xyxy[idx].tolist()
    return int(x1), int(y1), int(x2), int(y2)


def compute_embedding(
    embedder: PetEmbedder,
    transform,
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = embedder(tensor).squeeze(0)
    return F.normalize(embedding, p=2, dim=0)


def identify_image(
    path: Path,
    detector: YOLO,
    embedder: PetEmbedder,
    transform,
    gallery_vectors: torch.Tensor,
    gallery_labels: List[str],
    device: torch.device,
    args: argparse.Namespace,
    faiss_index=None,
) -> dict:
    """Identify a single image and return prediction details."""
    result = {"path": str(path), "prediction": None, "score": None}

    bgr = cv2.imread(str(path))
    if bgr is None:
        print(f"{path}: failed to read image")
        result["prediction"] = "READ_ERROR"
        return result

    aligned = try_align_from_cat(str(path))

    crop_bgr = aligned
    if crop_bgr is None:
        try:
            box = best_detection(detector, path, device)
        except Exception as exc:
            print(f"{path}: detection failed ({exc})")
            result["prediction"] = "DETECTION_ERROR"
            return result

        if box is None:
            print(f"{path}: no pet head detected")
            result["prediction"] = "NO_DETECTION"
            return result

        x1, y1, x2, y2 = box
        h, w = bgr.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            print(f"{path}: invalid bounding box {box}")
            result["prediction"] = "INVALID_BOX"
            return result
        crop_bgr = bgr[y1:y2, x1:x2]

    if crop_bgr is None or crop_bgr.size == 0:
        print(f"{path}: failed to obtain crop")
        result["prediction"] = "EMPTY_CROP"
        return result

    try:
        quality_ok = pass_quality(crop_bgr)
    except ValueError as exc:
        print(f"{path}: REJECT_LOW_QUALITY ({exc})")
        result["prediction"] = "REJECT_LOW_QUALITY"
        return result

    if not quality_ok:
        print(f"{path}: REJECT_LOW_QUALITY")
        result["prediction"] = "REJECT_LOW_QUALITY"
        return result

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)

    embedding = compute_embedding(embedder, transform, crop_pil, device)

    if faiss_index is not None:
        max_k = min(TOPK, len(gallery_labels))
        query = embedding.detach().cpu().numpy()[np.newaxis, :]
        scores, idxs = search_faiss(faiss_index, query, topk=max_k)
        values = torch.from_numpy(scores[0, :max_k])
        indices = torch.from_numpy(idxs[0, :max_k]).long()
    else:
        sims = torch.matmul(gallery_vectors, embedding)
        max_k = min(TOPK, sims.numel())
        values, indices = torch.topk(sims, k=max_k)

    actual_topk = values.numel()
    if actual_topk == 0:
        print(f"{path}: no gallery candidates found")
        result["prediction"] = "NO_CANDIDATES"
        return result

    top1_idx = int(indices[0].item())
    top1_label = gallery_labels[top1_idx]
    top1_score = values[0].item()
    top2_score = values[1].item() if actual_topk > 1 else float("-inf")
    gap = top1_score - top2_score if actual_topk > 1 else float("inf")

    topk_info = ", ".join(
        f"{gallery_labels[idx]}:{score:.3f}" for idx, score in zip(indices.tolist(), values.tolist())
    )

    if top1_score < args.reject or gap < args.margin:
        print(f"{path}: UNKNOWN (top-1 similarity {top1_score:.3f})")
        print(f"  Top-{actual_topk}: {topk_info}")
        result["prediction"] = "UNKNOWN"
        result["score"] = top1_score
        return result

    print(f"{path}: {top1_label} ({top1_score:.3f})")
    print(f"  Top-{actual_topk}: {topk_info}")

    result["prediction"] = top1_label
    result["score"] = top1_score
    return result


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if child.is_file() and child.suffix in SUPPORTED_EXTS:
                yield child
    elif path.is_file() and path.suffix in SUPPORTED_EXTS:
        yield path
    else:
        raise FileNotFoundError(f"No supported images found at {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify pet faces from an image or folder.")
    parser.add_argument("input", type=str, help="Image file or folder containing images.")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate predictions by majority vote when processing a folder.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Minimum similarity margin between top-1 and top-2 to accept a prediction.",
    )
    parser.add_argument(
        "--reject",
        type=float,
        default=REJECT_THRESHOLD,
        help="Open-set rejection threshold on top-1 similarity.",
    )
    parser.add_argument(
        "--faiss",
        action="store_true",
        help="Use FAISS inner-product index for nearest-neighbour search.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    device = get_device()

    try:
        gallery_vectors, gallery_labels = load_gallery(device)
    except Exception as exc:
        print(f"Failed to load gallery: {exc}")
        sys.exit(1)

    faiss_index = None
    if args.faiss:
        if build_faiss is None or search_faiss is None:
            print("FAISS support not available. Install faiss-cpu to use --faiss.")
            sys.exit(1)
        try:
            gallery_np = gallery_vectors.detach().cpu().numpy()
            faiss_index = build_faiss(gallery_np)
        except Exception as exc:
            print(f"Failed to build FAISS index: {exc}")
            sys.exit(1)

    try:
        detector = load_detector(device)
    except Exception as exc:
        print(f"Failed to load detector: {exc}")
        sys.exit(1)

    embedder = PetEmbedder().to(device)
    embedder.eval()
    transform = get_transforms()

    try:
        images = list(iter_images(input_path))
    except Exception as exc:
        print(exc)
        sys.exit(1)

    if not images:
        print(f"No images to process in {input_path}")
        sys.exit(1)

    results = []
    total_processed = 0
    for img_path in images:
        total_processed += 1
        res = identify_image(
            img_path,
            detector,
            embedder,
            transform,
            gallery_vectors,
            gallery_labels,
            device,
            args,
            faiss_index=faiss_index,
        )
        if res is not None:
            results.append(res)

    if args.aggregate and input_path.is_dir():
        invalid_labels = {
            "UNKNOWN",
            "REJECT_LOW_QUALITY",
            None,
            "READ_ERROR",
            "DETECTION_ERROR",
            "NO_DETECTION",
            "INVALID_BOX",
            "EMPTY_CROP",
            "NO_CANDIDATES",
        }
        valid_results = [r for r in results if r["prediction"] not in invalid_labels and r["score"] is not None]
        if not valid_results:
            print("AGGREGATE: UNKNOWN")
        else:
            label_counts = Counter(r["prediction"] for r in valid_results)
            max_votes = max(label_counts.values())
            candidates = [label for label, count in label_counts.items() if count == max_votes]
            mean_scores = {
                label: float(np.mean([r["score"] for r in valid_results if r["prediction"] == label]))
                for label in candidates
            }
            winner = max(candidates, key=lambda lbl: mean_scores[lbl])
            total_votes = len(valid_results)
            mean_score = mean_scores[winner]
            print(f"AGGREGATE: {winner} (mean={mean_score:.3f}, votes={label_counts[winner]}/{total_processed})")


if __name__ == "__main__":
    main()
