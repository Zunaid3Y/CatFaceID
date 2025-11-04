import os, sys
THIS_DIR = os.path.dirname(__file__)
PARENT   = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
import argparse
import os
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

# Try to import faiss (faiss-cpu installs as module 'faiss')
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    HAS_FAISS = False

try:
    from scripts.embedder_resnet import PetEmbedder, get_transforms
except ImportError:
    from embedder_resnet import PetEmbedder, get_transforms

def build_index(gallery_vectors: np.ndarray):
    """Build a cosine-similarity FAISS index over L2-normalized vectors.
    Returns a callable search(query_vecs, topk) -> (scores, idxs).
    """
    if not HAS_FAISS:
        raise RuntimeError("FAISS is not available; install faiss-cpu to use --faiss")
    if gallery_vectors.ndim != 2:
        raise ValueError("gallery_vectors must be 2D (n, d)")
    # Ensure float32 and L2-normalized
    V = gallery_vectors.astype(np.float32)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = V / norms
    dim = V.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(V)

    def _search(Q: np.ndarray, topk: int):
        if Q.ndim == 1:
            Q = Q[None, :]
        Q = Q.astype(np.float32)
        qn = np.linalg.norm(Q, axis=1, keepdims=True)
        qn[qn == 0] = 1.0
        Q = Q / qn
        scores, idxs = index.search(Q, topk)
        return scores, idxs

    return _search

from scripts.landmark_align import try_align_from_cat
from scripts.quality import pass_quality

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
# Defaults via env (overridable by CLI)
ENV_THRESHOLD = float(os.getenv("CATFACEID_THRESHOLD", os.getenv("REJECT_THRESHOLD", "0.72")))
ENV_MARGIN = float(os.getenv("CATFACEID_MARGIN", "0.03"))
ENV_WEIGHTS = os.getenv("CATFACEID_WEIGHTS", "runs/pet_head/weights/best.pt")
ENV_GALLERY = os.getenv("CATFACEID_GALLERY", os.getenv("GALLERY_DIR", "gallery"))
ENV_REJECT_LABEL = os.getenv("CATFACEID_REJECT_LABEL", "unknown")

TOPK = 5                 # top-k results to report

def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_gallery(gallery_dir: Path, device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    vec_p = Path(gallery_dir) / "vectors.npy"
    lab_p = Path(gallery_dir) / "labels.json"
    if not vec_p.exists() or not lab_p.exists():
        raise FileNotFoundError("Gallery files not found. Run scripts/enroll_gallery.py first.")
    vectors = torch.from_numpy(np.load(vec_p)).to(device=device, dtype=torch.float32)
    labels = json.loads(lab_p.read_text(encoding="utf-8"))
    if vectors.ndim != 2 or vectors.size(0) != len(labels):
        raise ValueError("Gallery vectors and labels are mismatched.")
    vectors = F.normalize(vectors, p=2, dim=1)
    return vectors, labels


def get_weights_path(cli_weights: Optional[str] = None) -> Path:
    """Resolve detector weights path with CLI/env/fallbacks.
    Order: CLI --weights → CATFACEID_WEIGHTS → runs/pet_head/...best.pt → last.pt → yolov8n.pt
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
        f"Detector weights not found. Tried: "
        f"CLI={cli_weights!r}, ENV={env_w!r}, fallbacks=['runs/pet_head/weights/best.pt','runs/pet_head/weights/last.pt','yolov8n.pt']"
    )


def load_detector(weights_path: Path, device: torch.device) -> YOLO:
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found at {weights_path}")
    model = YOLO(str(weights_path))
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

def best_detection_bgr(detector: YOLO, bgr: np.ndarray, device: torch.device) -> Optional[Tuple[int, int, int, int]]:
    results = detector.predict(
        source=bgr,
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

    used_faiss = faiss_index is not None
    if used_faiss:
        max_k = min(TOPK, len(gallery_labels))
        query = embedding.detach().cpu().numpy()[np.newaxis, :]
        scores, idxs = faiss_index(query, topk=max_k)
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

    # Decision with threshold + margin
    if top1_score < args.threshold:
        label_out = args.reject_label
    elif gap < args.margin:
        label_out = f"{top1_label}?"
    else:
        label_out = top1_label

    print(f"{path}: {label_out} ({top1_score:.3f})")
    print(f"  Top-{actual_topk}: {topk_info}")

    result["prediction"] = label_out
    result["score"] = top1_score
    result["label"] = label_out
    result["top3"] = [(gallery_labels[int(i)], float(v)) for i, v in zip(indices.tolist(), values.tolist())]
    result["threshold"] = float(args.threshold)
    result["margin"] = float(args.margin)
    result["used_faiss"] = bool(used_faiss)
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
        "--threshold",
        type=float,
        default=ENV_THRESHOLD,
        help="Acceptance threshold on top-1 similarity (env CATFACEID_THRESHOLD).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=ENV_MARGIN,
        help="Ambiguity margin: if top1-top2 < margin, mark as uncertain (env CATFACEID_MARGIN).",
    )
    parser.add_argument(
        "--faiss",
        action="store_true",
        help="Use FAISS inner-product index for nearest-neighbour search.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=ENV_WEIGHTS,
        help="Path to YOLO detector weights (env CATFACEID_WEIGHTS).",
    )
    parser.add_argument(
        "--gallery",
        type=str,
        default=ENV_GALLERY,
        help="Gallery directory path (env CATFACEID_GALLERY).",
    )
    parser.add_argument(
        "--reject-label",
        type=str,
        default=ENV_REJECT_LABEL,
        help="Label to emit when below threshold (default 'unknown').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    device = get_device()

    try:
        gallery_vectors, gallery_labels = load_gallery(Path(args.gallery), device)
    except Exception as exc:
        print(f"Failed to load gallery: {exc}")
        sys.exit(1)

    faiss_index = None
    if args.faiss:
        if not HAS_FAISS:
            print("[warn] --faiss requested but FAISS not available; falling back to torch.")
        else:
            try:
                gallery_np = gallery_vectors.detach().cpu().numpy()
                faiss_index = build_index(gallery_np)
            except Exception as exc:
                print(f"[warn] Failed to build FAISS index: {exc}; falling back to torch.")
                faiss_index = None

    try:
        resolved_weights = get_weights_path(args.weights)
        detector = load_detector(resolved_weights, device)
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

# --------- Programmatic API ---------

_CTX = {
    "built": False,
    "device": None,
    "detector": None,
    "embedder": None,
    "transform": None,
    "gallery_vectors": None,
    "gallery_labels": None,
    "faiss_index": None,
    "threshold": ENV_THRESHOLD,
    "margin": ENV_MARGIN,
    "reject_label": ENV_REJECT_LABEL,
    "used_faiss": False,
}

def _ensure_ctx():
    if _CTX["built"]:
        return _CTX
    device = get_device()
    gallery_dir = Path(ENV_GALLERY)
    try:
        weights = get_weights_path(ENV_WEIGHTS)
    except Exception as e:
        raise RuntimeError(str(e))
    try:
        gv, gl = load_gallery(gallery_dir, device)
    except Exception as e:
        raise RuntimeError(f"identify_bgr: failed to load gallery from {gallery_dir}: {e}")
    try:
        det = load_detector(weights, device)
    except Exception as e:
        raise RuntimeError(f"identify_bgr: failed to load detector at {weights}: {e}")
    emb = PetEmbedder().to(device)
    emb.eval()
    tfm = get_transforms()

    # Optional FAISS via env CATFACEID_FAISS=true
    faiss_idx = None
    used_faiss = False
    if os.getenv("CATFACEID_FAISS", "false").lower() in {"1","true","yes"} and HAS_FAISS:
        try:
            faiss_idx = build_index(gv.detach().cpu().numpy())
            used_faiss = True
        except Exception:
            faiss_idx = None
            used_faiss = False

    _CTX.update({
        "built": True,
        "device": device,
        "detector": det,
        "embedder": emb,
        "transform": tfm,
        "gallery_vectors": gv,
        "gallery_labels": gl,
        "faiss_index": faiss_idx,
        "used_faiss": used_faiss,
    })
    return _CTX

def identify_bgr(bgr: np.ndarray) -> dict:
    """Programmatic identification API.
    Uses env-based defaults for weights/gallery/threshold/margin unless CLI overrides are in effect.
    Returns: {label, score, top3, threshold, margin, used_faiss}
    """
    ctx = _ensure_ctx()
    device = ctx["device"]
    det = ctx["detector"]
    emb = ctx["embedder"]
    tfm = ctx["transform"]
    gv = ctx["gallery_vectors"]
    gl = ctx["gallery_labels"]
    faiss_idx = ctx["faiss_index"]
    threshold = float(_CTX["threshold"]) if _CTX["threshold"] is not None else ENV_THRESHOLD
    margin = float(_CTX["margin"]) if _CTX["margin"] is not None else ENV_MARGIN
    reject_label = str(_CTX["reject_label"]) if _CTX["reject_label"] else ENV_REJECT_LABEL

    if bgr is None or bgr.size == 0:
        return {"label": reject_label, "score": 0.0, "top3": [], "threshold": threshold, "margin": margin, "used_faiss": bool(faiss_idx is not None)}

    box = best_detection_bgr(det, bgr, device)
    if box is None:
        return {"label": reject_label, "score": 0.0, "top3": [], "threshold": threshold, "margin": margin, "used_faiss": bool(faiss_idx is not None)}
    x1,y1,x2,y2 = box
    h,w = bgr.shape[:2]
    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w)); y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return {"label": reject_label, "score": 0.0, "top3": [], "threshold": threshold, "margin": margin, "used_faiss": bool(faiss_idx is not None)}
    crop = bgr[y1:y2, x1:x2]
    if not pass_quality(crop):
        return {"label": reject_label, "score": 0.0, "top3": [], "threshold": threshold, "margin": margin, "used_faiss": bool(faiss_idx is not None)}

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    z = compute_embedding(emb, tfm, pil, device)

    # top-k via faiss or torch
    if faiss_idx is not None and HAS_FAISS:
        max_k = min(TOPK, len(gl))
        scores, idxs = faiss_idx(z.detach().cpu().numpy()[np.newaxis,:], topk=max_k)
        vals = scores[0, :max_k]
        inds = idxs[0, :max_k]
        values = vals
        indices = inds
        used_faiss = True
    else:
        sims = torch.matmul(gv, z)
        k = min(TOPK, sims.numel())
        vals, inds = torch.topk(sims, k=k)
        values = vals.detach().cpu().numpy()
        indices = inds.detach().cpu().numpy()
        used_faiss = False

    if values.size == 0:
        return {"label": reject_label, "score": 0.0, "top3": [], "threshold": threshold, "margin": margin, "used_faiss": used_faiss}

    top1_score = float(values[0])
    top1_label = gl[int(indices[0])]
    top2_score = float(values[1]) if values.size > 1 else float('-inf')
    gap = top1_score - top2_score if values.size > 1 else float('inf')
    if top1_score < threshold:
        label_out = reject_label
    elif gap < margin:
        label_out = f"{top1_label}?"
    else:
        label_out = top1_label

    top3 = [[gl[int(i)], float(s)] for i,s in zip(indices, values)]
    return {"label": label_out, "score": top1_score, "top3": top3, "threshold": threshold, "margin": margin, "used_faiss": used_faiss}
