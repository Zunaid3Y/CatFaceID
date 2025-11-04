"""Threshold sweep for CatFaceID (closed/open set).

Usage:
  python3 scripts/sweep_threshold.py --val-dir data/faceid --topk 1 \
      --thr-min 0.50 --thr-max 0.95 --thr-step 0.01 [--faiss]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from scripts.embedder_resnet import PetEmbedder, get_transforms
except ImportError:
    from embedder_resnet import PetEmbedder, get_transforms

try:
    from scripts.faiss_index import build_faiss, search_faiss
except Exception:
    build_faiss = search_faiss = None


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_gallery(gallery_dir: Path, device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    vec_p = gallery_dir / "vectors.npy"
    lab_p = gallery_dir / "labels.json"
    if not vec_p.exists() or not lab_p.exists():
        raise FileNotFoundError(f"Missing gallery files in {gallery_dir}; run scripts/enroll_gallery.py")
    vectors = torch.from_numpy(np.load(vec_p)).to(device=device, dtype=torch.float32)
    labels = json.loads(lab_p.read_text(encoding="utf-8"))
    if vectors.ndim != 2 or vectors.size(0) != len(labels):
        raise ValueError("Gallery vectors and labels are mismatched.")
    vectors = F.normalize(vectors, p=2, dim=1)
    return vectors, labels


def iter_val_images(val_dir: Path) -> Iterable[Tuple[str, Path]]:
    for pet_dir in sorted(val_dir.iterdir()):
        if not pet_dir.is_dir():
            continue
        for p in sorted(pet_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                yield pet_dir.name, p


def compute_embeddings(model: PetEmbedder, transform, device: torch.device, paths: List[Path]) -> torch.Tensor:
    batch = []
    for p in paths:
        with Image.open(p) as img:
            batch.append(transform(img.convert("RGB")))
    if not batch:
        return torch.empty(0, device=device)
    x = torch.stack(batch, dim=0).to(device)
    with torch.no_grad():
        z = model(x)
    return F.normalize(z, p=2, dim=1)


@dataclass
class Metrics:
    thr: float
    closed_acc: float
    open_reject: float
    precision: float
    recall: float
    f1: float


def sweep(
    gallery_vectors: torch.Tensor,
    gallery_labels: List[str],
    probes_labels: List[str],
    probes_embeddings: torch.Tensor,
    thr_min: float,
    thr_max: float,
    thr_step: float,
    use_faiss: bool,
) -> Tuple[List[Metrics], float]:
    # Prepare similarity function
    if use_faiss and build_faiss is not None:
        idx = build_faiss(gallery_vectors.detach().cpu().numpy())

        def top1_scores(emb: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
            scores, ids = search_faiss(idx, emb.detach().cpu().numpy(), topk=1)
            return scores[:, 0], ids[:, 0]

    else:

        def top1_scores(emb: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
            sims = torch.matmul(gallery_vectors, emb.T).T  # N x G
            vals, inds = torch.topk(sims, k=1, dim=1)
            return vals.squeeze(1).cpu().numpy(), inds.squeeze(1).cpu().numpy()

    known_set = set(gallery_labels)
    known_mask = np.array([lbl in known_set for lbl in probes_labels], dtype=bool)
    scores, indices = top1_scores(probes_embeddings)
    pred_labels = np.array([gallery_labels[i] for i in indices])

    results: List[Metrics] = []
    best_thr = thr_min
    best_f1 = -1.0

    thr = thr_min
    while thr <= thr_max + 1e-9:
        accept = scores >= thr  # accept if top1 score >= thr
        # Closed-set accuracy (known probes): correct accept with correct label
        correct_known = (accept & known_mask & (pred_labels == np.array(probes_labels)))
        total_known = int(known_mask.sum())
        closed_acc = float(correct_known.sum() / total_known) if total_known else float("nan")

        # Open-set reject rate (unknown probes): rejected fraction
        unknown_mask = ~known_mask
        total_unknown = int(unknown_mask.sum())
        open_reject = float((~accept & unknown_mask).sum() / total_unknown) if total_unknown else float("nan")

        # F1 for "correct accepts of known" as positive
        TP = int(correct_known.sum())
        FN = int((known_mask & ~correct_known).sum())  # rejected known + wrong-label accepts
        FP = int((accept & ~known_mask).sum()) + int((accept & known_mask & (pred_labels != np.array(probes_labels))).sum())
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append(Metrics(thr, closed_acc, open_reject, precision, recall, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

        thr = round(thr + thr_step, 6)

    return results, best_thr


def main():
    parser = argparse.ArgumentParser(description="Sweep acceptance threshold using current gallery.")
    parser.add_argument("--val-dir", type=str, default=str(Path(os.getenv("DATA_DIR", "data")) / "faceid"))
    parser.add_argument("--thr-min", type=float, default=0.50)
    parser.add_argument("--thr-max", type=float, default=0.95)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--topk", type=int, default=1)  # reserved, top-1 used for thresholding
    parser.add_argument("--faiss", action="store_true")
    args = parser.parse_args()

    device = get_device()
    model = PetEmbedder().to(device)
    model.eval()
    transform = get_transforms()

    gallery_dir = Path(os.getenv("GALLERY_DIR", "gallery"))
    gallery_vectors, gallery_labels = load_gallery(gallery_dir, device)

    # Load probes from val-dir
    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise FileNotFoundError(f"val-dir not found: {val_dir}")
    label_to_paths: Dict[str, List[Path]] = defaultdict(list)
    for label, path in iter_val_images(val_dir):
        label_to_paths[label].append(path)
    if not label_to_paths:
        raise RuntimeError("No images found under val-dir.")

    all_labels: List[str] = []
    all_paths: List[Path] = []
    for lbl, paths in label_to_paths.items():
        all_labels.extend([lbl] * len(paths))
        all_paths.extend(paths)

    embs = compute_embeddings(model, transform, device, all_paths)
    if embs.numel() == 0:
        raise RuntimeError("Failed to compute embeddings for validation images.")

    results, best_thr = sweep(
        gallery_vectors,
        gallery_labels,
        all_labels,
        embs,
        args.thr_min,
        args.thr_max,
        args.thr_step,
        args.faiss,
    )

    # Print small table
    print("thr  closed_acc  open_reject  precision  recall  f1")
    for m in results:
        ca = f"{m.closed_acc:.3f}" if not math.isnan(m.closed_acc) else "nan"
        orr = f"{m.open_reject:.3f}" if not math.isnan(m.open_reject) else "nan"
        print(f"{m.thr:0.2f}  {ca:>10}  {orr:>11}  {m.precision:>9.3f}  {m.recall:>6.3f}  {m.f1:>4.3f}")
    print(f"RECOMMEND threshold={best_thr:.2f} (max F1)")

    # Save report
    report = {
        "best_thr": best_thr,
        "curve": [
            {
                "thr": m.thr,
                "closed_acc": m.closed_acc,
                "open_reject": m.open_reject,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
            }
            for m in results
        ],
    }
    out_dir = gallery_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "threshold_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
