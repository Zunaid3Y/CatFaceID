"""Sweep similarity thresholds and margins to tune open-set identification performance."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
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


GALLERY_DIR = Path("gallery")
FACEID_DIR = Path("data/faceid")
RUNS_DIR = Path("runs")
CSV_PATH = RUNS_DIR / "threshold_sweep.csv"

THRESH_RANGE = np.arange(0.3, 0.9, 0.01)
MARGIN_VALUES = [0.0, 0.02, 0.05, 0.1]
MAX_K = 5


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_gallery(device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    vectors_path = GALLERY_DIR / "vectors.npy"
    labels_path = GALLERY_DIR / "labels.json"
    if not vectors_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Gallery vectors.npy or labels.json not found. Run enroll_gallery first.")
    vectors = torch.from_numpy(np.load(vectors_path)).to(device=device, dtype=torch.float32)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if vectors.ndim != 2 or vectors.size(0) != len(labels):
        raise ValueError("Gallery vectors and labels mismatch.")
    return F.normalize(vectors, p=2, dim=1), labels


def iter_pet_images() -> Iterable[Tuple[str, Path]]:
    if not FACEID_DIR.exists():
        raise FileNotFoundError(f"FaceID directory not found: {FACEID_DIR}")
    for pet_dir in sorted(FACEID_DIR.iterdir()):
        if not pet_dir.is_dir():
            continue
        images = sorted(
            p for p in pet_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for img_path in images:
            yield pet_dir.name, img_path


def build_pet_splits() -> Dict[str, List[Path]]:
    pet_images: Dict[str, List[Path]] = defaultdict(list)
    for pet_id, path in iter_pet_images():
        pet_images[pet_id].append(path)
    if not pet_images:
        raise RuntimeError("No pet images found under data/faceid.")
    return pet_images


def compute_embeddings(
    model: PetEmbedder,
    transform,
    device: torch.device,
    image_paths: List[Path],
) -> torch.Tensor:
    tensors = []
    for path in image_paths:
        with Image.open(path) as img:
            tensor = transform(img.convert("RGB"))
        tensors.append(tensor)
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        embeddings = model(batch)
    return F.normalize(embeddings, p=2, dim=1)


def kfold_indices(n: int, k: int) -> Iterable[Tuple[List[int], List[int]]]:
    k = min(k, n) if n > 1 else 1
    if k < 2:
        yield list(range(n)), []
        return
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1
    indices = list(range(n))
    start = 0
    for fold_size in fold_sizes:
        val_idx = indices[start:start + fold_size]
        train_idx = indices[:start] + indices[start + fold_size:]
        yield train_idx, val_idx
        start += fold_size


def evaluate_thresholds(
    gallery_vectors: torch.Tensor,
    gallery_labels: List[str],
    probes: List[Tuple[str, torch.Tensor]],
) -> Tuple[List[Tuple[float, float, float, float]], Tuple[float, float]]:
    results = []
    best_choice = None

    for thresh, margin in itertools.product(THRESH_RANGE, MARGIN_VALUES):
        total_genuine = total_impostor = 0
        reject_genuine = accept_impostor = 0

        for pet_id, scores in probes:
            total_scores = scores.tolist()
            top1 = max(total_scores)
            top1_idx = total_scores.index(top1)
            sorted_scores = sorted(total_scores, reverse=True)
            top2 = sorted_scores[1] if len(sorted_scores) > 1 else float("-inf")

            if gallery_labels[top1_idx] == pet_id:
                total_genuine += 1
                if top1 < thresh or (top1 - top2) < margin:
                    reject_genuine += 1
            else:
                total_impostor += 1
                if top1 >= thresh and (top1 - top2) >= margin:
                    accept_impostor += 1

        far = accept_impostor / total_impostor if total_impostor else math.nan
        frr = reject_genuine / total_genuine if total_genuine else math.nan

        results.append((thresh, margin, far, frr))

        if not math.isnan(far) and not math.isnan(frr):
            if far <= 0.01:
                if best_choice is None or frr < best_choice[2]:
                    best_choice = (thresh, margin, frr)
            elif far <= 0.02 and (best_choice is None or best_choice[2] > 0.01):
                if best_choice is None or frr < best_choice[2]:
                    best_choice = (thresh, margin, frr)

    if best_choice is None and results:
        # fallback to minimal FRR overall
        filtered = [(t, m, frr) for t, m, far, frr in results if not math.isnan(frr)]
        if filtered:
            best_choice = min(filtered, key=lambda x: x[2])

    best_pair = (best_choice[0], best_choice[1]) if best_choice else (float("nan"), float("nan"))
    return results, best_pair


def main() -> None:
    device = get_device()
    model = PetEmbedder().to(device)
    model.eval()
    transform = get_transforms()

    gallery_vectors, gallery_labels = load_gallery(device)
    pet_images = build_pet_splits()

    probes: List[Tuple[str, torch.Tensor]] = []
    label_to_idx = {label: idx for idx, label in enumerate(gallery_labels)}

    for pet_id, paths in pet_images.items():
        if len(paths) < 2:
            continue  # cannot form probe from single image
        if pet_id not in label_to_idx:
            print(f"Skipping {pet_id}: not present in gallery labels.")
            continue

        embeddings = compute_embeddings(model, transform, device, paths)

        for train_idx, val_idx in kfold_indices(len(paths), MAX_K):
            if not val_idx or not train_idx:
                continue
            support = embeddings[train_idx]
            prototype = support.mean(dim=0, keepdim=True)
            prototype = F.normalize(prototype, p=2, dim=1)

            # similarity of probe to all gallery prototypes
            probe_embeddings = embeddings[val_idx]
            class_vectors = gallery_vectors.clone()
            class_vectors[label_to_idx[pet_id]] = prototype.squeeze(0)
            sims = torch.matmul(class_vectors, probe_embeddings.T).T  # val x gallery

            for i, idx in enumerate(val_idx):
                probes.append((pet_id, sims[i].detach().cpu()))

    if not probes:
        raise RuntimeError("No probes generated. Need at least two images per pet.")

    results, best_pair = evaluate_thresholds(gallery_vectors, gallery_labels, probes)

    print("Threshold  Margin  FAR      FRR")
    for thresh, margin, far, frr in results:
        far_str = f"{far:.4f}" if not math.isnan(far) else "nan"
        frr_str = f"{frr:.4f}" if not math.isnan(frr) else "nan"
        print(f"{thresh:8.2f}  {margin:6.2f}  {far_str:>7}  {frr_str:>7}")

    if best_pair[0] == best_pair[0]:  # not nan
        print(f"RECOMMEND reject={best_pair[0]:.2f}, margin={best_pair[1]:.2f}")
    else:
        print("RECOMMEND reject=nan, margin=nan (no suitable thresholds)")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "margin", "FAR", "FRR"])
        for thresh, margin, far, frr in results:
            writer.writerow([f"{thresh:.2f}", f"{margin:.2f}", far, frr])


if __name__ == "__main__":
    main()
