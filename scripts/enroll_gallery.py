import os, sys
THIS_DIR = os.path.dirname(__file__)
PARENT   = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

try:
    from scripts.embedder_resnet import PetEmbedder, get_transforms
except ImportError:
    from embedder_resnet import PetEmbedder, get_transforms

try:
    from scripts.landmark_align import try_align_from_cat
    from scripts.quality import pass_quality
except ImportError:
    from landmark_align import try_align_from_cat
    from quality import pass_quality

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
FACEID_DIR = Path("data/faceid")
GALLERY_DIR = Path("gallery")
VECTORS_PATH = GALLERY_DIR / "vectors.npy"
LABELS_PATH = GALLERY_DIR / "labels.json"
META_PATH = GALLERY_DIR / "meta.json"


def list_image_paths(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix in SUPPORTED_EXTS
    )


def main() -> None:
    if not FACEID_DIR.exists():
        raise FileNotFoundError(f"Face gallery directory not found: {FACEID_DIR}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PetEmbedder().to(device)
    model.eval()
    transform = get_transforms()

    pet_labels: List[str] = []
    pet_vectors: List[np.ndarray] = []
    pet_meta: Dict[str, Dict[str, float]] = {}
    total_used = 0
    total_low_quality = 0

    for pet_dir in sorted(p for p in FACEID_DIR.iterdir() if p.is_dir()):
        image_paths = list_image_paths(pet_dir)
        if not image_paths:
            continue

        embeddings = []
        used = 0
        skipped_quality = 0

        for img_path in image_paths:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"Skipping {img_path}: failed to read image")
                continue

            aligned = try_align_from_cat(str(img_path))
            crop_bgr = aligned if aligned is not None else bgr

            try:
                quality_ok = pass_quality(crop_bgr)
            except ValueError as exc:
                print(f"Skipping {img_path}: REJECT_LOW_QUALITY ({exc})")
                skipped_quality += 1
                continue

            if not quality_ok:
                skipped_quality += 1
                continue

            try:
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            except Exception as exc:
                print(f"Skipping {img_path}: preprocessing error ({exc})")
                continue

            with torch.no_grad():
                emb = model(tensor).squeeze(0)
            embeddings.append(emb)
            used += 1

        if not embeddings:
            print(f"[skipped] {pet_dir.name} | used=0 / total={len(image_paths)} | skipped_low_quality={skipped_quality}")
            total_low_quality += skipped_quality
            continue

        pet_tensor = torch.stack(embeddings, dim=0)
        proto = F.normalize(pet_tensor.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
        sims = torch.matmul(pet_tensor, proto)
        std_sim = sims.std(unbiased=False).item() if sims.numel() > 1 else 0.0

        pet_labels.append(pet_dir.name)
        pet_vectors.append(proto.cpu().numpy().astype(np.float32))
        pet_meta[pet_dir.name] = {
            "count": int(used),
            "mean_norm": float(proto.norm().item()),
            "std_sim": float(std_sim),
        }
        total_used += used
        total_low_quality += skipped_quality
        print(f"[enrolled] {pet_dir.name} | used={used} / total={len(image_paths)} | skipped_low_quality={skipped_quality}")

    if not pet_vectors:
        raise RuntimeError("No embeddings generated; ensure data/faceid/* has usable images.")

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_PATH, np.stack(pet_vectors, axis=0))
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(pet_labels, f, indent=2)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(pet_meta, f, indent=2)

    print(
        f"Enrolled pets: {len(pet_labels)} | Total images used: {total_used} | "
        f"Skipped low-quality: {total_low_quality}"
    )


if __name__ == "__main__":
    main()
