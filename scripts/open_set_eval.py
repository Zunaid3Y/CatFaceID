import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


CHECKPOINT_PATH = Path("models/embedder.pt")
GALLERY_VECTORS = Path("gallery/vectors.npy")
GALLERY_METADATA = Path("gallery/metadata.json")
KNOWN_DIR = Path("data/crops")
UNKNOWN_DIR = Path("data/crops_unknown")

EMBED_DIM = 256
IMAGE_SIZE = 224
BATCH_SIZE = 64
TARGET_FPR = 0.001  # 0.1%

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.eps = 1e-7

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # pragma: no cover
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


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE + 32),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_model(device: torch.device) -> EmbedderModel:
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"Embedder checkpoint not found at {CHECKPOINT_PATH}.")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names = checkpoint.get("classes")
    if class_names is None:
        raise SystemExit("Checkpoint missing 'classes' key.")
    if checkpoint.get("emb_dim") != EMBED_DIM:
        raise SystemExit(f"Expected embedding dim {EMBED_DIM}, found {checkpoint.get('emb_dim')}.")

    model = EmbedderModel(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def load_gallery(device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    if not GALLERY_VECTORS.exists() or not GALLERY_METADATA.exists():
        raise SystemExit("Gallery not found. Run scripts/enroll_gallery.py first.")

    vectors = np.load(GALLERY_VECTORS)
    with GALLERY_METADATA.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if len(metadata) != vectors.shape[0]:
        raise SystemExit("Gallery metadata length mismatch with vectors.")

    tensor = torch.from_numpy(vectors).to(device=device, dtype=torch.float32)
    tensor = F.normalize(tensor, dim=1)
    labels = [entry["pet_id"] for entry in metadata]
    return tensor, labels


def gather_images(root: Path) -> List[Tuple[str, Path]]:
    if not root.exists():
        return []

    items: List[Tuple[str, Path]] = []
    for pet_dir in sorted(root.iterdir()):
        if not pet_dir.is_dir():
            continue
        for path in sorted(pet_dir.glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((pet_dir.name, path))
    return items


def compute_embeddings(
    model: EmbedderModel,
    image_items: Sequence[Tuple[str, Path]],
    transform: transforms.Compose,
    device: torch.device,
) -> Tuple[List[str], torch.Tensor]:
    labels: List[str] = []
    vectors: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(image_items), BATCH_SIZE):
            batch_items = image_items[start : start + BATCH_SIZE]
            tensors = []
            for label, path in batch_items:
                with Image.open(path) as img:
                    tensor = transform(img.convert("RGB"))
                tensors.append(tensor)
                labels.append(label)

            batch = torch.stack(tensors).to(device)
            embeddings = model.encode(batch)
            vectors.append(embeddings.cpu())

    if not vectors:
        return [], torch.empty((0, EMBED_DIM))
    return labels, F.normalize(torch.cat(vectors, dim=0), dim=1)


def summarize_distribution(name: str, values: np.ndarray) -> None:
    if values.size == 0:
        print(f"{name}: no samples.")
        return

    percentiles = np.percentile(values, [50, 90, 95, 99, 99.9])
    print(f"{name}: count={values.size}, mean={values.mean():.4f}, std={values.std():.4f}, min={values.min():.4f}, max={values.max():.4f}")
    print(
        f"  percentiles: 50%={percentiles[0]:.4f}, 90%={percentiles[1]:.4f}, 95%={percentiles[2]:.4f}, "
        f"99%={percentiles[3]:.4f}, 99.9%={percentiles[4]:.4f}"
    )


def determine_threshold(unknown_scores: np.ndarray) -> Tuple[float, float]:
    if unknown_scores.size == 0:
        return 1.0, 0.0

    sorted_desc = np.sort(unknown_scores)[::-1]
    n = sorted_desc.size

    threshold = min(1.0, sorted_desc[0] + 1e-6)
    actual_fpr = 0.0

    for idx, score in enumerate(sorted_desc):
        fpr = (idx + 1) / n
        if fpr <= TARGET_FPR:
            threshold = float(score)
            actual_fpr = float(fpr)
        else:
            break

    if actual_fpr > TARGET_FPR:
        threshold = min(1.0, sorted_desc[0] + 1e-6)
        actual_fpr = 0.0

    return threshold, actual_fpr


def main() -> None:
    device = select_device()
    transform = build_transform()
    model = load_model(device)
    gallery_vectors, gallery_labels = load_gallery(device)

    known_items = gather_images(KNOWN_DIR)
    known_labels, known_embeddings = compute_embeddings(model, known_items, transform, device)

    unknown_items = gather_images(UNKNOWN_DIR)
    unknown_labels, unknown_embeddings = compute_embeddings(model, unknown_items, transform, device)

    print(f"Loaded gallery templates: {len(gallery_labels)}")
    print(f"Known evaluation samples: {known_embeddings.shape[0]}")
    print(f"Unknown evaluation samples: {unknown_embeddings.shape[0]}")

    known_scores: List[float] = []
    missing_known = 0
    label_to_index = {label: idx for idx, label in enumerate(gallery_labels)}

    if known_embeddings.size(0) > 0:
        sims = torch.matmul(known_embeddings, gallery_vectors.T).numpy()
        for idx, label in enumerate(known_labels):
            if label not in label_to_index:
                missing_known += 1
                continue
            known_scores.append(float(sims[idx].max()))

    unknown_scores: List[float] = []
    if unknown_embeddings.size(0) > 0:
        sims_unknown = torch.matmul(unknown_embeddings, gallery_vectors.T).numpy()
        unknown_scores = [float(row.max()) for row in sims_unknown]

    known_scores_np = np.array(known_scores, dtype=np.float32)
    unknown_scores_np = np.array(unknown_scores, dtype=np.float32)

    summarize_distribution("Known max similarities", known_scores_np)
    summarize_distribution("Unknown max similarities", unknown_scores_np)

    if missing_known:
        print(f"Warning: {missing_known} known images skipped because their pet_id is absent from the gallery.")

    threshold, actual_fpr = determine_threshold(unknown_scores_np)
    tar = float((known_scores_np >= threshold).sum() / max(1, known_scores_np.size))
    print(
        f"Suggested threshold for FPR <= {TARGET_FPR:.4%}: {threshold:.4f} "
        f"(actual FPR={actual_fpr:.4%}, TAR on known={tar:.4%})"
    )


if __name__ == "__main__":
    main()
