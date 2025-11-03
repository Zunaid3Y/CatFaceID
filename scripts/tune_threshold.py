import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


DATA_DIR = Path("data/crops")
CHECKPOINT_PATH = Path("models/embedder.pt")
GALLERY_VECTORS = Path("gallery/vectors.npy")
GALLERY_METADATA = Path("gallery/metadata.json")
OUTPUT_CSV = Path("metrics.csv")

EMBED_DIM = 256
IMAGE_SIZE = 224
BATCH_SIZE = 64
THRESH_START = 0.1
THRESH_END = 0.9
THRESH_STEP = 0.01
TARGET_FARS = [1e-2, 1e-3]

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
        raise SystemExit("Gallery not found. Run scripts/enroll_gallery.py first.")

    vectors = np.load(GALLERY_VECTORS)
    with GALLERY_METADATA.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if len(metadata) != vectors.shape[0]:
        raise SystemExit("Gallery metadata length mismatch with vectors.")

    pet_ids = [entry["pet_id"] for entry in metadata]
    tensor = torch.from_numpy(vectors).to(device=device, dtype=torch.float32)
    tensor = F.normalize(tensor, dim=1)
    return tensor, pet_ids


def load_embedder(device: torch.device) -> EmbedderModel:
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"Embedder checkpoint not found at {CHECKPOINT_PATH}.")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names = checkpoint.get("classes")
    if class_names is None:
        raise SystemExit("Checkpoint missing 'classes' key.")
    if checkpoint.get("emb_dim") != EMBED_DIM:
        raise SystemExit(f"Expected embedding dim {EMBED_DIM}, found {checkpoint.get('emb_dim')}.")
    model = EmbedderModel(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def gather_images() -> List[Tuple[str, Path]]:
    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found at {DATA_DIR}.")

    images: List[Tuple[str, Path]] = []
    for pet_dir in sorted(DATA_DIR.iterdir()):
        if not pet_dir.is_dir():
            continue
        for image_path in sorted(pet_dir.glob("*")):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                images.append((pet_dir.name, image_path))
    if not images:
        raise SystemExit(f"No images discovered under {DATA_DIR}.")
    return images


def compute_embeddings(
    model: EmbedderModel,
    image_items: List[Tuple[str, Path]],
    transform: transforms.Compose,
    device: torch.device,
) -> Tuple[List[str], torch.Tensor]:
    pet_ids: List[str] = []
    embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(image_items), BATCH_SIZE):
            batch_items = image_items[start : start + BATCH_SIZE]
            tensors = []
            for pet_id, image_path in batch_items:
                with Image.open(image_path) as img:
                    tensor = transform(img.convert("RGB"))
                tensors.append(tensor)
                pet_ids.append(pet_id)

            batch_tensor = torch.stack(tensors).to(device)
            batch_embeddings = model.encode(batch_tensor)
            embeddings.append(batch_embeddings.cpu())

    return pet_ids, F.normalize(torch.cat(embeddings, dim=0), dim=1)


def compute_score_distributions(
    pet_ids: List[str],
    embeddings: torch.Tensor,
    template_labels: List[str],
    template_vectors: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    templates = template_vectors.cpu()
    label_to_index = {label: idx for idx, label in enumerate(template_labels)}
    genuine_scores: List[float] = []
    impostor_scores: List[float] = []

    for emb, pet_id in zip(embeddings, pet_ids):
        template_idx = label_to_index.get(pet_id)
        if template_idx is None:
            continue

        sims = torch.matmul(emb.unsqueeze(0), templates.T).squeeze(0)

        genuine_scores.append(float(sims[template_idx].item()))

        if sims.numel() > 1:
            mask = torch.ones_like(sims, dtype=torch.bool)
            mask[template_idx] = False
            impostor_scores.extend(sims[mask].tolist())

    if not genuine_scores:
        raise SystemExit("No genuine scores computed; ensure crops align with gallery.")
    if not impostor_scores:
        raise SystemExit("No impostor scores computed; need at least two pets in the gallery.")

    return np.array(genuine_scores, dtype=np.float32), np.array(impostor_scores, dtype=np.float32)


def sweep_thresholds(genuine: np.ndarray, impostor: np.ndarray) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []

    thresholds = np.arange(THRESH_START, THRESH_END + 1e-9, THRESH_STEP)
    total_genuine = float(len(genuine))
    total_impostor = float(len(impostor))

    for threshold in thresholds:
        false_rejects = float(np.sum(genuine < threshold))
        false_accepts = float(np.sum(impostor >= threshold))

        frr = false_rejects / total_genuine
        far = false_accepts / total_impostor
        tar = 1.0 - frr

        metrics.append(
            {
                "threshold": threshold,
                "FAR": far,
                "FRR": frr,
                "TAR": tar,
            }
        )

    return metrics


def find_operating_points(metrics: List[Dict[str, float]]) -> Dict[float, Dict[str, float]]:
    op_points: Dict[float, Dict[str, float]] = {}
    for target_far in TARGET_FARS:
        candidates = [m for m in metrics if m["FAR"] <= target_far]
        if not candidates:
            continue

        best = max(candidates, key=lambda m: (m["threshold"], m["TAR"]))
        op_points[target_far] = best

    tight_candidates = [m for m in metrics if m["FAR"] <= TARGET_FARS[-1]]
    if tight_candidates:
        recommended = max(tight_candidates, key=lambda m: (m["TAR"], m["threshold"]))
    else:
        recommended = max(metrics, key=lambda m: m["threshold"])
    op_points["recommended"] = recommended
    return op_points


def save_metrics_csv(metrics: List[Dict[str, float]], output_path: Path) -> None:
    fieldnames = ["threshold", "FAR", "FRR", "TAR"]
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({field: f"{row[field]:.6f}" if field != "threshold" else f"{row[field]:.2f}" for field in fieldnames})


def main() -> None:
    device = select_device()
    transform = build_transform()
    embedder = load_embedder(device)
    template_vectors, template_labels = load_gallery(device)

    image_items = gather_images()
    pet_ids, embeddings = compute_embeddings(embedder, image_items, transform, device)

    genuine_scores, impostor_scores = compute_score_distributions(
        pet_ids,
        embeddings,
        template_labels,
        template_vectors,
    )

    metrics = sweep_thresholds(genuine_scores, impostor_scores)

    save_metrics_csv(metrics, OUTPUT_CSV)

    op_points = find_operating_points(metrics)

    print("Operating points:")
    for target in TARGET_FARS:
        point = op_points.get(target)
        if point:
            print(
                f"  FAR <= {target:.0e}: threshold={point['threshold']:.2f}, FAR={point['FAR']:.4f}, TAR={point['TAR']:.4f}"
            )
        else:
            print(f"  FAR <= {target:.0e}: not achievable within sweep.")

    recommended = op_points.get("recommended")
    if recommended:
        print(
            f"Recommended threshold: {recommended['threshold']:.2f} "
            f"(FAR={recommended['FAR']:.4f}, TAR={recommended['TAR']:.4f})"
        )

    print(f"Saved metrics to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
