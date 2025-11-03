import csv
from pathlib import Path
from typing import Dict, List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


CHECKPOINT_PATH = Path("models/embedder.pt")
CROPS_DIR = Path("data/crops")
OUTPUT_CSV = Path("hard_pairs.csv")
EMBED_DIM = 256
BATCH_SIZE = 64
IMAGE_SIZE = 224
TOP_K = 5

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


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[EmbedderModel, List[str]]:
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found at {checkpoint_path}.")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint.get("classes")
    if class_names is None:
        raise SystemExit("Checkpoint missing 'classes' key.")
    if checkpoint.get("emb_dim") != EMBED_DIM:
        raise SystemExit(f"Expected embedding dim {EMBED_DIM}, found {checkpoint.get('emb_dim')}.")

    model = EmbedderModel(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, class_names


def gather_images() -> List[Tuple[str, Path]]:
    if not CROPS_DIR.exists():
        raise SystemExit(f"Crop directory not found at {CROPS_DIR}.")

    all_images: List[Tuple[str, Path]] = []
    for pet_dir in sorted(CROPS_DIR.iterdir()):
        if not pet_dir.is_dir():
            continue
        for image_path in sorted(pet_dir.glob("*")):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                all_images.append((pet_dir.name, image_path))
    if not all_images:
        raise SystemExit(f"No images found under {CROPS_DIR}.")
    return all_images


def compute_embeddings(
    model: EmbedderModel,
    images: List[Tuple[str, Path]],
    device: torch.device,
    transform: transforms.Compose,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    embeddings: List[torch.Tensor] = []
    pet_ids: List[str] = []
    rel_paths: List[str] = []

    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = images[start : start + BATCH_SIZE]
            tensors = []
            for pet_id, image_path in batch:
                with Image.open(image_path) as img:
                    tensor = transform(img.convert("RGB"))
                tensors.append(tensor)
                pet_ids.append(pet_id)
                rel_paths.append(str(image_path.relative_to(CROPS_DIR)))

            batch_tensor = torch.stack(tensors).to(device)
            batch_embeddings = model.encode(batch_tensor)
            embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0), pet_ids, rel_paths


def collect_hard_pairs(sim_matrix: torch.Tensor, pet_ids: List[str], rel_paths: List[str]) -> List[Tuple[str, str, str, str, float]]:
    pet_to_indices: Dict[str, List[int]] = {}
    for idx, pet_id in enumerate(pet_ids):
        pet_to_indices.setdefault(pet_id, []).append(idx)

    all_pairs: List[Tuple[str, str, str, str, float]] = []

    for pet_id, indices in pet_to_indices.items():
        other_indices = [i for i, other_pet in enumerate(pet_ids) if other_pet != pet_id]
        if not other_indices:
            continue

        candidate_indices = torch.tensor(other_indices, dtype=torch.long)
        pet_pairs: List[Tuple[float, int, int]] = []

        for anchor_idx in indices:
            sims = sim_matrix[anchor_idx, candidate_indices]
            if sims.numel() == 0:
                continue
            k = min(TOP_K, sims.numel())
            values, top_indices = torch.topk(sims, k)
            for value, pos in zip(values.tolist(), top_indices.tolist()):
                pet_pairs.append((value, anchor_idx, candidate_indices[pos].item()))

        pet_pairs.sort(key=lambda item: item[0], reverse=True)
        for value, anchor_idx, candidate_idx in pet_pairs[:TOP_K]:
            all_pairs.append(
                (
                    pet_id,
                    rel_paths[anchor_idx],
                    pet_ids[candidate_idx],
                    rel_paths[candidate_idx],
                    value,
                )
            )

    return all_pairs


def save_pairs(pairs: List[Tuple[str, str, str, str, float]], output_path: Path) -> None:
    header = ["pet_a", "img_a", "pet_b", "img_b", "similarity"]
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for pet_a, img_a, pet_b, img_b, similarity in pairs:
            writer.writerow([pet_a, img_a, pet_b, img_b, f"{similarity:.6f}"])


def main() -> None:
    device = select_device()
    model, _ = load_model(CHECKPOINT_PATH, device)
    transform = build_transform()

    images = gather_images()
    embeddings, pet_ids, rel_paths = compute_embeddings(model, images, device, transform)

    sim_matrix = torch.mm(embeddings, embeddings.T)
    num_samples = sim_matrix.size(0)
    if num_samples <= 1:
        raise SystemExit("Not enough samples to compute hard pairs.")

    indices = torch.arange(num_samples)
    sim_matrix[indices, indices] = -1.0

    pairs = collect_hard_pairs(sim_matrix, pet_ids, rel_paths)
    if not pairs:
        raise SystemExit("No hard pairs could be generated.")

    save_pairs(pairs, OUTPUT_CSV)
    print(f"Saved {len(pairs)} hard pairs to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
