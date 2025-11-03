import argparse
import json
from pathlib import Path
from typing import List, Sequence

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
EMBED_DIM = 256
IMAGE_SIZE = 224

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


def load_gallery(device: torch.device) -> tuple[np.ndarray, List[dict]]:
    if GALLERY_VECTORS.exists() and GALLERY_METADATA.exists():
        vectors = np.load(GALLERY_VECTORS)
        with GALLERY_METADATA.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if len(metadata) != vectors.shape[0]:
            raise SystemExit("Gallery metadata length mismatch with vectors.")

        return vectors.astype(np.float32), metadata

    return np.empty((0, EMBED_DIM), dtype=np.float32), []


def save_gallery(vectors: np.ndarray, metadata: List[dict]) -> None:
    GALLERY_VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.save(GALLERY_VECTORS, vectors)
    with GALLERY_METADATA.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def compute_embeddings(model: EmbedderModel, image_paths: Sequence[Path], transform: transforms.Compose, device: torch.device) -> torch.Tensor:
    tensors = []
    for path in image_paths:
        with Image.open(path) as img:
            tensor = transform(img.convert("RGB"))
        tensors.append(tensor)

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        embeddings = model.encode(batch)
    return F.normalize(embeddings, dim=1).cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update gallery templates for a specific pet.")
    parser.add_argument("--pet", required=True, help="Pet identifier to update or create.")
    parser.add_argument("images", nargs="+", help="Image file paths to include in the update.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pet_id = args.pet
    image_paths = [Path(path) for path in args.images]

    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing image files: {', '.join(missing)}")

    device = select_device()
    transform = build_transform()
    model = load_model(device)

    vectors, metadata = load_gallery(device)
    embeddings = compute_embeddings(model, image_paths, transform, device)
    new_vector = F.normalize(embeddings.mean(dim=0, keepdim=True), dim=1).squeeze(0).numpy()

    labels = [entry["pet_id"] for entry in metadata]
    if pet_id in labels:
        index = labels.index(pet_id)
        existing_vector = vectors[index]
        updated = existing_vector + new_vector
        updated = updated / np.linalg.norm(updated)
        vectors[index] = updated
        print(f"Updated template for pet '{pet_id}' with {len(image_paths)} image(s).")
    else:
        vectors = np.vstack([vectors, new_vector[np.newaxis, :]]) if vectors.size else new_vector[np.newaxis, :]
        metadata.append({"pet_id": pet_id})
        print(f"Added new pet '{pet_id}' with {len(image_paths)} image(s).")

    save_gallery(vectors, metadata)
    print("Gallery updated.")


if __name__ == "__main__":
    main()
