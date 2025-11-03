import os
import random
from pathlib import Path
from typing import List, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


TRAIN_RATIO = 0.85
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 3e-4
EMBED_DIM = 256
MARGIN_SCALE = 30.0
MARGIN_VALUE = 0.5
RANDOM_SEED = 42
IMAGE_SIZE = 224
DATA_DIR = Path("data/crops")
CHECKPOINT_PATH = Path("models/embedder.pt")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

if hasattr(torch, "set_float32_matmul_precision"):
    try:
        torch.set_float32_matmul_precision("high")
    except ValueError:
        torch.set_float32_matmul_precision("medium")

class ArcMarginProduct(nn.Module):
    """Implements an ArcFace-style cosine margin head."""

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


class SubsetWithTransform(Dataset):
    """Dataset wrapper that applies a transform after indexing into an ImageFolder subset."""

    def __init__(self, dataset: ImageFolder, indices: Sequence[int], transform: transforms.Compose) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class EmbedderModel(nn.Module):
    """Wraps the backbone, projection head, and margin classifier."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=True,
            num_classes=0,
        )
        backbone_dim = getattr(self.backbone, "num_features")
        self.projection = nn.Linear(backbone_dim, EMBED_DIM)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.margin_head = ArcMarginProduct(EMBED_DIM, num_classes, s=MARGIN_SCALE, m=MARGIN_VALUE)

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None):
        features = self.backbone(images)
        embeddings = F.normalize(self.projection(features), dim=1)
        logits = self.margin_head(embeddings, labels) if labels is not None else None
        return embeddings, logits


def select_device() -> torch.device:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available() and mps_backend.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE + 32),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def split_indices(total: int, train_ratio: float, rng: random.Random) -> tuple[List[int], List[int]]:
    if total < 2:
        raise SystemExit("Need at least two images to perform a train/val split.")

    indices = list(range(total))
    rng.shuffle(indices)

    val_size = max(1, int(round(total * (1 - train_ratio))))
    if val_size >= total:
        val_size = 1
    train_size = total - val_size
    if train_size <= 0:
        train_size = total - 1
        val_size = total - train_size

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def create_dataloaders(train_dataset: Dataset, val_dataset: Dataset, device: torch.device) -> tuple[DataLoader, DataLoader]:
    num_workers = min(4, os.cpu_count() or 0)
    pin_memory = device.type in {"cuda"}
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: EmbedderModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0
    non_blocking = device.type != "cpu"

    for images, labels in loader:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        _, logits = model(images, labels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(1, total_samples)


def evaluate(
    model: EmbedderModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    non_blocking = device.type != "cpu"

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            _, logits = model(images, labels)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    return running_loss / max(1, total_samples)


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Dataset directory not found at {DATA_DIR}.")

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    dataset = ImageFolder(str(DATA_DIR))
    if len(dataset) == 0:
        raise SystemExit(f"No images found under {DATA_DIR}.")

    class_names: List[str] = list(dataset.classes)
    train_transform, val_transform = build_transforms()

    rng = random.Random(RANDOM_SEED)
    train_indices, val_indices = split_indices(len(dataset), TRAIN_RATIO, rng)

    train_dataset = SubsetWithTransform(dataset, train_indices, train_transform)
    val_dataset = SubsetWithTransform(dataset, val_indices, val_transform)

    device = select_device()
    print(f"Using device: {device}")

    model = EmbedderModel(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, device)

    best_val_loss = float("inf")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "emb_dim": EMBED_DIM,
                "classes": class_names,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print("Saved models/embedder.pt")


if __name__ == "__main__":
    main()
