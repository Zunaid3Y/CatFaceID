import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List


CHECKPOINT_PATH = Path("models/embedder.pt")
ONNX_OUTPUT_PATH = Path("models/embedder.onnx")
EMBED_DIM = 256
IMAGE_SIZE = 224


class ArcMarginProduct(nn.Module):
    """ArcFace-style cosine margin layer kept for state dict compatibility."""

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        embeddings = F.normalize(self.projection(features), dim=1)
        return embeddings


def load_model(device: torch.device) -> nn.Module:
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"Checkpoint not found at {CHECKPOINT_PATH}.")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names: List[str] = checkpoint.get("classes")
    if class_names is None:
        raise SystemExit("Checkpoint missing 'classes'.")
    if checkpoint.get("emb_dim") != EMBED_DIM:
        raise SystemExit(f"Expected embedding dim {EMBED_DIM}, found {checkpoint.get('emb_dim')}.")

    model = EmbedderModel(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def export_to_onnx(model: EmbedderModel, device: torch.device) -> None:
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    dynamic_axes = {"input": {0: "batch"}, "emb": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUTPUT_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["emb"],
        dynamic_axes=dynamic_axes,
    )


def verify_export() -> None:
    if not ONNX_OUTPUT_PATH.exists():
        raise SystemExit(f"Failed to find exported ONNX at {ONNX_OUTPUT_PATH}.")

    onnx_model = onnx.load(str(ONNX_OUTPUT_PATH))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(ONNX_OUTPUT_PATH), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    outputs = session.run(["emb"], {"input": dummy})
    if not outputs or outputs[0].shape != (1, EMBED_DIM):
        raise SystemExit(f"Unexpected output shape: {outputs[0].shape if outputs else 'None'}")

    print(f"Export verified: output shape {outputs[0].shape}")


def main() -> None:
    device = torch.device("cpu")
    model = load_model(device)

    export_to_onnx(model, device)
    verify_export()
    print(f"Saved ONNX model to {ONNX_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
