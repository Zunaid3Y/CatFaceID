import os
import torch
import torchvision as tv
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_LOCAL_CKPT = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth")

class PetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        m = None
        # 1) Try official pretrained weights
        try:
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except Exception as e:
            print("WARN: could not load ResNet50 IMAGENET1K_V2 weights:", e)
            # 2) Fallback: uninitialized model
            m = resnet50(weights=None)
            # 3) If local checkpoint exists, try load it
            if os.path.exists(_LOCAL_CKPT):
                try:
                    sd = torch.load(_LOCAL_CKPT, map_location="cpu")
                    msg = m.load_state_dict(sd, strict=False)
                    print("INFO: loaded local ResNet50 weights:", msg)
                except Exception as e2:
                    print("WARN: failed to load local ResNet50 checkpoint:", e2)

        # chop off FC head -> 2048-d global avgpool output
        self.backbone = nn.Sequential(*(list(m.children())[:-1]))

    @torch.no_grad()
    def forward(self, x):  # x: Bx3x224x224
        z = self.backbone(x).squeeze(-1).squeeze(-1)  # Bx2048
        return nn.functional.normalize(z, dim=1)

def get_transforms():
    # Always safe defaults; if weights meta exists we’ll use it, else ImageNet stats.
    mean, std = list(_IMAGENET_MEAN), list(_IMAGENET_STD)
    try:
        w = ResNet50_Weights.IMAGENET1K_V2
        meta = getattr(w, "meta", {}) or {}
        mean = list(meta.get("mean", mean))
        std  = list(meta.get("std", std))
    except Exception:
        pass
    return tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=std),
    ])
