import os
import sys
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Body, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Robust import guard before importing embedder
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
try:
    from scripts.embedder_resnet import PetEmbedder, get_transforms
except ImportError:
    from embedder_resnet import PetEmbedder, get_transforms

from ultralytics import YOLO


app = FastAPI(title="Pet FaceID API")

# CORS: allow local dev origins
origins = [
    # Local dev UIs
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://localhost:5500",
    # TODO: add your deployed frontend domain here
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Globals loaded at startup
DEVICE_STR: str = "cpu"
DEVICE = torch.device("cpu")
DETECTOR: Optional[YOLO] = None
EMBEDDER: Optional[PetEmbedder] = None
TRANSFORM = None
GALLERY_VECTORS_T: Optional[torch.Tensor] = None  # (n, d) L2-normalized on DEVICE
GALLERY_LABELS: List[str] = []

GALLERY_DIR = Path("gallery")
GALLERY_VECTORS_PATH = GALLERY_DIR / "vectors.npy"
GALLERY_LABELS_PATH = GALLERY_DIR / "labels.json"
YOLO_WEIGHTS = Path("runs/cat_head/weights/best.pt")


def get_device_str() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_gallery_into_memory() -> None:
    global GALLERY_VECTORS_T, GALLERY_LABELS
    if not GALLERY_VECTORS_PATH.exists() or not GALLERY_LABELS_PATH.exists():
        GALLERY_VECTORS_T = None
        GALLERY_LABELS = []
        return
    import json

    vecs = np.load(GALLERY_VECTORS_PATH)
    labels = json.loads(GALLERY_LABELS_PATH.read_text(encoding="utf-8"))
    if vecs.ndim != 2 or len(labels) != vecs.shape[0]:
        GALLERY_VECTORS_T = None
        GALLERY_LABELS = []
        return
    GALLERY_VECTORS_T = F.normalize(
        torch.from_numpy(vecs).to(device=DEVICE, dtype=torch.float32), p=2, dim=1
    )
    GALLERY_LABELS = list(labels)


def best_detection(bgr: np.ndarray) -> Optional[tuple]:
    if DETECTOR is None:
        return None
    results = DETECTOR.predict(
        source=bgr,
        conf=0.25,
        iou=0.6,
        device=DEVICE_STR,
        imgsz=640,
        verbose=False,
    )
    if not results:
        return None
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return None
    idx = torch.argmax(r0.boxes.conf)
    x1, y1, x2, y2 = r0.boxes.xyxy[idx].tolist()
    return int(x1), int(y1), int(x2), int(y2)


def identify_image(bgr: np.ndarray) -> dict:
    # Gallery check
    if GALLERY_VECTORS_T is None or GALLERY_VECTORS_T.numel() == 0:
        return {"label": "NO_GALLERY"}

    # Detect head and crop
    box = best_detection(bgr)
    if box is None:
        return {"label": "NO_FACE"}

    x1, y1, x2, y2 = box
    h, w = bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return {"label": "NO_FACE"}
    crop = bgr[y1:y2, x1:x2]

    # Preprocess + embed
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = __import__("PIL.Image").Image.fromarray(rgb)
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = EMBEDDER(tensor).squeeze(0)
        emb = F.normalize(emb, p=2, dim=0)

    sims = torch.matmul(GALLERY_VECTORS_T, emb)
    k = min(3, sims.numel())
    values, indices = torch.topk(sims, k=k)
    top3 = [[GALLERY_LABELS[int(i.item())], float(v.item())] for v, i in zip(values, indices)]

    # Decision
    label = "UNKNOWN"
    score = float(values[0].item()) if k > 0 else 0.0
    margin = float(values[0].item() - (values[1].item() if k > 1 else -1.0)) if k > 0 else 0.0
    top1_label = GALLERY_LABELS[int(indices[0].item())] if k > 0 else ""

    REJECT_T = 0.75
    MARGIN_T = 0.05
    if k > 0 and (score >= REJECT_T) and (margin >= MARGIN_T):
        label = top1_label

    return {"label": label, "score": score, "top3": top3}


@app.on_event("startup")
async def _startup():
    print("CatFaceID API starting…")
    global DEVICE_STR, DEVICE, DETECTOR, EMBEDDER, TRANSFORM
    DEVICE_STR = get_device_str()
    DEVICE = torch.device(DEVICE_STR)
    # Models
    DETECTOR = YOLO(str(YOLO_WEIGHTS)) if YOLO_WEIGHTS.exists() else None
    if DETECTOR is not None:
        DETECTOR.to(DEVICE_STR)
    EMBEDDER = PetEmbedder().to(DEVICE)
    EMBEDDER.eval()
    TRANSFORM = get_transforms()
    # Gallery
    load_gallery_into_memory()


@app.get("/health")
async def health():
    return {"ok": True}


def _decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return im


class IdentifyB64(BaseModel):
    image_b64: str


@app.post("/identify")
async def identify(
    image: Optional[UploadFile] = File(None),
    payload: Optional[IdentifyB64] = None,
):
    bgr = None
    if image is not None:
        data = await image.read()
        bgr = _decode_image_bytes(data)
    elif payload is not None and payload.image_b64:
        try:
            data = base64.b64decode(payload.image_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")
        bgr = _decode_image_bytes(data)
    else:
        raise HTTPException(status_code=400, detail="Provide 'image' file or JSON with 'image_b64'.")

    if bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode image.")

    return identify_image(bgr)


@app.post("/enroll")
async def enroll(pet_id: str = Form(...), images: List[UploadFile] = File(...)):
    if not pet_id:
        raise HTTPException(status_code=400, detail="pet_id is required")

    out_dir = Path("data/faceid") / pet_id
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for up in images:
        data = await up.read()
        if not data:
            continue
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = out_dir / f"{ts}.jpg"
        with open(out_path, "wb") as f:
            f.write(data)
        saved += 1

    # Rebuild gallery
    try:
        try:
            import scripts.enroll_gallery as enroll_mod
        except ImportError:
            import enroll_gallery as enroll_mod
        enroll_mod.main()
    except Exception:
        # fallback via subprocess
        import subprocess

        subprocess.run(["python3", "scripts/enroll_gallery.py"], check=False)

    # Reload in-memory gallery
    load_gallery_into_memory()

    return {"ok": True, "pet_id": pet_id, "saved": saved}


if __name__ == "__main__":
    import uvicorn

    # For larger forms, keep workers=1 and rely on default body limits.
    # Recommend images <= 2MB each for snappy responses.
    uvicorn.run("api.server:app", host="0.0.0.0", port=9000, reload=True, workers=1)
