import os
import sys
import base64
import time
import traceback
from pathlib import Path
from typing import List, Optional
import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image

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


app = FastAPI()

# CORS: allow local dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Ensure required directories exist
os.makedirs("data/faceid", exist_ok=True)
os.makedirs("gallery", exist_ok=True)

try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pass

# Identification helpers (relying on scripts implementation when available)
def identify_image_from_bgr(bgr):
    # Use your existing scripts/identify.py implementation if available
    try:
        from scripts.identify import identify_bgr  # type: ignore
    except Exception:
        identify_bgr = None  # fallback to local implementation
    if identify_bgr is not None:
        return identify_bgr(bgr)
    # Fallback to server's built-in identify_image
    return identify_image(bgr)


def bytes_to_bgr(buf: bytes):
    # 1) try OpenCV fast path
    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # 2) fallback to PIL (handles HEIC/WebP if pillow-heif installed)
    pil = Image.open(BytesIO(buf)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _to_bgr_upload(f: UploadFile):
    return bytes_to_bgr(f.file.read())


@app.get("/health")
def health():
    return {"ok": True}





# Globals loaded at startup
DEVICE_STR: str = "cpu"
DEVICE = torch.device("cpu")
DETECTOR: Optional[YOLO] = None
EMBEDDER: Optional[PetEmbedder] = None
TRANSFORM = None
GALLERY_VECTORS_T: Optional[torch.Tensor] = None  # (n, d) L2-normalized on DEVICE
GALLERY_LABELS: List[str] = []

# Thresholds/margin from env (overridable via POST /config)
THR: float = float(os.getenv("CATFACEID_THRESHOLD", "0.72"))
MARGIN: float = float(os.getenv("CATFACEID_MARGIN", "0.03"))

GALLERY_DIR = Path(os.getenv("GALLERY_DIR", "gallery"))
GALLERY_VECTORS_PATH = GALLERY_DIR / "vectors.npy"
GALLERY_LABELS_PATH = GALLERY_DIR / "labels.json"

# Try common detector weight locations; pick the first that exists
_YOLO_CANDIDATES = [
    Path("runs/pet_head/weights/best.pt"),
    Path("runs/cat_head/weights/best.pt"),
    Path("runs/detect/pet_head/weights/best.pt"),
    Path("runs/detect/cat_head/weights/best.pt"),
]
YOLO_WEIGHTS = next((p for p in _YOLO_CANDIDATES if p.exists()), _YOLO_CANDIDATES[0])

# Ensure required directories exist
os.makedirs("data/faceid", exist_ok=True)
os.makedirs("gallery", exist_ok=True)


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
        conf=0.35,
        iou=0.6,
        device=DEVICE_STR,
        imgsz=416,
        max_det=1,
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

    return {"label": label, "score": score, "top3": top3, "box": [x1, y1, x2, y2]}


@app.on_event("startup")
async def _startup():
    print("CatFaceID API starting…")
    global DEVICE_STR, DEVICE, DETECTOR, EMBEDDER, TRANSFORM
    DEVICE_STR = get_device_str()
    DEVICE = torch.device(DEVICE_STR)
    # Models
    print(f"Detector weights path: {YOLO_WEIGHTS} (exists={YOLO_WEIGHTS.exists()})")
    DETECTOR = YOLO(str(YOLO_WEIGHTS)) if YOLO_WEIGHTS.exists() else None
    if DETECTOR is not None:
        DETECTOR.to(DEVICE_STR)
    EMBEDDER = PetEmbedder().to(DEVICE)
    EMBEDDER.eval()
    TRANSFORM = get_transforms()
    # Gallery
    load_gallery_into_memory()
    # Propagate thresholds to scripts.identify, if available
    try:
        import scripts.identify as ident_mod  # type: ignore
        ident_mod._CTX["threshold"] = THR
        ident_mod._CTX["margin"] = MARGIN
        # Also pass resolved paths via env for consistency
        os.environ["CATFACEID_GALLERY"] = str(GALLERY_DIR)
        os.environ["CATFACEID_WEIGHTS"] = str(YOLO_WEIGHTS)
    except Exception:
        pass


def _read_upload_to_bgr(f: UploadFile):
    try:
        buf = f.file.read()
        arr = np.frombuffer(buf, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


@app.post("/identify")
async def identify_endpoint(image: Optional[UploadFile] = File(default=None), image_b64: Optional[str] = None):
    try:
        bgr = None
        if image is not None:
            # Prefer async read; fall back to underlying file if needed
            data = await image.read()
            if not data:
                try:
                    image.file.seek(0)
                    data = image.file.read()
                except Exception:
                    data = b""
            if data:
                bgr = bytes_to_bgr(data)
        elif image_b64:
            raw = base64.b64decode(image_b64.split(",")[-1])
            bgr = bytes_to_bgr(raw)
        else:
            return JSONResponse({"error":"No image"}, status_code=400)
        if bgr is None:
            return JSONResponse({"error":"Decode fail"}, status_code=400)
        return JSONResponse(identify_image_from_bgr(bgr))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/enroll")
async def enroll_endpoint(
    pet_id: str = Form(...),
    images: Optional[List[UploadFile]] = File(default=None),
    images_alt: Optional[List[UploadFile]] = File(default=None, alias="images[]"),
):
    try:
        files = images or images_alt
        if not files:
            return JSONResponse(
                {"ok": False, "error": "No images provided (field name must be 'images')"},
                status_code=400,
            )
        base_dir = os.getenv("DATA_DIR", "data")
        td = os.path.join(base_dir, "faceid", pet_id)
        os.makedirs(td, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        saved = bad = small = 0
        for f in files:
            try:
                bgr = _to_bgr_upload(f)
                if bgr is None:
                    bad += 1; continue
                if min(bgr.shape[:2]) < 64:
                    small += 1; continue
                outp = os.path.join(td, f"enrolled_{ts}_{saved:03d}.jpg")
                cv2.imwrite(outp, bgr)
                saved += 1
            except Exception:
                bad += 1
        if saved == 0:
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"No valid images (bad:{bad}, too small:{small}). Try JPG/PNG or enable HEIC support.",
                },
                status_code=400,
            )
        # Rebuild gallery
        from scripts.enroll_gallery import main as rebuild
        rebuild()
        # Reload in-memory gallery so new pets are available immediately
        try:
            load_gallery_into_memory()
        except Exception:
            pass
        return {"ok": True, "pet_id": pet_id, "saved": saved, "skipped_bad": bad, "skipped_small": small}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/config")
def get_config():
    """Return current identification configuration."""
    faiss_env = os.getenv("CATFACEID_FAISS", "false").lower() in {"1", "true", "yes"}
    return {
        "threshold": THR,
        "margin": MARGIN,
        "faiss": faiss_env,
        "weights": str(YOLO_WEIGHTS),
        "gallery": str(GALLERY_DIR),
    }


@app.post("/config")
def set_config(threshold: float | None = None, margin: float | None = None):
    """Update threshold/margin in-memory (no persistence). Protect with auth in production."""
    global THR, MARGIN
    if threshold is not None:
        THR = float(threshold)
    if margin is not None:
        MARGIN = float(margin)
    try:
        import scripts.identify as ident_mod  # type: ignore
        ident_mod._CTX["threshold"] = THR
        ident_mod._CTX["margin"] = MARGIN
    except Exception:
        pass
    return {"ok": True, "threshold": THR, "margin": MARGIN}


if __name__ == "__main__":
    import uvicorn
    # For larger forms, keep workers=1 and rely on default body limits.
    # Recommend images <= 2MB each for snappy responses.
    uvicorn.run("api.server:app", host="0.0.0.0", port=9000, reload=True, workers=1)
