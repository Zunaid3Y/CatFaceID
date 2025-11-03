import os, sys
THIS_DIR = os.path.dirname(__file__)
PARENT   = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
try:
    from scripts.embedder_resnet import PetEmbedder, get_transforms
except ImportError:
    from embedder_resnet import PetEmbedder, get_transforms
from qual import pass_quality
from ultralytics import YOLO

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


GALLERY_VECTORS = Path("gallery") / "vectors.npy"
GALLERY_LABELS = Path("gallery") / "labels.json"
YOLO_WEIGHTS = Path("runs/cat_head/weights/best.pt")
SNAP_DIR = Path("runs/identify_snaps")


def get_device_str() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_device() -> torch.device:
    return torch.device(get_device_str())



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Webcam pet identification")
    p.add_argument("--reject", type=float, default=0.75, help="Open-set reject threshold on top-1 similarity")
    p.add_argument("--margin", type=float, default=0.05, help="Decision margin between top-1 and top-2")
    p.add_argument("--show", action="store_true", help="Show crop preview window")
    return p.parse_args()


def load_gallery(device: torch.device):
    if not GALLERY_VECTORS.exists() or not GALLERY_LABELS.exists():
        print("Gallery not found. Please run: python scripts/enroll_gallery.py")
        sys.exit(1)
    vectors = np.load(GALLERY_VECTORS)
    labels = list(np.load(GALLERY_LABELS.with_suffix("").with_suffix(".json")) if False else [])  # placeholder
    # Robust JSON read
    import json
    labels = json.loads(GALLERY_LABELS.read_text(encoding="utf-8"))

    gv = torch.from_numpy(vectors).to(device=device, dtype=torch.float32)
    gv = F.normalize(gv, p=2, dim=1)
    return gv, labels


def main() -> None:
    args = parse_args()
    device_str = get_device_str()
    device = get_device()

    # Load gallery
    gallery_vectors, gallery_labels = load_gallery(device)

    # Load YOLO
    if not YOLO_WEIGHTS.exists():
        print(f"Detector weights not found at {YOLO_WEIGHTS}")
        sys.exit(1)
    detector = YOLO(str(YOLO_WEIGHTS))
    detector.to(device_str)

    # Embedder + transforms
    embedder = PetEmbedder().to(device)
    embedder.eval()
    transform = get_transforms()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # macOS fallback: AVFoundation backend
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("ERROR: Unable to open webcam. Check camera permissions and backend.")
        sys.exit(1)

    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    window_title = "Identify  [S]=snapshot, [Q]=quit"
    crop_window = "Crop"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            display = frame.copy()
            best_box = None
            best_conf = -1.0

            # Detect
            results = detector.predict(
                source=frame,
                conf=0.25,
                iou=0.6,
                device=device_str,
                imgsz=640,
                verbose=False,
            )

            label_text = "NO FACE"
            label_color = (0, 0, 255)
            crop = None

            if results:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                        if float(conf) > best_conf:
                            best_conf = float(conf)
                            best_box = (int(x1), int(y1), int(x2), int(y2))

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]

            if crop is not None:
                if pass_quality(crop):
                    # Embed and compare
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    inp = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = embedder(inp).squeeze(0)
                        emb = F.normalize(emb, p=2, dim=0)
                    sims = torch.matmul(gallery_vectors, emb)
                    if sims.numel() > 0:
                        values, indices = torch.topk(sims, k=min(2, sims.numel()))
                        top1 = float(values[0].item())
                        top1_label = gallery_labels[int(indices[0].item())]
                        top2 = float(values[1].item()) if values.numel() > 1 else -1.0
                        if top1 < args.reject or (top1 - top2) < args.margin:
                            label_text = f"UNKNOWN  sim={top1:.3f}"
                            label_color = (0, 0, 255)
                        else:
                            label_text = f"{top1_label}  sim={top1:.3f}"
                            label_color = (0, 200, 0)
                    else:
                        label_text = "NO GALLERY"
                        label_color = (0, 0, 255)
                else:
                    label_text = "LOW QUALITY"
                    label_color = (0, 0, 255)

            # Draw overlays
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                cv2.rectangle(display, (x1, y1), (x2, y2), label_color, 2)
                cv2.putText(
                    display,
                    label_text,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    label_color,
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display,
                    label_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    label_color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_title, display)
            if args.show:
                if crop is not None:
                    cv2.imshow(crop_window, crop)
                else:
                    # show blank crop window if enabled but no crop
                    blank = np.zeros((120, 120, 3), dtype=np.uint8)
                    cv2.putText(blank, "No crop", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    cv2.imshow(crop_window, blank)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                out_path = SNAP_DIR / f"{ts}.jpg"
                cv2.imwrite(str(out_path), display)
                print(f"Snapshot saved: {out_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
