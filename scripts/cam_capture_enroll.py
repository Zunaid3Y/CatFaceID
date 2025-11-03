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

from datetime import datetime
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"



def main() -> None:
    pet_id = input("Enter pet ID: ").strip() or "unnamed"
    out_dir = Path("data/faceid") / pet_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # macOS fallback: AVFoundation backend
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("ERROR: Unable to open webcam. Check camera permissions and backend.")
        return

    device = get_device()
    detector = YOLO("runs/cat_head/weights/best.pt")
    detector.to(device)

    window_title = f"Enroll: {pet_id}  [SPACE]=save crop  [Q]=quit"
    saved_count = 0
    overlay_timer = {"saved": 0, "low_quality": 0, "no_detection": 0}

    print(f"Saving crops to {out_dir.resolve()}")
    print("Press SPACE to capture the highest-confidence detection. Press Q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            display = frame.copy()
            best_box = None
            best_conf = -1.0

            results = detector.predict(
                source=frame,
                conf=0.25,
                iou=0.6,
                device=device,
                verbose=False,
            )

            if results:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                        x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                        cv2.rectangle(display, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
                        cv2.putText(
                            display,
                            f"{conf:.2f}",
                            (x1_i, max(y1_i - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        if conf > best_conf:
                            best_conf = conf
                            best_box = (x1_i, y1_i, x2_i, y2_i)

            cv2.putText(
                display,
                "SPACE: save crop   Q: quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Pet: {pet_id} | Saved: {saved_count}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if overlay_timer["saved"] > 0:
                cv2.putText(
                    display,
                    "Saved",
                    (10, display.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 200, 0),
                    3,
                    cv2.LINE_AA,
                )
                overlay_timer["saved"] -= 1

            if overlay_timer["low_quality"] > 0:
                cv2.putText(
                    display,
                    "LOW QUALITY",
                    (10, display.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
                overlay_timer["low_quality"] -= 1

            if overlay_timer["no_detection"] > 0:
                cv2.putText(
                    display,
                    "NO DETECTION",
                    (10, display.shape[0] - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                overlay_timer["no_detection"] -= 1

            cv2.imshow(window_title, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key == ord(" "):
                if best_box is None:
                    overlay_timer["no_detection"] = 20
                    continue

                x1, y1, x2, y2 = best_box
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    overlay_timer["no_detection"] = 20
                    continue

                crop = frame[y1:y2, x1:x2]
                if not pass_quality(crop):
                    overlay_timer["low_quality"] = 20
                    continue

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                out_path = out_dir / f"{ts}.jpg"
                cv2.imwrite(str(out_path), crop)
                saved_count += 1
                overlay_timer["saved"] = 20
                print(f"Saved {out_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Running enrollment update...")
    subprocess.run(["python3", "scripts/enroll_gallery.py"], check=False)

    print(f"Captured {saved_count} images for pet '{pet_id}' in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
