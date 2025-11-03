import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

from PIL import Image
from ultralytics import YOLO


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
MARGIN_RATIO = 0.08  # 8% margin around the detected box


def iter_pet_images(raw_root: Path) -> Iterable[Tuple[str, Path]]:
    """Yield (pet_id, image_path) pairs for images under raw_root."""
    for pet_dir in sorted(raw_root.iterdir()):
        if not pet_dir.is_dir():
            continue
        pet_id = pet_dir.name
        for image_path in sorted(pet_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                yield pet_id, image_path


def largest_box_xyxy(result) -> Tuple[float, float, float, float] | None:
    """Return the largest detection box (xyxy) from a YOLO result."""
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
        return None
    xyxy = boxes.xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    max_idx = int(areas.argmax().item())
    return tuple(float(v) for v in xyxy[max_idx])


def expand_bbox(
    box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    margin_ratio: float = MARGIN_RATIO,
) -> Tuple[int, int, int, int]:
    """Expand the bounding box by margin_ratio and clamp to image bounds."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return 0, 0, img_width, img_height

    margin_x = width * margin_ratio / 2
    margin_y = height * margin_ratio / 2

    x1 = math.floor(max(0.0, x1 - margin_x))
    y1 = math.floor(max(0.0, y1 - margin_y))
    x2 = math.ceil(min(float(img_width), x2 + margin_x))
    y2 = math.ceil(min(float(img_height), y2 + margin_y))

    if x2 <= x1:
        x2 = min(img_width, x1 + 1)
    if y2 <= y1:
        y2 = min(img_height, y1 + 1)

    return x1, y1, x2, y2


def main() -> None:
    weights_path = Path("runs/detect/cat_head/weights/best.pt")
    raw_root = Path("data/raw")
    crops_root = Path("data/crops")

    if not weights_path.exists():
        raise SystemExit(f"YOLO weights not found at {weights_path}.")
    if not raw_root.exists():
        raise SystemExit(f"Raw images directory not found at {raw_root}.")

    model = YOLO(weights_path)
    crops_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: {"processed": 0, "skipped": 0})

    for pet_id, image_path in iter_pet_images(raw_root):
        dest_dir = crops_root / pet_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        results = model(image_path, verbose=False)
        if not results:
            summary[pet_id]["skipped"] += 1
            continue

        result = results[0]
        box = largest_box_xyxy(result)
        if box is None:
            summary[pet_id]["skipped"] += 1
            continue

        img_height, img_width = result.orig_shape
        x1, y1, x2, y2 = expand_bbox(box, img_width, img_height)

        with Image.open(image_path) as img:
            crop = img.crop((x1, y1, x2, y2))
            crop.save(dest_dir / image_path.name)

        summary[pet_id]["processed"] += 1

    if not summary:
        print("No images found to process.")
        return

    print("Per-pet summary:")
    total_processed = 0
    total_skipped = 0
    for pet_id in sorted(summary.keys()):
        processed = summary[pet_id]["processed"]
        skipped = summary[pet_id]["skipped"]
        total_processed += processed
        total_skipped += skipped
        print(f"  {pet_id}: processed {processed}, skipped {skipped}")

    print()
    print("Overall:")
    print(f"  Processed: {total_processed}")
    print(f"  Skipped:   {total_skipped}")


if __name__ == "__main__":
    main()
