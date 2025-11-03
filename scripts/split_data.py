import os
import random
import shutil
from glob import glob

random.seed(42)

IM_DIR = "data/raw/cats"
LB_DIR = "data/labels/cats"
OUT_DIR = "data/splits"
TRAIN_RATIO = 0.9  # 90/10 split


def ensure_split_dirs() -> None:
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, split, "labels"), exist_ok=True)


def copy_pair(image_path: str, split: str) -> bool:
    name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(LB_DIR, f"{name}.txt")
    if not os.path.exists(label_path):
        return False
    shutil.copy2(image_path, os.path.join(OUT_DIR, split, "images", os.path.basename(image_path)))
    shutil.copy2(label_path, os.path.join(OUT_DIR, split, "labels", f"{name}.txt"))
    return True


def main() -> None:
    images = sorted(glob(os.path.join(IM_DIR, "*.jpg")))
    if not images:
        raise FileNotFoundError(f"No images found in {IM_DIR}")

    ensure_split_dirs()

    random.shuffle(images)
    cut = int(len(images) * TRAIN_RATIO)
    train_imgs, val_imgs = images[:cut], images[cut:]

    train_count = sum(copy_pair(img, "train") for img in train_imgs)
    val_count = sum(copy_pair(img, "val") for img in val_imgs)

    print(f"Train pairs: {train_count} | Val pairs: {val_count}")


if __name__ == "__main__":
    main()
