import argparse
import random
import shutil
from collections import OrderedDict
from pathlib import Path
from statistics import median_low
from typing import Dict, List, Optional


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
RANDOM_SEED = 42


def gather_pet_images(crops_root: Path) -> Dict[str, List[Path]]:
    """Return image paths grouped by pet id under crops_root."""
    pet_images: Dict[str, List[Path]] = OrderedDict()
    for pet_dir in sorted(crops_root.iterdir()):
        if not pet_dir.is_dir():
            continue
        images = sorted(
            (path for path in pet_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: path.name,
        )
        pet_images[pet_dir.name] = images
    return pet_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balance cropped pet images per class.")
    parser.add_argument(
        "--median",
        action="store_true",
        help="Downsample classes with more images than the median count.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned balancing without copying any files.",
    )
    return parser.parse_args()


def select_images(
    pet_images: Dict[str, List[Path]],
    max_per_pet: Optional[int],
    rng: Optional[random.Random] = None,
) -> Dict[str, List[Path]]:
    """Select images for each pet, optionally downsampling to max_per_pet."""
    selections: Dict[str, List[Path]] = {}
    if not pet_images:
        return selections

    rng = rng or random.Random()
    for pet_id, images in pet_images.items():
        if max_per_pet is not None and len(images) > max_per_pet:
            chosen = rng.sample(images, k=max_per_pet)
            selections[pet_id] = sorted(chosen, key=lambda path: path.name)
        else:
            selections[pet_id] = images
    return selections


def print_counts(pet_images: Dict[str, List[Path]]) -> None:
    print("Counts per pet:")
    if not pet_images:
        print("  (none found)")
        return
    for pet_id, images in pet_images.items():
        print(f"  {pet_id}: {len(images)}")


def print_selection_summary(
    selections: Dict[str, List[Path]],
    pet_images: Dict[str, List[Path]],
    median_count: Optional[int],
) -> None:
    if not selections:
        return
    print()
    if median_count is not None:
        print(f"Median count: {median_count}")
    print("Selected images per pet:")
    for pet_id, selected in selections.items():
        original_count = len(pet_images.get(pet_id, []))
        note = ""
        if median_count is not None and original_count > median_count:
            note = " (downsampled)"
        print(f"  {pet_id}: {len(selected)} of {original_count}{note}")


def copy_images(selections: Dict[str, List[Path]], target_root: Path, dry_run: bool) -> None:
    if dry_run or not selections:
        return

    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for pet_id, images in selections.items():
        dest_dir = target_root / pet_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in images:
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            total_copied += 1

    print()
    print(f"Copied {total_copied} images into {target_root}.")


def main() -> None:
    args = parse_args()

    crops_root = Path("data/crops")
    target_root = Path("data/crops_balanced")

    if not crops_root.exists():
        raise SystemExit(f"Crops directory not found at {crops_root}.")

    pet_images = gather_pet_images(crops_root)
    print_counts(pet_images)

    use_median = args.median
    median_count = median_low([len(images) for images in pet_images.values()]) if use_median and pet_images else None
    rng = random.Random(RANDOM_SEED)
    selections = select_images(pet_images, max_per_pet=median_count, rng=rng)
    print_selection_summary(selections, pet_images, median_count)

    if args.dry_run:
        print()
        print("Dry run: no files were copied.")
        return

    copy_images(selections, target_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
