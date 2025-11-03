import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def collect_images(raw_root: Path) -> Dict[str, List[Path]]:
    images_by_pet: Dict[str, List[Path]] = defaultdict(list)
    for image_path in raw_root.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            rel_path = image_path.relative_to(raw_root)
            pet_id = rel_path.parts[0] if rel_path.parts else "<unknown>"
            images_by_pet[pet_id].append(image_path)
    return images_by_pet


def collect_labels(labels_root: Path) -> Dict[str, List[Path]]:
    labels_by_pet: Dict[str, List[Path]] = defaultdict(list)
    for label_path in labels_root.rglob("*.txt"):
        if label_path.is_file():
            rel_path = label_path.relative_to(labels_root)
            pet_id = rel_path.parts[0] if rel_path.parts else "<unknown>"
            labels_by_pet[pet_id].append(label_path)
    return labels_by_pet


def main() -> None:
    raw_root = Path("data/raw")
    labels_root = Path("data/labels")

    if not raw_root.exists():
        raise SystemExit(f"Raw directory not found at {raw_root}.")
    if not labels_root.exists():
        raise SystemExit(f"Labels directory not found at {labels_root}.")

    images_by_pet = collect_images(raw_root)
    labels_by_pet = collect_labels(labels_root)

    missing_labels: List[Path] = []
    orphan_labels: List[Path] = []
    pet_stats = defaultdict(lambda: {"images": 0, "labels": 0, "matched": 0, "missing": 0, "orphan": 0})

    all_pets = sorted(set(images_by_pet.keys()) | set(labels_by_pet.keys()))

    for pet_id in all_pets:
        pet_images = images_by_pet.get(pet_id, [])
        pet_labels = labels_by_pet.get(pet_id, [])

        for image_path in pet_images:
            rel_image = image_path.relative_to(raw_root)
            pet_stats[pet_id]["images"] += 1
            expected_label_rel = rel_image.with_suffix(".txt")
            expected_label_path = labels_root / expected_label_rel
            if expected_label_path.exists():
                pet_stats[pet_id]["matched"] += 1
            else:
                missing_labels.append(expected_label_rel)
                pet_stats[pet_id]["missing"] += 1

        pet_stats[pet_id]["labels"] += len(pet_labels)

        image_candidates = {image_path.relative_to(raw_root).with_suffix("").as_posix() for image_path in pet_images}
        for label_path in pet_labels:
            label_key = label_path.relative_to(labels_root).with_suffix("").as_posix()
            if label_key not in image_candidates:
                orphan_labels.append(label_path.relative_to(labels_root))
                pet_stats[pet_id]["orphan"] += 1

    if not all_pets:
        print("No pets found under data/raw or data/labels.")
        sys.exit(0)

    print("Per-pet summary:")
    for pet_id in all_pets:
        stats = pet_stats[pet_id]
        print(f"  {pet_id}:")
        print(f"    Images: {stats['images']} (matched: {stats['matched']}, missing labels: {stats['missing']})")
        print(f"    Labels: {stats['labels']} (orphans: {stats['orphan']})")

    print()
    if missing_labels:
        print("Missing label files:")
        for label in missing_labels:
            print(f"  {label}")
    else:
        print("No missing label files detected.")

    print()
    if orphan_labels:
        print("Orphan label files (no corresponding image):")
        for label in orphan_labels:
            print(f"  {label}")
    else:
        print("No orphan label files detected.")

    print()
    total_images = sum(pet_stats[pet]["images"] for pet in all_pets)
    total_labels = sum(pet_stats[pet]["labels"] for pet in all_pets)
    total_missing = len(missing_labels)
    total_orphan = len(orphan_labels)
    print("Overall summary:")
    print(f"  Images: {total_images}")
    print(f"  Labels: {total_labels}")
    print(f"  Missing labels: {total_missing}")
    print(f"  Orphan labels: {total_orphan}")

    if missing_labels or orphan_labels:
        sys.exit(1)


if __name__ == "__main__":
    main()
