#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Script: Filter Low-Resolution Images Out of a Cleaned Dataset
#
# What it does:
# - Reads images from <src>/images and labels from <src>/labels (flat folders).
# - Removes every image whose shorter side is below --min-side, together with
#   its label file. The default (2048) matches the default tile size of
#   tile_building_defect_images.py, which skips images smaller than a tile.
# - Works in place on the dataset. By default the removed files are moved to
#   <src>/rejected_low_res/{images,labels} so the step can be undone; nothing
#   downstream reads that folder.
# - Refuses to run on anything under data_raw/ (raw data is never modified):
#   run clean_to_x_classes.py first and point this at its output.
# - Prints a summary: images kept/removed and a resolution breakdown.
#
# How to use (paths are relative to the current directory; examples assume you
# are in data-cleaning/):
# - Default run (threshold 2048, rejects moved aside):
#   python scripts/filter_low_res.py --src ../data/data-cleaned/tf_r2_cleaned
# - Custom threshold (use the tile size you will tile with):
#   python scripts/filter_low_res.py --src ../data/data-cleaned/tf_r2_cleaned --min-side 1024
# - Only list what would be removed, without touching anything:
#   python scripts/filter_low_res.py --src ../data/data-cleaned/tf_r2_cleaned --dry-run
# - Delete the rejects instead of moving them aside:
#   python scripts/filter_low_res.py --src ../data/data-cleaned/tf_r2_cleaned --delete
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # Fallback if tqdm isn't installed.
    tqdm = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REJECTED_DIR_NAME = "rejected_low_res"
RAW_DIR_NAME = "data_raw"


def print_resolutions(title: str, resolutions: Counter) -> None:
    print(title)
    if not resolutions:
        print("  (none)")
        return
    for (width, height), count in sorted(resolutions.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {width}x{height}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove images (and labels) whose shorter side is below a minimum, in place."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Cleaned dataset root containing images/ and labels/.",
    )
    parser.add_argument(
        "--min-side",
        type=int,
        default=2048,
        help="Minimum length in pixels of the shorter image side (default: 2048, the tile size).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help=f"Delete rejected files instead of moving them to <src>/{REJECTED_DIR_NAME}/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be removed without changing anything.",
    )
    args = parser.parse_args()

    src_root = args.src.resolve()
    images_dir = src_root / "images"
    labels_dir = src_root / "labels"

    if RAW_DIR_NAME in src_root.parts:
        raise SystemExit(
            f"Refusing to modify raw data: {src_root} is under {RAW_DIR_NAME}/.\n"
            "Run clean_to_x_classes.py first and point --src at its output."
        )
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Expected images/ and labels/ under {src_root}")
    if args.min_side <= 0:
        raise SystemExit("--min-side must be a positive number of pixels.")

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    iterable = image_paths
    if tqdm is not None:
        iterable = tqdm(image_paths, desc="Checking resolutions", unit="image")

    rejected: list[tuple[Path, int, int]] = []
    unreadable: list[Path] = []
    kept_resolutions: Counter = Counter()
    rejected_resolutions: Counter = Counter()

    for img_path in iterable:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except OSError:
            unreadable.append(img_path)
            continue
        if min(width, height) < args.min_side:
            rejected.append((img_path, width, height))
            rejected_resolutions[(width, height)] += 1
        else:
            kept_resolutions[(width, height)] += 1

    if args.dry_run:
        action = "Would remove"
    elif args.delete:
        action = "Deleted"
    else:
        action = "Moved aside"

    rejected_images_dir = src_root / REJECTED_DIR_NAME / "images"
    rejected_labels_dir = src_root / REJECTED_DIR_NAME / "labels"
    if rejected and not args.dry_run and not args.delete:
        rejected_images_dir.mkdir(parents=True, exist_ok=True)
        rejected_labels_dir.mkdir(parents=True, exist_ok=True)

    removed_labels = 0
    for img_path, width, height in rejected:
        label_path = labels_dir / f"{img_path.stem}.txt"
        has_label = label_path.exists()
        removed_labels += int(has_label)
        print(f"{action}: {img_path.name} ({width}x{height})")
        if args.dry_run:
            continue
        if args.delete:
            img_path.unlink()
            if has_label:
                label_path.unlink()
        else:
            shutil.move(str(img_path), str(rejected_images_dir / img_path.name))
            if has_label:
                shutil.move(str(label_path), str(rejected_labels_dir / label_path.name))

    print()
    print(f"Dataset   : {src_root}")
    print(f"Threshold : shorter side >= {args.min_side} px")
    print(f"Images checked : {len(image_paths)}")
    print(f"Images kept    : {sum(kept_resolutions.values())}")
    print(f"{action:<15}: {len(rejected)} images, {removed_labels} labels")
    if unreadable:
        print(f"Unreadable images (left untouched): {len(unreadable)}")
        for img_path in unreadable:
            print(f"  {img_path.name}")
    print_resolutions("Resolutions kept:", kept_resolutions)
    print_resolutions("Resolutions below the threshold:", rejected_resolutions)
    if rejected and not args.dry_run and not args.delete:
        print(f"Rejected files are in: {src_root / REJECTED_DIR_NAME}")


if __name__ == "__main__":
    main()
