#!/usr/bin/env python3
"""Split a single source folder into training-data/{train,val,test}.

Example:
  python scripts/split_dataset.py \
    --src ../data/data-5defect/building-defect-5defect \
    --dst training-data
"""

import argparse
import random
import shutil
import sys
from pathlib import Path


DEFAULT_TRAIN_RATIO = 0.9
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.0
DEFAULT_SEED = 42


def normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return train_ratio / total, val_ratio / total, test_ratio / total


def split_source(
    src_dir: Path,
    dst_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    src_images = src_dir / "images"
    src_labels = src_dir / "labels"
    if not src_images.exists() or not src_labels.exists():
        raise FileNotFoundError(
            f"Expected images/ and labels/ under {src_dir}"
        )

    label_map = {p.stem: p for p in src_labels.glob("*.txt")}
    items = []
    for img in src_images.iterdir():
        if not img.is_file():
            continue
        label = label_map.get(img.stem)
        if label is None:
            continue
        items.append((img, label))

    if not items:
        print("No image/label pairs found to split.")
        return

    train_ratio, val_ratio, test_ratio = normalize_ratios(
        train_ratio, val_ratio, test_ratio
    )

    rng = random.Random(seed)
    rng.shuffle(items)

    total = len(items)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)

    assigned = train_count + val_count + test_count
    remainder = total - assigned
    if remainder > 0:
        # Always put leftover items into train.
        train_count += remainder

    splits = {
        "train": items[:train_count],
        "val": items[train_count : train_count + val_count],
        "test": items[train_count + val_count :],
    }

    for split, split_items in splits.items():
        dst_images = dst_root / split / "images"
        dst_labels = dst_root / split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        split_total = len(split_items)
        split_copied = 0
        for img, label in split_items:
            shutil.copy2(img, dst_images / img.name)
            shutil.copy2(label, dst_labels / label.name)
            split_copied += 1
            if split_total > 0:
                progress = split_copied / split_total
                bar_width = 40
                filled = int(bar_width * progress)
                bar = "#" * filled + "-" * (bar_width - filled)
                sys.stdout.write(
                    f"\rCopying {split}: [{bar}] {split_copied}/{split_total}"
                )
                sys.stdout.flush()

        if split_total > 0:
            sys.stdout.write("\n")
        print(f"{split}: copied {len(split_items)} image/label pairs")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split a single source folder into training-data/train, training-data/val, training-data/test"
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Path to the source folder containing images/ and labels/",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Destination root for train/val/test. Defaults to yolonas-training/training-data, "
        "anchored to this script's location regardless of CWD (not the shared top-level data/).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Train split ratio (default: {DEFAULT_TRAIN_RATIO})",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Val split ratio (default: {DEFAULT_VAL_RATIO})",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Test split ratio (default: {DEFAULT_TEST_RATIO})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for splitting (default: {DEFAULT_SEED})",
    )

    args = parser.parse_args()
    src_dir = Path(args.src)
    dst_root = args.dst if args.dst is not None else Path(__file__).resolve().parent.parent / "training-data"

    if not src_dir.exists():
        print(f"Source path not found: {src_dir}")
        return 1

    split_source(
        src_dir,
        dst_root,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
