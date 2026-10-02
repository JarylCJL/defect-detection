#!/usr/bin/env python3
"""
restore_selected_classes.py

Restores selected classes that cleaning removed, one dataset at a time:
  - Keeps corrosion, crack, spall labels from cleaned dataset versions
  - Adds desired labels (default: plant_growth, crack_glass) from original datasets
  - Only processes images that exist in both original and cleaned datasets

Desired labels are matched against the *raw* source dataset's own class
names (which may use a different naming convention, e.g. spaces instead of
underscores) but written to the output classes.txt using the pipeline's
canonical name (matching yolonas-training/training-data/classes.txt). See
DEFAULT_DESIRED_LABELS below for the (source_name, output_name) mapping.

Output class schema:
  0: corrosion
  1: crack
  2: spall
  3: plant_growth
  4: crack_glass

Usage:
    python scripts/restore_selected_classes.py
    python scripts/restore_selected_classes.py --base-dir /path/to/defect-detection/data
    python scripts/restore_selected_classes.py --desired-labels "plant growth=plant_growth" "crack glass=crack_glass"
"""

import argparse
import json
import shutil
from pathlib import Path

from tqdm import tqdm

# Dataset mappings: (original subdir, cleaned subdir, output name)
# Paths are relative to --base-dir, which now defaults to the shared top-level
# data/ directory (../../data from this script), with the raw source datasets
# under data_raw/ instead of data/.
DATASET_MAPPINGS = [
    (
        "data_raw/building-defect-detection-3.v3i.yolov7pytorch-001",
        "data-cleaned/building-defect-cleaned",
        "building-defect-5defect",
    ),
    (
        "data_raw/kelvin_bbox_all",
        "data-cleaned/kelvin_bbox_cleaned",
        "kelvin_bbox-5defect",
    ),
    (
        "data_raw/tf_rbb_all",
        "data-cleaned/tf_rbb_cleaned_half",
        "tf_rbb-5defect",
    ),
    (
        "data_raw/tf_r2",
        "data-cleaned/tf_r2_cleaned",
        "tf_r2-5defect",
    ),
]

# (source_name, output_name): source_name matches the raw dataset's own
# class list; output_name is what gets written into the produced classes.txt.
DEFAULT_DESIRED_LABELS = [
    ("plant growth", "plant_growth"),
    ("crack glass", "crack_glass"),
]

# Output class ordering: cleaned classes first, then desired labels in order
OUTPUT_BASE_CLASSES = ["corrosion", "crack", "spall"]


def read_classes(classes_file: Path) -> list[str]:
    """Read class names from a classes.txt file (one name per line)."""
    return [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]


def find_original_images_dir(original_dir: Path) -> Path:
    """Locate the images directory in the original dataset (flat or train/ structure)."""
    flat = original_dir / "images"
    if flat.is_dir():
        return flat
    train = original_dir / "train" / "images"
    if train.is_dir():
        return train
    raise FileNotFoundError(f"Cannot find images directory in {original_dir}")


def find_original_labels_dir(original_dir: Path) -> Path:
    """Locate the labels directory in the original dataset (flat or train/ structure)."""
    flat = original_dir / "labels"
    if flat.is_dir():
        return flat
    train = original_dir / "train" / "labels"
    if train.is_dir():
        return train
    raise FileNotFoundError(f"Cannot find labels directory in {original_dir}")


def build_label_lookup(labels_dir: Path) -> dict[str, Path]:
    """Build a stem -> Path mapping for all .txt label files in a directory."""
    return {p.stem: p for p in labels_dir.glob("*.txt")}


def process_dataset(
    original_dir: Path,
    cleaned_dir: Path,
    output_dir: Path,
    desired_labels: list[tuple[str, str]],
    output_classes: list[str],
) -> dict:
    """
    Process one dataset pair and write merged output.

    desired_labels is a list of (source_name, output_name) pairs: source_name
    is matched against the raw dataset's own class list, output_name is what
    ends up in the produced classes.txt / label files.

    Returns a stats dict with counts for reporting.
    """
    # Read class lists
    orig_classes_file = original_dir / "classes.txt"
    if not orig_classes_file.exists():
        # Fall back to data.yaml for Roboflow-style datasets
        orig_classes = []
    else:
        orig_classes = read_classes(orig_classes_file)

    cleaned_classes = read_classes(cleaned_dir / "classes.txt")

    # Map original class IDs directly to their output label name
    orig_class_to_id = {name: i for i, name in enumerate(orig_classes)}
    desired_orig_ids = {}
    for source_name, output_name in desired_labels:
        if source_name in orig_class_to_id:
            desired_orig_ids[orig_class_to_id[source_name]] = output_name

    # Map output label names to their output class IDs
    output_class_to_id = {name: i for i, name in enumerate(output_classes)}

    # Set up source directories
    cleaned_images_dir = cleaned_dir / "images"
    cleaned_labels_dir = cleaned_dir / "labels"
    orig_labels_dir = find_original_labels_dir(original_dir)

    # Build lookup of original label files by stem
    orig_label_lookup = build_label_lookup(orig_labels_dir)

    # Set up output directories
    out_images_dir = output_dir / "images"
    out_labels_dir = output_dir / "labels"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "images_processed": 0,
        "images_with_extra_labels": 0,
        "extra_annotations_added": 0,
        "desired_labels_not_in_original": [
            source_name for source_name, _ in desired_labels if source_name not in orig_class_to_id
        ],
    }

    # Process each image in the cleaned dataset
    images = sorted(cleaned_images_dir.iterdir())
    for img_path in tqdm(images, desc=output_dir.name, unit="img"):
        if not img_path.is_file():
            continue

        stem = img_path.stem
        cleaned_label_path = cleaned_labels_dir / (stem + ".txt")

        # Read cleaned labels (may be empty if image has no annotations)
        cleaned_lines = []
        if cleaned_label_path.exists():
            cleaned_lines = [l for l in cleaned_label_path.read_text().splitlines() if l.strip()]

        # Look for matching original label file
        extra_lines = []
        if stem in orig_label_lookup and desired_orig_ids:
            orig_label_path = orig_label_lookup[stem]
            for line in orig_label_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                orig_class_id = int(parts[0])
                if orig_class_id in desired_orig_ids:
                    label_name = desired_orig_ids[orig_class_id]
                    new_class_id = output_class_to_id[label_name]
                    extra_lines.append(f"{new_class_id} " + " ".join(parts[1:]))

        if extra_lines:
            stats["images_with_extra_labels"] += 1
            stats["extra_annotations_added"] += len(extra_lines)

        # Copy image
        shutil.copy2(img_path, out_images_dir / img_path.name)

        # Write merged label file
        all_lines = cleaned_lines + extra_lines
        out_label_path = out_labels_dir / (stem + ".txt")
        out_label_path.write_text("\n".join(all_lines) + ("\n" if all_lines else ""))

        stats["images_processed"] += 1

    # Write classes.txt
    (output_dir / "classes.txt").write_text("\n".join(output_classes) + "\n")

    # Write data.yaml alongside classes.txt (no pipeline step reads it any more;
    # build_training_data.py uses classes.txt)
    data_yaml_lines = [
        "names:",
        *[f"- {name}" for name in output_classes],
        f"nc: {len(output_classes)}",
        "train: images",
    ]
    (output_dir / "data.yaml").write_text("\n".join(data_yaml_lines) + "\n")

    # Write notes.json
    notes = {
        "categories": [{"id": i, "name": name} for i, name in enumerate(output_classes)],
        "info": {
            "year": 2026,
            "version": "1.0",
            "description": "5-defect consolidated dataset",
            "source_cleaned": str(cleaned_dir),
            "source_original": str(original_dir),
        },
    }
    (output_dir / "notes.json").write_text(json.dumps(notes, indent=2) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Restore selected classes dropped during cleaning: merge cleaned labels with desired labels from the raw originals."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Root data/ directory (shared across the consolidated repo). Defaults to ../../data relative to this script (i.e. <repo>/data).",
    )
    parser.add_argument(
        "--desired-labels",
        nargs="+",
        default=None,
        metavar="SOURCE_NAME[=OUTPUT_NAME]",
        help=(
            "Labels to extract from original datasets, matched by their name in the "
            "source dataset's own class list. Use SOURCE_NAME=OUTPUT_NAME to rename "
            "on output (e.g. 'plant growth=plant_growth'); a bare name is used for "
            f"both. Default: {DEFAULT_DESIRED_LABELS}"
        ),
    )
    args = parser.parse_args()

    # Resolve base directory: <repo>/data-cleaning/scripts/.. -> data-cleaning/.. -> <repo>, then /data
    if args.base_dir is not None:
        base_dir = args.base_dir.resolve()
    else:
        base_dir = Path(__file__).resolve().parents[2] / "data"

    if args.desired_labels is None:
        desired_labels = DEFAULT_DESIRED_LABELS
    else:
        desired_labels = [
            tuple(raw.split("=", 1)) if "=" in raw else (raw, raw) for raw in args.desired_labels
        ]

    output_classes = OUTPUT_BASE_CLASSES + [output_name for _, output_name in desired_labels]

    print(f"Base directory : {base_dir}")
    print(f"Desired labels : {desired_labels}")
    print(f"Output classes : {output_classes}")
    print()

    for orig_rel, cleaned_rel, output_name in DATASET_MAPPINGS:
        original_dir = base_dir / orig_rel
        cleaned_dir = base_dir / cleaned_rel
        output_dir = base_dir / "data-5defect" / output_name

        print(f"{'='*60}")
        print(f"Dataset : {output_name}")
        print(f"  Original : {original_dir}")
        print(f"  Cleaned  : {cleaned_dir}")
        print(f"  Output   : {output_dir}")

        if not original_dir.exists():
            print(f"  WARNING: original directory not found, skipping.")
            continue
        if not cleaned_dir.exists():
            print(f"  WARNING: cleaned directory not found, skipping.")
            continue

        stats = process_dataset(
            original_dir=original_dir,
            cleaned_dir=cleaned_dir,
            output_dir=output_dir,
            desired_labels=desired_labels,
            output_classes=output_classes,
        )

        print(f"  Images processed          : {stats['images_processed']}")
        print(f"  Images with extra labels  : {stats['images_with_extra_labels']}")
        print(f"  Extra annotations added   : {stats['extra_annotations_added']}")
        if stats["desired_labels_not_in_original"]:
            print(f"  Labels not in original    : {stats['desired_labels_not_in_original']}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
