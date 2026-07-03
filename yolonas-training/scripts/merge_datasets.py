#!/usr/bin/env python3
"""Merge multiple YOLO datasets into one, remapping class ids if needed.

Usage:
  python scripts/merge_datasets.py \
    --datasets ../data/data-tiled/building-defect-5defect-tiled ../data/data-tiled/kelvin_bbox-5defect-tiled \
    --classes corrosion,crack,spall,plant_growth,crack_glass --out datasets/merged

Args:
  --datasets  One or more dataset roots (e.g. under ../data/data-tiled or
              ../data/data-5defect), each containing images/ and labels/.
  --classes   Comma-separated target class list; index is the class id.
             If omitted, uses the first dataset's data.yaml names.
  --out       Output dataset root (default: datasets/merged, a local scratch
              staging dir — feed this into copy_dataset.py --src to produce
              the final train/val/test split under yolonas-training/training-data/).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def read_yaml_lines(data_yaml: Path) -> list[str]:
    return data_yaml.read_text(encoding="utf-8").splitlines()


def read_names(data_yaml: Path) -> list[str]:
    names: list[str] = []
    in_names = False
    for raw in read_yaml_lines(data_yaml):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if line.startswith("- "):
                names.append(line[2:].strip())
            else:
                break
    if not names:
        raise ValueError(f"No names found in {data_yaml}")
    return names


def read_train_images_root(data_yaml: Path) -> Path:
    train_value = ""
    for raw in read_yaml_lines(data_yaml):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("train:"):
            train_value = line.split(":", 1)[1].strip()
            break
    if not train_value:
        raise ValueError(f"No train path found in {data_yaml}")
    train_path = Path(train_value)
    if train_path.is_absolute():
        return train_path.resolve()
    primary = (data_yaml.parent / train_path).resolve()
    if primary.exists():
        return primary
    fallback = (data_yaml.parent.parent / train_path).resolve()
    return fallback


def build_id_map(source_names: list[str], target_names: list[str]) -> dict[int, int]:
    target_index = {name: idx for idx, name in enumerate(target_names)}
    mapping: dict[int, int] = {}
    for src_id, name in enumerate(source_names):
        if name not in target_index:
            raise ValueError(f"Class '{name}' not found in target classes.")
        mapping[src_id] = target_index[name]
    return mapping


def remap_label_file(src: Path, dst: Path, id_map: dict[int, int]) -> None:
    lines = []
    for raw in src.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        class_id = int(parts[0])
        if class_id not in id_map:
            raise ValueError(f"Class id {class_id} not in mapping for {src}")
        parts[0] = str(id_map[class_id])
        lines.append(" ".join(parts))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def unique_name(stem: str, suffix: str, used: set[str]) -> str:
    base = f"{stem}_{suffix}"
    candidate = base
    counter = 1
    while candidate in used:
        counter += 1
        candidate = f"{base}_{counter}"
    used.add(candidate)
    return candidate


def merge_dataset(
    dataset_dir: Path,
    out_images: Path,
    out_labels: Path,
    id_map: dict[int, int],
    used_stems: set[str],
) -> None:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml in {dataset_dir}")

    images_root = read_train_images_root(data_yaml)
    if images_root.name != "images":
        raise ValueError(
            f"Expected train path to end with images/, got {images_root}"
        )

    images_dir = images_root
    labels_dir = images_root.parent / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected train/images and train/labels in {dataset_dir}"
        )

    label_map = {p.stem: p for p in labels_dir.glob("*.txt")}
    dataset_suffix = dataset_dir.name
    image_files = [p for p in images_dir.iterdir() if p.is_file()]
    total = 0
    for img in image_files:
        if label_map.get(img.stem) is not None:
            total += 1

    copied = 0
    for img in image_files:
        if not img.is_file():
            continue
        label = label_map.get(img.stem)
        if label is None:
            continue
        new_stem = unique_name(img.stem, dataset_suffix, used_stems)
        dst_img = out_images / f"{new_stem}{img.suffix}"
        dst_lbl = out_labels / f"{new_stem}.txt"
        shutil.copy2(img, dst_img)
        remap_label_file(label, dst_lbl, id_map)
        copied += 1
        if total > 0:
            bar_width = 40
            filled = int(bar_width * copied / total)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"\rMerging {dataset_dir.name}: [{bar}] {copied}/{total}",
                end="",
                flush=True,
            )
    if total > 0:
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge multiple YOLO datasets into one."
    )
    parser.add_argument(
        "--datasets",
        required=True,
        nargs="+",
        help="Dataset roots containing images/ and labels/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dataset root. Defaults to yolonas-training/scripts/../datasets/merged, "
        "anchored to this script's location regardless of CWD.",
    )
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated target class list (optional)",
    )
    args = parser.parse_args()

    dataset_dirs = [Path(p) for p in args.datasets]
    out_root = args.out if args.out is not None else Path(__file__).resolve().parent.parent / "datasets" / "merged"
    out_images = out_root / "train" / "images"
    out_labels = out_root / "train" / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    target_names: list[str] = []
    if args.classes:
        target_names = [c.strip() for c in args.classes.split(",") if c.strip()]

    used_stems: set[str] = set()
    for dataset_dir in dataset_dirs:
        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"Missing data.yaml in {dataset_dir}")
        source_names = read_names(data_yaml)
        if not target_names:
            target_names = source_names
        id_map = build_id_map(source_names, target_names)
        merge_dataset(dataset_dir, out_images, out_labels, id_map, used_stems)

    if not target_names:
        raise ValueError("Target classes are empty.")
    data_yaml = "\n".join(
        [
            "names:",
            *[f"- {name}" for name in target_names],
            f"nc: {len(target_names)}",
            f"train: {out_root.name}/train/images",
        ]
    )
    (out_root / "data.yaml").write_text(data_yaml + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
