#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Script: Build training-data/{train,val,test} straight from one or more datasets
#
# What it does:
# - Takes one or more dataset folders (each with images/, labels/ and
#   classes.txt, e.g. ../data/data-tiled/*) and writes the train/val/test split
#   directly into training-data/, with no intermediate merged dataset.
# - Replaces the old merge_datasets.py + split_dataset.py pair.
# - Checks everything before touching any file. It stops with an error if:
#     1. a dataset is missing images/, labels/ or classes.txt
#        (every missing one is listed);
#     2. the classes.txt files are not all identical (same names, same order;
#        the error groups the datasets by their class list);
#     3. two datasets share a folder name;
#     4. a label row is not a plain YOLO box row: exactly 5 fields
#        (class cx cy w h), a whole-number class id inside the shared list,
#        and four numbers between 0 and 1. Polygon / oriented-box rows (more
#        than 5 fields) and COCO JSON are not supported: the trainer reads
#        5-field rows only, so convert those to boxes before this step.
# - Splits by source photo, per dataset: all tiles cut from one photo
#   (<photo>_s<size>_x<n>_y<n>) land in the same split, and every dataset
#   contributes its own share to val/test. A dataset's split depends only on
#   --seed and its folder name, so adding another dataset later does not
#   reshuffle it.
# - Clears <dst>/{train,val,test}/{images,labels} (keeping .gitkeep files) and
#   deletes the old split_manifest.csv and split_report.txt, then copies each
#   image and its label unchanged, named <file>_<dataset>. classes.txt is kept
#   and only rewritten if the class list has changed.
# - Writes <dst>/classes.txt, <dst>/split_manifest.csv (split, dataset,
#   source photo, file) and <dst>/split_report.txt.
# - Prints a report (also saved as split_report.txt): the class distribution
#   (boxes per class per split, each split's share of the class against the
#   --val-ratio/--test-ratio target, and photos/tiles containing each class),
#   photos/tiles per dataset per split, boxes per class per dataset, warnings
#   (a class with no boxes, missing from train or val, or more than 2
#   percentage points off target), and the class id mapping used for training.
# - This split is by photo per dataset, so each class's share usually lands
#   near, not on, the target. For a split that targets every class's share
#   directly, use build_training_data_class_balanced.py instead.
#
# How to use (paths are relative to the current directory; examples assume you
# are in yolonas-training/):
# - Default run (80/20 train/val, no test set, seed 42):
#   python scripts/build_training_data.py \
#     --datasets ../data/data-tiled/building-defect-5defect-tiled \
#                ../data/data-tiled/kelvin_bbox-5defect-tiled \
#                ../data/data-tiled/tf_r2-5defect-tiled
# - Check the datasets and preview the split without writing anything:
#   python scripts/build_training_data.py --datasets ... --dry-run
#   (the dry run prints the class distribution report too; it is only saved on a real run)
# - Hold out a test set as well (80/10/10):
#   python scripts/build_training_data.py --datasets ... --val-ratio 0.1 --test-ratio 0.1
# - Write somewhere other than yolonas-training/training-data:
#   python scripts/build_training_data.py --datasets ... --dst /path/to/folder
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # Fallback if tqdm isn't installed.
    tqdm = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
# Suffix added by tile_building_defect_images.py: <photo>_s<size>_x<x0>_y<y0>
TILE_SUFFIX_RE = re.compile(r"_s\d+_x\d+_y\d+$")
MAX_EXAMPLES = 10
# Allowed deviation (percentage points) before a class is flagged in the report.
OFF_TARGET_WARNING_PP = 2.0
# Classes found in fewer photos than this are flagged as hard to split reliably.
FEW_PHOTOS_WARNING = 10


@dataclass
class Item:
    image: Path
    label: Path
    photo: str
    class_ids: list[int]


@dataclass
class Dataset:
    root: Path
    name: str
    items: list[Item] = field(default_factory=list)
    images_without_label: int = 0
    labels_without_image: int = 0
    splits: dict[str, list[Item]] = field(default_factory=dict)


def read_classes(classes_path: Path) -> list[str]:
    return [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_layout_and_classes(dataset_dirs: list[Path]) -> list[str]:
    """Rules 1-3: every dataset is complete, all classes.txt are identical, names are unique."""
    problems: list[str] = []

    structure_lines = []
    classes_by_dataset: dict[Path, tuple[str, ...]] = {}
    missing_classes = []
    for root in dataset_dirs:
        if not root.is_dir():
            structure_lines.append(f"    {root}: folder not found")
            continue
        for sub in ("images", "labels"):
            if not (root / sub).is_dir():
                structure_lines.append(f"    {root}: missing {sub}/")
            elif any(p.is_dir() for p in (root / sub).iterdir()):
                structure_lines.append(f"    {root}: {sub}/ contains subfolders (must be flat)")
        classes_path = root / "classes.txt"
        if not classes_path.is_file():
            missing_classes.append(root)
            continue
        names = read_classes(classes_path)
        if not names:
            structure_lines.append(f"    {classes_path}: file is empty")
            continue
        classes_by_dataset[root] = tuple(names)

    if structure_lines:
        problems.append("Datasets with a missing or invalid layout:\n" + "\n".join(structure_lines))
    if missing_classes:
        problems.append(
            "Missing classes.txt (every dataset must have one):\n"
            + "\n".join(f"    {root / 'classes.txt'}" for root in missing_classes)
        )

    groups: dict[tuple[str, ...], list[Path]] = {}
    for root, names in classes_by_dataset.items():
        groups.setdefault(names, []).append(root)
    if len(groups) > 1:
        lines = ["classes.txt files do not match (all must list the same names in the same order):"]
        for names, roots in sorted(groups.items(), key=lambda item: -len(item[1])):
            lines.append(f"    [{', '.join(names)}]")
            lines.extend(f"        {root / 'classes.txt'}" for root in roots)
        problems.append("\n".join(lines))

    by_name: dict[str, list[Path]] = {}
    for root in dataset_dirs:
        by_name.setdefault(root.name, []).append(root)
    duplicates = {name: roots for name, roots in by_name.items() if len(roots) > 1}
    if duplicates:
        lines = ["Datasets share a folder name (output files are named <file>_<dataset>, so names must be unique):"]
        for name, roots in duplicates.items():
            lines.append(f"    {name}: " + ", ".join(str(root) for root in roots))
        problems.append("\n".join(lines))

    return problems


def check_yolo_row(parts: list[str], num_classes: int) -> str | None:
    """Return what is wrong with a label row, or None for a valid YOLO box row
    (class cx cy w h: 5 fields, class id in the class list, values within 0-1)."""
    if len(parts) != 5:
        return f"expected 5 fields (class cx cy w h), found {len(parts)}"
    try:
        class_id = int(parts[0])
    except ValueError:
        return f"class id '{parts[0]}' is not a whole number"
    if not 0 <= class_id < num_classes:
        return f"class id {class_id} is outside the class list (0-{num_classes - 1})"
    try:
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return f"box values are not all numbers: {' '.join(parts[1:])}"
    if not all(0.0 <= value <= 1.0 for value in values):
        return f"box values must be between 0 and 1: {' '.join(parts[1:])}"
    return None


def scan_dataset(root: Path, num_classes: int) -> tuple[Dataset, list[str], list[str]]:
    """Pair images with labels and validate every label row (rule 4)."""
    dataset = Dataset(root=root, name=root.name)
    bad_rows: list[str] = []
    duplicate_stems: list[str] = []

    image_paths = sorted(
        p for p in (root / "images").iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    label_by_stem = {p.stem: p for p in (root / "labels").glob("*.txt")}
    stem_counts = Counter(p.stem for p in image_paths)
    duplicate_stems = sorted(stem for stem, count in stem_counts.items() if count > 1)
    dataset.labels_without_image = len(set(label_by_stem) - set(stem_counts))

    for img_path in image_paths:
        label_path = label_by_stem.get(img_path.stem)
        if label_path is None:
            dataset.images_without_label += 1
            continue

        class_ids: list[int] = []
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if not parts:
                continue
            problem = check_yolo_row(parts, num_classes)
            if problem is not None:
                bad_rows.append(f"{label_path}:{line_no}: {problem}")
                continue
            class_ids.append(int(parts[0]))

        dataset.items.append(
            Item(
                image=img_path,
                label=label_path,
                photo=TILE_SUFFIX_RE.sub("", img_path.stem),
                class_ids=class_ids,
            )
        )

    return dataset, bad_rows, duplicate_stems


def split_counts(num_photos: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    """Number of photos for val and test; whatever is left goes to train."""
    num_val = round(num_photos * val_ratio)
    num_test = round(num_photos * test_ratio)
    # A dataset with enough photos always contributes at least one to each requested split.
    if val_ratio > 0 and num_photos >= 2:
        num_val = max(1, num_val)
    if test_ratio > 0 and num_photos >= 3:
        num_test = max(1, num_test)
    # Always leave at least one photo for train.
    while num_val + num_test > num_photos - 1 and (num_val or num_test):
        if num_test >= num_val and num_test:
            num_test -= 1
        else:
            num_val -= 1
    return num_val, num_test


def assign_splits(dataset: Dataset, val_ratio: float, test_ratio: float, seed: int) -> None:
    items_by_photo: dict[str, list[Item]] = {}
    for item in dataset.items:
        items_by_photo.setdefault(item.photo, []).append(item)

    photos = sorted(items_by_photo)
    # Seeded per dataset, so one dataset's split does not change when others are added.
    random.Random(f"{seed}:{dataset.name}").shuffle(photos)

    num_val, num_test = split_counts(len(photos), val_ratio, test_ratio)
    photo_split = {}
    for index, photo in enumerate(photos):
        if index < num_val:
            photo_split[photo] = "val"
        elif index < num_val + num_test:
            photo_split[photo] = "test"
        else:
            photo_split[photo] = "train"

    dataset.splits = {split: [] for split in SPLITS}
    for photo, items in items_by_photo.items():
        dataset.splits[photo_split[photo]].extend(items)


def clear_split_dirs(dst_root: Path) -> None:
    """Empty <dst>/{train,val,test}/{images,labels}, keeping .gitkeep files, and delete
    the old split_manifest.csv and split_report.txt, which describe the split being removed."""
    for split in SPLITS:
        for sub in ("images", "labels"):
            folder = dst_root / split / sub
            folder.mkdir(parents=True, exist_ok=True)
            for entry in folder.iterdir():
                if entry.name == ".gitkeep":
                    continue
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
    # Removed together with the split they describe, so a build that fails partway
    # leaves no report that looks valid. The new ones are written at the end of a build.
    for stale in ("split_manifest.csv", "split_report.txt"):
        (dst_root / stale).unlink(missing_ok=True)


def write_outputs(datasets: list[Dataset], class_names: list[str], dst_root: Path) -> None:
    clear_split_dirs(dst_root)

    jobs = [
        (dataset, split, item)
        for dataset in datasets
        for split in SPLITS
        for item in dataset.splits[split]
    ]
    iterable = jobs
    if tqdm is not None:
        iterable = tqdm(jobs, desc="Copying", unit="image")

    manifest_rows = []
    for dataset, split, item in iterable:
        out_stem = f"{item.image.stem}_{dataset.name}"
        out_image = dst_root / split / "images" / f"{out_stem}{item.image.suffix}"
        shutil.copy2(item.image, out_image)
        shutil.copy2(item.label, dst_root / split / "labels" / f"{out_stem}.txt")
        manifest_rows.append((split, dataset.name, item.photo, out_image.name))

    # Leave an identical classes.txt untouched so a rebuild shows no diff in git.
    classes_path = dst_root / "classes.txt"
    if not classes_path.is_file() or read_classes(classes_path) != class_names:
        classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    with (dst_root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "dataset", "source_photo", "file"])
        writer.writerows(sorted(manifest_rows))


def format_table(headers: list[str], rows: list[list], indent: str = "  ") -> list[str]:
    cells = [headers] + [[f"{v:,}" if isinstance(v, int) else str(v) for v in row] for row in rows]
    widths = [max(len(row[col]) for row in cells) for col in range(len(headers))]
    lines = []
    for row in cells:
        first = row[0].ljust(widths[0])
        rest = "  ".join(cell.rjust(widths[col + 1]) for col, cell in enumerate(row[1:]))
        lines.append(f"{indent}{first}  {rest}")
    return lines


def build_report(datasets: list[Dataset], class_names: list[str], ratios: dict[str, float]) -> list[str]:
    """Report lines: class distribution first, then per-dataset tables, warnings, class ids."""
    used_splits = [split for split in SPLITS if ratios.get(split, 0) > 0]
    held_out = [split for split in used_splits if split != "train"]
    split_items = {split: [item for dataset in datasets for item in dataset.splits[split]] for split in SPLITS}

    def box_counts(items: list[Item]) -> Counter:
        counts: Counter = Counter()
        for item in items:
            counts.update(item.class_ids)
        return counts

    boxes = {split: box_counts(split_items[split]) for split in SPLITS}
    totals = box_counts([item for dataset in datasets for item in dataset.items])
    lines: list[str] = []
    warnings: list[str] = []

    # --- class distribution, in boxes ---
    lines.append("Class distribution (boxes per class; share = that split's part of the class's boxes):")
    headers = ["class"] + [f"{split} boxes" for split in used_splits] + ["total"]
    for split in held_out:
        headers += [f"{split} share", "target", "off by"]
    rows = []
    for class_id, name in enumerate(class_names):
        row = [name] + [boxes[split][class_id] for split in used_splits] + [totals[class_id]]
        for split in held_out:
            share = 100.0 * boxes[split][class_id] / totals[class_id] if totals[class_id] else 0.0
            target = 100.0 * ratios[split]
            off_by = share - target
            row += [f"{share:.1f}%", f"{target:.0f}%", "0.0 pp" if abs(off_by) < 0.05 else f"{off_by:+.1f} pp"]
            if totals[class_id] and abs(share - target) > OFF_TARGET_WARNING_PP:
                warnings.append(
                    f"class '{name}': {split} share {share:.1f}% is {abs(share - target):.1f} pp from the {target:.0f}% target"
                )
        rows.append(row)
    lines += format_table(headers, rows)

    # --- photos and tiles containing each class ---
    lines.append("")
    lines.append("Photos / tiles containing each class:")
    photos_with = {split: Counter() for split in SPLITS}
    tiles_with = {split: Counter() for split in SPLITS}
    for split in SPLITS:
        seen: dict[tuple[str, str], set[int]] = {}
        for dataset in datasets:
            for item in dataset.splits[split]:
                classes_here = set(item.class_ids)
                tiles_with[split].update(classes_here)
                seen.setdefault((dataset.name, item.photo), set()).update(classes_here)
        for classes_here in seen.values():
            photos_with[split].update(classes_here)
    rows = []
    for class_id, name in enumerate(class_names):
        row = [name] + [f"{photos_with[s][class_id]:,} / {tiles_with[s][class_id]:,}" for s in used_splits]
        rows.append(row)
        total_photos = sum(photos_with[s][class_id] for s in SPLITS)
        if 0 < total_photos < FEW_PHOTOS_WARNING:
            warnings.append(f"class '{name}' appears in only {total_photos} photo{'' if total_photos == 1 else 's'}, too few to split reliably")
    lines += format_table(["class"] + [f"{s} photos / tiles" for s in used_splits], rows)

    # --- per dataset ---
    lines.append("")
    lines.append("Photos and tiles per dataset (photos / tiles):")
    rows = []
    split_totals = {split: [0, 0] for split in SPLITS}
    for dataset in datasets:
        row = [dataset.name]
        for split in SPLITS:
            items = dataset.splits[split]
            photos = len({item.photo for item in items})
            split_totals[split][0] += photos
            split_totals[split][1] += len(items)
            row.append(f"{photos:,} / {len(items):,}")
        rows.append(row)
    rows.append(["total"] + [f"{split_totals[s][0]:,} / {split_totals[s][1]:,}" for s in SPLITS])
    lines += format_table(["dataset", *SPLITS], rows)

    lines.append("")
    lines.append("Boxes per class per dataset:")
    rows = []
    for dataset in datasets:
        counts = box_counts(dataset.items)
        rows.append([dataset.name] + [counts[class_id] for class_id in range(len(class_names))])
    lines += format_table(["dataset", *class_names], rows)

    # --- warnings ---
    for dataset in datasets:
        if dataset.images_without_label:
            warnings.append(
                f"{dataset.name}: images with no label file (skipped): {dataset.images_without_label}"
            )
        if dataset.labels_without_image:
            warnings.append(f"{dataset.name}: label files with no image: {dataset.labels_without_image}")
        if ratios.get("val", 0) > 0 and not dataset.splits["val"]:
            warnings.append(f"{dataset.name}: contributes nothing to val")
    for class_id, name in enumerate(class_names):
        if totals[class_id] == 0:
            warnings.append(f"class '{name}' has no boxes in any dataset")
        elif boxes["train"][class_id] == 0:
            warnings.append(f"class '{name}' has no boxes in train")
        elif ratios.get("val", 0) > 0 and boxes["val"][class_id] == 0:
            warnings.append(f"class '{name}' has no boxes in val")

    lines.append("")
    if warnings:
        lines.append("Warnings:")
        lines += [f"  - {warning}" for warning in warnings]
    else:
        lines.append("Warnings: none")

    lines.append("")
    lines.append("Class ids used for training:")
    for class_id, name in enumerate(class_names):
        lines.append(f"  {class_id}: {name:<20} {totals[class_id]:>9,} boxes")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training-data/{train,val,test} directly from one or more datasets "
        "that share an identical classes.txt."
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        required=True,
        help="Dataset roots, each containing images/, labels/ and classes.txt.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Share of photos for val (default: 0.2).")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Share of photos for test (default: 0).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split (default: 42).")
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Destination root for train/val/test. Defaults to yolonas-training/training-data, "
        "anchored to this script's location regardless of the current directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every check and print the report without clearing or copying anything.",
    )
    args = parser.parse_args()

    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("--val-ratio and --test-ratio must be 0 or more and add up to less than 1.")

    dataset_dirs = [path.resolve() for path in args.datasets]
    dst_root = (
        args.dst.resolve()
        if args.dst is not None
        else Path(__file__).resolve().parent.parent / "training-data"
    )
    for root in dataset_dirs:
        if dst_root == root or dst_root in root.parents or root in dst_root.parents:
            raise SystemExit(f"--dst ({dst_root}) overlaps with a dataset ({root}); choose another folder.")

    # ---- Checks: nothing is written until all of them pass ----
    problems = check_layout_and_classes(dataset_dirs)
    if problems:
        raise SystemExit("ERROR: the datasets cannot be combined.\n\n" + "\n\n".join(problems))

    class_names = read_classes(dataset_dirs[0] / "classes.txt")

    datasets: list[Dataset] = []
    label_problems: list[str] = []
    iterable = dataset_dirs
    if tqdm is not None:
        iterable = tqdm(dataset_dirs, desc="Checking labels", unit="dataset")
    for root in iterable:
        dataset, bad_rows, duplicate_stems = scan_dataset(root, len(class_names))
        datasets.append(dataset)
        if bad_rows:
            lines = [
                f"{root.name}: {len(bad_rows)} label rows are not valid YOLO box rows "
                "(5 fields: class cx cy w h, class id in classes.txt, values within 0-1):"
            ]
            lines.extend(f"    {row}" for row in bad_rows[:MAX_EXAMPLES])
            if len(bad_rows) > MAX_EXAMPLES:
                lines.append(f"    ... and {len(bad_rows) - MAX_EXAMPLES} more")
            label_problems.append("\n".join(lines))
        if duplicate_stems:
            label_problems.append(
                f"{root.name}: {len(duplicate_stems)} image names are used by more than one file extension, "
                f"e.g. {', '.join(duplicate_stems[:3])}"
            )
        if not dataset.items:
            label_problems.append(f"{root.name}: no image/label pairs found")
    if label_problems:
        raise SystemExit("ERROR: the datasets cannot be combined.\n\n" + "\n\n".join(label_problems))

    # ---- Split ----
    for dataset in datasets:
        assign_splits(dataset, args.val_ratio, args.test_ratio, args.seed)

    # ---- Write ----
    if not args.dry_run:
        write_outputs(datasets, class_names, dst_root)

    ratios = {"train": 1.0 - args.val_ratio - args.test_ratio, "val": args.val_ratio, "test": args.test_ratio}
    report = [
        f"Report from  : build_training_data.py, {datetime.now():%Y-%m-%d %H:%M}",
        f"Destination  : {dst_root}",
        f"Split        : by source photo per dataset, val {args.val_ratio}, test {args.test_ratio}, seed {args.seed}",
        f"Datasets     : {', '.join(dataset.name for dataset in datasets)}",
        "",
        *build_report(datasets, class_names, ratios),
    ]
    print()
    print("\n".join(report))
    print()
    if args.dry_run:
        print("Dry run: all checks passed; nothing was cleared or copied.")
    else:
        (dst_root / "split_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"Training data written to: {dst_root}")
        print(f"Report saved to: {dst_root / 'split_report.txt'}")


if __name__ == "__main__":
    main()
