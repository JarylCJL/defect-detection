#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Script: Build training-data/{train,val,test} with a class-balanced split
#
# What it does:
# - An alternative to build_training_data.py: same inputs, same checks, same
#   output files, but it chooses the train/val/test split so that every defect
#   class is split at the target ratio (e.g. 80/20), measured in boxes, instead
#   of splitting each dataset's photos at that ratio. Run one script or the
#   other; train.ipynb works the same with either. This script is
#   self-contained and does not import build_training_data.py.
# - Takes one or more dataset folders (each with images/, labels/ and
#   classes.txt, e.g. ../data/data-tiled/*) and writes the split directly into
#   training-data/.
# - Checks everything before touching any file. It stops with an error if:
#     1. a dataset is missing images/, labels/ or classes.txt
#        (every missing one is listed);
#     2. the classes.txt files are not all identical (same names, same order;
#        the error groups the datasets by their class list);
#     3. two datasets share a folder name;
#     4. a label row is not a plain YOLO box row: exactly 5 fields
#        (class cx cy w h), a whole-number class id inside the shared list,
#        and four numbers between 0 and 1.
# - How the split is chosen (best effort):
#     * Source photos are kept whole: all tiles cut from one photo
#       (<photo>_s<size>_x<n>_y<n>) go to the same split, so overlapping tiles
#       never leak between train and val. Because a photo usually holds
#       several classes, an exact per-class ratio is not always possible.
#     * Target: for every class, the share of its boxes in val (and test) equals
#       --val-ratio (and --test-ratio). Boxes are counted on the tiles, as the
#       model sees them.
#     * Photos are first placed rarest-class-first, each going to the split
#       that most needs that class (iterative stratification). The split is
#       then refined by moving and swapping single photos while that brings
#       the classes closer to target. As a secondary goal, every dataset keeps
#       roughly its share of photos in each split, and a dataset with at least
#       two photos always keeps at least one in train and one in val.
#     * The same --seed gives the same split. Unlike build_training_data.py,
#       adding or removing a dataset can move other datasets' photos between
#       splits, because all classes are balanced across all datasets together.
# - Clears <dst>/{train,val,test}/{images,labels} (keeping .gitkeep files) and
#   deletes the old split_manifest.csv and split_report.txt, then copies each
#   image and its label unchanged, named <file>_<dataset>. classes.txt is kept
#   and only rewritten if the class list has changed.
# - Writes <dst>/classes.txt, <dst>/split_manifest.csv (split, dataset,
#   source photo, file) and <dst>/split_report.txt.
# - Prints a report (also saved as split_report.txt): the class distribution
#   (boxes per class per split, achieved share against target, how far off, and
#   photos/tiles containing each class), photos/tiles per dataset, boxes per
#   class per dataset, warnings, and the class id mapping used for training.
#
# Things to know:
# - Best effort, not exact. A class concentrated in very few photos cannot be
#   split at the target, because photos are never cut in two. For example, if
#   all of a class's boxes come from one photo, that class ends up entirely in
#   train or entirely in val. The report flags any class more than 2
#   percentage points off target, and any class found in fewer than 10 photos.
# - The target is in boxes, so photo and tile counts drift from it. Photos with
#   many boxes are the ones most often moved to balance the classes, so val can
#   hold a larger share of photos or tiles than --val-ratio (on the data in
#   October 2026: 20% of every class's boxes, 20.5% of the photos, about 24% of
#   the tiles). The report shows photos and tiles per class and per dataset.
# - Adding or removing a dataset can change the split of the others, because
#   all classes are balanced across all datasets together. Validation photos
#   can therefore change between builds, and scores from models trained on
#   different builds are not directly comparable. When the same validation
#   photos must be kept across builds, use build_training_data.py, which splits
#   each dataset independently.
# - Datasets keep roughly, not exactly, their share of photos in each split;
#   class balance takes priority. A dataset with at least two photos always
#   keeps at least one in train and one in val; a one-photo dataset goes to train.
#
# How to use (paths are relative to the current directory; examples assume you
# are in yolonas-training/):
# - Default run (80/20 train/val per class, no test set, seed 42):
#   python scripts/build_training_data_class_balanced.py \
#     --datasets ../data/data-tiled/building-defect-5defect-tiled \
#                ../data/data-tiled/kelvin_bbox-5defect-tiled \
#                ../data/data-tiled/tf_r2-5defect-tiled
# - Preview the split and the class distribution report without writing anything:
#   python scripts/build_training_data_class_balanced.py --datasets ... --dry-run
# - Another ratio, e.g. 70/30 per class:
#   python scripts/build_training_data_class_balanced.py --datasets ... --val-ratio 0.3
# - Hold out a test set as well (80/10/10 per class):
#   python scripts/build_training_data_class_balanced.py --datasets ... --val-ratio 0.1 --test-ratio 0.1
# - Write somewhere other than yolonas-training/training-data:
#   python scripts/build_training_data_class_balanced.py --datasets ... --dst /path/to/folder
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



# Allowed deviation (percentage points) before a class is flagged in the report.
OFF_TARGET_WARNING_PP = 2.0
# Classes found in fewer photos than this are flagged as hard to split reliably.
FEW_PHOTOS_WARNING = 10
# Weight of the secondary goal (each dataset keeps roughly its share of photos per split).
DATASET_WEIGHT = 0.01
# Penalty for leaving a dataset with no photos in a split it should contribute to.
EMPTY_SPLIT_PENALTY = 1000.0
MAX_MOVE_PASSES = 100
MAX_SWAP_ROUNDS = 20
SWAP_TRIES_PER_ROUND = 20000


@dataclass
class PhotoGroup:
    """All tiles cut from one source photo: the unit that is assigned to a split."""
    dataset_index: int
    photo: str
    items: list[Item]
    boxes: list[int]  # boxes per class, summed over the photo's tiles


def active_splits(val_ratio: float, test_ratio: float) -> dict[str, float]:
    ratios = {"train": 1.0 - val_ratio - test_ratio, "val": val_ratio, "test": test_ratio}
    return {split: ratio for split, ratio in ratios.items() if ratio > 0}


def build_groups(datasets: list[Dataset], num_classes: int) -> list[PhotoGroup]:
    groups: list[PhotoGroup] = []
    for dataset_index, dataset in enumerate(datasets):
        items_by_photo: dict[str, list[Item]] = {}
        for item in dataset.items:
            items_by_photo.setdefault(item.photo, []).append(item)
        for photo in sorted(items_by_photo):
            boxes = [0] * num_classes
            for item in items_by_photo[photo]:
                for class_id in item.class_ids:
                    boxes[class_id] += 1
            groups.append(PhotoGroup(dataset_index, photo, items_by_photo[photo], boxes))
    return groups


class SplitState:
    """Current assignment plus running totals, so a move can be scored cheaply."""

    def __init__(self, groups: list[PhotoGroup], num_datasets: int, num_classes: int, ratios: dict[str, float]):
        self.groups = groups
        self.ratios = ratios
        self.splits = list(ratios)
        self.num_classes = num_classes
        totals = [sum(group.boxes[c] for group in groups) for c in range(num_classes)]
        self.class_targets = {s: [ratios[s] * totals[c] for c in range(num_classes)] for s in self.splits}
        photos_per_dataset = Counter(group.dataset_index for group in groups)
        self.photos_per_dataset = [photos_per_dataset[d] for d in range(num_datasets)]
        self.dataset_targets = {s: [ratios[s] * n for n in self.photos_per_dataset] for s in self.splits}
        # A dataset must contribute to a split if it has enough photos to do so.
        self.min_photos = {
            s: [1 if (s == "train" or n >= (2 if s == "val" else 3)) and n > 0 else 0 for n in self.photos_per_dataset]
            for s in self.splits
        }
        self.assignment: list[str | None] = [None] * len(groups)
        self.class_counts = {s: [0] * num_classes for s in self.splits}
        self.dataset_counts = {s: [0] * num_datasets for s in self.splits}

    def place(self, index: int, split: str | None) -> None:
        group = self.groups[index]
        old = self.assignment[index]
        if old is not None:
            for c, n in enumerate(group.boxes):
                self.class_counts[old][c] -= n
            self.dataset_counts[old][group.dataset_index] -= 1
        if split is not None:
            for c, n in enumerate(group.boxes):
                self.class_counts[split][c] += n
            self.dataset_counts[split][group.dataset_index] += 1
        self.assignment[index] = split

    def score(self) -> float:
        """Lower is better: squared relative deviation of every class from its target
        share, plus a small term for dataset balance and a penalty for empty splits."""
        total = 0.0
        for s in self.splits:
            for c in range(self.num_classes):
                target = self.class_targets[s][c]
                if target > 0:
                    total += ((self.class_counts[s][c] - target) / target) ** 2
            for d, target in enumerate(self.dataset_targets[s]):
                if target > 0:
                    total += DATASET_WEIGHT * ((self.dataset_counts[s][d] - target) / target) ** 2
                if self.dataset_counts[s][d] < self.min_photos[s][d]:
                    total += EMPTY_SPLIT_PENALTY
        return total


def greedy_assign(state: SplitState, rng: random.Random) -> None:
    """Iterative stratification: place photos rarest-class-first, each into the
    split with the largest remaining need for that class."""
    groups = state.groups
    need = {s: list(state.class_targets[s]) for s in state.splits}
    need_photos = {s: state.ratios[s] * len(groups) for s in state.splits}
    unassigned = set(range(len(groups)))

    def assign(index: int, split: str) -> None:
        state.place(index, split)
        for c, n in enumerate(groups[index].boxes):
            need[split][c] -= n
        need_photos[split] -= 1
        unassigned.discard(index)

    while unassigned:
        remaining = [0] * state.num_classes
        for index in unassigned:
            for c, n in enumerate(groups[index].boxes):
                remaining[c] += n
        present = [c for c in range(state.num_classes) if remaining[c] > 0]
        if not present:  # photos without boxes: spread them by photo share
            for index in sorted(unassigned):
                assign(index, max(state.splits, key=lambda s: need_photos[s]))
            break
        rarest = min(present, key=lambda c: (remaining[c], c))
        candidates = sorted(index for index in unassigned if groups[index].boxes[rarest] > 0)
        rng.shuffle(candidates)
        candidates.sort(key=lambda index: -groups[index].boxes[rarest])  # heaviest first, ties shuffled
        for index in candidates:
            split = max(state.splits, key=lambda s: (need[s][rarest], need_photos[s]))
            assign(index, split)


def refine(state: SplitState, rng: random.Random) -> None:
    """Local search: move or swap single photos while it lowers the score."""
    if len(state.splits) < 2:
        return
    indices = list(range(len(state.groups)))
    for _ in range(MAX_SWAP_ROUNDS):
        # 1. single-photo moves until none helps
        for _ in range(MAX_MOVE_PASSES):
            improved = False
            rng.shuffle(indices)
            current = state.score()
            for index in indices:
                old = state.assignment[index]
                best_split, best_score = old, current
                for split in state.splits:
                    if split == old:
                        continue
                    state.place(index, split)
                    candidate = state.score()
                    if candidate < best_score - 1e-12:
                        best_split, best_score = split, candidate
                    state.place(index, old)
                if best_split != old:
                    state.place(index, best_split)
                    current = best_score
                    improved = True
            if not improved:
                break
        # 2. random pairwise swaps between splits, to get out of local minima
        swapped = False
        current = state.score()
        for _ in range(SWAP_TRIES_PER_ROUND):
            a, b = rng.randrange(len(indices)), rng.randrange(len(indices))
            split_a, split_b = state.assignment[a], state.assignment[b]
            if split_a == split_b:
                continue
            state.place(a, split_b)
            state.place(b, split_a)
            candidate = state.score()
            if candidate < current - 1e-12:
                current = candidate
                swapped = True
            else:
                state.place(a, split_a)
                state.place(b, split_b)
        if not swapped:
            break


def assign_balanced_splits(datasets: list[Dataset], num_classes: int, val_ratio: float, test_ratio: float, seed: int) -> None:
    ratios = active_splits(val_ratio, test_ratio)
    groups = build_groups(datasets, num_classes)
    state = SplitState(groups, len(datasets), num_classes, ratios)
    rng = random.Random(f"class-balanced:{seed}")
    greedy_assign(state, rng)
    refine(state, rng)

    for dataset in datasets:
        dataset.splits = {split: [] for split in SPLITS}
    for group, split in zip(groups, state.assignment):
        datasets[group.dataset_index].splits[split].extend(group.items)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training-data/{train,val,test} from one or more datasets that share an identical "
        "classes.txt, choosing the split so that every class is split at the target ratio (in boxes)."
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        required=True,
        help="Dataset roots, each containing images/, labels/ and classes.txt.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Target share of every class's boxes in val (default: 0.2)."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.0, help="Target share of every class's boxes in test (default: 0)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed; the same seed gives the same split (default: 42)."
    )
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
        help="Run every check, choose the split and print the report without clearing or copying anything.",
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
    print("Choosing a class-balanced split...", flush=True)
    assign_balanced_splits(datasets, len(class_names), args.val_ratio, args.test_ratio, args.seed)
    ratios = {"train": 1.0 - args.val_ratio - args.test_ratio, "val": args.val_ratio, "test": args.test_ratio}

    # ---- Write ----
    if not args.dry_run:
        write_outputs(datasets, class_names, dst_root)

    report = [
        f"Report from  : build_training_data_class_balanced.py, {datetime.now():%Y-%m-%d %H:%M}",
        f"Destination  : {dst_root}",
        f"Split        : class-balanced (in boxes), target val {args.val_ratio}, test {args.test_ratio}, "
        f"seed {args.seed}, source photos kept whole",
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
