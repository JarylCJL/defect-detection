#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Script: Clean a YOLO dataset down to a chosen set of classes
#
# What it does:
# - Reads images from --images and YOLO-format labels from --labels (both flat
#   folders, paired by filename stem). Or give --src, a dataset folder, and
#   --images, --labels and --classes are worked out from it (see "--src" below).
# - Accepts plain YOLO box rows only: exactly 5 fields (class cx cy w h), a
#   whole-number class id and four numbers between 0 and 1. Any other row
#   (e.g. polygon / oriented-box rows with more than 5 fields) is skipped and
#   counted in the summary as malformed; COCO JSON is not supported.
# - Reads the source dataset's class names from --classes (see "Class list
#   formats" below).
# - Normalises class names (trim, whitespace -> "_", lowercase), so
#   "plant growth" becomes "plant_growth". The same rule is applied to the
#   --keep values, so either spelling can be typed.
# - Keeps only the annotations whose class is listed in --keep and renumbers
#   them in --keep order (first name -> id 0, second -> id 1, ...).
# - Drops images (and their labels) left with no annotations after filtering.
# - Writes <dest>/images, <dest>/labels and <dest>/classes.txt. The source
#   folders are never modified.
# - Prints a summary: images kept/dropped, boxes per class, label problems.
#
# Class list formats (--classes):
# Only these 3 formats are accepted, chosen by file extension; any other
# extension stops with an error.
# - .txt (classes.txt): one class name per line, the line number (from 0) is
#   the class id. Comes with Label Studio YOLO exports (kelvin_bbox_all,
#   tf_rbb_all, tf_r2) and is written by this pipeline's own scripts.
# - .yaml / .yml (data.yaml): the `names:` entry, as a list (block or inline)
#   or an id -> name mapping. Comes with Roboflow / Ultralytics-style YOLO
#   exports (building-defect-detection-3, where it sits one level above
#   train/), and is written by restore_selected_classes.py.
# - .json (notes.json): the `categories` list of {"id", "name"} entries. Comes
#   with Label Studio YOLO exports, sometimes without a classes.txt
#   (Ray_CrackedGlass_Synthetic).
#
# --src (instead of --images / --labels / --classes):
# - --images = <src>/images, --labels = <src>/labels.
# - --classes = <src>/classes.txt and/or <src>/notes.json (as Label Studio
#   exports both). If both exist they must list the same classes in the same
#   order; if they disagree it stops, and you choose one with --classes. If
#   neither exists, data.yaml in <src> or one level up (Roboflow layout) is used.
# - Any of --images, --labels or --classes given explicitly overrides the
#   folder worked out from --src.
# - A folder of several Label Studio projects (one subfolder each) is not a
#   dataset: combine it first with combine_label_studio_projects.py.
#
# Safeguards:
# - Stops if two source class names become identical after normalisation.
# - Warns (and lists the available names) for each --keep name missing from the
#   source; stops only if none of them match.
# - Warns if the --keep order disagrees with
#   yolonas-training/training-data/classes.txt (ids are still assigned in the
#   order typed). build_training_data.py only combines datasets whose
#   classes.txt are identical, so keep the same --keep list and order for every
#   dataset you plan to train on together.
# - Refuses an existing, non-empty --dest unless --overwrite is passed.
# - Stops if --images or --labels contain subfolders.
#
# How to use (paths are relative to the current directory; examples assume you
# are in data-cleaning/):
# - 3 classes:
#   python scripts/clean_to_x_classes.py \
#     --images ../data/data_raw/tf_r2/images \
#     --labels ../data/data_raw/tf_r2/labels \
#     --classes ../data/data_raw/tf_r2/classes.txt \
#     --dest ../data/data-cleaned/tf_r2_cleaned \
#     --keep corrosion crack spall
# - 5 classes (names with spaces: use underscores, or quote them):
#   ... --keep corrosion crack spall plant_growth crack_glass
#   ... --keep corrosion crack spall "plant growth" "crack glass"
# - Roboflow-style dataset (class names live in data.yaml):
#   python scripts/clean_to_x_classes.py \
#     --images ../data/data_raw/<name>/train/images \
#     --labels ../data/data_raw/<name>/train/labels \
#     --classes ../data/data_raw/<name>/data.yaml \
#     --dest ../data/data-cleaned/<name>_cleaned \
#     --keep corrosion crack spall
# - Same, giving the dataset folder only (Label Studio export or the output of
#   combine_label_studio_projects.py):
#   python scripts/clean_to_x_classes.py --src ../data/data_raw/tf_rbb_all \
#     --dest ../data/data-5defect/tf_rbb-5defect \
#     --keep corrosion crack spall plant_growth crack_glass
# - Replace an existing output folder: add --overwrite
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

try:
    from tqdm import tqdm
except ImportError:  # Fallback if tqdm isn't installed.
    tqdm = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# <repo>/data-cleaning/scripts/.. -> data-cleaning/.. -> <repo>
CANONICAL_CLASSES_PATH = (
    Path(__file__).resolve().parents[2] / "yolonas-training" / "training-data" / "classes.txt"
)


def normalize_name(name: str) -> str:
    """Trim, collapse each run of whitespace into one underscore, lowercase."""
    return "_".join(str(name).split()).lower()


def load_source_classes(classes_path: Path) -> dict[int, str]:
    """Read a class list as {class id: raw name} from classes.txt, data.yaml or notes.json."""
    suffix = classes_path.suffix.lower()
    text = classes_path.read_text(encoding="utf-8")

    if suffix == ".txt":
        names = [line.strip() for line in text.splitlines() if line.strip()]
        classes = dict(enumerate(names))
    elif suffix in {".yaml", ".yml"}:
        names = (yaml.safe_load(text) or {}).get("names")
        if isinstance(names, dict):
            classes = {int(class_id): str(name) for class_id, name in names.items()}
        elif isinstance(names, list):
            classes = {class_id: str(name) for class_id, name in enumerate(names)}
        else:
            raise SystemExit(f"No `names:` list found in {classes_path}")
    elif suffix == ".json":
        categories = json.loads(text).get("categories")
        if not categories:
            raise SystemExit(f"No `categories` list found in {classes_path}")
        classes = {int(cat["id"]): str(cat["name"]) for cat in categories}
    else:
        raise SystemExit(
            f"Unsupported --classes file: {classes_path} "
            "(expected a .txt, .yaml/.yml or .json file: classes.txt, data.yaml or notes.json)"
        )

    if not classes:
        raise SystemExit(f"No class names found in {classes_path}")
    return classes


def normalize_source_classes(source_classes: dict[int, str]) -> dict[int, str]:
    """Normalise source names; stop if two of them collide after normalisation."""
    by_normalized: dict[str, list[tuple[int, str]]] = {}
    for class_id, raw_name in sorted(source_classes.items()):
        by_normalized.setdefault(normalize_name(raw_name), []).append((class_id, raw_name))

    collisions = {name: entries for name, entries in by_normalized.items() if len(entries) > 1}
    if collisions:
        lines = ["Source class names collide after normalisation:"]
        for name, entries in collisions.items():
            originals = ", ".join(f"{class_id}: '{raw_name}'" for class_id, raw_name in entries)
            lines.append(f"  '{name}' <- {originals}")
        lines.append("Fix the source class list (or merge those ids in the labels) and rerun.")
        raise SystemExit("\n".join(lines))

    return {class_id: normalize_name(raw_name) for class_id, raw_name in source_classes.items()}


def resolve_keep(keep_args: list[str], source_names: dict[int, str]) -> list[str]:
    """Normalise --keep, drop duplicates and names missing from the source."""
    available = set(source_names.values())
    keep: list[str] = []
    missing: list[str] = []
    for raw in keep_args:
        name = normalize_name(raw)
        if not name:
            continue
        if name in keep:
            print(f"WARNING: '{raw}' is listed more than once in --keep; using it once.")
            continue
        if name not in available:
            missing.append(raw)
            continue
        keep.append(name)

    available_text = ", ".join(source_names[class_id] for class_id in sorted(source_names))
    for raw in missing:
        print(f"WARNING: --keep name '{raw}' is not in the source classes; skipping it.")
    if missing:
        print(f"         Available source classes: {available_text}")
    if not keep:
        raise SystemExit(
            f"None of the --keep names exist in the source classes. Available: {available_text}"
        )
    return keep


def warn_if_order_differs(keep: list[str]) -> None:
    """Warn when --keep is in a different relative order than the training class list."""
    if not CANONICAL_CLASSES_PATH.exists():
        return
    canonical = [
        normalize_name(line)
        for line in CANONICAL_CLASSES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = [name for name in canonical if name in keep]
    actual = [name for name in keep if name in canonical]
    if actual != expected:
        print("WARNING: --keep order differs from yolonas-training/training-data/classes.txt.")
        print(f"         Typed order   : {actual}")
        print(f"         Training order: {expected}")
        print("         Class ids follow the typed order. build_training_data.py only combines")
        print("         datasets whose classes.txt are identical (same names, same order).")


def check_flat(folder: Path, flag: str) -> None:
    subfolders = sorted(p.name for p in folder.iterdir() if p.is_dir())
    if subfolders:
        raise SystemExit(
            f"{flag} must be a flat folder, but {folder} contains subfolders: "
            f"{', '.join(subfolders[:5])}. Point {flag} at the folder that holds the files."
        )


def prepare_dest(dest: Path, inputs: list[Path], overwrite: bool) -> None:
    """Create an empty --dest, refusing to clobber data unless --overwrite is given."""
    for src in inputs:
        if dest == src or dest in src.parents or src in dest.parents:
            raise SystemExit(f"--dest ({dest}) overlaps with an input path ({src}); choose another folder.")

    if dest.exists() and any(dest.iterdir()):
        if not overwrite:
            raise SystemExit(f"--dest already exists and is not empty: {dest}\nPass --overwrite to replace it.")
        shutil.rmtree(dest)

    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)


def filter_label_rows(
    label_path: Path,
    id_map: dict[int, int],
    source_names: dict[int, str],
    stats: Counter,
) -> list[tuple[int, str]]:
    """Return the kept rows of one label file as (new class id, rewritten line)."""
    kept: list[tuple[int, str]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        # Plain YOLO box rows only: class cx cy w h, with the box values within 0-1.
        try:
            old_id = int(parts[0])
            box_values = [float(value) for value in parts[1:]]
        except ValueError:
            stats["rows_malformed"] += 1
            continue
        if len(parts) != 5 or not all(0.0 <= value <= 1.0 for value in box_values):
            stats["rows_malformed"] += 1
            continue
        if old_id not in source_names:
            stats["rows_unknown_class_id"] += 1
            continue
        if old_id in id_map:
            new_id = id_map[old_id]
            kept.append((new_id, " ".join([str(new_id), *parts[1:]])))
    return kept


def paths_from_src(src: Path) -> tuple[Path, Path, Path | None]:
    """Work out --images, --labels and --classes from a dataset folder."""
    if not src.is_dir():
        raise SystemExit(f"--src folder not found: {src}")
    if not (src / "images").is_dir() or not (src / "labels").is_dir():
        projects = [p.name for p in sorted(src.iterdir()) if (p / "images").is_dir() and (p / "labels").is_dir()]
        if projects:
            raise SystemExit(
                f"--src {src} holds {len(projects)} project folders ({', '.join(projects[:3])}"
                f"{', ...' if len(projects) > 3 else ''}), not one dataset. Combine them first with "
                "combine_label_studio_projects.py, then pass the combined folder as --src."
            )
        raise SystemExit(f"--src {src} has no images/ and labels/ folders.")

    classes_txt, notes_json = src / "classes.txt", src / "notes.json"
    if classes_txt.is_file() and notes_json.is_file():
        from_txt = [n for _, n in sorted(load_source_classes(classes_txt).items())]
        from_json = [n for _, n in sorted(load_source_classes(notes_json).items())]
        if from_txt != from_json:
            raise SystemExit(
                f"classes.txt and notes.json in {src} disagree:\n"
                f"  classes.txt: {from_txt}\n  notes.json : {from_json}\n"
                "Choose the right one with --classes (the class ids in the labels follow only one of them)."
            )
        classes = classes_txt
    elif classes_txt.is_file():
        classes = classes_txt
    elif notes_json.is_file():
        classes = notes_json
    else:
        classes = next((p for p in (src / "data.yaml", src.parent / "data.yaml") if p.is_file()), None)
    return src / "images", src / "labels", classes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean a YOLO dataset down to a chosen set of classes and drop images left without annotations."
    )
    parser.add_argument(
        "--src",
        type=Path,
        help="Dataset folder: --images, --labels and --classes are worked out from it "
        "(images/, labels/, classes.txt and/or notes.json). Explicit flags override it.",
    )
    parser.add_argument("--images", type=Path, help="Source images folder (flat). Required unless --src is given.")
    parser.add_argument("--labels", type=Path, help="Source YOLO labels folder (flat). Required unless --src is given.")
    parser.add_argument(
        "--classes",
        type=Path,
        help="File listing the source dataset's classes. Only 3 formats are accepted: "
        "classes.txt (.txt), data.yaml (.yaml/.yml) or notes.json (.json). Required unless --src is given.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Output dataset root; images/, labels/ and classes.txt are written here.",
    )
    parser.add_argument(
        "--keep",
        nargs="+",
        required=True,
        metavar="CLASS",
        help="Classes to keep; the order sets the output class ids. "
        "Names with spaces can be typed with underscores (plant_growth) or quoted ('plant growth').",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --dest if it already exists and is not empty.",
    )
    args = parser.parse_args()

    if args.src is not None:
        src_images, src_labels, src_classes = paths_from_src(args.src.resolve())
        args.images = args.images or src_images
        args.labels = args.labels or src_labels
        args.classes = args.classes or src_classes
    missing = [flag for flag, value in (("--images", args.images), ("--labels", args.labels), ("--classes", args.classes)) if value is None]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}: give them, or give --src with the dataset folder.")

    images_dir = args.images.resolve()
    labels_dir = args.labels.resolve()
    classes_path = args.classes.resolve()
    dest = args.dest.resolve()

    if not images_dir.is_dir():
        raise SystemExit(f"--images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"--labels folder not found: {labels_dir}")
    if not classes_path.is_file():
        raise SystemExit(f"--classes file not found: {classes_path}")
    check_flat(images_dir, "--images")
    check_flat(labels_dir, "--labels")

    source_names = normalize_source_classes(load_source_classes(classes_path))
    keep = resolve_keep(args.keep, source_names)
    warn_if_order_differs(keep)

    new_id_by_name = {name: new_id for new_id, name in enumerate(keep)}
    id_map = {
        old_id: new_id_by_name[name] for old_id, name in source_names.items() if name in new_id_by_name
    }

    print(f"Images  : {images_dir}")
    print(f"Labels  : {labels_dir}")
    print(f"Classes : {classes_path}")
    print(f"Dest    : {dest}")
    print(f"Keeping : {keep}")
    print()

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    label_by_stem = {p.stem: p for p in labels_dir.glob("*.txt")}
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    prepare_dest(dest, [images_dir, labels_dir, classes_path], args.overwrite)
    out_images_dir = dest / "images"
    out_labels_dir = dest / "labels"

    stats: Counter = Counter()
    boxes_per_class: Counter = Counter()
    stem_counts = Counter(p.stem for p in image_paths)
    stats["image_stems_duplicated"] = sum(1 for count in stem_counts.values() if count > 1)
    stats["labels_without_image"] = len(set(label_by_stem) - set(stem_counts))

    iterable = image_paths
    if tqdm is not None:
        iterable = tqdm(image_paths, desc="Cleaning", unit="image")

    for img_path in iterable:
        label_path = label_by_stem.get(img_path.stem)
        if label_path is None:
            stats["images_without_label"] += 1
            stats["images_dropped"] += 1
            continue

        kept_rows = filter_label_rows(label_path, id_map, source_names, stats)
        if not kept_rows:
            stats["images_dropped"] += 1
            continue

        shutil.copy2(img_path, out_images_dir / img_path.name)
        (out_labels_dir / f"{img_path.stem}.txt").write_text(
            "\n".join(line for _, line in kept_rows) + "\n", encoding="utf-8"
        )
        stats["images_kept"] += 1
        for new_id, _ in kept_rows:
            boxes_per_class[new_id] += 1

    (dest / "classes.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")

    print()
    print(f"Images found            : {len(image_paths)}")
    print(f"Images kept             : {stats['images_kept']}")
    print(f"Images dropped (no kept annotations): {stats['images_dropped']}")
    print("Boxes per class:")
    for new_id, name in enumerate(keep):
        print(f"  {new_id}: {name:<20} {boxes_per_class[new_id]}")
    print(f"  total{'':<19} {sum(boxes_per_class.values())}")

    problems = [
        ("Images with no label file", stats["images_without_label"]),
        ("Label files with no image", stats["labels_without_image"]),
        ("Label rows with a class id not in the class list", stats["rows_unknown_class_id"]),
        ("Label rows that are not valid YOLO box rows (5 fields, values 0-1; skipped)", stats["rows_malformed"]),
        ("Image names shared by more than one file extension", stats["image_stems_duplicated"]),
    ]
    found = [(text, count) for text, count in problems if count]
    if found:
        print("Label problems found in the source:")
        for text, count in found:
            print(f"  {text}: {count}")
    else:
        print("Label problems found in the source: none")
    print(f"Cleaned dataset written to: {dest}")


if __name__ == "__main__":
    main()
