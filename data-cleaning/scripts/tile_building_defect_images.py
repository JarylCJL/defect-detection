#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Script: Tile Building Defect Images + Labels
#
# What it does:
# - Reads images from <src>/images and YOLO-format labels from <src>/labels.
# - Generates square tiles at one or more sizes with configurable overlap.
# - Clips bounding boxes to each tile and writes new YOLO labels.
# - Writes tiled images/labels to <dest>/images and <dest>/labels.
# - If <src>/classes.txt exists, copies it to <dest>/classes.txt and writes
#   <dest>/data.yaml (names/nc/train: images) so merge_datasets.py can
#   consume the output directly.
#
# How to use:
# - Default run (sizes=2048; overlap=0.5):
#   python3 scripts/tile_building_defect_images.py
# - Customize tile sizes/overlap:
#   python3 scripts/tile_building_defect_images.py --sizes 512,1024 --overlap 0.5
# - Keep empty tiles:
#   python3 scripts/tile_building_defect_images.py --keep-empty
# - Change input/output roots:
#   python3 scripts/tile_building_defect_images.py --src data-cleaned/building-defect-cleaned --dest data-cleaned/defect-data-tiled
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # Fallback if tqdm isn't installed.
    tqdm = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_sizes(value: str) -> list[int]:
    sizes = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append(int(part))
    if not sizes:
        raise ValueError("No tile sizes provided.")
    return sizes


def tile_positions(length: int, tile: int, stride: int) -> list[int]:
    if tile > length:
        return []
    positions = list(range(0, length - tile + 1, stride))
    last = length - tile
    if positions and positions[-1] != last:
        positions.append(last)
    elif not positions:
        positions = [0]
    return positions


def load_labels(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    labels = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            labels.append((cls, x * img_w, y * img_h, w * img_w, h * img_h))
    return labels


def clip_box_to_tile(
    box: tuple[int, float, float, float, float],
    x0: int,
    y0: int,
    size: int,
) -> tuple[int, float, float, float, float] | None:
    cls, cx, cy, bw, bh = box
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    tx1, ty1 = x0, y0
    tx2, ty2 = x0 + size, y0 + size

    ix1 = max(x1, tx1)
    iy1 = max(y1, ty1)
    ix2 = min(x2, tx2)
    iy2 = min(y2, ty2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    new_cx = (ix1 + ix2) / 2 - x0
    new_cy = (iy1 + iy2) / 2 - y0
    new_w = ix2 - ix1
    new_h = iy2 - iy1

    return cls, new_cx / size, new_cy / size, new_w / size, new_h / size


def write_labels(out_path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for cls, x, y, w, h in labels:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def write_dataset_metadata(src_root: Path, out_root: Path) -> None:
    """Propagate classes.txt and write a data.yaml so downstream tools
    (e.g. merge_datasets.py, which hard-requires a data.yaml with a
    `train:` key) can consume the tiled output directly.
    """
    src_classes = src_root / "classes.txt"
    if not src_classes.exists():
        return

    names = [line.strip() for line in src_classes.read_text().splitlines() if line.strip()]
    if not names:
        return

    (out_root / "classes.txt").write_text("\n".join(names) + "\n")

    data_yaml_lines = [
        "names:",
        *[f"- {name}" for name in names],
        f"nc: {len(names)}",
        "train: images",
    ]
    (out_root / "data.yaml").write_text("\n".join(data_yaml_lines) + "\n")


def tile_image(
    img_path: Path,
    label_path: Path,
    out_img_dir: Path,
    out_label_dir: Path,
    sizes: list[int],
    overlap: float,
    skip_empty: bool,
) -> None:
    with Image.open(img_path) as img:
        img_w, img_h = img.size
        labels = load_labels(label_path, img_w, img_h)

        for size in sizes:
            stride = max(1, int(math.floor(size * (1.0 - overlap))))
            xs = tile_positions(img_w, size, stride)
            ys = tile_positions(img_h, size, stride)
            if not xs or not ys:
                continue

            for y0 in ys:
                for x0 in xs:
                    tile = img.crop((x0, y0, x0 + size, y0 + size))
                    tile_name = f"{img_path.stem}_s{size}_x{x0}_y{y0}{img_path.suffix}"
                    out_img_path = out_img_dir / tile_name

                    tile_labels = []
                    for box in labels:
                        clipped = clip_box_to_tile(box, x0, y0, size)
                        if clipped is not None:
                            tile_labels.append(clipped)

                    if skip_empty and not tile_labels:
                        continue

                    tile.save(out_img_path)
                    out_label_path = out_label_dir / f"{img_path.stem}_s{size}_x{x0}_y{y0}.txt"
                    write_labels(out_label_path, tile_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile images and labels with overlap.")
    parser.add_argument("--sizes", default="2048", help="Comma-separated tile sizes.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio, e.g. 0.5.")
    parser.add_argument(
        "--src",
        default="data-cleaned/building-defect-detection",
        help="Source dataset root containing images/ and labels/.",
    )
    parser.add_argument(
        "--dest",
        default="data-cleaned/defect-data-tiled",
        help="Destination dataset root for tiled images/ and labels/.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep tiles that contain no annotations.",
    )
    args = parser.parse_args()

    # <repo>/data-cleaning/scripts/.. -> data-cleaning/.. -> <repo>, then /data
    # (relative --src/--dest values are resolved against the shared top-level data/ dir)
    repo_root = Path(__file__).resolve().parents[2] / "data"
    src_root = Path(args.src)
    if not src_root.is_absolute():
        src_root = repo_root / src_root
    src_img_dir = src_root / "images"
    src_label_dir = src_root / "labels"
    out_root = Path(args.dest)
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_img_dir = out_root / "images"
    out_label_dir = out_root / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    sizes = parse_sizes(args.sizes)

    items: list[tuple[Path, Path, Path, Path]] = []
    for root, _, files in os.walk(src_img_dir):
        rel_root = Path(root).relative_to(src_img_dir)
        out_img_sub = out_img_dir / rel_root
        out_label_sub = out_label_dir / rel_root
        out_img_sub.mkdir(parents=True, exist_ok=True)
        out_label_sub.mkdir(parents=True, exist_ok=True)

        for name in files:
            if Path(name).suffix.lower() not in IMG_EXTS:
                continue
            img_path = Path(root) / name
            label_path = src_label_dir / rel_root / f"{img_path.stem}.txt"
            items.append((img_path, label_path, out_img_sub, out_label_sub))

    iterable = items
    if tqdm is not None:
        iterable = tqdm(items, desc="Tiling images", unit="image")

    for img_path, label_path, out_img_sub, out_label_sub in iterable:
        tile_image(
            img_path,
            label_path,
            out_img_sub,
            out_label_sub,
            sizes,
            args.overlap,
            not args.keep_empty,
        )

    write_dataset_metadata(src_root, out_root)


if __name__ == "__main__":
    main()
