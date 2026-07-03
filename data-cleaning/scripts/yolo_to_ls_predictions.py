#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Purpose:
#   Convert YOLO-format label files (class_id xc yc w h, normalized 0-1) into
#   Label Studio import JSON with pre-annotated rectangle predictions.
#
# Usage:
#   python scripts/yolo_to_ls_predictions.py --base-dir ../data/data-cleaned/my-dataset \
#     --output yolo_predictions.json
#
# Note: --base-dir is resolved relative to the current working directory (not
# this script's location) — always pass it explicitly rather than relying on
# the default, e.g. ../data/data-cleaned/<dataset-name> in the consolidated repo.
#
#   Use either --image-url-base (for hosted images) or --local-files-root (for
#   Label Studio local files). If neither is set, --local-files-root defaults to
#   the dataset's images/ folder under --base-dir.
# -----------------------------------------------------------------------------
import argparse
import glob
import json
import os


def load_classes(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def yolo_box_to_ls_percent(xc, yc, w, h):
    x = (xc - w / 2.0) * 100.0
    y = (yc - h / 2.0) * 100.0
    return x, y, w * 100.0, h * 100.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert YOLO labels into Label Studio import JSON with predictions."
    )
    parser.add_argument(
        "--base-dir",
        default="../data/data-cleaned/building-defect-detection",
        help="Directory with images/, labels/, classes.txt. Resolved relative to CWD, not this script's location.",
    )
    parser.add_argument("--from-name", default="label")
    parser.add_argument("--to-name", default="image")
    parser.add_argument("--model-version", default="yolo-import")
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument(
        "--local-files-root",
        default=None,
        help="Absolute images directory for Label Studio local files, "
        "e.g. /home/user/images",
    )
    parser.add_argument(
        "--image-url-base",
        default=None,
        help="Base URL for images, e.g. http://localhost:8000. "
        "Overrides --local-files-root.",
    )
    parser.add_argument(
        "--output",
        default="yolo_predictions.json",
        help="Output JSON file to import into Label Studio.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    classes_path = os.path.join(args.base_dir, "classes.txt")
    labels_dir = os.path.join(args.base_dir, "labels")
    images_dir = os.path.join(args.base_dir, "images")

    classes = load_classes(classes_path)
    class_map = {i: name for i, name in enumerate(classes)}

    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if not label_files:
        raise SystemExit(f"No label files found in {labels_dir}")

    tasks = []
    for label_file in label_files:
        base = os.path.splitext(os.path.basename(label_file))[0]
        img_matches = glob.glob(os.path.join(images_dir, base + ".*"))
        if not img_matches:
            continue
        image_name = os.path.basename(img_matches[0])
        if args.image_url_base:
            image_path = args.image_url_base.rstrip("/") + "/" + image_name
        else:
            local_root = args.local_files_root or images_dir
            local_root = os.path.abspath(local_root)
            image_path = f"/data/local-files/?d={local_root}/{image_name}"

        results = []
        with open(label_file, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                x, y, w_pct, h_pct = yolo_box_to_ls_percent(xc, yc, w, h)
                label = class_map.get(cls_id, f"class_{cls_id}")
                results.append(
                    {
                        "from_name": args.from_name,
                        "to_name": args.to_name,
                        "type": "rectanglelabels",
                        "value": {
                            "x": x,
                            "y": y,
                            "width": w_pct,
                            "height": h_pct,
                            "rectanglelabels": [label],
                        },
                    }
                )

        if not results:
            continue

        task = {
            "data": {args.to_name: image_path},
            "predictions": [
                {
                    "result": results,
                    "model_version": args.model_version,
                    "score": args.score,
                }
            ],
        }
        tasks.append(task)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(tasks, handle, indent=2)
    print(f"Wrote {len(tasks)} tasks to {args.output}")


if __name__ == "__main__":
    main()
