#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:
    raise SystemExit(
        "Pillow is required. Install with: pip install pillow"
    ) from exc

try:
    from super_gradients.training import models
except Exception as exc:
    raise SystemExit(
        "super-gradients is required. Install with: pip install super-gradients"
    ) from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_model_name(name: str) -> str:
    return name.replace("-", "_").lower()


def list_images(input_dir: Path) -> List[Path]:
    images: List[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return images


def clear_output_dirs(output_dir: Path) -> Dict[str, Path]:
    subdirs = {
        "images": output_dir / "images",
        "labels": output_dir / "labels",
        "images_overlaid": output_dir / "images-overlaid",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in subdirs.values():
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def _to_numpy(x):
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore

        if isinstance(x, np.ndarray):
            return x
    except Exception:
        pass
    return x


def extract_predictions(preds) -> Tuple[List[List[float]], List[float], List[int]]:
    # Handle list-like prediction results
    if isinstance(preds, (list, tuple)):
        preds = preds[0] if preds else None
    if preds is None:
        return [], [], []

    pred_obj = None
    for attr in ("prediction", "predictions"):
        if hasattr(preds, attr):
            pred_obj = getattr(preds, attr)
            break
    if pred_obj is None:
        pred_obj = preds

    boxes = None
    for attr in ("bboxes_xyxy", "bboxes", "boxes", "xyxy"):
        if hasattr(pred_obj, attr):
            boxes = getattr(pred_obj, attr)
            break
    if boxes is None:
        return [], [], []

    confs = None
    for attr in ("confidence", "confidences", "scores", "confidence_scores"):
        if hasattr(pred_obj, attr):
            confs = getattr(pred_obj, attr)
            break

    labels = None
    for attr in ("labels", "classes", "class_ids"):
        if hasattr(pred_obj, attr):
            labels = getattr(pred_obj, attr)
            break

    boxes = _to_numpy(boxes)
    confs = _to_numpy(confs) if confs is not None else None
    labels = _to_numpy(labels) if labels is not None else None

    boxes_list = boxes.tolist() if hasattr(boxes, "tolist") else list(boxes)
    confs_list = (
        confs.tolist() if hasattr(confs, "tolist") else list(confs)
    ) if confs is not None else []
    labels_list = (
        labels.tolist() if hasattr(labels, "tolist") else list(labels)
    ) if labels is not None else []

    return boxes_list, confs_list, labels_list


def xyxy_to_yolo(
    xyxy: Iterable[float], image_w: int, image_h: int
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(x1), image_w))
    x2 = max(0.0, min(float(x2), image_w))
    y1 = max(0.0, min(float(y1), image_h))
    y2 = max(0.0, min(float(y2), image_h))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / image_w, cy / image_h, w / image_w, h / image_h


def draw_overlays(
    image_path: Path,
    output_path: Path,
    boxes_xyxy: List[List[float]],
    labels: List[int],
    confs: List[float],
    class_names: List[str],
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    font = None
    base_font_size = 12
    font_size = base_font_size * 5
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    palette = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
    ]
    box_width = 6

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [float(v) for v in box]
        label = labels[i] if i < len(labels) else -1
        conf = confs[i] if i < len(confs) else None
        class_name = (
            class_names[label] if 0 <= label < len(class_names) else str(label)
        )
        color = palette[label % len(palette)] if label >= 0 else "#ff0000"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
        text = f"{class_name}"
        if conf is not None:
            text += f" {conf:.2f}"
        if font:
            draw.text((x1 + 4, y1 + 4), text, fill=color, font=font)
        else:
            draw.text((x1 + 4, y1 + 4), text, fill=color)

    image.save(output_path)


def save_yolo_labels(
    output_path: Path,
    boxes_xyxy: List[List[float]],
    labels: List[int],
    confs: List[float],
    image_w: int,
    image_h: int,
) -> None:
    lines: List[str] = []
    for i, box in enumerate(boxes_xyxy):
        label = labels[i] if i < len(labels) else 0
        conf = confs[i] if i < len(confs) else None
        cx, cy, w, h = xyxy_to_yolo(box, image_w, image_h)
        if conf is None:
            lines.append(f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        else:
            lines.append(
                f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def merge_overlapping_annotations(
    boxes: List[List[float]],
    confs: List[float],
    labels: List[int],
    iou_threshold: float = 0.1,
) -> Tuple[List[List[float]], List[float], List[int]]:
    if not boxes:
        return boxes, confs, labels

    def rects_touch_or_overlap(b1, b2) -> bool:
        return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])

    annots = list(zip(boxes, confs, labels))
    while True:
        merged = []
        used = [False] * len(annots)
        for i, (box_i, conf_i, lbl_i) in enumerate(annots):
            if used[i]:
                continue
            group_boxes = [box_i]
            group_confs = [conf_i]
            used[i] = True
            for j in range(i + 1, len(annots)):
                if used[j]:
                    continue
                box_j, conf_j, lbl_j = annots[j]
                if lbl_i == lbl_j and (
                    rects_touch_or_overlap(box_i, box_j)
                    or calculate_iou(box_i, box_j) > iou_threshold
                ):
                    group_boxes.append(box_j)
                    group_confs.append(conf_j)
                    used[j] = True
            x1 = min(b[0] for b in group_boxes)
            y1 = min(b[1] for b in group_boxes)
            x2 = max(b[2] for b in group_boxes)
            y2 = max(b[3] for b in group_boxes)
            merged.append(([x1, y1, x2, y2], max(group_confs), lbl_i))
        if len(merged) == len(annots):
            break
        annots = merged

    out_boxes = [a[0] for a in annots]
    out_confs = [a[1] for a in annots]
    out_labels = [a[2] for a in annots]
    return out_boxes, out_confs, out_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference for YOLO-NAS")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    logging.getLogger("super_gradients").setLevel(logging.WARNING)

    input_dir = Path(cfg.get("input_data", "input-data"))
    output_dir = Path(cfg.get("output_dir", "output-preds"))
    model_type = cfg.get("model_type", "yolo-nas-m")
    checkpoint_path = Path(cfg.get("checkpoint_path", ""))
    num_classes = cfg.get("num_classes", None)
    class_names = cfg.get("class_names", [])
    conf_threshold = float(cfg.get("conf_threshold", 0.25))
    nms_iou_threshold = float(cfg.get("nms_iou_threshold", 0.7))
    image_size = cfg.get("image_size", 640)
    use_tiled_inference = bool(cfg.get("use_tiled_inference", False))
    tiled_cfg = cfg.get("tiled_inference", {}) or {}
    slice_height = int(tiled_cfg.get("slice_height", image_size))
    slice_width = int(tiled_cfg.get("slice_width", image_size))
    overlap_height_ratio = float(tiled_cfg.get("overlap_height_ratio", 0.2))
    overlap_width_ratio = float(tiled_cfg.get("overlap_width_ratio", 0.2))
    tiled_device = tiled_cfg.get("device", None)
    postprocess_type = tiled_cfg.get("postprocess_type", "GREEDYNMM")
    postprocess_match_metric = tiled_cfg.get("postprocess_match_metric", "IOU")
    postprocess_match_threshold = float(
        tiled_cfg.get("postprocess_match_threshold", 0.25)
    )
    merge_cfg = cfg.get("merge_annotations", {}) or {}
    do_merge = bool(merge_cfg.get("enabled", False))
    merge_iou_threshold = float(merge_cfg.get("iou_threshold", 0.1))

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    run_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_paths = {
        "images": run_dir / "images",
        "labels": run_dir / "labels",
        "images_overlaid": run_dir / "images-overlaid",
    }
    for path in output_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    run_config_path = run_dir / "config.yaml"
    run_config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    model_name = normalize_model_name(model_type)

    model_kwargs = {
        "checkpoint_path": str(checkpoint_path),
    }
    if num_classes is not None:
        model_kwargs["num_classes"] = int(num_classes)

    model = models.get(model_name, **model_kwargs)
    model.eval()

    images = list_images(input_dir)
    if not images:
        raise SystemExit(f"No images found in: {input_dir}")

    tiled_predictor = None
    if use_tiled_inference:
        from sahi_inference import SahiTiledPredictor

        tiled_predictor = SahiTiledPredictor(
            model=model,
            class_names=class_names,
            conf_threshold=conf_threshold,
            iou_threshold=nms_iou_threshold,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_type=postprocess_type,
            postprocess_match_metric=postprocess_match_metric,
            postprocess_match_threshold=postprocess_match_threshold,
            device=tiled_device,
        )

    for image_path in tqdm(
        images,
        desc="Batch inference",
        unit="image",
        dynamic_ncols=True,
        smoothing=0.1,
        mininterval=0.1,
    ):
        if tiled_predictor is not None:
            boxes, confs, labels = tiled_predictor.predict(image_path)
        else:
            try:
                preds = model.predict(
                    str(image_path),
                    conf=conf_threshold,
                    iou=nms_iou_threshold,
                    image_size=image_size,
                )
            except TypeError:
                preds = model.predict(
                    str(image_path),
                    conf=conf_threshold,
                    iou=nms_iou_threshold,
                )

            boxes, confs, labels = extract_predictions(preds)

        if do_merge:
            boxes, confs, labels = merge_overlapping_annotations(
                boxes, confs, labels, iou_threshold=merge_iou_threshold
            )

        # Save raw image
        shutil.copy2(image_path, output_paths["images"] / image_path.name)

        # Save labels
        with Image.open(image_path) as img:
            w, h = img.size
        label_path = output_paths["labels"] / f"{image_path.stem}.txt"
        save_yolo_labels(label_path, boxes, labels, confs, w, h)

        # Save overlay
        overlay_path = output_paths["images_overlaid"] / image_path.name
        draw_overlays(image_path, overlay_path, boxes, labels, confs, class_names)


if __name__ == "__main__":
    main()
