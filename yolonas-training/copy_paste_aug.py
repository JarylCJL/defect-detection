"""
Class-aware copy-paste augmentation for SuperGradients / YOLO-NAS.

Builds a crop pool from the training labels at init time, then during training
pastes instances of rare classes onto each image. This is different from weighted
sampling: rather than repeating whole images (which drags along majority-class
instances), it injects rare-class instances into diverse contexts.

Usage in train_transforms (insert before DetectionMosaic):
    from copy_paste_aug import ClassAwareCopyPaste
    copy_paste = ClassAwareCopyPaste(
        images_dir="data/train/images",
        labels_dir="data/train/labels",
        rare_classes=[0, 2, 3],   # corrosion, spall, plant_growth
        paste_prob=0.5,
        max_paste=3,
    )
    train_transforms = [{"ClassAwareCopyPaste": copy_paste}, ...]

Or use it directly as a transform object in the list (SG also accepts instances).
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from super_gradients.training.samples import DetectionSample
from super_gradients.training.transforms.detection.abstract_detection_transform import AbstractDetectionTransform


@dataclass
class _InstanceEntry:
    image_path: str
    # Normalized CXCYWH — converted to absolute XYXY on load
    cx: float
    cy: float
    w: float
    h: float
    label: int


class ClassAwareCopyPaste(AbstractDetectionTransform):
    """
    Pastes bbox-level crops of rare classes onto training images.

    :param images_dir:   Path to training images directory.
    :param labels_dir:   Path to training labels directory (YOLO txt format).
    :param rare_classes: List of class IDs to source crops from.
    :param paste_prob:   Probability of applying this transform to any given image.
    :param max_paste:    Maximum number of instances to paste per image.
    :param scale_range:  (min, max) scale of the pasted crop relative to the
                         target image's shorter side. E.g. (0.05, 0.25) keeps
                         pasted objects between 5–25% of the image.
    :param min_px:       Minimum width AND height of a source crop in pixels.
                         Crops smaller than this are skipped.
    :param max_overlap:  Maximum IoU between a candidate paste region and any
                         existing box before the placement is rejected.
    :param max_attempts: How many random placements to try before giving up on
                         a crop.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        rare_classes: List[int],
        paste_prob: float = 0.5,
        max_paste: int = 3,
        scale_range: Tuple[float, float] = (0.05, 0.25),
        min_px: int = 16,
        max_overlap: float = 0.1,
        max_attempts: int = 10,
    ):
        super().__init__()
        self.rare_classes = set(rare_classes)
        self.paste_prob = paste_prob
        self.max_paste = max_paste
        self.scale_range = scale_range
        self.min_px = min_px
        self.max_overlap = max_overlap
        self.max_attempts = max_attempts

        self._pool: Dict[int, List[_InstanceEntry]] = {c: [] for c in rare_classes}
        self._build_pool(images_dir, labels_dir)

    # ------------------------------------------------------------------
    # Pool construction (runs once at init)
    # ------------------------------------------------------------------

    def _build_pool(self, images_dir: str, labels_dir: str) -> None:
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        img_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

        for label_file in labels_dir.glob("*.txt"):
            lines = label_file.read_text().strip().splitlines()
            rare_lines = [
                ln for ln in lines
                if ln.strip() and int(ln.split()[0]) in self.rare_classes
            ]
            if not rare_lines:
                continue

            img_path = next(
                (images_dir / (label_file.stem + ext)
                 for ext in img_exts
                 if (images_dir / (label_file.stem + ext)).exists()),
                None,
            )
            if img_path is None:
                continue

            for ln in rare_lines:
                parts = ln.split()
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                self._pool[cls_id].append(
                    _InstanceEntry(str(img_path), cx, cy, w, h, cls_id)
                )

        for cls_id, entries in self._pool.items():
            print(f"[CopyPaste] class {cls_id}: {len(entries)} crops in pool")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_crop(self, entry: _InstanceEntry) -> Optional[np.ndarray]:
        """Return the cropped BGR image, or None if unusable."""
        img = cv2.imread(entry.image_path)
        if img is None:
            return None
        H, W = img.shape[:2]
        x1 = int((entry.cx - entry.w / 2) * W)
        y1 = int((entry.cy - entry.h / 2) * H)
        x2 = int((entry.cx + entry.w / 2) * W)
        y2 = int((entry.cy + entry.h / 2) * H)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < self.min_px or y2 - y1 < self.min_px:
            return None
        return img[y1:y2, x1:x2]

    @staticmethod
    def _max_iou(box: np.ndarray, boxes: np.ndarray) -> float:
        """Max IoU of box [4] against each row in boxes [N,4]. Returns 0 if empty."""
        if len(boxes) == 0:
            return 0.0
        ix1 = np.maximum(box[0], boxes[:, 0])
        iy1 = np.maximum(box[1], boxes[:, 1])
        ix2 = np.minimum(box[2], boxes[:, 2])
        iy2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        a_box = (box[2] - box[0]) * (box[3] - box[1])
        a_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = a_box + a_boxes - inter
        return float(np.max(inter / np.maximum(union, 1e-6)))

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        if random.random() > self.paste_prob:
            return sample

        all_entries = [e for entries in self._pool.values() for e in entries]
        if not all_entries:
            return sample

        img = sample.image.copy()
        H, W = img.shape[:2]
        short_side = min(H, W)

        new_bboxes = list(sample.bboxes_xyxy)
        new_labels = list(sample.labels)
        new_is_crowd = list(sample.is_crowd)

        random.shuffle(all_entries)
        pasted = 0

        for entry in all_entries:
            if pasted >= self.max_paste:
                break

            crop = self._load_crop(entry)
            if crop is None:
                continue

            ch, cw = crop.shape[:2]
            # Scale so the pasted object occupies scale_range fraction of the
            # image's shorter side (preserves aspect ratio)
            target_size = int(short_side * random.uniform(*self.scale_range))
            target_size = max(self.min_px, target_size)
            scale = target_size / max(cw, ch)
            new_cw = max(self.min_px, int(cw * scale))
            new_ch = max(self.min_px, int(ch * scale))
            if new_cw > W or new_ch > H:
                continue

            crop_resized = cv2.resize(crop, (new_cw, new_ch))

            placed = False
            for _ in range(self.max_attempts):
                x1 = random.randint(0, W - new_cw)
                y1 = random.randint(0, H - new_ch)
                x2, y2 = x1 + new_cw, y1 + new_ch
                candidate = np.array([x1, y1, x2, y2], dtype=np.float32)
                if new_bboxes and self._max_iou(candidate, np.array(new_bboxes)) > self.max_overlap:
                    continue
                img[y1:y2, x1:x2] = crop_resized
                new_bboxes.append(candidate)
                new_labels.append(entry.label)
                new_is_crowd.append(False)
                placed = True
                break

            if placed:
                pasted += 1

        if not new_bboxes:
            return sample

        return DetectionSample(
            image=img,
            bboxes_xyxy=np.array(new_bboxes, dtype=np.float32).reshape(-1, 4),
            labels=np.array(new_labels, dtype=sample.labels.dtype),
            is_crowd=np.array(new_is_crowd, dtype=bool),
        )

    def get_equivalent_preprocessing(self) -> List:
        return []
