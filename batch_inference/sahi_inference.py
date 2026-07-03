#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    from sahi.models.base import DetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.prediction import ObjectPrediction
    from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
except Exception as exc:
    raise SystemExit(
        "SAHI is required for tiled inference. Install with: pip install sahi"
    ) from exc


class YoloNASDetectionModel(DetectionModel):
    def __init__(
        self,
        model,
        class_names: List[str],
        confidence_threshold: float,
        iou_threshold: float,
        device: str,
        category_mapping: Optional[Dict[str, str]] = None,
        load_at_init: bool = True,
    ) -> None:
        self._provided_model = model
        self._class_names = class_names
        self.iou_threshold = iou_threshold

        if not category_mapping:
            category_mapping = {
                str(i): name for i, name in enumerate(self._class_names)
            }

        super().__init__(
            model_path=None,
            confidence_threshold=confidence_threshold,
            device=device,
            category_mapping=category_mapping,
            load_at_init=False,
        )

        if load_at_init:
            self.load_model()

    def load_model(self):
        if self._provided_model is None:
            raise ValueError("model must be provided for YoloNASDetectionModel")
        self.set_model(self._provided_model)

    def set_model(self, model):
        self.model = model.to(self.device)
        self.model.eval()

        if not self.category_mapping:
            self.category_mapping = {
                str(i): name for i, name in enumerate(self._class_names)
            }

    def perform_inference(self, image: np.ndarray):
        if self.model is None:
            self.load_model()

        # SAHI provides RGB images; SuperGradients expects BGR for cv2 arrays
        image_bgr = image[:, :, ::-1].copy()
        prediction = self.model.predict(
            image_bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
        )
        det = prediction.prediction
        self._original_predictions = {
            "bboxes": det.bboxes_xyxy,
            "scores": det.confidence,
            "labels": det.labels,
        }

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list=[[0, 0]],
        full_shape_list=None,
    ):
        original = self._original_predictions
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        object_prediction_list_per_image = []
        for image_ind, shift_amount in enumerate(shift_amount_list):
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []
            if original is not None and original["bboxes"].size > 0:
                for bbox, score, label in zip(
                    original["bboxes"], original["scores"], original["labels"]
                ):
                    category_id = int(label)
                    category_name = self.category_mapping.get(
                        str(category_id), str(category_id)
                    )
                    object_prediction_list.append(
                        ObjectPrediction(
                            bbox=[float(v) for v in bbox],
                            category_id=category_id,
                            category_name=category_name,
                            score=float(score),
                            shift_amount=shift_amount,
                            full_shape=full_shape,
                        )
                    )
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image


class SahiTiledPredictor:
    def __init__(
        self,
        model,
        class_names: List[str],
        conf_threshold: float,
        iou_threshold: float,
        slice_height: int,
        slice_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
        postprocess_type: str,
        postprocess_match_metric: str,
        postprocess_match_threshold: float,
        device: Optional[str] = None,
    ) -> None:
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.postprocess_type = postprocess_type
        self.postprocess_match_metric = postprocess_match_metric
        self.postprocess_match_threshold = postprocess_match_threshold

        if device is None:
            try:
                import torch  # type: ignore

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        self.detection_model = YoloNASDetectionModel(
            model=model,
            class_names=class_names,
            confidence_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            load_at_init=True,
        )

    def predict(self, image_path: Path) -> Tuple[List[List[float]], List[float], List[int]]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        result = get_sliced_prediction(
            image,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            postprocess_type=self.postprocess_type,
            postprocess_match_metric=self.postprocess_match_metric,
            postprocess_match_threshold=self.postprocess_match_threshold,
        )

        boxes: List[List[float]] = []
        confs: List[float] = []
        labels: List[int] = []

        for obj in result.object_prediction_list:
            bbox = obj.bbox
            if hasattr(bbox, "to_xyxy"):
                x1, y1, x2, y2 = bbox.to_xyxy()
            else:
                x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            score = obj.score.value if obj.score is not None else 0.0
            confs.append(float(score))
            category_id = obj.category.id if obj.category is not None else 0
            labels.append(int(category_id))

        return boxes, confs, labels
