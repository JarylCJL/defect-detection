# Building Defect Detection — Training Documentation

What this document covers: how the training data is prepared, how the model is trained and evaluated, and **why** each of those choices was made. It is written to be read on its own as a report, and alongside the repository by whoever takes the project over.

The step-by-step commands are in the repository [readme](../readme.md) and are not repeated here. Production inference is covered in the readme's Stage 3 and in [`batch_inference/README.md`](../batch_inference/README.md).

State described: the repository as of October 2026.

---

## Summary

- The model finds and draws boxes around five kinds of defect in drone photos of building façades: corrosion, crack, spall, plant growth and cracked glass.
- Photos are cut into overlapping 2048 px tiles for both training and prediction, so small defects such as hairline cracks are not lost when an image is shrunk to the model's 640 px input. It gave the largest improvement in accuracy of any change, and most of the pipeline follows from it (§2).
- The model is YOLO-NAS-M, starting from weights pre-trained on the general-purpose COCO dataset, trained for 100 epochs on about 17,400 tiles.
- Training and validation are split by source photo, so no photo contributes tiles to both (§3.4). Earlier runs split by tile, which let near-identical tiles appear on both sides and made their validation scores too optimistic.

---

## 1. Objective and taxonomy

The system detects and localises defects on building façades in drone (UAV) imagery. It is an **object detection** task, not classification: the operational requirement is to know *where* on the façade a defect is, not merely that one exists somewhere in the frame.

The model is trained on five classes. The order matters: a class's line number in `classes.txt` is its id, and every dataset, the training split and `batch_inference/config.yaml` must use the same order.

| ID | Class | Notes |
|---|---|---|
| 0 | `corrosion` | Rust on metal fixtures, brackets, air-conditioning units |
| 1 | `crack` | Concrete and render cracking |
| 2 | `spall` | Concrete spalling / delamination |
| 3 | `plant_growth` | Vegetation ingress |
| 4 | `crack_glass` | Cracked or shattered glazing |

**History.** Earlier runs used three classes (`crack`, `corrosion`, `spall`). `plant_growth` and `crack_glass` were added later. Checkpoints from before and after that change are not interchangeable, so always check that `num_classes` matches the checkpoint being loaded. Datasets are now reduced to the five classes with `clean_to_x_classes.py`. The older route, `restore_selected_classes.py` (formerly `consolidate_defects.py`), is kept for reference only; the readme explains why.

### Dataset size

Current split (`yolonas-training/training-data/`, built by `build_training_data.py`):

| Dataset | Train tiles (photos) | Val tiles (photos) |
|---|---|---|
| `building-defect-5defect-tiled` | 6,995 (934) | 1,869 (233) |
| `kelvin_bbox-5defect-tiled` | 5,110 (682) | 1,214 (171) |
| `ray_synthetic_crackglass_gt2_5defect_tiled` | 4,189 (656) | 1,034 (164) |
| `tf_rbb-5defect-tiled` | 711 (90) | 209 (22) |
| `tf_r2-5defect-tiled` | 422 (75) | 106 (19) |
| **Total** | **17,427 (2,437)** | **4,432 (609)** |

There is no separate test set. The counts are **tiles**, not photos (§3.3): about 3,000 source photos produce about 22,000 tiles.

Boxes per class:

| Class | Train | Val |
|---|---|---|
| corrosion | 6,307 | 2,418 |
| crack | 20,617 | 6,146 |
| spall | 3,440 | 678 |
| plant_growth | 1,799 | 373 |
| crack_glass | 4,584 | 917 |

`crack` dominates: it has about three times as many boxes as any other class. Almost all `crack_glass` boxes come from `ray_synthetic_crackglass_gt2`, a synthetic dataset: 908 of the 917 in validation. A validation score for `crack_glass` therefore says little about real cracked glazing.

---

## 2. The central design decision: tiled training + sliced inference

Of all the changes made during the project, this one gave the largest improvement in accuracy. It also constrains almost everything else in the pipeline.

*The accuracy observations in this document come from inspecting predictions and comparing runs during development.*

### The problem

Drone façade photos are very high resolution, but the detector takes a fixed 640×640 input. Feeding it a whole photo means shrinking the photo many times over. A hairline crack a few pixels wide in the original becomes less than a pixel wide after shrinking. The information is gone before the model sees the image, and no amount of training can recover it.

### The approach

Cut the photo into overlapping tiles, run the model on each tile separately, then merge the detections back into the original photo's coordinates. In effect the model "zooms in". A 2048 px tile is still shrunk to 640 px (3.2×), but that is far less than the 10× or more needed to fit a whole photo, so much more fine detail survives.

At prediction time this is done by SAHI (Slicing Aided Hyper Inference), wrapped in [`batch_inference/sahi_inference.py`](../batch_inference/sahi_inference.py) and configured in [`batch_inference/config.yaml`](../batch_inference/config.yaml):

```yaml
use_tiled_inference: true
tiled_inference:
  slice_height: 2048
  slice_width: 2048
  overlap_height_ratio: 0.5
  overlap_width_ratio: 0.5
  postprocess_match_threshold: 0.25
merge_annotations:
  enabled: true
  iou_threshold: 0.1
```

The 50% overlap ensures that a defect sitting on a tile boundary appears whole in at least one tile. Overlap creates duplicate detections by design, so two merge steps clean them up: SAHI's own merging (`GREEDYNMM`), then a second pass that joins overlapping boxes of the same class (IoU ≥ 0.1), so a defect split across tiles becomes one box.

### Why 2048 px tiles

The tile size is a middle ground between two failure modes:

- **Too large:** the tile has to be shrunk further to reach 640 px, and fine detail is lost again, the same problem tiling exists to solve.
- **Too small:** the model sees the façade at too fine a scale. More defects are cut across tile edges, more tiles are empty wall, and a single defect is spread over many small overlapping fragments. All of this makes the training labels noisier.

Façade inspection makes a single fixed size workable. The drone flies at a predetermined or predictable distance from the building, so defects of the same kind appear at a broadly similar size in pixels across the dataset.

2048 px was settled on because it works well with the current data. It is not a fixed requirement, and other sizes are worth testing as the data changes. Any change must be made in both the tiler and the inference config (Consequence 1, below).

### Consequence 1: training data must be tiled the same way

The model has to train on images at the same scale it will see at prediction time. If it trained on shrunken whole photos and then predicted on tiles, a crack would look a different size in pixels in the two cases, and the model would generalise poorly.

So training tiles are cut with exactly the same settings as the prediction slices: **2048 px, 50% overlap**, then rescaled to 640 px for the model. If one changes, the other must change too. This is why `tile_building_defect_images.py` and `data/data-tiled/` exist as a separate pipeline stage. Tiling is needed so that training and prediction match; it is not an optional optimisation.

### Consequence 2: prediction is slower

Processing N tiles instead of one image costs roughly N times as much compute. This trade-off is accepted:

- The task does not need real-time results. Inspection reports are produced after the flight, not during it.
- The cost is acceptable: one photo takes seconds, and a full project runs in minutes.
- The accuracy gain is large enough that the alternative is not competitive.

---

## 3. Data pipeline

```
data/data_raw/<name>                      raw photos + YOLO labels (any class list)
   │  clean_to_x_classes.py               keep the 5 classes, drop images left with no boxes
   ▼
data/data-5defect/<name>-5defect
   │  filter_low_res.py                   drop photos smaller than one tile
   │  tile_building_defect_images.py      2048 px tiles, 50% overlap
   ▼
data/data-tiled/<name>-5defect-tiled
   │  build_training_data.py              combine datasets, split by source photo
   ▼
yolonas-training/training-data/{train,val}
```

`dataset_eda.ipynb` is a read-only notebook for inspecting any dataset at any of these stages.

### 3.1 Annotation guidelines and relabelling

A meaningful share of the accuracy improvement came from fixing the annotations, not from changing training settings. Three systematic corrections were made to the existing labels.

**(a) Tightening bounding boxes to the defect.** Most inherited boxes were considerably larger than the defect inside them. This harms training in several compounding ways:

- *Background becomes evidence for the class.* Everything inside a box is treated as an example of that class. A loose box around a crack teaches the model that clean wall is part of what a crack looks like, so it fires on clean wall.
- *Box-edge targets become noisy.* The model learns where box edges belong (§4.2). If the "correct" edge is placed inconsistently from one annotation to the next, the model receives contradictory answers for similar inputs and hedges by predicting wide, low-confidence boxes.
- *The wrong parts of the image are trained as positives.* During training, the model's candidate predictions are matched to labelled boxes partly by overlap. Oversized boxes pull in candidates centred on background, which are then trained as if they were defects.
- *Measurements are wrong.* If box area is ever used as a proxy for defect extent or severity, loose boxes systematically overstate it.

**(b) Relabelling corrosion to cover only the corroded area.** Corrosion boxes used to enclose the whole corroded object, typically an entire air-conditioning unit. The model learned the obvious shortcut: it began detecting air-conditioning units themselves as corrosion, including clean, uncorroded ones. The visual feature that best predicted the label was "air-conditioning unit", not "rust". Restricting boxes to the rusted region removed the shortcut and forced the model to learn rust texture. After relabelling, false corrosion detections on clean units dropped.

This illustrates a general failure mode: the model learns whatever feature inside the box best predicts the label, which is not necessarily the feature that was intended.

**(c) Splitting long cracks into segments.** Some cracks run across an entire wall. One box around the whole run is mostly undamaged wall, an extreme case of (a), and its size is far outside the range of every other crack box. Long cracks are now annotated as a series of tighter boxes along the crack's path.

### 3.2 Class filtering and resolution filtering

**`clean_to_x_classes.py`** reads a raw dataset in any class list and keeps only the five target classes, renumbering them into the standard order. Images left with no boxes are dropped, for the reasons given in §3.3. It also reports label problems found in the source, such as unknown class ids or labels without an image. The raw folder is never modified.

**`filter_low_res.py`** removes photos whose shorter side is under **2048 px**, the tile size. Below that size there is no full tile to cut, and a small photo would have to be enlarged to match the training scale, which adds no real detail. The tiler skips such photos anyway; this step makes the removal explicit and moves the rejects to `rejected_low_res/` so it can be undone.

### 3.3 Tiling

`tile_building_defect_images.py` cuts each photo into **2048×2048 tiles with 50% overlap**, matching the prediction settings (§2). The last row and column of tiles are shifted to sit flush with the image edge, so the whole photo is covered. Tiles are named `<photo>_s2048_x<x>_y<y>`, which is how later steps trace a tile back to its photo.

Box handling:

- A box that crosses a tile edge is **clipped** to the tile. Any box with a non-zero overlap with the tile is kept.
- Tiles with no boxes are dropped (`--keep-empty` keeps them).

**Why defect-free images and tiles are dropped.** Most of a façade is undamaged, so once photos are tiled, most tiles are empty wall. Keeping them would multiply the dataset size for little gain. Dropping them:

- saves disk space;
- shortens training time;
- keeps training focused on what the model has to learn: identifying defects.

The model still learns what background looks like. Every training tile that contains a defect is mostly undamaged surface, and everything outside the labelled boxes is trained as "not a defect".

**The trade-off.** At prediction time, most SAHI slices of a building are clean façade, but the model never trains on a tile that is entirely clean. Whether adding some empty tiles back would reduce false detections is an open question. It can be tested by re-tiling with `--keep-empty` (or keeping a sample of the empty tiles) and comparing against a run without them.

**Known limitation:** there is no minimum fraction of a box that must fall inside a tile. A box that only just crosses into a tile becomes a thin sliver label for an object the tile barely shows. Training augmentation discards some slivers (§4.4), but not ones created at tiling time. Adding a minimum-retained-area rule to the tiler is a cheap candidate improvement.

### 3.4 Splitting

`build_training_data.py` builds the train/val split directly from one or more tiled datasets (it replaced the earlier `merge_datasets.py` + `split_dataset.py`).

- **Split by source photo, not by tile.** All tiles from one photo go to the same side of the split. With 50% overlap, neighbouring tiles share a quarter to a half of their pixels. If tiles from one photo appeared in both train and val, the model would be validated on images it had largely already seen, and the score would be too optimistic. This is what happened in earlier runs: the old `split_dataset.py` shuffled individual tiles 90/10.
- **Split per dataset.** Each dataset contributes its own 80/20 share to train and val, so every dataset is represented in validation.
- **80/20 ratio.** This is a common default. Early runs used 90/10 because there was little data and as much as possible was kept for training. As the dataset grew, 80/20 became affordable.
- **Reproducible.** The split depends only on a seed (42) and each dataset's folder name, so adding a dataset later does not reshuffle the others. `training-data/split_manifest.csv` records which photo went to which split.
- **Checked before anything is written.** The script stops if any dataset is missing `images/`, `labels/` or `classes.txt`, if the `classes.txt` files differ, or if a label row is not a plain five-field YOLO box.
- **Class distribution report.** Each build ends with a report, saved as `training-data/split_report.txt`, showing for every class how many boxes are in train and val, val's share against the target, and how many photos and tiles contain it.

**Class-balanced alternative.** Splitting each dataset 80/20 by photo does not split each class 80/20, because photos differ in which defects they contain and how many. On the current data, each class's share of boxes in validation ranges from 16.5% (spall) to 27.7% (corrosion). `build_training_data_class_balanced.py` is an alternative to `build_training_data.py` with the same inputs, checks and outputs. It chooses the split so that every class is divided at the target ratio, measured in boxes:

- Photos are still kept whole, for the leakage reason above. Since a photo usually contains several classes, an exact split per class is not always possible, so the script works on a best-effort basis and the report flags any class more than 2 percentage points off target.
- Photos are first placed rarest-class-first, each into the split that most needs that class (iterative stratification, a standard method for splitting data where each item carries several labels). The split is then refined by moving and swapping single photos while that brings the classes closer to target.
- On the current data, every class lands within 0.1 percentage points of 20%.
- The trade-off is stability. The same seed gives the same split, but adding a dataset can move existing photos between train and val, because the classes are balanced across all datasets together. `build_training_data.py` keeps each dataset's split fixed, which suits comparing models on the same validation photos.

The latest model was trained on the per-dataset split from `build_training_data.py`.

One dataset, `building-defect`, comes from a Roboflow export, and Roboflow exports can contain several augmented copies of the same original photo under different names. If they did, copies of one photo could still land on both sides of the split. A check of the current split found that 8 originals have more than one copy, and none of them spans train and val.

---

## 4. Training methodology

Everything in this section is in [`yolonas-training/train.ipynb`](../yolonas-training/train.ipynb).

**Most settings are standard defaults; the ones adjusted for this project are marked.** The training parameters follow the usual YOLO-NAS fine-tuning recipe, as published in Roboflow's [YOLO-NAS custom-dataset tutorial](https://github.com/roboflow/notebooks/blob/main/notebooks/train-yolo-nas-on-custom-dataset.ipynb). The augmentations are SuperGradients' default detection pipeline (`coco_detection_yolo_format_base_dataset_params.yaml`). The tables below list every setting with what it does. Settings that differ from those defaults are marked **Changed**, with the default value given for comparison. Anything unmarked is the default.

### 4.1 Model and initialisation

- **Architecture:** YOLO-NAS-M (51.13M parameters), via the SuperGradients library. **Changed:** the tutorial uses the larger YOLO-NAS-L.
- **Input:** 640×640. Each 2048 px tile is rescaled to this size.
- **Output:** `models/building_defect_yolo_nas_m/RUN_<timestamp>/`

**Why YOLO-NAS-M.** This project redeveloped a model from an earlier project, and YOLO-NAS-M was the model already in use. Keeping the architecture avoided a significant change at that stage. Within the YOLO-NAS family, M is the middle ground:

- **S** would have been less accurate, a downgrade from the existing model.
- **L** would have made training time too long because of its larger size.

Newer detector families, notably the transformer-based RT-DETR and RF-DETR, were also considered. There was not enough time to evaluate them properly before accreditation. They are the natural place to start if the model is moved to a newer architecture.

The starting weights are set by `INIT_SOURCE`:

- `"coco"` (default): start from weights pre-trained on COCO, a large general-purpose image dataset. This is transfer learning: the network already recognises general edges, textures and shapes, so on a dataset this size it trains much faster, and to a better result, than starting from random weights.
- `"checkpoint"`: fine-tune from an earlier run's `ckpt_best.pth`.

Separately, `RESUME_TRAINING` with `RESUME_PATH` continues an interrupted run from its `ckpt_latest.pth`. It loads with `resume_strict_load=False`, which tolerates a changed detection head, for example across the three-to-five-class change. The `"checkpoint"` option and the resume option were both added for this project.

### 4.2 Loss and label assignment

```python
PPYoloELoss(use_static_assigner=False, num_classes=5, reg_max=16)
```

This is the recipe's loss, unchanged. Two of its settings are worth understanding:

- **`use_static_assigner=False`** uses the Task-Aligned Assigner. During training, the model must decide which of its many candidate predictions is responsible for each labelled box. A static assigner decides this from geometry alone. The task-aligned assigner scores candidates on classification confidence and box overlap together, so the assignment improves as the model learns.
- **`reg_max=16`** uses Distribution Focal Loss for box edges. Instead of predicting each edge as a single number, the model predicts a probability spread over 16 positions. This copes better with ambiguous boundaries, which matters here: the edge of a spall or a corrosion patch is genuinely fuzzy.

The three loss terms in the training logs are `loss_cls` (is this the right class), `loss_iou` (does the box overlap the labelled box) and `loss_dfl` (the box-edge distribution above).

### 4.3 Optimisation

| Setting | Value | What it does |
|---|---|---|
| Optimiser | **Changed:** AdamW (default Adam) | The algorithm that updates the weights. AdamW applies weight decay separately from the gradient update |
| Weight decay | `1e-4` | Gently pulls weights toward zero to limit overfitting |
| Weight-decay exclusion | `zero_weight_decay_on_bias_and_bn=True` | Exempts biases and normalisation-layer parameters from weight decay, which would otherwise degrade them |
| Initial learning rate | `5e-4` | Size of each weight update at the start of training |
| Schedule | Cosine, `cosine_final_lr_ratio=0.1` | Lowers the learning rate smoothly to 10% of its initial value by the last epoch |
| Warm-up | `linear_epoch_step`, 3 epochs from `1e-6` | Ramps the learning rate up over the first 3 epochs, so early updates don't overwrite the useful COCO features |
| Epochs | **Changed:** 100 (default 25) | Number of passes over the training set |
| Batch size | 8 | Tiles processed together per step. Also the largest that fits in 8 GB of GPU memory at 640×640 |
| Gradient accumulation | **Changed:** `batch_accumulate=4` (default 1, i.e. none) | Adds up 4 batches before each update, for an effective batch of 32: steadier updates without the memory cost |
| Mixed precision | `True` | Uses 16-bit arithmetic where safe: about 2× faster and less memory |

With the current split this is about 2,180 batches per epoch, or about 545 weight updates after accumulation.

Mixed precision occasionally produces NaN (not-a-number) metrics on some GPUs. If metrics go NaN, turn it off before investigating anything else.

### 4.4 Augmentation

Augmentation randomly alters training images each epoch so the model sees more variety than the raw data contains. Validation applies only steps 6–7, so evaluation is deterministic.

The transforms are the SuperGradients default pipeline. Four settings were changed, all toning augmentation down: three are applied to fewer images, and the affine zoom range is narrower.

| # | Transform | Settings | What it does |
|---|---|---|---|
| 1 | `DetectionMosaic` | **Changed:** `prob=0.7` (default 1.0) | Combines 4 images into one: more variety of scale and context |
| 2 | `DetectionRandomAffine` | ±10° rotation, translate 0.1, shear 2.0; **changed:** scale 0.5–1.5 (default 0.1–2.0) | Randomly rotates, shifts, zooms and skews the image, as drone photos are taken at varying angles and distances |
| 3 | `DetectionMixup` | **Changed:** `prob=0.2` (default 1.0); scale 0.5–1.5 | Blends two images together; discourages over-confidence |
| 4 | `DetectionHSV` | **Changed:** `prob=0.5` (default 1.0); gains (5, 30, 30) | Randomly shifts hue, saturation and brightness, for robustness to the varied daylight façades are shot in |
| 5 | `DetectionHorizontalFlip` | `prob=0.5` | Mirrors the image left to right |
| 6 | `DetectionPaddedRescale` | 640×640 | Resizes to the model's input with padding, preserving aspect ratio |
| 7 | `DetectionTargetsFormatTransform` | `LABEL_CXCYWH` | Converts labels to the format the loss expects |

`DetectionRandomAffine` runs with `filter_box_candidates=True` (`wh_thr=2`, `area_thr=0.1`, `ar_thr=20`), also the default. This discards boxes that the transform has squeezed into slivers or near-zero areas.

**No vertical flip.** The default pipeline has none, and none was added on purpose. Façades have a consistent "down", and some defects depend on it (water staining runs downward, spalling sheds downward). Flipping vertically would produce physically implausible training images.

### 4.5 Class imbalance

The classes are heavily imbalanced (§1). Two methods to counter this were built and tested. **Both are currently off** (`USE_COPY_PASTE = False`, `USE_WEIGHTED_SAMPLING = False`) because neither gave better results than training without them. The latest run used neither.

**Class-aware copy-paste** ([`copy_paste_aug.py`](../yolonas-training/copy_paste_aug.py)). At start-up it collects crops of every rare-class box in the training set. During training it pastes up to three of them onto an image at random positions.

| Parameter | Value | Meaning |
|---|---|---|
| `rare_classes` | `[0, 2, 3]` | corrosion, spall, plant_growth |
| `paste_prob` | 0.5 | Applied to half of the images |
| `max_paste` | 3 | Up to 3 pasted instances per image |
| `scale_range` | (0.05, 0.25) | Pasted object spans 5–25% of the image's shorter side |
| `max_overlap` | 0.1 | Rejects positions overlapping an existing box by more than IoU 0.1 |
| `min_px` | 16 | Ignores source crops smaller than this |
| `max_attempts` | 10 | Placement retries before giving up on a crop |

It was introduced to replace weighted sampling, because weighted sampling repeats *whole images*. An image containing one rare spall also contains its common-class boxes, which get repeated along with it, and repeating the same image adds no new context, so the model can memorise those instances. Copy-paste puts rare instances onto **varied backgrounds**, which is closer to what the imbalance problem needs.

In practice it did not improve results, and it was switched off by default on 1 October 2026 (commit `97e7514`). That was before the data pipeline was rebuilt, so it has not yet been re-tested on the current split. Known limitations:

- The paste is a hard rectangle with no blending, so the seams are visible. This is tolerable for texture-like defects (corrosion, spall) but a poor fit for thin linear ones such as cracks. `crack` is not in `rare_classes` in any case, as it is the most common class.
- It can only recombine instances that already exist. A class with few *distinct* instances gains variety of background, not new examples.

**Inverse-frequency weighted sampling** (legacy). This gives each image a weight equal to the highest inverse-frequency weight among its classes, and draws images with replacement using `WeightedRandomSampler`. It sometimes made the model worse. Accuracy dropped on classes the model had been strong at, notably `crack`, while the less common classes did no better. This matches the whole-image problem above: images are picked for their rare classes, so the balance of everything else in the training set shifts with them. It was superseded by copy-paste, and then switched off.

### 4.6 Weight averaging

Two mechanisms run together:

- **EMA** (`decay_type="threshold"`; **changed:** `decay=0.9999`, default 0.9) keeps a slowly updated running average of the weights, which is what gets evaluated and saved. The threshold setting ramps the averaging in gradually, so the average is not dominated by the poor weights of the first few iterations.
- **`average_best_models=True`** averages the highest-scoring checkpoints of the run into `average_model.pth`.

Both reduce the noise of stopping at one arbitrary point in training.

A run writes three checkpoints: `ckpt_best.pth` (best validation mAP@0.50), `ckpt_latest.pth` (last epoch) and `average_model.pth`. Both the notebook's final evaluation and `batch_inference` use **`ckpt_best.pth`**.

### 4.7 Evaluation

```python
DetectionMetrics_050(
    score_thres=0.1, top_k_predictions=300, normalize_targets=True,
    include_classwise_ap=True,
    post_prediction_callback=PPYoloEPostPredictionCallback(
        score_threshold=0.01, nms_top_k=1000,
        max_predictions=300, nms_threshold=0.5))
metric_to_watch = 'mAP@0.50'
```

This is the recipe's metric, with two changes:
- **Changed:** `nms_threshold=0.5` (default 0.7). Overlapping predictions of the same class are merged more aggressively before scoring.
- **Added:** `include_classwise_ap=True`, which reports AP for each class as well as the overall mAP.

**mAP@0.50** is the standard detection score. A prediction counts as correct if it has the right class and overlaps the labelled box by at least 50% (IoU ≥ 0.5). For each class, average precision (AP) summarises the trade-off between precision and recall across all confidence thresholds. mAP is the mean of the per-class APs, so 1.0 is perfect. Per-class AP is reported every epoch.

**Why mAP@0.50 rather than the stricter mAP@0.50:0.95.** mAP@0.50:0.95, the COCO standard, averages the score over IoU thresholds from 0.50 up to 0.95, so it rewards boxes that match the label almost exactly. That suits objects with crisp edges, such as cars or people. It suits façade defects poorly:

- **The edges are genuinely ambiguous.** Where a spall or a patch of corrosion ends is a judgement call, and two careful annotators will draw different boxes. At high IoU thresholds the score would mostly measure agreement with one annotator's edge placement, not whether the defect was found.
- **Cracks are labelled as segments** (§3.1c). A prediction can cover a crack correctly but split it at different points from the labels. It then overlaps each labelled segment only partly and fails the strict thresholds, even though it is operationally correct.
- **The operational need is to locate a defect**, so an inspector can find and assess it on the façade. A box that clearly covers the defect does this, whether or not its edges are exact.

mAP@0.50 is also the recipe's default for fine-tuning. It is the metric that selects `ckpt_best.pth` (`metric_to_watch`).

**The trade-off:** mAP@0.50 says little about how tight the boxes are. If box size is ever used to estimate defect extent or severity (§3.1a), box accuracy should be checked separately, for example by also reporting mAP@0.50:0.95 or the average IoU of correct detections.

Evaluation is on 640×640 **tiles** from the validation photos, not on whole photos through SAHI. Scores on whole photos, after SAHI's merging, may differ.

A `NotebookMetricsRecorder` callback records every epoch's metrics for plotting in the notebook. After training, the notebook writes `run_info.txt` into the run folder, with the datasets, settings and best-epoch scores. Runs from before this feature have no `run_info.txt`; their scores are only in the training logs in the run folder.

---

## 5. Tooling

| Tool | Role |
|---|---|
| **SuperGradients** 3.7.1 (Deci) | Training framework: YOLO-NAS, `Trainer`, data loaders, augmentations, PPYoloE loss, detection metrics |
| **PyTorch** | Underlying framework: mixed precision, `DataLoader`, `WeightedRandomSampler` |
| **SAHI** | Sliced prediction on whole photos |
| **OpenCV / NumPy / Pillow** | Image reading, tiling, crop-and-paste, drawing boxes |
| **Label Studio** | Annotation review. `yolo_to_ls_predictions.py` converts YOLO labels into Label Studio import JSON, so existing boxes can be loaded and corrected instead of drawn from scratch |
| **Jupyter** | Training notebook and dataset inspection |
| **pyenv-virtualenv** | Python 3.10.19 environment (SuperGradients supports up to 3.10) |

Environment setup is `setup_venv.sh`, with dependencies in `requirements.txt`. Three environment problems were found and fixed. Each is recorded in the readme so it is not rediscovered:

- SuperGradients' download server for the COCO weights no longer exists. `setup_venv.sh` fetches the weights from a mirror into torch's cache.
- PyTorch 2.6 and later refuse to load a file SuperGradients writes during training, which crashes the run after the second epoch. The notebook's first cell sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` to prevent this.
- Newer glibc versions refuse to load the pinned `onnxruntime` library. It needs a one-off `patchelf --clear-execstack` per virtualenv.

---

## Appendix: settings quick reference

```python
MODEL_ARCH  = "yolo_nas_m"     # 51.13M params
IMG_SIZE    = [640, 640]
MAX_EPOCHS  = 100
BATCH_SIZE  = 8                # x4 accumulation = effective 32
INIT_SOURCE = "coco"
USE_COPY_PASTE        = False
USE_WEIGHTED_SAMPLING = False

train_params = {
    "average_best_models": True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "batch_accumulate": 4,
    "optimizer": "AdamW",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9999, "decay_type": "threshold"},
    "max_epochs": MAX_EPOCHS,
    "resume_strict_load": False,
    "mixed_precision": True,
    "loss": PPYoloELoss(use_static_assigner=False, num_classes=5, reg_max=16),
    "metric_to_watch": "mAP@0.50",
}
```

Data pipeline defaults: 5 classes in the order of §1; minimum photo side 2048 px; 2048 px tiles at 50% overlap, boxes clipped, empty tiles dropped; 80/20 split by source photo, seed 42. Inference: 2048 px slices at 50% overlap, confidence threshold 0.3, merge at IoU 0.1.
