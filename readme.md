# defect-detection

Consolidated pipeline for building-defect detection from drone/UAV imagery: raw data → cleaning → training (YOLO-NAS) → batch inference. This repo unifies three previously separate projects (`batch_inference`, `data-cleaning`, `yolonas-training`) around one shared `data/` and `models/` directory.

## Status

This is a **structural consolidation only**. Code (scripts + notebooks) has been migrated and path-adjusted; the actual image datasets and model checkpoints (~300GB combined across the original projects) have **not** been copied yet. All `data/`, `models/`, and per-project data folders exist as empty skeletons (tracked via `.gitkeep`).

## Pipeline flow

```
data-cleaning/                 data/                          yolonas-training/            batch_inference/
  scripts + notebooks   --->     data_raw/       (raw sources)   train.ipynb          --->    reads models/
  read + write data/             data-cleaned/   (filtered)      reads data/ to split         own input-data/
                                  data-tiled/     (tiled)         into a local copy at         + output-preds/
                                  data-5defect/   (5-class)       yolonas-training/data/
                                                                  writes checkpoints to
                                                                  ../models/
                                                        \
                                                         ---> models/  <--- batch_inference reads this
```

- **`data-cleaning/`** reads and writes `../data/{data_raw,data-cleaned,data-tiled,data-5defect}`.
- **`yolonas-training/`** reads from `../data/` (typically `data-tiled/` or `data-5defect/`) to build its train/val/test split, but persists that split as a single local copy at `yolonas-training/data/{train,val,test}` — not a second copy of the source data. It writes model checkpoints to the shared top-level `../models/`.
- **`batch_inference/`** reads checkpoints from `../models/` and keeps its own local `input-data/` and `output-preds/`, independent of the shared `data/`.

## Folder structure

```
defect-detection/
├── data/                  # shared raw/intermediate datasets (data-cleaning stage)
│   ├── data_raw/            # original/raw source datasets
│   ├── data-cleaned/        # filtered (e.g. min-resolution, empty-label removal)
│   ├── data-tiled/          # tiled crops for training
│   └── data-5defect/        # consolidated 5-class taxonomy (corrosion, crack, spall, plant_growth, crack_glass)
├── models/                 # shared checkpoints, written by yolonas-training, read by batch_inference
│   └── building_defect_yolo_nas_m/RUN_*/...   (created by SuperGradients Trainer)
├── data-cleaning/
│   ├── scripts/
│   │   ├── consolidate_defects.py       # builds data/data-5defect/* from data_raw/ + data-cleaned/
│   │   ├── tile_building_defect_images.py  # builds data/data-tiled/* from a cleaned dataset
│   │   └── yolo_to_ls_predictions.py    # YOLO labels -> Label Studio prediction JSON (annotation-review workflow)
│   ├── data-cleaning.ipynb              # filters raw building-defect-detection dataset
│   └── kelvin_bbox_cleaning.ipynb       # EDA + cleaning for tf_rbb_all (filename is a pre-existing misnomer)
├── yolonas-training/
│   ├── train.ipynb                      # main YOLO-NAS-M training pipeline
│   ├── copy_paste_aug.py                # custom class-aware copy-paste augmentation
│   ├── scripts/
│   │   ├── copy_dataset.py              # splits a source dataset into data/{train,val,test}
│   │   ├── merge_datasets.py            # merges multiple ../data/{data-tiled,data-5defect}/* sources
│   │   ├── cleanup_checkpoints.py       # prunes RUN_* checkpoint folders under ../models/
│   │   └── clear_splits.sh              # wipes and recreates local data/{train,val,test}
│   └── data/                            # the one persisted copy of the train/val/test split
└── batch_inference/
    ├── batch_inference.py               # main entry point
    ├── sahi_inference.py                # SAHI tiled-inference wrapper
    ├── clear_output_preds.py            # utility to clear output-preds/
    ├── config.yaml                      # runtime configuration (checkpoint_path -> ../models/...)
    ├── input-data/                      # local inference inputs (own copy, not shared)
    └── output-preds/                    # timestamped run outputs
```

## Running each stage

**1. Data cleaning** (from `data-cleaning/`):
```bash
jupyter notebook data-cleaning.ipynb        # filters raw dataset -> ../data/data-cleaned/
jupyter notebook kelvin_bbox_cleaning.ipynb # EDA + cleaning for tf_rbb_all -> ../data/data-cleaned/
python scripts/consolidate_defects.py       # -> ../data/data-5defect/ (defaults to ../data)
python scripts/tile_building_defect_images.py --src ../data/data-cleaned/<name> --dest ../data/data-tiled/<name>
```

**2. Training** (from `yolonas-training/`):
```bash
# optional: merge multiple tiled/5-defect sources from ../data/ into a local scratch dir
python scripts/merge_datasets.py --datasets ../data/data-tiled/<a> ../data/data-tiled/<b> --out datasets/merged

# split into the local train/val/test copy
python scripts/copy_dataset.py --src ../data/data-5defect/<name> --dst data

jupyter notebook train.ipynb   # writes checkpoints to ../models/building_defect_yolo_nas_m/RUN_*/
```

**3. Inference** (from `batch_inference/`):
```bash
# update config.yaml's checkpoint_path to a real RUN_* under ../models/building_defect_yolo_nas_m/
python batch_inference.py --config config.yaml
```

## Setup

```bash
pip install -r requirements.txt
```

## Known issues (pre-existing, not introduced by this consolidation)

- Several data-cleaning scripts/notebooks have output folder names that don't match what's actually on disk in the original projects (e.g. `building-defect-detection` vs `building-defect-cleaned`) — inherited as-is, not silently fixed during migration.
- `kelvin_bbox_cleaning.ipynb`'s filename doesn't match its actual content (it processes `tf_rbb_all`, not kelvin_bbox data).

## What was intentionally left out of this consolidation

A few files from the original projects were reviewed and dropped rather than migrated, since they were dead, redundant, or superseded:
- `data-cleaning/scripts/resize_building_defect_images.py` — hardcoded paths pointing at folders that don't exist anywhere in the data tree; no evidence it was ever run.
- `data-cleaning/data-overview.ipynb` — pure EDA/visualization, not a pipeline step.
- `yolonas-training/inference.ipynb` and `inference-tiling.ipynb` — broken hardcoded paths, and superseded by `batch_inference`'s production inference path.
- `yolonas-training/scripts/copy_dataset_tree.sh` — its only job (staging a copy of `data-cleaned` into a local `datasets/` duplicate) is obsoleted by the shared top-level `data/`.
- `yolonas-training/old scripts/building_defect_yolo_nas_m.ipynb` — legacy/superseded (older 3-class taxonomy, different checkpoint/data layout).
