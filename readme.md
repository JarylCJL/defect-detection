# defect-detection

Consolidated pipeline for building-defect detection from drone/UAV imagery: raw data → cleaning → training (YOLO-NAS) → batch inference. This repo unifies three previously separate projects (`batch_inference`, `data-cleaning`, `yolonas-training`) around one shared `data/` and `models/` directory.

## Status

Code (scripts + notebooks) has been migrated and path-adjusted, and the shared `data/` tree (`data_raw/`, `data-cleaned/`, `data-tiled/`, `data-5defect/` — ~168GB total) has been copied in from the original `data-cleaning` project. None of it is tracked in git (see `.gitignore`) — it exists only on disk, locally.

`models/`, `yolonas-training/training-data/{train,val,test}`, and `batch_inference/{input-data,output-preds}` are still empty skeletons (tracked via `.gitkeep`). These aren't filled by data migration — they get populated by actually running the pipeline: `build_training_data.py` produces the local train/val/test copy, `train.ipynb` writes checkpoints to `models/`, and dropping images into `input-data/` + running inference produces `output-preds/`.

## Pipeline flow

```
data-cleaning/                 data/                          yolonas-training/            batch_inference/
  scripts + notebooks   --->     data_raw/       (raw sources)   train.ipynb          --->    reads models/
  read + write data/             data-cleaned/   (filtered)      reads data/ to split         own input-data/
                                  data-tiled/     (tiled)         into a local copy at         + output-preds/
                                  data-5defect/   (5-class)       yolonas-training/training-data/
                                                                  writes checkpoints to
                                                                  ../models/
                                                        \
                                                         ---> models/  <--- batch_inference reads this
```

- **`data-cleaning/`** reads and writes `../data/{data_raw,data-cleaned,data-tiled,data-5defect}`.
- **`yolonas-training/`** reads from `../data/` (typically `data-tiled/` or `data-5defect/`) to build its train/val/test split, but persists that split as a single local copy at `yolonas-training/training-data/{train,val,test}` — not a second copy of the source data. It writes model checkpoints to the shared top-level `../models/`.
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
│   │   ├── clean_to_x_classes.py           # raw -> data/data-cleaned/*: keep chosen classes, drop images left without annotations
│   │   ├── filter_low_res.py               # drops images below a minimum resolution (default 2048) from a cleaned dataset, in place
│   │   ├── restore_selected_classes.py     # builds data/data-5defect/*: cleaned 3-class labels + selected classes restored from data_raw/
│   │   ├── tile_building_defect_images.py  # builds data/data-tiled/* from a cleaned dataset
│   │   └── yolo_to_ls_predictions.py    # YOLO labels -> Label Studio prediction JSON (annotation-review workflow)
│   └── dataset_eda.ipynb                # read-only EDA for any YOLO dataset (classes, resolutions, box sizes)
├── yolonas-training/
│   ├── train.ipynb                      # main YOLO-NAS-M training pipeline
│   ├── copy_paste_aug.py                # custom class-aware copy-paste augmentation
│   ├── scripts/
│   │   ├── build_training_data.py       # builds training-data/{train,val,test} directly from one or more ../data/data-tiled/* datasets
│   │   ├── cleanup_checkpoints.py       # prunes RUN_* checkpoint folders under ../models/
│   │   └── clear_splits.sh              # wipes and recreates local training-data/{train,val,test} (build_training_data.py now clears the split itself)
│   └── training-data/                   # the one persisted copy of the train/val/test split
└── batch_inference/
    ├── batch_inference.py               # main entry point
    ├── sahi_inference.py                # SAHI tiled-inference wrapper
    ├── clear_output_preds.py            # utility to clear output-preds/
    ├── config.yaml                      # runtime configuration (checkpoint_path -> ../models/...)
    ├── README.md                        # original project README (setup/usage detail)
    ├── input-data/                      # local inference inputs (own copy, not shared)
    └── output-preds/                    # timestamped run outputs
```

## Running each stage

**1. Data cleaning** (from `data-cleaning/`):
```bash
jupyter notebook dataset_eda.ipynb     # read-only EDA: set DATASET_DIR at the top (raw, cleaned or tiled dataset)

# keep chosen classes + drop images left without annotations -> ../data/data-cleaned/<name>_cleaned
# (--classes is the file listing the source classes: classes.txt, data.yaml or notes.json;
#  --keep order sets the output class ids; names with spaces: plant_growth or "plant growth")
python scripts/clean_to_x_classes.py \
  --images ../data/data_raw/<name>/images \
  --labels ../data/data_raw/<name>/labels \
  --classes ../data/data_raw/<name>/classes.txt \
  --dest ../data/data-cleaned/<name>_cleaned \
  --keep corrosion crack spall

# drop images whose shorter side is below 2048 px (in place; rejects moved to <src>/rejected_low_res/)
python scripts/filter_low_res.py --src ../data/data-cleaned/<name>_cleaned --min-side 2048

python scripts/restore_selected_classes.py  # -> ../data/data-5defect/ (defaults to ../data)

# NOTE: the tiler resolves relative --src/--dest against the shared data/ dir, not the current directory
python scripts/tile_building_defect_images.py --src data-cleaned/<name> --dest data-tiled/<name>
```

`clean_to_x_classes.py` and `filter_low_res.py` resolve paths against the current directory. Passing all five classes (`--keep corrosion crack spall plant_growth crack_glass`) produces a 5-defect dataset directly, without `restore_selected_classes.py`, and keeps images that only contain `plant_growth` / `crack_glass` (the 3-class route drops those before the restore step can see them).

**2. Training** (from `yolonas-training/`):
```bash
# build the local train/val/test copy straight from one or more datasets (no intermediate merged dataset)
# - every dataset needs images/, labels/ and an identical classes.txt (same names, same order), or it stops with an error
# - splits by source photo, per dataset, so tiles of one photo never straddle train and val
# - clears training-data/{train,val,test} itself; --dst defaults to yolonas-training/training-data regardless of CWD
python scripts/build_training_data.py \
  --datasets ../data/data-tiled/<a> ../data/data-tiled/<b> \
  --val-ratio 0.2            # the default (80/20 train/val); add --dry-run to check the datasets and preview the split without writing

jupyter notebook train.ipynb   # writes checkpoints to ../models/building_defect_yolo_nas_m/RUN_*/
```

`train.ipynb` starts from the YOLO-NAS-M COCO pretrained weights (`INIT_SOURCE = "coco"`). super-gradients 3.7.1 tries to download them from `sghub.deci.ai`, which no longer exists, so the file has to be in torch's cache first; `setup_venv.sh` puts it there. To do it by hand:
```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl -fL -o ~/.cache/torch/hub/checkpoints/yolo_nas_m_coco.pth \
  https://d2gjn4b69gu75n.cloudfront.net/models/yolo_nas_m_coco.pth
```
The weights are subject to the YOLO-NAS licence terms (`LICENSE.YOLONAS.md` in the super-gradients repository).

**3. Inference** (from `batch_inference/`):
```bash
# update config.yaml's checkpoint_path to a real RUN_* under ../models/building_defect_yolo_nas_m/
python batch_inference.py --config config.yaml
```

## Setup

Uses a dedicated `pyenv-virtualenv` named `defect-detection` on Python 3.10.19 (the same Python version as the reference `yolonas-training` env — `super-gradients` only officially supports up to 3.10). The `.python-version` file at the repo root auto-activates it via pyenv in any subdirectory.

**On a new machine**, run the bootstrap script — it installs `pyenv` itself if missing, installs Python 3.10.19, creates the `defect-detection` virtualenv, and installs `requirements.txt`, so you don't need to know any of the above up front:
```bash
./setup_venv.sh
```
If this is the first time `pyenv` has been installed on the machine, its shell hooks only take effect in a new shell session — the script will tell you to restart your terminal and rerun it once.

**If pyenv + the virtualenv already exist** (e.g. this machine), you can skip straight to:
```bash
pyenv virtualenv 3.10.19 defect-detection   # one-time setup
cd defect-detection                         # auto-activates via .python-version
pip install -r requirements.txt
```

## Data provenance (read before trying to regenerate anything)

- Several data-cleaning scripts/notebooks have output folder names that don't match what's actually on disk in the original projects (e.g. `building-defect-detection` vs `building-defect-cleaned`) — inherited as-is, not silently fixed during migration.
- `data/data-cleaned/building-defect-cleaned` (1,167 images) cannot be regenerated from `data_raw/` — treat it as a source of truth. It started from the raw `building-defect-detection-3` set with empty-label images dropped and a 6000 px minimum on the shorter side (stricter than the 2048 default used elsewhere; it keeps only the 8000x6000 images), which leaves 1,869 images, and was then reviewed and re-exported from Label Studio (hence its `notes.json`, the 702 further images removed, and a few edited label files). The equivalent of that first step today is `clean_to_x_classes.py --classes <raw>/data.yaml --keep corrosion crack spall` followed by `filter_low_res.py --min-side 6000`.

## What was intentionally left out of this consolidation

A few files from the original projects were reviewed and dropped rather than migrated, since they were dead, redundant, or superseded:
- `data-cleaning/scripts/resize_building_defect_images.py` — hardcoded paths pointing at folders that don't exist anywhere in the data tree; no evidence it was ever run.
- `data-cleaning/data-overview.ipynb` — pure EDA/visualization, not a pipeline step.
- `data-cleaning/data-cleaning.ipynb` — the original building-defect cleaning notebook (drop empty labels, 6000 px minimum). Superseded by `clean_to_x_classes.py` + `filter_low_res.py`; see "Data provenance" above for the parameters it used.
- `yolonas-training/inference.ipynb` and `inference-tiling.ipynb` — broken hardcoded paths, and superseded by `batch_inference`'s production inference path.
- `yolonas-training/scripts/copy_dataset_tree.sh` — its only job (staging a copy of `data-cleaned` into a local `datasets/` duplicate) is obsoleted by the shared top-level `data/`.
- `yolonas-training/old scripts/building_defect_yolo_nas_m.ipynb` — legacy/superseded (older 3-class taxonomy, different checkpoint/data layout).
