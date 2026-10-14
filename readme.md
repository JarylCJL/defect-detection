# defect-detection

Building-defect detection from drone/UAV imagery, end to end: raw labelled photos → cleaned datasets → tiles → YOLO-NAS-M training → batch inference on new photos.

The model detects five defect classes, always in this order:

| id | class |
|---|---|
| 0 | `corrosion` |
| 1 | `crack` |
| 2 | `spall` |
| 3 | `plant_growth` |
| 4 | `crack_glass` |

The repo has three stages, each in its own folder, sharing one `data/` and one `models/` directory:

| Stage | Folder | Reads | Writes |
|---|---|---|---|
| 1. Data preparation | `data-cleaning/` | `data/data_raw/` | `data/data-5defect/`, `data/data-tiled/` |
| 2. Training | `yolonas-training/` | `data/data-tiled/` | `yolonas-training/training-data/`, `models/` |
| 3. Inference | `batch_inference/` | `models/`, `batch_inference/input-data/` | `batch_inference/output-preds/` |

Only code is tracked in git. `data/`, `models/`, the train/val/test split and inference inputs/outputs exist on disk only (see `.gitignore`).

This readme says how to run each stage. [`docs/training.md`](docs/training.md) explains why the pipeline and training are set up this way.

## Quick start: run the model on new photos

If you have a trained model and only want predictions, this is all you need. Stages 1 and 2 further down are only for adding data or retraining.

1. Set up the environment once: `./setup_venv.sh` (see [Setup](#setup)).
2. Put the model's `ckpt_best.pth` somewhere under `models/`, and the photos in a folder under `batch_inference/input-data/`.
3. In `batch_inference/config.yaml`, set `input_data` to the photo folder and `checkpoint_path` to the model file. Both are relative to `batch_inference/`, e.g. `input-data/my-photos` and `../models/<folder>/ckpt_best.pth`.
4. Run it from `batch_inference/`:
   ```bash
   cd batch_inference
   python batch_inference.py --config config.yaml
   ```
5. Open `batch_inference/output-preds/<timestamp>/images-overlaid/` to see the photos with the detected defects drawn on them.

[Stage 3](#stage-3-inference) explains the settings and the other output files.

## Pipeline at a glance

```
data/data_raw/<name>                      raw photos + YOLO labels (any class list)
   │  dataset_eda.ipynb                   look before you clean (read-only)
   │  clean_to_x_classes.py               keep the 5 classes, drop images with nothing left
   ▼
data/data-5defect/<name>-5defect          images/ + labels/ + classes.txt
   │  filter_low_res.py                   drop photos smaller than a tile (in place)
   │  tile_building_defect_images.py      2048 px tiles, 50% overlap
   ▼
data/data-tiled/<name>-5defect-tiled      images/ + labels/ + classes.txt
   │  build_training_data.py              combine datasets, split by source photo
   ▼
yolonas-training/training-data/           train/ val/ test/ + classes.txt
   │  train.ipynb                         YOLO-NAS-M, COCO start, 640 px input
   ▼
models/building_defect_yolo_nas_m/RUN_*/  ckpt_best.pth
   │  batch_inference.py                  SAHI: 2048 px slices, 50% overlap
   ▼
batch_inference/output-preds/<timestamp>/ labels + overlaid images
```

## Conventions

These hold across every stage; most errors come from breaking one of them.

- **Dataset layout.** Every dataset folder, at every step, is `images/` + `labels/` + `classes.txt`. Images and labels are paired by file name; both folders are flat.
- **Class list.** `classes.txt` has one name per line, and the line number is the class id. Datasets that will be trained together must have identical `classes.txt` files (same names, same order): `build_training_data.py` refuses to combine them otherwise. Use the five-class list above.
- **Label format.** Plain YOLO box rows only: `class cx cy w h`, five fields, values normalised to 0–1. Polygon, oriented-box and COCO labels are not supported; convert them to boxes first.
- **Tile size = inference slice size.** Training tiles and SAHI inference slices are both 2048 px with 50% overlap, and both are rescaled to 640 px for the model. Change one and you must change the other.
- **Paths.** Scripts resolve relative paths against the current directory, with one exception: `tile_building_defect_images.py` resolves `--src`/`--dest` against `data/`.
- **Script headers.** Every script starts with a comment block saying what it does and how to use it; that block is the reference for its flags.

## Setup

**Requirements**

- Linux, or Windows with WSL2. Developed and tested on Ubuntu 26.04 under WSL2.
- An NVIDIA GPU with at least 8 GB of memory and a working driver: `nvidia-smi` must list the GPU. No separate CUDA install is needed; PyTorch brings its own.
- Disk space: the full set of datasets under `data/` is about 170 GB, the train/val split adds about 26 GB, and each trained run about 1.5 GB.
- `git`, `curl` and a C build toolchain, which `pyenv` uses to compile Python.

**Install**

Python 3.10.19 in a `pyenv-virtualenv` named `defect-detection` (`super-gradients` only supports up to 3.10). `.python-version` at the repo root activates it automatically in any subdirectory.

```bash
./setup_venv.sh
```

The script installs `pyenv` if missing, installs Python, creates the virtualenv, installs `requirements.txt`, and downloads the YOLO-NAS-M COCO pretrained weights into torch's cache. If it had to install `pyenv`, restart the shell and run it again.

Two things worth knowing:

- **COCO weights.** `super-gradients` 3.7.1 tries to download pretrained weights from `sghub.deci.ai`, which no longer exists. With the file already in `~/.cache/torch/hub/checkpoints/`, it loads from there instead. `setup_venv.sh` handles this; by hand it is:
  ```bash
  mkdir -p ~/.cache/torch/hub/checkpoints
  curl -fL -o ~/.cache/torch/hub/checkpoints/yolo_nas_m_coco.pth \
    https://d2gjn4b69gu75n.cloudfront.net/models/yolo_nas_m_coco.pth
  ```
  The weights are subject to the YOLO-NAS licence (`LICENSE.YOLONAS.md` in the super-gradients repository).
- **`cannot enable executable stack` when importing `super_gradients`.** Newer glibc (seen on Ubuntu 26.04, glibc 2.43) refuses to load the `onnxruntime` build that super-gradients pins. Fix it once per virtualenv:
  ```bash
  pip install patchelf
  patchelf --clear-execstack "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"/onnxruntime/capi/onnxruntime_pybind11_state*.so
  ```
  `setup_venv.sh` does not do this; repeat it if the virtualenv is rebuilt.

**Check that it worked**

Run these from the repo root. Nothing here changes any file.

```bash
# 1. Libraries import and the GPU is visible (keep the -u: without it the output is swallowed)
python -u -c "import torch, super_gradients, sahi; print('torch', torch.__version__, '| super-gradients', super_gradients.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')"

# 2. The COCO starting weights are in place
ls ~/.cache/torch/hub/checkpoints/yolo_nas_m_coco.pth

# 3. Once the datasets are in data/: they are complete and consistent (checks only, writes nothing)
python yolonas-training/scripts/build_training_data.py --dry-run \
  --datasets data/data-tiled/<a> data/data-tiled/<b> ...
```

Expected: the first prints the library versions and the GPU's name, the second lists the file, and the third ends with `Dry run: all checks passed`.

## Stage 1: Data preparation

Run from `data-cleaning/`. The example brings a new raw dataset called `<name>` into the pipeline.

**1. Look at the raw data.** Open `dataset_eda.ipynb`, set `DATASET_DIR` to `../data/data_raw/<name>` and run it. It only reads. It reports image/label pairing, boxes per class, resolutions, how many images are smaller than a tile, and box sizes in pixels per class. It works on cleaned and tiled datasets too.

**2. Keep the five classes.**
```bash
python scripts/clean_to_x_classes.py \
  --images  ../data/data_raw/<name>/images \
  --labels  ../data/data_raw/<name>/labels \
  --classes ../data/data_raw/<name>/classes.txt \
  --dest    ../data/data-5defect/<name>-5defect \
  --keep corrosion crack spall plant_growth crack_glass
```
- `--classes` is the file listing the *source* dataset's classes: `classes.txt`, `data.yaml` (Roboflow-style exports, usually one level above `train/`) or `notes.json` (Label Studio).
- `--keep` lists the classes to keep; its order sets the output ids, so always use the order above. Names are normalised (spaces become `_`, lowercase), so a source class `plant growth` is matched by `plant_growth`.
- Images left with no annotations are dropped. The raw folder is never modified.
- It prints a summary including label problems found in the source (unknown class ids, labels with no image, invalid rows).

**3. Drop photos smaller than a tile.**
```bash
python scripts/filter_low_res.py --src ../data/data-5defect/<name>-5defect
```
Removes images whose shorter side is under `--min-side` (default 2048), in place, moving them to `<src>/rejected_low_res/`. The tiler would skip these anyway; this makes it visible. `--dry-run` lists them without touching anything.

**4. Tile.**
```bash
python scripts/tile_building_defect_images.py \
  --src data-5defect/<name>-5defect \
  --dest data-tiled/<name>-5defect-tiled
```
Paths here are relative to `data/`. Defaults are 2048 px tiles with 50% overlap. Tiles are named `<photo>_s2048_x<x>_y<y>`, boxes are clipped to each tile, tiles with no boxes are dropped (`--keep-empty` keeps them), and `classes.txt` is carried over.

## Stage 2: Training

Run from `yolonas-training/`.

**1. Build the train/val split.**
```bash
python scripts/build_training_data.py \
  --datasets ../data/data-tiled/<a> ../data/data-tiled/<b> ...
```
- Writes straight into `training-data/{train,val,test}`; there is no intermediate merged dataset. It clears the previous split first.
- Checks everything before touching a file: every dataset has `images/`, `labels/` and `classes.txt`; all `classes.txt` are identical; every label row is a valid five-field YOLO row. On a mismatch it lists which datasets disagree.
- Splits by **source photo**, per dataset: all tiles cut from one photo go to the same split, so overlapping tiles never leak between train and val, and every dataset is represented in val.
- Default is 80/20 train/val. `--val-ratio`, `--test-ratio` and `--seed` change it; a dataset's split depends only on the seed and its folder name, so adding a dataset later does not reshuffle the others.
- `--dry-run` runs the checks and prints the report (tiles per dataset per split, boxes per class, class id mapping) without writing.
- Also writes `training-data/split_manifest.csv`, recording which photo went to which split.

**2. Train.** Open `train.ipynb`, select the `defect-detection` kernel, and run all cells.

| Setting | Value |
|---|---|
| Model | YOLO-NAS-M, starting from COCO weights (`INIT_SOURCE = "coco"`) |
| Input size | 640×640 |
| Epochs | 100 |
| Batch | 8, with 4× gradient accumulation (effective 32) |
| Optimiser | AdamW, lr 5e-4, 3 warmup epochs, cosine decay |
| Augmentation | mosaic, random affine, mixup, HSV, horizontal flip |
| Validation | mAP@0.50 overall and AP@0.50 per class, every epoch |
| Optional, off by default | `USE_COPY_PASTE` (rare-class copy-paste, `copy_paste_aug.py`), `USE_WEIGHTED_SAMPLING` |

- **Output.** Checkpoints go to `../models/building_defect_yolo_nas_m/RUN_<timestamp>/`: `ckpt_best.pth` (use this for inference), `ckpt_latest.pth`, `average_model.pth`.
- **Run record.** After training, the cell following the one that loads the best model writes `run_info.txt` into the same folder: the datasets and tile counts, the main settings, the best epoch and its validation scores overall and per class. It reads the metrics recorded during training, so run it in the same session as the training cell. Hand the whole `RUN_` folder over with the model so the record goes with it.
- **To continue from an existing model** instead of COCO weights, set `INIT_SOURCE = "checkpoint"` and `CHECKPOINT_PATH`.
- **To resume an interrupted run**, set `RESUME_PATH` to that run's `ckpt_latest.pth` and `RESUME_TRAINING = True`. The resumed run writes into a new `RUN_` folder.
- **PyTorch note.** The first cell sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`. Without it, training crashes after the second epoch on PyTorch 2.6 or newer; leave it in.

**Hardware guidance** (measured on an RTX 5070 Laptop GPU, 8 GB, 24-thread CPU, 15 GB RAM under WSL):

- Keep `BATCH_SIZE = 8`. It peaks around 7 GB of GPU memory on real batches. A larger batch is no faster, and going past 8 GB does not fail: the GPU spills into shared system memory and training runs about ten times slower.
- Keep `NUM_WORKERS = 8`. The GPU is the bottleneck; more workers only add CPU load.
- Expect about 3.5 iterations/s, roughly 10 minutes per training epoch on ~17,000 tiles. If the progress bar drops below 1 it/s, memory has spilled: stop and restart with `BATCH_SIZE = 4` and `batch_accumulate` 8.
- On WSL, a Windows sleep ends the run. Keep the machine awake and plugged in.

## Stage 3: Inference

Run from `batch_inference/`; paths in the config are relative to that folder.

1. Put the photos to score in a folder under `input-data/`.
2. Edit `config.yaml`: set `input_data` to that folder and `checkpoint_path` to the model, e.g. `../models/building_defect_yolo_nas_m/RUN_<timestamp>/ckpt_best.pth`. Both are specific to your machine and the run you want to use, so check them before every run.
3. Run:
   ```bash
   python batch_inference.py --config config.yaml
   ```

Each run writes a new `output-preds/<timestamp>/` folder containing `labels/` (YOLO rows with a confidence column), `images-overlaid/` (boxes drawn on the photos), `images/` (copies of the inputs) and the `config.yaml` used.

The main settings:

| Setting | Default | Meaning |
|---|---|---|
| `use_tiled_inference` | `true` | Slice each photo with SAHI. Needed for large photos; turning it off shrinks the whole photo to 640 px, which loses small defects. |
| `tiled_inference.slice_*`, `overlap_*` | 2048, 0.5 | Must match the training tiles. |
| `conf_threshold` | 0.3 | Minimum confidence to keep a detection. |
| `merge_annotations` | on | Merges touching or overlapping boxes of the same class into one. |

`batch_inference/README.md` is the original project's own readme and has more detail on the outputs; its example config predates the five-class model.

## Repository layout

```
defect-detection/
├── setup_venv.sh                         # environment bootstrap (+ COCO weights)
├── requirements.txt
├── docs/
│   └── training.md                       # training rationale
├── data/                                 # not tracked
│   ├── data_raw/                         #   raw source datasets
│   ├── data-cleaned/                     #   older 3-class cleaned sets
│   ├── data-5defect/                     #   5-class datasets
│   └── data-tiled/                       #   tiled datasets, the input to training
├── models/                               # not tracked; checkpoints
├── data-cleaning/
│   ├── dataset_eda.ipynb                 # read-only EDA for any dataset
│   └── scripts/
│       ├── clean_to_x_classes.py         # keep chosen classes, drop empty images
│       ├── filter_low_res.py             # drop photos below a minimum resolution
│       ├── tile_building_defect_images.py# cut photos into overlapping tiles
│       ├── restore_selected_classes.py   # legacy: 3-class set + classes restored from raw
│       └── yolo_to_ls_predictions.py     # YOLO labels -> Label Studio import JSON
├── yolonas-training/
│   ├── train.ipynb                       # training notebook
│   ├── copy_paste_aug.py                 # optional rare-class copy-paste augmentation
│   ├── training-data/                    # the train/val/test split (contents not tracked)
│   └── scripts/
│       ├── build_training_data.py        # datasets -> training-data/ split
│       ├── cleanup_checkpoints.py        # delete RUN_* folders not in a keep-list
│       └── clear_splits.sh               # empties training-data/ (redundant; see below)
└── batch_inference/
    ├── batch_inference.py                # entry point
    ├── sahi_inference.py                 # SAHI tiled-inference wrapper
    ├── config.yaml                       # run configuration
    ├── clear_output_preds.py             # empties output-preds/
    ├── input-data/                       # photos to score (not tracked)
    └── output-preds/                     # one timestamped folder per run (not tracked)
```

## Other scripts

- **`restore_selected_classes.py`** is the older route to a five-class dataset: it takes a three-class cleaned set and adds `plant_growth` / `crack_glass` back from raw. It cannot recover images that contained only those two classes, because the three-class cleaning already dropped them. For new datasets use `clean_to_x_classes.py` with all five classes instead. Its dataset list is hardcoded and it reprocesses every dataset on each run.
- **`yolo_to_ls_predictions.py`** turns a dataset's YOLO labels into Label Studio import JSON with pre-drawn boxes, for reviewing or correcting annotations.
- **`cleanup_checkpoints.py`** deletes `RUN_*` folders under `models/building_defect_yolo_nas_m/` that are not listed in `models/checkpoint-map.txt`. Run it with `--dry-run` first.
- **`clear_splits.sh`** empties `training-data/`. `build_training_data.py` now does this itself and keeps the tracked `.gitkeep` files, which this script deletes.
- **`clear_output_preds.py`** empties `batch_inference/output-preds/`.

## Superseded files

Earlier versions of this repo had these; they were removed in favour of the scripts above.

| Removed | Replaced by |
|---|---|
| `data-cleaning/clean_to_3class.ipynb` | `clean_to_x_classes.py`, `filter_low_res.py`, `dataset_eda.ipynb` |
| `data-cleaning/data-cleaning.ipynb` | `clean_to_x_classes.py`, `filter_low_res.py` |
| `yolonas-training/scripts/merge_datasets.py`, `split_dataset.py` | `build_training_data.py` |
| `data.yaml` in each dataset | `classes.txt` |
