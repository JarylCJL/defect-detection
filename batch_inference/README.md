# Batch Inference

This project runs batch inference on a folder of images using a YOLO‑NAS checkpoint. It can run either standard inference or tiled inference (SAHI) and writes predictions to a timestamped output folder.

## Quick start

```bash
# From your local clone of this repo
python3 -m pip install super-gradients pyyaml tqdm pillow

# If you want tiled inference
python3 -m pip install sahi opencv-python

python3 batch_inference.py --config config.yaml
```

## What you get

Each run writes outputs to a new timestamped directory under `output-preds/`:

```
output-preds/
  YYYYMMDD_HHMMSS/
    config.yaml
    images/
    images-overlaid/
    labels/
```

- `images/`: raw input images copied from the input directory
- `images-overlaid/`: images with bounding boxes and class labels drawn
- `labels/`: YOLO‑format predictions (`class cx cy w h [conf]`), normalized to image size
- `config.yaml`: exact settings used for that run

## Configuration

Edit `config.yaml` before running (paths are repo‑relative by default):

```yaml
# Batch inference configuration
input_data: input-data/PlainBackground(25photos)/spalling
output_dir: output-preds

# Model settings
model_type: yolo-nas-m
# models/ is the shared top-level directory (../models from this project),
# populated by yolonas-training. SuperGradients writes checkpoints to
# {models}/{EXPERIMENT_NAME}/RUN_*/... where EXPERIMENT_NAME="building_defect_yolo_nas_m".
checkpoint_path: ../models/building_defect_yolo_nas_m/RUN_20260123_101514_611070/ckpt_best.pth
num_classes: 3
class_names:
  - corrosion
  - crack
  - spall

# Inference settings
conf_threshold: 0.4
nms_iou_threshold: 0.5

# Tiled inference (SAHI)
use_tiled_inference: true
tiled_inference:
  slice_height: 2048
  slice_width: 2048
  overlap_height_ratio: 0.2
  overlap_width_ratio: 0.2
  device: null
```

### Key fields

- `input_data`: directory containing images to run inference on (can include subfolders). This project keeps its own local `input-data/` folder, independent of the shared top-level `data/`.
- `output_dir`: base output directory (each run creates a timestamped subfolder).
- `model_type`: model name (e.g., `yolo-nas-m`).
- `checkpoint_path`: path to the `.pth` checkpoint — now resolved against the shared top-level `../models/` directory (see note above).
- `num_classes`: number of classes for the model.
- `class_names`: list of class names (used for overlay labels and color mapping).
- `conf_threshold`, `nms_iou_threshold`: inference thresholds.
- `use_tiled_inference`: set to `true` to enable SAHI tiling.
- `tiled_inference.*`: tiling parameters (slice size + overlap). `device: null` lets SAHI pick GPU if available.

## Running inference

```bash
python3 batch_inference.py --config config.yaml
```

Notes:
- The script uses a progress bar for overall batch progress.
- SuperGradients INFO logs are silenced (set to WARNING) inside the script.

## Tiled inference (SAHI)

Tiled inference is enabled with `use_tiled_inference: true` and requires `sahi` and `opencv-python`.

SAHI integration uses a custom `YoloNASDetectionModel` wrapper to run YOLO‑NAS on image tiles and merge results.

If you see:
```
SAHI is required for tiled inference. Install with: pip install sahi
```
Install the dependencies or set `use_tiled_inference: false`.

## Clearing outputs

To clear all previous runs but keep the `output-preds/` folder:

```bash
python3 clear_output_preds.py
```

## File overview

- `batch_inference.py`: main entry point
- `sahi_inference.py`: SAHI tiling wrapper for YOLO‑NAS
- `config.yaml`: runtime configuration
- `clear_output_preds.py`: utility to clear `output-preds/`

## Troubleshooting

- **Checkpoint not found**: verify `checkpoint_path` in `config.yaml` and that the checkpoint exists on your machine.
- **No images found**: verify `input_data` and supported image extensions (jpg, jpeg, png, bmp, tiff).
- **SAHI errors**: ensure `sahi` + `opencv-python` are installed and `use_tiled_inference` is set correctly.
- **GPU vs CPU**: SAHI uses GPU if available; set `tiled_inference.device` to `"cuda"` or `"cpu"` to force a device.
