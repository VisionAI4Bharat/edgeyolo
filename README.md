# EdgeYOLO Wrapper Repo

A reusable wrapper repository around **EdgeYOLO** where the original upstream repository lives as a git submodule at:

```text
third_party/edgeyolo
```

The wrapper repo does the following in one place:

- inspects the source dataset automatically
- detects a supported annotation family
- cleans the dataset
- keeps only valid image/annotation pairs
- removes empty object-detection labels from the canonical prepared dataset
- converts **all images** into a single JPG image format
- converts the dataset into a **canonical COCO detection dataset**
- creates the EdgeYOLO dataset YAML
- creates the EdgeYOLO train YAML
- creates a custom model YAML with the correct `nc`
- patches the EdgeYOLO loader for newer PyTorch checkpoint-loading behavior
- downloads official pretrained weights when required
- starts training
- optionally exports the final trained checkpoint to ONNX

The goal is that, after the repo is initialized properly, your normal command can stay very short and only the **dataset root** is required.

---

## 1. Repository layout

```text
edgeyolo_wrapper_repo/
├── README.md
├── .gitignore
├── requirements-wrapper.txt
├── generated/
│   └── .gitkeep
├── third_party/
│   └── .gitkeep
├── workspace/
│   └── .gitkeep
└── scripts/
    ├── init_repo_with_submodule.sh
    ├── pipeline.py
    └── export_onnx.py
```

### What each area is for

#### `third_party/edgeyolo/`
This is where the original EdgeYOLO repository must live as a **git submodule**.

#### `workspace/extracted/`
Temporary extraction area if the input dataset is a `.zip` archive.

#### `workspace/prepared/`
Canonical prepared datasets produced by the wrapper.

#### `workspace/runs/`
Training outputs. Each run directory name contains:

```text
epoch_batch_inputresolution_date_time
```

Example:

```text
e300_b16_i416x416_20260408_170059
```

#### `workspace/exports/`
Wrapper-owned ONNX export copies and export manifests.

#### `generated/`
Generated YAML files and run manifests.

---

## 2. How to make this a git repo with EdgeYOLO as a submodule

If you start from the extracted zip of this wrapper repo, run the following from the wrapper repo root:

```bash
git init
git branch -M main
git submodule add https://github.com/LSH9832/edgeyolo.git third_party/edgeyolo
git submodule update --init --recursive
```

Or use the helper script:

```bash
bash scripts/init_repo_with_submodule.sh
```

---

## 3. Python packages needed by the wrapper

Install the wrapper-side packages first:

```bash
python3 -m pip install -r requirements-wrapper.txt
```

Then install EdgeYOLO's own requirements inside the submodule checkout:

```bash
cd third_party/edgeyolo
python3 -m pip install -r requirements.txt
cd ../..
```

---

## 4. Supported source dataset types

This wrapper detects and converts these source annotation families automatically.

### A. YOLO TXT
Expected common pattern:

```text
<dataset>/train/images/*
<dataset>/train/labels/*.txt
<dataset>/valid/images/*
<dataset>/valid/labels/*.txt
```

YOLO label rows are expected in this format:

```text
class_id x_center y_center width height
```

with normalized values in `[0, 1]`.

The wrapper validates that:

- each row has exactly 5 values
- the class id is non-negative
- the bbox values are normalized
- width and height are positive

If class names are not provided manually, the wrapper tries to read them from one of:

- `data.yaml`
- `data.yml`
- `dataset.yaml`
- `dataset.yml`

If it still cannot find names, it falls back to generic names like `class_0`, `class_1`, etc.

---

### B. COCO JSON
Supported common patterns:

#### global annotations directory

```text
<dataset>/train/images/*
<dataset>/valid/images/*
<dataset>/annotations/instances_train.json
<dataset>/annotations/instances_valid.json
```

#### split-local Roboflow-style JSON

```text
<dataset>/train/images/*
<dataset>/train/_annotations.coco.json
<dataset>/valid/images/*
<dataset>/valid/_annotations.coco.json
```

The wrapper:

- remaps category ids to contiguous `0..N-1`
- converts all referenced images to JPG
- drops images that have no detection annotations
- rewrites a clean canonical COCO dataset

---

### C. Pascal VOC XML
Supported common patterns:

```text
<dataset>/train/images/*
<dataset>/train/labels/*.xml
<dataset>/valid/images/*
<dataset>/valid/labels/*.xml
```

or:

```text
<dataset>/train/images/*
<dataset>/train/annotations/*.xml
<dataset>/valid/images/*
<dataset>/valid/annotations/*.xml
```

The wrapper:

- parses VOC XML files
- discovers class names automatically if not supplied
- converts all images to JPG
- writes canonical COCO JSON train/valid files

---

## 5. What the wrapper always produces

No matter which supported source family you start with, the wrapper always writes this canonical prepared dataset:

```text
workspace/prepared/<dataset_name>_prepared/
├── train/
│   └── images/*.jpg
├── valid/
│   └── images/*.jpg
├── annotations/
│   ├── instances_train.json
│   └── instances_valid.json
└── manifest.json
```

### Why this is useful

- mixed image extensions stop being a problem
- evaluation becomes consistent
- the training command does not need special handling per dataset family
- the prepared dataset can be reused for repeated experiments

---

## 6. Generated EdgeYOLO configs

For each training run, the wrapper writes generated files under:

```text
generated/<run_name>/
```

This includes:

- `dataset_<run_name>.yaml`
- `model_<model>_<run_name>.yaml`
- `train_<run_name>.yaml`
- `run_manifest.json`

### Important design choice

The generated YAML files live in the wrapper repo, not inside the EdgeYOLO submodule.

That makes the wrapper easier to version, inspect, and rerun without polluting the upstream repository more than necessary.

---

## 7. Output directory naming

Training output directories are created as:

```text
workspace/runs/e{epochs}_b{batch}_i{input}x{input}_{YYYYMMDD_HHMMSS}
```

Example:

```text
workspace/runs/e300_b16_i416x416_20260408_170059
```

That satisfies the requirement that the output contain:

- epoch
- batch
- input resolution
- date
- time

---

## 8. Normal training command

Once the submodule is present and dependencies are installed, the normal command only needs the dataset root.

### Example: prepare + train

```bash
python3 scripts/pipeline.py \
  --dataset-root /path/to/your/dataset_or_zip \
  --epochs 300 \
  --batch-size 16 \
  --input-size 416 \
  --fp16
```

### Example: prepare only

```bash
python3 scripts/pipeline.py \
  --dataset-root /path/to/your/dataset_or_zip \
  --epochs 300 \
  --batch-size 16 \
  --input-size 416 \
  --prepare-only
```

### Example: manual class names override

```bash
python3 scripts/pipeline.py \
  --dataset-root /path/to/your/dataset_or_zip \
  --class-names forklift person \
  --epochs 300 \
  --batch-size 16 \
  --input-size 416 \
  --fp16
```

### Example: comma-separated class names override

```bash
python3 scripts/pipeline.py \
  --dataset-root /path/to/your/dataset_or_zip \
  --class-names forklift,person \
  --epochs 300 \
  --batch-size 16 \
  --input-size 416 \
  --fp16
```

---

## 9. ONNX export support

This wrapper includes a dedicated ONNX export script:

```text
scripts/export_onnx.py
```

It also supports ONNX export directly from the main pipeline after training completes.

### Export through the main pipeline

```bash
python3 scripts/pipeline.py \
  --dataset-root /path/to/your/dataset_or_zip \
  --epochs 300 \
  --batch-size 16 \
  --input-size 416 \
  --fp16 \
  --export-onnx-after-train \
  --export-opset 13
```

### Export a finished checkpoint manually

```bash
python3 scripts/export_onnx.py \
  --weights /absolute/path/to/best.pth \
  --input-size 416 \
  --batch 1 \
  --opset 13
```

### Disable ONNX simplification

```bash
python3 scripts/export_onnx.py \
  --weights /absolute/path/to/best.pth \
  --input-size 416 \
  --batch 1 \
  --opset 13 \
  --no-simplify
```

### Where exported models are copied

The wrapper copies the newest exported ONNX file into:

```text
workspace/exports/<weights_stem>/
```

It also writes an `export_manifest.json` beside the copied ONNX file.

---

## 10. Why the wrapper defaults to ONNX opset 13

The wrapper defaults to **opset 13** because it is a practical, modern default for many ONNX Runtime workflows.

However, the value is configurable because:

- some toolchains still prefer older opsets
- the original EdgeYOLO export examples use opset 11
- deployment constraints can differ by target runtime

So the wrapper exposes:

```bash
--export-opset 13
```

and you can change it whenever needed.

---

## 11. What happens internally during the pipeline

The main pipeline performs these stages:

### Stage 1. Resolve the source dataset

The input may be:

- a directory
- a `.zip` archive

If it is a zip, the wrapper extracts it under `workspace/extracted/`.

### Stage 2. Detect the annotation family

The wrapper detects one of:

- YOLO TXT
- COCO JSON
- Pascal VOC XML

### Stage 3. Build a canonical dataset

The wrapper then creates a new prepared dataset that:

- keeps only valid detection examples
- converts all images to JPG
- converts the dataset into COCO train/valid JSON

### Stage 4. Patch EdgeYOLO if needed

The wrapper patches the EdgeYOLO checkpoint-loading path for newer PyTorch releases.

### Stage 5. Download pretrained weights if needed

For the selected model preset, the wrapper downloads official pretrained weights to:

```text
third_party/edgeyolo/weights/
```

### Stage 6. Generate YAML files

The wrapper writes model, dataset, and training YAML files under `generated/<run_name>/`.

### Stage 7. Start training

The wrapper runs EdgeYOLO training using the generated train YAML.

### Stage 8. Optionally export ONNX

If requested, the wrapper calls the export wrapper script after training finishes.

---

## 12. Example workflow from scratch

### Step A. Extract this wrapper repo zip

```bash
unzip edgeyolo_wrapper_repo.zip
cd edgeyolo_wrapper_repo
```

### Step B. Initialize git + submodule

```bash
bash scripts/init_repo_with_submodule.sh
```

### Step C. Install Python packages

```bash
python3 -m pip install -r requirements-wrapper.txt
cd third_party/edgeyolo
python3 -m pip install -r requirements.txt
cd ../..
```

### Step D. Run a 300-epoch training job

```bash
python3 scripts/pipeline.py \
  --dataset-root /data/dataforklift.zip \
  --epochs 300 \
  --batch-size 16 \
  --input-size 416 \
  --fp16
```

### Step E. Run training plus ONNX export

```bash
python3 scripts/pipeline.py \
  --dataset-root /data/dataforklift.zip \
  --epochs 300 \
  --batch-size 16 \
  --input-size 416 \
  --fp16 \
  --export-onnx-after-train \
  --export-opset 13
```

---

## 13. Notes and current limits

### Supported input families are common but not literally every possible custom format

The wrapper converts **common detection families** automatically:

- YOLO TXT
- COCO JSON
- Pascal VOC XML

If your source is an arbitrary custom CSV or some special JSON schema, you should add a new importer function rather than trying to force it through the existing checks.

### Empty-label images are dropped in the canonical dataset

This wrapper intentionally removes empty detection-label images from the prepared canonical dataset, because this was the safest working behavior for the EdgeYOLO setup this repo was built around.

If you later want to keep true negative images in the canonical set, modify the relevant preparation function in `scripts/pipeline.py`.

### The wrapper keeps generated YAML files outside the submodule

This is deliberate. It makes the wrapper easier to audit and reuse.

---

## 14. Files you will usually care about most

### Main pipeline

```text
scripts/pipeline.py
```

### Manual ONNX export

```text
scripts/export_onnx.py
```

### Submodule initialization helper

```text
scripts/init_repo_with_submodule.sh
```

### Generated training config for a specific run

```text
generated/<run_name>/train_<run_name>.yaml
```

### Final best checkpoint for a run

```text
workspace/runs/<run_name>/best.pth
```

### Copied ONNX artifact

```text
workspace/exports/<weights_stem>/
```

---

## 15. Class YAML Format

For ONNX and other inference backends, EdgeYOLO requires a YAML file that defines the class names and optionally the input size. This file should be placed next to the model file with the same base name (e.g., `model.onnx` and `model.yaml`) or specified explicitly in the configuration dialog.

### YAML Structure
```yaml
names:
  - person
  - forklift
  - # ... additional classes
```

### Optional Input Size
You can explicitly specify the input size if needed:
```yaml
names:
  - person
  - forklift
img_size: [416, 416]  # [height, width]
```

If `img_size` is not specified, the wrapper will attempt to infer the input size from the ONNX model metadata.

---

## 16. Persistent Settings Storage

The EdgeYOLO Qt GUI automatically saves and loads Region of Interest (ROI) configurations to enable persistence across sessions.

### ROI Storage Location
ROI configurations are stored in YAML files located in the same directory as the model file, using the naming convention:
```
<modelBaseName>_roi.yaml
```
For example, if your model is at `/path/to/yolov5s.onnx`, the ROI configuration will be stored in `/path/to/yolov5s_roi.yaml`.

### ROI YAML Format
```yaml
roi:
  x: 100      # X coordinate of top-left corner
  y: 50       # Y coordinate of top-left corner
  width: 300  # Width in pixels
  height: 200 # Height in pixels
```

The GUI also supports storing class names in the same YAML file under the `names` key, allowing a single configuration file to contain both ROI and class information:
```yaml
roi:
  x: 100
  y: 50
  width: 300
  height: 200
names:
  - person
  - forklift
```

### Automatic Loading/Saving
- When a model is loaded, the GUI automatically looks for `<modelBaseName>_roi.yaml` and loads the ROI if present
- When drawing or modifying an ROI in Edit mode, changes are immediately saved to the same YAML file
- If no ROI YAML exists, the GUI starts with no ROI filtering enabled

---

## 15. Minimal command summary

### Create repo + submodule

```bash
git init
git branch -M main
git submodule add https://github.com/LSH9832/edgeyolo.git third_party/edgeyolo
git submodule update --init --recursive
```

### Train

```bash
python3 scripts/pipeline.py --dataset-root /path/to/dataset_or_zip --epochs 300 --batch-size 16 --input-size 416 --fp16
```

### Train + export ONNX

```bash
python3 scripts/pipeline.py --dataset-root /path/to/dataset_or_zip --epochs 300 --batch-size 16 --input-size 416 --fp16 --export-onnx-after-train --export-opset 13
```

### Export a trained checkpoint manually

```bash
python3 scripts/export_onnx.py --weights /path/to/best.pth --input-size 416 --batch 1 --opset 13
```

