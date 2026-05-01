# EdgeYOLO Wrapper Pipeline

The `pipeline.py` script is an all-in-one automation tool designed to take raw image datasets and convert them into trained, deployment-ready EdgeYOLO models. It handles dataset normalization, environment patching, training orchestration, and ONNX export.

## 🚀 Quick Start

Ensure you have your dataset ready (either as a folder or a `.zip` file) and run:

```bash
python3 pipeline.py --dataset-root path/to/dataset.zip --epochs 100 --export-onnx-after-train
```

## 📋 Key Features

- **Automatic Format Detection**: Supports YOLO (.txt), COCO (.json), and Pascal VOC (.xml) annotations.
- **Data Normalization**: Converts all images to `.jpg` and all annotations to a canonical COCO JSON format.
- **Auto-Patching**: Automatically patches the EdgeYOLO submodule to support modern PyTorch (2.6+) and ensures pretrained weights are downloaded.
- **Stable Configs**: Generates editable YAML files in the `generated/` folder that persist across runs.
- **End-to-End**: Goes from raw data to a simplified ONNX model in a single command.

## 📂 Supported Input Structure

The script expects a `train/` and `valid/` split. Common patterns include:

### YOLO Format
```text
dataset/
├── train/
│   ├── images/*.jpg
│   └── labels/*.txt
└── valid/
    ├── images/*.jpg
    └── labels/*.txt
```

### COCO Format
```text
dataset/
├── annotations/
│   ├── instances_train.json
│   └── instances_valid.json
├── train/*.jpg
└── valid/*.jpg
```

## ⚙️ Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset-root` | **(Required)** Path to dataset folder or .zip file. | N/A |
| `--class-names` | Space or comma-separated list of classes. Auto-detected if omitted. | Auto |
| `--model` | Model preset to use. Currently supports `tiny`. | `tiny` |
| `--input-size` | Image resolution (e.g., 416 or 640). | `416` |
| `--epochs` | Number of training epochs. | `300` |
| `--batch-size` | Images per GPU. | `16` |
| `--device` | CUDA device ID(s) (e.g., `0` or `0,1`). | `0` |
| `--fp16` | Enable mixed-precision training. | `False` |
| `--prepare-only` | Stop after data normalization and YAML generation. | `False` |
| `--export-onnx-after-train` | Automatically export to ONNX after training finishes. | `False` |

## 🛠 Advanced Usage

### Just Prepare the Data
If you want to inspect the normalized dataset and generated configs before training:
```bash
python3 pipeline.py --dataset-root ./my_data --prepare-only
```

### Multigpu Training with custom resolution
```bash
python3 pipeline.py --dataset-root ./data.zip \
    --device 0,1 \
    --batch-size 32 \
    --input-size 640 \
    --export-onnx-after-train
```

## 📁 Output Directory Structure

- **`workspace/prepared/`**: The "canonical" version of your dataset (All JPGs + COCO JSON).
- **`workspace/runs/`**: Training logs, checkpoints (`best.pth`, `last.pth`), and per-run configs.
- **`generated/`**: Stable `model.yaml`, `dataset.yaml`, and `train.yaml`. These can be manually edited.
- **`weights/`**: Cached pretrained EdgeYOLO weights downloaded from upstream.

---

### Prerequisites
- Python 3.8+
- `pip install opencv-python pyyaml torch torchvision`
- EdgeYOLO submodule initialized in `third_party/edgeyolo`.
