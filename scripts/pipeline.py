#!/usr/bin/env python3
"""
Reusable EdgeYOLO wrapper pipeline.

Purpose
=======
This script turns a raw detection dataset into a canonical dataset that EdgeYOLO
can consume reliably, generates all required YAML files, optionally starts
training, and can optionally export the final trained model to ONNX.

Design goals
============
1. A single wrapper repository around EdgeYOLO.
2. EdgeYOLO lives at: third_party/edgeyolo (git submodule).
3. The user should normally need to provide only --dataset-root.
4. Mixed image extensions are normalized into a single JPG-only dataset.
5. Common annotation formats are detected automatically and converted into a
   canonical COCO dataset used for both training and validation.
6. All generated files are written outside the EdgeYOLO submodule where possible,
   keeping the wrapper repo organized and easier to reuse.
7. The pipeline is explicit, heavily commented, and safe to inspect/modify.

Supported source annotation families
====================================
- YOLO TXT detection labels
- COCO JSON detection labels
- Pascal VOC XML detection labels

Not supported
=============
- Arbitrary custom CSV/JSON/XML schemas that do not match common YOLO/COCO/VOC layouts
- Segmentation masks / polygons for segmentation training

Canonical output dataset
========================
The pipeline always prepares this structure:

    workspace/prepared/<dataset_name>/
        train/images/*.jpg
        valid/images/*.jpg
        annotations/instances_train.json
        annotations/instances_valid.json
        manifest.json

Why COCO as the canonical form?
===============================
EdgeYOLO can train on both YOLO and COCO-style dataset configs. In practice,
using a clean COCO JSON for train/valid tends to make evaluation behavior much
more predictable because the validation path is COCO-based.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import textwrap
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "OpenCV is required. Install it with: pip install opencv-python"
    ) from exc

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install it with: pip install pyyaml"
    ) from exc

# Accepted source image formats. All will be converted into final JPG images.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Official upstream tiny COCO checkpoint URL.
#
# Upstream EdgeYOLO currently publishes a single Tiny COCO checkpoint filename
# (`edgeyolo_tiny_coco.pth`) while reporting both 416 and 640 Tiny metrics in
# the README. To make the wrapper explicit and less ambiguous in local usage,
# this wrapper stores two local aliases:
#   - edgeyolo_tiny_416_coco.pth
#   - edgeyolo_tiny_640_coco.pth
# Both aliases currently download from the same upstream URL unless upstream
# later publishes resolution-specific weight files.
UPSTREAM_TINY_COCO_URL = (
    "https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_tiny_coco.pth"
)

MODEL_PRESETS = {
    "tiny": {
        "base_yaml": "params/model/edgeyolo_tiny.yaml",
        "weights_aliases": {
            416: {
                "url": UPSTREAM_TINY_COCO_URL,
                "name": "edgeyolo_tiny_416_coco.pth",
            },
            640: {
                "url": UPSTREAM_TINY_COCO_URL,
                "name": "edgeyolo_tiny_640_coco.pth",
            },
        },
    },
    # Add more presets later if you decide to use edgeyolo_s, etc.
}

# -----------------------------------------------------------------------------
# Logging and basic helpers
# -----------------------------------------------------------------------------

def log(msg: str) -> None:
    """Simple logger that flushes immediately, useful over SSH/tmux."""
    print(msg, flush=True)


def abort(msg: str, code: int = 1) -> None:
    raise SystemExit(f"ERROR: {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    """Run a subprocess and fail loudly if it returns non-zero."""
    log("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        abort(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def repo_root_from_script() -> Path:
    """The wrapper repo root is the parent of the scripts/ directory."""
    return Path(__file__).resolve().parents[1]


def parse_class_names(raw: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Accept either comma-separated or space-separated class names."""
    if not raw:
        return None
    if len(raw) == 1 and "," in raw[0]:
        vals = [x.strip() for x in raw[0].split(",") if x.strip()]
        return vals or None
    vals = [x.strip() for x in raw if x.strip()]
    return vals or None


def detect_train_valid_dirs(root: Path) -> Tuple[Path, Path]:
    """
    Accept either a direct dataset root or a wrapper root containing one child
    directory that is the true dataset root.
    """
    if (root / "train").exists() and (root / "valid").exists():
        return root / "train", root / "valid"

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and (subdirs[0] / "train").exists() and (subdirs[0] / "valid").exists():
        return subdirs[0] / "train", subdirs[0] / "valid"

    abort(
        "Could not detect train/ and valid/ directories under dataset root. "
        f"Checked: {root}"
    )


# -----------------------------------------------------------------------------
# Input extraction and format detection
# -----------------------------------------------------------------------------

def extract_input_if_needed(dataset_root: Path, working_root: Path) -> Path:
    """
    Accept either:
      - a directory path
      - a .zip archive

    If zip: extract it under workspace/extracted/<zip_stem>/ and return the path.
    If directory: return the directory.
    """
    if dataset_root.is_dir():
        return dataset_root.resolve()

    if not dataset_root.exists():
        abort(f"Dataset root does not exist: {dataset_root}")

    if dataset_root.suffix.lower() != ".zip":
        abort("Only a directory or a .zip archive is supported as --dataset-root")

    out_dir = working_root / "extracted" / dataset_root.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    log(f"[extract] Extracting zip archive: {dataset_root}")
    with zipfile.ZipFile(dataset_root, "r") as zf:
        zf.extractall(out_dir)

    return out_dir.resolve()


@dataclass
class DetectedLayout:
    """
    Describes the detected source dataset format and where its pieces live.

    format_name:
      'yolo', 'coco', or 'voc'

    train_root / valid_root:
      paths to the source split directories

    extra:
      format-specific metadata, such as JSON paths.
    """

    format_name: str
    source_root: str
    train_root: str
    valid_root: str
    extra: Dict[str, str]


def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def detect_dataset_layout(dataset_root: Path) -> DetectedLayout:
    """
    Detect one of the supported dataset layouts.

    Common supported patterns:
    1) YOLO
       root/train/images/*
       root/train/labels/*.txt
       root/valid/images/*
       root/valid/labels/*.txt

    2) COCO
       root/train/images/*
       root/valid/images/*
       root/annotations/instances_train.json
       root/annotations/instances_valid.json

       or Roboflow-style split-local JSON:
       root/train/images/*
       root/train/_annotations.coco.json
       root/valid/images/*
       root/valid/_annotations.coco.json

    3) VOC
       root/train/images/*
       root/train/labels/*.xml   (or annotations/*.xml)
       root/valid/images/*
       root/valid/labels/*.xml   (or annotations/*.xml)
    """
    train_dir, valid_dir = detect_train_valid_dirs(dataset_root)

    train_images = find_first_existing([train_dir / "images", train_dir])
    valid_images = find_first_existing([valid_dir / "images", valid_dir])
    if not train_images or not valid_images:
        abort("Could not find train/valid image directories")

    # COCO detection first, because some COCO datasets may also have a labels/ dir.
    global_train_json = dataset_root / "annotations" / "instances_train.json"
    global_valid_json = dataset_root / "annotations" / "instances_valid.json"
    split_train_json = train_dir / "_annotations.coco.json"
    split_valid_json = valid_dir / "_annotations.coco.json"

    if global_train_json.exists() and global_valid_json.exists():
        return DetectedLayout(
            format_name="coco",
            source_root=str(dataset_root),
            train_root=str(train_dir),
            valid_root=str(valid_dir),
            extra={
                "train_json": str(global_train_json),
                "valid_json": str(global_valid_json),
                "train_images": str(train_images),
                "valid_images": str(valid_images),
            },
        )

    if split_train_json.exists() and split_valid_json.exists():
        return DetectedLayout(
            format_name="coco",
            source_root=str(dataset_root),
            train_root=str(train_dir),
            valid_root=str(valid_dir),
            extra={
                "train_json": str(split_train_json),
                "valid_json": str(split_valid_json),
                "train_images": str(train_images),
                "valid_images": str(valid_images),
            },
        )

    # YOLO
    train_yolo_labels = train_dir / "labels"
    valid_yolo_labels = valid_dir / "labels"
    if train_yolo_labels.exists() and valid_yolo_labels.exists():
        if any(p.suffix.lower() == ".txt" for p in train_yolo_labels.iterdir()) and any(
            p.suffix.lower() == ".txt" for p in valid_yolo_labels.iterdir()
        ):
            return DetectedLayout(
                format_name="yolo",
                source_root=str(dataset_root),
                train_root=str(train_dir),
                valid_root=str(valid_dir),
                extra={
                    "train_images": str(train_images),
                    "valid_images": str(valid_images),
                    "train_labels": str(train_yolo_labels),
                    "valid_labels": str(valid_yolo_labels),
                },
            )

    # VOC
    train_voc_labels = find_first_existing([train_dir / "labels", train_dir / "annotations", train_dir / "Annotations"])
    valid_voc_labels = find_first_existing([valid_dir / "labels", valid_dir / "annotations", valid_dir / "Annotations"])
    if train_voc_labels and valid_voc_labels:
        if any(p.suffix.lower() == ".xml" for p in train_voc_labels.iterdir()) and any(
            p.suffix.lower() == ".xml" for p in valid_voc_labels.iterdir()
        ):
            return DetectedLayout(
                format_name="voc",
                source_root=str(dataset_root),
                train_root=str(train_dir),
                valid_root=str(valid_dir),
                extra={
                    "train_images": str(train_images),
                    "valid_images": str(valid_images),
                    "train_labels": str(train_voc_labels),
                    "valid_labels": str(valid_voc_labels),
                },
            )

    abort(
        "Could not detect a supported dataset format under the supplied dataset root. "
        "Supported families: YOLO TXT, COCO JSON, Pascal VOC XML."
    )


# -----------------------------------------------------------------------------
# Class-name discovery
# -----------------------------------------------------------------------------

def discover_class_names_from_yaml_files(dataset_root: Path) -> Optional[List[str]]:
    """
    Try to discover class names from data.yaml / dataset.yaml / data.yml files.
    This helps YOLO datasets so the user does not always need --class-names.
    """
    candidates = []
    for name in ["data.yaml", "data.yml", "dataset.yaml", "dataset.yml"]:
        candidates.extend(dataset_root.rglob(name))

    for cand in candidates:
        try:
            data = yaml.safe_load(cand.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            names = data.get("names")
            if isinstance(names, list) and names:
                return [str(x) for x in names]
            if isinstance(names, dict) and names:
                # Common YOLO variant where names is {0: "cls0", 1: "cls1"}
                ordered = [str(v) for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
                return ordered
        except Exception:
            continue
    return None


def discover_class_names_from_coco_json(coco_json: Path) -> Optional[List[str]]:
    try:
        data = json.loads(coco_json.read_text(encoding="utf-8"))
        cats = data.get("categories", [])
        if not cats:
            return None
        ordered = [x["name"] for x in sorted(cats, key=lambda c: int(c["id"]))]
        return [str(x) for x in ordered]
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Validation and conversion helpers
# -----------------------------------------------------------------------------

@dataclass
class CanonicalDatasetSummary:
    dataset_name: str
    format_detected: str
    class_names: List[str]
    train_images: int
    valid_images: int
    train_annotations: int
    valid_annotations: int
    source_root: str
    prepared_root: str


def safe_jpg_name(stem: str) -> str:
    """Return canonical image name for the prepared dataset."""
    return f"{stem}.jpg"


def read_image_shape(image_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        abort(f"OpenCV failed to read image: {image_path}")
    h, w = img.shape[:2]
    return h, w


def convert_image_to_jpg(src: Path, dst: Path, jpeg_quality: int = 95) -> Tuple[int, int]:
    """
    Convert/copy any supported image into a canonical JPG image.

    Why always convert, even for source JPG/JPEG?
    - It gives a uniform output extension.
    - It avoids mixed suffix problems in downstream loaders.
    - It avoids corner cases where an image extension and content disagree.
    """
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        abort(f"Could not decode image: {src}")
    ensure_dir(dst.parent)
    ok = cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        abort(f"Failed to write JPG image: {dst}")
    h, w = img.shape[:2]
    return h, w


def remap_categories_to_zero_based(categories: List[Dict]) -> Tuple[List[Dict], Dict[int, int], List[str]]:
    """
    Canonicalize category ids to contiguous 0..N-1 and return:
      - new categories list
      - old_id -> new_id map
      - ordered class names
    """
    ordered_src = sorted(categories, key=lambda x: int(x["id"]))
    old_to_new: Dict[int, int] = {}
    new_cats: List[Dict] = []
    names: List[str] = []
    for new_id, cat in enumerate(ordered_src):
        old_id = int(cat["id"])
        name = str(cat["name"])
        old_to_new[old_id] = new_id
        new_cats.append({"id": new_id, "name": name})
        names.append(name)
    return new_cats, old_to_new, names


def load_yolo_label_lines(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Validate and load a YOLO TXT label file.

    Empty labels are allowed in source datasets, but the canonical prepared
    dataset drops them because they caused skipping/confusion in this EdgeYOLO
    workflow.
    """
    raw = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return []

    rows: List[Tuple[int, float, float, float, float]] = []
    for i, line in enumerate(raw.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            abort(f"Invalid YOLO label row in {label_path} line {i}: expected 5 values")
        try:
            cls = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except ValueError:
            abort(f"Invalid numeric YOLO label row in {label_path} line {i}: {line}")

        if cls < 0:
            abort(f"Negative class id in {label_path} line {i}: {line}")
        vals = [xc, yc, bw, bh]
        if any(v < 0 or v > 1 for v in vals):
            abort(f"YOLO bbox values must be normalized to [0,1] in {label_path} line {i}: {line}")
        if bw <= 0 or bh <= 0:
            log(f"[warn] Skipping zero-area YOLO box in {label_path} line {i}: {line}")
            continue
        rows.append((cls, xc, yc, bw, bh))
    return rows


def clamp_box_xywh(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Clamp a COCO xywh box to image bounds."""
    x = max(0.0, min(x, float(img_w - 1)))
    y = max(0.0, min(y, float(img_h - 1)))
    w = max(0.0, min(w, float(img_w) - x))
    h = max(0.0, min(h, float(img_h) - y))
    return x, y, w, h


def build_coco_record(images: List[Dict], annotations: List[Dict], class_names: List[str]) -> Dict:
    categories = [{"id": idx, "name": name} for idx, name in enumerate(class_names)]
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# -----------------------------------------------------------------------------
# Format-specific preparation into the canonical COCO+JPG dataset
# -----------------------------------------------------------------------------

def prepare_from_yolo(
    layout: DetectedLayout,
    prepared_root: Path,
    class_names: Optional[List[str]],
) -> CanonicalDatasetSummary:
    """Prepare canonical dataset from a YOLO TXT source dataset."""
    source_root = Path(layout.source_root)
    train_img_dir = Path(layout.extra["train_images"])
    valid_img_dir = Path(layout.extra["valid_images"])
    train_lbl_dir = Path(layout.extra["train_labels"])
    valid_lbl_dir = Path(layout.extra["valid_labels"])

    if not class_names:
        class_names = discover_class_names_from_yaml_files(source_root)

    # If names are still missing, infer a generic class list from maximum class id.
    if not class_names:
        max_cls = -1
        for lbl_dir in [train_lbl_dir, valid_lbl_dir]:
            for p in lbl_dir.iterdir():
                if p.is_file() and p.suffix.lower() == ".txt":
                    for row in load_yolo_label_lines(p):
                        max_cls = max(max_cls, row[0])
        if max_cls < 0:
            abort("Could not infer class names from YOLO dataset because all labels were empty")
        class_names = [f"class_{i}" for i in range(max_cls + 1)]
        log(
            "[warn] No class names were found in data.yaml/dataset.yaml. "
            f"Using generic names: {class_names}"
        )

    for split in ["train", "valid"]:
        ensure_dir(prepared_root / split / "images")
    ensure_dir(prepared_root / "annotations")

    train_images: List[Dict] = []
    train_annotations: List[Dict] = []
    valid_images: List[Dict] = []
    valid_annotations: List[Dict] = []

    ann_id = 1

    def process_split(split_name: str, src_img_dir: Path, src_lbl_dir: Path, out_images: List[Dict], out_annotations: List[Dict]) -> None:
        nonlocal ann_id
        image_id = 1

        # Build maps by stem so we can keep only matched image/label pairs.
        src_images: Dict[str, Path] = {}
        for p in sorted(src_img_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                if p.stem in src_images:
                    abort(
                        f"Duplicate image stem detected in {src_img_dir}: {p.stem}. "
                        "This would collide after JPG normalization."
                    )
                src_images[p.stem] = p

        src_labels: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
        for p in sorted(src_lbl_dir.iterdir()):
            if p.is_file() and p.suffix.lower() == ".txt":
                rows = load_yolo_label_lines(p)
                # Keep only non-empty labels in the prepared set.
                if rows:
                    src_labels[p.stem] = rows

        matched_stems = sorted(set(src_images) & set(src_labels))

        for stem in matched_stems:
            src_img = src_images[stem]
            rows = src_labels[stem]
            dst_img = prepared_root / split_name / "images" / safe_jpg_name(stem)
            img_h, img_w = convert_image_to_jpg(src_img, dst_img)

            out_images.append({
                "id": image_id,
                "file_name": dst_img.name,
                "width": img_w,
                "height": img_h,
            })

            for cls, xc, yc, bw, bh in rows:
                if cls >= len(class_names):
                    abort(
                        f"YOLO class id {cls} in {src_lbl_dir / (stem + '.txt')} exceeds class name list length {len(class_names)}"
                    )
                x = (xc - bw / 2.0) * img_w
                y = (yc - bh / 2.0) * img_h
                box_w = bw * img_w
                box_h = bh * img_h
                x, y, box_w, box_h = clamp_box_xywh(x, y, box_w, box_h, img_w, img_h)
                out_annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": [x, y, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1

            image_id += 1

        log(
            f"[prepare:yolo] {split_name}: matched non-empty pairs={len(matched_stems)} "
            f"from images={len(src_images)} labels={len(src_labels)}"
        )

    process_split("train", train_img_dir, train_lbl_dir, train_images, train_annotations)
    process_split("valid", valid_img_dir, valid_lbl_dir, valid_images, valid_annotations)

    (prepared_root / "annotations" / "instances_train.json").write_text(
        json.dumps(build_coco_record(train_images, train_annotations, class_names), indent=2),
        encoding="utf-8",
    )
    (prepared_root / "annotations" / "instances_valid.json").write_text(
        json.dumps(build_coco_record(valid_images, valid_annotations, class_names), indent=2),
        encoding="utf-8",
    )

    return CanonicalDatasetSummary(
        dataset_name=prepared_root.name,
        format_detected="yolo",
        class_names=class_names,
        train_images=len(train_images),
        valid_images=len(valid_images),
        train_annotations=len(train_annotations),
        valid_annotations=len(valid_annotations),
        source_root=str(source_root),
        prepared_root=str(prepared_root),
    )


def prepare_from_coco(layout: DetectedLayout, prepared_root: Path) -> CanonicalDatasetSummary:
    """Prepare canonical dataset from a COCO JSON source dataset."""
    train_json = Path(layout.extra["train_json"])
    valid_json = Path(layout.extra["valid_json"])
    train_img_dir = Path(layout.extra["train_images"])
    valid_img_dir = Path(layout.extra["valid_images"])

    for split in ["train", "valid"]:
        ensure_dir(prepared_root / split / "images")
    ensure_dir(prepared_root / "annotations")

    # We use train.json categories as the reference category order. Then map valid
    # categories through names so both train/valid use the exact same zero-based ids.
    train_data = json.loads(train_json.read_text(encoding="utf-8"))
    train_categories = train_data.get("categories", [])
    if not train_categories:
        abort(f"No COCO categories found in: {train_json}")
    canonical_categories, train_old_to_new, class_names = remap_categories_to_zero_based(train_categories)
    name_to_new_id = {cat["name"]: int(cat["id"]) for cat in canonical_categories}

    def process_split(split_name: str, image_dir: Path, json_path: Path, is_train: bool) -> Tuple[List[Dict], List[Dict]]:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        categories = data.get("categories", [])
        images = data.get("images", [])
        annotations = data.get("annotations", [])

        if not images:
            abort(f"No COCO images found in: {json_path}")

        # Build source category mapping.
        if is_train:
            old_to_new = train_old_to_new
        else:
            old_to_name = {int(cat["id"]): str(cat["name"]) for cat in categories}
            old_to_new = {}
            for old_id, name in old_to_name.items():
                if name not in name_to_new_id:
                    abort(
                        f"Validation category '{name}' exists in {json_path} but not in training categories"
                    )
                old_to_new[old_id] = name_to_new_id[name]

        anns_by_image: Dict[int, List[Dict]] = {}
        for ann in annotations:
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        out_images: List[Dict] = []
        out_annotations: List[Dict] = []
        new_image_id = 1
        new_ann_id = 1

        for img in images:
            file_name = str(img["file_name"])
            src_img_path = image_dir / file_name
            if not src_img_path.exists():
                # Some datasets store only the basename in JSON even if JSON came from a subfolder.
                src_img_path = image_dir / Path(file_name).name
            if not src_img_path.exists():
                abort(f"COCO image referenced by JSON not found on disk: {file_name}")

            src_anns = anns_by_image.get(int(img["id"]), [])
            # Drop empty-annotation images to match the canonical behavior used in the rest of the workflow.
            if not src_anns:
                continue

            stem = Path(file_name).stem
            dst_img = prepared_root / split_name / "images" / safe_jpg_name(stem)
            img_h, img_w = convert_image_to_jpg(src_img_path, dst_img)

            out_images.append({
                "id": new_image_id,
                "file_name": dst_img.name,
                "width": img_w,
                "height": img_h,
            })

            for ann in src_anns:
                old_cat_id = int(ann["category_id"])
                if old_cat_id not in old_to_new:
                    abort(f"Unknown COCO category id {old_cat_id} in {json_path}")
                x, y, w, h = [float(v) for v in ann["bbox"]]
                x, y, w, h = clamp_box_xywh(x, y, w, h, img_w, img_h)
                if w <= 0 or h <= 0:
                    continue
                out_annotations.append({
                    "id": new_ann_id,
                    "image_id": new_image_id,
                    "category_id": old_to_new[old_cat_id],
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    "segmentation": ann.get("segmentation", []),
                })
                new_ann_id += 1

            new_image_id += 1

        return out_images, out_annotations

    train_images, train_annotations = process_split("train", train_img_dir, train_json, True)
    valid_images, valid_annotations = process_split("valid", valid_img_dir, valid_json, False)

    (prepared_root / "annotations" / "instances_train.json").write_text(
        json.dumps({"images": train_images, "annotations": train_annotations, "categories": canonical_categories}, indent=2),
        encoding="utf-8",
    )
    (prepared_root / "annotations" / "instances_valid.json").write_text(
        json.dumps({"images": valid_images, "annotations": valid_annotations, "categories": canonical_categories}, indent=2),
        encoding="utf-8",
    )

    log(
        f"[prepare:coco] train: kept images={len(train_images)} annotations={len(train_annotations)}; "
        f"valid: kept images={len(valid_images)} annotations={len(valid_annotations)}"
    )

    return CanonicalDatasetSummary(
        dataset_name=prepared_root.name,
        format_detected="coco",
        class_names=class_names,
        train_images=len(train_images),
        valid_images=len(valid_images),
        train_annotations=len(train_annotations),
        valid_annotations=len(valid_annotations),
        source_root=str(layout.source_root),
        prepared_root=str(prepared_root),
    )


def parse_voc_xml(xml_path: Path) -> Tuple[str, int, int, List[Tuple[str, float, float, float, float]]]:
    """
    Parse a Pascal VOC annotation file.

    Returns:
      image_filename, width, height, list of (class_name, xmin, ymin, xmax, ymax)
    """
    root = ET.parse(xml_path).getroot()

    filename = root.findtext("filename") or ""
    width = int(root.findtext("size/width") or 0)
    height = int(root.findtext("size/height") or 0)
    if width <= 0 or height <= 0:
        abort(f"Invalid or missing VOC image size in: {xml_path}")

    boxes: List[Tuple[str, float, float, float, float]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if not name:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin") or 0)
        ymin = float(bnd.findtext("ymin") or 0)
        xmax = float(bnd.findtext("xmax") or 0)
        ymax = float(bnd.findtext("ymax") or 0)
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append((name, xmin, ymin, xmax, ymax))

    return filename, width, height, boxes


def prepare_from_voc(
    layout: DetectedLayout,
    prepared_root: Path,
    class_names_override: Optional[List[str]],
) -> CanonicalDatasetSummary:
    """Prepare canonical dataset from a Pascal VOC XML source dataset."""
    train_img_dir = Path(layout.extra["train_images"])
    valid_img_dir = Path(layout.extra["valid_images"])
    train_lbl_dir = Path(layout.extra["train_labels"])
    valid_lbl_dir = Path(layout.extra["valid_labels"])

    for split in ["train", "valid"]:
        ensure_dir(prepared_root / split / "images")
    ensure_dir(prepared_root / "annotations")

    # Discover class names from XML files if the user did not provide them.
    discovered_names: List[str] = []
    if not class_names_override:
        names = set()
        for lbl_dir in [train_lbl_dir, valid_lbl_dir]:
            for p in lbl_dir.iterdir():
                if p.is_file() and p.suffix.lower() == ".xml":
                    _, _, _, boxes = parse_voc_xml(p)
                    for name, *_ in boxes:
                        names.add(name)
        discovered_names = sorted(names)
        if not discovered_names:
            abort("Could not discover any VOC class names from XML annotations")
        class_names = discovered_names
    else:
        class_names = class_names_override

    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    def process_split(split_name: str, img_dir: Path, lbl_dir: Path) -> Tuple[List[Dict], List[Dict]]:
        images: List[Dict] = []
        annotations: List[Dict] = []
        image_id = 1
        ann_id = 1

        # Build a source image lookup by stem and by exact name.
        by_stem: Dict[str, Path] = {}
        by_name: Dict[str, Path] = {}
        for p in img_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                by_stem[p.stem] = p
                by_name[p.name] = p

        for xml_path in sorted(lbl_dir.iterdir()):
            if not xml_path.is_file() or xml_path.suffix.lower() != ".xml":
                continue
            filename, _, _, boxes = parse_voc_xml(xml_path)
            if not boxes:
                continue
            src_img = by_name.get(filename) or by_stem.get(Path(filename).stem) or by_stem.get(xml_path.stem)
            if src_img is None:
                abort(f"VOC XML {xml_path} references missing image: {filename}")

            dst_img = prepared_root / split_name / "images" / safe_jpg_name(src_img.stem)
            img_h, img_w = convert_image_to_jpg(src_img, dst_img)

            images.append({
                "id": image_id,
                "file_name": dst_img.name,
                "width": img_w,
                "height": img_h,
            })

            for name, xmin, ymin, xmax, ymax in boxes:
                if name not in class_to_id:
                    abort(f"VOC class '{name}' not present in class name mapping")
                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin
                x, y, w, h = clamp_box_xywh(x, y, w, h, img_w, img_h)
                if w <= 0 or h <= 0:
                    continue
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_to_id[name],
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1

            image_id += 1

        return images, annotations

    train_images, train_annotations = process_split("train", train_img_dir, train_lbl_dir)
    valid_images, valid_annotations = process_split("valid", valid_img_dir, valid_lbl_dir)

    (prepared_root / "annotations" / "instances_train.json").write_text(
        json.dumps(build_coco_record(train_images, train_annotations, class_names), indent=2),
        encoding="utf-8",
    )
    (prepared_root / "annotations" / "instances_valid.json").write_text(
        json.dumps(build_coco_record(valid_images, valid_annotations, class_names), indent=2),
        encoding="utf-8",
    )

    log(
        f"[prepare:voc] train: kept images={len(train_images)} annotations={len(train_annotations)}; "
        f"valid: kept images={len(valid_images)} annotations={len(valid_annotations)}"
    )

    return CanonicalDatasetSummary(
        dataset_name=prepared_root.name,
        format_detected="voc",
        class_names=class_names,
        train_images=len(train_images),
        valid_images=len(valid_images),
        train_annotations=len(train_annotations),
        valid_annotations=len(valid_annotations),
        source_root=str(layout.source_root),
        prepared_root=str(prepared_root),
    )


def prepare_canonical_dataset(
    layout: DetectedLayout,
    prepared_root: Path,
    class_names_override: Optional[List[str]],
) -> CanonicalDatasetSummary:
    """Dispatch to the correct source-format preparation function."""
    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    ensure_dir(prepared_root)

    if layout.format_name == "yolo":
        summary = prepare_from_yolo(layout, prepared_root, class_names_override)
    elif layout.format_name == "coco":
        summary = prepare_from_coco(layout, prepared_root)
    elif layout.format_name == "voc":
        summary = prepare_from_voc(layout, prepared_root, class_names_override)
    else:  # pragma: no cover
        abort(f"Unsupported internal format dispatcher target: {layout.format_name}")

    manifest_path = prepared_root / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    log(f"[prepare] Wrote canonical dataset manifest: {manifest_path}")
    return summary


# -----------------------------------------------------------------------------
# EdgeYOLO integration
# -----------------------------------------------------------------------------

def edgeyolo_root_from_repo(repo_root: Path) -> Path:
    return repo_root / "third_party" / "edgeyolo"


def verify_edgeyolo_checkout(edgeyolo_root: Path) -> None:
    required = [
        edgeyolo_root / "train.py",
        edgeyolo_root / "export.py",
        edgeyolo_root / "params" / "model",
        edgeyolo_root / "edgeyolo" / "models" / "__init__.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        abort(
            "EdgeYOLO submodule checkout looks incomplete. Missing: " + ", ".join(missing)
        )


def patch_edgeyolo_for_torch26(edgeyolo_root: Path) -> None:
    """
    Patch the EdgeYOLO checkpoint loader so official .pth weights can be loaded
    under newer PyTorch versions where torch.load defaults changed.
    """
    target = edgeyolo_root / "edgeyolo" / "models" / "__init__.py"
    text = target.read_text(encoding="utf-8")

    backup = target.with_suffix(".py.bak_wrapper")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    changed = False
    if "add_safe_globals" not in text:
        text = text.replace(
            "import torch\n",
            "import torch\nimport numpy.core.multiarray\nfrom torch.serialization import add_safe_globals\n",
            1,
        )
        changed = True

    old = 'self.ckpt = torch.load(weights, map_location="cpu")'
    new = (
        'add_safe_globals({"scalar": numpy.core.multiarray.scalar})\n'
        '            self.ckpt = torch.load(weights, map_location="cpu", weights_only=False)'
    )
    if old in text and new not in text:
        text = text.replace(old, new, 1)
        changed = True

    if changed:
        target.write_text(text, encoding="utf-8")
        log(f"[patch] Patched EdgeYOLO torch.load compatibility: {target}")
    else:
        log("[patch] EdgeYOLO torch.load compatibility patch already present")



def ensure_pretrained_weights(edgeyolo_root: Path, model_key: str, input_size: int) -> Path:
    """
    Ensure pretrained weights exist locally.

    Behavior:
    - Keep explicit local aliases for 416 / 640 where defined.
    - If the requested alias is missing but another alias for the same upstream
      URL already exists, copy the existing alias instead of downloading again.
    - If download is required, use retry + temporary-part file + atomic rename
      to avoid leaving truncated checkpoints behind.
    """
    import time
    import shutil

    preset = MODEL_PRESETS[model_key]
    weights_dir = edgeyolo_root / "weights"
    ensure_dir(weights_dir)

    aliases = preset.get("weights_aliases")
    if aliases:
        # Pick the most suitable alias for the requested input size.
        if input_size in aliases:
            chosen_res = input_size
        else:
            ordered = sorted(aliases.keys())
            chosen_res = ordered[0] if input_size <= ordered[0] else ordered[-1]

        chosen = aliases[chosen_res]
        chosen_path = weights_dir / chosen["name"]
        chosen_url = chosen["url"]

        # Already present and non-empty.
        if chosen_path.exists() and chosen_path.stat().st_size > 0:
            log(f"[weights] Using existing pretrained weights alias: {chosen_path}")
            return chosen_path

        # Reuse a sibling alias if it is already present and comes from the same upstream URL.
        for _, meta in aliases.items():
            sibling_path = weights_dir / meta["name"]
            if sibling_path == chosen_path:
                continue
            if sibling_path.exists() and sibling_path.stat().st_size > 0 and meta["url"] == chosen_url:
                shutil.copy2(sibling_path, chosen_path)
                log(f"[weights] Reused sibling alias by copy: {sibling_path} -> {chosen_path}")
                return chosen_path

        # Download only the requested alias.
        tmp_path = chosen_path.with_suffix(chosen_path.suffix + ".part")
        if tmp_path.exists():
            tmp_path.unlink()

        last_err = None
        for attempt in range(1, 4):
            try:
                log(
                    f"[weights] Downloading pretrained weights alias {chosen['name']} "
                    f"from {chosen_url} (attempt {attempt}/3)"
                )
                urllib.request.urlretrieve(chosen_url, tmp_path)
                if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                    raise RuntimeError("Downloaded file is missing or empty")
                tmp_path.replace(chosen_path)
                log(f"[weights] Download complete: {chosen_path}")
                return chosen_path
            except Exception as exc:
                last_err = exc
                log(f"[weights] Download failed for {chosen['name']}: {exc}")
                if tmp_path.exists():
                    tmp_path.unlink()
                time.sleep(2 * attempt)

        abort(f"Failed to obtain pretrained weights after retries: {chosen['name']} ({last_err})")

    # Generic fallback for future non-tiny presets.
    weights_path = weights_dir / preset["weights_name"]
    if weights_path.exists() and weights_path.stat().st_size > 0:
        log(f"[weights] Using existing pretrained weights: {weights_path}")
        return weights_path

    tmp_path = weights_path.with_suffix(weights_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    last_err = None
    for attempt in range(1, 4):
        try:
            log(f"[weights] Downloading pretrained weights: {preset['weights_url']} (attempt {attempt}/3)")
            urllib.request.urlretrieve(preset["weights_url"], tmp_path)
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("Downloaded file is missing or empty")
            tmp_path.replace(weights_path)
            log(f"[weights] Download complete: {weights_path}")
            return weights_path
        except Exception as exc:
            last_err = exc
            log(f"[weights] Download failed: {exc}")
            if tmp_path.exists():
                tmp_path.unlink()
            time.sleep(2 * attempt)

    abort(f"Failed to obtain pretrained weights after retries: {weights_path.name} ({last_err})")


def make_run_name(epochs: int, batch_size: int, input_size: int, now: dt.datetime) -> str:
    return f"e{epochs}_b{batch_size}_i{input_size}x{input_size}_{now.strftime('%Y%m%d_%H%M%S')}"


def make_config_key(epochs: int, batch_size: int, input_size: int) -> str:
    """Stable configuration key with no timestamp."""
    return f"e{epochs}_b{batch_size}_i{input_size}x{input_size}"


def write_yaml(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_yaml_if_missing(path: Path, payload: Dict, label: str) -> None:
    """
    Create a YAML only if it does not already exist.

    This matches the requested workflow:
    - generated YAMLs are stable
    - they are not recreated every run
    - they can be edited manually and reused as-is
    """
    ensure_dir(path.parent)
    if path.exists():
        log(f"[configs] Reusing existing {label}: {path}")
        return
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    log(f"[configs] Created {label}: {path}")


def rewrite_model_nc(base_model_yaml: Path, dst_model_yaml: Path, num_classes: int) -> None:
    """
    Copy a model YAML and rewrite the 'nc:' line.

    The file is created only if missing. If it already exists, it is left
    untouched on purpose so the user can edit it manually.
    """
    if dst_model_yaml.exists():
        log(f"[configs] Reusing existing model yaml: {dst_model_yaml}")
        return

    text = base_model_yaml.read_text(encoding="utf-8")
    if "\nnc:" not in f"\n{text}":
        abort(f"Could not find 'nc:' in model YAML: {base_model_yaml}")

    lines = text.splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith("nc:"):
            lines[i] = f"nc: {num_classes}  # number of classes"
            replaced = True
            break
    if not replaced:
        abort(f"Failed to rewrite nc in model YAML: {base_model_yaml}")

    ensure_dir(dst_model_yaml.parent)
    dst_model_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[configs] Created model yaml: {dst_model_yaml}")


def load_yaml_or_abort(path: Path, label: str) -> Dict:
    if not path.exists():
        abort(f"Expected {label} YAML not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        abort(f"{label} YAML did not parse into a dictionary: {path}")
    return data


@dataclass
class GeneratedRunFiles:
    run_name: str
    config_key: str
    run_dir: str
    generated_root: str
    dataset_yaml: str
    train_yaml: str
    runtime_train_yaml: str
    model_yaml: str
    weights_path: str
    prepared_dataset: str


def generate_edgeyolo_configs(
    repo_root: Path,
    edgeyolo_root: Path,
    prepared_root: Path,
    summary: CanonicalDatasetSummary,
    epochs: int,
    batch_size: int,
    input_size: int,
    num_workers: int,
    model_key: str,
    fp16: bool,
    device_list: List[int],
    eval_interval: int,
) -> GeneratedRunFiles:
    """
    Generate wrapper-owned YAMLs and run directories.

    Design:
    - run/output directories remain timestamped per actual training launch
    - generated config directories are stable and *not* timestamped
    - stable config files are created only if missing
    - a per-run runtime train YAML is created inside the run directory so the
      output_dir can remain per-run without forcing timestamped generated YAMLs
    """
    now = dt.datetime.now()
    run_name = make_run_name(epochs, batch_size, input_size, now)
    config_key = make_config_key(epochs, batch_size, input_size)

    run_dir = repo_root / "workspace" / "runs" / run_name

    dataset_slug = prepared_root.name
    generated_root = repo_root / "generated" / dataset_slug / config_key

    ensure_dir(run_dir)
    ensure_dir(generated_root)

    preset = MODEL_PRESETS[model_key]
    base_model_yaml = edgeyolo_root / preset["base_yaml"]
    model_yaml = generated_root / f"model_{model_key}.yaml"
    rewrite_model_nc(base_model_yaml, model_yaml, len(summary.class_names))

    dataset_yaml = generated_root / "dataset_coco.yaml"
    dataset_payload = {
        "type": "coco",
        "dataset_path": str(prepared_root),
        "kwargs": {
            "suffix": "jpg",
            "use_cache": True,
        },
        "train": {
            "image_dir": "train/images",
            "label": "annotations/instances_train.json",
        },
        "val": {
            "image_dir": "valid/images",
            "label": "annotations/instances_valid.json",
        },
        "test": {
            "test_dir": "",
        },
        "segmentaion_enabled": False,
        "names": summary.class_names,
    }
    write_yaml_if_missing(dataset_yaml, dataset_payload, "dataset yaml")

    weights_path = ensure_pretrained_weights(edgeyolo_root, model_key, input_size)

    train_yaml = generated_root / "train.yaml"
    train_payload = {
        "model_cfg": str(model_yaml),
        "weights": str(weights_path),
        "use_cfg": False,
        "output_dir": "__SET_AT_RUNTIME__",
        "save_checkpoint_for_each_epoch": False,
        "log_file": "log.txt",
        "dataset_cfg": str(dataset_yaml),
        "batch_size_per_gpu": batch_size,
        "loader_num_workers": num_workers,
        "num_threads": 1,
        "device": device_list,
        "fp16": bool(fp16),
        "cudnn_benchmark": True,
        "optimizer": "SGD",
        "max_epoch": epochs,
        "close_mosaic_epochs": 15,
        "lr_per_img": 0.00015625,
        "warmup_epochs": 5,
        "warmup_lr_ratio": 0.0,
        "final_lr_ratio": 0.05,
        "loss_use": ["bce", "bce", "giou"],
        "input_size": [input_size, input_size],
        "multiscale_range": 5,
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "enhance_mosaic": False,
        "use_ema": True,
        "enable_mixup": True,
        "mixup_scale": [0.5, 1.5],
        "mosaic_scale": [0.1, 2.0],
        "flip_prob": 0.5,
        "mosaic_prob": 1,
        "mixup_prob": 1,
        "degrees": 10,
        "hsv_gain": [0.0138, 0.664, 0.464],
        "eval_at_start": False,
        "val_conf_thres": 0.001,
        "val_nms_thres": 0.65,
        "eval_only": False,
        "obj_conf_enabled": True,
        "eval_interval": int(eval_interval),
        "print_interval": 100,
        "load_optimizer_params": True,
        "train_backbone": True,
        "train_start_layers": 51,
        "force_start_epoch": -1,
    }
    write_yaml_if_missing(train_yaml, train_payload, "train yaml")

    # Create a per-run runtime train YAML. This is the file actually passed to
    # EdgeYOLO train.py. It is intentionally placed inside the run directory so:
    # - generated YAMLs stay stable and timestamp-free
    # - output_dir can still be unique per run
    runtime_train_yaml = run_dir / "train_runtime.yaml"
    runtime_payload = load_yaml_or_abort(train_yaml, "template train")
    runtime_payload["output_dir"] = str(run_dir)
    runtime_payload["model_cfg"] = str(model_yaml)
    runtime_payload["dataset_cfg"] = str(dataset_yaml)
    runtime_payload["weights"] = str(weights_path)
    write_yaml(runtime_train_yaml, runtime_payload)

    manifest = GeneratedRunFiles(
        run_name=run_name,
        config_key=config_key,
        run_dir=str(run_dir),
        generated_root=str(generated_root),
        dataset_yaml=str(dataset_yaml),
        train_yaml=str(train_yaml),
        runtime_train_yaml=str(runtime_train_yaml),
        model_yaml=str(model_yaml),
        weights_path=str(weights_path),
        prepared_dataset=str(prepared_root),
    )
    (generated_root / "config_manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2),
        encoding="utf-8",
    )
    return manifest


# -----------------------------------------------------------------------------
# Training and export
# -----------------------------------------------------------------------------

def start_training(edgeyolo_root: Path, train_yaml: Path) -> None:
    log(f"[train] Starting EdgeYOLO training with config: {train_yaml}")
    run_cmd([sys.executable, "train.py", "--cfg", str(train_yaml)], cwd=edgeyolo_root)


def maybe_export_onnx_after_train(
    repo_root: Path,
    run_dir: Path,
    input_size: int,
    opset: int,
    export_batch: int,
    simplify: bool,
) -> None:
    best = run_dir / "best.pth"
    last = run_dir / "last.pth"
    weights = best if best.exists() else last
    if not weights.exists():
        abort(f"Could not find best.pth or last.pth under completed run dir: {run_dir}")

    export_script = repo_root / "scripts" / "export_onnx.py"
    cmd = [
        sys.executable,
        str(export_script),
        "--weights",
        str(weights),
        "--input-size",
        str(input_size),
        "--batch",
        str(export_batch),
        "--opset",
        str(opset),
    ]
    if not simplify:
        cmd.append("--no-simplify")
    run_cmd(cmd, cwd=repo_root)


# -----------------------------------------------------------------------------
# Argument parsing and main orchestration
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reusable EdgeYOLO wrapper: dataset checks, cleanup, conversion, YAML generation, training, ONNX export.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to a source dataset directory or a .zip archive.",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help=(
            "Optional class names override.\n"
            "Examples:\n"
            "  --class-names forklift person\n"
            "  --class-names forklift,person"
        ),
    )
    parser.add_argument(
        "--model",
        default="tiny",
        choices=sorted(MODEL_PRESETS.keys()),
        help="EdgeYOLO model preset to use.",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU.")
    parser.add_argument("--input-size", type=int, default=416, help="Square input size, e.g. 416 or 640.")
    parser.add_argument("--num-workers", type=int, default=8, help="Data loader worker count.")
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device list, e.g. '0' or '0,1'.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable FP16 training.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="Run validation every N epochs.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare dataset and generate configs. Do not start training.",
    )
    parser.add_argument(
        "--export-onnx-after-train",
        action="store_true",
        help="After training completes, export ONNX using the wrapper export script.",
    )
    parser.add_argument(
        "--export-opset",
        type=int,
        default=13,
        help="ONNX opset for wrapper export. Default: 13.",
    )
    parser.add_argument(
        "--export-batch",
        type=int,
        default=1,
        help="Batch size for ONNX export. Default: 1.",
    )
    parser.add_argument(
        "--no-simplify-onnx",
        action="store_true",
        help="Disable ONNX simplification during export.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = repo_root_from_script()
    edgeyolo_root = edgeyolo_root_from_repo(repo_root)
    verify_edgeyolo_checkout(edgeyolo_root)
    patch_edgeyolo_for_torch26(edgeyolo_root)

    raw_class_names = parse_class_names(args.class_names)
    device_list = [int(x.strip()) for x in args.device.split(",") if x.strip()]
    if not device_list:
        abort("--device produced an empty device list")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    working_root = repo_root / "workspace"
    ensure_dir(working_root)

    extracted_root = extract_input_if_needed(dataset_root, working_root)
    layout = detect_dataset_layout(extracted_root)
    log(f"[detect] Source format detected: {layout.format_name}")
    log(f"[detect] Source dataset root: {layout.source_root}")

    prepared_name = f"{Path(dataset_root).stem}_prepared"
    prepared_root = working_root / "prepared" / prepared_name
    summary = prepare_canonical_dataset(layout, prepared_root, raw_class_names)

    log(
        f"[summary] train images={summary.train_images}, train annotations={summary.train_annotations}, "
        f"valid images={summary.valid_images}, valid annotations={summary.valid_annotations}, "
        f"classes={summary.class_names}"
    )

    run_files = generate_edgeyolo_configs(
        repo_root=repo_root,
        edgeyolo_root=edgeyolo_root,
        prepared_root=prepared_root,
        summary=summary,
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers,
        model_key=args.model,
        fp16=bool(args.fp16),
        device_list=device_list,
        eval_interval=args.eval_interval,
    )

    log(f"[configs] generated root : {run_files.generated_root}")
    log(f"[configs] dataset yaml   : {run_files.dataset_yaml}")
    log(f"[configs] train yaml     : {run_files.train_yaml}")
    log(f"[configs] model yaml     : {run_files.model_yaml}")
    log(f"[configs] runtime yaml   : {run_files.runtime_train_yaml}")
    log(f"[run] output directory    : {run_files.run_dir}")

    if args.prepare_only:
        log("[done] Prepare-only mode requested. Training was not started.")
        return

    start_training(edgeyolo_root, Path(run_files.runtime_train_yaml))

    if args.export_onnx_after_train:
        maybe_export_onnx_after_train(
            repo_root=repo_root,
            run_dir=Path(run_files.run_dir),
            input_size=args.input_size,
            opset=args.export_opset,
            export_batch=args.export_batch,
            simplify=not args.no_simplify_onnx,
        )

    log("[done] Pipeline completed successfully.")


if __name__ == "__main__":
    main()
