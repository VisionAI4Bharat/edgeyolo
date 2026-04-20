#!/usr/bin/env python3
"""
Convert an EdgeYOLO ONNX model to OpenVINO IR with optional INT8 quantization.

Workflow
========
1. (Optional) Export ONNX from a .pth checkpoint first — pass --weights instead of --onnx.
2. Convert the ONNX to OpenVINO IR (FP32 or FP16) using the openvino Python API.
3. (Optional) Quantize to INT8 using NNCF post-training quantization.
   INT8 needs a directory of representative calibration images (--calibration-dir).

Requirements
============
    pip install openvino          # conversion + FP16
    pip install nncf openvino-dev # only needed for --quantize int8

Output
======
All artefacts are written to:
    workspace/exports/<model_stem>/
        <name>.xml / <name>.bin       — OpenVINO FP32 (or FP16) IR
        <name>_int8.xml / _int8.bin   — INT8 IR (with --quantize int8)
        export_manifest_ov.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


def log(msg: str) -> None:
    print(msg, flush=True)


def abort(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


# ─── ONNX export helper (reuses export_onnx.py logic) ────────────────────────

def export_onnx_from_weights(weights: Path, input_size: int, opset: int,
                              repo_root: Path) -> Path:
    """Run scripts/export_onnx.py and return the path to the produced ONNX."""
    import subprocess
    script = repo_root / "scripts" / "export_onnx.py"
    cmd = [
        sys.executable, str(script),
        "--weights", str(weights),
        "--input-size", str(input_size),
        "--opset", str(opset),
    ]
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        abort(f"ONNX export failed (code {r.returncode})")

    export_dir = repo_root / "workspace" / "exports" / weights.stem
    onnx_files = sorted(export_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
    if not onnx_files:
        abort(f"No ONNX found under {export_dir} after export")
    return onnx_files[-1]


# ─── OpenVINO conversion ──────────────────────────────────────────────────────

def convert_to_ir(onnx_path: Path) -> object:
    """Convert ONNX → OpenVINO model object (always FP32 in memory).
    FP16 compression is applied at save time via compress_to_fp16=True."""
    try:
        import openvino as ov
    except ImportError:
        abort("openvino package not found. Install with: pip install openvino")

    log(f"[openvino] Converting {onnx_path.name} → IR")
    return ov.convert_model(str(onnx_path))


def save_ir(model, dst_xml: Path, fp16: bool = False) -> None:
    """Save IR; compress_to_fp16 is the stable 2024.x API for FP16 conversion."""
    import openvino as ov
    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(dst_xml), compress_to_fp16=fp16)
    tag = "FP16" if fp16 else "FP32"
    log(f"[openvino] Saved {tag} IR: {dst_xml}")


# ─── NNCF version compatibility probe ────────────────────────────────────────

def _check_nncf_ov_compat() -> None:
    """
    NNCF uses `ov.Node` as a type annotation inside OVOpMetatype.  That
    attribute was removed in OpenVINO 2024.x.  The incompatibility surfaces as
    an AttributeError at import time, before any user code runs.

    Compatible pairings (from NNCF release notes):
        NNCF 2.12  ↔  OpenVINO 2024.3
        NNCF 2.13  ↔  OpenVINO 2024.4 – 2024.6
        NNCF 2.14+ ↔  OpenVINO 2025.x

    If incompatible versions are detected we print actionable pip commands and
    raise SystemExit so the FP32/FP16 IR is still saved before we abort.
    """
    try:
        import openvino as ov
        import nncf
    except ImportError as e:
        abort(f"Missing package: {e}\n  pip install nncf openvino")

    ov_ver   = getattr(ov,   "__version__", "unknown")
    nncf_ver = getattr(nncf, "__version__", "unknown")

    try:
        # Trigger the import that fails when versions are mismatched
        from nncf.openvino.graph.metatypes import openvino_metatypes  # noqa: F401
    except AttributeError:
        abort(
            f"NNCF {nncf_ver} is incompatible with OpenVINO {ov_ver}.\n"
            f"  ov.Node was removed in OpenVINO 2024.x; NNCF must be updated to match.\n\n"
            f"  Fix for OpenVINO 2024.3:  pip install 'nncf==2.12.*'\n"
            f"  Fix for OpenVINO 2024.4–2024.6:  pip install 'nncf==2.13.*'\n"
            f"  Fix for OpenVINO 2025.x:  pip install 'nncf>=2.14'\n\n"
            f"  Your versions:  nncf={nncf_ver}  openvino={ov_ver}\n"
            f"  Check the compatibility table at:\n"
            f"  https://github.com/openvinotoolkit/nncf#nncf-pytorch-openvino-compatibility-table"
        )
    except ImportError as e:
        abort(f"NNCF import failed: {e}\n  pip install nncf openvino")


# ─── INT8 quantization via NNCF ───────────────────────────────────────────────

def build_calibration_dataset(calibration_dir: Path, input_shape: Sequence[int]):
    """
    Build an NNCF Dataset from JPEG/PNG images in calibration_dir.
    Images are letterbox-resized to (H, W) = input_shape[2:] and converted
    to float32 CHW BGR 0-255 — identical to the C++ inference preprocessing.
    """
    import cv2
    import numpy as np
    import nncf

    H, W = int(input_shape[2]), int(input_shape[3])
    image_paths = sorted(
        p for p in calibration_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        abort(f"No images found in calibration directory: {calibration_dir}")

    log(f"[nncf] Calibration dataset: {len(image_paths)} images from {calibration_dir}")

    def preprocess(path: Path):
        img = cv2.imread(str(path))
        if img is None:
            return None
        ih, iw = img.shape[:2]
        scale = min(W / iw, H / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = np.full((H, W, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = resized
        # HWC BGR → CHW BGR float32 0-255 (matches C++ blobFromBGR)
        return padded.transpose(2, 0, 1).astype(np.float32)[np.newaxis]

    samples = [s for p in image_paths if (s := preprocess(p)) is not None]
    return nncf.Dataset(samples)


def quantize_int8(model, calibration_dir: Path, input_shape: Sequence[int]):
    import nncf

    _check_nncf_ov_compat()  # aborts with a clear message if versions mismatch

    log("[nncf] Building calibration dataset…")
    dataset = build_calibration_dataset(calibration_dir, input_shape)

    log("[nncf] Running INT8 post-training quantization (this may take a few minutes)…")
    quantized = nncf.quantize(
        model,
        dataset,
        preset=nncf.QuantizationPreset.MIXED,  # symmetric weights, asymmetric activations
    )
    log("[nncf] INT8 quantization complete")
    return quantized


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export EdgeYOLO to OpenVINO IR with optional INT8 quantization"
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--onnx",    type=Path, help="Path to an already-exported .onnx file")
    src.add_argument("--weights", type=Path, help="Path to a .pth checkpoint (auto-exports ONNX first)")

    parser.add_argument("--input-size",  type=int, default=416,
                        help="Square input size used during ONNX export (ignored if --onnx is given)")
    parser.add_argument("--opset",       type=int, default=13,
                        help="ONNX opset for the intermediate export (ignored if --onnx is given)")
    parser.add_argument("--fp16",        action="store_true",
                        help="Convert weights to FP16 in the IR (faster on GPU/VPU, same inputs)")
    parser.add_argument("--quantize",    choices=["int8"], default=None,
                        help="Post-training quantization mode")
    parser.add_argument("--calibration-dir", type=Path, default=None,
                        help="Directory of representative images for INT8 calibration (required with --quantize int8)")
    parser.add_argument("--edgeyolo-root", type=Path, default=None,
                        help="Manual override for the EdgeYOLO submodule root")

    args = parser.parse_args()

    if args.quantize == "int8" and args.calibration_dir is None:
        abort("--calibration-dir is required when using --quantize int8")

    repo_root = Path(__file__).resolve().parents[1]

    # ── resolve ONNX path ─────────────────────────────────────────────────────
    if args.onnx:
        onnx_path = args.onnx.expanduser().resolve()
        if not onnx_path.exists():
            abort(f"ONNX file not found: {onnx_path}")
        stem = onnx_path.stem
    else:
        weights = args.weights.expanduser().resolve()
        if not weights.exists():
            abort(f"Weights file not found: {weights}")
        onnx_path = export_onnx_from_weights(weights, args.input_size, args.opset, repo_root)
        stem = weights.stem

    log(f"[openvino] Source ONNX: {onnx_path}")

    export_dir = repo_root / "workspace" / "exports" / stem
    export_dir.mkdir(parents=True, exist_ok=True)

    # ── convert ONNX → OV model (FP32 in memory) ─────────────────────────────
    ov_model = convert_to_ir(onnx_path)

    # Save FP32 IR (always produced)
    ir_fp32_xml = export_dir / f"{stem}_fp32.xml"
    save_ir(ov_model, ir_fp32_xml, fp16=False)

    manifest: dict = {
        "source_onnx": str(onnx_path),
        "ir_fp32_xml": str(ir_fp32_xml),
    }

    # Save FP16 IR if requested (uses compress_to_fp16 — stable across OV 2024.x)
    if args.fp16:
        ir_fp16_xml = export_dir / f"{stem}_fp16.xml"
        save_ir(ov_model, ir_fp16_xml, fp16=True)
        manifest["ir_fp16_xml"] = str(ir_fp16_xml)

    # ── optional INT8 quantization ────────────────────────────────────────────
    if args.quantize == "int8":
        # Probe NNCF/OV version compatibility before building the dataset
        _check_nncf_ov_compat()

        # Derive input shape directly from the model graph — no assumptions
        input_shape = list(ov_model.inputs[0].shape)

        calibration_dir = args.calibration_dir.expanduser().resolve()
        if not calibration_dir.is_dir():
            abort(f"Calibration directory not found: {calibration_dir}")

        int8_model = quantize_int8(ov_model, calibration_dir, input_shape)
        int8_xml = export_dir / f"{stem}_int8.xml"
        save_ir(int8_model, int8_xml, fp16=False)
        manifest["ir_int8_xml"] = str(int8_xml)
        log(f"[done] INT8 IR: {int8_xml}")

    (export_dir / "export_manifest_ov.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    log(f"[done] OpenVINO IR saved to: {export_dir}")


if __name__ == "__main__":
    main()
