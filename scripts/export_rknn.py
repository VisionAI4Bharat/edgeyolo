#!/usr/bin/env python3
# Copyright (C) 2026 swatah.ai. All rights reserved.
#
# This software is dual-licensed:
# 1. GNU General Public License v3.0 (GPLv3)
# 2. A proprietary license for commercial use.
#
# You may use this software under the terms of the GPLv3 if you are using it
# for non-commercial purposes. For commercial usage, a separate commercial 
# license must be obtained from swatah.ai (info@swatah.ai).
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
# for more details.
#
# Trademarks: All trademarks, service marks, and logos are the property of 
# their respective owners.

"""
Convert an EdgeYOLO ONNX model to RKNN INT8 for Rockchip RV1106 / RK3588.

Workflow
========
1. (Optional) Export ONNX from a .pth checkpoint first — pass --weights instead of --onnx.
2. Convert the ONNX to RKNN using rknn-toolkit2.
3. Quantization is always INT8 (required for NPU); output filename always contains '_int8'.

Requirements
============
    pip install rknn-toolkit2          # Rockchip official Python toolkit

Output
======
All artefacts are written to:
    workspace/exports/<model_stem>/
        <name>_int8.rknn          — INT8 RKNN model
        export_manifest_rknn.json

Quantization dataset
====================
RKNN quantization requires a plain-text dataset file: one image path per line.
Pass --dataset to specify it.  If omitted the script looks for
workspace/calibration/dataset.txt relative to the repo root.

Example
=======
    python scripts/export_rknn.py \\
        --onnx workspace/exports/my_model/my_model.onnx \\
        --dataset workspace/calibration/dataset.txt \\
        --target RV1106
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


# ─── Supported Rockchip targets ───────────────────────────────────────────────

SUPPORTED_TARGETS = [
    "RV1106", "RV1103",
    "RK3588",  "RK3566", "RK3568",
    "RK3562",
]


# ─── ONNX export helper ───────────────────────────────────────────────────────

def export_onnx_from_weights(weights: Path, input_size: int, opset: int,
                              repo_root: Path) -> Path:
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


# ─── RKNN conversion ──────────────────────────────────────────────────────────

def convert_to_rknn(
    onnx_path: Path,
    output_path: Path,
    dataset_file: Path,
    target: str,
    mean_values: Sequence[float],
    std_values: Sequence[float],
    input_size: int,
) -> None:
    try:
        from rknn.api import RKNN
    except ImportError:
        abort(
            "rknn-toolkit2 not found.\n"
            "  Install from Rockchip releases:\n"
            "  pip install rknn-toolkit2"
        )

    rknn = RKNN(verbose=True)

    # ── configure ────────────────────────────────────────────────────────────
    log(f"[rknn] Configuring for target={target}  input={input_size}x{input_size}")
    ret = rknn.config(
        mean_values=[list(mean_values)],
        std_values=[list(std_values)],
        target_platform=target,
        quantized_dtype="w8a8",  # INT8 weights + INT8 activations (rknn-toolkit2 2.x)
        quantized_algorithm="normal",
        optimization_level=3,
    )
    if ret != 0:
        abort(f"rknn.config failed (code {ret})")

    # ── load ONNX ─────────────────────────────────────────────────────────────
    log(f"[rknn] Loading ONNX: {onnx_path}")
    ret = rknn.load_onnx(
        model=str(onnx_path),
        input_size_list=[[1, 3, input_size, input_size]],
    )
    if ret != 0:
        abort(f"rknn.load_onnx failed (code {ret})")

    # ── build with INT8 quantization ──────────────────────────────────────────
    log(f"[rknn] Building with INT8 quantization — dataset: {dataset_file}")
    ret = rknn.build(
        do_quantization=True,
        dataset=str(dataset_file),
    )
    if ret != 0:
        abort(f"rknn.build failed (code {ret})")

    # ── export ────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[rknn] Exporting to: {output_path}")
    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        abort(f"rknn.export_rknn failed (code {ret})")

    rknn.release()
    log(f"[rknn] Done — {output_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Export EdgeYOLO ONNX to RKNN INT8 for Rockchip NPU"
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--onnx",    type=Path,
                     help="Path to an already-exported .onnx file")
    src.add_argument("--weights", type=Path,
                     help="Path to a .pth checkpoint (auto-exports ONNX first)")

    parser.add_argument("--target", choices=SUPPORTED_TARGETS, default="RV1106",
                        help=f"Rockchip target platform (default: RV1106). "
                             f"Supported: {', '.join(SUPPORTED_TARGETS)}")
    parser.add_argument("--input-size", type=int, default=416,
                        help="Square input size (default: 416)")
    parser.add_argument("--opset", type=int, default=13,
                        help="ONNX opset for intermediate export (ignored if --onnx is given)")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Path to dataset.txt for INT8 calibration "
                             "(default: workspace/calibration/dataset.txt)")
    parser.add_argument("--mean", type=float, nargs=3,
                        default=[0.0, 0.0, 0.0],
                        metavar=("R", "G", "B"),
                        help="Per-channel mean for normalisation (default: 0 0 0)")
    parser.add_argument("--std",  type=float, nargs=3,
                        default=[255.0, 255.0, 255.0],
                        metavar=("R", "G", "B"),
                        help="Per-channel std for normalisation (default: 255 255 255)")

    args = parser.parse_args()

    # ── resolve ONNX ─────────────────────────────────────────────────────────
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

    # ── resolve dataset ───────────────────────────────────────────────────────
    dataset_file = args.dataset.expanduser().resolve() if args.dataset else \
                   repo_root / "workspace" / "calibration" / "dataset.txt"
    if not dataset_file.exists():
        abort(
            f"Calibration dataset file not found: {dataset_file}\n"
            "  Create a plain-text file listing one representative image path per line.\n"
            "  Pass --dataset <path> to specify its location."
        )

    # ── output path: always contains _int8 ───────────────────────────────────
    export_dir  = repo_root / "workspace" / "exports" / stem
    output_name = f"{stem}_{args.target.lower()}_int8.rknn"
    output_path = export_dir / output_name

    log(f"[rknn] Source ONNX : {onnx_path}")
    log(f"[rknn] Target      : {args.target}")
    log(f"[rknn] Output      : {output_path}")
    log(f"[rknn] Dataset     : {dataset_file}")

    convert_to_rknn(
        onnx_path=onnx_path,
        output_path=output_path,
        dataset_file=dataset_file,
        target=args.target,
        mean_values=args.mean,
        std_values=args.std,
        input_size=args.input_size,
    )

    manifest = {
        "source_onnx":  str(onnx_path),
        "rknn_model":   str(output_path),
        "target":       args.target,
        "input_size":   args.input_size,
        "quantization": "int8",
        "dataset":      str(dataset_file),
        "mean":         args.mean,
        "std":          args.std,
    }
    (export_dir / "export_manifest_rknn.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    log(f"[done] RKNN INT8 model: {output_path}")


if __name__ == "__main__":
    main()
