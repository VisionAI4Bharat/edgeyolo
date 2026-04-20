#!/usr/bin/env python3
"""
Wrapper around EdgeYOLO's export.py for ONNX export.

Why this wrapper exists
=======================
- Keeps the command short and consistent from the wrapper repo.
- Defaults to the wrapper submodule location: third_party/edgeyolo
- Applies the same PyTorch 2.6+ checkpoint-loading patch used by the training pipeline.
- Uses opset 13 by default, but exposes --opset so you can override it easily.
- Copies the produced ONNX artifact into the wrapper repo's workspace/exports area.

Notes on opset choice
=====================
EdgeYOLO's own README examples show ONNX export with opset 11. This wrapper uses
13 as the default because it is widely supported in modern ONNX Runtime workflows,
but the flag is configurable so you can fall back to 11 if your downstream toolchain
requires it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence, Optional


def log(msg: str) -> None:
    print(msg, flush=True)


def abort(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    log("$ " + " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        abort(f"Command failed with code {result.returncode}: {' '.join(str(x) for x in cmd)}")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def edgeyolo_root_from_repo(repo_root: Path) -> Path:
    return repo_root / "third_party" / "edgeyolo"


def verify_edgeyolo_checkout(edgeyolo_root: Path) -> None:
    required = [
        edgeyolo_root / "export.py",
        edgeyolo_root / "edgeyolo" / "models" / "__init__.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        abort("EdgeYOLO submodule checkout looks incomplete. Missing: " + ", ".join(missing))


def patch_edgeyolo_for_torch26(edgeyolo_root: Path) -> None:
    target = edgeyolo_root / "edgeyolo" / "models" / "__init__.py"
    text = target.read_text(encoding="utf-8")

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


def quantize_onnx(src: Path, dst: Path, mode: str) -> None:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization import quantize_static, CalibrationDataReader
    except ImportError:
        raise SystemExit("ERROR: onnxruntime-tools not found. Install with: pip install onnxruntime-tools")

    if mode == "dynamic":
        log(f"[quantize] Dynamic INT8 quantization: {src} → {dst}")
        quantize_dynamic(
            str(src),
            str(dst),
            weight_type=QuantType.QInt8,
        )
    else:
        raise SystemExit(f"ERROR: Unknown quantize mode '{mode}'. Supported: dynamic")

    log(f"[quantize] Saved quantized model to: {dst}")


def find_latest_onnx(edgeyolo_root: Path) -> Path:
    export_root = edgeyolo_root / "output" / "export"
    if not export_root.exists():
        abort(f"Expected export root does not exist: {export_root}")
    onnx_files = sorted(export_root.rglob("*.onnx"), key=lambda p: p.stat().st_mtime)
    if not onnx_files:
        abort("No ONNX file was found under EdgeYOLO output/export after export step")
    return onnx_files[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained EdgeYOLO checkpoint to ONNX via the wrapper repo")
    parser.add_argument("--weights", required=True, help="Path to trained best.pth or last.pth")
    parser.add_argument("--input-size", type=int, default=640, help="Square input size, e.g. 416 or 640")
    parser.add_argument("--batch", type=int, default=1, help="Export batch size")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--edgeyolo-root", default=None, help="Optional manual EdgeYOLO root override")
    parser.add_argument("--no-simplify", action="store_true", help="Pass --no-simplify to EdgeYOLO export.py")
    parser.add_argument("--quantize", choices=["dynamic"], default=None,
                        help="Post-export quantization. 'dynamic' = dynamic INT8 via onnxruntime")
    args = parser.parse_args()

    repo_root = repo_root_from_script()
    edgeyolo_root = Path(args.edgeyolo_root).resolve() if args.edgeyolo_root else edgeyolo_root_from_repo(repo_root)
    verify_edgeyolo_checkout(edgeyolo_root)
    patch_edgeyolo_for_torch26(edgeyolo_root)

    weights = Path(args.weights).expanduser().resolve()
    if not weights.exists():
        abort(f"Weights file does not exist: {weights}")

    cmd = [
        sys.executable,
        "export.py",
        "--onnx-only",
        "--weights",
        str(weights),
        "--input-size",
        str(args.input_size),
        str(args.input_size),
        "--batch",
        str(args.batch),
        "--opset",
        str(args.opset),
    ]
    if args.no_simplify:
        cmd.append("--no-simplify")

    run_cmd(cmd, cwd=edgeyolo_root)

    latest_onnx = find_latest_onnx(edgeyolo_root)
    export_name = f"{weights.stem}_opset{args.opset}_{args.input_size}x{args.input_size}_b{args.batch}.onnx"
    export_dir = repo_root / "workspace" / "exports" / weights.stem
    export_dir.mkdir(parents=True, exist_ok=True)
    dst = export_dir / export_name
    shutil.copy2(latest_onnx, dst)

    manifest: dict = {
        "weights": str(weights),
        "edgeyolo_export_output": str(latest_onnx),
        "copied_export": str(dst),
        "opset": args.opset,
        "batch": args.batch,
        "input_size": args.input_size,
    }

    if args.quantize:
        q_name = f"{weights.stem}_opset{args.opset}_{args.input_size}x{args.input_size}_b{args.batch}_{args.quantize}int8.onnx"
        q_dst = export_dir / q_name
        quantize_onnx(dst, q_dst, args.quantize)
        manifest["quantized_export"] = str(q_dst)
        manifest["quantize_mode"]    = args.quantize
        log(f"[done] Quantized model: {q_dst}")

    (export_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log(f"[done] ONNX export copied to: {dst}")


if __name__ == "__main__":
    main()
