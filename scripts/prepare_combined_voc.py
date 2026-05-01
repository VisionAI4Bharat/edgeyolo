#!/usr/bin/env python3
"""
Prepare a combined (unsplit) Pascal VOC dataset for pipeline.py.

Handles datasets where images and XML annotations are spread across multiple
subdirectories with no existing train/valid split (e.g. LPRNet_Dataset with
google_images/, State-wise_OLX/, video_images/ subfolders).

Output written to:
    workspace/prepared/<dataset_stem>/
        train/images/
        train/labels/
        valid/images/
        valid/labels/

Standalone usage:
    python scripts/prepare_combined_voc.py --dataset-root D:/ml/datasets/LPRNet_Dataset

Called automatically by pipeline.py when no train/valid split is detected.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_combined_voc(root: Path) -> bool:
    """Return True when root has no train/valid dirs but contains XML files in subdirs."""
    if (root / "train").exists() or (root / "valid").exists():
        return False
    for subdir in root.iterdir():
        if subdir.is_dir() and any(p.suffix.lower() == ".xml" for p in subdir.rglob("*") if p.is_file()):
            return True
    return False


def _collect_pairs(dataset_root: Path) -> List[Tuple[Path, Path, str]]:
    imgs: dict[str, list[Path]] = {}
    xmls: dict[str, list[Path]] = {}

    for p in sorted(dataset_root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMG_EXTS:
            imgs.setdefault(p.stem.lower(), []).append(p)
        elif ext == ".xml":
            xmls.setdefault(p.stem.lower(), []).append(p)

    pairs: List[Tuple[Path, Path, str]] = []
    seen: set[str] = set()

    for stem_lower, img_list in sorted(imgs.items()):
        if stem_lower not in xmls:
            continue
        xml_list = xmls[stem_lower]

        for img_path in img_list:
            # Prefer the XML that lives in the same directory as the image.
            xml_path = next(
                (x for x in xml_list if x.parent == img_path.parent), xml_list[0]
            )
            dest_stem = img_path.stem
            if dest_stem.lower() in seen:
                dest_stem = f"{img_path.parent.name}__{img_path.stem}"
            if dest_stem.lower() in seen:
                dest_stem = f"{img_path.parent.parent.name}__{img_path.parent.name}__{img_path.stem}"
            seen.add(dest_stem.lower())
            pairs.append((img_path, xml_path, dest_stem))

    return pairs


def _copy_split(pairs: List[Tuple[Path, Path, str]], split_name: str, out_root: Path) -> None:
    img_dir = out_root / split_name / "images"
    lbl_dir = out_root / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    total = len(pairs)
    for i, (img_path, xml_path, dest_stem) in enumerate(pairs, 1):
        shutil.copy2(img_path, img_dir / (dest_stem + img_path.suffix.lower()))
        shutil.copy2(xml_path, lbl_dir / (dest_stem + ".xml"))
        if i % 100 == 0 or i == total:
            print(f"[split:{split_name}] {i}/{total} files copied", flush=True)


def split_combined_voc(
    source: Path,
    output: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[int, int]:
    """
    Collect all matched image+XML pairs from source, split, copy to output.
    Returns (n_train, n_valid).
    """
    if output.exists():
        shutil.rmtree(output)

    pairs = _collect_pairs(source)
    if not pairs:
        raise SystemExit(f"ERROR: No matched image+XML pairs found under: {source}")

    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n_train = max(1, round(len(shuffled) * train_ratio))
    _copy_split(shuffled[:n_train], "train", output)
    _copy_split(shuffled[n_train:], "valid", output)

    return n_train, len(shuffled) - n_train


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True, help="Path to raw combined VOC dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split fraction (default 0.8).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument("--output-dir", default=None, help="Output dir. Defaults to workspace/prepared/<dataset_stem>.")
    args = parser.parse_args()

    source = Path(args.dataset_root).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"ERROR: not found: {source}")

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _repo_root() / "workspace" / "prepared" / source.stem
    )

    print(f"[prepare_combined_voc] source : {source}")
    print(f"[prepare_combined_voc] output : {out_dir}")
    n_train, n_valid = split_combined_voc(source, out_dir, args.train_ratio, args.seed)
    print(f"[prepare_combined_voc] train  : {n_train}")
    print(f"[prepare_combined_voc] valid  : {n_valid}")
    print(f"[prepare_combined_voc] Done. Pass to pipeline.py:")
    print(f'          --dataset-root "{out_dir}"')


if __name__ == "__main__":
    main()
