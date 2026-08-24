#!/usr/bin/env python3
"""Convert preprocessed M&M masks to binary foreground masks.

The source M&M data in this project is already preprocessed as `.npy` volumes
with masks containing labels 0, 1, 2, and 3. For binary runs requested by the
project lead, labels greater than zero are treated as one foreground class.
The source tree is left untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("training", "validation", "testing")


def prepare_mnm_binary_dataset(
    input_root: str | Path = "preprocessed_data/mnm",
    output_root: str | Path = "preprocessed_data/mnm_binary",
    overwrite: bool = False,
) -> dict[str, Any]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    if not input_root.exists():
        raise FileNotFoundError(f"M&M input directory not found: {input_root}")

    summary: dict[str, Any] = {
        "dataset": "M&M-binary",
        "source_dataset": "M&M",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_remap": {"0": 0, ">0": 1},
        "num_classes": 2,
        "class_names": ["BG", "FG"],
        "splits": {},
    }

    for split in SPLITS:
        source_split = input_root / split
        if not source_split.exists():
            continue
        split_summary = _convert_split(source_split, output_root / split, overwrite=overwrite)
        summary["splits"][split] = split_summary

    if not summary["splits"]:
        raise FileNotFoundError(
            f"No supported split folders found under {input_root}. "
            f"Expected one or more of: {', '.join(SPLITS)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def _convert_split(source_split: Path, output_split: Path, overwrite: bool = False) -> dict[str, Any]:
    source_volumes = source_split / "volumes"
    source_masks = source_split / "masks"
    if not source_volumes.exists() or not source_masks.exists():
        raise FileNotFoundError(f"Expected volumes/ and masks/ under {source_split}")

    output_volumes = output_split / "volumes"
    output_masks = output_split / "masks"
    output_volumes.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    source_ids = {path.stem for path in source_volumes.glob("*.npy")}
    mask_ids = {path.stem for path in source_masks.glob("*.npy")}
    missing_masks = sorted(source_ids - mask_ids)
    missing_volumes = sorted(mask_ids - source_ids)
    if missing_masks or missing_volumes:
        raise FileNotFoundError(
            f"Volume/mask mismatch in {source_split}: "
            f"missing masks={missing_masks[:5]}, missing volumes={missing_volumes[:5]}"
        )

    processed = 0
    skipped = 0
    volume_info: dict[str, Any] = {}
    for case_id in sorted(source_ids):
        src_volume = source_volumes / f"{case_id}.npy"
        src_mask = source_masks / f"{case_id}.npy"
        dst_volume = output_volumes / f"{case_id}.npy"
        dst_mask = output_masks / f"{case_id}.npy"
        if not overwrite and dst_volume.exists() and dst_mask.exists():
            skipped += 1
            continue

        volume = np.load(src_volume, mmap_mode="r")
        mask = np.load(src_mask, mmap_mode="r")
        if volume.shape != mask.shape:
            raise ValueError(f"Volume/mask shape mismatch for {case_id}: {volume.shape} vs {mask.shape}")

        shutil.copy2(src_volume, dst_volume)
        binary_mask = (np.asarray(mask) > 0).astype(np.uint8)
        np.save(dst_mask, binary_mask)
        volume_info[case_id] = {
            "num_slices": int(volume.shape[-1]),
            "target_size": [int(volume.shape[0]), int(volume.shape[1])],
            "source_labels": [int(v) for v in np.unique(mask).tolist()],
            "binary_labels": [0, 1],
        }
        processed += 1

    metadata = {
        "dataset": "M&M-binary",
        "source_split": str(source_split),
        "num_classes": 2,
        "class_names": ["BG", "FG"],
        "label_remap": {"0": 0, ">0": 1},
        "total_volumes": len(source_ids),
        "processed": processed,
        "skipped_existing": skipped,
        "volume_info": volume_info,
    }
    with open(output_split / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    return {"total_volumes": len(source_ids), "processed": processed, "skipped_existing": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert preprocessed M&M masks to binary foreground masks")
    parser.add_argument("--input", default="preprocessed_data/mnm")
    parser.add_argument("--output", default="preprocessed_data/mnm_binary")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = prepare_mnm_binary_dataset(args.input, args.output, overwrite=args.overwrite)
    print(f"Wrote binary M&M dataset: {args.output}")
    for split, info in summary["splits"].items():
        print(f"  {split}: processed={info['processed']} skipped={info['skipped_existing']} total={info['total_volumes']}")


if __name__ == "__main__":
    main()
