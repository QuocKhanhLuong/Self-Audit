"""Audit CMRxMotion archives/trees, masks, spacing, and affine alignment."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_audit.data.cmrxmotion import CMRxMotionAdapter


def run_audit(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    allow_affine_mismatch: bool = False,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter = CMRxMotionAdapter(data_root, allow_affine_mismatch=allow_affine_mismatch)
    records = adapter.audit()
    path = output / "cmrxmotion_audit.csv"
    fields = [
        "subject_id", "acquisition_id", "phase", "case_id", "image_path",
        "mask_path", "has_mask", "image_shape", "mask_shape", "image_dtype",
        "mask_dtype", "spacing_x", "spacing_y", "spacing_z", "affine_equal",
        "unique_labels", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            spacing = tuple(record.spacing)
            writer.writerow({
                "subject_id": record.subject_id,
                "acquisition_id": record.acquisition_id,
                "phase": record.phase,
                "case_id": record.case_id,
                "image_path": record.image_path,
                "mask_path": record.mask_path,
                "has_mask": record.has_mask,
                "image_shape": "x".join(map(str, record.image_shape)),
                "mask_shape": "x".join(map(str, record.mask_shape)),
                "image_dtype": record.image_dtype,
                "mask_dtype": record.mask_dtype,
                "spacing_x": spacing[0] if len(spacing) > 0 else "",
                "spacing_y": spacing[1] if len(spacing) > 1 else "",
                "spacing_z": spacing[2] if len(spacing) > 2 else "",
                "affine_equal": record.affine_equal,
                "unique_labels": ",".join(map(str, record.unique_labels)),
                "error": record.error,
            })
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--allow-affine-mismatch", action="store_true")
    args = parser.parse_args()
    print(run_audit(
        args.data_root,
        args.output_dir,
        allow_affine_mismatch=args.allow_affine_mismatch,
    ))


if __name__ == "__main__":
    main()
