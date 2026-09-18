"""Audit CMR-MULTI pairing, flattened Z/T inference, labels, and geometry."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_audit.data.cmr_multi import CMRMultiAdapter


def _stats(array: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(array, dtype=np.float32)
    return float(values.min()), float(values.max()), float(values.mean()), float(values.std())


def run_audit(data_root: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter = CMRMultiAdapter(data_root, require_confident=False)
    records = adapter.audit()
    pairs = {pair.case_id: pair for pair in adapter.pairs}
    audit_path = output / "cmr_multi_audit.csv"
    zt_path = output / "cmr_multi_zt_inference.csv"
    audit_fields = [
        "case_id", "subject_id", "image_path", "mask_path", "image_shape",
        "mask_shape", "image_dtype", "mask_dtype", "spacing_x", "spacing_y",
        "spacing_3rd_axis", "affine_equal", "unique_labels", "image_min",
        "image_max", "image_mean", "image_std", "nonzero_mask_voxels", "error",
    ]
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        for record in records:
            image_min = image_max = image_mean = image_std = ""
            nonzero = ""
            try:
                image, mask, _, _, _ = adapter._load_pair_arrays(pairs[record.case_id])
                image_min, image_max, image_mean, image_std = _stats(image)
                nonzero = int(np.count_nonzero(mask))
            except Exception:
                pass
            spacing = tuple(record.spacing)
            writer.writerow({
                "case_id": record.case_id,
                "subject_id": record.subject_id,
                "image_path": record.image_path,
                "mask_path": record.mask_path,
                "image_shape": "x".join(map(str, record.shape)),
                "mask_shape": "x".join(map(str, record.shape)),
                "image_dtype": record.image_dtype,
                "mask_dtype": record.mask_dtype,
                "spacing_x": spacing[0] if len(spacing) > 0 else "",
                "spacing_y": spacing[1] if len(spacing) > 1 else "",
                "spacing_3rd_axis": spacing[2] if len(spacing) > 2 else "",
                "affine_equal": record.affine_equal,
                "unique_labels": ",".join(map(str, record.unique_labels)),
                "image_min": image_min,
                "image_max": image_max,
                "image_mean": image_mean,
                "image_std": image_std,
                "nonzero_mask_voxels": nonzero,
                "error": record.error,
            })
    zt_fields = [
        "case_id", "N", "Z", "T", "score", "temporal_dice", "wrap_dice",
        "spatial_dice", "derived_z_spacing", "confidence", "status",
        "score_gap", "reason",
    ]
    with zt_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=zt_fields)
        writer.writeheader()
        for record in records:
            inference = record.inference
            writer.writerow({
                "case_id": record.case_id,
                "N": record.shape[2] if len(record.shape) == 3 else "",
                "Z": inference.z if inference else "",
                "T": inference.t if inference else "",
                "score": inference.score if inference else "",
                "temporal_dice": inference.temporal_dice if inference else "",
                "wrap_dice": inference.wrap_dice if inference else "",
                "spatial_dice": inference.spatial_dice if inference else "",
                "derived_z_spacing": inference.derived_z_spacing if inference else "",
                "confidence": inference.confidence if inference else "error",
                "status": inference.status if inference else "error",
                "score_gap": inference.score_gap if inference else "",
                "reason": inference.reason if inference else record.error,
            })
    return audit_path, zt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    paths = run_audit(args.data_root, args.output_dir)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
