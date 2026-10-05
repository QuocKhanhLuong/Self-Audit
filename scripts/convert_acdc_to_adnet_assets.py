#!/usr/bin/env python3
"""Convert ACDC paired volumes into ADNet few-shot audit assets.

The output is an ``adnet.asset_spec.v1`` plus query/support/GT arrays laid out
for ``scripts/prepare_adnet_fewshot_manifests.py``.  Query records remain
image-only; GT paths are emitted only in the evaluator section of the spec.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_audit.data.acdc import discover_acdc_records, resolve_effective_acdc_splits  # noqa: E402
from self_audit.data.common import load_array, resize_in_plane, to_depth_first  # noqa: E402
from shared_benchmark.adnet_fewshot import atomic_write_json, atomic_write_npy, file_sha256  # noqa: E402
from shared_benchmark.provenance import sha256_json  # noqa: E402


CLASS_MAPPING = {"RV": 1, "MYO": 2, "LV": 3}
CLASS_NAMES = {1: "RV", 2: "MYO", 3: "LV"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--support-split", default="train")
    parser.add_argument("--query-split", default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--max-query-cases", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = convert_acdc_to_adnet_assets(
        data_root=args.data_root,
        output_root=args.output_root,
        split_manifest=args.split_manifest,
        support_split=args.support_split,
        query_split=args.query_split,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        target_size=args.target_size,
        max_query_cases=args.max_query_cases,
        force=args.force,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def convert_acdc_to_adnet_assets(
    *,
    data_root: Path,
    output_root: Path,
    split_manifest: Path | None = None,
    support_split: str = "train",
    query_split: str = "val",
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    target_size: int = 224,
    max_query_cases: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise ValueError(f"refusing to write into non-empty output root: {output_root}")
    for child in ("query", "support", "gt"):
        (output_root / child).mkdir(parents=True, exist_ok=True)

    records = discover_acdc_records(data_root)
    effective = resolve_effective_acdc_splits(
        records,
        split_manifest=split_manifest,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    support_records = list(effective[support_split])
    query_records = list(effective[query_split])
    if max_query_cases is not None:
        query_records = query_records[: int(max_query_cases)]
    if not support_records:
        raise ValueError(f"support split {support_split!r} selected zero ACDC records")
    if not query_records:
        raise ValueError(f"query split {query_split!r} selected zero ACDC records")

    support_assets = _select_support_assets(
        support_records,
        output_root=output_root,
        target_size=target_size,
    )
    query_assets = [
        _write_query_and_gt(record, output_root=output_root, target_size=target_size, split=query_split)
        for record in query_records
    ]

    spec = {
        "schema": "adnet.asset_spec.v1",
        "class_mapping": CLASS_MAPPING,
        "queries": [row["query"] for row in query_assets],
        "supports": support_assets,
        "gt": [row["gt"] for row in query_assets],
    }
    spec_path = output_root / "adnet_assets.json"
    atomic_write_json(spec_path, spec)

    receipt = {
        "schema": "adnet.acdc_conversion_receipt.v1",
        "asset_spec_path": str(spec_path),
        "asset_spec_sha256": file_sha256(spec_path),
        "asset_spec_scientific_sha256": sha256_json(spec),
        "data_root": str(Path(data_root).resolve()),
        "split_manifest": None if split_manifest is None else str(Path(split_manifest).resolve()),
        "split_strategy": effective.strategy,
        "split_signature": effective.signature,
        "support_split": support_split,
        "query_split": query_split,
        "target_size": int(target_size),
        "canonical_labels": {"0": "BG", "1": "RV", "2": "MYO", "3": "LV"},
        "query_count": len(query_assets),
        "support_count": len(support_assets),
        "gt_count": len(query_assets),
        "support_policy": "first split volume with non-empty class; median positive slice",
        "queries": [row["receipt"] for row in query_assets],
        "supports": [
            {
                "support_id": row["support_id"],
                "class_name": row["class_name"],
                "canonical_class_id": row["canonical_class_id"],
                "source_case_id": row["metadata"]["source_case_id"],
                "slice_indices": row["slice_indices"],
            }
            for row in support_assets
        ],
    }
    receipt_path = output_root / "conversion_receipt.json"
    atomic_write_json(receipt_path, receipt)
    return receipt


def _select_support_assets(records: list[Any], *, output_root: Path, target_size: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_cases: set[str] = set()
    for class_id, class_name in sorted(CLASS_NAMES.items()):
        for record in records:
            image, mask = _load_converted_pair(record, target_size=target_size)
            positive_slices = np.flatnonzero((mask == class_id).sum(axis=(1, 2)) > 0)
            if positive_slices.size == 0:
                continue
            slice_index = int(positive_slices[len(positive_slices) // 2])
            stem = _safe_stem(record.case_id)
            image_rel = Path("support") / f"{stem}.npy"
            mask_rel = Path("support") / f"{stem}_{class_name.lower()}_mask.npy"
            atomic_write_npy(output_root / image_rel, image)
            atomic_write_npy(output_root / mask_rel, (mask == class_id).astype(np.uint8, copy=False))
            selected.append({
                "support_id": f"{stem}_{class_name.lower()}",
                "class_name": class_name,
                "canonical_class_id": class_id,
                "image_path": str(image_rel),
                "mask_path": str(mask_rel),
                "slice_indices": [slice_index],
                "metadata": {"source_case_id": record.case_id},
            })
            used_cases.add(record.case_id)
            break
        else:
            raise ValueError(f"could not find non-empty ACDC support for class {class_id} ({class_name})")
    if not used_cases:
        raise ValueError("no support assets selected")
    return selected


def _write_query_and_gt(record: Any, *, output_root: Path, target_size: int, split: str) -> dict[str, Any]:
    image, mask = _load_converted_pair(record, target_size=target_size)
    stem = _safe_stem(record.case_id)
    image_rel = Path("query") / f"{stem}.npy"
    gt_rel = Path("gt") / f"{stem}.npy"
    atomic_write_npy(output_root / image_rel, image)
    atomic_write_npy(output_root / gt_rel, mask.astype(np.uint8, copy=False))
    return {
        "query": {
            "sample_id": stem,
            "image_path": str(image_rel),
            "split": str(split),
            "metadata": {
                "dataset": "ACDC",
                "case_id": record.case_id,
                "patient_id": record.patient_id,
                "source_format": record.source_format,
                "source_image_sha256": file_sha256(record.image_path),
            },
        },
        "gt": {"sample_id": stem, "gt_path": str(gt_rel)},
        "receipt": {
            "sample_id": stem,
            "source_case_id": record.case_id,
            "image_shape": [int(dim) for dim in image.shape],
            "gt_shape": [int(dim) for dim in mask.shape],
            "source_image": str(record.image_path),
            "source_mask": str(record.mask_path),
        },
    }


def _load_converted_pair(record: Any, *, target_size: int) -> tuple[np.ndarray, np.ndarray]:
    image_raw, _ = load_array(record.image_path)
    mask_raw, _ = load_array(record.mask_path)
    image = to_depth_first(np.asarray(image_raw), depth_axis=2 if record.source_format == "npy" else None)
    mask = to_depth_first(np.asarray(mask_raw), depth_axis=2 if record.source_format == "npy" else None)
    if image.shape != mask.shape:
        raise ValueError(f"ACDC image/mask shape mismatch for {record.case_id}: {image.shape} vs {mask.shape}")
    image = _zscore(image)
    image = resize_in_plane(image, int(target_size), is_mask=False).astype(np.float32, copy=False)
    mask = resize_in_plane(mask, int(target_size), is_mask=True).astype(np.uint8, copy=False)
    _validate_converted_pair(record.case_id, image, mask)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def _zscore(value: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(array, [0.5, 99.5])
    clipped = np.clip(array, low, high)
    std = float(clipped.std())
    if std <= 0.0 or not np.isfinite(std):
        raise ValueError("ACDC image is constant after clipping")
    return ((clipped - float(clipped.mean())) / std).astype(np.float32, copy=False)


def _validate_converted_pair(case_id: str, image: np.ndarray, mask: np.ndarray) -> None:
    if image.ndim != 3 or mask.ndim != 3:
        raise ValueError(f"converted ACDC assets must be [Z,H,W] for {case_id}")
    if image.shape != mask.shape:
        raise ValueError(f"converted ACDC image/GT shape mismatch for {case_id}")
    if not np.isfinite(image).all():
        raise ValueError(f"converted ACDC image contains non-finite values for {case_id}")
    if float(image.std()) <= 0.0:
        raise ValueError(f"converted ACDC image is constant for {case_id}")
    unknown = sorted(set(int(value) for value in np.unique(mask)) - {0, 1, 2, 3})
    if unknown:
        raise ValueError(f"converted ACDC mask has labels outside 0..3 for {case_id}: {unknown}")


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:96] or "case"


if __name__ == "__main__":
    raise SystemExit(main())
