#!/usr/bin/env python3
"""Preflight ADNet few-shot manifests before producer execution."""
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

from shared_benchmark.adnet_fewshot import (  # noqa: E402
    ADNetContractError,
    file_sha256,
    load_query_manifest,
    load_support_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--query-image-root", type=Path)
    parser.add_argument("--support-asset-root", type=Path)
    parser.add_argument("--required-classes", default="1,2,3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required_classes = _parse_required_classes(args.required_classes)
    result = preflight(
        query_manifest_path=args.query_manifest,
        support_manifest_path=args.support_manifest,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        required_classes=required_classes,
        query_image_root=args.query_image_root,
        support_asset_root=args.support_asset_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def preflight(
    *,
    query_manifest_path: Path,
    support_manifest_path: Path,
    checkpoint: Path,
    output_root: Path,
    required_classes: tuple[int, ...],
    query_image_root: Path | None = None,
    support_asset_root: Path | None = None,
) -> dict[str, Any]:
    query = load_query_manifest(query_manifest_path, image_root=query_image_root)
    support = load_support_manifest(support_manifest_path, asset_root=support_asset_root)
    if not checkpoint.is_file():
        raise ADNetContractError(f"checkpoint is missing: {checkpoint}")
    checkpoint_sha256 = file_sha256(checkpoint)
    if output_root.exists() and any(output_root.iterdir()):
        raise ADNetContractError(f"output root is not empty: {output_root}")
    observed_classes = {item.canonical_class_id for item in support.supports}
    missing = sorted(set(required_classes) - observed_classes)
    if missing:
        raise ADNetContractError(f"support manifest missing required classes: {missing}")

    query_shapes = {}
    for record in query.records:
        volume = _read_volume(record.image_path, role="query image")
        _require_nonconstant(volume, role=f"query image {record.sample_id}")
        query_shapes[record.sample_id] = [int(dim) for dim in volume.shape]

    support_shapes = {}
    for item in support.supports:
        image = _read_volume(item.image_path, role="support image")
        mask = _read_volume(item.mask_path, role="support mask")
        if image.shape != mask.shape:
            raise ADNetContractError(f"support image/mask shape mismatch: {item.support_id}")
        _require_nonconstant(image, role=f"support image {item.support_id}")
        if not np.isin(mask, [0, 1]).all():
            raise ADNetContractError(f"support mask must be binary: {item.support_id}")
        for index in item.slice_indices:
            if index >= mask.shape[0]:
                raise ADNetContractError(f"support slice index outside volume: {item.support_id}")
            if int(np.asarray(mask[index] != 0).sum()) <= 0:
                raise ADNetContractError(f"support slice is empty: {item.support_id}:{index}")
        support_shapes[item.support_id] = [int(dim) for dim in image.shape]

    return {
        "status": "READY",
        "schema": "adnet.preflight.v1",
        "query_manifest_sha256": query.sha256,
        "support_manifest_sha256": support.sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "query_count": len(query.records),
        "support_count": len(support.supports),
        "required_classes": [int(value) for value in required_classes],
        "query_shapes": query_shapes,
        "support_shapes": support_shapes,
    }


def _read_volume(path: Path, *, role: str) -> np.ndarray:
    name = path.name.lower()
    if name.endswith(".npy"):
        value = np.load(path, allow_pickle=False)
    elif name.endswith(".nii") or name.endswith(".nii.gz"):
        try:
            import SimpleITK as sitk
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ADNetContractError(f"SimpleITK is required to read {role}: {exc}") from exc
        value = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    else:
        raise ADNetContractError(f"unsupported {role} container: {path.name}")
    array = np.asarray(value)
    if array.ndim != 3 or not np.issubdtype(array.dtype, np.number):
        raise ADNetContractError(f"{role} must be numeric [Z,H,W]")
    if not np.isfinite(array).all():
        raise ADNetContractError(f"{role} contains non-finite values")
    return np.ascontiguousarray(array)


def _require_nonconstant(value: np.ndarray, *, role: str) -> None:
    std = float(np.asarray(value, dtype=np.float32).std())
    if std <= 0.0 or not np.isfinite(std):
        raise ADNetContractError(f"{role} has invalid standard deviation")


def _parse_required_classes(value: str) -> tuple[int, ...]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        class_id = int(item)
        if class_id < 1 or class_id > 3:
            raise ADNetContractError("required classes must be canonical foreground ids 1,2,3")
        result.append(class_id)
    if not result:
        raise ADNetContractError("at least one required class is needed")
    if len(set(result)) != len(result):
        raise ADNetContractError("required classes must be unique")
    return tuple(result)


if __name__ == "__main__":
    raise SystemExit(main())
