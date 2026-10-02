#!/usr/bin/env python3
"""Prepare ADNet few-shot query/support/GT manifests from an asset spec."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shared_benchmark.adnet_fewshot import (  # noqa: E402
    ADNetContractError,
    atomic_write_json,
    file_sha256,
    load_gt_manifest,
    load_query_manifest,
    load_support_manifest,
)
from shared_benchmark.provenance import sha256_json  # noqa: E402


SPEC_SCHEMA = "adnet.asset_spec.v1"
DEFAULT_CLASS_MAPPING = {"RV": 1, "MYO": 2, "LV": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--path-style", choices=("absolute", "relative-to-asset-root"), default="absolute")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = prepare_manifests(
        spec_path=args.spec,
        output_dir=args.output_dir,
        asset_root=args.asset_root,
        path_style=args.path_style,
        force=args.force,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def prepare_manifests(
    *,
    spec_path: Path,
    output_dir: Path,
    asset_root: Path | None = None,
    path_style: str = "absolute",
    force: bool = False,
) -> dict[str, Any]:
    spec_location = spec_path.resolve()
    spec = _read_spec(spec_location)
    root = _asset_root(spec_location, asset_root)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if path_style == "relative-to-asset-root" and asset_root is None:
        raise ADNetContractError("relative-to-asset-root requires --asset-root")
    if path_style not in {"absolute", "relative-to-asset-root"}:
        raise ADNetContractError("unsupported ADNet manifest path style")

    query_manifest_path = output_dir / "query_manifest.json"
    support_manifest_path = output_dir / "support_manifest.json"
    gt_manifest_path = output_dir / "gt_manifest.json"
    receipt_path = output_dir / "prepare_receipt.json"
    targets = [query_manifest_path, support_manifest_path, receipt_path]
    if "gt" in spec:
        targets.append(gt_manifest_path)
    if not force:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise ADNetContractError("refusing to overwrite ADNet manifests: " + ", ".join(existing))

    class_mapping = _class_mapping(spec.get("class_mapping", DEFAULT_CLASS_MAPPING))
    query_manifest = {
        "schema": "adnet.query_manifest.v1",
        "records": [
            _query_record(row, root, path_style=path_style)
            for row in _required_list(spec, "queries")
        ],
    }
    support_manifest = {
        "schema": "adnet.support_manifest.v1",
        "class_mapping": class_mapping,
        "supports": [
            _support_record(row, root, class_mapping, path_style=path_style)
            for row in _required_list(spec, "supports")
        ],
    }
    atomic_write_json(query_manifest_path, query_manifest)
    atomic_write_json(support_manifest_path, support_manifest)

    gt_manifest = None
    if "gt" in spec:
        gt_manifest = {
            "schema": "adnet.gt_manifest.v1",
            "records": [
                _gt_record(row, root, path_style=path_style)
                for row in _required_list(spec, "gt")
            ],
        }
        atomic_write_json(gt_manifest_path, gt_manifest)

    manifest_root = root if path_style == "relative-to-asset-root" else None
    query = load_query_manifest(query_manifest_path, image_root=manifest_root)
    support = load_support_manifest(support_manifest_path, asset_root=manifest_root)
    gt = load_gt_manifest(gt_manifest_path, gt_root=manifest_root) if gt_manifest is not None else None

    receipt = {
        "schema": "adnet.prepare_receipt.v1",
        "asset_spec_path": str(spec_location),
        "asset_spec_sha256": file_sha256(spec_location),
        "asset_spec_scientific_sha256": sha256_json(spec),
        "asset_root": str(root),
        "path_style": path_style,
        "query_manifest": str(query_manifest_path),
        "query_manifest_sha256": query.sha256,
        "support_manifest": str(support_manifest_path),
        "support_manifest_sha256": support.sha256,
        "query_count": len(query.records),
        "support_count": len(support.supports),
        "class_mapping": support.class_mapping,
    }
    if gt is not None:
        receipt.update({
            "gt_manifest": str(gt_manifest_path),
            "gt_manifest_sha256": gt.sha256,
            "gt_count": len(gt.records),
        })
    atomic_write_json(receipt_path, receipt)
    return receipt


def _read_spec(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ADNetContractError(f"cannot read ADNet asset spec: {path}") from exc
    if not isinstance(value, dict):
        raise ADNetContractError("ADNet asset spec must be a JSON object")
    if value.get("schema") != SPEC_SCHEMA:
        raise ADNetContractError("unknown ADNet asset spec schema")
    return value


def _asset_root(spec_path: Path, asset_root: Path | None) -> Path:
    root = asset_root if asset_root is not None else spec_path.parent
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ADNetContractError(f"ADNet asset root is missing: {resolved}")
    return resolved


def _required_list(spec: Mapping[str, Any], key: str) -> list[Any]:
    value = spec.get(key)
    if not isinstance(value, list) or not value:
        raise ADNetContractError(f"ADNet asset spec requires non-empty {key}")
    return value


def _class_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ADNetContractError("ADNet class_mapping must be a non-empty object")
    result = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ADNetContractError("ADNet class names must be non-empty strings")
        if isinstance(raw, bool) or not isinstance(raw, int) or int(raw) < 1 or int(raw) > 3:
            raise ADNetContractError("ADNet foreground class ids must be 1,2,3")
        result[key] = int(raw)
    return dict(sorted(result.items()))


def _query_record(row: Any, root: Path, *, path_style: str) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ADNetContractError("ADNet query spec rows must be objects")
    sample_id = _string(row, "sample_id")
    image = _resolve_asset(_string(row, "image_path"), root)
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ADNetContractError("ADNet query metadata must be an object")
    return {
        "sample_id": sample_id,
        "image_path": _emit_path(image, root, path_style=path_style),
        "image_sha256": file_sha256(image),
        "split": str(row.get("split", "dev")),
        "metadata": dict(metadata),
    }


def _support_record(row: Any, root: Path, class_mapping: Mapping[str, int], *, path_style: str) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ADNetContractError("ADNet support spec rows must be objects")
    class_name = _string(row, "class_name")
    canonical_class_id = _int(row, "canonical_class_id", minimum=1, maximum=3)
    if class_mapping.get(class_name) != canonical_class_id:
        raise ADNetContractError("ADNet support class disagrees with class_mapping")
    image = _resolve_asset(_string(row, "image_path"), root)
    mask = _resolve_asset(_string(row, "mask_path"), root)
    raw_slices = row.get("slice_indices")
    if not isinstance(raw_slices, list) or not raw_slices:
        raise ADNetContractError("ADNet support slice_indices must be a non-empty list")
    return {
        "support_id": _string(row, "support_id"),
        "class_name": class_name,
        "canonical_class_id": canonical_class_id,
        "image_path": _emit_path(image, root, path_style=path_style),
        "image_sha256": file_sha256(image),
        "mask_path": _emit_path(mask, root, path_style=path_style),
        "mask_sha256": file_sha256(mask),
        "slice_indices": [_int_value(value, "slice_indices", minimum=0) for value in raw_slices],
    }


def _gt_record(row: Any, root: Path, *, path_style: str) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ADNetContractError("ADNet GT spec rows must be objects")
    gt = _resolve_asset(_string(row, "gt_path"), root)
    return {
        "sample_id": _string(row, "sample_id"),
        "gt_path": _emit_path(gt, root, path_style=path_style),
        "gt_sha256": file_sha256(gt),
    }


def _resolve_asset(raw: str, root: Path) -> Path:
    path = Path(raw)
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ADNetContractError(f"ADNet asset path is not readable: {raw}") from exc
    if not resolved.is_file():
        raise ADNetContractError(f"ADNet asset path is not a file: {raw}")
    return resolved


def _emit_path(path: Path, root: Path, *, path_style: str) -> str:
    if path_style == "absolute":
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError as exc:
        raise ADNetContractError(f"ADNet asset path escapes asset root: {path}") from exc


def _string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ADNetContractError(f"ADNet {key} must be a non-empty string")
    return value


def _int(row: Mapping[str, Any], key: str, *, minimum: int, maximum: int) -> int:
    return _int_value(row.get(key), key, minimum=minimum, maximum=maximum)


def _int_value(value: Any, key: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ADNetContractError(f"ADNet {key} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ADNetContractError(f"ADNet {key} is outside the allowed range")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
