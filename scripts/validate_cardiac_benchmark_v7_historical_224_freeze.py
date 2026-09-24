#!/usr/bin/env python
"""Validate the immutable static v7 historical-224 CUTS/DFC baseline freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v7_historical_224"
SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v7_historical_224"
PREFIX = "cardiac-benchmark-v7-historical-224"


class FreezeValidationError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeValidationError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FreezeValidationError(f"JSON artifact must be an object: {path}")
    return value


def _validate_bindings(root: Path, values: list[Mapping[str, Any]], label: str) -> None:
    for value in values:
        relative = Path(str(value.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise FreezeValidationError(f"invalid bound path: {relative}")
        try:
            actual = _hash_file(root / relative)
        except OSError as exc:
            raise FreezeValidationError(f"{label} hash mismatch: {relative}") from exc
        if actual != value.get("sha256"):
            raise FreezeValidationError(f"{label} hash mismatch: {relative}")


def validate_freeze(freeze_dir: str | Path = FREEZE, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    freeze, repo = Path(freeze_dir), Path(repo_root)
    document = _read(freeze / "FREEZE_MANIFEST.json")
    payload = document.get("scientific_payload")
    if document.get("freeze_schema") != SCHEMA or not isinstance(payload, dict):
        raise FreezeValidationError("unsupported v7 historical-224 freeze schema")
    digest = _hash_bytes(_canonical(payload))
    if document.get("scientific_payload_sha256") != digest:
        raise FreezeValidationError("scientific payload hash mismatch")
    if document.get("freeze_id") != f"{PREFIX}-{digest[:16]}":
        raise FreezeValidationError("freeze ID does not derive from payload")
    _validate_bindings(repo, list(payload.get("bound_repository_files", [])), "repository")
    _validate_bindings(freeze, list(payload.get("bound_freeze_files", [])), "freeze")

    source = str(repo / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from shared_benchmark.manifest import manifest_hash, validate_manifest
    from shared_benchmark.semantic_contract import (
        ADAPTER_VERSION,
        FROZEN_ADAPTER_SPEC_SHA256,
        FROZEN_SELF_AUDIT_HISTORICAL_224_SHARED_GRID_SHA256,
    )
    from shared_benchmark.spatial import (
        SELF_AUDIT_HISTORICAL_224_SPATIAL_CONTRACT_VERSION,
        grid_hash,
        load_self_audit_historical_224_grid_spec,
    )

    grid = _read(freeze / "configs" / "resolved_shared_contract.json")
    expected_grid = load_self_audit_historical_224_grid_spec(repo)
    if (
        {key: value for key, value in grid.items() if key != "shared_grid_sha256"} != expected_grid
        or grid.get("shared_grid_sha256") != grid_hash(expected_grid)
        or grid_hash(expected_grid) != FROZEN_SELF_AUDIT_HISTORICAL_224_SHARED_GRID_SHA256
    ):
        raise FreezeValidationError("historical-224 grid does not reproduce")
    if (
        grid.get("version") != SELF_AUDIT_HISTORICAL_224_SPATIAL_CONTRACT_VERSION
        or grid.get("target_hw") != [224, 224]
        or grid.get("whole_fov") is not True
        or grid.get("crop") is not None
        or grid.get("forward_values") != "historical_preprocess_224_no_post_loader_resize"
    ):
        raise FreezeValidationError("wrong historical-224 central-image grid")
    manifest = _read(freeze / "data" / "acdc_shared_manifest.json")
    try:
        validate_manifest(manifest)
    except Exception as exc:
        raise FreezeValidationError(f"frozen manifest invalid: {exc}") from exc
    counts = {split: sum(row.get("split") == split for row in manifest.get("records", [])) for split in ("train", "dev", "test")}
    if (
        manifest.get("manifest_hash") != manifest_hash(manifest)
        or manifest.get("shared_grid") != expected_grid
        or manifest.get("shared_grid_hash") != grid_hash(expected_grid)
        or counts != {"train": 1526, "dev": 376, "test": 0}
        or any(row.get("split") == "test" for row in manifest.get("records", []))
    ):
        raise FreezeValidationError("wrong historical-224 Self-Audit train/dev-only cohort")
    spec = _read(freeze / "configs" / "adapter_v2_spec.json")
    if (
        spec != _read(repo / "configs" / "adapter_v2_spec_source.json")
        or spec.get("adapter_version") != ADAPTER_VERSION
        or _hash_file(freeze / "configs" / "adapter_v2_spec.json") != FROZEN_ADAPTER_SPEC_SHA256
    ):
        raise FreezeValidationError("adapter-v2 specification does not reproduce")
    adapter = payload.get("adapter", {})
    expected_contract = (
        "Self-Audit historical preprocess image branch: source 0.5/99.5 percentile clip + population z-score, "
        "per-slice skimage bilinear resize 224x224 (preserve_range=True, anti_aliasing=True, mode=reflect), "
        "then loader 0.5/99.5 percentile clip + population z-score; central plane at 224x224; no post-loader resize"
    )
    if adapter.get("central_image_contract") != expected_contract:
        raise FreezeValidationError("adapter historical central-image contract mismatch")
    lineage = payload.get("semantic_lineage", {})
    if (
        lineage.get("checkpoint_reuse") != "FORBIDDEN: v6 CUTS checkpoints bind the v6 manifest/grid and v6 input normalization"
        or lineage.get("raw_partition_reuse") != "FORBIDDEN: v6 CUTS/DFC raw partitions were generated from a different central-image contract"
    ):
        raise FreezeValidationError("v6 runtime artifact reuse was not fail-closed")
    expected_path = f"benchmark_freezes/{freeze.name}/configs/adapter_v2_spec.json"
    runner = payload.get("runner_binding", {})
    if runner.get("cuts_adapter_default") != expected_path or runner.get("dfc_adapter_default") != expected_path:
        raise FreezeValidationError("runner defaults do not bind v7 historical-224")
    return {"freeze_id": document["freeze_id"], "scientific_payload_sha256": digest, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, default=FREEZE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(json.dumps(validate_freeze(args.freeze_dir, args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
