#!/usr/bin/env python
"""Validate the active Self-Audit ACDC STEGO/PiCIE compat-224 v7 freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v7"
SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v7"
PREFIX = "cardiac-benchmark-v7"


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
        raise FreezeValidationError("unsupported v7 freeze schema")
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
        ADAPTER_V3_INPUT_SCHEMA_VERSION,
        ADAPTER_V3_OUTPUT_SCHEMA_VERSION,
        ADAPTER_V3_VERSION,
        FROZEN_ADAPTER_V3_SPEC_SHA256,
        load_and_validate_spec,
        semantic_artifact_stage,
    )
    from shared_benchmark.spatial import grid_hash, load_self_audit_compat_224_grid_spec

    spec = _read(freeze / "configs" / "adapter_v3_spec.json")
    source_spec = _read(repo / "configs" / "adapter_v3_spec_source.json")
    if spec != source_spec:
        raise FreezeValidationError("adapter-v3 specification does not reproduce")
    _, frozen_file_sha = load_and_validate_spec(spec)
    if frozen_file_sha != FROZEN_ADAPTER_V3_SPEC_SHA256 or _hash_file(freeze / "configs" / "adapter_v3_spec.json") != FROZEN_ADAPTER_V3_SPEC_SHA256:
        raise FreezeValidationError("adapter-v3 specification hash mismatch")
    if spec.get("adapter_version") != ADAPTER_V3_VERSION or spec.get("input_schema_version") != ADAPTER_V3_INPUT_SCHEMA_VERSION or spec.get("output_schema_version") != ADAPTER_V3_OUTPUT_SCHEMA_VERSION:
        raise FreezeValidationError("adapter-v3 version/schema mismatch")
    if spec.get("scientific_properties", {}).get("region_splitting") is not False:
        raise FreezeValidationError("adapter-v3 must remain merge-only")

    grid = _read(freeze / "configs" / "resolved_shared_contract.json")
    expected_grid = load_self_audit_compat_224_grid_spec(repo)
    if {key: value for key, value in grid.items() if key != "shared_grid_sha256"} != expected_grid or grid.get("shared_grid_sha256") != grid_hash(expected_grid):
        raise FreezeValidationError("compat-224 grid does not reproduce")
    if grid.get("target_hw") != [224, 224] or grid.get("whole_fov") is not True or grid.get("crop") is not None or grid.get("forward_values") != "bilinear_align_corners_false":
        raise FreezeValidationError("wrong compat-224 central-image grid")

    manifest = _read(freeze / "data" / "acdc_shared_manifest.json")
    try:
        validate_manifest(manifest)
    except Exception as exc:
        raise FreezeValidationError(f"frozen manifest invalid: {exc}") from exc
    counts = {split: sum(row.get("split") == split for row in manifest.get("records", [])) for split in ("train", "dev", "test")}
    if manifest.get("manifest_hash") != manifest_hash(manifest) or counts != {"train": 1526, "dev": 376, "test": 0} or any(row.get("split") == "test" for row in manifest.get("records", [])):
        raise FreezeValidationError("wrong Self-Audit train/dev-only cohort")
    if manifest.get("shared_grid_hash") != grid_hash(expected_grid) or manifest.get("shared_grid", {}).get("version") != expected_grid["version"]:
        raise FreezeValidationError("manifest does not bind the compat-224 grid")
    if any(row.get("shared_grid") != expected_grid or row.get("spatial_transform", {}).get("forward_resize", {}).get("output_hw") != [224, 224] for row in manifest.get("records", [])):
        raise FreezeValidationError("manifest record projection does not reproduce the compat-224 contract")

    shared_manifest = payload.get("shared_manifest", {})
    if shared_manifest.get("manifest_sha256") != manifest["manifest_hash"] or shared_manifest.get("manifest_copy_sha256") != _hash_file(freeze / "data" / "acdc_shared_manifest.json"):
        raise FreezeValidationError("shared manifest payload binding mismatch")
    adapter = payload.get("adapter", {})
    expected_contract = "Self-Audit full-volume 0.5/99.5 percentile clip + population z-score, then central plane whole-FOV bilinear resize 224x224 align_corners=False on the STEGO/PiCIE compat grid"
    if adapter.get("adapter_version") != ADAPTER_V3_VERSION or adapter.get("specification_sha256") != FROZEN_ADAPTER_V3_SPEC_SHA256 or adapter.get("central_image_contract") != expected_contract:
        raise FreezeValidationError("adapter-v3 publication mismatch")
    if adapter.get("intensity_resolution") != "disabled" or adapter.get("orientation_resolution") != "disabled":
        raise FreezeValidationError("adapter-v3 resolution contract mismatch")
    lineage = payload.get("semantic_lineage", {})
    if lineage.get("semantic_artifact_stage") != semantic_artifact_stage(ADAPTER_V3_VERSION) or lineage.get("historical_v2_semantic_stage") != "semantic-cardiac_adapter_v2" or lineage.get("overwrite_policy") != "forbidden_by_distinct_stage":
        raise FreezeValidationError("semantic artifact lineage mismatch")
    runner = payload.get("runner_binding", {})
    expected_v7 = "benchmark_freezes/cardiac_benchmark_v7/configs/adapter_v3_spec.json"
    expected_v6 = "benchmark_freezes/cardiac_benchmark_v6/configs/adapter_v2_spec.json"
    if runner.get("cuts_adapter_default") != expected_v6 or runner.get("dfc_adapter_default") != expected_v6:
        raise FreezeValidationError("legacy runner defaults do not preserve v6")
    if runner.get("stego_sa224_adapter_default") != expected_v7 or runner.get("picie_sa224_adapter_default") != expected_v7:
        raise FreezeValidationError("SA224 runner defaults do not bind v7")
    if runner.get("stego_sa224_fair_requires_checkpoint_contract") is not True or runner.get("picie_sa224_fair_requires_checkpoint_contract") is not True:
        raise FreezeValidationError("fair runner contract requirement mismatch")
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
