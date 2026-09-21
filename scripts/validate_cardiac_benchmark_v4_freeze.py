#!/usr/bin/env python
"""Validate the immutable, static Self-Audit ACDC adapter-v2/v4 freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v4"
FREEZE_ID_PREFIX = "cardiac-benchmark-v4"
DEFAULT_FREEZE_DIR = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v4"


class FreezeValidationError(ValueError):
    """The v4 freeze is incomplete, altered, or no longer reproducible."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeValidationError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FreezeValidationError(f"JSON artifact must be an object: {path}")
    return value


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise FreezeValidationError(f"bound path must be relative and contained: {value!r}")
    return path


def _validate_hashes(root: Path, bindings: list[Mapping[str, Any]], label: str) -> None:
    for binding in bindings:
        relative = _safe_relative(str(binding.get("path", "")))
        expected = str(binding.get("sha256", ""))
        try:
            actual = _sha256_file(root / relative)
        except OSError as exc:
            raise FreezeValidationError(f"{label} hash mismatch: {relative}") from exc
        if actual != expected:
            raise FreezeValidationError(f"{label} hash mismatch: {relative}")


def _counts(manifest: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    rows = list(manifest.get("records", []))
    return {
        split: {
            "patients": len({str(row.get("patient_id")) for row in rows if row.get("split") == split}),
            "volumes": len({str(row.get("volume_id")) for row in rows if row.get("split") == split}),
            "samples": sum(row.get("split") == split for row in rows),
        }
        for split in ("train", "dev", "test")
    }


def validate_freeze(freeze_dir: str | Path = DEFAULT_FREEZE_DIR, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    freeze = Path(freeze_dir)
    repo = Path(repo_root)
    document = _read_json(freeze / "FREEZE_MANIFEST.json")
    payload = document.get("scientific_payload")
    if document.get("freeze_schema") != FREEZE_SCHEMA or not isinstance(payload, dict):
        raise FreezeValidationError("unsupported v4 freeze schema")
    payload_hash = _sha256_bytes(_canonical_bytes(payload))
    if document.get("scientific_payload_sha256") != payload_hash:
        raise FreezeValidationError("scientific payload hash mismatch")
    if document.get("freeze_id") != f"{FREEZE_ID_PREFIX}-{payload_hash[:16]}":
        raise FreezeValidationError("freeze ID does not derive from the scientific payload")
    _validate_hashes(repo, list(payload.get("bound_repository_files", [])), "repository")
    _validate_hashes(freeze, list(payload.get("bound_freeze_files", [])), "freeze")

    source = str(repo / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from shared_benchmark.manifest import manifest_hash, validate_manifest
    from shared_benchmark.semantic_contract import (
        ADAPTER_INPUT_SCHEMA_VERSION, ADAPTER_OUTPUT_SCHEMA_VERSION, ADAPTER_VERSION,
        FROZEN_ADAPTER_SPEC_SHA256,
    )
    from shared_benchmark.spatial import grid_hash, load_self_audit_grid_spec

    spec = _read_json(freeze / "configs" / "adapter_v2_spec.json")
    source_spec = _read_json(repo / "configs" / "adapter_v2_spec_source.json")
    if spec != source_spec or _sha256_file(freeze / "configs" / "adapter_v2_spec.json") != FROZEN_ADAPTER_SPEC_SHA256:
        raise FreezeValidationError("adapter-v2 specification does not reproduce from source")
    if (spec.get("adapter_version"), spec.get("input_schema_version"), spec.get("output_schema_version")) != (
        ADAPTER_VERSION, ADAPTER_INPUT_SCHEMA_VERSION, ADAPTER_OUTPUT_SCHEMA_VERSION,
    ):
        raise FreezeValidationError("adapter-v2 specification/schema mismatch")
    if spec.get("scientific_properties", {}).get("region_splitting") is not False:
        raise FreezeValidationError("adapter-v2 may not split regions")

    contract = _read_json(freeze / "configs" / "resolved_shared_contract.json")
    expected_grid = load_self_audit_grid_spec(repo)
    stored_grid = {key: value for key, value in contract.items() if key != "shared_grid_sha256"}
    if stored_grid != expected_grid or contract.get("shared_grid_sha256") != grid_hash(expected_grid):
        raise FreezeValidationError("resolved Self-Audit grid does not reproduce")
    if contract.get("target_hw") != [256, 256] or contract.get("whole_fov") is not True or contract.get("crop") is not None or contract.get("forward_values") != "bilinear_align_corners_false":
        raise FreezeValidationError("v4 grid is not 256 whole-FOV bilinear align_corners=False")

    manifest = _read_json(freeze / "data" / "acdc_shared_manifest.json")
    try:
        validate_manifest(manifest)
    except Exception as exc:
        raise FreezeValidationError(f"frozen ACDC manifest is invalid: {exc}") from exc
    if manifest.get("manifest_hash") != manifest_hash(manifest):
        raise FreezeValidationError("frozen ACDC manifest hash does not reproduce")
    counts = _counts(manifest)
    expected_counts = {
        "train": {"patients": 80, "volumes": 160, "samples": 1526},
        "dev": {"patients": 20, "volumes": 40, "samples": 376},
        "test": {"patients": 0, "volumes": 0, "samples": 0},
    }
    if counts != expected_counts or any(row.get("split") == "test" for row in manifest.get("records", [])):
        raise FreezeValidationError(f"v4 manifest violates Self-Audit train/dev-only cohort: {counts}")
    if payload.get("shared_manifest", {}).get("sample_counts") != {key: value["samples"] for key, value in counts.items()}:
        raise FreezeValidationError("payload sample counts do not bind frozen manifest")
    if payload.get("shared_manifest", {}).get("manifest_sha256") != manifest["manifest_hash"]:
        raise FreezeValidationError("payload manifest identity mismatch")
    selection = _read_json(freeze / "data" / "acdc_selection_receipt.json")
    if selection.get("schema_version") != "self_audit.acdc.frame_selection.v1" or any(row.get("split") == "test" for row in selection.get("records", [])):
        raise FreezeValidationError("v4 selection receipt is not Self-Audit train/dev-only")
    split = _read_json(freeze / "data" / "acdc_patient_split_seed42.json")
    if split.get("seed") != 42 or len(split.get("train_patients", [])) != 80 or len(split.get("val_patients", [])) != 20:
        raise FreezeValidationError("v4 patient split is not seed-42 80/20")

    parent = _read_json(repo / "benchmark_freezes" / "cardiac_benchmark_v3" / "FREEZE_MANIFEST.json")
    parent_ref = payload.get("parent_freeze", {})
    if parent_ref.get("freeze_id") != parent.get("freeze_id") or parent_ref.get("scientific_payload_sha256") != parent.get("scientific_payload_sha256"):
        raise FreezeValidationError("v4 parent v3 provenance mismatch")
    adapter = payload.get("adapter", {})
    if adapter.get("central_image_contract") != "Self-Audit full-volume 0.5/99.5 percentile clip + population z-score, then central plane whole-FOV bilinear resize 256x256 align_corners=False":
        raise FreezeValidationError("v4 central-image contract mismatch")
    if adapter.get("intensity_resolution") != "disabled" or adapter.get("orientation_resolution") != "disabled":
        raise FreezeValidationError("v4 adapter unexpectedly enables intensity/orientation resolution")
    return {"freeze_id": document["freeze_id"], "scientific_payload_sha256": payload_hash, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(json.dumps(validate_freeze(args.freeze_dir, args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
