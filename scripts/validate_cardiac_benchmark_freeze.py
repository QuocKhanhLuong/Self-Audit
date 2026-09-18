#!/usr/bin/env python
"""Validate the immutable, GT-free cardiac benchmark freeze artifacts.

This validator deliberately validates only provenance/contracts.  It neither
opens reference annotations nor invokes a semantic adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE_DIR = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v1"


class FreezeValidationError(ValueError):
    """A freeze artifact does not match its immutable declared identity."""


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


def _safe_relative(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise FreezeValidationError(f"bound path must be relative and contained: {path!r}")
    return candidate


def _validate_hashes(root: Path, bindings: list[Mapping[str, Any]], field: str) -> None:
    for binding in bindings:
        relative = _safe_relative(str(binding["path"]))
        expected = str(binding["sha256"])
        actual = _sha256_file(root / relative)
        if actual != expected:
            raise FreezeValidationError(f"{field} hash mismatch: {relative}")


def _validate_fixture_set(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != "shared_benchmark.cardiac_adapter_fixture_set.v1":
        raise FreezeValidationError("unsupported adapter fixture schema")
    for fixture in value.get("fixtures", []):
        raw = fixture.get("partition_grid", [])
        semantic = fixture.get("expected_semantic_grid", [])
        validity = fixture.get("expected_validity_grid", [])
        if not raw or len(raw) != len(semantic) or len(raw) != len(validity):
            raise FreezeValidationError(f"invalid fixture dimensions: {fixture.get('id')}")
        ids = fixture.get("raw_id_by_symbol", {})
        for source, output, valid in zip(raw, semantic, validity):
            if len(source) != len(output) or len(source) != len(valid):
                raise FreezeValidationError(f"non-rectangular fixture: {fixture.get('id')}")
            if not set(source).issubset(ids):
                raise FreezeValidationError(f"undefined raw symbol: {fixture.get('id')}")
            if not set(output).issubset({"B", "R", "M", "L", "V"}) or not set(valid).issubset({"0", "1"}):
                raise FreezeValidationError(f"invalid expected fixture encoding: {fixture.get('id')}")


def validate_freeze(freeze_dir: str | Path = DEFAULT_FREEZE_DIR, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Validate core payload identity and every content-hash-bound artifact."""
    freeze = Path(freeze_dir)
    repo = Path(repo_root)
    manifest = _read_json(freeze / "FREEZE_MANIFEST.json")
    if manifest.get("freeze_schema") != "shared_benchmark.cardiac_benchmark_freeze.v1":
        raise FreezeValidationError("unsupported freeze schema")
    payload = manifest.get("scientific_payload")
    if not isinstance(payload, dict):
        raise FreezeValidationError("freeze scientific_payload is required")
    payload_hash = _sha256_bytes(_canonical_bytes(payload))
    if payload_hash != manifest.get("scientific_payload_sha256"):
        raise FreezeValidationError("scientific payload hash mismatch")
    expected_id = f"cardiac-benchmark-v1-{payload_hash[:16]}"
    if manifest.get("freeze_id") != expected_id:
        raise FreezeValidationError("freeze ID does not derive from the scientific payload")
    _validate_hashes(repo, list(payload.get("bound_repository_files", [])), "repository")
    _validate_hashes(freeze, list(payload.get("bound_freeze_files", [])), "freeze")
    datasets = payload.get("datasets", {})
    for dataset in ("acdc", "mnms"):
        status = datasets.get(dataset, {}).get("scientific_manifest_status")
        if status not in {"PASS", "DEFERRED"}:
            raise FreezeValidationError(f"invalid dataset freeze status: {dataset}")
        if status == "DEFERRED" and any(key in datasets[dataset] for key in ("shared_manifest_sha256", "patient_counts", "sample_counts")):
            raise FreezeValidationError(f"deferred dataset carries fabricated scientific counts: {dataset}")
    contract = _read_json(freeze / "configs" / "resolved_shared_contract.json")
    if contract.get("target_hw") != [224, 224] or contract.get("forward_values") != "masked_area_normalized_convolution":
        raise FreezeValidationError("shared spatial freeze is not the mandated 224 masked-area contract")
    spec = _read_json(freeze / "configs" / "adapter_v1_spec.json")
    if spec.get("adapter_version") != "cardiac_adapter_v1" or spec.get("scientific_properties", {}).get("region_splitting") is not False:
        raise FreezeValidationError("adapter freeze does not preserve v1 naming/merging-only boundary")
    _validate_fixture_set(_read_json(freeze / "configs" / "adapter_v1_synthetic_fixtures.json"))
    return {"freeze_id": expected_id, "scientific_payload_sha256": payload_hash, "datasets": datasets}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", default=str(DEFAULT_FREEZE_DIR))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args()
    result = validate_freeze(args.freeze_dir, args.repo_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
