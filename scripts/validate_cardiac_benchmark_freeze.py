#!/usr/bin/env python
"""Validate the immutable, GT-free cardiac benchmark freeze artifacts.

This validator deliberately validates only provenance/contracts.  It neither
opens reference annotations nor invokes a semantic adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v3"
FREEZE_ID_PREFIX = "cardiac-benchmark-v3"
DEFAULT_FREEZE_DIR = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v3"
_FREEZE_PROFILES = {
    "shared_benchmark.cardiac_benchmark_freeze.v1": ("cardiac-benchmark-v1", "freemask"),
    "shared_benchmark.cardiac_benchmark_freeze.v2": ("cardiac-benchmark-v2", "freemask"),
    "shared_benchmark.cardiac_benchmark_freeze.v3": ("cardiac-benchmark-v3", "self_audit"),
}


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


def _reject_gt_shaped_fixture_fields(value: Any, path: str = "fixture") -> None:
    forbidden = (
        "gt", "ground_truth", "mask", "label", "annotation", "dice", "iou",
        "hausdorff", "oracle", "hungarian", "candidate_bank", "auditor",
        "predictive", "o_fit", "o_select", "o_verify",
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in forbidden):
                raise FreezeValidationError(f"forbidden GT-shaped fixture field: {path}.{key}")
            _reject_gt_shaped_fixture_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_gt_shaped_fixture_fields(child, f"{path}[{index}]")


def _validate_fixture_set(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != "shared_benchmark.cardiac_adapter_fixture_set.v1":
        raise FreezeValidationError("unsupported adapter fixture schema")
    fixtures = value.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise FreezeValidationError("fixture set must contain a non-empty list")
    fixture_ids: list[str] = []
    allowed_semantic = {"B", "R", "M", "L", "V"}
    for fixture in fixtures:
        if not isinstance(fixture, Mapping):
            raise FreezeValidationError("fixture entries must be objects")
        _reject_gt_shaped_fixture_fields(fixture, f"fixtures[{fixture.get('id', '?')}]")
        fixture_id = fixture.get("id")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise FreezeValidationError("fixture id must be a non-empty string")
        fixture_ids.append(fixture_id)
        raw = fixture.get("partition_grid", [])
        semantic = fixture.get("expected_semantic_grid", [])
        validity = fixture.get("expected_validity_grid", [])
        if not raw or len(raw) != len(semantic) or len(raw) != len(validity):
            raise FreezeValidationError(f"invalid fixture dimensions: {fixture_id}")
        ids = fixture.get("raw_id_by_symbol", {})
        if not isinstance(ids, Mapping) or not ids:
            raise FreezeValidationError(f"raw_id_by_symbol is required: {fixture_id}")
        for source, output, valid in zip(raw, semantic, validity):
            if len(source) != len(output) or len(source) != len(valid):
                raise FreezeValidationError(f"non-rectangular fixture: {fixture_id}")
            if not set(source).issubset(ids):
                raise FreezeValidationError(f"undefined raw symbol: {fixture_id}")
            if not set(output).issubset(allowed_semantic) or not set(valid).issubset({"0", "1"}):
                raise FreezeValidationError(f"invalid expected fixture encoding: {fixture_id}")
            for semantic_symbol, validity_symbol in zip(output, valid):
                expected_validity = "0" if semantic_symbol == "V" else "1"
                if validity_symbol != expected_validity:
                    raise FreezeValidationError(
                        f"semantic/validity invariant violated: {fixture_id} "
                        f"semantic={semantic_symbol!r} validity={validity_symbol!r}"
                    )
    if len(set(fixture_ids)) != len(fixture_ids):
        raise FreezeValidationError("fixture ids must be unique")
    if fixture_ids != sorted(fixture_ids):
        raise FreezeValidationError("fixture ids must use canonical lexicographic ordering")


def _generator_module(repo: Path):
    source = repo / "scripts" / "regenerate_cardiac_benchmark_freeze.py"
    spec = importlib.util.spec_from_file_location("cardiac_freeze_generator", source)
    if spec is None or spec.loader is None:
        raise FreezeValidationError("cannot load cardiac freeze generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_generated_fixture(repo: Path, fixture: Mapping[str, Any]) -> None:
    generator = _generator_module(repo)
    source = generator._read_json(repo / "configs" / "adapter_v1_synthetic_fixtures_source.json")
    expected = generator._validate_fixture_source(source)
    if dict(fixture) != expected:
        raise FreezeValidationError("frozen fixture does not reproduce from its declared source")


def _validate_generated_grid(repo: Path, contract: Mapping[str, Any], profile: str) -> None:
    source_root = str(repo / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from shared_benchmark.spatial import grid_hash, load_pinned_grid_spec, load_self_audit_grid_spec

    expected = load_self_audit_grid_spec(repo) if profile == "self_audit" else load_pinned_grid_spec(repo)
    stored = {
        key: value for key, value in dict(contract).items()
        if key not in {"shared_grid_sha256", "golden_fixture"}
    }
    if stored != expected:
        raise FreezeValidationError("resolved shared grid does not reproduce from current pinned sources")
    if contract.get("shared_grid_sha256") != grid_hash(expected):
        raise FreezeValidationError("resolved shared grid hash does not derive from current pinned sources")
    if not isinstance(contract.get("golden_fixture"), Mapping):
        raise FreezeValidationError("resolved shared grid is missing its golden fixture")


def validate_freeze(freeze_dir: str | Path = DEFAULT_FREEZE_DIR, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Validate core payload identity and every content-hash-bound artifact."""
    freeze = Path(freeze_dir)
    repo = Path(repo_root)
    manifest = _read_json(freeze / "FREEZE_MANIFEST.json")
    schema = str(manifest.get("freeze_schema", ""))
    if schema not in _FREEZE_PROFILES:
        raise FreezeValidationError("unsupported freeze schema")
    id_prefix, profile = _FREEZE_PROFILES[schema]
    payload = manifest.get("scientific_payload")
    if not isinstance(payload, dict):
        raise FreezeValidationError("freeze scientific_payload is required")
    payload_hash = _sha256_bytes(_canonical_bytes(payload))
    if payload_hash != manifest.get("scientific_payload_sha256"):
        raise FreezeValidationError("scientific payload hash mismatch")
    expected_id = f"{id_prefix}-{payload_hash[:16]}"
    if manifest.get("freeze_id") != expected_id:
        raise FreezeValidationError("freeze ID does not derive from the scientific payload")
    _validate_hashes(repo, list(payload.get("bound_repository_files", [])), "repository")
    _validate_hashes(freeze, list(payload.get("bound_freeze_files", [])), "freeze")
    datasets = payload.get("datasets", {})
    for dataset in ("acdc", "mnms"):
        status = datasets.get(dataset, {}).get("scientific_manifest_status")
        if status not in {"PASS", "PASS_STATIC", "DEFERRED"}:
            raise FreezeValidationError(f"invalid dataset freeze status: {dataset}")
        if status == "DEFERRED" and any(key in datasets[dataset] for key in ("shared_manifest_sha256", "patient_counts", "sample_counts")):
            raise FreezeValidationError(f"deferred dataset carries fabricated scientific counts: {dataset}")
    contract = _read_json(freeze / "configs" / "resolved_shared_contract.json")
    if profile == "self_audit":
        if (
            contract.get("target_hw") != [256, 256]
            or contract.get("forward_values") != "bilinear_align_corners_false"
            or contract.get("crop") is not None
        ):
            raise FreezeValidationError("Self-Audit spatial freeze is not the mandated 256 bilinear whole-FOV contract")
    elif contract.get("target_hw") != [224, 224] or contract.get("forward_values") != "masked_area_normalized_convolution":
        raise FreezeValidationError("historical FreeMask spatial freeze is not the mandated 224 masked-area contract")
    _validate_generated_grid(repo, contract, profile)
    spec = _read_json(freeze / "configs" / "adapter_v1_spec.json")
    if spec.get("adapter_version") != "cardiac_adapter_v1" or spec.get("scientific_properties", {}).get("region_splitting") is not False:
        raise FreezeValidationError("adapter freeze does not preserve v1 naming/merging-only boundary")
    fixture = _read_json(freeze / "configs" / "adapter_v1_synthetic_fixtures.json")
    _validate_fixture_set(fixture)
    _validate_generated_fixture(repo, fixture)
    if profile == "self_audit":
        manifest_path = freeze / "data" / "acdc_shared_manifest.json"
        selection_path = freeze / "data" / "acdc_selection_receipt.json"
        split_path = freeze / "data" / "acdc_patient_split_seed42.json"
        manifest_value = _read_json(manifest_path)
        if manifest_value.get("split_policy_version") != "self_audit.acdc.patient_split.v1":
            raise FreezeValidationError("v3 frozen manifest does not bind the Self-Audit patient split")
        source = str(repo / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from shared_benchmark.manifest import validate_manifest, manifest_hash
        try:
            validate_manifest(manifest_value)
        except Exception as exc:
            raise FreezeValidationError(f"v3 scientific manifest is invalid: {exc}") from exc
        if manifest_value.get("manifest_hash") != manifest_hash(manifest_value):
            raise FreezeValidationError("v3 scientific manifest hash does not reproduce")
        if any(row.get("split") == "test" for row in manifest_value.get("records", [])):
            raise FreezeValidationError("v3 scientific manifest unexpectedly contains a test cohort")
        selection = _read_json(selection_path)
        if selection.get("schema_version") != "self_audit.acdc.frame_selection.v1" or any(
            row.get("split") == "test" for row in selection.get("records", [])
        ):
            raise FreezeValidationError("v3 selection receipt is not train/dev-only Self-Audit protocol")
        split = _read_json(split_path)
        if split.get("seed") != 42 or len(split.get("train_patients", [])) != 80 or len(split.get("val_patients", [])) != 20:
            raise FreezeValidationError("v3 copied patient split does not reproduce 80/20 seed-42 membership")
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
