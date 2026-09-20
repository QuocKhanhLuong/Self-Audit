#!/usr/bin/env python
"""Validate the active adapter-v2 v5 static freeze without a data workload."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v5"
FREEZE_ID_PREFIX = "cardiac-benchmark-v5"
DEFAULT_FREEZE_DIR = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v5"


class FreezeValidationError(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
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


def _check_hashes(root: Path, bindings: list[Mapping[str, Any]], label: str) -> None:
    for binding in bindings:
        relative = Path(str(binding.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise FreezeValidationError(f"invalid bound path: {relative}")
        try:
            actual = _hash_file(root / relative)
        except OSError as exc:
            raise FreezeValidationError(f"{label} hash mismatch: {relative}") from exc
        if actual != binding.get("sha256"):
            raise FreezeValidationError(f"{label} hash mismatch: {relative}")


def validate_freeze(freeze_dir: str | Path = DEFAULT_FREEZE_DIR, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    freeze, repo = Path(freeze_dir), Path(repo_root)
    document = _read(freeze / "FREEZE_MANIFEST.json")
    payload = document.get("scientific_payload")
    if document.get("freeze_schema") != FREEZE_SCHEMA or not isinstance(payload, dict):
        raise FreezeValidationError("unsupported v5 freeze schema")
    digest = _hash_bytes(_canonical_bytes(payload))
    if document.get("scientific_payload_sha256") != digest:
        raise FreezeValidationError("scientific payload hash mismatch")
    if document.get("freeze_id") != f"{FREEZE_ID_PREFIX}-{digest[:16]}":
        raise FreezeValidationError("freeze ID does not derive from payload")
    _check_hashes(repo, list(payload.get("bound_repository_files", [])), "repository")
    _check_hashes(freeze, list(payload.get("bound_freeze_files", [])), "freeze")

    source = str(repo / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from shared_benchmark.manifest import manifest_hash, validate_manifest
    from shared_benchmark.semantic_contract import ADAPTER_VERSION, FROZEN_ADAPTER_SPEC_SHA256
    from shared_benchmark.spatial import grid_hash, load_self_audit_grid_spec

    spec = _read(freeze / "configs" / "adapter_v2_spec.json")
    if spec != _read(repo / "configs" / "adapter_v2_spec_source.json") or spec.get("adapter_version") != ADAPTER_VERSION or _hash_file(freeze / "configs" / "adapter_v2_spec.json") != FROZEN_ADAPTER_SPEC_SHA256:
        raise FreezeValidationError("adapter-v2 specification does not reproduce")
    if spec.get("scientific_properties", {}).get("region_splitting") is not False:
        raise FreezeValidationError("adapter-v2 region splitting is forbidden")
    contract = _read(freeze / "configs" / "resolved_shared_contract.json")
    grid = load_self_audit_grid_spec(repo)
    if {key: value for key, value in contract.items() if key != "shared_grid_sha256"} != grid or contract.get("shared_grid_sha256") != grid_hash(grid):
        raise FreezeValidationError("Self-Audit grid does not reproduce")
    if contract.get("target_hw") != [256, 256] or contract.get("crop") is not None or contract.get("whole_fov") is not True or contract.get("forward_values") != "bilinear_align_corners_false":
        raise FreezeValidationError("wrong Self-Audit central-image grid")
    manifest = _read(freeze / "data" / "acdc_shared_manifest.json")
    try:
        validate_manifest(manifest)
    except Exception as exc:
        raise FreezeValidationError(f"ACDC manifest invalid: {exc}") from exc
    counts = {split: sum(row.get("split") == split for row in manifest.get("records", [])) for split in ("train", "dev", "test")}
    if manifest.get("manifest_hash") != manifest_hash(manifest) or counts != {"train": 1526, "dev": 376, "test": 0} or any(row.get("split") == "test" for row in manifest.get("records", [])):
        raise FreezeValidationError("wrong Self-Audit train/dev-only manifest")
    if payload.get("shared_manifest", {}).get("manifest_sha256") != manifest["manifest_hash"]:
        raise FreezeValidationError("payload/manifest identity mismatch")
    adapter = payload.get("adapter", {})
    if adapter.get("central_image_contract") != "Self-Audit full-volume 0.5/99.5 percentile clip + population z-score, then central plane whole-FOV bilinear resize 256x256 align_corners=False" or adapter.get("intensity_resolution") != "disabled" or adapter.get("orientation_resolution") != "disabled":
        raise FreezeValidationError("central-image or resolution policy mismatch")
    runner = payload.get("runner_binding", {})
    expected_path = "benchmark_freezes/cardiac_benchmark_v5/configs/adapter_v2_spec.json"
    if runner.get("cuts_adapter_default") != expected_path or runner.get("dfc_adapter_default") != expected_path:
        raise FreezeValidationError("runner default does not bind active v5 adapter spec")
    return {"freeze_id": document["freeze_id"], "scientific_payload_sha256": digest, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(json.dumps(validate_freeze(args.freeze_dir, args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
