"""Shared-manifest loader for STEGO cardiac benchmark.

Reuses the same frozen manifest schema as CUTS so that both baselines
consume identical cohort manifests produced by FreeMask discovery.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "cuts.cardiac.shared-manifest.v1"
SPLITS = ("train", "dev", "test")
SPLIT_SEED = 42


class ManifestError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _without_hash(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "manifest_hash"}


def _is_forbidden_key(key: str) -> bool:
    key = key.lower()
    return any(token in key for token in ("mask", "label", "annotation", "foreground_hint", "reference_path"))


def _assert_image_only(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _is_forbidden_key(str(key)):
                raise ManifestError(f"generation manifest contains forbidden key {key!r}")
            _assert_image_only(child)
    elif isinstance(value, list):
        for child in value:
            _assert_image_only(child)


def _validate_record(record: dict[str, Any]) -> None:
    required = {
        "dataset", "patient_id", "split", "sample_id", "acquisition_id", "source_path",
        "image_checksum", "slice_index", "context_indices", "native_shape", "spacing",
        "manifest_hash",
    }
    missing = required - set(record)
    if missing:
        raise ManifestError(f"record {record.get('sample_id', '<unknown>')} missing {sorted(missing)}")
    if record["split"] not in SPLITS:
        raise ManifestError(f"invalid split {record['split']!r}")
    if len(record["context_indices"]) != 3:
        raise ManifestError("context_indices must be [z-1,z,z+1]")
    _assert_image_only(record)


def _hash_payload_without_derived_record_hash(payload: dict[str, Any]) -> str:
    value = _without_hash(payload)
    value = dict(value)
    value["records"] = [{k: v for k, v in record.items() if k != "manifest_hash"} for record in value["records"]]
    return sha256_json(value)


def freeze_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload["schema_version"] = MANIFEST_SCHEMA_VERSION
    payload["split_seed"] = SPLIT_SEED
    payload["frozen"] = True
    payload["manifest_hash"] = sha256_json(_without_hash(payload))
    for record in payload.get("records", []):
        record["manifest_hash"] = payload["manifest_hash"]
    canonical_payload = dict(payload)
    canonical_payload["records"] = [
        {k: v for k, v in record.items() if k != "manifest_hash"} for record in payload.get("records", [])
    ]
    payload["manifest_hash"] = sha256_json(_without_hash(canonical_payload))
    for record in payload.get("records", []):
        record["manifest_hash"] = payload["manifest_hash"]
    return payload


def write_manifest(payload: dict[str, Any], path: str | Path) -> str:
    payload = freeze_manifest(payload)
    payload["manifest_hash"] = _hash_payload_without_derived_record_hash(payload)
    for record in payload["records"]:
        record["manifest_hash"] = payload["manifest_hash"]
    validate_manifest(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json(payload) + b"\n")
    return payload["manifest_hash"]


def validate_manifest(payload: dict[str, Any], *, check_paths: bool = False) -> None:
    required = {"schema_version", "manifest_kind", "frozen", "dataset", "split_seed",
                "freemask_source_sha", "freemask_discovery_contract", "image_roots",
                "records", "manifest_hash"}
    if required - set(payload):
        raise ManifestError(f"manifest missing {sorted(required - set(payload))}")
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION or payload["split_seed"] != SPLIT_SEED or not payload["frozen"]:
        raise ManifestError("invalid frozen shared manifest contract")
    if payload["manifest_hash"] != _hash_payload_without_derived_record_hash(payload):
        raise ManifestError("manifest hash mismatch")
    sample_ids: set[str] = set()
    patient_splits: dict[str, str] = {}
    for record in payload["records"]:
        _validate_record(record)
        if record["manifest_hash"] != payload["manifest_hash"]:
            raise ManifestError("record manifest hash mismatch")
        if record["dataset"].lower() != payload["dataset"].lower():
            raise ManifestError("record dataset differs from manifest dataset")
        if record["sample_id"] in sample_ids:
            raise ManifestError("duplicate sample_id")
        sample_ids.add(record["sample_id"])
        prior = patient_splits.setdefault(record["patient_id"], record["split"])
        if prior != record["split"]:
            raise ManifestError("patient crosses splits")
        if check_paths and not Path(record["source_path"]).is_file():
            raise ManifestError(f"image source does not exist: {record['source_path']}")
    if not payload["records"]:
        raise ManifestError("manifest cannot be empty")


def load_manifest(path: str | Path, *, check_paths: bool = False) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(payload, check_paths=check_paths)
    return payload


def require_scientific_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_manifest(path, check_paths=True)
    if manifest["manifest_kind"] != "scientific":
        raise ManifestError("scientific_run=true rejects fixture/mock manifests")
    roots = [Path(root) for root in manifest["image_roots"]]
    if not roots or any(not root.is_dir() for root in roots):
        raise ManifestError("scientific_run=true requires existing image roots")
    return manifest


def counts_by_split(manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    answer: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        selected = [record for record in manifest["records"] if record["split"] == split]
        answer[split] = {"patients": len({r["patient_id"] for r in selected}), "samples": len(selected)}
    return answer
