"""Canonical projection of an authoritative image-only source manifest.

The projector accepts the historical FreeMask receipt for reproducibility and
the original Self-Audit ACDC receipt for current CUTS/DFC scientific runs. It
never computes a split itself.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .firewall import FirewallError, validate_image_only_manifest
from .provenance import canonical_json_bytes as _canonical_json_bytes, sha256_file, sha256_json
from .spatial import (
    SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION,
    SELF_AUDIT_SPATIAL_CONTRACT_VERSION,
    SPATIAL_CONTRACT_VERSION,
    grid_hash,
    spatial_transform_for_record,
)


MANIFEST_SCHEMA_VERSION = "shared_benchmark_manifest.v1"
SPLIT_POLICY_VERSION = "maskfree150.data.discovery.v2"
SELF_AUDIT_SPLIT_POLICY_VERSION = "self_audit.acdc.patient_split.v1"
SUPPORTED_SPLIT_POLICY_VERSIONS = frozenset({SPLIT_POLICY_VERSION, SELF_AUDIT_SPLIT_POLICY_VERSION})
SUPPORTED_SOURCE_MANIFEST_SCHEMAS = frozenset({"maskfree150.data.v2", "self_audit.acdc.image_only.v1"})
_SPLITS = {"train", "dev", "test"}


class SharedManifestError(ValueError):
    """Raised for a noncanonical, non-image-only, or unsafe projection."""


def canonical_json_bytes(value: Any) -> bytes:
    """Public canonical serialization for a shared scientific payload."""
    return _canonical_json_bytes(value)


def _canonical_sample_id(record: Mapping[str, Any]) -> str:
    dataset = str(record["dataset"])
    patient = str(record["patient_id"])
    study = str(record["study_id"])
    volume = str(record["volume_id"])
    frame = int(record["frame_index"])
    slice_index = int(record["slice_index"])
    return f"{dataset}:patient-{patient}:study-{study}:volume-{volume}:frame-{frame:04d}:z-{slice_index:04d}"


def _logical_upstream_payload(upstream: Mapping[str, Any]) -> dict[str, Any]:
    """Strip host aliases and normalize discovery-record order before hashing."""
    payload = _strip_local_paths(copy.deepcopy(dict(upstream)))
    records = payload.get("records")
    if isinstance(records, list):
        # Discovery's record order is deliberately not part of sample identity.
        # Keep its complete image-only evidence, but bind it in canonical order.
        payload["records"] = sorted(
            records,
            key=lambda row: (
                str(row.get("dataset", "")),
                str(row.get("patient_id", "")),
                str(row.get("study_id", "")),
                str(row.get("volume_id", "")),
                int(row.get("frame_index", -1)),
                int(row.get("slice_index", -1)),
            ),
        )
    return payload


def _strip_local_paths(value: Any) -> Any:
    """Drop absolute-root diagnostics while retaining relative source identity."""
    if isinstance(value, Mapping):
        return {
            key: _strip_local_paths(child)
            for key, child in value.items()
            if key not in {"root", "path", "source_path", "primary_source_path", "reference_source_path", "source_paths"}
        }
    if isinstance(value, list):
        return [_strip_local_paths(child) for child in value]
    return value


def _source_record(record: Mapping[str, Any], grid: Mapping[str, Any]) -> dict[str, Any]:
    depth = int(record["depth"])
    z = int(record["slice_index"])
    context = [max(0, z - 1), z, min(depth - 1, z + 1)]
    projection = {
        "dataset": str(record["dataset"]),
        "patient_id": str(record["patient_id"]),
        "study_id": str(record["study_id"]),
        "volume_id": str(record["volume_id"]),
        "sample_id": _canonical_sample_id(record),
        "split": str(record["split"]),
        "frame_index": int(record["frame_index"]),
        "frame_axis": record.get("frame_axis"),
        "slice_index": z,
        "depth": depth,
        "depth_axis": int(record["depth_axis"]),
        "context_indices": context,
        "frame_selection_rule": str(record["frame_selection_rule"]),
        "source": {
            "locator": str(record["relative_path"]).replace("\\", "/"),
            "sha256": str(record.get("source_hash", record.get("source_fingerprint", ""))),
            "frame_fingerprint": str(record.get("frame_fingerprint", "")),
            "format": str(record["source_format"]),
            "dtype": str(record["dtype"]),
        },
        "native_shape": [int(value) for value in record["native_shape"]],
        "stored_shape": [int(value) for value in record["shape"]],
        "native_hw": [int(value) for value in record["native_hw"]],
        "axis_semantics": {
            "native_axis_order": {0: "ZHW", 1: "HZW", 2: "HWZ"}[int(record["depth_axis"])],
            "depth_axis": int(record["depth_axis"]),
            "frame_axis": record.get("frame_axis"),
        },
        "geometry": {
            "spacing_mm": record.get("spacing_mm") if record.get("spacing_valid") else None,
            "spacing_valid": bool(record["spacing_valid"]),
            "native_affine": record.get("native_affine"),
            "affine_valid": record.get("native_geometry") == "available",
            "orientation": record.get("orientation"),
            "orientation_valid": record.get("orientation") is not None,
            "native_grid_export": bool(record["native_grid_export"]),
            "export_grid": str(record["export_grid"]),
            "spatial_unit": record.get("spatial_unit"),
            "study_grid_compatibility": _strip_local_paths(record.get("study_grid_compatibility")),
        },
        "shared_grid": copy.deepcopy(dict(grid)),
    }
    projection["spatial_transform"] = spatial_transform_for_record(projection, grid)
    return projection


def _scientific_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(manifest))
    payload.pop("manifest_hash", None)
    payload.pop("local_receipt", None)
    return payload


def _split_provenance(upstream: Mapping[str, Any]) -> dict[str, Any]:
    """Keep split evidence while omitting annotation-named implementation fields.

    The source manifest's ``annotation_inputs=[]`` is useful internally, but a
    shared generation artifact must expose no annotation-shaped API surface.
    The projector verifies no split logic itself; it records only the resulting
    image-only policy facts.
    """
    source = upstream.get("split_provenance", {})
    # The historical source has the FreeMask split fields above; the original
    # Self-Audit source carries an explicit cohort and ED/ES selection rule.
    # Preserve both verbatim as provenance.  This function never computes or
    # changes membership.
    return copy.deepcopy(dict(source)) if isinstance(source, Mapping) else {}


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    return sha256_json(_scientific_payload(manifest))


def build_shared_manifest(
    upstream: Mapping[str, Any],
    grid: Mapping[str, Any],
    *,
    fixture: bool,
    scientific: bool,
    local_source_root: str | Path | None = None,
    upstream_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Project a completed source discovery result without re-splitting it."""
    if fixture == scientific:
        raise SharedManifestError("exactly one of fixture or scientific must be true")
    if upstream.get("schema_version") not in SUPPORTED_SOURCE_MANIFEST_SCHEMAS:
        raise SharedManifestError(
            "projection requires a supported authoritative source manifest: "
            f"{sorted(SUPPORTED_SOURCE_MANIFEST_SCHEMAS)}"
        )
    if int(upstream.get("seed", -1)) != 42:
        raise SharedManifestError("shared benchmark requires the authoritative split seed 42")
    if grid.get("version") not in {
        SPATIAL_CONTRACT_VERSION,
        SELF_AUDIT_SPATIAL_CONTRACT_VERSION,
        SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION,
    }:
        raise SharedManifestError("unsupported shared spatial contract")
    records = [_source_record(record, grid) for record in upstream.get("records", [])]
    records.sort(key=lambda record: record["sample_id"])
    if not records:
        raise SharedManifestError("upstream manifest has no generation records")
    if len({record["sample_id"] for record in records}) != len(records):
        raise SharedManifestError("canonical sample_id collision")
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "fixture": bool(fixture),
        "scientific": bool(scientific),
        "dataset": str(upstream["dataset"]),
        "split_seed": 42,
        "split_policy_version": str(
            upstream.get("split_provenance", {}).get("policy_version", SPLIT_POLICY_VERSION)
        ),
        "sample_order": "sample_id_lexicographic",
        "source_manifest": {
            "schema_version": str(upstream["schema_version"]),
            "manifest_id": str(upstream["manifest_id"]),
            "logical_sha256": sha256_json(_logical_upstream_payload(upstream)),
            "discovery_contract": copy.deepcopy(upstream.get("discovery_contract")),
            "split_provenance": _split_provenance(upstream),
        },
        "shared_grid": copy.deepcopy(dict(grid)),
        "shared_grid_hash": grid_hash(grid),
        "context_policy": "z_minus_1_z_z_plus_1_endpoint_replicated",
        "generation_membership": str(
            upstream.get(
                "generation_membership",
                "all_authoritatively_discovered_frames_x_all_z_slices",
            )
        ),
        "records": records,
    }
    if local_source_root is not None or upstream_file_sha256 is not None:
        payload["local_receipt"] = {
            "local_source_root": None if local_source_root is None else str(Path(local_source_root)),
            "upstream_file_sha256": upstream_file_sha256,
        }
    payload["manifest_hash"] = manifest_hash(payload)
    validate_manifest(payload)
    return payload


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SharedManifestError("unsupported shared manifest schema")
    if not isinstance(manifest.get("fixture"), bool) or not isinstance(manifest.get("scientific"), bool):
        raise SharedManifestError("fixture and scientific flags must be booleans")
    if bool(manifest["fixture"]) == bool(manifest["scientific"]):
        raise SharedManifestError("exactly one fixture/scientific status is required")
    # Check the image-only boundary before integrity: an injected GT-shaped
    # field must never be reported merely as a generic hash mismatch.
    try:
        validate_image_only_manifest(manifest)
    except FirewallError as exc:
        raise SharedManifestError(str(exc)) from exc
    if manifest.get("split_seed") != 42 or manifest.get("split_policy_version") not in SUPPORTED_SPLIT_POLICY_VERSIONS:
        raise SharedManifestError("shared split provenance is not a supported frozen policy")
    if manifest.get("sample_order") != "sample_id_lexicographic":
        raise SharedManifestError("unknown shared sample ordering")
    grid = manifest.get("shared_grid")
    if not isinstance(grid, Mapping) or grid.get("version") not in {
        SPATIAL_CONTRACT_VERSION,
        SELF_AUDIT_SPATIAL_CONTRACT_VERSION,
        SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION,
    }:
        raise SharedManifestError("shared grid contract is missing or unsupported")
    if manifest.get("shared_grid_hash") != grid_hash(grid):
        raise SharedManifestError("shared grid hash mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise SharedManifestError("shared manifest records must be nonempty")
    patient_splits: dict[str, str] = {}
    sample_ids: list[str] = []
    for record in records:
        required = {
            "dataset", "patient_id", "study_id", "volume_id", "sample_id", "split", "frame_index",
            "slice_index", "depth", "depth_axis", "context_indices", "source", "native_shape",
            "stored_shape", "native_hw", "axis_semantics", "geometry", "shared_grid", "spatial_transform",
        }
        missing = required - set(record)
        if missing:
            raise SharedManifestError(f"shared record missing {sorted(missing)}")
        if record["split"] not in _SPLITS:
            raise SharedManifestError("invalid split")
        prior = patient_splits.setdefault(record["patient_id"], record["split"])
        if prior != record["split"]:
            raise SharedManifestError("patient appears in multiple splits")
        z, depth = int(record["slice_index"]), int(record["depth"])
        expected_context = [max(0, z - 1), z, min(depth - 1, z + 1)]
        if record["context_indices"] != expected_context:
            raise SharedManifestError("record violates endpoint-replicated context policy")
        if record["shared_grid"] != grid:
            raise SharedManifestError("record reinterprets shared grid")
        if not record["source"].get("sha256"):
            raise SharedManifestError("record lacks source image content hash")
        sample_ids.append(record["sample_id"])
    if sample_ids != sorted(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise SharedManifestError("records must have unique lexicographically ordered sample IDs")
    if manifest.get("manifest_hash") != manifest_hash(manifest):
        raise SharedManifestError("scientific manifest payload hash mismatch")


def validate_scientific_manifest(
    manifest: Mapping[str, Any], *, image_root: str | Path, verify_source_hashes: bool = True
) -> None:
    """Validate a non-fixture projection against a separately mounted image-only root."""
    validate_manifest(manifest)
    if manifest["fixture"] or not manifest["scientific"]:
        raise SharedManifestError("fixture manifest cannot validate as scientific")
    root = Path(image_root)
    if not root.is_dir():
        raise SharedManifestError("scientific image-only root is unavailable")
    for record in manifest["records"]:
        path = root / record["source"]["locator"]
        if not path.is_file():
            raise SharedManifestError(f"scientific source missing: {record['sample_id']}")
        if verify_source_hashes and sha256_file(path) != record["source"]["sha256"]:
            raise SharedManifestError(f"scientific source hash mismatch: {record['sample_id']}")


def write_shared_manifest(manifest: Mapping[str, Any], path: str | Path) -> str:
    validate_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(dict(manifest)) + b"\n")
    return str(manifest["manifest_hash"])


def load_shared_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload
