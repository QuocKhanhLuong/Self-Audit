"""Crash-safe, GT-free artifacts for anonymous benchmark execution.

This module owns the boundary between an anonymous baseline and the frozen
semantic adapter.  It deliberately contains no baseline model code and no
semantic rules.  A raw artifact is committed first; only a verified
``RAW_COMPLETE`` artifact may be handed to :mod:`shared_benchmark.adapter`.

The scientific payloads in this module never contain absolute input roots,
timestamps, host names, or GT-shaped fields.  Those values may be recorded in
an external execution receipt by a caller, but cannot affect cache identity.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .adapter import adapt_partition
from .firewall import FirewallError, validate_image_only_source_locator, validate_image_only_value
from .manifest import SharedManifestError, load_shared_manifest, validate_scientific_manifest
from .provenance import canonical_json_bytes, sha256_bytes, sha256_file, sha256_json
from .semantic_contract import (
    ADAPTER_VERSION,
    FROZEN_ADAPTER_SPEC_SHA256,
    FROZEN_ADAPTER_V3_SPEC_SHA256,
    FROZEN_ADAPTER_V4_SPEC_SHA256,
    FROZEN_ADAPTER_V5_SPEC_SHA256,
    AdapterContractError,
    adapter_metadata_payload,
    array_hash,
    load_and_validate_spec,
    semantic_artifact_stage,
    validate_adapter_metadata,
)
from .spatial import grid_hash


RAW_SCHEMA_VERSION = "cardiac_raw_partition.v1"
SEMANTIC_SCHEMA_VERSION = "cardiac_semantic_partition.v2"
SEMANTIC_ARTIFACT_STAGE = semantic_artifact_stage(ADAPTER_VERSION)
STATE_SCHEMA_VERSION = "cardiac_execution_state.v1"

PENDING = "PENDING"
RUNNING = "RUNNING"
RAW_COMPLETE = "RAW_COMPLETE"
SEMANTIC_COMPLETE = "SEMANTIC_COMPLETE"
FAILED = "FAILED"

_TERMINAL_STAGES = {RAW_COMPLETE, SEMANTIC_COMPLETE, FAILED}
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_FORBIDDEN_METADATA_KEYS = {
    "gt", "groundtruth", "ground_truth", "mask", "masks", "label", "labels",
    "annotation", "annotations", "dice", "iou", "hausdorff", "oracle",
    "hungarian", "candidate_bank", "auditor", "predictive_evidence",
    "o_fit", "o_select", "o_verify", "freemask_semantic_output",
}
_FORBIDDEN_RAW_SEMANTIC_KEYS = {"semantic", "semantic_map", "validity", "validity_map", "bg", "rv", "myo", "lv", "void"}


class ArtifactError(ValueError):
    """Raised when an execution artifact is incomplete, unsafe, or stale."""


@dataclass(frozen=True)
class RawArtifact:
    """A verified raw partition and its canonical metadata."""

    directory: Path
    partition: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SemanticArtifact:
    """A verified semantic result and its canonical metadata."""

    directory: Path
    semantic_map: np.ndarray
    validity_map: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GeneratedSample:
    """Output of a baseline callback used by :func:`run_generation`.

    ``central_image`` is the already prepared central image in the shared
    target grid.  It is passed to the adapter only after raw sealing; v2 does
    not use its intensity values for semantic decisions.
    """

    partition: np.ndarray
    central_image: np.ndarray
    baseline_metadata: Mapping[str, Any]
    # Operational measurements deliberately live outside the scientific raw
    # payload. They describe one invocation, not the anonymous partition.
    execution_receipt: Mapping[str, Any] = field(default_factory=dict)


def _safe_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(child) for child in value]
    return value


def _validate_metadata(value: Mapping[str, Any], *, where: str = "artifact", allow_semantic: bool = False) -> None:
    """Reject GT-shaped fields before any artifact is written."""
    try:
        validate_image_only_value(value, where=where)
    except FirewallError as exc:
        raise ArtifactError(str(exc)) from exc
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in _FORBIDDEN_METADATA_KEYS or (
                not allow_semantic
                and (normalized in _FORBIDDEN_RAW_SEMANTIC_KEYS or normalized.startswith(("semantic", "validity")))
            ):
                raise ArtifactError(f"forbidden artifact field at {where}.{key}")
            if isinstance(child, Mapping):
                _validate_metadata(child, where=f"{where}.{key}", allow_semantic=allow_semantic)
            elif isinstance(child, (list, tuple)):
                for index, item in enumerate(child):
                    if isinstance(item, Mapping):
                        _validate_metadata(item, where=f"{where}.{key}[{index}]", allow_semantic=allow_semantic)


def _array_bytes(array: np.ndarray) -> bytes:
    value = np.ascontiguousarray(array)
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _file_hash(path: Path) -> str:
    return sha256_file(path)


def _atomic_bytes(path: str | Path, payload: bytes, *, before_replace: Callable[[Path], None] | None = None) -> None:
    """Write bytes beside ``path`` and atomically replace it.

    ``before_replace`` is intentionally public for interruption tests.  A
    failure leaves the previous complete file untouched and removes the
    temporary file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        if before_replace is not None:
            before_replace(temporary)
        os.replace(temporary, target)
        try:
            directory_fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def atomic_write_bytes(path: str | Path, payload: bytes, *, before_replace: Callable[[Path], None] | None = None) -> None:
    """Public atomic byte writer used by artifact and state receipts."""
    _atomic_bytes(path, payload, before_replace=before_replace)


def atomic_write_json(path: str | Path, value: Mapping[str, Any], *, before_replace: Callable[[Path], None] | None = None) -> None:
    _atomic_bytes(path, canonical_json_bytes(_safe_json(value)) + b"\n", before_replace=before_replace)


def atomic_write_npy(path: str | Path, array: np.ndarray, *, before_replace: Callable[[Path], None] | None = None) -> None:
    _atomic_bytes(path, _array_bytes(np.asarray(array)), before_replace=before_replace)


def _sample_slug(sample_id: str) -> str:
    if not isinstance(sample_id, str) or not sample_id:
        raise ArtifactError("sample_id must be a non-empty string")
    readable = _SAFE_ID_RE.sub("_", sample_id).strip("._")[:100] or "sample"
    return f"{readable}-{sha256_bytes(sample_id.encode('utf-8'))[:16]}"


def artifact_directory(output_root: str | Path, *, baseline_name: str, baseline_mode: str, sample_id: str, stage: str = "raw") -> Path:
    if stage != "raw" and stage != "semantic" and not str(stage).startswith("semantic-cardiac_adapter_"):
        raise ArtifactError("stage must be raw, historical semantic, or the current semantic adapter stage")
    if not baseline_name or not baseline_mode:
        raise ArtifactError("baseline name and mode are required")
    return Path(output_root) / stage / _sample_slug(baseline_name) / _sample_slug(baseline_mode) / _sample_slug(sample_id)


def repository_identity(repo_root: str | Path | None = None) -> dict[str, str]:
    """Return commit plus working-tree identity for cache invalidation.

    A local scientific invocation may be prepared before its infrastructure
    commit is created.  Rather than silently pretending that tree is the
    commit, bind the normalized patch as ``working_tree_sha256``.  A clean
    tree has the all-zero empty-patch digest and remains byte-stable.
    """
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        patch = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--binary"], text=False)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArtifactError(f"cannot resolve repository identity: {exc}") from exc
    return {
        "repository_commit_sha": commit,
        "working_tree_sha256": sha256_bytes(patch),
    }


def code_identity(paths: list[str | Path], *, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Hash behavior-affecting source text without binding machine paths."""
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    entries: dict[str, str] = {}
    for path_value in paths:
        path = Path(path_value)
        if not path.is_absolute():
            path = root / path
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise ArtifactError("code identity path must be inside repository") from exc
        entries[relative] = sha256_file(path)
    entries = dict(sorted(entries.items()))
    return {"files": entries, "sha256": sha256_json(entries)}


def config_hash(config: Mapping[str, Any] | str | Path) -> str:
    if isinstance(config, Mapping):
        return sha256_json(_safe_json(config))
    return sha256_file(Path(config))


def _record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    required = ("sample_id", "dataset", "patient_id", "split", "study_id", "volume_id", "frame_index", "slice_index", "source", "shared_grid")
    missing = [key for key in required if key not in record]
    if missing:
        raise ArtifactError(f"shared record missing required identity fields: {missing}")
    source = record["source"]
    if not isinstance(source, Mapping) or not isinstance(source.get("sha256"), str) or not source["sha256"]:
        raise ArtifactError("shared record lacks source image SHA-256")
    locator = source.get("locator")
    if not isinstance(locator, str) or not locator or Path(locator).is_absolute() or ".." in Path(locator).parts:
        raise ArtifactError("source locator must be a relative image-only locator")
    try:
        validate_image_only_source_locator(locator)
    except FirewallError as exc:
        raise ArtifactError(str(exc)) from exc
    return {
        "sample_id": str(record["sample_id"]), "dataset": str(record["dataset"]),
        "patient_id": str(record["patient_id"]), "split": str(record["split"]),
        "study_id": str(record["study_id"]), "volume_id": str(record["volume_id"]),
        "frame_index": int(record["frame_index"]), "slice_index": int(record["slice_index"]),
        "source_image_sha256": str(source["sha256"]),
        "source_image_hash": str(source["sha256"]),
    }


def _scientific_raw_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload = _safe_json(dict(metadata))
    for key in ("scientific_payload_hash", "local_receipt", "attempt", "state"):
        payload.pop(key, None)
    return payload


def _require_mapping(value: Mapping[str, Any], key: str, *, where: str) -> Mapping[str, Any]:
    candidate = value.get(key)
    if not isinstance(candidate, Mapping) or not candidate:
        raise ArtifactError(f"{where} requires non-empty mapping {key}")
    return candidate


def _require_string(value: Mapping[str, Any], key: str, *, where: str) -> str:
    candidate = value.get(key)
    if not isinstance(candidate, str) or not candidate:
        raise ArtifactError(f"{where} requires non-empty string {key}")
    return candidate


def _require_int(value: Mapping[str, Any], key: str, *, where: str) -> int:
    candidate = value.get(key)
    if isinstance(candidate, bool) or not isinstance(candidate, (int, np.integer)):
        raise ArtifactError(f"{where} requires integer {key}")
    return int(candidate)


def _validate_required_baseline_provenance(
    baseline_name: str, baseline_metadata: Mapping[str, Any] | None,
) -> None:
    """Require method evidence for production CUTS/DFC raw artifacts.

    Generic callers retain a baseline-agnostic writer.  The two production
    names are intentionally fail-closed so a hand-written incomplete mapping
    cannot become a RAW_COMPLETE scientific artifact.
    """
    if baseline_name not in {"CUTS", "DFC"}:
        return
    if not isinstance(baseline_metadata, Mapping):
        raise ArtifactError(f"{baseline_name} RAW_COMPLETE requires baseline provenance")
    where = f"{baseline_name} provenance"
    if baseline_name == "CUTS":
        _require_string(baseline_metadata, "checkpoint_sha256", where=where)
        _require_mapping(baseline_metadata, "checkpoint_training_provenance", where=where)
        _require_string(baseline_metadata, "cuts_mode", where=where)
        _require_mapping(baseline_metadata, "phate_configuration", where=where)
        _require_mapping(baseline_metadata, "kmeans_configuration", where=where)
        if _require_int(baseline_metadata, "primary_k", where=where) != 10:
            raise ArtifactError("CUTS provenance primary_k must be 10")
        _require_int(baseline_metadata, "clustering_seed", where=where)
        return
    _require_int(baseline_metadata, "sample_seed", where=where)
    derivation = _require_mapping(baseline_metadata, "seed_derivation", where=where)
    _require_string(derivation, "seed_derivation_version", where=where)
    _require_int(baseline_metadata, "update_count", where=where)
    _require_int(baseline_metadata, "iteration_count", where=where)
    _require_int(baseline_metadata, "final_active_count", where=where)
    _require_int(baseline_metadata, "minLabels", where=where)
    _require_int(baseline_metadata, "maxIter", where=where)
    _require_mapping(baseline_metadata, "optimizer", where=where)
    _require_string(baseline_metadata, "input_mode", where=where)
    if baseline_metadata.get("fresh_model_optimizer_bn_state_per_sample") is not True:
        raise ArtifactError("DFC provenance requires fresh_model_optimizer_bn_state_per_sample=true")
    _require_string(baseline_metadata, "final_forward_semantics", where=where)


def _raw_metadata(
    record: Mapping[str, Any], *, baseline_name: str, baseline_mode: str,
    manifest_hash: str, shared_grid_hash: str, repository: Mapping[str, Any],
    code_identity_value: Mapping[str, Any] | None,
    baseline_config_hash: str, seed: int, baseline_metadata: Mapping[str, Any] | None,
    raw_partition_path: str,
) -> dict[str, Any]:
    identity = _record_identity(record)
    _validate_required_baseline_provenance(baseline_name, baseline_metadata)
    grid = record.get("shared_grid")
    if not isinstance(grid, Mapping) or grid_hash(grid) != shared_grid_hash:
        raise ArtifactError("record shared-grid hash does not match execution contract")
    metadata: dict[str, Any] = {
        "schema_version": RAW_SCHEMA_VERSION,
        "completion_status": RAW_COMPLETE,
        "baseline_name": str(baseline_name),
        "baseline_mode": str(baseline_mode),
        **identity,
        "shared_manifest_hash": str(manifest_hash),
        "manifest_hash": str(manifest_hash),
        "shared_grid_hash": str(shared_grid_hash),
        "repository_identity": _safe_json(repository),
        "repository_commit_sha": str(repository.get("repository_commit_sha", "")),
        "code_identity": _safe_json(code_identity_value or {}),
        "baseline_config_hash": str(baseline_config_hash),
        "seed": int(seed),
        "raw_partition_path": raw_partition_path,
    }
    if baseline_metadata:
        clean_baseline_metadata = _safe_json(dict(baseline_metadata))
        metadata["baseline_provenance"] = clean_baseline_metadata
        # Keep the small set of method-required provenance fields discoverable
        # without making the shared schema depend on CUTS or DFC internals.
        for key in (
            "checkpoint_sha256", "checkpoint_identity", "checkpoint_training_provenance",
            "cuts_mode", "phate_configuration", "kmeans_configuration", "primary_k",
            "clustering_seed",
            "sample_seed", "seed_derivation", "iteration_count", "final_active_count",
            "update_count", "minLabels", "maxIter", "optimizer", "input_mode", "final_forward_semantics",
            "fresh_model_optimizer_bn_state_per_sample",
        ):
            if key in clean_baseline_metadata:
                metadata[key] = clean_baseline_metadata[key]
    _validate_metadata(metadata)
    return metadata


def _state_path(directory: Path) -> Path:
    return directory / "state.json"


def _read_state(directory: Path) -> dict[str, Any] | None:
    path = _state_path(directory)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid execution state: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ArtifactError(f"unsupported execution state: {path}")
    return value


def _write_state(
    directory: Path, *, stage: str, sample_id: str, attempt: int,
    error: Mapping[str, Any] | None = None, artifact_hash: str | None = None,
) -> dict[str, Any]:
    if stage not in {PENDING, RUNNING, RAW_COMPLETE, SEMANTIC_COMPLETE, FAILED}:
        raise ArtifactError(f"unsupported execution stage: {stage}")
    value: dict[str, Any] = {"schema_version": STATE_SCHEMA_VERSION, "sample_id": sample_id, "stage": stage, "attempt": int(attempt)}
    if error:
        value["failure"] = _safe_json(dict(error))
    if artifact_hash is not None:
        value["artifact_scientific_payload_hash"] = artifact_hash
    _validate_metadata(value, where="execution_state")
    atomic_write_json(_state_path(directory), value)
    return value


def mark_running(directory: str | Path, sample_id: str) -> dict[str, Any]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    prior = _read_state(target)
    attempt = int(prior.get("attempt", 0)) + 1 if prior else 1
    return _write_state(target, stage=RUNNING, sample_id=sample_id, attempt=attempt)


def mark_pending(directory: str | Path, sample_id: str) -> dict[str, Any]:
    """Create an explicit pending receipt before baseline computation starts."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    prior = _read_state(target)
    attempt = int(prior.get("attempt", 0)) if prior else 0
    return _write_state(target, stage=PENDING, sample_id=sample_id, attempt=attempt)


def mark_failed(directory: str | Path, sample_id: str, exc: BaseException) -> dict[str, Any]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    prior = _read_state(target)
    attempt = max(1, int(prior.get("attempt", 1))) if prior else 1
    message = str(exc).replace("\x00", " ")[:2000]
    return _write_state(target, stage=FAILED, sample_id=sample_id, attempt=attempt, error={"exception_type": type(exc).__name__, "message": message})


def _mark_complete(directory: Path, sample_id: str, stage: str, artifact_hash: str | None = None) -> dict[str, Any]:
    prior = _read_state(directory)
    attempt = int(prior.get("attempt", 1)) if prior else 1
    return _write_state(directory, stage=stage, sample_id=sample_id, attempt=attempt, artifact_hash=artifact_hash)


def _expected_match(metadata: Mapping[str, Any], expected: Mapping[str, Any] | None) -> None:
    if expected is None:
        return
    for key, value in expected.items():
        if key in {"repository_identity", "baseline_provenance"}:
            if metadata.get(key) != _safe_json(value):
                raise ArtifactError(f"raw artifact identity mismatch: {key}")
        elif metadata.get(key) != _safe_json(value):
            raise ArtifactError(f"raw artifact identity mismatch: {key}")


def seal_raw_partition(
    output_root: str | Path,
    partition: np.ndarray,
    *,
    record: Mapping[str, Any],
    baseline_name: str,
    baseline_mode: str,
    manifest_hash: str,
    shared_grid_hash: str,
    repository: Mapping[str, Any],
    code_identity_value: Mapping[str, Any] | None = None,
    code_identity: Mapping[str, Any] | None = None,
    baseline_config_hash: str,
    seed: int,
    baseline_metadata: Mapping[str, Any] | None = None,
    before_partition_replace: Callable[[Path], None] | None = None,
) -> RawArtifact:
    """Atomically write and seal a raw anonymous partition."""
    if code_identity is not None:
        if code_identity_value is not None and _safe_json(code_identity_value) != _safe_json(code_identity):
            raise ArtifactError("conflicting code identity arguments")
        code_identity_value = code_identity
    value = np.ascontiguousarray(np.asarray(partition))
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.integer) or value.size == 0:
        raise ArtifactError("raw partition must be a non-empty 2-D integer array")
    directory = artifact_directory(output_root, baseline_name=baseline_name, baseline_mode=baseline_mode, sample_id=str(record["sample_id"]))
    directory.mkdir(parents=True, exist_ok=True)
    mark_running(directory, str(record["sample_id"]))
    partition_path = directory / "raw_partition.npy"
    try:
        grid = record.get("shared_grid")
        if not isinstance(grid, Mapping) or list(grid.get("target_hw", [])) != [int(value) for value in value.shape]:
            raise ArtifactError("raw partition shape does not equal the shared target grid")
        metadata = _raw_metadata(
            record, baseline_name=baseline_name, baseline_mode=baseline_mode,
            manifest_hash=manifest_hash, shared_grid_hash=shared_grid_hash,
            repository=repository, code_identity_value=code_identity_value,
            baseline_config_hash=baseline_config_hash,
            seed=seed, baseline_metadata=baseline_metadata,
            raw_partition_path=partition_path.name,
        )
        atomic_write_npy(partition_path, value, before_replace=before_partition_replace)
        loaded = np.load(partition_path, allow_pickle=False)
        if array_hash(loaded) != array_hash(value):
            raise ArtifactError("raw partition hash verification failed")
        metadata.update({
            "dtype": str(loaded.dtype), "shape": [int(x) for x in loaded.shape],
            "raw_partition_dtype": str(loaded.dtype), "raw_partition_shape": [int(x) for x in loaded.shape],
            "raw_partition_sha256": array_hash(loaded),
            "raw_partition_file_sha256": _file_hash(partition_path),
        })
        metadata["scientific_payload_hash"] = sha256_json(_scientific_raw_payload(metadata))
        _validate_metadata(metadata)
        atomic_write_json(directory / "metadata.json", metadata)
        _mark_complete(directory, str(record["sample_id"]), RAW_COMPLETE, metadata["scientific_payload_hash"])
        return RawArtifact(directory=directory, partition=np.ascontiguousarray(loaded), metadata=metadata)
    except Exception as exc:
        mark_failed(directory, str(record["sample_id"]), exc)
        if isinstance(exc, ArtifactError):
            raise
        raise ArtifactError(str(exc)) from exc


def verify_raw_partition(directory: str | Path, *, expected: Mapping[str, Any] | None = None) -> RawArtifact:
    target = Path(directory)
    state = _read_state(target)
    if state is None or state.get("stage") != RAW_COMPLETE:
        raise ArtifactError("raw artifact is not RAW_COMPLETE")
    metadata_path = target / "metadata.json"
    if not metadata_path.is_file():
        raise ArtifactError("raw metadata is missing")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("raw metadata is invalid") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != RAW_SCHEMA_VERSION or metadata.get("completion_status") != RAW_COMPLETE:
        raise ArtifactError("raw metadata is not a complete v1 artifact")
    _validate_metadata(metadata)
    if metadata.get("sample_id") != state.get("sample_id"):
        raise ArtifactError("raw state/sample identity mismatch")
    if state.get("artifact_scientific_payload_hash") != metadata.get("scientific_payload_hash"):
        raise ArtifactError("raw completion receipt does not bind metadata")
    _expected_match(metadata, expected)
    partition_name = metadata.get("raw_partition_path")
    if not isinstance(partition_name, str) or Path(partition_name).name != partition_name:
        raise ArtifactError("raw partition path must be a local relative filename")
    partition_path = target / partition_name
    if not partition_path.is_file():
        raise ArtifactError("raw partition file is missing")
    try:
        partition = np.load(partition_path, allow_pickle=False)
    except Exception as exc:
        raise ArtifactError("raw partition cannot be decoded") from exc
    if partition.ndim != 2 or not np.issubdtype(partition.dtype, np.integer):
        raise ArtifactError("raw partition has invalid shape or dtype")
    if metadata.get("dtype") != str(partition.dtype) or metadata.get("shape") != [int(x) for x in partition.shape]:
        raise ArtifactError("raw partition shape/dtype metadata mismatch")
    if metadata.get("raw_partition_sha256") != array_hash(partition):
        raise ArtifactError("raw partition scientific hash mismatch")
    if metadata.get("raw_partition_file_sha256") != _file_hash(partition_path):
        raise ArtifactError("raw partition file hash mismatch")
    if metadata.get("scientific_payload_hash") != sha256_json(_scientific_raw_payload(metadata)):
        raise ArtifactError("raw scientific payload hash mismatch")
    return RawArtifact(directory=target, partition=np.ascontiguousarray(partition), metadata=metadata)


def raw_artifact_is_valid(directory: str | Path, *, expected: Mapping[str, Any] | None = None) -> bool:
    try:
        verify_raw_partition(directory, expected=expected)
        return True
    except (ArtifactError, OSError, ValueError):
        return False


def write_execution_receipt(
    directory: str | Path, *, sample_id: str, baseline_name: str, baseline_mode: str,
    raw_status: str, semantic_status: str | None, raw_generation_elapsed_seconds: float | None,
    adapter_elapsed_seconds: float | None, total_elapsed_seconds: float,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist observational execution metadata outside scientific cache identity."""
    receipt = {
        "schema_version": "cardiac_execution_receipt.v1",
        "sample_id": str(sample_id),
        "baseline_name": str(baseline_name),
        "baseline_mode": str(baseline_mode),
        "raw_status": str(raw_status),
        "adapter_status": semantic_status,
        "raw_generation_elapsed_seconds": None if raw_generation_elapsed_seconds is None else float(raw_generation_elapsed_seconds),
        "adapter_elapsed_seconds": None if adapter_elapsed_seconds is None else float(adapter_elapsed_seconds),
        "total_elapsed_seconds": float(total_elapsed_seconds),
        "environment": _safe_json(dict(environment or {})),
    }
    _validate_metadata(receipt, where="execution_receipt")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target / "execution_receipt.json", receipt)
    return receipt


def _adapter_implementation_files(repo_root: str | Path | None = None) -> list[Path]:
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    return [
        root / "src/shared_benchmark/adapter.py",
        root / "src/shared_benchmark/adapter_v4.py",
        root / "src/shared_benchmark/adapter_v5.py",
        root / "src/shared_benchmark/region_graph.py",
        root / "src/shared_benchmark/semantic_contract.py",
        root / "src/shared_benchmark/artifacts.py",
        root / "src/shared_benchmark/firewall.py",
        root / "src/shared_benchmark/spatial.py",
        root / "src/shared_benchmark/provenance.py",
        root / "src/self_audit_maskfree/data/firewall.py",
    ]


def _adapter_implementation_hash(repo_root: str | Path | None = None) -> str:
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    return code_identity(_adapter_implementation_files(root), repo_root=root)["sha256"]


def _semantic_scientific_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Everything except the outer self-referential scientific payload hash."""
    return {
        str(key): _safe_json(value)
        for key, value in metadata.items()
        if str(key) != "scientific_payload_hash"
    }


def _adapter_record_for_raw(
    record: Mapping[str, Any], raw_artifact: RawArtifact, central_image: np.ndarray,
    *,
    adapter_version: str = ADAPTER_VERSION,
    adapter_config_sha256: str = FROZEN_ADAPTER_SPEC_SHA256,
) -> dict[str, Any]:
    """Bind the adapter to a sealed raw partition and one central image."""
    adapter_record = dict(record)
    if "geometry_validity" not in adapter_record:
        adapter_record["geometry_validity"] = (
            dict(record.get("geometry", {}))
            if isinstance(record.get("geometry"), Mapping)
            else {"orientation_valid": False}
        )
    adapter_record.update({
        "shared_manifest_sha256": raw_artifact.metadata["shared_manifest_hash"],
        "partition_sha256": array_hash(raw_artifact.partition),
        "central_image_sha256": array_hash(np.asarray(central_image)),
        "adapter_version": str(adapter_version),
        "adapter_config_sha256": str(adapter_config_sha256),
    })
    return adapter_record


def seal_semantic_partition(
    output_root: str | Path,
    *,
    raw_artifact: RawArtifact,
    result: Any,
    baseline_name: str,
    baseline_mode: str,
    adapter_version: str = ADAPTER_VERSION,
    adapter_spec_sha256: str = FROZEN_ADAPTER_SPEC_SHA256,
    adapter_implementation_sha256: str | None = None,
) -> SemanticArtifact:
    """Persist adapter output only after a verified raw artifact."""
    try:
        semantic_input = np.asarray(result.semantic_map)
        validity_input = np.asarray(result.validity_map)
        result_metadata = dict(result.metadata)
    except AttributeError as exc:
        raise ArtifactError("adapter result must expose semantic_map, validity_map, and metadata") from exc
    if semantic_input.ndim != 2 or not np.issubdtype(semantic_input.dtype, np.integer) or not np.isin(semantic_input, [0, 1, 2, 3, 4]).all():
        raise ArtifactError("semantic map has invalid encoding")
    if validity_input.dtype != np.bool_:
        raise ArtifactError("validity map must be boolean")
    semantic = np.asarray(semantic_input, dtype=np.uint8)
    validity = np.asarray(validity_input, dtype=bool)
    if validity.shape != semantic.shape or not np.array_equal(validity, semantic != 4):
        raise ArtifactError("semantic artifact violates semantic/validity contract")
    if adapter_spec_sha256 not in {
        FROZEN_ADAPTER_SPEC_SHA256,
        FROZEN_ADAPTER_V3_SPEC_SHA256,
        FROZEN_ADAPTER_V4_SPEC_SHA256,
        FROZEN_ADAPTER_V5_SPEC_SHA256,
    }:
        raise ArtifactError("unsupported adapter specification hash")
    try:
        adapter_payload = validate_adapter_metadata(result_metadata)
    except AdapterContractError as exc:
        raise ArtifactError(str(exc)) from exc
    required_metadata = {
        "adapter_version": adapter_version,
        "adapter_config_sha256": adapter_spec_sha256,
        "sample_id": raw_artifact.metadata["sample_id"],
        "shared_manifest_sha256": raw_artifact.metadata["shared_manifest_hash"],
        "partition_sha256": raw_artifact.metadata["raw_partition_sha256"],
        "shared_grid_sha256": raw_artifact.metadata["shared_grid_hash"],
        "semantic_map_sha256": array_hash(semantic),
        "validity_map_sha256": array_hash(validity),
    }
    if any(result_metadata.get(key) != value for key, value in required_metadata.items()):
        raise ArtifactError("adapter metadata does not bind the sealed raw/result identity")
    if adapter_payload != adapter_metadata_payload(result_metadata):
        raise ArtifactError("adapter metadata payload is not canonical")
    directory = artifact_directory(
        output_root, baseline_name=baseline_name, baseline_mode=baseline_mode,
        sample_id=str(raw_artifact.metadata["sample_id"]), stage=semantic_artifact_stage(adapter_version),
    )
    directory.mkdir(parents=True, exist_ok=True)
    mark_running(directory, str(raw_artifact.metadata["sample_id"]))
    semantic_path = directory / "semantic_map.npy"
    validity_path = directory / "validity_map.npy"
    metadata: dict[str, Any] = {
        "schema_version": SEMANTIC_SCHEMA_VERSION,
        "completion_status": SEMANTIC_COMPLETE,
        "baseline_name": baseline_name,
        "baseline_mode": baseline_mode,
        "sample_id": raw_artifact.metadata["sample_id"],
        "raw_artifact_scientific_hash": raw_artifact.metadata["scientific_payload_hash"],
        "raw_partition_sha256": raw_artifact.metadata["raw_partition_sha256"],
        "adapter_version": adapter_version,
        "adapter_spec_sha256": adapter_spec_sha256,
        "adapter_implementation_sha256": adapter_implementation_sha256 or _adapter_implementation_hash(),
        "shared_manifest_hash": raw_artifact.metadata["shared_manifest_hash"],
        "shared_grid_hash": raw_artifact.metadata["shared_grid_hash"],
        "semantic_map_path": semantic_path.name,
        "validity_map_path": validity_path.name,
        "coverage": float(validity.mean()),
        "assignments": result_metadata.get("assignments", []),
        "assignment_reasons": result_metadata.get("assignment_reasons", {}),
        "void_reasons": result_metadata.get("void_reasons", {}),
        "role_reasons": result_metadata.get("role_reasons", {}),
        "unresolved_reasons": result_metadata.get("unresolved_reasons", {}),
        "component_graph_digest": result_metadata.get("component_graph_digest"),
        "central_image_sha256": result_metadata.get("central_image_sha256"),
        "adapter_metadata_sha256": result_metadata.get("metadata_sha256"),
        "scientific_result_sha256": result_metadata.get("scientific_result_sha256"),
        "intensity_resolution": result_metadata.get("intensity_resolution", "disabled"),
        "orientation_resolution": result_metadata.get("orientation_resolution", "disabled"),
        "region_splitting": result_metadata.get("region_splitting", False),
        "adapter_metadata": result_metadata,
    }
    try:
        atomic_write_npy(semantic_path, semantic)
        atomic_write_npy(validity_path, validity)
        semantic_loaded = np.load(semantic_path, allow_pickle=False)
        validity_loaded = np.load(validity_path, allow_pickle=False)
        metadata.update({
            "semantic_map_sha256": array_hash(semantic_loaded),
            "validity_map_sha256": array_hash(validity_loaded),
        })
        metadata["scientific_payload_hash"] = sha256_json(_semantic_scientific_payload(metadata))
        _validate_metadata(metadata, allow_semantic=True)
        atomic_write_json(directory / "metadata.json", metadata)
        _mark_complete(directory, str(raw_artifact.metadata["sample_id"]), SEMANTIC_COMPLETE, metadata["scientific_payload_hash"])
        return SemanticArtifact(directory=directory, semantic_map=np.ascontiguousarray(semantic_loaded), validity_map=np.ascontiguousarray(validity_loaded), metadata=metadata)
    except Exception as exc:
        mark_failed(directory, str(raw_artifact.metadata["sample_id"]), exc)
        if isinstance(exc, ArtifactError):
            raise
        raise ArtifactError(str(exc)) from exc


def verify_semantic_partition(
    directory: str | Path, *, raw_artifact: RawArtifact, record: Mapping[str, Any],
    central_image: np.ndarray, adapter_spec: Mapping[str, Any],
    adapter_version: str = ADAPTER_VERSION,
    adapter_spec_sha256: str = FROZEN_ADAPTER_SPEC_SHA256,
    adapter_implementation_sha256: str | None = None,
) -> SemanticArtifact:
    # Re-verify the on-disk raw seal at every semantic read.  Holding an old
    # in-memory ``RawArtifact`` must not allow a changed raw file to validate.
    current_raw = verify_raw_partition(raw_artifact.directory)
    if current_raw.metadata.get("scientific_payload_hash") != raw_artifact.metadata.get("scientific_payload_hash"):
        raise ArtifactError("semantic artifact is stale because the raw artifact changed")
    target = Path(directory)
    state = _read_state(target)
    if state is None or state.get("stage") != SEMANTIC_COMPLETE:
        raise ArtifactError("semantic artifact is not SEMANTIC_COMPLETE")
    try:
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("semantic metadata is invalid") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SEMANTIC_SCHEMA_VERSION:
        raise ArtifactError("unsupported semantic artifact schema")
    if state.get("sample_id") != metadata.get("sample_id") or state.get("artifact_scientific_payload_hash") != metadata.get("scientific_payload_hash"):
        raise ArtifactError("semantic completion receipt does not bind metadata")
    _validate_metadata(metadata, allow_semantic=True)
    if metadata.get("sample_id") != raw_artifact.metadata.get("sample_id"):
        raise ArtifactError("semantic/raw sample identity mismatch")
    if metadata.get("adapter_version") != adapter_version:
        raise ArtifactError("semantic artifact adapter version mismatch")
    if metadata.get("raw_artifact_scientific_hash") != raw_artifact.metadata.get("scientific_payload_hash") or metadata.get("raw_partition_sha256") != raw_artifact.metadata.get("raw_partition_sha256"):
        raise ArtifactError("semantic artifact is stale for the current raw partition")
    if metadata.get("adapter_spec_sha256") != adapter_spec_sha256:
        raise ArtifactError("semantic artifact adapter spec mismatch")
    expected_impl = adapter_implementation_sha256 or _adapter_implementation_hash()
    if metadata.get("adapter_implementation_sha256") != expected_impl:
        raise ArtifactError("semantic artifact adapter implementation mismatch")
    try:
        semantic = np.load(target / str(metadata["semantic_map_path"]), allow_pickle=False)
        validity = np.load(target / str(metadata["validity_map_path"]), allow_pickle=False)
    except Exception as exc:
        raise ArtifactError("semantic maps cannot be decoded") from exc
    if semantic.dtype != np.uint8 or validity.dtype != np.bool_ or semantic.shape != validity.shape or not np.isin(semantic, [0, 1, 2, 3, 4]).all() or not np.array_equal(validity, semantic != 4):
        raise ArtifactError("semantic map encoding/validity mismatch")
    if metadata.get("semantic_map_sha256") != array_hash(semantic) or metadata.get("validity_map_sha256") != array_hash(validity):
        raise ArtifactError("semantic map hash mismatch")
    if metadata.get("scientific_payload_hash") != sha256_json(_semantic_scientific_payload(metadata)):
        raise ArtifactError("semantic scientific payload hash mismatch")
    adapter_metadata = metadata.get("adapter_metadata")
    try:
        if not isinstance(adapter_metadata, Mapping):
            raise ArtifactError("semantic artifact adapter_metadata is required")
        validate_adapter_metadata(adapter_metadata)
        adapter_record = _adapter_record_for_raw(
            record,
            current_raw,
            np.asarray(central_image),
            adapter_version=adapter_version,
            adapter_config_sha256=adapter_spec_sha256,
        )
        recomputed = adapt_partition(
            adapter_record, current_raw.partition, np.asarray(central_image), adapter_spec=adapter_spec,
        )
    except (AdapterContractError, ValueError) as exc:
        raise ArtifactError(f"semantic adapter verification failed: {exc}") from exc
    if not np.array_equal(semantic, recomputed.semantic_map) or not np.array_equal(validity, recomputed.validity_map):
        raise ArtifactError("semantic maps do not reproduce from verified raw input")
    if _safe_json(dict(adapter_metadata)) != _safe_json(recomputed.metadata):
        raise ArtifactError("semantic adapter_metadata does not reproduce from verified raw input")
    copied_metadata = {
        "assignments": recomputed.metadata["assignments"],
        "assignment_reasons": recomputed.metadata["assignment_reasons"],
        "void_reasons": recomputed.metadata["void_reasons"],
        "role_reasons": recomputed.metadata["role_reasons"],
        "unresolved_reasons": recomputed.metadata["unresolved_reasons"],
        "component_graph_digest": recomputed.metadata["component_graph_digest"],
        "central_image_sha256": recomputed.metadata["central_image_sha256"],
        "adapter_metadata_sha256": recomputed.metadata["metadata_sha256"],
        "scientific_result_sha256": recomputed.metadata["scientific_result_sha256"],
        "coverage": float(recomputed.validity_map.mean()),
    }
    if any(_safe_json(metadata.get(key)) != _safe_json(value) for key, value in copied_metadata.items()):
        raise ArtifactError("semantic artifact copied adapter metadata mismatch")
    return SemanticArtifact(directory=target, semantic_map=np.ascontiguousarray(semantic), validity_map=np.ascontiguousarray(validity), metadata=metadata)


def select_manifest_records(manifest: Mapping[str, Any], *, split: str, limit: int | None = None, sample_list: list[str] | None = None) -> list[dict[str, Any]]:
    """Select records deterministically by canonical sample ID."""
    if split not in {"train", "dev", "test"}:
        raise ArtifactError("split must be train, dev, or test")
    records = sorted((dict(record) for record in manifest.get("records", []) if record.get("split") == split), key=lambda record: str(record["sample_id"]))
    if sample_list is not None:
        requested = list(sample_list)
        if len(set(requested)) != len(requested):
            raise ArtifactError("sample list contains duplicates")
        by_id = {record["sample_id"]: record for record in records}
        missing = [sample_id for sample_id in requested if sample_id not in by_id]
        if missing:
            raise ArtifactError(f"sample list contains records outside selected split: {missing}")
        records = [by_id[sample_id] for sample_id in requested]
    if limit is not None:
        if limit < 0:
            raise ArtifactError("limit must be non-negative")
        records = records[:limit]
    return records


def validate_scientific_execution(manifest_path: str | Path, image_root: str | Path) -> dict[str, Any]:
    """Load the image-only manifest and fail closed before baseline execution."""
    try:
        manifest = load_shared_manifest(manifest_path)
        validate_scientific_manifest(manifest, image_root=image_root, verify_source_hashes=True)
    except (SharedManifestError, OSError) as exc:
        raise ArtifactError(str(exc)) from exc
    root = Path(image_root).resolve()
    for record in manifest["records"]:
        locator = record["source"]["locator"]
        candidate = (root / locator).resolve()
        if root not in candidate.parents or candidate == root:
            raise ArtifactError(f"source escapes image-only root: {record['sample_id']}")
    if manifest.get("fixture") is True or manifest.get("scientific") is not True:
        raise ArtifactError("fixture manifests are forbidden in scientific execution")
    return manifest


def run_adapter_after_raw(
    raw_artifact: RawArtifact, *, semantic_root: str | Path, record: Mapping[str, Any],
    central_image: np.ndarray, adapter_spec: Mapping[str, Any], baseline_name: str,
    baseline_mode: str, adapter_version: str = ADAPTER_VERSION,
    adapter_spec_sha256: str = FROZEN_ADAPTER_SPEC_SHA256,
    adapter_implementation_sha256: str | None = None,
) -> SemanticArtifact:
    """Shared raw->adapter handoff; baseline modules are not imported here."""
    if raw_artifact.metadata.get("completion_status") != RAW_COMPLETE:
        raise ArtifactError("adapter handoff requires RAW_COMPLETE")
    partition = verify_raw_partition(raw_artifact.directory).partition
    adapter_record = _adapter_record_for_raw(
        record,
        raw_artifact,
        np.asarray(central_image),
        adapter_version=adapter_version,
        adapter_config_sha256=adapter_spec_sha256,
    )
    try:
        result = adapt_partition(adapter_record, partition, np.asarray(central_image), adapter_spec=adapter_spec)
    except (AdapterContractError, ValueError) as exc:
        raise ArtifactError(str(exc)) from exc
    return seal_semantic_partition(
        semantic_root, raw_artifact=raw_artifact, result=result,
        baseline_name=baseline_name, baseline_mode=baseline_mode,
        adapter_version=adapter_version,
        adapter_spec_sha256=adapter_spec_sha256,
        adapter_implementation_sha256=adapter_implementation_sha256,
    )


def run_generation(
    records: list[Mapping[str, Any]], *, output_root: str | Path, baseline_name: str, baseline_mode: str,
    manifest_hash: str, shared_grid_hash: str, repository: Mapping[str, Any], baseline_config_hash: str,
    code_identity_value: Mapping[str, Any] | None = None,
    code_identity: Mapping[str, Any] | None = None,
    seed_for_record: Callable[[Mapping[str, Any]], int], generate: Callable[[Mapping[str, Any]], GeneratedSample],
    semantic_root: str | Path | None = None, adapter_spec: Mapping[str, Any] | None = None,
    central_image_for_record: Callable[[Mapping[str, Any]], np.ndarray] | None = None,
    adapter_implementation_sha256: str | None = None, retry_failed: bool = True,
    extra_raw_identity_for_record: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Execute/resume a bounded set of samples with persistent stage receipts."""
    if code_identity is not None:
        if code_identity_value is not None and _safe_json(code_identity_value) != _safe_json(code_identity):
            raise ArtifactError("conflicting code identity arguments")
        code_identity_value = code_identity
    if len({str(record.get("sample_id")) for record in records}) != len(records):
        raise ArtifactError("generation record list contains duplicate sample IDs")
    resolved_adapter_spec = None
    adapter_spec_sha256 = None
    adapter_version = ADAPTER_VERSION
    adapter_stage = SEMANTIC_ARTIFACT_STAGE
    if adapter_spec is not None:
        resolved_adapter_spec, adapter_spec_sha256 = load_and_validate_spec(adapter_spec)
        adapter_version = str(resolved_adapter_spec["adapter_version"])
        adapter_stage = semantic_artifact_stage(adapter_version)
    results: list[dict[str, Any]] = []
    for record in records:
        sample_started = time.perf_counter()
        raw_generation_elapsed_seconds: float | None = None
        adapter_elapsed_seconds: float | None = None
        execution_environment: Mapping[str, Any] | None = None
        sample_id = str(record["sample_id"])
        directory = artifact_directory(output_root, baseline_name=baseline_name, baseline_mode=baseline_mode, sample_id=sample_id)
        expected_base = {
            "sample_id": sample_id, "shared_manifest_hash": manifest_hash,
            "shared_grid_hash": shared_grid_hash, "baseline_name": baseline_name,
            "baseline_mode": baseline_mode, "baseline_config_hash": baseline_config_hash,
            "seed": int(seed_for_record(record)),
            "repository_identity": _safe_json(repository),
            "code_identity": _safe_json(code_identity_value or {}),
            "source_image_sha256": str(record["source"]["sha256"]),
        }
        if extra_raw_identity_for_record is not None:
            extra = extra_raw_identity_for_record(record)
            if not isinstance(extra, Mapping):
                raise ArtifactError("extra raw identity callback must return a mapping")
            expected_base.update(_safe_json(dict(extra)))
        raw: RawArtifact | None = None
        try:
            if raw_artifact_is_valid(directory, expected=expected_base):
                raw = verify_raw_partition(directory, expected=expected_base)
                status = "SKIPPED_RAW_COMPLETE"
                prior_receipt_path = raw.directory / "execution_receipt.json"
                if prior_receipt_path.is_file():
                    try:
                        prior_receipt = json.loads(prior_receipt_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise ArtifactError("existing execution receipt is invalid") from exc
                    if not isinstance(prior_receipt, Mapping):
                        raise ArtifactError("existing execution receipt is invalid")
                    raw_generation_elapsed_seconds = prior_receipt.get("raw_generation_elapsed_seconds")
                    adapter_elapsed_seconds = prior_receipt.get("adapter_elapsed_seconds")
                    execution_environment = prior_receipt.get("environment")
            else:
                mark_pending(directory, sample_id)
                raw_started = time.perf_counter()
                generated = generate(record)
                raw_generation_elapsed_seconds = time.perf_counter() - raw_started
                execution_environment = generated.execution_receipt
                raw = seal_raw_partition(
                    output_root, generated.partition, record=record,
                    baseline_name=baseline_name, baseline_mode=baseline_mode,
                    manifest_hash=manifest_hash, shared_grid_hash=shared_grid_hash,
                    repository=repository, code_identity_value=code_identity_value,
                    baseline_config_hash=baseline_config_hash,
                    seed=int(seed_for_record(record)), baseline_metadata=generated.baseline_metadata,
                )
                status = "RAW_COMPLETE"
            semantic_status = None
            if semantic_root is not None and adapter_spec is not None:
                semantic_directory = artifact_directory(
                    semantic_root, baseline_name=baseline_name, baseline_mode=baseline_mode,
                    sample_id=sample_id, stage=adapter_stage,
                )
                if status == "RAW_COMPLETE":
                    central_image = np.asarray(generated.central_image)
                elif central_image_for_record is None:
                    raise ArtifactError("a central_image_for_record callback is required for adapter resume")
                else:
                    central_image = np.asarray(central_image_for_record(record))
                try:
                    semantic = verify_semantic_partition(
                        semantic_directory, raw_artifact=raw, record=record,
                        central_image=central_image, adapter_spec=resolved_adapter_spec,
                        adapter_version=adapter_version,
                        adapter_spec_sha256=str(adapter_spec_sha256),
                        adapter_implementation_sha256=adapter_implementation_sha256,
                    )
                    semantic_status = "SKIPPED_SEMANTIC_COMPLETE"
                except ArtifactError:
                    adapter_started = time.perf_counter()
                    try:
                        semantic = run_adapter_after_raw(
                            raw, semantic_root=semantic_root, record=record,
                            central_image=central_image, adapter_spec=resolved_adapter_spec,
                            baseline_name=baseline_name, baseline_mode=baseline_mode,
                            adapter_version=adapter_version,
                            adapter_spec_sha256=str(adapter_spec_sha256),
                            adapter_implementation_sha256=adapter_implementation_sha256,
                        )
                    finally:
                        adapter_elapsed_seconds = time.perf_counter() - adapter_started
                    semantic_status = "SEMANTIC_COMPLETE"
                receipt = write_execution_receipt(
                    raw.directory, sample_id=sample_id, baseline_name=baseline_name, baseline_mode=baseline_mode,
                    raw_status=status, semantic_status=semantic_status,
                    raw_generation_elapsed_seconds=raw_generation_elapsed_seconds,
                    adapter_elapsed_seconds=adapter_elapsed_seconds,
                    total_elapsed_seconds=time.perf_counter() - sample_started,
                    environment=execution_environment,
                )
                results.append({"sample_id": sample_id, "raw_status": status, "semantic_status": semantic_status, "raw_artifact": raw, "semantic_artifact": semantic, "execution_receipt": receipt})
            else:
                receipt = write_execution_receipt(
                    raw.directory, sample_id=sample_id, baseline_name=baseline_name, baseline_mode=baseline_mode,
                    raw_status=status, semantic_status=None,
                    raw_generation_elapsed_seconds=raw_generation_elapsed_seconds,
                    adapter_elapsed_seconds=None, total_elapsed_seconds=time.perf_counter() - sample_started,
                    environment=execution_environment,
                )
                results.append({"sample_id": sample_id, "raw_status": status, "semantic_status": None, "raw_artifact": raw, "execution_receipt": receipt})
        except Exception as exc:
            # The raw writer records its own failure receipt.  This outer path
            # also covers failures before a directory exists (e.g. callback
            # validation), making retry intent explicit and inspectable.
            if raw is None:
                mark_failed(directory, sample_id, exc)
                raw_status = FAILED
            else:
                # A semantic failure must never demote a sealed raw artifact.
                # Keep RAW_COMPLETE and persist the failure on the semantic
                # receipt so a later invocation can retry only the handoff.
                raw_status = status if "status" in locals() else RAW_COMPLETE
                if semantic_root is not None:
                    semantic_directory = artifact_directory(
                        semantic_root, baseline_name=baseline_name, baseline_mode=baseline_mode,
                        sample_id=sample_id, stage=adapter_stage,
                    )
                    mark_failed(semantic_directory, sample_id, exc)
            if raw is not None:
                receipt = write_execution_receipt(
                    raw.directory, sample_id=sample_id, baseline_name=baseline_name, baseline_mode=baseline_mode,
                    raw_status=raw_status, semantic_status=FAILED,
                    raw_generation_elapsed_seconds=raw_generation_elapsed_seconds,
                    adapter_elapsed_seconds=adapter_elapsed_seconds,
                    total_elapsed_seconds=time.perf_counter() - sample_started,
                    environment=execution_environment,
                )
            else:
                receipt = None
            if not retry_failed:
                raise
            results.append({"sample_id": sample_id, "raw_status": raw_status, "semantic_status": FAILED if raw is not None else None, "execution_receipt": receipt, "error": {"exception_type": type(exc).__name__, "message": str(exc)[:2000]}})
    return results


# Descriptive aliases used by orchestration callers that refer to the files
# as artifacts rather than partitions.  They intentionally preserve one
# implementation and therefore one validation path.
write_raw_artifact = seal_raw_partition
verify_raw_artifact = verify_raw_partition
write_semantic_artifact = seal_semantic_partition
verify_semantic_artifact = verify_semantic_partition


__all__ = [
    "ArtifactError", "GeneratedSample", "RawArtifact", "SemanticArtifact",
    "RAW_SCHEMA_VERSION", "SEMANTIC_SCHEMA_VERSION", "SEMANTIC_ARTIFACT_STAGE", "STATE_SCHEMA_VERSION",
    "PENDING", "RUNNING", "RAW_COMPLETE", "SEMANTIC_COMPLETE", "FAILED",
    "atomic_write_bytes", "atomic_write_json", "atomic_write_npy", "artifact_directory", "write_execution_receipt",
    "code_identity", "config_hash", "repository_identity", "mark_failed", "mark_pending", "mark_running",
    "raw_artifact_is_valid", "seal_raw_partition", "verify_raw_partition",
    "seal_semantic_partition", "verify_semantic_partition", "run_adapter_after_raw",
    "write_raw_artifact", "verify_raw_artifact", "write_semantic_artifact", "verify_semantic_artifact",
    "run_generation", "select_manifest_records", "validate_scientific_execution",
]
