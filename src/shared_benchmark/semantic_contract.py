"""Typed, GT-free contracts for the frozen cardiac adapter boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .firewall import FirewallError, validate_image_only_source_locator, validate_image_only_value
from .provenance import canonical_json_bytes, sha256_bytes, sha256_json
from .spatial import SPATIAL_CONTRACT_VERSION, grid_hash


ADAPTER_VERSION = "cardiac_adapter_v1"
ADAPTER_INPUT_SCHEMA_VERSION = "shared_benchmark.anonymous_partition.v1"
ADAPTER_OUTPUT_SCHEMA_VERSION = "shared_benchmark.cardiac-semantic.v1"
FROZEN_ADAPTER_SPEC_SHA256 = "34b1faeb7b40f77c2d4d9789a6e1a342e891fcb8ff8db4edd30957b2ae9d114a"
FROZEN_ADAPTER_SPEC_CANONICAL_SHA256 = "c15050164214414a532e862cef17f3a6151c40836e72ce1e6fa06288b2c1f013"
FROZEN_SHARED_GRID_SHA256 = "7c9d33fed0facbbabe736a5216bc599b65f1b465e3e26a3d19c624cf9be57949"

BG = 0
RV = 1
MYO = 2
LV = 3
VOID = 4
SEMANTIC_NAMES = {BG: "BG", RV: "RV", MYO: "MYO", LV: "LV", VOID: "VOID"}

_ALLOWED_RECORD_KEYS = frozenset({
    "dataset", "patient_id", "study_id", "volume_id", "sample_id", "split", "frame_index", "frame_axis",
    "slice_index", "depth", "depth_axis", "context_indices", "frame_selection_rule", "source",
    "native_shape", "stored_shape", "native_hw", "axis_semantics", "geometry", "shared_grid",
    "spatial_transform", "shared_manifest_sha256", "source_image_hash", "geometry_validity",
    "partition_sha256", "central_image_sha256", "adapter_version", "adapter_config_sha256",
})

_REQUIRED_RECORD_KEYS = frozenset({
    "sample_id", "shared_manifest_sha256", "shared_grid", "geometry_validity",
    "partition_sha256", "central_image_sha256", "adapter_version", "adapter_config_sha256",
})

_FORBIDDEN_INPUT_NAMES = frozenset({
    "gt", "groundtruth", "ground_truth", "mask", "masks", "label", "labels",
    "annotation", "annotations", "dice", "iou", "hausdorff", "metric", "metrics",
    "validation", "oracle", "oracle_mapping", "oracle_semantic_mapping", "hungarian", "hungarian_mapping",
    "freemask_semantic_output", "freemask_auditor", "candidate_bank",
    "predictive_evidence", "o_fit", "o_select", "o_verify", "method_semantic_hint",
})


class AdapterContractError(ValueError):
    """Raised when an adapter input is incomplete or carries forbidden evidence."""


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = f"dtype={value.dtype.str};shape={tuple(int(x) for x in value.shape)};".encode("ascii")
    return sha256_bytes(header + value.tobytes(order="C"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


@dataclass(frozen=True)
class AdapterResult:
    semantic_map: np.ndarray
    validity_map: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        semantic = np.asarray(self.semantic_map)
        validity = np.asarray(self.validity_map)
        if semantic.dtype != np.uint8 or validity.dtype != np.bool_:
            raise AdapterContractError("adapter outputs must be uint8 semantic and bool validity arrays")
        if not np.isin(semantic, [BG, RV, MYO, LV, VOID]).all():
            raise AdapterContractError("adapter semantic output contains an unknown encoding")
        if semantic.shape != validity.shape or not np.array_equal(validity, semantic != VOID):
            raise AdapterContractError("validity must be false exactly where semantic == VOID")


def load_and_validate_spec(adapter_spec: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    spec = _json_safe(adapter_spec)
    if not isinstance(spec, dict):
        raise AdapterContractError("adapter_spec must be a JSON object")
    digest = sha256_bytes(canonical_json_bytes(spec))
    if digest != FROZEN_ADAPTER_SPEC_CANONICAL_SHA256:
        raise AdapterContractError("adapter_spec canonical content does not match frozen cardiac_adapter_v1")
    if (
        spec.get("adapter_version") != ADAPTER_VERSION
        or spec.get("input_schema_version") != ADAPTER_INPUT_SCHEMA_VERSION
        or spec.get("output_schema_version") != ADAPTER_OUTPUT_SCHEMA_VERSION
    ):
        raise AdapterContractError("unsupported adapter specification")
    if spec.get("scientific_properties", {}).get("region_splitting") is not False:
        raise AdapterContractError("cardiac_adapter_v1 cannot split components")
    # The canonical digest proves the mapping has the frozen scientific
    # content; the bound output field is the SHA-256 of the frozen file.
    return spec, FROZEN_ADAPTER_SPEC_SHA256


def _normalize_input_name(name: object) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _validate_forbidden_input_names(value: Any, *, where: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalize_input_name(key)
            if normalized in _FORBIDDEN_INPUT_NAMES:
                raise AdapterContractError(f"forbidden adapter evidence at {where}.{key}")
            if normalized.endswith(("_path", "_root", "_file")) and any(
                token in normalized for token in ("gt", "mask", "label", "annotation", "reference", "groundtruth")
            ):
                raise AdapterContractError(f"forbidden adapter evidence at {where}.{key}")
            _validate_forbidden_input_names(child, where=f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_forbidden_input_names(child, where=f"{where}[{index}]")


def validate_adapter_record(record: Mapping[str, Any], partition: np.ndarray, central_image: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray, str]:
    if not isinstance(record, Mapping):
        raise AdapterContractError("record must be a mapping")
    unknown = sorted(set(record) - _ALLOWED_RECORD_KEYS, key=str)
    if unknown:
        raise AdapterContractError(f"adapter record has unsupported fields: {unknown}")
    missing = sorted(_REQUIRED_RECORD_KEYS - set(record))
    if missing:
        raise AdapterContractError(f"adapter record is missing required fields: {missing}")
    try:
        validate_image_only_value(record, where="adapter.record")
    except FirewallError as exc:
        raise AdapterContractError(str(exc)) from exc
    _validate_forbidden_input_names(record, where="adapter.record")
    source = record.get("source")
    if source is not None:
        if not isinstance(source, Mapping):
            raise AdapterContractError("record source must be a mapping")
        locator = source.get("locator")
        if locator is not None:
            if not isinstance(locator, str) or not locator:
                raise AdapterContractError("record source locator must be a non-empty string")
            try:
                validate_image_only_source_locator(locator)
            except FirewallError as exc:
                raise AdapterContractError(str(exc)) from exc
    partition_value = np.asarray(partition)
    image_value = np.asarray(central_image)
    if partition_value.ndim != 2 or not np.issubdtype(partition_value.dtype, np.integer):
        raise AdapterContractError("partition must be a 2-D integer array")
    if any(int(size) < 1 for size in partition_value.shape):
        raise AdapterContractError("partition must be non-empty")
    if image_value.ndim != 2 or image_value.shape != partition_value.shape or not np.issubdtype(image_value.dtype, np.number):
        raise AdapterContractError("central_image must be a numeric 2-D array matching partition shape")
    if not np.isfinite(image_value).all():
        raise AdapterContractError("central_image must be finite")
    if record.get("adapter_version") != ADAPTER_VERSION:
        raise AdapterContractError("record adapter_version is not cardiac_adapter_v1")
    if record.get("adapter_config_sha256") != FROZEN_ADAPTER_SPEC_SHA256:
        raise AdapterContractError("record adapter_config_sha256 does not bind the frozen adapter spec")
    if record.get("partition_sha256") != array_hash(partition_value):
        raise AdapterContractError("partition_sha256 does not match partition")
    if record.get("central_image_sha256") != array_hash(image_value):
        raise AdapterContractError("central_image_sha256 does not match central_image")
    grid = record.get("shared_grid")
    if not isinstance(grid, Mapping) or grid.get("version") != SPATIAL_CONTRACT_VERSION:
        raise AdapterContractError("record must carry a supported shared-grid contract")
    if list(grid.get("target_hw", [])) != [int(partition_value.shape[0]), int(partition_value.shape[1])]:
        raise AdapterContractError("partition shape must equal the declared shared target grid")
    if grid.get("whole_fov") is not True or grid.get("crop") is not None:
        raise AdapterContractError("adapter requires the whole-FOV shared grid")
    if grid.get("forward_values") != "masked_area_normalized_convolution":
        raise AdapterContractError("adapter requires the frozen masked-area shared grid")
    if not isinstance(record.get("geometry_validity"), Mapping):
        raise AdapterContractError("geometry_validity metadata is required")
    sample_id = record.get("sample_id")
    manifest_hash = record.get("shared_manifest_sha256")
    if not isinstance(sample_id, str) or not sample_id or not isinstance(manifest_hash, str) or not manifest_hash:
        raise AdapterContractError("sample_id and shared manifest hash are required")
    return dict(record), np.ascontiguousarray(partition_value), np.ascontiguousarray(image_value), str(manifest_hash)


def canonical_metadata_hash(metadata: Mapping[str, Any]) -> str:
    return sha256_json(_json_safe(metadata))
