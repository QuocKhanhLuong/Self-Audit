"""Typed, GT-free contracts for the frozen cardiac adapter boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .firewall import FirewallError, validate_image_only_source_locator, validate_image_only_value
from .provenance import canonical_json_bytes, sha256_bytes, sha256_json
from .spatial import (
    SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION,
    SELF_AUDIT_SPATIAL_CONTRACT_VERSION,
    SPATIAL_CONTRACT_VERSION,
    grid_hash,
)


ADAPTER_VERSION = "cardiac_adapter_v2"
ADAPTER_INPUT_SCHEMA_VERSION = "shared_benchmark.anonymous_partition.v2"
ADAPTER_OUTPUT_SCHEMA_VERSION = "shared_benchmark.cardiac-semantic.v2"
FROZEN_ADAPTER_SPEC_SHA256 = "be5b46f1c0b7bc71e3c49117b649d43d208b42b4d237c95ea06d43de9b6ea290"
FROZEN_ADAPTER_SPEC_CANONICAL_SHA256 = "9d4f6389f21ed33a2d423ff938526cdf6993ccb4d74021054ff972b89ed82c47"
ADAPTER_V3_VERSION = "cardiac_adapter_v3"
ADAPTER_V3_INPUT_SCHEMA_VERSION = "shared_benchmark.anonymous_partition.v3"
ADAPTER_V3_OUTPUT_SCHEMA_VERSION = "shared_benchmark.cardiac-semantic.v3"
FROZEN_ADAPTER_V3_SPEC_SHA256 = "c786dbf0733d380f71f808bff33c7299bdaa92a72d8c876a7fbfc8474faaab0e"
FROZEN_ADAPTER_V3_SPEC_CANONICAL_SHA256 = "20e973820a648a13a11a8b2af3a2e22ad1cbe3e26a05a975f4c41eafbf67a438"
# Historical v1/v2/v3 freeze artifacts remain independently verifiable.  New
# semantic artifacts must bind the Self-Audit-derived grid and adapter v2.
FROZEN_SHARED_GRID_SHA256 = "2d53b65aa94d6cc151a02444a03b2246034cc5334e089e4d86a8de69ce494fc1"
FROZEN_SELF_AUDIT_SHARED_GRID_SHA256 = "57858ddf831decee0ce40c0fcc66f68b794e9c94b8f1eebe0a3a146502e535bf"
FROZEN_SELF_AUDIT_COMPAT_224_SHARED_GRID_SHA256 = "e9cbf5b463286ed3a57b905c4dbe826bd89760eca6469d6a7b1d0c608d08b5a1"
SUPPORTED_FROZEN_GRID_SHA256 = frozenset({
    FROZEN_SHARED_GRID_SHA256,
    FROZEN_SELF_AUDIT_SHARED_GRID_SHA256,
    FROZEN_SELF_AUDIT_COMPAT_224_SHARED_GRID_SHA256,
})

_SPEC_BY_VERSION = {
    ADAPTER_VERSION: {
        "input_schema_version": ADAPTER_INPUT_SCHEMA_VERSION,
        "output_schema_version": ADAPTER_OUTPUT_SCHEMA_VERSION,
        "file_sha256": FROZEN_ADAPTER_SPEC_SHA256,
        "canonical_sha256": FROZEN_ADAPTER_SPEC_CANONICAL_SHA256,
    },
    ADAPTER_V3_VERSION: {
        "input_schema_version": ADAPTER_V3_INPUT_SCHEMA_VERSION,
        "output_schema_version": ADAPTER_V3_OUTPUT_SCHEMA_VERSION,
        "file_sha256": FROZEN_ADAPTER_V3_SPEC_SHA256,
        "canonical_sha256": FROZEN_ADAPTER_V3_SPEC_CANONICAL_SHA256,
    },
}

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
    version = spec.get("adapter_version")
    expected = _SPEC_BY_VERSION.get(str(version))
    if expected is None:
        raise AdapterContractError("unsupported adapter specification")
    digest = sha256_bytes(canonical_json_bytes(spec))
    if digest != expected["canonical_sha256"]:
        raise AdapterContractError(f"adapter_spec canonical content does not match frozen {version}")
    if (
        spec.get("input_schema_version") != expected["input_schema_version"]
        or spec.get("output_schema_version") != expected["output_schema_version"]
    ):
        raise AdapterContractError("unsupported adapter specification")
    if spec.get("scientific_properties", {}).get("region_splitting") is not False:
        raise AdapterContractError(f"{version} cannot split components")
    # The canonical digest proves the mapping has the frozen scientific
    # content; the bound output field is the SHA-256 of the frozen file.
    return spec, str(expected["file_sha256"])


def semantic_artifact_stage(adapter_version: str) -> str:
    return f"semantic-{str(adapter_version)}"


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


def validate_adapter_record(
    record: Mapping[str, Any],
    partition: np.ndarray,
    central_image: np.ndarray,
    *,
    adapter_version: str,
    adapter_config_sha256: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, str]:
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
    if record.get("adapter_version") != adapter_version:
        raise AdapterContractError(f"record adapter_version is not {adapter_version}")
    if record.get("adapter_config_sha256") != adapter_config_sha256:
        raise AdapterContractError("record adapter_config_sha256 does not bind the frozen adapter spec")
    if record.get("partition_sha256") != array_hash(partition_value):
        raise AdapterContractError("partition_sha256 does not match partition")
    if record.get("central_image_sha256") != array_hash(image_value):
        raise AdapterContractError("central_image_sha256 does not match central_image")
    grid = record.get("shared_grid")
    allowed_versions = {
        ADAPTER_VERSION: {SPATIAL_CONTRACT_VERSION, SELF_AUDIT_SPATIAL_CONTRACT_VERSION},
        ADAPTER_V3_VERSION: {SELF_AUDIT_SPATIAL_CONTRACT_VERSION, SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION},
    }
    if not isinstance(grid, Mapping) or grid.get("version") not in allowed_versions.get(adapter_version, set()):
        raise AdapterContractError("record must carry a supported shared-grid contract")
    expected_grid_hash = grid_hash(grid)
    # Historical topology fixtures intentionally use small arbitrary v1
    # grids.  Preserve that synthetic contract while requiring an exact hash
    # whenever a scientific Self-Audit grid is carried.
    if grid.get("version") == SELF_AUDIT_SPATIAL_CONTRACT_VERSION and expected_grid_hash != FROZEN_SELF_AUDIT_SHARED_GRID_SHA256:
        raise AdapterContractError("record Self-Audit shared-grid content is not the frozen scientific contract")
    if grid.get("version") == SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION and expected_grid_hash != FROZEN_SELF_AUDIT_COMPAT_224_SHARED_GRID_SHA256:
        raise AdapterContractError("record Self-Audit compat-224 shared-grid content is not the frozen scientific contract")
    if list(grid.get("target_hw", [])) != [int(partition_value.shape[0]), int(partition_value.shape[1])]:
        raise AdapterContractError("partition shape must equal the declared shared target grid")
    if grid.get("whole_fov") is not True or grid.get("crop") is not None:
        raise AdapterContractError("adapter requires the whole-FOV shared grid")
    expected_forward = "masked_area_normalized_convolution" if grid.get("version") == SPATIAL_CONTRACT_VERSION else "bilinear_align_corners_false"
    if grid.get("forward_values") != expected_forward:
        raise AdapterContractError("record shared-grid interpolation does not match its frozen contract")
    if not isinstance(record.get("geometry_validity"), Mapping):
        raise AdapterContractError("geometry_validity metadata is required")
    sample_id = record.get("sample_id")
    manifest_hash = record.get("shared_manifest_sha256")
    if not isinstance(sample_id, str) or not sample_id or not isinstance(manifest_hash, str) or not manifest_hash:
        raise AdapterContractError("sample_id and shared manifest hash are required")
    return dict(record), np.ascontiguousarray(partition_value), np.ascontiguousarray(image_value), str(manifest_hash)


def canonical_metadata_hash(metadata: Mapping[str, Any]) -> str:
    return sha256_json(_json_safe(metadata))


def adapter_metadata_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete adapter metadata covered by ``metadata_sha256``.

    The self-referential digest field is intentionally the only excluded key.
    This makes a canonical metadata seal useful both inside a semantic
    artifact and when a raw+semantic handoff is recomputed during verification.
    """
    return {
        str(key): _json_safe(value)
        for key, value in metadata.items()
        if str(key) != "metadata_sha256"
    }


def validate_adapter_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless all adapter metadata matches its canonical seal."""
    if not isinstance(metadata, Mapping):
        raise AdapterContractError("adapter metadata must be a mapping")
    digest = metadata.get("metadata_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise AdapterContractError("adapter metadata_sha256 is required")
    payload = adapter_metadata_payload(metadata)
    if canonical_metadata_hash(payload) != digest:
        raise AdapterContractError("adapter metadata_sha256 does not bind complete metadata")
    return payload
