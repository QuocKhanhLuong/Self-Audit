"""Shared whole-FOV target-grid contracts for frozen cardiac baselines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from self_audit_maskfree.data.geometry import inverse_transform_record, read_slice_stack, read_source_frame
from self_audit_maskfree.data.geometry import masked_resize

from .firewall import FirewallError, validate_image_only_source_locator
from .provenance import sha256_file, sha256_json


# v1 is retained for the historical FreeMask-derived freeze.  The original
# Self-Audit loader has a different input contract (256x256 and PyTorch
# bilinear interpolation), so it gets an explicitly versioned v2 contract
# instead of silently changing the meaning of the old hash.
SPATIAL_CONTRACT_VERSION = "shared_benchmark.spatial.v1"
SELF_AUDIT_SPATIAL_CONTRACT_VERSION = "shared_benchmark.spatial.self_audit.v1"
SELF_AUDIT_NORMALIZATION_VERSION = "self_audit.volume_percentile_clip_0p5_99p5_zscore.v1"
# This is deliberately *not* the current Self-Audit network grid.  It is a
# separately frozen, baseline-only reconstruction of the historical ACDC
# preprocessing default (224), followed by the loader's second volume
# normalization, with no subsequent 224->256 network resize.  Keeping a
# separate version prevents a 224 baseline result from being labelled as a
# result of the current 256 Self-Audit configuration.
SELF_AUDIT_HISTORICAL_224_SPATIAL_CONTRACT_VERSION = (
    "shared_benchmark.spatial.self_audit_historical_224.v1"
)
SELF_AUDIT_HISTORICAL_224_NORMALIZATION_VERSION = (
    "self_audit.preprocess_acdc_224_then_loader_volume_percentile_clip_0p5_99p5_zscore.v1"
)


class SharedSpatialError(ValueError):
    """Raised for an invalid common grid or source/grid incompatibility."""


def build_grid_spec(
    target_hw: tuple[int, int], *, config_provenance: Mapping[str, Any],
    version: str = SPATIAL_CONTRACT_VERSION,
    forward_values: str = "masked_area_normalized_convolution",
    forward_masks: str = "nearest_exact",
    inverse_labels: str = "nearest_exact",
    inverse_probabilities: str = "bilinear_then_renormalize",
) -> dict[str, Any]:
    """Create the canonical whole-FOV contract without prescribing intensity stats."""
    if len(target_hw) != 2 or any(int(value) < 2 for value in target_hw):
        raise SharedSpatialError("target grid must be [H,W] with each dimension >= 2")
    return {
        "version": str(version),
        "target_hw": [int(target_hw[0]), int(target_hw[1])],
        "whole_fov": True,
        "crop": None,
        "crop_rule": "whole_field_of_view_no_image_informed_crop",
        "forward_values": str(forward_values),
        "forward_masks": str(forward_masks),
        "inverse_labels": str(inverse_labels),
        "inverse_probabilities": str(inverse_probabilities),
        "config_provenance": dict(config_provenance),
    }


def grid_hash(grid: Mapping[str, Any]) -> str:
    return sha256_json(dict(grid))


def _load_yaml_image_size(path: Path) -> int:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project config dependency
        raise SharedSpatialError("PyYAML is required to read pinned FreeMask configs") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = payload.get("image_size") if isinstance(payload, dict) else None
    if not isinstance(value, int) or value < 2:
        raise SharedSpatialError(f"{path} has no valid image_size")
    return value


def load_pinned_grid_spec(repo_root: str | Path) -> dict[str, Any]:
    """Read both authoritative configs and fail if they do not share one grid."""
    root = Path(repo_root)
    paths = [root / "configs" / "maskfree_acdc_150.yaml", root / "configs" / "maskfree_mnms_150.yaml"]
    sizes = [_load_yaml_image_size(path) for path in paths]
    if len(set(sizes)) != 1:
        raise SharedSpatialError(f"FreeMask benchmark configs disagree on image_size: {sizes}")
    provenance = {
        "source": "FreeMask benchmark configs",
        "config_files": [str(path.relative_to(root)).replace("\\", "/") for path in paths],
        "config_sha256": {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path) for path in paths},
        "image_size": sizes[0],
    }
    return build_grid_spec((sizes[0], sizes[0]), config_provenance=provenance)


def _load_yaml_payload(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SharedSpatialError("PyYAML is required to read pinned Self-Audit configs") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SharedSpatialError(f"{path} must contain a mapping")
    return payload


def _self_audit_image_size(payload: Mapping[str, Any], path: Path) -> int:
    # Unified v1 nests the data section; the historical phase configs keep
    # dataset fields at the top level.  Both are source configs of the same
    # Self-Audit protocol and must agree before a grid can be frozen.
    dataset = payload.get("dataset")
    section = dataset if isinstance(dataset, Mapping) else payload
    value = section.get("image_size")
    if not isinstance(value, int) or value < 2:
        raise SharedSpatialError(f"{path} has no valid Self-Audit image_size")
    return int(value)


def load_self_audit_grid_spec(repo_root: str | Path) -> dict[str, Any]:
    """Load the grid stated by the original Self-Audit configs.

    This is deliberately separate from :func:`load_pinned_grid_spec`: the
    latter remains the historical FreeMask contract used by freeze v2.  The
    new contract binds all checked-in Self-Audit ACDC phase/unified configs,
    uses their 256x256 network grid, and records the loader-compatible
    PyTorch bilinear interpolation explicitly.
    """
    root = Path(repo_root)
    paths = [
        root / "configs" / name
        for name in (
            "self_audit_full.yaml",
            "self_audit_annotation.yaml",
            "self_audit_auditor.yaml",
            "self_audit_joint.yaml",
        )
    ]
    payloads = [_load_yaml_payload(path) for path in paths]
    sizes = [_self_audit_image_size(payload, path) for payload, path in zip(payloads, paths)]
    if len(set(sizes)) != 1:
        raise SharedSpatialError(f"Self-Audit configs disagree on image_size: {sizes}")
    full = payloads[0].get("dataset", {})
    preprocessing = full.get("preprocessing", {}) if isinstance(full, Mapping) else {}
    clipping = [preprocessing.get("clipping_min"), preprocessing.get("clipping_max")]
    if clipping != [0.5, 99.5]:
        raise SharedSpatialError(
            "Self-Audit canonical preprocessing must retain clipping_min=0.5 and clipping_max=99.5"
        )
    if full.get("depth_axis") != 2:
        raise SharedSpatialError("Self-Audit canonical ACDC depth_axis must be 2")
    provenance = {
        "source": "Self-Audit original ACDC loader/config protocol",
        "config_files": [str(path.relative_to(root)).replace("\\", "/") for path in paths],
        "config_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
            for path in paths
        },
        "image_size": sizes[0],
        "depth_axis": 2,
        "crop": None,
        "whole_fov": True,
        "preprocessing": {
            "clipping_min": 0.5,
            "clipping_max": 99.5,
            "normalization": "volume_percentile_clip_then_volume_zscore",
        },
        "interpolation": "torch.nn.functional.interpolate(mode=bilinear,align_corners=False)",
    }
    return build_grid_spec(
        (sizes[0], sizes[0]),
        config_provenance=provenance,
        version=SELF_AUDIT_SPATIAL_CONTRACT_VERSION,
        forward_values="bilinear_align_corners_false",
        forward_masks="nearest_exact",
        inverse_labels="nearest_exact",
        inverse_probabilities="bilinear_then_renormalize",
    )


def load_self_audit_historical_224_grid_spec(repo_root: str | Path) -> dict[str, Any]:
    """Return the isolated CUTS/DFC historical-224 input contract.

    ``scripts/preprocess_acdc.py`` historically defaulted to 224x224 and
    persisted the normalized/resized image array.  The checked-in loader then
    applied its own volume percentile/z-score operation.  The current
    Self-Audit runner additionally resizes that loader tensor to its configured
    256x256 network grid; this baseline-only contract intentionally does not.

    No current Self-Audit configuration or source loader is changed by this
    function.  The source files below are evidence for a reimplementation,
    rather than an assertion that a missing historical ``preprocessed_data``
    artifact is byte-identical to a new execution.
    """
    root = Path(repo_root)
    preprocessor = root / "scripts" / "preprocess_acdc.py"
    loader = root / "src" / "self_audit" / "data" / "common.py"
    for path in (preprocessor, loader):
        if not path.is_file():
            raise SharedSpatialError(f"historical Self-Audit source is missing: {path}")
    provenance = {
        "source": "Self-Audit historical ACDC image preprocessing reimplemented for CUTS/DFC only",
        "historical_preprocessor": {
            "path": "scripts/preprocess_acdc.py",
            "sha256": sha256_file(preprocessor),
            "target_size_cli_default": 224,
            "image_normalization": "normalize_zscore: percentile(0.5,99.5), clip, population_zscore",
            "image_resize": (
                "skimage.transform.resize(order=1,preserve_range=True,"
                "anti_aliasing=True,mode=reflect) per HW slice"
            ),
            "image_branch_only": True,
        },
        "loader_normalization": {
            "path": "src/self_audit/data/common.py",
            "sha256": sha256_file(loader),
            "function": "percentile_clip_and_zscore(lower_percentile=0.5,upper_percentile=99.5)",
        },
        "target_size": 224,
        "depth_axis": 2,
        "crop": None,
        "whole_fov": True,
        "network_resize_after_loader": "none_baseline_only",
        "numeric_parity_status": (
            "source-code reproduction; retained historical preprocessed metadata/artifacts unavailable, "
            "so bytewise parity is not asserted"
        ),
    }
    return build_grid_spec(
        (224, 224),
        config_provenance=provenance,
        version=SELF_AUDIT_HISTORICAL_224_SPATIAL_CONTRACT_VERSION,
        forward_values="historical_preprocess_224_no_post_loader_resize",
        forward_masks="not_applicable_image_only",
        inverse_labels="not_applicable_baseline_partition",
        inverse_probabilities="not_applicable_baseline_partition",
    )


def resolve_source_path(record: Mapping[str, Any], source_root: str | Path | None = None) -> Path:
    """Resolve only a declared relative image locator under a declared image root."""
    source = record.get("source", {})
    locator = source.get("locator")
    if not isinstance(locator, str) or not locator:
        raise SharedSpatialError("record source locator is required")
    validate_image_only_source_locator(locator)
    root_value = source_root
    if root_value is None:
        root_value = record.get("local_source_root")
    if root_value is None:
        raise SharedSpatialError("source_root is required; absolute local paths are not scientific identity")
    root = Path(root_value).resolve()
    candidate = (root / locator).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FirewallError("source locator escapes declared image-only root") from exc
    if not candidate.is_file():
        raise SharedSpatialError(f"declared source file does not exist: {candidate}")
    return candidate


def read_context_stack(record: Mapping[str, Any], *, source_root: str | Path | None = None) -> np.ndarray:
    """Decode a declared image-only source with endpoint-replicated context.

    The small rank-3/rank-4 decoder is shared implementation code; cohort,
    split, frame selection, and grid authority come from the source protocol
    bound in the manifest, not from the legacy mask-free discovery module.
    """
    source = resolve_source_path(record, source_root)
    source_info = record["source"]
    stack = read_slice_stack(
        source,
        depth_axis=int(record["depth_axis"]),
        slice_index=int(record["slice_index"]),
        frame_index=record.get("frame_index"),
        frame_axis=record.get("frame_axis"),
    )
    expected = [int(value) for value in record["context_indices"]]
    depth = int(record["depth"])
    observed = [max(0, min(depth - 1, int(record["slice_index"]) + offset)) for offset in (-1, 0, 1)]
    if expected != observed:
        raise SharedSpatialError("record context_indices disagree with canonical endpoint replication")
    if source_info.get("sha256") is None:
        raise SharedSpatialError("source image hash is required")
    return np.ascontiguousarray(stack, dtype=np.float32)


def normalize_self_audit_volume(volume: np.ndarray) -> np.ndarray:
    """Apply the checked-in Self-Audit image-only volume preprocessing.

    This is deliberately a source-protocol transform, not a CUTS or DFC
    adapter transform.  The paired Self-Audit loader applies the same
    0.5/99.5 clipping followed by a volume-wise population z-score before it
    constructs the endpoint-replicated context.  Keeping it here lets the
    image-only projection use the raw NIfTI mount without fitting statistics
    separately per baseline or per slice.
    """
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise SharedSpatialError(f"Self-Audit normalization expects a rank-3 volume, got {array.shape}")
    finite = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(finite, (0.5, 99.5))
    clipped = np.clip(finite, low, high)
    mean = float(clipped.mean())
    std = float(clipped.std())
    return ((clipped - mean) / max(std, 1e-6)).astype(np.float32, copy=False)


def _normalize_preprocess_acdc_image(volume: np.ndarray) -> np.ndarray:
    """Reproduce the image branch of ``preprocess_acdc.normalize_zscore``.

    The historical script obtains NIfTI data as float64, normalizes before
    slice resampling, and writes into a float32 output array.  Do not replace
    this with :func:`normalize_self_audit_volume`: the latter deliberately
    repairs non-finite values and uses an epsilon denominator, as the loader
    does, while the original preprocessing function did neither.
    """
    array = np.asarray(volume)
    if array.ndim != 3:
        raise SharedSpatialError(f"historical preprocess expects a rank-3 volume, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise SharedSpatialError("historical preprocess source must be finite numeric image data")
    low, high = np.percentile(array, (0.5, 99.5))
    clipped = np.clip(array, low, high)
    mean = np.mean(clipped)
    std = np.std(clipped)
    return (clipped - mean) / std if float(std) > 0.0 else clipped - mean


def _resize_preprocess_acdc_volume_to_224(volume_zhw: np.ndarray) -> np.ndarray:
    """Reproduce ``preprocess_acdc.py``'s per-slice image resize exactly."""
    try:
        from skimage.transform import resize
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SharedSpatialError("scikit-image is required for the historical 224 ACDC baseline contract") from exc
    volume = np.asarray(volume_zhw)
    if volume.ndim != 3:
        raise SharedSpatialError(f"historical preprocess expects [Z,H,W], got {volume.shape}")
    # The original script allocates float32 and assigns every result into it.
    resized = np.empty((int(volume.shape[0]), 224, 224), dtype=np.float32)
    for z in range(int(volume.shape[0])):
        resized[z] = resize(
            volume[z], (224, 224), order=1, preserve_range=True,
            anti_aliasing=True, mode="reflect",
        )
    return resized


def _read_historical_preprocess_source_frame(
    source: Path, *, depth_axis: int, frame_index: int | None, frame_axis: int | None,
) -> tuple[np.ndarray, int | None]:
    """Read only image data with the dtype semantics of ``nib.get_fdata()``.

    The general shared decoder intentionally converts frames to float32.  That
    is correct for the current 256 loader contract but differs from the old
    preprocessing script, whose NIfTI image branch calls ``get_fdata()``
    before the first normalization.  This isolated helper preserves that
    difference without modifying the Self-Audit decoder.
    """
    name = source.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SharedSpatialError("nibabel is required for NIfTI historical preprocessing") from exc
        raw = np.asarray(nib.load(str(source)).get_fdata())
    elif name.endswith(".npy"):
        raw = np.asarray(np.load(source))
    else:
        raise SharedSpatialError(f"unsupported historical preprocess image container: {source.name}")
    if raw.ndim not in (3, 4):
        raise SharedSpatialError(f"historical preprocess expected rank-3/4 source, got {raw.shape}")
    if depth_axis not in (0, 1, 2):
        raise SharedSpatialError(f"invalid historical preprocess depth_axis: {depth_axis}")
    actual_frame_axis: int | None = None
    if raw.ndim == 4:
        actual_frame_axis = 3 if frame_axis is None else int(frame_axis)
        if actual_frame_axis < 0 or actual_frame_axis >= 4 or actual_frame_axis == depth_axis:
            raise SharedSpatialError("historical preprocess frame_axis is invalid")
        if frame_index is None or not 0 <= int(frame_index) < int(raw.shape[actual_frame_axis]):
            raise SharedSpatialError("historical preprocess frame_index is invalid")
        raw = np.take(raw, int(frame_index), axis=actual_frame_axis)
    if not np.issubdtype(raw.dtype, np.number):
        raise SharedSpatialError(f"historical preprocess image is non-numeric: {raw.dtype}")
    return np.asarray(raw), actual_frame_axis


def read_self_audit_context_stack(
    record: Mapping[str, Any], *, source_root: str | Path | None = None,
) -> np.ndarray:
    """Read a full image-only frame, normalize it by source-volume statistics,
    then return the canonical endpoint-replicated ``[z-1,z,z+1]`` stack.

    The legacy ``read_context_stack`` remains available for historical fixture
    contracts.  Scientific Self-Audit v3 consumers use this function so the
    source normalization is shared and method adapters add no ungrounded
    percentile, min/max, or uint8 scaling.
    """
    source = resolve_source_path(record, source_root)
    source_info = record["source"]
    frame, actual_frame_axis = read_source_frame(
        source,
        depth_axis=int(record["depth_axis"]),
        frame_index=record.get("frame_index"),
        frame_axis=record.get("frame_axis"),
        _return_frame_axis=True,
    )
    depth_axis = int(record["depth_axis"])
    if actual_frame_axis is not None and int(actual_frame_axis) < depth_axis:
        depth_axis -= 1
    if frame.ndim != 3 or depth_axis not in (0, 1, 2):
        raise SharedSpatialError(f"selected Self-Audit source frame has invalid shape/axis: {frame.shape}, {depth_axis}")
    volume = np.moveaxis(frame, depth_axis, 0)
    normalized = normalize_self_audit_volume(volume)
    z = int(record["slice_index"])
    depth = int(normalized.shape[0])
    if not 0 <= z < depth:
        raise SharedSpatialError(f"slice_index {z} outside normalized depth extent {depth}")
    context = [max(0, z - 1), z, min(depth - 1, z + 1)]
    expected = [int(value) for value in record["context_indices"]]
    if expected != context:
        raise SharedSpatialError("record context_indices disagree with canonical endpoint replication")
    if source_info.get("sha256") is None:
        raise SharedSpatialError("source image hash is required")
    return np.ascontiguousarray(normalized[context], dtype=np.float32)


def read_self_audit_historical_224_context_stack(
    record: Mapping[str, Any], *, source_root: str | Path | None = None,
) -> np.ndarray:
    """Read one image-only context after the historical 224 preprocessing path.

    The order is fixed and recorded in the v7 baseline freeze: source image
    normalization -> per-slice skimage bilinear resize to 224 -> loader volume
    normalization -> endpoint-replicated context.  No core Self-Audit code is
    called or modified here, and no annotation path is opened.
    """
    source = resolve_source_path(record, source_root)
    source_info = record["source"]
    declared_depth_axis = int(record["depth_axis"])
    frame, actual_frame_axis = _read_historical_preprocess_source_frame(
        source,
        depth_axis=declared_depth_axis,
        frame_index=record.get("frame_index"),
        frame_axis=record.get("frame_axis"),
    )
    depth_axis = declared_depth_axis
    if actual_frame_axis is not None and actual_frame_axis < depth_axis:
        depth_axis -= 1
    if frame.ndim != 3 or depth_axis not in (0, 1, 2):
        raise SharedSpatialError(
            f"selected historical source frame has invalid shape/axis: {frame.shape}, {depth_axis}"
        )
    source_normalized = _normalize_preprocess_acdc_image(np.moveaxis(frame, depth_axis, 0))
    resized = _resize_preprocess_acdc_volume_to_224(source_normalized)
    normalized = normalize_self_audit_volume(resized)
    z = int(record["slice_index"])
    depth = int(normalized.shape[0])
    if not 0 <= z < depth:
        raise SharedSpatialError(f"slice_index {z} outside historical normalized depth extent {depth}")
    context = [max(0, z - 1), z, min(depth - 1, z + 1)]
    expected = [int(value) for value in record["context_indices"]]
    if expected != context:
        raise SharedSpatialError("record context_indices disagree with canonical endpoint replication")
    if source_info.get("sha256") is None:
        raise SharedSpatialError("source image hash is required")
    return np.ascontiguousarray(normalized[context], dtype=np.float32)


def resize_values_to_grid(values: torch.Tensor, grid: Mapping[str, Any]) -> torch.Tensor:
    """Apply the frozen value-resampling primitive to full-FOV method inputs.

    Input normalization happens before this operation.  The support is all
    true, so this invokes the declared value-resampling semantics without
    importing FreeMask's fit/select/verify observation partition.
    """
    if values.ndim != 3:
        raise SharedSpatialError("values must have [C,H,W] shape")
    target = grid.get("target_hw")
    if not isinstance(target, list) or len(target) != 2:
        raise SharedSpatialError("shared grid must declare target_hw")
    if grid.get("version") not in {
        SPATIAL_CONTRACT_VERSION,
        SELF_AUDIT_SPATIAL_CONTRACT_VERSION,
        SELF_AUDIT_HISTORICAL_224_SPATIAL_CONTRACT_VERSION,
    }:
        raise SharedSpatialError("unsupported shared spatial contract version")
    if grid.get("version") == SELF_AUDIT_HISTORICAL_224_SPATIAL_CONTRACT_VERSION:
        if grid.get("forward_values") != "historical_preprocess_224_no_post_loader_resize":
            raise SharedSpatialError("historical 224 grid declares an unexpected value transform")
        if [int(values.shape[-2]), int(values.shape[-1])] != [int(target[0]), int(target[1])]:
            raise SharedSpatialError("historical 224 values must already match the frozen target grid")
        return values
    if grid.get("version") == SELF_AUDIT_SPATIAL_CONTRACT_VERSION:
        if grid.get("forward_values") != "bilinear_align_corners_false":
            raise SharedSpatialError("Self-Audit grid must declare bilinear_align_corners_false")
        resized = F.interpolate(
            values.unsqueeze(0).float(),
            size=(int(target[0]), int(target[1])),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return resized.to(dtype=values.dtype)
    support = torch.ones(values.shape[-2:], dtype=torch.bool, device=values.device)
    resized, resized_support = masked_resize(values, support, (int(target[0]), int(target[1])))
    if not bool(resized_support.all()):  # full FOV cannot legitimately lose support
        raise SharedSpatialError("full-FOV shared resize lost output support")
    return resized


def spatial_transform_for_record(record: Mapping[str, Any], grid: Mapping[str, Any]) -> dict[str, Any]:
    """Expose FreeMask-compatible inverse provenance without resampling data."""
    transform = inverse_transform_record(
        native_shape=[int(value) for value in record["native_shape"]],
        depth_axis=int(record["depth_axis"]),
        slice_index=int(record["slice_index"]),
        frame_index=record.get("frame_index"),
        native_hw=(int(record["native_hw"][0]), int(record["native_hw"][1])),
        unit_hw=(int(grid["target_hw"][0]), int(grid["target_hw"][1])),
        native_grid_export=bool(record["geometry"]["native_grid_export"]),
        frame_axis=record.get("frame_axis"),
        full_native_shape=[int(value) for value in record["stored_shape"]]
        if len(record["stored_shape"]) == 4
        else None,
    )
    # Keep the historical v1 inverse helper byte-compatible while making the
    # Self-Audit v2 forward interpolation explicit in every record.
    transform["forward_resize"]["mode_values"] = grid.get(
        "forward_values", transform["forward_resize"].get("mode_values")
    )
    transform["forward_resize"]["mode_masks"] = grid.get(
        "forward_masks", transform["forward_resize"].get("mode_masks")
    )
    return transform
