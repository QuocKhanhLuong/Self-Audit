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
SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION = "shared_benchmark.spatial.self_audit_compat_224.v1"
SELF_AUDIT_NORMALIZATION_VERSION = "self_audit.volume_percentile_clip_0p5_99p5_zscore.v1"


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


def load_self_audit_compat_224_grid_spec(repo_root: str | Path) -> dict[str, Any]:
    """Load the 224x224 compat contract for STEGO/PiCIE over Self-Audit data.

    This contract deliberately combines the Self-Audit source normalization and
    whole-FOV decoder with the historical 224x224 benchmark grid used by the
    STEGO/PiCIE cardiac ports. It is separate from both the historical
    FreeMask v1 grid and the Self-Audit 256x256 contract so merges cannot
    silently reinterpret STEGO/PiCIE inputs.
    """
    root = Path(repo_root)
    self_audit_grid = load_self_audit_grid_spec(root)
    pinned_grid = load_pinned_grid_spec(root)
    target_hw = tuple(int(value) for value in pinned_grid["target_hw"])
    if target_hw != (224, 224):
        raise SharedSpatialError(f"STEGO/PiCIE compat grid must remain 224x224, got {target_hw}")
    provenance = {
        "source": "Self-Audit normalized ACDC source with STEGO/PiCIE compat 224 benchmark grid",
        "self_audit_protocol": dict(self_audit_grid["config_provenance"]),
        "compat_grid": {
            "source": pinned_grid["config_provenance"]["source"],
            "config_files": list(pinned_grid["config_provenance"]["config_files"]),
            "config_sha256": dict(pinned_grid["config_provenance"]["config_sha256"]),
            "image_size": 224,
        },
    }
    return build_grid_spec(
        target_hw,
        config_provenance=provenance,
        version=SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION,
        forward_values="bilinear_align_corners_false",
        forward_masks="nearest_exact",
        inverse_labels="nearest_exact",
        inverse_probabilities="bilinear_then_renormalize",
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
        SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION,
    }:
        raise SharedSpatialError("unsupported shared spatial contract version")
    if grid.get("version") in {
        SELF_AUDIT_SPATIAL_CONTRACT_VERSION,
        SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION,
    }:
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
