"""Shared whole-FOV target-grid contract backed by FreeMask resampling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from self_audit_maskfree.data.geometry import inverse_transform_record, read_slice_stack
from self_audit_maskfree.data.geometry import masked_resize

from .firewall import FirewallError, validate_image_only_source_locator
from .provenance import sha256_file, sha256_json


SPATIAL_CONTRACT_VERSION = "shared_benchmark.spatial.v1"


class SharedSpatialError(ValueError):
    """Raised for an invalid common grid or source/grid incompatibility."""


def build_grid_spec(
    target_hw: tuple[int, int], *, config_provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Create the canonical whole-FOV contract without prescribing intensity stats."""
    if len(target_hw) != 2 or any(int(value) < 2 for value in target_hw):
        raise SharedSpatialError("target grid must be [H,W] with each dimension >= 2")
    return {
        "version": SPATIAL_CONTRACT_VERSION,
        "target_hw": [int(target_hw[0]), int(target_hw[1])],
        "whole_fov": True,
        "crop": None,
        "crop_rule": "whole_field_of_view_no_image_informed_crop",
        "forward_values": "masked_area_normalized_convolution",
        "forward_masks": "nearest_exact",
        "inverse_labels": "nearest_exact",
        "inverse_probabilities": "bilinear_then_renormalize",
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
    """Use FreeMask's canonical rank-3/rank-4 image-only decoder and context rule."""
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


def resize_values_to_grid(values: torch.Tensor, grid: Mapping[str, Any]) -> torch.Tensor:
    """Apply FreeMask's actual masked-area primitive to full-FOV method inputs.

    Method-specific normalization happens before this operation.  The support is
    all true, so this invokes the same value-resampling semantics without
    importing FreeMask's fit/select/verify observation partition.
    """
    if values.ndim != 3:
        raise SharedSpatialError("values must have [C,H,W] shape")
    target = grid.get("target_hw")
    if not isinstance(target, list) or len(target) != 2:
        raise SharedSpatialError("shared grid must declare target_hw")
    if grid.get("version") != SPATIAL_CONTRACT_VERSION:
        raise SharedSpatialError("unsupported shared spatial contract version")
    support = torch.ones(values.shape[-2:], dtype=torch.bool, device=values.device)
    resized, resized_support = masked_resize(values, support, (int(target[0]), int(target[1])))
    if not bool(resized_support.all()):  # full FOV cannot legitimately lose support
        raise SharedSpatialError("full-FOV shared resize lost output support")
    return resized


def spatial_transform_for_record(record: Mapping[str, Any], grid: Mapping[str, Any]) -> dict[str, Any]:
    """Expose FreeMask-compatible inverse provenance without resampling data."""
    return inverse_transform_record(
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
