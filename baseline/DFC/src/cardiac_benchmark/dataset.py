"""DFC central-slice loading over the common image-only projection."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from shared_benchmark.spatial import (
    SELF_AUDIT_HISTORICAL_224_NORMALIZATION_VERSION,
    SELF_AUDIT_HISTORICAL_224_SPATIAL_CONTRACT_VERSION,
    SELF_AUDIT_NORMALIZATION_VERSION,
    SELF_AUDIT_SPATIAL_CONTRACT_VERSION,
    grid_hash,
    read_context_stack,
    read_self_audit_historical_224_context_stack,
    read_self_audit_context_stack,
    resize_values_to_grid,
)


class DataContractError(ValueError):
    pass


NORMALIZATION_VERSION = SELF_AUDIT_NORMALIZATION_VERSION
LEGACY_NORMALIZATION_VERSION = "source.float32_identity.legacy_fixture.v1"


def prepare_central_slice(image: np.ndarray) -> np.ndarray:
    """Validate a source-normalized central plane without re-normalizing it.

    Upstream DFC's direct demo only divides decoded uint8 BGR images by 255;
    it does not define a percentile or mean/std transform.  Scientific ACDC
    records therefore arrive already normalized by the shared Self-Audit
    volume contract.  This boundary must remain an identity operation.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) < 2:
        raise DataContractError("central MRI slice must be a 2-D image with both axes >=2")
    if not np.isfinite(array).all():
        raise DataContractError("MRI input contains NaN or Inf")
    return np.ascontiguousarray(array)


# Compatibility name for test/consumer code written against the pre-v3
# adapter.  It intentionally performs no transform now.
normalize_central_slice = prepare_central_slice


def apply_shared_grid(image: np.ndarray, grid: Mapping[str, Any]) -> np.ndarray:
    if np.asarray(image).ndim != 2:
        raise DataContractError("shared grid receives one central 2-D image")
    try:
        return resize_values_to_grid(torch.from_numpy(np.ascontiguousarray(image))[None], grid)[0].numpy()
    except Exception as exc:
        raise DataContractError(str(exc)) from exc


def load_primary_2d(
    record: Mapping[str, Any], root: str | Path | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decode the source-normalized context, use only its centre, and resize."""
    try:
        grid_version = record.get("shared_grid", {}).get("version")
        if grid_version == SELF_AUDIT_SPATIAL_CONTRACT_VERSION:
            stack = read_self_audit_context_stack(record, source_root=root)
            normalization = SELF_AUDIT_NORMALIZATION_VERSION
        elif grid_version == SELF_AUDIT_HISTORICAL_224_SPATIAL_CONTRACT_VERSION:
            stack = read_self_audit_historical_224_context_stack(record, source_root=root)
            normalization = SELF_AUDIT_HISTORICAL_224_NORMALIZATION_VERSION
        else:
            stack = read_context_stack(record, source_root=root)
            normalization = LEGACY_NORMALIZATION_VERSION
        normalized = prepare_central_slice(stack[1])
        shared = apply_shared_grid(normalized, record["shared_grid"])
    except Exception as exc:
        if isinstance(exc, DataContractError):
            raise
        raise DataContractError(str(exc)) from exc
    tensor = torch.from_numpy(np.ascontiguousarray(shared))[None, None]
    return tensor, {
        "sample_id": record["sample_id"], "patient_id": record["patient_id"], "split": record["split"],
        "frame_index": record["frame_index"], "slice_index": record["slice_index"], "central_slice": record["slice_index"],
        "context_indices": record["context_indices"], "source_image_hash": record["source"]["sha256"],
        "source_shape": record["native_hw"], "target_shape": record["shared_grid"]["target_hw"],
        "spatial_transform": record["spatial_transform"], "shared_grid_version": record["shared_grid"]["version"],
        "shared_grid_hash": grid_hash(record["shared_grid"]),
        "normalization": normalization,
    }
