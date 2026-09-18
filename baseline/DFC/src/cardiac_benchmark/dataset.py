"""DFC central-slice loading over the common image-only projection."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from shared_benchmark.spatial import grid_hash, read_context_stack, resize_values_to_grid


class DataContractError(ValueError):
    pass


NORMALIZATION_VERSION = "dfc.cardiac.central_percentile_0p5_99p5_population_zscore.v1"


def normalize_central_slice(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) < 2:
        raise DataContractError("central MRI slice must be a 2-D image with both axes >=2")
    if not np.isfinite(array).all():
        raise DataContractError("MRI input contains NaN or Inf")
    lo, hi = np.percentile(array, [0.5, 99.5])
    clipped = np.clip(array, lo, hi).astype(np.float32, copy=False)
    mean, std = float(clipped.mean(dtype=np.float64)), float(clipped.std(dtype=np.float64, ddof=0))
    return ((clipped - mean) / max(std, 1e-6)).astype(np.float32, copy=False)


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
    """Decode shared context, use only its centre, then DFC-normalize and grid-resize."""
    try:
        central = read_context_stack(record, source_root=root)[1]
        normalized = normalize_central_slice(central)
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
        "shared_grid_hash": grid_hash(record["shared_grid"]), "normalization": NORMALIZATION_VERSION,
    }
