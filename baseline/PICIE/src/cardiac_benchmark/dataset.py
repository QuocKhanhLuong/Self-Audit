"""PiCIE image-only view over the common shared benchmark manifest."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from shared_benchmark.manifest import SharedManifestError
from shared_benchmark.spatial import grid_hash, read_context_stack, resize_values_to_grid


PROFILES = ("PICIE-2D",)
NORMALIZATION_VERSION = "picie.cardiac.context_percentile_imagenet_v1"


@dataclass(frozen=True)
class ImageOnlySample:
    image: torch.Tensor
    provenance: dict[str, Any]


def _normalize_image_only(stack: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(stack.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(finite, (0.5, 99.5))
    if not high > low:
        high = low + 1.0
    return np.clip((finite - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def _imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image - mean) / std


class PICIECardiacDataset(Dataset[ImageOnlySample]):
    """Three-channel PiCIE input; membership, context, FOV and grid are shared-owned."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        split: str,
        profile: str = "PICIE-2D",
        source_root: str | Path | None = None,
        target_hw: tuple[int, int] | None = None,
    ):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")
        self.profile = profile
        self.manifest_hash = str(manifest["manifest_hash"])
        self.grid = dict(manifest["shared_grid"])
        if target_hw is not None and tuple(target_hw) != tuple(self.grid["target_hw"]):
            raise SharedManifestError("PiCIE target shape may not override the shared benchmark grid")
        self.source_root = source_root or manifest.get("local_receipt", {}).get("local_source_root")
        self.records = [record for record in manifest["records"] if record["split"] == split]
        if not self.records:
            raise SharedManifestError(f"manifest has no {split} records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ImageOnlySample:
        record = self.records[index]
        stack = _normalize_image_only(read_context_stack(record, source_root=self.source_root))
        image = resize_values_to_grid(torch.from_numpy(stack), self.grid).float()
        image = _imagenet_normalize(image)
        provenance = {
            "dataset": record["dataset"], "sample_id": record["sample_id"], "patient_id": record["patient_id"],
            "study_id": record["study_id"], "volume_id": record["volume_id"],
            "split": record["split"], "frame_index": record["frame_index"],
            "slice_index": record["slice_index"], "context_indices": record["context_indices"],
            "source_image_hash": record["source"]["sha256"], "source_shape": record["native_hw"],
            "target_shape": self.grid["target_hw"], "spatial_transform": record["spatial_transform"],
            "manifest_hash": self.manifest_hash, "shared_grid_version": self.grid["version"],
            "shared_grid_hash": grid_hash(self.grid), "profile": self.profile,
            "normalization": NORMALIZATION_VERSION, "augmentation_policy": "none_scientific",
            "input_channels": int(image.shape[0]),
        }
        return ImageOnlySample(image=image, provenance=provenance)


def collate_image_only(samples: list[ImageOnlySample]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    return torch.stack([sample.image for sample in samples], dim=0), [sample.provenance for sample in samples]
