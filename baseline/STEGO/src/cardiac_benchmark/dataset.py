"""STEGO image-only view over the common shared benchmark manifest."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from shared_benchmark.manifest import SharedManifestError
from shared_benchmark.spatial import (
    SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION,
    grid_hash,
    read_context_stack,
    read_self_audit_context_stack,
    resize_values_to_grid,
)


PROFILES = ("STEGO-2D", "STEGO-SA224", "STEGO-SA224-FAIR")
NORMALIZATION_VERSION = "stego.cardiac.central_fixed_affine_v1"
SA224_NORMALIZATION_VERSION = "stego.cardiac.sa224_self_audit_volume_then_central_fixed_affine_v1"


@dataclass(frozen=True)
class ImageOnlySample:
    image: torch.Tensor
    provenance: dict[str, Any]


def fixed_affine_normalize(plane: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(plane.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    clipped = np.clip(finite, -3.0, 3.0)
    return ((clipped + 3.0) / 6.0 * 255.0).astype(np.uint8, copy=False)


def fixed_affine_tensor(plane: torch.Tensor) -> torch.Tensor:
    finite = torch.nan_to_num(plane.float(), nan=0.0, posinf=0.0, neginf=0.0)
    clipped = finite.clamp(-3.0, 3.0)
    return (clipped + 3.0) / 6.0 * 255.0


class STEGOCardiacDataset(Dataset[ImageOnlySample]):
    """Central-slice STEGO input; membership, context, FOV and grid are shared-owned."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        split: str,
        profile: str = "STEGO-2D",
        source_root: str | Path | None = None,
        target_hw: tuple[int, int] | None = None,
        resolution: int | None = None,
    ):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")
        self.profile = profile
        self.manifest_hash = str(manifest["manifest_hash"])
        self.grid = dict(manifest["shared_grid"])
        expected_hw = tuple(int(v) for v in self.grid["target_hw"])
        requested_hw = target_hw or ((int(resolution), int(resolution)) if resolution is not None else None)
        if requested_hw is not None and tuple(requested_hw) != expected_hw:
            raise SharedManifestError("STEGO target shape may not override the shared benchmark grid")
        self.source_root = source_root or manifest.get("local_receipt", {}).get("local_source_root")
        self.records = [record for record in manifest["records"] if record["split"] == split]
        if not self.records:
            raise SharedManifestError(f"manifest has no {split} records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ImageOnlySample:
        record = self.records[index]
        if self.profile in {"STEGO-SA224", "STEGO-SA224-FAIR"}:
            if self.grid.get("version") != SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION:
                raise SharedManifestError("STEGO-SA224 requires the Self-Audit compat 224 shared grid")
            stack = read_self_audit_context_stack(record, source_root=self.source_root)
            central = resize_values_to_grid(torch.from_numpy(stack[1:2]), self.grid)
            image = fixed_affine_tensor(central).repeat(3, 1, 1).float()
            normalization_version = SA224_NORMALIZATION_VERSION
        else:
            stack = read_context_stack(record, source_root=self.source_root)
            central = fixed_affine_normalize(stack[1])
            central_tensor = torch.from_numpy(central[None]).float()
            image = resize_values_to_grid(central_tensor, self.grid).repeat(3, 1, 1).float()
            normalization_version = NORMALIZATION_VERSION
        provenance = {
            "dataset": record["dataset"], "sample_id": record["sample_id"], "patient_id": record["patient_id"],
            "study_id": record["study_id"], "volume_id": record["volume_id"],
            "split": record["split"], "frame_index": record["frame_index"],
            "slice_index": record["slice_index"], "context_indices": record["context_indices"],
            "source_image_hash": record["source"]["sha256"], "source_shape": record["native_hw"],
            "target_shape": self.grid["target_hw"], "spatial_transform": record["spatial_transform"],
            "manifest_hash": self.manifest_hash, "shared_grid_version": self.grid["version"],
            "shared_grid_hash": grid_hash(self.grid), "profile": self.profile,
            "normalization": normalization_version, "input_channels": 3,
            "benchmark_tier": "fair" if self.profile.endswith("-FAIR") else "compat",
        }
        return ImageOnlySample(image=image, provenance=provenance)


def collate_image_only(samples: list[ImageOnlySample]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    return torch.stack([sample.image for sample in samples], dim=0), [sample.provenance for sample in samples]
