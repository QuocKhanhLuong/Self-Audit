"""CUTS image preparation over the common image-only manifest projection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from shared_benchmark.manifest import SharedManifestError
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


PROFILES = ("CUTS-2D", "CUTS-2.5D")
LEGACY_NORMALIZATION_VERSION = "source.float32_identity.legacy_fixture.v1"
# These aliases are retained for callers that inspect the old profile names;
# scientific v3 provenance now records the source Self-Audit transform.
NORMALIZATION_VERSION_2D = SELF_AUDIT_NORMALIZATION_VERSION
NORMALIZATION_VERSION_25D = SELF_AUDIT_NORMALIZATION_VERSION
NORMALIZATION_VERSION = SELF_AUDIT_NORMALIZATION_VERSION


@dataclass(frozen=True)
class ImageOnlySample:
    image: torch.Tensor
    provenance: dict[str, Any]


def _prepare_profile_values(
    stack: np.ndarray, profile: str, *, normalization: str,
) -> tuple[np.ndarray, str]:
    """Select CUTS channels without adding a method-specific normalizer.

    The source Self-Audit volume transform is applied before this function for
    v3 records.  The legacy identity branch exists only for old synthetic
    fixtures and is never accepted by the scientific runner.
    """
    if profile == "CUTS-2D":
        return stack[1:2], normalization
    if profile == "CUTS-2.5D":
        return stack, normalization
    raise ValueError(f"profile must be one of {PROFILES}")


class ImageOnlyCardiacDataset(Dataset[ImageOnlySample]):
    """CUTS views; membership, context, FOV and grid remain shared-owned."""

    def __init__(
        self, manifest: Mapping[str, Any], *, split: str, profile: str,
        target_hw: tuple[int, int] | None = None, source_root: str | Path | None = None,
    ):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")
        self.profile = profile
        self.manifest_hash = str(manifest["manifest_hash"])
        self.grid = dict(manifest["shared_grid"])
        if target_hw is not None and list(target_hw) != self.grid["target_hw"]:
            raise SharedManifestError("CUTS target_hw may not override the shared benchmark grid")
        self.source_root = source_root or manifest.get("local_receipt", {}).get("local_source_root")
        self.records = [record for record in manifest["records"] if record["split"] == split]
        if not self.records:
            raise SharedManifestError(f"manifest has no {split} records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ImageOnlySample:
        record = self.records[index]
        grid_version = self.grid.get("version")
        if grid_version == SELF_AUDIT_SPATIAL_CONTRACT_VERSION:
            stack = read_self_audit_context_stack(record, source_root=self.source_root)
            normalization = SELF_AUDIT_NORMALIZATION_VERSION
        elif grid_version == SELF_AUDIT_HISTORICAL_224_SPATIAL_CONTRACT_VERSION:
            stack = read_self_audit_historical_224_context_stack(record, source_root=self.source_root)
            normalization = SELF_AUDIT_HISTORICAL_224_NORMALIZATION_VERSION
        else:
            stack = read_context_stack(record, source_root=self.source_root)
            normalization = LEGACY_NORMALIZATION_VERSION
        values, normalization_version = _prepare_profile_values(
            stack, self.profile, normalization=normalization,
        )
        image = resize_values_to_grid(torch.from_numpy(values), self.grid)
        provenance = {
            "dataset": record["dataset"], "sample_id": record["sample_id"], "patient_id": record["patient_id"],
            "study_id": record["study_id"], "volume_id": record["volume_id"],
            "split": record["split"], "frame_index": record["frame_index"],
            "slice_index": record["slice_index"], "context_indices": record["context_indices"],
            "source_image_hash": record["source"]["sha256"], "source_shape": record["native_hw"],
            "target_shape": self.grid["target_hw"], "spatial_transform": record["spatial_transform"],
            "manifest_hash": self.manifest_hash, "shared_grid_version": self.grid["version"],
            "shared_grid_hash": grid_hash(self.grid), "profile": self.profile,
            "normalization": normalization_version, "input_channels": int(image.shape[0]),
        }
        return ImageOnlySample(image=image, provenance=provenance)


def collate_image_only(samples: list[ImageOnlySample]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    return torch.stack([sample.image for sample in samples], dim=0), [sample.provenance for sample in samples]
