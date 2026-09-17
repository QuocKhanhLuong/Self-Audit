"""Image-only ACDC/M&Ms data loader over a frozen shared manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .manifest import ManifestError

PROFILES = ("CUTS-2D", "CUTS-2.5D")


@dataclass(frozen=True)
class ImageOnlySample:
    image: torch.Tensor
    provenance: dict[str, Any]


def _read_array(path: str) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        return np.asarray(np.load(source, allow_pickle=False))
    if source.suffix == ".npz":
        archive = np.load(source, allow_pickle=False)
        if len(archive.files) != 1:
            raise ManifestError("npz source must have exactly one image array")
        return np.asarray(archive[archive.files[0]])
    if source.name.endswith(".nii") or source.name.endswith(".nii.gz"):
        import nibabel as nib
        return np.asarray(nib.load(str(source)).dataobj)
    raise ManifestError(f"unsupported image-only source type: {source}")


def _select_frame(array: np.ndarray, frame_index: int | None, frame_axis: int | None, depth_axis: int) -> tuple[np.ndarray, int]:
    if frame_axis is None:
        if array.ndim != 3:
            raise ManifestError("rank-4 source requires manifest frame_axis")
        return array, depth_axis
    if frame_index is None:
        raise ManifestError("frame_axis requires frame_index")
    if array.ndim != 4:
        raise ManifestError("frame_axis present for non-rank-4 source")
    if not 0 <= int(frame_index) < array.shape[int(frame_axis)]:
        raise ManifestError("frame_index outside source")
    selected = np.take(array, int(frame_index), axis=int(frame_axis))
    return selected, int(depth_axis) - int(frame_axis < depth_axis)


def _slice_stack(record: dict[str, Any]) -> np.ndarray:
    array = _read_array(record["source_path"])
    volume, depth_axis = _select_frame(array, record.get("frame_index"), record.get("frame_axis"), int(record["depth_axis"]))
    if volume.ndim != 3:
        raise ManifestError("selected image volume must be rank 3")
    depth = volume.shape[depth_axis]
    requested = [int(v) for v in record["context_indices"]]
    clipped = [min(depth - 1, max(0, v)) for v in requested]
    if clipped[1] != int(record["slice_index"]):
        raise ManifestError("manifest central context index differs from slice_index")
    planes = [np.take(volume, index, axis=depth_axis).astype(np.float32, copy=False) for index in clipped]
    return np.stack(planes, axis=0)


def _normalize_image_only(stack: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(stack.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(finite, (0.5, 99.5))
    if not high > low:
        high = low + 1.0
    return np.clip((finite - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def _resize(image: torch.Tensor, target_hw: tuple[int, int] | None) -> torch.Tensor:
    if target_hw is None or tuple(image.shape[-2:]) == tuple(target_hw):
        return image
    return F.interpolate(image.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)


class ImageOnlyCardiacDataset(Dataset[ImageOnlySample]):
    """Dataset that returns image tensors and manifest-derived provenance only."""

    def __init__(self, manifest: dict[str, Any], *, split: str, profile: str, target_hw: tuple[int, int] | None = None):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        self.profile = profile
        self.target_hw = target_hw
        self.manifest_hash = manifest["manifest_hash"]
        self.records = [record for record in manifest["records"] if record["split"] == split]
        if not self.records:
            raise ManifestError(f"manifest has no {split} records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ImageOnlySample:
        record = self.records[index]
        stack = _normalize_image_only(_slice_stack(record))
        image = torch.from_numpy(stack[1:2] if self.profile == "CUTS-2D" else stack)
        image = _resize(image, self.target_hw)
        provenance = {key: value for key, value in record.items() if key not in {"source_path"}}
        provenance.update({"manifest_hash": self.manifest_hash, "profile": self.profile, "normalization": "per_input_p0_p5_clip_to_unit_interval"})
        return ImageOnlySample(image=image, provenance=provenance)


def collate_image_only(samples: list[ImageOnlySample]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    return torch.stack([sample.image for sample in samples], dim=0), [sample.provenance for sample in samples]
