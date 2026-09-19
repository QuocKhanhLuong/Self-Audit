"""Image-only ACDC/M&Ms data loader for STEGO over a frozen shared manifest.

Normalization: Fixed Affine — clip [-3.0, 3.0] → scale to [0, 255] uint8,
replicate grayscale → 3-channel RGB. No per-image min-max normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .manifest import ManifestError

PROFILES = ("STEGO-2D",)
NORMALIZATION_VERSION = "stego.cardiac.fixed_affine_v1"


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


def _select_frame(
    array: np.ndarray, frame_index: int | None, frame_axis: int | None, depth_axis: int
) -> tuple[np.ndarray, int]:
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


def _extract_center_slice(record: dict[str, Any]) -> np.ndarray:
    array = _read_array(record["source_path"])
    volume, depth_axis = _select_frame(
        array, record.get("frame_index"), record.get("frame_axis"), int(record["depth_axis"])
    )
    if volume.ndim != 3:
        raise ManifestError("selected image volume must be rank 3")
    center_z = int(record["slice_index"])
    depth = volume.shape[depth_axis]
    if not 0 <= center_z < depth:
        raise ManifestError("slice_index outside volume depth")
    plane = np.take(volume, center_z, axis=depth_axis).astype(np.float32, copy=False)
    return plane


def fixed_affine_normalize(plane: np.ndarray) -> np.ndarray:
    """Fixed Affine: clip [-3.0, 3.0] → scale to [0, 255] → uint8.

    Input is a z-score float from the shared pipeline.
    Output is uint8 [0, 255] — NOT per-image min-max.
    """
    finite = np.nan_to_num(plane.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    clipped = np.clip(finite, -3.0, 3.0)
    scaled = ((clipped + 3.0) / 6.0 * 255.0).astype(np.uint8)
    return scaled


class STEGOCardiacDataset(Dataset[ImageOnlySample]):
    """Image-only dataset: returns 3-channel uint8→float tensor and provenance.

    No GT/label access. Fixed Affine normalization.
    Resolution: 224×224, nearest-neighbor interpolation.
    """

    def __init__(
        self,
        manifest: dict[str, Any],
        *,
        split: str,
        profile: str = "STEGO-2D",
        resolution: int = 224,
    ):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        self.profile = profile
        self.resolution = resolution
        self.manifest_hash = manifest["manifest_hash"]
        self.records = [record for record in manifest["records"] if record["split"] == split]
        if not self.records:
            raise ManifestError(f"manifest has no {split} records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ImageOnlySample:
        record = self.records[index]
        plane = _extract_center_slice(record)
        normalized = fixed_affine_normalize(plane)
        rgb = np.stack([normalized, normalized, normalized], axis=0)
        image = torch.from_numpy(rgb).float()
        if image.shape[-2] != self.resolution or image.shape[-1] != self.resolution:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode="nearest",
            ).squeeze(0)
        provenance = {key: value for key, value in record.items() if key != "source_path"}
        provenance.update({
            "manifest_hash": self.manifest_hash,
            "profile": self.profile,
            "normalization": NORMALIZATION_VERSION,
            "input_channels": 3,
            "resolution": self.resolution,
        })
        return ImageOnlySample(image=image, provenance=provenance)


def collate_image_only(samples: list[ImageOnlySample]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    return torch.stack([s.image for s in samples], dim=0), [s.provenance for s in samples]
