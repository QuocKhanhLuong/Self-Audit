"""Common volume and 2.5-D sample contracts for Self-Audit datasets."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CLASS_NAMES = ("Background", "RV", "MYO", "LV")
NUM_CLASSES = 4
PATIENT_RE = re.compile(r"^(patient[^_\-]+)", re.IGNORECASE)
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "training": ("training", "train"),
    "val": ("val", "validation", "valid"),
    "validation": ("validation", "val", "valid"),
    "test": ("test", "testing"),
    "testing": ("testing", "test"),
}


@dataclass(frozen=True)
class VolumeRecord:
    """One paired image/mask volume owned by exactly one patient."""

    case_id: str
    patient_id: str
    image_path: Path
    mask_path: Path
    split: str = ""
    spacing: tuple[float, ...] | None = None
    source_format: str = "npy"


Sample = dict[str, Any]


def patient_id_from_case_id(case_id: str | Path) -> str:
    """Extract a stable patient identity before any frame/phase suffix."""

    clean = _strip_known_suffixes(Path(str(case_id)).name)
    match = PATIENT_RE.match(clean)
    if match:
        return match.group(1)
    return re.split(r"[_\-]", clean, maxsplit=1)[0]


def _strip_known_suffixes(name: str) -> str:
    for suffix in (".nii.gz", ".nii", ".npy", ".npz"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_array(path: str | Path) -> tuple[np.ndarray, tuple[float, ...] | None]:
    """Load NPY/NPZ or NIfTI without making nibabel a hard import dependency."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volume file does not exist: {path}")
    lower = path.name.lower()
    if lower.endswith(".npy"):
        return np.asarray(np.load(path, mmap_mode="r")), None
    if lower.endswith(".npz"):
        payload = np.load(path)
        if not payload.files:
            raise ValueError(f"NPZ volume contains no arrays: {path}")
        return np.asarray(payload[payload.files[0]]), None
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "NIfTI input requires nibabel. Install the repository environment or "
                "preprocess volumes to paired .npy files first."
            ) from exc
        image = nib.as_closest_canonical(nib.load(str(path)))
        return np.asarray(image.get_fdata()), tuple(float(x) for x in image.header.get_zooms()[:3])
    raise ValueError(f"Unsupported volume format: {path}")


def infer_depth_axis(
    shape: Sequence[int],
    expected_slices: int | None = None,
    depth_axis: int | None = None,
) -> int:
    if len(shape) != 3:
        raise ValueError(f"Expected a 3-D shape, got {tuple(shape)}")
    if depth_axis is not None:
        if depth_axis not in (0, 1, 2):
            raise ValueError(f"depth_axis must be 0, 1, or 2, got {depth_axis}")
        return int(depth_axis)
    if expected_slices is not None:
        matches = [axis for axis, size in enumerate(shape) if int(size) == int(expected_slices)]
        if len(matches) == 1:
            return matches[0]
    small_axes = [axis for axis, size in enumerate(shape) if int(size) <= 64]
    if len(small_axes) == 1:
        return small_axes[0]
    return int(np.argmin(np.asarray(shape)))


def to_depth_first(
    volume: np.ndarray,
    *,
    expected_slices: int | None = None,
    depth_axis: int | None = None,
) -> np.ndarray:
    """Normalize a volume to ``[Z,H,W]``; no through-plane interpolation."""

    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"Expected [*,*,*] volume, got shape {array.shape}")
    axis = infer_depth_axis(array.shape, expected_slices=expected_slices, depth_axis=depth_axis)
    return np.moveaxis(array, axis, 0)


def reorder_spacing(
    spacing: Sequence[float] | None,
    depth_axis: int,
) -> tuple[float, ...] | None:
    """Reorder source-axis spacing to match a ``[Z,H,W]`` array."""

    if spacing is None:
        return None
    values = tuple(float(value) for value in spacing)
    if len(values) < 3:
        return None
    axis = int(depth_axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"depth_axis must be 0, 1, or 2, got {depth_axis}")
    return (values[axis],) + tuple(values[index] for index in range(3) if index != axis)


def percentile_clip_and_zscore(
    volume: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply volume-wise percentile clipping followed by a volume-wise z-score."""

    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected [Z,H,W], got {array.shape}")
    finite = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(finite, [float(lower_percentile), float(upper_percentile)])
    clipped = np.clip(finite, low, high)
    mean = float(clipped.mean())
    std = float(clipped.std())
    return ((clipped - mean) / max(std, float(eps))).astype(np.float32, copy=False)


def resize_in_plane(
    volume_zhw: np.ndarray,
    size: int | tuple[int, int],
    *,
    is_mask: bool = False,
) -> np.ndarray:
    """Resize only H/W and preserve Z exactly."""

    array = np.asarray(volume_zhw)
    if array.ndim != 3:
        raise ValueError(f"Expected [Z,H,W], got {array.shape}")
    if isinstance(size, int):
        target = (int(size), int(size))
    else:
        target = (int(size[0]), int(size[1]))
    if tuple(array.shape[-2:]) == target:
        return array.copy()
    tensor = torch.from_numpy(np.ascontiguousarray(array.astype(np.float32, copy=False))).unsqueeze(1)
    mode = "nearest" if is_mask else "bilinear"
    resized = F.interpolate(tensor, size=target, mode=mode, align_corners=False if mode != "nearest" else None)
    result = resized[:, 0].cpu().numpy()
    return result.astype(np.int64 if is_mask else np.float32, copy=False)


def build_25d_triplet(volume_zhw: np.ndarray, center_slice: int) -> np.ndarray:
    """Construct ``[z-1,z,z+1]`` with replicated boundary slices."""

    volume = np.asarray(volume_zhw)
    if volume.ndim != 3:
        raise ValueError(f"Expected [Z,H,W], got {volume.shape}")
    n_slices = int(volume.shape[0])
    if n_slices == 0:
        raise ValueError("Cannot build a 2.5-D sample from an empty volume")
    index = int(center_slice)
    if index < 0 or index >= n_slices:
        raise IndexError(f"center_slice {index} outside [0, {n_slices})")
    neighbors = (max(index - 1, 0), index, min(index + 1, n_slices - 1))
    return np.stack([volume[neighbor] for neighbor in neighbors], axis=0).astype(np.float32, copy=False)


def resize_sample(image: torch.Tensor, mask: torch.Tensor, size: int | tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize a 2.5-D image and center mask with matching geometry."""

    if isinstance(size, int):
        target = (int(size), int(size))
    else:
        target = (int(size[0]), int(size[1]))
    if tuple(image.shape[-2:]) != target:
        image = F.interpolate(image.unsqueeze(0).float(), size=target, mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask[None, None].float(), size=target, mode="nearest").squeeze(0).squeeze(0).long()
    return image, mask


def validate_patient_split(case_ids_by_split: Mapping[str, Iterable[str]]) -> None:
    """Fail loudly if any patient occurs in more than one split."""

    patients: dict[str, str] = {}
    for split, case_ids in case_ids_by_split.items():
        for case_id in case_ids:
            patient = patient_id_from_case_id(case_id)
            previous = patients.get(patient)
            if previous is not None and previous != split:
                raise ValueError(
                    f"Patient leakage: {patient!r} appears in both {previous!r} and {split!r}"
                )
            patients[patient] = str(split)


def patient_level_split(
    case_ids: Iterable[str],
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Deterministically split cases by patient, never by individual slices."""

    if not 0.0 < train_fraction < 1.0 or not 0.0 <= val_fraction < 1.0:
        raise ValueError("train_fraction and val_fraction must be valid fractions")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")
    grouped: dict[str, list[str]] = {}
    for case_id in sorted(set(str(value) for value in case_ids)):
        grouped.setdefault(patient_id_from_case_id(case_id), []).append(case_id)
    patients = sorted(grouped)
    if len(patients) < 3:
        raise ValueError("Patient-level train/val/test split requires at least three patients")
    rng = np.random.default_rng(int(seed))
    shuffled = list(patients)
    rng.shuffle(shuffled)
    n_train = min(max(int(round(len(patients) * train_fraction)), 1), len(patients) - 2)
    n_val = min(max(int(round(len(patients) * val_fraction)), 1), len(patients) - n_train - 1)
    split_patients = {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }
    result = {
        split: sorted(case_id for patient in patient_ids for case_id in grouped[patient])
        for split, patient_ids in split_patients.items()
    }
    validate_patient_split(result)
    return result


def read_split_manifest(path: str | Path) -> dict[str, list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    result: dict[str, list[str]] = {}
    for split, aliases in (("train", ("train", "training")), ("val", ("val", "validation")), ("test", ("test", "testing"))):
        values: list[str] = []
        for alias in aliases:
            direct = payload.get(f"{alias}_cases") or payload.get(f"{alias}_volumes")
            if direct:
                values = [_strip_known_suffixes(Path(str(item)).name) for item in direct]
                break
            nested = payload.get("splits", {}).get(alias, {}) if isinstance(payload.get("splits"), dict) else {}
            if isinstance(nested, dict) and (nested.get("cases") or nested.get("volumes")):
                values = [_strip_known_suffixes(Path(str(item)).name) for item in (nested.get("cases") or nested.get("volumes"))]
                break
        if values:
            result[split] = sorted(set(values))
    if not result:
        raise ValueError(f"Split manifest contains no recognized case lists: {path}")
    validate_patient_split(result)
    return result


class VolumeSliceDataset(Dataset):
    """Shared implementation for ACDC and M&Ms sample construction."""

    def __init__(
        self,
        records: Sequence[VolumeRecord],
        *,
        image_size: int | tuple[int, int] = 256,
        augment: bool = False,
        transform: Any | None = None,
        foreground_only: bool = False,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        max_cache: int = 4,
        depth_axis: int | None = None,
        expected_slices: int | None = None,
    ) -> None:
        if not records:
            raise ValueError("Dataset has no volume records")
        self.records = list(records)
        self.image_size = image_size
        self.augment = bool(augment)
        self.transform = transform
        self.foreground_only = bool(foreground_only)
        self.lower_percentile = float(lower_percentile)
        self.upper_percentile = float(upper_percentile)
        self.max_cache = max(int(max_cache), 1)
        self.depth_axis = depth_axis
        self.expected_slices = expected_slices
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray, tuple[float, ...] | None]] = OrderedDict()
        self.index_map: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            volume, volume_spacing = load_array(record.image_path)
            mask, mask_spacing = load_array(record.mask_path)
            axis = self.depth_axis if self.depth_axis is not None else (2 if record.source_format == "nifti" else None)
            source_volume_axis = axis if axis is not None else infer_depth_axis(volume.shape, expected_slices=self.expected_slices)
            volume = to_depth_first(volume, expected_slices=self.expected_slices, depth_axis=axis)
            mask = to_depth_first(mask, expected_slices=volume.shape[0], depth_axis=axis)
            if volume.shape != mask.shape:
                raise ValueError(f"Image/mask shape mismatch for {record.case_id}: {volume.shape} vs {mask.shape}")
            for slice_index in range(volume.shape[0]):
                if self.foreground_only and not np.any(mask[slice_index] > 0):
                    continue
                self.index_map.append((record_index, slice_index))
            spacing = record.spacing or reorder_spacing(volume_spacing, source_volume_axis) or reorder_spacing(mask_spacing, source_volume_axis)
            self.records[record_index] = VolumeRecord(
                **{**record.__dict__, "spacing": spacing}
            )
        if not self.index_map:
            raise ValueError("Dataset contains no slices after filtering")

    def __len__(self) -> int:
        return len(self.index_map)

    def _load(self, record_index: int) -> tuple[np.ndarray, np.ndarray, tuple[float, ...] | None]:
        if record_index in self._cache:
            self._cache.move_to_end(record_index)
            return self._cache[record_index]
        record = self.records[record_index]
        volume, volume_spacing = load_array(record.image_path)
        mask, mask_spacing = load_array(record.mask_path)
        axis = self.depth_axis if self.depth_axis is not None else (2 if record.source_format == "nifti" else None)
        source_volume_axis = axis if axis is not None else infer_depth_axis(volume.shape, expected_slices=self.expected_slices)
        volume = to_depth_first(volume, expected_slices=self.expected_slices, depth_axis=axis)
        mask = to_depth_first(mask, expected_slices=volume.shape[0], depth_axis=axis)
        if volume.shape != mask.shape:
            raise ValueError(f"Image/mask shape mismatch for {record.case_id}: {volume.shape} vs {mask.shape}")
        volume = percentile_clip_and_zscore(volume, self.lower_percentile, self.upper_percentile)
        spacing = record.spacing or reorder_spacing(volume_spacing, source_volume_axis) or reorder_spacing(mask_spacing, source_volume_axis)
        value = (volume, mask.astype(np.int64, copy=False), spacing)
        self._cache[record_index] = value
        if len(self._cache) > max(int(self._cache_size), 1):
            self._cache.popitem(last=False)
        return value

    @property
    def _cache_size(self) -> int:
        return self.max_cache

    def __getitem__(self, index: int) -> Sample:
        record_index, slice_index = self.index_map[int(index)]
        volume, mask, spacing = self._load(record_index)
        image = torch.from_numpy(np.ascontiguousarray(build_25d_triplet(volume, slice_index))).float()
        target = torch.from_numpy(np.array(mask[slice_index], copy=True, order="C")).long()
        image, target = resize_sample(image, target, self.image_size)
        sample: Sample = {
            "image": image,
            "mask": target,
            "case_id": self.records[record_index].case_id,
            "patient_id": self.records[record_index].patient_id,
            "slice_idx": int(slice_index),
            "num_slices": int(volume.shape[0]),
            # Default-unit spacing keeps the shared dictionary collatable by
            # PyTorch's default DataLoader while remaining explicit that no
            # physical spacing was available in an NPY-only layout.
            "spacing": spacing if spacing is not None else (1.0, 1.0, 1.0),
            "spacing_known": bool(spacing is not None),
            "spacing_units": "mm" if spacing is not None else "pixel",
        }
        if self.transform is not None:
            sample = self.transform(sample)
        elif self.augment:
            from .transforms import RandomGeometricTransform

            sample = RandomGeometricTransform()(sample)
        return sample


# Readable aliases used by small protocol/integration scripts.
build_25d_sample = build_25d_triplet
construct_25d_input = build_25d_triplet
normalize_volume = percentile_clip_and_zscore
