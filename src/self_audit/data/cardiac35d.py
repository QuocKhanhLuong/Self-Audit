"""Reusable lazy 2.5-D sample construction for cardiac MRI adapters."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    build_25d_triplet,
    percentile_clip_and_zscore,
    resize_sample,
)


@dataclass(frozen=True)
class CardiacUnit:
    """One logical 3-D spatial unit presented to the common sample builder."""

    dataset: str
    case_id: str
    subject_id: str
    image_ref: Any
    mask_ref: Any | None
    split: str = ""
    phase: str = ""
    time_index: int = -1
    acquisition_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LoadedCardiacUnit:
    """Adapter output in [Z,H,W] geometry before shared preprocessing."""

    image_zhw: np.ndarray
    mask_zhw: np.ndarray
    spacing: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Cardiac35DAdapter:
    """Small adapter interface consumed by Cardiac35DSliceDataset."""

    dataset_name = "unknown"

    def discover_units(
        self,
        *,
        split: str | None = None,
        supervised_only: bool = True,
    ) -> Sequence[CardiacUnit]:
        raise NotImplementedError

    def load_unit(self, unit: CardiacUnit) -> LoadedCardiacUnit:
        raise NotImplementedError

    def remap_labels(self, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_metadata(self, unit: CardiacUnit) -> Mapping[str, Any]:
        return unit.metadata


class Cardiac35DSliceDataset(Dataset):
    """Lazy center-slice dataset preserving the existing [B,3,H,W] contract."""

    def __init__(
        self,
        adapter: Cardiac35DAdapter,
        *,
        units: Sequence[CardiacUnit] | None = None,
        split: str | None = None,
        supervised_only: bool = True,
        image_size: int | tuple[int, int] = 256,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        max_cache: int = 2,
        augment: bool = False,
        transform: Any | None = None,
    ) -> None:
        self.adapter = adapter
        self.units = list(
            units
            if units is not None
            else adapter.discover_units(split=split, supervised_only=supervised_only)
        )
        if not self.units:
            raise ValueError("Cardiac35DSliceDataset has no supervised cardiac units")
        self.image_size = image_size
        self.lower_percentile = float(lower_percentile)
        self.upper_percentile = float(upper_percentile)
        self.max_cache = max(int(max_cache), 1)
        self.augment = bool(augment)
        self.transform = transform
        self._cache: OrderedDict[int, LoadedCardiacUnit] = OrderedDict()
        self._index: list[tuple[int, int]] = []
        for unit_index, unit in enumerate(self.units):
            shape = self._unit_shape(unit_index)
            if len(shape) != 3 or int(shape[0]) < 1:
                raise ValueError(
                    f"Cardiac unit {unit.case_id!r} must expose [Z,H,W], got {shape}"
                )
            self._index.extend((unit_index, z) for z in range(int(shape[0])))

    def _unit_shape(self, unit_index: int) -> tuple[int, ...]:
        unit = self.units[unit_index]
        shape = unit.metadata.get("shape") if isinstance(unit.metadata, Mapping) else None
        if shape is not None:
            return tuple(int(value) for value in shape)
        loaded = self._load_unit(unit_index)
        return tuple(int(value) for value in loaded.image_zhw.shape)

    def _load_unit(self, unit_index: int) -> LoadedCardiacUnit:
        cached = self._cache.get(unit_index)
        if cached is not None:
            self._cache.move_to_end(unit_index)
            return cached
        unit = self.units[unit_index]
        loaded = self.adapter.load_unit(unit)
        image = np.asarray(loaded.image_zhw)
        mask = np.asarray(loaded.mask_zhw)
        if image.ndim != 3 or mask.ndim != 3:
            raise ValueError(
                f"Unit {unit.case_id!r} must load image/mask as [Z,H,W], "
                f"got {image.shape} and {mask.shape}"
            )
        if image.shape != mask.shape:
            raise ValueError(
                f"Unit {unit.case_id!r} image/mask shape mismatch: "
                f"{image.shape} vs {mask.shape}"
            )
        normalized_image = percentile_clip_and_zscore(
            image.astype(np.float32, copy=False),
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
        )
        mapped_mask = np.asarray(self.adapter.remap_labels(mask), dtype=np.int64)
        if mapped_mask.shape != image.shape:
            raise ValueError(
                f"Unit {unit.case_id!r} remapped mask shape mismatch: "
                f"{mapped_mask.shape} vs {image.shape}"
            )
        loaded = LoadedCardiacUnit(
            image_zhw=normalized_image,
            mask_zhw=mapped_mask,
            spacing=loaded.spacing,
            metadata=dict(loaded.metadata),
        )
        self._cache[unit_index] = loaded
        self._cache.move_to_end(unit_index)
        while len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        return loaded

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        unit_index, z = self._index[int(index)]
        unit = self.units[unit_index]
        loaded = self._load_unit(unit_index)
        image = torch.from_numpy(build_25d_triplet(loaded.image_zhw, z))
        mask = torch.from_numpy(np.asarray(loaded.mask_zhw[z], dtype=np.int64))
        image, mask = resize_sample(image, mask, self.image_size)
        metadata = dict(unit.metadata)
        metadata.update(loaded.metadata)
        phase = str(unit.phase or metadata.get("phase") or "")
        time_index = int(unit.time_index if unit.time_index is not None else metadata.get("time_index", -1))
        spacing = loaded.spacing
        sample: dict[str, Any] = {
            "image": image.float(),
            "mask": mask.long(),
            "target": mask.long(),
            "dataset": str(unit.dataset),
            "case_id": str(unit.case_id),
            "subject_id": str(unit.subject_id),
            "patient_id": str(unit.subject_id),
            "slice_idx": int(z),
            "z": int(z),
            "num_slices": int(loaded.image_zhw.shape[0]),
            "phase": phase,
            "time_index": time_index,
            "t": time_index,
            "acquisition_id": str(unit.acquisition_id or ""),
            "spacing": tuple(float(value) for value in spacing) if spacing is not None else (),
        }
        sample.update(
            {
                key: value
                for key, value in metadata.items()
                if key not in sample and value is not None
            }
        )
        if self.transform is not None:
            transformed = self.transform(sample)
            if not isinstance(transformed, Mapping):
                raise TypeError("cardiac 2.5-D transform must return a sample mapping")
            sample = dict(transformed)
        elif self.augment:
            from .transforms import RandomGeometricTransform
            sample = RandomGeometricTransform()(sample)
        return sample
