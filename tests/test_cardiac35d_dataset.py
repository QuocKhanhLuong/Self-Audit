from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

try:
    from self_audit.data.cardiac35d import (
        Cardiac35DAdapter,
        Cardiac35DSliceDataset,
        CardiacUnit,
        LoadedCardiacUnit,
    )
except ImportError:
    from src.self_audit.data.cardiac35d import (
        Cardiac35DAdapter,
        Cardiac35DSliceDataset,
        CardiacUnit,
        LoadedCardiacUnit,
    )


@dataclass
class _DummyAdapter(Cardiac35DAdapter):
    image: np.ndarray
    mask: np.ndarray

    def discover_units(self, *, split: str | None = None, supervised_only: bool = True):
        return [
            CardiacUnit(
                dataset="dummy",
                case_id="case-1",
                subject_id="subject-1",
                image_ref="image",
                mask_ref="mask",
                split=split or "train",
                phase="ED",
                metadata={"shape": self.image.shape},
            )
        ]

    def load_unit(self, unit: CardiacUnit) -> LoadedCardiacUnit:
        return LoadedCardiacUnit(
            image_zhw=self.image,
            mask_zhw=self.mask,
            spacing=(2.0, 1.0, 1.0),
            metadata={"phase": unit.phase},
        )

    def remap_labels(self, mask: np.ndarray) -> np.ndarray:
        observed = set(int(value) for value in np.unique(mask))
        if not observed.issubset({0, 1, 2, 3}):
            raise ValueError(f"mask contains unknown label values {sorted(observed)}")
        return np.asarray(mask, dtype=np.int64)


def test_dataset_emits_three_spatial_channels_and_center_target() -> None:
    image = np.stack([np.full((4, 4), value, dtype=np.float32) for value in range(4)])
    mask = np.zeros((4, 4, 4), dtype=np.int64)
    mask[1, 1, 1] = 1
    dataset = Cardiac35DSliceDataset(_DummyAdapter(image, mask), image_size=4)

    first = dataset[0]
    assert tuple(first["image"].shape) == (3, 4, 4)
    assert tuple(first["mask"].shape) == (4, 4)
    assert first["image"].dtype == torch.float32
    assert first["mask"].dtype == torch.int64
    assert first["case_id"] == "case-1"
    assert first["subject_id"] == "subject-1"
    assert first["dataset"] == "dummy"
    assert first["slice_idx"] == 0
    assert first["phase"] == "ED"
    assert torch.equal(first["image"][0], first["image"][1])

    center = dataset[1]
    assert center["slice_idx"] == 1
    assert int(center["mask"][1, 1]) == 1


def test_dataset_rejects_invalid_source_mask_labels() -> None:
    image = np.zeros((3, 4, 4), dtype=np.float32)
    mask = np.zeros((3, 4, 4), dtype=np.int64)
    mask[1, 1, 1] = 9
    dataset = Cardiac35DSliceDataset(_DummyAdapter(image, mask), image_size=4)
    with pytest.raises(ValueError, match="unknown label"):
        dataset[1]
