from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import DataLoader

from src.self_audit.data import (
    ACDCDataset,
    MNMSClassMapping,
    MNMSDataset,
    build_25d_triplet,
    patient_level_split,
    validate_patient_split,
)


def _write_pair(root: Path, case_id: str, raw_mask: bool = False) -> None:
    (root / "volumes").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    volume = np.arange(4 * 6 * 5, dtype=np.float32).reshape(4, 6, 5)
    mask = np.zeros_like(volume, dtype=np.int64)
    mask[1:3, 1:4, 0] = 1 if not raw_mask else 3
    mask[2:4, 2:5, -1] = 2 if not raw_mask else 1
    np.save(root / "volumes" / f"{case_id}.npy", volume)
    np.save(root / "masks" / f"{case_id}.npy", mask)


def test_25d_boundary_construction_repeats_end_slices() -> None:
    volume = np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2)
    assert np.array_equal(build_25d_triplet(volume, 0), volume[[0, 0, 1]])
    assert np.array_equal(build_25d_triplet(volume, 3), volume[[2, 3, 3]])


def test_acdc_sample_contract(tmp_path: Path) -> None:
    for case_id in ("patient001_ED", "patient002_ED"):
        _write_pair(tmp_path, case_id)
    dataset = ACDCDataset(tmp_path, image_size=8)
    sample = dataset[0]
    assert sample["image"].shape == (3, 8, 8)
    assert sample["mask"].shape == (8, 8)
    assert sample["patient_id"].startswith("patient")
    assert {int(value) for value in np.unique(sample["mask"].numpy())}.issubset({0, 1, 2, 3})
    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    assert batch["image"].shape[1:] == (3, 8, 8)


def test_mnms_mapping_is_explicit_and_shared_contract(tmp_path: Path) -> None:
    _write_pair(tmp_path, "subject001", raw_mask=True)
    mapping = MNMSClassMapping({0: 0, 1: 3, 2: 2, 3: 1})
    dataset = MNMSDataset(tmp_path, image_size=8, class_mapping=mapping)
    sample = dataset[0]
    assert sample["image"].shape == (3, 8, 8)
    assert sample["mask"].shape == (8, 8)
    assert dataset.raw_to_acdc == {0: 0, 1: 3, 2: 2, 3: 1}
    with pytest.raises(ValueError, match="background"):
        MNMSClassMapping({1: 1, 2: 2, 3: 3})


def test_patient_split_rejects_cross_split_patient_leakage() -> None:
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_patient_split({"train": ["patient001_ED"], "val": ["patient001_ES"]})
    split = patient_level_split(
        ["patient001_ED", "patient001_ES", "patient002_ED", "patient003_ED", "patient004_ED"],
        train_fraction=0.6,
        val_fraction=0.2,
        seed=42,
    )
    assert set(split["train"]).isdisjoint(split["val"])
    assert {case.split("_")[0] for case in split["train"]}.isdisjoint({case.split("_")[0] for case in split["val"]})
