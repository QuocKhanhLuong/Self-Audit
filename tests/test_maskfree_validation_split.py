"""Focused image-only checks for official validation split discovery."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from self_audit_maskfree.data import discover_dataset  # noqa: E402
from self_audit_maskfree.data.discovery import (  # noqa: E402
    OfficialSplitConflictError,
)


def _write_npy(path: Path, value: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full((6, 5, 3), value, dtype=np.float32))


def _split_by_patient(manifest: dict) -> dict[str, str]:
    return {
        record["patient_id"]: record["split"]
        for record in manifest["records"]
        if record["slice_index"] == 0
    }


def test_mnms_official_validation_membership_maps_to_dev(tmp_path):
    root = tmp_path / "mnms"
    _write_npy(root / "Training" / "case001_t01.npy", 1.0)
    _write_npy(root / "Validation" / "case002_t01.npy", 2.0)
    _write_npy(root / "val" / "case003_t01.npy", 3.0)
    _write_npy(root / "Testing" / "case004_t01.npy", 4.0)

    manifest = discover_dataset(root, "mnms", seed=42)

    assert _split_by_patient(manifest) == {
        "case001": "train",
        "case002": "dev",
        "case003": "dev",
        "case004": "test",
    }
    assert manifest["split_provenance"]["official_folder_membership"] is True
    assert manifest["split_provenance"]["official_folder_patients"] == {
        "train": 1,
        "dev": 2,
        "test": 1,
    }
    assert manifest["official_split_check"]["conflicts"] == []
    assert "synthetic_split_not_a_fresh_test_set" not in {
        item["code"] for item in manifest["limitations"]
    }
    assert {
        record["cohort_provenance"]["official_folder"]
        for record in manifest["records"]
        if record["slice_index"] == 0
    } == {"train", "dev", "test"}


def test_mnms_patient_in_multiple_official_folders_fails_closed(tmp_path):
    root = tmp_path / "mnms"
    training_path = root / "Training" / "case001_t01.npy"
    validation_path = root / "Validation" / "case001_t02.npy"
    _write_npy(training_path, 1.0)
    _write_npy(validation_path, 2.0)

    with pytest.raises(OfficialSplitConflictError) as raised:
        discover_dataset(root, "mnms", seed=42)

    details = raised.value.details
    assert details["code"] == "official_split_patient_conflict"
    assert details["conflicts"] == [
        {
            "patient_id": "case001",
            "official_folders": ["dev", "train"],
            "source_paths": {
                "dev": [str(validation_path)],
                "train": [str(training_path)],
            },
        }
    ]


def test_official_folder_detection_is_relative_to_dataset_root(tmp_path):
    # The absolute parent deliberately contains an exact ``test`` component;
    # it must not turn an otherwise untagged source into an official test case.
    root = tmp_path / "test" / "mnms_root"
    source = root / "case001.npy"
    _write_npy(source)

    manifest = discover_dataset(root, "mnms", seed=42)
    record = manifest["records"][0]

    assert record["cohort_provenance"]["official_folder"] is None
    assert manifest["split_provenance"]["official_folder_membership"] is False
    assert manifest["split_provenance"]["official_test_membership"] is False
    assert record["split"] == "train"


def test_acdc_training_folder_still_uses_patient_hash_split(tmp_path):
    root = tmp_path / "acdc" / "training"
    for index in range(1, 9):
        _write_npy(root / f"patient{index:03d}.npy", float(index))

    manifest = discover_dataset(tmp_path / "acdc", "acdc", seed=42)
    splits = _split_by_patient(manifest)

    assert manifest["split_provenance"]["rule"] == "rank_hashed_patient_id_70_15_15"
    assert manifest["split_provenance"]["official_folder_membership"] is True
    assert manifest["split_provenance"]["official_test_membership"] is False
    assert set(splits.values()) == {"train", "dev", "test"}
    assert all(
        record["cohort_provenance"]["official_folder"] == "train"
        for record in manifest["records"]
    )


def test_mnms_training_only_root_keeps_deterministic_dev_fallback(tmp_path):
    root = tmp_path / "mnms"
    for index in range(8):
        _write_npy(root / "Training" / f"patient{index:03d}_sa.npy", float(index))
    manifest = discover_dataset(root, "mnms", seed=42)
    assert set(_split_by_patient(manifest).values()) == {"train", "dev", "test"}
    assert manifest["split_provenance"]["rule"] == "rank_hashed_patient_id_70_15_15"
