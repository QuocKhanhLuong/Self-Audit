from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.self_audit.data.mnms import discover_mnms_records
from src.self_audit.training._utils import build_patient_dataset, validate_dataset_splits


def _write_mnms_split(root: Path, split: str = "testing", case_id: str = "A0S9V9_t00", *, label: int = 1) -> None:
    volumes = root / split / "volumes"
    masks = root / split / "masks"
    volumes.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    volume = np.zeros((16, 16, 4), dtype=np.float32)
    mask = np.zeros((16, 16, 4), dtype=np.uint8)
    mask[2:6, 2:6, 1] = label
    np.save(volumes / f"{case_id}.npy", volume)
    np.save(masks / f"{case_id}.npy", mask)


def test_mnms_config_dispatches_to_mnms_dataset(tmp_path: Path) -> None:
    _write_mnms_split(tmp_path)
    dataset = build_patient_dataset(
        {"dataset": "mnms", "data_root": str(tmp_path), "image_size": 16, "depth_axis": 2},
        split="testing",
        train=False,
    )
    assert dataset.__class__.__name__ == "MNMSDataset"
    assert dataset.records[0].case_id == "A0S9V9_t00"
    assert dataset.records[0].patient_id == "A0S9V9"


def test_mnms_validation_reports_real_pairs_and_signature(tmp_path: Path) -> None:
    _write_mnms_split(tmp_path)
    result = validate_dataset_splits(
        {"dataset": "mnms", "data_root": str(tmp_path), "test_split": "testing", "depth_axis": 2}
    )
    assert result["validated"] is True
    assert result["case_counts"]["test"] == 1
    assert result["patient_counts"]["test"] == 1
    assert result["membership_signature"]
    assert result["dataset"] == "mnms"


def test_mnms_binary_contract_is_rejected(tmp_path: Path) -> None:
    _write_mnms_split(tmp_path)
    with pytest.raises(ValueError, match="four-class"):
        build_patient_dataset(
            {"dataset": "mnms", "data_root": str(tmp_path), "num_classes": 2},
            split="testing",
            train=False,
        )


def test_mnms_mapping_must_cover_four_class_contract() -> None:
    with pytest.raises(ValueError, match="raw classes 0..3"):
        from src.self_audit.data.mnms import MNMSClassMapping

        MNMSClassMapping({0: 0, 1: 3, 2: 2})


def test_mnms_unknown_label_fails_validation(tmp_path: Path) -> None:
    _write_mnms_split(tmp_path, label=9)
    with pytest.raises(ValueError, match="without an explicit class mapping"):
        validate_dataset_splits({"dataset": "mnms", "data_root": str(tmp_path), "test_split": "testing", "depth_axis": 2})


def test_mnms_missing_pair_fails_closed(tmp_path: Path) -> None:
    volumes = tmp_path / "testing" / "volumes"
    volumes.mkdir(parents=True)
    np.save(volumes / "A0S9V9_t00.npy", np.zeros((4, 4, 2), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="No paired M&Ms"):
        discover_mnms_records(tmp_path, split="testing")


def test_external_command_contract_is_documented() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "phase_c_best.pt" in readme
    assert "--split testing" in readme
    assert "--tau-accept" in readme
    assert "preprocessed_data/mnm" in readme
    assert "mnm_binary" in readme
