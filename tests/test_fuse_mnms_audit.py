from pathlib import Path

import numpy as np

import csv
import json

import pytest

from scripts.fuse_audit_mnms import audit_header, summarize, write_audit


def _nifti(path: Path, affine: np.ndarray, unit: str = "mm") -> Path:
    import nibabel as nib

    image = nib.Nifti1Image(np.zeros((5, 6, 7), dtype=np.uint8), affine)
    image.header.set_xyzt_units(unit, "sec")
    nib.save(image, path)
    return path


def test_header_audit_accepts_valid_mm_geometry(tmp_path):
    path = _nifti(tmp_path / "good.nii.gz", np.diag([1.2, 1.3, 1.4, 1]))
    row = audit_header("P1", "train", path)
    assert row["status"] == "VALID"
    assert row["affine_pixdim_agree"]
    assert row["spatial_unit"] == "mm"


def test_header_audit_rejects_shear_unknown_units_and_missing_file(tmp_path):
    affine = np.eye(4)
    affine[0, 1] = 0.2
    shear = audit_header("P2", "val", _nifti(tmp_path / "shear.nii.gz", affine))
    assert "nonorthogonal_affine" in shear["findings"]
    unknown = audit_header("P3", "test", _nifti(tmp_path / "unknown.nii.gz", np.eye(4), "unknown"))
    assert "unknown_spatial_units" in unknown["findings"]
    missing = audit_header("P4", "val", tmp_path / "missing.nii.gz")
    assert missing["findings"] == ["missing_image_header"]


def test_summary_blocks_missing_test_and_invalid_val_test():
    rows = [
        {"patient_id": "T1", "split": "train", "status": "INVALID"},
        {"patient_id": "V1", "split": "val", "status": "INVALID"},
        {"patient_id": "X1", "split": "test", "status": "INVALID"},
    ]
    summary = summarize(rows)
    assert summary["status"] == "BLOCKED"
    assert summary["invalid_patient_ids_by_split"] == {"train": ["T1"], "val": ["V1"], "test": ["X1"]}
    assert "invalid_val_patient_ids=V1" in summary["blockers"]
    assert "invalid_test_patient_ids=X1" in summary["blockers"]


def test_nonfinite_header_values_write_as_null_and_csv_json(tmp_path):
    rows = [{
        "patient_id": "NAN", "split": "train", "status": "INVALID",
        "findings": ["invalid_pixdim"], "pixdim_xyz": [1.0, float("nan"), float("inf")],
        "affine": [[1.0, 0.0], [float("nan"), 1.0]],
    }]
    write_audit(rows, tmp_path)
    saved = json.loads((tmp_path / "per_patient.json").read_text())
    assert saved[0]["pixdim_xyz"] == [1.0, None, None]
    with (tmp_path / "per_patient.csv").open(newline="") as stream:
        record = next(csv.DictReader(stream))
    assert json.loads(record["findings"]) == ["invalid_pixdim"]
    assert json.loads(record["affine"])[1][0] is None


def test_train_exclusions_do_not_block_when_eight_valid_but_unknown_evidence_does():
    rows = ([{"patient_id": f"BAD{i}", "split": "train", "status": "INVALID"} for i in range(2)]
            + [{"patient_id": f"GOOD{i}", "split": "train", "status": "VALID"} for i in range(8)]
            + [{"patient_id": "V1", "split": "val", "status": "VALID"},
               {"patient_id": "X1", "split": "test", "status": "VALID"}])
    summary = summarize(rows)
    assert summary["train_exclusions"]["train"] == ["BAD0", "BAD1"]
    assert "fewer_than_8_valid_train_patients" not in summary["blockers"]
    assert "official_train_val_test_split_evidence_unverified" in summary["blockers"]
    assert "class_mapping_BG_RV_MYO_LV_unverified" in summary["blockers"]


def test_summary_fails_closed_on_unknown_split_or_duplicate_ids():
    with pytest.raises(ValueError, match="unknown split"):
        summarize([{"patient_id": "P1", "split": "development", "status": "VALID"}])
    with pytest.raises(ValueError, match="duplicate patient ID"):
        summarize([{"patient_id": "P1", "split": "train", "status": "VALID"},
                   {"patient_id": "P1", "split": "val", "status": "VALID"}])
