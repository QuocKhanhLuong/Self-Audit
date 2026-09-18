from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import numpy as np
import pytest

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

try:
    from self_audit.data.cmrxmotion import (
        CMRxMotionAdapter,
        CMRxMotionName,
        parse_cmrxmotion_filename,
    )
except ImportError:
    from src.self_audit.data.cmrxmotion import (
        CMRxMotionAdapter,
        CMRxMotionName,
        parse_cmrxmotion_filename,
    )


pytestmark = pytest.mark.skipif(nib is None, reason="nibabel is required")


def _nifti_bytes(array: np.ndarray, affine: np.ndarray | None = None) -> bytes:
    image = nib.Nifti1Image(array, np.eye(4) if affine is None else affine)
    return gzip.compress(image.to_bytes())


def _write_archive(root: Path, *, with_label: bool = True, mismatch: bool = False) -> Path:
    archive_path = root / "cmrxmotion training dataset.zip"
    image = np.zeros((6, 5, 4, 1), dtype=np.float32)
    image[2:4, 1:3, 1, 0] = 5
    mask = np.zeros((6, 5, 4), dtype=np.int16)
    mask[2:4, 1:3, 1] = 1
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data/P001-1/P001-1-ED.nii.gz", _nifti_bytes(image))
        if with_label:
            affine = np.diag([1.0, 1.0, 2.0, 1.0]) if mismatch else np.eye(4)
            archive.writestr(
                "data/P001-1/P001-1-ED-label.nii.gz",
                _nifti_bytes(mask, affine=affine),
            )
        archive.writestr("data/P001-1/P001-1-ES.nii.gz", _nifti_bytes(image))
        archive.writestr("data/P001-2/P001-2-ED.nii.gz", _nifti_bytes(image))
        if with_label:
            archive.writestr(
                "data/P001-2/P001-2-ED-label.nii.gz",
                _nifti_bytes(mask),
            )
    return archive_path


def test_filename_parser_extracts_subject_acquisition_phase_and_label() -> None:
    parsed = parse_cmrxmotion_filename("data/P003-4/P003-4-ES-label.nii.gz")
    assert isinstance(parsed, CMRxMotionName)
    assert parsed.subject_id == "P003"
    assert parsed.acquisition_id == "4"
    assert parsed.phase == "ES"
    assert parsed.is_label is True
    assert parsed.case_id == "P003-4-ES"


def test_adapter_excludes_missing_masks_from_supervised_units(tmp_path: Path) -> None:
    _write_archive(tmp_path, with_label=True)
    adapter = CMRxMotionAdapter(tmp_path)
    units = adapter.discover_units(supervised_only=True)
    assert [unit.case_id for unit in units] == ["P001-1-ED", "P001-2-ED"]
    assert {unit.subject_id for unit in units} == {"P001"}
    audit = adapter.audit()
    missing = {item.case_id for item in audit if not item.has_mask}
    assert missing == {"P001-1-ES"}


def test_adapter_squeezes_image_singleton_and_maps_labels(tmp_path: Path) -> None:
    _write_archive(tmp_path, with_label=True)
    adapter = CMRxMotionAdapter(tmp_path)
    unit = adapter.discover_units()[0]
    loaded = adapter.load_unit(unit)
    assert loaded.image_zhw.shape == (4, 5, 6)
    assert loaded.mask_zhw.shape == loaded.image_zhw.shape
    assert set(np.unique(loaded.mask_zhw)).issubset({0, 1, 2, 3})
    assert int(loaded.mask_zhw[1, 1, 2]) == 3


def test_affine_mismatch_is_reported_and_excluded_unless_explicitly_allowed(tmp_path: Path) -> None:
    _write_archive(tmp_path, with_label=True, mismatch=True)
    strict = CMRxMotionAdapter(tmp_path)
    strict_ids = {unit.case_id for unit in strict.discover_units()}
    assert "P001-1-ED" not in strict_ids
    record = next(item for item in strict.audit() if item.case_id == "P001-1-ED")
    assert record.affine_equal is False
    permissive = CMRxMotionAdapter(tmp_path, allow_affine_mismatch=True)
    assert "P001-1-ED" in {unit.case_id for unit in permissive.discover_units()}
