import gzip
import io
import json
import zipfile

import numpy as np
import pytest

from scripts.pg_audit_mnms_download import (
    CLASS_MAPPING, GEOMETRY_AFFINE_ATOL, _compare_geometry, _safe_member,
    _raw_findings, _header, audit_archives,
)


def _nifti_bytes(tmp_path, affine=None, unit="mm"):
    import nibabel as nib
    image = nib.Nifti1Image(np.zeros((4, 5, 6, 2), dtype=np.int16), affine if affine is not None else np.eye(4))
    image.header.set_xyzt_units(unit, "sec")
    path = tmp_path / f"fixture-{unit}.nii"
    nib.save(image, path)
    return path.read_bytes()


def _make_archive(path, root, count, *, tmp_path, offset=0, unit="mm", mask_payload=b"mask-not-read"):
    image = _nifti_bytes(tmp_path, unit=unit)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(count):
            pid = f"P{i + offset:05d}"
            base = f"{root}/{pid}/{pid}"
            zf.writestr(f"{base}_sa.nii.gz", gzip.compress(image))
            zf.writestr(f"{base}_sa_gt.nii.gz", mask_payload)
    return path


def test_member_path_rejects_traversal_and_absolute():
    for value in ("../outside.nii.gz", "/absolute/file", "Testing/../../bad"):
        with pytest.raises(ValueError, match="unsafe"):
            _safe_member(value)


def test_raw_header_geometry_flags_unknown_units(tmp_path):
    header, _ = _header(io.BytesIO(_nifti_bytes(tmp_path, unit="unknown")[:348]))
    finding = _raw_findings(header)
    assert finding["status"] == "INVALID"
    assert "unknown_spatial_units" in finding["findings"]
    assert finding["sform_code"] > 0


def test_archive_audit_verifies_crc_and_keeps_geometry_separate(tmp_path):
    paths = [
        _make_archive(tmp_path / "Training-001.zip", "Training/Labeled", 175, tmp_path=tmp_path, unit="unknown"),
        _make_archive(tmp_path / "M&M-20251118T142205Z-1-002.zip", "Validation", 34, tmp_path=tmp_path, offset=175),
        _make_archive(tmp_path / "Testing-003.zip", "Testing", 136, tmp_path=tmp_path, offset=209),
    ]
    out = tmp_path / "out"
    report = audit_archives(paths, out, source_document=tmp_path / "missing.pdf",
                            source_snapshot=tmp_path / "missing.json", geometry_summary=tmp_path / "missing-summary.json")
    assert report["status"] == "RAW_VERIFIED"
    assert report["training_ready"] is False
    assert report["archive_integrity"] == "PASS_ALL_THREE"
    assert report["split_counts"] == {"train": 175, "val": 34, "test": 136}
    assert report["raw_header_findings"]["unknown_unit_count"] == 175
    assert report["geometry_policy"]["status"] == "NOT_APPLIED"
    assert report["geometry_policy"]["geometry_validation_errors"]
    assert json.loads((out / "MNMS_AUDIT.json").read_text())["training"] == "NOT_RUN"


def test_canonical_mapping_and_svd_tolerance_are_explicit():
    assert CLASS_MAPPING == {"0": 0, "1": 3, "2": 2, "3": 1}
    expected = {"shape": [14, 440, 512], "spacing_zyx": [10.0, 0.7954999804496765, 0.7955000400543213],
                "affine_zyx": np.eye(4).tolist(), "context": "full_volume"}
    recomputed = {**expected, "affine_zyx": np.eye(4).tolist()}
    recomputed["affine_zyx"][0][1] = GEOMETRY_AFFINE_ATOL / 10
    assert _compare_geometry(expected, recomputed)["matches"]
    recomputed["affine_zyx"][0][1] = GEOMETRY_AFFINE_ATOL * 2
    comparison = _compare_geometry(expected, recomputed)
    assert not comparison["matches"]
    assert comparison["errors"][0]["field"] == "affine_zyx"
    recomputed["affine_zyx"] = expected["affine_zyx"]
    recomputed["spacing_zyx"] = [10.0, 0.7954999804496766, 0.7955000400543213]
    assert _compare_geometry(expected, recomputed)["errors"][0]["field"] == "spacing_zyx"


def test_historical_receipts_recompute_all_345_geometries_with_tight_affine_tolerance():
    import io
    from pathlib import Path
    import nibabel as nib
    from self_audit.candidate_fuse.geometry import adjudicate_geometry

    root = Path(__file__).resolve().parents[1]
    receipt_dir = root / "reports/fuse-mnms-geometry-20261113/adjudicated"
    snapshot = json.loads((receipt_dir / "source_snapshot.json").read_text())
    summary = json.loads((receipt_dir / "SUMMARY.json").read_text())
    rows = json.loads((receipt_dir / "per_patient.json").read_text())
    by_id = {row["patient_id"]: row for row in rows}
    maximum = 0.0
    matched = 0
    for source in snapshot["records"]:
        result = adjudicate_geometry(
            nib.Nifti1Header.from_fileobj(io.BytesIO(bytes.fromhex(source["header_hex"]))),
            policy=summary["policy"],
        )
        comparison = _compare_geometry(by_id[source["patient_id"]]["geometry"], result["geometry"])
        assert comparison["matches"], {"patient_id": source["patient_id"], "errors": comparison["errors"]}
        maximum = max(maximum, comparison["affine_max_abs_diff"])
        matched += 1
    assert matched == 345
    assert maximum < 1e-10


def test_actual_drive_validation_container_is_unwrapped_and_hash_bound(tmp_path):
    from scripts.pg_audit_mnms_download import _archive_inventory, sha256_file
    inner = _make_archive(tmp_path/'Validation.zip', 'Validation', 34, tmp_path=tmp_path)
    outer = tmp_path/'M&M-20251118T142205Z-1-002.zip'
    with zipfile.ZipFile(outer, 'w') as z:
        z.write(inner, 'M&M/Validation.zip')
        z.writestr('M&M/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv', 'External code,ED,ES\n')
    receipt, patients = _archive_inventory(outer, 'val')
    assert len(patients) == 34
    assert receipt['sha256'] == sha256_file(outer)
    assert receipt['nested_archive_sha256'] == sha256_file(inner)
    assert all(p['archive_sha256'] == sha256_file(outer) for p in patients)
    assert outer.exists()


def test_corrupt_member_crc_fails(tmp_path):
    paths = [
        _make_archive(tmp_path / "Training-001.zip", "Training/Labeled", 175, tmp_path=tmp_path),
        _make_archive(tmp_path / "M&M-20251118T142205Z-1-002.zip", "Validation", 34, tmp_path=tmp_path, offset=175),
        _make_archive(tmp_path / "Testing-003.zip", "Testing", 136, tmp_path=tmp_path, offset=209),
    ]
    # Rewrite one valid archive member under a wrong CRC in a deliberately malformed central directory.
    path = paths[0]
    raw = bytearray(path.read_bytes())
    central = raw.rfind(b"PK\x01\x02")
    assert central >= 0
    raw[central + 16] ^= 0x01
    path.write_bytes(raw)
    with pytest.raises((zipfile.BadZipFile, ValueError)):
        audit_archives(paths, tmp_path / "out", source_document=tmp_path / "missing.pdf")
