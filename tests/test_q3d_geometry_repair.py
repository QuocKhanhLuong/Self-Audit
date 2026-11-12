from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from self_audit.data import CLASS_MAPPING, GEOMETRY_MAP_SCHEMA, GEOMETRY_SCHEMA
from self_audit.candidate_q3d.geometry_repair import (
    ACDC_PIXDIM_NATIVE_V1,
    STRICT_GEOMETRY_POLICY,
    repair_manifest_geometry,
    validate_audit_geometry,
    validate_geometry_record,
    validate_geometry_repair_receipt,
)


_AFFINE_XYZ = np.asarray(
    [
        [1.25, 0.0, 0.0, 10.0],
        [0.0, 0.75, 0.0, 20.0],
        [0.0, 0.0, 2.5, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_nifti(
    path: Path,
    *,
    affine: np.ndarray = _AFFINE_XYZ,
    shape_xyz_t: tuple[int, int, int, int] = (4, 5, 6, 2),
    unit: str = "mm",
    sform_code: int = 2,
    qform_code: int = 2,
    zooms: tuple[float, float, float, float] | None = None,
) -> Path:
    data = np.zeros(shape_xyz_t, dtype=np.float32)
    image = nib.Nifti1Image(data, affine)
    image.set_sform(affine, code=sform_code)
    image.set_qform(affine, code=qform_code)
    image.header.set_xyzt_units(xyz=unit, t="sec")
    if zooms is not None:
        image.header["pixdim"][1:5] = zooms
    nib.save(image, str(path))
    return path


def _manifest(patient_ids: tuple[str, ...] = ("patient001",)) -> dict:
    by_patient = {}
    for patient_id in patient_ids:
        by_patient[patient_id] = {
            "schema": GEOMETRY_SCHEMA,
            "array_axes": ["T", "Z", "Y", "X"],
            "spatial_axes": ["Z", "Y", "X"],
            "shape_zyx": [6, 5, 4],
            # These are intentionally the stale values that the repair replaces.
            "spacing_zyx_mm": [1.0, 1.0, 1.0],
            "spacing_yx_mm": [1.0, 1.0],
            "affine_zyx": np.eye(4).tolist(),
            "class_mapping": dict(CLASS_MAPPING),
        }
    labeled = []
    for patient_id in patient_ids:
        labeled.append({
            "id": patient_id,
            "image": f"{patient_id}.npy",
            "image_sha256": f"image-{patient_id}-sha",
            "masks": {"0": {"path": f"{patient_id}_mask.npy", "sha256": f"mask-{patient_id}-sha"}},
            "selector": {"rank": 2, "seed": 42},
        })
    return {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": False,
        "dataset": "acdc",
        "label_budget": len(patient_ids),
        "label_seed": 42,
        "selector_sha256": "selector-sha",
        "labeled": labeled,
        "unlabeled": [],
        "evaluation_ids": [],
        "class_mapping": dict(CLASS_MAPPING),
        "geometry": {"schema": GEOMETRY_MAP_SCHEMA, "by_patient": by_patient},
    }


def _write_manifest(tmp_path: Path, manifest: dict, name: str = "source.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _mapping_inputs(tmp_path: Path, cine_paths: dict[str, Path]) -> tuple[str, str, str]:
    path = tmp_path / "cine-map.json"
    path.write_text(json.dumps({key: str(value) for key, value in cine_paths.items()}, indent=2) + "\n")
    return str(path), _sha(path), "synthetic fixture patient-to-CINE mapping"


def _call(
    tmp_path: Path,
    manifest_path: Path,
    cine_paths: dict[str, Path],
    *,
    output_manifest: Path | None = None,
    audit_only: bool = True,
    unit_declarations: dict | None = None,
    geometry_policy: str = STRICT_GEOMETRY_POLICY,
):
    mapping_source, mapping_sha, mapping_provenance = _mapping_inputs(tmp_path, cine_paths)
    return repair_manifest_geometry(
        manifest_path,
        _sha(manifest_path),
        cine_paths,
        axis_map="XYZ->ZYX",
        cine_mapping_source=mapping_source,
        cine_mapping_sha256=mapping_sha,
        cine_mapping_provenance=mapping_provenance,
        report_json_path=tmp_path / "audit.json",
        report_csv_path=tmp_path / "audit.csv",
        output_manifest_path=output_manifest,
        unit_declarations_by_patient=unit_declarations,
        audit_only=audit_only,
        geometry_policy=geometry_policy,
    )


def _forbid_voxel_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    load = nib.load

    class HeaderOnlyImage:
        def __init__(self, image):
            self.header = image.header
            self.shape = image.shape
            self.affine = image.affine

        @property
        def dataobj(self):
            raise AssertionError("geometry repair must not access NIfTI voxel data")

        def get_fdata(self, *args, **kwargs):
            raise AssertionError("geometry repair must not call get_fdata")

    def header_only_load(*args, **kwargs):
        return HeaderOnlyImage(load(*args, **kwargs))

    monkeypatch.setattr(nib, "load", header_only_load)


def test_header_only_audit_maps_native_xyz_geometry_to_zyx(tmp_path: Path, monkeypatch):
    cine = _write_nifti(tmp_path / "patient001_4d.nii.gz")
    manifest_path = _write_manifest(tmp_path, _manifest())
    source_manifest_bytes = manifest_path.read_bytes()
    cine_bytes = cine.read_bytes()
    _forbid_voxel_reads(monkeypatch)

    report = _call(tmp_path, manifest_path, {"patient001": cine})

    assert report["status"] == "audit_only_pass"
    assert report["receipt"]["status"] == "audit_only"
    assert report["receipt"]["voxel_bridge_status"] == "NOT_VERIFIED"
    assert report["cine_mapping"]["shape_match_is_not_voxel_equivalence"] is True
    assert len(report["audit_rows"]) == 1
    row = report["audit_rows"][0]
    assert row["header_zooms_xyz_native"] == pytest.approx([1.25, 0.75, 2.5])
    assert row["sform_affine_xyz"] is not None
    assert row["qform_affine_xyz"] is not None
    assert len(row["source_file_sha256"]) == 64
    assert len(row["source_header_sha256"]) == 64
    assert row["repaired_spacing_zyx_mm"] == pytest.approx([2.5, 0.75, 1.25])
    geometry = report["audit_rows"][0]
    assert (tmp_path / "audit.json").is_file()
    assert (tmp_path / "audit.csv").is_file()
    assert not (tmp_path / "corrected.json").exists()
    assert manifest_path.read_bytes() == source_manifest_bytes
    assert cine.read_bytes() == cine_bytes


def test_successful_new_manifest_changes_geometry_only_and_receipt_validates(tmp_path: Path, monkeypatch):
    cine = _write_nifti(tmp_path / "patient001_4d.nii.gz")
    source_manifest = _manifest()
    manifest_path = _write_manifest(tmp_path, source_manifest)
    source_bytes = manifest_path.read_bytes()
    cine_bytes = cine.read_bytes()
    output_path = tmp_path / "corrected.json"
    _forbid_voxel_reads(monkeypatch)

    report = _call(
        tmp_path,
        manifest_path,
        {"patient001": cine},
        output_manifest=output_path,
        audit_only=False,
    )

    repaired = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert {key: value for key, value in repaired.items() if key != "geometry"} == {
        key: value for key, value in source_manifest.items() if key != "geometry"
    }
    assert repaired["geometry"]["by_patient"]["patient001"]["shape_zyx"] == [6, 5, 4]
    assert repaired["geometry"]["by_patient"]["patient001"]["spacing_zyx_mm"] == pytest.approx(
        [2.5, 0.75, 1.25]
    )
    assert validate_geometry_record(repaired["geometry"]["by_patient"]["patient001"])
    validation = validate_geometry_repair_receipt(
        source_manifest,
        repaired,
        report["receipt"],
        source_manifest_sha256=_sha(manifest_path),
        repaired_manifest_sha256=_sha(output_path),
    )
    assert validation["valid"] is True
    assert validation["patient_count"] == 1
    assert manifest_path.read_bytes() == source_bytes
    assert cine.read_bytes() == cine_bytes


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"zooms": (1.0, 1.0, 1.0, 1.0), "affine": np.asarray([
            [1.0, 0.0, 0.0, 12.0], [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0], [0.0, 0.0, 0.0, 1.0],
        ])}, "spacing [1,1,1]"),
        ({"affine": np.eye(4)}, "identity affine"),
        ({"zooms": (float("nan"), 0.75, 2.5, 1.0)}, "finite and positive"),
        ({"zooms": (1.5, 0.75, 2.5, 1.0)}, "disagree with authoritative affine column lengths"),
        ({
            "affine": np.asarray([
                [1.25, 0.1, 0.0, 10.0],
                [0.0, np.sqrt(0.75**2 - 0.1**2), 0.0, 20.0],
                [0.0, 0.0, 2.5, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            "qform_code": 0,
        }, "sheared"),
    ],
)
def test_invalid_spacing_or_identity_affine_rejected(tmp_path: Path, monkeypatch, kwargs, message):
    cine = _write_nifti(tmp_path / "patient001_4d.nii.gz", **kwargs)
    manifest_path = _write_manifest(tmp_path, _manifest())
    _forbid_voxel_reads(monkeypatch)

    report = _call(tmp_path, manifest_path, {"patient001": cine})

    assert report["status"] == "rejected"
    assert any(message.lower() in row_error.lower() for row_error in report["audit_rows"][0]["errors"])
    assert report["audit_rows"][0]["source_file_sha256"]
    assert report["audit_rows"][0]["sform_affine_xyz"] is not None
    assert not (tmp_path / "corrected.json").exists()


def test_missing_authoritative_sform_and_qform_fails_closed(tmp_path: Path, monkeypatch):
    cine = _write_nifti(
        tmp_path / "patient001_4d.nii.gz",
        sform_code=0,
        qform_code=0,
    )
    manifest_path = _write_manifest(tmp_path, _manifest())
    _forbid_voxel_reads(monkeypatch)

    report = _call(tmp_path, manifest_path, {"patient001": cine})

    assert report["status"] == "rejected"
    assert any("no coded authoritative" in item for item in report["audit_rows"][0]["errors"])
    assert report["audit_rows"][0]["sform_code"] == 0
    assert report["audit_rows"][0]["qform_code"] == 0


@pytest.mark.parametrize("case", ["pixdim_conflict", "uncoded"])
def test_pixdim_policy_repairs_scale_or_uses_uncoded_base_affine_without_voxel_reads(
    tmp_path: Path, monkeypatch, case: str,
):
    if case == "pixdim_conflict":
        cine = _write_nifti(
            tmp_path / "patient001_4d.nii.gz",
            qform_code=0,
            zooms=(1.75, 0.75, 2.5, 1.0),
        )
    else:
        cine = _write_nifti(
            tmp_path / "patient001_4d.nii.gz",
            sform_code=0,
            qform_code=0,
        )
    manifest_path = _write_manifest(tmp_path, _manifest())
    output_path = tmp_path / "corrected.json"
    _forbid_voxel_reads(monkeypatch)

    report = _call(
        tmp_path,
        manifest_path,
        {"patient001": cine},
        output_manifest=output_path,
        audit_only=False,
        geometry_policy=ACDC_PIXDIM_NATIVE_V1,
    )

    row = report["audit_rows"][0]
    assert report["status"] == "success"
    assert report["geometry_policy"] == ACDC_PIXDIM_NATIVE_V1
    assert report["receipt"]["geometry_policy"] == ACDC_PIXDIM_NATIVE_V1
    assert row["geometry_policy"] == ACDC_PIXDIM_NATIVE_V1
    expected_pixdim = [1.75, 0.75, 2.5] if case == "pixdim_conflict" else [1.25, 0.75, 2.5]
    assert row["raw_pixdim_xyz_native"] == pytest.approx(expected_pixdim)
    assert row["errors"] == []
    if case == "pixdim_conflict":
        assert row["geometry_policy_waivers"] == [
            "AFFINE_COLUMN_LENGTHS_DISAGREE_WITH_PIXDIM"
        ]
        assert row["selected_affine_source"] == "sform"
        expected = _AFFINE_XYZ.copy()
        expected[:3, :3] = np.diag([1.75, 0.75, 2.5])
        assert np.allclose(np.asarray(row["reconstructed_affine_xyz"]), expected)
    else:
        assert row["geometry_policy_waivers"][0] == "UNCODED_FORMS_USE_NIBABEL_BASE_AFFINE"
        assert row["selected_affine_source"] == "nibabel_base_affine_uncoded_derived"
        assert row["affine_coordinate_convention"] == "derived_native_image_coordinate_convention"
        assert "UNCODED_AFFINE_IS_DERIVED_NATIVE_IMAGE_COORDINATE_CONVENTION" in row[
            "geometry_nonclaims"
        ]
        assert row["scanner_world_pose_status"] == "NOT_INDEPENDENTLY_VERIFIED"

    repaired = json.loads(output_path.read_text(encoding="utf-8"))
    assert {key: value for key, value in repaired.items() if key != "geometry"} == {
        key: value for key, value in _manifest().items() if key != "geometry"
    }
    corrected = repaired["geometry"]["by_patient"]["patient001"]
    validate_audit_geometry(row, corrected, geometry_policy=ACDC_PIXDIM_NATIVE_V1)
    validation = validate_geometry_repair_receipt(
        _manifest(),
        repaired,
        report["receipt"],
        source_manifest_sha256=_sha(manifest_path),
        repaired_manifest_sha256=_sha(output_path),
        geometry_policy=ACDC_PIXDIM_NATIVE_V1,
    )
    assert validation["geometry_policy"] == ACDC_PIXDIM_NATIVE_V1


def test_pixdim_policy_rejects_unit_pixdim_and_bad_policy_or_affine_tampering(tmp_path, monkeypatch):
    cine = _write_nifti(
        tmp_path / "patient001_4d.nii.gz",
        qform_code=0,
        zooms=(1.0, 1.0, 1.0, 1.0),
    )
    manifest_path = _write_manifest(tmp_path, _manifest())
    _forbid_voxel_reads(monkeypatch)
    report = _call(
        tmp_path,
        manifest_path,
        {"patient001": cine},
        geometry_policy=ACDC_PIXDIM_NATIVE_V1,
    )
    assert report["status"] == "rejected"
    assert any("raw pixdim [1,1,1]" in error for error in report["audit_rows"][0]["errors"])

    # Independently verify that a policy report cannot bless an altered affine.
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    good = _write_nifti(
        second_dir / "patient002_4d.nii.gz",
        qform_code=0,
        zooms=(1.75, 0.75, 2.5, 1.0),
    )
    manifest_path = _write_manifest(second_dir, _manifest(("patient002",)), "source.json")
    valid = _call(
        second_dir,
        manifest_path,
        {"patient002": good},
        geometry_policy=ACDC_PIXDIM_NATIVE_V1,
    )
    row = valid["audit_rows"][0]
    tampered = json.loads(json.dumps(row))
    tampered["reconstructed_affine_xyz"][0][3] += 10.0
    geometry = _manifest(("patient002",))["geometry"]["by_patient"]["patient002"]
    with pytest.raises(ValueError, match="reconstructed affine"):
        validate_audit_geometry(tampered, geometry, geometry_policy=ACDC_PIXDIM_NATIVE_V1)
    tampered["geometry_policy"] = STRICT_GEOMETRY_POLICY
    with pytest.raises(ValueError, match="geometry_policy"):
        validate_audit_geometry(tampered, geometry, geometry_policy=ACDC_PIXDIM_NATIVE_V1)


def test_missing_nifti_header_is_reported(tmp_path: Path, monkeypatch):
    cine = _write_nifti(tmp_path / "patient001_4d.nii.gz")
    manifest_path = _write_manifest(tmp_path, _manifest())

    class MissingHeaderImage:
        shape = (4, 5, 6, 2)
        header = None

    monkeypatch.setattr(nib, "load", lambda *args, **kwargs: MissingHeaderImage())
    report = _call(tmp_path, manifest_path, {"patient001": cine})

    assert report["status"] == "rejected"
    assert report["audit_rows"][0]["source_file_sha256"]
    assert any("header is missing" in item for item in report["audit_rows"][0]["errors"])


def test_unknown_units_need_explicit_declaration_with_provenance(tmp_path: Path, monkeypatch):
    cine = _write_nifti(tmp_path / "patient001_4d.nii.gz", unit="unknown")
    manifest_path = _write_manifest(tmp_path, _manifest())
    _forbid_voxel_reads(monkeypatch)

    rejected = _call(tmp_path, manifest_path, {"patient001": cine})
    assert rejected["status"] == "rejected"
    assert any("explicit unit declaration" in item for item in rejected["audit_rows"][0]["errors"])

    # Use separate output paths for the second audit because output files are never overwritten.
    mapping_source, mapping_sha, mapping_provenance = _mapping_inputs(tmp_path, {"patient001": cine})
    accepted = repair_manifest_geometry(
        manifest_path,
        _sha(manifest_path),
        {"patient001": cine},
        axis_map="XYZ->ZYX",
        cine_mapping_source=mapping_source,
        cine_mapping_sha256=mapping_sha,
        cine_mapping_provenance=mapping_provenance,
        report_json_path=tmp_path / "audit-explicit.json",
        report_csv_path=tmp_path / "audit-explicit.csv",
        unit_declarations_by_patient={
            "patient001": {"unit": "mm", "provenance": "synthetic explicit unit declaration"}
        },
    )
    assert accepted["status"] == "audit_only_pass"
    assert accepted["audit_rows"][0]["unit_resolution"] == "explicit_declaration"
    assert accepted["audit_rows"][0]["unit_provenance"] == "synthetic explicit unit declaration"


def test_pixdim_policy_requires_unknown_units_to_be_explicitly_declared_mm(tmp_path, monkeypatch):
    cine = _write_nifti(tmp_path / "patient001_4d.nii.gz", unit="unknown", qform_code=0)
    _forbid_voxel_reads(monkeypatch)

    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    missing_manifest = _write_manifest(missing_dir, _manifest())
    missing = _call(
        missing_dir, missing_manifest,
        {"patient001": cine}, geometry_policy=ACDC_PIXDIM_NATIVE_V1,
    )
    assert missing["status"] == "rejected"
    assert any("explicit unit declaration" in item for item in missing["audit_rows"][0]["errors"])

    mm_dir = tmp_path / "declared_mm"
    mm_dir.mkdir()
    mm_manifest = _write_manifest(mm_dir, _manifest())
    mm = _call(
        mm_dir,
        mm_manifest,
        {"patient001": cine},
        geometry_policy=ACDC_PIXDIM_NATIVE_V1,
        unit_declarations={"patient001": {"unit": "mm", "provenance": "synthetic explicit mm declaration"}},
    )
    assert mm["status"] == "audit_only_pass"
    assert mm["audit_rows"][0]["resolved_spatial_unit"] == "mm"
    assert mm["audit_rows"][0]["unit_provenance"] == "synthetic explicit mm declaration"

    m_dir = tmp_path / "declared_m"
    m_dir.mkdir()
    m_manifest = _write_manifest(m_dir, _manifest())
    meters = _call(
        m_dir,
        m_manifest,
        {"patient001": cine},
        geometry_policy=ACDC_PIXDIM_NATIVE_V1,
        unit_declarations={"patient001": {"unit": "m", "provenance": "synthetic non-mm declaration"}},
    )
    assert meters["status"] == "rejected"
    assert any("requires an explicit mm unit declaration" in item for item in meters["audit_rows"][0]["errors"])


def test_pixdim_policy_rejects_shear_and_conflicting_coded_orientation(tmp_path, monkeypatch):
    sheared_affine = np.asarray(
        [
            [1.25, 0.1, 0.0, 10.0],
            [0.0, np.sqrt(0.75**2 - 0.1**2), 0.0, 20.0],
            [0.0, 0.0, 2.5, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    sheared = _write_nifti(
        tmp_path / "patient001_4d.nii.gz", affine=sheared_affine, qform_code=0,
    )
    manifest_path = _write_manifest(tmp_path, _manifest())
    _forbid_voxel_reads(monkeypatch)
    rejected = _call(
        tmp_path,
        manifest_path,
        {"patient001": sheared},
        geometry_policy=ACDC_PIXDIM_NATIVE_V1,
    )
    assert rejected["status"] == "rejected"
    assert any("sheared" in item for item in rejected["audit_rows"][0]["errors"])

    conflict_dir = tmp_path / "orientation_conflict"
    conflict_dir.mkdir()
    conflict_affine = _AFFINE_XYZ.copy()
    conflict_affine[:3, 0] *= -1
    image = nib.Nifti1Image(np.zeros((4, 5, 6, 2), dtype=np.float32), _AFFINE_XYZ)
    image.set_sform(_AFFINE_XYZ, code=2)
    image.set_qform(conflict_affine, code=2)
    image.header.set_xyzt_units(xyz="mm", t="sec")
    conflict_path = conflict_dir / "patient001_4d.nii.gz"
    nib.save(image, str(conflict_path))
    conflict_manifest = _write_manifest(conflict_dir, _manifest())
    report = _call(
        conflict_dir,
        conflict_manifest,
        {"patient001": conflict_path},
        geometry_policy=ACDC_PIXDIM_NATIVE_V1,
    )
    assert report["status"] == "rejected"
    assert any("coded sform and qform disagree" in item for item in report["audit_rows"][0]["errors"])


def test_shape_mismatch_and_bad_patient_prevent_any_manifest_output(tmp_path: Path, monkeypatch):
    good = _write_nifti(tmp_path / "patient001_4d.nii.gz")
    bad_affine = np.eye(4)
    bad = _write_nifti(tmp_path / "patient002_4d.nii.gz", affine=bad_affine)
    manifest_path = _write_manifest(tmp_path, _manifest(("patient001", "patient002")))
    output_path = tmp_path / "corrected.json"
    _forbid_voxel_reads(monkeypatch)

    report = _call(
        tmp_path,
        manifest_path,
        {"patient001": good, "patient002": bad},
        output_manifest=output_path,
        audit_only=False,
    )

    assert report["status"] == "rejected"
    assert len(report["audit_rows"]) == 2
    assert all(row["source_header_sha256"] for row in report["audit_rows"])
    assert not output_path.exists()
    assert (tmp_path / "audit.json").exists()
    assert (tmp_path / "audit.csv").exists()


def test_shape_mismatch_is_reported_without_partial_manifest(tmp_path: Path, monkeypatch):
    cine = _write_nifti(tmp_path / "patient001_4d.nii.gz", shape_xyz_t=(4, 5, 7, 2))
    manifest_path = _write_manifest(tmp_path, _manifest())
    output_path = tmp_path / "corrected.json"
    _forbid_voxel_reads(monkeypatch)

    report = _call(
        tmp_path,
        manifest_path,
        {"patient001": cine},
        output_manifest=output_path,
        audit_only=False,
    )

    assert report["status"] == "rejected"
    assert any("does not match manifest" in item for item in report["audit_rows"][0]["errors"])
    assert not output_path.exists()


def test_record_guard_rejects_placeholder_spacing_and_identity_affines():
    base = _manifest()["geometry"]["by_patient"]["patient001"]
    nonidentity_unit_affine = np.eye(4)
    nonidentity_unit_affine[0, 3] = 5.0
    base["affine_zyx"] = nonidentity_unit_affine.tolist()
    with pytest.raises(ValueError, match="spacing \\[1,1,1\\]"):
        validate_geometry_record(base)

    base = _manifest()["geometry"]["by_patient"]["patient001"]
    with pytest.raises(ValueError, match="identity affine_zyx"):
        validate_geometry_record(base)

    base = _manifest()["geometry"]["by_patient"]["patient001"]
    # A unit-spacing source XYZ identity becomes a permutation in ZYX order.
    base["affine_zyx"] = np.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    ).tolist()
    with pytest.raises(ValueError, match="XYZ identity affine reordered"):
        validate_geometry_record(base)


def test_unit_conversion_cannot_hide_one_mm_placeholder(tmp_path, monkeypatch):
    affine = np.diag([1000.0, 1000.0, 1000.0, 1.0])
    affine[:3, 3] = [10000, 20000, 30000]
    cine = _write_nifti(tmp_path / "patient001_4d.nii.gz", affine=affine, unit="micron")
    manifest = _write_manifest(tmp_path, _manifest())
    _forbid_voxel_reads(monkeypatch)
    output = tmp_path / "corrected.json"
    report = _call(tmp_path, manifest, {"patient001": cine}, output_manifest=output, audit_only=False)
    assert report["status"] == "rejected"
    assert any("spacing [1,1,1]" in error for error in report["audit_rows"][0]["errors"])
    assert not output.exists()


def test_nonfinite_header_still_emits_valid_diagnostics(tmp_path, monkeypatch):
    cine = _write_nifti(tmp_path / "patient001_4d.nii.gz")
    image = nib.load(cine)
    header = image.header.copy()
    header["srow_x"][0] = np.nan
    monkeypatch.setattr(nib, "load", lambda *_args, **_kwargs: SimpleNamespace(header=header, shape=image.shape))
    manifest = _write_manifest(tmp_path, _manifest())
    report = _call(tmp_path, manifest, {"patient001": cine})
    assert report["status"] == "rejected"
    saved = json.loads((tmp_path / "audit.json").read_text())
    assert saved["audit_rows"][0]["selected_affine_xyz"][0][0] == "nan"
    assert any("finite" in error for error in saved["audit_rows"][0]["errors"])
    assert (tmp_path / "audit.csv").is_file()


def test_output_failure_rolls_back_only_new_batch(tmp_path, monkeypatch):
    from self_audit.candidate_q3d import geometry_repair

    cine = _write_nifti(tmp_path / "patient001_4d.nii.gz")
    manifest = _write_manifest(tmp_path, _manifest())
    source_bytes = manifest.read_bytes()
    output = tmp_path / "corrected.json"
    original_write = geometry_repair._write_new

    def fail_manifest(path, payload):
        if path == output:
            raise OSError("synthetic publication failure")
        return original_write(path, payload)

    monkeypatch.setattr(geometry_repair, "_write_new", fail_manifest)
    with pytest.raises(OSError, match="publication failure"):
        _call(tmp_path, manifest, {"patient001": cine}, output_manifest=output, audit_only=False)
    assert not output.exists()
    assert not (tmp_path / "audit.json").exists()
    assert not (tmp_path / "audit.csv").exists()
    assert manifest.read_bytes() == source_bytes
def test_output_close_failure_removes_only_new_partial_file(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from self_audit.candidate_q3d.geometry_repair import _write_new

    output = tmp_path / "new.json"
    original_open = Path.open

    @contextmanager
    def fail_on_close(path, *args, **kwargs):
        with original_open(path, *args, **kwargs) as stream:
            yield stream
        raise OSError("simulated close failure")

    monkeypatch.setattr(Path, "open", fail_on_close)
    with pytest.raises(OSError, match="simulated close failure"):
        _write_new(output, b"serialized output")
    assert not output.exists()
