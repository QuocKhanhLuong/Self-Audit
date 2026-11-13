from __future__ import annotations

import copy

import nibabel as nib
import numpy as np
import pytest

from self_audit.candidate_fuse.geometry import adjudicate_geometry
from self_audit.candidate_fuse.geometry import UNITS_EVIDENCE_SHA256, UNITS_EVIDENCE_URL


def _policy(**updates):
    value = {
        "schema": "self_audit.mnms_geometry_policy.v1",
        "name": "mnms_open345_mm_polar_v1",
        "dataset_release": "original_2020_open345",
        "units_evidence_url": UNITS_EVIDENCE_URL,
        "units_evidence_sha256": UNITS_EVIDENCE_SHA256,
        "source_identities_sha256": "b" * 64,
        "reason": "Locked public release acquisition metadata interpretation",
    }
    value.update(updates)
    return value


def _header(shape=(5, 6, 7), affine=None, unit="mm", *, sform_code=1, qform_code=0):
    affine = np.eye(4) if affine is None else np.asarray(affine, dtype=float)
    hdr = nib.Nifti1Header()
    hdr.set_data_shape(shape)
    hdr.set_data_dtype(np.float32)
    hdr.set_xyzt_units(unit, "sec")
    hdr.set_sform(affine, code=sform_code)
    if qform_code:
        hdr.set_qform(affine, code=qform_code)
    else:
        hdr["qform_code"] = 0
    hdr["pixdim"][1:4] = np.linalg.norm(affine[:3, :3], axis=0)
    return hdr


def test_generic_orthogonal_sform_preserves_native_contract_and_header_bytes():
    affine = np.array([
        [0.0, -1.25, 0.0, 4.0],
        [1.5, 0.0, 0.0, -3.0],
        [0.0, 0.0, -8.0, 12.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    header = _header((9, 11, 13, 2), affine)
    before = bytes(header.binaryblock)
    result = adjudicate_geometry(header, policy=_policy())
    assert bytes(header.binaryblock) == before
    assert result["canonical_valid"] is True
    assert result["policy_name"] == "mnms_open345_mm_polar_v1"
    assert result["geometry"]["shape"] == (13, 11, 9)
    assert result["geometry"]["spacing_zyx"] == (8.0, 1.25, 1.5)
    np.testing.assert_allclose(result["effective_affine_xyz"], affine, atol=1e-12)
    expected_zyx = np.eye(4)
    expected_zyx[:3, :3] = affine[:3, [2, 1, 0]]
    expected_zyx[:3, 3] = affine[:3, 3]
    np.testing.assert_allclose(result["geometry"]["affine_zyx"], expected_zyx)
    assert np.linalg.det(np.asarray(result["effective_affine_xyz"])[:3, :3]) < 0
    assert result["max_corner_displacement_mm"] == pytest.approx(0.0, abs=1e-12)


def test_actual_c8_j7l5_raw_header_is_polar_adjudicated_without_mutation():
    affine = np.array([
        [-0.5526717901229858, 1.0454227924346924, 6.04373025894165, -150.75],
        [-1.255924105644226, 0.015040089376270771, -5.3297624588012695, 216.1219940185547],
        [-0.5662614703178406, -1.0537117719650269, 5.921735763549805, 203.75100708007812],
        [0.0, 0.0, 0.0, 1.0],
    ])
    header = _header((256, 256, 10, 25), affine, unit="unknown")
    header["pixdim"][1:4] = (1.4844000339508057, 1.4844000339508057, 10.0)
    before = bytes(header.binaryblock)

    result = adjudicate_geometry(header, policy=_policy())

    assert bytes(header.binaryblock) == before
    assert result["raw_unit"] == "unknown"
    assert result["geometry"]["shape"] == (10, 256, 256)
    assert result["gram_error"] == pytest.approx(0.0001151592688, abs=1e-10)
    assert result["max_corner_displacement_mm"] == pytest.approx(0.0224900593, abs=1e-6)
    assert result["canonical_valid"] is True
    effective = np.asarray(result["effective_affine_xyz"])
    directions = effective[:3, :3] / np.linalg.norm(effective[:3, :3], axis=0)
    np.testing.assert_allclose(directions.T @ directions, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(effective[:3, :3], axis=0),
                               (1.4844000339508057, 1.4844000339508057, 10.0), atol=1e-12)


def test_header_only_result_is_deterministic_and_ignores_voxel_contents():
    header = _header()
    first = adjudicate_geometry(header, policy=_policy())
    second = adjudicate_geometry(copy.deepcopy(header), policy=_policy())
    assert first == second


@pytest.mark.parametrize("updates", [
    {"schema": "wrong"},
    {"name": "other"},
    {"dataset_release": "complete_challenge"},
    {"units_evidence_url": "http://example.org/evidence"},
    {"units_evidence_url": "https://"},
    {"units_evidence_url": "https://example.org/arbitrary"},
    {"units_evidence_sha256": "z" * 64},
    {"units_evidence_sha256": "a" * 64},
    {"source_identities_sha256": "0" * 63},
    {"reason": "  "},
    {"max_corner_displacement_mm": 100.0},
])
def test_invalid_policy_and_caller_tolerance_override_are_rejected(updates):
    with pytest.raises(ValueError):
        adjudicate_geometry(_header(), policy=_policy(**updates))


@pytest.mark.parametrize(("unit", "scale"), [("meter", 1000.0), ("micron", 0.001)])
def test_known_units_convert_affine_translation_and_spacing(unit, scale):
    native = 0.001 if unit == "meter" else 1000.0
    affine = np.diag([native, native * 1.5, native * 8.0, 1.0])
    affine[:3, 3] = np.asarray([12.0, -3.0, 25.0]) / scale
    result = adjudicate_geometry(_header(affine=affine, unit=unit), policy=_policy())
    assert result["raw_unit"] == unit
    assert result["effective_unit"] == "mm"
    np.testing.assert_allclose(result["effective_affine_xyz"], affine * np.array([[scale], [scale], [scale], [1]]))
    assert result["geometry"]["spacing_zyx"] == pytest.approx((8.0, 1.5, 1.0))


def test_unknown_units_require_policy_and_are_reported_as_interpretation():
    header = _header(unit="unknown")
    result = adjudicate_geometry(header, policy=_policy())
    assert result["raw_unit"] == "unknown"
    assert result["unit_interpretation"] == "unknown_raw_unit_interpreted_as_mm_by_explicit_policy"
    assert result["effective_unit"] == "mm"


@pytest.mark.parametrize("policy", [None, {"schema": "self_audit.mnms_geometry_policy.v1"}])
def test_unknown_units_without_complete_explicit_policy_are_rejected(policy):
    with pytest.raises(ValueError, match="policy"):
        adjudicate_geometry(_header(unit="unknown"), policy=policy)


def test_material_shear_exceeds_fixed_normalized_gram_bound():
    affine = np.eye(4)
    affine[0, 1] = 0.02
    with pytest.raises(ValueError, match="Gram error"):
        adjudicate_geometry(_header(affine=affine), policy=_policy())


def test_corner_displacement_bound_rejects_small_shear_on_large_volume():
    affine = np.eye(4)
    affine[0, 1] = 0.0002
    with pytest.raises(ValueError, match="native corner"):
        adjudicate_geometry(_header(shape=(512, 512, 64), affine=affine), policy=_policy())


def test_nonfinite_singular_pixdim_mismatch_and_invalid_shape_rejected():
    header = _header()
    header["srow_x"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        adjudicate_geometry(header, policy=_policy())
    header = _header()
    header["srow_z"][:3] = 0.0
    header["pixdim"][1:4] = 1.0
    with pytest.raises(ValueError, match="nonsingular"):
        adjudicate_geometry(header, policy=_policy())
    header = _header()
    header["pixdim"][1] = 2.0
    with pytest.raises(ValueError, match="pixdim"):
        adjudicate_geometry(header, policy=_policy())
    with pytest.raises(ValueError, match="shape"):
        adjudicate_geometry(_header(shape=(1, 6, 7)), policy=_policy())


def test_missing_transform_codes_rejected():
    header = _header(sform_code=0, qform_code=0)
    with pytest.raises(ValueError, match="coded sform or qform"):
        adjudicate_geometry(header, policy=_policy())


def test_coded_qform_is_accepted_when_sform_is_uncoded():
    affine = np.diag([1.25, 1.5, 8.0, 1.0])
    header = _header(affine=affine, sform_code=0, qform_code=1)
    result = adjudicate_geometry(header, policy=_policy())
    assert result["affine_source"] == "qform"
    assert result["canonical_valid"]


def test_explicit_axis_permutation_and_reflection_are_retained():
    affine = np.array([
        [0.0, -1.2, 0.0, 2.0],
        [0.0, 0.0, 2.5, 3.0],
        [7.0, 0.0, 0.0, -4.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    result = adjudicate_geometry(_header((4, 5, 6), affine), policy=_policy())
    assert result["geometry"]["shape"] == (6, 5, 4)
    assert result["geometry"]["spacing_zyx"] == pytest.approx((2.5, 1.2, 7.0))
    assert np.linalg.det(np.asarray(result["effective_affine_xyz"])[:3, :3]) < 0
