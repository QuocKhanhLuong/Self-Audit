"""SOFTWARE_ONLY checks: invalid ACDC geometry cannot reach voxel readers."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import q3d_preflight
from self_audit.candidate_q3d import engine
from self_audit.candidate_q3d.data import Q3DData, validate_acdc_geometry
from self_audit.data import CLASS_MAPPING, GEOMETRY_MAP_SCHEMA, GEOMETRY_SCHEMA


def _manifest():
    affine = [[0, 0, 1.5, 12], [0, 1.5, 0, -8], [10, 0, 0, 5], [0, 0, 0, 1]]
    geometry = {
        "schema": GEOMETRY_SCHEMA,
        "array_axes": ["T", "Z", "Y", "X"],
        "spatial_axes": ["Z", "Y", "X"],
        "shape_zyx": [9, 32, 32],
        "spacing_zyx_mm": [10, 1.5, 1.5],
        "spacing_yx_mm": [1.5, 1.5],
        "affine_zyx": affine,
        "class_mapping": CLASS_MAPPING,
    }
    return {
        "schema": "self_audit.data.v2", "dataset": "acdc", "role": "training",
        "synthetic": False,
        "labeled": [{"id": "patient001"}], "unlabeled": [{"id": "patient002"}],
        "geometry": {
            "schema": GEOMETRY_MAP_SCHEMA,
            "by_patient": {pid: copy.deepcopy(geometry) for pid in ("patient001", "patient002")},
        },
    }


def _invalidate(manifest, failure):
    geometry = manifest["geometry"]["by_patient"]["patient001"]
    if failure == "unit_spacing":
        geometry["spacing_zyx_mm"] = [1, 1, 1]
        geometry["spacing_yx_mm"] = [1, 1]
        geometry["affine_zyx"] = [[0, 0, 1, 12], [0, 1, 0, -8], [1, 0, 0, 5], [0, 0, 0, 1]]
    elif failure == "identity_affine":
        geometry["affine_zyx"] = np.eye(4).tolist()
    elif failure == "missing_affine":
        del geometry["affine_zyx"]
    elif failure == "missing_patient":
        del manifest["geometry"]["by_patient"]["patient002"]
    else:
        raise AssertionError(failure)


def _pin(tmp_path, manifest):
    path = tmp_path / "train.json"
    path.write_text(json.dumps(manifest))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _forbidden(*_args, **_kwargs):
    pytest.fail("invalid geometry reached a voxel/GT reader")


@pytest.mark.parametrize("failure", ["unit_spacing", "identity_affine", "missing_affine", "missing_patient"])
def test_engine_rejects_geometry_before_baseline_loader(tmp_path, monkeypatch, failure):
    from self_audit.candidate_a3d import data as baseline

    manifest = _manifest()
    _invalidate(manifest, failure)
    spec = _pin(tmp_path, manifest)
    monkeypatch.setattr(baseline, "CandidateA3DData", _forbidden)
    config = {"runtime": {"synthetic": False}, "dataset": {"manifests": {"train": spec}}}
    with pytest.raises(ValueError, match="ACDC"):
        engine._load_data(config, None)


@pytest.mark.parametrize("failure", ["unit_spacing", "identity_affine", "missing_affine", "missing_patient"])
def test_preflight_rejects_geometry_before_gt(tmp_path, monkeypatch, failure):
    manifest = _manifest()
    _invalidate(manifest, failure)
    spec = _pin(tmp_path, manifest)
    monkeypatch.setattr(q3d_preflight, "_ground_truth_physiology", _forbidden)
    monkeypatch.setattr(q3d_preflight, "_ground_truth_stage4_coverage", _forbidden)
    report = q3d_preflight._manifest_diagnostics("acdc", spec, config_path=tmp_path / "recipe.json")
    assert report["status"] == "INVALID_GEOMETRY"
    assert report["original_header_audit"]["status"] == "NOT_RUN"


def test_valid_metadata_is_not_promoted_to_original_header_evidence(tmp_path, monkeypatch):
    spec = _pin(tmp_path, _manifest())
    skipped = {"status": "NOT_RUN", "reason": "SOFTWARE_ONLY test has no GT arrays"}
    monkeypatch.setattr(q3d_preflight, "_ground_truth_physiology", lambda *_: skipped)
    monkeypatch.setattr(q3d_preflight, "_ground_truth_stage4_coverage", lambda *_: skipped)
    report = q3d_preflight._manifest_diagnostics("acdc", spec, config_path=tmp_path / "recipe.json")
    assert report["status"] == "READ_ONLY_MANIFEST_DIAGNOSTICS"
    assert report["spacing_zyx_mm"]["z"]["median"] == 10
    assert report["r_sv_r_m_on_labeled_train_ground_truth"]["status"] == "NOT_RUN"


def test_synthetic_override_requires_both_explicit_runtime_and_manifest_identity():
    manifest = _manifest()
    _invalidate(manifest, "unit_spacing")
    manifest["synthetic"] = True
    validate_acdc_geometry(manifest, synthetic=True)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_acdc_geometry(manifest, synthetic=False)
    manifest["synthetic"] = False
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_acdc_geometry(manifest, synthetic=True)


def test_engine_hash_check_precedes_geometry_and_array_access(tmp_path, monkeypatch):
    from self_audit.candidate_a3d import data as baseline

    spec = _pin(tmp_path, _manifest())
    spec["sha256"] = "0" * 64
    monkeypatch.setattr(baseline, "CandidateA3DData", _forbidden)
    config = {"runtime": {"synthetic": False}, "dataset": {"manifests": {"train": spec}}}
    with pytest.raises(ValueError, match="hash mismatch"):
        engine._load_data(config, None)


@pytest.mark.parametrize("value", ["false", "true", 1, 0, None])
def test_nonboolean_synthetic_identity_cannot_enable_fixture_bypass(value):
    manifest = _manifest()
    manifest["synthetic"] = value
    with pytest.raises(ValueError, match="booleans"):
        validate_acdc_geometry(manifest, synthetic=True)
    manifest["synthetic"] = False
    with pytest.raises(ValueError, match="booleans"):
        validate_acdc_geometry(manifest, synthetic=value)


@pytest.mark.parametrize("declared", [None, "mnms"])
def test_real_manifest_cannot_hide_acdc_identity_from_config(declared):
    manifest = _manifest()
    manifest["dataset"] = declared
    with pytest.raises(ValueError, match="dataset"):
        validate_acdc_geometry(manifest, synthetic=False, dataset_name="acdc")


def test_protocol_b_validates_teacher_metadata_before_any_array_reader(tmp_path, monkeypatch):
    from self_audit.candidate_a3d import data as baseline

    student_dir = tmp_path / "student"
    teacher_dir = tmp_path / "teacher"
    student_dir.mkdir()
    teacher_dir.mkdir()
    student = _pin(student_dir, _manifest())
    bad_teacher = _manifest()
    _invalidate(bad_teacher, "unit_spacing")
    teacher = _pin(teacher_dir, bad_teacher)
    monkeypatch.setattr(baseline, "CandidateA3DData", _forbidden)
    config = {
        "runtime": {"synthetic": False}, "protocol": {"variant": "B"},
        "dataset": {"name": "acdc", "manifests": {"train": student, "teacher_train": teacher}},
    }
    with pytest.raises(ValueError, match="ACDC"):
        engine._load_data(config, None)


def test_supplied_view_cannot_bypass_guard_before_further_engine_access():
    # No constructor or patient arrays: represent an externally supplied view
    # whose raw metadata has been changed after its creation.
    manifest = _manifest()
    _invalidate(manifest, "unit_spacing")
    data = object.__new__(Q3DData)
    data.student_data = SimpleNamespace(manifest=manifest)
    data.teacher_data = SimpleNamespace(manifest=manifest)
    config = {"runtime": {"synthetic": False}, "dataset": {"name": "acdc"}}
    with pytest.raises(ValueError, match="ACDC"):
        engine._load_data(config, data)
