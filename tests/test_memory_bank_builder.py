import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from self_audit.memory_bank import (
    PAIRS,
    boundary_pair_candidates,
    build_memory_bank,
    create_native_image_manifest,
    pack_bank,
    patient_balanced_select,
    _require_a0_student,
    sha256_file,
)


def _sha(path):
    return sha256_file(path)


def _save_nifti(nib, path, array, affine, units="mm"):
    image = nib.Nifti1Image(array, affine)
    image.header.set_xyzt_units(xyz=units)
    nib.save(image, path)


def _interface_prediction(shape=(2, 2, 2), confidence=.95):
    probs = np.zeros((4, *shape), dtype=np.float32)
    probs[0, 0] = confidence
    probs[1, 0] = 1 - confidence
    probs[1, 1] = confidence
    probs[0, 1] = 1 - confidence
    feats = np.zeros((32, *shape), dtype=np.float32)
    feats[0] = 1.0
    feats[1] = np.indices(shape)[0] + 1
    return probs, feats


def _record(pid, pair=(0, 1), rank=None):
    return {
        "patient": pid, "timepoint": 0, "class_pair": list(pair),
        "endpoints": [([1.0] + [0.0] * 31), ([0.0, 1.0] + [0.0] * 30)],
        "confidence": [.9, .9], "coord_c_xyz": [0, 0, 0], "coord_d_xyz": [1, 0, 0],
        "world_c_mm": [0.0, 0.0, 0.0], "world_d_mm": [1.0, 0.0, 0.0],
        "distance_mm": 1.0, "image_sha256": "a" * 64, "source_sha256": "b" * 64,
        "feature_source": "test", "_rank": rank or f"{pid}-rank",
    }


def test_pair_extraction_uses_physical_distance_confidence_and_native_coordinates():
    probs, feats = _interface_prediction()
    affine = np.eye(4)
    affine[:3, 3] = [10.0, 20.0, 30.0]
    found = boundary_pair_candidates(probs, feats, affine, patient="patient001", timepoint=2,
        image_sha256="a" * 64, source_sha256="b" * 64, source_kind="student")
    assert found[(0, 1)]
    entry = found[(0, 1)][0]
    assert entry["coord_c_xyz"][0] == 0
    assert entry["coord_d_xyz"][0] == 1
    np.testing.assert_allclose(entry["world_c_mm"], np.array(entry["coord_c_xyz"]) + [10, 20, 30])
    np.testing.assert_allclose(entry["world_d_mm"], np.array(entry["coord_d_xyz"]) + [10, 20, 30])
    assert entry["distance_mm"] <= 3.0
    assert entry["timepoint"] == 2
    assert entry["confidence"] == pytest.approx([.95, .95])

    spaced = np.eye(4)
    spaced[0, 0] = 4.0
    assert not boundary_pair_candidates(probs, feats, spaced, patient="patient001", timepoint=0,
        image_sha256="a" * 64, source_sha256="b" * 64, source_kind="student")[(0, 1)]


def test_confidence_below_threshold_is_not_selected():
    probs, feats = _interface_prediction(confidence=.79)
    found = boundary_pair_candidates(probs, feats, np.eye(4), patient="patient001", timepoint=0,
        image_sha256="a" * 64, source_sha256="b" * 64, source_kind="student")
    assert found[(0, 1)] == []


def test_teacher_valid_mask_excludes_unsupported_endpoints():
    probs, feats = _interface_prediction()
    valid = np.ones(probs.shape[1:], dtype=bool)
    valid[1, :, :] = False
    probs[:, 1, :, :] = 0.0
    found = boundary_pair_candidates(probs, feats, np.eye(4), patient="patient001", timepoint=0,
        image_sha256="a" * 64, source_sha256="b" * 64, source_kind="teacher", valid_mask=valid)
    assert found[(0, 1)] == []


def test_pair_candidates_cannot_cross_an_intervening_predicted_class():
    labels = np.array([0, 2, 1], dtype=np.int64).reshape(3, 1, 1)
    probs = np.zeros((4, 3, 1, 1), dtype=np.float32)
    for cls in range(3):
        probs[cls][labels == cls] = 1.0
    feats = np.ones((32, 3, 1, 1), dtype=np.float32)
    found = boundary_pair_candidates(probs, feats, np.diag([1.0, 1.0, 10.0, 1.0]),
        patient="patient001", timepoint=0, image_sha256="a" * 64,
        source_sha256="b" * 64, source_kind="test")
    assert found[(0, 1)] == []


def test_pair_candidates_cannot_bridge_repeated_c_d_interfaces():
    labels = np.array([0, 1, 0, 1], dtype=np.int64).reshape(4, 1, 1)
    probs = np.zeros((4, 4, 1, 1), dtype=np.float32)
    for cls in range(2):
        probs[cls][labels == cls] = .95
        probs[1 - cls][labels == cls] = .05
    # The nearer d endpoint is still predicted d, but has low confidence.
    probs[:, 1, 0, 0] = [0.49, 0.51, 0.0, 0.0]
    feats = np.ones((32, 4, 1, 1), dtype=np.float32)
    found = boundary_pair_candidates(probs, feats, np.diag([1.0, 1.0, 10.0, 1.0]),
        patient="patient001", timepoint=0, image_sha256="a" * 64,
        source_sha256="b" * 64, source_kind="test")
    outer = [entry for entry in found[(0, 1)]
             if entry["coord_c_xyz"] == [0, 0, 0] and entry["coord_d_xyz"] == [3, 0, 0]]
    assert outer == []


def test_bank_keeps_all_six_groups_and_empty_groups_zero():
    selected = {pair: [] for pair in PAIRS}
    selected[(0, 1)] = [_record("patient001")]
    packed = pack_bank(selected, {"fixture": True})
    assert tuple(packed["endpoints"].shape) == (6, 8, 2, 32)
    assert packed["valid"].sum().item() == 1
    assert not packed["valid"][1:].any()
    assert not packed["endpoints"][1:].any()
    assert not packed["confidences"][1:].any()


def test_patient_balanced_selection_is_deterministic():
    groups = {p: {pair: [] for pair in PAIRS} for p in ("patient001", "patient002", "patient003")}
    for pid in groups:
        groups[pid][(0, 1)] = [_record(pid, rank=f"{pid}-{j}") for j in range(3)]
    a = patient_balanced_select(groups)[(0, 1)]
    b = patient_balanced_select(groups)[(0, 1)]
    assert [x["_rank"] for x in a] == [x["_rank"] for x in b]
    assert len(a) == 8
    assert len({x["patient"] for x in a[:3]}) == 3


def test_prepare_manifest_never_reads_labeled_mask_paths(tmp_path):
    nib = pytest.importorskip("nibabel")
    prepared = tmp_path / "patient001.npy"
    np.save(prepared, np.zeros((2, 2, 2, 2), dtype=np.float32), allow_pickle=False)
    root = tmp_path / "native"
    folder = root / "patient001"
    folder.mkdir(parents=True)
    raw = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    native = folder / "patient001_4d.nii.gz"
    _save_nifti(nib, native, raw, np.eye(4))
    train = tmp_path / "train.json"
    train.write_text(json.dumps({
        "schema": "self_audit.data.v2", "role": "training", "synthetic": False,
        "label_budget": 1,
        "labeled": [{"id": "patient001", "image": prepared.name, "image_sha256": _sha(prepared),
                     "masks": {"0": {"path": "does-not-exist-mask.npy", "sha256": "0" * 64}}}],
        "unlabeled": [], "evaluation_ids": ["patient002"],
    }))
    output = tmp_path / "native_manifest.json"
    result = create_native_image_manifest(train, _sha(train), root, output)
    assert result["image_reads"]["mask_paths_read"] == 0
    assert result["patients"][0]["affine_xyz_to_world_mm"] == np.eye(4).tolist()


def test_train_evaluation_overlap_fails_before_native_image_open(tmp_path, monkeypatch):
    nib = pytest.importorskip("nibabel")
    prepared = tmp_path / "patient001.npy"
    np.save(prepared, np.zeros((1, 1, 1, 1), dtype=np.float32), allow_pickle=False)
    native = tmp_path / "native" / "patient001" / "patient001_4d.nii.gz"
    native.parent.mkdir(parents=True)
    _save_nifti(nib, native, np.zeros((1, 1, 1, 1), dtype=np.float32), np.eye(4))
    train = tmp_path / "train.json"
    train.write_text(json.dumps({
        "schema": "self_audit.data.v2", "role": "training", "synthetic": False,
        "label_budget": 0,
        "labeled": [], "unlabeled": [{"id": "patient001", "image": prepared.name, "image_sha256": _sha(prepared)}],
        "evaluation_ids": ["patient001"],
    }))
    monkeypatch.setattr(nib, "load", lambda *a, **kw: pytest.fail("image opened before overlap validation"))
    with pytest.raises(ValueError, match="overlap"):
        create_native_image_manifest(train, _sha(train), tmp_path / "native", tmp_path / "out.json")


def test_native_spatial_units_are_explicitly_converted_or_rejected(tmp_path):
    nib = pytest.importorskip("nibabel")
    prepared = tmp_path / "patient001.npy"
    np.save(prepared, np.zeros((1, 1, 1, 1), dtype=np.float32), allow_pickle=False)
    root = tmp_path / "native"
    native = root / "patient001" / "patient001_4d.nii.gz"
    native.parent.mkdir(parents=True)
    affine_meter = np.diag([.001, .001, .001, 1.0])
    _save_nifti(nib, native, np.zeros((1, 1, 1, 1), dtype=np.float32), affine_meter, units="meter")
    train = tmp_path / "train.json"
    train.write_text(json.dumps({"schema": "self_audit.data.v2", "role": "training", "synthetic": False,
        "label_budget": 0, "labeled": [], "unlabeled": [{"id": "patient001", "image": prepared.name,
        "image_sha256": _sha(prepared)}], "evaluation_ids": []}))
    result = create_native_image_manifest(train, _sha(train), root, tmp_path / "meter_manifest.json")
    assert result["patients"][0]["spatial_units"] == "meter"
    np.testing.assert_allclose(result["patients"][0]["affine_xyz_to_world_mm"], np.eye(4))

    _save_nifti(nib, native, np.zeros((1, 1, 1, 1), dtype=np.float32), np.eye(4), units="unknown")
    train.write_text(json.dumps({"schema": "self_audit.data.v2", "role": "training", "synthetic": False,
        "label_budget": 0, "labeled": [], "unlabeled": [{"id": "patient001", "image": prepared.name,
        "image_sha256": _sha(prepared)}], "evaluation_ids": []}))
    with pytest.raises(ValueError, match="spatial units"):
        create_native_image_manifest(train, _sha(train), root, tmp_path / "unknown_manifest.json")


def test_native_image_hash_is_checked_before_opening_image(tmp_path, monkeypatch):
    nib = pytest.importorskip("nibabel")
    prepared = tmp_path / "patient001.npy"
    np.save(prepared, np.zeros((1, 1, 1, 1), dtype=np.float32), allow_pickle=False)
    native = tmp_path / "native" / "patient001" / "patient001_4d.nii.gz"
    native.parent.mkdir(parents=True)
    _save_nifti(nib, native, np.zeros((1, 1, 1, 1), dtype=np.float32), np.eye(4))
    train = tmp_path / "train.json"
    train.write_text(json.dumps({"schema": "self_audit.data.v2", "role": "training", "synthetic": False,
        "label_budget": 0, "labeled": [], "unlabeled": [{"id": "patient001", "image": prepared.name,
        "image_sha256": _sha(prepared)}], "evaluation_ids": []}))
    native_manifest = create_native_image_manifest(train, _sha(train), tmp_path / "native", tmp_path / "manifest.json")
    native_manifest["patients"][0]["native_image_sha256"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(native_manifest))
    monkeypatch.setattr(nib, "load", lambda *a, **kw: pytest.fail("image opened before hash validation"))
    from self_audit.memory_bank import load_native_manifest
    with pytest.raises(ValueError, match="native image hash mismatch"):
        load_native_manifest(manifest_path, _sha(manifest_path))


def test_a5_refuses_external_teacher_object():
    with pytest.raises(ValueError, match="A5 requires only a frozen student"):
        build_memory_bank("unused", "0" * 64, mode="a5", adapter=object(),
                          student_checkpoint="unused", student_manifest_sha256="1" * 64)


def test_teacher_builder_refuses_unpinned_or_statusless_adapter():
    class UnpinnedAdapter:
        synthetic = False

    with pytest.raises(ValueError, match="source/config/weights"):
        build_memory_bank("unused", "0" * 64, mode="teacher", adapter=UnpinnedAdapter())


@pytest.mark.parametrize("arm", ["A1", "A2", "A3", "A4", "A5", "A6"])
def test_a5_rejects_memory_bearing_student_ancestry(arm):
    class Student:
        teacher_memory = None

    meta = {"synthetic": False, "teacher_memory": None,
        "data_sha256": "d" * 64, "labeled_ids": ["patient001"],
        "unlabeled_ids": ["patient002"], "label_budget": 1, "audit_label_reads": ["patient001"],
        "config": {"teacher_memory": {"arm": arm, "bank_path": None, "bank_sha256": None,
            "cache_path": None, "cache_sha256": None}}}
    with pytest.raises(ValueError, match="A0 student checkpoint"):
        _require_a0_student(Student(), meta, data_sha256="d" * 64,
                            labeled_ids=["patient001"], unlabeled_ids=["patient002"])


def test_a5_accepts_only_explicit_clean_a0_student_ancestry():
    class Student:
        teacher_memory = None

    meta = {"synthetic": False, "teacher_memory": None, "data_sha256": "d" * 64,
        "labeled_ids": ["patient001"], "unlabeled_ids": ["patient002"],
        "label_budget": 1, "audit_label_reads": ["patient001"],
        "config": {"teacher_memory": {"arm": "A0", "bank_path": None, "bank_sha256": None,
            "cache_path": None, "cache_sha256": None}}}
    assert _require_a0_student(Student(), meta, data_sha256="d" * 64,
        labeled_ids=["patient001"], unlabeled_ids=["patient002"]) == "A0"


@pytest.mark.parametrize("field,value", [
    ("data_sha256", "e" * 64), ("labeled_ids", ["patient003"]),
    ("unlabeled_ids", ["patient004"]), ("label_budget", 2),
])
def test_a5_rejects_mismatched_train_data_or_label_budget(field, value):
    class Student:
        teacher_memory = None

    meta = {"synthetic": False, "teacher_memory": None, "data_sha256": "d" * 64,
        "labeled_ids": ["patient001"], "unlabeled_ids": ["patient002"],
        "label_budget": 1, "audit_label_reads": [],
        "config": {"teacher_memory": {"arm": "A0", "bank_path": None, "bank_sha256": None,
            "cache_path": None, "cache_sha256": None}}}
    meta[field] = value
    with pytest.raises(ValueError, match="A0 student checkpoint"):
        _require_a0_student(Student(), meta, data_sha256="d" * 64,
            labeled_ids=["patient001"], unlabeled_ids=["patient002"])


def test_a1_writer_refuses_partial_coverage():
    from self_audit.memory_bank import write_a1_cache
    with pytest.raises(ValueError, match="partial A1 cache"):
        write_a1_cache("unused", {"a1_records": [("patient001", 0, np.zeros((1, 4, 1, 1)), {}, {})],
            "coverage": {"complete": False}})


def test_a1_writer_declares_canonical_semantic_class_map(tmp_path):
    from self_audit.memory_bank import write_a1_cache
    affine = np.eye(4).tolist()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    probability_path = cache_dir / "patient001_probabilities.npy"
    support_path = cache_dir / "patient001_valid_mask.npy"
    np.save(probability_path, np.full((1, 2, 4, 3, 4), .25, dtype=np.float32), allow_pickle=False)
    np.save(support_path, np.ones((1, 2, 3, 4), dtype=bool), allow_pickle=False)
    build_result = {
        "a1_records": [{"id": "patient001", "probabilities_path": str(probability_path),
                         "valid_support_path": str(support_path), "timepoints": [0]}],
        "coverage": {"complete": True, "train_patients_total": 1},
        "bank": {"metadata": {"provenance": {
            "image_hashes": {"patient001": "a" * 64},
            "native_image_hashes": {"patient001": "b" * 64},
            "native_manifest_sha256": "c" * 64,
        }}},
        "manifest": {"evaluation_ids": ["patient002"], "patients": [{
            "id": "patient001", "shape_xyzt": [4, 3, 2, 1], "affine_xyz_to_world_mm": affine,
        }]},
    }
    cache_path = write_a1_cache(cache_dir, build_result)
    cache = json.loads(cache_path.read_text())
    assert cache["class_map"] == ["BG", "RV", "MYO", "LV"]
    assert cache["records"][0]["shape"] == [1, 2, 4, 3, 4]
    support_meta = cache["provenance"]["native_geometry_by_patient"]["patient001"]["valid_support"]
    assert support_meta["shape_tzyx"] == [1, 2, 3, 4]
    saved_support = np.load(cache_path.parent / support_meta["path"], allow_pickle=False)
    assert saved_support.dtype == np.bool_ and saved_support.all()
