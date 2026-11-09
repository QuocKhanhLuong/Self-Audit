import json

import numpy as np
import pytest

from prepare_acdc import native_geometry_zyx
from self_audit.candidate_a.data import CandidateAData, synthetic_candidate_a_fixture
from self_audit.data import TrainingData, digest, write_json


def test_native_geometry_reorders_spacing_and_affine():
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag([1.1, 2.2, 3.3])
    affine[:3, 3] = [10.0, -2.0, 4.0]
    geometry = native_geometry_zyx(affine, (5, 6, 7))
    assert geometry["spacing_zyx_mm"] == [3.3, 2.2, 1.1]
    assert geometry["spacing_yx_mm"] == [2.2, 1.1]
    cache_affine = np.asarray(geometry["affine_zyx"])
    np.testing.assert_allclose(
        cache_affine[:3, :3],
        np.asarray([[0.0, 0.0, 1.1], [0.0, 2.2, 0.0], [3.3, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(cache_affine[:3, 3], affine[:3, 3])


def test_sheared_geometry_is_rejected():
    affine = np.eye(4, dtype=np.float64)
    affine[0, 1] = 0.1
    with pytest.raises(ValueError, match="sheared"):
        native_geometry_zyx(affine, (2, 3, 4))


def test_candidate_wrapper_returns_anisotropic_spacing_and_deterministic_slice_keys(tmp_path):
    fixture = synthetic_candidate_a_fixture(tmp_path / "fixture")
    data = CandidateAData(fixture["train"], fixture["train_sha256"], synthetic=True)
    keys1 = data.labeled_slice_keys()
    keys2 = data.labeled_slice_keys()
    assert keys1 == keys2
    x1, y1, spacing1, metadata1 = data.load_labeled_slice(keys1[0])
    x2, y2, spacing2, metadata2 = data.load_labeled_slice(keys2[0])
    np.testing.assert_array_equal(x1.numpy(), x2.numpy())
    np.testing.assert_array_equal(y1.numpy(), y2.numpy())
    assert spacing1 == (0.7, 1.3) and spacing2 == spacing1
    assert metadata1 == metadata2
    assert tuple(x1.shape) == (1, 3, 12, 13)
    assert tuple(y1.shape) == (1, 12, 13)


def test_old_manifest_stays_compatible_but_candidate_requires_geometry(tmp_path):
    fixture = synthetic_candidate_a_fixture(tmp_path / "fixture")
    manifest = json.loads(fixture["train"].read_text())
    manifest.pop("geometry")
    manifest.pop("class_mapping")
    old = tmp_path / "fixture" / "old.json"
    write_json(old, manifest)
    base = TrainingData(old, digest(old), synthetic=True)
    assert base.geometry_for(manifest["labeled"][0]["id"]) is None
    with pytest.raises(ValueError, match="requires validated native geometry"):
        CandidateAData(old, digest(old), synthetic=True)


def test_u_mask_and_class_mapping_contract_remain_strict(tmp_path):
    fixture = synthetic_candidate_a_fixture(tmp_path / "fixture")
    manifest = json.loads(fixture["train"].read_text())
    manifest["unlabeled"][0]["mask"] = "forbidden.npy"
    bad_u = tmp_path / "bad_u.json"
    write_json(bad_u, manifest)
    with pytest.raises(ValueError, match="image only"):
        TrainingData(bad_u, digest(bad_u), synthetic=True)

    manifest = json.loads(fixture["train"].read_text())
    manifest["class_mapping"]["1"] = "wrong"
    bad_mapping = tmp_path / "bad_mapping.json"
    write_json(bad_mapping, manifest)
    with pytest.raises(ValueError, match="class_mapping"):
        TrainingData(bad_mapping, digest(bad_mapping), synthetic=True)


def test_mask_frame_keys_are_canonical_and_unique(tmp_path):
    fixture = synthetic_candidate_a_fixture(tmp_path / "fixture")
    manifest = json.loads(fixture["train"].read_text())
    record = manifest["labeled"][0]
    first = record["masks"].pop("0")
    record["masks"]["01"] = first
    malformed = fixture["train"].parent / "malformed_frames.json"
    write_json(malformed, manifest)
    with pytest.raises(ValueError, match="canonical decimal"):
        TrainingData(malformed, digest(malformed), synthetic=True)


def test_dense_epoch_preflight_rejects_ignore_pixels(tmp_path):
    fixture = synthetic_candidate_a_fixture(tmp_path / "fixture")
    manifest = json.loads(fixture["train"].read_text())
    record = manifest["labeled"][0]
    mask_record = record["masks"]["0"]
    mask_path = fixture["train"].parent / mask_record["path"]
    mask = np.load(mask_path)
    mask[0, 0, 0] = 255
    np.save(mask_path, mask)
    mask_record["sha256"] = digest(mask_path)
    malformed = fixture["train"].parent / "ignore_pixel.json"
    write_json(malformed, manifest)
    data = CandidateAData(malformed, digest(malformed), synthetic=True)
    with pytest.raises(ValueError, match="dense TRAIN masks"):
        data.labeled_slice_keys()


def test_fixture_train_and_evaluation_ids_are_disjoint(tmp_path):
    fixture = synthetic_candidate_a_fixture(tmp_path / "fixture")
    train = json.loads(fixture["train"].read_text())
    evaluation = json.loads(fixture["evaluation"].read_text())
    train_ids = {r["id"] for r in train["labeled"] + train["unlabeled"]}
    eval_ids = {r["id"] for r in evaluation["patients"]}
    assert train_ids.isdisjoint(eval_ids)
    assert set(train["evaluation_ids"]) == eval_ids
    assert train["geometry"]["schema"] == "self_audit.geometry_map.v1"
    assert all(len(r["image"]) > 0 for r in evaluation["patients"])
