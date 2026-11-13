import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from self_audit.candidate_fuse.data import FuseData, FuseSampler, write_synthetic_manifests
from scripts.fuse_build_manifest import _split
from scripts.fuse_prepare_sources import prepare_sources


def _data(tmp_path):
    refs = write_synthetic_manifests(tmp_path / "synthetic")
    return refs, FuseData(refs["train"]["path"], refs["train"]["sha256"], role="train", synthetic=True)


def test_synthetic_manifest_hash_roles_and_exact_ed_es_exclusion(tmp_path):
    refs, data = _data(tmp_path)
    assert set(refs) == {"train", "development", "test"}
    assert all(len(ref["sha256"]) == 64 for ref in refs.values())
    assert data.patients == ("train001", "train002")
    assert len(data.labeled_keys()) == 4
    assert all(frame in {0, 2} for _, frame in data.labeled_keys())
    assert all(frame in {1, 3} for _, frame in data.unlabeled_keys())
    with pytest.raises(ValueError, match="role"):
        FuseData(refs["test"]["path"], refs["test"]["sha256"], role="development", synthetic=True)
    with pytest.raises(ValueError, match="manifest SHA"):
        FuseData(refs["train"]["path"], "0" * 64, role="train", synthetic=True)


def test_u_load_has_no_target_and_ed_es_cannot_be_loaded_as_u(tmp_path):
    _, data = _data(tmp_path)
    frame = data.load(("train001", 1), labeled=False)
    assert frame.image.shape == (1, 1, 3, 8, 9)
    assert frame.target is None
    with pytest.raises(ValueError, match="excludes"):
        data.load(("train001", 0), labeled=False)
    with pytest.raises(ValueError, match="ED/ES"):
        data.load(("train001", 1), labeled=True)


def test_sampler_full_permutations_and_resume(tmp_path):
    _, data = _data(tmp_path)
    sampler = FuseSampler(data, seed=42)
    epoch = [sampler.next_labeled() for _ in range(sampler.teacher_epoch_size)]
    assert len(epoch) == len(set(epoch)) == sampler.teacher_epoch_size
    pairs = [sampler.next_pair() for _ in range(sampler.joint_epoch_size)]
    assert len({u[0] for _, u in pairs}) == sampler.joint_epoch_size
    assert all(u[1] in {1, 3} for _, u in pairs)
    state = sampler.state_dict()
    expected = [sampler.next_pair() for _ in range(4)]
    restored = FuseSampler(data, seed=42)
    restored.load_state_dict(state)
    assert [restored.next_pair() for _ in range(4)] == expected


def test_sampler_roundtrip_before_first_joint_pair(tmp_path):
    _, data = _data(tmp_path)
    sampler = FuseSampler(data, seed=42)
    for _ in range(sampler.teacher_epoch_size):
        sampler.next_labeled()
    state = sampler.state_dict()
    assert state["l_stream"] == []
    assert state["l_cursor"] == 0

    restored = FuseSampler(data, seed=42)
    restored.load_state_dict(state)
    expected_pair = sampler.next_pair()
    actual_pair = restored.next_pair()
    assert actual_pair == expected_pair
    assert restored.py_rng.getstate() == sampler.py_rng.getstate()
    assert restored.epoch_rng.getstate() == sampler.epoch_rng.getstate()


def test_manifest_payload_hash_mismatch_fails(tmp_path):
    refs, _ = _data(tmp_path)
    root = tmp_path / "synthetic"
    manifest_path = root / "train.json"
    manifest = json.loads(manifest_path.read_text())
    image = root / "train001.npy"
    with image.open("ab") as stream:
        stream.write(b"changed")
    # Pinning the changed manifest itself must not hide a payload hash mismatch.
    with pytest.raises(ValueError, match="image SHA256 mismatch"):
        FuseData(manifest_path, __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest(),
                 role="train", synthetic=True)


def test_split_overlap_rejected(tmp_path):
    refs, _ = _data(tmp_path)
    path = tmp_path / "synthetic" / "train.json"
    manifest = json.loads(path.read_text())
    manifest["split"]["development"].append("train001")
    path.write_text(json.dumps(manifest))
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="overlap"):
        FuseData(path, sha, role="train", synthetic=True)


def test_sampler_rejects_tampered_duplicate_resume_permutation(tmp_path):
    _, data = _data(tmp_path)
    sampler = FuseSampler(data, seed=42)
    state = sampler.state_dict()
    state["teacher_stream"][0] = state["teacher_stream"][1]
    with pytest.raises(ValueError, match="permutation"):
        FuseSampler(data, seed=42).load_state_dict(state)


def test_sampler_rejects_invalid_empty_or_started_joint_stream(tmp_path):
    _, data = _data(tmp_path)
    sampler = FuseSampler(data, seed=42)
    initial = sampler.state_dict()
    initial["l_cursor"] = 1
    with pytest.raises(ValueError, match="empty initial"):
        FuseSampler(data, seed=42).load_state_dict(initial)

    sampler.next_pair()
    started = sampler.state_dict()
    tampered_empty = dict(started, l_stream=[], l_cursor=0)
    with pytest.raises(ValueError, match="empty initial"):
        FuseSampler(data, seed=42).load_state_dict(tampered_empty)
    started["l_stream"][0] = started["l_stream"][1]
    with pytest.raises(ValueError, match="permutation"):
        FuseSampler(data, seed=42).load_state_dict(started)


def test_builder_split_accepts_locked_80_18_50_and_rejects_legacy_80_20(tmp_path):
    path = tmp_path / "split.json"
    split = {"train": [f"tr{i:03}" for i in range(80)],
             "val": [f"va{i:03}" for i in range(18)],
             "test": [f"te{i:03}" for i in range(50)]}
    path.write_text(json.dumps(split))
    parsed = _split(path)
    assert {role: len(ids) for role, ids in parsed.items()} == {
        "train": 80, "development": 18, "test": 50,
    }
    split["val"] = [f"va{i:03}" for i in range(20)]
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="TRAIN80/VAL18/TEST50"):
        _split(path)


def test_prepare_sources_tiny_raw_cine_without_test_masks(tmp_path):
    import nibabel as nib

    root = tmp_path / "raw"
    for role, patient in (("training", "patient001"), ("training", "patient002"),
                          ("training", "patient003"), ("testing", "patient101")):
        (root / role / patient).mkdir(parents=True)

    geometry = {pid: {"shape_zyx": [3, 5, 4]} for pid in ("patient001", "patient002")}
    train = {
        "schema": "self_audit.data.v2", "role": "training", "synthetic": False, "dataset": "acdc",
        "labeled": [{"id": "patient001"}], "unlabeled": [{"id": "patient002"}],
        "geometry": {"by_patient": geometry},
    }
    development = {
        "schema": "self_audit.data.v2", "role": "development", "synthetic": False, "dataset": "acdc",
        "patients": [{"id": "patient003"}],
    }
    train_path, development_path = tmp_path / "train.json", tmp_path / "development.json"
    train_path.write_text(json.dumps(train)); development_path.write_text(json.dumps(development))

    raw = np.arange(4 * 5 * 3 * 3, dtype=np.int16).reshape(4, 5, 3, 3)
    image = nib.Nifti1Image(raw, np.diag([1.5, 2.0, 4.0, 1.0]))
    image.header.set_zooms((1.5, 2.0, 4.0, 1.0))
    image.header.set_xyzt_units(xyz=None, t=None)
    image.set_sform(np.diag([1.5, 2.0, 4.0, 1.0]), code=1)
    image.set_qform(np.diag([1.5, 2.0, 4.0, 1.0]), code=1)
    cine_path = root / "testing" / "patient101" / "patient101_4d.nii"
    nib.save(image, cine_path)

    refs = prepare_sources(
        train_path, development_path, root, tmp_path / "prepared",
        _expected_counts=(2, 1, 1), _expected_test_ids=["patient101"],
    )
    split = json.loads(Path(refs["split"]["path"]).read_text())
    manifest = json.loads(Path(refs["test"]["path"]).read_text())
    record = manifest["patients"][0]
    output = np.load(record["image"], allow_pickle=False)
    lo, hi = np.percentile(raw, [0.5, 99.5])
    expected = np.clip((raw - lo) / max(float(hi - lo), 1e-6), 0, 1).transpose(3, 2, 1, 0).astype(np.float32)
    assert split["train"] == ["patient001", "patient002"]
    assert split["development"] == ["patient003"] and split["test"] == ["patient101"]
    assert np.array_equal(output, expected)
    assert record["image_sha256"] == hashlib.sha256(Path(record["image"]).read_bytes()).hexdigest()
    assert manifest["geometry"]["by_patient"]["patient101"]["spacing_zyx_mm"] == [4.0, 2.0, 1.5]
    assert json.loads(Path(refs["receipt"]).read_text())["test_mask_reads"] == 0
    assert not list((root / "testing" / "patient101").glob("*_gt.nii*"))
