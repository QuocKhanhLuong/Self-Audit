import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from self_audit.candidate_fuse.data import FuseData, FuseSampler, write_synthetic_manifests
from scripts.fuse_build_semi_manifest import build as build_semi_manifest


def _semi_fixture(tmp_path, *, patients=80, labeled_count=8):
    """Construct synthetic source metadata using the reviewed K8 record shape."""
    root = tmp_path / "semi-source"
    root.mkdir()
    rng = np.random.default_rng(420)
    class_mapping = {"0": "BG", "1": "RV", "2": "MYO", "3": "LV"}
    train_ids = [f"train{i:03d}" for i in range(patients)]
    geometry = {"shape": [3, 8, 9], "spacing_zyx": [2.5, .7, 1.3],
                "affine_zyx": np.diag([2.5, .7, 1.3, 1.]).tolist(), "context": "full_volume",
                "provenance": {"synthetic": True}}
    labeled, unlabeled, geometries, original_mask_inventory = [], [], {}, {}
    for pid in train_ids:
        cine = rng.random((4, 3, 8, 9), dtype=np.float32)
        ip = root / f"{pid}.npy"
        np.save(ip, cine)
        image_sha = hashlib.sha256(ip.read_bytes()).hexdigest()
        masks = {}
        for frame in (0, 2):
            mask = np.zeros((3, 8, 9), dtype=np.uint8)
            mask[:, 2:6, 2:7] = 2
            mask[:, 3:5, 3:6] = 3
            mp = root / f"{pid}_t{frame:03d}_mask.npy"
            np.save(mp, mask)
            source_mask_path = str(root / "raw" / f"{pid}_frame{frame + 1:02d}_gt.nii")
            source_mask_sha = f"{frame + 1:064x}"
            masks[str(frame)] = {
                "path": str(mp), "sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
                "source_path": source_mask_path, "source_sha256": source_mask_sha,
                "transpose": [2, 1, 0],
            }
            original_mask_inventory[source_mask_path] = source_mask_sha
        labeled.append({"id": pid, "image": str(ip), "image_sha256": image_sha,
                        "phases": {"ED": 0, "ES": 2}, "masks": masks})
        unlabeled.append({"id": pid, "image": str(ip), "image_sha256": image_sha})
        geometries[pid] = geometry
    split = {"train": train_ids, "development": ["dev001"], "test": ["test001"]}
    train = {
        "schema": "self_audit.candidate_fuse.data.v1", "role": "train", "synthetic": True,
        "dataset": "synthetic", "class_mapping": class_mapping, "split": split,
        "source_original_mask_sha256": original_mask_inventory,
        "geometry": {"by_patient": geometries}, "patients": train_ids,
        "labeled": labeled, "unlabeled": unlabeled,
    }
    source_path = root / "full-train.json"
    source_bytes = json.dumps(train, sort_keys=True).encode()
    source_path.write_bytes(source_bytes)
    chosen = set(train_ids[:labeled_count])
    locked = {
        "schema": "self_audit.data.v2", "role": "training", "synthetic": True,
        "dataset": "synthetic", "class_mapping": class_mapping, "label_seed": 42,
        "label_budget": labeled_count, "evaluation_ids": ["dev001", "test001"],
        "geometry": {},
        "labeled": [
            {"id": r["id"], "image": r["image"], "image_sha256": r["image_sha256"],
             "masks": {k: {"path": v["path"], "sha256": v["sha256"]}
                       for k, v in r["masks"].items()}}
            for r in labeled if r["id"] in chosen
        ],
        "unlabeled": [r for r in unlabeled if r["id"] not in chosen],
    }
    locked_path = root / "train.q3d-pixdim.json"
    locked_path.write_text(json.dumps(locked, sort_keys=True))
    expected_locked_sha = hashlib.sha256(locked_path.read_bytes()).hexdigest()
    return source_path, source_bytes, locked_path, expected_locked_sha, train


def _build(tmp_path, *, patients=80, labeled_count=8):
    source_path, source_bytes, locked_path, locked_sha, source = _semi_fixture(
        tmp_path, patients=patients, labeled_count=labeled_count)
    result = build_semi_manifest(source_path, locked_path, tmp_path / "semi-output",
                                 expected_locked_sha256=locked_sha, synthetic=True)
    return result, source_path, source_bytes, locked_path, source


def test_builder_strips_non_l_masks_and_loader_sampler_preserve_v2_contract(tmp_path, monkeypatch):
    source_path, source_bytes, locked_path, locked_sha, source = _semi_fixture(tmp_path)
    non_l = set(source["patients"]) - {r["id"] for r in json.loads(locked_path.read_text())["labeled"]}
    forbidden = {r["masks"][frame]["path"] for r in source["labeled"] if r["id"] in non_l
                 for frame in r["masks"]}
    original_load = np.load

    def guarded_load(file, *args, **kwargs):
        if str(file) in forbidden:
            raise AssertionError(f"non-L TRAIN mask array opened: {file}")
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    result = build_semi_manifest(source_path, locked_path, tmp_path / "semi-output",
                                 expected_locked_sha256=locked_sha, synthetic=True)
    manifest = json.loads(Path(result["manifest"]).read_text())
    non_l = set(source["patients"]) - set(manifest["labeled_contract"]["patient_ids"])
    assert manifest["schema"] == "self_audit.candidate_fuse.data.v2"
    assert len(manifest["labeled"]) == 8
    assert len(manifest["unlabeled"]) == 80
    assert all(set(record) == {"id", "image", "image_sha256"} for record in manifest["unlabeled"])
    assert result["locked_l_ids"] == sorted(record["id"] for record in manifest["labeled"])
    assert result["counts"] == {
        "train_patient_count": 80, "labeled_patient_count": 8,
        "labeled_frame_count": 16, "train_frame_count": 320,
        "unlabeled_frame_count": 304, "unlabeled_patient_count": 80,
    }
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["mask_array_reads"] == receipt["nifti_reads"] == 0
    assert receipt["non_l_mask_references_in_output"] == 0
    assert receipt["original_mask_inventory_retained"] == 16
    assert len(manifest["source_original_mask_sha256"]) == 16
    assert source_path.read_bytes() == source_bytes
    assert locked_path.exists()

    data = FuseData(result["manifest"], result["sha256"], role="train", synthetic=True)
    assert len(data.labeled_keys()) == 16
    keys = data.unlabeled_keys()
    assert len(keys) == 304
    assert all(frame not in {0, 2} for pid, frame in keys if pid not in non_l)
    assert all({0, 2}.issubset({frame for p, frame in keys if p == pid}) for pid in non_l)
    assert data.load((sorted(non_l)[0], 0), labeled=False).target is None

    sampler = FuseSampler(data, seed=42)
    epoch = [sampler.next_pair() for _ in range(80)]
    assert len({u[0] for _, u in epoch}) == sampler.joint_epoch_size == 80


def test_v2_locked_id_tampering_fails_even_when_contract_hash_is_recomputed(tmp_path):
    result, *_ = _build(tmp_path)
    path = Path(result["manifest"])
    manifest = json.loads(path.read_text())
    ids = manifest["labeled_contract"]["patient_ids"]
    ids[0] = "train999"
    ids.sort()
    canonical = json.dumps(ids, separators=(",", ":"))
    manifest["labeled_contract"]["patient_ids_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(manifest, sort_keys=True))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="TRAIN L records differ from locked patient IDs"):
        FuseData(path, digest, role="train", synthetic=True)


def test_v1_manifest_hashes_and_loading_behavior_are_unchanged(tmp_path):
    refs = write_synthetic_manifests(tmp_path / "v1")
    path = Path(refs["train"]["path"])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    data = FuseData(path, refs["train"]["sha256"], role="train", synthetic=True)
    assert len(data.labeled_keys()) == 4
    assert len(data.unlabeled_keys()) == 4
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before == refs["train"]["sha256"]


def test_synthetic_target_domain_mask_isolation_without_real_split_claim(tmp_path, monkeypatch):
    # Software-only coverage; this does not fabricate M&Ms membership or geometry.
    result, _, _, _, source = _build(tmp_path, patients=3, labeled_count=1)
    manifest = json.loads(Path(result["manifest"]).read_text())
    non_l = set(source["patients"]) - set(manifest["labeled_contract"]["patient_ids"])
    assert manifest["dataset"] == "synthetic"
    forbidden = {r["masks"][frame]["path"] for r in source["labeled"] if r["id"] in non_l
                 for frame in r["masks"]}
    original_load = np.load

    def guarded_load(file, *args, **kwargs):
        if str(file) in forbidden:
            raise AssertionError(f"non-L target mask array opened: {file}")
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    data = FuseData(result["manifest"], result["sha256"], role="train", synthetic=True)
    assert len(data.unlabeled_keys()) == 10
    assert all(frame in {0, 1, 2, 3} for pid, frame in data.unlabeled_keys() if pid in non_l)


def test_builder_requires_exact_locked_k8_source_sha(tmp_path):
    source, _, locked, _, _ = _semi_fixture(tmp_path)
    with pytest.raises(ValueError, match="source manifest SHA256 mismatch"):
        build_semi_manifest(source, locked, tmp_path / "wrong-sha", expected_locked_sha256="0" * 64,
                            synthetic=True)


def test_builder_rejects_k8_l_records_that_disagree_with_full_source(tmp_path):
    source_path, _, locked_path, locked_sha, _ = _semi_fixture(tmp_path)
    source = json.loads(source_path.read_text())
    source["labeled"][0]["masks"]["0"]["sha256"] = "0" * 64
    source_path.write_text(json.dumps(source, sort_keys=True))
    with pytest.raises(ValueError, match="locked L mask hashes differ"):
        build_semi_manifest(source_path, locked_path, tmp_path / "inconsistent",
                            expected_locked_sha256=locked_sha, synthetic=True)
