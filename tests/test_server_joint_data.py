from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np

from self_audit.compact_data import CompactTrainingData, synthetic_compact_fixture
from self_audit.data import TrainingData, write_json
from self_audit.model import digest
from scripts.prepare_server_mnm import (
    EVALUATION_MAPS_SCHEMA,
    MNM_SOURCE_SCHEMA,
    prepare_joint_data,
)


class _FakeNifti:
    def __init__(self, data: np.ndarray):
        self.dataobj = data
        self.affine = np.eye(4, dtype=np.float64)


class _FakeNibabel:
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = arrays
        self.calls: list[str] = []

    def load(self, path: str) -> _FakeNifti:
        key = str(Path(path).resolve())
        self.calls.append(key)
        return _FakeNifti(self.arrays[key])


def _save_array(path: Path, array: np.ndarray) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return str(path), digest(path)


def _write_fake_nifti(root: Path, relative: str, array: np.ndarray, fake_arrays: dict[str, np.ndarray]) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("fake-nifti:" + relative).encode("utf-8"))
    fake_arrays[str(path.resolve())] = array
    return {"path": relative, "sha256": digest(path)}


def _make_eval_map(folder: Path, domain: str, patient_id: str, seed: int) -> tuple[Path, str]:
    rng = np.random.default_rng(seed)
    image = folder / f"{domain}_{patient_id}.npy"
    _save_array(image, rng.random((2, 2, 6, 6), dtype=np.float32))
    mask = folder / f"{domain}_{patient_id}_mask.npy"
    _save_array(mask, rng.integers(0, 4, size=(2, 6, 6), dtype=np.uint8))
    manifest = folder / f"{domain}_{patient_id}_manifest.json"
    write_json(manifest, {
        "schema": "self_audit.native_manifest.v1",
        "role": "development",
        "synthetic": True,
        "patients": [{
            "id": patient_id,
            "domain": domain,
            "image": image.name,
            "image_sha256": digest(image),
            "axes": ["T", "Z", "Y", "X"],
            "affine": np.eye(4).tolist(),
            "masks": {"0": {"path": mask.name, "sha256": digest(mask)}},
        }],
    })
    return manifest, digest(manifest)


def _build_fixture(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(721)
    acdc_dir = tmp_path / "acdc"
    acdc_dir.mkdir()
    acdc_labeled = []
    acdc_unlabeled = []
    for index in range(1, 6):
        pid = f"patient{index:03d}"
        image = acdc_dir / f"{pid}.npy"
        _save_array(image, rng.random((2, 2, 6, 6), dtype=np.float32))
        mask = acdc_dir / f"{pid}_mask.npy"
        _save_array(mask, rng.integers(0, 4, size=(2, 6, 6), dtype=np.uint8))
        acdc_labeled.append({"id": pid, "image": image.name, "image_sha256": digest(image),
                             "masks": {"0": {"path": mask.name, "sha256": digest(mask)}}})
    # Known ACDC quarantine member has deliberately absent payloads. It must
    # be filtered from metadata before any source image or mask lookup.
    acdc_labeled.append({"id": "patient055", "image": "never_open_acdc.npy", "image_sha256": "a" * 64,
                         "masks": {"0": {"path": "never_open_acdc_mask.npy", "sha256": "b" * 64}}})
    for index in (6, 7):
        pid = f"patient{index:03d}"
        image = acdc_dir / f"{pid}.npy"
        _save_array(image, rng.random((2, 2, 6, 6), dtype=np.float32))
        acdc_unlabeled.append({"id": pid, "image": image.name, "image_sha256": digest(image)})
    acdc_manifest = acdc_dir / "train.json"
    write_json(acdc_manifest, {
        "schema": "self_audit.data.v2", "role": "training", "synthetic": True,
        "label_budget": len(acdc_labeled), "labeled": acdc_labeled,
        "unlabeled": acdc_unlabeled, "evaluation_ids": ["patient100"],
    })

    mnm_root = tmp_path / "mnm_raw"
    mnm_root.mkdir()
    split_path = mnm_root / "locked_split.json"
    mnm_clean_ids = [f"M{i:02d}" for i in range(5)]
    mnm_qid = "A7G0P5"
    mnm_uids = ["U00", "U01"]
    split_ids = mnm_clean_ids + [mnm_qid] + mnm_uids
    write_json(split_path, {"dataset": "M&Ms", "train": split_ids, "val": ["MV00"], "test": []})
    source_rows = []
    fake_arrays: dict[str, np.ndarray] = {}
    for index, pid in enumerate(mnm_clean_ids):
        image = _write_fake_nifti(mnm_root, f"Training-001/Training/Labeled/{pid}/{pid}_sa.nii.gz",
                                  rng.normal(index, 1, size=(6, 6, 2, 2)).astype(np.float32), fake_arrays)
        raw_mask = np.zeros((6, 6, 2), dtype=np.float32)
        raw_mask[0, 0, 0] = 1
        raw_mask[1, 1, 1] = 2
        raw_mask[2, 2, 0] = 3
        raw_mask[3, 3, 1] = 255
        mask = _write_fake_nifti(mnm_root, f"Training-001/Training/Labeled/{pid}/{pid}_sa_gt.nii.gz",
                                 raw_mask, fake_arrays)
        source_rows.append({"id": pid, "image": image["path"], "image_sha256": image["sha256"],
                            "masks": {"0": mask}})
    # Quarantined source files are intentionally absent; metadata alone is
    # enough to exclude them before any image or mask I/O.
    source_rows.append({"id": mnm_qid, "image": "Training-001/Training/Labeled/A7G0P5/A7G0P5_sa.nii.gz",
                        "image_sha256": "c" * 64,
                        "masks": {"0": {"path": "Training-001/Training/Labeled/A7G0P5/A7G0P5_sa_gt.nii.gz",
                                         "sha256": "d" * 64}}})
    source_unlabeled = []
    for pid in mnm_uids:
        image = _write_fake_nifti(mnm_root, f"Training-001/Training/Unlabeled/{pid}/{pid}_sa.nii.gz",
                                  rng.normal(5, 1, size=(6, 6, 2, 2)).astype(np.float32), fake_arrays)
        source_unlabeled.append({"id": pid, "image": image["path"], "image_sha256": image["sha256"]})
    # A decoy mask on disk proves preparation does not discover U masks.
    u_decoy = mnm_root / "Training-001/Training/Unlabeled/U00/U00_sa_gt.nii.gz"
    u_decoy.parent.mkdir(parents=True, exist_ok=True)
    u_decoy.write_bytes(b"must-not-be-opened")
    source_manifest = mnm_root / "raw_source.json"
    write_json(source_manifest, {
        "schema": MNM_SOURCE_SCHEMA,
        "dataset": "M&Ms",
        "synthetic": True,
        "source_split": {"path": split_path.name, "sha256": digest(split_path)},
        "train_labeled_candidates": source_rows,
        "train_unlabeled": source_unlabeled,
    })

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    acdc_eval_manifest, acdc_eval_sha = _make_eval_map(eval_dir, "acdc", "patient100", 83)
    mnm_eval_manifest, mnm_eval_sha = _make_eval_map(eval_dir, "mnm", "MV00", 84)
    eval_maps = eval_dir / "maps.json"
    write_json(eval_maps, {
        "schema": EVALUATION_MAPS_SCHEMA,
        "maps": {
            "acdc_val": {"domain": "acdc", "manifest": acdc_eval_manifest.name, "sha256": acdc_eval_sha},
            "mnm_val": {"domain": "mnm", "manifest": mnm_eval_manifest.name, "sha256": mnm_eval_sha},
        },
    })

    fake_nib = _FakeNibabel(fake_arrays)
    monkeypatch.setitem(sys.modules, "nibabel", fake_nib)
    return {
        "acdc_manifest": acdc_manifest,
        "mnm_root": mnm_root,
        "mnm_source_manifest": source_manifest,
        "evaluation_maps": eval_maps,
        "fake_nib": fake_nib,
        "u_decoy": u_decoy,
        "mnm_clean_ids": mnm_clean_ids,
        "acdc_dir": acdc_dir,
    }


def _prepare(fixture, out: Path):
    return prepare_joint_data(
        acdc_manifest_path=fixture["acdc_manifest"],
        acdc_manifest_sha256=digest(fixture["acdc_manifest"]),
        mnm_root=fixture["mnm_root"],
        mnm_source_manifest_path=fixture["mnm_source_manifest"],
        mnm_source_manifest_sha256=digest(fixture["mnm_source_manifest"]),
        evaluation_maps_path=fixture["evaluation_maps"],
        evaluation_maps_sha256=digest(fixture["evaluation_maps"]),
        out_dir=out,
        synthetic=True,
    )


def test_server_prepare_pins_joint_l8_mapping_maps_and_mask_access(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    result = _prepare(fixture, tmp_path / "joint_prepared")

    train = json.loads(Path(result["train_manifest"]).read_text())
    prep = json.loads(Path(result["preparation"]).read_text())
    maps = json.loads(Path(result["evaluation_maps"]).read_text())
    assert train["label_budget"] == 8
    assert train["domain_budgets"] == {"acdc": 4, "mnm": 4}
    assert train["label_seed"] == 42
    assert len(prep["selected_labeled_ids"]["acdc"]) == 4
    assert len(prep["selected_labeled_ids"]["mnm"]) == 4
    assert all(pid.startswith("mnm:") for pid in prep["selected_labeled_ids"]["mnm"])
    assert "acdc:patient055" in prep["quarantined_ids_excluded"]["acdc"]
    assert "mnm:A7G0P5" in prep["quarantined_ids_excluded"]["mnm"]
    assert set(maps["maps"]) == {"acdc_val", "mnm_val"}
    assert prep["mask_mapping"]["mapping_applied_once"] == {"0": 0, "1": 3, "2": 2, "3": 1, "255": 255}
    assert prep["mask_access"]["unlabeled_mask_reads"] == 0
    assert prep["mask_access"]["unlabeled_output_records_have_mask_paths"] is False
    assert all(set(record) == {"id", "image", "image_sha256"} for record in train["unlabeled"])
    assert not any(path.endswith("U00_sa_gt.nii.gz") for path in fixture["fake_nib"].calls)

    opened_masks = {path for path in fixture["fake_nib"].calls if path.endswith("_sa_gt.nii.gz")}
    selected_raw_ids = {pid.split(":", 1)[1] for pid in prep["selected_labeled_ids"]["mnm"]}
    assert len(opened_masks) == 4
    assert {Path(path).parent.name for path in opened_masks} == selected_raw_ids

    mask_path = next(iter(Path(result["train_manifest"]).parent.glob("masks/*.npy")))
    canonical = np.load(mask_path)
    assert canonical[0, 0, 0] == 3
    assert canonical[1, 1, 1] == 2
    assert canonical[0, 2, 2] == 1
    assert canonical[1, 3, 3] == 255

    data = CompactTrainingData(result["train_manifest"], result["train_manifest_sha256"], synthetic=True)
    rng = np.random.default_rng(42)
    observed_domains = []
    for step in range(40):
        li, ai, ui = data.sample_indices(step, rng)
        supervised_domain = data.domain_of(data.labeled[li]["id"])
        audit_domain = data.domain_of(data.labeled[ai]["id"])
        unlabeled_domain = data.domain_of(data.unlabeled[ui]["id"])
        assert supervised_domain == audit_domain == unlabeled_domain
        assert data.labeled[li]["id"] != data.labeled[ai]["id"]
        assert supervised_domain == ("acdc" if step % 2 == 0 else "mnm")
        observed_domains.append(supervised_domain)
    assert observed_domains.count("acdc") == observed_domains.count("mnm") == 20

    # The sampler accepts the checkpointed NumPy generator state. Replaying
    # from a segment boundary returns the exact same L/A/U patient sequence.
    prefix_rng = np.random.default_rng(904)
    for step in range(12):
        data.sample_indices(step, prefix_rng)
    saved_state = copy.deepcopy(prefix_rng.bit_generator.state)
    expected = [data.sample_indices(step, prefix_rng) for step in range(12, 24)]
    resumed_rng = np.random.default_rng(1)
    resumed_rng.bit_generator.state = saved_state
    replay = [data.sample_indices(step, resumed_rng) for step in range(12, 24)]
    assert replay == expected


def test_acdc_only_compact_sampler_matches_training_data(tmp_path):
    manifest, sha = synthetic_compact_fixture(tmp_path / "acdc_only", seed=42, n_labeled=4, n_unlabeled=3)
    compact = CompactTrainingData(manifest, sha, synthetic=True)
    ordinary = TrainingData(manifest, sha, synthetic=True, allow_k1=True)
    compact_rng = np.random.default_rng(83)
    ordinary_rng = np.random.default_rng(83)
    assert [compact.sample_indices(step, compact_rng) for step in range(12)] == [
        ordinary.sample_indices(step, ordinary_rng) for step in range(12)
    ]


def test_mnm_split_mismatch_blocks_before_raw_nifti_reads(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    split_path = fixture["mnm_root"] / "locked_split.json"
    split = json.loads(split_path.read_text())
    split["train"] = split["train"][:-1]
    write_json(split_path, split)
    source = json.loads(fixture["mnm_source_manifest"].read_text())
    source["source_split"]["sha256"] = digest(split_path)
    write_json(fixture["mnm_source_manifest"], source)

    with np.testing.assert_raises_regex(ValueError, "do not exactly cover the pinned TRAIN split"):
        _prepare(fixture, tmp_path / "blocked")
    assert fixture["fake_nib"].calls == []


def test_unlabeled_source_mask_reference_is_rejected_before_raw_nifti_reads(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    source = json.loads(fixture["mnm_source_manifest"].read_text())
    source["train_unlabeled"][0]["masks"] = {"0": {"path": "never_open.nii.gz", "sha256": "e" * 64}}
    write_json(fixture["mnm_source_manifest"], source)

    with np.testing.assert_raises_regex(ValueError, "unlabeled record must contain exactly"):
        _prepare(fixture, tmp_path / "blocked_u_mask")
    assert fixture["fake_nib"].calls == []
