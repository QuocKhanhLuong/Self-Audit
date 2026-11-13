import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from self_audit.candidate_fuse.config import load_config, normalize_config
from self_audit.candidate_fuse.engine import _validate_semi_track_a, run_training
from self_audit.candidate_fuse.data import write_synthetic_manifests

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", [50, 999])
def test_v2_rejects_supplied_nonlocked_teacher_epoch(value):
    cfg = json.loads((ROOT / "configs/candidate_fuse/semi_k8.json").read_text())
    cfg["training"]["teacher_epochs"] = value
    with pytest.raises(ValueError):
        normalize_config(cfg)


def test_semi_k8_config_locks_v2_and_budget():
    cfg = load_config(ROOT / "configs/candidate_fuse/semi_k8.json")
    assert cfg["experiment"] == {"track": "A", "dataset": "ACDC"}
    assert cfg["training"]["teacher_epochs"] == 500
    assert cfg["training"]["teacher_batch_size"] == 2
    assert cfg["training"]["joint_epochs"] == 100
    with pytest.raises(ValueError):
        cfg["experiment"] = {"track": "B", "dataset": "MNMS"}
        normalize_config(cfg)


def test_v1_config_file_bytes_and_hash_are_unchanged():
    raw = (ROOT / "configs/candidate_fuse/v1.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "d90c276611e82cb42036e77c8482143c897f6f0ee4cf34423bff4c81b9f19ee4"
    assert load_config(ROOT / "configs/candidate_fuse/v1.json")["schema"].endswith("v1")


def _semi_fixture(root: Path, *, val_overlap=False):
    from tests.test_fuse_semi_data import _semi_fixture as build_source
    from scripts.fuse_build_semi_manifest import build as build_semi
    from self_audit.candidate_fuse.data import FuseData

    root.mkdir(parents=True)
    source_path, _, locked_path, locked_sha, source = build_source(root, patients=9, labeled_count=8)
    train_ids = source["split"]["train"]
    l_ids = set(r["id"] for r in json.loads(locked_path.read_text())["labeled"])
    non_l = set(train_ids) - l_ids
    assert len(train_ids) >= 9 and len(l_ids) == 8 and non_l
    result = build_semi(source_path, locked_path, root / "semi-output",
                        expected_locked_sha256=locked_sha, synthetic=True)
    manifest = json.loads(Path(result["manifest"]).read_text())
    train_path = Path(result["manifest"])
    train_data = FuseData(str(train_path), result["sha256"], role="train", synthetic=True)
    assert len(train_data.patients) >= 9 and len(train_data.labeled_keys()) == 16
    assert any({0, 2}.issubset({f for p, f in train_data.unlabeled_keys() if p == pid}) for pid in non_l)

    # Reuse a deterministic synthetic development manifest, but update its
    # locked split identity to match this L8+U-only TRAIN source.
    val_root = root / "val-v1"
    refs = write_synthetic_manifests(val_root)
    val_path = Path(refs["development"]["path"])
    val_manifest = json.loads(val_path.read_text())
    split = manifest["split"]
    val_ids = [next(iter(non_l))] if val_overlap else ["dev001"]
    split["development"] = val_ids
    val_manifest["split"] = split
    if val_overlap:
        record = val_manifest["patients"][0]
        record["id"] = val_ids[0]
        val_manifest["geometry"]["by_patient"] = {val_ids[0]: val_manifest["geometry"]["by_patient"].pop("dev001")}
    val_path.write_text(json.dumps(val_manifest, sort_keys=True))
    cfg = load_config(ROOT / "configs/candidate_fuse/semi_k8.json")
    cfg["runtime"]["synthetic"] = True; cfg["runtime"]["amp_dtype"] = "float32"
    cfg["model"]["student"]["channels"] = [4, 8, 12, 16, 32]
    cfg["training"]["teacher_epochs"] = cfg["training"]["joint_epochs"] = 1
    cfg["data"].update(train_manifest=str(train_path), train_manifest_sha256=result["sha256"],
        val_manifest=str(val_path), val_manifest_sha256=hashlib.sha256(val_path.read_bytes()).hexdigest())
    return cfg, l_ids, non_l


def test_track_a_gate_rejects_train_validation_patient_overlap(tmp_path):
    from self_audit.candidate_fuse.data import FuseData
    cfg, _, _ = _semi_fixture(tmp_path / "overlap", val_overlap=True)
    with pytest.raises(ValueError, match="overlap"):
        run_training(cfg, tmp_path / "run-overlap", stage="T", max_updates=1)


@pytest.mark.parametrize("stage", ["T", "all"])
def test_v2_synthetic_run_never_loads_train_masks_outside_l8(tmp_path, monkeypatch, stage):
    from self_audit.candidate_fuse import data as data_module
    cfg, l_ids, u_only = _semi_fixture(tmp_path / f"semi-{stage}")
    original = data_module.FuseData.load
    seen_train_labeled = []
    def checked_load(self, key, *, labeled):
        if self.role == "train" and labeled:
            seen_train_labeled.append(key[0])
            assert key[0] in l_ids, f"non-L8 training mask access: {key[0]}"
        return original(self, key, labeled=labeled)
    monkeypatch.setattr(data_module.FuseData, "load", checked_load)
    result = run_training(cfg, tmp_path / f"run-{stage}", stage=stage, max_updates=1 if stage == "T" else None)
    assert result["status"] in {"PAUSED", "COMPLETED"}
    assert len(u_only) >= 1
    assert seen_train_labeled and set(seen_train_labeled).issubset(l_ids)
