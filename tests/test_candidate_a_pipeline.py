"""Actual CLI stages, export and held-out synthetic evaluation integration."""
import copy
import csv
import json
import shutil
from pathlib import Path
import pytest
import torch
from scripts.smoke_candidate_a import run_smoke
from self_audit.candidate_a.config import validate_config
from self_audit.candidate_a.engine import train
from self_audit.candidate_a.io import read_checkpoint
from self_audit.model import digest


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("candidate_a_parent") / "run"
    return out, run_smoke(out)


def test_cli_stages_export_and_frozen_evaluation(smoke):
    out, receipt = smoke
    assert receipt["status"] == "PASS_SYNTHETIC_ONLY"
    assert receipt["export_bitwise_parity"]
    assert receipt["export_payload_keys"] == ["student"]
    assert receipt["stages"]["teacher_pretrain"]["student_updates"] == 0
    assert receipt["stages"]["teacher_warmup"]["student_updates"] == 0
    assert receipt["stages"]["joint"]["teacher_snapshots"] > 0
    assert (out / "joint" / "metrics.csv").is_file()
    frozen = out / "evaluation" / "PREDICTIONS_FROZEN.json"
    assert frozen.is_file()
    assert receipt["patient_training"] == "NOT_RUN"


def test_epoch_plan_covers_each_annotated_frame_slice_and_logs_all_classes(smoke):
    out, _ = smoke
    epoch = json.loads((out / "joint" / "epochs.jsonl").read_text().splitlines()[0])
    # Fixture shape is T=1, Z=2 with two labeled patients: 1*2*2 updates.
    assert epoch["updates"] == 4
    with (out / "joint" / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    branches = {row["branch"] for row in rows}
    assert {"student", "teacher_region", "teacher_boundary"}.issubset(branches)
    assert {row["class"] for row in rows} == {"BG", "RV", "MYO", "LV"}
    for stage in ("teacher_pretrain", "teacher_warmup"):
        validation_rows = list(csv.DictReader((out / stage / "metrics.csv").open()))
        assert {row["branch"] for row in validation_rows if row["split"] == "validation"} == {
            "teacher_region", "teacher_boundary"
        }


def test_frozen_teacher_and_warmup_encoder(smoke, tmp_path, monkeypatch):
    out, _ = smoke
    pre, _ = read_checkpoint(out / "teacher_pretrain" / "final", digest(out / "teacher_pretrain" / "final" / "MANIFEST.json"))
    warm, _ = read_checkpoint(out / "teacher_warmup" / "final", digest(out / "teacher_warmup" / "final" / "MANIFEST.json"))
    encoder_keys = [k for k in pre["teacher"] if k.startswith("encoder.")]
    assert encoder_keys, "shared encoder must be explicit in teacher state"
    assert all(torch.equal(pre["teacher"][k], warm["teacher"][k]) for k in encoder_keys)
    assert any(not torch.equal(pre["teacher"][k], warm["teacher"][k]) for k in pre["teacher"] if k not in encoder_keys)
    c = json.loads((out / "joint.json").read_text())
    c.update(epochs=1, freeze_teacher=True, validation_every=0)
    c["teacher_init"] = {"kind": "candidate_a", "path": str(out / "teacher_warmup" / "final"),
        "sha256": digest(out / "teacher_warmup" / "final" / "MANIFEST.json")}
    from self_audit.candidate_a.teacher import DualBranchTeacher
    from self_audit.candidate_a.model import CandidateAStudent
    seen = {"teacher": [], "student": []}
    teacher_forward, student_encode = DualBranchTeacher.forward, CandidateAStudent.encode
    def observe_teacher(self, x):
        seen["teacher"].append(x.detach().clone())
        return teacher_forward(self, x)
    def observe_student(self, x):
        seen["student"].append(x.detach().clone())
        return student_encode(self, x)
    monkeypatch.setattr(DualBranchTeacher, "forward", observe_teacher)
    monkeypatch.setattr(CandidateAStudent, "encode", observe_student)
    report = train(c, out / "data" / "train.json", digest(out / "data" / "train.json"), tmp_path / "frozen", device="cpu", synthetic=True)
    state, _ = read_checkpoint(tmp_path / "frozen" / "final", report["final_manifest_sha256"])
    assert report["teacher_updates"] == 0 and report["student_updates"] > 0
    assert all(torch.equal(warm["teacher"][k], state["teacher"][k]) for k in warm["teacher"])
    assert len(seen["student"]) == report["student_updates"]
    assert len(seen["teacher"]) == report["teacher_snapshots"]
    assert all(torch.equal(a, b) for a, b in zip(seen["teacher"], seen["student"]))


def test_config_has_finite_total_budget_and_native_classes():
    root = Path(__file__).resolve().parents[1]
    c = json.loads((root / "configs/candidate_a/smoke.json").read_text())
    validate_config(c)
    for path, value in (("bit_budget", 1), ("bit_budget", None), ("baseline_ids", [0, 0])):
        bad = copy.deepcopy(c)
        bad["rollout"][path] = value
        with pytest.raises((ValueError, TypeError)):
            validate_config(bad)
    bad = copy.deepcopy(c)
    bad["class_mapping"] = ["BG", "LV", "MYO", "RV"]
    with pytest.raises(ValueError, match="class mapping"):
        validate_config(bad)


def test_teacher_stage_rejects_changed_training_manifest(smoke, tmp_path):
    out, _ = smoke
    checkpoint = tmp_path / "teacher"
    shutil.copytree(out / "teacher_warmup" / "final", checkpoint)
    mp = checkpoint / "MANIFEST.json"
    meta = json.loads(mp.read_text())
    meta["data_sha256"] = "0" * 64
    mp.write_text(json.dumps(meta))
    c = json.loads((out / "joint.json").read_text())
    c["validation_every"] = 0
    c["teacher_init"] = {"kind": "candidate_a", "path": str(checkpoint), "sha256": digest(mp)}
    with pytest.raises(ValueError, match="exact hashed TRAIN manifest"):
        train(c, out / "data" / "train.json", digest(out / "data" / "train.json"),
              tmp_path / "rejected", device="cpu", synthetic=True)


def test_checkpoint_rejects_changed_source_even_with_new_manifest_hash(smoke, tmp_path):
    out, _ = smoke
    checkpoint = tmp_path / "student"
    shutil.copytree(out / "student", checkpoint)
    mp = checkpoint / "MANIFEST.json"
    meta = json.loads(mp.read_text())
    meta["source_files"]["self_audit/candidate_a/model.py"] = "0" * 64
    mp.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="executed source differs"):
        read_checkpoint(checkpoint, digest(mp))
