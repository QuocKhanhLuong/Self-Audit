"""Fail-closed binding and domain-selection fixtures; no patient/GPU claims."""
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from self_audit import server_ready_arm as ready
from self_audit import server_runner as runner
from self_audit.server_evaluation import shared_selection_summary


def pin(path, value):
    runner.atomic_json(path, value)
    return {"path": str(path), "sha256": runner.sha256_file(path)}


def joint_plan(tmp_path):
    lids = [f"acdc:patient{x}" for x in ("007", "024", "026", "041")]
    lids += sorted(runner.JOINT_SUPERVISED_MNM_IDS)
    uids = ["acdc:patient001", "mnm:U001"]
    train = {"schema": "self_audit.data.v2", "role": "training", "synthetic": True,
             "shared": True, "domains": ["acdc", "mnm"], "label_budget": 8,
             "domain_budgets": {"acdc": 4, "mnm": 4},
             "labeled": [{"id": i} for i in lids], "unlabeled": [{"id": i} for i in uids],
             "domain_map": {i: i.split(":")[0] for i in lids + uids}}
    plan = {"joint_train": pin(tmp_path / "train.json", train)}
    def evaluation(key, domain, role, patient):
        return pin(tmp_path / (key + ".json"), {
            "schema": "self_audit.native_manifest.v1", "role": role, "synthetic": True,
            "patients": [{"id": patient, "domain": domain, "image": "image.npy", "image_sha256": "a" * 64,
                          "axes": ["T", "Z", "Y", "X"], "shape_tzyx": [2, 3, 8, 8],
                          "affine": np.eye(4).tolist(),
                          "masks": {"0": {"path": "mask.npy", "sha256": "b" * 64}}}]})
    plan["val"] = evaluation("val", "acdc", "development", "patient002")
    plan["mnm_val"] = evaluation("mnm_val", "mnm", "development", "V001")
    plan["test"] = evaluation("test", "acdc", "test", "patient101")
    template = json.loads((runner.ROOT / "configs/server_compact/joint_supervised.json").read_text())
    plan["joint_template"] = pin(tmp_path / "template.json", template)
    return plan


def test_joint_config_exact_namespaced_4_plus_4_and_no_teacher(tmp_path):
    plan = joint_plan(tmp_path)
    config = ready.bind_config(plan, "joint_supervised")
    assert config["teacher"] == {} and config["method"] == "supervised"
    assert config["labeled_batch_size"] == config["unlabeled_batch_size"] == 32
    config_pin = pin(tmp_path / "config.json", config)
    normalized, *_ = runner._validate_server_arm_config(
        config_pin["path"], config_pin["sha256"], plan["joint_train"]["path"], plan["joint_train"]["sha256"])
    assert normalized["steps"] == 10000
    # ACDC-L8 plus M&Ms would exceed this experiment's fixed label lineage.
    bad = copy.deepcopy(config)
    bad["teacher"] = {"source": "acdc_l8_checkpoint"}
    bad_pin = pin(tmp_path / "bad.json", bad)
    with pytest.raises(ValueError, match="teacher"):
        runner._validate_server_arm_config(bad_pin["path"], bad_pin["sha256"],
                                          plan["joint_train"]["path"], plan["joint_train"]["sha256"])


def test_joint_views_are_metadata_only_preserve_geometry_and_test_role(tmp_path, monkeypatch):
    plan = joint_plan(tmp_path)
    # No payload exists. Preparing these views must not try to read MRI or GT.
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("payload read during metadata composition"))
    val_pin, test_pin = ready.joint_eval_manifests(plan, tmp_path / "views")
    val, test = [json.loads(Path(p["path"]).read_text()) for p in (val_pin, test_pin)]
    assert [r["id"] for r in val["patients"]] == ["acdc:patient002", "mnm:V001"]
    assert test["role"] == "test" and test["patients"][0]["id"] == "acdc:patient101"
    assert test["patients"][0]["affine"] == np.eye(4).tolist()
    assert val["patients"][0]["masks"]["0"]["sha256"] == "b" * 64
    assert Path(val["patients"][0]["image"]).is_absolute()
    original = json.loads(Path(plan["val"]["path"]).read_text())
    original["patients"][0]["id"] = "patient007"
    plan["val"] = pin(Path(plan["val"]["path"]), original)
    with pytest.raises(ValueError, match="overlaps TRAIN"):
        ready.joint_eval_manifests(plan, tmp_path / "bad_views")


def test_equal_domain_selection_does_not_use_pooled_macro():
    classes = {c: {"dice": 0.5, "present_patients": 1} for c in ("BG", "RV", "MYO", "LV")}
    metrics = {"coverage": {"complete": True}, "frames": [], "class_first_macro4": .95,
               "datasets": {"acdc": {"class_first_macro4": .9, "class_first_foreground_macro": .8, "macro_patient": classes},
                            "mnm": {"class_first_macro4": .5, "class_first_foreground_macro": .4, "macro_patient": classes}}}
    summary = shared_selection_summary(metrics)
    assert summary["macro4"] == .7
    assert summary["foreground_macro"] == pytest.approx(.6)
    assert set(summary["domain_results"]) == {"acdc", "mnm"}
    metrics["coverage"]["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        shared_selection_summary(metrics)


def test_cache_disk_estimate_never_reads_u_masks(tmp_path):
    array = np.zeros((2, 3, 8, 8), np.float32)
    np.save(tmp_path / "image.npy", array)
    train = {"schema": "self_audit.data.v2", "role": "training", "label_budget": 1,
             "labeled": [{"id": "L", "masks": {"0": {"path": "missing_mask.npy"}}}],
             "unlabeled": [{"id": "U", "image": "image.npy"}]}
    p = pin(tmp_path / "train.json", train)
    assert ready.cache_disk_bytes(Path(p["path"]), p["sha256"]) == array.size * 16 + 4096
    train["unlabeled"][0]["masks"] = {"0": {"path": "missing.npy"}}
    p = pin(tmp_path / "train.json", train)
    with pytest.raises(ValueError, match="image-only"):
        ready.cache_disk_bytes(Path(p["path"]), p["sha256"])


def test_no_duplicate_run_or_gpu_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(ready, "bind_config", lambda *_: {})
    (tmp_path / "S1").mkdir()
    with pytest.raises(FileExistsError, match="exact resume"):
        ready.run_arm({"output_root": str(tmp_path)}, "S1")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, stdout="1234\n"))
    with pytest.raises(RuntimeError, match="existing compute"):
        ready.assert_gpu_idle()


def test_teacher_requires_completed_verified_b8_before_any_cache(tmp_path):
    b8 = pin(tmp_path / "b8.json", {"source": "missing"})
    runner.atomic_json(tmp_path / "teacher/STATUS.json", {"status": "UPLOAD_PENDING"})
    plan = {"b8_plan": b8, "teacher_root": str(tmp_path / "teacher")}
    with pytest.raises(ValueError, match="verified weight backup"):
        ready.verify_teacher(plan)


def test_teacher_binding_checks_physical_metrics_and_exact_source(tmp_path, monkeypatch):
    from self_audit import compact_teacher
    root = tmp_path / "teacher"
    b8 = pin(tmp_path / "b8.json", {"source": "/fixture/source", "source_commit": "c" * 40,
                                   "source_sha256": "d" * 64})
    runner.atomic_json(root / "STATUS.json", {"status": "TEACHER_COMPLETED_REMOTE_VERIFIED"})
    identity = {"source_commit": "c" * 40, "source_sha256": "d" * 64, "quality_gate": "GO",
                "metrics": {d: {"Macro4": .9} for d in ("val18", "test50")}}
    runner.atomic_json(root / "TEACHER_IDENTITY.json", identity)
    def validate(path, *, expected_receipt_sha256, checkpoint_dir, raw_student_path):
        assert expected_receipt_sha256 == runner.sha256_file(path)
        assert checkpoint_dir == root / "training/final"
        assert raw_student_path == root / "raw_student/raw_student.pt"
        return copy.deepcopy(identity)
    def verify_source(path, *, expected_commit, expected_closure):
        assert path == "/fixture/source"
        assert expected_commit == "c" * 40 and expected_closure == "d" * 64
    monkeypatch.setattr(compact_teacher, "validate_reproduction_receipt", validate)
    monkeypatch.setattr(compact_teacher, "verify_reproduction_source", verify_source)
    for domain, count in (("val18", 18), ("test50", 50)):
        runner.atomic_json(root / (domain + "_evaluation") / "METRICS.json", {
            "macro_patient": {c: {"dice": .9} for c in ("BG", "RV", "MYO", "LV")},
            "patients": [f"case{i}" for i in range(count)]})
    plan = {"b8_plan": b8, "teacher_root": str(root), "output_root": str(tmp_path / "out")}
    assert ready.teacher_binding(plan)["status"] == "COMPLETED"
    runner.atomic_json(root / "test50_evaluation/METRICS.json", {
        "macro_patient": {c: {"dice": .8} for c in ("BG", "RV", "MYO", "LV")}, "patients": list(range(50))})
    with pytest.raises(ValueError, match="physical evaluation"):
        ready.verify_teacher(plan)


def test_plan_pins_source_and_assets_before_gpu(tmp_path, monkeypatch):
    p = pin(tmp_path / "plan.json", {"schema": "self_audit.server_ready_arm.v1", "git_sha": "wrong"})
    monkeypatch.setattr(ready, "source_identity", lambda: {"git_head": "new", "closure_sha256": "a" * 64})
    with pytest.raises(ValueError, match="pinned commit/closure"):
        ready.load_plan(p["path"], p["sha256"])
