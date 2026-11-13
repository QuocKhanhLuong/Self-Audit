import random

import numpy as np
import torch
import pytest
from torch import nn

from self_audit.candidate_fuse.config import default_config
from self_audit.candidate_fuse.evaluation import (
    choose_deployment, evaluate_export, evaluate_exports, evaluate_modes,
)
from self_audit.candidate_fuse.export import export_model, load_export, verify_cold_load
from self_audit.candidate_fuse.student import FuseStudent
from self_audit.candidate_fuse.teacher import build_teacher


class _Frame:
    patient_id = "p1"
    frame_index = 0
    geometry = {"shape": (2, 2, 2), "spacing_zyx": (1.5, 1.0, 1.0),
                "affine_zyx": np.eye(4).tolist(), "context": "full_volume"}
    image = torch.zeros((1, 1, 2, 2, 2))
    target = torch.tensor([[[0, 1], [2, 3]], [[0, 1], [2, 3]]])


class _Data:
    def labeled_keys(self):
        return [("p1", 0)]

    def load(self, key, *, labeled):
        assert labeled
        return _Frame()


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = nn.Dropout3d()

    def forward(self, image, geometry, *, return_features=True):
        assert not self.training and not self.child.training
        logits = torch.zeros((1, 4, 2, 2, 2))
        logits[:, 0] = 1
        return {"region_logits": logits, "features": (torch.zeros(1),) * 5}


class _Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = nn.Dropout3d()

    def forward(self, image, geometry, *, teacher_features=None, keep_mask=None):
        assert not self.training and not self.child.training
        logits = torch.zeros((1, 4, 2, 2, 2))
        logits[:, 0] = 1
        return {"logits": logits}


def test_choose_deployment_uses_locked_inclusive_margin():
    assert choose_deployment({"student": {"fg_dice": 0.49},
                              "fusion": {"fg_dice": 0.50}})["mode"] == "student"
    assert choose_deployment({"student": {"fg_dice": 0.489},
                              "fusion": {"fg_dice": 0.50}})["mode"] == "fusion"


def test_evaluate_modes_uses_same_frames_and_restores_rng_and_modes():
    teacher, student = _Teacher().train(), _Student().train()
    teacher.child.training = False
    student.child.training = False
    module_flags = [(module, module.training)
                    for model in (teacher, student) for module in model.modules()]
    random.seed(8); np.random.seed(8); torch.manual_seed(8)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    scores = evaluate_modes(teacher, student, _Data())
    assert set(scores) == {"teacher", "student", "fusion"}
    assert scores["student"]["patient_count"] == 1
    assert all(module.training == flag for module, flag in module_flags)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_fusion_only_evaluation_sets_both_models_to_eval_and_restores_children():
    teacher, student = _Teacher().train(), _Student().train()
    teacher.child.training = False
    student.child.training = False
    flags = [(module, module.training)
             for model in (teacher, student) for module in model.modules()]
    result = evaluate_modes(teacher, student, _Data(), modes=("fusion",))
    assert set(result) == {"fusion"}
    assert all(module.training == flag for module, flag in flags)


def test_student_export_strict_load_and_cold_load_bitwise(tmp_path):
    config = {"model": {"student": {"channels": [4, 4, 4, 4, 4]}}}
    student = FuseStudent(config, include_fusion=True, include_mimic=True).eval()
    artifact = tmp_path / "student.pt"
    export_model(student, None, config, artifact, mode="student")
    loaded = load_export(artifact)
    geometry = {"shape": (2, 16, 16), "spacing_zyx": (1.0, 1.0, 1.0),
                "affine_zyx": np.eye(4).tolist(), "context": "full_volume"}
    volumes = [(torch.zeros((1, 1, 2, 16, 16)), geometry) for _ in range(3)]
    assert all(torch.equal(student.infer_logits(image, geometry), loaded(image, geometry))
               for image, geometry in volumes)
    assert verify_cold_load(artifact, volumes, original_model=student)["bitwise_equal"]


def test_teacher_export_load_is_self_contained_without_pretrained_weights(tmp_path, monkeypatch):
    import self_audit.candidate_q3d.teacher_resenc as teacher_api
    from self_audit.candidate_fuse.export import _build_teacher_architecture

    original_builder = teacher_api.build_res_enc_l
    monkeypatch.setattr(
        teacher_api, "build_res_enc_l",
        lambda strides: original_builder(
            strides, features_per_stage=(4, 8, 12, 16, 20, 20),
            n_blocks_per_stage=(1, 1, 1, 1, 1, 1)),
    )
    config = default_config(synthetic=False)
    # The config's pinned source path is deliberately unavailable; export load
    # must construct architecture only, then strict-load the full FUSE state.
    config["model"]["teacher"]["weights_path"] = "/missing/pretrained/weights.pth"
    teacher = _build_teacher_architecture(config)
    artifact = tmp_path / "teacher.pt"
    export_model(None, teacher, config, artifact, mode="teacher")
    loaded = load_export(artifact)
    assert loaded.mode == "teacher"
    assert set(loaded.teacher.model.state_dict()) == set(teacher.model.state_dict())
    assert all(torch.equal(loaded.teacher.model.state_dict()[key], value)
               for key, value in teacher.model.state_dict().items())


def test_load_export_checks_sidecar_hash_before_deserialization(tmp_path, monkeypatch):
    config = {"model": {"student": {"channels": [4, 4, 4, 4, 4]}}}
    student = FuseStudent(config, include_fusion=False, include_mimic=False).eval()
    artifact = tmp_path / "student.pt"
    export_model(student, None, config, artifact, mode="student")
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("torch.load called")))
    try:
        load_export(artifact)
    except ValueError as exc:
        assert "sidecar" in str(exc) or "SHA-256" in str(exc)
    else:
        raise AssertionError("corrupt FUSE artifact passed sidecar validation")


def test_evaluate_export_loads_role_checked_native_manifest(tmp_path):
    from self_audit.candidate_fuse.data import write_synthetic_manifests

    manifests = write_synthetic_manifests(tmp_path / "data")
    config = {"model": {"student": {"channels": [4, 4, 4, 4, 4]}}}
    student = FuseStudent(config, include_fusion=True, include_mimic=True).eval()
    artifact = tmp_path / "student.pt"
    result = export_model(student, None, config, artifact, mode="student")
    measured = evaluate_export(artifact, manifests["development"]["path"],
                               manifests["development"]["sha256"],
                               role="development", device="cpu")
    assert measured["mode"] == "student"
    assert measured["metrics"]["patient_count"] == 1
    assert measured["artifact_sha256"] == result["sha256"]
    cli_result = evaluate_export(artifact, result["sha256"],
                                 manifests["development"]["path"],
                                 manifests["development"]["sha256"],
                                 str(tmp_path / "cli-out"), role="development")
    assert (tmp_path / "cli-out" / "FUSE_DEVELOPMENT_STUDENT_METRICS.json").is_file()
    assert cli_result["artifact_sha256"] == result["sha256"]


def test_evaluate_exports_commits_one_three_mode_test_transaction(tmp_path):
    from self_audit.candidate_fuse.data import write_synthetic_manifests

    config = default_config(synthetic=True)
    teacher, _ = build_teacher(config, synthetic=True)
    student = FuseStudent(config, teacher_channels=teacher.feature_channels,
                          teacher_phases=teacher.feature_phases,
                          include_fusion=True, include_mimic=True).eval()
    paths = {mode: tmp_path / f"{mode}.pt" for mode in ("teacher", "student", "fusion")}
    receipts = {
        "teacher": export_model(None, teacher, config, paths["teacher"], mode="teacher"),
        "student": export_model(student, None, config, paths["student"], mode="student"),
        "fusion": export_model(student, teacher, config, paths["fusion"], mode="fusion"),
    }
    geometry = {"shape": (3, 8, 9), "spacing_zyx": (2.5, .7, 1.3),
                "affine_zyx": np.diag([2.5, .7, 1.3, 1.]).tolist(),
                "context": "full_volume"}
    cold_volumes = [(torch.zeros((1, 1, 3, 8, 9)), geometry) for _ in range(3)]
    assert verify_cold_load(paths["teacher"], cold_volumes,
                            original_model=teacher)["bitwise_equal"]
    assert verify_cold_load(paths["fusion"], cold_volumes,
                            original_model={"teacher": teacher, "student": student})["bitwise_equal"]
    manifests = write_synthetic_manifests(tmp_path / "fixtures")
    test_manifest = manifests["test"]
    ledger = evaluate_exports(receipts, test_manifest["path"], tmp_path / "once", "cpu",
                              "test", test_manifest["sha256"])
    assert ledger["status"] == "COMMITTED"
    assert set(ledger["metrics"]) == {"teacher", "student", "fusion"}
    assert (tmp_path / "once" / "TEST_TRANSACTION.json").is_file()
    try:
        evaluate_exports(receipts, test_manifest["path"], tmp_path / "once", "cpu",
                         "test", test_manifest["sha256"])
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("completed FUSE TEST transaction was replayed")


def test_single_mode_test_rejected_before_fusedata_or_mask_reads(tmp_path, monkeypatch):
    from self_audit.candidate_fuse import data as fuse_data
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"artifact")
    import hashlib
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    sidecar = artifact.with_suffix(".pt.manifest.json")
    sidecar.write_text(__import__("json").dumps({
        "schema": "self_audit.candidate_fuse.export.v1", "artifact": artifact.name,
        "artifact_sha256": digest, "artifact_bytes": artifact.stat().st_size,
    }))
    monkeypatch.setattr(fuse_data, "FuseData", lambda *a, **k: (_ for _ in ()).throw(AssertionError("FuseData read")))
    with pytest.raises(ValueError, match="locked three-mode TEST transaction"):
        evaluate_export(artifact, "unused.json", "0" * 64, role="test")
