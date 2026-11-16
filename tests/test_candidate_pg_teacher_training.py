from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from self_audit.candidate_pg.config import default_config, validate_config
from self_audit.candidate_pg.logging import EpochLogger
from self_audit.candidate_pg.teacher_training import (
    _lambda_u,
    _q3d_teacher_loss,
    _soft_consistency,
    run_teacher_stage,
)


torch.set_num_threads(1)


class _TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Conv3d(1, 8, 1)
        self.region = nn.Conv3d(8, 4, 1)
        self.boundary = nn.Conv3d(8, 4, 1)
        self.sdf_head = nn.Conv3d(8, 4, 1)
        self.skip_dropout_calls: list[float | None] = []

    def forward(self, image, geometry, return_features=True, skip_dropout=None, dropout_generator=None):
        self.skip_dropout_calls.append(skip_dropout)
        feature = self.body(image)
        if skip_dropout is not None:
            feature = torch.nn.functional.dropout3d(feature, p=float(skip_dropout), training=self.training)
        sdf_all = torch.tanh(self.sdf_head(feature))
        return {
            "region_logits": self.region(feature),
            "boundary_logits": self.boundary(feature),
            "sdf": sdf_all[:, 1:],
            "sdf_all": sdf_all,
            "features": (feature,) * 5,
        }


def _frame(index: int, *, labeled: bool, patient: str, shape: tuple[int, int, int] = (2, 16, 16)) -> dict:
    target = torch.zeros(shape, dtype=torch.long)
    target[:, 4:12, 4:12] = (index % 3) + 1
    return {
        "image": torch.full((1, 1, *shape), 0.5 + 0.01 * (index % 2)),
        "target": target if labeled else None,
        "geometry": {
            "shape": shape,
            "spacing_zyx": (1.0, 1.0, 1.0),
            "affine_zyx": torch.eye(4).tolist(),
            "context": "full_volume",
        },
        "patient_id": patient,
        "frame_index": index,
    }


def _config(updates: int = 1) -> dict:
    config = validate_config(default_config(synthetic=True))
    config["teacher"]["updates"] = updates
    return config


def test_lambda_u_locked_warmup_and_ramp():
    assert _lambda_u(0, 8000, warmup=800, ramp_fraction=0.1, maximum=1.0) == 0.0
    assert _lambda_u(799, 8000, warmup=800, ramp_fraction=0.1, maximum=1.0) == 0.0
    assert _lambda_u(800, 8000, warmup=800, ramp_fraction=0.1, maximum=1.0) == pytest.approx(1 / 800)
    assert _lambda_u(1600, 8000, warmup=800, ramp_fraction=0.1, maximum=1.0) == 1.0


def test_unimatch_consistency_uses_detached_hard_argmax_ce():
    logits = torch.tensor([[[[[2.0]]], [[[0.0]]], [[[-1.0]]], [[[-2.0]]]]])
    target_prob = torch.tensor([[[[[0.49]]], [[[0.51]]], [[[0.0]]], [[[0.0]]]]])
    confident = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    expected = torch.nn.functional.cross_entropy(logits, torch.ones((1, 1, 1, 1), dtype=torch.long))
    torch.testing.assert_close(_soft_consistency(logits, target_prob, confident), expected)


def test_teacher_stage_active_unimatch_smoke_uses_weak_fp_and_strict16_canvas(tmp_path):
    config = _config(1)
    config["teacher"]["semi"]["lambda_u"]["warmup_updates_labeled_only"] = 0
    train = {
        "labeled": [_frame(0, labeled=True, patient="L0", shape=(2, 64, 64))],
        "unlabeled": [_frame(1, labeled=False, patient="U0", shape=(2, 64, 64)), _frame(2, labeled=False, patient="U1", shape=(2, 64, 64))],
    }
    teacher = _TinyTeacher()
    receipt = run_teacher_stage(config, tmp_path, train, train["labeled"], train["labeled"], teacher=teacher, max_updates=1)
    assert receipt["status"] == "COMPLETED"
    assert 0.5 in teacher.skip_dropout_calls  # FP is weak-A with feature dropout


def test_q3d_adapter_has_exact_three_keys(monkeypatch):
    import self_audit.candidate_q3d.losses as q3d_losses
    import self_audit.candidate_pg.teacher_training as training

    seen = {}

    def fake_loss(outputs, labels, sdf_targets, **kwargs):
        seen["keys"] = set(outputs)
        return outputs["region_logits"].sum() * 0.0, {"region": torch.tensor(0.0), "boundary": torch.tensor(0.0), "sdf_near_far": torch.tensor(0.0)}

    monkeypatch.setattr(q3d_losses, "q3d_teacher_loss", fake_loss)
    monkeypatch.setattr(training, "_sdf_targets4", lambda labels, geometry, radius_mm: torch.zeros((labels.shape[0], 4, *labels.shape[-3:]), device=labels.device))
    logits = torch.zeros((1, 4, 2, 4, 4), requires_grad=True)
    output = {"region_logits": logits, "boundary_logits": logits, "sdf_all": logits, "sdf": logits[:, 1:]}
    labels = torch.zeros((1, 2, 4, 4), dtype=torch.long)
    _q3d_teacher_loss(output, labels, {"spacing_zyx": (1.0, 1.0, 1.0)}, sdf_radius=5.0, sdf_near=2.5, sdf_weight=1.0)
    assert seen["keys"] == {"region_logits", "boundary_logits", "sdf"}


def test_teacher_stage_labeled_step_and_checkpoint_state(tmp_path):
    config = _config(1)
    train = {
        "labeled": [_frame(0, labeled=True, patient="L0")],
        "unlabeled": [_frame(1, labeled=False, patient="U0"), _frame(2, labeled=False, patient="U1")],
    }
    teacher = _TinyTeacher()
    receipt = run_teacher_stage(config, tmp_path, train, train["labeled"], train["labeled"], teacher=teacher, max_updates=1)
    assert receipt["status"] == "COMPLETED"
    assert receipt["selected_update"] is None  # no epoch boundary in a one-step fixture
    payload = torch.load(tmp_path / "S2" / "teacher_latest.pt", map_location="cpu", weights_only=False)
    assert {"model", "optimizer", "sampler", "augmentation", "fp_generator_state", "threshold", "rng"} <= set(payload)
    assert any(value is None for value in teacher.skip_dropout_calls)  # labeled-only warmup
    assert json.loads((tmp_path / "S2" / "TEACHER_BEST_RECEIPT.json").read_text())["status"] == "COMPLETED"


def test_teacher_stage_epoch_logs_val_only_selection(tmp_path):
    config = _config(80)
    train = {
        "labeled": [_frame(0, labeled=True, patient="L0"), _frame(1, labeled=True, patient="L1")],
        "unlabeled": [_frame(2, labeled=False, patient="U0"), _frame(3, labeled=False, patient="U1")],
    }
    val = [_frame(4, labeled=True, patient="V0")]
    receipt = run_teacher_stage(config, tmp_path, train, val, val, teacher=_TinyTeacher())
    assert receipt["selected_update"] == 80
    rows = (tmp_path / "S2" / "epochs.jsonl").read_text().splitlines()
    assert len(rows) == 3
    assert {json.loads(row)["split"] for row in rows} == {"train", "val", "test"}
    assert sum(json.loads(row)["best_val"] for row in rows) == 1


def test_teacher_stage_center_pads_unequal_u_partner(tmp_path):
    config = _config(1)
    train = {
        "labeled": [_frame(0, labeled=True, patient="L0")],
        "unlabeled": [_frame(1, labeled=False, patient="U0", shape=(2, 16, 16)), _frame(2, labeled=False, patient="U1", shape=(2, 20, 20))],
    }
    receipt = run_teacher_stage(config, tmp_path, train, train["labeled"], train["labeled"], teacher=_TinyTeacher(), max_updates=1)
    assert receipt["status"] == "COMPLETED"


def test_teacher_stage_resume_after_checkpoint_before_commit_is_row_exact(tmp_path, monkeypatch):
    config = _config(80)
    train = {
        "labeled": [_frame(0, labeled=True, patient="L0"), _frame(1, labeled=True, patient="L1")],
        "unlabeled": [_frame(2, labeled=False, patient="U0"), _frame(3, labeled=False, patient="U1")],
    }
    val = [_frame(4, labeled=True, patient="V0")]
    interrupted = tmp_path / "interrupted"
    clean = tmp_path / "clean"
    original = EpochLogger.commit_epoch
    calls = {"count": 0}

    def fail_once(self, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("injected S2 crash after checkpoint")
        return original(self, **kwargs)

    monkeypatch.setattr(EpochLogger, "commit_epoch", fail_once)
    with pytest.raises(RuntimeError, match="injected S2"):
        run_teacher_stage(config, interrupted, train, val, val, teacher=_TinyTeacher())
    monkeypatch.setattr(EpochLogger, "commit_epoch", original)
    run_teacher_stage(config, interrupted, train, val, val, teacher=_TinyTeacher(), resume=True)
    run_teacher_stage(config, clean, train, val, val, teacher=_TinyTeacher())

    resumed = [json.loads(line) for line in (interrupted / "S2" / "epochs.jsonl").read_text().splitlines()]
    expected = [json.loads(line) for line in (clean / "S2" / "epochs.jsonl").read_text().splitlines()]
    assert {(row["epoch"], row["split"], row["path"]) for row in resumed} == {
        (1, "train", "teacher"), (1, "val", "teacher"), (1, "test", "teacher")
    }
    for row in resumed + expected:
        row.pop("seconds", None)
    assert resumed == expected
