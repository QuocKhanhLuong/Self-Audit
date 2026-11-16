from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from self_audit.candidate_pg.codebook import (
    StudentCodeBranch,
    TeacherCodebook,
    assign_kl,
    rho,
    soft_cross_entropy,
    usage_metrics,
    vq_ste,
)


class _LoopFrame:
    def __init__(self, patient_id: str, *, labeled: bool) -> None:
        self.patient_id = patient_id
        self.frame_index = 0
        self.image = torch.linspace(0.0, 1.0, 2 * 16 * 16, dtype=torch.float32).reshape(1, 2, 16, 16)
        self.target = torch.zeros(2, 16, 16, dtype=torch.long) if labeled else None
        self.geometry = {
            "shape": (2, 16, 16),
            "spacing_zyx": (1.0, 1.0, 1.0),
            "affine_zyx": torch.eye(4).tolist(),
            "context": "full_volume",
        }


class _LoopData:
    """Explicit synthetic L1/U2 fixture with two different U patients."""

    def __init__(self) -> None:
        self.frames = {
            ("L0", 0): _LoopFrame("L0", labeled=True),
            ("U0", 0): _LoopFrame("U0", labeled=False),
            ("U1", 0): _LoopFrame("U1", labeled=False),
        }
        self.loaded_pairs: list[tuple[str, str]] = []

    def labeled_keys(self) -> list[tuple[str, int]]:
        return [("L0", 0)]

    def unlabeled_keys(self) -> list[tuple[str, int]]:
        return [("U0", 0), ("U1", 0)]

    def load(self, key: tuple[str, int], *, labeled: bool) -> _LoopFrame:
        frame = self.frames[key]
        if labeled != (frame.target is not None):
            raise AssertionError("U/L mask access contract was violated")
        if not labeled:
            # The fixture itself records the two U patients used by each step.
            self.loaded_pairs.append((key[0], "image_only"))
        return frame


class _PhaseTeacher(torch.nn.Module):
    feature_phases = (
        ((1, 1, 1), (0.0, 0.0, 0.0)),
        ((1, 2, 2), (0.0, 0.0, 0.0)),
        ((1, 4, 4), (0.0, 0.0, 0.0)),
        ((1, 8, 8), (0.0, 0.0, 0.0)),
        ((1, 16, 16), (0.0, 0.0, 0.0)),
    )

    def __init__(self) -> None:
        super().__init__()
        self.feature3 = torch.nn.Conv3d(1, 256, 1)
        self.region = torch.nn.Conv3d(1, 4, 1)
        self.anchor = torch.nn.Parameter(torch.tensor(0.25))
        self.last_packets: list[dict[str, torch.Tensor]] = []

    def forward(self, image: torch.Tensor, geometry: dict, *, return_features: bool = True):
        pooled = F.avg_pool3d(image, kernel_size=(1, 8, 8), stride=(1, 8, 8))
        feature3 = self.feature3(pooled)
        region = self.region(image) + self.anchor
        packet = {
            "region_logits": region,
            "features": (
                pooled.repeat(1, 32, 1, 1, 1),
                pooled.repeat(1, 64, 1, 1, 1),
                pooled.repeat(1, 128, 1, 1, 1),
                feature3,
                pooled.repeat(1, 512, 1, 1, 1),
            ),
        }
        self.last_packets.append(packet)
        return packet


class _PhaseStudent(torch.nn.Module):
    phases = _PhaseTeacher.feature_phases


def _cfg() -> dict[str, object]:
    return {
        "teacher_channels": 5,
        "student_channels": 2,
        "embedding_dim": 3,
        "codebook_size": 4,
        "num_classes": 4,
        "rho_temperature": 1.0,
        "commit_beta": 0.25,
        "usage_entropy_weight": 0.01,
        "dead_code_restart_every": 1,
        "seed": 42,
    }


def test_rho_known_answer_preserves_code_order() -> None:
    z = torch.tensor([[0.0, 0.0]])
    C = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    actual = rho(z, C)
    expected = torch.softmax(torch.tensor([[0.0, -1.0]]), dim=-1)
    torch.testing.assert_close(actual, expected)
    # A caller's C order is part of the KD contract; no sorting is permitted.
    torch.testing.assert_close(rho(z, C.flip(0)), expected.flip(1))


def test_vq_ste_has_identity_z_gradient_and_no_codebook_gradient() -> None:
    z = torch.tensor([[0.2, -0.1]], requires_grad=True)
    C = torch.tensor([[0.0, 0.0], [2.0, 2.0]], requires_grad=True)
    quantized = vq_ste(z, C)
    quantized.sum().backward()
    torch.testing.assert_close(z.grad, torch.ones_like(z))
    assert C.grad is None


def test_teacher_forward_loss_and_trainable_codebook_gradient() -> None:
    torch.manual_seed(4)
    teacher = TeacherCodebook(_cfg())
    features = torch.randn(1, 5, 2, 2, 2, requires_grad=True)
    valid = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    valid[:, 0, 0, 0] = False
    labels = torch.randint(0, 4, (1, 2, 2, 2))
    output = teacher(features, valid)
    assert output["z"].shape == (1, 3, 2, 2, 2)
    assert output["rho"].shape == (1, 4, 2, 2, 2)
    assert output["q_logits"].shape == (1, 4, 2, 2, 2)
    components = teacher.loss(output, labels, return_components=True)
    components["loss"].backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert teacher.C.grad is not None and torch.isfinite(teacher.C.grad).all()
    assert components["metrics"]["valid_count"] == 7


def test_student_codebook_is_frozen_and_assign_kl_detaches_teacher() -> None:
    torch.manual_seed(7)
    teacher = TeacherCodebook(_cfg())
    student = StudentCodeBranch(_cfg(), teacher)
    t_feature = torch.randn(1, 5, 1, 2, 2, requires_grad=True)
    s_feature = torch.randn(1, 2, 1, 2, 2, requires_grad=True)
    t_output = teacher(t_feature)
    s_output = student(s_feature)
    loss = student.loss(s_output, t_output, return_components=True)["loss"]
    loss.backward()
    assert s_feature.grad is not None and torch.isfinite(s_feature.grad).all()
    assert t_feature.grad is None
    assert student.C.requires_grad is False
    assert student.C.grad is None
    # The pure helper has the same detach boundary.
    t_logits = torch.randn(5, 4, requires_grad=True)
    s_logits = torch.randn(5, 4, requires_grad=True)
    t = torch.softmax(t_logits, dim=-1)
    s = torch.softmax(s_logits, dim=-1)
    value = assign_kl(t, s)
    value.backward()
    assert t_logits.grad is None


def test_labeled_unlabeled_qtask_isolation() -> None:
    teacher = TeacherCodebook(_cfg())
    features = torch.randn(1, 5, 1, 2, 2)
    output = teacher(features)
    labels = torch.zeros(1, 1, 2, 2, dtype=torch.long)
    labeled = teacher.loss(output, labels, labeled=True, return_components=True)
    assert float(labeled["qtask"].detach()) > 0.0
    with pytest.raises(ValueError, match="cannot consume GT"):
        teacher.loss(output, labels, labeled=False)
    pseudo = torch.zeros(1, 1, 2, 2, dtype=torch.long)
    unlabeled = teacher.loss(output, pseudo_labels=pseudo, labeled=False, return_components=True)
    assert float(unlabeled["qtask"].detach()) > 0.0
    student = StudentCodeBranch(_cfg(), teacher)
    s_output = student(torch.randn(1, 2, 1, 2, 2))
    # CBKD L_Qtask is soft CE only; Dice belongs to the S3 teacher task.
    torch.testing.assert_close(
        student.loss_qtask(s_output, labels),
        soft_cross_entropy(s_output["q_logits"], labels, s_output["valid_mask"]),
    )
    with pytest.raises(ValueError, match="cannot consume GT"):
        student.loss(s_output, output, labels=labels, labeled=False)


def test_dead_code_restart_state_and_usage_entropy_metrics() -> None:
    teacher = TeacherCodebook(_cfg())
    # Make all observed assignments code 0 so codes 1..3 are dead at step 1.
    assignments = torch.zeros(1, 1, 2, 2, dtype=torch.long)
    features = torch.randn(1, 3, 1, 2, 2)
    result = teacher.update_usage(assignments, features=features)
    assert result["step"] == 1
    assert result["restarted_codes"] == [1, 2, 3]
    assert result["restart_count"] == 3
    assert result["last_restart_step"] == 1
    metrics = usage_metrics(assignments, 4)
    assert math.isclose(metrics["usage_entropy"], 0.0, abs_tol=1e-12)
    assert math.isclose(metrics["perplexity"], 1.0, abs_tol=1e-12)
    assert metrics["dead_code_count"] == 3
    assert metrics["codes_per_class"] == {0: 0, 1: 0, 2: 0, 3: 0}
    assert "_extra_state" in teacher.state_dict()


def test_dead_code_window_restarts_previously_used_code_after_window_expires() -> None:
    cfg = _cfg()
    cfg["dead_code_restart_every"] = 2
    teacher = TeacherCodebook(cfg)
    features = torch.randn(1, 3, 1, 2, 2)
    used_code_one = torch.ones(1, 1, 2, 2, dtype=torch.long)
    used_code_zero = torch.zeros(1, 1, 2, 2, dtype=torch.long)
    teacher.update_usage(used_code_one, features=features, step=1)
    first_boundary = teacher.update_usage(used_code_zero, features=features, step=2)
    assert 1 not in first_boundary["restarted_codes"]  # code 1 was used in window 1
    teacher.update_usage(used_code_zero, features=features, step=3)
    second_boundary = teacher.update_usage(used_code_zero, features=features, step=4)
    assert 1 in second_boundary["restarted_codes"]  # no assignment in window 2
    assert second_boundary["last_restart_step"] == 4


def test_internal_pad_is_excluded_from_student_commitment() -> None:
    teacher = TeacherCodebook(_cfg())
    student = StudentCodeBranch(_cfg(), teacher)
    feature = torch.randn(1, 2, 1, 2, 2)
    output = student(feature)
    valid = torch.tensor([[[[True, False], [False, False]]]])
    # One valid cell gives a finite scalar and no accidental denominator-zero.
    teacher_output = teacher(torch.randn(1, 5, 1, 2, 2))
    loss = student.loss(output, teacher_output, valid_mask=valid, labeled=False)
    assert torch.isfinite(loss)


def _loop_config() -> dict[str, object]:
    return {
        "runtime": {"synthetic": True},
        "codebook": {
            "P_T": "conv1x1x1_256_to_64",
            "H_T_Q": "conv1x1x1_64_64_relu_64_4",
            "codebook": {
                "size": 4,
                "dim": 64,
                "dead_code_restart_every": 1,
                "usage_entropy_weight": 0.01,
            },
            "commit_beta": 0.25,
            "rho_temperature": 1.0,
            "updates": 2,
            "eval_every_updates": 1,
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
            # Force the tiny fixture to exercise the persisted restart state.
            "dead_code_threshold": 0.9,
        },
    }


def _run_loop(out: Path, *, updates: int, resume: bool = False):
    from self_audit.candidate_pg.codebook import run_codebook

    config = _loop_config()
    config["codebook"] = dict(config["codebook"])
    config["codebook"]["updates"] = updates
    data = _LoopData()
    torch.manual_seed(1234)
    teacher = _PhaseTeacher()
    before = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
    result = run_codebook(
        config,
        out,
        data,
        data,
        data,
        teacher,
        _PhaseStudent(),
        max_updates=updates,
        resume=resume,
    )
    return result, data, teacher, before


def test_run_codebook_synthetic_loop_checkpoint_resume_and_receipts(tmp_path: Path) -> None:
    """Exercise S3's real loop, native phases, L1/U2, logging, and replay."""

    from self_audit.candidate_pg.codebook import TeacherCodebook

    full_out = tmp_path / "full"
    partial_out = tmp_path / "resume"
    full, full_data, full_teacher, full_before = _run_loop(full_out, updates=2)
    _run_loop(partial_out, updates=1)
    resumed, resumed_data, resumed_teacher, resumed_before = _run_loop(partial_out, updates=2, resume=True)

    assert full["status"] == resumed["status"] == "COMPLETED"
    assert full["codebook_frozen"] is True and resumed["codebook_frozen"] is True
    # The fallback synthetic path still enforces one anchor plus a different
    # U patient partner on every update.
    for loaded, expected_count in ((full_data.loaded_pairs, 4), (resumed_data.loaded_pairs, 2)):
        assert len(loaded) == expected_count
        assert all(loaded[index][0] != loaded[index + 1][0] for index in range(0, len(loaded), 2))
    # Teacher is a frozen target in S3; neither run may mutate its parameters,
    # and packets observed inside the teacher are detached.
    assert all(torch.equal(value, full_teacher.state_dict()[name]) for name, value in full_before.items())
    assert all(torch.equal(value, resumed_teacher.state_dict()[name]) for name, value in resumed_before.items())
    assert full_teacher.last_packets and all(not tensor.requires_grad for packet in full_teacher.last_packets for tensor in packet["features"])

    full_state = torch.load(full_out / "codebook.pt", map_location="cpu", weights_only=False)["codebook"]
    resumed_state = torch.load(partial_out / "codebook.pt", map_location="cpu", weights_only=False)["codebook"]
    assert full_state["C"].shape == (4, 64)
    for name in ("P_T.weight", "P_T.bias", "C", "H_T_Q.0.weight", "H_T_Q.2.weight"):
        torch.testing.assert_close(full_state[name], resumed_state[name], rtol=0, atol=0)
    assert int(full_state["restart_count"]) > 0
    reloaded = TeacherCodebook(_loop_config())
    reloaded.load_state_dict(full_state)
    assert all(not parameter.requires_grad for parameter in reloaded.parameters())
    assert reloaded.C.shape == (4, 64)

    required_columns = {
        "fg_dice", "rv_dice", "myo_dice", "lv_dice",
        "fg_hd95", "rv_hd95", "myo_hd95", "lv_hd95",
        "fg_assd", "rv_assd", "myo_assd", "lv_assd",
    }
    with (full_out / "epochs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        assert required_columns.issubset(set(rows[0]))
    assert len(rows) == 6
    assert {(row["split"], row["path"]) for row in rows} == {("train", "qT"), ("val", "qT"), ("test", "qT")}
    assert {int(row["epoch"]) for row in rows} == {1, 2}
    jsonl_rows = (full_out / "epochs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl_rows) == len(rows)
    for epoch in (1, 2):
        patient_path = full_out / f"patients_epoch{epoch:03d}.csv"
        assert patient_path.is_file()
        patient_rows = list(csv.DictReader(patient_path.open(newline="", encoding="utf-8")))
        assert {row["split"] for row in patient_rows} == {"train", "val", "test"}
        assert {row["path"] for row in patient_rows} == {"qT"}


def test_run_codebook_resume_after_checkpoint_before_commit_is_row_exact(tmp_path: Path, monkeypatch) -> None:
    from self_audit.candidate_pg.logging import EpochLogger
    from self_audit.candidate_pg.codebook import run_codebook

    clean_out = tmp_path / "clean"
    interrupted_out = tmp_path / "interrupted"
    _run_loop(clean_out, updates=2)
    original = EpochLogger.commit_epoch
    calls = {"count": 0}

    def fail_once(self, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("injected S3 crash after checkpoint")
        return original(self, **kwargs)

    monkeypatch.setattr(EpochLogger, "commit_epoch", fail_once)
    with pytest.raises(RuntimeError, match="injected S3"):
        _run_loop(interrupted_out, updates=2)
    monkeypatch.setattr(EpochLogger, "commit_epoch", original)
    _run_loop(interrupted_out, updates=2, resume=True)

    clean_rows = [json.loads(line) for line in (clean_out / "epochs.jsonl").read_text().splitlines()]
    resumed_rows = [json.loads(line) for line in (interrupted_out / "epochs.jsonl").read_text().splitlines()]
    assert len(resumed_rows) == len(clean_rows) == 6
    assert {(row["epoch"], row["split"], row["path"]) for row in resumed_rows} == {
        (1, "train", "qT"), (1, "val", "qT"), (1, "test", "qT"),
        (2, "train", "qT"), (2, "val", "qT"), (2, "test", "qT"),
    }
    for row in clean_rows + resumed_rows:
        row.pop("seconds", None)
    assert resumed_rows == clean_rows
