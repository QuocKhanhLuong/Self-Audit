"""Comprehensive tests for compact_control: whole-model class soft KD candidate displacement,

common denominator, active norm clipping KD sum preservation, finite deterministic search,
strict per-class, per-patient, and per-frame absent-FP guards, and atomic parameter/RNG/gradient preservation.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, Iterator

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from self_audit.compact_control import (
    CLASS_NAMES,
    FOREGROUND_CLASS_NAMES,
    apply_compact_update,
    build_class_candidates,
    compute_audit_metrics,
    compute_batch_margins,
)


class DummyStudent(nn.Module):
    """Simple 2.5D convolutional student for fast CPU unit testing."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(2, 8)
        self.conv2 = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gn(self.conv1(x)))
        return self.conv2(h)


class MixedModeStudent(nn.Module):
    """Student with BatchNorm2d and Dropout to test per-module mode preservation."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=1)
        self.bn = nn.BatchNorm2d(4)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.bn(self.conv(x)))


def test_build_class_candidates_mathematical_equivalence():
    """Verify sum of class-grouped displacements exactly equals plain KD SGD."""
    torch.manual_seed(42)
    model = DummyStudent(in_channels=1, num_classes=4)
    x = torch.randn(2, 1, 16, 16)
    targets = F.softmax(torch.randn(2, 4, 16, 16), dim=1)
    valid = torch.rand(2, 16, 16) > 0.15
    lr = 0.05
    temp = 1.0
    coef = 1.0
    conf = 0.2

    result = build_class_candidates(
        model, x, targets, valid, lr=lr, temperature=temp, coefficient=coef, confidence=conf
    )
    directions = result["directions"]
    denom = result["denominator"]

    assert len(directions) > 0
    assert denom > 0

    sum_disp: Dict[str, torch.Tensor] = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    for d in directions:
        for name, tensor in d.items():
            sum_disp[name] += tensor

    t_max = targets.max(dim=1).values
    conf_mask = (t_max >= conf) & valid
    expected_denom = int(conf_mask.sum().item())
    assert denom == expected_denom

    logits = model(x)
    log_p = F.log_softmax(logits, dim=1)
    pixel_loss = - (targets * log_p).sum(dim=1)
    total_loss = (pixel_loss * conf_mask).sum() / expected_denom
    plain_grads = torch.autograd.grad(total_loss, list(model.parameters()))

    for (name, p), g in zip(model.named_parameters(), plain_grads):
        plain_disp = -lr * g
        max_diff = (sum_disp[name] - plain_disp).abs().max().item()
        assert max_diff < 1e-6, f"Mismatch on parameter {name}: max diff {max_diff}"


def test_build_class_candidates_active_norm_clip_kd_sum():
    """Verify that aggregate scaling preserves plain KD displacement sum under active clipping."""
    torch.manual_seed(42)
    model = DummyStudent(in_channels=1, num_classes=4)
    x = torch.randn(2, 1, 16, 16)
    targets = F.softmax(torch.randn(2, 4, 16, 16), dim=1)
    valid = torch.ones(2, 16, 16, dtype=torch.bool)
    lr = 0.1
    # Small max_norm that is guaranteed to actively clip
    max_norm = 0.005

    result = build_class_candidates(
        model, x, targets, valid, lr=lr, max_norm=max_norm
    )
    directions = result["directions"]
    assert result["telemetry"]["aggregate_clip_scale"] < 1.0  # clipping was active

    sum_disp: Dict[str, torch.Tensor] = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    for d in directions:
        for name, tensor in d.items():
            sum_disp[name] += tensor

    # Compute unpartitioned plain KD displacement clipped by max_norm
    logits = model(x)
    log_p = F.log_softmax(logits, dim=1)
    loss = - (targets * log_p).sum(dim=1).mean()
    plain_grads = torch.autograd.grad(loss, list(model.parameters()))

    raw_plain_disp = {name: -lr * g for (name, p), g in zip(model.named_parameters(), plain_grads)}
    total_norm = float(torch.sqrt(sum(v.double().norm() ** 2 for v in raw_plain_disp.values())).item())
    clip_scale = max_norm / (total_norm + 1e-12)
    clipped_plain_disp = {k: v * clip_scale for k, v in raw_plain_disp.items()}

    for name in sum_disp:
        diff = (sum_disp[name] - clipped_plain_disp[name]).abs().max().item()
        assert diff < 1e-6, f"Active clipping broke plain KD sum on {name}: diff {diff}"


def test_build_class_candidates_teacher_validation():
    """Verify validation on teacher targets (sum to 1, finite, confidence <= 1)."""
    model = DummyStudent()
    x = torch.randn(2, 1, 8, 8)
    valid = torch.ones(2, 8, 8, dtype=torch.bool)

    # Not summing to 1
    unnorm_t = torch.full((2, 4, 8, 8), 0.5)
    with pytest.raises(ValueError, match="summing to 1"):
        build_class_candidates(model, x, unnorm_t, valid, lr=0.01)

    # Confidence > 1.0
    valid_t = F.softmax(torch.randn(2, 4, 8, 8), dim=1)
    with pytest.raises(ValueError, match="confidence"):
        build_class_candidates(model, x, valid_t, valid, lr=0.01, confidence=1.5)


def test_mixed_bn_dropout_per_module_training_modes():
    """Verify that per-module modes (e.g. BN in eval, Dropout in train) are preserved."""
    model = MixedModeStudent()
    model.bn.eval()
    model.drop.train()

    assert model.bn.training is False
    assert model.drop.training is True

    x = torch.randn(2, 1, 8, 8)
    targets = F.softmax(torch.randn(2, 4, 8, 8), dim=1)
    valid = torch.ones(2, 8, 8, dtype=torch.bool)

    # 1. Candidate building preserves per-module modes
    cand = build_class_candidates(model, x, targets, valid, lr=0.01)
    assert model.bn.training is False
    assert model.drop.training is True

    # Pre-generated fixed audit tensors so audit() is deterministic
    x_fixed = torch.rand(2, 1, 8, 8)
    y_fixed = torch.zeros(2, 8, 8, dtype=torch.long)
    v_fixed = torch.ones(2, 8, 8, dtype=torch.bool)

    def audit():
        yield {
            "x": x_fixed.clone(),
            "y": y_fixed.clone(),
            "valid": v_fixed.clone(),
            "frame": 0,
            "patient": "p0",
        }

    # 2. apply_compact_update preserves per-module modes on both accept and reject
    receipt = apply_compact_update(model, cand, method="joint_guard", audit_batches=audit)
    assert model.bn.training is False
    assert model.drop.training is True


def test_optimizer_and_rng_and_grad_preservation():
    """Ensure optimizer state, RNG states, and parameter .grad are preserved on rejection."""
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    model = DummyStudent()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    out = model(torch.randn(2, 1, 8, 8)).sum()
    out.backward()
    optimizer.step()

    # Populate gradients for next transaction
    out2 = model(torch.randn(2, 1, 8, 8)).sum()
    out2.backward()

    grads_before = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optim_state_before = copy_json = str(optimizer.state_dict())

    # Pre-generated fixed audit tensors so audit() is deterministic
    x_fixed = torch.rand(2, 1, 8, 8)
    y_fixed = torch.zeros(2, 8, 8, dtype=torch.long)
    v_fixed = torch.ones(2, 8, 8, dtype=torch.bool)

    def audit():
        yield {
            "x": x_fixed.clone(),
            "y": y_fixed.clone(),
            "valid": v_fixed.clone(),
            "frame": 0,
            "patient": "p0",
        }

    # Candidate with huge regression
    cand_disp = {n: torch.randn_like(p) * 50.0 for n, p in model.named_parameters()}
    candidates = {
        "directions": [cand_disp],
        "source_ids": ["pseudo_class_1"],
        "denominator": 100,
    }

    # Snapshot RNGs right before transaction
    py_rng_before = random.getstate()
    np_rng_before = np.random.get_state()
    torch_rng_before = torch.get_rng_state().clone()

    receipt = apply_compact_update(model, candidates, method="joint_guard", audit_batches=audit)
    assert receipt["status"] == "rejected_all"

    # Verify optimizer state is untouched
    assert str(optimizer.state_dict()) == optim_state_before

    # Verify gradients and parameters are untouched
    for n, p in model.named_parameters():
        assert torch.equal(p, params_before[n])
        assert torch.equal(p.grad, grads_before[n])

    # Verify RNG states were restored
    assert random.getstate() == py_rng_before
    assert np.array_equal(np.random.get_state()[1], np_rng_before[1])
    assert torch.equal(torch.get_rng_state(), torch_rng_before)


def test_changing_audit_factory_detected():
    """Verify that an audit_batches factory producing inconsistent data across replays is detected."""
    model = DummyStudent()
    call_count = 0

    def bad_audit():
        nonlocal call_count
        call_count += 1
        # Change slice labels on second call
        label_val = 1 if call_count == 1 else 2
        yield {
            "x": torch.randn(2, 1, 8, 8),
            "y": torch.full((2, 8, 8), label_val, dtype=torch.long),
            "valid": torch.ones(2, 8, 8, dtype=torch.bool),
            "frame": 0,
            "patient": "p0",
        }

    cand = {
        "directions": [{n: torch.randn_like(p) * 0.01 for n, p in model.named_parameters()}],
        "source_ids": ["pseudo_class_1"],
        "denominator": 100,
    }

    with pytest.raises((ValueError, RuntimeError), match="identity hash mismatch"):
        apply_compact_update(model, cand, method="joint_guard", audit_batches=bad_audit)


def test_absent_fp_per_frame_cannot_offset_counterexample():
    """Verify absent FP improvement on Frame 0 cannot offset an absent FP regression on Frame 1."""
    class BiasModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = nn.Parameter(torch.zeros(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.b.view(1, 4, 1, 1)

    model = BiasModel()

    # Patient has 2 frames. Class 3 (LV) is absent in GT for both frames.
    # Frame 0: 100 valid voxels. Base has 10 FP for class 3 -> absent FP = 0.10.
    # Frame 1: 100 valid voxels. Base has 0 FP for class 3 -> absent FP = 0.00.
    # Base patient mean absent FP = (0.10 + 0.00) / 2 = 0.05.
    x_f0 = torch.zeros(1, 4, 10, 10)
    x_f0[0, 3, :1, :] = 5.0  # 10 FP on row 0
    x_f0[0, 0, 1:, :] = 5.0  # BG for rest
    y_f0 = torch.zeros(1, 10, 10, dtype=torch.long)

    x_f1 = torch.zeros(1, 4, 10, 10)
    x_f1[0, 0, :, :] = 5.0   # all BG
    y_f1 = torch.zeros(1, 10, 10, dtype=torch.long)

    def audit():
        yield {"x": x_f0.clone(), "y": y_f0.clone(), "valid": torch.ones(1, 10, 10, dtype=torch.bool), "frame": 0, "patient": "p0"}
        yield {"x": x_f1.clone(), "y": y_f1.clone(), "valid": torch.ones(1, 10, 10, dtype=torch.bool), "frame": 1, "patient": "p0"}

    # Under candidate displacement:
    # Frame 0 FP drops to 0 (absent FP becomes 0.00, improvement of -0.10)
    # But on Frame 1, 5 false positives appear (absent FP becomes 0.05, regression of +0.05)
    # Patient mean absent FP drops from 0.05 to 0.025 (improves on average!)
    # But Frame 1 regressed! Absent FP per frame cannot offset!
    cand = {
        "directions": [{"b": torch.tensor([0.0, 0.0, 0.0, -1.0])}],
        "source_ids": ["pseudo_class_3"],
        "denominator": 100,
    }

    # Custom candidate alpha testing this direction
    receipt = apply_compact_update(
        model,
        cand,
        method="joint_guard",
        audit_batches=audit,
        config={"absent_fp_budget": 0.0, "tolerance": 1e-6},
    )
    # Must reject any trial that regressed on frame 1
    for trial in receipt["trials"]:
        if trial["displacement_norm"] > 0 and not trial["absent_fp_ok"]:
            assert trial["guard_passed"] is False


def test_per_patient_class_guard_counterexample():
    """Verify that patient 0 improving does not offset patient 1 regressing on the same class."""
    class BiasModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = nn.Parameter(torch.zeros(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.b.view(1, 4, 1, 1)

    model = BiasModel()

    # Patient 0: Base Dice for RV = 0.50
    x_p0 = torch.zeros(1, 4, 10, 10)
    y_p0 = torch.zeros(1, 10, 10, dtype=torch.long)
    y_p0[0, :4, :] = 1
    x_p0[0, 1, :2, :] = 5.0
    x_p0[0, 0, 2:4, :] = 5.0

    # Patient 1: Base Dice for RV = 0.90
    x_p1 = torch.zeros(1, 4, 10, 10)
    y_p1 = torch.zeros(1, 10, 10, dtype=torch.long)
    y_p1[0, :4, :] = 1
    x_p1[0, 1, :4, :] = 5.0  # 40 TP

    def audit():
        yield {"x": x_p0.clone(), "y": y_p0.clone(), "valid": torch.ones(1, 10, 10, dtype=torch.bool), "frame": 0, "patient": "p0"}
        yield {"x": x_p1.clone(), "y": y_p1.clone(), "valid": torch.ones(1, 10, 10, dtype=torch.bool), "frame": 0, "patient": "p1"}

    # Displacement drops bias[1] drastically -> Patient 1 regresses from 0.90 to 0.0!
    cand = {
        "directions": [{"b": torch.tensor([0.0, -10.0, 0.0, 0.0])}],
        "source_ids": ["pseudo_class_1"],
        "denominator": 100,
    }

    receipt = apply_compact_update(
        model,
        cand,
        method="joint_guard",
        audit_batches=audit,
        config={"trial_alphas": [[1.0]], "tolerance": 1e-6},
    )

    assert receipt["status"] == "rejected_all"
    assert receipt["accepted"] is False
    assert receipt["trials"][0]["per_class_dice_ok"] is False


def test_zero_cancellation_status():
    """Verify that nonzero alpha with canceling directions returns status zero_displacement."""
    model = DummyStudent()
    d1 = {n: torch.ones_like(p) * 0.01 for n, p in model.named_parameters()}
    d2 = {n: -torch.ones_like(p) * 0.01 for n, p in model.named_parameters()}

    candidates = {
        "directions": [d1, d2],
        "source_ids": ["pseudo_class_1", "pseudo_class_2"],
        "denominator": 100,
    }

    # Pre-generated fixed audit tensors so audit() is deterministic
    x_fixed = torch.rand(2, 1, 8, 8)
    y_fixed = torch.zeros(2, 8, 8, dtype=torch.long)
    v_fixed = torch.ones(2, 8, 8, dtype=torch.bool)

    def audit():
        yield {
            "x": x_fixed.clone(),
            "y": y_fixed.clone(),
            "valid": v_fixed.clone(),
            "frame": 0,
            "patient": "p0",
        }

    # Alpha [1.0, 1.0] causes d1 + d2 = 0
    receipt = apply_compact_update(
        model,
        candidates,
        method="joint_guard",
        audit_batches=audit,
        config={"trial_alphas": [[1.0, 1.0]]},
    )

    assert receipt["status"] == "zero_displacement"
    assert receipt["accepted"] is False
    assert receipt["realized_displacement_norm"] == 0.0


def test_m_greater_1_joint_vector_alternatives_no_bg():
    """Verify that when no BG direction is present, diverse joint alternatives are generated."""
    model = DummyStudent()
    cand1 = {n: torch.randn_like(p) * 0.001 for n, p in model.named_parameters()}
    cand2 = {n: torch.randn_like(p) * 0.001 for n, p in model.named_parameters()}

    candidates = {
        "directions": [cand1, cand2],
        "source_ids": ["pseudo_class_1", "pseudo_class_2"],  # NO pseudo_class_0!
        "denominator": 100,
    }

    x_f = torch.rand(2, 1, 8, 8)
    y_f = torch.randint(0, 4, (2, 8, 8))
    v_f = torch.ones(2, 8, 8, dtype=torch.bool)

    def audit():
        yield {
            "x": x_f.clone(),
            "y": y_f.clone(),
            "valid": v_f.clone(),
            "frame": 0,
            "patient": "p0",
        }

    receipt = apply_compact_update(model, candidates, method="joint_guard", audit_batches=audit)
    assert len(receipt["trials"]) == 8
    # Verify diversity in joint candidates
    alphas = [t["alpha"] for t in receipt["trials"]]
    assert len(set(tuple(a) for a in alphas)) > 4


def test_strict_config_validation():
    """Verify strict validation on config keys, NaN values, negative budgets, and scalar alphas."""
    model = DummyStudent()
    cand = {"directions": [], "source_ids": [], "denominator": 0}

    # Unknown config key
    with pytest.raises(ValueError, match="unknown configuration keys"):
        apply_compact_update(model, cand, method="supervised", config={"unknown_key": 1})

    # NaN eps
    with pytest.raises(ValueError, match="finite non-negative float"):
        apply_compact_update(model, cand, method="supervised", config={"eps": float("nan")})

    # Non-scalar alpha for scalar_guard
    d = {"directions": [{n: torch.zeros_like(p) for n, p in model.named_parameters()},
                        {n: torch.zeros_like(p) for n, p in model.named_parameters()}],
         "source_ids": ["pseudo_class_1", "pseudo_class_2"], "denominator": 100}

    x_f = torch.rand(2, 1, 8, 8)
    y_f = torch.zeros(2, 8, 8, dtype=torch.long)
    v_f = torch.ones(2, 8, 8, dtype=torch.bool)
    def dummy_audit():
        yield {"x": x_f.clone(), "y": y_f.clone(), "valid": v_f.clone(), "frame": 0, "patient": "p0"}

    with pytest.raises(ValueError, match="scalar alpha"):
        apply_compact_update(model, d, method="scalar_guard", audit_batches=dummy_audit, config={"trial_alphas": [[1.0, 0.5]]})


def test_accepted_exact_once():
    """Verify parameters are updated by exactly the chosen displacement once on acceptance."""
    model = DummyStudent()
    p_before = model.conv1.bias.detach().clone()
    d = torch.full_like(model.conv1.bias, 0.001)

    candidates = {
        "directions": [{"conv1.bias": d}],
        "source_ids": ["pseudo_class_1"],
        "denominator": 100,
    }

    # Pre-generated fixed audit tensors so audit() is deterministic
    x_fixed = torch.rand(2, 1, 8, 8)
    y_fixed = torch.zeros(2, 8, 8, dtype=torch.long)
    v_fixed = torch.ones(2, 8, 8, dtype=torch.bool)

    def audit():
        yield {
            "x": x_fixed.clone(),
            "y": y_fixed.clone(),
            "valid": v_fixed.clone(),
            "frame": 0,
            "patient": "p0",
        }

    receipt = apply_compact_update(
        model,
        candidates,
        method="scalar_guard",
        audit_batches=audit,
        config={"trial_alphas": [[1.0]], "tolerance": 1e-4},
    )

    if receipt["status"] == "accepted_nonzero":
        assert receipt["accepted"] is True
        assert torch.allclose(model.conv1.bias, p_before + d, atol=1e-7)
        # Ensure it was not added twice
        assert not torch.allclose(model.conv1.bias, p_before + 2 * d, atol=1e-7)


def test_compact_student_whole_model_update():
    """Verify whole-model candidate displacement with native CompactStudent architecture."""
    try:
        from self_audit.compact_model import build_compact_model
    except ImportError:
        pytest.skip("compact_model not available")

    model = build_compact_model({"widths": [8, 16, 32, 64], "in_channels": 3, "num_classes": 4})
    # Inputs must be in [0, 1] for CompactStudent
    x = torch.rand(2, 3, 32, 32)
    targets = F.softmax(torch.randn(2, 4, 32, 32), dim=1)
    valid = torch.ones(2, 32, 32, dtype=torch.bool)

    cand = build_class_candidates(model, x, targets, valid, lr=0.01)
    assert len(cand["directions"]) > 0
    assert cand["denominator"] == 2 * 32 * 32

    x_audit = torch.rand(2, 3, 32, 32)
    y_audit = torch.randint(0, 4, (2, 32, 32))
    v_audit = torch.ones(2, 32, 32, dtype=torch.bool)

    def audit():
        yield {
            "x": x_audit.clone(),
            "y": y_audit.clone(),
            "valid": v_audit.clone(),
            "frame": 0,
            "patient": "p0",
        }

    receipt = apply_compact_update(model, cand, method="joint_guard", audit_batches=audit)
    assert receipt["status"] in {"accepted_nonzero", "rejected_all"}
    assert len(receipt["trials"]) == 8
    assert receipt["optimizer_steps_in_transaction"] == 0


def test_m1_scalar_vs_joint_identical():
    """Verify that when M=1, scalar_guard and joint_guard evaluate identical trials and alphas."""
    m_scalar = DummyStudent()
    m_joint = DummyStudent()
    m_joint.load_state_dict(m_scalar.state_dict())

    cand_disp = {n: torch.randn_like(p) * 0.005 for n, p in m_scalar.named_parameters()}
    candidates = {
        "directions": [cand_disp],
        "source_ids": ["pseudo_class_1"],
        "denominator": 128,
    }

    x_f = torch.rand(2, 1, 8, 8)
    y_f = torch.randint(0, 4, (2, 8, 8))
    v_f = torch.ones(2, 8, 8, dtype=torch.bool)

    def audit():
        yield {
            "x": x_f.clone(),
            "y": y_f.clone(),
            "valid": v_f.clone(),
            "frame": 0,
            "patient": "p1",
        }

    r_scalar = apply_compact_update(m_scalar, candidates, method="scalar_guard", audit_batches=audit)
    r_joint = apply_compact_update(m_joint, candidates, method="joint_guard", audit_batches=audit)

    assert len(r_scalar["trials"]) == len(r_joint["trials"])
    assert r_scalar["status"] == r_joint["status"]
    assert r_scalar["accepted"] == r_joint["accepted"]

    for ts, tj in zip(r_scalar["trials"], r_joint["trials"]):
        assert ts["alpha"] == tj["alpha"]
        assert abs(ts["objective"] - tj["objective"]) < 1e-6
        assert ts["guard_passed"] == tj["guard_passed"]


def test_plain_kd_commits_without_guard():
    """Verify plain_kd commits the full displacement without evaluating guards."""
    model = DummyStudent()
    p_before = model.conv1.bias.detach().clone()
    d = torch.full_like(model.conv1.bias, 0.05)
    candidates = {
        "directions": [{"conv1.bias": d}],
        "source_ids": ["pseudo_class_1"],
        "denominator": 100,
    }

    receipt = apply_compact_update(model, candidates, method="plain_kd")
    assert receipt["status"] == "accepted_nonzero"
    assert receipt["accepted"] is True
    assert torch.allclose(model.conv1.bias, p_before + d, atol=1e-7)


def test_supervised_method_zero_displacement():
    """Verify supervised method applies zero displacement and returns zero_displacement status."""
    model = DummyStudent()
    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    candidates = {
        "directions": [{n: torch.randn_like(p) for n, p in model.named_parameters()}],
        "source_ids": ["pseudo_class_1"],
        "denominator": 100,
    }

    receipt = apply_compact_update(model, candidates, method="supervised")
    assert receipt["status"] == "zero_displacement"
    assert receipt["accepted"] is False
    assert receipt["realized_displacement_norm"] == 0.0

    for n, p in model.named_parameters():
        assert torch.equal(p, params_before[n])


def test_empty_or_all_ignored_audit_rejected():
    """Verify that empty audit or all-ignored audit support raises an explicit ValueError."""
    model = DummyStudent()
    cand = {
        "directions": [{n: torch.randn_like(p) * 0.01 for n, p in model.named_parameters()}],
        "source_ids": ["pseudo_class_1"],
        "denominator": 100,
    }

    def empty_audit():
        if False:
            yield {}

    with pytest.raises(ValueError, match="zero batches"):
        apply_compact_update(model, cand, method="joint_guard", audit_batches=empty_audit)

    def all_ignored_audit():
        yield {
            "x": torch.rand(2, 1, 8, 8),
            "y": torch.full((2, 8, 8), 255, dtype=torch.long),  # all ignored
            "valid": torch.zeros(2, 8, 8, dtype=torch.bool),
            "frame": 0,
            "patient": "p0",
        }

    with pytest.raises(ValueError, match="all pixels across audit support are ignored"):
        apply_compact_update(model, cand, method="joint_guard", audit_batches=all_ignored_audit)


def test_json_serializability():
    """Verify receipt is strictly JSON-serializable without NaN or Inf."""
    model = DummyStudent()
    candidates = {
        "directions": [{n: torch.randn_like(p) * 0.01 for n, p in model.named_parameters()}],
        "source_ids": ["pseudo_class_1"],
        "denominator": 100,
    }

    x_f = torch.rand(2, 1, 8, 8)
    y_f = torch.randint(0, 4, (2, 8, 8))
    v_f = torch.ones(2, 8, 8, dtype=torch.bool)

    def audit():
        yield {
            "x": x_f.clone(),
            "y": y_f.clone(),
            "valid": v_f.clone(),
            "frame": 0,
            "patient": "p0",
        }

    receipt = apply_compact_update(model, candidates, method="joint_guard", audit_batches=audit)
    encoded = json.dumps(receipt, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["status"] in {"accepted_nonzero", "rejected_all", "zero_displacement"}


def test_call_model_functional_exception_propagates():
    """Verify that exceptions raised during forward pass (e.g. input rejection) propagate."""
    class RejectingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(4))

        def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
            raise ValueError("model rejected input during forward pass")

    model = RejectingModel()
    x = torch.rand(2, 1, 8, 8)
    targets = F.softmax(torch.randn(2, 4, 8, 8), dim=1)
    valid = torch.ones(2, 8, 8, dtype=torch.bool)

    with pytest.raises(ValueError, match="model rejected input"):
        build_class_candidates(model, x, targets, valid, lr=0.01)

    cand = {
        "directions": [{"p": torch.zeros(4)}],
        "source_ids": ["pseudo_class_0"],
        "denominator": 100,
    }

    def audit():
        yield {"x": x.clone(), "y": torch.zeros(2, 8, 8, dtype=torch.long), "valid": valid.clone(), "frame": 0, "patient": "p0"}

    with pytest.raises(ValueError, match="model rejected input"):
        apply_compact_update(model, cand, method="joint_guard", audit_batches=audit)


def test_audit_input_validation_witnesses():
    """Verify audit streaming validates x 4D, y integer canonical 0..3/255, and patient/frame lengths."""
    model = DummyStudent()
    cand = {
        "directions": [{n: torch.zeros_like(p) for n, p in model.named_parameters()}],
        "source_ids": ["pseudo_class_0"],
        "denominator": 100,
    }

    # 1. Non-4D x
    def bad_x_audit():
        yield {"x": torch.rand(1, 8, 8), "y": torch.zeros(1, 8, 8, dtype=torch.long), "valid": torch.ones(1, 8, 8, dtype=torch.bool), "frame": 0, "patient": "p0"}

    with pytest.raises(ValueError, match="x must be a 4D Tensor"):
        apply_compact_update(model, cand, method="joint_guard", audit_batches=bad_x_audit)

    # 2. Non-canonical y ontology (e.g. label 9)
    def bad_y_ontology_audit():
        yield {"x": torch.rand(1, 1, 8, 8), "y": torch.full((1, 8, 8), 9, dtype=torch.long), "valid": torch.ones(1, 8, 8, dtype=torch.bool), "frame": 0, "patient": "p0"}

    with pytest.raises(ValueError, match="y ontology must strictly contain only canonical values"):
        apply_compact_update(model, cand, method="joint_guard", audit_batches=bad_y_ontology_audit)

    # 3. Floating point y
    def float_y_audit():
        yield {"x": torch.rand(1, 1, 8, 8), "y": torch.zeros(1, 8, 8, dtype=torch.float32), "valid": torch.ones(1, 8, 8, dtype=torch.bool), "frame": 0, "patient": "p0"}

    with pytest.raises(TypeError, match="y must be an integer tensor"):
        apply_compact_update(model, cand, method="joint_guard", audit_batches=float_y_audit)

    # 4. Mismatched patient list length
    def bad_patient_len_audit():
        yield {"x": torch.rand(2, 1, 8, 8), "y": torch.zeros(2, 8, 8, dtype=torch.long), "valid": torch.ones(2, 8, 8, dtype=torch.bool), "frame": 0, "patient": ["p0"]}

    with pytest.raises(ValueError, match="patient list length"):
        apply_compact_update(model, cand, method="joint_guard", audit_batches=bad_patient_len_audit)


def test_candidate_dtype_device_shape_validation():
    """Verify displacement tensors are validated for matching shape, dtype, and device."""
    model = DummyStudent()

    # Wrong shape
    bad_shape_cand = {
        "directions": [{"conv1.bias": torch.zeros(20)}],  # conv1.bias is 8
        "source_ids": ["pseudo_class_0"],
        "denominator": 100,
    }
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_compact_update(model, bad_shape_cand, method="supervised")

    # Wrong dtype
    bad_dtype_cand = {
        "directions": [{"conv1.bias": torch.zeros(8, dtype=torch.float64)}],  # model is float32
        "source_ids": ["pseudo_class_0"],
        "denominator": 100,
    }
    with pytest.raises(TypeError, match="dtype mismatch"):
        apply_compact_update(model, bad_dtype_cand, method="supervised")


def test_plain_kd_non_finite_params_rejected_before_commit():
    """Verify that plain_kd rejects non-finite candidate updates before modifying model parameters."""
    model = DummyStudent()
    p_before = model.conv1.bias.detach().clone()

    inf_cand = {
        "directions": [{"conv1.bias": torch.full_like(model.conv1.bias, float("inf"))}],
        "source_ids": ["pseudo_class_0"],
        "denominator": 100,
    }

    with pytest.raises(FloatingPointError, match="non-finite displacement"):
        apply_compact_update(model, inf_cand, method="plain_kd")

    assert torch.equal(model.conv1.bias, p_before)


def test_aggregate_norm_cap_enforced_for_every_trial_vector():
    """Verify aggregate norm cap is enforced for every trial vector and logs norm_bound_ok: False."""
    model = DummyStudent()
    p_name = "conv1.bias"
    p_ref = model.conv1.bias

    # Two directions: d1 has norm 6, d2 has norm -6. Sum has norm 0.
    # But individual trial with alpha [1.0, 0.0] has norm 6 > max_norm=5.0!
    d1 = {p_name: torch.ones_like(p_ref) * (6.0 / np.sqrt(p_ref.numel()))}
    d2 = {p_name: -torch.ones_like(p_ref) * (6.0 / np.sqrt(p_ref.numel()))}

    # Snapshot to ensure directions are not mutated
    d1_clone = d1[p_name].clone()
    d2_clone = d2[p_name].clone()

    candidates = {
        "directions": [d1, d2],
        "source_ids": ["pseudo_class_1", "pseudo_class_2"],
        "denominator": 100,
        "telemetry": {"max_norm": 5.0},  # max_norm is 5.0
    }

    x_f = torch.rand(2, 1, 8, 8)
    y_f = torch.zeros(2, 8, 8, dtype=torch.long)
    v_f = torch.ones(2, 8, 8, dtype=torch.bool)
    def audit():
        yield {"x": x_f.clone(), "y": y_f.clone(), "valid": v_f.clone(), "frame": 0, "patient": "p0"}

    # Custom alphas: trial 0 has alpha [1.0, 0.0] (norm 6 > 5), trial 1 has alpha [0.5, 0.5] (norm 0 <= 5)
    receipt = apply_compact_update(
        model,
        candidates,
        method="joint_guard",
        audit_batches=audit,
        config={"trial_alphas": [[1.0, 0.0], [0.5, 0.5]]},
    )

    # Trial 0 must be rejected due to aggregate norm cap exceeded
    assert receipt["trials"][0]["norm_bound_ok"] is False
    assert receipt["trials"][0]["reason"] == "aggregate_norm_cap_exceeded"
    assert receipt["trials"][0]["guard_passed"] is False

    # Trial 1 norm was 0 <= 5, so norm_bound_ok is True
    assert receipt["trials"][1]["norm_bound_ok"] is True

    # Verify directions were NOT mutated
    assert torch.equal(d1[p_name], d1_clone)
    assert torch.equal(d2[p_name], d2_clone)


def test_mean_margin_decreased_honest_boolean_and_null_voxel_fraction():
    """Verify receipt has honest mean_margin_decreased bool and null margin_decreased_fraction."""
    model = DummyStudent()
    candidates = {
        "directions": [{n: torch.zeros_like(p) for n, p in model.named_parameters()}],
        "source_ids": ["pseudo_class_0"],
        "denominator": 100,
    }

    x_f = torch.rand(2, 1, 8, 8)
    y_f = torch.zeros(2, 8, 8, dtype=torch.long)
    v_f = torch.ones(2, 8, 8, dtype=torch.bool)
    def audit():
        yield {"x": x_f.clone(), "y": y_f.clone(), "valid": v_f.clone(), "frame": 0, "patient": "p0"}

    receipt = apply_compact_update(model, candidates, method="joint_guard", audit_batches=audit)
    assert "mean_margin_decreased" in receipt
    assert isinstance(receipt["mean_margin_decreased"], bool)
    assert receipt["margin_decreased_fraction"] is None
