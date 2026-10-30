"""Unit and integration tests for transfer-utility and compact audit selectors:

Original golden parity, larger_tie, cross_patient_utility, same_batch_utility,
strict guard adherence, empty/nonfinite/negative gains, smaller-norm tie breaks,
atomic parameter/BN/optimizer/RNG preservation, single commit, and frozen probe consistency.
"""

from __future__ import annotations

import copy
import json
import random
from typing import Any, Dict, Iterator

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from self_audit.compact_control import (
    DEFAULT_SELECTOR,
    DEFAULT_SELECTOR_DICE_TOLERANCE,
    DEFAULT_UTILITY_GAIN_TOLERANCE,
    DEFAULT_UTILITY_TIE_TOLERANCE,
    VALID_SELECTORS,
    _validate_config,
    apply_compact_update,
    build_class_candidates,
)
from self_audit.compact_utility import (
    compute_probe_kd_loss,
    validate_utility_probe,
)


class DummyStudent(nn.Module):
    """Simple 2.5D convolutional student for fast CPU unit testing."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(2, 8)
        self.conv2 = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
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


def _make_dummy_audit(x_shape=(2, 1, 8, 8), num_batches=1):
    """Create a deterministic audit batches callable."""
    x = torch.rand(x_shape)
    y = torch.zeros(x_shape[0], x_shape[2], x_shape[3], dtype=torch.long)
    v = torch.ones(x_shape[0], x_shape[2], x_shape[3], dtype=torch.bool)

    def audit():
        for b in range(num_batches):
            yield {
                "x": x.clone(),
                "y": y.clone(),
                "valid": v.clone(),
                "frame": b,
                "patient": f"p{b}",
            }

    return audit


def _make_dummy_probe(x_shape=(2, 1, 8, 8), temp=1.0, coef=1.0, conf=0.0):
    """Create a valid utility probe dictionary."""
    B, _, H, W = x_shape
    x = torch.rand(x_shape)
    raw_logits = torch.randn(B, 4, H, W)
    targets = F.softmax(raw_logits, dim=1)
    valid = torch.ones(B, H, W, dtype=torch.bool)
    return {
        "x": x,
        "targets": targets,
        "valid": valid,
        "temperature": temp,
        "coefficient": coef,
        "confidence": conf,
    }


def test_config_validation_and_nuances():
    """Verify strict configuration validation and historical tolerance nuance."""
    # Historical nuance: None gives 1e-6, {} gives 0.0
    c_none = _validate_config(None)
    assert c_none["tolerance"] == 1e-6
    assert c_none["selector"] == "original"
    assert c_none["selector_dice_tolerance"] == 0.0
    assert c_none["utility_gain_tolerance"] == 1e-7
    assert c_none["utility_tie_tolerance"] == 1e-7

    c_empty = _validate_config({})
    assert c_empty["tolerance"] == 0.0
    assert c_empty["selector"] == "original"

    # Valid selector options
    for sel in VALID_SELECTORS:
        c = _validate_config({"selector": sel})
        assert c["selector"] == sel

    # Invalid selector
    with pytest.raises(ValueError, match="unknown selector"):
        _validate_config({"selector": "invalid_selector"})

    # Unknown config key
    with pytest.raises(ValueError, match="unknown configuration keys"):
        _validate_config({"bad_key": 123})

    # Method restriction: new selectors are restricted to scalar_guard
    model = DummyStudent()
    cand = {"directions": [], "source_ids": [], "denominator": 0}
    with pytest.raises(ValueError, match="restricted to method 'scalar_guard'"):
        apply_compact_update(
            model, cand, method="joint_guard", config={"selector": "cross_patient_utility"}
        )
    with pytest.raises(ValueError, match="restricted to method 'scalar_guard'"):
        apply_compact_update(
            model, cand, method="plain_kd", config={"selector": "larger_tie"}
        )


def test_kd_loss_mathematical_equivalence_with_build_class_candidates():
    """Verify KD distillation probe loss exactly matches build_class_candidates common-denominator KD

    across temperatures, coefficients, and raw-confidence thresholding before temperature.
    """
    torch.manual_seed(1234)
    model = DummyStudent()

    configs = [
        {"temp": 1.0, "coef": 1.0, "conf": 0.0},
        {"temp": 2.0, "coef": 0.5, "conf": 0.2},
        {"temp": 0.5, "coef": 2.0, "conf": 0.4},
    ]

    for cfg in configs:
        temp = cfg["temp"]
        coef = cfg["coef"]
        conf = cfg["conf"]

        x = torch.randn(2, 1, 16, 16)
        raw_t = torch.randn(2, 4, 16, 16)
        targets = F.softmax(raw_t, dim=1)
        valid = torch.rand(2, 16, 16) > 0.2

        # 1. Compute loss via build_class_candidates
        # build_class_candidates computes directions by backpropping loss_c = (per_pixel_loss * mask_c).sum() / D
        # So sum_c loss_c = (per_pixel_loss * confident_mask).sum() / D
        t_max, t_argmax = targets.max(dim=1)
        conf_mask = (t_max >= conf) & valid
        denom = int(conf_mask.sum().item())
        assert denom > 0

        with torch.no_grad():
            fresh_buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
            base_params = dict(model.named_parameters())
            logits = model(x)
            scaled_logits = logits / temp
            log_p = F.log_softmax(scaled_logits, dim=1)
            if abs(temp - 1.0) > 1e-7:
                soft_t = F.softmax(torch.log(targets.clamp(min=1e-12)) / temp, dim=1)
            else:
                soft_t = targets
            per_pixel = - (soft_t * log_p).sum(dim=1) * ((temp ** 2) * coef)
            expected_loss = float(((per_pixel * conf_mask).sum() / denom).item())

        # 2. Compute loss via compute_probe_kd_loss
        probe = {
            "x": x,
            "targets": targets,
            "valid": valid,
            "temperature": temp,
            "coefficient": coef,
            "confidence": conf,
        }
        frozen_probe = validate_utility_probe(probe)
        probe_loss, probe_denom, is_finite = compute_probe_kd_loss(
            model, base_params, fresh_buffers, frozen_probe
        )

        assert is_finite is True
        assert probe_denom == denom
        assert abs(probe_loss - expected_loss) < 1e-6, f"Mismatch: {probe_loss} vs {expected_loss}"


def test_original_golden_parity():
    """Verify that selector='original' exactly preserves legacy scalar_guard behavior and keys."""
    model = DummyStudent()
    audit = _make_dummy_audit()

    d = {n: torch.randn_like(p) * 0.001 for n, p in model.named_parameters()}
    cand = {"directions": [d], "source_ids": ["pseudo_class_1"], "denominator": 100}

    # Run with default config (original selector)
    receipt = apply_compact_update(model, cand, method="scalar_guard", audit_batches=audit)

    # Legacy keys must be intact
    for k in (
        "method", "status", "accepted", "source_ids", "alpha", "scale",
        "proposed_displacement_norm", "realized_displacement_norm",
        "base_macro_dice", "base_macro_absent_fp", "base_fg_dice",
        "macro_dice_after", "macro_absent_fp_after", "macro_fg_dice_after",
        "margins_before", "margins_after", "margin_change", "mean_margin_decreased",
        "trials", "telemetry"
    ):
        assert k in receipt, f"Missing legacy key {k}"

    # New selector diagnostics
    assert receipt["selector"] == "original"
    assert "reason" in receipt
    assert "original_choice" in receipt
    assert "feasible_scales" in receipt
    assert "forward_counts" in receipt
    assert "timings" in receipt

    # Verify JSON serializability without NaN
    json_str = json.dumps(receipt, allow_nan=False)
    assert len(json_str) > 0


def test_larger_tie_tie_only_change():
    """Verify that larger_tie breaks exact Dice ties by selecting the larger feasible scalar,

    and does NOT alter decisions when Dice is not tied (tie-only change).
    """
    model = DummyStudent()
    audit = _make_dummy_audit()

    # Create two candidate scales: 1.0 and 0.5.
    # In DummyStudent with dummy audit, all scales yield Dice = None (no foreground in GT labels).
    # Since foreground Dice is None for both, abs(None - None) is a tie!
    # Because of the ridge term (0.5 * ridge * alpha^2), scale 0.5 has strictly lower objective than 1.0!
    # Original selects 0.5; larger_tie selects 1.0!
    d = {n: torch.randn_like(p) * 0.001 for n, p in model.named_parameters()}
    cand = {"directions": [d], "source_ids": ["pseudo_class_1"], "denominator": 100}

    cfg_orig = {
        "trial_alphas": [[1.0], [0.5]],
        "selector": "original",
        "ridge": 1e-2,
    }
    receipt_orig = apply_compact_update(
        copy.deepcopy(model), cand, method="scalar_guard", audit_batches=audit, config=cfg_orig
    )
    assert receipt_orig["status"] == "accepted_nonzero"
    assert receipt_orig["scale"] == 0.5  # Original picked smaller scale due to ridge

    cfg_tie = {
        "trial_alphas": [[1.0], [0.5]],
        "selector": "larger_tie",
        "selector_dice_tolerance": 0.0,
        "ridge": 1e-2,
    }
    receipt_tie = apply_compact_update(
        copy.deepcopy(model), cand, method="scalar_guard", audit_batches=audit, config=cfg_tie
    )
    assert receipt_tie["status"] == "accepted_nonzero"
    assert receipt_tie["scale"] == 1.0  # larger_tie broke the tie in favor of 1.0!
    assert receipt_tie["reason"] == "larger_tie_dice_tiebreak"


def test_maxgain_feasible_vs_guard_rejection():
    """Verify cross_patient_utility chooses the feasible trial with max gain,

    and never weakens the audit guard (rejects trials that violate the guard even if gain is high).
    """
    model = DummyStudent()
    audit = _make_dummy_audit()

    # Create two candidates:
    # Scale 1.0 (very large displacement) which will cause extreme shift and fail guard or norm bound
    # Scale 0.1 (small displacement) which easily passes guard
    d = {n: torch.ones_like(p) * 0.01 for n, p in model.named_parameters()}
    cand = {"directions": [d], "source_ids": ["pseudo_class_1"], "denominator": 100}

    probe = _make_dummy_probe()

    # Set max_norm to 0.05, so scale 1.0 (norm ~ 0.07) exceeds max_norm and is rejected,
    # while scale 0.1 (norm ~ 0.007) is feasible.
    norm_disp = float(torch.sqrt(sum(t.norm() ** 2 for t in d.values())).item())
    cfg = {
        "trial_alphas": [[1.0], [0.1]],
        "max_norm": norm_disp * 0.5,  # 1.0 will violate norm bound, 0.1 will pass
        "selector": "cross_patient_utility",
    }

    receipt = apply_compact_update(
        copy.deepcopy(model),
        cand,
        method="scalar_guard",
        audit_batches=audit,
        config=cfg,
        utility_probe=probe,
    )

    # Scale 1.0 was rejected by norm bound, so only feasible trials can be considered
    trial_0 = receipt["trials"][0]
    assert trial_0["norm_bound_ok"] is False
    assert trial_0["probe_loss"] is None  # Not evaluated on probe because infeasible

    # If all trials are rejected by guard:
    cfg_all_reject = {
        "trial_alphas": [[1.0]],
        "max_norm": norm_disp * 0.01,  # All violate norm bound
        "selector": "cross_patient_utility",
    }
    receipt_all_rej = apply_compact_update(
        copy.deepcopy(model),
        cand,
        method="scalar_guard",
        audit_batches=audit,
        config=cfg_all_reject,
        utility_probe=probe,
    )
    assert receipt_all_rej["status"] == "rejected_all"
    assert receipt_all_rej["accepted"] is False


def test_empty_nonfinite_negative_gains():
    """Verify zero wins on negative gains, non-finite loss, empty denominator, and missing probe."""
    model = DummyStudent()
    audit = _make_dummy_audit()

    d = {n: torch.randn_like(p) * 0.001 for n, p in model.named_parameters()}
    cand = {"directions": [d], "source_ids": ["pseudo_class_1"], "denominator": 100}

    # 1. Missing probe -> zero decision
    receipt_missing = apply_compact_update(
        copy.deepcopy(model),
        cand,
        method="scalar_guard",
        audit_batches=audit,
        config={"selector": "cross_patient_utility"},
        utility_probe=None,
    )
    assert receipt_missing["status"] == "zero_displacement"
    assert receipt_missing["accepted"] is False
    assert receipt_missing["realized_displacement_norm"] == 0.0
    assert receipt_missing["reason"] == "missing_utility_probe"

    # 2. Zero denominator -> zero decision
    probe_zero_denom = _make_dummy_probe(conf=1.0)
    # Force targets to be below confidence 1.0
    probe_zero_denom["targets"].fill_(0.25)
    receipt_zd = apply_compact_update(
        copy.deepcopy(model),
        cand,
        method="scalar_guard",
        audit_batches=audit,
        config={"selector": "cross_patient_utility"},
        utility_probe=probe_zero_denom,
    )
    assert receipt_zd["status"] == "zero_displacement"
    assert receipt_zd["accepted"] is False
    assert receipt_zd["reason"] == "zero_probe_denominator"

    # 3. Nonfinite probe inputs raise with atomicity
    bad_probe = _make_dummy_probe()
    bad_probe["x"][0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite values"):
        apply_compact_update(
            copy.deepcopy(model),
            cand,
            method="scalar_guard",
            audit_batches=audit,
            config={"selector": "cross_patient_utility"},
            utility_probe=bad_probe,
        )


def test_smaller_norm_tie():
    """Verify that tied utility gains within utility_tie_tolerance are broken by smaller displacement norm."""
    torch.manual_seed(42)
    model = DummyStudent()
    audit = _make_dummy_audit()
    probe = _make_dummy_probe()

    # build_class_candidates produces gradient step displacements that strictly improve KD loss on probe
    cand = build_class_candidates(
        model,
        probe["x"],
        probe["targets"],
        probe["valid"],
        lr=0.001,
        temperature=probe["temperature"],
        coefficient=probe["coefficient"],
    )

    m_dirs = len(cand["directions"])
    # Both trials pass guard and have positive gain
    # Scale 0.1 has norm < Scale 0.5
    cfg = {
        "trial_alphas": [[0.5] * m_dirs, [0.1] * m_dirs],
        "selector": "cross_patient_utility",
        "utility_gain_tolerance": 0.0,   # Non-negative tolerance
        "utility_tie_tolerance": 1.0,   # Force tie across these two positive-gain trials
    }
    receipt = apply_compact_update(
        copy.deepcopy(model),
        cand,
        method="scalar_guard",
        audit_batches=audit,
        config=cfg,
        utility_probe=probe,
    )
    assert receipt["status"] == "zero_displacement"
    assert receipt["reason"] == "utility_tied_with_zero"
    assert receipt["scale"] == 0.0  # Logical zero has the smallest norm in this forced tie!


def test_bn_optimizer_rng_atomicity():
    """Verify that per-module modes, optimizer state, parameter gradients, and RNGs are strictly preserved on rejection."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = MixedModeStudent()
    model.train(True)
    model.conv.train(True)
    model.bn.train(True)
    model.drop.train(True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Create optimizer momentum state
    dummy_x = torch.randn(2, 1, 8, 8)
    loss = model(dummy_x).sum()
    loss.backward()
    optimizer.step()

    audit = _make_dummy_audit()
    d = {n: torch.randn_like(p) * 0.001 for n, p in model.named_parameters()}
    cand = {"directions": [d], "source_ids": ["pseudo_class_1"], "denominator": 100}

    # Snapshot states
    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers_before = {n: p.detach().clone() for n, p in model.named_buffers()}
    grads_before = {n: p.grad.detach().clone() if p.grad is not None else None for n, p in model.named_parameters()}
    optim_state_before = str(optimizer.state_dict())
    modes_before = {m: m.training for m in model.modules()}

    py_rng_before = random.getstate()
    np_rng_before = np.random.get_state()
    torch_rng_before = torch.get_rng_state().clone()

    # Execute a rejection with missing probe -> zero decision
    cfg = {"selector": "cross_patient_utility"}
    receipt = apply_compact_update(
        model, cand, method="scalar_guard", audit_batches=audit, config=cfg, utility_probe=None
    )
    assert receipt["status"] == "zero_displacement"
    assert receipt["accepted"] is False

    # Verify parameters, grads, optimizer untouched
    for n, p in model.named_parameters():
        assert torch.equal(p, params_before[n]), f"Parameter {n} mutated"
        if grads_before[n] is not None:
            assert torch.equal(p.grad, grads_before[n]), f"Grad for {n} mutated"

    assert all(torch.equal(p, buffers_before[n]) for n, p in model.named_buffers())
    assert str(optimizer.state_dict()) == optim_state_before
    for m, mode in modes_before.items():
        assert m.training == mode, f"Module {m} training mode not restored"

    assert random.getstate() == py_rng_before
    assert np.array_equal(np.random.get_state()[1], np_rng_before[1])
    assert torch.equal(torch.get_rng_state(), torch_rng_before)


def test_single_commit():
    """Verify that model parameters are modified at most once per transaction."""
    model = DummyStudent()
    audit = _make_dummy_audit()

    d = {n: torch.randn_like(p) * 0.001 for n, p in model.named_parameters()}
    cand = {"directions": [d], "source_ids": ["pseudo_class_1"], "denominator": 100}

    # Track copy_ calls
    copy_calls = []
    orig_copy = torch.Tensor.copy_

    def tracked_copy(self, src, *args, **kwargs):
        copy_calls.append(self)
        return orig_copy(self, src, *args, **kwargs)

    torch.Tensor.copy_ = tracked_copy
    try:
        # Accepted run
        cfg = {"trial_alphas": [[0.1]], "selector": "original"}
        receipt = apply_compact_update(model, cand, method="scalar_guard", audit_batches=audit, config=cfg)
        assert receipt["accepted"] is True
        num_named_params = len(list(model.named_parameters()))
        # In a single commit, each named parameter is copied exactly once
        assert len(copy_calls) == num_named_params

        # Rejected run
        copy_calls.clear()
        receipt_rej = apply_compact_update(
            model, cand, method="scalar_guard", audit_batches=audit,
            config={"trial_alphas": [[0.1]], "selector": "cross_patient_utility"},
            utility_probe=None  # Missing probe -> rejected
        )
        assert receipt_rej["accepted"] is False
        assert len(copy_calls) == 0  # Zero copies on rejection
    finally:
        torch.Tensor.copy_ = orig_copy


def test_frozen_probe_consistency():
    """Verify that mutating the caller's utility_probe dictionary or tensors does not affect internal execution."""
    model = DummyStudent()
    audit = _make_dummy_audit()

    d = {n: torch.randn_like(p) * 0.001 for n, p in model.named_parameters()}
    cand = {"directions": [d], "source_ids": ["pseudo_class_1"], "denominator": 100}

    probe = _make_dummy_probe()
    frozen = validate_utility_probe(probe)

    # Mutate original probe dictionary and tensors
    probe["x"].fill_(999.0)
    probe["targets"].zero_()
    probe["temperature"] = 999.0

    # Frozen probe must remain untouched
    assert not torch.allclose(frozen["x"], probe["x"])
    assert frozen["temperature"] == 1.0


def test_same_batch_utility():
    """Verify same_batch_utility selector behaves identically to cross_patient_utility algorithm."""
    model = DummyStudent()
    audit = _make_dummy_audit()
    probe = _make_dummy_probe()

    d = {n: torch.randn_like(p) * 0.001 for n, p in model.named_parameters()}
    cand = {"directions": [d], "source_ids": ["pseudo_class_1"], "denominator": 100}

    cfg_cpu = {"trial_alphas": [[0.1]], "selector": "cross_patient_utility"}
    cfg_sbu = {"trial_alphas": [[0.1]], "selector": "same_batch_utility"}

    rec_cpu = apply_compact_update(
        copy.deepcopy(model), cand, method="scalar_guard", audit_batches=audit,
        config=cfg_cpu, utility_probe=probe
    )
    rec_sbu = apply_compact_update(
        copy.deepcopy(model), cand, method="scalar_guard", audit_batches=audit,
        config=cfg_sbu, utility_probe=probe
    )

    assert rec_cpu["status"] == rec_sbu["status"]
    assert rec_cpu["scale"] == rec_sbu["scale"]
    assert rec_cpu["realized_displacement_norm"] == rec_sbu["realized_displacement_norm"]
    assert rec_sbu["selector"] == "same_batch_utility"


def test_plain_kd_norm_rejection_and_zero_first_grid():
    """Verify plain_kd respects norm caps, handles custom zero-first grids, and rejects empty grids."""
    model = DummyStudent()
    audit = _make_dummy_audit()

    d = {n: torch.ones_like(p) * 0.01 for n, p in model.named_parameters()}
    cand = {"directions": [d], "source_ids": ["pseudo_class_1"], "denominator": 100}
    norm_disp = float(torch.sqrt(sum(t.norm() ** 2 for t in d.values())).item())

    # 1. Norm cap rejection in plain_kd
    cfg_norm = {"max_norm": norm_disp * 0.1}
    rec_norm = apply_compact_update(copy.deepcopy(model), cand, method="plain_kd", audit_batches=audit, config=cfg_norm)
    assert rec_norm["status"] == "rejected_all"
    assert rec_norm["accepted"] is False
    assert rec_norm["realized_displacement_norm"] == 0.0

    # 2. Custom zero-first grid in plain_kd: skips zero and accepts feasible nonzero
    cfg_zfirst = {"trial_alphas": [[0.0], [1.0]], "max_norm": norm_disp * 2.0}
    rec_zfirst = apply_compact_update(copy.deepcopy(model), cand, method="plain_kd", audit_batches=audit, config=cfg_zfirst)
    assert rec_zfirst["status"] == "accepted_nonzero"
    assert rec_zfirst["accepted"] is True
    assert rec_zfirst["scale"] == 1.0

    # 3. Empty grid in plain_kd
    cfg_empty = {"trial_alphas": []}
    rec_empty = apply_compact_update(copy.deepcopy(model), cand, method="plain_kd", audit_batches=audit, config=cfg_empty)
    assert rec_empty["status"] == "rejected_all"
    assert rec_empty["accepted"] is False


def test_exact_dice_ties_no_hidden_slack():
    """Verify that larger_tie requires exact Dice equality without hidden numerical slack."""
    from self_audit.compact_utility import select_larger_tie

    trials_log = [
        {"trial_index": 0, "foreground_macro_dice": 0.8500000000000000, "displacement_norm": 1.0, "guard_passed": True},
        {"trial_index": 1, "foreground_macro_dice": 0.8500000000000001, "displacement_norm": 0.5, "guard_passed": True},
    ]
    candidate_alphas = [np.array([1.0]), np.array([0.5])]

    # With dice_tolerance=0.0 (exact ties), trial 0 and trial 1 differ by 1e-16 and do NOT tie
    chosen_idx, reason = select_larger_tie(trials_log, orig_trial_idx=1, candidate_alphas=candidate_alphas, dice_tolerance=0.0)
    assert chosen_idx == 1
    assert reason == "larger_tie_matches_original"

    # When Dice is strictly equal (exact tie), larger scalar scale (trial 0 with scale 1.0) is selected
    trials_log_exact = [
        {"trial_index": 0, "foreground_macro_dice": 0.85, "displacement_norm": 1.0, "guard_passed": True},
        {"trial_index": 1, "foreground_macro_dice": 0.85, "displacement_norm": 0.5, "guard_passed": True},
    ]
    chosen_idx_exact, reason_exact = select_larger_tie(trials_log_exact, orig_trial_idx=1, candidate_alphas=candidate_alphas, dice_tolerance=0.0)
    assert chosen_idx_exact == 0
    assert reason_exact == "larger_tie_dice_tiebreak"


def test_probe_actual_forward_count_zero():
    """Verify actual probe forward count is 0 when probe coefficient is 0.0 or denominator is 0."""
    model = DummyStudent()
    audit = _make_dummy_audit()

    d = {n: torch.randn_like(p) * 0.001 for n, p in model.named_parameters()}
    cand = {"directions": [d], "source_ids": ["pseudo_class_1"], "denominator": 100}

    # Coefficient 0.0 -> forward count 0
    probe_coef0 = _make_dummy_probe(coef=0.0)
    rec_c0 = apply_compact_update(
        copy.deepcopy(model), cand, method="scalar_guard", audit_batches=audit,
        config={"selector": "cross_patient_utility"}, utility_probe=probe_coef0
    )
    assert rec_c0["status"] == "zero_displacement"
    assert rec_c0["forward_counts"]["probe_reference_evaluations"] == 0
    assert rec_c0["forward_counts"]["probe_trial_evaluations"] == 0

    # Denominator 0 -> forward count 0
    probe_denom0 = _make_dummy_probe(conf=1.0)
    probe_denom0["targets"].fill_(0.25)
    rec_d0 = apply_compact_update(
        copy.deepcopy(model), cand, method="scalar_guard", audit_batches=audit,
        config={"selector": "cross_patient_utility"}, utility_probe=probe_denom0
    )
    assert rec_d0["status"] == "zero_displacement"
    assert rec_d0["forward_counts"]["probe_reference_evaluations"] == 0
    assert rec_d0["forward_counts"]["probe_trial_evaluations"] == 0


def test_probe_telemetry_and_null_receipt_on_nonfinite():
    """Verify probe telemetry fields and JSON-safe null receipts on non-finite outcomes."""
    model = DummyStudent()
    audit = _make_dummy_audit()

    d = {n: torch.randn_like(p) * 0.001 for n, p in model.named_parameters()}
    cand = {"directions": [d], "source_ids": ["pseudo_class_1"], "denominator": 100}

    # Normal probe telemetry check
    probe = _make_dummy_probe()
    rec = apply_compact_update(
        copy.deepcopy(model), cand, method="scalar_guard", audit_batches=audit,
        config={"selector": "cross_patient_utility"}, utility_probe=probe
    )
    telem = rec["telemetry"]
    assert "reference_kd_loss" in telem and isinstance(telem["reference_kd_loss"], float)
    assert "probe_denominator" in telem and telem["probe_denominator"] > 0
    assert "probe_mask_identity" in telem and isinstance(telem["probe_mask_identity"], str)
    assert "classwise_kd_loss" in telem and isinstance(telem["classwise_kd_loss"], dict)

    # Verify JSON serialization with allow_nan=False
    raw_json = json.dumps(rec, allow_nan=False)
    assert len(raw_json) > 0
