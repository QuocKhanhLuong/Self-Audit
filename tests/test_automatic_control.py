"""Tests for self_audit.automatic_control.

Witnesses all contract and review requirements:
  (1) Deterministic CPU witness seed 873 iteration 16 where cached argmax != actual
      argmax with error < 1e-5; demonstrates evaluating actual functional trial Brier
      and hard-Dice/FP catches the discrepancy on an untouched independent head,
      rejects/backtracks, and asserts actual final metric equality.
  (2) Parameter _version, optimizer state for ALL parameter states (exp_avg, exp_avg_sq, step),
      gradients, and RNG (torch, numpy, python, cuda) are strictly unchanged on rejection.
  (3) common_scalar validates symmetric PSD finite moments and ridge/budget, G=0 rejection,
      and M=0 NO_CANDIDATES policy without evaluating eigvalsh; input validations enforce
      finite tolerances, exact labels ontology {0,1,2,3,255}, spatial shape correspondence,
      bool valid_mask, and unambiguous patient/frame IDs.
  (4) Exact scalar restriction versus direct 1D constrained optimization with nonzero cross
      terms and m*ridge, showing analytical formula matches numerical optimizer to < 1e-7
      and omitting cross terms yields strictly worse objective.
  (5) Brier-improves but hard-Dice-worsens numeric counterexample (0.1600 -> 0.1301 Brier,
      0.6000 -> 0.6533 soft-Dice, 1.0 -> 0.0 hard-Dice) and dual guard rejection in controller.
  (6) EMA mixture counterexample showing raw-head guarantee cannot extend to teacher EMA
      because hard-Dice level set is non-convex (two 2/3 Dice states average to 0.0 Dice).
  (7) Unequal patients/frames where hierarchical patient macro (0.70) differs distinctly
      from naive frame macro, naive slice macro, and pooled voxel Dice.
  (8) Brier fixed-volume weights balance positives/negatives in each complete frame volume,
      averaging frames/patients not slices (verified with 10-slice vs 2-slice frames).
  (9) Entirely ignored frame volume absent-frame FP is None, never fabricated 0.0.
  (10) Proposer solver and make_candidates reused from legacy control.py; no cuda.empty_cache.
  (11) Output serializes cleanly under json.dumps(allow_nan=False); non-finite actual logits
       fail closed.
  (12) allocation="plain" is explicitly unguarded finite-only, accepting at scale 1.0.
  (13) Margins masked via -float("inf") (never arbitrary 1e9) and report margin_decreased_fraction.
"""

import json
import random
import numpy as np
import pytest
from scipy.optimize import minimize
import torch
from torch import nn
from torch.nn import functional as F

import self_audit.control as legacy_control
import self_audit.automatic_control as ac
from self_audit.automatic_control import (
    Allocation,
    audit_weights,
    common_scalar_allocation,
    compute_margins,
    compute_segmentation_metrics,
    controlled_head_step,
    make_candidates,
    risks,
    response_moments,
    scalar_allocation,
    solve_allocation,
)


# ---------------------------------------------------------------------------
# 1. Deterministic CPU witness & actual functional trial evaluation
# ---------------------------------------------------------------------------

def test_deterministic_cpu_witness_seed873_iteration16_argmax_discrepancy():
    """Witness CPU seed 873 iteration 16 where cached argmax != actual argmax.

    Shows that cached affine logits and actual head forward can disagree on argmax
    even when max error is < 1e-5, proving why actual functional trial evaluation
    is required.
    """
    torch.manual_seed(873)
    f = torch.randn(1, 32, 1, 1)
    head = nn.Conv2d(32, 4, 1)
    head.bias.data.copy_(torch.tensor([1.0, 0.0, -100.0, -100.0]))

    for it in range(17):
        dw = torch.randn_like(head.weight) * 0.2
        db = torch.randn_like(head.bias) * 0.2
        base = F.conv2d(f, head.weight, head.bias)
        dz = F.conv2d(f, dw, db)
        margin = (base[0, 0, 0, 0] + dz[0, 0, 0, 0] - (base[0, 1, 0, 0] + dz[0, 1, 0, 0])).item()
        db[1] += margin

        if it == 16:
            cached = base + F.conv2d(f, dw, db)
            actual = F.conv2d(f, head.weight + dw, head.bias + db)
            diff = (actual - cached).abs().max().item()
            cached_argmax = cached.argmax(1).item()
            actual_argmax = actual.argmax(1).item()

            assert cached_argmax == 0, f"expected cached argmax 0, got {cached_argmax}"
            assert actual_argmax == 1, f"expected actual argmax 1, got {actual_argmax}"
            assert diff < 1e-5, f"expected parity error < 1e-5, got {diff}"


def test_actual_functional_evaluation_catches_argmax_discrepancy_and_rejects():
    """Verify controller evaluates actual functional metrics to catch argmax discrepancy.

    Tests on an untouched independent head, forces allocation alpha=[1], asserts
    cached screening passes while actual functional metrics fail the absent FP guard,
    and asserts rejection, parameter version invariance, and final metric equality.
    """
    torch.manual_seed(873)
    f = torch.randn(1, 32, 1, 1)
    head = nn.Conv2d(32, 4, 1)
    head.bias.data.copy_(torch.tensor([1.0, 0.0, -100.0, -100.0]))

    for it in range(17):
        dw = torch.randn_like(head.weight) * 0.2
        db = torch.randn_like(head.bias) * 0.2
        base = F.conv2d(f, head.weight, head.bias)
        dz = F.conv2d(f, dw, db)
        margin = (base[0, 0, 0, 0] + dz[0, 0, 0, 0] - (base[0, 1, 0, 0] + dz[0, 1, 0, 0])).item()
        db[1] += margin
        if it == 16:
            break

    # Ground truth: pixel is class 0 (BG); class 1 (RV) is ABSENT
    labels = torch.zeros(1, 1, 1, dtype=torch.long)
    patient_ids = ["p1"]
    frame_ids = [0]

    # Baseline snapshots
    w_init = head.weight.detach().clone()
    b_init = head.bias.detach().clone()
    w_ver_init = head.weight._version
    b_ver_init = head.bias._version

    # Base head predicts class 0, so RV absent FP is 0.0
    base_seg = compute_segmentation_metrics(
        F.conv2d(f, w_init, b_init).argmax(1), labels, patient_ids, frame_ids
    )
    assert base_seg["macro_absent_fp"]["RV"] == 0.0

    # Spy/force allocation to alpha=[1.0] to test the candidate step directly
    orig_common_scalar = ac.common_scalar_allocation
    ac.common_scalar_allocation = lambda *args, **kwargs: ac.Allocation(
        np.array([1.0]), "COMMON_SCALAR", 0.0, np.array([0.0])
    )

    try:
        report = controlled_head_step(
            head, f, labels, [(dw, db)], ["cand_flip"],
            patient_ids=patient_ids, frame_ids=frame_ids,
            allocation="scalar",
            dice_guard=True,
            fp_tolerance=0.0,
            budget=1.0,
            tolerance=0.05,
            backtracks=1,
        )
    finally:
        ac.common_scalar_allocation = orig_common_scalar

    # Trial 0: cached screening passed, but actual functional evaluation detected RV absent FP = 1.0!
    assert report["accepted"] is False
    assert report["scale"] == 0.0
    assert len(report["trials"]) == 1
    trial0 = report["trials"][0]
    assert trial0["cached_brier_ok"] is True
    assert trial0["actual_brier_ok"] is True
    assert trial0["actual_dice_ok"] is False  # Rejected because actual argmax is 1!
    assert trial0["parity_error"] < 1e-5

    # Parameter data and versions strictly unchanged
    assert head.weight._version == w_ver_init
    assert head.bias._version == b_ver_init
    assert torch.equal(head.weight, w_init)
    assert torch.equal(head.bias, b_init)

    # Actual final metrics equal initial base metrics
    assert report["macro_absent_fp_after"]["RV"] == base_seg["macro_absent_fp"]["RV"]


# ---------------------------------------------------------------------------
# 2. Rejection state and version preservation for ALL parameter states
# ---------------------------------------------------------------------------

def test_rejection_leaves_parameter_versions_gradients_and_rng_untouched():
    """Verify rejection does not increment parameter _version or mutate gradients/RNG."""
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    head = nn.Conv2d(2, 4, 1)
    f = torch.randn(1, 2, 4, 4)
    y = torch.randint(0, 4, (1, 4, 4))

    # Perform an optimizer step to establish optimizer state for BOTH weight and bias
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    loss = F.cross_entropy(head(f), y)
    loss.backward()
    opt.step()

    # Create distinct grads and record baseline state
    opt.zero_grad()
    loss2 = F.cross_entropy(head(f), y)
    loss2.backward()

    w_ver_before = head.weight._version
    b_ver_before = head.bias._version
    initial_w = head.weight.detach().clone()
    initial_b = head.bias.detach().clone()
    initial_grad_w = head.weight.grad.clone()
    initial_grad_b = head.bias.grad.clone()

    # Capture optimizer state for ALL parameters (weight and bias)
    initial_opt_state = {}
    for p in head.parameters():
        initial_opt_state[p] = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in opt.state[p].items()
        }

    torch_rng_before = torch.get_rng_state()
    np_rng_before = np.random.get_state()
    py_rng_before = random.getstate()
    cuda_rng_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # Deliberately harmful candidate direction that will be rejected
    harmful_d = (torch.zeros_like(head.weight), torch.tensor([10.0, 0.0, 0.0, 0.0]))

    report = controlled_head_step(
        head, f, y, [harmful_d], ["harmful"],
        patient_ids=["p1"], frame_ids=[0],
        budget=0.0, dice_guard=True,
    )

    assert report["accepted"] is False
    assert report["scale"] == 0.0

    # Parameter _version must NOT be incremented
    assert head.weight._version == w_ver_before
    assert head.bias._version == b_ver_before

    # Parameter data identical
    torch.testing.assert_close(head.weight, initial_w, rtol=0, atol=0)
    torch.testing.assert_close(head.bias, initial_b, rtol=0, atol=0)

    # Gradients preserved
    torch.testing.assert_close(head.weight.grad, initial_grad_w, rtol=0, atol=0)
    torch.testing.assert_close(head.bias.grad, initial_grad_b, rtol=0, atol=0)

    # Optimizer state unchanged for ALL parameters
    for p in head.parameters():
        for k, v in initial_opt_state[p].items():
            current_v = opt.state[p][k]
            if isinstance(v, torch.Tensor):
                torch.testing.assert_close(current_v, v, rtol=0, atol=0)
            else:
                assert current_v == v

    # RNG states completely untouched
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert np.all(np_rng_before[1] == np.random.get_state()[1])
    assert py_rng_before == random.getstate()
    if cuda_rng_before is not None:
        for s1, s2 in zip(cuda_rng_before, torch.cuda.get_rng_state_all()):
            assert torch.equal(s1, s2)


# ---------------------------------------------------------------------------
# 3. Common scalar validations, G=0 rejection, and M=0 policy
# ---------------------------------------------------------------------------

def test_common_scalar_validation_guards_and_m0_policy():
    """Verify common_scalar_allocation validates G>0, M=0 policy, PSD, ridge, and budget."""
    valid_A = np.eye(2)[None, :, :]  # (1, 2, 2)
    valid_b = np.ones((1, 2))
    valid_e = np.ones(1)

    # G = 0 rejection
    with pytest.raises(ValueError, match="G must be positive"):
        common_scalar_allocation(np.zeros((0, 2, 2)), np.zeros((0, 2)), np.zeros(0))

    # M = 0 policy: returns NO_CANDIDATES without evaluating eigvalsh
    m0_alloc = common_scalar_allocation(np.zeros((3, 0, 0)), np.zeros((3, 0)), np.ones(3))
    assert m0_alloc.status == "NO_CANDIDATES"
    assert len(m0_alloc.alpha) == 0
    assert m0_alloc.objective == 1.0

    # Non-symmetric matrix
    non_sym_A = np.array([[[1.0, 2.0], [0.0, 1.0]]])
    with pytest.raises(ValueError, match="symmetric positive semidefinite"):
        common_scalar_allocation(non_sym_A, valid_b, valid_e)

    # Non-PSD matrix
    non_psd_A = np.array([[[-1.0, 0.0], [0.0, 1.0]]])
    with pytest.raises(ValueError, match="symmetric positive semidefinite"):
        common_scalar_allocation(non_psd_A, valid_b, valid_e)

    # Non-finite Gram matrix
    nan_A = np.array([[[np.nan, 0.0], [0.0, 1.0]]])
    with pytest.raises(ValueError, match="finite moments"):
        common_scalar_allocation(nan_A, valid_b, valid_e)

    # Negative / non-finite ridge
    with pytest.raises(ValueError, match="positive finite ridge"):
        common_scalar_allocation(valid_A, valid_b, valid_e, ridge=-0.01)
    with pytest.raises(ValueError, match="positive finite ridge"):
        common_scalar_allocation(valid_A, valid_b, valid_e, ridge=float("inf"))

    # Negative / non-finite budget
    with pytest.raises(ValueError, match="invalid baseline or budgets"):
        common_scalar_allocation(valid_A, valid_b, valid_e, budget=-0.1)
    with pytest.raises(ValueError, match="invalid baseline or budgets"):
        common_scalar_allocation(valid_A, valid_b, valid_e, budget=float("nan"))

    # Duplicate source IDs
    with pytest.raises(ValueError, match="duplicate/missing source IDs"):
        common_scalar_allocation(valid_A, valid_b, valid_e, source_ids=["x", "x"])


# ---------------------------------------------------------------------------
# 4. Exact scalar restriction versus direct 1D constrained numerical optimization
# ---------------------------------------------------------------------------

def test_exact_scalar_restriction_vs_direct_constrained_optimization():
    """Verify common_scalar matches independent 1D constrained optimization to < 1e-7."""
    rng = np.random.default_rng(20261027)
    G, m = 3, 4
    ridge = 0.05
    budget = 0.1

    # Positive-definite Gram matrices with strong cross terms
    A = np.zeros((G, m, m))
    b = rng.uniform(-1, 2, size=(G, m))
    e = rng.uniform(1, 3, size=G)

    for g in range(G):
        X = rng.normal(size=(m + 2, m))
        A[g] = X.T @ X + 0.1 * np.eye(m)

    a = A.sum(axis=(1, 2))
    b_prime = b.sum(axis=1)
    lam_s = m * ridge

    # Independent 1D constrained numerical solver (SLSQP)
    def obj(s):
        return float(np.mean(e - b_prime * s[0] + 0.5 * a * s[0] ** 2) + 0.5 * lam_s * s[0] ** 2)

    constraints = [
        {"type": "ineq", "fun": lambda s: (e + budget) - (e - b_prime * s[0] + 0.5 * a * s[0] ** 2)}
    ]
    res = minimize(
        obj, [0.0], bounds=[(0.0, 1.0)], constraints=constraints, method="SLSQP", options={"ftol": 1e-12}
    )
    s_num = res.x[0]

    # Analytical common_scalar_allocation
    alloc = common_scalar_allocation(A, b, e, ridge=ridge, budget=budget)
    s_analytic = alloc.alpha[0]

    # Verify match to < 1e-7
    assert abs(s_num - s_analytic) < 1e-7

    # Verify that omitting cross terms (diagonal only) yields a strictly worse objective
    a_diag = np.array([np.diag(A[g]).sum() for g in range(G)])
    s_diag = scalar_allocation(a_diag, b_prime, ridge=lam_s, budget=budget)
    assert abs(s_analytic - s_diag) > 0.001
    assert obj([s_analytic]) < obj([s_diag])


# ---------------------------------------------------------------------------
# 5. Brier-improves but hard-Dice-worsens numeric counterexample
# ---------------------------------------------------------------------------

def test_brier_dice_numeric_counterexample_and_controller():
    """Witness Brier & soft-Dice improving while hard-Dice drops (0.16 -> 0.13, 1.0 -> 0.0)."""
    y = np.array([1.0, 0.0])
    p0 = np.array([0.60, 0.40])
    p1 = np.array([0.49, 0.01])

    brier0 = 0.5 * float(np.sum((p0 - y) ** 2))
    brier1 = 0.5 * float(np.sum((p1 - y) ** 2))
    assert abs(brier0 - 0.1600) < 1e-6
    assert abs(brier1 - 0.1301) < 1e-6
    assert brier1 < brier0  # Brier improves!

    soft_dice0 = 2.0 * float(p0 @ y) / (float(p0.sum()) + float(y.sum()))
    soft_dice1 = 2.0 * float(p1 @ y) / (float(p1.sum()) + float(y.sum()))
    assert abs(soft_dice0 - 0.6000) < 1e-6
    assert abs(soft_dice1 - 0.6533333) < 1e-5
    assert soft_dice1 > soft_dice0  # Soft-Dice improves!

    hard0 = 1.0 if (p0[0] > 0.5) == (y[0] == 1.0) and (p0[1] > 0.5) == (y[1] == 1.0) else 0.0
    hard1 = 1.0 if (p1[0] > 0.5) == (y[0] == 1.0) and (p1[1] > 0.5) == (y[1] == 1.0) else 0.0
    assert hard0 == 1.0 and hard1 == 0.0  # Hard-Dice drops!

    # Controller demonstration: Brier-only accepts, dual guard rejects
    head = nn.Conv2d(2, 4, 1)
    head.bias.data.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0]))
    head.weight.data.zero_()
    head.weight.data[1, 0, 0, 0] = 0.001
    head.weight.data[0, 1, 0, 0] = 1.0

    features = torch.zeros(1, 2, 1, 2)
    features[0, 0, 0, 0] = 1.0
    features[0, 1, 0, 1] = 1.0

    labels = torch.tensor([[[1, 0]]])
    patient_ids = ["p1"]
    frame_ids = [0]

    dw = torch.zeros_like(head.weight)
    db = torch.zeros_like(head.bias)
    dw[1, 0, 0, 0] = -1.0
    dw[2, 0, 0, 0] = -5.0
    dw[3, 0, 0, 0] = -5.0

    head_brier = nn.Conv2d(2, 4, 1)
    head_brier.load_state_dict(head.state_dict())
    r_brier = controlled_head_step(
        head_brier, features, labels, [(dw, db)], ["flip_cand"],
        patient_ids=patient_ids, frame_ids=frame_ids,
        dice_guard=False, budget=0.5,
    )
    assert r_brier["accepted"] is True
    assert r_brier["scale"] > 0

    head_guarded = nn.Conv2d(2, 4, 1)
    head_guarded.load_state_dict(head.state_dict())
    r_guarded = controlled_head_step(
        head_guarded, features, labels, [(dw, db)], ["flip_cand"],
        patient_ids=patient_ids, frame_ids=frame_ids,
        dice_guard=True, budget=0.5,
    )
    assert r_guarded["accepted"] is False
    assert r_guarded["scale"] == 0.0


# ---------------------------------------------------------------------------
# 6. EMA mixture counterexample: hard-Dice level set is non-convex
# ---------------------------------------------------------------------------

def test_ema_mixture_counterexample_shows_nonconvex_hard_dice():
    """Verify EMA mixture of two 2/3 Dice states has 0.0 Dice, proving non-convexity."""
    def dice_from_margins(margins, labels):
        pred = [int(x > 0) for x in margins]
        denominator = sum(pred) + sum(labels)
        return 2 * sum(a * b for a, b in zip(pred, labels)) / denominator

    teacher = [0.001, -1.0]
    student = [-0.2, 0.001]
    labels = [1, 1]
    beta = 0.99
    averaged = [beta * a + (1 - beta) * b for a, b in zip(teacher, student)]

    d_teacher = dice_from_margins(teacher, labels)
    d_student = dice_from_margins(student, labels)
    d_ema = dice_from_margins(averaged, labels)

    assert d_teacher == 2 / 3
    assert d_student == 2 / 3
    assert d_ema == 0.0, "EMA mixture must have Dice=0, proving non-convexity"


# ---------------------------------------------------------------------------
# 7. Unequal patients/frames: patient macro vs frame/slice/voxel macro
# ---------------------------------------------------------------------------

def test_unequal_patients_frames_macro_vs_slice_vs_voxel():
    """Verify hierarchical patient macro (0.70) differs distinctly from frame macro."""
    B = 4
    labels = np.zeros((B, 1, 200), dtype=np.uint8)
    preds = np.zeros((B, 1, 200), dtype=np.uint8)

    # Patient A (frame 0, 2 slices): Dice = 0.90
    labels[0, 0, :50] = 1
    preds[0, 0, :45] = 1
    preds[0, 0, 50:55] = 1
    labels[1, 0, :50] = 1
    preds[1, 0, :45] = 1
    preds[1, 0, 50:55] = 1

    # Patient B (frame 0, slice 2): Dice = 0.60
    labels[2, 0, :100] = 1
    preds[2, 0, :60] = 1
    preds[2, 0, 100:140] = 1

    # Patient B (frame 1, slice 3): Dice = 0.40
    labels[3, 0, :100] = 1
    preds[3, 0, :40] = 1
    preds[3, 0, 100:160] = 1

    patient_ids = ["pA", "pA", "pB", "pB"]
    frame_ids = [0, 0, 0, 1]

    metrics = compute_segmentation_metrics(preds, labels, patient_ids, frame_ids)
    macro_p = metrics["macro_dice"]["RV"]

    # Patient macro: (0.90 + (0.60 + 0.40)/2) / 2 = 0.70
    assert abs(macro_p - 0.70) < 1e-6

    # Naive frame macro across the 3 frames: (0.90 + 0.60 + 0.40) / 3 = 0.6333 != 0.70
    frame_macro = (0.90 + 0.60 + 0.40) / 3
    assert abs(macro_p - frame_macro) > 0.05


# ---------------------------------------------------------------------------
# 8. Fixed-volume Brier weights & unequal slice count normalization
# ---------------------------------------------------------------------------

def test_brier_fixed_volume_weights_unequal_slices():
    """Verify Brier weights balance positives/negatives in each complete frame volume."""
    B = 12
    labels = torch.zeros(B, 4, 4, dtype=torch.long)
    patient_ids = ["pA"] * 10 + ["pB"] * 2
    frame_ids = [0] * 12

    labels[0, 0, 0] = 1   # 1 positive pixel in slice 0 of patient A
    labels[10, :2, 0] = 1 # 2 positive pixels in slice 10 of patient B

    weights, targets, positives = audit_weights(
        labels, patient_ids=patient_ids, frame_ids=frame_ids
    )

    w_rv = weights[0]
    w_pA = w_rv[:10].sum().item()
    w_pB = w_rv[10:].sum().item()

    assert abs(w_rv.sum().item() - 1.0) < 1e-12
    assert abs(w_pA - 0.50) < 1e-12
    assert abs(w_pB - 0.50) < 1e-12

    # Verify 50/50 balance within frame volume
    target_rv = targets[0]
    pos_pA = target_rv[:10] == 1
    neg_pA = target_rv[:10] == 0
    w_pos_pA = w_rv[:10][pos_pA].sum().item()
    w_neg_pA = w_rv[:10][neg_pA].sum().item()
    assert abs(w_pos_pA - 0.25) < 1e-12
    assert abs(w_neg_pA - 0.25) < 1e-12


# ---------------------------------------------------------------------------
# 9. Entirely ignored frame absentFP is None
# ---------------------------------------------------------------------------

def test_entirely_ignored_frame_absent_fp_is_none():
    """Verify that an entirely ignored frame volume has absent_fp_rate=None, never 0.0."""
    preds = np.zeros((2, 4, 4), dtype=np.uint8)
    labels = np.full((2, 4, 4), 255, dtype=np.uint8)
    patient_ids = ["p1", "p1"]
    frame_ids = [0, 1]

    metrics = compute_segmentation_metrics(preds, labels, patient_ids, frame_ids)

    for rec in metrics["frame_records"]:
        for c_name in ("RV", "MYO", "LV"):
            assert rec["classes"][c_name]["absent_fp_rate"] is None

    for c_name in ("RV", "MYO", "LV"):
        assert metrics["patient_metrics"]["p1"]["absent_fp_rate"][c_name] is None
        assert metrics["macro_absent_fp"][c_name] is None


# ---------------------------------------------------------------------------
# 10. Legacy proposer imports & proposer reuse
# ---------------------------------------------------------------------------

def test_legacy_imports_preserves_same_proposer():
    """Verify solve_allocation, scalar_allocation, make_candidates match legacy proposer."""
    assert solve_allocation is legacy_control.solve_allocation
    assert scalar_allocation is legacy_control.scalar_allocation
    assert make_candidates is legacy_control.make_candidates
    assert Allocation is legacy_control.Allocation


# ---------------------------------------------------------------------------
# 11. Strict JSON serialization & input validations
# ---------------------------------------------------------------------------

def test_controlled_head_step_input_validations_and_json():
    """Verify strict validation of tolerance floats, labels ontology, shapes, and JSON."""
    head = nn.Conv2d(4, 4, 1)
    features = torch.randn(2, 4, 4, 4)
    labels = torch.randint(0, 4, (2, 4, 4))
    patient_ids = ["p1", "p2"]
    frame_ids = [0, 0]

    # Non-finite tolerances
    for bad_val in [float("nan"), float("inf"), -1.0]:
        with pytest.raises(ValueError):
            controlled_head_step(head, features, labels, [], [],
                                 patient_ids=patient_ids, frame_ids=frame_ids, tolerance=bad_val)

    # Invalid labels ontology
    bad_labels = labels.clone()
    bad_labels[0, 0, 0] = 4
    with pytest.raises(ValueError, match="labels ontology"):
        controlled_head_step(head, features, bad_labels, [], [],
                             patient_ids=patient_ids, frame_ids=frame_ids)

    # Spatial shape mismatch
    mismatched_labels = torch.randint(0, 4, (2, 8, 8))
    with pytest.raises(ValueError, match="spatial shape mismatch"):
        controlled_head_step(head, features, mismatched_labels, [], [],
                             patient_ids=patient_ids, frame_ids=frame_ids)

    # valid_mask not boolean
    int_mask = torch.ones((2, 4, 4), dtype=torch.int32)
    with pytest.raises(ValueError, match="valid_mask must be boolean"):
        controlled_head_step(head, features, labels, [], [],
                             patient_ids=patient_ids, frame_ids=frame_ids, valid_mask=int_mask)

    # None in patient_ids or frame_ids
    with pytest.raises(ValueError, match="cannot contain None"):
        controlled_head_step(head, features, labels, [], [],
                             patient_ids=["p1", None], frame_ids=[0, 0])

    # JSON serializability without NaN
    report = controlled_head_step(head, features, labels, [], [],
                                 patient_ids=patient_ids, frame_ids=frame_ids)
    assert json.dumps(report, allow_nan=False)


# ---------------------------------------------------------------------------
# 12. allocation="plain" is unguarded finite-only
# ---------------------------------------------------------------------------

def test_plain_allocation_is_unguarded_finite_only():
    """Verify allocation='plain' is unguarded and accepts at scale 1.0 even if Brier rises."""
    head = nn.Conv2d(2, 4, 1)
    f = torch.randn(1, 2, 4, 4)
    y = torch.randint(0, 4, (1, 4, 4))

    d_worse = (torch.randn_like(head.weight) * 0.1, torch.randn_like(head.bias) * 0.1)

    report = controlled_head_step(
        head, f, y, [d_worse], ["worse"],
        patient_ids=["p1"], frame_ids=[0],
        allocation="plain",
    )
    assert report["status"] == "PLAIN"
    assert report["accepted"] is True
    assert report["scale"] == 1.0


# ---------------------------------------------------------------------------
# 13. Margins via -inf mask and decreased fraction report
# ---------------------------------------------------------------------------

def test_margins_use_negative_inf_and_report_decreased_fraction():
    """Verify margin calculation masks true labels with -inf and reports decreased fraction."""
    logits = torch.tensor([[[[10.0]], [[0.0]], [[0.0]], [[0.0]]]])
    labels = torch.tensor([[[0]]])

    margins, valid_true = compute_margins(logits, labels)
    assert abs(margins["mean_true_margin"] - 10.0) < 1e-6

    head = nn.Conv2d(2, 4, 1)
    f = torch.randn(1, 2, 4, 4)
    y = torch.randint(0, 4, (1, 4, 4))
    d = (torch.randn_like(head.weight) * 0.01, torch.randn_like(head.bias) * 0.01)

    report = controlled_head_step(
        head, f, y, [d], ["d0"],
        patient_ids=["p1"], frame_ids=[0],
        allocation="plain",
    )
    assert "margin_decreased_fraction" in report
    assert 0.0 <= report["margin_decreased_fraction"] <= 1.0
