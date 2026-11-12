"""Pure Q-3D weighting contracts, independent of teacher/student modules."""

from __future__ import annotations

import math
import struct

import pytest
import torch
from torch.nn import functional as F

from self_audit.candidate_q3d.losses import (
    cohort_omega,
    weighted_class_balanced_reduce,
)


torch.set_num_threads(1)


def _student_kl(omega: float) -> tuple[float, torch.Tensor]:
    student_logits = torch.tensor(
        [[[[1.2]]], [[[0.1]]], [[[-0.4]]], [[[0.7]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher_probabilities = torch.tensor(
        [[[[0.05]]], [[[0.25]]], [[[0.60]]], [[[0.10]]]],
        dtype=torch.float32,
    )
    per_cell = F.kl_div(
        F.log_softmax(student_logits, dim=0),
        teacher_probabilities,
        reduction="none",
    ).sum(dim=0)
    cls = torch.zeros((1, 1, 1), dtype=torch.long)
    weights = torch.ones_like(cls, dtype=torch.float32)
    loss, _ = weighted_class_balanced_reduce(per_cell, cls, weights, omega)
    loss.backward()
    return float(loss.detach()), student_logits.grad.detach().clone()


def test_student_logit_kl_scales_loss_and_gradient_outside_class_denominator():
    loss_one, grad_one = _student_kl(1.0)
    loss_tenth, grad_tenth = _student_kl(0.1)
    loss_zero, grad_zero = _student_kl(0.0)

    assert loss_one > 0.0
    assert loss_tenth / loss_one == pytest.approx(0.1, rel=1e-6, abs=0.0)
    torch.testing.assert_close(grad_tenth, grad_one * 0.1, rtol=1e-6, atol=0.0)
    assert loss_zero == 0.0
    assert torch.equal(grad_zero, torch.zeros_like(grad_zero))


def test_inline_old_formula_exhibits_patient_weight_cancellation():
    per_cell = torch.tensor([[[2.0, 5.0]]])
    cls = torch.zeros((1, 1, 2), dtype=torch.long)
    weights = torch.ones_like(per_cell)

    def old_formula(patient_weight: float) -> float:
        numerator = (per_cell * weights * patient_weight).sum()
        denominator = (weights * patient_weight).sum() + 1e-8
        return float((numerator / denominator).detach())

    old_one = old_formula(1.0)
    old_tenth = old_formula(0.1)
    old_tiny = old_formula(1e-4)
    assert old_one == pytest.approx(old_tenth, rel=1e-6)
    assert old_one == pytest.approx(old_tiny, rel=1e-4)

    new_one = weighted_class_balanced_reduce(per_cell, cls, weights, 1.0)[0]
    new_tenth = weighted_class_balanced_reduce(per_cell, cls, weights, 0.1)[0]
    new_tiny = weighted_class_balanced_reduce(per_cell, cls, weights, 1e-4)[0]
    assert float(new_tenth) == pytest.approx(float(new_one) * 0.1, rel=1e-6)
    assert float(new_tiny) == pytest.approx(float(new_one) * 1e-4, rel=1e-6)


def test_hand_two_class_weighted_means_are_equal_across_active_classes():
    per_cell = torch.tensor([[[1.0, 3.0, 10.0, 14.0]]])
    cls = torch.tensor([[[0, 0, 1, 1]]], dtype=torch.long)
    weights = torch.tensor([[[1.0, 3.0, 2.0, 1.0]]])
    loss, stats = weighted_class_balanced_reduce(per_cell, cls, weights, 0.5)

    class_zero = (1.0 + 3.0 * 3.0) / 4.0
    class_one = (2.0 * 10.0 + 14.0) / 3.0
    assert float(loss) == pytest.approx(0.5 * ((class_zero + class_one) / 2.0))
    assert stats["active_classids"] == [0, 1]
    assert stats["excluded_classids"] == [2, 3]
    assert stats["counts"] == {0: pytest.approx(4.0), 1: pytest.approx(3.0), 2: 0.0, 3: 0.0}
    assert stats["base_loss"] == pytest.approx(class_zero / 2.0 + class_one / 2.0)
    assert stats["omega"] == 0.5


def test_zero_weight_class_is_excluded_without_diluting_active_class():
    per_cell = torch.tensor([[[2.0, 2.0, 100.0]]])
    cls = torch.tensor([[[0, 0, 1]]], dtype=torch.long)
    weights = torch.tensor([[[1.0, 1.0, 0.0]]])
    loss, stats = weighted_class_balanced_reduce(per_cell, cls, weights, 1.0)
    assert float(loss) == pytest.approx(2.0)
    assert stats["active_classids"] == [0]
    assert stats["excluded_classids"] == [1, 2, 3]


def test_all_zero_support_returns_graph_connected_zero_and_zero_gradient():
    per_cell = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    cls = torch.tensor([[[0, 1]]], dtype=torch.long)
    weights = torch.zeros_like(per_cell)
    loss, stats = weighted_class_balanced_reduce(per_cell, cls, weights, 0.7)
    assert loss.ndim == 0
    assert loss.grad_fn is not None
    assert float(loss.detach()) == 0.0
    assert stats["active_classids"] == []
    loss.backward()
    assert per_cell.grad is not None
    assert torch.equal(per_cell.grad, torch.zeros_like(per_cell))


def test_cohort_v1_normalizes_mean_and_reports_ess():
    result = cohort_omega({"p0": 1.0, "p1": 0.5, "p2": 0.25}, "V1")
    expected_ess = 1.75**2 / (1.0**2 + 0.5**2 + 0.25**2)
    assert result["mean_w_pat"] == pytest.approx(1.75 / 3.0)
    assert result["ess"] == pytest.approx(expected_ess)
    assert result["ess_fraction"] == pytest.approx(expected_ess / 3.0)
    assert result["mean_omega"] == pytest.approx(1.0)
    assert result["max_omega"] == pytest.approx(1.0 / (1.75 / 3.0))
    assert result["zero_fraction"] == 0.0


def test_l_is_unit_weighted_and_skips_physiology_gate():
    result = cohort_omega({"p0": 0.0, "p1": 0.0}, "L")
    assert result["omega"] == {"p0": 1.0, "p1": 1.0}
    assert result["mean_w_pat"] == 0.0
    assert result["ess"] == 0.0
    assert result["ess_fraction"] == 0.0
    assert result["mean_omega"] == 1.0
    assert result["max_omega"] == 1.0
    assert result["zero_fraction"] == 0.0


def test_v1u_is_unit_weighted_but_shares_u_gate():
    with pytest.raises(ValueError, match="mean_w_pat>0"):
        cohort_omega({"p0": 0.0, "p1": 0.0}, "V1-U")
    result = cohort_omega({"p0": 1.0, "p1": 0.0, "p2": 0.0, "p3": 0.0}, "V1u")
    assert result["omega"] == {patient: 1.0 for patient in ("p0", "p1", "p2", "p3")}
    assert result["ess_fraction"] == pytest.approx(0.25)


@pytest.mark.parametrize("arm", ["V1", "C1", "V1-U"])
def test_zero_mean_weighted_u_cohort_fails_pre_j_gate(arm):
    with pytest.raises(ValueError, match="mean_w_pat>0"):
        cohort_omega({"p0": 0.0, "p1": 0.0}, arm)


def test_ess_gate_accepts_exact_boundary_and_rejects_below_boundary():
    boundary = cohort_omega({"p0": 1.0, "p1": 0.0, "p2": 0.0, "p3": 0.0}, "V1")
    assert boundary["ess_fraction"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="ESS/N>=0.25"):
        cohort_omega({"p0": 1.0, "p1": 0.0, "p2": 0.0, "p3": 0.0, "p4": 0.0}, "V1")


def test_equal_positive_weights_emit_bitwise_identical_unit_omegas():
    source = {"p0": 0.3, "p1": 0.3, "p2": 0.3}
    v1 = cohort_omega(source, "V1")["omega"]
    v1u = cohort_omega(source, "V1-U")["omega"]
    assert v1 == v1u
    assert all(struct.pack("!d", value) == struct.pack("!d", 1.0) for value in v1.values())


def test_tiny_equal_positive_weights_are_stable_and_bitwise_unit():
    source = {"p0": 1e-200, "p1": 1e-200, "p2": 1e-200, "p3": 1e-200}
    v1 = cohort_omega(source, "V1")
    v1u = cohort_omega(source, "V1-U")
    assert v1["ess"] == 4.0
    assert v1["ess_fraction"] == 1.0
    assert v1["omega"] == v1u["omega"]
    assert all(struct.pack("!d", value) == struct.pack("!d", 1.0)
               for value in v1["omega"].values())


def test_batch_frames_use_one_shared_class_weight_map_then_average_frames():
    per_cell = torch.tensor([
        [[[1.0, 3.0, 10.0, 14.0]]],
        [[[5.0, 7.0, 20.0, 22.0]]],
    ])
    cls = torch.tensor([[[0, 0, 1, 1]]], dtype=torch.long)
    weights = torch.tensor([[[1.0, 3.0, 2.0, 1.0]]])
    loss, stats = weighted_class_balanced_reduce(per_cell, cls, weights, 1.0)
    frame_zero = ((1.0 + 3.0 * 3.0) / 4.0 + (2.0 * 10.0 + 14.0) / 3.0) / 2.0
    frame_one = ((5.0 + 3.0 * 7.0) / 4.0 + (2.0 * 20.0 + 22.0) / 3.0) / 2.0
    assert float(loss) == pytest.approx((frame_zero + frame_one) / 2.0)
    assert stats["active_classids"] == [0, 1]


def test_cell_weights_are_detached_from_loss_graph():
    per_cell = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    cls = torch.tensor([[[0, 0]]], dtype=torch.long)
    weights = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    loss, _ = weighted_class_balanced_reduce(per_cell, cls, weights, 1.0)
    loss.backward()
    assert per_cell.grad is not None
    assert weights.grad is None


def test_tiny_single_cell_support_has_finite_unit_gradient():
    per_cell = torch.tensor([[[3.0]]], dtype=torch.float32, requires_grad=True)
    cls = torch.zeros((1, 1, 1), dtype=torch.long)
    weights = torch.tensor([[[1e-320]]], dtype=torch.float64)
    loss, _ = weighted_class_balanced_reduce(per_cell, cls, weights, 1.0)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(3.0)
    assert per_cell.grad is not None
    assert torch.isfinite(per_cell.grad).all()
    assert per_cell.grad.item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bad_loss,bad_cls,bad_weights,bad_omega,match",
    [
        (torch.ones((1, 1)), torch.zeros((1, 1), dtype=torch.long), torch.ones((1, 1)), 1.0, "shape"),
        (torch.tensor([[[math.nan]]]), torch.zeros((1, 1, 1), dtype=torch.long), torch.ones((1, 1, 1)), 1.0, "NaN"),
        (torch.ones((1, 1, 1)), torch.zeros((1, 1, 1), dtype=torch.float32), torch.ones((1, 1, 1)), 1.0, "integer"),
        (torch.ones((1, 1, 1)), torch.full((1, 1, 1), 4, dtype=torch.long), torch.ones((1, 1, 1)), 1.0, "outside"),
        (torch.ones((1, 1, 1)), torch.zeros((1, 1, 1), dtype=torch.long), -torch.ones((1, 1, 1)), 1.0, "nonnegative"),
        (torch.ones((1, 1, 1)), torch.zeros((1, 1, 1), dtype=torch.long), torch.ones((1, 1, 1)), 1.0, ""),
    ],
)
def test_weighted_reduce_rejects_invalid_inputs(bad_loss, bad_cls, bad_weights, bad_omega, match):
    if match == "":
        with pytest.raises((TypeError, ValueError)):
            weighted_class_balanced_reduce(bad_loss, bad_cls, bad_weights, [1.0])
    else:
        with pytest.raises((TypeError, ValueError, FloatingPointError), match=match):
            weighted_class_balanced_reduce(bad_loss, bad_cls, bad_weights, bad_omega)


def test_weighted_reduce_rejects_nonfinite_or_negative_omega_and_cohort_inputs():
    per_cell = torch.ones((1, 1, 1))
    cls = torch.zeros((1, 1, 1), dtype=torch.long)
    weights = torch.ones_like(per_cell)
    with pytest.raises(ValueError, match="nonnegative"):
        weighted_class_balanced_reduce(per_cell, cls, weights, -1.0)
    with pytest.raises(ValueError, match="finite"):
        weighted_class_balanced_reduce(per_cell, cls, weights, float("inf"))
    with pytest.raises(ValueError, match="nonempty"):
        cohort_omega({}, "V1")
    with pytest.raises(ValueError, match="finite"):
        cohort_omega({"p0": float("nan")}, "V1")
    with pytest.raises(ValueError, match="nonnegative"):
        cohort_omega({"p0": -0.1}, "V1")
    with pytest.raises(ValueError, match="one of"):
        cohort_omega({"p0": 1.0}, "other")
