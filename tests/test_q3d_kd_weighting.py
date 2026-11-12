"""Regression coverage for the public KD losses consumed by the train engine."""

import math

import pytest
import torch

from self_audit.candidate_q3d.losses import codekd_loss, feature_kd_loss


def _kd_inputs(kind):
    student = torch.tensor(
        [[[[[1.2, -0.4, 0.3, 0.8]]], [[[0.2, 0.7, 1.1, -0.3]]]]],
        dtype=torch.float64, requires_grad=True,
    )
    teacher = torch.tensor(
        [[[[[0.1, 0.8, 0.4, 0.2]]], [[[0.9, 0.2, 0.6, 0.8]]]]],
        dtype=torch.float64, requires_grad=True,
    )
    classes = torch.tensor([[[[0, 0, 1, 2]]]])
    weights = torch.tensor([[[[0.7, 1.0, 0.9, 0.0]]]], dtype=torch.float64,
                           requires_grad=True)
    target = student.softmax(dim=1) if kind == "code" else student
    function = codekd_loss if kind == "code" else feature_kd_loss
    return function, teacher, target, student, classes, weights


@pytest.mark.parametrize("kind", ["code", "feature"])
def test_public_kd_loss_and_student_gradient_scale_with_outer_omega(kind):
    measured = []
    for omega in (1.0, 0.1, 0.0):
        function, teacher, target, student, classes, weights = _kd_inputs(kind)
        loss, stats = function(teacher, target, classes, weights, omega=omega)
        loss.backward()
        assert teacher.grad is None and weights.grad is None
        assert stats["omega"] == omega
        assert stats["used_classes"] == [0, 1]
        assert stats["zero_weight_classes"] == [2]
        measured.append((loss.detach(), student.grad.clone()))
    loss, grad = measured[0]
    assert loss > 0 and torch.linalg.vector_norm(grad) > 0
    for index, omega in ((1, 0.1), (2, 0.0)):
        torch.testing.assert_close(measured[index][0], loss * omega, rtol=1e-6, atol=0)
        torch.testing.assert_close(measured[index][1], grad * omega, rtol=1e-6, atol=0)


@pytest.mark.parametrize("kind", ["code", "feature"])
def test_public_kd_zero_cell_support_is_finite_graph_connected_zero(kind):
    function, teacher, target, student, classes, weights = _kd_inputs(kind)
    loss, stats = function(teacher, target, classes, torch.zeros_like(weights), omega=0.1)
    assert torch.isfinite(loss) and loss.item() == 0 and loss.grad_fn is not None
    assert stats["used_classes"] == []
    assert stats["zero_weight_classes"] == [0, 1, 2]
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


@pytest.mark.parametrize("kind", ["code", "feature"])
def test_public_kd_tiny_cell_support_does_not_create_epsilon_gate(kind):
    function, teacher, target, student, classes, weights = _kd_inputs(kind)
    normal, _ = function(teacher, target, classes, weights, omega=0.1)
    tiny, _ = function(teacher, target, classes, weights * 1e-100, omega=0.1)
    normal_grad = torch.autograd.grad(normal, student, retain_graph=True)[0]
    tiny_grad = torch.autograd.grad(tiny, student)[0]
    torch.testing.assert_close(tiny, normal, rtol=1e-6, atol=0)
    torch.testing.assert_close(tiny_grad, normal_grad, rtol=1e-6, atol=0)


def test_codekd_balances_each_frame_before_averaging_batch():
    teacher = torch.tensor([[[[[1., 1.]]], [[[0., 0.]]]]]).repeat(2, 1, 1, 1, 1)
    student = torch.tensor([
        [[[[0.5, 0.5]]], [[[0.5, 0.5]]]],
        [[[[0.25, 0.125]]], [[[0.75, 0.875]]]],
    ], requires_grad=True)
    classes = torch.tensor([[[[0, 0]]], [[[0, 1]]]])
    weights = torch.ones_like(classes, dtype=torch.float32)
    loss, stats = codekd_loss(teacher, student, classes, weights, omega=0.1)
    expected = 0.1 * (math.log(2) + (math.log(4) + math.log(8)) / 2) / 2
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    assert [frame["active_classids"] for frame in stats["frames"]] == [[0], [0, 1]]


@pytest.mark.parametrize("kind", ["code", "feature"])
def test_public_kd_zero_frame_remains_in_frame_average(kind):
    function, teacher, target, student, classes, weights = _kd_inputs(kind)
    single, _ = function(teacher, target, classes, weights, omega=0.1)
    batch, stats = function(
        teacher.repeat(2, 1, 1, 1, 1), target.repeat(2, 1, 1, 1, 1),
        classes.repeat(2, 1, 1, 1), torch.cat([weights, torch.zeros_like(weights)]),
        omega=0.1,
    )
    torch.testing.assert_close(batch, single / 2, rtol=1e-6, atol=0)
    assert stats["frames"][1]["active_classids"] == []


@pytest.mark.parametrize("kind", ["code", "feature"])
@pytest.mark.parametrize("omega", [-0.1, float("nan"), float("inf")])
def test_public_kd_rejects_invalid_patient_multiplier(kind, omega):
    function, teacher, target, _student, classes, weights = _kd_inputs(kind)
    with pytest.raises(ValueError, match="omega"):
        function(teacher, target, classes, weights, omega=omega)
