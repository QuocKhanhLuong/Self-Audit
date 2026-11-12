"""Regression coverage for the unweighted, per-cell public KD maps."""

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
    function = codekd_loss if kind == "code" else feature_kd_loss
    target = student.softmax(dim=1) if kind == "code" else student
    return function, teacher, target, student


@pytest.mark.parametrize("kind", ["code", "feature"])
def test_public_kd_map_preserves_cells_and_student_gradient(kind):
    function, teacher, target, student = _kd_inputs(kind)
    per_cell = function(teacher, target)

    assert per_cell.shape == (1, 1, 1, 4)
    assert torch.isfinite(per_cell).all()
    assert per_cell.dtype == torch.float64
    assert torch.count_nonzero(per_cell) > 0
    per_cell.sum().backward()

    assert teacher.grad is None
    assert student.grad is not None
    assert torch.linalg.vector_norm(student.grad) > 0


def test_codekd_map_matches_each_cell_kl_without_frame_or_class_reduction():
    teacher = torch.tensor(
        [
            [[[[0.8, 0.2]]], [[[0.2, 0.8]]]],
            [[[[0.9, 0.1]]], [[[0.1, 0.9]]]],
        ], dtype=torch.float64,
    )
    student = torch.tensor(
        [
            [[[[0.5, 0.1]]], [[[0.5, 0.9]]]],
            [[[[0.3, 0.7]]], [[[0.7, 0.3]]]],
        ], dtype=torch.float64, requires_grad=True,
    )
    actual = codekd_loss(teacher, student)
    p_t = teacher / teacher.sum(dim=1, keepdim=True)
    p_s = student / student.sum(dim=1, keepdim=True)
    expected = (p_t * (p_t.log() - p_s.log())).sum(dim=1)

    assert actual.shape == (2, 1, 1, 2)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual[0].mean(), actual[1].mean())


@pytest.mark.parametrize("kind", ["code", "feature"])
def test_public_kd_maps_accept_no_reducer_inputs(kind):
    function, teacher, target, _student = _kd_inputs(kind)
    # The two-tensor signature deliberately leaves cls, confidence weights,
    # cohort omega, and frame aggregation to the engine.
    assert function(teacher, target).shape == (1, 1, 1, 4)


def test_codekd_rejects_invalid_probability_mass():
    teacher = torch.full((1, 2, 1, 1, 1), 0.5)
    student = torch.zeros_like(teacher)
    with pytest.raises(ValueError, match="positive mass"):
        codekd_loss(teacher, student)
