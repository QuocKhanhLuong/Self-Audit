import pytest
import torch
from torch import nn

from self_audit.gradient_bridge import (
    InvalidGradientDiagnostic,
    KDLossContext,
    compute_gradient_arms,
)


class PointwiseHead(nn.Module):
    def __init__(self, channels=3, classes=4, scale=1.0):
        super().__init__()
        self.conv = nn.Conv2d(channels, classes, 1, bias=True)
        with torch.no_grad():
            values = torch.arange(classes * channels, dtype=torch.float32).reshape(classes, channels, 1, 1)
            self.conv.weight.copy_((values + 1) * (0.013 * scale))
            self.conv.bias.copy_(torch.linspace(-0.2, 0.2, classes) * scale)
        self.register_buffer("frozen_offset", torch.linspace(-0.03, 0.03, classes))
        self.eval()

    def forward(self, h):
        return self.conv(h) + self.frozen_offset[None, :, None, None]


def _snapshot(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _assert_state_equal(before, module):
    after = module.state_dict()
    assert before.keys() == after.keys()
    for key in before:
        assert torch.equal(before[key], after[key]), key


def test_exact_current_kd_vjps_shared_weighted_mask_denominator_and_native_outputs():
    torch.manual_seed(71)
    h = torch.randn(2, 3, 3, 4)
    teacher = torch.softmax(torch.randn(2, 4, 3, 4), dim=1)
    valid = torch.tensor(
        [[[1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 0]],
         [[1, 0, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1]]], dtype=torch.bool,
    )
    weights = torch.tensor(
        [[[1.0, 0.3, 8.0, 0.7], [1.2, 2.0, 0.8, 1.4], [0.5, 1.1, 0.9, 4.0]],
         [[0.4, 3.0, 1.5, 0.9], [1.8, 0.6, 0.2, 7.0], [2.0, 0.7, 1.0, 0.5]]],
    )
    kd = KDLossContext(teacher, valid, temperature=2.3, coefficient=0.7,
                       confidence=0.42, class_weights=(0.5, 1.0, 1.5, 2.0),
                       denominator="valid_weight_sum", pixel_weights=weights)
    student, bridge = PointwiseHead(scale=0.8), PointwiseHead(scale=-1.1)
    for module in (student, bridge):
        for parameter in module.parameters():
            parameter.grad = torch.full_like(parameter, 0.125)
    student_grad_before = {name: parameter.grad.clone() for name, parameter in student.named_parameters()}
    bridge_grad_before = {name: parameter.grad.clone() for name, parameter in bridge.named_parameters()}
    student_state, bridge_state = _snapshot(student), _snapshot(bridge)
    rng_before = torch.get_rng_state().clone()

    result = compute_gradient_arms(
        h,
        student,
        bridge,
        kd,
        feature_mask=torch.tensor([[[1, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1]],
                                   [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1]]], dtype=torch.bool),
        rho=0.04,
        seed=902,
        sample_ids=("p01/t0/z1", "p02/t1/z4"),
        frozen_modules=(student, bridge),
    )

    expected_denom = float((kd.pixel_weights * kd.confidence_mask).double().sum().item())
    assert result.denominator == expected_denom
    assert result.gradient_sources["D3"] == "valid_coordinate_shuffle(D2)"
    assert torch.allclose(result.gradient_norms["D2"], result.gradient_norms["D3"], atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.displacement_norms["D2"], result.displacement_norms["D3"], atol=1e-8, rtol=1e-6)
    assert result.logits["D0"].shape == teacher.shape
    assert result.forward_counts == {"student_model": 6, "bridge_model": 6, "kd_objective": 12}
    assert result.forward_count == 12
    assert result.backward_counts == {
        "autograd_grad_calls": 8,
        "cotangent_vjps": 2,
        "direct_kd_gradient_checks": 2,
        "native_vjps": 2,
        "cross_vjps": 2,
    }
    assert result.backward_count == 8
    for name in ("D0", "D1", "D2", "D3", "D4", "D5"):
        expected_logits = student(h + result.deltas[name])
        assert torch.allclose(result.logits[name], expected_logits, atol=0, rtol=0)
        expected_student_kd = kd(expected_logits)
        expected_bridge_kd = kd(bridge(h + result.deltas[name]))
        assert torch.allclose(result.per_arm_kd_losses[name]["student"], expected_student_kd)
        assert torch.allclose(result.per_arm_kd_losses[name]["bridge"], expected_bridge_kd)
    assert torch.equal(result.kd_losses["student"], result.per_arm_kd_losses["D0"]["student"])
    assert torch.equal(result.kd_losses["bridge"], result.per_arm_kd_losses["D0"]["bridge"])
    for name in ("D1", "D2", "D3", "D4", "D5"):
        invalid = ~torch.tensor([[[1, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1]],
                                 [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1]]], dtype=torch.bool)
        assert torch.count_nonzero(result.deltas[name] * invalid[:, None]) == 0

    # The VJPs are direct autograd derivatives of the actual temperature-scaled
    # soft-CE objective, not a hand-written p-q substitute.
    h_ref = h.detach().clone().requires_grad_(True)
    z_s = student(h_ref)
    z_b = bridge(h_ref)
    loss_s, loss_b = kd(z_s), kd(z_b)
    direct_s = torch.autograd.grad(loss_s, h_ref, retain_graph=True)[0]
    direct_b = torch.autograd.grad(loss_b, h_ref, retain_graph=True)[0]
    u_s = torch.autograd.grad(loss_s, z_s, retain_graph=True)[0]
    u_b = torch.autograd.grad(loss_b, z_b, retain_graph=True)[0]
    output_vjp_s = torch.autograd.grad(z_s, h_ref, grad_outputs=u_s, retain_graph=True)[0]
    output_vjp_b = torch.autograd.grad(z_b, h_ref, grad_outputs=u_b, retain_graph=True)[0]
    direct_d4 = torch.autograd.grad(z_b, h_ref, grad_outputs=u_s, retain_graph=True)[0]
    direct_d5 = torch.autograd.grad(z_s, h_ref, grad_outputs=u_b)[0]
    assert torch.allclose(output_vjp_s, direct_s, atol=1e-7, rtol=1e-6)
    assert torch.allclose(output_vjp_b, direct_b, atol=1e-7, rtol=1e-6)
    for name, expected in (("D1", output_vjp_s), ("D2", output_vjp_b), ("D4", direct_d4), ("D5", direct_d5)):
        assert torch.allclose(result.gradients[name], expected.detach(), atol=1e-7, rtol=1e-6)

    assert torch.equal(rng_before, torch.get_rng_state())
    _assert_state_equal(student_state, student)
    _assert_state_equal(bridge_state, bridge)
    for name, parameter in student.named_parameters():
        assert torch.equal(parameter.grad, student_grad_before[name])
    for name, parameter in bridge.named_parameters():
        assert torch.equal(parameter.grad, bridge_grad_before[name])

    repeated = compute_gradient_arms(
        h, student, bridge, kd,
        feature_mask=torch.tensor([[[1, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1]],
                                   [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1]]], dtype=torch.bool),
        rho=0.04, seed=902, sample_ids=("p01/t0/z1", "p02/t1/z4"),
        frozen_modules=(student, bridge),
    )
    assert torch.equal(result.deltas["D3"], repeated.deltas["D3"])
    assert torch.equal(rng_before, torch.get_rng_state())


def test_equal_jacobians_with_opposite_output_cotangents_are_distinguished():
    h = torch.full((1, 1, 1, 1), 0.5)
    target = torch.full((1, 2, 1, 1), 0.5)
    kd = KDLossContext(target, torch.ones((1, 1, 1), dtype=torch.bool), temperature=1.7)

    def logits_with_offset(offset):
        def call(value):
            scalar = value[:, :1] + offset
            return torch.cat((scalar, -scalar), dim=1)
        return call

    with torch.no_grad():
        result = compute_gradient_arms(
            h,
            logits_with_offset(2.5),
            logits_with_offset(-3.5),
            kd,
            rho=0.1,
            seed=19,
            sample_ids=("patient-a-sample",),
            frozen_modules=(nn.Identity().eval(),),
        )
    # Both output maps have the same Jacobian [1,-1]. The logits are on
    # opposite sides of q, so current-KD cotangents and the crossed VJPs flip.
    assert torch.allclose(result.gradients["D1"], -result.gradients["D2"], atol=1e-7, rtol=1e-6)
    assert torch.allclose(result.gradients["D4"], result.gradients["D1"], atol=1e-7, rtol=1e-6)
    assert torch.allclose(result.gradients["D5"], result.gradients["D2"], atol=1e-7, rtol=1e-6)
    assert result.statuses["D3"] == result.statuses["D2"] == ("active",)
    assert result.shuffle_modes["D3"] == ("identity_noop_single_scalar",)
    assert torch.equal(result.gradients["D3"], result.gradients["D2"])
    assert torch.equal(result.deltas["D3"], result.deltas["D2"])
    assert torch.equal(result.gradient_norms["D3"], result.gradient_norms["D2"])
    assert torch.equal(result.displacement_norms["D3"], result.displacement_norms["D2"])


def test_d3_single_spatial_coordinate_shuffles_channels_without_losing_d2_norm():
    h = torch.tensor([[[[0.7]], [[-0.3]]]])
    target = torch.tensor([[[[0.9]], [[0.1]]]])
    kd = KDLossContext(target, torch.ones((1, 1, 1), dtype=torch.bool), temperature=2.1)

    def student(value):
        scalar = value[:, :1] * 1.7 + value[:, 1:2] * 0.2
        return torch.cat((scalar, -scalar), dim=1)

    def bridge(value):
        scalar = value[:, :1] * -0.4 + value[:, 1:2] * 2.3 + 0.7
        return torch.cat((scalar, -scalar), dim=1)

    result = compute_gradient_arms(
        h, student, bridge, kd, feature_mask=torch.ones((1, 1, 1), dtype=torch.bool),
        rho=0.08, seed=41, sample_ids=("one-coordinate",), frozen_modules=(nn.Identity().eval(),),
    )
    assert result.statuses["D2"] == result.statuses["D3"] == ("active",)
    assert result.shuffle_modes["D3"] == ("shuffled",)
    assert torch.allclose(result.gradient_norms["D2"], result.gradient_norms["D3"], atol=0, rtol=1e-12)
    assert torch.allclose(result.displacement_norms["D2"], result.displacement_norms["D3"], atol=1e-8, rtol=1e-6)


def test_near_zero_is_per_sample_fallback_and_empty_support_stays_in_cohort():
    h = torch.zeros(2, 1, 1, 3)
    h[0].fill_(1.0)
    p = torch.softmax(torch.tensor([1.0, -1.0]), dim=0)
    target = torch.tensor([[[[float(p[0]), float(p[0]), float(p[0])]],
                            [[float(p[1]), float(p[1]), float(p[1])]]],
                           [[[0.9, 0.9, 0.9]], [[0.1, 0.1, 0.1]]]])
    valid = torch.ones((2, 1, 3), dtype=torch.bool)
    kd = KDLossContext(target, valid)

    def head(value):
        scalar = value[:, :1]
        return torch.cat((scalar, -scalar), dim=1)

    feature_mask = torch.tensor([[[True, True, True]], [[False, False, False]]])
    # Sample 0 has exact zero KD cotangent; sample 1 has empty feature support.
    result = compute_gradient_arms(
        h, head, head, kd, feature_mask=feature_mask, rho=0.2, seed=8,
        sample_ids=("p-a", "p-b"), min_gradient_norm=1e-6,
        frozen_modules=(nn.Identity().eval(),),
    )
    assert result.statuses["D1"][0] == "near_zero_gradient"
    assert result.statuses["D1"][1] == "empty_support"
    assert torch.count_nonzero(result.deltas["D1"]) == 0
    assert result.statuses["D2"][0] == "near_zero_gradient"
    assert result.statuses["D2"][1] == "empty_support"
    assert torch.count_nonzero(result.deltas["D2"]) == 0
    assert result.logits["D0"].shape[0] == 2


def test_nonfinite_logits_invalidate_instead_of_becoming_zero_fallback():
    h = torch.ones(1, 2, 1, 2)
    q = torch.full((1, 2, 1, 2), 0.5)
    kd = KDLossContext(q, torch.ones((1, 1, 2), dtype=torch.bool))

    def student(value):
        return value[:, :1].repeat(1, 2, 1, 1)

    def nonfinite_bridge(value):
        return value[:, :1].repeat(1, 2, 1, 1) * torch.tensor(float("nan"))

    with pytest.raises(InvalidGradientDiagnostic, match="non-finite bridge logits"):
        compute_gradient_arms(h, student, nonfinite_bridge, kd, rho=0.1, seed=4,
                             sample_ids=("patient-1/source-1",), frozen_modules=(nn.Identity().eval(),))


def test_duplicate_sample_ids_are_rejected_for_deterministic_shuffle():
    h = torch.randn(2, 1, 1, 2)
    q = torch.full((2, 2, 1, 2), 0.5)
    kd = KDLossContext(q, torch.ones((2, 1, 2), dtype=torch.bool))
    with pytest.raises(ValueError, match="unique"):
        compute_gradient_arms(h, lambda z: z[:, :1].repeat(1, 2, 1, 1),
                              lambda z: z[:, :1].repeat(1, 2, 1, 1), kd,
                              rho=0.1, seed=1, sample_ids=("same", "same"))
