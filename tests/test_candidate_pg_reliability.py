from __future__ import annotations

import torch
import pytest

from self_audit.candidate_pg.losses import weighted_ce, weighted_dice
from self_audit.candidate_pg.reliability import (
    ClassAdaptiveThreshold,
    boundary_band,
    class_adaptive_threshold_update,
    voxel_weights,
)
from self_audit.candidate_pg.training_step import _pg_loss


torch.set_num_threads(1)


def _padded_probabilities(*, requires_grad: bool = False):
    """One valid and one center-padded cell, with no fabricated pad mass."""
    shape = (1, 4, 1, 1, 2)
    p_w = torch.zeros(shape, dtype=torch.float32)
    p_t = torch.zeros(shape, dtype=torch.float32)
    p_s = torch.zeros(shape, dtype=torch.float32)
    p_w[..., 0] = torch.tensor([.05, .80, .10, .05]).reshape(1, 4, 1, 1)
    p_t[..., 0] = torch.tensor([.10, .70, .10, .10]).reshape(1, 4, 1, 1)
    p_s[..., 0] = torch.tensor([.10, .60, .15, .15]).reshape(1, 4, 1, 1)
    if requires_grad:
        p_w.requires_grad_()
        p_t.requires_grad_()
        p_s.requires_grad_()
    valid = torch.tensor([[[[True, False]]]])
    return p_w, p_t, p_s, valid


def test_class_adaptive_threshold_update_hand_computation_and_resume():
    probabilities = torch.tensor([[[[.8, .2]], [[.2, .8]]]], dtype=torch.float32)
    valid = torch.tensor([[[True, True]]])
    tau_g, p_tilde, tau_c = class_adaptive_threshold_update(
        .5, torch.tensor([.5, .5]), probabilities, valid, ema=.5, clip=(.0, 1.0)
    )
    assert float(tau_g) == pytest.approx(.65)
    torch.testing.assert_close(p_tilde, torch.tensor([.5, .5]))
    torch.testing.assert_close(tau_c, torch.tensor([.65, .65]))
    state = ClassAdaptiveThreshold(2, ema=.5, clip=(0.0, 1.0))
    snapshot = state.state_dict()
    state.update(probabilities, valid)
    resumed = ClassAdaptiveThreshold(2, ema=.5, clip=(0.0, 1.0))
    resumed.load_state_dict(snapshot)
    assert float(resumed.tau_g) == pytest.approx(1 / 2)
    resumed.update(probabilities, valid)
    assert torch.equal(state.tau_g, resumed.tau_g)
    assert torch.equal(state.p_tilde, resumed.p_tilde)


def test_boundary_band_is_strict_normalized_sdf_band():
    sdf = torch.tensor([[[[[.5, .49]]], [[[.6, .7]]]]])
    band = boundary_band(sdf, radius_mm=5, near_mm=2.5)
    assert torch.equal(band, torch.tensor([[[[False, True]]]]))


def test_voxel_weights_hand_formula_and_detachment():
    fusion = torch.tensor([[[[[.2, .4]]], [[[.8, .6]]]]])
    teacher = torch.tensor([[[[[.8, .4]]], [[[.2, .6]]]]], requires_grad=True)
    student = torch.tensor([[[[[.5, .9]]], [[[.5, .1]]]]])
    sdf = torch.tensor([[[[[.0, .6]]], [[[.6, .6]]]]])
    valid = torch.ones((1, 1, 1, 2), dtype=torch.bool)
    weights = voxel_weights(fusion, teacher, student, valid, torch.tensor([.7, .7]), sdf)
    expected0 = (.2 * .5) ** .5 * .8  # fusion class-1 argmax, inside band
    expected1 = (.6 * .1) ** .5 * 0.0  # outside band and confidence below tau
    torch.testing.assert_close(weights, torch.tensor([[[[expected0, expected1]]]]))
    assert not weights.requires_grad


def test_voxel_weights_masks_zero_mass_invalid_padding_without_fabricating_support():
    fusion = torch.tensor([[[[[.8, 0.0]]], [[[.2, 0.0]]]]])
    teacher = torch.tensor([[[[[.5, 0.0]]], [[[.5, 0.0]]]]])
    student = torch.tensor([[[[[.25, 0.0]]], [[[.75, 0.0]]]]])
    valid = torch.tensor([[[[True, False]]]])
    weights = voxel_weights(fusion, teacher, student, valid, torch.tensor([.7, .7]))
    expected = (.5 * .25) ** .5
    torch.testing.assert_close(weights, torch.tensor([[[[expected, 0.0]]]]))

    bad_valid = fusion.clone()
    bad_valid[..., 0] = 0.0
    with pytest.raises(ValueError, match="positive class mass on valid support"):
        voxel_weights(bad_valid, teacher, student, valid, torch.tensor([.7, .7]))

    bad_negative = fusion.clone()
    bad_negative[..., 1] = -1.0
    with pytest.raises(ValueError, match="finite and nonnegative"):
        voxel_weights(bad_negative, teacher, student, valid, torch.tensor([.7, .7]))


def test_threshold_update_ignores_zero_mass_invalid_padding():
    probabilities = torch.tensor([[[[.8, 0.0]], [[.2, 0.0]]]])
    valid = torch.tensor([[[True, False]]])
    tau_g, p_tilde, tau_c = class_adaptive_threshold_update(
        .5, torch.tensor([.5, .5]), probabilities, valid, ema=.5, clip=(0.0, 1.0)
    )
    assert float(tau_g) == pytest.approx(.65)
    torch.testing.assert_close(p_tilde, torch.tensor([.65, .35]))
    torch.testing.assert_close(tau_c, torch.tensor([.65, .35]))


@pytest.mark.parametrize("field", ("pW", "pT", "pS"))
def test_voxel_weights_rejects_zero_mass_on_every_valid_probability_field(field):
    p_w, p_t, p_s, valid = _padded_probabilities()
    values = {"pW": p_w, "pT": p_t, "pS": p_s}
    values[field][..., 0] = 0.0
    with pytest.raises(ValueError, match="positive class mass on valid support"):
        voxel_weights(p_w, p_t, p_s, valid, torch.full((4,), .7))


@pytest.mark.parametrize("field", ("pW", "pT", "pS"))
@pytest.mark.parametrize("cell", (0, 1))
@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), -float("inf")))
def test_voxel_weights_rejects_nonfinite_probability_anywhere(field, cell, bad_value):
    p_w, p_t, p_s, valid = _padded_probabilities()
    values = {"pW": p_w, "pT": p_t, "pS": p_s}
    values[field][..., cell] = bad_value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        voxel_weights(p_w, p_t, p_s, valid, torch.full((4,), .7))


def test_pg_loss_masks_invalid_padding_and_preserves_valid_gradient_and_detached_targets():
    p_w, p_t, p_s, valid = _padded_probabilities(requires_grad=True)
    target = torch.tensor([[[[1, 0]]]], dtype=torch.long)
    packet = {
        "p_W": p_w,
        "y": target,
        "w": p_w.detach().amax(dim=1),
        "valid": valid,
    }
    logits = torch.tensor(
        [[[[[0.2, 8.0]]], [[[-0.4, -8.0]]], [[[0.7, 5.0]]], [[[-0.1, -5.0]]]]],
        requires_grad=True,
    )
    loss = _pg_loss(logits, packet, p_t, p_s, torch.full((4,), .7), None)
    grad = torch.autograd.grad(loss, (logits, p_w, p_t, p_s), allow_unused=True)

    baseline_packet = {
        "p_W": p_w.detach()[..., :1],
        "y": target[..., :1],
        "w": packet["w"][..., :1],
        "valid": valid[..., :1],
    }
    baseline_logits = logits.detach()[..., :1].clone().requires_grad_()
    baseline_loss = _pg_loss(
        baseline_logits,
        baseline_packet,
        p_t.detach()[..., :1],
        p_s.detach()[..., :1],
        torch.full((4,), .7),
        None,
    )
    baseline_grad = torch.autograd.grad(baseline_loss, baseline_logits)[0]

    torch.testing.assert_close(loss, baseline_loss)
    torch.testing.assert_close(grad[0][..., :1], baseline_grad)
    assert torch.equal(grad[0][..., 1:], torch.zeros_like(grad[0][..., 1:]))
    assert grad[1:] == (None, None, None)
    assert not packet["y"].requires_grad


def test_weighted_ce_and_dice_match_hand_computation():
    logits = torch.tensor([[[[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]], requires_grad=True)
    target = torch.tensor([[[1, 2]]])
    weights = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    valid = torch.ones_like(target, dtype=torch.bool)
    ce = weighted_ce(logits, target, weights, valid)
    assert float(ce.detach()) == pytest.approx(torch.log(torch.tensor(4.0)).item() * 1.5)
    probs = torch.tensor([[[[.1, .0]], [[.8, .3]], [[.1, .7]], [[.0, .0]]]])
    dice = weighted_dice(probs, target, weights, valid)
    d1 = 2 * (.8 * 1) / (.8 + .3 * 2 + 1)
    d2 = 2 * (2 * .7) / (.1 + 2 * .7 + 2)
    assert float(dice) == pytest.approx(1 - (d1 + d2) / 2)
    ce.backward()
    assert weights.grad is None
