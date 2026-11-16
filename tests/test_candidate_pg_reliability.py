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


torch.set_num_threads(1)


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
