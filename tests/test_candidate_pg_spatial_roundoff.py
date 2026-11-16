"""Regression for S2 update 1989's one-ULP bilinear interpolation overshoot."""

import copy

import pytest
import torch

from self_audit.candidate_fuse.augmentation import StrongInplaneAugment
from self_audit.candidate_pg.views import StatefulViewGenerator


def _inputs():
    image = torch.ones(1, 1, 1, 174, 208)
    valid = torch.ones(1, 1, 174, 208, dtype=torch.bool)
    geometry = {"shape": (1, 174, 208), "spacing_zyx": (10.0, 1.0, 1.0)}
    return image, valid, geometry


@pytest.mark.parametrize("batch,labeled", [(1, True), (1, False), (2, True), (2, False)])
def test_real_bilinear_roundoff_is_repaired_only_at_pg_weak_output(monkeypatch, batch, labeled):
    image, valid, geometry = _inputs()
    target = torch.arange(image.numel()).reshape_as(valid).remainder(4).long()
    matrix = torch.tensor([[0.7102487087249756, 0.703950822353363],
                           [-0.703950822353363, 0.7102487087249756]])
    raw, expected_target = StrongInplaneAugment._warp(image, target, matrix)
    _, expected_valid = StrongInplaneAugment._warp(image, valid.long(), matrix)
    assert float(raw.max()) == 1.0 + torch.finfo(torch.float32).eps
    assert int((raw > 1).sum()) > 0
    # The historical primitive and strict public strong-input guard stay intact.
    view = StatefulViewGenerator()
    with pytest.raises(ValueError, match="normalized"):
        view.strong(raw)

    def spatial(x, mask, geom):
        out, labels = StrongInplaneAugment._warp(x, mask, matrix)
        return out, labels, geom

    monkeypatch.setattr(view.augment, "spatial", spatial)
    image = image.expand(batch, -1, -1, -1, -1)
    valid = valid.expand(batch, -1, -1, -1)
    target = target.expand(batch, -1, -1, -1)
    result = view.joint_views(image, valid, geometry, target if labeled else None)
    if labeled:
        weak, strong, labels, support, geom = result
        assert torch.equal(labels, expected_target.expand_as(target))
    else:
        weak, strong, support, geom = result
    raw = raw.expand_as(image)
    assert torch.equal(weak, raw.clamp_max(1))
    assert torch.equal(weak[raw <= 1], raw[raw <= 1])
    assert torch.equal(support, expected_valid.bool().expand_as(valid))
    assert geom is geometry
    assert strong.shape == image.shape and strong.dtype == torch.float32
    assert torch.isfinite(strong).all() and strong.min() >= 0 and strong.max() <= 1
    assert torch.equal(image, torch.ones_like(image))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"),
                                  -torch.finfo(torch.float32).eps,
                                  1.0 + 2 * torch.finfo(torch.float32).eps, 1.01])
def test_invalid_spatial_output_still_fails_closed(monkeypatch, value):
    image, valid, geometry = _inputs()
    view = StatefulViewGenerator()

    def spatial(x, mask, geom):
        out = x.clone()
        out.flatten()[0] = value
        return out, mask, geom

    monkeypatch.setattr(view.augment, "spatial", spatial)
    with pytest.raises(ValueError, match="spatial"):
        view.weak(image, valid, geometry)


def test_one_ulp_invalid_native_input_is_not_repaired():
    image, valid, geometry = _inputs()
    image.flatten()[0] = 1.0 + torch.finfo(torch.float32).eps
    view = StatefulViewGenerator()
    with pytest.raises(ValueError, match="normalized"):
        view.weak(image, valid, geometry)


def test_in_range_outputs_rng_and_resume_are_unchanged():
    image, valid, geometry = _inputs()
    image.mul_(0.5)
    view = StatefulViewGenerator()
    before = copy.deepcopy(view.state_dict())
    global_rng = torch.get_rng_state()
    reference = StrongInplaneAugment(seed=view.seed)
    reference.load_state_dict(before["augmentation"])
    expected_weak, expected_valid, _ = reference.spatial(image, valid.long(), geometry)
    expected_strong = reference.intensity(expected_weak)
    actual = view.joint_views(image, valid, geometry)
    assert torch.equal(actual[0], expected_weak)
    assert torch.equal(actual[1], expected_strong)
    assert torch.equal(actual[2], expected_valid.bool())
    assert torch.equal(view.augment.generator.get_state(), reference.generator.get_state())
    assert torch.equal(view.cutmix_generator.get_state(), before["cutmix_generator_state"])
    assert torch.equal(torch.get_rng_state(), global_rng)
    after = copy.deepcopy(view.state_dict())
    view.load_state_dict(before)
    replay = view.joint_views(image, valid, geometry)
    assert all(torch.equal(a, b) for a, b in zip(actual[:3], replay[:3]))
    assert torch.equal(view.augment.generator.get_state(), after["augmentation"]["generator_state"])


def test_disabled_augmentation_remains_bitwise_identity():
    image, valid, geometry = _inputs()
    view = StatefulViewGenerator()
    view.augment.probabilities = dict.fromkeys(view.augment.probabilities, 0.0)
    before = view.augment.generator.get_state()
    weak, strong, support, geom = view.joint_views(image, valid, geometry)
    assert weak is image and torch.equal(strong, image)
    assert torch.equal(support, valid) and geom is geometry
    assert torch.equal(view.augment.generator.get_state(), before)
