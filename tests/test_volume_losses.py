import pytest
import torch
from torch.nn import functional as F

from self_audit.losses import supervised_loss
from self_audit.volume.losses import (
    combine_accumulated_losses,
    pseudo_volume_ce,
    pseudo_weight_ramp,
    supervised_volume_loss,
)


def _old(logits, labels):
    return supervised_loss(logits, labels)


def _independent_volume_mean_oracle(logits, labels, geometric_valid, label_valid):
    """Reference estimator written in flattened-voxel form.

    This deliberately does not call the production helpers (or reuse their
    axis tuples).  Each non-empty volume is flattened to ``N`` voxels, so a
    wrong reduction that leaves a spatial axis cannot agree by construction.
    """
    values = []
    for index in range(logits.shape[0]):
        supported = geometric_valid[index] & label_valid[index] & (labels[index] != 255)
        if not bool(supported.any()):
            continue
        volume_logits = logits[index].permute(1, 2, 3, 0).reshape(-1, logits.shape[1])
        volume_labels = labels[index].reshape(-1).clone()
        volume_labels[~supported.reshape(-1)] = 255
        ce = F.cross_entropy(volume_logits, volume_labels, ignore_index=255)

        probabilities = volume_logits.softmax(dim=1).transpose(0, 1)
        encoded = F.one_hot(volume_labels.clamp_max(3), 4).transpose(0, 1).to(logits.dtype)
        valid = volume_labels != 255
        probabilities = probabilities * valid
        encoded = encoded * valid
        denominator = probabilities.sum(dim=1) + encoded.sum(dim=1)
        dice = (2 * (probabilities * encoded).sum(dim=1) + 1e-5) / (denominator + 1e-5)
        present = encoded[1:].sum(dim=1) > 0
        dice_loss = (1 - dice[1:][present]).mean() if bool(present.any()) else volume_logits.sum() * 0
        values.append(ce + dice_loss)
    if not values:
        raise ValueError("empty labeled support")
    return torch.stack(values).mean()


def test_legacy_pooled_matches_existing_loss_when_all_volume_support_is_valid():
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 3, 5, 7, requires_grad=True)
    labels = torch.randint(0, 4, (2, 3, 5, 7), dtype=torch.long)
    volume_loss = supervised_volume_loss(
        logits, labels, torch.ones_like(labels, dtype=torch.bool), torch.ones_like(labels, dtype=torch.bool)
    )
    reference_logits = logits.detach().permute(0, 2, 1, 3, 4).reshape(-1, 4, 5, 7).requires_grad_()
    reference_labels = labels.reshape(-1, 5, 7)
    reference = _old(reference_logits, reference_labels)
    assert torch.allclose(volume_loss.detach(), reference.detach(), atol=1e-7, rtol=1e-6)
    volume_grad = torch.autograd.grad(volume_loss, logits, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference, reference_logits)[0].reshape(2, 3, 4, 5, 7).permute(0, 2, 1, 3, 4)
    assert torch.allclose(volume_grad, reference_grad, atol=1e-6, rtol=1e-5)


def test_padding_and_unknown_are_excluded_but_absent_foreground_still_has_ce():
    logits = torch.zeros(1, 4, 2, 3, 3, requires_grad=True)
    labels = torch.zeros(1, 2, 3, 3, dtype=torch.long)
    labels[:, 0, 0, 0] = 255
    geometric = torch.ones_like(labels, dtype=torch.bool)
    geometric[:, 1, 2, 2] = False
    loss, metrics = supervised_volume_loss(logits, labels, geometric, torch.ones_like(geometric), return_metrics=True)
    assert torch.isfinite(loss)
    assert int(metrics["support_voxels"]) == 16
    assert int(metrics["present_foreground"]) == 0
    assert int(metrics["absent_foreground"]) == 3
    loss.backward()
    # An all-background target still yields a nonzero CE gradient on foreground
    # classes; the test prevents accidentally dropping absent-class negatives.
    assert logits.grad[:, 1:].abs().sum() > 0


def test_volume_mean_accumulation_matches_complete_logical_batch_gradient():
    torch.manual_seed(11)
    logits = torch.randn(3, 4, 2, 3, 3)
    labels = torch.randint(0, 4, (3, 2, 3, 3), dtype=torch.long)
    valid = torch.ones_like(labels, dtype=torch.bool)
    full = logits.clone().requires_grad_()
    full_loss = supervised_volume_loss(full, labels, valid, valid, mode="volume_mean")
    full_grad = torch.autograd.grad(full_loss, full)[0]
    first = logits[:1].clone().requires_grad_()
    second = logits[1:].clone().requires_grad_()
    first_loss = supervised_volume_loss(first, labels[:1], valid[:1], valid[:1], mode="volume_mean")
    second_loss = supervised_volume_loss(second, labels[1:], valid[1:], valid[1:], mode="volume_mean")
    accumulated = combine_accumulated_losses([first_loss, second_loss], [1, 2], mode="volume_mean")
    first_grad, second_grad = torch.autograd.grad(accumulated, [first, second])
    assert torch.allclose(accumulated.detach(), full_loss.detach(), atol=1e-6)
    assert torch.allclose(first_grad, full_grad[:1], atol=1e-6, rtol=1e-5)
    assert torch.allclose(second_grad, full_grad[1:], atol=1e-6, rtol=1e-5)


def test_volume_mean_matches_independent_flattened_volume_oracle_value_gradient_and_counts():
    """Guard the B,C,Z,H,W reduction and per-volume presence accounting."""
    torch.manual_seed(11)
    logits = torch.randn(2, 4, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 4, (2, 3, 4, 5), dtype=torch.long)
    geometric = torch.ones_like(labels, dtype=torch.bool)
    label_valid = torch.ones_like(labels, dtype=torch.bool)
    geometric[0, 0, 0, 0] = False
    label_valid[1, 2, 3, 4] = False

    expected = _independent_volume_mean_oracle(logits, labels, geometric, label_valid)
    actual, metrics = supervised_volume_loss(
        logits, labels, geometric, label_valid, mode="volume_mean", return_metrics=True
    )
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert int(metrics["support_voxels"]) == 118
    assert int(metrics["volumes_used"]) == 2
    assert int(metrics["present_foreground"]) == 6
    assert int(metrics["absent_foreground"]) == 0
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    assert torch.isfinite(actual_grad).all()
    assert torch.allclose(actual_grad, expected_grad, atol=1e-12, rtol=1e-11)


def test_volume_mean_skips_empty_volume_and_accumulation_uses_supported_volume_weights():
    torch.manual_seed(23)
    logits = torch.randn(3, 4, 2, 3, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.zeros(3, 2, 3, 5, dtype=torch.long)
    labels[1, :, 0, 0] = 1  # foreground class 1 present in one supported volume
    labels[2, :, 1, 1] = 2  # a different foreground class in another volume
    geometric = torch.ones_like(labels, dtype=torch.bool)
    label_valid = torch.ones_like(labels, dtype=torch.bool)
    # Volume 0 is entirely padding/unknown and must not dilute the mean.
    geometric[0] = False
    labels[0] = 255
    # Partial support and an explicit unknown voxel exercise both masks.
    geometric[1, 0, 0, 1] = False
    labels[2, 1, 2, 3] = 255

    complete, metrics = supervised_volume_loss(
        logits, labels, geometric, label_valid, mode="volume_mean", return_metrics=True
    )
    oracle = _independent_volume_mean_oracle(logits, labels, geometric, label_valid)
    assert torch.allclose(complete, oracle, atol=1e-12, rtol=1e-12)
    assert int(metrics["volumes_used"]) == 2
    assert int(metrics["present_foreground"]) == 2
    assert int(metrics["absent_foreground"]) == 4
    assert torch.isfinite(torch.autograd.grad(complete, logits, retain_graph=True)[0]).all()

    # The all-empty microbatch is omitted by the caller; supported volume
    # counts then make scalar accumulation equal the complete logical batch.
    first = supervised_volume_loss(
        logits[1:2], labels[1:2], geometric[1:2], label_valid[1:2], mode="volume_mean"
    )
    second = supervised_volume_loss(
        logits[2:3], labels[2:3], geometric[2:3], label_valid[2:3], mode="volume_mean"
    )
    accumulated = combine_accumulated_losses([first, second], [1, 1], mode="volume_mean")
    assert torch.allclose(accumulated, complete, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("case", ["all_unknown", "all_geometry_invalid", "all_label_invalid"])
def test_volume_mean_rejects_empty_support_cases(case):
    logits = torch.zeros(1, 4, 1, 2, 3, requires_grad=True)
    labels = torch.zeros(1, 1, 2, 3, dtype=torch.long)
    geometric = torch.ones_like(labels, dtype=torch.bool)
    label_valid = torch.ones_like(labels, dtype=torch.bool)
    if case == "all_unknown":
        labels.fill_(255)
    elif case == "all_geometry_invalid":
        geometric.fill_(False)
    else:
        label_valid.fill_(False)
    with pytest.raises(ValueError, match="empty labeled support"):
        supervised_volume_loss(logits, labels, geometric, label_valid, mode="volume_mean")


def test_legacy_pooled_accumulation_fails_closed_and_empty_support_fails():
    scalar = torch.tensor(1.0, requires_grad=True)
    with pytest.raises(ValueError, match="graph-retaining"):
        combine_accumulated_losses([scalar, scalar], [1, 1], mode="legacy_pooled")
    logits = torch.randn(1, 4, 1, 2, 2)
    labels = torch.full((1, 1, 2, 2), 255, dtype=torch.long)
    with pytest.raises(ValueError, match="empty labeled support"):
        supervised_volume_loss(logits, labels, torch.ones_like(labels, dtype=torch.bool), torch.ones_like(labels, dtype=torch.bool))


def test_pseudo_targets_detached_and_empty_acceptance_has_zero_grad():
    logits = torch.randn(1, 4, 2, 3, 3, requires_grad=True)
    targets = torch.zeros(1, 2, 3, 3, dtype=torch.long)
    accepted = torch.zeros_like(targets, dtype=torch.bool)
    loss, metrics = pseudo_volume_ce(logits, targets, torch.ones_like(accepted), accepted, weight=2.0, return_metrics=True)
    assert int(metrics["accepted_voxels"]) == 0 and bool(metrics["empty"])
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() == 0
    assert pseudo_weight_ramp(0, base_weight=1.0, ramp_steps=3, pseudo_delay=2) == 0
    assert pseudo_weight_ramp(2, base_weight=1.0, ramp_steps=3, pseudo_delay=2) == pytest.approx(1 / 3)
    assert pseudo_weight_ramp(10, base_weight=1.0, ramp_steps=3, pseudo_delay=2) == 1.0


def test_low_precision_large_volume_uses_finite_fp32_dice_accumulators():
    # 4 * 128 * 128 = 65,536 supported voxels; half-precision reductions can
    # overflow to inf/NaN if the Dice sums are not promoted before reduction.
    logits = torch.zeros(1, 4, 4, 128, 128, dtype=torch.float16, requires_grad=True)
    labels = torch.ones(1, 4, 128, 128, dtype=torch.long)
    valid = torch.ones_like(labels, dtype=torch.bool)
    loss = supervised_volume_loss(logits, labels, valid, valid, mode="volume_mean")
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
