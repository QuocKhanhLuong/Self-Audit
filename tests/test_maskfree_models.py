"""Focused checks for the mask-free producer, students and losses (W4).

These are software checks on synthetic CPU tensors. They are evidence that the
implementation has the contracted shapes, budgets, gradient firewalls and
degenerate-case behaviour. They are not evidence about real data, anatomy or
label quality, and no real dataset exists in this checkout.

Run with the package importable under its canonical name, so that no second
copy of the module is created under a different top-level name::

    PYTHONPATH=src python3 -m pytest tests/test_maskfree_models.py -v
"""
from __future__ import annotations

import pytest
import torch

from self_audit_maskfree.losses import (
    MAX_SAMPLES_PER_IMAGE,
    _sample_support_locations,
    producer_loss,
    student_loss,
)
from self_audit_maskfree.models import (
    MAX_PRODUCER_PARAMETERS,
    MIN_CHANNELS_PER_GROUP,
    Producer,
    Student,
    _groups,
    count_parameters,
    make_models,
)


def _batch(batch: int = 2, height: int = 32, width: int = 32, *, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    context = torch.randn(batch, 3, height, width, generator=generator)
    support = torch.zeros(batch, height, width, dtype=torch.bool)
    support[:, : height * 3 // 4, :] = True
    context = context * support.unsqueeze(1)  # withheld pixels are physically zero
    return context, support


def test_shapes_parameter_budget_and_small_images():
    """Contracted shapes, a measured parameter budget, and tiny/odd inputs."""
    bundle = make_models(seed=42)
    producer, student = bundle["producer"], bundle["student_no_audit"]
    counts = bundle["parameter_counts"]

    assert counts["producer"] == count_parameters(producer)
    assert counts["producer"] <= MAX_PRODUCER_PARAMETERS
    assert counts["student_no_audit"] == counts["student_audited"]

    context, _ = _batch()
    out = producer(context)
    assert out["features"].shape == (2, 16, 32, 32)
    assert out["reconstruction"].shape == (2, 1, 32, 32)
    assert student(context).shape == (2, 4, 32, 32)

    # GroupNorm and the pooling ladder must survive small and non-square inputs.
    for height, width in ((5, 7), (1, 1), (3, 16)):
        tiny = torch.randn(1, 3, height, width)
        assert Producer()(tiny)["features"].shape == (1, 16, height, width)
        assert Student()(tiny).shape == (1, 4, height, width)


def test_students_start_identical_without_shared_storage():
    """Both arms start byte-identical and own separate storage."""
    bundle = make_models(seed=42)
    first, second = bundle["student_no_audit"], bundle["student_audited"]

    left = dict(first.state_dict())
    right = dict(second.state_dict())
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key

    pointers_first = {p.data_ptr() for p in first.parameters()}
    pointers_second = {p.data_ptr() for p in second.parameters()}
    assert not (pointers_first & pointers_second)

    with torch.no_grad():
        for param in first.parameters():
            param.add_(1.0)
    assert not torch.equal(
        first.state_dict()["head.weight"], second.state_dict()["head.weight"]
    )

    # Same seed reproduces the same initialization; a different seed does not.
    again = make_models(seed=42)["student_no_audit"].state_dict()
    other = make_models(seed=43)["student_no_audit"].state_dict()
    assert torch.equal(again["head.weight"], make_models(seed=42)["student_audited"].state_dict()["head.weight"])
    assert not torch.equal(again["head.weight"], other["head.weight"])


def test_producer_loss_is_finite_bounded_and_fit_only():
    """Finite gradients, bounded sampling, fit-only targets, all-invalid support."""
    producer = make_models(seed=42)["producer"]
    context, support = _batch(batch=2, height=32, width=32)
    generator = torch.Generator().manual_seed(7)

    loss, metrics = producer_loss(producer, context, support, generator=generator)
    assert torch.isfinite(loss)
    assert loss.requires_grad
    for arm in ("contrastive", "reconstruction", "equivariance"):
        assert torch.isfinite(metrics[arm])
    assert 0 < float(metrics["contrastive_samples"]) <= MAX_SAMPLES_PER_IMAGE * 2
    assert not bool(metrics["skipped"])
    assert torch.isfinite(metrics["feature_std_mean"])

    loss.backward()
    grads = [p.grad for p in producer.parameters() if p.grad is not None]
    assert grads, "producer received no gradients"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)

    # Sampling never leaves the fit support, which is how "targets only O_fit"
    # is enforced for the contrastive arm.
    rows, cols = _sample_support_locations(support[0], generator)
    assert rows.numel() <= MAX_SAMPLES_PER_IMAGE
    assert bool(support[0][rows, cols].all())

    # A completely withheld unit produces an explicit differentiable zero.
    producer.zero_grad(set_to_none=True)
    empty = torch.zeros(1, 32, 32, dtype=torch.bool)
    zero_loss, zero_metrics = producer_loss(
        producer, torch.zeros(1, 3, 32, 32), empty, generator=generator
    )
    assert bool(zero_metrics["skipped"])
    assert float(zero_loss.detach()) == 0.0
    zero_loss.backward()
    assert all(
        p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad))
        for p in producer.parameters()
    )


def test_student_loss_validity_weighting_and_all_invalid():
    """Valid-support denominator, explicit skip flag, and nan rejection."""
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    probabilities = torch.softmax(torch.randn(2, 4, 8, 8), dim=1)
    validity = torch.ones(2, 8, 8)

    loss, metrics = student_loss(logits, probabilities, validity)
    assert torch.isfinite(loss) and loss.requires_grad
    assert float(metrics["valid_support"]) == pytest.approx(128.0)
    assert not bool(metrics["skipped"])
    loss.backward()
    assert torch.isfinite(logits.grad).all()

    # Halving the validity of every pixel leaves the normalized loss unchanged,
    # because the denominator is the valid support, not the pixel count.
    half, half_metrics = student_loss(logits.detach(), probabilities, validity * 0.5)
    assert float(half) == pytest.approx(float(loss.detach()), rel=1e-5)
    assert float(half_metrics["valid_support"]) == pytest.approx(64.0)

    # All-invalid batch: exact differentiable zero, explicit flag, no nan.
    logits2 = torch.randn(2, 4, 8, 8, requires_grad=True)
    zero_loss, zero_metrics = student_loss(logits2, probabilities, torch.zeros(2, 8, 8))
    assert bool(zero_metrics["skipped"])
    assert float(zero_loss.detach()) == 0.0
    zero_loss.backward()
    assert torch.equal(logits2.grad, torch.zeros_like(logits2.grad))

    with pytest.raises(FloatingPointError):
        student_loss(torch.full((1, 4, 4, 4), float("nan")), probabilities[:1, :, :4, :4],
                     torch.ones(1, 4, 4))
    with pytest.raises(ValueError):
        student_loss(torch.randn(1, 4, 4, 4), torch.full((1, 4, 4, 4), float("nan")),
                     torch.ones(1, 4, 4))


def test_no_cross_arm_or_producer_gradients():
    """A student step touches neither the other arm nor the producer."""
    bundle = make_models(seed=42)
    producer = bundle["producer"]
    audited, no_audit = bundle["student_audited"], bundle["student_no_audit"]
    context, _ = _batch(batch=1, height=16, width=16)

    # Pseudo-labels are produced from the producer's own graph on purpose: the
    # loss must still detach them, otherwise the audited arm would train the
    # shared producer and the two arms would stop being comparable.
    features = producer(context)["features"]
    probabilities = torch.softmax(features[:, :4], dim=1)
    validity = torch.sigmoid(features[:, 4])
    assert probabilities.requires_grad and validity.requires_grad

    loss, _ = student_loss(audited(context), probabilities, validity)
    loss.backward()

    assert any(p.grad is not None for p in audited.parameters())
    assert all(p.grad is None for p in no_audit.parameters())
    assert all(p.grad is None for p in producer.parameters())


def test_revision_hidden_intensity_immunity_targets_and_narrow_width():
    """Root-review revision: fit-support immunity, target validation, width 4.

    One combined check for the three defects found in root review:

    1. ``producer_loss`` forwarded the whole context, so hidden intensities
       reached the model even though the docstring promised immunity;
    2. ``student_loss`` clamped and renormalized invalid pseudo-labels, which
       let an all-zero class vector on valid support contribute a silent zero;
    3. ``_groups`` allowed one channel per group, which at ``1x1`` has zero
       variance and collapses a narrow-width block to a constant.
    """
    # --- 1. mutating pixels outside the fit support changes nothing ----------
    producer = make_models(seed=42)["producer"]
    context, support = _batch(batch=2, height=32, width=32)
    assert bool((context * ~support.unsqueeze(1)).eq(0).all()), "fixture is not legal"

    clean_loss, clean_metrics = producer_loss(
        producer, context, support, generator=torch.Generator().manual_seed(11)
    )
    clean_loss.backward()
    clean_grads = [p.grad.detach().clone() for p in producer.parameters() if p.grad is not None]

    hidden = torch.randn(context.shape, generator=torch.Generator().manual_seed(99))
    polluted = torch.where(support.unsqueeze(1), context, hidden)
    assert not torch.equal(polluted, context)

    producer.zero_grad(set_to_none=True)
    dirty_loss, dirty_metrics = producer_loss(
        producer, polluted, support, generator=torch.Generator().manual_seed(11)
    )
    dirty_loss.backward()
    dirty_grads = [p.grad.detach().clone() for p in producer.parameters() if p.grad is not None]

    assert float(dirty_loss.detach()) == float(clean_loss.detach())
    assert len(dirty_grads) == len(clean_grads)
    assert all(torch.equal(a, b) for a, b in zip(clean_grads, dirty_grads))
    # The violation is reported, not absorbed.
    assert float(clean_metrics["hidden_intensity_elements"]) == 0.0
    assert float(dirty_metrics["hidden_intensity_elements"]) > 0.0

    # --- 2. invalid pseudo-labels are rejected, never repaired ---------------
    logits = torch.randn(1, 4, 4, 4)
    good = torch.softmax(torch.randn(1, 4, 4, 4), dim=1)
    ones = torch.ones(1, 4, 4)

    all_zero = torch.zeros(1, 4, 4, 4)
    with pytest.raises(ValueError, match="sum to 1"):
        student_loss(logits, all_zero, ones)
    with pytest.raises(ValueError, match="sum to 1"):
        student_loss(logits, good * 2.0 / 3.0, ones)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        student_loss(logits, good - 0.5, ones)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        student_loss(logits, good, ones * 1.5)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        student_loss(logits, good, ones * -1.0)

    # Zero-validity pixels stay unconstrained, and the legal all-invalid path is
    # unchanged: an entirely unlabelled unit is a skip, not an error.
    mixed = good.clone()
    mixed[:, :, 0, 0] = 0.0
    partial = ones.clone()
    partial[:, 0, 0] = 0.0
    mixed_loss, mixed_metrics = student_loss(logits, mixed, partial)
    assert torch.isfinite(mixed_loss)
    assert float(mixed_metrics["valid_support"]) == pytest.approx(15.0)

    zero_loss, zero_metrics = student_loss(logits, all_zero, torch.zeros(1, 4, 4))
    assert bool(zero_metrics["skipped"])
    assert float(zero_loss) == 0.0

    # --- 3. every GroupNorm keeps at least two channels per group ------------
    assert _groups(4) == 2 and _groups(8) == 4 and _groups(16) == 8
    for module in list(Producer(width=4).modules()) + list(Student(width=4).modules()):
        if isinstance(module, torch.nn.GroupNorm):
            assert module.num_channels // module.num_groups >= MIN_CHANNELS_PER_GROUP

    # With one channel per group the 1x1 variance is exactly zero, GroupNorm
    # emits zeros and the whole narrow network degenerates to a constant. Two
    # channels per group keep it input dependent. The output is still heavily
    # quantized at a single pixel, because normalizing two values can only
    # record which is larger, so the honest assertion is "not a constant
    # function", not "separates any two inputs".
    torch.manual_seed(0)
    narrow = Producer(width=4)
    outputs = [narrow(torch.randn(1, 3, 1, 1))["features"].detach() for _ in range(32)]
    assert outputs[0].shape == (1, 16, 1, 1)
    assert all(torch.isfinite(out).all() for out in outputs)
    distinct: list[torch.Tensor] = []
    for out in outputs:
        if not any(torch.allclose(out, seen) for seen in distinct):
            distinct.append(out)
    assert len(distinct) > 1, "width-4 producer is a constant function at 1x1"
