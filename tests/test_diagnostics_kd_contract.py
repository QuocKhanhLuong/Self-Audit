"""Independent regression against the canonical compact distillation update."""
import pytest
import copy
import random
import numpy as np
import torch
from torch import nn

from self_audit.compact_control import build_class_candidates
from self_audit.gradient_bridge import KDLossContext, compute_gradient_arms


@pytest.mark.parametrize("temperature", [1.0, 2.3])
def test_diagnostic_kd_gradient_matches_canonical_plain_kd(temperature):
    # Compare with the actual trainer's four-class displacement builder, not a
    # second hand-written soft-CE formula. Masks and confidence are nonuniform.
    generator = torch.Generator().manual_seed(372)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        model = nn.Conv2d(3, 4, 1).eval()
    x = torch.randn(2, 3, 5, 7, generator=generator)
    q = torch.softmax(torch.randn(2, 4, 5, 7, generator=generator) * 2, dim=1)
    valid = torch.rand(2, 5, 7, generator=generator) > 0.2
    lr, coefficient, confidence = 0.013, 0.7, 0.55
    result = build_class_candidates(
        model, x, q, valid, lr=lr, temperature=temperature,
        coefficient=coefficient, confidence=confidence,
    )
    objective = KDLossContext(
        q, valid, temperature=temperature, coefficient=coefficient,
        confidence=confidence, denominator="valid_pixel_count",
    )
    gradients = torch.autograd.grad(objective(model(x)), tuple(model.parameters()))
    assert objective.denominator == result["denominator"] > 0
    assert len(result["directions"]) == 4
    for (name, parameter), gradient in zip(model.named_parameters(), gradients):
        canonical = sum(direction[name] for direction in result["directions"])
        torch.testing.assert_close(canonical, -lr * gradient, atol=2e-8, rtol=2e-6)
        assert parameter.grad is None


def _same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    else:
        assert left == right


def test_feature_diagnostics_preserve_bn_optimizer_scheduler_and_all_rng():
    generator = torch.Generator().manual_seed(19)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(991)
        student = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 4, 1)).eval()
        teacher_suffix = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 4, 1)).eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
    h = torch.randn(2, 3, 3, 4, generator=generator)
    student(h).square().mean().backward()
    optimizer.step()
    scheduler.step()
    # Preserve existing grads too: diagnostics must not zero or accumulate them.
    for module in (student, teacher_suffix):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    states = [copy.deepcopy(x.state_dict()) for x in (student, teacher_suffix, optimizer, scheduler)]
    grads = [None if p.grad is None else p.grad.clone() for p in student.parameters()]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    q = torch.softmax(torch.randn(2, 4, 3, 4, generator=generator), dim=1)
    kd = KDLossContext(q, torch.ones(2, 3, 4, dtype=torch.bool), temperature=1.8)
    first = compute_gradient_arms(
        h, student, teacher_suffix, kd, rho=0.02, seed=801,
        sample_ids=("fixture/p1/z0", "fixture/p2/z0"),
        frozen_modules=(student, teacher_suffix),
    )
    second = compute_gradient_arms(
        h, student, teacher_suffix, kd, rho=0.02, seed=801,
        sample_ids=("fixture/p1/z0", "fixture/p2/z0"),
        frozen_modules=(student, teacher_suffix),
    )
    for original, obj in zip(states, (student, teacher_suffix, optimizer, scheduler)):
        _same(original, obj.state_dict())
    for original, parameter in zip(grads, student.parameters()):
        _same(original, parameter.grad)
    _same(rng, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
    for arm in ("D0", "D1", "D2", "D3", "D4", "D5"):
        torch.testing.assert_close(first.logits[arm], second.logits[arm], atol=0, rtol=0)


def test_teacher_fit_tap_matches_actual_native_forward_hook():
    from self_audit.model import build_model
    from self_audit.diagnostic_models import extract_teacher_pre_refine0

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(182)
        teacher, _ = build_model({"backbone": "tiny_test"}, synthetic=True)
        teacher.eval()
        x = torch.rand(1, 3, 35, 41)
    captured = []
    handle = teacher.refine[0].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        with torch.no_grad():
            teacher(x)
    finally:
        handle.remove()
    assert len(captured) == 1
    extracted = extract_teacher_pre_refine0(teacher, x)
    torch.testing.assert_close(extracted, captured[0], atol=0, rtol=0)
