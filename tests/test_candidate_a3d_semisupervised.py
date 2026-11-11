"""Focused software checks for the opt-in A-3D semi-supervised helpers."""

from __future__ import annotations

import torch
from torch import nn

from self_audit.candidate_a3d.semisupervised import (
    consistency_loss,
    make_ema,
    ramp_weight,
    strong_intensity,
    teacher_consistency_loss,
    update_ema,
)


class _TinyOutputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=1)
        self.sdf = nn.Conv3d(1, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.conv(x)
        return {"region_logits": logits, "boundary_logits": logits + 0.1, "sdf": self.sdf(x).tanh()}


def test_ema_is_frozen_and_updates_only_when_called() -> None:
    torch.manual_seed(4)
    online = _TinyOutputModel()
    ema = make_ema(online, decay=0.5)
    original = {name: value.detach().clone() for name, value in ema.named_parameters()}
    assert not ema.training
    assert all(not parameter.requires_grad for parameter in ema.parameters())

    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(2.0)
    for name, value in ema.named_parameters():
        assert torch.equal(value, original[name])
    update_ema(ema, online)
    for name, value in ema.named_parameters():
        assert torch.allclose(value, original[name] * 0.5 + dict(online.named_parameters())[name] * 0.5)
        assert not value.requires_grad


def test_intensity_transform_is_deterministic_and_keeps_shape() -> None:
    image = torch.linspace(0.0, 1.0, 2 * 1 * 3 * 4 * 5).reshape(2, 1, 3, 4, 5)
    config = {"semi_supervised": {"intensity_gamma": [0.8, 1.2], "intensity_noise_std": 0.02}}
    first = strong_intensity(image, config, generator=torch.Generator().manual_seed(91))
    second = strong_intensity(image, config, generator=torch.Generator().manual_seed(91))
    assert first.shape == image.shape
    assert torch.equal(first, second)
    assert bool((first >= 0).all()) and bool((first <= 1).all())


def test_consistency_has_student_gradient_and_downweights_background() -> None:
    student = torch.zeros((1, 4, 2, 3, 3), requires_grad=True)
    # One reliable foreground voxel and a background-dominated remainder.
    target = torch.full_like(student, -8.0)
    target[:, 0] = 8.0
    target[:, 2, 0, 1, 1] = 12.0
    loss, stats = consistency_loss(student, target, threshold=0.7, background_weight=0.1)
    assert torch.isfinite(loss)
    assert stats["foreground_coverage"] > 0.0
    loss.backward()
    assert student.grad is not None and bool(torch.isfinite(student.grad).all())
    assert float(student.grad.abs().sum()) > 0.0


def test_all_background_target_is_finite_and_reported_without_inventing_foreground() -> None:
    student = torch.randn((1, 4, 2, 3, 3), requires_grad=True)
    target = torch.zeros_like(student)
    target[:, 0] = 9.0
    loss, stats = consistency_loss(student, target, threshold=0.7, background_weight=0.1)
    assert torch.isfinite(loss)
    assert stats["foreground_coverage"] == 0.0
    assert stats["background_coverage"] == 1.0
    loss.backward()
    assert student.grad is not None and bool(torch.isfinite(student.grad).all())


def test_teacher_consistency_uses_soft_sdf_target_without_gradients() -> None:
    online = _TinyOutputModel()
    image = torch.rand((1, 1, 2, 3, 3))
    student_outputs = online(image)
    target_model = make_ema(online)
    target_outputs = target_model(image)
    loss, stats = teacher_consistency_loss(
        student_outputs,
        target_outputs,
        {"semi_supervised": {"confidence_threshold": 0.0, "background_weight": 0.1}},
    )
    assert torch.isfinite(loss)
    assert stats["region"]["selected_fraction"] == 1.0
    assert all(value.grad is None for value in target_model.parameters())
    loss.backward()
    assert any(value.grad is not None for value in online.parameters())


def test_ramp_weight_is_bounded() -> None:
    config = {"semi_supervised": {"rampup_epochs": 4}}
    assert ramp_weight(0, config) == 0.0
    assert ramp_weight(2, config) == 0.5
    assert ramp_weight(99, config) == 1.0
