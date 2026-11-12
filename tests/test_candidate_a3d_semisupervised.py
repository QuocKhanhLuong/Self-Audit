"""Focused software checks for the opt-in A-3D semi-supervised helpers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from self_audit.candidate_a3d.bank import BankConfig3D, QueryBank3D
from self_audit.candidate_a3d.codec import MeasurementCodec3D
from self_audit.candidate_a3d.k1 import K1Config, infer_student, student_rollout
from self_audit.candidate_a3d.model import CandidateA3DStudent, StudentConfig3D
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
    # Noise must be voxel-wise, not one global brightness offset per volume.
    assert float(first[0].flatten().std()) > 0.0


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
    full_background_loss, _ = consistency_loss(student.detach().clone().requires_grad_(), target, threshold=0.7, background_weight=1.0)
    assert torch.isfinite(loss)
    assert stats["foreground_coverage"] == 0.0
    assert stats["background_coverage"] == 1.0
    # The all-background contribution remains capped instead of being
    # renormalized back to the full loss when support is entirely BG.
    assert float(loss.detach()) < float(full_background_loss.detach())
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


def test_code_only_unlabeled_rollout_keeps_gradient_through_quantized_decoder() -> None:
    bank = QueryBank3D(BankConfig3D(
        region_grid=(1, 1, 1), boundary_grid=(1, 1, 1), region_scales=(1.0,),
    ))
    codec = MeasurementCodec3D(bank, value_bits=4, base_ids=(0,))
    model = CandidateA3DStudent(
        bank, StudentConfig3D(width=8, decoder_dim=16, num_heads=4, chunk_size=8),
    )
    geometry = {
        "shape": (2, 3, 4), "spacing_zyx": (2.0, 1.0, 0.5),
        "affine_zyx": ((2.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                       (0.0, 0.0, 0.5, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "context": "full_volume",
    }
    image = torch.rand(1, 1, 2, 3, 4)
    base_bits = codec.cost((0,), shape=geometry["shape"], spacing=geometry["spacing_zyx"], affine_zyx=geometry["affine_zyx"])
    result = student_rollout(
        model, image, bank, codec=codec, geometry=geometry,
        labels=None, quantize_ste=True,
        config=K1Config(max_steps=1, candidate_limit=1, teacher_topk=0,
                        student_topk=0, random_candidates=0,
                        bit_budget=base_bits + 256, baseline_ids=(0,)),
    )
    assert result.segmentation_loss is None and result.total_loss is None
    result.final_logits.square().mean().backward()
    assert model.encoder[0].weight.grad is not None
    assert model.value_heads["REGION"][0].weight.grad is not None
    assert model.decoder_head[0].weight.grad is not None
    with torch.no_grad():
        assert torch.isfinite(model.encoder[0].weight.grad).all()
    with pytest.raises(ValueError, match="training-only quantize_ste"):
        infer_student(
            model, image, bank, codec=codec, geometry=geometry,
            quantize_ste=True,
            config=K1Config(max_steps=0, bit_budget=base_bits, baseline_ids=(0,)),
        )
