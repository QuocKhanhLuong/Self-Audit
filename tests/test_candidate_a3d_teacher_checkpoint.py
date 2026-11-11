"""Activation-checkpointing parity for the A-3D boundary path."""

from __future__ import annotations

import copy

import pytest
import torch

from self_audit.candidate_a3d.teacher import build_teacher3d, set_teacher_stage


def _config(*, checkpoint_boundary: bool = False) -> dict:
    return {
        "backbone": "tiny_test3d",
        "input_channels": 1,
        "feature_channels": [8, 16, 24, 32],
        "class_mapping": ["BG", "RV", "MYO", "LV"],
        "checkpoint_boundary": checkpoint_boundary,
    }


def _geometry(shape: tuple[int, int, int]) -> dict:
    spacing = (2.0, 1.0, 0.8)
    return {
        "shape": shape,
        "spacing_zyx": spacing,
        "affine_zyx": ((2.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                       (0.0, 0.0, 0.8, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "context": "full_volume",
    }


def _paired_teachers() -> tuple[torch.nn.Module, torch.nn.Module]:
    plain, _ = build_teacher3d(_config(checkpoint_boundary=False), synthetic=True)
    checkpointed, _ = build_teacher3d(_config(checkpoint_boundary=True), synthetic=True)
    checkpointed.load_state_dict(copy.deepcopy(plain.state_dict()), strict=True)
    return plain, checkpointed


def test_boundary_checkpoint_matches_inference_outputs() -> None:
    plain, checkpointed = _paired_teachers()
    plain.eval()
    checkpointed.eval()
    x = torch.rand(1, 1, 5, 7, 9)
    geometry = _geometry(tuple(x.shape[-3:]))
    with torch.inference_mode():
        expected = plain(x, geometry=geometry)
        actual = checkpointed(x, geometry=geometry)
    for key in ("region_logits", "boundary_logits", "sdf"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)


def test_boundary_checkpoint_matches_full_forward_gradients() -> None:
    plain, checkpointed = _paired_teachers()
    plain.train()
    checkpointed.train()
    x = torch.rand(1, 1, 5, 7, 9)
    geometry = _geometry(tuple(x.shape[-3:]))
    plain(x, geometry=geometry)["boundary_logits"].square().mean().backward()
    checkpointed(x, geometry=geometry)["boundary_logits"].square().mean().backward()
    for (name_plain, parameter_plain), (name_checkpointed, parameter_checkpointed) in zip(
        plain.named_parameters(), checkpointed.named_parameters(), strict=True,
    ):
        assert name_plain == name_checkpointed
        if parameter_plain.grad is None or parameter_checkpointed.grad is None:
            assert parameter_plain.grad is None and parameter_checkpointed.grad is None
        else:
            torch.testing.assert_close(parameter_checkpointed.grad, parameter_plain.grad, rtol=0.0, atol=0.0)


def test_boundary_checkpoint_matches_warmup_with_frozen_encoder() -> None:
    plain, checkpointed = _paired_teachers()
    set_teacher_stage(plain, "teacher_warmup")
    set_teacher_stage(checkpointed, "teacher_warmup")
    x = torch.rand(1, 1, 5, 7, 9)
    geometry = _geometry(tuple(x.shape[-3:]))
    expected = plain(x, geometry=geometry)
    actual = checkpointed(x, geometry=geometry)
    for key in ("region_logits", "boundary_logits", "sdf"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)
    expected["boundary_logits"].square().mean().backward()
    actual["boundary_logits"].square().mean().backward()
    assert all(parameter.grad is None for parameter in plain.encoder.parameters())
    assert all(parameter.grad is None for parameter in checkpointed.encoder.parameters())
    assert any(parameter.grad is not None for parameter in checkpointed.boundary_path.parameters())


def test_checkpoint_boundary_flag_is_strict() -> None:
    with pytest.raises(ValueError, match="checkpoint_boundary"):
        build_teacher3d({**_config(), "checkpoint_boundary": 1}, synthetic=True)
