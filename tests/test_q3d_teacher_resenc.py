"""Contracts for the Q-3D nnFoundation ResEnc-L teacher."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from self_audit.candidate_q3d.teacher_features import (
    forward_teacher_with_features,
    teacher_feature_phase,
    teacher_feature_source,
)
from self_audit.candidate_q3d.teacher_resenc import (
    NNFOUNDATION_HF_REVISION,
    NNFOUNDATION_WEIGHTS_SHA256,
    NNSSL_COMMIT,
    Q3DResEncTeacher,
    build_res_enc_l,
    build_resenc_l,
    load_nnfoundation_encoder,
)


def _geometry(shape: tuple[int, int, int]) -> dict:
    return {
        "shape": shape,
        "spacing_zyx": (2.0, 1.0, 1.0),
        "affine_zyx": torch.eye(4, dtype=torch.float64),
        "context": "full_volume",
    }


def test_default_resenc_matches_pinned_shape_and_keys() -> None:
    encoder = build_res_enc_l()
    alias = build_resenc_l()
    assert tuple(encoder.encoder.output_channels) == (32, 64, 128, 256, 320, 320)
    assert tuple(encoder.encoder.strides[0]) == (1, 1, 1)
    assert tuple(encoder.encoder.strides[-1]) == (2, 2, 2)
    assert len(encoder.encoder.state_dict()) == 448
    assert all(key.startswith(("stem.", "stages.")) for key in encoder.encoder.state_dict())
    assert any(key.startswith("decoder.") for key in encoder.state_dict())
    assert set(alias.state_dict()) == set(encoder.state_dict())
    assert all(alias.state_dict()[key].shape == encoder.state_dict()[key].shape for key in encoder.state_dict())


def _pinned_registry_path() -> Path | None:
    value = os.environ.get("Q3D_NNSSL_REGISTRY")
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_dir():
        candidates = (
            path / "architecture_registry.py",
            path / "src" / "nnssl" / "architectures" / "architecture_registry.py",
            path / "nnssl" / "architectures" / "architecture_registry.py",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    return path if path.is_file() else None


def _load_pinned_registry(path: Path):
    """Load the pinned source file without requiring the optional no-skip module."""

    module_names = (
        "nnssl",
        "nnssl.architectures",
        "nnssl.architectures.noskipResEncUNet",
    )
    saved = {name: sys.modules.get(name) for name in module_names}
    root = types.ModuleType("nnssl")
    architectures = types.ModuleType("nnssl.architectures")
    noskip = types.ModuleType("nnssl.architectures.noskipResEncUNet")
    noskip.ResidualEncoderUNet_noskip = object
    sys.modules.update({
        "nnssl": root,
        "nnssl.architectures": architectures,
        "nnssl.architectures.noskipResEncUNet": noskip,
    })
    module_name = "_q3d_pinned_architecture_registry"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned registry: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


@pytest.mark.skipif(
    _pinned_registry_path() is None,
    reason="set Q3D_NNSSL_REGISTRY to the pinned nnssl architecture_registry.py",
)
def test_resenc_factory_matches_pinned_nnssl_parameters_and_state_shapes() -> None:
    path = _pinned_registry_path()
    assert path is not None
    registry = _load_pinned_registry(path)
    upstream = registry.get_res_enc_l(
        num_input_channels=1,
        num_output_channels=4,
        deep_supervision=False,
    )
    ours = build_res_enc_l()
    assert tuple(ours.encoder.output_channels) == tuple(upstream.encoder.output_channels)
    assert tuple(tuple(item) for item in ours.encoder.strides) == tuple(
        tuple(item) for item in upstream.encoder.strides
    )
    our_parameters = dict(ours.named_parameters())
    upstream_parameters = dict(upstream.named_parameters())
    assert set(our_parameters) == set(upstream_parameters)
    assert sum(parameter.numel() for parameter in our_parameters.values()) == sum(
        parameter.numel() for parameter in upstream_parameters.values()
    )
    our_state = ours.state_dict()
    upstream_state = upstream.state_dict()
    assert set(our_state) == set(upstream_state)
    assert all(our_state[key].shape == upstream_state[key].shape for key in our_state)


def test_same_seed_synthetic_builds_have_identical_state() -> None:
    torch.manual_seed(123)
    first = Q3DResEncTeacher(synthetic=True)
    torch.manual_seed(123)
    second = Q3DResEncTeacher(synthetic=True)
    assert set(first.state_dict()) == set(second.state_dict())
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def test_real_teacher_requires_an_explicit_loaded_encoder() -> None:
    with pytest.raises(ValueError, match="explicitly loaded encoder"):
        Q3DResEncTeacher()


def test_region_decoder_encoder_alias_is_shared_and_checked_on_load() -> None:
    teacher = Q3DResEncTeacher(synthetic=True)
    assert teacher.region_decoder.encoder is teacher.encoder
    encoder_parameter = next(teacher.encoder.parameters())
    decoder_parameter = next(teacher.region_decoder.encoder.parameters())
    assert encoder_parameter is decoder_parameter

    state = teacher.state_dict()
    key = next(key for key in state if key.startswith("encoder."))
    alias = f"region_decoder.{key}"
    assert alias in state
    state[alias] = state[alias].clone()
    state[alias].reshape(-1)[0] += 1.0
    with pytest.raises(ValueError, match="divergent shared encoder aliases"):
        teacher.load_state_dict(state)


def test_synthetic_teacher_uses_same_class_and_native_crop() -> None:
    torch.manual_seed(7)
    teacher = Q3DResEncTeacher(synthetic=True).eval()
    shape = (64, 65, 67)
    image = torch.randn(1, 1, *shape)
    with torch.no_grad():
        outputs = teacher(image, geometry=_geometry(shape), return_features=True)
    assert outputs["region_logits"].shape == (1, 4, *shape)
    assert outputs["boundary_logits"].shape == (1, 4, *shape)
    assert outputs["sdf"].shape == (1, 4, *shape)
    # Feature tensors stay on their padded encoder lattices; only logits/SDF
    # are cropped to native D/H/W.
    assert outputs["feature_encoder_stride8"].shape == (1, 16, 8, 12, 12)
    assert outputs["feature_encoder_stride16"].shape == (1, 20, 4, 6, 6)
    assert teacher_feature_source(teacher, "encoder_stride8")["channels"] == 16
    assert teacher_feature_phase(teacher, "encoder_stride16")[0] == (16, 16, 16)


def test_synthetic_teacher_handles_thin_native_cardiac_volume_in_training_mode() -> None:
    # Keep the production padding rule (smallest cumulative-stride multiple)
    # and make the synthetic geometry valid explicitly: no z downsampling and
    # a 16x in-plane cumulative stride leaves a 2x2 deepest grid.
    teacher = Q3DResEncTeacher(
        synthetic=True,
        strides=(
            (1, 1, 1), (1, 2, 2), (1, 2, 2),
            (1, 2, 2), (1, 2, 2), (1, 1, 1),
        ),
    ).train()
    shape = (8, 32, 32)
    image = torch.randn(1, 1, *shape)
    _, pads = teacher._pad_high(image)
    assert pads == (0, 0, 0)
    outputs = teacher(image, geometry=_geometry(shape))
    assert outputs["region_logits"].shape == (1, 4, *shape)
    assert outputs["boundary_logits"].shape == (1, 4, *shape)
    assert outputs["sdf"].shape == (1, 4, *shape)
    sum(value.square().mean() for value in outputs.values()).backward()


def test_teacher_zscore_is_invariant_to_affine_intensity_change() -> None:
    torch.manual_seed(11)
    teacher = Q3DResEncTeacher(synthetic=True).eval()
    shape = (64, 64, 64)
    image = torch.randn(1, 1, *shape)
    with torch.no_grad():
        first = teacher(image, geometry=_geometry(shape))["region_logits"]
        second = teacher(image.mul(3.25).add(8.0), geometry=_geometry(shape))["region_logits"]
    assert torch.allclose(first, second, atol=2e-4, rtol=2e-4)


def test_feature_forward_is_detached_and_uses_encoder_alias() -> None:
    torch.manual_seed(13)
    teacher = Q3DResEncTeacher(synthetic=True).eval()
    shape = (64, 64, 64)
    image = torch.randn(1, 1, *shape, requires_grad=True)
    outputs = forward_teacher_with_features(
        teacher, image, geometry=_geometry(shape), source="encoder_stride8", synthetic=True,
    )
    assert outputs["feature"].shape[1] == 16
    assert all(not value.requires_grad for value in outputs.values())


def _checkpoint_path() -> Path | None:
    value = os.environ.get("Q3D_NNFOUNDATION_CHECKPOINT")
    if value and Path(value).is_file():
        return Path(value)
    return None


@pytest.mark.skipif(_checkpoint_path() is None, reason="nnFoundation checkpoint is not available")
def test_official_checkpoint_strict_encoder_load_and_ignore_decoder() -> None:
    path = _checkpoint_path()
    assert path is not None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == NNFOUNDATION_WEIGHTS_SHA256
    encoder, metadata = load_nnfoundation_encoder(path, digest)
    assert len(encoder.state_dict()) == 448
    assert metadata["nnssl_commit"] == NNSSL_COMMIT
    assert metadata["hf_revision"] == NNFOUNDATION_HF_REVISION
    assert metadata["encoder_key_count"] == 448
    assert metadata["missing_keys"] == []
    assert metadata["unexpected_keys"] == []
    assert metadata["decoder_loaded"] is False
    assert metadata["segmentation_head_loaded"] is False


def test_checkpoint_missing_sha_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "fake.pth"
    payload = {
        "network_weights": {},
        "nnssl_adaptation_plan": {
            "architecture_plans": {"arch_class_name": "ResEncL"},
            "key_to_encoder": "encoder.stages",
            "key_to_stem": "encoder.stem",
        },
    }
    torch.save(payload, path)
    with pytest.raises(ValueError, match="64-character hash"):
        load_nnfoundation_encoder(path, "")
