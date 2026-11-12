from __future__ import annotations

import numpy as np
import pytest
import torch
import hashlib

from self_audit.candidate_q3d.student import (
    CandidateQ3DStudent,
    _EPA3DBlock,
    _valid_grid_mask,
)


def _tiny_config(
    arm: str = "V1",
    *,
    support: str = "uniform",
    residual: bool = False,
    teacher_channels: int = 12,
) -> dict:
    return {
        "schema": "candidate_q3d.v1",
        "arm": arm,
        "model": {
            "student": {
                "architecture": "unetrpp_lite_3d",
                "channels": [8, 16, 24, 32, 48],
                "num_classes": 4,
                "attention": {"support": support},
                "residual2": {"enabled": residual, "zero_init": True},
            },
            "teacher": {"feature_channels": teacher_channels},
        },
        "tokenizer": {
            "codebook_size": 8,
            "embedding_dim": 8,
            "temperature": 0.1,
        },
    }


def _geometry(shape: tuple[int, int, int]) -> dict:
    return {
        "shape": shape,
        "spacing_zyx": (1.0, 1.25, 1.5),
        "affine_zyx": np.diag([1.0, 1.25, 1.5, 1.0]),
        "context": "full_volume",
    }


def _frozen_codebook(size: int = 8, dim: int = 8) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(29)
    codes = torch.randn(size, dim, generator=generator)
    digest = hashlib.sha256(codes.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    return {
        "schema": "self_audit.q3d.codebook.v1",
        "codes": codes,
        "version": "q3d_teacher_codebook.v1",
        "sha256": digest,
        "size": size,
        "dimension": dim,
        "initialized": True,
        "frozen": True,
    }


def test_native_shape_crop_and_pad_for_odd_inplane_sizes() -> None:
    torch.set_num_threads(1)
    model = CandidateQ3DStudent(codebook=_frozen_codebook(256, 64)).eval()
    for depth in (6, 9, 10, 13, 20):
        image = torch.rand(1, 1, depth, 23, 31)
        with torch.inference_mode():
            output = model(image)
        assert output["logits"].shape == (1, 4, depth, 23, 31)
        assert output["f4"].shape == (1, 256, depth, 2, 2)
        assert "sdf" not in output
        assert output["codes"].shape == (1, depth, 2, 2)


def test_default_student_stays_under_ten_million_parameters() -> None:
    model = CandidateQ3DStudent()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert model.codebook_size == 256
    assert model.codebook_dim == 64
    assert model.C0.shape == (256, 64)
    assert not model.C0.requires_grad
    assert parameter_count <= 10_000_000


def test_vq_requires_a_frozen_codebook_and_persists_its_identity() -> None:
    torch.set_num_threads(1)
    config = _tiny_config("V1")
    uninitialized = CandidateQ3DStudent(config)
    assert uninitialized.C0.shape == (8, 8)
    assert torch.count_nonzero(uninitialized.C0) == 0
    assert uninitialized.codebook_version.item() == 0
    with pytest.raises(RuntimeError, match="requires frozen tokenizer C0"):
        uninitialized(torch.rand(1, 1, 6, 17, 25))

    mapping = _frozen_codebook()
    model = CandidateQ3DStudent(config, codebook=mapping)
    assert model.codebook_version.item() == 1
    assert model.codebook_version_name == mapping["version"]
    assert model.codebook_sha256 == mapping["sha256"]
    state = model.state_dict()
    restored = CandidateQ3DStudent(config)
    restored.load_state_dict(state)
    assert restored.codebook_version.item() == 1
    assert restored.codebook_version_name == mapping["version"]
    assert restored.codebook_sha256 == mapping["sha256"]

    # This is the engine's post-Q installation path: the model exists before
    # C0 is learned, then receives the frozen codebook and numeric version.
    engine_style = CandidateQ3DStudent(config)
    with pytest.raises(RuntimeError, match="requires frozen tokenizer C0"):
        engine_style(torch.rand(1, 1, 6, 17, 25))
    installed = engine_style.install_codebook(mapping)
    assert installed == {
        "version": mapping["version"],
        "sha256": mapping["sha256"],
        "size": mapping["size"],
        "dimension": mapping["dimension"],
        "initialized": True,
        "frozen": True,
    }
    assert engine_style.install_codebook(mapping) == installed
    replacement_codes = mapping["codes"] + 0.1
    replacement = dict(mapping, codes=replacement_codes)
    replacement["sha256"] = hashlib.sha256(
        replacement_codes.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="cannot replace an already frozen"):
        engine_style.install_codebook(replacement)
    engine_state = engine_style.state_dict()
    resumed = CandidateQ3DStudent(config)
    resumed.load_state_dict(engine_state)
    assert resumed.codebook_version_name == "q3d_teacher_codebook.v1"
    assert resumed.codebook_sha256 == mapping["sha256"]

    tampered = dict(state)
    tampered["C0"] = state["C0"] + 0.1
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        CandidateQ3DStudent(config).load_state_dict(tampered)
    with pytest.raises(ValueError, match="requires frozen codes"):
        CandidateQ3DStudent(config, codebook=mapping["codes"])
    unfrozen = dict(mapping, frozen=False)
    with pytest.raises(ValueError, match="frozen=true"):
        CandidateQ3DStudent(config, codebook=unfrozen)


def test_externally_padded_values_are_discarded_before_network() -> None:
    torch.set_num_threads(1)
    model = CandidateQ3DStudent(_tiny_config("C0")).eval()
    native = torch.rand(1, 1, 6, 17, 25)
    geometry = _geometry((6, 17, 25))
    externally_padded = torch.full((1, 1, 6, 32, 32), float("nan"))
    externally_padded[..., :17, :25] = native
    with torch.inference_mode():
        expected = model(native, geometry=geometry)["logits"]
        actual = model(externally_padded, geometry=geometry)["logits"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (1, 4, 6, 17, 25)


def test_geometry_rejects_missing_native_coverage_and_z_padding() -> None:
    model = CandidateQ3DStudent(_tiny_config("C0")).eval()
    image = torch.rand(1, 1, 6, 16, 24)
    with pytest.raises(ValueError, match="smaller than the full native geometry"):
        model(image, geometry=_geometry((6, 17, 24)))
    with pytest.raises(ValueError, match="z axis"):
        model(image, geometry=_geometry((5, 16, 24)))


def test_continuous_and_quantized_arms_expose_their_declared_interfaces() -> None:
    torch.set_num_threads(1)
    image = torch.rand(1, 1, 6, 17, 25)

    c0 = CandidateQ3DStudent(_tiny_config("C0"))
    c0_result = c0(image)
    assert c0_result["sdf"].shape == (1, 4, 6, 17, 25)
    assert c0_result["f4"].shape[-3:] == (6, 2, 2)
    assert c0_result["u"] is None
    assert c0_result["pi"] is None
    assert c0_result["codes"] is None
    assert c0_result["quantized"] is None
    assert c0.C0.shape == (0, 0)

    deploy_config = _tiny_config("C0")
    deploy_config["model"]["student"]["sdf_head"] = {"enabled": False}
    deploy_c0 = CandidateQ3DStudent(deploy_config).train()
    deploy_result = deploy_c0(image)
    assert deploy_c0.sdf_head is None
    assert "sdf" not in deploy_result

    c1 = CandidateQ3DStudent(_tiny_config("C1"))
    c1_result = c1(image)
    assert c1_result["feature_kd"].shape == (1, 12, 6, 2, 2)
    assert c1_result["u"].shape == (1, 8, 6, 2, 2)
    assert c1_result["pi"] is None
    assert c1_result["quantized"] is None

    for arm in ("V1", "V0", "V1-u"):
        model = CandidateQ3DStudent(_tiny_config(arm), codebook=_frozen_codebook()).eval()
        result = model(image)
        assert result["pi"].shape == (1, 8, 6, 2, 2)
        assert result["codes"].shape == (1, 6, 2, 2)
        assert result["quantized"].shape == (1, 8, 6, 2, 2)
        assert result["codebook_version"] == "q3d_teacher_codebook.v1"
        normalized_codes = torch.nn.functional.normalize(model.C0, p=2.0, dim=-1)
        selected = normalized_codes[result["codes"]].permute(0, 4, 1, 2, 3)
        torch.testing.assert_close(result["quantized"], selected, rtol=0, atol=1e-6)

    b1 = CandidateQ3DStudent(_tiny_config("B1"), codebook=_frozen_codebook()).eval()
    assert b1.code_only and not b1.uses_dense_skips
    assert b1.infer_logits(image).shape == (1, 4, 6, 17, 25)


def test_vq_straight_through_gradient_reaches_adapter_but_not_c0() -> None:
    torch.set_num_threads(1)
    model = CandidateQ3DStudent(_tiny_config("V1"), codebook=_frozen_codebook())
    with torch.no_grad():
        model.vq_gate.fill_(0.25)
    image = torch.rand(1, 1, 6, 17, 25)
    result = model(image)
    loss = result["logits"].square().mean() + result["sdf"].square().mean()
    loss.backward()
    assert model.adapter[0].weight.grad is not None
    assert torch.isfinite(model.adapter[0].weight.grad).all()
    assert model.C0.grad is None
    assert "C0" not in dict(model.named_parameters())


def test_uniform_and_learned_content_support_are_shape_aware_and_masked() -> None:
    torch.set_num_threads(1)
    feature = torch.rand(1, 8, 6, 4, 4)
    valid = _valid_grid_mask(
        1, 6, 4, 4, native_shape=(6, 17, 25), stride_yx=8, device=feature.device
    )
    block = _EPA3DBlock(8, "uniform", support_count=8).eval()
    changed_padding = feature.clone()
    changed_padding.masked_fill_(valid == 0, 1000.0)
    with torch.inference_mode():
        base = block(feature, valid, (6, 17, 25), 8)
        changed = block(changed_padding, valid, (6, 17, 25), 8)
    valid_cells = valid.expand_as(base) > 0
    torch.testing.assert_close(base[valid_cells], changed[valid_cells], rtol=0, atol=0)
    assert valid[0, 0, 0, 2, 0].item() == pytest.approx(1 / 8)
    assert valid[0, 0, 0, 2, 3].item() == pytest.approx(1 / 64)

    adaptive_block = _EPA3DBlock(8, "learned_content", support_count=64).eval()
    with torch.no_grad():
        adaptive_block.detail_score.weight.zero_()
        adaptive_block.detail_score.weight[0, 0] = 1.0
    large_feature = torch.arange(150, dtype=torch.float32).view(1, 1, 6, 5, 5)
    large_feature = large_feature.expand(1, 8, 6, 5, 5).clone()
    large_mask = _valid_grid_mask(
        1, 6, 5, 5, native_shape=(6, 65, 65), stride_yx=16, device=feature.device
    )
    large_weight = large_mask.flatten(2).squeeze(1)
    tokens_a = large_feature.flatten(2).transpose(1, 2)
    tokens_b = tokens_a.clone()
    tokens_b[..., 0] = -tokens_b[..., 0]
    support_a = adaptive_block._build_adaptive_support(
        tokens_a, large_weight, (6, 65, 65), (6, 5, 5)
    )
    support_b = adaptive_block._build_adaptive_support(
        tokens_b, large_weight, (6, 65, 65), (6, 5, 5)
    )
    assert support_a.shape == (1, 64, 150)
    torch.testing.assert_close(
        support_a.sum(dim=-1), torch.ones(1, 64), rtol=0, atol=1e-6
    )
    assert not torch.allclose(support_a[:, 32:], support_b[:, 32:])

    small_mask = _valid_grid_mask(
        1, 6, 2, 2, native_shape=(6, 17, 25), stride_yx=16, device=feature.device
    )
    small_weight = small_mask.flatten(2).squeeze(1)
    small_tokens = torch.rand(1, 24, 8)
    identity_support = adaptive_block._build_adaptive_support(
        small_tokens, small_weight, (6, 17, 25), (6, 2, 2)
    )
    torch.testing.assert_close(identity_support[0], torch.eye(24), rtol=0, atol=0)

    adaptive = CandidateQ3DStudent(
        _tiny_config("E1", support="adaptive"), codebook=_frozen_codebook()
    ).eval()
    with torch.inference_mode():
        output = adaptive(torch.rand(1, 1, 9, 65, 65))
    assert output["logits"].shape == (1, 4, 9, 65, 65)
    assert any(name.endswith("detail_score.weight") for name, _ in adaptive.named_parameters())


def test_uniform_epa_pools_projected_keys_and_values_with_fractional_support(monkeypatch) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(41)
    block = _EPA3DBlock(8, "uniform", support_count=32).eval()
    feature = torch.randn(1, 8, 2, 5, 5)
    valid = _valid_grid_mask(
        1, 2, 5, 5, native_shape=(2, 9, 9), stride_yx=2, device=feature.device
    )
    token_mask = valid.flatten(2).squeeze(1) > 0
    valid_x = feature * token_mask.reshape(1, 1, 2, 5, 5)
    tokens = valid_x.flatten(2).transpose(1, 2)
    normalized = block.norm(tokens)
    projected = block.qk(normalized).reshape(1, 50, 2, 8)
    k_full = projected[:, :, 1]
    v_full = block.spatial_value(normalized)

    def weighted_pool(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values_5d = values.transpose(1, 2).reshape(1, 8, 2, 5, 5)
        weights = valid.to(dtype=values.dtype)
        numerator = torch.nn.functional.adaptive_avg_pool3d(values_5d * weights, (2, 4, 4))
        denominator = torch.nn.functional.adaptive_avg_pool3d(weights, (2, 4, 4))
        pooled = numerator / denominator.clamp_min(torch.finfo(values.dtype).eps)
        support_valid = denominator.flatten(2).squeeze(1) > 0
        return pooled.flatten(2).transpose(1, 2), support_valid

    expected_k, support_valid = weighted_pool(k_full)
    expected_v, _ = weighted_pool(v_full)
    # The old operation order pools x first and then applies LayerNorm/projections.
    input_weights = valid.to(dtype=feature.dtype)
    pooled_input = torch.nn.functional.adaptive_avg_pool3d(valid_x * input_weights, (2, 4, 4))
    input_denominator = torch.nn.functional.adaptive_avg_pool3d(input_weights, (2, 4, 4))
    pooled_input = pooled_input / input_denominator.clamp_min(torch.finfo(feature.dtype).eps)
    old_tokens = block.norm(pooled_input.flatten(2).transpose(1, 2))
    old_projected = block.qk(old_tokens).reshape(1, 32, 2, 8)
    old_k = old_projected[:, :, 1]
    old_v = block.spatial_value(old_tokens)

    observed: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def capture_attention(query, key, value, **kwargs):
        observed.append((query.detach(), key.detach(), value.detach()))
        return torch.zeros_like(query)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", capture_attention)
    with torch.inference_mode():
        block(feature, valid, (2, 9, 9), 2)

    assert len(observed) == 1
    _, actual_k, actual_v = observed[0]
    expected_k = expected_k.reshape(1, 32, 4, 2).transpose(1, 2)
    expected_v = expected_v.reshape(1, 32, 4, 2).transpose(1, 2)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=1e-6)
    torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=1e-6)
    assert support_valid.any()
    old_k = old_k.reshape(1, 32, 4, 2).transpose(1, 2)
    old_v = old_v.reshape(1, 32, 4, 2).transpose(1, 2)
    assert not torch.allclose(actual_k, old_k, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(actual_v, old_v, rtol=1e-4, atol=1e-5)
    assert ((valid > 0) & (valid < 1)).any()  # Partial native-cell weights are present.


def test_residual_second_level_is_zero_gated_at_initialization() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(27)
    base = CandidateQ3DStudent(_tiny_config("V1"), codebook=_frozen_codebook()).eval()
    torch.manual_seed(89)
    residual = CandidateQ3DStudent(
        _tiny_config("V1", residual=True), codebook=_frozen_codebook()
    ).eval()
    base_state = base.state_dict()
    residual_state = residual.state_dict()
    for key, value in base_state.items():
        if key in residual_state and torch.is_tensor(value) and residual_state[key].shape == value.shape:
            residual_state[key].copy_(value)
    assert residual.residual_gate.item() == 0.0
    assert residual.residual_codebook.shape == (8, 8)

    image = torch.rand(1, 1, 6, 17, 25)
    with torch.inference_mode():
        expected = base(image)["logits"]
        actual = residual(image)["logits"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
