"""Tests for typed entropy contracts (Wave 2.2).

Validates explicit entropy_from_logits and entropy_from_probabilities contracts,
eliminating heuristic min/max inference, ensuring logit shift and batch invariance,
simplex validation, and backwards compatibility.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

try:
    from self_audit.models.annotation_expert import (
        AnnotationExpert,
        MODEL_ENTROPY_VERSION,
        annotation_entropy,
        entropy_from_logits,
        entropy_from_probabilities,
    )
except ImportError:
    from src.self_audit.models.annotation_expert import (
        AnnotationExpert,
        MODEL_ENTROPY_VERSION,
        annotation_entropy,
        entropy_from_logits,
        entropy_from_probabilities,
    )


# ==============================================================================
# 1. Constant logit shift invariance and batch-composition invariance
# ==============================================================================


def test_constant_logit_shift_invariance() -> None:
    """Logits shifted by an arbitrary constant along channels must have identical entropy."""
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 8, 8)
    base_entropy = entropy_from_logits(logits)

    # Positive scalar shift
    shifted_pos = entropy_from_logits(logits + 10.0)
    assert torch.allclose(base_entropy, shifted_pos, atol=1e-6)

    # Negative scalar shift
    shifted_neg = entropy_from_logits(logits - 25.0)
    assert torch.allclose(base_entropy, shifted_neg, atol=1e-6)

    # Spatially broadcastable constant shift per-pixel [B, 1, H, W]
    spatial_shift = torch.randn(2, 1, 8, 8) * 5.0
    shifted_spatial = entropy_from_logits(logits + spatial_shift)
    assert torch.allclose(base_entropy, shifted_spatial, atol=1e-6)


def test_batch_composition_invariance() -> None:
    """Entropy of sample i must be independent of other samples in the batch."""
    torch.manual_seed(42)
    sample_a = torch.randn(1, 4, 8, 8)
    sample_b = torch.randn(1, 4, 8, 8) * 10.0 + 50.0

    entropy_a_alone = entropy_from_logits(sample_a)
    entropy_b_alone = entropy_from_logits(sample_b)

    # Joint batch
    batch = torch.cat([sample_a, sample_b], dim=0)
    entropy_batch = entropy_from_logits(batch)

    assert torch.allclose(entropy_batch[0:1], entropy_a_alone, atol=1e-6)
    assert torch.allclose(entropy_batch[1:2], entropy_b_alone, atol=1e-6)


# ==============================================================================
# 2. Astra review probe: [0, 1] range input shift error eliminated
# ==============================================================================


def test_astra_review_probe_shift_error_zero() -> None:
    """Replicates Astra's exact audit probe from evidence.md:

    z = torch.tensor([0., 0.1, 0.2, 0.3]).view(1, 4, 1, 1).expand(1, 4, 2, 2)
    Old bug produced shift_error = 0.2659366 when shifting by -2 because
    z was erroneously treated as probabilities instead of logits.
    """
    z = torch.tensor([0.0, 0.1, 0.2, 0.3]).view(1, 4, 1, 1).expand(1, 4, 2, 2)
    correct = -(z.softmax(1) * z.log_softmax(1)).sum(1, keepdim=True) / np.log(4)

    actual = entropy_from_logits(z)
    assert torch.allclose(actual, correct, atol=1e-6)

    shifted = entropy_from_logits(z - 2.0)
    shift_error = (actual - shifted).abs().max().item()
    assert shift_error < 1e-6, f"Expected shift error ~0, got {shift_error}"


def test_expert_forward_uses_logits_formula_for_unit_interval_logits() -> None:
    """Verify AnnotationExpert.forward actually uses entropy_from_logits even when

    annotation_logits has all values in [0, 1].
    """
    torch.manual_seed(42)
    expert = AnnotationExpert(feature_channels=16, num_classes=4, audit_channels=3, window_k=4)
    expert.eval()

    shared = torch.randn(1, 16, 4, 4)
    # Logits strictly inside [0.1, 0.9], which tripped the old min/max heuristic
    logits = torch.tensor([0.1, 0.3, 0.6, 0.8]).view(1, 4, 1, 1).expand(1, 4, 8, 8)

    # Pass explicitly computed entropy vs None (expert computes it internally)
    expected_entropy = entropy_from_logits(logits)
    out_with_entropy = expert(shared, logits, entropy=expected_entropy)
    out_default = expert(shared, logits, entropy=None)

    assert torch.allclose(out_with_entropy.candidate_logits, out_default.candidate_logits, atol=1e-6)
    assert torch.allclose(out_with_entropy.delta_logits, out_default.delta_logits, atol=1e-6)


# ==============================================================================
# 3. Probability simplex contracts and validation
# ==============================================================================


def test_uniform_logits_and_probabilities_entropy_one() -> None:
    """Uniform distributions across C classes must have normalized entropy = 1.0."""
    # 4 classes
    uniform_logits_4 = torch.zeros(2, 4, 8, 8)
    assert torch.allclose(entropy_from_logits(uniform_logits_4), torch.ones(2, 1, 8, 8), atol=1e-6)

    uniform_probs_4 = torch.full((2, 4, 8, 8), 0.25)
    assert torch.allclose(entropy_from_probabilities(uniform_probs_4), torch.ones(2, 1, 8, 8), atol=1e-6)

    # 3 classes
    uniform_logits_3 = torch.full((1, 3, 4, 4), 2.5)
    assert torch.allclose(entropy_from_logits(uniform_logits_3), torch.ones(1, 1, 4, 4), atol=1e-6)

    uniform_probs_3 = torch.full((1, 3, 4, 4), 1.0 / 3.0)
    assert torch.allclose(entropy_from_probabilities(uniform_probs_3), torch.ones(1, 1, 4, 4), atol=1e-6)


def test_one_hot_probabilities_entropy_zero() -> None:
    """One-hot probability tensors must have exact 0.0 entropy."""
    for c in (2, 4, 8):
        for class_idx in range(c):
            one_hot = torch.zeros(1, c, 4, 4)
            one_hot[:, class_idx] = 1.0
            ent = entropy_from_probabilities(one_hot)
            assert torch.allclose(ent, torch.zeros(1, 1, 4, 4), atol=1e-7)
            assert (ent >= 0.0).all()


def test_valid_probabilities_parity_with_logits() -> None:
    """For any valid logits, entropy_from_probabilities(softmax(logits)) == entropy_from_logits(logits)."""
    torch.manual_seed(42)
    logits = torch.randn(3, 4, 16, 16)
    probs = F.softmax(logits, dim=1)

    ent_from_logits = entropy_from_logits(logits)
    ent_from_probs = entropy_from_probabilities(probs)

    assert torch.allclose(ent_from_logits, ent_from_probs, atol=1e-6)


def test_invalid_probability_simplex_rejected() -> None:
    """Probabilities that violate the simplex contract must be rejected loudly with ValueError."""
    # 1. Sum does not equal 1.0 (silent renormalization prohibited)
    unnormalized = torch.full((1, 4, 8, 8), 0.5)  # sum is 2.0
    with pytest.raises(ValueError, match="do not sum to 1.0"):
        entropy_from_probabilities(unnormalized)

    # 2. Values outside [0, 1]
    bad_range = torch.zeros(1, 4, 8, 8)
    bad_range[:, 0] = 1.5
    bad_range[:, 1] = -0.5
    with pytest.raises(ValueError, match="out of valid range"):
        entropy_from_probabilities(bad_range)

    # 3. NaN values
    nan_tensor = torch.full((1, 4, 8, 8), float("nan"))
    with pytest.raises(ValueError, match="finite values"):
        entropy_from_probabilities(nan_tensor)

    # 4. Inf values
    inf_tensor = torch.full((1, 4, 8, 8), float("inf"))
    with pytest.raises(ValueError, match="finite values"):
        entropy_from_probabilities(inf_tensor)

    # 5. Non-4D tensor
    with pytest.raises(ValueError, match="Expected 4D"):
        entropy_from_probabilities(torch.tensor([0.25, 0.25, 0.25, 0.25]))

    with pytest.raises(ValueError, match="Expected 4D"):
        entropy_from_logits(torch.tensor([0.25, 0.25, 0.25, 0.25]))


# ==============================================================================
# 4. Finite gradients and differentiability
# ==============================================================================


def test_finite_gradients_logits_backward() -> None:
    """entropy_from_logits must support backpropagation with strictly finite gradients."""
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    ent = entropy_from_logits(logits)
    loss = ent.sum()
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0.0


def test_finite_gradients_probabilities_backward() -> None:
    """entropy_from_probabilities must support backpropagation with finite gradients."""
    torch.manual_seed(42)
    # Start from logits to ensure valid simplex during forward pass
    raw = torch.randn(2, 4, 8, 8, requires_grad=True)
    probs = F.softmax(raw, dim=1)
    ent = entropy_from_probabilities(probs)
    loss = ent.sum()
    loss.backward()

    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


# ==============================================================================
# 5. Backwards compatibility alias and semantic versioning
# ==============================================================================


def test_annotation_entropy_alias_contract() -> None:
    """annotation_entropy default is logits; explicit input_type is supported; invalid types fail."""
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 8, 8)
    probs = F.softmax(logits, dim=1)

    # Default is logits
    assert torch.allclose(annotation_entropy(logits), entropy_from_logits(logits))
    assert torch.allclose(annotation_entropy(logits, input_type="logits"), entropy_from_logits(logits))

    # Explicit probabilities
    assert torch.allclose(
        annotation_entropy(probs, input_type="probabilities"),
        entropy_from_probabilities(probs),
    )

    # Invalid input_type raises ValueError with clear message
    with pytest.raises(ValueError, match="Invalid input_type"):
        annotation_entropy(logits, input_type="auto")

    with pytest.raises(ValueError, match="Heuristic range-based dispatch has been removed"):
        annotation_entropy(logits, input_type="unknown")


def test_model_entropy_version_metadata() -> None:
    """Semantic version constant must be exposed for lineage and hashing."""
    assert isinstance(MODEL_ENTROPY_VERSION, str)
    assert len(MODEL_ENTROPY_VERSION.split(".")) >= 3
    assert AnnotationExpert.entropy_version == MODEL_ENTROPY_VERSION


def test_annotation_expert_state_dict_keys_unchanged() -> None:
    """AnnotationExpert must not introduce new parameter tensors or state_dict keys."""
    expert = AnnotationExpert(feature_channels=16, num_classes=4, audit_channels=3, window_k=4)
    state_keys = set(expert.state_dict().keys())

    # All state keys belong to known architectural submodules
    for key in state_keys:
        prefix = key.split(".")[0]
        assert prefix in (
            "input_projection",
            "refinement_block",
            "refinement_residual",
            "delta_head",
            "gate_head",
        ), f"Unexpected key in state_dict: {key}"

    # Verify no 'entropy' tensor is stored in state_dict
    assert not any("entropy" in k for k in state_keys)


# ==============================================================================
# 6. FP16 / BFloat16 / Float64 precision stability and numerical edge cases
# ==============================================================================


def test_fp16_onehot_entropy_finite_and_gradients_finite() -> None:
    """Regression test for Astra's review probe: fp16 one-hot input must produce

    finite output (0.0) and strictly finite gradients without NaN.
    """
    p = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float16).reshape(1, 4, 1, 1).requires_grad_()
    h = entropy_from_probabilities(p)
    assert h.dtype == torch.float16
    assert torch.isfinite(h).all()
    assert torch.allclose(h, torch.zeros_like(h), atol=1e-3)

    h.sum().backward()
    assert p.grad is not None
    assert p.grad.dtype == torch.float16
    assert torch.isfinite(p.grad).all()


def test_fp16_extreme_logits_finite_entropy_and_gradients() -> None:
    """Regression test for Astra's review probe: extreme fp16 logits [65504, -65504, 0, 0]

    must produce finite output and strictly finite gradients without NaN.
    """
    x = torch.tensor([65504.0, -65504.0, 0.0, 0.0], dtype=torch.float16).reshape(1, 4, 1, 1).requires_grad_()
    hx = entropy_from_logits(x)
    assert hx.dtype == torch.float16
    assert torch.isfinite(hx).all()
    assert torch.allclose(hx, torch.zeros_like(hx), atol=1e-3)

    hx.sum().backward()
    assert x.grad is not None
    assert x.grad.dtype == torch.float16
    assert torch.isfinite(x.grad).all()


def test_bfloat16_precision_stability() -> None:
    """bfloat16 inputs must compute stably in float32 and return bfloat16 with finite gradients."""
    p = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.bfloat16).reshape(1, 4, 1, 1).requires_grad_()
    h = entropy_from_probabilities(p)
    assert h.dtype == torch.bfloat16
    assert torch.isfinite(h).all()
    h.sum().backward()
    assert p.grad is not None
    assert p.grad.dtype == torch.bfloat16
    assert torch.isfinite(p.grad).all()

    x = torch.tensor([1e38, -1e38, 0.0, 0.0], dtype=torch.bfloat16).reshape(1, 4, 1, 1).requires_grad_()
    hx = entropy_from_logits(x)
    assert hx.dtype == torch.bfloat16
    assert torch.isfinite(hx).all()
    hx.sum().backward()
    assert x.grad is not None
    assert x.grad.dtype == torch.bfloat16
    assert torch.isfinite(x.grad).all()


def test_float64_precision_preserved() -> None:
    """float64 inputs must compute in float64 and preserve float64 output and gradient dtypes."""
    p = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64).reshape(1, 4, 1, 1).requires_grad_()
    h = entropy_from_probabilities(p)
    assert h.dtype == torch.float64
    assert torch.allclose(h, torch.ones_like(h), atol=1e-8)
    h.sum().backward()
    assert p.grad is not None
    assert p.grad.dtype == torch.float64
    assert torch.isfinite(p.grad).all()

    x = torch.randn(2, 4, 8, 8, dtype=torch.float64, requires_grad=True)
    hx = entropy_from_logits(x)
    assert hx.dtype == torch.float64
    hx.sum().backward()
    assert x.grad is not None
    assert x.grad.dtype == torch.float64
    assert torch.isfinite(x.grad).all()


def test_nonfinite_logits_rejected() -> None:
    """Regression test: NaN, +Inf, and -Inf logits must raise ValueError instead of

    silently returning finite zeros.
    """
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        x = torch.tensor([bad_val, 0.0, 0.0, 0.0]).reshape(1, 4, 1, 1)
        with pytest.raises(ValueError, match="finite values"):
            entropy_from_logits(x)


def test_nonfinite_rejected_before_channel_shortcut() -> None:
    """Regression test: C<=1 tensors with NaN, +Inf, or -Inf must raise ValueError

    before the C<=1 shortcut returns zero.
    """
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        x1 = torch.tensor([bad_val]).reshape(1, 1, 1, 1)
        with pytest.raises(ValueError, match="finite values"):
            entropy_from_logits(x1)

        p1 = torch.tensor([bad_val]).reshape(1, 1, 1, 1)
        with pytest.raises(ValueError, match="finite values"):
            entropy_from_probabilities(p1)


