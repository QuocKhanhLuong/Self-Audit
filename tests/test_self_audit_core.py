from __future__ import annotations

import torch

from src.self_audit.models import (
    AnnotationExpert,
    CounterfactualAuditor,
    DynamicWindowAttention,
    SelfAuditNet,
)


def test_convnext_fpn_initial_and_recurrent_shapes() -> None:
    model = SelfAuditNet(pretrained_encoder=False, shared_channels=32, window_k=8, max_turns=3)
    images = torch.randn(2, 3, 32, 32)
    encoded = model.encode(images)
    assert encoded["shared"].shape == (2, 32, 8, 8)
    output = model.forward_annotation(images, turns=3)
    assert output["initial_logits"].shape == (2, 4, 32, 32)
    assert len(output["states"]) == 3
    assert all(state.shape == (2, 4, 32, 32) for state in output["states"])


def test_dynamic_window_coordinates_and_backward_are_valid() -> None:
    module = DynamicWindowAttention(16, heads=4, k=8)
    features = torch.randn(1, 16, 8, 8, requires_grad=True)
    output, metadata = module(features, return_metadata=True)
    assert output.shape == features.shape
    coordinates = metadata["coordinates"]
    assert coordinates.shape == (1, 8, 8, 8, 2)
    assert torch.isfinite(coordinates).all()
    assert float(coordinates.min()) >= -1.0
    assert float(coordinates.max()) <= 1.0
    output.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_annotation_expert_is_soft_residual_with_shared_depth() -> None:
    expert = AnnotationExpert(feature_channels=16, num_classes=4, window_k=8, max_turns=3)
    shared = torch.randn(1, 16, 8, 8)
    annotation = torch.randn(1, 4, 32, 32)
    first = expert(shared, annotation, turn_index=0)
    third = expert(shared, annotation, turn_index=2)
    assert first.depth == 1
    assert third.depth == 3
    assert expert.refinement_block is expert.refinement_block
    expected = annotation + torch.sigmoid(first.update_gate) * first.delta_logits
    assert torch.allclose(first.candidate_logits, expected)
    assert first.candidate_logits.dtype.is_floating_point


def test_auditor_transition_inputs_are_detached() -> None:
    auditor = CounterfactualAuditor(feature_channels=16, hidden_channels=16)
    shared = torch.randn(2, 16, 8, 8, requires_grad=True)
    previous_logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    candidate_logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    previous = torch.softmax(previous_logits, dim=1)
    candidate = torch.softmax(candidate_logits, dim=1)
    output = auditor(shared, previous, candidate, candidate - previous)
    assert output.local_logits.shape == (2, 3, 32, 32)
    assert output.delta_q.shape == (2, 1)
    assert output.local_logits.requires_grad
    assert output.delta_q.requires_grad
    (output.local_logits.square().mean() + output.delta_q.square().mean()).backward()
    assert shared.grad is None
    assert previous_logits.grad is None
    assert candidate_logits.grad is None
