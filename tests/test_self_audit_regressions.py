from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.self_audit.audit.counterfactual import CounterfactualGenerator
from src.self_audit.models.auditor import CounterfactualAuditor
from src.self_audit.models.annotation_expert import AnnotationExpert
from src.self_audit.models.dynamic_window import DynamicWindowAttention
from src.self_audit.models.self_audit_net import SelfAuditNet
from src.self_audit.training._utils import (
    adjacent_annotation_pairs,
    build_model_from_config,
    extract_annotation_trajectory,
    load_config,
)
from src.self_audit.training.finetune_joint import compute_joint_losses
from src.self_audit.training.train_auditor import build_auditor_transitions, train_auditor_epoch


ROOT = Path(__file__).resolve().parents[1]


def _counterfactual_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.zeros(1, 48, 48, dtype=torch.long)
    labels[:, 12:34, 12:34] = 1
    labels[:, 36:41, 36:41] = 2
    ground_truth = labels.clone()
    ground_truth[:, 18:23, 18:23] = 2
    probabilities = torch.nn.functional.one_hot(labels, 4).permute(0, 3, 1, 2).float()
    return probabilities, ground_truth, labels


@pytest.mark.parametrize(
    "config_name",
    (
        "self_audit_annotation.yaml",
        "self_audit_auditor.yaml",
        "self_audit_joint.yaml",
        "self_audit_acdc_to_mnms.yaml",
    ),
)
def test_model_builds_from_each_baseline_yaml_without_data_kwargs(config_name: str) -> None:
    config = load_config(ROOT / "configs" / config_name)
    # The YAML is the real baseline contract.  Disable only external weight
    # fetching so this regression remains offline and synthetic-test friendly.
    config = copy.deepcopy(config)
    config["model"]["pretrained_encoder"] = False
    model = build_model_from_config(config, torch.device("cpu"))
    assert model.num_classes == int(config["model"].get("num_classes", config.get("num_classes", 4)))
    assert model.max_turns == int(config["model"]["max_turns"])


def test_model_builder_rejects_image_size_inside_model_config() -> None:
    with pytest.raises(ValueError, match="image_size"):
        build_model_from_config({"image_size": 64, "model": {"image_size": 64}}, torch.device("cpu"))


def test_annotation_trajectory_and_phase_b_pairs_are_adjacent() -> None:
    states = [torch.full((1, 4, 4, 4), float(index)) for index in range(4)]
    output = {"initial_logits": states[0], "states": states[1:]}
    trajectory = extract_annotation_trajectory(output)
    assert all(actual is expected for actual, expected in zip(trajectory, states))
    pairs = adjacent_annotation_pairs(trajectory)
    assert len(pairs) == 3
    assert all(previous is states[index] and candidate is states[index + 1] for index, (previous, candidate) in enumerate(pairs))

    previous, ground_truth, _ = _counterfactual_fixture()
    trajectory = [previous.log(), previous.log() + 0.1, previous.log() + 0.2, previous.log() + 0.3]
    transitions = build_auditor_transitions(
        {"initial_logits": trajectory[0], "states": trajectory[1:]},
        ground_truth,
        CounterfactualGenerator(positive_fraction=1.0, negative_fraction=0.0, hard_neutral_fraction=0.0),
    )
    on_policy = [item for item in transitions if item["provenance"] == "on_policy"]
    assert len(on_policy) == 3
    for index, item in enumerate(on_policy):
        assert torch.allclose(item["previous"], trajectory[index].softmax(dim=1))
        assert torch.allclose(item["candidate"], trajectory[index + 1].softmax(dim=1))


class _CountingAuditor(CounterfactualAuditor):
    def __init__(self) -> None:
        super().__init__(feature_channels=2, hidden_channels=8, residual_blocks=1)
        self.seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, H: torch.Tensor, previous: torch.Tensor, candidate: torch.Tensor, *args, **kwargs):
        self.seen.append((previous.detach().clone(), candidate.detach().clone()))
        return super().forward(H, previous, candidate, *args, **kwargs)


class _ToyPhaseBModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 2, 1)
        self.auditor = _CountingAuditor()

    def forward_annotation(self, images: torch.Tensor):
        initial = images.new_zeros((images.shape[0], 4, images.shape[-2], images.shape[-1]))
        initial[:, 1] = 5.0
        states = [initial + float(index) * 0.1 for index in range(1, 4)]
        return {"initial_logits": initial, "states": states}

    def encode(self, images: torch.Tensor):
        return {"shared": self.features(images)}


def test_phase_b_auditor_receives_every_adjacent_on_policy_transition() -> None:
    model = _ToyPhaseBModel()
    images = torch.randn(1, 3, 16, 16)
    ground_truth = torch.zeros(1, 16, 16, dtype=torch.long)
    loader = DataLoader([{"image": images[0], "mask": ground_truth[0]}], batch_size=1)
    optimizer = torch.optim.SGD(model.auditor.parameters(), lr=1e-3)
    generator = CounterfactualGenerator(positive_fraction=1.0, negative_fraction=0.0, hard_neutral_fraction=0.0)
    stats = train_auditor_epoch(model, loader, optimizer, torch.device("cpu"), generator=generator)
    assert stats["transitions"] == 7.0  # 3 adjacent on-policy + 4 A_t synthetic
    expected_initial = model.forward_annotation(images)["initial_logits"]
    expected = [expected_initial] + model.forward_annotation(images)["states"]
    for index in range(3):
        previous, candidate = model.auditor.seen[index]
        assert torch.allclose(previous, expected[index].softmax(dim=1))
        assert torch.allclose(candidate, expected[index + 1].softmax(dim=1))


def test_positive_counterfactual_is_local_partial_and_samples_strength() -> None:
    previous, ground_truth, labels = _counterfactual_fixture()
    generator = CounterfactualGenerator()
    strengths = []
    target = torch.nn.functional.one_hot(ground_truth, 4).permute(0, 3, 1, 2).float()
    for seed in range(6):
        torch.manual_seed(seed)
        sample = generator.generate(previous, ground_truth, kind="positive")
        assert sample.valid
        assert sample.valid_mask is not None and bool(sample.valid_mask.all())
        assert 0.3 <= float(sample.strength[0]) <= 0.6
        strengths.append(round(float(sample.strength[0]), 5))
        assert sample.edit_mask is not None
        assert bool(sample.edit_mask[0][labels[0] != ground_truth[0]].any())
        assert not torch.allclose(sample.candidate_probs, target)
        assert torch.allclose(sample.candidate_probs.sum(dim=1), torch.ones_like(sample.candidate_probs[:, 0]), atol=1e-5)
    assert len(set(strengths)) > 1


def test_positive_counterfactual_without_error_is_explicitly_invalid() -> None:
    previous, _, labels = _counterfactual_fixture()
    ground_truth = labels.clone()
    sample = CounterfactualGenerator().generate(previous, ground_truth, kind="positive")
    assert not sample.valid
    assert sample.valid_mask is not None and not bool(sample.valid_mask.any())
    assert torch.allclose(sample.previous_probs, sample.candidate_probs)


@pytest.mark.parametrize(
    "operation",
    (
        "local_erosion",
        "local_dilation",
        "boundary_displacement",
        "hole_insertion",
        "false_island",
        "component_deletion",
        "semantic_class_swap",
    ),
)
def test_named_negative_operations_make_distinct_local_edits(operation: str) -> None:
    previous, ground_truth, _ = _counterfactual_fixture()
    torch.manual_seed(7)
    sample = CounterfactualGenerator().generate(previous, ground_truth, kind="negative", operation=operation)
    assert sample.valid
    assert sample.operation == operation
    assert sample.edit_mask is not None and bool(sample.edit_mask.any())
    assert sample.source_class is not None and sample.target_class is not None
    assert torch.isfinite(sample.candidate_probs).all()
    assert bool((sample.candidate_probs >= 0).all())
    assert torch.allclose(sample.candidate_probs.sum(dim=1), torch.ones_like(sample.candidate_probs[:, 0]), atol=1e-5)
    changed = (sample.candidate_probs - sample.previous_probs).abs().sum(dim=1) > 1e-6
    assert bool(changed[0][sample.edit_mask[0]].any())
    assert not bool(changed[0][~sample.edit_mask[0]].any())

    source = int(sample.source_class[0])
    target = int(sample.target_class[0])
    mask = sample.edit_mask[0]
    before_source = sample.previous_probs[0, source][mask].mean()
    after_source = sample.candidate_probs[0, source][mask].mean()
    if operation in {"local_erosion", "hole_insertion", "component_deletion", "semantic_class_swap"}:
        assert after_source < before_source
    if operation in {"local_dilation", "false_island"}:
        assert sample.candidate_probs[0, target][mask].mean() > sample.previous_probs[0, target][mask].mean()
    if operation == "semantic_class_swap":
        assert target != source


def test_hard_neutral_reports_actual_delta_and_neutral_status() -> None:
    previous, ground_truth, _ = _counterfactual_fixture()
    generator = CounterfactualGenerator(epsilon_neutral=0.02, neutral_max_retries=10)
    found_neutral = False
    for seed in range(6):
        torch.manual_seed(seed)
        sample = generator.generate(previous, ground_truth, kind="hard_neutral")
        assert sample.valid
        actual = float(sample.actual_delta_dice[0])
        metadata = sample.metadata["per_sample"][0]
        assert actual == pytest.approx(float(metadata["actual_delta_dice"]), abs=1e-6)
        assert 1 <= int(metadata["retry_count"]) <= generator.neutral_max_retries
        if metadata["neutral_satisfied"]:
            found_neutral = True
            assert abs(actual) < generator.epsilon_neutral
        else:
            assert abs(actual) >= generator.epsilon_neutral
    assert found_neutral


class _ToyJointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.annotation = nn.Conv2d(3, 4, 1)
        self.features = nn.Conv2d(3, 2, 1)
        self.auditor = CounterfactualAuditor(feature_channels=2, hidden_channels=8, residual_blocks=1)

    def infer(self, images: torch.Tensor, *, mode: str, tau_accept: float, t_max: int):
        assert mode == "self_audit"
        previous = self.annotation(images)
        candidate = previous + 0.1 * torch.tanh(previous)
        previous_probs = previous.softmax(dim=1)
        candidate_probs = candidate.softmax(dim=1)
        audit = self.auditor(
            self.features(images),
            previous_probs,
            candidate_probs,
            candidate_probs - previous_probs,
        )
        return {
            "logits": candidate,
            "transition_previous": [previous],
            "transition_candidates": [candidate],
            "audits": [{"local_logits": audit.local_logits, "delta_q": audit.delta_q}],
        }


def test_phase_c_audit_gradients_are_auditor_only() -> None:
    torch.manual_seed(3)
    model = _ToyJointModel()
    batch = {"image": torch.randn(2, 3, 16, 16), "mask": torch.randint(0, 4, (2, 16, 16))}
    total, details = compute_joint_losses(model, batch, lambda_audit=1.0, t_max=1)
    assert details["transition_count"] == 1
    total.backward()
    annotation_with_audit = model.annotation.weight.grad.detach().clone()
    auditor_gradients = [parameter.grad for parameter in model.auditor.parameters()]
    assert any(gradient is not None and torch.isfinite(gradient).all() and bool(gradient.abs().sum() > 0) for gradient in auditor_gradients)

    model.zero_grad(set_to_none=True)
    annotation_only = compute_joint_losses(model, batch, lambda_audit=0.0, t_max=1)[0]
    annotation_only.backward()
    assert torch.allclose(annotation_with_audit, model.annotation.weight.grad, atol=1e-6, rtol=1e-5)


def test_phase_c_separate_loss_terms_have_separate_gradient_paths() -> None:
    torch.manual_seed(31)
    model = _ToyJointModel()
    batch = {"image": torch.randn(2, 3, 16, 16), "mask": torch.randint(0, 4, (2, 16, 16))}
    _, details = compute_joint_losses(model, batch, t_max=1)
    details["audit_loss_tensor"].backward()
    assert model.annotation.weight.grad is None
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in model.auditor.parameters())

    model.zero_grad(set_to_none=True)
    _, details = compute_joint_losses(model, batch, t_max=1)
    details["annotation_loss_tensor"].backward()
    assert model.annotation.weight.grad is not None
    assert not any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in model.auditor.parameters())


def test_actual_self_audit_net_keeps_phase_c_gradient_paths_separate() -> None:
    torch.manual_seed(41)
    model = SelfAuditNet(pretrained_encoder=False, shared_channels=16, window_k=8, max_turns=1)
    batch = {"image": torch.randn(1, 3, 32, 32), "mask": torch.randint(0, 4, (1, 32, 32))}

    _, annotation_details = compute_joint_losses(model, batch, tau_accept=-float("inf"), t_max=1)
    annotation_details["annotation_loss_tensor"].backward()
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in model.fpn.parameters())
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in model.initial_head.parameters())
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in model.annotation_expert.parameters())
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in model.annotation_expert.refinement_block.generator.parameters()
    )

    model.zero_grad(set_to_none=True)
    _, audit_details = compute_joint_losses(model, batch, tau_accept=-float("inf"), t_max=1)
    audit_details["audit_loss_tensor"].backward()
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in model.auditor.parameters())
    annotation_modules = (model.encoder, model.fpn, model.initial_head, model.annotation_expert)
    assert not any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for module in annotation_modules
        for parameter in module.parameters()
    )


def test_dynamic_window_responds_to_audit_conditioning_and_backpropagates() -> None:
    torch.manual_seed(11)
    module = DynamicWindowAttention(16, k=8)
    features = torch.randn(1, 16, 8, 8)
    first = module.generate_coordinates(features, condition=torch.zeros(1, 3, 8, 8))
    second = module.generate_coordinates(features, condition=torch.ones(1, 3, 8, 8))
    assert not torch.allclose(first.coordinates, second.coordinates)

    audit = torch.randn(1, 3, 8, 8, requires_grad=True)
    output = module(features, condition=audit)
    output.square().mean().backward()
    assert audit.grad is not None and torch.isfinite(audit.grad).all()
    generator_gradients = [parameter.grad for parameter in module.generator.parameters()]
    assert any(gradient is not None and torch.isfinite(gradient).all() and bool(gradient.abs().sum() > 0) for gradient in generator_gradients)

    expert = AnnotationExpert(feature_channels=16, window_k=8, max_turns=3)
    shared = torch.randn(1, 16, 8, 8)
    annotation = torch.randn(1, 4, 32, 32)
    audit_zero = torch.zeros(1, 3, 8, 8)
    audit_one = torch.ones(1, 3, 8, 8)
    expert_params_zero = expert.refinement_block.generate_coordinates(shared, condition=audit_zero)
    expert_params_one = expert.refinement_block.generate_coordinates(shared, condition=audit_one)
    assert not torch.allclose(expert_params_zero.coordinates, expert_params_one.coordinates)
    expert_output = expert(
        shared,
        annotation,
        previous_audit_evidence=torch.randn(1, 3, 32, 32),
        turn_index=1,
    )
    expert_output.candidate_logits.square().mean().backward()
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in expert.refinement_block.generator.parameters())
