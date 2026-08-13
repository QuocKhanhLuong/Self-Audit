from __future__ import annotations

import torch

from src.self_audit.audit import (
    FIX,
    REGRESS,
    UNCHANGED,
    CounterfactualGenerator,
    ThresholdGate,
    build_transition_targets,
)


def _labels_to_probs(labels: torch.Tensor, classes: int = 4) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, classes).permute(0, 3, 1, 2).float()


def test_fix_unchanged_regress_target_semantics() -> None:
    ground_truth = torch.tensor([[[1, 1], [2, 2]]])
    previous = torch.tensor([[[0, 1], [2, 3]]])
    candidate = torch.tensor([[[1, 0], [3, 2]]])
    targets = build_transition_targets(_labels_to_probs(previous), _labels_to_probs(candidate), ground_truth)
    assert targets.local.tolist() == [[[FIX, REGRESS], [REGRESS, FIX]]]
    assert UNCHANGED == 1


def test_counterfactual_mix_produces_valid_soft_transitions() -> None:
    previous = torch.softmax(torch.randn(4, 4, 16, 16), dim=1)
    ground_truth = torch.randint(0, 4, (4, 16, 16))
    generator = CounterfactualGenerator()
    for kind in ("positive", "negative", "hard_neutral"):
        sample = generator.generate(previous, ground_truth, kind=kind)
        assert sample.previous_probs.shape == sample.candidate_probs.shape
        assert torch.isfinite(sample.candidate_probs).all()
        assert torch.allclose(sample.candidate_probs.sum(dim=1), torch.ones(4, 16, 16), atol=1e-5)
        assert sample.candidate_probs.min() > 0


def test_threshold_gate_halts_after_rejected_transition() -> None:
    initial = torch.zeros(2, 1, 2, 2)

    def transition(state: torch.Tensor, turn: int):
        return state + 1.0, torch.tensor([-1.0, 1.0]) if turn == 0 else torch.tensor([-1.0, -1.0])

    result = ThresholdGate(tau_accept=0.0, t_max=3).run(initial, transition)
    assert result["accepted_turns"] == [0]
    assert result["halted_turns"] == [0, 1]
    assert torch.equal(result["state"][0], initial[0])
    assert torch.equal(result["state"][1], initial[1] + 1.0)
