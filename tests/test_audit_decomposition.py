from __future__ import annotations

import math

import pytest
import torch

from src.self_audit.evaluation.audit_decomposition import (
    decompose_annotation_output,
    decompose_self_audit_output,
)


def _logits(labels: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).float() * 8.0


def test_stage_attribution_separates_candidate_and_gate_value() -> None:
    gt = torch.ones(2, 2, 2, dtype=torch.long)
    previous_labels = torch.stack(
        [
            torch.zeros(2, 2, dtype=torch.long),
            torch.ones(2, 2, dtype=torch.long),
        ]
    )
    candidate_labels = torch.stack(
        [
            torch.ones(2, 2, dtype=torch.long),
            torch.zeros(2, 2, dtype=torch.long),
        ]
    )
    previous = _logits(previous_labels)
    candidate = _logits(candidate_labels)
    output = {
        "transition_previous": [previous],
        "transition_candidates": [candidate],
        "transition_active_masks": [torch.tensor([True, True])],
        "transition_state_masks": [torch.tensor([True, False])],
        "audits": [{"accepted": torch.tensor([True, False])}],
    }

    metrics = decompose_self_audit_output(output, gt, neutral_margin=0.005)
    assert metrics["stage_0/beneficial_capture_rate"] == pytest.approx(1.0)
    assert metrics["stage_0/harmful_block_rate"] == pytest.approx(1.0)
    assert metrics["stage_0/audit_gate_value"] > 0.0
    assert metrics["stage_0/realized_gain"] > metrics["stage_0/candidate_gain"]
    assert metrics["stage_0/attribution_residual"] < 1e-7


def test_phase_a_headroom_reports_a0_saturation() -> None:
    gt = torch.ones(1, 2, 2, dtype=torch.long)
    a0 = _logits(gt)
    a1 = _logits(gt)
    output = {
        "initial_logits": a0,
        "state_trace": [a0, a1],
        "states": [a1],
    }
    metrics = decompose_annotation_output(output, gt)
    assert metrics["phase_a/a0_dice"] == pytest.approx(1.0)
    assert metrics["phase_a/total_refinement_gain"] == pytest.approx(0.0)
    assert metrics["phase_a/stage_0/headroom"] == pytest.approx(0.0)
    assert metrics["phase_a/headroom_collapse_ratio"] == pytest.approx(1.0)


def test_neutral_candidate_is_not_counted_as_beneficial_or_harmful() -> None:
    gt = torch.ones(1, 2, 2, dtype=torch.long)
    state = _logits(gt)
    output = {
        "transition_previous": [state],
        "transition_candidates": [state.clone()],
        "transition_active_masks": [torch.tensor([True])],
        "transition_state_masks": [torch.tensor([True])],
    }
    metrics = decompose_self_audit_output(output, gt, neutral_margin=0.005)
    assert metrics["stage_0/neutral_candidate_rate"] == pytest.approx(1.0)
    assert metrics["stage_0/beneficial_candidate_rate"] == pytest.approx(0.0)
    assert metrics["stage_0/harmful_candidate_rate"] == pytest.approx(0.0)
    assert math.isnan(metrics["stage_0/beneficial_capture_rate"])
    assert math.isnan(metrics["stage_0/harmful_block_rate"])
