from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.self_audit.audit.gate import ThresholdGate
from src.self_audit.evaluation.metrics import annotation_metrics, transition_audit_metrics
from src.self_audit.evaluation.threshold import evaluate_threshold, select_threshold, sweep_thresholds
from src.self_audit.evaluation.volume_inference import canonicalize_depth_first
from src.self_audit.losses.audit import signed_ranking_loss
from src.self_audit.training.train_annotation import resolve_stage_weights
from src.self_audit.training._utils import (
    build_adamw_optimizer,
    build_warmup_cosine_scheduler,
    load_config,
    load_checkpoint,
    save_checkpoint,
)


def test_threshold_calibration_simulates_per_sample_halting() -> None:
    rows = sweep_thresholds(
        {
            "cache_schema_version": 1,
            "initial_dice": torch.tensor([0.50, 0.50]),
            "delta_q": torch.tensor([[0.10, -0.10], [0.10, -0.10]]),
            "actual_delta_dice": torch.tensor([[0.10, -0.20], [0.05, 0.02]]),
            "q_previous": torch.tensor([[0.50, 0.60], [0.50, 0.55]]),
            "q_candidate": torch.tensor([[0.60, 0.40], [0.55, 0.57]]),
            "active_mask": torch.ones(2, 2, dtype=torch.bool),
            "metric_contract": "foreground_dice_exclude_v1",
        },
        [0.0, 0.2],
    )
    best = select_threshold(rows)
    assert best["tau_accept"] == 0.0
    assert best["final_macro_dice"] == pytest.approx(0.575)
    assert best["mean_attempted_turns"] == pytest.approx(2.0)
    assert best["harmful_acceptance_rate"] == pytest.approx(0.0)


def test_threshold_calibration_rejects_metadata_only_delta_cache() -> None:
    from src.self_audit.evaluation.contracts import ContractMismatchError

    metadata_only_cache = {
        "cache_schema_version": 1,
        "initial_dice": torch.tensor([0.50, 0.50]),
        "delta_q": torch.tensor([[0.10, -0.10], [0.10, -0.10]]),
        "actual_delta_dice": torch.tensor([[0.10, -0.20], [0.05, 0.02]]),
        "active_mask": torch.ones(2, 2, dtype=torch.bool),
        "metric_contract": "foreground_dice_exclude_v1",
    }
    with pytest.raises(ContractMismatchError, match="Strict cache requires both 'q_previous' and 'q_candidate'"):
        sweep_thresholds(metadata_only_cache, [0.0, 0.2], strict_contract=True)


def test_threshold_calibration_rejects_nonfinite_cached_values() -> None:
    try:
        evaluate_threshold(0.0, [0.5], [[np.nan]], [[0.1]])
    except ValueError as exc:
        assert "NaN" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("non-finite cached transitions must fail")


def test_threshold_gate_reports_per_sample_bookkeeping() -> None:
    initial = torch.zeros(2, 1, 2, 2)

    def transition(state: torch.Tensor, turn: int):
        del state
        return torch.full_like(initial, float(turn + 1)), torch.tensor([-1.0, 1.0]) if turn == 0 else torch.tensor([1.0, -1.0])

    result = ThresholdGate(tau_accept=0.0, t_max=2).run(initial, transition)
    assert result["accepted_count"].tolist() == [0, 1]
    assert result["halt_turn"].tolist() == [0, 1]
    assert result["num_attempted_turns"].tolist() == [1, 2]
    assert len(result["transition_active_masks"]) == 2
    assert result["transition_active_masks"][0].tolist() == [True, True]
    assert result["transition_active_masks"][1].tolist() == [False, True]


def test_neutral_ranking_examples_are_calibrated_to_zero() -> None:
    value = signed_ranking_loss(torch.tensor([[0.4], [-0.2]]), torch.tensor([[0.001], [-0.002]]), neutral_margin=0.005)
    assert torch.isfinite(value)
    assert float(value) > 0.0


def test_phase_a_stage_weights_follow_locked_baseline_and_generalize() -> None:
    assert resolve_stage_weights(4) == [0.5, 0.7, 0.8, 1.0]
    assert resolve_stage_weights(2) == [0.5, 0.7]
    assert resolve_stage_weights(5) == [0.5, 0.7, 0.8, 1.0, 1.0]
    assert resolve_stage_weights(3, [1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_explicit_depth_axes_produce_depth_first_contract() -> None:
    volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    assert np.array_equal(canonicalize_depth_first(volume, depth_axis=0), volume)
    assert canonicalize_depth_first(np.moveaxis(volume, 0, 2), depth_axis=2).shape == (2, 3, 4)


def test_checked_in_acdc_npy_convention_is_explicit() -> None:
    config = load_config("configs/self_audit_annotation.yaml")
    assert config["depth_axis"] == 2


def test_metrics_use_consistent_both_empty_dice_convention() -> None:
    """A class absent from both prediction and target is excluded, not scored 1.0.

    This test previously pinned the pre-remediation convention
    (``empty_score=1.0``).  That convention inflated every macro: on ACDC,
    apical and basal slices routinely contain no RV or MYO at all, so a class
    that is empty in both prediction and target was contributing a perfect
    score while carrying no information about segmentation quality.  The
    default is now ``empty_policy="exclude"`` (nan, dropped from the macro).
    The original intent of the test -- that the convention is *consistent* --
    is preserved by also checking the legacy opt-in still works.
    """

    prediction = np.zeros((8, 8), dtype=np.int64)
    target = np.zeros((8, 8), dtype=np.int64)

    result = annotation_metrics(prediction, target, num_classes=4)
    assert all(np.isnan(value) for value in result["per_class"]["dice"].values())
    # Every foreground class excluded => the macro is nan, not 1.0 and not 0.0.
    assert np.isnan(result["dice"])

    legacy = annotation_metrics(prediction, target, num_classes=4, empty_policy="legacy_one")
    assert legacy["per_class"]["dice"] == {1: 1.0, 2: 1.0, 3: 1.0}
    assert legacy["dice"] == pytest.approx(1.0)


def test_class_present_in_target_but_missing_from_prediction_is_never_excluded() -> None:
    """Exclusion applies only when BOTH sides are empty; a total miss scores 0.0."""

    prediction = np.zeros((8, 8), dtype=np.int64)
    target = np.zeros((8, 8), dtype=np.int64)
    target[2:5, 2:5] = 1  # class 1 present in GT, absent from the prediction

    for policy in ("exclude", "legacy_one"):
        result = annotation_metrics(prediction, target, num_classes=4, empty_policy=policy)
        assert result["per_class"]["dice"][1] == pytest.approx(0.0)


def test_transition_metrics_accept_cpu_bfloat16_outputs() -> None:
    local = torch.zeros(1, 3, 4, 4, dtype=torch.bfloat16)
    target_local = torch.zeros(1, 4, 4, dtype=torch.long)
    result = transition_audit_metrics(
        local,
        target_local,
        torch.tensor([[0.1]], dtype=torch.bfloat16),
        torch.tensor([[0.2]], dtype=torch.bfloat16),
    )
    assert result["improve_regress_accuracy"] == pytest.approx(1.0)
    assert np.isfinite(result["auprc"])


def test_checkpoint_round_trip_restores_model_optimizer_and_progress(tmp_path) -> None:
    model = nn.Linear(3, 2)
    optimizer = build_adamw_optimizer(model.parameters(), lr=1e-3)
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=4, warmup_steps=1)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer=optimizer, scheduler=scheduler, epoch=3, global_step=7, optimizer_step=4, config={"model": {"num_classes": 2}})
    restored = nn.Linear(3, 2)
    restored_optimizer = build_adamw_optimizer(restored.parameters(), lr=1e-3)
    restored_scheduler = build_warmup_cosine_scheduler(restored_optimizer, total_steps=4, warmup_steps=1)
    payload = load_checkpoint(path, model=restored, optimizer=restored_optimizer, scheduler=restored_scheduler)
    assert payload["epoch"] == 3
    assert payload["global_step"] == 7
    assert payload["optimizer_step"] == 4
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
