from __future__ import annotations

import numpy as np
import pytest
import torch

from self_audit.evaluation.metrics import transition_audit_metrics
from self_audit.evaluation.transition_accumulator import TransitionMetricAccumulator


def _confusion(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = np.zeros((3, 3), dtype=np.int64)
    for pred, true in zip(prediction.reshape(-1), target.reshape(-1)):
        result[int(true), int(pred)] += 1
    return result


def test_chunked_accumulator_matches_transition_metrics() -> None:
    local_pred = np.array([0, 1, 2, 0, 2, 1, 0, 2])
    local_target = np.array([0, 2, 2, 1, 2, 1, 0, 1])
    delta_q = np.array([0.8, -0.4, 0.01, 0.2, -0.8, 0.0, 0.3, -0.2], dtype=np.float64)
    delta_dice = np.array([0.4, -0.2, 0.001, 0.1, -0.4, 0.0, 0.2, -0.1], dtype=np.float64)
    expected = transition_audit_metrics(
        local_pred,
        local_target,
        delta_q,
        delta_dice,
        neutral_margin=0.005,
    )

    accumulator = TransitionMetricAccumulator()
    accumulator.update(_confusion(local_pred[:4], local_target[:4]), delta_q[:4], delta_dice[:4])
    accumulator.update(_confusion(local_pred[4:], local_target[4:]), delta_q[4:], delta_dice[4:])
    actual = accumulator.finalize(neutral_margin=0.005)

    for key, value in expected.items():
        if isinstance(value, float) and np.isnan(value):
            assert np.isnan(actual[key])
        elif isinstance(value, float):
            assert actual[key] == pytest.approx(value)
        else:
            assert actual[key] == value


def test_accumulator_merge_preserves_counts() -> None:
    left = TransitionMetricAccumulator()
    right = TransitionMetricAccumulator()
    left.update(np.eye(3, dtype=np.int64), torch.tensor([0.1]), torch.tensor([0.2]))
    right.update(np.eye(3, dtype=np.int64), torch.tensor([-0.1]), torch.tensor([-0.2]))
    left.merge(right)
    result = left.finalize(neutral_margin=0.005)
    assert result["transition_count"] == 2
    assert result["beneficial_count"] == 1
    assert result["harmful_count"] == 1


def test_accumulator_rejects_invalid_confusion_shape() -> None:
    with pytest.raises(ValueError, match="3x3"):
        TransitionMetricAccumulator().update(np.zeros((2, 2), dtype=np.int64), [], [])


def test_empty_accumulator_preserves_nan_population_metrics() -> None:
    result = TransitionMetricAccumulator().finalize(neutral_margin=0.005)
    assert result["transition_count"] == 0
    assert np.isnan(result["auroc"])
    assert np.isnan(result["local_fix_f1"])
    assert np.isnan(result["local_regress_f1"])
