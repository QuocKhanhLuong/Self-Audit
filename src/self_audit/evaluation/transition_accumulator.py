"""CPU streaming statistics for transition-audit validation.

Phase-B validation can visit thousands of transitions.  Keeping every local
logit tensor on the accelerator until the epoch ends causes the validation
loader to become an avoidable second training-sized memory consumer.  This
module keeps only a 3x3 local confusion matrix and the two scalar transition
vectors on CPU, then reconstructs the public metric dictionary at finalize.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..audit.semantics import BENEFICIAL, HARMFUL, NEUTRAL, classify_delta, resolve_neutral_margin
from .metrics import _safe_correlation, binary_auprc, binary_auroc


def _cpu_vector(value: Any, name: str) -> np.ndarray:
    """Detach a scalar vector to CPU and validate finite values."""

    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu")
        if getattr(value, "is_floating_point", lambda: False)():
            value = value.float()
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _cpu_confusion(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu")
        value = value.numpy()
    array = np.asarray(value)
    if array.shape != (3, 3):
        raise ValueError(f"local_confusion must be a 3x3 matrix, got {array.shape}")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("local_confusion must contain finite non-negative counts")
    if not np.equal(array, np.asarray(array, dtype=np.int64)).all():
        raise ValueError("local_confusion must contain integer counts")
    return np.asarray(array, dtype=np.int64).copy()


def _f1_from_confusion(confusion: np.ndarray, label: int) -> float:
    tp = float(confusion[label, label])
    fp = float(confusion[:, label].sum() - confusion[label, label])
    fn = float(confusion[label, :].sum() - confusion[label, label])
    denominator = 2.0 * tp + fp + fn
    return 1.0 if denominator == 0.0 else 2.0 * tp / denominator


class TransitionMetricAccumulator:
    """Accumulate transition metrics without retaining accelerator tensors."""

    def __init__(self) -> None:
        self.local_confusion = np.zeros((3, 3), dtype=np.int64)
        self._delta_q: list[np.ndarray] = []
        self._delta_dice: list[np.ndarray] = []

    @property
    def transition_count(self) -> int:
        return int(sum(values.size for values in self._delta_q))

    def update(self, local_confusion: Any, delta_pred: Any, delta_target: Any) -> None:
        """Add one batch of CPU-reduced local counts and scalar transitions."""

        confusion = _cpu_confusion(local_confusion)
        predicted = _cpu_vector(delta_pred, "delta_pred")
        target = _cpu_vector(delta_target, "delta_target")
        if predicted.shape != target.shape:
            raise ValueError(
                "delta_pred and delta_target must have the same number of transitions, "
                f"got {predicted.size} vs {target.size}"
            )
        self.local_confusion += confusion
        if predicted.size:
            self._delta_q.append(predicted)
            self._delta_dice.append(target)

    def merge(self, other: "TransitionMetricAccumulator") -> "TransitionMetricAccumulator":
        if not isinstance(other, TransitionMetricAccumulator):
            raise TypeError("other must be a TransitionMetricAccumulator")
        self.local_confusion += other.local_confusion
        self._delta_q.extend(values.copy() for values in other._delta_q)
        self._delta_dice.extend(values.copy() for values in other._delta_dice)
        return self

    def finalize(self, *, neutral_margin: float | None = None, tau: float = 0.0) -> dict[str, Any]:
        """Return the same public fields as :func:`transition_audit_metrics`."""

        margin = resolve_neutral_margin(neutral_margin)
        if self._delta_q:
            delta_q = np.concatenate(self._delta_q)
            delta_dice = np.concatenate(self._delta_dice)
        else:
            delta_q = np.empty((0,), dtype=np.float64)
            delta_dice = np.empty((0,), dtype=np.float64)
        if delta_q.size == 0:
            return {
                "improve_regress_accuracy": float("nan"),
                "auroc": float("nan"),
                "auprc": float("nan"),
                "correlation_delta_q_delta_dice": float("nan"),
                "local_fix_f1": float("nan"),
                "local_regress_f1": float("nan"),
                "neutral_count": 0,
                "beneficial_count": 0,
                "harmful_count": 0,
                "ranking_count": 0,
                "transition_count": 0,
                "neutral_margin": margin,
                "tau": float(tau),
            }

        labels = classify_delta(delta_dice, margin)
        non_neutral = labels != NEUTRAL
        ranking_scores = delta_q[non_neutral]
        ranking_labels = labels[non_neutral] == BENEFICIAL
        predicted_sign = ranking_scores > float(tau)
        accuracy = (
            float((predicted_sign == ranking_labels).mean()) if ranking_scores.size else float("nan")
        )
        return {
            "improve_regress_accuracy": accuracy,
            "auroc": binary_auroc(ranking_scores, ranking_labels),
            "auprc": binary_auprc(ranking_scores, ranking_labels),
            "correlation_delta_q_delta_dice": _safe_correlation(delta_q, delta_dice),
            "local_fix_f1": _f1_from_confusion(self.local_confusion, 0),
            "local_regress_f1": _f1_from_confusion(self.local_confusion, 2),
            "neutral_count": int((labels == NEUTRAL).sum()),
            "beneficial_count": int((labels == BENEFICIAL).sum()),
            "harmful_count": int((labels == HARMFUL).sum()),
            "ranking_count": int(ranking_scores.size),
            "transition_count": int(delta_q.size),
            "neutral_margin": margin,
            "tau": float(tau),
        }


__all__ = ["TransitionMetricAccumulator"]
