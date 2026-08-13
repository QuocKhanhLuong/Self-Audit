"""Annotation and transition-audit metrics for the Self-Audit baseline.

The functions in this module deliberately accept NumPy arrays as well as
PyTorch tensors.  That keeps volume-level reporting independent from the
training framework and makes small synthetic protocol checks easy to run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _labels(value: Any) -> np.ndarray:
    array = _numpy(value)
    if array.ndim >= 4:
        # Accept logits/probabilities in [B,C,...] and use the class axis.
        array = array.argmax(axis=1)
    return array.astype(np.int64, copy=False)


def _foreground_classes(num_classes: int, include_background: bool) -> range:
    return range(int(num_classes)) if include_background else range(1, int(num_classes))


def per_class_dice(
    prediction: Any,
    target: Any,
    num_classes: int = 4,
    include_background: bool = False,
    empty_score: float = 1.0,
) -> dict[int, float]:
    """Return Dice for each requested semantic class."""

    pred = _labels(prediction)
    true = _labels(target)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    scores: dict[int, float] = {}
    for cls in _foreground_classes(num_classes, include_background):
        p = pred == cls
        t = true == cls
        denom = int(p.sum() + t.sum())
        scores[cls] = float(empty_score if denom == 0 else 2.0 * (p & t).sum() / denom)
    return scores


def dice_score(
    prediction: Any,
    target: Any,
    num_classes: int = 4,
    include_background: bool = False,
    empty_score: float = 1.0,
) -> float:
    """Return mean multiclass Dice over foreground classes by default."""

    scores = per_class_dice(
        prediction,
        target,
        num_classes=num_classes,
        include_background=include_background,
        empty_score=empty_score,
    )
    return float(np.mean(list(scores.values()))) if scores else float(empty_score)


def per_class_precision_recall(
    prediction: Any,
    target: Any,
    num_classes: int = 4,
    include_background: bool = False,
    empty_score: float = 1.0,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return class-wise precision and recall with explicit empty handling."""

    pred = _labels(prediction)
    true = _labels(target)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    precision: dict[int, float] = {}
    recall: dict[int, float] = {}
    for cls in _foreground_classes(num_classes, include_background):
        p = pred == cls
        t = true == cls
        tp = int((p & t).sum())
        fp = int((p & ~t).sum())
        fn = int((~p & t).sum())
        precision[cls] = float(empty_score if tp + fp == 0 else tp / (tp + fp))
        recall[cls] = float(empty_score if tp + fn == 0 else tp / (tp + fn))
    return precision, recall


def _surface(mask: np.ndarray) -> np.ndarray:
    """Return surface voxels without requiring SciPy."""

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    interior = mask.copy()
    for axis in range(mask.ndim):
        for shift in (-1, 1):
            shifted = np.zeros_like(mask)
            source = [slice(None)] * mask.ndim
            destination = [slice(None)] * mask.ndim
            if shift < 0:
                source[axis] = slice(1, None)
                destination[axis] = slice(None, -1)
            else:
                source[axis] = slice(None, -1)
                destination[axis] = slice(1, None)
            shifted[tuple(destination)] = mask[tuple(source)]
            interior &= shifted
    return mask & ~interior


def _pairwise_min_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.empty((0,), dtype=np.float64)
    # Volumes in the intended baseline are small enough for this fallback.
    distances = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
    return distances.min(axis=1)


def surface_metrics(
    prediction: Any,
    target: Any,
    spacing: Iterable[float] | None = None,
    empty_score: float = 0.0,
) -> tuple[float, float]:
    """Return ``(HD95, ASSD)`` for binary masks.

    SciPy is used by the repository when available, but a NumPy fallback is
    included so protocol tests do not require the full medical-imaging stack.
    A one-sided empty mask is reported as ``inf`` rather than silently
    claiming a good surface score.
    """

    pred = _numpy(prediction).astype(bool)
    true = _numpy(target).astype(bool)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    if not pred.any() and not true.any():
        return float(empty_score), float(empty_score)
    if not pred.any() or not true.any():
        return float("inf"), float("inf")
    scale = np.ones(pred.ndim, dtype=np.float64)
    if spacing is not None:
        values = tuple(float(x) for x in spacing)
        if len(values) != pred.ndim:
            raise ValueError(f"spacing must have {pred.ndim} values, got {values}")
        scale[:] = values
    p_surface = np.argwhere(_surface(pred)).astype(np.float64) * scale
    t_surface = np.argwhere(_surface(true)).astype(np.float64) * scale
    p_to_t = _pairwise_min_distances(p_surface, t_surface)
    t_to_p = _pairwise_min_distances(t_surface, p_surface)
    all_distances = np.concatenate([p_to_t, t_to_p])
    return float(np.percentile(all_distances, 95)), float(all_distances.mean())


def annotation_metrics(
    prediction: Any,
    target: Any,
    num_classes: int = 4,
    spacing: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Compute the minimum volume-reporting metric set."""

    pred = _labels(prediction)
    true = _labels(target)
    dice = per_class_dice(pred, true, num_classes=num_classes)
    precision, recall = per_class_precision_recall(pred, true, num_classes=num_classes)
    hd95: dict[int, float] = {}
    assd: dict[int, float] = {}
    if spacing is None:
        spacing = (1.0,) * pred.ndim
    for cls in range(1, int(num_classes)):
        hd95[cls], assd[cls] = surface_metrics(pred == cls, true == cls, spacing=spacing)
    return {
        "dice": float(np.mean(list(dice.values()))) if dice else 0.0,
        "hd95": _finite_mean(hd95.values()),
        "assd": _finite_mean(assd.values()),
        "precision": float(np.mean(list(precision.values()))) if precision else 0.0,
        "recall": float(np.mean(list(recall.values()))) if recall else 0.0,
        "per_class": {
            "dice": dice,
            "hd95": hd95,
            "assd": assd,
            "precision": precision,
            "recall": recall,
        },
    }


def _finite_mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("inf")


def binary_auroc(scores: Any, labels: Any) -> float:
    """Dependency-free AUROC using the rank-sum definition."""

    score = _numpy(scores).reshape(-1).astype(np.float64)
    label = _numpy(labels).reshape(-1).astype(bool)
    positives = score[label]
    negatives = score[~label]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def binary_auprc(scores: Any, labels: Any) -> float:
    """Dependency-free average precision for binary transition labels."""

    score = _numpy(scores).reshape(-1).astype(np.float64)
    label = _numpy(labels).reshape(-1).astype(bool)
    positives = int(label.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    sorted_labels = label[order].astype(np.float64)
    precision = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def f1_for_label(prediction: Any, target: Any, label: int) -> float:
    pred = _numpy(prediction).reshape(-1) == int(label)
    true = _numpy(target).reshape(-1) == int(label)
    tp = float((pred & true).sum())
    fp = float((pred & ~true).sum())
    fn = float((~pred & true).sum())
    denom = 2.0 * tp + fp + fn
    return 1.0 if denom == 0.0 else 2.0 * tp / denom


def transition_audit_metrics(
    predicted_local: Any,
    target_local: Any,
    predicted_delta_q: Any,
    actual_delta_dice: Any,
) -> dict[str, float]:
    """Summarize transition-level and local audit quality."""

    local_pred = _labels(predicted_local)
    local_target = _labels(target_local)
    delta_q = _numpy(predicted_delta_q).reshape(-1)
    delta_dice = _numpy(actual_delta_dice).reshape(-1)
    predicted_sign = delta_q > 0.0
    actual_sign = delta_dice > 0.0
    return {
        "improve_regress_accuracy": float((predicted_sign == actual_sign).mean()),
        "auroc": binary_auroc(delta_q, actual_sign),
        "auprc": binary_auprc(delta_q, actual_sign),
        "correlation_delta_q_delta_dice": _safe_correlation(delta_q, delta_dice),
        "local_fix_f1": f1_for_label(local_pred, local_target, 0),
        "local_regress_f1": f1_for_label(local_pred, local_target, 2),
    }


def acceptance_metrics(accepted: Any, actual_delta_dice: Any, turns: Any | None = None) -> dict[str, float]:
    """Report gate outcomes required by the Self-Audit protocol."""

    accepted_array = _numpy(accepted).reshape(-1).astype(bool)
    delta = _numpy(actual_delta_dice).reshape(-1).astype(np.float64)
    if accepted_array.shape != delta.shape:
        raise ValueError("accepted and actual_delta_dice must have the same number of transitions")
    accepted_count = max(int(accepted_array.sum()), 1)
    rejected_count = max(int((~accepted_array).sum()), 1)
    harmful = accepted_array & (delta <= 0.0)
    beneficial_rejection = (~accepted_array) & (delta > 0.0)
    result = {
        "harmful_acceptance_rate": float(harmful.sum() / accepted_count),
        "beneficial_rejection_rate": float(beneficial_rejection.sum() / rejected_count),
        "net_dice_gain_after_auditing": float(delta[accepted_array].sum()) if accepted_array.any() else 0.0,
    }
    if turns is not None:
        result["mean_refinement_turns"] = float(_numpy(turns).astype(np.float64).mean())
    return result


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
