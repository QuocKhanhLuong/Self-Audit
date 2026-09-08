"""Annotation and transition-audit metrics for the Self-Audit baseline.

The functions in this module deliberately accept NumPy arrays as well as
PyTorch tensors.  That keeps volume-level reporting independent from the
training framework and makes small synthetic protocol checks easy to run.

Two semantic rules are imported from :mod:`self_audit.audit.semantics` rather
than re-implemented here, so the project has exactly one copy of each:

* **Signed transition quality.**  ``delta > +eps`` is BENEFICIAL,
  ``delta < -eps`` is HARMFUL, ``abs(delta) <= eps`` is NEUTRAL, with
  ``eps = DEFAULT_NEUTRAL_MARGIN``.  Neutral transitions are *neither* class:
  they are dropped from ranking populations (AUROC/AUPRC/accuracy) and from
  the acceptance-rate denominators instead of being silently counted as
  harmful, which is what the old ``delta <= 0.0`` test did.
* **Empty-class policy.**  A foreground class absent from *both* prediction
  and target carries no information, so by default it is excluded from the
  macro (score ``nan``) instead of scoring ``1.0``.  A class present in
  exactly one of prediction/target scores ``0.0`` by the ordinary Dice
  formula and is **never** excluded -- excluding it would hide a total miss.

**Metric space.**  :func:`slice_proxy_dice` computes 2-D per-slice foreground
macro Dice on the network grid.  That is
:data:`~self_audit.audit.semantics.METRIC_SPACE_SLICE_PROXY`: a training and
monitoring proxy, *not* a paper metric.  Quotable numbers are per-volume
(``volume_resized`` / ``volume_native``) and live in ``volume_inference``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
import warnings

import numpy as np

from ..audit.semantics import (
    BENEFICIAL,
    DEFAULT_NEUTRAL_MARGIN,
    HARMFUL,
    NEUTRAL,
    METRIC_SPACE_SLICE_PROXY,
    beneficial_mask,
    classify_delta,
    empty_class_score,
    harmful_mask,
    macro_mean,
    neutral_mask,
    resolve_empty_policy,
    resolve_neutral_margin,
)


#: Sentinel marking "caller did not pass the deprecated ``empty_score``".
class _Unset:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<unset>"


_UNSET = _Unset()

#: Metric-space label for :func:`slice_proxy_dice` (training proxy, not a
#: paper metric).  Re-exported so callers can stamp reports without importing
#: the semantics module directly.
SLICE_PROXY_METRIC_SPACE = METRIC_SPACE_SLICE_PROXY


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # NumPy has no bfloat16 dtype.  CPU AMP validation can legitimately
        # produce bfloat16 Auditor logits, so normalize floating tensors at
        # this boundary without changing integer label semantics.
        if getattr(value, "is_floating_point", lambda: False)():
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def _labels(value: Any) -> np.ndarray:
    array = _numpy(value)
    if array.ndim >= 4:
        # Accept logits/probabilities in [B,C,...] and use the class axis.
        array = array.argmax(axis=1)
    return array.astype(np.int64, copy=False)


def _foreground_classes(num_classes: int, include_background: bool) -> range:
    return range(int(num_classes)) if include_background else range(1, int(num_classes))


def _empty_values(
    empty_score: Any,
    empty_policy: str | None,
    *,
    function: str,
) -> tuple[float, float]:
    """Resolve ``(both_empty_value, one_sided_empty_value)``.

    Under a policy, only the both-empty case is special: a one-sided empty
    class scores ``0.0`` (the ordinary Dice/precision/recall value for a total
    miss) and is never excluded.  When the deprecated ``empty_score`` is passed
    explicitly it wins and reproduces the pre-remediation behaviour exactly:
    *any* zero denominator takes that value.
    """

    if isinstance(empty_score, _Unset):
        resolve_empty_policy(empty_policy)  # validate the name eagerly
        return empty_class_score(empty_policy), 0.0
    warnings.warn(
        f"{function}(empty_score=...) is deprecated and overrides empty_policy; "
        f"pass empty_policy='exclude' (default), 'legacy_one', or 'zero' instead. "
        f"empty_score also applies to one-sided-empty classes, which the policies "
        f"deliberately score 0.0 so a total miss cannot be hidden.",
        DeprecationWarning,
        stacklevel=3,
    )
    value = float(empty_score)
    return value, value


def per_class_dice(
    prediction: Any,
    target: Any,
    num_classes: int = 4,
    include_background: bool = False,
    empty_score: Any = _UNSET,
    *,
    empty_policy: str | None = None,
) -> dict[int, float]:
    """Return Dice for each requested semantic class.

    By default a class empty in both prediction and target is *excluded*
    (``nan``); callers must aggregate with :func:`macro_mean` so the value does
    not silently become ``1.0`` or ``0.0``.  ``empty_score`` remains accepted
    as a deprecated explicit override.
    """

    both_empty, _one_sided = _empty_values(empty_score, empty_policy, function="per_class_dice")
    pred = _labels(prediction)
    true = _labels(target)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    scores: dict[int, float] = {}
    for cls in _foreground_classes(num_classes, include_background):
        p = pred == cls
        t = true == cls
        denom = int(p.sum() + t.sum())
        # denom == 0 happens only when the class is empty in BOTH arrays.
        scores[cls] = float(both_empty if denom == 0 else 2.0 * (p & t).sum() / denom)
    return scores


def dice_score(
    prediction: Any,
    target: Any,
    num_classes: int = 4,
    include_background: bool = False,
    empty_score: Any = _UNSET,
    *,
    empty_policy: str | None = None,
) -> float:
    """Return the nan-aware mean multiclass Dice over foreground classes.

    If every class was excluded the result is ``nan`` -- never ``1.0``.
    """

    scores = per_class_dice(
        prediction,
        target,
        num_classes=num_classes,
        include_background=include_background,
        empty_score=empty_score,
        empty_policy=empty_policy,
    )
    if not scores and not isinstance(empty_score, _Unset):
        return float(empty_score)
    return macro_mean(list(scores.values()))


def per_class_precision_recall(
    prediction: Any,
    target: Any,
    num_classes: int = 4,
    include_background: bool = False,
    empty_score: Any = _UNSET,
    *,
    empty_policy: str | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return class-wise precision and recall with explicit empty handling.

    Both-empty classes take the policy value (``nan`` under ``"exclude"``).  A
    class that is empty on exactly one side has an undefined ratio; it scores
    ``0.0``, matching Dice, so a class the model never predicts cannot inflate
    precision, and a class the model hallucinates cannot inflate recall.
    """

    both_empty, one_sided = _empty_values(
        empty_score, empty_policy, function="per_class_precision_recall"
    )
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
        class_empty = (tp + fp + fn) == 0
        empty_value = both_empty if class_empty else one_sided
        precision[cls] = float(empty_value if tp + fp == 0 else tp / (tp + fp))
        recall[cls] = float(empty_value if tp + fn == 0 else tp / (tp + fn))
    return precision, recall


def slice_proxy_dice(
    prediction: Any,
    target: Any,
    *,
    num_classes: int = 4,
    empty_policy: str | None = None,
) -> np.ndarray:
    """Return per-sample foreground macro Dice, shape ``[B]``.

    This is the single 2-D training proxy (metric space
    ``"slice_proxy"``): Phase A and Phase C both call it so their headline
    numbers are comparable.  It is computed **per sample and then averaged by
    the caller**, which makes it invariant to how a set of slices is split into
    batches -- unlike pooling confusion counts over a whole ``[B,H,W]`` block,
    which makes the result batch-size dependent.

    ``prediction`` may be ``[B,C,H,W]`` logits/probabilities (argmax over the
    class axis) or ``[B,H,W]`` labels; ``target`` is ``[B,H,W]`` labels.  A
    sample whose foreground classes are all empty in both prediction and
    target yields ``nan`` under the default ``"exclude"`` policy; aggregate
    with :func:`~self_audit.audit.semantics.macro_mean` or ``np.nanmean``.
    """

    pred = _labels(prediction)
    true = _labels(target)
    if pred.ndim == 2:
        pred = pred[None]
    if true.ndim == 2:
        true = true[None]
    if pred.ndim < 3 or true.ndim < 3:
        raise ValueError(
            f"slice_proxy_dice expects [B,C,H,W] or [B,H,W]; got {pred.shape} vs {true.shape}"
        )
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    scores = np.empty(pred.shape[0], dtype=np.float64)
    for index in range(pred.shape[0]):
        per_class = per_class_dice(
            pred[index],
            true[index],
            num_classes=num_classes,
            include_background=False,
            empty_policy=empty_policy,
        )
        scores[index] = macro_mean(list(per_class.values()))
    return scores


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

    ``empty_score`` here is a *distance*, not a Dice value, so it is untouched
    by the empty-class policy.
    """

    pred = _numpy(prediction).astype(bool)
    true = _numpy(target).astype(bool)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    scale = np.ones(pred.ndim, dtype=np.float64)
    if spacing is not None:
        values = tuple(float(x) for x in spacing)
        if len(values) != pred.ndim:
            raise ValueError(f"spacing must have {pred.ndim} values, got {values}")
        if not all(np.isfinite(x) and x > 0.0 for x in values):
            raise ValueError(f"spacing must be positive and finite, got {values}")
        scale[:] = values
    if not pred.any() and not true.any():
        return float(empty_score), float(empty_score)
    if not pred.any() or not true.any():
        return float("inf"), float("inf")
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
    spacing_known: bool | None = None,
    *,
    empty_policy: str | None = None,
    empty_score: Any = _UNSET,
) -> dict[str, Any]:
    """Compute reporting metrics with explicit physical-vs-pixel semantics.

    When physical spacing is unavailable, HD95/ASSD are reported in pixel
    units and the result includes ``distance_space='pixel'``.  Physical metrics
    require both spacing values and ``spacing_known=True``.

    Note the key is ``distance_space``, not ``metric_space``.  ``metric_space``
    is reserved project-wide for the grid/aggregation axis
    (``slice_proxy`` / ``volume_resized`` / ``volume_native``, see
    ``self_audit.audit.semantics``).  Distance units are an independent axis:
    a ``volume_native`` Dice can carry either ``physical`` or ``pixel``
    surface distances.  Callers such as ``evaluate_comparison_modes`` nest this
    dict inside a block that stamps its own ``metric_space``, so reusing the
    name here would produce two different meanings in one result.

    Headline ``dice`` / ``precision`` / ``recall`` are nan-aware macros over
    the per-class values, so an excluded class neither inflates nor deflates
    them; ``nan`` means every foreground class was excluded.
    """

    pred = _labels(prediction)
    true = _labels(target)
    dice = per_class_dice(
        pred, true, num_classes=num_classes, empty_score=empty_score, empty_policy=empty_policy
    )
    precision, recall = per_class_precision_recall(
        pred, true, num_classes=num_classes, empty_score=empty_score, empty_policy=empty_policy
    )
    hd95: dict[int, float] = {}
    assd: dict[int, float] = {}
    if spacing_known is None:
        spacing_known = spacing is not None
    if spacing_known and spacing is None:
        raise ValueError("spacing_known=True requires physical spacing metadata")
    distance_space = "physical" if spacing_known else "pixel"
    if spacing_known:
        values = tuple(float(x) for x in spacing)  # type: ignore[arg-type]
        if len(values) != pred.ndim:
            raise ValueError(f"spacing must have {pred.ndim} values, got {values}")
        if not all(np.isfinite(x) and x > 0.0 for x in values):
            raise ValueError(f"physical spacing must be positive and finite, got {values}")
        metric_spacing: tuple[float, ...] | None = values
    else:
        metric_spacing = (1.0,) * pred.ndim
    for cls in range(1, int(num_classes)):
        hd95[cls], assd[cls] = surface_metrics(pred == cls, true == cls, spacing=metric_spacing)
    return {
        "dice": macro_mean(list(dice.values())),
        "hd95": _finite_mean(hd95.values()),
        "assd": _finite_mean(assd.values()),
        "precision": macro_mean(list(precision.values())),
        "recall": macro_mean(list(recall.values())),
        "spacing_known": bool(spacing_known),
        "distance_space": distance_space,
        "empty_policy": (
            "explicit_empty_score"
            if not isinstance(empty_score, _Unset)
            else resolve_empty_policy(empty_policy)
        ),
        "physical_metrics_available": bool(spacing_known),
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
    *,
    neutral_margin: float | None = None,
    tau: float = 0.0,
) -> dict[str, Any]:
    """Summarize transition-level and local audit quality.

    The ranking task is *beneficial vs harmful*.  Transitions whose measured
    ``abs(delta_dice) <= neutral_margin`` are NEUTRAL: they belong to neither
    class, so they are dropped from the AUROC/AUPRC/accuracy population rather
    than being folded into the negative class by a ``delta > 0.0`` test.  Their
    number is reported as ``neutral_count`` so the drop is never invisible.

    ``tau`` is the decision threshold applied to the predicted score for the
    hard accuracy figure only; it does not affect the ranking metrics.

    ``correlation_delta_q_delta_dice`` is computed over the **full** population
    (it is a correlation of continuous values and needs no margin), and the two
    local F1 figures are pixel-label metrics that the margin does not touch.
    """

    margin = resolve_neutral_margin(neutral_margin)
    local_pred = _labels(predicted_local)
    local_target = _labels(target_local)
    delta_q = _numpy(predicted_delta_q).reshape(-1).astype(np.float64)
    delta_dice = _numpy(actual_delta_dice).reshape(-1).astype(np.float64)
    if delta_q.shape != delta_dice.shape:
        raise ValueError(
            "predicted_delta_q and actual_delta_dice must have the same number of transitions, "
            f"got {delta_q.size} vs {delta_dice.size}"
        )
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
        "local_fix_f1": f1_for_label(local_pred, local_target, 0),
        "local_regress_f1": f1_for_label(local_pred, local_target, 2),
        "neutral_count": int((labels == NEUTRAL).sum()),
        "beneficial_count": int((labels == BENEFICIAL).sum()),
        "harmful_count": int((labels == HARMFUL).sum()),
        "ranking_count": int(ranking_scores.size),
        "transition_count": int(delta_q.size),
        "neutral_margin": margin,
        "tau": float(tau),
    }


def acceptance_metrics(
    accepted: Any,
    actual_delta_dice: Any,
    turns: Any | None = None,
    *,
    neutral_margin: float | None = None,
) -> dict[str, Any]:
    """Report gate outcomes required by the Self-Audit protocol.

    NEUTRAL transitions are excluded from every *numerator*: accepting an edit
    that changed nothing measurable is neither a harmful acceptance nor a
    missed benefit.  The headline denominators, however, are the full accepted
    and rejected populations.

    That denominator choice is deliberate and it matters.  Restricting the
    denominator to materially signed transitions answers a discriminative
    question ("when there was a real call to make, how often was it wrong?"),
    but it is the wrong headline: a gate that accepts a thousand no-op edits
    and one harmful edit would report a 100% harmful-acceptance rate.  That
    both misstates safety and, because ``select_threshold`` tie-breaks on this
    rate, would push the calibrated ``tau_accept`` *more* conservative than the
    original zero-boundary bug did -- the opposite of the intended fix.

    * ``harmful_acceptance_rate`` -- of **all** accepted transitions, the
      fraction that were harmful.  Headline; consumed by threshold selection.
    * ``beneficial_rejection_rate`` -- of **all** rejected transitions, the
      fraction that were beneficial.  Headline.
    * ``harmful_acceptance_rate_signed`` / ``beneficial_rejection_rate_signed``
      -- the same quantities conditioned on materially signed transitions.
      Diagnostic companions, not used for selection.
    * ``beneficial_acceptance_rate`` -- share of all acceptances that were
      beneficial.
    * ``neutral_acceptance_rate`` -- share of all acceptances that were
      neutral, i.e. how much of the gate's traffic is no-op.

    ``net_dice_gain_after_auditing`` is a raw sum over accepted transitions and
    is margin-free by definition; it is unchanged.

    Raw counts (``harmful_accepted_count``, ``beneficial_rejected_count``,
    ``neutral_accepted_count``, ``accepted_count``, ...) are returned so any
    caller can re-aggregate across cohorts, or recover a rate under a different
    denominator convention, without re-deriving the classification.
    """

    margin = resolve_neutral_margin(neutral_margin)
    accepted_array = _numpy(accepted).reshape(-1).astype(bool)
    delta = _numpy(actual_delta_dice).reshape(-1).astype(np.float64)
    if accepted_array.shape != delta.shape:
        raise ValueError("accepted and actual_delta_dice must have the same number of transitions")
    if accepted_array.size == 0:
        result: dict[str, Any] = {
            "harmful_acceptance_rate": 0.0,
            "beneficial_rejection_rate": 0.0,
            "harmful_acceptance_rate_signed": 0.0,
            "beneficial_rejection_rate_signed": 0.0,
            "net_dice_gain_after_auditing": 0.0,
            "beneficial_acceptance_rate": 0.0,
            "neutral_acceptance_rate": 0.0,
            "neutral_count": 0,
            "beneficial_count": 0,
            "harmful_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "harmful_accepted_count": 0,
            "beneficial_accepted_count": 0,
            "neutral_accepted_count": 0,
            "beneficial_rejected_count": 0,
            "transition_count": 0,
            "neutral_margin": margin,
        }
        if turns is not None:
            result["mean_refinement_turns"] = 0.0
        return result

    rejected_array = ~accepted_array
    neutral = neutral_mask(delta, margin)
    beneficial = beneficial_mask(delta, margin)
    harmful = harmful_mask(delta, margin)

    accepted_count = int(accepted_array.sum())
    rejected_count = int(rejected_array.sum())
    accepted_signed = int((accepted_array & ~neutral).sum())
    rejected_signed = int((rejected_array & ~neutral).sum())
    harmful_accepted = int((accepted_array & harmful).sum())
    beneficial_accepted = int((accepted_array & beneficial).sum())
    neutral_accepted = int((accepted_array & neutral).sum())
    beneficial_rejected = int((rejected_array & beneficial).sum())

    result = {
        "harmful_acceptance_rate": float(harmful_accepted / max(accepted_count, 1)),
        "beneficial_rejection_rate": float(beneficial_rejected / max(rejected_count, 1)),
        "harmful_acceptance_rate_signed": float(harmful_accepted / max(accepted_signed, 1)),
        "beneficial_rejection_rate_signed": float(beneficial_rejected / max(rejected_signed, 1)),
        "net_dice_gain_after_auditing": (
            float(delta[accepted_array].sum()) if accepted_array.any() else 0.0
        ),
        "beneficial_acceptance_rate": float(beneficial_accepted / max(accepted_count, 1)),
        "neutral_acceptance_rate": float(neutral_accepted / max(accepted_count, 1)),
        "neutral_count": int(neutral.sum()),
        "beneficial_count": int(beneficial.sum()),
        "harmful_count": int(harmful.sum()),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "accepted_signed_count": accepted_signed,
        "rejected_signed_count": rejected_signed,
        "harmful_accepted_count": harmful_accepted,
        "beneficial_accepted_count": beneficial_accepted,
        "neutral_accepted_count": neutral_accepted,
        "beneficial_rejected_count": beneficial_rejected,
        "transition_count": int(delta.size),
        "neutral_margin": margin,
    }
    if turns is not None:
        result["mean_refinement_turns"] = float(_numpy(turns).astype(np.float64).mean())
    return result


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
