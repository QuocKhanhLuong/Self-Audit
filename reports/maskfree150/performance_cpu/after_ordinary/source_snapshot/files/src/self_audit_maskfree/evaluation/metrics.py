"""Metric primitives for the mask-free pipeline (W6).

Scope
-----
Two very different kinds of number live in this package and they are kept
separate on purpose:

* **Image-only numbers** (coverage, ambiguity, predictive NLL, stability). These
  may be computed at any time and never touch a manual mask.
* **Reference numbers** (Dice, IoU, HD95, ASSD, true FIX/REGRESS attribution).
  These require manual segmentation masks and are therefore only ever called
  from :mod:`self_audit_maskfree.evaluation.reference`, which is a standalone
  post-freeze process. Nothing in this module reads a file; callers hand in
  arrays that the isolated evaluator already loaded.

Contract obligations implemented here
-------------------------------------
* Every reported number is wrapped by :func:`metric_row` with dataset, split,
  protocol, epoch, checkpoint, population, count, availability, unit and
  contract version. A missing metric is ``available=False`` with a reason and a
  ``None`` value; it is never silently converted to zero.
* Dice/IoU: both-empty classes are *excluded* from the aggregate and counted
  separately; one-empty classes count as an error (score 0.0). There is no
  undefined-to-zero conversion anywhere else.
* Surface metrics require valid native geometry (finite positive spacing). With
  stored-grid data only, HD95/ASSD are reported as unavailable rather than
  relabelled pixel distances; a caller that explicitly wants pixel units must
  pass ``spacing=(1,1,1)`` *and* gets ``unit='pixel'`` in the row.
* Bootstrap resampling is at the patient level. Slices are never resampled as
  independent units.
* A score margin is not a probability. Nothing here emits ECE/Brier or any
  calibration claim.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..contracts import SEMANTIC_ORDER, VERSION

#: Foreground classes in the frozen named semantic order (0=BG, 1=RV, 2=MYO, 3=LV).
FOREGROUND_CLASSES: tuple[int, ...] = (1, 2, 3)
#: Preregistered coverage levels for coverage-versus-error curves.
COVERAGE_LEVELS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
#: Minimum number of defined (positive, negative) pairs before a ranking metric
#: is reported at all. Below this the row is unavailable, not optimistic.
MIN_RANKING_SUPPORT = 3


class MetricContractError(ValueError):
    """Raised when a metric call violates the frozen maskfree150 metric contract."""


# ----------------------------------------------------------------------------
# row schema
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricRow:
    """One fully provenanced metric observation.

    ``value`` is ``None`` whenever ``available`` is ``False``. ``count`` is the
    number of units that actually entered the number (patients, pixels, pairs -
    whatever ``population`` names), so a reader can always see the denominator.
    """

    name: str
    value: float | None
    unit: str
    dataset: str
    split: str
    protocol: str
    epoch: int | None
    checkpoint: str
    population: str
    count: int
    available: bool = True
    reason: str | None = None
    contract_version: str = VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def metric_row(
    name: str,
    value: float | None,
    *,
    unit: str,
    dataset: str,
    split: str,
    protocol: str,
    epoch: int | None,
    checkpoint: str,
    population: str,
    count: int,
    available: bool = True,
    reason: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build one metric row dict. Shared by every W6 report and by W5/W7 callers.

    An unavailable metric must carry a reason, and an available metric must
    carry a finite value: this function refuses the two states that hide a
    fabricated number.
    """
    if available and (value is None or not math.isfinite(float(value))):
        raise MetricContractError(
            f"metric {name!r} claims availability but has no finite value"
        )
    if not available:
        if value is not None:
            raise MetricContractError(
                f"unavailable metric {name!r} must carry value None, not {value!r}"
            )
        if not reason:
            raise MetricContractError(f"unavailable metric {name!r} must state a reason")
    if count < 0:
        raise MetricContractError("count must be non-negative")
    row = MetricRow(
        name=name,
        value=None if value is None else float(value),
        unit=unit,
        dataset=dataset,
        split=split,
        protocol=protocol,
        epoch=epoch,
        checkpoint=checkpoint,
        population=population,
        count=int(count),
        available=bool(available),
        reason=reason,
        extra=dict(extra),
    )
    return row.to_dict()


# ----------------------------------------------------------------------------
# volume overlap
# ----------------------------------------------------------------------------
def _as_label_volume(volume: Any, name: str) -> np.ndarray:
    array = np.asarray(volume)
    if array.ndim != 3:
        raise MetricContractError(f"{name} must be a 3-D [Z,H,W] label volume, got {array.ndim}-D")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.equal(np.mod(array, 1), 0)):
            raise MetricContractError(f"{name} must contain integer class ids")
        array = array.astype(np.int64)
    return array.astype(np.int64)


def dice_iou_volume(
    prediction: Any,
    reference: Any,
    *,
    classes: Sequence[int] = FOREGROUND_CLASSES,
) -> dict[int, dict[str, Any]]:
    """Per-class 3-D Dice and IoU for one patient volume, with explicit emptiness.

    Returned per class:

    ``defined``
        ``False`` when both prediction and reference are empty for that class.
        Such a class is excluded from aggregates and counted as ``both_empty``.
    ``one_empty``
        Exactly one side empty. This is an error, scored 0.0, and stays in the
        aggregate. It is never converted to "undefined" to protect the mean.

    The class mapping is the frozen named order; no per-case permutation or
    Hungarian matching happens anywhere in this function.
    """
    predicted = _as_label_volume(prediction, "prediction")
    truth = _as_label_volume(reference, "reference")
    if predicted.shape != truth.shape:
        raise MetricContractError(
            f"prediction {predicted.shape} and reference {truth.shape} must share a grid"
        )
    results: dict[int, dict[str, Any]] = {}
    for class_id in classes:
        predicted_mask = predicted == class_id
        truth_mask = truth == class_id
        predicted_count = int(predicted_mask.sum())
        truth_count = int(truth_mask.sum())
        intersection = int(np.logical_and(predicted_mask, truth_mask).sum())
        union = int(np.logical_or(predicted_mask, truth_mask).sum())
        both_empty = predicted_count == 0 and truth_count == 0
        one_empty = (predicted_count == 0) != (truth_count == 0)
        if both_empty:
            results[class_id] = {
                "class_id": class_id,
                "class_name": SEMANTIC_ORDER[class_id],
                "dice": None,
                "iou": None,
                "defined": False,
                "both_empty": True,
                "one_empty": False,
                "prediction_voxels": 0,
                "reference_voxels": 0,
                "reason": "class absent from both prediction and reference",
            }
            continue
        dice = 2.0 * intersection / float(predicted_count + truth_count)
        iou = intersection / float(union)
        results[class_id] = {
            "class_id": class_id,
            "class_name": SEMANTIC_ORDER[class_id],
            "dice": float(dice),
            "iou": float(iou),
            "defined": True,
            "both_empty": False,
            "one_empty": bool(one_empty),
            "prediction_voxels": predicted_count,
            "reference_voxels": truth_count,
            "reason": None,
        }
    return results


def patient_macro(
    per_patient: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    key: str = "dice",
    classes: Sequence[int] = FOREGROUND_CLASSES,
) -> dict[str, Any]:
    """Aggregate per-patient per-class overlap into a patient macro mean.

    Two levels, both explicit:

    1. Within a patient, the foreground aggregate averages that patient's
       *defined* classes. A patient with no defined foreground class contributes
       nothing and is counted in ``patients_undefined``.
    2. Across patients, the macro mean averages those per-patient values, so one
       patient is one unit regardless of how many slices or voxels it has.

    ``per_class`` reports the same mean class by class, each with its own
    denominator and its own both-empty count.
    """
    per_class: dict[int, dict[str, Any]] = {}
    for class_id in classes:
        values: list[float] = []
        both_empty = 0
        one_empty = 0
        for classes_map in per_patient.values():
            entry = classes_map.get(class_id)
            if entry is None:
                continue
            if not entry.get("defined", False):
                both_empty += 1
                continue
            one_empty += int(bool(entry.get("one_empty", False)))
            values.append(float(entry[key]))
        per_class[class_id] = {
            "class_id": class_id,
            "class_name": SEMANTIC_ORDER[class_id],
            "mean": float(np.mean(values)) if values else None,
            "count": len(values),
            "both_empty_excluded": both_empty,
            "one_empty_errors": one_empty,
            "available": bool(values),
            "reason": None if values else "no patient had this class defined",
        }

    patient_values: dict[str, float] = {}
    patients_undefined: list[str] = []
    for patient_id, classes_map in per_patient.items():
        defined = [
            float(entry[key])
            for class_id, entry in classes_map.items()
            if class_id in classes and entry.get("defined", False)
        ]
        if not defined:
            patients_undefined.append(patient_id)
            continue
        patient_values[patient_id] = float(np.mean(defined))

    values = list(patient_values.values())
    return {
        "key": key,
        "per_class": per_class,
        "per_patient_foreground": patient_values,
        "macro_mean": float(np.mean(values)) if values else None,
        "macro_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "patients_counted": len(values),
        "patients_undefined": patients_undefined,
        "available": bool(values),
        "reason": None if values else "no patient had any defined foreground class",
    }


# ----------------------------------------------------------------------------
# surface metrics
# ----------------------------------------------------------------------------
def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels: set members with at least one 6-neighbour outside the set."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = np.ones_like(mask, dtype=bool)
    for axis in range(mask.ndim):
        shifted_up = np.roll(mask, 1, axis=axis)
        shifted_down = np.roll(mask, -1, axis=axis)
        index_up = [slice(None)] * mask.ndim
        index_up[axis] = 0
        shifted_up[tuple(index_up)] = False
        index_down = [slice(None)] * mask.ndim
        index_down[axis] = -1
        shifted_down[tuple(index_down)] = False
        eroded &= shifted_up & shifted_down
    return mask & ~eroded


def surface_metrics(
    prediction: Any,
    reference: Any,
    *,
    class_id: int,
    spacing: Sequence[float] | None,
    geometry_valid: bool,
    percentile: float = 95.0,
) -> dict[str, Any]:
    """HD95 and ASSD for one class of one volume, or an explicit unavailability.

    ``geometry_valid`` must come from the data layer's native-geometry status;
    this function will not guess. Without valid geometry the metrics are
    unavailable with a reason - a pixel-space number is only produced when a
    caller passes unit spacing deliberately, and then the returned ``unit`` says
    ``pixel`` so no reader can mistake it for millimetres.
    """
    predicted = _as_label_volume(prediction, "prediction") == class_id
    truth = _as_label_volume(reference, "reference") == class_id
    if predicted.shape != truth.shape:
        raise MetricContractError("prediction and reference must share a grid")

    unit = "mm"
    if not geometry_valid or spacing is None:
        return {
            "class_id": class_id,
            "class_name": SEMANTIC_ORDER[class_id],
            "hd95": None,
            "assd": None,
            "available": False,
            "unit": "unavailable",
            "reason": "native geometry unavailable; surface distances are not defined",
            "undefined_case": True,
        }
    spacing_array = np.asarray(spacing, dtype=np.float64)
    if spacing_array.shape != (3,) or not np.all(np.isfinite(spacing_array)) or np.any(spacing_array <= 0):
        raise MetricContractError("spacing must be three finite positive values (z,y,x)")
    if np.allclose(spacing_array, 1.0):
        unit = "pixel"

    if not predicted.any() or not truth.any():
        return {
            "class_id": class_id,
            "class_name": SEMANTIC_ORDER[class_id],
            "hd95": None,
            "assd": None,
            "available": False,
            "unit": unit,
            "reason": (
                "surface distance undefined: class empty in "
                + ("prediction" if not predicted.any() else "reference")
            ),
            "undefined_case": True,
        }

    from scipy import ndimage  # imported lazily; only the isolated evaluator needs it

    predicted_surface = _surface_voxels(predicted)
    truth_surface = _surface_voxels(truth)
    distance_to_truth = ndimage.distance_transform_edt(~truth_surface, sampling=spacing_array)
    distance_to_predicted = ndimage.distance_transform_edt(~predicted_surface, sampling=spacing_array)
    forward = distance_to_truth[predicted_surface]
    backward = distance_to_predicted[truth_surface]
    both = np.concatenate([forward, backward])
    return {
        "class_id": class_id,
        "class_name": SEMANTIC_ORDER[class_id],
        "hd95": float(np.percentile(both, percentile)),
        "assd": float(both.mean()),
        "available": True,
        "unit": unit,
        "reason": None,
        "undefined_case": False,
        "percentile": percentile,
        "surface_voxels_prediction": int(predicted_surface.sum()),
        "surface_voxels_reference": int(truth_surface.sum()),
    }


# ----------------------------------------------------------------------------
# paired patient bootstrap
# ----------------------------------------------------------------------------
def paired_patient_bootstrap(
    audited: Mapping[str, float],
    unaudited: Mapping[str, float],
    *,
    iterations: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired bootstrap over patients of ``audited - unaudited``.

    The resampling unit is the patient id present in both mappings. Slices are
    never treated as independent draws, so the interval does not shrink with
    slice count. Returns an unavailable result rather than a degenerate interval
    when fewer than two patients are paired.
    """
    shared = sorted(set(audited) & set(unaudited))
    differences = np.array([float(audited[p]) - float(unaudited[p]) for p in shared], dtype=np.float64)
    result: dict[str, Any] = {
        "patients": shared,
        "count": len(shared),
        "mean_difference": float(differences.mean()) if differences.size else None,
        "iterations": iterations,
        "alpha": alpha,
        "seed": seed,
        "ci_low": None,
        "ci_high": None,
        "available": False,
        "reason": None,
        "unpaired_audited": sorted(set(audited) - set(unaudited)),
        "unpaired_unaudited": sorted(set(unaudited) - set(audited)),
    }
    if differences.size < 2:
        result["reason"] = "fewer than two paired patients; a bootstrap interval is not defined"
        return result
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, size=(iterations, differences.size))
    means = differences[indices].mean(axis=1)
    result["ci_low"] = float(np.percentile(means, 100.0 * alpha / 2.0))
    result["ci_high"] = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    result["available"] = True
    return result


# ----------------------------------------------------------------------------
# true edit attribution
# ----------------------------------------------------------------------------
def edit_attribution(
    initial: Any,
    final: Any,
    reference: Any,
    *,
    valid: Any | None = None,
) -> dict[str, Any]:
    """True initial-to-final FIX/REGRESS accounting against a reference volume.

    Categories over the evaluated pixels:

    ``fix``
        initial wrong, final correct.
    ``regress``
        initial correct, final wrong - this is destruction of a previously
        correct pixel and is reported with its own denominator.
    ``preserved_correct`` / ``preserved_wrong``
        unchanged correctness, whether or not the label changed.

    ``net_quality_change`` is ``fix - regress`` over the evaluated pixels, and
    ``harm_rate`` divides regressions by the number of initially correct pixels,
    so the two numbers cannot be confused. Only pixels with ``valid > 0`` are
    evaluated when a validity map is supplied; the ignored count is reported.

    This function needs a manual reference and therefore belongs exclusively to
    the isolated post-freeze evaluator.
    """
    initial_volume = _as_label_volume(initial, "initial")
    final_volume = _as_label_volume(final, "final")
    truth = _as_label_volume(reference, "reference")
    if not (initial_volume.shape == final_volume.shape == truth.shape):
        raise MetricContractError("initial, final and reference must share a grid")
    if valid is None:
        evaluated = np.ones_like(truth, dtype=bool)
        ignored = 0
    else:
        validity = np.asarray(valid)
        if validity.shape != truth.shape:
            raise MetricContractError("validity must share the reference grid")
        evaluated = validity > 0
        ignored = int((~evaluated).sum())

    initial_correct = (initial_volume == truth) & evaluated
    final_correct = (final_volume == truth) & evaluated
    fix = int((~initial_correct & final_correct & evaluated).sum())
    regress = int((initial_correct & ~final_correct).sum())
    preserved_correct = int((initial_correct & final_correct).sum())
    preserved_wrong = int((~initial_correct & ~final_correct & evaluated).sum())
    changed = int(((initial_volume != final_volume) & evaluated).sum())
    evaluated_count = int(evaluated.sum())
    initially_correct = int(initial_correct.sum())
    return {
        "fix": fix,
        "regress": regress,
        "preserved_correct": preserved_correct,
        "preserved_wrong": preserved_wrong,
        "changed_pixels": changed,
        "evaluated_pixels": evaluated_count,
        "ignored_pixels": ignored,
        "initially_correct_pixels": initially_correct,
        "net_quality_change": fix - regress,
        "net_quality_rate": (fix - regress) / evaluated_count if evaluated_count else None,
        "harm_rate": regress / initially_correct if initially_correct else None,
        "harm_rate_reason": None if initially_correct else "no initially correct evaluated pixel",
        "initial_accuracy": initially_correct / evaluated_count if evaluated_count else None,
        "final_accuracy": int(final_correct.sum()) / evaluated_count if evaluated_count else None,
        "available": evaluated_count > 0,
        "reason": None if evaluated_count else "no evaluated pixel after validity masking",
    }


# ----------------------------------------------------------------------------
# association between image-only evidence and true quality
# ----------------------------------------------------------------------------
def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2:
        return None
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = math.sqrt(float((a_centered ** 2).sum()) * float((b_centered ** 2).sum()))
    if denominator <= 0.0:
        return None
    return float((a_centered * b_centered).sum() / denominator)


def _auroc(scores: np.ndarray, positives: np.ndarray) -> float | None:
    """Rank-based AUROC with correct handling of tied scores (0.5 credit)."""
    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = _average_ranks(scores)
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


def _average_precision(scores: np.ndarray, positives: np.ndarray) -> float | None:
    """Average precision with tied scores resolved as one group (no tie-order luck)."""
    positive_count = int(positives.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_positives = positives[order]
    total_seen = 0
    true_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    index = 0
    while index < sorted_scores.size:
        stop = index + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[index]:
            stop += 1
        total_seen += stop - index
        true_positives += int(sorted_positives[index:stop].sum())
        precision = true_positives / total_seen
        recall = true_positives / positive_count
        average_precision += precision * (recall - previous_recall)
        previous_recall = recall
        index = stop
    return float(average_precision)


def evidence_quality_association(
    evidence_gain: Sequence[float],
    quality_gain: Sequence[float],
    *,
    tie_policy: str = "neutral_excluded",
    min_support: int = MIN_RANKING_SUPPORT,
) -> dict[str, Any]:
    """Relate image-only evidence gain to true reference quality gain.

    ``evidence_gain`` is a score difference in nats per pixel; ``quality_gain``
    is a reference-based difference. Correlations are reported without a
    p-value: a single-seed pilot does not support an inferential claim.

    Ranking metrics treat "quality actually improved" as the positive class.
    Exact ties in quality gain are neutral and, under the default policy, are
    excluded from the positive/negative support and counted. The metrics become
    unavailable - never optimistic - when either class has too little support.
    """
    if tie_policy not in ("neutral_excluded", "neutral_negative"):
        raise MetricContractError("tie_policy must be 'neutral_excluded' or 'neutral_negative'")
    evidence = np.asarray(evidence_gain, dtype=np.float64)
    quality = np.asarray(quality_gain, dtype=np.float64)
    if evidence.shape != quality.shape or evidence.ndim != 1:
        raise MetricContractError("evidence_gain and quality_gain must be matching 1-D sequences")

    ties = int(np.sum(quality == 0.0))
    if tie_policy == "neutral_excluded":
        keep = quality != 0.0
    else:
        keep = np.ones_like(quality, dtype=bool)
    ranking_scores = evidence[keep]
    ranking_positives = quality[keep] > 0.0

    pearson = _pearson(evidence, quality)
    spearman = (
        _pearson(_average_ranks(evidence), _average_ranks(quality)) if evidence.size >= 2 else None
    )
    positives = int(ranking_positives.sum())
    negatives = int(ranking_positives.size - positives)
    enough = positives >= min_support and negatives >= min_support
    reason = None
    if not enough:
        reason = (
            f"insufficient ranking support: {positives} positive and {negatives} negative "
            f"pairs, minimum {min_support} each"
        )
    return {
        "pairs": int(evidence.size),
        "ties": ties,
        "tie_policy": tie_policy,
        "positives": positives,
        "negatives": negatives,
        "pearson": pearson,
        "spearman": spearman,
        "auroc": _auroc(ranking_scores, ranking_positives) if enough else None,
        "auprc": _average_precision(ranking_scores, ranking_positives) if enough else None,
        "ranking_available": bool(enough),
        "reason": reason,
        "note": "correlation of an image-only score with reference quality; not a calibration claim",
    }


# ----------------------------------------------------------------------------
# coverage versus error
# ----------------------------------------------------------------------------
def coverage_order(validity: np.ndarray) -> np.ndarray:
    """Stable descending order of pixel indices by validity, ties by flat index.

    Deterministic and reference-free: the same validity map always yields the
    same retained set, so a coverage curve cannot be tuned by re-running.
    """
    flat = np.asarray(validity, dtype=np.float64).reshape(-1)
    return np.lexsort((np.arange(flat.size), -flat))


def coverage_error_curve(
    prediction: Any,
    reference: Any,
    validity: Any,
    *,
    levels: Sequence[float] = COVERAGE_LEVELS,
    evaluated: Any | None = None,
) -> list[dict[str, Any]]:
    """Error rate at preregistered coverage levels, ordered by image-only validity.

    The retained pixel set at each level is chosen by validity alone; the
    reference is only read to count errors inside the retained set. This keeps
    the coverage decision free of ground truth while still exposing the
    accuracy-for-coverage trade that a selective method buys.

    Every row carries its own coverage so that no accuracy number can be quoted
    without the coverage that produced it.
    """
    predicted = _as_label_volume(prediction, "prediction").reshape(-1)
    truth = _as_label_volume(reference, "reference").reshape(-1)
    validity_flat = np.asarray(validity, dtype=np.float64).reshape(-1)
    if not (predicted.size == truth.size == validity_flat.size):
        raise MetricContractError("prediction, reference and validity must share a grid")
    if evaluated is None:
        mask = np.ones_like(predicted, dtype=bool)
    else:
        mask = np.asarray(evaluated).reshape(-1).astype(bool)
        if mask.size != predicted.size:
            raise MetricContractError("evaluated mask must share the grid")
    candidate_indices = np.nonzero(mask)[0]
    total = candidate_indices.size
    ordering = coverage_order(validity_flat[candidate_indices])
    ordered_indices = candidate_indices[ordering]

    rows: list[dict[str, Any]] = []
    for level in levels:
        if not (0.0 < float(level) <= 1.0):
            raise MetricContractError("coverage levels must lie in (0,1]")
        retained = int(round(float(level) * total))
        if total > 0:
            retained = max(1, min(total, retained))
        selected = ordered_indices[:retained]
        if selected.size == 0:
            rows.append(
                {
                    "coverage_level": float(level),
                    "retained_pixels": 0,
                    "total_pixels": total,
                    "realized_coverage": None,
                    "error_rate": None,
                    "accuracy": None,
                    "available": False,
                    "reason": "no evaluable pixel at this coverage level",
                }
            )
            continue
        errors = int((predicted[selected] != truth[selected]).sum())
        rows.append(
            {
                "coverage_level": float(level),
                "retained_pixels": int(selected.size),
                "total_pixels": total,
                "realized_coverage": float(selected.size) / float(total),
                "error_rate": errors / float(selected.size),
                "accuracy": 1.0 - errors / float(selected.size),
                "available": True,
                "reason": None,
            }
        )
    return rows


# ----------------------------------------------------------------------------
# report writers
# ----------------------------------------------------------------------------
def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r} into a metric report")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomic JSON write: temp file then replace, so a crash leaves no half report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


CSV_COLUMNS = (
    "name",
    "value",
    "unit",
    "dataset",
    "split",
    "protocol",
    "epoch",
    "checkpoint",
    "population",
    "count",
    "available",
    "reason",
    "contract_version",
)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write metric rows as CSV with the frozen provenance columns plus extras."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    extra_keys = sorted({key for row in materialized for key in row.get("extra", {})})
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*CSV_COLUMNS, *extra_keys])
        for row in materialized:
            extra = row.get("extra", {})
            writer.writerow(
                [row.get(column) for column in CSV_COLUMNS]
                + [extra.get(key) for key in extra_keys]
            )
    temporary.replace(destination)
    return destination


def write_markdown(path: str | Path, title: str, rows: Iterable[Mapping[str, Any]],
                   *, notes: Sequence[str] = ()) -> Path:
    """Render metric rows as a Markdown table, unavailable rows included verbatim."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for note in notes:
        lines.append(f"> {note}")
    if notes:
        lines.append("")
    header = ["name", "value", "unit", "dataset", "split", "protocol", "epoch",
              "checkpoint", "population", "count", "available", "reason"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        cells = []
        for column in header:
            value = row.get(column)
            if value is None:
                cells.append("unavailable" if column == "value" else "")
            elif isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(destination)
    return destination
