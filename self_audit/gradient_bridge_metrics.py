"""Scoring and paired inference for frozen gradient-bridge predictions.

This module accepts already-frozen D0--D5 logits and reference labels. It does
not load data, select an intervention, construct a gradient, or mutate a model.
Callers must complete the diagnostic protocol's scoring authorization before
opening reference masks and must pass the same scored patient cohort for every
arm.

Dice follows the historical evaluator reduction:

    full-frame class Dice (GT-positive frames only)
      -> equal defined-frame mean per patient and class
      -> equal defined-patient mean per class
      -> mean of defined classes (foreground or Macro4).

The bootstrap resamples patient rows with one shared local-RNG index matrix for
both primary contrasts. Leave-one-patient-out values are recomputed from the
remaining frozen patient rows; no model is retrained.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

import numpy as np
import torch

from self_audit.context_metrics import _class_statistics


SCHEMA = "self_audit.gradient_bridge_metrics.v1"
FRAME_SCHEMA = "self_audit.gradient_bridge_frame.v1"
PATIENT_SCHEMA = "self_audit.gradient_bridge_patient.v1"
GATE_SCHEMA = "self_audit.gradient_bridge_gate.v1"
ARMS = ("D0", "D1", "D2", "D3", "D4", "D5")
GATE_CONTRASTS = (("D2", "D0"), ("D2", "D1"))
CLASS_NAMES = ("BG", "RV", "MYO", "LV")
FOREGROUND_CLASSES = CLASS_NAMES[1:]
IGNORE_INDEX = 255
_EPS = 1e-5

# context_metrics.analyze_frame scores both model maps at three strengths. The
# assignments below are only a read-only adapter for its class/count code; its
# unrelated pairwise context comparisons are deliberately not returned.
class MetricsIntegrityError(ValueError):
    """Malformed or mismatched scoring inputs that invalidate a gate."""


class NonFiniteMetricsError(MetricsIntegrityError):
    """Nonfinite frozen logits or metric values that invalidate a gate."""


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _validate_ids(patient_id: Any, frame_id: Any | None = None) -> tuple[str, str | None]:
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise MetricsIntegrityError("patient_id must be a nonempty string")
    normalized_frame = None
    if frame_id is not None:
        normalized_frame = str(frame_id)
        if not normalized_frame.strip():
            raise MetricsIntegrityError("frame_id must be nonempty")
    return patient_id, normalized_frame


def _empty_arm_state() -> dict[str, Any]:
    return {
        "confusion": np.zeros((4, 4), dtype=np.int64),
        "ce_sum": 0.0,
        "ce_count": 0,
        "probability_sum": np.zeros(4, dtype=np.float64),
        "gt_sum": np.zeros(4, dtype=np.int64),
        "intersection_sum": np.zeros(4, dtype=np.float64),
    }


def _native_loss_from_sums(state: Mapping[str, Any]) -> dict[str, Any]:
    support_voxels = int(state["ce_count"])
    if support_voxels == 0:
        return {
            "ce": None,
            "dice_loss": None,
            "total": None,
            "support_voxels": 0,
            "present_foreground_classes": [],
            "absent_foreground_classes": list(FOREGROUND_CLASSES),
            "undefined_reason": "empty_valid_support",
        }
    ce = float(state["ce_sum"] / support_voxels)
    dice_losses = []
    present = []
    absent = []
    for class_id, class_name in enumerate(CLASS_NAMES[1:], start=1):
        gt_sum = int(state["gt_sum"][class_id])
        if gt_sum > 0:
            present.append(class_name)
            class_dice = (
                2.0 * float(state["intersection_sum"][class_id]) + _EPS
            ) / (
                float(state["probability_sum"][class_id]) + gt_sum + _EPS
            )
            dice_losses.append(1.0 - class_dice)
        else:
            absent.append(class_name)
    dice_loss = float(np.mean(dice_losses)) if dice_losses else 0.0
    total = ce + dice_loss
    if not all(math.isfinite(value) for value in (ce, dice_loss, total)):
        raise NonFiniteMetricsError("streamed native supervised scorer produced a nonfinite loss")
    return {
        "ce": ce,
        "dice_loss": dice_loss,
        "total": total,
        "support_voxels": support_voxels,
        "present_foreground_classes": present,
        "absent_foreground_classes": absent,
        "undefined_reason": None,
    }


class GradientBridgeFrameAccumulator:
    """Bounded-memory scorer for one frozen native frame across D0--D5.

    Call ``update_chunk`` once per contiguous Z chunk in increasing order. The
    accumulator stores only class confusion counts, CE sum/count, and per-class
    soft-Dice probability/target/intersection sums. It never retains logits,
    per-pixel probabilities, or one-hot tensors between chunks.
    """

    def __init__(
        self,
        patient_id: str,
        frame_id: str | int,
        frame_shape: Sequence[int],
        *,
        fallback_by_arm: Mapping[str, bool] | None = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.patient_id, normalized_frame = _validate_ids(patient_id, frame_id)
        self.frame_id = normalized_frame
        if ignore_index != IGNORE_INDEX:
            raise MetricsIntegrityError("the native label contract requires ignore_index=255")
        if len(frame_shape) != 3 or any(
            isinstance(size, bool) or not isinstance(size, (int, np.integer)) or int(size) <= 0
            for size in frame_shape
        ):
            raise MetricsIntegrityError("frame_shape must be positive integer (Z,H,W)")
        self.frame_shape = tuple(int(size) for size in frame_shape)
        if fallback_by_arm is None:
            self.fallbacks = {arm: False for arm in ARMS}
        else:
            if not isinstance(fallback_by_arm, Mapping) or set(fallback_by_arm) != set(ARMS):
                raise MetricsIntegrityError(f"fallback_by_arm must contain exactly {ARMS}")
            if any(not isinstance(value, (bool, np.bool_)) for value in fallback_by_arm.values()):
                raise MetricsIntegrityError("fallback flags must be booleans")
            self.fallbacks = {arm: bool(fallback_by_arm[arm]) for arm in ARMS}
        self._states = {arm: _empty_arm_state() for arm in ARMS}
        self._next_z = 0
        self._valid_voxels = 0
        self._finalized = False

    def update_chunk(
        self,
        z_start: int,
        ground_truth_chunk: np.ndarray,
        logits_by_arm: Mapping[str, np.ndarray],
        valid_mask_chunk: np.ndarray | None = None,
    ) -> None:
        """Accumulate one contiguous chunk without retaining its input arrays."""
        if self._finalized:
            raise MetricsIntegrityError("cannot add chunks after finalize")
        if isinstance(z_start, bool) or not isinstance(z_start, (int, np.integer)):
            raise MetricsIntegrityError("z_start must be an integer")
        if int(z_start) != self._next_z:
            raise MetricsIntegrityError(f"chunks must be contiguous and ordered; expected z_start={self._next_z}")
        truth = np.asarray(ground_truth_chunk)
        if truth.ndim != 3 or truth.shape[1:] != self.frame_shape[1:] or truth.shape[0] <= 0:
            raise MetricsIntegrityError("ground_truth_chunk must have shape (positive Zc,H,W)")
        stop_z = self._next_z + int(truth.shape[0])
        if stop_z > self.frame_shape[0]:
            raise MetricsIntegrityError("ground-truth chunk exceeds declared frame depth")
        if not np.issubdtype(truth.dtype, np.integer):
            raise MetricsIntegrityError("ground_truth_chunk must have an integer dtype")
        if np.any(~np.isin(truth, (0, 1, 2, 3, IGNORE_INDEX))):
            raise MetricsIntegrityError("ground_truth_chunk labels must be 0,1,2,3,255")
        support = truth != IGNORE_INDEX
        if valid_mask_chunk is not None:
            supplied_valid = np.asarray(valid_mask_chunk)
            if supplied_valid.shape != truth.shape or supplied_valid.dtype != np.bool_:
                raise MetricsIntegrityError("valid_mask_chunk must be boolean with the chunk shape")
            support &= supplied_valid
        expected_keys = set(ARMS)
        if not isinstance(logits_by_arm, Mapping) or set(logits_by_arm) != expected_keys:
            raise MetricsIntegrityError(f"logits_by_arm must contain exactly {ARMS}")
        expected_shape = (4, *truth.shape)
        chunk_logits: dict[str, np.ndarray] = {}
        for arm in ARMS:
            values = np.asarray(logits_by_arm[arm])
            if values.shape != expected_shape or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise MetricsIntegrityError(f"{arm} chunk logits must be real numeric values with shape {expected_shape}")
            if np.any(~np.isfinite(values)):
                raise NonFiniteMetricsError(f"{arm} frozen chunk logits contain NaN or infinity")
            chunk_logits[arm] = values

        # The reference copy is chunk-local. Ignore and geometric invalidity
        # both carry zero mass and are represented by UNKNOWN to the canonical
        # supervised loss.
        labels_np = np.where(support, truth, IGNORE_INDEX).astype(np.int64, copy=False)
        truth_valid = truth[support].astype(np.int64, copy=False)
        self._valid_voxels += int(support.sum())
        for arm in ARMS:
            values = chunk_logits[arm]
            predicted = np.argmax(values, axis=0)
            encoded = truth_valid * 4 + predicted[support].astype(np.int64, copy=False)
            self._states[arm]["confusion"] += np.bincount(encoded, minlength=16).reshape(4, 4)

            score_dtype = torch.float64 if values.dtype == np.float64 else torch.float32
            logits_tensor = torch.as_tensor(
                np.array(np.transpose(values, (1, 0, 2, 3)), copy=True, order="C"), dtype=score_dtype
            )  # Zc,C,H,W; bounded by this chunk only
            labels_tensor = torch.as_tensor(np.ascontiguousarray(labels_np), dtype=torch.long)
            support_tensor = labels_tensor != IGNORE_INDEX
            with torch.no_grad():
                ce_sum = torch.nn.functional.cross_entropy(
                    logits_tensor,
                    labels_tensor,
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                probabilities = logits_tensor.softmax(dim=1)
                state = self._states[arm]
                state["ce_sum"] += float(ce_sum.to(torch.float64).item())
                state["ce_count"] += int(support.sum())
                for class_id in range(4):
                    gt_class = labels_tensor == class_id
                    selected = probabilities[:, class_id] * support_tensor
                    probability_sum = selected.sum(dtype=torch.float64)
                    gt_sum = gt_class.sum()
                    intersection_sum = (selected * gt_class).sum(dtype=torch.float64)
                    state["probability_sum"][class_id] += float(probability_sum.item())
                    state["gt_sum"][class_id] += int(gt_sum.item())
                    state["intersection_sum"][class_id] += float(intersection_sum.item())
        self._next_z = stop_z

    def finalize(self) -> dict[str, Any]:
        """Return JSON-ready metrics after all declared Z slices were scored."""
        if self._finalized:
            raise MetricsIntegrityError("frame accumulator can be finalized only once")
        if self._next_z != self.frame_shape[0]:
            raise MetricsIntegrityError(
                f"incomplete frame coverage: scored {self._next_z}/{self.frame_shape[0]} Z slices"
            )
        arms: dict[str, Any] = {}
        for arm in ARMS:
            state = self._states[arm]
            confusion = state["confusion"].tolist()
            native_counts = _class_statistics(confusion)
            class_metrics = {}
            for class_name in CLASS_NAMES:
                counts = native_counts[class_name]
                tp = int(counts["tp"])
                class_metrics[class_name] = {
                    **counts,
                    "recall": _safe_ratio(tp, int(counts["reference_count"])),
                    "precision": _safe_ratio(tp, int(counts["prediction_count"])),
                }
            native = _native_loss_from_sums(state)
            foreground_dice = _mean_defined([class_metrics[name]["dice"] for name in FOREGROUND_CLASSES])
            macro4_dice = _mean_defined([class_metrics[name]["dice"] for name in CLASS_NAMES])
            arms[arm] = {
                "class_metrics": class_metrics,
                "class_dice": {name: class_metrics[name]["dice"] for name in CLASS_NAMES},
                "foreground_dice": foreground_dice,
                "macro4_dice": macro4_dice,
                "rv_recall": class_metrics["RV"]["recall"],
                "rv_precision": class_metrics["RV"]["precision"],
                "native_supervised": native,
                "fallback": self.fallbacks[arm],
            }
        self._finalized = True
        total_voxels = int(np.prod(self.frame_shape))
        return {
            "schema": FRAME_SCHEMA,
            "patient_id": self.patient_id,
            "frame_id": self.frame_id,
            "shape": list(self.frame_shape),
            "valid_voxels": self._valid_voxels,
            "ignored_or_invalid_voxels": total_voxels - self._valid_voxels,
            "arms": arms,
        }


def score_frame_logits(
    patient_id: str,
    frame_id: str | int,
    ground_truth: np.ndarray,
    logits_by_arm: Mapping[str, np.ndarray],
    valid_mask: np.ndarray | None = None,
    *,
    fallback_by_arm: Mapping[str, bool] | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, Any]:
    """Convenience wrapper scoring one frozen full-frame D0--D5 logit set.

    Production inference should use :class:`GradientBridgeFrameAccumulator`
    and submit one slice/chunk at a time. Each array here has ``(4,Z,H,W)``
    class-first layout; the wrapper streams it as one chunk for fixture use.
    """
    truth = np.asarray(ground_truth)
    accumulator = GradientBridgeFrameAccumulator(
        patient_id,
        frame_id,
        truth.shape,
        fallback_by_arm=fallback_by_arm,
        ignore_index=ignore_index,
    )
    accumulator.update_chunk(0, truth, logits_by_arm, valid_mask)
    return accumulator.finalize()


def _validate_frame_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != FRAME_SCHEMA:
        raise MetricsIntegrityError("unsupported gradient bridge frame record schema")
    _validate_ids(record.get("patient_id"), record.get("frame_id"))
    valid_voxels = record.get("valid_voxels")
    if isinstance(valid_voxels, bool) or not isinstance(valid_voxels, (int, np.integer)) or valid_voxels < 0:
        raise MetricsIntegrityError("frame valid_voxels must be a nonnegative integer")
    if not isinstance(record.get("arms"), Mapping) or set(record["arms"]) != set(ARMS):
        raise MetricsIntegrityError("frame record must retain all six D0-D5 arms")
    reference_counts_by_arm = []
    for arm in ARMS:
        details = record["arms"][arm]
        if not isinstance(details, Mapping) or not isinstance(details.get("class_dice"), Mapping):
            raise MetricsIntegrityError(f"{arm} frame metrics are malformed")
        if set(details["class_dice"]) != set(CLASS_NAMES):
            raise MetricsIntegrityError(f"{arm} frame metrics must contain all four class Dice values")
        metrics = details.get("class_metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(CLASS_NAMES):
            raise MetricsIntegrityError(f"{arm} frame metrics must retain counts for all four classes")
        for class_name in CLASS_NAMES:
            counts = metrics[class_name]
            if not isinstance(counts, Mapping):
                raise MetricsIntegrityError(f"{arm}/{class_name} count metrics are malformed")
            required_counts = ("tp", "fp", "fn", "reference_count", "prediction_count", "reference_present")
            if not set(required_counts).issubset(counts):
                raise MetricsIntegrityError(f"{arm}/{class_name} count metrics are incomplete")
            for count_name in required_counts[:-1]:
                value = counts[count_name]
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                    raise MetricsIntegrityError(f"{arm}/{class_name}/{count_name} must be a nonnegative integer")
            if not isinstance(counts["reference_present"], (bool, np.bool_)):
                raise MetricsIntegrityError(f"{arm}/{class_name}/reference_present must be boolean")
            reference_count = int(counts["reference_count"])
            prediction_count = int(counts["prediction_count"])
            tp = int(counts["tp"])
            if tp + int(counts["fn"]) != reference_count or tp + int(counts["fp"]) != prediction_count:
                raise MetricsIntegrityError(f"{arm}/{class_name} counts do not reconcile")
            if bool(counts["reference_present"]) != (reference_count > 0):
                raise MetricsIntegrityError(f"{arm}/{class_name} reference presence disagrees with its count")
            if (details["class_dice"][class_name] is None) != (not bool(counts["reference_present"])):
                raise MetricsIntegrityError(f"{arm}/{class_name} Dice undefined status disagrees with GT presence")
            metric_dice = counts.get("dice")
            if metric_dice is not None and not math.isfinite(float(metric_dice)):
                raise NonFiniteMetricsError(f"{arm}/{class_name} context Dice is nonfinite")
            if metric_dice != details["class_dice"][class_name]:
                raise MetricsIntegrityError(f"{arm}/{class_name} class Dice fields disagree")
        reference_counts = sum(int(metrics[name]["reference_count"]) for name in CLASS_NAMES)
        prediction_counts = sum(int(metrics[name]["prediction_count"]) for name in CLASS_NAMES)
        if reference_counts != valid_voxels or prediction_counts != valid_voxels:
            raise MetricsIntegrityError(f"{arm} class counts do not sum to frame valid_voxels")
        reference_counts_by_arm.append(tuple(int(metrics[name]["reference_count"]) for name in CLASS_NAMES))
        for value in details["class_dice"].values():
            if value is not None and (isinstance(value, bool) or not np.isscalar(value)):
                raise MetricsIntegrityError("class Dice values must be finite scalars or null")
            if value is not None and not math.isfinite(float(value)):
                raise NonFiniteMetricsError("frame record contains nonfinite class Dice")
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise MetricsIntegrityError("class Dice values must lie in [0,1]")
    if any(counts != reference_counts_by_arm[0] for counts in reference_counts_by_arm[1:]):
        raise MetricsIntegrityError("reference class counts differ across arms for one frame")


def aggregate_patient_rows(frame_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reduce frame metrics to one historical-policy row for every patient.

    No frame or patient is dropped for fallback, absent classes, ignore-only
    support, or undefined native loss. Frame Dice is first averaged over
    defined frames within a patient/class, after which the bootstrap performs
    the patient/class and class-first reductions.
    """
    if not isinstance(frame_records, Sequence) or isinstance(frame_records, (str, bytes)) or not frame_records:
        raise MetricsIntegrityError("at least one frozen frame score is required")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    keys: set[tuple[str, str]] = set()
    for record in frame_records:
        _validate_frame_record(record)
        key = (record["patient_id"], str(record["frame_id"]))
        if key in keys:
            raise MetricsIntegrityError(f"duplicate patient/frame score: {key}")
        keys.add(key)
        grouped[record["patient_id"]].append(record)

    patient_rows: list[dict[str, Any]] = []
    for patient_id in sorted(grouped):
        frames = sorted(grouped[patient_id], key=lambda row: str(row["frame_id"]))
        arms: dict[str, Any] = {}
        for arm in ARMS:
            class_dice: dict[str, float | None] = {}
            class_coverage: dict[str, dict[str, int]] = {}
            class_counts: dict[str, dict[str, int]] = {}
            absent_class_fp: dict[str, dict[str, Any]] = {}
            for class_name in CLASS_NAMES:
                values = [frame["arms"][arm]["class_dice"][class_name] for frame in frames]
                defined = [float(value) for value in values if value is not None]
                class_dice[class_name] = float(np.mean(defined)) if defined else None
                class_coverage[class_name] = {
                    "defined_frames": len(defined),
                    "undefined_frames": len(values) - len(defined),
                }
                counts = [frame["arms"][arm]["class_metrics"][class_name] for frame in frames]
                class_counts[class_name] = {
                    key: sum(int(row[key]) for row in counts)
                    for key in ("tp", "fp", "fn", "reference_count", "prediction_count")
                }
                absent_frames = [
                    (frame, count)
                    for frame, count in zip(frames, counts)
                    if not bool(count["reference_present"])
                ]
                absent_valid = sum(int(frame["valid_voxels"]) for frame, _ in absent_frames)
                absent_fp = sum(int(count["fp"]) for _, count in absent_frames)
                absent_class_fp[class_name] = {
                    "absent_frames": len(absent_frames),
                    "absent_valid_voxels": absent_valid,
                    "false_positive_voxels": absent_fp,
                    "rate": _safe_ratio(absent_fp, absent_valid),
                }
            rv_counts = class_counts["RV"]
            fallback_frames = sum(bool(frame["arms"][arm].get("fallback", False)) for frame in frames)
            losses: dict[str, Any] = {}
            for loss_name in ("ce", "dice_loss", "total"):
                loss_values = [
                    frame["arms"][arm]["native_supervised"][loss_name]
                    for frame in frames
                    if frame["arms"][arm].get("native_supervised") is not None
                    and frame["arms"][arm]["native_supervised"].get(loss_name) is not None
                ]
                losses[loss_name] = float(np.mean(loss_values)) if loss_values else None
            arms[arm] = {
                "class_dice": class_dice,
                "class_coverage": class_coverage,
                "foreground_dice": _mean_defined([class_dice[name] for name in FOREGROUND_CLASSES]),
                "macro4_dice": _mean_defined([class_dice[name] for name in CLASS_NAMES]),
                "class_counts": class_counts,
                "absent_class_fp": absent_class_fp,
                "rv_counts": {key: rv_counts[key] for key in ("tp", "fp", "fn")},
                "rv_recall": _safe_ratio(rv_counts["tp"], rv_counts["reference_count"]),
                "rv_precision": _safe_ratio(rv_counts["tp"], rv_counts["prediction_count"]),
                "native_supervised_frame_mean": losses,
                "native_loss_defined_frames": sum(
                    frame["arms"][arm].get("native_supervised", {}).get("total") is not None
                    for frame in frames
                ),
                "fallback_frames": fallback_frames,
                "fallback_any": fallback_frames > 0,
            }
        # A shared reference volume implies the same supported/absent-class
        # pattern in every arm. A difference here is an integrity failure.
        for class_name in CLASS_NAMES:
            null_pattern = {arms[arm]["class_dice"][class_name] is None for arm in ARMS}
            if len(null_pattern) != 1:
                raise MetricsIntegrityError(f"class-support mismatch across arms for {patient_id}/{class_name}")
        patient_rows.append({
            "schema": PATIENT_SCHEMA,
            "patient_id": patient_id,
            "frame_count": len(frames),
            "arms": arms,
        })
    return patient_rows


def _normalize_patient_rows(patient_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(patient_rows, Sequence) or isinstance(patient_rows, (str, bytes)) or not patient_rows:
        raise MetricsIntegrityError("patient cohort must contain at least one row")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_support: dict[tuple[str, str], bool] = {}
    for row in patient_rows:
        if not isinstance(row, Mapping):
            raise MetricsIntegrityError("patient row must be a mapping")
        patient_id, _ = _validate_ids(row.get("patient_id"))
        if patient_id in seen:
            raise MetricsIntegrityError(f"duplicate patient row: {patient_id}")
        seen.add(patient_id)
        arms_value = row.get("arms")
        if not isinstance(arms_value, Mapping) or set(arms_value) != set(ARMS):
            raise MetricsIntegrityError(f"{patient_id}: patient row must retain all six arms")
        arms: dict[str, dict[str, float | None]] = {}
        for arm in ARMS:
            arm_row = arms_value[arm]
            if not isinstance(arm_row, Mapping) or not isinstance(arm_row.get("class_dice"), Mapping):
                raise MetricsIntegrityError(f"{patient_id}/{arm}: class_dice mapping required")
            scores = arm_row["class_dice"]
            if set(scores) != set(CLASS_NAMES):
                raise MetricsIntegrityError(f"{patient_id}/{arm}: all four class Dice keys are required")
            arms[arm] = {}
            for class_name in CLASS_NAMES:
                value = scores[class_name]
                is_undefined = value is None
                key = (patient_id, class_name)
                previous = expected_support.setdefault(key, is_undefined)
                if previous != is_undefined:
                    raise MetricsIntegrityError(f"{patient_id}/{class_name}: arm class support does not match")
                if value is None:
                    arms[arm][class_name] = None
                    continue
                if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                    raise MetricsIntegrityError(f"{patient_id}/{arm}/{class_name}: Dice must be scalar or null")
                number = float(value)
                if not math.isfinite(number):
                    raise NonFiniteMetricsError(f"{patient_id}/{arm}/{class_name}: Dice is nonfinite")
                if not 0.0 <= number <= 1.0:
                    raise MetricsIntegrityError(f"{patient_id}/{arm}/{class_name}: Dice must lie in [0,1]")
                arms[arm][class_name] = number
        normalized.append({"patient_id": patient_id, "arms": arms})
    return sorted(normalized, key=lambda row: row["patient_id"])


def _cohort_summary(patient_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        class_means = {
            class_name: _mean_defined([row["arms"][arm][class_name] for row in patient_rows])
            for class_name in CLASS_NAMES
        }
        result[arm] = {
            "class_dice": class_means,
            "foreground_dice": _mean_defined([class_means[name] for name in FOREGROUND_CLASSES]),
            "macro4_dice": _mean_defined([class_means[name] for name in CLASS_NAMES]),
            "defined_patients_by_class": {
                class_name: sum(row["arms"][arm][class_name] is not None for row in patient_rows)
                for class_name in CLASS_NAMES
            },
            "patient_count": len(patient_rows),
        }
    return result


def summarize_patient_rows(patient_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return class-first cohort metrics without dropping fallback patients."""
    normalized = _normalize_patient_rows(patient_rows)
    return _cohort_summary(normalized)


def _contrast_delta(summary: Mapping[str, Any], candidate: str, reference: str, metric: str) -> float | None:
    candidate_value = summary[candidate][metric]
    reference_value = summary[reference][metric]
    if candidate_value is None or reference_value is None:
        return None
    return float(candidate_value - reference_value)


def _paired_bootstrap(
    patient_rows: Sequence[Mapping[str, Any]],
    *,
    contrasts: Sequence[tuple[str, str]],
    n_resamples: int,
    seed: int,
    confidence_level: float,
    minimum_defined_fraction: float,
) -> dict[str, Any]:
    n_patients = len(patient_rows)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_patients, size=(n_resamples, n_patients), dtype=np.int64)
    sample_bytes = np.ascontiguousarray(indices).tobytes()
    pairing_sha256 = hashlib.sha256(sample_bytes).hexdigest()
    alpha = (1.0 - confidence_level) / 2.0
    point_summary = _cohort_summary(patient_rows)
    contrast_results: dict[str, Any] = {}

    raw: dict[str, dict[str, list[float | None]]] = {
        f"{candidate}-{reference}": {"foreground_dice": [], "macro4_dice": []}
        for candidate, reference in contrasts
    }
    for sample in indices:
        sampled_rows = [patient_rows[int(index)] for index in sample]
        sampled_summary = _cohort_summary(sampled_rows)
        for candidate, reference in contrasts:
            key = f"{candidate}-{reference}"
            for metric in ("foreground_dice", "macro4_dice"):
                raw[key][metric].append(_contrast_delta(sampled_summary, candidate, reference, metric))

    for candidate, reference in contrasts:
        key = f"{candidate}-{reference}"
        metrics: dict[str, Any] = {}
        for metric in ("foreground_dice", "macro4_dice"):
            values = raw[key][metric]
            defined = np.asarray([value for value in values if value is not None], dtype=np.float64)
            count_defined = int(defined.size)
            coverage = count_defined / n_resamples
            if count_defined:
                lower, upper = np.quantile(defined, [alpha, 1.0 - alpha], method="linear").tolist()
                ci = {"lower": float(lower), "upper": float(upper), "level": confidence_level, "method": "paired_percentile"}
            else:
                ci = {"lower": None, "upper": None, "level": confidence_level, "method": "paired_percentile"}
            metrics[metric] = {
                "point_delta": _contrast_delta(point_summary, candidate, reference, metric),
                "bootstrap_deltas": values,
                "bootstrap_coverage": {
                    "defined": count_defined,
                    "undefined": n_resamples - count_defined,
                    "fraction": coverage,
                    "minimum_fraction": minimum_defined_fraction,
                },
                "confidence_interval": ci,
            }

        loo_values: dict[str, dict[str, float | None]] = {"foreground_dice": {}, "macro4_dice": {}}
        patient_ids = [row["patient_id"] for row in patient_rows]
        for omitted_index, omitted_patient in enumerate(patient_ids):
            remaining = [row for index, row in enumerate(patient_rows) if index != omitted_index]
            loo_summary = _cohort_summary(remaining)
            for metric in ("foreground_dice", "macro4_dice"):
                loo_values[metric][omitted_patient] = _contrast_delta(loo_summary, candidate, reference, metric)

        loo_metrics: dict[str, Any] = {}
        for metric, values in loo_values.items():
            defined = [value for value in values.values() if value is not None]
            loo_metrics[metric] = {
                "deltas_by_omitted_patient": values,
                "coverage": {
                    "defined": len(defined),
                    "undefined": len(values) - len(defined),
                    "total": len(values),
                },
            }
        contrast_results[key] = {
            "candidate": candidate,
            "reference": reference,
            "metrics": metrics,
            "leave_one_patient_out": loo_metrics,
        }

    return {
        "n_patients": n_patients,
        "n_resamples": n_resamples,
        "seed": seed,
        "shared_resample_indices_sha256": pairing_sha256,
        "shared_resample_matrix": indices.tolist(),
        "contrasts": contrast_results,
    }


def paired_patient_bootstrap(
    patient_rows: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int,
    seed: int,
    confidence_level: float,
    minimum_defined_fraction: float = 1.0,
) -> dict[str, Any]:
    """Compute paired D2-D0/D2-D1 patient bootstrap and LOO deltas.

    ``patient_rows`` must already be one row per patient and contain all six
    arms. The one index matrix stored in the output is applied to every arm and
    both contrasts, retaining fallback patients and repeated bootstrap draws.
    """
    normalized = _normalize_patient_rows(patient_rows)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples <= 0:
        raise MetricsIntegrityError("n_resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise MetricsIntegrityError("seed must be a nonnegative integer")
    confidence_level = float(confidence_level)
    minimum_defined_fraction = float(minimum_defined_fraction)
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise MetricsIntegrityError("confidence_level must lie strictly between zero and one")
    if not math.isfinite(minimum_defined_fraction) or not 0.0 < minimum_defined_fraction <= 1.0:
        raise MetricsIntegrityError("minimum_defined_fraction must lie in (0,1]")
    return _paired_bootstrap(
        normalized,
        contrasts=GATE_CONTRASTS,
        n_resamples=int(n_resamples),
        seed=int(seed),
        confidence_level=confidence_level,
        minimum_defined_fraction=minimum_defined_fraction,
    )


def _validate_gate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or config.get("schema") != GATE_SCHEMA:
        raise MetricsIntegrityError(f"gate config schema must be {GATE_SCHEMA}")
    required = {"minimum_fg_absolute_delta", "bootstrap", "confidence_rule", "loo_rule"}
    if not required.issubset(config):
        raise MetricsIntegrityError(f"gate config is missing required keys: {sorted(required - set(config))}")
    minimum_delta = config["minimum_fg_absolute_delta"]
    if isinstance(minimum_delta, (bool, np.bool_)) or not np.isscalar(minimum_delta):
        raise MetricsIntegrityError("minimum_fg_absolute_delta must be a finite scalar")
    minimum_delta = float(minimum_delta)
    if not math.isfinite(minimum_delta) or not 0.0 <= minimum_delta <= 1.0:
        raise MetricsIntegrityError("minimum_fg_absolute_delta must lie in [0,1]")
    bootstrap = config["bootstrap"]
    confidence = config["confidence_rule"]
    loo = config["loo_rule"]
    if not isinstance(bootstrap, Mapping) or not {"n_resamples", "seed"}.issubset(bootstrap):
        raise MetricsIntegrityError("bootstrap config requires n_resamples and seed")
    if not isinstance(confidence, Mapping) or not {
        "method", "level", "minimum_defined_fraction", "acceptance"
    }.issubset(confidence):
        raise MetricsIntegrityError("confidence_rule requires method, level, minimum_defined_fraction, and acceptance")
    if confidence["method"] != "paired_percentile":
        raise MetricsIntegrityError("confidence_rule.method must be paired_percentile")
    if confidence["acceptance"] != "lower_bound_strictly_above_minimum":
        raise MetricsIntegrityError("unsupported confidence-rule acceptance criterion")
    if not isinstance(loo, Mapping) or not {"method", "require_all_defined"}.issubset(loo):
        raise MetricsIntegrityError("loo_rule requires method and require_all_defined")
    if loo["method"] != "all_deltas_at_least_minimum":
        raise MetricsIntegrityError("loo_rule.method must be all_deltas_at_least_minimum")
    if loo["require_all_defined"] is not True:
        raise MetricsIntegrityError("loo_rule.require_all_defined must be true for the locked rule")
    n_resamples = bootstrap["n_resamples"]
    seed = bootstrap["seed"]
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples <= 0:
        raise MetricsIntegrityError("bootstrap.n_resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise MetricsIntegrityError("bootstrap.seed must be a nonnegative integer")
    level = confidence["level"]
    minimum_defined_fraction = confidence["minimum_defined_fraction"]
    if isinstance(level, bool) or not np.isscalar(level):
        raise MetricsIntegrityError("confidence_rule.level must be a finite scalar")
    if isinstance(minimum_defined_fraction, bool) or not np.isscalar(minimum_defined_fraction):
        raise MetricsIntegrityError("confidence_rule.minimum_defined_fraction must be a finite scalar")
    level, minimum_defined_fraction = float(level), float(minimum_defined_fraction)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise MetricsIntegrityError("confidence_rule.level must lie strictly between zero and one")
    if not math.isfinite(minimum_defined_fraction) or not 0.0 < minimum_defined_fraction <= 1.0:
        raise MetricsIntegrityError("confidence_rule.minimum_defined_fraction must lie in (0,1]")
    return {
        "schema": GATE_SCHEMA,
        "minimum_fg_absolute_delta": minimum_delta,
        "bootstrap": {"n_resamples": int(n_resamples), "seed": int(seed)},
        "confidence_rule": {
            "method": "paired_percentile",
            "level": level,
            "minimum_defined_fraction": minimum_defined_fraction,
            "acceptance": "lower_bound_strictly_above_minimum",
        },
        "loo_rule": {"method": "all_deltas_at_least_minimum", "require_all_defined": True},
    }


def _blocked_result(
    reasons: Sequence[str], *, integrity_reasons: Sequence[str] = (), nonfinite_reasons: Sequence[str] = ()
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "BLOCKED",
        "decision": "BLOCKED",
        "blocking": {
            "integrity": list(integrity_reasons),
            "nonfinite": list(nonfinite_reasons),
            "all": list(reasons),
        },
        "contrasts": {},
    }


def _loo_rule_passes(contrast: Mapping[str, Any], minimum_delta: float) -> tuple[bool, bool, list[str]]:
    details = contrast["leave_one_patient_out"]["foreground_dice"]
    values = details["deltas_by_omitted_patient"]
    defined = [value for value in values.values() if value is not None]
    reasons: list[str] = []
    if len(defined) != len(values):
        reasons.append("LOO contains undefined deltas")
    if any(value < minimum_delta for value in defined):
        reasons.append("at least one LOO delta is below the minimum foreground delta")
    return len(defined) == len(values) and all(value >= minimum_delta for value in defined), any(
        value < minimum_delta for value in defined
    ), reasons


def evaluate_paired_gate(
    patient_rows: Sequence[Mapping[str, Any]],
    gate_config: Mapping[str, Any],
    *,
    integrity_errors: Sequence[str] = (),
    nonfinite_errors: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a locked D2-over-D0-and-D1 gate result with four distinct states.

    ``GO`` requires both point deltas to meet the locked practical minimum, both
    paired percentile lower bounds to be strictly above it, sufficient defined
    bootstrap coverage, and every leave-one-patient-out delta to meet it.
    ``NO_GO`` requires the locked LOO rule to fail on defined values or an upper
    confidence bound at/below the minimum. Remaining statistical uncertainty or
    undefined support is ``INCONCLUSIVE``. Integrity and nonfinite issues are
    ``BLOCKED`` and are kept in separate fields.
    """
    if integrity_errors or nonfinite_errors:
        reasons = [str(value) for value in integrity_errors] + [str(value) for value in nonfinite_errors]
        return _blocked_result(
            reasons,
            integrity_reasons=[str(value) for value in integrity_errors],
            nonfinite_reasons=[str(value) for value in nonfinite_errors],
        )
    try:
        config = _validate_gate_config(gate_config)
        rows = _normalize_patient_rows(patient_rows)
        bootstrap = _paired_bootstrap(
            rows,
            contrasts=GATE_CONTRASTS,
            n_resamples=config["bootstrap"]["n_resamples"],
            seed=config["bootstrap"]["seed"],
            confidence_level=config["confidence_rule"]["level"],
            minimum_defined_fraction=config["confidence_rule"]["minimum_defined_fraction"],
        )
    except NonFiniteMetricsError as exc:
        return _blocked_result([str(exc)], nonfinite_reasons=[str(exc)])
    except (MetricsIntegrityError, ValueError, TypeError, KeyError) as exc:
        return _blocked_result([str(exc)], integrity_reasons=[str(exc)])

    point_summary = _cohort_summary(rows)
    minimum_delta = config["minimum_fg_absolute_delta"]
    all_go = True
    any_no_go = False
    any_inconclusive = False
    contrast_decisions: dict[str, Any] = {}
    for contrast_key, contrast in bootstrap["contrasts"].items():
        fg = contrast["metrics"]["foreground_dice"]
        point_delta = fg["point_delta"]
        ci = fg["confidence_interval"]
        coverage = fg["bootstrap_coverage"]
        loo_pass, loo_observed_failure, loo_reasons = _loo_rule_passes(contrast, minimum_delta)
        coverage_pass = coverage["fraction"] >= config["confidence_rule"]["minimum_defined_fraction"]
        point_pass = point_delta is not None and point_delta >= minimum_delta
        ci_lower_pass = ci["lower"] is not None and ci["lower"] > minimum_delta
        upper_clearly_fails = ci["upper"] is not None and ci["upper"] <= minimum_delta
        reasons: list[str] = []
        if point_delta is None:
            reasons.append("foreground Dice is undefined for the full cohort")
        elif not point_pass:
            reasons.append("point delta does not meet the minimum foreground delta")
        if not coverage_pass:
            reasons.append("defined bootstrap coverage is below the locked minimum")
        if not ci_lower_pass:
            reasons.append("paired confidence lower bound does not clear the minimum foreground delta")
        reasons.extend(loo_reasons)

        if upper_clearly_fails or loo_observed_failure:
            decision = "NO_GO"
            any_no_go = True
            all_go = False
        elif point_pass and coverage_pass and ci_lower_pass and loo_pass:
            decision = "GO"
        else:
            decision = "INCONCLUSIVE"
            any_inconclusive = True
            all_go = False
        contrast_decisions[contrast_key] = {
            "status": decision,
            "candidate": "D2",
            "reference": contrast["reference"],
            "minimum_fg_absolute_delta": minimum_delta,
            "point_delta": point_delta,
            "confidence_interval": ci,
            "bootstrap_coverage": coverage,
            "leave_one_patient_out": contrast["leave_one_patient_out"]["foreground_dice"],
            "loo_rule_passed": loo_pass,
            "reasons": reasons,
        }

    status = "GO" if all_go else ("NO_GO" if any_no_go else "INCONCLUSIVE")
    return {
        "schema": SCHEMA,
        "status": status,
        "decision": status,
        "criterion": "D2 must beat both D0 and D1 on class-first native foreground Dice",
        "gate_config": config,
        "cohort": {
            "patient_count": len(rows),
            "patient_ids": [row["patient_id"] for row in rows],
            "same_cohort_all_arms": True,
            "fallback_patients_retained": sum(
                any(bool(arm_row.get("fallback_any", arm_row.get("fallback", False))) for arm_row in row["arms"].values())
                for row in patient_rows
            ),
        },
        "arm_summaries": point_summary,
        "bootstrap": {
            "n_patients": bootstrap["n_patients"],
            "n_resamples": bootstrap["n_resamples"],
            "seed": bootstrap["seed"],
            "shared_resample_indices_sha256": bootstrap["shared_resample_indices_sha256"],
            "shared_resample_matrix": bootstrap["shared_resample_matrix"],
            "contrasts": bootstrap["contrasts"],
        },
        "contrasts": contrast_decisions,
        "blocking": {"integrity": [], "nonfinite": [], "all": []},
        "notes": [
            "D3-D5 are explanatory; none is an additional gate criterion.",
            "Native CE and present-foreground Dice loss are scorer-only diagnostics.",
            "LOO excludes one scored patient and recomputes summaries without retraining.",
        ],
    }
