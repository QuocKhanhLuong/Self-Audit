"""Frozen-prediction metrics for the adjacent-slice context diagnostic.

This module only scores supplied hard predictions. It neither chooses context
strengths nor changes model state. Reference masks must be opened only after all
teacher/student predictions at the prespecified strengths are frozen.

``analyze_frame`` scores a full ``(Z, H, W)`` frame volume. ``aggregate_records``
uses the historical frame -> patient/class -> classwise patient mean reduction
for Dice. Transition rates pool counts within each patient, then give each
defined patient equal weight; pooled cohort rates are a separate output.

``edge`` is the union of axial slices 0 and Z-1. ``interior`` is its complement.
All denominators exclude ignore=255 and pixels outside the optional valid mask.
Absent-reference Dice is None, including when a prediction has false positives.
The common-failure four cells are deliberately separate from the broader
teacher-rescue/student-wrong opportunity, which can include student regressions.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping

import numpy as np


CLASS_NAMES = ("BG", "RV", "MYO", "LV")
FOREGROUND_CLASS_NAMES = CLASS_NAMES[1:]
LAMBDAS = ("0", "0.5", "1")
TARGET_LAMBDAS = LAMBDAS[1:]
STRATA = ("all", "interior", "edge")
MODELS = ("teacher", "student")
CELLS = ("teacher_only_fix", "student_only_fix", "both_fix", "neither_fix")
SCHEMA = "self_audit.context_metrics.v1"


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _mean(values: Iterable[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def _prediction_mapping(
    predictions: Mapping[float | str, np.ndarray],
    *,
    shape: tuple[int, ...],
    valid: np.ndarray,
    model: str,
) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for key, prediction in predictions.items():
        try:
            strength = float(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{model}: invalid context lambda {key!r}") from exc
        if strength not in (0.0, 0.5, 1.0):
            raise ValueError(f"{model}: lambda keys must uniquely equal 0, 0.5, 1")
        canonical = {0.0: "0", 0.5: "0.5", 1.0: "1"}[strength]
        if canonical not in LAMBDAS or canonical in normalized:
            raise ValueError(f"{model}: lambda keys must uniquely equal 0, 0.5, 1")
        array = np.asarray(prediction)
        if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{model} lambda={canonical}: expected integer prediction of shape {shape}")
        if np.any((array[valid] < 0) | (array[valid] >= len(CLASS_NAMES))):
            raise ValueError(f"{model} lambda={canonical}: invalid label on a valid reference voxel")
        normalized[canonical] = array
    if set(normalized) != set(LAMBDAS):
        raise ValueError(f"{model}: all frozen predictions at 0, 0.5, 1 are required")
    return normalized


def _confusion(reference: np.ndarray, prediction: np.ndarray) -> list[list[int]]:
    encoded = reference.astype(np.int64) * 4 + prediction.astype(np.int64)
    return np.bincount(encoded, minlength=16).reshape(4, 4).tolist()


def _class_statistics(confusion: list[list[int]]) -> dict[str, dict[str, Any]]:
    matrix = np.asarray(confusion, dtype=np.int64)
    valid_count = int(matrix.sum())
    classes = {}
    for class_id, name in enumerate(CLASS_NAMES):
        tp = int(matrix[class_id, class_id])
        reference_count = int(matrix[class_id].sum())
        prediction_count = int(matrix[:, class_id].sum())
        fp, fn = prediction_count - tp, reference_count - tp
        classes[name] = {
            "class_id": class_id,
            "reference_present": reference_count > 0,
            "reference_count": reference_count,
            "prediction_count": prediction_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "dice": _ratio(2 * tp, reference_count + prediction_count) if reference_count else None,
            "absent_fp_rate": _ratio(fp, valid_count) if not reference_count else None,
        }
    return classes


def _prediction_statistics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    confusion = _confusion(reference, prediction)
    return {"confusion": confusion, "classes": _class_statistics(confusion)}


def _overall_rates(counts: Mapping[str, int]) -> dict[str, float | None]:
    return {
        "rescue_per_baseline_wrong": _ratio(counts["rescued"], counts["baseline_wrong"]),
        "harm_per_baseline_correct": _ratio(counts["harmed"], counts["baseline_correct"]),
        "rescue_per_valid": _ratio(counts["rescued"], counts["valid_voxels"]),
        "harm_per_valid": _ratio(counts["harmed"], counts["valid_voxels"]),
        "net_correct_change_per_valid": _ratio(counts["net_correct_change"], counts["valid_voxels"]),
    }


def _overall_transition(reference: np.ndarray, baseline: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    correct_before, correct_after = baseline == reference, target == reference
    rescued = ~correct_before & correct_after
    harmed = correct_before & ~correct_after
    counts = {
        "valid_voxels": int(reference.size),
        "baseline_correct": int(correct_before.sum()),
        "baseline_wrong": int((~correct_before).sum()),
        "target_correct": int(correct_after.sum()),
        "rescued": int(rescued.sum()),
        "harmed": int(harmed.sum()),
        "unchanged_correct": int((correct_before & correct_after).sum()),
        "unchanged_wrong": int((~correct_before & ~correct_after).sum()),
        "net_correct_change": int(correct_after.sum()) - int(correct_before.sum()),
    }
    return {
        "counts": counts,
        "rates": _overall_rates(counts),
        # Rows are true classes, columns are the new wrong prediction.
        "regression_confusion": _confusion(reference[harmed], target[harmed]),
        # Rows are true classes, columns are the previous wrong prediction.
        "rescue_origin_confusion": _confusion(reference[rescued], baseline[rescued]),
        # Rows/columns are baseline/target predictions, over all valid voxels.
        "prediction_transition_confusion": _confusion(baseline, target),
    }


def _class_rates(counts: Mapping[str, int]) -> dict[str, float | None]:
    rates = {"common_failure_per_gt": _ratio(counts["common_failure"], counts["gt_count"])}
    for cell in CELLS:
        rates[f"{cell}_per_common_failure"] = _ratio(counts[cell], counts["common_failure"])
        rates[f"{cell}_per_gt"] = _ratio(counts[cell], counts["gt_count"])
    for model in MODELS:
        rates[f"{model}_rescue_per_baseline_wrong"] = _ratio(
            counts[f"{model}_rescue"], counts[f"{model}_baseline_wrong"]
        )
        rates[f"{model}_harm_per_baseline_correct"] = _ratio(
            counts[f"{model}_harm"], counts[f"{model}_baseline_correct"]
        )
        rates[f"{model}_rescue_per_gt"] = _ratio(counts[f"{model}_rescue"], counts["gt_count"])
        rates[f"{model}_harm_per_gt"] = _ratio(counts[f"{model}_harm"], counts["gt_count"])
        rates[f"{model}_fp_introduced_per_negative"] = _ratio(
            counts[f"{model}_fp_introduced"], counts["valid_voxels"] - counts["gt_count"]
        )
        rates[f"{model}_fp_removed_per_baseline_fp"] = _ratio(
            counts[f"{model}_fp_removed"], counts[f"{model}_baseline_fp"]
        )
    for supplemental in ("supplemental_teacher_rescue_student_wrong", "supplemental_student_regression"):
        rates[f"{supplemental}_per_teacher_baseline_wrong"] = _ratio(
            counts[supplemental], counts["teacher_baseline_wrong"]
        )
        rates[f"{supplemental}_per_gt"] = _ratio(counts[supplemental], counts["gt_count"])
    return rates


def _check_class_counts(counts: Mapping[str, int]) -> None:
    if sum(counts[cell] for cell in CELLS) != counts["common_failure"]:
        raise ValueError("Common-failure four-cell partition does not equal its denominator")
    if counts["supplemental_common_failure_teacher_only"] != counts["teacher_only_fix"]:
        raise ValueError("Supplemental common-failure opportunity does not match teacher-only fixes")
    if counts["supplemental_teacher_rescue_student_wrong"] != (
        counts["supplemental_common_failure_teacher_only"] + counts["supplemental_student_regression"]
    ):
        raise ValueError("Supplemental opportunity does not partition into baseline student states")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
        raise ValueError("Class transition counts must be nonnegative integers")
    if not counts["common_failure"] <= counts["gt_count"] <= counts["valid_voxels"]:
        raise ValueError("Invalid common-failure/GT/valid denominator order")
    for model in MODELS:
        if counts[f"{model}_baseline_wrong"] + counts[f"{model}_baseline_correct"] != counts["gt_count"]:
            raise ValueError("Baseline class states do not partition GT class voxels")
        if counts[f"{model}_rescue"] > counts[f"{model}_baseline_wrong"]:
            raise ValueError("Rescues exceed baseline wrong class voxels")
        if counts[f"{model}_harm"] > counts[f"{model}_baseline_correct"]:
            raise ValueError("Harm exceeds baseline correct class voxels")
        if counts["common_failure"] > counts[f"{model}_baseline_wrong"]:
            raise ValueError("Common failures exceed a model's baseline wrong class voxels")
        if counts[f"{model}_only_fix"] + counts["both_fix"] > counts[f"{model}_rescue"]:
            raise ValueError("Common-failure fixes exceed model rescues")
    if counts["supplemental_teacher_rescue_student_wrong"] > counts["teacher_rescue"]:
        raise ValueError("Supplemental opportunities exceed teacher rescues")
    if counts["supplemental_student_regression"] > counts["student_harm"]:
        raise ValueError("Supplemental regressions exceed student harm")


def _class_transition(
    reference: np.ndarray,
    teacher0: np.ndarray,
    student0: np.ndarray,
    teacher: np.ndarray,
    student: np.ndarray,
    class_id: int,
) -> dict[str, Any]:
    gt = reference == class_id
    teacher0_right, student0_right = teacher0 == class_id, student0 == class_id
    teacher_right, student_right = teacher == class_id, student == class_id
    common = gt & ~teacher0_right & ~student0_right
    supplemental = gt & ~teacher0_right & teacher_right & ~student_right
    counts = {
        "valid_voxels": int(reference.size),
        "gt_count": int(gt.sum()),
        "common_failure": int(common.sum()),
        "teacher_only_fix": int((common & teacher_right & ~student_right).sum()),
        "student_only_fix": int((common & ~teacher_right & student_right).sum()),
        "both_fix": int((common & teacher_right & student_right).sum()),
        "neither_fix": int((common & ~teacher_right & ~student_right).sum()),
        "supplemental_teacher_rescue_student_wrong": int(supplemental.sum()),
        "supplemental_common_failure_teacher_only": int((supplemental & ~student0_right).sum()),
        "supplemental_student_regression": int((supplemental & student0_right).sum()),
    }
    for model, before, after in (
        ("teacher", teacher0_right, teacher_right), ("student", student0_right, student_right)
    ):
        counts.update({
            f"{model}_baseline_wrong": int((gt & ~before).sum()),
            f"{model}_baseline_correct": int((gt & before).sum()),
            f"{model}_rescue": int((gt & ~before & after).sum()),
            f"{model}_harm": int((gt & before & ~after).sum()),
            f"{model}_baseline_fp": int((~gt & before).sum()),
            f"{model}_fp_introduced": int((~gt & ~before & after).sum()),
            f"{model}_fp_removed": int((~gt & before & ~after).sum()),
        })
    _check_class_counts(counts)
    return {
        "class_id": class_id,
        "reference_present": counts["gt_count"] > 0,
        "counts": counts,
        "rates": _class_rates(counts),
    }


def analyze_frame(
    patient_id: str,
    frame_id: str | int,
    ground_truth: np.ndarray,
    teacher_predictions: Mapping[float | str, np.ndarray],
    student_predictions: Mapping[float | str, np.ndarray],
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score frozen predictions and return a JSON-ready full-frame record.

    Predictions must be integer labels BG=0/RV=1/MYO=2/LV=3 at each valid
    reference voxel. Lambda mappings accept numeric keys or ``"0"/"0.5"/"1"``.
    Reference 255 and false mask pixels are excluded from every count. Only
    positional axial endpoints define edges; anatomy never selects the stratum.
    """
    if not isinstance(patient_id, str) or not patient_id.strip() or not str(frame_id).strip():
        raise ValueError("A nonempty patient_id and frame_id are required")
    reference = np.asarray(ground_truth)
    if reference.ndim != 3 or any(size == 0 for size in reference.shape):
        raise ValueError("ground_truth must be a nonempty (Z,H,W) frame volume")
    if not np.issubdtype(reference.dtype, np.integer):
        raise ValueError("ground_truth must contain integer class labels")
    if np.any(~np.isin(reference, (0, 1, 2, 3, 255))):
        raise ValueError("ground_truth contains labels other than 0,1,2,3,255")
    valid = reference != 255
    if valid_mask is not None:
        supplied_mask = np.asarray(valid_mask)
        if supplied_mask.shape != reference.shape or supplied_mask.dtype != np.bool_:
            raise ValueError("valid_mask must be boolean with the same (Z,H,W) shape")
        valid = valid & supplied_mask
    predictions = {
        model: _prediction_mapping(mapping, shape=reference.shape, valid=valid, model=model)
        for model, mapping in (("teacher", teacher_predictions), ("student", student_predictions))
    }
    edge_slices = np.zeros(reference.shape[0], dtype=bool)
    edge_slices[0] = edge_slices[-1] = True
    strata_masks = {"all": np.ones_like(edge_slices), "interior": ~edge_slices, "edge": edge_slices}
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "patient_id": patient_id,
        "frame_id": str(frame_id),
        "shape": [int(size) for size in reference.shape],
        "valid_voxels": int(valid.sum()),
        "strata": {},
    }
    for stratum in STRATA:
        selected = valid & strata_masks[stratum][:, None, None]
        gt = reference[selected]
        pred = {model: {lam: predictions[model][lam][selected] for lam in LAMBDAS} for model in MODELS}
        record["strata"][stratum] = {
            "slice_indices": np.flatnonzero(strata_masks[stratum]).tolist(),
            "valid_voxels": int(selected.sum()),
            "predictions": {
                model: {lam: _prediction_statistics(gt, pred[model][lam]) for lam in LAMBDAS}
                for model in MODELS
            },
            "comparisons": {
                lam: {
                    "all_pixels": {
                        model: _overall_transition(gt, pred[model]["0"], pred[model][lam]) for model in MODELS
                    },
                    "foreground": {
                        name: _class_transition(
                            gt, pred["teacher"]["0"], pred["student"]["0"],
                            pred["teacher"][lam], pred["student"][lam], class_id,
                        )
                        for class_id, name in enumerate(CLASS_NAMES) if class_id > 0
                    },
                }
                for lam in TARGET_LAMBDAS
            },
        }
    validate_frame_record(record)
    return record


def validate_frame_record(record: Mapping[str, Any]) -> None:
    """Fail on altered partitions, confusion totals, or inconsistent regressions."""
    if record.get("schema") != SCHEMA:
        raise ValueError("Unsupported context metric record schema")
    strata = record["strata"]
    shape = record["shape"]
    if len(shape) != 3 or any(not isinstance(size, int) or size <= 0 for size in shape):
        raise ValueError("Invalid frame shape")
    expected_edge = sorted({0, shape[0] - 1})
    expected_slices = {"all": list(range(shape[0])), "edge": expected_edge, "interior": list(range(1, shape[0] - 1))}
    if any(strata[name]["slice_indices"] != expected_slices[name] for name in STRATA):
        raise ValueError("Stratum slices do not match the declared axial endpoint partition")
    if record["valid_voxels"] != strata["all"]["valid_voxels"]:
        raise ValueError("Frame valid count differs from all-stratum valid count")
    if strata["all"]["valid_voxels"] != strata["edge"]["valid_voxels"] + strata["interior"]["valid_voxels"]:
        raise ValueError("Edge/interior valid counts do not partition the frame")
    for stratum in STRATA:
        row = strata[stratum]
        reference_counts = np.asarray(row["predictions"]["teacher"]["0"]["confusion"]).sum(axis=1)
        for model in MODELS:
            for lam in LAMBDAS:
                matrix = np.asarray(row["predictions"][model][lam]["confusion"])
                if matrix.shape != (4, 4) or not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0) or int(matrix.sum()) != row["valid_voxels"]:
                    raise ValueError("Prediction confusion does not match valid voxel count")
                if row["predictions"][model][lam]["classes"] != _class_statistics(matrix.tolist()):
                    raise ValueError("Class statistics disagree with prediction confusion")
                if not np.array_equal(matrix.sum(axis=1), reference_counts):
                    raise ValueError("Prediction confusions disagree on the fixed reference class counts")
        for lam in TARGET_LAMBDAS:
            comparison = row["comparisons"][lam]
            for model in MODELS:
                transition = comparison["all_pixels"][model]
                counts = transition["counts"]
                for key, value in counts.items():
                    if not isinstance(value, int) or isinstance(value, bool) or (key != "net_correct_change" and value < 0):
                        raise ValueError("Overall transition counts must be integer voxel counts")
                baseline_matrix = np.asarray(row["predictions"][model]["0"]["confusion"])
                target_matrix = np.asarray(row["predictions"][model][lam]["confusion"])
                if counts["valid_voxels"] != row["valid_voxels"]:
                    raise ValueError("Overall transition valid count is inconsistent")
                if counts["baseline_correct"] != int(np.trace(baseline_matrix)) or counts["target_correct"] != int(np.trace(target_matrix)):
                    raise ValueError("Overall correctness counts disagree with prediction confusion")
                if counts["rescued"] - counts["harmed"] != counts["net_correct_change"]:
                    raise ValueError("Rescue/harm counts disagree with net correctness change")
                if counts["target_correct"] - counts["baseline_correct"] != counts["net_correct_change"]:
                    raise ValueError("Baseline/target correctness disagrees with net change")
                if counts["baseline_correct"] + counts["baseline_wrong"] != row["valid_voxels"]:
                    raise ValueError("Overall baseline states do not partition valid voxels")
                if counts["harmed"] + counts["unchanged_correct"] != counts["baseline_correct"] or counts["rescued"] + counts["unchanged_wrong"] != counts["baseline_wrong"]:
                    raise ValueError("Overall transitions do not partition baseline correctness states")
                for matrix_key in ("regression_confusion", "rescue_origin_confusion", "prediction_transition_confusion"):
                    matrix = np.asarray(transition[matrix_key])
                    if matrix.shape != (4, 4) or not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0):
                        raise ValueError("Invalid transition confusion matrix")
                    if matrix_key != "prediction_transition_confusion" and np.trace(matrix):
                        raise ValueError("Regression/rescue confusion may not include unchanged correct classes")
                if sum(map(sum, transition["regression_confusion"])) != counts["harmed"]:
                    raise ValueError("Regression confusion does not sum to harm")
                if sum(map(sum, transition["rescue_origin_confusion"])) != counts["rescued"]:
                    raise ValueError("Rescue-origin confusion does not sum to rescues")
                transition_matrix = np.asarray(transition["prediction_transition_confusion"])
                if not np.array_equal(transition_matrix.sum(axis=1), baseline_matrix.sum(axis=0)) or not np.array_equal(transition_matrix.sum(axis=0), target_matrix.sum(axis=0)):
                    raise ValueError("Prediction transition margins disagree with baseline/target predictions")
                if transition["rates"] != _overall_rates(counts):
                    raise ValueError("Overall transition rates disagree with their denominators")
            for name in FOREGROUND_CLASS_NAMES:
                class_result = comparison["foreground"][name]
                counts = class_result["counts"]
                _check_class_counts(counts)
                if counts["valid_voxels"] != row["valid_voxels"] or class_result["rates"] != _class_rates(counts):
                    raise ValueError("Class transition denominators or rates are inconsistent")
                for model in MODELS:
                    baseline_class = row["predictions"][model]["0"]["classes"][name]
                    target_class = row["predictions"][model][lam]["classes"][name]
                    if counts["gt_count"] != baseline_class["reference_count"] or counts[f"{model}_baseline_correct"] != baseline_class["tp"] or counts[f"{model}_baseline_fp"] != baseline_class["fp"]:
                        raise ValueError("Class transition baseline disagrees with prediction statistics")
                    if baseline_class["tp"] + counts[f"{model}_rescue"] - counts[f"{model}_harm"] != target_class["tp"]:
                        raise ValueError("Class rescue/harm does not explain the TP change")
                    if baseline_class["fp"] + counts[f"{model}_fp_introduced"] - counts[f"{model}_fp_removed"] != target_class["fp"]:
                        raise ValueError("Class FP transitions do not explain the FP change")
    for model in MODELS:
        for lam in LAMBDAS:
            matrices = [np.asarray(strata[s]["predictions"][model][lam]["confusion"]) for s in STRATA]
            if not np.array_equal(matrices[0], matrices[1] + matrices[2]):
                raise ValueError("Edge/interior confusions do not partition all voxels")
    for lam in TARGET_LAMBDAS:
        for model in MODELS:
            transitions = [strata[s]["comparisons"][lam]["all_pixels"][model] for s in STRATA]
            if any(transitions[0]["counts"][key] != transitions[1]["counts"][key] + transitions[2]["counts"][key] for key in transitions[0]["counts"]):
                raise ValueError("Edge/interior overall transitions do not partition all voxels")
            for key in ("regression_confusion", "rescue_origin_confusion", "prediction_transition_confusion"):
                if not np.array_equal(np.asarray(transitions[0][key]), np.asarray(transitions[1][key]) + np.asarray(transitions[2][key])):
                    raise ValueError("Edge/interior transition confusions do not partition all voxels")
        for name in FOREGROUND_CLASS_NAMES:
            counts = [strata[s]["comparisons"][lam]["foreground"][name]["counts"] for s in STRATA]
            if any(counts[0][key] != counts[1][key] + counts[2][key] for key in counts[0]):
                raise ValueError("Edge/interior class transitions do not partition all voxels")


def _sum_counts(rows: list[Mapping[str, int]], template: Mapping[str, int]) -> dict[str, int]:
    return {key: sum(row[key] for row in rows) for key in template}


def _sum_matrices(matrices: Iterable[list[list[int]]]) -> list[list[int]]:
    total = np.zeros((4, 4), dtype=np.int64)
    for matrix in matrices:
        total += np.asarray(matrix, dtype=np.int64)
    return total.tolist()


def _empty_stratum() -> dict[str, Any]:
    empty = np.array([], dtype=np.int64)
    return {
        "valid_voxels": 0,
        "predictions": {model: {lam: _prediction_statistics(empty, empty) for lam in LAMBDAS} for model in MODELS},
        "comparisons": {
            lam: {
                "all_pixels": {model: _overall_transition(empty, empty, empty) for model in MODELS},
                "foreground": {
                    name: _class_transition(empty, empty, empty, empty, empty, class_id)
                    for class_id, name in enumerate(CLASS_NAMES) if class_id > 0
                },
            }
            for lam in TARGET_LAMBDAS
        },
    }


def _pool_stratum(rows: list[Mapping[str, Any]], *, average_frame_dice: bool) -> dict[str, Any]:
    result = _empty_stratum()
    result["frame_count"] = len(rows)
    result["valid_voxels"] = sum(row["valid_voxels"] for row in rows)
    for model in MODELS:
        for lam in LAMBDAS:
            predictions = [row["predictions"][model][lam] for row in rows]
            matrix = _sum_matrices(pred["confusion"] for pred in predictions)
            classes = _class_statistics(matrix)
            for name in CLASS_NAMES:
                frame_classes = [pred["classes"][name] for pred in predictions]
                scores = [item["dice"] for item in frame_classes if item["dice"] is not None]
                classes[name]["pooled_voxel_dice"] = classes[name]["dice"]
                if average_frame_dice:
                    classes[name]["dice"] = _mean(scores)
                classes[name]["defined_frames"] = len(scores)
                classes[name]["absent_frames"] = sum(not item["reference_present"] for item in frame_classes)
                absent_fp = sum(item["fp"] for item in frame_classes if not item["reference_present"])
                absent_valid = sum(
                    row["valid_voxels"] for row, item in zip(rows, frame_classes) if not item["reference_present"]
                )
                classes[name]["absent_frame_fp"] = absent_fp
                classes[name]["absent_frame_valid_voxels"] = absent_valid
                classes[name]["absent_fp_rate"] = _ratio(absent_fp, absent_valid)
            result["predictions"][model][lam] = {
                "confusion": matrix,
                "classes": classes,
                "all_class_mean": _mean(classes[name]["dice"] for name in CLASS_NAMES),
                "foreground_mean": _mean(classes[name]["dice"] for name in FOREGROUND_CLASS_NAMES),
            }
    for lam in TARGET_LAMBDAS:
        comparisons = [row["comparisons"][lam] for row in rows]
        target = result["comparisons"][lam]
        for model in MODELS:
            transitions = [comparison["all_pixels"][model] for comparison in comparisons]
            template = target["all_pixels"][model]
            counts = _sum_counts([transition["counts"] for transition in transitions], template["counts"])
            target["all_pixels"][model] = {
                "counts": counts,
                "rates": _overall_rates(counts),
                **{
                    key: _sum_matrices(transition[key] for transition in transitions)
                    for key in ("regression_confusion", "rescue_origin_confusion", "prediction_transition_confusion")
                },
            }
        for name in FOREGROUND_CLASS_NAMES:
            template = target["foreground"][name]
            counts = _sum_counts([comparison["foreground"][name]["counts"] for comparison in comparisons], template["counts"])
            _check_class_counts(counts)
            target["foreground"][name] = {
                "class_id": CLASS_NAMES.index(name),
                "reference_present": counts["gt_count"] > 0,
                "counts": counts,
                "rates": _class_rates(counts),
            }
    return result


def _macro_rates(rows: list[Mapping[str, Any]], template: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: {"mean": _mean(row[key] for row in rows), "defined_patients": sum(row[key] is not None for row in rows)}
        for key in template
    }


def _patient_macro_stratum(patient_rows: list[Mapping[str, Any]], template: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"patient_count": len(patient_rows), "predictions": {}, "comparisons": {}}
    for model in MODELS:
        result["predictions"][model] = {}
        for lam in LAMBDAS:
            predictions = [row["predictions"][model][lam] for row in patient_rows]
            classes = {
                name: {
                    "dice": _mean(pred["classes"][name]["dice"] for pred in predictions),
                    "defined_patients": sum(pred["classes"][name]["dice"] is not None for pred in predictions),
                    "absent_fp_rate": _mean(pred["classes"][name]["absent_fp_rate"] for pred in predictions),
                    "absent_fp_defined_patients": sum(pred["classes"][name]["absent_fp_rate"] is not None for pred in predictions),
                }
                for name in CLASS_NAMES
            }
            result["predictions"][model][lam] = {
                "classes": classes,
                "class_first_macro4": _mean(classes[name]["dice"] for name in CLASS_NAMES),
                "class_first_foreground_macro": _mean(classes[name]["dice"] for name in FOREGROUND_CLASS_NAMES),
                "defined_classes_macro4": sum(classes[name]["dice"] is not None for name in CLASS_NAMES),
                "defined_classes_foreground": sum(classes[name]["dice"] is not None for name in FOREGROUND_CLASS_NAMES),
                "patient_first_all_class_macro": _mean(pred["all_class_mean"] for pred in predictions),
                "patient_first_foreground_macro": _mean(pred["foreground_mean"] for pred in predictions),
            }
    for lam in TARGET_LAMBDAS:
        comparisons = [row["comparisons"][lam] for row in patient_rows]
        result["comparisons"][lam] = {
            "all_pixels": {
                model: {"rates": _macro_rates(
                    [comparison["all_pixels"][model]["rates"] for comparison in comparisons],
                    template["comparisons"][lam]["all_pixels"][model]["rates"],
                )}
                for model in MODELS
            },
            "foreground": {
                name: {
                    "class_id": CLASS_NAMES.index(name),
                    "present_patients": sum(comparison["foreground"][name]["reference_present"] for comparison in comparisons),
                    "common_failure_patients": sum(comparison["foreground"][name]["counts"]["common_failure"] > 0 for comparison in comparisons),
                    "rates": _macro_rates(
                        [comparison["foreground"][name]["rates"] for comparison in comparisons],
                        template["comparisons"][lam]["foreground"][name]["rates"],
                    ),
                }
                for name in FOREGROUND_CLASS_NAMES
            },
        }
    return result


def _summarize(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["patient_id"]].append(record)
    patients = {
        patient_id: {
            "frame_count": len(frames),
            "frame_ids": sorted(frame["frame_id"] for frame in frames),
            "strata": {stratum: _pool_stratum([frame["strata"][stratum] for frame in frames], average_frame_dice=True) for stratum in STRATA},
        }
        for patient_id, frames in sorted(grouped.items())
    }
    pooled = {
        stratum: _pool_stratum([record["strata"][stratum] for record in records], average_frame_dice=False)
        for stratum in STRATA
    }
    return {
        "patient_count": len(patients),
        "frame_count": len(records),
        "patients": patients,
        "pooled": pooled,
        "patient_macro": {
            stratum: _patient_macro_stratum([patient["strata"][stratum] for patient in patients.values()], pooled[stratum])
            for stratum in STRATA
        },
    }


def aggregate_records(records: Iterable[Mapping[str, Any]], *, leave_one_out: bool = True) -> dict[str, Any]:
    """Aggregate unique patient/frame records without mixing patient identities.

    ``patient_macro`` is the primary equal-patient output. ``pooled`` combines
    counts and therefore gives larger volumes more weight. Leave-one-patient-out
    summaries exclude every frame for the named patient and are sensitivity
    analyses, not independent validations or confidence intervals.
    """
    frames = sorted(records, key=lambda row: (row["patient_id"], row["frame_id"]))
    keys = set()
    for frame in frames:
        validate_frame_record(frame)
        key = (frame["patient_id"], frame["frame_id"])
        if key in keys:
            raise ValueError(f"Duplicate patient/frame record: {key}")
        keys.add(key)
    summary = _summarize(frames)
    summary.update({
        "schema": SCHEMA,
        "class_names": list(CLASS_NAMES),
        "lambdas": [0.0, 0.5, 1.0],
        "aggregation": {
            "dice": "full frame volume Dice -> equal defined-frame patient/class mean -> equal defined-patient class mean -> mean defined classes",
            "transition_patient_macro": "pool transition counts within each patient, form rates, then average defined patients equally",
            "pooled": "sum counts across all frame volumes; pooled voxel Dice/rates are separate from patient macros",
            "absent_class": "Dice omitted even with false positives; false positives and fixed-valid-voxel rates are explicit",
            "edge": "first/last axial slice union; Z=1 counted once; interior empty for Z<=2",
            "supplemental": "teacher0 wrong, teacher lambda correct, student lambda wrong; includes student0-correct regressions; never divided by common failure",
        },
        "integrity": {"unique_patient_frame_ids": True, "common_failure_partitions": True, "edge_interior_partitions": True},
    })
    summary["leave_one_patient_out"] = {}
    if leave_one_out:
        for omitted in summary["patients"]:
            reduced = _summarize([frame for frame in frames if frame["patient_id"] != omitted])
            reduced.pop("patients")
            summary["leave_one_patient_out"][omitted] = {"excluded_patient_id": omitted, **reduced}
    return summary


def frame_csv_rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flat per-patient/frame/stratum/lambda/foreground transition CSV rows."""
    rows = []
    for record in sorted(records, key=lambda row: (row["patient_id"], row["frame_id"])):
        validate_frame_record(record)
        for stratum in STRATA:
            item = record["strata"][stratum]
            for lam in TARGET_LAMBDAS:
                for name in FOREGROUND_CLASS_NAMES:
                    transition = item["comparisons"][lam]["foreground"][name]
                    row = {
                        "patient_id": record["patient_id"], "frame_id": record["frame_id"],
                        "stratum": stratum, "lambda": float(lam), "class_id": transition["class_id"],
                        "class_name": name, "reference_present": transition["reference_present"],
                        **transition["counts"], **transition["rates"],
                    }
                    for model in MODELS:
                        for label, strength in (("baseline", "0"), ("target", lam)):
                            metrics = item["predictions"][model][strength]["classes"][name]
                            for key in ("tp", "fp", "fn", "dice", "absent_fp_rate"):
                                row[f"{model}_{label}_{key}"] = metrics[key]
                    rows.append(row)
    return rows


def frame_metric_csv_rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flat Dice/TP/FP/FN rows for every model/lambda/class, including BG."""
    rows = []
    for record in sorted(records, key=lambda row: (row["patient_id"], row["frame_id"])):
        validate_frame_record(record)
        for stratum in STRATA:
            for model in MODELS:
                for lam in LAMBDAS:
                    for name in CLASS_NAMES:
                        rows.append({
                            "patient_id": record["patient_id"], "frame_id": record["frame_id"],
                            "stratum": stratum, "model": model, "lambda": float(lam), "class_name": name,
                            "valid_voxels": record["strata"][stratum]["valid_voxels"],
                            **record["strata"][stratum]["predictions"][model][lam]["classes"][name],
                        })
    return rows


def summary_csv_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Lossless long-form numeric CSV leaves for patient, pooled, macro and LOO summaries.

    ``metric_path`` uses slash-delimited JSON keys; matrix indices are zero-based.
    None remains None for a CSV writer to emit an empty cell, never a fabricated
    zero. No output here should be interpreted as a trained-model improvement.
    """
    rows: list[dict[str, Any]] = []

    def walk(value: Any, scope: str, patient_id: str, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                walk(child, scope, patient_id, f"{path}/{key}" if path else key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, scope, patient_id, f"{path}/{index}")
        elif value is None or isinstance(value, (int, float, bool)):
            rows.append({"scope": scope, "patient_id": patient_id, "metric_path": path, "value": value})

    for scope in ("patient_macro", "pooled"):
        walk(summary[scope], scope, "", "")
    for patient_id, patient in summary["patients"].items():
        walk(patient, "patient", patient_id, "")
    for patient_id, reduced in summary.get("leave_one_patient_out", {}).items():
        walk(reduced, "leave_one_patient_out", patient_id, "")
    return rows


class ContextMetricsAccumulator:
    """Convenience wrapper retaining only small, JSON-ready frame summaries."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._keys: set[tuple[str, str]] = set()

    def update_frame(
        self,
        patient_id: str,
        frame_id: str | int,
        ground_truth: np.ndarray,
        teacher_predictions: Mapping[float | str, np.ndarray],
        student_predictions: Mapping[float | str, np.ndarray],
        valid_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        key = (patient_id, str(frame_id))
        if key in self._keys:
            raise ValueError(f"Duplicate patient/frame record: {key}")
        record = analyze_frame(patient_id, frame_id, ground_truth, teacher_predictions, student_predictions, valid_mask)
        self._keys.add(key)
        self._records.append(record)
        return deepcopy(record)

    @property
    def records(self) -> list[dict[str, Any]]:
        return deepcopy(self._records)

    def finalize(self, *, leave_one_out: bool = True) -> dict[str, Any]:
        return aggregate_records(self._records, leave_one_out=leave_one_out)
