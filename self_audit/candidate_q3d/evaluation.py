"""Native-volume Q-3D evaluation and paired decision gates.

Evaluation consumes one full native frame at a time.  Development and test
labels are only opened after every requested prediction has been materialized
and hashed.  Physical distances use the declared anisotropic ZYX spacing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage, stats

from self_audit.candidate_a3d.geometry import (
    VolumeGeometry3D,
    require_full_volume_shape,
)
from self_audit.data import CLASS_MAPPING, digest, geometry_map, write_json

CLASS_NAMES = ("BG", "RV", "MYO", "LV")
FOREGROUND_IDS = (1, 2, 3)
FOREGROUND_NAMES = ("RV", "MYO", "LV")
BOUNDARY_TOLERANCE_MM = 1.0
HARM_MARGIN = 0.02
MIN_FG_DICE_DELTA = 0.01
DEAD_CODE_FREQUENCY = 0.001
HARM_REDUCTION_CI95_LOWER_BOUND = 0.0


def _runtime_geometry(value: Mapping[str, Any] | VolumeGeometry3D,
                      shape_zyx: Sequence[int]) -> dict[str, Any]:
    geometry = VolumeGeometry3D.from_mapping(value) if not isinstance(value, VolumeGeometry3D) else value
    require_full_volume_shape(geometry, shape_zyx)
    return geometry.as_dict()


def validate_native_image(image: Any, geometry: Mapping[str, Any] | VolumeGeometry3D) -> tuple[torch.Tensor, dict[str, Any]]:
    """Validate a single normalized full-volume frame and its physical geometry."""
    tensor = torch.as_tensor(image)
    if tensor.ndim == 3:
        tensor = tensor[None, None]
    if tensor.ndim != 5 or tuple(tensor.shape[:2]) != (1, 1):
        raise ValueError("Q-3D native image must have shape [D,H,W] or [1,1,D,H,W]")
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    else:
        tensor = tensor.to(dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError("Q-3D native image contains NaN or Inf")
    if torch.any(tensor < 0) or torch.any(tensor > 1):
        raise ValueError("Q-3D native image must be normalized to [0,1]")
    spatial = tuple(int(value) for value in tensor.shape[-3:])
    canonical = _runtime_geometry(geometry, spatial)
    return tensor.contiguous(), canonical


def _validate_label(label: Any, shape_zyx: Sequence[int]) -> np.ndarray:
    value = np.asarray(label)
    if value.shape != tuple(int(v) for v in shape_zyx):
        raise ValueError("Q-3D reference mask shape differs from the native image")
    if not np.isin(value, (0, 1, 2, 3)).all():
        raise ValueError("Q-3D reference mask must use the declared BG/RV/MYO/LV IDs")
    return value.astype(np.uint8, copy=False)


def _dice(reference: np.ndarray, prediction: np.ndarray, class_id: int) -> float | None:
    ref = reference == class_id
    pred = prediction == class_id
    denominator = int(ref.sum()) + int(pred.sum())
    if denominator == 0:
        return None
    return float(2.0 * np.count_nonzero(ref & pred) / denominator)


def frame_class_dice(reference: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, Any]]:
    """Return the shared per-frame class Dice values and reference support."""
    return {
        name: {
            "dice": _dice(reference, prediction, class_id),
            "reference_present": bool(np.any(reference == class_id)),
        }
        for class_id, name in zip(FOREGROUND_IDS, FOREGROUND_NAMES)
    }


def aggregate_patient_class_dice(frame_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average frame Dice within class, then average supported classes.

    A frame contributes to a class only when that class is present in the
    reference. This preserves Q-3D's empty-class policy for patient selection
    and reporting while preventing large frames from dominating a patient.
    """
    per_class: dict[str, dict[str, Any]] = {}
    for name in FOREGROUND_NAMES:
        values = [
            float(row[name]["dice"])
            for row in frame_rows
            if row[name]["reference_present"] and row[name]["dice"] is not None
        ]
        per_class[name] = {
            "dice": float(np.mean(values)) if values else None,
            "frames_with_reference": len(values),
        }
    supported = [entry["dice"] for entry in per_class.values() if entry["dice"] is not None]
    return {
        "dice_by_class": per_class,
        "fg_dice": float(np.mean(supported)) if supported else None,
    }


def surface_distances_3d(reference: np.ndarray, prediction: np.ndarray,
                         spacing_zyx: Sequence[float], *,
                         tolerance_mm: float = BOUNDARY_TOLERANCE_MM) -> dict[str, Any]:
    """Return voxel-surface HD95/ASSD in mm with explicit absent-class status."""
    ref = np.asarray(reference, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    spacing = tuple(float(v) for v in spacing_zyx)
    if ref.ndim != 3 or pred.shape != ref.shape:
        raise ValueError("surface metric inputs must be matching 3-D masks")
    if len(spacing) != 3 or any(not math.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError("surface metrics require positive anisotropic ZYX spacing in mm")
    if not math.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("surface tolerance must be positive millimetres")
    if not ref.any() and not pred.any():
        return {"status": "both_absent", "hd95_mm": None, "assd_mm": None,
                "surface_dice_1mm": None}
    if not ref.any() or not pred.any():
        return {"status": "one_absent", "hd95_mm": None, "assd_mm": None,
                "surface_dice_1mm": 0.0}
    connectivity = ndimage.generate_binary_structure(3, 1)
    ref_surface = ref ^ ndimage.binary_erosion(ref, structure=connectivity, border_value=0)
    pred_surface = pred ^ ndimage.binary_erosion(pred, structure=connectivity, border_value=0)
    if not ref_surface.any() or not pred_surface.any():
        return {"status": "no_surface", "hd95_mm": None, "assd_mm": None,
                "surface_dice_1mm": None}
    d_ref = ndimage.distance_transform_edt(~ref_surface, sampling=spacing)
    d_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    ref_to_pred = d_pred[ref_surface]
    pred_to_ref = d_ref[pred_surface]
    distances = np.concatenate((ref_to_pred, pred_to_ref)).astype(np.float64, copy=False)
    return {
        "status": "ok",
        "hd95_mm": float(np.percentile(distances, 95)),
        "assd_mm": float(distances.mean()),
        "surface_dice_1mm": float(
            (np.count_nonzero(ref_to_pred <= tolerance_mm)
             + np.count_nonzero(pred_to_ref <= tolerance_mm)) / len(distances)
        ),
    }


def _frame_metrics(reference: np.ndarray, prediction: np.ndarray,
                   spacing_zyx: Sequence[float]) -> dict[str, Any]:
    overlap: dict[str, Any] = {}
    boundary: dict[str, Any] = {}
    class_dice = frame_class_dice(reference, prediction)
    for class_id, name in enumerate(CLASS_NAMES):
        if class_id == 0:
            continue
        overlap[name] = {
            **class_dice[name],
            "reference_voxels": int(np.count_nonzero(reference == class_id)),
            "prediction_voxels": int(np.count_nonzero(prediction == class_id)),
        }
        boundary[name] = surface_distances_3d(
            reference == class_id, prediction == class_id, spacing_zyx
        )
    fg_values = [overlap[name]["dice"] for name in FOREGROUND_NAMES
                 if overlap[name]["dice"] is not None]
    hd95_values = [boundary[name]["hd95_mm"] for name in FOREGROUND_NAMES
                   if boundary[name]["hd95_mm"] is not None]
    assd_values = [boundary[name]["assd_mm"] for name in FOREGROUND_NAMES
                   if boundary[name]["assd_mm"] is not None]
    surface_status_counts = {
        status: sum(boundary[name]["status"] == status for name in FOREGROUND_NAMES)
        for status in ("ok", "both_absent", "one_absent", "no_surface")
    }
    return {
        "dice": overlap,
        "fg_dice": float(np.mean(fg_values)) if fg_values else None,
        "boundary": boundary,
        "hd95_mm": float(np.mean(hd95_values)) if hd95_values else None,
        "assd_mm": float(np.mean(assd_values)) if assd_values else None,
        "surface_status_counts": surface_status_counts,
        "one_absent_count": surface_status_counts["one_absent"],
    }


def _clinical_frame(reference: np.ndarray, prediction: np.ndarray,
                    voxel_mm3: float) -> dict[str, Any]:
    def volume(mask: np.ndarray, class_id: int) -> float:
        return float(np.count_nonzero(mask == class_id) * voxel_mm3 / 1000.0)
    return {
        "LV_volume_ml": {"reference": volume(reference, 3), "prediction": volume(prediction, 3)},
        "RV_volume_ml": {"reference": volume(reference, 1), "prediction": volume(prediction, 1)},
        "MYO_mass_g": {
            "reference": volume(reference, 2) * 1.05,
            "prediction": volume(prediction, 2) * 1.05,
        },
    }


def _patient_aggregate(frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_aggregate = aggregate_patient_class_dice(
        [row["metrics"]["dice"] for row in frame_rows]
    )
    per_class: dict[str, Any] = {}
    for name in FOREGROUND_NAMES:
        h95_values = [row["metrics"]["boundary"][name]["hd95_mm"] for row in frame_rows
                      if row["metrics"]["boundary"][name]["hd95_mm"] is not None]
        assd_values = [row["metrics"]["boundary"][name]["assd_mm"] for row in frame_rows
                       if row["metrics"]["boundary"][name]["assd_mm"] is not None]
        per_class[name] = {
            "dice": class_aggregate["dice_by_class"][name]["dice"],
            "hd95_mm": float(np.mean(h95_values)) if h95_values else None,
            "assd_mm": float(np.mean(assd_values)) if assd_values else None,
            "frames_with_reference": class_aggregate["dice_by_class"][name]["frames_with_reference"],
        }
    fg_dice = class_aggregate["fg_dice"]
    h95 = [per_class[name]["hd95_mm"] for name in FOREGROUND_NAMES
           if per_class[name]["hd95_mm"] is not None]
    assd = [per_class[name]["assd_mm"] for name in FOREGROUND_NAMES
            if per_class[name]["assd_mm"] is not None]
    surface_status_counts = {
        status: sum(
            int(row["metrics"].get("surface_status_counts", {}).get(status, 0))
            for row in frame_rows
        )
        for status in ("ok", "both_absent", "one_absent", "no_surface")
    }
    return {
        "dice_by_class": per_class,
        "fg_dice": fg_dice,
        "hd95_mm": float(np.mean(h95)) if h95 else None,
        "assd_mm": float(np.mean(assd)) if assd else None,
        "surface_status_counts": surface_status_counts,
        "one_absent_count": surface_status_counts["one_absent"],
    }


def _clinical_patient_metrics(phases: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    if "ED" not in phases or "ES" not in phases:
        return {"status": "NOT_RUN_MISSING_EXPLICIT_ED_ES_PHASES", "errors": {}}
    result: dict[str, Any] = {"status": "OK", "errors": {}}
    for ventricle, label in (("LV", "LV_volume_ml"), ("RV", "RV_volume_ml")):
        ed = phases["ED"][label]
        es = phases["ES"][label]
        result["errors"][f"{ventricle}_EDV_abs_error_ml"] = abs(ed["prediction"] - ed["reference"])
        result["errors"][f"{ventricle}_ESV_abs_error_ml"] = abs(es["prediction"] - es["reference"])
        ref_edv, pred_edv = ed["reference"], ed["prediction"]
        ref_esv, pred_esv = es["reference"], es["prediction"]
        if ref_edv <= 0 or pred_edv <= 0:
            result["errors"][f"{ventricle}_EF_abs_error_percentage_points"] = None
            result["status"] = "PARTIAL_INVALID_EF_DENOMINATOR"
        else:
            ref_ef = 100.0 * (ref_edv - ref_esv) / ref_edv
            pred_ef = 100.0 * (pred_edv - pred_esv) / pred_edv
            result["errors"][f"{ventricle}_EF_abs_error_percentage_points"] = abs(pred_ef - ref_ef)
    for phase in ("ED", "ES"):
        entry = phases[phase]["MYO_mass_g"]
        result["errors"][f"MYO_mass_{phase}_abs_error_g"] = abs(entry["prediction"] - entry["reference"])
    return result


def _spearman(values_x: Sequence[float], values_y: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Spearman arrays must be matching vectors")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return {"status": "NOT_RUN_INSUFFICIENT_PAIRS", "n": len(x), "rho": None, "p_value": None}
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"status": "NOT_RUN_CONSTANT_INPUT", "n": len(x), "rho": None, "p_value": None}
    result = stats.spearmanr(x, y)
    return {"status": "COMPUTED", "n": len(x), "rho": float(result.statistic),
            "p_value": float(result.pvalue)}


def physiology_spearman(rows: Sequence[Mapping[str, Any]], *, split: str) -> dict[str, Any]:
    """Report DEV w_pat versus RV Dice; refuse to calculate this diagnostic on TEST."""
    if split.lower() not in {"dev", "development"}:
        raise ValueError("physiologic w_pat correlation is a DEV-only diagnostic")
    return _spearman(
        [float(row["w_pat"]) for row in rows],
        [float(row["rv_dice"]) for row in rows],
    )


def codebook_diagnostics(student_codes: Any, *, codebook_size: int,
                         teacher_codes: Any | None = None) -> dict[str, Any]:
    """Measure student occupancy and optional aligned teacher-code agreement."""
    if type(codebook_size) is not int or codebook_size < 2:
        raise ValueError("codebook_size must be an integer >= 2")
    student = np.asarray(student_codes)
    if student.size == 0 or not np.issubdtype(student.dtype, np.integer):
        raise ValueError("student hard codes must be a nonempty integer array")
    if np.any(student < 0) or np.any(student >= codebook_size):
        raise ValueError("student hard code lies outside the frozen codebook")
    counts = np.bincount(student.reshape(-1), minlength=codebook_size).astype(np.int64)
    probabilities = counts / counts.sum()
    nonzero = probabilities[probabilities > 0]
    perplexity = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    dead = counts / counts.sum() < DEAD_CODE_FREQUENCY
    result: dict[str, Any] = {
        "status": "COMPUTED",
        "student_assignment_count": int(counts.sum()),
        "perplexity": perplexity,
        "dead_code_count_below_0_1_percent": int(dead.sum()),
        "dead_code_fraction_below_0_1_percent": float(dead.mean()),
        "counts": counts.tolist(),
        "teacher_student_agreement": None,
        "teacher_student_comparison_count": 0,
    }
    if teacher_codes is not None:
        teacher = np.asarray(teacher_codes)
        if teacher.shape != student.shape or not np.issubdtype(teacher.dtype, np.integer):
            raise ValueError("teacher hard codes must be aligned to the student code lattice")
        if np.any(teacher < 0) or np.any(teacher >= codebook_size):
            raise ValueError("teacher hard code lies outside the frozen codebook")
        result["teacher_student_agreement"] = float(np.mean(teacher == student))
        result["teacher_student_comparison_count"] = int(student.size)
    else:
        result["teacher_student_status"] = "NOT_RUN_NO_FROZEN_TEACHER_CODES_SUPPLIED"
    return result


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, samples: int = 5000) -> list[float] | None:
    if len(values) < 2:
        return None
    if type(samples) is not int or samples < 100:
        raise ValueError("bootstrap samples must be an integer >= 100")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def _paired_comparison(reference: Mapping[str, Mapping[str, Any]],
                       candidate: Mapping[str, Mapping[str, Any]], *,
                       reference_arm: str, candidate_arm: str, seed: int,
                       bootstrap_samples: int) -> dict[str, Any]:
    common = sorted(set(reference) & set(candidate))
    dice_pairs = [(pid, reference[pid].get("fg_dice"), candidate[pid].get("fg_dice")) for pid in common]
    dice_pairs = [(pid, float(a), float(b)) for pid, a, b in dice_pairs
                  if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    h95_pairs = [(pid, reference[pid].get("hd95_mm"), candidate[pid].get("hd95_mm")) for pid in common]
    h95_pairs = [(pid, float(a), float(b)) for pid, a, b in h95_pairs
                 if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    dice_delta = np.asarray([b - a for _, a, b in dice_pairs], dtype=np.float64)
    h95_delta = np.asarray([b - a for _, a, b in h95_pairs], dtype=np.float64)
    enough = len(dice_delta) >= 2
    paired_ids = [pid for pid, _, _ in dice_pairs]
    absent_pairs: list[tuple[str, int, int]] = []
    missing_absent_ids: list[str] = []
    for patient_id in paired_ids:
        ref_row, cand_row = reference[patient_id], candidate[patient_id]
        ref_count = ref_row.get("one_absent_count")
        cand_count = cand_row.get("one_absent_count")
        if ref_count is None:
            ref_count = ref_row.get("surface_status_counts", {}).get("one_absent")
        if cand_count is None:
            cand_count = cand_row.get("surface_status_counts", {}).get("one_absent")
        if (type(ref_count) is not int or ref_count < 0
                or type(cand_count) is not int or cand_count < 0):
            missing_absent_ids.append(patient_id)
            continue
        absent_pairs.append((patient_id, ref_count, cand_count))
    absent_worse_ids = [pid for pid, ref_count, cand_count in absent_pairs if cand_count > ref_count]
    absent_better_ids = [pid for pid, ref_count, cand_count in absent_pairs if cand_count < ref_count]
    absent_count_ready = not missing_absent_ids and len(absent_pairs) == len(paired_ids)
    absent_nonworse = absent_count_ready and not absent_worse_ids
    return {
        "reference_arm": reference_arm,
        "candidate_arm": candidate_arm,
        "paired_patient_count": len(dice_delta),
        "paired_patient_ids": paired_ids,
        "fg_dice_delta_candidate_minus_reference": float(dice_delta.mean()) if len(dice_delta) else None,
        "fg_dice_delta_bootstrap_ci95": _bootstrap_mean_ci(dice_delta, seed=seed, samples=bootstrap_samples),
        "fg_dice_win_rate": float(np.mean(dice_delta > 0)) if len(dice_delta) else None,
        "fg_dice_tie_rate": float(np.mean(dice_delta == 0)) if len(dice_delta) else None,
        "hd95_delta_mm_candidate_minus_reference": float(h95_delta.mean()) if len(h95_delta) else None,
        "hd95_delta_bootstrap_ci95_mm": _bootstrap_mean_ci(h95_delta, seed=seed + 1, samples=bootstrap_samples),
        "hd95_win_rate": float(np.mean(h95_delta < 0)) if len(h95_delta) else None,
        "harm_rate_candidate_vs_reference": float(np.mean(dice_delta < -HARM_MARGIN)) if len(dice_delta) else None,
        "one_absent": {
            "status": "COMPUTED" if absent_count_ready else "NOT_REPORTED_FAIL_CLOSED",
            "reference_total": sum(ref_count for _, ref_count, _ in absent_pairs),
            "candidate_total": sum(cand_count for _, _, cand_count in absent_pairs),
            "candidate_minus_reference_total": sum(
                cand_count - ref_count for _, ref_count, cand_count in absent_pairs
            ),
            "paired_patient_count": len(absent_pairs),
            "patient_counts": {
                pid: {"reference": ref_count, "candidate": cand_count,
                      "candidate_minus_reference": cand_count - ref_count}
                for pid, ref_count, cand_count in absent_pairs
            },
            "patients_worse": absent_worse_ids,
            "patients_better": absent_better_ids,
            "missing_patient_ids": missing_absent_ids,
            "nonworse": bool(absent_nonworse),
        },
        "decision_status": (
            "NO_GO_INSUFFICIENT_PAIRED_PATIENTS" if not enough else
            "NO_GO_MISSING_ONE_ABSENT_COUNTS" if not absent_count_ready else "COMPUTED"
        ),
    }


def compare_primary_arms(patient_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]], *,
                         split: str, seed: int = 42, bootstrap_samples: int = 5000) -> dict[str, Any]:
    """Apply the predeclared V1-vs-C1/C0/V1-U decision gates on paired patients."""
    if split.lower() not in {"test", "external_test", "external_frozen"}:
        raise ValueError("primary Q-3D decision gates may only consume the locked TEST split")
    required = ("C0", "C1", "V1", "V1-U")
    missing = [arm for arm in required if arm not in patient_metrics]
    if missing:
        raise ValueError(f"paired primary-arm comparison is missing arms: {missing}")
    comparisons = {
        "vq_vs_continuous_kd": _paired_comparison(
            patient_metrics["C1"], patient_metrics["V1"], reference_arm="C1", candidate_arm="V1",
            seed=seed, bootstrap_samples=bootstrap_samples,
        ),
        "teacher_vs_student_only": _paired_comparison(
            patient_metrics["C0"], patient_metrics["V1"], reference_arm="C0", candidate_arm="V1",
            seed=seed + 10, bootstrap_samples=bootstrap_samples,
        ),
        "physiology_weight_vs_uniform": _paired_comparison(
            patient_metrics["V1-U"], patient_metrics["V1"], reference_arm="V1-U", candidate_arm="V1",
            seed=seed + 20, bootstrap_samples=bootstrap_samples,
        ),
    }
    uniform_harms: list[float] = []
    physiology_harms: list[float] = []
    harm_patient_ids: list[str] = []
    for patient_id in sorted(set(patient_metrics["C0"]) & set(patient_metrics["V1-U"])
                             & set(patient_metrics["V1"])):
        baseline = patient_metrics["C0"][patient_id].get("fg_dice")
        uniform = patient_metrics["V1-U"][patient_id].get("fg_dice")
        weighted = patient_metrics["V1"][patient_id].get("fg_dice")
        if baseline is None or uniform is None or weighted is None:
            continue
        if not all(np.isfinite(float(value)) for value in (baseline, uniform, weighted)):
            continue
        harm_patient_ids.append(patient_id)
        uniform_harms.append(float(uniform) < float(baseline) - HARM_MARGIN)
        physiology_harms.append(float(weighted) < float(baseline) - HARM_MARGIN)
    if uniform_harms:
        comparisons["physiology_weight_vs_uniform"]["harm_rate_v1_u_vs_c0"] = float(np.mean(uniform_harms))
        comparisons["physiology_weight_vs_uniform"]["harm_rate_v1_vs_c0"] = float(np.mean(physiology_harms))
        harm_delta = np.asarray(uniform_harms, dtype=np.float64) - np.asarray(
            physiology_harms, dtype=np.float64
        )
        harm_ci = _bootstrap_mean_ci(harm_delta, seed=seed + 30, samples=bootstrap_samples)
        comparisons["physiology_weight_vs_uniform"].update({
            "harm_reduction_paired_patient_count": len(harm_patient_ids),
            "harm_reduction_patient_ids": harm_patient_ids,
            "harm_rate_reduction_v1_u_minus_v1": float(harm_delta.mean()),
            "harm_rate_reduction_v1_u_minus_v1_bootstrap_ci95": harm_ci,
            "harm_rate_reduction_status": (
                "COMPUTED" if harm_ci is not None else "INCONCLUSIVE_INSUFFICIENT_PAIRS"
            ),
        })
    harm_vs_c0: dict[str, Any] = {}
    for arm in ("C1", "V1", "V1-U"):
        paired_harms: list[bool] = []
        for patient_id in sorted(set(patient_metrics["C0"]) & set(patient_metrics[arm])):
            baseline = patient_metrics["C0"][patient_id].get("fg_dice")
            candidate = patient_metrics[arm][patient_id].get("fg_dice")
            if baseline is None or candidate is None:
                continue
            if not np.isfinite(float(baseline)) or not np.isfinite(float(candidate)):
                continue
            paired_harms.append(float(candidate) < float(baseline) - HARM_MARGIN)
        harm_vs_c0[arm] = {
            "paired_patient_count": len(paired_harms),
            "harm_margin_fg_dice": HARM_MARGIN,
            "rate": float(np.mean(paired_harms)) if paired_harms else None,
            "status": "COMPUTED" if paired_harms else "NO_GO_INSUFFICIENT_PAIRED_PATIENTS",
        }
    def gate(key: str) -> str:
        row = comparisons[key]
        if row["decision_status"] != "COMPUTED":
            return "NO_GO"
        dice_delta = row["fg_dice_delta_candidate_minus_reference"]
        h95_delta = row["hd95_delta_mm_candidate_minus_reference"]
        if not row["one_absent"]["nonworse"]:
            return "NO_GO"
        if key == "physiology_weight_vs_uniform":
            if dice_delta is not None and dice_delta > 0:
                return "GO"
            if row.get("harm_rate_reduction_status") != "COMPUTED":
                return "INCONCLUSIVE"
            harm_reduction = row.get("harm_rate_reduction_v1_u_minus_v1")
            harm_ci = row.get("harm_rate_reduction_v1_u_minus_v1_bootstrap_ci95")
            harm_clear = (
                harm_reduction is not None and harm_reduction > 0
                and harm_ci is not None
                and harm_ci[0] > HARM_REDUCTION_CI95_LOWER_BOUND
            )
            return "GO" if harm_clear else "INCONCLUSIVE"
        return "GO" if (dice_delta is not None and dice_delta >= MIN_FG_DICE_DELTA
                        and h95_delta is not None and h95_delta <= 0) else "NO_GO"
    return {
        "split": split,
        "thresholds_locked_from_spec": {
            "minimum_fg_dice_delta": MIN_FG_DICE_DELTA,
            "hd95_not_worse": True,
            "harm_margin": HARM_MARGIN,
            "bootstrap_confidence": 0.95,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "physiology_harm_rate_delta": "paired_harm_V1U_minus_harm_V1",
            "physiology_harm_rate_ci95_lower_bound_must_exceed": HARM_REDUCTION_CI95_LOWER_BOUND,
            "physiology_minimum_patient_pairs_for_bootstrap": 2,
            "one_absent_no_worse_rule": "candidate one_absent count must not exceed reference per paired patient",
        },
        "comparisons": comparisons,
        "harm_rate_vs_c0": harm_vs_c0,
        "decisions": {
            "vq_contribution": gate("vq_vs_continuous_kd"),
            "teacher_contribution": gate("teacher_vs_student_only"),
            "physiology_contribution": gate("physiology_weight_vs_uniform"),
        },
        "evidence_status": "PATIENT_EVIDENCE_ONLY_IF_INPUTS_ARE_LOCKED_REAL_TEST_PATIENTS",
    }


def _phase_for_frame(record: Mapping[str, Any], metadata: Mapping[str, Any], frame: int) -> str | None:
    phases = record.get("phases", metadata.get("phases", {}))
    if isinstance(phases, Mapping):
        for phase_name in ("ED", "ES", "ed", "es"):
            if phases.get(phase_name) is not None and int(phases[phase_name]) == frame:
                return phase_name.upper()
    for phase_name in ("ED", "ES"):
        raw = record.get(f"{phase_name.lower()}_frame", metadata.get(f"{phase_name.lower()}_frame"))
        if raw is not None and int(raw) == frame:
            return phase_name
    return None


def _validated_manifest(manifest_path: str | Path, manifest_sha256: str, *,
                        train_patient_ids: Iterable[str] = (), expected_role: str | None = None,
                        expected_synthetic: bool | None = None) -> tuple[Path, dict, list[dict], dict, Any]:
    path = Path(manifest_path).resolve()
    if digest(path) != manifest_sha256:
        raise ValueError("Q-3D evaluation manifest SHA-256 mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "self_audit.data.v2":
        raise ValueError("Q-3D evaluation requires the declared self_audit.data.v2 manifest")
    role = manifest.get("role")
    allowed_roles = {"development", "test", "external_frozen"}
    if role not in allowed_roles or (expected_role is not None and role != expected_role):
        raise ValueError("Q-3D evaluation manifest role does not match the requested frozen split")
    if manifest.get("class_mapping") != CLASS_MAPPING:
        raise ValueError("Q-3D evaluation requires the explicit BG/RV/MYO/LV class mapping")
    if expected_synthetic is not None and bool(manifest.get("synthetic", False)) != bool(expected_synthetic):
        raise ValueError("Q-3D synthetic/patient identity mismatch")
    patients = manifest.get("patients")
    if not isinstance(patients, list) or not patients:
        raise ValueError("Q-3D evaluation manifest must contain patient records")
    ids: list[str] = []
    for record in patients:
        if not isinstance(record, dict) or {"id", "image", "image_sha256", "masks"} - set(record):
            raise ValueError("Q-3D evaluation patient record is incomplete")
        if not isinstance(record["id"], str) or not record["id"]:
            raise ValueError("Q-3D patient id must be a nonempty string")
        ids.append(record["id"])
        masks = record["masks"]
        if not isinstance(masks, dict) or not masks:
            raise ValueError("Q-3D evaluation patient requires explicit labeled frames")
        for frame, mask_record in masks.items():
            if not str(frame).isdigit() or not isinstance(mask_record, dict) or {"path", "sha256"} - set(mask_record):
                raise ValueError("Q-3D mask manifest record is invalid")
    if len(ids) != len(set(ids)):
        raise ValueError("Q-3D evaluation patient IDs must be unique")
    overlap = set(ids) & set(train_patient_ids)
    if overlap:
        raise ValueError(f"Q-3D evaluation patients overlap with TRAIN: {sorted(overlap)}")
    geometries = geometry_map(manifest)
    if set(geometries) != set(ids):
        raise ValueError("every Q-3D evaluation patient requires explicit native geometry")
    resolve = lambda value: Path(value) if Path(value).is_absolute() else path.parent / value
    return path, manifest, patients, geometries, resolve


def _predict_logits(model: torch.nn.Module, image: torch.Tensor,
                    geometry: Mapping[str, Any], device: str | torch.device) -> tuple[np.ndarray, np.ndarray | None]:
    output = model(image.to(device), geometry=geometry)
    if not isinstance(output, Mapping):
        raise TypeError("Q-3D student forward must return the canonical output mapping")
    logits = output.get("logits", output.get("logits_S"))
    expected = (1, 4, *geometry["shape"])
    if not torch.is_tensor(logits) or tuple(logits.shape) != expected:
        raise ValueError(f"Q-3D student logits must have shape {expected}")
    if not torch.isfinite(logits).all():
        raise ValueError("Q-3D student logits contain NaN or Inf")
    predicted = logits.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8, copy=False)
    hard_codes = output.get("codes")
    codes = None
    if hard_codes is not None:
        if not torch.is_tensor(hard_codes) or hard_codes.ndim != 4 or hard_codes.shape[0] != 1:
            raise ValueError("student hard codes must have shape [1,D,h,w]")
        codes = hard_codes[0].detach().to("cpu").numpy()
    return predicted, codes


def evaluate_manifest(model: torch.nn.Module, manifest_path: str | Path, manifest_sha256: str,
                      out_dir: str | Path, *, device: str | torch.device = "cpu",
                      split: str | None = None, train_patient_ids: Iterable[str] = (),
                      expected_synthetic: bool | None = None,
                      teacher_codes_by_frame: Mapping[tuple[str, int], Any] | None = None,
                      physiology_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Evaluate a frozen native manifest without using labels before prediction freeze."""
    requested_role = {"dev": "development", "external_test": "external_frozen"}.get(
        split.lower(), split
    ) if isinstance(split, str) else split
    path, manifest, patients, geometries, resolve = _validated_manifest(
        manifest_path, manifest_sha256, train_patient_ids=train_patient_ids,
        expected_role=requested_role, expected_synthetic=expected_synthetic,
    )
    role = str(manifest["role"])
    out = Path(out_dir).resolve()
    if out.exists() or out.is_relative_to(path.parent) or path.parent.is_relative_to(out):
        raise ValueError("Q-3D evaluation output must be a new path outside the input manifest tree")
    out.mkdir(parents=True)
    prediction_dir = out / "predictions"
    prediction_dir.mkdir()
    old_training = model.training
    model = model.to(device).eval()
    patient_ids = [str(record["id"]) for record in patients]
    prediction_files: list[dict[str, Any]] = []
    code_arrays: list[np.ndarray] = []
    student_codes_by_frame: dict[tuple[str, int], np.ndarray] = {}
    code_size = int(getattr(model, "codebook_size", 0) or 0)
    try:
        with torch.inference_mode():
            for record in patients:
                patient_id = str(record["id"])
                image_path = resolve(record["image"])
                if digest(image_path) != record["image_sha256"]:
                    raise ValueError(f"Q-3D image hash mismatch for patient {patient_id}")
                image = np.load(image_path, mmap_mode="r", allow_pickle=False)
                if image.ndim != 4 or image.shape[0] < 1:
                    raise ValueError("Q-3D cached image must have native shape [T,D,H,W]")
                if not np.isfinite(image).all() or float(image.min()) < 0 or float(image.max()) > 1:
                    raise ValueError("Q-3D cached image must be finite and normalized to [0,1]")
                depth, height, width = (int(v) for v in image.shape[1:])
                raw_geometry = geometries[patient_id]
                cache_geometry = VolumeGeometry3D.from_cache_geometry(
                    raw_geometry, expected_shape=(depth, height, width)
                )
                runtime_geometry = cache_geometry.as_dict()
                for frame_text in sorted(record["masks"], key=int):
                    frame = int(frame_text)
                    if frame < 0 or frame >= image.shape[0]:
                        raise ValueError(f"Q-3D labeled frame index is outside cine for {patient_id}")
                    x, canonical = validate_native_image(np.array(image[frame], copy=True), runtime_geometry)
                    predicted, codes = _predict_logits(model, x, canonical, device)
                    file_path = prediction_dir / f"{patient_id}_t{frame:04d}.npy"
                    np.save(file_path, predicted, allow_pickle=False)
                    prediction_files.append({"patient": patient_id, "frame": frame,
                                             "file": file_path.name, "sha256": digest(file_path),
                                             "shape_zyx": [depth, height, width]})
                    if codes is not None:
                        hard = codes.astype(np.int64, copy=False)
                        code_arrays.append(hard.reshape(-1))
                        student_codes_by_frame[(patient_id, frame)] = hard.copy()
    finally:
        model.train(old_training)
    write_json(out / "PREDICTIONS_FROZEN.json", {
        "schema": "q3d.predictions.v1", "manifest_sha256": manifest_sha256,
        "role": role, "synthetic": bool(manifest.get("synthetic", False)),
        "predictions": prediction_files,
    })

    rows: list[dict[str, Any]] = []
    clinical_by_patient: dict[str, dict[str, dict[str, Any]]] = {pid: {} for pid in patient_ids}
    metadata_by_id = manifest.get("patient_metadata", {}) or {}
    for record in patients:
        patient_id = str(record["id"])
        metadata = metadata_by_id.get(patient_id, {})
        if not isinstance(metadata, Mapping):
            raise TypeError(f"patient_metadata[{patient_id!r}] must be a mapping")
        raw_geometry = geometries[patient_id]
        image_path = resolve(record["image"])
        image = np.load(image_path, mmap_mode="r", allow_pickle=False)
        depth, height, width = (int(v) for v in image.shape[1:])
        cache_geometry = VolumeGeometry3D.from_cache_geometry(
            raw_geometry, expected_shape=(depth, height, width)
        )
        for frame_text, mask_record in sorted(record["masks"].items(), key=lambda item: int(item[0])):
            frame = int(frame_text)
            mask_path = resolve(mask_record["path"])
            if digest(mask_path) != mask_record["sha256"]:
                raise ValueError(f"Q-3D reference mask hash mismatch for patient {patient_id} frame {frame}")
            reference = _validate_label(np.load(mask_path, allow_pickle=False), (depth, height, width))
            prediction_path = prediction_dir / f"{patient_id}_t{frame:04d}.npy"
            if digest(prediction_path) != next(
                row["sha256"] for row in prediction_files
                if row["patient"] == patient_id and row["frame"] == frame
            ):
                raise ValueError("Q-3D frozen prediction changed before scoring")
            prediction = np.load(prediction_path, allow_pickle=False)
            if prediction.shape != reference.shape or not np.isin(prediction, (0, 1, 2, 3)).all():
                raise ValueError("Q-3D frozen prediction has an invalid native mask")
            frame_metric = _frame_metrics(reference, prediction, cache_geometry.spacing_zyx)
            phase = _phase_for_frame(record, metadata, frame)
            row = {"patient": patient_id, "frame": frame, "phase": phase, "metrics": frame_metric}
            rows.append(row)
            if phase is not None:
                if phase in clinical_by_patient[patient_id]:
                    raise ValueError(f"duplicate explicit {phase} frame for patient {patient_id}")
                clinical_by_patient[patient_id][phase] = _clinical_frame(
                    reference, prediction, cache_geometry.voxel_volume_mm3
                )

    per_patient = {
        patient_id: {
            **_patient_aggregate([row for row in rows if row["patient"] == patient_id]),
            "clinical": _clinical_patient_metrics(clinical_by_patient[patient_id]),
        }
        for patient_id in patient_ids
    }
    surface_status_totals = {
        status: sum(
            int(entry.get("surface_status_counts", {}).get(status, 0))
            for entry in per_patient.values()
        )
        for status in ("ok", "both_absent", "one_absent", "no_surface")
    }
    def macro(metric: str) -> float | None:
        values = [entry.get(metric) for entry in per_patient.values()
                  if entry.get(metric) is not None and np.isfinite(entry[metric])]
        return float(np.mean(values)) if values else None
    class_macro: dict[str, Any] = {}
    for name in FOREGROUND_NAMES:
        values = [entry["dice_by_class"][name]["dice"] for entry in per_patient.values()
                  if entry["dice_by_class"][name]["dice"] is not None]
        h95 = [entry["dice_by_class"][name]["hd95_mm"] for entry in per_patient.values()
               if entry["dice_by_class"][name]["hd95_mm"] is not None]
        assd = [entry["dice_by_class"][name]["assd_mm"] for entry in per_patient.values()
                if entry["dice_by_class"][name]["assd_mm"] is not None]
        class_macro[name] = {
            "patient_macro_dice": float(np.mean(values)) if values else None,
            "patient_macro_hd95_mm": float(np.mean(h95)) if h95 else None,
            "patient_macro_assd_mm": float(np.mean(assd)) if assd else None,
            "patients_counted": len(values),
        }
    fg = [entry["fg_dice"] for entry in per_patient.values() if entry["fg_dice"] is not None]
    h95 = [entry["hd95_mm"] for entry in per_patient.values() if entry["hd95_mm"] is not None]
    assd = [entry["assd_mm"] for entry in per_patient.values() if entry["assd_mm"] is not None]
    clinical_names = sorted({key for entry in per_patient.values()
                             for key in entry["clinical"]["errors"]})
    clinical_macro = {}
    for name in clinical_names:
        values = [entry["clinical"]["errors"].get(name) for entry in per_patient.values()
                  if entry["clinical"]["errors"].get(name) is not None]
        clinical_macro[name] = {"patient_macro_mae": float(np.mean(values)) if values else None,
                                "patients_counted": len(values)}
    code_metrics: dict[str, Any]
    if code_arrays and code_size >= 2:
        code_metrics = codebook_diagnostics(np.concatenate(code_arrays), codebook_size=code_size)
    else:
        code_metrics = {"status": "NOT_RUN_STUDENT_HARD_CODES_UNAVAILABLE",
                        "teacher_student_agreement": None,
                        "teacher_student_status": "NOT_RUN_NO_FROZEN_TEACHER_CODES_SUPPLIED"}
    if teacher_codes_by_frame is not None and student_codes_by_frame:
        if role != "development":
            raise ValueError("teacher/student code agreement is a DEV-only diagnostic")
        student_aligned: list[np.ndarray] = []
        teacher_aligned: list[np.ndarray] = []
        for key, teacher_code_value in teacher_codes_by_frame.items():
            if key not in student_codes_by_frame:
                continue
            student_code = student_codes_by_frame[key]
            teacher_code = np.asarray(teacher_code_value)
            if teacher_code.shape != student_code.shape:
                raise ValueError("frozen teacher hard codes must match the student stage-4 lattice")
            student_aligned.append(student_code.reshape(-1))
            teacher_aligned.append(teacher_code.reshape(-1))
        if student_aligned:
            code_metrics["teacher_student_agreement"] = float(
                np.mean(np.concatenate(teacher_aligned) == np.concatenate(student_aligned))
            )
            code_metrics["teacher_student_comparison_count"] = int(sum(a.size for a in student_aligned))
            code_metrics.pop("teacher_student_status", None)
        else:
            code_metrics["teacher_student_status"] = "NOT_RUN_NO_MATCHING_FROZEN_TEACHER_CODES"
    physiology_metric = {
        "status": "NOT_RUN_REQUIRES_DEV_W_PAT_AND_TEACHER_RV_DICE_ROWS",
        "n": 0, "rho": None, "p_value": None,
    }
    if physiology_rows is not None:
        physiology_metric = physiology_spearman(physiology_rows, split=role)
    result = {
        "schema": "q3d.evaluation.v1",
        "role": role,
        "split": role,
        "dataset": manifest.get("dataset", manifest.get("dataset_name", "UNDECLARED")),
        "manifest_sha256": manifest_sha256,
        "synthetic": bool(manifest.get("synthetic", False)),
        "labels_used_for": "frozen segmentation and clinical scoring only; no training or checkpoint selection",
        "patients": len(patient_ids),
        "frames": len(rows),
        "per_patient": per_patient,
        "surface_status_counts": surface_status_totals,
        "patient_macro": {
            "dice_by_class": class_macro,
            "fg_dice": float(np.mean(fg)) if fg else None,
            "hd95_mm": float(np.mean(h95)) if h95 else None,
            "assd_mm": float(np.mean(assd)) if assd else None,
            "one_absent_count": surface_status_totals["one_absent"],
            "clinical_abs_error": clinical_macro,
        },
        "frame_metrics": rows,
        "codebook": code_metrics,
        "physiology_w_pat_spearman": physiology_metric,
        "cost": {
            "parameters": "NOT_RUN_USE_Q3D_PROFILE",
            "flops": "NOT_RUN_USE_Q3D_PROFILE",
            "peak_vram_mb": "NOT_RUN_USE_Q3D_PROFILE",
            "gpu_latency_ms_per_volume": "NOT_RUN_USE_Q3D_PROFILE",
            "cpu_latency_ms_per_volume": "NOT_RUN_USE_Q3D_PROFILE",
        },
        "prediction_manifest": "PREDICTIONS_FROZEN.json",
    }
    if result["synthetic"]:
        result["evidence_status"] = "SOFTWARE_ONLY_SYNTHETIC"
        result["warning"] = "Synthetic values are not patient metrics or clinical evidence."
    else:
        result["evidence_status"] = "PATIENT_METRICS_FROM_FROZEN_MANIFEST"
    write_json(out / "Q3D_METRICS.json", result)
    return result


__all__ = [
    "BOUNDARY_TOLERANCE_MM", "CLASS_NAMES", "FOREGROUND_NAMES", "aggregate_patient_class_dice",
    "frame_class_dice", "codebook_diagnostics",
    "compare_primary_arms", "evaluate_manifest", "physiology_spearman", "surface_distances_3d",
    "validate_native_image",
]
