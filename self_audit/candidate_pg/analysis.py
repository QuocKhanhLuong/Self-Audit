"""Analysis helpers for PG's locked selection and paired diagnostics."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scripts.fuse_compare_results import paired_statistics

ARMS = ("A1", "A2", "A3", "A4")
PAIRS = (("A2", "A1"), ("A3", "A1"), ("A4", "A1"), ("A4", "A2"), ("A4", "A3"))


def _patient_metric(record: Mapping[str, Any], metric: str = "FG") -> float:
    if metric in {"FG", "fg_dice", "fg"}:
        value = record.get("fg_dice")
    else:
        aliases = {"RV": "RV", "MYO": "MYO", "LV": "LV"}
        if "." in metric:
            cls, field = metric.split(".", 1)
        else:
            cls, field = metric, "dice"
        value = record.get("per_class", {}).get(aliases.get(cls, cls), {}).get(field)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"missing/nonfinite PG patient metric {metric!r}")
    return float(value)


def _patient_rows(value: Any) -> Mapping[str, Mapping[str, Any]]:
    if isinstance(value, Mapping) and isinstance(value.get("per_patient"), Mapping):
        return value["per_patient"]
    if isinstance(value, Mapping):
        return value
    raise TypeError("PG arm metrics must be a patient mapping or contain per_patient")


def paired_comparisons(arms: Mapping[str, Any], *, split: str = "TEST50",
                       metrics: Sequence[str] = ("FG", "RV.dice", "MYO.dice", "LV.dice",
                                                 "RV.hd95_mm", "MYO.hd95_mm", "LV.hd95_mm",
                                                 "RV.assd_mm", "MYO.assd_mm", "LV.assd_mm"),
                       bootstrap_resamples: int = 10_000, seed: int = 42,
                       opened_test: bool = True) -> dict[str, Any]:
    """Run the five predeclared paired arm comparisons at patient granularity."""
    result: dict[str, Any] = {
        "schema": "self_audit.candidate_pg.paired_comparisons.v1",
        "split": split, "independent_unit": "patient", "bootstrap_seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "analysis_status": "EXPLORATORY_POST_TEST_OPEN" if opened_test else "EXPLORATORY",
        "comparisons": {},
    }
    split_values = {}
    for arm, value in arms.items():
        if isinstance(value, Mapping) and split in value and isinstance(value[split], Mapping):
            split_values[arm] = value[split]
        else:
            split_values[arm] = value
    for left, right in PAIRS:
        key = f"{left}_minus_{right}"
        if left not in split_values or right not in split_values:
            raise ValueError(f"PG paired comparison requires both arms for {key}")
        left_rows, right_rows = _patient_rows(split_values[left]), _patient_rows(split_values[right])
        metrics_result = {}
        for metric in metrics:
            first = {str(pid): _patient_metric(record, metric) for pid, record in left_rows.items()}
            second = {str(pid): _patient_metric(record, metric) for pid, record in right_rows.items()}
            # paired_statistics is intentionally allowed to raise on missing,
            # mismatched, or non-finite patient data: NOT_RUN is reserved for
            # absent campaigns, not malformed evidence.
            metrics_result[metric] = paired_statistics(first, second,
                                                       bootstrap_resamples=bootstrap_resamples,
                                                       seed=seed)
        result["comparisons"][key] = metrics_result
    return result


def select_val_epoch(rows: Sequence[Mapping[str, Any]], *, metric: str = "fg_dice",
                     path: str = "student_only") -> dict[str, Any]:
    """Compatibility alias for the central logger VAL-only selector."""
    from .logging import select_best_epoch as central_select
    normalized = [dict(row, update=int(row.get("update", int(row.get("epoch", 0)) * 80)))
                  for row in rows]
    return central_select(normalized, split="VAL18", path=path, metric=metric)


def compare_logged_export(logged: Mapping[str, Any], recomputed: Mapping[str, Any], *,
                          fields: Sequence[str] = ("fg_dice", "rv_dice", "myo_dice", "lv_dice",
                                                   "fg_hd95", "rv_hd95", "myo_hd95", "lv_hd95",
                                                   "fg_assd", "rv_assd", "myo_assd", "lv_assd"),
                          atol: float = 1e-7) -> dict[str, Any]:
    """Compare a selected-epoch CSV/JSON row against exported recomputation."""
    differences = {}
    def value(row: Mapping[str, Any], field: str) -> Any:
        direct = row.get(field)
        if direct is not None:
            return direct
        if field.startswith("fg_"):
            return row.get(field)
        parts = field.split("_", 1)
        if len(parts) == 2:
            class_name = {"rv": "RV", "myo": "MYO", "lv": "LV"}.get(parts[0])
            metric_name = {"dice": "dice", "hd95": "hd95_mm", "assd": "assd_mm"}.get(parts[1])
            if class_name and metric_name:
                return row.get("per_class", {}).get(class_name, {}).get(metric_name)
        return None
    for field in fields:
        a, b = value(logged, field), value(recomputed, field)
        if a is None or b is None:
            differences[field] = {"status": "FAIL_MISSING_METRIC", "logged": a, "recomputed": b}
            continue
        delta = abs(float(a) - float(b))
        differences[field] = {"status": "PASS" if delta <= atol else "FAIL",
                              "logged": float(a), "recomputed": float(b), "abs_diff": delta}
    return {"status": ("PASS" if all(v["status"] == "PASS"
                                        for v in differences.values()) else "FAIL"),
            "atol": atol, "fields": differences}


def _shuffle_codes(codes: np.ndarray, rng: np.random.Generator,
                   valid_mask: np.ndarray | None = None) -> np.ndarray:
    result = np.array(codes, copy=True, order="C")
    if result.ndim not in (4, 5):
        raise ValueError("qS codes must be [B,D,H,W] IDs or [B,C,D,H,W] vectors")
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != result.shape:
            # Permit a batch dimension in the mask when codes are batched.
            if result.ndim == 4 and valid_mask.shape == result.shape[1:]:
                valid_mask = np.broadcast_to(valid_mask, result.shape)
            else:
                raise ValueError("qS valid_mask shape differs from code lattice")
    for batch in range(result.shape[0]):
        vectors = result[batch].reshape((-1, int(np.prod(result.shape[-3:]))))
        if valid_mask is None:
            eligible = np.arange(vectors.shape[-1])
        else:
            masks = valid_mask[batch].reshape(vectors.shape)
            if not np.all(masks == masks[:1]):
                raise ValueError("all code channels must share spatial validity")
            eligible = np.flatnonzero(masks[0])
        if len(eligible) > 1:
            # Move whole code vectors, using one permutation for every channel.
            # Independent channel permutations would invent vectors absent in C.
            source = vectors[:, eligible].copy()
            vectors[:, eligible] = source[:, rng.permutation(len(eligible))]
    return result


def code_spatial_shuffle_diagnostic(model: Any, codes: Any, *, baseline_logits: Any = None,
                                    seed: int = 42, path: str = "qS", target: Any = None,
                                    valid_mask: Any = None) -> dict[str, Any]:
    """Shuffle qS code positions (not images) for the A3/A4 VAL diagnostic."""
    if valid_mask is None:
        raise ValueError("qS spatial shuffle requires an explicit valid_mask")
    original = np.asarray(codes)
    valid_array = np.asarray(valid_mask, dtype=bool)
    if valid_array.shape != original.shape and not (original.ndim == 4 and valid_array.shape == original.shape[1:]):
        raise ValueError("qS valid_mask shape differs from code lattice")
    rng = np.random.default_rng(seed)
    shuffled = _shuffle_codes(original, rng, valid_array)
    if model is None:
        return {"status": "NOT_RUN_NO_QS_MODEL", "seed": seed,
                "changed_positions": int(np.count_nonzero(original != shuffled))}
    code_head = getattr(model, "H_S_Q", None)
    if not callable(code_head):
        raise TypeError("qS diagnostic requires the H_S_Q code head")

    def call(value):
        output = code_head(value)
        if isinstance(output, Mapping):
            output = output.get("logits", output.get("q_logits", output.get("H_S_Q")))
        if output is None:
            raise TypeError("H_S_Q returned no logits")
        return output
    baseline = np.asarray(call(original)) if baseline_logits is None else np.asarray(baseline_logits)
    shuffled_logits = np.asarray(call(shuffled))
    if baseline.shape != shuffled_logits.shape:
        raise ValueError("qS baseline and shuffled logits shapes differ")
    delta = np.abs(baseline.astype(np.float64) - shuffled_logits.astype(np.float64))
    # qS's point is spatial code identity, therefore report a numeric shift;
    # segmentation Dice is added by the caller when labels are available.
    result = {"status": "PASS", "seed": seed, "path": path,
            "changed_positions": int(np.count_nonzero(original != shuffled)),
            "logit_mean_abs_delta": float(delta.mean()),
            "logit_max_abs_delta": float(delta.max()),
            "spatial_shuffle": True}
    if target is not None:
        truth = np.asarray(target)
        base = np.argmax(baseline, axis=1 if baseline.ndim == 5 else 0)
        shuffled_pred = np.argmax(shuffled_logits, axis=1 if shuffled_logits.ndim == 5 else 0)
        if base.ndim == 4 and base.shape[0] == 1:
            base = base[0]
        if shuffled_pred.ndim == 4 and shuffled_pred.shape[0] == 1:
            shuffled_pred = shuffled_pred[0]
        if base.shape != truth.shape or shuffled_pred.shape != truth.shape:
            raise ValueError("qS diagnostic target and logits shapes differ")
        values = []
        for class_id in (1, 2, 3):
            ref = truth == class_id
            den_base = ref.sum() + (base == class_id).sum()
            den_shuffle = ref.sum() + (shuffled_pred == class_id).sum()
            values.append((2 * np.count_nonzero(ref & (base == class_id)) / den_base) if den_base else None)
            values.append((2 * np.count_nonzero(ref & (shuffled_pred == class_id)) / den_shuffle) if den_shuffle else None)
        base_values = [v for v in values[::2] if v is not None]; shuffle_values = [v for v in values[1::2] if v is not None]
        result.update({"baseline_fg_dice": float(np.mean(base_values)) if base_values else None,
                       "shuffled_fg_dice": float(np.mean(shuffle_values)) if shuffle_values else None})
    return result


def add_shuffle_metrics(report: Mapping[str, Any], *, arms: Sequence[str] = ("A3", "A4")) -> dict[str, Any]:
    output = copy.deepcopy(dict(report))
    diagnostics = output.setdefault("shuffle_diagnostics", {})
    for arm in arms:
        diagnostics.setdefault(arm, {"status": "NOT_RUN_NO_QS_DIAGNOSTIC"})
    return output


paired_five_comparisons = paired_comparisons


__all__ = ["ARMS", "PAIRS", "add_shuffle_metrics", "code_spatial_shuffle_diagnostic",
           "compare_logged_export", "paired_comparisons", "paired_five_comparisons",
           "select_val_epoch"]
