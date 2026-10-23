#!/usr/bin/env python3
"""Retrospective endpoint-selection bounds; not estimates of training-guard benefit."""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess

REV = "8960b6baac4c6600553d900c303f192c2fcfa4f0"
PATH = "results/PATIENT_METRICS.csv"
raw = subprocess.check_output(["git", "show", f"{REV}:{PATH}"])
rows = list(csv.DictReader(io.StringIO(raw.decode())))
classes = ["RV", "MYO", "LV"]
data = collections.defaultdict(dict)
for row in rows:
    method = row["run"].removeprefix("L8_S42_")
    cm = json.loads(row["confusion_4x5"])
    p = data[method].setdefault(row["patient"], {"dice": {}, "cm": cm})
    assert p["cm"] == cm
    p["dice"][row["class_name"]] = float(row["dice_mean_frames"])
pids = sorted(data["supervised"])
assert len(pids) == 18 and len(rows) == 288
for method in data:
    assert set(data[method]) == set(pids)
    for pid in pids:
        assert [sum(r) for r in data[method][pid]["cm"]] == [sum(r) for r in data["supervised"][pid]["cm"]]


def score(method, pid):
    return statistics.mean(data[method][pid]["dice"][c] for c in classes)


def oracle(methods):
    constant = {m: statistics.mean(score(m, p) for p in pids) for m in methods}
    best = max(methods, key=constant.get)
    # Whole-patient switching preserves one complete multiclass predictor per patient.
    select = {p: max(methods, key=lambda m: score(m, p)) for p in pids}
    value = statistics.mean(score(select[p], p) for p in pids)
    # Independent class-score maxima relax multiclass consistency and are NOT masks.
    relaxation = statistics.mean(max(data[m][p]["dice"][c] for m in methods) for p in pids for c in classes)
    rvselect = {p: max(methods, key=lambda m: data[m][p]["dice"]["RV"]) for p in pids}
    assert constant[best] <= value + 1e-15 <= relaxation + 1e-15
    return {"methods": methods, "best_constant_method": best, "best_constant_foreground_dice": constant[best],
            "ORACLE_retrospective_whole_patient_foreground_dice": value,
            "ORACLE_gain_over_best_constant_pp": 100 * (value - constant[best]),
            "ORACLE_selection_counts": dict(collections.Counter(select.values())),
            "RELAXATION_independent_patient_class_score_maximum": relaxation,
            "RELAXATION_gain_over_best_constant_pp": 100 * (relaxation - constant[best]),
            "ORACLE_RV_only_selection_counts": dict(collections.Counter(rvselect.values())),
            "ORACLE_RV_only_selector_class_dice": {c: statistics.mean(data[rvselect[p]][p]["dice"][c] for p in pids) for c in classes},
            "status": "Retrospective label-informed bound for switching among these final endpoint predictions ONLY; not an admissible learned policy or bound on a new guard."}


def pair(left, right):
    # left is the comparator; right is ema_block. Bounds are stratified by patient and GT class.
    result = {"comparator": left, "reference": right, "class": {}}
    for k, c in enumerate(["BG", "RV", "MYO", "LV"]):
        totals = {m: {"TP": 0, "FP": 0, "FN": 0} for m in [left, right]}
        corrected_lo = corrected_hi = destroyed_lo = destroyed_hi = 0
        for p in pids:
            a, b = data[right][p]["cm"], data[left][p]["cm"]
            n = sum(a[k]); assert n == sum(b[k])
            ta, tb = a[k][k], b[k][k]
            # Intersection of two correct sets has Frechet bounds.
            corrected_lo += max(0, tb - ta)
            corrected_hi += min(tb, n - ta)
            destroyed_lo += max(0, ta - tb)
            destroyed_hi += min(ta, n - tb)
            for m, cm in [(left, b), (right, a)]:
                tp = cm[k][k]
                totals[m]["TP"] += tp
                totals[m]["FN"] += sum(cm[k]) - tp
                totals[m]["FP"] += sum(r[k] for r in cm) - tp
        delta = {f: totals[left][f] - totals[right][f] for f in ["TP", "FP", "FN"]}
        assert corrected_lo - destroyed_lo == delta["TP"]
        assert corrected_hi - destroyed_hi == delta["TP"]
        result["class"][c] = {"comparator_counts": totals[left], "ema_block_counts": totals[right],
                               "comparator_minus_block_counts": delta,
                               "block_wrong_comparator_correct_count_bounds": [corrected_lo, corrected_hi],
                               "block_correct_comparator_wrong_count_bounds": [destroyed_lo, destroyed_hi],
                               "bound_type": "Sharp marginal-overlap bounds conditional on the common patient/GT domains; not confidence intervals."}
        if c in classes:
            vals = [data[left][p]["dice"][c] - data[right][p]["dice"][c] for p in pids]
            result["class"][c]["mean_dice_delta_pp"] = 100 * statistics.mean(vals)
            result["class"][c]["patients_better_worse_equal"] = [sum(v > 0 for v in vals), sum(v < 0 for v in vals), sum(v == 0 for v in vals)]
    result["patient_foreground_better_worse_equal"] = [sum(score(left,p)>score(right,p) for p in pids), sum(score(left,p)<score(right,p) for p in pids), sum(score(left,p)==score(right,p) for p in pids)]
    result["block_RV_lower_MYO_higher_patients"] = sum(data[right][p]["dice"]["RV"] < data[left][p]["dice"]["RV"] and data[right][p]["dice"]["MYO"] > data[left][p]["dice"]["MYO"] for p in pids)
    return result


def bg_rv_correction(comparator):
    lo = hi = 0
    for p in pids:
        a, b = data["ema_block"][p]["cm"][0], data[comparator][p]["cm"][0]
        n = sum(a); assert n == sum(b)
        lo += max(0, a[1] + b[0] - n)
        hi += min(a[1], b[0])
    return {"comparator": comparator, "block_BG_to_RV_and_comparator_BG_count_bounds": [lo, hi],
            "meaning": "Bounds on coincident endpoint predictions only; no pixel coordinates or responsible training updates identified."}


oracles = [oracle(["ema_block", "supervised"]), oracle(["ema_block", "ema_full"]), oracle(["supervised", "supervised_extra", "ema_full", "ema_block"])]
result = {"revision": REV, "input_path": PATH, "input_sha256": hashlib.sha256(raw).hexdigest(),
          "scope": "Local arithmetic on Git-tracked endpoint tables; no patient arrays, inference or training; all oracle calculations are retrospective and forbidden for training/selection on VAL labels.",
          "score_definition": "Mean over18 patients of mean RV/MYO/LV; each class input is stored mean of reference-present3D frame Dice.",
          "oracles": oracles, "paired_tradeoffs": [pair("supervised", "ema_block"), pair("ema_full", "ema_block")],
          "BG_RV_endpoint_overlap_bounds": [bg_rv_correction("supervised"), bg_rv_correction("ema_full")],
          "full_minus_block_supervised_patient_oracle_pp": 100 * (oracles[1]["best_constant_foreground_dice"] - oracles[0]["ORACLE_retrospective_whole_patient_foreground_dice"]),
          "nonidentifiable_from_input": ["Actual pixelwise cross-model overlap", "Any transaction counterfactual", "Audit Brier values or local acceptance", "Pseudo-update responsibility", "Learning a legal selector from TRAIN", "Realized benefit of a retrained joint/scalar guard"]}
out = Path(__file__).resolve().parents[1] / "evidence" / "C_endpoint_headroom.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, indent=2))
