#!/usr/bin/env python3
"""Audit Git-tracked aggregates only; no patient files, GPU, or weight loading."""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess

REV = "8960b6baac4c6600553d900c303f192c2fcfa4f0"
OUT = Path(__file__).resolve().parents[1] / "evidence" / "C_recomputation.json"


def blob(path, rev=REV):
    return subprocess.check_output(["git", "show", f"{rev}:{path}"])


def table(path):
    return list(csv.DictReader(io.StringIO(blob(path).decode())))


def source_digest(rev):
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", rev, "self_audit"], text=True
    ).splitlines()
    paths = sorted(p for p in paths if p.count("/") == 1 and p.endswith(".py")) + ["train.py"]
    material = "".join(f"{Path(p).name}:{hashlib.sha256(blob(p, rev)).hexdigest()}\n" for p in paths)
    return hashlib.sha256(material.encode()).hexdigest()


rows = table("results/PATIENT_METRICS.csv")
main = table("results/RESULTS_MAIN.csv")
ablation = {r["method"]: r for r in table("results/RESULTS_ABLATION.csv")}
completed = json.loads(blob("results/logs/COMPLETED_SUMMARY.json"))["completed_methods"]
split = json.loads(blob("splits/acdc_reviewed_locked_train80_val18.json"))
checks = collections.Counter()
by_method = collections.defaultdict(dict)
for r in rows:
    method = r["run"].removeprefix("L8_S42_")
    cls = r["class_name"]
    k = ["BG", "RV", "MYO", "LV"].index(cls)
    cm = json.loads(r["confusion_4x5"])
    assert len(cm) == 4 and all(len(x) == 5 for x in cm)
    tp = cm[k][k]
    gt = sum(cm[k])
    pred = sum(x[k] for x in cm)
    expected = {"TP": tp, "FP": pred - tp, "FN": gt - tp,
                "bg_to_myo_fp": cm[0][2], "iou_pooled_frames": tp / (gt + pred - tp),
                "precision": tp / pred if pred else None, "recall": tp / gt if gt else None}
    for field, value in expected.items():
        assert value is not None and abs(float(r[field]) - value) < 1e-12, (method, r["patient"], field)
        checks["confusion_derived_fields"] += 1
    assert r["reference_present"] == str(bool(gt)) and not r["hd95_mm"]
    checks["presence_and_missing_hd95"] += 2
    assert r["patient"] in split["val"]
    patient = by_method[method].setdefault(r["patient"], {"confusion": cm, "scores": {}})
    assert patient["confusion"] == cm and cls not in patient["scores"]
    patient["scores"][cls] = float(r["dice_mean_frames"])

summary = {}
for r in completed:
    method = r["method"]
    patients = by_method[method]
    assert set(patients) == set(split["val"])
    assert len(patients) == 18
    mainrow = next(x for x in main if x["method"] == method)
    macro = {c: statistics.mean(p["scores"][c] for p in patients.values()) for c in ["BG", "RV", "MYO", "LV"]}
    for c, v in macro.items():
        assert abs(v - r["macro_patient"][c]["dice"]) < 1e-12
        checks["patient_dice_to_summary"] += 1
        if c != "BG":
            assert abs(v - float(mainrow[c + "_dice_mean"])) < 1e-12
            assert mainrow[c + "_dice_std"] == ""
            checks["patient_dice_to_main"] += 1
    cm = [[sum(p["confusion"][i][j] for p in patients.values()) for j in range(5)] for i in range(4)]
    pooled = {}
    for k, c in enumerate(["BG", "RV", "MYO", "LV"]):
        tp, gt, pred = cm[k][k], sum(cm[k]), sum(x[k] for x in cm)
        pooled[c] = {"TP": tp, "FP": pred - tp, "FN": gt - tp, "dice": 2 * tp / (gt + pred)}
    assert abs(r["elapsed_seconds"] / 3600 - float(ablation[method]["wall_gpu_hours"])) < 1e-12
    checks["wall_hours"] += 1
    summary[method] = {"macro_patient_dice": macro,
                       "macro_foreground_dice": statistics.mean(macro[c] for c in ["RV", "MYO", "LV"]),
                       "pooled_confusion": cm, "pooled_class_metrics": pooled,
                       "bg_to_myo_fp": cm[0][2], "wall_hours": r["elapsed_seconds"] / 3600,
                       "peak_vram_GiB": int(ablation[method]["peak_vram_bytes"]) / 2**30,
                       "pseudo_accepted": r["pseudo_accepted"]}

comparisons = {}
for left, right in [("ema_full", "supervised"), ("ema_block", "supervised"),
                    ("ema_full", "ema_block"), ("supervised_extra", "supervised")]:
    d = {}
    for c in ["RV", "MYO", "LV"]:
        vals = [100 * (by_method[left][p]["scores"][c] - by_method[right][p]["scores"][c]) for p in split["val"]]
        d[c] = {"mean_delta_percentage_points": statistics.mean(vals), "patients_better": sum(v > 0 for v in vals),
                "patients_worse": sum(v < 0 for v in vals), "min_delta": min(vals), "max_delta": max(vals)}
    for p in split["val"]:
        assert [sum(x) for x in by_method[left][p]["confusion"]] == [sum(x) for x in by_method[right][p]["confusion"]]
        checks["paired_reference_marginals"] += 1
    d["foreground_macro_delta_percentage_points"] = 100 * (summary[left]["macro_foreground_dice"] - summary[right]["macro_foreground_dice"])
    d["wall_time_ratio"] = summary[left]["wall_hours"] / summary[right]["wall_hours"]
    comparisons[f"{left}_minus_{right}"] = d

source = {rev: source_digest(rev) for rev in ["637f38ced217fe1db2347a87924695533056d8f0", "970f142a76f417c516db1c9f18f60a1dfb38d342", "7940ec2", REV]}
assert source[REV] == completed[0]["source_sha256"]
assert all(r["source_sha256"] == source[REV] for r in completed)
checks["executed_source_receipt_matches_git_tree"] += len(completed)
tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", REV], text=True).splitlines()
status = collections.Counter(r["status"] for r in table("results/EXPERIMENT_STATUS.csv"))
assert len(rows) == 288 and status == {"SCORED_EXPOSED_VAL": 4, "NOT_RUN_USER_STOP": 77}
result = {"scope": "Local arithmetic and source-hash checks over Git-tracked aggregates; no raw prediction/reference replay or GPU verification",
          "revision": REV, "status_counts": status, "row_count": len(rows), "checks_passed": checks,
          "summary": summary, "paired_descriptive_comparisons": comparisons, "source_digests": source,
          "tracked_payloads": [p for p in tracked if p.endswith((".pt", ".pth", ".npy", ".npz", ".jsonl"))],
          "limitations": ["Mean-frame patient Dice is stored, not reconstructable from pooled-frame confusion alone.",
                          "Patient means can be aggregated and paired; per-frame Dice needs untracked METRICS.json/raw frame counts.",
                          "Confusion counts and reference marginals are checked for internal consistency, not truth against patient files.",
                          "Source digest covers self_audit/*.py plus train.py, not the entire runtime/dependencies/data.",
                          "No confidence intervals or significance claim; comparisons concern one exposed-development run per method."]}
OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"artifact": str(OUT), "checks_passed": checks, "source_digests": source,
                  "summary": summary, "paired_descriptive_comparisons": comparisons, "tracked_payloads": result["tracked_payloads"]}, indent=2))
