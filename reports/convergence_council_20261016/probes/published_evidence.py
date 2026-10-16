#!/usr/bin/env python3
"""Recheck published text hashes and count arithmetic; never read binary data."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COUNCIL = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "reports/stable_guidance_main513ddda_b12_20261016"
EXPECTED_HEAD = "92a60744efef85558d7c0226c84b7c49d47217ec"
EXPECTED_SOURCE = "513ddda8cec3a166f13cfe68fe6379cab014a1a7"
CLASSES = ("BG", "RV", "MYO", "LV")


def read_json(relative):
    return json.loads((PACKAGE / relative).read_text())


def mean_defined(values):
    defined = [value for value in values if value is not None]
    return statistics.mean(defined) if defined else None


def git_bytes(*args):
    return subprocess.check_output(["rtk", "proxy", "git", *args], cwd=ROOT)


def main():
    issues = []

    def require(ok, label):
        if not ok:
            issues.append(label)

    def close(a, b):
        if a is None or b is None:
            return a is b
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)

    head = git_bytes("rev-parse", "HEAD").decode().strip()
    # Delivery commits may add reports after the evidence commit. Pin the input
    # manifest itself rather than requiring HEAD to remain its parent forever.
    pinned_manifest = git_bytes("show", EXPECTED_HEAD + ":" + str((PACKAGE / "MANIFEST.json").relative_to(ROOT)))
    require((PACKAGE / "MANIFEST.json").read_bytes() == pinned_manifest, "publication manifest differs from pinned evidence")
    manifest = read_json("MANIFEST.json")
    require(manifest["source_commit"] == EXPECTED_SOURCE, "unexpected source SHA")
    source_count = source_matches = 0
    for entry in manifest["files"]:
        relative = entry["published_path"]
        data = (PACKAGE / relative).read_bytes()
        require(hashlib.sha256(data).hexdigest() == entry["published_sha256"], "hash: " + relative)
        require(len(data) == entry["published_bytes"], "size: " + relative)
        if "/source/" in relative:
            source_count += 1
            original_path = relative.split("/source/", 1)[1]
            original = git_bytes("show", EXPECTED_SOURCE + ":" + original_path)
            source_matches += int(original == data)
            require(original == data, "source identity: " + original_path)
    require(len(manifest["files"]) == 118, "unexpected manifest file count")
    require(source_count == 24, "unexpected archived source file count")

    report = read_json("completed_run/run/evaluation_val_verified_grid/METRICS.json")
    published = read_json("summary.json")
    frames = report["frames"]
    patients = sorted({frame["patient_id"] for frame in frames})
    require(len(frames) == 36 and len(patients) == 18, "unexpected evaluation cohort")
    class_results = {}
    for name in CLASSES:
        totals = {key: sum(frame["metrics"]["classes"][name][key] for frame in frames)
                  for key in ("tp", "fp", "fn", "predicted_pixels", "reference_pixels")}
        result = {"pooled_counts": totals}
        for metric in ("dice", "iou", "precision", "recall"):
            per_patient = []
            for patient in patients:
                values = []
                for frame in frames:
                    if frame["patient_id"] != patient:
                        continue
                    row = frame["metrics"]["classes"][name]
                    tp, fp, fn = (row[key] for key in ("tp", "fp", "fn"))
                    require(all(type(v) is int and v >= 0 for v in (tp, fp, fn)), "invalid saved count")
                    require(tp + fp == row["predicted_pixels"], "prediction count identity")
                    require(tp + fn == row["reference_pixels"], "reference count identity")
                    denominator = {"dice": 2 * tp + fp + fn, "iou": tp + fp + fn,
                                   "precision": tp + fp, "recall": tp + fn}[metric]
                    value = ((2 * tp if metric == "dice" else tp) / denominator
                             if denominator else None)
                    if metric in ("dice", "iou"):
                        require(close(value, row[metric]), "saved frame " + metric)
                    values.append(value)
                per_patient.append(mean_defined(values))
            macro = mean_defined(per_patient)
            defined = sum(v is not None for v in per_patient)
            require(close(macro, published["metrics"][f"val/{name}/{metric}"]), "macro " + name + metric)
            require(defined == published["metrics"][f"val/{name}/{metric}_defined_patients"], "defined count")
            result[metric] = {"macro": macro, "defined_patients": defined}
        class_results[name] = result
    for metric in ("dice", "iou", "precision", "recall"):
        require(close(mean_defined([class_results[c][metric]["macro"] for c in CLASSES[1:]]),
                      published["metrics"]["val/" + metric]), "foreground macro " + metric)
    macro_coverage = mean_defined([
        mean_defined([frame["metrics"]["coverage"] for frame in frames if frame["patient_id"] == pid])
        for pid in patients])
    total_pixels = sum(frame["metrics"]["pixels"] for frame in frames)
    unknown_pixels = sum(frame["metrics"]["unknown_pixels"] for frame in frames)
    for frame in frames:
        metrics = frame["metrics"]
        require(close(1 - metrics["unknown_pixels"] / metrics["pixels"], metrics["coverage"]), "frame coverage")
        require(sum(metrics["classes"][c]["predicted_pixels"] for c in CLASSES)
                + metrics["unknown_pixels"] == metrics["pixels"], "frame label partition")
    require(close(macro_coverage, published["metrics"]["val/coverage"]), "macro coverage")
    require(close(1 - macro_coverage, published["metrics"]["val/unknown_fraction"]), "unknown macro")

    entropy = read_json("completed_run/analysis/semantic_convergence/REPORT.json")
    history = read_json("completed_run/run/student/training.json")
    with (PACKAGE / "completed_run/analysis/semantic_convergence/epochs.csv").open() as stream:
        epochs = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    require(len(epochs) == len(history) == 50, "unexpected epoch count")
    for row, logged in zip(epochs, history):
        for key in ("epoch", "steps", "feature_loss"):
            require(close(row[key], logged[key]), "epoch history " + key)
        require(close(row["semantic_ce"], logged["semantic_loss"]), "CE history")
        require(close(row["online_kl_estimate"], row["semantic_ce"] - row["target_entropy_floor"]), "CE-H identity")
    steps = int(sum(row["steps"] for row in epochs))
    active = int(sum(row["active_steps"] for row in epochs))
    require(steps == entropy["optimizer_steps"] == 85700, "optimizer steps")
    require(active == entropy["active_semantic_steps"] == 79221, "active steps")
    for key, a, b in (("epochs_1_10", 0, 10), ("epochs_31_40", 30, 40), ("epochs_41_50", 40, 50)):
        for metric, expected in entropy["means"][key].items():
            require(close(statistics.mean(row[metric] for row in epochs[a:b]), expected), "epoch window " + key + metric)
    for metric, expected in entropy["relative_reductions_percent"].items():
        require(close(100 * (1 - epochs[-1][metric] / epochs[0][metric]), expected), "relative reduction " + metric)
    config = read_json("completed_run/source/configs/stable_guidance_experimental.json")
    probability_margin_lower_bound = 2 * config["min_probability"] - 1
    receipt = {
        "status": "PASS_PUBLISHED_TEXT_ONLY" if not issues else "FAIL",
        "scope": "Local hashes, pinned Git source bytes, saved-count and saved-epoch arithmetic only. No tensors, patient images, checkpoints, GT masks, training, GPU, or network read.",
        "audit_head": head, "executed_source_commit": EXPECTED_SOURCE,
        "manifest_files": len(manifest["files"]), "archived_source_files": source_count,
        "exact_source_matches": source_matches, "issues": issues,
        "class_metrics": class_results,
        "evaluation": {"frames": len(frames), "patients": len(patients), "macro_coverage": macro_coverage,
                       "pooled_coverage": 1 - unknown_pixels / total_pixels,
                       "pixels": total_pixels, "unknown_pixels": unknown_pixels,
                       "verified_grid_frames_reported": report["verified_frame_geometry_count"]},
        "training": {"epochs": len(epochs), "steps": steps, "active_steps": active,
                     "active_fraction": active / steps,
                     "first_active_step_kl": epochs[0]["online_kl_estimate"] * epochs[0]["steps"] / epochs[0]["active_steps"],
                     "last_active_step_kl": epochs[-1]["online_kl_estimate"] * epochs[-1]["steps"] / epochs[-1]["active_steps"],
                     "late_window_kl_reduction_percent": 100 * (1 - entropy["means"]["epochs_41_50"]["online_kl_estimate"] / entropy["means"]["epochs_31_40"]["online_kl_estimate"]),
                     "epoch50_entropy_over_ce": epochs[-1]["target_entropy_floor"] / epochs[-1]["semantic_ce"],
                     "epoch50_entropy_over_uniform_ce": epochs[-1]["target_entropy_floor"] / epochs[-1]["uniform_prediction_ce"]},
        "gate_algebra": {"probability_threshold": config["min_probability"], "margin_threshold": config["min_margin"],
                         "implied_margin_lower_bound": probability_margin_lower_bound,
                         "margin_gate_redundant_for_probability_simplex": probability_margin_lower_bound >= config["min_margin"]},
        "not_reproduced": ["80 original bank binary SHA checks and entropy/RNG replay",
                           "original checkpoint or native prediction authenticity",
                           "14 frame/cine voxel identities", "GPU training or model convergence"],
    }
    output = COUNCIL / "checks/published_evidence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "manifest_files": len(manifest["files"]),
                      "exact_source_matches": source_matches, "issues": issues,
                      "output": str(output.relative_to(ROOT))}, indent=2))
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
