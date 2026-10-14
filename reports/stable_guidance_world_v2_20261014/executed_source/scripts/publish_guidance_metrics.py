#!/usr/bin/env python3
"""Publish final segmentation metrics after the original training owner exits.

Reads frozen evaluator confusion counts only; never opens images, masks, model
weights or checkpoints. No checkpoint/threshold selection is performed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

CLASSES = ("BG", "RV", "MYO", "LV")


def mean_defined(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def evaluation_scalars(report, split="val"):
    """Average frames within patients, patients within classes, then foreground.

    Precision with no predicted positive and recall with no reference positive
    are undefined (None). Missing predictions still yield Dice/IoU/recall zero
    when the reference contains the class. UNKNOWN remains a false negative.
    """
    patients = {}
    for row in report["frames"]:
        per_patient = patients.setdefault(row["patient_id"], {})
        for name in CLASSES:
            counts = row["metrics"]["classes"][name]
            tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
            if any(type(x) is not int or x < 0 for x in (tp, fp, fn)):
                raise ValueError("invalid confusion counts")
            values = dict(dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                          iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                          precision=tp / (tp + fp) if tp + fp else None,
                          recall=tp / (tp + fn) if tp + fn else None)
            per_patient.setdefault(name, []).append(values)
    if not patients:
        raise ValueError("no evaluated patient frames")
    metrics = {f"{split}/patient_count": len(patients), f"{split}/labeled_frames": len(report["frames"])}
    for metric in ("dice", "iou", "precision", "recall"):
        class_means = {}
        for name in CLASSES:
            values = [mean_defined([frame[metric] for frame in patient[name]]) for patient in patients.values()]
            class_means[name] = mean_defined(values)
            metrics[f"{split}/{name}/{metric}"] = class_means[name]
            metrics[f"{split}/{name}/{metric}_defined_patients"] = sum(v is not None for v in values)
        metrics[f"{split}/{metric}"] = mean_defined([class_means[c] for c in CLASSES[1:]])
    expected = report["summary"]["macro_foreground_dice"]
    actual = metrics[f"{split}/dice"]
    if (expected is None) != (actual is None) or (expected is not None and not math.isclose(actual, expected, abs_tol=1e-12)):
        raise ValueError("Dice reduction disagrees with the frozen evaluator")
    metrics[f"{split}/coverage"] = report["summary"]["coverage_patient_macro"]
    return metrics


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv)
    job = args.job.resolve()
    tracking = job / "tracking"
    destination = tracking / "FINAL_METRICS.json"
    state_path = tracking / "METRICS_PUBLISHER.json"
    if state_path.exists() and json.loads(state_path.read_text()).get("status") == "published":
        print("Final metrics already published", flush=True)
        return 0
    state = dict(status="waiting_for_training_exit", started_at=time.time(),
                 checkpoint_selection_changed=False, reads_reference_masks=False)
    write_json(state_path, state)
    try:
        while not (job / "EXIT_CODE").exists():
            if not args.wait:
                raise RuntimeError("original W&B owner has not exited; refusing simultaneous writers")
            time.sleep(15)
        original = json.loads((tracking / "WANDB_RUN.json").read_text())
        if original["status"] != "finished":
            state.update(status="skipped_pipeline_failed", pipeline_status=original.get("pipeline_status"))
            write_json(state_path, state)
            return 1
        output = job / "run"
        lock = json.loads((output / "PREDICTIONS_LOCKED.json").read_text())
        # Use the run's original artifact verifier, without changing its source.
        sys.path.insert(0, str(job / "source/src"))
        from self_audit_guidance.io import verify
        metrics = {}
        for split in ("val", "test"):
            directory = output / f"evaluation_{split}"
            if not directory.exists():
                continue
            manifest = verify(directory, "independent_evaluation")
            if manifest["prediction_manifest_sha256"] != lock["predictions"][split]:
                raise ValueError("evaluation is not bound to the frozen prediction pair")
            report = json.loads((directory / "METRICS.json").read_text())
            metrics.update(evaluation_scalars(report, split))
        if not metrics:
            raise RuntimeError("no final reference evaluation is available")
        recipe = json.loads((output / "RECIPE.json").read_text())
        metrics["evaluation/epoch"] = recipe["epochs"]
        payload = dict(metrics=metrics, aggregation="frame_mean_then_patient_mean_then_foreground_class_mean",
                       undefined_precision_recall="null_excluded_from_macro_with_defined_patient_counts",
                       auroc_status="unavailable: frozen exports contain hard labels only; no transition auditor in this pipeline",
                       checkpoint_selection_changed=False, raw_reference_masks_read=False)
        write_json(destination, payload)
        state.update(status="publishing", original_run_id=original["id"])
        write_json(state_path, state)
        import wandb
        log_dir = tracking / "final_metrics_wandb"
        log_dir.mkdir(exist_ok=True)
        with wandb.init(project=original["project"], entity=original["entity"], id=original["id"],
                        resume="must", mode=original["mode"], dir=str(log_dir)) as run:
            run.define_metric("evaluation/epoch")
            for split in ("val", "test"):
                run.define_metric(f"{split}/*", step_metric="evaluation/epoch")
            run.log({k: v for k, v in metrics.items() if v is not None})
            run.summary.update(metrics)
            run.summary["evaluation/auroc_status"] = payload["auroc_status"]
            run.summary["evaluation/aggregation"] = payload["aggregation"]
            run.summary["evaluation/undefined_precision_recall"] = payload["undefined_precision_recall"]
            run.config.update({k: recipe[k] for k in ("epochs", "batch_size", "width", "lr", "seed", "checkpoint_selection")}, allow_val_change=True)
        state.update(status="published", finished_at=time.time(), metrics_file=str(destination))
        write_json(state_path, state)
        print(json.dumps(state), flush=True)
        return 0
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc), failed_at=time.time())
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
