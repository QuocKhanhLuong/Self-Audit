#!/usr/bin/env python3
"""Independent post-freeze evaluator. Never imported by method/training.

GT matching below is diagnostic only. It never rewrites model predictions, chooses
checkpoints, calibrates thresholds, or provides any training gradient.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.optimize import linear_sum_assignment


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(pred, truth, support=None):
    m = np.ones(truth.shape, bool) if support is None else support
    return np.array([[((pred == k) & (truth == k) & m).sum(),
                      ((pred == k) & m).sum(), ((truth == k) & m).sum()]
                     for k in range(4)], np.int64)


def dice(c):
    denominator = c[:, 1] + c[:, 2]
    return np.divide(2 * c[:, 0], denominator, out=np.full(4, np.nan), where=denominator > 0)


def adjusted_rand(confusion):
    """ARI from the contingency table; no model or extra ML dependency."""
    c = np.asarray(confusion, dtype=np.float64)
    pairs = lambda a: (a * (a - 1) / 2).sum()
    total = c.sum() * (c.sum() - 1) / 2
    if total == 0:
        return 1.0
    index = pairs(c)
    row_pairs, col_pairs = pairs(c.sum(1)), pairs(c.sum(0))
    expected = row_pairs * col_pairs / total
    denominator = .5 * (row_pairs + col_pairs) - expected
    return 1.0 if denominator == 0 else float((index - expected) / denominator)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads((args.run / "FROZEN.json").read_text())
    # Validate EVERY prediction before opening the first reference.
    for relative, digest in frozen["predictions"].items():
        if file_hash(args.run / relative) != digest:
            raise RuntimeError("Prediction changed after freeze: " + relative)
    if frozen["checkpoint_sha256"] and file_hash(args.run / "final.pt") != frozen["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed after freeze")
    totals, conditional, cov = {}, {}, {}
    volume_rows, read_log = [], []
    for relative in sorted(frozen["predictions"]):
        path = args.run / relative
        volume = path.stem
        patient = volume.split("_")[0]
        step = path.parent.name
        key = step + ":" + patient
        reference = args.references / patient / (volume + "_gt.nii")
        if not reference.exists():
            reference = reference.with_suffix(".nii.gz")
        ref = nib.load(str(reference))
        truth = np.asarray(ref.dataobj).astype(np.uint8)
        data = np.load(path)
        pred, partition = data["named"], data["partition"]
        if pred.shape != truth.shape or not np.allclose(data["affine"], ref.affine, atol=1e-5):
            raise RuntimeError("Native geometry mismatch: " + volume)
        if not set(np.unique(truth)) <= {0, 1, 2, 3}:
            raise RuntimeError("Unexpected reference class order")
        read_log.append(str(reference))
        c = counts(pred, truth)
        cc = counts(pred, truth, pred != 255)
        coverage = np.array([[(truth == k).sum(), ((truth == k) & (pred != 255)).sum()]
                             for k in range(4)], np.int64)
        totals[key] = totals.get(key, np.zeros((4, 3), np.int64)) + c
        conditional[key] = conditional.get(key, np.zeros((4, 3), np.int64)) + cc
        cov[key] = cov.get(key, np.zeros((4, 2), np.int64)) + coverage
        # Oracle alignment quantifies partition information; NEVER output a relabeled mask.
        confusion = np.bincount(partition.ravel().astype(int) * 4 + truth.ravel(), minlength=16).reshape(4, 4)
        a, b = linear_sum_assignment(-confusion)
        oracle_accuracy = float(confusion[a, b].sum() / truth.size)
        per_pair_dice = 2 * confusion / np.maximum(confusion.sum(1)[:, None] + confusion.sum(0)[None], 1)
        ad, bd = linear_sum_assignment(-per_pair_dice)
        volume_rows.append({"step": step, "patient": patient, "volume": volume,
                            "named_dice": dice(c), "coverage": float((pred != 255).mean()),
                            "partition_ari": adjusted_rand(confusion),
                            "oracle_aligned_accuracy_ANALYSIS_ONLY": oracle_accuracy,
                            "oracle_macro4_dice_ANALYSIS_ONLY": float(per_pair_dice[ad, bd].mean())})
    patients = []
    for key in sorted(totals):
        step, patient = key.split(":")
        d = dice(totals[key])
        v = cov[key]
        patients.append({"step": step, "patient": patient, "class_dice": d,
                         "foreground_mean_dice": float(np.nanmean(d[1:])),
                         "conditional_dice_known_pixels_ONLY": dice(conditional[key]),
                         "class_coverage": np.divide(v[:, 1], v[:, 0], out=np.full(4, np.nan), where=v[:, 0] > 0)})
    summaries = []
    for step in sorted({p["step"] for p in patients}):
        rows = [p for p in patients if p["step"] == step]
        volumes = [p for p in volume_rows if p["step"] == step]
        summaries.append({"step": step, "patients": len(rows),
                          "foreground_mean_dice": float(np.mean([p["foreground_mean_dice"] for p in rows])),
                          "class_dice": np.nanmean([p["class_dice"] for p in rows], axis=0),
                          "class_coverage": np.nanmean([p["class_coverage"] for p in rows], axis=0),
                          "partition_ari_volume_mean": float(np.mean([p["partition_ari"] for p in volumes])),
                          "oracle_macro4_dice_volume_mean_ANALYSIS_ONLY": float(np.mean([p["oracle_macro4_dice_ANALYSIS_ONLY"] for p in volumes]))})
    result = {"evaluated_unix": time.time(), "frozen_manifest_sha256": file_hash(args.run / "FROZEN.json"),
              "prediction_hashes_verified_before_first_reference": True,
              "gt_read_only_evaluator": True, "gt_mapping_exported": False,
              "checkpoint_selected_using_gt": False, "unit_of_independence": "patient",
              "set_status": "development, NOT untouched test", "reference_reads": sorted(set(read_log)),
              "summary": summaries, "patients": patients, "volumes": volume_rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(clean(result), indent=2, allow_nan=False))
    print(json.dumps(clean(summaries), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
