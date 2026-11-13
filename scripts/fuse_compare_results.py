#!/usr/bin/env python3
"""Compare locked model metrics using patient as the independent unit.

Input is a JSON document with ``splits`` mapping split names to ``models``;
each model maps patient IDs to metric mappings, e.g. ``{"fg_dice": .9}`` or
``{"per_class": {"RV": {"dice": .9, "hd95_mm": 2.}}}``. No labels or
voxel arrays are read. Paired inference is restricted to the shared patient IDs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon


def _metric_value(record: Mapping[str, Any], metric: str) -> float:
    if metric == "FG":
        value = record.get("fg_dice")
    else:
        cls, field = metric.split(".", 1)
        value = record.get("per_class", {}).get(cls, {}).get(field)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"missing/nonfinite metric {metric!r} in patient record")
    return float(value)


def paired_statistics(
    first: Mapping[str, float], second: Mapping[str, float], *,
    bootstrap_resamples: int = 10_000, seed: int = 42,
) -> dict[str, Any]:
    """Return paired first-minus-second inference over matched patients."""
    if not isinstance(bootstrap_resamples, int) or isinstance(bootstrap_resamples, bool) or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if set(first) != set(second):
        missing_second = sorted(set(first) - set(second))
        missing_first = sorted(set(second) - set(first))
        raise ValueError(f"patient ID sets differ; missing from second={missing_second}, missing from first={missing_first}")
    ids = sorted(set(first) & set(second))
    if len(ids) < 2:
        raise ValueError("at least two paired patients are required")
    a = np.asarray([first[i] for i in ids], dtype=np.float64)
    b = np.asarray([second[i] for i in ids], dtype=np.float64)
    delta = a - b
    if not np.isfinite(delta).all():
        raise ValueError("paired values must be finite")
    nonzero = delta[delta != 0]
    if not len(nonzero):
        statistic, p_value = 0.0, 1.0
    else:
        # Force asymptotic for larger cohorts to avoid exact-test work scaling
        # steeply; for small samples scipy chooses exact when applicable.
        test = wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="auto")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(bootstrap_resamples, dtype=np.float64)
    # Chunk to bound memory while keeping a deterministic iid patient bootstrap.
    chunk = 1024
    for start in range(0, bootstrap_resamples, chunk):
        stop = min(start + chunk, bootstrap_resamples)
        sample_indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        boot_means[start:stop] = delta[sample_indices].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return {
        "n_paired_patients": len(ids), "patient_ids": ids,
        "mean_difference": float(delta.mean()),
        "median_difference": float(np.median(delta)),
        "wilcoxon_two_sided": {"statistic": statistic, "p_value": p_value},
        "patient_bootstrap_mean_difference_95ci": [float(low), float(high)],
        "bootstrap_resamples": bootstrap_resamples, "bootstrap_seed": seed,
    }


def analyze_split(split: Mapping[str, Any], *, models: Sequence[str],
                  metrics: Sequence[str], bootstrap_resamples: int, seed: int) -> dict[str, Any]:
    model_rows = split["models"]
    output: dict[str, Any] = {"models": {}, "paired": {}}
    names = list(models)
    for name in names:
        if name not in model_rows:
            continue
        rows = model_rows[name]
        output["models"][name] = {}
        for metric in metrics:
            values = [_metric_value(row, metric) for row in rows.values()]
            output["models"][name][metric] = {
                "n_patients": len(values), "mean_patient_metric": float(np.mean(values)),
                "median_patient_metric": float(np.median(values)),
            }
    if len(names) >= 3 and all(name in model_rows for name in names[:3]):
        # Conventional FUSE inputs use teacher/student/fusion as the first three.
        teacher, student, fusion = names[:3]
        for metric in metrics:
            vals = {name: {pid: _metric_value(row, metric) for pid, row in model_rows[name].items()}
                    for name in (teacher, student, fusion)}
            output["paired"][metric] = {
                f"{fusion}_minus_{teacher}": paired_statistics(
                    vals[fusion], vals[teacher], bootstrap_resamples=bootstrap_resamples, seed=seed),
                f"{fusion}_minus_{student}": paired_statistics(
                    vals[fusion], vals[student], bootstrap_resamples=bootstrap_resamples, seed=seed),
            }
    return output


def build_input(result: Mapping[str, Any], transaction: Mapping[str, Any]) -> dict[str, Any]:
    """Convert FUSE receipt schema into portable patient-metric input."""
    splits: dict[str, Any] = {}
    for name, val in result["validation"].items():
        splits.setdefault("VAL18", {"models": {}})["models"][name] = val["per_patient"]
    for name, test in transaction["metrics"].items():
        splits.setdefault("TEST50", {"models": {}})["models"][name] = test["per_patient"]
    return {"schema": "self_audit.paired_model_metrics.v1", "splits": splits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="portable paired-metrics JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["teacher", "student", "fusion"])
    parser.add_argument("--metrics", nargs="+", default=["FG", "RV.dice", "MYO.dice", "LV.dice",
                                                            "RV.hd95_mm", "MYO.hd95_mm", "LV.hd95_mm"])
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opened-test", action="store_true",
                        help="mark outputs as exploratory because TEST outcomes were opened")
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    splits = {}
    for split_name, split in source["splits"].items():
        splits[split_name] = analyze_split(split, models=args.models, metrics=args.metrics,
                                           bootstrap_resamples=args.bootstrap_resamples, seed=args.seed)
    output = {
        "schema": "self_audit.paired_comparison_analysis.v1",
        "analysis_status": "EXPLORATORY_POST_TEST_OPEN" if args.opened_test else "EXPLORATORY",
        "independent_unit": "patient", "pairing": "same patient IDs across compared models",
        "statistical_methods": {"wilcoxon": "paired two-sided signed-rank, zero differences discarded",
                                "bootstrap": "paired patient resampling; mean difference; percentile 95% CI"},
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
