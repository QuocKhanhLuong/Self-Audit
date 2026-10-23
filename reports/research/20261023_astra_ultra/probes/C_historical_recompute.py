#!/usr/bin/env python3
"""Recompute archived full50 frame Dice from tracked confusion counts only."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

REV = "924265f7df604d3dca84dda49578b1a0cf28d821"
PATH = "reports/dinov3_full50_val_20261019/METRICS.json"
raw = subprocess.check_output(["git", "show", f"{REV}:{PATH}"])
data = json.loads(raw)
frames = data["frames"]
assert len(frames) == 36 and len({r["patient"] for r in frames}) == 18
checks = collections.Counter()
aggregates = {kind: [[0] * 5 for _ in range(4)] for kind in ["raw", "gated"]}
scores = collections.defaultdict(list)
for row in frames:
    for kind in ["raw", "gated"]:
        cm = row[kind + "_confusion"]
        for k, name in enumerate(["BG", "RV", "MYO", "LV"]):
            gt, pred, tp = sum(cm[k]), sum(v[k] for v in cm), cm[k][k]
            value = 2 * tp / (gt + pred) if gt else None
            assert value == row["metrics"][name][kind + "_dice"]
            checks["frame_dice_from_counts"] += 1
            if value is not None:
                scores[(kind, name)].append(value)
            for j in range(5):
                aggregates[kind][k][j] += cm[k][j]
    cm = row["gated_confusion"]
    coverage = sum(sum(v[:4]) for v in cm) / sum(sum(v) for v in cm)
    fgcoverage = sum(sum(v[:4]) for v in cm[1:]) / sum(sum(v) for v in cm[1:])
    assert coverage == row["coverage"] == 0 and fgcoverage == row["foreground_coverage"] == 0
    checks["coverage_from_counts"] += 2
for kind in ["raw", "gated"]:
    assert aggregates[kind] == data["confusion"][kind]
    checks["pooled_confusion"] += 1
    for name in ["BG", "RV", "MYO", "LV"]:
        assert abs(statistics.mean(scores[(kind, name)]) - data["macro_per_frame"][name][kind + "_dice"]) < 1e-12
        checks["macro_dice"] += 1
result = {"scope": "Local arithmetic from archived Git-tracked frame confusion, not image/prediction replay",
          "revision": REV, "metrics_path": PATH, "metrics_sha256": hashlib.sha256(raw).hexdigest(),
          "checks_passed": checks, "patients": 18, "frames": 36,
          "macro_raw_dice": {n: statistics.mean(scores[("raw", n)]) for n in ["BG", "RV", "MYO", "LV"]},
          "macro_foreground_raw_dice": statistics.mean(statistics.mean(scores[("raw", n)]) for n in ["RV", "MYO", "LV"]),
          "gated_dice_all_classes": 0, "coverage": 0, "foreground_coverage": 0,
          "AUROC_AP": "Reported only; cannot reconstruct pixel rankings from confusion counts",
          "protocol_comparability_to_v2": "Not matched: historical frozen-Q supervision, architecture, objective, optimizer, budget, preprocessing, hardware and schedule differ"}
out = Path(__file__).resolve().parents[1] / "evidence" / "C_historical_recomputation.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
