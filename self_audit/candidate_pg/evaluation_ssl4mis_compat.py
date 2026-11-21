"""SSL4MIS-compatible 3D evaluation (the metric computation of HiLab-git/SSL4MIS ``code/test_2D_fully.py``).

SSL4MIS (commit 06df6047): for every test volume, per class c in {RV=1, MYO=2, LV=3}
``medpy.metric.binary.dc / hd95 / asd`` on the boolean masks ``prediction == c`` and ``label == c`` of the assembled
``(slice, H, W)`` volume, with **no voxelspacing** (HD95 and ASD are therefore in voxels), averaged over the volumes of
``test.list`` per class and then over the three classes.  The upstream script does not handle empty masks (medpy raises).
``empty_policy="zero"`` is this repo's documented choice: a class with an empty prediction or an empty label contributes
Dice 0 (medpy's own value), HD95 0 and ASD 0, and every such case is counted in ``empty_cases``.

Predictions here come from our 3D student on the native grid; nothing is resampled.  ``voxelspacing`` can be passed to
obtain the same statistics in mm (not the SSL4MIS protocol; used for unit tests and cross-checks only).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from medpy import metric

CLASSES = {1: "RV", 2: "MYO", 3: "LV"}
HD95_UNIT = "voxels (SSL4MIS protocol: medpy hd95, no voxelspacing)"


def class_metrics(pred: np.ndarray, gt: np.ndarray, *, voxelspacing: Sequence[float] | None = None,
                  empty_policy: str = "zero") -> tuple[float, float, float, bool]:
    """``(dice, hd95, asd, was_empty)`` for two boolean masks; ``was_empty`` flags an empty prediction or label."""

    pred, gt = pred.astype(bool), gt.astype(bool)
    if empty_policy != "zero":
        raise ValueError("only empty_policy='zero' is implemented")
    if not pred.any() or not gt.any():
        dice = float(metric.binary.dc(pred, gt)) if pred.any() or gt.any() else 0.0
        return dice, 0.0, 0.0, True
    kwargs = {} if voxelspacing is None else {"voxelspacing": tuple(float(v) for v in voxelspacing)}
    return float(metric.binary.dc(pred, gt)), float(metric.binary.hd95(pred, gt, **kwargs)), float(metric.binary.asd(pred, gt, **kwargs)), False


def jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    """medpy Jaccard; 0.0 when both masks are empty (medpy would divide by zero)."""

    pred, gt = pred.astype(bool), gt.astype(bool)
    return float(metric.binary.jc(pred, gt)) if pred.any() or gt.any() else 0.0


def evaluate_volumes(records: Sequence[Mapping[str, Any]], *, voxelspacing_key: str | None = None) -> dict[str, Any]:
    """SSL4MIS-protocol metrics over ``records`` with ``prediction``/``target`` ``[D,H,W]`` integer arrays."""

    per_volume, empty = [], 0
    for rec in records:
        pred, gt = np.asarray(rec["prediction"]), np.asarray(rec["target"])
        if pred.shape != gt.shape:
            raise ValueError("prediction and target must share the native grid")
        spacing = None
        if voxelspacing_key is not None:
            spacing = rec["geometry"][voxelspacing_key]
        row = {"patient": str(rec.get("patient", "unknown")), "frame": int(rec.get("frame", 0))}
        for cls, name in CLASSES.items():
            dice, hd95, asd, was_empty = class_metrics(pred == cls, gt == cls, voxelspacing=spacing)
            row[name] = {"dice": dice, "jaccard": jaccard(pred == cls, gt == cls), "hd95": hd95, "asd": asd, "empty": was_empty}
            empty += int(was_empty)
        per_volume.append(row)
    classes = {name: {key: float(np.mean([r[name][key] for r in per_volume])) for key in ("dice", "jaccard", "hd95", "asd")} for name in CLASSES.values()}
    mean = {key: float(np.mean([classes[name][key] for name in classes])) for key in ("dice", "jaccard", "hd95", "asd")}
    return {"protocol": "SSL4MIS test_2D_fully.py metric computation", "hd95_unit": HD95_UNIT if voxelspacing_key is None else "mm",
            "volumes": len(per_volume), "per_class": classes, "mean": mean, "empty_cases": empty,
            "empty_policy": "zero (Dice per medpy, HD95=0, ASD=0; counted in empty_cases)", "per_volume": per_volume}


__all__ = ["CLASSES", "HD95_UNIT", "class_metrics", "evaluate_volumes", "jaccard"]
