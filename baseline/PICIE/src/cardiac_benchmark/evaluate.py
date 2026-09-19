"""Evaluation wrapper for PICIE cardiac benchmark.

Hungarian matching from anonymous clusters to GT classes.
3D volume reconstruction: group slices by patient_id → sort slice_index → compute Dice/HD95/ASSD.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import sys

def _add_picie_src_to_path() -> None:
    picie_src = str(Path(__file__).resolve().parents[2])
    if picie_src not in sys.path:
        sys.path.insert(0, picie_src)

from .provenance import write_json, environment_identity

N_CLASSES = 4
CLASS_NAMES = ["Background", "RV", "MYO", "LV"]
FOREGROUND_CLASSES = [1, 2, 3]


def hungarian_match(pred: np.ndarray, gt: np.ndarray, n_pred_clusters: int, n_gt_classes: int = N_CLASSES) -> dict[int, int]:
    """Hungarian matching from predicted clusters to GT classes."""
    confusion = np.zeros((n_pred_clusters, n_gt_classes), dtype=np.int64)
    for pc in range(n_pred_clusters):
        for gc in range(n_gt_classes):
            confusion[pc, gc] = np.sum((pred == pc) & (gt == gc))
    row_ind, col_ind = linear_sum_assignment(confusion, maximize=True)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[int(r)] = int(c)
    return mapping


def apply_mapping(partition: np.ndarray, mapping: dict[int, int], default: int = 0) -> np.ndarray:
    result = np.full_like(partition, default)
    for src, dst in mapping.items():
        result[partition == src] = dst
    return result


class ClusterProbeEvaluator:
    """Hungarian matching evaluation.

    WARNING: Metrics computed after Hungarian matching — GT used for assignment only.
    """

    def evaluate(
        self,
        partitions: list[np.ndarray],
        labels: list[np.ndarray],
        provenances: list[dict[str, Any]],
    ) -> dict[str, Any]:
        _add_picie_src_to_path()
        from medical_metrics import compute_medical_metrics

        all_pred = np.concatenate([p.ravel() for p in partitions])
        all_gt = np.concatenate([l.ravel() for l in labels])
        n_pred_clusters = max(int(all_pred.max()) + 1, 4)
        mapping = hungarian_match(all_pred, all_gt, n_pred_clusters)

        all_mapped_pred = []
        for part in partitions:
            mapped = apply_mapping(part, mapping)
            all_mapped_pred.append(mapped)

        patient_volumes_pred = defaultdict(list)
        patient_volumes_gt = defaultdict(list)
        patient_spacings = {}
        for mapped, label, prov in zip(all_mapped_pred, labels, provenances):
            pid = prov["patient_id"]
            z = int(prov.get("slice_index", 0))
            patient_volumes_pred[pid].append((z, mapped))
            patient_volumes_gt[pid].append((z, label))
            if "spacing" in prov:
                patient_spacings[pid] = prov["spacing"]

        volume_metrics = []
        for pid in sorted(patient_volumes_pred.keys()):
            pred_slices = sorted(patient_volumes_pred[pid], key=lambda x: x[0])
            gt_slices = sorted(patient_volumes_gt[pid], key=lambda x: x[0])
            # (H, W, Z) format as expected by PICIE medical_metrics
            pred_vol = np.stack([s for _, s in pred_slices], axis=-1)
            gt_vol = np.stack([s for _, s in gt_slices], axis=-1)

            # Original spacing in manifest is typically [Z, Y, X].
            # PICIE's compute_medical_metrics expects (spacing_y, spacing_x, spacing_z)
            sp = patient_spacings.get(pid, [1.0, 1.0, 1.0])
            spacing_3d = (sp[1], sp[2], sp[0])

            metrics = compute_medical_metrics(pred_vol, gt_vol, spacing_3d, num_classes=N_CLASSES)

            vol_dice = metrics["dice"]
            vol_hd95 = metrics["hd95"]
            vol_assd = metrics["assd"]

            volume_metrics.append({
                "patient_id": pid,
                "dice_per_class": {CLASS_NAMES[c]: float(vol_dice[c]) for c in range(N_CLASSES)},
                "hd95_per_class": {CLASS_NAMES[c]: float(vol_hd95[c]) for c in FOREGROUND_CLASSES},
                "assd_per_class": {CLASS_NAMES[c]: float(vol_assd[c]) for c in FOREGROUND_CLASSES},
            })

        mean_dice = {}
        mean_hd95 = {}
        mean_assd = {}
        for c in range(N_CLASSES):
            vals = [vm["dice_per_class"][CLASS_NAMES[c]] for vm in volume_metrics if not np.isnan(vm["dice_per_class"][CLASS_NAMES[c]])]
            mean_dice[CLASS_NAMES[c]] = float(np.mean(vals)) if vals else float("nan")
        for c in FOREGROUND_CLASSES:
            h_vals = [vm["hd95_per_class"][CLASS_NAMES[c]] for vm in volume_metrics if not np.isnan(vm["hd95_per_class"][CLASS_NAMES[c]]) and not np.isinf(vm["hd95_per_class"][CLASS_NAMES[c]])]
            a_vals = [vm["assd_per_class"][CLASS_NAMES[c]] for vm in volume_metrics if not np.isnan(vm["assd_per_class"][CLASS_NAMES[c]]) and not np.isinf(vm["assd_per_class"][CLASS_NAMES[c]])]
            mean_hd95[CLASS_NAMES[c]] = float(np.mean(h_vals)) if h_vals else float("nan")
            mean_assd[CLASS_NAMES[c]] = float(np.mean(a_vals)) if a_vals else float("nan")

        return {
            "evaluation_type": "cluster_probe_hungarian",
            "role": "DIAGNOSTIC ONLY — GT LEAKAGE BY DESIGN",
            "warning": "Metrics computed after Hungarian matching — GT used for assignment only.",
            "n_pred_clusters": n_pred_clusters,
            "cluster_to_class_mapping": {str(k): v for k, v in mapping.items()},
            "volume_metrics": volume_metrics,
            "summary": {
                "mean_dice": mean_dice,
                "mean_hd95_mm": mean_hd95,
                "mean_assd_mm": mean_assd,
            },
        }


def write_evaluation_report(
    report: dict[str, Any],
    output_dir: str | Path,
    *,
    config_hash: str = "",
    checkpoint_sha256: str = "",
) -> str:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    eval_type = report.get("evaluation_type", "unknown")
    full_report = {
        "schema_version": "picie.cardiac.evaluation.v1",
        "config_hash": config_hash,
        "checkpoint_sha256": checkpoint_sha256,
        "environment": environment_identity(),
        **report,
    }
    filename = f"evaluation_{eval_type}.json"
    return write_json(root / filename, full_report)