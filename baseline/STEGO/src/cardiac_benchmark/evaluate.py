"""Evaluation wrapper for STEGO cardiac benchmark.

Two evaluation branches:
1. Linear Probe (supervised) — OFFICIAL METRIC for baseline comparison.
   Trains a linear head on frozen STEGO features using train-set GT.
2. Cluster Probe (Hungarian matching) — DIAGNOSTIC ONLY.
   Uses GT on test set for matching. Not for official comparison due to GT leakage.

3D volume reconstruction: group slices by patient_id, compute Dice/HD95/ASSD.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.optimize import linear_sum_assignment

from .provenance import write_json, sha256_json, environment_identity

N_CLASSES = 4
CLASS_NAMES = ["Background", "RV", "MYO", "LV"]
FOREGROUND_CLASSES = [1, 2, 3]


def _dice_per_class(pred: np.ndarray, gt: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    dice = np.zeros(n_classes)
    for c in range(n_classes):
        p = (pred == c)
        g = (gt == c)
        intersection = np.sum(p & g)
        union = np.sum(p) + np.sum(g)
        dice[c] = 2.0 * intersection / union if union > 0 else float("nan")
    return dice


def _surface_distances(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    """Compute symmetric surface distances between binary masks."""
    pred_border = pred_mask ^ binary_erosion(pred_mask, iterations=1)
    gt_border = gt_mask ^ binary_erosion(gt_mask, iterations=1)

    if not np.any(pred_border) or not np.any(gt_border):
        return np.array([])

    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)

    d_gt_to_pred = dt_pred[gt_border]
    d_pred_to_gt = dt_gt[pred_border]

    return np.concatenate([d_gt_to_pred, d_pred_to_gt])


def _hd95(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    distances = _surface_distances(pred_mask, gt_mask, spacing)
    if len(distances) == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def _assd(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    distances = _surface_distances(pred_mask, gt_mask, spacing)
    if len(distances) == 0:
        return float("nan")
    return float(np.mean(distances))


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


def reconstruct_3d_volumes(
    partitions: dict[str, np.ndarray],
    provenances: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Group 2D slices by patient_id → 3D volumes sorted by slice_index."""
    patient_slices: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for sample_id, partition in partitions.items():
        prov = provenances[sample_id]
        patient_id = prov["patient_id"]
        z = int(prov["slice_index"])
        patient_slices[patient_id].append((z, partition))
    volumes = {}
    for patient_id, slices in patient_slices.items():
        slices.sort(key=lambda x: x[0])
        volumes[patient_id] = np.stack([s for _, s in slices], axis=0)
    return volumes


class LinearProbeEvaluator:
    """Train a linear probe on frozen STEGO features; evaluate on test.

    This is the OFFICIAL METRIC for Self-Audit baseline comparison.
    """

    def __init__(self, feature_dim: int = 70, n_classes: int = N_CLASSES, lr: float = 5e-3, epochs: int = 50):
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs

    def train_and_evaluate(
        self,
        train_features: list[np.ndarray],
        train_labels: list[np.ndarray],
        test_features: list[np.ndarray],
        test_labels: list[np.ndarray],
        test_provenances: list[dict[str, Any]],
        *,
        spacing_mm: tuple[float, ...] = (1.0, 1.0),
    ) -> dict[str, Any]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        probe = nn.Conv2d(self.feature_dim, self.n_classes, (1, 1)).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        start = time.perf_counter()
        for _ in range(self.epochs):
            probe.train()
            for feat, label in zip(train_features, train_labels):
                feat_t = torch.from_numpy(feat).unsqueeze(0).float().to(device)
                label_t = torch.from_numpy(label).long().to(device)
                h, w = label_t.shape
                logits = probe(feat_t)
                logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
                loss = loss_fn(logits, label_t.unsqueeze(0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_time = time.perf_counter() - start

        probe.eval()
        per_sample_metrics = []
        all_pred = []
        all_gt = []
        with torch.no_grad():
            for feat, label, prov in zip(test_features, test_labels, test_provenances):
                feat_t = torch.from_numpy(feat).unsqueeze(0).float().to(device)
                logits = probe(feat_t)
                h, w = label.shape
                logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
                pred = logits.argmax(1).squeeze(0).cpu().numpy()
                dice = _dice_per_class(pred, label)
                per_sample_metrics.append({
                    "sample_id": prov["sample_id"],
                    "patient_id": prov["patient_id"],
                    "dice_per_class": {CLASS_NAMES[c]: float(dice[c]) for c in range(self.n_classes)},
                })
                all_pred.append(pred)
                all_gt.append(label)

        patient_volumes_pred = defaultdict(list)
        patient_volumes_gt = defaultdict(list)
        for pred, label, prov in zip(all_pred, all_gt, test_provenances):
            pid = prov["patient_id"]
            z = int(prov.get("slice_index", 0))
            patient_volumes_pred[pid].append((z, pred))
            patient_volumes_gt[pid].append((z, label))

        volume_metrics = []
        for pid in sorted(patient_volumes_pred.keys()):
            pred_slices = sorted(patient_volumes_pred[pid], key=lambda x: x[0])
            gt_slices = sorted(patient_volumes_gt[pid], key=lambda x: x[0])
            pred_vol = np.stack([s for _, s in pred_slices], axis=0)
            gt_vol = np.stack([s for _, s in gt_slices], axis=0)
            spacing_3d = (1.0,) + tuple(spacing_mm)
            vol_dice = _dice_per_class(pred_vol, gt_vol)
            vol_hd95 = {}
            vol_assd = {}
            for c in FOREGROUND_CLASSES:
                vol_hd95[CLASS_NAMES[c]] = _hd95(pred_vol == c, gt_vol == c, spacing_3d)
                vol_assd[CLASS_NAMES[c]] = _assd(pred_vol == c, gt_vol == c, spacing_3d)
            volume_metrics.append({
                "patient_id": pid,
                "dice_per_class": {CLASS_NAMES[c]: float(vol_dice[c]) for c in range(self.n_classes)},
                "hd95_per_class": vol_hd95,
                "assd_per_class": vol_assd,
            })

        mean_dice = {}
        mean_hd95 = {}
        mean_assd = {}
        for c in range(self.n_classes):
            vals = [vm["dice_per_class"][CLASS_NAMES[c]] for vm in volume_metrics if not np.isnan(vm["dice_per_class"][CLASS_NAMES[c]])]
            mean_dice[CLASS_NAMES[c]] = float(np.mean(vals)) if vals else float("nan")
        for c in FOREGROUND_CLASSES:
            h_vals = [vm["hd95_per_class"][CLASS_NAMES[c]] for vm in volume_metrics if not np.isnan(vm["hd95_per_class"][CLASS_NAMES[c]])]
            a_vals = [vm["assd_per_class"][CLASS_NAMES[c]] for vm in volume_metrics if not np.isnan(vm["assd_per_class"][CLASS_NAMES[c]])]
            mean_hd95[CLASS_NAMES[c]] = float(np.mean(h_vals)) if h_vals else float("nan")
            mean_assd[CLASS_NAMES[c]] = float(np.mean(a_vals)) if a_vals else float("nan")

        return {
            "evaluation_type": "linear_probe",
            "role": "OFFICIAL_METRIC",
            "n_classes": self.n_classes,
            "probe_epochs": self.epochs,
            "probe_lr": self.lr,
            "train_time_seconds": train_time,
            "per_sample_metrics": per_sample_metrics,
            "volume_metrics": volume_metrics,
            "summary": {
                "mean_dice": mean_dice,
                "mean_hd95_mm": mean_hd95,
                "mean_assd_mm": mean_assd,
            },
        }


class ClusterProbeEvaluator:
    """Hungarian matching evaluation — DIAGNOSTIC ONLY.

    WARNING: This uses GT on the test set for matching.
    GT LEAKAGE BY DESIGN — do NOT use as official comparison metric.
    """

    def evaluate(
        self,
        partitions: list[np.ndarray],
        labels: list[np.ndarray],
        provenances: list[dict[str, Any]],
        *,
        spacing_mm: tuple[float, ...] = (1.0, 1.0),
    ) -> dict[str, Any]:
        all_pred = np.concatenate([p.ravel() for p in partitions])
        all_gt = np.concatenate([l.ravel() for l in labels])
        n_pred_clusters = int(all_pred.max()) + 1
        mapping = hungarian_match(all_pred, all_gt, n_pred_clusters)

        per_sample = []
        all_mapped_pred = []
        for part, label, prov in zip(partitions, labels, provenances):
            mapped = apply_mapping(part, mapping)
            dice = _dice_per_class(mapped, label)
            per_sample.append({
                "sample_id": prov["sample_id"],
                "patient_id": prov["patient_id"],
                "dice_per_class": {CLASS_NAMES[c]: float(dice[c]) for c in range(N_CLASSES)},
            })
            all_mapped_pred.append(mapped)

        patient_volumes_pred = defaultdict(list)
        patient_volumes_gt = defaultdict(list)
        for mapped, label, prov in zip(all_mapped_pred, labels, provenances):
            pid = prov["patient_id"]
            z = int(prov.get("slice_index", 0))
            patient_volumes_pred[pid].append((z, mapped))
            patient_volumes_gt[pid].append((z, label))

        volume_metrics = []
        for pid in sorted(patient_volumes_pred.keys()):
            pred_slices = sorted(patient_volumes_pred[pid], key=lambda x: x[0])
            gt_slices = sorted(patient_volumes_gt[pid], key=lambda x: x[0])
            pred_vol = np.stack([s for _, s in pred_slices], axis=0)
            gt_vol = np.stack([s for _, s in gt_slices], axis=0)
            spacing_3d = (1.0,) + tuple(spacing_mm)
            vol_dice = _dice_per_class(pred_vol, gt_vol)
            vol_hd95 = {}
            vol_assd = {}
            for c in FOREGROUND_CLASSES:
                vol_hd95[CLASS_NAMES[c]] = _hd95(pred_vol == c, gt_vol == c, spacing_3d)
                vol_assd[CLASS_NAMES[c]] = _assd(pred_vol == c, gt_vol == c, spacing_3d)
            volume_metrics.append({
                "patient_id": pid,
                "dice_per_class": {CLASS_NAMES[c]: float(vol_dice[c]) for c in range(N_CLASSES)},
                "hd95_per_class": vol_hd95,
                "assd_per_class": vol_assd,
            })

        mean_dice = {}
        mean_hd95 = {}
        mean_assd = {}
        for c in range(N_CLASSES):
            vals = [vm["dice_per_class"][CLASS_NAMES[c]] for vm in volume_metrics if not np.isnan(vm["dice_per_class"][CLASS_NAMES[c]])]
            mean_dice[CLASS_NAMES[c]] = float(np.mean(vals)) if vals else float("nan")
        for c in FOREGROUND_CLASSES:
            h_vals = [vm["hd95_per_class"][CLASS_NAMES[c]] for vm in volume_metrics if not np.isnan(vm["hd95_per_class"][CLASS_NAMES[c]])]
            a_vals = [vm["assd_per_class"][CLASS_NAMES[c]] for vm in volume_metrics if not np.isnan(vm["assd_per_class"][CLASS_NAMES[c]])]
            mean_hd95[CLASS_NAMES[c]] = float(np.mean(h_vals)) if h_vals else float("nan")
            mean_assd[CLASS_NAMES[c]] = float(np.mean(a_vals)) if a_vals else float("nan")

        return {
            "evaluation_type": "cluster_probe_hungarian",
            "role": "DIAGNOSTIC ONLY — GT LEAKAGE BY DESIGN",
            "warning": "Do NOT use as official comparison metric. "
                       "Hungarian matching requires GT labels on test set.",
            "n_pred_clusters": n_pred_clusters,
            "cluster_to_class_mapping": {str(k): v for k, v in mapping.items()},
            "per_sample_metrics": per_sample,
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
        "schema_version": "stego.cardiac.evaluation.v1",
        "config_hash": config_hash,
        "checkpoint_sha256": checkpoint_sha256,
        "environment": environment_identity(),
        **report,
    }
    filename = f"evaluation_{eval_type}.json"
    return write_json(root / filename, full_report)
