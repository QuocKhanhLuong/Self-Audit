"""Fully automatic head pilot controller: metric-aligned semi-supervised adaptation.

Implements the locked contract from docs/AUTOMATIC_PILOT_CONTRACT.md and
FULLY_AUTOMATIC_REVIEW_VI.md:
  - Exact common scalar restriction with cross terms and m*ridge.
  - Cached affine logit screening + actual functional trial evaluation for finite Brier
    and optional hard-Dice guards on complete fixed support.
  - Full volume / frame / patient reduction matching evaluation absent-class policy.
  - Fixed observed-voxel denominator for absent-frame FP rates.
  - Parity check between cached affine trial logits and actual head forward.
  - Atomic direct commit on acceptance; completely non-mutating on rejection preserving
    parameter _version, gradients, optimizer state, and RNG.
  - JSON-safe comprehensive diagnostics including displacement, margins, and margin
    decreased fraction.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

# Reuse legacy solver, scalar allocation, make_candidates, and Allocation dataclass
from self_audit.control import (
    Allocation,
    make_candidates,
    scalar_allocation,
    solve_allocation,
)

__all__ = [
    "Allocation",
    "CLASS_NAMES",
    "FOREGROUND_CLASS_INDICES",
    "FOREGROUND_CLASS_NAMES",
    "audit_weights",
    "common_scalar_allocation",
    "compute_margins",
    "compute_segmentation_metrics",
    "controlled_head_step",
    "make_candidates",
    "risks",
    "response_moments",
    "scalar_allocation",
    "solve_allocation",
]

CLASS_NAMES = ("BG", "RV", "MYO", "LV")
FOREGROUND_CLASS_INDICES = (1, 2, 3)
FOREGROUND_CLASS_NAMES = ("RV", "MYO", "LV")


def common_scalar_allocation(
    grams: np.ndarray,
    alignment: np.ndarray,
    baseline: np.ndarray,
    *,
    ridge: float = 1e-6,
    budget: float = 0.0,
    source_ids: Sequence[str] | None = None,
) -> Allocation:
    """Exact common scalar restriction alpha = s * 1 with all cross terms and m * ridge.

    For each group c:
        a_c = 1^T A_c 1 = sum_{i,j} A_{c,i,j}
        b'_c = 1^T b_c = sum_i b_{c,i}
        lambda_s = m * ridge
    """
    A = np.asarray(grams, dtype=np.float64)
    b = np.asarray(alignment, dtype=np.float64)
    e = np.asarray(baseline, dtype=np.float64)

    if A.ndim != 3 or A.shape[1] != A.shape[2] or b.shape != A.shape[:2] or e.shape != A.shape[:1]:
        raise ValueError("expected GxMxM, GxM, G")

    G = e.shape[0]
    if G == 0:
        raise ValueError("G must be positive")

    if not isinstance(ridge, (int, float)) or ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("positive finite ridge required")

    eps = np.broadcast_to(np.asarray(budget, dtype=np.float64), e.shape)
    if (eps < 0).any() or not np.isfinite(eps).all() or (e < 0).any() or not np.isfinite(e).all():
        raise ValueError("invalid baseline or budgets")

    m = b.shape[1]
    if source_ids is not None and (len(source_ids) != m or len(set(source_ids)) != m):
        raise ValueError("duplicate/missing source IDs: aggregate by class before allocation")
    if m == 0:
        return Allocation(np.zeros(0, dtype=np.float64), "NO_CANDIDATES", float(e.mean()), e.copy())

    if not (np.isfinite(A).all() and np.isfinite(b).all()):
        raise ValueError("finite moments required")
    if not np.allclose(A, A.swapaxes(1, 2), atol=1e-10) or np.linalg.eigvalsh(A).min() < -1e-9:
        raise ValueError("Gram matrices must be symmetric positive semidefinite")

    # a_c = 1^T A_c 1 (summing all elements includes all off-diagonal cross terms)
    a = A.sum(axis=(1, 2))
    # b'_c = 1^T b_c
    b_prime = b.sum(axis=1)
    # lambda_s = m * ridge
    lambda_s = float(m * ridge)

    s = scalar_allocation(a, b_prime, ridge=lambda_s, budget=budget)
    alpha = np.full(m, s, dtype=np.float64)
    local_risks = e - b_prime * s + 0.5 * a * (s * s)
    objective = float(local_risks.mean() + 0.5 * lambda_s * (s * s))
    return Allocation(alpha, "COMMON_SCALAR", objective, local_risks)


def audit_weights(
    labels: Tensor,
    patient_ids: Sequence[Any] | None = None,
    frame_ids: Sequence[Any] | None = None,
    valid_mask: Tensor | None = None,
    classes: int = 4,
) -> tuple[Tensor, Tensor, list[list[int]]]:
    """Foreground one-vs-rest Brier groups on complete fixed frame volumes.

    Balances positives and negatives in each complete frame volume.
    Averages frames within each patient, then averages patients; does not bias by slice count.
    Ignored 255 and geometry padding contribute zero mass.
    """
    if labels.ndim != 3:
        raise ValueError("expected BHW class IDs or 255")
    if not torch.isin(labels, labels.new_tensor([0, 1, 2, 3, 255])).all():
        raise ValueError("labels ontology must strictly contain only {0, 1, 2, 3, 255}")

    valid = labels != 255
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool or valid_mask.shape != labels.shape:
            raise ValueError("valid_mask must be boolean BHW")
        valid = valid & valid_mask

    if not valid.any():
        raise ValueError("no observed valid pixels in support")

    B = labels.shape[0]
    if patient_ids is None:
        patient_ids = list(range(B))
    if frame_ids is None:
        frame_ids = [0] * B

    if len(patient_ids) != B or len(frame_ids) != B:
        raise ValueError(f"patient_ids and frame_ids must match batch size {B}")

    unique_patients = list(dict.fromkeys(patient_ids))

    # Identify valid frames per patient (frames with at least one valid pixel)
    valid_frames_per_patient: dict[Any, list[Any]] = {}
    for pid in unique_patients:
        p_fids = list(dict.fromkeys(frame_ids[i] for i in range(B) if patient_ids[i] == pid))
        valid_fids = [
            fid for fid in p_fids
            if valid[[i for i in range(B) if patient_ids[i] == pid and frame_ids[i] == fid]].any()
        ]
        valid_frames_per_patient[pid] = valid_fids

    valid_patients = [pid for pid in unique_patients if len(valid_frames_per_patient[pid]) > 0]
    num_valid_patients = len(valid_patients)
    if num_valid_patients == 0:
        raise ValueError("no observed valid pixels in support")

    weights: list[Tensor] = []
    targets: list[Tensor] = []
    positives: list[list[int]] = []

    for c in range(1, classes):
        target = labels == c
        w = torch.zeros_like(labels, dtype=torch.float64)
        pos_counts: list[int] = []

        for n in range(B):
            pos_slice = (target[n] & valid[n]).sum().item()
            pos_counts.append(int(pos_slice))

        p_mass = 1.0 / num_valid_patients
        for pid in valid_patients:
            valid_fids = valid_frames_per_patient[pid]
            frame_mass = p_mass / len(valid_fids)

            for fid in valid_fids:
                indices = [i for i in range(B) if patient_ids[i] == pid and frame_ids[i] == fid]
                f_target = target[indices]
                f_valid = valid[indices]

                pos_mask = f_target & f_valid
                neg_mask = (~f_target) & f_valid

                np_ = int(pos_mask.sum().item())
                nn_ = int(neg_mask.sum().item())
                v_total = int(f_valid.sum().item())

                if np_ > 0 and nn_ > 0:
                    pos_weight = (0.5 * frame_mass) / np_
                    neg_weight = (0.5 * frame_mass) / nn_
                    for idx_pos, slice_idx in enumerate(indices):
                        if pos_mask[idx_pos].any():
                            w[slice_idx, pos_mask[idx_pos]] = pos_weight
                        if neg_mask[idx_pos].any():
                            w[slice_idx, neg_mask[idx_pos]] = neg_weight
                else:
                    absent_weight = frame_mass / v_total
                    for idx_pos, slice_idx in enumerate(indices):
                        if f_valid[idx_pos].any():
                            w[slice_idx, f_valid[idx_pos]] = absent_weight

        weights.append(w)
        targets.append(target.to(torch.float64))
        positives.append(pos_counts)

    return torch.stack(weights), torch.stack(targets), positives


def risks(probabilities: Tensor, weights: Tensor, targets: Tensor) -> np.ndarray:
    """Compute one-vs-rest Brier risk per foreground class."""
    p = probabilities[:, 1:].permute(1, 0, 2, 3).double()
    return (0.5 * weights * (p - targets).square()).flatten(1).sum(1).detach().cpu().numpy()


def response_moments(
    p: Tensor,
    dp: Tensor,
    weights: Tensor,
    targets: Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute quadratic Brier response moments A (GxMxM), b (GxM), e (G)."""
    m = dp.shape[0]
    r = (targets - p[:, 1:].permute(1, 0, 2, 3).double()).flatten(1)
    w = weights.flatten(1)
    v = dp[:, :, 1:].permute(2, 1, 3, 4, 0).double().reshape(3, -1, m)
    A = torch.einsum("gnm,gn,gnk->gmk", v, w, v)
    b = torch.einsum("gn,gn,gnm->gm", r, w, v)
    e = 0.5 * (w * r.square()).sum(1)
    return tuple(x.detach().cpu().numpy() for x in (A, b, e))


def compute_segmentation_metrics(
    predictions: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
    patient_ids: Sequence[Any],
    frame_ids: Sequence[Any],
    valid_mask: Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute full volume / frame / patient reduction matching evaluation semantics.

    - Multiclass argmax.
    - Sum TP, FP, FN over all valid voxels/slices of each frame volume before ratios.
    - If reference class is absent (GT == 0) in a frame volume:
        * Frame Dice is None (regardless of FP).
        * Absent-frame FP rate is computed as FP / valid_voxels with fixed observed denominator.
    - Entirely ignored frame volume has absent-frame FP rate = None (never fabricated 0.0).
    - Patient aggregation:
        * patient_dice[c] is mean of defined frame Dices for class c.
        * patient_absent_fp[c] is mean of defined absent-frame FP rates for class c.
    - Macro aggregation:
        * macro_dice[c] is mean of patient_dice[c] over defined patients.
        * macro_absent_fp[c] is mean of patient_absent_fp[c] over defined patients.
        * foreground_macro_dice is mean of macro_dice across defined foreground classes.
    """
    if isinstance(predictions, Tensor):
        preds_np = predictions.detach().cpu().numpy()
    else:
        preds_np = np.asarray(predictions)

    if isinstance(labels, Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    B = len(patient_ids)
    if len(frame_ids) != B or preds_np.shape[0] != B or labels_np.shape[0] != B:
        raise ValueError("batch dimension mismatch between predictions, labels, and IDs")

    valid = labels_np != 255
    if valid_mask is not None:
        if isinstance(valid_mask, Tensor):
            vm_np = valid_mask.detach().cpu().numpy()
        else:
            vm_np = np.asarray(valid_mask)
        valid = valid & vm_np

    # Group slice indices by (patient_id, frame_id)
    unique_patients = list(dict.fromkeys(patient_ids))
    frame_map: dict[tuple[Any, Any], list[int]] = {}
    for idx in range(B):
        key = (patient_ids[idx], frame_ids[idx])
        frame_map.setdefault(key, []).append(idx)

    # Compute metrics per frame volume
    frame_records: list[dict[str, Any]] = []
    for (pid, fid), slice_indices in frame_map.items():
        rec: dict[str, Any] = {"patient": pid, "frame": fid, "classes": {}}
        f_valid = valid[slice_indices]
        f_preds = preds_np[slice_indices]
        f_labels = labels_np[slice_indices]
        valid_voxels = int(f_valid.sum())

        for c_idx, c_name in enumerate(CLASS_NAMES):
            if valid_voxels == 0:
                rec["classes"][c_name] = {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "gt": 0,
                    "pred": 0,
                    "valid_voxels": 0,
                    "dice": None,
                    "absent_fp_rate": None,
                }
                continue

            c_gt = int(((f_labels == c_idx) & f_valid).sum())
            c_pred = int(((f_preds == c_idx) & f_valid).sum())
            c_tp = int(((f_preds == c_idx) & (f_labels == c_idx) & f_valid).sum())
            c_fp = c_pred - c_tp
            c_fn = c_gt - c_tp

            if c_gt > 0:
                dice = float(2.0 * c_tp / (c_gt + c_pred))
                absent_fp_rate = None
            else:
                dice = None
                absent_fp_rate = float(c_fp / valid_voxels)

            rec["classes"][c_name] = {
                "tp": c_tp,
                "fp": c_fp,
                "fn": c_fn,
                "gt": c_gt,
                "pred": c_pred,
                "valid_voxels": valid_voxels,
                "dice": dice,
                "absent_fp_rate": absent_fp_rate,
            }
        frame_records.append(rec)

    # Patient aggregation
    patient_metrics: dict[str, Any] = {}
    for pid in unique_patients:
        p_frames = [r for r in frame_records if r["patient"] == pid]
        p_dice: dict[str, float | None] = {}
        p_absent_fp: dict[str, float | None] = {}

        for c_name in FOREGROUND_CLASS_NAMES:
            defined_dices = [
                f["classes"][c_name]["dice"]
                for f in p_frames
                if f["classes"][c_name]["dice"] is not None
            ]
            p_dice[c_name] = float(np.mean(defined_dices)) if defined_dices else None

            defined_fps = [
                f["classes"][c_name]["absent_fp_rate"]
                for f in p_frames
                if f["classes"][c_name]["absent_fp_rate"] is not None
            ]
            p_absent_fp[c_name] = float(np.mean(defined_fps)) if defined_fps else None

        patient_metrics[str(pid)] = {
            "dice": p_dice,
            "absent_fp_rate": p_absent_fp,
        }

    # Macro aggregation per foreground class
    macro_dice: dict[str, float | None] = {}
    macro_absent_fp: dict[str, float | None] = {}

    for c_name in FOREGROUND_CLASS_NAMES:
        defined_p_dices = [
            patient_metrics[str(pid)]["dice"][c_name]
            for pid in unique_patients
            if patient_metrics[str(pid)]["dice"][c_name] is not None
        ]
        macro_dice[c_name] = float(np.mean(defined_p_dices)) if defined_p_dices else None

        defined_p_fps = [
            patient_metrics[str(pid)]["absent_fp_rate"][c_name]
            for pid in unique_patients
            if patient_metrics[str(pid)]["absent_fp_rate"][c_name] is not None
        ]
        macro_absent_fp[c_name] = float(np.mean(defined_p_fps)) if defined_p_fps else None

    defined_macros = [d for d in macro_dice.values() if d is not None]
    foreground_macro_dice = float(np.mean(defined_macros)) if defined_macros else None

    return {
        "macro_dice": macro_dice,
        "macro_absent_fp": macro_absent_fp,
        "foreground_macro_dice": foreground_macro_dice,
        "patient_metrics": patient_metrics,
        "frame_records": frame_records,
    }


def compute_margins(
    logits: Tensor,
    labels: Tensor,
    valid_mask: Tensor | None = None,
) -> tuple[dict[str, float | None], Tensor | None]:
    """Compute true margin (z_true - max_{c!=true} z_c) via -inf mask and predicted margin."""
    with torch.no_grad():
        valid = labels != 255
        if valid_mask is not None:
            valid = valid & valid_mask
        if not valid.any():
            return {
                "mean_true_margin": None,
                "min_true_margin": None,
                "mean_predicted_margin": None,
                "min_predicted_margin": None,
            }, None

        labels_clamped = labels.clamp(0, 3)
        true_logits = logits.gather(1, labels_clamped.unsqueeze(1)).squeeze(1)

        # Mask true class with -inf (never arbitrary finite constants)
        mask = F.one_hot(labels_clamped, 4).permute(0, 3, 1, 2).bool()
        other_logits = logits.masked_fill(mask, -float("inf"))
        max_other_logits = other_logits.max(1).values
        true_margins = true_logits - max_other_logits
        valid_true = true_margins[valid]

        top2 = logits.topk(2, dim=1).values
        pred_margins = top2[:, 0] - top2[:, 1]
        valid_pred = pred_margins[valid]

        summary = {
            "mean_true_margin": float(valid_true.mean().item()),
            "min_true_margin": float(valid_true.min().item()),
            "mean_predicted_margin": float(valid_pred.mean().item()),
            "min_predicted_margin": float(valid_pred.min().item()),
        }
        return summary, valid_true


def controlled_head_step(
    head: nn.Conv2d,
    features: Tensor,
    labels: Tensor,
    directions: list[tuple[Tensor, Tensor]],
    source_ids: list[str],
    *,
    patient_ids: Sequence[Any],
    frame_ids: Sequence[Any],
    valid_mask: Tensor | None = None,
    allocation: str = "joint",
    dice_guard: bool = False,
    ridge: float = 1e-6,
    budget: float = 0.0,
    backtracks: int = 8,
    tolerance: float = 1e-8,
    dice_tolerance: float = 0.0,
    fp_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Atomic extra unlabeled update on fixed support.

    Evaluates cached affine logit trial steps, verifies actual functional trial
    Brier and hard-Dice/FP before accepting, enforces parity against actual head forward,
    directly commits if accepted, or atomically rejects leaving all parameter versions,
    gradients, RNG and optimizers strictly untouched.
    """
    valid_allocations = {"joint", "scalar", "independent", "plain"}
    if allocation not in valid_allocations:
        raise ValueError(f"unknown allocation: {allocation}, expected one of {valid_allocations}")

    # Validate all tolerance and budget args are finite numbers
    for arg_name, arg_val in [
        ("ridge", ridge),
        ("budget", budget),
        ("tolerance", tolerance),
        ("dice_tolerance", dice_tolerance),
        ("fp_tolerance", fp_tolerance),
    ]:
        if not isinstance(arg_val, (int, float)) or not np.isfinite(arg_val):
            raise ValueError(f"{arg_name} must be a finite float")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    if budget < 0 or tolerance < 0 or dice_tolerance < 0 or fp_tolerance < 0:
        raise ValueError("tolerances and budget must be non-negative")
    if not isinstance(backtracks, int) or backtracks < 1:
        raise ValueError("backtracks must be an integer >= 1")

    # Validate tensor types, dimensions, ontology, and spatial correspondence
    if not isinstance(features, Tensor) or features.ndim != 4:
        raise ValueError("features must be a 4D Tensor (BCHW)")
    if not isinstance(labels, Tensor) or labels.ndim != 3:
        raise ValueError("labels must be a 3D Tensor (BHW)")
    B = features.shape[0]
    if labels.shape[0] != B:
        raise ValueError(f"batch size mismatch: features has {B}, labels has {labels.shape[0]}")
    if features.shape[2:] != labels.shape[1:]:
        raise ValueError(f"spatial shape mismatch: features has {features.shape[2:]}, labels has {labels.shape[1:]}")
    if not torch.isin(labels, labels.new_tensor([0, 1, 2, 3, 255])).all():
        raise ValueError("labels ontology must strictly contain only {0, 1, 2, 3, 255}")

    if valid_mask is not None:
        if not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool or valid_mask.shape != labels.shape:
            raise ValueError("valid_mask must be boolean Tensor with same shape as labels (BHW)")

    # Validate patient and frame IDs
    if len(patient_ids) != B or len(frame_ids) != B:
        raise ValueError(f"patient_ids and frame_ids must have length equal to batch size {B}")
    if any(p is None for p in patient_ids) or any(f is None for f in frame_ids):
        raise ValueError("patient_ids and frame_ids cannot contain None values")

    if len(directions) != len(source_ids) or len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate/missing source IDs: each candidate direction must have unique ID")

    params = tuple(head.parameters())
    if len(params) != 2 or head.kernel_size != (1, 1):
        raise ValueError("controlled_head_step requires affine 1x1 head with weight and bias")

    # Snapshot references (without writing to head parameters or RNG)
    base_w = head.weight.detach().clone()
    base_b = head.bias.detach().clone()

    guarded = allocation != "plain"

    with torch.no_grad():
        f = features.detach()
        base_logits = F.conv2d(f, base_w, base_b)
        if not torch.isfinite(base_logits).all():
            raise FloatingPointError("non-finite base logits from head")

        p_base = base_logits.softmax(1)
        weights, targets, positives = audit_weights(
            labels, patient_ids=patient_ids, frame_ids=frame_ids, valid_mask=valid_mask
        )
        e_base = risks(p_base, weights, targets)

        base_preds = base_logits.argmax(1)
        base_seg = compute_segmentation_metrics(
            base_preds, labels, patient_ids, frame_ids, valid_mask=valid_mask
        )
        base_margins, valid_true_before = compute_margins(base_logits, labels, valid_mask=valid_mask)

        report: dict[str, Any] = {
            "allocation": allocation,
            "dice_guard": dice_guard,
            "source_ids": list(source_ids),
            "before_brier": e_base.tolist(),
            "positive_pixels": positives,
            "scope": "observed_TRAIN_fixed_support",
            "optimizer_steps_in_transaction": 0,
            "ema_updates_in_transaction": 0,
            "macro_dice_before": base_seg["macro_dice"],
            "macro_absent_fp_before": base_seg["macro_absent_fp"],
            "macro_fg_dice_before": base_seg["foreground_macro_dice"],
            "margins_before": base_margins,
        }

        m = len(directions)
        if m == 0:
            report.update({
                "alpha": [],
                "scale": 0.0,
                "accepted": False,
                "after_brier": e_base.tolist(),
                "status": "NO_CANDIDATES",
                "proposed_displacement_norm": 0.0,
                "realized_displacement_norm": 0.0,
                "weight_displacement_norm": 0.0,
                "bias_displacement_norm": 0.0,
                "realized_max_brier_regression": 0.0,
                "linearized_after_brier": e_base.tolist(),
                "cache_parity_max_error": None,
                "macro_dice_after": base_seg["macro_dice"],
                "macro_absent_fp_after": base_seg["macro_absent_fp"],
                "macro_fg_dice_after": base_seg["foreground_macro_dice"],
                "margins_after": base_margins,
                "margin_change": 0.0,
                "margin_decreased_fraction": 0.0,
                "trials": [],
            })
            return report

        # Validate candidate directions
        for dw, db in directions:
            if (
                dw.shape != base_w.shape
                or db.shape != base_b.shape
                or not torch.isfinite(dw).all()
                or not torch.isfinite(db).all()
            ):
                raise ValueError("invalid or non-finite head displacement")

        # Compute logit responses and moments
        dz_list = [F.conv2d(f, dw, db) for dw, db in directions]
        dz = torch.stack(dz_list)
        dp = p_base[None] * (dz - (p_base[None] * dz).sum(2, keepdim=True))
        A, b, _ = response_moments(p_base, dp, weights, targets)
        del dp

        # Solve allocation
        if allocation == "joint":
            sol = solve_allocation(
                A, b, e_base, ridge=ridge, budget=budget, constrained=True, source_ids=source_ids
            )
            alpha, status = sol.alpha, sol.status
        elif allocation == "scalar":
            sol = common_scalar_allocation(
                A, b, e_base, ridge=ridge, budget=budget, source_ids=source_ids
            )
            alpha, status = sol.alpha, sol.status
        elif allocation == "independent":
            alpha = np.array([
                scalar_allocation(A[:, j, j], b[:, j], ridge=ridge, budget=budget)
                for j in range(m)
            ], dtype=np.float64)
            status = "INDEPENDENT_SCALARS"
        elif allocation == "plain":
            alpha = np.ones(m, dtype=np.float64)
            status = "PLAIN"

        delta_w = sum(float(a) * d[0] for a, d in zip(alpha, directions))
        delta_b = sum(float(a) * d[1] for a, d in zip(alpha, directions))
        dz_comb = sum(float(a) * z for a, z in zip(alpha, dz_list))
        del dz_list, dz

        proposed_disp_norm = float(
            torch.sqrt(delta_w.double().norm() ** 2 + delta_b.double().norm() ** 2).item()
        )

        trials: list[dict[str, Any]] = []
        chosen_scale = 0.0
        chosen_w = base_w
        chosen_b = base_b
        after_e = e_base
        after_seg = base_seg
        after_margins = base_margins
        valid_true_after = valid_true_before
        cache_parity_error: float | None = None

        if np.any(alpha > 0) and proposed_disp_norm > 0:
            num_trials = 1 if not guarded else backtracks
            for k in range(num_trials):
                s = 1.0 if not guarded else (0.5 ** k)
                trial_logits = base_logits + s * dz_comb
                trial_dict: dict[str, Any] = {"backtrack": k, "scale": s}

                if not torch.isfinite(trial_logits).all():
                    trial_dict.update({
                        "finite": False,
                        "brier_ok": False,
                        "dice_ok": False,
                        "accepted": False,
                    })
                    trials.append(trial_dict)
                    continue

                # Preliminary cached check
                p_cached = trial_logits.softmax(1)
                e_cached = risks(p_cached, weights, targets)
                cached_brier_within = bool(np.all(e_cached <= e_base + budget + tolerance))
                cached_brier_improves = bool(float(e_cached.mean()) <= float(e_base.mean()) + tolerance)
                cached_brier_ok = cached_brier_within and cached_brier_improves
                trial_dict["cached_brier_ok"] = cached_brier_ok

                if guarded and not cached_brier_ok:
                    trial_dict.update({
                        "finite": True,
                        "accepted": False,
                    })
                    trials.append(trial_dict)
                    continue

                # Actual functional trial head forward
                cand_w = base_w + s * delta_w
                cand_b = base_b + s * delta_b
                actual_logits = F.conv2d(f, cand_w, cand_b)

                if not torch.isfinite(actual_logits).all():
                    trial_dict.update({
                        "finite": False,
                        "actual_finite": False,
                        "accepted": False,
                    })
                    trials.append(trial_dict)
                    continue

                parity_err = float((actual_logits - trial_logits).abs().max().item())
                parity_tol = max(1e-5, tolerance)
                if parity_err > parity_tol:
                    trial_dict.update({
                        "finite": True,
                        "parity_error": parity_err,
                        "parity_ok": False,
                        "accepted": False,
                    })
                    trials.append(trial_dict)
                    continue

                # Evaluate ACTUAL functional Brier on actual logits
                actual_p = actual_logits.softmax(1)
                actual_e = risks(actual_p, weights, targets)
                actual_brier_within = bool(np.all(actual_e <= e_base + budget + tolerance))
                actual_brier_improves = bool(float(actual_e.mean()) <= float(e_base.mean()) + tolerance)
                actual_brier_ok = actual_brier_within and actual_brier_improves

                # Evaluate ACTUAL functional Hard-Dice / FP on actual logits
                actual_dice_ok = True
                actual_seg = None
                if guarded and dice_guard:
                    actual_preds = actual_logits.argmax(1)
                    actual_seg = compute_segmentation_metrics(
                        actual_preds, labels, patient_ids, frame_ids, valid_mask=valid_mask
                    )
                    trial_dict["macro_dice"] = actual_seg["macro_dice"]
                    trial_dict["macro_absent_fp"] = actual_seg["macro_absent_fp"]
                    trial_dict["foreground_macro_dice"] = actual_seg["foreground_macro_dice"]

                    for c_name in FOREGROUND_CLASS_NAMES:
                        base_d = base_seg["macro_dice"].get(c_name)
                        if base_d is not None:
                            tr_d = actual_seg["macro_dice"].get(c_name)
                            if tr_d is None or tr_d < base_d - dice_tolerance:
                                actual_dice_ok = False
                                break

                        base_fp = base_seg["macro_absent_fp"].get(c_name)
                        if base_fp is not None:
                            tr_fp = actual_seg["macro_absent_fp"].get(c_name)
                            if tr_fp is None or tr_fp > base_fp + fp_tolerance:
                                actual_dice_ok = False
                                break

                    if actual_dice_ok and base_seg["foreground_macro_dice"] is not None:
                        tr_fg = actual_seg["foreground_macro_dice"]
                        if tr_fg is None or tr_fg < base_seg["foreground_macro_dice"] - dice_tolerance:
                            actual_dice_ok = False

                accepted = (not guarded) or (actual_brier_ok and actual_dice_ok)
                trial_dict.update({
                    "finite": True,
                    "actual_brier_ok": actual_brier_ok if guarded else True,
                    "actual_dice_ok": actual_dice_ok if guarded and dice_guard else True,
                    "parity_error": parity_err,
                    "accepted": accepted,
                })
                trials.append(trial_dict)

                if accepted:
                    chosen_scale = s
                    chosen_w = cand_w
                    chosen_b = cand_b
                    after_e = actual_e
                    cache_parity_error = parity_err
                    if actual_seg is not None:
                        after_seg = actual_seg
                    else:
                        after_preds = actual_logits.argmax(1)
                        after_seg = compute_segmentation_metrics(
                            after_preds, labels, patient_ids, frame_ids, valid_mask=valid_mask
                        )
                    after_margins, valid_true_after = compute_margins(actual_logits, labels, valid_mask=valid_mask)
                    break

        if chosen_scale > 0:
            # Commit directly to head parameters
            head.weight.copy_(chosen_w)
            head.bias.copy_(chosen_b)
        else:
            # Pure functional: do NOT touch parameters, gradients, optimizer state, or RNG!
            if status not in ("NO_CANDIDATES", "SOLVER_FALLBACK_ZERO"):
                status = "REJECTED_ALL_TRIALS"
            after_e = e_base
            after_seg = base_seg
            after_margins = base_margins
            valid_true_after = valid_true_before

        # Compute displacement norms
        realized_dw = chosen_scale * delta_w if chosen_scale > 0 else torch.zeros_like(base_w)
        realized_db = chosen_scale * delta_b if chosen_scale > 0 else torch.zeros_like(base_b)
        realized_disp_norm = float(
            torch.sqrt(realized_dw.double().norm() ** 2 + realized_db.double().norm() ** 2).item()
        )
        weight_disp_norm = float(realized_dw.double().norm().item())
        bias_disp_norm = float(realized_db.double().norm().item())

        eff_alpha = alpha * chosen_scale
        linearized_after = (
            e_base - b @ eff_alpha + 0.5 * np.einsum("i,gij,j->g", eff_alpha, A, eff_alpha)
        ).tolist()

        before_m = base_margins.get("mean_true_margin")
        after_m = after_margins.get("mean_true_margin")
        margin_change = (
            float(after_m - before_m)
            if before_m is not None and after_m is not None
            else 0.0
        )

        # Margin decreased fraction on observed valid voxels
        if valid_true_before is not None and valid_true_after is not None and valid_true_before.numel() > 0:
            decreased = valid_true_after < (valid_true_before - 1e-7)
            margin_decreased_fraction = float(decreased.float().mean().item())
        else:
            margin_decreased_fraction = 0.0

        report.update({
            "alpha": alpha.tolist(),
            "scale": chosen_scale,
            "accepted": bool(chosen_scale > 0),
            "status": status,
            "after_brier": after_e.tolist(),
            "realized_max_brier_regression": float(np.max(after_e - e_base)),
            "linearized_after_brier": linearized_after,
            "proposed_displacement_norm": proposed_disp_norm,
            "realized_displacement_norm": realized_disp_norm,
            "weight_displacement_norm": weight_disp_norm,
            "bias_displacement_norm": bias_disp_norm,
            "cache_parity_max_error": cache_parity_error,
            "macro_dice_after": after_seg["macro_dice"],
            "macro_absent_fp_after": after_seg["macro_absent_fp"],
            "macro_fg_dice_after": after_seg["foreground_macro_dice"],
            "margins_after": after_margins,
            "margin_change": margin_change,
            "margin_decreased_fraction": margin_decreased_fraction,
            "trials": trials,
            "patient_metrics_before": base_seg["patient_metrics"],
            "patient_metrics_after": after_seg["patient_metrics"],
        })

        # Strict JSON check with allow_nan=False
        try:
            json.dumps(report, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"control diagnostics must be JSON-safe without NaN: {exc}") from exc

        return report
