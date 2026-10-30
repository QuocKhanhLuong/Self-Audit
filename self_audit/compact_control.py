"""Whole-model class soft KD candidate displacement with common denominator and audited trials.

Implements the contract for Worker audit from COMPACT_IMPLEMENTATION_CONTRACT.md
with strict adherence to coordinator integration reviews:
  - build_class_candidates:
      Functional forward with cloned buffers and per-module mode preservation.
      Soft CE T^2 * coefficient normalized by COMMON valid teacher-confident pixels.
      Grouped by teacher argmax into up to 4 classes.
      COMMON aggregate scaling under active norm bound (preserves plain KD sum).
      Detached finite sum-to-1 teacher probabilities and 0 <= confidence <= 1.
      Does not mutate parameter .grad or model buffers.
  - apply_compact_update:
      Post-supervised reference on audit support.
      Methods: 'supervised', 'plain_kd', 'scalar_guard', 'joint_guard'.
      Repeatable callable audit_batches streaming whole labeled frames.
      Audit input validation: x BCHW, y BHW int canonical 0..3/255, valid bool, frame/patient length B.
      Audit identity hashing across replays to detect non-deterministic audit factories.
      Rejects empty, all-ignored, or non-finite audit references.
      Hard multiclass Dice per frame then patient class averages.
      Separate absent-frame FP checks with fixed observed valid-voxel count denominator.
      Strict per-class and per-patient guards; absent FP per frame cannot offset.
      Enforces aggregate norm cap for EVERY trial vector; logs norm bound rejections.
      Finite deterministic search under identical trial cap (default max_trials=8).
      Scalar common alpha (alpha = s * 1) and joint vector candidates in box [0, 1]; M=1 identical.
      Functional trials with FRESH cloned buffers each evaluation via torch.func.functional_call.
      Streamed margin sum/count/min without retaining volume GPU or CPU tensor lists.
      Honest mean_margin_decreased boolean and null/unavailable margin_decreased_fraction.
      Zero cancellation correctly yields status 'zero_displacement'.
      Receipt validation (strict JSON without NaN) BEFORE direct copy (exception-atomic).
      Atomic state and RNG restoration: module parameters, .grad, per-module modes, and
      Python/NumPy/Torch/CUDA RNG are strictly preserved on rejection.
      Statuses: 'no_candidate', 'zero_displacement', 'rejected_all', 'accepted_nonzero'.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F

from self_audit.compact_utility import (
    DEFAULT_SELECTOR,
    DEFAULT_SELECTOR_DICE_TOLERANCE,
    DEFAULT_UTILITY_GAIN_TOLERANCE,
    DEFAULT_UTILITY_TIE_TOLERANCE,
    VALID_SELECTORS,
    compute_probe_kd_loss,
    select_larger_tie,
    select_utility,
    validate_utility_probe,
)

__all__ = [
    "CLASS_NAMES",
    "DEFAULT_SELECTOR",
    "DEFAULT_SELECTOR_DICE_TOLERANCE",
    "DEFAULT_UTILITY_GAIN_TOLERANCE",
    "DEFAULT_UTILITY_TIE_TOLERANCE",
    "FOREGROUND_CLASS_INDICES",
    "FOREGROUND_CLASS_NAMES",
    "VALID_SELECTORS",
    "apply_compact_update",
    "build_class_candidates",
    "compute_audit_metrics",
    "compute_batch_margins",
    "compute_probe_kd_loss",
    "select_larger_tie",
    "select_utility",
    "validate_utility_probe",
]

CLASS_NAMES = ("BG", "RV", "MYO", "LV")
FOREGROUND_CLASS_INDICES = (1, 2, 3)
FOREGROUND_CLASS_NAMES = ("RV", "MYO", "LV")


def _supports_valid_mask(model: nn.Module) -> bool:
    """Check whether model.forward explicitly declares a valid_mask parameter."""
    try:
        sig = inspect.signature(model.forward)
        return "valid_mask" in sig.parameters
    except (TypeError, ValueError):
        return False


def _call_model_functional(
    model: nn.Module,
    params: Dict[str, Tensor],
    buffers: Dict[str, Tensor],
    x: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Call model functional forward pass, passing valid_mask when supported by model.forward.

    Exception handling is restricted to signature inspection only; model execution exceptions propagate.
    """
    if valid_mask is not None and _supports_valid_mask(model):
        return functional_call(model, (params, buffers), (x,), {"valid_mask": valid_mask})
    return functional_call(model, (params, buffers), (x,))


class _ExecutionStateContext:
    """Saves and restores Python, NumPy, PyTorch CPU and CUDA RNGs, and per-module training modes."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.py_rng = random.getstate()
        self.np_rng = np.random.get_state()
        self.torch_cpu_rng = torch.get_rng_state().clone()
        self.torch_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.module_modes = {mod: mod.training for mod in model.modules()}

    def set_eval(self) -> None:
        for mod in self.model.modules():
            mod.eval()

    def restore(self) -> None:
        for mod, mode in self.module_modes.items():
            mod.train(mode)
        random.setstate(self.py_rng)
        np.random.set_state(self.np_rng)
        torch.set_rng_state(self.torch_cpu_rng)
        if self.torch_cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.torch_cuda_rng)


def build_class_candidates(
    model: nn.Module,
    x: Tensor,
    targets: Tensor,
    valid: Tensor,
    *,
    lr: float,
    temperature: float = 1.0,
    coefficient: float = 1.0,
    confidence: float = 0.0,
    max_norm: Optional[float] = None,
) -> Dict[str, Any]:
    """Build whole-model class soft KD candidate displacements normalized by common denominator.

    Args:
        model: Student model returning B4HW logits.
        x: Input image tensor (B, C, H, W).
        targets: Teacher probability tensor (B, 4, H, W).
        valid: Boolean validity mask (B, H, W).
        lr: Positive learning rate.
        temperature: Distillation temperature (default 1.0).
        coefficient: Overall loss multiplier (default 1.0).
        confidence: Minimum teacher max-probability threshold in [0, 1] (default 0.0).
        max_norm: Optional maximum L2 displacement norm applied to aggregate plain KD sum.

    Returns:
        Dictionary with:
          - directions: list of dicts mapping param_name -> displacement Tensor
          - source_ids: list of str (e.g. ['pseudo_class_0', ...])
          - denominator: total valid confident pixels across all classes
          - telemetry: dict with diagnostics
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(x, Tensor) or x.ndim != 4:
        raise ValueError("x must be a 4D Tensor (BCHW)")
    if not isinstance(targets, Tensor) or targets.ndim != 4 or targets.shape[1] != 4:
        raise ValueError("targets must be a 4D Tensor (B4HW)")
    if not isinstance(valid, Tensor) or valid.dtype != torch.bool or valid.ndim != 3:
        raise ValueError("valid must be a 3D boolean Tensor (BHW)")

    B, _, H, W = targets.shape
    if x.shape[0] != B or x.shape[2] != H or x.shape[3] != W:
        raise ValueError("spatial and batch dimensions of x and targets must match")
    if valid.shape != (B, H, W):
        raise ValueError("valid mask shape must match (B, H, W)")

    if not isinstance(lr, (int, float)) or lr <= 0 or not np.isfinite(lr):
        raise ValueError("lr must be a positive finite float")
    if not isinstance(temperature, (int, float)) or temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be a positive finite float")
    if not isinstance(coefficient, (int, float)) or coefficient < 0 or not np.isfinite(coefficient):
        raise ValueError("coefficient must be a non-negative finite float")
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0) or not np.isfinite(confidence):
        raise ValueError("confidence must be a finite float in [0.0, 1.0]")
    if max_norm is not None:
        if not isinstance(max_norm, (int, float)) or max_norm <= 0 or not np.isfinite(max_norm):
            raise ValueError("max_norm must be a positive finite float if provided")

    # Validate teacher targets
    targets = targets.detach()
    if not torch.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("teacher targets must be non-negative and finite")

    prob_sums = targets.sum(dim=1)
    if valid.any():
        if not torch.allclose(prob_sums[valid], torch.ones_like(prob_sums[valid]), atol=1e-3):
            raise ValueError("teacher targets must be probabilities summing to 1 across class dimension")

    trainable_named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable_named_params:
        raise ValueError("model has no trainable parameters (requires_grad=True)")

    trainable_params = [p for _, p in trainable_named_params]
    param_names = [name for name, _ in trainable_named_params]

    teacher_max, teacher_argmax = targets.max(dim=1)
    confident_mask = (teacher_max >= confidence) & valid
    denominator = int(confident_mask.sum().item())

    telemetry: Dict[str, Any] = {
        "accepted_pixels": denominator,
        "total_valid_pixels": int(valid.sum().item()),
        "confidence_threshold": float(confidence),
        "temperature": float(temperature),
        "coefficient": float(coefficient),
        "lr": float(lr),
        "max_norm": float(max_norm) if max_norm is not None else None,
        "class_pixels": [],
        "source_ids": [],
        "status": "NO_CANDIDATES" if denominator == 0 or coefficient == 0 else "CANDIDATES_BUILT",
    }

    if denominator == 0 or coefficient == 0.0:
        telemetry["class_pixels"] = [0, 0, 0, 0]
        return {
            "directions": [],
            "source_ids": [],
            "denominator": denominator,
            "telemetry": telemetry,
        }

    # Functional forward pass with cloned buffers and per-module mode preservation
    state_ctx = _ExecutionStateContext(model)
    try:
        state_ctx.set_eval()
        all_params = dict(model.named_parameters())
        fresh_buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
        logits = _call_model_functional(model, all_params, fresh_buffers, x, valid_mask=valid)

        if not isinstance(logits, Tensor) or logits.shape != targets.shape:
            raise ValueError(f"model output logits shape {logits.shape} does not match targets shape {targets.shape}")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite logits produced by model functional forward pass")

        scaled_logits = logits / temperature
        log_probs = F.log_softmax(scaled_logits, dim=1)

        if abs(temperature - 1.0) > 1e-7:
            soft_targets = F.softmax(torch.log(targets.clamp(min=1e-12)) / temperature, dim=1)
        else:
            soft_targets = targets

        per_pixel_loss = - (soft_targets * log_probs).sum(dim=1) * ((temperature ** 2) * coefficient)

        raw_directions: List[Dict[str, Tensor]] = []
        source_ids: List[str] = []
        class_pixels: List[int] = []

        for c in range(4):
            mask_c = confident_mask & (teacher_argmax == c)
            c_count = int(mask_c.sum().item())
            class_pixels.append(c_count)
            if c_count > 0:
                # Normalized by the COMMON denominator across all classes:
                # sum_c loss_c = sum_{i in M} loss_i / D = loss_plain_kd
                loss_c = (per_pixel_loss * mask_c).sum() / denominator
                grads = torch.autograd.grad(loss_c, trainable_params, retain_graph=True)

                disp_dict: Dict[str, Tensor] = {}
                for name, g in zip(param_names, grads):
                    if not torch.isfinite(g).all():
                        raise FloatingPointError(f"non-finite gradient for parameter {name} in class {c}")
                    disp_dict[name] = (-lr * g.detach()).clone()

                raw_directions.append(disp_dict)
                source_ids.append(f"pseudo_class_{c}")

        # COMMON aggregate scaling under active norm bound:
        # Scale all candidate displacements by the SAME factor so sum_c d_c exactly matches clipped plain KD
        if raw_directions and max_norm is not None:
            sum_disp: Dict[str, Tensor] = {
                name: sum(d[name] for d in raw_directions) for name in param_names
            }
            total_norm_sq = sum(t.double().norm() ** 2 for t in sum_disp.values())
            total_disp_norm = float(torch.sqrt(total_norm_sq).item())
            if total_disp_norm > max_norm:
                common_scale = max_norm / (total_disp_norm + 1e-12)
                directions = [{k: (v * common_scale).clone() for k, v in d.items()} for d in raw_directions]
                telemetry["aggregate_clip_scale"] = float(common_scale)
            else:
                directions = raw_directions
                telemetry["aggregate_clip_scale"] = 1.0
        else:
            directions = raw_directions
            telemetry["aggregate_clip_scale"] = 1.0

        telemetry["class_pixels"] = class_pixels
        telemetry["source_ids"] = source_ids

        return {
            "directions": directions,
            "source_ids": source_ids,
            "denominator": denominator,
            "telemetry": telemetry,
        }
    finally:
        state_ctx.restore()


def compute_batch_margins(
    logits: Tensor,
    labels: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> Tuple[Dict[str, Optional[float]], Optional[Tensor]]:
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


def compute_audit_metrics(
    frame_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate frame records into patient-level and macro-level Dice and absent-frame FP metrics.

    Frame records contain per-frame-volume metrics for each class:
      - If GT > 0: Dice is 2*TP/(GT+Pred), absent_fp_rate is None.
      - If GT == 0: Dice is None, absent_fp_rate is FP / valid_voxels (fixed valid count denominator).
      - If valid_voxels == 0: both are None.

    Aggregation:
      1. Average over defined frames within each patient.
      2. Average patient means over defined patients for each class.
      3. Macro foreground Dice is mean of defined foreground classes (1, 2, 3).
    """
    unique_patients = list(dict.fromkeys(r["patient"] for r in frame_records))
    patient_metrics: Dict[str, Any] = {}

    for pid in unique_patients:
        p_frames = [r for r in frame_records if r["patient"] == pid]
        p_dice: Dict[str, Optional[float]] = {}
        p_absent_fp: Dict[str, Optional[float]] = {}

        for c_name in FOREGROUND_CLASS_NAMES:
            dices = [
                f["classes"][c_name]["dice"]
                for f in p_frames
                if f["classes"][c_name]["dice"] is not None
            ]
            p_dice[c_name] = float(np.mean(dices)) if dices else None

            fps = [
                f["classes"][c_name]["absent_fp_rate"]
                for f in p_frames
                if f["classes"][c_name]["absent_fp_rate"] is not None
            ]
            p_absent_fp[c_name] = float(np.mean(fps)) if fps else None

        patient_metrics[str(pid)] = {
            "dice": p_dice,
            "absent_fp_rate": p_absent_fp,
        }

    macro_dice: Dict[str, Optional[float]] = {}
    macro_absent_fp: Dict[str, Optional[float]] = {}

    for c_name in FOREGROUND_CLASS_NAMES:
        p_dices = [
            patient_metrics[str(pid)]["dice"][c_name]
            for pid in unique_patients
            if patient_metrics[str(pid)]["dice"][c_name] is not None
        ]
        macro_dice[c_name] = float(np.mean(p_dices)) if p_dices else None

        p_fps = [
            patient_metrics[str(pid)]["absent_fp_rate"][c_name]
            for pid in unique_patients
            if patient_metrics[str(pid)]["absent_fp_rate"][c_name] is not None
        ]
        macro_absent_fp[c_name] = float(np.mean(p_fps)) if p_fps else None

    defined_macros = [d for d in macro_dice.values() if d is not None]
    foreground_macro_dice = float(np.mean(defined_macros)) if defined_macros else None

    return {
        "macro_dice": macro_dice,
        "macro_absent_fp": macro_absent_fp,
        "foreground_macro_dice": foreground_macro_dice,
        "patient_metrics": patient_metrics,
        "frame_records": frame_records,
    }


def _evaluate_model_on_audit_streamed(
    model: nn.Module,
    param_dict: Dict[str, Tensor],
    audit_batches: Callable[[], Iterable[Mapping[str, Any]]],
    expected_identity_hash: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Optional[float]], str]:
    """Stream whole audit frames through functional_call with fresh buffers per batch.

    Maintains scalar margin accumulators without holding lists of GPU or CPU tensors.
    Computes an audit identity hash across batches to detect changing audit factories.

    Returns:
        (metrics_dict, margins_summary, identity_hash)
    """
    hasher = hashlib.sha256()
    frame_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    margin_count = 0
    margin_sum = 0.0
    margin_min = float("inf")
    batch_idx = 0

    for batch in audit_batches():
        if not isinstance(batch, Mapping):
            raise TypeError("each audit batch must be a mapping with x, y, valid, frame, patient")
        for req_k in ("x", "y", "frame", "patient"):
            if req_k not in batch:
                raise ValueError(f"audit batch missing required key '{req_k}'")

        x = batch["x"]
        y = batch["y"]
        raw_frame = batch["frame"]
        raw_patient = batch["patient"]

        if not isinstance(x, Tensor) or x.ndim != 4:
            raise ValueError("x must be a 4D Tensor (BCHW)")
        if not torch.isfinite(x).all():
            raise FloatingPointError("non-finite values in audit input x")

        B, _, H, W = x.shape

        if not isinstance(y, Tensor) or y.ndim != 3:
            raise ValueError("y must be a 3D Tensor (BHW)")
        if y.shape != (B, H, W):
            raise ValueError(f"y shape {y.shape} does not match x batch/spatial shape {(B, H, W)}")
        if y.dtype.is_floating_point:
            raise TypeError(f"y must be an integer tensor, got {y.dtype}")
        if not torch.isin(y, y.new_tensor([0, 1, 2, 3, 255])).all():
            raise ValueError("y ontology must strictly contain only canonical values {0, 1, 2, 3, 255}")

        if "valid" in batch:
            valid = batch["valid"]
            if not isinstance(valid, Tensor) or valid.dtype != torch.bool:
                raise TypeError("valid must be a boolean Tensor")
            if valid.shape != (B, H, W):
                raise ValueError(f"valid mask shape {valid.shape} does not match (B, H, W) {(B, H, W)}")
        else:
            valid = y != 255

        if isinstance(raw_patient, str):
            patient_list = [raw_patient] * B
        elif isinstance(raw_patient, (list, tuple)):
            if len(raw_patient) != B:
                raise ValueError(f"patient list length {len(raw_patient)} must match batch size {B}")
            patient_list = [str(p) for p in raw_patient]
        else:
            raise TypeError("patient must be a string or sequence of strings matching batch size")

        if isinstance(raw_frame, (int, np.integer)):
            frame_list = [int(raw_frame)] * B
        elif isinstance(raw_frame, (list, tuple)):
            if len(raw_frame) != B:
                raise ValueError(f"frame list length {len(raw_frame)} must match batch size {B}")
            frame_list = [int(f) for f in raw_frame]
        elif isinstance(raw_frame, Tensor) and raw_frame.ndim == 1:
            if len(raw_frame) != B:
                raise ValueError(f"frame tensor length {len(raw_frame)} must match batch size {B}")
            frame_list = [int(f.item()) for f in raw_frame]
        else:
            raise TypeError("frame must be an integer or sequence of integers matching batch size")

        # Update running identity hash of the complete audit stream
        hasher.update(str(patient_list).encode("utf-8"))
        hasher.update(str(frame_list).encode("utf-8"))
        if "slice" in batch:
            hasher.update(str(batch["slice"]).encode("utf-8"))
        if "slices" in batch:
            hasher.update(str(batch["slices"]).encode("utf-8"))
        hasher.update(str(x.dtype).encode("utf-8"))
        hasher.update(str(x.shape).encode("utf-8"))
        hasher.update(str(y.dtype).encode("utf-8"))
        hasher.update(str(y.shape).encode("utf-8"))
        hasher.update(str(valid.dtype).encode("utf-8"))
        hasher.update(str(valid.shape).encode("utf-8"))
        hasher.update(x.detach().cpu().numpy().tobytes())
        hasher.update(y.detach().cpu().numpy().tobytes())
        hasher.update(valid.detach().cpu().numpy().tobytes())

        # Fresh cloned buffers for EVERY batch evaluation
        fresh_buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
        with torch.no_grad():
            logits = _call_model_functional(model, param_dict, fresh_buffers, x, valid_mask=valid)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite logits produced during audit trial forward")

            batch_margins, valid_true = compute_batch_margins(logits, y, valid)
            if valid_true is not None and valid_true.numel() > 0:
                n_v = valid_true.numel()
                margin_count += n_v
                margin_sum += float(valid_true.sum().item())
                m_min = float(valid_true.min().item())
                if m_min < margin_min:
                    margin_min = m_min

            preds = logits.argmax(dim=1)
            preds_np = preds.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            valid_np = (y_np != 255) & valid.detach().cpu().numpy().astype(bool)

            for i in range(B):
                pid = patient_list[i]
                fid = frame_list[i]
                key = (pid, fid)
                if key not in frame_map:
                    frame_map[key] = {
                        "patient": pid,
                        "frame": fid,
                        "classes": {
                            c_name: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0, "valid_voxels": 0}
                            for c_name in CLASS_NAMES
                        },
                    }

                rec = frame_map[key]
                v_slice = valid_np[i]
                p_slice = preds_np[i]
                y_slice = y_np[i]
                v_count = int(v_slice.sum())

                for c_idx, c_name in enumerate(CLASS_NAMES):
                    rec["classes"][c_name]["valid_voxels"] += v_count
                    gt_mask = (y_slice == c_idx) & v_slice
                    pred_mask = (p_slice == c_idx) & v_slice
                    tp = int((gt_mask & pred_mask).sum())
                    gt_count = int(gt_mask.sum())
                    pred_count = int(pred_mask.sum())

                    rec["classes"][c_name]["gt"] += gt_count
                    rec["classes"][c_name]["pred"] += pred_count
                    rec["classes"][c_name]["tp"] += tp
                    rec["classes"][c_name]["fp"] += (pred_count - tp)
                    rec["classes"][c_name]["fn"] += (gt_count - tp)

        batch_idx += 1

    stream_hash = hasher.hexdigest()
    if expected_identity_hash is not None and stream_hash != expected_identity_hash:
        raise ValueError("audit_batches callable produced inconsistent data across replays (identity hash mismatch)")

    if batch_idx == 0:
        raise ValueError("audit_batches callable yielded zero batches (empty audit support)")

    # Build per-frame records
    frame_records: List[Dict[str, Any]] = []
    total_valid_across_frames = 0
    for key, rec in frame_map.items():
        frame_rec: Dict[str, Any] = {"patient": rec["patient"], "frame": rec["frame"], "classes": {}}
        for c_name in CLASS_NAMES:
            c_info = rec["classes"][c_name]
            v_vox = c_info["valid_voxels"]
            gt = c_info["gt"]
            pred = c_info["pred"]
            tp = c_info["tp"]
            fp = c_info["fp"]

            if v_vox == 0:
                dice = None
                absent_fp = None
            elif gt > 0:
                dice = float(2.0 * tp / (gt + pred))
                absent_fp = None
                total_valid_across_frames += v_vox
            else:
                dice = None
                absent_fp = float(fp / v_vox)  # fixed valid-voxel count denominator
                total_valid_across_frames += v_vox

            frame_rec["classes"][c_name] = {
                "tp": tp,
                "fp": fp,
                "fn": c_info["fn"],
                "gt": gt,
                "pred": pred,
                "valid_voxels": v_vox,
                "dice": dice,
                "absent_fp_rate": absent_fp,
            }
        frame_records.append(frame_rec)

    if total_valid_across_frames == 0:
        raise ValueError("all pixels across audit support are ignored (valid_voxels == 0)")

    metrics = compute_audit_metrics(frame_records)

    margins_summary = {
        "mean_true_margin": float(margin_sum / margin_count) if margin_count > 0 else None,
        "min_true_margin": float(margin_min) if margin_count > 0 else None,
    }

    return metrics, margins_summary, stream_hash


def _validate_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Strict configuration validation: check unknown keys, types, finite non-negative values."""
    if config is None:
        return {
            "eps": 0.0,
            "absent_fp_budget": 0.0,
            "tolerance": 1e-6,
            "ridge": 1e-6,
            "max_trials": 8,
            "trial_alphas": None,
            "max_norm": None,
            "selector": DEFAULT_SELECTOR,
            "selector_dice_tolerance": DEFAULT_SELECTOR_DICE_TOLERANCE,
            "utility_gain_tolerance": DEFAULT_UTILITY_GAIN_TOLERANCE,
            "utility_tie_tolerance": DEFAULT_UTILITY_TIE_TOLERANCE,
        }
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    allowed_keys = {
        "eps",
        "absent_fp_budget",
        "tolerance",
        "ridge",
        "max_trials",
        "trial_alphas",
        "joint_candidates",
        "max_norm",
        "selector",
        "selector_dice_tolerance",
        "utility_gain_tolerance",
        "utility_tie_tolerance",
    }
    unknown = set(config.keys()) - allowed_keys
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")

    validated: Dict[str, Any] = {}
    for k in ("eps", "absent_fp_budget", "tolerance"):
        val = config.get(k, 0.0)
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not np.isfinite(val) or val < 0:
            raise ValueError(f"config key '{k}' must be a finite non-negative float, got {val}")
        validated[k] = float(val)

    ridge = config.get("ridge", 1e-6)
    if not isinstance(ridge, (int, float)) or isinstance(ridge, bool) or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError(f"config key 'ridge' must be a finite positive float, got {ridge}")
    validated["ridge"] = float(ridge)

    max_trials = config.get("max_trials", 8)
    if not isinstance(max_trials, int) or isinstance(max_trials, bool) or max_trials < 1:
        raise ValueError(f"config key 'max_trials' must be a positive integer, got {max_trials}")
    validated["max_trials"] = max_trials

    if "max_norm" in config and config["max_norm"] is not None:
        mn = config["max_norm"]
        if not isinstance(mn, (int, float)) or isinstance(mn, bool) or not np.isfinite(mn) or mn <= 0:
            raise ValueError(f"config key 'max_norm' must be a finite positive float, got {mn}")
        validated["max_norm"] = float(mn)
    else:
        validated["max_norm"] = None

    trial_alphas = config.get("trial_alphas", config.get("joint_candidates", None))
    if trial_alphas is not None:
        if not isinstance(trial_alphas, (list, tuple)):
            raise TypeError("trial_alphas must be a list or tuple of candidate vectors")
        validated["trial_alphas"] = trial_alphas
    else:
        validated["trial_alphas"] = None

    selector = config.get("selector", DEFAULT_SELECTOR)
    if not isinstance(selector, str):
        raise TypeError(f"config key 'selector' must be a string, got {type(selector)}")
    if selector not in VALID_SELECTORS:
        raise ValueError(f"unknown selector '{selector}', expected one of {sorted(VALID_SELECTORS)}")
    validated["selector"] = selector

    for k, default_val in (
        ("selector_dice_tolerance", DEFAULT_SELECTOR_DICE_TOLERANCE),
        ("utility_gain_tolerance", DEFAULT_UTILITY_GAIN_TOLERANCE),
        ("utility_tie_tolerance", DEFAULT_UTILITY_TIE_TOLERANCE),
    ):
        val = config.get(k, default_val)
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not np.isfinite(val) or val < 0:
            raise ValueError(f"config key '{k}' must be a finite non-negative float, got {val}")
        validated[k] = float(val)

    return validated


def _generate_candidate_alphas(
    method: str,
    num_directions: int,
    source_ids: Sequence[str],
    max_trials: int,
    config_alphas: Optional[Sequence[Any]] = None,
) -> List[np.ndarray]:
    """Generate and validate finite deterministic candidate alpha vectors in box [0, 1]."""
    M = num_directions
    if M == 0:
        return []

    if config_alphas is not None:
        result: List[np.ndarray] = []
        for a in config_alphas[:max_trials]:
            arr = np.asarray(a, dtype=np.float64)
            if arr.ndim != 1 or len(arr) != M:
                raise ValueError(f"candidate alpha must have length {M}, got shape {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError("candidate alpha contains non-finite values")
            if (arr < 0.0).any() or (arr > 1.0).any():
                raise ValueError(f"candidate alpha values must lie in box [0, 1], got {arr}")
            if method == "scalar_guard" and not np.allclose(arr, arr[0]):
                raise ValueError(f"scalar_guard requires common scalar alpha = s * 1, got {arr}")
            result.append(arr)
        return result

    base_scales = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125][:max_trials]

    if method == "plain_kd":
        return [np.ones(M, dtype=np.float64)]

    if method == "scalar_guard" or M == 1:
        # Scalar common alpha vector: alpha = s * 1; M=1 identical for joint_guard
        return [np.full(M, s, dtype=np.float64) for s in base_scales]

    # method == "joint_guard" and M > 1:
    joint_candidates: List[np.ndarray] = []
    # 1-4: Uniform scalings
    joint_candidates.append(np.ones(M, dtype=np.float64))
    joint_candidates.append(np.full(M, 0.5, dtype=np.float64))
    joint_candidates.append(np.full(M, 0.25, dtype=np.float64))
    joint_candidates.append(np.full(M, 0.125, dtype=np.float64))

    # 5-8: Class-differentiated joint vector alternatives (with or without background)
    if "pseudo_class_0" in source_ids:
        bg_idx = source_ids.index("pseudo_class_0")
        c5 = np.ones(M, dtype=np.float64)
        c5[bg_idx] = 0.25
        joint_candidates.append(c5)

        c6 = np.ones(M, dtype=np.float64)
        c6[bg_idx] = 0.5
        joint_candidates.append(c6)

        c7 = np.ones(M, dtype=np.float64)
        c7[bg_idx] = 0.0
        joint_candidates.append(c7)

        c8 = np.full(M, 0.5, dtype=np.float64)
        c8[bg_idx] = 0.0
        joint_candidates.append(c8)
    else:
        # Diverse alternatives when no BG direction is present
        c5 = np.full(M, 0.5, dtype=np.float64)
        c5[0] = 1.0
        joint_candidates.append(c5)

        c6 = np.full(M, 0.5, dtype=np.float64)
        c6[-1] = 1.0
        joint_candidates.append(c6)

        c7 = np.full(M, 0.75, dtype=np.float64)
        joint_candidates.append(c7)

        c8 = np.full(M, 0.375, dtype=np.float64)
        joint_candidates.append(c8)

    return joint_candidates[:max_trials]


def apply_compact_update(
    model: nn.Module,
    candidates: Union[Dict[str, Any], Sequence[Dict[str, Tensor]]],
    *,
    method: str,
    audit_batches: Optional[Callable[[], Iterable[Mapping[str, Any]]]] = None,
    config: Optional[Dict[str, Any]] = None,
    utility_probe: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Audited whole-network parameter update with post-supervised reference and atomic state.

    Args:
        model: Compact student model.
        candidates: Return dictionary from build_class_candidates or list of direction dicts.
        method: One of 'supervised', 'plain_kd', 'scalar_guard', 'joint_guard'.
        audit_batches: Repeatable callable yielding mappings with 'x', 'y', 'valid', 'frame', 'patient'.
        config: Optional configuration dictionary with keys:
            - eps: float >= 0 (Dice budget, default 0.0)
            - absent_fp_budget: float >= 0 (absent FP budget, default 0.0)
            - tolerance: float >= 0 (numerical tolerance, default 1e-6)
            - ridge: float > 0 (ridge regularizer, default 1e-6)
            - max_trials: int >= 1 (maximum candidate trials, default 8)
            - max_norm: optional aggregate displacement norm bound
            - trial_alphas: optional custom sequence of alpha vectors in box [0, 1]
            - selector: str ('original', 'larger_tie', 'cross_patient_utility', 'same_batch_utility')
            - selector_dice_tolerance: float >= 0 (default 0.0)
            - utility_gain_tolerance: float >= 0 (default 1e-7)
            - utility_tie_tolerance: float >= 0 (default 1e-7)
        utility_probe: Optional mapping with detached tensors 'x', 'targets', 'valid', and
            optional 'temperature', 'coefficient', 'confidence', 'metadata'. Sampled once per transaction.

    Returns:
        JSON-safe receipt dictionary with update diagnostics and explicit statuses:
        'no_candidate', 'zero_displacement', 'rejected_all', 'accepted_nonzero'.
    """
    valid_methods = {"supervised", "plain_kd", "scalar_guard", "joint_guard"}
    if method not in valid_methods:
        raise ValueError(f"unknown method '{method}', expected one of {valid_methods}")

    cfg = _validate_config(config)
    eps = cfg["eps"]
    absent_fp_budget = cfg["absent_fp_budget"]
    tolerance = cfg["tolerance"]
    ridge = cfg["ridge"]
    max_trials = cfg["max_trials"]
    config_alphas = cfg["trial_alphas"]
    selector = cfg["selector"]
    selector_dice_tolerance = cfg["selector_dice_tolerance"]
    utility_gain_tolerance = cfg["utility_gain_tolerance"]
    utility_tie_tolerance = cfg["utility_tie_tolerance"]

    if selector != DEFAULT_SELECTOR and method != "scalar_guard":
        raise ValueError(f"selector '{selector}' is restricted to method 'scalar_guard', got method '{method}'")

    # Extract aggregate max_norm from candidates telemetry or config
    max_norm = None
    if isinstance(candidates, dict):
        max_norm = candidates.get("telemetry", {}).get("max_norm", None)
    if max_norm is None:
        max_norm = cfg["max_norm"]

    # Unpack candidates
    if isinstance(candidates, dict):
        directions: List[Dict[str, Tensor]] = candidates.get("directions", [])
        source_ids: List[str] = candidates.get("source_ids", [])
        denominator = int(candidates.get("denominator", 0))
    else:
        directions = list(candidates)
        source_ids = [f"direction_{i}" for i in range(len(directions))]
        denominator = 1 if directions else 0

    m = len(directions)
    if m > 4:
        raise ValueError(f"directions count {m} exceeds maximum allowed 4 classes")
    if len(source_ids) != m:
        raise ValueError(f"source_ids count {len(source_ids)} must match directions count {m}")
    if len(set(source_ids)) != m:
        raise ValueError(f"source_ids must be unique, got {source_ids}")

    # Validate direction tensor shapes, dtypes, devices, and finiteness
    model_param_dict = dict(model.named_parameters())
    for idx_d, d_dict in enumerate(directions):
        if not isinstance(d_dict, dict):
            raise TypeError(f"direction at index {idx_d} must be a dictionary")
        for p_name, d_tensor in d_dict.items():
            if p_name not in model_param_dict:
                raise KeyError(f"displacement parameter '{p_name}' not in model parameters")
            p_ref = model_param_dict[p_name]
            if not isinstance(d_tensor, Tensor):
                raise TypeError(f"displacement for '{p_name}' must be a torch.Tensor, got {type(d_tensor)}")
            if d_tensor.shape != p_ref.shape:
                raise ValueError(f"shape mismatch for '{p_name}': {d_tensor.shape} vs {p_ref.shape}")
            if d_tensor.dtype != p_ref.dtype:
                raise TypeError(f"dtype mismatch for '{p_name}': {d_tensor.dtype} vs {p_ref.dtype}")
            if d_tensor.device != p_ref.device:
                raise ValueError(f"device mismatch for '{p_name}': {d_tensor.device} vs {p_ref.device}")
            if not torch.isfinite(d_tensor).all():
                raise FloatingPointError(f"non-finite displacement for parameter '{p_name}'")

    # Fast-path for empty candidates or supervised mode
    if m == 0 or denominator == 0 or method == "supervised":
        fast_reason = "supervised_mode" if method == "supervised" else "no_candidate"
        fast_counts = {
            "audit_stream_batches": 0,
            "audit_reference_evaluations": 0,
            "nonzero_trial_evaluations": 0,
            "probe_reference_evaluations": 0,
            "probe_trial_evaluations": 0,
        }
        fast_timings = {
            "audit_reference_seconds": 0.0,
            "audit_trials_seconds": 0.0,
            "probe_reference_seconds": 0.0,
            "probe_trials_seconds": 0.0,
            "total_seconds": 0.0,
        }
        return {
            "method": method,
            "status": "zero_displacement" if method == "supervised" and m > 0 else "no_candidate",
            "accepted": False,
            "source_ids": list(source_ids),
            "alpha": [],
            "scale": 0.0,
            "proposed_displacement_norm": 0.0,
            "realized_displacement_norm": 0.0,
            "base_macro_dice": {c: None for c in FOREGROUND_CLASS_NAMES},
            "base_macro_absent_fp": {c: None for c in FOREGROUND_CLASS_NAMES},
            "base_fg_dice": None,
            "macro_dice_after": {c: None for c in FOREGROUND_CLASS_NAMES},
            "macro_absent_fp_after": {c: None for c in FOREGROUND_CLASS_NAMES},
            "macro_fg_dice_after": None,
            "margins_before": {"mean_true_margin": None, "min_true_margin": None},
            "margins_after": {"mean_true_margin": None, "min_true_margin": None},
            "margin_change": 0.0,
            "mean_margin_decreased": False,
            "margin_decreased_fraction": None,
            "trials": [],
            "optimizer_steps_in_transaction": 0,
            "ema_updates_in_transaction": 0,
            "selector": selector,
            "reason": fast_reason,
            "original_choice": None,
            "feasible_scales": [],
            "forward_counts": fast_counts,
            "timings": fast_timings,
            "telemetry": {
                "info": "no candidates or supervised-only mode",
                "selector": selector,
                "selector_dice_tolerance": selector_dice_tolerance,
                "utility_gain_tolerance": utility_gain_tolerance,
                "utility_tie_tolerance": utility_tie_tolerance,
                "original_choice": None,
                "feasible_scales": [],
                "forward_counts": fast_counts,
                "timings": fast_timings,
                "reason": fast_reason,
            },
        }

    if audit_batches is None and method in {"scalar_guard", "joint_guard"}:
        raise ValueError("audit_batches callable is required for guarded methods")

    t_total_start = time.monotonic()
    t_audit_ref = 0.0
    t_audit_trials = 0.0
    t_probe_ref = 0.0
    t_probe_trials = 0.0
    audit_stream_batches = 0
    nonzero_trial_evaluations = 0
    probe_reference_evaluations = 0
    probe_trial_evaluations = 0

    # Execution state preservation: RNGs and per-module modes
    state_ctx = _ExecutionStateContext(model)
    base_params = {name: p.detach().clone() for name, p in model.named_parameters()}

    try:
        state_ctx.set_eval()

        # Step 1: Evaluate post-supervised reference baseline on audit support
        def counted_audit():
            nonlocal audit_stream_batches
            for batch in audit_batches():
                audit_stream_batches += 1
                yield batch

        if audit_batches is not None:
            t0_ref = time.monotonic()
            base_metrics, base_margins, audit_hash = _evaluate_model_on_audit_streamed(
                model, base_params, counted_audit
            )
            t_audit_ref = time.monotonic() - t0_ref
            base_macro_dice = base_metrics["macro_dice"]
            base_macro_absent_fp = base_metrics["macro_absent_fp"]
            base_fg_dice = base_metrics["foreground_macro_dice"]
        else:
            base_metrics = None
            base_macro_dice = {c: None for c in FOREGROUND_CLASS_NAMES}
            base_macro_absent_fp = {c: None for c in FOREGROUND_CLASS_NAMES}
            base_fg_dice = None
            base_margins = {"mean_true_margin": None, "min_true_margin": None}
            audit_hash = None

        # Step 2: Generate candidate alpha vectors under max_trials cap
        candidate_alphas = _generate_candidate_alphas(
            method, m, source_ids, max_trials, config_alphas
        )

        guarded = method in {"scalar_guard", "joint_guard"}
        trials_log: List[Dict[str, Any]] = []
        best_trial_idx: Optional[int] = None
        best_objective = float("inf")
        chosen_disp: Optional[Dict[str, Tensor]] = None
        after_metrics = base_metrics
        after_margins = base_margins
        trial_metrics_list: List[Optional[Dict[str, Any]]] = []
        trial_margins_list: List[Optional[Dict[str, Any]]] = []
        combined_displacements: List[Dict[str, Tensor]] = []

        # Step 3: Evaluate all candidate trials under identical cap
        for k, alpha in enumerate(candidate_alphas):
            combined_disp: Dict[str, Tensor] = {}
            for name in base_params:
                param_delta = None
                for a_val, d_dict in zip(alpha, directions):
                    if name in d_dict and abs(float(a_val)) > 1e-12:
                        term = float(a_val) * d_dict[name]
                        param_delta = term if param_delta is None else (param_delta + term)
                if param_delta is not None:
                    combined_disp[name] = param_delta
            combined_displacements.append(combined_disp)

            disp_norm_sq = sum(t.double().norm() ** 2 for t in combined_disp.values())
            disp_norm = float(torch.sqrt(disp_norm_sq).item()) if combined_disp else 0.0

            # Enforce aggregate norm cap for EVERY trial vector
            norm_bound_ok = True
            if max_norm is not None and disp_norm > (max_norm + 1e-7):
                norm_bound_ok = False

            if not norm_bound_ok:
                trial_rec = {
                    "trial_index": k,
                    "alpha": alpha.tolist(),
                    "scale": float(alpha[0]) if len(alpha) > 0 else 0.0,
                    "displacement_norm": disp_norm,
                    "finite": True,
                    "macro_dice": base_macro_dice,
                    "macro_absent_fp": base_macro_absent_fp,
                    "foreground_macro_dice": base_fg_dice,
                    "margins": base_margins,
                    "objective": 0.5 * ridge * float(np.sum(alpha ** 2)),
                    "per_class_dice_ok": True,
                    "absent_fp_ok": True,
                    "norm_bound_ok": False,
                    "guard_passed": False,
                    "accepted": False,
                    "probe_loss": None,
                    "utility_gain": None,
                    "reason": "aggregate_norm_cap_exceeded",
                }
                trials_log.append(trial_rec)
                trial_metrics_list.append(None)
                trial_margins_list.append(None)
                continue

            if disp_norm == 0.0:
                trial_rec = {
                    "trial_index": k,
                    "alpha": alpha.tolist(),
                    "scale": float(alpha[0]) if len(alpha) > 0 else 0.0,
                    "displacement_norm": 0.0,
                    "finite": True,
                    "macro_dice": base_macro_dice,
                    "macro_absent_fp": base_macro_absent_fp,
                    "foreground_macro_dice": base_fg_dice,
                    "margins": base_margins,
                    "objective": 0.5 * ridge * float(np.sum(alpha ** 2)),
                    "per_class_dice_ok": True,
                    "absent_fp_ok": True,
                    "norm_bound_ok": True,
                    "guard_passed": True,
                    "accepted": False,
                    "probe_loss": None,
                    "utility_gain": None,
                    "reason": "zero_displacement",
                }
                trials_log.append(trial_rec)
                trial_metrics_list.append(base_metrics)
                trial_margins_list.append(base_margins)
                if best_trial_idx is None:
                    best_trial_idx = k
                    chosen_disp = combined_disp
                continue

            trial_params = {
                name: (base_params[name] + combined_disp[name] if name in combined_disp else base_params[name])
                for name in base_params
            }

            # Verify resulting parameter values are finite before trial evaluation or commit
            for p_name, t_val in trial_params.items():
                if not torch.isfinite(t_val).all():
                    raise FloatingPointError(f"resulting parameter '{p_name}' contains non-finite values")

            if not guarded:
                # plain_kd: verify evaluation and commit without guard
                if audit_batches is not None:
                    t0_tr = time.monotonic()
                    tr_metrics, tr_margins, _ = _evaluate_model_on_audit_streamed(
                        model, trial_params, counted_audit, expected_identity_hash=audit_hash
                    )
                    t_audit_trials += time.monotonic() - t0_tr
                    nonzero_trial_evaluations += 1
                    tr_dice = tr_metrics["macro_dice"]
                    tr_fp = tr_metrics["macro_absent_fp"]
                    tr_fg = tr_metrics["foreground_macro_dice"]
                else:
                    tr_metrics = None
                    tr_dice = base_macro_dice
                    tr_fp = base_macro_absent_fp
                    tr_fg = base_fg_dice
                    tr_margins = base_margins

                trial_rec = {
                    "trial_index": k,
                    "alpha": alpha.tolist(),
                    "scale": float(alpha[0]) if len(alpha) > 0 else 0.0,
                    "displacement_norm": disp_norm,
                    "finite": True,
                    "macro_dice": tr_dice,
                    "macro_absent_fp": tr_fp,
                    "foreground_macro_dice": tr_fg,
                    "margins": tr_margins,
                    "objective": 0.5 * ridge * float(np.sum(alpha ** 2)),
                    "per_class_dice_ok": True,
                    "absent_fp_ok": True,
                    "norm_bound_ok": True,
                    "guard_passed": True,
                    "accepted": True,
                    "probe_loss": None,
                    "utility_gain": None,
                }
                trials_log.append(trial_rec)
                trial_metrics_list.append(tr_metrics)
                trial_margins_list.append(tr_margins)
                best_trial_idx = k
                chosen_disp = combined_disp
                after_metrics = tr_metrics
                after_margins = tr_margins
                break

            # Guarded evaluation (scalar_guard / joint_guard)
            t0_tr = time.monotonic()
            tr_metrics, tr_margins, _ = _evaluate_model_on_audit_streamed(
                model, trial_params, counted_audit, expected_identity_hash=audit_hash
            )
            t_audit_trials += time.monotonic() - t0_tr
            nonzero_trial_evaluations += 1

            tr_dice = tr_metrics["macro_dice"]
            tr_fp = tr_metrics["macro_absent_fp"]
            tr_fg = tr_metrics["foreground_macro_dice"]

            # Guard check 1: Strict per-class and per-patient Dice (no macro-only accept)
            per_class_dice_ok = True
            for c_name in FOREGROUND_CLASS_NAMES:
                # Patient-level checks
                for pid, p_data in base_metrics["patient_metrics"].items():
                    b_p_d = p_data["dice"].get(c_name)
                    if b_p_d is not None:
                        tr_p_d = tr_metrics["patient_metrics"][pid]["dice"].get(c_name)
                        if tr_p_d is None or tr_p_d < (b_p_d - eps - tolerance):
                            per_class_dice_ok = False
                            break
                if not per_class_dice_ok:
                    break

                # Macro-level check
                b_d = base_macro_dice.get(c_name)
                if b_d is not None:
                    t_d = tr_dice.get(c_name)
                    if t_d is None or t_d < (b_d - eps - tolerance):
                        per_class_dice_ok = False
                        break

            # Guard check 2: Strict absent FP per-frame cannot offset
            absent_fp_ok = True
            base_frame_recs = base_metrics["frame_records"]
            tr_frame_recs = tr_metrics["frame_records"]
            for b_f_rec, tr_f_rec in zip(base_frame_recs, tr_frame_recs):
                for c_name in FOREGROUND_CLASS_NAMES:
                    b_f_fp = b_f_rec["classes"][c_name]["absent_fp_rate"]
                    if b_f_fp is not None:
                        tr_f_fp = tr_f_rec["classes"][c_name]["absent_fp_rate"]
                        if tr_f_fp is None or tr_f_fp > (b_f_fp + absent_fp_budget + tolerance):
                            absent_fp_ok = False
                            break
                if not absent_fp_ok:
                    break

            # Guard check 3: Overall foreground macro Dice
            fg_ok = True
            if base_fg_dice is not None:
                if tr_fg is None or tr_fg < (base_fg_dice - eps - tolerance):
                    fg_ok = False

            guard_passed = bool(norm_bound_ok and per_class_dice_ok and absent_fp_ok and fg_ok)

            fg_val = tr_fg if tr_fg is not None else 0.0
            obj = float((1.0 - fg_val) + 0.5 * ridge * float(np.sum(alpha ** 2)))

            trial_rec = {
                "trial_index": k,
                "alpha": alpha.tolist(),
                "scale": float(alpha[0]) if len(alpha) > 0 else 0.0,
                "displacement_norm": disp_norm,
                "finite": True,
                "macro_dice": tr_dice,
                "macro_absent_fp": tr_fp,
                "foreground_macro_dice": tr_fg,
                "margins": tr_margins,
                "objective": obj,
                "per_class_dice_ok": per_class_dice_ok,
                "absent_fp_ok": absent_fp_ok,
                "norm_bound_ok": norm_bound_ok,
                "guard_passed": guard_passed,
                "accepted": False,
                "probe_loss": None,
                "utility_gain": None,
            }
            trials_log.append(trial_rec)
            trial_metrics_list.append(tr_metrics)
            trial_margins_list.append(tr_margins)

            if guard_passed and obj < best_objective:
                best_objective = obj
                best_trial_idx = k
                chosen_disp = combined_disp
                after_metrics = tr_metrics
                after_margins = tr_margins

        # Step 4: Selector resolution
        feasible_scales = [
            float(candidate_alphas[k][0]) if len(candidate_alphas[k]) > 0 else 0.0
            for k, tr in enumerate(trials_log)
            if tr.get("guard_passed") and tr.get("displacement_norm", 0.0) > 0
        ]

        best_orig_idx = best_trial_idx
        if best_orig_idx is not None:
            original_choice: Optional[Dict[str, Any]] = {
                "trial_index": best_orig_idx,
                "scale": float(candidate_alphas[best_orig_idx][0]) if len(candidate_alphas[best_orig_idx]) > 0 else 0.0,
                "objective": float(trials_log[best_orig_idx]["objective"]) if np.isfinite(trials_log[best_orig_idx]["objective"]) else None,
                "foreground_macro_dice": trials_log[best_orig_idx]["foreground_macro_dice"],
            }
        else:
            original_choice = None

        status_utility = None
        probe_telemetry: Dict[str, Any] = {
            "reference_kd_loss": None,
            "probe_denominator": None,
            "probe_mask_identity": None,
            "classwise_kd_loss": None,
        }

        if selector == DEFAULT_SELECTOR or method != "scalar_guard":
            if best_trial_idx is not None:
                if trials_log[best_trial_idx]["displacement_norm"] == 0.0:
                    reason = "zero_displacement"
                elif method == "plain_kd":
                    reason = "plain_kd_commit"
                else:
                    reason = "original_strict_objective"
            else:
                reason = "guard_rejected_all" if guarded else "plain_kd_rejected"

        elif selector == "larger_tie":
            best_trial_idx, reason = select_larger_tie(
                trials_log=trials_log,
                orig_trial_idx=best_orig_idx,
                candidate_alphas=candidate_alphas,
                dice_tolerance=selector_dice_tolerance,
            )

        elif selector in ("cross_patient_utility", "same_batch_utility"):
            (
                best_trial_idx,
                status_utility,
                reason,
                probe_counts,
                probe_times,
                probe_telemetry,
            ) = select_utility(
                model=model,
                base_params=base_params,
                combined_displacements=combined_displacements,
                candidate_alphas=candidate_alphas,
                trials_log=trials_log,
                utility_probe=utility_probe,
                gain_tolerance=utility_gain_tolerance,
                tie_tolerance=utility_tie_tolerance,
            )
            probe_reference_evaluations += probe_counts["probe_reference_evaluations"]
            probe_trial_evaluations += probe_counts["probe_trial_evaluations"]
            t_probe_ref += probe_times["probe_reference_seconds"]
            t_probe_trials += probe_times["probe_trials_seconds"]

        # Step 5: Finalize outcome
        if best_trial_idx is not None and (chosen_disp is not None or selector != DEFAULT_SELECTOR):
            chosen_alpha = candidate_alphas[best_trial_idx]
            chosen_disp_norm = trials_log[best_trial_idx]["displacement_norm"]

            if chosen_disp_norm > 0:
                status = "accepted_nonzero"
                accepted = True
                trials_log[best_trial_idx]["accepted"] = True
                after_metrics = trial_metrics_list[best_trial_idx]
                after_margins = trial_margins_list[best_trial_idx]
                chosen_disp = combined_displacements[best_trial_idx]
            else:
                status = "zero_displacement"
                accepted = False
                trials_log[best_trial_idx]["accepted"] = False
                after_metrics = base_metrics
                after_margins = base_margins
                chosen_disp = None
        else:
            if status_utility is not None:
                status = status_utility
            else:
                status = "rejected_all"
            accepted = False
            chosen_alpha = np.zeros(m, dtype=np.float64)
            chosen_disp_norm = 0.0
            after_metrics = base_metrics
            after_margins = base_margins
            chosen_disp = None

        before_m = base_margins.get("mean_true_margin") if base_margins else None
        after_m = after_margins.get("mean_true_margin") if after_margins else None
        margin_change = float(after_m - before_m) if (before_m is not None and after_m is not None) else 0.0
        mean_margin_decreased = bool(after_m < (before_m - 1e-7)) if (before_m is not None and after_m is not None) else False

        t_total = time.monotonic() - t_total_start
        forward_counts = {
            "audit_stream_batches": audit_stream_batches,
            "audit_reference_evaluations": 1 if base_metrics is not None else 0,
            "nonzero_trial_evaluations": nonzero_trial_evaluations,
            "probe_reference_evaluations": probe_reference_evaluations,
            "probe_trial_evaluations": probe_trial_evaluations,
        }
        timings = {
            "audit_reference_seconds": float(t_audit_ref),
            "audit_trials_seconds": float(t_audit_trials),
            "probe_reference_seconds": float(t_probe_ref),
            "probe_trials_seconds": float(t_probe_trials),
            "total_seconds": float(t_total),
        }

        receipt: Dict[str, Any] = {
            "method": method,
            "status": status,
            "accepted": accepted,
            "source_ids": list(source_ids),
            "alpha": chosen_alpha.tolist(),
            "scale": float(np.mean(chosen_alpha)) if m > 0 else 0.0,
            "proposed_displacement_norm": float(trials_log[0]["displacement_norm"]) if trials_log else 0.0,
            "realized_displacement_norm": float(chosen_disp_norm),
            "base_macro_dice": base_macro_dice,
            "base_macro_absent_fp": base_macro_absent_fp,
            "base_fg_dice": base_fg_dice,
            "macro_dice_after": after_metrics["macro_dice"] if after_metrics else base_macro_dice,
            "macro_absent_fp_after": after_metrics["macro_absent_fp"] if after_metrics else base_macro_absent_fp,
            "macro_fg_dice_after": after_metrics["foreground_macro_dice"] if after_metrics else base_fg_dice,
            "margins_before": base_margins,
            "margins_after": after_margins,
            "margin_change": margin_change,
            "mean_margin_decreased": mean_margin_decreased,
            "margin_decreased_fraction": None,
            "trials": trials_log,
            "optimizer_steps_in_transaction": 0,
            "ema_updates_in_transaction": 0,
            "selector": selector,
            "reason": reason,
            "original_choice": original_choice,
            "feasible_scales": feasible_scales,
            "forward_counts": forward_counts,
            "timings": timings,
            "telemetry": {
                "eps": eps,
                "absent_fp_budget": absent_fp_budget,
                "tolerance": tolerance,
                "ridge": ridge,
                "max_trials": max_trials,
                "max_norm": max_norm,
                "num_directions": m,
                "audit_identity_hash": audit_hash,
                "selector": selector,
                "selector_dice_tolerance": selector_dice_tolerance,
                "utility_gain_tolerance": utility_gain_tolerance,
                "utility_tie_tolerance": utility_tie_tolerance,
                "original_choice": original_choice,
                "feasible_scales": feasible_scales,
                "forward_counts": forward_counts,
                "timings": timings,
                "reason": reason,
                "reference_kd_loss": probe_telemetry.get("reference_kd_loss"),
                "probe_denominator": probe_telemetry.get("probe_denominator"),
                "probe_mask_identity": probe_telemetry.get("probe_mask_identity"),
                "probe_identity_sha256": probe_telemetry.get("probe_identity_sha256"),
                "classwise_kd_loss": probe_telemetry.get("classwise_kd_loss"),
            },
        }

        # Step 6: Validate receipt BEFORE any live parameter modification (exception-atomic)
        try:
            json.dumps(receipt, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"receipt must be JSON-safe without NaN: {exc}") from exc

        # Step 7: Direct parameter commit ONLY after successful validation
        if accepted and chosen_disp is not None and chosen_disp_norm > 0:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in chosen_disp:
                        proposed_val = p.data + chosen_disp[name].to(device=p.device, dtype=p.dtype)
                        if not torch.isfinite(proposed_val).all():
                            raise FloatingPointError(f"resulting parameter '{name}' contains non-finite values")
                        p.data.copy_(proposed_val)

        return receipt

    finally:
        # Atomic restoration of execution mode and RNG
        state_ctx.restore()
