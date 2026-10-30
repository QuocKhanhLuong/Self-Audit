"""Utility probe validation, distillation probe evaluation, and selector algorithms for compact audit control.

Implements the contract for transfer-utility selection per CPTU_IMPLEMENTATION_CONTRACT.md:
  - validate_utility_probe:
      Validates probe mapping, types, shapes, dimensions, and finite sum-to-1 probabilities.
      Detaches and clones frozen copies so caller mutations cannot affect controller execution.
  - compute_probe_kd_loss:
      Evaluates whole-model KD loss on frozen probe inputs using functional forward calls.
      Raw teacher confidence thresholding before temperature.
      Exact KD common denominator and temperature scaling matching build_class_candidates.
  - select_larger_tie:
      First resolves original strict objective, then selects the largest feasible scalar
      with foreground Dice equal to that original choice within selector_dice_tolerance.
      Exact ties without hidden numerical slack.
  - select_utility:
      Evaluates probe distillation loss on post-supervised reference and feasible trials.
      Zero competitor with gain zero; requires gain strictly above utility_gain_tolerance.
      Tied gains within utility_tie_tolerance broken by smaller displacement norm.
      Nonfinite probe loss triggers fail-closed zero decision with JSON-safe null receipts.
      Missing probe, zero denominator, or zero coefficient yields actual forward count 0.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F

DEFAULT_SELECTOR = "original"
DEFAULT_SELECTOR_DICE_TOLERANCE = 0.0
DEFAULT_UTILITY_GAIN_TOLERANCE = 1e-7
DEFAULT_UTILITY_TIE_TOLERANCE = 1e-7

VALID_SELECTORS = ("original", "larger_tie", "cross_patient_utility", "same_batch_utility")


def _supports_valid_mask(model: nn.Module) -> bool:
    """Check whether model.forward explicitly declares a valid_mask parameter."""
    import inspect

    try:
        sig = inspect.signature(model.forward)
        return "valid_mask" in sig.parameters
    except (TypeError, ValueError):
        return False


def _call_functional_probe(
    model: nn.Module,
    params: Dict[str, Tensor],
    buffers: Dict[str, Tensor],
    x: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Call model functional forward pass for utility probe evaluation."""
    if valid_mask is not None and _supports_valid_mask(model):
        return functional_call(model, (params, buffers), (x,), {"valid_mask": valid_mask})
    return functional_call(model, (params, buffers), (x,))


def validate_utility_probe(probe: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate and freeze utility probe mapping.

    Args:
        probe: Mapping with 'x', 'targets', 'valid', and optional 'temperature',
               'coefficient', 'confidence', 'metadata'.

    Returns:
        Frozen copy dictionary with detached cloned tensors, or None if probe is None.

    Raises:
        TypeError, ValueError, FloatingPointError: If probe structures, types, shapes, or values are invalid.
    """
    if probe is None:
        return None

    if not isinstance(probe, Mapping):
        raise TypeError(f"utility_probe must be a mapping, got {type(probe)}")

    for req_k in ("x", "targets", "valid"):
        if req_k not in probe:
            raise ValueError(f"utility_probe missing required key '{req_k}'")

    x = probe["x"]
    targets = probe["targets"]
    valid = probe["valid"]

    if not isinstance(x, Tensor) or x.ndim != 4:
        raise ValueError(f"utility_probe 'x' must be a 4D Tensor (BCHW), got {type(x)} ndim={getattr(x, 'ndim', None)}")
    if not torch.isfinite(x).all():
        raise FloatingPointError("utility_probe 'x' contains non-finite values")

    if not isinstance(targets, Tensor) or targets.ndim != 4 or targets.shape[1] != 4:
        raise ValueError(f"utility_probe 'targets' must be a 4D Tensor (B4HW), got shape {getattr(targets, 'shape', None)}")
    if not torch.isfinite(targets).all():
        raise FloatingPointError("utility_probe 'targets' contains non-finite values")
    if (targets < 0).any():
        raise ValueError("utility_probe 'targets' contains negative probability values")

    if not isinstance(valid, Tensor) or valid.dtype != torch.bool or valid.ndim != 3:
        raise ValueError(f"utility_probe 'valid' must be a 3D boolean Tensor (BHW), got {type(valid)} {getattr(valid, 'dtype', None)}")

    B, _, H, W = targets.shape
    if x.shape[0] != B or x.shape[2] != H or x.shape[3] != W:
        raise ValueError(f"spatial/batch shape mismatch between probe x {x.shape} and targets {targets.shape}")
    if valid.shape != (B, H, W):
        raise ValueError(f"shape mismatch between probe valid {valid.shape} and expected {(B, H, W)}")

    # Probabilities sum to 1 check across class dimension for valid voxels
    if valid.any():
        prob_sums = targets.sum(dim=1)
        if not torch.allclose(prob_sums[valid], torch.ones_like(prob_sums[valid]), atol=1e-3):
            raise ValueError("utility_probe 'targets' must sum to 1 across class dimension for valid pixels")

    temperature = probe.get("temperature", 1.0)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"utility_probe 'temperature' must be a finite positive float, got {temperature}")

    coefficient = probe.get("coefficient", 1.0)
    if not isinstance(coefficient, (int, float)) or isinstance(coefficient, bool) or not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError(f"utility_probe 'coefficient' must be a finite non-negative float, got {coefficient}")

    confidence = probe.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not np.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
        raise ValueError(f"utility_probe 'confidence' must be a finite float in [0.0, 1.0], got {confidence}")

    metadata = probe.get("metadata", None)
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("utility_probe 'metadata' must be a mapping if provided")

    # Freeze detached copies
    return {
        "x": x.detach().clone(),
        "targets": targets.detach().clone(),
        "valid": valid.detach().clone(),
        "temperature": float(temperature),
        "coefficient": float(coefficient),
        "confidence": float(confidence),
        "metadata": copy.deepcopy(dict(metadata)) if metadata is not None else None,
    }


def compute_probe_kd_loss(
    model: nn.Module,
    params: Dict[str, Tensor],
    buffers: Dict[str, Tensor],
    probe: Dict[str, Any],
    *,
    return_diagnostics: bool = False,
) -> Tuple[Any, ...]:
    """Compute KD distillation loss on frozen probe inputs.

    Normalizes by common valid teacher-confident pixels using raw teacher confidence
    thresholding before temperature, matching build_class_candidates.

    Returns:
        If return_diagnostics is False:
            (loss_float, denominator, is_finite)
        If return_diagnostics is True:
            (loss_float, denominator, is_finite, diagnostics_dict)
    """
    x = probe["x"]
    targets = probe["targets"]
    valid = probe["valid"]
    temp = probe["temperature"]
    coef = probe["coefficient"]
    conf = probe["confidence"]

    # Raw teacher confidence before temperature
    raw_teacher_max, raw_teacher_argmax = targets.max(dim=1)
    confident_mask = (raw_teacher_max >= conf) & valid
    denominator = int(confident_mask.sum().item())
    mask_identity = hashlib.sha256(confident_mask.cpu().numpy().tobytes()).hexdigest()

    diagnostics: Dict[str, Any] = {
        "denominator": denominator,
        "mask_identity": mask_identity,
        "classwise_kd_loss": {c_name: 0.0 for c_name in ("BG", "RV", "MYO", "LV")},
    }

    if denominator == 0 or coef == 0.0:
        if return_diagnostics:
            return 0.0, denominator, True, diagnostics
        return 0.0, denominator, True

    # Functional forward call
    with torch.no_grad():
        logits = _call_functional_probe(model, params, buffers, x, valid_mask=valid)
        if not torch.isfinite(logits).all():
            diagnostics["classwise_kd_loss"] = {c_name: None for c_name in ("BG", "RV", "MYO", "LV")}
            if return_diagnostics:
                return float("nan"), denominator, False, diagnostics
            return float("nan"), denominator, False

        scaled_logits = logits / temp
        log_probs = F.log_softmax(scaled_logits, dim=1)

        if abs(temp - 1.0) > 1e-7:
            soft_targets = F.softmax(torch.log(targets.clamp(min=1e-12)) / temp, dim=1)
        else:
            soft_targets = targets

        per_pixel_loss = - (soft_targets * log_probs).sum(dim=1) * ((temp ** 2) * coef)
        loss = (per_pixel_loss * confident_mask).sum() / denominator
        loss_val = float(loss.item())
        is_finite = bool(np.isfinite(loss_val))

        classwise = {}
        for c_idx, c_name in enumerate(("BG", "RV", "MYO", "LV")):
            mask_c = confident_mask & (raw_teacher_argmax == c_idx)
            if mask_c.any() and is_finite:
                c_loss = float(((per_pixel_loss * mask_c).sum() / denominator).item())
                classwise[c_name] = c_loss
            else:
                classwise[c_name] = 0.0 if is_finite else None
        diagnostics["classwise_kd_loss"] = classwise

        if return_diagnostics:
            return loss_val, denominator, is_finite, diagnostics
        return loss_val, denominator, is_finite


def select_larger_tie(
    trials_log: Sequence[Dict[str, Any]],
    orig_trial_idx: Optional[int],
    candidate_alphas: Sequence[np.ndarray],
    dice_tolerance: float = DEFAULT_SELECTOR_DICE_TOLERANCE,
) -> Tuple[Optional[int], str]:
    """First resolves original strict objective, then selects the largest feasible scalar

    with foreground Dice equal to that original choice within dice_tolerance.
    Exact Dice equality without hidden numerical slack.

    Returns:
        (chosen_trial_index, reason)
    """
    if orig_trial_idx is None:
        return None, "guard_rejected_all"

    orig_tr = trials_log[orig_trial_idx]
    if orig_tr.get("displacement_norm", 0.0) == 0.0:
        return orig_trial_idx, "zero_displacement"

    orig_fg_dice = orig_tr.get("foreground_macro_dice")

    # Find feasible trials with displacement_norm > 0
    feasible_indices = [
        k for k, tr in enumerate(trials_log)
        if tr.get("guard_passed") and tr.get("displacement_norm", 0.0) > 0
    ]

    if not feasible_indices:
        return orig_trial_idx, "larger_tie_matches_original"

    # Match foreground Dice without hidden numerical slack
    matching: List[int] = []
    for k in feasible_indices:
        k_fg = trials_log[k].get("foreground_macro_dice")
        if orig_fg_dice is None:
            if k_fg is None:
                matching.append(k)
        else:
            if k_fg is not None:
                if dice_tolerance == 0.0:
                    if k_fg == orig_fg_dice:
                        matching.append(k)
                else:
                    if abs(k_fg - orig_fg_dice) <= dice_tolerance:
                        matching.append(k)

    if not matching:
        return orig_trial_idx, "larger_tie_matches_original"

    # Select the trial with the largest scalar scale
    # In scalar_guard, scale is alpha[0]
    scales = [float(candidate_alphas[k][0]) if len(candidate_alphas[k]) > 0 else 0.0 for k in matching]
    max_scale = max(scales)
    tied_for_max = [k for k, s in zip(matching, scales) if s == max_scale]

    # Break ties by smaller trial index (first encountered)
    chosen_idx = min(tied_for_max)

    if chosen_idx == orig_trial_idx:
        reason = "larger_tie_matches_original"
    else:
        reason = "larger_tie_dice_tiebreak"

    return chosen_idx, reason


def select_utility(
    model: nn.Module,
    base_params: Dict[str, Tensor],
    combined_displacements: Sequence[Dict[str, Tensor]],
    candidate_alphas: Sequence[np.ndarray],
    trials_log: List[Dict[str, Any]],
    utility_probe: Optional[Mapping[str, Any]],
    *,
    gain_tolerance: float = DEFAULT_UTILITY_GAIN_TOLERANCE,
    tie_tolerance: float = DEFAULT_UTILITY_TIE_TOLERANCE,
) -> Tuple[Optional[int], str, str, Dict[str, int], Dict[str, float], Dict[str, Any]]:
    """Transfer-utility evaluation on post-supervised reference and feasible trials.

    Returns:
        (best_trial_idx, status, reason, forward_counts, timings, probe_telemetry)
    """
    forward_counts = {
        "probe_reference_evaluations": 0,
        "probe_trial_evaluations": 0,
    }
    timings = {
        "probe_reference_seconds": 0.0,
        "probe_trials_seconds": 0.0,
    }
    probe_telemetry: Dict[str, Any] = {
        "reference_kd_loss": None,
        "probe_denominator": 0,
        "probe_mask_identity": None,
        "classwise_kd_loss": {c_name: None for c_name in ("BG", "RV", "MYO", "LV")},
    }

    # Missing probe is a zero decision, not a fallback to proposal patient
    if utility_probe is None:
        return None, "zero_displacement", "missing_utility_probe", forward_counts, timings, probe_telemetry

    frozen_probe = validate_utility_probe(utility_probe)
    assert frozen_probe is not None

    probe_hash = hashlib.sha256()
    for key in ("x", "targets", "valid"):
        tensor = frozen_probe[key].detach().cpu().contiguous()
        probe_hash.update(str((key, tensor.dtype, tuple(tensor.shape))).encode())
        probe_hash.update(tensor.numpy().tobytes())
    probe_hash.update(json.dumps({k: frozen_probe[k] for k in
        ("temperature", "coefficient", "confidence", "metadata")}, sort_keys=True, allow_nan=False).encode())
    probe_telemetry["probe_identity_sha256"] = probe_hash.hexdigest()

    # Check denominator and coefficient before evaluating model (actual forward count 0 when empty/coefficient 0)
    raw_teacher_max = frozen_probe["targets"].max(dim=1).values
    confident_mask = (raw_teacher_max >= frozen_probe["confidence"]) & frozen_probe["valid"]
    denom = int(confident_mask.sum().item())
    probe_telemetry["probe_denominator"] = denom
    probe_telemetry["probe_mask_identity"] = hashlib.sha256(confident_mask.cpu().numpy().tobytes()).hexdigest()

    if denom == 0:
        return None, "zero_displacement", "zero_probe_denominator", forward_counts, timings, probe_telemetry

    if frozen_probe["coefficient"] == 0.0:
        probe_telemetry["reference_kd_loss"] = 0.0
        probe_telemetry["classwise_kd_loss"] = {c_name: 0.0 for c_name in ("BG", "RV", "MYO", "LV")}
        return None, "zero_displacement", "zero_probe_coefficient", forward_counts, timings, probe_telemetry

    # Evaluate reference probe loss
    fresh_buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    t0_ref = time.monotonic()
    ref_loss, ref_denom, ref_finite, ref_diag = compute_probe_kd_loss(
        model, base_params, fresh_buffers, frozen_probe, return_diagnostics=True
    )
    timings["probe_reference_seconds"] = time.monotonic() - t0_ref
    forward_counts["probe_reference_evaluations"] = 1

    probe_telemetry["reference_kd_loss"] = ref_loss if ref_finite else None
    probe_telemetry["probe_denominator"] = ref_denom
    probe_telemetry["probe_mask_identity"] = ref_diag["mask_identity"]
    probe_telemetry["classwise_kd_loss"] = ref_diag["classwise_kd_loss"]

    # Nonfinite reference loss fails-closed to zero with JSON null receipt
    if not ref_finite:
        return None, "zero_displacement", "nonfinite_probe_loss", forward_counts, timings, probe_telemetry

    # Feasible candidate trials under audit guard
    feasible_indices = [
        k for k, tr in enumerate(trials_log)
        if tr.get("guard_passed") and tr.get("displacement_norm", 0.0) > 0
    ]

    # If no trials passed the guard, guard rejected all
    if not feasible_indices:
        if any(tr.get("guard_passed") and tr.get("displacement_norm") == 0.0 for tr in trials_log):
            return None, "zero_displacement", "zero_displacement", forward_counts, timings, probe_telemetry
        return None, "rejected_all", "guard_rejected_all", forward_counts, timings, probe_telemetry

    # Evaluate feasible trials on probe
    qualifying_trials: List[int] = []
    gains: Dict[int, float] = {}

    for k in feasible_indices:
        t0_tr = time.monotonic()
        d_k = combined_displacements[k]
        trial_p = {
            name: (base_params[name] + d_k[name] if name in d_k else base_params[name])
            for name in base_params
        }
        fresh_b = {name: b.detach().clone() for name, b in model.named_buffers()}
        tr_loss, _, tr_finite, tr_diag = compute_probe_kd_loss(
            model, trial_p, fresh_b, frozen_probe, return_diagnostics=True)
        trials_log[k]["probe_identity_sha256"] = probe_telemetry["probe_identity_sha256"]
        trials_log[k]["probe_mask_identity"] = tr_diag["mask_identity"]
        trials_log[k]["probe_classwise_kd_loss"] = tr_diag["classwise_kd_loss"]
        timings["probe_trials_seconds"] += time.monotonic() - t0_tr
        forward_counts["probe_trial_evaluations"] += 1

        # Nonfinite loss at any evaluated trial fails-closed the entire transaction to zero
        if not tr_finite:
            trials_log[k]["probe_loss"] = None
            trials_log[k]["utility_gain"] = None
            return None, "zero_displacement", "nonfinite_probe_loss", forward_counts, timings, probe_telemetry

        gain = ref_loss - tr_loss
        trials_log[k]["probe_loss"] = tr_loss
        trials_log[k]["utility_gain"] = gain
        gains[k] = gain

        if gain > gain_tolerance:
            qualifying_trials.append(k)

    # Zero competitor includes zero with gain zero. If no trial beats gain tolerance, zero wins
    if not qualifying_trials:
        return None, "zero_displacement", "no_positive_utility_gain", forward_counts, timings, probe_telemetry

    # Select maximum gain
    max_gain = max(gains[k] for k in qualifying_trials)

    # Logical zero participates in the same tie rule, with norm exactly zero.
    if max_gain <= tie_tolerance:
        return None, "zero_displacement", "utility_tied_with_zero", forward_counts, timings, probe_telemetry

    # Break tied gains within tie_tolerance by smaller displacement norm
    if tie_tolerance == 0.0:
        tied_trials = [k for k in qualifying_trials if gains[k] == max_gain]
    else:
        tied_trials = [
            k for k in qualifying_trials
            if abs(gains[k] - max_gain) <= tie_tolerance
        ]

    norms = [float(trials_log[k]["displacement_norm"]) for k in tied_trials]
    min_norm = min(norms)
    tied_for_min_norm = [
        k for k, n in zip(tied_trials, norms)
        if n == min_norm
    ]

    # Deterministic tie break: smallest trial index
    winner_idx = min(tied_for_min_norm)

    return winner_idx, "accepted_nonzero", "max_utility_gain", forward_counts, timings, probe_telemetry
