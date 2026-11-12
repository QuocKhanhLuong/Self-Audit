"""Losses for Q-3D teacher, tokenizer, and student training stages.

These functions keep the Q-3D near/far SDF weighting and class-balanced
distillation separate from the legacy A-3D ``teacher_loss`` contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from self_audit.losses import supervised_loss


def _labels4(labels: Tensor) -> Tensor:
    if not isinstance(labels, Tensor):
        raise TypeError("labels must be a tensor")
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 4 or labels.dtype.is_floating_point or labels.dtype.is_complex:
        raise ValueError("labels must be integer [B,D,H,W] or [B,1,D,H,W]")
    return labels.long()


def balanced_near_far_sdf_loss(
    prediction: Tensor,
    target: Tensor,
    labels: Tensor | None = None,
    *,
    radius_mm: float = 5.0,
    near_mm: float = 2.5,
    alpha: float = 0.5,
) -> Tensor:
    """Near/far balanced L1 for normalized signed distance fields.

    ``target`` follows the A-3D physical voxel-face convention and is already
    divided by ``radius_mm`` and clipped to ``[-1, 1]``.  Since it is clipped,
    the near band ``|d| < near_mm`` is exactly ``|target| < near_mm/radius_mm``.
    If either band is empty its weight is removed and the remaining band is
    renormalized, as required by the Q-3D teacher/student protocol.
    """

    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be tensors")
    if prediction.ndim != 5 or prediction.shape[1] != 4 or prediction.shape != target.shape:
        raise ValueError("SDF prediction/target must share [B,4,D,H,W]")
    if not (0.0 < near_mm < radius_mm) or not (0.0 <= alpha <= 1.0):
        raise ValueError("SDF radius, near band, and alpha are invalid")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("SDF prediction/target contains NaN or Inf")
    if bool((target < -1.00001).any()) or bool((target > 1.00001).any()):
        raise ValueError("normalized SDF target must be in [-1,1]")

    valid = torch.ones_like(target, dtype=torch.bool)
    if labels is not None:
        y = _labels4(labels)
        if tuple(y.shape) != (target.shape[0], *target.shape[-3:]):
            raise ValueError("SDF labels must match target batch/spatial dimensions")
        if bool(((y != 255) & ((y < 0) | (y > 3))).any()):
            raise ValueError("SDF labels contain unknown class IDs")
        valid = (y != 255).unsqueeze(1).expand_as(target)
    if not bool(valid.any()):
        raise ValueError("SDF loss has no valid support")

    error = (prediction.float() - target.float()).abs()
    near = target.abs() < (float(near_mm) / float(radius_mm))
    far = ~near
    near_support = near & valid
    far_support = far & valid
    terms: list[tuple[float, Tensor]] = []
    if bool(near_support.any()):
        terms.append((float(alpha), error[near_support].mean()))
    if bool(far_support.any()):
        terms.append((1.0 - float(alpha), error[far_support].mean()))
    if not terms:
        raise ValueError("SDF near/far loss has no support")
    weight_sum = sum(weight for weight, _ in terms)
    if weight_sum <= 0.0:
        # alpha=0 or 1 can otherwise silently discard the only populated band.
        return torch.stack([value for _, value in terms]).mean()
    loss = sum(weight * value for weight, value in terms) / weight_sum
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite balanced near/far SDF loss")
    return loss


def q3d_teacher_loss(
    outputs: Mapping[str, Tensor],
    labels: Tensor,
    sdf_targets: Tensor,
    *,
    lambda_distance: float = 1.0,
    radius_mm: float = 5.0,
    near_mm: float = 2.5,
    alpha: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Q-3D stage-T loss: two CE+foreground-Dice heads plus near/far SDF."""

    if set(outputs) != {"region_logits", "boundary_logits", "sdf"}:
        raise ValueError("teacher outputs must contain region_logits, boundary_logits and sdf")
    region = outputs["region_logits"]
    boundary = outputs["boundary_logits"]
    sdf = outputs["sdf"]
    if region.ndim != 5 or region.shape[1] != 4 or boundary.shape != region.shape or sdf.shape != region.shape:
        raise ValueError("teacher outputs must share [B,4,D,H,W]")
    if not torch.isfinite(torch.tensor(float(lambda_distance))) or float(lambda_distance) < 0:
        raise ValueError("lambda_distance must be finite and nonnegative")
    labels4 = _labels4(labels)
    region_loss = supervised_loss(region, labels4)
    boundary_loss = supervised_loss(boundary, labels4)
    distance_loss = balanced_near_far_sdf_loss(
        sdf, sdf_targets, labels4, radius_mm=radius_mm, near_mm=near_mm, alpha=alpha,
    )
    total = region_loss + boundary_loss + float(lambda_distance) * distance_loss
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite Q-3D teacher loss")
    return total, {"region": region_loss.detach(), "boundary": boundary_loss.detach(),
                   "sdf_near_far": distance_loss.detach()}


def student_supervised_loss(
    logits: Tensor,
    sdf: Tensor | None,
    labels: Tensor,
    sdf_targets: Tensor | None,
    *,
    lambda_distance: float = 1.0,
    radius_mm: float = 5.0,
    near_mm: float = 2.5,
    alpha: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Labeled student CE+Dice and optional Q-3D near/far SDF objective."""

    labels4 = _labels4(labels)
    segmentation = supervised_loss(logits, labels4)
    if sdf is None:
        if sdf_targets is not None:
            raise ValueError("sdf_targets cannot be supplied when student SDF output is disabled")
        distance = logits.sum() * 0.0
    else:
        if sdf_targets is None:
            raise ValueError("student SDF output requires physical SDF targets")
        distance = balanced_near_far_sdf_loss(
            sdf, sdf_targets, labels4, radius_mm=radius_mm, near_mm=near_mm, alpha=alpha,
        )
    total = segmentation + float(lambda_distance) * distance
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite Q-3D student supervised loss")
    return total, {"segmentation": segmentation.detach(), "sdf_near_far": distance.detach()}


def commitment_loss(u: Tensor, quantized: Tensor) -> Tensor:
    """Squared commitment with gradients only through the adapter output."""

    if not isinstance(u, Tensor) or not isinstance(quantized, Tensor) or u.shape != quantized.shape:
        raise ValueError("commitment inputs must be matching tensors")
    if not torch.isfinite(u).all() or not torch.isfinite(quantized).all():
        raise FloatingPointError("commitment inputs contain NaN or Inf")
    return (u.float() - quantized.detach().float()).square().mean()


def _class_balanced_weighted_mean(
    values: Tensor,
    classes: Tensor,
    weights: Tensor,
    *,
    class_count: int = 4,
) -> tuple[Tensor, dict[str, Any]]:
    if values.shape != classes.shape or weights.shape != classes.shape:
        raise ValueError("values, class IDs, and weights must have the same [B,...] shape")
    if classes.dtype.is_floating_point or classes.dtype.is_complex:
        raise TypeError("teacher class IDs must be integer")
    if not torch.isfinite(values).all() or not torch.isfinite(weights).all():
        raise FloatingPointError("distillation values/weights contain NaN or Inf")
    if bool((weights < 0).any()):
        raise ValueError("distillation weights must be nonnegative")
    if bool(((classes < 0) | (classes >= class_count)).any()):
        raise ValueError("teacher class IDs are outside the declared class range")
    classes = classes.detach().long()
    weights = weights.detach().float()
    class_terms: list[Tensor] = []
    used: list[int] = []
    dropped: list[int] = []
    for class_id in range(class_count):
        selected = classes.eq(class_id)
        if not bool(selected.any()):
            continue
        class_weights = weights[selected]
        denom = class_weights.sum()
        if not bool(denom > 0):
            dropped.append(class_id)
            continue
        class_terms.append((values[selected].float() * class_weights).sum() / (denom + 1e-8))
        used.append(class_id)
    if not class_terms:
        raise ValueError("all class-balanced distillation weights are zero; refusing to disable the loss")
    result = torch.stack(class_terms).mean()
    if not torch.isfinite(result):
        raise FloatingPointError("nonfinite class-balanced distillation loss")
    return result, {"used_classes": used, "zero_weight_classes": dropped}


def codekd_loss(
    pi_teacher: Tensor,
    pi_student: Tensor,
    teacher_classes: Tensor,
    weights: Tensor,
) -> tuple[Tensor, dict[str, Any]]:
    """Class-balanced weighted KL(teacher assignments || student assignments).

    Background is class zero and receives the same class mean weight as each
    present foreground class.  A zero-weight present class is reported and
    omitted; if every class has zero weight this raises instead of returning a
    fake zero loss.
    """

    if not isinstance(pi_teacher, Tensor) or not isinstance(pi_student, Tensor):
        raise TypeError("assignment distributions must be tensors")
    if pi_teacher.ndim != 5 or pi_teacher.shape[0] != pi_student.shape[0] \
            or pi_teacher.shape[1] != pi_student.shape[1] or pi_teacher.shape[2:] != pi_student.shape[2:]:
        raise ValueError("assignment distributions must share [B,K,D,H,W]")
    if pi_teacher.shape[1] <= 1:
        raise ValueError("codeKD requires at least two shared codebook entries")
    expected = (pi_teacher.shape[0], *pi_teacher.shape[2:])
    if teacher_classes.shape != expected:
        raise ValueError("teacher_classes must match assignment [B,D,H,W]")
    if weights.shape not in {expected, (expected[0], 1, *expected[1:])}:
        raise ValueError("weights must be [B,D,H,W] or [B,1,D,H,W]")
    if weights.ndim == 5:
        weights = weights[:, 0]
    if not torch.isfinite(pi_teacher).all() or not torch.isfinite(pi_student).all():
        raise FloatingPointError("codeKD distributions contain NaN or Inf")
    if bool((pi_teacher < 0).any()) or bool((pi_student < 0).any()):
        raise ValueError("codeKD distributions must be nonnegative")
    if bool((pi_teacher.sum(1) <= 0).any()) or bool((pi_student.sum(1) <= 0).any()):
        raise ValueError("codeKD distributions must have positive mass at every cell")
    # Normalize defensively only to correct floating point sum drift.  Inputs
    # are still required to represent probability mass, not arbitrary scores.
    teacher_probabilities = pi_teacher.detach().float()
    p_t = teacher_probabilities / teacher_probabilities.sum(1, keepdim=True)
    p_s = pi_student.float() / pi_student.float().sum(1, keepdim=True)
    kl = (p_t * (p_t.clamp_min(1e-8).log() - p_s.clamp_min(1e-8).log())).sum(1)
    loss, stats = _class_balanced_weighted_mean(kl, teacher_classes, weights)
    return loss, {"loss": loss.detach(), **stats}


def feature_kd_loss(
    feature_teacher: Tensor,
    feature_student: Tensor,
    teacher_classes: Tensor,
    weights: Tensor,
) -> tuple[Tensor, dict[str, Any]]:
    """C1 class-balanced weighted cosine alignment of continuous features."""

    if feature_teacher.ndim != 5 or feature_student.ndim != 5 \
            or feature_teacher.shape[0] != feature_student.shape[0] \
            or feature_teacher.shape[2:] != feature_student.shape[2:]:
        raise ValueError("feature KD tensors must share [B,C,D,H,W] spatial support")
    if feature_teacher.shape[1] != feature_student.shape[1]:
        raise ValueError("feature KD channels must match")
    if not torch.isfinite(feature_teacher).all() or not torch.isfinite(feature_student).all():
        raise FloatingPointError("feature KD tensors contain NaN or Inf")
    per_cell = 1.0 - F.cosine_similarity(
        feature_teacher.detach().float(), feature_student.float(), dim=1, eps=1e-8,
    )
    loss, stats = _class_balanced_weighted_mean(per_cell, teacher_classes, weights)
    return loss, {"loss": loss.detach(), **stats}


def mean_teacher_consistency_loss(student_logits: Tensor, ema_logits: Tensor) -> Tensor:
    """Detached soft-probability consistency for the Mean Teacher baseline.

    The baseline has no teacher MedNeXt, tokenizer, codebook, or GT on U.  It
    compares the online student's strong-view probabilities to the student EMA
    weak-view probabilities with a symmetric-shape mean-square discrepancy.
    """

    if not isinstance(student_logits, Tensor) or not isinstance(ema_logits, Tensor):
        raise TypeError("Mean Teacher logits must be tensors")
    if student_logits.ndim != 5 or student_logits.shape != ema_logits.shape \
            or student_logits.shape[1] != 4:
        raise ValueError("Mean Teacher logits must share [B,4,D,H,W]")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(ema_logits).all():
        raise FloatingPointError("Mean Teacher logits contain NaN or Inf")
    target = torch.softmax(ema_logits.detach().float(), dim=1)
    prediction = torch.softmax(student_logits.float(), dim=1)
    loss = (prediction - target).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite Mean Teacher consistency loss")
    return loss


def tokenizer_stage_loss(
    *,
    commitment: Tensor,
    reconstruction: Tensor,
    anatomy: Tensor,
    beta: float = 0.25,
    anatomy_weight: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine Q-stage tokenizer objectives using the frozen teacher targets."""

    terms = {"commitment": commitment, "reconstruction": reconstruction, "anatomy": anatomy}
    if any(not isinstance(value, Tensor) or value.ndim != 0 for value in terms.values()):
        raise ValueError("tokenizer stage losses must be scalar tensors")
    if beta < 0 or anatomy_weight < 0 or not torch.isfinite(torch.tensor([beta, anatomy_weight])).all():
        raise ValueError("tokenizer loss weights must be finite and nonnegative")
    total = float(beta) * commitment + reconstruction + float(anatomy_weight) * anatomy
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite tokenizer stage loss")
    return total, {name: value.detach() for name, value in terms.items()}


_Q3D_CLASS_COUNT = 4
_Q3D_WEIGHTED_ARMS = {"C1", "V1", "V1-U"}


def weighted_class_balanced_reduce(
    per_cell_loss: Tensor,
    cls: Tensor,
    w_cell: Tensor,
    omega: float,
) -> tuple[Tensor, dict[str, Any]]:
    """Reduce per-cell losses with exact class balancing and a frame weight.

    ``per_cell_loss`` is either ``[D,H,W]`` or ``[B,D,H,W]``.  ``cls`` and
    ``w_cell`` are shared over the frames and must both be ``[D,H,W]``.  For
    each frame, every class whose detached cell-weight total is positive gets
    one equally weighted class mean.  Classes with zero total are excluded;
    if no class has support, the return value is a graph-connected zero.

    ``counts`` and ``class_weight_totals`` are maps from class ID to the
    detached sum of cell weights.  ``cell_counts`` keeps unweighted counts.
    ``omega`` is applied after class balancing, so a patient weight cannot
    cancel in a class denominator.
    """

    if not isinstance(per_cell_loss, Tensor) or not isinstance(cls, Tensor) \
            or not isinstance(w_cell, Tensor):
        raise TypeError("per_cell_loss, cls, and w_cell must be tensors")
    if per_cell_loss.ndim not in (3, 4):
        raise ValueError("per_cell_loss must have shape [D,H,W] or [B,D,H,W]")
    if any(int(size) <= 0 for size in per_cell_loss.shape):
        raise ValueError("per_cell_loss dimensions must be positive")
    spatial_shape = tuple(per_cell_loss.shape[-3:])
    if cls.ndim != 3 or tuple(cls.shape) != spatial_shape:
        raise ValueError("cls must have the shared [D,H,W] shape")
    if w_cell.ndim != 3 or tuple(w_cell.shape) != spatial_shape:
        raise ValueError("w_cell must have the shared [D,H,W] shape")
    if cls.device != per_cell_loss.device or w_cell.device != per_cell_loss.device:
        raise ValueError("per_cell_loss, cls, and w_cell must share a device")
    if not per_cell_loss.dtype.is_floating_point or per_cell_loss.dtype.is_complex:
        raise TypeError("per_cell_loss must be a floating-point tensor")
    if cls.dtype == torch.bool or cls.dtype.is_floating_point or cls.dtype.is_complex:
        raise TypeError("cls must be an integer tensor")
    if w_cell.dtype == torch.bool or w_cell.dtype.is_complex:
        raise TypeError("w_cell must be a real numeric tensor")
    if not bool(torch.isfinite(per_cell_loss).all()):
        raise FloatingPointError("per_cell_loss contains NaN or Inf")
    if not bool(torch.isfinite(w_cell).all()):
        raise FloatingPointError("w_cell contains NaN or Inf")
    if bool((w_cell < 0).any()):
        raise ValueError("w_cell must be nonnegative")
    if bool(((cls < 0) | (cls >= _Q3D_CLASS_COUNT)).any()):
        raise ValueError("cls contains an ID outside BG/RV/MYO/LV (0..3)")
    if isinstance(omega, Tensor):
        if omega.ndim != 0:
            raise ValueError("omega must be a scalar")
        omega = omega.detach().item()
    if isinstance(omega, bool) or not isinstance(omega, Real):
        raise TypeError("omega must be a real scalar")
    omega_value = float(omega)
    if not math.isfinite(omega_value) or omega_value < 0.0:
        raise ValueError("omega must be finite and nonnegative")

    # Float64 weights preserve a positive support such as 1e-200 even when
    # the loss itself is float32.  This is still an exact epsilon-free divide.
    compute_dtype = torch.float64 if per_cell_loss.dtype == torch.float64 else torch.float32
    values = per_cell_loss.to(dtype=compute_dtype)
    detached_weights = w_cell.detach().to(dtype=torch.float64)
    detached_classes = cls.detach().long()
    weight_totals: dict[int, float] = {}
    cell_counts: dict[int, int] = {}
    active: list[int] = []
    excluded: list[int] = []
    class_frame_means: list[Tensor] = []
    flattened_values = values.flatten(start_dim=1) if values.ndim == 4 else None
    flattened_classes = detached_classes.reshape(-1)

    for class_id in range(_Q3D_CLASS_COUNT):
        selected = detached_classes.eq(class_id)
        count = int(selected.sum().item())
        cell_counts[class_id] = count
        total_weight_tensor = detached_weights[selected].sum()
        total_weight = float(total_weight_tensor.item())
        if not math.isfinite(total_weight):
            raise FloatingPointError("nonfinite class weight total")
        weight_totals[class_id] = total_weight
        if total_weight <= 0.0:
            excluded.append(class_id)
            continue
        active.append(class_id)
        normalized_weights = detached_weights[selected] / total_weight_tensor
        if values.ndim == 3:
            class_frame_means.append((values[selected] * normalized_weights).sum())
        else:
            mask = flattened_classes.eq(class_id)
            class_frame_means.append(
                (flattened_values[:, mask] * normalized_weights).sum(dim=1)
            )

    if not class_frame_means:
        # Multiplication before reduction avoids overflow from a finite but
        # very large loss tensor while retaining the autograd graph.
        base_loss_tensor = (values * 0.0).sum()
    elif values.ndim == 3:
        base_loss_tensor = torch.stack(class_frame_means).mean()
    else:
        # Stack [class, batch], average classes per frame, then average frames.
        base_loss_tensor = torch.stack(class_frame_means, dim=0).mean(dim=0).mean()
    if not bool(torch.isfinite(base_loss_tensor)):
        raise FloatingPointError("nonfinite weighted class-balanced base loss")
    loss = base_loss_tensor * omega_value
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite weighted class-balanced loss")
    weight_totals = dict(weight_totals)
    stats: dict[str, Any] = {
        "active_classids": active,
        "excluded_classids": excluded,
        "counts": weight_totals,
        "class_weight_totals": dict(weight_totals),
        "cell_counts": cell_counts,
        "base_loss": float(base_loss_tensor.detach().item()),
        "omega": omega_value,
    }
    return loss, stats


def cohort_omega(
    w_pat: Mapping[Any, float],
    arm: str,
) -> dict[str, Any]:
    """Build patient weights and diagnostics for one locked Q-3D cohort.

    C1 and V1 normalize physiology scores by their cohort mean after requiring
    positive mean and effective-sample-size fraction ``ESS/N >= 0.25``.  V1-U
    emits unit patient weights but applies the same gate, making it an
    eligibility-matched control.  L always emits unit weights and skips the U
    gate.  Source diagnostics are retained separately from emitted omega.
    """

    if not isinstance(w_pat, Mapping):
        raise TypeError("w_pat must be a nonempty patient-to-weight mapping")
    if not w_pat:
        raise ValueError("w_pat cohort must be nonempty")
    if not isinstance(arm, str):
        raise TypeError("arm must be a string")
    arm_key = arm.strip().upper().replace("_", "-")
    if arm_key == "V1U":
        arm_key = "V1-U"
    if arm_key not in {"L", "C1", "V1", "V1-U"}:
        raise ValueError("arm must be one of L, C1, V1, or V1-U")

    patients = list(w_pat)
    source_values: list[float] = []
    for patient in patients:
        value = w_pat[patient]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"w_pat[{patient!r}] must be a real scalar")
        value_float = float(value)
        if not math.isfinite(value_float) or value_float < 0.0:
            raise ValueError("w_pat values must be finite and nonnegative")
        source_values.append(value_float)

    n = len(source_values)
    scale = max(source_values)
    if scale > 0.0:
        scaled = [value / scale for value in source_values]
        scaled_total = math.fsum(scaled)
        scaled_squared_total = math.fsum(value * value for value in scaled)
        mean_scaled = scaled_total / n
        mean_w_pat = scale * mean_scaled
        ess = scaled_total * scaled_total / scaled_squared_total
    else:
        scaled = [0.0] * n
        mean_scaled = 0.0
        mean_w_pat = 0.0
        ess = 0.0
    ess_fraction = ess / n

    if arm_key in _Q3D_WEIGHTED_ARMS and (mean_w_pat <= 0.0 or ess_fraction < 0.25):
        raise ValueError("physiology cohort fails mean_w_pat>0 and ESS/N>=0.25 gate")

    if arm_key in {"L", "V1-U"}:
        emitted = [1.0] * n
    elif source_values and all(value == source_values[0] and value > 0.0 for value in source_values):
        # Preserve the literal IEEE-754 one-bit value for equal weights.
        emitted = [1.0] * n
    else:
        # The ratio is evaluated on scaled values to avoid underflow for
        # Gaussian-tail weights or overflow for large finite diagnostics.
        emitted = [value / mean_scaled for value in scaled]
    omega = dict(zip(patients, emitted))
    mean_omega = math.fsum(emitted) / n
    max_omega = max(emitted)
    zero_fraction = sum(value == 0.0 for value in emitted) / n
    return {
        "omega": omega,
        "mean_w_pat": float(mean_w_pat),
        "ess": float(ess),
        "ess_fraction": float(ess_fraction),
        "mean_omega": float(mean_omega),
        "max_omega": float(max_omega),
        "zero_fraction": float(zero_fraction),
    }


__all__ = [
    "balanced_near_far_sdf_loss",
    "codekd_loss",
    "cohort_omega",
    "commitment_loss",
    "feature_kd_loss",
    "mean_teacher_consistency_loss",
    "q3d_teacher_loss",
    "student_supervised_loss",
    "tokenizer_stage_loss",
    "weighted_class_balanced_reduce",
]
