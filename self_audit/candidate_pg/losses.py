"""Exact detached-weight segmentation losses for Candidate PG."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _labels4(target: Tensor, *, shape: tuple[int, ...]) -> Tensor:
    if not isinstance(target, Tensor):
        raise TypeError("target must be a tensor")
    if target.ndim == len(shape) + 1 and target.shape[1] == 1:
        target = target[:, 0]
    if tuple(target.shape) != shape or target.dtype.is_floating_point or target.dtype.is_complex:
        raise ValueError(f"target must be integer with shape {shape}")
    return target.long()


def _weight4(weight: Tensor, shape: tuple[int, ...]) -> Tensor:
    if not isinstance(weight, Tensor):
        raise TypeError("w must be a tensor")
    if weight.ndim == len(shape) + 1 and weight.shape[1] == 1:
        weight = weight[:, 0]
    if tuple(weight.shape) != shape or weight.dtype.is_complex or not weight.dtype.is_floating_point:
        raise ValueError(f"w must be floating point with shape {shape}")
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        raise ValueError("w must be finite and nonnegative")
    return weight.detach()


def _valid4(valid: Tensor | None, labels: Tensor) -> Tensor:
    if valid is None:
        support = labels != 255
    else:
        if not isinstance(valid, Tensor):
            raise TypeError("valid must be a tensor")
        if valid.ndim == labels.ndim + 1 and valid.shape[1] == 1:
            valid = valid[:, 0]
        if tuple(valid.shape) != tuple(labels.shape):
            raise ValueError("valid must align with target")
        if valid.dtype.is_complex or (valid.dtype.is_floating_point and not bool(torch.isfinite(valid).all())):
            raise ValueError("valid must be finite and real")
        support = valid.bool() & labels.ne(255)
    if bool((support & ((labels < 0) | (labels > 3))).any()):
        raise ValueError("target labels must be in 0..3 or 255(ignore)")
    return support


def weighted_ce(
    logits: Tensor,
    target: Tensor,
    w: Tensor,
    valid: Tensor | None = None,
) -> Tensor:
    """Return ``sum(valid * w * CE) / sum(valid)``.

    The denominator deliberately uses valid-voxel count, not weight mass.  The
    weight map is detached so confidence/reliability computation never receives
    gradients from this objective.
    """

    if not isinstance(logits, Tensor) or logits.ndim < 3 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [B,C,*spatial] with C>=2")
    if not logits.dtype.is_floating_point or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite floating point")
    shape = (int(logits.shape[0]), *[int(v) for v in logits.shape[2:]])
    labels = _labels4(target, shape=shape)
    support = _valid4(valid, labels)
    weights = _weight4(w, shape).to(device=logits.device)
    safe_labels = labels.clamp(0, logits.shape[1] - 1)
    per_voxel = F.cross_entropy(logits, safe_labels, reduction="none")
    support_f = support.to(dtype=per_voxel.dtype)
    denominator = support_f.sum()
    numerator = (per_voxel * weights.to(dtype=per_voxel.dtype) * support_f).sum()
    if float(denominator.detach()) == 0.0:
        return logits.sum() * 0.0
    loss = numerator / denominator
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("weighted CE is nonfinite")
    return loss


def weighted_dice(
    probabilities: Tensor,
    target: Tensor,
    w: Tensor,
    valid: Tensor | None = None,
) -> Tensor:
    """Return one minus weighted soft Dice averaged over present target classes.

    A class is active only when its weighted target mass is positive.  This is
    the locked absent-class convention and avoids inventing a score for a class
    that does not occur in the current packet.
    """

    if not isinstance(probabilities, Tensor) or probabilities.ndim < 3 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [B,C,*spatial] with C>=2")
    if not probabilities.dtype.is_floating_point or not bool(torch.isfinite(probabilities).all()):
        raise ValueError("probabilities must be finite floating point")
    if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
        raise ValueError("probabilities must lie in [0,1]")
    shape = (int(probabilities.shape[0]), *[int(v) for v in probabilities.shape[2:]])
    labels = _labels4(target, shape=shape)
    support = _valid4(valid, labels)
    weights = _weight4(w, shape).to(device=probabilities.device, dtype=probabilities.dtype)
    mask = weights * support.to(dtype=weights.dtype)
    target_valid = labels.clamp(0, probabilities.shape[1] - 1)
    terms: list[Tensor] = []
    # Dice is reported over the three foreground structures only.  Background
    # contributes to CE but is not a foreground quality term.
    for class_id in range(1, int(probabilities.shape[1])):
        target_mask = target_valid.eq(class_id) & support
        target_mass = (mask * target_mask.to(dtype=mask.dtype)).sum()
        if float(target_mass.detach()) <= 0.0:
            continue
        probability_mass = (mask * probabilities[:, class_id]).sum()
        numerator = 2.0 * (mask * probabilities[:, class_id] * target_mask.to(dtype=mask.dtype)).sum()
        denominator = probability_mass + target_mass
        if float(denominator.detach()) > 0.0:
            terms.append(numerator / denominator)
    if not terms:
        return probabilities.sum() * 0.0
    loss = 1.0 - torch.stack(terms).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("weighted Dice is nonfinite")
    return loss


__all__ = ["weighted_ce", "weighted_dice"]
