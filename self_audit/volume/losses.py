"""Losses for the native-volume experiments.

The original single-slice loss in :mod:`self_audit.losses` is intentionally
left untouched.  ``legacy_pooled`` below is its exact estimator after a
``B x Z`` flattening when all voxels are valid.  ``volume_mean`` is an
explicitly different estimator: every supported volume contributes one CE
plus present-foreground Dice value, and the volume values are averaged.

Masks are deliberately separate.  ``geometric_valid`` describes image/grid
support while ``label_valid`` describes label availability.  ``255`` is an
unknown label and is never converted to background.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


NUM_CLASSES = 4
UNKNOWN = 255
_EPS = 1e-5


def _loss_logits(logits: Tensor) -> Tensor:
    """Use FP32 accumulators for low-precision logits only.

    Half/bfloat16 Dice reductions can overflow when a supported slab contains
    65,536 or more voxels (the denominator is summed in the input dtype).
    Keeping FP32/FP64 tensors untouched preserves the historical estimator and
    its normal-precision numerical behavior exactly; casting low precision is
    differentiable and returns finite gradients to the original tensor.
    """
    if logits.dtype in {torch.float16, torch.bfloat16}:
        return logits.float()
    return logits


def _as_bool_mask(mask: Tensor | None, shape: tuple[int, ...], name: str, device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if not isinstance(mask, Tensor) or mask.shape != shape or mask.dtype is not torch.bool:
        raise ValueError(f"{name} must be a bool tensor with shape {shape}")
    return mask.to(device=device)


def _validate_inputs(
    logits: Tensor,
    labels: Tensor,
    geometric_valid: Tensor | None,
    label_valid: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(logits, Tensor) or logits.ndim != 5 or logits.shape[1] != NUM_CLASSES:
        raise ValueError("volume logits must have shape Bx4xZxHxW")
    if not isinstance(labels, Tensor) or labels.ndim != 4 or labels.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("volume labels must have shape BxZxHxW matching logits")
    if labels.dtype is not torch.long:
        raise ValueError("volume labels must use torch.long")
    shape = tuple(labels.shape)
    geom = _as_bool_mask(geometric_valid, shape, "geometric_valid", logits.device)
    label = _as_bool_mask(label_valid, shape, "label_valid", logits.device)
    labels = labels.to(device=logits.device)

    # Values outside {0, 1, 2, 3, 255} are ambiguous.  Fail closed rather
    # than silently turning a malformed label into a foreground class.
    allowed = (labels == UNKNOWN) | ((labels >= 0) & (labels < NUM_CLASSES))
    if not bool(allowed.all().detach()):
        raise ValueError("labels must contain classes 0..3 or UNKNOWN=255")
    support = geom & label & (labels != UNKNOWN)
    safe_labels = torch.where(support, labels, torch.full_like(labels, UNKNOWN))
    return safe_labels, support, geom


def _one_hot_probabilities(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Return masked probabilities and one-hot targets in B,C,Z,H,W layout."""
    valid = labels != UNKNOWN
    encoded = F.one_hot(labels.clamp_max(NUM_CLASSES - 1), NUM_CLASSES)
    target = encoded.movedim(-1, 1).to(dtype=logits.dtype)
    target = target * valid[:, None]
    probability = logits.softmax(dim=1) * valid[:, None]
    return probability, target


def _dice_loss(probability: Tensor, target: Tensor, dims: tuple[int, ...], *, present: Tensor | None = None) -> tuple[Tensor, Tensor]:
    den = probability.sum(dims) + target.sum(dims)
    dice = (2.0 * (probability * target).sum(dims) + _EPS) / (den + _EPS)
    if present is None:
        present = target.sum(dims) > 0
    foreground_present = present[1:]
    if bool(foreground_present.any().detach()):
        loss = (1.0 - dice[1:][foreground_present]).mean()
    else:
        loss = probability.sum() * 0.0
    return loss, dice


def _detached_metrics(**values: Tensor | int | float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, Tensor):
            result[key] = value.detach()
        else:
            result[key] = value
    return result


def _legacy_pooled(logits: Tensor, labels: Tensor, support: Tensor) -> tuple[Tensor, dict[str, Any]]:
    b, classes, z, h, w = logits.shape
    flat_logits = logits.permute(0, 2, 1, 3, 4).reshape(b * z, classes, h, w)
    flat_labels = labels.reshape(b * z, h, w)
    valid_count = support.sum()
    if int(valid_count.detach()) == 0:
        raise ValueError("empty labeled support")

    # ``flat_labels`` uses UNKNOWN for both padding and unknown labels.  With
    # all geometry/label masks true this is byte-for-byte the old CE call.
    ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=UNKNOWN)
    probability, target = _one_hot_probabilities(flat_logits, flat_labels)
    dice_loss, dice = _dice_loss(probability, target, (0, 2, 3))
    loss = ce + dice_loss
    present = target[:, 1:].sum((0, 2, 3)) > 0
    metrics = _detached_metrics(
        ce=ce,
        dice_loss=dice_loss,
        loss=loss,
        support_voxels=valid_count,
        volumes_used=torch.tensor(b, device=logits.device),
        present_foreground=present.sum(),
        absent_foreground=torch.tensor(NUM_CLASSES - 1, device=logits.device) - present.sum(),
        dice_foreground=dice[1:].mean(),
        dice_macro4=dice.mean(),
        estimator_code=0,
    )
    return loss, metrics


def _volume_mean(logits: Tensor, labels: Tensor, support: Tensor) -> tuple[Tensor, dict[str, Any]]:
    values: list[Tensor] = []
    ce_values: list[Tensor] = []
    dice_values: list[Tensor] = []
    dice4_values: list[Tensor] = []
    present_values: list[Tensor] = []
    for index in range(logits.shape[0]):
        if not bool(support[index].any().detach()):
            continue
        current_logits = logits[index : index + 1]
        current_labels = labels[index : index + 1]
        ce = F.cross_entropy(current_logits, current_labels, ignore_index=UNKNOWN)
        probability, target = _one_hot_probabilities(current_logits, current_labels)
        # ``current_logits`` retains a singleton B dimension and the volume
        # layout is B,C,Z,H,W.  Reduce every sample/spatial axis here.  The
        # legacy flattened 2-D path correctly uses (0,2,3); reusing those
        # axes after slicing a volume would leave W as a vector axis and
        # silently count per-column presence (and return a wrong objective).
        volume_dims = (0, 2, 3, 4)
        present = target.sum(volume_dims) > 0
        dice_loss, dice = _dice_loss(probability, target, volume_dims, present=present)
        values.append(ce + dice_loss)
        ce_values.append(ce)
        dice_values.append(dice_loss)
        dice4_values.append(dice.mean())
        present_values.append(present[1:].sum())
    if not values:
        raise ValueError("empty labeled support")
    loss = torch.stack(values).mean()
    ce = torch.stack(ce_values).mean()
    dice_loss = torch.stack(dice_values).mean()
    present_total = torch.stack(present_values).sum()
    volumes = len(values)
    metrics = _detached_metrics(
        ce=ce,
        dice_loss=dice_loss,
        loss=loss,
        support_voxels=support.sum(),
        volumes_used=torch.tensor(volumes, device=logits.device),
        present_foreground=present_total,
        # Presence is counted per supported volume, not as a single batch
        # class count.  This remains non-negative even when a foreground
        # class is absent from every volume.
        absent_foreground=torch.tensor(NUM_CLASSES - 1, device=logits.device) * volumes - present_total,
        dice_foreground=1.0 - dice_loss,
        dice_macro4=torch.stack(dice4_values).mean(),
        estimator_code=1,
    )
    return loss, metrics


def supervised_volume_loss(
    logits: Tensor,
    labels: Tensor,
    geometric_valid: Tensor | None = None,
    label_valid: Tensor | None = None,
    *,
    mode: str = "legacy_pooled",
    return_metrics: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """Compute dense CE + foreground Dice for native volume logits.

    ``legacy_pooled`` flattens ``B x Z`` and exactly reproduces
    :func:`self_audit.losses.supervised_loss` when all support is valid.
    ``volume_mean`` averages one CE+present-foreground-Dice value per volume,
    so it is intentionally not equivalent to a pooled or microbatch mean.
    """
    compute_logits = _loss_logits(logits)
    labels, support, _ = _validate_inputs(compute_logits, labels, geometric_valid, label_valid)
    if mode == "legacy_pooled":
        loss, metrics = _legacy_pooled(compute_logits, labels, support)
    elif mode == "volume_mean":
        loss, metrics = _volume_mean(compute_logits, labels, support)
    else:
        raise ValueError("mode must be legacy_pooled or volume_mean")
    return (loss, metrics) if return_metrics else loss


def combine_accumulated_losses(
    losses: Sequence[Tensor],
    volume_counts: Sequence[int],
    *,
    mode: str = "volume_mean",
) -> Tensor:
    """Combine microbatch losses using the declared logical-batch estimator.

    For ``volume_mean``, each microbatch loss is weighted by its number of
    supported volumes, giving the same value and gradients as one complete
    logical batch (assuming the same per-volume support).  Pooled Dice cannot
    be reconstructed from scalar microbatch losses, so ``legacy_pooled``
    fails closed when more than one microbatch is supplied.
    """
    if not losses or len(losses) != len(volume_counts):
        raise ValueError("losses and volume_counts must be non-empty and aligned")
    if any(not isinstance(value, Tensor) or value.ndim != 0 for value in losses):
        raise ValueError("losses must be scalar tensors")
    if any(type(count) is not int or count <= 0 for count in volume_counts):
        raise ValueError("volume_counts must contain positive integers")
    if mode == "legacy_pooled":
        if len(losses) != 1:
            raise ValueError("legacy_pooled accumulation requires graph-retaining concatenation")
        return losses[0]
    if mode != "volume_mean":
        raise ValueError("mode must be legacy_pooled or volume_mean")
    total = float(sum(volume_counts))
    return sum((loss * (count / total) for loss, count in zip(losses, volume_counts, strict=True)))


def pseudo_volume_ce(
    logits: Tensor,
    targets: Tensor,
    geometric_valid: Tensor | None = None,
    pseudo_valid: Tensor | None = None,
    *,
    weight: float = 1.0,
    return_metrics: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """Detached-target pseudo CE with explicit geometric/acceptance support."""
    if not isinstance(logits, Tensor) or logits.ndim != 5 or logits.shape[1] != NUM_CLASSES:
        raise ValueError("volume logits must have shape Bx4xZxHxW")
    if not isinstance(targets, Tensor) or targets.shape != logits.shape[:1] + logits.shape[2:] or targets.dtype is not torch.long:
        raise ValueError("pseudo targets must be long BxZxHxW")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
        raise ValueError("pseudo weight must be non-negative")
    shape = tuple(targets.shape)
    geom = _as_bool_mask(geometric_valid, shape, "geometric_valid", logits.device)
    accepted = _as_bool_mask(pseudo_valid, shape, "pseudo_valid", logits.device)
    targets = targets.to(device=logits.device).detach()
    allowed = (targets == UNKNOWN) | ((targets >= 0) & (targets < NUM_CLASSES))
    if not bool(allowed.all().detach()):
        raise ValueError("pseudo targets must contain classes 0..3 or UNKNOWN=255")
    support = geom & accepted & (targets != UNKNOWN)
    safe_targets = torch.where(support, targets, torch.full_like(targets, UNKNOWN))
    count = support.sum()
    compute_logits = _loss_logits(logits)
    if int(count.detach()) == 0 or float(weight) == 0.0:
        loss = compute_logits.sum() * 0.0
    else:
        flat_logits = compute_logits.permute(0, 2, 1, 3, 4).reshape(-1, NUM_CLASSES, *compute_logits.shape[-2:])
        loss = F.cross_entropy(flat_logits, safe_targets.reshape(-1, *logits.shape[-2:]), ignore_index=UNKNOWN)
        loss = loss * float(weight)
    metrics = _detached_metrics(
        loss=loss,
        unweighted_loss=(loss / float(weight) if float(weight) > 0 else loss),
        accepted_voxels=count,
        weight=float(weight),
        empty=(int(count.detach()) == 0),
    )
    return (loss, metrics) if return_metrics else loss


def pseudo_weight_ramp(step: int, *, base_weight: float, ramp_steps: int = 0, pseudo_delay: int = 0) -> float:
    """Return a linear pseudo-loss weight after the supervised-only delay."""
    if type(step) is not int or step < 0 or type(pseudo_delay) is not int or pseudo_delay < 0:
        raise ValueError("step and pseudo_delay must be non-negative integers")
    if isinstance(base_weight, bool) or not isinstance(base_weight, (int, float)) or base_weight < 0:
        raise ValueError("base_weight must be non-negative")
    if type(ramp_steps) is not int or ramp_steps < 0:
        raise ValueError("ramp_steps must be a non-negative integer")
    if step < pseudo_delay:
        return 0.0
    if ramp_steps == 0:
        return float(base_weight)
    return float(base_weight) * min(1.0, (step - pseudo_delay + 1) / ramp_steps)


# Explicit aliases used by a few callers while keeping the public API small.
supervised_loss_volume = supervised_volume_loss
pseudo_loss_volume = pseudo_volume_ce
