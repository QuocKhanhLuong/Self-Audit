"""Auditor losses with local classification and signed transition ranking."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


def signed_ranking_loss(
    predicted_delta_q: Tensor,
    actual_delta_dice: Tensor,
    margin: float = 0.05,
    neutral_margin: float = 0.0,
) -> Tensor:
    predicted = predicted_delta_q.reshape(-1)
    actual = actual_delta_dice.reshape(-1).detach()
    neutral = actual.abs() < float(neutral_margin)
    ranking_actual = actual[~neutral]
    ranking_predicted = predicted[~neutral]
    positive = ranking_predicted[ranking_actual > 0]
    negative = ranking_predicted[ranking_actual < 0]
    if positive.numel() and negative.numel():
        pairwise = F.relu(float(margin) - positive[:, None] + negative[None, :]).mean()
        regression = F.smooth_l1_loss(ranking_predicted, ranking_actual)
        ranking_term = pairwise + 0.25 * regression
    elif ranking_predicted.numel():
        ranking_term = F.smooth_l1_loss(ranking_predicted, ranking_actual)
    else:
        ranking_term = predicted.new_zeros(())
    if neutral.any():
        # Near-zero GT changes are calibration examples, not noisy ordering
        # pairs.  Pull their predicted quality toward the neutral point.
        neutral_term = F.smooth_l1_loss(predicted[neutral], torch.zeros_like(predicted[neutral]))
        if ranking_predicted.numel():
            return ranking_term + neutral_term
        return neutral_term
    return ranking_term


def _get(output: Any, *names: str) -> Tensor:
    if isinstance(output, dict):
        for name in names:
            if torch.is_tensor(output.get(name)):
                return output[name]
    for name in names:
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    raise TypeError(f"Output does not contain any of {names}")


def _targets(targets: Any) -> tuple[Tensor, Tensor]:
    if isinstance(targets, dict):
        local = targets.get("local", targets.get("local_target"))
        global_target = targets.get("delta_dice", targets.get("global_target"))
    else:
        local = getattr(targets, "local", getattr(targets, "local_target", None))
        global_target = getattr(targets, "delta_dice", getattr(targets, "global_target", None))
    if not torch.is_tensor(local) or not torch.is_tensor(global_target):
        raise TypeError("Audit targets must provide local and delta_dice tensors")
    return local, global_target


def audit_loss(
    output: Any,
    targets: Any,
    *,
    local_weight: float = 1.0,
    global_weight: float = 1.0,
    margin: float = 0.05,
    neutral_margin: float = 0.0,
    local_class_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute auditor loss without propagating into annotation inputs."""

    local_logits = _get(output, "local_logits", "local")
    delta_q = _get(output, "delta_q", "global_delta_q", "delta_quality")
    local_target, delta_dice = _targets(targets)
    if local_class_weights is not None:
        local_class_weights = local_class_weights.to(device=local_logits.device, dtype=local_logits.dtype)
        if local_class_weights.numel() != local_logits.shape[1]:
            raise ValueError(
                f"local_class_weights must have {local_logits.shape[1]} values, got {local_class_weights.numel()}"
            )
    local_term = F.cross_entropy(local_logits, local_target.long(), weight=local_class_weights)
    global_term = signed_ranking_loss(
        delta_q,
        delta_dice,
        margin=margin,
        neutral_margin=neutral_margin,
    )
    total = float(local_weight) * local_term + float(global_weight) * global_term
    return total, {"local": local_term.detach(), "global": global_term.detach(), "loss": total.detach()}


def transition_audit_loss(output: Any, targets: Any, **kwargs):
    return audit_loss(output, targets, **kwargs)
