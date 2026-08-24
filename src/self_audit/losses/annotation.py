"""Initial and recurrent annotation losses."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def soft_dice_loss(logits: Tensor, target: Tensor, *, num_classes: int | None = None, include_background: bool = True, eps: float = 1e-6) -> Tensor:
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError(f"Expected logits [B,C,H,W] and target [B,H,W], got {tuple(logits.shape)}, {tuple(target.shape)}")
    classes = int(num_classes or logits.shape[1])
    target_one_hot = F.one_hot(target.long().clamp(0, classes - 1), classes).permute(0, 3, 1, 2).to(logits.dtype)
    probabilities = logits.softmax(dim=1)
    reduce_classes = range(classes) if include_background else range(1, classes)
    scores = []
    for cls in reduce_classes:
        prediction = probabilities[:, cls]
        truth = target_one_hot[:, cls]
        denominator = prediction.flatten(1).sum(1) + truth.flatten(1).sum(1)
        numerator = 2.0 * (prediction * truth).flatten(1).sum(1)
        scores.append((numerator + eps) / (denominator + eps))
    return 1.0 - torch.stack(scores, dim=1).mean()


def annotation_loss(
    logits: Tensor,
    target: Tensor,
    *,
    dice_weight: float = 1.0,
    cross_entropy_weight: float = 1.0,
    include_background: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    dice = soft_dice_loss(logits, target, include_background=include_background)
    cross_entropy = F.cross_entropy(logits, target.long())
    total = float(dice_weight) * dice + float(cross_entropy_weight) * cross_entropy
    return total, {"dice": dice.detach(), "cross_entropy": cross_entropy.detach(), "loss": total.detach()}


def dice_ce_loss(logits: Tensor, target: Tensor, **kwargs):
    return annotation_loss(logits, target, **kwargs)
