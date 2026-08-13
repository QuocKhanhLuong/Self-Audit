"""GT-only training targets for transition auditing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


FIX = 0
UNCHANGED = 1
REGRESS = 2
LOCAL_AUDIT_NAMES = ("FIX", "UNCHANGED", "REGRESS")


@dataclass
class TransitionTargets:
    local: Tensor
    delta_dice: Tensor

    @property
    def local_target(self) -> Tensor:
        return self.local

    @property
    def global_target(self) -> Tensor:
        return self.delta_dice

    def __getitem__(self, key: str):
        if key in {"local", "local_target"}:
            return self.local
        if key in {"delta_dice", "global_target"}:
            return self.delta_dice
        raise KeyError(key)


def _probabilities(value: Tensor) -> Tensor:
    if value.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(value.shape)}")
    if bool((value.detach().min() < 0).item()) or bool((value.detach().max() > 1).item()):
        return value.softmax(dim=1)
    return value / value.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _labels(value: Tensor) -> Tensor:
    if value.ndim == 4:
        return _probabilities(value).argmax(dim=1)
    if value.ndim == 3:
        return value.long()
    raise ValueError(f"Expected logits/probabilities [B,C,H,W] or labels [B,H,W], got {tuple(value.shape)}")


def local_audit_targets(previous: Tensor, candidate: Tensor, ground_truth: Tensor) -> Tensor:
    """Return exact FIX/UNCHANGED/REGRESS semantics for each pixel."""

    previous_label = _labels(previous)
    candidate_label = _labels(candidate)
    ground_truth = _labels(ground_truth)
    if previous_label.shape != candidate_label.shape or previous_label.shape != ground_truth.shape:
        raise ValueError(
            f"Transition/GT shape mismatch: {previous_label.shape}, {candidate_label.shape}, {ground_truth.shape}"
        )
    previous_correct = previous_label == ground_truth
    candidate_correct = candidate_label == ground_truth
    target = torch.full_like(ground_truth, UNCHANGED)
    target[~previous_correct & candidate_correct] = FIX
    target[previous_correct & ~candidate_correct] = REGRESS
    return target


def multiclass_dice(prediction: Tensor, ground_truth: Tensor, num_classes: int = 4) -> Tensor:
    prediction = _labels(prediction)
    ground_truth = _labels(ground_truth)
    values = []
    for cls in range(1, int(num_classes)):
        pred = prediction == cls
        true = ground_truth == cls
        denominator = pred.flatten(1).sum(1) + true.flatten(1).sum(1)
        score = torch.where(
            denominator == 0,
            torch.ones_like(denominator, dtype=torch.float32),
            2.0 * (pred & true).flatten(1).sum(1).float() / denominator.clamp_min(1).float(),
        )
        values.append(score)
    return torch.stack(values, dim=1).mean(dim=1)


def delta_dice_target(previous: Tensor, candidate: Tensor, ground_truth: Tensor, num_classes: int = 4) -> Tensor:
    previous_labels = _labels(previous)
    candidate_labels = _labels(candidate)
    ground_truth = _labels(ground_truth)
    return multiclass_dice(candidate_labels, ground_truth, num_classes) - multiclass_dice(previous_labels, ground_truth, num_classes)


def build_transition_targets(
    previous: Tensor,
    candidate: Tensor,
    ground_truth: Tensor,
    *,
    num_classes: int = 4,
) -> TransitionTargets:
    return TransitionTargets(
        local=local_audit_targets(previous, candidate, ground_truth),
        delta_dice=delta_dice_target(previous, candidate, ground_truth, num_classes=num_classes).unsqueeze(1),
    )


build_local_audit_targets = local_audit_targets
build_counterfactual_targets = build_transition_targets
