"""Soft counterfactual transitions around real model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
import torch.nn.functional as F

from .targets import _labels, _probabilities


CounterfactualKind = Literal["positive", "negative", "hard_neutral", "on_policy"]


@dataclass
class CounterfactualSample:
    previous_probs: Tensor
    candidate_probs: Tensor
    kind: str
    operation: str

    @property
    def previous(self) -> Tensor:
        return self.previous_probs

    @property
    def candidate(self) -> Tensor:
        return self.candidate_probs


def _one_hot(labels: Tensor, num_classes: int) -> Tensor:
    return F.one_hot(labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


def _local_patch(region: Tensor, radius: int = 2) -> Tensor:
    kernel = max(1, int(radius) * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    return F.max_pool2d(region.float().unsqueeze(1), kernel, stride=1, padding=kernel // 2).squeeze(1).clamp(0, 1)


class CounterfactualGenerator:
    """Generate controlled soft transitions with the requested sampling mix."""

    def __init__(
        self,
        *,
        positive_fraction: float = 0.4,
        negative_fraction: float = 0.4,
        hard_neutral_fraction: float = 0.2,
        min_repair_fraction: float = 0.3,
        max_repair_fraction: float = 0.6,
        num_classes: int = 4,
    ) -> None:
        fractions = (float(positive_fraction), float(negative_fraction), float(hard_neutral_fraction))
        if any(value < 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-5:
            raise ValueError("counterfactual sampling fractions must be non-negative and sum to one")
        if not 0.0 < min_repair_fraction <= max_repair_fraction <= 1.0:
            raise ValueError("repair fraction must satisfy 0 < min <= max <= 1")
        self.fractions = fractions
        self.min_repair_fraction = float(min_repair_fraction)
        self.max_repair_fraction = float(max_repair_fraction)
        self.num_classes = int(num_classes)
        self.negative_operations = (
            "local_erosion",
            "local_dilation",
            "boundary_displacement",
            "hole_insertion",
            "false_island",
            "component_deletion",
            "semantic_class_swap",
        )

    def _pick_kind(self) -> str:
        value = float(torch.rand(()).item())
        if value < self.fractions[0]:
            return "positive"
        if value < self.fractions[0] + self.fractions[1]:
            return "negative"
        return "hard_neutral"

    def _pick_region(self, mask: Tensor) -> Tensor:
        regions = []
        for item in mask:
            positions = torch.nonzero(item, as_tuple=False)
            if positions.numel() == 0:
                regions.append(torch.zeros_like(item, dtype=torch.float32))
                continue
            center = positions[positions.shape[0] // 2]
            seed = torch.zeros_like(item, dtype=torch.float32)
            seed[center[0], center[1]] = 1.0
            regions.append(_local_patch(seed.unsqueeze(0), radius=2)[0])
        return torch.stack(regions, dim=0)

    def _repair(self, previous: Tensor, ground_truth: Tensor) -> tuple[Tensor, Tensor, str]:
        previous_labels = _labels(previous)
        gt = ground_truth.long()
        error = previous_labels != gt
        region = self._pick_region(error)
        amount = self.min_repair_fraction + (self.max_repair_fraction - self.min_repair_fraction) * 0.5
        amount_map = region.unsqueeze(1) * float(amount)
        target_probs = _one_hot(gt, self.num_classes).to(previous)
        candidate = previous * (1.0 - amount_map) + target_probs * amount_map
        return candidate, region, "local_connected_error_repair"

    def _regress(self, previous: Tensor, ground_truth: Tensor, operation: str) -> tuple[Tensor, Tensor, str]:
        previous_labels = _labels(previous)
        correct = previous_labels == ground_truth.long()
        region = self._pick_region(correct)
        wrong_labels = (previous_labels + 1) % self.num_classes
        wrong_probs = _one_hot(wrong_labels, self.num_classes).to(previous)
        amount = 0.35
        if operation == "boundary_displacement":
            shifted = torch.roll(previous, shifts=(1, -1), dims=(-2, -1))
            candidate = previous * (1.0 - region.unsqueeze(1) * amount) + shifted * (region.unsqueeze(1) * amount)
        elif operation == "semantic_class_swap":
            swapped = previous.clone()
            if self.num_classes > 2:
                swapped[:, 1], swapped[:, 2] = previous[:, 2], previous[:, 1]
            candidate = previous * (1.0 - region.unsqueeze(1) * amount) + swapped * (region.unsqueeze(1) * amount)
        else:
            candidate = previous * (1.0 - region.unsqueeze(1) * amount) + wrong_probs * (region.unsqueeze(1) * amount)
        return candidate, region, operation

    def generate(
        self,
        previous: Tensor,
        ground_truth: Tensor,
        *,
        kind: CounterfactualKind | None = None,
        operation: str | None = None,
    ) -> CounterfactualSample:
        previous_probs = _probabilities(previous).detach()
        gt = _labels(ground_truth).detach()
        selected = kind or self._pick_kind()
        if selected == "positive":
            candidate, _, op = self._repair(previous_probs, gt)
        elif selected == "negative":
            op = operation or self.negative_operations[int(torch.randint(len(self.negative_operations), ()).item())]
            candidate, _, op = self._regress(previous_probs, gt, op)
        elif selected == "hard_neutral":
            positive, positive_region, _ = self._repair(previous_probs, gt)
            negative, negative_region, op = self._regress(previous_probs, gt, operation or "boundary_displacement")
            candidate = positive * (1.0 - negative_region.unsqueeze(1) * 0.35) + negative * (negative_region.unsqueeze(1) * 0.35)
            op = f"repair_plus_{op}"
        elif selected == "on_policy":
            candidate = previous_probs.clone()
            op = "on_policy_placeholder"
        else:
            raise ValueError(f"Unknown counterfactual kind: {selected!r}")
        candidate = candidate.clamp_min(1e-8)
        candidate = candidate / candidate.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if not torch.isfinite(candidate).all():
            raise FloatingPointError("Counterfactual generator produced non-finite probabilities")
        return CounterfactualSample(previous_probs, candidate, str(selected), op)

    def on_policy_transition(self, previous: Tensor, candidate: Tensor) -> CounterfactualSample:
        previous_probs = _probabilities(previous).detach()
        candidate_probs = _probabilities(candidate).detach()
        if previous_probs.shape != candidate_probs.shape:
            raise ValueError("on-policy transition tensors must have identical shapes")
        return CounterfactualSample(previous_probs, candidate_probs, "on_policy", "annotation_expert_transition")

    def generate_batch(self, previous: Tensor, ground_truth: Tensor, **kwargs) -> CounterfactualSample:
        return self.generate(previous, ground_truth, **kwargs)


generate_counterfactual = CounterfactualGenerator().generate
