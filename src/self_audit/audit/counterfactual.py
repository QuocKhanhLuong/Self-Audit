"""Soft, localized counterfactual transitions around real model predictions.

The generator is used only by training.  It never replaces a prediction with
GT globally: every valid edit has an explicit spatial mask and remains on the
probability simplex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor
import torch.nn.functional as F

from .targets import _labels, _probabilities, multiclass_dice


CounterfactualKind = Literal["positive", "negative", "hard_neutral", "on_policy"]


@dataclass
class CounterfactualSample:
    """A generated transition plus inspectable edit provenance.

    ``valid_mask`` is per batch item.  ``valid`` is true only when every item
    in the returned batch has a valid requested edit; training code should use
    ``valid_mask`` when a mixed-validity batch is possible.
    """

    previous_probs: Tensor
    candidate_probs: Tensor
    kind: str
    operation: str
    valid: bool = True
    edit_mask: Tensor | None = None
    strength: Tensor | None = None
    repair_strength: Tensor | None = None
    regression_strength: Tensor | None = None
    source_class: Tensor | None = None
    target_class: Tensor | None = None
    actual_delta_dice: Tensor | None = None
    valid_mask: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def previous(self) -> Tensor:
        return self.previous_probs

    @property
    def candidate(self) -> Tensor:
        return self.candidate_probs

    @property
    def delta_dice(self) -> Tensor | None:
        return self.actual_delta_dice


@dataclass
class _SingleEdit:
    candidate: Tensor
    edit_mask: Tensor
    strength: Tensor
    repair_strength: Tensor
    regression_strength: Tensor
    source_class: Tensor
    target_class: Tensor
    valid: bool
    operation: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalise(probabilities: Tensor) -> Tensor:
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("Counterfactual input contains non-finite probabilities")
    probabilities = probabilities.clamp_min(1e-8)
    if probabilities.ndim == 3:
        class_dim = 0
    elif probabilities.ndim == 4:
        class_dim = 1
    else:
        raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got {tuple(probabilities.shape)}")
    result = probabilities / probabilities.sum(dim=class_dim, keepdim=True).clamp_min(1e-8)
    if not torch.isfinite(result).all() or bool((result < 0).any()):
        raise FloatingPointError("Counterfactual normalization produced invalid probabilities")
    return result


def _one_hot(labels: Tensor, num_classes: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    return F.one_hot(labels.long(), num_classes=num_classes).permute(2, 0, 1).to(device=device, dtype=dtype)


def _uniform(low: float, high: float, device: torch.device) -> Tensor:
    return torch.empty((), device=device, dtype=torch.float32).uniform_(float(low), float(high))


def _binary_dilate(mask: Tensor, radius: int = 1) -> Tensor:
    radius = max(int(radius), 0)
    if radius == 0:
        return mask.bool()
    kernel = 2 * radius + 1
    return F.max_pool2d(mask.float()[None, None], kernel, stride=1, padding=radius)[0, 0] > 0


def _binary_erode(mask: Tensor, radius: int = 1) -> Tensor:
    radius = max(int(radius), 0)
    if radius == 0:
        return mask.bool()
    kernel = 2 * radius + 1
    inverse = (~mask.bool()).float()[None, None]
    return F.max_pool2d(inverse, kernel, stride=1, padding=radius)[0, 0] == 0


def _connected_components(mask: Tensor) -> list[Tensor]:
    """Return 4-connected components without adding a SciPy dependency."""

    mask_cpu = mask.detach().to(device="cpu", dtype=torch.bool)
    visited = torch.zeros_like(mask_cpu)
    components: list[Tensor] = []
    height, width = mask_cpu.shape
    for y, x in torch.nonzero(mask_cpu, as_tuple=False).tolist():
        if bool(visited[y, x]):
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        component = torch.zeros_like(mask_cpu)
        while stack:
            cy, cx = stack.pop()
            component[cy, cx] = True
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and bool(mask_cpu[ny, nx]) and not bool(visited[ny, nx]):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        components.append(component.to(device=mask.device))
    return components


def _grow_region(
    component: Tensor,
    *,
    fraction_low: float = 0.35,
    fraction_high: float = 0.75,
    max_count: int | None = None,
) -> Tensor:
    """Select a connected partial region from one connected component."""

    positions = torch.nonzero(component, as_tuple=False)
    if positions.numel() == 0:
        return torch.zeros_like(component, dtype=torch.bool)
    target_fraction = float(_uniform(fraction_low, fraction_high, component.device))
    target_count = max(1, min(int(positions.shape[0]), int(round(positions.shape[0] * target_fraction))))
    if max_count is not None:
        target_count = min(target_count, max(int(max_count), 1))
    start = positions[int(torch.randint(positions.shape[0], (), device=component.device).item())]
    selected = torch.zeros_like(component, dtype=torch.bool)
    frontier = [(int(start[0]), int(start[1]))]
    selected[start[0], start[1]] = True
    while frontier and int(selected.sum()) < target_count:
        cy, cx = frontier.pop(0)
        neighbours = [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]
        # A random order makes repeated counterfactuals explore different
        # local regions while retaining connectedness.
        order = torch.randperm(len(neighbours), device=component.device).tolist()
        for index in order:
            ny, nx = neighbours[index]
            if 0 <= ny < component.shape[0] and 0 <= nx < component.shape[1] and bool(component[ny, nx]) and not bool(selected[ny, nx]):
                selected[ny, nx] = True
                frontier.append((ny, nx))
                if int(selected.sum()) >= target_count:
                    break
    return selected


def _choose_component(mask: Tensor, *, min_area: int = 1, avoid: Tensor | None = None) -> Tensor | None:
    components = [component for component in _connected_components(mask) if int(component.sum()) >= int(min_area)]
    if avoid is not None:
        components = [component for component in components if not bool((component & avoid).any())]
    if not components:
        return None
    index = int(torch.randint(len(components), (), device=mask.device).item())
    return components[index]


def _choose_foreground(
    labels: Tensor,
    num_classes: int,
    *,
    min_area: int = 1,
    avoid: Tensor | None = None,
) -> tuple[int, Tensor] | None:
    choices: list[tuple[int, Tensor]] = []
    for cls in range(1, int(num_classes)):
        selected = _choose_component(labels == cls, min_area=min_area, avoid=avoid)
        if selected is not None:
            choices.append((cls, selected))
    if not choices:
        return None
    index = int(torch.randint(len(choices), (), device=labels.device).item())
    return choices[index]


def _invalid_edit(previous: Tensor, operation: str, *, metadata: dict[str, Any] | None = None) -> _SingleEdit:
    zeros = torch.zeros(previous.shape[-2:], dtype=torch.bool, device=previous.device)
    zero = previous.new_zeros(())
    minus_one = torch.tensor(-1, dtype=torch.long, device=previous.device)
    return _SingleEdit(
        candidate=previous.clone(),
        edit_mask=zeros,
        strength=zero,
        repair_strength=zero,
        regression_strength=zero,
        source_class=minus_one,
        target_class=minus_one,
        valid=False,
        operation=operation,
        metadata={"reason": "no_valid_local_region", **(metadata or {})},
    )


def _transfer_to_background(previous: Tensor, region: Tensor, source: int, strength: Tensor) -> Tensor:
    candidate = previous.clone()
    mask = region.to(dtype=previous.dtype) * strength.to(dtype=previous.dtype)
    transfer = previous[source] * mask
    candidate[source] = candidate[source] - transfer
    candidate[0] = candidate[0] + transfer
    return _normalise(candidate)


def _transfer_between(previous: Tensor, region: Tensor, source: int, target: int, strength: Tensor) -> Tensor:
    candidate = previous.clone()
    mask = region.to(dtype=previous.dtype) * strength.to(dtype=previous.dtype)
    transfer = previous[source] * mask
    candidate[source] = candidate[source] - transfer
    candidate[target] = candidate[target] + transfer
    return _normalise(candidate)


def _localized_shift(previous: Tensor, region: Tensor, dy: int, dx: int, strength: Tensor) -> Tensor:
    """Shift probabilities only at selected boundary pixels."""

    candidate = previous.clone()
    height, width = region.shape
    for y, x in torch.nonzero(region, as_tuple=False).tolist():
        source_y = min(max(int(y) - int(dy), 0), height - 1)
        source_x = min(max(int(x) - int(dx), 0), width - 1)
        amount = strength.to(dtype=previous.dtype)
        candidate[:, y, x] = (1.0 - amount) * previous[:, y, x] + amount * previous[:, source_y, source_x]
    return _normalise(candidate)


class CounterfactualGenerator:
    """Generate distinct localized soft repairs, regressions, and mixtures."""

    _ALIASES = {
        "erosion": "local_erosion",
        "dilation": "local_dilation",
        "boundary": "boundary_displacement",
        "hole": "hole_insertion",
        "island": "false_island",
        "deletion": "component_deletion",
        "class_swap": "semantic_class_swap",
    }

    def __init__(
        self,
        *,
        positive_fraction: float = 0.4,
        negative_fraction: float = 0.4,
        hard_neutral_fraction: float = 0.2,
        min_repair_fraction: float = 0.3,
        max_repair_fraction: float = 0.6,
        epsilon_neutral: float = 0.02,
        neutral_max_retries: int = 8,
        num_classes: int = 4,
    ) -> None:
        fractions = (float(positive_fraction), float(negative_fraction), float(hard_neutral_fraction))
        if any(value < 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-5:
            raise ValueError("counterfactual sampling fractions must be non-negative and sum to one")
        if not 0.0 < min_repair_fraction <= max_repair_fraction <= 1.0:
            raise ValueError("repair fraction must satisfy 0 < min <= max <= 1")
        if float(epsilon_neutral) < 0:
            raise ValueError("epsilon_neutral must be non-negative")
        if int(neutral_max_retries) < 1:
            raise ValueError("neutral_max_retries must be positive")
        self.fractions = fractions
        self.min_repair_fraction = float(min_repair_fraction)
        self.max_repair_fraction = float(max_repair_fraction)
        self.epsilon_neutral = float(epsilon_neutral)
        self.neutral_max_retries = int(neutral_max_retries)
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

    def _positive_single(self, previous: Tensor, ground_truth: Tensor) -> _SingleEdit:
        labels = previous.argmax(dim=0)
        error = labels != ground_truth
        component = _choose_component(error)
        if component is None:
            return _invalid_edit(previous, "local_connected_error_repair")
        region = _grow_region(component)
        strength = _uniform(self.min_repair_fraction, self.max_repair_fraction, previous.device).to(previous.dtype)
        target = _one_hot(ground_truth, self.num_classes, previous.dtype, previous.device)
        mask = region.to(dtype=previous.dtype) * strength
        candidate = _normalise(previous * (1.0 - mask.unsqueeze(0)) + target * mask.unsqueeze(0))
        positions = torch.nonzero(region, as_tuple=False)
        source_class = labels[positions[0, 0], positions[0, 1]].long()
        target_class = ground_truth[positions[0, 0], positions[0, 1]].long()
        return _SingleEdit(
            candidate=candidate,
            edit_mask=region,
            strength=strength,
            repair_strength=strength,
            regression_strength=previous.new_zeros(()),
            source_class=source_class,
            target_class=target_class,
            valid=True,
            operation="local_connected_error_repair",
            metadata={"repair_strength": float(strength.detach()), "component_area": int(component.sum())},
        )

    def _negative_single(
        self,
        previous: Tensor,
        ground_truth: Tensor,
        operation: str,
        *,
        avoid: Tensor | None = None,
    ) -> _SingleEdit:
        operation = self._ALIASES.get(operation, operation)
        labels = previous.argmax(dim=0)
        strength = _uniform(0.35, 0.75, previous.device).to(previous.dtype)
        chosen = _choose_foreground(labels, self.num_classes, min_area=1, avoid=avoid)

        if operation == "false_island":
            foreground = labels > 0
            background = (labels == 0) & ~_binary_dilate(foreground, radius=2)
            if avoid is not None:
                background = background & ~avoid
            component = _choose_component(background, min_area=1)
            if component is None:
                return _invalid_edit(previous, operation)
            small_island_limit = max(4, min(32, int(component.shape[0] * component.shape[1] / 64)))
            region = _grow_region(component, fraction_low=0.15, fraction_high=0.4, max_count=small_island_limit)
            source = int(torch.randint(1, self.num_classes, (), device=previous.device).item())
            candidate = _transfer_between(previous, region, 0, source, strength)
            return _SingleEdit(
                candidate, region, strength, previous.new_zeros(()), strength,
                torch.tensor(0, dtype=torch.long, device=previous.device),
                torch.tensor(source, dtype=torch.long, device=previous.device), True, operation,
                {"regression_strength": float(strength.detach()), "topology": "false_positive_island"},
            )

        if chosen is None:
            return _invalid_edit(previous, operation)
        source, component = chosen

        if operation == "local_erosion":
            region = component & ~_binary_erode(component)
            if avoid is not None:
                region = region & ~avoid
            if not bool(region.any()):
                return _invalid_edit(previous, operation)
            region = _grow_region(region, fraction_low=0.5, fraction_high=0.9)
            candidate = _transfer_to_background(previous, region, source, strength)
            metadata = {"morphology": "boundary_removed"}
        elif operation == "local_dilation":
            region = _binary_dilate(component) & ~component
            if avoid is not None:
                region = region & ~avoid
            if not bool(region.any()):
                return _invalid_edit(previous, operation)
            region = _grow_region(region, fraction_low=0.35, fraction_high=0.8)
            candidate = _transfer_between(previous, region, 0, source, strength)
            metadata = {"morphology": "foreground_expanded"}
        elif operation == "boundary_displacement":
            region = component & ~_binary_erode(component)
            if avoid is not None:
                region = region & ~avoid
            if not bool(region.any()):
                return _invalid_edit(previous, operation)
            region = _grow_region(region, fraction_low=0.4, fraction_high=0.85)
            dy = int(torch.randint(-2, 3, (), device=previous.device).item())
            dx = int(torch.randint(-2, 3, (), device=previous.device).item())
            if dy == 0 and dx == 0:
                dx = 1
            candidate = _localized_shift(previous, region, dy, dx, strength)
            metadata = {"shift": (dy, dx)}
        elif operation == "hole_insertion":
            interior = _binary_erode(component)
            if int(interior.sum()) < 4:
                return _invalid_edit(previous, operation)
            if avoid is not None:
                interior = interior & ~avoid
            if not bool(interior.any()):
                return _invalid_edit(previous, operation)
            region = _grow_region(interior, fraction_low=0.35, fraction_high=0.7)
            candidate = _transfer_to_background(previous, region, source, strength)
            metadata = {"topology": "interior_hole"}
        elif operation == "component_deletion":
            chosen = _choose_foreground(labels, self.num_classes, min_area=2, avoid=avoid)
            if chosen is None:
                return _invalid_edit(previous, operation)
            source, component = chosen
            region = component
            candidate = _transfer_to_background(previous, region, source, strength)
            metadata = {"component_area": int(component.sum()), "partial": True}
        elif operation == "semantic_class_swap":
            if self.num_classes <= 2:
                return _invalid_edit(previous, operation)
            region = _grow_region(component, fraction_low=0.3, fraction_high=0.7)
            target_choices = [cls for cls in range(1, self.num_classes) if cls != source]
            target = target_choices[int(torch.randint(len(target_choices), (), device=previous.device).item())]
            candidate = _transfer_between(previous, region, source, target, strength)
            metadata = {"semantic": "local_foreground_class_swap", "target_class": target}
        else:
            raise ValueError(f"Unknown negative counterfactual operation: {operation!r}")

        target_class_value = 0 if operation in {"local_erosion", "hole_insertion", "component_deletion"} else source
        if operation == "semantic_class_swap":
            target_class_value = int(metadata["target_class"])
        return _SingleEdit(
            candidate=candidate,
            edit_mask=region,
            strength=strength,
            repair_strength=previous.new_zeros(()),
            regression_strength=strength,
            source_class=torch.tensor(source, dtype=torch.long, device=previous.device),
            target_class=torch.tensor(target_class_value, dtype=torch.long, device=previous.device),
            valid=True,
            operation=operation,
            metadata={"regression_strength": float(strength.detach()), **metadata},
        )

    def _hard_neutral_single(self, previous: Tensor, ground_truth: Tensor, operation: str | None) -> _SingleEdit:
        best: tuple[float, _SingleEdit, Tensor] | None = None
        for retry_index in range(self.neutral_max_retries):
            positive = self._positive_single(previous, ground_truth)
            if not positive.valid:
                continue
            avoid = _binary_dilate(positive.edit_mask, radius=1)
            selected_operation = operation or self.negative_operations[int(torch.randint(len(self.negative_operations), (), device=previous.device).item())]
            negative = self._negative_single(previous, ground_truth, selected_operation, avoid=avoid)
            if not negative.valid:
                continue
            candidate = _normalise(previous + (positive.candidate - previous) + (negative.candidate - previous))
            delta = multiclass_dice(candidate[None], ground_truth[None], self.num_classes)[0] - multiclass_dice(previous[None], ground_truth[None], self.num_classes)[0]
            combined_mask = positive.edit_mask | negative.edit_mask
            result = _SingleEdit(
                candidate=candidate,
                edit_mask=combined_mask,
                strength=(positive.strength + negative.strength) / 2.0,
                repair_strength=positive.repair_strength,
                regression_strength=negative.regression_strength,
                source_class=positive.source_class,
                target_class=negative.target_class,
                valid=True,
                operation=f"local_repair_plus_{negative.operation}",
                metadata={
                    "actual_delta_dice": float(delta.detach()),
                    "epsilon_neutral": self.epsilon_neutral,
                    "neutral_satisfied": bool(abs(float(delta.detach())) < self.epsilon_neutral),
                    "retry_count": retry_index + 1,
                    "regression_operation": negative.operation,
                },
            )
            distance = abs(float(delta.detach()))
            if best is None or distance < best[0]:
                best = (distance, result, delta)
            if distance < self.epsilon_neutral:
                return result
        if best is None:
            return _invalid_edit(
                previous,
                "local_repair_plus_regression",
                metadata={"neutral_satisfied": False, "retry_count": self.neutral_max_retries},
            )
        return best[1]

    def _pack(
        self,
        previous_probs: Tensor,
        ground_truth: Tensor,
        edits: list[_SingleEdit],
        *,
        kind: str,
    ) -> CounterfactualSample:
        candidates = _normalise(torch.stack([edit.candidate for edit in edits], dim=0))
        edit_mask = torch.stack([edit.edit_mask for edit in edits], dim=0)
        strength = torch.stack([edit.strength for edit in edits], dim=0)
        repair_strength = torch.stack([edit.repair_strength for edit in edits], dim=0)
        regression_strength = torch.stack([edit.regression_strength for edit in edits], dim=0)
        source_class = torch.stack([edit.source_class for edit in edits], dim=0)
        target_class = torch.stack([edit.target_class for edit in edits], dim=0)
        valid_mask = torch.tensor([edit.valid for edit in edits], device=previous_probs.device, dtype=torch.bool)
        actual_delta = multiclass_dice(candidates, ground_truth, self.num_classes) - multiclass_dice(previous_probs, ground_truth, self.num_classes)
        operations = sorted({edit.operation for edit in edits})
        operation = operations[0] if len(operations) == 1 else "mixed"
        metadata = {
            "per_sample": [edit.metadata for edit in edits],
            "valid_mask": valid_mask,
        }
        return CounterfactualSample(
            previous_probs=previous_probs,
            candidate_probs=candidates,
            kind=kind,
            operation=operation,
            valid=bool(valid_mask.all()),
            edit_mask=edit_mask,
            strength=strength,
            repair_strength=repair_strength,
            regression_strength=regression_strength,
            source_class=source_class,
            target_class=target_class,
            actual_delta_dice=actual_delta,
            valid_mask=valid_mask,
            metadata=metadata,
        )

    def generate(
        self,
        previous: Tensor,
        ground_truth: Tensor,
        *,
        kind: CounterfactualKind | None = None,
        operation: str | None = None,
    ) -> CounterfactualSample:
        previous_probs = _normalise(_probabilities(previous).detach())
        gt = _labels(ground_truth).detach()
        if previous_probs.shape[0] != gt.shape[0] or previous_probs.shape[-2:] != gt.shape[-2:]:
            raise ValueError(f"previous/ground_truth shape mismatch: {tuple(previous_probs.shape)}, {tuple(gt.shape)}")
        selected = str(kind or self._pick_kind())
        if selected not in {"positive", "negative", "hard_neutral", "on_policy"}:
            raise ValueError(f"Unknown counterfactual kind: {selected!r}")
        edits: list[_SingleEdit] = []
        for index in range(previous_probs.shape[0]):
            current = previous_probs[index]
            target = gt[index]
            if selected == "positive":
                edit = self._positive_single(current, target)
            elif selected == "negative":
                chosen_operation = operation or self.negative_operations[int(torch.randint(len(self.negative_operations), (), device=current.device).item())]
                edit = self._negative_single(current, target, chosen_operation)
            elif selected == "hard_neutral":
                edit = self._hard_neutral_single(current, target, operation)
            else:
                edit = _invalid_edit(current, "on_policy_requires_candidate")
            edits.append(edit)
        return self._pack(previous_probs, gt, edits, kind=selected)

    def on_policy_transition(self, previous: Tensor, candidate: Tensor) -> CounterfactualSample:
        previous_probs = _normalise(_probabilities(previous).detach())
        candidate_probs = _normalise(_probabilities(candidate).detach())
        if previous_probs.shape != candidate_probs.shape:
            raise ValueError("on-policy transition tensors must have identical shapes")
        edit_mask = (candidate_probs.argmax(dim=1) != previous_probs.argmax(dim=1))
        valid_mask = torch.ones(previous_probs.shape[0], dtype=torch.bool, device=previous_probs.device)
        return CounterfactualSample(
            previous_probs=previous_probs,
            candidate_probs=candidate_probs,
            kind="on_policy",
            operation="annotation_expert_transition",
            valid=True,
            edit_mask=edit_mask,
            strength=edit_mask.float().flatten(1).any(dim=1).to(dtype=previous_probs.dtype),
            valid_mask=valid_mask,
            metadata={"valid_mask": valid_mask},
        )

    def generate_batch(self, previous: Tensor, ground_truth: Tensor, **kwargs: Any) -> CounterfactualSample:
        return self.generate(previous, ground_truth, **kwargs)


generate_counterfactual = CounterfactualGenerator().generate
