"""W2S-Match arbiter: teacher/student target arbitration for unlabeled data.

Pure, detached tensor functions.  Nothing here owns a parameter or an RNG.

Rules (first match wins, per voxel and per view):

1. y_T == y_S and max(c_T, c_S) >= tau          -> (y_T, 1)          agree
2. y_T != y_S and c_T >= tau_v                  -> (y_T, 1)          teacher veto
3. y_T != y_S and c_S >= tau and stable         -> (y_S, alpha)      student override
4. y_T != y_S and c_T >= tau                    -> (y_T, 1 - alpha)  teacher fallback
5. otherwise                                    -> weight 0
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

CutMixBox = tuple[int, int, int, int]  # (y0, y1, x0, x1), full depth

RULE_NAMES = ("pad", "agree", "veto", "override", "fallback", "drop")


def alpha_schedule(update: int, total: int, *, start_fraction: float = 0.2,
                   end_fraction: float = 0.5, alpha_max: float = 0.5) -> float:
    """0 until ``start_fraction``, linear to ``alpha_max`` at ``end_fraction``, then constant."""

    if total <= 0 or update < 0:
        raise ValueError("alpha_schedule requires total > 0 and update >= 0")
    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("alpha schedule fractions must satisfy 0 <= start < end <= 1")
    progress = update / total
    if progress <= start_fraction:
        return 0.0
    if progress >= end_fraction:
        return float(alpha_max)
    return float(alpha_max) * (progress - start_fraction) / (end_fraction - start_fraction)


def _probs(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 5:
        raise ValueError(f"{name} must be [B,C,D,H,W] probabilities")
    return value.detach().float()


@torch.no_grad()
def arbitrate(
    p_teacher: Tensor,
    p_student: Tensor,
    stable: Tensor,
    valid: Tensor,
    alpha: float,
    *,
    tau: float = 0.95,
    tau_v: float = 0.99,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(y, w, rule)``; ``rule`` is 0 on invalid voxels, 1..5 elsewhere."""

    p_t = _probs(p_teacher, "p_teacher")
    p_s = _probs(p_student, "p_student")
    if p_t.shape != p_s.shape:
        raise ValueError("teacher and student probabilities must share shape")
    spatial = (p_t.shape[0], *p_t.shape[2:])
    stable = stable.detach().bool()
    valid = valid.detach().bool()
    if tuple(stable.shape) != spatial or tuple(valid.shape) != spatial:
        raise ValueError("stable/valid must be [B,D,H,W]")
    if not (math.isfinite(alpha) and 0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0,1]")
    c_t, y_t = p_t.max(dim=1)
    c_s, y_s = p_s.max(dim=1)
    same = y_t == y_s
    r1 = same & (torch.maximum(c_t, c_s) >= tau)
    r2 = ~same & (c_t >= tau_v)
    r3 = ~same & (c_s >= tau) & stable
    r4 = ~same & (c_t >= tau)
    rule = torch.full_like(y_t, 5)
    # Apply in reverse priority so that the first matching rule wins.
    for index, hit in ((4, r4), (3, r3), (2, r2), (1, r1)):
        rule = torch.where(hit, torch.full_like(rule, index), rule)
    rule = torch.where(valid, rule, torch.zeros_like(rule))
    ones = torch.ones_like(c_t)
    weight = torch.zeros_like(c_t)
    weight = torch.where(rule == 1, ones, weight)
    weight = torch.where(rule == 2, ones, weight)
    weight = torch.where(rule == 3, ones * float(alpha), weight)
    weight = torch.where(rule == 4, ones * (1.0 - float(alpha)), weight)
    y = torch.where(rule == 3, y_s, y_t)
    return y.long(), weight, rule.long()


def _in_box(shape: Sequence[int], box: CutMixBox, device: torch.device) -> Tensor:
    """Boolean ``[1,1,H,W]`` mask of the replaced region (full depth)."""

    mask = torch.zeros((1, 1, int(shape[-2]), int(shape[-1])), dtype=torch.bool, device=device)
    y0, y1, x0, x1 = (int(v) for v in box)
    mask[..., y0:y1, x0:x1] = True
    return mask


@torch.no_grad()
def source_aligned_stability(
    y_student_a: Tensor,
    y_student_b: Tensor,
    strong_argmax: Sequence[Tensor],
    fp_argmax: Tensor,
    boxes: Sequence[CutMixBox],
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """Source-aligned stability flags and per-view mapping into mixed coordinates.

    ``y_student_a`` / ``y_student_b``: clean-weak student argmax of sources A and B
    on A's canvas ``[B,D,H,W]``.  ``strong_argmax[v]``: student argmax on the
    mixed strong view ``v`` (A outside ``boxes[v]``, B inside).  ``fp_argmax``:
    student argmax on the unmixed FP view of A.

    Visible views of a source-A voxel: fp always, view v iff x not in box v.
    Visible views of a source-B voxel: view v iff x in box v (no fp).
    A voxel is stable iff at least two views are visible and every visible view
    equals the clean-weak student label of its own source voxel.

    Returns ``(stable_a, stable_b, per_view)`` where ``stable_a``/``stable_b``
    are source-coordinate flags and ``per_view[v]`` maps them into view ``v``
    (fp is the last entry and equals ``stable_a``).
    """

    if len(strong_argmax) != len(boxes):
        raise ValueError("one box per strong view is required")
    shape = y_student_a.shape
    device = y_student_a.device
    count_a = torch.ones(shape, dtype=torch.long, device=device)  # fp is always visible for A
    ok_a = fp_argmax == y_student_a
    count_b = torch.zeros(shape, dtype=torch.long, device=device)
    ok_b = torch.ones(shape, dtype=torch.bool, device=device)
    masks = []
    for view, box in zip(strong_argmax, boxes):
        inside = _in_box(shape, box, device).expand(shape)
        masks.append(inside)
        outside = ~inside
        count_a = count_a + outside.long()
        ok_a = ok_a & (~outside | (view == y_student_a))
        count_b = count_b + inside.long()
        ok_b = ok_b & (~inside | (view == y_student_b))
    stable_a = ok_a & (count_a >= 2)
    stable_b = ok_b & (count_b >= 2)
    per_view = []
    for inside in masks:
        per_view.append(torch.where(inside, stable_b, stable_a))
    per_view.append(stable_a)
    return stable_a, stable_b, per_view


@torch.no_grad()
def fg_disagreement(p_teacher: Tensor, p_student: Tensor, valid: Tensor) -> Tensor:
    """Voxels where teacher and student argmax differ and at least one is foreground."""

    y_t, y_s = p_teacher.argmax(1), p_student.argmax(1)
    return (y_t != y_s) & ((y_t > 0) | (y_s > 0)) & valid.bool()


def rule_statistics(rule: Tensor, y: Tensor, valid: Tensor, *, num_classes: int = 4) -> dict[str, object]:
    """Detached per-rule voxel fractions overall and per (output) class."""

    valid = valid.bool()
    denominator = float(valid.sum().clamp_min(1))
    overall = {RULE_NAMES[i]: float(((rule == i) & valid).sum()) / denominator for i in range(1, 6)}
    per_class: dict[int, dict[str, float]] = {}
    for cls in range(num_classes):
        selected = valid & (y == cls)
        denominator_c = float(selected.sum().clamp_min(1))
        per_class[cls] = {RULE_NAMES[i]: float(((rule == i) & selected).sum()) / denominator_c for i in range(1, 6)}
    return {"overall": overall, "per_class": per_class}


__all__ = ["RULE_NAMES", "alpha_schedule", "arbitrate", "fg_disagreement", "rule_statistics", "source_aligned_stability"]
