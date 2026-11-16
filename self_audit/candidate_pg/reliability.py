"""Reliability state and detached voxel weighting for Candidate PG."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn


def _valid4(valid: Tensor, shape: Sequence[int]) -> Tensor:
    if not isinstance(valid, Tensor):
        raise TypeError("valid must be a tensor")
    if valid.ndim == len(shape) + 1 and valid.shape[1] == 1:
        valid = valid[:, 0]
    if tuple(valid.shape) != tuple(shape):
        raise ValueError(f"valid must have shape {tuple(shape)}")
    if valid.dtype.is_complex or (valid.dtype.is_floating_point and not bool(torch.isfinite(valid).all())):
        raise ValueError("valid must be finite and real")
    return valid.bool()


def _probabilities(probabilities: Tensor) -> Tensor:
    if not isinstance(probabilities, Tensor) or probabilities.ndim < 3:
        raise ValueError("probabilities must have shape [B,C,*spatial]")
    if probabilities.shape[1] < 2 or not probabilities.dtype.is_floating_point:
        raise ValueError("probabilities must be floating point with at least two classes")
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
        raise ValueError("probabilities must be finite and nonnegative")
    mass = probabilities.sum(dim=1)
    if bool((mass <= 0).any()):
        raise ValueError("probabilities must have positive class mass at every cell")
    # The engine emits softmax values.  Normalize tiny floating-point drift but
    # do not silently turn arbitrary scores into probabilities.
    return probabilities / mass.unsqueeze(1)


def _clip_pair(clip: Sequence[float]) -> tuple[float, float]:
    if not isinstance(clip, Sequence) or len(clip) != 2:
        raise ValueError("clip must be [low, high]")
    low, high = (float(v) for v in clip)
    if not math.isfinite(low) or not math.isfinite(high) or not (0 <= low <= high <= 1):
        raise ValueError("clip must satisfy 0 <= low <= high <= 1")
    return low, high


@torch.no_grad()
def class_adaptive_threshold_update(
    tau_g: float | Tensor,
    p_tilde: Tensor | Sequence[float],
    probabilities: Tensor,
    valid: Tensor,
    ema: float = 0.999,
    clip: Sequence[float] = (0.5, 0.95),
) -> tuple[Tensor, Tensor, Tensor]:
    """Update global/class probability EMAs and return class thresholds.

    ``probabilities`` is the one combined A/B observation for one teacher (or
    one student) state.  Only cells selected by ``valid`` contribute.  The
    returned state is detached and can be stored directly in a checkpoint.
    """

    ema = float(ema)
    if not math.isfinite(ema) or not 0 <= ema < 1:
        raise ValueError("ema must be finite and in [0,1)")
    low, high = _clip_pair(clip)
    probs = _probabilities(probabilities)
    support = _valid4(valid, (probs.shape[0], *probs.shape[2:]))
    if not bool(support.any()):
        raise ValueError("threshold update has no valid support")
    dtype = torch.float64 if probs.dtype == torch.float64 else torch.float32
    values = probs.to(dtype=dtype)
    selected = support.unsqueeze(1)
    count = support.sum().to(dtype=dtype)
    observed_global = values.amax(dim=1)[support].mean()
    observed_class = (values * selected.to(dtype=values.dtype)).sum(dim=tuple([0, *range(2, values.ndim)])) / count
    if isinstance(tau_g, Tensor):
        if tau_g.numel() != 1:
            raise ValueError("tau_g must be scalar")
        tau_old = tau_g.detach().to(device=values.device, dtype=dtype).reshape(())
    else:
        tau_old = torch.tensor(float(tau_g), device=values.device, dtype=dtype)
    if not bool(torch.isfinite(tau_old)) or float(tau_old) < 0 or float(tau_old) > 1:
        raise ValueError("tau_g must be finite and in [0,1]")
    p_old = torch.as_tensor(p_tilde, device=values.device, dtype=dtype).detach().reshape(-1)
    if p_old.numel() != values.shape[1] or not bool(torch.isfinite(p_old).all()) or bool((p_old < 0).any()) or float(p_old.sum()) <= 0:
        raise ValueError("p_tilde must be a finite nonnegative vector with one value per class")
    # ``count`` is intentionally evaluated above so a future caller cannot
    # accidentally replace the mean with an unmasked reduction.
    del count
    tau_new = ema * tau_old + (1.0 - ema) * observed_global
    p_new = ema * p_old + (1.0 - ema) * observed_class
    denominator = p_new.max().clamp_min(torch.finfo(p_new.dtype).tiny)
    tau_c = (tau_new * p_new / denominator).clamp(min=low, max=high)
    return tau_new.detach(), p_new.detach(), tau_c.detach()


class ClassAdaptiveThreshold(nn.Module):
    """Checkpointable per-class threshold state for one prediction stream."""

    def __init__(
        self,
        num_classes: int,
        *,
        ema: float = 0.999,
        clip: Sequence[float] = (0.5, 0.95),
        tau_g: float | None = None,
        p_tilde: Tensor | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or int(num_classes) < 2:
            raise ValueError("num_classes must be an integer >= 2")
        self.num_classes = int(num_classes)
        self.ema = float(ema)
        self.clip = _clip_pair(clip)
        if not math.isfinite(self.ema) or not 0 <= self.ema < 1:
            raise ValueError("ema must be finite and in [0,1)")
        initial_tau = 1.0 / self.num_classes if tau_g is None else float(tau_g)
        initial_p = torch.full((self.num_classes,), 1.0 / self.num_classes, dtype=torch.float32)
        if p_tilde is not None:
            initial_p = torch.as_tensor(p_tilde, dtype=torch.float32).reshape(-1)
        if initial_p.numel() != self.num_classes or not bool(torch.isfinite(initial_p).all()) or bool((initial_p < 0).any()):
            raise ValueError("p_tilde must have one finite nonnegative value per class")
        if not math.isfinite(initial_tau):
            raise ValueError("tau_g must be finite")
        self.register_buffer("tau_g", torch.tensor(initial_tau, dtype=torch.float32))
        self.register_buffer("p_tilde", initial_p.clone())

    @property
    def thresholds(self) -> Tensor:
        denominator = self.p_tilde.max().clamp_min(torch.finfo(self.p_tilde.dtype).tiny)
        return (self.tau_g * self.p_tilde / denominator).clamp(*self.clip)

    def update(self, probabilities: Tensor, valid: Tensor) -> Tensor:
        tau_g, p_tilde, tau_c = class_adaptive_threshold_update(
            self.tau_g, self.p_tilde, probabilities, valid, ema=self.ema, clip=self.clip,
        )
        # Rebind (rather than mutate) the buffers so a caller holding the
        # ordinary ``state_dict()`` snapshot gets an immutable resume point.
        # Both names remain registered buffers under nn.Module.
        self.tau_g = tau_g.to(device=self.tau_g.device, dtype=self.tau_g.dtype)
        self.p_tilde = p_tilde.to(device=self.p_tilde.device, dtype=self.p_tilde.dtype)
        return tau_c.to(device=self.tau_g.device, dtype=self.tau_g.dtype)

    def forward(self, probabilities: Tensor, valid: Tensor) -> Tensor:
        return self.update(probabilities, valid)

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}, ema={self.ema:g}, clip={self.clip}"


def boundary_band(
    d_T: Tensor,
    *,
    radius_mm: float = 5.0,
    near_mm: float = 2.5,
) -> Tensor:
    """Return voxels with any class SDF inside the normalized near band."""

    if not isinstance(d_T, Tensor) or d_T.ndim < 4:
        raise ValueError("d_T must have shape [B,C,*spatial] or [B,*spatial]")
    radius_mm, near_mm = float(radius_mm), float(near_mm)
    if not math.isfinite(radius_mm) or not math.isfinite(near_mm) or not (0 < near_mm < radius_mm):
        raise ValueError("near_mm and radius_mm must satisfy 0 < near_mm < radius_mm")
    if not d_T.dtype.is_floating_point or not bool(torch.isfinite(d_T).all()):
        raise ValueError("d_T must be finite floating point")
    values = d_T.abs() if d_T.ndim >= 5 else d_T.abs().unsqueeze(1)
    return values.amin(dim=1) < (near_mm / radius_mm)


def voxel_weights(
    pW: Tensor,
    pT_region: Tensor,
    pS_cleanweak: Tensor,
    valid: Tensor,
    thresholds: Tensor | Sequence[float] | float,
    d_T: Tensor | None = None,
    *,
    boundary: Tensor | None = None,
    radius_mm: float = 5.0,
    near_mm: float = 2.5,
) -> Tensor:
    """Compute detached PG weights from fusion, teacher, and student views.

    The class label and confidence come from fusion weak ``pW``; teacher and
    clean-student probabilities are sampled at that same class for agreement.
    This distinction matters when the fusion argmax disagrees with the frozen
    teacher argmax.
    """

    fusion = _probabilities(pW)
    teacher = _probabilities(pT_region)
    student = _probabilities(pS_cleanweak)
    if fusion.shape != teacher.shape or teacher.shape != student.shape:
        raise ValueError("pW, pT_region, and pS_cleanweak must share shape")
    expected_valid_shape = (fusion.shape[0], *fusion.shape[2:])
    support = _valid4(valid, expected_valid_shape)
    confidence, labels = fusion.max(dim=1)
    selected_teacher = teacher.gather(1, labels.unsqueeze(1)).squeeze(1)
    selected_student = student.gather(1, labels.unsqueeze(1)).squeeze(1)
    agreement = (selected_teacher * selected_student).clamp_min(0).sqrt()
    threshold_values = torch.as_tensor(thresholds, device=fusion.device, dtype=fusion.dtype).reshape(-1)
    if threshold_values.numel() == 1:
        tau_y = threshold_values.expand_as(confidence)
    elif threshold_values.numel() == fusion.shape[1]:
        tau_y = threshold_values[labels]
    else:
        raise ValueError("thresholds must be scalar or one value per class")
    if boundary is None:
        boundary_mask = boundary_band(d_T, radius_mm=radius_mm, near_mm=near_mm) if d_T is not None else torch.zeros_like(confidence, dtype=torch.bool)
    else:
        boundary_mask = _valid4(boundary, tuple(confidence.shape)).bool()
    if tuple(boundary_mask.shape) != tuple(confidence.shape):
        raise ValueError("boundary mask must align with probability spatial shape")
    inside = confidence
    outside = (confidence >= tau_y).to(dtype=fusion.dtype)
    result = torch.where(boundary_mask, inside, outside) * agreement * support.to(dtype=fusion.dtype)
    return result.detach()


__all__ = [
    "ClassAdaptiveThreshold",
    "boundary_band",
    "class_adaptive_threshold_update",
    "voxel_weights",
]
