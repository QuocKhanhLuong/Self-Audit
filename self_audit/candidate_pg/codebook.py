"""Candidate PG S3 teacher codebook and frozen CBKD branch.

This module deliberately has no dependency on the PG engine.  The teacher is
trainable during S3 and owns the complete ``[K,D]`` codebook; the student branch
copies that codebook after S3 and never exposes it to autograd.  All tensors use
native ``[B,C,D,H,W]`` grids and an optional boolean valid-cell mask.  Padding
cells are excluded from every reducer when a mask is supplied.

The small pure functions at the top are also the engine's stable loss API:
``rho`` computes the ordered soft assignments, ``vq_ste`` is the identity
straight-through estimator, and ``assign_kl`` is KL(sg(rho_T)||rho_S).
"""

from __future__ import annotations

import math
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CODEBOOK_VERSION = "candidate_pg.codebook.v2"
_IGNORE_INDEX = 255
_DEFAULT_CLASSES = 4


def _autocast_disabled(device: torch.device):
    """Return a context that keeps code assignments in float32.

    S3 may run under the global GPU autocast context.  Distances and nearest
    code selection are part of the code-order contract, so they must not be
    evaluated in bf16/fp16.
    """

    return torch.autocast(device_type=device.type, enabled=False)


def _as_options(config: Any, *, preferred: str | None = None) -> dict[str, Any]:
    """Extract codebook options from a PG config without requiring a schema.

    The engine passes the complete PG mapping while focused unit tests often
    pass the ``codebook`` or ``cbkd`` mapping directly.  We intentionally do
    not reject unrelated PG keys here; PG's config validator owns that schema.
    """

    if config is None:
        return {}
    if isinstance(config, Mapping):
        root = dict(config)
    elif hasattr(config, "__dict__"):
        root = dict(vars(config))
    else:
        raise TypeError("codebook config must be a mapping or object with attributes")
    keys = (preferred,) if preferred is not None else ("codebook", "cbkd")
    for key in keys:
        value = root.get(key)
        if isinstance(value, Mapping):
            selected = dict(value)
            nested = selected.get("codebook")
            if isinstance(nested, Mapping):
                selected.update(dict(nested))
            return selected
    model = root.get("model")
    if isinstance(model, Mapping):
        for key in keys:
            value = model.get(key)
            if isinstance(value, Mapping):
                selected = dict(value)
                nested = selected.get("codebook")
                if isinstance(nested, Mapping):
                    selected.update(dict(nested))
                return selected
    return root


def _option(options: Mapping[str, Any], *names: str, default: Any) -> Any:
    for name in names:
        if name in options:
            return options[name]
    return default


def _finite_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _validate_codebook(C: Tensor) -> tuple[Tensor, int, int]:
    if not isinstance(C, Tensor) or C.ndim != 2:
        raise ValueError("codebook must have shape [K,D]")
    if C.shape[0] < 2 or C.shape[1] < 1:
        raise ValueError("codebook must contain at least two entries and one dimension")
    if not torch.is_floating_point(C) or not torch.isfinite(C).all():
        raise ValueError("codebook must be a finite floating tensor")
    return C, int(C.shape[0]), int(C.shape[1])


def _validate_z(z: Tensor, dimension: int | None = None) -> tuple[Tensor, tuple[int, ...], int]:
    if not isinstance(z, Tensor) or z.ndim < 2:
        raise ValueError("z must have shape [...,D]")
    if not torch.is_floating_point(z) or not torch.isfinite(z).all():
        raise ValueError("z must be a finite floating tensor")
    dim = int(z.shape[-1])
    if dimension is not None and dim != int(dimension):
        raise ValueError(f"z embedding dimension must be {dimension}, got {dim}")
    return z, tuple(z.shape[:-1]), dim


def rho(
    z: Tensor,
    C: Tensor,
    temperature: float = 1.0,
    *,
    channel_first: bool = False,
) -> Tensor:
    """Return ordered soft assignments ``softmax(-||z-C||² / temperature)``.

    ``z`` is any tensor whose final axis is the embedding dimension and ``C``
    is ``[K,D]``.  Code order is never sorted or otherwise transformed.
    """

    C, _, code_dim = _validate_codebook(C)
    if channel_first is None:
        channel_first = bool(z.ndim == 5 and z.shape[1] == code_dim and z.shape[-1] != code_dim)
    if channel_first:
        if z.ndim != 5 or z.shape[1] != code_dim:
            raise ValueError("channel-first z must have shape [B,C,D,H,W]")
        original = tuple(z.shape)
        z = z.permute(0, 2, 3, 4, 1)
    z, shape, dim = _validate_z(z)
    if dim != code_dim:
        raise ValueError("z and codebook embedding dimensions differ")
    tau = _finite_positive(temperature, "temperature")
    # Assignment precision is a contract: a surrounding AMP context must not
    # change code ordering or nearest-code IDs.  ``.float()`` is differentiable
    # and therefore keeps the teacher's codebook/projection gradients intact.
    with _autocast_disabled(z.device):
        z_fp32 = z.float()
        C_fp32 = C.to(device=z.device, dtype=torch.float32)
        distances = (z_fp32.reshape(-1, dim).square().sum(1, keepdim=True)
                     + C_fp32.square().sum(1).unsqueeze(0)
                     - 2.0 * z_fp32.reshape(-1, dim) @ C_fp32.T)
        # Round-off can make exact self-distances slightly negative.  This does
        # not change ordering and keeps the function finite.
        distances = distances.clamp_min(0.0)
        probabilities = torch.softmax((-distances / tau), dim=-1).reshape(*shape, C.shape[0])
    if channel_first:
        probabilities = probabilities.permute(0, 4, 1, 2, 3).contiguous()
        if tuple(probabilities.shape) != (original[0], C.shape[0], original[2], original[3], original[4]):
            raise RuntimeError("channel-first rho shape construction failed")
    return probabilities


def vq_ste(
    z: Tensor,
    C: Tensor,
    assignments: Tensor | None = None,
    *,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Nearest-code quantization with identity STE through ``z``.

    The default return is the straight-through tensor.  ``return_indices``
    additionally returns hard indices and the selected code vectors (the latter
    remains connected to ``C`` for the teacher's codebook loss).
    """

    z, shape, dim = _validate_z(z)
    C, K, code_dim = _validate_codebook(C)
    if dim != code_dim:
        raise ValueError("z and codebook embedding dimensions differ")
    with _autocast_disabled(z.device):
        z_fp32 = z.float()
        C_fp32 = C.to(device=z.device, dtype=torch.float32)
        if assignments is None:
            assignments = rho(z_fp32, C_fp32).argmax(dim=-1)
        else:
            if tuple(assignments.shape) != shape or assignments.dtype.is_floating_point:
                raise ValueError("assignments must match z's leading shape and be integer")
            assignments = assignments.to(device=z.device, dtype=torch.long)
            if bool((assignments < 0).any()) or bool((assignments >= K).any()):
                raise ValueError("assignments contain an invalid code index")
        selected = F.embedding(assignments.reshape(-1), C_fp32).reshape(*shape, dim)
        straight_through = z_fp32 + (selected - z_fp32).detach()
    if return_indices:
        return straight_through, assignments, selected
    return straight_through


def _broadcast_mask(mask: Tensor | None, shape: tuple[int, ...], *, device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if not isinstance(mask, Tensor):
        mask = torch.as_tensor(mask, device=device)
    mask = mask.to(device=device)
    if mask.ndim == len(shape) + 1 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(f"valid_mask must have shape {shape} (or [B,1,*shape[1:]])")
    if mask.dtype != torch.bool:
        if not torch.is_floating_point(mask) and mask.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("valid_mask must be boolean or binary numeric")
        if not torch.isfinite(mask.float()).all() or bool(((mask != 0) & (mask != 1)).any()):
            raise ValueError("valid_mask must contain only finite binary values")
    return mask.bool()


def _masked_mean(values: Tensor, mask: Tensor | None = None, weights: Tensor | None = None) -> Tensor:
    if values.ndim == 0:
        return values
    support = torch.ones_like(values, dtype=torch.bool) if mask is None else mask.bool()
    if tuple(support.shape) != tuple(values.shape):
        try:
            support = torch.broadcast_to(support, values.shape)
        except RuntimeError as exc:
            raise ValueError("mask is not broadcastable to loss values") from exc
    factor = torch.ones_like(values, dtype=values.dtype)
    if weights is not None:
        factor = torch.as_tensor(weights, device=values.device, dtype=values.dtype)
        try:
            factor = torch.broadcast_to(factor, values.shape)
        except RuntimeError as exc:
            raise ValueError("weights are not broadcastable to loss values") from exc
        if not torch.isfinite(factor).all() or bool((factor < 0).any()):
            raise ValueError("loss weights must be finite and nonnegative")
    support_factor = factor * support.to(values.dtype)
    denominator = support_factor.sum()
    if float(denominator.detach()) <= 0.0:
        return values.sum() * 0.0
    return (values * support_factor).sum() / denominator


def assign_kl(
    rho_teacher: Tensor,
    rho_student: Tensor,
    valid_mask: Tensor | None = None,
    weights: Tensor | None = None,
    *,
    reduction: str = "mean",
    channel_first: bool = False,
) -> Tensor:
    """KL(sg(rho_teacher)||rho_student), optionally reduced over valid cells."""

    if not isinstance(rho_teacher, Tensor) or not isinstance(rho_student, Tensor):
        raise TypeError("assignment distributions must be tensors")
    if rho_teacher.shape != rho_student.shape or rho_teacher.ndim < 2:
        raise ValueError("teacher and student assignments must share [...,K], K>=2")
    if not torch.is_floating_point(rho_teacher) or not torch.is_floating_point(rho_student):
        raise ValueError("assignment distributions must be floating tensors")
    if not torch.isfinite(rho_teacher).all() or not torch.isfinite(rho_student).all():
        raise ValueError("assignment distributions must be finite")
    if bool((rho_teacher < 0).any()) or bool((rho_student < 0).any()):
        raise ValueError("assignment distributions must be nonnegative")
    # Engine/model outputs use [B,K,D,H,W], while the pure helper naturally
    # uses [...,K].  A valid mask with a singleton channel is an unambiguous
    # channel-first marker; callers can also state it explicitly.
    if channel_first:
        if rho_teacher.ndim != 5 or rho_teacher.shape[1] < 2:
            raise ValueError("channel-first assignments must have shape [B,K,D,H,W]")
        pt = rho_teacher.detach().permute(0, 2, 3, 4, 1)
        ps = rho_student.permute(0, 2, 3, 4, 1)
        if valid_mask is not None:
            valid_mask = _broadcast_mask(
                valid_mask, tuple(rho_teacher.shape[0:1] + rho_teacher.shape[2:]), device=rho_teacher.device,
            )
        if weights is not None and isinstance(weights, Tensor) and weights.ndim == 5 and weights.shape[1] == 1:
            weights = weights[:, 0].to(device=rho_teacher.device, dtype=torch.float32)
    else:
        if rho_teacher.shape[-1] < 2:
            raise ValueError("teacher and student assignments must have K>=2")
        pt = rho_teacher.detach()
        ps = rho_student
    # KL is evaluated in float32 even when the student branch is under AMP.
    # Casting the live student tensor preserves its gradient while the teacher
    # remains an explicit stop-gradient target.
    with _autocast_disabled(rho_student.device):
        pt = pt.float()
        ps = ps.float()
        pt = pt / pt.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        ps = ps / ps.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        values = (pt * (pt.clamp_min(1e-12).log() - ps.clamp_min(1e-12).log())).sum(dim=-1)
    if reduction == "none":
        return values
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'none'")
    return _masked_mean(values, valid_mask, weights)


def commitment_loss(z: Tensor, e_nearest: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    """Mean squared ``z - sg(e_nearest)`` over valid cells."""

    if z.shape != e_nearest.shape or z.ndim < 2:
        raise ValueError("commitment tensors must share [...,D]")
    values = (z - e_nearest.detach()).square().mean(dim=-1)
    return _masked_mean(values, valid_mask)


def _labels_for_logits(labels: Tensor, spatial: tuple[int, int, int]) -> Tensor:
    if not isinstance(labels, Tensor):
        labels = torch.as_tensor(labels)
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 4 or labels.dtype.is_floating_point or labels.dtype.is_complex:
        raise ValueError("labels must be integer [B,D,H,W] or [B,1,D,H,W]")
    labels = labels.long()
    if tuple(labels.shape[1:]) != spatial:
        # S3's GT is already pooled with physical cell support by
        # ``_pooled_gt``.  Resizing raw labels here would silently change the
        # declared (1,8,8) lattice and hide a geometry bug.
        raise ValueError("hard labels must already match the pooled q_logits grid")
    observed = torch.unique(labels.detach())
    if bool(((observed != _IGNORE_INDEX) & ((observed < 0) | (observed >= _DEFAULT_CLASSES))).any()):
        raise ValueError("labels contain values outside 0..3 and ignore 255")
    return labels


def soft_cross_entropy(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    weights: Tensor | None = None,
) -> Tensor:
    """Masked CE for hard labels or soft class probabilities."""

    if not isinstance(logits, Tensor) or logits.ndim != 5 or logits.shape[1] != _DEFAULT_CLASSES:
        raise ValueError("logits must have shape [B,4,D,H,W]")
    spatial = tuple(int(v) for v in logits.shape[2:])
    if target.ndim == logits.ndim and torch.is_floating_point(target) and target.shape != logits.shape:
        raise ValueError("pooled soft CE targets must exactly match q_logits grid")
    if target.ndim == logits.ndim and target.shape == logits.shape:
        if not torch.is_floating_point(target) or not torch.isfinite(target).all() or bool((target < 0).any()):
            raise ValueError("soft CE targets must be finite nonnegative probabilities")
        probabilities = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)
        values = -(probabilities * F.log_softmax(logits, dim=1)).sum(dim=1)
        target_valid = torch.ones_like(values, dtype=torch.bool)
    else:
        labels = _labels_for_logits(target, spatial).to(device=logits.device)
        target_valid = labels != _IGNORE_INDEX
        safe = labels.clamp(0, _DEFAULT_CLASSES - 1)
        values = F.cross_entropy(logits, safe, reduction="none")
    if weights is not None and isinstance(weights, Tensor) and weights.ndim == 5 and weights.shape[1] == 1:
        weights = weights[:, 0]
    if valid_mask is None:
        support = target_valid
    else:
        support = _broadcast_mask(valid_mask, tuple(logits.shape[0:1] + logits.shape[2:]), device=logits.device)
        support = support & target_valid
    return _masked_mean(values, support, weights)


def soft_dice_loss(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    weights: Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Soft-target multiclass Dice, skipping classes with no target mass."""

    if not isinstance(logits, Tensor) or logits.ndim != 5 or logits.shape[1] != _DEFAULT_CLASSES:
        raise ValueError("logits must have shape [B,4,D,H,W]")
    spatial = tuple(int(v) for v in logits.shape[2:])
    if target.ndim == logits.ndim and torch.is_floating_point(target) and target.shape != logits.shape:
        raise ValueError("pooled soft Dice targets must exactly match q_logits grid")
    if target.ndim == logits.ndim and target.shape == logits.shape:
        if not torch.is_floating_point(target) or not torch.isfinite(target).all() or bool((target < 0).any()):
            raise ValueError("soft Dice targets must be finite nonnegative probabilities")
        target_prob = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)
        target_valid = torch.ones(logits.shape[0], *spatial, dtype=torch.bool, device=logits.device)
    else:
        labels = _labels_for_logits(target, spatial).to(device=logits.device)
        target_valid = labels != _IGNORE_INDEX
        safe = labels.clamp(0, _DEFAULT_CLASSES - 1)
        target_prob = F.one_hot(safe, num_classes=_DEFAULT_CLASSES).permute(0, 4, 1, 2, 3).to(logits.dtype)
    if valid_mask is None:
        support = target_valid
    else:
        support = _broadcast_mask(valid_mask, tuple(logits.shape[0:1] + logits.shape[2:]), device=logits.device) & target_valid
    factor = torch.ones_like(support, dtype=logits.dtype) if weights is None else torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)
    if factor.ndim == 5 and factor.shape[1] == 1:
        factor = factor[:, 0]
    factor = torch.broadcast_to(factor, support.shape)
    if not torch.isfinite(factor).all() or bool((factor < 0).any()):
        raise ValueError("soft Dice weights must be finite and nonnegative")
    mask = factor * support.to(logits.dtype)
    probabilities = torch.softmax(logits, dim=1)
    target_prob = target_prob.to(device=logits.device, dtype=logits.dtype)
    intersection = (probabilities * target_prob * mask.unsqueeze(1)).sum(dim=(0, 2, 3, 4))
    denominator = ((probabilities + target_prob) * mask.unsqueeze(1)).sum(dim=(0, 2, 3, 4))
    target_mass = (target_prob * mask.unsqueeze(1)).sum(dim=(0, 2, 3, 4))
    active = target_mass > 0
    if not bool(active.any()):
        return logits.sum() * 0.0
    dice = (2.0 * intersection[active] + eps) / (denominator[active] + eps)
    return 1.0 - dice.mean()


def soft_task_loss(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    weights: Tensor | None = None,
) -> Tensor:
    """The S3/CBKD Q-task: soft-target CE plus soft-target Dice."""

    return soft_cross_entropy(logits, target, valid_mask, weights) + soft_dice_loss(logits, target, valid_mask, weights)


def usage_metrics(
    assignments: Tensor,
    codebook_size: int,
    valid_mask: Tensor | None = None,
    labels: Tensor | None = None,
) -> dict[str, Any]:
    """Compute usage entropy/perplexity, dead-code count, and class coverage."""

    if not isinstance(assignments, Tensor) or assignments.ndim != 4:
        raise ValueError("assignments must have shape [B,D,H,W]")
    K = _integer(codebook_size, "codebook_size", minimum=2)
    if assignments.dtype.is_floating_point or bool((assignments < 0).any()) or bool((assignments >= K).any()):
        raise ValueError("assignments contain invalid code indices")
    mask = _broadcast_mask(valid_mask, tuple(assignments.shape), device=assignments.device)
    flat = assignments.long()[mask]
    counts = torch.bincount(flat, minlength=K).to(dtype=torch.float64)
    total = counts.sum()
    frequencies = counts / total.clamp_min(1.0)
    entropy_tensor = -(frequencies * frequencies.clamp_min(1e-12).log()).sum()
    entropy = float(entropy_tensor.item()) if total > 0 else 0.0
    perplexity = float(entropy_tensor.exp().item()) if total > 0 else 0.0
    dead = int((counts == 0).sum().item())
    result: dict[str, Any] = {
        "usage": frequencies.detach(),
        "usage_frequency": frequencies.detach().cpu().tolist(),
        "usage_entropy": entropy,
        "entropy": entropy,
        "perplexity": perplexity,
        "dead_code_count": dead,
        "dead_code_fraction": dead / K,
        "valid_count": int(flat.numel()),
        "codes_per_class": {class_id: 0 for class_id in range(_DEFAULT_CLASSES)},
    }
    if labels is not None:
        if isinstance(labels, Tensor) and labels.ndim == 5 and labels.shape[1] == _DEFAULT_CLASSES and torch.is_floating_point(labels):
            if tuple(labels.shape[2:]) != tuple(assignments.shape[1:]):
                raise ValueError("soft labels must share the assignment grid")
            y = labels.detach().argmax(dim=1).to(device=assignments.device)
        else:
            y = _labels_for_logits(labels, tuple(assignments.shape[1:])).to(device=assignments.device)
        class_coverage: dict[int, int] = {}
        for class_id in range(_DEFAULT_CLASSES):
            selected = mask & (y == class_id)
            class_codes = torch.unique(assignments.long()[selected])
            class_coverage[class_id] = int(class_codes.numel())
        result["codes_per_class"] = class_coverage
    return result


def _feature_to_flat(z: Tensor, channels: int) -> tuple[Tensor, tuple[int, int, int], tuple[int, int, int, int]]:
    if not isinstance(z, Tensor) or z.ndim != 5 or z.shape[1] != channels:
        raise ValueError(f"features must have shape [B,{channels},D,H,W]")
    if not torch.is_floating_point(z) or not torch.isfinite(z).all():
        raise ValueError("features must be finite floating tensors")
    batch, _, depth, height, width = z.shape
    flat = z.permute(0, 2, 3, 4, 1).reshape(-1, channels)
    return flat, (depth, height, width), (batch, depth, height, width)


def _mask_for_feature(valid_mask: Tensor | None, shape: tuple[int, int, int, int], device: torch.device) -> Tensor:
    return _broadcast_mask(valid_mask, shape, device=device)


class TeacherCodebook(nn.Module):
    """Trainable S3 teacher projection, codebook and pooled anatomy head."""

    def __init__(self, config: Mapping[str, Any] | Any | None = None) -> None:
        super().__init__()
        options = _as_options(config, preferred="codebook")
        self.teacher_channels = _integer(_option(options, "teacher_channels", "P_T_in", default=256), "teacher_channels")
        self.embedding_dim = _integer(_option(options, "embedding_dim", "code_dim", "dim", default=64), "embedding_dim")
        self.codebook_size = _integer(_option(options, "codebook_size", "size", default=128), "codebook_size")
        self.num_classes = _integer(_option(options, "num_classes", "classes", default=4), "num_classes")
        if self.num_classes != 4:
            raise ValueError("Candidate PG codebook has exactly four anatomy classes")
        self.temperature = _finite_positive(_option(options, "rho_temperature", "temperature", default=1.0), "rho_temperature")
        self.commit_beta = float(_option(options, "commit_beta", "commitment_beta", default=0.25))
        self.usage_entropy_weight = float(_option(options, "usage_entropy_weight", "entropy_weight", default=0.01))
        self.dead_code_restart_every = _integer(_option(options, "dead_code_restart_every", default=500), "dead_code_restart_every")
        self.dead_code_threshold = float(_option(options, "dead_code_threshold", default=0.0))
        self.seed = int(_option(options, "seed", default=42))
        if not math.isfinite(self.commit_beta) or self.commit_beta < 0:
            raise ValueError("commit_beta must be finite and nonnegative")
        if not math.isfinite(self.usage_entropy_weight) or self.usage_entropy_weight < 0:
            raise ValueError("usage_entropy_weight must be finite and nonnegative")
        if not math.isfinite(self.dead_code_threshold) or not 0.0 <= self.dead_code_threshold < 1.0:
            raise ValueError("dead_code_threshold must be in [0,1)")

        self.P_T = nn.Conv3d(self.teacher_channels, self.embedding_dim, kernel_size=1)
        self.C = nn.Parameter(torch.empty(self.codebook_size, self.embedding_dim))
        nn.init.normal_(self.C, mean=0.0, std=0.02)
        self.H_T_Q = nn.Sequential(
            nn.Conv3d(self.embedding_dim, self.embedding_dim, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv3d(self.embedding_dim, self.num_classes, kernel_size=1),
        )
        self.register_buffer("update_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_restart_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("restart_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("usage_ema", torch.zeros(self.codebook_size, dtype=torch.float64))
        # ``usage_ema`` is diagnostic only.  Dead-code decisions use exact
        # assignment counts in the current restart window so a code that was
        # used long ago does not remain alive forever.
        self.register_buffer("usage_window_counts", torch.zeros(self.codebook_size, dtype=torch.float64))
        self._restart_generator = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        self._frozen = False

    @property
    def codebook(self) -> Tensor:
        return self.C

    @property
    def codebook_sha256(self) -> str:
        payload = self.C.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    @property
    def frozen(self) -> bool:
        return self._frozen

    def _output(self, feature: Tensor, valid_mask: Tensor | None) -> dict[str, Tensor]:
        flat, spatial, shape = _feature_to_flat(feature, self.teacher_channels)
        # Keep the projection and all code assignments in fp32 under AMP.  The
        # projection remains trainable because ``feature.float()`` is a normal
        # differentiable cast and C is not detached on this teacher path.
        with _autocast_disabled(feature.device):
            z_map = self.P_T(feature.float())
            z_flat = z_map.permute(0, 2, 3, 4, 1).reshape(-1, self.embedding_dim).float()
            probabilities = rho(z_flat, self.C.float(), self.temperature)
            assignments = probabilities.argmax(dim=-1)
            quantized_flat = F.embedding(assignments, self.C.float())
            straight_flat = z_flat + (quantized_flat - z_flat).detach()
        quantized = quantized_flat.reshape(shape[0], *spatial, self.embedding_dim).permute(0, 4, 1, 2, 3)
        straight = straight_flat.reshape(shape[0], *spatial, self.embedding_dim).permute(0, 4, 1, 2, 3)
        assignments_map = assignments.reshape(shape)
        rho_map = probabilities.reshape(shape[0], *spatial, self.codebook_size).permute(0, 4, 1, 2, 3)
        q_logits = self.H_T_Q(straight)
        valid = _mask_for_feature(valid_mask, shape, feature.device).unsqueeze(1)
        return {
            "z": z_map,
            "rho": rho_map,
            "assignments": assignments_map,
            "nearest": assignments_map,
            "e_nearest": quantized,
            "quantized": quantized,
            "straight_through": straight,
            "q_logits": q_logits,
            "q_probs": torch.softmax(q_logits, dim=1),
            "valid_mask": valid,
        }

    def forward(self, feature: Tensor, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        return self._output(feature, valid_mask)

    def _coerce_output(self, output_or_feature: Mapping[str, Tensor] | Tensor, valid_mask: Tensor | None) -> dict[str, Tensor]:
        if isinstance(output_or_feature, Mapping):
            output = dict(output_or_feature)
            required = {"z", "rho", "assignments", "e_nearest", "q_logits"}
            if not required.issubset(output):
                raise ValueError(f"codebook output missing keys: {sorted(required - set(output))}")
            return output
        return self.forward(output_or_feature, valid_mask)

    def loss(
        self,
        output_or_feature: Mapping[str, Tensor] | Tensor,
        labels: Tensor | None = None,
        *,
        valid_mask: Tensor | None = None,
        pseudo_labels: Tensor | None = None,
        pseudo_confidence: Tensor | None = None,
        labeled: bool = True,
        return_components: bool = False,
    ) -> Tensor | dict[str, Tensor | dict[str, Any]]:
        """S3 objective, with strict L/U target isolation.

        ``labeled=True`` consumes pooled GT only; ``labeled=False`` consumes
        detached pseudo labels and confidence weights only.  Both paths train
        P_T/C/H_T_Q through VQ codebook, commitment and anatomy terms.
        """

        output = self._coerce_output(output_or_feature, valid_mask)
        z = output["z"]
        if z.ndim != 5 or z.shape[1] != self.embedding_dim:
            raise ValueError("output z has an unexpected shape")
        shape = tuple(int(v) for v in z.shape[0:1] + z.shape[2:])
        mask = _mask_for_feature(output.get("valid_mask", valid_mask), shape, z.device)
        rho_map = output["rho"]
        assignments = output["assignments"]
        e_nearest = output["e_nearest"]
        if tuple(rho_map.shape[2:]) != tuple(z.shape[2:]):
            raise ValueError("rho and z grids differ")
        z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, self.embedding_dim)
        e_flat = e_nearest.permute(0, 2, 3, 4, 1).reshape(-1, self.embedding_dim)
        mask_flat = mask.reshape(-1)
        vq_values = (z_flat.detach() - e_flat).square().mean(dim=1)
        commit_values = (z_flat - e_flat.detach()).square().mean(dim=1)
        vq = _masked_mean(vq_values, mask_flat)
        commit = _masked_mean(commit_values, mask_flat)
        usage = rho_map.permute(0, 2, 3, 4, 1).reshape(-1, self.codebook_size)
        usage = _masked_mean(usage, mask_flat.unsqueeze(-1)).clamp_min(1e-12)
        usage = usage / usage.sum().clamp_min(1e-12)
        entropy_penalty = (usage * usage.log()).sum()
        qtask = z.sum() * 0.0
        if labeled:
            if pseudo_labels is not None:
                raise ValueError("labeled S3 loss cannot consume U pseudo labels")
            if labels is not None:
                qtask = soft_task_loss(output["q_logits"], labels, mask)
        else:
            if labels is not None:
                raise ValueError("U S3 loss cannot consume GT labels")
            if pseudo_labels is not None:
                pseudo = pseudo_labels.detach()
                confidence = None if pseudo_confidence is None else pseudo_confidence.detach()
                qtask = soft_task_loss(output["q_logits"], pseudo, mask, confidence)
        qpred = z.sum() * 0.0  # S3 defines this diagnostic term, but no Qpred weight.
        total = vq + self.commit_beta * commit + self.usage_entropy_weight * entropy_penalty + qtask
        result: dict[str, Tensor | dict[str, Any]] = {
            "loss": total,
            "vq": vq,
            "codebook": vq,
            "commitment": commit,
            "qtask": qtask,
            "qpred": qpred,
            "entropy_penalty": entropy_penalty,
            "usage_entropy": -entropy_penalty.detach(),
            "metrics": usage_metrics(assignments.detach(), self.codebook_size, mask, labels),
        }
        return result if return_components else total

    @torch.no_grad()
    def update_usage_batch(
        self,
        samples: Sequence[tuple[Tensor, Tensor | None, Tensor | None]],
        *,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Accumulate one declared L/U1/U2 batch and restart once per window.

        Deadness is defined as *no assignment in the last restart window*,
        not as an EMA threshold.  The exact window counts are checkpointed in
        ``usage_window_counts``; ``usage_ema`` remains a diagnostic signal.
        """

        if not isinstance(samples, Sequence) or not samples:
            raise ValueError("S3 usage batch must contain at least one sample")
        current = int(self.update_count.item()) + 1 if step is None else _integer(step, "step", minimum=0)
        self.update_count.fill_(current)
        counts = torch.zeros(self.codebook_size, dtype=torch.float64, device=self.C.device)
        candidates: list[Tensor] = []
        for item in samples:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ValueError("S3 usage samples must be (assignments, valid_mask, features)")
            assignments, valid_mask, features = item
            if not isinstance(assignments, Tensor) or assignments.ndim != 4:
                raise ValueError("assignments must have shape [B,D,H,W]")
            mask = _broadcast_mask(valid_mask, tuple(assignments.shape), device=assignments.device)
            selected = assignments.detach().long()[mask]
            counts += torch.bincount(selected, minlength=self.codebook_size).to(device=counts.device, dtype=counts.dtype)
            if features is not None:
                flat, _, feature_shape = _feature_to_flat(features.detach(), self.embedding_dim)
                support = _mask_for_feature(valid_mask, feature_shape, features.device).reshape(-1)
                if bool(support.any()):
                    candidates.append(flat[support])
        total = counts.sum()
        frequency = counts / total.clamp_min(1.0)
        self.usage_window_counts.add_(counts.to(device=self.usage_window_counts.device))
        self.usage_ema.mul_(0.99).add_(frequency, alpha=0.01)
        restarted: list[int] = []
        if current % self.dead_code_restart_every == 0:
            dead = torch.nonzero(self.usage_window_counts == 0, as_tuple=False).flatten()
            if dead.numel() and candidates:
                candidate_pool = torch.cat(candidates, dim=0)
                if candidate_pool.numel():
                    indices = torch.randint(candidate_pool.shape[0], (dead.numel(),), generator=self._restart_generator, device="cpu")
                    replacement = candidate_pool.index_select(0, indices.to(candidate_pool.device)).to(self.C.device, self.C.dtype)
                    self.C.index_copy_(0, dead.to(self.C.device), replacement)
                    restarted = [int(v) for v in dead.cpu().tolist()]
                    self.last_restart_step.fill_(current)
                    self.restart_count.add_(len(restarted))
            # A code used in this window is alive for the next window only if
            # it receives a fresh assignment; do not let historical EMA usage
            # keep it alive indefinitely.
            self.usage_window_counts.zero_()
        entropy_tensor = -(frequency * frequency.clamp_min(1e-12).log()).sum()
        metrics = {
            "usage": frequency.detach(),
            "usage_frequency": frequency.detach().cpu().tolist(),
            "usage_entropy": float(entropy_tensor.item()) if total > 0 else 0.0,
            "entropy": float(entropy_tensor.item()) if total > 0 else 0.0,
            "perplexity": float(entropy_tensor.exp().item()) if total > 0 else 0.0,
            "dead_code_count": int((counts == 0).sum().item()),
            "dead_code_fraction": float((counts == 0).sum().item()) / self.codebook_size,
            "valid_count": int(total.item()),
            "window_dead_code_count": int(dead.numel()) if current % self.dead_code_restart_every == 0 else int((self.usage_window_counts == 0).sum().item()),
        }
        metrics.update({
            "step": current,
            "restarted_codes": restarted,
            "restart_count": int(self.restart_count.item()),
            "last_restart_step": int(self.last_restart_step.item()),
        })
        return metrics

    @torch.no_grad()
    def update_usage(
        self,
        assignments: Tensor,
        valid_mask: Tensor | None = None,
        *,
        step: int | None = None,
        features: Tensor | None = None,
    ) -> dict[str, Any]:
        """Compatibility wrapper for a one-frame usage update."""

        return self.update_usage_batch([(assignments, valid_mask, features)], step=step)

    def metrics(self, assignments: Tensor, valid_mask: Tensor | None = None, labels: Tensor | None = None) -> dict[str, Any]:
        result = usage_metrics(assignments.detach(), self.codebook_size, valid_mask, labels)
        result.update({
            "step": int(self.update_count.item()),
            "restart_count": int(self.restart_count.item()),
            "last_restart_step": int(self.last_restart_step.item()),
            "frozen": self._frozen,
        })
        return result

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "version": CODEBOOK_VERSION,
            "sha256": self.codebook_sha256,
            "size": self.codebook_size,
            "dimension": self.embedding_dim,
            "frozen": self._frozen,
        }

    @torch.no_grad()
    def restart_dead_codes(
        self,
        assignments: Tensor,
        features: Tensor,
        valid_mask: Tensor | None = None,
        *,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Explicit alias used by the S3 loop for usage/restart bookkeeping."""

        return self.update_usage(assignments, valid_mask, step=step, features=features)

    @torch.no_grad()
    def freeze_codebook(self) -> "TeacherCodebook":
        return self.freeze()

    @torch.no_grad()
    def freeze(self) -> "TeacherCodebook":
        self.requires_grad_(False)
        self._frozen = True
        super().eval()
        return self

    def train(self, mode: bool = True) -> "TeacherCodebook":
        super().train(mode)
        if self._frozen:
            super().train(False)
        return self

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": CODEBOOK_VERSION,
            "frozen": self._frozen,
            "restart_generator_state": self._restart_generator.get_state().clone(),
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("version") != CODEBOOK_VERSION:
            raise ValueError("codebook checkpoint version mismatch")
        self._frozen = bool(state.get("frozen", False))
        generator_state = state.get("restart_generator_state")
        if isinstance(generator_state, Tensor):
            self._restart_generator.set_state(generator_state.detach().cpu())
        if self._frozen:
            self.requires_grad_(False)
            super().eval()


class StudentCodeBranch(nn.Module):
    """Frozen-codebook CBKD branch operating on student post-EPA stage3."""

    def __init__(self, config: Mapping[str, Any] | Any | None, codebook: TeacherCodebook | Tensor) -> None:
        super().__init__()
        options = _as_options(config, preferred="cbkd")
        self.student_channels = _integer(_option(options, "student_channels", "P_S_in", default=128), "student_channels")
        self.embedding_dim = _integer(_option(options, "embedding_dim", "code_dim", "dim", default=64), "embedding_dim")
        self.num_classes = 4
        self.temperature = _finite_positive(_option(options, "rho_temperature", "temperature", default=1.0), "rho_temperature")
        self.commit_beta = float(_option(options, "commit_beta", "commitment_beta", default=0.25))
        self.qtask_weight = float(_option(options, "qtask_weight", default=1.0))
        self.assign_weight = float(_option(options, "assign_weight", default=1.0))
        self.commit_weight = float(_option(options, "commit_weight", default=1.0))
        self.qpred_weight = float(_option(options, "qpred_weight", default=0.0))
        if isinstance(codebook, TeacherCodebook):
            source = codebook.C.detach()
        elif isinstance(codebook, Tensor):
            source = codebook.detach()
        elif hasattr(codebook, "C") and isinstance(codebook.C, Tensor):
            source = codebook.C.detach()
        else:
            raise TypeError("codebook must be TeacherCodebook or [K,D] tensor")
        source, self.codebook_size, source_dim = _validate_codebook(source)
        if source_dim != self.embedding_dim:
            raise ValueError("student embedding_dim differs from frozen codebook")
        # A buffer is intentionally used instead of a Parameter or nested
        # TeacherCodebook, so StudentCodeBranch.parameters() cannot update C.
        self.register_buffer("C", source.clone())
        self.P_S = nn.Conv3d(self.student_channels, self.embedding_dim, kernel_size=1)
        self.H_S_Q = nn.Sequential(
            nn.Conv3d(self.embedding_dim, self.embedding_dim, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv3d(self.embedding_dim, self.num_classes, kernel_size=1),
        )

    @property
    def codebook(self) -> Tensor:
        return self.C

    @property
    def frozen(self) -> bool:
        return True

    @property
    def codebook_sha256(self) -> str:
        payload = self.C.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    def forward(self, feature: Tensor, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        flat, spatial, shape = _feature_to_flat(feature, self.student_channels)
        del flat
        # Student assignments must be bit-stable between fp32 and AMP.  Keep
        # P_S, rho, and nearest-code selection in a disabled-autocast fp32
        # region; gradients through P_S and the STE remain intact while C is a
        # frozen buffer.
        with _autocast_disabled(feature.device):
            z_map = self.P_S(feature.float())
            z_flat = z_map.permute(0, 2, 3, 4, 1).reshape(-1, self.embedding_dim).float()
            probabilities = rho(z_flat, self.C.float(), self.temperature)
            assignments = probabilities.argmax(dim=-1)
            quantized_flat = F.embedding(assignments, self.C.float())
            straight_flat = z_flat + (quantized_flat - z_flat).detach()
        quantized = quantized_flat.reshape(shape[0], *spatial, self.embedding_dim).permute(0, 4, 1, 2, 3)
        straight = straight_flat.reshape(shape[0], *spatial, self.embedding_dim).permute(0, 4, 1, 2, 3)
        assignments_map = assignments.reshape(shape)
        rho_map = probabilities.reshape(shape[0], *spatial, self.codebook_size).permute(0, 4, 1, 2, 3)
        q_logits = self.H_S_Q(straight)
        valid = _mask_for_feature(valid_mask, shape, feature.device).unsqueeze(1)
        return {
            "z": z_map,
            "rho": rho_map,
            "assignments": assignments_map,
            "nearest": assignments_map,
            "e_nearest": quantized,
            "quantized": quantized,
            "straight_through": straight,
            "q_logits": q_logits,
            "q_probs": torch.softmax(q_logits, dim=1),
            "valid_mask": valid,
        }

    def _coerce_output(self, output_or_feature: Mapping[str, Tensor] | Tensor, valid_mask: Tensor | None) -> dict[str, Tensor]:
        if isinstance(output_or_feature, Mapping):
            output = dict(output_or_feature)
            required = {"z", "rho", "assignments", "e_nearest", "q_logits"}
            if not required.issubset(output):
                raise ValueError(f"student code output missing keys: {sorted(required - set(output))}")
            return output
        return self.forward(output_or_feature, valid_mask)

    def loss_assign_kl(self, teacher_rho: Tensor, student_rho: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        return assign_kl(teacher_rho, student_rho, valid_mask, channel_first=True)

    def loss_commit(self, output: Mapping[str, Tensor], valid_mask: Tensor | None = None) -> Tensor:
        z = output["z"].permute(0, 2, 3, 4, 1)
        e = output["e_nearest"].permute(0, 2, 3, 4, 1)
        shape = tuple(int(v) for v in z.shape[:-1])
        mask = _broadcast_mask(output.get("valid_mask", valid_mask), shape, device=z.device)
        return commitment_loss(z, e, mask)

    def loss_qtask(self, output: Mapping[str, Tensor], labels: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        # CBKD's Q-task is soft CE only.  S3's teacher anatomy task is the
        # separate CE+soft-Dice objective.
        return soft_cross_entropy(output["q_logits"], labels, output.get("valid_mask", valid_mask))

    @torch.no_grad()
    def shuffle_codes(
        self,
        output: Mapping[str, Tensor],
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        """Shuffle spatial code vectors while keeping the input/student z fixed.

        This is a VAL-only diagnostic for A3/A4.  It deliberately permutes
        ``e_nearest`` rather than the image or the student feature and never
        feeds shuffled tensors into the optimizer.
        """

        codes = output.get("e_nearest", output.get("quantized"))
        if not isinstance(codes, Tensor) or codes.ndim != 5 or codes.shape[1] != self.embedding_dim:
            raise ValueError("student output must contain [B,embedding_dim,D,H,W] code vectors")
        batch, channels, depth, height, width = codes.shape
        count = depth * height * width
        shuffled = []
        shuffled_assignments = []
        assignments = output.get("assignments", output.get("nearest"))
        if not isinstance(assignments, Tensor) or tuple(assignments.shape) != (batch, depth, height, width):
            raise ValueError("student output assignments must match code-vector spatial shape")
        for index in range(batch):
            permutation = torch.randperm(count, generator=generator, device="cpu").to(codes.device)
            flat = codes[index].permute(1, 2, 3, 0).reshape(count, channels)
            shuffled.append(flat.index_select(0, permutation).reshape(depth, height, width, channels).permute(3, 0, 1, 2))
            shuffled_assignments.append(assignments[index].reshape(-1).index_select(0, permutation).reshape(depth, height, width))
        vectors = torch.stack(shuffled, dim=0)
        return {
            "quantized": vectors,
            "q_logits": self.H_S_Q(vectors),
            "assignments": torch.stack(shuffled_assignments, dim=0).detach(),
            "valid_mask": output.get("valid_mask", torch.ones(batch, 1, depth, height, width, dtype=torch.bool, device=codes.device)),
        }

    def loss(
        self,
        output_or_feature: Mapping[str, Tensor] | Tensor,
        teacher_rho: Tensor | Mapping[str, Tensor] | None = None,
        labels: Tensor | None = None,
        *,
        valid_mask: Tensor | None = None,
        labeled: bool = True,
        return_components: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """CBKD objective; qtask is deliberately L-only and qpred weight is zero."""

        output = self._coerce_output(output_or_feature, valid_mask)
        if teacher_rho is None:
            raise ValueError("student CBKD loss requires teacher rho assignments")
        if isinstance(teacher_rho, Mapping):
            teacher_rho = teacher_rho.get("rho")
        if not isinstance(teacher_rho, Tensor):
            raise TypeError("teacher_rho must be a tensor or output mapping")
        mask = output.get("valid_mask", valid_mask)
        assign = self.loss_assign_kl(teacher_rho, output["rho"], mask)
        commit = self.loss_commit(output, mask)
        qtask = output["z"].sum() * 0.0
        if labeled:
            if labels is not None:
                qtask = self.loss_qtask(output, labels, mask)
        elif labels is not None:
            raise ValueError("U CBKD loss cannot consume GT labels")
        qpred = output["z"].sum() * 0.0
        total = self.assign_weight * assign + self.qtask_weight * qtask + self.commit_weight * self.commit_beta * commit + self.qpred_weight * qpred
        result = {"loss": total, "assign": assign, "assign_kl": assign, "commit": commit, "commitment": commit, "qtask": qtask, "qpred": qpred}
        return result if return_components else total


def _code_output(value: Any) -> Mapping[str, Tensor] | None:
    """Find an already-computed code-branch output in an engine packet."""

    if isinstance(value, Mapping):
        nested = value.get("code_branch")
        if isinstance(nested, Mapping) and "rho" in nested:
            return nested
        if {"rho", "z", "e_nearest", "q_logits"}.issubset(value):
            return value
    return None


def cbkd_loss(
    labeled_student: Mapping[str, Tensor] | Tensor,
    unlabeled_student: Mapping[str, Tensor] | Tensor,
    unlabeled_teacher: Mapping[str, Tensor] | Tensor,
    labels: Tensor | None = None,
    labeled_teacher: Mapping[str, Tensor] | Tensor | None = None,
    *,
    weights: Mapping[str, float] | None = None,
) -> Tensor:
    """Engine-facing CBKD reducer.

    The preferred packet contains ``code_branch`` outputs from
    :class:`StudentCodeBranch` and a teacher packet containing ``rho``.  The
    optional fifth argument supplies the labeled teacher assignment so both L
    and U KL terms can be included.  During early synthetic engine fixtures the
    code branch is intentionally absent; returning a graph-connected zero keeps
    that fixture runnable without inventing a teacher feature/codebook.
    """

    if labeled_teacher is None:
        raise ValueError("CBKD requires labeled_teacher assignments for the L KL term")
    if not isinstance(weights, Mapping):
        raise ValueError("CBKD requires explicit weights for assign/qtask/commit/qpred")
    required_weights = {"assign", "qtask", "commit", "qpred"}
    if set(weights) != required_weights:
        raise ValueError(f"CBKD weights must contain exactly {sorted(required_weights)}")
    parsed_weights = {key: float(weights[key]) for key in required_weights}
    if any(not math.isfinite(value) or value < 0.0 for value in parsed_weights.values()):
        raise ValueError("CBKD weights must be finite and nonnegative")
    s_l = _code_output(labeled_student)
    s_u = _code_output(unlabeled_student)
    t_u = _code_output(unlabeled_teacher)
    if t_u is None and isinstance(unlabeled_teacher, Tensor):
        t_u = {"rho": unlabeled_teacher}
    if t_u is None and isinstance(unlabeled_teacher, Mapping) and isinstance(unlabeled_teacher.get("rho"), Tensor):
        t_u = unlabeled_teacher
    if s_l is None or s_u is None or t_u is None:
        raise ValueError("CBKD requires labeled/U StudentCodeBranch packets and teacher rho")
    t_l = _code_output(labeled_teacher)
    if t_l is None and isinstance(labeled_teacher, Tensor):
        t_l = {"rho": labeled_teacher}
    if t_l is None:
        raise ValueError("CBKD labeled_teacher packet is missing rho assignments")
    valid_u = s_u.get("valid_mask")
    assign_u = assign_kl(t_u["rho"], s_u["rho"], valid_u, channel_first=True)
    assign_l = assign_kl(t_l["rho"], s_l["rho"], s_l.get("valid_mask"), channel_first=True)
    total = parsed_weights["assign"] * (assign_l + assign_u)
    # L-only pooled-GT qtask; no qtask is ever evaluated on U.
    qtask = s_l["z"].sum() * 0.0
    if labels is not None:
        qtask = soft_cross_entropy(s_l["q_logits"], labels, s_l.get("valid_mask"))
    total = total + parsed_weights["qtask"] * qtask
    commits = []
    for packet in (s_l, s_u):
        z = packet["z"].permute(0, 2, 3, 4, 1)
        e = packet["e_nearest"].permute(0, 2, 3, 4, 1)
        mask = packet.get("valid_mask")
        commits.append(commitment_loss(z, e, None if mask is None else mask[:, 0]))
    total = total + parsed_weights["commit"] * sum(commits)
    # qpred remains a declared zero-weight diagnostic in the locked contract.
    total = total + parsed_weights["qpred"] * (s_l["z"].sum() * 0.0)
    return total


def _frame_lists(data: Any) -> tuple[list[Any], list[Any]]:
    """Return labeled/U frame keys or frame objects from a PG/FUSE view."""

    view = getattr(data, "train", data)
    if hasattr(view, "labeled_keys") and hasattr(view, "unlabeled_keys"):
        return list(view.labeled_keys()), list(view.unlabeled_keys())
    labeled = getattr(view, "labeled", None)
    unlabeled = getattr(view, "unlabeled", None)
    if labeled is not None and unlabeled is not None:
        return list(labeled), list(unlabeled)
    if isinstance(view, Mapping) and "labeled" in view and "unlabeled" in view:
        return list(view["labeled"]), list(view["unlabeled"])
    raise TypeError("S3 data must expose labeled_keys/unlabeled_keys or labeled/unlabeled sequences")


def _load_s3_frame(data: Any, key_or_frame: Any, *, labeled: bool) -> Any:
    view = getattr(data, "train", data)
    if hasattr(view, "load") and not hasattr(key_or_frame, "image") and not isinstance(key_or_frame, Mapping):
        key = tuple(key_or_frame) if isinstance(key_or_frame, (tuple, list)) else key_or_frame
        return view.load(key, labeled=labeled)
    return key_or_frame


def _key_patient(key: Any, frame: Any | None = None) -> str | None:
    if isinstance(key, (tuple, list)) and key:
        return str(key[0])
    value = getattr(frame, "patient_id", None)
    if value is None and isinstance(frame, Mapping):
        value = frame.get("patient_id")
    return None if value is None else str(value)


def _s3_frame_tensors(frame: Any, device: torch.device, *, labeled: bool) -> tuple[Tensor, Tensor | None, Mapping[str, Any]]:
    image = torch.as_tensor(getattr(frame, "image", frame["image"] if isinstance(frame, Mapping) else frame), dtype=torch.float32, device=device)
    if image.ndim == 4:
        image = image.unsqueeze(0)
    if image.ndim != 5 or image.shape[1] != 1 or not torch.isfinite(image).all():
        raise ValueError("S3 frame image must be finite [B,1,D,H,W]")
    target = getattr(frame, "target", None)
    if isinstance(frame, Mapping):
        target = frame.get("target", target)
    if labeled:
        if target is None:
            raise ValueError("S3 labeled frame is missing GT")
        target = torch.as_tensor(target, dtype=torch.long, device=device)
        if target.ndim == 3:
            target = target.unsqueeze(0)
        if tuple(target.shape) != tuple(image.shape[0:1] + image.shape[2:]):
            raise ValueError("S3 frame target/image geometry mismatch")
    geometry = getattr(frame, "geometry", None)
    if isinstance(frame, Mapping):
        geometry = frame.get("geometry", geometry)
    if not isinstance(geometry, Mapping) or tuple(geometry.get("shape", ())) != tuple(image.shape[2:]):
        # Synthetic tiny fixtures commonly omit geometry; only use an explicit
        # identity geometry when the tensor itself supplies all native cells.
        if geometry is None:
            geometry = {"shape": tuple(image.shape[2:]), "spacing_zyx": (1.0, 1.0, 1.0), "context": "full_volume"}
        else:
            raise ValueError("S3 frame geometry must declare native shape")
    return image, target, geometry


def _teacher_packet(teacher: nn.Module, image: Tensor, geometry: Mapping[str, Any]) -> Mapping[str, Any]:
    with torch.no_grad():
        packet = teacher(image, geometry, return_features=True)
    if not isinstance(packet, Mapping) or "features" not in packet or "region_logits" not in packet:
        raise ValueError("S3 teacher must return region_logits and five features")
    features = packet["features"]
    if not isinstance(features, (tuple, list)) or len(features) < 4 or not isinstance(features[3], Tensor):
        raise ValueError("S3 teacher features must include level3 256-channel tensor")
    if features[3].shape[1] != 256:
        raise ValueError("S3 teacher level3 feature must have 256 channels")
    return packet


def _aligned_level3(
    feature: Tensor,
    geometry: Mapping[str, Any],
    *,
    teacher: nn.Module,
    student: nn.Module | None,
) -> tuple[Tensor, tuple[tuple[int, int, int], tuple[float, float, float]], Tensor]:
    """Align level3 using live phase metadata; no crop/resample fallback."""

    teacher_phases = getattr(teacher, "feature_phases", None)
    if teacher_phases is None:
        teacher_phases = getattr(teacher, "teacher_phases", None)
    student_phases = getattr(student, "phases", None)
    teacher_phase = teacher_phases[3] if isinstance(teacher_phases, (tuple, list)) and len(teacher_phases) >= 4 else None
    student_phase = student_phases[3] if isinstance(student_phases, (tuple, list)) and len(student_phases) >= 4 else None
    if teacher_phase is None or student_phase is None:
        raise ValueError("S3 alignment requires live teacher.phases[3] and student.phases[3]")
    if feature.ndim != 5 or feature.shape[1] != 256:
        raise ValueError("aligned S3 level3 feature must be [B,256,D,H,W]")
    native = tuple(int(v) for v in geometry["shape"])
    target_shape = tuple((native[i] + int(student_phase[0][i]) - 1) // int(student_phase[0][i]) for i in range(3))
    from ..candidate_q3d.alignment import pool_native_cells

    support_native = torch.ones(1, 1, *native, device=feature.device, dtype=feature.dtype)
    support = pool_native_cells(support_native, native, student_phase)[0, 0] > 0
    support = support.unsqueeze(0).expand(feature.shape[0], -1, -1, -1).contiguous()
    if tuple(feature.shape[-3:]) == target_shape and tuple(teacher_phase[0]) == tuple(student_phase[0]):
        return feature.detach(), student_phase, support
    from ..candidate_fuse.fusion import align_feature

    aligned = align_feature(feature, native, teacher_phase, student_phase, target_shape)
    if tuple(aligned.shape[-3:]) != target_shape:
        raise RuntimeError("S3 level3 alignment did not produce the student grid")
    return aligned.detach(), student_phase, support


@torch.no_grad()
def _initialize_codebook_from_train(
    codebook: TeacherCodebook,
    train_data: Any,
    teacher: nn.Module,
    student: nn.Module | None,
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    """Seed C from randomly selected valid TRAIN teacher projections.

    Visit TRAIN L/U frames in seeded random order until K valid vectors are
    available, then sample K cells without replacement. VAL/TEST are never
    consulted. The support returned by ``_aligned_level3``
    excludes high-side internal padding, and non-finite projected vectors are
    discarded before sampling.  A short pool is an error rather than a reason
    to duplicate a cell or fall back to random weights.
    """

    labeled_keys, unlabeled_keys = _frame_lists(train_data)
    stream = [(key, True) for key in labeled_keys] + [(key, False) for key in unlabeled_keys]
    if not stream:
        raise ValueError("S3 codebook initialization requires TRAIN L/U frames")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    required = codebook.codebook_size
    candidates: list[Tensor] = []
    candidate_count = 0
    frames_visited = 0
    for frame_index in torch.randperm(len(stream), generator=generator).tolist():
        key, labeled = stream[frame_index]
        frames_visited += 1
        frame = _load_s3_frame(train_data, key, labeled=labeled)
        image, _, geometry = _s3_frame_tensors(frame, device, labeled=labeled)
        packet = _teacher_packet(teacher, image, geometry)
        feature, _phase, feature_valid = _aligned_level3(
            packet["features"][3], geometry, teacher=teacher, student=student,
        )
        with _autocast_disabled(device):
            projected = codebook.P_T(feature.float()).float()
        flat = projected.permute(0, 2, 3, 4, 1).reshape(-1, codebook.embedding_dim)
        support = _broadcast_mask(
            feature_valid, tuple(projected.shape[0:1] + projected.shape[2:]), device=device,
        ).reshape(-1)
        finite = torch.isfinite(flat).all(dim=1)
        selected = flat[support & finite]
        if selected.numel():
            candidates.append(selected.detach().cpu())
            candidate_count += int(selected.shape[0])
        if candidate_count >= required:
            break
    if not candidates:
        raise ValueError("S3 codebook initialization found no valid projected TRAIN vectors")
    pool = torch.cat(candidates, dim=0)
    if int(pool.shape[0]) < required:
        raise ValueError(
            f"S3 codebook initialization needs {required} valid TRAIN vectors, found {int(pool.shape[0])}"
        )
    indices = torch.randperm(int(pool.shape[0]), generator=generator)[:required]
    sampled = pool.index_select(0, indices).to(device=codebook.C.device, dtype=codebook.C.dtype)
    codebook.C.copy_(sampled)
    return {
        "source": "TRAIN_L_U_projected_z_T",
        "seed": int(seed),
        "valid_vectors": int(pool.shape[0]),
        "sampled_vectors": int(required),
        "codebook_size": int(required),
        "frames_visited": frames_visited,
        "sampling": "seeded_frame_order_then_cells_without_replacement",
    }


def _pooled_gt(
    target: Tensor,
    native_shape: tuple[int, int, int],
    phase: tuple[tuple[int, int, int], tuple[float, float, float]],
) -> tuple[Tensor, Tensor]:
    from ..candidate_q3d.alignment import pool_native_cells

    labels = target[:, : native_shape[0], : native_shape[1], : native_shape[2]]
    valid_native = labels != _IGNORE_INDEX
    safe = labels.clamp(0, _DEFAULT_CLASSES - 1)
    one_hot = F.one_hot(safe, num_classes=_DEFAULT_CLASSES).permute(0, 4, 1, 2, 3).float()
    pooled = pool_native_cells(one_hot, native_shape, phase, native_mask=valid_native.unsqueeze(1).float())
    support = pool_native_cells(valid_native.unsqueeze(1).float(), native_shape, phase)[:, 0] > 0
    pooled = pooled / pooled.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return pooled.detach(), support.detach()


@torch.no_grad()
def _qval_metrics(
    codebook: TeacherCodebook,
    data: Any,
    teacher: nn.Module,
    student: nn.Module | None,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate qT on the pooled (1,8,8) lattice with patient macro metrics."""

    from .evaluation import patient_macro, pooled_grid_metrics

    labeled, _ = _frame_lists(data)
    if not labeled:
        return {"fg_dice": float("nan"), "perplexity": 0.0, "dead_code_count": codebook.codebook_size}
    frame_rows: list[dict[str, Any]] = []
    usages: list[Tensor] = []
    for key in labeled:
        frame = _load_s3_frame(data, key, labeled=True)
        image, target, geometry = _s3_frame_tensors(frame, device, labeled=True)
        packet = _teacher_packet(teacher, image, geometry)
        feature, phase, feature_valid = _aligned_level3(packet["features"][3], geometry, teacher=teacher, student=student)
        output = codebook(feature, valid_mask=feature_valid)
        pooled, valid = _pooled_gt(target, tuple(image.shape[2:]), phase)
        prediction = output["q_logits"].argmax(1)
        # The pooled GT and qT prediction are already on the live level-3
        # grid.  Pass the physical cell factor and support mask explicitly so
        # HD95/ASSD are measured in mm and internal pad never contributes.
        pooled_reference = pooled.argmax(1)
        pooled_valid = valid & feature_valid
        frame_metric = pooled_grid_metrics(
            pooled_reference[0].detach().cpu().numpy(),
            prediction[0].detach().cpu().numpy(),
            geometry["spacing_zyx"],
            (1, 8, 8),
            valid_mask=pooled_valid[0].detach().cpu().numpy(),
        )
        frame_metric["valid_voxels"] = int(pooled_valid.sum().item())
        frame_metric["valid_grid_shape"] = list(pooled_valid.shape[1:])
        frame_metric["valid_mask_present"] = True
        patient = _key_patient(key, frame)
        if patient is None:
            raise ValueError("S3 qT evaluation frame requires patient_id")
        frame_rows.append({
            "patient": patient,
            "patient_id": patient,
            "frame": int(getattr(frame, "frame_index", 0) if not isinstance(frame, Mapping) else frame.get("frame_index", 0)),
            "metrics": frame_metric,
            "valid_mask": pooled_valid[0].detach().cpu().numpy(),
        })
        usages.append(output["assignments"])
    metrics = patient_macro(frame_rows)
    valid_by_patient: dict[str, int] = {}
    for row in frame_rows:
        valid_by_patient[row["patient"]] = valid_by_patient.get(row["patient"], 0) + int(row["metrics"]["valid_voxels"])
    for patient, count in valid_by_patient.items():
        if isinstance(metrics.get("per_patient"), Mapping) and isinstance(metrics["per_patient"].get(patient), Mapping):
            metrics["per_patient"][patient]["valid_voxels"] = int(count)
    usage = torch.cat(usages, dim=0) if usages else torch.empty(0)
    usage_metrics_value = usage_metrics(usage, codebook.codebook_size) if usages else {"perplexity": 0.0, "dead_code_count": codebook.codebook_size, "dead_code_fraction": 1.0}
    metrics.update({key: value for key, value in usage_metrics_value.items() if key not in {"usage", "codes_per_class"}})
    metrics.pop("usage", None)
    # Keep valid-mask provenance compact in JSON receipts while the actual
    # boolean mask remains attached to each in-memory frame row.
    metrics["valid_voxels"] = int(sum(int(row["metrics"]["valid_voxels"]) for row in frame_rows))
    metrics["valid_grid_shape"] = frame_rows[0]["metrics"]["valid_grid_shape"]
    metrics["frame_metrics"] = [
        {key: value for key, value in row.items() if key != "valid_mask"} | {"valid_mask_present": True}
        for row in frame_rows
    ]
    return metrics


def _atomic_torch_save(value: Any, path: Any) -> str:
    """Atomically publish an S3 checkpoint and return its content hash."""

    target = path if hasattr(path, "parent") else __import__("pathlib").Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save(value, temp)
    os.replace(temp, target)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _logger_checkpoint_state(logger: Any, *, epoch: int, update: int) -> dict[str, Any]:
    """Serialize the append/commit boundary carried by an S3 checkpoint."""

    return {
        "schema": "self_audit.candidate_pg.epoch_logger.v1",
        "row_count": int(len(logger.rows)),
        "epoch": int(epoch),
        "update": int(update),
        "csv_offset": int(logger.csv_path.stat().st_size if logger.csv_path.exists() else 0),
        "jsonl_offset": int(logger.jsonl_path.stat().st_size if logger.jsonl_path.exists() else 0),
    }


def _open_logger_for_resume(root: Any, payload: Mapping[str, Any] | None) -> Any:
    """Recover S3 rows from either side of the checkpoint/commit crash window."""

    from .logging import EpochLogger

    commit_path = root / "epochs.commit.json"
    pending = None
    if isinstance(payload, Mapping):
        pending = payload.get("epoch_logger", payload.get("metrics_receipt"))
    suppress = False
    if isinstance(pending, Mapping) and commit_path.is_file():
        try:
            committed = json.loads(commit_path.read_text(encoding="utf-8"))
            suppress = int(pending["row_count"]) > int(committed["row_count"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            suppress = False
    saved = commit_path.read_bytes() if suppress else None
    if suppress:
        commit_path.unlink()
    try:
        logger = EpochLogger(root, stage="S3", arm="S3")
        if isinstance(pending, Mapping):
            logger.recover_from_checkpoint(pending)
        return logger
    except Exception:
        if saved is not None and not commit_path.exists():
            commit_path.write_bytes(saved)
        raise


def _metric_epoch_row(metric: Mapping[str, Any], *, epoch: int, update: int,
                      split: str, loss_total: float | None) -> dict[str, Any]:
    """Flatten pooled qT patient-macro metrics into the fixed epoch schema."""

    per = metric.get("per_class", {})
    rv = per.get("RV", {}) if isinstance(per, Mapping) else {}
    myo = per.get("MYO", {}) if isinstance(per, Mapping) else {}
    lv = per.get("LV", {}) if isinstance(per, Mapping) else {}
    return {
        "stage": "S3", "arm": "S3", "epoch": int(epoch), "update": int(update),
        "split": split, "path": "qT", "loss_total": loss_total,
        "loss_sup": None, "loss_pg": None, "loss_cbkd": None,
        "fg_dice": metric.get("fg_dice"), "rv_dice": rv.get("dice"),
        "myo_dice": myo.get("dice"), "lv_dice": lv.get("dice"),
        "fg_hd95": metric.get("fg_hd95"), "rv_hd95": rv.get("hd95_mm"),
        "myo_hd95": myo.get("hd95_mm"), "lv_hd95": lv.get("hd95_mm"),
        "fg_assd": metric.get("fg_assd"), "rv_assd": rv.get("assd_mm"),
        "myo_assd": myo.get("assd_mm"), "lv_assd": lv.get("assd_mm"),
        "lr": None, "seconds": 0.0, "best_val": False,
    }


def run_codebook(
    config: Mapping[str, Any],
    out: str | Any,
    train_data: Any,
    val_data: Any,
    test_data: Any,
    teacher: nn.Module,
    student: nn.Module | None = None,
    *,
    device: str | torch.device = "cpu",
    max_updates: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the complete S3 codebook stage with checkpointable CPU/GPU state.

    The function performs no execution in this delivery; it is a production
    entrypoint for the engine's later real run.  All target construction is
    native-grid pooled and all U labels are teacher-detached.  No quality gate
    can stop or rewrite the run.
    """

    if not isinstance(config, Mapping):
        raise TypeError("S3 config must be a mapping")
    from pathlib import Path

    target_device = torch.device(device)
    from .runtime import configure_runtime
    runtime_flags = configure_runtime(target_device, str(config.get("runtime", {}).get("cublas_workspace_config", ":4096:8")))
    options = _as_options(config, preferred="codebook")
    raw_codebook = config.get("codebook")
    if not isinstance(raw_codebook, Mapping) or not isinstance(raw_codebook.get("codebook"), Mapping):
        raise ValueError("S3 requires nested codebook.size and codebook.dim")
    nested_codebook = raw_codebook["codebook"]
    if "size" not in nested_codebook or "dim" not in nested_codebook:
        raise ValueError("S3 requires exact nested codebook.size and codebook.dim")
    codebook_size = _integer(nested_codebook["size"], "codebook.codebook.size")
    embedding_dim = _integer(nested_codebook["dim"], "codebook.codebook.dim")
    model_options = dict(options)
    model_options.update({"teacher_channels": 256, "embedding_dim": embedding_dim, "codebook_size": codebook_size})
    codebook = TeacherCodebook(model_options).to(target_device)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "checkpoint.pt"
    optimizer_options = raw_codebook.get("optimizer")
    if not isinstance(optimizer_options, Mapping):
        raise ValueError("S3 requires the exact codebook.optimizer mapping")
    if optimizer_options.get("name") != "adamw" or "lr" not in optimizer_options or "weight_decay" not in optimizer_options:
        raise ValueError("S3 training.optimizer must declare AdamW lr and weight_decay")
    labeled_keys, unlabeled_keys = _frame_lists(train_data)
    if not labeled_keys or not unlabeled_keys:
        raise ValueError("S3 requires nonempty labeled and U TRAIN streams")
    seed_value = int(config["seed"]) if "seed" in config else int(options.get("seed", 42))
    initialization_meta: dict[str, Any]
    if resume and checkpoint.is_file():
        initialization_meta = {
            "source": "checkpoint",
            "seed": seed_value,
            "codebook_size": int(codebook.codebook_size),
        }
    else:
        # Cold S3 starts from real TRAIN teacher projections, not a random C.
        # Resume loads C below and must never reinitialize it.
        initialization_meta = _initialize_codebook_from_train(
            codebook, train_data, teacher, student, target_device, seed=seed_value,
        )
    optimizer = torch.optim.AdamW(
        codebook.parameters(),
        lr=float(optimizer_options["lr"]),
        weight_decay=float(optimizer_options["weight_decay"]),
    )
    sampler = None
    synthetic = bool(isinstance(config.get("runtime"), Mapping) and config["runtime"].get("synthetic", False))
    try:
        from .data import PGSampler

        sampler = PGSampler(getattr(train_data, "train", train_data), seed=42)
    except (TypeError, ValueError):
        if not synthetic:
            raise
        # Explicit synthetic fixtures may expose sequences without a
        # patient-balanced FuseData view; real PGData always takes this path.
        sampler = None
    if "updates" not in raw_codebook:
        raise ValueError("S3 codebook config must declare updates")
    configured_updates = _integer(raw_codebook["updates"], "codebook updates")
    updates = configured_updates
    training_cfg = config.get("training")
    if isinstance(training_cfg, Mapping) and "updates_per_epoch" in training_cfg:
        updates_per_epoch = _integer(training_cfg["updates_per_epoch"], "training.updates_per_epoch")
        if not synthetic and updates_per_epoch != 80:
            raise ValueError("real PG S3 requires training.updates_per_epoch=80")
        if not synthetic and "eval_every_epochs" not in training_cfg:
            raise ValueError("real PG S3 requires explicit training.eval_every_epochs")
        eval_every_epochs = _integer(training_cfg.get("eval_every_epochs", 1), "training.eval_every_epochs")
        eval_every = updates_per_epoch * eval_every_epochs
    elif synthetic:
        # Tiny CPU fixtures predate the materialized production training block;
        # they may explicitly choose an eval interval for a short replay.
        eval_every = _integer(options.get("eval_every_updates", 80), "codebook eval_every_updates")
        updates_per_epoch = eval_every
    else:
        raise ValueError("real PG S3 requires training.updates_per_epoch")
    truncated_run = max_updates is not None and int(max_updates) < configured_updates
    if max_updates is not None:
        updates = min(configured_updates, int(max_updates))
    if updates <= 0:
        raise ValueError("S3 updates must be positive")
    start = 0
    resume_payload: Mapping[str, Any] | None = None
    epoch_loss_sum = {"total": 0.0, "l": 0.0, "u": 0.0}
    epoch_loss_count = 0
    last_assignments: Tensor | None = None
    last_valid_mask: Tensor | None = None
    last_phase: tuple[tuple[int, int, int], tuple[float, float, float]] | None = None
    if resume and checkpoint.is_file():
        state = torch.load(checkpoint, map_location=target_device, weights_only=False)
        if state.get("schema") != CODEBOOK_VERSION:
            raise ValueError("S3 checkpoint schema mismatch")
        resume_payload = state
        codebook.load_state_dict(state["codebook"])
        optimizer.load_state_dict(state["optimizer"])
        start = int(state.get("update", 0))
        if sampler is not None and state.get("sampler") is not None:
            sampler.load_state_dict(state["sampler"])
        epoch_loss_sum = {key: float(state.get("epoch_loss_sum", {}).get(key, 0.0)) for key in epoch_loss_sum}
        epoch_loss_count = int(state.get("epoch_loss_count", 0))
        if isinstance(state.get("last_assignments"), Tensor):
            last_assignments = state["last_assignments"].to(target_device)
        if isinstance(state.get("last_valid_mask"), Tensor):
            last_valid_mask = state["last_valid_mask"].to(target_device)
    log_path = root / "diag.jsonl"
    run_log = root / "run.log"
    metrics_path = root / "qT_metrics.jsonl"
    epoch_logger = _open_logger_for_resume(root, resume_payload) if resume else __import__("self_audit.candidate_pg.logging", fromlist=["EpochLogger"]).EpochLogger(root, stage="S3", arm="S3")
    total_epochs = max(1, (configured_updates + eval_every - 1) // eval_every)
    print("[S3] start", flush=True)
    epoch_started = time.perf_counter()
    for update in range(start, updates):
        step_started = time.perf_counter()
        if sampler is not None:
            l_key, u_key, u2_key = sampler.next_step()
        else:
            l_key = labeled_keys[update % len(labeled_keys)]
            u_key = unlabeled_keys[update % len(unlabeled_keys)]
            anchor_patient = _key_patient(u_key)
            partners = [key for key in unlabeled_keys if _key_patient(key) != anchor_patient]
            if not partners:
                raise ValueError("synthetic S3 U fixture must provide a different-patient partner")
            u2_key = partners[update % len(partners)]
        l_frame = _load_s3_frame(train_data, l_key, labeled=True)
        u_frame = _load_s3_frame(train_data, u_key, labeled=False)
        u2_frame = _load_s3_frame(train_data, u2_key, labeled=False)
        l_image, l_target, l_geometry = _s3_frame_tensors(l_frame, target_device, labeled=True)
        u_image, _, u_geometry = _s3_frame_tensors(u_frame, target_device, labeled=False)
        u2_image, _, u2_geometry = _s3_frame_tensors(u2_frame, target_device, labeled=False)
        l_packet = _teacher_packet(teacher, l_image, l_geometry)
        u_packet = _teacher_packet(teacher, u_image, u_geometry)
        u2_packet = _teacher_packet(teacher, u2_image, u2_geometry)
        l_feature, l_phase, l_valid = _aligned_level3(l_packet["features"][3], l_geometry, teacher=teacher, student=student)
        u_feature, u_phase, u_valid_feature = _aligned_level3(u_packet["features"][3], u_geometry, teacher=teacher, student=student)
        u2_feature, u2_phase, u2_valid_feature = _aligned_level3(u2_packet["features"][3], u2_geometry, teacher=teacher, student=student)
        l_out = codebook(l_feature, valid_mask=l_valid)
        u_out = codebook(u_feature, valid_mask=u_valid_feature)
        u2_out = codebook(u2_feature, valid_mask=u2_valid_feature)
        l_probs, l_valid = _pooled_gt(l_target, tuple(l_image.shape[2:]), l_phase)
        from ..candidate_q3d.alignment import pool_teacher_targets

        u_probs, _u_classes, u_confidence = pool_teacher_targets(u_packet["region_logits"], tuple(u_image.shape[2:]), u_phase)
        u_valid = u_valid_feature & (u_confidence > 0)
        u2_probs, _u2_classes, u2_confidence = pool_teacher_targets(u2_packet["region_logits"], tuple(u2_image.shape[2:]), u2_phase)
        u2_valid = u2_valid_feature & (u2_confidence > 0)
        optimizer.zero_grad(set_to_none=True)
        l_terms = codebook.loss(l_out, labels=l_probs, valid_mask=l_valid, labeled=True, return_components=True)
        u_terms = codebook.loss(u_out, pseudo_labels=u_probs, pseudo_confidence=u_confidence, valid_mask=u_valid, labeled=False, return_components=True)
        u2_terms = codebook.loss(u2_out, pseudo_labels=u2_probs, pseudo_confidence=u2_confidence, valid_mask=u2_valid, labeled=False, return_components=True)
        # The declared S3 batch is one labeled frame plus two U frames.  The
        # U contribution is explicitly the equal frame average, never a
        # hidden per-voxel or patient reweighting: L + mean(U1, U2).
        u_frame_average = 0.5 * (u_terms["loss"] + u2_terms["loss"])
        total = l_terms["loss"] + u_frame_average
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite S3 codebook loss at update {update}")
        total.backward()
        optimizer.step()
        usage_update = codebook.update_usage_batch(
            [
                (l_out["assignments"].detach(), l_out["valid_mask"], l_out["z"].detach()),
                (u_out["assignments"].detach(), u_out["valid_mask"], u_out["z"].detach()),
                (u2_out["assignments"].detach(), u2_out["valid_mask"], u2_out["z"].detach()),
            ],
            step=update + 1,
        )
        l_usage_diag = usage_metrics(
            l_out["assignments"].detach(), codebook.codebook_size,
            l_out["valid_mask"], l_probs.argmax(dim=1),
        )
        last_assignments = l_out["assignments"].detach()
        last_valid_mask = l_out["valid_mask"].detach()
        last_phase = l_phase
        epoch_loss_sum["total"] += float(total.detach())
        epoch_loss_sum["l"] += float(l_terms["loss"].detach())
        epoch_loss_sum["u"] += float(u_frame_average.detach())
        epoch_loss_count += 1
        update_no = update + 1
        if update_no % 50 == 0 or update_no == updates:
            memory_bytes = int(torch.cuda.memory_allocated(target_device)) if target_device.type == "cuda" else 0
            record = {
                "stage": "S3", "update": update_no,
                "loss_total": float(total.detach()), "loss_l": float(l_terms["loss"].detach()),
                "loss_u": float(u_frame_average.detach()),
                "u_reduction": "mean(U1,U2)",
                "loss_l_vq": float(l_terms["vq"].detach()), "loss_l_commitment": float(l_terms["commitment"].detach()),
                "loss_l_qtask": float(l_terms["qtask"].detach()), "loss_l_entropy_penalty": float(l_terms["entropy_penalty"].detach()),
                "loss_u_vq": float((0.5 * (u_terms["vq"] + u2_terms["vq"])).detach()),
                "loss_u_commitment": float((0.5 * (u_terms["commitment"] + u2_terms["commitment"])).detach()),
                "loss_u_qtask": float((0.5 * (u_terms["qtask"] + u2_terms["qtask"])).detach()),
                "loss_u_entropy_penalty": float((0.5 * (u_terms["entropy_penalty"] + u2_terms["entropy_penalty"])).detach()),
                "perplexity": float(l_usage_diag["perplexity"]),
                "dead_code_count": int(l_usage_diag["dead_code_count"]),
                "codes_per_class": l_usage_diag["codes_per_class"],
                "window_dead_code_count": int(usage_update.get("window_dead_code_count", l_usage_diag["dead_code_count"])),
                "restarted_codes": usage_update.get("restarted_codes", []),
                "memory_bytes": memory_bytes, "seconds": float(time.perf_counter() - step_started),
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            with run_log.open("a", encoding="utf-8") as stream:
                stream.write("DEBUG " + json.dumps(record, sort_keys=True) + "\n")
        is_epoch_boundary = update_no % eval_every == 0 or (update_no == updates and not truncated_run)
        if is_epoch_boundary:
            epoch_loss_value = epoch_loss_sum["total"] / max(1, epoch_loss_count)
            split_metrics = {
                "train_labeled": _qval_metrics(codebook, train_data, teacher, student, target_device),
                "VAL18": _qval_metrics(codebook, val_data, teacher, student, target_device) if val_data is not None else None,
                "TEST50": _qval_metrics(codebook, test_data, teacher, student, target_device) if test_data is not None else None,
            }
            epoch = (update_no + eval_every - 1) // eval_every
            split_aliases = {"train_labeled": "train", "VAL18": "val", "TEST50": "test"}
            patient_rows: list[dict[str, Any]] = []
            for split_name, metric in split_metrics.items():
                if metric is None:
                    continue
                split = split_aliases[split_name]
                train_loss = epoch_loss_value if split_name == "train_labeled" else None
                epoch_logger.log_epoch(_metric_epoch_row(metric, epoch=epoch, update=update_no, split=split, loss_total=train_loss))
                for patient_id, patient_metric in (metric.get("per_patient") or {}).items():
                    per = patient_metric.get("per_class", {}) if isinstance(patient_metric, Mapping) else {}
                    patient_rows.append({
                        "patient_id": str(patient_id), "split": split, "path": "qT",
                        "fg_dice": patient_metric.get("fg_dice"), "fg_hd95": patient_metric.get("fg_hd95"),
                        "fg_assd": patient_metric.get("fg_assd"), "rv_dice": (per.get("RV") or {}).get("dice"),
                        "myo_dice": (per.get("MYO") or {}).get("dice"), "lv_dice": (per.get("LV") or {}).get("dice"),
                        "rv_hd95": (per.get("RV") or {}).get("hd95_mm"), "myo_hd95": (per.get("MYO") or {}).get("hd95_mm"),
                        "lv_hd95": (per.get("LV") or {}).get("hd95_mm"), "rv_assd": (per.get("RV") or {}).get("assd_mm"),
                        "myo_assd": (per.get("MYO") or {}).get("assd_mm"), "lv_assd": (per.get("LV") or {}).get("assd_mm"),
                        "valid_voxels": patient_metric.get("valid_voxels"), "valid_mask_present": True,
                    })
            if patient_rows:
                epoch_logger.log_patient_rows(epoch, patient_rows)
            val_metric = split_metrics.get("VAL18")
            if isinstance(val_metric, Mapping) and val_metric.get("fg_dice") is not None:
                # Selection is VAL-only and uses the same fixed qT path as
                # the epoch rows; TEST metrics cannot influence this marker.
                try:
                    epoch_logger.select_best(split="val", path="qT", metric="fg_dice")
                except ValueError:
                    pass
            # S3 emits exactly one compact progress row per completed epoch;
            # detailed diagnostics remain file-only in diag.jsonl/run.log.
            from .logging import epoch_console_line
            train_metric = split_metrics.get("train_labeled") or {}
            val_metric = split_metrics.get("VAL18") or {}
            test_metric = split_metrics.get("TEST50") or {}
            best_row = epoch_logger.best_val
            best_fg = best_row.get("fg_dice") if isinstance(best_row, Mapping) else None
            best_epoch = best_row.get("epoch") if isinstance(best_row, Mapping) else epoch
            epoch_seconds = time.perf_counter() - epoch_started
            print(epoch_console_line(
                "S3", epoch, total_epochs, epoch_loss_value,
                train_metric.get("fg_dice"), val_metric.get("fg_dice"),
                best_fg, best_epoch, test_metric.get("fg_dice"),
                epoch_seconds,
            ), flush=True)
            epoch_started = time.perf_counter()
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"stage": "S3", "update": update_no, "epoch": epoch, "splits": split_metrics}, sort_keys=True, allow_nan=False) + "\n")
            # Boundary rows are now durable in the append receipts.  Reset the
            # partial epoch accumulator before publishing the checkpoint so a
            # resume after commit starts the next epoch at zero.
            epoch_loss_sum = {"total": 0.0, "l": 0.0, "u": 0.0}
            epoch_loss_count = 0
            if update_no % 50 == 0 or is_epoch_boundary:
                logger_state = _logger_checkpoint_state(epoch_logger, epoch=update_no // eval_every, update=update_no)
            checkpoint_payload = {
                "schema": CODEBOOK_VERSION, "update": update_no,
                "codebook": codebook.state_dict(), "optimizer": optimizer.state_dict(),
                "sampler": sampler.state_dict() if sampler is not None else None,
                    "last_assignments": last_assignments, "last_valid_mask": last_valid_mask,
                "epoch_loss_sum": epoch_loss_sum, "epoch_loss_count": epoch_loss_count,
                    "epoch_logger": logger_state, "metrics_receipt": logger_state,
                "batch_reduction": "L + mean(U1,U2)",
            }
            _atomic_torch_save(checkpoint_payload, checkpoint)
            if is_epoch_boundary:
                # Crash-safe boundary: epoch rows -> checkpoint -> commit.
                epoch_logger.commit_epoch(epoch=epoch, update=update_no)
    codebook.freeze()
    _atomic_torch_save({"schema": CODEBOOK_VERSION, "codebook": codebook.state_dict(), "update": updates}, root / "codebook.pt")
    if last_assignments is None or last_valid_mask is None or last_phase is None:
        # A checkpoint from the pre-boundary state contract may contain no
        # cached labeled packet (for example, resume after a one-update smoke).
        # Recompute one deterministic L packet solely for the usage receipt.
        fallback_key = labeled_keys[0]
        fallback_frame = _load_s3_frame(train_data, fallback_key, labeled=True)
        fallback_image, _, fallback_geometry = _s3_frame_tensors(fallback_frame, target_device, labeled=True)
        fallback_packet = _teacher_packet(teacher, fallback_image, fallback_geometry)
        fallback_feature, _fallback_phase, fallback_valid = _aligned_level3(
            fallback_packet["features"][3], fallback_geometry, teacher=teacher, student=student
        )
        fallback_output = codebook(fallback_feature, valid_mask=fallback_valid)
        last_assignments = fallback_output["assignments"].detach()
        last_valid_mask = fallback_output["valid_mask"]
        last_phase = _fallback_phase
    metrics = codebook.metrics(last_assignments, last_valid_mask)
    if isinstance(metrics.get("usage"), Tensor):
        metrics["usage"] = metrics["usage"].detach().cpu().tolist()
    val_metrics = _qval_metrics(codebook, val_data, teacher, student, target_device) if val_data is not None else None
    test_metrics = _qval_metrics(codebook, test_data, teacher, student, target_device) if test_data is not None else None
    runtime_cfg = config.get("runtime") if isinstance(config.get("runtime"), Mapping) else {}
    receipt = {
        "schema": CODEBOOK_VERSION, "status": "COMPLETED", "seed": seed_value,
        "runtime": {
            "device": str(target_device), "synthetic": bool(runtime_cfg.get("synthetic", False)),
            "deterministic": bool(runtime_cfg.get("deterministic", False)),
            "tf32": bool(runtime_cfg.get("tf32", False)),
            **runtime_flags,
        },
        "updates": updates, "updates_per_epoch": int(updates_per_epoch),
        "batch_reduction": "L + mean(U1,U2)",
        "codebook_initialization": initialization_meta,
        "data_identity": json.loads(json.dumps(config.get("data", {}), default=str)),
        "source_identity": json.loads(json.dumps({
            "teacher": config.get("teacher", {}), "codebook": config.get("codebook", {}),
        }, default=str)),
        "checkpoint": str(checkpoint.resolve()), "artifact": str((root / "codebook.pt").resolve()),
        "epochs_jsonl": str(epoch_logger.jsonl_path.resolve()),
    }
    receipt_path = root / "S3_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("[S3] end", flush=True)
    return {
        "status": "COMPLETED",
        "updates": updates,
        "codebook_frozen": True,
        "artifact": str((root / "codebook.pt").resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "diag": str(log_path.resolve()),
        "qT_metrics": str(metrics_path.resolve()),
        "epochs_csv": str(epoch_logger.csv_path.resolve()),
        "epochs_jsonl": str(epoch_logger.jsonl_path.resolve()),
        "metrics": metrics,
        "qT_VAL": val_metrics,
        "qT_TEST": test_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "batch_reduction": "L + mean(U1,U2)",
        "codebook_initialization": initialization_meta,
        "updates_per_epoch": int(updates_per_epoch),
        "receipt": str(receipt_path.resolve()),
        "artifact_metadata": codebook.artifact_metadata(),
        "aligned_feature_phase": {
            "teacher_tap": "level3",
                "student_phase": [list(last_phase[0]), list(last_phase[1])],
        },
    }


__all__ = [
    "CODEBOOK_VERSION",
    "TeacherCodebook",
    "StudentCodeBranch",
    "assign_kl",
    "cbkd_loss",
    "commitment_loss",
    "rho",
    "soft_cross_entropy",
    "soft_dice_loss",
    "soft_task_loss",
    "usage_metrics",
    "vq_ste",
    "run_codebook",
]
