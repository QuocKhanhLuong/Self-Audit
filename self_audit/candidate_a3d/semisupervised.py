"""Small, explicit semi-supervised helpers for the native A-3D track.

The A-3D pipeline has two different mentor roles.  The dual-branch teacher
proposes measurements for labelled examples; an EMA copy is used here only as
a temporal consistency target for image-only TRAIN examples.  The EMA target
is always detached, is never given masks, and is updated *after* the online
optimizer step.  This module deliberately does not create pseudo masks,
signed-distance targets, or K1 utility targets.

Unlabelled consistency is applied to the dense predictions produced by the
online path.  The caller remains responsible for obtaining those predictions
through the code-only rollout, so this module cannot introduce a feature skip
or a raw-image decoder bypass.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

_OUTPUT_KEYS = ("region_logits", "boundary_logits", "sdf")


def _semi_value(config: Mapping[str, Any] | None, name: str, default: Any) -> Any:
    """Read a semi-supervised option from either a nested or flat mapping."""

    if config is None:
        return default
    if name in config:
        return config[name]
    nested = config.get("semi_supervised")
    if isinstance(nested, Mapping) and name in nested:
        return nested[name]
    return default


def _float_option(config: Mapping[str, Any] | None, name: str, default: float) -> float:
    value = float(_semi_value(config, name, default))
    if not math.isfinite(value):
        raise ValueError(f"semi-supervised {name} must be finite")
    return value


def _check_probability_options(
    config: Mapping[str, Any] | None,
    *,
    threshold: float | None = None,
    background_weight: float | None = None,
) -> tuple[float, float]:
    threshold = _float_option(config, "confidence_threshold", 0.7) if threshold is None else float(threshold)
    background_weight = (
        _float_option(config, "background_weight", 0.1)
        if background_weight is None
        else float(background_weight)
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0,1]")
    if not 0.0 < background_weight <= 1.0:
        raise ValueError("background_weight must be in (0,1]")
    return threshold, background_weight


def make_ema(model: nn.Module, decay: float = 0.99) -> nn.Module:
    """Create a frozen, evaluation-mode EMA copy of ``model``.

    ``copy.deepcopy`` keeps the exact module topology and buffers, which makes
    the result directly serializable with ``state_dict``.  It also ensures that
    the consistency target cannot accidentally share parameters with the online
    network.
    """

    decay = float(decay)
    if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be finite and in [0,1)")
    target = copy.deepcopy(model)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    # Keep the value available to checkpoint wrappers without making it a
    # trainable tensor.  The parent trainer may overwrite this after loading.
    target._semi_ema_decay = decay
    return target


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float | None = None) -> None:
    """Update EMA parameters after an online optimizer step.

    Floating-point parameters are exponentially averaged; buffers are copied
    from the online model.  Copying buffers is the safe choice for this model
    family (GroupNorm has no running statistics) and avoids stale geometry or
    integer buffers.  The function never creates gradients on either side.
    """

    if type(ema_model) is not type(model):
        raise TypeError("EMA and online models must have the same module type")
    value = getattr(ema_model, "_semi_ema_decay", 0.99) if decay is None else decay
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("EMA decay must be finite and in [0,1)")
    online_params = dict(model.named_parameters())
    ema_params = dict(ema_model.named_parameters())
    if set(online_params) != set(ema_params):
        raise ValueError("EMA and online parameter structures do not match")
    for name, target in ema_params.items():
        source = online_params[name]
        target.mul_(value).add_(source.detach(), alpha=1.0 - value)
        target.requires_grad_(False)
    online_buffers = dict(model.named_buffers())
    ema_buffers = dict(ema_model.named_buffers())
    if set(online_buffers) != set(ema_buffers):
        raise ValueError("EMA and online buffer structures do not match")
    for name, target in ema_buffers.items():
        target.copy_(online_buffers[name].detach())
    ema_model.eval()
    ema_model._semi_ema_decay = value


@torch.no_grad()
def detached_outputs(outputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Detach the public teacher/EMA outputs before using them as targets."""

    if not isinstance(outputs, Mapping) or set(outputs) != set(_OUTPUT_KEYS):
        raise ValueError(f"outputs must contain exactly {list(_OUTPUT_KEYS)}")
    result: dict[str, Tensor] = {}
    for key in _OUTPUT_KEYS:
        value = outputs[key]
        if not isinstance(value, Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"outputs[{key!r}] must be a finite tensor")
        result[key] = value.detach()
    return result


def strong_intensity(
    image: Tensor,
    config: Mapping[str, Any] | None = None,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply intensity-only perturbation while preserving native geometry.

    A per-volume gamma and additive Gaussian noise are sampled from the given
    generator.  Sampling on CPU and moving the random factors to the image
    device keeps a CPU-seeded rollout deterministic on CUDA as well.  No
    spatial operation, crop, resampling, or affine change is performed.
    """

    if not isinstance(image, Tensor) or image.ndim < 3:
        raise ValueError("semi-supervised image must be a tensor with batch and spatial dimensions")
    if not torch.isfinite(image).all() or bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError("semi-supervised image must be finite and normalized to [0,1]")
    gamma_range = _semi_value(config, "intensity_gamma", (0.7, 1.5))
    if not isinstance(gamma_range, (list, tuple)) or len(gamma_range) != 2:
        raise ValueError("intensity_gamma must contain [low, high]")
    gamma_low, gamma_high = (float(item) for item in gamma_range)
    noise_std = _float_option(config, "intensity_noise_std", 0.03)
    if not math.isfinite(gamma_low) or not math.isfinite(gamma_high) or not 0.0 < gamma_low <= gamma_high:
        raise ValueError("intensity_gamma must be finite positive [low, high]")
    if noise_std < 0.0:
        raise ValueError("intensity_noise_std must be non-negative")
    batch = int(image.shape[0])
    # Per-example factors, broadcast over C and native D/H/W.
    random_device = torch.device("cpu")
    if generator is not None:
        # ``torch.Generator.device`` is available on supported PyTorch builds;
        # retain CPU as the portable default for callers that pass no generator.
        random_device = torch.device(getattr(generator, "device", "cpu"))
    uniform = torch.rand((batch,) + (1,) * (image.ndim - 1), generator=generator, device=random_device)
    gamma = (gamma_low + (gamma_high - gamma_low) * uniform).to(device=image.device, dtype=image.dtype)
    perturbed = image.clamp(0.0, 1.0).pow(gamma)
    if noise_std:
        # Noise is voxel-wise.  A ``[B,1,1,1,1]`` draw would be one global
        # brightness offset per volume and would not provide a spatial
        # consistency perturbation.
        noise = torch.randn(image.shape, generator=generator, device=random_device, dtype=torch.float32)
        perturbed = perturbed + noise.to(device=image.device, dtype=image.dtype) * noise_std
    return perturbed.clamp(0.0, 1.0)


def ramp_weight(epoch: int, config: Mapping[str, Any] | None = None) -> float:
    """Return a deterministic linear consistency ramp for one-based epochs."""

    epoch = int(epoch)
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    ramp_epochs = int(_semi_value(config, "rampup_epochs", 10))
    if ramp_epochs < 0:
        raise ValueError("rampup_epochs must be non-negative")
    if ramp_epochs == 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch) / float(ramp_epochs)))


def _validate_logits(student_logits: Tensor, target_logits: Tensor) -> None:
    if not isinstance(student_logits, Tensor) or not isinstance(target_logits, Tensor):
        raise TypeError("consistency inputs must be tensors")
    if student_logits.ndim < 3 or student_logits.shape != target_logits.shape:
        raise ValueError("consistency logits must have matching [B,C,...] shapes")
    if student_logits.shape[1] < 2:
        raise ValueError("consistency logits need background plus foreground classes")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(target_logits).all():
        raise ValueError("consistency logits must be finite")


def consistency_loss(
    student_logits: Tensor,
    target_logits: Tensor,
    *,
    threshold: float = 0.7,
    background_weight: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    """Confidence-gated soft segmentation consistency.

    The target is a detached soft distribution.  Reliable foreground voxels
    receive unit weight; reliable background receives ``background_weight`` so
    a model predicting background everywhere cannot dominate the unsupervised
    objective.  If no target voxel clears the threshold, a zero loss retaining
    the student autograd graph is returned and coverage is reported as zero.
    """

    _validate_logits(student_logits, target_logits)
    threshold, background_weight = _check_probability_options(
        None, threshold=threshold, background_weight=background_weight
    )
    target_prob = target_logits.detach().softmax(dim=1)
    student_prob = student_logits.softmax(dim=1)
    confidence, target_class = target_prob.max(dim=1)
    selected = confidence >= threshold
    foreground = selected & target_class.ne(0)
    background = selected & target_class.eq(0)
    # Aggregate each predicted class separately.  A denominator made from the
    # selected voxel weights would cancel ``background_weight`` on all-
    # background images and let the largest support dominate.  The fixed
    # denominator keeps the configured background cap effective and makes the
    # loss comparable as confidence coverage changes.
    class_terms: list[Tensor] = []
    class_coverage: dict[str, float] = {}
    per_class_error = (student_prob - target_prob).pow(2).mean(dim=1)
    for class_id in range(student_logits.shape[1]):
        support = selected & target_class.eq(class_id)
        class_coverage[str(class_id)] = float(support.float().mean().detach())
        if bool(support.any()):
            weight = background_weight if class_id == 0 else 1.0
            class_terms.append(per_class_error[support].mean() * weight)
    # One unit for each foreground class and the configured capped background
    # term.  Empty supports contribute zero to the numerator but remain in the
    # fixed normalizer, avoiding a hidden all-background rescaling.
    normalizer = float((student_logits.shape[1] - 1) + background_weight)
    if class_terms:
        loss = torch.stack(class_terms).sum() / normalizer
    else:
        loss = student_logits.sum() * 0.0
    total = float(selected.numel())
    stats = {
        "selected_fraction": float(selected.float().mean().detach()),
        "foreground_coverage": float(foreground.float().mean().detach()),
        "background_coverage": float(background.float().mean().detach()),
        "selected_foreground_fraction": float(foreground.sum().detach() / selected.sum().clamp_min(1)),
        "class_coverage": class_coverage,
        "loss": float(loss.detach()),
        "selected_voxels": float(selected.sum().detach()),
        "total_voxels": total,
    }
    return loss, stats


def _sdf_consistency(
    student_sdf: Tensor,
    target_sdf: Tensor,
    target_region_logits: Tensor,
    *,
    threshold: float,
    background_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    if student_sdf.shape != target_sdf.shape or student_sdf.shape != target_region_logits.shape:
        raise ValueError("student/target SDF and region logits must have matching [B,C,...] shapes")
    if not torch.isfinite(student_sdf).all() or not torch.isfinite(target_sdf).all():
        raise ValueError("SDF consistency tensors must be finite")
    probs = target_region_logits.detach().softmax(dim=1)
    confidence, target_class = probs.max(dim=1)
    selected = confidence >= threshold
    error = (student_sdf - target_sdf.detach()).abs()
    class_losses: list[Tensor] = []
    class_coverages: dict[str, float] = {}
    for class_id in range(student_sdf.shape[1]):
        support = selected & target_class.eq(class_id)
        class_coverages[str(class_id)] = float(support.float().mean().detach())
        if bool(support.any()):
            value = error[:, class_id][support].mean()
            if class_id == 0:
                value = value * background_weight
            class_losses.append(value)
    if class_losses:
        loss = torch.stack(class_losses).mean()
    else:
        loss = student_sdf.sum() * 0.0
    foreground = selected & target_class.ne(0)
    return loss, {
        "loss": float(loss.detach()),
        "selected_fraction": float(selected.float().mean().detach()),
        "foreground_coverage": float(foreground.float().mean().detach()),
        "class_coverage": class_coverages,
        "selected_voxels": float(selected.sum().detach()),
    }


def teacher_consistency_loss(
    student_outputs: Mapping[str, Tensor],
    target_outputs: Mapping[str, Tensor],
    config: Mapping[str, Any] | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Consistency for region, boundary, and soft SDF outputs on image-only U."""

    if not isinstance(student_outputs, Mapping) or not isinstance(target_outputs, Mapping):
        raise TypeError("teacher consistency outputs must be mappings")
    if set(student_outputs) != set(_OUTPUT_KEYS) or set(target_outputs) != set(_OUTPUT_KEYS):
        raise ValueError(f"outputs must contain exactly {list(_OUTPUT_KEYS)}")
    threshold, background_weight = _check_probability_options(config)
    region, region_stats = consistency_loss(
        student_outputs["region_logits"], target_outputs["region_logits"],
        threshold=threshold, background_weight=background_weight,
    )
    boundary, boundary_stats = consistency_loss(
        student_outputs["boundary_logits"], target_outputs["boundary_logits"],
        threshold=threshold, background_weight=background_weight,
    )
    sdf, sdf_stats = _sdf_consistency(
        student_outputs["sdf"], target_outputs["sdf"], target_outputs["boundary_logits"],
        threshold=threshold, background_weight=background_weight,
    )
    total = (
        region * _float_option(config, "teacher_weight", 1.0)
        + boundary * _float_option(config, "teacher_weight", 1.0)
        + sdf * _float_option(config, "teacher_weight", 1.0)
    )
    return total, {"region": region_stats, "boundary": boundary_stats, "sdf": sdf_stats,
                   "loss": float(total.detach())}


def student_unlabeled_loss(
    student_logits: Tensor,
    ema_logits: Tensor,
    config: Mapping[str, Any] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Code-output consistency used for U; no labels or K1 utilities are accepted."""

    threshold, background_weight = _check_probability_options(config)
    loss, stats = consistency_loss(
        student_logits, ema_logits,
        threshold=threshold, background_weight=background_weight,
    )
    weight = _float_option(config, "student_weight", 1.0)
    return loss * weight, stats | {"loss_weight": weight, "weighted_loss": float((loss * weight).detach())}


def ema_forward(
    ema_model: nn.Module,
    forward: Callable[..., Mapping[str, Tensor]],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Tensor]:
    """Run an EMA target forward with gradients disabled and detached outputs."""

    was_training = ema_model.training
    ema_model.eval()
    try:
        with torch.no_grad():
            return detached_outputs(forward(*args, **kwargs))
    finally:
        ema_model.train(was_training)


__all__ = [
    "consistency_loss",
    "detached_outputs",
    "ema_forward",
    "make_ema",
    "ramp_weight",
    "strong_intensity",
    "student_unlabeled_loss",
    "teacher_consistency_loss",
    "update_ema",
]
