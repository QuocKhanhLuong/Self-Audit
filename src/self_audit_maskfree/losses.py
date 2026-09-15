"""Image-only producer objectives and the shared student objective (W4).

Scientific role
---------------
Two entry points, deliberately separated by what they are allowed to see:

* :func:`producer_loss` — self-supervision for :class:`~.models.Producer` from
  fitting observations alone. Three arms, all restricted to ``O_fit`` pixels:

  1. **Bounded dense contrastive correspondence.** A dihedral transform (flip
     and/or 90-degree rotation) is applied to the context; because the transform
     is known and exactly invertible, the same anatomical pixel in the two views
     is a *known* positive pair, with no learned matcher and no mask. At most
     :data:`MAX_SAMPLES_PER_IMAGE` locations are sampled per image, so the
     similarity matrix is at most ``128x128`` per image; a dense ``HW x HW``
     similarity is never allocated.
  2. **Local masked reconstruction.** A few small blocks of the context are
     physically zeroed and the reconstruction head must predict the centre
     channel at exactly those blocks, scored only where the fit support is true.
  3. **Geometric equivariance.** Features computed on the transformed context
     and mapped back through the inverse transform must agree with features
     computed on the untransformed context, again only on fit pixels.

* :func:`student_loss` — validity-weighted soft cross entropy against detached
  generated pseudo-labels. The identical function is used by both student arms.

Supervision firewall
--------------------
No manual mask, no reference intensity and no student output can reach
:func:`producer_loss`: it takes a model, a fitting context and a fit support
mask, nothing else. It zeroes all three context channels outside that support
*before the first forward pass*, so the immunity is enforced rather than
assumed, and every target it constructs is then read from the fit support of the
context itself. :func:`student_loss` detaches both the probabilities and the
validity map before use, so no gradient can flow from a student back into the
producer, the hypothesis bank, or the other arm.

:func:`student_loss` validates its pseudo-labels instead of repairing them. Out
of range or non-finite probabilities and validity weights are rejected, and on
valid support the class axis must already be a unit-sum distribution. Silently
renormalizing an all-zero class vector would manufacture a target no hypothesis
produced and would let a degenerate bank contribute an unnoticed zero loss, so
that path raises.

Stated negative-sampling rule and its limitation
------------------------------------------------
Negatives for the contrastive arm are other sampled locations **from the same
image of the same patient**, kept at least :data:`MIN_NEGATIVE_SEPARATION`
pixels away from the anchor in native pixel coordinates. This avoids the common
"every other patient is a true negative" assumption, which is false for cardiac
anatomy.

It is still wrong in the other direction, and this is a real limitation, not a
detail: two spatially separated patches of the same slice are frequently the
*same* tissue class — background on opposite sides of the field of view, or the
septal and lateral myocardial walls. Those pairs are **false negatives**. The
objective therefore actively pushes apart some pixels that anatomy says belong
together, and the learned features should be read as "locally discriminative",
not as a class-pure embedding. The separation threshold trades false negatives
(too small) against trivial negatives (too large); it is a fixed provisional
constant, not a tuned optimum, and it was not selected using any mask.

Batch size, accumulation and the contrastive arm
------------------------------------------------
Negatives are drawn **within one image**, never across the batch. Consequently
the *negative set* is invariant to physical batch size: microbatch 4 with
accumulation 2 draws exactly the same per-image negatives as physical batch 8,
unlike a cross-batch InfoNCE where shrinking the physical batch shrinks the
negative pool and changes the objective.

What is still not identical under accumulation: this function returns the mean
over the images in the tensor it was given. Summing two microbatch means and
dividing by two equals the batch-8 mean only when both microbatches contribute
the same number of contributing images. Images with empty fit support, or with
fewer than two sampled locations, are skipped and counted in
``metrics['contrastive_images']``, so unequal skipping makes accumulated
gradients differ slightly from the single-batch gradient. The trainer must
report physical and effective batch separately and must not claim numerical
identity; the honest statement is that the negative distribution is unchanged
and only the weighting across skipped images can differ.

Numerical guards
----------------
Non-finite inputs are rejected rather than silently propagated. Masked
similarity entries use a large finite negative constant instead of ``-inf`` so
that ``softmax`` cannot produce ``nan`` gradients. Every denominator is a
counted support that is clamped away from zero, and each arm with no usable
support contributes an exact differentiable zero instead of ``0/0``. Clamping
appears only in denominators and in the log of the reported target entropy;
it is never used to coerce an invalid target into a valid one.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import VERSION
from .models import CONTEXT_CHANNELS, NUM_CLASSES

#: Upper bound on sampled locations per image for the contrastive arm. Keeps the
#: similarity matrix at most ``128x128`` per image instead of ``HW x HW``.
MAX_SAMPLES_PER_IMAGE = 128
#: InfoNCE temperature. Fixed provisional constant.
CONTRASTIVE_TEMPERATURE = 0.07
#: Minimum Euclidean pixel distance between an anchor and an admissible negative.
MIN_NEGATIVE_SEPARATION = 8.0
#: Finite stand-in for ``-inf`` in masked similarity logits.
MASKED_LOGIT = -1.0e4
#: Number of masked blocks per image in the reconstruction arm.
RECONSTRUCTION_BLOCKS = 4
#: Loss weights. Fixed provisional constants, not tuned against any mask.
WEIGHT_CONTRASTIVE = 1.0
WEIGHT_RECONSTRUCTION = 1.0
WEIGHT_EQUIVARIANCE = 1.0

EPS = 1.0e-6
#: Tolerance on ``probabilities.sum(dim=1) == 1`` over valid pseudo-label pixels.
#: Wide enough for float32 accumulation over four classes, far too tight to let
#: an unnormalized or all-zero class vector through.
PROBABILITY_SUM_TOLERANCE = 1.0e-3

#: The eight dihedral operations: ``(number of 90-degree rotations, horizontal flip)``.
DIHEDRAL_OPS: tuple[tuple[int, bool], ...] = tuple(
    (k, flip) for k in (0, 1, 2, 3) for flip in (False, True)
)


def _randperm(n: int, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    """Deterministic permutation that tolerates a generator on another device."""
    if generator is None:
        return torch.randperm(n, device=device)
    if generator.device.type == device.type:
        return torch.randperm(n, generator=generator, device=device)
    return torch.randperm(n, generator=generator, device=generator.device).to(device)


def _randint(high: int, shape: tuple[int, ...], device: torch.device,
             generator: torch.Generator | None) -> torch.Tensor:
    if high <= 1:
        return torch.zeros(shape, dtype=torch.long, device=device)
    if generator is None:
        return torch.randint(0, high, shape, device=device)
    if generator.device.type == device.type:
        return torch.randint(0, high, shape, generator=generator, device=device)
    return torch.randint(0, high, shape, generator=generator, device=generator.device).to(device)


def _apply_op(x: torch.Tensor, op: tuple[int, bool]) -> torch.Tensor:
    """Apply a dihedral operation to a ``[...,H,W]`` tensor."""
    rotations, flip = op
    out = torch.rot90(x, rotations, dims=(-2, -1)) if rotations else x
    return torch.flip(out, dims=(-1,)) if flip else out


def _invert_op(x: torch.Tensor, op: tuple[int, bool]) -> torch.Tensor:
    """Exact inverse of :func:`_apply_op`, so correspondences stay pixel-exact."""
    rotations, flip = op
    out = torch.flip(x, dims=(-1,)) if flip else x
    return torch.rot90(out, -rotations, dims=(-2, -1)) if rotations else out


def _sample_op(generator: torch.Generator | None, device: torch.device) -> tuple[int, bool]:
    index = int(_randint(len(DIHEDRAL_OPS), (1,), device, generator).item())
    return DIHEDRAL_OPS[index]


def _sample_support_locations(support: torch.Tensor, generator: torch.Generator | None,
                              max_samples: int = MAX_SAMPLES_PER_IMAGE
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample at most ``max_samples`` ``(row, col)`` pairs from a ``[H,W]`` bool mask.

    Only true positions are ever returned, which is how "targets only ``O_fit``"
    is enforced for the contrastive arm.
    """
    if support.ndim != 2 or support.dtype != torch.bool:
        raise ValueError("support must be a bool [H,W] tensor")
    width = support.shape[1]
    flat = support.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if flat.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=support.device)
        return empty, empty
    if flat.numel() > max_samples:
        flat = flat[_randperm(flat.numel(), support.device, generator)[:max_samples]]
    return torch.div(flat, width, rounding_mode="floor"), flat % width


def _block_mask(shape: tuple[int, int, int], device: torch.device,
                generator: torch.Generator | None) -> torch.Tensor:
    """Random local square blocks, ``[B,H,W]`` bool."""
    batch, height, width = shape
    block = max(1, min(8, min(height, width) // 4))
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    rows = _randint(height - block + 1, (batch, RECONSTRUCTION_BLOCKS), device, generator)
    cols = _randint(width - block + 1, (batch, RECONSTRUCTION_BLOCKS), device, generator)
    for image in range(batch):
        for corner in range(RECONSTRUCTION_BLOCKS):
            row = int(rows[image, corner].item())
            col = int(cols[image, corner].item())
            mask[image, row:row + block, col:col + block] = True
    return mask


def _masked_mean_squared_error(prediction: torch.Tensor, target: torch.Tensor,
                               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean squared error over ``mask``, with the counted support returned.

    ``mask`` is broadcast to the prediction shape before counting, so a channel
    axis is normalized by the number of scored *elements* rather than by the
    number of scored pixels.
    """
    weight = mask.expand_as(prediction).to(prediction.dtype)
    count = weight.sum()
    if count.item() <= 0:
        return prediction.sum() * 0.0, count
    error = ((prediction - target) ** 2 * weight).sum() / count.clamp_min(1.0)
    return error, count


def _scalar(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, bool):
        return torch.tensor(value, device=device)
    return torch.tensor(float(value), device=device)


def producer_loss(model: nn.Module, context: torch.Tensor, fit_support: torch.Tensor,
                  *, generator: torch.Generator | None = None
                  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Image-only self-supervision for the representation producer.

    Parameters
    ----------
    model:
        A :class:`~.models.Producer`; called three times per step (untransformed
        context, dihedral-transformed context, block-masked context).
    context:
        ``[B,3,H,W]`` float fitting context. Withheld pixels and their guard band
        are already physically zero by the data contract.
    fit_support:
        ``[B,H,W]`` bool mask of pixels that belong to ``O_fit``. All three
        context channels are physically zeroed outside this mask before the
        first forward pass, and every sampled location, every reconstruction
        target and every equivariance residual is restricted to it. The loss is
        therefore a function of the fit observations alone: mutating any pixel
        outside the support cannot change the returned value or its gradient.
    generator:
        Optional :class:`torch.Generator` for reproducible sampling. A generator
        living on another device is used on its own device and the indices are
        moved, so determinism survives a CPU generator with CUDA tensors.

    Returns
    -------
    ``(loss, metrics)`` where ``loss`` is a differentiable scalar and ``metrics``
    maps names to detached zero-dimensional tensors, including the per-arm
    losses, the counted supports, the number of contributing images, and the
    SSL collapse indicators ``feature_std_mean`` / ``feature_std_min`` /
    ``feature_cosine_mean``, and ``hidden_intensity_elements``, the number of
    non-zero context elements outside the fit support that had to be cleared.

    An arm with no usable support contributes an exact differentiable zero and
    reports a zero support count; a batch with no fit support at all returns a
    differentiable zero total with ``skipped`` true.
    """
    if context.ndim != 4 or context.shape[1] != CONTEXT_CHANNELS:
        raise ValueError(f"context must be [B,{CONTEXT_CHANNELS},H,W], got {tuple(context.shape)}")
    if fit_support.dtype != torch.bool or fit_support.shape != (context.shape[0], *context.shape[2:]):
        raise ValueError("fit_support must be a bool [B,H,W] tensor matching context")
    if not torch.isfinite(context).all():
        raise ValueError("context contains non-finite values")

    device = context.device
    batch, _, height, width = context.shape

    # Physically zero every channel outside the fit support *before the model
    # sees anything*. The data layer is already required to deliver withheld
    # pixels and their guard band as zeros in all three context channels, so for
    # a legal input this is a no-op. For an illegal one it makes the documented
    # immunity real instead of merely claimed: without it the producer would
    # forward hidden intensities, and every feature, hence every arm of this
    # loss, would depend on observations the fitting side is not allowed to see.
    # The count of elements this had to clear is reported as
    # ``hidden_intensity_elements``; a non-zero value is an upstream contract
    # violation that the trainer must surface rather than absorb.
    inside = fit_support.unsqueeze(1)
    hidden_elements = int(((context != 0) & ~inside).sum().item())
    context = context.float()
    context = context * inside.to(context.dtype)

    op = _sample_op(generator, device)
    transformed = _apply_op(context, op)

    out_reference = model(context)
    out_transformed = model(transformed)
    features_reference = out_reference["features"]
    features_aligned = _invert_op(out_transformed["features"], op)
    if features_aligned.shape != features_reference.shape:
        raise RuntimeError("inverse transform did not restore the feature geometry")

    zero = features_reference.sum() * 0.0

    # --- arm 3: geometric equivariance, fit pixels only -----------------------
    equivariance, equivariance_support = _masked_mean_squared_error(
        features_aligned, features_reference, fit_support.unsqueeze(1)
    )

    # --- arm 1: bounded dense contrastive correspondence ----------------------
    contrastive_terms: list[torch.Tensor] = []
    cosine_terms: list[torch.Tensor] = []
    sampled_total = 0
    for image in range(batch):
        rows, cols = _sample_support_locations(fit_support[image], generator)
        if rows.numel() < 2:
            continue
        anchor = F.normalize(features_reference[image, :, rows, cols].t(), dim=1)
        positive = F.normalize(features_aligned[image, :, rows, cols].t(), dim=1)
        similarity = anchor @ positive.t() / CONTRASTIVE_TEMPERATURE
        distance = (rows[:, None] - rows[None, :]) ** 2 + (cols[:, None] - cols[None, :]) ** 2
        identity = torch.eye(rows.numel(), dtype=torch.bool, device=device)
        too_close = (distance.float() < MIN_NEGATIVE_SEPARATION ** 2) & ~identity
        # An anchor with no admissible negative is a degenerate all-positive
        # softmax that would contribute an uninformative near-zero term, so that
        # row is dropped. Columns stay intact, so surviving anchors keep their
        # full negative pool.
        keep = (~too_close & ~identity).any(dim=1)
        if not bool(keep.any()):
            continue
        similarity = similarity.masked_fill(too_close, MASKED_LOGIT)
        target = torch.arange(rows.numel(), device=device)
        contrastive_terms.append(F.cross_entropy(similarity[keep], target[keep]))
        cosine_terms.append((anchor @ anchor.t()).masked_select(~identity).mean().detach())
        sampled_total += int(keep.sum().item())
    if contrastive_terms:
        contrastive = torch.stack(contrastive_terms).mean()
        cosine_mean = torch.stack(cosine_terms).mean()
    else:
        contrastive = zero
        cosine_mean = torch.zeros((), device=device)

    # --- arm 2: local masked reconstruction, fit targets only -----------------
    block = _block_mask((batch, height, width), device, generator)
    masked_context = context.masked_fill(block.unsqueeze(1), 0.0)
    reconstruction = model(masked_context)["reconstruction"]
    recon_mask = block & fit_support
    reconstruction_loss, reconstruction_support = _masked_mean_squared_error(
        reconstruction[:, 0], context[:, CONTEXT_CHANNELS // 2], recon_mask
    )

    total = (
        WEIGHT_CONTRASTIVE * contrastive
        + WEIGHT_RECONSTRUCTION * reconstruction_loss
        + WEIGHT_EQUIVARIANCE * equivariance
    )
    skipped = not fit_support.any()
    if skipped:
        total = zero
    if not torch.isfinite(total):
        raise FloatingPointError("producer loss is non-finite")

    with torch.no_grad():
        flat_features = features_reference.detach().permute(1, 0, 2, 3).reshape(
            features_reference.shape[1], -1
        )
        flat_support = fit_support.reshape(-1)
        if int(flat_support.sum().item()) > 1:
            observed = flat_features[:, flat_support]
            channel_std = observed.std(dim=1)
        else:
            channel_std = torch.zeros(features_reference.shape[1], device=device)

    metrics = {
        "loss": _scalar(total, device),
        "contrastive": _scalar(contrastive, device),
        "reconstruction": _scalar(reconstruction_loss, device),
        "equivariance": _scalar(equivariance, device),
        "contrastive_images": _scalar(len(contrastive_terms), device),
        "contrastive_samples": _scalar(sampled_total, device),
        "reconstruction_support": _scalar(reconstruction_support, device),
        "equivariance_support": _scalar(equivariance_support, device),
        "feature_std_mean": _scalar(channel_std.mean(), device),
        "feature_std_min": _scalar(channel_std.min() if channel_std.numel() else 0.0, device),
        "feature_cosine_mean": _scalar(cosine_mean, device),
        "hidden_intensity_elements": _scalar(hidden_elements, device),
        "transform_rotations": _scalar(op[0], device),
        "transform_flip": _scalar(op[1], device),
        "skipped": _scalar(skipped, device),
    }
    return total, metrics


def student_loss(logits: torch.Tensor, probabilities: torch.Tensor, validity: torch.Tensor
                 ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Validity-weighted soft cross entropy against detached pseudo-labels.

    Parameters
    ----------
    logits:
        ``[B,4,H,W]`` student output in the frozen named semantic order.
    probabilities:
        ``[B,4,H,W]`` generated soft labels, already a normalized distribution
        over the class axis wherever ``validity > 0``. Detached here, so no
        gradient reaches the producer, the hypothesis bank or the other student
        arm even if the caller forgot to detach them. Values outside ``[0,1]``,
        non-finite values, and class vectors on valid support that do not sum to
        one within :data:`PROBABILITY_SUM_TOLERANCE` are rejected. Pixels with
        zero validity are unconstrained, so a bank may leave them at zero.
    validity:
        ``[B,H,W]`` weights in ``[0,1]``; values outside that range are rejected.
        Ambiguous pixels are down-weighted, never relabelled as background.

    Returns
    -------
    ``(loss, metrics)``. The denominator is ``validity.sum()``, that is the
    valid support, not the pixel count, so down-weighting pixels cannot inflate
    or deflate the reported loss scale. When the whole batch is invalid the
    function returns a differentiable exact zero and sets ``metrics['skipped']``
    true; the trainer is expected to log that flag rather than silently average
    a zero into its history.

    The same function is used by both student arms, so the arms differ only in
    the pseudo-labels they receive.
    """
    if logits.ndim != 4 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"logits must be [B,{NUM_CLASSES},H,W], got {tuple(logits.shape)}")
    if probabilities.shape != logits.shape:
        raise ValueError("probabilities must match the logits shape [B,4,H,W]")
    if validity.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("validity must be [B,H,W] matching the logits")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("student logits are non-finite")
    if not torch.isfinite(probabilities).all() or not torch.isfinite(validity).all():
        raise ValueError("pseudo-label probabilities or validity contain non-finite values")

    device = logits.device
    targets = probabilities.detach().float()
    weights = validity.detach().float()
    if bool(((targets < 0.0) | (targets > 1.0)).any()):
        raise ValueError("pseudo-label probabilities must lie in [0,1]; received "
                         f"min {float(targets.min()):.6g} max {float(targets.max()):.6g}")
    if bool(((weights < 0.0) | (weights > 1.0)).any()):
        raise ValueError("validity weights must lie in [0,1]; received "
                         f"min {float(weights.min()):.6g} max {float(weights.max()):.6g}")

    support = weights.sum()
    skipped = bool(support.item() <= 0.0)
    if not skipped:
        # On valid support the class axis must already be a distribution. This
        # is checked, never repaired: renormalizing would invent a target the
        # hypothesis bank never produced, and an all-zero class vector would
        # otherwise contribute exactly zero to the loss while still counting in
        # the denominator, quietly diluting a real gradient.
        valid = weights > 0.0
        deviation = (targets.sum(dim=1) - 1.0).abs()[valid]
        if bool((deviation > PROBABILITY_SUM_TOLERANCE).any()):
            raise ValueError(
                "pseudo-label probabilities must sum to 1 over the class axis on "
                f"valid support; worst deviation {float(deviation.max()):.6g} at "
                f"{int((deviation > PROBABILITY_SUM_TOLERANCE).sum())} of "
                f"{int(valid.sum())} valid pixels"
            )

    log_probabilities = F.log_softmax(logits.float(), dim=1)
    per_pixel = -(targets * log_probabilities).sum(dim=1)

    if skipped:
        loss = logits.sum() * 0.0
    else:
        loss = (per_pixel * weights).sum() / support.clamp_min(EPS)
    if not torch.isfinite(loss):
        raise FloatingPointError("student loss is non-finite")

    with torch.no_grad():
        entropy = -(targets * targets.clamp_min(EPS).log()).sum(dim=1)
        weighted_entropy = (
            (entropy * weights).sum() / support.clamp_min(EPS) if not skipped
            else torch.zeros((), device=device)
        )

    metrics = {
        "loss": _scalar(loss, device),
        "valid_support": _scalar(support, device),
        "valid_fraction": _scalar(support / max(1, weights.numel()), device),
        "pixels": _scalar(weights.numel(), device),
        "target_entropy": _scalar(weighted_entropy, device),
        "skipped": _scalar(skipped, device),
    }
    return loss, metrics


__all__ = [
    "VERSION",
    "MAX_SAMPLES_PER_IMAGE",
    "MIN_NEGATIVE_SEPARATION",
    "PROBABILITY_SUM_TOLERANCE",
    "producer_loss",
    "student_loss",
]
