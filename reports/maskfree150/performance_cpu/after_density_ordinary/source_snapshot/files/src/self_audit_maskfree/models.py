"""Small random-initialized dense models: one producer, two identical students (W4).

Scientific role
---------------
This module supplies the *only* learned parameters in the mask-free pipeline:

* :class:`Producer` — a small GroupNorm encoder/decoder that turns the three-slice
  fitting context into a dense feature map plus a reconstruction head. Its
  features are the input to bounded hypothesis generation; its reconstruction
  head exists so that masked-reconstruction self-supervision has a target-free
  output. It is trained by image-only objectives in :mod:`.losses` and receives
  no gradient from either student.
* :class:`Student` — a lightweight dense classifier over the fixed named semantic
  order ``0=BG, 1=RV, 2=MYO, 3=LV``. Two students are created with byte-identical
  initial parameters and no shared storage: ``student_no_audit`` and
  ``student_audited``. They differ only in the pseudo-labels they consume, so any
  measured difference between the arms cannot be an initialization difference.

Supervision firewall
--------------------
Initialization is random. There is no checkpoint loading, no download, no
``pretrained`` flag, no import of :mod:`self_audit` or
:mod:`self_audit_candidate_c`, and no architecture choice informed by manual
masks. The parameter budget for the first profile is 5M parameters or less and
is enforced, not assumed: :func:`make_models` measures the real counts and
raises if the producer exceeds :data:`MAX_PRODUCER_PARAMETERS`.

Shape and small-image behaviour
-------------------------------
All modules accept ``[B, 3, H, W]`` float32 and return dense maps at the input
resolution. Down-sampling uses fixed 2x2 average pooling whenever both spatial
dimensions are even and at least two pixels; adaptive average pooling to
``max(1, size // 2)`` remains the fallback when either dimension is odd or
singleton. Up-sampling interpolates back to the stored skip size, so odd,
non-square and very small inputs (down to ``1x1``) are supported without
padding hacks.
GroupNorm group counts are chosen per channel width so that the channel count is
always divisible by the group count **and every group holds at least
:data:`MIN_CHANNELS_PER_GROUP` channels**; normalization statistics are computed
over ``C/G * H * W`` elements, which stays well defined for a single pixel
because GroupNorm never normalizes over the batch axis. A one-channel group
would have zero variance at ``1x1`` and would emit a constant zero, so the
narrowest supported width (4) uses two channels per group rather than four
single-channel groups.
"""
from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import VERSION

#: First-profile parameter budget for the representation producer. Measured, not
#: assumed; :func:`make_models` raises when it is exceeded.
MAX_PRODUCER_PARAMETERS = 5_000_000

#: Number of semantic classes in the frozen named order ``0=BG,1=RV,2=MYO,3=LV``.
NUM_CLASSES = 4

#: Number of context channels (neighbour slices, withheld blocks physically zero).
CONTEXT_CHANNELS = 3


#: Minimum number of channels inside one GroupNorm group. A group holding a
#: single channel normalizes over ``H*W`` elements only, so at ``1x1`` its
#: variance is exactly zero and the block would output a constant zero
#: regardless of its input. Two channels per group keep the statistic defined
#: down to a single pixel, which is what the small-image promise requires.
MIN_CHANNELS_PER_GROUP = 2


def _groups(channels: int) -> int:
    """Largest group count in ``(8, 4, 2, 1)`` that divides ``channels`` and
    leaves at least :data:`MIN_CHANNELS_PER_GROUP` channels per group."""
    for group in (8, 4, 2):
        if channels % group == 0 and channels // group >= MIN_CHANNELS_PER_GROUP:
            return group
    return 1


def _down(x: torch.Tensor) -> torch.Tensor:
    """Halve spatial size, never below one pixel.

    The production 128 -> 64 -> 32 path uses the fixed-kernel operator, which
    has a deterministic 2x2 stencil on even grids.  Adaptive pooling remains
    the correctness-compatible fallback for odd or singleton dimensions.
    """
    height, width = x.shape[-2:]
    if height >= 2 and width >= 2 and height % 2 == 0 and width % 2 == 0:
        return F.avg_pool2d(x, kernel_size=2, stride=2)
    return F.adaptive_avg_pool2d(x, (max(1, height // 2), max(1, width // 2)))


def _up(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Resize ``x`` to the spatial size of ``like``."""
    size = tuple(like.shape[-2:])
    mode = "bilinear" if min(size) > 1 else "nearest"
    if mode == "bilinear":
        return F.interpolate(x, size=size, mode=mode, align_corners=False)
    return F.interpolate(x, size=size, mode=mode)


class _ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by GroupNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _DenseTrunk(nn.Module):
    """Two-level GroupNorm encoder/decoder producing a dense map at input size."""

    def __init__(self, in_channels: int = CONTEXT_CHANNELS, width: int = 16) -> None:
        super().__init__()
        if width < 4:
            raise ValueError("width must be at least 4")
        self.enc0 = _ConvBlock(in_channels, width)
        self.enc1 = _ConvBlock(width, 2 * width)
        self.enc2 = _ConvBlock(2 * width, 4 * width)
        self.dec1 = _ConvBlock(4 * width + 2 * width, 2 * width)
        self.dec0 = _ConvBlock(2 * width + width, width)
        self.out_channels = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip0 = self.enc0(x)
        skip1 = self.enc1(_down(skip0))
        bottleneck = self.enc2(_down(skip1))
        up1 = self.dec1(torch.cat([_up(bottleneck, skip1), skip1], dim=1))
        return self.dec0(torch.cat([_up(up1, skip0), skip0], dim=1))


def _validate_input(x: torch.Tensor, name: str) -> None:
    if x.ndim != 4 or x.shape[1] != CONTEXT_CHANNELS:
        raise ValueError(f"{name} must be [B,{CONTEXT_CHANNELS},H,W], got {tuple(x.shape)}")
    if x.shape[0] == 0 or x.shape[2] == 0 or x.shape[3] == 0:
        raise ValueError(f"{name} has an empty axis: {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values")


class Producer(nn.Module):
    """Dense self-supervised representation model.

    ``forward(context)`` takes ``[B,3,H,W]`` and returns a dict with

    * ``features`` ``[B,feature_dim,H,W]`` — dense descriptors for hypothesis
      generation and for the contrastive / equivariance objectives;
    * ``reconstruction`` ``[B,1,H,W]`` — prediction of the centre context
      channel, used only by the masked-reconstruction objective on fit pixels.

    The reconstruction head has no skip connection from the target intensities
    beyond the ordinary trunk input, which is the masked image itself; the loss
    supervises only pixels that were masked out of that input.
    """

    def __init__(self, feature_dim: int = 16, width: int = 16) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = feature_dim
        self.width = width
        self.trunk = _DenseTrunk(CONTEXT_CHANNELS, width)
        self.feature_head = nn.Conv2d(width, feature_dim, 1)
        self.reconstruction_head = nn.Conv2d(width, 1, 1)

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        _validate_input(context, "context")
        hidden = self.trunk(context)
        return {
            "features": self.feature_head(hidden),
            "reconstruction": self.reconstruction_head(hidden),
        }


class Student(nn.Module):
    """Lightweight dense segmentation student over the frozen semantic order."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.width = width
        self.trunk = _DenseTrunk(CONTEXT_CHANNELS, width)
        self.head = nn.Conv2d(width, NUM_CLASSES, 1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        _validate_input(context, "context")
        return self.head(self.trunk(context))


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Measured parameter count of ``model``."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in model.parameters() if p.requires_grad)
    return int(sum(p.numel() for p in params))


@contextlib.contextmanager
def _seeded(seed: int):
    """Seed the global CPU RNG, then restore the caller's stream."""
    state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(state)


def make_models(seed: int = 42, width: int = 16, feature_dim: int = 16) -> dict[str, Any]:
    """Build the producer and the two identically initialized students.

    Returns a dict with ``producer``, ``student_no_audit``, ``student_audited``,
    the measured ``parameter_counts`` and the contract ``version``.

    ``student_audited`` is constructed independently and then loaded from a
    deep copy of ``student_no_audit``'s state dict, so the two arms start from
    byte-identical values while owning separate storage. Sharing storage would
    silently couple the arms and invalidate the whole comparison, so the caller
    is expected to assert both properties; :func:`make_models` itself checks
    that no parameter storage is aliased between the arms.

    The global CPU RNG is seeded for construction and restored afterwards, so
    calling this function does not perturb a trainer's sampling stream.
    """
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    with _seeded(seed):
        producer = Producer(feature_dim=feature_dim, width=width)
        student_no_audit = Student(width=width)
        student_audited = Student(width=width)
    reference = {k: v.detach().clone() for k, v in student_no_audit.state_dict().items()}
    student_audited.load_state_dict(reference)

    no_audit_ptrs = {p.data_ptr() for p in student_no_audit.parameters()}
    audited_ptrs = {p.data_ptr() for p in student_audited.parameters()}
    if no_audit_ptrs & audited_ptrs:
        raise RuntimeError("student arms share parameter storage")

    counts = {
        "producer": count_parameters(producer),
        "student_no_audit": count_parameters(student_no_audit),
        "student_audited": count_parameters(student_audited),
    }
    counts["total_trainable"] = sum(counts.values())
    if counts["producer"] > MAX_PRODUCER_PARAMETERS:
        raise ValueError(
            f"producer has {counts['producer']} parameters, budget is "
            f"{MAX_PRODUCER_PARAMETERS}"
        )
    return {
        "producer": producer,
        "student_no_audit": student_no_audit,
        "student_audited": student_audited,
        "parameter_counts": counts,
        "version": VERSION,
        "seed": int(seed),
        "width": int(width),
        "feature_dim": int(feature_dim),
    }
