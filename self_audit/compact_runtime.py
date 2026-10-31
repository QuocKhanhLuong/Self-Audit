"""Explicit deterministic execution for matched compact runs, without RNG draws."""
from contextlib import contextmanager
from collections.abc import Mapping
from functools import wraps
import os

import torch
from torch.nn import functional as F
from .compact_model import CompactDecoder


class _DeterministicBilinear(torch.autograd.Function):
    """Keep the native forward; replace only CUDA's atomic backward reduction.

    ATen's CPU backward implements the same align_corners=False bilinear
    derivative deterministically. Transfer/CPU overhead belongs to every arm's
    training cost. This path is never needed for no-grad student inference.
    """
    @staticmethod
    def forward(ctx, x, size):
        ctx.input_size = list(x.shape)
        ctx.output_size = list(size)
        ctx.input_device = x.device
        # F.interpolate switches to a different decomposition under strict
        # determinism in newer torch releases. Call the existing native forward
        # kernel explicitly so training, guard and exported inference agree.
        return torch.ops.aten.upsample_bilinear2d.default(x, list(size), False, None, None)

    @staticmethod
    def backward(ctx, grad_output):
        gradient = torch.ops.aten.upsample_bilinear2d_backward.default(
            grad_output.to("cpu"), ctx.output_size, ctx.input_size, False, None, None)
        return gradient.to(ctx.input_device), None


def compact_bilinear(x, size):
    if torch.are_deterministic_algorithms_enabled() and x.is_cuda:
        if torch.is_grad_enabled() and x.requires_grad:
            return _DeterministicBilinear.apply(x, tuple(size))
        return torch.ops.aten.upsample_bilinear2d.default(x, list(size), False, None, None)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _deterministic_decoder_forward(self, stages):
    """The existing decoder computation with an explicit bilinear backend."""
    d = stages[-1]
    for index, i in enumerate(range(len(self.widths) - 2, -1, -1)):
        skip = stages[i]
        up = compact_bilinear(d, skip.shape[-2:])
        merged = torch.cat([up, skip], dim=1)
        stage = self.up_stages[index]
        d = stage["refine"](stage["fuse"](merged))
    return d


@contextmanager
def deterministic_runtime(enabled=False):
    if not isinstance(enabled, bool):
        raise ValueError("deterministic_algorithms must be boolean")
    if not enabled:
        yield
        return
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8"):
        raise ValueError("deterministic compact execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if workspace is None and torch.cuda.is_initialized():
        raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        CompactDecoder.forward,
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Scoped to this training process; model definitions and exports stay unchanged.
        CompactDecoder.forward = _deterministic_decoder_forward
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic = previous[2]
        torch.backends.cudnn.benchmark = previous[3]
        CompactDecoder.forward = previous[4]
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def with_compact_runtime(function):
    @wraps(function)
    def wrapped(config, *args, **kwargs):
        if not isinstance(config, Mapping):
            return function(config, *args, **kwargs)
        with deterministic_runtime(config.get("deterministic_algorithms", False)):
            return function(config, *args, **kwargs)
    return wrapped
