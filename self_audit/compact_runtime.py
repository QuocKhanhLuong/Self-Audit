"""Explicit deterministic execution for matched compact runs, without RNG draws."""
from contextlib import contextmanager
from collections.abc import Mapping
from functools import wraps
import os

import torch


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
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic = previous[2]
        torch.backends.cudnn.benchmark = previous[3]
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
