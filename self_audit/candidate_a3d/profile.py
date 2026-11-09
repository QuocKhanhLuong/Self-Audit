"""Wall-clock profiling for full-volume A-3D inference."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


def validate_device(device: str | torch.device) -> torch.device:
    """Accept only the profiled CPU/CUDA backends; never silently fall back."""

    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"unsupported A-3D device {device!r}; use CPU or CUDA") from exc
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported A-3D device {resolved}; use CPU or CUDA")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for A-3D evaluation but CUDA is unavailable")
    return resolved


def synchronize(device: str | torch.device) -> None:
    resolved = validate_device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class VolumeProfile:
    """Profile fields are wall-clock milliseconds, including GPU synchronisation."""

    fields: dict[str, float] = field(default_factory=dict)

    def measure(self, name: str, device: str | torch.device) -> _Measure:
        return _Measure(self, name, device)


class _Measure:
    def __init__(self, profile: VolumeProfile, name: str, device: str | torch.device):
        self.profile, self.name, self.device = profile, name, device

    def __enter__(self):
        synchronize(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        synchronize(self.device)
        self.profile.fields[self.name] = (time.perf_counter() - self.started) * 1000.0
        return False


__all__ = ["VolumeProfile", "synchronize", "validate_device"]
