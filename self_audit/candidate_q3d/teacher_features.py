"""One-pass feature capture from the verified Candidate A-3D MedNeXt teacher.

The hook observes an existing module execution inside
``MedNeXtL3DAdapter._native_forward_with_features``.  It does not run a second
trunk pass and never substitutes logits for a feature representation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from ..candidate_a3d.teacher import DualBranchTeacher3D, MedNeXtL3DAdapter


_FEATURE_MODULES = {
    "stride8": ("dec_block_3", 3),
    "stride16": ("bottleneck", 4),
    "stride4": ("dec_block_2", 2),
}


def teacher_feature_source(teacher: DualBranchTeacher3D, source: str = "stride8") -> dict[str, Any]:
    """Return the actual validated module and channel count for a source."""

    if not isinstance(teacher, DualBranchTeacher3D):
        raise TypeError("teacher must be the existing DualBranchTeacher3D")
    if source not in _FEATURE_MODULES:
        raise ValueError("source must be stride8, stride16, or stride4")
    encoder = teacher.encoder
    if not isinstance(encoder, MedNeXtL3DAdapter):
        raise ValueError("Q-3D feature capture requires the verified MedNeXt-L adapter")
    model = encoder.model
    if not getattr(model, "_candidate_a_native_v1", False):
        raise ValueError("MedNeXt native v1 module sequence has not been validated")
    module_name, channel_index = _FEATURE_MODULES[source]
    module = getattr(model, module_name, None)
    if not isinstance(module, Module):
        raise ValueError(f"validated MedNeXt module {module_name!r} is unavailable")
    # The official L factory can checkpoint a sequential by iterating its
    # children. In that mode the outer Sequential.forward hook is bypassed;
    # capture the final executed child, whose output is the block output.
    capture_name = module_name
    if bool(getattr(model, "outside_block_checkpointing", False)):
        children = tuple(module.children())
        if not children:
            raise ValueError(
                f"cannot observe checkpointed MedNeXt {module_name}: no executed child module"
            )
        module = children[-1]
        capture_name = f"{module_name}.{len(children) - 1}"
    if channel_index >= len(encoder.channels):
        raise ValueError("MedNeXt adapter.channels does not cover the selected feature source")
    channels = int(encoder.channels[channel_index])
    if channels <= 0:
        raise ValueError("selected MedNeXt feature channel count must be positive")
    return {
        "source": source,
        "module_name": capture_name,
        "channels": channels,
        "module": module,
        "features_api": encoder.features_api,
    }


def forward_teacher_with_features(
    teacher: DualBranchTeacher3D,
    images: Tensor,
    *,
    geometry: Mapping[str, Any],
    source: str = "stride8",
) -> dict[str, Tensor]:
    """Run the frozen teacher once and capture one MedNeXt decoder feature.

    The hook is installed only for the duration of this call and removed even
    if the teacher raises.  Q-3D uses this after teacher stage T, so the teacher
    must already be frozen and in evaluation mode.
    """

    if teacher.training:
        raise ValueError("Q-3D feature extraction requires teacher.eval()")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("Q-3D feature extraction requires a frozen teacher")
    if not isinstance(images, Tensor) or images.ndim != 5:
        raise ValueError("images must be native [B,1,D,H,W]")

    details = teacher_feature_source(teacher, source)
    captured: list[Tensor] = []

    def capture(_module: Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, Tensor):
            raise ValueError(f"MedNeXt {details['module_name']} output must be a tensor")
        captured.append(output.detach())

    handle = details["module"].register_forward_hook(capture)
    try:
        with torch.no_grad():
            outputs = teacher(images, geometry=geometry)
    finally:
        handle.remove()

    if len(captured) != 1:
        raise RuntimeError(
            f"expected one execution of MedNeXt {details['module_name']}, got {len(captured)}"
        )
    feature = captured[0]
    if feature.ndim != 5 or feature.shape[0] != images.shape[0] \
            or feature.shape[1] != details["channels"]:
        raise ValueError(
            f"MedNeXt {details['module_name']} must return [B,{details['channels']},D,H,W]"
        )
    if not torch.isfinite(feature).all():
        raise ValueError(f"MedNeXt {details['module_name']} feature contains NaN or Inf")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "region_logits", "boundary_logits", "sdf",
    }:
        raise ValueError("A-3D teacher must return its canonical region, boundary and SDF mapping")
    return {**dict(outputs), "feature": feature}


__all__ = ["forward_teacher_with_features", "teacher_feature_source"]
