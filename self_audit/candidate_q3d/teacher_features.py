"""Feature and phase accessors for the Q-3D nnFoundation ResEnc teacher."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from .alignment import GridPhase
from .teacher_resenc import Q3DResEncTeacher

_SOURCE_ALIASES = {
    "encoder_stride8": "encoder_stride8",
    "stride8": "encoder_stride8",
    "encoder_stride16": "encoder_stride16",
    "stride16": "encoder_stride16",
}


def _canonical_source(source: str) -> str:
    if not isinstance(source, str) or source not in _SOURCE_ALIASES:
        raise ValueError("teacher feature source must be encoder_stride8 or encoder_stride16")
    return _SOURCE_ALIASES[source]


def teacher_feature_source(
    teacher: Q3DResEncTeacher,
    source: str = "encoder_stride8",
    *,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Describe one encoder skip without registering hooks or reading logits."""

    if not isinstance(teacher, Q3DResEncTeacher):
        raise TypeError("teacher must be Q3DResEncTeacher")
    canonical = _canonical_source(source)
    index = 3 if canonical == "encoder_stride8" else 4
    channels = int(teacher.encoder_channels[index])
    return {
        "source": canonical,
        "module_name": f"encoder.stages.{index}",
        "channels": channels,
        "module": None,
        "features_api": "encoder_return_skips",
        "feature_index": index,
        "capture_api": "forward_with_features",
        "synthetic_proxy": bool(synthetic),
        "input_padding": "high_side_replicate_to_cumulative_stride",
        "phase": teacher.phase_for_source(canonical),
    }


def teacher_feature_phase(
    teacher: Q3DResEncTeacher,
    source: str = "encoder_stride8",
    *,
    synthetic: bool = False,
) -> GridPhase:
    if not isinstance(teacher, Q3DResEncTeacher):
        raise TypeError("teacher must be Q3DResEncTeacher")
    del synthetic  # Synthetic smoke uses the same ResEnc implementation.
    return teacher.phase_for_source(_canonical_source(source))


def teacher_grid_phase(
    teacher: Q3DResEncTeacher,
    source: str = "encoder_stride8",
    synthetic: bool = False,
) -> GridPhase:
    return teacher_feature_phase(teacher, source=source, synthetic=synthetic)


def forward_teacher_with_features(
    teacher: Q3DResEncTeacher,
    images: Tensor,
    *,
    geometry: Mapping[str, Any],
    source: str = "encoder_stride8",
    synthetic: bool = False,
) -> dict[str, Tensor]:
    """Run the teacher once and return its selected encoder feature.

    The teacher feature is detached because it is a target for tokenizer/KD;
    region/boundary outputs are detached as well, matching target generation.
    """

    details = teacher_feature_source(teacher, source, synthetic=synthetic)
    index = int(details["feature_index"])
    with torch.no_grad():
        outputs = teacher.forward_with_features(images, geometry=geometry)
    if not isinstance(outputs, Mapping):
        raise ValueError("Q3D ResEnc teacher output must be a mapping")
    feature_key = "feature_encoder_stride8" if index == 3 else "feature_encoder_stride16"
    feature = outputs.get(feature_key)
    if not isinstance(feature, Tensor):
        raise ValueError(f"Q3D ResEnc teacher did not return {feature_key}")
    if feature.ndim != 5 or feature.shape[1] != details["channels"]:
        raise ValueError("selected ResEnc teacher feature has unexpected shape")
    required = {"region_logits", "boundary_logits", "sdf"}
    if not required.issubset(outputs):
        raise ValueError("Q3D ResEnc teacher output is missing region/boundary/SDF tensors")
    return {
        "region_logits": outputs["region_logits"].detach(),
        "boundary_logits": outputs["boundary_logits"].detach(),
        "sdf": outputs["sdf"].detach(),
        "feature": feature.detach(),
    }


__all__ = [
    "forward_teacher_with_features",
    "teacher_feature_phase",
    "teacher_feature_source",
    "teacher_grid_phase",
]
