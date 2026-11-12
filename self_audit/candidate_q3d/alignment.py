"""Native-grid alignment for Q-3D teacher features.

Feature padding is decided by the feature producer. The student keeps native z
and has stride 16 only in-plane. This module maps teacher cells to student S4
without assuming a specific network minimum, consulting affine geometry, or
allowing gradients to flow into the teacher.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import torch
from torch import Tensor
from torch.nn import functional as F


def _native_shape(value: Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("native_shape must contain positive integer D,H,W")
    shape: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension <= 0:
            raise ValueError("native_shape must contain positive integer D,H,W")
        shape.append(int(dimension))
    return tuple(shape)  # type: ignore[return-value]


def align_teacher_to_student4(
    F_T_src: Tensor,
    native_shape: Sequence[int],
    *,
    teacher_stride: int = 8,
) -> Tensor:
    """Align MedNeXt features to student S4 on the native grid.

    Args:
        F_T_src: ``[B,C,Dp/s,Hp/s,Wp/s]`` features from the feature producer,
            where ``s`` is 4, 8, or 16. Padding policy belongs to the producer;
            the helper requires only native-depth coverage and full S4 pooling
            support in-plane.
        native_shape: The unpadded input shape ``(D,H,W)``.
        teacher_stride: Isotropic MedNeXt feature stride. Stride 8 is the main
            path; 4 is the detail support and 16 is the bottleneck ablation.

    Returns:
        A detached ``[B,C,D,ceil(H/16),ceil(W/16)]`` tensor. In-plane cells
        use exact non-overlapping average pooling over full teacher cells,
        including cells that contain replicate padding. Through-plane
        alignment linearly samples each native slice center.
    """

    depth, height, width = _native_shape(native_shape)
    if isinstance(teacher_stride, bool) or not isinstance(teacher_stride, Integral) \
            or int(teacher_stride) not in (4, 8, 16):
        raise ValueError("teacher_stride must be one of 4, 8, or 16")
    teacher_stride = int(teacher_stride)
    if not isinstance(F_T_src, Tensor) or F_T_src.ndim != 5:
        raise ValueError("F_T_src must be a [B,C,D,H,W] tensor")
    if any(size <= 0 for size in F_T_src.shape):
        raise ValueError("F_T_src dimensions must be positive")
    if not torch.is_floating_point(F_T_src):
        raise ValueError("F_T_src must use a floating dtype")
    if not torch.isfinite(F_T_src).all():
        raise ValueError("F_T_src contains NaN or Inf")
    target_h = (height + 15) // 16
    target_w = (width + 15) // 16
    pool_factor = 16 // teacher_stride
    if F_T_src.shape[-3] * teacher_stride < depth:
        raise ValueError("teacher feature does not cover native depth")
    if F_T_src.shape[-2] < target_h * pool_factor \
            or F_T_src.shape[-1] < target_w * pool_factor:
        raise ValueError("teacher feature does not provide full in-plane pooling coverage")
    work_dtype = torch.float32 if F_T_src.dtype in (torch.float16, torch.bfloat16) else F_T_src.dtype
    feature = F_T_src.detach().to(dtype=work_dtype)
    if pool_factor > 1:
        pooled = F.avg_pool3d(
            feature,
            kernel_size=(1, pool_factor, pool_factor),
            stride=(1, pool_factor, pool_factor),
        )
    else:
        pooled = feature
    pooled = pooled[..., :target_h, :target_w]

    # With align_corners=False, native slice center z+0.5 maps to this
    # teacher-cell coordinate. Clamp before floor/ceil so endpoint values are
    # replicated beyond the first/last feature-cell centers.
    coordinates = (
        (torch.arange(depth, device=pooled.device, dtype=pooled.dtype) + 0.5)
        / float(teacher_stride)
        - 0.5
    ).clamp_(0.0, float(pooled.shape[2] - 1))
    lower = coordinates.floor().to(torch.long)
    upper = coordinates.ceil().to(torch.long)
    fraction = (coordinates - lower.to(coordinates.dtype)).view(1, 1, depth, 1, 1)

    low_values = pooled.index_select(2, lower)
    high_values = pooled.index_select(2, upper)
    aligned = low_values + (high_values - low_values) * fraction
    return aligned.to(dtype=F_T_src.dtype).detach()


__all__ = ["align_teacher_to_student4"]
