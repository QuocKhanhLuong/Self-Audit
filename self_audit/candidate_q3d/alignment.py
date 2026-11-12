"""Native-grid phase, alignment, and centered teacher-target pooling for Q-3D."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import ceil, isfinite
from numbers import Integral, Real
from typing import TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F


GridPhase: TypeAlias = tuple[tuple[int, int, int], tuple[float, float, float]]


def _native_shape(value: Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("native_shape must contain positive integer D,H,W")
    shape: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension <= 0:
            raise ValueError("native_shape must contain positive integer D,H,W")
        shape.append(int(dimension))
    return tuple(shape)  # type: ignore[return-value]


def _phase(value: GridPhase, *, name: str) -> GridPhase:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be (step_zyx, offset_zyx)")
    step, offset = value
    if not isinstance(step, (tuple, list)) or not isinstance(offset, (tuple, list)) \
            or len(step) != 3 or len(offset) != 3:
        raise ValueError(f"{name} step and offset must each contain z,y,x")
    parsed_step: list[int] = []
    parsed_offset: list[float] = []
    for axis_step, axis_offset in zip(step, offset):
        if isinstance(axis_step, bool) or not isinstance(axis_step, Integral) or axis_step <= 0:
            raise ValueError(f"{name} step values must be positive integers")
        if isinstance(axis_offset, bool) or not isinstance(axis_offset, Real) \
                or not isfinite(float(axis_offset)):
            raise ValueError(f"{name} offset values must be finite numbers")
        parsed_step.append(int(axis_step))
        parsed_offset.append(float(axis_offset))
    return tuple(parsed_step), tuple(parsed_offset)  # type: ignore[return-value]


def grid_phase(conv_chain: Iterable[nn.Module]) -> GridPhase:
    """Derive the native-grid step and offset for an ordered sampling path.

    ``conv_chain`` must list only actual Conv3d or AvgPool3d sampling modules
    along one forward path, in execution order. In particular, callers with
    residual blocks must name the main path explicitly and validate branch
    phases separately; automatically walking a module tree could mistakenly
    count parallel paths.

    Each input sample has center coordinate 0. For a convolution with kernel
    ``k``, stride ``a``, left padding ``p`` and dilation ``d``, the new offset
    is ``o + (d*(k-1)/2 - p)*j`` and the new step is ``a*j``, axis by axis.
    """

    try:
        modules = tuple(conv_chain)
    except TypeError as exc:
        raise ValueError("conv_chain must be an ordered iterable of Conv3d/AvgPool3d modules") from exc
    step = [1, 1, 1]
    offset = [0.0, 0.0, 0.0]
    for index, conv in enumerate(modules):
        if isinstance(conv, nn.Conv3d):
            if isinstance(conv.padding, str):
                raise ValueError("string Conv3d padding is unsupported for phase derivation")
            kernel = tuple(int(value) for value in conv.kernel_size)
            stride = tuple(int(value) for value in conv.stride)
            padding = tuple(int(value) for value in conv.padding)
            dilation = tuple(int(value) for value in conv.dilation)
        elif isinstance(conv, nn.AvgPool3d):
            kernel = _as_3tuple(conv.kernel_size, "AvgPool3d kernel_size")
            stride = _as_3tuple(conv.stride or conv.kernel_size, "AvgPool3d stride")
            padding = _as_3tuple(conv.padding, "AvgPool3d padding")
            dilation = (1, 1, 1)
        else:
            raise TypeError(f"conv_chain[{index}] must be Conv3d or AvgPool3d")
        if any(value <= 0 for value in (*kernel, *stride, *dilation)):
            raise ValueError("sampling kernel, stride, and dilation must be positive")
        for axis in range(3):
            offset[axis] += (
                dilation[axis] * (kernel[axis] - 1) / 2.0 - padding[axis]
            ) * step[axis]
            step[axis] *= stride[axis]
    return tuple(step), tuple(offset)  # type: ignore[return-value]


def _as_3tuple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, Integral) and not isinstance(value, bool):
        result = (int(value),) * 3
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        result = tuple(int(item) for item in value)
    else:
        raise ValueError(f"{name} must be an int or a three-value sequence")
    if any(item < 0 for item in result):
        raise ValueError(f"{name} values must be nonnegative")
    return result


def _student_phase_convs(student: nn.Module) -> tuple[nn.Conv3d, ...]:
    """Read the actual student stem and four down-convolution modules."""

    try:
        modules = (
            student.stem[0].conv,
            student.stage1[0].conv,
            student.stage2[0].conv,
            student.stage3.down.conv,
            student.stage4.down.conv,
        )
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("student does not expose the Q-3D stem and four down convolutions") from exc
    if any(not isinstance(module, nn.Conv3d) for module in modules):
        raise ValueError("student phase path must contain actual Conv3d modules")
    return modules


def student_grid_phase(student: nn.Module) -> GridPhase:
    """Derive the Q-3D S4 grid phase from the live student down modules."""

    return grid_phase(_student_phase_convs(student))


def teacher_grid_phase(teacher: nn.Module, source: str, synthetic: bool = False) -> GridPhase:
    """Read the phase of an actual loaded teacher feature source.

    The implementation lives beside the source selection and one-pass hook in
    ``teacher_features``. This public alias keeps phase consumers independent
    of the feature-capture implementation module.
    """

    from .teacher_features import teacher_feature_phase

    return teacher_feature_phase(teacher, source, synthetic=synthetic)


def _target_shape(native_shape: tuple[int, int, int], phase: GridPhase) -> tuple[int, int, int]:
    step, _offset = phase
    return tuple(ceil(length / stride) for length, stride in zip(native_shape, step))


def _native_axis_centers(
    native_length: int,
    target_length: int,
    step: int,
    offset: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if target_length != ceil(native_length / step):
        raise ValueError("target grid shape does not match native_shape and student phase")
    return offset + torch.arange(target_length, device=device, dtype=dtype) * step


def _sample_axis(feature: Tensor, coordinates: Tensor, axis: int) -> Tensor:
    size = int(feature.shape[axis])
    if size <= 0:
        raise ValueError("teacher feature axes must be nonempty")
    coordinates = coordinates.clamp(min=0.0, max=float(size - 1))
    lower = coordinates.floor().to(torch.long)
    upper = (lower + 1).clamp(max=size - 1)
    fraction = coordinates - lower.to(coordinates.dtype)
    shape = [1] * feature.ndim
    shape[axis] = coordinates.numel()
    fraction = fraction.view(shape)
    low_values = feature.index_select(axis, lower)
    high_values = feature.index_select(axis, upper)
    return low_values + (high_values - low_values) * fraction


def align_teacher_to_student4(
    F_T_src: Tensor,
    native_shape: Sequence[int],
    teacher_phase: GridPhase,
    student_phase: GridPhase,
) -> Tensor:
    """Trilinearly sample teacher features at every native student S4 center.

    The source and target phases are measured from their actual convolution
    chains. Values before/after the first/last teacher feature center use
    endpoint replication. Input padding is high-side only in the supported
    producers, so it does not change the phase origin.
    """

    shape = _native_shape(native_shape)
    teacher_phase = _phase(teacher_phase, name="teacher_phase")
    student_phase = _phase(student_phase, name="student_phase")
    if not isinstance(F_T_src, Tensor) or F_T_src.ndim != 5:
        raise ValueError("F_T_src must be a [B,C,D,H,W] tensor")
    if any(size <= 0 for size in F_T_src.shape):
        raise ValueError("F_T_src dimensions must be positive")
    if not torch.is_floating_point(F_T_src) or not torch.isfinite(F_T_src).all():
        raise ValueError("F_T_src must contain finite floating-point values")

    output_shape = _target_shape(shape, student_phase)
    work_dtype = torch.float64 if F_T_src.dtype == torch.float64 else torch.float32
    feature = F_T_src.detach().to(dtype=work_dtype)
    coordinates: list[Tensor] = []
    teacher_step, teacher_offset = teacher_phase
    student_step, student_offset = student_phase
    for axis, (length, target_length, t_step, t_offset, s_step, s_offset) in enumerate(
        zip(shape, output_shape, teacher_step, teacher_offset, student_step, student_offset)
    ):
        native_centers = _native_axis_centers(
            length, target_length, s_step, s_offset,
            device=feature.device, dtype=work_dtype,
        )
        source_coordinates = (native_centers - t_offset) / t_step
        minimum_source_length = ceil(length / t_step)
        if feature.shape[axis + 2] < minimum_source_length:
            raise ValueError(
                "teacher feature grid does not cover native_shape at its declared stride"
            )
        coordinates.append(source_coordinates)
    for axis, values in enumerate(coordinates, start=2):
        feature = _sample_axis(feature, values, axis)
    return feature.to(dtype=F_T_src.dtype).detach()


def _axis_overlap_weights(
    native_length: int,
    target_length: int,
    step: int,
    offset: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Voxel-support overlap with Voronoi cells centered on the target grid."""

    centers = _native_axis_centers(
        native_length, target_length, step, offset, device=device, dtype=dtype,
    )
    left = centers - step / 2.0
    right = centers + step / 2.0
    voxel_centers = torch.arange(native_length, device=device, dtype=dtype)
    voxel_left = voxel_centers - 0.5
    voxel_right = voxel_centers + 0.5
    return (
        torch.minimum(right[:, None], voxel_right[None, :])
        - torch.maximum(left[:, None], voxel_left[None, :])
    ).clamp_min_(0.0)


def _centered_kernel(step: tuple[int, int, int], *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, tuple[int, int, int]]:
    vectors: list[Tensor] = []
    padding: list[int] = []
    for value in step:
        pad = value // 2
        if value % 2 == 0:
            vector = torch.ones(value + 1, device=device, dtype=dtype)
            vector[0] = 0.5
            vector[-1] = 0.5
        else:
            vector = torch.ones(value, device=device, dtype=dtype)
        vectors.append(vector)
        padding.append(pad)
    kernel = (
        vectors[0][:, None, None]
        * vectors[1][None, :, None]
        * vectors[2][None, None, :]
    )
    return kernel.view(1, 1, *kernel.shape), tuple(padding)  # type: ignore[return-value]


def _pool_with_zero_offset(values: Tensor, mask: Tensor, step: tuple[int, int, int]) -> tuple[Tensor, Tensor]:
    kernel, padding = _centered_kernel(step, device=values.device, dtype=values.dtype)
    channels = values.shape[1]
    numerator = F.conv3d(
        values * mask,
        kernel.expand(channels, 1, *kernel.shape[-3:]),
        stride=step,
        padding=padding,
        groups=channels,
    )
    denominator = F.conv3d(mask, kernel, stride=step, padding=padding)
    return numerator / denominator.clamp_min(torch.finfo(values.dtype).eps), denominator


def _pool_with_fractional_offset(
    values: Tensor,
    mask: Tensor,
    native_shape: tuple[int, int, int],
    phase: GridPhase,
) -> tuple[Tensor, Tensor]:
    target_shape = _target_shape(native_shape, phase)
    numerator = values * mask
    denominator = mask
    step, offset = phase
    for axis, (length, count, axis_step, axis_offset) in enumerate(
        zip(native_shape, target_shape, step, offset),
    ):
        weights = _axis_overlap_weights(
            length, count, axis_step, axis_offset,
            device=values.device, dtype=values.dtype,
        )
        tensor_axis = axis + 2
        numerator = torch.matmul(numerator.movedim(tensor_axis, -1), weights.T).movedim(-1, tensor_axis)
        denominator = torch.matmul(denominator.movedim(tensor_axis, -1), weights.T).movedim(-1, tensor_axis)
    pooled = numerator / denominator.clamp_min(torch.finfo(values.dtype).eps)
    return pooled, denominator


def _native_mask(
    native_mask: Tensor | None,
    *,
    batch: int,
    native_shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    depth, height, width = native_shape
    if native_mask is None:
        return torch.ones((batch, 1, depth, height, width), device=device, dtype=dtype)
    if not isinstance(native_mask, Tensor) or native_mask.ndim not in (4, 5):
        raise ValueError("native_mask must have shape [B,D,H,W] or [B,1,D,H,W]")
    if native_mask.ndim == 4:
        native_mask = native_mask[:, None]
    if native_mask.shape[1] != 1 or any(
        native_mask.shape[axis + 2] < length for axis, length in enumerate(native_shape)
    ):
        raise ValueError("native_mask must cover native_shape with one channel")
    if native_mask.shape[0] not in (1, batch):
        raise ValueError("native_mask batch must be one or match values")
    mask = native_mask[..., :depth, :height, :width].to(device=device, dtype=dtype)
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1, -1, -1)
    if not torch.isfinite(mask).all() or torch.any(mask < 0) or torch.any(mask > 1):
        raise ValueError("native_mask values must be finite fractions in [0,1]")
    return mask.detach()


def pool_native_cells(
    values: Tensor,
    native_shape: Sequence[int],
    student_phase: GridPhase,
    native_mask: Tensor | None = None,
) -> Tensor:
    """Average native values over centered student cells using valid overlap.

    The input can include high-side padding and may have any positive channel
    count. A per-voxel support mask is applied before both numerator and
    denominator are pooled. For even strides, boundary voxel supports receive
    half weights when a cell edge cuts through their centers; this keeps the
    integration symmetric around each student sampling center.
    """

    shape = _native_shape(native_shape)
    phase = _phase(student_phase, name="student_phase")
    if not isinstance(values, Tensor) or values.ndim != 5 or values.shape[1] <= 0:
        raise ValueError("values must have shape [B,C,D,H,W] with C > 0")
    if any(values.shape[axis + 2] < length for axis, length in enumerate(shape)):
        raise ValueError("values do not cover native_shape")
    if not torch.is_floating_point(values) or not torch.isfinite(values).all():
        raise ValueError("values must contain finite floating-point values")

    work_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    cropped = values[..., :shape[0], :shape[1], :shape[2]].to(dtype=work_dtype)
    mask = _native_mask(
        native_mask,
        batch=values.shape[0],
        native_shape=shape,
        device=values.device,
        dtype=work_dtype,
    )
    step, offset = phase
    if all(value == 0.0 for value in offset):
        pooled, _support = _pool_with_zero_offset(cropped, mask, step)
    else:
        pooled, _support = _pool_with_fractional_offset(cropped, mask, shape, phase)
    return pooled.to(dtype=values.dtype)


def pool_teacher_targets(
    region_logits: Tensor,
    native_shape: Sequence[int],
    student_phase: GridPhase,
    native_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pool teacher class fractions and confidence on centered student cells.

    ``region_logits`` may include high-side producer padding. It is cropped to
    the declared native shape before softmax, and ``native_mask`` can further
    describe valid native support. Cell windows are centered at actual student
    grid centers; their voxel-support overlaps are used at even-stride edges.
    Padded logits never enter either numerator or denominator.

    Returns ``(class_probs, classes, confidence)`` with shapes ``[B,4,D,h,w]``,
    ``[B,D,h,w]``, and ``[B,D,h,w]`` respectively.
    """

    shape = _native_shape(native_shape)
    phase = _phase(student_phase, name="student_phase")
    if not isinstance(region_logits, Tensor) or region_logits.ndim != 5 \
            or region_logits.shape[1] != 4:
        raise ValueError("teacher region logits must have shape [B,4,D,H,W]")
    if not torch.is_floating_point(region_logits) or not torch.isfinite(region_logits).all():
        raise ValueError("teacher region logits must be finite floating-point values")
    if any(region_logits.shape[axis + 2] < length for axis, length in enumerate(shape)):
        raise ValueError("teacher region logits do not cover native_shape")

    logits = region_logits[..., :shape[0], :shape[1], :shape[2]].detach().float()
    probabilities = torch.softmax(logits, dim=1)
    confidence = probabilities.max(dim=1, keepdim=True).values
    pooled = pool_native_cells(
        torch.cat((probabilities, confidence), dim=1), shape, phase,
        native_mask=native_mask,
    )
    pooled_probs = pooled[:, :4].float()
    pooled_confidence = pooled[:, 4:5].float()
    pooled_probs = pooled_probs.detach()
    classes = pooled_probs.argmax(dim=1).detach()
    pooled_confidence = pooled_confidence[:, 0].detach()
    return pooled_probs, classes, pooled_confidence


__all__ = [
    "GridPhase",
    "align_teacher_to_student4",
    "grid_phase",
    "pool_native_cells",
    "pool_teacher_targets",
    "student_grid_phase",
    "teacher_grid_phase",
]
