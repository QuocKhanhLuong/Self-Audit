from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from self_audit.candidate_q3d.alignment import (
    align_teacher_to_student4,
    grid_phase,
    pool_native_cells,
    pool_teacher_targets,
    student_grid_phase,
    teacher_grid_phase,
)
from self_audit.candidate_q3d.student import CandidateQ3DStudent
from self_audit.candidate_q3d.teacher_features import (
    teacher_feature_phase,
    teacher_feature_source,
)
from self_audit.candidate_q3d.teacher_resenc import Q3DResEncTeacher


def _student() -> CandidateQ3DStudent:
    return CandidateQ3DStudent({"arm": "C0"})


def _synthetic_teacher(
    *,
    strides: tuple[tuple[int, int, int], ...] | None = None,
    phase_offsets: dict[str, tuple[float, float, float]] | None = None,
) -> Q3DResEncTeacher:
    teacher = Q3DResEncTeacher(
        synthetic=True,
        strides=strides,
        phase_offsets=phase_offsets,
    )
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def _conv_params(module: nn.Module) -> tuple[tuple[int, int, int], ...]:
    if isinstance(module, nn.Conv3d):
        return (
            tuple(module.kernel_size), tuple(module.stride), tuple(module.padding),
            tuple(module.dilation),
        )
    if isinstance(module, nn.AvgPool3d):
        kernel = module.kernel_size
        stride = module.stride or kernel
        kernel = (kernel,) * 3 if isinstance(kernel, int) else tuple(kernel)
        stride = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
        padding = module.padding
        padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
        return kernel, stride, padding, (1, 1, 1)
    raise TypeError(type(module))


def _box_probe(chain: tuple[nn.Module, ...], shape: tuple[int, int, int], axis: int) -> Tensor:
    coordinate = torch.arange(shape[axis], dtype=torch.float64)
    view = [1, 1, 1, 1, 1]
    view[axis + 2] = shape[axis]
    value = coordinate.view(view).expand(1, 1, *shape).contiguous()
    for module in chain:
        kernel_size, stride, padding, dilation = _conv_params(module)
        if isinstance(module, nn.AvgPool3d):
            value = F.avg_pool3d(
                value, kernel_size=kernel_size, stride=stride, padding=padding,
                ceil_mode=module.ceil_mode, count_include_pad=module.count_include_pad,
            )
        else:
            kernel = torch.ones((1, 1, *kernel_size), dtype=value.dtype)
            kernel /= math.prod(kernel_size)
            value = F.conv3d(value, kernel, stride=stride, padding=padding, dilation=dilation)
    return value


def _check_network_phase_ramps(
    chain: tuple[nn.Module, ...],
    phase: tuple[tuple[int, int, int], tuple[float, float, float]],
    shape: tuple[int, int, int],
) -> None:
    step, offset = phase
    receptive_radius = [0.0, 0.0, 0.0]
    jump = [1, 1, 1]
    for module in chain:
        kernel, stride, _padding, dilation = _conv_params(module)
        for axis in range(3):
            receptive_radius[axis] += dilation[axis] * (kernel[axis] - 1) / 2 * jump[axis]
            jump[axis] *= stride[axis]

    for axis in range(3):
        values = _box_probe(chain, shape, axis)
        centers_by_axis = [
            offset[index] + torch.arange(values.shape[index + 2], dtype=torch.float64) * step[index]
            for index in range(3)
        ]
        valid_by_axis = [
            (centers_by_axis[index] >= receptive_radius[index])
            & (centers_by_axis[index] <= shape[index] - 1 - receptive_radius[index])
            for index in range(3)
        ]
        assert all(mask.any() for mask in valid_by_axis), "no interior probe points"
        observed = values
        for selected_axis, valid in enumerate(valid_by_axis):
            observed = observed.index_select(selected_axis + 2, valid.nonzero().flatten())
        expected = centers_by_axis[axis][valid_by_axis[axis]]
        expected_shape = [1, 1, 1, 1, 1]
        expected_shape[axis + 2] = expected.numel()
        expected = expected.view(expected_shape).expand_as(observed)
        assert torch.max(torch.abs(observed - expected)).item() <= 1e-5


def test_student_phase_is_derived_from_live_sampling_modules() -> None:
    student = _student()
    chain = (
        student.stem[0].conv,
        student.stage1[0].conv,
        student.stage2[0].conv,
        student.stage3.down.conv,
        student.stage4.down.conv,
    )
    phase = student_grid_phase(student)
    assert phase == ((1, 16, 16), (0.0, 0.0, 0.0))
    assert grid_phase(chain) == phase
    _check_network_phase_ramps(chain, phase, (16, 161, 161))


def test_resenc_phase_uses_configured_stride_and_measured_offset() -> None:
    strides = (
        (1, 1, 1), (2, 2, 1), (2, 2, 2),
        (2, 1, 2), (2, 2, 2), (1, 2, 2),
    )
    offsets = {
        "encoder_stride8": (0.5, -1.0, 2.0),
        "stride16": (1.5, 0.0, -2.0),
    }
    teacher = _synthetic_teacher(strides=strides, phase_offsets=offsets)
    assert teacher_feature_phase(teacher, "encoder_stride8") == (
        (8, 4, 4), offsets["encoder_stride8"],
    )
    assert teacher_grid_phase(teacher, "stride16") == (
        (16, 8, 8), offsets["stride16"],
    )
    assert teacher_feature_source(teacher, "stride8")["module_name"] == "encoder.stages.3"


def test_resenc_phase_offsets_fail_closed() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _synthetic_teacher(phase_offsets={"encoder_stride8": (0.0, float("nan"), 0.0)})
    with pytest.raises(ValueError, match="unsupported"):
        _synthetic_teacher(phase_offsets={"unknown_source": (0.0, 0.0, 0.0)})


def test_alignment_endpoint_replication_handles_shallow_teacher_depth() -> None:
    teacher_phase = ((8, 8, 8), (0.0, 0.0, 0.0))
    student_phase = ((1, 16, 16), (0.0, 0.0, 0.0))
    source = torch.full((1, 1, 1, 3, 3), 2.5, dtype=torch.float64)
    aligned = align_teacher_to_student4(source, (6, 17, 19), teacher_phase, student_phase)
    assert aligned.shape == (1, 1, 6, 2, 2)
    torch.testing.assert_close(aligned, torch.full_like(aligned, 2.5))


def test_alignment_samples_all_native_student_centers() -> None:
    native_shape = (32, 33, 41)
    teacher_phase = ((8, 8, 8), (0.0, 0.0, 0.0))
    student_phase = ((1, 16, 16), (0.0, 0.0, 0.0))
    source_shape = tuple(math.ceil(length / stride) for length, stride in zip(native_shape, teacher_phase[0]))
    z = torch.arange(source_shape[0], dtype=torch.float64) * 8
    y = torch.arange(source_shape[1], dtype=torch.float64) * 8
    x = torch.arange(source_shape[2], dtype=torch.float64) * 8
    source = (z[:, None, None] + y[None, :, None] + x[None, None, :])[None, None]
    aligned = align_teacher_to_student4(source, native_shape, teacher_phase, student_phase)
    expected_z = torch.arange(native_shape[0], dtype=torch.float64).view(-1, 1, 1)
    expected_y = (torch.arange(3, dtype=torch.float64) * 16).view(1, -1, 1)
    expected_x = (torch.arange(3, dtype=torch.float64) * 16).view(1, 1, -1)
    expected = (expected_z + expected_y + expected_x).view(1, 1, native_shape[0], 3, 3)
    torch.testing.assert_close(aligned[..., :25, :, :], expected[..., :25, :, :], atol=1e-5, rtol=0)
    assert not aligned.requires_grad


def test_pool_native_cells_uses_centered_overlap_and_excludes_padding() -> None:
    native_shape = (16, 65, 65)
    student_phase = ((1, 16, 16), (0.0, 0.0, 0.0))
    depth, height, width = native_shape
    z = torch.arange(depth, dtype=torch.float64).view(1, 1, depth, 1, 1)
    y = torch.arange(height, dtype=torch.float64).view(1, 1, 1, height, 1)
    x = torch.arange(width, dtype=torch.float64).view(1, 1, 1, 1, width)
    native = z + y + x
    pooled = pool_native_cells(native, native_shape, student_phase)
    torch.testing.assert_close(pooled[0, 0, 7, 2, 2], torch.tensor(71.0, dtype=torch.float64))
    padded = torch.full((1, 1, 17, 80, 80), 10000.0, dtype=torch.float64)
    padded[..., :depth, :height, :width] = native
    torch.testing.assert_close(pool_native_cells(padded, native_shape, student_phase), pooled)


def test_pool_teacher_targets_returns_fractions_classes_and_confidence() -> None:
    native_shape = (1, 33, 33)
    student_phase = ((1, 16, 16), (0.0, 0.0, 0.0))
    logits = torch.zeros((1, 4, 2, 48, 48), dtype=torch.float32)
    logits[:, 3] = 100.0  # high-side padding must be removed before pooling
    logits[:, :, :1, :33, :33] = 0.0
    logits[:, 0, :, :, :16] = 9.0
    logits[:, 1, :, :, 16:33] = 9.0
    probs, classes, confidence = pool_teacher_targets(logits, native_shape, student_phase)
    assert probs.shape == (1, 4, 1, 3, 3)
    assert classes.shape == confidence.shape == (1, 1, 3, 3)
    assert classes[0, 0, 1, 1].item() in (0, 1)
    assert probs[0, 0, 0, 1, 1] > 0.4
    assert probs[0, 1, 0, 1, 1] > 0.4
    assert confidence[0, 0, 1, 1] > 0.99


def test_resenc_feature_source_is_encoder_grid_and_requires_no_hooks() -> None:
    teacher = _synthetic_teacher()
    details = teacher_feature_source(teacher, "encoder_stride8", synthetic=True)
    assert details["feature_index"] == 3
    assert details["module_name"] == "encoder.stages.3"
    assert details["capture_api"] == "forward_with_features"
    assert details["module"] is None
    assert details["synthetic_proxy"] is True


def test_resenc_feature_phase_rejects_removed_sources() -> None:
    teacher = _synthetic_teacher()
    for source in ("unknown_source", "decoder", "feature_4"):
        with pytest.raises(ValueError, match="encoder_stride8 or encoder_stride16"):
            teacher_feature_phase(teacher, source)
