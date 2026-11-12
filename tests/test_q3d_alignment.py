from __future__ import annotations

import math
import os
from collections.abc import Sequence

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from self_audit.candidate_a3d.teacher import (
    DualBranchTeacher3D,
    TinyEncoder3D,
    build_teacher3d,
)
from self_audit.candidate_q3d.alignment import (
    align_teacher_to_student4,
    grid_phase,
    pool_native_cells,
    pool_teacher_targets,
    student_grid_phase,
    teacher_grid_phase,
)
from self_audit.candidate_q3d.config import UPSTREAM_MEDNEXT_COMMIT
from self_audit.candidate_q3d.student import CandidateQ3DStudent
from self_audit.candidate_q3d.teacher_features import (
    _mednext_phase_chain,
    forward_teacher_with_features,
    teacher_feature_phase,
    teacher_feature_source,
)


def _student() -> CandidateQ3DStudent:
    return CandidateQ3DStudent({"arm": "C0"})


def _tiny_teacher() -> DualBranchTeacher3D:
    teacher = DualBranchTeacher3D(TinyEncoder3D(widths=(4, 8, 12, 16)))
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


def _box_probe(chain: Sequence[nn.Module], shape: tuple[int, int, int], axis: int) -> Tensor:
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
            value = F.conv3d(
                value, kernel, stride=stride, padding=padding, dilation=dilation,
            )
    return value


def _check_network_phase_ramps(
    chain: Sequence[nn.Module],
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
            offset[index]
            + torch.arange(values.shape[index + 2], dtype=torch.float64) * step[index]
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
        assert torch.max(torch.abs(observed - expected)).item() <= 0.5
        assert torch.max(torch.abs(observed - expected)).item() <= 1e-5


def test_student_and_tiny_teacher_phases_come_from_actual_modules() -> None:
    student = _student()
    student_chain = (
        student.stem[0].conv,
        student.stage1[0].conv,
        student.stage2[0].conv,
        student.stage3.down.conv,
        student.stage4.down.conv,
    )
    assert student_grid_phase(student) == ((1, 16, 16), (0.0, 0.0, 0.0))
    assert grid_phase(student_chain) == student_grid_phase(student)
    _check_network_phase_ramps(student_chain, student_grid_phase(student), (16, 161, 161))

    teacher = _tiny_teacher()
    tiny_chain = (
        teacher.encoder.stem[0],
        teacher.encoder.down[0][0],
        teacher.encoder.down[1][0],
        teacher.encoder.down[2][0],
    )
    assert teacher_feature_phase(teacher, "encoder_stride8", synthetic=True) == (
        (8, 8, 8), (0.0, 0.0, 0.0)
    )
    assert teacher_grid_phase(teacher, "stride8", synthetic=True) == (
        (8, 8, 8), (0.0, 0.0, 0.0)
    )
    _check_network_phase_ramps(tiny_chain, teacher_grid_phase(
        teacher, "encoder_stride8", synthetic=True
    ), (97, 97, 97))


def test_synthetic_bottleneck_phase_uses_the_attached_pool_module() -> None:
    teacher = _tiny_teacher()
    with pytest.raises(ValueError, match="actual AvgPool3d"):
        teacher_grid_phase(teacher, "bottleneck_stride16", synthetic=True)
    teacher._q3d_stride16_pool = nn.AvgPool3d(
        kernel_size=(1, 2, 2), stride=(1, 2, 2)
    )
    assert teacher_grid_phase(teacher, "bottleneck_stride16", synthetic=True) == (
        (8, 16, 16), (0.0, 4.0, 4.0)
    )
    chain = (
        teacher.encoder.stem[0],
        teacher.encoder.down[0][0],
        teacher.encoder.down[1][0],
        teacher.encoder.down[2][0],
        teacher._q3d_stride16_pool,
    )
    _check_network_phase_ramps(chain, teacher_grid_phase(
        teacher, "bottleneck_stride16", synthetic=True
    ), (97, 97, 97))


@pytest.mark.parametrize(
    ("source", "feature_index", "step"),
    [
        ("decoder_stride8", 3, (8, 8, 8)),
        ("stride4_decoder_block", 2, (4, 4, 4)),
    ],
)
def test_synthetic_secondary_teacher_sources_probe_their_actual_proxy_chain(
    source: str,
    feature_index: int,
    step: tuple[int, int, int],
) -> None:
    teacher = _tiny_teacher()
    details = teacher_feature_source(teacher, source, synthetic=True)
    assert details["synthetic_proxy"] is True
    assert details["feature_index"] == feature_index
    chain = (teacher.encoder.stem[0], *(
        teacher.encoder.down[index][0] for index in range(feature_index)
    ))
    phase = teacher_feature_phase(teacher, source, synthetic=True)
    assert phase == (step, (0.0, 0.0, 0.0))
    _check_network_phase_ramps(chain, phase, (97, 97, 97))


@pytest.mark.parametrize("depth", [6, 8])
def test_alignment_endpoint_replication_handles_shallow_teacher_depth(depth: int) -> None:
    teacher_phase = ((8, 8, 8), (0.0, 0.0, 0.0))
    student_phase = ((1, 16, 16), (0.0, 0.0, 0.0))
    source = torch.full((1, 1, 1, 3, 3), 2.5, dtype=torch.float64)
    aligned = align_teacher_to_student4(source, (depth, 17, 19), teacher_phase, student_phase)
    assert aligned.shape == (1, 1, depth, 2, 2)
    torch.testing.assert_close(aligned, torch.full_like(aligned, 2.5))


def test_alignment_samples_all_native_student_centers_with_precise_interior() -> None:
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
    expected = (expected_z + expected_y + expected_x).view(1, 1, *expected_z.shape[:1], 3, 3)
    torch.testing.assert_close(aligned[..., :25, :, :], expected[..., :25, :, :], atol=1e-5, rtol=0)
    assert not aligned.requires_grad


def test_pool_native_cells_uses_centered_overlap_and_excludes_padded_values() -> None:
    native_shape = (16, 65, 65)
    student_phase = ((1, 16, 16), (0.0, 0.0, 0.0))
    depth, height, width = native_shape
    z = torch.arange(depth, dtype=torch.float64).view(1, 1, depth, 1, 1)
    y = torch.arange(height, dtype=torch.float64).view(1, 1, 1, height, 1)
    x = torch.arange(width, dtype=torch.float64).view(1, 1, 1, 1, width)
    native = z + y + x
    pooled = pool_native_cells(native, native_shape, student_phase)
    expected = torch.tensor(7 + 32 + 32, dtype=torch.float64)
    torch.testing.assert_close(pooled[0, 0, 7, 2, 2], expected, atol=1e-5, rtol=0)

    padded = torch.full((1, 1, 17, 80, 80), 10000.0, dtype=torch.float64)
    padded[..., :depth, :height, :width] = native
    padded_pooled = pool_native_cells(padded, native_shape, student_phase)
    torch.testing.assert_close(padded_pooled, pooled)


def test_pool_teacher_targets_returns_native_fractions_class_and_mean_confidence() -> None:
    native_shape = (1, 33, 33)
    student_phase = ((1, 16, 16), (0.0, 0.0, 0.0))
    logits = torch.zeros((1, 4, 2, 48, 48), dtype=torch.float32)
    logits[:, 3] = 100.0  # high-side padding must be removed before softmax/pooling
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


def test_tiny_feature_capture_observes_forward_tuple_once_and_preserves_teacher_outputs() -> None:
    teacher = _tiny_teacher()
    image = torch.rand((1, 1, 6, 17, 19), dtype=torch.float32)
    geometry = {
        "shape": (6, 17, 19),
        "spacing_zyx": (5.0, 1.2, 1.2),
        "affine_zyx": torch.eye(4),
        "context": "full_volume",
    }
    with torch.no_grad():
        reference = teacher(image, geometry=geometry)
    calls = 0
    original = teacher.encoder.forward_features

    def counted(value: Tensor) -> tuple[Tensor, ...]:
        nonlocal calls
        calls += 1
        return original(value)

    teacher.encoder.forward_features = counted
    captured = forward_teacher_with_features(
        teacher, image, geometry=geometry, source="encoder_stride8", synthetic=True,
    )
    assert calls == 1
    assert set(captured) == {"region_logits", "boundary_logits", "sdf", "feature"}
    assert captured["feature"].shape == (1, 16, 1, 3, 3)
    assert not captured["feature"].requires_grad
    for key in ("region_logits", "boundary_logits", "sdf"):
        torch.testing.assert_close(captured[key], reference[key])


@pytest.mark.skipif(
    not os.environ.get("Q3D_OFFICIAL_MEDNEXT_SOURCE"),
    reason="set Q3D_OFFICIAL_MEDNEXT_SOURCE to the clean pinned MedNeXt checkout",
)
def test_actual_pinned_mednext_feature_phases_and_source_modules() -> None:
    source_repo = os.environ["Q3D_OFFICIAL_MEDNEXT_SOURCE"]
    teacher, _provenance = build_teacher3d(
        {
            "backbone": "mednext_l_k3",
            "input_channels": 1,
            "source_repo": source_repo,
            "source_commit": UPSTREAM_MEDNEXT_COMMIT,
            "weights": None,
            "weights_sha256": None,
        },
        synthetic=False,
    )
    assert teacher.encoder.source_commit == "0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba"
    assert teacher_feature_phase(teacher, "encoder_stride8") == (
        (8, 8, 8), (0.0, 0.0, 0.0)
    )
    main_chain = _mednext_phase_chain(teacher.encoder.model, "encoder_stride8")
    sampling_chain = tuple(
        module for index, module in enumerate(main_chain)
        if index == 0 or module.stride != (1, 1, 1)
    )
    _check_network_phase_ramps(
        sampling_chain,
        teacher_feature_phase(teacher, "encoder_stride8"),
        (33, 33, 33),
    )
    main = teacher_feature_source(teacher, "encoder_stride8")
    assert main["feature_index"] == 3
    assert main["capture_api"] == "forward_with_features"
    assert main["module"] is None
    decoder = teacher_feature_source(teacher, "decoder_stride8")
    assert decoder["module_name"] == "dec_block_3.7"
    assert decoder["capture_api"] == "module_hook"
    assert decoder["channels"] == 256
    with pytest.raises(ValueError, match="upsample array-index lattice.*encoder skip lattice"):
        teacher_feature_phase(teacher, "decoder_stride8")
    with pytest.raises(ValueError, match="upsample array-index lattice.*encoder skip lattice"):
        teacher_feature_phase(teacher, "stride4_decoder_block")
    assert teacher_feature_phase(teacher, "bottleneck_stride16") == (
        (16, 16, 16), (0.0, 0.0, 0.0)
    )

    teacher.requires_grad_(False)
    teacher.eval()
    image = torch.rand((1, 1, 32, 32, 32))
    geometry = {
        "shape": (32, 32, 32),
        "spacing_zyx": (1.0, 1.0, 1.0),
        "affine_zyx": torch.eye(4),
        "context": "full_volume",
    }
    forward_calls = 0
    original_forward_with_features = teacher.encoder.forward_with_features

    def counted_forward_with_features(value: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        nonlocal forward_calls
        forward_calls += 1
        return original_forward_with_features(value)

    teacher.encoder.forward_with_features = counted_forward_with_features
    decoder_output: list[Tensor] = []
    decoder_handle = decoder["module"].register_forward_hook(
        lambda _module, _inputs, output: decoder_output.append(output.detach())
    )
    try:
        captured = forward_teacher_with_features(
            teacher, image, geometry=geometry, source="encoder_stride8",
        )
    finally:
        decoder_handle.remove()
    assert forward_calls == 1
    assert captured["feature"].shape == (1, 256, 4, 4, 4)
    assert len(decoder_output) == 1
    assert decoder_output[0].shape == (1, 256, 4, 4, 4)
