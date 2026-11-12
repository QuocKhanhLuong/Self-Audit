"""Synthetic effective-receptive-field diagnostics for the ResEnc teacher.

The probe is deliberately separate from the training engine.  It differentiates
one encoder stage-3 cell with respect to a random, volume-wise Z-scored input;
it never reads patient data and never uses decoder or segmentation heads.  The
nominal comparison phase is the theoretical main convolution path (zero
offset for symmetric 3x3/p1 convolutions), even when a teacher carries a
configured empirical correction.  That keeps the measurement diagnostic
instead of circularly agreeing with the correction it is meant to check.
"""

from __future__ import annotations

import importlib.metadata
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from numbers import Integral
from typing import Any

import torch
from torch import Tensor, nn

from .teacher_resenc import Q3DResEncTeacher, cumulative_strides

_STAGE_INDEX = 3
_MIN_SAMPLES = 16
_OFFSET_THRESHOLD = 2.0


def _positive_shape(value: Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError("input_shape must contain positive D,H,W")
    if any(not isinstance(axis, Integral) or isinstance(axis, bool) for axis in value):
        raise ValueError("input_shape must contain positive D,H,W")
    shape = tuple(int(axis) for axis in value)
    if any(axis <= 0 for axis in shape):
        raise ValueError("input_shape must contain positive D,H,W")
    return shape


def _teacher_device(teacher: nn.Module) -> torch.device:
    try:
        return next(teacher.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@contextmanager
def _frozen_eval(module: nn.Module) -> Iterator[None]:
    training = module.training
    requires_grad = [(parameter, parameter.requires_grad) for parameter in module.parameters()]
    module.eval()
    for parameter, _was_enabled in requires_grad:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, was_enabled in requires_grad:
            parameter.requires_grad_(was_enabled)
        module.train(training)


def _encoder_prefix(teacher: Q3DResEncTeacher, image: Tensor, stage_index: int) -> Tensor:
    encoder = teacher.encoder
    stem = getattr(encoder, "stem", None)
    stages = getattr(encoder, "stages", None)
    if not isinstance(stem, nn.Module) or not isinstance(stages, nn.Module):
        raise TypeError("ResEnc teacher encoder must expose stem and stages")
    stage_modules = tuple(stages.children())
    if stage_index >= len(stage_modules):
        raise ValueError("requested ERF stage is not present in the encoder")
    feature = stem(image)
    for stage in stage_modules[: stage_index + 1]:
        feature = stage(feature)
    return feature


def _nominal_stage_stride(teacher: Q3DResEncTeacher, stage_index: int) -> tuple[int, int, int]:
    strides = tuple(tuple(int(axis) for axis in stride) for stride in teacher.encoder_strides)
    if len(strides) <= stage_index:
        raise ValueError("ResEnc teacher does not expose enough stage strides")
    return cumulative_strides(strides[: stage_index + 1])


def _sample(
    teacher: Q3DResEncTeacher,
    *,
    input_shape: tuple[int, int, int],
    seed: int,
    stage_index: int,
    nominal_stride: tuple[int, int, int],
    device: torch.device,
) -> dict[str, Any]:
    # Generate on CPU so a fixed seed has the same input contract across CPU
    # and CUDA probes, then move the already-Z-scored leaf to the teacher.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    raw = torch.randn((1, 1, *input_shape), generator=generator, dtype=torch.float32)
    raw = raw.to(device=device)
    mean = raw.mean()
    std = raw.std(unbiased=False).clamp_min(1e-6)
    image = ((raw - mean) / std).detach().requires_grad_(True)
    padded, padding_after = teacher._pad_high(image)
    feature = _encoder_prefix(teacher, padded, stage_index)
    logical_feature_shape = tuple(
        math.ceil(size / stride) for size, stride in zip(input_shape, nominal_stride)
    )
    cell = tuple(size // 2 for size in logical_feature_shape)
    if any(index >= int(feature.shape[axis + 2]) for axis, index in enumerate(cell)):
        raise RuntimeError(
            "ERF logical center is outside the measured encoder feature grid: "
            f"cell={cell}, feature_shape={tuple(feature.shape)}"
        )
    channel_count = int(feature.shape[1])
    if channel_count < 16:
        raise ValueError("ERF stage-3 feature must expose at least 16 channels")
    scalar = feature[(0, slice(None), *cell)].mean()
    gradient = torch.autograd.grad(scalar, image, retain_graph=False, create_graph=False)[0]
    mass = gradient.detach().abs()[0, 0]
    mass_total = mass.sum()
    if not torch.isfinite(mass).all() or not torch.isfinite(mass_total) or float(mass_total) <= 0:
        raise RuntimeError("ERF gradient mass is non-finite or zero")
    coordinate_axes = [
        torch.arange(size, dtype=torch.float64, device=device) for size in input_shape
    ]
    coordinates = torch.stack(torch.meshgrid(*coordinate_axes, indexing="ij"), dim=0)
    mass64 = mass.to(dtype=torch.float64)
    centroid = (coordinates * mass64.unsqueeze(0)).flatten(1).sum(dim=1) / mass64.sum()
    phase_center = torch.tensor(
        [stride * index for stride, index in zip(nominal_stride, cell)],
        dtype=torch.float64,
        device=device,
    )
    offset = centroid - phase_center
    return {
        "seed": int(seed),
        "input_shape_zyx": list(input_shape),
        "model_input_shape_zyx": list(padded.shape[-3:]),
        "padding_after_zyx": list(padding_after),
        "feature_shape": list(feature.shape),
        "logical_feature_shape": list(logical_feature_shape),
        "cell_index_zyx": list(cell),
        "channel_aggregation": "signed_mean_all_stage_channels",
        "signed_mean_cell_activation": float(scalar.detach().item()),
        "channel_count": channel_count,
        "nominal_main_phase_center_voxels": [float(value) for value in phase_center.cpu()],
        "gradient_centroid_voxels": [float(value) for value in centroid.cpu()],
        "centroid_offset_from_nominal_main_phase_voxels": [
            float(value) for value in offset.cpu()
        ],
        "max_abs_offset_voxels": float(offset.abs().max().item()),
        "gradient_mass": float(mass_total.item()),
    }


def run_resenc_erf(
    teacher: Q3DResEncTeacher,
    *,
    samples: int = _MIN_SAMPLES,
    seed: int = 42,
    input_shape: Sequence[int] = (33, 97, 97),
) -> dict[str, Any]:
    """Measure the stage-3 gradient ERF against the theoretical main phase.

    ``teacher`` must be a ResEnc teacher with a six-stage encoder.  The probe
    uses only ``encoder.stem`` and ``encoder.stages[:4]``; its decoder,
    boundary path, labels, and configured phase offsets are not consulted for
    the gradient.  A real ResEnc-L has 256 channels at stage 3; synthetic
    fixtures are accepted when they expose at least 16 channels.
    """

    if not isinstance(teacher, Q3DResEncTeacher):
        raise TypeError("run_resenc_erf requires Q3DResEncTeacher")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < _MIN_SAMPLES:
        raise ValueError(f"ERF requires at least {_MIN_SAMPLES} samples")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("ERF seed must be a nonnegative integer")
    shape = _positive_shape(input_shape)
    nominal_stride = _nominal_stage_stride(teacher, _STAGE_INDEX)
    device = _teacher_device(teacher)
    configured_phase = None
    try:
        configured_phase = teacher.phase_for_source("encoder_stride8")
    except (AttributeError, ValueError):
        pass
    rows: list[dict[str, Any]] = []
    with _frozen_eval(teacher):
        for index in range(samples):
            rows.append(
                _sample(
                    teacher,
                    input_shape=shape,
                    seed=seed + index,
                    stage_index=_STAGE_INDEX,
                    nominal_stride=nominal_stride,
                    device=device,
                )
            )
    offsets = torch.tensor([
        row["centroid_offset_from_nominal_main_phase_voxels"] for row in rows
    ], dtype=torch.float64)
    mean_offset = offsets.mean(dim=0)
    absolute_mean_offset = mean_offset.abs()
    status = (
        "PASS_MAIN_PHASE"
        if bool(torch.all(absolute_mean_offset <= _OFFSET_THRESHOLD))
        else "MEASURED_OFFSET_REQUIRED"
    )
    proposed_offset = torch.zeros(3, dtype=torch.float64) if status == "PASS_MAIN_PHASE" else mean_offset
    dynamic_version = None
    try:
        dynamic_version = importlib.metadata.version("dynamic-network-architectures")
    except importlib.metadata.PackageNotFoundError:
        dynamic_version = None
    return {
        "schema": "self_audit.q3d_resenc.erf_probe.v1",
        "status": status,
        "decision_rule": {
            "metric": "absolute mean centroid offset per axis from theoretical main phase",
            "threshold_voxels": _OFFSET_THRESHOLD,
            "gate": "all(abs(mean_offset_axis)) <= threshold",
        },
        "compute": {
            "device": str(device),
            "samples": int(samples),
            "seed_start": int(seed),
            "random_zscore_input": True,
            "patient_data": "NOT_USED",
            "padding_mode": "replicate",
            "input_shape_zyx": list(shape),
            "torch_version": torch.__version__,
            "dynamic_network_architectures_version": dynamic_version,
        },
        "model": {
            "architecture": "ResEncL" if tuple(teacher.encoder_channels) == (32, 64, 128, 256, 320, 320) else "ResEnc_fixture",
            "stage_index": _STAGE_INDEX,
            "stage_name": "encoder.stages[3]",
            "stage_channels": int(teacher.encoder_channels[_STAGE_INDEX]),
            "stride_schedule_zyx": [list(value) for value in teacher.encoder_strides],
            "cumulative_stride_zyx": list(nominal_stride),
            "phase_reference": "theoretical main convolution path; configured phase correction is reported but not used",
            "configured_phase": configured_phase,
        },
        "summary": {
            "mean_centroid_offset_voxels": [float(value) for value in mean_offset],
            "absolute_mean_centroid_offset_voxels": [float(value) for value in absolute_mean_offset],
            "mean_abs_centroid_offset_voxels": [float(value) for value in offsets.abs().mean(dim=0)],
            "max_abs_centroid_offset_voxels": float(offsets.abs().max().item()),
            "proposed_phase_offset_zyx": [float(value) for value in proposed_offset],
        },
        "proposed_offset_zyx": [float(value) for value in proposed_offset],
        "selected_phase": {
            "step_zyx": list(nominal_stride),
            "offset_zyx": [float(value) for value in proposed_offset],
        },
        "samples": rows,
        "limitations": [
            "Random Z-score inputs are architecture evidence only and contain no patient data.",
            "The probe measures encoder stage 3 only; it does not measure decoder or patient-dependent fields.",
            "The logical input shape is a bounded synthetic probe and is not a full-volume efficacy claim.",
        ],
    }


__all__ = ["run_resenc_erf"]
