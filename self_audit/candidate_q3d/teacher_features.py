"""One-pass Q-3D teacher feature capture and actual feature-grid phase."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from ..candidate_a3d.teacher import (
    UPSTREAM_MAIN_SHA,
    DualBranchTeacher3D,
    MedNeXtL3DAdapter,
    TinyEncoder3D,
)
from .alignment import GridPhase, grid_phase


_SOURCE_ALIASES = {
    "encoder_stride8": "encoder_stride8",
    "stride8": "encoder_stride8",
    "decoder_stride8": "decoder_stride8",
    "stride4_decoder_block": "stride4_decoder_block",
    "stride4": "stride4_decoder_block",
    "bottleneck_stride16": "bottleneck_stride16",
    "stride16": "bottleneck_stride16",
}
_MEDNEXT_SOURCES = {
    "encoder_stride8": ("enc_block_3", 3),
    "decoder_stride8": ("dec_block_3", 3),
    "stride4_decoder_block": ("dec_block_2", 2),
    "bottleneck_stride16": ("bottleneck", 4),
}


def _canonical_source(source: str) -> str:
    if not isinstance(source, str) or source not in _SOURCE_ALIASES:
        raise ValueError(
            "teacher feature source must be encoder_stride8, decoder_stride8, "
            "stride4_decoder_block, or bottleneck_stride16"
        )
    return _SOURCE_ALIASES[source]


def _capture_module(module: nn.Module, module_name: str, *, checkpointed: bool) -> tuple[nn.Module, str]:
    if not checkpointed:
        return module, module_name
    children = tuple(module.children())
    if not children:
        raise ValueError(f"checkpointed MedNeXt module {module_name!r} has no children to observe")
    # The pinned iterative_checkpoint method iterates over Sequential children,
    # so the container's forward hook is bypassed. The final child output is
    # exactly the completed block output and executes once in the frozen pass.
    return children[-1], f"{module_name}.{len(children) - 1}"


def teacher_feature_source(
    teacher: DualBranchTeacher3D,
    source: str = "encoder_stride8",
    *,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Return the actual source module and channel width for one Q-3D feature."""

    if not isinstance(teacher, DualBranchTeacher3D):
        raise TypeError("teacher must be the existing DualBranchTeacher3D")
    canonical = _canonical_source(source)
    encoder = teacher.encoder

    if isinstance(encoder, TinyEncoder3D):
        if not synthetic:
            raise ValueError("tiny_test3d teacher features are permitted only in synthetic runs")
        tiny_sources = {
            "encoder_stride8": (3, 3),
            "decoder_stride8": (3, 3),
            "stride4_decoder_block": (2, 2),
            "bottleneck_stride16": (3, 3),
        }
        if canonical not in tiny_sources:
            raise ValueError(f"tiny synthetic teacher has no {canonical} feature source")
        feature_index, channel_index = tiny_sources[canonical]
        pool = None
        if canonical == "bottleneck_stride16":
            pool = getattr(teacher, "_q3d_stride16_pool", None)
            if not isinstance(pool, nn.AvgPool3d):
                raise ValueError(
                    "synthetic bottleneck_stride16 requires the engine's actual AvgPool3d module"
                )
        channels = int(encoder.channels[channel_index])
        return {
            "source": canonical,
            "module_name": f"forward_features[{feature_index}]",
            "channels": channels,
            "module": None,
            "features_api": "tiny_encoder_forward_features",
            "feature_index": feature_index,
            "capture_api": "forward_features",
            "synthetic_proxy": canonical != "encoder_stride8",
            "pool": pool,
            "input_padding": "high_side_replicate_no_phase_shift",
        }

    if not isinstance(encoder, MedNeXtL3DAdapter):
        raise ValueError("Q-3D feature capture requires the verified MedNeXt-L adapter")
    model = encoder.model
    if not getattr(model, "_candidate_a_native_v1", False):
        raise ValueError("MedNeXt native v1 module sequence has not been validated")
    module_name, channel_index = _MEDNEXT_SOURCES[canonical]
    module = getattr(model, module_name, None)
    if not isinstance(module, nn.Module):
        raise ValueError(f"validated MedNeXt module {module_name!r} is unavailable")
    if canonical == "encoder_stride8":
        # The adapter already returns x_res_3 as features[3] from its one
        # fused encoder/region forward. Observe that tuple without a hook.
        capture_module, capture_name = None, "forward_with_features[0][3]"
        capture_api, feature_index = "forward_with_features", 3
    else:
        checkpointed = bool(getattr(model, "outside_block_checkpointing", False))
        capture_module, capture_name = _capture_module(
            module, module_name, checkpointed=checkpointed,
        )
        capture_api, feature_index = "module_hook", None
    if channel_index >= len(encoder.channels):
        raise ValueError("MedNeXt adapter.channels does not cover the selected feature source")
    channels = int(encoder.channels[channel_index])
    if channels <= 0:
        raise ValueError("selected MedNeXt feature channel count must be positive")
    return {
        "source": canonical,
        "module_name": capture_name,
        "channels": channels,
        "module": capture_module,
        "features_api": encoder.features_api,
        "feature_index": feature_index,
        "capture_api": capture_api,
        "synthetic_proxy": False,
        "pool": None,
        "input_padding": "high_side_replicate_no_phase_shift",
    }


def _require_conv(module: Any, label: str) -> nn.Conv3d:
    if not isinstance(module, nn.Conv3d):
        raise ValueError(f"verified MedNeXt {label} must be an actual Conv3d module")
    return module


def _mednext_block_main_convs(block: nn.Module, label: str) -> tuple[nn.Conv3d, ...]:
    """Return conv1/2/3 on the pinned MedNeXt block's main forward path."""

    blocks = tuple(block.children()) if isinstance(block, nn.Sequential) else (block,)
    if not blocks:
        raise ValueError(f"verified MedNeXt {label} cannot be empty")
    result: list[nn.Conv3d] = []
    for index, item in enumerate(blocks):
        for attribute in ("conv1", "conv2", "conv3"):
            result.append(_require_conv(
                getattr(item, attribute, None), f"{label}.{index}.{attribute}"
            ))
    return tuple(result)


def _transpose_phase(module: nn.ConvTranspose3d, phase: GridPhase) -> GridPhase:
    """Derive a transposed convolution's output lattice from its live module."""

    step, offset = phase
    output_step: list[int] = []
    output_offset: list[float] = []
    for axis, (kernel, stride, padding, dilation) in enumerate(zip(
        module.kernel_size, module.stride, module.padding, module.dilation,
    )):
        if step[axis] % stride:
            raise ValueError("MedNeXt upsample stride does not divide its input grid step")
        axis_step = step[axis] // stride
        output_step.append(axis_step)
        output_offset.append(
            offset[axis]
            + (padding - dilation * (kernel - 1) / 2.0) * axis_step
        )
    return tuple(output_step), tuple(output_offset)  # type: ignore[return-value]


def _mednext_up_phase(model: nn.Module, up_name: str, input_phase: GridPhase) -> GridPhase:
    """Measure the pinned UpBlock lattice, including its explicit low pad."""

    up = getattr(model, up_name, None)
    if not isinstance(up, nn.Module):
        raise ValueError(f"verified MedNeXt {up_name} is unavailable")
    main = getattr(up, "conv1", None)
    if not isinstance(main, nn.ConvTranspose3d):
        raise ValueError(f"verified MedNeXt {up_name}.conv1 must be ConvTranspose3d")
    branch_phase = _transpose_phase(main, input_phase)
    residual = getattr(up, "res_conv", None)
    if residual is not None:
        if not isinstance(residual, nn.ConvTranspose3d):
            raise ValueError(f"verified MedNeXt {up_name}.res_conv must be ConvTranspose3d")
        if _transpose_phase(residual, input_phase) != branch_phase:
            raise ValueError(f"MedNeXt {up_name} residual upsample phase differs from main branch")

    # The pinned source applies F.pad(x, (1,0,1,0,1,0)) after ConvTranspose3d:
    # one sample is inserted at the low side of W, H, and D.
    step, offset = branch_phase
    return step, tuple(value - jump for value, jump in zip(offset, step))


def _mednext_phase_chain(model: nn.Module, source: str) -> tuple[nn.Conv3d, ...]:
    """Build the executed encoder path that defines the selected feature grid."""

    if source == "encoder_stride8":
        down_count, final_encoder_stage = 3, 3
    elif source == "decoder_stride8":
        down_count, final_encoder_stage = 3, 3
    elif source == "stride4_decoder_block":
        down_count, final_encoder_stage = 2, 2
    elif source == "bottleneck_stride16":
        down_count, final_encoder_stage = 4, None
    else:  # pragma: no cover - canonicalization rejects unknown sources
        raise ValueError(f"unsupported MedNeXt source {source!r}")

    chain: list[nn.Conv3d] = [_require_conv(getattr(model, "stem", None), "stem")]
    for index in range(down_count):
        encoder_block = getattr(model, f"enc_block_{index}", None)
        if not isinstance(encoder_block, nn.Module):
            raise ValueError(f"verified MedNeXt enc_block_{index} is unavailable")
        chain.extend(_mednext_block_main_convs(encoder_block, f"enc_block_{index}"))

        down_block = getattr(model, f"down_{index}", None)
        if not isinstance(down_block, nn.Module):
            raise ValueError(f"verified MedNeXt down_{index} is unavailable")
        prefix = tuple(chain)
        main_convs = _mednext_block_main_convs(down_block, f"down_{index}")
        chain.extend(main_convs)

        # MedNeXt-L k3 is built with do_res_up_down=True. The residual
        # res_conv is parallel to the depthwise stride-2 conv, so validate its
        # measured phase against the full main path rather than counting both.
        residual = getattr(down_block, "res_conv", None)
        if residual is not None:
            residual_conv = _require_conv(residual, f"down_{index}.res_conv")
            if grid_phase((*prefix, residual_conv)) != grid_phase(tuple(chain)):
                raise ValueError(f"MedNeXt down_{index} residual branch phase does not match main path")
        elif bool(getattr(down_block, "resample_do_res", False)):
            raise ValueError(f"MedNeXt down_{index} declares a missing residual sampling convolution")

    if final_encoder_stage is not None:
        encoder_block = getattr(model, f"enc_block_{final_encoder_stage}", None)
        if not isinstance(encoder_block, nn.Module):
            raise ValueError(f"verified MedNeXt enc_block_{final_encoder_stage} is unavailable")
        chain.extend(_mednext_block_main_convs(
            encoder_block, f"enc_block_{final_encoder_stage}"
        ))

    if source in {"decoder_stride8", "stride4_decoder_block"}:
        target_stage = 3 if source == "decoder_stride8" else 2
        target_skip_phase = grid_phase(tuple(chain))

        # Build the actual deepest skip path even for a stride-4 source, whose
        # encoder prefix ends at stage 2. The decoder reaches stage 2 through
        # the stage-3 upsample and addition first.
        skip3_chain = list(chain)
        if final_encoder_stage != 3:
            down2 = getattr(model, "down_2", None)
            encoder3 = getattr(model, "enc_block_3", None)
            if not isinstance(down2, nn.Module) or not isinstance(encoder3, nn.Module):
                raise ValueError("verified MedNeXt down_2 and enc_block_3 are required")
            down2_prefix = tuple(skip3_chain)
            skip3_chain.extend(_mednext_block_main_convs(down2, "down_2"))
            residual_down2 = getattr(down2, "res_conv", None)
            if residual_down2 is not None:
                residual_conv = _require_conv(residual_down2, "down_2.res_conv")
                if grid_phase((*down2_prefix, residual_conv)) != grid_phase(tuple(skip3_chain)):
                    raise ValueError("MedNeXt down_2 residual branch phase does not match main path")
            skip3_chain.extend(_mednext_block_main_convs(encoder3, "enc_block_3"))
        skip3_phase = grid_phase(tuple(skip3_chain))

        down3 = getattr(model, "down_3", None)
        bottleneck = getattr(model, "bottleneck", None)
        if not isinstance(down3, nn.Module) or not isinstance(bottleneck, nn.Module):
            raise ValueError("verified MedNeXt down_3 and bottleneck are required for decoder phase")
        up3_input_chain = list(skip3_chain)
        down3_prefix = tuple(up3_input_chain)
        up3_input_chain.extend(_mednext_block_main_convs(down3, "down_3"))
        residual_down3 = getattr(down3, "res_conv", None)
        if residual_down3 is not None:
            residual_conv = _require_conv(residual_down3, "down_3.res_conv")
            if grid_phase((*down3_prefix, residual_conv)) != grid_phase(tuple(up3_input_chain)):
                raise ValueError("MedNeXt down_3 residual branch phase does not match main path")
        up3_input_chain.extend(_mednext_block_main_convs(bottleneck, "bottleneck"))
        up3_phase = _mednext_up_phase(
            model, "up_3", grid_phase(tuple(up3_input_chain)),
        )
        if up3_phase != skip3_phase:
            raise ValueError(
                f"MedNeXt dec_block_3 mixes upsample array-index lattice {up3_phase} "
                f"with encoder skip lattice {skip3_phase}; the transposed convolution "
                "has overlapping k3/s2 centers and its low-side pad leaves no single "
                "native sampling phase for the sum"
            )

        if target_stage == 2:
            up2_phase = _mednext_up_phase(model, "up_2", skip3_phase)
            if up2_phase != target_skip_phase:
                raise ValueError(
                    f"MedNeXt dec_block_2 mixes upsample array-index lattice {up2_phase} "
                    f"with encoder skip lattice {target_skip_phase}; the transposed "
                    "convolution has overlapping centers and no single native sampling phase"
                )

    if source == "bottleneck_stride16":
        bottleneck = getattr(model, "bottleneck", None)
        if not isinstance(bottleneck, nn.Module):
            raise ValueError("verified MedNeXt bottleneck is unavailable")
        chain.extend(_mednext_block_main_convs(bottleneck, "bottleneck"))
    return tuple(chain)


def teacher_feature_phase(
    teacher: DualBranchTeacher3D,
    source: str = "encoder_stride8",
    synthetic: bool = False,
) -> GridPhase:
    """Return the exact native sampling phase for a loaded teacher feature.

    The pinned MedNeXt-L source pads only at the high sides of the input, so
    those minimum-size/multiple-of-16 pads do not shift the grid origin. The
    tiny synthetic encoder also pads on high sides only.
    """

    if not isinstance(teacher, DualBranchTeacher3D):
        raise TypeError("teacher must be the existing DualBranchTeacher3D")
    canonical = _canonical_source(source)
    encoder = teacher.encoder
    if isinstance(encoder, TinyEncoder3D):
        details = teacher_feature_source(teacher, canonical, synthetic=synthetic)
        down_count = int(details["feature_index"])
        stem = encoder.stem[0]
        chain = [stem]
        chain.extend(encoder.down[index][0] for index in range(down_count))
        if details["pool"] is not None:
            chain.append(details["pool"])
        phase = grid_phase(chain)
        if details["channels"] != encoder.channels[down_count]:
            raise ValueError("tiny teacher feature channel does not match its actual stage")
        return phase

    details = teacher_feature_source(teacher, canonical, synthetic=synthetic)
    if not isinstance(encoder, MedNeXtL3DAdapter):  # pragma: no cover - checked above
        raise ValueError("unsupported Q-3D teacher encoder")
    if encoder.source_commit != UPSTREAM_MAIN_SHA:
        raise ValueError(
            f"Q-3D MedNeXt phase is verified only at pinned commit {UPSTREAM_MAIN_SHA}"
        )
    model = encoder.model
    phase = grid_phase(_mednext_phase_chain(model, canonical))
    if details["input_padding"] != "high_side_replicate_no_phase_shift":
        raise ValueError("teacher input padding origin is not a verified high-side-only pad")
    return phase


def teacher_grid_phase(
    teacher: DualBranchTeacher3D,
    source: str = "encoder_stride8",
    synthetic: bool = False,
) -> GridPhase:
    """Alias for callers that name the result as a grid rather than a feature phase."""

    return teacher_feature_phase(teacher, source=source, synthetic=synthetic)


def forward_teacher_with_features(
    teacher: DualBranchTeacher3D,
    images: Tensor,
    *,
    geometry: Mapping[str, Any],
    source: str = "encoder_stride8",
    synthetic: bool = False,
) -> dict[str, Tensor]:
    """Run the frozen teacher once and capture one actual feature output.

    The hook is installed for one teacher.forward call and always removed. This
    preserves the teacher wrapper's synthetic manufactured-logit behavior as
    well as the real MedNeXt region and boundary heads.
    """

    if teacher.training:
        raise ValueError("Q-3D feature extraction requires teacher.eval()")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("Q-3D feature extraction requires a frozen teacher")
    if not isinstance(images, Tensor) or images.ndim != 5:
        raise ValueError("images must be native [B,1,D,H,W]")

    details = teacher_feature_source(teacher, source, synthetic=synthetic)
    captured: list[Tensor] = []

    def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, Tensor):
            raise ValueError(f"teacher feature {details['module_name']} output must be a tensor")
        captured.append(output.detach())

    original_method = None
    patched_attribute: str | None = None
    handle = None
    if details["capture_api"] in {"forward_features", "forward_with_features"}:
        patched_attribute = str(details["capture_api"])
        original_method = getattr(teacher.encoder, patched_attribute)

        def capture_feature_tuple(*args: Any, **kwargs: Any) -> Any:
            result = original_method(*args, **kwargs)
            feature_values = result[0] if patched_attribute == "forward_with_features" else result
            index = int(details["feature_index"])
            if not isinstance(feature_values, (tuple, list)) or len(feature_values) <= index:
                raise ValueError(
                    f"teacher {patched_attribute} did not return feature index {index}"
                )
            feature = feature_values[index]
            if not isinstance(feature, Tensor):
                raise ValueError(f"teacher feature index {index} must be a tensor")
            captured.append(feature.detach())
            return result

        setattr(teacher.encoder, patched_attribute, capture_feature_tuple)
    else:
        if not isinstance(details["module"], nn.Module):
            raise ValueError("teacher feature hook module is unavailable")
        handle = details["module"].register_forward_hook(capture)
    try:
        with torch.no_grad():
            outputs = teacher(images, geometry=geometry)
    finally:
        if handle is not None:
            handle.remove()
        if patched_attribute is not None and original_method is not None:
            setattr(teacher.encoder, patched_attribute, original_method)

    if len(captured) != 1:
        raise RuntimeError(
            f"expected one execution of teacher feature {details['module_name']}, got {len(captured)}"
        )
    feature = captured[0]
    if details["pool"] is not None:
        with torch.no_grad():
            feature = details["pool"](feature)
    if feature.ndim != 5 or feature.shape[0] != images.shape[0] \
            or feature.shape[1] != details["channels"]:
        raise ValueError(
            f"teacher feature {details['module_name']} must return "
            f"[B,{details['channels']},D,H,W]"
        )
    if not torch.isfinite(feature).all():
        raise ValueError(f"teacher feature {details['module_name']} contains NaN or Inf")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "region_logits", "boundary_logits", "sdf",
    }:
        raise ValueError("teacher must return its canonical region, boundary and SDF mapping")
    return {**dict(outputs), "feature": feature}


__all__ = [
    "forward_teacher_with_features",
    "teacher_feature_phase",
    "teacher_feature_source",
    "teacher_grid_phase",
]
