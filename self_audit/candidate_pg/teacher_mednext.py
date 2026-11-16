"""Candidate PG MedNeXt-L teacher adapter.

This module keeps the PG teacher separate from the historical A/FUSE/Q3D
tracks.  The production path is the hash-pinned MedNeXt-v1 factory used by the
existing A-3D adapter; no substitute backbone is silently selected.  The
teacher exposes the original full MedNeXt region decoder, a separate
high-resolution boundary/SDF compatibility head, and the five live encoder
features consumed by :class:`~self_audit.candidate_fuse.student.FuseStudent`.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..candidate_a3d.teacher import (
    CLASS_NAMES,
    UPSTREAM_MAIN_SHA,
    UPSTREAM_URL,
    UPSTREAM_VERIFIED_FILES,
    MedNeXtL3DAdapter,
    _geometry_spacing,
    _load_mednext,
)
from ..candidate_q3d.alignment import GridPhase, grid_phase


MEDNEXT_CHANNELS = (32, 64, 128, 256, 512)
MEDNEXT_BLOCK_COUNTS = (3, 4, 8, 8, 8, 8, 8, 4, 3)
MEDNEXT_EXP_RATIOS = (3, 4, 8, 8, 8, 8, 8, 4, 3)
MEDNEXT_DOWN_STRIDES = ((1, 2, 2), (1, 2, 2), (2, 2, 2), (1, 2, 2))
MEDNEXT_PHASE_STEPS = ((1, 1, 1), (1, 2, 2), (1, 4, 4), (2, 8, 8), (2, 16, 16))
MEDNEXT_PHASE_OFFSETS = ((0.0, 0.0, 0.0),) * 5


def _mednext_block_forward(base: nn.Module, x: Tensor) -> Tensor:
    """Run the upstream MedNeXtBlock internals on a resampling tensor."""

    x1 = base.conv1(x)
    x1 = base.act(base.conv2(base.norm(x1)))
    if bool(getattr(base, "grn", False)):
        dims = (-3, -2, -1) if x1.ndim == 5 else (-2, -1)
        gx = torch.norm(x1, p=2, dim=dims, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        x1 = base.grn_gamma * (x1 * nx) + base.grn_beta + x1
    x1 = base.conv3(x1)
    if bool(getattr(base, "do_res", False)):
        x1 = x + x1
    return x1


class _AnisotropicDown(nn.Module):
    """Upstream MedNeXtDownBlock with per-axis sampling stride."""

    def __init__(self, base: nn.Module, stride: Sequence[int]):
        super().__init__()
        self.base = copy.deepcopy(base)
        self.stride = tuple(int(v) for v in stride)
        self.base.conv1.stride = self.stride
        res_conv = getattr(self.base, "res_conv", None)
        if isinstance(res_conv, nn.Conv3d):
            res_conv.stride = self.stride

    @property
    def conv1(self) -> nn.Conv3d:
        return self.base.conv1

    @property
    def conv3(self) -> nn.Conv3d:
        return self.base.conv3

    @property
    def res_conv(self) -> nn.Conv3d | None:
        value = getattr(self.base, "res_conv", None)
        return value if isinstance(value, nn.Conv3d) else None

    def forward(self, x: Tensor, dummy_tensor: Tensor | None = None) -> Tensor:
        del dummy_tensor
        output = _mednext_block_forward(self.base, x)
        if bool(getattr(self.base, "resample_do_res", False)):
            res = self.base.res_conv(x)
            output = output + res
        return output


def _left_pad_for_stride(x: Tensor, stride: Sequence[int]) -> Tensor:
    sz, sy, sx = (int(v) for v in stride)
    return F.pad(x, (1 if sx > 1 else 0, 0, 1 if sy > 1 else 0, 0, 1 if sz > 1 else 0, 0))


def _dropout3d(value: Tensor, probability: float, training: bool,
               generator: torch.Generator | None = None) -> Tensor:
    if not training or probability <= 0.0:
        return value
    if probability >= 1.0:
        return torch.zeros_like(value)
    if generator is None:
        return F.dropout3d(value, p=probability, training=True)
    keep = torch.rand((value.shape[0], value.shape[1], 1, 1, 1), device=value.device, generator=generator) >= probability
    return value * keep.to(value.dtype) / (1.0 - probability)


class _AnisotropicUp(nn.Module):
    """Upstream MedNeXtUpBlock with padding only on sampled axes."""

    def __init__(self, base: nn.Module, stride: Sequence[int]):
        super().__init__()
        self.base = copy.deepcopy(base)
        self.stride = tuple(int(v) for v in stride)
        self.base.conv1.stride = self.stride
        res_conv = getattr(self.base, "res_conv", None)
        if isinstance(res_conv, nn.ConvTranspose3d):
            res_conv.stride = self.stride

    @property
    def conv1(self) -> nn.ConvTranspose3d:
        return self.base.conv1

    @property
    def conv3(self) -> nn.Conv3d:
        return self.base.conv3

    def forward(self, x: Tensor, dummy_tensor: Tensor | None = None) -> Tensor:
        del dummy_tensor
        output = _mednext_block_forward(self.base, x)
        output = _left_pad_for_stride(output, self.stride)
        if bool(getattr(self.base, "resample_do_res", False)):
            res = _left_pad_for_stride(self.base.res_conv(x), self.stride)
            output = output + res
        return output


class _PGAnisotropicMedNeXt(nn.Module):
    """A full official-v1 module sequence with anisotropic resampling."""

    _candidate_pg_native_v1 = True

    def __init__(self, source: nn.Module, strides: Sequence[Sequence[int]]):
        super().__init__()
        self.stem = copy.deepcopy(source.stem)
        for name in ("enc_block_0", "enc_block_1", "enc_block_2", "enc_block_3", "bottleneck",
                     "dec_block_3", "dec_block_2", "dec_block_1", "dec_block_0", "out_0"):
            setattr(self, name, copy.deepcopy(getattr(source, name)))
        strides = tuple(tuple(int(v) for v in value) for value in strides)
        self.down_0 = _AnisotropicDown(source.down_0, strides[0])
        self.down_1 = _AnisotropicDown(source.down_1, strides[1])
        self.down_2 = _AnisotropicDown(source.down_2, strides[2])
        self.down_3 = _AnisotropicDown(source.down_3, strides[3])
        self.up_3 = _AnisotropicUp(source.up_3, strides[3])
        self.up_2 = _AnisotropicUp(source.up_2, strides[2])
        self.up_1 = _AnisotropicUp(source.up_1, strides[1])
        self.up_0 = _AnisotropicUp(source.up_0, strides[0])
        self.do_ds = bool(getattr(source, "do_ds", False))

    @staticmethod
    def _run_module(module: nn.Module, value: Tensor) -> Tensor:
        # Preserve the native module/parameter layout while recomputing its
        # internal activations during backward. FP skip-dropout draws occur
        # outside these modules, so its private generator is never replayed.
        if module.training and torch.is_grad_enabled() and (value.requires_grad or any(p.requires_grad for p in module.parameters())):
            from torch.utils.checkpoint import checkpoint

            if isinstance(module, nn.Sequential):
                for block in module:
                    value = _PGAnisotropicMedNeXt._run_module(block, value)
                return value
            return checkpoint(module, value, use_reentrant=False, preserve_rng_state=True)
        return module(value)

    def forward_with_features(
        self, x: Tensor, *, skip_dropout: float = 0.0,
        dropout_generator: torch.Generator | None = None,
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        x = self.stem(x)
        x_res_0 = self._run_module(self.enc_block_0, x)
        x = self._run_module(self.down_0, x_res_0)
        x_res_1 = self._run_module(self.enc_block_1, x)
        x = self._run_module(self.down_1, x_res_1)
        x_res_2 = self._run_module(self.enc_block_2, x)
        x = self._run_module(self.down_2, x_res_2)
        x_res_3 = self._run_module(self.enc_block_3, x)
        x = self._run_module(self.down_3, x_res_3)
        x = self._run_module(self.bottleneck, x)
        features = (x_res_0, x_res_1, x_res_2, x_res_3, x)
        x = _dropout3d(x, skip_dropout, self.training, dropout_generator)
        x = self._run_module(self.up_3, x)
        x = self._run_module(self.dec_block_3, _dropout3d(x_res_3, skip_dropout, self.training, dropout_generator) + x)
        x = self._run_module(self.up_2, x)
        x = self._run_module(self.dec_block_2, _dropout3d(x_res_2, skip_dropout, self.training, dropout_generator) + x)
        x = self._run_module(self.up_1, x)
        x = self._run_module(self.dec_block_1, _dropout3d(x_res_1, skip_dropout, self.training, dropout_generator) + x)
        x = self._run_module(self.up_0, x)
        x = self._run_module(self.dec_block_0, _dropout3d(x_res_0, skip_dropout, self.training, dropout_generator) + x)
        return features, self._run_module(self.out_0, x)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_features(x)[1]


class _PGBoundaryPath(nn.Module):
    """Boundary logits plus foreground-3 and background-1 SDF heads."""

    def __init__(self, in_channels: int, deep_channels: int, width: int):
        super().__init__()
        self.detail = nn.Sequential(
            nn.Conv3d(in_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, width // 2)), width), nn.GELU(),
            nn.Conv3d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, width // 2)), width), nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv3d(deep_channels, width, 1, bias=False),
            nn.GroupNorm(max(1, min(8, width // 2)), width), nn.GELU(),
        )
        self.spacing_context = nn.Sequential(nn.Linear(3, width), nn.LayerNorm(width), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, width // 2)), width), nn.GELU(),
            nn.Conv3d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, width // 2)), width), nn.GELU(),
        )
        self.boundary_head = nn.Conv3d(width, 4, 1)
        self.sdf_head = nn.Conv3d(width, 3, 1)
        self.sdf_background_head = nn.Conv3d(width, 1, 1)

    def forward(self, high: Tensor, deep: Tensor, spacing_zyx: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        detail = self.detail(high)
        context = self.context(deep)
        if spacing_zyx.ndim != 2 or spacing_zyx.shape[0] != detail.shape[0] or spacing_zyx.shape[1] != 3:
            raise ValueError("boundary spacing conditioning must have shape [B,3]")
        if not torch.isfinite(spacing_zyx).all() or bool((spacing_zyx <= 0).any()):
            raise ValueError("boundary spacing must be finite and positive")
        token = self.spacing_context(torch.log(spacing_zyx).to(next(self.spacing_context.parameters()).dtype))
        context = F.interpolate(context + token.to(context.dtype)[..., None, None, None], size=detail.shape[-3:], mode="trilinear", align_corners=False)
        fused = self.fuse(detail + context)
        return self.boundary_head(fused), torch.tanh(self.sdf_head(fused)), torch.tanh(self.sdf_background_head(fused))


def _first_conv(module: nn.Module, *, name: str) -> nn.Conv3d:
    """Return the actual sampling convolution on one verified live path.

    In the official MedNeXtDownBlock the stride belongs to ``conv1`` while
    ``conv3`` is the 1x1 channel-compression path.  Synthetic fixtures may
    expose only ``conv3`` as their sampling layer, hence the stride-aware
    selection below.
    """

    if not isinstance(module, nn.Module):
        raise ValueError(f"MedNeXt {name} has no live module")
    for attribute in ("conv1", "conv3"):
        preferred = getattr(module, attribute, None)
        if isinstance(preferred, nn.Conv3d) and tuple(int(v) for v in preferred.stride) != (1, 1, 1):
            return preferred
    for child in module.modules():
        if isinstance(child, nn.Conv3d) and tuple(int(v) for v in child.stride) != (1, 1, 1):
            return child
    for attribute in ("conv1", "conv3"):
        preferred = getattr(module, attribute, None)
        if isinstance(preferred, nn.Conv3d):
            return preferred
    for child in module.modules():
        if isinstance(child, nn.Conv3d):
            return child
    raise ValueError(f"MedNeXt {name} has no live Conv3d sampling module")


def mednext_phase_chains(model: nn.Module) -> tuple[tuple[nn.Conv3d, ...], ...]:
    """Build the five phase chains from live MedNeXt sampling modules.

    The encoder blocks contain residual and depthwise convolutions.  They do
    not change the sampling lattice, so only the stem and the four explicit
    ``down_*`` paths are included.  This mirrors the verified upstream forward
    sequence and avoids guessing phase offsets from configuration text.
    """

    stem = _first_conv(getattr(model, "stem", None), name="stem")
    downs = tuple(_first_conv(getattr(model, f"down_{i}", None), name=f"down_{i}") for i in range(4))
    return tuple((stem, *downs[:index]) for index in range(5))


def derive_mednext_feature_phases(
    model: nn.Module,
    *,
    expected_channels: Sequence[int] = MEDNEXT_CHANNELS,
    expected_steps: Sequence[Sequence[int]] = MEDNEXT_PHASE_STEPS,
    expected_offsets: Sequence[Sequence[float]] = MEDNEXT_PHASE_OFFSETS,
) -> tuple[GridPhase, ...]:
    """Derive and validate feature phases from the actual MedNeXt modules."""

    chains = mednext_phase_chains(model)
    phases = tuple(grid_phase(chain) for chain in chains)
    expected_channels = tuple(int(value) for value in expected_channels)
    if len(expected_channels) != 5:
        raise ValueError("MedNeXt-L PG teacher requires five feature widths")
    live_channels = [int(getattr(getattr(model, "stem"), "out_channels", -1))]
    for index in range(4):
        down = getattr(model, f"down_{index}")
        conv = getattr(down, "conv3", None)
        if not isinstance(conv, nn.Conv3d):
            raise ValueError(f"MedNeXt down_{index} has no live channel projection conv3")
        live_channels.append(int(conv.out_channels))
    if tuple(live_channels) != expected_channels:
        raise ValueError(
            f"MedNeXt-L live channels {tuple(live_channels)} disagree with configured {expected_channels}"
        )
    expected_steps = tuple(tuple(int(v) for v in step) for step in expected_steps)
    if len(expected_steps) != 5:
        raise ValueError("MedNeXt-L PG teacher requires five feature phases")
    expected_offsets = tuple(tuple(float(v) for v in offset) for offset in expected_offsets)
    if len(expected_offsets) != 5:
        raise ValueError("MedNeXt-L PG teacher requires five phase offsets")
    for index, (phase, expected, expected_offset) in enumerate(zip(phases, expected_steps, expected_offsets)):
        if tuple(phase[0]) != expected:
            raise ValueError(
                f"MedNeXt-L phase {index} derives as {phase[0]}, expected configured {expected}"
            )
        if tuple(float(v) for v in phase[1]) != expected_offset:
            raise ValueError(
                f"MedNeXt-L phase {index} derives offset {phase[1]}, expected {expected_offset}"
            )
    # A residual resampling branch, when enabled by the upstream factory, must
    # represent the same native lattice as the main depthwise sampler.  Check
    # it independently rather than assuming branch equivalence.
    stem = _first_conv(getattr(model, "stem", None), name="stem")
    main_downs = tuple(_first_conv(getattr(model, f"down_{i}"), name=f"down_{i}") for i in range(4))
    for index in range(4):
        residual = getattr(getattr(model, f"down_{index}"), "res_conv", None)
        if not isinstance(residual, nn.Conv3d):
            continue
        branch_phase = grid_phase((stem, *main_downs[:index], residual))
        if branch_phase != phases[index + 1]:
            raise ValueError(
                f"MedNeXt down_{index} residual phase {branch_phase} disagrees with main {phases[index + 1]}"
            )
    return phases


def _teacher_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("PG teacher config must be a mapping")
    model = config.get("model")
    teacher = model.get("teacher") if isinstance(model, Mapping) and isinstance(model.get("teacher"), Mapping) else config.get("teacher")
    if teacher is None:
        teacher = config
    if not isinstance(teacher, Mapping):
        raise ValueError("PG config requires a teacher mapping")
    result = copy.deepcopy(dict(teacher))
    arch = str(result.get("arch", result.get("backbone", "mednext_l")))
    if arch not in {"mednext_l", "mednext_l_k3", "mednext_l_3d_k3"}:
        raise ValueError("PG teacher requires the verified MedNeXt-L architecture")
    kernel = int(result.get("kernel", result.get("kernel_size", 3)))
    if kernel != 3:
        raise ValueError("PG teacher requires MedNeXt-L kernel size 3")
    channels = tuple(int(v) for v in result.get("channels", MEDNEXT_CHANNELS))
    if channels != MEDNEXT_CHANNELS:
        raise ValueError("PG teacher channels must be [32,64,128,256,512]")
    block_counts = tuple(int(v) for v in result.get("block_counts", MEDNEXT_BLOCK_COUNTS))
    if block_counts != MEDNEXT_BLOCK_COUNTS:
        raise ValueError("PG teacher block_counts must match MedNeXt-L [3,4,8,8,8,8,8,4,3]")
    exp_ratios = tuple(int(v) for v in result.get("exp_ratios", MEDNEXT_EXP_RATIOS))
    if exp_ratios != MEDNEXT_EXP_RATIOS:
        raise ValueError("PG teacher exp_ratios must match MedNeXt-L [3,4,8,8,8,8,8,4,3]")
    strides = tuple(tuple(int(v) for v in s) for s in result.get("down_strides", MEDNEXT_DOWN_STRIDES))
    if strides != MEDNEXT_DOWN_STRIDES:
        raise ValueError("PG teacher down_strides must be anisotropic [1,2,2],[1,2,2],[2,2,2],[1,2,2]")
    phases = tuple(tuple(int(v) for v in p) for p in result.get("feature_phases", MEDNEXT_PHASE_STEPS))
    if phases != MEDNEXT_PHASE_STEPS:
        raise ValueError("PG teacher feature_phases do not match the locked anisotropic schedule")
    heads = result.get("heads", {})
    if isinstance(heads, Mapping):
        sdf = heads.get("sdf")
        if sdf is not None and not isinstance(sdf, Mapping):
            raise ValueError("PG teacher heads.sdf must be a mapping")
        sdf_channels = int(sdf.get("channels", 3)) if isinstance(sdf, Mapping) else int(heads.get("sdf_channels", 3))
        sdf_radius = float(sdf.get("radius_mm", 5.0)) if isinstance(sdf, Mapping) else float(heads.get("radius_mm", 5.0))
        if int(heads.get("region", 4)) != 4 or sdf_channels != 3:
            raise ValueError("PG teacher public heads require region=4 and foreground sdf_channels=3")
        if sdf_radius != 5.0:
            raise ValueError("PG teacher SDF radius must be 5 mm")
    result.update({"backbone": "mednext_l_k3", "input_channels": 1, "channels": list(channels),
                   "block_counts": list(block_counts), "exp_ratios": list(exp_ratios)})
    return result


class MedNeXtPGTeacher(nn.Module):
    """Full anisotropic MedNeXt-L teacher with a Q3D-compatible boundary head."""

    def __init__(self, config: Mapping[str, Any], *, model: nn.Module | None = None) -> None:
        super().__init__()
        teacher_config = _teacher_config(config)
        if model is None:
            # ``_load_mednext`` enforces the official source checkout, commit,
            # clean tree, factory and optional hash-pinned checkpoint.
            source_adapter, provenance = _load_mednext(teacher_config)
            encoder = _PGAnisotropicMedNeXt(source_adapter.model, MEDNEXT_DOWN_STRIDES)
        else:
            if isinstance(model, MedNeXtL3DAdapter):
                encoder = model
            elif isinstance(model, _PGAnisotropicMedNeXt):
                encoder = model
            else:
                raise TypeError("PG teacher model must be a verified MedNeXt adapter or PG anisotropic core")
            provenance = {
                "backbone": "MedNeXt-L-k3", "weights_status": "INJECTED_VERIFIED_ADAPTER",
                "source_commit": getattr(encoder, "source_commit", UPSTREAM_MAIN_SHA),
                "source_repo": getattr(encoder, "source_repo", "injected"),
                "features_api": getattr(encoder, "features_api", "native_mednext_v1_sequence"),
            }
        live_model = encoder.model if isinstance(encoder, MedNeXtL3DAdapter) else encoder
        phases = derive_mednext_feature_phases(live_model)
        channels = tuple(int(v) for v in (
            encoder.channels if isinstance(encoder, MedNeXtL3DAdapter)
            else (int(encoder.stem.out_channels), *(int(getattr(encoder, f"down_{i}").conv3.out_channels) for i in range(4)))
        ))
        if channels != MEDNEXT_CHANNELS:
            raise ValueError("PG MedNeXt-L adapter must expose channels [32,64,128,256,512]")
        self.encoder = encoder
        self.feature_channels = channels
        self.feature_phases = phases
        self.teacher_phases = phases
        self.class_mapping = CLASS_NAMES
        self.num_classes = 4
        self.sdf_channels = 3
        self.sdf_radius_mm = 5.0
        self.sdf_near_boundary_mm = 2.5
        self.padding_multiple = 16
        self.padding_multiples = (2, 16, 16)
        self.padding_policy = "high_side_replicate_to_anisotropic_bottleneck"
        self.boundary_path = _PGBoundaryPath(channels[0], channels[-1], channels[0])
        self.provenance = {
            "model_class": "MedNeXtPGTeacher",
            "architecture": "MedNeXt-L-k3-full-encoder-decoder-plus-boundary-sdf",
            "feature_channels": list(channels),
            "block_counts": list(MEDNEXT_BLOCK_COUNTS),
            "exp_ratios": list(MEDNEXT_EXP_RATIOS),
            "feature_phases": [[list(step), list(offset)] for step, offset in phases],
            "class_mapping": list(CLASS_NAMES),
            "sdf_channels": self.sdf_channels,
            "sdf_all_channels": 4,
            "sdf_radius_mm": self.sdf_radius_mm,
            "boundary_compatibility_head": "boundary_logits",
            "q3d_teacher_loss_keys": ["region_logits", "boundary_logits", "sdf"],
            "q3d_sdf_source": "sdf_all (bg1 + foreground3); public sdf is foreground3",
            "upstream_url": UPSTREAM_URL,
            "reference_main_sha": UPSTREAM_MAIN_SHA,
            "verified_source_files": [f"{UPSTREAM_URL}/blob/{UPSTREAM_MAIN_SHA}/{item}" for item in UPSTREAM_VERIFIED_FILES],
            **provenance,
        }
        self._joint_frozen = False

    @property
    def model(self) -> nn.Module:
        """Return the underlying official MedNeXt module without alias keys."""

        return self.encoder.model if isinstance(self.encoder, MedNeXtL3DAdapter) else self.encoder

    def _pad(self, image: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        shape = tuple(int(v) for v in image.shape[-3:])
        target = tuple(max(multiple, ((value + multiple - 1) // multiple) * multiple)
                       for value, multiple in zip(shape, self.padding_multiples))
        padding = tuple(target_value - value for target_value, value in zip(target, shape))
        return F.pad(image, (0, padding[2], 0, padding[1], 0, padding[0]), mode="replicate"), shape

    @staticmethod
    def _check_image(image: Tensor) -> tuple[int, int, int]:
        if image.ndim != 5 or image.shape[1] != 1 or not image.is_floating_point():
            raise ValueError("PG teacher input must be floating [B,1,D,H,W]")
        if not torch.isfinite(image).all():
            raise ValueError("PG teacher input contains NaN or Inf")
        if any(int(v) <= 0 for v in image.shape[-3:]):
            raise ValueError("PG teacher volume dimensions must be positive")
        return tuple(int(v) for v in image.shape[-3:])

    def forward(
        self,
        image: Tensor,
        geometry: Mapping[str, Any],
        *,
        return_features: bool = True,
        skip_dropout: float = 0.0,
        dropout_generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        if isinstance(skip_dropout, bool) or not isinstance(skip_dropout, (int, float)):
            raise TypeError("skip_dropout must be a finite scalar in [0,1]")
        skip_dropout = float(skip_dropout)
        if not torch.isfinite(torch.tensor(skip_dropout)) or not 0.0 <= skip_dropout <= 1.0:
            raise ValueError("skip_dropout must be a finite scalar in [0,1]")
        original_shape = self._check_image(image)
        spacing = _geometry_spacing(geometry, original_shape, device=image.device, dtype=image.dtype)
        if spacing.shape[0] == 1 and image.shape[0] != 1:
            spacing = spacing.expand(image.shape[0], -1)
        if spacing.shape[0] != image.shape[0]:
            raise ValueError("runtime geometry spacing batch must equal image batch")
        padded, _ = self._pad(image)
        if isinstance(self.encoder, _PGAnisotropicMedNeXt):
            features, region = self.encoder.forward_with_features(
                padded, skip_dropout=skip_dropout, dropout_generator=dropout_generator,
            )
        else:
            if skip_dropout:
                raise ValueError("skip_dropout requires the PG-owned anisotropic decoder")
            features, region = self.encoder.forward_with_features(padded)
        if len(features) != 5:
            raise RuntimeError("verified MedNeXt forward must return five encoder features")
        boundary, sdf, sdf_background = self.boundary_path(features[0], features[-1], spacing)
        crop = (..., slice(0, original_shape[0]), slice(0, original_shape[1]), slice(0, original_shape[2]))
        output: dict[str, Any] = {
            "region_logits": region[crop],
            "boundary_logits": boundary[crop],
            "boundary_logits4": boundary[crop],
            "sdf": sdf[crop],
            "sdf_all": torch.cat((sdf_background[crop], sdf[crop]), dim=1),
        }
        if return_features:
            output["features"] = tuple(features)
        return output

    def as_q3d_outputs(self, output: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Select exactly the unchanged Q3D teacher-loss head contract."""

        required = ("region_logits", "boundary_logits", "sdf_all")
        if any(key not in output for key in required):
            raise ValueError("PG teacher output is missing a Q3D compatibility head")
        return {
            "region_logits": output["region_logits"],
            "boundary_logits": output["boundary_logits"],
            "sdf": output["sdf_all"],
        }

    def freeze(self) -> "MedNeXtPGTeacher":
        for parameter in self.parameters():
            parameter.grad = None
        self.requires_grad_(False)
        self._joint_frozen = True
        super().train(False)
        return self

    freeze_for_joint = freeze

    def train(self, mode: bool = True):
        if self._joint_frozen and mode:
            super().train(False)
            return self
        super().train(mode)
        return self


def build_teacher(
    config: Mapping[str, Any], *, synthetic: bool = False, model: nn.Module | None = None
) -> tuple[MedNeXtPGTeacher, dict[str, Any]]:
    # ``synthetic`` is accepted for engine symmetry, but never selects a
    # manufactured/random backbone.  Tests must inject a shape-faithful
    # adapter explicitly; real construction remains source-verified.
    del synthetic
    teacher = MedNeXtPGTeacher(config, model=model)
    return teacher, copy.deepcopy(teacher.provenance)


__all__ = [
    "MEDNEXT_CHANNELS", "MEDNEXT_BLOCK_COUNTS", "MEDNEXT_EXP_RATIOS", "MEDNEXT_DOWN_STRIDES", "MEDNEXT_PHASE_STEPS", "MEDNEXT_PHASE_OFFSETS",
    "MedNeXtPGTeacher", "build_teacher", "derive_mednext_feature_phases", "mednext_phase_chains",
]
