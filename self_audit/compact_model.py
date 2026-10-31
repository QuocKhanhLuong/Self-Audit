"""Compact patient student: 2.5D residual UNet with native geometry and GroupNorm.

Fully learned encoder, decoder and head with random Kaiming initialization.
No teacher, prototype, DINO or feature-caching dependencies.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence
import torch
from torch import nn
from torch.nn import functional as F
from .compact_runtime import compact_bilinear

ALLOWED_ARCHITECTURE_KEYS = {
    "name",
    "model_class",
    "widths",
    "in_channels",
    "num_classes",
    "norm",
    "init",
    "init_lineage",
    "pretrained",
    "dino_required",
}


def _find_groups(channels: int, max_groups: int = 8) -> int:
    """Find the largest divisor of channels up to max_groups."""
    limit = min(max_groups, channels)
    for g in range(limit, 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResidualBlock(nn.Module):
    """GroupNorm residual block with optional 1x1 projection."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        g1 = _find_groups(out_channels, groups)
        g2 = _find_groups(out_channels, groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(g1, out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.act2 = nn.GELU()
        if in_channels != out_channels:
            g_skip = _find_groups(out_channels, groups)
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(g_skip, out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act2(h + res)


class CompactEncoder(nn.Module):
    """Learned multi-stage 2.5D encoder with residual blocks and GroupNorm."""

    def __init__(self, widths: Sequence[int], in_channels: int = 3) -> None:
        super().__init__()
        self.widths = tuple(widths)
        self.in_channels = in_channels
        g0 = _find_groups(self.widths[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.widths[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g0, self.widths[0]),
            nn.GELU(),
            ResidualBlock(self.widths[0], self.widths[0]),
        )
        self.down_stages = nn.ModuleList()
        for i in range(1, len(self.widths)):
            c_in = self.widths[i - 1]
            c_out = self.widths[i]
            gi = _find_groups(c_out)
            stage = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(gi, c_out),
                nn.GELU(),
                ResidualBlock(c_out, c_out),
            )
            self.down_stages.append(stage)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        stages = [self.stem(x)]
        for stage in self.down_stages:
            stages.append(stage(stages[-1]))
        return stages


class CompactDecoder(nn.Module):
    """Learned multi-stage decoder with skip fusion, GroupNorm and residual refinement."""

    def __init__(self, widths: Sequence[int]) -> None:
        super().__init__()
        self.widths = tuple(widths)
        self.up_stages = nn.ModuleList()
        # Decode from bottleneck (last stage) up to stage 0
        for i in range(len(self.widths) - 2, -1, -1):
            c_deep = self.widths[i + 1]
            c_skip = self.widths[i]
            c_out = self.widths[i]
            g = _find_groups(c_out)
            stage = nn.ModuleDict({
                "fuse": nn.Sequential(
                    nn.Conv2d(c_deep + c_skip, c_out, kernel_size=1, bias=False),
                    nn.GroupNorm(g, c_out),
                    nn.GELU(),
                ),
                "refine": ResidualBlock(c_out, c_out),
            })
            self.up_stages.append(stage)

    def forward(self, stages: list[torch.Tensor]) -> torch.Tensor:
        d = stages[-1]
        for idx, i in enumerate(range(len(self.widths) - 2, -1, -1)):
            skip = stages[i]
            up = compact_bilinear(d, skip.shape[-2:])
            merged = torch.cat([up, skip], dim=1)
            stage = self.up_stages[idx]
            d = stage["refine"](stage["fuse"](merged))
        return d


def validate_architecture(architecture: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate an architecture mapping without silent type coercion."""
    if not isinstance(architecture, Mapping):
        raise ValueError(f"architecture must be a Mapping, got {type(architecture).__name__}")

    cfg = dict(architecture)
    if "architecture" in cfg and isinstance(cfg["architecture"], Mapping):
        cfg.update(cfg["architecture"])

    unknown = set(cfg) - ALLOWED_ARCHITECTURE_KEYS
    if unknown:
        raise ValueError(f"unknown architecture fields: {sorted(unknown)}")

    if "name" in cfg:
        name = cfg["name"]
        if not isinstance(name, str) or not (
            name in {"CompactStudent", "compact_student", "compact_student_v1"}
            or name.lower().startswith("compact_student")
        ):
            raise ValueError(f"invalid architecture name: {name!r}")

    if "model_class" in cfg:
        mc = cfg["model_class"]
        if not isinstance(mc, str) or mc != "CompactStudent":
            raise ValueError(f"invalid architecture model_class: {mc!r}")

    raw_widths = cfg.get("widths", (16, 32, 64, 128))
    if not isinstance(raw_widths, (list, tuple)):
        raise ValueError(f"architecture widths must be a list or tuple, got {type(raw_widths).__name__}")
    if len(raw_widths) < 2:
        raise ValueError(f"architecture widths must contain at least 2 stages, got {len(raw_widths)}")
    for i, w in enumerate(raw_widths):
        if type(w) is not int or w <= 0:
            raise ValueError(f"architecture widths[{i}] must be a positive integer without coercion, got {w!r}")
    widths = tuple(raw_widths)

    raw_in_channels = cfg.get("in_channels", 3)
    if type(raw_in_channels) is not int or raw_in_channels != 3:
        raise ValueError(f"architecture in_channels must be exactly integer 3, got {raw_in_channels!r}")
    in_channels = 3

    raw_num_classes = cfg.get("num_classes", 4)
    if type(raw_num_classes) is not int or raw_num_classes != 4:
        raise ValueError(f"architecture num_classes must be exactly integer 4, got {raw_num_classes!r}")
    num_classes = 4

    norm = cfg.get("norm", "group_norm")
    if norm not in {"group_norm", "GroupNorm"}:
        raise ValueError(f"architecture norm must be 'group_norm', got {norm!r}")

    init = cfg.get("init", "kaiming_normal")
    if init not in {"kaiming_normal", "kaiming"}:
        raise ValueError(f"architecture init must be 'kaiming_normal', got {init!r}")

    if "pretrained" in cfg:
        if type(cfg["pretrained"]) is not bool or cfg["pretrained"]:
            raise ValueError(f"architecture pretrained must be exact False, got {cfg['pretrained']!r}")

    if "dino_required" in cfg:
        if type(cfg["dino_required"]) is not bool or cfg["dino_required"]:
            raise ValueError(f"architecture dino_required must be exact False, got {cfg['dino_required']!r}")

    return {
        "name": "CompactStudent",
        "model_class": "CompactStudent",
        "widths": list(widths),
        "in_channels": in_channels,
        "num_classes": num_classes,
        "norm": "group_norm",
        "init": "kaiming_normal",
        "init_lineage": "random_kaiming_distinct_from_dino",
        "pretrained": False,
        "dino_required": False,
    }


class CompactStudent(nn.Module):
    """Patient CompactStudent: 2.5D residual UNet with configurable widths and native geometry.

    Exposes module_parameters() grouping encoder, decoder and head.
    model(x, valid_mask=None) returns B4HW logits.
    """

    def __init__(
        self,
        widths: Sequence[int] = (16, 32, 64, 128),
        in_channels: int = 3,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        arch_spec = validate_architecture({
            "widths": list(widths),
            "in_channels": in_channels,
            "num_classes": num_classes,
        })
        self.widths = tuple(arch_spec["widths"])
        self.in_channels = arch_spec["in_channels"]
        self.num_classes = arch_spec["num_classes"]
        self.encoder = CompactEncoder(self.widths, in_channels=self.in_channels)
        self.decoder = CompactDecoder(self.widths)
        self.head = nn.Conv2d(self.widths[0], self.num_classes, kernel_size=1)
        self._init_kaiming()
        self.architecture = arch_spec

    def _init_kaiming(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def module_parameters(self) -> dict[str, list[nn.Parameter]]:
        """Return parameters partitioned into encoder, decoder, and head groups."""
        enc_params = list(self.encoder.parameters())
        dec_params = list(self.decoder.parameters())
        head_params = list(self.head.parameters())
        enc_ids = {id(p) for p in enc_params}
        dec_ids = {id(p) for p in dec_params}
        head_ids = {id(p) for p in head_params}
        if enc_ids & dec_ids or enc_ids & head_ids or dec_ids & head_ids:
            raise RuntimeError("parameter groups are not disjoint")
        all_params = list(self.parameters())
        if len(enc_params) + len(dec_params) + len(head_params) != len(all_params):
            raise RuntimeError("module_parameters did not partition all model parameters")
        return {
            "encoder": enc_params,
            "decoder": dec_params,
            "head": head_params,
        }

    def module_named_parameters(self) -> dict[str, dict[str, nn.Parameter]]:
        """Return named parameters grouped by canonical module name."""
        return {
            "encoder": dict(self.encoder.named_parameters()),
            "decoder": dict(self.decoder.named_parameters()),
            "head": dict(self.head.named_parameters()),
        }

    def _pad_geometry(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        h, w = x.shape[-2:]
        if h <= 0 or w <= 0:
            raise ValueError(f"spatial dimensions must be positive, got height={h}, width={w}")
        stride_factor = 2 ** (len(self.widths) - 1)
        pad_w = (-w) % stride_factor
        pad_h = (-h) % stride_factor
        if pad_w > 0 or pad_h > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            return x_pad, h, w
        return x, h, w

    def features(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Extract spatial features before the output head with native geometry support."""
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"expected torch.Tensor input, got {type(x).__name__}")
        if not torch.is_floating_point(x):
            raise ValueError(f"expected floating point input tensor, got {x.dtype}")
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected 4D tensor with {self.in_channels} channels [B, {self.in_channels}, H, W], got {x.shape}")
        if not torch.isfinite(x).all():
            raise ValueError("input contains non-finite values (NaN or Inf)")
        if x.numel() > 0:
            if x.min() < -1e-5 or x.max() > 1.0 + 1e-5:
                raise ValueError(f"expected normalized input values in [0, 1], got min={x.min().item()}, max={x.max().item()}")
        if valid_mask is not None:
            if not isinstance(valid_mask, torch.Tensor):
                raise ValueError(f"valid_mask must be a torch.Tensor, got {type(valid_mask).__name__}")
            if valid_mask.dtype is not torch.bool:
                raise ValueError(f"valid_mask must have boolean dtype, got {valid_mask.dtype}")
            if valid_mask.ndim != 3 or valid_mask.shape != (x.shape[0], x.shape[2], x.shape[3]):
                raise ValueError(
                    f"valid_mask shape {valid_mask.shape} != expected {(x.shape[0], x.shape[2], x.shape[3])}"
                )
        x_pad, orig_h, orig_w = self._pad_geometry(x)
        stages = self.encoder(x_pad)
        f_pad = self.decoder(stages)
        if f_pad.shape[-2:] != (orig_h, orig_w):
            f_pad = f_pad[..., :orig_h, :orig_w]
        return f_pad

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass returning [B, 4, H, W] logits."""
        feat = self.features(x, valid_mask=valid_mask)
        return self.head(feat)


def build_compact_model(
    architecture: Mapping[str, Any],
    *,
    return_metadata: bool = False,
) -> CompactStudent | tuple[CompactStudent, dict[str, Any]]:
    """Build a CompactStudent from an explicit architecture specification.

    Requires an explicit Mapping with strictly validated fields (no float/bool coercion).
    """
    if architecture is None or not isinstance(architecture, Mapping):
        raise ValueError("build_compact_model requires an explicit architecture Mapping")

    arch_spec = validate_architecture(architecture)
    model = CompactStudent(
        widths=arch_spec["widths"],
        in_channels=arch_spec["in_channels"],
        num_classes=arch_spec["num_classes"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weights_only_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    metadata = {
        "schema": "self_audit.compact.model.v1",
        "architecture": copy.deepcopy(model.architecture),
        "parameters_total": total_params,
        "parameters_trainable": trainable_params,
        "weights_only_bytes": weights_only_bytes,
        "dino_required": False,
        "pretrained": False,
        "init": "kaiming_normal",
        "init_lineage": "random_kaiming_distinct_from_dino",
    }
    model.metadata = metadata

    if return_metadata:
        return model, metadata
    return model


__all__ = [
    "ALLOWED_ARCHITECTURE_KEYS",
    "CompactDecoder",
    "CompactEncoder",
    "CompactStudent",
    "ResidualBlock",
    "build_compact_model",
    "validate_architecture",
]
