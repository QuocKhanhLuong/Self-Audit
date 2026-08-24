"""ConvNeXt-Tiny feature encoder used by the locked 2.5-D baseline."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _FallbackConvNeXtBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.pwconv1 = nn.Conv2d(channels, channels * 4, 1)
        self.pwconv2 = nn.Conv2d(channels * 4, channels, 1)
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-6)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        y = self.dwconv(x)
        y = self.norm(y)
        y = self.pwconv2(F.gelu(self.pwconv1(y)))
        return residual + self.gamma.view(1, -1, 1, 1) * y


class _FallbackHierarchicalEncoder(nn.Module):
    """Small shape-compatible fallback for offline unit tests.

    It is not presented as the pretrained baseline; it only permits synthetic
    tests on machines where timm or ImageNet weights are unavailable.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        channels = (96, 192, 384, 768)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 4, stride=4),
            nn.GroupNorm(1, channels[0]),
        )
        stages: list[nn.Module] = []
        for index, channel in enumerate(channels):
            blocks = [_FallbackConvNeXtBlock(channel)]
            if index:
                blocks.insert(0, nn.Sequential(nn.GroupNorm(1, channels[index - 1]), nn.Conv2d(channels[index - 1], channel, 2, stride=2)))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.out_channels = channels
        self.reductions = (4, 8, 16, 32)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        features = []
        for index, stage in enumerate(self.stages):
            if index == 0:
                y = stage(x)
            else:
                previous = features[-1]
                # ConvNeXt's normal input is much larger, but padding the
                # synthetic smoke path keeps the fallback well-defined down
                # to tiny spatial tensors without changing real resolutions.
                pad_h = max(0, 2 - previous.shape[-2])
                pad_w = max(0, 2 - previous.shape[-1])
                if pad_h or pad_w:
                    previous = F.pad(previous, (0, pad_w, 0, pad_h))
                y = stage(previous)
            features.append(y)
        return features


class ConvNeXtTinyEncoder(nn.Module):
    """Expose ConvNeXt-Tiny multi-scale features for RGB-shaped MRI slices."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        pretrained: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        allow_fallback: bool = True,
    ) -> None:
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError("The locked baseline expects [B,3,H,W] neighboring-slice input")
        self.name = "convnext_tiny"
        self.pretrained = bool(pretrained)
        self.using_timm = False
        try:
            import timm

            self.model = timm.create_model(
                "convnext_tiny",
                features_only=True,
                pretrained=bool(pretrained),
                in_chans=3,
                out_indices=tuple(int(i) for i in out_indices),
            )
            self.out_channels = tuple(int(value) for value in self.model.feature_info.channels())
            self.reductions = tuple(int(value) for value in self.model.feature_info.reduction())
            self.using_timm = True
        except Exception as exc:
            if pretrained or not allow_fallback:
                raise ImportError(
                    "ConvNeXt-Tiny ImageNet support requires timm and accessible weights. "
                    "Install timm or set pretrained=False for dependency-light synthetic tests."
                ) from exc
            self.model = _FallbackHierarchicalEncoder(in_channels=3)
            self.out_channels = self.model.out_channels
            self.reductions = self.model.reductions

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"ConvNeXt-Tiny encoder expects [B,3,H,W], got {tuple(x.shape)}")
        features = self.model(x)
        return list(features)


def build_encoder(
    *,
    name: str = "convnext_tiny",
    pretrained: bool = True,
    in_channels: int = 3,
    allow_fallback: bool = True,
) -> ConvNeXtTinyEncoder:
    normalized = str(name).lower().replace("-", "_")
    if normalized not in {"convnext_tiny", "convnexttiny"}:
        raise ValueError(f"Only convnext_tiny is supported by the locked baseline, got {name!r}")
    return ConvNeXtTinyEncoder(
        in_channels=in_channels,
        pretrained=pretrained,
        allow_fallback=allow_fallback,
    )
