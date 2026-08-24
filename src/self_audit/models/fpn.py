"""Lightweight top-down FPN for ConvNeXt multi-scale features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class LightweightFPN(nn.Module):
    """Fuse hierarchical features into one ``[B,96,H/4,W/4]`` feature."""

    def __init__(self, in_channels: Sequence[int], out_channels: int = 96) -> None:
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels must contain at least one feature scale")
        self.in_channels = tuple(int(value) for value in in_channels)
        self.out_channels = int(out_channels)
        self.lateral = nn.ModuleList([nn.Conv2d(channel, self.out_channels, 1) for channel in self.in_channels])
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                    nn.GroupNorm(_groups(self.out_channels), self.out_channels),
                    nn.GELU(),
                )
                for _ in self.in_channels
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
            nn.GroupNorm(_groups(self.out_channels), self.out_channels),
            nn.GELU(),
        )

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != len(self.lateral):
            raise ValueError(f"Expected {len(self.lateral)} feature scales, got {len(features)}")
        pyramid = [lateral(feature) for lateral, feature in zip(self.lateral, features)]
        for index in range(len(pyramid) - 2, -1, -1):
            pyramid[index] = pyramid[index] + F.interpolate(
                pyramid[index + 1], size=pyramid[index].shape[-2:], mode="bilinear", align_corners=False
            )
        refined = [block(feature) for block, feature in zip(self.refine, pyramid)]
        return self.output(refined[0])


FPN = LightweightFPN
