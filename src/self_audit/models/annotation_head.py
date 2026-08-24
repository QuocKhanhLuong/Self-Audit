"""Initial soft cardiac annotation head."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class InitialAnnotationHead(nn.Module):
    def __init__(self, in_channels: int = 96, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        hidden = max(int(in_channels) // 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), hidden, 3, padding=1),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_classes, 1),
        )

    def forward(self, shared: Tensor, output_size: tuple[int, int] | None = None) -> Tensor:
        logits = self.net(shared)
        if output_size is None:
            output_size = (shared.shape[-2] * 4, shared.shape[-1] * 4)
        if tuple(logits.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


AnnotationHead = InitialAnnotationHead
