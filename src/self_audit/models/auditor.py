"""Counterfactual transition auditor with local and global heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


@dataclass
class AuditOutput:
    local_logits: Tensor
    delta_q: Tensor

    @property
    def global_delta_q(self) -> Tensor:
        return self.delta_q

    @property
    def local_audit_logits(self) -> Tensor:
        return self.local_logits

    def __iter__(self):
        yield self.local_logits
        yield self.delta_q


class CounterfactualAuditor(nn.Module):
    """Predict the signed quality of ``P_previous -> P_candidate``."""

    def __init__(self, feature_channels: int = 96, num_classes: int = 4, hidden_channels: int = 96, residual_blocks: int = 3) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        input_channels = self.feature_channels + self.num_classes * 3 + 2
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_channels, 1),
            nn.GroupNorm(8 if self.hidden_channels % 8 == 0 else 1, self.hidden_channels),
            nn.GELU(),
        )
        self.residual = nn.Sequential(*[_ResidualBlock(self.hidden_channels) for _ in range(max(int(residual_blocks), 1))])
        self.local_head = nn.Conv2d(self.hidden_channels, 3, 1)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.hidden_channels, max(self.hidden_channels // 2, 16)),
            nn.GELU(),
            nn.Linear(max(self.hidden_channels // 2, 16), 1),
        )

    @staticmethod
    def _probabilities(value: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError(f"Annotation transition tensors must be [B,C,H,W], got {tuple(value.shape)}")
        if bool((value.detach().min() < 0).item()) or bool((value.detach().max() > 1).item()):
            return value.softmax(dim=1)
        return value / value.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        H: Tensor,
        P_previous: Tensor,
        P_candidate: Tensor,
        delta_P: Tensor | None = None,
        entropy_previous: Tensor | None = None,
        entropy_candidate: Tensor | None = None,
    ) -> AuditOutput:
        if H.ndim != 4:
            raise ValueError(f"H must be [B,C,H,W], got {tuple(H.shape)}")
        if H.shape[1] != self.feature_channels:
            raise ValueError(f"Expected H with {self.feature_channels} channels, got {H.shape[1]}")
        # This detachment is intentional and is part of the v1 gradient
        # contract: the auditor cannot teach the annotation network to collude.
        H_audit = H.detach()
        previous = self._probabilities(P_previous.detach())
        candidate = self._probabilities(P_candidate.detach())
        delta = (candidate - previous) if delta_P is None else delta_P.detach()
        if delta.shape[1] != self.num_classes:
            raise ValueError(f"Expected delta_P with {self.num_classes} classes, got {delta.shape[1]}")
        spatial = H_audit.shape[-2:]
        previous = F.interpolate(previous, size=spatial, mode="bilinear", align_corners=False)
        candidate = F.interpolate(candidate, size=spatial, mode="bilinear", align_corners=False)
        delta = F.interpolate(delta, size=spatial, mode="bilinear", align_corners=False)
        if entropy_previous is None:
            entropy_previous = -(previous.clamp_min(1e-8) * previous.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        else:
            entropy_previous = entropy_previous.detach()
            entropy_previous = F.interpolate(entropy_previous, size=spatial, mode="bilinear", align_corners=False)
        if entropy_candidate is None:
            entropy_candidate = -(candidate.clamp_min(1e-8) * candidate.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        else:
            entropy_candidate = entropy_candidate.detach()
            entropy_candidate = F.interpolate(entropy_candidate, size=spatial, mode="bilinear", align_corners=False)
        features = self.input_projection(torch.cat([H_audit, previous, candidate, delta, entropy_previous, entropy_candidate], dim=1))
        features = self.residual(features)
        local = self.local_head(features)
        local = F.interpolate(local, size=P_previous.shape[-2:], mode="bilinear", align_corners=False)
        delta_q = self.global_head(features)
        return AuditOutput(local, delta_q)


Auditor = CounterfactualAuditor
