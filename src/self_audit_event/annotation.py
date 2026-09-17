"""Event-audit annotation actor over the canonical DynamicWindowAttention leaf.

Initial annotation already uses the dynamic reading: the coarse head seeds
logits, the shared dynamic-window writer produces A0, and refine reuses the
same writer. Guidance enters ONLY the coordinate generator's condition; it has
no path to values or the writer. Coordinates are returned, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from self_audit.models.dynamic_window import DynamicWindowAttention


@dataclass
class AnnotationState:
    features: Tensor
    logits: Tensor
    coordinates: Tensor


class _SharedWriter(nn.Module):
    """Dynamic attention followed by a shared residual refinement."""

    def __init__(self, channels: int, *, k: int, condition_channels: int) -> None:
        super().__init__()
        self.attention = DynamicWindowAttention(
            channels,
            k=k,
            condition_channels=condition_channels,
            offset_mode="structured",
        )
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(
        self,
        features: Tensor,
        *,
        condition: Tensor | None = None,
        turn_index: int | Tensor = 0,
        iteration_index: int | Tensor = 0,
        coordinate_override: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        window, metadata = self.attention(
            features,
            condition=condition,
            turn_index=turn_index,
            iteration_index=iteration_index,
            coordinate_override=coordinate_override,
            return_metadata=True,
        )
        return features + self.residual(window), metadata


class AnnotationExpert(nn.Module):
    """Two-stride encoder, coarse seed, one shared dynamic-window writer."""

    def __init__(self, channels: int = 16, num_classes: int = 4, k: int = 8) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.k = int(k)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 5, stride=2, padding=2),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.GELU(),
        )
        self.coarse_head = nn.Conv2d(channels, num_classes, 1)
        self.writer = _SharedWriter(channels, k=self.k, condition_channels=num_classes+1)
        self.logit_head = nn.Conv2d(channels, num_classes, 1)

    @property
    def feature_scale(self) -> int:
        return 4

    def _validate_image(self, image: Tensor) -> None:
        if not torch.is_tensor(image):
            raise TypeError(f"image must be a tensor, got {type(image)!r}")
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image [B,3,H,W], got {tuple(image.shape)}")
        if not image.is_floating_point():
            raise TypeError(f"image must be floating point, got {image.dtype}")

    def encode(self, image: Tensor) -> Tensor:
        self._validate_image(image)
        if int(image.shape[-1]) % 4 or int(image.shape[-2]) % 4:
            raise ValueError("image H and W must be divisible by 4 for the /4 feature grid")
        return self.encoder(image)

    def _write(
        self,
        features: Tensor,
        logits: Tensor,
        *,
        guidance: Tensor | None,
        coordinate_override: Tensor | None = None,
    ) -> AnnotationState:
        guidance = torch.zeros_like(logits[:, :1]) if guidance is None else guidance.detach()
        condition = torch.cat([logits.softmax(1), guidance], dim=1)
        written, metadata = self.writer(
            features,
            condition=condition,
            turn_index=0,
            iteration_index=0,
            coordinate_override=coordinate_override,
        )
        logits = logits + F.interpolate(
            self.logit_head(written),
            size=logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        coordinates = metadata["coordinates"]
        return AnnotationState(features, logits, coordinates)

    def start(self, image: Tensor) -> AnnotationState:
        self._validate_image(image)
        features = self.encode(image)
        seed = self.coarse_head(features)
        logits = F.interpolate(seed, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self._write(features, logits, guidance=None)

    def refine(
        self,
        state: AnnotationState,
        guidance: Tensor | None,
        *,
        coordinate_override: Tensor | None = None,
    ) -> AnnotationState:
        if not isinstance(state, AnnotationState):
            raise TypeError(f"state must be AnnotationState, got {type(state)!r}")
        self._validate_guidance(state, guidance)
        return self._write(
            state.features,
            state.logits,
            guidance=None if guidance is None else guidance.detach(),
            coordinate_override=coordinate_override,
        )

    def _validate_guidance(self, state: AnnotationState, guidance: Tensor | None) -> None:
        if guidance is None:
            return
        if not torch.is_tensor(guidance):
            raise TypeError(f"guidance must be a tensor, got {type(guidance)!r}")
        if guidance.shape != state.logits[:, :1].shape:
            raise ValueError(f"guidance must be [B,1,H,W], got {tuple(guidance.shape)}")
        if not guidance.is_floating_point():
            raise TypeError(f"guidance must be floating point, got {guidance.dtype}")
