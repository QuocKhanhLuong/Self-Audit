"""Stage-conditioned structured + free-form dynamic window attention.

The operator samples a learned, bounded K-point support per query location.
It is intentionally implemented with ``grid_sample`` and ordinary PyTorch
layers; no fixed square windows or custom CUDA kernels are involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class DynamicWindowParameters:
    center_displacement: Tensor
    radius: Tensor
    orientation: Tensor
    residual_offsets: Tensor
    coordinates: Tensor


def _as_index(value: int | Tensor, batch: int, device: torch.device) -> Tensor:
    if torch.is_tensor(value):
        index = value.to(device=device, dtype=torch.long).reshape(-1)
        if index.numel() == 1:
            index = index.expand(batch)
        if index.numel() != batch:
            raise ValueError(f"conditioning index has {index.numel()} values for batch {batch}")
        return index
    return torch.full((batch,), int(value), device=device, dtype=torch.long)


class DynamicWindowGenerator(nn.Module):
    """Predict bounded center/scale/rotation/free-form support parameters."""

    def __init__(
        self,
        channels: int,
        *,
        condition_channels: int = 3,
        k: int = 8,
        turn_dim: int = 16,
        iteration_dim: int = 16,
        max_turns: int = 8,
        max_center_displacement: float = 0.25,
        min_radius: float = 0.03,
        max_radius: float = 0.55,
        max_residual_offset: float = 0.12,
    ) -> None:
        super().__init__()
        if int(k) < 1:
            raise ValueError("k must be positive")
        if not 0.0 < min_radius <= max_radius:
            raise ValueError("radius bounds must satisfy 0 < min_radius <= max_radius")
        self.channels = int(channels)
        self.condition_channels = int(condition_channels)
        self.k = int(k)
        self.turn_embedding = nn.Embedding(max(int(max_turns), 1) + 1, int(turn_dim))
        self.iteration_embedding = nn.Embedding(max(int(max_turns), 1) + 1, int(iteration_dim))
        state_channels = self.channels + self.condition_channels + int(turn_dim) + int(iteration_dim)
        hidden = max(self.channels // 2, 32)
        self.state_net = nn.Sequential(
            nn.Conv2d(state_channels, hidden, 1),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.parameter_head = nn.Conv2d(hidden, 5 + 2 * self.k, 1)
        canonical = torch.arange(self.k, dtype=torch.float32) * (2.0 * math.pi / self.k)
        self.register_buffer("canonical_support", torch.stack([canonical.cos(), canonical.sin()], dim=-1), persistent=False)
        self.max_center_displacement = float(max_center_displacement)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.max_residual_offset = float(max_residual_offset)

    def _condition(self, condition: Tensor | None, x: Tensor) -> Tensor:
        if condition is None:
            return x.new_zeros((x.shape[0], self.condition_channels, x.shape[2], x.shape[3]))
        if condition.ndim != 4:
            raise ValueError(f"condition must be [B,C,H,W], got {tuple(condition.shape)}")
        condition = F.interpolate(condition.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)
        if condition.shape[1] == self.condition_channels:
            return condition
        if condition.shape[1] == 1:
            return condition.expand(-1, self.condition_channels, -1, -1)
        if condition.shape[1] > self.condition_channels:
            return condition[:, : self.condition_channels]
        repeats = math.ceil(self.condition_channels / condition.shape[1])
        return condition.repeat(1, repeats, 1, 1)[:, : self.condition_channels]

    def forward(
        self,
        x: Tensor,
        *,
        condition: Tensor | None = None,
        audit_map: Tensor | None = None,
        turn_index: int | Tensor = 0,
        iteration_index: int | Tensor = 0,
    ) -> DynamicWindowParameters:
        if x.ndim != 4:
            raise ValueError(f"Expected feature map [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} feature channels, got {x.shape[1]}")
        if condition is None:
            condition = audit_map
        batch = x.shape[0]
        turn = _as_index(turn_index, batch, x.device).clamp(0, self.turn_embedding.num_embeddings - 1)
        iteration = _as_index(iteration_index, batch, x.device).clamp(0, self.iteration_embedding.num_embeddings - 1)
        turn_feature = self.turn_embedding(turn).to(dtype=x.dtype).view(batch, -1, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        iteration_feature = self.iteration_embedding(iteration).to(dtype=x.dtype).view(batch, -1, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        state = torch.cat([x, self._condition(condition, x), turn_feature, iteration_feature], dim=1)
        raw = self.parameter_head(self.state_net(state))
        center = torch.tanh(raw[:, 0:2]).permute(0, 2, 3, 1).contiguous() * self.max_center_displacement
        radius = (
            self.min_radius + (self.max_radius - self.min_radius) * torch.sigmoid(raw[:, 2:4])
        ).permute(0, 2, 3, 1).contiguous()
        orientation = torch.tanh(raw[:, 4:5]).permute(0, 2, 3, 1).contiguous() * math.pi
        residual = (
            torch.tanh(raw[:, 5:])
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(batch, x.shape[2], x.shape[3], self.k, 2)
            * self.max_residual_offset
        )
        return DynamicWindowParameters(center, radius, orientation, residual, coordinates=raw.new_empty(0))


class DynamicWindowAttention(nn.Module):
    """Sparse attention over each query's predicted structured support."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int = 4,
        k: int = 8,
        condition_channels: int = 3,
        max_turns: int = 8,
        max_center_displacement: float = 0.25,
        min_radius: float = 0.03,
        max_radius: float = 0.55,
        max_residual_offset: float = 0.12,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.heads = max(1, min(int(heads), self.channels))
        while self.channels % self.heads != 0:
            self.heads -= 1
        self.head_dim = self.channels // self.heads
        self.k = int(k)
        self.generator = DynamicWindowGenerator(
            self.channels,
            condition_channels=condition_channels,
            k=self.k,
            max_turns=max_turns,
            max_center_displacement=max_center_displacement,
            min_radius=min_radius,
            max_radius=max_radius,
            max_residual_offset=max_residual_offset,
        )
        self.query = nn.Conv2d(self.channels, self.channels, 1)
        self.key = nn.Conv2d(self.channels, self.channels, 1)
        self.value = nn.Conv2d(self.channels, self.channels, 1)
        self.output = nn.Conv2d(self.channels, self.channels, 1)

    @staticmethod
    def _base_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1).view(1, height, width, 1, 2)

    def generate_coordinates(
        self,
        x: Tensor,
        *,
        condition: Tensor | None = None,
        audit_map: Tensor | None = None,
        turn_index: int | Tensor = 0,
        iteration_index: int | Tensor = 0,
    ) -> DynamicWindowParameters:
        parameters = self.generator(
            x,
            condition=condition,
            audit_map=audit_map,
            turn_index=turn_index,
            iteration_index=iteration_index,
        )
        center = parameters.center_displacement[:, :, :, None, :]
        radius = parameters.radius[:, :, :, None, :]
        theta = parameters.orientation[:, :, :, None, :]
        canonical = self.generator.canonical_support.to(device=x.device, dtype=x.dtype).view(1, 1, 1, self.k, 2)
        support = canonical * radius
        cos_theta = theta[..., 0:1].cos()
        sin_theta = theta[..., 0:1].sin()
        support_x = support[..., 0:1] * cos_theta - support[..., 1:2] * sin_theta
        support_y = support[..., 0:1] * sin_theta + support[..., 1:2] * cos_theta
        support = torch.cat([support_x, support_y], dim=-1)
        residual = parameters.residual_offsets
        base = self._base_grid(x.shape[-2], x.shape[-1], x.device, x.dtype)
        coordinates = (base + center + support + residual).clamp(-1.0, 1.0)
        parameters.coordinates = coordinates
        return parameters

    def forward(
        self,
        x: Tensor,
        *,
        condition: Tensor | None = None,
        audit_map: Tensor | None = None,
        turn_index: int | Tensor = 0,
        iteration_index: int | Tensor = 0,
        return_metadata: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        parameters = self.generate_coordinates(
            x,
            condition=condition,
            audit_map=audit_map,
            turn_index=turn_index,
            iteration_index=iteration_index,
        )
        batch, _, height, width = x.shape
        grid = parameters.coordinates.reshape(batch, height, width * self.k, 2)
        key_values = self.key(x)
        value_values = self.value(x)
        sampled_key = F.grid_sample(key_values, grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled_value = F.grid_sample(value_values, grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled_key = sampled_key.view(batch, self.channels, height, width, self.k)
        sampled_value = sampled_value.view(batch, self.channels, height, width, self.k)
        query = self.query(x).view(batch, self.heads, self.head_dim, height, width)
        key = sampled_key.view(batch, self.heads, self.head_dim, height, width, self.k)
        value = sampled_value.view(batch, self.heads, self.head_dim, height, width, self.k)
        attention = (query.unsqueeze(-1) * key).sum(dim=2) / math.sqrt(float(self.head_dim))
        attention = attention.softmax(dim=-1)
        aggregate = (attention.unsqueeze(2) * value).sum(dim=-1).reshape(batch, self.channels, height, width)
        output = self.output(aggregate)
        if not return_metadata:
            return output
        return output, {
            "coordinates": parameters.coordinates,
            "center_displacement": parameters.center_displacement,
            "radius": parameters.radius,
            "orientation": parameters.orientation,
            "residual_offsets": parameters.residual_offsets,
            "attention": attention,
        }


DynamicWindow = DynamicWindowAttention
