"""One shared recurrent Annotation Expert for soft annotation refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .dynamic_window import DynamicWindowAttention


def annotation_entropy(logits_or_probs: Tensor, eps: float = 1e-8) -> Tensor:
    if logits_or_probs.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(logits_or_probs.shape)}")
    if logits_or_probs.min().detach() < 0 or logits_or_probs.max().detach() > 1:
        probs = logits_or_probs.softmax(dim=1)
    else:
        probs = logits_or_probs / logits_or_probs.sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=1, keepdim=True)
    return entropy / torch.log(torch.tensor(float(max(logits_or_probs.shape[1], 2)), device=logits_or_probs.device, dtype=logits_or_probs.dtype))


@dataclass
class AnnotationExpertOutput:
    delta_logits: Tensor
    update_gate: Tensor
    candidate_logits: Tensor
    depth: int
    window_metadata: dict[str, Tensor] | None = None

    @property
    def A_candidate(self) -> Tensor:
        return self.candidate_logits

    def __iter__(self):
        yield self.delta_logits
        yield self.update_gate


class AnnotationExpert(nn.Module):
    """Shared-weight recurrent residual updater.

    The dynamic window is conditioned by the projected previous local audit
    evidence.  A high audit value is not assigned a hand-written window size;
    the generator learns all support parameters from the joint state.
    """

    def __init__(
        self,
        feature_channels: int = 96,
        num_classes: int = 4,
        *,
        audit_channels: int = 3,
        window_k: int = 8,
        max_turns: int = 3,
    ) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.num_classes = int(num_classes)
        self.audit_channels = int(audit_channels)
        self.max_turns = int(max_turns)
        input_channels = self.feature_channels + self.num_classes + 1 + self.audit_channels
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, self.feature_channels, 1),
            nn.GroupNorm(8 if self.feature_channels % 8 == 0 else 1, self.feature_channels),
            nn.GELU(),
        )
        # Exactly one shared dynamic-window refinement block is reused for all
        # recurrent turns and all depth iterations.
        self.refinement_block = DynamicWindowAttention(
            self.feature_channels,
            k=int(window_k),
            condition_channels=self.audit_channels,
            max_turns=max(self.max_turns, 3),
        )
        self.refinement_residual = nn.Sequential(
            nn.Conv2d(self.feature_channels, self.feature_channels, 3, padding=1),
            nn.GroupNorm(8 if self.feature_channels % 8 == 0 else 1, self.feature_channels),
            nn.GELU(),
            nn.Conv2d(self.feature_channels, self.feature_channels, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, self.feature_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.feature_channels // 2, self.num_classes, 1),
        )
        self.gate_head = nn.Conv2d(self.feature_channels, 1, 1)
        self.last_window_metadata: dict[str, Tensor] | None = None

    def forward(
        self,
        shared_features: Tensor,
        annotation_logits: Tensor,
        entropy: Tensor | None = None,
        previous_audit_evidence: Tensor | None = None,
        *,
        turn_index: int | Tensor = 0,
        iteration_index: int | Tensor = 0,
        return_metadata: bool = False,
    ) -> AnnotationExpertOutput:
        if shared_features.ndim != 4 or annotation_logits.ndim != 4:
            raise ValueError("shared_features and annotation_logits must be four-dimensional")
        if annotation_logits.shape[1] != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} annotation classes, got {annotation_logits.shape[1]}")
        spatial = shared_features.shape[-2:]
        annotation_low = F.interpolate(annotation_logits, size=spatial, mode="bilinear", align_corners=False)
        entropy = annotation_entropy(annotation_logits) if entropy is None else entropy
        entropy_low = F.interpolate(entropy, size=spatial, mode="bilinear", align_corners=False)
        if previous_audit_evidence is None:
            audit_low = shared_features.new_zeros((shared_features.shape[0], self.audit_channels, *spatial))
        else:
            if previous_audit_evidence.ndim != 4:
                raise ValueError("previous_audit_evidence must be [B,C,H,W]")
            audit_low = F.interpolate(previous_audit_evidence, size=spatial, mode="bilinear", align_corners=False)
            if audit_low.shape[1] == 1:
                audit_low = audit_low.expand(-1, self.audit_channels, -1, -1)
            elif audit_low.shape[1] != self.audit_channels:
                audit_low = audit_low[:, : self.audit_channels]
                if audit_low.shape[1] < self.audit_channels:
                    padding = audit_low.new_zeros(
                        audit_low.shape[0],
                        self.audit_channels - audit_low.shape[1],
                        audit_low.shape[2],
                        audit_low.shape[3],
                    )
                    audit_low = torch.cat([audit_low, padding], dim=1)
        state = self.input_projection(torch.cat([shared_features, annotation_low, entropy_low, audit_low], dim=1))
        if torch.is_tensor(turn_index):
            turn_value = int(turn_index.reshape(-1)[0].item())
        else:
            turn_value = int(turn_index)
        depth = min(turn_value + 1, 3)
        depth = max(depth, 1)
        metadata: dict[str, Tensor] | None = None
        for iteration in range(depth):
            iteration_value = iteration
            if isinstance(iteration_index, int):
                iteration_value += int(iteration_index)
            window_output, window_metadata = self.refinement_block(
                state,
                condition=audit_low,
                turn_index=turn_index,
                iteration_index=iteration_value,
                return_metadata=True,
            )
            state = state + self.refinement_residual(window_output)
            if return_metadata:
                metadata = window_metadata
        delta_low = self.delta_head(state)
        gate_low = self.gate_head(state)
        delta_logits = F.interpolate(delta_low, size=annotation_logits.shape[-2:], mode="bilinear", align_corners=False)
        update_gate = F.interpolate(gate_low, size=annotation_logits.shape[-2:], mode="bilinear", align_corners=False)
        candidate_logits = annotation_logits + torch.sigmoid(update_gate) * delta_logits
        self.last_window_metadata = metadata
        return AnnotationExpertOutput(delta_logits, update_gate, candidate_logits, depth, metadata)


SharedAnnotationExpert = AnnotationExpert
