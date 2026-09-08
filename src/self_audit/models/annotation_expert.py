"""One shared recurrent Annotation Expert for soft annotation refinement."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .dynamic_window import DynamicWindowAttention


MODEL_ENTROPY_VERSION: str = "2.0.0"


def entropy_from_logits(logits: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute normalized Shannon entropy from unnormalized logits.

    The channel dimension (dim=1) represents classes.
    Computes Shannon entropy normalized by ln(num_classes) so that:
      - Uniform distribution produces entropy ≈ 1.0
      - One-hot (pure certainty) distribution produces entropy = 0.0

    This implementation is strictly invariant to constant shifts along the
    channel dimension (shift invariance) and invariant to other batch items
    (batch-composition invariance).

    Computations are performed in float32 for float16 and bfloat16 inputs
    to ensure numerical stability against overflow/underflow, while preserving
    the caller's original output dtype, float64 precision, and backpropagation gradients.

    Parameters
    ----------
    logits : Tensor
        Unnormalized logit tensor of shape [B, C, H, W].
    eps : float, default 1e-8
        Numerical stability epsilon (reserved for API parity).

    Returns
    -------
    Tensor
        Normalized entropy tensor of shape [B, 1, H, W] in range [0, 1],
        matching the device and dtype of the input logits.
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected 4D logits tensor [B, C, H, W], got shape {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError("Logits must contain only finite values (found NaN or Inf)")
    num_classes = logits.shape[1]
    if num_classes <= 1:
        return logits.new_zeros((logits.shape[0], 1, logits.shape[2], logits.shape[3]))

    orig_dtype = logits.dtype
    compute_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    logits_comp = logits.to(compute_dtype)

    # Invariant to constant shifts along channel dimension via log_softmax
    probs = F.softmax(logits_comp, dim=1)
    log_probs = F.log_softmax(logits_comp, dim=1)
    # Clamp extreme negative infinities to avoid 0 * -inf = NaN in pathological logit differences
    log_probs_safe = torch.nan_to_num(log_probs, neginf=-1e30)
    p_log_p = torch.where(probs > 0, probs * log_probs_safe, torch.zeros_like(probs))
    entropy = -p_log_p.sum(dim=1, keepdim=True)
    normalizer = float(math.log(num_classes))
    normalized_entropy = entropy / normalizer
    return normalized_entropy.to(orig_dtype)


def entropy_from_probabilities(
    probs: Tensor,
    eps: float = 1e-8,
    *,
    atol: float = 1e-3,
) -> Tensor:
    """Compute normalized Shannon entropy from probability simplex tensors.

    Validates that the input represents a valid probability distribution on
    the simplex:
      1. 4D shape [B, C, H, W]
      2. Strictly finite values (no NaNs or Infs)
      3. Valid probability range [0, 1] within numerical tolerance `atol`
      4. Valid per-pixel probability simplex (sum over classes equals 1.0 within `atol`)

    Does NOT silently renormalize invalid inputs.

    Computations are performed in float32 for float16 and bfloat16 inputs
    to prevent underflow of eps or overflow during exponentiation/logarithms,
    while preserving the caller's original output dtype, float64 precision,
    and backpropagation gradients.

    Parameters
    ----------
    probs : Tensor
        Probability tensor of shape [B, C, H, W].
    eps : float, default 1e-8
        Numerical clamping epsilon for logarithm in compute precision.
    atol : float, default 1e-3
        Absolute numerical tolerance for range [0, 1] and simplex sum == 1.0.

    Returns
    -------
    Tensor
        Normalized entropy tensor of shape [B, 1, H, W] in range [0, 1],
        matching the device and dtype of the input probabilities.
    """
    if probs.ndim != 4:
        raise ValueError(f"Expected 4D probabilities tensor [B, C, H, W], got shape {tuple(probs.shape)}")
    if not torch.isfinite(probs).all():
        raise ValueError("Probabilities must contain only finite values (found NaN or Inf)")
    num_classes = probs.shape[1]
    if num_classes <= 1:
        return probs.new_zeros((probs.shape[0], 1, probs.shape[2], probs.shape[3]))

    orig_dtype = probs.dtype
    compute_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    probs_comp = probs.to(compute_dtype)

    if (probs_comp < -atol).any() or (probs_comp > 1.0 + atol).any():
        min_val = float(probs_comp.min().detach().item())
        max_val = float(probs_comp.max().detach().item())
        raise ValueError(
            f"Probabilities out of valid range [0, 1] (min={min_val:.6f}, max={max_val:.6f})"
        )

    pixel_sums = probs_comp.sum(dim=1, keepdim=True)
    discrepancy = torch.abs(pixel_sums - 1.0)
    if (discrepancy > atol).any():
        max_disc = float(discrepancy.max().detach().item())
        raise ValueError(
            f"Probabilities do not sum to 1.0 along channel dimension (max deviation={max_disc:.6f} > atol={atol}). "
            "Silent renormalization of invalid inputs is prohibited."
        )

    # Standard convention 0 * log(0) -> 0 limit for Shannon entropy
    safe_probs = torch.clamp(probs_comp, min=eps, max=1.0)
    log_probs = torch.log(safe_probs)
    p_log_p = torch.where(probs_comp > 0, probs_comp * log_probs, torch.zeros_like(probs_comp))
    entropy = -p_log_p.sum(dim=1, keepdim=True)
    normalizer = float(math.log(num_classes))
    normalized_entropy = entropy / normalizer
    return normalized_entropy.to(orig_dtype)


def annotation_entropy(
    x: Tensor,
    eps: float = 1e-8,
    *,
    input_type: str = "logits",
    atol: float = 1e-3,
) -> Tensor:
    """Compute normalized Shannon entropy from logits or probabilities.

    .. note::
        Range-based heuristic dispatch has been removed. Callers must use
        explicit input typing: 'logits' (default) or 'probabilities'.
        Direct callers should prefer :func:`entropy_from_logits` or
        :func:`entropy_from_probabilities`.

    Parameters
    ----------
    x : Tensor
        Input 4D tensor [B, C, H, W].
    eps : float, default 1e-8
        Numerical stability epsilon.
    input_type : {"logits", "probabilities", "probs"}, default "logits"
        Explicit input representation type. Range-based heuristic dispatch is not performed.
    atol : float, default 1e-3
        Simplex tolerance when input_type is 'probabilities'.

    Returns
    -------
    Tensor
        Normalized entropy tensor [B, 1, H, W].
    """
    clean_type = str(input_type).lower().strip()
    if clean_type in ("logits", "logit"):
        return entropy_from_logits(x, eps=eps)
    elif clean_type in ("probabilities", "probs", "probability"):
        return entropy_from_probabilities(x, eps=eps, atol=atol)
    else:
        raise ValueError(
            f"Invalid input_type: {input_type!r}. Must be 'logits' or 'probabilities'. "
            "Heuristic range-based dispatch has been removed."
        )


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

    entropy_version: str = MODEL_ENTROPY_VERSION

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
        entropy = entropy_from_logits(annotation_logits) if entropy is None else entropy
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
