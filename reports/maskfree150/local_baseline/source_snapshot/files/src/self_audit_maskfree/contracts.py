"""Shared types. Fitting callers never receive sealed observation intensities.

All image tensors use channel/depth first axes. Data constructs capability views;
fit code accepts FittingView, score code accepts ScoringView. Hidden intensities
must be physically zeroed before constructing a fitting view, including context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

VERSION = "maskfree150.v1"
SEMANTIC_ORDER = {0: "BG", 1: "RV", 2: "MYO", 3: "LV"}

FITTING_METADATA_KEYS = frozenset({
    "unit_id", "study_id", "patient_id", "dataset", "split", "manifest_id", "protocol",
    "protocol_reason", "partition_id", "partition_spec", "image_size", "native_hw", "depth",
    "depth_axis", "slice_index", "frame_index", "frame_selection_rule", "source_format",
    "native_geometry", "native_geometry_reason", "spacing_valid", "orientation", "zooms",
    "native_affine", "export_grid", "native_grid_export", "inverse_transform", "normalization",
    "support_counts", "capabilities", "cohort_provenance", "acquisition_id", "mask_inputs_used",
    "deployment_only", "volume_id", "num_slices", "epoch", "checkpoint", "view",
    "inplane_left_axis", "geometry_valid", "spacing", "spacing_units", "spacing_mm", "temporal_spacing",
})


@dataclass(frozen=True)
class FittingView:
    image: torch.Tensor  # [1,H,W], normalized ONLY from fit intensities, zero elsewhere
    support: torch.Tensor  # [H,W] bool
    context: torch.Tensor  # [3,H,W]; all neighbor slices have withheld blocks zeroed
    study_id: str
    unit_id: str
    protocol: str = "spatial_predictive"
    metadata: dict[str, Any] = field(default_factory=dict)
    partition_id: str = ""

    def validate(self) -> None:
        unknown = set(self.metadata) - FITTING_METADATA_KEYS
        if unknown:
            raise ValueError(f"unsupported fitting metadata keys: {sorted(unknown)}")
        if self.image.ndim != 3 or self.image.shape[0] != 1:
            raise ValueError("image must be [1,H,W]")
        if self.support.dtype != torch.bool or self.support.shape != self.image.shape[1:]:
            raise ValueError("support must be bool [H,W]")
        if self.context.shape != (3, *self.support.shape):
            raise ValueError("context must be [3,H,W]")
        if not torch.isfinite(self.image).all() or not torch.isfinite(self.context).all():
            raise ValueError("nonfinite fitting input")
        if torch.any(self.image[:, ~self.support] != 0):
            raise ValueError("fitting image contains hidden intensities")
        if torch.any(self.context[:, ~self.support] != 0):
            raise ValueError("context contains hidden intensities")
        if not self.support.any():
            raise ValueError("empty fitting support")


@dataclass(frozen=True)
class ScoringView:
    image: torch.Tensor  # [1,H,W], zero outside this scoring support
    support: torch.Tensor  # [H,W] bool
    study_id: str
    unit_id: str
    role: Literal["select", "verify"]
    partition_id: str = ""

    def validate(self) -> None:
        if self.role not in ("select", "verify"):
            raise ValueError("invalid observation role")
        if self.image.shape != (1, *self.support.shape) or self.support.dtype != torch.bool:
            raise ValueError("scoring shapes invalid")
        if not torch.isfinite(self.image).all() or torch.any(self.image[:, ~self.support] != 0):
            raise ValueError("invalid scoring intensities")


@dataclass
class Hypothesis:
    candidate_id: str
    labels: torch.Tensor  # [H,W] long, fixed named semantic order; still a draft if unresolved
    probabilities: torch.Tensor  # [4,H,W], normalized
    validity: torch.Tensor  # [H,W] float in [0,1]; ambiguity is NOT background
    source: str
    semantic_unresolved: bool = False
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceScore:
    nll_sum: float
    count: int
    normalized_nll: float | None
    complexity: float
    prior: float
    total: float | None
    role: str
    available: bool = True
    reason: str | None = None


@dataclass
class FittedHypothesis:
    hypothesis: Hypothesis
    parameters: dict[str, Any]
    fit_score: EvidenceScore
    fitting_steps: int
    capacity: int
    study_id: str
    unit_id: str
    partition_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditResult:
    initial: Hypothesis
    selected: Hypothesis
    bank: list[Hypothesis]
    fitted: list[FittedHypothesis]
    scores: list[EvidenceScore]
    regional_margin: torch.Tensor  # [H,W], not a correctness probability
    validity: torch.Tensor
    trace: dict[str, Any]


@dataclass
class TrainingUnit:
    fitting: FittingView
    selection: ScoringView
    record: dict[str, Any]  # no images or masks, no verify intensities


def hypothesis_from_labels(labels: torch.Tensor, candidate_id: str, source: str,
                           validity: torch.Tensor | None = None, **kwargs: Any) -> Hypothesis:
    labels = labels.detach().long()
    probs = torch.nn.functional.one_hot(labels, 4).permute(2, 0, 1).float()
    return Hypothesis(candidate_id, labels, probs,
                      torch.ones_like(labels, dtype=torch.float32) if validity is None else validity,
                      source, **kwargs)
