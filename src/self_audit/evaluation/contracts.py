"""Metric contracts and sufficient statistics primitives for Self-Audit.

This module defines explicit, versioned metric contracts that prevent metric mixing:
- ``audit_target_legacy_one_v1``: historical training target contract where both-empty
  foreground classes score 1.0 (empty_policy="legacy_one").
- ``foreground_dice_exclude_v1``: canonical evaluation contract where both-empty foreground
  classes are excluded (empty_policy="exclude", score is NaN).

Metric spaces (``slice_proxy``, ``volume_resized``, ``volume_native``) are strictly
separated and must not be interchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import torch

from ..audit.semantics import (
    AUDIT_TARGET_LEGACY_ONE_V1,
    DEFAULT_NEUTRAL_MARGIN,
    FOREGROUND_DICE_EXCLUDE_V1,
    METRIC_SPACES,
    METRIC_SPACE_SLICE_PROXY,
    METRIC_SPACE_VOLUME_NATIVE,
    METRIC_SPACE_VOLUME_RESIZED,
    empty_class_score,
    macro_mean,
    resolve_empty_policy,
    resolve_neutral_margin,
)


class ContractMismatchError(ValueError):
    """Raised when two metric contracts cannot be combined or compared."""


@dataclass(frozen=True)
class MetricContract:
    """Explicit, versioned metric contract definition."""

    name: str
    version: int = 1
    metric_space: str = METRIC_SPACE_SLICE_PROXY
    empty_policy: str = "exclude"
    classes: tuple[int, ...] = (1, 2, 3)
    include_background: bool = False
    neutral_margin: float = DEFAULT_NEUTRAL_MARGIN
    aggregation: str = "slice_macro_mean"
    description: str = ""

    def __post_init__(self) -> None:
        if self.metric_space not in METRIC_SPACES:
            raise ValueError(
                f"metric_space must be one of {list(METRIC_SPACES)}, got {self.metric_space!r}"
            )
        resolve_empty_policy(self.empty_policy)
        resolve_neutral_margin(self.neutral_margin)
        if not self.classes:
            raise ValueError("classes must not be empty")

    def validate_compatibility(self, other: MetricContract) -> None:
        """Validate that another contract is compatible for combined operations."""
        if not isinstance(other, MetricContract):
            raise TypeError(f"Expected MetricContract, got {type(other)!r}")
        mismatches: list[str] = []
        if self.name != other.name:
            mismatches.append(f"name ({self.name!r} vs {other.name!r})")
        if self.version != other.version:
            mismatches.append(f"version ({self.version} vs {other.version})")
        if self.metric_space != other.metric_space:
            mismatches.append(f"metric_space ({self.metric_space!r} vs {other.metric_space!r})")
        if self.empty_policy != other.empty_policy:
            mismatches.append(f"empty_policy ({self.empty_policy!r} vs {other.empty_policy!r})")
        if self.classes != other.classes:
            mismatches.append(f"classes ({self.classes} vs {other.classes})")
        if self.include_background != other.include_background:
            mismatches.append(
                f"include_background ({self.include_background} vs {other.include_background})"
            )
        if abs(self.neutral_margin - other.neutral_margin) > 1e-7:
            mismatches.append(
                f"neutral_margin ({self.neutral_margin} vs {other.neutral_margin})"
            )
        if self.aggregation != other.aggregation:
            mismatches.append(
                f"aggregation ({self.aggregation!r} vs {other.aggregation!r})"
            )
        if mismatches:
            joined = "; ".join(mismatches)
            raise ContractMismatchError(
                f"Metric contract mismatch: {joined}. "
                "Cannot perform arithmetic or replaying across incompatible metric contracts."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "metric_space": self.metric_space,
            "empty_policy": self.empty_policy,
            "classes": list(self.classes),
            "include_background": self.include_background,
            "neutral_margin": self.neutral_margin,
            "aggregation": self.aggregation,
            "description": self.description,
        }

    REQUIRED_CONTRACT_KEYS: tuple[str, ...] = (
        "name",
        "version",
        "metric_space",
        "empty_policy",
        "classes",
        "neutral_margin",
        "aggregation",
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricContract:
        if not isinstance(data, Mapping):
            raise TypeError(f"Expected mapping for contract dictionary, got {type(data)!r}")

        missing = [k for k in cls.REQUIRED_CONTRACT_KEYS if k not in data]
        if missing:
            raise ContractMismatchError(
                f"Contract dictionary missing required fields: {', '.join(missing)}"
            )

        name = str(data["name"]).strip()
        if not name:
            raise ContractMismatchError("Contract dictionary 'name' cannot be empty")

        version = data["version"]
        if not isinstance(version, int) or isinstance(version, bool) or version != 1:
            raise ContractMismatchError(
                f"Unsupported contract version {version!r}; only version 1 is supported"
            )

        metric_space = str(data["metric_space"])
        if metric_space not in METRIC_SPACES:
            raise ContractMismatchError(
                f"Unsupported metric_space {metric_space!r}; must be one of {list(METRIC_SPACES)}"
            )

        empty_policy = resolve_empty_policy(data["empty_policy"])
        neutral_margin = resolve_neutral_margin(data["neutral_margin"])

        classes_raw = data["classes"]
        if not isinstance(classes_raw, (list, tuple)) or not classes_raw:
            raise ContractMismatchError(
                f"classes must be a non-empty list or tuple, got {classes_raw!r}"
            )
        classes = tuple(int(c) for c in classes_raw)

        aggregation = str(data["aggregation"]).strip()
        if not aggregation:
            raise ContractMismatchError("Contract dictionary 'aggregation' cannot be empty")

        include_background = bool(data.get("include_background", False))
        description = str(data.get("description", ""))

        # Strict validation: compare against registered definition if known name
        if name in _REGISTRY:
            reg = _REGISTRY[name]
            mismatches: list[str] = []
            if empty_policy != reg.empty_policy:
                mismatches.append(f"empty_policy ({empty_policy!r} vs expected {reg.empty_policy!r})")
            if version != reg.version:
                mismatches.append(f"version ({version} vs expected {reg.version})")
            if metric_space != reg.metric_space:
                mismatches.append(f"metric_space ({metric_space!r} vs expected {reg.metric_space!r})")
            if classes != reg.classes:
                mismatches.append(f"classes ({classes} vs expected {reg.classes})")
            if abs(neutral_margin - reg.neutral_margin) > 1e-7:
                mismatches.append(f"neutral_margin ({neutral_margin} vs expected {reg.neutral_margin})")
            if aggregation != reg.aggregation:
                mismatches.append(f"aggregation ({aggregation!r} vs expected {reg.aggregation!r})")
            if mismatches:
                raise ContractMismatchError(
                    f"Forged or invalid contract definition for registered name {name!r}: {'; '.join(mismatches)}"
                )

        return cls(
            name=name,
            version=version,
            metric_space=metric_space,
            empty_policy=empty_policy,
            classes=classes,
            include_background=include_background,
            neutral_margin=neutral_margin,
            aggregation=aggregation,
            description=description,
        )


#: Training target contract: both-empty foreground classes score 1.0 (legacy_one).
AUDIT_TARGET_LEGACY_ONE_V1_CONTRACT = MetricContract(
    name=AUDIT_TARGET_LEGACY_ONE_V1,
    version=1,
    metric_space=METRIC_SPACE_SLICE_PROXY,
    empty_policy="legacy_one",
    classes=(1, 2, 3),
    include_background=False,
    neutral_margin=DEFAULT_NEUTRAL_MARGIN,
    aggregation="slice_macro_mean",
    description=(
        "Historical training target contract: both-empty foreground classes score 1.0, "
        "2-D slice proxy."
    ),
)

#: Primary evaluation contract: both-empty foreground classes are excluded (NaN).
FOREGROUND_DICE_EXCLUDE_V1_CONTRACT = MetricContract(
    name=FOREGROUND_DICE_EXCLUDE_V1,
    version=1,
    metric_space=METRIC_SPACE_SLICE_PROXY,
    empty_policy="exclude",
    classes=(1, 2, 3),
    include_background=False,
    neutral_margin=DEFAULT_NEUTRAL_MARGIN,
    aggregation="slice_macro_mean",
    description=(
        "Canonical evaluation contract: both-empty foreground classes are excluded (NaN); "
        "one-sided empty classes score 0.0."
    ),
)

#: Volume evaluation contract on resized network grid (256x256 in-plane).
FOREGROUND_DICE_VOLUME_RESIZED_V1_CONTRACT = MetricContract(
    name="foreground_dice_volume_resized_v1",
    version=1,
    metric_space=METRIC_SPACE_VOLUME_RESIZED,
    empty_policy="exclude",
    classes=(1, 2, 3),
    include_background=False,
    neutral_margin=DEFAULT_NEUTRAL_MARGIN,
    aggregation="volume_sufficient_stats",
    description="Volume-level evaluation contract on resized grid, aggregated from sufficient statistics.",
)

#: Volume evaluation contract on native acquisition grid.
FOREGROUND_DICE_VOLUME_NATIVE_V1_CONTRACT = MetricContract(
    name="foreground_dice_volume_native_v1",
    version=1,
    metric_space=METRIC_SPACE_VOLUME_NATIVE,
    empty_policy="exclude",
    classes=(1, 2, 3),
    include_background=False,
    neutral_margin=DEFAULT_NEUTRAL_MARGIN,
    aggregation="volume_sufficient_stats",
    description="Volume-level evaluation contract on native grid with physical spacing.",
)

_REGISTRY: dict[str, MetricContract] = {
    AUDIT_TARGET_LEGACY_ONE_V1: AUDIT_TARGET_LEGACY_ONE_V1_CONTRACT,
    FOREGROUND_DICE_EXCLUDE_V1: FOREGROUND_DICE_EXCLUDE_V1_CONTRACT,
    "foreground_dice_volume_resized_v1": FOREGROUND_DICE_VOLUME_RESIZED_V1_CONTRACT,
    "foreground_dice_volume_native_v1": FOREGROUND_DICE_VOLUME_NATIVE_V1_CONTRACT,
}


def resolve_metric_contract(contract: MetricContract | str | None = None) -> MetricContract:
    """Resolve a contract name or object to a validated MetricContract instance."""
    if contract is None:
        return FOREGROUND_DICE_EXCLUDE_V1_CONTRACT
    if isinstance(contract, MetricContract):
        return contract
    name = str(contract).strip()
    if name in _REGISTRY:
        return _REGISTRY[name]
    raise ValueError(
        f"Unknown metric contract {contract!r}. Supported contracts: {sorted(_REGISTRY.keys())}"
    )


def validate_contract_compatibility(
    contract_a: MetricContract | str,
    contract_b: MetricContract | str,
) -> None:
    """Raise ContractMismatchError if two contracts are not compatible."""
    resolved_a = resolve_metric_contract(contract_a)
    resolved_b = resolve_metric_contract(contract_b)
    resolved_a.validate_compatibility(resolved_b)


def validate_contract(contract: Any) -> MetricContract:
    """Validate a MetricContract or deserialized mapping against known semantics."""
    if isinstance(contract, MetricContract):
        if contract.version != 1:
            raise ContractMismatchError(
                f"Unsupported contract version {contract.version!r}; only version 1 is supported"
            )
        if contract.metric_space not in METRIC_SPACES:
            raise ContractMismatchError(
                f"Unsupported metric_space {contract.metric_space!r}; must be one of {list(METRIC_SPACES)}"
            )
        resolve_empty_policy(contract.empty_policy)
        resolve_neutral_margin(contract.neutral_margin)
        if contract.name in _REGISTRY:
            contract.validate_compatibility(_REGISTRY[contract.name])
        return contract
    if isinstance(contract, Mapping):
        return MetricContract.from_dict(contract)
    if isinstance(contract, str):
        return resolve_metric_contract(contract)
    raise TypeError(f"Expected MetricContract, Mapping, or str, got {type(contract)!r}")


# --------------------------------------------------------------------------
# Sufficient statistics
# --------------------------------------------------------------------------


@dataclass
class SufficientStatistics:
    """Class-wise True Positive, False Positive, and False Negative voxel counts.

    These counts form the minimal sufficient statistics for exact volumetric or
    multi-slice Dice recomputation without loss of information.
    """

    tp: dict[int, int]
    fp: dict[int, int]
    fn: dict[int, int]

    def merged_with(self, other: SufficientStatistics) -> SufficientStatistics:
        """Merge statistics across slices to compute exact volume-level statistics."""
        all_classes = sorted(set(self.tp.keys()) | set(other.tp.keys()))
        return SufficientStatistics(
            tp={c: self.tp.get(c, 0) + other.tp.get(c, 0) for c in all_classes},
            fp={c: self.fp.get(c, 0) + other.fp.get(c, 0) for c in all_classes},
            fn={c: self.fn.get(c, 0) + other.fn.get(c, 0) for c in all_classes},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": {int(k): int(v) for k, v in self.tp.items()},
            "fp": {int(k): int(v) for k, v in self.fp.items()},
            "fn": {int(k): int(v) for k, v in self.fn.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SufficientStatistics:
        return cls(
            tp={int(k): int(v) for k, v in data.get("tp", {}).items()},
            fp={int(k): int(v) for k, v in data.get("fp", {}).items()},
            fn={int(k): int(v) for k, v in data.get("fn", {}).items()},
        )


def _numpy_labels(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if getattr(value, "is_floating_point", lambda: False)():
            value = value.float()
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim >= 4:
        array = array.argmax(axis=1)
    return array.astype(np.int64, copy=False)


def compute_sufficient_statistics(
    prediction: Any,
    target: Any,
    classes: tuple[int, ...] = (1, 2, 3),
) -> SufficientStatistics:
    """Compute sufficient statistics for a single sample or volume."""
    pred = _numpy_labels(prediction)
    true = _numpy_labels(target)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    tp: dict[int, int] = {}
    fp: dict[int, int] = {}
    fn: dict[int, int] = {}
    for cls in classes:
        p = pred == cls
        t = true == cls
        tp[cls] = int((p & t).sum())
        fp[cls] = int((p & ~t).sum())
        fn[cls] = int((~p & t).sum())
    return SufficientStatistics(tp=tp, fp=fp, fn=fn)


def compute_batch_sufficient_statistics(
    prediction: Any,
    target: Any,
    classes: tuple[int, ...] = (1, 2, 3),
) -> list[SufficientStatistics]:
    """Compute sufficient statistics per sample for a batch [B, H, W]."""
    pred = _numpy_labels(prediction)
    true = _numpy_labels(target)
    if pred.ndim == 2:
        pred = pred[None]
    if true.ndim == 2:
        true = true[None]
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    batch_size = pred.shape[0]
    return [
        compute_sufficient_statistics(pred[i], true[i], classes=classes)
        for i in range(batch_size)
    ]


def compute_dice_from_stats(
    stats: SufficientStatistics,
    contract: MetricContract | str | None = None,
) -> tuple[dict[int, float], float]:
    """Compute class-wise Dice and macro Dice from sufficient statistics under a contract.

    Empty policy rules:
    - Both empty (tp=0, fp=0, fn=0):
      under exclude: nan (excluded from macro mean)
      under legacy_one: 1.0
      under zero: 0.0
    - One-sided empty (tp=0, fp+fn > 0):
      always 0.0 (total miss, never excluded).
    """
    resolved_contract = resolve_metric_contract(contract)
    both_empty_val = empty_class_score(resolved_contract.empty_policy)
    per_class: dict[int, float] = {}
    for cls in resolved_contract.classes:
        tp = stats.tp.get(cls, 0)
        fp = stats.fp.get(cls, 0)
        fn = stats.fn.get(cls, 0)
        denom = 2 * tp + fp + fn
        if denom == 0:
            per_class[cls] = both_empty_val
        elif tp == 0:
            per_class[cls] = 0.0
        else:
            per_class[cls] = float(2.0 * tp / denom)
    macro = macro_mean(list(per_class.values()))
    return per_class, macro


@dataclass
class StateScore:
    """Evaluation score of a single annotation state."""

    macro_dice: float
    per_class_dice: dict[int, float]
    stats: SufficientStatistics
    is_defined: bool
    contract: MetricContract


@dataclass
class TransitionScore:
    """Evaluation score of a transition from previous to candidate state."""

    q_previous: float
    q_candidate: float
    delta_dice: float
    stats_previous: SufficientStatistics
    stats_candidate: SufficientStatistics
    is_defined_previous: bool
    is_defined_candidate: bool
    is_defined_delta: bool
    contract: MetricContract


def score_state(
    prediction: Any,
    target: Any,
    contract: MetricContract | str | None = None,
) -> StateScore:
    """Score an annotation state against target under the specified contract."""
    resolved_contract = resolve_metric_contract(contract)
    stats = compute_sufficient_statistics(
        prediction, target, classes=resolved_contract.classes
    )
    per_class, macro = compute_dice_from_stats(stats, resolved_contract)
    is_defined = np.isfinite(macro)
    return StateScore(
        macro_dice=float(macro),
        per_class_dice=per_class,
        stats=stats,
        is_defined=bool(is_defined),
        contract=resolved_contract,
    )


def score_transition(
    previous: Any,
    candidate: Any,
    target: Any,
    contract: MetricContract | str | None = None,
) -> TransitionScore:
    """Score a state transition against target under the specified contract."""
    resolved_contract = resolve_metric_contract(contract)
    s_prev = score_state(previous, target, resolved_contract)
    s_cand = score_state(candidate, target, resolved_contract)
    is_defined_delta = s_prev.is_defined and s_cand.is_defined
    delta = (
        float(s_cand.macro_dice - s_prev.macro_dice)
        if is_defined_delta
        else float("nan")
    )
    return TransitionScore(
        q_previous=s_prev.macro_dice,
        q_candidate=s_cand.macro_dice,
        delta_dice=delta,
        stats_previous=s_prev.stats,
        stats_candidate=s_cand.stats,
        is_defined_previous=s_prev.is_defined,
        is_defined_candidate=s_cand.is_defined,
        is_defined_delta=is_defined_delta,
        contract=resolved_contract,
    )


def score_volume_from_stats(
    stats_list: Sequence[SufficientStatistics],
    contract: MetricContract | str | None = None,
) -> StateScore:
    """Aggregate slice-level sufficient statistics to compute exact volume Dice.

    Volume scorer guarantees the returned score is stamped with a volume-level
    metric space (e.g. ``METRIC_SPACE_VOLUME_RESIZED``), never ``slice_proxy``.
    """
    resolved_contract = resolve_metric_contract(contract)
    if resolved_contract.metric_space == METRIC_SPACE_SLICE_PROXY:
        # Promote to volume contract: volume scorer must not report slice_proxy contract
        resolved_contract = MetricContract(
            name=(
                "foreground_dice_volume_resized_v1"
                if resolved_contract.name == FOREGROUND_DICE_EXCLUDE_V1
                else f"{resolved_contract.name}_volume"
            ),
            version=resolved_contract.version,
            metric_space=METRIC_SPACE_VOLUME_RESIZED,
            empty_policy=resolved_contract.empty_policy,
            classes=resolved_contract.classes,
            include_background=resolved_contract.include_background,
            neutral_margin=resolved_contract.neutral_margin,
            aggregation="volume_sufficient_stats",
            description=f"Volume-level aggregation derived from {resolved_contract.name}",
        )
    if not stats_list:
        raise ValueError("Cannot score volume from zero slice statistics")
    volume_stats = stats_list[0]
    for s in stats_list[1:]:
        volume_stats = volume_stats.merged_with(s)
    per_class, macro = compute_dice_from_stats(volume_stats, resolved_contract)
    return StateScore(
        macro_dice=float(macro),
        per_class_dice=per_class,
        stats=volume_stats,
        is_defined=bool(np.isfinite(macro)),
        contract=resolved_contract,
    )


__all__ = [
    "AUDIT_TARGET_LEGACY_ONE_V1_CONTRACT",
    "ContractMismatchError",
    "FOREGROUND_DICE_EXCLUDE_V1_CONTRACT",
    "FOREGROUND_DICE_VOLUME_NATIVE_V1_CONTRACT",
    "FOREGROUND_DICE_VOLUME_RESIZED_V1_CONTRACT",
    "MetricContract",
    "StateScore",
    "SufficientStatistics",
    "TransitionScore",
    "compute_batch_sufficient_statistics",
    "compute_dice_from_stats",
    "compute_sufficient_statistics",
    "resolve_metric_contract",
    "score_state",
    "score_transition",
    "score_volume_from_stats",
    "validate_contract",
    "validate_contract_compatibility",
]
