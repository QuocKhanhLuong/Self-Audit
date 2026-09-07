"""Canonical transition semantics shared by training, evaluation, and reporting.

One decision margin governs every improve / neutral / regress judgement in the
project.  Before this module existed the same question was answered with three
different thresholds (``0.0`` in the metric layer, ``0.005`` in the audit loss
and stage decomposition, ``0.02`` inside the counterfactual generator), so a
transition could be reported as harmful by one component and neutral by
another.  Everything that classifies a signed quality change now imports
:data:`DEFAULT_NEUTRAL_MARGIN` and :func:`classify_delta` from here.

The generator's ``epsilon_neutral`` is deliberately *not* unified away: it is a
generation-time search tolerance, not a decision margin.  See
:func:`check_generation_tolerance`.
"""

from __future__ import annotations

from typing import Any, Literal
import warnings

import numpy as np


# --------------------------------------------------------------------------
# Canonical decision margin
# --------------------------------------------------------------------------

#: The single decision margin for signed transition quality (Dice units).
#: ``delta > +eps`` is BENEFICIAL, ``delta < -eps`` is HARMFUL, and
#: ``abs(delta) <= eps`` is NEUTRAL.  Neutral transitions are neither.
DEFAULT_NEUTRAL_MARGIN: float = 0.005

BENEFICIAL = 1
NEUTRAL = 0
HARMFUL = -1

TRANSITION_CLASS_NAMES = {BENEFICIAL: "BENEFICIAL", NEUTRAL: "NEUTRAL", HARMFUL: "HARMFUL"}


def resolve_neutral_margin(value: Any = None) -> float:
    """Validate and return a decision margin, defaulting to the canonical one."""

    if value is None:
        return float(DEFAULT_NEUTRAL_MARGIN)
    margin = float(value)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError(f"neutral_margin must be finite and non-negative, got {value!r}")
    return margin


def classify_delta(delta: Any, neutral_margin: float | None = None) -> np.ndarray:
    """Return ``+1`` beneficial, ``0`` neutral, ``-1`` harmful, elementwise.

    The boundary is closed on the neutral side: ``abs(delta) == eps`` is
    NEUTRAL, matching the ``abs(delta) <= eps`` contract.
    """

    margin = resolve_neutral_margin(neutral_margin)
    values = np.asarray(delta, dtype=np.float64)
    out = np.zeros(values.shape, dtype=np.int64)
    out[values > margin] = BENEFICIAL
    out[values < -margin] = HARMFUL
    return out


def beneficial_mask(delta: Any, neutral_margin: float | None = None) -> np.ndarray:
    return classify_delta(delta, neutral_margin) == BENEFICIAL


def harmful_mask(delta: Any, neutral_margin: float | None = None) -> np.ndarray:
    return classify_delta(delta, neutral_margin) == HARMFUL


def neutral_mask(delta: Any, neutral_margin: float | None = None) -> np.ndarray:
    return classify_delta(delta, neutral_margin) == NEUTRAL


def check_generation_tolerance(
    epsilon_neutral: float,
    neutral_margin: float | None = None,
    *,
    context: str = "CounterfactualGenerator",
) -> float:
    """Warn when the generator's search tolerance is looser than the decision margin.

    ``epsilon_neutral`` is the tolerance used while *searching* for a
    hard-neutral counterfactual: the generator retries until it finds a
    combined repair+regression edit whose measured ``abs(delta)`` falls below
    it.  That is a generation-time parameter and is intentionally distinct from
    the decision margin used to report or gate transitions.

    They are not independent, though.  If the search tolerance exceeds the
    decision margin, the generator can emit a sample it calls "hard neutral"
    whose delta the loss and the reporting layer both classify as strictly
    beneficial or harmful.  That is a real semantic conflict, so it is surfaced
    here rather than silently reconciled.
    """

    tolerance = float(epsilon_neutral)
    margin = resolve_neutral_margin(neutral_margin)
    if tolerance > margin:
        warnings.warn(
            f"{context}: hard-neutral search tolerance epsilon_neutral={tolerance:g} exceeds the "
            f"canonical decision margin neutral_margin={margin:g}. Generated 'hard neutral' "
            f"transitions with {margin:g} < |delta| <= {tolerance:g} will be trained and reported "
            f"as strictly beneficial or harmful. Set epsilon_neutral <= neutral_margin, or accept "
            f"this gap deliberately.",
            RuntimeWarning,
            stacklevel=2,
        )
    return tolerance


# --------------------------------------------------------------------------
# Empty-class policy for Dice
# --------------------------------------------------------------------------

EmptyClassPolicy = Literal["exclude", "legacy_one", "zero"]

#: Headline default.  A foreground class absent from BOTH prediction and ground
#: truth carries no information about segmentation quality, so scoring it 1.0
#: inflates every macro average -- on ACDC, apical and basal slices routinely
#: lack RV or MYO entirely.  Such a class is excluded from the macro instead.
DEFAULT_EMPTY_CLASS_POLICY: EmptyClassPolicy = "exclude"

#: Pre-remediation behaviour, retained only for explicit backwards comparison.
LEGACY_EMPTY_CLASS_POLICY: EmptyClassPolicy = "legacy_one"

_EMPTY_POLICY_SCORES: dict[str, float] = {
    "exclude": float("nan"),
    "legacy_one": 1.0,
    "zero": 0.0,
}


def resolve_empty_policy(policy: Any = None) -> EmptyClassPolicy:
    if policy is None:
        return DEFAULT_EMPTY_CLASS_POLICY
    name = str(policy)
    if name not in _EMPTY_POLICY_SCORES:
        raise ValueError(
            f"empty_policy must be one of {sorted(_EMPTY_POLICY_SCORES)}, got {policy!r}"
        )
    return name  # type: ignore[return-value]


def empty_class_score(policy: Any = None) -> float:
    """Return the Dice value assigned when prediction and target are both empty.

    ``nan`` marks the class as excluded from the macro; callers must aggregate
    with a nan-aware mean.  Only a class that is empty in *both* prediction and
    target takes this value: a class present in exactly one of them scores 0.0
    by the ordinary Dice formula and is never excluded.
    """

    return _EMPTY_POLICY_SCORES[resolve_empty_policy(policy)]


def macro_mean(values: Any) -> float:
    """Nan-aware macro mean; ``nan`` when every class was excluded."""

    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    if array.size == 0 or bool(np.all(np.isnan(array))):
        return float("nan")
    return float(np.nanmean(array))


# --------------------------------------------------------------------------
# Metric-space labels
# --------------------------------------------------------------------------

#: 2-D per-slice Dice on the network grid.  Training/monitoring proxy only.
METRIC_SPACE_SLICE_PROXY = "slice_proxy"
#: 3-D per-volume Dice on the resized network grid (e.g. 256x256 in-plane).
METRIC_SPACE_VOLUME_RESIZED = "volume_resized"
#: 3-D per-volume Dice after inverse-mapping to the original acquisition grid.
METRIC_SPACE_VOLUME_NATIVE = "volume_native"

METRIC_SPACES = (
    METRIC_SPACE_SLICE_PROXY,
    METRIC_SPACE_VOLUME_RESIZED,
    METRIC_SPACE_VOLUME_NATIVE,
)


__all__ = [
    "BENEFICIAL",
    "DEFAULT_EMPTY_CLASS_POLICY",
    "DEFAULT_NEUTRAL_MARGIN",
    "EmptyClassPolicy",
    "HARMFUL",
    "LEGACY_EMPTY_CLASS_POLICY",
    "METRIC_SPACES",
    "METRIC_SPACE_SLICE_PROXY",
    "METRIC_SPACE_VOLUME_NATIVE",
    "METRIC_SPACE_VOLUME_RESIZED",
    "NEUTRAL",
    "TRANSITION_CLASS_NAMES",
    "beneficial_mask",
    "check_generation_tolerance",
    "classify_delta",
    "empty_class_score",
    "harmful_mask",
    "macro_mean",
    "neutral_mask",
    "resolve_empty_policy",
    "resolve_neutral_margin",
]
