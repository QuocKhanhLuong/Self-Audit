"""Validation-only threshold calibration for the Self-Audit gate.

The calibration utility consumes cached validation transitions.  It never
changes the deployable inference path and never uses test data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _array(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _transition_arrays(
    delta_q: Any,
    actual_delta_dice: Any,
    active_mask: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quality = _array(delta_q, name="delta_q")
    actual = _array(actual_delta_dice, name="actual_delta_dice")
    if quality.ndim == 3 and quality.shape[-1] == 1:
        quality = quality[..., 0]
    if actual.ndim == 3 and actual.shape[-1] == 1:
        actual = actual[..., 0]
    if quality.ndim == 1:
        quality = quality[:, None]
    if actual.ndim == 1:
        actual = actual[:, None]
    if quality.ndim != 2 or actual.ndim != 2 or quality.shape != actual.shape:
        raise ValueError(
            "delta_q and actual_delta_dice must have matching [N,T] or [N,T,1] shapes; "
            f"got {quality.shape} and {actual.shape}"
        )
    if active_mask is None:
        active = np.ones(quality.shape, dtype=bool)
    else:
        active = _array(active_mask, name="active_mask").astype(bool, copy=False)
        if active.ndim == 1:
            active = active[:, None]
        if active.shape != quality.shape:
            raise ValueError(f"active_mask shape {active.shape} does not match transitions {quality.shape}")
    return quality.astype(np.float64), actual.astype(np.float64), active


def evaluate_threshold(
    tau_accept: float,
    initial_dice: Any,
    delta_q: Any,
    actual_delta_dice: Any,
    active_mask: Any | None = None,
) -> dict[str, float]:
    """Simulate per-sample threshold halting on cached validation transitions."""

    initial = _array(initial_dice, name="initial_dice").reshape(-1).astype(np.float64)
    quality, actual, valid = _transition_arrays(delta_q, actual_delta_dice, active_mask)
    if initial.shape[0] != quality.shape[0]:
        raise ValueError(f"initial_dice has {initial.shape[0]} samples, transitions have {quality.shape[0]}")

    active = np.ones(quality.shape[0], dtype=bool)
    final = initial.copy()
    attempted = np.zeros_like(initial, dtype=np.int64)
    accepted_count = np.zeros_like(initial, dtype=np.int64)
    accepted_total = 0
    rejected_total = 0
    harmful_total = 0
    beneficial_rejection_total = 0
    for turn in range(quality.shape[1]):
        eligible = active & valid[:, turn]
        attempted += eligible.astype(np.int64)
        accepted = eligible & (quality[:, turn] > float(tau_accept))
        rejected = eligible & ~accepted
        accepted_count += accepted.astype(np.int64)
        accepted_total += int(accepted.sum())
        rejected_total += int(rejected.sum())
        harmful_total += int((accepted & (actual[:, turn] <= 0.0)).sum())
        beneficial_rejection_total += int((rejected & (actual[:, turn] > 0.0)).sum())
        final += np.where(accepted, actual[:, turn], 0.0)
        # A rejected transition halts only that sample.  A sample with no
        # cached transition at a turn is also no longer eligible.
        active = accepted

    accepted_denominator = max(accepted_total, 1)
    rejected_denominator = max(rejected_total, 1)
    attempted_total = max(int(attempted.sum()), 1)
    return {
        "tau_accept": float(tau_accept),
        "final_macro_dice": float(final.mean()) if final.size else 0.0,
        "net_dice_gain": float((final - initial).mean()) if final.size else 0.0,
        "harmful_acceptance_rate": float(harmful_total / accepted_denominator),
        "beneficial_rejection_rate": float(beneficial_rejection_total / rejected_denominator),
        "mean_attempted_turns": float(attempted.mean()) if attempted.size else 0.0,
        "mean_accepted_turns": float(accepted_count.mean()) if accepted_count.size else 0.0,
        "acceptance_rate": float(accepted_total / attempted_total),
    }


def sweep_thresholds(
    transitions: Mapping[str, Any],
    thresholds: Iterable[float],
    *,
    max_harmful_acceptance_rate: float | None = None,
) -> list[dict[str, float]]:
    """Evaluate a threshold grid from a cached validation-transition mapping."""

    required = {"initial_dice", "delta_q", "actual_delta_dice"}
    missing = sorted(required - set(transitions))
    if missing:
        raise ValueError(f"Cached transitions are missing: {', '.join(missing)}")
    rows = [
        evaluate_threshold(
            float(tau),
            transitions["initial_dice"],
            transitions["delta_q"],
            transitions["actual_delta_dice"],
            transitions.get("active_mask"),
        )
        for tau in thresholds
    ]
    if max_harmful_acceptance_rate is not None:
        allowed = [row for row in rows if row["harmful_acceptance_rate"] <= float(max_harmful_acceptance_rate)]
        if allowed:
            return allowed
    return rows


def select_threshold(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Select the best validation row, with lower harmful acceptance as tie-break."""

    candidates = [dict(row) for row in rows]
    if not candidates:
        raise ValueError("Cannot select a threshold from zero calibration rows")
    return max(
        candidates,
        key=lambda row: (
            float(row.get("final_macro_dice", -np.inf)),
            -float(row.get("harmful_acceptance_rate", np.inf)),
            -abs(float(row.get("tau_accept", 0.0))),
        ),
    )
