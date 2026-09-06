"""Validation-only threshold calibration for the Self-Audit gate.

The calibration utility consumes cached validation transitions.  It never
changes the deployable inference path and never uses test data.

Two things are deliberate here and easy to get wrong:

1.  A transition whose measured ``delta_dice`` sits inside the canonical
    neutral margin is *neither* harmful nor beneficial.  It is excluded from
    both the harmful-acceptance and beneficial-rejection denominators rather
    than being folded into one of them.  Counting ``delta == 0`` as harmful
    (the pre-remediation behaviour) biased ``harmful_acceptance_rate`` upward,
    and because that rate is the tie-break inside :func:`select_threshold`, it
    systematically pushed the calibrated ``tau_accept`` more conservative than
    the project's own neutral semantics warrant.
2.  ``tau_accept`` is selected by argmax over a grid on the *same* split whose
    Dice is then reported.  That is an optimistically-biased diagnostic, not
    held-out evidence.  The persisted artifact records this in-band; see
    :data:`VALIDITY_DIAGNOSTIC_ONLY` and :func:`save_calibration`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..audit.semantics import (
    METRIC_SPACES,
    beneficial_mask,
    harmful_mask,
    neutral_mask,
    resolve_neutral_margin,
)


# --------------------------------------------------------------------------
# Persisted calibration artifact
# --------------------------------------------------------------------------

#: Schema version of the calibration JSON artifact written by
#: :func:`save_calibration`.  :func:`load_calibration` refuses any other value.
CALIBRATION_SCHEMA_VERSION = 1

#: The only validity class this artifact may carry.  ``tau_accept`` is chosen
#: on the split it is measured on, so the artifact is never held-out evidence.
VALIDITY_DIAGNOSTIC_ONLY = "diagnostic_only"

#: Verbatim statement of why the artifact is diagnostic only.  Written into
#: every artifact so the caveat travels with the number rather than living in
#: a report someone may not read.
VALIDITY_REASON = (
    "tau_accept was selected by argmax over the threshold grid on the same "
    "validation split on which its final_macro_dice is reported. Selecting "
    "and reporting on one split makes every number in selected_row an "
    "optimistically-biased diagnostic, not held-out evidence: the reported "
    "gain includes the selection bias of the grid search. The current "
    "checkpoint has no independent test set -- the project's 80/20 ACDC split "
    "has a train half and a validation half and nothing else -- so no "
    "unbiased estimate of this threshold's benefit exists yet. Treat this "
    "artifact as a reproducible record of how tau_accept was chosen and of "
    "which tau a run actually deployed, never as evidence of generalisation."
)

_CALIBRATION_KEYS: tuple[str, ...] = (
    "schema_version",
    "tau_accept",
    "neutral_margin",
    "source_split",
    "checkpoint_path",
    "checkpoint_sha256",
    "t_max",
    "threshold_grid",
    "selected_row",
    "created_at",
    "metric_space",
    "validity",
    "validity_reason",
    "extra",
)

_CALIBRATION_REQUIRED: tuple[str, ...] = tuple(k for k in _CALIBRATION_KEYS if k != "extra")


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
    *,
    neutral_margin: float | None = None,
) -> dict[str, float]:
    """Simulate per-sample threshold halting on cached validation transitions.

    The halting simulation is unchanged and remains faithful to
    ``SelfAuditNet.infer``: a rejected transition halts only that sample, and
    only accepted transitions contribute their measured ``delta_dice``.

    What changed is the *classification* of each simulated decision.  Actual
    deltas are classified with the canonical
    :func:`~self_audit.audit.semantics.classify_delta`, so

    * ``harmful_total`` counts accepted transitions that are strictly HARMFUL
      (``delta < -neutral_margin``), not everything with ``delta <= 0``;
    * ``beneficial_rejection_total`` counts rejected transitions that are
      strictly BENEFICIAL (``delta > +neutral_margin``);
    * neutral transitions are excluded from both *numerators*, so a gate that
      accepts only no-op edits scores ``harmful_acceptance_rate == 0.0``
      instead of ``1.0``.

    The headline denominators are the **full** accepted and rejected
    populations, matching :func:`self_audit.evaluation.metrics.acceptance_metrics`.
    Conditioning the denominator on materially signed transitions instead would
    report 100% harmful acceptance for a gate that accepted a thousand no-ops
    and one harmful edit -- and since :func:`select_threshold` tie-breaks on
    this rate, that would drive ``tau_accept`` *more* conservative than the
    original zero-boundary bug did.  The conditional forms are still emitted as
    ``harmful_acceptance_rate_signed`` / ``beneficial_rejection_rate_signed``
    for diagnosis, and ``neutral_acceptance_rate`` keeps the excluded
    population visible.
    """

    margin = resolve_neutral_margin(neutral_margin)
    initial = _array(initial_dice, name="initial_dice").reshape(-1).astype(np.float64)
    quality, actual, valid = _transition_arrays(delta_q, actual_delta_dice, active_mask)
    if initial.shape[0] != quality.shape[0]:
        raise ValueError(f"initial_dice has {initial.shape[0]} samples, transitions have {quality.shape[0]}")

    is_harmful = harmful_mask(actual, margin)
    is_beneficial = beneficial_mask(actual, margin)
    is_neutral = neutral_mask(actual, margin)

    active = np.ones(quality.shape[0], dtype=bool)
    final = initial.copy()
    attempted = np.zeros_like(initial, dtype=np.int64)
    accepted_count = np.zeros_like(initial, dtype=np.int64)
    accepted_total = 0
    rejected_total = 0
    accepted_non_neutral_total = 0
    rejected_non_neutral_total = 0
    harmful_total = 0
    beneficial_rejection_total = 0
    neutral_accepted_total = 0
    for turn in range(quality.shape[1]):
        eligible = active & valid[:, turn]
        attempted += eligible.astype(np.int64)
        accepted = eligible & (quality[:, turn] > float(tau_accept))
        rejected = eligible & ~accepted
        accepted_count += accepted.astype(np.int64)
        accepted_total += int(accepted.sum())
        rejected_total += int(rejected.sum())
        accepted_non_neutral_total += int((accepted & ~is_neutral[:, turn]).sum())
        rejected_non_neutral_total += int((rejected & ~is_neutral[:, turn]).sum())
        harmful_total += int((accepted & is_harmful[:, turn]).sum())
        beneficial_rejection_total += int((rejected & is_beneficial[:, turn]).sum())
        neutral_accepted_total += int((accepted & is_neutral[:, turn]).sum())
        final += np.where(accepted, actual[:, turn], 0.0)
        # A rejected transition halts only that sample.  A sample with no
        # cached transition at a turn is also no longer eligible.
        active = accepted

    attempted_total = max(int(attempted.sum()), 1)
    return {
        "tau_accept": float(tau_accept),
        "neutral_margin": float(margin),
        "final_macro_dice": float(final.mean()) if final.size else 0.0,
        "net_dice_gain": float((final - initial).mean()) if final.size else 0.0,
        # Neutral transitions are excluded from the numerators only; the
        # headline denominators are the full accepted / rejected populations.
        "harmful_acceptance_rate": float(harmful_total / max(accepted_total, 1)),
        "beneficial_rejection_rate": float(beneficial_rejection_total / max(rejected_total, 1)),
        "harmful_acceptance_rate_signed": float(harmful_total / max(accepted_non_neutral_total, 1)),
        "beneficial_rejection_rate_signed": float(
            beneficial_rejection_total / max(rejected_non_neutral_total, 1)
        ),
        "neutral_acceptance_rate": float(neutral_accepted_total / max(accepted_total, 1)),
        "mean_attempted_turns": float(attempted.mean()) if attempted.size else 0.0,
        "mean_accepted_turns": float(accepted_count.mean()) if accepted_count.size else 0.0,
        "acceptance_rate": float(accepted_total / attempted_total),
        "accepted_total": int(accepted_total),
        "rejected_total": int(rejected_total),
        "accepted_non_neutral_total": int(accepted_non_neutral_total),
        "rejected_non_neutral_total": int(rejected_non_neutral_total),
        "harmful_total": int(harmful_total),
        "beneficial_rejection_total": int(beneficial_rejection_total),
        "neutral_accepted_total": int(neutral_accepted_total),
    }


def sweep_thresholds(
    transitions: Mapping[str, Any],
    thresholds: Iterable[float],
    *,
    max_harmful_acceptance_rate: float | None = None,
    neutral_margin: float | None = None,
) -> list[dict[str, float]]:
    """Evaluate a threshold grid from a cached validation-transition mapping."""

    required = {"initial_dice", "delta_q", "actual_delta_dice"}
    missing = sorted(required - set(transitions))
    if missing:
        raise ValueError(f"Cached transitions are missing: {', '.join(missing)}")
    margin = resolve_neutral_margin(neutral_margin)
    rows = [
        evaluate_threshold(
            float(tau),
            transitions["initial_dice"],
            transitions["delta_q"],
            transitions["actual_delta_dice"],
            transitions.get("active_mask"),
            neutral_margin=margin,
        )
        for tau in thresholds
    ]
    if max_harmful_acceptance_rate is not None:
        allowed = [row for row in rows if row["harmful_acceptance_rate"] <= float(max_harmful_acceptance_rate)]
        if allowed:
            return allowed
    return rows


#: Human-readable statement of what :func:`select_threshold` maximises.
SELECTION_OBJECTIVE = (
    "argmax over the threshold grid of the lexicographic key "
    "(final_macro_dice, -harmful_acceptance_rate, -abs(tau_accept)); "
    "harmful_acceptance_rate counts only strictly harmful accepts over all accepts; "
    "selection and reporting share one split, so the result is diagnostic only"
)


def select_threshold(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Select the best validation row, with lower harmful acceptance as tie-break.

    The lexicographic key is unchanged.  The returned dict now additionally
    carries ``objective`` (what was maximised, in words), ``n_rows`` (how many
    grid points that argmax ranged over -- the size of the selection bias) and
    the ``neutral_margin`` the rows were classified with, so a persisted
    selection is self-describing.
    """

    candidates = [dict(row) for row in rows]
    if not candidates:
        raise ValueError("Cannot select a threshold from zero calibration rows")
    best = max(
        candidates,
        key=lambda row: (
            float(row.get("final_macro_dice", -np.inf)),
            -float(row.get("harmful_acceptance_rate", np.inf)),
            -abs(float(row.get("tau_accept", 0.0))),
        ),
    )
    taus = [float(row.get("tau_accept", 0.0)) for row in candidates]
    best = dict(best)
    best["objective"] = SELECTION_OBJECTIVE
    best["n_rows"] = int(len(candidates))
    best["neutral_margin"] = resolve_neutral_margin(best.get("neutral_margin"))
    best["grid"] = {"min": float(min(taus)), "max": float(max(taus)), "steps": int(len(taus))}
    return best


# --------------------------------------------------------------------------
# Artifact I/O
# --------------------------------------------------------------------------


def _file_sha256(path: Any) -> str | None:
    """Return the SHA-256 of ``path``, or ``None`` if it cannot be read.

    A missing or unreadable checkpoint must never fail a calibration write --
    the artifact simply records that the checkpoint identity is unknown.
    """

    if path is None:
        return None
    try:
        candidate = Path(os.fspath(path))
        if not candidate.is_file():
            return None
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, TypeError, ValueError):
        return None


def _grid_dict(threshold_grid: Any) -> dict[str, float | int]:
    if isinstance(threshold_grid, Mapping):
        missing = sorted({"min", "max", "steps"} - set(threshold_grid))
        if missing:
            raise ValueError(f"threshold_grid is missing: {', '.join(missing)}")
        return {
            "min": float(threshold_grid["min"]),
            "max": float(threshold_grid["max"]),
            "steps": int(threshold_grid["steps"]),
        }
    values = [float(value) for value in threshold_grid]
    if not values:
        raise ValueError("threshold_grid must contain at least one threshold")
    return {"min": float(min(values)), "max": float(max(values)), "steps": int(len(values))}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    return value


def save_calibration(
    path: Any,
    *,
    tau_accept: float,
    neutral_margin: float | None,
    source_split: str,
    checkpoint_path: Any = None,
    t_max: int,
    threshold_grid: Any,
    selected_row: Mapping[str, Any],
    metric_space: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the schema-v1 calibration artifact and return the payload.

    The artifact is the single record of which ``tau_accept`` a run selected,
    on which split, under which neutral margin, in which metric space, and
    against which checkpoint.  It always carries ``validity ==
    "diagnostic_only"`` plus the verbatim :data:`VALIDITY_REASON`, because
    ``tau_accept`` is selected by argmax on the split it is measured on.

    ``checkpoint_sha256`` is computed when the checkpoint file exists and is
    readable, and is ``None`` otherwise -- a missing checkpoint never fails
    the write.
    """

    if metric_space not in METRIC_SPACES:
        raise ValueError(f"metric_space must be one of {list(METRIC_SPACES)}, got {metric_space!r}")
    payload: dict[str, Any] = {
        "schema_version": int(CALIBRATION_SCHEMA_VERSION),
        "tau_accept": float(tau_accept),
        "neutral_margin": resolve_neutral_margin(neutral_margin),
        "source_split": str(source_split),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "t_max": int(t_max),
        "threshold_grid": _grid_dict(threshold_grid),
        "selected_row": _json_safe(dict(selected_row)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_space": str(metric_space),
        "validity": VALIDITY_DIAGNOSTIC_ONLY,
        "validity_reason": VALIDITY_REASON,
        "extra": _json_safe(dict(extra)) if extra is not None else {},
    }
    destination = Path(os.fspath(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_calibration(path: Any) -> dict[str, Any]:
    """Read a schema-v1 calibration artifact, validating it strictly.

    An unknown ``schema_version``, an unknown top-level key, a missing
    required key, or a ``validity`` other than ``"diagnostic_only"`` raises.
    Nothing is silently ignored: a calibration artifact this loader does not
    fully understand must not be allowed to hand a threshold to inference.
    """

    source = Path(os.fspath(path))
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Calibration artifact {source} must be a JSON object")
    version = payload.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported calibration schema_version {version!r} in {source}; "
            f"this build understands version {CALIBRATION_SCHEMA_VERSION} only"
        )
    unknown = sorted(set(payload) - set(_CALIBRATION_KEYS))
    if unknown:
        raise ValueError(
            f"Calibration artifact {source} has unknown top-level keys: {', '.join(unknown)}"
        )
    missing = sorted(set(_CALIBRATION_REQUIRED) - set(payload))
    if missing:
        raise ValueError(f"Calibration artifact {source} is missing: {', '.join(missing)}")
    if payload["validity"] != VALIDITY_DIAGNOSTIC_ONLY:
        raise ValueError(
            f"Calibration artifact {source} declares validity {payload['validity']!r}; "
            f"only {VALIDITY_DIAGNOSTIC_ONLY!r} is a valid calibration class"
        )
    payload["tau_accept"] = float(payload["tau_accept"])
    payload["neutral_margin"] = resolve_neutral_margin(payload["neutral_margin"])
    payload["t_max"] = int(payload["t_max"])
    payload.setdefault("extra", {})
    return payload


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "SELECTION_OBJECTIVE",
    "VALIDITY_DIAGNOSTIC_ONLY",
    "VALIDITY_REASON",
    "evaluate_threshold",
    "load_calibration",
    "save_calibration",
    "select_threshold",
    "sweep_thresholds",
]
