"""Explicit exploratory quality-waiver policy for Candidate Q3D.

Strict behavior is the default everywhere.  A waiver exists only when a
caller passes a nonempty human-readable reason explicitly; it never comes from
a config file, so config hashes are unchanged.  The waiver permits a real run
to *continue past* three quality stops (tokenizer gate, physiology support,
global KD support).  It never rewrites a measured gate: measured gates keep
their NO_GO status and ``passed=False``.  Mathematical finite checks, shape,
data identity and hash validation are outside its scope.

All helpers in this module are pure (no I/O, no torch) so that the engine,
stage-artifact code and CLI can share one policy contract.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from decimal import Decimal, localcontext
from typing import Any

WAIVER_SCHEMA = "q3d.exploratory_quality_waiver.v1"
WAIVER_SCOPE = ("tokenizer_gate", "physiology_support", "kd_support")
WAIVER_STATE_KEY = "exploratory_quality_waiver"
WAIVER_EVENTS_KEY = "exploratory_quality_waiver_events"
_POLICY_KEYS = frozenset({"schema", "reason", "scope", "exploratory"})


def quality_waiver(reason: str | None) -> dict[str, Any] | None:
    """Return the canonical waiver policy for ``reason`` or ``None``.

    ``None`` means strict (default) behavior.  Any other value must be a
    string with non-whitespace content; the reason is stored stripped.
    """

    if reason is None:
        return None
    if not isinstance(reason, str):
        raise TypeError("exploratory quality waiver reason must be a string or None")
    stripped = reason.strip()
    if not stripped:
        raise ValueError("exploratory quality waiver reason must be a nonempty string")
    return {
        "schema": WAIVER_SCHEMA,
        "reason": stripped,
        "scope": list(WAIVER_SCOPE),
        "exploratory": True,
    }


def validate_quality_waiver(policy: Any) -> dict[str, Any] | None:
    """Validate a persisted/supplied policy exactly; return a deep copy."""

    if policy is None:
        return None
    if not isinstance(policy, Mapping):
        raise TypeError("exploratory quality waiver policy must be a mapping or None")
    if set(policy) != _POLICY_KEYS:
        raise ValueError("exploratory quality waiver policy has unexpected or missing fields")
    reason = policy.get("reason")
    expected = quality_waiver(reason if isinstance(reason, str) else "")
    assert expected is not None
    if expected["reason"] != reason:
        raise ValueError("exploratory quality waiver reason is not canonical (stripped)")
    scope = policy.get("scope")
    if (
        policy.get("schema") != expected["schema"]
        or not isinstance(scope, (list, tuple))
        or list(scope) != expected["scope"]
        or policy.get("exploratory") is not True
    ):
        raise ValueError("exploratory quality waiver policy does not match the v1 contract")
    return expected


def waiver_from_state(state: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(state, Mapping):
        return None
    return validate_quality_waiver(state.get(WAIVER_STATE_KEY))


def waiver_covers(policy_or_state: Any, scope: str) -> bool:
    """Return True only for a valid policy (or state holding one) covering scope."""

    if scope not in WAIVER_SCOPE:
        raise ValueError(f"unknown exploratory quality waiver scope {scope!r}")
    if isinstance(policy_or_state, Mapping) and WAIVER_STATE_KEY in policy_or_state:
        policy = waiver_from_state(policy_or_state)
    elif isinstance(policy_or_state, Mapping) and policy_or_state.get("schema") == WAIVER_SCHEMA:
        policy = validate_quality_waiver(policy_or_state)
    else:
        policy = None
    return bool(policy is not None and scope in policy["scope"])


def bind_quality_waiver(
    state: dict[str, Any], policy: Mapping[str, Any] | None, *, context: str,
) -> dict[str, Any] | None:
    """Bind the explicitly supplied policy to ``state``.

    Rules:
    * no persisted waiver, no supplied waiver: strict, nothing recorded;
    * persisted waiver, no supplied waiver: rejected (reason must be explicit
      on every resume/import of waived state);
    * persisted and supplied waivers differ: rejected;
    * no persisted waiver, supplied waiver: bound now (e.g. continuing a strict
      run whose measured gate failed); the binding event is recorded.
    """

    supplied = validate_quality_waiver(policy)
    persisted = waiver_from_state(state)
    if persisted is not None and supplied is None:
        raise RuntimeError(
            f"{context}: state carries an exploratory quality waiver; resuming or importing "
            "it requires the same explicit --q3d-exploratory-quality-waiver REASON"
        )
    if persisted is not None and supplied is not None and persisted != supplied:
        raise RuntimeError(
            f"{context}: supplied exploratory quality waiver reason differs from the persisted one"
        )
    if supplied is None:
        state.pop(WAIVER_STATE_KEY, None)
        return None
    state[WAIVER_STATE_KEY] = copy.deepcopy(supplied)
    if persisted is None:
        record_waiver_event(state, {"event": "exploratory_quality_waiver_bound", "context": context})
    return copy.deepcopy(supplied)


def record_waiver_event(state: dict[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    """Append a waiver/fallback event to state and return the stored record."""

    policy = waiver_from_state(state)
    record = {
        "schema": "q3d.exploratory_quality_waiver_event.v1",
        "exploratory": True,
        "waiver_reason": policy["reason"] if policy is not None else None,
        **copy.deepcopy(dict(event)),
    }
    state.setdefault(WAIVER_EVENTS_KEY, []).append(record)
    return record


def exploratory_relative_omega(w_pat: Mapping[str, float], arm: str) -> dict[str, Any]:
    """Waived omega for a U cohort that failed the mean/ESS gate.

    * V1-U: always unit omega (``fallback="v1u_unit_omega"``).
    * positive mean (low ESS): existing relative weights normalized by their
      cohort mean, with the same canonical-decimal arithmetic as
      ``losses.cohort_omega`` (``fallback="low_ess_relative_normalized"``).
    * zero mean: unit omega, logged explicitly as a uniform fallback
      (``fallback="zero_mean_uniform_unit_omega"``).
    Every emitted value is finite and nonnegative.
    """

    if not isinstance(w_pat, Mapping) or not w_pat:
        raise ValueError("w_pat cohort must be a nonempty mapping")
    values: dict[str, float] = {}
    for patient, raw in w_pat.items():
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("w_pat values must be finite and nonnegative")
        values[str(patient)] = value
    arm_key = str(arm).strip().upper().replace("_", "-")
    if arm_key == "V1U":
        arm_key = "V1-U"
    if arm_key not in {"V1", "C1", "V1-U"}:
        raise ValueError("exploratory omega applies only to V1, C1 or V1-U")
    n = len(values)
    positive_mean = math.fsum(values.values()) > 0.0
    if arm_key == "V1-U":
        omega = {patient: 1.0 for patient in values}
        fallback = "v1u_unit_omega"
    elif not positive_mean:
        omega = {patient: 1.0 for patient in values}
        fallback = "zero_mean_uniform_unit_omega"
    else:
        with localcontext() as context:
            context.prec = 50
            canonical = {p: Decimal(format(v, ".15g")) for p, v in values.items()}
            mean = sum(canonical.values(), Decimal(0)) / Decimal(n)
            omega = {p: float(v / mean) for p, v in canonical.items()}
        fallback = "low_ess_relative_normalized"
    emitted = list(omega.values())
    if not all(math.isfinite(value) and value >= 0.0 for value in emitted):
        raise FloatingPointError("exploratory omega produced a nonfinite or negative value")
    return {
        "omega": omega,
        "fallback": fallback,
        "mean_omega": math.fsum(emitted) / n,
        "max_omega": max(emitted),
        "zero_fraction": sum(value == 0.0 for value in emitted) / n,
    }


__all__ = [
    "WAIVER_EVENTS_KEY",
    "WAIVER_SCHEMA",
    "WAIVER_SCOPE",
    "WAIVER_STATE_KEY",
    "bind_quality_waiver",
    "exploratory_relative_omega",
    "quality_waiver",
    "record_waiver_event",
    "validate_quality_waiver",
    "waiver_covers",
    "waiver_from_state",
]
