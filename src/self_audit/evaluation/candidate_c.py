"""Evaluation-only diagnostics for the Candidate C restitution path.

This module *measures* what the runtime Candidate C solver already did.  It
never proposes a coordinate, never re-runs the solver, and never re-derives the
Auditor decision: every solver-side number (replay error, objective, evaluation
counts, ``delta_q``, acceptance) is read out of the per-attempt rows the model
emitted during inference.  If the model did not emit a field, the field is
reported as ``None`` -- an unavailable metric is never reported as ``0``.

Two clearly separated families live here.

``summarize_diagnostic_rows`` / ``summarize_geometry``
    Ground-truth-free.  These describe solver behaviour and support geometry
    and may be computed on any inference output.

``ground_truth_restitution_metrics``
    **Evaluation only.**  It consumes ground truth to score three already-
    computed logit tensors.  Nothing it returns is ever passed back into
    ``infer``, the Annotation Expert, the Auditor or the solver; there is no
    code path from this function into a model call.  See
    :data:`GT_METRICS_ARE_EVALUATION_ONLY`.

Coordinate conventions
----------------------
Realized supports are normalized ``align_corners=True`` ``grid_sample``
coordinates ordered ``(x, y)``, where ``x`` indexes *width* and ``y`` indexes
*height*.  One feature pixel is ``2 / (size - 1)`` normalized units; a
degenerate size-one axis has **zero** physical extent, so displacement along it
is reported as exactly zero and the axis is named in ``degenerate_axes``
instead of being silently divided by zero.

Geometry payload
----------------
``infer(..., capture_geometry=True)`` emits ``candidate_c_geometry``: a list
the same length and in the same order as ``candidate_c_diagnostics``.  Element
``i`` describes row ``i``.  An available element carries ``turn``,
``record_turn``, ``sample_index`` (an index into the FULL batch), ``feature_hw``,
``k``, and two equal-length depth-ordered tuples ``factual`` and ``chosen``.
Each depth entry carries ``coordinates`` ``[1,H,W,K,2]``, optional
``coordinates_preclamp``, optional ``attention`` (native ``[1,heads,H,W,K]`` or
flattened ``[1,heads,H*W,K]``), ``depth_index``, ``turn_index``,
``iteration_index`` and ``state_identity``.  An element that instead carries
``unavailable_reason`` is reported unavailable, not zero-filled.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

__all__ = [
    "CANDIDATE_C_DIAGNOSTIC_SCHEMA_VERSION",
    "DEFAULT_DUPLICATE_TOLERANCE_PIXELS",
    "DEFAULT_SATURATION_EPS",
    "GT_METRICS_ARE_EVALUATION_ONLY",
    "EFFECT_COUNT_KEYS",
    "SOLVER_EVAL_DEDUPLICATION_NOTE",
    "SOLVER_EVAL_SCOPE_NOTE",
    "SOLVER_EVAL_ZERO_WORK_NOTE",
    "EFFECT_FAMILIES",
    "ROW_COUNT_FIELDS",
    "ROW_EVAL_FIELDS",
    "ROW_FLAG_FIELDS",
    "ROW_NUMERIC_FIELDS",
    "evaluate_candidate_c",
    "geometry_row_metrics",
    "summarize_geometry_rows",
    "ground_truth_restitution_metrics",
    "merge_ground_truth_blocks",
    "normalized_pixel_step",
    "summarize_diagnostic_rows",
    "summarize_geometry",
]


#: Bumped whenever the dictionaries returned here change shape non-additively.
CANDIDATE_C_DIAGNOSTIC_SCHEMA_VERSION = 1

#: Read by tests and by the audit CLI as an executable statement of intent: the
#: ground-truth block below is scoring-only and has no path into a model call.
GT_METRICS_ARE_EVALUATION_ONLY = True

#: Two support points are "duplicates" when they agree on *both* axes to within
#: this many feature pixels.  Reported alongside every duplicate count, because
#: a duplicate rate without its tolerance is meaningless.
DEFAULT_DUPLICATE_TOLERANCE_PIXELS = 0.5

#: A coordinate component is "saturated" when it sits within this distance of
#: the +/-1 domain edge.
DEFAULT_SATURATION_EPS = 1e-6

#: Per-attempt row fields summarised as continuous quantities.
ROW_NUMERIC_FIELDS: tuple[str, ...] = (
    "c1_max_abs_err",
    "regress_mass",
    "fix_mass",
    "objective_before",
    "objective_after",
    "constraint_violation",
    "coordinate_displacement",
    "innovation_magnitude",
    "delta_q",
)

#: Per-attempt row fields summarised as non-negative integer counts.
ROW_COUNT_FIELDS: tuple[str, ...] = ("num_protected", "num_ties_excluded", "num_fix_ties")

#: Per-attempt row fields summarised as tri-state booleans (True/False/None).
ROW_FLAG_FIELDS: tuple[str, ...] = ("eligible", "c1_passed", "feasible", "improved", "accepted")

#: Actual solver work, as counted by the solver itself.  These are the honest
#: forward/backward counts; nothing here re-derives them from the schedule.
ROW_EVAL_FIELDS: tuple[str, ...] = (
    "factual_replay",
    "coordinate_backward",
    "candidate_checks",
    "total_forward",
)

#: Row fields summarised as string histograms.
ROW_CATEGORICAL_FIELDS: tuple[str, ...] = ("record_kind", "fallback_reason", "accepted_path")

#: Accepted paths whose row admits a pretransition/factual/current triplet.
RESTITUTION_ACCEPTED_PATHS: frozenset[str] = frozenset({"counterfactual", "rollback"})


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------


def _finite_float(value: Any) -> float | None:
    """Coerce to a finite float, or ``None``.  Never invents ``0.0``."""

    if value is None or isinstance(value, bool):
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.item()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_nonfinite(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if torch.is_tensor(value):
        if value.numel() != 1:
            return False
        value = value.item()
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _tri_state(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return bool(value.item())
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _rate(numerator: int, denominator: int) -> float | None:
    """A rate is ``None`` when its denominator is zero -- never ``0.0``."""

    return float(numerator) / float(denominator) if denominator else None


def _summarize_numbers(values: Sequence[Any]) -> dict[str, Any]:
    """Summarise one row field, keeping missing and non-finite entries visible."""

    present: list[float] = []
    null_count = 0
    nonfinite_count = 0
    for value in values:
        if value is None:
            null_count += 1
            continue
        if _is_nonfinite(value):
            nonfinite_count += 1
            continue
        number = _finite_float(value)
        if number is None:
            null_count += 1
            continue
        present.append(number)
    count = len(present)
    return {
        "count": count,
        "null_count": int(null_count),
        "nonfinite_count": int(nonfinite_count),
        "rows_considered": int(len(values)),
        "mean": (sum(present) / float(count)) if count else None,
        "min": min(present) if count else None,
        "max": max(present) if count else None,
        "sum": float(sum(present)) if count else None,
    }


def _summarize_flag(values: Sequence[Any]) -> dict[str, Any]:
    true_count = 0
    false_count = 0
    null_count = 0
    for value in values:
        state = _tri_state(value)
        if state is None:
            null_count += 1
        elif state:
            true_count += 1
        else:
            false_count += 1
    denominator = true_count + false_count
    return {
        "true_count": int(true_count),
        "false_count": int(false_count),
        "null_count": int(null_count),
        "denominator": int(denominator),
        "rate": _rate(true_count, denominator),
        "rows_considered": int(len(values)),
    }


def _histogram(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = "__null__" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _validate_geometry_limits(duplicate_tolerance_pixels: Any, saturation_eps: Any) -> tuple[float, float]:
    """Reject a caller-supplied limit that is not a finite, non-negative number.

    These are the denominators of the duplicate and saturation definitions. A
    ``nan`` tolerance silently makes every comparison false and a negative one
    makes every count zero, so both would publish a confident zero that means
    nothing.  A bad limit is a caller bug and is raised, not absorbed.
    """

    limits: list[float] = []
    for name, value in (
        ("duplicate_tolerance_pixels", duplicate_tolerance_pixels),
        ("saturation_eps", saturation_eps),
    ):
        number = _finite_float(value)
        if number is None:
            raise ValueError(f"{name} must be a finite number, got {value!r}")
        if number < 0.0:
            raise ValueError(f"{name} must be non-negative, got {number!r}")
        limits.append(number)
    return limits[0], limits[1]


def normalized_pixel_step(height: int, width: int) -> tuple[float, float]:
    """One feature pixel in ``align_corners=True`` normalized units, ``(x, y)``.

    Mirrors ``self_audit.models.dynamic_window.normalized_pixel_step``.  A
    size-one axis yields ``0.0``: it has no physical extent, so no normalized
    displacement along it corresponds to real movement.
    ``tests/test_candidate_c_diagnostics.py`` asserts the two definitions agree
    whenever the core helper is importable.
    """

    height = int(height)
    width = int(width)
    step_x = 2.0 / float(width - 1) if width > 1 else 0.0
    step_y = 2.0 / float(height - 1) if height > 1 else 0.0
    return step_x, step_y


# --------------------------------------------------------------------------
# Per-attempt row summary (ground-truth free)
# --------------------------------------------------------------------------


#: What the solver-only evaluation counters do and do not include.  The
#: restitution solver's forwards are the ONLY thing counted: the official
#: Auditor's forward, the ordinary Annotation Expert's forward, and the
#: separate ordinary-fallback evaluation that follows a C1 replay failure all
#: happen outside the solver and are not in these numbers.
SOLVER_EVAL_SCOPE_NOTE = (
    "Solver-only. These are the restitution solver's own forward/backward counts. They EXCLUDE "
    "the official Auditor's forward, the ordinary Annotation Expert's forward, and the separate "
    "ordinary-fallback evaluation that a C1 replay failure falls back to."
)

#: A group whose counters are all explicitly zero did not execute a solver
#: invocation.  Ordinary annotation turns, direct rollback (which consumes an
#: accepted record but never calls the solver) and a stale-record preflight
#: that aborts before any replay all land here.  They are KNOWN zero work, not
#: unknown work, and they are not phantom invocations.
SOLVER_EVAL_ZERO_WORK_NOTE = (
    "A group whose four counters are all explicitly zero executed no solver invocation: an "
    "ordinary annotation turn, a direct rollback (record consumed, solver never called) or a "
    "stale-record preflight that aborted before any replay. Such groups are excluded from "
    "'invocations' and counted under 'invocation_groups_attempted'. They contribute a KNOWN "
    "zero to the totals; they never make a total unknown."
)

#: Why the raw per-row sum is not the work that was done.
SOLVER_EVAL_DEDUPLICATION_NOTE = (
    "One solver invocation runs on a batched group and the SAME counts are copied into every "
    "surviving row of that group, so adding them up multiplies real work by the number of "
    "survivors. Totals here are deduplicated per invocation. Dividing by solver_group_size "
    "would also be wrong: a halted row emits no diagnostics row at all."
)


def _nonnegative_int(value: Any) -> int | None:
    """Coerce to a non-negative int, or ``None``.  An evaluation count is a tally."""

    if value is None or isinstance(value, bool):
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.item()
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _invocation_key(row: Mapping[str, Any], scope: Any) -> tuple[Any, str] | None:
    """Identify the solver invocation a row's counters came from.

    ``solver_invocation_id`` is preferred whenever core supplies it.  Otherwise
    the composite ``(scope, turn, record_turn)`` is exact for the current
    architecture, which runs at most one restitution solver invocation per turn
    per ``infer`` call; ``scope`` is the caller's per-batch namespace, so a
    repeated sample/turn index in a later batch cannot collide with an earlier
    one.  A row that carries neither is unkeyable and is excluded from the
    totals rather than guessed at.
    """

    identifier = row.get("solver_invocation_id")
    if identifier is not None and not isinstance(identifier, (dict, list, set)):
        return (("id", scope, identifier), "solver_invocation_id")
    turn = row.get("turn")
    if isinstance(turn, bool) or not isinstance(turn, int):
        return None
    record_turn = row.get("record_turn")
    if isinstance(record_turn, bool) or not isinstance(record_turn, int):
        record_turn = None
    return (("composite", scope, turn, record_turn), "batch_scope+turn+record_turn")


def _summarize_solver_evals(
    rows: Sequence[Mapping[str, Any]],
    scopes: Sequence[Any],
) -> dict[str, Any]:
    """Separate ACTUAL solver work from per-row attribution.

    Three distinct quantities are published, none of them a rebranding of
    another:

    ``actual_invocation_totals``
        real work, deduplicated per solver invocation.  This is what the
        ``*_sum`` flat keys carry.
    ``sample_associated_mean``
        the per-row mean of the shared counters -- a per-sample attribution of
        a shared cost, never a total.
    ``per_row_opportunity``
        how many rows shared each invocation, and the solver group sizes.  A
        row is an opportunity to restitute, not a unit of work.

    An invocation whose rows disagree on a counter is reported ``unknown``; the
    affected field's total becomes ``None`` and the disagreement is listed.
    Nothing is averaged or invented to fill the hole.
    """

    groups: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    key_sources: set[str] = set()
    rows_without_evals = 0
    rows_unkeyable_uncertain = 0
    rows_unkeyable_zero_work = 0
    record_consuming_rows = 0
    record_consuming_paths: list[Any] = []

    for row, scope in zip(rows, scopes):
        record_turn = row.get("record_turn")
        if isinstance(record_turn, int) and not isinstance(record_turn, bool):
            record_consuming_rows += 1
            record_consuming_paths.append(row.get("accepted_path"))

        block = row.get("evals")
        if not isinstance(block, Mapping):
            # No counters at all: this row could be hiding solver work, so it
            # makes the exact totals unknown.  It is never assumed to be zero.
            rows_without_evals += 1
            continue
        counts = {field: _nonnegative_int(block.get(field)) for field in ROW_EVAL_FIELDS}
        all_zero = all(value == 0 for value in counts.values())

        keyed = _invocation_key(row, scope)
        if keyed is None:
            # An unkeyable row that explicitly reported zero work is still
            # KNOWN zero work; only an unkeyable row that might carry work
            # makes the totals unknown.
            if all_zero:
                rows_unkeyable_zero_work += 1
            else:
                rows_unkeyable_uncertain += 1
            continue
        key, source = keyed
        key_sources.add(source)
        group = groups.get(key)
        if group is None:
            group = {
                "rows": 0,
                "counts": dict(counts),
                "inconsistent": set(),
                "all_zero": all_zero,
                "scope": scope,
                "turn": row.get("turn"),
                "record_turn": row.get("record_turn"),
                "accepted_paths": set(),
            }
            groups[key] = group
            order.append(key)
        else:
            for field, value in counts.items():
                if group["counts"][field] != value:
                    group["inconsistent"].add(field)
            group["all_zero"] = bool(group["all_zero"] and all_zero)
        group["rows"] += 1
        path = row.get("accepted_path")
        if path is not None:
            group["accepted_paths"].add(str(path))

    # A group that executed no solver invocation is known zero work, not a
    # phantom invocation.  ``executed`` is what ``invocations`` reports.
    executed = [
        key
        for key in order
        if groups[key]["inconsistent"] or not groups[key]["all_zero"]
    ]
    zero_work = [key for key in order if key not in set(executed)]

    # Any row that might be hiding solver work makes the EXACT totals unknown,
    # even when other groups are perfectly known.  The known part is retained
    # separately and is never presented as the full figure.
    uncertain_rows = int(rows_without_evals + rows_unkeyable_uncertain)

    totals: dict[str, int | None] = {}
    known_totals: dict[str, int] = {}
    unknown_per_field: dict[str, int] = {}
    for field in ROW_EVAL_FIELDS:
        running = 0
        unknown = 0
        for key in executed:
            group = groups[key]
            value = group["counts"][field]
            if value is None or field in group["inconsistent"]:
                unknown += 1
                continue
            running += value
        known_totals[field] = int(running)
        unknown_per_field[field] = int(unknown)
        totals[field] = None if (unknown or uncertain_rows) else int(running)

    inconsistent = [
        {
            "scope": str(groups[key]["scope"]),
            "turn": groups[key]["turn"],
            "record_turn": groups[key]["record_turn"],
            "rows": int(groups[key]["rows"]),
            "fields": sorted(groups[key]["inconsistent"]),
        }
        for key in order
        if groups[key]["inconsistent"]
    ]
    unmeasured = sum(
        1
        for key in executed
        if not groups[key]["inconsistent"]
        and any(groups[key]["counts"][field] is None for field in ROW_EVAL_FIELDS)
    )

    row_sums: dict[str, Any] = {}
    sample_mean: dict[str, Any] = {}
    for field in ROW_EVAL_FIELDS:
        values = [
            row.get("evals", {}).get(field) if isinstance(row.get("evals"), Mapping) else None
            for row in rows
        ]
        summary = _summarize_numbers(values)
        sample_mean[field] = summary
        row_sums[field] = summary["sum"]

    return {
        "actual_invocation_totals": {
            # EXECUTED solver invocations only.
            "invocations": int(len(executed)),
            # Every keyed group, including the ones that executed nothing.
            "invocation_groups_attempted": int(len(order)),
            "groups_without_executed_work": int(len(zero_work)),
            "invocations_inconsistent": int(len(inconsistent)),
            "invocations_unmeasured": int(unmeasured),
            "rows_without_evals_block": int(rows_without_evals),
            "rows_without_invocation_key": int(
                rows_unkeyable_uncertain + rows_unkeyable_zero_work
            ),
            "rows_without_invocation_key_zero_work": int(rows_unkeyable_zero_work),
            "rows_without_invocation_key_uncertain": int(rows_unkeyable_uncertain),
            "rows_that_could_hide_work": int(uncertain_rows),
            "key_source": (
                "none"
                if not key_sources
                else (sorted(key_sources)[0] if len(key_sources) == 1 else "mixed")
            ),
            "totals": totals,
            "totals_over_known_invocations": known_totals,
            "unknown_invocations_per_field": unknown_per_field,
            "inconsistent_invocations": inconsistent[:8],
            "inconsistent_invocations_truncated": int(max(0, len(inconsistent) - 8)),
            "record_consumption": {
                "rows_consuming_a_record": int(record_consuming_rows),
                "by_accepted_path": _histogram(record_consuming_paths),
                "note": (
                    "Consuming an accepted-transition record is not the same as calling the "
                    "solver. Direct rollback consumes a record and never calls it; an ordinary "
                    "fallback after a failed C1 preflight consumes one too. Only the counters "
                    "decide whether an invocation executed."
                ),
            },
            "note": SOLVER_EVAL_SCOPE_NOTE,
            "zero_work_note": SOLVER_EVAL_ZERO_WORK_NOTE,
            "deduplication_note": SOLVER_EVAL_DEDUPLICATION_NOTE,
            "unknown_note": (
                "A row with no counters, or an unkeyable row that did not explicitly report "
                "zero work, could be hiding solver work: the exact totals are then null and "
                "totals_over_known_invocations carries the known subtotal. A partial sum is "
                "never presented as the full figure."
            ),
        },
        "sample_associated_mean": {
            **sample_mean,
            "note": (
                "Per-ROW statistics of the shared counters: an attribution of one invocation's "
                "cost to each sample that shared it. The 'sum' entries are the raw per-row sums "
                "and OVER-COUNT real work; they are published only so the raw number is never "
                "hidden. Use actual_invocation_totals for work done."
            ),
        },
        "row_sums_overcounted": row_sums,
        "per_row_opportunity": {
            "rows_total": int(len(rows)),
            "rows_with_evals": int(len(rows) - rows_without_evals),
            "rows_without_evals_block": int(rows_without_evals),
            "rows_per_invocation": _summarize_numbers(
                [groups[key]["rows"] for key in executed]
            ),
            "solver_group_size": _summarize_numbers(
                [row.get("solver_group_size") for row in rows]
            ),
            "rows_per_group": _summarize_numbers([groups[key]["rows"] for key in order]),
            "note": (
                "A diagnostics row is an opportunity to restitute, not a unit of solver work. "
                "solver_group_size is the solver's own group size and is NOT a divisor: a halted "
                "row emits no diagnostics row, so rows_per_invocation can be smaller. "
                "rows_per_invocation counts only groups that executed work; rows_per_group "
                "counts every keyed group, executed or not."
            ),
        },
    }


def summarize_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    scopes: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Aggregate the per-attempt ``candidate_c_diagnostics`` rows.

    Every rate carries its own explicit denominator, and a field the model did
    not supply raises ``null_count`` rather than contributing a zero.  The
    solver-side numbers are *reported*, not recomputed: this function never
    replays anything and never re-decides acceptance.

    ``scopes`` is an optional parallel sequence naming the namespace each row
    came from -- in practice the loader batch index.  It exists because the
    solver's evaluation counters are shared by every surviving row of one
    batched invocation, so they must be deduplicated per invocation rather than
    summed; without a per-batch namespace, a repeated ``(turn, record_turn)``
    pair in a later batch would be mistaken for the same invocation.  A caller
    summarising the rows of a single ``infer`` output may omit it.
    """

    if rows is None:
        return {
            "available": False,
            "reason": "model inference emitted no candidate_c_diagnostics key",
            "schema_version": int(CANDIDATE_C_DIAGNOSTIC_SCHEMA_VERSION),
            "rows_total": 0,
        }
    rows = list(rows)
    summary: dict[str, Any] = {
        "available": True,
        "schema_version": int(CANDIDATE_C_DIAGNOSTIC_SCHEMA_VERSION),
        "rows_total": int(len(rows)),
    }

    for field in ROW_NUMERIC_FIELDS:
        summary[f"numeric/{field}"] = _summarize_numbers([row.get(field) for row in rows])
    for field in ROW_COUNT_FIELDS:
        summary[f"count/{field}"] = _summarize_numbers([row.get(field) for row in rows])
    for field in ROW_FLAG_FIELDS:
        summary[f"flag/{field}"] = _summarize_flag([row.get(field) for row in rows])
    for field in ROW_CATEGORICAL_FIELDS:
        summary[f"histogram/{field}"] = _histogram(row.get(field) for row in rows)

    # Objective movement is only meaningful when BOTH endpoints were measured.
    deltas: list[float] = []
    pairs_missing = 0
    for row in rows:
        before = _finite_float(row.get("objective_before"))
        after = _finite_float(row.get("objective_after"))
        if before is None or after is None:
            pairs_missing += 1
            continue
        deltas.append(after - before)
    summary["numeric/objective_delta"] = _summarize_numbers(deltas)
    summary["numeric/objective_delta"]["pairs_missing"] = int(pairs_missing)
    summary["numeric/objective_delta"]["rows_considered"] = int(len(rows))
    strict_improved = sum(1 for delta in deltas if delta < 0.0)
    summary["objective_strictly_decreased_count"] = int(strict_improved)
    summary["objective_strictly_decreased_denominator"] = int(len(deltas))
    summary["objective_strictly_decreased_rate"] = _rate(strict_improved, len(deltas))

    # Actual solver work, deduplicated per invocation, kept strictly apart from
    # the per-row attribution of that shared cost.
    if scopes is None:
        scopes = [0] * len(rows)
    else:
        scopes = list(scopes)
        if len(scopes) != len(rows):
            raise ValueError(
                f"scopes must be parallel to rows: {len(scopes)} scopes for {len(rows)} rows"
            )
    summary["evals"] = _summarize_solver_evals(rows, scopes)

    # C1 replay health, restricted to rows that actually attempted a replay.
    replay_rows = [row for row in rows if _tri_state(row.get("c1_passed")) is not None]
    replay_passed = sum(1 for row in replay_rows if _tri_state(row.get("c1_passed")))
    summary["c1"] = {
        "attempted_count": int(len(replay_rows)),
        "passed_count": int(replay_passed),
        "failed_count": int(len(replay_rows) - replay_passed),
        "pass_rate": _rate(replay_passed, len(replay_rows)),
        "max_abs_err": _summarize_numbers([row.get("c1_max_abs_err") for row in replay_rows]),
        "rows_without_replay_attempt": int(len(rows) - len(replay_rows)),
        "note": (
            "Reported from the runtime solver's own replay result. This module does not "
            "re-execute the replay and cannot independently confirm it."
        ),
    }

    summary["note"] = (
        "Every value is read out of the inference-time rows. Missing fields are null, never 0. "
        "Rates are null when their denominator is 0."
    )
    return summary


# --------------------------------------------------------------------------
# Geometry metrics (ground-truth free)
# --------------------------------------------------------------------------


def _as_points(coordinates: Any, *, k: int | None = None) -> Tensor | None:
    """Normalise a coordinate tensor to ``[N, K, 2]`` float64 on CPU."""

    if not torch.is_tensor(coordinates):
        return None
    tensor = coordinates.detach().to(device="cpu", dtype=torch.float64)
    if tensor.ndim < 3 or tensor.shape[-1] != 2:
        return None
    if k is not None and int(tensor.shape[-2]) != int(k):
        return None
    return tensor.reshape(-1, int(tensor.shape[-2]), 2)


def _attention_probabilities(attention: Any) -> tuple[Tensor | None, bool, str | None]:
    """Validate attention and normalise it to ``[N, K]`` probabilities.

    Accepts the native ``[1, heads, H, W, K]`` layout and the flattened
    ``[1, heads, H*W, K]`` layout; the last axis is always ``K``.

    A malformed map is **rejected with a reason**, never repaired.  In
    particular a negative weight is not a probability: rescaling a row that
    contains one still yields negative "probabilities", and ``-p log p`` on
    those is not an entropy at all (``log`` of a negative number). Reporting
    such a map as unavailable is the only honest option.  Renormalisation is
    applied only to non-negative rows whose mass is merely off by scale, and
    the fact that it happened is reported.
    """

    if not torch.is_tensor(attention):
        return None, False, "attention is not a tensor"
    tensor = attention.detach().to(device="cpu", dtype=torch.float64)
    if tensor.ndim < 2 or tensor.shape[-1] < 1:
        return None, False, f"attention has unusable shape {tuple(tensor.shape)}"
    flat = tensor.reshape(-1, int(tensor.shape[-1]))
    if flat.numel() == 0:
        return None, False, "attention is empty"
    if not bool(torch.isfinite(flat).all()):
        return None, False, "attention contains non-finite values"
    if bool((flat < 0.0).any()):
        return (
            None,
            False,
            "attention contains negative weights; a negative weight is not a probability and "
            "renormalising it would produce an invalid entropy",
        )
    totals = flat.sum(dim=-1)
    if bool((totals <= 0.0).any()):
        return None, False, "attention has a row with zero total mass"
    renormalized = bool((totals - 1.0).abs().max().item() > 1e-3)
    return flat / totals.unsqueeze(-1), renormalized, None


def _saturation_block(
    points: Tensor,
    preclamp: Tensor | None,
    *,
    eps: float,
) -> dict[str, Any]:
    components = int(points.numel())
    at_edge = int((points.abs() >= (1.0 - float(eps))).sum().item())
    block: dict[str, Any] = {
        "components": components,
        "saturated_count": at_edge,
        "saturated_rate": _rate(at_edge, components),
        "saturation_eps": float(eps),
    }
    if preclamp is None:
        block["clamped_count"] = None
        block["clamped_rate"] = None
        block["max_preclamp_overshoot"] = None
        block["clamp_note"] = "coordinates_preclamp unavailable; clamp activity not measurable"
    else:
        overshoot = (preclamp.abs() - 1.0).clamp_min(0.0)
        clamped = int((overshoot > 0.0).sum().item())
        block["clamped_count"] = clamped
        block["clamped_rate"] = _rate(clamped, components)
        block["max_preclamp_overshoot"] = float(overshoot.max().item()) if components else None
    return block


def _duplicate_block(points_pixels: Tensor, *, tolerance_pixels: float) -> dict[str, Any]:
    """Count support points that coincide with an earlier point of the same query."""

    queries, k, _ = points_pixels.shape
    if k < 2 or queries == 0:
        return {
            "tolerance_pixels": float(tolerance_pixels),
            "points": int(queries * k),
            "duplicate_point_count": 0 if queries else None,
            "duplicate_point_rate": _rate(0, queries * k),
            "queries": int(queries),
            "queries_with_duplicate_count": 0 if queries else None,
            "queries_with_duplicate_rate": _rate(0, queries),
            "note": "K < 2: no intra-query pair exists" if k < 2 else None,
        }
    delta = (points_pixels.unsqueeze(2) - points_pixels.unsqueeze(1)).abs()
    close = (delta <= float(tolerance_pixels)).all(dim=-1)
    earlier = torch.tril(torch.ones(k, k, dtype=torch.bool), diagonal=-1)
    duplicate_point = (close & earlier.unsqueeze(0)).any(dim=-1)
    duplicate_points = int(duplicate_point.sum().item())
    queries_with = int(duplicate_point.any(dim=-1).sum().item())
    return {
        "tolerance_pixels": float(tolerance_pixels),
        "points": int(queries * k),
        "duplicate_point_count": duplicate_points,
        "duplicate_point_rate": _rate(duplicate_points, queries * k),
        "queries": int(queries),
        "queries_with_duplicate_count": queries_with,
        "queries_with_duplicate_rate": _rate(queries_with, queries),
        "note": None,
    }


def _spread_block(points_pixels: Tensor) -> dict[str, Any]:
    """Mean radial distance to the per-query support centroid, in feature pixels."""

    queries, k, _ = points_pixels.shape
    if queries == 0 or k == 0:
        return {"queries": int(queries), "mean_radius_pixels": None, "max_radius_pixels": None,
                "mean_axis_extent_pixels": None}
    centroid = points_pixels.mean(dim=1, keepdim=True)
    radius = (points_pixels - centroid).pow(2).sum(dim=-1).sqrt()
    extent = points_pixels.amax(dim=1) - points_pixels.amin(dim=1)
    return {
        "queries": int(queries),
        "mean_radius_pixels": float(radius.mean().item()),
        "max_radius_pixels": float(radius.max().item()),
        "mean_axis_extent_pixels": float(extent.mean().item()),
    }


def _to_pixels(points: Tensor, *, height: int, width: int) -> Tensor:
    """Map normalized ``align_corners=True`` coordinates to absolute pixels.

    A size-one axis maps to a constant ``0.0``: it has exactly one sample
    location, so every normalized value denotes the same pixel.
    """

    scale_x = (float(width) - 1.0) / 2.0 if int(width) > 1 else 0.0
    scale_y = (float(height) - 1.0) / 2.0 if int(height) > 1 else 0.0
    scale = torch.tensor([scale_x, scale_y], dtype=points.dtype)
    return (points + 1.0) * scale


def _displacement_block(
    factual: Tensor,
    chosen: Tensor,
    *,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Per-point displacement in FEATURE-PIXEL units, not normalized units."""

    step_x, step_y = normalized_pixel_step(height, width)
    degenerate = [name for name, step in (("x", step_x), ("y", step_y)) if step == 0.0]
    delta = chosen - factual
    scale = torch.tensor(
        [1.0 / step_x if step_x > 0.0 else 0.0, 1.0 / step_y if step_y > 0.0 else 0.0],
        dtype=delta.dtype,
    )
    pixels = delta * scale
    magnitude = pixels.pow(2).sum(dim=-1).sqrt().reshape(-1)
    points = int(magnitude.numel())
    moved = int((magnitude > 0.0).sum().item())
    return {
        "points": points,
        "degenerate_axes": degenerate,
        "mean_pixels": float(magnitude.mean().item()) if points else None,
        "max_pixels": float(magnitude.max().item()) if points else None,
        "max_abs_x_pixels": float(pixels[..., 0].abs().max().item()) if points else None,
        "max_abs_y_pixels": float(pixels[..., 1].abs().max().item()) if points else None,
        "moved_point_count": moved,
        "moved_point_rate": _rate(moved, points),
        "mean_moved_pixels": (
            float(magnitude[magnitude > 0.0].mean().item()) if moved else None
        ),
        "max_normalized": float(delta.abs().max().item()) if points else None,
        "note": (
            "A degenerate size-one axis contributes exactly 0 displacement: it has no physical "
            "extent, so no normalized move along it is real."
        ),
    }


def geometry_row_metrics(
    element: Mapping[str, Any],
    *,
    duplicate_tolerance_pixels: float = DEFAULT_DUPLICATE_TOLERANCE_PIXELS,
    saturation_eps: float = DEFAULT_SATURATION_EPS,
) -> dict[str, Any]:
    """Measure one ``candidate_c_geometry`` element.

    Returns ``{"available": False, "reason": ...}`` when the element is a
    declared-unavailable stub or is structurally unusable.  It never fabricates
    a coordinate, an attention map or a zero.
    """

    duplicate_tolerance_pixels, saturation_eps = _validate_geometry_limits(
        duplicate_tolerance_pixels, saturation_eps
    )
    base: dict[str, Any] = {
        "turn": element.get("turn"),
        "record_turn": element.get("record_turn"),
        "sample_index": element.get("sample_index"),
    }
    reason = element.get("unavailable_reason")
    if reason is not None:
        return {**base, "available": False, "reason": str(reason)}

    feature_hw = element.get("feature_hw")
    if not isinstance(feature_hw, (tuple, list)) or len(feature_hw) != 2:
        return {**base, "available": False, "reason": "feature_hw missing or malformed"}
    try:
        height, width = int(feature_hw[0]), int(feature_hw[1])
    except (TypeError, ValueError):
        return {**base, "available": False, "reason": "feature_hw is not a pair of integers"}
    if height < 1 or width < 1:
        return {
            **base,
            "available": False,
            "reason": f"feature_hw must be positive, got ({height}, {width})",
        }

    factual_depths = element.get("factual")
    chosen_depths = element.get("chosen")
    if not isinstance(factual_depths, (tuple, list)) or not isinstance(chosen_depths, (tuple, list)):
        return {**base, "available": False, "reason": "factual/chosen depth sequences missing"}
    if len(factual_depths) != len(chosen_depths):
        return {
            **base,
            "available": False,
            "reason": (
                f"factual depth count {len(factual_depths)} != chosen depth count "
                f"{len(chosen_depths)}"
            ),
        }
    if not factual_depths:
        return {**base, "available": False, "reason": "no internal depths captured"}

    k = element.get("k")
    k = int(k) if isinstance(k, int) else None

    depth_rows: list[dict[str, Any]] = []
    displacement_all: list[Tensor] = []
    chosen_all: list[Tensor] = []
    preclamp_all: list[Tensor] = []
    preclamp_missing = 0
    attention_entropies: list[Tensor] = []
    attention_missing = 0
    attention_renormalized = 0
    attention_reasons: list[str] = []

    for index, (factual_entry, chosen_entry) in enumerate(zip(factual_depths, chosen_depths)):
        if not isinstance(factual_entry, Mapping) or not isinstance(chosen_entry, Mapping):
            depth_rows.append({"depth_index": index, "available": False,
                               "reason": "depth entry is not a mapping"})
            continue
        factual_points = _as_points(factual_entry.get("coordinates"), k=k)
        chosen_points = _as_points(chosen_entry.get("coordinates"), k=k)
        if factual_points is None or chosen_points is None:
            depth_rows.append({"depth_index": index, "available": False,
                               "reason": "coordinates missing or wrong shape"})
            continue
        if factual_points.shape != chosen_points.shape:
            depth_rows.append({"depth_index": index, "available": False,
                               "reason": "factual/chosen coordinate shapes differ"})
            continue
        if not bool(torch.isfinite(factual_points).all()) or not bool(torch.isfinite(chosen_points).all()):
            depth_rows.append({"depth_index": index, "available": False,
                               "reason": "non-finite coordinates"})
            continue
        depth_k = int(chosen_points.shape[-2])
        if k is None:
            k = depth_k
        elif depth_k != k:
            # Pooling across depths assumes one support size; a mixed-K element
            # is reported as such rather than silently reshaped.
            depth_rows.append({"depth_index": index, "available": False,
                               "reason": f"support size {depth_k} != element k {k}"})
            continue

        chosen_pixels = _to_pixels(chosen_points, height=height, width=width)
        preclamp = _as_points(chosen_entry.get("coordinates_preclamp"), k=int(chosen_points.shape[-2]))
        if (
            preclamp is None
            or preclamp.shape != chosen_points.shape
            or not bool(torch.isfinite(preclamp).all())
        ):
            # A non-finite preclamp cannot measure clamp activity; it is
            # reported missing rather than folded into a bogus overshoot.
            preclamp = None
            preclamp_missing += 1
        else:
            preclamp_all.append(preclamp)

        probabilities, renormalized, attention_reason = _attention_probabilities(
            chosen_entry.get("attention")
        )
        if probabilities is None:
            attention_missing += 1
            attention_reasons.append(attention_reason or "attention unavailable")
            attention_block: dict[str, Any] = {
                "available": False,
                "reason": attention_reason or "attention unavailable",
            }
        else:
            if renormalized:
                attention_renormalized += 1
            entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            attention_entropies.append(entropy)
            support = int(probabilities.shape[-1])
            attention_block = {
                "available": True,
                "queries": int(entropy.numel()),
                "support": support,
                "mean_entropy_nats": float(entropy.mean().item()),
                "min_entropy_nats": float(entropy.min().item()),
                "max_entropy_nats": float(entropy.max().item()),
                "max_possible_entropy_nats": float(math.log(support)) if support > 1 else 0.0,
                "mean_normalized_entropy": (
                    float((entropy / math.log(support)).mean().item()) if support > 1 else None
                ),
                "renormalized": bool(renormalized),
            }

        displacement_all.append(
            torch.stack([factual_points, chosen_points], dim=0)
        )
        chosen_all.append(chosen_points)
        depth_rows.append(
            {
                "depth_index": int(factual_entry.get("depth_index", index) or index),
                "available": True,
                "turn_index": factual_entry.get("turn_index"),
                "iteration_index": factual_entry.get("iteration_index"),
                "state_identity": factual_entry.get("state_identity"),
                "state_identity_matches_chosen": (
                    factual_entry.get("state_identity") == chosen_entry.get("state_identity")
                ),
                "displacement": _displacement_block(
                    factual_points, chosen_points, height=height, width=width
                ),
                "saturation": _saturation_block(chosen_points, preclamp, eps=saturation_eps),
                "duplicates": _duplicate_block(
                    chosen_pixels, tolerance_pixels=duplicate_tolerance_pixels
                ),
                "spread": _spread_block(chosen_pixels),
                "attention_entropy": attention_block,
            }
        )

    usable = [row for row in depth_rows if row.get("available")]
    if not usable:
        return {
            **base,
            "available": False,
            "reason": "no usable depth entry",
            "depths": depth_rows,
        }

    stacked = torch.cat([pair.reshape(2, -1, int(pair.shape[-2]), 2) for pair in displacement_all], dim=1)
    all_chosen = torch.cat(chosen_all, dim=0)
    all_chosen_pixels = _to_pixels(all_chosen, height=height, width=width)
    all_preclamp = torch.cat(preclamp_all, dim=0) if len(preclamp_all) == len(usable) else None

    entropy_summary: dict[str, Any]
    if attention_entropies:
        merged = torch.cat([entry.reshape(-1) for entry in attention_entropies], dim=0)
        support = int(all_chosen.shape[-2])
        entropy_summary = {
            "available": True,
            "queries": int(merged.numel()),
            "support": support,
            "mean_entropy_nats": float(merged.mean().item()),
            "min_entropy_nats": float(merged.min().item()),
            "max_entropy_nats": float(merged.max().item()),
            "mean_normalized_entropy": (
                float((merged / math.log(support)).mean().item()) if support > 1 else None
            ),
            "depths_missing_attention": int(attention_missing),
            "depths_renormalized": int(attention_renormalized),
        }
    else:
        # Keep the *specific* rejection reason: "contains negative weights" and
        # "is not a tensor" mean different things and must not collapse into a
        # single opaque "unavailable".
        distinct = sorted(set(attention_reasons))
        entropy_summary = {
            "available": False,
            "reason": (
                distinct[0]
                if len(distinct) == 1
                else "no depth supplied a usable attention map"
            ),
            "reasons": _histogram(attention_reasons),
            "depths_missing_attention": int(attention_missing),
        }

    return {
        **base,
        "available": True,
        "feature_hw": [height, width],
        "k": int(all_chosen.shape[-2]),
        "depths_captured": int(len(depth_rows)),
        "depths_usable": int(len(usable)),
        "displacement": _displacement_block(stacked[0], stacked[1], height=height, width=width),
        "saturation": _saturation_block(all_chosen, all_preclamp, eps=saturation_eps),
        "saturation_preclamp_depths_missing": int(preclamp_missing),
        "duplicates": _duplicate_block(
            all_chosen_pixels, tolerance_pixels=duplicate_tolerance_pixels
        ),
        "spread": _spread_block(all_chosen_pixels),
        "attention_entropy": entropy_summary,
        "depths": depth_rows,
    }


def summarize_geometry_rows(
    row_metrics: Sequence[Mapping[str, Any]],
    *,
    duplicate_tolerance_pixels: float = DEFAULT_DUPLICATE_TOLERANCE_PIXELS,
    saturation_eps: float = DEFAULT_SATURATION_EPS,
    keep_rows: bool = True,
) -> dict[str, Any]:
    """Pool already-measured :func:`geometry_row_metrics` results.

    This is the streaming entry point.  ``geometry_row_metrics`` returns pure
    numbers -- no tensors, no views onto the payload it measured -- so a caller
    can measure one batch, keep only these small dictionaries and drop the raw
    ``candidate_c_geometry`` payload (image-sized coordinate and attention
    tensors, possibly on the GPU) before the next batch is fetched.  Holding
    the raw payload for a whole validation cohort is what made the earlier
    version's memory grow without bound.
    """

    duplicate_tolerance_pixels, saturation_eps = _validate_geometry_limits(
        duplicate_tolerance_pixels, saturation_eps
    )
    rows = list(row_metrics)
    usable = [row for row in rows if row.get("available")]
    unavailable_reasons = _histogram(
        row.get("reason") for row in rows if not row.get("available")
    )

    def _pool(path: Sequence[str]) -> dict[str, Any]:
        values: list[Any] = []
        for row in usable:
            node: Any = row
            for key in path:
                node = node.get(key) if isinstance(node, Mapping) else None
            values.append(node)
        return _summarize_numbers(values)

    summary: dict[str, Any] = {
        "available": bool(usable),
        "schema_version": int(CANDIDATE_C_DIAGNOSTIC_SCHEMA_VERSION),
        "rows_total": int(len(rows)),
        "rows_usable": int(len(usable)),
        "rows_unavailable": int(len(rows) - len(usable)),
        "unavailable_reasons": unavailable_reasons,
        "duplicate_tolerance_pixels": float(duplicate_tolerance_pixels),
        "saturation_eps": float(saturation_eps),
    }
    if not usable:
        summary.setdefault(
            "reason", "no geometry element was usable; see unavailable_reasons"
        )
    else:
        summary["displacement_mean_pixels"] = _pool(("displacement", "mean_pixels"))
        summary["displacement_max_pixels"] = _pool(("displacement", "max_pixels"))
        summary["displacement_moved_point_rate"] = _pool(("displacement", "moved_point_rate"))
        summary["saturation_rate"] = _pool(("saturation", "saturated_rate"))
        summary["clamped_rate"] = _pool(("saturation", "clamped_rate"))
        summary["duplicate_point_rate"] = _pool(("duplicates", "duplicate_point_rate"))
        summary["queries_with_duplicate_rate"] = _pool(("duplicates", "queries_with_duplicate_rate"))
        summary["attention_mean_entropy_nats"] = _pool(("attention_entropy", "mean_entropy_nats"))
        summary["attention_mean_normalized_entropy"] = _pool(
            ("attention_entropy", "mean_normalized_entropy")
        )
        summary["spread_mean_radius_pixels"] = _pool(("spread", "mean_radius_pixels"))
        summary["spread_mean_axis_extent_pixels"] = _pool(("spread", "mean_axis_extent_pixels"))
    if keep_rows:
        summary["rows"] = rows
    summary["note"] = (
        "Displacement, duplicate tolerance and spread are in FEATURE PIXELS. Attention entropy is "
        "in nats; the normalized form divides by log(K) and is null when K == 1. Every value here "
        "is a plain number: no tensor from the measured payload is retained."
    )
    return summary


def summarize_geometry(
    geometry: Sequence[Mapping[str, Any]] | None,
    *,
    duplicate_tolerance_pixels: float = DEFAULT_DUPLICATE_TOLERANCE_PIXELS,
    saturation_eps: float = DEFAULT_SATURATION_EPS,
    keep_rows: bool = True,
) -> dict[str, Any]:
    """Measure every ``candidate_c_geometry`` element and pool the results.

    A convenience wrapper for a caller that already holds the whole payload.
    A streaming caller should call :func:`geometry_row_metrics` per batch and
    pool with :func:`summarize_geometry_rows`, so the raw tensors can be
    released between batches.
    """

    duplicate_tolerance_pixels, saturation_eps = _validate_geometry_limits(
        duplicate_tolerance_pixels, saturation_eps
    )
    if geometry is None:
        return {
            "available": False,
            "reason": "inference emitted no candidate_c_geometry (capture_geometry was not enabled)",
            "schema_version": int(CANDIDATE_C_DIAGNOSTIC_SCHEMA_VERSION),
            "rows_total": 0,
        }
    return summarize_geometry_rows(
        [
            geometry_row_metrics(
                element,
                duplicate_tolerance_pixels=duplicate_tolerance_pixels,
                saturation_eps=saturation_eps,
            )
            for element in geometry
        ],
        duplicate_tolerance_pixels=duplicate_tolerance_pixels,
        saturation_eps=saturation_eps,
        keep_rows=keep_rows,
    )


# --------------------------------------------------------------------------
# Evaluation-only ground-truth metrics
# --------------------------------------------------------------------------


def _argmax_labels(logits: Tensor) -> Tensor:
    if logits.ndim != 4:
        raise ValueError(f"expected logits [N,C,H,W], got {tuple(logits.shape)}")
    return logits.detach().argmax(dim=1)


def _record_turn_hint(
    geometry: Sequence[Mapping[str, Any]] | None,
    position: int,
    row: Mapping[str, Any],
) -> int | None:
    """Recover ``record_turn`` from the index-aligned geometry element.

    ``candidate_c_geometry`` is emitted in the same order as
    ``candidate_c_diagnostics``, so element ``position`` describes this row.
    The hint is only trusted when the element agrees with the row on both
    ``turn`` and ``sample_index``; a disagreement means the alignment
    assumption is broken and no hint is returned.  The row's own
    ``record_turn`` always wins when it is present.
    """

    if geometry is None or position >= len(geometry):
        return None
    element = geometry[position]
    if not isinstance(element, Mapping):
        return None
    if element.get("turn") != row.get("turn"):
        return None
    if element.get("sample_index") != row.get("sample_index"):
        return None
    hint = element.get("record_turn")
    if isinstance(hint, bool) or not isinstance(hint, int):
        return None
    return hint


def _row_triplet_indices(
    row: Mapping[str, Any],
    *,
    num_turns: int,
    batch_size: int,
    record_turn_hint: int | None = None,
) -> tuple[int, int, str | None]:
    """Resolve ``(record_turn, sample_index)`` or explain why the row is unusable.

    Admission is decided by **history, not by outcome**.  Any row that names a
    valid earlier ``record_turn`` has a real pretransition/factual/current
    triplet and is scored, whatever its ``accepted_path`` was and whether or
    not the Auditor accepted it.  Selecting only accepted successful
    restitutions would condition the correction rates on success and report a
    number no deployment ever sees.

    Only a row with no history at all -- an ordinary annotation turn that
    replayed nothing -- is unavailable, because for it no pretransition state
    exists to attribute against.
    """

    record_turn = row.get("record_turn")
    if record_turn is None and record_turn_hint is not None:
        record_turn = record_turn_hint
    turn = row.get("turn")
    sample_index = row.get("sample_index")
    if not isinstance(record_turn, int) or isinstance(record_turn, bool):
        return -1, -1, "no accepted-history record_turn: nothing to attribute against"
    if not isinstance(turn, int) or isinstance(turn, bool):
        return -1, -1, "turn missing or not an int"
    if not isinstance(sample_index, int) or isinstance(sample_index, bool):
        return -1, -1, "sample_index missing or not an int"
    if not 0 <= record_turn < num_turns or not 0 <= turn < num_turns:
        return -1, -1, "turn/record_turn out of range for the supplied transition lists"
    if record_turn >= turn:
        return -1, -1, "record_turn must precede the current turn"
    if not 0 <= sample_index < batch_size:
        return -1, -1, "sample_index out of range for the supplied batch"
    return record_turn, sample_index, None


def _effect_counts(
    *,
    p_ok: Tensor,
    f_ok: Tensor,
    after_ok: Tensor,
    valid: Tensor,
) -> dict[str, int]:
    """Attribute one "after" state against the prior transition ``P -> F``."""

    prior_true_fix = (~p_ok) & f_ok & valid
    prior_true_regress = p_ok & (~f_ok) & valid
    outside = f_ok & (~prior_true_fix)
    return {
        "prior_true_fix": int(prior_true_fix.sum().item()),
        "prior_true_fix_retained": int((prior_true_fix & after_ok).sum().item()),
        "prior_true_fix_destroyed": int((prior_true_fix & (~after_ok)).sum().item()),
        "prior_true_regress": int(prior_true_regress.sum().item()),
        "prior_true_regress_repaired": int((prior_true_regress & after_ok).sum().item()),
        "factual_correct": int(f_ok.sum().item()),
        "new_error": int((f_ok & (~after_ok)).sum().item()),
        "factual_correct_outside_prior_fix": int(outside.sum().item()),
        "new_error_outside_prior_fix": int((outside & (~after_ok)).sum().item()),
        "valid_pixels": int(valid.sum().item()),
        "after_correct": int(after_ok.sum().item()),
    }


#: The count keys every effect family carries.  Kept explicit so a merge can
#: pool an empty family without inventing keys.
EFFECT_COUNT_KEYS: tuple[str, ...] = (
    "prior_true_fix",
    "prior_true_fix_retained",
    "prior_true_fix_destroyed",
    "prior_true_regress",
    "prior_true_regress_repaired",
    "factual_correct",
    "new_error",
    "factual_correct_outside_prior_fix",
    "new_error_outside_prior_fix",
    "valid_pixels",
    "after_correct",
)

#: The two effect families.  ``proposal`` is what the solver put forward;
#: ``retained`` is what the row actually kept after the official audit gate.
EFFECT_FAMILIES: tuple[str, ...] = ("proposal", "retained")


def _effect_block(totals: Mapping[str, int], *, rows_scored: int, family: str) -> dict[str, Any]:
    """Turn pooled counts into rates, each beside the denominator it used."""

    def total(key: str) -> int:
        return int(totals.get(key, 0))

    return {
        "family": str(family),
        "rows_scored": int(rows_scored),
        "counts": {key: total(key) for key in EFFECT_COUNT_KEYS},
        "denominators": {
            "prior_true_fix": total("prior_true_fix"),
            "prior_true_regress": total("prior_true_regress"),
            "factual_correct": total("factual_correct"),
            "factual_correct_outside_prior_fix": total("factual_correct_outside_prior_fix"),
            "valid_pixels": total("valid_pixels"),
        },
        "prior_true_fix_retained_rate": _rate(
            total("prior_true_fix_retained"), total("prior_true_fix")
        ),
        "prior_true_fix_destroyed_rate": _rate(
            total("prior_true_fix_destroyed"), total("prior_true_fix")
        ),
        "prior_true_regress_repaired_rate": _rate(
            total("prior_true_regress_repaired"), total("prior_true_regress")
        ),
        "new_error_rate": _rate(total("new_error"), total("factual_correct")),
        "new_error_outside_prior_fix_rate": _rate(
            total("new_error_outside_prior_fix"), total("factual_correct_outside_prior_fix")
        ),
        "net_correct_pixel_delta": int(total("after_correct") - total("factual_correct")),
        "net_correct_rate_delta": (
            None
            if not total("valid_pixels")
            else float(total("after_correct") - total("factual_correct"))
            / float(total("valid_pixels"))
        ),
    }


PROPOSAL_NOTE = (
    "proposal = the candidate the Auditor judged at this turn, scored whether or not it was "
    "accepted. It measures what the solver put forward, never what the model kept."
)
RETAINED_NOTE = (
    "retained = the state the row actually kept: the candidate when the Auditor accepted it, "
    "and the unchanged factual state when it rejected (reject HALTs the row). A rejected "
    "proposed repair therefore never appears as a realized repair."
)
GT_EVALUATION_ONLY_NOTE = (
    "Evaluation-only. Ground truth scores already-computed logit tensors and never reaches "
    "infer(), the Annotation Expert, the Auditor or the Candidate C solver."
)


def ground_truth_restitution_metrics(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    transition_previous: Sequence[Tensor] | None,
    transition_candidates: Sequence[Tensor] | None,
    target: Tensor | None,
    ignore_index: int | None = None,
    geometry: Sequence[Mapping[str, Any]] | None = None,
    keep_per_row: bool = True,
) -> dict[str, Any]:
    """EVALUATION ONLY.  Score already-computed restitution attempts against truth.

    Nothing here is ever handed back to the model.  Every logit tensor and
    every row field was produced ground-truth-free by inference; this function
    only *labels* them.

    Temporal alignment (fixed by the architecture contract), for a row whose
    replayed record came from ``record_turn`` and that was judged at ``turn``:

    ``P`` pretransition
        ``transition_previous[record_turn][sample_index]`` -- the state before
        the previous accepted ORDINARY transition.
    ``F`` factual
        ``transition_candidates[record_turn][sample_index]`` -- that ordinary
        transition's candidate, which is also the state retained entering this
        turn.
    ``Q`` proposal
        ``transition_candidates[turn][sample_index]`` -- the candidate the
        official Auditor judged at this turn.
    ``R`` retained
        ``Q`` when the row was accepted, otherwise ``F``.

    Two families are reported from the same masks, differing only in which
    tensor plays the "after" role:

    ``proposal``
        after = ``Q``.  What the solver put forward, scored regardless of the
        audit decision.
    ``retained``
        after = ``R``.  What the model actually kept.  A rejected proposal
        contributes its *unchanged* factual state here, so a rejected proposed
        repair is never counted as a realized repair.

    Masks over pixels valid in ``target`` (``g``), with ``p = argmax P``,
    ``f = argmax F`` and ``a`` the family's "after" argmax:

    ``prior_true_fix``   ``(p != g) & (f == g)``
        the previous ordinary transition genuinely corrected the pixel.
        ``retained`` adds ``a == g``; ``destroyed`` adds ``a != g``.
        Denominator ``|prior_true_fix|``.
    ``prior_true_regress``   ``(p == g) & (f != g)``
        the previous ordinary transition genuinely broke the pixel.
        ``repaired`` adds ``a == g``.  Denominator ``|prior_true_regress|``.
    ``new_error``   ``(f == g) & (a != g)``
        correct before, wrong after.  Denominator ``|f == g|``.
        ``new_error_outside_prior_fix`` removes ``prior_true_fix`` from both
        numerator and denominator so a destroyed fix is not double counted.

    Row admission is by history, not by outcome: every row naming a valid
    earlier ``record_turn`` is scored, including ``factual_support`` identity
    fallbacks, rejected proposals and eligible rows that fell back to ordinary
    annotation.  ``strata`` breaks the scored rows down by acceptance,
    ``accepted_path``, ``fallback_reason`` and ``eligible`` so a conditional
    read is still possible without the pooled numbers being conditioned.

    A rate whose denominator is zero is ``None``.  A row that cannot be aligned
    is counted in ``rows_unavailable`` with its reason; it is never scored as 0.
    """

    block: dict[str, Any] = {
        "schema_version": int(CANDIDATE_C_DIAGNOSTIC_SCHEMA_VERSION),
        "evaluation_only": bool(GT_METRICS_ARE_EVALUATION_ONLY),
        "note": GT_EVALUATION_ONLY_NOTE,
        "proposal_note": PROPOSAL_NOTE,
        "retained_note": RETAINED_NOTE,
    }
    if rows is None:
        return {**block, "available": False, "reason": "no candidate_c_diagnostics rows",
                "rows_total": 0}
    rows = list(rows)
    if target is None:
        return {**block, "available": False, "reason": "no ground-truth target supplied",
                "rows_total": int(len(rows))}
    if not transition_previous or not transition_candidates:
        return {**block, "available": False,
                "reason": "transition_previous/transition_candidates unavailable",
                "rows_total": int(len(rows))}
    num_turns = min(len(transition_previous), len(transition_candidates))
    batch_size = int(target.shape[0])

    truth = target.detach()
    if truth.ndim == 4 and truth.shape[1] == 1:
        truth = truth[:, 0]
    if truth.ndim != 3:
        return {**block, "available": False,
                "reason": f"target must be [N,H,W] or [N,1,H,W], got {tuple(target.shape)}",
                "rows_total": int(len(rows))}
    truth = truth.to(torch.long)

    totals: dict[str, dict[str, int]] = {
        family: {key: 0 for key in EFFECT_COUNT_KEYS} for family in EFFECT_FAMILIES
    }
    scored_per_family = {family: 0 for family in EFFECT_FAMILIES}
    per_row: list[dict[str, Any]] = []
    unavailable_reasons: list[str | None] = []
    strata_accepted: list[Any] = []
    strata_path: list[Any] = []
    strata_fallback: list[Any] = []
    strata_eligible: list[Any] = []

    for position, row in enumerate(rows):
        record_turn, sample_index, reason = _row_triplet_indices(
            row,
            num_turns=num_turns,
            batch_size=batch_size,
            record_turn_hint=_record_turn_hint(geometry, position, row),
        )
        if reason is not None:
            unavailable_reasons.append(reason)
            continue
        turn = int(row["turn"])
        try:
            pre = transition_previous[record_turn][sample_index : sample_index + 1]
            factual = transition_candidates[record_turn][sample_index : sample_index + 1]
            proposal = transition_candidates[turn][sample_index : sample_index + 1]
        except (IndexError, TypeError) as error:  # pragma: no cover - defensive
            unavailable_reasons.append(f"transition lookup failed: {error}")
            continue
        gt = truth[sample_index : sample_index + 1]
        if pre.shape[-2:] != gt.shape[-2:]:
            unavailable_reasons.append("logit spatial shape does not match the target grid")
            continue

        accepted = _tri_state(row.get("accepted"))
        if accepted is None:
            unavailable_reasons.append(
                "acceptance is unmeasured, so the retained state is undetermined"
            )
            continue

        valid = torch.ones_like(gt, dtype=torch.bool)
        if ignore_index is not None:
            valid &= gt != int(ignore_index)
        valid &= (gt >= 0) & (gt < int(pre.shape[1]))

        p_ok = (_argmax_labels(pre) == gt) & valid
        f_ok = (_argmax_labels(factual) == gt) & valid
        q_ok = (_argmax_labels(proposal) == gt) & valid
        # Reject HALTs the row, so the retained state is the unchanged factual
        # one.  This is a selection of tensors, never a re-decision.
        r_ok = q_ok if accepted else f_ok

        row_counts = {
            "proposal": _effect_counts(p_ok=p_ok, f_ok=f_ok, after_ok=q_ok, valid=valid),
            "retained": _effect_counts(p_ok=p_ok, f_ok=f_ok, after_ok=r_ok, valid=valid),
        }
        for family in EFFECT_FAMILIES:
            scored_per_family[family] += 1
            for key, value in row_counts[family].items():
                totals[family][key] += value

        strata_accepted.append(accepted)
        strata_path.append(row.get("accepted_path"))
        strata_fallback.append(row.get("fallback_reason"))
        strata_eligible.append(_tri_state(row.get("eligible")))
        if keep_per_row:
            per_row.append(
                {
                    "turn": turn,
                    "record_turn": record_turn,
                    "sample_index": sample_index,
                    "accepted": accepted,
                    "accepted_path": row.get("accepted_path"),
                    "eligible": _tri_state(row.get("eligible")),
                    "fallback_reason": row.get("fallback_reason"),
                    "proposal": row_counts["proposal"],
                    "retained": row_counts["retained"],
                }
            )

    scored = scored_per_family["proposal"]
    result: dict[str, Any] = {
        **block,
        "available": scored > 0,
        "rows_total": int(len(rows)),
        "rows_scored": int(scored),
        "rows_unavailable": int(len(rows) - scored),
        "unavailable_reasons": _histogram(unavailable_reasons),
        "strata": {
            "by_accepted": _histogram(strata_accepted),
            "by_accepted_path": _histogram(strata_path),
            "by_fallback_reason": _histogram(strata_fallback),
            "by_eligible": _histogram(strata_eligible),
        },
        "proposal": _effect_block(
            totals["proposal"], rows_scored=scored_per_family["proposal"], family="proposal"
        ),
        "retained": _effect_block(
            totals["retained"], rows_scored=scored_per_family["retained"], family="retained"
        ),
        "per_row": per_row,
    }
    if not scored:
        result["reason"] = "no row carried an alignable pretransition/factual/current triplet"
    return result


def merge_ground_truth_blocks(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool per-batch :func:`ground_truth_restitution_metrics` blocks.

    Counts add and rates are recomputed from the pooled counts, so a batch that
    scored one pixel never gets the same weight as a batch that scored
    thousands.  A pooled rate with a zero denominator stays ``None``.  Both
    effect families and all four strata are pooled; ``per_row`` is dropped,
    because the pooled block is what the report carries.
    """

    totals: dict[str, dict[str, int]] = {
        family: {key: 0 for key in EFFECT_COUNT_KEYS} for family in EFFECT_FAMILIES
    }
    scored_per_family = {family: 0 for family in EFFECT_FAMILIES}
    strata: dict[str, dict[str, int]] = {
        name: {} for name in ("by_accepted", "by_accepted_path", "by_fallback_reason", "by_eligible")
    }
    rows_total = 0
    rows_scored = 0
    reasons: list[str] = []
    for block in blocks:
        rows_total += int(block.get("rows_total", 0) or 0)
        rows_scored += int(block.get("rows_scored", 0) or 0)
        for key, count in (block.get("unavailable_reasons") or {}).items():
            reasons.extend([str(key)] * int(count))
        for name, bucket in (block.get("strata") or {}).items():
            if name not in strata:
                strata[name] = {}
            for key, count in (bucket or {}).items():
                strata[name][str(key)] = strata[name].get(str(key), 0) + int(count)
        for family in EFFECT_FAMILIES:
            family_block = block.get(family)
            if not isinstance(family_block, Mapping):
                continue
            scored_per_family[family] += int(family_block.get("rows_scored", 0) or 0)
            for key, value in (family_block.get("counts") or {}).items():
                if key in totals[family]:
                    totals[family][key] += int(value)

    return {
        "schema_version": int(CANDIDATE_C_DIAGNOSTIC_SCHEMA_VERSION),
        "evaluation_only": bool(GT_METRICS_ARE_EVALUATION_ONLY),
        "available": rows_scored > 0,
        "reason": (
            None
            if rows_scored
            else "no row across the measured batches carried an alignable triplet"
        ),
        "rows_total": int(rows_total),
        "rows_scored": int(rows_scored),
        "rows_unavailable": int(rows_total - rows_scored),
        "unavailable_reasons": _histogram(reasons),
        "strata": {name: dict(sorted(bucket.items())) for name, bucket in strata.items()},
        "proposal": _effect_block(
            totals["proposal"], rows_scored=scored_per_family["proposal"], family="proposal"
        ),
        "retained": _effect_block(
            totals["retained"], rows_scored=scored_per_family["retained"], family="retained"
        ),
        "note": GT_EVALUATION_ONLY_NOTE,
        "proposal_note": PROPOSAL_NOTE,
        "retained_note": RETAINED_NOTE,
    }


# --------------------------------------------------------------------------
# Convenience entry point
# --------------------------------------------------------------------------


def evaluate_candidate_c(
    infer_output: Mapping[str, Any],
    *,
    target: Tensor | None = None,
    duplicate_tolerance_pixels: float = DEFAULT_DUPLICATE_TOLERANCE_PIXELS,
    saturation_eps: float = DEFAULT_SATURATION_EPS,
    ignore_index: int | None = None,
    keep_geometry_rows: bool = False,
    keep_ground_truth_rows: bool = True,
) -> dict[str, Any]:
    """Measure one ``SelfAuditNet.infer`` output.

    ``target`` is optional and is used **only** for the evaluation-only
    ground-truth block; omitting it leaves that block explicitly unavailable
    and changes nothing else.  This function does not call the model.
    """

    rows = infer_output.get("candidate_c_diagnostics")
    if rows is not None and not isinstance(rows, Sequence):
        rows = None
    geometry = infer_output.get("candidate_c_geometry")
    if geometry is not None and not isinstance(geometry, Sequence):
        geometry = None

    return {
        "schema_version": int(CANDIDATE_C_DIAGNOSTIC_SCHEMA_VERSION),
        "rows": summarize_diagnostic_rows(rows),
        "geometry": summarize_geometry(
            geometry,
            duplicate_tolerance_pixels=duplicate_tolerance_pixels,
            saturation_eps=saturation_eps,
            keep_rows=keep_geometry_rows,
        ),
        "ground_truth": ground_truth_restitution_metrics(
            rows,
            transition_previous=infer_output.get("transition_previous"),
            transition_candidates=infer_output.get("transition_candidates"),
            target=target,
            ignore_index=ignore_index,
            geometry=geometry,
            keep_per_row=bool(keep_ground_truth_rows),
        ),
    }
