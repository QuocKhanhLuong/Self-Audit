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
import math
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from ..artifact_io import atomic_write_json, json_safe_artifact

from ..audit.semantics import (
    METRIC_SPACES,
    METRIC_SPACE_SLICE_PROXY,
    METRIC_SPACE_VOLUME_NATIVE,
    METRIC_SPACE_VOLUME_RESIZED,
    beneficial_mask,
    empty_class_score,
    harmful_mask,
    neutral_mask,
    resolve_neutral_margin,
)
from .contracts import (
    ContractMismatchError,
    FOREGROUND_DICE_EXCLUDE_V1_CONTRACT,
    FOREGROUND_DICE_VOLUME_RESIZED_V1_CONTRACT,
    MetricContract,
    SufficientStatistics,
    compute_dice_from_stats,
    resolve_metric_contract,
    validate_contract,
    validate_contract_compatibility,
)


class NoFeasibleThresholdError(ValueError):
    """Raised when no threshold in a grid satisfies the specified constraint or cohort has no usable outcome."""



# --------------------------------------------------------------------------
# Persisted calibration artifact
# --------------------------------------------------------------------------

#: Schema v2 adds the top-level ``lineage`` block: the exact weights,
#: preprocessing recipe, metric semantics, split membership and cohort the
#: threshold was measured under.  v1 artifacts predate it and are *not*
#: loadable here; they are readable only through
#: ``evaluation.calibration_lineage.inspect_legacy_calibration``, which always
#: reports them as unverified.
CALIBRATION_SCHEMA_VERSION = 2

#: Versions this build recognises but refuses to load for calibrated use.
LEGACY_CALIBRATION_SCHEMA_VERSIONS: tuple[int, ...] = (1,)

#: Schema version of the validation transition cache mapping produced by
#: :func:`collect_validation_transition_cache` and consumed by :func:`sweep_thresholds`.
CACHE_SCHEMA_VERSION = 1
SUPPORTED_CACHE_SCHEMA_VERSIONS = (CACHE_SCHEMA_VERSION,)

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
    "metric_contract",
    "metric_contract_version",
    "lineage",
    "validity",
    "validity_reason",
    "extra",
)

#: ``lineage`` is structurally optional so a producer that cannot build one
#: still writes an honest artifact -- but it is written as an explicit ``null``
#: and every deployable consumer rejects that artifact.  Absence is never
#: silence here; see :func:`load_calibration`.
_CALIBRATION_OPTIONAL: tuple[str, ...] = (
    "extra",
    "metric_contract",
    "metric_contract_version",
    "lineage",
)
_CALIBRATION_REQUIRED: tuple[str, ...] = tuple(
    k for k in _CALIBRATION_KEYS if k not in _CALIBRATION_OPTIONAL
)


def _array(value: Any, *, name: str, allow_nan: bool = False) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not allow_nan:
        if not np.isfinite(result).all():
            raise ValueError(f"{name} contains NaN or Inf")
    else:
        if np.isinf(result).any():
            raise ValueError(f"{name} contains Inf")
    return result


def _transition_arrays(
    delta_q: Any,
    actual_delta_dice: Any,
    active_mask: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quality = _array(delta_q, name="delta_q", allow_nan=False)
    actual = _array(actual_delta_dice, name="actual_delta_dice", allow_nan=True)
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
        active = _array(active_mask, name="active_mask", allow_nan=False).astype(bool, copy=False)
        if active.ndim == 1:
            active = active[:, None]
        if active.shape != quality.shape:
            raise ValueError(f"active_mask shape {active.shape} does not match transitions {quality.shape}")
    return quality.astype(np.float64), actual.astype(np.float64), active


def _validate_stats_array(
    raw: Any,
    name: str,
    expected_ndim: int,
    expected_prefix: tuple[int, ...],
) -> np.ndarray:
    if raw is None:
        raise ContractMismatchError(f"Sufficient statistics '{name}' cannot be None")
    arr = np.asarray(raw)
    if arr.dtype == bool or not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise ContractMismatchError(f"Sufficient statistics '{name}' must have numeric integer dtype, got {arr.dtype}")
    if arr.ndim != expected_ndim:
        raise ContractMismatchError(
            f"Bogus or mismatched shape for '{name}': got {arr.shape}, expected {expected_ndim} dimensions with prefix {expected_prefix}"
        )
    if arr.shape[: len(expected_prefix)] != expected_prefix:
        raise ContractMismatchError(
            f"Shape mismatch: got {arr.shape}, expected prefix {expected_prefix} for '{name}'"
        )
    if not np.isfinite(arr).all():
        raise ContractMismatchError(f"Sufficient statistics '{name}' contains NaN or Inf values")
    if (arr < 0).any():
        raise ContractMismatchError(f"Sufficient statistics '{name}' cannot contain negative counts")
    if np.issubdtype(arr.dtype, np.floating) and not np.all(arr == np.floor(arr)):
        raise ContractMismatchError(
            f"Sufficient statistics '{name}' cannot contain fractional counts; counts must be exact integers"
        )
    return arr.astype(np.int64)


def _check_stats_scores_consistency(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    scores: np.ndarray,
    contract: MetricContract,
    score_name: str,
) -> None:
    both_empty_val = empty_class_score(contract.empty_policy)
    class_scores = []
    for c in contract.classes:
        tpc = tp[..., c].astype(np.float64)
        fpc = fp[..., c].astype(np.float64)
        fnc = fn[..., c].astype(np.float64)
        denom = 2.0 * tpc + fpc + fnc
        safe_denom = np.where(denom == 0, 1.0, denom)
        cs = np.where(denom == 0, both_empty_val, np.where(tpc == 0, 0.0, 2.0 * tpc / safe_denom))
        class_scores.append(cs)
    stacked = np.stack(class_scores, axis=-1)
    all_nan = np.all(np.isnan(stacked), axis=-1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        computed_macro = np.nanmean(stacked, axis=-1)
    computed_macro[all_nan] = np.nan

    scores_nan = np.isnan(scores)
    computed_nan = np.isnan(computed_macro)
    nan_mismatch = scores_nan ^ computed_nan
    if nan_mismatch.any():
        idx = tuple(np.argwhere(nan_mismatch)[0])
        raise ContractMismatchError(
            f"Finite/NaN mismatch between {score_name}{idx} ({scores[idx]}) and "
            f"sufficient statistics ({computed_macro[idx]}) under contract '{contract.name}'"
        )
    both_finite = ~scores_nan & ~computed_nan
    if both_finite.any():
        diff = np.abs(scores[both_finite] - computed_macro[both_finite])
        if (diff > 1e-4).any():
            worst_sub_idx = np.argmax(diff)
            full_idx = tuple(np.argwhere(both_finite)[worst_sub_idx])
            raise ContractMismatchError(
                f"Inconsistency detected between {score_name}{full_idx} ({scores[full_idx]:.6f}) and "
                f"Dice computed from sufficient statistics ({computed_macro[full_idx]:.6f}) "
                f"under contract '{contract.name}'"
            )


def _validate_sufficient_statistics_bundle(
    transitions: Mapping[str, Any],
    *,
    N: int,
    T: int,
    contract: MetricContract,
    initial_dice: np.ndarray,
    q_candidate: np.ndarray | None,
    require_volume: bool = False,
) -> dict[str, np.ndarray]:
    has_init_stats = any(k in transitions and transitions[k] is not None for k in ("tp_initial", "fp_initial", "fn_initial"))
    has_cand_stats = any(k in transitions and transitions[k] is not None for k in ("tp_candidate", "fp_candidate", "fn_candidate"))

    if require_volume and not has_init_stats:
        raise ContractMismatchError(
            "Volume scoring was explicitly requested (require_volume=True) but sufficient statistics "
            "are missing from transitions cache."
        )

    if not has_init_stats and not has_cand_stats:
        return {}

    # Step 1: individual array validation
    init_arrays: dict[str, np.ndarray] = {}
    for k in ("tp_initial", "fp_initial", "fn_initial"):
        if k in transitions and transitions[k] is not None:
            init_arrays[k] = _validate_stats_array(transitions[k], k, expected_ndim=2, expected_prefix=(N,))

    cand_arrays: dict[str, np.ndarray] = {}
    for k in ("tp_candidate", "fp_candidate", "fn_candidate"):
        if k in transitions and transitions[k] is not None:
            cand_arrays[k] = _validate_stats_array(transitions[k], k, expected_ndim=3, expected_prefix=(N, T))

    # Step 2: bundle completeness
    missing_init = [k for k in ("tp_initial", "fp_initial", "fn_initial") if k not in init_arrays]
    if missing_init:
        raise ContractMismatchError(
            f"Incomplete sufficient statistics bundle: initial stats missing {', '.join(missing_init)}"
        )

    if T > 0:
        missing_cand = [k for k in ("tp_candidate", "fp_candidate", "fn_candidate") if k not in cand_arrays]
        if missing_cand:
            raise ContractMismatchError(
                f"Incomplete sufficient statistics bundle: candidate stats missing {', '.join(missing_cand)}"
            )
    else:  # T == 0
        if cand_arrays and len(cand_arrays) != 3:
            missing_cand = [k for k in ("tp_candidate", "fp_candidate", "fn_candidate") if k not in cand_arrays]
            raise ContractMismatchError(
                f"Incomplete sufficient statistics bundle: candidate stats missing {', '.join(missing_cand)}"
            )

    # Step 3: check class dimensions
    tp_init = init_arrays["tp_initial"]
    fp_init = init_arrays["fp_initial"]
    fn_init = init_arrays["fn_initial"]
    if tp_init.shape != fp_init.shape or tp_init.shape != fn_init.shape:
        raise ContractMismatchError(
            f"Shape mismatch among initial statistics: tp {tp_init.shape}, fp {fp_init.shape}, fn {fn_init.shape}"
        )
    C_init = tp_init.shape[1]
    max_req_class = max(contract.classes)
    if C_init <= max_req_class:
        raise ContractMismatchError(
            f"Sufficient statistics class dimension C={C_init} is too small for contract classes {contract.classes}; "
            f"requires at least {max_req_class + 1} classes."
        )

    if cand_arrays:
        tp_cand = cand_arrays["tp_candidate"]
        fp_cand = cand_arrays["fp_candidate"]
        fn_cand = cand_arrays["fn_candidate"]
        if tp_cand.shape != fp_cand.shape or tp_cand.shape != fn_cand.shape:
            raise ContractMismatchError(
                f"Shape mismatch among candidate statistics: tp {tp_cand.shape}, fp {fp_cand.shape}, fn {fn_cand.shape}"
            )
        if tp_cand.shape[2] != C_init:
            raise ContractMismatchError(
                f"Class dimension mismatch: initial has C={C_init}, candidate has C={tp_cand.shape[2]}"
            )

    # Step 4: Q / stat consistency check
    _check_stats_scores_consistency(tp_init, fp_init, fn_init, initial_dice, contract, "initial_dice")
    if cand_arrays and q_candidate is not None and T > 0:
        _check_stats_scores_consistency(
            cand_arrays["tp_candidate"],
            cand_arrays["fp_candidate"],
            cand_arrays["fn_candidate"],
            q_candidate,
            contract,
            "q_candidate",
        )

    out = {**init_arrays, **cand_arrays}
    return out


def _evaluate_threshold_core(
    tau_accept: float,
    *,
    initial: np.ndarray,
    quality: np.ndarray,
    actual: np.ndarray,
    valid: np.ndarray,
    margin: float,
    cand_scores: np.ndarray | None = None,
    resolved_contract: MetricContract | None = None,
    transitions: Mapping[str, Any] | None = None,
    require_volume: bool = False,
) -> dict[str, Any]:
    """Private threshold evaluation core executed only after boundary validation."""
    active = np.ones(quality.shape[0], dtype=bool)
    final = initial.copy()
    final_state_idx = np.zeros(quality.shape[0], dtype=np.int64)
    attempted = np.zeros_like(initial, dtype=np.int64)
    accepted_count = np.zeros_like(initial, dtype=np.int64)
    accepted_total = 0
    rejected_total = 0
    accepted_non_neutral_total = 0
    rejected_non_neutral_total = 0
    harmful_total = 0
    beneficial_rejection_total = 0
    neutral_accepted_total = 0
    undefined_accepted_total = 0
    undefined_rejected_total = 0
    undefined_transition_total = 0

    for turn in range(quality.shape[1]):
        eligible = active & valid[:, turn]
        attempted += eligible.astype(np.int64)
        accepted = eligible & (quality[:, turn] > float(tau_accept))
        rejected = eligible & ~accepted
        accepted_count += accepted.astype(np.int64)
        accepted_total += int(accepted.sum())
        rejected_total += int(rejected.sum())

        delta_turn = actual[:, turn]
        is_undefined_delta = ~np.isfinite(delta_turn)
        is_harmful = harmful_mask(delta_turn, margin) & ~is_undefined_delta
        is_beneficial = beneficial_mask(delta_turn, margin) & ~is_undefined_delta
        is_neutral = neutral_mask(delta_turn, margin) & ~is_undefined_delta

        accepted_non_neutral_total += int((accepted & ~is_neutral & ~is_undefined_delta).sum())
        rejected_non_neutral_total += int((rejected & ~is_neutral & ~is_undefined_delta).sum())
        harmful_total += int((accepted & is_harmful).sum())
        beneficial_rejection_total += int((rejected & is_beneficial).sum())
        neutral_accepted_total += int((accepted & is_neutral).sum())
        undefined_accepted_total += int((accepted & is_undefined_delta).sum())
        undefined_rejected_total += int((rejected & is_undefined_delta).sum())
        undefined_transition_total += int((eligible & is_undefined_delta).sum())

        if cand_scores is not None:
            final = np.where(accepted, cand_scores[:, turn], final)
            final_state_idx = np.where(accepted, turn + 1, final_state_idx)
        else:
            final += np.where(accepted, np.nan_to_num(delta_turn, nan=0.0), 0.0)
            final_state_idx = np.where(accepted, turn + 1, final_state_idx)

        # A rejected transition halts only that sample.
        active = accepted

    attempted_total = max(int(attempted.sum()), 1)
    finite_final = np.isfinite(final)
    finite_initial = np.isfinite(initial)
    final_macro_dice = float(np.nanmean(final)) if finite_final.any() else float("nan")
    initial_macro_dice = float(np.nanmean(initial)) if finite_initial.any() else float("nan")
    both_finite = finite_initial & finite_final
    if both_finite.any():
        net_dice_gain = float(np.mean(final[both_finite] - initial[both_finite]))
    elif not finite_initial.any() and not finite_final.any():
        net_dice_gain = 0.0
    else:
        net_dice_gain = float("nan")

    result: dict[str, Any] = {
        "tau_accept": float(tau_accept),
        "neutral_margin": float(margin),
        "final_macro_dice": final_macro_dice,
        "initial_macro_dice": initial_macro_dice,
        "net_dice_gain": net_dice_gain,
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
        "undefined_accepted_total": int(undefined_accepted_total),
        "undefined_rejected_total": int(undefined_rejected_total),
        "undefined_transition_total": int(undefined_transition_total),
        "undefined_initial_count": int((~finite_initial).sum()),
        "undefined_final_count": int((~finite_final).sum()),
        "blank_to_hallucination_count": int((~finite_initial & finite_final).sum()),
        "hallucination_to_blank_count": int((finite_initial & ~finite_final).sum()),
        "finite_slice_count": int(finite_final.sum()),
        "sample_count": int(initial.size),
        "final_dice": final.copy(),
    }
    if resolved_contract is not None:
        result["metric_contract"] = resolved_contract.name
        result["metric_space"] = resolved_contract.metric_space

    # Check for volume-level sufficient statistics aggregation
    if transitions is not None and "tp_initial" in transitions and transitions["tp_initial"] is not None:
        tp_init = np.asarray(transitions["tp_initial"])
        fp_init = np.asarray(transitions["fp_initial"])
        fn_init = np.asarray(transitions["fn_initial"])
        contract_for_vol = resolved_contract or FOREGROUND_DICE_EXCLUDE_V1_CONTRACT

        final_tp = tp_init.copy()
        final_fp = fp_init.copy()
        final_fn = fn_init.copy()
        if "tp_candidate" in transitions and transitions["tp_candidate"] is not None and quality.shape[1] > 0:
            tp_cand = np.asarray(transitions["tp_candidate"])
            fp_cand = np.asarray(transitions["fp_candidate"])
            fn_cand = np.asarray(transitions["fn_candidate"])
            for i in range(initial.shape[0]):
                if accepted_count[i] > 0:
                    t_last = accepted_count[i] - 1
                    final_tp[i] = tp_cand[i, t_last]
                    final_fp[i] = fp_cand[i, t_last]
                    final_fn[i] = fn_cand[i, t_last]

        case_ids_raw = None
        for k in ("case_ids", "volume_ids", "subject_ids"):
            if k in transitions and transitions[k] is not None:
                case_ids_raw = transitions[k]
                break

        if case_ids_raw is not None:
            if isinstance(case_ids_raw, np.ndarray):
                case_ids = [str(c).strip() for c in case_ids_raw.tolist()]
            elif isinstance(case_ids_raw, (list, tuple)):
                case_ids = [str(c).strip() for c in case_ids_raw]
            else:
                raise ContractMismatchError(
                    f"Unsupported case_ids type: {type(case_ids_raw).__name__}"
                )
            if len(case_ids) != initial.shape[0]:
                raise ValueError(
                    f"case_ids length ({len(case_ids)}) does not match samples ({initial.shape[0]})"
                )
            if any(not c or c.lower() in ("none", "nan", "null") for c in case_ids):
                raise ContractMismatchError(
                    "case_ids contains invalid, empty, or None identifiers"
                )
            unique_cases = sorted(set(case_ids))
            case_macros: list[float] = []
            case_per_classes: list[dict[int, float]] = []
            for cid in unique_cases:
                c_mask = np.array([c == cid for c in case_ids], dtype=bool)
                c_tp = {cls: int(final_tp[c_mask, cls].sum()) for cls in contract_for_vol.classes}
                c_fp = {cls: int(final_fp[c_mask, cls].sum()) for cls in contract_for_vol.classes}
                c_fn = {cls: int(final_fn[c_mask, cls].sum()) for cls in contract_for_vol.classes}
                c_stats = SufficientStatistics(tp=c_tp, fp=c_fp, fn=c_fn)
                c_per_class, c_macro = compute_dice_from_stats(c_stats, contract_for_vol)
                case_macros.append(c_macro)
                case_per_classes.append(c_per_class)

            finite_case_macros = [m for m in case_macros if np.isfinite(m)]
            result["volume_macro_dice"] = float(np.mean(finite_case_macros)) if finite_case_macros else float("nan")
            result["volume_per_class_dice"] = {
                cls: float(np.nanmean([p[cls] for p in case_per_classes]))
                if any(np.isfinite(p[cls]) for p in case_per_classes) else float("nan")
                for cls in contract_for_vol.classes
            }
            result["volume_case_count"] = int(len(unique_cases))
            result["volume_metrics_available"] = True
            result["volume_metric_space"] = METRIC_SPACE_VOLUME_RESIZED
            result["volume_metric_contract"] = (
                FOREGROUND_DICE_VOLUME_RESIZED_V1_CONTRACT.name
                if contract_for_vol.empty_policy == "exclude" and contract_for_vol.classes == (1, 2, 3)
                else f"{contract_for_vol.name}_volume_resized"
            )
        else:
            if require_volume:
                raise ContractMismatchError(
                    "Volume scoring was explicitly requested (require_volume=True) but case_ids are missing; "
                    "cohort pooling across unidentified slices is forbidden."
                )
            result["volume_macro_dice"] = None
            result["volume_per_class_dice"] = None
            result["volume_case_count"] = None
            result["volume_metrics_available"] = False
            result["volume_unavailable_reason"] = (
                "Missing case_ids; cohort pooling across unidentified slices is forbidden"
            )
    elif require_volume:
        raise ContractMismatchError(
            "Volume scoring was explicitly requested (require_volume=True) but sufficient statistics "
            "are missing from transitions cache."
        )
    else:
        result["volume_metrics_available"] = False

    return result


def evaluate_threshold(
    tau_accept: float,
    initial_dice: Any = None,
    delta_q: Any = None,
    actual_delta_dice: Any = None,
    active_mask: Any | None = None,
    *,
    neutral_margin: float | None = None,
    q_previous: Any | None = None,
    q_candidate: Any | None = None,
    stats_initial: Any | None = None,
    stats_candidate: Any | None = None,
    metric_contract: MetricContract | str | None = None,
    transitions: Mapping[str, Any] | None = None,
    cache_schema_version: Any = None,
    require_volume: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Simulate per-sample threshold halting on cached validation transitions.

    Public entry point. Revalidates transitions strictly at the boundary.
    No serialized or caller-supplied dictionary key may skip validation.
    """
    if transitions is not None:
        norm = validate_and_normalize_transition_cache(
            transitions,
            expected_contract=metric_contract,
            neutral_margin=neutral_margin,
            strict_contract=True,
            require_volume=require_volume,
        )
        return _evaluate_threshold_core(
            float(tau_accept),
            initial=norm["initial_dice"],
            quality=norm["delta_q"],
            actual=norm["actual_delta_dice"],
            valid=norm["active_mask"],
            margin=norm["neutral_margin"],
            cand_scores=norm.get("q_candidate"),
            resolved_contract=norm.get("metric_contract"),
            transitions=norm,
            require_volume=require_volume,
        )

    if cache_schema_version is not None:
        if isinstance(cache_schema_version, bool) or not isinstance(cache_schema_version, (int, np.integer)):
            raise ContractMismatchError(
                f"Invalid cache_schema_version {cache_schema_version!r}: version must be an integer, got {type(cache_schema_version).__name__}"
            )
        if int(cache_schema_version) not in SUPPORTED_CACHE_SCHEMA_VERSIONS:
            raise ContractMismatchError(
                f"Unsupported cache_schema_version {cache_schema_version!r}; "
                f"supported versions: {SUPPORTED_CACHE_SCHEMA_VERSIONS}"
            )

    resolved_contract = resolve_metric_contract(metric_contract) if metric_contract else None
    margin = resolve_neutral_margin(neutral_margin)
    initial = _array(initial_dice, name="initial_dice", allow_nan=True).reshape(-1).astype(np.float64)
    quality, actual, valid = _transition_arrays(delta_q, actual_delta_dice, active_mask)
    if initial.shape[0] != quality.shape[0]:
        raise ValueError(f"initial_dice has {initial.shape[0]} samples, transitions have {quality.shape[0]}")

    cand_scores: np.ndarray | None = None
    if q_candidate is not None:
        cand_scores = _array(q_candidate, name="q_candidate", allow_nan=True)
        if cand_scores.ndim == 3 and cand_scores.shape[-1] == 1:
            cand_scores = cand_scores[..., 0]
        if cand_scores.ndim == 1 and quality.shape[0] == 1:
            cand_scores = cand_scores[None, :]
        elif cand_scores.ndim == 1:
            cand_scores = cand_scores[:, None]
        if cand_scores.shape != quality.shape:
            raise ValueError(
                f"q_candidate shape {cand_scores.shape} does not match transitions {quality.shape}"
            )

    if resolved_contract is not None:
        if cand_scores is None:
            raise ContractMismatchError(
                "Contract-governed evaluate_threshold requires explicit q_candidate. "
                "Mixed-contract delta addition is strictly forbidden."
            )
        if q_previous is None:
            raise ContractMismatchError(
                "Contract-governed evaluate_threshold requires explicit q_previous."
            )

    if q_previous is not None and cand_scores is not None:
        prev_scores = _array(q_previous, name="q_previous", allow_nan=True)
        if prev_scores.ndim == 3 and prev_scores.shape[-1] == 1:
            prev_scores = prev_scores[..., 0]
        if prev_scores.ndim == 1 and quality.shape[0] == 1:
            prev_scores = prev_scores[None, :]
        elif prev_scores.ndim == 1:
            prev_scores = prev_scores[:, None]
        if prev_scores.shape != quality.shape:
            raise ContractMismatchError(
                f"Shape mismatch: q_previous shape {prev_scores.shape} does not match transitions {quality.shape}"
            )
        diff = cand_scores - prev_scores
        prev_fin = np.isfinite(prev_scores)
        cand_fin = np.isfinite(cand_scores)
        act_fin = np.isfinite(actual)
        both_fin = prev_fin & cand_fin & valid
        if both_fin.any():
            if not act_fin[both_fin].all():
                raise ContractMismatchError(
                    "Fabricated NaN in actual_delta_dice: actual_delta_dice is NaN where both q_previous and q_candidate are finite."
                )
            if not np.allclose(actual[both_fin], diff[both_fin], atol=1e-4):
                raise ContractMismatchError(
                    "Inconsistency detected between actual_delta_dice and (q_candidate - q_previous). "
                    "State scores and deltas must originate from the exact same metric contract."
                )
        either_nan = (~both_fin) & valid
        if either_nan.any():
            if act_fin[either_nan].any():
                raise ContractMismatchError(
                    "Inconsistency in actual_delta_dice: actual_delta_dice is finite where q_previous or q_candidate is NaN."
                )

    extra_transitions: dict[str, Any] = {}
    if stats_initial is not None:
        extra_transitions["stats_initial"] = stats_initial
    if stats_candidate is not None:
        extra_transitions["stats_candidate"] = stats_candidate
    for k in ("tp_initial", "fp_initial", "fn_initial", "tp_candidate", "fp_candidate", "fn_candidate", "case_ids"):
        if k in kwargs and kwargs[k] is not None:
            extra_transitions[k] = kwargs[k]

    if extra_transitions or require_volume:
        contract_for_check = resolved_contract or FOREGROUND_DICE_EXCLUDE_V1_CONTRACT
        validated_extra = _validate_sufficient_statistics_bundle(
            extra_transitions,
            N=initial.shape[0],
            T=quality.shape[1],
            contract=contract_for_check,
            initial_dice=initial,
            q_candidate=cand_scores,
            require_volume=require_volume,
        )
        extra_transitions.update(validated_extra)

    return _evaluate_threshold_core(
        float(tau_accept),
        initial=initial,
        quality=quality,
        actual=actual,
        valid=valid,
        margin=margin,
        cand_scores=cand_scores,
        resolved_contract=resolved_contract,
        transitions=extra_transitions if extra_transitions else None,
        require_volume=require_volume,
    )


def validate_and_normalize_transition_cache(
    transitions: Mapping[str, Any],
    *,
    expected_contract: MetricContract | str | None = None,
    neutral_margin: float | None = None,
    strict_contract: bool = True,
    require_volume: bool = False,
) -> dict[str, Any]:
    """Validate and normalize a transition cache under strict metric contracts.

    Enforces:
    1. Contract metadata presence and validation (rejecting unversioned or forged contracts).
    2. Exact alignment of neutral_margin and metric_space with contract.
    3. Mandatory presence of initial_dice, delta_q, actual_delta_dice, q_previous, and q_candidate.
       (Presence of stats_candidate or tp_candidate alone is NEVER enough to bypass state scores).
    4. Exact (N,) and (N, T) shape alignment across all arrays (supporting T=0).
    5. Valid score bounds (finite Dice in [-1e-6, 1.0 + 1e-6], finite delta_q).
    6. Mask and delta parity consistency:
       - If both q_previous and q_candidate are finite, actual_delta_dice MUST be finite
         and equal (q_candidate - q_previous) within 1e-4. (Fabricated NaNs are rejected!).
       - If either is NaN, actual_delta_dice MUST be NaN.
       - At turn 0 (T > 0), q_previous[:, 0] MUST match initial_dice (both NaN or both finite within 1e-4).
       - For T > 1, state continuity: q_previous[:, t] MUST match q_candidate[:, t-1].
    7. Complete validation of sufficient statistics if present (rejecting bogus stats/empty lists/wrong shapes).
    """
    if not isinstance(transitions, Mapping):
        raise TypeError(f"Transitions must be a Mapping, got {type(transitions).__name__}")

    # 0. Cache schema version verification
    if "cache_schema_version" not in transitions:
        if strict_contract:
            raise ContractMismatchError(
                "Cached transitions missing required 'cache_schema_version'. "
                f"Expected version {CACHE_SCHEMA_VERSION}."
            )
    else:
        ver = transitions["cache_schema_version"]
        if isinstance(ver, bool) or not isinstance(ver, (int, np.integer)):
            raise ContractMismatchError(
                f"Invalid cache_schema_version {ver!r}: version must be an integer, got {type(ver).__name__}"
            )
        if int(ver) not in SUPPORTED_CACHE_SCHEMA_VERSIONS:
            raise ContractMismatchError(
                f"Unsupported cache_schema_version {ver!r}; "
                f"supported versions: {SUPPORTED_CACHE_SCHEMA_VERSIONS}"
            )

    # 1. Contract metadata verification
    cached_contract_raw = transitions.get("metric_contract")
    if cached_contract_raw is None:
        if strict_contract:
            target_contract_name = (
                expected_contract.name if isinstance(expected_contract, MetricContract)
                else (str(expected_contract) if expected_contract is not None else FOREGROUND_DICE_EXCLUDE_V1_CONTRACT.name)
            )
            raise ContractMismatchError(
                "Cached transitions do not declare a metric_contract. "
                "Unversioned transition caches are rejected by default. "
                f"Expected contract: {target_contract_name}"
            )
        contract = FOREGROUND_DICE_EXCLUDE_V1_CONTRACT
    else:
        contract = validate_contract(cached_contract_raw)

    if expected_contract is not None:
        target_contract = resolve_metric_contract(expected_contract)
        validate_contract_compatibility(target_contract, contract)

    # Neutral margin consistency: if neutral_margin is explicitly supplied, must match contract
    if neutral_margin is not None:
        resolved_nm = resolve_neutral_margin(neutral_margin)
        if abs(resolved_nm - contract.neutral_margin) > 1e-7:
            raise ContractMismatchError(
                f"Sweep neutral_margin ({resolved_nm}) conflicts with "
                f"declared contract '{contract.name}' neutral_margin ({contract.neutral_margin})"
            )
    effective_neutral_margin = contract.neutral_margin

    # Cache metadata metric_space consistency
    cached_space = transitions.get("metric_space")
    if cached_space is not None and str(cached_space) != contract.metric_space:
        raise ContractMismatchError(
            f"Cache metric_space '{cached_space}' conflicts with "
            f"declared contract '{contract.name}' space '{contract.metric_space}'"
        )

    if contract.metric_space != METRIC_SPACE_SLICE_PROXY:
        raise ContractMismatchError(
            f"Transition cache represents per-slice transitions and requires metric_space='{METRIC_SPACE_SLICE_PROXY}', "
            f"got '{contract.metric_space}' from contract '{contract.name}'. "
            "Per-slice Q values must never be labeled volume scores."
        )

    # 2. Strict presence of required keys
    required_keys = ("initial_dice", "delta_q", "actual_delta_dice")
    missing = sorted(set(required_keys) - set(transitions))
    if missing:
        raise ValueError(f"Cached transitions are missing: {', '.join(missing)}")

    # Check for bogus stats container before checking state scores
    for k in ("stats_candidate", "stats_initial"):
        if k in transitions:
            val = transitions[k]
            if val is None or (isinstance(val, (list, tuple, dict)) and len(val) == 0):
                raise ContractMismatchError(
                    f"Bogus or empty sufficient statistics container '{k}' is rejected."
                )

    if strict_contract:
        missing_state_scores = [k for k in ("q_previous", "q_candidate") if k not in transitions or transitions[k] is None]
        if missing_state_scores:
            raise ContractMismatchError(
                f"Strict cache requires both 'q_previous' and 'q_candidate'. "
                f"Missing: {', '.join(missing_state_scores)}. "
                "Presence of sufficient statistics keys alone is never sufficient to bypass state scores."
            )

    # Convert arrays
    initial = _array(transitions["initial_dice"], name="initial_dice", allow_nan=True).reshape(-1).astype(np.float64)
    N = int(initial.shape[0])
    if N == 0:
        raise ValueError("Cache cannot be empty (0 samples)")

    raw_dq = transitions["delta_q"]
    raw_act = transitions["actual_delta_dice"]
    delta_q = _array(raw_dq, name="delta_q", allow_nan=False)
    actual_delta = _array(raw_act, name="actual_delta_dice", allow_nan=True)

    if delta_q.ndim == 3 and delta_q.shape[-1] == 1:
        delta_q = delta_q[..., 0]
    if delta_q.ndim == 1 and N == 1:
        delta_q = delta_q[None, :]
    elif delta_q.ndim == 1:
        delta_q = delta_q[:, None]
    if delta_q.ndim != 2 or delta_q.shape[0] != N:
        raise ContractMismatchError(
            f"Shape mismatch: delta_q must have shape (N={N}, T), got {delta_q.shape}"
        )
    T = int(delta_q.shape[1])

    if actual_delta.ndim == 3 and actual_delta.shape[-1] == 1:
        actual_delta = actual_delta[..., 0]
    if actual_delta.ndim == 1 and N == 1:
        actual_delta = actual_delta[None, :]
    elif actual_delta.ndim == 1:
        actual_delta = actual_delta[:, None]
    if actual_delta.shape != (N, T):
        raise ContractMismatchError(
            f"Shape mismatch: actual_delta_dice has shape {actual_delta.shape}, expected (N={N}, T={T})"
        )

    # Score bounds for initial
    fin_init = np.isfinite(initial)
    if fin_init.any():
        init_vals = initial[fin_init]
        if (init_vals < -1e-6).any() or (init_vals > 1.0 + 1e-6).any():
            raise ContractMismatchError(
                f"initial_dice values must be in [0.0, 1.0] or NaN, got range [{init_vals.min()}, {init_vals.max()}]"
            )

    if not np.isfinite(delta_q).all():
        raise ContractMismatchError("delta_q values must all be finite real numbers")

    raw_active = transitions.get("active_mask")
    if raw_active is not None:
        active_mask = np.asarray(raw_active, dtype=bool)
        if active_mask.ndim == 3 and active_mask.shape[-1] == 1:
            active_mask = active_mask[..., 0]
        if active_mask.ndim == 1 and N == 1:
            active_mask = active_mask[None, :]
        elif active_mask.ndim == 1:
            active_mask = active_mask[:, None]
        if active_mask.shape != (N, T):
            raise ContractMismatchError(
                f"Shape mismatch: active_mask has shape {active_mask.shape}, expected (N={N}, T={T})"
            )
    else:
        active_mask = np.ones((N, T), dtype=bool)

    q_prev: np.ndarray | None = None
    q_cand: np.ndarray | None = None

    if "q_previous" in transitions and transitions["q_previous"] is not None:
        raw_prev = transitions["q_previous"]
        q_prev = _array(raw_prev, name="q_previous", allow_nan=True)
        if q_prev.ndim == 3 and q_prev.shape[-1] == 1:
            q_prev = q_prev[..., 0]
        if q_prev.ndim == 1 and N == 1:
            q_prev = q_prev[None, :]
        elif q_prev.ndim == 1:
            q_prev = q_prev[:, None]
        if q_prev.shape != (N, T):
            raise ContractMismatchError(
                f"Shape mismatch: q_previous has shape {q_prev.shape}, expected (N={N}, T={T})"
            )
        fin_prev = np.isfinite(q_prev)
        if fin_prev.any():
            prev_vals = q_prev[fin_prev]
            if (prev_vals < -1e-6).any() or (prev_vals > 1.0 + 1e-6).any():
                raise ContractMismatchError(
                    f"q_previous values must be in [0.0, 1.0] or NaN, got range [{prev_vals.min()}, {prev_vals.max()}]"
                )

    if "q_candidate" in transitions and transitions["q_candidate"] is not None:
        raw_cand = transitions["q_candidate"]
        q_cand = _array(raw_cand, name="q_candidate", allow_nan=True)
        if q_cand.ndim == 3 and q_cand.shape[-1] == 1:
            q_cand = q_cand[..., 0]
        if q_cand.ndim == 1 and N == 1:
            q_cand = q_cand[None, :]
        elif q_cand.ndim == 1:
            q_cand = q_cand[:, None]
        if q_cand.shape != (N, T):
            raise ContractMismatchError(
                f"Shape mismatch: q_candidate has shape {q_cand.shape}, expected (N={N}, T={T})"
            )
        fin_cand = np.isfinite(q_cand)
        if fin_cand.any():
            cand_vals = q_cand[fin_cand]
            if (cand_vals < -1e-6).any() or (cand_vals > 1.0 + 1e-6).any():
                raise ContractMismatchError(
                    f"q_candidate values must be in [0.0, 1.0] or NaN, got range [{cand_vals.min()}, {cand_vals.max()}]"
                )

    if q_prev is not None and q_cand is not None:
        if T > 0:
            # Check 1: Turn 0 previous state matches initial_dice
            prev_0 = q_prev[:, 0]
            init_f = np.isfinite(initial)
            prev0_f = np.isfinite(prev_0)
            if not np.array_equal(init_f, prev0_f):
                raise ContractMismatchError(
                    "Finite/NaN mask mismatch between initial_dice and turn 0 q_previous"
                )
            both_0_f = init_f & prev0_f
            if both_0_f.any() and not np.allclose(initial[both_0_f], prev_0[both_0_f], atol=1e-4):
                raise ContractMismatchError(
                    "Turn 0 q_previous values do not match initial_dice"
                )

            # Check 2: Delta parity and fabricated NaN rejection
            prev_f = np.isfinite(q_prev)
            cand_f = np.isfinite(q_cand)
            act_f = np.isfinite(actual_delta)

            both_finite = prev_f & cand_f
            if both_finite.any():
                if not act_f[both_finite].all():
                    raise ContractMismatchError(
                        "Fabricated NaN in actual_delta_dice: actual_delta_dice is NaN where both q_previous and q_candidate are finite."
                    )
                expected_diff = q_cand[both_finite] - q_prev[both_finite]
                actual_vals = actual_delta[both_finite]
                if not np.allclose(actual_vals, expected_diff, atol=1e-4):
                    raise ContractMismatchError(
                        "Inconsistency detected between actual_delta_dice and (q_candidate - q_previous). "
                        "State scores and deltas must originate from the exact same metric contract."
                    )

            either_nan = ~both_finite
            if either_nan.any():
                if act_f[either_nan].any():
                    raise ContractMismatchError(
                        "Inconsistency in actual_delta_dice: actual_delta_dice is finite where q_previous or q_candidate is NaN."
                    )

            # Check 3: State continuity across turns (T > 1)
            for t in range(1, T):
                prev_t = q_prev[:, t]
                cand_prev = q_cand[:, t - 1]
                prev_t_f = np.isfinite(prev_t)
                cand_prev_f = np.isfinite(cand_prev)
                if not np.array_equal(prev_t_f, cand_prev_f):
                    raise ContractMismatchError(
                        f"State continuity broken: finite/NaN mask mismatch at turn {t} vs turn {t-1} candidate"
                    )
                both_t_f = prev_t_f & cand_prev_f
                if both_t_f.any() and not np.allclose(prev_t[both_t_f], cand_prev[both_t_f], atol=1e-4):
                    raise ContractMismatchError(
                        f"State continuity broken at turn {t}: q_previous[:, {t}] does not match q_candidate[:, {t-1}]"
                    )

    # 3. Sufficient statistics validation
    validated_stats = _validate_sufficient_statistics_bundle(
        transitions,
        N=N,
        T=T,
        contract=contract,
        initial_dice=initial,
        q_candidate=q_cand,
        require_volume=require_volume,
    )

    normalized_case_ids: list[str] | None = None
    case_ids_source = None
    for k in ("case_ids", "volume_ids", "subject_ids"):
        if k in transitions and transitions[k] is not None:
            case_ids_source = transitions[k]
            break

    if case_ids_source is not None:
        cids = case_ids_source
        if isinstance(cids, np.ndarray):
            cids_list = [str(c).strip() for c in cids.tolist()]
        elif isinstance(cids, (list, tuple)):
            cids_list = [str(c).strip() for c in cids]
        else:
            raise ContractMismatchError(
                f"case_ids must be a list, tuple, or 1D ndarray, got {type(cids).__name__}"
            )
        if len(cids_list) != N:
            raise ContractMismatchError(
                f"case_ids length ({len(cids_list)}) must match sample count N={N}"
            )
        if any(not c or c.lower() in ("none", "nan", "null") for c in cids_list):
            raise ContractMismatchError(
                "case_ids contains invalid, empty, or None identifiers"
            )
        normalized_case_ids = cids_list
    elif require_volume:
        raise ContractMismatchError(
            "Volume scoring was explicitly requested (require_volume=True) but case_ids are missing; "
            "cohort pooling across unidentified slices is forbidden."
        )

    normalized: dict[str, Any] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "initial_dice": initial,
        "delta_q": delta_q,
        "actual_delta_dice": actual_delta,
        "active_mask": active_mask,
        "metric_contract": contract,
        "metric_space": contract.metric_space,
        "neutral_margin": effective_neutral_margin,
    }
    if q_prev is not None:
        normalized["q_previous"] = q_prev
    if q_cand is not None:
        normalized["q_candidate"] = q_cand

    for k, v in validated_stats.items():
        normalized[k] = v

    if normalized_case_ids is not None:
        normalized["case_ids"] = normalized_case_ids

    for k in (
        "legacy_actual_delta_dice",
        "empty_class_policy", "excluded_empty_slice_count",
    ):
        if k in transitions and transitions[k] is not None:
            normalized[k] = transitions[k]

    return normalized


def sweep_thresholds(
    transitions: Mapping[str, Any],
    thresholds: Iterable[float],
    *,
    max_harmful_acceptance_rate: float | None = None,
    neutral_margin: float | None = None,
    expected_contract: MetricContract | str | None = None,
    strict_contract: bool = True,
    require_volume: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate a threshold grid from a cached validation-transition mapping."""

    norm = validate_and_normalize_transition_cache(
        transitions,
        expected_contract=expected_contract,
        neutral_margin=neutral_margin,
        strict_contract=strict_contract,
        require_volume=require_volume,
    )
    margin = norm["neutral_margin"]

    rows = [
        _evaluate_threshold_core(
            float(tau),
            initial=norm["initial_dice"],
            quality=norm["delta_q"],
            actual=norm["actual_delta_dice"],
            valid=norm["active_mask"],
            margin=margin,
            cand_scores=norm.get("q_candidate"),
            resolved_contract=norm.get("metric_contract"),
            transitions=norm,
            require_volume=require_volume,
        )
        for tau in thresholds
    ]
    if max_harmful_acceptance_rate is not None:
        allowed = [
            row
            for row in rows
            if row["harmful_acceptance_rate"] <= float(max_harmful_acceptance_rate)
        ]
        if not allowed:
            min_harmful = min(r["harmful_acceptance_rate"] for r in rows) if rows else float("nan")
            raise NoFeasibleThresholdError(
                f"No threshold in grid satisfied constraint max_harmful_acceptance_rate <= {max_harmful_acceptance_rate:g}. "
                f"Minimum harmful_acceptance_rate was {min_harmful:g}."
            )
        return allowed
    return rows


#: Human-readable statement of what :func:`select_threshold` maximises.
SELECTION_OBJECTIVE = (
    "argmax over the threshold grid of the lexicographic key "
    "(final_macro_dice, -harmful_acceptance_rate, -abs(tau_accept)); "
    "harmful_acceptance_rate counts only strictly harmful accepts over all accepts; "
    "selection and reporting share one split, so the result is diagnostic only"
)


def select_threshold(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the best validation row, with lower harmful acceptance as tie-break.

    The lexicographic key is unchanged. The returned dict now additionally
    carries ``objective``, ``n_rows``, and ``neutral_margin``.
    If cohort has no usable outcome (all final_macro_dice are NaN), raises NoFeasibleThresholdError.
    """

    candidates = [dict(row) for row in rows]
    if not candidates:
        raise ValueError("Cannot select a threshold from zero calibration rows")

    finite_candidates = [
        row
        for row in candidates
        if np.isfinite(float(row.get("final_macro_dice", float("nan"))))
    ]
    if not finite_candidates:
        raise NoFeasibleThresholdError(
            "Cohort has no usable outcome: final_macro_dice is NaN for all evaluated thresholds."
        )

    best = max(
        finite_candidates,
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


def _get_semantic(lineage: Mapping[str, Any], key: str) -> Any:
    """Read one measured semantic out of a lineage record.

    ``validate_lineage_completeness`` has already proved the field exists and
    is non-``None``, so a lookup failure here is a programming error rather
    than a tolerable absence.
    """

    semantics = lineage.get("semantics")
    if not isinstance(semantics, Mapping) or key not in semantics:
        raise ValueError(f"lineage record has no semantics.{key}")
    return semantics[key]


def json_safe(value: Any) -> Any:
    """Convert values into strictly JSON-compliant structures (replacing NaN/Inf with null)."""
    return json_safe_artifact(value)


_json_safe = json_safe


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
    metric_contract: str | None = None,
    lineage: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the schema-v2 calibration artifact and return the payload.

    The artifact is the single record of which ``tau_accept`` a run selected,
    on which split, under which neutral margin, in which metric space, and
    against which checkpoint.  It always carries ``validity ==
    "diagnostic_only"`` plus the verbatim :data:`VALIDITY_REASON`, because
    ``tau_accept`` is selected by argmax on the split it is measured on.

    ``checkpoint_sha256`` is computed when the checkpoint file exists and is
    readable, and is ``None`` otherwise -- a missing checkpoint never fails
    the write.

    ``lineage`` is the producer's record of the weights, preprocessing recipe,
    metric semantics, split membership and cohort the threshold was measured
    under.  When supplied it is checked for completeness *at write time*, so an
    artifact never ships a half-filled lineage that would only fail later.
    When omitted, ``lineage`` is written as an explicit ``null``: the artifact
    stays readable, but every deployable consumer refuses it as unverified.
    Serialising a lineage is not verifying one -- verification happens at the
    consumer, against the runtime, in
    :func:`self_audit.evaluation.calibration_lineage.verify_calibration_lineage`.
    """

    if metric_space not in METRIC_SPACES:
        raise ValueError(f"metric_space must be one of {list(METRIC_SPACES)}, got {metric_space!r}")
    row_space = selected_row.get("metric_space")
    if row_space is not None and str(row_space) != str(metric_space):
        raise ContractMismatchError(
            f"save_calibration metric_space '{metric_space}' conflicts with selected_row metric_space '{row_space}'"
        )

    from .calibration_lineage import (
        validate_calibration_header,
        validate_header_lineage_consistency,
        validate_lineage_completeness,
    )

    validate_calibration_header(
        {
            "schema_version": int(CALIBRATION_SCHEMA_VERSION),
            "tau_accept": tau_accept,
            "t_max": t_max,
            "selected_row": selected_row,
        },
        where=f"save_calibration for {path}",
        error_cls=ContractMismatchError,
    )

    file_sha = _file_sha256(checkpoint_path) if checkpoint_path is not None else None
    lineage_sha: str | None = None
    lineage_record: dict[str, Any] | None = None
    if lineage is not None:
        lineage_record = _json_safe(
            dict(validate_lineage_completeness(lineage, where=f"lineage for {path}"))
        )
        lineage_ckpt = lineage_record.get("checkpoint")
        if isinstance(lineage_ckpt, Mapping):
            raw_sha = lineage_ckpt.get("checkpoint_sha256")
            if isinstance(raw_sha, str):
                lineage_sha = raw_sha

        if file_sha is not None and lineage_sha is not None and file_sha != lineage_sha:
            raise ContractMismatchError(
                f"save_calibration checkpoint_path '{checkpoint_path}' SHA-256 {file_sha} conflicts with "
                f"lineage checkpoint_sha256 {lineage_sha}"
            )

        recorded_space = _get_semantic(lineage_record, "metric_space")
        if str(recorded_space) != str(metric_space):
            raise ContractMismatchError(
                f"save_calibration metric_space '{metric_space}' conflicts with the lineage's "
                f"measured metric_space '{recorded_space}'"
            )
        recorded_margin = _get_semantic(lineage_record, "neutral_margin")
        resolved_margin = resolve_neutral_margin(neutral_margin)
        if float(recorded_margin) != float(resolved_margin):
            raise ContractMismatchError(
                f"save_calibration neutral_margin {resolved_margin!r} conflicts with the "
                f"lineage's measured neutral_margin {recorded_margin!r}"
            )
        recorded_t_max = _get_semantic(lineage_record, "t_max")
        if int(recorded_t_max) != int(t_max):
            raise ContractMismatchError(
                f"save_calibration t_max {int(t_max)!r} conflicts with the lineage's measured "
                f"t_max {recorded_t_max!r}"
            )
        lineage_split = lineage_record.get("split")
        if isinstance(lineage_split, Mapping):
            recorded_split = lineage_split.get("split_name")
            if recorded_split is not None and str(source_split) != str(recorded_split):
                raise ContractMismatchError(
                    f"save_calibration source_split '{source_split}' conflicts with lineage "
                    f"split_name '{recorded_split}'"
                )
        recorded_contract = _get_semantic(lineage_record, "metric_contract")
        if metric_contract is not None and recorded_contract is not None and str(metric_contract) != str(recorded_contract):
            raise ContractMismatchError(
                f"save_calibration metric_contract '{metric_contract}' conflicts with lineage "
                f"metric_contract '{recorded_contract}'"
            )
        if metric_contract is None and recorded_contract is not None:
            metric_contract = str(recorded_contract)

        recorded_contract_version = _get_semantic(lineage_record, "metric_contract_version")
        if recorded_contract_version is not None:
            metric_contract_version = int(recorded_contract_version)
        else:
            metric_contract_version = 1 if metric_contract is not None else None
    else:
        metric_contract_version = 1 if metric_contract is not None else None

    # Derive declared identity from lineage when checkpoint_path is omitted or unreadable
    if file_sha is not None:
        recorded_checkpoint_sha = file_sha
    elif lineage_sha is not None:
        recorded_checkpoint_sha = lineage_sha
    else:
        recorded_checkpoint_sha = None

    payload: dict[str, Any] = {
        "schema_version": int(CALIBRATION_SCHEMA_VERSION),
        "tau_accept": float(tau_accept),
        "neutral_margin": resolve_neutral_margin(neutral_margin),
        "source_split": str(source_split),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": recorded_checkpoint_sha,
        "t_max": int(t_max),
        "threshold_grid": _grid_dict(threshold_grid),
        "selected_row": _json_safe(dict(selected_row)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_space": str(metric_space),
        "metric_contract": str(metric_contract) if metric_contract is not None else None,
        "metric_contract_version": metric_contract_version,
        "lineage": lineage_record,
        "validity": VALIDITY_DIAGNOSTIC_ONLY,
        "validity_reason": VALIDITY_REASON,
        "extra": _json_safe(dict(extra)) if extra is not None else {},
    }
    if lineage_record is not None:
        validate_header_lineage_consistency(
            payload, lineage_record, where=f"save_calibration for {path}", error_cls=ContractMismatchError
        )
    destination = Path(os.fspath(path))
    atomic_write_json(destination, payload, indent=2, sort_keys=True)
    return payload


def load_calibration(path: Any) -> dict[str, Any]:
    """Read a schema-v2 calibration artifact, validating it strictly.

    An unknown ``schema_version``, an unknown top-level key, a missing
    required key, or a ``validity`` other than ``"diagnostic_only"`` raises.
    Nothing is silently ignored: a calibration artifact this loader does not
    fully understand must not be allowed to hand a threshold to inference.

    A schema-v1 artifact predates the lineage block and is refused with a
    pointer to
    :func:`self_audit.evaluation.calibration_lineage.inspect_legacy_calibration`,
    which reads it as explicitly unverified.  Loading succeeds with
    ``lineage is None``; that payload is readable but is rejected by every
    deployable consumer, which must call ``verify_calibration_lineage``
    against a runtime-derived expectation before using ``tau_accept``.
    """

    source = Path(os.fspath(path))
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Calibration artifact {source} must be a JSON object")
    version = payload.get("schema_version")
    if (
        isinstance(version, int)
        and not isinstance(version, bool)
        and version in LEGACY_CALIBRATION_SCHEMA_VERSIONS
    ):
        raise ValueError(
            f"Calibration artifact {source} is schema_version {version}, which predates the "
            "required lineage block. It cannot supply a threshold to a calibrated evaluation. "
            "Read it with self_audit.evaluation.calibration_lineage.inspect_legacy_calibration(), "
            "which reports it as explicitly unverified, or re-run calibration."
        )
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
        raise ValueError(
            f"Calibration artifact {source} top-level {', '.join(missing)} is missing"
        )
    if payload["validity"] != VALIDITY_DIAGNOSTIC_ONLY:
        raise ValueError(
            f"Calibration artifact {source} declares validity {payload['validity']!r}; "
            f"only {VALIDITY_DIAGNOSTIC_ONLY!r} is a valid calibration class"
        )

    from .calibration_lineage import (
        validate_calibration_header,
        validate_header_lineage_consistency,
    )

    validate_calibration_header(payload, where=f"Calibration artifact {source}", error_cls=ValueError)
    if payload.get("lineage") is not None and isinstance(payload["lineage"], Mapping):
        validate_header_lineage_consistency(
            payload, payload["lineage"], where=f"Calibration artifact {source}", error_cls=ValueError
        )

    payload["tau_accept"] = float(payload["tau_accept"])
    payload["neutral_margin"] = resolve_neutral_margin(payload["neutral_margin"])
    payload["t_max"] = int(payload["t_max"])
    payload.setdefault("extra", {})
    payload.setdefault("lineage", None)
    return payload


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "SUPPORTED_CACHE_SCHEMA_VERSIONS",
    "CALIBRATION_SCHEMA_VERSION",
    "LEGACY_CALIBRATION_SCHEMA_VERSIONS",
    "NoFeasibleThresholdError",
    "SELECTION_OBJECTIVE",
    "VALIDITY_DIAGNOSTIC_ONLY",
    "VALIDITY_REASON",
    "evaluate_threshold",
    "json_safe",
    "load_calibration",
    "save_calibration",
    "select_threshold",
    "sweep_thresholds",
    "validate_and_normalize_transition_cache",
]
