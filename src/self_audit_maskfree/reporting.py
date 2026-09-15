"""Bounded image-only reporting for the mask-free 150 epoch run.

This module is deliberately downstream of the freeze and image-only
verification calls.  It consumes manifests and already materialized rows; it
does not read images, masks, references, or checkpoints.  The report keeps
appearance evidence (predictive NLL) separate from any later reference
evaluation and preserves unavailable values as ``None`` with a reason.

The public entry point is :func:`write_image_only_reports`.  It writes one
JSON summary, one metric-row CSV, and one Markdown view below
``<run_root>/reports`` and, for multi-split input, the same set below each
``<run_root>/splits/<split>/reports`` directory.  It returns the all-split
paths.
"""
from __future__ import annotations

import dataclasses
import json
import math
import numbers
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import VERSION
from .evaluation.metrics import (
    metric_row,
    paired_patient_bootstrap,
    write_csv,
    write_json,
    write_markdown,
)

REPORT_SCHEMA_VERSION = "maskfree150.reporting.v1"
REPORT_FILENAMES = {
    "json": "image_only_summary.json",
    "csv": "image_only_summary.csv",
    "md": "image_only_summary.md",
}

_STANDARD_ROW_KEYS = {
    "name",
    "value",
    "unit",
    "dataset",
    "split",
    "protocol",
    "epoch",
    "checkpoint",
    "population",
    "count",
    "available",
    "reason",
    "contract_version",
    "extra",
}

_PAIR_COMPARISONS = (
    ("E5_evidence_plus_challenge", "E1_cuts_inspired_control"),
    ("E5_evidence_plus_challenge", "E2_confidence_consistency"),
    ("E5_evidence_plus_challenge", "E3_fitting_score"),
    ("E5_evidence_plus_challenge", "E4_selection_evidence"),
    ("student_audited", "student_no_audit"),
)

_METHOD_ALIASES = {
    "E1_cuts_inspired_control": ("E1_cuts_inspired_control", "E1"),
    "E2_confidence_consistency": ("E2_confidence_consistency", "E2"),
    "E3_fitting_score": ("E3_fitting_score", "E3"),
    "E4_selection_evidence": ("E4_selection_evidence", "E4"),
    "E5_evidence_plus_challenge": ("E5_evidence_plus_challenge", "E5"),
    "student_audited": ("student_audited", "audited_student"),
    "student_no_audit": ("student_no_audit", "unaudited_student"),
}


class ReportingContractError(ValueError):
    """Raised when a report input cannot be represented honestly."""


# ---------------------------------------------------------------------------
# input and scalar helpers
# ---------------------------------------------------------------------------
def _read_jsonish(value: Any) -> Any:
    """Load an optional JSON/JSONL path while accepting in-memory rows."""
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            return value
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        return json.loads(text)
    return value


def _mapping(value: Any) -> dict[str, Any]:
    """Return a shallow mapping for ordinary dicts and simple dataclasses."""
    value = _read_jsonish(value)
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if dataclasses.is_dataclass(value):
        return {str(key): item for key, item in dataclasses.asdict(value).items()}
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return {str(key): item for key, item in vars(value).items() if not key.startswith("_")}
    return {}


def _rows(value: Any, *preferred_keys: str) -> list[dict[str, Any]]:
    """Materialize row collections from a list or a ``{"rows": [...]}`` payload."""
    value = _read_jsonish(value)
    if isinstance(value, Mapping):
        selected: Any = None
        for key in preferred_keys or ("rows",):
            if key in value:
                selected = value[key]
                break
        if selected is None and "rows" in value:
            selected = value["rows"]
        if selected is None:
            # A single row is useful for a one-unit synthetic caller.
            selected = [value] if value else []
    else:
        selected = value
    if selected is None or isinstance(selected, (str, bytes)):
        return []
    if not isinstance(selected, Iterable):
        return []
    materialized: list[dict[str, Any]] = []
    for item in selected:
        row = _mapping(item)
        if row:
            materialized.append(row)
    return materialized


def _safe(value: Any) -> Any:
    """Convert tensors/scalars/dataclasses into finite JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _safe(dataclasses.asdict(value))
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return _safe(value.detach().cpu().tolist())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _safe(value.tolist())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _number(value: Any) -> float | None:
    """Return a finite scalar, excluding booleans and non-scalar arrays."""
    if value is None or isinstance(value, bool):
        return None
    if hasattr(value, "numel"):
        try:
            if int(value.numel()) != 1:
                return None
            value = value.detach().cpu().item()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    elif hasattr(value, "item") and not isinstance(value, numbers.Number):
        try:
            value = value.item()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_count(value: Any, default: int = 0) -> int:
    number = _number(value)
    if number is None or number < 0 or not number.is_integer():
        return int(default)
    return int(number)


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _provenance(
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    epoch: int | None,
    checkpoint: Any,
) -> dict[str, Any]:
    """Resolve row-level provenance without manufacturing a numeric metric."""
    dataset = _first(row, "dataset") or _first(manifest, "dataset") or "unknown"
    split = _first(row, "split") or _first(manifest, "split") or "unknown"
    protocol = (
        _first(row, "protocol")
        or _first(manifest, "resolved_protocol", "protocol")
        or "unknown"
    )
    row_epoch = _first(row, "epoch", "global_epoch")
    row_checkpoint = _first(row, "checkpoint")
    row_epoch_number = _number(row_epoch)
    resolved_epoch = (
        int(row_epoch_number)
        if row_epoch_number is not None and row_epoch_number.is_integer() and row_epoch_number >= 0
        else epoch
    )
    return {
        "dataset": str(dataset),
        "split": str(split),
        "protocol": str(protocol),
        "epoch": resolved_epoch,
        "checkpoint": str(row_checkpoint if row_checkpoint is not None else checkpoint),
    }


def _metric(
    name: str,
    value: Any,
    *,
    provenance: Mapping[str, Any],
    unit: str,
    population: str,
    count: Any,
    available: bool | None = None,
    reason: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a strict metric row, making a missing/nonfinite value unavailable."""
    number = _number(value)
    if available is None:
        available = number is not None
    if number is None:
        available = False
    if not available and not reason:
        reason = "metric was not available in the supplied rows"
    return metric_row(
        name,
        number if available else None,
        unit=unit,
        dataset=str(provenance.get("dataset", "unknown")),
        split=str(provenance.get("split", "unknown")),
        protocol=str(provenance.get("protocol", "unknown")),
        epoch=provenance.get("epoch"),
        checkpoint=str(provenance.get("checkpoint", "unknown")),
        population=population,
        count=_nonnegative_count(count),
        available=bool(available),
        reason=reason,
        **_safe(extra),
    )


def _normalise_input_metric_row(
    row: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    epoch: int | None,
    checkpoint: Any,
    prefix: str,
) -> dict[str, Any]:
    """Re-wrap an existing experiment/audit row through the strict row helper."""
    provenance = _provenance(row, manifest, epoch, checkpoint)
    name = str(_first(row, "name", "metric", "experiment") or prefix)
    value = row.get("value")
    if value is None:
        selection = _mapping(row.get("selection_score"))
        if selection:
            value = _first(selection, "total", "normalized_nll")
    available = bool(row.get("available", value is not None))
    reason = row.get("reason")
    if not available and not reason:
        reason = "input row marked unavailable"
    if available and _number(value) is None:
        available = False
        reason = reason or "input row has no finite metric value"
    extra = _mapping(row.get("extra"))
    for key, item in row.items():
        if key not in _STANDARD_ROW_KEYS:
            extra.setdefault(key, item)
    return _metric(
        name,
        value,
        provenance=provenance,
        unit=str(row.get("unit") or "scalar"),
        population=str(row.get("population") or "observation unit"),
        count=row.get("count", 0),
        available=available,
        reason=str(reason) if reason else None,
        **extra,
    )


# ---------------------------------------------------------------------------
# verification score extraction and aggregation
# ---------------------------------------------------------------------------
def _score_map(candidate: Mapping[str, Any]) -> dict[str, Any]:
    nested = _mapping(candidate.get("verify_score"))
    if nested:
        return nested
    nested = _mapping(candidate.get("score"))
    return nested or dict(candidate)


def _score_info(candidate: Mapping[str, Any]) -> dict[str, Any]:
    score = _score_map(candidate)
    count = _nonnegative_count(score.get("count"))
    nll_sum = _number(_first(score, "nll_sum", "raw_nll_sum", "rawNLLsum"))
    normalized = _number(_first(score, "normalized_nll", "normalizedNLL"))
    derived_normalized = False
    if normalized is None and nll_sum is not None and count > 0:
        normalized = nll_sum / count
        derived_normalized = True
    declared_available = score.get("available")
    available = bool(declared_available) if declared_available is not None else normalized is not None
    if count <= 0:
        available = False
    reason = score.get("reason")
    if not available and not reason:
        reason = "verification score has no positive observed-pixel support"
    gap = _number(
        _first(
            candidate,
            "selection_to_verification_gap",
            "selection_gap",
            "selectiongap",
            "verification_gap",
        )
    )
    if gap is None:
        gap = _number(
            _first(
                score,
                "selection_to_verification_gap",
                "selection_gap",
                "selectiongap",
                "verification_gap",
            )
        )
    return {
        "nll_sum": nll_sum,
        "count": count,
        "normalized_nll": normalized if available else None,
        "selection_to_verification_gap": gap if available else None,
        "available": available and normalized is not None,
        "reason": None if available and normalized is not None else str(reason),
        "derived_normalized_nll": derived_normalized,
        "complexity": _number(score.get("complexity")),
        "prior": _number(score.get("prior")),
        "total": _number(score.get("total")),
        "role": score.get("role"),
    }


def _candidate_entries(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("candidates")
    entries: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        for candidate_id, value in raw.items():
            item = _mapping(value)
            item.setdefault("candidate_id", str(candidate_id))
            entries.append(item)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for value in raw:
            item = _mapping(value)
            if item:
                entries.append(item)
    return entries


def _method_maps(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str | None]]:
    result: dict[str, dict[str, str | None]] = {}
    for row in rows:
        unit_id = str(_first(row, "unit_id", "study_unit_id", "id") or "unknown_unit")
        mapping = _mapping(row.get("method_to_candidate"))
        # W6 verification exposes the same relationship both as the compact
        # ``method_to_candidate`` mapping and as per-method score records.  A
        # report may receive the latter when a caller persisted only the
        # richer verification result, so retain that relationship here rather
        # than dropping the student or control methods.
        if not mapping:
            methods = _mapping(row.get("methods"))
            mapping = {
                str(method): _mapping(details).get("candidate_id")
                for method, details in methods.items()
            }
        result[unit_id] = {
            str(method): None if candidate is None else str(candidate)
            for method, candidate in mapping.items()
        }
    return result


def _verification_records(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    epoch: int | None,
    checkpoint: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str | None]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    candidate_metric_rows: list[dict[str, Any]] = []
    maps = _method_maps(rows)
    for row in rows:
        provenance = _provenance(row, manifest, epoch, checkpoint)
        unit_id = str(_first(row, "unit_id", "study_unit_id", "id") or "unknown_unit")
        patient_id = str(_first(row, "patient_id", "study_id") or unit_id)
        candidates = _candidate_entries(row)
        by_id = {
            str(item.get("candidate_id")): item
            for item in candidates
            if item.get("candidate_id") is not None
        }
        for item in candidates:
            candidate_id = str(item.get("candidate_id", "unknown_candidate"))
            info = _score_info(item)
            base = f"verification.candidate.{candidate_id}"
            candidate_metric_rows.append(
                _metric(
                    f"{base}.raw_nll_sum",
                    info["nll_sum"],
                    provenance=provenance,
                    unit="nats",
                    population="observation pixels",
                    count=info["count"],
                    available=info["nll_sum"] is not None and info["available"],
                    reason=info["reason"],
                    candidate_id=candidate_id,
                )
            )
            candidate_metric_rows.append(
                _metric(
                    f"{base}.normalized_nll",
                    info["normalized_nll"],
                    provenance=provenance,
                    unit="nats/pixel",
                    population="observation pixels",
                    count=info["count"],
                    available=info["available"],
                    reason=info["reason"],
                    candidate_id=candidate_id,
                    derived_from_raw_count=info["derived_normalized_nll"],
                )
            )
            candidate_metric_rows.append(
                _metric(
                    f"{base}.selection_to_verification_gap",
                    info["selection_to_verification_gap"],
                    provenance=provenance,
                    unit="nats/pixel",
                    population="observation pixels",
                    count=info["count"],
                    available=info["selection_to_verification_gap"] is not None,
                    reason=(
                        None
                        if info["selection_to_verification_gap"] is not None
                        else "selection-to-verification gap unavailable for this candidate"
                    ),
                    candidate_id=candidate_id,
                )
            )
        for method, candidate_id in maps.get(unit_id, {}).items():
            if candidate_id is None or candidate_id not in by_id:
                records.append(
                    {
                        "method": method,
                        "candidate_id": candidate_id,
                        "unit_id": unit_id,
                        "patient_id": patient_id,
                        "provenance": provenance,
                        "score": {
                            "available": False,
                            "reason": "method-to-candidate mapping has no matching verification candidate",
                            "count": 0,
                            "normalized_nll": None,
                            "nll_sum": None,
                            "selection_to_verification_gap": None,
                        },
                    }
                )
                continue
            records.append(
                {
                    "method": method,
                    "candidate_id": candidate_id,
                    "unit_id": unit_id,
                    "patient_id": patient_id,
                    "provenance": provenance,
                    "score": _score_info(by_id[candidate_id]),
                }
            )
    return records, maps, candidate_metric_rows


def _candidate_payload(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expose the frozen candidate score fields in a compact JSON table."""
    payload: list[dict[str, Any]] = []
    for row in rows:
        unit_id = str(_first(row, "unit_id", "study_unit_id", "id") or "unknown_unit")
        patient_id = str(_first(row, "patient_id", "study_id") or unit_id)
        for candidate in _candidate_entries(row):
            info = _score_info(candidate)
            payload.append(
                {
                    "unit_id": unit_id,
                    "patient_id": patient_id,
                    "candidate_id": candidate.get("candidate_id"),
                    "verify_score": {
                        "nll_sum": info["nll_sum"],
                        "count": info["count"],
                        "normalized_nll": info["normalized_nll"],
                        "available": info["available"],
                        "reason": info["reason"],
                    },
                    "selection_to_verification_gap": info["selection_to_verification_gap"],
                }
            )
    return payload


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _method_values(method_values: Mapping[str, Mapping[str, float]], name: str) -> dict[str, float]:
    """Resolve current long method names and compact E1-E5 aliases."""
    for alias in _METHOD_ALIASES.get(name, (name,)):
        if alias in method_values:
            return dict(method_values[alias])
    return {}


def _aggregate_method(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate units within patients, then patients; retain a raw pooled view."""
    usable = [
        row
        for row in records
        if _mapping(row.get("score")).get("available")
        and _number(_mapping(row.get("score")).get("normalized_nll")) is not None
        and _nonnegative_count(_mapping(row.get("score")).get("count")) > 0
    ]
    by_patient: dict[str, list[float]] = defaultdict(list)
    gap_by_patient: dict[str, list[float]] = defaultdict(list)
    total_nll = 0.0
    total_count = 0
    for row in usable:
        score = _mapping(row["score"])
        patient = str(row.get("patient_id") or row.get("unit_id") or "unknown_patient")
        normalized = float(score["normalized_nll"])
        by_patient[patient].append(normalized)
        gap = _number(score.get("selection_to_verification_gap"))
        if gap is not None:
            gap_by_patient[patient].append(gap)
        nll_sum = _number(score.get("nll_sum"))
        count = _nonnegative_count(score.get("count"))
        if nll_sum is not None and count > 0:
            total_nll += nll_sum
            total_count += count
    patient_values = {patient: sum(values) / len(values) for patient, values in by_patient.items()}
    patient_gaps = {patient: sum(values) / len(values) for patient, values in gap_by_patient.items()}
    return {
        "unit_count": len(usable),
        "patient_count": len(patient_values),
        "patient_values": patient_values,
        "patient_gap_values": patient_gaps,
        "patient_macro_normalized_nll": _mean(list(patient_values.values())),
        "patient_macro_gap": _mean(list(patient_gaps.values())),
        "raw_nll_sum": total_nll if total_count > 0 else None,
        "raw_count": total_count,
        "raw_pooled_normalized_nll": total_nll / total_count if total_count > 0 else None,
        "available": bool(patient_values),
        "reason": None if patient_values else "no verification candidate had valid positive support",
    }


def _paired_bootstrap_rows(
    method_values: Mapping[str, Mapping[str, float]],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for audited_name, comparator_name in _PAIR_COMPARISONS:
        audited = _method_values(method_values, audited_name)
        comparator = _method_values(method_values, comparator_name)
        if not audited or not comparator:
            result = {
                "available": False,
                "reason": "one or both methods lack valid patient-level support/count",
                "count": len(set(audited) & set(comparator)),
                "patients": sorted(set(audited) & set(comparator)),
                "mean_difference": None,
                "ci_low": None,
                "ci_high": None,
            }
        else:
            result = paired_patient_bootstrap(audited, comparator, seed=42)
        key = f"{audited_name}_minus_{comparator_name}"
        results[key] = {
            **_safe(result),
            "direction": f"{audited_name} - {comparator_name}; negative means lower appearance NLL for the first method",
        }
        rows.append(
            _metric(
                f"paired_patient_bootstrap.{key}.mean_difference",
                result.get("mean_difference"),
                provenance=provenance,
                unit="nats/pixel",
                population="paired patients",
                count=result.get("count", 0),
                available=bool(result.get("available")),
                reason=result.get("reason"),
                ci_low=result.get("ci_low"),
                ci_high=result.get("ci_high"),
                iterations=result.get("iterations"),
                alpha=result.get("alpha"),
                seed=result.get("seed"),
                comparator=comparator_name,
                audited=audited_name,
            )
        )
    return results, rows


def _repeat_stability_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate an explicit independent pre-freeze repeat by method.

    ``repeated_annotation_stability`` is intentionally a separate payload from
    W6's top-two verification ``bank_agreement``.  Agreement values are first
    averaged over units within each patient, then over patients.  Support
    counts remain additive metadata and are never used to turn an unavailable
    valid-mask agreement into a fabricated number.
    """
    entries_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    payload_rows = 0
    payload_units: set[str] = set()
    payload_patients: set[str] = set()
    protocols: set[str] = set()
    scopes: set[str] = set()
    for row in rows:
        payload = _mapping(row.get("repeated_annotation_stability"))
        if not payload or payload.get("available") is False:
            continue
        payload_rows += 1
        unit_id = str(_first(row, "unit_id", "study_unit_id", "id") or "unknown_unit")
        patient_id = str(_first(row, "patient_id", "study_id") or unit_id)
        payload_units.add(unit_id)
        payload_patients.add(patient_id)
        protocol = payload.get("protocol")
        scope = payload.get("scope")
        if protocol is not None:
            protocols.add(str(protocol))
        if scope is not None:
            scopes.add(str(scope))
        for method, raw_entry in _mapping(payload.get("methods")).items():
            entry = _mapping(raw_entry)
            if entry.get("available") is False:
                continue
            agreement = _number(entry.get("agreement"))
            if agreement is None:
                continue
            valid_agreement = _number(entry.get("valid_agreement"))
            common_valid_count = _number(entry.get("common_valid_count"))
            if common_valid_count is not None and (
                common_valid_count < 0 or not common_valid_count.is_integer()
            ):
                common_valid_count = None
            support_count = _number(entry.get("count"))
            if support_count is not None and (
                support_count < 0 or not support_count.is_integer()
            ):
                support_count = None
            entries_by_method[str(method)].append(
                {
                    "unit_id": unit_id,
                    "patient_id": patient_id,
                    "agreement": agreement,
                    "valid_agreement": valid_agreement,
                    "common_valid_count": (
                        int(common_valid_count) if common_valid_count is not None else None
                    ),
                    "count": int(support_count) if support_count is not None else None,
                }
            )

    methods: dict[str, dict[str, Any]] = {}
    for method, entries in sorted(entries_by_method.items()):
        by_patient: dict[str, list[float]] = defaultdict(list)
        valid_by_patient: dict[str, list[float]] = defaultdict(list)
        common_counts: list[int] = []
        support_counts: list[int] = []
        for entry in entries:
            patient_id = str(entry["patient_id"])
            by_patient[patient_id].append(float(entry["agreement"]))
            if entry["valid_agreement"] is not None:
                valid_by_patient[patient_id].append(float(entry["valid_agreement"]))
            if entry["common_valid_count"] is not None:
                common_counts.append(int(entry["common_valid_count"]))
            if entry["count"] is not None:
                support_counts.append(int(entry["count"]))
        patient_agreement = {
            patient: _mean(values) for patient, values in by_patient.items()
        }
        patient_valid_agreement = {
            patient: _mean(values) for patient, values in valid_by_patient.items()
        }
        methods[method] = {
            "agreement": _mean(list(patient_agreement.values())),
            "valid_agreement": _mean(list(patient_valid_agreement.values())),
            "common_valid_count": sum(common_counts) if common_counts else None,
            "count": sum(support_counts) if support_counts else len(entries),
            "unit_count": len(entries),
            "patient_count": len(patient_agreement),
            "valid_agreement_count": len(patient_valid_agreement),
            "available": bool(patient_agreement),
            "reason": None
            if patient_agreement
            else "repeat agreement had no finite unit-level values",
            "patient_agreement": patient_agreement,
            "patient_valid_agreement": patient_valid_agreement,
        }

    def _declared(values: set[str]) -> str | None:
        if not values:
            return None
        return next(iter(values)) if len(values) == 1 else "mixed"

    complete = bool(rows) and payload_rows == len(rows)
    return {
        "available": bool(methods),
        "value": None,
        "count": payload_rows,
        "unit_count": len(payload_units),
        "patient_count": len(payload_patients),
        "complete": complete,
        "protocol": _declared(protocols),
        "scope": _declared(scopes),
        "methods": methods,
        "reason": None
        if methods and complete
        else (
            "repeat stability payload was missing for one or more verification rows"
            if methods
            else "independent repeated-annotation stability was not supplied"
        ),
    }


def _stability_metric_rows(
    stability: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build strict metric rows for explicit repeated-annotation stability."""
    rows: list[dict[str, Any]] = []
    for method, summary in sorted(_mapping(stability.get("methods")).items()):
        item = _mapping(summary)
        unit_count = item.get("unit_count", 0)
        rows.append(
            _metric(
                f"verification.stability.{method}.agreement",
                item.get("agreement"),
                provenance=provenance,
                unit="fraction",
                population="patients",
                count=item.get("patient_count", 0),
                available=bool(item.get("available")),
                reason=item.get("reason"),
                method=method,
                unit_count=unit_count,
                aggregation="mean per unit, then mean per patient",
                repeat_protocol=stability.get("protocol"),
                repeat_scope=stability.get("scope"),
            )
        )
        valid_value = item.get("valid_agreement")
        valid_available = valid_value is not None and bool(item.get("available"))
        rows.append(
            _metric(
                f"verification.stability.{method}.valid_agreement",
                valid_value,
                provenance=provenance,
                unit="fraction",
                population="patients with common valid support",
                count=item.get("valid_agreement_count", 0),
                available=valid_available,
                reason=(
                    None
                    if valid_available
                    else "valid-mask agreement was unavailable for this method"
                ),
                method=method,
                unit_count=unit_count,
                aggregation="mean per unit, then mean per patient",
                repeat_protocol=stability.get("protocol"),
                repeat_scope=stability.get("scope"),
            )
        )
        common_count = item.get("common_valid_count")
        rows.append(
            _metric(
                f"verification.stability.{method}.common_valid_count",
                common_count,
                provenance=provenance,
                unit="pixels",
                population="repeated image-only observations",
                count=unit_count,
                available=common_count is not None,
                reason=(
                    None
                    if common_count is not None
                    else "common valid support count was not supplied"
                ),
                method=method,
                repeat_protocol=stability.get("protocol"),
                repeat_scope=stability.get("scope"),
            )
        )
    return rows


def _verification_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: str,
) -> dict[str, Any]:
    total = len(rows)
    if not total:
        unavailable = {"available": False, "value": None, "reason": "no verification rows supplied"}
        return {
            "degenerate_flags": unavailable,
            "ambiguity": unavailable,
            "stability": {
                **_repeat_stability_summary(rows),
                "reason": "independent repeated-annotation stability was not supplied",
            },
            "ranking_agreement": unavailable,
            "sensitivity": unavailable,
            "temporal": {
                "available": False,
                "value": None,
                "reason": "no verification rows supplied; temporal status is unknown",
            },
        }

    # Degenerate flags are available only when each row exposes the relevant
    # pre-freeze diagnostic field. Missing fields remain unknown.
    flags: list[dict[str, Any]] = []
    missing_degenerate = 0
    for row in rows:
        if "falsification_flag" in row:
            if bool(row.get("falsification_flag")):
                flags.append({"unit_id": row.get("unit_id"), "kind": "falsification_flag"})
            continue
        challenges = row.get("degenerate_challenges")
        if isinstance(challenges, Sequence) and not isinstance(challenges, (str, bytes)):
            for challenger in challenges:
                item = _mapping(challenger)
                if bool(item.get("beats_selected")):
                    flags.append(
                        {
                            "unit_id": row.get("unit_id"),
                            "kind": "degenerate_beats_selected",
                            "candidate_id": item.get("candidate_id"),
                        }
                    )
            continue
        missing_degenerate += 1
    degenerate_available = missing_degenerate == 0
    degenerate = {
        "available": degenerate_available,
        "flag": bool(flags) if degenerate_available else None,
        "flag_count": len(flags) if degenerate_available else None,
        "flags": flags,
        "count": total,
        "reason": None if degenerate_available else "some verification rows omit degenerate challenge outcomes",
    }

    unresolved: list[bool] = []
    alternatives = 0
    missing_ambiguity = 0
    for row in rows:
        if "semantic_unresolved" in row:
            unresolved.append(bool(row.get("semantic_unresolved")))
        elif "candidates" in row:
            candidates = _candidate_entries(row)
            values = [item.get("semantic_unresolved") for item in candidates if "semantic_unresolved" in item]
            if values:
                unresolved.append(any(bool(value) for value in values))
            else:
                missing_ambiguity += 1
        else:
            missing_ambiguity += 1
        for item in _candidate_entries(row):
            value = item.get("alternatives")
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                alternatives += len(value)
            elif _number(value) is not None:
                alternatives += _nonnegative_count(value)
    ambiguity_available = bool(unresolved) and missing_ambiguity == 0
    ambiguity = {
        "available": ambiguity_available,
        "semantic_unresolved_units": sum(unresolved) if ambiguity_available else None,
        "semantic_unresolved_rate": (
            sum(unresolved) / len(unresolved) if ambiguity_available and unresolved else None
        ),
        "alternative_count": alternatives if ambiguity_available else None,
        "count": len(unresolved) if ambiguity_available else 0,
        "reason": None if ambiguity_available else "semantic ambiguity was not recorded for every verification row",
    }

    # W6's ``bank_agreement``/legacy ``annotation_stability`` is agreement
    # between the top two verification-ranked candidates.  It is a useful
    # ranking diagnostic, but it is not repeated-annotation stability.  Keep
    # the two labels separate and leave true stability unavailable unless an
    # independent repeated annotation is explicitly supplied.
    ranking_values: dict[str, list[float]] = defaultdict(list)
    sensitivity_values: list[float] = []
    missing_ranking_agreement = 0
    missing_sensitivity = 0
    for row in rows:
        agreement = _mapping(row.get("bank_agreement"))
        if not agreement:
            agreement = _mapping(row.get("annotation_stability"))
        if not agreement or agreement.get("available") is False:
            missing_ranking_agreement += 1
        else:
            for key, value in _flatten_scalars(agreement):
                if key in {"count", "available"}:
                    continue
                number = _number(value)
                if number is not None:
                    ranking_values[key].append(number)
        perturbations = row.get("perturbation_sensitivity")
        if not isinstance(perturbations, Sequence) or isinstance(perturbations, (str, bytes)):
            missing_sensitivity += 1
            continue
        found = False
        for perturbation in perturbations:
            delta = _number(_mapping(perturbation).get("delta_vs_unperturbed"))
            if delta is not None:
                sensitivity_values.append(delta)
                found = True
        if not found:
            missing_sensitivity += 1
    ranking_available = bool(ranking_values) and missing_ranking_agreement == 0
    ranking_agreement = {
        "available": ranking_available,
        "metrics": {
            key: {"mean": _mean(values), "count": len(values)}
            for key, values in sorted(ranking_values.items())
        },
        "count": total - missing_ranking_agreement,
        "reason": None
        if ranking_available
        else "top-2 verification-ranking agreement was unavailable for one or more units",
        "note": "candidate agreement in the verification ranking is not repeated-annotation stability",
    }
    stability = _repeat_stability_summary(rows)
    stability["ranking_agreement"] = ranking_agreement
    sensitivity_available = bool(sensitivity_values) and missing_sensitivity == 0
    sensitivity = {
        "available": sensitivity_available,
        "mean_delta_vs_unperturbed": _mean(sensitivity_values) if sensitivity_available else None,
        "count": len(sensitivity_values) if sensitivity_available else 0,
        "reason": None if sensitivity_available else "perturbation sensitivity unavailable for one or more units",
    }

    temporal_values = []
    for row in rows:
        for key, value in _flatten_scalars(row):
            if "temporal" in key.lower() and key.lower().endswith(("nll", "error", "residual", "delta")):
                number = _number(value)
                if number is not None:
                    temporal_values.append(number)
    if str(protocol) != "cine_predictive":
        temporal = {
            "available": False,
            "value": None,
            "count": 0,
            "reason": f"temporal predictive transport unavailable; resolved protocol is {protocol!r}",
        }
    elif temporal_values:
        temporal = {
            "available": True,
            "value": _mean(temporal_values),
            "count": len(temporal_values),
            "reason": None,
        }
    else:
        temporal = {
            "available": False,
            "value": None,
            "count": 0,
            "reason": "cine protocol declared but no temporal verification metric was supplied",
        }
    return {
        "degenerate_flags": degenerate,
        "ambiguity": ambiguity,
        "stability": stability,
        "ranking_agreement": ranking_agreement,
        "sensitivity": sensitivity,
        "temporal": temporal,
    }


# ---------------------------------------------------------------------------
# coverage and epoch audit summaries
# ---------------------------------------------------------------------------
def _coverage_value(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def _coverage_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
    student_coverage_rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    arms = ("student_no_audit", "student_audited")
    # W6 stores matched predictive scores inside each verification row.  Keep
    # those rows separate from ordinary coverage rows so a caller that already
    # supplied flat support counts cannot have them double-counted, while still
    # allowing the nested score payload to provide support when no flat count
    # was persisted.
    nested_student_rows = [
        row
        for row in student_coverage_rows
        if isinstance(row.get("student_coverage"), Mapping)
    ]

    def _nested_arm_entry(row: Mapping[str, Any], arm: str, field: str) -> Any:
        nested = _mapping(row.get("student_coverage"))
        section = _mapping(nested.get(field))
        return _mapping(section.get(arm))

    def _nested_support(row: Mapping[str, Any], arm: str) -> float | None:
        entry = _nested_arm_entry(row, arm, "natural")
        return _coverage_value(entry, ("supported_pixels", "support_pixels", "count"))

    flat_matched_keys = (
        "matched_support",
        "matched_supported_pixels",
        "matched_coverage_support",
        "matched_support_count",
        "matched_coverage_support_count",
    )

    def _nested_matched_support(row: Mapping[str, Any]) -> float | None:
        nested = _mapping(row.get("student_coverage"))
        return _coverage_value(
            nested,
            ("matched_supported_pixels", "matched_support", "matched_support_count"),
        )

    def _nested_quality() -> dict[str, dict[str, Any]]:
        qualities: dict[str, dict[str, Any]] = {}
        for arm in arms:
            raw_sum = 0.0
            count = 0
            reasons: list[str] = []
            for row in nested_student_rows:
                entry = _nested_arm_entry(row, arm, "matched")
                score = (_mapping(entry.get("score")) or entry) if entry else {}
                info = _score_info(score)
                if info["available"] and info["normalized_nll"] is not None and info["count"] > 0:
                    contribution = info["nll_sum"]
                    if contribution is None:
                        contribution = info["normalized_nll"] * info["count"]
                    raw_sum += contribution
                    count += info["count"]
                elif info["reason"]:
                    reasons.append(str(info["reason"]))
            available = count > 0 and math.isfinite(raw_sum)
            normalized = raw_sum / count if available else None
            qualities[arm] = {
                "value": normalized,
                "nll_sum": raw_sum if available else None,
                "count": count,
                "normalized_nll": normalized,
                "available": available,
                "reason": None
                if available
                else (reasons[0] if reasons else "matched predictive score was not supplied"),
                "support": "matched masks",
            }
        return qualities

    natural: dict[str, Any] = {}
    valid_foreground: dict[str, Any] = {}
    for arm in arms:
        support_values = [
            _coverage_value(
                row,
                (
                    f"natural_support_{arm}",
                    f"natural_support_pixels_{arm}",
                    f"natural_coverage_count_{arm}",
                ),
            )
            for row in rows
        ]
        support_values = [value for value in support_values if value is not None and value >= 0]
        support_values.extend(
            value
            for row in nested_student_rows
            for value in [_nested_support(row, arm)]
            if value is not None
            and value >= 0
            and (
                not any(row is existing for existing in rows)
                or _coverage_value(
                    row,
                    (
                        f"natural_support_{arm}",
                        f"natural_support_pixels_{arm}",
                        f"natural_coverage_count_{arm}",
                    ),
                )
                is None
            )
        )
        pixel_values = [
            _coverage_value(row, ("pixels", "total_pixels", "support_pixels")) for row in rows
        ]
        pixel_values = [value for value in pixel_values if value is not None and value >= 0]
        fraction_values = [
            _coverage_value(
                row,
                (
                    f"natural_coverage_{arm}",
                    f"{arm}_natural_coverage",
                    f"natural_fraction_{arm}",
                ),
            )
            for row in rows
        ]
        fraction_values = [value for value in fraction_values if value is not None and value >= 0]
        support = sum(support_values) if support_values else None
        pixels = sum(pixel_values) if pixel_values else None
        fraction = support / pixels if support is not None and pixels and pixels > 0 else _mean(fraction_values)
        support_available = support is not None
        fraction_available = fraction is not None
        natural[arm] = {
            "support_pixels": int(support) if support_available else None,
            "total_pixels": int(pixels) if pixels is not None else None,
            "fraction": fraction,
            "count": len(support_values),
            "available": support_available or fraction_available,
            "reason": None if support_available or fraction_available else "natural coverage counts were not supplied",
        }
        metric_rows.append(
            _metric(
                f"coverage.{arm}.natural_support_pixels",
                support,
                provenance=provenance,
                unit="pixels",
                population="coverage units",
                count=len(support_values),
                available=support_available,
                reason=None if support_available else "natural coverage support count unavailable",
                arm=arm,
            )
        )
        metric_rows.append(
            _metric(
                f"coverage.{arm}.natural_fraction",
                fraction,
                provenance=provenance,
                unit="fraction",
                population="coverage pixels",
                count=pixels if pixels is not None else len(fraction_values),
                available=fraction_available,
                reason=None if fraction_available else "natural coverage fraction unavailable",
                arm=arm,
            )
        )

        fg_values = [
            _coverage_value(
                row,
                (
                    f"valid_foreground_{arm}",
                    f"valid_foreground_pixels_{arm}",
                    f"valid_fg_{arm}",
                    f"validFG_{arm}",
                ),
            )
            for row in rows
        ]
        fg_values = [value for value in fg_values if value is not None and value >= 0]
        fg = sum(fg_values) if fg_values else None
        fg_available = bool(rows) and len(fg_values) == len(rows)
        collapsed = bool(fg_available and fg == 0)
        valid_foreground[arm] = {
            "pixels": int(fg) if fg is not None and fg_available else None,
            "count": len(fg_values),
            "available": fg_available,
            "collapsed": collapsed if fg_available else None,
            "reason": None if fg_available else "valid foreground support was not recorded for every coverage row",
        }
        metric_rows.append(
            _metric(
                f"coverage.{arm}.valid_foreground_pixels",
                fg if fg_available else None,
                provenance=provenance,
                unit="pixels",
                population="coverage units",
                count=len(fg_values),
                available=fg_available,
                reason=None if fg_available else valid_foreground[arm]["reason"],
                arm=arm,
            )
        )
        metric_rows.append(
            _metric(
                f"coverage.{arm}.valid_fg_collapse",
                1.0 if collapsed else 0.0 if fg_available else None,
                provenance=provenance,
                unit="boolean",
                population="coverage units",
                count=len(fg_values),
                available=fg_available,
                reason=None if fg_available else valid_foreground[arm]["reason"],
                arm=arm,
            )
        )

    matched_values = [
        _coverage_value(
            row,
            flat_matched_keys,
        )
        for row in rows
    ]
    matched_values = [value for value in matched_values if value is not None and value >= 0]
    matched_values.extend(
        value
        for row in nested_student_rows
        for value in [_nested_matched_support(row)]
        if value is not None
        and value >= 0
        and (
            not any(row is existing for existing in rows)
            or _coverage_value(row, flat_matched_keys) is None
        )
    )
    matched_support = sum(matched_values) if matched_values else None
    total_pixels = sum(
        value
        for value in (
            _coverage_value(row, ("pixels", "total_pixels")) for row in rows
        )
        if value is not None and value >= 0
    )
    matched_fraction = matched_support / total_pixels if matched_support is not None and total_pixels > 0 else None
    quality_by_arm = _nested_quality()
    quality_available = any(item["available"] for item in quality_by_arm.values())
    quality_reason = (
        None
        if quality_available
        else "matched support was counted, but no metric was scored on the matched masks"
    )
    quality: dict[str, Any] = {
        # ``value`` remains null because matched quality has one score per
        # student arm.  The actual values live under ``by_arm`` and are also
        # emitted as strict metric rows below.
        "value": None,
        "available": quality_available,
        "reason": quality_reason,
        "by_arm": quality_by_arm,
        "student_no_audit": quality_by_arm["student_no_audit"],
        "student_audited": quality_by_arm["student_audited"],
    }
    audited_quality = quality_by_arm["student_audited"]
    no_audit_quality = quality_by_arm["student_no_audit"]
    if audited_quality["available"] and no_audit_quality["available"]:
        quality["difference_audited_minus_no_audit"] = {
            "value": audited_quality["normalized_nll"] - no_audit_quality["normalized_nll"],
            "normalized_nll": audited_quality["normalized_nll"] - no_audit_quality["normalized_nll"],
            "count": min(audited_quality["count"], no_audit_quality["count"]),
            "available": True,
            "reason": None,
            "support": "matched masks",
        }
    else:
        quality["difference_audited_minus_no_audit"] = {
            "value": None,
            "normalized_nll": None,
            "count": 0,
            "available": False,
            "reason": "both student arms require valid matched verification scores",
            "support": "matched masks",
        }
    matched = {
        "support_pixels": int(matched_support) if matched_support is not None else None,
        "support_count": int(matched_support) if matched_support is not None else None,
        "total_pixels": int(total_pixels) if total_pixels > 0 else None,
        "fraction": matched_fraction,
        "count": len(matched_values),
        "available": matched_support is not None,
        "quality": quality,
        "reason": None if matched_support is not None else "matched coverage support count was not supplied",
    }
    metric_rows.append(
        _metric(
            "coverage.matched_support_pixels",
            matched_support,
            provenance=provenance,
            unit="pixels",
            population="coverage units",
            count=len(matched_values),
            available=matched_support is not None,
            reason=matched["reason"],
        )
    )
    metric_rows.append(
        _metric(
            "coverage.matched_fraction",
            matched_fraction,
            provenance=provenance,
            unit="fraction",
            population="coverage pixels",
            count=total_pixels,
            available=matched_fraction is not None,
            reason=(None if matched_fraction is not None else "matched coverage denominator was not supplied"),
        )
    )
    for arm in arms:
        item = quality_by_arm[arm]
        metric_rows.append(
            _metric(
                f"coverage.matched.{arm}.normalized_nll",
                item["normalized_nll"],
                provenance=provenance,
                unit="nats/pixel",
                population="matched verification pixels",
                count=item["count"],
                available=item["available"],
                reason=item["reason"],
                arm=arm,
                support="matched masks",
            )
        )
    difference = quality["difference_audited_minus_no_audit"]
    metric_rows.append(
        _metric(
            "coverage.matched.student_audited_minus_student_no_audit.normalized_nll",
            difference["normalized_nll"],
            provenance=provenance,
            unit="nats/pixel",
            population="matched verification pixels",
            count=difference["count"],
            available=difference["available"],
            reason=difference["reason"],
            comparison="student_audited - student_no_audit",
            support="matched masks",
        )
    )
    return {
        "rows": [_safe(row) for row in rows],
        "natural_coverage": natural,
        "matched_coverage": matched,
        "valid_foreground": valid_foreground,
        "valid_fg_collapse": {
            arm: valid_foreground[arm]["collapsed"] for arm in arms
        },
        "validFGcollapse": {
            arm: valid_foreground[arm]["collapsed"] for arm in arms
        },
        # Compact aliases make the two coverage populations explicit to readers
        # using the report as a JSON interchange rather than Markdown.
        "naturalcoverage": natural,
        "matchedcoverage": matched,
    }, metric_rows


def _flatten_scalars(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_scalars(item, name)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        return
    if _number(value) is not None:
        yield prefix, value


def _epoch_unit(name: str) -> str:
    lower = name.lower()
    if "nll" in lower:
        return "nats" if any(token in lower for token in ("sum", "raw")) else "nats/pixel"
    if any(token in lower for token in ("coverage", "rate", "fraction", "occupancy", "ramp")):
        return "fraction"
    if "loss" in lower:
        return "loss"
    if any(token in lower for token in ("second", "latency", "time")):
        return "seconds"
    if any(token in lower for token in ("step", "batch", "unit", "pixel", "candidate", "count")):
        return "count"
    if lower.endswith("lr") or ".lr" in lower:
        return "optimizer"
    return "scalar"


def _epoch_audit_rows(
    history: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    epoch: int | None,
    checkpoint: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in history:
        record_epoch = _first(record, "global_epoch", "epoch")
        provenance = _provenance(
            {**record, "epoch": record_epoch if record_epoch is not None else epoch},
            manifest,
            epoch,
            checkpoint,
        )
        units = _nonnegative_count(_first(record, "units_visited", "count"), 1)
        for name, value in _flatten_scalars(record):
            lower = name.lower()
            if not name or lower in {
                "global_epoch",
                "epoch",
                "units_visited",
                "units_available",
                "count",
            }:
                continue
            if lower.endswith((".available", ".count")):
                continue
            rows.append(
                _metric(
                    f"epoch_audit.{name}",
                    value,
                    provenance=provenance,
                    unit=_epoch_unit(name),
                    population="observation units",
                    count=units,
                    available=True,
                )
            )
    return rows


def _aggregate_provenance(
    base: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Keep aggregate provenance truthful when rows span several splits.

    A method summary is often assembled from multiple units.  Copying the
    first row's provenance onto that summary silently turns an all-split
    aggregate into (for example) a ``train`` result.  Preserve the compact
    single-split label only when every contributing row agrees; otherwise
    explicitly retain ``mixed`` and the complete split list.
    """
    result = dict(base)
    split_values = sorted(
        {
            str(_mapping(record.get("provenance")).get("split"))
            for record in records
            if _mapping(record.get("provenance")).get("split") is not None
        }
    )
    if len(split_values) == 1:
        result["split"] = split_values[0]
        result["splits"] = split_values
    elif len(split_values) > 1:
        result["split"] = "mixed"
        result["splits"] = split_values
    return result


# ---------------------------------------------------------------------------
# report writer implementation
# ---------------------------------------------------------------------------
def _write_image_only_reports_single(
    run_root: str | Path,
    manifest: Mapping[str, Any] | str | Path,
    epoch_history: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    verification_rows: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    experiment_rows: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    coverage_rows: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    epoch: int | None,
    checkpoint: str | Path,
) -> dict[str, Path]:
    """Write the bounded image-only JSON/CSV/Markdown report set.

    Parameters are intentionally row-oriented and generic so the trainer can
    call this after ``freeze_predictions`` and image-only verification without
    passing models or tensors.  ``manifest`` and any row collection may also be
    a path to the JSON/JSONL artifact the trainer already wrote.

    The JSON report contains the nested summaries.  CSV contains every strict
    metric row, including unavailable rows.  Markdown contains the same rows
    plus compact method, coverage, and diagnostic tables.  No reference metric
    is inferred when a matched mask was only counted and never scored.
    """
    root = Path(run_root)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_map = _mapping(manifest)
    history = _rows(epoch_history, "history", "epoch_metrics", "metrics", "rows")
    verification = _rows(verification_rows, "verification_rows", "verification", "rows")
    experiments = _rows(experiment_rows, "experiment_rows", "experiments", "rows")
    coverage_rows_list = _rows(coverage_rows, "coverage_rows", "coverage", "rows")
    first_row = next(iter([*verification, *experiments, *coverage_rows_list]), {})
    dataset = str(_first(manifest_map, "dataset") or _first(first_row, "dataset") or "unknown")
    protocol = str(
        _first(manifest_map, "resolved_protocol", "protocol")
        or _first(first_row, "protocol")
        or "unknown"
    )

    splits = sorted(
        {
            str(value)
            for row in [*verification, *experiments, *coverage_rows_list]
            for value in [_first(row, "split")]
            if value is not None and str(value) not in {"mixed", "unknown"}
        }
        | (
            {str(manifest_map["split"])}
            if manifest_map.get("split") is not None
            and str(manifest_map["split"]) not in {"mixed", "unknown"}
            else set()
        )
    )
    split = splits[0] if len(splits) == 1 else "mixed" if splits else "unknown"
    provenance = {
        "dataset": dataset,
        "split": split,
        "splits": splits,
        "protocol": protocol,
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "population": "image-only observation units",
        "count": len(verification),
        "available": bool(verification or experiments or coverage_rows_list or history),
        "unit": "observation unit",
        "contract_version": VERSION,
    }
    if not provenance["available"]:
        provenance["reason"] = "no image-only rows were supplied"
    else:
        provenance["reason"] = None

    records, maps, candidate_rows = _verification_records(
        verification, manifest_map, epoch, checkpoint
    )
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_method[str(record["method"])].append(record)
    method_summaries: dict[str, Any] = {}
    method_values: dict[str, Mapping[str, float]] = {}
    verification_metric_rows = list(candidate_rows)
    for method, method_records in sorted(by_method.items()):
        aggregate = _aggregate_method(method_records)
        method_provenance = _aggregate_provenance(provenance, method_records)
        method_summaries[method] = {
            "candidate_ids_by_unit": {
                str(record["unit_id"]): record.get("candidate_id") for record in method_records
            },
            "provenance": method_provenance,
            **_safe(aggregate),
        }
        patient_values = aggregate["patient_values"]
        method_values[method] = patient_values
        verification_metric_rows.extend(
            [
                _metric(
                    f"verification.method.{method}.patient_macro_normalized_nll",
                    aggregate["patient_macro_normalized_nll"],
                    provenance=method_provenance,
                    unit="nats/pixel",
                    population="patients",
                    count=aggregate["patient_count"],
                    available=aggregate["patient_macro_normalized_nll"] is not None,
                    reason=aggregate["reason"],
                    method=method,
                    aggregation="mean per unit, then mean per patient",
                ),
                _metric(
                    f"verification.method.{method}.raw_pooled_normalized_nll",
                    aggregate["raw_pooled_normalized_nll"],
                    provenance=method_provenance,
                    unit="nats/pixel",
                    population="observation pixels",
                    count=aggregate["raw_count"],
                    available=aggregate["raw_pooled_normalized_nll"] is not None,
                    reason=(None if aggregate["raw_pooled_normalized_nll"] is not None else aggregate["reason"]),
                    method=method,
                    aggregation="sum raw NLL / sum observed-pixel count",
                ),
                _metric(
                    f"verification.method.{method}.patient_macro_selection_gap",
                    aggregate["patient_macro_gap"],
                    provenance=method_provenance,
                    unit="nats/pixel",
                    population="patients",
                    count=len(aggregate["patient_gap_values"]),
                    available=aggregate["patient_macro_gap"] is not None,
                    reason=(
                        None
                        if aggregate["patient_macro_gap"] is not None
                        else "selection-to-verification gaps are unavailable for the patient set"
                    ),
                    method=method,
                ),
            ]
        )
    paired_results, paired_rows = _paired_bootstrap_rows(method_values, provenance)
    verification_metric_rows.extend(paired_rows)
    diagnostics = _verification_diagnostics(verification, protocol=protocol)
    verification_metric_rows.extend(
        _stability_metric_rows(diagnostics["stability"], provenance=provenance)
    )

    experiment_metric_rows = [
        _normalise_input_metric_row(
            row,
            manifest=manifest_map,
            epoch=epoch,
            checkpoint=checkpoint,
            prefix="experiment",
        )
        for row in experiments
    ]
    epoch_metric_rows = _epoch_audit_rows(
        history, manifest=manifest_map, epoch=epoch, checkpoint=checkpoint
    )
    coverage_summary, coverage_metric_rows = _coverage_summary(
        coverage_rows_list,
        provenance=provenance,
        student_coverage_rows=verification,
    )
    all_metric_rows = [
        *epoch_metric_rows,
        *experiment_metric_rows,
        *verification_metric_rows,
        *coverage_metric_rows,
    ]
    temporal = diagnostics["temporal"]
    temporal_metric_row = _metric(
        "verification.temporal_predictive_metric",
        temporal.get("value"),
        provenance=provenance,
        unit="nats/pixel",
        population="temporal verification observations",
        count=temporal.get("count", 0),
        available=bool(temporal.get("available")),
        reason=temporal.get("reason"),
    )
    all_metric_rows.append(temporal_metric_row)

    method_to_candidate: Any
    if len(maps) == 1:
        # The common one-unit case has the compact contract requested by the
        # trainer: method -> candidate id.
        method_to_candidate = next(iter(maps.values()))
    else:
        # Candidate ids are unit-local.  Keeping methods as the outer keys
        # avoids pretending that ``E5`` has one global candidate across a run.
        method_to_candidate = {
            method: {
                unit_id: mapping.get(method)
                for unit_id, mapping in sorted(maps.items())
                if method in mapping
            }
            for method in sorted({method for mapping in maps.values() for method in mapping})
        }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract_version": VERSION,
        "report_type": "image_only",
        "provenance": provenance,
        "dataset": dataset,
        "split": split,
        "protocol": protocol,
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "population": provenance["population"],
        "count": provenance["count"],
        "available": provenance["available"],
        "reason": provenance["reason"],
        "method_to_candidate": _safe(method_to_candidate),
        "method_to_candidate_by_unit": _safe(maps),
        "verification": {
            "candidates": _candidate_payload(verification),
            "method_summaries": method_summaries,
            "patient_macro": {
                method: summary["patient_macro_normalized_nll"]
                for method, summary in method_summaries.items()
            },
            "raw_pooled": {
                method: summary["raw_pooled_normalized_nll"]
                for method, summary in method_summaries.items()
            },
            "paired_patient_bootstrap": paired_results,
            "diagnostics": _safe(diagnostics),
            "rows": _safe(verification),
            "metric_rows": _safe(verification_metric_rows),
        },
        "experiments": {
            "rows": _safe(experiments),
            "metric_rows": _safe(experiment_metric_rows),
        },
        "coverage": coverage_summary,
        "epoch_audit": {
            "rows": _safe(history),
            "metric_rows": _safe(epoch_metric_rows),
            "available": bool(epoch_metric_rows),
            "count": len(history),
            "reason": None if epoch_metric_rows else "no finite epoch audit metric was supplied",
        },
        "rows": _safe(all_metric_rows),
        "metric_rows": _safe(all_metric_rows),
        "scores_are_appearance_not_segmentation_accuracy": True,
        "manual_reference_metrics": {
            "available": False,
            "reason": "this writer emits image-only reports; reference evaluation is a separate post-freeze process",
        },
        "notes": [
            "Predictive NLL and evidence margins describe appearance observations; they are not segmentation accuracy or correctness probabilities.",
            "Patient macro metrics average per-unit values within each patient before averaging patients; raw pooled metrics use summed NLL and observed-pixel counts.",
            "Natural and matched support are reported separately. Matched support has no quality value unless a metric was explicitly scored on the matched masks.",
            "Unavailable values remain null with an explicit reason; no unknown metric is converted to zero.",
        ],
    }

    json_path = write_json(reports_dir / REPORT_FILENAMES["json"], report)
    csv_path = write_csv(reports_dir / REPORT_FILENAMES["csv"], all_metric_rows)
    md_path = write_markdown(
        reports_dir / REPORT_FILENAMES["md"],
        f"Image-only maskfree150 summary - {dataset} ({split})",
        all_metric_rows,
        notes=report["notes"]
        + [
            f"Resolved protocol: {protocol}.",
            f"Temporal status: {temporal.get('reason') if not temporal.get('available') else 'available'}.",
        ],
    )
    # The shared writer gives a stable strict-metric table. Add compact
    # structural sections without duplicating every bounded input row.
    with md_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Method to candidate\n\n")
        handle.write("| Unit | Method | Candidate |\n|---|---|---|\n")
        for unit_id, mapping in sorted(maps.items()):
            for method, candidate_id in sorted(mapping.items()):
                handle.write(
                    f"| {unit_id} | {method} | {candidate_id if candidate_id is not None else 'unavailable'} |\n"
                )
        if not maps:
            handle.write("| unavailable | unavailable | unavailable |\n")
        handle.write("\n## Coverage\n\n")
        for arm, summary in coverage_summary["natural_coverage"].items():
            handle.write(
                f"- `{arm}` natural support: {summary.get('support_pixels')}; "
                f"natural fraction: {summary.get('fraction')}; count: {summary.get('count')}.\n"
            )
        matched = coverage_summary["matched_coverage"]
        matched_quality = matched["quality"]
        if matched_quality.get("available"):
            quality_parts = []
            for arm in ("student_no_audit", "student_audited"):
                item = matched_quality.get(arm, {})
                if item.get("available"):
                    quality_parts.append(f"{arm}={item.get('normalized_nll')}")
            quality_text = ", ".join(quality_parts) or "available"
        else:
            quality_text = f"unavailable ({matched_quality.get('reason')})"
        handle.write(
            f"- matched support: {matched.get('support_pixels')}; "
            f"matched fraction: {matched.get('fraction')}; matched quality: {quality_text}.\n"
        )
        handle.write("\n## Image-only diagnostics\n\n")
        for name in (
            "degenerate_flags",
            "ambiguity",
            "stability",
            "ranking_agreement",
            "sensitivity",
            "temporal",
        ):
            diagnostic = diagnostics[name]
            handle.write(
                f"- `{name}`: available={diagnostic.get('available')}; "
                f"reason={diagnostic.get('reason') or 'none'}.\n"
            )
        stability = diagnostics["stability"]
        if stability.get("methods"):
            handle.write("\n### Repeated-annotation stability\n\n")
            handle.write("| Method | Agreement | Valid agreement | Common valid count |\n|---|---:|---:|---:|\n")
            for method, item in sorted(stability["methods"].items()):
                handle.write(
                    f"| {method} | {item.get('agreement')} | {item.get('valid_agreement')} | "
                    f"{item.get('common_valid_count')} |\n"
                )
            handle.write(
                f"\nRepeat protocol: `{stability.get('protocol')}`; scope: "
                f"`{stability.get('scope')}`.\n"
            )
        handle.write(
            "\n`valid_fg_collapse` is a support diagnostic only; it does not establish label quality.\n"
        )

    return {"json": json_path, "csv": csv_path, "md": md_path}


def _manifest_and_row_splits(
    manifest: Mapping[str, Any],
    row_sets: Sequence[Sequence[Mapping[str, Any]]],
) -> list[str]:
    """Collect declared split names without guessing a row's split."""
    values: set[str] = set()
    declared = manifest.get("split")
    if declared is not None and str(declared) not in {"mixed", "unknown"}:
        values.add(str(declared))
    # Dataset manifests in the trainer have used ``records`` and ``units``
    # over time; freeze manifests may instead expose ``predictions``.  Read
    # only their shallow record entries so split discovery remains bounded.
    for key in ("records", "units", "entries", "predictions", "rows"):
        nested = manifest.get(key)
        if isinstance(nested, Mapping):
            nested_values = nested.values()
        elif isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            nested_values = nested
        else:
            nested_values = ()
        for item in nested_values:
            item_map = _mapping(item)
            split = _first(item_map, "split")
            if split is not None and str(split) not in {"mixed", "unknown"}:
                values.add(str(split))
    for rows in row_sets:
        for row in rows:
            split = _first(row, "split")
            if split is not None and str(split) not in {"mixed", "unknown"}:
                values.add(str(split))
    return sorted(values)


def _rows_for_split(
    rows: Sequence[Mapping[str, Any]], split: str, *, all_splits: Sequence[str]
) -> list[dict[str, Any]]:
    """Select explicitly labelled rows for one split.

    An unlabelled row can be inherited only when the complete input declares a
    single split.  With multiple splits it is left out of per-split reports so
    it cannot be silently attributed to train or held-out test.
    """
    selected: list[dict[str, Any]] = []
    single = len(all_splits) == 1
    for row in rows:
        row_split = _first(row, "split")
        if row_split is None:
            if single and str(all_splits[0]) == split:
                selected.append(dict(row))
        elif str(row_split) == split:
            selected.append(dict(row))
    return selected


def write_image_only_reports(
    run_root: str | Path,
    manifest: Mapping[str, Any] | str | Path,
    epoch_history: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    verification_rows: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    experiment_rows: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    coverage_rows: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    epoch: int | None,
    checkpoint: str | Path,
) -> dict[str, Path]:
    """Write an all-split index plus one report set for every declared split.

    The returned mapping keeps the stable ``json``/``csv``/``md`` contract and
    points at the all-split report.  Its JSON ``split_reports`` index links the
    split-specific report sets below ``run_root/splits/<split>/reports``.
    """
    root = Path(run_root)
    manifest_map = _mapping(manifest)
    history = _rows(epoch_history, "history", "epoch_metrics", "metrics", "rows")
    verification = _rows(verification_rows, "verification_rows", "verification", "rows")
    experiments = _rows(experiment_rows, "experiment_rows", "experiments", "rows")
    coverage = _rows(coverage_rows, "coverage_rows", "coverage", "rows")
    all_splits = _manifest_and_row_splits(
        manifest_map, (history, verification, experiments, coverage)
    )

    overall_manifest = dict(manifest_map)
    if len(all_splits) > 1:
        overall_manifest["split"] = "mixed"
    elif len(all_splits) == 1:
        overall_manifest["split"] = all_splits[0]
    overall_paths = _write_image_only_reports_single(
        root,
        overall_manifest,
        history,
        verification,
        experiments,
        coverage,
        epoch,
        checkpoint,
    )

    split_reports: dict[str, dict[str, str]] = {}
    for split in all_splits:
        split_manifest = dict(manifest_map)
        split_manifest["split"] = split
        split_paths = _write_image_only_reports_single(
            root / "splits" / split,
            split_manifest,
            history,
            _rows_for_split(verification, split, all_splits=all_splits),
            _rows_for_split(experiments, split, all_splits=all_splits),
            _rows_for_split(coverage, split, all_splits=all_splits),
            epoch,
            checkpoint,
        )
        split_reports[split] = {key: str(path) for key, path in split_paths.items()}

    if split_reports:
        report_payload = json.loads(overall_paths["json"].read_text(encoding="utf-8"))
        report_payload["split_reports"] = split_reports
        report_payload["split_report_layout"] = "run_root/splits/<split>/reports"
        write_json(overall_paths["json"], report_payload)
        with overall_paths["md"].open("a", encoding="utf-8") as handle:
            handle.write("\n## Split reports\n\n")
            for split, paths in split_reports.items():
                handle.write(
                    f"- `{split}`: JSON `{paths['json']}`; CSV `{paths['csv']}`; Markdown `{paths['md']}`.\n"
                )

    return overall_paths


__all__ = [
    "REPORT_FILENAMES",
    "REPORT_SCHEMA_VERSION",
    "ReportingContractError",
    "write_image_only_reports",
]
