#!/usr/bin/env python3
"""Aggregate observed Candidate A-3D campaign artifacts.

The reporter is deliberately read-only with respect to experiment outputs.  It
reads the durable supervisor state and the final validation receipts, keeps
missing or failed jobs as such, and emits JSON, CSV and Markdown summaries.
Undefined metrics remain ``null``; no score is inferred from a checkpoint,
parameter count or a synthetic fixture.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_a_offload import OFFLOAD_RECEIPT_NAME, validate_offload_receipt
from self_audit.model import digest

SCHEMA = "self_audit.candidate_a3d.campaign_report.v1"
# Metrics/evaluation use the declared four-class display names.  The data
# loader's CLASS_MAPPING is an integer-keyed manifest mapping (and also
# contains the ignore value 255), so it is not suitable as this report's class
# list.
CLASS_NAMES = ("BG", "RV", "MYO", "LV")
FOREGROUND = tuple(name for name in CLASS_NAMES if name != "BG")
METRICS = ("dice", "f1", "precision", "recall")
BOUNDARY_METRICS = ("surface_dice", "assd_mm", "hd95_mm")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON object expected: {path}")
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _finite(value)) is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _path_hash(path: Path) -> str | None:
    return digest(path) if path.is_file() else None


def _child_plan(job: Mapping[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    key = str(job.get("key", ""))
    if "::" not in key:
        return None, None
    child_path = Path(key.split("::", 1)[0]).expanduser().resolve()
    if not child_path.is_file():
        return child_path, None
    try:
        return child_path, _read_json(child_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return child_path, None


def _job_plan_record(job: Mapping[str, Any], child: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if child is None:
        return None
    wanted = str(job.get("key", "")).split("::", 1)[-1]
    for item in child.get("jobs", []):
        if isinstance(item, Mapping) and item.get("id") == wanted:
            return dict(item)
    return None


def _validation_payload(out: Path, report: Mapping[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    """Read only the declared final validation epoch for this job."""

    epoch = report.get("epochs")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        return None, None
    validation = out / "validation" / f"epoch_{epoch:04d}"
    candidates = (validation / "EPOCH_METRICS.json", validation / "TEACHER_EPOCH_METRICS.json")
    for candidate in candidates:
        if candidate.is_file():
            try:
                return _read_json(candidate), candidate
            except (OSError, ValueError, json.JSONDecodeError):
                return None, candidate
    return None, None


def _branch_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one student or teacher validation result without filling gaps."""

    pooled = payload.get("per_class")
    patient = payload.get("macro_patient")
    if not isinstance(pooled, Mapping):
        pooled = {}
    if not isinstance(patient, Mapping):
        patient = {}
    # Validation metrics use patient macro scores for the headline and retain
    # pooled voxel scores separately.  Teacher validation has the same schema.
    headline = patient if patient else pooled
    result: dict[str, Any] = {
        "pooled_per_class": {name: dict(pooled[name]) for name in CLASS_NAMES
                             if isinstance(pooled.get(name), Mapping)},
        "macro_patient_per_class": {name: dict(patient[name]) for name in CLASS_NAMES
                                     if isinstance(patient.get(name), Mapping)},
        "per_class": {},
        "foreground_macro": {},
    }
    for name in CLASS_NAMES:
        values = headline.get(name)
        values = values if isinstance(values, Mapping) else {}
        result["per_class"][name] = {metric: _finite(values.get(metric)) for metric in METRICS}
    for metric in METRICS:
        result["foreground_macro"][metric] = _mean(
            result["per_class"][name].get(metric) for name in FOREGROUND
        )
    boundary = payload.get("boundary")
    if not isinstance(boundary, Mapping):
        boundary = {}
    result["boundary"] = {
        str(name): {metric: _finite(values.get(metric)) for metric in BOUNDARY_METRICS}
        for name, values in boundary.items()
        if isinstance(values, Mapping)
    }
    result["source"] = "macro_patient" if patient else ("pooled_voxels" if pooled else None)
    return result


def _extract_metrics(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "branches": {}, "primary": None, "validation": "MISSING", "epoch": None,
            "boundary": {}, "code": {"packets": None, "total_bytes": None,
                                       "mean_bytes": None, "query_count_mean": None,
                                       "query_count_min": None, "query_count_max": None},
            "profile": {"total_ms": None, "mean_ms": None,
                        "student_rollout_total_ms": None,
                        "student_rollout_mean_ms": None,
                        "timing_scope": None,
                        "peak_vram_allocated_bytes": None,
                        "peak_vram_reserved_bytes": None},
        }
    branches: dict[str, Any] = {}
    if isinstance(payload.get("branches"), Mapping):
        for name, branch in payload["branches"].items():
            if isinstance(branch, Mapping):
                branches[str(name)] = _branch_summary(branch)
    elif "per_class" in payload or "macro_patient" in payload:
        branches["student"] = _branch_summary(payload)
    primary = branches.get("student")
    if primary is None and branches:
        # Teacher stages have no student branch; preserve both branches and use
        # region as a clearly named headline when it is available.
        primary = branches.get("region") or next(iter(branches.values()))
    profile = payload.get("profile") if isinstance(payload.get("profile"), Mapping) else {}
    profile_frames = profile.get("frames") if isinstance(profile.get("frames"), list) else []
    rollout_ms = [frame.get("rollout_ms") for frame in profile_frames
                  if isinstance(frame, Mapping) and _finite(frame.get("rollout_ms")) is not None]
    code = payload.get("code") if isinstance(payload.get("code"), Mapping) else {}
    query_counts = code.get("query_counts")
    query_numbers = [number for value in query_counts or [] if (number := _finite(value)) is not None]
    epoch = payload.get("epoch")
    return {
        "branches": branches,
        "primary": primary,
        "validation": "PASS" if branches else "INVALID_OR_EMPTY",
        "epoch": epoch if isinstance(epoch, int) else None,
        "boundary": primary.get("boundary", {}) if isinstance(primary, Mapping) else {},
        "code": {
            "packets": code.get("packets"),
            "total_bytes": code.get("total_bytes"),
            "mean_bytes": code.get("mean_bytes"),
            "query_count_mean": _mean(query_numbers),
            "query_count_min": min(query_numbers) if query_numbers else None,
            "query_count_max": max(query_numbers) if query_numbers else None,
        },
        "profile": {
            # ``rollout_ms`` is the student-only forward/policy/decoder path.
            # The raw profile total also includes packet serialization and
            # roundtrip checks, so it is intentionally not exposed as
            # student-only latency.
            "student_rollout_total_ms": sum(float(value) for value in rollout_ms) if rollout_ms else None,
            "student_rollout_mean_ms": _mean(rollout_ms),
            "timing_scope": profile.get("timing_scope"),
            "peak_vram_allocated_bytes": _finite(profile.get("peak_vram_allocated_bytes")),
            "peak_vram_reserved_bytes": _finite(profile.get("peak_vram_reserved_bytes")),
        },
    }


def _epoch_summary(out: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    epoch = report.get("epochs")
    if not isinstance(epoch, int):
        return {}
    path = out / "epochs.jsonl"
    if not path.is_file():
        return {}
    selected: dict[str, Any] = {}
    try:
        for line in path.read_text().splitlines():
            value = json.loads(line)
            if isinstance(value, Mapping) and value.get("epoch") == epoch:
                selected = dict(value)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return selected


def _artifact_status(
    manifest: Mapping[str, Any] | None,
    report: Mapping[str, Any] | None,
    manifest_path: Path,
    payload_path: Path,
    *,
    state_job: Mapping[str, Any] | None = None,
    plan: Mapping[str, Any] | None = None,
) -> str:
    if manifest is None or report is None or not manifest_path.is_file():
        return "MISSING_OR_INVALID"
    if report.get("status") != "COMPLETED":
        return "INVALID"
    expected_payload = manifest.get("payload_sha256")
    expected_manifest = report.get("final_manifest_sha256")
    if expected_manifest is not None and expected_manifest != _path_hash(manifest_path):
        return "INVALID"
    if payload_path.is_file():
        return "PASS" if isinstance(expected_payload, str) and _path_hash(payload_path) == expected_payload else "INVALID"
    if (state_job is None or plan is None or state_job.get("stage") != "joint"):
        return "MISSING_OR_INVALID"
    if report.get("stage") != "joint":
        return "INVALID"
    expected = {
        "job_key": state_job.get("key"), "dataset": state_job.get("dataset"),
        "k": state_job.get("k"), "arm": state_job.get("arm"), "stage": "joint",
        "config_sha256": state_job.get("config_sha256"),
        "source_sha256": manifest.get("source_sha256"),
        "data_sha256": plan.get("train_sha256"),
        "manifest_sha256": _path_hash(manifest_path),
        "report_sha256": _path_hash(payload_path.parent.parent / "REPORT.json"),
        "payload_sha256": expected_payload,
    }
    valid, _reason = validate_offload_receipt(
        payload_path.parent / OFFLOAD_RECEIPT_NAME, expected=expected, verify_local=False)
    if valid:
        return "OFFLOADED"
    return "MISSING_OR_INVALID"


def _delta(current: Mapping[str, Any], full: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if full is None:
        return None
    result: dict[str, Any] = {}
    for metric in METRICS:
        left = _finite(current.get("foreground_macro", {}).get(metric))
        right = _finite(full.get("foreground_macro", {}).get(metric))
        result[f"foreground_macro_{metric}"] = left - right if left is not None and right is not None else None
    result["per_class"] = {}
    for name in CLASS_NAMES:
        result["per_class"][name] = {}
        for metric in METRICS:
            left = _finite(current.get("per_class", {}).get(name, {}).get(metric))
            right = _finite(full.get("per_class", {}).get(name, {}).get(metric))
            result["per_class"][name][metric] = left - right if left is not None and right is not None else None
    return result


def _row(state_job: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    out = Path(str(state_job.get("out", ""))).expanduser().resolve()
    report_path = out / "REPORT.json"
    final_manifest = out / "final" / "MANIFEST.json"
    payload_path = out / "final" / "weights.pt"
    report: dict[str, Any] | None = None
    if report_path.is_file():
        try:
            report = _read_json(report_path)
        except (OSError, ValueError, json.JSONDecodeError):
            report = None
    manifest: dict[str, Any] | None = None
    if final_manifest.is_file():
        try:
            manifest = _read_json(final_manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            manifest = None
    child_path, child = _child_plan(state_job)
    plan_job = _job_plan_record(state_job, child)
    validation, validation_path = _validation_payload(out, report or {})
    metrics = _extract_metrics(validation)
    primary = metrics.get("primary") or {}
    epoch = _epoch_summary(out, report or {})
    source_hash = manifest.get("source_sha256") if manifest else None
    data_hash = (child or {}).get("train_sha256")
    dev_hash = (child or {}).get("development_sha256")
    checkpoint_hash = _path_hash(final_manifest)
    payload_hash = manifest.get("payload_sha256") if manifest else None
    if payload_hash is None and payload_path.is_file():
        payload_hash = _path_hash(payload_path)
    offload_path = final_manifest.parent / OFFLOAD_RECEIPT_NAME
    offload_receipt: dict[str, Any] | None = None
    if offload_path.is_file():
        try:
            offload_receipt = _read_json(offload_path)
        except (OSError, ValueError, json.JSONDecodeError):
            offload_receipt = None
    artifact_status = _artifact_status(
        manifest, report, final_manifest, payload_path,
        state_job=state_job, plan=child,
    )
    row: dict[str, Any] = {
        "key": state_job.get("key"),
        "dataset": state_job.get("dataset"),
        "k": state_job.get("k"),
        "seed": (child or {}).get("training_seeds", [None])[0],
        "arm": state_job.get("arm"),
        "stage": state_job.get("stage"),
        "phase": state_job.get("phase"),
        "status": state_job.get("status"),
        "synthetic": (manifest.get("synthetic") if manifest else (child or {}).get("synthetic")),
        "artifact_status": artifact_status,
        "out": str(out),
        "config_path": state_job.get("config"),
        "config_sha256": state_job.get("config_sha256"),
        "data_sha256": data_hash,
        "development_sha256": dev_hash,
        "source_sha256": source_hash,
        "checkpoint_manifest_sha256": checkpoint_hash,
        "checkpoint_payload_sha256": payload_hash,
        "offload_receipt_path": str(offload_path) if offload_path.is_file() else None,
        "offload_receipt_sha256": _path_hash(offload_path),
        "offload_local_backup_path": (offload_receipt or {}).get("local_backup_path"),
        "offload_local_payload_sha256": (offload_receipt or {}).get("local_backup_sha256"),
        "validation_epoch": metrics.get("epoch"),
        "validation_path": str(validation_path) if validation_path else None,
        "validation_status": metrics.get("validation"),
        "metrics": metrics,
        "train_elapsed_seconds": _finite(report.get("elapsed_seconds")) if report else None,
        "student_only_rollout_total_ms": metrics.get("profile", {}).get("student_rollout_total_ms"),
        "student_only_rollout_mean_ms": metrics.get("profile", {}).get("student_rollout_mean_ms"),
        "peak_vram_allocated_bytes": metrics.get("profile", {}).get("peak_vram_allocated_bytes"),
        "peak_vram_reserved_bytes": metrics.get("profile", {}).get("peak_vram_reserved_bytes"),
        "code_total_bytes": metrics.get("code", {}).get("total_bytes"),
        "code_mean_bytes": metrics.get("code", {}).get("mean_bytes"),
        "code_packets": metrics.get("code", {}).get("packets"),
        "query_count_mean": metrics.get("code", {}).get("query_count_mean"),
        "training_rollout_steps_mean": _finite(epoch.get("mean_rollout_steps")),
        "error": state_job.get("error"),
    }
    # Keep epoch values as observed even when validation did not expose code
    # packets; do not turn mean code bits into bytes.
    if row["query_count_mean"] is None:
        row["query_count_mean"] = _finite(epoch.get("mean_query_count"))
    if row["code_total_bytes"] is None:
        row["code_total_bytes"] = None
    row["_primary"] = primary
    row["_child_path"] = str(child_path) if child_path else None
    row["_plan_job"] = plan_job
    return row


def _attach_deltas(rows: list[dict[str, Any]]) -> None:
    full: dict[tuple[Any, Any, Any], Mapping[str, Any]] = {}
    for row in rows:
        key = (row.get("dataset"), row.get("k"), row.get("seed"))
        if row.get("arm") == "full" and row.get("stage") == "joint":
            full[key] = row.get("_primary", {})
    for row in rows:
        key = (row.get("dataset"), row.get("k"), row.get("seed"))
        if row.get("stage") == "joint" and row.get("arm") != "full":
            row["delta_vs_full"] = _delta(row.get("_primary", {}), full.get(key))
        else:
            row["delta_vs_full"] = None


def _clean_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


CSV_FIELDS = (
    "key", "dataset", "k", "seed", "arm", "stage", "phase", "status", "synthetic",
    "artifact_status", "config_sha256", "data_sha256", "development_sha256", "source_sha256",
    "checkpoint_manifest_sha256", "checkpoint_payload_sha256", "validation_epoch", "validation_status",
    "offload_receipt_sha256", "offload_local_backup_path", "offload_local_payload_sha256",
    "train_elapsed_seconds", "student_only_rollout_total_ms", "student_only_rollout_mean_ms",
    "peak_vram_allocated_bytes",
    "peak_vram_reserved_bytes", "code_total_bytes", "code_mean_bytes", "code_packets",
    "query_count_mean", "training_rollout_steps_mean", "foreground_macro_dice", "foreground_macro_f1",
    "foreground_macro_precision", "foreground_macro_recall", "error",
)

CSV_FIELDS = CSV_FIELDS + tuple(
    f"{metric}_{name}" for name in CLASS_NAMES for metric in METRICS
) + tuple(
    f"boundary_{name}_{metric}" for name in FOREGROUND for metric in BOUNDARY_METRICS
) + tuple(f"delta_foreground_macro_{metric}" for metric in METRICS)


def _csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = {field: row.get(field) for field in CSV_FIELDS}
    primary = row.get("metrics", {}).get("primary", {}) if isinstance(row.get("metrics"), Mapping) else {}
    foreground = primary.get("foreground_macro", {}) if isinstance(primary, Mapping) else {}
    for metric in METRICS:
        output[f"foreground_macro_{metric}"] = foreground.get(metric)
    per_class = primary.get("per_class", {}) if isinstance(primary, Mapping) else {}
    for name in CLASS_NAMES:
        values = per_class.get(name, {}) if isinstance(per_class, Mapping) else {}
        for metric in METRICS:
            output[f"{metric}_{name}"] = values.get(metric)
    boundary = primary.get("boundary", {}) if isinstance(primary, Mapping) else {}
    for name in FOREGROUND:
        values = boundary.get(name, {}) if isinstance(boundary, Mapping) else {}
        for metric in BOUNDARY_METRICS:
            output[f"boundary_{name}_{metric}"] = values.get(metric)
    delta = row.get("delta_vs_full")
    for metric in METRICS:
        output[f"delta_foreground_macro_{metric}"] = (
            delta.get(f"foreground_macro_{metric}") if isinstance(delta, Mapping) else None
        )
    return output


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Candidate A-3D campaign report", "",
        f"State: `{report.get('state_path')}`  ",
        f"Status: `{report.get('campaign_status')}`; jobs: `{report.get('total_jobs')}`", "",
        "Metrics are copied from observed final validation receipts. `null` means the artifact was absent or the metric was undefined. Foreground macro excludes BG.",
        "",
        "| dataset | k | seed | arm | stage | status | artifact | FG Dice | FG F1 | precision | recall | code bytes | queries | rollout steps | train s | inference ms |",
        "|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows", []):
        primary = row.get("metrics", {}).get("primary", {}) if isinstance(row.get("metrics"), Mapping) else {}
        fg = primary.get("foreground_macro", {}) if isinstance(primary, Mapping) else {}
        values = [
            row.get("dataset"), row.get("k"), row.get("seed"), row.get("arm"), row.get("stage"),
            row.get("status"), row.get("artifact_status"), fg.get("dice"), fg.get("f1"),
            fg.get("precision"), fg.get("recall"), row.get("code_total_bytes"),
            row.get("query_count_mean"), row.get("training_rollout_steps_mean"), row.get("train_elapsed_seconds"),
            row.get("student_only_rollout_mean_ms"),
        ]
        lines.append("| " + " | ".join("" if value is None else str(value) for value in values) + " |")
    lines.extend(["", "No variance estimate is reported: this campaign uses one seed.", ""])
    return "\n".join(lines)


def build_report(state_path: str | Path) -> dict[str, Any]:
    state_file = Path(state_path).expanduser().resolve()
    state = _read_json(state_file)
    if not isinstance(state.get("jobs"), list):
        raise TypeError("campaign state must contain a jobs list")
    rows = [_row(job, state) for job in state["jobs"] if isinstance(job, Mapping)]
    _attach_deltas(rows)
    clean = [_clean_row(row) for row in rows]
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state_path": str(state_file),
        "state_sha256": digest(state_file),
        "campaign_status": state.get("status"),
        "total_jobs": len(clean),
        "status_counts": dict(Counter(row.get("status") for row in clean)),
        "synthetic_rows": sum(bool(row.get("synthetic")) for row in clean),
        "real_data_rows": sum(row.get("synthetic") is False for row in clean),
        "rows": clean,
        "limitations": [
            "One seed is reported without mean or standard deviation.",
            "Missing validation receipts remain null; no metric is inferred from weights or parameters.",
            "Peak VRAM is reported only when a receipt records it.",
        ],
    }


def write_report(report: Mapping[str, Any], out: str | Path) -> dict[str, str]:
    destination = Path(out).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path, csv_path, md_path = destination / "REPORT.json", destination / "REPORT.csv", destination / "REPORT.md"
    _atomic_write(json_path, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    fields = list(CSV_FIELDS)
    lines: list[str] = []
    import io
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(_csv_row(row) for row in report.get("rows", []))
    lines.append(buffer.getvalue())
    _atomic_write(csv_path, "".join(lines))
    _atomic_write(md_path, _markdown(report))
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, help="durable candidate_a_campaign state JSON")
    parser.add_argument("--out", required=True, help="new or owned report directory")
    args = parser.parse_args(argv)
    report = build_report(args.state)
    print(json.dumps({"status": "WRITTEN", "jobs": report["total_jobs"], **write_report(report, args.out)}, indent=2))


if __name__ == "__main__":
    main()
