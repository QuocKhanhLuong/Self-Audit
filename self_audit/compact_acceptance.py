"""Deployment acceptance from pinned exported-student evidence; never software tests.

This assessor does not certify provenance on its own: callers must use the frozen
evaluation and isolated benchmark producers. Missing/invalid evidence is BLOCKED;
complete measurements failing a gate are NO_GO. Research superiority is separate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .compact_artifacts import digest, _require_sha256


def read_pinned_json(path, expected_sha):
    expected_sha = _require_sha256(expected_sha, "evidence SHA256")
    if digest(path) != expected_sha:
        raise ValueError(f"evidence hash mismatch: {path}")
    return json.loads(Path(path).read_text())


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


def assess_compact_deployment(export, quality, benchmark):
    """Assess one exported raw student using the fixed Macro4 >= .87 protocol."""
    blockers, gates = [], {}
    if export.get("schema") != "self_audit.compact.export.v1":
        blockers.append("unsupported export schema")
    if export.get("synthetic") is not False or quality.get("synthetic") is not False:
        blockers.append("patient evidence is missing or synthetic")
    export_sha = export.get("manifest_sha256")
    for field, expected in (("export_manifest_sha256", export_sha),
                            ("export_payload_sha256", export.get("payload_sha256")),
                            ("train_manifest_sha256", export.get("lineage", {}).get("data_sha256"))):
        if expected is None or quality.get(field) != expected:
            blockers.append(f"quality identity mismatch: {field}")
    if not quality.get("frozen_predictions_sha256") or not quality.get("recipe_sha256"):
        blockers.append("frozen prediction/recipe evidence missing")

    if quality.get("evaluation_source_manifest_sha256") != "22a9ea63821f266bd59e810bcd8f9a6958e6f92cc363e4a360b5ab9d4eb1e25c":
        blockers.append("quality split is not the locked historical TEST50 development reference")
    coverage = quality.get("coverage", {})
    if coverage.get("complete") is not True:
        blockers.append("complete prediction and scoring coverage evidence missing")

    # Recompute the historical class-first reduction from retained frame confusion.
    # Do not trust a separately typed Macro4 or a pooled aggregate.
    from .evaluation import class_metrics
    frames = quality.get("frames", [])
    by_patient = {}
    seen = set()
    try:
        if not frames:
            raise ValueError("no scored frames")
        for row in frames:
            key = (row["patient"], row["frame"])
            if key in seen:
                raise ValueError("duplicate reference frame")
            seen.add(key)
            cm = np.asarray(row["confusion"])
            if cm.shape != (4, 5) or not np.issubdtype(cm.dtype, np.integer) or (cm < 0).any() or cm.sum() == 0:
                raise ValueError("invalid/empty reference confusion")
            by_patient.setdefault(key[0], []).append(class_metrics(cm))
        expected_patients = set(quality.get("patients", {}))
        if len(expected_patients) != 50:
            raise ValueError("locked TEST50 protocol requires 50 scored patients")
        if set(by_patient) != expected_patients:
            raise ValueError("incomplete patient scoring coverage")
        expected_frames = coverage.get("expected_frames", {})
        expected_keys = {(pid, frame) for pid, frames_for_patient in expected_frames.items() for frame in frames_for_patient}
        if expected_keys != seen or coverage.get("scored_frames") != expected_frames:
            raise ValueError("incomplete reference frame coverage")
        class_means = []
        for c in range(4):
            values = []
            for rows in by_patient.values():
                present = [r[c]["dice"] for r in rows if r[c]["dice"] is not None]
                if present:
                    values.append(float(np.mean(present)))
            if not values:
                raise ValueError("Macro4 requires four defined class means")
            class_means.append(float(np.mean(values)))
        macro4 = float(np.mean(class_means))
        reported = quality.get("class_first_macro4")
        if not _number(reported) or not math.isclose(macro4, reported, abs_tol=1e-12):
            raise ValueError("reported Macro4 differs from frame/class/patient reduction")
        gates["macro4_at_least_0_87"] = macro4 >= .87
    except (ValueError, KeyError, TypeError) as exc:
        macro4, class_means = None, None
        blockers.append(f"quality reduction: {exc}")

    if benchmark.get("schema") != "self_audit.compact.benchmark.v1":
        blockers.append("unsupported benchmark schema")
    compact = benchmark.get("compact_student", {})
    baseline = benchmark.get("baseline_student", {})
    if compact.get("status") != "COMPLETED" or baseline.get("status") != "COMPLETED":
        blockers.append("both inference benchmarks must be completed")
    for arm in (compact, baseline):
        if arm.get("synthetic") is not False:
            blockers.append("benchmark patient evidence missing or synthetic")
        if arm.get("evaluation_manifest_sha256") != quality.get("evaluation_manifest_sha256"):
            blockers.append("benchmark cases differ from quality evaluation")
    if compact.get("artifact_manifest_sha256") != export_sha:
        blockers.append("benchmark did not use the evaluated export")
    if compact.get("artifact_payload_sha256") != export.get("payload_sha256"):
        blockers.append("benchmark export payload mismatch")
    for key in ("device", "precision", "workload", "io_policy"):
        if key not in compact or compact.get(key) != baseline.get(key):
            blockers.append(f"unmatched benchmark {key}")
    if not str(compact.get("device", "")).startswith("cuda"):
        blockers.append("target GPU inference memory was not measured")
    reference = benchmark.get("reference_identity", {})
    if baseline.get("artifact_manifest_sha256") != reference.get("deploy_manifest_sha256") or not reference.get("verified"):
        blockers.append("single EMA-run reference identity not verified")
    # Milestone one is the pinned ACDC L8 reference, whose published inference
    # used the raw student. Another teacher/state/protocol needs another receipt.
    if reference.get("source_manifest_sha256") != "03c1730f88941ada898bed7fcf1b9f13cdd8f08bcce4390539fec665c82e97de" or reference.get("inference_state") != "student":
        blockers.append("reference is not the locked historical raw student")
    if export.get("lineage", {}).get("data_sha256") != "e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4":
        blockers.append("this acceptance protocol is the locked ACDC L8 milestone")

    comparisons = (
        ("e2e_seconds_per_volume", ("latency", "steady", "e2e_median_seconds_per_volume")),
        ("peak_gpu_allocated", ("memory", "cuda_peak_allocated_mb")),
        ("peak_gpu_reserved", ("memory", "cuda_peak_reserved_mb")),
        ("peak_host_ram", ("memory", "peak_inference_process_tree_ram_mb")),
        ("weights_only_bytes", ("model_size", "deployed_weights_bytes")),
    )
    measured = {}
    for name, path in comparisons:
        a, b = compact, baseline
        for key in path:
            a = a.get(key) if isinstance(a, dict) else None
            b = b.get(key) if isinstance(b, dict) else None
        measured[name] = {"compact": a, "reference": b}
        if not _number(a) or not _number(b) or a <= 0 or b <= 0:
            blockers.append(f"missing positive finite measurement: {name}")
        else:
            gates[name] = a < b
    return {
        "schema": "self_audit.compact.acceptance.v1",
        "status": "BLOCKED" if blockers else ("GO" if all(gates.values()) else "NO_GO"),
        "threshold_macro4": .87,
        "macro4_recomputed": macro4,
        "class_means_BG_RV_MYO_LV": class_means,
        "gates": gates,
        "measurements": measured,
        "blockers": sorted(set(blockers)),
        "audit_contribution": "NOT_ESTABLISHED_BY_DEPLOYMENT_GATES",
        "joint_superiority": "REQUIRES_MATCHED_EMPIRICAL_COMPARISON",
    }
