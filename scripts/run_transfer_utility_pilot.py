#!/usr/bin/env python3
"""Sequential four-arm pilot runner for scalar transfer-utility distillation study.

Contracted in docs/CPTU_IMPLEMENTATION_CONTRACT.md and docs/TRANSFER_UTILITY_RUNBOOK.md.
Modes:
  - Arm 1: plain_kd (original learning rate, baseline without audit candidate selection)
  - Arm 2: scalar_guard with selector=original
  - Arm 3: scalar_guard with selector=larger_tie
  - Arm 4: scalar_guard with selector=cross_patient_utility
  - Conditional Arm 5: scalar_guard with selector=same_batch_utility (triggered only if Arm 4 > Arm 3 by > 0.002)

Dispatches strictly through canonical train.py CLI, exports via export_compact_student.py,
and evaluates on VAL using historical evaluate_compact_student.py and benchmark_compact_student.py.
Never invents duplicate trainers or independent evaluators.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.compact_artifacts import (
    digest,
    compact_training_source_sha256,
)
from self_audit.compact_data import CompactTrainingData, synthetic_compact_fixture
from self_audit.transfer_preflight import (
    check_preflight, check_gpu_occupancy, check_active_training_jobs, validate_locked_protocol,
)

# Immutable Protocol Specification Pins
EXPECTED_WIDTHS = [16, 32, 64, 128]
EXPECTED_STEPS = 2000
EXPECTED_WARMUP = 1000
EXPECTED_EVAL_STEPS = (1000, 1500, 2000)
EXPECTED_SEED = 42
EXPECTED_LR = 3e-4
EXPECTED_PSEUDO_LR = 1e-3
EXPECTED_TEMPERATURE = 1.0
EXPECTED_PSEUDO_THRESHOLD = 0.7
EXPECTED_RIDGE = 1e-6
EXPECTED_GUARD_TOLERANCE = 1e-6
EXPECTED_MAX_TRIALS = 8
EXPECTED_BATCH_SIZE = 8
EXPECTED_CROP_SIZE = 160
EXPECTED_TRIAL_SCALES = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
WATCHDOG_DEFAULT_SECONDS = 7200
SCREENING_DELTA_THRESHOLD = 0.002
FOREGROUND_CLASSES = ["RV", "MYO", "LV"]
ALL_CLASSES = ["BG", "RV", "MYO", "LV"]
L8_PATIENT_IDS = [
    "patient007", "patient010", "patient011", "patient024",
    "patient026", "patient041", "patient077", "patient099",
]
EXPECTED_TRAIN_MANIFEST_SHA = "e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4"
EXPECTED_TEACHER_MANIFEST_SHA = "ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa"
EXPECTED_TEACHER_PAYLOAD_SHA = "0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6"


def require_sha256(value: Any, name: str) -> str:
    """Validate 64-hex SHA-256 string."""
    if not isinstance(value, str) or len(value) != 64 or not all(c in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"{name} must be a valid 64-hex SHA-256 string, got {value!r}")
    return value.lower()


def read_json(path: Path | str) -> Dict[str, Any]:
    """Read JSON file with strict parsing."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return json.loads(p.read_text())


def write_json(path: Path | str, data: Any) -> None:
    """Atomic write JSON file with indentation and no NaN."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp_{os.getpid()}_{int(time.time()*1000)}")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


# ==============================================================================
# Preflight & Resource Validation
# ==============================================================================







# ==============================================================================
# Bounded Profile Runner (up to 32 steps)
# ==============================================================================



# ==============================================================================
# Canonical Command Generation (Zero invented flags)
# ==============================================================================

def build_train_command(
    config_path: Path | str,
    out_dir: Path | str,
    *,
    data_manifest: Optional[str | Path] = None,
    data_manifest_sha256: Optional[str] = None,
    device: str = "cuda",
    synthetic: bool = False,
    steps: Optional[int] = None,
    stop_after_step: Optional[int] = None,
    allow_full_run: bool = True,
    resume: Optional[str | Path] = None,
    resume_sha256: Optional[str] = None,
    py_exec: str = sys.executable,
) -> Tuple[List[str], Dict[str, Any]]:
    """Build canonical train.py CLI command and execution manifest."""
    cmd = [
        py_exec,
        str(ROOT / "train.py"),
        "--config", str(config_path),
        "--out", str(out_dir),
        "--device", str(device),
    ]

    if synthetic:
        cmd.append("--synthetic")

    if data_manifest and data_manifest_sha256:
        cmd.extend(["--data", str(data_manifest), "--data-sha256", str(data_manifest_sha256)])

    if steps is not None:
        cmd.extend(["--steps", str(steps)])

    if stop_after_step is not None:
        cmd.extend(["--stop-after-step", str(stop_after_step)])

    if allow_full_run:
        cmd.append("--allow-full-run")

    if resume:
        if not resume_sha256:
            raise ValueError("--resume requires --resume-sha256")
        cmd.extend(["--resume", str(resume), "--resume-sha256", str(resume_sha256)])

    manifest = {
        "command": cmd,
        "config_path": str(config_path),
        "config_sha256": digest(config_path),
        "out_dir": str(out_dir),
        "device": str(device),
        "synthetic": bool(synthetic),
        "steps": steps,
        "stop_after_step": stop_after_step,
        "resume": str(resume) if resume else None,
        "resume_sha256": resume_sha256,
        "source_sha256": compact_training_source_sha256(),
    }
    return cmd, manifest


def build_export_command(
    checkpoint_dir: Path | str,
    checkpoint_sha256: str,
    out_dir: Path | str,
    *,
    py_exec: str = sys.executable,
) -> Tuple[List[str], Dict[str, Any]]:
    """Build canonical export_compact_student.py command."""
    cmd = [
        py_exec,
        str(ROOT / "scripts" / "export_compact_student.py"),
        "--checkpoint", str(checkpoint_dir),
        "--checkpoint-sha256", str(checkpoint_sha256),
        "--out", str(out_dir),
    ]
    manifest = {
        "command": cmd,
        "checkpoint": str(checkpoint_dir),
        "checkpoint_sha256": checkpoint_sha256,
        "out": str(out_dir),
    }
    return cmd, manifest


def build_evaluate_command(
    export_dir: Path | str,
    export_sha256: str,
    manifest_path: Path | str,
    manifest_sha256: str,
    train_manifest_path: Path | str,
    train_manifest_sha256: str,
    out_dir: Path | str,
    *,
    device: str = "cpu",
    batch_size: int = 16,
    normalization: str = "prepared",
    py_exec: str = sys.executable,
) -> Tuple[List[str], Dict[str, Any]]:
    """Build canonical evaluate_compact_student.py command."""
    cmd = [
        py_exec,
        str(ROOT / "scripts" / "evaluate_compact_student.py"),
        "--export", str(export_dir),
        "--export-sha256", str(export_sha256),
        "--manifest", str(manifest_path),
        "--manifest-sha256", str(manifest_sha256),
        "--train-manifest", str(train_manifest_path),
        "--train-manifest-sha256", str(train_manifest_sha256),
        "--out", str(out_dir),
        "--device", str(device),
        "--batch-size", str(batch_size),
        "--normalization", str(normalization),
    ]
    manifest = {
        "command": cmd,
        "export": str(export_dir),
        "export_sha256": export_sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "train_manifest": str(train_manifest_path),
        "train_manifest_sha256": train_manifest_sha256,
        "out": str(out_dir),
        "device": device,
    }
    return cmd, manifest


def build_benchmark_command(
    compact_export_dir: Path | str,
    compact_export_sha256: str,
    manifest_path: Path | str,
    manifest_sha256: str,
    out_file: Path | str,
    *,
    device: str = "cpu",
    batch_size: int = 16,
    warmup: int = 1,
    repeats: int = 3,
    normalization: str = "prepared",
    py_exec: str = sys.executable,
) -> Tuple[List[str], Dict[str, Any]]:
    """Build canonical benchmark_compact_student.py command."""
    cmd = [
        py_exec,
        str(ROOT / "scripts" / "benchmark_compact_student.py"),
        "--compact-export", str(compact_export_dir),
        "--compact-export-sha256", str(compact_export_sha256),
        "--manifest", str(manifest_path),
        "--manifest-sha256", str(manifest_sha256),
        "--out", str(out_file),
        "--device", str(device),
        "--batch-size", str(batch_size),
        "--warmup", str(warmup),
        "--repeats", str(repeats),
        "--normalization", str(normalization),
    ]
    manifest = {
        "command": cmd,
        "compact_export": str(compact_export_dir),
        "compact_export_sha256": compact_export_sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "out": str(out_file),
        "device": device,
    }
    return cmd, manifest


# ==============================================================================
# Transaction Diagnostics & Valuation Analysis
# ==============================================================================

def analyze_transaction_diagnostics(train_log_path: Path | str) -> Dict[str, Any]:
    """Parse TRAIN_LOG.jsonl (or steps.jsonl) extracting transaction diagnostics, alpha histogram, gains, norms."""
    p = Path(train_log_path)
    if not p.is_file():
        alt = p.parent / "TRAIN_LOG.jsonl" if p.name != "TRAIN_LOG.jsonl" else p.parent / "steps.jsonl"
        if alt.is_file():
            p = alt
        else:
            return {"status": "NO_STEPS_LOG", "path": str(p)}

    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return {"status": "EMPTY_STEPS_LOG", "path": str(p)}

    # Alpha histogram across candidate scales
    alpha_counts: Dict[str, int] = {str(s): 0 for s in EXPECTED_TRIAL_SCALES}
    alpha_counts["0.0"] = 0
    alpha_counts["other"] = 0

    warmup_steps = 0
    no_candidate_steps = 0
    accepted_nonzero_steps = 0
    zero_displacement_steps = 0
    rejected_all_steps = 0

    proposed_norms: List[float] = []
    realized_norms: List[float] = []
    gains: List[float] = []
    selected_gains: List[float] = []
    class_gains = {name: [] for name in ALL_CLASSES}
    reasons: Dict[str, int] = {}
    step_latencies: List[float] = []
    audit_latencies: List[float] = []
    supervised_latencies: List[float] = []

    for r in rows:
        step_lat = r.get("timings", {}).get("step_elapsed_seconds")
        if step_lat is not None:
            step_latencies.append(float(step_lat))
        audit_lat = r.get("timings", {}).get("audit_trials_seconds")
        if audit_lat is not None:
            audit_latencies.append(float(audit_lat))
        sup_lat = r.get("timings", {}).get("supervised_seconds")
        if sup_lat is not None:
            supervised_latencies.append(float(sup_lat))

        status = r.get("status", "unknown")
        is_warmup = status == "warmup" or bool(r.get("warm_start", {}).get("in_warmup", False))

        if is_warmup:
            warmup_steps += 1
            continue

        if status == "no_candidate":
            no_candidate_steps += 1
            continue

        # U displacement attempted
        p_norm = r.get("proposed_displacement_norm")
        if p_norm is not None:
            proposed_norms.append(float(p_norm))
        r_norm = r.get("realized_displacement_norm")
        if r_norm is not None:
            realized_norms.append(float(r_norm))

        if status == "accepted_nonzero" or bool(r.get("accepted", False)):
            accepted_nonzero_steps += 1
        elif status == "zero_displacement":
            zero_displacement_steps += 1
        elif status == "rejected_all":
            rejected_all_steps += 1

        scale = r.get("scale", 0.0)
        matched_scale = False
        for s in EXPECTED_TRIAL_SCALES:
            if abs(scale - s) < 1e-5:
                alpha_counts[str(s)] += 1
                matched_scale = True
                break
        if not matched_scale:
            if abs(scale) < 1e-5:
                alpha_counts["0.0"] += 1
            else:
                alpha_counts["other"] += 1

        # Check for selector probe gain (safely handle None)
        tx = r.get("transaction", {})
        reason = tx.get("reason", "not_recorded")
        reasons[reason] = reasons.get(reason, 0) + 1
        trials = tx.get("trials", [])
        for t in trials:
            g = t.get("utility_gain") if "utility_gain" in t else t.get("gain")
            if g is not None and isinstance(g, (int, float)):
                gains.append(float(g))
            if t.get("accepted") is True and isinstance(g, (int, float)):
                selected_gains.append(float(g))
                reference = tx.get("telemetry", {}).get("classwise_kd_loss") or {}
                trial_classes = t.get("probe_classwise_kd_loss") or {}
                for name in ALL_CLASSES:
                    if isinstance(reference.get(name), (int, float)) and isinstance(trial_classes.get(name), (int, float)):
                        class_gains[name].append(reference[name] - trial_classes[name])
        if status in ("zero_displacement", "rejected_all") and tx.get("telemetry", {}).get("reference_kd_loss") is not None:
            # Logical zero reuses the reference and contributes exactly zero gain.
            selected_gains.append(0.0)
            for name in ALL_CLASSES:
                if (tx.get("telemetry", {}).get("classwise_kd_loss") or {}).get(name) is not None:
                    class_gains[name].append(0.0)

    total_steps = len(rows)
    u_attempted_steps = total_steps - warmup_steps - no_candidate_steps

    u_acceptance_rate = (accepted_nonzero_steps / u_attempted_steps) if u_attempted_steps > 0 else None
    overall_acceptance_rate = (accepted_nonzero_steps / total_steps) if total_steps > 0 else None
    u_zero_rate = (zero_displacement_steps / u_attempted_steps) if u_attempted_steps > 0 else None
    u_reject_rate = (rejected_all_steps / u_attempted_steps) if u_attempted_steps > 0 else None

    return {
        "status": "COMPLETED",
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "no_candidate_steps": no_candidate_steps,
        "u_attempted_steps": u_attempted_steps,
        "alpha_histogram": alpha_counts,
        "transaction_outcomes": {
            "accepted_nonzero": accepted_nonzero_steps,
            "zero_displacement": zero_displacement_steps,
            "rejected_all": rejected_all_steps,
            "u_acceptance_rate": round(u_acceptance_rate, 4) if u_acceptance_rate is not None else None,
            "u_zero_rate": round(u_zero_rate, 4) if u_zero_rate is not None else None,
            "u_reject_rate": round(u_reject_rate, 4) if u_reject_rate is not None else None,
            "overall_acceptance_rate": round(overall_acceptance_rate, 4) if overall_acceptance_rate is not None else None,
        },
        "displacement_norms": {
            "mean_proposed": float(sum(proposed_norms) / len(proposed_norms)) if proposed_norms else None,
            "mean_realized": float(sum(realized_norms) / len(realized_norms)) if realized_norms else None,
            "max_realized": float(max(realized_norms)) if realized_norms else None,
        },
        "utility_probe_gains": {
            "scope": "all evaluated feasible nonzero trials, including unselected trials",
            "count": len(gains),
            "mean": float(sum(gains) / len(gains)) if gains else None,
            "min": float(min(gains)) if gains else None,
            "max": float(max(gains)) if gains else None,
        },
        "selected_probe_gains": {
            "scope": "selected nonzero or valid-reference logical-zero decisions; unavailable probes excluded",
            "count": len(selected_gains),
            "mean": sum(selected_gains) / len(selected_gains) if selected_gains else None,
            "classwise_mean": {name: sum(values) / len(values) if values else None
                               for name, values in class_gains.items()},
            "classwise_count": {name: len(values) for name, values in class_gains.items()},
        },
        "selection_reasons": reasons,
        "latencies": {
            "mean_step_seconds": float(sum(step_latencies) / len(step_latencies)) if step_latencies else None,
            "mean_audit_seconds": float(sum(audit_latencies) / len(audit_latencies)) if audit_latencies else None,
            "mean_supervised_seconds": float(sum(supervised_latencies) / len(supervised_latencies)) if supervised_latencies else None,
            "total_wall_seconds": float(sum(step_latencies)) if step_latencies else None,
        },
    }


def analyze_checkpoint_evaluations(eval_dir: Path | str) -> Dict[str, Any]:
    """Ingest METRICS.json from an evaluation output directory and derive pooled RV counts."""
    p = Path(eval_dir)
    mfile = p / "METRICS.json"
    if not mfile.is_file():
        return {"status": "NOT_EVALUATED", "path": str(p)}

    data = read_json(mfile)
    macro4 = data.get("class_first_macro4")
    fg_macro = data.get("class_first_foreground_macro")
    macro_patient = data.get("macro_patient", {})

    class_dices: Dict[str, Optional[float]] = {}
    for c in ALL_CLASSES:
        class_dices[c] = macro_patient.get(c, {}).get("dice")

    # Pooled RV Precision and Recall derived from actual frame records
    rv_tp = 0
    rv_fp = 0
    rv_fn = 0
    has_rv = False

    for f in data.get("frames", []):
        raw_list = f.get("raw", [])
        for entry in raw_list:
            if isinstance(entry, Mapping) and entry.get("class") == "RV":
                tp_val = entry.get("TP")
                fp_val = entry.get("FP")
                fn_val = entry.get("FN")
                if tp_val is not None:
                    rv_tp += int(tp_val)
                    has_rv = True
                if fp_val is not None:
                    rv_fp += int(fp_val)
                if fn_val is not None:
                    rv_fn += int(fn_val)

    rv_precision = float(rv_tp / (rv_tp + rv_fp)) if has_rv and (rv_tp + rv_fp) > 0 else None
    rv_recall = float(rv_tp / (rv_tp + rv_fn)) if has_rv and (rv_tp + rv_fn) > 0 else None

    # Per-patient metrics from actual compute_historical_metrics output
    patient_metrics = data.get("patients", {})

    return {
        "status": "COMPLETED",
        "class_first_macro4": macro4,
        "class_first_foreground_macro": fg_macro,
        "class_dices": class_dices,
        "rv_precision": round(rv_precision, 5) if rv_precision is not None else None,
        "rv_recall": round(rv_recall, 5) if rv_recall is not None else None,
        "rv_pooled_counts": {"TP": rv_tp, "FP": rv_fp, "FN": rv_fn} if has_rv else None,
        "patients": patient_metrics,
        "synthetic": data.get("synthetic"),
        "export_manifest_sha256": data.get("export_manifest_sha256"),
        "frozen_predictions_sha256": data.get("frozen_predictions_sha256"),
        "metrics_sha256": digest(mfile),
        "class_counts": data.get("class_counts"),
        "failure_cases": data.get("failure_cases"),
        "evaluation_manifest_sha256": data.get("evaluation_manifest_sha256"),
        "coverage": data.get("coverage", {}),
    }


# ==============================================================================
# Screening Decision Rule
# ==============================================================================



# ==============================================================================
# Arm Execution Engine
# ==============================================================================



# ==============================================================================
# Full Sequential Pilot Orchestrator
# ==============================================================================



# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def bind_teacher_config(config, cache_path=None, cache_sha=None):
    """Bind the actual cache identity; never manufacture a verified teacher."""
    cfg = copy.deepcopy(config)
    teacher = cfg.setdefault("teacher", {})
    path = cache_path or teacher.get("cache_path")
    expected = cache_sha or teacher.get("cache_sha256")
    if path:
        path = Path(path).resolve()
        if path.is_dir():
            path /= "CACHE_MANIFEST.json"
        if digest(path) != require_sha256(expected, "teacher cache SHA"):
            raise ValueError("teacher cache hash mismatch")
        cache = read_json(path)
        identity = cache["teacher_identity"]
        teacher.update(cache_path=str(path), cache_sha256=digest(path),
                       teacher_identity=identity)
        # Costs remain evidence, not a claim that we reran the source teacher.
        teacher["source_training_cost"] = identity.get(
            "source_training_cost", teacher.get("source_training_cost", "NOT_KNOWN"))
    return cfg


def _run_logged(cmd, root, tag, timeout):
    """Bound only our child process; preserve stdout/stderr even on failure."""
    if timeout <= 0:
        raise TimeoutError("locked watchdog exhausted")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / f"{tag}_COMMAND.json", {"argv": cmd})
    with (root / f"{tag}.stdout.log").open("w") as stdout, (root / f"{tag}.stderr.log").open("w") as stderr:
        result = subprocess.run(cmd, cwd=ROOT, stdout=stdout, stderr=stderr,
                                timeout=timeout, text=True)
    if result.returncode:
        raise RuntimeError(f"{tag} exited {result.returncode}; see {root / (tag + '.stderr.log')}")


def validate_arm_config(cfg, protocol, arm_name):
    """Refuse hidden experimental factors in locked real pilot configurations."""
    from self_audit.compact_training import validate_compact_config
    canonical = read_json(ROOT / "configs/transfer_utility" / f"{arm_name}.json")
    actual = validate_compact_config(cfg, allow_full_run=True)
    expected = validate_compact_config(canonical, allow_full_run=True)
    for key, value in protocol['hyperparameters'].items():
        if key != 'trial_scales' and actual.get(key) != value:
            raise ValueError(f"locked hyperparameter mismatch: {key}")
    for key, value in protocol['selector_parameters'].items():
        if actual.get(key) != value:
            raise ValueError(f"locked selector parameter mismatch: {key}")
    if actual['model'] != protocol['architecture'] or actual.get('data') != protocol['data']:
        raise ValueError("locked architecture/data mismatch")
    if actual['steps'] != protocol['budget']['total_steps'] or actual['warmup_steps'] != protocol['budget']['warmup_steps']:
        raise ValueError("locked steps mismatch")
    # Runtime assets are hash-verified separately. All other settings are frozen.
    for item in (actual, expected):
        teacher = item.get("teacher", {})
        for key in ("cache_path", "cache_sha256", "teacher_identity", "source", "source_training_cost"):
            teacher.pop(key, None)
    if actual != expected:
        raise ValueError(f"{arm_name}: config differs from the locked arm")
    errors = validate_locked_protocol(protocol)
    if errors:
        raise ValueError("; ".join(errors))


def run_bounded_profile(config_path, out_dir, *, device="cpu", synthetic=True,
                        data_manifest=None, data_manifest_sha256=None, max_steps=32,
                        py_exec=sys.executable, teacher_cache=None, teacher_cache_sha256=None):
    """Resource measurement only, separate initialization and budget from pilot."""
    if isinstance(max_steps, bool) or not 2 <= max_steps <= 32:
        raise ValueError("bounded profile requires 2..32 steps")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = bind_teacher_config(read_json(config_path), teacher_cache, teacher_cache_sha256)
    cfg.update(steps=max_steps, warmup_steps=min(8, max_steps // 2), save_every=max_steps)
    if synthetic and not data_manifest:
        data_manifest, data_manifest_sha256 = synthetic_compact_fixture(
            out / "fixture", n_labeled=2, n_unlabeled=2, size=16, num_frames=2, num_slices=2)
        cmd = [py_exec, str(ROOT / "scripts/build_compact_teacher_cache.py"),
               "--data-manifest", str(data_manifest), "--data-sha256", data_manifest_sha256,
               "--output-dir", str(out / "cache"), "--synthetic-teacher", "--device", "cpu"]
        _run_logged(cmd, out, "cache", 60)
        cache = out / "cache/CACHE_MANIFEST.json"
        ids = CompactTrainingData(data_manifest, data_manifest_sha256, synthetic=True, allow_k1=True).lids
        cfg["data"] = {"manifest_sha256": data_manifest_sha256, "budget_k": len(ids), "labeled_ids": ids}
        cfg["teacher"] = {"lineage": {"budget_k": len(ids), "exact_labeled_ids": ids}}
        cfg = bind_teacher_config(cfg, cache, digest(cache))
    config_file = out / "profile_config.json"
    write_json(config_file, cfg)
    command, _ = build_train_command(config_file, out / "training", data_manifest=data_manifest,
                                    data_manifest_sha256=data_manifest_sha256, device=device,
                                    synthetic=synthetic, steps=max_steps, py_exec=py_exec)
    start = time.monotonic()
    receipt = {"schema": "self_audit.compact.transfer_utility.profile.v1", "synthetic": synthetic,
               "max_steps": max_steps, "warmup_steps": cfg["warmup_steps"],
               "u_steps": max_steps - cfg["warmup_steps"], "device": device,
               "source_sha256": compact_training_source_sha256(), "config_sha256": digest(config_file),
               "train_manifest_sha256": data_manifest_sha256,
               "teacher_cache_sha256": cfg.get("teacher", {}).get("cache_sha256")}
    try:
        _run_logged(command, out, "profile", 600)
        report = read_json(out / "training/REPORT.json")
        diag = analyze_transaction_diagnostics(out / "training/TRAIN_LOG.jsonl")
        if report["steps"] != max_steps or report["status"] != "COMPLETED":
            raise RuntimeError("bounded profile did not finish its declared steps")
        receipt.update(status="PASS", training_report=report, diagnostics=diag,
                       checkpoint_manifest_sha256=digest(out / "training/final/MANIFEST.json"))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        receipt.update(status="FAIL", reason=str(exc))
    elapsed = time.monotonic() - start
    receipt.update(elapsed_seconds=elapsed, seconds_per_step=elapsed / max_steps,
                   projected_2000_steps_seconds=elapsed / max_steps * 2000,
                   within_watchdog_7200s=elapsed / max_steps * 2000 < 7200)
    write_json(out / "PROFILE_RECEIPT.json", receipt)
    return receipt


def evaluate_screening_decision(arm_results):
    """Screen final exported VAL18 only after cohort and training comparability checks."""
    names = ["plain_kd", "scalar_guard_original", "scalar_guard_larger_tie", "scalar_guard_cross_patient_utility"]
    missing = [(name, arm_results.get(name, {}).get("status", "MISSING"))
               for name in names if arm_results.get(name, {}).get("status") != "COMPLETED"]
    def blocked(reason, verdict="BLOCKED", **extra):
        return {"verdict": verdict, "reason": reason, "screening_passed": False,
                "should_run_conditional_5th_arm": False, **extra}
    if missing:
        return blocked("Required arms missing or incomplete", "INCOMPLETE", missing_arms=missing)
    evals = {name: arm_results[name].get("evaluations", {}).get("step_002000", {}) for name in names}
    synthetic = any(e.get("synthetic") is True for e in evals.values())
    if any(e.get("synthetic") not in (True, False) for e in evals.values()):
        return blocked("Missing explicit synthetic/real evidence flag")
    cohorts = [sorted(e.get("patients", {})) for e in evals.values()]
    if not cohorts[0] or any(x != cohorts[0] for x in cohorts):
        return blocked("Missing or unmatched evaluated patient cohorts")
    expected_sha = "cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50"
    for name, evaluation in evals.items():
        if evaluation.get("status") != "COMPLETED" or evaluation.get("coverage", {}).get("complete") is not True:
            return blocked(f"{name}: evaluation incomplete")
        score = evaluation.get("class_first_macro4")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            return blocked(f"{name}: missing/nonfinite Macro4")
        if not synthetic and (evaluation.get("evaluation_manifest_sha256") != expected_sha or len(cohorts[0]) != 18):
            return blocked("Real evaluation must bind the pinned VAL18 manifest and all 18 patients")
        if not synthetic:
            if arm_results[name].get("steps") != 2000 or evaluation.get("checkpoint_step") != 2000:
                return blocked(f"{name}: not the locked final step")
            try:
                require_sha256(evaluation.get("export_manifest_sha256"), "export manifest")
                require_sha256(evaluation.get("frozen_predictions_sha256"), "frozen predictions")
            except ValueError as exc:
                return blocked(str(exc))
    if not synthetic:
        for key in ("initial_state_sha256", "data_schedule_sha256", "source_sha256", "train_manifest_sha256", "teacher_cache_sha256"):
            values = [arm_results[n].get(key) for n in names]
            if not values[0] or any(v != values[0] for v in values):
                return blocked(f"Unmatched or missing {key}")
    scores = {name: e["class_first_macro4"] for name, e in evals.items()}
    utility = scores[names[-1]]
    deltas = {"vs_" + name: utility - scores[name] for name in names[:-1]}
    passed = all(value > SCREENING_DELTA_THRESHOLD for value in deltas.values())
    paired = {}
    for name in names[:-1]:
        paired[name] = {}
        for pid in cohorts[0]:
            a, b = evals[names[-1]]["patients"][pid], evals[name]["patients"][pid]
            paired[name][pid] = {key: (a[key] - b[key] if a.get(key) is not None and b.get(key) is not None else None)
                                 for key in (*ALL_CLASSES, "foreground_mean", "all_class_mean")}
    return {"verdict": "SYNTHETIC_PASS" if synthetic else "PILOT_GO" if passed else "NO_GO",
            "synthetic": synthetic, "screening_passed": passed and not synthetic,
            "screening_threshold": SCREENING_DELTA_THRESHOLD, "final_macro4s": scores,
            "deltas": deltas, "paired_patient_differences": paired,
            "criteria": {"exceeds_plain_kd_by_002": deltas["vs_plain_kd"] > .002,
                         "exceeds_original_by_002": deltas["vs_scalar_guard_original"] > .002,
                         "exceeds_larger_tie_by_002": deltas["vs_scalar_guard_larger_tie"] > .002},
            "should_run_conditional_5th_arm": not synthetic and deltas["vs_scalar_guard_larger_tie"] > .002,
            "notes": "Synthetic data cannot certify clinical PILOT_GO." if synthetic else
                     "Development screening only; no deployment or joint superiority claim. Matched-displacement and same-batch controls required before a strong claim."}


def execute_arm(arm_name, config_path, arm_out_dir, *, data_manifest=None,
                data_manifest_sha256=None, val_manifest=None, val_manifest_sha256=None,
                device="cpu", synthetic=False, watchdog_seconds=WATCHDOG_DEFAULT_SECONDS,
                py_exec=sys.executable, teacher_cache=None, teacher_cache_sha256=None,
                stop_after_steps=None):
    """Fresh output per segment. Export/reload and VAL evaluation at every boundary."""
    cfg = bind_teacher_config(read_json(config_path), teacher_cache, teacher_cache_sha256)
    if not synthetic:
        validate_arm_config(cfg, read_json(ROOT / "configs/transfer_utility/protocol.json"), arm_name)
    if not all((data_manifest, data_manifest_sha256, val_manifest, val_manifest_sha256)):
        raise ValueError("arm execution requires hash-pinned TRAIN and VAL manifests")
    total = cfg["steps"]
    boundaries = list(stop_after_steps or (EXPECTED_EVAL_STEPS if not synthetic else [total]))
    if boundaries != sorted(set(boundaries)) or boundaries[0] < 1 or boundaries[-1] != total:
        raise ValueError("stop boundaries must increase and end at the immutable total steps")
    if not synthetic and boundaries != list(EXPECTED_EVAL_STEPS):
        raise ValueError("real pilot eval boundaries are locked")
    out = Path(arm_out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config_file = out / "ARM_CONFIG.json"
    write_json(config_file, cfg)
    rec = {"arm_name": arm_name, "selector": cfg.get("selector", "original"), "status": "RUNNING",
           "synthetic": synthetic, "evaluations": {}, "segments": [], "steps": 0,
           "watchdog_seconds": watchdog_seconds, "source_sha256": compact_training_source_sha256(),
           "train_manifest_sha256": data_manifest_sha256,
           "teacher_cache_sha256": cfg.get("teacher", {}).get("cache_sha256"),
           "config_sha256": digest(config_file), "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    receipt_path = out / "ARM_MANIFEST.json"
    write_json(receipt_path, rec)
    started = time.monotonic()
    resume = resume_sha = None
    try:
        for stop in boundaries:
            tag = f"step_{stop:06d}"
            segment = out / "segments" / tag
            cmd, _ = build_train_command(config_file, segment, data_manifest=data_manifest,
                data_manifest_sha256=data_manifest_sha256, device=device, synthetic=synthetic,
                steps=total, stop_after_step=stop, resume=resume, resume_sha256=resume_sha, py_exec=py_exec)
            segment_start = time.monotonic()
            _run_logged(cmd, out / "logs", f"train_{tag}", watchdog_seconds - (time.monotonic() - started))
            train_report = read_json(segment / "REPORT.json")
            expected_status = "COMPLETED" if stop == total else "STOPPED_BOUNDARY"
            if train_report.get("status") != expected_status or train_report.get("steps") != stop:
                raise RuntimeError(f"training stopped outside boundary {stop}")
            checkpoint = segment / "final"
            if read_json(checkpoint / "MANIFEST.json").get("step") != stop:
                raise RuntimeError("checkpoint step mismatch")
            resume, resume_sha = str(checkpoint), digest(checkpoint / "MANIFEST.json")
            export = out / "exports" / tag
            cmd, _ = build_export_command(checkpoint, resume_sha, export, py_exec=py_exec)
            _run_logged(cmd, out / "logs", f"export_{tag}", min(180, watchdog_seconds - (time.monotonic() - started)))
            export_sha = digest(export / "MANIFEST.json")
            evaluation = out / "evaluations" / tag
            cmd, _ = build_evaluate_command(export, export_sha, val_manifest, val_manifest_sha256,
                    data_manifest, data_manifest_sha256, evaluation, device=device, py_exec=py_exec)
            _run_logged(cmd, out / "logs", f"eval_{tag}", min(600, watchdog_seconds - (time.monotonic() - started)))
            metrics = analyze_checkpoint_evaluations(evaluation)
            if metrics.get("status") != "COMPLETED" or metrics.get("coverage", {}).get("complete") is not True:
                raise RuntimeError("evaluation failed or incomplete prediction coverage")
            if metrics.get("export_manifest_sha256") != export_sha or metrics.get("evaluation_manifest_sha256") != val_manifest_sha256:
                raise RuntimeError("evaluation/export provenance mismatch")
            metrics.update(checkpoint_step=stop, checkpoint_manifest_sha256=resume_sha)
            rec["evaluations"][tag] = metrics
            rec["segments"].append({"step": stop, "directory": str(segment),
                "wall_seconds_including_export_eval": time.monotonic() - segment_start,
                "checkpoint_manifest_sha256": resume_sha, "report_sha256": digest(segment / "REPORT.json")})
            rec.update(steps=stop, final_checkpoint=resume, final_export=str(export),
                       final_export_manifest_sha256=export_sha, training_report=train_report)
            write_json(receipt_path, rec)
        log = segment / "TRAIN_LOG.jsonl"
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        if len(rows) != total or [r["step"] for r in rows] != list(range(total)):
            raise RuntimeError("training log does not cover the immutable step schedule")
        (out / "TRAIN_LOG.jsonl").write_bytes(log.read_bytes())
        rec["data_schedule_sha256"] = hashlib.sha256(json.dumps([r["data_schedule"] for r in rows], sort_keys=True).encode()).hexdigest()
        rec["initial_state_sha256"] = train_report.get("initial_state_sha256")
        if not rec["initial_state_sha256"]:
            rec["initial_state_sha256"] = read_json(checkpoint / "MANIFEST.json").get("protocol", {}).get("initial_state_sha256")
        require_sha256(rec["initial_state_sha256"], "initial student weights")
        rec["diagnostics"] = analyze_transaction_diagnostics(out / "TRAIN_LOG.jsonl")
        write_json(out / "DIAGNOSTICS.json", rec["diagnostics"])
        rec["status"] = "COMPLETED"
    except (OSError, ValueError, KeyError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
        rec.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}")
    rec["train_export_eval_wall_seconds"] = time.monotonic() - started
    rec["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(receipt_path, rec)
    return rec


def orchestrate_pilot(protocol_path, matrix_path, out_dir, *, data_manifest=None,
        data_manifest_sha256=None, val_manifest=None, val_manifest_sha256=None, device="cuda",
        synthetic=False, skip_preflight=False, skip_jobs_check=False, teacher_checkpoint=None,
        teacher_checkpoint_sha256=None, teacher_payload_sha256=None, teacher_cache=None,
        teacher_cache_sha256=None, py_exec=sys.executable):
    """Locked profile then four sequential arms; no implicit restarts or budget changes."""
    if skip_preflight or (skip_jobs_check and not synthetic):
        raise ValueError("real pilot safety/provenance preflight cannot be bypassed")
    out = Path(out_dir).resolve()
    protocol, matrix = read_json(protocol_path), read_json(matrix_path)
    preflight = check_preflight(protocol=protocol, data_manifest=data_manifest,
        data_manifest_sha256=data_manifest_sha256, out_dir=out, device=device, synthetic=synthetic,
        check_jobs=not skip_jobs_check, teacher_checkpoint=teacher_checkpoint,
        teacher_checkpoint_sha256=teacher_checkpoint_sha256, teacher_payload_sha256=teacher_payload_sha256,
        teacher_cache=teacher_cache, teacher_cache_sha256=teacher_cache_sha256,
        val_manifest=val_manifest, val_manifest_sha256=val_manifest_sha256)
    if preflight["status"] != "PASS":
        return {"status": "BLOCKED", "preflight": preflight, "arm_results": {}}
    expected_matrix = read_json(ROOT / "configs/transfer_utility/matrix.json")
    if matrix != expected_matrix:
        raise ValueError("matrix differs from locked pilot")
    if synthetic:
        raise ValueError("synthetic integration uses smoke_compact_pipeline.py --transfer-utility; all mode is the locked real pilot")
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "self_audit.compact.transfer_utility.report.v1", "status": "RUNNING",
              "protocol": protocol, "protocol_sha256": digest(protocol_path), "matrix_sha256": digest(matrix_path),
              "preflight": preflight, "arm_results": {}, "source_sha256": compact_training_source_sha256()}
    report_path = out / "TRANSFER_UTILITY_PILOT_REPORT.json"
    write_json(out / "PREFLIGHT_RECEIPT.json", preflight)
    write_json(report_path, report)
    common = dict(data_manifest=data_manifest, data_manifest_sha256=data_manifest_sha256,
                  device=device, synthetic=False, teacher_cache=teacher_cache,
                  teacher_cache_sha256=teacher_cache_sha256, py_exec=py_exec)
    profile = run_bounded_profile(ROOT / "configs/transfer_utility/profile_32steps.json", out / "profile", **common)
    report["profile"] = profile
    if profile["status"] != "PASS":
        report["status"] = "BLOCKED_PROFILE"
        write_json(report_path, report)
        return report
    for job in matrix["jobs"]:
        if not job.get("required", False):
            screening = evaluate_screening_decision(report["arm_results"])
            report["screening"] = screening
            if not screening.get("should_run_conditional_5th_arm"):
                continue
        gpu = check_gpu_occupancy(device)
        if gpu["status"] != "PASS":
            report.update(status="BLOCKED_GPU", gpu_check=gpu)
            write_json(report_path, report)
            return report
        arm = execute_arm(job["name"], ROOT / job["config"], out / "runs" / job["name"],
                          val_manifest=val_manifest, val_manifest_sha256=val_manifest_sha256,
                          stop_after_steps=protocol["budget"]["eval_steps"], **common)
        report["arm_results"][job["name"]] = arm
        write_json(report_path, report)
        if arm["status"] != "COMPLETED":
            report["status"] = "INCOMPLETE"
            write_json(report_path, report)
            return report
    report["screening"] = evaluate_screening_decision(report["arm_results"])
    report["status"] = report["screening"]["verdict"]
    report["pilot_completed"] = all(x.get("status") == "COMPLETED" for x in report["arm_results"].values())
    report["benchmark"] = {"status": "NOT_RUN", "reason": "Run matched isolated benchmark command with verified historical source and raw B8 identity; see runbook."}
    write_json(report_path, report)
    from scripts.analyze_transfer_utility import run_analysis
    run_analysis(out)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequential four-arm pilot runner for scalar transfer-utility distillation study."
    )
    parser.add_argument("--mode", choices=["all", "preflight", "profile", "run_arm", "analyze"], default="all")
    parser.add_argument("--protocol", default="configs/transfer_utility/protocol.json", help="Path to immutable protocol JSON")
    parser.add_argument("--matrix", default="configs/transfer_utility/matrix.json", help="Path to matrix config JSON")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--synthetic", action="store_true", help="Synthetic smoke run without patient data")
    parser.add_argument("--data", help="TRAIN manifest path")
    parser.add_argument("--data-sha256", help="TRAIN manifest SHA-256")
    parser.add_argument("--val-manifest", help="VAL manifest path")
    parser.add_argument("--val-manifest-sha256", help="VAL manifest SHA-256")
    parser.add_argument("--teacher-checkpoint", help="Path to teacher checkpoint directory")
    parser.add_argument("--teacher-checkpoint-sha256", help="Expected SHA256 of teacher MANIFEST.json")
    parser.add_argument("--teacher-payload-sha256", help="Expected SHA256 of teacher weights payload")
    parser.add_argument("--teacher-cache", help="Path to CACHE_MANIFEST.json (directory accepted)")
    parser.add_argument("--teacher-cache-sha256", help="Expected SHA256 of CACHE_MANIFEST.json")
    parser.add_argument("--arm", choices=["plain_kd", "scalar_guard_original", "scalar_guard_larger_tie", "scalar_guard_cross_patient_utility", "scalar_guard_same_batch_control"])
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-jobs-check", action="store_true", help="Bypass active jobs preflight check")
    args = parser.parse_args()

    if (args.skip_preflight or args.skip_jobs_check) and not args.synthetic:
        parser.error("real pilot preflight cannot be bypassed")
    protocol_p = Path(args.protocol).resolve()
    matrix_p = Path(args.matrix).resolve()
    out_p = Path(args.out).resolve()

    if args.mode == "preflight":
        protocol = read_json(protocol_p)
        receipt = check_preflight(
            protocol=protocol,
            data_manifest=args.data,
            data_manifest_sha256=args.data_sha256,
            out_dir=out_p,
            device=args.device,
            synthetic=args.synthetic,
            check_jobs=not args.skip_jobs_check,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_checkpoint_sha256=args.teacher_checkpoint_sha256,
            teacher_payload_sha256=args.teacher_payload_sha256,
            teacher_cache=args.teacher_cache,
            teacher_cache_sha256=args.teacher_cache_sha256,
            val_manifest=args.val_manifest, val_manifest_sha256=args.val_manifest_sha256,
        )
        print(json.dumps(receipt, indent=2))
        return 0 if receipt["status"] == "PASS" else 2

    elif args.mode in ("profile", "run_arm"):
        if not args.synthetic:
            preflight = check_preflight(protocol=read_json(protocol_p), data_manifest=args.data,
                data_manifest_sha256=args.data_sha256, out_dir=out_p, device=args.device,
                synthetic=False, check_jobs=True, teacher_checkpoint=args.teacher_checkpoint,
                teacher_checkpoint_sha256=args.teacher_checkpoint_sha256,
                teacher_payload_sha256=args.teacher_payload_sha256, teacher_cache=args.teacher_cache,
                teacher_cache_sha256=args.teacher_cache_sha256, val_manifest=args.val_manifest,
                val_manifest_sha256=args.val_manifest_sha256)
            if preflight["status"] != "PASS":
                print(json.dumps(preflight, indent=2))
                return 2
        if args.mode == "run_arm":
            if not args.arm or args.arm == "scalar_guard_same_batch_control":
                parser.error("run_arm requires one of the four primary arms; conditional control is dispatched by all mode")
            result = execute_arm(args.arm, ROOT / f"configs/transfer_utility/{args.arm}.json", out_p,
                data_manifest=args.data, data_manifest_sha256=args.data_sha256,
                val_manifest=args.val_manifest, val_manifest_sha256=args.val_manifest_sha256,
                device=args.device, synthetic=args.synthetic, teacher_cache=args.teacher_cache,
                teacher_cache_sha256=args.teacher_cache_sha256)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "COMPLETED" else 2
        receipt = run_bounded_profile(
            config_path=ROOT / "configs/transfer_utility/profile_32steps.json",
            out_dir=out_p,
            device=args.device,
            synthetic=args.synthetic,
            data_manifest=args.data,
            data_manifest_sha256=args.data_sha256,
            max_steps=32, teacher_cache=args.teacher_cache, teacher_cache_sha256=args.teacher_cache_sha256,
        )
        print(json.dumps(receipt, indent=2))
        return 0 if receipt["status"] == "PASS" else 2

    elif args.mode == "analyze":
        report_p = out_p / "TRANSFER_UTILITY_PILOT_REPORT.json"
        if not report_p.is_file():
            print(f"Report not found at {report_p}")
            return 2
        report = read_json(report_p)
        print(json.dumps(report.get("screening", {}), indent=2))
        return 0

    else:
        report = orchestrate_pilot(
            protocol_path=protocol_p,
            matrix_path=matrix_p,
            out_dir=out_p,
            data_manifest=args.data,
            data_manifest_sha256=args.data_sha256,
            val_manifest=args.val_manifest,
            val_manifest_sha256=args.val_manifest_sha256,
            device=args.device,
            synthetic=args.synthetic,
            skip_preflight=args.skip_preflight,
            skip_jobs_check=args.skip_jobs_check,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_checkpoint_sha256=args.teacher_checkpoint_sha256,
            teacher_payload_sha256=args.teacher_payload_sha256,
            teacher_cache=args.teacher_cache,
            teacher_cache_sha256=args.teacher_cache_sha256,
        )
        print(json.dumps({"status": report["status"], "screening": report.get("screening")}, indent=2))
        return 0 if report["status"] in ("PILOT_GO", "NO_GO", "PASS", "SYNTHETIC_PASS") else 2


if __name__ == "__main__":
    sys.exit(main())
