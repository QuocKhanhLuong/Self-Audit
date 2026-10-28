#!/usr/bin/env python3
"""Run and evaluate the 10-arm fully automatic head pilot matrix.

Fixed arms contracted in docs/AUTOMATIC_PILOT_CONTRACT.md:
  1. scalar_brier
  2. joint_brier
  3. scalar_dice
  4. joint_dice
  5. frozen
  6. support_only
  7. plain_head_u
  8. extra_supervised
  9. smaller_lr
  10. full_ema

All arms branch from the exact same verified support-only warm-start checkpoint.
Never invents a duplicate trainer; all training runs dispatch through train.py.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import torch
from typing import Any, Mapping, Sequence
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.automatic import (
    CHECKPOINT_SCHEMA,
    DEFAULT_AUTOMATIC_CONFIG,
    NAMED_ARMS,
    SCHEMA,
    automatic_source_digest,
    validate_automatic_config,
    _verify_and_read_checkpoint,
)

FIXED_ARMS = (
    "scalar_brier",
    "joint_brier",
    "scalar_dice",
    "joint_dice",
    "frozen",
    "support_only",
    "plain_head_u",
    "extra_supervised",
    "smaller_lr",
    "full_ema",
)

PRO4000_MAX_VRAM_MB = 24576.0


def compute_file_sha256(path: str | Path) -> str:
    """Compute sha256 of file in 1MB chunks."""
    path = Path(path).resolve()
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_registry(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the automatic pilot configuration registry."""
    dir_path = Path(config_dir or ROOT / "configs/automatic_pilot").resolve()
    reg_path = dir_path / "REGISTRY.json"
    if not reg_path.is_file():
        raise FileNotFoundError(f"Missing registry at {reg_path}")
    data = json.loads(reg_path.read_text())
    if data.get("schema") != "self_audit.automatic_registry.v1":
        raise ValueError(f"Unexpected registry schema: {data.get('schema')}")
    return data


def load_arm_config(arm_name: str, config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load config for a specific arm from configs/automatic_pilot/{arm_name}.json."""
    if arm_name not in FIXED_ARMS:
        raise ValueError(f"Unknown arm '{arm_name}'; must be one of {FIXED_ARMS}")
    dir_path = Path(config_dir or ROOT / "configs/automatic_pilot").resolve()
    cfg_path = dir_path / f"{arm_name}.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config for arm '{arm_name}' at {cfg_path}")
    data = json.loads(cfg_path.read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"Arm config {cfg_path} must have schema {SCHEMA}")
    if data.get("arm") != arm_name:
        raise ValueError(f"Config arm mismatch: expected {arm_name}, got {data.get('arm')}")
    # Enforce strict adherence to engine config schema
    allowed_keys = set(DEFAULT_AUTOMATIC_CONFIG.keys())
    extra = set(data.keys()) - allowed_keys
    if extra:
        raise ValueError(f"Arm config {cfg_path} contains unexpected fields: {extra}")
    return data


def validate_matrix_configs(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Validate all 10 arm configurations against contract invariants and engine schema."""
    reg = load_registry(config_dir)
    report: dict[str, Any] = {
        "status": "PASS",
        "total_arms": len(FIXED_ARMS),
        "arms": {},
        "shared_invariants": {},
    }
    configs = {}
    for arm in FIXED_ARMS:
        cfg = load_arm_config(arm, config_dir)
        validate_automatic_config(cfg, allow_full_run=True)
        configs[arm] = cfg
        meta = reg["arms"].get(arm, {})
        report["arms"][arm] = {
            "arm": cfg.get("arm"),
            "allocation": meta.get("allocation"),
            "dice_guard": meta.get("dice_guard"),
            "pseudo_lr": cfg.get("pseudo_lr"),
            "body_lr": cfg.get("body_lr"),
            "head_lr": cfg.get("head_lr"),
            "steps": cfg.get("steps"),
        }

    # Verify distinct arm contracts
    reg_arms = reg["arms"]
    assert reg_arms["scalar_brier"]["allocation"] == "scalar" and not reg_arms["scalar_brier"]["dice_guard"]
    assert reg_arms["joint_brier"]["allocation"] == "joint" and not reg_arms["joint_brier"]["dice_guard"]
    assert reg_arms["scalar_dice"]["allocation"] == "scalar" and reg_arms["scalar_dice"]["dice_guard"]
    assert reg_arms["joint_dice"]["allocation"] == "joint" and reg_arms["joint_dice"]["dice_guard"]
    assert configs["frozen"]["pseudo_lr"] == 0.0
    assert configs["support_only"]["pseudo_lr"] == 0.0
    assert reg_arms["plain_head_u"]["allocation"] == "plain" and not reg_arms["plain_head_u"]["dice_guard"]
    assert configs["extra_supervised"]["pseudo_lr"] == 0.0
    assert configs["smaller_lr"]["pseudo_lr"] < configs["plain_head_u"]["pseudo_lr"]
    assert configs["full_ema"]["body_lr"] > 0.0

    # Verify shared baseline hyperparameters
    ref_seed = configs["joint_dice"]["seed"]
    ref_model = configs["joint_dice"]["model"]["backbone"]
    ref_window = configs["joint_dice"]["window"]["mode"]
    for arm, cfg in configs.items():
        if cfg.get("seed") != ref_seed:
            raise ValueError(f"Seed mismatch in arm {arm}: {cfg.get('seed')} != {ref_seed}")
        if cfg.get("model", {}).get("backbone") != ref_model:
            raise ValueError(f"Backbone mismatch in arm {arm}")
        if cfg.get("window", {}).get("mode") != ref_window:
            raise ValueError(f"Window mode mismatch in arm {arm}")

    report["shared_invariants"] = {
        "seed": ref_seed,
        "backbone": ref_model,
        "window_mode": ref_window,
        "support_warm_start_shared": True,
    }
    return report


def verify_warm_start_checkpoint(
    checkpoint_folder: str | Path,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify shared support-only warm-start checkpoint manifest and payload hash.

    Enforces:
      - Valid MANIFEST.json matching CHECKPOINT_SCHEMA
      - Expected manifest SHA-256 (if provided)
      - Phase == 'warm_start'
      - Arm == 'support_only'
      - Prediction_model == 'student'
      - Valid non-empty source_sha256 present
      - Payload weights.pt matching payload_sha256
      - Non-empty target_label_lineage.patients
    """
    if not expected_sha256:
        raise ValueError("Expected warm-start MANIFEST SHA-256 is required")
    ckpt_dir = Path(checkpoint_folder).resolve()
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Warm-start checkpoint directory does not exist: {ckpt_dir}")

    manifest_file = ckpt_dir / "MANIFEST.json"
    if not manifest_file.is_file():
        if (ckpt_dir / "final" / "MANIFEST.json").is_file():
            ckpt_dir = ckpt_dir / "final"
            manifest_file = ckpt_dir / "MANIFEST.json"
        else:
            raise FileNotFoundError(f"Missing checkpoint manifest: {manifest_file}")

    manifest_sha = compute_file_sha256(manifest_file)
    if expected_sha256 and manifest_sha != expected_sha256:
        raise ValueError(
            f"Warm-start manifest SHA-256 mismatch: expected {expected_sha256}, got {manifest_sha}"
        )

    manifest = json.loads(manifest_file.read_text())
    schema = manifest.get("schema")
    if schema != CHECKPOINT_SCHEMA:
        raise ValueError(f"Unrecognized checkpoint manifest schema: {schema}")

    if manifest.get("phase") != "warm_start":
        raise ValueError(
            f"Warm-start checkpoint phase must be 'warm_start', got {manifest.get('phase')!r}"
        )

    if manifest.get("arm") != "support_only":
        raise ValueError(
            f"Warm-start checkpoint arm must be 'support_only', got {manifest.get('arm')!r}"
        )

    if manifest.get("prediction_model") != "student":
        raise ValueError(
            f"Warm-start checkpoint prediction_model must be 'student', got {manifest.get('prediction_model')!r}"
        )

    source_sha = manifest.get("source_sha256")
    if not source_sha or not isinstance(source_sha, str):
        raise ValueError("Warm-start checkpoint must declare non-empty source_sha256")

    weights_file = ckpt_dir / "weights.pt"
    if not weights_file.is_file():
        raise FileNotFoundError(f"Missing checkpoint weights file: {weights_file}")

    actual_weights_sha = compute_file_sha256(weights_file)
    if not manifest.get("payload_sha256") or actual_weights_sha != manifest.get("payload_sha256"):
        raise ValueError(
            f"Payload weights SHA-256 mismatch: manifest specifies {manifest.get('payload_sha256')}, got {actual_weights_sha}"
        )

    lineage = manifest.get("target_label_lineage", {})
    patients = lineage.get("patients", [])
    if not lineage or not patients:
        raise ValueError("Warm-start checkpoint lineage must declare non-empty target patients")

    # Reuse the engine's manifest-only pin, source and lineage checks.
    _verify_and_read_checkpoint(ckpt_dir, expected_sha256)
    declared_budget = manifest["declared_target_budget"]

    return {
        "status": "VERIFIED",
        "checkpoint_folder": str(ckpt_dir),
        "manifest_sha256": manifest_sha,
        "weights_sha256": actual_weights_sha,
        "schema": schema,
        "phase": manifest.get("phase"),
        "arm": manifest.get("arm"),
        "prediction_model": manifest.get("prediction_model"),
        "source_sha256": source_sha,
        "manifest": manifest,
        "lineage": lineage,
        "declared_target_budget": declared_budget,
    }


def invoke_trainer(
    config_path: str | Path,
    manifest_path: str | Path,
    manifest_sha256: str,
    out_dir: str | Path,
    *,
    device: str = "cuda",
    synthetic: bool = False,
    allow_full_run: bool = False,
    steps: int | None = None,
    resume: str | None = None,
    resume_sha256: str | None = None,
) -> dict[str, Any]:
    """Invoke canonical train.py entrypoint via subprocess."""
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--config",
        str(Path(config_path).resolve()),
        "--data",
        str(Path(manifest_path).resolve()),
        "--data-sha256",
        manifest_sha256,
        "--out",
        str(Path(out_dir).resolve()),
        "--device",
        device,
    ]
    if synthetic:
        cmd.append("--synthetic")
    if allow_full_run:
        cmd.append("--allow-full-run")
    if steps is not None:
        cmd.extend(["--steps", str(steps)])
    if resume is not None:
        cmd.extend(["--resume", str(resume)])
    if resume_sha256 is not None:
        cmd.extend(["--resume-sha256", resume_sha256])

    if Path(out_dir).exists():
        raise ValueError("Trainer output must be a new directory")
    start_time = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    subprocess_wall_time = time.perf_counter() - start_time

    out_path = Path(out_dir).resolve()
    if proc.returncode != 0:
        raise RuntimeError(
            f"train.py failed with exit code {proc.returncode}:\n{proc.stderr}\n{proc.stdout}"
        )
    (out_path / "train_stdout.log").write_text(proc.stdout)
    (out_path / "train_stderr.log").write_text(proc.stderr)

    # Engine produces REPORT.json in out_dir
    report_file = out_path / "REPORT.json"
    if not report_file.is_file():
        raise RuntimeError("train.py exited without REPORT.json; execution is unverified")
    receipt = json.loads(report_file.read_text())
    if receipt.get("status") != "COMPLETED":
        raise RuntimeError("train.py did not report a completed run")

    receipt["subprocess_wall_seconds"] = subprocess_wall_time
    return receipt


def compute_resource_accounting(
    arm_receipt: dict[str, Any],
    warm_start_receipt: dict[str, Any] | None = None,
    evaluation_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute cold total cost vs marginal adaptation cost with disjoint measured components.

    No fabricated defaults: missing metrics remain strictly None or NOT_MEASURED.
    Cold total cost is INCOMPLETE if mask export or warm-start has not run (never converted to 0).
    """
    marginal_adaptation_seconds = (
        arm_receipt.get("marginal_adaptation_seconds")
        if arm_receipt.get("marginal_adaptation_seconds") is not None
        else arm_receipt.get("elapsed_seconds")
    )
    if marginal_adaptation_seconds is not None:
        marginal_adaptation_seconds = float(marginal_adaptation_seconds)
    else:
        marginal_adaptation_seconds = arm_receipt.get("subprocess_wall_seconds")
        if marginal_adaptation_seconds is not None:
            marginal_adaptation_seconds = float(marginal_adaptation_seconds)

    extraction_seconds = arm_receipt.get("extraction_seconds")
    if extraction_seconds is not None:
        extraction_seconds = float(extraction_seconds)
    else:
        extraction_seconds = "NOT_MEASURED"

    def _safe_float(val: Any) -> float | str:
        if val is None or val == "NOT_MEASURED":
            return "NOT_MEASURED"
        try:
            f = float(val)
            return f if math.isfinite(f) else "NOT_MEASURED"
        except (ValueError, TypeError):
            return "NOT_MEASURED"

    throughput = _safe_float(arm_receipt.get("throughput_updates_per_sec"))
    if throughput == "NOT_MEASURED":
        steps = arm_receipt.get("steps")
        if steps is not None and marginal_adaptation_seconds and marginal_adaptation_seconds > 0:
            throughput = float(steps) / marginal_adaptation_seconds

    p50_latency_ms = _safe_float(arm_receipt.get("latency_p50_ms"))
    p95_latency_ms = _safe_float(arm_receipt.get("latency_p95_ms"))

    if p50_latency_ms == "NOT_MEASURED" or p95_latency_ms == "NOT_MEASURED":
        step_latencies = arm_receipt.get("step_latencies_ms")
        if step_latencies and isinstance(step_latencies, list):
            arr = sorted(float(x) for x in step_latencies if math.isfinite(float(x)))
            if arr:
                if p50_latency_ms == "NOT_MEASURED":
                    p50_latency_ms = arr[int(round(0.50 * (len(arr) - 1)))]
                if p95_latency_ms == "NOT_MEASURED":
                    p95_latency_ms = arr[int(round(0.95 * (len(arr) - 1)))]

    peak_vram_alloc_mb = _safe_float(arm_receipt.get("peak_vram_allocated_mb"))
    if peak_vram_alloc_mb == "NOT_MEASURED":
        peak_vram_alloc_bytes = arm_receipt.get("peak_allocated_bytes")
        if peak_vram_alloc_bytes is not None and peak_vram_alloc_bytes != "NOT_MEASURED":
            try:
                peak_vram_alloc_mb = float(peak_vram_alloc_bytes) / (1024 * 1024)
            except (ValueError, TypeError):
                peak_vram_alloc_mb = "NOT_MEASURED"

    peak_vram_res_mb = _safe_float(arm_receipt.get("peak_vram_reserved_mb"))
    if peak_vram_res_mb == "NOT_MEASURED":
        peak_vram_res_bytes = arm_receipt.get("peak_reserved_bytes")
        if peak_vram_res_bytes is not None and peak_vram_res_bytes != "NOT_MEASURED":
            try:
                peak_vram_res_mb = float(peak_vram_res_bytes) / (1024 * 1024)
            except (ValueError, TypeError):
                peak_vram_res_mb = "NOT_MEASURED"

    pseudo_proposals = arm_receipt.get("pseudo_proposals")
    pseudo_accepted = arm_receipt.get("pseudo_accepted")
    acceptance_rate = arm_receipt.get("acceptance_rate")
    if acceptance_rate is not None:
        if isinstance(acceptance_rate, (int, float)):
            acceptance_rate = float(acceptance_rate)
    elif pseudo_proposals is not None and pseudo_accepted is not None:
        if int(pseudo_proposals) > 0:
            acceptance_rate = float(pseudo_accepted) / float(pseudo_proposals)
        else:
            acceptance_rate = "NOT_APPLICABLE"
    else:
        acceptance_rate = "NOT_MEASURED"

    # Measured displacement updates from engine report, never equated to pseudo_accepted
    nonzero_displacement_updates = arm_receipt.get("nonzero_displacement_updates")
    if nonzero_displacement_updates is not None and nonzero_displacement_updates != "NOT_MEASURED":
        try:
            nonzero_displacement_updates = int(nonzero_displacement_updates)
        except (ValueError, TypeError):
            nonzero_displacement_updates = "NOT_MEASURED"
    else:
        nonzero_displacement_updates = "NOT_MEASURED"

    margin_changes = arm_receipt.get("margin_changes")
    if margin_changes is None:
        margin_changes = "NOT_MEASURED"

    marginal_cost = {
        "marginal_adaptation_seconds": marginal_adaptation_seconds,
        "extraction_seconds": extraction_seconds,
        "throughput_updates_per_sec": throughput,
        "latency_p50_ms": p50_latency_ms,
        "latency_p95_ms": p95_latency_ms,
        "peak_vram_allocated_mb": peak_vram_alloc_mb,
        "peak_vram_reserved_mb": peak_vram_res_mb,
        "acceptance_rate": acceptance_rate,
        "nonzero_displacement_updates": nonzero_displacement_updates,
        "margin_changes": margin_changes,
    }

    warm_start_seconds = None
    if warm_start_receipt:
        ws_elapsed = warm_start_receipt.get("elapsed_seconds") or warm_start_receipt.get("marginal_adaptation_seconds")
        if ws_elapsed is not None:
            warm_start_seconds = float(ws_elapsed)
        elif "subprocess_wall_seconds" in warm_start_receipt:
            warm_start_seconds = float(warm_start_receipt["subprocess_wall_seconds"])

    # Disjoint cold total cost components
    output_export_seconds = None
    if evaluation_receipt is not None:
        eval_sec = evaluation_receipt.get("output_export_seconds") or evaluation_receipt.get("total_seconds")
        if eval_sec is not None:
            output_export_seconds = float(eval_sec)

    if (
        output_export_seconds is not None
        and warm_start_seconds is not None
        and marginal_adaptation_seconds is not None
        and not (isinstance(arm_receipt.get("step_latencies_ms"), list)
                 and arm_receipt.get("steps", 0) > len(arm_receipt["step_latencies_ms"]))
    ):
        cold_total_seconds = warm_start_seconds + marginal_adaptation_seconds + output_export_seconds
        cold_cost_status = "COMPLETE"
    else:
        cold_total_seconds = "INCOMPLETE"
        cold_cost_status = "INCOMPLETE"

    cold_total_cost = {
        "status": cold_cost_status,
        "warm_start_seconds": warm_start_seconds if warm_start_seconds is not None else "NOT_MEASURED",
        "marginal_adaptation_seconds": marginal_adaptation_seconds if marginal_adaptation_seconds is not None else "NOT_MEASURED",
        "output_export_seconds": output_export_seconds if output_export_seconds is not None else "NOT_MEASURED",
        "cold_total_seconds": cold_total_seconds,
        "resume_cost_policy": "Fresh phase clocks only; resumed checkpoint timing omits the previous checkpoint write boundary",
    }

    return {
        "marginal_adaptation_cost": marginal_cost,
        "cold_total_cost": cold_total_cost,
    }


def evaluate_decision_gate(
    arm_metrics: Mapping[str, Mapping[str, Any]],
    *,
    synthetic: bool = False,
    verified_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate research Go / No-Go protocol strictly according to contract.

    Requires all of:
      1. Real execution (synthetic=False, quality_status MEASURED)
      2. Positive verified context:
         - Non-empty query provenance (query_provenance)
         - Raw student prediction model (prediction_model == "student")
         - 100% output coverage (output_coverage == 1.0)
         - Matched compute across arms (matched_compute is True)
         - Matched initialization across arms (matched_initialization is True)
         - Matched lineage provenance across arms (matched_lineage is True)
      3. Valid finite scalar metrics for all required arms (0.0 <= f <= 1.0; no NaN/inf).
         Note: Dice 0.0 is a valid finite outcome.
      4. Three contracted inequality gates:
         - joint_dice > scalar_dice
         - joint_dice > smaller_lr
         - joint_dice > extra_supervised

    If synthetic: strictly returns NO_GO_SOFTWARE_ONLY.
    If context is missing/unverified or metrics are missing/invalid: returns INCONCLUSIVE_NOT_RUN.
    """
    if synthetic:
        return {
            "decision": "NO_GO_SOFTWARE_ONLY",
            "quality_status": "NOT_RUN",
            "reason": "Software-only or synthetic fixture execution strictly cannot yield research GO; real patient verification required.",
            "gates": {
                "joint_beats_scalar": False,
                "dice_guard_beats_smaller_lr": False,
                "u_beats_extra_supervised": False,
            },
        }

    if verified_context is None:
        return {
            "decision": "INCONCLUSIVE_NOT_RUN",
            "quality_status": "NOT_RUN",
            "reason": "Positive verified_context is required for decision gate; context missing.",
            "gates": {
                "joint_beats_scalar": None,
                "dice_guard_beats_smaller_lr": None,
                "u_beats_extra_supervised": None,
            },
        }

    ctx = verified_context

    # 1. Query provenance check
    if not ctx.get("query_provenance"):
        return {
            "decision": "INCONCLUSIVE_NOT_RUN",
            "quality_status": "NOT_RUN",
            "reason": "Missing positive query_provenance in verified_context.",
            "gates": {"joint_beats_scalar": None, "dice_guard_beats_smaller_lr": None, "u_beats_extra_supervised": None},
        }

    # 2. Prediction model check: must be raw student
    if ctx.get("prediction_model") != "student":
        return {
            "decision": "INCONCLUSIVE_NOT_RUN",
            "quality_status": "NOT_RUN",
            "reason": f"Prediction model must be 'student', got {ctx.get('prediction_model')!r}.",
            "gates": {"joint_beats_scalar": None, "dice_guard_beats_smaller_lr": None, "u_beats_extra_supervised": None},
        }

    # 3. Output coverage check: 100% required
    coverage = ctx.get("output_coverage")
    try:
        cov_val = float(coverage) if coverage is not None else -1.0
        if not math.isfinite(cov_val) or cov_val != 1.0:
            return {
                "decision": "INCONCLUSIVE_NOT_RUN",
                "quality_status": "NOT_RUN",
                "reason": f"Incomplete output coverage ({coverage}); 100% output coverage required.",
                "gates": {"joint_beats_scalar": None, "dice_guard_beats_smaller_lr": None, "u_beats_extra_supervised": None},
            }
    except (ValueError, TypeError):
        return {
            "decision": "INCONCLUSIVE_NOT_RUN",
            "quality_status": "NOT_RUN",
            "reason": "Invalid output coverage metadata in verified_context.",
            "gates": {"joint_beats_scalar": None, "dice_guard_beats_smaller_lr": None, "u_beats_extra_supervised": None},
        }

    # 4. Matched compute check
    if ctx.get("matched_compute") is not True:
        return {
            "decision": "INCONCLUSIVE_NOT_RUN",
            "quality_status": "NOT_RUN",
            "reason": "Matched compute verification failed or missing in verified_context.",
            "gates": {"joint_beats_scalar": None, "dice_guard_beats_smaller_lr": None, "u_beats_extra_supervised": None},
        }

    # 5. Matched initialization check
    if ctx.get("matched_initialization") is not True:
        return {
            "decision": "INCONCLUSIVE_NOT_RUN",
            "quality_status": "NOT_RUN",
            "reason": "Matched initialization verification failed or missing in verified_context.",
            "gates": {"joint_beats_scalar": None, "dice_guard_beats_smaller_lr": None, "u_beats_extra_supervised": None},
        }

    # 6. Matched lineage check
    if ctx.get("matched_lineage") is not True:
        return {
            "decision": "INCONCLUSIVE_NOT_RUN",
            "quality_status": "NOT_RUN",
            "reason": "Matched lineage verification failed or missing in verified_context.",
            "gates": {"joint_beats_scalar": None, "dice_guard_beats_smaller_lr": None, "u_beats_extra_supervised": None},
        }

    # Extract patient-macro FG Dice scores (0.0 is a valid finite outcome)
    def get_valid_dice(arm_name: str) -> float | None:
        metrics = arm_metrics.get(arm_name, {})
        val = (
            metrics.get("query_patient_macro_fg_dice")
            if metrics.get("query_patient_macro_fg_dice") is not None
            else metrics.get("foreground_patient_macro")
        )
        if val is None and "receipt" in metrics and isinstance(metrics["receipt"], dict):
            val = (
                metrics["receipt"].get("query_patient_macro_fg_dice")
                if metrics["receipt"].get("query_patient_macro_fg_dice") is not None
                else metrics["receipt"].get("foreground_patient_macro")
            )
        if val is None or val in ("NOT_MEASURED", "NOT_RUN", "INCOMPLETE"):
            return None
        try:
            f = float(val)
            if not math.isfinite(f) or math.isnan(f) or not (0.0 <= f <= 1.0):
                return None
            return f
        except (ValueError, TypeError):
            return None

    dice_joint = get_valid_dice("joint_dice")
    dice_scalar = get_valid_dice("scalar_dice")
    dice_smaller_lr = get_valid_dice("smaller_lr")
    dice_extra_sup = get_valid_dice("extra_supervised")

    missing = []
    if dice_joint is None:
        missing.append("joint_dice")
    if dice_scalar is None:
        missing.append("scalar_dice")
    if dice_smaller_lr is None:
        missing.append("smaller_lr")
    if dice_extra_sup is None:
        missing.append("extra_supervised")

    if missing:
        return {
            "decision": "INCONCLUSIVE_NOT_RUN",
            "quality_status": "NOT_RUN",
            "missing_arms": missing,
            "reason": f"Independent query metrics missing or invalid for arms: {missing}. Quality endpoints remain NOT_RUN.",
            "gates": {
                "joint_beats_scalar": None,
                "dice_guard_beats_smaller_lr": None,
                "u_beats_extra_supervised": None,
            },
        }

    # Per-class primary results (RV, MYO, LV)
    per_class_results: dict[str, dict[str, Any]] = {}
    for arm_name in ("joint_dice", "scalar_dice", "smaller_lr", "extra_supervised"):
        arm_data = arm_metrics.get(arm_name, {})
        cls_data = (
            arm_data.get("per_class")
            or arm_data.get("macro_patient")
            or (arm_data.get("receipt", {}).get("macro_patient") if isinstance(arm_data.get("receipt"), dict) else None)
            or {}
        )
        per_class_results[arm_name] = {}
        for cname in ("RV", "MYO", "LV"):
            c_val = cls_data.get(cname)
            if isinstance(c_val, dict):
                per_class_results[arm_name][cname] = c_val.get("dice", "NOT_MEASURED")
            elif isinstance(c_val, (int, float)):
                per_class_results[arm_name][cname] = float(c_val)
            else:
                per_class_results[arm_name][cname] = "NOT_MEASURED"

    matched_compute_details = {
        "status": "VERIFIED_MATCHED",
        "comparison_mode": ctx.get("comparison_mode", "matched"),
        "steps": ctx.get("steps"),
        "time_budget_seconds": ctx.get("time_budget_seconds"),
        "device": ctx.get("device"),
        "hardware": ctx.get("hardware", "PRO4000 24GB"),
    }

    if any(not isinstance(v, (int, float)) or isinstance(v, bool)
           or not math.isfinite(v) or not 0 <= v <= 1
           for classes in per_class_results.values() for v in classes.values()):
        return {"decision": "INCONCLUSIVE_NOT_RUN", "quality_status": "NOT_RUN",
                "reason": "A primary foreground class endpoint is unavailable.",
                "per_class_primary_results": per_class_results}

    g1 = dice_joint > dice_scalar
    g2 = dice_joint > dice_smaller_lr
    g3 = dice_joint > dice_extra_sup

    passed_all = g1 and g2 and g3
    decision = "GO" if passed_all else "NO_GO"

    return {
        "decision": decision,
        "quality_status": "MEASURED",
        "gates": {
            "joint_beats_scalar": bool(g1),
            "dice_guard_beats_smaller_lr": bool(g2),
            "u_beats_extra_supervised": bool(g3),
        },
        "scores": {
            "joint_dice": dice_joint,
            "scalar_dice": dice_scalar,
            "smaller_lr": dice_smaller_lr,
            "extra_supervised": dice_extra_sup,
        },
        "per_class_primary_results": per_class_results,
        "matched_compute": matched_compute_details,
        "reason": (
            "All 3 independent query gates passed under verified matched compute."
            if passed_all
            else "One or more independent query gates failed."
        ),
    }


def run_warm_start_pipeline(
    out_dir: Path,
    manifest_path: Path,
    manifest_sha256: str,
    *,
    support_patients: list[str],
    device: str = "cuda",
    synthetic: bool = False,
    allow_full_run: bool = False,
    steps: int = 8,
    model_config: dict[str, Any] | None = None,
    pseudo_threshold: float | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    """Run support_only arm to create the shared support-only warm-start checkpoint."""
    ws_out = out_dir / "warm_start_support_only"
    ws_cfg = load_arm_config("support_only")
    ws_cfg["phase"] = "warm_start"
    ws_cfg["steps"] = steps
    ws_cfg["support"]["patient_ids"] = list(support_patients)
    ws_cfg["initialization"]["declared_target_budget"] = len(support_patients)
    ws_cfg["initialization"]["warm_start_checkpoint"] = None
    ws_cfg["initialization"]["warm_start_sha256"] = None
    if model_config is not None:
        ws_cfg["model"] = copy.deepcopy(model_config)
    if pseudo_threshold is not None:
        ws_cfg["pseudo_threshold"] = pseudo_threshold
    if synthetic:
        ws_cfg["model"]["backbone"] = "tiny_test"

    ws_cfg_path = out_dir / "warm_start_config.json"
    ws_cfg_path.write_text(json.dumps(ws_cfg, indent=2) + "\n")

    receipt = invoke_trainer(
        ws_cfg_path,
        manifest_path,
        manifest_sha256,
        ws_out,
        device=device,
        synthetic=synthetic,
        allow_full_run=allow_full_run,
        steps=steps,
    )

    final_dir = ws_out / "final"
    manifest_path_file = final_dir / "MANIFEST.json"
    if not manifest_path_file.is_file():
        raise FileNotFoundError(f"train.py did not produce warm-start checkpoint manifest at {manifest_path_file}")

    weights_path = final_dir / "weights.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f"train.py did not produce warm-start weights payload at {weights_path}")

    ws_manifest_sha = compute_file_sha256(manifest_path_file)
    return final_dir, ws_manifest_sha, receipt


def run_experiment_matrix(
    arms: Sequence[str],
    manifest_path: str | Path,
    manifest_sha256: str,
    out_dir: str | Path,
    *,
    comparison: str = "equal_update",
    time_budget: float | None = None,
    steps: int = 32,
    device: str = "cuda",
    synthetic: bool = False,
    allow_full_run: bool = False,
    warm_start_checkpoint: str | Path | None = None,
    warm_start_sha256: str | None = None,
    model_config: dict[str, Any] | None = None,
    pseudo_threshold: float | None = None,
    max_steps: int = 1000,
) -> dict[str, Any]:
    """Execute the configured subset of arms under equal-update or equal-time constraints."""
    out_path = Path(out_dir).resolve()
    # Output safety: never write inside protected source trees
    protected_roots = [
        ROOT / "self_audit",
        ROOT / "configs",
        ROOT / "docs",
        ROOT / "scripts",
        ROOT / "tests",
    ]
    for prot in protected_roots:
        if out_path == prot or prot in out_path.parents:
            raise ValueError(f"Output directory {out_path} cannot be inside protected source tree {prot}")

    manifest_path = Path(manifest_path).resolve()
    actual_manifest_sha = compute_file_sha256(manifest_path)
    if actual_manifest_sha != manifest_sha256:
        raise ValueError(
            f"Manifest SHA-256 mismatch: expected {manifest_sha256}, got {actual_manifest_sha}"
        )

    manifest_data = json.loads(manifest_path.read_text())
    prepare_output(out_path, manifest_path, manifest_data, warm_start_checkpoint)
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(FIXED_ARMS):
        raise ValueError("arms must be a nonempty unique subset of the fixed matrix")
    if comparison not in {"equal_update", "equal_time"}:
        raise ValueError("invalid comparison mode")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be positive")
    labeled_records = manifest_data.get("labeled", [])
    if not labeled_records:
        raise ValueError("Training manifest has no labeled records for support")

    all_labeled_ids = [str(r["id"]) for r in labeled_records]

    # Step 1: Ensure shared support-only warm-start and lineage accounting
    ws_receipt: dict[str, Any] | None = None
    if warm_start_checkpoint:
        ws_info = verify_warm_start_checkpoint(warm_start_checkpoint, warm_start_sha256)
        ws_path = Path(ws_info["checkpoint_folder"])
        ws_sha = ws_info["manifest_sha256"]
        prior_lineage_patients = ws_info.get("lineage", {}).get("patients", [])
        support_patients = list(all_labeled_ids)
        touched_patients = sorted(list(set(prior_lineage_patients) | set(support_patients)))
        declared_budget = len(touched_patients)
    else:
        support_patients = list(all_labeled_ids)
        ws_path, ws_sha, ws_receipt = run_warm_start_pipeline(
            out_path,
            manifest_path,
            manifest_sha256,
            support_patients=support_patients,
            device=device,
            synthetic=synthetic,
            allow_full_run=allow_full_run,
            steps=min(steps, 8) if not allow_full_run else steps,
            model_config=model_config,
            pseudo_threshold=pseudo_threshold,
        )
        declared_budget = len(support_patients)

    # Step 2: If equal-time mode without explicit budget, run reference baseline to measure time
    effective_time_budget: float | None = time_budget
    if comparison == "equal_time" and effective_time_budget is None:
        ref_arm = "plain_head_u" if "plain_head_u" in arms else arms[0]
        ref_cfg = load_arm_config(ref_arm)
        ref_cfg["phase"] = "adaptation"
        ref_cfg["support"]["patient_ids"] = list(support_patients)
        ref_cfg["initialization"]["warm_start_checkpoint"] = str(ws_path)
        ref_cfg["initialization"]["warm_start_sha256"] = ws_sha
        ref_cfg["initialization"]["declared_target_budget"] = declared_budget
        ref_cfg["steps"] = steps
        if model_config is not None:
            ref_cfg["model"] = copy.deepcopy(model_config)
        if pseudo_threshold is not None:
            ref_cfg["pseudo_threshold"] = pseudo_threshold
        if synthetic:
            ref_cfg["model"]["backbone"] = "tiny_test"

        ref_cfg_path = out_path / f"{ref_arm}_reference_time_cfg.json"
        ref_cfg_path.write_text(json.dumps(ref_cfg, indent=2) + "\n")
        ref_receipt = invoke_trainer(
            ref_cfg_path,
            manifest_path,
            manifest_sha256,
            out_path / f"{ref_arm}_time_ref",
            device=device,
            synthetic=synthetic,
            allow_full_run=allow_full_run,
            steps=steps,
        )
        measured_time = (
            ref_receipt.get("elapsed_seconds")
            or ref_receipt.get("marginal_adaptation_seconds")
            or ref_receipt.get("subprocess_wall_seconds")
        )
        if measured_time is None or float(measured_time) <= 0:
            raise RuntimeError(
                f"Failed to measure actual time budget from reference baseline; got {measured_time}"
            )
        effective_time_budget = float(measured_time)

    # Step 3: Run all selected arms
    results: dict[str, Any] = {
        "schema": "self_audit.automatic_pilot_report.v1",
        "comparison_mode": comparison,
        "target_hardware": "PRO4000 24GB",
        "device": device,
        "synthetic": synthetic,
        "environment": {"python": platform.python_version(), "system": platform.platform(),
                        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                        "cuda_available": torch.cuda.is_available(),
                        "device_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else platform.machine()},
        "training_manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "warm_start": {
            "checkpoint_folder": str(ws_path),
            "manifest_sha256": ws_sha,
            "declared_target_budget": declared_budget,
            "receipt": ws_receipt,
        },
        "arms": {},
    }

    all_equal_time_certified = True
    for arm_name in arms:
        arm_out = out_path / arm_name
        cfg = load_arm_config(arm_name)
        cfg["phase"] = "adaptation"
        cfg["support"]["patient_ids"] = list(support_patients)
        cfg["initialization"]["warm_start_checkpoint"] = str(ws_path)
        cfg["initialization"]["warm_start_sha256"] = ws_sha
        cfg["initialization"]["declared_target_budget"] = declared_budget
        if model_config is not None:
            cfg["model"] = copy.deepcopy(model_config)
        if pseudo_threshold is not None:
            cfg["pseudo_threshold"] = pseudo_threshold
        if synthetic:
            cfg["model"]["backbone"] = "tiny_test"

        arm_steps = steps
        if comparison == "equal_time":
            cfg["time_budget_seconds"] = effective_time_budget
            cfg["steps"] = max_steps
        else:
            cfg["steps"] = arm_steps

        arm_cfg_path = out_path / f"{arm_name}_resolved_config.json"
        arm_cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

        arm_receipt = invoke_trainer(
            arm_cfg_path,
            manifest_path,
            manifest_sha256,
            arm_out,
            device=device,
            synthetic=synthetic,
            allow_full_run=allow_full_run,
            steps=arm_steps if comparison == "equal_update" else None,
        )

        if comparison == "equal_time":
            arm_certified = bool(arm_receipt.get("equal_time_certified", False))
            if not arm_certified:
                all_equal_time_certified = False

        accounting = compute_resource_accounting(
            arm_receipt,
            warm_start_receipt=ws_receipt,
            evaluation_receipt=None,
        )

        results["arms"][arm_name] = {
            "receipt": arm_receipt,
            "output_directory": str(arm_out),
            "report_sha256": compute_file_sha256(arm_out / "REPORT.json"),
            "config_sha256": compute_file_sha256(arm_cfg_path),
            "accounting": accounting,
        }

    if comparison == "equal_time":
        measured = [v["receipt"]["phase_cost_seconds"] for v in results["arms"].values()]
        positive = all(v["receipt"]["steps"] > 0 for v in results["arms"].values())
        results["all_time_budgets_reached"] = all_equal_time_certified
        results["equal_time_certified"] = bool(all_equal_time_certified and positive
                                               and max(measured) <= 1.05 * min(measured))
        results["wall_time_match_tolerance_fraction"] = 0.05
        results["effective_time_budget_seconds"] = effective_time_budget

    # Step 4: Decision gate
    gate_result = evaluate_decision_gate(results["arms"], synthetic=synthetic)
    results["decision_gate"] = gate_result

    return results


def prepare_output(out: Path, manifest: Path, data: dict, checkpoint=None) -> None:
    """Validate a new run destination before writing configs or trainer logs."""
    out = out.resolve()
    if out.exists():
        raise ValueError(f"Output directory must be new: {out}")
    protected = [manifest.resolve().parent]
    for r in data.get("labeled", []) + data.get("unlabeled", []):
        paths = [r["image"]] + [m["path"] for m in r.get("masks", {}).values()]
        for p in paths:
            p = Path(p)
            protected.append((p if p.is_absolute() else manifest.parent / p).resolve().parent)
    if checkpoint:
        protected.append(Path(checkpoint).resolve())
    for p in protected:
        if out == p or p in out.parents or out in p.parents:
            raise ValueError(f"Output overlaps data/checkpoint directory: {p}")
    out.mkdir(parents=True, exist_ok=False)


def _read_pinned(path: Path, sha: str) -> dict:
    if not sha or compute_file_sha256(path) != sha:
        raise ValueError(f"Receipt hash mismatch: {path}")
    return json.loads(path.read_text())


def collate_evaluations(index_path: str | Path, index_sha256: str) -> dict[str, Any]:
    """Verify an offline receipt chain, complete costs, and assess pilot contrasts.

    The index pins the matrix summary and each evaluation METRICS.json. Evaluation
    receipts pin predictions, recipe, query manifest and the trained checkpoint.
    This reads frozen outputs, never supplies reference masks to a learner.
    """
    index_path = Path(index_path).resolve()
    index = _read_pinned(index_path, index_sha256)
    if index.get("schema") != "self_audit.automatic_evaluation_index.v1":
        raise ValueError("Invalid evaluation index schema")
    resolve = lambda p: (index_path.parent / p).resolve()
    mr = index["matrix_report"]
    matrix_path = resolve(mr["path"])
    matrix = _read_pinned(matrix_path, mr["sha256"])
    if matrix.get("schema") != "self_audit.automatic_pilot_report.v1":
        raise ValueError("Invalid matrix report schema")
    required = ("joint_dice", "scalar_dice", "smaller_lr", "extra_supervised")
    missing = [a for a in required if a not in index.get("evaluations", {})]
    if missing:
        return {"decision_gate": {"decision": "INCONCLUSIVE_NOT_RUN",
                "quality_status": "NOT_RUN", "missing_arms": missing}}
    ws = matrix["warm_start"]
    verify_warm_start_checkpoint(ws["checkpoint_folder"], ws["manifest_sha256"])
    arms, configs, receipts, cohorts, queries, train_shas = {}, [], [], [], [], []
    synthetic = bool(matrix.get("synthetic"))
    coverage_ok = True
    for arm, item in index["evaluations"].items():
        if arm not in matrix["arms"]:
            raise ValueError(f"Evaluation arm is absent from matrix: {arm}")
        run_item = matrix["arms"][arm]
        run_dir = Path(run_item["output_directory"])
        report = _read_pinned(run_dir / "REPORT.json", run_item["report_sha256"])
        if report != {k: v for k, v in run_item["receipt"].items() if k != "subprocess_wall_seconds"}:
            raise ValueError("Matrix/report receipt contents differ")
        ckpt_dir = run_dir / "final"
        _, ckpt, _, _ = _verify_and_read_checkpoint(ckpt_dir, report["final_manifest_sha256"])
        cfg = ckpt["config"]
        if ckpt["arm"] != arm or ckpt["phase"] != "adaptation":
            raise ValueError("Checkpoint arm/phase mismatch")
        if cfg["initialization"]["warm_start_sha256"] != ws["manifest_sha256"]:
            raise ValueError("Arms do not share the pinned warm-start")
        ev_dir = resolve(item["directory"])
        metrics = _read_pinned(ev_dir / "METRICS.json", item["metrics_sha256"])
        frozen = _read_pinned(ev_dir / "PREDICTIONS_FROZEN.json", metrics["frozen_receipt_sha256"])
        recipe = _read_pinned(ev_dir / "RECIPE.json", frozen["recipe_sha256"])
        for info in frozen["files"]:
            if Path(info["file"]).name != info["file"]:
                raise ValueError("Invalid prediction receipt filename")
            if compute_file_sha256(ev_dir / "predictions" / info["file"]) != info["sha256"]:
                raise ValueError("Frozen prediction hash mismatch")
        if recipe["checkpoint_sha256"] != report["final_manifest_sha256"] or frozen["checkpoint_sha256"] != report["final_manifest_sha256"]:
            raise ValueError("Evaluation belongs to a different checkpoint")
        if recipe.get("prediction_model") != "raw_student" or recipe["data_sha256"] != frozen["data_sha256"]:
            raise ValueError("Invalid raw-student evaluation recipe")
        synthetic |= bool(metrics.get("synthetic")) or bool(ckpt.get("synthetic"))
        coverage_ok &= (metrics.get("status") == "COMPLETED"
                        and metrics.get("role") == "external_frozen"
                        and recipe.get("role") == "external_frozen"
                        and metrics.get("prediction_coverage") == 1.0
                        and metrics.get("scoring_coverage") == 1.0
                        and frozen.get("coverage") == 1.0)
        query_ids = {f["id"] for f in frozen["files"]}
        if query_ids & (set(ckpt["labeled_ids"]) | set(ckpt["unlabeled_ids"])):
            raise ValueError("Query overlaps TRAIN lineage")
        queries.append(frozen["data_sha256"])
        cohorts.append(sorted((f["patient"], f["frame"]) for f in metrics.get("frames", [])))
        train_shas.append(ckpt["training_manifest_sha256"])
        configs.append({k: cfg[k] for k in (
            "seed", "head_lr", "weight_decay", "ema_decay", "pseudo_threshold", "ridge",
            "risk_budget", "guard_tolerance", "dice_tolerance", "fp_tolerance", "backtracks",
            "batch_size", "crop_size", "support", "initialization", "window", "model")})
        receipts.append(report)
        arms[arm] = {**metrics, "accounting": compute_resource_accounting(
            report, ws.get("receipt"), metrics)}
    same = lambda xs: bool(xs) and all(x == xs[0] for x in xs)
    matched_init = same(configs)
    matched_lineage = same(train_shas) and same(queries) and same(cohorts)
    # Equal outer updates are reported separately; they do not equal wall compute.
    # A declared 5% wall-time tolerance is fixed here before any query comparison.
    times = [r["phase_cost_seconds"] for r in receipts]
    budgets = [r.get("time_budget_seconds") for r in receipts]
    matched_compute = (matrix.get("comparison_mode") == "equal_time" and same(budgets)
                       and all(isinstance(t, (int, float)) and math.isfinite(t) and t > 0 for t in times)
                       and all(r.get("equal_time_certified") and r["steps"] > 0 for r in receipts)
                       and all(r["steps"] == len(r.get("step_latencies_ms", [])) for r in receipts)
                       and max(times) <= 1.05 * min(times))
    ctx = {"query_provenance": queries[0], "prediction_model": "student",
           "output_coverage": 1.0 if coverage_ok else 0.0,
           "matched_initialization": matched_init, "matched_lineage": matched_lineage,
           "matched_compute": matched_compute, "comparison_mode": matrix["comparison_mode"],
           "time_budget_seconds": budgets[0], "device": matrix["device"]}
    return {"schema": "self_audit.automatic_collation.v1", "arms": arms,
            "verified_context": ctx, "actual_phase_seconds": times,
            "wall_time_match_tolerance_fraction": 0.05,
            "decision_gate": evaluate_decision_gate(arms, synthetic=synthetic, verified_context=ctx)}



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["matrix", "warm_start", "smoke", "profile", "run", "evaluate"], default="smoke")
    parser.add_argument("--arms", default="all")
    parser.add_argument("--comparison", choices=["equal_update", "equal_time"], default="equal_update")
    parser.add_argument("--time-budget", type=float)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000, help="Explicit safety cap for equal-time runs")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--data")
    parser.add_argument("--data-sha256")
    parser.add_argument("--model-config", help="Pinned local backbone source/commit/weights JSON; no downloads")
    parser.add_argument("--pseudo-threshold", type=float, help="Recorded shared override; use 0 for manufactured nonempty-candidate smoke")
    parser.add_argument("--out")
    parser.add_argument("--warm-start-checkpoint")
    parser.add_argument("--warm-start-sha256")
    parser.add_argument("--report-path")
    parser.add_argument("--evaluation-index")
    parser.add_argument("--evaluation-index-sha256")
    parser.add_argument("--validate-configs", action="store_true")
    args = parser.parse_args()
    if args.validate_configs or args.mode == "matrix":
        print(json.dumps(validate_matrix_configs(), indent=2, allow_nan=False))
        return
    if args.mode == "evaluate":
        if not all([args.evaluation_index, args.evaluation_index_sha256, args.report_path]):
            parser.error("evaluate requires --evaluation-index, --evaluation-index-sha256 and a NEW --report-path")
        target = Path(args.report_path).resolve()
        if target.exists():
            parser.error("Evaluation report path must be new; matrix summary is never overwritten")
        report = collate_evaluations(args.evaluation_index, args.evaluation_index_sha256)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report["decision_gate"], indent=2, allow_nan=False))
        return
    arms = list(FIXED_ARMS) if args.arms == "all" else args.arms.split(",")
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(FIXED_ARMS):
        parser.error("--arms must be unique names from the fixed matrix")
    if args.steps < 1 or args.max_steps < 1:
        parser.error("Step counts must be positive")
    synthetic = args.synthetic or args.mode == "smoke"
    device = "cpu" if synthetic else args.device
    if not synthetic and args.steps > 8 and not args.allow_full_run:
        parser.error("More than 8 patient steps requires --allow-full-run")
    if args.comparison == "equal_time" and args.time_budget is not None:
        if not math.isfinite(args.time_budget) or args.time_budget <= 0:
            parser.error("--time-budget must be positive and finite")
    model_config = json.loads(Path(args.model_config).read_text()) if args.model_config else None
    out = Path(args.out).resolve() if args.out else Path(tempfile.gettempdir()) / ("automatic-pilot-" + uuid.uuid4().hex)
    if out.exists():
        parser.error("--out must be a new directory")
    manifest = Path(args.data).resolve() if args.data else None
    data_sha = args.data_sha256
    if manifest is None:
        if not synthetic:
            parser.error("Patient runs require --data and --data-sha256")
        from self_audit.data import synthetic_fixture
        # Separate fixture and output trees, so destination safety remains enforced.
        fixture = Path(tempfile.mkdtemp(prefix="automatic-pilot-fixture-")) / "data"
        manifest, data_sha = synthetic_fixture(fixture, seed=42)
    if not data_sha:
        parser.error("--data-sha256 is required")
    data = _read_pinned(manifest, data_sha)
    if not synthetic and model_config is None and not args.warm_start_checkpoint:
        parser.error("Cold patient initialization requires --model-config with pre-existing source and weights")
    if model_config is None and args.warm_start_checkpoint:
        ws = verify_warm_start_checkpoint(args.warm_start_checkpoint, args.warm_start_sha256)
        model_config = copy.deepcopy(ws["manifest"]["config"]["model"])
    if args.mode == "warm_start":
        if args.warm_start_checkpoint:
            parser.error("warm_start creates a new checkpoint; use run/profile to branch from an existing one")
        prepare_output(out, manifest, data)
        ws_path, ws_sha, receipt = run_warm_start_pipeline(
            out, manifest, data_sha, support_patients=[r["id"] for r in data["labeled"]],
            device=device, synthetic=synthetic, allow_full_run=args.allow_full_run,
            steps=args.steps, model_config=model_config, pseudo_threshold=args.pseudo_threshold)
        report = {"schema": "self_audit.automatic_warm_start_report.v1",
                  "checkpoint_folder": str(ws_path), "manifest_sha256": ws_sha, "receipt": receipt}
    else:
        report = run_experiment_matrix(
            arms, manifest, data_sha, out, comparison=args.comparison,
            time_budget=args.time_budget, steps=min(args.steps, 2) if args.mode == "smoke" else args.steps,
            device=device, synthetic=synthetic, allow_full_run=args.allow_full_run or synthetic,
            warm_start_checkpoint=args.warm_start_checkpoint,
            warm_start_sha256=args.warm_start_sha256, model_config=model_config,
            pseudo_threshold=args.pseudo_threshold, max_steps=args.max_steps)
    target = Path(args.report_path).resolve() if args.report_path else out / "summary.json"
    if target.exists():
        parser.error("Report path already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
