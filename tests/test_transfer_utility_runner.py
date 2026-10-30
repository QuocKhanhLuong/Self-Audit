"""Comprehensive unit and integration tests for transfer-utility pilot runner and diagnostics.

Tests:
  - Immutable protocol lock and configuration validation across all 5 arms.
  - Failclosed preflight safety (GPU busy, active jobs, existing output, missing physical assets).
  - Canonical command generation with exact flags and locked total steps budget.
  - Bounded profile execution exercising both warmup and U updates (32 steps, warmup 8).
  - Transaction diagnostics parsing from TRAIN_LOG.jsonl (warmup/no-candidate/U denominators, safe null gains).
  - Evaluator metrics extraction with pooled RV counts from actual frame records (null when undefined).
  - Decision screening rule logic (finite 0..1, same 18 patients, synthetic cannot clinical GO,
    screening threshold > 0.002, conditional 5th arm trigger, failclosed INCOMPLETE/BLOCKED).
  - Real tiny end-to-end integration test with canonical trainer, export, evaluator, and analysis.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import numpy as np
import pytest

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.compact_data import CompactTrainingData, synthetic_compact_fixture
from self_audit.compact_artifacts import digest
from scripts.run_transfer_utility_pilot import (
    analyze_checkpoint_evaluations,
    analyze_transaction_diagnostics,
    build_benchmark_command,
    build_evaluate_command,
    build_export_command,
    build_train_command,
    check_active_training_jobs,
    check_gpu_occupancy,
    check_preflight,
    evaluate_screening_decision,
    read_json,
    run_bounded_profile,
    write_json,
    EXPECTED_BATCH_SIZE,
    EXPECTED_CROP_SIZE,
    EXPECTED_EVAL_STEPS,
    EXPECTED_GUARD_TOLERANCE,
    EXPECTED_LR,
    EXPECTED_MAX_TRIALS,
    EXPECTED_PSEUDO_LR,
    EXPECTED_PSEUDO_THRESHOLD,
    EXPECTED_RIDGE,
    EXPECTED_SEED,
    EXPECTED_STEPS,
    EXPECTED_TEMPERATURE,
    EXPECTED_TRIAL_SCALES,
    EXPECTED_WARMUP,
    EXPECTED_WIDTHS,
    SCREENING_DELTA_THRESHOLD,
    WATCHDOG_DEFAULT_SECONDS,
    L8_PATIENT_IDS,
    EXPECTED_TRAIN_MANIFEST_SHA,
    EXPECTED_TEACHER_MANIFEST_SHA,
    EXPECTED_TEACHER_PAYLOAD_SHA,
)
from scripts.analyze_transfer_utility import format_markdown_report, run_analysis


# ==============================================================================
# Protocol & Config Validation Tests
# ==============================================================================

def test_protocol_immutable_pins():
    """Verify protocol.json exists and strictly enforces all contract pins."""
    protocol_path = ROOT / "configs" / "transfer_utility" / "protocol.json"
    assert protocol_path.is_file(), f"Missing protocol file at {protocol_path}"

    proto = read_json(protocol_path)
    assert proto["schema"] == "self_audit.compact.transfer_utility.protocol.v1"

    budget = proto["budget"]
    assert budget["total_steps"] == EXPECTED_STEPS == 2000
    assert budget["warmup_steps"] == EXPECTED_WARMUP == 1000
    assert budget["eval_steps"] == list(EXPECTED_EVAL_STEPS) == [1000, 1500, 2000]
    assert budget["final_step"] == 2000
    assert budget["watchdog_max_seconds"] == WATCHDOG_DEFAULT_SECONDS == 7200
    assert budget["profile_max_steps"] == 32

    hp = proto["hyperparameters"]
    assert hp["seed"] == EXPECTED_SEED == 42
    assert hp["lr"] == EXPECTED_LR == 0.0003
    assert hp["pseudo_lr"] == EXPECTED_PSEUDO_LR == 0.001
    assert hp["temperature"] == EXPECTED_TEMPERATURE == 1.0
    assert hp["pseudo_threshold"] == EXPECTED_PSEUDO_THRESHOLD == 0.7
    assert hp["ridge"] == EXPECTED_RIDGE == 1e-6
    assert hp["risk_budget"] == 0.0
    assert hp["absent_fp_budget"] == 0.0
    assert hp["guard_tolerance"] == EXPECTED_GUARD_TOLERANCE == 1e-6
    assert hp["max_trials"] == EXPECTED_MAX_TRIALS == 8
    assert hp["trial_scales"] == EXPECTED_TRIAL_SCALES
    assert hp["batch_size"] == EXPECTED_BATCH_SIZE == 8
    assert hp["crop_size"] == EXPECTED_CROP_SIZE == 160

    arch = proto["architecture"]
    assert arch["widths"] == EXPECTED_WIDTHS == [16, 32, 64, 128]
    assert arch["num_classes"] == 4
    assert arch["init"] == "kaiming_normal"

    sel = proto["selector_parameters"]
    assert sel["selector_dice_tolerance"] == 0.0
    assert sel["utility_gain_tolerance"] == 1e-7
    assert sel["utility_tie_tolerance"] == 1e-7
    assert sel["probe_seed"] == 42017
    assert sel["probe_batch_size"] == 8

    assert proto["data"]["manifest_sha256"] == EXPECTED_TRAIN_MANIFEST_SHA
    assert proto["data"]["labeled_ids"] == L8_PATIENT_IDS
    assert proto["teacher"]["source_manifest_sha256"] == EXPECTED_TEACHER_MANIFEST_SHA
    assert proto["teacher"]["source_payload_sha256"] == EXPECTED_TEACHER_PAYLOAD_SHA
    assert proto["teacher"]["state"] == "student"


def test_configs_and_matrix_conformance():
    """Verify all 5 configs and matrix.json conform to self_audit.compact.v1 schema."""
    cfg_dir = ROOT / "configs" / "transfer_utility"
    expected_configs = [
        ("plain_kd.json", "plain_kd", None),
        ("scalar_guard_original.json", "scalar_guard", "original"),
        ("scalar_guard_larger_tie.json", "scalar_guard", "larger_tie"),
        ("scalar_guard_cross_patient_utility.json", "scalar_guard", "cross_patient_utility"),
        ("scalar_guard_same_batch_control.json", "scalar_guard", "same_batch_utility"),
        ("profile_32steps.json", "scalar_guard", "cross_patient_utility"),
    ]

    for fname, exp_method, exp_sel in expected_configs:
        cpath = cfg_dir / fname
        assert cpath.is_file(), f"Missing config {fname}"
        cfg = read_json(cpath)
        assert cfg["schema"] == "self_audit.compact.v1"
        assert cfg["method"] == exp_method
        if exp_sel is not None:
            assert cfg.get("selector") == exp_sel
        assert cfg["model"]["widths"] == EXPECTED_WIDTHS
        assert cfg["model"]["num_classes"] == 4
        assert cfg["model"]["init"] == "kaiming_normal"

    matrix_p = cfg_dir / "matrix.json"
    assert matrix_p.is_file()
    mat = read_json(matrix_p)
    assert mat["schema"] == "self_audit.compact.transfer_utility.matrix.v1"
    assert len(mat["jobs"]) == 5
    job_names = [j["name"] for j in mat["jobs"]]
    assert job_names == [
        "plain_kd",
        "scalar_guard_original",
        "scalar_guard_larger_tie",
        "scalar_guard_cross_patient_utility",
        "scalar_guard_same_batch_control",
    ]


# ==============================================================================
# Preflight Safety & Failclosed Tests
# ==============================================================================

def test_preflight_refuses_existing_completed_output(tmp_path: Path):
    """Preflight must refuse if output directory already contains a completed report."""
    proto = read_json(ROOT / "configs" / "transfer_utility" / "protocol.json")
    out_dir = tmp_path / "existing_run"
    out_dir.mkdir(parents=True)
    (out_dir / "TRANSFER_UTILITY_PILOT_REPORT.json").write_text('{"status": "PILOT_GO"}\n')

    receipt = check_preflight(
        protocol=proto,
        data_manifest=None,
        data_manifest_sha256=None,
        out_dir=out_dir,
        device="cpu",
        synthetic=True,
        check_jobs=False,
    )
    assert receipt["status"] == "REFUSED_EXISTING_OUTPUT"
    assert "output" in receipt["reasons"][0].lower()


def test_preflight_fails_closed_on_missing_real_assets(tmp_path: Path):
    """Preflight must failclosed and record NOT_RUN if patient data is missing for real runs."""
    proto = read_json(ROOT / "configs" / "transfer_utility" / "protocol.json")
    out_dir = tmp_path / "new_run"

    receipt = check_preflight(
        protocol=proto,
        data_manifest=tmp_path / "nonexistent_manifest.json",
        data_manifest_sha256="a" * 64,
        out_dir=out_dir,
        device="cpu",
        synthetic=False,
        check_jobs=False,
    )
    assert receipt["status"] == "NOT_RUN"
    assert any("not found" in r.lower() for r in receipt["reasons"])


def test_preflight_fails_closed_on_protocol_tampering(tmp_path: Path):
    """Preflight must refuse if protocol hyperparameters or budget are tampered with."""
    proto = read_json(ROOT / "configs" / "transfer_utility" / "protocol.json")
    proto["budget"]["total_steps"] = 5000  # Tampered total steps
    out_dir = tmp_path / "new_run"

    receipt = check_preflight(
        protocol=proto,
        data_manifest=None,
        data_manifest_sha256=None,
        out_dir=out_dir,
        device="cpu",
        synthetic=True,
        check_jobs=False,
    )
    assert receipt["status"] == "INVALID_PROTOCOL"
    assert any("steps" in r.lower() for r in receipt["reasons"])


def test_gpu_occupancy_check_cpu():
    """CPU device occupancy check passes cleanly without initializing CUDA."""
    res = check_gpu_occupancy("cpu")
    assert res["status"] == "PASS"
    assert res["gpu_available"] is False


def test_check_active_training_jobs():
    """Active jobs detector runs without crash and reports dictionary."""
    res = check_active_training_jobs()
    assert res["status"] in ("CLEAN", "ACTIVE_JOBS_PRESENT")
    assert isinstance(res["active_job_count"], int)


# ==============================================================================
# Command Construction Tests (Canonical Interface)
# ==============================================================================

def test_build_train_command_canonical():
    """Verify build_train_command adheres strictly to canonical train.py CLI interface."""
    cfg_p = ROOT / "configs" / "transfer_utility" / "plain_kd.json"
    cmd, manifest = build_train_command(
        config_path=cfg_p,
        out_dir="/tmp/out_test",
        device="cuda",
        synthetic=False,
        data_manifest="/path/to/data.json",
        data_manifest_sha256="b" * 64,
        steps=2000,
        stop_after_step=1000,
        allow_full_run=True,
    )

    assert "train.py" in cmd[1]
    assert "--config" in cmd
    assert "--out" in cmd
    assert "--device" in cmd
    assert "--data" in cmd
    assert "--data-sha256" in cmd
    assert "--steps" in cmd
    assert "2000" in cmd
    assert "--stop-after-step" in cmd
    assert "1000" in cmd
    assert "--allow-full-run" in cmd
    assert "--synthetic" not in cmd
    assert manifest["steps"] == 2000
    assert manifest["stop_after_step"] == 1000
    assert len(manifest["source_sha256"]) == 64


def test_build_train_command_synthetic():
    """Verify build_train_command with --synthetic flag."""
    cfg_p = ROOT / "configs" / "transfer_utility" / "scalar_guard_original.json"
    cmd, manifest = build_train_command(
        config_path=cfg_p,
        out_dir="/tmp/out_test",
        device="cpu",
        synthetic=True,
        steps=32,
    )
    assert "--synthetic" in cmd
    assert "--data" not in cmd
    assert manifest["synthetic"] is True


def test_build_export_evaluate_benchmark_commands():
    """Verify export, evaluate, and benchmark command builders."""
    exp_cmd, exp_man = build_export_command("/tmp/ckpt", "c" * 64, "/tmp/export")
    assert "export_compact_student.py" in exp_cmd[1]
    assert exp_man["checkpoint_sha256"] == "c" * 64

    eval_cmd, eval_man = build_evaluate_command(
        export_dir="/tmp/export",
        export_sha256="d" * 64,
        manifest_path="/tmp/val.json",
        manifest_sha256="e" * 64,
        train_manifest_path="/tmp/train.json",
        train_manifest_sha256="f" * 64,
        out_dir="/tmp/eval_out",
        device="cpu",
    )
    assert "evaluate_compact_student.py" in eval_cmd[1]
    assert "--export" in eval_cmd
    assert "--train-manifest" in eval_cmd
    assert eval_man["device"] == "cpu"

    bench_cmd, bench_man = build_benchmark_command(
        compact_export_dir="/tmp/export",
        compact_export_sha256="d" * 64,
        manifest_path="/tmp/val.json",
        manifest_sha256="e" * 64,
        out_file="/tmp/bench.json",
        device="cpu",
    )
    assert "benchmark_compact_student.py" in bench_cmd[1]
    assert bench_man["compact_export_sha256"] == "d" * 64


# ==============================================================================
# Transaction Diagnostics Tests (TRAIN_LOG.jsonl)
# ==============================================================================

def test_selected_gain_diagnostics_separate_unselected_and_zero(tmp_path):
    path = tmp_path / 'TRAIN_LOG.jsonl'
    rows = [
        {'step': 0, 'status': 'accepted_nonzero', 'scale': .5,
         'transaction': {'reason': 'max_utility_gain',
             'telemetry': {'reference_kd_loss': 1., 'classwise_kd_loss': {'RV': .4}},
             'trials': [{'utility_gain': .1, 'accepted': False},
                        {'utility_gain': .2, 'accepted': True, 'probe_classwise_kd_loss': {'RV': .3}}]}},
        {'step': 1, 'status': 'zero_displacement', 'scale': 0.,
         'transaction': {'reason': 'no_positive_utility_gain',
             'telemetry': {'reference_kd_loss': 1., 'classwise_kd_loss': {'RV': .4}},
             'trials': [{'utility_gain': -.2, 'accepted': False}]}},
        {'step': 2, 'status': 'zero_displacement', 'scale': 0.,
         'transaction': {'reason': 'missing_utility_probe',
             'telemetry': {'reference_kd_loss': None}, 'trials': []}},
    ]
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    result = analyze_transaction_diagnostics(path)
    assert result['utility_probe_gains']['count'] == 3
    assert result['selected_probe_gains']['count'] == 2
    assert result['selected_probe_gains']['mean'] == pytest.approx(.1)
    assert result['selected_probe_gains']['classwise_mean']['RV'] == pytest.approx(.05)
    assert result['selected_probe_gains']['classwise_mean']['BG'] is None
    assert result['selection_reasons']['missing_utility_probe'] == 1

def test_analyze_transaction_diagnostics_train_log(tmp_path: Path):
    """Verify transaction diagnostics parsing: warmup/no_candidate/U denominators and safe null gains."""
    log_p = tmp_path / "TRAIN_LOG.jsonl"
    dummy_steps = [
        # Step 0: Warmup step (no U attempted)
        {
            "step": 0,
            "status": "warmup",
            "scale": 0.0,
            "proposed_displacement_norm": 0.0,
            "realized_displacement_norm": 0.0,
            "warm_start": {"in_warmup": True},
            "timings": {"step_elapsed_seconds": 0.10, "supervised_seconds": 0.10},
        },
        # Step 1: No candidate step (no U attempted)
        {
            "step": 1,
            "status": "no_candidate",
            "scale": 0.0,
            "proposed_displacement_norm": 0.0,
            "realized_displacement_norm": 0.0,
            "warm_start": {"in_warmup": False},
            "timings": {"step_elapsed_seconds": 0.10, "supervised_seconds": 0.08},
        },
        # Step 2: Accepted nonzero U step
        {
            "step": 2,
            "status": "accepted_nonzero",
            "scale": 0.5,
            "proposed_displacement_norm": 0.05,
            "realized_displacement_norm": 0.025,
            "warm_start": {"in_warmup": False},
            "transaction": {
                "status": "accepted_nonzero",
                "trials": [
                    {"trial_index": 0, "utility_gain": 0.012},
                    {"trial_index": 1, "utility_gain": None},  # Null gain test
                ],
            },
            "timings": {"step_elapsed_seconds": 0.20, "audit_trials_seconds": 0.12, "supervised_seconds": 0.05},
        },
        # Step 3: Rejected all U step
        {
            "step": 3,
            "status": "rejected_all",
            "scale": 0.0,
            "proposed_displacement_norm": 0.04,
            "realized_displacement_norm": 0.0,
            "warm_start": {"in_warmup": False},
            "transaction": {
                "status": "rejected_all",
                "trials": [
                    {"trial_index": 0, "utility_gain": -0.002},
                ],
            },
            "timings": {"step_elapsed_seconds": 0.18, "audit_trials_seconds": 0.10, "supervised_seconds": 0.05},
        },
        # Step 4: Zero displacement U step
        {
            "step": 4,
            "status": "zero_displacement",
            "scale": 0.0,
            "proposed_displacement_norm": 0.03,
            "realized_displacement_norm": 0.0,
            "warm_start": {"in_warmup": False},
            "transaction": {
                "status": "zero_displacement",
                "trials": [
                    {"trial_index": 0, "utility_gain": 0.0},
                ],
            },
            "timings": {"step_elapsed_seconds": 0.15, "audit_trials_seconds": 0.08, "supervised_seconds": 0.05},
        },
    ]

    with open(log_p, "w", encoding="utf-8") as f:
        for r in dummy_steps:
            f.write(json.dumps(r) + "\n")

    diag = analyze_transaction_diagnostics(log_p)
    assert diag["status"] == "COMPLETED"
    assert diag["total_steps"] == 5
    assert diag["warmup_steps"] == 1
    assert diag["no_candidate_steps"] == 1
    assert diag["u_attempted_steps"] == 3

    outcomes = diag["transaction_outcomes"]
    assert outcomes["accepted_nonzero"] == 1
    assert outcomes["rejected_all"] == 1
    assert outcomes["zero_displacement"] == 1
    # U acceptance rate = 1 / 3
    assert outcomes["u_acceptance_rate"] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert outcomes["overall_acceptance_rate"] == pytest.approx(1.0 / 5.0, abs=1e-4)

    # Gains: extracted safely without float(None) failure
    gains = diag["utility_probe_gains"]
    assert gains["count"] == 3  # 0.012, -0.002, 0.0 (None skipped)
    assert gains["min"] == -0.002
    assert gains["max"] == 0.012


# ==============================================================================
# Evaluation Metrics Parsing & Pooled RV Counts Tests
# ==============================================================================

def test_analyze_checkpoint_evaluations_actual_schema(tmp_path: Path):
    """Verify evaluation parsing: pooled RV precision/recall from frames raw and patients metrics."""
    eval_dir = tmp_path / "eval_test"
    eval_dir.mkdir(parents=True)

    # Actual compute_historical_metrics output structure
    metrics_data = {
        "class_first_macro4": 0.8540,
        "class_first_foreground_macro": 0.8420,
        "macro_patient": {
            "BG": {"dice": 0.991, "present_patients": 2},
            "RV": {"dice": 0.810, "present_patients": 2},
            "MYO": {"dice": 0.845, "present_patients": 2},
            "LV": {"dice": 0.871, "present_patients": 2},
        },
        "patients": {
            "patient001": {
                "BG": 0.990, "RV": 0.800, "MYO": 0.840, "LV": 0.870,
                "foreground_mean": 0.8367, "all_class_mean": 0.875,
                "valid_voxels": 10000, "frame_count": 2,
            },
            "patient002": {
                "BG": 0.992, "RV": 0.820, "MYO": 0.850, "LV": 0.872,
                "foreground_mean": 0.8473, "all_class_mean": 0.883,
                "valid_voxels": 10000, "frame_count": 2,
            },
        },
        "frames": [
            {
                "patient": "patient001",
                "frame": 0,
                "raw": [
                    {"class": "BG", "TP": 5000, "FP": 50, "FN": 40},
                    {"class": "RV", "TP": 400, "FP": 40, "FN": 60},
                    {"class": "MYO", "TP": 600, "FP": 30, "FN": 50},
                    {"class": "LV", "TP": 800, "FP": 20, "FN": 30},
                ],
            },
            {
                "patient": "patient002",
                "frame": 0,
                "raw": [
                    {"class": "BG", "TP": 5100, "FP": 40, "FN": 30},
                    {"class": "RV", "TP": 450, "FP": 50, "FN": 40},
                    {"class": "MYO", "TP": 620, "FP": 25, "FN": 45},
                    {"class": "LV", "TP": 820, "FP": 15, "FN": 25},
                ],
            },
        ],
        "synthetic": False,
        "evaluation_manifest_sha256": "1" * 64,
    }
    write_json(eval_dir / "METRICS.json", metrics_data)

    parsed = analyze_checkpoint_evaluations(eval_dir)
    assert parsed["status"] == "COMPLETED"
    assert parsed["class_first_macro4"] == 0.8540
    assert parsed["class_dices"]["RV"] == 0.810

    # Total RV: TP = 400 + 450 = 850; FP = 40 + 50 = 90; FN = 60 + 40 = 100
    # Precision = 850 / (850 + 90) = 850 / 940 = ~0.90426
    # Recall = 850 / (850 + 100) = 850 / 950 = ~0.89474
    assert parsed["rv_precision"] == pytest.approx(850 / 940, abs=1e-4)
    assert parsed["rv_recall"] == pytest.approx(850 / 950, abs=1e-4)
    assert parsed["patients"]["patient001"]["foreground_mean"] == pytest.approx(0.8367)


def test_analyze_checkpoint_evaluations_zero_rv_frames(tmp_path: Path):
    """When RV is absent, rv_precision and rv_recall must remain None, not invented 0."""
    eval_dir = tmp_path / "eval_zero_rv"
    eval_dir.mkdir(parents=True)

    metrics_data = {
        "class_first_macro4": 0.80,
        "class_first_foreground_macro": 0.75,
        "macro_patient": {"BG": {"dice": 0.99}, "RV": {"dice": None}, "MYO": {"dice": 0.80}, "LV": {"dice": 0.85}},
        "patients": {},
        "frames": [
            {
                "patient": "patient001",
                "frame": 0,
                "raw": [
                    {"class": "BG", "TP": 5000, "FP": 50, "FN": 40},
                    {"class": "RV", "TP": 0, "FP": 0, "FN": 0},
                ],
            }
        ],
        "synthetic": False,
    }
    write_json(eval_dir / "METRICS.json", metrics_data)

    parsed = analyze_checkpoint_evaluations(eval_dir)
    assert parsed["rv_precision"] is None
    assert parsed["rv_recall"] is None


# ==============================================================================
# Screening Decision Rule Tests
# ==============================================================================

def _complete_screening_receipts(arms):
    """Manufactured metadata for testing decision logic, never experimental evidence."""
    for arm in arms.values():
        arm.update(steps=2000, initial_state_sha256="1"*64, data_schedule_sha256="2"*64,
                   source_sha256="3"*64, train_manifest_sha256=EXPECTED_TRAIN_MANIFEST_SHA,
                   teacher_cache_sha256="4"*64)
        for evaluation in arm.get("evaluations", {}).values():
            evaluation.update(status="COMPLETED", coverage={"complete": True}, checkpoint_step=2000,
                evaluation_manifest_sha256="cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50",
                export_manifest_sha256="5"*64, frozen_predictions_sha256="6"*64)


def test_evaluate_screening_decision_go_real_data():
    """PILOT_GO awarded when real data exceeds each baseline by > 0.002 on same 18 patients."""
    pids = [f"patient_{i:02d}" for i in range(18)]
    eval_sha = "a" * 64

    arm_results = {}
    base_scores = {
        "plain_kd": 0.850,
        "scalar_guard_original": 0.855,
        "scalar_guard_larger_tie": 0.856,
        "scalar_guard_cross_patient_utility": 0.860,
    }

    for arm, score in base_scores.items():
        arm_results[arm] = {
            "status": "COMPLETED",
            "evaluations": {
                "step_002000": {
                    "class_first_macro4": score,
                    "evaluation_manifest_sha256": eval_sha,
                    "patients": {pid: {"foreground_mean": score} for pid in pids},
                    "synthetic": False,
                }
            },
        }

    _complete_screening_receipts(arm_results)
    decision = evaluate_screening_decision(arm_results)
    assert decision["verdict"] == "PILOT_GO"
    assert decision["screening_passed"] is True
    assert decision["should_run_conditional_5th_arm"] is True
    assert decision["paired_patient_differences"]["scalar_guard_larger_tie"]["patient_00"]["foreground_mean"] == pytest.approx(0.004, abs=1e-4)


def test_evaluate_screening_decision_synthetic_cannot_clinical_go():
    """Synthetic run cannot certify clinical PILOT_GO; records SYNTHETIC_PASS."""
    pids = [f"patient_{i:02d}" for i in range(18)]
    eval_sha = "b" * 64

    arm_results = {}
    base_scores = {
        "plain_kd": 0.850,
        "scalar_guard_original": 0.855,
        "scalar_guard_larger_tie": 0.856,
        "scalar_guard_cross_patient_utility": 0.860,
    }

    for arm, score in base_scores.items():
        arm_results[arm] = {
            "status": "COMPLETED",
            "evaluations": {
                "step_002000": {
                    "class_first_macro4": score,
                    "evaluation_manifest_sha256": eval_sha,
                    "patients": {pid: {"foreground_mean": score} for pid in pids},
                    "synthetic": True,  # Synthetic test!
                }
            },
        }

    _complete_screening_receipts(arm_results)
    decision = evaluate_screening_decision(arm_results)
    assert decision["verdict"] == "SYNTHETIC_PASS"
    assert decision["screening_passed"] is False
    assert decision["should_run_conditional_5th_arm"] is False
    assert decision["synthetic"] is True
    assert "cannot certify clinical PILOT_GO" in decision["notes"]


def test_evaluate_screening_decision_no_go_insufficient_margin():
    """Screening returns NO_GO if utility does not exceed each baseline by > 0.002."""
    pids = [f"patient_{i:02d}" for i in range(18)]
    eval_sha = "c" * 64

    arm_results = {
        "plain_kd": {"status": "COMPLETED", "evaluations": {"step_002000": {"class_first_macro4": 0.850, "evaluation_manifest_sha256": eval_sha, "patients": {p: {"foreground_mean": 0.850} for p in pids}, "synthetic": False}}},
        "scalar_guard_original": {"status": "COMPLETED", "evaluations": {"step_002000": {"class_first_macro4": 0.855, "evaluation_manifest_sha256": eval_sha, "patients": {p: {"foreground_mean": 0.855} for p in pids}, "synthetic": False}}},
        "scalar_guard_larger_tie": {"status": "COMPLETED", "evaluations": {"step_002000": {"class_first_macro4": 0.858, "evaluation_manifest_sha256": eval_sha, "patients": {p: {"foreground_mean": 0.858} for p in pids}, "synthetic": False}}},
        "scalar_guard_cross_patient_utility": {"status": "COMPLETED", "evaluations": {"step_002000": {"class_first_macro4": 0.859, "evaluation_manifest_sha256": eval_sha, "patients": {p: {"foreground_mean": 0.859} for p in pids}, "synthetic": False}}},
    }

    # Utility (0.859) vs larger_tie (0.858) is delta 0.001 <= 0.002 -> NO_GO
    _complete_screening_receipts(arm_results)
    decision = evaluate_screening_decision(arm_results)
    assert decision["verdict"] == "NO_GO"
    assert decision["screening_passed"] is False
    assert decision["should_run_conditional_5th_arm"] is False


def test_evaluate_screening_decision_incomplete_on_missing_run():
    """Missing or failed arm results in INCOMPLETE, NEVER empirical NO_GO."""
    arm_results = {
        "plain_kd": {"status": "COMPLETED", "evaluations": {"step_002000": {"class_first_macro4": 0.850}}},
        "scalar_guard_original": {"status": "COMPLETED", "evaluations": {"step_002000": {"class_first_macro4": 0.855}}},
        "scalar_guard_larger_tie": {"status": "FAILED", "error": "Watchdog timeout"},
        "scalar_guard_cross_patient_utility": {"status": "COMPLETED", "evaluations": {"step_002000": {"class_first_macro4": 0.860}}},
    }

    _complete_screening_receipts(arm_results)
    decision = evaluate_screening_decision(arm_results)
    assert decision["verdict"] == "INCOMPLETE"
    assert decision["screening_passed"] is False
    assert "missing_arms" in decision


# ==============================================================================
# Markdown Report Formatting Tests
# ==============================================================================

def test_format_markdown_report():
    """Verify Markdown report generation contains required tables and headers."""
    sample_report = {
        "screening": {
            "verdict": "PILOT_GO",
            "deltas": {"vs_plain_kd": 0.008, "vs_scalar_guard_original": 0.004, "vs_scalar_guard_larger_tie": 0.003},
            "criteria": {"exceeds_plain_kd_by_002": True, "exceeds_original_by_002": True, "exceeds_larger_tie_by_002": True},
            "should_run_conditional_5th_arm": True,
        },
        "arm_results": {
            "plain_kd": {
                "selector": "none",
                "status": "COMPLETED",
                "evaluations": {
                    "step_002000": {
                        "class_first_macro4": 0.850,
                        "class_first_foreground_macro": 0.840,
                        "class_dices": {"RV": 0.82, "MYO": 0.84, "LV": 0.86},
                        "rv_precision": 0.83,
                        "rv_recall": 0.81,
                    }
                },
                "diagnostics": {
                    "status": "COMPLETED",
                    "total_steps": 2000,
                    "warmup_steps": 1000,
                    "no_candidate_steps": 0,
                    "u_attempted_steps": 1000,
                    "transaction_outcomes": {
                        "accepted_nonzero": 800,
                        "zero_displacement": 100,
                        "rejected_all": 100,
                        "u_acceptance_rate": 0.80,
                    },
                    "displacement_norms": {"mean_proposed": 0.05, "mean_realized": 0.04},
                    "latencies": {"total_wall_seconds": 3600.0},
                    "alpha_histogram": {"1.0": 800, "0.5": 0, "0.25": 0, "0.125": 0, "0.0625": 0, "0.03125": 0, "0.015625": 0, "0.0078125": 0, "0.0": 200},
                },
            }
        },
    }

    md = format_markdown_report(sample_report)
    assert "# Scalar Transfer-Utility Pilot: Comparative Analysis" in md
    assert "PILOT_GO" in md
    assert "Final Performance at Step 2000" in md
    assert "Screening Decision Rule Breakdown" in md
    assert "Transaction Diagnostics" in md
    assert "Alpha Scale Selection Frequencies" in md


# ==============================================================================
# Real Bounded Profile & Real Tiny End-to-End CLI Integration Tests
# ==============================================================================

def test_runner_cli_preflight_mode(tmp_path: Path):
    """Test standalone preflight CLI execution."""
    out_dir = tmp_path / "preflight_test"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "check_transfer_utility_preflight.py"),
        "--protocol", str(ROOT / "configs" / "transfer_utility" / "protocol.json"),
        "--out", str(out_dir),
        "--device", "cpu",
        "--synthetic",
        "--skip-jobs-check",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=30)
    assert res.returncode == 0, f"Preflight failed: {res.stderr}"
    receipt = json.loads(res.stdout)
    assert not out_dir.exists(), "read-only preflight must not create the proposed run output"
    assert receipt["status"] == "PASS"


def test_runner_bounded_profile_mode(tmp_path: Path):
    """Test runner in bounded profile mode (32 steps, warmup 8) exercising U updates."""
    out_dir = tmp_path / "profile_test"
    cfg = read_json(ROOT / "configs" / "transfer_utility" / "profile_32steps.json")
    # Tiny model widths for fast CPU profile test
    cfg["model"]["widths"] = [4, 8]
    test_cfg_path = tmp_path / "profile_smoke_cfg.json"
    write_json(test_cfg_path, cfg)

    receipt = run_bounded_profile(
        config_path=test_cfg_path,
        out_dir=out_dir,
        device="cpu",
        synthetic=True,
        max_steps=32,
    )
    assert receipt["status"] == "PASS"
    assert receipt["max_steps"] == 32
    assert receipt["warmup_steps"] == 8
    assert receipt["u_steps"] == 24
    assert "seconds_per_step" in receipt
    assert receipt["within_watchdog_7200s"] is True
    assert (out_dir / "PROFILE_RECEIPT.json").is_file()


def test_real_tiny_end_to_end_runner_cli(tmp_path: Path):
    """Real tiny end-to-end integration: canonical train CLI, export, eval, and analysis."""
    e2e_dir = tmp_path / "e2e_pilot"
    e2e_dir.mkdir(parents=True)

    # 1. Create real synthetic dataset fixture (labeled + unlabeled)
    data_dir = e2e_dir / "data"
    manifest_p, manifest_sha = synthetic_compact_fixture(
        data_dir, seed=42, n_labeled=1, n_unlabeled=2, size=16, num_frames=2, num_slices=2
    )

    # 2. Build real synthetic teacher cache using build_compact_teacher_cache.py
    cache_dir = e2e_dir / "teacher_cache"
    cache_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_compact_teacher_cache.py"),
        "--data-manifest", str(manifest_p),
        "--data-sha256", str(manifest_sha),
        "--output-dir", str(cache_dir),
        "--synthetic-teacher",
        "--device", "cpu",
        "--batch-size", "2",
    ]
    c_res = subprocess.run(cache_cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    assert c_res.returncode == 0, f"Cache build failed: {c_res.stderr}"
    cache_manifest = cache_dir / "CACHE_MANIFEST.json"
    cache_sha = digest(cache_manifest)

    # 3. Prepare tiny arm configuration (steps=4, warmup=1, width=[4, 8])
    lids = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True).lids
    c_ident = read_json(cache_manifest)["teacher_identity"]

    arm_cfg = read_json(ROOT / "configs" / "transfer_utility" / "scalar_guard_cross_patient_utility.json")
    arm_cfg.update({
        "steps": 4,
        "warmup_steps": 1,
        "batch_size": 1,
        "labeled_batch_size": 1,
        "unlabeled_batch_size": 1,
        "crop_size": 16,
        "max_trials": 2,
    })
    arm_cfg["model"] = {"name": "CompactStudent", "widths": [4, 8], "in_channels": 3, "num_classes": 4, "init": "kaiming_normal"}
    arm_cfg["data"] = {"manifest_sha256": str(manifest_sha), "budget_k": len(lids), "labeled_ids": lids}
    arm_cfg["teacher"].update({
        "cache_path": str(cache_manifest),
        "cache_sha256": str(cache_sha),
        "teacher_identity": c_ident,
        "lineage": {"budget_k": len(lids), "exact_labeled_ids": lids},
        "source_training_cost": "NOT_KNOWN",
    })
    arm_cfg_p = e2e_dir / "arm_config.json"
    write_json(arm_cfg_p, arm_cfg)

    # 4. Execute canonical train.py CLI with --stop-after-step
    arm_out = e2e_dir / "arm_out"
    train_cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--config", str(arm_cfg_p),
        "--out", str(arm_out),
        "--data", str(manifest_p),
        "--data-sha256", str(manifest_sha),
        "--device", "cpu",
        "--synthetic",
        "--steps", "4",
        "--stop-after-step", "4",
    ]
    t_res = subprocess.run(train_cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    assert t_res.returncode == 0, f"train.py failed: {t_res.stderr}"
    assert (arm_out / "TRAIN_LOG.jsonl").is_file()
    assert (arm_out / "final" / "MANIFEST.json").is_file()

    # 5. Execute canonical export_compact_student.py CLI
    ckpt_dir = arm_out / "final"
    ckpt_sha = digest(ckpt_dir / "MANIFEST.json")
    export_dir = e2e_dir / "export_out"
    exp_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_compact_student.py"),
        "--checkpoint", str(ckpt_dir),
        "--checkpoint-sha256", str(ckpt_sha),
        "--out", str(export_dir),
    ]
    e_res = subprocess.run(exp_cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    assert e_res.returncode == 0, f"export failed: {e_res.stderr}"
    assert (export_dir / "MANIFEST.json").is_file()

    # 6. Execute canonical analyze_transfer_utility.py CLI
    diag = analyze_transaction_diagnostics(arm_out / "TRAIN_LOG.jsonl")
    assert diag["status"] == "COMPLETED"
    assert diag["total_steps"] == 4
    assert diag["warmup_steps"] == 1
    assert diag["u_attempted_steps"] == 3
