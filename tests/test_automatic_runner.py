"""Unit and integration tests for scripts/run_automatic_pilot.py and configs/automatic_pilot.

Verifies:
  1. CLI entrypoint: --help, --validate-configs, argument parsing.
  2. Matrix configuration invariants: all 10 arms strictly match DEFAULT_AUTOMATIC_CONFIG schema.
  3. Shared support-only warm-start verification: MANIFEST hash pinning, lineage accountability, fail-closed on corrupt.
  4. Resource accounting: disjoint components, no fabricated metrics (missing stays NOT_MEASURED),
     incomplete mask export leaves cold cost INCOMPLETE (never 0).
  5. Go / No-Go decision rules: verified conditions required (coverage, matched compute/lineage/init),
     joint>scalar, dice>smaller_lr, u>extra_supervised; no software-only GO; NaN/inf rejected.
  6. Synthetic smoke run through train.py router without duplicate trainer.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_automatic_pilot import (
    FIXED_ARMS,
    compute_file_sha256,
    compute_resource_accounting,
    evaluate_decision_gate,
    load_arm_config,
    load_registry,
    run_experiment_matrix,
    validate_matrix_configs,
    verify_warm_start_checkpoint,
)
from self_audit.automatic import DEFAULT_AUTOMATIC_CONFIG, SCHEMA, automatic_source_digest
from self_audit.data import synthetic_fixture


# ---------------------------------------------------------------------------
# 1. CLI and configuration validation tests
# ---------------------------------------------------------------------------

def test_runner_cli_help():
    """Verify scripts/run_automatic_pilot.py --help succeeds and documents contract flags."""
    cmd = [sys.executable, str(ROOT / "scripts/run_automatic_pilot.py"), "--help"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0
    assert "--mode" in proc.stdout
    assert "--arms" in proc.stdout
    assert "--comparison" in proc.stdout
    assert "--synthetic" in proc.stdout
    assert "--warm-start-checkpoint" in proc.stdout
    assert "--validate-configs" in proc.stdout


def test_registry_metadata():
    """Verify configs/automatic_pilot/REGISTRY.json invariants."""
    reg = load_registry()
    assert reg["schema"] == "self_audit.automatic_registry.v1"
    assert reg["total_arms"] == 10
    assert tuple(reg["fixed_arms"]) == FIXED_ARMS
    assert "PRO 4000" in reg["hardware_target"]
    assert "no_software_only_go" in reg["decision_gates"]
    for arm in FIXED_ARMS:
        assert arm in reg["arms"]
        assert Path(ROOT / reg["arms"][arm]["config_file"]).is_file()


def test_matrix_configs_validation():
    """Verify all 10 arm configurations adhere to schema self_audit.automatic.v1 and contract factors."""
    report = validate_matrix_configs()
    assert report["status"] == "PASS"
    assert report["total_arms"] == 10
    assert report["shared_invariants"]["seed"] == 42
    assert report["shared_invariants"]["window_mode"] == "fixed"
    assert report["shared_invariants"]["support_warm_start_shared"] is True

    # Validate individual arm contracts
    arms = report["arms"]
    assert arms["scalar_brier"]["allocation"] == "scalar" and not arms["scalar_brier"]["dice_guard"]
    assert arms["joint_brier"]["allocation"] == "joint" and not arms["joint_brier"]["dice_guard"]
    assert arms["scalar_dice"]["allocation"] == "scalar" and arms["scalar_dice"]["dice_guard"]
    assert arms["joint_dice"]["allocation"] == "joint" and arms["joint_dice"]["dice_guard"]
    assert arms["frozen"]["pseudo_lr"] == 0.0
    assert arms["support_only"]["pseudo_lr"] == 0.0
    assert arms["plain_head_u"]["allocation"] == "plain" and not arms["plain_head_u"]["dice_guard"]
    assert arms["extra_supervised"]["pseudo_lr"] == 0.0
    assert arms["smaller_lr"]["pseudo_lr"] < arms["plain_head_u"]["pseudo_lr"]
    assert arms["full_ema"]["body_lr"] > 0.0


def test_arm_config_rejects_unknown():
    """Verify load_arm_config rejects unregistered arm names."""
    with pytest.raises(ValueError, match="Unknown arm"):
        load_arm_config("unregistered_speculative_arm")


def test_arm_config_rejects_unexpected_fields(tmp_path):
    """Verify load_arm_config strictly rejects unknown configuration fields."""
    bad_cfg = copy.deepcopy(DEFAULT_AUTOMATIC_CONFIG)
    bad_cfg["arm"] = "joint_dice"
    bad_cfg["invented_speculative_field"] = "illegal"

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "joint_dice.json").write_text(json.dumps(bad_cfg))
    with pytest.raises(ValueError, match="unexpected fields"):
        load_arm_config("joint_dice", config_dir=cfg_dir)


# ---------------------------------------------------------------------------
# 2. Shared warm-start verification tests
# ---------------------------------------------------------------------------

def test_verify_warm_start_checkpoint_success(tmp_path):
    """Verify valid warm-start checkpoint manifest and payload hashes."""
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    weights_file = ckpt_dir / "weights.pt"
    weights_file.write_bytes(b"TEST_WEIGHTS_BINARY_PAYLOAD_12345")
    weights_sha = compute_file_sha256(weights_file)

    manifest_file = ckpt_dir / "MANIFEST.json"
    manifest_data = {
        "schema": "self_audit.automatic_checkpoint.v1",
        "phase": "warm_start",
        "arm": "support_only",
        "prediction_model": "student",
        "source_sha256": automatic_source_digest(),
        "config": {**copy.deepcopy(DEFAULT_AUTOMATIC_CONFIG), "phase": "warm_start", "arm": "support_only", "initialization": {"warm_start_checkpoint": None, "warm_start_sha256": None, "declared_target_budget": 2}},
        "labeled_ids": ["patient001", "patient002"],
        "unlabeled_ids": ["patient003"],
        "training_manifest_sha256": "a" * 64,
        "payload_sha256": weights_sha,
        "declared_target_budget": 2,
        "target_label_lineage": {
            "patients": ["patient001", "patient002"],
            "count": 2,
        },
    }
    manifest_file.write_text(json.dumps(manifest_data))
    manifest_sha = compute_file_sha256(manifest_file)

    info = verify_warm_start_checkpoint(ckpt_dir, expected_sha256=manifest_sha)
    assert info["status"] == "VERIFIED"
    assert info["manifest_sha256"] == manifest_sha
    assert info["weights_sha256"] == weights_sha
    assert info["declared_target_budget"] == 2
    assert info["phase"] == "warm_start"
    assert info["arm"] == "support_only"
    assert info["prediction_model"] == "student"


def test_verify_warm_start_fails_closed_on_corrupt(tmp_path):
    """Verify warm-start checkpoint fails closed on hash mismatch, corrupt metadata, or missing files."""
    ckpt_dir = tmp_path / "corrupt_ckpt"
    ckpt_dir.mkdir()

    # Missing manifest
    with pytest.raises(FileNotFoundError):
        verify_warm_start_checkpoint(ckpt_dir, expected_sha256=compute_file_sha256(ckpt_dir / "MANIFEST.json") if (ckpt_dir / "MANIFEST.json").exists() else "0" * 64)

    # Missing weights
    manifest_file = ckpt_dir / "MANIFEST.json"
    manifest_file.write_text(json.dumps({
        "schema": "self_audit.automatic_checkpoint.v1",
        "phase": "warm_start",
        "arm": "support_only",
        "prediction_model": "student",
        "source_sha256": "0" * 64,
        "payload_sha256": "fake_sha",
        "target_label_lineage": {"patients": ["p1"]},
    }))
    with pytest.raises(FileNotFoundError):
        verify_warm_start_checkpoint(ckpt_dir, expected_sha256=compute_file_sha256(ckpt_dir / "MANIFEST.json") if (ckpt_dir / "MANIFEST.json").exists() else "0" * 64)

    # Corrupt payload hash
    weights_file = ckpt_dir / "weights.pt"
    weights_file.write_bytes(b"dummy")
    manifest_file.write_text(json.dumps({
        "schema": "self_audit.automatic_checkpoint.v1",
        "phase": "warm_start",
        "arm": "support_only",
        "prediction_model": "student",
        "source_sha256": "0" * 64,
        "payload_sha256": "wrong_payload_hash_does_not_match",
        "target_label_lineage": {"patients": ["p1"]},
    }))
    with pytest.raises(ValueError, match="Payload weights SHA-256 mismatch"):
        verify_warm_start_checkpoint(ckpt_dir, expected_sha256=compute_file_sha256(ckpt_dir / "MANIFEST.json") if (ckpt_dir / "MANIFEST.json").exists() else "0" * 64)

    # Missing lineage patients
    manifest_file.write_text(json.dumps({
        "schema": "self_audit.automatic_checkpoint.v1",
        "phase": "warm_start",
        "arm": "support_only",
        "prediction_model": "student",
        "source_sha256": "0" * 64,
        "payload_sha256": compute_file_sha256(weights_file),
        "target_label_lineage": {"patients": []},
    }))
    with pytest.raises(ValueError, match="target patients"):
        verify_warm_start_checkpoint(ckpt_dir, expected_sha256=compute_file_sha256(ckpt_dir / "MANIFEST.json") if (ckpt_dir / "MANIFEST.json").exists() else "0" * 64)

    # Invalid phase
    manifest_file.write_text(json.dumps({
        "schema": "self_audit.automatic_checkpoint.v1",
        "phase": "adaptation",
        "arm": "support_only",
        "prediction_model": "student",
        "source_sha256": "0" * 64,
        "payload_sha256": compute_file_sha256(weights_file),
        "target_label_lineage": {"patients": ["p1"]},
    }))
    with pytest.raises(ValueError, match="phase must be 'warm_start'"):
        verify_warm_start_checkpoint(ckpt_dir, expected_sha256=compute_file_sha256(ckpt_dir / "MANIFEST.json") if (ckpt_dir / "MANIFEST.json").exists() else "0" * 64)

    # Invalid prediction_model
    manifest_file.write_text(json.dumps({
        "schema": "self_audit.automatic_checkpoint.v1",
        "phase": "warm_start",
        "arm": "support_only",
        "prediction_model": "teacher",
        "source_sha256": "0" * 64,
        "payload_sha256": compute_file_sha256(weights_file),
        "target_label_lineage": {"patients": ["p1"]},
    }))
    with pytest.raises(ValueError, match="prediction_model must be 'student'"):
        verify_warm_start_checkpoint(ckpt_dir, expected_sha256=compute_file_sha256(ckpt_dir / "MANIFEST.json") if (ckpt_dir / "MANIFEST.json").exists() else "0" * 64)

    # Missing source_sha256
    manifest_file.write_text(json.dumps({
        "schema": "self_audit.automatic_checkpoint.v1",
        "phase": "warm_start",
        "arm": "support_only",
        "prediction_model": "student",
        "source_sha256": "",
        "payload_sha256": compute_file_sha256(weights_file),
        "target_label_lineage": {"patients": ["p1"]},
    }))
    with pytest.raises(ValueError, match="non-empty source_sha256"):
        verify_warm_start_checkpoint(ckpt_dir, expected_sha256=compute_file_sha256(ckpt_dir / "MANIFEST.json") if (ckpt_dir / "MANIFEST.json").exists() else "0" * 64)


# ---------------------------------------------------------------------------
# 3. Resource accounting tests (cold total cost vs marginal cost)
# ---------------------------------------------------------------------------

def test_resource_accounting_no_fabricated_defaults():
    """Verify missing metrics remain NOT_MEASURED / None without invented defaults."""
    minimal_receipt = {
        "steps": 32,
        "elapsed_seconds": 12.0,
    }
    costs = compute_resource_accounting(minimal_receipt, warm_start_receipt=None, evaluation_receipt=None)
    marginal = costs["marginal_adaptation_cost"]
    cold = costs["cold_total_cost"]

    # Invariants: no fabricated p95=p50*1.25, no default acceptance 1.0, no default nonzero updates
    assert marginal["latency_p50_ms"] == "NOT_MEASURED"
    assert marginal["latency_p95_ms"] == "NOT_MEASURED"
    assert marginal["acceptance_rate"] == "NOT_MEASURED"
    assert marginal["nonzero_displacement_updates"] == "NOT_MEASURED"
    assert marginal["peak_vram_allocated_mb"] == "NOT_MEASURED"
    assert marginal["peak_vram_reserved_mb"] == "NOT_MEASURED"
    assert marginal["extraction_seconds"] == "NOT_MEASURED"

    # Cold cost status must remain INCOMPLETE and output export NOT_MEASURED
    assert cold["status"] == "INCOMPLETE"
    assert cold["output_export_seconds"] == "NOT_MEASURED"
    assert cold["cold_total_seconds"] == "INCOMPLETE"


def test_resource_accounting_disjoint_components_complete():
    """Verify disjoint components when all phases are measured (no double-counting)."""
    arm_receipt = {
        "steps": 32,
        "elapsed_seconds": 10.0,
        "extraction_seconds": 2.0,
        "pseudo_proposals": 20,
        "pseudo_accepted": 15,
        "nonzero_displacement_updates": 12,
        "peak_allocated_bytes": 104857600,
        "peak_reserved_bytes": 209715200,
        "step_latencies_ms": [100.0] * 10 + [120.0] * 12 + [150.0] * 10,
    }
    ws_receipt = {"elapsed_seconds": 4.0}
    eval_receipt = {"total_seconds": 6.0}

    costs = compute_resource_accounting(arm_receipt, warm_start_receipt=ws_receipt, evaluation_receipt=eval_receipt)
    marginal = costs["marginal_adaptation_cost"]
    cold = costs["cold_total_cost"]

    assert marginal["marginal_adaptation_seconds"] == 10.0
    assert marginal["extraction_seconds"] == 2.0
    assert marginal["acceptance_rate"] == 15 / 20
    # Verified: nonzero_displacement_updates is 12 (from engine receipt), not equated to pseudo_accepted (15)
    assert marginal["nonzero_displacement_updates"] == 12
    assert marginal["peak_vram_allocated_mb"] == 100.0
    assert marginal["peak_vram_reserved_mb"] == 200.0
    assert marginal["latency_p50_ms"] == 120.0
    assert marginal["latency_p95_ms"] == 150.0

    # Cold total cost: warm_start (4.0) + adaptation (10.0) + output_export (6.0) = 20.0
    assert cold["status"] == "COMPLETE"
    assert cold["warm_start_seconds"] == 4.0
    assert cold["marginal_adaptation_seconds"] == 10.0
    assert cold["output_export_seconds"] == 6.0
    assert cold["cold_total_seconds"] == 20.0


def test_resource_accounting_engine_keys_direct_extraction():
    """Verify direct extraction of engine report keys without fallback formulas."""
    arm_receipt = {
        "steps": 10,
        "elapsed_seconds": 5.0,
        "throughput_updates_per_sec": 2.5,
        "latency_p50_ms": 42.0,
        "latency_p95_ms": 68.0,
        "peak_vram_allocated_mb": 512.0,
        "peak_vram_reserved_mb": 1024.0,
        "acceptance_rate": 0.85,
        "nonzero_displacement_updates": 7,
    }
    costs = compute_resource_accounting(arm_receipt)
    marginal = costs["marginal_adaptation_cost"]
    assert marginal["throughput_updates_per_sec"] == 2.5
    assert marginal["latency_p50_ms"] == 42.0
    assert marginal["latency_p95_ms"] == 68.0
    assert marginal["peak_vram_allocated_mb"] == 512.0
    assert marginal["peak_vram_reserved_mb"] == 1024.0
    assert marginal["acceptance_rate"] == 0.85
    assert marginal["nonzero_displacement_updates"] == 7


# ---------------------------------------------------------------------------
# 4. Go / No-Go Decision Gate Protocol tests
# ---------------------------------------------------------------------------

def test_decision_gate_synthetic_fails_closed_no_software_go():
    """Verify synthetic / software-only runs strictly cannot yield research GO."""
    manufactured_metrics = {
        "joint_dice": {"query_patient_macro_fg_dice": 0.85},
        "scalar_dice": {"query_patient_macro_fg_dice": 0.80},
        "smaller_lr": {"query_patient_macro_fg_dice": 0.78},
        "extra_supervised": {"query_patient_macro_fg_dice": 0.76},
    }
    result = evaluate_decision_gate(manufactured_metrics, synthetic=True)
    assert result["decision"] == "NO_GO_SOFTWARE_ONLY"
    assert result["quality_status"] == "NOT_RUN"
    assert "software-only" in result["reason"].lower()


def test_decision_gate_incomplete_coverage_inconclusive():
    """Verify incomplete output coverage (< 1.0) yields INCONCLUSIVE_NOT_RUN."""
    metrics = {
        "joint_dice": {"query_patient_macro_fg_dice": 0.85},
        "scalar_dice": {"query_patient_macro_fg_dice": 0.80},
        "smaller_lr": {"query_patient_macro_fg_dice": 0.78},
        "extra_supervised": {"query_patient_macro_fg_dice": 0.76},
    }
    ctx = {
        "query_provenance": "eval_manifest_sha",
        "prediction_model": "student",
        "output_coverage": 0.95,
        "matched_initialization": True,
        "matched_lineage": True,
        "matched_compute": True,
    }
    result = evaluate_decision_gate(metrics, synthetic=False, verified_context=ctx)
    assert result["decision"] == "INCONCLUSIVE_NOT_RUN"
    assert result["quality_status"] == "NOT_RUN"
    assert "coverage" in result["reason"].lower()


def test_decision_gate_rejects_nan_inf_metrics():
    """Verify NaN or inf metric values yield INCONCLUSIVE_NOT_RUN."""
    metrics_with_nan = {
        "joint_dice": {"query_patient_macro_fg_dice": float("nan")},
        "scalar_dice": {"query_patient_macro_fg_dice": 0.80},
        "smaller_lr": {"query_patient_macro_fg_dice": 0.78},
        "extra_supervised": {"query_patient_macro_fg_dice": 0.76},
    }
    ctx = {
        "query_provenance": "eval_manifest_sha",
        "prediction_model": "student",
        "output_coverage": 1.0,
        "matched_initialization": True,
        "matched_lineage": True,
        "matched_compute": True,
    }
    result = evaluate_decision_gate(metrics_with_nan, synthetic=False, verified_context=ctx)
    assert result["decision"] == "INCONCLUSIVE_NOT_RUN"
    assert result["quality_status"] == "NOT_RUN"


def test_decision_gate_real_evaluation_go_and_no_go():
    """Verify real independent query metrics evaluate the 3 contracted gates."""
    verified_ctx = {
        "query_provenance": "sha256:eval_query_manifest_12345",
        "prediction_model": "student",
        "output_coverage": 1.0,
        "matched_initialization": True,
        "matched_lineage": True,
        "matched_compute": True,
        "steps": 32,
        "comparison_mode": "equal_update",
    }

    # Scenario A: All 3 pass
    passing_metrics = {
        "joint_dice": {
            "query_patient_macro_fg_dice": 0.85,
            "macro_patient": {
                "RV": {"dice": 0.83},
                "MYO": {"dice": 0.84},
                "LV": {"dice": 0.88},
            },
        },
        "scalar_dice": {
            "query_patient_macro_fg_dice": 0.81,      # joint > scalar (0.85 > 0.81)
            "macro_patient": {
                "RV": {"dice": 0.79},
                "MYO": {"dice": 0.80},
                "LV": {"dice": 0.84},
            },
        },
        "smaller_lr": {
            "query_patient_macro_fg_dice": 0.82,       # dice_guard > smaller_lr (0.85 > 0.82)
            "macro_patient": {
                "RV": {"dice": 0.80},
                "MYO": {"dice": 0.81},
                "LV": {"dice": 0.85},
            },
        },
        "extra_supervised": {
            "query_patient_macro_fg_dice": 0.80, # u > extra_supervised (0.85 > 0.80)
            "macro_patient": {
                "RV": {"dice": 0.78},
                "MYO": {"dice": 0.79},
                "LV": {"dice": 0.83},
            },
        },
    }
    res_pass = evaluate_decision_gate(passing_metrics, synthetic=False, verified_context=verified_ctx)
    assert res_pass["decision"] == "GO"
    assert res_pass["quality_status"] == "MEASURED"
    assert res_pass["gates"]["joint_beats_scalar"] is True
    assert res_pass["gates"]["dice_guard_beats_smaller_lr"] is True
    assert res_pass["gates"]["u_beats_extra_supervised"] is True
    assert res_pass["scores"]["joint_dice"] == 0.85
    assert res_pass["per_class_primary_results"]["joint_dice"]["RV"] == 0.83
    assert res_pass["per_class_primary_results"]["joint_dice"]["LV"] == 0.88
    assert res_pass["matched_compute"]["status"] == "VERIFIED_MATCHED"
    assert res_pass["matched_compute"]["steps"] == 32

    # Scenario B: joint does not beat scalar
    failing_joint = copy.deepcopy(passing_metrics)
    failing_joint["scalar_dice"]["query_patient_macro_fg_dice"] = 0.86
    res_fail = evaluate_decision_gate(failing_joint, synthetic=False, verified_context=verified_ctx)
    assert res_fail["decision"] == "NO_GO"
    assert res_fail["gates"]["joint_beats_scalar"] is False

    # Scenario C: dice_guard does not beat smaller_lr
    failing_lr = copy.deepcopy(passing_metrics)
    failing_lr["smaller_lr"]["query_patient_macro_fg_dice"] = 0.86
    res_fail_lr = evaluate_decision_gate(failing_lr, synthetic=False, verified_context=verified_ctx)
    assert res_fail_lr["decision"] == "NO_GO"
    assert res_fail_lr["gates"]["dice_guard_beats_smaller_lr"] is False

    # Scenario D: u does not beat extra_supervised
    failing_u = copy.deepcopy(passing_metrics)
    failing_u["extra_supervised"]["query_patient_macro_fg_dice"] = 0.86
    res_fail_u = evaluate_decision_gate(failing_u, synthetic=False, verified_context=verified_ctx)
    assert res_fail_u["decision"] == "NO_GO"
    assert res_fail_u["gates"]["u_beats_extra_supervised"] is False


def test_decision_gate_missing_context_inconclusive():
    """Verify missing verified_context strictly yields INCONCLUSIVE_NOT_RUN."""
    metrics = {
        "joint_dice": {"query_patient_macro_fg_dice": 0.85},
        "scalar_dice": {"query_patient_macro_fg_dice": 0.80},
        "smaller_lr": {"query_patient_macro_fg_dice": 0.78},
        "extra_supervised": {"query_patient_macro_fg_dice": 0.76},
    }
    result = evaluate_decision_gate(metrics, synthetic=False, verified_context=None)
    assert result["decision"] == "INCONCLUSIVE_NOT_RUN"
    assert result["quality_status"] == "NOT_RUN"
    assert "positive verified_context is required" in result["reason"].lower()


def test_decision_gate_missing_individual_context_fields():
    """Verify missing any individual verified_context invariant yields INCONCLUSIVE_NOT_RUN."""
    base_ctx = {
        "query_provenance": "manifest_sha_ok",
        "prediction_model": "student",
        "output_coverage": 1.0,
        "matched_initialization": True,
        "matched_lineage": True,
        "matched_compute": True,
    }
    metrics = {
        "joint_dice": {"query_patient_macro_fg_dice": 0.85},
        "scalar_dice": {"query_patient_macro_fg_dice": 0.80},
        "smaller_lr": {"query_patient_macro_fg_dice": 0.78},
        "extra_supervised": {"query_patient_macro_fg_dice": 0.76},
    }

    # Missing query_provenance
    c = copy.deepcopy(base_ctx)
    c["query_provenance"] = ""
    res = evaluate_decision_gate(metrics, synthetic=False, verified_context=c)
    assert res["decision"] == "INCONCLUSIVE_NOT_RUN"
    assert "query_provenance" in res["reason"]

    # Wrong prediction_model
    c = copy.deepcopy(base_ctx)
    c["prediction_model"] = "teacher"
    res = evaluate_decision_gate(metrics, synthetic=False, verified_context=c)
    assert res["decision"] == "INCONCLUSIVE_NOT_RUN"
    assert "prediction model must be 'student'" in res["reason"].lower()

    # Incomplete coverage
    c = copy.deepcopy(base_ctx)
    c["output_coverage"] = 0.99
    res = evaluate_decision_gate(metrics, synthetic=False, verified_context=c)
    assert res["decision"] == "INCONCLUSIVE_NOT_RUN"
    assert "coverage" in res["reason"].lower()

    # Unmatched compute
    c = copy.deepcopy(base_ctx)
    c["matched_compute"] = False
    res = evaluate_decision_gate(metrics, synthetic=False, verified_context=c)
    assert res["decision"] == "INCONCLUSIVE_NOT_RUN"
    assert "compute" in res["reason"].lower()

    # Unmatched initialization
    c = copy.deepcopy(base_ctx)
    c["matched_initialization"] = False
    res = evaluate_decision_gate(metrics, synthetic=False, verified_context=c)
    assert res["decision"] == "INCONCLUSIVE_NOT_RUN"
    assert "initialization" in res["reason"].lower()

    # Unmatched lineage
    c = copy.deepcopy(base_ctx)
    c["matched_lineage"] = False
    res = evaluate_decision_gate(metrics, synthetic=False, verified_context=c)
    assert res["decision"] == "INCONCLUSIVE_NOT_RUN"
    assert "lineage" in res["reason"].lower()


def test_decision_gate_zero_dice_is_valid_finite():
    """Verify Dice of 0.0 is treated as a valid finite metric (0.0 <= f <= 1.0), not missing."""
    verified_ctx = {
        "query_provenance": "eval_query_sha",
        "prediction_model": "student",
        "output_coverage": 1.0,
        "matched_initialization": True,
        "matched_lineage": True,
        "matched_compute": True,
    }
    metrics = {
        "joint_dice": {"query_patient_macro_fg_dice": 0.85},
        "scalar_dice": {"query_patient_macro_fg_dice": 0.0},
        "smaller_lr": {"query_patient_macro_fg_dice": 0.0},
        "extra_supervised": {"query_patient_macro_fg_dice": 0.0},
    }
    for values in metrics.values():
        values["per_class"] = {c: values["query_patient_macro_fg_dice"] for c in ("RV", "MYO", "LV")}
    result = evaluate_decision_gate(metrics, synthetic=False, verified_context=verified_ctx)
    assert result["decision"] == "GO"
    assert result["quality_status"] == "MEASURED"
    assert result["scores"]["scalar_dice"] == 0.0
    assert result["scores"]["smaller_lr"] == 0.0
    assert result["scores"]["extra_supervised"] == 0.0


def test_output_directory_safety_prevents_protected_tree(tmp_path):
    """Verify runner refuses to write into protected repository directories."""
    data_dir = tmp_path / "safety_data"
    manifest_path, manifest_sha = synthetic_fixture(data_dir, size=32)

    with pytest.raises(ValueError, match="protected source tree"):
        run_experiment_matrix(
            ["scalar_brier"],
            manifest_path,
            manifest_sha,
            out_dir=ROOT / "self_audit",
            device="cpu",
            synthetic=True,
        )


# ---------------------------------------------------------------------------
# 5. Synthetic smoke run integration test
# ---------------------------------------------------------------------------

def test_synthetic_smoke_matrix_execution(tmp_path):
    """Execute bounded synthetic smoke through train.py router and verify report."""
    data_dir = tmp_path / "synth_data"
    manifest_path, manifest_sha = synthetic_fixture(data_dir, size=32)

    out_dir = tmp_path / "matrix_smoke"
    selected_arms = ["scalar_brier", "joint_dice"]

    report = run_experiment_matrix(
        selected_arms,
        manifest_path,
        manifest_sha,
        out_dir,
        comparison="equal_update",
        steps=1,
        device="cpu",
        synthetic=True,
        allow_full_run=True,
    )

    assert report["schema"] == "self_audit.automatic_pilot_report.v1"
    assert report["synthetic"] is True
    assert "warm_start" in report
    assert report["warm_start"]["manifest_sha256"] is not None

    for arm in selected_arms:
        assert arm in report["arms"]
        receipt = report["arms"][arm]["receipt"]
        assert receipt["status"] == "COMPLETED"
        assert receipt["steps"] == 1
        assert receipt["arm"] == arm

        accounting = report["arms"][arm]["accounting"]
        assert accounting["marginal_adaptation_cost"]["marginal_adaptation_seconds"] > 0
        assert accounting["cold_total_cost"]["status"] == "INCOMPLETE"

    # Software-only gate check
    assert report["decision_gate"]["decision"] == "NO_GO_SOFTWARE_ONLY"


def test_synthetic_smoke_equal_time_mode(tmp_path):
    """Execute bounded synthetic equal-time smoke and verify time accounting and certification flags."""
    data_dir = tmp_path / "equal_time_synth_data"
    manifest_path, manifest_sha = synthetic_fixture(data_dir, size=32)

    out_dir = tmp_path / "equal_time_smoke"
    selected_arms = ["support_only", "joint_dice"]

    report = run_experiment_matrix(
        selected_arms,
        manifest_path,
        manifest_sha,
        out_dir,
        comparison="equal_time",
        time_budget=0.5,
        steps=2,
        device="cpu",
        synthetic=True,
        allow_full_run=True,
    )

    assert report["schema"] == "self_audit.automatic_pilot_report.v1"
    assert report["comparison_mode"] == "equal_time"
    assert "equal_time_certified" in report
    assert report["effective_time_budget_seconds"] == 0.5
    for arm in selected_arms:
        assert arm in report["arms"]
        receipt = report["arms"][arm]["receipt"]
        assert receipt["status"] == "COMPLETED"
        assert receipt["time_budget_seconds"] == 0.5
