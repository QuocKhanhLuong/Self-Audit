"""Tests for independent transfer-utility evidence and audit oracle.

Contract: docs/CPTU_IMPLEMENTATION_CONTRACT.md and AGENTS.md.
Must run cleanly on Linux / CI without requiring external user directories.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.audit_transfer_utility_evidence import (
    STUDY_MATERIAL_DIR,
    PREEXISTING_STUDY_HASHES,
    EXPECTED_B8_TEACHER_PAYLOAD_SHA256,
    HISTORICAL_B36_TEACHER_PAYLOAD_SHA256,
    EXPECTED_L8_IDS,
    EXPECTED_VAL18_MANIFEST_SHA256,
    EXPECTED_TRAIN_MANIFEST_SHA256,
    ARCHIVE_COMMIT,
    audit_study_inventory,
    audit_source_hashes,
    independent_selection_oracle,
    audit_teacher_identity_and_lineage,
    extract_pilot_budget_recommendation,
    run_full_evidence_audit,
    compute_narrow_student_parameters,
)


RESULTS_DIR = Path("results/compact_acdc_l8_20261029")


def test_study_material_hashes_with_fixture(tmp_path: Path):
    """Test inventory audit using a temporary fixture; runs on Linux/CI without external study folder."""
    # Create fixture files matching a subset of expected hashes
    test_files = {
        "file_a.txt": b"content_a",
        "file_b.txt": b"content_b",
    }
    expected = {}
    for name, content in test_files.items():
        fp = tmp_path / name
        fp.write_bytes(content)
        import hashlib
        expected[name] = hashlib.sha256(content).hexdigest()

    # Add an expected file that is missing from disk
    expected["file_c.txt"] = "deadbeef" * 8

    inv = audit_study_inventory(study_dir=tmp_path, expected_hashes=expected)
    assert inv["directory_exists"] is True
    assert inv["files"]["file_a.txt"]["comparison_status"] == "MATCHED"
    assert inv["files"]["file_b.txt"]["comparison_status"] == "MATCHED"
    assert inv["files"]["file_c.txt"]["comparison_status"] == "MISSING_FROM_DISK"
    assert inv["all_verified"] is False
    assert "file_c.txt" in inv["missing_files"]

    # Test when all match
    valid_expected = {k: v for k, v in expected.items() if k != "file_c.txt"}
    inv_valid = audit_study_inventory(study_dir=tmp_path, expected_hashes=valid_expected)
    assert inv_valid["all_verified"] is True
    assert len(inv_valid["missing_files"]) == 0

    # Also, if local external study directory exists on this specific host, verify it as well
    if STUDY_MATERIAL_DIR.exists():
        real_inv = audit_study_inventory(STUDY_MATERIAL_DIR)
        assert real_inv["all_verified"] is True
        for name, data in real_inv["files"].items():
            if name in PREEXISTING_STUDY_HASHES:
                assert data["comparison_status"] == "MATCHED"


def test_study_material_nonexistent_directory(tmp_path: Path):
    """Test graceful handling when study directory does not exist."""
    nonexistent = tmp_path / "does_not_exist"
    inv = audit_study_inventory(study_dir=nonexistent)
    assert inv["directory_exists"] is False
    assert inv["all_verified"] is False
    assert len(inv["missing_files"]) == len(PREEXISTING_STUDY_HASHES)


def test_scalar_larger_feasible_equal_dice_finding():
    """Independently recompute 761/7028 finding from scalar_guard_full TRAIN_LOG in repo archive."""
    log_path = RESULTS_DIR / "experiment_20261028" / "scalar_guard_full" / "TRAIN_LOG.jsonl.gz"
    res = independent_selection_oracle(log_path, warmup_steps=1000, ridge=1e-6, method="scalar_guard")
    stats = res["statistics"]

    assert stats["steps"] == 9000
    assert stats["accepted"] == 7028
    assert stats["no_feasible"] == 1972
    assert stats["oracle_selection_matches"] == 7028
    assert stats["selected_below_largest_feasible_rows"] == 1971
    assert stats["larger_feasible_pairs"] == 3458
    assert stats["equal_dice_larger_pairs"] == 1276
    assert stats["equal_dice_larger_rows"] == 761
    assert stats["larger_better_dice_pairs"] == 12


def test_ridge_gap_exact_decomposition():
    """Verify that objective gaps on equal-Dice scalar ties are explained entirely by ridge penalty."""
    log_path = RESULTS_DIR / "experiment_20261028" / "scalar_guard_full" / "TRAIN_LOG.jsonl.gz"
    res = independent_selection_oracle(log_path, warmup_steps=1000, ridge=1e-6, method="scalar_guard")
    stats = res["statistics"]
    assert stats["max_ridge_decomposition_error"] < 1e-15


def test_zero_competitor_behavior():
    """Verify historical behavior where zero is not a competitor."""
    # Scalar guard
    scalar_log = RESULTS_DIR / "experiment_20261028" / "scalar_guard_full" / "TRAIN_LOG.jsonl.gz"
    scalar_res = independent_selection_oracle(scalar_log, warmup_steps=1000, ridge=1e-6, method="scalar_guard")
    assert scalar_res["statistics"]["worse_than_zero_objective_rows"] == 713
    assert scalar_res["statistics"]["worse_than_zero_equal_base_dice_rows"] == 693

    # Joint guard
    joint_log = RESULTS_DIR / "experiment_20261028" / "joint_guard_full" / "TRAIN_LOG.jsonl.gz"
    joint_res = independent_selection_oracle(joint_log, warmup_steps=1000, ridge=1e-6, method="joint_guard")
    assert joint_res["statistics"]["accepted"] == 4506
    assert joint_res["statistics"]["no_feasible"] == 4494
    assert joint_res["statistics"]["worse_than_zero_objective_rows"] == 145
    assert joint_res["statistics"]["worse_than_zero_equal_base_dice_rows"] == 144


def test_teacher_identity_and_b36_mismatch():
    """Verify B8 teacher payload sha, L8 patient IDs, and b36 mismatch."""
    lineage = audit_teacher_identity_and_lineage(RESULTS_DIR)
    b8 = lineage["recovered_b8_teacher"]
    assert b8["payload_sha256"] == EXPECTED_B8_TEACHER_PAYLOAD_SHA256
    assert b8["matches_expected"] is True
    assert b8["gate"] == "PASS"

    # Mismatch with historical b36 teacher
    b36 = lineage["historical_b36_teacher"]
    assert b36["required_payload_sha256"] == HISTORICAL_B36_TEACHER_PAYLOAD_SHA256
    assert b36["required_payload_sha256"] != b8["payload_sha256"]

    # L8 IDs
    lids = lineage["labeled_train_ids"]
    assert lids["matches_expected"] is True
    assert lids["count"] == 8
    assert sorted(lids["ids"]) == sorted(EXPECTED_L8_IDS)


def test_narrow_student_parameters_and_pilot_specs():
    """Verify exact narrow CompactStudent parameter counts, spatial Z context, and pilot budget."""
    params, raw_bytes = compute_narrow_student_parameters()
    assert params == 603956, f"Expected 603956 narrow parameters, got {params}"
    assert raw_bytes == 2415824, f"Expected 2415824 raw bytes, got {raw_bytes}"

    specs = extract_pilot_budget_recommendation(RESULTS_DIR)
    arch = specs["architecture"]
    assert arch["parameters"] == 603956
    assert arch["raw_tensor_bytes"] == 2415824
    assert arch["widths"] == [16, 32, 64, 128]
    assert arch["wide_contrast"]["wide_parameters"] == 2410596

    # Context definition: spatial Z, not temporal
    aug = specs["augmentation_and_sampling"]
    assert "Spatial Z" in aug["context_definition"]
    assert "NOT temporal" in aug["context_definition"]

    # Normalization provenance: CANONICAL_PREPROCESSING percentile normalization
    norm = specs["normalization"]
    assert "percentiles" in norm["method"]
    assert norm["provenance"] == "CANONICAL_PREPROCESSING from self_audit/shared_data.py"

    # Locked pilot budget: 2000 total steps (1000 warmup + 1000 KD)
    pilot = specs["pilot_budget_recommendation"]
    assert pilot["locked_total_steps"] == 2000
    assert pilot["warmup_steps"] == 1000
    assert pilot["active_kd_steps"] == 1000
    assert pilot["eval_steps"] == [1000, 1500, 2000]
    assert len(pilot["arms"]) == 4


def test_source_hashes_derivation_from_git():
    """Verify original source hashes derived from git show d93c4de:path or graceful reporting."""
    sources = audit_source_hashes(archive_commit=ARCHIVE_COMMIT)
    src_map = sources["sources"]
    assert "ACTIVE_PIPELINE.json" in src_map
    assert "docs/METHOD.md" in src_map
    assert "self_audit/compact_control.py" in src_map
    assert "self_audit/compact_training.py" in src_map

    # If git commit is reachable, verify exact pinned archive hashes
    if sources["git_commit_reachable"]:
        assert (
            src_map["ACTIVE_PIPELINE.json"]["archived_d93c4de_sha256"]
            == "618de5c3ab6bbb4b2e88a9ff4bc7ca0eb90594e281c7728e041adf3e675abccf"
        )
        assert (
            src_map["docs/METHOD.md"]["archived_d93c4de_sha256"]
            == "d7a62f699eae6035496333178406c3d2c243f4364aed9d84ba1923e9ededb7b6"
        )
        assert (
            src_map["self_audit/compact_control.py"]["archived_d93c4de_sha256"]
            == "17878be6becada6f6fefa2ed7d56f52357c7d049f6ff07f853ea319e91d082b8"
        )
        assert (
            src_map["self_audit/compact_training.py"]["archived_d93c4de_sha256"]
            == "b7a5123ebc93e9271e111430f4818b8654d0c63d1b31e7a5b34dc05749c06a70"
        )


def test_full_evidence_audit_entrypoint():
    """Verify complete audit pipeline executes cleanly."""
    res = run_full_evidence_audit(results_root=RESULTS_DIR)
    assert res["status"] == "PASS"
    assert res["budget_specs"]["architecture"]["parameters"] == 603956
    assert res["budget_specs"]["pilot_budget_recommendation"]["locked_total_steps"] == 2000
    assert res["not_run_statuses"]["full_patient_training"].startswith("NOT_RUN")
