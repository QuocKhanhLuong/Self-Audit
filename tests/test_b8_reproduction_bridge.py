"""Focused tests for explicit opt-in B8 reproduction identity binding.

Tests cover:
  1. Historical baseline pins and default locked protocol remain unchanged.
  2. Reproduction source bridge strictly verifies git repo, commit, closure,
     clean tree, and exact historical model.py / window.py hashes.
  3. Reproduction receipt validation fails closed on bad schema, Macro4 < 0.87,
     failed quality gate, bad L8 IDs, unverified Drive readback, contradictory files.
  4. Explicit teacher_kind='reproduction' required (no loose inference from dict).
  5. Protocol derivation binds full checkpoint payload, cache hash, and runtime policy
     (deterministic_algorithms=True) while locking all pilot hyperparameters.
  6. Teacher cache and preflight reject unverified physical files.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch

from self_audit.compact_artifacts import digest
from self_audit.compact_data import synthetic_compact_fixture
from self_audit.compact_teacher import (
    HISTORICAL_BASELINE_PINS,
    HISTORICAL_MANIFEST_SHA256,
    HISTORICAL_MODEL_SHA256,
    HISTORICAL_PAYLOAD_SHA256,
    HISTORICAL_SOURCE_CLOSURE,
    HISTORICAL_SOURCE_COMMIT,
    HISTORICAL_WINDOW_SHA256,
    REPRODUCTION_RECEIPT_SCHEMA,
    FixedTeacherCache,
    is_historical_baseline_checkpoint,
    load_teacher_for_cache,
    validate_reproduction_receipt,
    validate_teacher_identity,
    verify_historical_source,
    verify_reproduction_source,
)
from self_audit.transfer_preflight import (
    PINNED_B8_TEACHER_MANIFEST_SHA256,
    PINNED_L8_PATIENT_IDS,
    PINNED_LABEL_BUDGET_K,
    PINNED_RAW_PAYLOAD_SHA256,
    PINNED_TEACHER_CACHE_SHA256,
    PINNED_TRAIN_MANIFEST_SHA256,
    PINNED_VAL_MANIFEST_SHA256,
    check_preflight,
    derive_reproduction_protocol,
    validate_locked_protocol,
)
from scripts.run_transfer_utility_pilot import (
    bind_teacher_config,
    validate_arm_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mutation", [None, "legacy_method", "missing_lineage", "raw_schema", "ema_only"])
def test_real_v2_manifest_layout_and_strict_student(tmp_path, valid_reproduction_receipt_dict, mutation):
    """Use the schema actually emitted by the historical trainer, not a surrogate layout."""
    receipt = copy.deepcopy(valid_reproduction_receipt_dict)
    state = {"head.weight": torch.tensor([0.1, 0.2])}
    full = tmp_path / "weights.pt"
    raw = tmp_path / "raw_student.pt"
    torch.save({"teacher" if mutation == "ema_only" else "student": state}, full)
    torch.save(state, raw)
    manifest = {
        "schema": "self_audit.checkpoint.v2", "config": {"method": "ema_full"},
        "step": 10000, "source_sha256": receipt["source_sha256"],
        "labeled_ids": receipt["labeled_ids"], "payload_sha256": digest(full),
    }
    if mutation == "legacy_method":
        manifest.pop("config")
        manifest["method"] = "ema_full"
    if mutation == "missing_lineage":
        manifest.pop("labeled_ids")
    if mutation == "raw_schema":
        manifest["schema"] = "self_audit.b8.raw_student.v1"
    mf = tmp_path / "MANIFEST.json"
    mf.write_text(json.dumps(manifest))
    for key, name, path in [
        ("checkpoint_payload_sha256", "full_checkpoint/weights.pt", full),
        ("checkpoint_manifest_sha256", "full_checkpoint/MANIFEST.json", mf),
        ("raw_student_sha256", "raw_student/raw_student.pt", raw),
    ]:
        receipt[key] = digest(path)
        item = next(r for r in receipt["remote_files"] if r["file"] == name)
        item.update(sha256=digest(path), bytes=path.stat().st_size)
    if mutation is None:
        result = validate_reproduction_receipt(receipt, checkpoint_dir=tmp_path, raw_student_path=raw)
        assert result["verified"] is True
        assert result["payload_sha256"] == digest(full)
    else:
        with pytest.raises(ValueError):
            validate_reproduction_receipt(receipt, checkpoint_dir=tmp_path, raw_student_path=raw)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_reproduction_receipt_dict() -> dict:
    """A valid self_audit.b8.reproduction.v1 receipt dictionary."""
    ckpt_payload_sha = "1111111111111111111111111111111111111111111111111111111111111111"
    raw_student_sha = "2222222222222222222222222222222222222222222222222222222222222222"
    ckpt_manifest_sha = "3333333333333333333333333333333333333333333333333333333333333333"
    source_closure = "07809c252ae67d3ef17b148511c5973296c2367f18284a41766e8248fa3fdc2d"
    source_commit = "c06c4c9ae08610e047f1e960712f551690de4067"

    return {
        "schema": REPRODUCTION_RECEIPT_SCHEMA,
        "inference_state": "student",
        "new_reproduction": True,
        "original_payload_recovered": False,
        "checkpoint_payload_sha256": ckpt_payload_sha,
        "raw_student_sha256": raw_student_sha,
        "checkpoint_manifest_sha256": ckpt_manifest_sha,
        "source_commit": source_commit,
        "source_sha256": source_closure,
        "labeled_ids": list(PINNED_L8_PATIENT_IDS),
        "label_budget": 8,
        "metrics": {
            "val18": {
                "Macro4": 0.885,
                "foreground": 0.845,
                "class_dice": {"BG": 0.99, "RV": 0.84, "MYO": 0.83, "LV": 0.88},
                "patients": 18,
            },
            "test50": {
                "Macro4": 0.875,
                "foreground": 0.838,
                "class_dice": {"BG": 0.99, "RV": 0.83, "MYO": 0.82, "LV": 0.86},
                "patients": 50,
            },
        },
        "remote_files": [
            {
                "file": "full_checkpoint/weights.pt",
                "bytes": 1048576,
                "sha256": ckpt_payload_sha,
                "drive_id": "drive_file_id_weights",
                "readback": "SHA256_MATCH",
            },
            {
                "file": "full_checkpoint/MANIFEST.json",
                "bytes": 2048,
                "sha256": ckpt_manifest_sha,
                "drive_id": "drive_file_id_manifest",
                "readback": "SHA256_MATCH",
            },
            {
                "file": "raw_student/raw_student.pt",
                "bytes": 524288,
                "sha256": raw_student_sha,
                "drive_id": "drive_file_id_raw_student",
                "readback": "SHA256_MATCH",
            },
        ],
        "historical_reference": {"Macro4": 0.8800492397526454, "foreground": 0.8412102397257094},
        "quality_gate": "GO",
        "test50_status": "EXPOSED_DEVELOPMENT; final checkpoint only; no tuning",
    }


@pytest.fixture
def mock_git_repro_source(tmp_path) -> Path:
    """Create a temporary clean git repository mimicking the reproduction source."""
    src = tmp_path / "repro_source"
    src.mkdir()
    subprocess.run(["git", "init", str(src)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(src), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(src), "config", "user.email", "test@test.com"], check=True)

    pkg = src / "self_audit"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("# reproduction package\n")

    # model.py and window.py must have the EXACT historical SHA256 hashes!
    model_bytes = subprocess.check_output(
        ["git", "show", f"{HISTORICAL_SOURCE_COMMIT}:self_audit/model.py"]
    )
    window_bytes = subprocess.check_output(
        ["git", "show", f"{HISTORICAL_SOURCE_COMMIT}:self_audit/window.py"]
    )
    (pkg / "model.py").write_bytes(model_bytes)
    (pkg / "window.py").write_bytes(window_bytes)

    # Add dummy files for other self_audit modules
    (pkg / "engine.py").write_text("# reproduction engine\n")
    (src / "train.py").write_text("# reproduction train\n")

    subprocess.run(["git", "-C", str(src), "add", "."], check=True)
    subprocess.run(["git", "-C", str(src), "commit", "-m", "init repro source"], check=True)
    return src


# ---------------------------------------------------------------------------
# Test 1: Historical baseline pins & behavior unaffected
# ---------------------------------------------------------------------------

def test_historical_baseline_unaffected():
    """Verify historical baseline pins and protocol validation remain bit-for-bit unchanged."""
    assert HISTORICAL_SOURCE_COMMIT == "0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca"
    assert HISTORICAL_SOURCE_CLOSURE == "3c7fa85f00d54a6e087af3cd6c7045351fb72d532f1069bf09d95f9cd72359b7"
    assert HISTORICAL_MANIFEST_SHA256 == "03c1730f88941ada898bed7fcf1b9f13cdd8f08bcce4390539fec665c82e97de"
    assert HISTORICAL_PAYLOAD_SHA256 == "b36bcafe9613325d85ee70dabd6a36fef7c17429882967831a151c0cce6d8437"

    assert is_historical_baseline_checkpoint({
        "manifest_sha256": HISTORICAL_MANIFEST_SHA256,
        "source_sha256": HISTORICAL_SOURCE_CLOSURE,
    })
    assert not is_historical_baseline_checkpoint({
        "manifest_sha256": "different_manifest_hash",
        "source_sha256": HISTORICAL_SOURCE_CLOSURE,
    })

    # Default locked protocol validation passes with no reproduction receipt
    proto_path = ROOT / "configs" / "transfer_utility" / "protocol.json"
    if proto_path.is_file():
        proto = json.loads(proto_path.read_text())
        errors = validate_locked_protocol(proto)
        assert errors == [], f"Historical locked protocol validation failed: {errors}"


# ---------------------------------------------------------------------------
# Test 2: Reproduction source verification
# ---------------------------------------------------------------------------

def test_verify_reproduction_source_strict(mock_git_repro_source):
    """Reproduction source requires git repo, matching commit, closure, clean tree, and exact model hashes."""
    src = mock_git_repro_source
    head = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()

    # Compute closure
    pkg_files = sorted((src / "self_audit").glob("*.py")) + [src / "train.py"]
    closure = hashlib.sha256("".join(f"{p.name}:{digest(p)}\n" for p in pkg_files).encode()).hexdigest()

    # Successful verification
    res = verify_reproduction_source(src, expected_commit=head, expected_closure=closure)
    assert res["status"] == "PASS"
    assert res["source_commit"] == head
    assert res["source_closure"] == closure
    assert res["model_sha256"] == HISTORICAL_MODEL_SHA256
    assert res["window_sha256"] == HISTORICAL_WINDOW_SHA256

    # Failure 1: Non-git directory
    non_git = src / "subdir"
    non_git.mkdir()
    (non_git / "self_audit").mkdir()
    (non_git / "train.py").write_text("# non git")
    with pytest.raises(ValueError, match="not a git repository"):
        verify_reproduction_source(non_git, expected_commit=head, expected_closure=closure)

    # Failure 2: Missing expected commit or closure
    with pytest.raises(ValueError, match="requires explicit expected_commit"):
        verify_reproduction_source(src, expected_commit=None, expected_closure=closure)
    with pytest.raises(ValueError, match="requires explicit expected_commit"):
        verify_reproduction_source(src, expected_commit=head, expected_closure=None)

    # Failure 3: Commit mismatch
    with pytest.raises(ValueError, match="commit mismatch"):
        verify_reproduction_source(src, expected_commit="0000000000000000000000000000000000000000", expected_closure=closure)

    # Failure 4: Closure mismatch
    with pytest.raises(ValueError, match="closure mismatch"):
        verify_reproduction_source(src, expected_commit=head, expected_closure="0000000000000000000000000000000000000000000000000000000000000000")

    # Failure 5: Dirty working tree
    (src / "self_audit" / "engine.py").write_text("# dirty edit")
    with pytest.raises(ValueError, match="uncommitted changes"):
        verify_reproduction_source(src, expected_commit=head, expected_closure=closure)
    subprocess.run(["git", "-C", str(src), "checkout", "--", "."], check=True)

    # Failure 6: Altered model.py
    (src / "self_audit" / "model.py").write_text("# altered model architecture")
    subprocess.run(["git", "-C", str(src), "commit", "-am", "bad model"], check=True)
    new_head = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    with pytest.raises(ValueError, match="model.py hash mismatch"):
        verify_reproduction_source(src, expected_commit=new_head, expected_closure=closure)


# ---------------------------------------------------------------------------
# Test 3: Reproduction receipt validation fail-closed modes
# ---------------------------------------------------------------------------

def test_validate_reproduction_receipt_failures(valid_reproduction_receipt_dict, tmp_path):
    """Verify strict fail-closed validation of reproduction receipts."""
    # 1. Valid receipt passes validation (unverified without physical paths)
    rec = copy.deepcopy(valid_reproduction_receipt_dict)
    res = validate_reproduction_receipt(rec)
    assert res["teacher_kind"] == "reproduction"
    assert res["payload_sha256"] == rec["checkpoint_payload_sha256"]
    assert res["verified"] is False  # Claims alone cannot stamp physical verified!
    assert res["physical_payload_status"] == "BLOCKED_MISSING_PAYLOAD"

    # 2. Schema mismatch
    bad_schema = copy.deepcopy(valid_reproduction_receipt_dict)
    bad_schema["schema"] = "self_audit.checkpoint.v2"
    with pytest.raises(ValueError, match="invalid reproduction receipt schema"):
        validate_reproduction_receipt(bad_schema)

    # 3. Macro4 < 0.87 or > 1.0
    bad_macro = copy.deepcopy(valid_reproduction_receipt_dict)
    bad_macro["metrics"]["test50"]["Macro4"] = 0.869
    with pytest.raises(ValueError, match="Macro4=0.869 must be finite in"):
        validate_reproduction_receipt(bad_macro)

    bad_macro_high = copy.deepcopy(valid_reproduction_receipt_dict)
    bad_macro_high["metrics"]["test50"]["Macro4"] = 1.001
    with pytest.raises(ValueError, match="Macro4=1.001 must be finite in"):
        validate_reproduction_receipt(bad_macro_high)

    # 3b. Remote file bytes non-int or <= 0
    bad_bytes_bool = copy.deepcopy(valid_reproduction_receipt_dict)
    bad_bytes_bool["remote_files"][0]["bytes"] = True
    with pytest.raises(ValueError, match="must be positive int"):
        validate_reproduction_receipt(bad_bytes_bool)

    bad_bytes_zero = copy.deepcopy(valid_reproduction_receipt_dict)
    bad_bytes_zero["remote_files"][0]["bytes"] = 0
    with pytest.raises(ValueError, match="must be positive int"):
        validate_reproduction_receipt(bad_bytes_zero)

    # 4. Quality gate != "GO"
    bad_gate = copy.deepcopy(valid_reproduction_receipt_dict)
    bad_gate["quality_gate"] = "NO_GO"
    with pytest.raises(ValueError, match="quality_gate is 'NO_GO'; must be 'GO'"):
        validate_reproduction_receipt(bad_gate)

    # 5. Remote file readback != "SHA256_MATCH"
    bad_readback = copy.deepcopy(valid_reproduction_receipt_dict)
    bad_readback["remote_files"][0]["readback"] = "FAILED"
    with pytest.raises(ValueError, match="readback verification status 'FAILED'"):
        validate_reproduction_receipt(bad_readback)

    # 6. Remote file missing drive_id
    bad_drive = copy.deepcopy(valid_reproduction_receipt_dict)
    bad_drive["remote_files"][0]["drive_id"] = ""
    with pytest.raises(ValueError, match="missing verified drive_id"):
        validate_reproduction_receipt(bad_drive)

    # 7. Contradictory duplicate remote_files
    dup_contradictory = copy.deepcopy(valid_reproduction_receipt_dict)
    dup_contradictory["remote_files"].append({
        "file": "full_checkpoint/weights.pt",
        "bytes": 999999,
        "sha256": "4444444444444444444444444444444444444444444444444444444444444444",
        "drive_id": "drive_dup",
        "readback": "SHA256_MATCH",
    })
    with pytest.raises(ValueError, match="duplicate contradictory remote_file entry"):
        validate_reproduction_receipt(dup_contradictory)

    # 8. Essential file remote hash mismatch
    bad_hash_map = copy.deepcopy(valid_reproduction_receipt_dict)
    bad_hash_map["remote_files"][0]["sha256"] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(ValueError, match="remote_files hash mismatch for 'full_checkpoint/weights.pt'"):
        validate_reproduction_receipt(bad_hash_map)

    # 9. Wrong L8 labeled patient IDs
    bad_lids = copy.deepcopy(valid_reproduction_receipt_dict)
    bad_lids["labeled_ids"] = ["patient001", "patient002", "patient003", "patient004", "patient005", "patient006", "patient007", "patient008"]
    with pytest.raises(ValueError, match="reproduction labeled_ids mismatch"):
        validate_reproduction_receipt(bad_lids)

    # 10. File hash mismatch when validating receipt from disk
    receipt_file = tmp_path / "TEACHER_IDENTITY.json"
    receipt_file.write_text(json.dumps(valid_reproduction_receipt_dict))
    with pytest.raises(ValueError, match="reproduction receipt file hash mismatch"):
        validate_reproduction_receipt(receipt_file, expected_receipt_sha256="0000000000000000000000000000000000000000000000000000000000000000")

    # 11. Physical checkpoint MANIFEST size mismatch vs remote bytes
    phys_dir = tmp_path / "phys_cp"
    phys_dir.mkdir()
    phys_mfile = phys_dir / "MANIFEST.json"
    m_content = json.dumps({
        "schema": "self_audit.checkpoint.v2",
        "config": {"method": "ema_full"},
        "step": 10000,
        "source_sha256": valid_reproduction_receipt_dict["source_sha256"],
        "labeled_ids": valid_reproduction_receipt_dict["labeled_ids"],
    })
    phys_mfile.write_text(m_content)
    m_sha = digest(phys_mfile)
    m_bytes = phys_mfile.stat().st_size

    phys_rec = copy.deepcopy(valid_reproduction_receipt_dict)
    phys_rec["checkpoint_manifest_sha256"] = m_sha
    for rf in phys_rec["remote_files"]:
        if rf["file"] == "full_checkpoint/MANIFEST.json":
            rf["sha256"] = m_sha
            rf["bytes"] = m_bytes + 1  # Intentionally wrong bytes count!

    with pytest.raises(ValueError, match="full checkpoint MANIFEST.json size mismatch"):
        validate_reproduction_receipt(phys_rec, checkpoint_dir=phys_dir)

    # 12. Physical checkpoint MANIFEST step != 10000
    phys_bad_step = phys_dir / "bad_step"
    phys_bad_step.mkdir()
    (phys_bad_step / "MANIFEST.json").write_text(json.dumps({
        "schema": "self_audit.checkpoint.v2", "config": {"method": "ema_full"}, "step": 9000,
        "source_sha256": valid_reproduction_receipt_dict["source_sha256"],
        "labeled_ids": valid_reproduction_receipt_dict["labeled_ids"],
    }))
    bad_step_sha = digest(phys_bad_step / "MANIFEST.json")
    phys_rec_bad_step = copy.deepcopy(valid_reproduction_receipt_dict)
    phys_rec_bad_step["checkpoint_manifest_sha256"] = bad_step_sha
    for rf in phys_rec_bad_step["remote_files"]:
        if rf["file"] == "full_checkpoint/MANIFEST.json":
            rf["sha256"] = bad_step_sha
            rf["bytes"] = (phys_bad_step / "MANIFEST.json").stat().st_size
    with pytest.raises(ValueError, match="checkpoint MANIFEST.json step must be 10000"):
        validate_reproduction_receipt(phys_rec_bad_step, checkpoint_dir=phys_bad_step)

    # 13. Tensor mismatch between raw_student.pt and checkpoint student weights
    w_file = phys_dir / "weights.pt"
    raw_pt = phys_dir / "raw_student.pt"
    t1 = {"conv.weight": torch.tensor([1.0, 2.0])}
    t2 = {"conv.weight": torch.tensor([1.0, 99.0])}
    torch.save(t1, raw_pt)
    torch.save({"student": t2, "method": "ema_full"}, w_file)

    for rf in phys_rec["remote_files"]:
        if rf["file"] == "full_checkpoint/MANIFEST.json":
            rf["bytes"] = m_bytes
        elif rf["file"] == "full_checkpoint/weights.pt":
            rf["sha256"] = digest(w_file)
            rf["bytes"] = w_file.stat().st_size
        elif rf["file"] == "raw_student/raw_student.pt":
            rf["sha256"] = digest(raw_pt)
            rf["bytes"] = raw_pt.stat().st_size
    phys_rec["checkpoint_payload_sha256"] = digest(w_file)
    phys_rec["raw_student_sha256"] = digest(raw_pt)

    with pytest.raises(ValueError, match="weight tensor mismatch"):
        validate_reproduction_receipt(phys_rec, checkpoint_dir=phys_dir, raw_student_path=raw_pt)


# ---------------------------------------------------------------------------
# Test 4: Explicit teacher_kind='reproduction' required
# ---------------------------------------------------------------------------

def test_validate_teacher_identity_no_loose_inference(valid_reproduction_receipt_dict):
    """Passing a dict with new_reproduction=True without teacher_kind='reproduction' defaults to canonical and fails."""
    loose_dict = copy.deepcopy(valid_reproduction_receipt_dict)
    # No teacher_kind specified: defaults to canonical, which requires mandatory quality_receipt mapping with Macro4
    with pytest.raises(ValueError, match="canonical teacher requires mandatory 64-hex 'manifest_sha256'"):
        validate_teacher_identity(loose_dict)

    # Explicit teacher_kind="reproduction" succeeds
    explicit_dict = copy.deepcopy(valid_reproduction_receipt_dict)
    explicit_dict["teacher_kind"] = "reproduction"
    receipt = validate_teacher_identity(explicit_dict)
    assert receipt["teacher_kind"] == "reproduction"
    assert receipt["inference_state"] == "student"


# ---------------------------------------------------------------------------
# Test 5: Protocol derivation & locking
# ---------------------------------------------------------------------------

def test_derive_reproduction_protocol(valid_reproduction_receipt_dict):
    """derive_reproduction_protocol updates only teacher and runtime_policy, locking all other parameters."""
    base_proto = json.loads((ROOT / "configs" / "transfer_utility" / "protocol.json").read_text())
    cache_sha = "7777777777777777777777777777777777777777777777777777777777777777"

    derived = derive_reproduction_protocol(
        base_proto,
        valid_reproduction_receipt_dict,
        cache_sha,
        deterministic_algorithms=True,
    )

    # 1. Teacher section updated with reproduction hashes
    assert derived["teacher"]["source_manifest_sha256"] == valid_reproduction_receipt_dict["checkpoint_manifest_sha256"]
    assert derived["teacher"]["source_payload_sha256"] == valid_reproduction_receipt_dict["checkpoint_payload_sha256"]
    assert derived["teacher"]["cache_sha256"] == cache_sha
    assert derived["teacher"]["state"] == "student"
    assert derived["teacher"]["lineage"]["budget_k"] == 8
    assert derived["teacher"]["lineage"]["exact_labeled_ids"] == list(PINNED_L8_PATIENT_IDS)
    assert derived["teacher"]["reproduction"]["quality_gate"] == "GO"
    assert derived["teacher"]["reproduction"]["Macro4"] >= 0.87

    # 2. Runtime policy added
    assert derived["runtime_policy"]["deterministic_algorithms"] is True

    # 3. All other sections are bit-for-bit identical to base protocol
    for key in ("budget", "hyperparameters", "selector_parameters", "architecture", "data", "evaluation", "decision_rules", "arms", "conditional_arms"):
        assert derived[key] == base_proto[key], f"Section {key} differed from base protocol!"

    # 4. Validates cleanly against validate_locked_protocol
    errors = validate_locked_protocol(derived, reproduction_receipt=valid_reproduction_receipt_dict)
    assert errors == [], f"Derived protocol failed validation: {errors}"

    # 5. Fail closed if budget or hyperparameters are tampered with
    tampered_steps = copy.deepcopy(derived)
    tampered_steps["budget"]["total_steps"] = 1000
    assert len(validate_locked_protocol(tampered_steps, reproduction_receipt=valid_reproduction_receipt_dict)) > 0

    tampered_lr = copy.deepcopy(derived)
    tampered_lr["hyperparameters"]["lr"] = 0.001
    assert len(validate_locked_protocol(tampered_lr, reproduction_receipt=valid_reproduction_receipt_dict)) > 0


# ---------------------------------------------------------------------------
# Test 6: CLI flags in cache generator and pilot runner
# ---------------------------------------------------------------------------

def test_cache_and_pilot_cli_reproduction_support(valid_reproduction_receipt_dict, tmp_path):
    """Test CLI argument parsing and validation for reproduction flags."""
    receipt_file = tmp_path / "TEACHER_IDENTITY.json"
    receipt_file.write_text(json.dumps(valid_reproduction_receipt_dict))
    receipt_sha = digest(receipt_file)

    data_manifest, data_sha = synthetic_compact_fixture(tmp_path / "cli_data", n_labeled=1, n_unlabeled=1)

    # 1. build_compact_teacher_cache.py rejects reproduction receipt without --checkpoint-dir
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_compact_teacher_cache.py"),
            "--data-manifest", str(data_manifest),
            "--data-sha256", data_sha,
            "--output-dir", str(tmp_path / "cache_out"),
            "--reproduction-receipt", str(receipt_file),
            "--reproduction-receipt-sha256", receipt_sha,
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "--checkpoint-dir" in res.stderr

    # 2. Both CLIs reject --reproduction-receipt without mandatory --reproduction-receipt-sha256
    res_no_sha_cache = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_compact_teacher_cache.py"),
            "--data-manifest", str(data_manifest),
            "--data-sha256", data_sha,
            "--output-dir", str(tmp_path / "cache_out2"),
            "--reproduction-receipt", str(receipt_file),
            "--checkpoint-dir", str(tmp_path),
            "--reproduction-source", str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert res_no_sha_cache.returncode != 0
    assert "--reproduction-receipt requires --reproduction-receipt-sha256" in res_no_sha_cache.stderr

    res_no_sha_pilot = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_transfer_utility_pilot.py"),
            "--protocol", str(ROOT / "configs" / "transfer_utility" / "protocol.json"),
            "--out", str(tmp_path / "pilot_out2"),
            "--reproduction-receipt", str(receipt_file),
        ],
        capture_output=True,
        text=True,
    )
    assert res_no_sha_pilot.returncode != 0
    assert "--reproduction-receipt requires --reproduction-receipt-sha256" in res_no_sha_pilot.stderr

    # 3. run_transfer_utility_pilot.py supports --derive-reproduction-protocol
    out_proto = tmp_path / "derived_proto.json"
    res_proto = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_transfer_utility_pilot.py"),
            "--protocol", str(ROOT / "configs" / "transfer_utility" / "protocol.json"),
            "--out", str(tmp_path / "pilot_out"),
            "--reproduction-receipt", str(receipt_file),
            "--reproduction-receipt-sha256", receipt_sha,
            "--teacher-cache-sha256", "8" * 64,
            "--derive-reproduction-protocol", str(out_proto),
        ],
        capture_output=True,
        text=True,
    )
    assert res_proto.returncode == 0, f"derive-protocol failed: {res_proto.stderr}"
    assert out_proto.is_file()
    derived_data = json.loads(out_proto.read_text())
    assert derived_data["teacher"]["source_payload_sha256"] == valid_reproduction_receipt_dict["checkpoint_payload_sha256"]
    assert derived_data["runtime_policy"]["deterministic_algorithms"] is True


# ---------------------------------------------------------------------------
# Test 7: Integration check: profile and all 4 arm configs receive reproduction pins,
#         runtime_policy, and custom training cost (no archived fallback)
# ---------------------------------------------------------------------------

def test_emitted_configs_receive_reproduction_pins_and_runtime_policy(valid_reproduction_receipt_dict, tmp_path):
    """Integration test verifying that profile and all 4 arm configs receive
    reproduction pins, deterministic_algorithms=True, and measured source_training_cost."""
    rec = copy.deepcopy(valid_reproduction_receipt_dict)
    rec["parent_receipt_sha256"] = "1" * 64
    rec["source_training_cost"] = {
        "status": "MEASURED_BY_TRAINER",
        "steps": 10000,
        "elapsed_seconds": 1234.56,
        "source": "/workspace/training/REPORT.json",
    }

    # Create mock teacher cache
    cache_dir = tmp_path / "mock_teacher_cache"
    cache_dir.mkdir()
    cache_manifest = cache_dir / "CACHE_MANIFEST.json"

    # Teacher identity inside cache manifest
    teacher_id = validate_reproduction_receipt(rec)
    teacher_id["teacher_kind"] = "reproduction"

    cache_data = {
        "schema": "self_audit.compact_teacher_cache.v1",
        "teacher_identity": teacher_id,
        "train_manifest_sha256": "3" * 64,
        "patients": {},
    }
    cache_manifest.write_text(json.dumps(cache_data))
    cache_sha = digest(cache_manifest)

    base_proto = json.loads((ROOT / "configs" / "transfer_utility" / "protocol.json").read_text())
    derived_proto = derive_reproduction_protocol(
        base_proto,
        rec,
        cache_sha,
        deterministic_algorithms=True,
    )

    # 1. Profile config receives reproduction pins and deterministic runtime
    profile_base = ROOT / "configs" / "transfer_utility" / "profile_32steps.json"
    profile_bound = bind_teacher_config(
        json.loads(profile_base.read_text()),
        cache_path=cache_manifest,
        cache_sha=cache_sha,
        protocol=derived_proto,
        reproduction_receipt=rec,
    )
    assert profile_bound["deterministic_algorithms"] is True
    assert profile_bound["runtime_policy"]["deterministic_algorithms"] is True
    assert profile_bound["teacher"]["source_manifest_sha256"] == rec["checkpoint_manifest_sha256"]
    assert profile_bound["teacher"]["source_payload_sha256"] == rec["checkpoint_payload_sha256"]
    assert profile_bound["teacher"]["source_training_cost"]["elapsed_seconds"] == 1234.56
    assert profile_bound["teacher"]["source_training_cost"]["status"] == "MEASURED_BY_TRAINER"
    assert profile_bound["teacher"]["parent_receipt_sha256"] == "1" * 64

    # 2. All 4 primary arm configs receive reproduction pins, deterministic runtime,
    #    and validate cleanly against derived protocol
    arm_names = [
        "plain_kd",
        "scalar_guard_original",
        "scalar_guard_larger_tie",
        "scalar_guard_cross_patient_utility",
    ]
    for arm_name in arm_names:
        arm_cfg_path = ROOT / "configs" / "transfer_utility" / f"{arm_name}.json"
        arm_bound = bind_teacher_config(
            json.loads(arm_cfg_path.read_text()),
            cache_path=cache_manifest,
            cache_sha=cache_sha,
            protocol=derived_proto,
            reproduction_receipt=rec,
        )

        # Assert bound reproduction identity
        assert arm_bound["deterministic_algorithms"] is True
        assert arm_bound["runtime_policy"]["deterministic_algorithms"] is True
        assert arm_bound["teacher"]["source_manifest_sha256"] == rec["checkpoint_manifest_sha256"]
        assert arm_bound["teacher"]["source_payload_sha256"] == rec["checkpoint_payload_sha256"]
        assert arm_bound["teacher"]["lineage"]["exact_labeled_ids"] == list(PINNED_L8_PATIENT_IDS)
        assert arm_bound["teacher"]["source_training_cost"]["elapsed_seconds"] == 1234.56
        assert arm_bound["teacher"]["source_training_cost"]["elapsed_seconds"] != 2440.043427208904
        assert arm_bound["teacher"]["parent_receipt_sha256"] == "1" * 64

        # Validate against derived protocol
        validate_arm_config(arm_bound, derived_proto, arm_name, reproduction_receipt=rec)

    # 3. If receipt has no source_training_cost, it defaults to "NOT_KNOWN" and never falls back
    rec_no_cost = copy.deepcopy(valid_reproduction_receipt_dict)
    rec_no_cost.pop("source_training_cost", None)
    teacher_id_no_cost = validate_reproduction_receipt(rec_no_cost)
    teacher_id_no_cost["teacher_kind"] = "reproduction"
    assert teacher_id_no_cost["source_training_cost"] == "NOT_KNOWN"

    cache_data_no_cost = {
        "schema": "self_audit.compact_teacher_cache.v1",
        "teacher_identity": teacher_id_no_cost,
        "train_manifest_sha256": "3" * 64,
        "patients": {},
    }
    cache_manifest_no_cost = tmp_path / "cache_no_cost.json"
    cache_manifest_no_cost.write_text(json.dumps(cache_data_no_cost))
    cache_sha_no_cost = digest(cache_manifest_no_cost)

    arm_bound_no_cost = bind_teacher_config(
        json.loads((ROOT / "configs" / "transfer_utility" / "plain_kd.json").read_text()),
        cache_path=cache_manifest_no_cost,
        cache_sha=cache_sha_no_cost,
        protocol=derived_proto,
        reproduction_receipt=rec_no_cost,
    )
    assert arm_bound_no_cost["teacher"]["source_training_cost"] == "NOT_KNOWN"
    assert arm_bound_no_cost["teacher"]["source_training_cost"] != 2440.043427208904
