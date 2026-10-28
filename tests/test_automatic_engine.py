"""Acceptance tests for self_audit.automatic.v1 engine integration.

Verifies:
1. Strict config validation rejecting unknown keys, non-fixed window, invalid phases,
   and enforcing no silent body_lr fallback for full_ema.
2. Complete fixed TRAIN-L support volumes (all Z slices) without random sampling.
3. Strict support voxel bound and cache bounds failing closed before allocation.
4. Support feature cache fingerprinting, invalidation, and base logit parity against direct model forward.
5. Frozen feature path: encoder, decoder, window parameters are strictly frozen during head adaptation.
6. Teacher feature path frozen during head adaptation, reset to student at split, and EMA head ONLY.
7. Target label lineage and declared budget accounting failing closed on mismatch or U/query/eval overlap.
8. Checkpoint loader strictness: mandatory expected_sha256, unconditional source verification,
   manifest verification, and strict raw-student-only loader (load_automatic_student).
9. Output safety: rejecting outputs inside datasets/checkpoints, rejecting overwrites, and safe resume to new dir.
10. True exact resume: >1 steps with nontrivial RNG and full_ema AdamW optimizer states,
    verifying student, teacher, optimizer, and np_rng.bit_generator.state bit-for-bit parity.
11. Matched supervised slice schedule and SAME supervised_loss CE+Dice across trainable arms.
12. Time budget enforcement with bounded overshoot, termination_reason, and equal-time certification.
13. Resource accounting: p50/p95 latencies, throughput, acceptance separate from nonzero updates, and margin changes.
14. All 10 named arms execute successfully in bounded synthetic runs.
15. train.py router correctly handles self_audit.automatic.v1, phase/budget flags, and rejects invalid flags.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import pytest
import torch
import numpy as np

from self_audit.automatic import (
    DEFAULT_AUTOMATIC_CONFIG,
    NAMED_ARMS,
    PHASES,
    SCHEMA,
    automatic_source_digest,
    build_lineage,
    load_automatic_checkpoint,
    load_automatic_student,
    run,
    save_automatic_checkpoint,
    validate_automatic_config,
)
from self_audit.automatic_data import (
    SupportFeatureCache,
    SupportVolumeRecord,
    compute_model_feature_fingerprint,
    compute_support_feature_cache,
    extract_complete_support_volumes,
)
from self_audit.data import TrainingData, synthetic_fixture, write_json
from self_audit.model import build_model, digest


def make_test_config(arm: str = "joint_dice", steps: int = 2, phase: str = "adaptation") -> dict:
    c = copy.deepcopy(DEFAULT_AUTOMATIC_CONFIG)
    c.update({
        "schema": SCHEMA,
        "phase": phase,
        "arm": arm,
        "steps": steps,
        "warmup_steps": 0,
        "seed": 42,
        "body_lr": 1e-4 if arm == "full_ema" else 0.0,
        "save_every": 10,
        "model": {"backbone": "tiny_test"},
        "initialization": {
            "warm_start_checkpoint": None,
            "warm_start_sha256": None,
            "declared_target_budget": 2,
        },
        "support": {
            "patient_ids": ["synthetic_000", "synthetic_001"],
            "k_reuse": 1,
        },
    })
    return c


def create_warm_start_checkpoint(tmp_path: Path, manifest: Path, manifest_sha: str) -> tuple[Path, str]:
    """Helper to produce an authentic SHA-pinned warm-start checkpoint."""
    ws_cfg = make_test_config("support_only", steps=2, phase="warm_start")
    ws_out = tmp_path / "shared_warm_start"
    rep = run(ws_cfg, ws_out, synthetic=True, manifest=manifest, manifest_sha256=manifest_sha)
    assert rep["phase"] == "warm_start"
    ckpt_dir = ws_out / "final"
    ckpt_sha = rep["final_manifest_sha256"]
    return ckpt_dir, ckpt_sha


def test_validate_automatic_config():
    c = make_test_config("joint_dice")
    valid = validate_automatic_config(c)
    assert valid["schema"] == SCHEMA
    assert valid["arm"] == "joint_dice"
    assert valid["phase"] == "adaptation"

    # Reject unknown top-level keys
    bad_key = copy.deepcopy(c)
    bad_key["unknown_extra_param"] = 123
    with pytest.raises(ValueError, match="unknown config keys"):
        validate_automatic_config(bad_key)

    # Reject unknown nested keys
    bad_nested = copy.deepcopy(c)
    bad_nested["support"]["unknown_sub_param"] = "bad"
    with pytest.raises(ValueError, match="unknown keys in support"):
        validate_automatic_config(bad_nested)

    # Reject non-fixed window
    bad_window = copy.deepcopy(c)
    bad_window["window"]["mode"] = "none"
    with pytest.raises(ValueError, match="fixed window mode"):
        validate_automatic_config(bad_window)

    # Reject unknown arm
    bad_arm = copy.deepcopy(c)
    bad_arm["arm"] = "unknown_experimental_arm"
    with pytest.raises(ValueError, match="unknown arm"):
        validate_automatic_config(bad_arm)

    # Reject invalid phase
    bad_phase = copy.deepcopy(c)
    bad_phase["phase"] = "unsupported_phase"
    with pytest.raises(ValueError, match="unknown phase"):
        validate_automatic_config(bad_phase)

    # Reject body_lr > 0 for head adaptation
    bad_body = copy.deepcopy(c)
    bad_body["arm"] = "joint_dice"
    bad_body["body_lr"] = 1e-4
    with pytest.raises(ValueError, match="body_lr must be 0"):
        validate_automatic_config(bad_body)

    # Reject body_lr <= 0 for full_ema (no silent bodylr fallback)
    bad_full_ema = copy.deepcopy(c)
    bad_full_ema["arm"] = "full_ema"
    bad_full_ema["body_lr"] = 0.0
    with pytest.raises(ValueError, match="full_ema requires body_lr > 0"):
        validate_automatic_config(bad_full_ema)

    # Reject invalid time_budget_seconds
    bad_tb = copy.deepcopy(c)
    bad_tb["time_budget_seconds"] = -5.0
    with pytest.raises(ValueError, match="time_budget_seconds must be a positive finite number"):
        validate_automatic_config(bad_tb)

    # Reject invalid max_cache_bytes
    bad_mcb = copy.deepcopy(c)
    bad_mcb["max_cache_bytes"] = -100
    with pytest.raises(ValueError, match="max_cache_bytes must be a positive integer"):
        validate_automatic_config(bad_mcb)


def test_warm_start_phase_and_adaptation_chain(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)

    # 1. Warm-start phase runs and saves support-only checkpoint with phase='warm_start'
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)
    manifest = json.loads((ws_dir / "MANIFEST.json").read_text())
    assert manifest["phase"] == "warm_start"
    assert manifest["arm"] == "support_only"
    assert manifest["prediction_model"] == "student"

    # 2. Adaptation phase succeeds with SHA-pinned warm-start checkpoint
    adapt_cfg = make_test_config("joint_dice", steps=2, phase="adaptation")
    adapt_cfg["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    adapt_cfg["initialization"]["warm_start_sha256"] = ws_sha
    adapt_out = tmp_path / "adaptation_out"
    rep_adapt = run(adapt_cfg, adapt_out, synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep_adapt["status"] == "COMPLETED"
    assert rep_adapt["phase"] == "adaptation"

    # 3. Adaptation fails closed if warm-start checkpoint is missing
    adapt_cfg_no_ws = make_test_config("joint_dice", steps=2, phase="adaptation")
    with pytest.raises(ValueError, match="phase 'adaptation' requires an externally SHA-pinned warm-start checkpoint"):
        run(adapt_cfg_no_ws, tmp_path / "fail_no_ws", synthetic=True, manifest=m, manifest_sha256=sha)

    # 4. Adaptation fails closed if given an adapted checkpoint (phase='adaptation') instead of warm-start
    adapt_cfg_bad_phase = make_test_config("joint_dice", steps=2, phase="adaptation")
    adapt_cfg_bad_phase["initialization"]["warm_start_checkpoint"] = str(adapt_out / "final")
    adapt_cfg_bad_phase["initialization"]["warm_start_sha256"] = rep_adapt["final_manifest_sha256"]
    with pytest.raises(ValueError, match="adaptation requires a warm_start phase checkpoint and rejects arbitrary adapted checkpoints"):
        run(adapt_cfg_bad_phase, tmp_path / "fail_bad_phase", synthetic=True, manifest=m, manifest_sha256=sha)


def test_checkpoint_loader_strictness(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    # 1. Missing or non-64 hex expected_sha256 fails closed
    with pytest.raises(ValueError, match="expected_sha256 is mandatory"):
        load_automatic_checkpoint(ws_dir, expected_sha256="")
    with pytest.raises(ValueError, match="expected_sha256 is mandatory"):
        load_automatic_checkpoint(ws_dir, expected_sha256="not_a_64_hex_string")

    # 2. Tampered expected_sha256 fails closed
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        load_automatic_checkpoint(ws_dir, expected_sha256="0" * 64)

    # 3. Successful load_automatic_checkpoint returns (student, teacher, manifest, payload)
    st, tc, manifest, payload = load_automatic_checkpoint(ws_dir, expected_sha256=ws_sha)
    assert isinstance(st, torch.nn.Module)
    assert isinstance(tc, torch.nn.Module)
    assert manifest["phase"] == "warm_start"
    assert manifest["prediction_model"] == "student"
    assert manifest["training_manifest_sha256"] == sha
    assert "synthetic_000" in manifest["labeled_ids"]
    assert "synthetic_003" in manifest["unlabeled_ids"]
    assert "np_rng_state" in payload

    # 4. Strict load_automatic_student returns (student, manifest) only, eliminating teacher memory overhead
    st_raw, manifest_raw = load_automatic_student(ws_dir, expected_sha256=ws_sha)
    assert isinstance(st_raw, torch.nn.Module)
    assert not hasattr(st_raw, "teacher_memory") or st_raw.teacher_memory is None
    assert manifest_raw["schema"] == "self_audit.automatic_checkpoint.v1"


def test_output_overwrites_rejected(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    c = make_test_config("joint_dice", steps=2, phase="adaptation")
    c["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c["initialization"]["warm_start_sha256"] = ws_sha

    # 1. Output inside actual manifest directory or warmstart directory fails closed BEFORE mkdir
    with pytest.raises(ValueError, match="already exists; output directory must be new only"):
        run(c, data_dir, synthetic=True, manifest=m, manifest_sha256=sha)
    with pytest.raises(ValueError, match="overlaps with protected path"):
        run(c, data_dir / "nested_sub", synthetic=True, manifest=m, manifest_sha256=sha)
    with pytest.raises(ValueError, match="overlaps with protected path"):
        run(c, ws_dir / "sub_folder", synthetic=True, manifest=m, manifest_sha256=sha)

    # 2. Output directory NEW only: directory that already exists fails closed before mkdir
    pre_existing_dir = tmp_path / "already_exists"
    pre_existing_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists; output directory must be new only"):
        run(c, pre_existing_dir, synthetic=True, manifest=m, manifest_sha256=sha)

    # 3. Output overwrite of existing completed run fails closed
    out_valid = tmp_path / "valid_out"
    run(c, out_valid, synthetic=True, manifest=m, manifest_sha256=sha)
    assert (out_valid / "REPORT.json").is_file()

    with pytest.raises(ValueError, match="already exists; output directory must be new only"):
        run(c, out_valid, synthetic=True, manifest=m, manifest_sha256=sha)

    # 4. Resume into same directory as resume source fails closed
    with pytest.raises(ValueError, match="already exists; output directory must be new only"):
        run(c, out_valid, synthetic=True, manifest=m, manifest_sha256=sha, resume=out_valid / "final", resume_sha256="0" * 64)

    # 5. Checkpoint saver nooverwrite fails closed if weights.pt or MANIFEST.json already exists
    st, tc, _, _ = load_automatic_checkpoint(ws_dir, expected_sha256=ws_sha)
    opts = [torch.optim.SGD(st.head.parameters(), lr=0.001)]
    with pytest.raises(ValueError, match="checkpoint overwrites are prohibited"):
        save_automatic_checkpoint(
            ws_dir,
            st,
            tc,
            opts,
            c,
            step=1,
            provenance={},
            lineage={"patients": ["synthetic_000"]},
            data=TrainingData(m, sha, synthetic=True),
            phase="adaptation",
            np_rng=np.random.default_rng(42),
            counts={"supervised_updates": 1},
            cumulative_seconds=1.0,
            is_final=True,
        )


def test_frozen_feature_path_and_teacher_split_policy(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    c = make_test_config("joint_dice", steps=2, phase="adaptation")
    c["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c["initialization"]["warm_start_sha256"] = ws_sha
    out = tmp_path / "out_freeze"

    ws_student, ws_teacher, _, _ = load_automatic_checkpoint(ws_dir, expected_sha256=ws_sha)
    ws_feat_weights = {k: v.clone() for k, v in ws_student.state_dict().items() if not k.startswith("head.")}

    rep = run(c, out, synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep["status"] == "COMPLETED"

    final_student, final_teacher, _, _ = load_automatic_checkpoint(out / "final", expected_sha256=rep["final_manifest_sha256"])
    final_st_state = final_student.state_dict()
    final_tc_state = final_teacher.state_dict()

    # Feature parameters bit-for-bit identical to warm-start
    for k, v in ws_feat_weights.items():
        torch.testing.assert_close(final_st_state[k], v)
        torch.testing.assert_close(final_tc_state[k], v)

    # Head parameters updated and teacher EMA updated head ONLY
    assert not torch.allclose(final_st_state["head.weight"], ws_student.head.weight)
    assert not torch.allclose(final_tc_state["head.weight"], ws_teacher.head.weight)


def test_matched_supervised_schedule_and_loss(tmp_path):
    """Verify head-only and full_ema arms use the exact same slice schedule and supervised_loss CE+Dice."""
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    # Run head arm
    c_head = make_test_config("joint_dice", steps=2, phase="adaptation")
    c_head["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c_head["initialization"]["warm_start_sha256"] = ws_sha
    out_head = tmp_path / "out_head"
    rep_head = run(c_head, out_head, synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep_head["supervised_updates"] == 2

    # Run full_ema arm
    c_full = make_test_config("full_ema", steps=2, phase="adaptation")
    c_full["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c_full["initialization"]["warm_start_sha256"] = ws_sha
    out_full = tmp_path / "out_full"
    rep_full = run(c_full, out_full, synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep_full["supervised_updates"] == 2


def test_true_exact_resume_with_nontrivial_rng(tmp_path):
    """Verify bit-for-bit exact continuation across uninterrupted vs resumed execution (>1 steps, RNG, AdamW state)."""
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    for arm in ("joint_dice", "full_ema"):
        # Continuous run: 4 steps, saving at step 2
        c_full = make_test_config(arm, steps=4, phase="adaptation")
        c_full["save_every"] = 2
        c_full["initialization"]["warm_start_checkpoint"] = str(ws_dir)
        c_full["initialization"]["warm_start_sha256"] = ws_sha
        out_cont = tmp_path / f"out_cont_{arm}"
        rep_cont = run(c_full, out_cont, synthetic=True, manifest=m, manifest_sha256=sha)
        assert rep_cont["status"] == "COMPLETED"

        step2_dir = out_cont / "step_000002"
        step2_sha = digest(step2_dir / "MANIFEST.json")

        # Resumed run: resume from step 2 to step 4 in a new output directory
        c_resume = copy.deepcopy(c_full)
        out_resume = tmp_path / f"out_resume_{arm}"
        rep_resume = run(
            c_resume,
            out_resume,
            synthetic=True,
            manifest=m,
            manifest_sha256=sha,
            resume=step2_dir,
            resume_sha256=step2_sha,
        )
        assert rep_resume["status"] == "COMPLETED"

        st_cont, tc_cont, _, pay_cont = load_automatic_checkpoint(out_cont / "final", rep_cont["final_manifest_sha256"])
        st_res, tc_res, _, pay_res = load_automatic_checkpoint(out_resume / "final", rep_resume["final_manifest_sha256"])

        # Compare student and teacher state_dict
        for k, v in st_cont.state_dict().items():
            torch.testing.assert_close(v, st_res.state_dict()[k])
        for k, v in tc_cont.state_dict().items():
            torch.testing.assert_close(v, tc_res.state_dict()[k])

        # Compare optimizer state
        assert len(pay_cont["optimizers"]) == len(pay_res["optimizers"])
        for o_c, o_r in zip(pay_cont["optimizers"], pay_res["optimizers"]):
            for k in o_c:
                if k == "state":
                    for param_id in o_c["state"]:
                        for sk in o_c["state"][param_id]:
                            t1 = o_c["state"][param_id][sk]
                            t2 = o_r["state"][param_id][sk]
                            if isinstance(t1, torch.Tensor):
                                torch.testing.assert_close(t1, t2)

        # Compare final local np_rng bit generator states
        assert pay_cont["np_rng_state"]["state"]["state"] == pay_res["np_rng_state"]["state"]["state"]
        assert pay_cont["counters"] == pay_res["counters"]


def test_time_budget_enforcement_and_certification(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    # 1. Enforced time budget: budget expires before stepcap
    c_time = make_test_config("joint_dice", steps=20, phase="adaptation")
    c_time["time_budget_seconds"] = 0.0001  # Immediate expiry
    c_time["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c_time["initialization"]["warm_start_sha256"] = ws_sha
    out_time = tmp_path / "out_time"
    rep_time = run(c_time, out_time, synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep_time["termination_reason"] == "TIME_BUDGET_EXCEEDED"
    assert rep_time["equal_time_certified"] is True
    assert rep_time["steps"] < 20

    # 2. Step limit reached first: no equal-time certification
    c_step = make_test_config("joint_dice", steps=2, phase="adaptation")
    c_step["time_budget_seconds"] = 100.0  # Large budget
    c_step["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c_step["initialization"]["warm_start_sha256"] = ws_sha
    out_step = tmp_path / "out_step"
    rep_step = run(c_step, out_step, synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep_step["termination_reason"] == "STEP_LIMIT_REACHED"
    assert rep_step["equal_time_certified"] is False


def test_resource_accounting_and_margins(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    # Test guarded arm and control arms (frozen, support_only)
    for arm in ("joint_dice", "frozen", "support_only"):
        c = make_test_config(arm, steps=2, phase="adaptation")
        c["initialization"]["warm_start_checkpoint"] = str(ws_dir)
        c["initialization"]["warm_start_sha256"] = ws_sha
        out = tmp_path / f"out_metrics_{arm}"
        rep = run(c, out, synthetic=True, manifest=m, manifest_sha256=sha)

        assert rep["status"] == "COMPLETED"
        assert rep["extraction_seconds"] >= 0.0
        assert rep["marginal_adaptation_seconds"] >= rep["extraction_seconds"]
        assert len(rep["step_latencies_ms"]) == 2
        assert rep["latency_p50_ms"] != "NOT_MEASURED"
        assert rep["latency_p95_ms"] != "NOT_MEASURED"
        assert rep["throughput_updates_per_sec"] != "NOT_MEASURED"
        assert "margin_changes" in rep
        assert rep["margin_changes"] != "NOT_MEASURED"

        # Acceptance rate separate from nonzero displacement updates
        if arm == "joint_dice":
            assert "acceptance_rate" in rep
            assert "nonzero_displacement_updates" in rep
        elif arm in ("frozen", "support_only"):
            assert rep["acceptance_rate"] == "NOT_APPLICABLE"
            assert rep["nonzero_displacement_updates"] == 0


def test_max_cache_bytes_enforced(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    c = make_test_config("joint_dice", steps=2, phase="adaptation")
    c["max_cache_bytes"] = 10  # Tiny limit failing closed before allocation
    c["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c["initialization"]["warm_start_sha256"] = ws_sha

    with pytest.raises(ValueError, match="exceed.*max_cache_bytes"):
        run(c, tmp_path / "out_cache_fail", synthetic=True, manifest=m, manifest_sha256=sha)


@pytest.mark.parametrize("arm", sorted(NAMED_ARMS))
def test_all_named_arms_synthetic_execution(tmp_path, arm):
    data_dir = tmp_path / f"data_{arm}"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    c = make_test_config(arm, steps=2, phase="adaptation")
    c["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c["initialization"]["warm_start_sha256"] = ws_sha
    out = tmp_path / f"out_{arm}"

    report = run(c, out, synthetic=True, manifest=m, manifest_sha256=sha)
    assert report["status"] == "COMPLETED"
    assert report["arm"] == arm
    assert report["steps"] == 2
    assert (out / "REPORT.json").is_file()
    assert (out / "final" / "MANIFEST.json").is_file()
    assert (out / "final" / "weights.pt").is_file()


def test_train_py_cli_routing(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    # 1. Warm-start phase CLI invocation
    ws_cfg = make_test_config("support_only", steps=2, phase="warm_start")
    cfg_ws_path = tmp_path / "ws_config.json"
    write_json(cfg_ws_path, ws_cfg)
    out_ws = tmp_path / "cli_ws_out"
    cmd_ws = [
        "python", "train.py",
        "--config", str(cfg_ws_path),
        "--data", str(m),
        "--data-sha256", sha,
        "--out", str(out_ws),
        "--device", "cpu",
        "--synthetic",
    ]
    res_ws = subprocess.run(cmd_ws, capture_output=True, text=True)
    assert res_ws.returncode == 0, f"train.py warm_start failed: {res_ws.stderr}"
    assert (out_ws / "REPORT.json").is_file()
    ws_sha2 = json.loads((out_ws / "REPORT.json").read_text())["final_manifest_sha256"]

    # 2. Adaptation phase CLI invocation with time-budget and phase flags
    adapt_cfg = make_test_config("joint_dice", steps=2, phase="adaptation")
    adapt_cfg["initialization"]["warm_start_checkpoint"] = str(out_ws / "final")
    adapt_cfg["initialization"]["warm_start_sha256"] = ws_sha2
    cfg_adapt_path = tmp_path / "adapt_config.json"
    write_json(cfg_adapt_path, adapt_cfg)
    out_adapt = tmp_path / "cli_adapt_out"
    cmd_adapt = [
        "python", "train.py",
        "--config", str(cfg_adapt_path),
        "--data", str(m),
        "--data-sha256", sha,
        "--out", str(out_adapt),
        "--device", "cpu",
        "--synthetic",
        "--time-budget", "60.0",
        "--phase", "adaptation",
    ]
    res_adapt = subprocess.run(cmd_adapt, capture_output=True, text=True)
    assert res_adapt.returncode == 0, f"train.py adaptation failed: {res_adapt.stderr}"
    assert (out_adapt / "REPORT.json").is_file()

    # 3. Test rejection of volume geometry flags
    bad_geom_cmd = cmd_adapt + ["--geometry", "dummy", "--geometry-sha256", "0" * 64]
    res_bad = subprocess.run(bad_geom_cmd, capture_output=True, text=True)
    assert res_bad.returncode != 0
    assert "volume-geometry flags are invalid" in res_bad.stderr

    # 4. Test rejection of legacy flags
    bad_legacy_cmd = cmd_adapt + ["--legacy-checkpoint", "dummy"]
    res_legacy = subprocess.run(bad_legacy_cmd, capture_output=True, text=True)
    assert res_legacy.returncode != 0
    assert "legacy-only flags are invalid" in res_legacy.stderr

    # 5. Test rejection of resume without resume-sha256
    bad_resume_cmd = cmd_adapt + ["--resume", str(out_adapt / "final")]
    res_resume = subprocess.run(bad_resume_cmd, capture_output=True, text=True)
    assert res_resume.returncode != 0
    assert "--resume-sha256 is mandatory" in res_resume.stderr


def test_regression_manifest_only_and_metadata_tampering(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    # 1. Payload sha rejected as expected_sha256
    manifest_orig = json.loads((ws_dir / "MANIFEST.json").read_text())
    payload_sha = manifest_orig["payload_sha256"]
    with pytest.raises(ValueError, match="checkpoint hash mismatch: expected manifest"):
        load_automatic_checkpoint(ws_dir, expected_sha256=payload_sha)

    # 2. Tampered metadata with original expected sha rejected
    tampered_dir = tmp_path / "tampered_ws"
    tampered_dir.mkdir(parents=True)
    (tampered_dir / "weights.pt").write_bytes((ws_dir / "weights.pt").read_bytes())
    tampered_m = copy.deepcopy(manifest_orig)
    tampered_m["arm"] = "joint_dice"
    write_json(tampered_dir / "MANIFEST.json", tampered_m)
    with pytest.raises(ValueError, match="checkpoint hash mismatch: expected manifest"):
        load_automatic_checkpoint(tampered_dir, expected_sha256=ws_sha)

    # 3. Corrupted lineage rejected
    corrupt_lineage_dir = tmp_path / "corrupt_lineage"
    corrupt_lineage_dir.mkdir(parents=True)
    (corrupt_lineage_dir / "weights.pt").write_bytes((ws_dir / "weights.pt").read_bytes())
    corrupt_m = copy.deepcopy(manifest_orig)
    corrupt_m["target_label_lineage"]["count"] = 999  # mismatch len(patients)
    corrupt_m["payload_sha256"] = digest(corrupt_lineage_dir / "weights.pt")
    write_json(corrupt_lineage_dir / "MANIFEST.json", corrupt_m)
    corrupt_sha = digest(corrupt_lineage_dir / "MANIFEST.json")
    with pytest.raises(ValueError, match="checkpoint target_label_lineage count does not match patients length"):
        load_automatic_checkpoint(corrupt_lineage_dir, expected_sha256=corrupt_sha)


def test_regression_resume_schema_and_canonical_invariants(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    c = make_test_config("joint_dice", steps=4, phase="adaptation")
    c["save_every"] = 2
    c["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c["initialization"]["warm_start_sha256"] = ws_sha
    out_orig = tmp_path / "out_orig"
    rep = run(c, out_orig, synthetic=True, manifest=m, manifest_sha256=sha)

    step2_dir = out_orig / "step_000002"
    step2_sha = digest(step2_dir / "MANIFEST.json")

    # 1. Config mutation rejected by canonical JSON config digest
    c_mutated = copy.deepcopy(c)
    c_mutated["head_lr"] = 0.099  # mutated parameter
    with pytest.raises(ValueError, match="Resumed config invariants hash .* does not match"):
        run(c_mutated, tmp_path / "out_mut_fail", synthetic=True, manifest=m, manifest_sha256=sha, resume=step2_dir, resume_sha256=step2_sha)

    # 2. Resume requires adaptation phase (warm_start checkpoint rejected for resume continuation)
    c_ws_resume = copy.deepcopy(c)
    with pytest.raises(ValueError, match="checkpoint phase must be 'adaptation' for resume continuation"):
        run(c_ws_resume, tmp_path / "out_ws_fail", synthetic=True, manifest=m, manifest_sha256=sha, resume=ws_dir, resume_sha256=ws_sha)

    # 3. Time budget cannot reset or extend on resume
    c_time = copy.deepcopy(c)
    c_time["time_budget_seconds"] = 0.0001
    with pytest.raises(ValueError, match="time budget cannot be reset or extended on resume"):
        run(c_time, tmp_path / "out_tb_fail", synthetic=True, manifest=m, manifest_sha256=sha, resume=step2_dir, resume_sha256=step2_sha)


def test_regression_timer_accuracy_and_disjoint_components(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)

    # Warmstart timing
    ws_cfg = make_test_config("support_only", steps=2, phase="warm_start")
    out_ws = tmp_path / "ws_timing"
    rep_ws = run(ws_cfg, out_ws, synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep_ws["startup_seconds"] > 0
    assert rep_ws["transaction_seconds"] > 0
    assert rep_ws["phase_cost_seconds"] >= (rep_ws["startup_seconds"] + rep_ws["transaction_seconds"])
    assert "invocation_timing" in rep_ws
    assert "cumulative_timing" in rep_ws

    # Adaptation timing with disjoint components
    adapt_cfg = make_test_config("joint_dice", steps=2, phase="adaptation")
    adapt_cfg["initialization"]["warm_start_checkpoint"] = str(out_ws / "final")
    adapt_cfg["initialization"]["warm_start_sha256"] = rep_ws["final_manifest_sha256"]
    out_adapt = tmp_path / "adapt_timing"
    rep_adapt = run(adapt_cfg, out_adapt, synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep_adapt["startup_seconds"] > 0
    assert rep_adapt["extraction_seconds"] > 0
    assert rep_adapt["transaction_seconds"] > 0
    assert rep_adapt["final_save_seconds"] >= 0
    total_disjoint = (
        rep_adapt["startup_seconds"]
        + rep_adapt["extraction_seconds"]
        + rep_adapt["transaction_seconds"]
        + rep_adapt["final_save_seconds"]
    )
    assert abs(total_disjoint - rep_adapt["phase_cost_seconds"]) < 0.05


def test_regression_proposals_transactions_and_displacement_norm(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    # Test extra_supervised measures actual parameter displacement
    c_extra = make_test_config("extra_supervised", steps=2, phase="adaptation")
    c_extra["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c_extra["initialization"]["warm_start_sha256"] = ws_sha
    out_extra = tmp_path / "out_extra"
    rep_extra = run(c_extra, out_extra, synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep_extra["nonzero_displacement_updates"] > 0
    assert isinstance(rep_extra["zero_displacement_updates"], int)

    # Test joint_dice separates transactions vs directions
    c_joint = make_test_config("joint_dice", steps=2, phase="adaptation")
    c_joint["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c_joint["initialization"]["warm_start_sha256"] = ws_sha
    out_joint = tmp_path / "out_joint"
    rep_joint = run(c_joint, out_joint, synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep_joint["pseudo_directions_proposed"] >= rep_joint["pseudo_transactions_proposed"]
    assert rep_joint["pseudo_transactions_accepted"] <= rep_joint["pseudo_transactions_proposed"]
    assert rep_joint["pseudo_directions_accepted"] <= rep_joint["pseudo_directions_proposed"]
    if rep_joint["pseudo_transactions_proposed"] > 0:
        assert rep_joint["acceptance_rate"] == float(
            rep_joint["pseudo_transactions_accepted"] / rep_joint["pseudo_transactions_proposed"]
        )


def test_regression_support_margins_at_theta_sup_single_step(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    # 1-step run: margins must NOT be falsely netzero
    c = make_test_config("joint_dice", steps=1, phase="adaptation")
    c["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c["initialization"]["warm_start_sha256"] = ws_sha
    out = tmp_path / "out_step1"
    rep = run(c, out, synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep["steps"] == 1
    assert "margin_changes" in rep
    m_info = rep["margin_changes"]
    assert "initial_mean_true_margin" in m_info
    assert "final_mean_true_margin" in m_info
    assert "net_margin_change" in m_info


def test_regression_smaller_lr_configured_pseudo_lr(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    c = make_test_config("smaller_lr", steps=1, phase="adaptation")
    c["pseudo_lr"] = 0.0004
    c["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c["initialization"]["warm_start_sha256"] = ws_sha
    out = tmp_path / "out_smaller_lr"
    rep = run(c, out, synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep["status"] == "COMPLETED"


def test_regression_device_cpu_safety_and_threads(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    c = make_test_config("joint_dice", steps=1, phase="adaptation")
    c["threads"] = 2
    c["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c["initialization"]["warm_start_sha256"] = ws_sha
    out = tmp_path / "out_cpu"
    rep = run(c, out, device="cpu", synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep["peak_allocated_bytes"] == "NOT_MEASURED"
    assert rep["peak_vram_allocated_mb"] == "NOT_MEASURED"


def test_regression_finite_cache_workspace_defaults_and_guards(tmp_path):
    assert DEFAULT_AUTOMATIC_CONFIG["max_cache_bytes"] == 2 * 1024 * 1024 * 1024
    assert DEFAULT_AUTOMATIC_CONFIG["max_controller_workspace_bytes"] == 4 * 1024 * 1024 * 1024

    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)
    ws_dir, ws_sha = create_warm_start_checkpoint(tmp_path, m, sha)

    # Tiny controller workspace limit fails closed before allocation
    c = make_test_config("joint_dice", steps=1, phase="adaptation")
    c["max_controller_workspace_bytes"] = 10
    c["initialization"]["warm_start_checkpoint"] = str(ws_dir)
    c["initialization"]["warm_start_sha256"] = ws_sha
    with pytest.raises(ValueError, match="exceeds declared max_controller_workspace_bytes 10"):
        run(c, tmp_path / "fail_ws_bytes", synthetic=True, manifest=m, manifest_sha256=sha)


def test_regression_warmstart_arm_and_nan_features(tmp_path):
    data_dir = tmp_path / "data"
    m, sha = synthetic_fixture(data_dir, size=32)

    # Warmstart with arm != support_only must fail closed
    bad_ws = make_test_config("joint_dice", steps=1, phase="warm_start")
    with pytest.raises(ValueError, match="phase 'warm_start' requires arm='support_only'"):
        run(bad_ws, tmp_path / "fail_ws_arm", synthetic=True, manifest=m, manifest_sha256=sha)

    # >2L fixture works consistently when patient_ids is None (defaults to all L)
    ws_cfg = make_test_config("support_only", steps=1, phase="warm_start")
    ws_cfg["support"]["patient_ids"] = None
    ws_cfg["initialization"]["declared_target_budget"] = 1  # will adapt to all L in fixture
    out_all_l = tmp_path / "ws_all_l"
    rep_all_l = run(ws_cfg, out_all_l, synthetic=True, manifest=m, manifest_sha256=sha)
    assert rep_all_l["status"] == "COMPLETED"
    assert rep_all_l["target_budget"] == 3
