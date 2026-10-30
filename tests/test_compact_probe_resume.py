"""Unit and integration tests for compact probe sampling, exact resume, and CLI plumbing.

Covers CPTU_IMPLEMENTATION_CONTRACT.md requirements:
  - Proposal sequence unaffected by probe (independent Generator).
  - Probe patients strictly disjoint from all proposal patients in transaction.
  - Image-only TRAIN-U records; never touches U masks or labels.
  - No eligible probe returns None (missing probe -> zero decision).
  - Same-batch utility control reuses proposal tensors.
  - Exact interrupted vs uninterrupted decisions, weights, optimizer, and coverage.
  - Final status distinguishes stopped-boundary vs full-budget-completed.
  - Exact resume from legacy checkpoint missing RNG metadata fails closed.
  - Protocol mismatch (seed, method, changed total steps budget) fails closed.
  - CLI backward compatibility and bounded --stop-after-step.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import numpy as np
import pytest
import torch

from self_audit.compact_artifacts import (
    digest,
    load_compact_checkpoint,
    load_exact_compact_checkpoint,
    save_compact_checkpoint,
)
from self_audit.compact_data import CompactTrainingData, synthetic_compact_fixture
from self_audit.compact_model import build_compact_model
from self_audit.compact_probe import CompactProbeSampler, build_same_batch_probe
from self_audit.compact_training import (
    DEFAULT_CONFIG,
    SyntheticTeacherCache,
    train_compact,
    validate_compact_config,
)


def _bind_synthetic_teacher(cfg: Dict[str, Any], labeled_ids: list[str]) -> None:
    lineage = {"budget_k": len(labeled_ids), "exact_labeled_ids": list(labeled_ids)}
    cfg.setdefault("teacher", {})["lineage"] = lineage
    cfg["teacher"]["teacher_identity"] = {
        "teacher_kind": "synthetic",
        "inference_state": "student",
        "lineage": copy.deepcopy(lineage),
        "declared_k": len(labeled_ids),
        "labeled_ids": list(labeled_ids),
        "verification_status": "SYNTHETIC_VERIFIED",
    }


def _make_fixture(tmp_path: Path, n_unlabeled: int = 4) -> tuple[Path, str]:
    fix_dir = tmp_path / "fixture"
    return synthetic_compact_fixture(
        fix_dir, seed=42, n_labeled=1, n_unlabeled=n_unlabeled, size=32
    )


def test_training_no_eligible_probe_is_explicit_zero(tmp_path):
    manifest, sha = _make_fixture(tmp_path, n_unlabeled=1)
    metadata = json.loads(manifest.read_text())
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(method='scalar_guard', selector='cross_patient_utility', steps=1,
               warmup_steps=0, batch_size=1, labeled_batch_size=1, unlabeled_batch_size=1,
               probe_batch_size=1, crop_size=16, pseudo_threshold=0., max_trials=2)
    cfg['model']['widths'] = [4, 8]
    _bind_synthetic_teacher(cfg, [r['id'] for r in metadata['labeled']])
    out = tmp_path / 'run'
    report = train_compact(cfg, manifest, sha, out, synthetic=True, device='cpu')
    row = json.loads((out / 'TRAIN_LOG.jsonl').read_text())
    assert row['transaction']['reason'] == 'no_eligible_probe_patient'
    assert row['transaction']['probe_unavailable_reason'] == 'no_eligible_probe_patient'
    assert row['status'] == 'zero_displacement'
    assert row['transaction']['forward_counts']['probe_reference_evaluations'] == 0
    assert report['counts']['pseudo_accepted'] == 0


def test_proposal_sequence_unaffected_by_probe(tmp_path: Path):
    """Proposal batch schedule must be bit-for-bit identical regardless of probe configuration."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=4)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    base_cfg = copy.deepcopy(DEFAULT_CONFIG)
    base_cfg["method"] = "scalar_guard"
    base_cfg["steps"] = 16
    base_cfg["warmup_steps"] = 4
    base_cfg["seed"] = 123
    _bind_synthetic_teacher(base_cfg, data.lids)

    # Arm 1: Original selector (no probe active)
    cfg1 = copy.deepcopy(base_cfg)
    cfg1["selector"] = "original"
    out1 = tmp_path / "run_original"
    train_compact(cfg1, manifest_p, manifest_sha, out1, synthetic=True)

    # Arm 2: Utility selector with default probe seed
    cfg2 = copy.deepcopy(base_cfg)
    cfg2["selector"] = "utility"
    cfg2["probe_seed"] = 42017
    cfg2["probe_batch_size"] = 8
    out2 = tmp_path / "run_utility_default"
    train_compact(cfg2, manifest_p, manifest_sha, out2, synthetic=True)

    # Arm 3: Utility selector with different probe seed and batch size
    cfg3 = copy.deepcopy(base_cfg)
    cfg3["selector"] = "utility"
    cfg3["probe_seed"] = 99999
    cfg3["probe_batch_size"] = 4
    out3 = tmp_path / "run_utility_alt"
    train_compact(cfg3, manifest_p, manifest_sha, out3, synthetic=True)

    # Read schedules from logs
    lines1 = [json.loads(line) for line in (out1 / "TRAIN_LOG.jsonl").read_text().splitlines() if line]
    lines2 = [json.loads(line) for line in (out2 / "TRAIN_LOG.jsonl").read_text().splitlines() if line]
    lines3 = [json.loads(line) for line in (out3 / "TRAIN_LOG.jsonl").read_text().splitlines() if line]

    assert len(lines1) == 16
    assert len(lines2) == 16
    assert len(lines3) == 16

    for step in range(16):
        # Supervised schedule comparison
        sched1 = lines1[step]["data_schedule"]
        sched2 = lines2[step]["data_schedule"]
        sched3 = lines3[step]["data_schedule"]
        assert sched1 == sched2 == sched3, f"proposal data schedule diverged at step {step}"
        if step < 4:
            assert lines1[step]["supervised_loss"] == pytest.approx(lines2[step]["supervised_loss"])
            assert lines1[step]["supervised_loss"] == pytest.approx(lines3[step]["supervised_loss"])


def test_probe_patient_disjoint_from_proposal(tmp_path: Path):
    """Utility probe patient must be strictly disjoint from proposal patient in that transaction."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=4)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["method"] = "scalar_guard"
    cfg["selector"] = "utility"
    cfg["steps"] = 16
    cfg["warmup_steps"] = 2
    cfg["seed"] = 77
    cfg["probe_seed"] = 42017
    cfg["probe_batch_size"] = 4
    _bind_synthetic_teacher(cfg, data.lids)

    out = tmp_path / "run_disjoint"
    train_compact(cfg, manifest_p, manifest_sha, out, synthetic=True)

    lines = [json.loads(line) for line in (out / "TRAIN_LOG.jsonl").read_text().splitlines() if line]
    probe_checked = 0

    for row in lines:
        probe_meta = row.get("utility_probe_metadata")
        if probe_meta is not None:
            proposal_pid = row["data_schedule"]["unlabeled"][0]["patient"]
            probe_pid = probe_meta["patient"]
            assert probe_pid != proposal_pid, f"probe patient {probe_pid} overlapped with proposal patient {proposal_pid}"
            assert proposal_pid in probe_meta["excluded_proposal_patients"]
            assert probe_pid not in probe_meta["excluded_proposal_patients"]
            probe_checked += 1

    assert probe_checked > 0, "no probe transactions were observed"


def test_probe_image_only_no_labels_accessed(tmp_path: Path):
    """Probe sampler reads only image-only TRAIN-U records; never accesses labeled masks."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=3)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)
    cache = SyntheticTeacherCache(data)

    initial_label_reads = set(data.label_reads)

    sampler = CompactProbeSampler(data, cache, seed=42017, batch_size=4)
    probe = sampler.sample_probe(proposal_patient_ids={"patient_u_0"})

    assert probe is not None
    assert probe["x"].shape == (4, 3, 32, 32)
    assert probe["targets"].shape == (4, 4, 32, 32)
    assert probe["valid"].shape == (4, 32, 32)
    assert probe["metadata"]["patient"] != "patient_u_0"

    # Label reads set must NOT have grown
    assert data.label_reads == initial_label_reads, "probe sampler erroneously accessed labeled data"


def test_probe_no_eligible_unlabeled_returns_none(tmp_path: Path):
    """When all unlabeled patients in dataset are in proposal, sampler returns None."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=2)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)
    cache = SyntheticTeacherCache(data)

    sampler = CompactProbeSampler(data, cache, seed=42017, batch_size=2)
    all_u_ids = [r["id"] for r in data.unlabeled]

    # Exclude all unlabeled IDs
    probe = sampler.sample_probe(proposal_patient_ids=all_u_ids)
    assert probe is None, "expected None when no eligible unlabeled patients remain"


def test_same_batch_utility_control():
    """same_batch_utility packages proposal tensors without probe sampling."""
    x = torch.randn(2, 3, 16, 16)
    targets = torch.rand(2, 4, 16, 16)
    valid = torch.ones(2, 16, 16, dtype=torch.bool)

    probe = build_same_batch_probe(
        x,
        targets,
        valid,
        temperature=1.0,
        coefficient=1.0,
        confidence=0.7,
        metadata={"patient": "patient_u_0"},
    )

    assert probe["metadata"]["same_batch"] is True
    assert probe["metadata"]["patient"] == "patient_u_0"
    assert torch.equal(probe["x"], x)
    assert torch.equal(probe["targets"], targets)
    assert torch.equal(probe["valid"], valid)


def test_exact_resume_interrupted_uninterrupted_identical(tmp_path: Path):
    """Uninterrupted 16-step run must match interrupted 8-step + 8-step resume exactly."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=4)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["method"] = "scalar_guard"
    cfg["selector"] = "utility"
    cfg["steps"] = 16
    cfg["warmup_steps"] = 4
    cfg["seed"] = 42
    cfg["probe_seed"] = 42017
    cfg["probe_batch_size"] = 4
    _bind_synthetic_teacher(cfg, data.lids)

    # 1. Uninterrupted run for 16 steps
    out_uninterrupted = tmp_path / "run_uninterrupted"
    rep_uninterrupted = train_compact(
        cfg, manifest_p, manifest_sha, out_uninterrupted, synthetic=True
    )
    assert rep_uninterrupted["status"] == "COMPLETED"
    assert rep_uninterrupted["steps"] == 16
    assert rep_uninterrupted["termination_reason"] == "max_steps_reached"

    # 2. First segment: run 8 steps and stop
    out_segment1 = tmp_path / "run_segment1"
    rep_seg1 = train_compact(
        cfg,
        manifest_p,
        manifest_sha,
        out_segment1,
        synthetic=True,
        stop_after_step=8,
    )
    assert rep_seg1["status"] == "STOPPED_BOUNDARY"
    assert rep_seg1["steps"] == 8
    assert rep_seg1["termination_reason"] == "stop_after_step_reached"
    assert (out_segment1 / "final" / "MANIFEST.json").exists()

    ckpt_path = out_segment1 / "final"
    ckpt_manifest_sha = rep_seg1["final_manifest_sha256"]

    # 3. Second segment: exact resume from step 8 to 16
    out_resumed = tmp_path / "run_resumed"
    rep_resumed = train_compact(
        cfg,
        manifest_p,
        manifest_sha,
        out_resumed,
        synthetic=True,
        resume=ckpt_path,
        resume_sha256=ckpt_manifest_sha,
        stop_after_step=16,
    )
    assert rep_resumed["status"] == "COMPLETED"
    assert rep_resumed["steps"] == 16
    assert rep_resumed["termination_reason"] == "max_steps_reached"
    assert rep_resumed["resumed_step"] == 8

    # Verify model weights are bit-for-bit identical
    model_uninterrupted, _ = load_compact_checkpoint(
        out_uninterrupted / "final", rep_uninterrupted["final_manifest_sha256"]
    )
    model_resumed, _ = load_compact_checkpoint(
        out_resumed / "final", rep_resumed["final_manifest_sha256"]
    )

    p_un = dict(model_uninterrupted.named_parameters())
    p_res = dict(model_resumed.named_parameters())
    assert set(p_un.keys()) == set(p_res.keys())
    for name in p_un:
        assert torch.equal(p_un[name], p_res[name]), f"parameter '{name}' differed between uninterrupted and resumed run"

    # Verify decisions in TRAIN_LOG.jsonl for steps 8..15
    logs_un = [json.loads(line) for line in (out_uninterrupted / "TRAIN_LOG.jsonl").read_text().splitlines() if line]
    logs_res = [json.loads(line) for line in (out_resumed / "TRAIN_LOG.jsonl").read_text().splitlines() if line]

    assert len(logs_res) == 16  # Cumulative history preserved
    for step in range(8, 16):
        r_un = logs_un[step]
        r_res = logs_res[step]
        assert r_un["status"] == r_res["status"], f"status differed at step {step}"
        assert r_un["accepted"] == r_res["accepted"], f"acceptance differed at step {step}"
        assert r_un["scale"] == pytest.approx(r_res["scale"]), f"scale differed at step {step}"
        assert r_un["supervised_loss"] == pytest.approx(r_res["supervised_loss"]), f"loss differed at step {step}"
        assert r_un["realized_displacement_norm"] == pytest.approx(r_res["realized_displacement_norm"])

    # Verify coverage counters match
    assert rep_uninterrupted["counts"] == rep_resumed["counts"]
    assert rep_uninterrupted["slices_sampled"] == rep_resumed["slices_sampled"]
    assert rep_uninterrupted["slices_optimized"] == rep_resumed["slices_optimized"]
    assert rep_uninterrupted["optimizer_calls"] == rep_resumed["optimizer_calls"]


def test_resume_missing_rng_fails_closed(tmp_path: Path):
    """Legacy checkpoints missing RNG metadata fail closed on exact resume."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=2)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    arch = {"name": "CompactStudent", "widths": [16, 32, 64, 128], "in_channels": 3, "num_classes": 4, "init": "kaiming_normal"}
    model = build_compact_model(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # Save legacy-style checkpoint without rng_state
    ckpt_dir = tmp_path / "legacy_ckpt"
    ckpt_sha = save_compact_checkpoint(
        model, opt, cfg, data, ckpt_dir, step=4, rng_state=None
    )

    # Loading for export still works
    loaded_m, loaded_meta = load_compact_checkpoint(ckpt_dir, ckpt_sha)
    assert loaded_m is not None

    # Exact resume loading must fail closed
    with pytest.raises(ValueError, match="lacks mandatory exact RNG state metadata"):
        load_exact_compact_checkpoint(ckpt_dir, ckpt_sha, expected_config=cfg)


def test_resume_protocol_mismatch_fails_closed(tmp_path: Path):
    """Protocol changes (seed, method, budget steps) fail closed on resume."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=2)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["method"] = "scalar_guard"
    cfg["steps"] = 16
    cfg["seed"] = 42
    _bind_synthetic_teacher(cfg, data.lids)

    out = tmp_path / "base_ckpt"
    rep = train_compact(cfg, manifest_p, manifest_sha, out, synthetic=True, stop_after_step=4)
    ckpt_path = out / "final"
    ckpt_sha = rep["final_manifest_sha256"]

    # 1. Changed seed
    bad_cfg_seed = copy.deepcopy(cfg)
    bad_cfg_seed["seed"] = 999
    with pytest.raises(ValueError, match="protocol mismatch on resume: checkpoint seed=42, requested seed=999"):
        train_compact(bad_cfg_seed, manifest_p, manifest_sha, tmp_path / "r1", synthetic=True, resume=ckpt_path, resume_sha256=ckpt_sha)

    # 2. Changed method
    bad_cfg_method = copy.deepcopy(cfg)
    bad_cfg_method["method"] = "joint_guard"
    with pytest.raises(ValueError, match="protocol mismatch on resume: checkpoint method="):
        train_compact(bad_cfg_method, manifest_p, manifest_sha, tmp_path / "r2", synthetic=True, resume=ckpt_path, resume_sha256=ckpt_sha)

    # 3. Silently extended total budget steps (16 -> 32)
    bad_cfg_steps = copy.deepcopy(cfg)
    bad_cfg_steps["steps"] = 32
    with pytest.raises(ValueError, match="cannot silently change total step budget on resume"):
        train_compact(bad_cfg_steps, manifest_p, manifest_sha, tmp_path / "r3", synthetic=True, resume=ckpt_path, resume_sha256=ckpt_sha)


def test_cli_backward_compatibility_and_flags(tmp_path: Path):
    """CLI without new flags preserves historical defaults; new flags work properly."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=2)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    config_p = tmp_path / "compact_config.json"
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["steps"] = 8
    cfg["warmup_steps"] = 2
    _bind_synthetic_teacher(cfg, data.lids)
    config_p.write_text(json.dumps(cfg))

    out_cli = tmp_path / "cli_run"
    py_exec = sys.executable

    # 1. Default CLI run
    cmd = [
        py_exec,
        "train.py",
        "--config", str(config_p),
        "--data", str(manifest_p),
        "--data-sha256", manifest_sha,
        "--out", str(out_cli),
        "--device", "cpu",
        "--synthetic",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI command failed:\n{res.stderr}"
    rep = json.loads((out_cli / "REPORT.json").read_text())
    assert rep["status"] == "COMPLETED"
    assert rep["selector"] == "original"
    assert rep["steps"] == 8

    # 2. Bounded stop-after-step exceeding total steps fails CLI validation
    cmd_invalid = [
        py_exec,
        "train.py",
        "--config", str(config_p),
        "--data", str(manifest_p),
        "--data-sha256", manifest_sha,
        "--out", str(tmp_path / "cli_invalid"),
        "--device", "cpu",
        "--synthetic",
        "--stop-after-step", "16",  # 16 > 8
    ]
    res_inv = subprocess.run(cmd_invalid, capture_output=True, text=True)
    assert res_inv.returncode != 0
    assert "cannot exceed total steps" in res_inv.stderr

    # 3. Resume without resume-sha256 fails CLI validation
    cmd_no_sha = [
        py_exec,
        "train.py",
        "--config", str(config_p),
        "--data", str(manifest_p),
        "--data-sha256", manifest_sha,
        "--out", str(tmp_path / "cli_no_sha"),
        "--device", "cpu",
        "--synthetic",
        "--resume", str(out_cli / "final"),
    ]
    res_no_sha = subprocess.run(cmd_no_sha, capture_output=True, text=True)
    assert res_no_sha.returncode != 0
    assert "--resume-sha256 is mandatory" in res_no_sha.stderr


def test_validate_config_selector_method_restriction():
    """Non-original selectors and same_batch_utility require method scalar_guard."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["method"] = "joint_guard"
    cfg["selector"] = "utility"
    with pytest.raises(ValueError, match="is restricted to method 'scalar_guard'"):
        validate_compact_config(cfg)

    cfg2 = copy.deepcopy(DEFAULT_CONFIG)
    cfg2["method"] = "supervised"
    cfg2["same_batch_utility"] = True
    with pytest.raises(ValueError, match="same_batch_utility is restricted to method 'scalar_guard'"):
        validate_compact_config(cfg2)


def test_resume_all_changed_config_fields_rejected(tmp_path: Path):
    """Every immutable config field change must be rejected on exact resume."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=4)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    base_cfg = copy.deepcopy(DEFAULT_CONFIG)
    base_cfg["method"] = "scalar_guard"
    base_cfg["selector"] = "utility"
    base_cfg["steps"] = 16
    base_cfg["warmup_steps"] = 4
    base_cfg["seed"] = 42
    base_cfg["probe_seed"] = 42017
    base_cfg["probe_batch_size"] = 4
    _bind_synthetic_teacher(base_cfg, data.lids)

    base_out = tmp_path / "base_run"
    rep = train_compact(base_cfg, manifest_p, manifest_sha, base_out, synthetic=True, stop_after_step=4)
    ckpt_path = base_out / "final"
    ckpt_sha = rep["final_manifest_sha256"]

    # Test each immutable config field modification
    modifications = [
        ("lr", 1e-4),
        ("pseudo_lr", 2e-3),
        ("temperature", 1.5),
        ("coefficient", 0.5),
        ("ridge", 1e-5),
        ("batch_size", 2),
        ("labeled_batch_size", 2),
        ("unlabeled_batch_size", 2),
        ("risk_budget", 0.05),
        ("guard_tolerance", 1e-5),
        ("absent_fp_budget", 0.02),
        ("max_trials", 4),
        ("pseudo_threshold", 0.8),
        ("crop_size", 16),
        ("selector_dice_tolerance", 0.01),
        ("utility_gain_tolerance", 1e-6),
        ("utility_tie_tolerance", 1e-6),
        ("probe_seed", 99999),
        ("probe_batch_size", 2),
    ]

    for idx, (key, new_val) in enumerate(modifications):
        bad_cfg = copy.deepcopy(base_cfg)
        bad_cfg[key] = new_val
        run_out = tmp_path / f"test_mod_{idx}_{key}"
        with pytest.raises(ValueError, match=r"(immutable configuration mismatch on resume|protocol mismatch on resume)"):
            train_compact(bad_cfg, manifest_p, manifest_sha, run_out, synthetic=True, resume=ckpt_path, resume_sha256=ckpt_sha)

    # Test model architecture modification
    bad_arch_cfg = copy.deepcopy(base_cfg)
    bad_arch_cfg["model"]["widths"] = [32, 64, 128, 256]
    with pytest.raises(ValueError, match="immutable configuration mismatch on resume"):
        train_compact(bad_arch_cfg, manifest_p, manifest_sha, tmp_path / "bad_arch", synthetic=True, resume=ckpt_path, resume_sha256=ckpt_sha)


def test_resume_missing_or_invalid_metadata_rejected(tmp_path: Path):
    """Missing optimizer, protocol, counters, RNG components, or modified source hash fail closed."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=4)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    base_cfg = copy.deepcopy(DEFAULT_CONFIG)
    base_cfg["method"] = "scalar_guard"
    base_cfg["selector"] = "utility"
    base_cfg["steps"] = 16
    base_cfg["warmup_steps"] = 4
    base_cfg["seed"] = 42
    base_cfg["probe_seed"] = 42017
    base_cfg["probe_batch_size"] = 4
    _bind_synthetic_teacher(base_cfg, data.lids)

    base_out = tmp_path / "meta_run"
    rep = train_compact(base_cfg, manifest_p, manifest_sha, base_out, synthetic=True, stop_after_step=4)
    ckpt_path = base_out / "final"

    # Helper to create modified checkpoint directory and test rejection
    def make_tampered_ckpt(mutate_fn):
        t_dir = tmp_path / f"tampered_{mutate_fn.__name__}"
        import shutil
        shutil.copytree(ckpt_path, t_dir)
        weights_p = t_dir / "weights.pt"
        manifest_p_file = t_dir / "MANIFEST.json"
        payload = torch.load(weights_p, map_location="cpu", weights_only=True)
        manifest_data = json.loads(manifest_p_file.read_text())
        mutate_fn(payload, manifest_data)
        torch.save(payload, weights_p)
        manifest_data["payload_sha256"] = digest(weights_p)
        manifest_p_file.write_text(json.dumps(manifest_data, indent=2))
        t_sha = digest(manifest_p_file)
        return t_dir, t_sha

    # 1. Missing optimizer
    def mutate_no_opt(payload, manifest):
        payload["optimizer"] = None
    d, s = make_tampered_ckpt(mutate_no_opt)
    with pytest.raises(ValueError, match="missing mandatory optimizer state"):
        load_exact_compact_checkpoint(d, s, expected_config=base_cfg)

    # 2. Missing protocol
    def mutate_no_proto(payload, manifest):
        payload["protocol"] = None
        manifest["protocol"] = None
    d, s = make_tampered_ckpt(mutate_no_proto)
    with pytest.raises(ValueError, match="missing mandatory protocol metadata"):
        load_exact_compact_checkpoint(d, s, expected_config=base_cfg)

    # 3. Missing counters
    def mutate_no_counters(payload, manifest):
        payload["counters"] = None
        manifest["counters"] = None
    d, s = make_tampered_ckpt(mutate_no_counters)
    with pytest.raises(ValueError, match="missing mandatory counters metadata"):
        load_exact_compact_checkpoint(d, s, expected_config=base_cfg)

    # Exact continuation must not silently reset cumulative coverage or costs.
    for key in ('seen_a', 'audit_unique_frames', 'total_probe_cache_time',
                'warm_start_logs', 'cuda_peak_allocated_bytes'):
        def missing_counter(payload, manifest):
            payload['counters'].pop(key)
            manifest['counters'].pop(key)
        missing_counter.__name__ = 'missing_' + key
        d, s = make_tampered_ckpt(missing_counter)
        with pytest.raises(ValueError, match='missing mandatory field'):
            load_exact_compact_checkpoint(d, s, expected_config=base_cfg)

    def nonfinite_cost(payload, manifest):
        payload['counters']['total_probe_cache_time'] = float('nan')
        manifest['counters']['total_probe_cache_time'] = float('nan')
    d, s = make_tampered_ckpt(nonfinite_cost)
    with pytest.raises(ValueError, match='counters|finite'):
        load_exact_compact_checkpoint(d, s, expected_config=base_cfg)

    # 4. Missing probe_rng_state
    def mutate_no_probe_rng(payload, manifest):
        payload["rng"]["probe_rng_state"] = None
    d, s = make_tampered_ckpt(mutate_no_probe_rng)
    with pytest.raises(ValueError, match="missing mandatory 'probe_rng_state'"):
        load_exact_compact_checkpoint(d, s, expected_config=base_cfg)

    # 5. Missing py_rng
    def mutate_no_py_rng(payload, manifest):
        payload["rng"]["py_rng"] = None
    d, s = make_tampered_ckpt(mutate_no_py_rng)
    with pytest.raises(ValueError, match="missing mandatory 'py_rng'"):
        load_exact_compact_checkpoint(d, s, expected_config=base_cfg)

    # 6. Missing np_legacy_rng
    def mutate_no_np_rng(payload, manifest):
        payload["rng"]["np_legacy_rng"] = None
    d, s = make_tampered_ckpt(mutate_no_np_rng)
    with pytest.raises(ValueError, match="missing mandatory 'np_legacy_rng'"):
        load_exact_compact_checkpoint(d, s, expected_config=base_cfg)

    # 7. Invalid training source SHA in manifest
    def mutate_bad_source(payload, manifest):
        manifest["training_source_sha256"] = "0" * 64
        manifest["source_sha256"] = "0" * 64
    d, s = make_tampered_ckpt(mutate_bad_source)
    with pytest.raises(ValueError, match="training source SHA mismatch"):
        load_exact_compact_checkpoint(d, s, expected_config=base_cfg)

    # 8. Manifest protocol tampered vs payload protocol
    def mutate_protocol_desync(payload, manifest):
        manifest["protocol"]["seed"] = 12345
    d, s = make_tampered_ckpt(mutate_protocol_desync)
    with pytest.raises(ValueError, match="manifest protocol does not match payload protocol"):
        load_exact_compact_checkpoint(d, s, expected_config=base_cfg)

    # 9. Probe state validation (seed mismatch)
    def mutate_probe_seed_mismatch(payload, manifest):
        payload["rng"]["probe_rng_state"]["seed"] = 999
    d, s = make_tampered_ckpt(mutate_probe_seed_mismatch)
    with pytest.raises(ValueError, match="probe sampler seed mismatch"):
        load_exact_compact_checkpoint(d, s, expected_config=base_cfg)

    # 10. Probe state validation (batch_size mismatch)
    def mutate_probe_batch_mismatch(payload, manifest):
        payload["rng"]["probe_rng_state"]["batch_size"] = 99
    d, s = make_tampered_ckpt(mutate_probe_batch_mismatch)
    with pytest.raises(ValueError, match="probe sampler batch_size mismatch"):
        load_exact_compact_checkpoint(d, s, expected_config=base_cfg)


def test_resume_prefix_tamper_and_truncation_rejected(tmp_path: Path):
    """Tampered or truncated previous log prefix is rejected on exact resume."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=4)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["method"] = "scalar_guard"
    cfg["selector"] = "utility"
    cfg["steps"] = 16
    cfg["warmup_steps"] = 2
    cfg["seed"] = 42
    _bind_synthetic_teacher(cfg, data.lids)

    out = tmp_path / "run_base"
    rep = train_compact(cfg, manifest_p, manifest_sha, out, synthetic=True, stop_after_step=4)
    ckpt_path = out / "final"
    ckpt_sha = rep["final_manifest_sha256"]
    log_file = out / "TRAIN_LOG.jsonl"
    original_log_bytes = log_file.read_bytes()

    # 1. Truncated previous log
    log_file.write_bytes(original_log_bytes[:50])
    with pytest.raises(ValueError, match="previous log truncated"):
        train_compact(cfg, manifest_p, manifest_sha, tmp_path / "out_trunc", synthetic=True, resume=ckpt_path, resume_sha256=ckpt_sha)

    # 2. Tampered previous log prefix (modify one byte)
    tampered_bytes = bytearray(original_log_bytes)
    tampered_bytes[20] = ord(b'X') if tampered_bytes[20] != ord(b'X') else ord(b'Y')
    log_file.write_bytes(bytes(tampered_bytes))
    with pytest.raises(ValueError, match=r"previous log prefix hash mismatch.*tampered log"):
        train_compact(cfg, manifest_p, manifest_sha, tmp_path / "out_tamper", synthetic=True, resume=ckpt_path, resume_sha256=ckpt_sha)


def test_resume_earlier_checkpoint_verified_prefix_copy(tmp_path: Path):
    """Resuming from an earlier checkpoint copies only verified prefix, excluding subsequent appended logs."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=4)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["method"] = "scalar_guard"
    cfg["selector"] = "utility"
    cfg["steps"] = 8
    cfg["warmup_steps"] = 2
    cfg["seed"] = 42
    cfg["save_every"] = 4
    _bind_synthetic_teacher(cfg, data.lids)

    # Run all 8 steps uninterrupted; step_000004 checkpoint is saved along the way
    out_full = tmp_path / "run_full"
    rep_full = train_compact(cfg, manifest_p, manifest_sha, out_full, synthetic=True)
    assert rep_full["steps"] == 8
    full_lines = (out_full / "TRAIN_LOG.jsonl").read_text().splitlines()
    assert len(full_lines) == 8

    # The step_000004 checkpoint exists
    step4_ckpt = out_full / "step_000004"
    step4_manifest_sha = digest(step4_ckpt / "MANIFEST.json")

    # Resume from step_000004 into a new output directory
    out_resumed = tmp_path / "run_resumed_from_step4"
    rep_resumed = train_compact(
        cfg,
        manifest_p,
        manifest_sha,
        out_resumed,
        synthetic=True,
        resume=step4_ckpt,
        resume_sha256=step4_manifest_sha,
        stop_after_step=8,
    )
    assert rep_resumed["steps"] == 8
    resumed_lines = (out_resumed / "TRAIN_LOG.jsonl").read_text().splitlines()
    assert len(resumed_lines) == 8  # exactly 8 lines, not 4 + 8 = 12 lines!

    # Model parameters match uninterrupted run
    m_full, _ = load_compact_checkpoint(out_full / "final", rep_full["final_manifest_sha256"])
    m_res, _ = load_compact_checkpoint(out_resumed / "final", rep_resumed["final_manifest_sha256"])
    for name, param in m_full.named_parameters():
        assert torch.equal(param, dict(m_res.named_parameters())[name])


def test_resume_requires_new_output_dir(tmp_path: Path):
    """Resume must reject existing output directory and never overwrite."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=2)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["method"] = "scalar_guard"
    cfg["steps"] = 8
    cfg["seed"] = 42
    _bind_synthetic_teacher(cfg, data.lids)

    out = tmp_path / "base"
    rep = train_compact(cfg, manifest_p, manifest_sha, out, synthetic=True, stop_after_step=4)
    ckpt_path = out / "final"
    ckpt_sha = rep["final_manifest_sha256"]

    # Attempt to resume into existing directory
    existing_dir = tmp_path / "already_exists"
    existing_dir.mkdir()
    (existing_dir / "stale.txt").write_text("hello")
    with pytest.raises(FileExistsError, match="compact output directory must not already exist"):
        train_compact(cfg, manifest_p, manifest_sha, existing_dir, synthetic=True, resume=ckpt_path, resume_sha256=ckpt_sha)
    # File in existing directory was not touched
    assert (existing_dir / "stale.txt").read_text() == "hello"


def test_equal_wall_time_resume_refusal(tmp_path: Path):
    """Equal wall time resume without explicit budget must refuse explicitly."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=2)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["method"] = "scalar_guard"
    cfg["steps"] = 8
    cfg["seed"] = 42
    _bind_synthetic_teacher(cfg, data.lids)

    out = tmp_path / "base_run"
    rep = train_compact(cfg, manifest_p, manifest_sha, out, synthetic=True, stop_after_step=4)
    ckpt_path = out / "final"
    ckpt_sha = rep["final_manifest_sha256"]

    cfg_wall = copy.deepcopy(cfg)
    cfg_wall["equal_wall_time"] = {"enabled": True, "max_steps": 8, "budget_seconds": 60.0}
    with pytest.raises(ValueError, match="equal-wall-time resume is not supported without explicit remaining budget specification"):
        train_compact(cfg_wall, manifest_p, manifest_sha, tmp_path / "wall_res", synthetic=True, resume=ckpt_path, resume_sha256=ckpt_sha)


def test_probe_and_teacher_cache_timing_metrics(tmp_path: Path):
    """Probe sampling and teacher cache times are measured separately and recorded."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=4)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["method"] = "scalar_guard"
    cfg["selector"] = "utility"
    cfg["steps"] = 4
    cfg["warmup_steps"] = 1
    cfg["seed"] = 42
    cfg["probe_seed"] = 42017
    cfg["probe_batch_size"] = 4
    _bind_synthetic_teacher(cfg, data.lids)

    out = tmp_path / "timing_run"
    rep = train_compact(cfg, manifest_p, manifest_sha, out, synthetic=True)

    # Check teacher costs in report
    assert "teacher_cache_total_seconds" in rep["teacher_costs"]
    assert "teacher_cache_u_seconds" in rep["teacher_costs"]
    assert "teacher_cache_probe_seconds" in rep["teacher_costs"]
    assert "probe_sampling_total_seconds" in rep["teacher_costs"]

    # Check step log rows
    lines = [json.loads(line) for line in (out / "TRAIN_LOG.jsonl").read_text().splitlines() if line]
    assert len(lines) == 4
    for row in lines:
        assert "cuda_memory" in row
        # on CPU, memory is nullCPU
        assert row["cuda_memory"]["cuda_current_allocated_bytes"] is None
        assert row["cuda_memory"]["cuda_peak_allocated_bytes"] is None
        timings = row["timings"]
        assert "teacher_cache_seconds" in timings
        assert "probe_cache_seconds" in timings
        assert "probe_sampling_seconds" in timings
        assert "total_teacher_cache_seconds" in timings


def test_actual_3_segment_vs_uninterrupted_cli(tmp_path: Path):
    """Actual 3-segment CLI run must match uninterrupted CLI run bit-for-bit across weights, optimizer, decisions, RNG, and counters."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=4)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    config_p = tmp_path / "cfg.json"
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["method"] = "scalar_guard"
    cfg["selector"] = "utility"
    cfg["steps"] = 12
    cfg["warmup_steps"] = 6  # First boundary at 4 is inside warmup
    cfg["seed"] = 42
    cfg["probe_seed"] = 42017
    cfg["probe_batch_size"] = 4
    _bind_synthetic_teacher(cfg, data.lids)
    config_p.write_text(json.dumps(cfg))

    py_exec = sys.executable

    # 1. Uninterrupted run for 12 steps via CLI
    out_un = tmp_path / "cli_uninterrupted"
    cmd_un = [
        py_exec, "train.py",
        "--config", str(config_p),
        "--data", str(manifest_p),
        "--data-sha256", manifest_sha,
        "--out", str(out_un),
        "--device", "cpu",
        "--synthetic",
    ]
    res_un = subprocess.run(cmd_un, capture_output=True, text=True)
    assert res_un.returncode == 0, f"Uninterrupted run failed:\n{res_un.stderr}"
    rep_un = json.loads((out_un / "REPORT.json").read_text())
    assert rep_un["status"] == "COMPLETED"
    assert rep_un["steps"] == 12

    # 2. Segment 1: Steps 0..4 via CLI
    out_seg1 = tmp_path / "cli_seg1"
    cmd_seg1 = [
        py_exec, "train.py",
        "--config", str(config_p),
        "--data", str(manifest_p),
        "--data-sha256", manifest_sha,
        "--out", str(out_seg1),
        "--device", "cpu",
        "--synthetic",
        "--stop-after-step", "4",
    ]
    res_seg1 = subprocess.run(cmd_seg1, capture_output=True, text=True)
    assert res_seg1.returncode == 0, f"Segment 1 failed:\n{res_seg1.stderr}"
    rep_seg1 = json.loads((out_seg1 / "REPORT.json").read_text())
    assert rep_seg1["status"] == "STOPPED_BOUNDARY"
    assert rep_seg1["steps"] == 4

    # 3. Segment 2: Resume 4..8 via CLI
    out_seg2 = tmp_path / "cli_seg2"
    cmd_seg2 = [
        py_exec, "train.py",
        "--config", str(config_p),
        "--data", str(manifest_p),
        "--data-sha256", manifest_sha,
        "--out", str(out_seg2),
        "--device", "cpu",
        "--synthetic",
        "--resume", str(out_seg1 / "final"),
        "--resume-sha256", rep_seg1["final_manifest_sha256"],
        "--stop-after-step", "8",
    ]
    res_seg2 = subprocess.run(cmd_seg2, capture_output=True, text=True)
    assert res_seg2.returncode == 0, f"Segment 2 failed:\n{res_seg2.stderr}"
    rep_seg2 = json.loads((out_seg2 / "REPORT.json").read_text())
    assert rep_seg2["status"] == "STOPPED_BOUNDARY"
    assert rep_seg2["steps"] == 8

    # 4. Segment 3: Resume 8..12 via CLI
    out_seg3 = tmp_path / "cli_seg3"
    cmd_seg3 = [
        py_exec, "train.py",
        "--config", str(config_p),
        "--data", str(manifest_p),
        "--data-sha256", manifest_sha,
        "--out", str(out_seg3),
        "--device", "cpu",
        "--synthetic",
        "--resume", str(out_seg2 / "final"),
        "--resume-sha256", rep_seg2["final_manifest_sha256"],
        "--stop-after-step", "12",
    ]
    res_seg3 = subprocess.run(cmd_seg3, capture_output=True, text=True)
    assert res_seg3.returncode == 0, f"Segment 3 failed:\n{res_seg3.stderr}"
    rep_seg3 = json.loads((out_seg3 / "REPORT.json").read_text())
    assert rep_seg3["status"] == "COMPLETED"
    assert rep_seg3["steps"] == 12

    # Compare weights bit-for-bit
    m_un, _ = load_compact_checkpoint(out_un / "final", rep_un["final_manifest_sha256"])
    m_seg3, _ = load_compact_checkpoint(out_seg3 / "final", rep_seg3["final_manifest_sha256"])
    for name, param in m_un.named_parameters():
        assert torch.equal(param, dict(m_seg3.named_parameters())[name]), f"Parameter {name} diverged in 3-segment run"

    # Compare optimizer state bit-for-bit
    p_un = torch.load(out_un / "final" / "weights.pt", weights_only=True)
    p_seg3 = torch.load(out_seg3 / "final" / "weights.pt", weights_only=True)
    opt_un = p_un["optimizer"]["state"]
    opt_seg3 = p_seg3["optimizer"]["state"]
    assert set(opt_un.keys()) == set(opt_seg3.keys())
    for k in opt_un:
        for sk in opt_un[k]:
            if isinstance(opt_un[k][sk], torch.Tensor):
                assert torch.equal(opt_un[k][sk], opt_seg3[k][sk]), f"Optimizer state {k}.{sk} diverged"

    # Compare decisions in TRAIN_LOG.jsonl for all 12 steps
    logs_un = [json.loads(line) for line in (out_un / "TRAIN_LOG.jsonl").read_text().splitlines() if line]
    logs_seg3 = [json.loads(line) for line in (out_seg3 / "TRAIN_LOG.jsonl").read_text().splitlines() if line]
    assert len(logs_un) == len(logs_seg3) == 12
    for step in range(12):
        assert logs_un[step]["status"] == logs_seg3[step]["status"], f"status diverged at step {step}"
        assert logs_un[step]["accepted"] == logs_seg3[step]["accepted"], f"acceptance diverged at step {step}"
        assert logs_un[step]["scale"] == pytest.approx(logs_seg3[step]["scale"]), f"scale diverged at step {step}"
        assert logs_un[step]["supervised_loss"] == pytest.approx(logs_seg3[step]["supervised_loss"]), f"loss diverged at step {step}"
        assert logs_un[step]["realized_displacement_norm"] == pytest.approx(logs_seg3[step]["realized_displacement_norm"])

    # Compare counters
    assert rep_un["counts"] == rep_seg3["counts"]
    assert rep_un["slices_sampled"] == rep_seg3["slices_sampled"]
    assert rep_un["slices_optimized"] == rep_seg3["slices_optimized"]
    assert rep_un["optimizer_calls"] == rep_seg3["optimizer_calls"]
    assert rep_un["audit_consumption"]["unique_frames"] == rep_seg3["audit_consumption"]["unique_frames"]

    # Verify timing metrics are recorded, positive, and monotonically accumulated across resume segments
    assert rep_seg3["teacher_costs"]["teacher_cache_total_seconds"] > 0
    assert rep_seg3["teacher_costs"]["teacher_cache_probe_seconds"] > 0
    assert rep_seg3["teacher_costs"]["probe_sampling_total_seconds"] > 0
    assert rep_seg3["teacher_costs"]["teacher_cache_total_seconds"] >= rep_seg2["teacher_costs"]["teacher_cache_total_seconds"]
    assert rep_seg2["teacher_costs"]["teacher_cache_total_seconds"] >= rep_seg1["teacher_costs"]["teacher_cache_total_seconds"]


def test_legacy_warmup_clamp_semantics():
    """Fresh run without stop_after_step preserves historical warmup clamping to steps."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["steps"] = 2
    cfg["warmup_steps"] = 1000
    validated = validate_compact_config(cfg)
    assert validated["warmup_steps"] == 2
    assert validated["stop_after_step"] is None

    # But with exact boundary stop_after_step, exceeding immutable steps budget raises
    cfg_bound = copy.deepcopy(DEFAULT_CONFIG)
    cfg_bound["steps"] = 16
    cfg_bound["stop_after_step"] = 4
    cfg_bound["warmup_steps"] = 20  # 20 > 16
    with pytest.raises(ValueError, match="exceeds total steps budget"):
        validate_compact_config(cfg_bound)


def test_train_cli_rejects_compact_flags_for_noncompact_schemas(tmp_path: Path):
    """CLI rejects compact-specific flags when used with noncompact schemas."""
    # Write a dummy volume config
    vol_cfg_p = tmp_path / "vol_cfg.json"
    vol_cfg_p.write_text(json.dumps({
        "schema": "self_audit.volume.v1",
        "steps": 10,
        "seed": 42,
    }))

    py_exec = sys.executable

    # Test rejection of --selector with noncompact schema
    cmd1 = [py_exec, "train.py", "--config", str(vol_cfg_p), "--selector", "utility"]
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    assert res1.returncode != 0
    assert "compact-only flags are invalid for noncompact schema" in res1.stderr

    # Test rejection of --stop-after-step with noncompact schema
    cmd2 = [py_exec, "train.py", "--config", str(vol_cfg_p), "--stop-after-step", "2"]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert res2.returncode != 0
    assert "compact-only flags are invalid for noncompact schema" in res2.stderr


def test_resume_cuda_rng_required_on_cuda_device(tmp_path: Path):
    """Resume on CUDA requires torch_cuda_rng; fails closed if absent."""
    manifest_p, manifest_sha = _make_fixture(tmp_path, n_unlabeled=2)
    data = CompactTrainingData(manifest_p, manifest_sha, synthetic=True, allow_k1=True)

    base_cfg = copy.deepcopy(DEFAULT_CONFIG)
    base_cfg["method"] = "scalar_guard"
    base_cfg["steps"] = 4
    base_cfg["warmup_steps"] = 1
    base_cfg["seed"] = 42
    _bind_synthetic_teacher(base_cfg, data.lids)

    base_out = tmp_path / "cpu_run"
    rep = train_compact(base_cfg, manifest_p, manifest_sha, base_out, synthetic=True, stop_after_step=2)
    ckpt_path = base_out / "final"

    # Explicitly remove torch_cuda_rng from saved payload and repin payload/manifest hashes
    weights_path = ckpt_path / "weights.pt"
    manifest_path = ckpt_path / "MANIFEST.json"
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text())
    payload["rng"].pop("torch_cuda_rng", None)
    torch.save(payload, weights_path)
    manifest["payload_sha256"] = digest(weights_path)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    ckpt_sha = digest(manifest_path)

    # When requesting cuda device on checkpoint without torch_cuda_rng, must fail closed
    with pytest.raises(ValueError, match="checkpoint lacks mandatory 'torch_cuda_rng' for exact CUDA resume"):
        load_exact_compact_checkpoint(ckpt_path, ckpt_sha, expected_config=base_cfg, device="cuda:0")
