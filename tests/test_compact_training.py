"""Comprehensive unit and integration tests for compact student training pipeline.

Covers:
  - Strict opt-in config validation and budget steps guard.
  - Locked Macro4 >= 0.87 acceptance metadata.
  - Full learned supervised AdamW with warm-start module gradient and weight change tracking.
  - Post-supervised reference state and whole-network additive SGD U update.
  - No compact EMA (raw student deployment only).
  - Weak student still receives teacher KD.
  - Explicit K=1 label budget reuse.
  - Matched sampling RNG schedule across all compact arms.
  - Separate L/U counts, timings, and teacher cost reporting (NOT_KNOWN, never 0).
  - Public CLI synthetic smoke with label, cache, and flag guards.
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
import torch
from torch import nn

from self_audit.compact_artifacts import digest
from self_audit.compact_control import build_class_candidates, apply_compact_update
from self_audit.compact_data import CompactTrainingData, synthetic_compact_fixture
from self_audit.compact_model import CompactStudent, build_compact_model
from self_audit.compact_training import (
    DEFAULT_CONFIG,
    METHODS,
    SyntheticTeacherCache,
    train_compact,
    validate_compact_config,
)


def test_validate_compact_config_all_methods():
    for method in METHODS:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["method"] = method
        validated = validate_compact_config(cfg)
        assert validated["method"] == method
        assert validated["schema"] == "self_audit.compact.v1"
        assert validated["acceptance"]["min_macro4"] >= 0.87


def _bind_synthetic_teacher(cfg: Dict[str, Any], labeled_ids: list[str]) -> None:
    lineage = {"budget_k": len(labeled_ids), "exact_labeled_ids": list(labeled_ids)}
    cfg.setdefault("teacher", {})["lineage"] = lineage
    cfg["teacher"]["teacher_identity"] = {
        "teacher_kind": "synthetic",
        "inference_state": "student",
        "lineage": copy.deepcopy(lineage),
    }


def _synthetic_ids(count: int) -> list[str]:
    return [f"synthetic_p{i:03d}" for i in range(count)]


def test_validate_compact_config_rejections():
    # 1. Invalid schema
    bad_schema = copy.deepcopy(DEFAULT_CONFIG)
    bad_schema["schema"] = "self_audit.semi_v2"
    with pytest.raises(ValueError, match="invalid schema"):
        validate_compact_config(bad_schema)

    # 2. Unknown method
    bad_method = copy.deepcopy(DEFAULT_CONFIG)
    bad_method["method"] = "ema_full"
    with pytest.raises(ValueError, match="invalid compact method"):
        validate_compact_config(bad_method)

    # 3. Budget steps guard (> 128 updates requires allow_full_run)
    long_job = copy.deepcopy(DEFAULT_CONFIG)
    long_job["steps"] = 256
    with pytest.raises(ValueError, match="more than 128 updates requires --allow-full-run"):
        validate_compact_config(long_job, allow_full_run=False)
    # Allowed when allow_full_run=True
    assert validate_compact_config(long_job, allow_full_run=True)["steps"] == 256

    # 4. Locked Macro4 acceptance metadata (< 0.87 rejected)
    low_macro = copy.deepcopy(DEFAULT_CONFIG)
    low_macro["acceptance"] = {"min_macro4": 0.82}
    with pytest.raises(ValueError, match="acceptance min_macro4 must be finite and exactly 0.87"):
        validate_compact_config(low_macro)
    nan_macro = copy.deepcopy(DEFAULT_CONFIG)
    nan_macro["acceptance"]["min_macro4"] = float("nan")
    with pytest.raises(ValueError, match="finite and exactly 0.87"):
        validate_compact_config(nan_macro)

    # 4.1 Enabled EMA/prototype/window are rejected at any config depth.
    for path, value in (("ema", True), ("prototype", {"enabled": True}),
                        ("model", {"window": {"enabled": True}})):
        cfg_with_feature = copy.deepcopy(DEFAULT_CONFIG)
        if path == "model":
            cfg_with_feature["model"].update(value)
        else:
            cfg_with_feature[path] = value
        with pytest.raises(ValueError, match="forbidden feature"):
            validate_compact_config(cfg_with_feature)

    # Trial limits above eight are retained as configured.
    cfg_more_trials = copy.deepcopy(DEFAULT_CONFIG)
    cfg_more_trials["max_trials"] = 12
    assert validate_compact_config(cfg_more_trials)["max_trials"] == 12

    timed = copy.deepcopy(DEFAULT_CONFIG)
    timed["equal_wall_time"] = {"enabled": True, "max_steps": 4, "budget_seconds": 10.0}
    assert validate_compact_config(timed)["equal_wall_time"]["max_steps"] == 4
    timed["equal_wall_time"]["max_steps"] = 33
    with pytest.raises(ValueError, match="cannot exceed the immutable steps limit"):
        validate_compact_config(timed)

    # 5. Prohibited student EMA
    ema_cfg = copy.deepcopy(DEFAULT_CONFIG)
    ema_cfg["student_ema"] = True
    with pytest.raises(ValueError, match="compact student EMA is prohibited"):
        validate_compact_config(ema_cfg)

    # 6. Placeholder without matching lineage
    placeholder_cfg = copy.deepcopy(DEFAULT_CONFIG)
    placeholder_cfg["teacher"]["lineage"] = {
        "budget_k": 2,
        "exact_labeled_ids": [],
        "matching_lineage_required": True,
    }
    with pytest.raises(ValueError, match="requires matching teacher label lineage"):
        validate_compact_config(placeholder_cfg)

    joint_placeholder = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/compact/joint_4_plus_4_placeholder.json").read_text()
    )
    assert joint_placeholder["baseline_gate"]["status"] == "REQUIRED_BEFORE_LAUNCH"
    assert joint_placeholder["teacher"]["lineage"]["matching_lineage_required"] is True
    assert joint_placeholder["experiment"]["launchable"] is False
    with pytest.raises(ValueError, match="placeholder and is not launchable"):
        validate_compact_config(joint_placeholder)

    full_profile = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/compact/acdc_l8_full.json").read_text()
    )
    assert full_profile["steps"] == 10000
    assert full_profile["warmup_steps"] == 1000
    assert (full_profile["labeled_batch_size"], full_profile["unlabeled_batch_size"], full_profile["crop_size"]) == (8, 8, 160)
    with pytest.raises(ValueError, match="more than 128 updates requires --allow-full-run"):
        validate_compact_config(full_profile)
    assert validate_compact_config(full_profile, allow_full_run=True)["steps"] == 10000
    historic_profile = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/compact/acdc_l8_historic.json").read_text()
    )
    assert historic_profile["steps"] == 32
    assert historic_profile["experiment"]["status"] == "PROFILE_ONLY"


def test_training_rejects_budget_mismatch_missing_identity_and_cache(tmp_path: Path):
    synthetic_manifest, synthetic_sha = synthetic_compact_fixture(
        tmp_path / "mismatch_data", n_labeled=2, n_unlabeled=1, size=8,
        num_frames=1, num_slices=1,
    )
    synthetic_data = CompactTrainingData(synthetic_manifest, synthetic_sha, synthetic=True, allow_k1=True)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["method"] = "plain_kd"
    cfg["steps"] = 1
    cfg["warmup_steps"] = 0
    _bind_synthetic_teacher(cfg, synthetic_data.lids)
    cfg["data"] = {
        "manifest_sha256": synthetic_sha,
        "budget_k": 1,
        "labeled_ids": list(synthetic_data.lids),
    }
    with pytest.raises(ValueError, match="data.budget_k"):
        train_compact(cfg, synthetic_manifest, synthetic_sha, tmp_path / "bad_budget", synthetic=True)

    cfg["data"]["budget_k"] = 2
    cfg["data"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="data.manifest_sha256 mismatch"):
        train_compact(cfg, synthetic_manifest, synthetic_sha, tmp_path / "bad_manifest_sha", synthetic=True)

    cfg["data"]["manifest_sha256"] = synthetic_sha
    cfg["teacher"]["teacher_identity"] = None
    with pytest.raises(ValueError, match="teacher_identity is required"):
        train_compact(cfg, synthetic_manifest, synthetic_sha, tmp_path / "missing_identity", synthetic=True)

    _bind_synthetic_teacher(cfg, synthetic_data.lids)
    cfg["auxiliary_labeled_ids"] = ["extra_patient"]
    with pytest.raises(ValueError, match="union of all labeled patient IDs"):
        train_compact(cfg, synthetic_manifest, synthetic_sha, tmp_path / "bad_auxiliary_union", synthetic=True)

    # A manufactured manifest marked non-synthetic exercises the real-data
    # fail-closed cache requirement without training or reading any labels.
    manifest_obj = json.loads(synthetic_manifest.read_text())
    manifest_obj["synthetic"] = False
    real_manifest = synthetic_manifest
    real_manifest.write_text(json.dumps(manifest_obj, indent=2) + "\n")
    real_sha = digest(real_manifest)
    real_data = CompactTrainingData(real_manifest, real_sha, synthetic=False, allow_k1=True)
    canonical_lineage = {"budget_k": len(real_data.lids), "exact_labeled_ids": list(real_data.lids)}
    canonical_identity = {
        "teacher_kind": "canonical",
        "inference_state": "teacher_ema",
        "lineage": canonical_lineage,
        "manifest_sha256": "1" * 64,
        "source_sha256": "2" * 64,
        "payload_sha256": "3" * 64,
        "training_method": "ema_full",
        "quality_receipt": {"Macro4": 0.88},
    }
    real_cfg = copy.deepcopy(DEFAULT_CONFIG)
    real_cfg.update(method="plain_kd", steps=1, warmup_steps=0)
    real_cfg["data"] = {
        "manifest_sha256": real_sha,
        "budget_k": len(real_data.lids),
        "labeled_ids": list(real_data.lids),
    }
    real_cfg["teacher"]["lineage"] = canonical_lineage
    real_cfg["teacher"]["teacher_identity"] = canonical_identity
    with pytest.raises(ValueError, match="explicit, verified teacher cache"):
        train_compact(real_cfg, real_manifest, real_sha, tmp_path / "missing_cache", synthetic=False)


def test_teacher_union_preflight_precedes_image_and_mask_loading(monkeypatch, tmp_path: Path):
    import self_audit.compact_training as compact_training

    manifest, manifest_sha = synthetic_compact_fixture(
        tmp_path / "preflight_order", n_labeled=2, n_unlabeled=1, size=8,
        num_frames=1, num_slices=1,
    )
    original_data_type = compact_training.CompactTrainingData
    metadata_data = original_data_type(manifest, manifest_sha, synthetic=True, allow_k1=True)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(method="plain_kd", steps=1, warmup_steps=1)
    _bind_synthetic_teacher(cfg, metadata_data.lids)
    cfg["data"] = {
        "manifest_sha256": manifest_sha,
        "budget_k": len(metadata_data.lids),
        "labeled_ids": list(metadata_data.lids),
    }
    preflight_called = {"value": False}
    original_preflight = compact_training.preflight_teacher_lineage

    def checked_preflight(*args, **kwargs):
        result = original_preflight(*args, **kwargs)
        preflight_called["value"] = True
        return result

    def checked_data_loader(*args, **kwargs):
        assert preflight_called["value"], "training images/masks loaded before teacher label-union preflight"
        return original_data_type(*args, **kwargs)

    monkeypatch.setattr(compact_training, "preflight_teacher_lineage", checked_preflight)
    monkeypatch.setattr(compact_training, "CompactTrainingData", checked_data_loader)
    report = train_compact(
        cfg, manifest, manifest_sha, tmp_path / "preflight_order_run", synthetic=True
    )
    assert report["teacher_preflight"]["status"] == "PASS"


def test_output_path_must_be_new_and_prior_run_is_preserved(tmp_path: Path):
    out = tmp_path / "existing_run"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("previous run")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["method"] = "supervised"
    with pytest.raises(FileExistsError, match="must not already exist"):
        train_compact(cfg, None, None, out, synthetic=True)
    assert marker.read_text() == "previous run"
    assert not (out / "config.json").exists()


def test_group_norm_accumulator_uses_parameter_device():
    from self_audit.compact_training import _group_l2

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        values = [torch.tensor([3.0], device=device), torch.tensor([4.0], device=device)]
        assert _group_l2(values) == pytest.approx(5.0)


def test_equal_wall_time_uses_predeclared_step_cap(tmp_path: Path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(method="supervised", steps=2, warmup_steps=0)
    cfg["equal_wall_time"] = {"enabled": True, "max_steps": 1, "budget_seconds": 30.0}
    report = train_compact(cfg, None, None, tmp_path / "bounded_equal_time", synthetic=True)
    assert report["steps_requested"] == 2
    assert report["max_steps"] == 1
    assert report["steps"] == 1
    assert report["equal_wall_time"] == cfg["equal_wall_time"]


def test_compact_teacher_import_failure_does_not_fall_back():
    root = Path(__file__).resolve().parents[1]
    script = r"""
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.rsplit('.', 1)[-1] == 'compact_teacher':
        raise ImportError('injected changed-source failure')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
try:
    import self_audit.compact_training
except ImportError as exc:
    assert 'injected changed-source failure' in str(exc)
else:
    raise SystemExit('compact_training silently fell back without compact_teacher')
"""
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(root), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_warm_start_gradient_and_weight_changes(tmp_path: Path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["steps"] = 1
    cfg["warmup_steps"] = 0
    cfg["method"] = "supervised"

    report = train_compact(
        cfg,
        None,
        None,
        tmp_path / "out_warm_start",
        device="cpu",
        synthetic=True,
    )
    assert report["status"] == "COMPLETED"

    # Verify warm-start tracking in summary
    ws = report["warm_start_summary"]
    for group in ("encoder", "decoder", "head"):
        assert group in ws["initial_gradient_norms"]
        assert ws["initial_gradient_norms"][group] > 0
        assert np.isfinite(ws["initial_gradient_norms"][group])

        assert group in ws["initial_weight_changes"]
        assert ws["initial_weight_changes"][group] > 0
        assert np.isfinite(ws["initial_weight_changes"][group])

    # Verify log entry in TRAIN_LOG.jsonl
    log_file = tmp_path / "out_warm_start" / "TRAIN_LOG.jsonl"
    assert log_file.exists()
    row = json.loads(log_file.read_text().splitlines()[0])
    assert "warm_start" in row
    assert "gradient_norms" in row["warm_start"]
    assert "weight_changes" in row["warm_start"]


def test_weak_student_still_gets_teacher_kd():
    torch.manual_seed(101)
    # Freshly initialized weak student
    model = CompactStudent(widths=[8, 16, 32, 64])
    x = torch.rand(2, 3, 16, 16)
    # Teacher produces confident targets
    targets = torch.zeros(2, 4, 16, 16)
    targets[:, 1, :, :] = 0.95
    targets[:, 0, :, :] = 0.05
    valid = torch.ones(2, 16, 16, dtype=torch.bool)

    candidates = build_class_candidates(
        model, x, targets, valid, lr=0.01, confidence=0.5
    )
    assert len(candidates["directions"]) > 0
    assert candidates["denominator"] == 2 * 16 * 16

    # Apply candidates via plain_kd and verify student parameters are updated
    before_p = {n: p.detach().clone() for n, p in model.named_parameters()}
    receipt = apply_compact_update(model, candidates, method="plain_kd")
    assert receipt["accepted"] is True
    assert receipt["status"] == "accepted_nonzero"

    # Parameters must have changed
    changed = any(
        not torch.equal(before_p[n], p.detach()) for n, p in model.named_parameters()
    )
    assert changed, "weak student parameters must be updated by teacher KD"


def test_post_supervised_reference_and_no_compact_ema(tmp_path: Path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["steps"] = 2
    cfg["warmup_steps"] = 0
    cfg["method"] = "joint_guard"
    cfg["pseudo_threshold"] = 0.0
    _bind_synthetic_teacher(cfg, _synthetic_ids(1))

    out_dir = tmp_path / "out_no_ema"
    report = train_compact(
        cfg,
        None,
        None,
        out_dir,
        device="cpu",
        synthetic=True,
    )
    assert report["status"] == "COMPLETED"
    assert report["initialization"]["initial_state_sha256"]
    assert report["initialization"]["first_step_optimizer"]["name"] == "AdamW"
    assert set(report["initialization"]["first_step_optimizer"]["parameter_ownership"]) == {
        name for name, _ in CompactStudent(widths=[16, 32, 64, 128]).named_parameters()
    }
    rows = [json.loads(line) for line in (out_dir / "TRAIN_LOG.jsonl").read_text().splitlines()]
    assert all(row["margin_decreased_fraction"] is None for row in rows)
    assert report["audit_consumption"]["slice_reads"] > 0
    assert report["audit_consumption"]["frame_reads"] > 0
    assert report["audit_consumption"]["patient_reads"] > 0

    # Verify final checkpoint has no student EMA
    manifest_path = out_dir / "final" / "MANIFEST.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest.get("teacher_included") is False
    assert "ema" not in manifest.get("schema", "")

    # Payload weights must have student only, no EMA weights
    weights = torch.load(out_dir / "final" / "weights.pt", map_location="cpu", weights_only=True)
    assert "student" in weights
    assert "ema" not in weights
    assert "teacher" not in weights


def test_k1_reuse_explicit(tmp_path: Path):
    # Create fixture with exactly K=1 labeled patient
    synth_dir = tmp_path / "k1_data"
    manifest_path, manifest_sha = synthetic_compact_fixture(
        synth_dir, n_labeled=1, n_unlabeled=2, size=16
    )

    data = CompactTrainingData(manifest_path, manifest_sha, synthetic=True, allow_k1=True)
    assert len(data.lids) == 1

    rng = np.random.default_rng(42)
    # Verify sample_indices reuses audit patient without crash
    for step in range(5):
        li, ai, ui = data.sample_indices(step, rng)
        assert li == 0
        assert ai == 0  # K1 reuse!
        assert ui in (0, 1)

    # Test full training run with K=1
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["steps"] = 2
    cfg["warmup_steps"] = 0
    cfg["pseudo_threshold"] = 0.0
    _bind_synthetic_teacher(cfg, data.lids)
    report = train_compact(
        cfg,
        manifest_path,
        manifest_sha,
        tmp_path / "out_k1",
        device="cpu",
        synthetic=True,
    )
    assert report["label_budget"] == 1
    assert len(report["seen_supervised"]) == 1
    assert report["seen_supervised"] == report["seen_audit"]


def test_matched_sampling_rng_schedule_across_arms(tmp_path: Path):
    synth_dir = tmp_path / "matched_data"
    manifest_path, manifest_sha = synthetic_compact_fixture(
        synth_dir, n_labeled=2, n_unlabeled=3, size=16
    )

    log_rows: Dict[str, list] = {}
    for arm in ("supervised", "plain_kd", "joint_guard"):
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["method"] = arm
        cfg["steps"] = 3
        cfg["warmup_steps"] = 1
        cfg["seed"] = 999
        if arm != "supervised":
            _bind_synthetic_teacher(cfg, _synthetic_ids(2))
        out_arm = tmp_path / f"out_rng_{arm}"
        train_compact(
            cfg,
            manifest_path,
            manifest_sha,
            out_arm,
            device="cpu",
            synthetic=True,
        )
        lines = (out_arm / "TRAIN_LOG.jsonl").read_text().splitlines()
        log_rows[arm] = [json.loads(l) for l in lines]

    # Compare step-by-step sampled slice counts and losses across arms
    for step in range(3):
        # Supervised loss in warmup / step 0 must match exactly across arms
        sup_loss_0 = log_rows["supervised"][step]["supervised_loss"]
        plain_loss_0 = log_rows["plain_kd"][step]["supervised_loss"]
        if step == 0:
            assert np.isclose(sup_loss_0, plain_loss_0, atol=1e-5), \
                f"step {step} supervised loss must match under identical RNG"

        # L/U schedules stay matched. Guarded arms report their extra audit reads;
        # those are intentionally absent from plain KD and supervised.
        for arm in ("plain_kd", "joint_guard"):
            assert log_rows[arm][step]["data_schedule_sha256"] == log_rows["supervised"][step]["data_schedule_sha256"]
            for role in ("labeled", "unlabeled"):
                assert log_rows[arm][step]["slices_sampled"][role] == log_rows["supervised"][step]["slices_sampled"][role]
        assert log_rows["plain_kd"][step]["slices_sampled"]["audit"] == 0


def test_separate_l_u_counts_costs_and_teacher_cost(tmp_path: Path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["steps"] = 2
    cfg["warmup_steps"] = 0
    cfg["method"] = "joint_guard"
    cfg["labeled_batch_size"] = 2
    cfg["unlabeled_batch_size"] = 1
    cfg["pseudo_threshold"] = 0.0
    _bind_synthetic_teacher(cfg, _synthetic_ids(1))

    report = train_compact(
        cfg,
        None,
        None,
        tmp_path / "out_costs",
        device="cpu",
        synthetic=True,
    )

    # Separate L and U counts
    assert "labeled" in report["slices_sampled"]
    assert "unlabeled" in report["slices_sampled"]
    assert report["slices_sampled"]["labeled"] == 4
    assert report["slices_sampled"]["unlabeled"] == 2
    assert report["batch_sizes"] == {"labeled": 2, "unlabeled": 1}
    assert report["effective_batch_counts"] == {"labeled": 4, "unlabeled": 2}

    assert "labeled" in report["slices_optimized"]
    assert "unlabeled" in report["slices_optimized"]
    assert report["slices_optimized"]["labeled"] == 4

    # Optimizer calls separate AdamW and SGD
    assert "adamw" in report["optimizer_calls"]
    assert "sgd" in report["optimizer_calls"]
    assert report["optimizer_calls"]["adamw"] == 2

    # Teacher costs reported: cache cost float and training cost NOT_KNOWN
    tc = report["teacher_costs"]
    assert isinstance(tc["teacher_cache_total_seconds"], float)
    assert tc["teacher_training_cost"] == "NOT_KNOWN"
    assert tc["teacher_training_cost"] != 0


def test_public_cli_synthetic_smoke_and_guards(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    py_exec = sys.executable
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    train_manifest, train_sha = synthetic_compact_fixture(
        tmp_path / "cli_train_data", n_labeled=2, n_unlabeled=2, size=16,
        num_frames=2, num_slices=2,
    )
    cache_dir = tmp_path / "cli_teacher_cache"
    cache_cmd = [
        py_exec,
        str(root / "scripts" / "build_compact_teacher_cache.py"),
        "--data-manifest", str(train_manifest),
        "--data-sha256", train_sha,
        "--output-dir", str(cache_dir),
        "--synthetic-teacher",
        "--device", "cpu",
        "--batch-size", "2",
    ]
    cache_res = subprocess.run(cache_cmd, capture_output=True, text=True, cwd=str(root), env=env, timeout=120)
    assert cache_res.returncode == 0, f"teacher cache build failed: {cache_res.stderr}"
    cache_manifest = cache_dir / "CACHE_MANIFEST.json"
    cache_sha = digest(cache_manifest)
    cache_identity = json.loads(cache_manifest.read_text())["teacher_identity"]
    labeled_ids = CompactTrainingData(train_manifest, train_sha, synthetic=True, allow_k1=True).lids

    # Every public method gets its own bounded CLI run with the same generated,
    # hash-pinned FixedTeacherCache. The cache builder and reader are real code.
    reports = {}
    for method in ("supervised", "plain_kd", "scalar_guard", "joint_guard"):
        cfg = json.loads((root / "configs" / "compact" / f"{method}.json").read_text())
        cfg.update(
            steps=2,
            warmup_steps=0,
            seed=217,
            batch_size=1,
            labeled_batch_size=1,
            unlabeled_batch_size=2,
            crop_size=16,
            pseudo_threshold=0.0,
            max_trials=2,
        )
        cfg["model"] = {"name": "CompactStudent", "widths": [4, 8], "in_channels": 3, "num_classes": 4}
        cfg["data"] = {"manifest_sha256": train_sha, "budget_k": len(labeled_ids), "labeled_ids": labeled_ids}
        if method != "supervised":
            lineage = {"budget_k": len(labeled_ids), "exact_labeled_ids": labeled_ids}
            cfg["teacher"].update(
                cache_path=str(cache_manifest),
                cache_sha256=cache_sha,
                teacher_identity=cache_identity,
                lineage=lineage,
                source_training_cost="NOT_KNOWN",
            )
        config_path = tmp_path / f"cli_{method}.json"
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
        out_dir = tmp_path / f"cli_smoke_{method}"
        cmd = [
            py_exec,
            str(root / "train.py"),
            "--config", str(config_path),
            "--data", str(train_manifest),
            "--data-sha256", train_sha,
            "--synthetic",
            "--out", str(out_dir),
            "--device", "cpu",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root), env=env, timeout=120)
        assert res.returncode == 0, f"{method} CLI smoke failed: {res.stderr}"
        report = json.loads((out_dir / "REPORT.json").read_text())
        assert report["status"] == "COMPLETED"
        assert report["quality"] == "NOT_EVALUATED"
        assert report["quality_gate"]["threshold_macro4"] == 0.87
        assert report["train_manifest_sha256"] == train_sha
        assert report["final_manifest_sha256"] == digest(out_dir / "final" / "MANIFEST.json")
        assert report["final_weights_sha256"] == digest(out_dir / "final" / "weights.pt")
        assert report["final_checkpoint_path"] == str(out_dir / "final")
        assert report["report_path"] == str(out_dir / "REPORT.json")
        for source_name, source_sha in report["source_hashes"].items():
            assert source_sha == digest(root / source_name)
        if method != "supervised":
            assert report["teacher_preflight"]["status"] == "PASS"
        reports[method] = report

    for method in ("plain_kd", "scalar_guard", "joint_guard"):
        assert reports[method]["teacher_cache_manifest_sha256"] == cache_sha
    schedule_hashes = {
        method: [row["data_schedule_sha256"] for row in (
            json.loads(line) for line in (tmp_path / f"cli_smoke_{method}" / "TRAIN_LOG.jsonl").read_text().splitlines()
        )]
        for method in reports
    }
    assert len({tuple(values) for values in schedule_hashes.values()}) == 1
    assert reports["plain_kd"]["audit_consumption"]["slice_reads"] == 0
    assert reports["plain_kd"]["runtime_comparability"] == "PLAIN_KD_EXCLUDES_AUDIT_NOT_LATENCY_MATCHED"

    # 3. Guard: Reject legacy-only flags on compact config
    cmd_bad_legacy = [
        py_exec,
        str(root / "train.py"),
        "--config",
        str(root / "configs" / "compact" / "supervised.json"),
        "--legacy-checkpoint",
        "dummy_ckpt",
        "--out",
        str(tmp_path / "dummy"),
    ]
    res_bad_legacy = subprocess.run(cmd_bad_legacy, capture_output=True, text=True, cwd=str(root))
    assert res_bad_legacy.returncode != 0
    assert "legacy-only flags are invalid for self_audit.compact.v1" in res_bad_legacy.stderr

    # 4. Guard: Reject volume-only flags on compact config
    cmd_bad_vol = [
        py_exec,
        str(root / "train.py"),
        "--config",
        str(root / "configs" / "compact" / "supervised.json"),
        "--geometry",
        "dummy_geom",
        "--out",
        str(tmp_path / "dummy"),
    ]
    res_bad_vol = subprocess.run(cmd_bad_vol, capture_output=True, text=True, cwd=str(root))
    assert res_bad_vol.returncode != 0
    assert "volume flags are invalid for self_audit.compact.v1" in res_bad_vol.stderr

    # 5. Guard: Reject automatic-only flags on compact config
    cmd_bad_auto = [
        py_exec,
        str(root / "train.py"),
        "--config",
        str(root / "configs" / "compact" / "supervised.json"),
        "--phase",
        "adaptation",
        "--out",
        str(tmp_path / "dummy"),
    ]
    res_bad_auto = subprocess.run(cmd_bad_auto, capture_output=True, text=True, cwd=str(root))
    assert res_bad_auto.returncode != 0
    assert "automatic-only flags are invalid for self_audit.compact.v1" in res_bad_auto.stderr

    # 6. Guard: Label protection: non-synthetic runs require --data and --data-sha256
    cmd_no_data = [
        py_exec,
        str(root / "train.py"),
        "--config",
        str(root / "configs" / "compact" / "supervised.json"),
        "--out",
        str(tmp_path / "dummy"),
    ]
    res_no_data = subprocess.run(cmd_no_data, capture_output=True, text=True, cwd=str(root))
    assert res_no_data.returncode != 0
    assert "compact patient runs require --data/--data-sha256" in res_no_data.stderr

    # 7. Guard: Budget steps guard: > 128 updates requires --allow-full-run
    cmd_too_long = [
        py_exec,
        str(root / "train.py"),
        "--config",
        str(root / "configs" / "compact" / "supervised.json"),
        "--synthetic",
        "--steps",
        "200",
        "--out",
        str(tmp_path / "dummy"),
    ]
    res_too_long = subprocess.run(cmd_too_long, capture_output=True, text=True, cwd=str(root))
    assert res_too_long.returncode != 0
    assert "more than 128 updates requires --allow-full-run" in res_too_long.stderr


def test_public_cli_forwards_allow_full_run_without_starting_long_training(monkeypatch, tmp_path: Path):
    import train as train_module
    import self_audit.engine as engine

    root = Path(__file__).resolve().parents[1]
    seen = {}

    def fake_train(config, manifest, manifest_sha256, out, **kwargs):
        seen.update(kwargs)
        seen["config_steps"] = config["steps"]
        return {"status": "STUBBED_NO_TRAINING"}

    monkeypatch.setattr(engine, "train", fake_train)
    monkeypatch.setattr(sys, "argv", [
        str(root / "train.py"),
        "--config", str(root / "configs/compact/supervised.json"),
        "--synthetic",
        "--steps", "129",
        "--allow-full-run",
        "--out", str(tmp_path / "never_created"),
    ])
    train_module.main()
    assert seen["allow_full_run"] is True
    assert seen["stop_steps"] == 129
    assert seen["config_steps"] == 129
    assert not (tmp_path / "never_created").exists()


def test_compact_cuda_startup_initialization_ordering(monkeypatch, tmp_path: Path):
    """Verify fresh CUDA startup initializes CUDA state before resetting peak stats,

    and resets peak stats before model build/load to retain peak memory scope.
    Simulates the fresh-process failure where torch.cuda.reset_peak_memory_stats(device)
    raises RuntimeError('Invalid device argument') if called prior to torch.cuda.init().
    """
    import self_audit.compact_training as compact_training

    manifest, manifest_sha = synthetic_compact_fixture(
        tmp_path / "cuda_startup",
        n_labeled=2,
        n_unlabeled=1,
        size=8,
        num_frames=1,
        num_slices=1,
    )
    metadata_data = compact_training.CompactTrainingData(manifest, manifest_sha, synthetic=True, allow_k1=True)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(method="plain_kd", steps=2, warmup_steps=1)
    _bind_synthetic_teacher(cfg, metadata_data.lids)
    cfg["data"] = {
        "manifest_sha256": manifest_sha,
        "budget_k": len(metadata_data.lids),
        "labeled_ids": list(metadata_data.lids),
    }

    # 1. Positive fresh training startup: torch.cuda.init() precedes reset_peak_memory_stats,
    # which precedes build_compact_model (ensuring student parameters are counted in peak stats).
    events = []
    cuda_state = {"initialized": False}

    def fake_init():
        cuda_state["initialized"] = True
        events.append("torch.cuda.init")

    def fake_reset_peak_memory_stats(dev):
        if not cuda_state["initialized"]:
            raise RuntimeError("Invalid device argument ")
        events.append(("torch.cuda.reset_peak_memory_stats", str(dev)))

    class _StartupHalt(Exception):
        pass

    def tracking_build_compact_model(*args, **kwargs):
        events.append("build_compact_model")
        raise _StartupHalt("startup_verification_halt")

    monkeypatch.setattr(torch.cuda, "init", fake_init)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", fake_reset_peak_memory_stats)
    monkeypatch.setattr(compact_training, "build_compact_model", tracking_build_compact_model)

    with pytest.raises(_StartupHalt, match="startup_verification_halt"):
        train_compact(
            cfg,
            manifest,
            manifest_sha,
            tmp_path / "out_startup",
            device="cuda:0",
            synthetic=True,
            stop_steps=1,
        )

    assert events == [
        "torch.cuda.init",
        ("torch.cuda.reset_peak_memory_stats", "cuda:0"),
        "build_compact_model",
    ]
    assert cuda_state["initialized"] is True

    # 2. Negative test: uninitialized CUDA call reproduces RuntimeError('Invalid device argument')
    cuda_state["initialized"] = False
    monkeypatch.setattr(torch.cuda, "init", lambda: None)  # simulate missing initialization
    with pytest.raises(RuntimeError, match="Invalid device argument"):
        train_compact(
            cfg,
            manifest,
            manifest_sha,
            tmp_path / "out_uninit",
            device="cuda:0",
            synthetic=True,
            stop_steps=1,
        )

    # 3. Positive resume startup: torch.cuda.init() and reset precede load_exact_compact_checkpoint.
    events.clear()
    cuda_state["initialized"] = False
    monkeypatch.setattr(torch.cuda, "init", fake_init)

    dummy_resume = tmp_path / "dummy_resume_dir"
    dummy_resume.mkdir()

    def tracking_load_exact(*args, **kwargs):
        events.append("load_exact_compact_checkpoint")
        raise _StartupHalt("resume_verification_halt")

    monkeypatch.setattr(compact_training, "load_exact_compact_checkpoint", tracking_load_exact)

    with pytest.raises(_StartupHalt, match="resume_verification_halt"):
        train_compact(
            cfg,
            manifest,
            manifest_sha,
            tmp_path / "out_resume",
            device="cuda:0",
            synthetic=True,
            stop_steps=2,
            resume=dummy_resume,
            resume_sha256="0" * 64,
        )

    assert events == [
        "torch.cuda.init",
        ("torch.cuda.reset_peak_memory_stats", "cuda:0"),
        "load_exact_compact_checkpoint",
    ]
    assert cuda_state["initialized"] is True
