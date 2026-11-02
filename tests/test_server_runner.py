from __future__ import annotations

import copy
import hashlib
import json
import random
import os
from pathlib import Path
import stat
import time

import numpy as np
import pytest
import torch

from self_audit.compact_artifacts import digest, load_exact_compact_checkpoint
from self_audit.compact_data import synthetic_compact_fixture
from self_audit.compact_training import train_compact
from self_audit import compact_training
from self_audit import server_runner
from self_audit.server_runner import (
    SubprocessRclone,
    atomic_json,
    backup_artifacts,
    build_s0_config,
    canonical_sha256,
    create_protocol_lock,
    evaluate_checkpoint,
    profile_batches,
    profile_arm_config,
    retry_backup,
    run_locked_experiment,
    sha256_file,
    source_identity,
    _train_metadata,
    verify_protocol_lock,
    _validate_server_arm_config,
)


def _small_config(steps: int = 3) -> dict:
    return {
        "schema": "self_audit.compact.v1",
        "method": "supervised",
        "selector": "original",
        "seed": 42,
        "steps": steps,
        "warmup_steps": 1,
        "save_every": 2,
        "batch_size": 1,
        "labeled_batch_size": 1,
        "unlabeled_batch_size": 1,
        "crop_size": 8,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "model": {
            "name": "CompactStudent",
            "in_channels": 3,
            "num_classes": 4,
            "widths": [4, 8, 16, 32],
            "init": "kaiming_normal",
        },
        "teacher": {},
        "acceptance": {"metric": "Macro4", "min_macro4": 0.87},
        "equal_wall_time": {"enabled": False},
    }


def test_server_s0_recipe_and_profile_batches_are_explicit():
    config = build_s0_config(64)
    assert config["method"] == "supervised"
    assert config["steps"] == 10000
    assert config["seed"] == 42
    assert config["crop_size"] == 160
    assert config["model"]["widths"] == [32, 64, 128, 256]
    assert config["model"]["init"] == "kaiming_normal"
    assert not config["model"].get("pretrained")
    assert config["labeled_batch_size"] == config["unlabeled_batch_size"] == 64
    assert config["eval_batch_size"] == 8
    assert build_s0_config(8)["labeled_batch_size"] == 8
    with pytest.raises(ValueError, match="batch 32 or 64"):
        create_protocol_lock(
            Path("unused-lock.json"), profile_path=Path("unused-profile.json"), profile_sha256="0" * 64,
            train_manifest=Path("unused-train.json"), train_sha256="0" * 64,
            val_manifest=Path("unused-val.json"), val_sha256="0" * 64, batch_size=8,
        )


def test_hash_pinned_real_train_manifest_without_synthetic_field_is_supported(tmp_path):
    train_path = tmp_path / "train_real_schema.json"
    atomic_json(train_path, {
        "schema": "self_audit.data.v2",
        "role": "training",
        "label_budget": 1,
        "labeled": [{"id": "L001"}],
        "unlabeled": [{"id": "U001"}],
    })
    _manifest, lids, synthetic = _train_metadata(train_path, sha256_file(train_path))
    assert lids == ["L001"]
    assert synthetic is False


def test_source_identity_hashes_full_python_closure_configs_and_git_head():
    identity = source_identity()
    runtime_files = {path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
                     for path in (Path(__file__).resolve().parents[1] / "self_audit").rglob("*.py")}
    assert runtime_files <= set(identity["files"])
    assert "configs/fresh_s0/supervised_wide_seed42.json" in identity["files"]
    assert "prepare_acdc.py" in identity["files"]
    assert identity["git_head"] and len(identity["git_head"]) >= 40
    assert identity["closure_sha256"] == canonical_sha256(
        {"git_head": identity["git_head"].lower(), "files": identity["files"]}
    )


def test_report_backup_keeps_metrics_without_intermediate_prediction_payloads(tmp_path):
    from self_audit.server_runner import _add_backup_tree
    folder = tmp_path / "validation"
    folder.mkdir()
    (folder / "METRICS.json").write_text('{}\n')
    (folder / "PREDICTIONS_FROZEN.json").write_text('{}\n')
    (folder / "prediction.npy").write_bytes(b"local prediction payload")
    (folder / "weights.pt").write_bytes(b"intermediate export")
    files = []
    _add_backup_tree(files, folder, "validation", reports_only=True)
    assert {relative for _, relative in files} == {
        "validation/METRICS.json", "validation/PREDICTIONS_FROZEN.json"
    }
    assert (folder / "prediction.npy").read_bytes() == b"local prediction payload"


def test_profile_and_protocol_lock_use_manufactured_train(monkeypatch, tmp_path):
    def fake_train(config, manifest, manifest_sha, out, *, device, synthetic, stop_after_step, allow_full_run):
        out = Path(out)
        out.mkdir(parents=True)
        rows = []
        for step in range(stop_after_step):
            rows.append({
                "step": step,
                "timings": {"step_elapsed_seconds": 0.02},
                "effective_batch_counts": {
                    "labeled": config["labeled_batch_size"],
                    "unlabeled": config["unlabeled_batch_size"],
                },
                "cuda_memory": {"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None},
            })
        (out / "TRAIN_LOG.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        atomic_json(out / "REPORT.json", {"status": "STOPPED_BOUNDARY", "steps": stop_after_step})
        return {"status": "STOPPED_BOUNDARY", "steps": stop_after_step}

    monkeypatch.setattr("self_audit.compact_training.train_compact", fake_train)
    profile_dir = tmp_path / "profile"
    profile = profile_batches(profile_dir, synthetic=True, device="cpu", warmup_steps=1,
                              measured_steps=2)
    assert profile["synthetic"] is True
    assert [result["batch_size"] for result in profile["results"]] == [8, 32, 64]
    assert all(result["status"] == "MEASURED" for result in profile["results"])
    assert all(result["cuda_peak_allocated_bytes"] is None for result in profile["results"])

    val_path = tmp_path / "val.json"
    val = {"schema": "self_audit.eval.v2", "role": "development", "synthetic": True,
           "patients": [{"id": "synthetic_val_001"}]}
    atomic_json(val_path, val)
    lock_path = tmp_path / "protocol_lock.json"
    lock = create_protocol_lock(
        lock_path,
        profile_path=profile_dir / "PROFILE.json",
        profile_sha256=sha256_file(profile_dir / "PROFILE.json"),
        train_manifest=profile["train_manifest"],
        train_sha256=profile["train_manifest_sha256"],
        val_manifest=val_path,
        val_sha256=sha256_file(val_path),
        batch_size=32,
    )
    verified = verify_protocol_lock(lock_path, sha256_file(lock_path))
    assert lock["schema"] == "self_audit.server_compact.lock.v1"
    assert "arm_config_path" not in lock
    assert verified["config"]["batch_size"] == 32
    assert verified["config"]["labeled_batch_size"] == verified["config"]["unlabeled_batch_size"] == 32
    assert lock["protocol"]["steps"] == 10000
    assert lock["protocol"]["eval_batch_size"] == 8
    assert lock["config"]["eval_batch_size"] == 8
    assert lock["inputs"]["test"] is None


def _synthetic_server_kd_config(tmp_path, *, arm_id="S1", crop_size=160):
    train_manifest, train_sha = synthetic_compact_fixture(
        tmp_path / "train", seed=41, n_labeled=2, n_unlabeled=2,
        size=16, num_frames=2, num_slices=2,
    )
    train = json.loads(train_manifest.read_text())
    labeled_ids = [record["id"] for record in train["labeled"]]
    unlabeled_ids = [record["id"] for record in train["unlabeled"]]
    identity = {
        "schema": "self_audit.compact_teacher_identity.v1",
        "teacher_kind": "synthetic",
        "inference_state": "student",
        "training_method": "ema_full",
        "label_budget": len(labeled_ids),
        "labeled_ids": labeled_ids,
        "lineage": {"budget_k": len(labeled_ids), "exact_labeled_ids": labeled_ids},
    }
    cache_path = tmp_path / "cache" / "CACHE_MANIFEST.json"
    atomic_json(cache_path, {
        "schema": "self_audit.compact_teacher_cache.v1",
        "native_axes": "TZYX",
        "num_classes": 4,
        "train_manifest_sha256": train_sha,
        "teacher_identity": identity,
        "patients": {uid: {} for uid in unlabeled_ids},
    })
    config = {
        "schema": "self_audit.compact.v1",
        "server_arm": arm_id,
        "experiment": {"id": f"server_compact_{arm_id.lower()}_test", "launchable": True},
        "method": "plain_kd",
        "selector": "original",
        "seed": 42,
        "steps": 10000,
        "warmup_steps": 1000,
        "save_every": 1000,
        "model": {
            "name": "CompactStudent", "in_channels": 3, "num_classes": 4,
            "widths": [32, 64, 128, 256], "init": "kaiming_normal",
        },
        "crop_size": crop_size,
        "batch_size": 32,
        "labeled_batch_size": 32,
        "unlabeled_batch_size": 32,
        "eval_batch_size": 8,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "pseudo_lr": 0.001,
        "temperature": 1.0,
        "coefficient": 1.0,
        "pseudo_threshold": 0.7,
        "ridge": 1e-6,
        "risk_budget": 0.0,
        "absent_fp_budget": 0.0,
        "guard_tolerance": 1e-6,
        "max_trials": 8,
        "equal_wall_time": {"enabled": False},
        "acceptance": {"metric": "Macro4", "min_macro4": 0.87},
        "data": {
            "manifest_sha256": train_sha,
            "budget_k": len(labeled_ids),
            "labeled_ids": labeled_ids,
        },
        "teacher": {
            "cache_path": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "teacher_identity_sha256": canonical_sha256(identity),
            "teacher_identity": identity,
            "lineage": {"budget_k": len(labeled_ids), "exact_labeled_ids": labeled_ids},
        },
    }
    return train_manifest, train_sha, config


def test_generic_kd_arm_config_and_profile_lock_are_pinned_and_bounded(monkeypatch, tmp_path):
    train_manifest, train_sha, config = _synthetic_server_kd_config(tmp_path)
    config_path = tmp_path / "s1.json"
    atomic_json(config_path, config)
    config_sha = sha256_file(config_path)
    val_path = tmp_path / "val.json"
    val = {"schema": "self_audit.eval.v2", "role": "development", "synthetic": True,
           "patients": [{"id": "synthetic_val_001"}]}
    atomic_json(val_path, val)
    val_sha = sha256_file(val_path)

    def fake_source_identity(root=server_runner.ROOT):
        return {"git_head": "1" * 40, "git_status_paths": [], "files": {}, "closure_sha256": "a" * 64}

    observed = {}

    def fake_train(profile_config, manifest, manifest_sha, out, *, device, synthetic, stop_after_step, allow_full_run):
        observed.update({"steps": profile_config["steps"], "warmup_steps": profile_config["warmup_steps"],
                         "stop_after_step": stop_after_step, "allow_full_run": allow_full_run})
        out = Path(out)
        out.mkdir(parents=True)
        rows = [{
            "step": step,
            "timings": {"step_elapsed_seconds": 0.02},
            "effective_batch_counts": {"labeled": 32, "unlabeled": 32},
            "cuda_memory": {"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None},
        } for step in range(stop_after_step)]
        (out / "TRAIN_LOG.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        atomic_json(out / "REPORT.json", {"status": "STOPPED_BOUNDARY", "steps": stop_after_step})
        return {"status": "STOPPED_BOUNDARY", "steps": stop_after_step}

    monkeypatch.setattr(server_runner, "source_identity", fake_source_identity)
    monkeypatch.setattr(compact_training, "train_compact", fake_train)
    profile_dir = tmp_path / "profile"
    profile = profile_arm_config(
        profile_dir,
        arm_config_path=config_path,
        arm_config_sha256=config_sha,
        train_manifest=train_manifest,
        train_sha256=train_sha,
        synthetic=True,
        device="cpu",
        warmup_steps=2,
        measured_steps=3,
    )
    assert observed == {"steps": 5, "warmup_steps": 2, "stop_after_step": 5, "allow_full_run": True}
    assert profile["recipe_warmup_steps"] == 1000
    assert profile["profile_warmup_steps"] == 2
    assert profile["arm_config_sha256"] == config_sha

    lock_path = tmp_path / "lock.json"
    lock = create_protocol_lock(
        lock_path,
        profile_path=profile_dir / "PROFILE.json",
        profile_sha256=sha256_file(profile_dir / "PROFILE.json"),
        train_manifest=train_manifest,
        train_sha256=train_sha,
        val_manifest=val_path,
        val_sha256=val_sha,
        batch_size=32,
        arm_config_path=config_path,
        arm_config_sha256=config_sha,
    )
    verified = verify_protocol_lock(lock_path, sha256_file(lock_path))
    assert lock["schema"] == "self_audit.server_compact.lock.v2"
    assert verified["config"]["crop_size"] == 160
    assert verified["protocol"]["warmup_steps"] == 1000
    assert verified["protocol"]["arm_config_sha256"] == config_sha
    assert verified["protocol"]["kd_loss"] == "soft_cross_entropy"
    assert verified["protocol"]["kd_denominator"] == "common_valid_confident_pixel_count"
    run_root = tmp_path / "run_without_arm_pin"
    with pytest.raises(ValueError, match="generic arm run requires --arm-config"):
        run_locked_experiment(lock_path, sha256_file(lock_path), run_root, device="cpu",
                              rclone_base="drive:unused")
    assert not run_root.exists()
    with pytest.raises(ValueError, match="differs from the verified protocol lock"):
        run_locked_experiment(lock_path, sha256_file(lock_path), run_root, device="cpu",
                              rclone_base="drive:unused", arm_config_path=config_path,
                              arm_config_sha256="f" * 64)
    assert not run_root.exists()


def test_generic_kd_arm_rejects_mismatched_teacher_crop_batch_and_steps(tmp_path):
    train_manifest, train_sha, baseline = _synthetic_server_kd_config(tmp_path)
    config_path = tmp_path / "arm.json"

    variants = []
    changed = copy.deepcopy(baseline)
    changed["crop_size"] = 256
    variants.append((changed, "S1/crop160 or S2/crop256"))
    changed = copy.deepcopy(baseline)
    changed.update(batch_size=16, labeled_batch_size=16, unlabeled_batch_size=16)
    variants.append((changed, "batch, labeled batch, and unlabeled batch must all be 32"))
    changed = copy.deepcopy(baseline)
    changed["steps"] = 9999
    variants.append((changed, "steps and checkpoint cadence must remain 10,000 and 1,000"))
    changed = copy.deepcopy(baseline)
    changed["teacher"]["teacher_identity"]["payload_sha256"] = "f" * 64
    changed["teacher"]["teacher_identity_sha256"] = canonical_sha256(changed["teacher"]["teacher_identity"])
    variants.append((changed, "cache identity differs"))
    changed = copy.deepcopy(baseline)
    changed["teacher"]["cache_path"] = None
    variants.append((changed, "fixed teacher cache path"))

    for config, message in variants:
        atomic_json(config_path, config)
        with pytest.raises((TypeError, ValueError), match=message):
            _validate_server_arm_config(config_path, sha256_file(config_path), train_manifest, train_sha)


def test_server_arm_templates_match_historical_wide_kd_without_old_teacher_pins():
    root = Path(__file__).resolve().parents[1]
    s1 = json.loads((root / "configs/server_compact/s1_plain_kd.json").read_text())
    s2 = json.loads((root / "configs/server_compact/s2_plain_kd.json").read_text())
    historical = json.loads((root / "results/compact_acdc_l8_20261029/experiment_20261028/wide_plain_kd_config.json").read_text())
    for config in (s1, s2):
        assert config["experiment"]["launchable"] is False
        for key in ("steps", "warmup_steps", "seed", "lr", "weight_decay", "pseudo_lr", "temperature",
                    "coefficient", "pseudo_threshold"):
            assert config[key] == historical[key]
        assert config["batch_size"] == config["labeled_batch_size"] == config["unlabeled_batch_size"] == 32
        assert config["model"]["widths"] == historical["model"]["widths"]
        assert config["teacher"]["teacher_identity"] is None
        assert config["teacher"]["cache_sha256"] is None
    assert s1["crop_size"] == 160
    assert s2["crop_size"] == 256


def test_canonical_compact_trainer_exact_resume_on_manufactured_data(tmp_path):
    train_manifest, train_sha = synthetic_compact_fixture(
        tmp_path / "data", seed=17, n_labeled=2, n_unlabeled=2,
        size=16, num_frames=2, num_slices=2,
    )
    config = _small_config(steps=3)
    first = tmp_path / "segment_1"
    partial = train_compact(config, train_manifest, train_sha, first, device="cpu", synthetic=True,
                            stop_after_step=1, allow_full_run=True)
    assert partial["status"] == "STOPPED_BOUNDARY"
    resume_checkpoint = first / "final"
    resume_sha = sha256_file(resume_checkpoint / "MANIFEST.json")
    resumed = train_compact(config, train_manifest, train_sha, tmp_path / "segment_2", device="cpu",
                            synthetic=True, stop_after_step=3, resume=resume_checkpoint,
                            resume_sha256=resume_sha, allow_full_run=True)
    uninterrupted = train_compact(config, train_manifest, train_sha, tmp_path / "uninterrupted",
                                  device="cpu", synthetic=True, stop_after_step=3, allow_full_run=True)
    assert resumed["status"] == uninterrupted["status"] == "COMPLETED"
    assert resumed["steps"] == uninterrupted["steps"] == 3
    resumed_model, _, _ = load_exact_compact_checkpoint(
        resumed["final_checkpoint_path"], resumed["final_manifest_sha256"], device="cpu"
    )
    baseline_model, _, _ = load_exact_compact_checkpoint(
        uninterrupted["final_checkpoint_path"], uninterrupted["final_manifest_sha256"], device="cpu"
    )
    assert resumed_model.state_dict().keys() == baseline_model.state_dict().keys()
    assert all(torch.equal(resumed_model.state_dict()[key], baseline_model.state_dict()[key])
               for key in resumed_model.state_dict())
    log = Path(resumed["output_dir"]) / "TRAIN_LOG.jsonl"
    assert len(log.read_text().splitlines()) == 3


def test_checkpoint_validation_preserves_training_state_and_keeps_earliest_tie(monkeypatch, tmp_path):
    import self_audit.server_runner as runner

    config = {"steps": 10000, "method": "supervised"}
    lock = {"config": config,
            "inputs": {"train": {"sha256": "a" * 64}, "val": {"sha256": "b" * 64},
                       "synthetic": True},
            "protocol": {"batch_size": 32, "best_rule": "highest VAL class_first_macro4; exact ties keep earliest step"}}
    run_root = tmp_path / "run"
    run_root.mkdir()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.train()
    before_params = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before_lr = optimizer.param_groups[0]["lr"]
    before_python_rng = random.getstate()
    before_numpy_rng = np.random.get_state()
    before_torch_rng = torch.get_rng_state().clone()
    scores = iter([0.8, 0.9, 0.9])

    def fake_evaluate(checkpoint, checkpoint_sha, *, lock, validation_dir, device):
        import self_audit.server_runner as sr

        validation_dir.mkdir(parents=True)
        export_dir = validation_dir / "export"
        export_dir.mkdir()
        (export_dir / "weights.pt").write_bytes(f"raw-weights-{Path(checkpoint).name}".encode())
        atomic_json(export_dir / "MANIFEST.json", {
            "architecture": {"name": "CompactStudent", "widths": [4, 8, 16, 32]},
            "preprocessing": {"normalization": "prepared"},
            "class_map": {"0": "background"},
        })
        eval_dir = validation_dir / "val"
        eval_dir.mkdir()
        macro4 = next(scores)
        atomic_json(eval_dir / "METRICS.json", {"class_first_macro4": macro4,
                                                "class_first_foreground_macro": macro4 - 0.1})
        atomic_json(eval_dir / "PREDICTIONS_FROZEN.json", {
            "evaluation_manifest_sha256": lock["inputs"]["val"]["sha256"],
            "coverage": {"complete": True},
        })
        random.random()
        np.random.random()
        torch.rand(1)
        with torch.no_grad():
            model.weight.add_(2)
        optimizer.param_groups[0]["lr"] = 0.7
        model.eval()
        return {
            "schema": "self_audit.server_compact.validation.v1", "status": "VERIFIED",
            "macro4": macro4, "foreground_macro": macro4 - 0.1, "synthetic": True,
            "export_path": str(export_dir), "export_manifest_sha256": sr.sha256_file(export_dir / "MANIFEST.json"),
            "validation_dir": str(eval_dir), "metrics_sha256": sr.sha256_file(eval_dir / "METRICS.json"),
            "predictions_frozen_sha256": sr.sha256_file(eval_dir / "PREDICTIONS_FROZEN.json"),
            "prediction_freeze_status": "complete", "selection_metric": "class_first_macro4",
            "selection_note": "development only",
        }

    monkeypatch.setattr(runner, "_evaluate_checkpoint_payload", fake_evaluate)
    selected_steps = []
    for step in (1, 2, 3):
        checkpoint = tmp_path / f"checkpoint_{step:06d}"
        checkpoint.mkdir()
        atomic_json(checkpoint / "MANIFEST.json", {
            "step": step, "config": {**config, "data": {"manifest_sha256": "a" * 64}},
        })
        (checkpoint / "weights.pt").write_bytes(f"full-{step}".encode())
        checkpoint_sha = sha256_file(checkpoint / "MANIFEST.json")
        evaluate_checkpoint(checkpoint, checkpoint_sha, lock=lock, run_root=run_root,
                            device="cpu", model=model, optimizer=optimizer)
        best = json.loads((run_root / "BEST_SELECTION.json").read_text())
        selected_steps.append(best["selected_step"])
        assert model.training is True
        assert optimizer.param_groups[0]["lr"] == before_lr
        assert all(torch.equal(model.state_dict()[key], value) for key, value in before_params.items())
        assert random.getstate() == before_python_rng
        assert np.array_equal(np.random.get_state()[1], before_numpy_rng[1])
        assert torch.equal(torch.get_rng_state(), before_torch_rng)
    assert selected_steps == [1, 2, 2]
    assert (run_root / "best.pt").read_bytes() == b"raw-weights-checkpoint_000002"
    selection = json.loads((run_root / "BEST_SELECTION.json").read_text())
    assert selection["architecture"]["widths"] == [4, 8, 16, 32]
    assert selection["preprocessing"]["normalization"] == "prepared"


class _FakeRclone:
    def __init__(self, fail_first_copy=False, mismatch=False):
        self.files = {}
        self.destinations = set()
        self.fail_first_copy = fail_first_copy
        self.mismatch = mismatch
        self.calls = {}

    def ensure_fresh_destination(self, base, destination):
        assert destination.startswith(base.rstrip("/") + "/")
        assert destination not in self.destinations
        self.destinations.add(destination)

    def copyto(self, local_path, remote_path):
        count = self.calls.get(remote_path, 0) + 1
        self.calls[remote_path] = count
        if self.fail_first_copy and count == 1:
            raise RuntimeError("temporary network failure")
        self.files[remote_path] = Path(local_path).read_bytes()

    def readback_sha256(self, remote_path):
        payload = self.files[remote_path]
        if self.mismatch:
            payload = payload + b"tampered"
        return len(payload), hashlib.sha256(payload).hexdigest()


def _make_backup_tree(root: Path):
    final = root / "segments/segment_0001/final"
    final_export = root / "final_export"
    best_export = root / "best_export"
    final.mkdir(parents=True)
    final_export.mkdir()
    best_export.mkdir()
    (final / "weights.pt").write_bytes(b"full-checkpoint-weights")
    atomic_json(final / "MANIFEST.json", {"step": 10000})
    (final_export / "weights.pt").write_bytes(b"terminal-raw-export-weights")
    atomic_json(final_export / "MANIFEST.json", {"source_checkpoint_manifest_sha256": "manifest"})
    (best_export / "weights.pt").write_bytes(b"best-raw-export-weights")
    atomic_json(best_export / "MANIFEST.json", {"schema": "compact.export"})
    (root / "best.pt").write_bytes(b"best-raw-export-weights")
    atomic_json(root / "BEST.json", {"step": 5000})
    atomic_json(root / "FULL_RESUME_CHECKPOINT.json", {"step": 10000})
    atomic_json(root / "RUN_LOCK.json", {"lock": "pinned"})
    atomic_json(root / "LOCKED_CONFIG.json", {"batch_size": 32})
    atomic_json(root / "LOCKED_PROFILE.json", {"profile": True})
    atomic_json(root / "LOCKED_BASE_RECIPE.json", {"recipe": True})
    atomic_json(root / "PROVENANCE.json", {"source": {"closure_sha256": "source"}})
    (root / "SHA256SUMS.txt").write_text("sum  receipt\n")
    atomic_json(root / "BEST_SELECTION.json", {"step": 5000})
    (root / "segments/segment_0001/TRAIN_LOG.jsonl").write_text("{}\n")
    (root / "segments/segment_0001/config.json").write_text("{}\n")
    return final


def test_backup_retries_and_stream_verifies_real_local_artifacts(tmp_path):
    final = _make_backup_tree(tmp_path)
    client = _FakeRclone(fail_first_copy=True)
    status = backup_artifacts(tmp_path, final_checkpoint=final, rclone_base="drive:server-runs",
                              lock_sha256="c" * 64, client=client, attempts=3,
                              retry_delay_seconds=0, sleep_fn=lambda _seconds: None)
    assert status["status"] == "VERIFIED"
    assert status["remote_verified"] is True
    assert status["local_retained"] is True
    assert len(status["files_verified"]) >= 10
    assert any(attempt["attempts"] == 2 for attempt in status["files_verified"])
    assert (tmp_path / "best.pt").read_bytes() == b"best-raw-export-weights"


def test_backup_hash_mismatch_reports_upload_pending_without_deleting_local(tmp_path):
    final = _make_backup_tree(tmp_path)
    client = _FakeRclone(mismatch=True)
    status = backup_artifacts(tmp_path, final_checkpoint=final, rclone_base="drive:server-runs",
                              lock_sha256="d" * 64, client=client, attempts=2,
                              retry_delay_seconds=0, sleep_fn=lambda _seconds: None)
    assert status["status"] == "UPLOAD_PENDING"
    assert status["remote_verified"] is False
    assert status["pending"]
    assert (tmp_path / "best.pt").is_file()
    persisted = json.loads((tmp_path / "BACKUP_STATUS.json").read_text())
    assert persisted["status"] == "UPLOAD_PENDING"
    assert client.calls and set(client.calls.values()) == {2}


def test_rclone_hard_deadlines_bound_command_and_streaming_cat_without_logging_stderr(tmp_path, monkeypatch):
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "rclone"
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        "sys.stderr.write('client_secret=never-log-this-token\\n'); sys.stderr.flush()\n"
        "time.sleep(5)\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(shim_dir) + os.pathsep + os.environ.get("PATH", ""))
    log_path = tmp_path / "rclone.log"
    rclone = SubprocessRclone(log_path, hard_timeout_seconds=0.15)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="hard deadline"):
        rclone._run(["rclone", "lsjson", "drive:private"])
    assert time.monotonic() - started < 1.5
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="hard deadline"):
        rclone.readback_sha256("drive:private/file")
    assert time.monotonic() - started < 1.5
    log = log_path.read_text()
    assert "never-log-this-token" not in log
    assert "client_secret" not in log
    assert "argv" not in log


def _write_eval_manifest(folder: Path, role: str, patient_id: str, *, seed: int) -> tuple[Path, str]:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cine = rng.random((2, 2, 9, 11), dtype=np.float32)
    image = folder / f"{patient_id}.npy"
    mask = folder / f"{patient_id}_frame0.npy"
    np.save(image, cine)
    np.save(mask, (cine[0] * 4).astype(np.uint8))
    affine = np.diag([1.25, 1.5, 7.5, 1.0])
    affine[0, 3] = seed * 10 + 3
    manifest = folder / "manifest.json"
    atomic_json(manifest, {
        "schema": "self_audit.native_manifest.v1", "role": role, "synthetic": True,
        "patients": [{
            "id": patient_id, "domain": "manufactured", "image": image.name,
            "image_sha256": sha256_file(image), "axes": ["T", "Z", "Y", "X"],
            "affine": affine.tolist(),
            "masks": {"0": {"path": mask.name, "sha256": sha256_file(mask)}},
        }],
    })
    return manifest, sha256_file(manifest)


def test_failed_core_backup_retry_completes_manufactured_test_without_retraining(monkeypatch, tmp_path):
    from self_audit.compact_artifacts import build_compact_model

    train_manifest, train_sha = synthetic_compact_fixture(
        tmp_path / "train", seed=51, n_labeled=1, n_unlabeled=1, size=16, num_frames=2, num_slices=2
    )
    train_obj = json.loads(Path(train_manifest).read_text())
    val_manifest, val_sha = _write_eval_manifest(tmp_path / "val", "development", "synthetic_val", seed=61)
    test_manifest, test_sha = _write_eval_manifest(tmp_path / "test", "test", "synthetic_test", seed=71)

    def fake_train(config, manifest_path, manifest_sha256, out_dir, *, device, synthetic,
                   allow_full_run, stop_after_step=None, resume=None, resume_sha256=None):
        if stop_after_step is not None:
            profile_out = Path(out_dir)
            profile_out.mkdir(parents=True)
            rows = [{
                "step": step, "timings": {"step_elapsed_seconds": 0.02},
                "effective_batch_counts": {"labeled": config["labeled_batch_size"],
                                           "unlabeled": config["unlabeled_batch_size"]},
                "cuda_memory": {"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None},
            } for step in range(stop_after_step)]
            (profile_out / "TRAIN_LOG.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            atomic_json(profile_out / "REPORT.json", {"status": "STOPPED_BOUNDARY", "steps": stop_after_step})
            return {"status": "STOPPED_BOUNDARY", "steps": stop_after_step}
        assert synthetic and device == "cpu" and allow_full_run
        out = Path(out_dir)
        out.mkdir(parents=True)
        (out / "TRAIN_LOG.jsonl").write_text("")
        cfg = dict(config)
        cfg["data"] = {
            "manifest_sha256": manifest_sha256, "budget_k": 1,
            "labeled_ids": [record["id"] for record in train_obj["labeled"]],
        }
        model = build_compact_model(cfg["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        data = {
            "lids": [record["id"] for record in train_obj["labeled"]],
            "uids": [record["id"] for record in train_obj["unlabeled"]],
            "manifest": train_obj, "path": str(manifest_path), "data_sha256": manifest_sha256,
            "synthetic": True,
        }
        rng_state = {
            "torch_cpu_rng": torch.get_rng_state(), "torch_cuda_rng": None,
            "np_legacy_rng": np.random.get_state(), "py_rng": random.getstate(),
            "data_rng_state": np.random.default_rng(42).bit_generator.state, "probe_rng_state": {},
        }
        initial_state_sha = compact_training._state_sha256(model)
        log_payload = b"{}\n"
        for step, name in ((9000, "step_009000"), (10000, "final")):
            checkpoint = out / name
            manifest_sha = compact_training.save_compact_checkpoint(
                model, optimizer, cfg, data, checkpoint, step=step, rng_state=rng_state,
                counters={
                    "completed_steps": step, "counts": {"supervised": step},
                    "slices_sampled": {"labeled": step}, "slices_optimized": {"labeled": step},
                    "optimizer_calls": {"supervised": step}, "cumulative_training_walltime": 0.0,
                    "total_cache_time": 0.0, "total_u_cache_time": 0.0,
                    "total_probe_cache_time": 0.0, "total_probe_sample_time": 0.0,
                    "seen_l": [train_obj["labeled"][0]["id"]], "seen_a": [], "seen_u": [],
                    "seen_probe": [], "audit_totals": {}, "audit_unique_frames": [],
                    "warm_start_logs": [], "cuda_peak_allocated_bytes": None,
                    "cuda_peak_reserved_bytes": None,
                },
                protocol={
                    "schema": cfg["schema"], "method": cfg["method"], "selector": cfg["selector"],
                    "steps": cfg["steps"], "warmup_steps": cfg["warmup_steps"], "seed": cfg["seed"],
                    "probe_seed": cfg["probe_seed"], "probe_batch_size": cfg["probe_batch_size"],
                    "selector_dice_tolerance": cfg["selector_dice_tolerance"],
                    "utility_gain_tolerance": cfg["utility_gain_tolerance"],
                    "utility_tie_tolerance": cfg["utility_tie_tolerance"],
                    "same_batch_utility": cfg["same_batch_utility"], "max_trials": cfg["max_trials"],
                    "train_manifest_sha256": manifest_sha256,
                    "initial_state_sha256": initial_state_sha, "log_bytes": len(log_payload),
                    "log_sha256": hashlib.sha256(log_payload).hexdigest(), "log_step_count": step,
                },
            )
            if step == 10000:
                final_path, final_sha = checkpoint, manifest_sha
        atomic_json(out / "REPORT.json", {"status": "COMPLETED", "steps": 10000})
        return {"status": "COMPLETED", "steps": 10000,
                "final_checkpoint_path": str(final_path), "final_manifest_sha256": final_sha,
                "output_dir": str(out)}

    monkeypatch.setattr(compact_training, "train_compact", fake_train)
    from self_audit import compact_inference
    original_evaluate = compact_inference.run_two_phase_evaluation
    evaluation_batches = []

    def record_evaluation_batch(*args, **kwargs):
        evaluation_batches.append(kwargs["batch_size"])
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(compact_inference, "run_two_phase_evaluation", record_evaluation_batch)
    profile_dir = tmp_path / "profile"
    profile = profile_batches(
        profile_dir, train_manifest=train_manifest, train_sha256=train_sha,
        synthetic=True, device="cpu", warmup_steps=0, measured_steps=1,
    )
    lock_path = tmp_path / "protocol_lock.json"
    create_protocol_lock(
        lock_path, profile_path=profile_dir / "PROFILE.json",
        profile_sha256=sha256_file(profile_dir / "PROFILE.json"),
        train_manifest=train_manifest, train_sha256=train_sha,
        val_manifest=val_manifest, val_sha256=val_sha, test_manifest=test_manifest,
        test_sha256=test_sha, batch_size=32,
    )
    initial_client = _FakeRclone(mismatch=True)
    run_root = tmp_path / "run"
    first = run_locked_experiment(
        lock_path, sha256_file(lock_path), run_root, device="cpu", rclone_base="drive:server-runs",
        backup_client=initial_client, retry_delay_seconds=0,
    )
    assert first["status"] == "UPLOAD_PENDING"
    assert first["backup_phase"] == "core"
    assert first["final_checkpoint"] == str(run_root / "segments/segment_0001/final")
    assert first["final_checkpoint_manifest_sha256"] == sha256_file(Path(first["final_checkpoint"]) / "MANIFEST.json")
    assert not (run_root / "TEST_RESULT.json").exists()
    assert (run_root / "final_export/MANIFEST.json").is_file()
    calls_before_retry = dict(initial_client.calls)

    resumed = retry_backup(
        run_root, rclone_base="drive:server-runs", client=_FakeRclone(),
        retry_delay_seconds=0, device="cpu",
    )
    assert resumed["status"] == "COMPLETED"
    assert resumed["test_status"] == "COMPLETED"
    assert evaluation_batches == [8, 8, 8]
    assert (run_root / "TEST_RESULT.json").is_file()
    assert json.loads((run_root / "BEST.json").read_text())["validation_dir"]
    assert json.loads((run_root / "SUPPLEMENTAL_EVALUATION_REPORT.json").read_text())["status"] == "TEST_SCORED"
    sums = (run_root / "SHA256SUMS.txt").read_text()
    assert "TEST_RESULT.json" in sums
    assert "TRAIN_LOG.jsonl" in sums
    remote = json.loads((run_root / "BACKUP_STATUS_post_test.json").read_text())
    remote_paths = {item["path"] for item in remote["files_verified"]}
    assert "TEST_RESULT.json" in remote_paths
    assert "SUPPLEMENTAL_EVALUATION_REPORT.json" in remote_paths
    assert "PROVENANCE.json" in remote_paths
    assert "SHA256SUMS.txt" in remote_paths
    assert any(path.endswith("TRAIN_LOG.jsonl") for path in remote_paths)
    assert calls_before_retry
