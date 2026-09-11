"""Comprehensive Wave 3 tests for UnifiedTrainer lifecycle and failure reporting.

Tests:
1. Subprocess injected checkpoint failure exits nonzero and writes failure.json.
2. Failures at training, validation, checkpoint, report, cache collection/save,
   calibration sweep/write/lineage, diagnostics, resume, and init stages.
3. Old report preservation on rejected output ownership.
4. W&B mocked cleanup on success, failure, interrupt, and disabled states.
5. Primary exception preserved when failure-writer and finish both throw secondary errors.
6. Per-epoch report survives later failure and preserves previous validation metrics.
7. KeyboardInterrupt handling: best-effort failure recording and clean propagation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from self_audit.artifact_io import atomic_write_json
from self_audit.models.self_audit_net import SelfAuditNet
from self_audit.training.unified_config import apply_overrides, load_unified_config, parse_unified_config
from self_audit.training.unified_trainer import UnifiedTrainer


class _SyntheticDataset(Dataset):
    def __init__(self, count: int = 4, size: int = 32) -> None:
        torch.manual_seed(42)
        self.images = torch.randn(count, 3, size, size)
        self.masks = torch.randint(0, 4, (count, size, size))
        self.case_ids = [f"case_{index // 2:03d}" for index in range(count)]

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.images[index],
            "mask": self.masks[index],
            "case_id": self.case_ids[index],
        }


def _make_tiny_net(seed: int = 42, shared_channels: int = 8, window_k: int = 2) -> SelfAuditNet:
    torch.manual_seed(seed)
    return SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=shared_channels,
        window_k=window_k,
        max_turns=2,
    ).eval()


def _get_base_config_dict() -> dict[str, Any]:
    with open("configs/self_audit_full.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _setup_tiny_trainer(tmp_path: Path, *, total_epochs: int = 2, max_turns: int = 1) -> UnifiedTrainer:
    cfg_dict = _get_base_config_dict()
    cfg_dict["model"]["pretrained_encoder"] = False
    cfg_dict["model"]["fallback"] = True
    cfg_dict["model"]["shared_channels"] = 8
    cfg_dict["model"]["window_k"] = 2
    cfg_dict["model"]["max_turns"] = max_turns
    cfg_dict["checkpoint"]["output_dir"] = str(tmp_path / "weights")
    cfg_dict["checkpoint"]["best_selection_min_epoch"] = 0
    cfg_dict["dataset"]["dataloader"]["num_workers"] = 0
    cfg_dict["logging"]["report_dir"] = str(tmp_path / "reports")
    cfg_dict["training"]["amp"]["enabled"] = False
    cfg_dict["training"]["schedule"]["total_epochs"] = total_epochs
    # Best selection is defined only inside the threshold-gated joint interval
    # and only on final_foreground_macro_dice, so this fixture uses the real
    # gated interval shape.  A bootstrap-only interval would exercise no
    # selection at all, which is not what these lifecycle tests are checking.
    cfg_dict["training"]["schedule"]["intervals"] = [
        {
            "start_epoch": 0,
            "end_epoch": total_epochs,
            "name": "joint_gated",
            "trainable": "all",
            "encoder_lr": 0.0001,
            "annotation_lr": 0.001,
            "auditor_lr": 0.001,
            "annotation_weight": 1.0,
            "audit_weight": 1.0,
            "objective": "retained_final_annotation",
            "transition_population": "active_attempted",
            "rollout": "threshold_gate",
            "batch_size": 2,
            "accumulation_steps": 1,
            "augment": False,
            "reset_optimizer": True,
        }
    ]
    cfg = parse_unified_config(cfg_dict)
    trainer = UnifiedTrainer(cfg, model=_make_tiny_net(99), device=torch.device("cpu"), disable_tqdm=True)

    ds = _SyntheticDataset(count=4)
    train_loader = DataLoader(ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(ds, batch_size=2, shuffle=False)
    for int_cfg in cfg.training.schedule.intervals:
        trainer._loader_cache[(int_cfg.batch_size, int_cfg.augment)] = (train_loader, val_loader)

    return trainer


# ============================================================================
# 1. Subprocess Injected Checkpoint Failure Exits Nonzero + failure.json
# ============================================================================


def test_subprocess_injected_checkpoint_failure_exits_nonzero_and_writes_failure_json(tmp_path: Path) -> None:
    """A real subprocess running train_self_audit.py with an injected checkpoint failure exits nonzero and writes failure.json."""
    import os
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vol_dir = data_dir / "volumes"
    msk_dir = data_dir / "masks"
    vol_dir.mkdir()
    msk_dir.mkdir()

    for cid in ["patient001", "patient002"]:
        np.save(vol_dir / f"{cid}.npy", np.random.randn(32, 32, 2).astype(np.float32))
        np.save(msk_dir / f"{cid}.npy", np.random.randint(0, 4, size=(32, 32, 2)).astype(np.int64))

    split_manifest = {"train": ["patient001"], "val": ["patient002"]}
    manifest_file = data_dir / "splits.json"
    manifest_file.write_text(json.dumps(split_manifest), encoding="utf-8")

    weights_dir = tmp_path / "weights"
    reports_dir = tmp_path / "reports"

    cfg_dict = _get_base_config_dict()
    cfg_dict["model"]["pretrained_encoder"] = False
    cfg_dict["model"]["fallback"] = True
    cfg_dict["model"]["shared_channels"] = 8
    cfg_dict["model"]["window_k"] = 2
    cfg_dict["model"]["max_turns"] = 1
    cfg_dict["dataset"]["data_root"] = str(data_dir)
    cfg_dict["dataset"]["split_manifest"] = str(manifest_file)
    cfg_dict["dataset"]["image_size"] = 32
    cfg_dict["dataset"]["depth_axis"] = 2
    cfg_dict["checkpoint"]["output_dir"] = str(weights_dir)
    cfg_dict["checkpoint"]["best_selection_min_epoch"] = 0
    cfg_dict["dataset"]["dataloader"]["num_workers"] = 0
    cfg_dict["logging"]["report_dir"] = str(reports_dir)
    cfg_dict["training"]["amp"]["enabled"] = False
    cfg_dict["training"]["schedule"]["total_epochs"] = 1
    cfg_dict["training"]["schedule"]["intervals"] = [
        {
            "start_epoch": 0,
            "end_epoch": 1,
            "name": "joint_gated",
            "trainable": "all",
            "encoder_lr": 0.0001,
            "annotation_lr": 0.001,
            "auditor_lr": 0.001,
            "annotation_weight": 1.0,
            "audit_weight": 1.0,
            "objective": "retained_final_annotation",
            "transition_population": "active_attempted",
            "rollout": "threshold_gate",
            "batch_size": 2,
            "accumulation_steps": 1,
            "augment": False,
            "reset_optimizer": True,
        }
    ]

    conf_path = tmp_path / "test_conf.yaml"
    conf_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    runner_script = tmp_path / "run_injected.py"
    runner_script.write_text(
        """
import sys
from pathlib import Path
import self_audit.training.unified_trainer as ut

orig_save_checkpoint = ut.save_checkpoint

def failing_save_checkpoint(*args, **kwargs):
    raise IOError("Injected checkpoint write failure: disk full")

ut.save_checkpoint = failing_save_checkpoint

if __name__ == "__main__":
    from scripts.train_self_audit import main
    main()
""",
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(runner_script),
        "--config",
        str(conf_path),
        "--device",
        "cpu",
        "--num_workers",
        "0",
        "--no_tqdm",
    ]

    env = {**os.environ, "PYTHONPATH": "src:."}
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    assert res.returncode != 0, f"Subprocess should have failed nonzero, got returncode 0.\\nStdout: {res.stdout}"
    assert "Injected checkpoint write failure" in res.stderr

    failure_file = reports_dir / "failure.json"
    assert failure_file.exists(), f"failure.json was not created at {failure_file}"
    failure_data = json.loads(failure_file.read_text(encoding="utf-8"))

    assert failure_data["schema_version"] == 1
    assert failure_data["completed"] is False
    assert failure_data["status"] == "failed"
    assert failure_data["failure_stage"] == "checkpoint"
    assert failure_data["exception_type"] in ("IOError", "OSError")
    assert "Injected checkpoint write failure" in failure_data["exception_message"]
    assert failure_data["current_epoch"] == 0
    assert failure_data["global_epoch"] == 0
    assert failure_data["completed_epochs"] == 0
    assert failure_data["run_id"] is not None
    assert failure_data["config_signature"] is not None
    assert failure_data["config_identity"] is not None
    assert failure_data["git_sha"] == failure_data.get("git_commit")


# ============================================================================
# 2. Granular Lifecycle Stage Failures
# ============================================================================


def test_failure_at_training_stage(tmp_path: Path) -> None:
    """Failure during train_epoch sets failure_stage='training' and completed=False."""
    trainer = _setup_tiny_trainer(tmp_path)
    reports_dir = tmp_path / "reports"

    def mock_train_epoch(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Injected gradient divergence during training")

    with patch.object(trainer, "train_epoch", side_effect=mock_train_epoch):
        with pytest.raises(RuntimeError, match="Injected gradient divergence"):
            trainer.train(max_steps=5)

    failure_file = reports_dir / "failure.json"
    assert failure_file.exists()
    failure_data = json.loads(failure_file.read_text(encoding="utf-8"))
    assert failure_data["failure_stage"] == "training"
    assert failure_data["completed"] is False
    assert failure_data["status"] == "failed"
    assert failure_data["exception_type"] == "RuntimeError"


def test_failure_at_validation_stage_never_fabricates_metrics(tmp_path: Path) -> None:
    """Failure during validate_epoch sets failure_stage='validation' and does not fabricate metrics."""
    trainer = _setup_tiny_trainer(tmp_path)
    reports_dir = tmp_path / "reports"

    def mock_validate_epoch(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("Injected validation dimension mismatch")

    with patch.object(trainer, "validate_epoch", side_effect=mock_validate_epoch):
        with pytest.raises(ValueError, match="Injected validation dimension mismatch"):
            trainer.train(max_steps=5)

    failure_file = reports_dir / "failure.json"
    assert failure_file.exists()
    failure_data = json.loads(failure_file.read_text(encoding="utf-8"))
    assert failure_data["failure_stage"] == "validation"
    assert failure_data["completed"] is False
    assert failure_data["last_completed_validation"] is None


def test_failure_at_report_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failure during pipeline_report.json writing sets failure_stage='report'."""
    trainer = _setup_tiny_trainer(tmp_path)
    reports_dir = tmp_path / "reports"

    import self_audit.training.unified_trainer as ut_mod
    real_atomic_write = ut_mod.atomic_write_json

    def mock_atomic_write(path: Any, payload: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path).name == "pipeline_report.json":
            raise OSError("Injected disk quota exceeded on report write")
        return real_atomic_write(path, payload, *args, **kwargs)

    monkeypatch.setattr(ut_mod, "atomic_write_json", mock_atomic_write)

    with pytest.raises(OSError, match="Injected disk quota exceeded"):
        trainer.train(max_steps=5)

    failure_file = reports_dir / "failure.json"
    assert failure_file.exists()
    failure_data = json.loads(failure_file.read_text(encoding="utf-8"))
    assert failure_data["failure_stage"] == "report"


def test_failures_inside_post_training_calibration_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Granular postprocessing stage failures: cache collection, sweep, write, lineage check, cache save, diagnostics."""
    stages = [
        ("cache_collection", "collect_validation_transition_cache", "collect error"),
        ("calibration_sweep", "sweep_thresholds", "sweep error"),
        ("calibration_write", "save_calibration", "write error"),
        ("calibration_lineage", "verify_calibration_lineage", "lineage error"),
        ("cache_save", "atomic_save_torch", "cache save error"),
        ("diagnostics", "evaluate_annotation_headroom", "diagnostics headroom error"),
    ]

    for stage_name, target_symbol, err_msg in stages:
        sub_dir = tmp_path / f"stage_{stage_name}"
        sub_dir.mkdir()
        trainer = _setup_tiny_trainer(sub_dir, total_epochs=1)
        trainer.completed = True
        trainer.written_best_path = sub_dir / "weights" / "best.pt"
        trainer.written_best_path.parent.mkdir(parents=True, exist_ok=True)

        from self_audit.training._utils import save_checkpoint
        save_checkpoint(
            trainer.written_best_path,
            trainer.model,
            epoch=1,
            global_step=1,
            optimizer_step=1,
            config=trainer.config.to_dict(),
            extra={
                "run_id": trainer.run_id,
                "config_signature": trainer.config_signature,
                "recipe_signature": trainer.config_signature,
                "source_signature": trainer.source_signature,
                "cohort_descriptor": trainer.get_cohort_descriptor(),
                "effective_cohort_descriptor": trainer.get_cohort_descriptor(),
                "resumable": True,
                "incomplete_epoch": False,
                "validation_complete": True,
                "completed_epoch": 1,
            },
        )

        with patch(f"self_audit.training.unified_trainer.{target_symbol}", side_effect=RuntimeError(err_msg)):
            with pytest.raises(RuntimeError, match=err_msg):
                trainer.run_post_training_calibration(sub_dir / "weights", sub_dir / "reports")

        assert trainer.failure_stage == stage_name


def test_failure_at_resume_stage(tmp_path: Path) -> None:
    """Failure during resume_from_checkpoint sets failure_stage='resume'."""
    trainer = _setup_tiny_trainer(tmp_path)
    corrupt_ckpt = tmp_path / "corrupt.pt"
    corrupt_ckpt.write_text("not a valid torch checkpoint", encoding="utf-8")

    with pytest.raises(Exception):
        trainer.resume_from_checkpoint(corrupt_ckpt)

    assert trainer.failure_stage == "resume"


def test_failure_at_init_stage(tmp_path: Path) -> None:
    """Pre-existing checkpoint or report without resume causes FileExistsError with failure_stage='init'."""
    trainer = _setup_tiny_trainer(tmp_path)
    rep_dir = tmp_path / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    stale_report = rep_dir / "pipeline_report.json"
    stale_report.write_text(json.dumps({"run_id": "old_run"}), encoding="utf-8")

    with pytest.raises(FileExistsError, match="Fresh training run refused"):
        trainer.train(start_epoch=0)

    assert trainer.failure_stage == "init"


# ============================================================================
# 3. Old Report Preservation on Rejected Output Ownership
# ============================================================================


def test_old_report_preservation_on_rejected_output_ownership(tmp_path: Path) -> None:
    """When a fresh run is rejected due to existing report artifacts, the old report is completely preserved."""
    rep_dir = tmp_path / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    report_file = rep_dir / "pipeline_report.json"

    original_report_payload = {
        "run_id": "run_historical_123",
        "completed": True,
        "epochs": [{"global_epoch": 1, "dice": 0.85}],
        "authoritative_evidence": "do_not_touch",
    }
    report_file.write_text(json.dumps(original_report_payload, indent=2), encoding="utf-8")
    original_bytes = report_file.read_bytes()

    trainer = _setup_tiny_trainer(tmp_path)

    with pytest.raises(FileExistsError, match="Fresh training run refused: existing report artifact found"):
        trainer.train(start_epoch=0)

    assert report_file.exists()
    assert report_file.read_bytes() == original_bytes
    reloaded = json.loads(report_file.read_text(encoding="utf-8"))
    assert reloaded["run_id"] == "run_historical_123"
    assert reloaded["completed"] is True
    assert reloaded["authoritative_evidence"] == "do_not_touch"


# ============================================================================
# 4. W&B Telemetry Mocked Cleanup (Success, Failure, Interrupt, Disabled)
# ============================================================================


def test_wandb_telemetry_cleanup_on_success(tmp_path: Path) -> None:
    """Successful run sets summary completed=True, status='succeeded', and finish(exit_code=0)."""
    trainer = _setup_tiny_trainer(tmp_path, total_epochs=1)
    trainer.logger.set_summary = MagicMock()
    trainer.logger.finish = MagicMock()

    trainer.config = apply_overrides(trainer.config, {"skip_calibration": True})

    report = trainer.train()
    assert report["completed"] is True

    trainer.logger.set_summary.assert_called()
    last_summary = trainer.logger.set_summary.call_args[0][0]
    assert last_summary["completed"] is True
    assert last_summary["status"] == "succeeded"

    trainer.logger.finish.assert_called_with(exit_code=0)


def test_wandb_telemetry_cleanup_on_failure(tmp_path: Path) -> None:
    """Failing run sets summary completed=False, status='failed', and finish(exit_code=1)."""
    trainer = _setup_tiny_trainer(tmp_path, total_epochs=1)
    trainer.logger.set_summary = MagicMock()
    trainer.logger.finish = MagicMock()

    with patch.object(trainer, "train_epoch", side_effect=RuntimeError("Training failed")):
        with pytest.raises(RuntimeError):
            trainer.train()

    trainer.logger.set_summary.assert_called()
    last_summary = trainer.logger.set_summary.call_args[0][0]
    assert last_summary["completed"] is False
    assert last_summary["status"] == "failed"
    assert last_summary["failure_stage"] == "training"

    trainer.logger.finish.assert_called_with(exit_code=1)


def test_wandb_telemetry_cleanup_on_keyboard_interrupt(tmp_path: Path) -> None:
    """KeyboardInterrupt sets summary completed=False, status='interrupted', and finish(exit_code=130)."""
    trainer = _setup_tiny_trainer(tmp_path, total_epochs=1)
    trainer.logger.set_summary = MagicMock()
    trainer.logger.finish = MagicMock()

    with patch.object(trainer, "train_epoch", side_effect=KeyboardInterrupt()):
        with pytest.raises(KeyboardInterrupt):
            trainer.train()

    trainer.logger.set_summary.assert_called()
    last_summary = trainer.logger.set_summary.call_args[0][0]
    assert last_summary["completed"] is False
    assert last_summary["status"] == "interrupted"

    trainer.logger.finish.assert_called_with(exit_code=130)

    failure_file = tmp_path / "reports" / "failure.json"
    assert failure_file.exists()
    failure_data = json.loads(failure_file.read_text(encoding="utf-8"))
    assert failure_data["status"] == "interrupted"
    assert failure_data["failure_stage"] == "interrupted"


def test_disabled_wandb_works_cleanly(tmp_path: Path) -> None:
    """Disabled W&B executes all logging and finish calls without raising."""
    trainer = _setup_tiny_trainer(tmp_path, total_epochs=1)
    trainer.config = apply_overrides(trainer.config, {"skip_calibration": True, "wandb": False})
    trainer.logger.enabled = False

    report = trainer.train()
    assert report["completed"] is True
    assert report["telemetry"]["enabled"] is False
    assert report["telemetry"]["telemetry_status"] == "disabled"


# ============================================================================
# 5. Original Exception Preserved When Failure Writer & Finish Both Fail
# ============================================================================


def test_original_exception_preserved_if_failure_writer_and_finish_both_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Primary exception must propagate even if writing failure.json and logger.finish both raise secondary errors."""
    trainer = _setup_tiny_trainer(tmp_path, total_epochs=1)

    import self_audit.training.unified_trainer as ut_mod

    def mock_atomic_write(path: Any, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Secondary error: read-only filesystem")

    def mock_finish(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionResetError("Secondary error: wandb connection reset")

    monkeypatch.setattr(ut_mod, "atomic_write_json", mock_atomic_write)
    monkeypatch.setattr(trainer.logger, "finish", mock_finish)

    with patch.object(trainer, "train_epoch", side_effect=RuntimeError("Primary mathematical divergence")):
        with pytest.raises(RuntimeError, match="Primary mathematical divergence") as exc_info:
            trainer.train()

    assert isinstance(exc_info.value, RuntimeError)
    assert "Primary mathematical divergence" in str(exc_info.value)


# ============================================================================
# 6. Per-Epoch Report Survives Later Failure
# ============================================================================


def test_per_epoch_report_survives_later_failure(tmp_path: Path) -> None:
    """Epoch 1 progress report survives when Epoch 2 crashes, retaining epoch 1 validation metrics."""
    trainer = _setup_tiny_trainer(tmp_path, total_epochs=2)
    reports_dir = tmp_path / "reports"

    real_train_epoch = trainer.train_epoch

    def crashing_train_epoch(interval: Any, loader: Any, **kwargs: Any) -> Any:
        if trainer.global_epoch == 1:
            raise RuntimeError("Crashed on epoch 2 training")
        return real_train_epoch(interval, loader, **kwargs)

    with patch.object(trainer, "train_epoch", side_effect=crashing_train_epoch):
        with pytest.raises(RuntimeError, match="Crashed on epoch 2 training"):
            trainer.train()

    # 1. Check pipeline_report.json exists and has Epoch 1
    progress_report = reports_dir / "pipeline_report.json"
    assert progress_report.exists()
    rep_data = json.loads(progress_report.read_text(encoding="utf-8"))
    assert len(rep_data["epochs"]) == 1
    assert rep_data["epochs"][0]["global_epoch"] == 1
    assert rep_data["completed"] is False
    assert rep_data["incomplete_reason"] == "failed_at_training"

    # 2. Check failure.json has exact completed counts and validation from Epoch 1
    failure_file = reports_dir / "failure.json"
    assert failure_file.exists()
    failure_data = json.loads(failure_file.read_text(encoding="utf-8"))
    assert failure_data["completed_epochs"] == 1
    assert failure_data["current_epoch"] == 1
    assert failure_data["last_checkpoint_committed_epoch"] == 1
    assert failure_data["last_completed_validation"] is not None
    assert "primary_metric" in failure_data["last_completed_validation"]


# ============================================================================
# 7. Regressions: summary-skips-finish & record_failure metadata masking
# ============================================================================


def test_summary_failure_does_not_skip_finish(tmp_path: Path) -> None:
    """When logger.set_summary throws, logger.finish must still be called with correct exit code."""
    trainer = _setup_tiny_trainer(tmp_path, total_epochs=1)
    trainer.logger.set_summary = MagicMock(side_effect=RuntimeError("W&B set_summary broken pipe"))
    trainer.logger.finish = MagicMock()

    with patch.object(trainer, "train_epoch", side_effect=RuntimeError("Primary failure")):
        with pytest.raises(RuntimeError, match="Primary failure"):
            trainer.train()

    # Even though set_summary threw, finish MUST be called with exit_code=1
    trainer.logger.finish.assert_called_with(exit_code=1)


def test_metadata_extraction_failure_does_not_mask_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When metadata extraction inside record_failure throws, primary exception must not be masked."""
    trainer = _setup_tiny_trainer(tmp_path, total_epochs=1)

    # Invalidate metadata extraction properties
    def exploding_property(self: Any) -> Any:
        raise RuntimeError("Fatal explosion calculating signature metadata")

    monkeypatch.setattr(UnifiedTrainer, "config_signature", property(exploding_property))
    monkeypatch.setattr(UnifiedTrainer, "telemetry_summary", property(exploding_property))
    monkeypatch.setattr(UnifiedTrainer, "git_commit", property(exploding_property))

    with patch.object(trainer, "train_epoch", side_effect=RuntimeError("Primary core computation failure")):
        with pytest.raises(RuntimeError, match="Primary core computation failure") as exc_info:
            trainer.train()

    # The raised exception must be the primary exception, NOT the metadata explosion
    assert "Primary core computation failure" in str(exc_info.value)
    assert "Fatal explosion" not in str(exc_info.value)


def test_metadata_preparation_failure_still_finishes(tmp_path: Path) -> None:
    """When metadata preparation inside record_failure throws (e.g. str(exc) raises), logger.finish must still execute in finally."""
    trainer = _setup_tiny_trainer(tmp_path, total_epochs=1)
    trainer.logger.finish = MagicMock()

    class UnstringableError(RuntimeError):
        def __str__(self) -> str:
            raise TypeError("Cannot stringify this error")

    with patch.object(trainer, "train_epoch", side_effect=UnstringableError("Primary unstringable")):
        with pytest.raises(UnstringableError):
            trainer.train()

    # Logger finish must still have been called with exit_code=1 in finally block!
    trainer.logger.finish.assert_called_with(exit_code=1)


def test_ownership_refusal_preserves_old_failure_json(tmp_path: Path) -> None:
    """When output ownership is refused on a fresh run, pre-existing failure.json is preserved unmodified."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    canonical_failure = reports_dir / "failure.json"
    original_bytes = b'{"old_run": "precious_data"}'
    canonical_failure.write_bytes(original_bytes)

    # Pre-existing report causes FileExistsError on fresh run
    existing_report = reports_dir / "pipeline_report.json"
    existing_report.write_text('{"run": "old"}', encoding="utf-8")

    trainer = _setup_tiny_trainer(tmp_path, total_epochs=1)
    with pytest.raises(FileExistsError, match="Fresh training run refused"):
        trainer.train()

    # 1. Canonical failure.json bytes must be untouched!
    assert canonical_failure.read_bytes() == original_bytes

    # 2. A uniquely named attempt artifact was written
    attempt_artifacts = list(reports_dir.glob("failure_attempt_*.json"))
    assert len(attempt_artifacts) == 1
    attempt_data = json.loads(attempt_artifacts[0].read_text(encoding="utf-8"))
    assert attempt_data["status"] == "failed"
    assert attempt_data["exception_type"] == "FileExistsError"


def test_logger_failure_after_durable_report_has_correct_committed_counters(tmp_path: Path) -> None:
    """When logger.log throws after checkpoint and report commit, failure.json and report have correct completed_epochs."""
    trainer = _setup_tiny_trainer(tmp_path, total_epochs=2)
    reports_dir = tmp_path / "reports"

    trainer.logger.log = MagicMock(side_effect=RuntimeError("W&B logger network drop"))

    with pytest.raises(RuntimeError, match="W&B logger network drop"):
        trainer.train()

    # Injected error during logging stage on epoch 0: checkpoint & report commits already succeeded,
    # so completed_epochs was incremented to 1 before calling logger.log!
    failure_file = reports_dir / "failure.json"
    assert failure_file.exists()
    failure_data = json.loads(failure_file.read_text(encoding="utf-8"))
    assert failure_data["failure_stage"] == "logging"
    assert failure_data["completed_epochs"] == 1
    assert failure_data["last_checkpoint_committed_epoch"] == 1

    # Pipeline report also reflects committed epoch 1
    report_file = reports_dir / "pipeline_report.json"
    assert report_file.exists()
    report_data = json.loads(report_file.read_text(encoding="utf-8"))
    assert report_data["completed_epochs"] == 1
    assert report_data["last_checkpoint_committed_epoch"] == 1


def test_final_telemetry_failure_report_truthful(tmp_path: Path) -> None:
    """When logger.finish fails, telemetry degradation is captured in the report and pipeline_report.json."""
    trainer = _setup_tiny_trainer(tmp_path, total_epochs=1)
    trainer.logger.enabled = True
    reports_dir = tmp_path / "reports"

    # Mock logger.finish to fail
    def failing_finish(exit_code: int | None = None) -> None:
        trainer.logger.telemetry_errors.append(f"finish: mock finish failed with exit_code {exit_code}")
        trainer.logger.failed_log_count += 1
        raise ConnectionError("W&B final sync timed out")

    trainer.logger.finish = failing_finish

    # Complete run succeeds without raising (telemetry errors are nonfatal)
    report = trainer.train()
    assert report["completed"] is True
    assert report["status"] == "succeeded"

    # But telemetry status must truthfully reflect degraded/error state captured AFTER finish!
    assert report["telemetry"]["telemetry_status"] == "degraded"
    assert report["telemetry"]["failed_log_count"] > 0
    assert any("finish: mock finish failed" in e for e in report["telemetry"]["telemetry_errors"])

    # Durable pipeline_report.json on disk must also have the truthful degraded telemetry!
    saved_report = json.loads((reports_dir / "pipeline_report.json").read_text(encoding="utf-8"))
    assert saved_report["telemetry"]["telemetry_status"] == "degraded"


def test_diagnostics_when_calibration_disabled(tmp_path: Path) -> None:
    """When calibration is disabled, requested diagnostics execute with configured fixed tau and report status."""
    cfg_dict = _get_base_config_dict()
    cfg_dict["model"]["pretrained_encoder"] = False
    cfg_dict["model"]["fallback"] = True
    cfg_dict["model"]["shared_channels"] = 8
    cfg_dict["model"]["window_k"] = 2
    cfg_dict["model"]["max_turns"] = 1
    cfg_dict["checkpoint"]["output_dir"] = str(tmp_path / "weights")
    cfg_dict["checkpoint"]["best_selection_min_epoch"] = 0
    cfg_dict["dataset"]["dataloader"]["num_workers"] = 0
    cfg_dict["logging"]["report_dir"] = str(tmp_path / "reports")
    cfg_dict["training"]["amp"]["enabled"] = False
    cfg_dict["training"]["schedule"]["total_epochs"] = 1
    cfg_dict["training"]["schedule"]["intervals"] = [
        {
            "start_epoch": 0,
            "end_epoch": 1,
            "name": "joint_gated",
            "trainable": "all",
            "encoder_lr": 0.0001,
            "annotation_lr": 0.001,
            "auditor_lr": 0.001,
            "annotation_weight": 1.0,
            "audit_weight": 1.0,
            "objective": "retained_final_annotation",
            "transition_population": "active_attempted",
            "rollout": "threshold_gate",
            "batch_size": 2,
            "accumulation_steps": 1,
            "augment": False,
            "reset_optimizer": True,
        }
    ]
    cfg_dict["calibration"]["enabled"] = False
    cfg_dict["diagnostics"]["evaluate_headroom"] = True
    cfg_dict["diagnostics"]["evaluate_decomposition"] = True

    cfg = parse_unified_config(cfg_dict)
    trainer = UnifiedTrainer(cfg, model=_make_tiny_net(99), device=torch.device("cpu"), disable_tqdm=True)

    ds = _SyntheticDataset(count=4)
    train_loader = DataLoader(ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(ds, batch_size=2, shuffle=False)
    for int_cfg in cfg.training.schedule.intervals:
        trainer._loader_cache[(int_cfg.batch_size, int_cfg.augment)] = (train_loader, val_loader)

    with patch.object(trainer, "run_independent_diagnostics") as mock_diag:
        report = trainer.train()
        assert report["completed"] is True
        assert report["calibration"]["status"] == "skipped"
        assert report["calibration"]["reason"] == "calibration_disabled"
        mock_diag.assert_called_once()


def test_final_report_failure_prevents_success_finish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the initial final report write fails, finish(exit_code=0) is NEVER called, and failure handler finishes with exit_code=1."""
    trainer = _setup_tiny_trainer(tmp_path, total_epochs=1)
    trainer.config = apply_overrides(trainer.config, {"skip_calibration": True})
    finish_calls: list[int | None] = []

    def recording_finish(exit_code: int | None = None) -> None:
        finish_calls.append(exit_code)

    trainer.logger.finish = recording_finish

    import self_audit.training.unified_trainer as ut_mod
    orig_atomic_write = ut_mod.atomic_write_json

    def failing_atomic_write(path: Any, payload: Any, *args: Any, **kwargs: Any) -> Any:
        # Fail specifically when writing pipeline_report.json
        if str(path).endswith("pipeline_report.json"):
            raise OSError("Disk quota exceeded while writing final pipeline report")
        return orig_atomic_write(path, payload, *args, **kwargs)

    monkeypatch.setattr(ut_mod, "atomic_write_json", failing_atomic_write)

    with pytest.raises(OSError, match="Disk quota exceeded"):
        trainer.train()

    # finish(exit_code=0) was NEVER called because report write failed before success finalization!
    assert 0 not in finish_calls
    # Failure handler must have finalized with exit_code=1
    assert 1 in finish_calls

    # Failure artifact was written with failure_stage='report'
    failure_file = tmp_path / "reports" / "failure.json"
    assert failure_file.exists()
    failure_data = json.loads(failure_file.read_text(encoding="utf-8"))
    assert failure_data["failure_stage"] == "report"
    assert failure_data["status"] == "failed"


def test_pre_init_failure_preserves_canonical_failure_on_resume_value_error(tmp_path: Path) -> None:
    """Pre-existing failure.json is preserved unmodified on resume ValueError in _write_pre_init_failure."""
    from scripts.train_self_audit import _write_pre_init_failure

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    canonical_failure = reports_dir / "failure.json"
    old_data = b'{"preexisting_run_id": "old-unrelated-run", "precious": true}'
    canonical_failure.write_bytes(old_data)

    exc = ValueError("Resume checkpoint /path/to/missing.pt does not exist")
    _write_pre_init_failure(reports_dir, exc, stage="resume", status="failed")

    # 1. Canonical failure.json must still have the old data!
    assert canonical_failure.read_bytes() == old_data

    # 2. A uniquely named attempt file was written
    attempts = list(reports_dir.glob("failure_attempt_*.json"))
    assert len(attempts) == 1
    attempt_data = json.loads(attempts[0].read_text(encoding="utf-8"))
    assert attempt_data["failure_stage"] == "resume"
    assert attempt_data["exception_type"] == "ValueError"
    assert "Resume checkpoint" in attempt_data["exception_message"]


def test_record_failure_preserves_canonical_failure_when_file_exists(tmp_path: Path) -> None:
    """UnifiedTrainer.record_failure unconditionally preserves existing failure.json and writes unique attempt file."""
    trainer = _setup_tiny_trainer(tmp_path, total_epochs=1)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    canonical_failure = reports_dir / "failure.json"
    old_data = b'{"preexisting_run_id": "old-unrelated-run"}'
    canonical_failure.write_bytes(old_data)

    path = trainer.record_failure(RuntimeError("Runtime explosion"), failure_stage="train")

    assert canonical_failure.read_bytes() == old_data
    assert path is not None
    assert path != canonical_failure
    assert path.name.startswith("failure_attempt_")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["exception_type"] == "RuntimeError"
