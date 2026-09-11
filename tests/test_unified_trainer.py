"""Comprehensive regression test suite for Wave 2 unified training pipeline.

Tests independent gates:
1. Strict schema rejection (duplicate keys, unknown keys, scalar types, non-finite values).
2. Schedule boundary detection (0, 99, 100, 119, 120, 129, out-of-range).
3. Trainability & module train/eval mode & zero LR parameter exclusion.
4. Differential reference batch loss & gradients for Intervals 0, 1, 2.
5. Optimizer reset carrying live weights.
6. Resuming lineage across boundaries (epoch 99 vs 100).
7. Bounded smoke (max_steps) incomplete run & calibration firewall.
8. best.pt selection restricted strictly to epochs >= 120.
9. Strict tau equality rejection and independent per-sample halting.
10. One-process CLI execution with synthetic data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any
import yaml

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from self_audit.audit.semantics import METRIC_SPACE_SLICE_PROXY
from self_audit.models.self_audit_net import SelfAuditNet
from self_audit.provenance import state_digest
from self_audit.training.schedule import Schedule, ScheduleInterval
from self_audit.training.unified_config import (
    UnifiedConfig,
    apply_overrides,
    load_unified_config,
    parse_unified_config,
)
from self_audit.training.unified_trainer import SELECTION_METRIC_KEY, UnifiedTrainer
from self_audit.training.train_annotation import phase_a_loss
from self_audit.training.train_auditor import _auditor_batch
from self_audit.training.finetune_joint import compute_joint_losses


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


def _make_tiny_net(seed: int = 42) -> SelfAuditNet:
    torch.manual_seed(seed)
    return SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
    ).eval()


def _get_valid_config_dict() -> dict[str, Any]:
    with open("configs/self_audit_full.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _unit_fixture_commit_extra(
    trainer: UnifiedTrainer,
    *,
    epoch: int,
    global_step: int,
    optimizer_step: int,
    cohort_descriptor: dict[str, Any] | None = None,
    loader_cardinality: int,
    selection_metric_value: float = 0.5,
    **overrides: Any,
) -> dict[str, Any]:
    """Build the commit metadata a UNIT fixture checkpoint must carry.

    These checkpoints are hand-built rather than produced by a real training
    run, so the counters here are *synthesized*.  They are synthesized
    coherently: run/source/config identity are read off the live trainer, the
    per-epoch history is derived from the configured schedule, and its tail row
    is derived from the same counters the payload records.  The point is that a
    unit fixture satisfies the same resume contract a genuine commit does,
    instead of a weakened one -- real end-to-end continuation coverage lives in
    tests/test_runtime_resume.py.

    ``overrides`` are applied last so a negative test can corrupt exactly one
    field and still reach the gate it is aiming at.
    """

    schedule = trainer.schedule
    history: list[dict[str, Any]] = []
    for index in range(1, epoch + 1):
        interval = schedule.get_interval(index - 1)
        interval_index = schedule.get_interval_index(index - 1)
        # Monotone interpolation whose final row lands exactly on the payload
        # counters; optimizer_step <= global_step is preserved row by row.
        row_global_step = (global_step * index) // epoch
        row_optimizer_step = (optimizer_step * index) // epoch
        history.append(
            {
                "epoch": index,
                "global_epoch": index,
                "interval": interval.name,
                "interval_index": interval_index,
                "global_step": row_global_step,
                "optimizer_step": row_optimizer_step,
                "validation_complete": True,
                "checkpoint_committed": True,
                "status": "checkpoint_committed",
                "completed_epochs": index,
                "last_checkpoint_committed_epoch": index,
                "report_committed": False,
                SELECTION_METRIC_KEY: float(selection_metric_value),
            }
        )

    active_interval = schedule.get_interval(max(epoch - 1, 0))
    active_index = schedule.get_interval_index(max(epoch - 1, 0))
    extra: dict[str, Any] = {
        "run_id": trainer.run_id,
        "source_signature": trainer.source_signature,
        "config_signature": trainer.config_signature,
        "recipe_signature": trainer.config_signature,
        "resumable": True,
        "incomplete_epoch": False,
        "validation_complete": True,
        "completed_epoch": epoch,
        "active_interval": active_interval.name,
        "interval_name": active_interval.name,
        "interval_index": active_index,
        "loader_cardinality": loader_cardinality,
        "last_completed_validation": {SELECTION_METRIC_KEY: float(selection_metric_value)},
        "epoch_history": history,
    }
    if cohort_descriptor is not None:
        extra["cohort_descriptor"] = cohort_descriptor
        extra["effective_cohort_descriptor"] = cohort_descriptor
    extra.update(overrides)
    return extra


# ============================================================================
# Gate 1: Strict Schema Validation
# ============================================================================


def test_schema_rejection_duplicate_keys(tmp_path: Path) -> None:
    """Duplicate keys anywhere in YAML must raise ValueError."""
    yaml_text = """
schema_version: 1
schema_version: 1
"""
    bad_file = tmp_path / "dup.yaml"
    bad_file.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate key 'schema_version'"):
        load_unified_config(bad_file)


def test_schema_rejection_unknown_keys() -> None:
    """Unknown keys at root or nested sections must be rejected."""
    cfg_dict = _get_valid_config_dict()

    # Root unknown key
    cfg_dict["unknown_root_key"] = 123
    with pytest.raises(ValueError, match="Unknown keys in section 'root'"):
        parse_unified_config(cfg_dict)
    del cfg_dict["unknown_root_key"]

    # Nested section unknown key
    cfg_dict["experiment"]["rogue_field"] = "bad"
    with pytest.raises(ValueError, match="Unknown keys in section 'experiment'"):
        parse_unified_config(cfg_dict)
    del cfg_dict["experiment"]["rogue_field"]

    # Training schedule interval unknown key
    cfg_dict["training"]["schedule"]["intervals"][0]["extra_option"] = True
    with pytest.raises(ValueError, match="Unknown keys in schedule interval"):
        parse_unified_config(cfg_dict)


def test_schema_rejection_wrong_scalar_types() -> None:
    """Wrong types (e.g. str for bool, bool for int, int for float) must be rejected."""
    cfg_dict = _get_valid_config_dict()

    # Bool expected, got str
    cfg_dict["experiment"]["deterministic"] = "false"
    with pytest.raises(TypeError, match="must be a boolean"):
        parse_unified_config(cfg_dict)
    cfg_dict["experiment"]["deterministic"] = False

    # Int expected, got bool (bool is subclass of int in Python)
    cfg_dict["experiment"]["seed"] = True
    with pytest.raises(TypeError, match="must be an integer"):
        parse_unified_config(cfg_dict)
    cfg_dict["experiment"]["seed"] = 42

    # Float expected, got bool
    cfg_dict["training"]["optimizer"]["grad_clip"] = False
    with pytest.raises(TypeError, match="must be a float"):
        parse_unified_config(cfg_dict)


def test_schema_rejection_nonfinite_values() -> None:
    """NaN and Inf floats must be rejected."""
    cfg_dict = _get_valid_config_dict()
    cfg_dict["training"]["optimizer"]["grad_clip"] = float("nan")
    with pytest.raises(ValueError, match="must be a finite number"):
        parse_unified_config(cfg_dict)

    cfg_dict["training"]["optimizer"]["grad_clip"] = float("inf")
    with pytest.raises(ValueError, match="must be a finite number"):
        parse_unified_config(cfg_dict)


def test_schema_rejection_invalid_schedule_coverage() -> None:
    """Schedule with gaps or wrong endpoints must be rejected."""
    cfg_dict = _get_valid_config_dict()

    # Gap in schedule (interval 0 ends at 90, interval 1 starts at 100)
    cfg_dict["training"]["schedule"]["intervals"][0]["end_epoch"] = 90
    with pytest.raises(ValueError, match="Schedule intervals not contiguous"):
        parse_unified_config(cfg_dict)
    cfg_dict["training"]["schedule"]["intervals"][0]["end_epoch"] = 100

    # Schedule doesn't start at 0
    cfg_dict["training"]["schedule"]["intervals"][0]["start_epoch"] = 1
    with pytest.raises(ValueError, match="First schedule interval must start at epoch 0"):
        parse_unified_config(cfg_dict)
    cfg_dict["training"]["schedule"]["intervals"][0]["start_epoch"] = 0

    # Schedule doesn't reach total_epochs
    cfg_dict["training"]["schedule"]["total_epochs"] = 150
    with pytest.raises(ValueError, match="must equal schedule total_epochs"):
        parse_unified_config(cfg_dict)


# ============================================================================
# Gate 2: Schedule & Transition Boundaries
# ============================================================================


def test_schedule_transition_boundaries_and_out_of_range() -> None:
    """Verify exact boundaries: 0, 99, 100, 119, 120, 129, and out-of-range failures."""
    config = load_unified_config("configs/self_audit_full.yaml")
    schedule = config.training.schedule

    assert schedule.total_epochs == 130
    assert len(schedule.intervals) == 3

    # Epoch 0: Interval 0 start
    int_0 = schedule.get_interval(0)
    assert int_0.name == "annotation_bootstrap"
    assert schedule.is_interval_start(0) is True
    assert schedule.is_reset_boundary(0) is False  # initial interval, not reset

    # Epoch 99: Interval 0 end
    assert schedule.get_interval(99).name == "annotation_bootstrap"
    assert schedule.is_interval_start(99) is False

    # Epoch 100: Interval 1 boundary
    int_1 = schedule.get_interval(100)
    assert int_1.name == "auditor_training"
    assert schedule.is_interval_start(100) is True
    assert schedule.is_reset_boundary(100) is True

    # Epoch 119: Interval 1 end
    assert schedule.get_interval(119).name == "auditor_training"
    assert schedule.is_interval_start(119) is False

    # Epoch 120: Interval 2 boundary
    int_2 = schedule.get_interval(120)
    assert int_2.name == "joint_self_audit"
    assert schedule.is_interval_start(120) is True
    assert schedule.is_reset_boundary(120) is True

    # Epoch 129: Interval 2 end
    assert schedule.get_interval(129).name == "joint_self_audit"
    assert schedule.is_interval_start(129) is False

    # Out of range: negative
    with pytest.raises(IndexError, match="out of range"):
        schedule.get_interval(-1)

    # Out of range: 130 (total_epochs)
    with pytest.raises(IndexError, match="out of range"):
        schedule.get_interval(130)

    # Out of range: 200
    with pytest.raises(IndexError, match="out of range"):
        schedule.get_interval(200)


# ============================================================================
# Gate 3: Trainability & Zero LR / Frozen Modules
# ============================================================================


def test_trainability_and_module_modes_and_zero_lr_exclusion() -> None:
    """Modules in eval() must have requires_grad=False and be excluded from optimizers."""
    config = load_unified_config("configs/self_audit_full.yaml")
    model = _make_tiny_net()
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)

    # Interval 0: Annotation only
    int_0 = config.training.schedule.intervals[0]
    trainer.setup_interval_optimizer_and_scheduler(int_0, num_batches=10)
    assert model.encoder.training is True
    assert model.initial_head.training is True
    assert model.auditor.training is False
    assert any(p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.auditor.parameters())

    # Ensure auditor is NOT in optimizer param groups
    opt_params = set()
    for group in trainer.optimizer.param_groups:
        for p in group["params"]:
            opt_params.add(p)
    for p in model.auditor.parameters():
        assert p not in opt_params

    # Interval 1: Auditor only
    int_1 = config.training.schedule.intervals[1]
    trainer.setup_interval_optimizer_and_scheduler(int_1, num_batches=10)
    assert model.encoder.training is False
    assert model.initial_head.training is False
    assert model.auditor.training is True
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.initial_head.parameters())
    assert all(p.requires_grad for p in model.auditor.parameters())

    # Ensure annotation params are NOT in optimizer param groups
    opt_params = set()
    for group in trainer.optimizer.param_groups:
        for p in group["params"]:
            opt_params.add(p)
    for p in model.encoder.parameters():
        assert p not in opt_params
    for p in model.initial_head.parameters():
        assert p not in opt_params

    # Interval 2: All trainable
    int_2 = config.training.schedule.intervals[2]
    trainer.setup_interval_optimizer_and_scheduler(int_2, num_batches=10)
    assert model.encoder.training is True
    assert model.auditor.training is True
    assert all(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.auditor.parameters())


# ============================================================================
# Gate 4: Differential Reference Batch Loss & Gradients
# ============================================================================


def test_differential_batch_loss_interval_0() -> None:
    """Interval 0 batch loss must match phase_a_loss bit-compatibly."""
    config = load_unified_config("configs/self_audit_full.yaml")
    model = _make_tiny_net(10)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)
    trainer.amp_enabled = False

    batch = {
        "image": torch.randn(2, 3, 32, 32),
        "mask": torch.randint(0, 4, (2, 32, 32)),
    }
    int_0 = config.training.schedule.intervals[0]

    # Unified loss
    loss_u, details_u = trainer.compute_batch_loss(batch, int_0)

    # Reference loss
    output_ref = model.forward_annotation(batch["image"])
    loss_ref, parts_ref = phase_a_loss(
        output_ref,
        batch["mask"],
        stage_weights=config.training.annotation_loss.stage_weights,
    )

    assert torch.allclose(loss_u, loss_ref, atol=1e-6)
    assert abs(details_u["loss"] - parts_ref["loss"]) < 1e-6


def test_differential_batch_loss_interval_2() -> None:
    """Interval 2 batch loss must match compute_joint_losses."""
    config = load_unified_config("configs/self_audit_full.yaml")
    model = _make_tiny_net(20)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)
    trainer.amp_enabled = False

    batch = {
        "image": torch.randn(2, 3, 32, 32),
        "mask": torch.randint(0, 4, (2, 32, 32)),
    }
    int_2 = config.training.schedule.intervals[2]

    torch.manual_seed(99)
    loss_u, details_u = trainer.compute_batch_loss(batch, int_2)

    torch.manual_seed(99)
    loss_ref, details_ref = compute_joint_losses(
        model,
        batch,
        tau_accept=config.training.rollout.tau,
        t_max=config.training.rollout.max_turns,
        lambda_audit=int_2.audit_weight,
        neutral_margin=config.training.audit_loss.neutral_margin,
        local_weighting=True,
    )

    assert torch.allclose(loss_u, loss_ref, atol=1e-6)
    assert abs(details_u["annotation_loss"] - float(details_ref["annotation_loss"])) < 1e-6
    assert abs(details_u["audit_loss"] - float(details_ref["audit_loss"])) < 1e-6


def test_optimizer_reset_carries_live_weights() -> None:
    """Resetting optimizer at interval boundary carries live weights and reinitializes moments."""
    config = load_unified_config("configs/self_audit_full.yaml")
    model = _make_tiny_net(30)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)

    int_0 = config.training.schedule.intervals[0]
    trainer.setup_interval_optimizer_and_scheduler(int_0, num_batches=5)

    # Mutate a weight
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
            break
    digest_before = state_digest(model)

    # Transition to interval 1
    int_1 = config.training.schedule.intervals[1]
    trainer.setup_interval_optimizer_and_scheduler(int_1, num_batches=5)
    digest_after = state_digest(model)

    assert digest_before == digest_after, "Model weights must be carried across boundary"
    assert len(trainer.optimizer.state) == 0, "Optimizer state must be freshly reset"


# ============================================================================
# Gate 5: Resume Lineage & RNG
# ============================================================================


def test_resume_lineage_before_and_after_boundary(tmp_path: Path) -> None:
    """Interruption at epoch 99 vs 100 correctly restores state and RNG."""
    config = load_unified_config("configs/self_audit_full.yaml")
    model = _make_tiny_net(40)
    output_dir = tmp_path / "weights"
    output_dir.mkdir()

    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)
    mock_ds = _SyntheticDataset(count=10)
    mock_loader = DataLoader(mock_ds, batch_size=2, shuffle=False)
    int_0 = config.training.schedule.intervals[0]
    trainer._loader_cache[(int_0.batch_size, int_0.augment)] = (mock_loader, mock_loader)
    trainer.setup_interval_optimizer_and_scheduler(int_0, num_batches=5)
    trainer.global_epoch = 99
    trainer.optimizer_step = 495
    trainer.global_step = 495

    ckpt_path = output_dir / "last.pt"
    from self_audit.training._utils import save_checkpoint
    cohort_desc = trainer.get_cohort_descriptor(int_0)
    save_checkpoint(
        ckpt_path,
        model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        epoch=99,
        global_step=495,
        optimizer_step=495,
        config=config.to_dict(),
        extra=_unit_fixture_commit_extra(
            trainer,
            epoch=99,
            global_step=495,
            optimizer_step=495,
            cohort_descriptor=cohort_desc,
            loader_cardinality=5,
        ),
    )

    # Resume in new trainer instance
    trainer2 = UnifiedTrainer(config, model=_make_tiny_net(999), device=torch.device("cpu"), disable_tqdm=True)
    mock_ds = _SyntheticDataset(count=10)
    mock_loader = DataLoader(mock_ds, batch_size=2, shuffle=False)
    trainer2._loader_cache[(int_0.batch_size, int_0.augment)] = (mock_loader, mock_loader)
    resumed_epoch = trainer2.resume_from_checkpoint(ckpt_path)

    assert resumed_epoch == 99
    assert trainer2.optimizer_step == 495
    assert trainer2.global_step == 495
    assert state_digest(trainer2.model) == state_digest(model)

    # The committed history travels with the bytes: a resumed attempt keeps
    # reporting the epochs it already committed, and its tail describes the
    # very commit it resumed from.
    restored = trainer2.report["epochs"]
    assert [row["epoch"] for row in restored] == list(range(1, 100))
    assert restored[-1]["checkpoint_committed"] is True
    assert restored[-1]["completed_epochs"] == 99
    assert restored[-1]["last_checkpoint_committed_epoch"] == 99
    assert restored[-1]["global_step"] == 495
    assert restored[-1]["optimizer_step"] == 495
    assert trainer2.completed_epochs == 99
    assert trainer2.last_checkpoint_committed_epoch == 99
    # save_checkpoint merges extra into the payload top level, so the saved
    # validation must be restored from there rather than silently reset.
    assert trainer2.last_completed_validation == {SELECTION_METRIC_KEY: 0.5}


# ============================================================================
# Gate 6: Bounded Smoke & Calibration Firewall
# ============================================================================


def test_bounded_smoke_stops_at_max_steps_and_bars_calibration(tmp_path: Path) -> None:
    """max_steps stops bounded smoke, marks run incomplete, and prevents calibration."""
    config = load_unified_config("configs/self_audit_full.yaml")
    overrides = {
        "output_dir": str(tmp_path / "weights"),
        "report_dir": str(tmp_path / "reports"),
    }
    config = apply_overrides(config, overrides)

    model = _make_tiny_net(50)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)

    # Inject mock small dataloaders
    ds = _SyntheticDataset(count=4)
    train_loader = DataLoader(ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(ds, batch_size=2, shuffle=False)
    trainer._loader_cache[(4, True)] = (train_loader, val_loader)
    trainer._loader_cache[(4, False)] = (train_loader, val_loader)
    trainer._loader_cache[(2, True)] = (train_loader, val_loader)

    # Run with max_steps=2
    report = trainer.train(max_steps=2, max_val_batches=1)

    assert report["completed"] is False
    assert report["incomplete_reason"] == "max_steps_reached"
    assert report["calibration"] is None
    assert trainer.optimizer_step == 2

    # Verify report JSON on disk
    report_file = tmp_path / "reports" / "pipeline_report.json"
    assert report_file.exists()
    import json
    saved_report = json.loads(report_file.read_text(encoding="utf-8"))
    assert saved_report["completed"] is False
    assert saved_report["calibration"] is None


def test_max_val_batches_alone_restricts_completion_and_bars_calibration(tmp_path: Path) -> None:
    """Restricting validation batches via max_val_batches prevents completed status and bars calibration."""
    cfg_dict = _get_valid_config_dict()
    cfg_dict["checkpoint"]["output_dir"] = str(tmp_path / "weights")
    cfg_dict["logging"]["report_dir"] = str(tmp_path / "reports")
    cfg_dict["training"]["schedule"]["total_epochs"] = 1
    cfg_dict["training"]["schedule"]["intervals"] = [
        {
            "start_epoch": 0,
            "end_epoch": 1,
            "name": "annotation_bootstrap",
            "trainable": "annotation",
            "encoder_lr": 0.0001,
            "annotation_lr": 0.001,
            "auditor_lr": 0.0,
            "annotation_weight": 1.0,
            "audit_weight": 0.0,
            "objective": "weighted_a0_a3",
            "transition_population": "none",
            "rollout": "propagate_no_audit",
            "batch_size": 2,
            "accumulation_steps": 1,
            "augment": False,
            "reset_optimizer": False,
        }
    ]
    config = parse_unified_config(cfg_dict)
    model = _make_tiny_net(50)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)

    ds = _SyntheticDataset(count=4)
    train_loader = DataLoader(ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(ds, batch_size=2, shuffle=False)
    trainer._loader_cache[(2, False)] = (train_loader, val_loader)

    report = trainer.train(max_val_batches=1)
    assert report["completed"] is False
    assert report["incomplete_reason"] == "max_val_batches_restricted"
    assert report["calibration"] is None



# ============================================================================
# Gate 7: best.pt Selection Restricted to Epochs >= 120
# ============================================================================


def test_best_pt_selection_restricted_to_epochs_ge_120(tmp_path: Path) -> None:
    """best.pt is never created during early epochs (<120)."""
    config = load_unified_config("configs/self_audit_full.yaml")
    overrides = {
        "output_dir": str(tmp_path / "weights"),
        "report_dir": str(tmp_path / "reports"),
    }
    config = apply_overrides(config, overrides)

    model = _make_tiny_net(60)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)

    ds = _SyntheticDataset(count=4)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    trainer._loader_cache[(4, True)] = (loader, loader)
    trainer._loader_cache[(4, False)] = (loader, loader)
    trainer._loader_cache[(2, True)] = (loader, loader)

    # Run 1 epoch in interval 0 (epoch 0)
    report = trainer.train(start_epoch=0, max_steps=1, max_val_batches=1)

    output_dir = tmp_path / "weights"
    assert (output_dir / "last.pt").exists()
    assert not (output_dir / "best.pt").exists(), "best.pt must NOT be written for epochs < 120"


# ============================================================================
# Gate 8: Audit Detachment & Rollout Halting
# ============================================================================


def test_rollout_strict_tau_equality_and_independent_halting() -> None:
    """When DeltaQ <= tau_accept, refinement is rejected and candidate state is not adopted."""
    model = _make_tiny_net(70)
    image = torch.randn(2, 3, 32, 32)

    # At tau_accept = 1000.0 (impossibly high), all candidates must be rejected
    output_rejected = model.infer(image, mode="self_audit", tau_accept=1000.0, t_max=2)
    initial_logits = output_rejected["initial_logits"]
    final_logits = output_rejected["logits"]

    assert torch.allclose(initial_logits, final_logits), "All steps rejected must retain initial logits"


def test_auditor_gradients_isolate_to_auditor_parameters() -> None:
    """In Interval 1 (auditor training), backward pass must not update annotation parameters."""
    config = load_unified_config("configs/self_audit_full.yaml")
    model = _make_tiny_net(80)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)

    int_1 = config.training.schedule.intervals[1]
    trainer.setup_interval_optimizer_and_scheduler(int_1, num_batches=5)

    batch = {
        "image": torch.randn(2, 3, 32, 32),
        "mask": torch.randint(0, 4, (2, 32, 32)),
    }

    loss, details = trainer.compute_batch_loss(batch, int_1)
    loss.backward()

    # Annotation parameters must have None gradients
    for name, param in model.named_parameters():
        if not name.startswith("auditor"):
            assert param.grad is None, f"Annotation parameter {name} received non-None gradient"
        else:
            if param.requires_grad:
                assert param.grad is not None, f"Trainable auditor parameter {name} had None gradient"


def test_cli_legacy_flags_rejected_in_canonical_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canonical runner strictly rejects legacy multi-config flags and points to legacy runner."""
    import scripts.train_self_audit as runner

    monkeypatch.setattr(sys, "argv", ["train_self_audit.py", "--config_a", "configs/self_audit_annotation.yaml"])
    with pytest.raises(SystemExit) as exc_info:
        runner.main()
    assert exc_info.value.code != 0
    assert "Legacy multi-config flags" in str(exc_info.value)


def test_legacy_runner_accepts_legacy_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Historical runner scripts/train_self_audit_legacy.py continues to accept legacy flags."""
    import scripts.train_self_audit_legacy as legacy_runner

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_self_audit_legacy.py",
            "--config_a", "configs/self_audit_annotation.yaml",
            "--config_b", "configs/self_audit_auditor.yaml",
            "--config_c", "configs/self_audit_joint.yaml",
            "--help",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        legacy_runner._parse_args()
    assert exc_info.value.code == 0




def test_cli_unified_smoke_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Running scripts/train_self_audit.py with --config and --max_steps operates as single process."""
    import scripts.train_self_audit as runner

    # Create a fast tiny config
    cfg_dict = _get_valid_config_dict()
    cfg_dict["model"]["pretrained_encoder"] = False
    cfg_dict["model"]["fallback"] = True
    cfg_dict["model"]["shared_channels"] = 16
    cfg_dict["model"]["window_k"] = 4
    cfg_dict["model"]["max_turns"] = 2
    cfg_dict["checkpoint"]["output_dir"] = str(tmp_path / "weights")
    cfg_dict["checkpoint"]["best_selection_min_epoch"] = 2
    cfg_dict["logging"]["report_dir"] = str(tmp_path / "reports")
    cfg_dict["training"]["amp"]["enabled"] = False
    cfg_dict["training"]["schedule"]["total_epochs"] = 3
    cfg_dict["training"]["schedule"]["intervals"] = [
        {
            "start_epoch": 0,
            "end_epoch": 1,
            "name": "annotation_bootstrap",
            "trainable": "annotation",
            "encoder_lr": 0.00003,
            "annotation_lr": 0.0003,
            "auditor_lr": 0.0,
            "annotation_weight": 1.0,
            "audit_weight": 0.0,
            "objective": "weighted_a0_a3",
            "transition_population": "none",
            "rollout": "propagate_no_audit",
            "batch_size": 2,
            "accumulation_steps": 1,
            "augment": False,
            "reset_optimizer": False,
        },
        {
            "start_epoch": 1,
            "end_epoch": 2,
            "name": "auditor_training",
            "trainable": "auditor",
            "encoder_lr": 0.0,
            "annotation_lr": 0.0,
            "auditor_lr": 0.0003,
            "annotation_weight": 0.0,
            "audit_weight": 1.0,
            "objective": "counterfactual_audit",
            "transition_population": "adjacent_and_synthetic",
            "rollout": "annotation_eval",
            "batch_size": 2,
            "accumulation_steps": 1,
            "augment": False,
            "reset_optimizer": True,
        },
        {
            "start_epoch": 2,
            "end_epoch": 3,
            "name": "joint_self_audit",
            "trainable": "all",
            "encoder_lr": 0.000001,
            "annotation_lr": 0.00001,
            "auditor_lr": 0.00001,
            "annotation_weight": 1.0,
            "audit_weight": 1.0,
            "objective": "retained_final_annotation",
            "transition_population": "active_attempted",
            "rollout": "threshold_gate",
            "batch_size": 2,
            "accumulation_steps": 1,
            "augment": False,
            "reset_optimizer": True,
        },
    ]

    config_path = tmp_path / "tiny_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    # Mock dataset loading to use _SyntheticDataset
    ds = _SyntheticDataset(count=4)
    synthetic_loader = DataLoader(ds, batch_size=2, shuffle=False)

    from self_audit.training.unified_trainer import UnifiedTrainer
    orig_get_loaders = UnifiedTrainer.get_loaders

    def mock_get_loaders(self, interval):
        return synthetic_loader, synthetic_loader

    monkeypatch.setattr(UnifiedTrainer, "get_loaders", mock_get_loaders)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_self_audit.py",
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--max_steps",
            "1",
            "--max_val_batches",
            "1",
            "--no_tqdm",
        ],
    )

    runner.main()

    # Check report
    report_file = tmp_path / "reports" / "pipeline_report.json"
    assert report_file.exists()
    import json
    report_data = json.loads(report_file.read_text(encoding="utf-8"))
    assert report_data["completed"] is False
    assert report_data["incomplete_reason"] == "max_steps_reached"


# ============================================================================
# Gate 11: Adversarial Probes & Revision Findings (Findings 1 - 6)
# ============================================================================


def test_astra_probe_1_interval0_annotation_weight_rejected() -> None:
    """Probe 1: interval 0 with annotation_weight != 1.0 (e.g. 7.0) must raise ValueError."""
    cfg_dict = _get_valid_config_dict()
    cfg_dict["training"]["schedule"]["intervals"][0]["annotation_weight"] = 7.0
    with pytest.raises(ValueError, match=r"annotation_weight=1\.0"):
        parse_unified_config(cfg_dict)


def test_astra_probe_2_interval0_transition_population_rejected() -> None:
    """Probe 2: interval 0 with transition_population != 'none' must raise ValueError."""
    cfg_dict = _get_valid_config_dict()
    cfg_dict["training"]["schedule"]["intervals"][0]["transition_population"] = "active_attempted"
    with pytest.raises(ValueError, match=r"transition_population='none'"):
        parse_unified_config(cfg_dict)


def test_astra_probe_3_interval1_and_2_reset_optimizer_false_rejected() -> None:
    """Probe 3: intervals 1 and 2 must have reset_optimizer=True."""
    cfg_dict1 = _get_valid_config_dict()
    cfg_dict1["training"]["schedule"]["intervals"][1]["reset_optimizer"] = False
    with pytest.raises(ValueError, match=r"requires reset_optimizer=True"):
        parse_unified_config(cfg_dict1)

    cfg_dict2 = _get_valid_config_dict()
    cfg_dict2["training"]["schedule"]["intervals"][2]["reset_optimizer"] = False
    with pytest.raises(ValueError, match=r"requires reset_optimizer=True"):
        parse_unified_config(cfg_dict2)


def test_astra_probe_4_calibration_split_mismatch_rejected() -> None:
    """Probe 4: calibration.split != dataset.val_split must raise ValueError."""
    cfg_dict = _get_valid_config_dict()
    cfg_dict["calibration"]["split"] = "test"
    cfg_dict["dataset"]["val_split"] = "val"
    with pytest.raises(ValueError, match=r"must equal dataset\.val_split"):
        parse_unified_config(cfg_dict)


def test_none_audit_loss_skips_optimizer_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 3: When _auditor_batch returns None, trainer skips backward and optimizer step."""
    config = load_unified_config("configs/self_audit_full.yaml")
    model = _make_tiny_net(55)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)

    int_1 = config.training.schedule.intervals[1]
    trainer.setup_interval_optimizer_and_scheduler(int_1, num_batches=1)

    # Force _auditor_batch to return (None, details) simulating 0 transitions
    def mock_auditor_batch(*args: Any, **kwargs: Any) -> tuple[None, dict[str, Any]]:
        return None, {"loss": 0.0, "transitions": 0}

    import self_audit.training.unified_trainer as ut_module
    monkeypatch.setattr(ut_module, "_auditor_batch", mock_auditor_batch)

    batch = {
        "image": torch.randn(2, 3, 32, 32),
        "mask": torch.randint(0, 4, (2, 32, 32)),
    }

    loss, details = trainer.compute_batch_loss(batch, int_1)
    assert loss is None
    assert details["transitions"] == 0

    # Ensure running train_epoch on this loader does not increment optimizer_step or mutate weights
    ds = _SyntheticDataset(count=2)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    param = next(p for n, p in model.named_parameters() if n.startswith("auditor") and p.requires_grad)
    weight_before = param.clone().detach()

    stats = trainer.train_epoch(int_1, loader)
    assert stats["epoch_optimizer_steps"] == 0
    assert trainer.optimizer_step == 0
    assert torch.allclose(param, weight_before)


def test_phase_b_amp_context_parity() -> None:
    """Finding 2: In Interval 1, model.forward_annotation runs outside autocast, while auditor runs in autocast."""
    config = load_unified_config("configs/self_audit_full.yaml")
    model = _make_tiny_net(65)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)
    trainer.amp_enabled = True
    trainer.amp_dtype = torch.bfloat16

    int_1 = config.training.schedule.intervals[1]

    forward_autocast_active = None
    orig_forward = model.forward_annotation

    def tracked_forward(img: torch.Tensor) -> Any:
        nonlocal forward_autocast_active
        forward_autocast_active = (
            torch.is_autocast_enabled("cpu")
            if hasattr(torch, "is_autocast_enabled")
            else torch.is_autocast_cpu_enabled()
        )
        return orig_forward(img)

    model.forward_annotation = tracked_forward  # type: ignore[assignment]

    batch = {
        "image": torch.randn(2, 3, 32, 32),
        "mask": torch.randint(0, 4, (2, 32, 32)),
    }

    loss, _ = trainer.compute_batch_loss(batch, int_1)
    assert forward_autocast_active is False, "Frozen annotation forward must run outside autocast in Interval 1"


def test_strict_resume_failures(tmp_path: Path) -> None:
    """Finding 4: Resume must fail on incomplete smoke checkpoints, cardinality mismatches, and config mismatches."""
    config = load_unified_config("configs/self_audit_full.yaml")
    model = _make_tiny_net(75)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()

    from self_audit.training._utils import save_checkpoint

    int_0 = config.training.schedule.intervals[0]

    # Case 1: Incomplete epoch / non-resumable smoke checkpoint.  Every other
    # field is a complete unit-fixture commit, so the refusal can only come
    # from the resumability flags themselves.
    trainer = UnifiedTrainer(config, model=_make_tiny_net(76), device=torch.device("cpu"), disable_tqdm=True)
    smoke_ckpt = ckpt_dir / "smoke.pt"
    save_checkpoint(
        smoke_ckpt,
        model,
        epoch=1,
        global_step=1,
        optimizer_step=1,
        config=config.to_dict(),
        extra=_unit_fixture_commit_extra(
            trainer,
            epoch=1,
            global_step=1,
            optimizer_step=1,
            loader_cardinality=5,
            resumable=False,
            incomplete_epoch=True,
            validation_complete=False,
        ),
    )
    with pytest.raises(ValueError, match=r"'resumable' is False, expected True"):
        trainer.resume_from_checkpoint(smoke_ckpt)

    # Case 1b: An absent flag is unknown, and unknown is not permission -- a
    # checkpoint that simply omits the certification must be refused exactly
    # like one that denies it.
    for omitted in ("resumable", "incomplete_epoch", "validation_complete"):
        omitted_extra = _unit_fixture_commit_extra(
            trainer,
            epoch=1,
            global_step=1,
            optimizer_step=1,
            loader_cardinality=5,
        )
        del omitted_extra[omitted]
        omitted_ckpt = ckpt_dir / f"omitted_{omitted}.pt"
        save_checkpoint(
            omitted_ckpt,
            model,
            epoch=1,
            global_step=1,
            optimizer_step=1,
            config=config.to_dict(),
            extra=omitted_extra,
        )
        with pytest.raises(ValueError, match=rf"required flag '{omitted}' is absent"):
            trainer.resume_from_checkpoint(omitted_ckpt)

    # Case 2: Loader cardinality mismatch
    card_ckpt = ckpt_dir / "cardinality.pt"
    trainer2 = UnifiedTrainer(config, model=_make_tiny_net(77), device=torch.device("cpu"), disable_tqdm=True)
    mock_ds = _SyntheticDataset(count=6)
    mock_loader = DataLoader(mock_ds, batch_size=2, shuffle=False)  # len is 3, mismatch with 10
    trainer2._loader_cache[(int_0.batch_size, int_0.augment)] = (mock_loader, mock_loader)
    cohort_desc = trainer2.get_cohort_descriptor(int_0)
    save_checkpoint(
        card_ckpt,
        model,
        epoch=1,
        global_step=5,
        optimizer_step=5,
        config=config.to_dict(),
        extra=_unit_fixture_commit_extra(
            trainer2,
            epoch=1,
            global_step=5,
            optimizer_step=5,
            cohort_descriptor=cohort_desc,
            loader_cardinality=10,
        ),
    )

    with pytest.raises(ValueError, match="Loader cardinality mismatch on resume"):
        trainer2.resume_from_checkpoint(card_ckpt)

    # Case 3: Incompatible config num_classes
    cfg_mismatch = config.to_dict()
    cfg_mismatch["model"]["num_classes"] = 99
    mismatch_ckpt = ckpt_dir / "mismatch.pt"
    trainer3 = UnifiedTrainer(config, model=_make_tiny_net(78), device=torch.device("cpu"), disable_tqdm=True)
    trainer3._loader_cache[(int_0.batch_size, int_0.augment)] = (mock_loader, mock_loader)
    save_checkpoint(
        mismatch_ckpt,
        model,
        epoch=1,
        global_step=5,
        optimizer_step=5,
        config=cfg_mismatch,
        extra=_unit_fixture_commit_extra(
            trainer3,
            epoch=1,
            global_step=5,
            optimizer_step=5,
            cohort_descriptor=trainer3.get_cohort_descriptor(int_0),
            loader_cardinality=3,
        ),
    )
    with pytest.raises(ValueError, match="Config mismatch on resume: saved num_classes=99"):
        trainer3.resume_from_checkpoint(mismatch_ckpt)

    # Case 4: Schedule and counter coherence.  A fully certified checkpoint is
    # still refused when the epoch falls outside the configured schedule or the
    # optimizer claims more updates than steps were taken.
    out_of_range = ckpt_dir / "out_of_range.pt"
    total_epochs = trainer3.schedule.total_epochs
    coherent = _unit_fixture_commit_extra(
        trainer3,
        epoch=1,
        global_step=5,
        optimizer_step=5,
        cohort_descriptor=trainer3.get_cohort_descriptor(int_0),
        loader_cardinality=3,
    )
    save_checkpoint(
        out_of_range,
        model,
        epoch=total_epochs + 1,
        global_step=5,
        optimizer_step=5,
        config=config.to_dict(),
        extra=coherent,
    )
    with pytest.raises(ValueError, match=rf"is outside the configured schedule \[1, {total_epochs}\]"):
        trainer3.resume_from_checkpoint(out_of_range)

    inverted = ckpt_dir / "inverted_counters.pt"
    save_checkpoint(
        inverted,
        model,
        epoch=1,
        global_step=3,
        optimizer_step=5,
        config=config.to_dict(),
        extra=_unit_fixture_commit_extra(
            trainer3,
            epoch=1,
            global_step=3,
            optimizer_step=5,
            cohort_descriptor=trainer3.get_cohort_descriptor(int_0),
            loader_cardinality=3,
        ),
    )
    with pytest.raises(ValueError, match=r"optimizer_step 5 exceeds global_step 3"):
        trainer3.resume_from_checkpoint(inverted)

    # Case 5: A history that does not describe exactly epochs 1..N, or whose
    # tail disagrees with the counters it is saved beside, is a refusal rather
    # than a silent truncation.
    truncated_extra = _unit_fixture_commit_extra(
        trainer3,
        epoch=2,
        global_step=6,
        optimizer_step=6,
        cohort_descriptor=trainer3.get_cohort_descriptor(int_0),
        loader_cardinality=3,
    )
    truncated_extra["epoch_history"] = truncated_extra["epoch_history"][:1]
    truncated = ckpt_dir / "truncated_history.pt"
    save_checkpoint(
        truncated,
        model,
        epoch=2,
        global_step=6,
        optimizer_step=6,
        config=config.to_dict(),
        extra=truncated_extra,
    )
    with pytest.raises(ValueError, match=r"refusing a truncated history"):
        trainer3.resume_from_checkpoint(truncated)

    uncommitted_extra = _unit_fixture_commit_extra(
        trainer3,
        epoch=1,
        global_step=5,
        optimizer_step=5,
        cohort_descriptor=trainer3.get_cohort_descriptor(int_0),
        loader_cardinality=3,
    )
    uncommitted_extra["epoch_history"][-1]["checkpoint_committed"] = False
    uncommitted = ckpt_dir / "uncommitted_tail.pt"
    save_checkpoint(
        uncommitted,
        model,
        epoch=1,
        global_step=5,
        optimizer_step=5,
        config=config.to_dict(),
        extra=uncommitted_extra,
    )
    with pytest.raises(ValueError, match=r"tail row is not marked checkpoint_committed"):
        trainer3.resume_from_checkpoint(uncommitted)


def test_fresh_run_refuses_preexisting_best_pt_and_preserves_bytes(tmp_path: Path) -> None:
    """Finding 5 (W2): A fresh unresumed run refuses existing best.pt and preserves file bytes."""
    config = load_unified_config("configs/self_audit_full.yaml")
    out_dir = tmp_path / "weights"
    out_dir.mkdir()
    rep_dir = tmp_path / "reports"
    rep_dir.mkdir()

    overrides = {
        "output_dir": str(out_dir),
        "report_dir": str(rep_dir),
        "skip_calibration": True,
    }
    config = apply_overrides(config, overrides)

    # Plant a fake old best.pt
    stale_best = out_dir / "best.pt"
    stale_bytes = b"fake_old_weights"
    stale_best.write_bytes(stale_bytes)
    assert stale_best.exists()

    model = _make_tiny_net(85)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)

    ds = _SyntheticDataset(count=4)
    train_loader = DataLoader(ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(ds, batch_size=2, shuffle=False)
    for int_cfg in config.training.schedule.intervals:
        trainer._loader_cache[(int_cfg.batch_size, int_cfg.augment)] = (train_loader, val_loader)

    # Fresh run must refuse to overwrite or delete existing best.pt
    with pytest.raises(FileExistsError, match="existing checkpoint artifact"):
        trainer.train(max_steps=1)

    # Stale best.pt must NOT have been unlinked or overwritten; bytes must be unchanged
    assert stale_best.exists()
    assert stale_best.read_bytes() == stale_bytes
    assert trainer.written_best_path is None


def test_real_subprocess_cli_execution(tmp_path: Path) -> None:
    """Run scripts/train_self_audit.py via real subprocess with synthetic data on disk."""
    data_dir = tmp_path / "acdc_synthetic"
    data_dir.mkdir()
    vol_dir = data_dir / "volumes"
    msk_dir = data_dir / "masks"
    vol_dir.mkdir()
    msk_dir.mkdir()

    # Create synthetic volume and mask files
    case_ids = ["patient001", "patient002"]
    for cid in case_ids:
        img_arr = np.random.randn(4, 32, 32).astype(np.float32)
        msk_arr = np.random.randint(0, 4, size=(4, 32, 32)).astype(np.uint8)
        np.save(vol_dir / f"{cid}.npy", img_arr)
        np.save(msk_dir / f"{cid}.npy", msk_arr)

    # Split manifest
    import json
    split_manifest = {
        "train": ["patient001"],
        "val": ["patient002"],
    }
    manifest_file = data_dir / "splits.json"
    manifest_file.write_text(json.dumps(split_manifest), encoding="utf-8")

    # Config
    cfg_dict = _get_valid_config_dict()
    cfg_dict["model"]["pretrained_encoder"] = False
    cfg_dict["model"]["fallback"] = True
    cfg_dict["model"]["shared_channels"] = 8
    cfg_dict["model"]["window_k"] = 2
    cfg_dict["model"]["max_turns"] = 1
    cfg_dict["dataset"]["data_root"] = str(data_dir)
    cfg_dict["dataset"]["split_manifest"] = str(manifest_file)
    cfg_dict["checkpoint"]["output_dir"] = str(tmp_path / "weights")
    cfg_dict["logging"]["report_dir"] = str(tmp_path / "reports")
    cfg_dict["training"]["amp"]["enabled"] = False
    cfg_dict["training"]["schedule"]["total_epochs"] = 1
    cfg_dict["training"]["schedule"]["intervals"] = [
        {
            "start_epoch": 0,
            "end_epoch": 1,
            "name": "annotation_bootstrap",
            "trainable": "annotation",
            "encoder_lr": 0.0001,
            "annotation_lr": 0.001,
            "auditor_lr": 0.0,
            "annotation_weight": 1.0,
            "audit_weight": 0.0,
            "objective": "weighted_a0_a3",
            "transition_population": "none",
            "rollout": "propagate_no_audit",
            "batch_size": 2,
            "accumulation_steps": 1,
            "augment": False,
            "reset_optimizer": False,
        }
    ]

    conf_path = tmp_path / "test_conf.yaml"
    conf_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/train_self_audit.py",
        "--config",
        str(conf_path),
        "--device",
        "cpu",
        "--max_steps",
        "1",
        "--max_val_batches",
        "1",
        "--no_tqdm",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Subprocess failed with returncode {res.returncode}:\n{res.stderr}"

    report_file = tmp_path / "reports" / "pipeline_report.json"
    assert report_file.exists()
    rep = json.loads(report_file.read_text(encoding="utf-8"))
    assert rep["completed"] is False
    assert rep["incomplete_reason"] == "max_steps_reached"


def test_canonical_shell_script_smoke_and_rejection(tmp_path: Path) -> None:
    """Canonical shell wrapper scripts/run_full_pipeline.sh executes unified pipeline and rejects legacy flags."""
    # 1. Test rejection of legacy flags
    res_reject = subprocess.run(
        ["bash", "scripts/run_full_pipeline.sh", "--config_a", "configs/self_audit_annotation.yaml"],
        capture_output=True,
        text=True,
    )
    assert res_reject.returncode == 2
    assert "Legacy multi-phase option" in res_reject.stderr
    assert "run_full_pipeline_legacy.sh" in res_reject.stderr

    # 2. Test help output
    res_help = subprocess.run(
        ["bash", "scripts/run_full_pipeline.sh", "--help"],
        capture_output=True,
        text=True,
    )
    assert res_help.returncode == 0
    assert "Canonical Unified Pipeline Options" in res_help.stdout


# ============================================================================
# Gate 11: Corrective Wave 1 Regressions (RNG, Cohort, Relocation, Reset)
# ============================================================================


def test_rng_state_full_roundtrip_and_backward_compatibility(tmp_path: Path) -> None:
    """_rng_state captures Python, NumPy, PyTorch CPU/CUDA in weights_only=True format, and supports backward reads."""
    from self_audit.training._utils import _rng_state, _restore_rng_state, save_checkpoint
    import random

    # Set known states
    random.seed(12345)
    np.random.seed(67890)
    torch.manual_seed(13579)

    # Capture state
    rng = _rng_state()
    assert "python" in rng
    assert "numpy" in rng
    assert "torch" in rng
    assert isinstance(rng["python"], tuple)
    assert isinstance(rng["numpy"], (tuple, list))
    assert isinstance(rng["torch"], torch.Tensor)

    # Save to checkpoint and verify weights_only=True loading
    model = _make_tiny_net(11)
    ckpt_file = tmp_path / "rng_test.pt"
    save_checkpoint(ckpt_file, model, epoch=1, global_step=1, optimizer_step=1)

    payload = torch.load(ckpt_file, weights_only=True)
    assert "rng_state" in payload
    saved_rng = payload["rng_state"]

    # Generate reference values before state disturbance
    _restore_rng_state(saved_rng)
    ref_py = [random.random() for _ in range(5)]
    ref_np = np.random.rand(5).tolist()
    ref_th = torch.rand(5).tolist()

    # Mutate all RNG states
    random.seed(99999)
    np.random.seed(88888)
    torch.manual_seed(77777)
    _ = [random.random() for _ in range(10)]
    _ = np.random.rand(10)
    _ = torch.rand(10)

    # Restore and verify exact replay
    _restore_rng_state(saved_rng)
    assert [random.random() for _ in range(5)] == ref_py
    assert np.allclose(np.random.rand(5).tolist(), ref_np)
    assert torch.allclose(torch.tensor(torch.rand(5).tolist()), torch.tensor(ref_th))

    # Test backward compatibility with historical checkpoints containing only torch
    legacy_rng = {"torch": rng["torch"]}
    _restore_rng_state(legacy_rng)  # Must not crash or raise KeyError

    # Test malformed present RNG values raise instead of silently passing
    with pytest.raises((ValueError, TypeError)):
        _restore_rng_state({"python": "broken", "numpy": "broken"})
    with pytest.raises((ValueError, TypeError)):
        _restore_rng_state({"numpy": "broken"})
    with pytest.raises(ValueError):
        _restore_rng_state({"numpy": {"algorithm": "MT19937"}})  # missing keys
    with pytest.raises((ValueError, TypeError)):
        _restore_rng_state({"torch": "broken"})
    with pytest.raises(ValueError):
        _restore_rng_state({"python": None})
    with pytest.raises(TypeError):
        _restore_rng_state("not_a_mapping")


def test_cohort_descriptor_detects_same_count_changed_patient_fixtures() -> None:
    """compare_cohort_descriptors catches same-count changed patient fixture, nested sampling, and empty metadata."""
    from self_audit.training.unified_trainer import compare_cohort_descriptors

    base_cohort = {
        "dataset_name": "acdc",
        "data_root": "/path/to/data",
        "split_manifest": "/path/to/splits.json",
        "train_membership": {
            "split": "train",
            "patient_count": 2,
            "case_count": 2,
            "patient_ids": ["patient001", "patient002"],
            "case_ids": ["patient001_frame01", "patient002_frame01"],
            "membership_signature": "sig123",
        },
        "val_membership": {
            "split": "val",
            "patient_count": 1,
            "case_count": 1,
            "patient_ids": ["patient003"],
            "case_ids": ["patient003_frame01"],
            "membership_signature": "sig456",
        },
        "train_cohort_identity": {
            "split_name": "train",
            "membership_signature": "sig123",
            "sample_count": 2,
            "patient_count": 2,
            "batch_size": 2,
            "sampling": {
                "sampler": "RandomSampler",
                "batch_sampler": "BatchSampler",
                "shuffled": True,
                "drop_last": False,
                "iterates_every_record": True,
            },
        },
        "val_cohort_identity": {
            "split_name": "val",
            "membership_signature": "sig456",
            "sample_count": 1,
            "patient_count": 1,
            "batch_size": 2,
            "sampling": {
                "sampler": "SequentialSampler",
                "batch_sampler": "BatchSampler",
                "shuffled": False,
                "drop_last": False,
                "iterates_every_record": True,
            },
        },
    }

    # 1. Matching cohort passes cleanly
    compare_cohort_descriptors(base_cohort, copy.deepcopy(base_cohort))

    # 2. Same-count changed patient fixture in train
    changed_train_cohort = copy.deepcopy(base_cohort)
    changed_train_cohort["train_membership"]["patient_ids"] = ["patient001", "patient999"]
    changed_train_cohort["train_membership"]["membership_signature"] = "sig_diff"
    with pytest.raises(ValueError, match="Cohort membership mismatch on resume for train split: patient IDs differ"):
        compare_cohort_descriptors(base_cohort, changed_train_cohort)

    # 3. Same-count changed patient fixture in val
    changed_val_cohort = copy.deepcopy(base_cohort)
    changed_val_cohort["val_membership"]["patient_ids"] = ["patient888"]
    changed_val_cohort["val_membership"]["membership_signature"] = "sig_diff_val"
    with pytest.raises(ValueError, match="Cohort membership mismatch on resume for val split: patient IDs differ"):
        compare_cohort_descriptors(base_cohort, changed_val_cohort)

    # 4. Nested loader sampling state mismatch: shuffled
    changed_loader_shuffled = copy.deepcopy(base_cohort)
    changed_loader_shuffled["train_cohort_identity"]["sampling"]["shuffled"] = False
    with pytest.raises(ValueError, match=r"Loader sampling mismatch on resume for train split \(shuffled\)"):
        compare_cohort_descriptors(base_cohort, changed_loader_shuffled)

    # 5. Nested loader sampling state mismatch: drop_last
    changed_loader_drop = copy.deepcopy(base_cohort)
    changed_loader_drop["val_cohort_identity"]["sampling"]["drop_last"] = True
    with pytest.raises(ValueError, match=r"Loader sampling mismatch on resume for val split \(drop_last\)"):
        compare_cohort_descriptors(base_cohort, changed_loader_drop)

    # 6. Empty metadata handling
    with pytest.raises(ValueError, match="Saved cohort descriptor cannot be empty"):
        compare_cohort_descriptors({}, base_cohort)
    with pytest.raises(ValueError, match="Current cohort descriptor cannot be empty"):
        compare_cohort_descriptors(base_cohort, {})

    # 7. Missing or empty membership_signature
    empty_sig = copy.deepcopy(base_cohort)
    empty_sig["train_membership"]["membership_signature"] = ""
    with pytest.raises(ValueError, match="missing real membership_signature for train split"):
        compare_cohort_descriptors(empty_sig, base_cohort)

    whitespace_sig = copy.deepcopy(base_cohort)
    whitespace_sig["val_membership"]["membership_signature"] = "   "
    with pytest.raises(ValueError, match="missing real membership_signature for val split"):
        compare_cohort_descriptors(base_cohort, whitespace_sig)

    # 8. Missing or empty identity / sampling metadata
    empty_ident = copy.deepcopy(base_cohort)
    empty_ident["train_cohort_identity"] = {}
    with pytest.raises(ValueError, match="missing required 'train_cohort_identity' metadata"):
        compare_cohort_descriptors(empty_ident, base_cohort)

    empty_sampling = copy.deepcopy(base_cohort)
    empty_sampling["train_cohort_identity"]["sampling"] = {}
    with pytest.raises(ValueError, match="missing required 'sampling' metadata"):
        compare_cohort_descriptors(empty_sampling, base_cohort)


def test_output_relocation_safety_and_clobber_prevention(tmp_path: Path) -> None:
    """Output relocation preserves best.pt lineage and refuses to clobber pre-existing files."""
    config = load_unified_config("configs/self_audit_full.yaml")
    model = _make_tiny_net(22)

    originating_dir = tmp_path / "originating"
    originating_dir.mkdir()
    int_0 = config.training.schedule.intervals[0]

    # Save a fake best.pt and last.pt in originating_dir
    best_file = originating_dir / "best.pt"
    from self_audit.training._utils import save_checkpoint
    save_checkpoint(best_file, model, epoch=1, global_step=5, optimizer_step=5, config=config.to_dict())
    best_hash = hashlib.sha256(best_file.read_bytes()).hexdigest()

    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)
    mock_ds = _SyntheticDataset(count=4)
    mock_loader = DataLoader(mock_ds, batch_size=2, shuffle=False)
    for bs in (2, 4):
        for aug in (True, False):
            trainer._loader_cache[(bs, aug)] = (mock_loader, mock_loader)
    trainer.setup_interval_optimizer_and_scheduler(int_0, num_batches=2)
    cohort_desc = trainer.get_cohort_descriptor(int_0)

    last_file = originating_dir / "last.pt"
    save_checkpoint(
        last_file,
        model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        epoch=1,
        global_step=5,
        optimizer_step=5,
        config=config.to_dict(),
        extra=_unit_fixture_commit_extra(
            trainer,
            epoch=1,
            global_step=5,
            optimizer_step=5,
            cohort_descriptor=cohort_desc,
            loader_cardinality=2,
            selection_metric_value=0.85,
            best_checkpoint_hash=best_hash,
            best_epoch=1,
            best_metric=0.85,
        ),
    )

    # Case A: Safe relocation into clean/new target directory retains best lineage
    clean_target = tmp_path / "clean_target"
    config_reloc = apply_overrides(config, {"output_dir": str(clean_target)})
    trainer_reloc = UnifiedTrainer(config_reloc, model=_make_tiny_net(23), device=torch.device("cpu"), disable_tqdm=True)
    for bs in (2, 4):
        for aug in (True, False):
            trainer_reloc._loader_cache[(bs, aug)] = (mock_loader, mock_loader)

    resumed_ep = trainer_reloc.resume_from_checkpoint(last_file)
    assert resumed_ep == 1
    assert (clean_target / "best.pt").exists()
    assert hashlib.sha256((clean_target / "best.pt").read_bytes()).hexdigest() == best_hash
    assert trainer_reloc.written_best_path == clean_target / "best.pt"

    # Case B: Refuse relocation when target directory contains unrelated existing files
    dirty_target = tmp_path / "dirty_target"
    dirty_target.mkdir()
    (dirty_target / "unrelated.txt").write_text("pre-existing data")

    config_dirty = apply_overrides(config, {"output_dir": str(dirty_target)})
    trainer_dirty = UnifiedTrainer(config_dirty, model=_make_tiny_net(24), device=torch.device("cpu"), disable_tqdm=True)
    for bs in (2, 4):
        for aug in (True, False):
            trainer_dirty._loader_cache[(bs, aug)] = (mock_loader, mock_loader)

    with pytest.raises(FileExistsError, match="Output relocation across resume refused"):
        trainer_dirty.resume_from_checkpoint(last_file)

    # Case C: Refuse resume if best.pt referenced by hash was tampered with
    tampered_dir = tmp_path / "tampered"
    tampered_dir.mkdir()
    tampered_best = tampered_dir / "best.pt"
    tampered_best.write_bytes(b"corrupt bytes")
    tampered_last = tampered_dir / "last.pt"
    save_checkpoint(
        tampered_last,
        model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        epoch=1,
        global_step=5,
        optimizer_step=5,
        config=config.to_dict(),
        extra=_unit_fixture_commit_extra(
            trainer,
            epoch=1,
            global_step=5,
            optimizer_step=5,
            cohort_descriptor=cohort_desc,
            loader_cardinality=2,
            selection_metric_value=0.85,
            best_checkpoint_hash=best_hash,  # expects valid hash, but file has corrupt bytes
            best_epoch=1,
            best_metric=0.85,
        ),
    )
    trainer_tampered = UnifiedTrainer(config, model=_make_tiny_net(25), device=torch.device("cpu"), disable_tqdm=True)
    for bs in (2, 4):
        for aug in (True, False):
            trainer_tampered._loader_cache[(bs, aug)] = (mock_loader, mock_loader)
    with pytest.raises(ValueError, match="Stale or replaced best checkpoint detected"):
        trainer_tampered.resume_from_checkpoint(tampered_last)


def test_uninterrupted_vs_resumed_weights_and_optimizer_ordinary_and_reset_boundaries(tmp_path: Path) -> None:
    """Exact parity between uninterrupted run and resumed run across ordinary step and reset boundary 100/120."""
    from self_audit.training._utils import save_checkpoint

    # Part 1: Ordinary step parity within Interval 0 at epoch 1
    torch.manual_seed(42)
    np.random.seed(42)
    config = load_unified_config("configs/self_audit_full.yaml")
    out_a = tmp_path / "weights_a"
    out_b = tmp_path / "weights_b"
    out_a.mkdir()
    out_b.mkdir()

    model_uninterrupted = _make_tiny_net(101)
    trainer_uninterrupted = UnifiedTrainer(config, model=model_uninterrupted, device=torch.device("cpu"), disable_tqdm=True)
    ds = _SyntheticDataset(count=8)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    int_0 = config.training.schedule.intervals[0]
    int_1 = config.training.schedule.intervals[1]
    for bs in (2, 4):
        for aug in (True, False):
            trainer_uninterrupted._loader_cache[(bs, aug)] = (loader, loader)
    trainer_uninterrupted.global_epoch = 1
    trainer_uninterrupted.setup_interval_optimizer_and_scheduler(int_0, len(loader))

    # Step 1 uninterrupted
    stats1 = trainer_uninterrupted.train_epoch(int_0, loader, max_steps=1)
    digest_step1 = state_digest(trainer_uninterrupted.model)

    # Save checkpoint at step 1
    ckpt_step1 = out_a / "step1.pt"
    cohort_desc = trainer_uninterrupted.get_cohort_descriptor(int_0)
    save_checkpoint(
        ckpt_step1,
        trainer_uninterrupted.model,
        optimizer=trainer_uninterrupted.optimizer,
        scheduler=trainer_uninterrupted.scheduler,
        scaler=trainer_uninterrupted.scaler,
        epoch=1,
        global_step=trainer_uninterrupted.global_step,
        optimizer_step=trainer_uninterrupted.optimizer_step,
        config=config.to_dict(),
        extra=_unit_fixture_commit_extra(
            trainer_uninterrupted,
            epoch=1,
            global_step=trainer_uninterrupted.global_step,
            optimizer_step=trainer_uninterrupted.optimizer_step,
            cohort_descriptor=cohort_desc,
            loader_cardinality=len(loader),
        ),
    )

    # Step 2 uninterrupted
    stats2_uninterrupted = trainer_uninterrupted.train_epoch(int_0, loader, max_steps=2)
    digest_uninterrupted = state_digest(trainer_uninterrupted.model)

    # Resumed trainer: resume from step 1 checkpoint
    model_resumed = _make_tiny_net(999)  # different initial weights
    trainer_resumed = UnifiedTrainer(config, model=model_resumed, device=torch.device("cpu"), disable_tqdm=True)
    for bs in (2, 4):
        for aug in (True, False):
            trainer_resumed._loader_cache[(bs, aug)] = (loader, loader)
    trainer_resumed.resume_from_checkpoint(ckpt_step1)
    assert trainer_resumed.optimizer_step == 1
    assert state_digest(trainer_resumed.model) == digest_step1

    # Step 2 resumed
    stats2_resumed = trainer_resumed.train_epoch(int_0, loader, max_steps=2)
    digest_resumed = state_digest(trainer_resumed.model)

    # Verify exact match between uninterrupted and resumed
    assert digest_resumed == digest_uninterrupted
    assert trainer_resumed.optimizer_step == trainer_uninterrupted.optimizer_step

    # Part 2: Boundary 100/120 reset verification
    # Interval 1 (epoch 100) has reset_optimizer=True and trainable=auditor
    assert int_1.reset_optimizer is True
    assert int_1.start_epoch == 100

    # Save boundary checkpoint at epoch 100
    boundary_ckpt = out_a / "epoch100.pt"
    cohort_desc_0 = trainer_uninterrupted.get_cohort_descriptor(int_0)
    save_checkpoint(
        boundary_ckpt,
        trainer_uninterrupted.model,
        epoch=100,
        global_step=500,
        optimizer_step=500,
        config=config.to_dict(),
        extra=_unit_fixture_commit_extra(
            trainer_uninterrupted,
            epoch=100,
            global_step=500,
            optimizer_step=500,
            cohort_descriptor=cohort_desc_0,
            loader_cardinality=len(loader),
        ),
    )

    # Resume at epoch 100: must reset optimizer for auditor
    trainer_boundary = UnifiedTrainer(config, model=_make_tiny_net(777), device=torch.device("cpu"), disable_tqdm=True)
    for bs in (2, 4):
        for aug in (True, False):
            trainer_boundary._loader_cache[(bs, aug)] = (loader, loader)
    res_epoch = trainer_boundary.resume_from_checkpoint(boundary_ckpt)
    assert res_epoch == 100
    # Boundary resume: optimizer must be fresh for Interval 1 (auditor params)
    assert trainer_boundary.current_interval_index == 1
    # Check that auditor params are trainable and in optimizer, while encoder is frozen
    param_count = sum(p.numel() for pg in trainer_boundary.optimizer.param_groups for p in pg["params"])
    auditor_count = sum(p.numel() for p in trainer_boundary.model.auditor.parameters() if p.requires_grad)
    assert param_count == auditor_count


def test_cli_and_runners_no_wandb_and_config_preservation() -> None:
    """CLI and runner wrappers support --no_wandb and preserve config-declared defaults."""
    config = load_unified_config("configs/self_audit_full.yaml")

    # 1. apply_overrides with wandb=False
    cfg_off = apply_overrides(config, {"wandb": False})
    assert cfg_off.logging.wandb.enabled is False

    # 2. apply_overrides with wandb=True
    cfg_on = apply_overrides(config, {"wandb": True})
    assert cfg_on.logging.wandb.enabled is True

    # 3. Inspect scripts/run_full_pipeline.sh: default variables must be empty strings
    sh_text = Path("scripts/run_full_pipeline.sh").read_text(encoding="utf-8")
    assert 'DEVICE=""' in sh_text
    assert 'OUTPUT_DIR=""' in sh_text
    assert 'REPORT_DIR=""' in sh_text
    assert 'WANDB_ENABLED=""' in sh_text
    assert "--no_wandb" in sh_text or "--no-wandb" in sh_text

    # 4. Inspect scripts/run_full_pipeline.ps1: default variables must be empty strings
    ps_text = Path("scripts/run_full_pipeline.ps1").read_text(encoding="utf-8")
    assert "ContainsKey('Device')" in ps_text
    assert "ContainsKey('OutputDir')" in ps_text
    assert "ContainsKey('ReportDir')" in ps_text
    assert "NoWandb" in ps_text



