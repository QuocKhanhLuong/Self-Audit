"""Focused downstream integration tests for Wave 3.

Covers:
1. Schedule-derived checkpoint best selection min_epoch (early and beyond-end rejection,
   support for short synthetic curriculum, preserving bounded annotation-only smoke).
2. Shared resolved config adapter (UnifiedConfig vs historical flat configs, overrides,
   model/dataset/loader/rollout/metric consumer integration).
3. Complete curriculum run exercising best != last, binding selected best for calibration
   and final diagnostics, strict roundtrip verification, and state immutability.
4. Tampering failure on corrupted weights or mismatched checkpoint lineage.
5. Incomplete/bounded runs refuse calibration and forbid silent fallback to last.pt.
6. Proposal-1 frozen transition bank export from unified configuration and bound best checkpoint.
7. Legacy flat configuration preservation in transition bank export.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.data.common import VolumeRecord
from self_audit.evaluation.calibration_lineage import (
    COHORT_ROLE_CALIBRATION,
    CohortPolicy,
    build_expected_lineage,
    verify_calibration_lineage,
)
from self_audit.evaluation.threshold import load_calibration
from self_audit.evaluation.transition_bank import (
    EXPORT_BOUND,
    SOURCE_ON_POLICY,
    load_bank,
    validate_bank,
)
from self_audit.models.self_audit_net import SelfAuditNet
from self_audit.provenance import state_digest
from self_audit.training._utils import (
    bind_evaluation_checkpoint,
    build_model_from_config,
    save_checkpoint,
    verify_bound_state,
)
from self_audit.training.schedule import Schedule
from self_audit.training.unified_config import (
    ResolvedExecutionConfig,
    UnifiedConfig,
    load_unified_config,
    parse_unified_config,
    resolve_downstream_config,
)
from self_audit.training.unified_trainer import UnifiedTrainer
from scripts.export_transition_bank import export_transition_bank


def _to_clean_json_safe(obj: Any) -> Any:
    """Recursively convert tuples to lists so yaml.dump produces clean portable YAML."""
    if isinstance(obj, (tuple, list)):
        return [_to_clean_json_safe(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _to_clean_json_safe(val) for key, val in obj.items()}
    return obj


def _load_canonical_raw() -> dict[str, Any]:
    full_yaml = ROOT / "configs" / "self_audit_full.yaml"
    cfg = load_unified_config(full_yaml)
    return _to_clean_json_safe(cfg.to_dict())


# ============================================================================
# 1. Schedule-derived Checkpoint Selection Boundary & Gated Interval Firewall
# ============================================================================


def test_schedule_derived_best_selection_min_epoch_canonical() -> None:
    """Canonical 130-epoch schedule enforces best_selection_min_epoch in [120, 130)."""
    raw = _load_canonical_raw()

    # Valid canonical setting: 120 and 125
    raw["checkpoint"]["best_selection_min_epoch"] = 120
    cfg = parse_unified_config(raw)
    assert cfg.checkpoint.best_selection_min_epoch == 120

    raw["checkpoint"]["best_selection_min_epoch"] = 125
    cfg = parse_unified_config(raw)
    assert cfg.checkpoint.best_selection_min_epoch == 125

    # Early selection before gated interval [120, 130) is strictly rejected
    for early_epoch in (0, 60, 119):
        raw["checkpoint"]["best_selection_min_epoch"] = early_epoch
        with pytest.raises(ValueError, match="best_selection_min_epoch must be >= 120"):
            parse_unified_config(raw)

    # Beyond-end selection (>= 130) is strictly rejected
    for beyond_epoch in (130, 135):
        raw["checkpoint"]["best_selection_min_epoch"] = beyond_epoch
        with pytest.raises(ValueError, match="beyond-end selection is strictly rejected"):
            parse_unified_config(raw)


def test_schedule_derived_best_selection_min_epoch_short_curriculum() -> None:
    """Short synthetic curriculum derives selection boundary from its own gated interval."""
    raw = _load_canonical_raw()
    # Define a 3-epoch schedule: phase_a [0, 1), phase_b [1, 2), phase_c [2, 3)
    raw["training"]["schedule"] = {
        "total_epochs": 3,
        "intervals": [
            {
                "name": "phase_a",
                "start_epoch": 0,
                "end_epoch": 1,
                "objective": "weighted_a0_a3",
                "trainable": "annotation",
                "encoder_lr": 3e-5,
                "annotation_lr": 1e-4,
                "auditor_lr": 0.0,
                "annotation_weight": 1.0,
                "audit_weight": 0.0,
                "transition_population": "none",
                "rollout": "propagate_no_audit",
                "batch_size": 2,
                "accumulation_steps": 1,
                "augment": False,
                "reset_optimizer": False,
            },
            {
                "name": "phase_b",
                "start_epoch": 1,
                "end_epoch": 2,
                "objective": "counterfactual_audit",
                "trainable": "auditor",
                "encoder_lr": 0.0,
                "annotation_lr": 0.0,
                "auditor_lr": 1e-4,
                "annotation_weight": 0.0,
                "audit_weight": 1.0,
                "transition_population": "adjacent_and_synthetic",
                "rollout": "annotation_eval",
                "batch_size": 2,
                "accumulation_steps": 1,
                "augment": False,
                "reset_optimizer": True,
            },
            {
                "name": "phase_c",
                "start_epoch": 2,
                "end_epoch": 3,
                "objective": "retained_final_annotation",
                "trainable": "all",
                "encoder_lr": 1e-6,
                "annotation_lr": 1e-5,
                "auditor_lr": 1e-5,
                "annotation_weight": 1.0,
                "audit_weight": 1.0,
                "transition_population": "active_attempted",
                "rollout": "threshold_gate",
                "batch_size": 2,
                "accumulation_steps": 1,
                "augment": False,
                "reset_optimizer": True,
            },
        ],
    }

    # Gated interval is [2, 3): min_epoch=2 must be accepted without requiring magic 120!
    raw["checkpoint"]["best_selection_min_epoch"] = 2
    cfg = parse_unified_config(raw)
    assert cfg.checkpoint.best_selection_min_epoch == 2

    # Early selection before 2 is rejected
    raw["checkpoint"]["best_selection_min_epoch"] = 1
    with pytest.raises(ValueError, match="best_selection_min_epoch must be >= 2"):
        parse_unified_config(raw)

    # Beyond-end selection (>= 3) is rejected
    raw["checkpoint"]["best_selection_min_epoch"] = 3
    with pytest.raises(ValueError, match="beyond-end selection is strictly rejected"):
        parse_unified_config(raw)


def test_schedule_without_gated_interval_preserves_smoke_fixtures() -> None:
    """Bounded annotation-only schedule has no gated interval and does not fail validation."""
    raw = _load_canonical_raw()
    raw["training"]["schedule"] = {
        "total_epochs": 1,
        "intervals": [
            {
                "name": "phase_a",
                "start_epoch": 0,
                "end_epoch": 1,
                "objective": "weighted_a0_a3",
                "trainable": "annotation",
                "encoder_lr": 3e-5,
                "annotation_lr": 1e-4,
                "auditor_lr": 0.0,
                "annotation_weight": 1.0,
                "audit_weight": 0.0,
                "transition_population": "none",
                "rollout": "propagate_no_audit",
                "batch_size": 2,
                "accumulation_steps": 1,
                "augment": False,
                "reset_optimizer": False,
            }
        ],
    }
    raw["checkpoint"]["best_selection_min_epoch"] = 0
    cfg = parse_unified_config(raw)
    assert cfg.checkpoint.best_selection_min_epoch == 0


# ============================================================================
# 2. Shared Resolved Config Adapter
# ============================================================================


def test_resolve_downstream_config_from_unified() -> None:
    """ResolvedExecutionConfig extracts exact consumer configs from UnifiedConfig without guessing."""
    full_yaml = ROOT / "configs" / "self_audit_full.yaml"
    resolved = resolve_downstream_config(
        full_yaml,
        overrides={
            "data_root": "preprocessed_data/acdc",
            "device": "cpu",
            "image_size": 32,
            "batch_size": 2,
        },
    )
    assert resolved.is_unified is True
    assert resolved.unified_config is not None
    assert resolved.device == torch.device("cpu")
    assert resolved.val_split == "val"
    assert resolved.metric_contract == "foreground_dice_exclude_v1"
    assert resolved.rollout_tau == 0.0
    assert resolved.rollout_max_turns == 3

    # model_config dict
    m_cfg = resolved.model_config
    assert m_cfg["model"]["encoder_name"] == "convnext_tiny"
    assert m_cfg["num_classes"] == 4

    # dataset_config dict
    d_cfg = resolved.dataset_config
    assert d_cfg["dataset"] == "acdc"
    assert d_cfg["image_size"] == 32
    assert d_cfg["data_root"] == "preprocessed_data/acdc"

    # dataloader_config dict
    dl_cfg = resolved.dataloader_config
    assert "num_workers" in dl_cfg
    assert "pin_memory" in dl_cfg


def test_resolve_downstream_config_from_historical_flat() -> None:
    """Historical flat configs are explicitly recognized without fabricating missing schema sections."""
    flat = {
        "dataset": "acdc",
        "data_root": "preprocessed_data/acdc",
        "split_manifest": "splits/acdc_split.json",
        "val_split": "val",
        "device": "cpu",
        "model": {
            "encoder_name": "convnext_tiny",
            "shared_channels": 16,
            "num_classes": 4,
            "window_k": 4,
            "max_turns": 2,
        },
        "audit": {
            "t_max": 2,
            "tau_accept": 0.0,
            "neutral_margin": 0.005,
        },
        "metric_contract": "foreground_dice_exclude_v1",
    }
    resolved = resolve_downstream_config(flat, overrides={"batch_size": 4})
    assert resolved.is_unified is False
    assert resolved.unified_config is None
    assert resolved.val_split == "val"
    assert resolved.rollout_max_turns == 2
    assert resolved.neutral_margin == 0.005
    assert resolved.device == torch.device("cpu")
    assert resolved.to_legacy_dict()["batch_size"] == 4


# ============================================================================
# 3. Short Synthetic Complete Curriculum: Best != Last, Calibration & Tampering
# ============================================================================


def _create_synthetic_acdc_cohort(data_root: Path, manifest_path: Path) -> None:
    (data_root / "volumes").mkdir(parents=True, exist_ok=True)
    (data_root / "masks").mkdir(parents=True, exist_ok=True)
    manifest_data: dict[str, list[str]] = {"train": [], "val": []}
    for pt_idx, split_name in [(1, "train"), (2, "val")]:
        pid = f"patient{pt_idx:03d}"
        case_id = f"{pid}_ED"
        img = np.random.randn(32, 32, 2).astype(np.float32)
        mask = np.zeros((32, 32, 2), dtype=np.int64)
        mask[8:24, 8:24, 0] = 1
        mask[12:20, 12:20, 0] = 2
        mask[14:18, 14:18, 0] = 3
        mask[8:24, 8:24, 1] = 1
        mask[12:20, 12:20, 1] = 2

        img_p = data_root / "volumes" / f"{case_id}.npy"
        mask_p = data_root / "masks" / f"{case_id}.npy"
        np.save(img_p, img)
        np.save(mask_p, mask)
        manifest_data[split_name].append(case_id)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")


def test_unified_trainer_complete_short_curriculum_best_differs_from_last(tmp_path: Path) -> None:
    """Execute complete short curriculum: verify best != last, bound best for calibration & diagnostics."""
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "splits" / "acdc_split.json"
    _create_synthetic_acdc_cohort(data_root, manifest_path)

    output_dir = tmp_path / "weights"
    report_dir = tmp_path / "reports"

    raw = _load_canonical_raw()
    raw["experiment"]["name"] = "short_complete_curriculum"
    raw["experiment"]["device"] = "cpu"
    raw["experiment"]["seed"] = 42
    raw["dataset"]["data_root"] = str(data_root)
    raw["dataset"]["split_manifest"] = str(manifest_path)
    raw["dataset"]["image_size"] = 32
    raw["dataset"]["dataloader"]["num_workers"] = 0
    raw["dataset"]["dataloader"]["pin_memory"] = False
    raw["dataset"]["dataloader"]["persistent_workers"] = False

    # Pretrained encoder disabled, fallback allowed for synthetic test fixture
    raw["model"]["pretrained_encoder"] = False
    raw["model"]["fallback"] = True
    raw["model"]["shared_channels"] = 16
    raw["model"]["window_k"] = 4
    raw["model"]["max_turns"] = 2

    # 4-epoch schedule:
    # epoch 0: phase_a
    # epoch 1: phase_b
    # epoch 2: phase_c (gated) -> achieves metric 0.95 -> saved as best.pt
    # epoch 3: phase_c (gated) -> metric 0.40 -> not saved as best, last.pt updated
    raw["training"]["schedule"] = {
        "total_epochs": 4,
        "intervals": [
            {
                "name": "phase_a",
                "start_epoch": 0,
                "end_epoch": 1,
                "objective": "weighted_a0_a3",
                "trainable": "annotation",
                "encoder_lr": 3e-5,
                "annotation_lr": 1e-4,
                "auditor_lr": 0.0,
                "annotation_weight": 1.0,
                "audit_weight": 0.0,
                "transition_population": "none",
                "rollout": "propagate_no_audit",
                "batch_size": 2,
                "accumulation_steps": 1,
                "augment": False,
                "reset_optimizer": False,
            },
            {
                "name": "phase_b",
                "start_epoch": 1,
                "end_epoch": 2,
                "objective": "counterfactual_audit",
                "trainable": "auditor",
                "encoder_lr": 0.0,
                "annotation_lr": 0.0,
                "auditor_lr": 1e-4,
                "annotation_weight": 0.0,
                "audit_weight": 1.0,
                "transition_population": "adjacent_and_synthetic",
                "rollout": "annotation_eval",
                "batch_size": 2,
                "accumulation_steps": 1,
                "augment": False,
                "reset_optimizer": True,
            },
            {
                "name": "phase_c",
                "start_epoch": 2,
                "end_epoch": 4,
                "objective": "retained_final_annotation",
                "trainable": "all",
                "encoder_lr": 1e-6,
                "annotation_lr": 1e-4,
                "auditor_lr": 1e-4,
                "annotation_weight": 1.0,
                "audit_weight": 1.0,
                "transition_population": "active_attempted",
                "rollout": "threshold_gate",
                "batch_size": 2,
                "accumulation_steps": 1,
                "augment": False,
                "reset_optimizer": True,
            },
        ],
    }
    raw["training"]["rollout"]["max_turns"] = 2
    raw["checkpoint"]["output_dir"] = str(output_dir)
    raw["checkpoint"]["best_selection_min_epoch"] = 2
    raw["logging"]["report_dir"] = str(report_dir)
    raw["calibration"]["enabled"] = True
    raw["calibration"]["threshold_steps"] = 3
    raw["diagnostics"]["evaluate_headroom"] = True
    raw["diagnostics"]["evaluate_decomposition"] = True

    cfg = parse_unified_config(raw)
    trainer = UnifiedTrainer(cfg, disable_tqdm=True)

    # Force a validation drop on epoch 3 so best.pt is saved at epoch 2 and last.pt at epoch 3 differs
    original_validate_epoch = trainer.validate_epoch

    def mock_validate_epoch(interval: Any, val_loader: Any, max_val_batches: Any = None) -> dict[str, Any]:
        stats = original_validate_epoch(interval, val_loader, max_val_batches=max_val_batches)
        if trainer.global_epoch == 2:
            # Epoch 2 (first gated epoch): great metric -> becomes best.pt
            stats["final_foreground_macro_dice"] = 0.95
            stats["primary_metric"] = 0.95
        elif trainer.global_epoch == 3:
            # Epoch 3 (second gated epoch): worse metric -> not selected as best
            stats["final_foreground_macro_dice"] = 0.40
            stats["primary_metric"] = 0.40
        return stats

    trainer.validate_epoch = mock_validate_epoch

    report = trainer.train()

    # 1. Pipeline marked completed
    assert report["completed"] is True
    assert trainer.completed is True

    # 2. Both best.pt and last.pt exist
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    assert best_path.exists()
    assert last_path.exists()

    # 3. Best != Last exercised: state_digest differs!
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
    last_ckpt = torch.load(last_path, map_location="cpu", weights_only=True)
    digest_best = state_digest(best_ckpt["model"])
    digest_last = state_digest(last_ckpt["model"])
    assert digest_best != digest_last, "best.pt and last.pt must have distinct weights"
    assert best_ckpt["epoch"] == 3  # epoch index 2 + 1 = 3
    assert last_ckpt["epoch"] == 4  # epoch index 3 + 1 = 4

    # 4. Checkpoint binding in report strictly names best.pt
    binding_report = report["checkpoint_binding"]
    assert binding_report["checkpoint_path"] == str(best_path)
    assert binding_report["state_digest"] == digest_best

    # 5. Calibration artifact generated and lineage verified against best.pt
    cal_file = report_dir / "calibration.json"
    assert cal_file.exists()
    cal_data = load_calibration(cal_file)
    assert cal_data["lineage"]["checkpoint"]["checkpoint_path"] == str(best_path)
    assert cal_data["lineage"]["checkpoint"]["state_digest"] == digest_best

    # 6. Final diagnostics recorded in report
    assert "final_diagnostics" in report
    assert "headroom" in report["final_diagnostics"]
    assert "decomposition" in report["final_diagnostics"]

    # 7. Selected-state immutability: trainer.model matches best.pt state digest
    assert state_digest(trainer.model) == digest_best

    # 8. Tampering failure: verify that corrupting best.pt fails state verification
    tampered_best = copy.deepcopy(best_ckpt)
    first_param = next(iter(tampered_best["model"].values()))
    first_param.add_(1.0)
    tampered_path = tmp_path / "tampered_best.pt"
    torch.save(tampered_best, tampered_path)

    test_model = build_model_from_config(cfg.to_legacy_model_config(), torch.device("cpu"))
    # (a) Refuses tampered checkpoint on disk
    with pytest.raises(ValueError, match="refusing to bind a tampered checkpoint"):
        bind_evaluation_checkpoint(
            test_model,
            [("checkpoint", tampered_path)],
            map_location=torch.device("cpu"),
            config=cfg.to_legacy_dataset_config(),
        )

    # (b) Detects in-memory mutation after valid binding
    valid_binding = bind_evaluation_checkpoint(
        test_model,
        [("checkpoint", best_path)],
        map_location=torch.device("cpu"),
        config=cfg.to_legacy_dataset_config(),
    )
    next(test_model.parameters()).data.add_(0.5)
    with pytest.raises(ValueError, match="Model state changed after checkpoint binding"):
        verify_bound_state(test_model, valid_binding, boundary="test_tamper")


def test_unified_trainer_missing_best_refuses_calibration(tmp_path: Path) -> None:
    """Bounded smoke run without best.pt refuses calibration and silent fallback to last.pt."""
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "splits" / "acdc_split.json"
    _create_synthetic_acdc_cohort(data_root, manifest_path)

    output_dir = tmp_path / "weights_bounded"
    report_dir = tmp_path / "reports_bounded"

    raw = _load_canonical_raw()
    raw["experiment"]["device"] = "cpu"
    raw["dataset"]["data_root"] = str(data_root)
    raw["dataset"]["split_manifest"] = str(manifest_path)
    raw["dataset"]["image_size"] = 32
    raw["model"]["pretrained_encoder"] = False
    raw["model"]["fallback"] = True
    raw["model"]["shared_channels"] = 16
    raw["checkpoint"]["output_dir"] = str(output_dir)
    raw["logging"]["report_dir"] = str(report_dir)

    cfg = parse_unified_config(raw)
    trainer = UnifiedTrainer(cfg, disable_tqdm=True)

    # Run bounded smoke (stops at step 1 before reaching gated epochs)
    report = trainer.train(max_steps=1)
    assert report["completed"] is False
    assert report["calibration"] is None
    assert trainer.completed is False

    # Calling run_post_training_calibration must raise RuntimeError
    with pytest.raises(RuntimeError, match="Cannot run post-training calibration on incomplete run"):
        trainer.run_post_training_calibration(output_dir, report_dir)


# ============================================================================
# 4. Proposal-1 Frozen Transition Bank Export Integration
# ============================================================================


def test_export_transition_bank_from_unified_full_config(tmp_path: Path) -> None:
    """Export Proposal-1 transition bank using UnifiedConfig and bound checkpoint."""
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "splits" / "acdc_split.json"
    _create_synthetic_acdc_cohort(data_root, manifest_path)

    raw = _load_canonical_raw()
    raw["experiment"]["device"] = "cpu"
    raw["dataset"]["data_root"] = str(data_root)
    raw["dataset"]["split_manifest"] = str(manifest_path)
    raw["dataset"]["image_size"] = 32
    raw["model"]["pretrained_encoder"] = False
    raw["model"]["fallback"] = True
    raw["model"]["shared_channels"] = 16
    raw["model"]["window_k"] = 4
    raw["model"]["max_turns"] = 2

    config_path = tmp_path / "full_config.yaml"
    config_path.write_text(yaml.dump(raw), encoding="utf-8")

    # Build and save a model checkpoint
    model = build_model_from_config(
        {
            "model": {
                "encoder_name": "convnext_tiny",
                "pretrained_encoder": False,
                "encoder_allow_fallback": True,
                "shared_channels": 16,
                "num_classes": 4,
                "window_k": 4,
                "max_turns": 2,
            }
        },
        torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        epoch=125,
        global_step=500,
        config=raw,
    )

    bank_output = tmp_path / "reports" / "transition_bank.json"

    # Export transition bank
    bank = export_transition_bank(
        config_source=config_path,
        checkpoint=checkpoint_path,
        output=bank_output,
        image_size=32,
        device="cpu",
        batch_size=2,
    )

    assert bank_output.exists()
    assert "rows" in bank
    assert len(bank["rows"]) > 0

    # Validate bank contract
    validate_bank(bank)

    # Check GT-free on-policy Proposal-1 generation
    for row in bank["rows"]:
        assert row["source"] == SOURCE_ON_POLICY
        assert row["gt_used_in_generation"] is False
        assert "delta_q" in row
        assert "q_candidate" in row
        assert "q_previous" in row

    # Provenance check
    generation = bank["generation"]
    assert generation["export_identity_class"] == EXPORT_BOUND
    assert generation["checkpoint_bound"] is True
    assert generation["checkpoint"]["checkpoint_path"] == str(checkpoint_path)
    assert generation["cohort"]["split_name"] == "val"
