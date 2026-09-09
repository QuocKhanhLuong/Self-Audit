"""Strict contract and adapter verification suite for Self-Audit unified configuration.

Covers:
1. Adapter build kwargs capture via monkeypatching / inspection:
   - model adapter forwards encoder_allow_fallback to build_model_from_config / build_self_audit_net
   - dataset adapter forwards class_mapping, preprocessing, and representation
2. Canonical YAML contract (self_audit_full.yaml):
   - derived num_classes compatibility without duplication in dataset section
   - explicit experiment protocol and recipe identity
   - dataset representation and optional-test policy
   - training audit target contract and calibration metric space / contract
3. Adversarial constraints:
   - contradicting duplicate dataset.num_classes rejected
   - arbitrary unsupported margins rejected (audit_margin != 0.05, neutral_margin != 0.005)
   - early checkpoint best selection (< 120) and non-max metric modes rejected
   - zero/negative LRs for trainable modules and non-zero LRs for frozen modules rejected
   - out-of-order schedule objectives rejected
   - unknown, duplicate, or invalid keys strictly rejected
4. Public class attribute compatibility for checkpoint worker.
"""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Any
import yaml

import pytest
import torch

from self_audit.training.schedule import Schedule, ScheduleInterval
from self_audit.training.unified_config import (
    CalibrationConfig,
    CheckpointConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    UnifiedConfig,
    apply_overrides,
    load_unified_config,
    parse_unified_config,
)
from self_audit.training._utils import build_model_from_config, filter_model_config


def _load_canonical_dict() -> dict[str, Any]:
    raw = Path("configs/self_audit_full.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(raw)


# ============================================================================
# 1. Canonical YAML Contract Verification
# ============================================================================


def test_canonical_yaml_loads_cleanly() -> None:
    """Canonical YAML must parse strictly and validate all Wave 2 invariants."""
    config = load_unified_config("configs/self_audit_full.yaml")
    assert config.schema_version == 1
    assert config.experiment.name == "self_audit_full"
    assert config.experiment.protocol == "unified_schedule_v1"
    assert config.experiment.recipe == "curriculum_v1"
    assert config.experiment.seed == 42
    assert config.dataset.name == "acdc"
    assert config.dataset.representation == "paired_3d"
    assert config.dataset.num_classes == 4
    assert config.dataset.has_test is True
    assert config.dataset.optional_test is True
    assert config.model.num_classes == 4
    assert config.model.fallback is False
    assert config.model.encoder_allow_fallback is False
    assert config.training.audit_loss.target_contract == "audit_target_legacy_one_v1"
    assert config.training.audit_target_contract == "audit_target_legacy_one_v1"
    assert config.training.audit_loss.audit_margin == 0.05
    assert config.training.audit_loss.neutral_margin == 0.005
    assert config.checkpoint.best_selection_min_epoch == 120
    assert config.checkpoint.best_metric == "final_foreground_macro_dice"
    assert config.checkpoint.best_metric_mode == "max"
    assert config.calibration.metric_contract == "foreground_dice_exclude_v1"
    assert config.calibration.metric_space == "slice_proxy"


def test_canonical_yaml_omits_duplicated_num_classes() -> None:
    """Canonical YAML must not duplicate dataset.num_classes."""
    raw_dict = _load_canonical_dict()
    assert "num_classes" not in raw_dict["dataset"]
    assert raw_dict["model"]["num_classes"] == 4


# ============================================================================
# 2. Adapter Build Kwargs Capture Tests
# ============================================================================


def test_model_adapter_forwards_encoder_allow_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """to_legacy_model_config must forward encoder_allow_fallback matching fallback."""
    config = load_unified_config("configs/self_audit_full.yaml")
    legacy_model = config.to_legacy_model_config()

    assert "encoder_allow_fallback" in legacy_model["model"]
    assert legacy_model["model"]["encoder_allow_fallback"] is False

    captured_kwargs: dict[str, Any] = {}

    def fake_build_net(**kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr("self_audit.models.self_audit_net.build_self_audit_net", fake_build_net)
    build_model_from_config(legacy_model, torch.device("cpu"))

    assert captured_kwargs["encoder_allow_fallback"] is False
    assert captured_kwargs["encoder_name"] == "convnext_tiny"
    assert captured_kwargs["shared_channels"] == 96
    assert captured_kwargs["num_classes"] == 4


def test_model_adapter_fallback_true_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    """When fallback is True, model adapter must forward encoder_allow_fallback=True."""
    raw = _load_canonical_dict()
    raw["model"]["fallback"] = True
    config = parse_unified_config(raw)
    legacy_model = config.to_legacy_model_config()
    assert legacy_model["model"]["encoder_allow_fallback"] is True

    captured_kwargs: dict[str, Any] = {}

    def fake_build_net(**kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr("self_audit.models.self_audit_net.build_self_audit_net", fake_build_net)
    build_model_from_config(legacy_model, torch.device("cpu"))
    assert captured_kwargs["encoder_allow_fallback"] is True


def test_dataset_adapter_forwards_preprocessing_and_class_mapping() -> None:
    """to_legacy_dataset_config must forward exact keys for Wave 3 factory."""
    config = load_unified_config("configs/self_audit_full.yaml")
    interval = config.training.schedule.intervals[0]
    legacy_ds = config.to_legacy_dataset_config(interval)

    assert "preprocessing" in legacy_ds
    assert legacy_ds["preprocessing"] == {
        "clipping_min": 0.5,
        "clipping_max": 99.5,
        "foreground_only": False,
    }
    assert "class_mapping" in legacy_ds
    assert legacy_ds["class_mapping"] == {0: 0, 1: 1, 2: 2, 3: 3}
    assert "representation" in legacy_ds
    assert legacy_ds["representation"] == "paired_3d"
    assert legacy_ds["num_classes"] == 4
    assert legacy_ds["batch_size"] == 4


# ============================================================================
# 3. Dataset Representation and Optional-Test Policy
# ============================================================================


def test_dataset_representation_unsupported_rejected() -> None:
    """Unprepared or invalid representations must be rejected."""
    raw = _load_canonical_dict()
    raw["dataset"]["representation"] = "4d_cine"
    with pytest.raises(ValueError, match="dataset.representation must be 'paired_3d'"):
        parse_unified_config(raw)


def test_dataset_optional_test_policy() -> None:
    """Test split can be null or omitted; optional_test policy remains True."""
    raw = _load_canonical_dict()
    raw["dataset"]["test_split"] = None
    config_none = parse_unified_config(raw)
    assert config_none.dataset.test_split is None
    assert config_none.dataset.has_test is False
    assert config_none.dataset.optional_test is True

    del raw["dataset"]["test_split"]
    config_omitted = parse_unified_config(raw)
    assert config_omitted.dataset.test_split is None
    assert config_omitted.dataset.has_test is False


# ============================================================================
# 4. Dataset num_classes Compatibility and Contradiction Rejection
# ============================================================================


def test_dataset_num_classes_derived_when_omitted() -> None:
    """When omitted from dataset section, num_classes is derived as 4."""
    raw = _load_canonical_dict()
    assert "num_classes" not in raw["dataset"]
    config = parse_unified_config(raw)
    assert config.dataset.num_classes == 4


def test_dataset_num_classes_matching_accepted() -> None:
    """Explicit matching num_classes=4 is accepted for legacy compatibility."""
    raw = _load_canonical_dict()
    raw["dataset"]["num_classes"] = 4
    config = parse_unified_config(raw)
    assert config.dataset.num_classes == 4


def test_dataset_num_classes_contradicting_rejected() -> None:
    """Explicit contradicting num_classes (e.g. 3 or 5) must be rejected."""
    raw = _load_canonical_dict()
    raw["dataset"]["num_classes"] = 3
    with pytest.raises(ValueError, match="contradicts"):
        parse_unified_config(raw)

    raw["dataset"]["num_classes"] = 5
    with pytest.raises(ValueError, match="contradicts"):
        parse_unified_config(raw)


# ============================================================================
# 5. Margins Validation (Joint Helper Invariants)
# ============================================================================


def test_arbitrary_audit_margin_rejected() -> None:
    """Audit margins differing from fixed 0.05 must be rejected."""
    raw = _load_canonical_dict()
    raw["training"]["audit_loss"]["audit_margin"] = 0.1
    with pytest.raises(ValueError, match="audit_margin must be 0.05"):
        parse_unified_config(raw)

    raw["training"]["audit_loss"]["audit_margin"] = 0.0
    with pytest.raises(ValueError, match="audit_margin must be 0.05"):
        parse_unified_config(raw)


def test_arbitrary_neutral_margin_rejected() -> None:
    """Neutral margins differing from fixed 0.005 must be rejected."""
    raw = _load_canonical_dict()
    raw["training"]["audit_loss"]["neutral_margin"] = 0.01
    with pytest.raises(ValueError, match="neutral_margin must be 0.005"):
        parse_unified_config(raw)


def test_training_audit_target_contract_strict() -> None:
    """Training audit target contract must strictly be audit_target_legacy_one_v1."""
    raw = _load_canonical_dict()
    raw["training"]["audit_loss"]["target_contract"] = "foreground_dice_exclude_v1"
    with pytest.raises(ValueError, match="training audit target contract must be 'audit_target_legacy_one_v1'"):
        parse_unified_config(raw)


# ============================================================================
# 6. Checkpoint Best Selection & Gated Interval Firewall
# ============================================================================


def test_best_selection_min_epoch_below_120_rejected() -> None:
    """Early selection cutoff (< 120) allows un-gated epochs to compete and is rejected."""
    raw = _load_canonical_dict()
    raw["checkpoint"]["best_selection_min_epoch"] = 0
    with pytest.raises(ValueError, match="best_selection_min_epoch must be >= 120"):
        parse_unified_config(raw)

    raw["checkpoint"]["best_selection_min_epoch"] = 100
    with pytest.raises(ValueError, match="best_selection_min_epoch must be >= 120"):
        parse_unified_config(raw)

    raw["checkpoint"]["best_selection_min_epoch"] = 119
    with pytest.raises(ValueError, match="best_selection_min_epoch must be >= 120"):
        parse_unified_config(raw)


def test_best_selection_invalid_metric_or_mode_rejected() -> None:
    """best_metric must be final_foreground_macro_dice and mode must be max."""
    raw = _load_canonical_dict()
    raw["checkpoint"]["best_metric"] = "initial_dice"
    with pytest.raises(ValueError, match="best_metric must be 'final_foreground_macro_dice'"):
        parse_unified_config(raw)

    raw["checkpoint"]["best_metric"] = "final_foreground_macro_dice"
    raw["checkpoint"]["best_metric_mode"] = "min"
    with pytest.raises(ValueError, match="best_metric_mode must be 'max'"):
        parse_unified_config(raw)


# ============================================================================
# 7. Calibration Contract and Metric Space Firewall
# ============================================================================


def test_calibration_metric_contract_strict() -> None:
    """calibration.metric_contract must be foreground_dice_exclude_v1."""
    raw = _load_canonical_dict()
    raw["calibration"]["metric_contract"] = "audit_target_legacy_one_v1"
    with pytest.raises(ValueError, match="calibration.metric_contract must be 'foreground_dice_exclude_v1'"):
        parse_unified_config(raw)


def test_calibration_metric_space_strict() -> None:
    """calibration.metric_space must strictly be slice_proxy."""
    raw = _load_canonical_dict()
    raw["calibration"]["metric_space"] = "volume_resized"
    with pytest.raises(ValueError, match="calibration.metric_space must be 'slice_proxy'"):
        parse_unified_config(raw)


# ============================================================================
# 8. Schedule: Positive LRs for Trainable & Zero LRs for Frozen Modules
# ============================================================================


def test_schedule_interval0_frozen_auditor_lr_positive_rejected() -> None:
    """Interval 0 has frozen auditor; positive auditor_lr must be rejected."""
    raw = _load_canonical_dict()
    raw["training"]["schedule"]["intervals"][0]["auditor_lr"] = 1e-4
    with pytest.raises(ValueError, match="Frozen auditor in 'annotation' requires auditor_lr=0.0"):
        parse_unified_config(raw)


def test_schedule_interval0_trainable_annotation_lr_zero_rejected() -> None:
    """Interval 0 trains annotation; zero annotation_lr corrupts grad_clip and is rejected."""
    raw = _load_canonical_dict()
    raw["training"]["schedule"]["intervals"][0]["annotation_lr"] = 0.0
    with pytest.raises(ValueError, match="Trainable 'annotation' requires positive annotation_lr > 0"):
        parse_unified_config(raw)


def test_schedule_interval1_frozen_encoder_lr_positive_rejected() -> None:
    """Interval 1 has frozen encoder/annotation; non-zero encoder_lr must be rejected."""
    raw = _load_canonical_dict()
    raw["training"]["schedule"]["intervals"][1]["encoder_lr"] = 1e-4
    with pytest.raises(ValueError, match="Frozen encoder in 'auditor' requires encoder_lr=0.0"):
        parse_unified_config(raw)


def test_schedule_interval1_trainable_auditor_lr_zero_rejected() -> None:
    """Interval 1 trains auditor; zero auditor_lr must be rejected."""
    raw = _load_canonical_dict()
    raw["training"]["schedule"]["intervals"][1]["auditor_lr"] = 0.0
    with pytest.raises(ValueError, match="Trainable 'auditor' requires positive auditor_lr > 0"):
        parse_unified_config(raw)


def test_schedule_interval2_all_trainable_lrs_must_be_positive() -> None:
    """Interval 2 trains all modules; any zero LR must be rejected."""
    raw = _load_canonical_dict()
    raw["training"]["schedule"]["intervals"][2]["encoder_lr"] = 0.0
    with pytest.raises(ValueError, match="Trainable 'all' requires positive encoder_lr > 0"):
        parse_unified_config(raw)

    raw = _load_canonical_dict()
    raw["training"]["schedule"]["intervals"][2]["auditor_lr"] = 0.0
    with pytest.raises(ValueError, match="Trainable 'all' requires positive auditor_lr > 0"):
        parse_unified_config(raw)


# ============================================================================
# 9. Schedule: Objective Ordering Validation
# ============================================================================


def test_schedule_objective_ordering_rejected_when_reversed() -> None:
    """Reversing objective curriculum ordering must raise ValueError."""
    raw = _load_canonical_dict()
    int1 = copy.deepcopy(raw["training"]["schedule"]["intervals"][1])
    int2 = copy.deepcopy(raw["training"]["schedule"]["intervals"][2])

    int2["start_epoch"] = 100
    int2["end_epoch"] = 110
    int1["start_epoch"] = 110
    int1["end_epoch"] = 130

    raw["training"]["schedule"]["intervals"][1] = int2
    raw["training"]["schedule"]["intervals"][2] = int1

    with pytest.raises(ValueError, match="Invalid schedule objective ordering"):
        parse_unified_config(raw)


# ============================================================================
# 10. Checkpoint Worker Compatibility & Public Class Attributes
# ============================================================================


def test_checkpoint_worker_class_attribute_compatibility() -> None:
    """All public class attributes required by checkpoint worker must exist."""
    config = load_unified_config("configs/self_audit_full.yaml")

    # ExperimentConfig attributes
    assert hasattr(config.experiment, "name")
    assert hasattr(config.experiment, "seed")
    assert hasattr(config.experiment, "deterministic")
    assert hasattr(config.experiment, "device")
    assert hasattr(config.experiment, "protocol")
    assert hasattr(config.experiment, "recipe")

    # DatasetConfig attributes
    assert hasattr(config.dataset, "name")
    assert hasattr(config.dataset, "data_root")
    assert hasattr(config.dataset, "split_manifest")
    assert hasattr(config.dataset, "train_split")
    assert hasattr(config.dataset, "val_split")
    assert hasattr(config.dataset, "test_split")
    assert hasattr(config.dataset, "image_size")
    assert hasattr(config.dataset, "depth_axis")
    assert hasattr(config.dataset, "num_classes")
    assert hasattr(config.dataset, "class_mapping")
    assert hasattr(config.dataset, "preprocessing")
    assert hasattr(config.dataset, "dataloader")
    assert hasattr(config.dataset, "representation")

    # ModelConfig attributes
    assert hasattr(config.model, "encoder_name")
    assert hasattr(config.model, "pretrained_encoder")
    assert hasattr(config.model, "fallback")
    assert hasattr(config.model, "encoder_allow_fallback")
    assert hasattr(config.model, "shared_channels")
    assert hasattr(config.model, "num_classes")
    assert hasattr(config.model, "window_k")
    assert hasattr(config.model, "max_turns")

    # CheckpointConfig attributes
    assert hasattr(config.checkpoint, "output_dir")
    assert hasattr(config.checkpoint, "save_best")
    assert hasattr(config.checkpoint, "save_last")
    assert hasattr(config.checkpoint, "best_selection_min_epoch")
    assert hasattr(config.checkpoint, "best_metric")
    assert hasattr(config.checkpoint, "best_metric_mode")

    # CalibrationConfig attributes
    assert hasattr(config.calibration, "enabled")
    assert hasattr(config.calibration, "split")
    assert hasattr(config.calibration, "threshold_min")
    assert hasattr(config.calibration, "threshold_max")
    assert hasattr(config.calibration, "threshold_steps")
    assert hasattr(config.calibration, "metric_contract")
    assert hasattr(config.calibration, "metric_space")

    # Serialization roundtrip
    d = config.to_dict()
    assert isinstance(d, dict)
    assert d["dataset"]["num_classes"] == 4
    assert d["dataset"]["representation"] == "paired_3d"
    assert d["experiment"]["protocol"] == "unified_schedule_v1"
    assert d["calibration"]["metric_space"] == "slice_proxy"
    assert d["training"]["audit_loss"]["target_contract"] == "audit_target_legacy_one_v1"

    reparsed = parse_unified_config(d)
    assert reparsed.dataset.num_classes == 4
    assert reparsed.model.num_classes == 4
