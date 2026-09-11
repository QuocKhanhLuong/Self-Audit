from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from self_audit.data.common import VolumeRecord, VolumeSliceDataset, load_array
from self_audit.data.mnms import (
    DEFAULT_MNMS_TO_ACDC,
    EXTERNAL_MNMS_PROTOCOL,
    NATIVE_MNMS_PROTOCOL,
    MNMSClassMapping,
    MNMSDataset,
    discover_mnms_records,
    is_mnms_binary_path,
)
from self_audit.training._utils import (
    build_model_from_config,
    build_patient_dataset,
    save_checkpoint,
    validate_dataset_splits,
)
from self_audit.training.unified_config import load_unified_config


def _populate_mnms_case(
    split_dir: Path,
    case_id: str,
    *,
    shape: tuple[int, int, int] = (16, 16, 4),
    labels: tuple[int, ...] = (0, 1, 2, 3),
) -> tuple[Path, Path]:
    vol_dir = split_dir / "volumes"
    mask_dir = split_dir / "masks"
    vol_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    vol = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    for idx, lbl in enumerate(labels):
        if idx == 0:
            continue
        start = (idx - 1) * 3
        mask[start : start + 3, start : start + 3, :] = lbl
    vol_path = vol_dir / f"{case_id}.npy"
    mask_path = mask_dir / f"{case_id}.npy"
    np.save(vol_path, vol)
    np.save(mask_path, mask)
    return vol_path, mask_path


# =========================================================================
# 1. Scientific Protocol Labels & Legitimate Multi-Substr Path Loading
# =========================================================================


def test_scientific_protocol_labels() -> None:
    assert NATIVE_MNMS_PROTOCOL == "mnms_unified_native_v1"
    assert EXTERNAL_MNMS_PROTOCOL == "acdc_frozen_to_mnms_external_v1"
    assert MNMSDataset.scientific_protocol_label == NATIVE_MNMS_PROTOCOL
    assert MNMSDataset.external_protocol_label == EXTERNAL_MNMS_PROTOCOL


def test_legitimate_mnms_loading_under_acdc_and_mnms_path(tmp_path: Path) -> None:
    """Loading legitimate M&Ms under path containing both 'acdc' and 'mnms' with patient ID 'patient001'."""
    dual_root = tmp_path / "test_acdc_to_mnms" / "mnm"
    _populate_mnms_case(dual_root / "testing", "patient001_t00", shape=(16, 16, 4))

    records = discover_mnms_records(dual_root, split="testing")
    assert len(records) == 1
    assert records[0].patient_id == "patient001"

    # MNMSDataset loads correctly
    ds = MNMSDataset(data_root=dual_root, split="testing", depth_axis=2)
    assert len(ds) == 4
    vol, mask, _ = ds.get_volume("patient001_t00")
    assert vol.shape == (4, 16, 16)
    assert mask.shape == (4, 16, 16)

    # Factory build_patient_dataset loads correctly
    p_ds = build_patient_dataset(
        {"dataset": "mnms", "data_root": str(dual_root), "depth_axis": 2},
        split="testing",
        train=False,
    )
    assert len(p_ds) == 4


# =========================================================================
# 2. Four-Class Contract, Label Mapping & Fail Unknown Raw Labels
# =========================================================================


def test_four_class_mapping_contract() -> None:
    # Locked semantic mapping: raw {0:0, 1:3, 2:2, 3:1}
    mapping = MNMSClassMapping(DEFAULT_MNMS_TO_ACDC)
    assert mapping.raw_to_acdc == {0: 0, 1: 3, 2: 2, 3: 1}

    # Verify semantic assignment: raw 1 -> ACDC 3 (LV), raw 2 -> ACDC 2 (MYO), raw 3 -> ACDC 1 (RV)
    raw = np.array([0, 1, 2, 3], dtype=np.int64)
    mapped = mapping.apply(raw)
    np.testing.assert_array_equal(mapped, np.array([0, 3, 2, 1], dtype=np.int64))


def test_mapping_fails_unknown_raw_labels() -> None:
    mapping = MNMSClassMapping(DEFAULT_MNMS_TO_ACDC)
    bad_labels = np.array([0, 1, 2, 4], dtype=np.int64)
    with pytest.raises(ValueError, match="without an explicit class mapping"):
        mapping.apply(bad_labels)

    bad_label_9 = np.array([0, 9], dtype=np.int64)
    with pytest.raises(ValueError, match="without an explicit class mapping"):
        mapping.apply(bad_label_9)


def test_mapping_rejects_fractional_and_boolean_types() -> None:
    # Boolean rejected
    with pytest.raises(TypeError, match="cannot be booleans"):
        MNMSClassMapping({True: 0, 1: 3, 2: 2, 3: 1})

    # Fractional key/value rejected
    with pytest.raises(ValueError, match="cannot be fractional|must be an integer"):
        MNMSClassMapping({0: 0, 1.2: 3, 2: 2, 3: 1})

    # Fractional mask labels rejected before cast
    mapping = MNMSClassMapping(DEFAULT_MNMS_TO_ACDC)
    with pytest.raises(ValueError, match="finite integers"):
        mapping.apply(np.array([0.0, 1.5, 2.0]))

    # Non-finite mask labels (NaN) rejected
    with pytest.raises(ValueError, match="finite integers"):
        mapping.apply(np.array([0.0, np.nan, 2.0]))


# =========================================================================
# 3. Binary Derivative Dataset Refusal Across All Entrypoints
# =========================================================================


def test_binary_dataset_refusal(tmp_path: Path) -> None:
    bin_dir = tmp_path / "mnm_binary"
    _populate_mnms_case(bin_dir / "testing", "A001_t00")

    # is_mnms_binary_path
    assert is_mnms_binary_path(bin_dir) is True

    # discover_mnms_records
    with pytest.raises(ValueError, match="binary derivative"):
        discover_mnms_records(bin_dir, split="testing")

    # MNMSDataset
    with pytest.raises(ValueError, match="binary derivative"):
        MNMSDataset(data_root=bin_dir, split="testing")

    # build_patient_dataset
    with pytest.raises(ValueError, match="binary derivative"):
        build_patient_dataset({"dataset": "mnms", "data_root": str(bin_dir)}, split="testing", train=False)

    # validate_dataset_splits
    with pytest.raises(ValueError, match="binary derivative"):
        validate_dataset_splits({"dataset": "mnms", "data_root": str(bin_dir), "test_split": "testing"})


# =========================================================================
# 4. Correct Depth Axis & Paired Image/Mask Integrity
# =========================================================================


def test_depth_axis_and_geometry_normalization(tmp_path: Path) -> None:
    # M&Ms preprocessed shape [H=16, W=16, Z=4] with depth_axis=2
    _populate_mnms_case(tmp_path / "testing", "A001_t00", shape=(16, 16, 4))

    ds = MNMSDataset(data_root=tmp_path, split="testing", image_size=16, depth_axis=2)
    assert len(ds) == 4  # 4 slices along Z

    sample = ds[0]
    # Sample image should be 2.5-D triplet [3, H, W]
    assert sample["image"].shape == (3, 16, 16)
    # Target mask should be [H, W]
    assert sample["mask"].shape == (16, 16)

    # get_volume returns normalized [Z, H, W] volume and mask
    vol, mask, spacing = ds.get_volume("A001_t00")
    assert vol.shape == (4, 16, 16)
    assert mask.shape == (4, 16, 16)


def test_shape_mismatch_fails_closed(tmp_path: Path) -> None:
    split_dir = tmp_path / "testing"
    vol_dir = split_dir / "volumes"
    mask_dir = split_dir / "masks"
    vol_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    # Volume has 4 slices, mask has 2 slices -> shape mismatch
    np.save(vol_dir / "A001_t00.npy", np.zeros((16, 16, 4), dtype=np.float32))
    np.save(mask_dir / "A001_t00.npy", np.zeros((16, 16, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="shape mismatch"):
        validate_dataset_splits({
            "dataset": "mnms",
            "data_root": str(tmp_path),
            "test_split": "testing",
            "depth_axis": 2,
        })


def test_missing_paired_file_fails_closed(tmp_path: Path) -> None:
    split_dir = tmp_path / "testing"
    vol_dir = split_dir / "volumes"
    vol_dir.mkdir(parents=True)
    np.save(vol_dir / "A001_t00.npy", np.zeros((16, 16, 4), dtype=np.float32))
    # No masks directory or mask file

    with pytest.raises(FileNotFoundError, match="No paired M&Ms"):
        discover_mnms_records(tmp_path, split="testing")


# =========================================================================
# 5. Patient-Level Disjoint Splits & Leakage Prevention
# =========================================================================


def test_patient_level_disjoint_splits(tmp_path: Path) -> None:
    _populate_mnms_case(tmp_path / "train", "A001_t00")
    _populate_mnms_case(tmp_path / "val", "A002_t00")
    _populate_mnms_case(tmp_path / "test", "A003_t00")

    cfg = {
        "dataset": "mnms",
        "data_root": str(tmp_path),
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "depth_axis": 2,
    }
    validation = validate_dataset_splits(cfg)
    assert validation["validated"] is True
    assert validation["case_counts"] == {"train": 1, "val": 1, "test": 1}
    assert validation["patient_counts"] == {"train": 1, "val": 1, "test": 1}


def test_patient_leakage_rejected(tmp_path: Path) -> None:
    # Same patient A001 in both train and val
    _populate_mnms_case(tmp_path / "train", "A001_t00")
    _populate_mnms_case(tmp_path / "val", "A001_t01")

    cfg = {
        "dataset": "mnms",
        "data_root": str(tmp_path),
        "train_split": "train",
        "val_split": "val",
        "depth_axis": 2,
    }
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_dataset_splits(cfg)


# =========================================================================
# 6. Candidate C Canonical Config Parity & Window Mode Support
# =========================================================================


def test_canonical_mnms_config_model_parity() -> None:
    cfg = load_unified_config("configs/self_audit_full_mnms.yaml")
    assert cfg.schema_version == 1
    assert cfg.dataset.name == "mnms"
    assert cfg.dataset.depth_axis == 2
    assert cfg.dataset.class_mapping == {0: 0, 1: 3, 2: 2, 3: 1}

    # Model configuration has canonical defaults
    assert cfg.model.encoder_name == "convnext_tiny"
    assert cfg.model.shared_channels == 96
    assert cfg.model.num_classes == 4
    assert cfg.model.window_k == 8
    assert cfg.model.max_turns == 3
    assert cfg.model.window_mode == "current"

    # Candidate C solver settings carry standard defaults
    assert cfg.model.candidate_c.rho_feature_pixels == 1.0
    assert cfg.model.candidate_c.lam == 1.0
    assert cfg.model.candidate_c.fix_threshold == 0.5
    assert cfg.model.candidate_c.regress_threshold == 0.5
    assert cfg.model.candidate_c.max_backtracks == 2


def test_candidate_c_model_construction_parity() -> None:
    # Verify both acdc and mnms build identical model architectures under Candidate C settings
    acdc_cfg = {
        "model": {
            "encoder_name": "convnext_tiny",
            "pretrained_encoder": False,
            "encoder_allow_fallback": True,
            "shared_channels": 16,
            "num_classes": 4,
            "window_k": 4,
            "max_turns": 2,
            "window_mode": "candidate_c",
            "candidate_c": {"rho_feature_pixels": 1.0, "lam": 1.0},
        },
        "dataset": {"name": "acdc"}
    }
    mnms_cfg = {
        "model": {
            "encoder_name": "convnext_tiny",
            "pretrained_encoder": False,
            "encoder_allow_fallback": True,
            "shared_channels": 16,
            "num_classes": 4,
            "window_k": 4,
            "max_turns": 2,
            "window_mode": "candidate_c",
            "candidate_c": {"rho_feature_pixels": 1.0, "lam": 1.0},
        },
        "dataset": {"name": "mnms"}
    }
    m_acdc = build_model_from_config(acdc_cfg, torch.device("cpu"))
    m_mnms = build_model_from_config(mnms_cfg, torch.device("cpu"))

    # Parameter counts match exactly (no dataset-specific model code)
    p_acdc = sum(p.numel() for p in m_acdc.parameters())
    p_mnms = sum(p.numel() for p in m_mnms.parameters())
    assert p_acdc == p_mnms


# =========================================================================
# 7. External Evaluation Protocol & Strict Checkpoint Binding
# =========================================================================


def test_external_evaluator_rejects_mnms_checkpoint(tmp_path: Path) -> None:
    from scripts.evaluate_external_mnms import _inspect_checkpoint_dataset

    mnms_ckpt = tmp_path / "mnms_best.pt"
    torch.save({"training_dataset": "mnms", "model": {}}, mnms_ckpt)

    with pytest.raises(ValueError, match="cannot evaluate M&Ms-trained model"):
        _inspect_checkpoint_dataset(mnms_ckpt)


def test_external_evaluator_requires_testing_split(tmp_path: Path) -> None:
    from scripts.evaluate_external_mnms import _normalize_external_config

    with pytest.raises(ValueError, match="requires the M&Ms test/testing split"):
        _normalize_external_config(
            {"external_test": {"dataset": "mnms", "split": "train", "data_root": str(tmp_path)}},
            data_root=None,
            split=None,
        )


def test_external_evaluator_subprocess_smoke_with_candidate_c(tmp_path: Path) -> None:
    # 1. Create synthetic M&Ms test dataset (32x32x2) compatible with ConvNeXt downsampling
    _populate_mnms_case(tmp_path / "mnm" / "testing", "A001_t00", shape=(32, 32, 2))

    # 2. Build model and save checkpoint declaring ACDC training dataset
    model_cfg = {
        "model": {
            "encoder_name": "convnext_tiny",
            "pretrained_encoder": False,
            "encoder_allow_fallback": True,
            "shared_channels": 16,
            "num_classes": 4,
            "window_k": 4,
            "max_turns": 2,
            "window_mode": "candidate_c",
            "candidate_c": {
                "rho_feature_pixels": 1.0,
                "lam": 1.0,
                "lr": None,
                "fix_threshold": 0.5,
                "regress_threshold": 0.5,
                "margin_fraction": 0.5,
                "min_regress_mass": 0.001,
                "replay_atol": 0.00001,
                "replay_rtol": 0.00001,
                "max_backtracks": 2,
            },
        },
        "dataset": {"name": "acdc"},
        "training_dataset": "acdc",
    }
    model = build_model_from_config(model_cfg, torch.device("cpu"))
    ckpt_path = tmp_path / "phase_c_best.pt"
    save_checkpoint(ckpt_path, model, config=model_cfg)

    # 3. Create evaluation configuration
    eval_cfg = {
        "protocol": "acdc_to_mnms_domain_shift",
        "external_test": {
            "dataset": "mnms",
            "split": "testing",
            "data_root": str(tmp_path / "mnm"),
            "raw_to_acdc": {0: 0, 1: 3, 2: 2, 3: 1},
        },
        "model": model_cfg["model"],
        "image_size": 32,
        "depth_axis": 2,
        "num_classes": 4,
        "audit": {"tau_accept": 0.0, "t_max": 2},
    }
    cfg_path = tmp_path / "eval_config.yaml"
    cfg_path.write_text(yaml.safe_dump(eval_cfg), encoding="utf-8")
    out_path = tmp_path / "eval_report.json"

    # 4. Run external evaluation entrypoint via subprocess
    cmd = [
        sys.executable,
        "scripts/evaluate_external_mnms.py",
        "--config", str(cfg_path),
        "--checkpoint", str(ckpt_path),
        "--window-mode", "candidate_c",
        "--device", "cpu",
        "--output", str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"External evaluation failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # 5. Verify report schema and invariants
    assert out_path.is_file()
    report = json.loads(out_path.read_text())
    assert report["evidence_class"] == "independent_external_evaluation"
    assert report["scientific_protocol_label"] == "acdc_frozen_to_mnms_external_v1"
    assert report["training_dataset"] == "acdc"
    assert report["dataset"] == "mnms"
    assert report["split"] == "testing"
    assert report["tau_accept"] == 0.0
    assert report["window_mode"] == "candidate_c"
    assert isinstance(report["candidate_c_settings"], dict)
    assert report["candidate_c_settings"]["rho_feature_pixels"] == 1.0
    assert report["candidate_c_settings"]["max_backtracks"] == 2
    assert "metrics" in report
    assert "cohorts" in report["metrics"]
    assert "all" in report["metrics"]["cohorts"]
    assert "self_audit" in report["metrics"]["cohorts"]["all"]["modes"]
    assert "oracle_accept" in report["metrics"]["cohorts"]["all"]["modes"]
    assert report["comparison_modes"] == ["initial_only", "always_accept_refinement", "self_audit", "oracle_accept"]


# =========================================================================
# 8. Strict Frozen Mode & Settings Conflict Rejection
# =========================================================================


def test_external_evaluator_rejects_conflicting_cli_window_mode(tmp_path: Path) -> None:
    from scripts.evaluate_external_mnms import run_external_evaluation

    _populate_mnms_case(tmp_path / "mnm" / "testing", "A001_t00", shape=(32, 32, 2))
    model_cfg = {
        "model": {
            "encoder_name": "convnext_tiny",
            "pretrained_encoder": False,
            "encoder_allow_fallback": True,
            "shared_channels": 16,
            "num_classes": 4,
            "window_k": 4,
            "max_turns": 2,
            "window_mode": "current",
        },
        "dataset": {"name": "acdc"},
        "training_dataset": "acdc",
    }
    model = build_model_from_config(model_cfg, torch.device("cpu"))
    ckpt_path = tmp_path / "ckpt_current.pt"
    save_checkpoint(ckpt_path, model, config=model_cfg)

    eval_cfg = {
        "protocol": "acdc_to_mnms_domain_shift",
        "external_test": {
            "dataset": "mnms",
            "split": "testing",
            "data_root": str(tmp_path / "mnm"),
            "raw_to_acdc": {0: 0, 1: 3, 2: 2, 3: 1},
        },
        "image_size": 32,
        "depth_axis": 2,
        "num_classes": 4,
        "model": model_cfg["model"],
    }

    # Overriding explicitly to candidate_c when checkpoint was trained under current must fail
    with pytest.raises(ValueError, match="conflicts with checkpoint declared window_mode"):
        run_external_evaluation(
            config=eval_cfg,
            checkpoint=ckpt_path,
            window_mode="candidate_c",
            device="cpu",
        )


def test_external_evaluator_rejects_conflicting_candidate_c_settings(tmp_path: Path) -> None:
    from scripts.evaluate_external_mnms import run_external_evaluation

    _populate_mnms_case(tmp_path / "mnm" / "testing", "A001_t00", shape=(32, 32, 2))
    model_cfg = {
        "model": {
            "encoder_name": "convnext_tiny",
            "pretrained_encoder": False,
            "encoder_allow_fallback": True,
            "shared_channels": 16,
            "num_classes": 4,
            "window_k": 4,
            "max_turns": 2,
            "window_mode": "candidate_c",
            "candidate_c": {
                "rho_feature_pixels": 1.0,
                "lam": 1.0,
                "max_backtracks": 2,
            },
        },
        "dataset": {"name": "acdc"},
        "training_dataset": "acdc",
    }
    model = build_model_from_config(model_cfg, torch.device("cpu"))
    ckpt_path = tmp_path / "ckpt_c.pt"
    save_checkpoint(ckpt_path, model, config=model_cfg)

    conflicting_eval_cfg = {
        "protocol": "acdc_to_mnms_domain_shift",
        "external_test": {
            "dataset": "mnms",
            "split": "testing",
            "data_root": str(tmp_path / "mnm"),
            "raw_to_acdc": {0: 0, 1: 3, 2: 2, 3: 1},
        },
        "image_size": 32,
        "depth_axis": 2,
        "num_classes": 4,
        "model": {
            "encoder_name": "convnext_tiny",
            "shared_channels": 16,
            "num_classes": 4,
            "window_k": 4,
            "max_turns": 2,
            "window_mode": "candidate_c",
            "candidate_c": {
                "rho_feature_pixels": 0.5,  # Conflicts with checkpoint 1.0
                "lam": 1.0,
                "max_backtracks": 2,
            },
        },
    }

    with pytest.raises(ValueError, match="conflicts with checkpoint candidate_c"):
        run_external_evaluation(
            config=conflicting_eval_cfg,
            checkpoint=ckpt_path,
            device="cpu",
        )


def test_external_evaluator_rejects_string_candidate_c_settings(tmp_path: Path) -> None:
    from scripts.evaluate_external_mnms import _normalize_external_config

    with pytest.raises(ValueError, match="candidate_c.rho_feature_pixels must be numeric"):
        _normalize_external_config(
            {
                "external_test": {"dataset": "mnms", "split": "testing"},
                "model": {
                    "window_mode": "candidate_c",
                    "candidate_c": {"rho_feature_pixels": "1.0"},
                },
            },
            data_root=str(tmp_path),
            split="testing",
        )


def test_external_evaluator_rejects_conflicting_checkpoint_mode_declarations(tmp_path: Path) -> None:
    from scripts.evaluate_external_mnms import run_external_evaluation
    _populate_mnms_case(tmp_path / "mnm" / "testing", "patient001_t00")
    ckpt_path = tmp_path / "conflicting_ckpt_mode.pt"
    torch.save(
        {
            "training_dataset": "acdc",
            "model": {},
            "config": {"model": {"window_mode": "current"}},
            "provenance": {"model_identity": {"window_mode": "candidate_c"}},
        },
        ckpt_path,
    )
    eval_cfg = {
        "external_test": {"dataset": "mnms", "split": "testing", "data_root": str(tmp_path / "mnm")},
        "model": {"num_classes": 4},
    }
    with pytest.raises(ValueError, match="Conflicting window_mode declarations in checkpoint"):
        run_external_evaluation(config=eval_cfg, checkpoint=ckpt_path, device="cpu")


def test_external_evaluator_rejects_conflicting_config_mode_declarations(tmp_path: Path) -> None:
    from scripts.evaluate_external_mnms import run_external_evaluation
    _populate_mnms_case(tmp_path / "mnm" / "testing", "patient001_t00")
    ckpt_path = tmp_path / "valid_ckpt.pt"
    torch.save({"training_dataset": "acdc", "model": {}, "config": {"model": {"window_mode": "current"}}}, ckpt_path)
    eval_cfg = {
        "external_test": {"dataset": "mnms", "split": "testing", "data_root": str(tmp_path / "mnm")},
        "window_mode": "candidate_c",
        "model": {"window_mode": "current", "num_classes": 4},
    }
    with pytest.raises(ValueError, match="Conflicting window_mode declarations in evaluation config"):
        run_external_evaluation(config=eval_cfg, checkpoint=ckpt_path, device="cpu")


def test_external_evaluator_rejects_conflicting_checkpoint_candidate_c_declarations(tmp_path: Path) -> None:
    from scripts.evaluate_external_mnms import run_external_evaluation
    _populate_mnms_case(tmp_path / "mnm" / "testing", "patient001_t00")
    ckpt_path = tmp_path / "conflicting_c_ckpt.pt"
    torch.save(
        {
            "training_dataset": "acdc",
            "model": {},
            "config": {
                "model": {
                    "window_mode": "candidate_c",
                    "candidate_c": {"rho_feature_pixels": 1.0},
                }
            },
            "provenance": {
                "model_identity": {
                    "window_mode": "candidate_c",
                    "candidate_c_settings": {"rho_feature_pixels": 2.0},
                }
            },
        },
        ckpt_path,
    )
    eval_cfg = {
        "external_test": {"dataset": "mnms", "split": "testing", "data_root": str(tmp_path / "mnm")},
        "model": {"num_classes": 4, "window_mode": "candidate_c"},
    }
    with pytest.raises(ValueError, match="Conflicting candidate_c declarations in checkpoint"):
        run_external_evaluation(config=eval_cfg, checkpoint=ckpt_path, device="cpu")


def test_external_evaluator_rejects_explicit_null_candidate_c_when_checkpoint_is_custom(tmp_path: Path) -> None:
    from scripts.evaluate_external_mnms import run_external_evaluation
    _populate_mnms_case(tmp_path / "mnm" / "testing", "patient001_t00")
    ckpt_path = tmp_path / "custom_c_ckpt.pt"
    torch.save(
        {
            "training_dataset": "acdc",
            "model": {},
            "config": {
                "model": {
                    "window_mode": "candidate_c",
                    "candidate_c": {"rho_feature_pixels": 2.0},
                }
            },
        },
        ckpt_path,
    )
    eval_cfg = {
        "external_test": {"dataset": "mnms", "split": "testing", "data_root": str(tmp_path / "mnm")},
        "model": {
            "num_classes": 4,
            "window_mode": "candidate_c",
            "candidate_c": None,
        },
    }
    with pytest.raises(ValueError, match="conflicts with checkpoint candidate_c"):
        run_external_evaluation(config=eval_cfg, checkpoint=ckpt_path, device="cpu")


def test_external_evaluator_rejects_invalid_null_window_mode(tmp_path: Path) -> None:
    from scripts.evaluate_external_mnms import run_external_evaluation
    _populate_mnms_case(tmp_path / "mnm" / "testing", "patient001_t00")
    ckpt_path = tmp_path / "baseline_ckpt.pt"
    torch.save({"training_dataset": "acdc", "model": {}}, ckpt_path)
    eval_cfg = {
        "external_test": {"dataset": "mnms", "split": "testing", "data_root": str(tmp_path / "mnm")},
        "model": {"num_classes": 4, "window_mode": None},
    }
    with pytest.raises(ValueError, match="must be a string"):
        run_external_evaluation(config=eval_cfg, checkpoint=ckpt_path, device="cpu")


def test_external_evaluator_legacy_checkpoint_requires_baseline(tmp_path: Path) -> None:
    from scripts.evaluate_external_mnms import run_external_evaluation
    _populate_mnms_case(tmp_path / "mnm" / "testing", "patient001_t00")
    ckpt_path = tmp_path / "legacy_ckpt.pt"
    torch.save({"training_dataset": "acdc", "model": {}}, ckpt_path)

    eval_cfg = {
        "external_test": {"dataset": "mnms", "split": "testing", "data_root": str(tmp_path / "mnm")},
        "model": {"num_classes": 4, "window_mode": "candidate_c"},
    }
    with pytest.raises(ValueError, match="Cannot evaluate legacy checkpoint.*source mechanism identity cannot be established"):
        run_external_evaluation(config=eval_cfg, checkpoint=ckpt_path, device="cpu")

