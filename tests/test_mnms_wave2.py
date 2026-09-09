from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch

from self_audit.data.common import VolumeSliceDataset, load_array
from self_audit.data.mnms import (
    DEFAULT_MNMS_TO_ACDC,
    MNMSClassMapping,
    MNMSDataset,
    discover_mnms_records,
)
from self_audit.training._utils import (
    build_model_from_config,
    build_patient_dataset,
    save_checkpoint,
    validate_dataset_splits,
)
from self_audit.training.unified_config import UnifiedConfig, load_unified_config
from self_audit.training.unified_trainer import UnifiedTrainer


def _make_volume_mask(shape: tuple[int, int, int], labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    volume = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    for idx, lbl in enumerate(labels):
        if idx == 0:
            continue
        start = idx * 4
        mask[start:start + 4, start:start + 4, :] = lbl
    return volume, mask


def _populate_split(
    split_dir: Path,
    cases: list[tuple[str, tuple[int, int, int], list[int]]],
) -> None:
    voldir = split_dir / "volumes"
    maskdir = split_dir / "masks"
    voldir.mkdir(parents=True, exist_ok=True)
    maskdir.mkdir(parents=True, exist_ok=True)
    for case_id, shape, labels in cases:
        vol, mask = _make_volume_mask(shape, labels)
        np.save(voldir / f"{case_id}.npy", vol)
        np.save(maskdir / f"{case_id}.npy", mask)


def test_mnms_split_aliases_and_factories(tmp_path: Path) -> None:
    # Setup directories for train/training, val/validation, test/testing
    _populate_split(tmp_path / "train", [("A001_t00", (16, 16, 2), [0, 1, 2, 3])])
    _populate_split(tmp_path / "val", [("A002_t00", (16, 16, 2), [0, 1, 2, 3])])
    _populate_split(tmp_path / "testing", [("A003_t00", (16, 16, 2), [0, 1, 2, 3])])

    # Test alias resolution in discover_mnms_records
    train_recs = discover_mnms_records(tmp_path, split="training")
    assert len(train_recs) == 1 and train_recs[0].patient_id == "A001"

    val_recs = discover_mnms_records(tmp_path, split="validation")
    assert len(val_recs) == 1 and val_recs[0].patient_id == "A002"

    test_recs = discover_mnms_records(tmp_path, split="test")
    assert len(test_recs) == 1 and test_recs[0].patient_id == "A003"

    # Test factory build_patient_dataset with aliases
    ds_train = build_patient_dataset(
        {"dataset": "mnms", "data_root": str(tmp_path), "image_size": 16, "depth_axis": 2},
        split="training",
        train=True,
    )
    assert len(ds_train) == 2  # 2 slices for 1 volume
    assert ds_train.records[0].patient_id == "A001"

    ds_test = build_patient_dataset(
        {"dataset": "mnms", "data_root": str(tmp_path), "image_size": 16, "depth_axis": 2},
        split="test",
        train=False,
    )
    assert len(ds_test) == 2
    assert ds_test.records[0].patient_id == "A003"


def test_mnms_validator_factory_record_parity(tmp_path: Path) -> None:
    _populate_split(tmp_path / "train", [
        ("A001_t00", (16, 16, 2), [0, 1, 2, 3]),
        ("A002_t00", (16, 16, 3), [0, 1, 2, 3]),
    ])
    _populate_split(tmp_path / "val", [
        ("A003_t00", (16, 16, 2), [0, 1, 2, 3]),
    ])
    _populate_split(tmp_path / "test", [
        ("A004_t00", (16, 16, 2), [0, 1, 2, 3]),
    ])

    config = {
        "dataset": "mnms",
        "data_root": str(tmp_path),
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "image_size": 16,
        "depth_axis": 2,
    }

    validation = validate_dataset_splits(config)
    assert validation["validated"] is True
    assert validation["case_counts"] == {"train": 2, "val": 1, "test": 1}
    assert validation["patient_counts"] == {"train": 2, "val": 1, "test": 1}

    # Verify factory record parity for each split
    for split, expected_cases in [
        ("train", ["A001_t00", "A002_t00"]),
        ("val", ["A003_t00"]),
        ("test", ["A004_t00"]),
    ]:
        ds = build_patient_dataset(config, split=split, train=False)
        factory_cases = [r.case_id for r in ds.records]
        validator_cases = validation["effective_identities"][split]
        assert factory_cases == expected_cases
        assert factory_cases == validator_cases


def test_mnms_fractional_and_nonfinite_labels_rejected_before_int_cast(tmp_path: Path) -> None:
    voldir = tmp_path / "testing" / "volumes"
    maskdir = tmp_path / "testing" / "masks"
    voldir.mkdir(parents=True)
    maskdir.mkdir(parents=True)

    vol = np.zeros((16, 16, 2), dtype=np.float32)
    # Fractional mask: 1.5 would silently truncate to 1 if cast to int64 first
    frac_mask = np.full((16, 16, 2), 1.5, dtype=np.float32)
    np.save(voldir / "A001_t00.npy", vol)
    np.save(maskdir / "A001_t00.npy", frac_mask)

    recs = discover_mnms_records(tmp_path, split="testing")

    # 1. VolumeSliceDataset._load directly with float mask containing 1.5
    ds_slice = VolumeSliceDataset(
        records=recs,
        image_size=16,
        depth_axis=2,
    )
    with pytest.raises(ValueError, match="finite integers"):
        _ = ds_slice[0]

    # 2. Non-finite mask (NaN)
    nan_mask = np.zeros((16, 16, 2), dtype=np.float32)
    nan_mask[0, 0, 0] = np.nan
    np.save(maskdir / "A001_t00.npy", nan_mask)
    ds_nan = VolumeSliceDataset(
        records=recs,
        image_size=16,
        depth_axis=2,
    )
    with pytest.raises(ValueError, match="finite integers"):
        _ = ds_nan[0]

    # 3. Direct MNMSDataset rejects fractional labels
    np.save(maskdir / "A001_t00.npy", frac_mask)
    ds_mnms = MNMSDataset(data_root=tmp_path, split="testing", image_size=16, depth_axis=2)
    with pytest.raises(ValueError, match="finite integers"):
        _ = ds_mnms[0]

    # 4. validate_dataset_splits rejects fractional labels
    with pytest.raises(ValueError, match="finite integers"):
        validate_dataset_splits({"dataset": "mnms", "data_root": str(tmp_path), "test_split": "testing", "depth_axis": 2})


def test_mnms_mapping_validation_rejects_booleans_and_fractionals() -> None:
    # Boolean key or value rejected with TypeError
    with pytest.raises(TypeError, match="cannot be booleans"):
        MNMSClassMapping({True: 0, 1: 3, 2: 2, 3: 1})

    with pytest.raises(TypeError, match="cannot be booleans"):
        MNMSClassMapping({0: False, 1: 3, 2: 2, 3: 1})

    # Fractional key or value rejected with ValueError
    with pytest.raises(ValueError, match="cannot be fractional|must be an integer"):
        MNMSClassMapping({0: 0, 1.5: 3, 2: 2, 3: 1})

    with pytest.raises(ValueError, match="cannot be fractional|must be an integer"):
        MNMSClassMapping({0: 0, 1: 2.7, 2: 2, 3: 1})

    # String fractional rejected
    with pytest.raises(ValueError, match="cannot be fractional|must be an integer"):
        MNMSClassMapping({"0": 0, "1.5": 3, "2": 2, "3": 1})

    # Integer strings from JSON accepted and properly cast to int
    cm = MNMSClassMapping({"0": "0", "1": "3", "2": "2", "3": "1"})
    assert cm.raw_to_acdc == {0: 0, 1: 3, 2: 2, 3: 1}

    # Incomplete mapping missing classes
    with pytest.raises(ValueError, match="raw classes 0..3"):
        MNMSClassMapping({0: 0, 1: 1, 2: 2})


def test_mnms_binary_derivative_path_rejected_across_entrypoints(tmp_path: Path) -> None:
    binary_root = tmp_path / "mnm_binary"
    _populate_split(binary_root / "testing", [("A001_t00", (16, 16, 2), [0, 1, 2, 3])])

    # 1. discover_mnms_records
    with pytest.raises(ValueError, match="binary derivative"):
        discover_mnms_records(binary_root, split="testing")

    # 2. MNMSDataset
    with pytest.raises(ValueError, match="binary derivative"):
        MNMSDataset(data_root=binary_root, split="testing", image_size=16)

    # 3. build_patient_dataset
    with pytest.raises(ValueError, match="binary derivative"):
        build_patient_dataset({"dataset": "mnms", "data_root": str(binary_root), "image_size": 16}, split="testing", train=False)

    # 4. validate_dataset_splits
    with pytest.raises(ValueError, match="binary derivative"):
        validate_dataset_splits({"dataset": "mnms", "data_root": str(binary_root), "test_split": "testing"})

    # 5. External evaluator normalization
    from scripts.evaluate_external_mnms import _normalize_external_config
    with pytest.raises(ValueError, match="binary derivative"):
        _normalize_external_config(
            {"external_test": {"dataset": "mnms", "data_root": str(binary_root), "split": "testing"}},
            data_root=None,
            split=None,
        )

    # 6. Benign parent directory containing 'binary' must NOT be rejected
    benign_root = tmp_path / "binary_experiments" / "mnm"
    _populate_split(benign_root / "testing", [("A001_t00", (16, 16, 2), [0, 1, 2, 3])])
    recs = discover_mnms_records(benign_root, split="testing")
    assert len(recs) == 1
    ds = MNMSDataset(data_root=benign_root, split="testing", image_size=16)
    assert len(ds) == 2
    ds_b = build_patient_dataset({"dataset": "mnms", "data_root": str(benign_root), "image_size": 16}, split="testing", train=False)
    assert len(ds_b) == 2
    v_res = validate_dataset_splits({"dataset": "mnms", "data_root": str(benign_root), "test_split": "testing"})
    assert v_res["validated"] is True
    norm_cfg = _normalize_external_config(
        {"external_test": {"dataset": "mnms", "data_root": str(benign_root), "split": "testing"}},
        data_root=None,
        split=None,
    )
    assert norm_cfg["data_root"] == str(benign_root)


def test_mnms_small_case_lacking_foreground_class_succeeds(tmp_path: Path) -> None:
    # Small case with only classes {0, 1, 2} (lacking RV=3) must succeed without error
    _populate_split(tmp_path / "testing", [("A001_t00", (16, 16, 2), [0, 1, 2])])

    validation = validate_dataset_splits({
        "dataset": "mnms",
        "data_root": str(tmp_path),
        "test_split": "testing",
        "depth_axis": 2,
    })
    assert validation["validated"] is True
    assert validation["case_counts"]["test"] == 1

    ds = MNMSDataset(data_root=tmp_path, split="testing", image_size=16, depth_axis=2)
    sample = ds[0]
    # Check that mask was mapped: raw class 1 -> 3, raw class 2 -> 2
    unique_labels = torch.unique(sample["mask"]).tolist()
    for lbl in unique_labels:
        assert lbl in {0, 2, 3}  # mapped from {0, 1, 2}


def test_mnms_native_supervision_train_val_validation(tmp_path: Path) -> None:
    # Distinct patients in train and val
    _populate_split(tmp_path / "train", [("A001_t00", (16, 16, 2), [0, 1, 2, 3])])
    _populate_split(tmp_path / "val", [("A002_t00", (16, 16, 2), [0, 1, 2, 3])])

    cfg = {
        "dataset": "mnms",
        "data_root": str(tmp_path),
        "train_split": "train",
        "val_split": "val",
        "image_size": 16,
        "depth_axis": 2,
    }
    val_res = validate_dataset_splits(cfg)
    assert val_res["validated"] is True
    assert val_res["case_counts"] == {"train": 1, "val": 1}

    # Patient overlap across train and val fails
    _populate_split(tmp_path / "val", [("A001_t01", (16, 16, 2), [0, 1, 2, 3])])
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_dataset_splits(cfg)

    # Patient prefix overlap across train and val fails (A001 vs A001-B)
    import shutil
    shutil.rmtree(tmp_path / "val")
    _populate_split(tmp_path / "val", [("A001B_t00", (16, 16, 2), [0, 1, 2, 3])])
    with pytest.raises(ValueError, match="Patient prefix overlap"):
        validate_dataset_splits(cfg)

    # Shape mismatch (volume vs mask) fails
    shutil.rmtree(tmp_path / "val")
    _populate_split(tmp_path / "val", [("A002_t00", (16, 16, 2), [0, 1, 2, 3])])
    bad_vol = np.zeros((16, 16, 2), dtype=np.float32)
    bad_mask = np.zeros((16, 16, 4), dtype=np.uint8)  # 4 slices instead of 2
    np.save(tmp_path / "val" / "volumes" / "A002_t00.npy", bad_vol)
    np.save(tmp_path / "val" / "masks" / "A002_t00.npy", bad_mask)
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_dataset_splits(cfg)


def test_mnms_checkpoint_training_dataset_binding(tmp_path: Path) -> None:
    from scripts.evaluate_external_mnms import _inspect_checkpoint_dataset, EVIDENCE_CLASS

    # 1. ACDC-trained checkpoint
    acdc_ckpt = tmp_path / "acdc.pt"
    torch.save({"training_dataset": "acdc", "model": {}}, acdc_ckpt)
    ds, ev = _inspect_checkpoint_dataset(acdc_ckpt)
    assert ds == "acdc"
    assert ev == EVIDENCE_CLASS

    # Checkpoint with config.dataset.name = "acdc"
    acdc_cfg_ckpt = tmp_path / "acdc_cfg.pt"
    torch.save({"config": {"dataset": {"name": "acdc"}}, "model": {}}, acdc_cfg_ckpt)
    ds, ev = _inspect_checkpoint_dataset(acdc_cfg_ckpt)
    assert ds == "acdc"

    # 2. M&Ms-trained checkpoint MUST be rejected
    mnms_ckpt = tmp_path / "mnms.pt"
    torch.save({"training_dataset": "mnms", "model": {}}, mnms_ckpt)
    with pytest.raises(ValueError, match="cannot evaluate M&Ms-trained model"):
        _inspect_checkpoint_dataset(mnms_ckpt)

    mnms_cfg_ckpt = tmp_path / "mnms_cfg.pt"
    torch.save({"config": {"dataset": {"name": "mnms"}}, "model": {}}, mnms_cfg_ckpt)
    with pytest.raises(ValueError, match="cannot evaluate M&Ms-trained model"):
        _inspect_checkpoint_dataset(mnms_cfg_ckpt)

    # 3. Substring non-ACDC bypass attempt must be rejected (notacdc contains acdc)
    notacdc_ckpt = tmp_path / "notacdc.pt"
    torch.save({"config": {"dataset": "notacdc"}, "model": {}}, notacdc_ckpt)
    with pytest.raises(ValueError, match="Incompatible checkpoint training dataset"):
        _inspect_checkpoint_dataset(notacdc_ckpt)

    # 4. Contradictory claims across metadata sources must be rejected
    contra_ckpt = tmp_path / "contradictory.pt"
    torch.save(
        {"training_dataset": "acdc", "config": {"dataset": {"name": "mnms"}}, "model": {}},
        contra_ckpt,
    )
    with pytest.raises(ValueError, match="Contradictory training dataset identities"):
        _inspect_checkpoint_dataset(contra_ckpt)

    contra2_ckpt = tmp_path / "contradictory2.pt"
    torch.save(
        {"training_dataset": "acdc", "cohort_descriptor": {"dataset": "mnms"}, "model": {}},
        contra2_ckpt,
    )
    with pytest.raises(ValueError, match="Contradictory training dataset identities"):
        _inspect_checkpoint_dataset(contra2_ckpt)

    # 5. Historical unannotated checkpoint
    hist_ckpt = tmp_path / "hist.pt"
    torch.save({"epoch": 100, "model": {}}, hist_ckpt)
    ds, ev = _inspect_checkpoint_dataset(hist_ckpt)
    assert ds == "unknown"
    assert ev == "uncertified_historical_checkpoint"


def test_mnms_stored_grid_uniformity(tmp_path: Path) -> None:
    from scripts.evaluate_external_mnms import _stored_grid

    # Case 1: Uniform grid (all 32x32)
    _populate_split(tmp_path / "testing", [
        ("A001_t00", (32, 32, 2), [0, 1, 2, 3]),
        ("A002_t00", (32, 32, 4), [0, 1, 2, 3]),
    ])
    ds = MNMSDataset(data_root=tmp_path, split="testing", image_size=32, depth_axis=2)
    grid, is_uniform = _stored_grid(ds, depth_axis=2)
    assert grid == [32, 32]
    assert is_uniform is True

    # Case 2: Heterogeneous grid (32x32 and 48x48)
    _populate_split(tmp_path / "testing", [
        ("A003_t00", (48, 48, 2), [0, 1, 2, 3]),
    ])
    ds_hetero = MNMSDataset(data_root=tmp_path, split="testing", image_size=32, depth_axis=2)
    grid_h, is_uniform_h = _stored_grid(ds_hetero, depth_axis=2)
    assert is_uniform_h is False
    assert grid_h is None


def test_mnms_full_config_contract() -> None:
    cfg = load_unified_config("configs/self_audit_full_mnms.yaml")
    assert cfg.schema_version == 1
    assert cfg.experiment.name == "self_audit_full_mnms"
    assert cfg.dataset.name == "mnms"
    assert cfg.dataset.data_root == "preprocessed_data/mnm"
    assert cfg.dataset.split_manifest is None
    assert cfg.dataset.class_mapping == {0: 0, 1: 3, 2: 2, 3: 1}
    assert cfg.model.pretrained_encoder is True
    assert cfg.checkpoint.output_dir == "weights/self_audit_full_mnms"
    assert cfg.logging.report_dir == "reports/self_audit_full_mnms"

    legacy = cfg.to_legacy_dataset_config()
    assert legacy["dataset"] == "mnms"
    assert legacy["split_manifest"] is None
    assert legacy["class_mapping"] == {0: 0, 1: 3, 2: 2, 3: 1}


def test_unified_trainer_guard_outputs_validates_splits(tmp_path: Path) -> None:
    cfg = load_unified_config("configs/self_audit_full_mnms.yaml")
    cfg_dict = cfg.to_dict()

    # Set data_root to tmp_path
    cfg_dict["dataset"]["data_root"] = str(tmp_path)
    cfg_dict["checkpoint"]["output_dir"] = str(tmp_path / "weights")
    cfg_dict["logging"]["report_dir"] = str(tmp_path / "reports")

    # When data_root exists with invalid splits (e.g. empty directory), guard_training_outputs must fail
    from self_audit.training.unified_config import (
        CalibrationConfig,
        CheckpointConfig,
        DatasetConfig,
        DiagnosticsConfig,
        ExperimentConfig,
        LoggingConfig,
        ModelConfig,
        TrainingConfig,
    )
    modified_cfg = UnifiedConfig(
        schema_version=1,
        experiment=ExperimentConfig(**cfg_dict["experiment"]),
        dataset=DatasetConfig(**cfg_dict["dataset"]),
        model=ModelConfig(**cfg_dict["model"]),
        training=TrainingConfig(
            amp=cfg.training.amp,
            optimizer=cfg.training.optimizer,
            lr_curve=cfg.training.lr_curve,
            annotation_loss=cfg.training.annotation_loss,
            audit_loss=cfg.training.audit_loss,
            counterfactual=cfg.training.counterfactual,
            rollout=cfg.training.rollout,
            schedule=cfg.training.schedule,
        ),
        checkpoint=CheckpointConfig(**cfg_dict["checkpoint"]),
        calibration=CalibrationConfig(**cfg_dict["calibration"]),
        logging=LoggingConfig(
            report_dir=cfg_dict["logging"]["report_dir"],
            wandb=cfg.logging.wandb,
        ),
        diagnostics=DiagnosticsConfig(**cfg_dict["diagnostics"]),
    )

    trainer = UnifiedTrainer(modified_cfg, model=torch.nn.Linear(1, 1), device=torch.device("cpu"), disable_tqdm=True)
    with pytest.raises((ValueError, FileNotFoundError), match="zero records|validation failed|split directory|not found"):
        trainer.guard_training_outputs()


def test_mnms_external_cpu_subprocess_smoke(tmp_path: Path) -> None:
    # 1. Create real M&Ms test fixture (32x32x2)
    voldir = tmp_path / "mnm" / "testing" / "volumes"
    maskdir = tmp_path / "mnm" / "testing" / "masks"
    voldir.mkdir(parents=True)
    maskdir.mkdir(parents=True)
    vol = np.zeros((32, 32, 2), dtype=np.float32)
    mask = np.zeros((32, 32, 2), dtype=np.uint8)
    mask[4:12, 4:12, 0] = 1
    mask[12:20, 12:20, 0] = 2
    np.save(voldir / "A001_t00.npy", vol)
    np.save(maskdir / "A001_t00.npy", mask)

    # 2. Build real model & save checkpoint with ACDC dataset identity
    model_cfg = {
        "model": {
            "encoder_name": "convnext_tiny",
            "pretrained_encoder": False,
            "encoder_allow_fallback": True,
            "shared_channels": 16,
            "num_classes": 4,
            "window_k": 4,
            "max_turns": 2,
        },
        "dataset": {"name": "acdc"}
    }
    model = build_model_from_config(model_cfg, torch.device("cpu"))
    ckpt_path = tmp_path / "phase_c_best.pt"
    save_checkpoint(ckpt_path, model, config=model_cfg)

    # 3. Write protocol config
    eval_cfg = {
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
    cfg_file = tmp_path / "eval_config.json"
    cfg_file.write_text(json.dumps(eval_cfg))
    out_file = tmp_path / "external_report.json"

    # 4. Run real subprocess
    cmd = [
        sys.executable,
        "scripts/evaluate_external_mnms.py",
        "--config", str(cfg_file),
        "--checkpoint", str(ckpt_path),
        "--device", "cpu",
        "--output", str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    assert out_file.is_file()
    report = json.loads(out_file.read_text())
    assert report["training_dataset"] == "acdc"
    assert report["evidence_class"] == "independent_external_evaluation"
    assert report["dataset"] == "mnms"
    assert report["split"] == "testing"
    assert report["stored_grid"] == [32, 32]
    assert report["stored_grid_uniform"] is True
    assert report["tau_accept"] == 0.0
    assert report["comparison_modes"] == ["initial_only", "always_accept_refinement", "self_audit", "oracle_accept"]
    assert "metrics" in report
