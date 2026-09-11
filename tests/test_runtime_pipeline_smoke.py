"""Bounded real subprocess CLI integration smoke tests for runtime hardening.

Verifies end-to-end execution across the complete pipeline:
1. Short 3-interval tiny synthetic ACDC fixture with full validation,
   checkpointing (best.pt/last.pt), post-training calibration, and diagnostics.
2. Standalone scripts/export_transition_bank.py using the frozen selected best checkpoint.
3. Standalone scripts/evaluate_external_mnms.py on held-out synthetic M&Ms using fixed
   calibrated tau and frozen checkpoint, asserting class mapping, shape preservation,
   and that external labels cannot tune model parameters.
4. Native M&Ms separate tiny configuration smoke.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import yaml

import numpy as np
import pytest

from self_audit.artifact_io import read_json_artifact


def _get_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    root_dir = str(Path(__file__).resolve().parents[1])
    src_dir = str(Path(root_dir) / "src")
    env["PYTHONPATH"] = f"{src_dir}:{root_dir}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def _assert_strict_json(path: Path) -> dict[str, Any]:
    """Ensure file exists, parses strictly without nonstandard float constants."""
    assert path.is_file(), f"Expected JSON artifact does not exist: {path}"
    # 1. read_json_artifact raises on raw NaN/Infinity
    payload = read_json_artifact(path)

    # 2. json.loads with parse_constant reject
    def _reject_constant(c: str) -> None:
        raise ValueError(f"Nonstandard JSON constant {c!r} in {path}")

    raw_text = path.read_text(encoding="utf-8")
    verified = json.loads(raw_text, parse_constant=_reject_constant)
    assert isinstance(verified, (dict, list))
    return payload


def _create_synthetic_acdc_dataset(root: Path, seed: int = 42) -> tuple[Path, Path]:
    """Create a minimal 4-class synthetic ACDC dataset with disjoint train/val splits."""
    rng = np.random.default_rng(seed)
    vols = root / "volumes"
    msks = root / "masks"
    vols.mkdir(parents=True, exist_ok=True)
    msks.mkdir(parents=True, exist_ok=True)

    # 2 train cases (patient001), 2 val cases (patient002)
    train_cases = ["patient001_ED", "patient001_ES"]
    val_cases = ["patient002_ED", "patient002_ES"]

    for cid in train_cases + val_cases:
        # [H, W, Z] = (32, 32, 2)
        img = rng.standard_normal((32, 32, 2)).astype(np.float32)
        msk = np.zeros((32, 32, 2), dtype=np.uint8)
        # Populate foreground classes 1, 2, 3
        msk[5:15, 5:15, 0] = 1
        msk[15:25, 5:15, 0] = 2
        msk[10:20, 15:25, 0] = 3
        msk[5:15, 5:15, 1] = 1
        msk[15:25, 5:15, 1] = 2
        msk[10:20, 15:25, 1] = 3
        np.save(vols / f"{cid}.npy", img)
        np.save(msks / f"{cid}.npy", msk)

    manifest_path = root / "splits.json"
    manifest_data = {
        "train": train_cases,
        "val": val_cases,
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    return root, manifest_path


def _create_synthetic_mnms_dataset(root: Path, split: str = "testing", seed: int = 123) -> Path:
    """Create a minimal synthetic M&Ms cohort."""
    rng = np.random.default_rng(seed)
    vols = root / split / "volumes"
    msks = root / split / "masks"
    vols.mkdir(parents=True, exist_ok=True)
    msks.mkdir(parents=True, exist_ok=True)

    cases = ["m001_t00", "m002_t00"]
    for cid in cases:
        img = rng.standard_normal((32, 32, 2)).astype(np.float32)
        msk = np.zeros((32, 32, 2), dtype=np.uint8)
        # Raw M&Ms classes: 0=background, 1=LV, 2=MYO, 3=RV
        msk[4:14, 4:14, 0] = 1
        msk[14:24, 14:24, 0] = 2
        msk[8:18, 14:24, 0] = 3
        msk[4:14, 4:14, 1] = 1
        msk[14:24, 14:24, 1] = 2
        msk[8:18, 14:24, 1] = 3
        np.save(vols / f"{cid}.npy", img)
        np.save(msks / f"{cid}.npy", msk)
    return root


def test_bounded_pipeline_e2e_acdc_to_mnms(tmp_path: Path) -> None:
    """Execute real 3-interval ACDC training, bank export, and external M&Ms evaluation."""
    env = _get_subprocess_env()

    # -------------------------------------------------------------------------
    # 1. Setup synthetic ACDC dataset and tiny unified config
    # -------------------------------------------------------------------------
    acdc_root, manifest_path = _create_synthetic_acdc_dataset(tmp_path / "acdc", seed=42)

    with open("configs/self_audit_full.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"]["data_root"] = str(acdc_root)
    cfg["dataset"]["split_manifest"] = str(manifest_path)
    cfg["dataset"]["image_size"] = 32
    cfg["dataset"]["depth_axis"] = 2
    cfg["dataset"]["dataloader"]["num_workers"] = 0
    cfg["dataset"]["dataloader"]["pin_memory"] = False
    cfg["dataset"]["dataloader"]["persistent_workers"] = False

    cfg["model"]["pretrained_encoder"] = False
    cfg["model"]["fallback"] = True
    cfg["model"]["shared_channels"] = 16
    cfg["model"]["window_k"] = 4
    cfg["model"]["max_turns"] = 2

    weights_dir = tmp_path / "weights"
    reports_dir = tmp_path / "reports"
    cfg["checkpoint"]["output_dir"] = str(weights_dir)
    cfg["checkpoint"]["save_best"] = True
    cfg["checkpoint"]["save_last"] = True
    cfg["checkpoint"]["best_selection_min_epoch"] = 2

    cfg["logging"]["report_dir"] = str(reports_dir)
    cfg["logging"]["wandb"]["enabled"] = False
    cfg["training"]["amp"]["enabled"] = False

    cfg["calibration"]["enabled"] = True
    cfg["calibration"]["threshold_min"] = -0.02
    cfg["calibration"]["threshold_max"] = 0.02
    cfg["calibration"]["threshold_steps"] = 5

    cfg["diagnostics"]["evaluate_headroom"] = True
    cfg["diagnostics"]["evaluate_decomposition"] = True

    cfg["training"]["schedule"]["total_epochs"] = 3
    cfg["training"]["schedule"]["intervals"] = [
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
        },
        {
            "start_epoch": 1,
            "end_epoch": 2,
            "name": "auditor_training",
            "trainable": "auditor",
            "encoder_lr": 0.0,
            "annotation_lr": 0.0,
            "auditor_lr": 0.001,
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
            "encoder_lr": 0.00001,
            "annotation_lr": 0.0001,
            "auditor_lr": 0.0001,
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

    conf_path = tmp_path / "full_pipeline_smoke.yaml"
    conf_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # -------------------------------------------------------------------------
    # 2. Subprocess execution: scripts/train_self_audit.py
    # -------------------------------------------------------------------------
    cmd_train = [
        sys.executable,
        "scripts/train_self_audit.py",
        "--config",
        str(conf_path),
        "--device",
        "cpu",
        "--no_tqdm",
    ]
    res_train = subprocess.run(cmd_train, capture_output=True, text=True, env=env, timeout=180)
    assert res_train.returncode == 0, f"train_self_audit failed:\nSTDERR:\n{res_train.stderr}\nSTDOUT:\n{res_train.stdout}"

    # Verify produced artifacts
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    report_json = reports_dir / "pipeline_report.json"
    cal_json = reports_dir / "calibration.json"
    cache_pt = reports_dir / "validation_transitions.pt"

    assert best_pt.is_file()
    assert last_pt.is_file()
    assert cache_pt.is_file()

    # Strict JSON assertions
    report_data = _assert_strict_json(report_json)
    cal_data = _assert_strict_json(cal_json)

    # Completion and counters assertion
    assert report_data["completed"] is True
    assert len(report_data.get("epochs", [])) == 3
    assert report_data["protocol"]["total_epochs"] == 3
    assert report_data["epochs"][-1]["global_epoch"] == 3

    # Checkpoint hash verification
    best_sha256 = hashlib.sha256(best_pt.read_bytes()).hexdigest()
    binding_dict = report_data.get("checkpoint_binding", {})
    assert binding_dict.get("checkpoint_sha256") == best_sha256

    # Calibration identity verification
    calibrated_tau = float(report_data["calibration"]["calibrated_tau"])
    assert cal_data["tau_accept"] == calibrated_tau
    assert "lineage" in cal_data
    assert "selected_row" in cal_data
    assert cal_data["checkpoint_sha256"] == best_sha256

    # Diagnostics assertion
    diagnostics = report_data.get("final_diagnostics", {})
    assert "headroom" in diagnostics
    assert "decomposition" in diagnostics

    # -------------------------------------------------------------------------
    # 3. Subprocess execution: scripts/export_transition_bank.py
    # -------------------------------------------------------------------------
    bank_json = reports_dir / "transition_bank.json"
    cmd_bank = [
        sys.executable,
        "scripts/export_transition_bank.py",
        "--config",
        str(conf_path),
        "--checkpoint",
        str(best_pt),
        "--output",
        str(bank_json),
        "--data_root",
        str(acdc_root),
        "--split_manifest",
        str(manifest_path),
        "--split",
        "val",
        "--device",
        "cpu",
        "--image_size",
        "32",
        "--tau_accept",
        str(calibrated_tau),
        "--batch_size",
        "2",
    ]
    res_bank = subprocess.run(cmd_bank, capture_output=True, text=True, env=env, timeout=120)
    assert res_bank.returncode == 0, f"export_transition_bank failed:\nSTDERR:\n{res_bank.stderr}\nSTDOUT:\n{res_bank.stdout}"

    bank_data = _assert_strict_json(bank_json)
    assert bank_data["bank_schema_version"] == 1
    rows = bank_data.get("rows", [])
    assert len(rows) > 0, "Transition bank rows must not be empty"

    first_row = rows[0]
    for required_key in ("patient_id", "case_id", "slice_index", "delta_class", "delta_dice", "q_candidate"):
        assert required_key in first_row, f"Row missing required key {required_key}"

    # Provenance matches selected checkpoint
    gen_prov = bank_data.get("generation", {})
    assert gen_prov.get("checkpoint", {}).get("checkpoint_sha256") == best_sha256

    # -------------------------------------------------------------------------
    # 4. Subprocess execution: scripts/evaluate_external_mnms.py
    # -------------------------------------------------------------------------
    mnms_root = _create_synthetic_mnms_dataset(tmp_path / "mnms", split="testing", seed=123)

    mnms_cfg = {
        "protocol": "acdc_to_mnms_domain_shift",
        "external_test": {
            "dataset": "mnms",
            "split": "testing",
            "data_root": str(mnms_root),
            "raw_to_acdc": {0: 0, 1: 3, 2: 2, 3: 1},
        },
        "image_size": 32,
        "num_classes": 4,
        "depth_axis": 2,
        "model": {
            "encoder_name": "convnext_tiny",
            "pretrained_encoder": False,
            "shared_channels": 16,
            "num_classes": 4,
            "window_k": 4,
            "max_turns": 2,
        },
        "audit": {
            "tau_accept": calibrated_tau,
            "t_max": 2,
            "neutral_margin": 0.005,
        },
    }
    mnms_conf_path = tmp_path / "external_mnms.yaml"
    mnms_conf_path.write_text(yaml.safe_dump(mnms_cfg), encoding="utf-8")

    external_json = reports_dir / "external_mnms.json"
    cmd_mnms = [
        sys.executable,
        "scripts/evaluate_external_mnms.py",
        "--config",
        str(mnms_conf_path),
        "--checkpoint",
        str(best_pt),
        "--data_root",
        str(mnms_root),
        "--split",
        "testing",
        "--tau_accept",
        str(calibrated_tau),
        "--device",
        "cpu",
        "--output",
        str(external_json),
    ]
    res_mnms = subprocess.run(cmd_mnms, capture_output=True, text=True, env=env, timeout=120)
    assert res_mnms.returncode == 0, f"evaluate_external_mnms failed:\nSTDERR:\n{res_mnms.stderr}\nSTDOUT:\n{res_mnms.stdout}"

    mnms_data = _assert_strict_json(external_json)
    assert mnms_data["evidence_class"] == "independent_external_evaluation"
    assert mnms_data["training_dataset"] == "acdc"
    assert mnms_data["dataset"] == "mnms"
    assert mnms_data["split"] == "testing"
    assert mnms_data["tau_accept"] == calibrated_tau
    assert mnms_data["tau_accept_source"] == "cli:--tau_accept"

    # Class mapping and shape preservation
    assert mnms_data["network_input_grid"] == [32, 32]
    assert mnms_data["stored_grid"] == [32, 32]
    assert mnms_data["stored_grid_uniform"] is True
    assert mnms_data["label_mapping"] == {0: 0, 1: 3, 2: 2, 3: 1} or mnms_data["label_mapping"] == {"0": 0, "1": 3, "2": 2, "3": 1}

    # Binding reflects selected best checkpoint hash
    assert mnms_data["checkpoint_binding"]["checkpoint_sha256"] == best_sha256


def test_external_labels_cannot_tune_model(tmp_path: Path) -> None:
    """Verify that external evaluation does not mutate the bound model weights or state digest."""
    env = _get_subprocess_env()

    from self_audit.models.self_audit_net import SelfAuditNet
    from self_audit.training._utils import save_checkpoint

    model = SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        num_classes=4,
        window_k=4,
        max_turns=2,
    ).eval()

    ckpt_path = tmp_path / "frozen_best.pt"
    save_checkpoint(
        ckpt_path,
        model,
        epoch=1,
        config={
            "dataset": {"name": "acdc"},
            "model": {
                "shared_channels": 16,
                "window_k": 4,
                "max_turns": 2,
                "num_classes": 4,
            },
        },
        extra={"training_dataset": "acdc"},
    )
    initial_bytes = ckpt_path.read_bytes()
    initial_hash = hashlib.sha256(initial_bytes).hexdigest()

    mnms_root = _create_synthetic_mnms_dataset(tmp_path / "mnms_eval", split="testing", seed=999)
    mnms_cfg = {
        "protocol": "acdc_to_mnms_domain_shift",
        "external_test": {
            "dataset": "mnms",
            "split": "testing",
            "data_root": str(mnms_root),
            "raw_to_acdc": {0: 0, 1: 3, 2: 2, 3: 1},
        },
        "image_size": 32,
        "num_classes": 4,
        "depth_axis": 2,
        "model": {
            "encoder_name": "convnext_tiny",
            "pretrained_encoder": False,
            "shared_channels": 16,
            "num_classes": 4,
            "window_k": 4,
            "max_turns": 2,
        },
        "audit": {"tau_accept": 0.0, "t_max": 2},
    }
    mnms_conf = tmp_path / "mnms_run.yaml"
    mnms_conf.write_text(yaml.safe_dump(mnms_cfg), encoding="utf-8")

    out1 = tmp_path / "out1.json"
    cmd = [
        sys.executable,
        "scripts/evaluate_external_mnms.py",
        "--config",
        str(mnms_conf),
        "--checkpoint",
        str(ckpt_path),
        "--device",
        "cpu",
        "--output",
        str(out1),
    ]
    res1 = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    assert res1.returncode == 0, f"Run 1 failed: {res1.stderr}"

    # Checkpoint bytes must be strictly unchanged
    assert ckpt_path.read_bytes() == initial_bytes
    assert hashlib.sha256(ckpt_path.read_bytes()).hexdigest() == initial_hash

    # Now tamper with masks on disk (e.g. set all mask values to 0)
    for msk_file in (mnms_root / "testing" / "masks").glob("*.npy"):
        np.save(msk_file, np.zeros((32, 32, 2), dtype=np.uint8))

    out2 = tmp_path / "out2.json"
    cmd[cmd.index(str(out1))] = str(out2)
    res2 = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    assert res2.returncode == 0, f"Run 2 failed: {res2.stderr}"

    # Checkpoint bytes and state digest remain invariant
    assert ckpt_path.read_bytes() == initial_bytes
    assert hashlib.sha256(ckpt_path.read_bytes()).hexdigest() == initial_hash

    d1 = _assert_strict_json(out1)
    d2 = _assert_strict_json(out2)
    assert d1["live_state_digest"] == d2["live_state_digest"]


def test_native_mnms_tiny_config_smoke(tmp_path: Path) -> None:
    """Execute bounded smoke training with configs/self_audit_full_mnms.yaml."""
    env = _get_subprocess_env()

    rng = np.random.default_rng(seed=777)
    mnms_root = tmp_path / "mnms_data"
    cases_by_split = {
        "train": ["patient001_t00", "patient002_t00"],
        "val": ["patient003_t00", "patient004_t00"],
    }
    for split, cids in cases_by_split.items():
        s_vols = mnms_root / split / "volumes"
        s_msks = mnms_root / split / "masks"
        s_vols.mkdir(parents=True, exist_ok=True)
        s_msks.mkdir(parents=True, exist_ok=True)

        for cid in cids:
            img = rng.standard_normal((32, 32, 2)).astype(np.float32)
            msk = np.zeros((32, 32, 2), dtype=np.uint8)
            msk[5:15, 5:15, 0] = 1
            msk[15:25, 5:15, 0] = 2
            msk[10:20, 15:25, 0] = 3
            msk[5:15, 5:15, 1] = 1
            msk[15:25, 5:15, 1] = 2
            msk[10:20, 15:25, 1] = 3
            np.save(s_vols / f"{cid}.npy", img)
            np.save(s_msks / f"{cid}.npy", msk)

    with open("configs/self_audit_full_mnms.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"]["data_root"] = str(mnms_root)
    cfg["dataset"]["image_size"] = 32
    cfg["dataset"]["depth_axis"] = 2
    cfg["dataset"]["dataloader"]["num_workers"] = 0
    cfg["dataset"]["dataloader"]["pin_memory"] = False
    cfg["dataset"]["dataloader"]["persistent_workers"] = False

    cfg["model"]["pretrained_encoder"] = False
    cfg["model"]["fallback"] = True
    cfg["model"]["shared_channels"] = 16
    cfg["model"]["window_k"] = 4
    cfg["model"]["max_turns"] = 2

    weights_dir = tmp_path / "weights"
    reports_dir = tmp_path / "reports"
    cfg["checkpoint"]["output_dir"] = str(weights_dir)
    cfg["logging"]["report_dir"] = str(reports_dir)
    cfg["logging"]["wandb"]["enabled"] = False
    cfg["training"]["amp"]["enabled"] = False

    cfg["training"]["schedule"]["total_epochs"] = 1
    cfg["training"]["schedule"]["intervals"] = [
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

    conf_path = tmp_path / "mnms_pipeline_smoke.yaml"
    conf_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cmd_train = [
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
    res = subprocess.run(cmd_train, capture_output=True, text=True, env=env, timeout=60)
    assert res.returncode == 0, f"Native MNMS smoke failed:\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"

    report_path = reports_dir / "pipeline_report.json"
    rep = _assert_strict_json(report_path)
    assert rep["completed"] is False
    assert rep["incomplete_reason"] == "max_steps_reached"
