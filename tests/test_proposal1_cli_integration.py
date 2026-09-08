"""Public-CLI and subprocess integration tests for Proposal 1.

Guards the complete pipeline:
  scripts/cache_validation_transitions.py
  -> scripts/calibrate_threshold.py
  -> scripts/audit_checkpoint.py

Verifies:
1. Positive matching pipeline works end-to-end via public CLI subprocesses.
2. Replaced weights behind the same checkpoint filename fail evaluation before
   any calibrated report is emitted.
3. CLI tau override (--tau_accept, --allow_tau_override) cannot bypass invalid calibration.
4. Emitted report names actual bound checkpoint digest, file SHA-256, metric contract,
   and cohort identity.
5. Independently permitted disjoint evaluation cohort positive control succeeds
   when authorized and patient-disjoint.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.evaluation.calibration_lineage import runtime_membership_signature
from self_audit.models.self_audit_net import SelfAuditNet
from self_audit.provenance import state_digest
from self_audit.training._utils import (
    build_data_loader,
    build_patient_dataset,
    load_config,
    save_checkpoint,
)

TINY_MODEL_CONFIG: dict[str, Any] = {
    "encoder_name": "convnext_tiny",
    "pretrained_encoder": False,
    "encoder_allow_fallback": True,
    "shared_channels": 16,
    "num_classes": 4,
    "window_k": 4,
    "max_turns": 2,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subproc_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}:{SRC}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _create_synthetic_data(data_dir: Path) -> None:
    vol_dir = data_dir / "volumes"
    mask_dir = data_dir / "masks"
    vol_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "patient001_ED": 10,
        "patient001_ES": 11,
        "patient002_ED": 20,
        "patient002_ES": 21,
        "patient003_ED": 30,
        "patient003_ES": 31,
    }
    for case_id, seed in cases.items():
        np.random.seed(seed)
        volume = np.random.randn(4, 32, 32).astype(np.float32)
        mask = np.random.randint(0, 4, (4, 32, 32)).astype(np.int64)
        np.save(vol_dir / f"{case_id}.npy", volume)
        np.save(mask_dir / f"{case_id}.npy", mask)


def _create_fixture_env(tmp_path: Path) -> dict[str, Any]:
    data_dir = tmp_path / "data"
    _create_synthetic_data(data_dir)

    manifest = {
        "train_cases": ["patient001_ED", "patient001_ES"],
        "val_cases": ["patient002_ED", "patient002_ES"],
        "test_cases": ["patient003_ED", "patient003_ES"],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    config_dict = {
        "dataset": "acdc",
        "data_root": str(data_dir),
        "split_manifest": str(manifest_path),
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "image_size": 32,
        "depth_axis": 0,
        "num_classes": 4,
        "batch_size": 2,
        "num_workers": 0,
        "device": "cpu",
        "seed": 42,
        "deterministic": True,
        "audit": {
            "tau_accept": 0.0,
            "t_max": 2,
            "neutral_margin": 0.005,
        },
        "model": TINY_MODEL_CONFIG,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict), encoding="utf-8")

    torch.manual_seed(42)
    net = SelfAuditNet(**TINY_MODEL_CONFIG).eval()
    ckpt_path = tmp_path / "checkpoint.pt"
    save_checkpoint(ckpt_path, net, epoch=1, config={"model": TINY_MODEL_CONFIG})
    expected_digest = state_digest(net)

    return {
        "data_dir": data_dir,
        "manifest_path": manifest_path,
        "config_path": config_path,
        "ckpt_path": ckpt_path,
        "expected_digest": expected_digest,
        "net": net,
    }


def test_cli_pipeline_positive_matching_end_to_end(tmp_path: Path) -> None:
    """Positive matching pipeline: cache -> calibrate -> audit CLI runs cleanly end-to-end."""
    fix = _create_fixture_env(tmp_path)
    env = _subproc_env()

    # Step 1: cache_validation_transitions.py
    cache_path = tmp_path / "transitions.pt"
    cmd_cache = [
        sys.executable,
        str(ROOT / "scripts" / "cache_validation_transitions.py"),
        "--config", str(fix["config_path"]),
        "--checkpoint", str(fix["ckpt_path"]),
        "--output", str(cache_path),
        "--device", "cpu",
        "--batch_size", "2",
        "--no_tqdm",
    ]
    proc_cache = subprocess.run(cmd_cache, capture_output=True, text=True, env=env)
    assert proc_cache.returncode == 0, f"Cache CLI failed: {proc_cache.stderr}\n{proc_cache.stdout}"
    assert cache_path.is_file(), "Transitions cache file was not created"

    cache_payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert "lineage" in cache_payload
    cache_lineage = cache_payload["lineage"]
    assert cache_lineage["checkpoint"]["state_digest"] == fix["expected_digest"]
    assert cache_lineage["semantics"]["metric_contract"] == "foreground_dice_exclude_v1"
    assert cache_payload["initial_dice"].shape[0] == 8

    # Step 2: calibrate_threshold.py
    calib_path = tmp_path / "calibration.json"
    cmd_calib = [
        sys.executable,
        str(ROOT / "scripts" / "calibrate_threshold.py"),
        "--transitions", str(cache_path),
        "--output", str(calib_path),
        "--checkpoint", str(fix["ckpt_path"]),
        "--min_tau", "-0.1",
        "--max_tau", "0.1",
        "--num_thresholds", "5",
        "--neutral_margin", "0.005",
        "--metric_contract", "foreground_dice_exclude_v1",
    ]
    proc_calib = subprocess.run(cmd_calib, capture_output=True, text=True, env=env)
    assert proc_calib.returncode == 0, f"Calibrate CLI failed: {proc_calib.stderr}\n{proc_calib.stdout}"
    assert calib_path.is_file(), "Calibration artifact was not created"

    with calib_path.open("r", encoding="utf-8") as f:
        calib_data = json.load(f)
    assert calib_data["schema_version"] == 2
    assert calib_data["validity"] == "diagnostic_only"
    assert calib_data["metric_contract"] == "foreground_dice_exclude_v1"
    assert "lineage" in calib_data
    assert calib_data["lineage"]["checkpoint"]["state_digest"] == fix["expected_digest"]

    # Step 3: audit_checkpoint.py
    report_path = tmp_path / "audit_report.json"
    cmd_audit = [
        sys.executable,
        str(ROOT / "scripts" / "audit_checkpoint.py"),
        "--config", str(fix["config_path"]),
        "--checkpoint", str(fix["ckpt_path"]),
        "--calibration", str(calib_path),
        "--output", str(report_path),
        "--device", "cpu",
        "--batch_size", "2",
        "--num_workers", "0",
        "--firewall_batches", "1",
        "--probe_batches", "1",
        "--skip_volume",
        "--no_tqdm",
    ]
    proc_audit = subprocess.run(cmd_audit, capture_output=True, text=True, env=env)
    assert proc_audit.returncode == 0, f"Audit CLI failed: {proc_audit.stderr}\n{proc_audit.stdout}"
    assert report_path.is_file(), "Audit report was not created"

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["diagnostic_schema_version"] == 1
    assert report["evidence_class"] == "diagnostic_only"
    assert report["tau_accept_source"] == "calibration_artifact"
    assert report["threshold_resolution"]["lineage_verified"] is True
    assert report["checkpoint_binding"]["state_digest"] == fix["expected_digest"]
    assert report["checkpoint_binding"]["checkpoint_sha256"] == _file_sha256(fix["ckpt_path"])
    assert report["expected_calibration_lineage"]["semantics"]["metric_contract"] == "foreground_dice_exclude_v1"
    assert report["metric_space"] == "slice_proxy"
    assert report["cohort_policy"]["role"] == "calibration"
    assert report["gt_firewall"]["gt_firewall/passed"] is True
    assert report["required_keys_present"]["ok"] is True


def test_cli_pipeline_replaced_weights_fails_before_calibrated_report(tmp_path: Path) -> None:
    """Same filename replaced weights must fail evaluation before a calibrated report is emitted."""
    fix = _create_fixture_env(tmp_path)
    env = _subproc_env()

    # Build legitimate calibration
    cache_path = tmp_path / "transitions.pt"
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "cache_validation_transitions.py"),
        "--config", str(fix["config_path"]),
        "--checkpoint", str(fix["ckpt_path"]),
        "--output", str(cache_path),
        "--device", "cpu", "--batch_size", "2", "--no_tqdm",
    ], check=True, capture_output=True, text=True, env=env)

    calib_path = tmp_path / "calibration.json"
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "calibrate_threshold.py"),
        "--transitions", str(cache_path),
        "--output", str(calib_path),
        "--checkpoint", str(fix["ckpt_path"]),
        "--num_thresholds", "5", "--neutral_margin", "0.005",
        "--metric_contract", "foreground_dice_exclude_v1",
    ], check=True, capture_output=True, text=True, env=env)

    # Overwrite ckpt_path with DIFFERENT weights under the same filename
    torch.manual_seed(999)
    net_mutated = SelfAuditNet(**TINY_MODEL_CONFIG).eval()
    mutated_ckpt_tmp = tmp_path / "mutated_checkpoint.pt"
    save_checkpoint(mutated_ckpt_tmp, net_mutated, epoch=1, config={"model": TINY_MODEL_CONFIG})
    shutil.copyfile(mutated_ckpt_tmp, fix["ckpt_path"])

    unwanted_report = tmp_path / "should_not_exist_report.json"
    if unwanted_report.exists():
        unwanted_report.unlink()

    cmd_audit = [
        sys.executable,
        str(ROOT / "scripts" / "audit_checkpoint.py"),
        "--config", str(fix["config_path"]),
        "--checkpoint", str(fix["ckpt_path"]),
        "--calibration", str(calib_path),
        "--output", str(unwanted_report),
        "--device", "cpu",
        "--batch_size", "2",
        "--num_workers", "0",
        "--firewall_batches", "1",
        "--probe_batches", "1",
        "--skip_volume",
        "--no_tqdm",
    ]
    proc_audit = subprocess.run(cmd_audit, capture_output=True, text=True, env=env)

    assert proc_audit.returncode != 0, "Audit CLI must fail when weights differ from calibration!"
    assert not unwanted_report.exists(), "Audit report must NOT be emitted when weights mismatch!"
    assert "LineageMismatchError" in proc_audit.stderr or "refusing to apply its threshold" in proc_audit.stderr


def test_cli_tau_override_cannot_bypass_invalid_calibration(tmp_path: Path) -> None:
    """CLI tau override must not bypass lineage verification of an invalid calibration artifact."""
    fix = _create_fixture_env(tmp_path)
    env = _subproc_env()

    cache_path = tmp_path / "transitions.pt"
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "cache_validation_transitions.py"),
        "--config", str(fix["config_path"]),
        "--checkpoint", str(fix["ckpt_path"]),
        "--output", str(cache_path),
        "--device", "cpu", "--batch_size", "2", "--no_tqdm",
    ], check=True, capture_output=True, text=True, env=env)

    calib_path = tmp_path / "calibration.json"
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "calibrate_threshold.py"),
        "--transitions", str(cache_path),
        "--output", str(calib_path),
        "--checkpoint", str(fix["ckpt_path"]),
        "--num_thresholds", "5", "--neutral_margin", "0.005",
        "--metric_contract", "foreground_dice_exclude_v1",
    ], check=True, capture_output=True, text=True, env=env)

    # Overwrite ckpt_path with different weights
    torch.manual_seed(888)
    net_mutated = SelfAuditNet(**TINY_MODEL_CONFIG).eval()
    mutated_ckpt_tmp = tmp_path / "mutated_checkpoint2.pt"
    save_checkpoint(mutated_ckpt_tmp, net_mutated, epoch=1, config={"model": TINY_MODEL_CONFIG})
    shutil.copyfile(mutated_ckpt_tmp, fix["ckpt_path"])

    unwanted_report = tmp_path / "unwanted_override_report.json"
    if unwanted_report.exists():
        unwanted_report.unlink()

    cmd_audit_override = [
        sys.executable,
        str(ROOT / "scripts" / "audit_checkpoint.py"),
        "--config", str(fix["config_path"]),
        "--checkpoint", str(fix["ckpt_path"]),
        "--calibration", str(calib_path),
        "--tau_accept", "0.15",
        "--allow_tau_override",
        "--output", str(unwanted_report),
        "--device", "cpu",
        "--batch_size", "2",
        "--num_workers", "0",
        "--firewall_batches", "1",
        "--probe_batches", "1",
        "--skip_volume",
        "--no_tqdm",
    ]
    proc_audit_override = subprocess.run(cmd_audit_override, capture_output=True, text=True, env=env)

    assert proc_audit_override.returncode != 0, "CLI tau override must not bypass invalid calibration!"
    assert not unwanted_report.exists(), "Audit report must NOT be emitted when calibration is invalid!"
    assert "LineageMismatchError" in proc_audit_override.stderr or "refusing to apply its threshold" in proc_audit_override.stderr


def test_cli_disjoint_independent_evaluation_cohort_positive(tmp_path: Path) -> None:
    """Tiny independently permitted disjoint evaluation positive control succeeds."""
    fix = _create_fixture_env(tmp_path)
    env = _subproc_env()

    # Step 1 & 2: calibrate on original val split (patient002)
    cache_path = tmp_path / "transitions.pt"
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "cache_validation_transitions.py"),
        "--config", str(fix["config_path"]),
        "--checkpoint", str(fix["ckpt_path"]),
        "--output", str(cache_path),
        "--device", "cpu", "--batch_size", "2", "--no_tqdm",
    ], check=True, capture_output=True, text=True, env=env)

    calib_path = tmp_path / "calibration.json"
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "calibrate_threshold.py"),
        "--transitions", str(cache_path),
        "--output", str(calib_path),
        "--checkpoint", str(fix["ckpt_path"]),
        "--num_thresholds", "5", "--neutral_margin", "0.005",
        "--metric_contract", "foreground_dice_exclude_v1",
    ], check=True, capture_output=True, text=True, env=env)

    # Step 3: create disjoint manifest where val contains patient003 (disjoint from patient002)
    disjoint_manifest = {
        "train_cases": ["patient001_ED", "patient001_ES"],
        "val_cases": ["patient003_ED", "patient003_ES"],
        "test_cases": ["patient002_ED", "patient002_ES"],
    }
    disjoint_manifest_path = tmp_path / "disjoint_manifest.json"
    disjoint_manifest_path.write_text(json.dumps(disjoint_manifest), encoding="utf-8")

    disjoint_config = load_config(fix["config_path"])
    disjoint_config["split_manifest"] = str(disjoint_manifest_path)
    disjoint_config_path = tmp_path / "disjoint_config.yaml"
    disjoint_config_path.write_text(yaml.dump(disjoint_config), encoding="utf-8")

    # Compute protocol's authorized membership signature for disjoint val split
    disjoint_dataset = build_patient_dataset(disjoint_config, split="val", train=False)
    disjoint_loader = build_data_loader(
        disjoint_dataset, disjoint_config, device=torch.device("cpu"), train=False, batch_size=2
    )
    auth_sig = runtime_membership_signature(disjoint_loader, split_name="val")

    disjoint_report_path = tmp_path / "disjoint_report.json"
    cmd_audit_disjoint = [
        sys.executable,
        str(ROOT / "scripts" / "audit_checkpoint.py"),
        "--config", str(disjoint_config_path),
        "--checkpoint", str(fix["ckpt_path"]),
        "--calibration", str(calib_path),
        "--cohort_role", "independent_evaluation",
        "--authorized_cohort_signature", auth_sig,
        "--authorized_cohort_split", "val",
        "--output", str(disjoint_report_path),
        "--device", "cpu",
        "--batch_size", "2",
        "--num_workers", "0",
        "--firewall_batches", "1",
        "--probe_batches", "1",
        "--skip_volume",
        "--no_tqdm",
    ]
    proc_disjoint = subprocess.run(cmd_audit_disjoint, capture_output=True, text=True, env=env)
    assert proc_disjoint.returncode == 0, f"Disjoint evaluation failed: {proc_disjoint.stderr}\n{proc_disjoint.stdout}"
    assert disjoint_report_path.is_file()

    with disjoint_report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["cohort_policy"]["role"] == "independent_evaluation"
    verification = report["calibration_lineage_verification"]
    assert verification["verified"] is True
    assert verification["cohort"]["patients_disjoint"] is True
    assert verification["cohort"]["role"] == "independent_evaluation"
    assert verification["cohort"]["calibration_patient_count"] == 1
    assert verification["cohort"]["evaluation_patient_count"] == 1
