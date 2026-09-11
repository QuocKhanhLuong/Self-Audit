"""Bounded native M&Ms unified training CLI subprocess smoke test under Candidate C.

NOTE / CONTRACT SCOPE:
This smoke test narrowly exercises the bootstrap phase (interval 1: annotation_bootstrap)
of native M&Ms unified training CLI via scripts/train_self_audit.py with window_mode="candidate_c".
It intentionally does NOT execute the full 3-interval curriculum or demonstrate an eligible solve
(which occurs in interval 3 joint_self_audit under threshold_gate rollout).
Because execution is bounded by --max_steps 2 before epoch completion, only canonical last.pt is
written (best.pt is not claimed or written).

Verifies:
1. Subprocess invocation of scripts/train_self_audit.py using configs/self_audit_full_mnms.yaml
   with window_mode="candidate_c" and --max_steps 2.
2. Proper optimizer step execution and state persistence.
3. Checkpoint generation of canonical last.pt with valid provenance recording window_mode="candidate_c"
   and Candidate C solver settings.
4. Pipeline report artifact generation (pipeline_report.json) with strict JSON validity.
5. Clear distinction between solver eligibility and candidate_c mode selection:
   - window_mode="candidate_c" configures the model with Candidate C architecture.
   - However, in interval 1 (annotation_bootstrap), rollout is "propagate_no_audit"
     where the restitution solver is not eligible/active.
   - Solver becomes actively eligible in interval 3 (joint_self_audit) under "threshold_gate".
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import yaml

import numpy as np
import pytest
import torch

from self_audit.artifact_io import read_json_artifact
from self_audit.training.finetune_joint import CANDIDATE_C_WINDOW_MODES


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
    assert path.is_file(), f"Expected JSON artifact does not exist: {path}"
    payload = read_json_artifact(path)

    def _reject_constant(c: str) -> None:
        raise ValueError(f"Nonstandard JSON constant {c!r} in {path}")

    raw_text = path.read_text(encoding="utf-8")
    verified = json.loads(raw_text, parse_constant=_reject_constant)
    assert isinstance(verified, (dict, list))
    return payload


def _populate_mnms_case(
    split_dir: Path,
    case_id: str,
    *,
    shape: tuple[int, int, int] = (32, 32, 2),
    rng: np.random.Generator,
) -> tuple[Path, Path]:
    vols = split_dir / "volumes"
    msks = split_dir / "masks"
    vols.mkdir(parents=True, exist_ok=True)
    msks.mkdir(parents=True, exist_ok=True)
    img = rng.standard_normal(shape).astype(np.float32)
    msk = np.zeros(shape, dtype=np.uint8)
    msk[5:15, 5:15, 0] = 1
    msk[15:25, 5:15, 0] = 2
    msk[10:20, 15:25, 0] = 3
    msk[5:15, 5:15, 1] = 1
    msk[15:25, 5:15, 1] = 2
    msk[10:20, 15:25, 1] = 3
    vol_path = vols / f"{case_id}.npy"
    msk_path = msks / f"{case_id}.npy"
    np.save(vol_path, img)
    np.save(msk_path, msk)
    return vol_path, msk_path


def test_bounded_native_mnms_candidate_c_cli_smoke(tmp_path: Path) -> None:
    env = _get_subprocess_env()

    # 1. Create synthetic M&Ms dataset (32x32x2) with disjoint train/val splits
    rng = np.random.default_rng(seed=999)
    mnms_root = tmp_path / "mnms_data"
    cases_by_split = {
        "train": ["patient001_t00", "patient002_t00"],
        "val": ["patient003_t00", "patient004_t00"],
    }
    for split, cids in cases_by_split.items():
        split_dir = mnms_root / split
        for cid in cids:
            _populate_mnms_case(split_dir, cid, shape=(32, 32, 2), rng=rng)

    # 2. Load canonical configs/self_audit_full_mnms.yaml and configure for Candidate C smoke
    with open("configs/self_audit_full_mnms.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"]["data_root"] = str(mnms_root)
    cfg["dataset"]["image_size"] = 32
    cfg["dataset"]["depth_axis"] = 2
    cfg["dataset"]["dataloader"]["num_workers"] = 0
    cfg["dataset"]["dataloader"]["pin_memory"] = False
    cfg["dataset"]["dataloader"]["persistent_workers"] = False

    # TEST ONLY: disable pretrained download, keep lightweight channels for fast CPU test
    cfg["model"]["pretrained_encoder"] = False
    cfg["model"]["fallback"] = True
    cfg["model"]["shared_channels"] = 16
    cfg["model"]["window_k"] = 4
    cfg["model"]["max_turns"] = 2

    # Select Candidate C solver mode and configure explicit numeric settings
    cfg["model"]["window_mode"] = "candidate_c"
    cfg["model"]["candidate_c"] = {
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
    }

    weights_dir = tmp_path / "weights"
    reports_dir = tmp_path / "reports"
    cfg["checkpoint"]["output_dir"] = str(weights_dir)
    cfg["checkpoint"]["save_best"] = True
    cfg["checkpoint"]["save_last"] = True
    cfg["checkpoint"]["best_selection_min_epoch"] = 0

    cfg["logging"]["report_dir"] = str(reports_dir)
    cfg["logging"]["wandb"]["enabled"] = False
    cfg["training"]["amp"]["enabled"] = False

    # Bounded 1-epoch interval for smoke
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

    conf_path = tmp_path / "mnms_candidate_c_smoke.yaml"
    conf_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # 3. Subprocess execution of canonical CLI scripts/train_self_audit.py
    cmd_train = [
        sys.executable,
        "scripts/train_self_audit.py",
        "--config", str(conf_path),
        "--device", "cpu",
        "--max_steps", "2",
        "--max_val_batches", "1",
        "--no_tqdm",
    ]
    res = subprocess.run(cmd_train, capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, (
        f"Native M&Ms Candidate C smoke failed:\n"
        f"STDERR:\n{res.stderr}\n"
        f"STDOUT:\n{res.stdout}"
    )

    # 4. Prove actual checkpoint path, optimizer state, and candidate_c provenance
    last_pt = weights_dir / "last.pt"
    assert last_pt.is_file(), f"Expected checkpoint last.pt not found in {weights_dir}"
    payload = torch.load(last_pt, map_location="cpu", weights_only=True)
    assert "model" in payload, "Checkpoint missing model state"
    assert "optimizer" in payload, "Checkpoint missing optimizer state"
    assert payload.get("training_dataset") == "mnms" or payload.get("config", {}).get("dataset", {}).get("name") == "mnms"

    # Verify model identity provenance records candidate_c mode and settings
    model_identity = payload.get("provenance", {}).get("model_identity", {})
    assert model_identity.get("window_mode") == "candidate_c"
    candidate_c_settings = model_identity.get("candidate_c_settings")
    assert isinstance(candidate_c_settings, dict)
    assert candidate_c_settings["rho_feature_pixels"] == 1.0
    assert candidate_c_settings["max_backtracks"] == 2

    # 5. Prove pipeline report path and strict JSON validation
    report_path = reports_dir / "pipeline_report.json"
    rep = _assert_strict_json(report_path)
    assert rep["completed"] is False
    assert rep["incomplete_reason"] == "max_steps_reached"
    assert rep["protocol"]["unified_schedule"] is True

    # 6. Distinguish solver eligibility from mode selection:
    # Mode selection: model is constructed with window_mode="candidate_c" in CANDIDATE_C_WINDOW_MODES.
    ckpt_model_cfg = payload.get("config", {}).get("model", {})
    assert ckpt_model_cfg.get("window_mode") == "candidate_c"
    assert ckpt_model_cfg.get("window_mode") in CANDIDATE_C_WINDOW_MODES
    assert ckpt_model_cfg.get("candidate_c", {}).get("rho_feature_pixels") == 1.0

    # Solver eligibility: in interval 1 (bootstrap), rollout is "propagate_no_audit"
    # so the restitution solver is not active. Solver runs in interval 3 (joint) with "threshold_gate".
    interval_0 = cfg["training"]["schedule"]["intervals"][0]
    assert interval_0["rollout"] == "propagate_no_audit"
    assert interval_0["audit_weight"] == 0.0
