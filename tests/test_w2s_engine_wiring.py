"""Engine wiring for W2S arms (synthetic CPU toys; software evidence only)."""
from __future__ import annotations

import copy

import pytest

from self_audit.candidate_pg.config import default_config, validate_config
from self_audit.candidate_pg.engine import run_training


def _config(mode: str):
    cfg = default_config(synthetic=True)
    cfg["pg"]["target_mode"] = mode
    cfg["pg"]["w2s"] = {"tau": 0.95, "tau_v": 0.99, "alpha_max": 0.5, "alpha_start_fraction": 0.2, "alpha_end_fraction": 0.5, "min_visible_views": 2}
    return validate_config(cfg)


def test_self_training_arm_runs_without_teacher_or_codebook_receipts(tmp_path) -> None:
    cfg = _config("self")
    run_training(cfg, tmp_path, device="cpu", stage="S1", max_updates=1)
    result = run_training(cfg, tmp_path, device="cpu", stage="arms", arm="A2", max_updates=1)
    assert result["status"] == "COMPLETED"
    assert not (tmp_path / "S2_RECEIPT.json").exists() and not (tmp_path / "S3_RECEIPT.json").exists()


def test_arbiter_arm_without_codebook_needs_teacher_but_not_s3(tmp_path) -> None:
    cfg = _config("w2s_arbiter")
    for stage in ("S1", "S2"):
        run_training(cfg, tmp_path, device="cpu", stage=stage, max_updates=1)
    result = run_training(cfg, tmp_path, device="cpu", stage="arms", arm="A2", max_updates=1)
    assert result["status"] == "COMPLETED"
    assert not (tmp_path / "S3_RECEIPT.json").exists()


def test_val20_config_key_is_validated() -> None:
    cfg = copy.deepcopy(default_config(synthetic=True))
    cfg["data"]["w2s_val20"] = {"path": "x.json", "sha256": "0" * 64}
    validate_config(cfg)
    cfg["data"]["w2s_val20"] = {"path": "x.json"}
    with pytest.raises(ValueError, match="w2s_val20"):
        validate_config(cfg)
