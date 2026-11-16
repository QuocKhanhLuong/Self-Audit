from __future__ import annotations

import json
import pytest

from self_audit.candidate_pg.config import default_config, validate_config
from self_audit.candidate_pg.engine import _lambda_u, run_training


def test_pg_lambda_has_labeled_only_warmup() -> None:
    assert _lambda_u(0, 8000) == 0.0
    assert _lambda_u(799, 8000) == 0.0
    assert _lambda_u(800, 8000) > 0.0


def test_pg_synthetic_full_flow_and_four_exports(tmp_path) -> None:
    config = validate_config(default_config(synthetic=True))
    result = run_training(config, tmp_path, device="cpu", max_updates=1)
    assert result["status"] == "COMPLETED"
    assert tuple(result["arms"]) == ("A1", "A2", "A3", "A4")
    assert all((tmp_path / "exports" / f"{arm}.pt").is_file() for arm in result["arms"])
    assert json.loads((tmp_path / "EXPERIMENT_LOCK.json").read_text())["seed"] == 42


def test_pg_resume_reloads_arm_checkpoint(tmp_path) -> None:
    config = validate_config(default_config(synthetic=True))
    run_training(config, tmp_path, device="cpu", stage="S1", max_updates=1)
    s1 = run_training(config, tmp_path, device="cpu", stage="S1", resume=True, max_updates=1)
    assert s1["stages"]["S1"]["resume_reused"] is True
    for stage in ("S2", "S3"):
        run_training(config, tmp_path, device="cpu", stage=stage, max_updates=1)
    run_training(config, tmp_path, device="cpu", stage="arms", arm="A1", max_updates=1)
    result = run_training(config, tmp_path, device="cpu", stage="arms", arm="A1", resume=True, max_updates=1)
    assert result["status"] == "COMPLETED"


def test_pg_resume_recovers_checkpoint_before_logger_commit(tmp_path, monkeypatch) -> None:
    config = validate_config(default_config(synthetic=True))
    from self_audit.candidate_pg import logging as pg_logging

    for stage in ("S1", "S2", "S3"):
        run_training(config, tmp_path, device="cpu", stage=stage, max_updates=1)

    original_commit = pg_logging.EpochLogger.commit_epoch

    def crash_after_checkpoint(self, *, epoch, update):
        if self.default_stage == "arms":
            raise RuntimeError("injected crash after atomic checkpoint")
        return original_commit(self, epoch=epoch, update=update)

    monkeypatch.setattr(pg_logging.EpochLogger, "commit_epoch", crash_after_checkpoint)
    with pytest.raises(RuntimeError, match="injected crash"):
        run_training(config, tmp_path, device="cpu", stage="arms", arm="A1", max_updates=1)
    assert (tmp_path / "resume_A1.pt").is_file()
    monkeypatch.setattr(pg_logging.EpochLogger, "commit_epoch", original_commit)

    resumed = run_training(config, tmp_path, device="cpu", stage="arms", arm="A1", resume=True, max_updates=1)
    rows = resumed["arms"]["A1"]["metrics"]
    assert resumed["status"] == "COMPLETED"
    assert len(rows) == 6
    assert len({(row["epoch"], row["split"], row["path"]) for row in rows}) == 6
