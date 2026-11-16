from __future__ import annotations

import json
import runpy
import subprocess
import sys
import re
import pytest


def test_smoke_pg_cli_is_local_software_only(tmp_path) -> None:
    output = tmp_path / "smoke"
    completed = subprocess.run(
        [sys.executable, "scripts/smoke_pg.py", "--out", str(output), "--updates", "1"],
        check=True, capture_output=True, text=True,
    )
    lines = completed.stdout.splitlines()
    names = ("S2", "S3", "A1", "A2", "A3", "A4")
    epoch_line = re.compile(r"\[(S2|S3|A[1-4])\] ep \d+/\d+ \| loss [\d.]+ \| train FG [\d.]+ \| val FG [\d.]+ \(best [\d.]+@\d+\) \| test FG [\d.]+ \| \d+s")
    for name in names:
        assert sum(line.startswith(f"[{name}] start") for line in lines) == 1
        assert sum(line.startswith(f"[{name}] end") for line in lines) == 1
    assert all(epoch_line.fullmatch(line) or any(line.startswith(f"[{name}] {event}")
               for name in names for event in ("start", "end")) for line in lines)
    for name in ("S3", "A1", "A2", "A3", "A4"):
        assert sum(line.startswith(f"[{name}] ep ") for line in lines) == 1
    result = json.loads((output / "RESULT.json").read_text())
    assert result["software_only"] is True
    assert result["real_runs"] == "NOT_RUN"
    assert (output / "RESULT.json").is_file()
    override = json.loads((output / "SYNTHETIC_WARMUP_OVERRIDE.json").read_text())
    assert override["status"] == "APPLIED"
    assert override["teacher_lambda_u_warmup_updates_labeled_only"] == 0
    assert override["teacher_threshold_clip"] == [0.0, 0.0]
    s2_diag = json.loads((output / "S2" / "diag.jsonl").read_text().splitlines()[-1])
    assert s2_diag["loss_u"] > 0.0
    # The one-update software fixture must exercise PG and CBKD rather than
    # merely traversing the locked 800-update labelled-only warmup.
    for arm, expected in (("A2", "loss_pg"), ("A3", "loss_cbkd"), ("A4", "loss_pg")):
        row = next(item for item in result["arms"][arm]["metrics"]
                   if item["split"] == "train" and item["path"] == "student_only")
        assert row[expected] > 0.0
    row_a4 = next(item for item in result["arms"]["A4"]["metrics"]
                  if item["split"] == "train" and item["path"] == "student_only")
    assert row_a4["loss_cbkd"] > 0.0
    assert result["finalization"]["status"] == "SOFTWARE_ONLY"


def test_canonical_train_dispatches_all_switch_only_arms(monkeypatch, tmp_path) -> None:
    import self_audit.candidate_pg.engine as pg_engine

    seen = []

    def fake_train(config, out, **kwargs):
        seen.append((kwargs.get("arm"), kwargs["stage"], bool(config["pg"]["enabled"]), bool(config["cbkd"]["enabled"])))
        return {"status": "MOCKED"}

    monkeypatch.setattr(pg_engine, "train", fake_train)
    for arm in ("base", "A1", "A2", "A3", "A4"):
        config_name = "pg_k8_base.json" if arm == "base" else f"pg_k8_{arm}.json"
        monkeypatch.setattr(sys, "argv", [
            "train.py", "--config", f"configs/candidate_pg/{config_name}",
            "--out", str(tmp_path / arm), "--device", "cpu",
        ])
        runpy.run_path("train.py", run_name="__main__")
    assert seen == [(None, "all", True, True), ("A1", "arms", False, False), ("A2", "arms", True, False), ("A3", "arms", False, True), ("A4", "arms", True, True)]


def test_single_arm_cannot_rerun_shared_prerequisites(tmp_path):
    from self_audit.candidate_pg.config import default_config
    from self_audit.candidate_pg.engine import run_training

    with pytest.raises(ValueError, match="shared S1/S2/S3"):
        run_training(default_config(synthetic=True), tmp_path / "run", stage="all", arm="A1")
    assert not (tmp_path / "run").exists()
