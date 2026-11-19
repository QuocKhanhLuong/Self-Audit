"""Pipeline orchestration: DONE markers, resume/skip, budget-guard abort (toy stages, no GPU)."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cbkd_sa_pipeline", ROOT / "scripts/cbkd_sa_pipeline.py")
pipe = importlib.util.module_from_spec(spec); sys.modules["cbkd_sa_pipeline"] = pipe; spec.loader.exec_module(pipe)


def _patch(monkeypatch, tmp_path, script: str):
    toy = tmp_path / "toy.py"
    toy.write_text(script)
    monkeypatch.setattr(pipe, "stage_command", lambda stage, config, out, device: [sys.executable, str(toy), stage, str(out)])
    monkeypatch.setattr(pipe, "artifacts", lambda stage, out: [])


def _cfg(tmp_path):
    cfg = tmp_path / "cfg.json"; cfg.write_text(json.dumps({"a": 1})); return cfg


def test_done_markers_skip_finished_stages_and_resume_failed_one(monkeypatch, tmp_path) -> None:
    out = tmp_path / "run"
    _patch(monkeypatch, tmp_path, """
import sys, pathlib
stage, out = sys.argv[1], pathlib.Path(sys.argv[2]); out.mkdir(exist_ok=True, parents=True)
(out / f"{stage}.ran").write_text(str(len(list(out.glob(stage + '.ran*')))))
with open(out / "calls.txt", "a") as fh: fh.write(stage + "\\n")
if stage == "S2" and not (out / "S2.ok").exists():
    (out / "S2.ok").write_text("x"); sys.exit(7)   # first S2 attempt crashes
""")
    cfg = _cfg(tmp_path)
    assert pipe.main(["--config", str(cfg), "--out", str(out), "--stages", "S1,S2,S3"]) == 1
    assert (out / "markers/S1.DONE.json").is_file() and (out / "markers/S2.FAILED.json").is_file() and not (out / "markers/S3.DONE.json").exists()
    assert pipe.main(["--config", str(cfg), "--out", str(out), "--stages", "S1,S2,S3"]) == 0
    calls = (out / "calls.txt").read_text().split()
    assert calls == ["S1", "S2", "S2", "S3"]  # S1 not re-run; S2 retried once; S3 ran once
    # a changed config invalidates the marker
    cfg.write_text(json.dumps({"a": 2}))
    assert pipe.main(["--config", str(cfg), "--out", str(out), "--stages", "S1"]) == 0
    assert (out / "calls.txt").read_text().split()[-1] == "S1" and (out / "calls.txt").read_text().split().count("S1") == 2


def test_budget_guard_aborts_stage_and_total(monkeypatch, tmp_path) -> None:
    out = tmp_path / "run"
    _patch(monkeypatch, tmp_path, "import time; time.sleep(60)")
    monkeypatch.setenv("CAP_HOURS_S1", str(0.5 / 3600 / pipe.OVERRUN))  # 0.5 s cap
    t0 = time.time()
    code = pipe.main(["--config", str(_cfg(tmp_path)), "--out", str(out), "--stages", "S1", "--poll-seconds", "0.2"])
    assert code == 3 and time.time() - t0 < 20
    marker = json.loads((out / "markers/S1.ABORTED_BUDGET.json").read_text())
    assert marker["reason"] == "stage_cap" and not (out / "markers/S1.DONE.json").exists()
    monkeypatch.delenv("CAP_HOURS_S1")
    monkeypatch.setenv("MAX_HOURS", str(0.5 / 3600))
    code = pipe.main(["--config", str(_cfg(tmp_path)), "--out", str(out), "--stages", "S2", "--poll-seconds", "0.2"])
    assert code == 3 and json.loads((out / "markers/S2.ABORTED_BUDGET.json").read_text())["reason"] == "max_hours"


def test_oracle_patch_roundtrip_and_pooled_oracle() -> None:
    import torch

    spec2 = importlib.util.spec_from_file_location("cbkd_sa_oracles", ROOT / "scripts/cbkd_sa_oracles.py")
    o = importlib.util.module_from_spec(spec2); sys.modules["cbkd_sa_oracles"] = o; spec2.loader.exec_module(o)
    t = torch.zeros(5, 37, 41, dtype=torch.long); t[:, 8:30, 6:33] = 1; t[:, 14:22, 14:24] = 3  # odd sizes -> padded cells excluded
    from self_audit.candidate_pg import shape_atoms as sa

    for patch in ((2, 8, 8), (1, 8, 8)):
        assert torch.equal(sa.assemble_full_res(sa.patch_split(t, patch), tuple(t.shape), patch), t)  # exact round trip, pad cells dropped
    pred, stats = o.pooled_oracle(t)
    assert pred.shape == t.shape and stats["mixed_cells"] > 0 and (pred == t).float().mean() > 0.8


def test_s2_resume_flag_only_when_checkpoint_exists(tmp_path) -> None:
    cfg = tmp_path / "c.json"
    assert "--resume" not in pipe.stage_command("S2", cfg, tmp_path, "cuda")
    (tmp_path / "S2").mkdir(); (tmp_path / "S2" / "teacher_latest.pt").write_bytes(b"x")
    assert "--resume" in pipe.stage_command("S2", cfg, tmp_path, "cuda")
    assert "--resume" in pipe.stage_command("S3", cfg, tmp_path, "cuda") and "--resume" in pipe.stage_command("A4", cfg, tmp_path, "cuda")


def test_pipeline_triggers_upload_after_done_and_on_budget_abort_and_survives_failure(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(pipe, "trigger_upload", lambda stage, out, include_resume=False: calls.append((stage, include_resume)))
    _patch(monkeypatch, tmp_path, "pass")
    assert pipe.main(["--config", str(_cfg(tmp_path)), "--out", str(tmp_path / "r1"), "--stages", "S1,S2"]) == 0
    assert calls == [("S1", False), ("S2", False)]
    calls.clear()
    _patch(monkeypatch, tmp_path, "import time; time.sleep(60)")
    monkeypatch.setenv("MAX_HOURS", str(0.5 / 3600))
    assert pipe.main(["--config", str(_cfg(tmp_path)), "--out", str(tmp_path / "r2"), "--stages", "S3", "--poll-seconds", "0.2"]) == 3
    assert calls == [("S3", True)]  # aborted stage uploads its last resume checkpoint
    monkeypatch.undo()


def test_trigger_upload_is_bounded_and_never_raises(monkeypatch, tmp_path) -> None:
    import subprocess

    seen = {}

    def slow(cmd, timeout=None, **kw):
        seen["cmd"], seen["timeout"] = cmd, timeout
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(pipe.subprocess, "run", slow)
    monkeypatch.setenv("CBKD_SA_UPLOAD_TIMEOUT", "7")
    pipe.trigger_upload("S2", tmp_path, include_resume=True)  # timeout -> warning only
    assert seen["timeout"] == 7 and "--include-resume" in seen["cmd"] and "S2" in seen["cmd"]

    def broken(cmd, timeout=None, **kw):
        raise OSError("no rclone")

    monkeypatch.setattr(pipe.subprocess, "run", broken)
    pipe.trigger_upload("S1", tmp_path)  # any error -> warning only
    monkeypatch.setenv("CBKD_SA_UPLOAD", "0")
    monkeypatch.setattr(pipe.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    pipe.trigger_upload("S1", tmp_path)
