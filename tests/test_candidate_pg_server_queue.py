from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pg_server_queue.py"
spec = importlib.util.spec_from_file_location("pg_server_queue", SCRIPT)
queue_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(queue_module)


def _setup(tmp_path: Path):
    manifests = {}
    for role in ("train", "val", "test"):
        path = tmp_path / f"{role}.manifest.json"
        path.write_text(json.dumps({"role": role}))
        manifests[role] = {"path": str(path), "sha256": queue_module.sha256(path)}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema": "self_audit.candidate_pg.v2", "data": {"manifests": manifests}}))
    data = tmp_path / "data.json"
    source = tmp_path / "source.json"
    data.write_text(json.dumps({"status": "READY", "kind": "data"}))
    source.write_text(json.dumps({"status": "READY", "kind": "source"}))
    return config, data, source


def test_tee_keeps_training_visible_and_durable(tmp_path, capsys):
    log_path = tmp_path / 'stage.log'
    with log_path.open('wb') as log:
        result = queue_module.tee_run([sys.executable, '-c', 'print("epoch=1 loss=0.4")'],
            cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0
    assert 'epoch=1 loss=0.4' in capsys.readouterr().out
    assert log_path.read_text().strip() == 'epoch=1 loss=0.4'


def test_failed_boundary_update_cannot_launch_next_stage(tmp_path):
    config, data, source = _setup(tmp_path)
    hook = tmp_path / 'hook.py'
    hook.write_text('import sys\nsys.exit(17 if sys.argv[sys.argv.index("--stage")+1]=="S2" else 0)\n')
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)
    state = queue_module.Queue(config_path=config, out=tmp_path/'out', data_ready=data,
        source_ready=source, device='cpu', poll_seconds=0.001, max_wait_seconds=0,
        runner=runner, before_stage_script=hook).run()
    assert state['status'] == 'FAILED'
    assert state['stages']['S2']['phase'] == 'boundary_hook'
    assert len(calls) == 1 and calls[0][calls[0].index('--pg-stage')+1] == 'S1'


def test_queue_waits_for_ready_receipts_and_runs_canonical_order(tmp_path, monkeypatch):
    config, data, source = _setup(tmp_path)
    data.write_text(json.dumps({"status": "BUILDING", "kind": "data"}))
    calls = []

    def runner(argv, **kwargs):
        state = json.loads((tmp_path / "out" / "QUEUE_STATE.json").read_text())
        stage = state["active_stage"]
        assert state["stages"][stage]["status"] == "RUNNING"
        assert argv == state["stages"][stage]["argv"]
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    q = queue_module.Queue(config_path=config, out=tmp_path / "out", data_ready=data,
                           source_ready=source, device="cuda", poll_seconds=0.001,
                           max_wait_seconds=0, runner=runner)
    state = q.run()
    assert state["status"] == "WAITING"
    assert calls == []
    data.write_text(json.dumps({"status": "READY", "kind": "data"}))
    q = queue_module.Queue(config_path=config, out=tmp_path / "out", data_ready=data,
                           source_ready=source, device="cuda", poll_seconds=0.001,
                           max_wait_seconds=1, runner=runner)
    state = q.run()
    assert state["status"] == "COMPLETED"
    observed = []
    for c in calls:
        stage = c[c.index("--pg-stage") + 1]
        arm = c[c.index("--pg-arm") + 1] if "--pg-arm" in c else None
        observed.append((stage, arm))
    assert observed == [("S1", None), ("S2", None), ("S3", None),
                        ("arms", "A4"), ("arms", "A3"), ("arms", "A2"),
                        ("arms", "A1"), ("arms", None)]
    assert "--pg-resume" not in calls[1]  # S2 has never trained.
    assert all("--pg-resume" in c for i, c in enumerate(calls) if i != 1)
    assert state["arm_order"] == ["A4", "A3", "A2", "A1"]
    assert all(Path(rec["log_path"]).is_file() and rec["returncode"] == 0
               for rec in state["stages"].values())


def test_queue_stops_at_first_failed_stage(tmp_path):
    config, data, source = _setup(tmp_path)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        rc = 17 if argv[argv.index("--pg-stage") + 1] == "S2" else 0
        return subprocess.CompletedProcess(argv, rc)

    state = queue_module.Queue(config_path=config, out=tmp_path / "out", data_ready=data,
                               source_ready=source, device="cpu", poll_seconds=0.001,
                               max_wait_seconds=0, runner=runner).run()
    assert state["status"] == "FAILED"
    assert [c[c.index("--pg-stage") + 1] for c in calls] == ["S1", "S2"]
    assert state["stages"]["S2"]["returncode"] == 17
    assert "S3" not in state["stages"]


def test_completed_stages_are_not_replayed_on_queue_resume(tmp_path):
    config, data, source = _setup(tmp_path)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    kwargs = dict(config_path=config, out=tmp_path / "out", data_ready=data,
                  source_ready=source, device="cpu", poll_seconds=0.001,
                  max_wait_seconds=0, runner=runner)
    assert queue_module.Queue(**kwargs).run()["status"] == "COMPLETED"
    first_count = len(calls)
    assert queue_module.Queue(**kwargs).run()["status"] == "COMPLETED"
    assert len(calls) == first_count


def test_failed_attempt_is_not_retried_by_second_invocation(tmp_path):
    import pytest

    config, data, source = _setup(tmp_path)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        rc = 17 if len(calls) == 2 else 0
        return subprocess.CompletedProcess(argv, rc)

    kwargs = dict(config_path=config, out=tmp_path / "out", data_ready=data,
                  source_ready=source, device="cpu", poll_seconds=0.001,
                  max_wait_seconds=0, runner=runner)
    state = queue_module.Queue(**kwargs).run()
    assert state["status"] == "FAILED"
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="prior FAILED/RUNNING"):
        queue_module.Queue(**kwargs).run()
    assert len(calls) == 2
    recovered = queue_module.Queue(**{**kwargs, "resume_failed_reason": "root reviewed logs and checkpoint identity"}).run()
    assert recovered["status"] == "COMPLETED"
    assert [c[c.index("--pg-stage") + 1] for c in calls[:3]] == ["S1", "S2", "S2"]
    assert recovered["attempt_history"][0]["recovery_reason"] == "root reviewed logs and checkpoint identity"
    assert recovered["attempt_history"][0]["stages"]["S2"]["status"] == "FAILED"
    assert recovered["attempt_history"][0]["stages"]["S2"]["returncode"] == 17
    assert "--pg-resume" not in calls[2]


def test_s2_empty_pretraining_directory_starts_fresh(tmp_path):
    out = tmp_path / "out"
    (out / "S2").mkdir(parents=True)
    commands = queue_module.planned_commands(tmp_path / "config.json", out, "cuda")
    assert "--pg-resume" not in commands[1]


def test_s2_retained_progress_requires_strict_resume_even_without_checkpoint(tmp_path):
    out = tmp_path / "out"
    (out / "S2").mkdir(parents=True)
    (out / "S2" / "diag.jsonl").write_text('{"update": 1}\n')
    commands = queue_module.planned_commands(tmp_path / "config.json", out, "cuda")
    assert "--pg-resume" in commands[1]
    (out / "S2" / "teacher_latest.pt").write_bytes(b"checkpoint validated by S2 trainer")
    commands = queue_module.planned_commands(tmp_path / "config.json", out, "cuda")
    assert "--pg-resume" in commands[1]


@pytest.mark.parametrize("active_stage,checkpoint", [
    ("S2", "S2/teacher_latest.pt"),
    ("S3", "S3/checkpoint.pt"),
    ("arm:A4", "resume_A4.pt"),
])
def test_recovery_resumes_stage_and_preserves_completed_shared_artifacts(tmp_path, active_stage, checkpoint):
    config, data, source = _setup(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    completed = queue_module.STAGES[:queue_module.STAGES.index(active_stage)]
    artifacts = {}
    for stage in completed:
        path = out / f"{stage}_RECEIPT.json"
        path.write_text(json.dumps({"status": "COMPLETED", "stage": stage}))
        artifacts[path] = path.read_bytes()
    cp = out / checkpoint
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_bytes(b"retained stage state; validated by canonical trainer")
    artifacts[cp] = cp.read_bytes()
    stages = {stage: {"status": "COMPLETED"} for stage in completed}
    stages[active_stage] = {"status": "FAILED", "returncode": 17}
    queue_module.atomic_json(out / "QUEUE_STATE.json", {
        "schema": queue_module.SCHEMA, "config_sha256": queue_module.sha256(config),
        "status": "FAILED", "active_stage": active_stage, "stages": stages,
    })
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    state = queue_module.Queue(config_path=config, out=out, data_ready=data,
        source_ready=source, device="cpu", poll_seconds=0.001, max_wait_seconds=0,
        runner=runner, resume_failed_reason="reviewed retained stage checkpoint").run()
    assert state["status"] == "COMPLETED"
    assert len(calls) == len(queue_module.STAGES) - len(completed)
    assert "--pg-resume" in calls[0]
    expected_stage = "arms" if active_stage.startswith("arm:") else active_stage
    assert calls[0][calls[0].index("--pg-stage") + 1] == expected_stage
    if active_stage.startswith("arm:"):
        assert calls[0][calls[0].index("--pg-arm") + 1] == "A4"
    assert all(path.read_bytes() == expected for path, expected in artifacts.items())
    assert state["attempt_history"][0]["stages"][active_stage] == {"status": "FAILED", "returncode": 17}


def test_ready_data_receipt_still_requires_config_manifest_bytes(tmp_path):
    config, data, source = _setup(tmp_path)
    cfg = json.loads(config.read_text())
    cfg["data"]["manifests"] = {"train": {"path": str(tmp_path / "missing.json"), "sha256": "0" * 64},
                                  "val": {"path": str(tmp_path / "val.json"), "sha256": "1" * 64},
                                  "test": {"path": str(tmp_path / "test.json"), "sha256": "2" * 64}}
    config.write_text(json.dumps(cfg))
    qcfg = json.loads(config.read_text())
    with __import__("pytest").raises(queue_module.ReadinessPending, match="manifest artifact missing"):
        queue_module.readiness(data, source, qcfg, queue_module.sha256(config))


def test_failed_or_malformed_readiness_fails_promptly(tmp_path):
    import pytest

    config, data, source = _setup(tmp_path)
    data.write_text(json.dumps({"status": "FAILED", "kind": "data"}))
    with pytest.raises(RuntimeError, match="reports FAILED"):
        queue_module.readiness(data, source, json.loads(config.read_text()), queue_module.sha256(config))
    data.write_text(json.dumps({"status": "not-a-status"}))
    with pytest.raises(ValueError, match="malformed status"):
        queue_module.readiness(data, source, json.loads(config.read_text()), queue_module.sha256(config))


def test_committed_test_and_finalization_are_reused_during_explicit_recovery(tmp_path):
    config, data, source = _setup(tmp_path)
    calls = []
    out = tmp_path / "out"

    def runner(argv, **kwargs):
        calls.append(argv)
        if "--pg-arm" not in argv and argv[argv.index("--pg-stage") + 1] == "arms":
            test_pin = json.loads(config.read_text())["data"]["manifests"]["test"]
            artifacts = {}
            for arm in ("A1", "A2", "A3", "A4"):
                artifact = out / "exports" / f"{arm}.pt"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(arm.encode())
                artifacts[arm] = {"path": str(artifact), "sha256": queue_module.sha256(artifact)}
            (out / "TEST_TRANSACTION.json").write_text(json.dumps({
                "status": "COMMITTED", "manifest": test_pin["path"],
                "manifest_sha256": test_pin["sha256"], "artifacts": artifacts,
            }))
            (out / "FINALIZATION.json").write_text(json.dumps({"test_transaction": {"status": "COMMITTED"}}))
            (out / "FINAL_METRICS.json").write_text(json.dumps({"test_transaction": {"status": "COMMITTED"}}))
            return subprocess.CompletedProcess(argv, 9)
        return subprocess.CompletedProcess(argv, 0)

    kwargs = dict(config_path=config, out=out, data_ready=data, source_ready=source,
                  device="cpu", poll_seconds=0.001, max_wait_seconds=0, runner=runner)
    first = queue_module.Queue(**kwargs).run()
    assert first["status"] == "FAILED"
    count = len(calls)
    recovered = queue_module.Queue(**{**kwargs, "resume_failed_reason": "root reviewed committed transaction"}).run()
    assert recovered["status"] == "COMPLETED"
    assert len(calls) == count
    assert recovered["stages"]["finalize"]["reused"] is True
