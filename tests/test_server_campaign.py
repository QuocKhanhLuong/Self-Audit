from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from self_audit.server_campaign import (
    CampaignError,
    process_identity,
    request_stop,
    run_foreground,
    sha256_file,
    start_campaign,
    status_state,
    validate_campaign,
)


def _script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _pin(path: Path, pin_id: str, kind: str, payload: bytes) -> dict:
    path.write_bytes(payload)
    return {"id": pin_id, "kind": kind, "path": str(path), "sha256": sha256_file(path)}


def _base_plan(tmp_path: Path, stages: list[dict], pins: list[dict], *, continue_independent: bool = True) -> dict:
    state_dir = tmp_path / "campaign_state"
    return {
        "schema": "self_audit.server_campaign.v1",
        "campaign_id": f"fixture_{tmp_path.name.replace('-', '_')[:40]}",
        "campaign_mode": "training",
        "state_dir": str(state_dir),
        "progress_receipts_jsonl": str(state_dir / "progress_receipts.jsonl"),
        "max_wall_clock_seconds": 30,
        "poll_seconds": 0.1,
        "continue_independent_on_failure": continue_independent,
        "gpu": {"visible_device": "0"},
        "pins": pins,
        "stages": stages,
    }


def _training_stage(
    tmp_path: Path,
    *,
    stage_id: str,
    role: str,
    script: Path,
    argv_tail: list[str],
    receipt: Path,
    progress: Path,
    pin_ids: list[str],
    source_pin_ids: list[str] | None = None,
    requires: list[str] | None = None,
    resume: dict | None = None,
) -> dict:
    stage = {
        "id": stage_id,
        "kind": "command",
        "role": role,
        "requires": requires or [],
        "training": {
            "steps": 10000,
            "recipe_pin": "recipe",
            "source_pin_ids": source_pin_ids or ["source"],
            "asset_pin_ids": ["train_asset"],
            "progress_source": {
                "path": str(progress),
                "step_field": "step",
                "eta_seconds_field": "eta_seconds",
            },
        },
        "pin_ids": pin_ids,
        "argv": [sys.executable, str(script), *argv_tail],
        "cwd": str(tmp_path),
        "completion": {"path": str(receipt), "field": "status", "equals": "COMPLETED"},
    }
    if resume is not None:
        stage["resume"] = resume
    return stage


def _write_plan(tmp_path: Path, plan: dict) -> tuple[Path, str]:
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, sha256_file(path)


def _wait_for(predicate, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError("timed out waiting for campaign fixture")


def _mk_training_pins(tmp_path: Path, *extra_sources: tuple[str, Path]) -> list[dict]:
    pins = [
        _pin(tmp_path / "recipe.json", "recipe", "recipe", b'{"steps":10000}\n'),
        _pin(tmp_path / "runner.py", "source", "source", b"pinned runner source\n"),
        _pin(tmp_path / "train_asset.json", "train_asset", "asset", b"manufactured train asset\n"),
    ]
    for pin_id, source_path in extra_sources:
        pins.append(_pin(source_path, pin_id, "source", source_path.read_bytes()))
    return pins


def _wait_campaign_terminal(state_dir: Path, timeout: float = 10.0) -> dict:
    return _wait_for(
        lambda: (state if (state := status_state(state_dir)).get("status") in {
            "completed", "completed_with_blocks", "blocked_hash_drift", "blocked_safety", "stopped", "deadline_exceeded"
        } else None),
        timeout=timeout,
    )


def test_fast_owned_child_and_zero_based_cumulative_progress(tmp_path, monkeypatch):
    from self_audit import server_campaign as campaign
    receipt, progress = tmp_path / "done.json", tmp_path / "progress.jsonl"
    script = _script(tmp_path / "fast.py",
                     "import json,pathlib,sys\n"
                     "pathlib.Path(sys.argv[1]).write_text(json.dumps({'status':'COMPLETED'}))\n"
                     "pathlib.Path(sys.argv[2]).write_text(json.dumps({'step':0,'elapsed':2})+'\\n')\n")
    pins = _mk_training_pins(tmp_path)
    stage = _training_stage(tmp_path, stage_id="fast", role="supervised", script=script,
                            argv_tail=[str(receipt), str(progress)], receipt=receipt, progress=progress,
                            pin_ids=[p["id"] for p in pins])
    stage["training"]["progress_source"] = {"path": str(progress), "step_field": "step",
                                              "step_offset": 1, "elapsed_seconds_field": "elapsed"}
    plan = _base_plan(tmp_path, [stage], pins)
    path, sha = _write_plan(tmp_path, plan)
    # Deliberately miss the tiny owned child's process identity observation.
    monkeypatch.setattr(campaign.CampaignController, "_wait_identity",
                        staticmethod(lambda pid: (time.sleep(.3), None)[1]))
    log, execution, result = (tmp_path / name for name in ("stdout.log", "execution.json", "result.json"))
    assert campaign._stage_child_main(str(path), sha, "fast", 1, str(log), str(execution), str(result)) == 0
    assert json.loads(result.read_text())["exit_code"] == 0
    assert json.loads(execution.read_text())["command_finished_before_identity_observation"] is True
    controller = campaign.CampaignController(path, sha)
    controller._poll_progress(stage)
    rows = [json.loads(line) for line in Path(plan["progress_receipts_jsonl"]).read_text().splitlines()]
    assert rows[-1]["step"] == 1 and rows[-1]["eta_seconds"] == 19998
    progress.write_text("")
    controller._poll_progress(stage)
    assert controller._state["safety_stop_reason"] == "invalid_training_progress_receipt"


def test_detached_adopts_running_s0_once_and_duplicate_start_is_refused(tmp_path: Path):
    s0 = _script(
        tmp_path / "existing_s0.py",
        "import json, pathlib, sys, time\n"
        "started, release, receipt = map(pathlib.Path, sys.argv[1:])\n"
        "started.write_text('started\\n', encoding='utf-8')\n"
        "while not release.exists(): time.sleep(0.03)\n"
        "receipt.write_text(json.dumps({'status':'COMPLETED','training_complete':True,'steps':10000})+'\\n', encoding='utf-8')\n",
    )
    started = tmp_path / "s0.started"
    release = tmp_path / "s0.release"
    receipt = tmp_path / "s0.receipt.json"
    log_path = tmp_path / "existing_s0.stdout.log"
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            [sys.executable, str(s0), str(started), str(release), str(receipt)],
            cwd=tmp_path,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        _wait_for(lambda: started.exists())
        identity = process_identity(proc.pid)
        assert identity is not None
        pins = _mk_training_pins(tmp_path)
        progress = tmp_path / "s0.TRAIN_LOG.jsonl"
        progress.write_text("", encoding="utf-8")
        plan = _base_plan(tmp_path, [{
            "id": "s0_existing",
            "kind": "adopt",
            "role": "supervised",
            "requires": [],
            "training": {
                "steps": 10000,
                "recipe_pin": "recipe",
                "source_pin_ids": ["source"],
                "asset_pin_ids": ["train_asset"],
                "progress_source": {"path": str(progress), "step_field": "step", "eta_seconds_field": "eta_seconds"},
            },
            "process": {
                "pid": proc.pid,
                "start_ticks": identity["start_ticks"],
                "cwd": identity["cwd"],
                "argv_sha256": identity["argv_sha256"],
                "log_path": str(log_path),
                "receipt_sha256_before": None,
            },
            "completion": {"path": str(receipt), "field": "training_complete", "equals": True},
        }], pins)
        plan_path, plan_sha = _write_plan(tmp_path, plan)

        started_result = start_campaign(str(plan_path), plan_sha)
        assert started_result["status"] == "started"
        duplicate = start_campaign(str(plan_path), plan_sha)
        assert duplicate["status"] == "already_running"
        running = _wait_for(lambda: (state if (state := status_state(plan["state_dir"]))["active_stage"] == "s0_existing" else None))
        assert running["stages"]["s0_existing"]["pid"] == proc.pid

        release.write_text("continue\n", encoding="utf-8")
        proc.wait(timeout=5)
        final = _wait_campaign_terminal(Path(plan["state_dir"]))
        assert final["status"] == "completed"
        assert final["stages"]["s0_existing"]["status"] == "succeeded"
        assert started.read_text(encoding="utf-8") == "started\n"
    finally:
        release.write_text("cleanup\n", encoding="utf-8")
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_detached_controller_restarts_with_only_explicit_hash_verified_resume(tmp_path: Path):
    initial = _script(
        tmp_path / "initial.py",
        "import json, pathlib, sys, time\n"
        "marker, checkpoint, progress, receipt = map(pathlib.Path, sys.argv[1:])\n"
        "marker.open('a', encoding='utf-8').write('initial\\n')\n"
        "checkpoint.write_bytes(b'fixed manufactured checkpoint\\n')\n"
        "progress.write_text(json.dumps({'step':10,'eta_seconds':2.5})+'\\n', encoding='utf-8')\n"
        "while True: time.sleep(0.1)\n",
    )
    resume_script = _script(
        tmp_path / "resume.py",
        "import hashlib, json, pathlib, sys\n"
        "marker, checkpoint, expected_sha, progress, receipt = sys.argv[1:]\n"
        "checkpoint = pathlib.Path(checkpoint)\n"
        "assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected_sha\n"
        "pathlib.Path(marker).open('a', encoding='utf-8').write('resume\\n')\n"
        "with pathlib.Path(progress).open('a', encoding='utf-8') as out: out.write(json.dumps({'step':10000,'eta_seconds':0})+'\\n')\n"
        "pathlib.Path(receipt).write_text(json.dumps({'status':'COMPLETED','steps':10000})+'\\n', encoding='utf-8')\n",
    )
    marker = tmp_path / "run-count.txt"
    checkpoint = tmp_path / "step_001000.MANIFEST.json"
    progress = tmp_path / "train.jsonl"
    receipt = tmp_path / "run.receipt.json"
    checkpoint_sha = hashlib.sha256(b"fixed manufactured checkpoint\n").hexdigest()
    pins = _mk_training_pins(tmp_path, ("source_resume", resume_script))
    progress.write_text("", encoding="utf-8")
    resume = {
        "argv": [sys.executable, str(resume_script), str(marker), str(checkpoint), checkpoint_sha, str(progress), str(receipt)],
        "checkpoint_pins": [{"path": str(checkpoint), "sha256": checkpoint_sha}],
        "required_argv_values": [str(checkpoint), checkpoint_sha],
    }
    stage = _training_stage(
        tmp_path,
        stage_id="supervised_s0",
        role="supervised",
        script=initial,
        argv_tail=[str(marker), str(checkpoint), str(progress), str(receipt)],
        receipt=receipt,
        progress=progress,
        pin_ids=["recipe", "source", "source_resume", "train_asset"],
        source_pin_ids=["source", "source_resume"],
        resume=resume,
    )
    plan = _base_plan(tmp_path, [stage], pins)
    plan_path, plan_sha = _write_plan(tmp_path, plan)
    try:
        first = start_campaign(str(plan_path), plan_sha)
        assert first["status"] == "started"
        state = _wait_for(lambda: (s if (s := status_state(plan["state_dir"]))["active_stage"] == "supervised_s0"
                                  and s["stages"]["supervised_s0"].get("wrapper_pid") else None))
        stage_state = state["stages"]["supervised_s0"]
        execution_path = Path(stage_state["execution_path"])
        execution = _wait_for(lambda: json.loads(execution_path.read_text()) if execution_path.is_file() and json.loads(execution_path.read_text()).get("command_identity") else None)
        wrapper_pid = stage_state["wrapper_pid"]
        command_pid = execution["command_identity"]["pid"]
        supervisor = state["supervisor"]
        # /proc identity exists before the child's Python body has run. Wait
        # for its durable checkpoint, not just for the process to exist.
        _wait_for(lambda: marker.is_file() and checkpoint.is_file()
                  and sha256_file(checkpoint) == checkpoint_sha)
        assert marker.read_text(encoding="utf-8") == "initial\n"
        assert checkpoint.is_file() and sha256_file(checkpoint) == checkpoint_sha

        os.kill(supervisor["pid"], signal.SIGKILL)
        _wait_for(lambda: process_identity(supervisor["pid"]) is None)
        os.kill(wrapper_pid, signal.SIGKILL)
        _wait_for(lambda: process_identity(wrapper_pid) is None)
        os.kill(command_pid, signal.SIGKILL)
        _wait_for(lambda: process_identity(command_pid) is None)

        restarted = start_campaign(str(plan_path), plan_sha)
        if restarted["status"] == "already_running":
            restarted = _wait_for(lambda: (r if (r := start_campaign(str(plan_path), plan_sha))["status"] != "already_running" else None))
        assert restarted["status"] == "started"
        final = _wait_campaign_terminal(Path(plan["state_dir"]))
        assert final["status"] == "completed"
        assert final["stages"]["supervised_s0"]["status"] == "succeeded"
        assert final["stages"]["supervised_s0"]["attempt"] == 2
        assert final["stages"]["supervised_s0"]["mode"] == "resume"
        assert marker.read_text(encoding="utf-8").splitlines() == ["initial", "resume"]
        progress_receipts = [json.loads(line) for line in Path(plan["progress_receipts_jsonl"]).read_text().splitlines()]
        assert progress_receipts
        assert all("step" in row and "eta_seconds" in row for row in progress_receipts)
    finally:
        state_path = Path(plan["state_dir"]) / "STATE.json"
        if state_path.exists():
            current = json.loads(state_path.read_text())
            active = current.get("active_stage")
            if active:
                item = current["stages"][active]
                execution_file = Path(item.get("execution_path", "missing"))
                identities = [item.get("wrapper_identity"), current.get("supervisor")]
                if execution_file.is_file():
                    identities.insert(0, json.loads(execution_file.read_text()).get("command_identity"))
                for identity in identities:
                    if identity:
                        actual = process_identity(identity["pid"])
                        if actual and all(actual[k] == identity[k] for k in ("pid", "start_ticks", "cwd", "argv_sha256")):
                            os.kill(identity["pid"], signal.SIGKILL)


def test_teacher_quality_gate_blocks_kd_without_threshold_relaxation(tmp_path: Path):
    teacher = _script(
        tmp_path / "teacher_eval.py",
        "import json, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'status':'COMPLETED','metrics':{'Macro4':0.869}})+'\\n', encoding='utf-8')\n",
    )
    kd = _script(tmp_path / "kd.py", "import pathlib,sys\npathlib.Path(sys.argv[1]).write_text('KD RAN\\n')\n")
    pins = _mk_training_pins(tmp_path, ("source_kd", kd))
    eval_asset = _pin(tmp_path / "teacher_eval_manifest.json", "eval_asset", "asset", b"manufactured teacher evaluation inputs\n")
    pins.append(eval_asset)
    teacher_receipt = tmp_path / "teacher_metrics.json"
    kd_receipt = tmp_path / "kd_receipt.json"
    kd_marker = tmp_path / "kd.started"
    kd_progress = tmp_path / "kd.TRAIN_LOG.jsonl"
    teacher_stage = {
        "id": "teacher_evaluation",
        "kind": "command",
        "role": "evaluation",
        "requires": [],
        "pin_ids": ["eval_asset"],
        "argv": [sys.executable, str(teacher), str(teacher_receipt)],
        "cwd": str(tmp_path),
        "completion": {"path": str(teacher_receipt), "field": "status", "equals": "COMPLETED"},
    }
    kd_stage = _training_stage(
        tmp_path, stage_id="kd_student", role="kd", script=kd,
        argv_tail=[str(kd_marker), str(kd_receipt), str(kd_progress)],
        receipt=kd_receipt, progress=kd_progress,
        pin_ids=["recipe", "source", "source_kd", "train_asset"],
        source_pin_ids=["source", "source_kd"], requires=["teacher_evaluation"],
    )
    kd_stage["quality_gate"] = {
        "stage": "teacher_evaluation",
        "metric": "Macro4",
        "field": "metrics.Macro4",
        "minimum": 0.87,
    }
    plan = _base_plan(tmp_path, [teacher_stage, kd_stage], pins)
    path, digest = _write_plan(tmp_path, plan)
    state = run_foreground(str(path), digest)
    assert state["status"] == "completed_with_blocks"
    assert state["stages"]["teacher_evaluation"]["status"] == "succeeded"
    assert state["stages"]["kd_student"]["status"] == "blocked"
    assert "below_0.87" in state["stages"]["kd_student"]["reason"]
    assert not kd_marker.exists()

    relaxed = json.loads(path.read_text())
    relaxed["stages"][1]["quality_gate"]["minimum"] = 0.86
    with pytest.raises(CampaignError, match="Macro4 >= 0.87 exactly"):
        validate_campaign(relaxed)


def test_upload_pending_is_recorded_and_never_retrains_completed_stage(tmp_path: Path):
    train_script = _script(
        tmp_path / "train_fixture.py",
        "import json, pathlib, sys\n"
        "marker, receipt, progress = map(pathlib.Path, sys.argv[1:])\n"
        "marker.open('a', encoding='utf-8').write('trained\\n')\n"
        "progress.write_text(json.dumps({'step':10000,'eta_seconds':0})+'\\n', encoding='utf-8')\n"
        "receipt.write_text(json.dumps({'status':'COMPLETED','steps':10000})+'\\n', encoding='utf-8')\n",
    )
    backup_script = _script(
        tmp_path / "backup_fixture.py",
        "import json, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'status':'UPLOAD_PENDING','remote_verified':False,'local_retained':True})+'\\n', encoding='utf-8')\n"
        "raise SystemExit(2)\n",
    )
    eval_script = _script(
        tmp_path / "independent_eval.py",
        "import json, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'status':'COMPLETED'})+'\\n', encoding='utf-8')\n",
    )
    pins = _mk_training_pins(tmp_path)
    backup_asset = _pin(tmp_path / "backup_plan.json", "backup_asset", "asset", b"exact local backup manifest\n")
    pins.append(backup_asset)
    train_marker = tmp_path / "train.count"
    train_receipt = tmp_path / "train.receipt.json"
    train_progress = tmp_path / "train.jsonl"
    s0 = _training_stage(
        tmp_path, stage_id="supervised_s0", role="supervised", script=train_script,
        argv_tail=[str(train_marker), str(train_receipt), str(train_progress)],
        receipt=train_receipt, progress=train_progress,
        pin_ids=["recipe", "source", "train_asset"],
    )
    backup_receipt = tmp_path / "backup.receipt.json"
    backup = {
        "id": "core_backup",
        "kind": "command",
        "role": "backup",
        "requires": ["supervised_s0"],
        "pin_ids": ["backup_asset"],
        "argv": [sys.executable, str(backup_script), str(backup_receipt)],
        "cwd": str(tmp_path),
        "completion": {"path": str(backup_receipt), "field": "status", "equals": "VERIFIED"},
        "failure_outcomes": [{"field": "status", "equals": "UPLOAD_PENDING", "label": "UPLOAD_PENDING"}],
    }
    independent_receipt = tmp_path / "independent.receipt.json"
    independent = {
        "id": "independent_eval",
        "kind": "command",
        "role": "evaluation",
        "requires": [],
        "pin_ids": ["backup_asset"],
        "argv": [sys.executable, str(eval_script), str(independent_receipt)],
        "cwd": str(tmp_path),
        "completion": {"path": str(independent_receipt), "field": "status", "equals": "COMPLETED"},
    }
    dependent_receipt = tmp_path / "backup_dependent.receipt.json"
    dependent = {
        "id": "post_backup_eval",
        "kind": "command",
        "role": "evaluation",
        "requires": ["core_backup"],
        "pin_ids": ["backup_asset"],
        "argv": [sys.executable, str(eval_script), str(dependent_receipt)],
        "cwd": str(tmp_path),
        "completion": {"path": str(dependent_receipt), "field": "status", "equals": "COMPLETED"},
    }
    plan = _base_plan(tmp_path, [s0, backup, independent, dependent], pins, continue_independent=True)
    path, digest = _write_plan(tmp_path, plan)

    state = run_foreground(str(path), digest)
    assert state["status"] == "completed_with_blocks"
    assert state["stages"]["supervised_s0"]["status"] == "succeeded"
    assert state["stages"]["core_backup"]["status"] == "failed"
    assert state["stages"]["core_backup"]["upload_state"] == "UPLOAD_PENDING"
    assert state["stages"]["core_backup"]["remote_verified"] is False
    assert state["stages"]["core_backup"]["local_artifacts_retained"] is True
    assert state["stages"]["independent_eval"]["status"] == "succeeded"
    assert state["stages"]["post_backup_eval"]["status"] == "blocked"
    assert train_marker.read_text(encoding="utf-8") == "trained\n"

    replay = run_foreground(str(path), digest)
    assert replay["status"] == "completed_with_blocks"
    assert train_marker.read_text(encoding="utf-8") == "trained\n"


def test_oom_output_fails_stage_even_when_receipt_says_complete(tmp_path: Path):
    training = _script(
        tmp_path / "oom_fixture.py",
        "import json, pathlib, sys\n"
        "receipt, progress = map(pathlib.Path, sys.argv[1:])\n"
        "progress.write_text(json.dumps({'step':3,'eta_seconds':9})+'\\n', encoding='utf-8')\n"
        "receipt.write_text(json.dumps({'status':'COMPLETED','steps':10000})+'\\n', encoding='utf-8')\n"
        "print('CUDA out of memory', flush=True)\n",
    )
    pins = _mk_training_pins(tmp_path)
    receipt = tmp_path / "oom.receipt.json"
    progress = tmp_path / "oom.jsonl"
    stage = _training_stage(
        tmp_path, stage_id="supervised_s0", role="supervised", script=training,
        argv_tail=[str(receipt), str(progress)], receipt=receipt, progress=progress,
        pin_ids=["recipe", "source", "train_asset"],
    )
    plan = _base_plan(tmp_path, [stage], pins)
    path, digest = _write_plan(tmp_path, plan)
    state = run_foreground(str(path), digest)
    assert state["status"] == "completed_with_blocks"
    assert state["stages"]["supervised_s0"]["status"] == "failed"
    assert state["stages"]["supervised_s0"]["reason"] == "OOM_detected"


def test_progress_eta_is_written_from_configured_step_duration_jsonl(tmp_path: Path):
    training = _script(
        tmp_path / "timed_training.py",
        "import json, pathlib, sys\n"
        "receipt, progress = map(pathlib.Path, sys.argv[1:])\n"
        "with progress.open('w', encoding='utf-8') as out:\n"
        "  for step in (0, 5000, 10000): out.write(json.dumps({'step':step,'timings':{'step_elapsed_seconds':0.01}})+'\\n')\n"
        "receipt.write_text(json.dumps({'status':'COMPLETED'})+'\\n', encoding='utf-8')\n",
    )
    pins = _mk_training_pins(tmp_path)
    receipt = tmp_path / "timed.receipt.json"
    progress = tmp_path / "timed.jsonl"
    stage = _training_stage(
        tmp_path, stage_id="supervised_s0", role="supervised", script=training,
        argv_tail=[str(receipt), str(progress)], receipt=receipt, progress=progress,
        pin_ids=["recipe", "source", "train_asset"],
    )
    stage["training"]["progress_source"] = {
        "path": str(progress),
        "step_field": "step",
        "step_duration_seconds_field": "timings.step_elapsed_seconds",
    }
    plan = _base_plan(tmp_path, [stage], pins)
    path, digest = _write_plan(tmp_path, plan)
    state = run_foreground(str(path), digest)
    assert state["status"] == "completed"
    rows = [json.loads(line) for line in Path(plan["progress_receipts_jsonl"]).read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 5000, 10000]
    assert [row["eta_seconds"] for row in rows] == [100.0, 50.0, 0.0]


def test_stop_waits_for_active_stage_and_prevents_following_stage(tmp_path: Path):
    active_script = _script(
        tmp_path / "active_stage.py",
        "import json, pathlib, sys, time\n"
        "started, release, receipt = map(pathlib.Path, sys.argv[1:])\n"
        "started.write_text('started\\n', encoding='utf-8')\n"
        "while not release.exists(): time.sleep(0.02)\n"
        "receipt.write_text(json.dumps({'status':'COMPLETED'})+'\\n', encoding='utf-8')\n",
    )
    second_script = _script(tmp_path / "next_stage.py", "import pathlib,sys\npathlib.Path(sys.argv[1]).write_text('ran\\n')\n")
    started, release = tmp_path / "active.started", tmp_path / "active.release"
    first_receipt, next_receipt, next_marker = tmp_path / "first.json", tmp_path / "next.json", tmp_path / "next.started"
    first = {
        "id": "active_stage", "kind": "command", "role": "maintenance", "requires": [],
        "argv": [sys.executable, str(active_script), str(started), str(release), str(first_receipt)],
        "cwd": str(tmp_path), "completion": {"path": str(first_receipt), "field": "status", "equals": "COMPLETED"},
    }
    second = {
        "id": "following_stage", "kind": "command", "role": "maintenance", "requires": [],
        "argv": [sys.executable, str(second_script), str(next_marker)], "cwd": str(tmp_path),
        "completion": {"path": str(next_receipt), "field": "status", "equals": "COMPLETED"},
    }
    plan = _base_plan(tmp_path, [first, second], [])
    path, digest = _write_plan(tmp_path, plan)
    try:
        assert start_campaign(str(path), digest)["status"] == "started"
        state = _wait_for(lambda: (s if (s := status_state(plan["state_dir"]))["active_stage"] == "active_stage" else None))
        execution_path = Path(state["stages"]["active_stage"]["execution_path"])
        execution = _wait_for(lambda: json.loads(execution_path.read_text()) if execution_path.exists() and json.loads(execution_path.read_text()).get("command_identity") else None)
        pid = execution["command_identity"]["pid"]
        assert request_stop(plan["state_dir"])["status"] == "stop_requested"
        assert process_identity(pid) is not None
        release.write_text("finish current stage\n", encoding="utf-8")
        final = _wait_campaign_terminal(Path(plan["state_dir"]))
        assert final["status"] == "stopped"
        assert final["stages"]["active_stage"]["status"] == "succeeded"
        assert final["stages"]["following_stage"]["status"] == "stopped"
        assert not next_marker.exists()
    finally:
        release.write_text("cleanup\n", encoding="utf-8")


def test_hash_pins_and_training_budget_fail_closed(tmp_path: Path):
    script = _script(tmp_path / "noop.py", "pass\n")
    pins = _mk_training_pins(tmp_path)
    receipt = tmp_path / "receipt.json"
    progress = tmp_path / "progress.jsonl"
    stage = _training_stage(
        tmp_path, stage_id="s0", role="supervised", script=script,
        argv_tail=[str(receipt), str(progress)], receipt=receipt, progress=progress,
        pin_ids=["recipe", "source", "train_asset"],
    )
    plan = _base_plan(tmp_path, [stage], pins)
    validate_campaign(plan)
    plan["stages"][0]["training"]["steps"] = 9999
    with pytest.raises(CampaignError, match="exactly 10000"):
        validate_campaign(plan)
    plan["stages"][0]["training"]["steps"] = 10000
    plan["campaign_mode"] = "backup_recovery"
    with pytest.raises(CampaignError, match="may not contain training stages"):
        validate_campaign(plan)


def test_changed_recipe_is_refused_before_command_launch(tmp_path: Path):
    script = _script(
        tmp_path / "must_not_run.py",
        "import pathlib,sys\npathlib.Path(sys.argv[1]).write_text('launched\\n')\n",
    )
    pins = _mk_training_pins(tmp_path)
    marker, receipt, progress = tmp_path / "launched.txt", tmp_path / "receipt.json", tmp_path / "progress.jsonl"
    stage = _training_stage(
        tmp_path, stage_id="supervised_s0", role="supervised", script=script,
        argv_tail=[str(marker)], receipt=receipt, progress=progress,
        pin_ids=["recipe", "source", "train_asset"],
    )
    plan = _base_plan(tmp_path, [stage], pins)
    path, digest = _write_plan(tmp_path, plan)
    (tmp_path / "recipe.json").write_text('{"steps":9999}\n', encoding="utf-8")

    state = run_foreground(str(path), digest)
    assert state["status"] == "completed_with_blocks"
    assert state["stages"]["supervised_s0"]["status"] == "failed"
    assert "pinned recipe hash changed" in state["stages"]["supervised_s0"]["reason"]
    assert not marker.exists()


def test_argv_environment_and_gpu_policy_reject_shell_eval_tokens_and_multiple_devices(tmp_path: Path):
    script = _script(tmp_path / "noop.py", "pass\n")
    pins = _mk_training_pins(tmp_path)
    receipt, progress = tmp_path / "receipt.json", tmp_path / "progress.jsonl"
    stage = _training_stage(
        tmp_path, stage_id="s0", role="supervised", script=script,
        argv_tail=[str(receipt), str(progress)], receipt=receipt, progress=progress,
        pin_ids=["recipe", "source", "train_asset"],
    )
    plan = _base_plan(tmp_path, [stage], pins)

    plan["stages"][0]["argv"] = ["/bin/sh", "-c", "touch /tmp/unsafe"]
    with pytest.raises(CampaignError, match="may not invoke a shell"):
        validate_campaign(plan)
    plan["stages"][0]["argv"] = [sys.executable, "-c", "pass"]
    with pytest.raises(CampaignError, match="eval/command strings"):
        validate_campaign(plan)
    plan["stages"][0]["argv"] = [sys.executable, str(script), "--api-key=not-allowed"]
    with pytest.raises(CampaignError, match="credentials or token-like"):
        validate_campaign(plan)
    plan["stages"][0]["argv"] = [sys.executable, str(script)]
    plan["gpu"]["visible_device"] = "0,1"
    with pytest.raises(CampaignError, match="exactly one numeric device"):
        validate_campaign(plan)
    plan["gpu"]["visible_device"] = "0"
    plan["env_allowlist"] = {"SERVICE_TOKEN": str(tmp_path)}
    with pytest.raises(CampaignError, match="forbidden or invalid name"):
        validate_campaign(plan)
