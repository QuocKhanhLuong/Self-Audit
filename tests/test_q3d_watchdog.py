"""CPU software-only tests for the server-resident Q3D watchdog (all I/O faked)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import q3d_watchdog as wd  # noqa: E402

NOW = 1_800_000_000.0


class FakeSystem:
    """Fake ``subprocess.run`` covering supervisorctl and ps."""

    def __init__(self, state="RUNNING", processes=(), ps_ok=True, start_rc=0):
        self.state = state
        self.processes = list(processes)
        self.ps_ok = ps_ok
        self.start_rc = start_rc
        self.calls: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.calls.append(list(command))
        assert kwargs.get("timeout"), "every external command must be bounded"
        if command[0] == "ps":
            if not self.ps_ok:
                return SimpleNamespace(returncode=1, stdout="", stderr="ps: failed")
            lines = [f"{pid} {stat} {args}" for pid, stat, args in self.processes]
            return SimpleNamespace(returncode=0, stdout="\n".join(lines) + "\n", stderr="")
        if command[-2:] == ["status", "q3d-campaign"]:
            detail = "pid 4242, uptime 1:00:00" if self.state == "RUNNING" else "Nov 12 10:00 AM"
            rc = 0 if self.state == "RUNNING" else 3
            return SimpleNamespace(returncode=rc, stdout=f"q3d-campaign   {self.state}   {detail}\n", stderr="")
        if command[-2:] == ["start", "q3d-campaign"]:
            if self.start_rc == 0:
                self.state = "RUNNING"
            return SimpleNamespace(returncode=self.start_rc, stdout="q3d-campaign: started\n", stderr="")
        raise AssertionError(f"unexpected command {command}")

    @property
    def starts(self):
        return [call for call in self.calls if call[-2:] == ["start", "q3d-campaign"]]


def _touch(path: Path, when: float) -> None:
    os.utime(path, (when, when))


def make_base(tmp_path: Path, *, queue_status="RUNNING", stage="J", age=60.0, metric=True) -> Path:
    base = tmp_path / "base"
    campaign = base / "campaign-v2"
    output = base / "jobs" / f"v1-k8-{stage}"
    output.mkdir(parents=True)
    campaign.mkdir(parents=True, exist_ok=True)
    (base / "QUEUE_STATUS.json").write_text(json.dumps({
        "status": queue_status, "setting": "v1", "k": 8, "through_stage": "J",
        "updated_utc": "2027-01-15T08:00:00+00:00", "error": None}))
    job_id = f"v1-k8-{stage}"
    (campaign / "PLAN.json").write_text(json.dumps({"stage_jobs": [
        {"id": job_id, "stage": stage, "output": str(output), "experiment_id": "v1-k8"}]}))
    (campaign / "RUN_STATE.json").write_text(json.dumps({"jobs": {
        "v1-k8-T": {"status": "COMPLETED", "exit_code": 0, "command": ["python", "train.py", "--secret"]},
        job_id: {"status": "RUNNING", "exit_code": None, "command": ["python", "train.py", "/data/case_001"]},
    }}))
    if metric:
        (output / "metrics.jsonl").write_text(
            json.dumps({"event": "train", "stage": stage, "update": 10, "loss": 0.5}) + "\n"
            + json.dumps({"event": "development_epoch", "stage": stage, "epoch": 3, "update": 120,
                          "loss": 0.25, "patient_id": "case_001", "case_ids": ["case_001"],
                          "per_class": {"RV": 0.8, "MYO": 0.7, "LV": 0.9, "case_001": 0.1}}) + "\n")
    for path in [base / "QUEUE_STATUS.json", campaign / "RUN_STATE.json", campaign / "PLAN.json",
                 output, *output.iterdir()]:
        _touch(path, NOW - age)
    return base


def run(base, system, **kwargs):
    kwargs.setdefault("now", lambda: NOW)
    return wd.check_once(base, runner=system, **kwargs)


TRAIN = (777, "Rl", "/venv/main/bin/python /root/Self-Audit/train.py --config c.yaml --out /x")
QUEUE = (555, "Sl", "/venv/main/bin/python -u /root/mount_data/runs/q3d-resenc-pixdim-20261112/run_ordered.py a b")


def test_healthy_running_writes_status_and_jsonl(tmp_path):
    base = make_base(tmp_path)
    system = FakeSystem("RUNNING", [TRAIN, QUEUE])
    record = run(base, system)
    assert record["health"] == "OK" and record["alerts"] == []
    assert record["status"] == "OK" and record["attention_required"] is False
    assert record["updated_utc"] == record["checked_utc"]
    assert record["current_stage"] == "J"
    assert record["supervisor"] == {"program": "q3d-campaign", "state": "RUNNING", "pid": 4242,
                                    "returncode": 0, "detail": "status_observed"}
    assert {p["kind"] for p in record["processes"]["matching"]} == {"train", "queue"}
    assert record["metrics"]["last_record"]["epoch"] == 3
    assert record["metrics"]["last_update"] == 120
    assert record["metrics"]["mtime_utc"] is not None
    assert system.starts == []
    written = json.loads((base / "MONITOR_STATUS.json").read_text())
    assert written["health"] == "OK"
    lines = (base / "monitor.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["health"] == "OK"
    text = (base / "MONITOR_STATUS.json").read_text() + lines[0]
    for forbidden in ("case_001", "patient_id", "--secret", "c.yaml", "/data/", "--config"):
        assert forbidden not in text
    assert record["metrics"]["last_record"]["per_class"] == {"LV": 0.9, "MYO": 0.7, "RV": 0.8}
    assert not list(base.glob(".*.tmp"))


@pytest.mark.parametrize("state", ["EXITED", "FATAL", "STOPPED"])
def test_exited_supervisor_is_recovered_once(tmp_path, state):
    base = make_base(tmp_path, queue_status="FAILED")
    system = FakeSystem(state, [])
    record = run(base, system)
    assert system.starts == [["supervisorctl", "start", "q3d-campaign"]]
    assert record["restart"]["attempted"] and record["restart"]["result"]["ok"]
    assert record["health"] == "RECOVERING"
    assert record["status"] == "RECOVERING" and record["attention_required"] is False
    history = json.loads((base / "MONITOR_STATUS.json").read_text())["restart"]["history"]
    assert len(history) == 1 and history[0]["from_state"] == state


def test_live_writer_prevents_restart(tmp_path):
    base = make_base(tmp_path)
    system = FakeSystem("EXITED", [TRAIN])
    record = run(base, system)
    assert system.starts == []
    assert "live train.py/queue process present" in record["restart"]["blocked_by"]
    assert "SUPERVISOR_DOWN_BUT_WRITER_ALIVE" in record["alerts"]
    assert record["health"] == "ALERT"


def test_queue_writer_alone_also_prevents_restart(tmp_path):
    base = make_base(tmp_path)
    system = FakeSystem("FATAL", [QUEUE])
    assert run(base, system)["restart"]["attempted"] is False
    assert system.starts == []


def test_failed_process_listing_prevents_restart(tmp_path):
    base = make_base(tmp_path)
    system = FakeSystem("EXITED", ps_ok=False)
    record = run(base, system)
    assert system.starts == [] and "PROCESS_LISTING_FAILED" in record["alerts"]


@pytest.mark.parametrize("output", ["unix:///var/run/supervisor.sock refused connection",
                                    "q3d-campaignX EXITED Nov 12", "other-program EXITED Nov 12"])
def test_unknown_supervisor_state_never_restarts(tmp_path, output):
    base = make_base(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[0] == "ps":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=2, stdout=output, stderr="")

    record = wd.check_once(base, runner=runner, now=lambda: NOW)
    assert record["supervisor"]["state"] == "UNKNOWN"
    assert not any(c[-1:] == ["q3d-campaign"] and "start" in c for c in calls)


def test_completed_queue_records_final_and_never_restarts(tmp_path):
    base = make_base(tmp_path, queue_status="COMPLETED")
    system = FakeSystem("EXITED", [])
    record = run(base, system)
    assert record["health"] == "COMPLETED" and record["final"] is True
    assert record["attention_required"] is True and record["status"] == "COMPLETED"
    assert system.starts == []
    assert "queue COMPLETED" in record["restart"]["blocked_by"]
    code = wd.main(["--base", str(base)], runner=system, now=lambda: NOW,
                   sleep=lambda _: pytest.fail("loop must exit on COMPLETED"))
    assert code == 0
    assert json.loads((base / "MONITOR_STATUS.json").read_text())["final"] is True


def test_missing_queue_status_never_restarts(tmp_path):
    base = make_base(tmp_path)
    (base / "QUEUE_STATUS.json").unlink()
    system = FakeSystem("EXITED", [])
    record = run(base, system)
    assert system.starts == [] and "QUEUE_STATUS_MISSING" in record["alerts"]


def test_stale_alive_job_alerts_without_kill_or_restart(tmp_path):
    base = make_base(tmp_path, age=1801 + 60)
    system = FakeSystem("RUNNING", [TRAIN])
    record = run(base, system)
    assert "STALE_NO_PROGRESS" in record["alerts"] and record["health"] == "ALERT"
    assert record["attention_required"] is True
    assert all(call[0] in {"ps", "supervisorctl"} and "start" not in call for call in system.calls)
    assert not any(call[0] in {"kill", "pkill"} or "stop" in call for call in system.calls)
    assert wd.main(["--base", str(base), "--once"], runner=system, now=lambda: NOW) == wd.EXIT_ALERT


def test_stage_p_without_metrics_uses_output_activity_and_30_minute_threshold(tmp_path):
    base = make_base(tmp_path, stage="P", metric=False, age=1200)
    system = FakeSystem("RUNNING", [TRAIN])
    record = run(base, system)
    assert record["current_stage"] == "P" and record["metrics"]["last_record"] is None
    assert record["progress"]["stale_threshold_seconds"] == wd.DEFAULT_P_STALE
    assert wd.DEFAULT_P_STALE == 1800
    assert record["alerts"] == []
    later = run(base, system, now=lambda: NOW + wd.DEFAULT_P_STALE)
    assert "STALE_NO_PROGRESS_STAGE_P" in later["alerts"]


def test_corrupt_history_remains_blocked_across_cycles(tmp_path):
    base = make_base(tmp_path)
    (base / "MONITOR_STATUS.json").write_text("not-json")
    system = FakeSystem("EXITED")
    for delta in (0, 420, 7200):
        record = run(base, system, now=lambda: NOW + delta)
        assert record["restart"]["history_error"]
        assert record["health"] == "ALERT"
    assert not system.starts


@pytest.mark.parametrize("history", [None, {}, [{"epoch": "bad"}], [{"epoch": float("nan")} ]])
def test_malformed_history_never_silently_resets_budget(tmp_path, history):
    base = make_base(tmp_path)
    (base / "MONITOR_STATUS.json").write_text(json.dumps({"restart": {"history": history}}))
    system = FakeSystem("EXITED")
    assert run(base, system)["restart"]["history_error"]
    assert not system.starts


def test_restart_budget_reserved_before_crash_and_survives_next_cycle(tmp_path):
    base = make_base(tmp_path)
    system = FakeSystem("EXITED")

    def crash(command, **kwargs):
        if "start" in command:
            saved = json.loads((base / "MONITOR_STATUS.json").read_text())
            assert len(saved["restart"]["history"]) == 1
            assert saved["health"] == "RECOVERY_PENDING"
            assert json.loads((base / "monitor.jsonl").read_text().splitlines()[0])["event"] == "restart_reserved"
            raise KeyboardInterrupt("simulate watchdog death")
        return system(command, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        wd.check_once(base, runner=crash, now=lambda: NOW)
    record = run(base, system, now=lambda: NOW + 1)
    assert not system.starts
    assert any("backoff" in reason for reason in record["restart"]["blocked_by"])


def test_writer_appearing_during_safety_recheck_prevents_start(tmp_path):
    base = make_base(tmp_path)
    system = FakeSystem("EXITED")
    calls = 0

    def runner(command, **kwargs):
        nonlocal calls
        if command[0] == "ps":
            calls += 1
            if calls >= 2:
                system.processes = [TRAIN]
        return system(command, **kwargs)

    record = wd.check_once(base, runner=runner, now=lambda: NOW)
    assert not system.starts
    assert "recheck" in record["restart"]["result"]["detail"]


@pytest.mark.parametrize("status", [None, "", "unrecognized", {"bad": "status"}])
def test_missing_or_invalid_queue_status_is_not_positive_permission(tmp_path, status):
    base = make_base(tmp_path)
    (base / "QUEUE_STATUS.json").write_text(json.dumps({"status": status}))
    system = FakeSystem("EXITED")
    record = run(base, system)
    assert not record["queue"]["readable"] and not system.starts


def test_progress_log_noise_cannot_mask_stale_metrics(tmp_path):
    base = make_base(tmp_path, age=1900)
    (base / "launch.log").write_text("heartbeat only")
    _touch(base / "launch.log", NOW)
    record = run(base, FakeSystem("RUNNING", [TRAIN]))
    assert "STALE_NO_PROGRESS" in record["alerts"]


def test_external_output_and_queue_error_do_not_leak_patient_details(tmp_path):
    base = make_base(tmp_path, queue_status="FAILED")
    queue_path = base / "QUEUE_STATUS.json"
    data = json.loads(queue_path.read_text())
    data["error"] = "patient Alice /patient/private.nii failed"
    queue_path.write_text(json.dumps(data))
    system = FakeSystem("EXITED")

    def runner(command, **kwargs):
        result = system(command, **kwargs)
        if "start" in command:
            result.stderr = "patient Alice /patient/private.nii"
        return result

    wd.check_once(base, runner=runner, now=lambda: NOW)
    for path in (base / "MONITOR_STATUS.json", base / "monitor.jsonl"):
        assert "Alice" not in path.read_text() and "/patient/" not in path.read_text()


def test_malformed_successful_process_listing_blocks_restart(tmp_path):
    base = make_base(tmp_path)
    system = FakeSystem("EXITED")

    def runner(command, **kwargs):
        if command[0] == "ps":
            return SimpleNamespace(returncode=0, stdout="unparseable process table", stderr="")
        return system(command, **kwargs)

    record = wd.check_once(base, runner=runner, now=lambda: NOW)
    assert not record["processes"]["listing_ok"] and not system.starts


def test_queue_completing_during_observation_prevents_restart(tmp_path):
    base = make_base(tmp_path)
    system = FakeSystem("EXITED")

    def runner(command, **kwargs):
        if command[0] == "ps":
            (base / "QUEUE_STATUS.json").write_text(json.dumps({"status": "COMPLETED"}))
        return system(command, **kwargs)

    wd.check_once(base, runner=runner, now=lambda: NOW)
    assert not system.starts


def test_start_timeout_consumes_budget(tmp_path):
    base = make_base(tmp_path)
    system = FakeSystem("EXITED")
    starts = []

    def runner(command, **kwargs):
        if "start" in command:
            starts.append(command)
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return system(command, **kwargs)

    first = wd.check_once(base, runner=runner, now=lambda: NOW)
    second = wd.check_once(base, runner=runner, now=lambda: NOW + 1)
    assert len(starts) == 1
    assert first["restart"]["result"]["ok"] is False
    assert len(second["restart"]["history"]) == 1
    assert "RESTART_BUDGET_BLOCKED" in second["alerts"]


def test_deleted_status_does_not_reset_attempt_budget(tmp_path):
    base = make_base(tmp_path)
    system = FakeSystem("EXITED", start_rc=1)
    run(base, system)
    (base / "MONITOR_STATUS.json").unlink()
    record = run(base, system, now=lambda: NOW + 1)
    assert len(system.starts) == 1 and record["restart"]["history_error"]


def test_no_restart_flag_reports_but_does_not_start(tmp_path):
    base = make_base(tmp_path)
    system = FakeSystem("EXITED", [])
    record = run(base, system, no_restart=True)
    assert system.starts == []
    assert record["restart"]["would_restart"] is True
    assert "RESTART_SUPPRESSED" in record["alerts"]


def test_restart_is_bounded_with_backoff_and_hourly_budget(tmp_path):
    base = make_base(tmp_path)
    system = FakeSystem("EXITED", [], start_rc=1)  # start keeps failing
    clock = {"t": NOW}
    attempts = []
    for step in range(40):  # 40 checks, 120 s apart = 80 min
        clock["t"] = NOW + step * 120
        record = run(base, system, now=lambda: clock["t"])
        if record["restart"]["attempted"]:
            attempts.append(clock["t"] - NOW)
    first_hour = [a for a in attempts if a < 3600]
    assert len(first_hour) == 3
    assert attempts[1] - attempts[0] >= 300 and attempts[2] - attempts[1] >= 600
    assert all(c == ["supervisorctl", "start", "q3d-campaign"] for c in system.starts)
    blocked = run(base, system, now=lambda: attempts[2] + NOW + 1)
    assert any("budget" in reason or "backoff" in reason for reason in blocked["restart"]["blocked_by"])


def test_lifetime_budget_and_unreadable_history_block(tmp_path):
    base = make_base(tmp_path)
    history = [{"epoch": NOW - 7200 - i, "utc": "x", "ok": False} for i in range(12)]
    (base / "MONITOR_STATUS.json").write_text(json.dumps({"restart": {"history": history}}))
    system = FakeSystem("EXITED", [])
    record = run(base, system)
    assert system.starts == [] and "RESTART_BUDGET_BLOCKED" in record["alerts"]
    (base / "MONITOR_STATUS.json").write_text("{corrupt")
    record = run(base, system)
    assert system.starts == []
    assert any("unreadable" in reason for reason in record["restart"]["blocked_by"])


def test_lock_held_blocks_restart_and_writes(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    base = make_base(tmp_path)
    with (base / "MONITOR.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        system = FakeSystem("EXITED", [])
        record = run(base, system)
    assert system.starts == [] and record["lock_acquired"] is False
    assert not (base / "MONITOR_STATUS.json").exists()


def test_program_name_is_not_configurable():
    parser = wd.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--program", "other"])
    assert wd.PROGRAM == "q3d-campaign"


def test_supervisor_config_is_forwarded(tmp_path):
    base = make_base(tmp_path)
    system = FakeSystem("RUNNING", [TRAIN])
    assert wd.main(["--base", str(base), "--once", "--supervisor-config", "/etc/sv.conf"],
                   runner=system, now=lambda: NOW) == 0
    assert ["supervisorctl", "-c", "/etc/sv.conf", "status", "q3d-campaign"] in system.calls


def test_recovery_executable_must_be_root_owned_and_safe(tmp_path):
    script = tmp_path / "recover.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    uid = os.getuid()
    assert wd.validate_recovery_executable(str(script), expected_uid=uid) is None
    assert "owned by root" in wd.validate_recovery_executable(str(script), expected_uid=uid + 1)
    assert "absolute" in wd.validate_recovery_executable("recover.sh", expected_uid=uid)
    script.chmod(0o777)
    assert "writable" in wd.validate_recovery_executable(str(script), expected_uid=uid)
    link = tmp_path / "link"
    link.symlink_to(script)
    assert "regular file" in wd.validate_recovery_executable(str(link), expected_uid=uid)
    base = make_base(tmp_path)
    assert wd.main(["--base", str(base), "--once", "--recovery-executable", str(script)],
                   runner=FakeSystem(), now=lambda: NOW, recovery_uid=uid) == wd.EXIT_INTERNAL


def test_recovery_executable_runs_before_start_and_failure_skips_start(tmp_path):
    script = tmp_path / "recover.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    base = make_base(tmp_path)
    system = FakeSystem("EXITED", [])
    seen = []

    def runner(command, **kwargs):
        if command[0] == str(script):
            seen.append(command)
            return SimpleNamespace(returncode=seen_rc["rc"], stdout="", stderr="")
        return system(command, **kwargs)

    seen_rc = {"rc": 1}
    record = wd.check_once(base, runner=runner, now=lambda: NOW, recovery_executable=str(script),
                           recovery_uid=os.getuid())
    assert seen == [[str(script), "--base", str(base)]] and system.starts == []
    assert record["restart"]["result"]["ok"] is False
    seen_rc["rc"] = 0
    record = wd.check_once(base, runner=runner, now=lambda: NOW + 3600 * 2, recovery_executable=str(script),
                           recovery_uid=os.getuid())
    assert system.starts == [["supervisorctl", "start", "q3d-campaign"]]


def test_classify_process_ignores_non_python_and_watchdog():
    base = Path(wd.DEFAULT_BASE)
    assert wd.classify_process(f"tail -f {base}/launch.log", base) is None
    assert wd.classify_process(f"/venv/main/bin/python scripts/q3d_watchdog.py --base {base}", base) is None
    assert wd.classify_process("python3 train.py --config x", base) == "train"
    assert wd.classify_process("/usr/bin/vim train.py", base) is None
    assert wd.classify_process("python3 /opt/q3d_resume_queue.py", base) == "queue"
    assert wd.classify_process("python3 train.py --out /tmp/q3d_watchdog.py", base) == "train"


def test_timeouts_and_missing_supervisorctl_fail_closed(tmp_path):
    base = make_base(tmp_path)

    def runner(command, **kwargs):
        if command[0] == "ps":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "status" in command:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        raise AssertionError("start must not be called")

    record = wd.check_once(base, runner=runner, now=lambda: NOW)
    assert record["supervisor"]["state"] == "UNKNOWN" and record["supervisor"]["detail"] == "timeout"


def test_cli_validates_base(tmp_path):
    assert wd.main(["--base", "relative/path", "--once"], runner=FakeSystem()) == wd.EXIT_INTERNAL
    assert wd.main(["--base", str(tmp_path / "missing"), "--once"], runner=FakeSystem()) == wd.EXIT_INTERNAL
    assert wd.main(["--base", str(tmp_path), "--once", "--interval-seconds", "0"],
                   runner=FakeSystem()) == wd.EXIT_INTERNAL
    assert wd.build_parser().parse_args([]).interval_seconds == 420
