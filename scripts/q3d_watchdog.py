#!/usr/bin/env python3
"""Server-resident, fail-closed watchdog for the supervised Q3D campaign queue.

Every ``--interval-seconds`` (default 420) the watchdog reads, without
modifying them:

* ``BASE/QUEUE_STATUS.json`` written by the queue driver,
* ``BASE/campaign-v2/RUN_STATE.json`` (and ``PLAN.json`` for job outputs),
* the newest ``metrics.jsonl`` of the running job(s),
* ``supervisorctl status q3d-campaign``,
* the process table (``ps``), to find live ``train.py`` / queue writers.

It writes only ``BASE/MONITOR_STATUS.json`` (atomic replace),
``BASE/monitor.jsonl`` (append) and the advisory lock ``BASE/MONITOR.lock``.

Recovery is deliberately narrow.  The only mutation the watchdog can perform is
``supervisorctl start q3d-campaign`` (the program name is a constant, never an
argument), and only when ALL of the following hold:

* supervisor positively reports ``EXITED``, ``FATAL`` or ``STOPPED``;
* ``QUEUE_STATUS.json`` is readable and its status is not ``COMPLETED``;
* the process listing succeeded and shows no live ``train.py`` or Q3D queue
  process;
* the monitor lock was acquired and the previous restart history is readable;
* the bounded budget allows it (default 3 attempts per rolling hour, 12 in
  total, exponential back-off between attempts);
* ``--no-restart`` was not given.

The watchdog never kills or signals any process.  A job that is alive but makes
no observable progress for ``--stale-seconds`` (default 30 min; stage P uses
``--p-stale-seconds`` because it may not emit metrics) only raises an alert.

When the queue reports ``COMPLETED`` the watchdog records a final status and
exits 0.  Output contains job ids, stages, counters and whitelisted numeric
metrics only; commands, manifests and case identifiers are never copied.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import signal
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SCHEMA = "q3d.watchdog.status.v1"
VERSION = "1"
PROGRAM = "q3d-campaign"  # the only supervisor program this watchdog may start
DEFAULT_BASE = "/root/mount_data/runs/q3d-resenc-pixdim-20261112"
DEFAULT_INTERVAL = 420
DEFAULT_STALE = 1800
DEFAULT_P_STALE = DEFAULT_STALE
RESTARTABLE_STATES = frozenset({"EXITED", "FATAL", "STOPPED"})
ALIVE_STATES = frozenset({"RUNNING", "STARTING", "BACKOFF", "STOPPING"})
KNOWN_STATES = RESTARTABLE_STATES | ALIVE_STATES | {"UNKNOWN"}
QUEUE_SCRIPT_MARKERS = ("run_ordered.py", "resume_queue.py", "q3d_resume_queue.py",
                        "q3d_campaign.py", "q3d_profile_gpu.py")
SAFE_METRIC_SCALARS = frozenset({
    "event", "stage", "epoch", "update", "step", "loss", "total_loss", "lr",
    "score", "selection_score", "mean_dice", "dev_dice", "dice", "macro_dice",
    "perplexity", "dead_code_fraction", "epoch_logged", "global_update",
})
TAIL_BYTES = 65536
OUTPUT_SCAN_LIMIT = 2000
COMMAND_TIMEOUT = 30
EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_ALERT = 3

Runner = Callable[..., Any]


# --------------------------------------------------------------------- helpers
def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return _dt.datetime.fromtimestamp(epoch, _dt.timezone.utc).isoformat()


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _excerpt(text: Any, limit: int = 200) -> str | None:
    if text is None:
        return None
    value = " ".join(str(text).split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _read_json(path: Path) -> tuple[Any, str | None]:
    """Return ``(document, error)``; error is ``"missing"`` or a short reason."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeError) as exc:
        return None, f"unreadable: {exc.__class__.__name__}"
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        return None, "invalid_json"


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sanitize_metric(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only whitelisted scalar fields and numeric per-class summaries."""
    safe: dict[str, Any] = {}
    for key in sorted(SAFE_METRIC_SCALARS):
        value = record.get(key)
        if key in {"event", "stage"}:
            if isinstance(value, str) and (
                (key == "stage" and value in {"T", "Q", "P", "J"})
                or (key == "event" and value in {
                    "train", "train_update", "train_epoch", "development_epoch",
                    "validation", "stage_start", "stage_end", "epoch",
                })
            ):
                safe[key] = _excerpt(value, 40)
        elif isinstance(value, bool) or _finite_number(value):
            safe[key] = value
    for key in ("per_class", "teacher_dev_per_class"):
        classes = record.get(key)
        if isinstance(classes, Mapping):
            filtered = {name: value for name, value in classes.items()
                        if name in {"BG", "RV", "MYO", "LV"} and _finite_number(value)}
            if filtered:
                safe[key] = dict(sorted(filtered.items())[:8])
    return safe


def last_jsonl_record(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - TAIL_BYTES))
            chunk = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(chunk.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, Mapping):
            return dict(record)
    return None


def newest_mtime_under(directory: Path, limit: int = OUTPUT_SCAN_LIMIT) -> float | None:
    """Newest mtime among ``directory`` and entries up to two levels deep (bounded)."""
    newest = _mtime(directory)
    if newest is None:
        return None
    seen = 0
    stack = [(directory, 0)]
    while stack and seen < limit:
        current, depth = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            seen += 1
            if seen > limit:
                break
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            newest = max(newest, info.st_mtime)
            if depth < 1 and stat.S_ISDIR(info.st_mode):
                stack.append((Path(entry.path), depth + 1))
    return newest


def _run(runner: Runner, command: Sequence[str], timeout: int = COMMAND_TIMEOUT) -> tuple[int | None, str]:
    try:
        result = runner(list(command), capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return None, "executable_not_found"
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except OSError as exc:
        return None, f"os_error: {exc.__class__.__name__}"
    output = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
    return getattr(result, "returncode", None), output


# ---------------------------------------------------------------- observations
def supervisor_status(runner: Runner, supervisorctl: Sequence[str]) -> dict[str, Any]:
    code, output = _run(runner, [*supervisorctl, "status", PROGRAM])
    info: dict[str, Any] = {"program": PROGRAM, "state": "UNKNOWN", "pid": None,
                            "returncode": code, "detail": None}
    if code not in {0, 3}:
        info["detail"] = output if output in {"timeout", "executable_not_found"} else "status_command_failed"
        return info
    for line in output.splitlines():
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0] == PROGRAM and tokens[1] in KNOWN_STATES:
            info["state"] = tokens[1]
            match = re.search(r"\bpid (\d+)", line)
            info["pid"] = int(match.group(1)) if match else None
            info["detail"] = "status_observed"
            return info
    info["detail"] = "no_status_line"
    return info


def _is_python(argv0: str) -> bool:
    return os.path.basename(argv0).startswith("python")


def classify_process(args: str, base: Path) -> str | None:
    try:
        tokens = shlex.split(args)
    except ValueError:
        tokens = args.split()
    if not tokens:
        return None
    if not any(_is_python(token) for token in tokens[:2]):
        return None
    if any(os.path.basename(token) == "train.py" for token in tokens):
        return "train"
    if any(os.path.basename(token) == "q3d_watchdog.py" for token in tokens):
        return None
    if any(marker in args for marker in QUEUE_SCRIPT_MARKERS) or str(base) in args:
        return "queue"
    return None


def live_writers(runner: Runner, base: Path) -> dict[str, Any]:
    code, output = _run(runner, ["ps", "-eo", "pid=,stat=,args="])
    if code != 0:
        return {"listing_ok": False, "matching": [], "detail": "process_listing_failed"}
    own = os.getpid()
    matching = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit():
            return {"listing_ok": False, "matching": matching, "detail": "malformed_process_listing"}
        pid, state, args = int(parts[0]), parts[1], parts[2]
        if pid == own or state.startswith("Z"):
            continue
        kind = classify_process(args, base)
        if kind is not None:
            matching.append({"pid": pid, "kind": kind})
    return {"listing_ok": True, "matching": matching, "detail": None}


def queue_status(base: Path, now: float) -> dict[str, Any]:
    path = base / "QUEUE_STATUS.json"
    document, error = _read_json(path)
    mtime = _mtime(path)
    info: dict[str, Any] = {"readable": False, "status": None, "error": error,
                            "mtime_utc": _iso(mtime),
                            "age_seconds": None if mtime is None else round(now - mtime, 1)}
    if error is None and not isinstance(document, Mapping):
        info["error"] = "not_an_object"
    if info["error"] is not None:
        return info
    status = document.get("status")
    if not isinstance(status, str) or status not in {
        "RUNNING", "FAILED", "COMPLETED", "STARTING", "PENDING", "NOT_RUN", "READY"
    }:
        info["error"] = "invalid_status"
        return info
    info["readable"] = True
    info["status"] = status if isinstance(status, str) else None
    for key in ("setting", "k", "through_stage", "current_stage", "updated_utc"):
        value = document.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            info[key] = value if not isinstance(value, str) else _excerpt(value, 64)
    info["has_error"] = document.get("error") is not None
    return info


def run_state(base: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarise RUN_STATE.json; also return the running jobs joined with PLAN outputs."""
    campaign = base / "campaign-v2"
    state_path = campaign / "RUN_STATE.json"
    document, error = _read_json(state_path)
    plan, _ = _read_json(campaign / "PLAN.json")
    plan_jobs: dict[str, Mapping[str, Any]] = {}
    if isinstance(plan, Mapping) and isinstance(plan.get("stage_jobs"), list):
        plan_jobs = {job["id"]: job for job in plan["stage_jobs"]
                     if isinstance(job, Mapping) and isinstance(job.get("id"), str)}
    mtime = _mtime(state_path)
    info: dict[str, Any] = {"readable": error is None and isinstance(document, Mapping),
                            "error": error, "mtime_utc": _iso(mtime), "counts": {},
                            "running_jobs": [], "failed_jobs": []}
    running: list[dict[str, Any]] = []
    if not info["readable"]:
        return info, running
    jobs = document.get("jobs") if isinstance(document.get("jobs"), Mapping) else {}
    for job_id, record in sorted(jobs.items()):
        if not isinstance(record, Mapping):
            continue
        status = record.get("status") if isinstance(record.get("status"), str) else "UNKNOWN"
        info["counts"][status] = info["counts"].get(status, 0) + 1
        plan_job = plan_jobs.get(job_id, {})
        stage = plan_job.get("stage")
        if not isinstance(stage, str) or stage not in {"T", "Q", "P", "J"}:
            stage = None
        if status == "RUNNING":
            entry = {"id": job_id, "stage": stage}
            info["running_jobs"].append(entry)
            output = plan_job.get("output")
            output_path = Path(output) if isinstance(output, str) else None
            if output_path is not None and not output_path.is_absolute():
                output_path = campaign / output_path
            running.append({**entry, "output": output_path})
        elif status == "FAILED":
            code = record.get("exit_code")
            info["failed_jobs"].append({"id": job_id, "stage": stage,
                                        "exit_code": code if isinstance(code, int) else None})
    return info, running


def metrics_view(running: Iterable[Mapping[str, Any]], now: float) -> tuple[dict[str, Any], float | None]:
    """Return the freshest running-job metric summary and newest output activity."""
    best: dict[str, Any] = {"job_id": None, "stage": None, "mtime_utc": None,
                            "age_seconds": None, "last_record": None, "last_update": None}
    best_mtime: float | None = None
    activity: float | None = None
    for job in running:
        output = job.get("output")
        if not isinstance(output, Path):
            continue
        dir_mtime = newest_mtime_under(output)
        if dir_mtime is not None:
            activity = dir_mtime if activity is None else max(activity, dir_mtime)
        metrics = output / "metrics.jsonl"
        mtime = _mtime(metrics)
        if mtime is None or (best_mtime is not None and mtime <= best_mtime):
            if best["job_id"] is None:
                best.update(job_id=job.get("id"), stage=job.get("stage"))
            continue
        best_mtime = mtime
        record = last_jsonl_record(metrics)
        safe = sanitize_metric(record) if record else None
        best.update(job_id=job.get("id"), stage=(safe or {}).get("stage", job.get("stage")), mtime_utc=_iso(mtime),
                    age_seconds=round(now - mtime, 1),
                    last_record=safe,
                    last_update=next(((safe or {})[key] for key in ("update", "global_update", "step")
                                      if key in (safe or {})), None))
    return best, activity


# -------------------------------------------------------------------- recovery
def validate_recovery_executable(path: str, expected_uid: int = 0) -> str | None:
    """Return None when ``path`` is an absolute, root-owned, non-writable executable."""
    candidate = Path(path)
    if not candidate.is_absolute():
        return "recovery executable must be an absolute path"
    try:
        info = os.stat(candidate, follow_symlinks=False)
    except OSError:
        return "recovery executable does not exist"
    if not stat.S_ISREG(info.st_mode):
        return "recovery executable must be a regular file (no symlink)"
    if info.st_uid != expected_uid:
        return "recovery executable must be owned by root"
    if info.st_mode & 0o022:
        return "recovery executable must not be group/world writable"
    if not info.st_mode & 0o111:
        return "recovery executable is not executable"
    return None


def _previous_history(status_path: Path) -> tuple[list[dict[str, Any]], str | None]:
    document, error = _read_json(status_path)
    if error == "missing":
        if status_path.with_name("monitor.jsonl").exists():
            return [], "previous MONITOR_STATUS.json missing after monitoring began"
        return [], None
    if error is not None or not isinstance(document, Mapping):
        return [], "previous MONITOR_STATUS.json unreadable; restart history unknown"
    restart = document.get("restart")
    if not isinstance(restart, Mapping):
        return [], "restart history malformed"
    if restart.get("history_error"):
        return [], "restart history previously unreadable; operator repair required"
    history = restart.get("history")
    if not isinstance(history, list) or any(
        not isinstance(entry, Mapping) or not _finite_number(entry.get("epoch")) for entry in history
    ):
        return [], "restart history malformed"
    return [{key: entry[key] for key in ("epoch", "utc", "from_state", "ok") if key in entry}
            for entry in history], None


def restart_budget(history: Sequence[Mapping[str, Any]], now: float, *, per_hour: int,
                   total: int, backoff_seconds: int) -> str | None:
    recent = [entry for entry in history if now - entry["epoch"] < 3600]
    if len(history) >= total:
        return f"lifetime restart budget exhausted ({len(history)}/{total})"
    if len(recent) >= per_hour:
        return f"hourly restart budget exhausted ({len(recent)}/{per_hour})"
    if recent:
        wait = backoff_seconds * (2 ** (len(recent) - 1))
        elapsed = now - max(entry["epoch"] for entry in recent)
        if elapsed < wait:
            return f"backoff active ({int(elapsed)}s of {wait}s since last attempt)"
    return None


class MonitorLock:
    """Non-blocking advisory lock (fcntl) on ``BASE/MONITOR.lock``."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def acquire(self) -> bool:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX
            return False
        try:
            self.handle = self.path.open("a")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            self.release()
            return False

    def release(self) -> None:
        if self.handle is not None:
            try:
                self.handle.close()
            finally:
                self.handle = None


# ----------------------------------------------------------------- one cycle
def check_once(base: Path, *, runner: Runner = subprocess.run, now: Callable[[], float] = time.time,
               no_restart: bool = False, interval: int = DEFAULT_INTERVAL,
               stale_seconds: int = DEFAULT_STALE, p_stale_seconds: int = DEFAULT_P_STALE,
               max_restarts_per_hour: int = 3, max_restarts_total: int = 12,
               backoff_seconds: int = 300, supervisorctl: Sequence[str] = ("supervisorctl",),
               recovery_executable: str | None = None, recovery_uid: int = 0,
               once: bool = False) -> dict[str, Any]:
    """Observe, decide, maybe restart, and persist one status record."""
    base = Path(base)
    status_path = base / "MONITOR_STATUS.json"
    lock = MonitorLock(base / "MONITOR.lock")
    locked = lock.acquire()
    try:
        current = now()
        queue = queue_status(base, current)
        state, running = run_state(base)
        metrics, activity = metrics_view(running, current)
        supervisor = supervisor_status(runner, supervisorctl)
        processes = live_writers(runner, base)
        history, history_error = _previous_history(status_path)

        metric_record = metrics.get("last_record") or {}
        current_stage = (metric_record.get("stage") or metrics.get("stage")
                         or queue.get("current_stage"))
        progress_sources = {
            "queue_status": _mtime(base / "QUEUE_STATUS.json"),
            "run_state": _mtime(base / "campaign-v2" / "RUN_STATE.json"),
            "job_output": activity,
        }
        known = {name: value for name, value in progress_sources.items() if value is not None}
        last_source = max(known, key=known.get) if known else None
        last_progress = known[last_source] if last_source else None
        threshold = p_stale_seconds if current_stage == "P" else stale_seconds
        progress = {"last_progress_utc": _iso(last_progress), "source": last_source,
                    "age_seconds": None if last_progress is None else round(current - last_progress, 1),
                    "stale_threshold_seconds": threshold, "current_stage": current_stage,
                    "stage_p_metrics_optional": current_stage == "P"}

        alerts: list[str] = []
        completed = queue.get("status") == "COMPLETED"
        sup_state = supervisor["state"]
        if not queue["readable"]:
            alerts.append(f"QUEUE_STATUS_{str(queue['error']).upper().split(':')[0]}")
        if not state["readable"]:
            alerts.append(f"RUN_STATE_{str(state['error']).upper().split(':')[0]}")
        if queue.get("status") == "FAILED":
            alerts.append("QUEUE_FAILED")
        if sup_state == "UNKNOWN":
            alerts.append("SUPERVISOR_STATUS_UNKNOWN")
        if not processes["listing_ok"]:
            alerts.append("PROCESS_LISTING_FAILED")
        if not locked:
            alerts.append("MONITOR_LOCK_UNAVAILABLE")
        if history_error:
            alerts.append("RESTART_HISTORY_UNREADABLE")
        alive = sup_state in ALIVE_STATES or bool(processes["matching"])
        if not completed and alive and progress["age_seconds"] is not None and progress["age_seconds"] > threshold:
            alerts.append("STALE_NO_PROGRESS_STAGE_P" if current_stage == "P" else "STALE_NO_PROGRESS")
        if sup_state in {"BACKOFF", "STARTING"}:
            alerts.append(f"SUPERVISOR_{sup_state}")
        if sup_state == "RUNNING" and processes["listing_ok"] and not processes["matching"]:
            alerts.append("SUPERVISOR_RUNNING_NO_MATCHING_PROCESS")
        if sup_state in RESTARTABLE_STATES and processes["matching"]:
            alerts.append("SUPERVISOR_DOWN_BUT_WRITER_ALIVE")

        blocked: list[str] = []
        if completed:
            blocked.append("queue COMPLETED")
        if sup_state not in RESTARTABLE_STATES:
            blocked.append(f"supervisor state {sup_state} is not EXITED/FATAL/STOPPED")
        if not queue["readable"]:
            blocked.append("QUEUE_STATUS.json unreadable")
        if not processes["listing_ok"]:
            blocked.append("process listing failed")
        if processes["matching"]:
            blocked.append("live train.py/queue process present")
        if not locked:
            blocked.append("monitor lock held by another watchdog")
        if history_error:
            blocked.append(history_error)
        budget = restart_budget(history, current, per_hour=max_restarts_per_hour,
                                total=max_restarts_total, backoff_seconds=backoff_seconds)
        if budget:
            blocked.append(budget)
        eligible_without_flag = not blocked
        if no_restart:
            blocked.append("--no-restart")
        restart: dict[str, Any] = {"eligible": not blocked, "would_restart": eligible_without_flag,
                                   "blocked_by": blocked, "attempted": False, "result": None,
                                   "history": list(history), "history_error": history_error}
        if eligible_without_flag and budget is None and sup_state in RESTARTABLE_STATES:
            if not no_restart:
                restart["attempted"] = True
                # Reserve the budget durably BEFORE any recovery/start command.
                # A timeout, crash or power loss must still count this attempt.
                restart["result"] = {"ok": False, "detail": "attempt_reserved"}
                restart["history"].append({"epoch": current, "utc": _iso(current),
                                           "from_state": sup_state, "ok": False})
            else:
                alerts.append("RESTART_SUPPRESSED")
        elif budget and sup_state in RESTARTABLE_STATES and not completed:
            alerts.append("RESTART_BUDGET_BLOCKED")
        elif sup_state in RESTARTABLE_STATES and not completed:
            alerts.append("SUPERVISOR_DOWN_NOT_RESTARTED")

        if completed:
            health = "COMPLETED"
        elif restart["attempted"] and restart["result"]["ok"]:
            health = "RECOVERING"
        elif alerts:
            health = "ALERT"
        else:
            health = "OK"
        record = {
            "schema": SCHEMA, "watchdog_version": VERSION, "checked_utc": _iso(current),
            "updated_utc": _iso(current),
            "base": str(base), "program": PROGRAM, "interval_seconds": interval,
            "mode": {"once": once, "no_restart": no_restart}, "lock_acquired": locked,
            "health": health, "status": health,
            "attention_required": health not in {"OK", "RECOVERING"},
            "final": completed and locked, "alerts": alerts,
            "supervisor": supervisor, "queue": queue, "run_state": state,
            "current_stage": current_stage, "metrics": metrics, "progress": progress,
            "processes": processes, "restart": restart,
        }
        if locked:
            if restart["attempted"]:
                record["health"] = "RECOVERY_PENDING"
                record["status"] = "RECOVERY_PENDING"
                record["attention_required"] = True
                atomic_write_json(status_path, record)
                append_jsonl(base / "monitor.jsonl", {
                    "schema": SCHEMA, "updated_utc": _iso(current),
                    "event": "restart_reserved", "attempt": restart["history"][-1],
                })
                restart["result"] = _attempt_restart(runner, supervisorctl, recovery_executable,
                                                     recovery_uid, base)
                restart["history"][-1]["ok"] = restart["result"]["ok"]
                alerts.append("RESTART_ATTEMPTED" if restart["result"]["ok"] else "RESTART_FAILED")
                record["health"] = "RECOVERING" if restart["result"]["ok"] else "ALERT"
                record["status"] = record["health"]
                record["attention_required"] = not restart["result"]["ok"]
            atomic_write_json(status_path, record)
            compact = {**record, "restart": {k: v for k, v in restart.items() if k != "history"}}
            append_jsonl(base / "monitor.jsonl", compact)
        return record
    finally:
        lock.release()


def _attempt_restart(runner: Runner, supervisorctl: Sequence[str], recovery_executable: str | None,
                     recovery_uid: int, base: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "recovery": None, "start_returncode": None, "detail": None}

    def still_safe() -> bool:
        queue = queue_status(base, time.time())
        supervisor = supervisor_status(runner, supervisorctl)
        processes = live_writers(runner, base)
        return (queue["readable"] and queue["status"] != "COMPLETED"
                and supervisor["state"] in RESTARTABLE_STATES
                and processes["listing_ok"] and not processes["matching"])

    if not still_safe():
        result["detail"] = "safety recheck changed; start skipped"
        return result
    if recovery_executable is not None:
        problem = validate_recovery_executable(recovery_executable, recovery_uid)
        if problem:
            result["recovery"] = {"ok": False, "detail": problem}
            result["detail"] = "recovery executable rejected; start skipped"
            return result
        code, output = _run(runner, [recovery_executable, "--base", str(base)], timeout=300)
        result["recovery"] = {"ok": code == 0, "returncode": code}
        if code != 0:
            result["detail"] = "recovery executable failed; start skipped"
            return result
        if not still_safe():
            result["detail"] = "safety recheck changed after recovery; start skipped"
            return result
    code, output = _run(runner, [*supervisorctl, "start", PROGRAM])
    result["start_returncode"] = code
    result["detail"] = "start_accepted" if code == 0 else "start_failed_or_unconfirmed"
    result["ok"] = code == 0
    return result


# ------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=DEFAULT_BASE, help="campaign base directory")
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--no-restart", action="store_true", help="observe and report only")
    parser.add_argument("--stale-seconds", type=int, default=DEFAULT_STALE)
    parser.add_argument("--p-stale-seconds", type=int, default=DEFAULT_P_STALE,
                        help="stale threshold while stage P (may not emit metrics) is running")
    parser.add_argument("--max-restarts-per-hour", type=int, default=3)
    parser.add_argument("--max-restarts-total", type=int, default=12)
    parser.add_argument("--backoff-seconds", type=int, default=300,
                        help="base back-off between attempts; doubles per recent attempt")
    parser.add_argument("--supervisorctl", default="supervisorctl",
                        help="supervisorctl executable (the program name is fixed to q3d-campaign)")
    parser.add_argument("--supervisor-config", default=None, help="optional supervisord config for -c")
    parser.add_argument("--recovery-executable", default=None,
                        help="optional absolute root-owned executable run with --base BASE before start; "
                             "never inferred")
    return parser


def main(argv: Sequence[str] | None = None, *, runner: Runner = subprocess.run,
         now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
         recovery_uid: int = 0) -> int:
    args = build_parser().parse_args(argv)
    base = Path(args.base)
    for name in ("interval_seconds", "stale_seconds", "p_stale_seconds", "backoff_seconds"):
        if getattr(args, name) <= 0:
            print(f"--{name.replace('_', '-')} must be positive", file=sys.stderr)
            return EXIT_INTERNAL
    if args.max_restarts_per_hour < 0 or args.max_restarts_total < 0:
        print("restart budgets must be non-negative", file=sys.stderr)
        return EXIT_INTERNAL
    if not base.is_absolute() or not base.is_dir():
        print(f"--base must be an existing absolute directory: {base}", file=sys.stderr)
        return EXIT_INTERNAL
    if args.recovery_executable is not None:
        problem = validate_recovery_executable(args.recovery_executable, recovery_uid)
        if problem:
            print(problem, file=sys.stderr)
            return EXIT_INTERNAL
    supervisorctl = [args.supervisorctl] + (["-c", args.supervisor_config] if args.supervisor_config else [])
    stopping = {"flag": False}

    def _stop(*_: Any) -> None:
        stopping["flag"] = True

    if not args.once:
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
    while True:
        try:
            record = check_once(
                base, runner=runner, now=now, no_restart=args.no_restart,
                interval=args.interval_seconds, stale_seconds=args.stale_seconds,
                p_stale_seconds=args.p_stale_seconds,
                max_restarts_per_hour=args.max_restarts_per_hour,
                max_restarts_total=args.max_restarts_total, backoff_seconds=args.backoff_seconds,
                supervisorctl=supervisorctl, recovery_executable=args.recovery_executable,
                recovery_uid=recovery_uid, once=args.once,
            )
        except Exception as exc:  # noqa: BLE001 - a watchdog must survive one bad cycle
            print(json.dumps({"health": "WATCHDOG_ERROR", "error": _excerpt(repr(exc), 300)}), flush=True)
            if args.once:
                return EXIT_INTERNAL
        else:
            print(json.dumps({"checked_utc": record["checked_utc"], "health": record["health"],
                              "alerts": record["alerts"], "stage": record["current_stage"],
                              "supervisor": record["supervisor"]["state"]}), flush=True)
            if record["final"]:
                return EXIT_OK
            if args.once:
                return EXIT_OK if record["health"] in {"OK", "RECOVERING"} else EXIT_ALERT
        deadline = now() + args.interval_seconds
        while not stopping["flag"] and now() < deadline:
            sleep(min(5.0, max(0.0, deadline - now())))
        if stopping["flag"]:
            return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
