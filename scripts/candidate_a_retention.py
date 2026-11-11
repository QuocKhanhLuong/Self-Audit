#!/usr/bin/env python3
"""Safely reclaim A-3D evaluation prediction arrays after scoring.

The A-3D evaluator materializes full-volume ``.npy`` probabilities under a
job's ``validation/epoch_*`` directory.  This utility is deliberately
receipt-driven: it only considers output directories declared by a
hash-pinned few-shot plan (or its campaign state), validates the frozen
prediction inventory and terminal metrics marker, and removes only those
listed prediction files.  It never searches the data tree and never reads a
label file.

``EPOCH_METRICS.json`` and ``TEACHER_EPOCH_METRICS.json`` are the evaluator's
terminal score markers.  The evaluator writes them after scoring and leaves
``PREDICTIONS_FROZEN.json`` and code packets alongside them.  A dry run is the
default; ``--execute`` is required before any prediction file is unlinked.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import sys
import time
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_a_fewshot import _verify_root_plan
from self_audit.model import digest

SCHEMA = "self_audit.candidate_a3d.retention.v1"
EPOCH_RE = re.compile(r"^epoch_[0-9]{4,}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _finite(value: Any, *, path: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, path=f"{path}[{index}]")


def _stable_json(path: Path) -> tuple[dict[str, Any], str, int, int]:
    """Read a terminal marker only when one stable byte snapshot exists."""

    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise ValueError(f"terminal marker changed while reading: {path}")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"terminal marker is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"terminal marker root must be an object: {path}")
    _finite(value, path=str(path))
    return value, hashlib.sha256(payload).hexdigest(), after.st_size, after.st_mtime_ns


def _safe_child(root: Path, relative: str, *, kind: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{kind} path must be a nonempty relative path")
    candidate = (root / relative).resolve()
    if candidate.parent != root.resolve():
        raise ValueError(f"{kind} path escapes its declared directory: {relative!r}")
    return candidate


def _job_records(plan_paths: Iterable[Path], state: dict[str, Any] | None) -> list[dict[str, Any]]:
    state_jobs: dict[str, dict[str, Any]] = {}
    if state is not None:
        if state.get("schema") != "self_audit.candidate_a3d.campaign_state.v1":
            raise ValueError("campaign state schema mismatch")
        for item in state.get("jobs", []):
            if not isinstance(item, dict) or not isinstance(item.get("key"), str):
                raise TypeError("campaign state contains an invalid job record")
            state_jobs[item["key"]] = item
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root_path in plan_paths:
        root_path = root_path.resolve()
        root = _verify_root_plan(root_path)
        for child_entry in root["child_plans"]:
            child_path = Path(child_entry["plan"]).resolve()
            if digest(child_path) != child_entry["plan_sha256"]:
                raise ValueError(f"child plan changed after planning: {child_path}")
            child = _json(child_path)
            if child.get("schema") != "self_audit.candidate_a3d.plan.v1":
                raise ValueError(f"child plan schema mismatch: {child_path}")
            for job in child.get("jobs", []):
                if not isinstance(job, dict) or not isinstance(job.get("out"), str):
                    raise TypeError(f"invalid job output in child plan: {child_path}")
                key = f"{child_path}::{job.get('id')}"
                state_job = state_jobs.get(key)
                # A running job may already have terminal, hash-verified
                # validation epochs.  Pending and failed jobs remain outside
                # the retention allow-list; an epoch-level check below still
                # leaves partial output untouched.
                if state is not None and (state_job is None or state_job.get("status") not in {"DONE", "RUNNING"}):
                    continue
                raw_planned_out = Path(job["out"])
                if raw_planned_out.is_symlink():
                    raise ValueError(f"planned job output must not be a symlink: {raw_planned_out}")
                planned_out = raw_planned_out.resolve()
                state_out = planned_out
                if state_job is not None and state_job.get("out") is not None:
                    if not isinstance(state_job["out"], str) or not state_job["out"]:
                        raise TypeError(f"campaign state job output is invalid: {key}")
                    raw_state_out = Path(state_job["out"])
                    state_out = raw_state_out.resolve()
                    same_planned = state_out == planned_out
                    retry_name = f"{planned_out.name}.retry-"
                    safe_retry = (raw_state_out.parent.resolve() == planned_out.parent.resolve()
                                  and raw_state_out.name.startswith(retry_name))
                    if not same_planned and not safe_retry:
                        raise ValueError(f"campaign state job output is outside its planned retry path: {key}")
                    if raw_state_out.is_symlink():
                        raise ValueError(f"campaign state job output must not be a symlink: {raw_state_out}")
                out = state_out
                if out in seen:
                    raise ValueError(f"duplicate planned output: {out}")
                seen.add(out)
                records.append({"key": key, "out": out, "dataset": root["dataset"],
                                "k": child_entry["k"], "arm": str(job.get("id")),
                                "plan": root_path, "plan_sha256": digest(root_path),
                                "child": child_path})
    return records


def _load_inventory(eval_dir: Path) -> tuple[list[tuple[Path, str, int, int]], dict[str, Any], str, int]:
    frozen_path = eval_dir / "PREDICTIONS_FROZEN.json"
    if not frozen_path.is_file() or frozen_path.is_symlink():
        raise ValueError(f"missing frozen prediction inventory: {frozen_path}")
    frozen, frozen_sha, _, frozen_mtime = _stable_json(frozen_path)
    entries: list[tuple[Path, str, int, int]] = []

    def branch(name: str, items: Any) -> None:
        if not isinstance(items, list) or not items:
            raise ValueError(f"empty or invalid frozen branch {name}: {frozen_path}")
        directory = eval_dir / ("predictions" if name == "student" else f"predictions_{name}")
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"missing frozen prediction directory: {directory}")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("file"), str):
                raise TypeError(f"invalid frozen prediction entry in {name}")
            sha = item.get("sha256")
            if not isinstance(sha, str) or len(sha) != 64:
                raise ValueError(f"invalid frozen prediction hash in {name}")
            path = _safe_child(directory, item["file"], kind="prediction")
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"frozen prediction is missing or symlinked: {path}")
            before = path.stat()
            actual = digest(path)
            stat = path.stat()
            if before.st_size != stat.st_size or before.st_mtime_ns != stat.st_mtime_ns:
                raise ValueError(f"frozen prediction changed while hashing: {path}")
            if actual != sha:
                raise ValueError(f"frozen prediction hash mismatch: {path}")
            entries.append((path, actual, stat.st_size, stat.st_mtime_ns))

    if isinstance(frozen.get("files"), list):
        branch("student", frozen["files"])
    elif isinstance(frozen.get("branches"), dict):
        for name, items in frozen["branches"].items():
            if name not in {"region", "boundary"}:
                raise ValueError(f"unknown teacher prediction branch {name}")
            branch(name, items)
    else:
        raise TypeError(f"frozen inventory has no student files or teacher branches: {frozen_path}")
    if len({path for path, *_ in entries}) != len(entries):
        raise ValueError(f"duplicate frozen prediction path: {frozen_path}")
    return entries, frozen, frozen_sha, frozen_mtime


def _validate_score_marker(eval_dir: Path, frozen: dict[str, Any]) -> tuple[Path, dict[str, Any], str]:
    teacher = isinstance(frozen.get("branches"), dict)
    marker = eval_dir / ("TEACHER_EPOCH_METRICS.json" if teacher else "EPOCH_METRICS.json")
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(f"score-finished marker is missing: {marker}")
    metrics, metric_sha, _, _marker_mtime = _stable_json(marker)
    required = {"epoch", "scope", "profile", "labels_used_for"}
    if teacher:
        required.add("branches")
    else:
        required.update({"per_class", "code"})
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"score marker is incomplete ({', '.join(missing)}): {marker}")
    if type(metrics["epoch"]) is not int or metrics["epoch"] < 0:
        raise ValueError(f"score marker epoch is invalid: {marker}")
    if not isinstance(metrics["labels_used_for"], str) or "development scoring" not in metrics["labels_used_for"]:
        raise ValueError(f"score marker does not identify read-only development scoring: {marker}")
    return marker, metrics, metric_sha


def _inspect_eval(eval_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    if eval_dir.is_symlink() or not eval_dir.is_dir():
        raise ValueError(f"evaluation directory is not a real directory: {eval_dir}")
    entries, frozen, frozen_sha, frozen_mtime = _load_inventory(eval_dir)
    # Check that every listed array was complete before opening the terminal
    # marker.  The marker is read last and therefore acts as the commit point.
    marker, metrics, metric_sha = _validate_score_marker(eval_dir, frozen)
    marker_mtime = marker.stat().st_mtime_ns
    try:
        expected_epoch = int(eval_dir.name.removeprefix("epoch_"))
    except ValueError as exc:
        raise ValueError(f"evaluation directory has an invalid epoch: {eval_dir}") from exc
    if metrics["epoch"] != expected_epoch:
        raise ValueError(f"score marker epoch does not match directory: {eval_dir}")
    if "epoch" in frozen and frozen["epoch"] != expected_epoch:
        raise ValueError(f"frozen inventory epoch does not match directory: {eval_dir}")
    if frozen_mtime > marker_mtime:
        raise ValueError(f"frozen inventory is newer than score marker: {eval_dir}")
    stale = [str(path) for path, _, _, mtime_ns in entries if mtime_ns > marker_mtime]
    if stale:
        raise ValueError(f"prediction is newer than score marker (evaluation may still be running): {stale[0]}")
    return {"job_key": record["key"], "job_out": str(record["out"]),
            "evaluation_dir": str(eval_dir), "score_marker": marker.name,
            "score_marker_sha256": metric_sha, "frozen_sha256": frozen_sha,
            "epoch": metrics["epoch"],
            "files": [{"path": str(path), "sha256": sha, "bytes": size, "mtime_ns": mtime_ns}
                      for path, sha, size, mtime_ns in entries]}


def discover(paths: Iterable[str | Path], *, state_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return validated removable evaluation artifacts without modifying them."""

    explicit = [Path(path).resolve() for path in paths]
    state = None
    if state_path is not None:
        state = _json(Path(state_path).resolve())
        if not explicit:
            explicit = [Path(item["path"]).resolve() for item in state.get("plans", [])]
        elif {str(path) for path in explicit} != {str(Path(item["path"]).resolve()) for item in state.get("plans", [])}:
            raise ValueError("explicit plans do not match the campaign state's hash-pinned plans")
        for item in state.get("plans", []):
            path = Path(item["path"]).resolve()
            if digest(path) != item["sha256"]:
                raise ValueError(f"campaign plan changed after state creation: {path}")
    if not explicit:
        raise ValueError("at least one --plan or --state is required")
    records = _job_records(explicit, state)
    findings: list[dict[str, Any]] = []
    for record in records:
        validation = record["out"] / "validation"
        if not validation.is_dir() or validation.is_symlink():
            continue
        for item in sorted(validation.iterdir(), key=lambda path: path.name):
            if item.is_symlink() or not EPOCH_RE.fullmatch(item.name):
                continue
            try:
                findings.append(_inspect_eval(item, record))
            except (OSError, TypeError, ValueError):
                # A partial/active epoch is not a deletion candidate.  The
                # caller receives a full finding list plus skipped diagnostics.
                continue
    return findings


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def _retain_once(plans: list[str | Path], *, state_path: Path | None,
                 execute: bool, receipt_path: Path) -> dict[str, Any]:
    findings = discover(plans, state_path=state_path)
    if not execute:
        return {"schema": SCHEMA, "status": "DRY_RUN", "evaluations": findings,
                "candidate_files": sum(len(item["files"]) for item in findings),
                "receipt": str(receipt_path)}
    existing: dict[str, Any] | None = None
    if receipt_path.is_file():
        existing = _json(receipt_path)
    old = {item["path"]: item for item in (existing or {}).get("removed_files", [])
           if isinstance(item, dict) and isinstance(item.get("path"), str)}
    # Recheck every candidate immediately before the first unlink.  This is a
    # fail-closed preflight for a writer that races the evaluator; no deletion
    # starts while an inventory hash is already stale.
    for finding in findings:
        for item in finding["files"]:
            path = Path(item["path"])
            if not path.exists():
                if str(path) not in old:
                    raise ValueError(f"prediction is absent without a prior receipt: {path}")
                continue
            stat = path.stat() if path.exists() else None
            if (path.is_symlink() or not path.is_file() or stat is None
                    or stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]
                    or digest(path) != item["sha256"]):
                raise ValueError(f"prediction changed before removal: {path}")
    removed: list[dict[str, Any]] = list(old.values())
    for finding in findings:
        for item in finding["files"]:
            path = Path(item["path"])
            if path.exists():
                stat = path.stat()
                if (path.is_symlink() or not path.is_file() or stat.st_size != item["bytes"]
                        or stat.st_mtime_ns != item["mtime_ns"]
                        or digest(path) != item["sha256"]):
                    raise ValueError(f"prediction changed immediately before removal: {path}")
                path.unlink()
                removed.append({**item, "evaluation_dir": finding["evaluation_dir"],
                                "job_key": finding["job_key"], "removed_at": _now()})
            elif str(path) not in old:
                raise ValueError(f"prediction is absent without a prior receipt: {path}")
    unique: dict[str, dict[str, Any]] = {item["path"]: item for item in removed}
    status = "APPLIED" if findings else ("ALREADY_APPLIED" if old else "NO_CANDIDATES")
    result = {"schema": SCHEMA, "status": status,
              "created_at": (existing or {}).get("created_at", _now()),
              "updated_at": _now(), "plans": [str(Path(path).resolve()) for path in plans],
              "plan_hashes": [{"path": str(Path(path).resolve()), "sha256": digest(Path(path))}
                              for path in plans],
              "state": {"path": str(state_path), "sha256": digest(state_path)} if state_path else None,
              "evaluations": findings, "removed_files": list(unique.values()),
              "removed_count": len(unique),
              "removed_bytes": int(sum(int(item.get("bytes", 0)) for item in unique.values())),
              "receipt": str(receipt_path),
              "policy": "remove only frozen prediction arrays after finite terminal score marker; keep metrics, frozen inventory, code packets and checkpoints"}
    _atomic_write(receipt_path, result)
    return result


@contextmanager
def _exclusive_lock(path: Path):
    """Serialize retention watchers that share one receipt path."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def retain(plans: Iterable[str | Path], *, state: str | Path | None = None,
           execute: bool = False, receipt: str | Path | None = None,
           watch_seconds: float | None = None) -> dict[str, Any]:
    plans = list(plans)
    state_path = Path(state).resolve() if state is not None else None
    receipt_path = Path(receipt).resolve() if receipt is not None else (
        state_path.parent / "RETENTION_RECEIPT.json" if state_path else Path("RETENTION_RECEIPT.json").resolve())
    if watch_seconds is not None:
        if watch_seconds <= 0 or not math.isfinite(watch_seconds):
            raise ValueError("watch interval must be a finite positive number of seconds")
        if state_path is None:
            raise ValueError("--watch requires --state so each poll can validate campaign progress")
    lock_path = receipt_path.with_name(receipt_path.name + ".lock")
    with _exclusive_lock(lock_path):
        if watch_seconds is None:
            return _retain_once(plans, state_path=state_path, execute=execute,
                                receipt_path=receipt_path)
        stop = False
        previous_handlers: dict[int, Any] = {}

        def request_stop(_signum: int, _frame: Any) -> None:
            nonlocal stop
            stop = True

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        result: dict[str, Any] = {"schema": SCHEMA, "status": "WATCHING"}
        try:
            while not stop:
                result = _retain_once(plans, state_path=state_path, execute=execute,
                                      receipt_path=receipt_path)
                result["watch_interval_seconds"] = watch_seconds
                if stop:
                    break
                deadline = time.monotonic() + watch_seconds
                while not stop:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.25))
            result = dict(result)
            result["status"] = "WATCH_STOPPED"
            result["stopped_at"] = _now()
            return result
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="append", default=[], help="hash-pinned FEWSHOT_PLAN.json; repeat per dataset")
    parser.add_argument("--state", help="campaign state JSON; its plan hashes are checked")
    parser.add_argument("--receipt", help="central retention receipt path")
    parser.add_argument("--execute", action="store_true", help="unlink validated prediction arrays")
    parser.add_argument("--watch", type=float, metavar="SECONDS",
                        help="poll a hash-pinned campaign state until SIGINT/SIGTERM")
    args = parser.parse_args(argv)
    result = retain(args.plan, state=args.state, execute=args.execute, receipt=args.receipt,
                    watch_seconds=args.watch)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
