#!/usr/bin/env python3
"""Supervise the locked FUSE T -> J -> VAL/deploy/export -> one TEST flow.

This is a foreground supervisor. It never starts work on import; execution
requires ``--execute`` and every stage delegates to the canonical train.py or
evaluate.py entrypoint.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATE_SCHEMA = "self_audit.candidate_fuse.campaign_state.v1"
STAGES = ("T", "J", "VAL3", "EXPORT", "TEST_ONCE")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"schema": STATE_SCHEMA, "status": "NOT_STARTED", "completed": [], "active": None,
                "test_ledger": {"status": "NOT_RUN"}, "attempts": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != STATE_SCHEMA:
        raise ValueError("invalid FUSE campaign state schema")
    completed = value.get("completed")
    if not isinstance(completed, list) or any(s not in STAGES for s in completed) or len(set(completed)) != len(completed):
        raise ValueError("invalid or duplicate completed stage ledger")
    if completed != list(STAGES[:len(completed)]):
        raise ValueError("campaign ledger is out of order")
    if "TEST_ONCE" in completed and value.get("test_ledger", {}).get("status") != "COMMITTED":
        raise ValueError("TEST stage lacks a committed one-shot ledger")
    if value.get("test_ledger", {}).get("status") == "COMMITTED" and "TEST_ONCE" not in completed:
        raise ValueError("one-shot TEST ledger is inconsistent with campaign stages")
    return value


def _record(path: Path, state: dict, **updates) -> None:
    state.update(updates)
    state["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _atomic_json(path, state)


def has_committed_t_checkpoint(out: Path) -> bool:
    """Accept T resume only when engine state, SHA receipt and checkpoint agree."""
    checkpoint = out / "resume.pt"
    receipt_path = out / "resume.sha256.json"
    state_path = out / "STATE.json"
    if not all(path.is_file() for path in (checkpoint, receipt_path, state_path)):
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        actual_sha = _sha(checkpoint)
        return (receipt.get("path") == checkpoint.name
                and receipt.get("sha256") == actual_sha
                and state.get("stage") == "T"
                and Path(state.get("checkpoint_path", "")).resolve() == checkpoint.resolve()
                and state.get("checkpoint_sha256") == actual_sha)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def archive_uncommitted_t_attempt(out: Path, state: dict) -> Path | None:
    """Preserve every prior pre-checkpoint artifact before a fresh T attempt."""
    candidates = [out / name for name in (
        "metrics.jsonl", "STATE.json", "RESULT.json", "resume.pt", "resume.sha256.json",
        "teacher.pt", "teacher.pt.meta.json", "joint_best.pt", "joint_best.sha256.json",
        "exports",
    )]
    t_log = out / "campaign_logs" / "T.log"
    if t_log.exists():
        candidates.append(t_log)
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive = out / "recovery_archive" / f"T-before-checkpoint-{stamp}-{os.getpid()}"
    archive.mkdir(parents=True, exist_ok=False)
    _atomic_json(archive / "CAMPAIGN_STATE.json", state)
    moved = []
    for source in existing:
        if source.is_dir():
            destination = archive / ("exports" if source.name == "exports" else source.name)
        else:
            destination = archive / source.relative_to(out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved.append({"path": str(source.relative_to(out)), "archive_path": str(destination.relative_to(out))})
    _atomic_json(archive / "ARCHIVE.json", {
        "schema": "self_audit.candidate_fuse.t_recovery_archive.v1",
        "reason": "explicit resume requested but no committed T checkpoint was available",
        "moved": moved,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    return archive


def _run(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("\n$ " + " ".join(command) + "\n")
        proc = subprocess.Popen(list(command), cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return proc.wait()


def commands(config: Path, out: Path, max_updates: int | None = None,
             device: str = "cuda") -> dict[str, list[str]]:
    py = sys.executable
    budget = ["--fuse-max-updates", str(max_updates)] if max_updates is not None else []
    device_args = ["--device", device]
    return {
        "T": [py, "train.py", "--config", str(config),
              "--out", str(out), "--fuse-stage", "T", *device_args, *budget],
        "J": [py, "train.py", "--config", str(config),
              "--out", str(out), "--fuse-stage", "J", "--fuse-resume", "--fuse-teacher-artifact",
              str(out / "teacher.pt"), *device_args, *budget],
        # J performs locked VAL3 selection and writes the selected deployment export.
        "VAL3": [],
        "EXPORT": [],
        "TEST_ONCE": [],
    }


def export_bundle(out: Path) -> dict[str, dict[str, str]]:
    """Load and verify all three engine export receipts for the TEST transaction."""
    exports = out / "exports"
    bundle = {}
    for mode in ("teacher", "student", "fusion"):
        artifact = exports / f"{mode}.pt"
        sidecar = artifact.with_suffix(artifact.suffix + ".manifest.json")
        if not artifact.is_file() or not sidecar.is_file():
            raise ValueError(f"FUSE {mode} export or sidecar is missing")
        receipt = json.loads(sidecar.read_text(encoding="utf-8"))
        artifact_sha = _sha(artifact)
        if (receipt.get("schema") != "self_audit.candidate_fuse.export.v1"
                or receipt.get("artifact") != artifact.name
                or receipt.get("artifact_sha256") != artifact_sha
                or receipt.get("artifact_bytes") != artifact.stat().st_size
                or receipt.get("mode") != mode):
            raise ValueError(f"FUSE {mode} sidecar does not identify current artifact bytes")
        bundle[mode] = {"path": str(artifact.resolve()), "sha256": artifact_sha}
    return bundle


def evaluate_test_once(config_path: Path, out: Path, *, device: str = "cuda") -> dict:
    """Call the shared three-mode transaction API exactly once."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = config.get("data", {})
    manifest, manifest_sha = data.get("test_manifest"), data.get("test_manifest_sha256")
    if not isinstance(manifest, str) or not isinstance(manifest_sha, str):
        raise ValueError("locked TEST requires a pinned test manifest and SHA in the FUSE config")
    from self_audit.candidate_fuse.evaluation import evaluate_exports
    return evaluate_exports(export_bundle(out), manifest, out / "TEST_ONCE", device, "test", manifest_sha)


def run_campaign(config: Path, out: Path, *, execute: bool, max_updates: int | None = None,
                resume: bool = False, device: str = "cuda") -> dict:
    config, out = config.resolve(), out.resolve()
    if not config.is_file():
        raise ValueError(f"FUSE config does not exist: {config}")
    if max_updates is not None and max_updates < 1:
        raise ValueError("--max-updates must be positive")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("--device must be a nonempty torch device string")
    device = device.strip()
    config_doc = json.loads(config.read_text(encoding="utf-8"))
    if max_updates is not None and config_doc.get("runtime", {}).get("synthetic") is not True:
        raise ValueError("bounded real FUSE profiles must use scripts/fuse_profile.py")
    state_path = out / "RUN_STATE.json"
    state = load_state(state_path)
    config_sha = _sha(config)
    if state_path.exists() and state.get("config_sha256") != config_sha:
        raise ValueError("campaign config SHA differs from existing RUN_STATE; use a new run directory")
    if state_path.exists() and resume and state.get("device", device) != device:
        raise ValueError("campaign device differs from existing RUN_STATE; use a new run directory")
    state.setdefault("attempts", {})
    state.setdefault("test_ledger", {"status": "NOT_RUN"})
    state["config"] = str(config)
    state["config_sha256"] = config_sha
    state["stage_logs"] = state.get("stage_logs", {})
    state["device"] = device
    if not execute:
        return {"status": "PLAN_ONLY", "commands": commands(config, out, max_updates, device), "state": state}
    if state_path.exists() and not resume:
        raise ValueError("existing campaign requires explicit --resume; refusing to overwrite progress")
    if resume and state.get("status") not in {"FAILED", "PAUSED", "RUNNING"}:
        raise ValueError("--resume requires an interrupted or failed campaign")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "CAMPAIGN.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise ValueError("campaign writer lock exists; refuse a duplicate writer") from exc
    os.ftruncate(fd, 0)
    os.write(fd, f"pid={os.getpid()}\n".encode())
    plan = commands(config, out, max_updates, device)
    resume_t_checkpoint = False
    if resume and "T" not in state["completed"]:
        resume_t_checkpoint = has_committed_t_checkpoint(out)
        if not resume_t_checkpoint:
            archive = archive_uncommitted_t_attempt(out, state)
            if archive is not None:
                state.setdefault("recovery_archives", []).append(str(archive))
            if "error" in state:
                state["previous_error"] = state.pop("error")
    state["status"] = "RUNNING"
    _record(state_path, state)
    for stage in STAGES[len(state["completed"]):]:
        if stage == "TEST_ONCE" and state["test_ledger"].get("status") != "NOT_RUN":
            raise ValueError("TEST transaction was already attempted; refusing rerun")
        state["active"] = stage
        state["attempts"][stage] = int(state["attempts"].get(stage, 0)) + 1
        log = out / "campaign_logs" / f"{stage}.log"
        state["stage_logs"][stage] = str(log)
        if stage == "TEST_ONCE":
            state["test_ledger"] = {"status": "RUNNING", "attempt": 1,
                                     "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        _record(state_path, state)
        if stage == "TEST_ONCE":
            started = dt.datetime.now(dt.timezone.utc).isoformat()
            try:
                transaction = evaluate_test_once(config, out, device=device)
                if transaction.get("status") != "COMMITTED" or transaction.get("role") != "test":
                    raise ValueError("three-mode TEST transaction did not commit")
            except Exception as exc:
                state["test_ledger"].update(status="FAILED_LOCKED", started_utc=started,
                                             error=f"{type(exc).__name__}: {exc}")
                state["status"] = "FAILED"
                state["error"] = {"stage": stage, "reason": str(exc)}
                _record(state_path, state)
                return state
            state["test_ledger"].update(status="COMMITTED", started_utc=started,
                                         completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                                         transaction_path=str(out / "TEST_ONCE" / "TEST_TRANSACTION.json"))
            state["completed"].append(stage)
            state["active"] = None
            _record(state_path, state)
            continue
        elif stage in {"VAL3", "EXPORT"}:
            # These are committed as one engine transaction by J; verify bytes
            # before recording each milestone.
            required = [out / "joint_best.pt"]
            if stage == "VAL3" and not (out / "joint_best.pt").is_file():
                raise ValueError("J completed without its selected VAL3 checkpoint")
            if stage == "EXPORT":
                bundle = export_bundle(out)
                state["export_bundle"] = bundle
                _atomic_json(out / "TEST_EXPORT_BUNDLE.json", bundle)
                required.extend(Path(bundle[mode]["path"]) for mode in ("teacher", "student", "fusion"))
            state["artifacts"] = {str(p.name): _sha(p) for p in required if p.is_file()}
            state["completed"].append(stage)
            state["active"] = None
            _record(state_path, state)
            continue
        else:
            command = list(plan[stage])
            if stage == "T" and resume_t_checkpoint:
                command.append("--fuse-resume")
        code = _run(command, log)
        if code:
            if stage == "TEST_ONCE":
                state["test_ledger"].update(status="FAILED_LOCKED", exit_code=code)
            state["status"] = "FAILED"
            state["error"] = {"stage": stage, "exit_code": code}
            _record(state_path, state)
            return state
        state["completed"].append(stage)
        state["active"] = None
        _record(state_path, state)
    state["status"] = "COMPLETED"
    state["error"] = None
    _record(state_path, state)
    return state


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--execute", action="store_true", help="launch the foreground campaign")
    p.add_argument("--resume", action="store_true", help="resume an interrupted campaign after its cause is fixed")
    p.add_argument("--max-updates", type=int, help="predeclared bound for a profile or smoke")
    p.add_argument("--device", default="cuda", help="torch device passed to T, J and TEST (default: cuda)")
    a = p.parse_args(argv)
    result = run_campaign(a.config, a.out, execute=a.execute, max_updates=a.max_updates,
                          resume=a.resume, device=a.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"PLAN_ONLY", "COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
