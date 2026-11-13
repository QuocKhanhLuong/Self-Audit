#!/usr/bin/env python3
"""Wait for the committed A campaign before running a locked Track B transfer.

Plan-only is the default. Execution requires an explicit predecessor receipt,
new output directory, and --execute. This queue never retries failures.
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
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
QUEUE_SCHEMA = "self_audit.candidate_fuse.transfer_queue.v1"
REQUIRED_A_STAGES = ["T", "J", "VAL3", "EXPORT", "TEST_ONCE"]


def real_data_preflight(config_path: Path) -> dict[str, Any]:
    """Validate pinned real inputs and source containers without decoding voxels."""
    from self_audit.candidate_fuse.config import load_config
    from self_audit.candidate_fuse.data import open_fuse_data

    cfg = load_config(config_path, real_run=True)
    if cfg.get("runtime", {}).get("synthetic") is not False:
        raise ValueError("transfer queue requires a real, non-synthetic FUSE config")
    receipts = {}
    for role, key in (("train", "train"), ("development", "val"), ("test", "test")):
        reader = open_fuse_data(cfg["data"][f"{key}_manifest"],
                                cfg["data"][f"{key}_manifest_sha256"],
                                role=role, synthetic=False)
        verify = getattr(reader, "verify_sources", None)
        if not callable(verify):
            raise RuntimeError(f"{role} data reader does not support metadata-only source verification")
        receipts[role] = verify()
    return receipts


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def verify_predecessor(path: Path) -> dict[str, Any]:
    """Verify A's campaign and committed one-shot TEST without touching A."""
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("status") != "COMPLETED" or run.get("completed") != REQUIRED_A_STAGES:
        raise ValueError("predecessor A campaign must be COMPLETED through TEST_ONCE")
    ledger = run.get("test_ledger", {})
    if ledger.get("status") != "COMMITTED":
        raise ValueError("predecessor A TEST ledger is not committed")
    if not isinstance(run.get("config_sha256"), str) or len(run["config_sha256"]) != 64:
        raise ValueError("predecessor A config identity is missing")
    if not isinstance(run.get("export_bundle"), dict) and not isinstance(run.get("artifacts"), dict):
        raise ValueError("predecessor A has no committed artifact identity")
    transaction_path = ledger.get("transaction_path")
    if (not isinstance(transaction_path, str) or not Path(transaction_path).is_file()
            or not isinstance(ledger.get("completed_utc"), str)):
        raise ValueError("predecessor A committed TEST result receipt is missing")
    transaction = json.loads(Path(transaction_path).read_text(encoding="utf-8"))
    if transaction.get("status") != "COMMITTED" or transaction.get("role") != "test":
        raise ValueError("predecessor A TEST transaction receipt is not committed")
    return run


def _failed_predecessor(path: Path) -> str | None:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if state.get("status") == "FAILED":
        return "predecessor A campaign is FAILED; no automatic retry is allowed"
    if state.get("test_ledger", {}).get("status") == "FAILED_LOCKED":
        return "predecessor A TEST is FAILED_LOCKED; no automatic TEST retry is allowed"
    return None


def gpu_idle(*, smi: str = "nvidia-smi") -> tuple[bool, str]:
    """Read-only process query used only at execution time."""
    completed = subprocess.run([smi, "--query-compute-apps=pid", "--format=csv,noheader"],
                               check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"cannot verify GPU idle state: {completed.stderr.strip()}")
    pids = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return not pids, ",".join(pids)


def disk_headroom(path: Path, required_bytes: int = 8 * 1024**3) -> dict[str, int]:
    free = shutil.disk_usage(path).free
    return {"free_bytes": free, "required_bytes": required_bytes}


def run_queue(config: Path, out: Path, predecessor: Path, *, execute: bool,
              device: str = "cuda", poll_seconds: int = 30,
              idle_probe=gpu_idle, wait_for_predecessor: bool = True) -> dict[str, Any]:
    config, out, predecessor = config.resolve(), out.resolve(), predecessor.resolve()
    if not config.is_file():
        raise ValueError(f"B config does not exist: {config}")
    if not predecessor.is_file():
        raise ValueError(f"predecessor RUN_STATE does not exist: {predecessor}")
    cfg = json.loads(config.read_text(encoding="utf-8"))
    if cfg.get("schema") != "self_audit.candidate_fuse.v3" or cfg.get("experiment", {}).get("track") != "B":
        raise ValueError("transfer queue requires a locked FUSE v3 Track B config")
    source = cfg.get("experiment", {}).get("source_teacher", {})
    if (not isinstance(source, dict) or not isinstance(source.get("path"), str)
            or len(source.get("sha256", "")) != 64
            or len(source.get("metadata_sha256", "")) != 64
            or not isinstance(source.get("identity"), dict)):
        raise ValueError("Track B queue requires a fully pinned selected source teacher")
    if out == predecessor.parent or predecessor.parent in out.parents:
        raise ValueError("B output must be separate from and outside the predecessor A run")
    if out.exists() and any(out.iterdir()):
        raise ValueError("B output must be a new empty directory; use a separate run directory")
    plan = {"schema": QUEUE_SCHEMA, "status": "PLAN_ONLY", "config": str(config),
            "config_sha256": _sha(config), "predecessor": str(predecessor),
            "device": device, "stages": ["PROFILE_TJ", "CAMPAIGN_TJ_VAL_EXPORT_TEST_ONCE"],
            "automatic_retries": 0, "output": str(out)}
    if not execute:
        return plan
    out.mkdir(parents=True, exist_ok=False)
    lock_path = out / "TRANSFER_QUEUE.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.write(fd, f"pid={os.getpid()}\n".encode())
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = out / "TRANSFER_QUEUE_STATE.json"
    state = {**plan, "status": "WAITING", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "predecessor_sha256": None, "profile": "NOT_RUN", "campaign": "NOT_RUN"}
    _atomic_json(state_path, state)
    try:
        while True:
            try:
                a_state = verify_predecessor(predecessor)
                break
            except (ValueError, json.JSONDecodeError):
                terminal = _failed_predecessor(predecessor)
                if terminal:
                    state.update(status="BLOCKED", blocker=terminal)
                    _atomic_json(state_path, state)
                    return state
                if not wait_for_predecessor:
                    raise
                time.sleep(max(1, poll_seconds))
        source_path = Path(source["path"]).expanduser().resolve()
        if not source_path.is_file() or _sha(source_path) != source.get("sha256"):
            state.update(status="BLOCKED", blocker="pinned selected source teacher SHA mismatch or source missing")
            _atomic_json(state_path, state)
            return state
        sidecar = Path(str(source_path) + ".meta.json")
        if not sidecar.is_file() or _sha(sidecar) != source.get("metadata_sha256"):
            state.update(status="BLOCKED", blocker="pinned selected source teacher metadata SHA mismatch")
            _atomic_json(state_path, state)
            return state
        source_meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if source_meta.get("status") != "SELECTED" or source_meta.get("identity") != source.get("identity"):
            state.update(status="BLOCKED", blocker="source teacher is not SELECTED or identity mismatches")
            _atomic_json(state_path, state)
            return state
        # The real-run validation and source-container preflight happen only
        # after A is committed, and before any profile/campaign CUDA process.
        state["active"] = "PREFLIGHT"
        _atomic_json(state_path, state)
        state["data_preflight"] = real_data_preflight(config)
        _atomic_json(state_path, state)
        idle, pids = idle_probe()
        if not idle:
            state.update(status="BLOCKED", blocker=f"GPU not idle; compute PIDs={pids}")
            _atomic_json(state_path, state)
            return state
        headroom = disk_headroom(out.parent)
        state["disk_headroom"] = headroom
        if headroom["free_bytes"] < headroom["required_bytes"]:
            state.update(status="BLOCKED", blocker="insufficient disk headroom: need at least 8 GiB for atomic checkpoint replacement")
            _atomic_json(state_path, state)
            return state
        state.update(status="READY", predecessor_config_sha256=a_state["config_sha256"],
                     predecessor_test_transaction=a_state["test_ledger"],
                     predecessor_sha256=_sha(predecessor), gpu_processes="none")
        _atomic_json(state_path, state)
        # Profile in a separate directory before creating production campaign state.
        profile_out = out / "profile"
        cmd = [sys.executable, "scripts/fuse_profile.py", "--config", str(config),
               "--out", str(profile_out), "--device", device, "--execute"]
        state["active"] = "PROFILE_TJ"
        state["log"] = str(out / "transfer_queue.log")
        _atomic_json(state_path, state)
        with Path(state["log"]).open("a", encoding="utf-8", buffering=1) as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line); sys.stdout.flush(); log.write(line)
            rc = proc.wait()
            if rc:
                state.update(status="FAILED", active=None, error={"stage": "PROFILE_TJ", "exit_code": rc})
                _atomic_json(state_path, state)
                return state
            profile_path = profile_out / "PROFILE.json"
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            if profile.get("status") != "PROFILE_ONLY" or profile.get("track") != "B":
                state.update(status="FAILED", active=None, error={"stage": "PROFILE_TJ", "reason": "invalid profile receipt"})
                _atomic_json(state_path, state)
                return state
            state["profile"] = {"status": "PROFILE_ONLY", "path": str(profile_path.resolve()), "sha256": _sha(profile_path)}
            # Profiling consumes CUDA and time; recheck both execution gates
            # immediately before starting the real campaign. Never retry.
            idle, pids = idle_probe()
            if not idle:
                state.update(status="BLOCKED", active=None,
                             blocker=f"GPU not idle after profile; compute PIDs={pids}")
                _atomic_json(state_path, state)
                return state
            headroom = disk_headroom(out.parent)
            state["disk_headroom_after_profile"] = headroom
            if headroom["free_bytes"] < headroom["required_bytes"]:
                state.update(status="BLOCKED", active=None,
                             blocker="insufficient disk headroom after profile: need at least 8 GiB for atomic checkpoint replacement")
                _atomic_json(state_path, state)
                return state
            state["active"] = "CAMPAIGN"
            _atomic_json(state_path, state)
            campaign_cmd = [sys.executable, "scripts/fuse_campaign.py", "--config", str(config),
                            "--out", str(out / "campaign"), "--execute", "--device", device]
            proc = subprocess.Popen(campaign_cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line); sys.stdout.flush(); log.write(line)
            rc = proc.wait()
        state.update(status="COMPLETED" if rc == 0 else "FAILED", active=None,
                     campaign="COMPLETED" if rc == 0 else "FAILED",
                     error=None if rc == 0 else {"stage": "CAMPAIGN", "exit_code": rc})
        _atomic_json(state_path, state)
        return state
    except Exception as exc:
        # A queue execution is single-pass: persist the failure before exiting
        # so a later invocation cannot mistake an interrupted stage for work
        # that should be retried automatically.
        failed_stage = state.get("active")
        state.update(status="FAILED", active=None,
                     error={"stage": failed_stage, "type": type(exc).__name__, "message": str(exc)})
        _atomic_json(state_path, state)
        return state
    finally:
        os.close(fd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--predecessor", required=True, type=Path, help="A RUN_STATE.json")
    parser.add_argument("--execute", action="store_true", help="wait, profile, and launch the B campaign")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    result = run_queue(args.config, args.out, args.predecessor, execute=args.execute,
                       device=args.device, poll_seconds=args.poll_seconds)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] in {"PLAN_ONLY", "COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
