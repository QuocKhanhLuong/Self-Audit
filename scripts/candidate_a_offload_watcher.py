#!/usr/bin/env python3
"""Poll campaign state and offload completed joint payloads safely.

Dry-run polling is the default. ``--execute`` additionally requires an
explicit enable file containing ``ENABLE_CANDIDATE_A_OFFLOAD``. The remote
transaction re-reads the remote state immediately before unlink, so a stale
local state cannot authorize a changed or running job.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_a_offload import offload_remote_job

ENABLE_TOKEN = "ENABLE_CANDIDATE_A_OFFLOAD"


def _read_state(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("jobs"), list):
        raise TypeError("campaign state must contain a jobs list")
    return value


def _enabled(path: Path) -> bool:
    try:
        return path.is_file() and path.read_text().strip() == ENABLE_TOKEN
    except OSError:
        return False


def _eligible_jobs(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = [job for job in state.get("jobs", [])
            if isinstance(job, Mapping) and job.get("status") == "DONE"
            and job.get("stage") == "joint"]
    return sorted((dict(job) for job in jobs), key=lambda job: str(job.get("key", "")))


def poll_once(*, state_path: Path, remote_state_path: str, remote_host: str,
              remote_campaign_root: str, backup_root: Path, enable_file: Path,
              execute: bool, python_bin: str) -> dict[str, Any]:
    state = _read_state(state_path)
    enabled = _enabled(enable_file)
    if execute and not enabled:
        raise PermissionError(f"execute requested but enable file is absent or invalid: {enable_file}")
    results: list[dict[str, Any]] = []
    for job in _eligible_jobs(state):
        key = str(job.get("key", ""))
        try:
            result = offload_remote_job(
                state=state, state_path=state_path, job_key=key,
                remote_host=remote_host, remote_campaign_root=remote_campaign_root,
                remote_state_path=remote_state_path, backup_root=backup_root,
                execute=execute and enabled, python_bin=python_bin,
            )
            results.append({"job_key": key, **result})
        except (OSError, RuntimeError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            results.append({"job_key": key, "status": "FAILED", "error": repr(exc)})
    return {
        "schema": "self_audit.candidate_a3d.offload_poll.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state": str(state_path), "execute": execute, "enabled": enabled,
        "jobs": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, help="refreshed local campaign state JSON")
    parser.add_argument("--remote-state", required=True, help="absolute remote campaign state JSON")
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-campaign-root", required=True)
    parser.add_argument("--backup-root", required=True)
    parser.add_argument("--enable-file", required=True,
                        help=f"must contain {ENABLE_TOKEN!r} before execute")
    parser.add_argument("--python-bin", default="/venv/main/bin/python")
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--execute", action="store_true", help="allow deletion only with enable file")
    parser.add_argument("--once", action="store_true", help="poll once and exit")
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.execute and not _enabled(Path(args.enable_file)):
        parser.error("--execute requires an enable file containing the exact enable token")
    kwargs = {
        "state_path": Path(args.state).expanduser().resolve(),
        "remote_state_path": args.remote_state, "remote_host": args.remote_host,
        "remote_campaign_root": args.remote_campaign_root,
        "backup_root": Path(args.backup_root).expanduser().resolve(),
        "enable_file": Path(args.enable_file).expanduser().resolve(),
        "execute": args.execute, "python_bin": args.python_bin,
    }
    while True:
        result = poll_once(**kwargs)
        print(json.dumps(result, sort_keys=True), flush=True)
        failed = any(item.get("status") == "FAILED" for item in result["jobs"])
        if args.once:
            return 1 if failed else 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
