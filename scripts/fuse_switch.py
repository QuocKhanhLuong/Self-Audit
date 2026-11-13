#!/usr/bin/env python3
"""Create a hash inventory and SUPERSEDED receipt for a Q3D boundary.

This helper is intentionally offline: importing it never inspects services,
stops processes, changes alarms, or deletes source history.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_boundary(run_dir: Path, receipt_path: Path, *, checkpoint: Path,
                      stage: str, update: int, logs: list[Path] | None = None,
                      metadata: list[Path] | None = None,
                      writers_stopped: bool = False) -> dict:
    run_dir, receipt_path = run_dir.resolve(), receipt_path.resolve()
    if writers_stopped is not True:
        raise ValueError("coordinator must confirm old-run writers are stopped before snapshot")
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError("explicit committed Q3D checkpoint must exist")
    if not checkpoint.is_relative_to(run_dir):
        raise ValueError("checkpoint must be under the old Q3D run directory")
    if stage not in {"T", "Q", "P", "J"} or type(update) is not int or update < 0:
        raise ValueError("checkpoint boundary requires a Q3D stage and nonnegative update")

    def inventory(path: Path, kind: str) -> dict:
        resolved = path.resolve()
        if not resolved.is_file() or not resolved.is_relative_to(run_dir):
            raise ValueError(f"explicit {kind} inventory path must be an existing file under old run")
        return {"kind": kind, "path": str(resolved.relative_to(run_dir)),
                "size_bytes": resolved.stat().st_size, "sha256": sha256(resolved)}

    artifacts = [inventory(checkpoint, "checkpoint")]
    artifacts.extend(inventory(path, "log") for path in (logs or []))
    artifacts.extend(inventory(path, "metadata") for path in (metadata or []))
    paths = [item["path"] for item in artifacts]
    if len(paths) != len(set(paths)):
        raise ValueError("checkpoint, log and metadata inventory paths must be unique")
    receipt = {"schema": "self_audit.candidate_fuse.switch_receipt.v1",
               "status": "SNAPSHOT_READY", "source_run": str(run_dir),
               "checkpoint_boundary": {"stage": stage, "update": update,
                                        "checkpoint": artifacts[0]},
               "artifacts": artifacts, "superseded_unrun_jobs": [],
               "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = receipt_path.with_name(receipt_path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, receipt_path)
    return receipt


def mark_superseded(receipt_path: Path, job_ids: list[str]) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "self_audit.candidate_fuse.switch_receipt.v1" or receipt.get("status") != "SNAPSHOT_READY":
        raise ValueError("valid snapshot receipt is required before superseding jobs")
    previous = receipt.get("superseded_unrun_jobs", [])
    if previous:
        raise ValueError("superseded job inventory is already committed")
    if len(set(job_ids)) != len(job_ids) or any(not x.strip() for x in job_ids):
        raise ValueError("job ids must be nonempty and unique")
    receipt["superseded_unrun_jobs"] = [{"job_id": x, "status": "SUPERSEDED"} for x in job_ids]
    receipt["status"] = "SWITCH_READY"
    tmp = receipt_path.with_name(receipt_path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, receipt_path)
    return receipt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--run-dir", required=True, type=Path)
    snap.add_argument("--receipt", required=True, type=Path)
    snap.add_argument("--checkpoint", required=True, type=Path)
    snap.add_argument("--stage", required=True, choices=("T", "Q", "P", "J"))
    snap.add_argument("--update", required=True, type=int)
    snap.add_argument("--writers-stopped", action="store_true", required=True,
                      help="coordinator-confirmed assertion after old-run writers were stopped")
    snap.add_argument("--log", action="append", default=[], type=Path)
    snap.add_argument("--metadata", action="append", default=[], type=Path)
    supersede = sub.add_parser("supersede")
    supersede.add_argument("--receipt", required=True, type=Path)
    supersede.add_argument("--job-id", action="append", default=[])
    a = p.parse_args(argv)
    result = (snapshot_boundary(a.run_dir, a.receipt, checkpoint=a.checkpoint,
                                stage=a.stage, update=a.update, logs=a.log, metadata=a.metadata,
                                writers_stopped=a.writers_stopped)
              if a.action == "snapshot" else mark_superseded(a.receipt, a.job_id))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
