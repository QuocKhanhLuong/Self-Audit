#!/usr/bin/env python3
"""Resumable A4-SA pipeline: S1 -> S2 -> S3-SA -> A4-SA -> FINAL, with DONE markers and a budget guard.

* Each stage writes ``<out>/markers/<stage>.DONE.json`` (config hash, receipt/artifact hashes, seconds).
  A restart skips stages whose marker matches the config hash, and resumes an unfinished stage
  from its last checkpoint (the stage commands are always run with ``--resume``).
* Budget guard: per-stage caps from env ``CAP_HOURS_<STAGE>`` (projected hours); a stage is aborted
  (SIGTERM, then ``ABORTED_BUDGET.json``) when it exceeds ``1.3 x`` its cap, or when the total wall
  clock exceeds ``MAX_HOURS``.  Training checkpoints atomically, so an aborted stage resumes later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("S1", "S2", "S3", "A4", "FINAL")
OVERRUN = 1.3


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_command(stage: str, config: Path, out: Path, device: str) -> list[str]:
    py = sys.executable
    if stage in ("S1", "S2", "S3"):
        return [py, str(ROOT / "scripts/pg_campaign.py"), "--config", str(config), "--out", str(out), "--device", device, "--stage", stage, "--resume"]
    if stage == "A4":
        return [py, str(ROOT / "scripts/pg_campaign.py"), "--config", str(config), "--out", str(out), "--device", device, "--stage", "arms", "--arm", "A4", "--resume"]
    return [py, str(ROOT / "scripts/cbkd_sa_finalize.py"), "--config", str(config), "--out", str(out), "--device", device]


def artifacts(stage: str, out: Path) -> list[Path]:
    mapping = {"S1": ["S1_RECEIPT.json"], "S2": ["S2_RECEIPT.json"], "S3": ["S3_RECEIPT.json"],
               "A4": ["arms/A4/epochs.csv"], "FINAL": ["FINAL_RESULT.json"]}
    return [out / name for name in mapping[stage]]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run_stage(stage: str, cmd: list[str], cap_hours: float | None, deadline: float | None, poll: float = 5.0) -> tuple[int, str]:
    """Run one stage; returns (exit_code, reason) with reason in {"", "stage_cap", "max_hours"}."""

    start = time.time()
    proc = subprocess.Popen(cmd)
    reason = ""
    while True:
        try:
            return proc.wait(timeout=poll), reason
        except subprocess.TimeoutExpired:
            pass
        now = time.time()
        if cap_hours is not None and now - start > OVERRUN * cap_hours * 3600:
            reason = "stage_cap"
        elif deadline is not None and now > deadline:
            reason = "max_hours"
        if reason:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            return 124, reason


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    config_sha = sha_file(args.config)
    pipeline_start = time.time()
    max_hours = float(os.environ["MAX_HOURS"]) if os.environ.get("MAX_HOURS") else None
    deadline = pipeline_start + max_hours * 3600 if max_hours else None
    for stage in [s for s in args.stages.split(",") if s]:
        marker = out / "markers" / f"{stage}.DONE.json"
        if marker.is_file() and json.loads(marker.read_text()).get("config_sha256") == config_sha:
            print(f"[pipeline] {stage}: DONE marker present, skipping", flush=True)
            continue
        cap = float(os.environ[f"CAP_HOURS_{stage}"]) if os.environ.get(f"CAP_HOURS_{stage}") else None
        print(f"[pipeline] {stage}: start (cap {cap} h, max {max_hours} h)", flush=True)
        t0 = time.time()
        code, reason = run_stage(stage, stage_command(stage, args.config, out, args.device), cap, deadline, args.poll_seconds)
        if code != 0:
            status = "ABORTED_BUDGET" if reason else "FAILED"
            write_json(out / "markers" / f"{stage}.{status}.json", {"stage": stage, "reason": reason or f"exit {code}", "exit_code": code,
                                                                   "elapsed_seconds": time.time() - t0, "config_sha256": config_sha})
            print(f"[pipeline] {stage}: {status} ({reason or code}); re-run the same command to resume", flush=True)
            return 3 if reason else 1
        write_json(marker, {"stage": stage, "config_sha256": config_sha, "elapsed_seconds": time.time() - t0,
                            "artifact_sha256": {str(p.relative_to(out)): sha_file(p) for p in artifacts(stage, out) if p.is_file()}})
        print(f"[pipeline] {stage}: DONE in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
