#!/usr/bin/env python
"""CLI diagnostic harness for system, cgroup, process, filesystem, and GPU resources.

Designed for self-audit maskfree runtime diagnostics without interfering with
training, modifying cgroups or system caches, or leaking environment secrets.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src/ is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from self_audit_maskfree.resources import (
    RESOURCE_SCHEMA_VERSION,
    compute_resource_deltas,
    resource_snapshot,
)


def _bounded_interval(val: str) -> float:
    try:
        f = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid interval: {val!r}")
    if not math.isfinite(f) or f < 0.1:
        raise argparse.ArgumentTypeError("Interval cannot be less than 0.1s")
    if f > 60.0:
        raise argparse.ArgumentTypeError("Interval cannot exceed 60.0s")
    return f


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose maskfree execution environment and system resources."
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Target process ID to inspect (default: current PID).",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        dest="data_roots",
        help="Data root directory to inspect (repeatable).",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Report directory to inspect.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (writes atomically). If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--interval",
        type=_bounded_interval,
        default=5.0,
        help="Bounded sampling interval in seconds (default 5.0s, max 60.0s).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        choices=range(1, 601),
        help="Number of snapshots to take (default: 1; if >= 2, computes interval deltas).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable bounded nvidia-smi GPU sampling.",
    )
    return parser.parse_args(args)


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    target_pid = args.pid if args.pid is not None else os.getpid()
    include_gpu = not args.no_gpu

    initial_snapshot = resource_snapshot(
        pid=target_pid,
        data_roots=args.data_roots,
        report_dir=args.report_dir,
        include_gpu=include_gpu,
    )

    if args.count <= 1:
        payload: dict[str, Any] = {
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "mode": "single_snapshot",
            "snapshot": initial_snapshot,
        }
    else:
        snapshots = [initial_snapshot]
        for _ in range(args.count - 1):
            time.sleep(args.interval)
            snapshots.append(resource_snapshot(
                pid=target_pid, data_roots=args.data_roots,
                report_dir=args.report_dir, include_gpu=include_gpu))
        second_snapshot = snapshots[-1]
        deltas = compute_resource_deltas(initial_snapshot, second_snapshot)
        payload = {
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "mode": "interval_deltas",
            "interval_seconds": args.interval,
            "initial_snapshot": initial_snapshot,
            "final_snapshot": second_snapshot,
            "deltas": deltas,
            "snapshots": snapshots,
            "interval_deltas": [compute_resource_deltas(a, b) for a, b in zip(snapshots, snapshots[1:])],
        }

    return payload


def main() -> int:
    args = parse_args()
    payload = run_diagnostics(args)
    text = json.dumps(payload, indent=2, sort_keys=True)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f"{out_path.name}.tmp-{os.getpid()}")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
    else:
        sys.stdout.write(text + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
