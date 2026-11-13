#!/usr/bin/env python3
"""Read-only campaign watchdog; never kills workers or retries jobs itself."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
from pathlib import Path
from typing import Mapping

DEFAULT_INTERVAL = 420
TERMINAL = {"COMPLETED", "FAILED", "PAUSED"}


def _safe_metric(row: Mapping) -> dict:
    """Keep aggregate scalar metrics; do not copy per-patient rows to monitor."""
    safe = {key: row[key] for key in ("type", "event", "stage", "epoch", "update")
            if key in row and (key not in {"type", "event", "stage"} or isinstance(row[key], str))}
    sources = [row]
    if isinstance(row.get("losses"), Mapping):
        sources.append(row["losses"])
    for source in sources:
        for key in ("loss", "supervised", "soft_ce", "mimic_l", "mimic_u", "lambda_u"):
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                safe[key] = value
    modes = row.get("modes")
    if isinstance(modes, Mapping):
        safe_modes = {}
        for mode in ("teacher", "student", "fusion"):
            metric = modes.get(mode)
            if not isinstance(metric, Mapping):
                continue
            selected = {}
            fg = metric.get("fg_dice")
            if isinstance(fg, (int, float)) and math.isfinite(fg):
                selected["fg_dice"] = fg
            classes = metric.get("per_class")
            if isinstance(classes, Mapping):
                class_values = {}
                for name in ("RV", "MYO", "LV"):
                    value = classes.get(name)
                    value = value.get("dice") if isinstance(value, Mapping) else value
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        class_values[name] = value
                selected["per_class"] = class_values
            if selected:
                safe_modes[mode] = selected
        if safe_modes:
            safe["modes"] = safe_modes
    gates = row.get("gate_means")
    if isinstance(gates, (list, tuple)) and len(gates) == 5 and all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in gates):
        safe["gate_means"] = list(gates)
    return safe


def _read_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value, None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError):
        return None, "unreadable_or_invalid"


def inspect_run(run_dir: Path, *, now: float | None = None, stale_seconds: int = 1800) -> dict:
    now = time.time() if now is None else now
    state, err = _read_json(run_dir / "STATE.json")
    campaign, cerr = _read_json(run_dir / "RUN_STATE.json")
    result = {"status": "UNKNOWN", "alerts": [], "epoch": None, "stage": None,
              "update": None, "last_progress_age_seconds": None}
    campaign_running = isinstance(campaign, Mapping) and campaign.get("status") == "RUNNING"
    if err and not (err == "missing" and isinstance(campaign, Mapping) and campaign.get("status") == "RUNNING"):
        result["alerts"].append(f"state_{err}")
    elif isinstance(state, Mapping):
        result.update({k: state.get(k) for k in ("status", "stage", "epoch", "update")})
        stamp = state.get("updated_utc")
        if isinstance(stamp, str):
            try:
                age = now - dt.datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
                result["last_progress_age_seconds"] = max(0, int(age))
                if campaign_running and age > stale_seconds:
                    result["alerts"].append("stale_progress")
            except ValueError:
                result["alerts"].append("invalid_state_timestamp")
    if cerr:
        result["alerts"].append(f"campaign_{cerr}")
    elif isinstance(campaign, Mapping):
        # Engine STATE.json covers one stage. Campaign RUN_STATE.json owns
        # lifecycle, so a completed T/J snapshot cannot stop a live campaign.
        result["status"] = campaign.get("status", result["status"])
        if campaign.get("active") is not None:
            result["stage"] = campaign["active"]
        if campaign.get("status") == "RUNNING" and campaign.get("active") is None:
            result["alerts"].append("campaign_running_without_active_stage")
        if campaign.get("status") == "FAILED":
            result["alerts"].append("campaign_failed")
        if campaign.get("active") == "TEST_ONCE" and campaign.get("test_ledger", {}).get("status") not in {"RUNNING", "COMMITTED", "FAILED_LOCKED"}:
            result["alerts"].append("test_ledger_invalid")
    metrics = run_dir / "metrics.jsonl"
    try:
        if metrics.stat().st_size:
            metric_mtime = metrics.stat().st_mtime
            current_age = max(0, int(now - metric_mtime))
            previous_age = result.get("last_progress_age_seconds")
            if previous_age is None or current_age < previous_age:
                result["last_progress_age_seconds"] = current_age
            if campaign_running and current_age > stale_seconds:
                if "stale_progress" not in result["alerts"]:
                    result["alerts"].append("stale_progress")
            for line in reversed(metrics.read_text(encoding="utf-8", errors="replace").splitlines()):
                try:
                    row = json.loads(line)
                    if isinstance(row, Mapping):
                        result["latest_metric"] = _safe_metric(row)
                        break
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    result["healthy"] = not result["alerts"]
    return result


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL)
    p.add_argument("--stale-seconds", type=int, default=1800)
    p.add_argument("--once", action="store_true")
    a = p.parse_args(argv)
    if a.interval_seconds < 1 or a.stale_seconds < 1:
        p.error("interval and stale thresholds must be positive")
    run_dir = a.run_dir.resolve()
    monitor = run_dir / "WATCHDOG_STATUS.json"
    while True:
        status = inspect_run(run_dir, stale_seconds=a.stale_seconds)
        status["observed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_write(monitor, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        if a.once or status.get("status") in TERMINAL:
            return 0 if status.get("status") in {"COMPLETED", "PAUSED"} else 3
        time.sleep(a.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
