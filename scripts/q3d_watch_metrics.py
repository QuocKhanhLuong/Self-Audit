#!/usr/bin/env python3
"""Read-only compact view of per-class Q3D development metrics."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def _percent(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "    NA"
    return f"{100 * value:6.2f}" if math.isfinite(value) else "    NA"


def metric_line(job, record):
    if record.get("event") not in {"development_selection", "development_epoch"}:
        return None
    classes = record.get("per_class") or record.get("teacher_dev_per_class") or {}
    epoch = record.get("epoch")
    return (
        f"{job['setting']:>5} K={job['k']:<2} {record['stage']} "
        f"epoch={epoch if epoch is not None else '-':>4} update={record['update']:>5} | "
        f"RV {_percent(classes.get('RV'))}%  MYO {_percent(classes.get('MYO'))}%  "
        f"LV {_percent(classes.get('LV'))}%  FG {_percent(record.get('score'))}%"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    jobs = {Path(job['output']) / 'metrics.jsonl': job for job in plan['stage_jobs']}
    positions = {}
    print('DEV patient-macro Dice (%) — measured values only; epoch=- means no epoch was recorded.', flush=True)
    while True:
        for path, job in jobs.items():
            if not path.exists():
                continue
            with path.open() as handle:
                handle.seek(positions.get(path, 0))
                while True:
                    start = handle.tell()
                    raw = handle.readline()
                    if not raw or not raw.endswith('\n'):
                        handle.seek(start)
                        break
                    row = json.loads(raw)
                    line = metric_line(job, row)
                    if line is not None:
                        print(line, flush=True)
                positions[path] = handle.tell()
        if args.once:
            return
        time.sleep(2)


if __name__ == '__main__':
    main()
