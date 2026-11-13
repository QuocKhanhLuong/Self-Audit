#!/usr/bin/env python3
"""Tail measured epoch loss, VAL3 modes and fusion gates from a FUSE run."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Mapping


def _pct(x):
    try:
        x = float(x)
        return f"{100*x:6.2f}" if math.isfinite(x) else "    NA"
    except (TypeError, ValueError):
        return "    NA"


def format_metric(record: Mapping) -> str | None:
    event = record.get("event", record.get("type"))
    if event not in {"train_epoch", "epoch", "validation", "development_epoch", "train_update", "validation_selected"}:
        return None
    modes = record.get("modes", {})
    epoch = record.get("epoch", "-")
    if event == "train_update" and isinstance(epoch, int):
        epoch += 1
    parts = [f"{record.get('stage', '?')} epoch={epoch} update={record.get('update', '-')}"]
    losses = record.get("losses", {})
    if isinstance(losses, Mapping):
        for name, value in losses.items():
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                parts.append(f"{name}={float(value):.5g}")
    if isinstance(record.get("loss"), (float, int)) and math.isfinite(float(record["loss"])):
        parts.append(f"loss={float(record['loss']):.5g}")
    has_modes = isinstance(modes, Mapping) and bool(modes)
    if has_modes:
        for mode in ("teacher", "student", "fusion"):
            metrics = modes.get(mode, {})
            if isinstance(metrics, Mapping):
                fg = metrics.get("fg_dice", metrics.get("score"))
                classes = metrics.get("per_class", {})
                if isinstance(classes, Mapping):
                    values = [classes.get(k) for k in ("RV", "MYO", "LV")]
                    rv, myo, lv = (v.get("dice") if isinstance(v, Mapping) else v for v in values)
                else:
                    rv = myo = lv = None
                parts.append(f"{mode} FG/RV/MYO/LV={_pct(fg)}/{_pct(rv)}/{_pct(myo)}/{_pct(lv)}%")
    gates = record.get("gate_means")
    fusion = modes.get("fusion", {}) if isinstance(modes, Mapping) else {}
    if gates is None and isinstance(fusion, Mapping):
        gates = fusion.get("gate_means")
    if gates is None:
        losses_for_gates = record.get("losses", record)
        if isinstance(losses_for_gates, Mapping):
            left, right = losses_for_gates.get("gate_means_l"), losses_for_gates.get("gate_means_u")
        else:
            left, right = None, None
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right) == 5:
            gates = []
            for a, b in zip(left, right):
                finite = [float(value) for value in (a, b)
                          if isinstance(value, (int, float)) and math.isfinite(float(value))]
                gates.append(sum(finite) / len(finite) if finite else None)
    if isinstance(gates, (list, tuple)) and len(gates) == 5:
        parts.append("G=" + "/".join(_pct(x) for x in gates))
    return " | ".join(parts)


def read_complete_rows(path: Path, offset: int = 0):
    with path.open("rb") as f:
        f.seek(offset)
        while True:
            start = f.tell()
            raw = f.readline()
            if not raw or not raw.endswith(b"\n"):
                f.seek(start)
                break
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(value, Mapping):
                yield value
        return f.tell()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval-seconds", type=float, default=2)
    a = p.parse_args(argv)
    path = a.run_dir / "metrics.jsonl"
    offset = 0
    print("FUSE live metrics: observed values only", flush=True)
    while True:
        if path.exists():
            with path.open("rb") as f:
                f.seek(offset)
                while True:
                    start = f.tell()
                    raw = f.readline()
                    if not raw or not raw.endswith(b"\n"):
                        f.seek(start)
                        break
                    offset = f.tell()
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, Mapping):
                        line = format_metric(row)
                        if line:
                            print(line, flush=True)
        if a.once:
            return 0
        time.sleep(max(0.1, a.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
