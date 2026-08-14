#!/usr/bin/env python3
"""Calibrate ``tau_accept`` from cached validation transitions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.evaluation.threshold import select_threshold, sweep_thresholds


def _load(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Cached validation transitions must be a mapping")
    if "transitions" in payload and isinstance(payload["transitions"], dict):
        payload = payload["transitions"]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", required=True, type=Path, help="torch.save mapping from validation inference")
    parser.add_argument("--output", required=True, type=Path, help="JSON artifact for the selected threshold")
    parser.add_argument("--min_tau", type=float, default=-0.20)
    parser.add_argument("--max_tau", type=float, default=0.20)
    parser.add_argument("--num_thresholds", type=int, default=41)
    parser.add_argument("--max_harmful_acceptance_rate", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_thresholds < 1:
        raise ValueError("--num_thresholds must be positive")
    transitions = _load(args.transitions)
    thresholds = np.linspace(float(args.min_tau), float(args.max_tau), int(args.num_thresholds)).tolist()
    rows = sweep_thresholds(
        transitions,
        thresholds,
        max_harmful_acceptance_rate=args.max_harmful_acceptance_rate,
    )
    selected = select_threshold(rows)
    artifact = {
        "source": str(args.transitions),
        "objective": "maximize final validation macro Dice",
        "constraint": args.max_harmful_acceptance_rate,
        "selected": selected,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(selected, sort_keys=True))
    print(f"saved={args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
