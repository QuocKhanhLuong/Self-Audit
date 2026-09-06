#!/usr/bin/env python3
"""Calibrate ``tau_accept`` from cached validation transitions only.

The selected threshold is written as a schema-v1 calibration artifact
(``self_audit.evaluation.threshold.save_calibration``) so that whatever runs
inference can read back the exact ``tau_accept`` and ``neutral_margin`` that
were calibrated, instead of silently falling back to a hard-coded default.

The artifact is stamped ``"validity": "diagnostic_only"``: ``tau_accept`` is
chosen by argmax over the grid on the same validation split whose Dice it then
reports, and this checkpoint has no held-out test set.
"""

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

from self_audit.audit.semantics import (
    METRIC_SPACES,
    METRIC_SPACE_SLICE_PROXY,
    resolve_neutral_margin,
)
from self_audit.evaluation.threshold import save_calibration, select_threshold, sweep_thresholds


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


def _cached_t_max(transitions: dict[str, object]) -> int:
    delta_q = transitions.get("delta_q")
    shape = getattr(delta_q, "shape", None)
    if shape is None:
        return 0
    return int(shape[1]) if len(shape) > 1 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", required=True, type=Path, help="torch.save mapping from validation inference")
    parser.add_argument("--output", required=True, type=Path, help="JSON artifact for the selected threshold")
    parser.add_argument("--min_tau", type=float, default=-0.20)
    parser.add_argument("--max_tau", type=float, default=0.20)
    parser.add_argument("--num_thresholds", type=int, default=41)
    parser.add_argument("--max_harmful_acceptance_rate", type=float, default=None)
    parser.add_argument(
        "--neutral_margin",
        type=float,
        default=None,
        help="Decision margin for beneficial/neutral/harmful; defaults to the canonical DEFAULT_NEUTRAL_MARGIN",
    )
    parser.add_argument(
        "--source_split",
        default="val",
        help="Name of the split the cached transitions came from (recorded in the artifact)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint the transitions were cached from; its SHA-256 is recorded when readable",
    )
    parser.add_argument(
        "--metric_space",
        default=METRIC_SPACE_SLICE_PROXY,
        choices=list(METRIC_SPACES),
        help="Metric space the cached Dice values live in",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_thresholds < 1:
        raise ValueError("--num_thresholds must be positive")
    neutral_margin = resolve_neutral_margin(args.neutral_margin)
    transitions = _load(args.transitions)
    thresholds = np.linspace(float(args.min_tau), float(args.max_tau), int(args.num_thresholds)).tolist()
    rows = sweep_thresholds(
        transitions,
        thresholds,
        max_harmful_acceptance_rate=args.max_harmful_acceptance_rate,
        neutral_margin=neutral_margin,
    )
    selected = select_threshold(rows)
    payload = save_calibration(
        args.output,
        tau_accept=float(selected["tau_accept"]),
        neutral_margin=neutral_margin,
        source_split=str(args.source_split),
        checkpoint_path=args.checkpoint,
        t_max=_cached_t_max(transitions),
        threshold_grid={"min": float(args.min_tau), "max": float(args.max_tau), "steps": int(args.num_thresholds)},
        selected_row=selected,
        metric_space=str(args.metric_space),
        extra={
            "source_transitions": str(args.transitions),
            "max_harmful_acceptance_rate": args.max_harmful_acceptance_rate,
            "rows": rows,
        },
    )
    print(json.dumps(selected, sort_keys=True))
    print(f"validity={payload['validity']} neutral_margin={payload['neutral_margin']}")
    print(f"saved={args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
