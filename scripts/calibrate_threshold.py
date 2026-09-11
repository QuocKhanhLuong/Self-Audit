#!/usr/bin/env python3
"""Calibrate ``tau_accept`` from cached validation transitions only.

The selected threshold is written as a schema-v2 calibration artifact
(``self_audit.evaluation.threshold.save_calibration``) so that whatever runs
inference can read back the exact ``tau_accept`` and ``neutral_margin`` that
were calibrated, instead of silently falling back to a hard-coded default.

The artifact carries the producer's ``lineage`` block, copied out of the
transition cache that was measured.  That block names the weights, the
preprocessing recipe, the metric contract, the split membership and the
cohort the threshold was measured under, and a consumer verifies it against
its own runtime before applying the threshold.  A cache without a lineage
cannot produce a verifiable artifact and is refused here rather than written
out as a threshold nobody can check.

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
from self_audit.evaluation.contracts import (
    ContractMismatchError,
    resolve_metric_contract,
)
from self_audit.evaluation.calibration_lineage import (
    CalibrationLineageError,
    validate_lineage_completeness,
)
from self_audit.evaluation.threshold import (
    NoFeasibleThresholdError,
    json_safe,
    save_calibration,
    select_threshold,
    sweep_thresholds,
    validate_and_normalize_transition_cache,
)


def _load(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Failed to load transition cache from {path} with weights_only=True: "
            f"unsupported legacy cache format or unsafe objects detected: {exc}"
        ) from exc
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


def _artifact_lineage(
    transitions: dict[str, object],
    *,
    metric_space: str,
    neutral_margin: float,
    t_max: int,
) -> dict[str, object]:
    """Return the cache's lineage with the measured semantics stamped in.

    The producer records the metric contract in full but may leave the
    top-level ``metric_space`` / ``neutral_margin`` / ``t_max`` slots unset.
    They are filled here from the *validated cache*, never invented: a slot the
    producer already filled must agree with what the cache validates to, and a
    disagreement is a hard error rather than an overwrite.
    """

    lineage = transitions.get("lineage")
    if not isinstance(lineage, dict):
        raise CalibrationLineageError(
            "Cached transitions carry no 'lineage' block, so the calibrated threshold could "
            "not be tied to the weights, preprocessing recipe, metric contract or cohort it "
            "was measured on. Re-cache with scripts/cache_validation_transitions.py, which "
            "records one."
        )
    record = json.loads(json.dumps(json_safe(lineage)))
    semantics = record.get("semantics")
    if not isinstance(semantics, dict):
        raise CalibrationLineageError("Cached lineage has no 'semantics' block")
    measured = {
        "metric_space": str(metric_space),
        "neutral_margin": float(neutral_margin),
        "t_max": int(t_max),
    }
    for key, value in measured.items():
        existing = semantics.get(key)
        if existing is not None and existing != value:
            raise CalibrationLineageError(
                f"Cached lineage declares semantics.{key}={existing!r} but the validated cache "
                f"measures {value!r}; refusing to overwrite a producer-recorded semantic"
            )
        semantics[key] = value
    return validate_lineage_completeness(record, where="cached transition lineage")


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
    parser.add_argument(
        "--metric_contract",
        default="foreground_dice_exclude_v1",
        help="Expected metric contract name (default: foreground_dice_exclude_v1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_thresholds < 1:
        raise ValueError("--num_thresholds must be positive")
    neutral_margin = resolve_neutral_margin(args.neutral_margin)
    transitions = _load(args.transitions)

    try:
        norm = validate_and_normalize_transition_cache(
            transitions,
            expected_contract=args.metric_contract,
            neutral_margin=args.neutral_margin,
            strict_contract=True,
        )
    except (ContractMismatchError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    # Validate metric_space compatibility: CLI cannot stamp a mismatched metric space onto cached transitions
    if str(norm["metric_space"]) != str(args.metric_space):
        print(
            f"Error: CLI --metric_space '{args.metric_space}' conflicts with cached metric_space '{norm['metric_space']}'. "
            "Cannot stamp a different metric space onto cached transitions.",
            file=sys.stderr,
        )
        sys.exit(1)

    thresholds = np.linspace(float(args.min_tau), float(args.max_tau), int(args.num_thresholds)).tolist()
    try:
        rows = sweep_thresholds(
            norm,
            thresholds,
            max_harmful_acceptance_rate=args.max_harmful_acceptance_rate,
            neutral_margin=args.neutral_margin,
            expected_contract=args.metric_contract,
        )
        selected = select_threshold(rows)
    except (NoFeasibleThresholdError, ContractMismatchError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    t_max = _cached_t_max(transitions)
    try:
        lineage = _artifact_lineage(
            transitions,
            metric_space=str(norm["metric_space"]),
            neutral_margin=float(selected["neutral_margin"]),
            t_max=t_max,
        )
    except CalibrationLineageError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    try:
        payload = save_calibration(
            args.output,
            tau_accept=float(selected["tau_accept"]),
            neutral_margin=float(selected["neutral_margin"]),
            source_split=str(args.source_split),
            checkpoint_path=args.checkpoint,
            t_max=t_max,
            threshold_grid={"min": float(args.min_tau), "max": float(args.max_tau), "steps": int(args.num_thresholds)},
            selected_row=selected,
            metric_space=str(args.metric_space),
            metric_contract=str(args.metric_contract),
            lineage=lineage,
            extra={
                "source_transitions": str(args.transitions),
                "max_harmful_acceptance_rate": args.max_harmful_acceptance_rate,
                "rows": rows,
            },
        )
    except (ContractMismatchError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(json_safe(selected), sort_keys=True))
    print(f"validity={payload['validity']} neutral_margin={payload['neutral_margin']}")
    print(
        f"lineage_state_digest={lineage['checkpoint']['state_digest']} "
        f"lineage_split={lineage['split']['split_name']}"
    )
    print(f"saved={args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
