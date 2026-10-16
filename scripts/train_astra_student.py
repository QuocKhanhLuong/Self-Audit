#!/usr/bin/env python3
"""Train the explicit DINOv3 layer-guided student; never selects the old CNN."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO / "experiments" / "layer_guided_student_v1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank",
        type=Path,
        required=True,
        help="Original frozen TRAIN bank with patient .npz arrays",
    )
    parser.add_argument(
        "--bank-manifest-sha256",
        required=True,
        help="Externally pinned original FROZEN.json digest",
    )
    parser.add_argument(
        "--approval",
        type=Path,
        required=True,
        help="Reviewed local official DINOv3 source/weights receipt",
    )
    parser.add_argument(
        "--approval-sha256",
        required=True,
        help="Externally pinned approval receipt digest",
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=EXPERIMENT / "training_v1" / "pilot.json",
        help="Defaults to the bounded 32-update pilot",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Fresh output directory; never overwritten",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="No automatic device fallback",
    )
    args = parser.parse_args(argv)
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(EXPERIMENT))
    try:
        from training_v1.runner import run_training

        result = run_training(
            bank=args.bank,
            expected_bank_sha256=args.bank_manifest_sha256,
            approval=args.approval,
            expected_approval_sha256=args.approval_sha256,
            config=json.loads(args.recipe.read_text()),
            out=args.out,
            device=args.device,
        )
    except (OSError, ValueError, RuntimeError, FloatingPointError) as exc:
        print(
            f"[ASTRA_FAILED] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True
        )
        return 1
    print("[ASTRA_COMPLETE] " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
