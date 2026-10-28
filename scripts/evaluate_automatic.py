#!/usr/bin/env python3
"""Evaluate automatic pilot checkpoints with strict image-only freeze before scoring.

Export full native uint8 argmax volumes before opening any reference masks.
Deterministic, no postprocessing.
Reports macro patient Dice, absent-reference FP diagnostics, and cold total output cost accounting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self_audit.automatic_evaluation import evaluate_automatic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strict evaluation of automatic NativeStudent pilot checkpoints"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory containing MANIFEST.json")
    parser.add_argument("--checkpoint-sha256", required=True, help="Expected SHA256 of checkpoint MANIFEST.json")
    parser.add_argument("--data", required=True, help="Path to evaluation data manifest (self_audit.data.v2)")
    parser.add_argument("--data-sha256", required=True, help="Expected SHA256 of evaluation data manifest")
    parser.add_argument("--out", required=True, help="New destination output directory (must not exist)")
    parser.add_argument("--device", default="cuda", help="Computation device (e.g. cuda, cpu)")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--threads", type=int, default=2, help="CPU thread count")
    parser.add_argument(
        "--compress-predictions",
        action="store_true",
        help="Compress predictions with deterministic gzip (.npy.gz)",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Perform image-only export without reference scoring",
    )

    args = parser.parse_args(argv)

    result = evaluate_automatic(
        checkpoint=args.checkpoint,
        checkpoint_sha=args.checkpoint_sha256,
        data_manifest=args.data,
        data_sha=args.data_sha256,
        out=args.out,
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
        compress_predictions=args.compress_predictions,
        inference_only=args.inference_only,
    )

    summary = {
        "status": result.get("status"),
        "role": result.get("role"),
        "coverage": result.get("coverage"),
        "scoring_coverage": result.get("scoring_coverage"),
        "macro_patient": result.get("macro_patient"),
        "foreground_patient_macro": result.get("foreground_patient_macro"),
        "all_class_patient_macro": result.get("all_class_patient_macro"),
        "cold_cost_diagnostics": result.get("cold_cost_diagnostics"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
