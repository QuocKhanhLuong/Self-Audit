#!/usr/bin/env python3
"""Isolated per-epoch reference CLI; never imported by the training process."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-config")
    args = parser.parse_args(argv)
    from self_audit_maskfree.evaluation.epoch_reference import evaluate_epoch
    from self_audit_maskfree.evaluation.metrics import write_json
    try:
        evaluate_epoch(args.frozen_manifest, args.image_manifest, args.output, args.reference_config)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "epoch_validation.json", {
            "schema_version": "maskfree150.epoch_validation.v1", "status": "FAILED", "available": False,
            "reason": f"{type(error).__name__}: {error}", "students": {},
        })
        traceback.print_exc()
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
