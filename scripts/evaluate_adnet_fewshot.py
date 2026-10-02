#!/usr/bin/env python3
"""Evaluate sealed ADNet few-shot outputs in a GT-only post-producer phase."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shared_benchmark.adnet_fewshot import (  # noqa: E402
    ADNetContractError,
    atomic_write_json,
    evaluate_adnet_outputs,
    load_gt_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--gt-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path)
    parser.add_argument("--classes", default="1,2,3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise ADNetContractError(f"refusing to overwrite evaluator output: {args.output}")
    class_ids = tuple(_parse_classes(args.classes))
    gt_manifest = load_gt_manifest(args.gt_manifest, gt_root=args.gt_root)
    result = evaluate_adnet_outputs(args.outputs_root, gt_manifest, class_ids=class_ids)
    atomic_write_json(args.output, result)
    print(json.dumps({
        "status": "COMPLETE",
        "schema": result["schema"],
        "macro_dice": result["macro_dice"],
        "macro_iou": result["macro_iou"],
        "sample_count": len(result["samples"]),
        "output": str(args.output),
    }, sort_keys=True))
    return 0


def _parse_classes(value: str) -> list[int]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        class_id = int(item)
        if class_id < 1 or class_id > 3:
            raise ADNetContractError("ADNet evaluator classes must be canonical foreground ids 1,2,3")
        result.append(class_id)
    if not result:
        raise ADNetContractError("at least one class id is required")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
