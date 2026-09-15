#!/usr/bin/env python
"""Isolated reference evaluation for a frozen mask-free prediction set.

This is a standalone process. It is the only entry point in the mask-free
pipeline that may read manual segmentation masks, and it can only run after a
freeze manifest exists and validates:

    python scripts/evaluate_maskfree_reference.py \
        --frozen-manifest runs/maskfree150/acdc/<run_id>/freeze_manifest.json \
        --reference-config configs/reference_acdc.json \
        --output runs/maskfree150/acdc/<run_id>/reference

Nothing here imports the trainer, the producer, the auditor or any config used
for training, and nothing it computes is returned to them. Its only outputs are
``reference_metrics.json``, ``.csv`` and ``.md`` under ``--output``.

A missing or unreadable reference produces a report whose metrics are marked
unavailable with a reason. It never produces a zero Dice and never blocks the
image-only pipeline result, which stands on its own.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from self_audit_maskfree.evaluation.freeze import (  # noqa: E402
    FreezeValidationError,
    load_freeze_manifest,
)
from self_audit_maskfree.evaluation.reference import (  # noqa: E402
    ReferenceConfig,
    ReferenceConfigError,
    evaluate_reference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_maskfree_reference.py",
        description=(
            "Evaluate a frozen mask-free prediction set against manual reference masks. "
            "Runs only after the freeze manifest validates by file hash; reads masks only "
            "through an explicit reference config; returns nothing to training."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Reference masks are optional. Without them every reference metric is reported "
            "as unavailable with a reason - never as zero."
        ),
    )
    parser.add_argument(
        "--frozen-manifest", required=True, type=Path,
        help="Path to the freeze manifest JSON produced at finalization.",
    )
    parser.add_argument(
        "--reference-config", required=True, type=Path,
        help="Explicit reference configuration (JSON or YAML). Separate from any training config.",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Directory for reference_metrics.json / .csv / .md.",
    )
    parser.add_argument(
        "--freeze-root", type=Path, default=None,
        help="Base directory for relative paths in the freeze manifest. "
             "Defaults to the manifest's own output_dir, then to its directory.",
    )
    parser.add_argument(
        "--print-summary", action="store_true",
        help="Print the available metric rows to stdout after writing the reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = ReferenceConfig.load(arguments.reference_config)
    except ReferenceConfigError as error:
        print(f"reference config error: {error}", file=sys.stderr)
        return 2
    try:
        manifest = load_freeze_manifest(arguments.frozen_manifest)
        report = evaluate_reference(
            manifest, config, output_dir=arguments.output, freeze_root=arguments.freeze_root
        )
    except FreezeValidationError as error:
        print(f"freeze validation failed, refusing to evaluate: {error}", file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError) as error:
        print(f"reference evaluation failed: {error}", file=sys.stderr)
        return 4

    print(f"wrote reference reports to {arguments.output}")
    print(f"freeze validated: {report['freeze_receipt']['files_checked']} file(s)")
    if not report["available"]:
        print(f"reference metrics unavailable: {report['reason']}")
    if arguments.print_summary:
        for row in report["rows"]:
            state = f"{row['value']:.6g} {row['unit']}" if row["available"] else f"unavailable ({row['reason']})"
            print(f"  {row['name']}: {state} [n={row['count']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
