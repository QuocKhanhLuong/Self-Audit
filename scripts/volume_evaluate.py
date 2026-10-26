#!/usr/bin/env python3
"""Run hash-pinned frozen native-volume inference and optional scoring.

The command performs real image-only sliding/slab inference for a separate
evaluation manifest. Reference masks are opened only with ``--score`` (or its
explicit ``--allow-labels`` alias), after all native probabilities are merged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.volume.checkpoint import load_volume_checkpoint
from self_audit.volume.evaluation import evaluate_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--prediction-model", choices=("student", "ema"), default="student")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default=None)
    parser.add_argument("--slab-depth", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=0)
    parser.add_argument("--slab-overlap", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--score", "--allow-labels", dest="allow_labels", action="store_true",
                        help="explicitly open frozen evaluation masks after predictions are complete")
    parser.add_argument("--predictions-dir", type=Path,
                        help="optional new directory for merged native probability/support arrays")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    manifest_root = args.manifest.resolve().parent
    output_path = args.output.resolve()
    if output_path == manifest_root or output_path.is_relative_to(manifest_root):
        raise ValueError("evaluation output must be outside the source manifest directory")
    if args.predictions_dir is not None:
        prediction_path = args.predictions_dir.resolve()
        if prediction_path == manifest_root or prediction_path.is_relative_to(manifest_root):
            raise ValueError("prediction output must be outside the source manifest directory")
        if prediction_path.exists():
            raise ValueError(f"prediction output directory already exists: {prediction_path}")
    trainer, manifest = load_volume_checkpoint(args.checkpoint, args.checkpoint_sha256, device=args.device)
    result = evaluate_manifest(
        trainer.model, args.manifest, args.manifest_sha256, args.geometry, args.geometry_sha256,
        teacher=trainer.teacher, prediction_model=args.prediction_model, split=args.split,
        allow_labels=args.allow_labels, synthetic=args.synthetic, slab_depth=args.slab_depth,
        tile_size=args.tile_size, tile_overlap=args.tile_overlap, slab_overlap=args.slab_overlap,
        config=manifest.get("config") if isinstance(manifest, dict) else None,
        checkpoint_sha256=args.checkpoint_sha256, predictions_dir=args.predictions_dir, device=args.device,
        evaluation_source_sha256=(manifest.get("source_sha256") if isinstance(manifest, dict) else None),
        cli_flags={
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": args.checkpoint_sha256,
            "manifest": str(args.manifest), "manifest_sha256": args.manifest_sha256,
            "geometry": str(args.geometry), "geometry_sha256": args.geometry_sha256,
            "prediction_model": args.prediction_model, "device": args.device,
            "split": args.split, "slab_depth": args.slab_depth, "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap, "slab_overlap": args.slab_overlap,
            "synthetic": args.synthetic, "allow_labels": args.allow_labels,
            "predictions_dir": str(args.predictions_dir) if args.predictions_dir is not None else None,
            "output": str(args.output),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
