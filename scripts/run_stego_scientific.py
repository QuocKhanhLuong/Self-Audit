#!/usr/bin/env python3
"""Scientific runner for STEGO cardiac benchmark.

Usage:
    python scripts/run_stego_scientific.py \
        --manifest /path/to/manifest.json \
        --output-dir /path/to/output \
        --checkpoint /path/to/cityscapes_vit_base_1.ckpt \
        --split train \
        [--with-adapter] \
        [--profile STEGO-2D]

Follows the same pattern as CUTS:
  Load manifest → iterate records → STEGO inference → anonymous partition
  → seal raw bundle → optionally run evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Inject STEGO src into path
STEGO_ROOT = Path(__file__).resolve().parents[1] / "baseline" / "STEGO"
STEGO_SRC = STEGO_ROOT / "src"
sys.path.insert(0, str(STEGO_SRC))

from cardiac_benchmark.config import STEGOConfig
from cardiac_benchmark.dataset import STEGOCardiacDataset, collate_image_only
from cardiac_benchmark.freeze import create_freeze, seal_raw_bundle
from cardiac_benchmark.manifest import load_manifest, require_scientific_manifest
from cardiac_benchmark.output import write_raw_partition
from cardiac_benchmark.provenance import (
    checkpoint_provenance,
    current_stego_sha,
    environment_identity,
    rng_contract,
    sha256_json,
    write_json,
)
from cardiac_benchmark.stego_runner import (
    build_run_metadata,
    load_stego_model,
    run_stego_on_sample,
    seed_deterministic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="STEGO cardiac benchmark scientific runner")
    parser.add_argument("--manifest", required=True, help="Path to shared manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for artifacts")
    parser.add_argument("--checkpoint", required=True, help="Path to STEGO checkpoint")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--profile", default="STEGO-2D")
    parser.add_argument("--with-adapter", action="store_true", help="Run semantic adapter after inference")
    parser.add_argument("--scientific", action="store_true", help="Enforce scientific manifest")
    parser.add_argument("--device", default=None, help="Compute device (default: auto)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = STEGOConfig(
        profile=args.profile,
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        scientific_run=args.scientific,
    )
    config.validate()

    if args.scientific:
        manifest = require_scientific_manifest(args.manifest)
    else:
        manifest = load_manifest(args.manifest)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    seed_deterministic(config.benchmark_seed)

    print(f"Loading STEGO model from {args.checkpoint}...")
    model = load_stego_model(config, device=device)

    dataset = STEGOCardiacDataset(manifest, split=args.split, profile=args.profile)
    print(f"Processing {len(dataset)} samples (split={args.split})...")

    run_start = time.perf_counter()
    partitions = []
    raw_dir = output_dir / "raw_partitions"

    for i in range(len(dataset)):
        sample = dataset[i]
        result = run_stego_on_sample(model, sample, config=config, device=device)
        result = write_raw_partition(result, raw_dir)
        partitions.append(result)
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(dataset)}] processed")

    run_elapsed = time.perf_counter() - run_start

    sample_ids = {r["sample_id"] for r in manifest["records"] if r["split"] == args.split}
    bundle = seal_raw_bundle(partitions, raw_dir, required_sample_ids=sample_ids)

    ckpt_prov = checkpoint_provenance(args.checkpoint)
    freeze = create_freeze({
        "model_type": config.model_type,
        "checkpoint_sha256": ckpt_prov["checkpoint_sha256"],
        "n_classes": config.n_classes,
        "resolution": config.resolution,
        "dim": config.dim,
        "normalization": "stego.cardiac.fixed_affine_v1",
        "projection_type": config.projection_type,
        "benchmark_seed": config.benchmark_seed,
    })

    gpu_hours = 0.0
    if torch.cuda.is_available():
        gpu_hours = run_elapsed / 3600.0

    receipt = {
        "schema_version": "stego.cardiac.run-receipt.v1",
        "split": args.split,
        "profile": args.profile,
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "freeze": freeze,
        "checkpoint_provenance": ckpt_prov,
        "rng_contract": rng_contract(),
        "environment": environment_identity(),
        "manifest_hash": manifest["manifest_hash"],
        "bundle_hash": bundle["bundle_hash"],
        "total_samples": len(partitions),
        "successful_samples": sum(1 for p in partitions if p["status"] == "success"),
        "total_elapsed_seconds": run_elapsed,
        "gpu_hours": gpu_hours,
    }
    try:
        receipt["stego_git_sha"] = current_stego_sha()
    except Exception:
        receipt["stego_git_sha"] = "unavailable"

    write_json(output_dir / "run_receipt.json", receipt)
    print(f"\nDone: {len(partitions)} partitions in {run_elapsed:.1f}s")
    print(f"Bundle hash: {bundle['bundle_hash']}")
    print(f"Receipt: {output_dir / 'run_receipt.json'}")


if __name__ == "__main__":
    main()
