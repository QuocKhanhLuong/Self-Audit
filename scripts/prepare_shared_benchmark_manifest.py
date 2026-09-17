#!/usr/bin/env python
"""Materialize the shared projection from FreeMask's authoritative discovery."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_audit_maskfree.data.discovery import discover_dataset
from shared_benchmark.manifest import build_shared_manifest, write_shared_manifest
from shared_benchmark.provenance import sha256_file
from shared_benchmark.spatial import load_pinned_grid_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a shared image-only benchmark manifest")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--dataset", required=True, choices=("acdc", "mnms"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--upstream-manifest", help="existing FreeMask manifest; otherwise run discovery")
    args = parser.parse_args()
    if args.upstream_manifest:
        from self_audit_maskfree.data.discovery import load_manifest

        upstream_path = Path(args.upstream_manifest)
        upstream = load_manifest(upstream_path)
        upstream_file_hash = sha256_file(upstream_path)
    else:
        upstream = discover_dataset(args.image_root, args.dataset, seed=42, protocol="auto", depth_axis=2)
        upstream_file_hash = None
    if upstream["dataset"] != args.dataset:
        raise SystemExit("dataset argument does not match authoritative FreeMask manifest")
    manifest = build_shared_manifest(
        upstream,
        load_pinned_grid_spec(ROOT),
        fixture=False,
        scientific=True,
        local_source_root=args.image_root,
        upstream_file_sha256=upstream_file_hash,
    )
    write_shared_manifest(manifest, args.output)
    print(f"shared manifest: {args.output}\nmanifest_hash: {manifest['manifest_hash']}")


if __name__ == "__main__":
    main()
