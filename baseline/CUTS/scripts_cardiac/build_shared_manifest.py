"""Materialize a shared scientific manifest on a real-data server only."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardiac_benchmark.manifest import FREEMASK_SOURCE_SHA, from_freemask_discovery, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freemask-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--dataset", choices=("acdc", "mnms"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    freemask_root = Path(args.freemask_root).resolve()
    observed_sha = subprocess.check_output(["git", "-C", str(freemask_root), "rev-parse", "HEAD"], text=True).strip()
    if observed_sha != FREEMASK_SOURCE_SHA:
        raise SystemExit(f"pinned FreeMask SHA required: {FREEMASK_SOURCE_SHA}, got {observed_sha}")
    sys.path.insert(0, str(freemask_root / "src"))
    from self_audit_maskfree.data.discovery import discover_dataset
    discovery = discover_dataset(args.image_root, args.dataset, seed=42, protocol="auto", depth_axis=2)
    payload = from_freemask_discovery(
        discovery, source_root=args.image_root, repo_root=freemask_root,
        fixture=False, scientific=True,
    )
    manifest_hash = write_manifest(payload, args.output)
    print(f"shared scientific manifest: {args.output}\nmanifest_hash: {manifest_hash}")


if __name__ == "__main__":
    main()
