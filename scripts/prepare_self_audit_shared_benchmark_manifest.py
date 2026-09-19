#!/usr/bin/env python
"""Project the original Self-Audit ACDC protocol into the shared manifest.

The command is designed for a bwrap namespace that exposes only the selected
image-only tree, this script's code, the Python environment, and the mounted
selection receipt.  It never discovers patients, reads ``Info.cfg``, or
accesses annotations; membership is already fixed by the receipt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shared_benchmark.manifest import (  # noqa: E402
    build_shared_manifest,
    validate_scientific_manifest,
    write_shared_manifest,
)
from shared_benchmark.provenance import sha256_file  # noqa: E402
from shared_benchmark.self_audit_protocol import discover_self_audit_acdc  # noqa: E402
from shared_benchmark.spatial import grid_hash, load_self_audit_grid_spec  # noqa: E402


def _assert_image_only_root(root: Path) -> None:
    if not root.is_dir():
        raise ValueError(f"image-only root is unavailable: {root}")
    forbidden: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if lowered == "info.cfg" or "_gt.nii" in lowered or lowered.endswith("_gt.nii.gz"):
            forbidden.append(str(path))
    if forbidden:
        raise ValueError("image-only root exposes annotation/selection metadata: " + ", ".join(forbidden[:5]))


def prepare(image_root: Path, selection: Path, output: Path) -> dict[str, Any]:
    _assert_image_only_root(image_root)
    upstream = discover_self_audit_acdc(image_root, selection, seed=42, depth_axis=2)
    grid = load_self_audit_grid_spec(ROOT)
    manifest = build_shared_manifest(
        upstream,
        grid,
        fixture=False,
        scientific=True,
        local_source_root=image_root,
        upstream_file_sha256=sha256_file(selection),
    )
    write_shared_manifest(manifest, output)
    # Hash verification is intentionally done while the image-only namespace
    # is still mounted.  No algorithmic workload is invoked.
    validate_scientific_manifest(manifest, image_root=image_root, verify_source_hashes=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.image_root, args.selection, args.output)
    counts = {split: sum(record["split"] == split for record in manifest["records"]) for split in ("train", "dev", "test")}
    patients = {}
    for split in ("train", "dev", "test"):
        patients[split] = len({record["patient_id"] for record in manifest["records"] if record["split"] == split})
    print(json.dumps({
        "manifest": str(args.output),
        "manifest_hash": manifest["manifest_hash"],
        "grid_hash": grid_hash(manifest["shared_grid"]),
        "records": counts,
        "patients": patients,
        "source_manifest": manifest["source_manifest"]["schema_version"],
        "source_root": os.path.realpath(args.image_root),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
