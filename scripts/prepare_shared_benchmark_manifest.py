#!/usr/bin/env python
"""Materialize a shared image-only manifest from an explicit protocol.

ACDC defaults to the original Self-Audit receipt.  The historical FreeMask
projection remains available only via ``--protocol maskfree`` (and is not the
scientific CUTS/DFC default).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shared_benchmark.manifest import build_shared_manifest, write_shared_manifest
from shared_benchmark.provenance import sha256_file


def _assert_image_only_root(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if lowered == "info.cfg" or "_gt.nii" in lowered or lowered.endswith("_gt.nii.gz"):
            raise SystemExit(f"image-only root exposes forbidden file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a shared image-only benchmark manifest")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--dataset", required=True, choices=("acdc", "mnms"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--upstream-manifest", help="existing FreeMask manifest; otherwise run discovery")
    parser.add_argument(
        "--protocol", choices=("self_audit", "maskfree"),
        help="ACDC defaults to self_audit; choose maskfree only for historical compatibility.",
    )
    parser.add_argument("--selection", help="Self-Audit ED/ES selection receipt (required by --protocol self_audit)")
    args = parser.parse_args()
    protocol = args.protocol or ("self_audit" if args.dataset == "acdc" else "maskfree")
    if protocol == "self_audit":
        if args.dataset != "acdc":
            raise SystemExit("Self-Audit protocol is defined only for ACDC")
        if not args.selection:
            raise SystemExit("--selection is required for the ACDC Self-Audit protocol")
        from shared_benchmark.self_audit_protocol import discover_self_audit_acdc
        from shared_benchmark.spatial import load_self_audit_grid_spec

        _assert_image_only_root(Path(args.image_root))
        selection_path = Path(args.selection)
        upstream = discover_self_audit_acdc(args.image_root, selection_path, seed=42, depth_axis=2)
        grid = load_self_audit_grid_spec(ROOT)
        upstream_file_hash = sha256_file(selection_path)
    else:
        from self_audit_maskfree.data.discovery import discover_dataset
        from self_audit_maskfree.data.discovery import load_manifest
        from shared_benchmark.spatial import load_pinned_grid_spec

        if args.upstream_manifest:
            upstream_path = Path(args.upstream_manifest)
            upstream = load_manifest(upstream_path)
            upstream_file_hash = sha256_file(upstream_path)
        else:
            upstream = discover_dataset(args.image_root, args.dataset, seed=42, protocol="auto", depth_axis=2)
            upstream_file_hash = None
        grid = load_pinned_grid_spec(ROOT)
    if upstream["dataset"] != args.dataset:
        raise SystemExit("dataset argument does not match the selected source protocol manifest")
    manifest = build_shared_manifest(
        upstream,
        grid,
        fixture=False,
        scientific=True,
        local_source_root=args.image_root,
        upstream_file_sha256=upstream_file_hash,
    )
    write_shared_manifest(manifest, args.output)
    print(f"shared manifest: {args.output}\nmanifest_hash: {manifest['manifest_hash']}")


if __name__ == "__main__":
    main()
