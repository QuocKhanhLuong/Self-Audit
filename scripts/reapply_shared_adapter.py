#!/usr/bin/env python
"""Apply a versioned shared adapter to verified raw artifacts only.

This command deliberately does not call ``run_generation``.  It is for a new
adapter version over already sealed raw partitions, so a changed runner/code
identity can never regenerate a baseline partition while reapplying semantics.
Only the declared image-only source root is read to reconstruct the same
central image bound by the selected producer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shared_benchmark.artifacts import (  # noqa: E402
    ArtifactError,
    artifact_directory,
    run_adapter_after_raw,
    select_manifest_records,
    validate_scientific_execution,
    verify_raw_partition,
)
from shared_benchmark.semantic_contract import load_and_validate_spec  # noqa: E402


def _load_spec(path: Path) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"adapter spec is invalid: {path}") from exc
    try:
        return load_and_validate_spec(value)
    except ValueError as exc:
        raise ArtifactError(str(exc)) from exc


def _central_image_loader(args: argparse.Namespace, manifest: Mapping[str, Any]) -> tuple[str, str, Callable[[Mapping[str, Any]], np.ndarray]]:
    if args.producer == "cuts":
        cuts_root = ROOT / "baseline" / "CUTS" / "src"
        if str(cuts_root) not in sys.path:
            sys.path.insert(0, str(cuts_root))
        from cardiac_benchmark.dataset import ImageOnlyCardiacDataset

        profile = "CUTS-2D" if args.mode == "2d" else "CUTS-2.5D"
        dataset = ImageOnlyCardiacDataset(manifest, split=args.split, profile=profile, source_root=args.image_root)
        positions = {record["sample_id"]: index for index, record in enumerate(dataset.records)}

        def central(record: Mapping[str, Any]) -> np.ndarray:
            try:
                sample = dataset[positions[str(record["sample_id"])]]
            except KeyError as exc:
                raise ArtifactError(f"CUTS dataset lacks selected sample: {record['sample_id']}") from exc
            return np.asarray(sample.image[0 if profile == "CUTS-2D" else 1])

        return "CUTS", profile, central

    dfc_root = ROOT / "baseline" / "DFC" / "src"
    if str(dfc_root) not in sys.path:
        sys.path.insert(0, str(dfc_root))
    from cardiac_benchmark.dataset import load_primary_2d

    def central(record: Mapping[str, Any]) -> np.ndarray:
        tensor, _ = load_primary_2d(record, args.image_root)
        return np.asarray(tensor[0, 0])

    return "DFC", "DFC-Direct-2D-Default-MinL3", central


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", choices=("cuts", "dfc"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--adapter-spec", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--mode", choices=("2d", "2.5d"), default="2d", help="CUTS profile; ignored for DFC")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-list", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sample_list = args.sample_list.read_text(encoding="utf-8").splitlines() if args.sample_list else None
    manifest = validate_scientific_execution(args.manifest, args.image_root)
    records = select_manifest_records(manifest, split=args.split, limit=args.limit, sample_list=sample_list)
    spec, spec_hash = _load_spec(args.adapter_spec)
    adapter_version = str(spec["adapter_version"])
    baseline_name, baseline_mode, central_image = _central_image_loader(args, manifest)

    raw_by_id = {}
    for record in records:
        raw_directory = artifact_directory(
            args.raw_root, baseline_name=baseline_name, baseline_mode=baseline_mode,
            sample_id=str(record["sample_id"]),
        )
        try:
            raw_by_id[str(record["sample_id"])] = verify_raw_partition(raw_directory)
        except ArtifactError as exc:
            raise ArtifactError(f"verified raw artifact required for {record['sample_id']}: {exc}") from exc

    completed: list[dict[str, str]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        semantic = run_adapter_after_raw(
            raw_by_id[sample_id], semantic_root=args.semantic_root, record=record,
            central_image=central_image(record), adapter_spec=spec,
            baseline_name=baseline_name, baseline_mode=baseline_mode,
            adapter_version=adapter_version, adapter_spec_sha256=spec_hash,
        )
        completed.append({"sample_id": sample_id, "semantic_directory": str(semantic.directory)})
    print(json.dumps({
        "adapter_version": adapter_version, "adapter_spec_sha256": spec_hash,
        "baseline_name": baseline_name, "baseline_mode": baseline_mode,
        "sample_count": len(completed), "semantic_root": str(args.semantic_root),
        "results": completed,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
