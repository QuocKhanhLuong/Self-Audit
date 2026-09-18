#!/usr/bin/env python
"""Run per-sample DFC, seal anonymous maps, and optionally run the adapter."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "baseline" / "DFC" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from shared_benchmark.artifacts import (  # noqa: E402
    ArtifactError,
    GeneratedSample,
    code_identity,
    config_hash,
    repository_identity,
    run_generation,
    select_manifest_records,
    validate_scientific_execution,
)
from shared_benchmark.provenance import sha256_file  # noqa: E402
from shared_benchmark.semantic_contract import FROZEN_ADAPTER_SPEC_SHA256  # noqa: E402
from cardiac_benchmark.config import load_primary_config  # noqa: E402
from cardiac_benchmark.dataset import load_primary_2d  # noqa: E402
from cardiac_benchmark.dfc_runner import run_dfc  # noqa: E402
from cardiac_benchmark.provenance import derive_sample_seed  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _adapter_spec(path: Path) -> dict[str, Any]:
    spec = _load_json(path)
    if spec.get("adapter_version") != "cardiac_adapter_v1" or sha256_file(path) != FROZEN_ADAPTER_SPEC_SHA256:
        raise ValueError("adapter spec hash is not the frozen cardiac_adapter_v1 contract")
    return spec


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = validate_scientific_execution(args.manifest, args.image_root)
    records = select_manifest_records(manifest, split=args.split, limit=args.limit, sample_list=args.sample_list)
    config = load_primary_config(args.config)
    if config.profile_id != "DFC-Direct-2D-Default-MinL3":
        raise ArtifactError("scientific DFC runner requires the frozen primary profile")
    repository = repository_identity(ROOT)
    baseline_code = code_identity([
        ROOT / "baseline" / "DFC" / "src" / "cardiac_benchmark" / "dataset.py",
        ROOT / "baseline" / "DFC" / "src" / "cardiac_benchmark" / "dfc_runner.py",
        ROOT / "baseline" / "DFC" / "src" / "cardiac_benchmark" / "config.py",
        ROOT / "baseline" / "DFC" / "src" / "cardiac_benchmark" / "provenance.py",
        ROOT / "src" / "shared_benchmark" / "artifacts.py",
        ROOT / "scripts" / "run_dfc_scientific.py",
    ], repo_root=ROOT)
    baseline_config_hash = args.config_hash or config_hash(args.config)

    def prepared(record: dict[str, Any]):
        tensor, metadata = load_primary_2d(record, args.image_root)
        return tensor, metadata

    def generate(record: dict[str, Any]) -> GeneratedSample:
        tensor, metadata = prepared(record)
        seed_info = derive_sample_seed(record["sample_id"])
        result = run_dfc(tensor.float(), config, seed_info["sample_seed"], device=args.device)
        provenance = {
            "sample_seed": seed_info["sample_seed"],
            "seed_derivation": seed_info,
            "update_count": result.update_count,
            "forward_count": result.forward_count,
            "stop_reason": result.stop_reason,
            "pre_stop_active_ids": result.pre_stop_active_ids,
            "pre_stop_active_count": result.pre_stop_active_count,
            "final_active_ids": result.final_active_ids,
            "final_active_count": result.final_active_count,
            "iteration_count": result.update_count,
            "minLabels": config.minLabels,
            "maxIter": config.maxIter,
            "optimizer": asdict(config),
            "input_mode": "central_slice_2d",
            "final_forward_semantics": config.final_forward_policy,
            "fresh_model_optimizer_bn_state_per_sample": True,
            "loader_metadata": metadata,
        }
        return GeneratedSample(partition=np.asarray(result.raw_cluster_map, dtype=np.int32), central_image=np.asarray(tensor[0, 0]), baseline_metadata=provenance)

    adapter_spec = None
    semantic_root = None
    if args.apply_adapter:
        adapter_spec = _adapter_spec(args.adapter_spec)
        semantic_root = args.semantic_root or Path(args.output_root)
    return run_generation(
        records,
        output_root=args.output_root,
        baseline_name="DFC",
        baseline_mode="DFC-Direct-2D-Default-MinL3",
        manifest_hash=str(manifest["manifest_hash"]),
        shared_grid_hash=str(manifest["shared_grid_hash"]),
        repository=repository,
        code_identity_value=baseline_code,
        baseline_config_hash=baseline_config_hash,
        seed_for_record=lambda record: int(derive_sample_seed(record["sample_id"])["sample_seed"]),
        generate=generate,
        semantic_root=semantic_root,
        adapter_spec=adapter_spec,
        central_image_for_record=lambda record: np.asarray(prepared(record)[0][0, 0]),
        retry_failed=args.retry_failed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "baseline" / "DFC" / "config" / "cardiac" / "dfc_direct_2d_minl3.yaml")
    parser.add_argument("--config-hash")
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-list", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--apply-adapter", action="store_true")
    parser.add_argument("--adapter-spec", type=Path, default=ROOT / "benchmark_freezes" / "cardiac_benchmark_v1" / "configs" / "adapter_v1_spec.json")
    parser.add_argument("--semantic-root", type=Path)
    parser.add_argument("--no-retry-failed", dest="retry_failed", action="store_false")
    parser.set_defaults(retry_failed=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.sample_list = args.sample_list.read_text(encoding="utf-8").splitlines() if args.sample_list else None
    results = run(args)
    print(json.dumps({"baseline": "DFC", "mode": "DFC-Direct-2D-Default-MinL3", "results": [{key: value for key, value in row.items() if key in {"sample_id", "raw_status", "semantic_status", "error"}} for row in results]}, sort_keys=True))
    return 0 if all(row.get("raw_status") != "FAILED" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
