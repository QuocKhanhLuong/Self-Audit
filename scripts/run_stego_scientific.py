#!/usr/bin/env python
"""Run STEGO raw partition generation and optional shared semantic handoff."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "baseline" / "STEGO" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from shared_benchmark.artifacts import (  # noqa: E402
    ArtifactError,
    code_identity,
    repository_identity,
    run_generation,
    select_manifest_records,
    validate_scientific_execution,
)
from shared_benchmark.checkpoint_contract import (  # noqa: E402
    CheckpointContractError,
    load_checkpoint_contract,
    validate_checkpoint_contract,
)
from shared_benchmark.provenance import sha256_file  # noqa: E402
from shared_benchmark.semantic_contract import (  # noqa: E402
    FROZEN_ADAPTER_SPEC_SHA256,
    FROZEN_ADAPTER_V3_SPEC_SHA256,
    FROZEN_SHARED_GRID_SHA256,
)
from shared_benchmark.spatial import (  # noqa: E402
    SPATIAL_CONTRACT_VERSION,
    grid_hash,
    load_self_audit_compat_224_grid_spec,
)
from cardiac_benchmark.config import STEGOConfig  # noqa: E402
from cardiac_benchmark.dataset import SA224_NORMALIZATION_VERSION, STEGOCardiacDataset  # noqa: E402
from cardiac_benchmark.stego_runner import generate_sample, load_stego_model, seed_deterministic  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _adapter_spec(path: Path) -> dict[str, Any]:
    spec = _load_json(path)
    version = spec.get("adapter_version")
    digest = sha256_file(path)
    if version == "cardiac_adapter_v2" and digest == FROZEN_ADAPTER_SPEC_SHA256:
        return spec
    if version == "cardiac_adapter_v3" and digest == FROZEN_ADAPTER_V3_SPEC_SHA256:
        return spec
    raise ValueError("adapter spec hash is not a supported frozen cardiac adapter contract")


def _is_sa224_profile(profile: str) -> bool:
    return profile in {"STEGO-SA224", "STEGO-SA224-FAIR"}


def _is_fair_profile(profile: str) -> bool:
    return profile == "STEGO-SA224-FAIR"


def _default_adapter_spec(profile: str) -> Path:
    if _is_sa224_profile(profile):
        return ROOT / "benchmark_freezes" / "cardiac_benchmark_v7" / "configs" / "adapter_v3_spec.json"
    return ROOT / "benchmark_freezes" / "cardiac_benchmark_v1" / "configs" / "adapter_v1_spec.json"


def _require_frozen_grid(manifest: dict[str, Any]) -> None:
    grid = manifest.get("shared_grid")
    if (
        manifest.get("schema_version") != "shared_benchmark_manifest.v1"
        or not isinstance(grid, dict)
        or grid.get("version") != SPATIAL_CONTRACT_VERSION
        or grid.get("target_hw") != [224, 224]
        or grid.get("whole_fov") is not True
        or manifest.get("shared_grid_hash") != FROZEN_SHARED_GRID_SHA256
    ):
        raise ArtifactError("STEGO scientific runner requires the frozen 224x224 whole-FOV shared grid")


def _require_sa224_grid(manifest: dict[str, Any]) -> None:
    grid = manifest.get("shared_grid")
    expected = load_self_audit_compat_224_grid_spec(ROOT)
    if (
        manifest.get("schema_version") != "shared_benchmark_manifest.v1"
        or manifest.get("split_policy_version") != "self_audit.acdc.patient_split.v1"
        or not isinstance(grid, dict)
        or grid != expected
        or manifest.get("shared_grid_hash") != grid_hash(expected)
    ):
        raise ArtifactError("STEGO-SA224 requires the Self-Audit compat 224 shared grid")


def _require_manifest_contract(manifest: dict[str, Any], profile: str) -> None:
    if _is_sa224_profile(profile):
        _require_sa224_grid(manifest)
        return
    _require_frozen_grid(manifest)


def _require_profile_semantic_handoff(profile: str, apply_adapter: bool) -> None:
    if apply_adapter and not _is_sa224_profile(profile):
        raise ArtifactError("shared semantic handoff is published only for STEGO SA224 profiles")


def _require_fair_checkpoint_contract(
    profile: str,
    contract_path: Path | None,
    *,
    checkpoint_sha256: str,
    manifest: dict[str, Any],
) -> None:
    if not _is_fair_profile(profile):
        return
    if contract_path is None:
        raise ArtifactError("STEGO-SA224-FAIR requires --checkpoint-contract")
    try:
        contract = load_checkpoint_contract(contract_path)
        validate_checkpoint_contract(
            contract,
            baseline_name="STEGO",
            baseline_mode=profile,
            checkpoint_sha256=checkpoint_sha256,
            dataset=str(manifest["dataset"]),
            split_policy_version=str(manifest["split_policy_version"]),
            shared_grid_hash=str(manifest["shared_grid_hash"]),
            normalization_version=SA224_NORMALIZATION_VERSION,
        )
    except CheckpointContractError as exc:
        raise ArtifactError(str(exc)) from exc


def _require_expected_config_hash(expected: str | None, computed: str) -> None:
    if expected is not None and str(expected) != computed:
        raise ArtifactError("STEGO --config-hash does not match the effective scientific config")


def _execution_environment(device: str) -> dict[str, Any]:
    runtime_device = torch.device(device)
    cuda_active = runtime_device.type == "cuda" and torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_device": str(device),
        "device_name": torch.cuda.get_device_name(runtime_device) if cuda_active else None,
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = validate_scientific_execution(args.manifest, args.image_root)
    _require_manifest_contract(manifest, args.profile)
    _require_profile_semantic_handoff(args.profile, args.apply_adapter)
    records = select_manifest_records(manifest, split=args.split, limit=args.limit, sample_list=args.sample_list)
    config = STEGOConfig(
        profile=args.profile,
        dataset=str(manifest["dataset"]),
        manifest_path=str(args.manifest),
        checkpoint_path=str(args.checkpoint),
        scientific_run=True,
    )
    config.validate()
    baseline_config_hash = config.config_hash()
    _require_expected_config_hash(args.config_hash, baseline_config_hash)
    checkpoint_hash = sha256_file(args.checkpoint)
    _require_fair_checkpoint_contract(args.profile, args.checkpoint_contract, checkpoint_sha256=checkpoint_hash, manifest=manifest)
    repository = repository_identity(ROOT)
    baseline_code = code_identity([
        ROOT / "baseline" / "STEGO" / "src" / "modules.py",
        ROOT / "baseline" / "STEGO" / "src" / "utils.py",
        ROOT / "baseline" / "STEGO" / "src" / "dino" / "vision_transformer.py",
        ROOT / "baseline" / "STEGO" / "src" / "cardiac_benchmark" / "dataset.py",
        ROOT / "baseline" / "STEGO" / "src" / "cardiac_benchmark" / "stego_runner.py",
        ROOT / "baseline" / "STEGO" / "src" / "cardiac_benchmark" / "config.py",
        ROOT / "src" / "shared_benchmark" / "artifacts.py",
        ROOT / "scripts" / "run_stego_scientific.py",
    ], repo_root=ROOT)
    seed_deterministic(config.benchmark_seed)
    model = load_stego_model(config, device=args.device)
    dataset = STEGOCardiacDataset(manifest, split=args.split, profile=args.profile, source_root=args.image_root)
    by_id = {record["sample_id"]: index for index, record in enumerate(dataset.records)}
    missing = [record["sample_id"] for record in records if record["sample_id"] not in by_id]
    if missing:
        raise ArtifactError(f"STEGO dataset inventory mismatch: {missing}")

    def prepared(record: dict[str, Any]):
        return dataset[by_id[record["sample_id"]]]

    environment = _execution_environment(args.device)

    def generate(record: dict[str, Any]):
        generated = generate_sample(model, prepared(record), config=config, device=args.device, checkpoint_sha256=checkpoint_hash)
        return type(generated)(
            partition=generated.partition,
            central_image=generated.central_image,
            baseline_metadata=generated.baseline_metadata,
            execution_receipt=environment,
        )

    adapter_spec = None
    semantic_root = None
    if args.apply_adapter:
        adapter_spec = _adapter_spec(args.adapter_spec or _default_adapter_spec(args.profile))
        semantic_root = args.semantic_root or Path(args.output_root)
    return run_generation(
        records,
        output_root=args.output_root,
        baseline_name="STEGO",
        baseline_mode=args.profile,
        manifest_hash=str(manifest["manifest_hash"]),
        shared_grid_hash=str(manifest["shared_grid_hash"]),
        repository=repository,
        code_identity_value=baseline_code,
        baseline_config_hash=baseline_config_hash,
        seed_for_record=lambda _record: config.benchmark_seed,
        generate=generate,
        semantic_root=semantic_root,
        adapter_spec=adapter_spec,
        central_image_for_record=lambda record: np.asarray(prepared(record).image[0]),
        retry_failed=args.retry_failed,
        extra_raw_identity_for_record=lambda _record: {"checkpoint_sha256": checkpoint_hash},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--profile", default="STEGO-2D")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-list", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config-hash")
    parser.add_argument("--checkpoint-contract", type=Path)
    parser.add_argument("--apply-adapter", action="store_true")
    parser.add_argument("--adapter-spec", type=Path)
    parser.add_argument("--semantic-root", type=Path)
    parser.add_argument("--no-retry-failed", dest="retry_failed", action="store_false")
    parser.set_defaults(retry_failed=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.sample_list = args.sample_list.read_text(encoding="utf-8").splitlines() if args.sample_list else None
    results = run(args)
    print(json.dumps({"baseline": "STEGO", "mode": args.profile, "results": [{key: value for key, value in row.items() if key in {"sample_id", "raw_status", "semantic_status", "error"}} for row in results]}, sort_keys=True))
    return 0 if all(row.get("raw_status") != "FAILED" and row.get("semantic_status") != "FAILED" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
