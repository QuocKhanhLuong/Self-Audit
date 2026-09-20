#!/usr/bin/env python
"""Run CUTS inference, seal anonymous partitions, and optionally hand off v2.

This runner owns orchestration only.  CUTS preprocessing, encoder inference,
PHATE, and K-means remain the existing baseline implementations.  The runner
never discovers annotations and never opens a reference mask.
"""
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
for _path in (ROOT / "src", ROOT / "baseline" / "CUTS" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from shared_benchmark.artifacts import (  # noqa: E402
    ArtifactError,
    GeneratedSample,
    code_identity,
    repository_identity,
    run_generation,
    select_manifest_records,
    validate_scientific_execution,
)
from shared_benchmark.provenance import sha256_file  # noqa: E402
from shared_benchmark.semantic_contract import FROZEN_ADAPTER_SPEC_SHA256  # noqa: E402
from shared_benchmark.semantic_contract import FROZEN_SELF_AUDIT_SHARED_GRID_SHA256  # noqa: E402
from shared_benchmark.spatial import SELF_AUDIT_SPATIAL_CONTRACT_VERSION  # noqa: E402
from cardiac_benchmark.cluster_kmeans import cluster_latent  # noqa: E402
from cardiac_benchmark.dataset import ImageOnlyCardiacDataset  # noqa: E402
from cardiac_benchmark.manifest import load_manifest  # noqa: E402
from cardiac_benchmark.train_stage1 import Stage1Config, load_checkpoint_for_export, scientific_config_hash  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _adapter_spec(path: Path) -> dict[str, Any]:
    spec = _load_json(path)
    if spec.get("adapter_version") != "cardiac_adapter_v2":
        raise ValueError("unsupported adapter specification")
    if sha256_file(path) != FROZEN_ADAPTER_SPEC_SHA256:
        raise ValueError("adapter spec hash is not the frozen cardiac_adapter_v2 contract")
    return spec


def _require_frozen_grid(manifest: dict[str, Any]) -> None:
    grid = manifest.get("shared_grid")
    if (
        manifest.get("schema_version") != "shared_benchmark_manifest.v1"
        or not isinstance(grid, dict)
        or manifest.get("split_policy_version") != "self_audit.acdc.patient_split.v1"
        or grid.get("version") != SELF_AUDIT_SPATIAL_CONTRACT_VERSION
        or grid.get("target_hw") != [256, 256]
        or grid.get("whole_fov") is not True
        or grid.get("crop") is not None
        or grid.get("forward_values") != "bilinear_align_corners_false"
        or manifest.get("shared_grid_hash") != FROZEN_SELF_AUDIT_SHARED_GRID_SHA256
    ):
        raise ArtifactError("CUTS scientific runner requires the frozen Self-Audit 256x256 whole-FOV shared grid")


def _require_expected_config_hash(expected: str | None, computed: str) -> None:
    if expected is not None and str(expected) != computed:
        raise ArtifactError("CUTS --config-hash does not match the effective scientific config")


def _central_image(sample: Any, profile: str) -> np.ndarray:
    return np.asarray(sample.image[0 if profile == "CUTS-2D" else 1])


def _start_execution_receipt(device: str) -> dict[str, Any]:
    runtime_device = torch.device(device)
    cuda_active = runtime_device.type == "cuda" and torch.cuda.is_available()
    receipt: dict[str, Any] = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_device": str(device),
        "device_name": torch.cuda.get_device_name(runtime_device) if cuda_active else None,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    try:
        import phate
        receipt["phate_version"] = phate.__version__
    except Exception:
        receipt["phate_version"] = None
    try:
        import sklearn
        receipt["scikit_learn_version"] = sklearn.__version__
    except Exception:
        receipt["scikit_learn_version"] = None
    if cuda_active:
        torch.cuda.reset_peak_memory_stats(runtime_device)
    return receipt


def _finish_execution_receipt(receipt: dict[str, Any], device: str) -> dict[str, Any]:
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and torch.cuda.is_available():
        receipt["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(runtime_device))
        receipt["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(runtime_device))
    return receipt


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = validate_scientific_execution(args.manifest, args.image_root)
    _require_frozen_grid(manifest)
    # Re-load through the baseline compatibility surface as a contract check;
    # it delegates to the same shared schema and does not split or translate.
    baseline_manifest = load_manifest(args.manifest)
    if baseline_manifest["manifest_hash"] != manifest["manifest_hash"]:
        raise ArtifactError("CUTS/shared manifest identity mismatch")
    profile = "CUTS-2D" if args.mode == "2d" else "CUTS-2.5D"
    records = select_manifest_records(manifest, split=args.split, limit=args.limit, sample_list=args.sample_list)
    config = Stage1Config(
        profile=profile,
        dataset=str(manifest["dataset"]),
        manifest_path=str(args.manifest),
        scientific_run=True,
        image_root=str(args.image_root),
    )
    computed_config_hash = scientific_config_hash(config)
    _require_expected_config_hash(args.config_hash, computed_config_hash)
    model = load_checkpoint_for_export(config, args.checkpoint, device=args.device)
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_payload.get("config_hash") != computed_config_hash:
        raise ArtifactError("CUTS checkpoint config hash does not match the effective scientific config")
    if checkpoint_payload.get("checkpoint_selection_policy") != "final_epoch":
        raise ArtifactError("CUTS scientific checkpoint must declare final_epoch selection")
    checkpoint_config_hash = computed_config_hash
    repository = repository_identity(ROOT)
    baseline_code = code_identity([
        ROOT / "baseline" / "CUTS" / "src" / "cardiac_benchmark" / "dataset.py",
        ROOT / "baseline" / "CUTS" / "src" / "cardiac_benchmark" / "train_stage1.py",
        ROOT / "baseline" / "CUTS" / "src" / "cardiac_benchmark" / "cluster_kmeans.py",
        ROOT / "baseline" / "CUTS" / "src" / "cardiac_benchmark" / "export_latents.py",
        ROOT / "src" / "shared_benchmark" / "artifacts.py",
        ROOT / "scripts" / "run_cuts_scientific.py",
    ], repo_root=ROOT)
    model.eval()
    dataset = ImageOnlyCardiacDataset(manifest, split=args.split, profile=profile, source_root=args.image_root)
    by_id = {record["sample_id"]: index for index, record in enumerate(dataset.records)}
    missing = [record["sample_id"] for record in records if record["sample_id"] not in by_id]
    if missing:
        raise ArtifactError(f"CUTS dataset inventory mismatch: {missing}")

    def prepared(record: dict[str, Any]):
        return dataset[by_id[record["sample_id"]]]

    def generate(record: dict[str, Any]) -> GeneratedSample:
        execution_receipt = _start_execution_receipt(args.device)
        sample = prepared(record)
        with torch.no_grad():
            latent = model(sample.image.unsqueeze(0).float().to(args.device)).detach().cpu().numpy()[0].astype(np.float32, copy=False)
        clustered = cluster_latent(latent, num_workers=args.num_workers)
        if clustered.get("status") != "success":
            raise RuntimeError(f"CUTS clustering failed for {record['sample_id']}: {clustered.get('exceptions', [])}")
        raw = np.asarray(clustered["raw_cluster_map"], dtype=np.int64)
        payload_keys = ("epoch", "dev_metrics", "source_cuts_sha", "source_manifest_logical_sha", "shared_grid_hash", "cuts_mode", "benchmark_seed", "checkpoint_selection_policy", "scientific_config")
        training = {key: checkpoint_payload[key] for key in payload_keys if key in checkpoint_payload}
        provenance = {
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_identity": checkpoint_hash,
            "checkpoint_training_provenance": training,
            "cuts_mode": profile,
            "input_normalization": sample.provenance["normalization"],
            "input_channels": sample.provenance["input_channels"],
            "phate_configuration": clustered.get("phate_configuration", {"n_components": 3, "knn": 100, "n_landmark": 500, "t": 2}),
            "kmeans_configuration": clustered.get("kmeans_configuration", {"n_clusters": 10}),
            "primary_k": 10,
            "clustering_seed": clustered.get("actual_clustering_seed_used"),
            "clustering_retry_occurred": bool(clustered.get("retry_occurred", False)),
            "latent_sha256": clustered.get("latent_hash"),
            "raw_partition_sha256": clustered.get("raw_partition_hash"),
        }
        return GeneratedSample(partition=raw, central_image=_central_image(sample, profile), baseline_metadata=provenance,
                               execution_receipt=_finish_execution_receipt(execution_receipt, args.device))

    adapter_spec = None
    semantic_root = None
    if args.apply_adapter:
        adapter_spec = _adapter_spec(args.adapter_spec)
        semantic_root = args.semantic_root or Path(args.output_root)
    return run_generation(
        records,
        output_root=args.output_root,
        baseline_name="CUTS",
        baseline_mode=profile,
        manifest_hash=str(manifest["manifest_hash"]),
        shared_grid_hash=str(manifest["shared_grid_hash"]),
        repository=repository,
        code_identity_value=baseline_code,
        baseline_config_hash=checkpoint_config_hash,
        seed_for_record=lambda _record: 42,
        generate=generate,
        semantic_root=semantic_root,
        adapter_spec=adapter_spec,
        central_image_for_record=lambda record: _central_image(prepared(record), profile),
        adapter_implementation_sha256=None,
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
    parser.add_argument("--mode", choices=("2d", "2.5d"), default="2d")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-list", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--config-hash")
    parser.add_argument("--apply-adapter", action="store_true")
    parser.add_argument("--adapter-spec", type=Path, default=ROOT / "benchmark_freezes" / "cardiac_benchmark_v6" / "configs" / "adapter_v2_spec.json")
    parser.add_argument("--semantic-root", type=Path)
    parser.add_argument("--no-retry-failed", dest="retry_failed", action="store_false")
    parser.set_defaults(retry_failed=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.sample_list = args.sample_list.read_text(encoding="utf-8").splitlines() if args.sample_list else None
    results = run(args)
    print(json.dumps({"baseline": "CUTS", "mode": "CUTS-2D" if args.mode == "2d" else "CUTS-2.5D", "results": [{key: value for key, value in row.items() if key in {"sample_id", "raw_status", "semantic_status", "error"}} for row in results]}, sort_keys=True))
    return 0 if all(row.get("raw_status") != "FAILED" and row.get("semantic_status") != "FAILED" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
