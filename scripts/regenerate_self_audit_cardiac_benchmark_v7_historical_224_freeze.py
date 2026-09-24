#!/usr/bin/env python
"""Create a new baseline-only historical-224 freeze without touching v6.

This is a static re-projection of the already verified image-only v6 source
records.  It never discovers patients, opens NIfTI files, reads GT, trains,
or generates baseline outputs.  The only scientific change is the frozen
input-grid/preprocessing contract, so all v6 runtime artifacts are retained
but explicitly non-reusable for v7.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v6"
OUTPUT = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v7_historical_224"
SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v7_historical_224"
PREFIX = "cardiac-benchmark-v7-historical-224"
GENERATOR_PATH = "scripts/regenerate_self_audit_cardiac_benchmark_v7_historical_224_freeze.py"
VALIDATOR_PATH = "scripts/validate_cardiac_benchmark_v7_historical_224_freeze.py"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _binding(relative: str) -> dict[str, str]:
    path = REPO_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "sha256": _hash_file(path)}


def _counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    return {split: sum(row.get("split") == split for row in manifest.get("records", [])) for split in ("train", "dev", "test")}


def _regrid_verified_v6_manifest(parent: Mapping[str, Any], grid: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve source membership/identity while replacing only grid evidence."""
    import sys

    source = str(REPO_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from shared_benchmark.manifest import manifest_hash, validate_manifest
    from shared_benchmark.spatial import grid_hash, spatial_transform_for_record

    result = copy.deepcopy(dict(parent))
    result["shared_grid"] = copy.deepcopy(dict(grid))
    result["shared_grid_hash"] = grid_hash(grid)
    for record in result.get("records", []):
        record["shared_grid"] = copy.deepcopy(dict(grid))
        record["spatial_transform"] = spatial_transform_for_record(record, grid)
    result["manifest_hash"] = manifest_hash(result)
    validate_manifest(result)
    return result


def regenerate(output: Path = OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing freeze: {output}")
    parent_document = _read(PARENT / "FREEZE_MANIFEST.json")
    parent_payload = parent_document.get("scientific_payload")
    if (
        parent_document.get("freeze_schema") != "shared_benchmark.cardiac_benchmark_freeze.v6"
        or not isinstance(parent_payload, dict)
    ):
        raise ValueError("v7 historical-224 requires retained v6 static provenance")
    parent_manifest = _read(PARENT / "data" / "acdc_shared_manifest.json")
    if _counts(parent_manifest) != {"train": 1526, "dev": 376, "test": 0}:
        raise ValueError("v7 historical-224 requires the verified Self-Audit train/dev cohort")

    import sys

    source = str(REPO_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from shared_benchmark.spatial import grid_hash, load_self_audit_historical_224_grid_spec

    grid = load_self_audit_historical_224_grid_spec(REPO_ROOT)
    manifest = _regrid_verified_v6_manifest(parent_manifest, grid)
    counts = _counts(manifest)
    if counts != {"train": 1526, "dev": 376, "test": 0}:
        raise ValueError("historical-224 re-projection changed cohort membership")

    output.mkdir(parents=True)
    for directory in ("configs", "data", "receipts"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for relative in (
        "configs/adapter_v2_spec.json",
        "configs/adapter_v1_synthetic_fixtures.json",
        "data/acdc_selection_receipt.json",
        "data/acdc_patient_split_seed42.json",
    ):
        (output / relative).write_bytes((PARENT / relative).read_bytes())
    resolved_grid = dict(grid)
    resolved_grid["shared_grid_sha256"] = grid_hash(grid)
    _write(output / "configs" / "resolved_shared_contract.json", resolved_grid)
    _write(output / "data" / "acdc_shared_manifest.json", manifest)
    (output / "data" / "README.md").write_text(
        "# Baseline-only historical-224 ACDC manifest\n\n"
        "This v7 freeze preserves the verified v6 image-only source records, patient split, ED/ES selection, "
        "and all-Z inventory (train=1526, dev=376, test=0). It changes only the CUTS/DFC baseline input contract: "
        "the historical image preprocessing branch is reproduced at 224x224, followed by the loader volume "
        "normalization, with no 224-to-256 resize. No NIfTI/GT access or runtime algorithm workload occurred "
        "while creating this freeze. v6 checkpoints/raw/semantic artifacts remain retained but cannot be reused.\n",
        encoding="utf-8",
    )
    _write(output / "receipts" / "repository.json", {
        "receipt_schema": "shared_benchmark.freeze_repository_receipt.v1",
        "repository_commit_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "worktree_clean_at_start": False,
        "note": "Static baseline-only re-freeze; existing user changes and v6 runtime artifacts were preserved.",
    })
    _write(output / "receipts" / "firewall.json", {
        "receipt_schema": "shared_benchmark.freeze_firewall_receipt.v1",
        "schema_firewall": "PASS_STATIC",
        "data_access": "NONE; projected previously verified image-only source records",
        "gt_access": "NONE",
        "runtime_workloads": "NOT_RUN_BY_USER_PROTOCOL",
    })
    _write(output / "receipts" / "tests.json", {
        "receipt_schema": "shared_benchmark.freeze_test_receipt.v1",
        "status": "STATIC_VALIDATION_RECORDED_IN_SERVER_PREFLIGHT_STATE",
        "runtime_gates": "NOT_RUN",
    })

    frozen_files = [
        "configs/adapter_v2_spec.json",
        "configs/adapter_v1_synthetic_fixtures.json",
        "configs/resolved_shared_contract.json",
        "data/acdc_shared_manifest.json",
        "data/acdc_selection_receipt.json",
        "data/acdc_patient_split_seed42.json",
    ]
    bound = [
        "configs/adapter_v2_spec_source.json",
        "configs/adapter_v1_synthetic_fixtures_source.json",
        "scripts/preprocess_acdc.py",  # evidence only; never modified by this baseline change
        "src/self_audit/data/common.py",  # evidence only; never modified by this baseline change
        "src/shared_benchmark/manifest.py",
        "src/shared_benchmark/spatial.py",
        "src/shared_benchmark/semantic_contract.py",
        "src/shared_benchmark/artifacts.py",
        "src/shared_benchmark/adapter.py",
        "src/shared_benchmark/region_graph.py",
        "src/shared_benchmark/firewall.py",
        "src/shared_benchmark/provenance.py",
        "src/self_audit_maskfree/data/firewall.py",
        "scripts/run_cuts_scientific.py",
        "scripts/run_dfc_scientific.py",
        GENERATOR_PATH,
        VALIDATOR_PATH,
        "baseline/CUTS/src/cardiac_benchmark/manifest.py",
        "baseline/CUTS/src/cardiac_benchmark/dataset.py",
        "baseline/CUTS/src/cardiac_benchmark/train_stage1.py",
        "baseline/CUTS/src/cardiac_benchmark/export_latents.py",
        "baseline/CUTS/src/cardiac_benchmark/cluster_kmeans.py",
        "baseline/DFC/src/cardiac_benchmark/dataset.py",
        "baseline/DFC/src/cardiac_benchmark/config.py",
        "baseline/DFC/src/cardiac_benchmark/dfc_runner.py",
        "baseline/DFC/config/cardiac/dfc_direct_2d_minl3.yaml",
    ]
    adapter = copy.deepcopy(parent_payload["adapter"])
    adapter["central_image_contract"] = (
        "Self-Audit historical preprocess image branch: source 0.5/99.5 percentile clip + population z-score, "
        "per-slice skimage bilinear resize 224x224 (preserve_range=True, anti_aliasing=True, mode=reflect), "
        "then loader 0.5/99.5 percentile clip + population z-score; central plane at 224x224; no post-loader resize"
    )
    freeze_relative = f"benchmark_freezes/{output.name}"
    payload = {
        "freeze_schema_version": SCHEMA,
        "parent_freeze": {
            "freeze_id": parent_document["freeze_id"],
            "scientific_payload_sha256": parent_document["scientific_payload_sha256"],
            "status": "RETAINED_STATIC_PREDECESSOR_NOT_RUNTIME_REUSABLE",
        },
        "lineage": parent_payload["lineage"],
        "authoritative_self_audit_policy": parent_payload["authoritative_self_audit_policy"],
        "shared_manifest": {
            "manifest_sha256": manifest["manifest_hash"],
            "patient_counts": {"train": 80, "dev": 20, "test": 0},
            "volume_counts": {"train": 160, "dev": 40, "test": 0},
            "sample_counts": counts,
            "projection": "v6 verified image-only source records re-gridded without re-splitting or data access",
        },
        "shared_spatial_contract": {
            "version": grid["version"],
            "target_hw": grid["target_hw"],
            "whole_fov": grid["whole_fov"],
            "crop": grid["crop"],
            "resampling": grid["forward_values"],
            "grid_sha256": grid_hash(grid),
            "config_provenance": grid["config_provenance"],
        },
        "adapter": adapter,
        "semantic_lineage": {
            "prior_semantic_outputs": "SUPERSEDED_PENDING_REGENERATION_FROM_VERIFIED_RAW_IMAGES_WITH_V7",
            "v6_runtime_artifacts": "RETAINED_NOT_REUSABLE: input grid/preprocessing identity changed from direct-256 to historical-224",
            "checkpoint_reuse": "FORBIDDEN: v6 CUTS checkpoints bind the v6 manifest/grid and v6 input normalization",
            "raw_partition_reuse": "FORBIDDEN: v6 CUTS/DFC raw partitions were generated from a different central-image contract",
            "semantic_artifact_stage": "semantic-cardiac_adapter_v2-v7-historical-224",
            "overwrite_policy": "forbidden_by_distinct_stage",
        },
        "runner_binding": {
            "cuts_adapter_default": f"{freeze_relative}/configs/adapter_v2_spec.json",
            "dfc_adapter_default": f"{freeze_relative}/configs/adapter_v2_spec.json",
            "cuts_training_input_normalization": "self_audit.preprocess_acdc_224_then_loader_volume_percentile_clip_0p5_99p5_zscore.v1",
        },
        "datasets": {
            "acdc": {"scientific_manifest_status": "PASS_STATIC", "runtime_status": "NOT_RUN"},
            "mnms": {"scientific_manifest_status": "DEFERRED", "runtime_status": "NOT_RUN"},
        },
        "scientific_readiness": "CODE_AND_STATIC_PREFLIGHT_PASS_WAITING_FOR_USER_DECISION",
        "bound_repository_files": [_binding(path) for path in bound],
        "bound_freeze_files": [{"path": path, "sha256": _hash_file(output / path)} for path in frozen_files],
        "pre_freeze_sources": {
            "parent_v6_freeze_manifest_sha256": _hash_file(PARENT / "FREEZE_MANIFEST.json"),
            "parent_v6_manifest_sha256": _hash_file(PARENT / "data" / "acdc_shared_manifest.json"),
            "generator": GENERATOR_PATH,
        },
    }
    digest = _hash_bytes(_canonical(payload))
    document = {
        "freeze_schema": SCHEMA,
        "freeze_id": f"{PREFIX}-{digest[:16]}",
        "scientific_payload_sha256": digest,
        "scientific_payload": payload,
    }
    _write(output / "FREEZE_MANIFEST.json", document)
    (output / "FREEZE_REPORT.md").write_text(
        f"# Cardiac benchmark {output.name} historical-224 baseline freeze\n\n"
        f"- freeze ID: `{document['freeze_id']}`\n"
        "- ACDC samples: train=1526 / dev=376 / test=0\n"
        "- v6 runtime artifacts: retained but not reusable for v7\n"
        "- Runtime training, DFC, generation, PHATE/KMeans, and evaluation: NOT_RUN\n",
        encoding="utf-8",
    )
    return {"freeze_id": document["freeze_id"], "scientific_payload_sha256": digest, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(regenerate(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
