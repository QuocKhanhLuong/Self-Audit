#!/usr/bin/env python
"""Regenerate the image-free cardiac benchmark contract freeze.

The freeze is intentionally generated from explicit pre-freeze sources.  This
script never discovers MRI data and never opens reference annotations; real
ACDC/M&Ms manifests remain deferred until an image-only root is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPEC = REPO_ROOT / "configs" / "adapter_v1_spec_source.json"
SOURCE_FIXTURES = REPO_ROOT / "configs" / "adapter_v1_synthetic_fixtures_source.json"
DEFAULT_OUTPUT = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v1"
SHARED_GRID_SHA256 = "7c9d33fed0facbbabe736a5216bc599b65f1b465e3e26a3d19c624cf9be57949"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _assert_fixture_gt_free(value: Any, path: str = "fixture") -> None:
    forbidden = ("gt", "ground_truth", "mask", "label", "annotation", "dice", "iou", "hausdorff", "oracle", "hungarian")
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in forbidden):
                raise ValueError(f"forbidden GT-shaped fixture field: {path}.{key}")
            _assert_fixture_gt_free(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_fixture_gt_free(child, f"{path}[{index}]")


def _validate_fixture_source(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != "shared_benchmark.cardiac_adapter_fixture_set.v1":
        raise ValueError("unsupported adapter fixture schema")
    fixtures = value.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("fixture source must contain a non-empty list")
    ordered = sorted(fixtures, key=lambda item: str(item.get("id", "")))
    ids: list[str] = []
    allowed = {"B", "R", "M", "L", "V"}
    for fixture in ordered:
        if not isinstance(fixture, Mapping):
            raise ValueError("fixture entries must be objects")
        fixture_id = fixture.get("id")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise ValueError("fixture id must be a non-empty string")
        ids.append(fixture_id)
        raw = fixture.get("partition_grid")
        semantic = fixture.get("expected_semantic_grid")
        validity = fixture.get("expected_validity_grid")
        if not isinstance(raw, list) or not raw or not isinstance(semantic, list) or not isinstance(validity, list):
            raise ValueError(f"invalid fixture dimensions: {fixture_id}")
        if len(raw) != len(semantic) or len(raw) != len(validity):
            raise ValueError(f"invalid fixture dimensions: {fixture_id}")
        raw_ids = fixture.get("raw_id_by_symbol")
        if not isinstance(raw_ids, Mapping) or not raw_ids:
            raise ValueError(f"raw_id_by_symbol is required: {fixture_id}")
        for source_row, semantic_row, validity_row in zip(raw, semantic, validity):
            if not all(isinstance(row, str) for row in (source_row, semantic_row, validity_row)):
                raise ValueError(f"fixture rows must be strings: {fixture_id}")
            if not (len(source_row) == len(semantic_row) == len(validity_row)):
                raise ValueError(f"non-rectangular fixture: {fixture_id}")
            if not set(source_row).issubset(raw_ids):
                raise ValueError(f"undefined raw symbol: {fixture_id}")
            if not set(semantic_row).issubset(allowed) or not set(validity_row).issubset({"0", "1"}):
                raise ValueError(f"invalid expected encoding: {fixture_id}")
            for semantic_symbol, validity_symbol in zip(semantic_row, validity_row):
                expected = "0" if semantic_symbol == "V" else "1"
                if validity_symbol != expected:
                    raise ValueError(
                        f"semantic/validity invariant violated: {fixture_id} "
                        f"semantic={semantic_symbol!r} validity={validity_symbol!r}"
                    )
        _assert_fixture_gt_free(fixture, f"fixtures[{fixture_id}]")
    if len(set(ids)) != len(ids):
        raise ValueError("fixture ids must be unique")
    output = dict(value)
    output["fixtures"] = ordered
    return output


def _run(*args: str) -> str:
    return subprocess.run(args, cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _repo_binding(relative: str) -> dict[str, str]:
    path = REPO_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative.replace("\\", "/"), "sha256": _sha256_file(path)}


def _environment_receipt() -> dict[str, Any]:
    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:
        numpy_version = "unavailable"
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        pytorch_version = "unavailable"
        cuda_available = False
    return {
        "receipt_schema": "shared_benchmark.freeze_environment_receipt.v1",
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": pytorch_version,
        "numpy": numpy_version,
        "cuda_available": cuda_available,
        "gpu": "not bound into scientific payload",
        "note": "Execution evidence only; excluded from scientific payload hash.",
    }


def _grid_contract() -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from shared_benchmark.spatial import grid_hash, load_pinned_grid_spec

    grid = load_pinned_grid_spec(REPO_ROOT)
    actual = grid_hash(grid)
    if actual != SHARED_GRID_SHA256:
        raise ValueError(f"shared grid hash changed: {actual}")
    grid["shared_grid_sha256"] = actual
    grid["golden_fixture"] = {
        "fixture_version": "shared_benchmark.masked_resize_golden.v1",
        "input_dtype": "float32",
        "input_shape": [2, 3, 5],
        "input_values": "arange(30).reshape(2,3,5)",
        "target_hw": [4, 7],
        "output_dtype": "float32",
        "output_shape": [2, 4, 7],
        "output_little_endian_c_order_sha256": "7f697a8d3b3cd2c0ddb8c944cde5deaf5cf4a6e992fd328d79edcee56a84166c",
        "expected_values": [
            [[0.0, 0.5, 1.5, 2.0, 2.5, 3.5, 4.0], [2.5, 3.0, 4.0, 4.5, 5.0, 6.0, 6.5], [7.5, 8.0, 9.0, 9.5, 10.0, 11.0, 11.5], [10.0, 10.5, 11.5, 12.0, 12.5, 13.5, 14.0]],
            [[15.0, 15.5, 16.5, 17.0, 17.5, 18.5, 19.0], [17.5, 18.0, 19.0, 19.5, 20.0, 21.0, 21.5], [22.5, 23.0, 24.0, 24.5, 25.0, 26.0, 26.5], [25.0, 25.5, 26.5, 27.0, 27.5, 28.5, 29.0]],
        ],
    }
    return grid


def regenerate(output: Path) -> dict[str, Any]:
    if output.exists():
        if output.is_dir():
            leftovers = [child for child in output.rglob("*") if child.is_file()]
            if leftovers:
                raise FileExistsError(f"refusing to overwrite existing freeze: {output}")
            for child in sorted((child for child in output.rglob("*") if child.is_dir()), key=lambda path: len(path.parts), reverse=True):
                child.rmdir()
            output.rmdir()
        else:
            raise FileExistsError(f"refusing to overwrite existing freeze: {output}")
    spec = _read_json(SOURCE_SPEC)
    fixtures = _validate_fixture_source(_read_json(SOURCE_FIXTURES))
    if spec.get("adapter_version") != "cardiac_adapter_v1":
        raise ValueError("source adapter spec is not cardiac_adapter_v1")
    grid = _grid_contract()
    output.mkdir(parents=True)
    # Preserve the authoritative spec bytes exactly; JSON parsing above is only
    # for schema checks.  This avoids a formatting-only scientific change.
    (output / "configs").mkdir(parents=True, exist_ok=True)
    (output / "configs" / "adapter_v1_spec.json").write_bytes(SOURCE_SPEC.read_bytes())
    _write_json(output / "configs" / "adapter_v1_synthetic_fixtures.json", fixtures)
    _write_json(output / "configs" / "resolved_shared_contract.json", grid)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "data" / "README.md").write_text(
        "# Scientific dataset manifests\n\n"
        "ACDC and M&Ms image-only roots were unavailable during this pre-data "
        "contract freeze. No synthetic scientific manifest is included.\n",
        encoding="utf-8",
    )
    commit = _run("git", "rev-parse", "HEAD")
    branch = _run("git", "branch", "--show-current")
    origin = _run("git", "remote", "get-url", "origin")
    try:
        origin_main = _run("git", "rev-parse", "origin/main")
    except subprocess.CalledProcessError:
        origin_main = "unavailable"
    _write_json(output / "receipts" / "repository.json", {
        "receipt_schema": "shared_benchmark.freeze_repository_receipt.v1",
        "branch_at_freeze_attempt": branch,
        "repository_commit_sha": commit,
        "origin": origin,
        "origin_main_sha": origin_main,
        "worktree_clean_at_start": not bool(_run("git", "status", "--porcelain")),
        "note": "Execution metadata; excluded from scientific payload hash.",
    })
    _write_json(output / "receipts" / "environment.json", _environment_receipt())
    _write_json(output / "receipts" / "firewall.json", {
        "receipt_schema": "shared_benchmark.freeze_firewall_receipt.v1",
        "schema_firewall": "PASS_LOCAL",
        "physical_gt_isolation": "DEFERRED",
        "physical_gt_isolation_reason": "No separately mounted real image-only ACDC/M&Ms root was available.",
        "manifest_materialization": "DEFERRED_NO_REAL_IMAGE_ONLY_ROOT",
        "evidence": [
            "shared manifest schema rejects GT-shaped fields",
            "root-bounded image-only locator rejects traversal",
            "fixture scientific promotion is not permitted",
        ],
    })
    _write_json(output / "receipts" / "tests.json", {
        "receipt_schema": "shared_benchmark.freeze_test_receipt.v1",
        "status": "PENDING_VALIDATION",
        "commands": [],
        "real_manifest_validation": "DEFERRED_NO_REAL_IMAGE_ONLY_ROOT",
    })

    bound_repository_paths = [
        "configs/maskfree_acdc_150.yaml", "configs/maskfree_mnms_150.yaml",
        "configs/adapter_v1_spec_source.json", "configs/adapter_v1_synthetic_fixtures_source.json",
        "src/self_audit_maskfree/data/discovery.py", "src/self_audit_maskfree/data/dataset.py",
        "src/self_audit_maskfree/data/geometry.py", "src/self_audit_maskfree/data/firewall.py",
        "src/shared_benchmark/__init__.py", "src/shared_benchmark/manifest.py",
        "src/shared_benchmark/spatial.py", "src/shared_benchmark/firewall.py", "src/shared_benchmark/provenance.py",
        "scripts/prepare_maskfree_data.py", "scripts/prepare_shared_benchmark_manifest.py",
        "scripts/validate_cardiac_benchmark_freeze.py", "scripts/regenerate_cardiac_benchmark_freeze.py",
        "baseline/CUTS/src/cardiac_benchmark/manifest.py", "baseline/CUTS/src/cardiac_benchmark/dataset.py",
        "baseline/CUTS/src/cardiac_benchmark/train_stage1.py", "baseline/CUTS/src/cardiac_benchmark/export_latents.py",
        "baseline/CUTS/src/cardiac_benchmark/cluster_kmeans.py", "baseline/CUTS/src/cardiac_benchmark/provenance.py",
        "baseline/CUTS/scripts_cardiac/p0_local_smoke.py",
        "baseline/DFC/src/cardiac_benchmark/manifest.py", "baseline/DFC/src/cardiac_benchmark/dataset.py",
        "baseline/DFC/src/cardiac_benchmark/config.py", "baseline/DFC/src/cardiac_benchmark/dfc_runner.py",
        "baseline/DFC/config/cardiac/dfc_direct_2d_minl3.yaml",
    ]
    bound_repo = [_repo_binding(path) for path in bound_repository_paths]
    freeze_configs = [
        "configs/resolved_shared_contract.json",
        "configs/adapter_v1_spec.json",
        "configs/adapter_v1_synthetic_fixtures.json",
    ]
    bound_freeze = [{"path": path, "sha256": _sha256_file(output / path)} for path in freeze_configs]
    fixture_hash = _sha256_file(output / "configs" / "adapter_v1_synthetic_fixtures.json")
    spec_hash = _sha256_file(output / "configs" / "adapter_v1_spec.json")
    scientific_payload: dict[str, Any] = {
        "freeze_schema_version": "shared_benchmark.cardiac_benchmark_freeze.v1",
        "repository_commit_sha": commit,
        "authoritative_freemask_policy": {
            "source_manifest_schema": "maskfree150.data.v2",
            "split_policy_version": "maskfree150.data.discovery.v2",
            "split_seed": 42,
            "patient_level": True,
            "fallback_ratio": [0.7, 0.15, 0.15],
            "fallback_ranking": "blake2b(dataset, seed, patient_id)",
            "official_membership": "preserved before fallback for unassigned patients",
            "generation_inventory": "all acquired frames x all target z slices",
            "context_policy": "[max(z-1,0),z,min(z+1,Z-1)]",
            "annotation_derived_generation_eligibility": False,
        },
        "shared_manifest": {
            "schema_version": "shared_benchmark_manifest.v1",
            "sample_id": "dataset+patient_id+study_id+volume_id+frame_index+slice_index",
            "scientific_manifest_materialization": "DEFERRED_NO_REAL_IMAGE_ONLY_ROOT",
        },
        "datasets": {
            "acdc": {"scientific_manifest_status": "DEFERRED", "root_availability": False, "root_logical_identity": "configured_acdc_image_only_root_not_available", "image_only_firewall_result": "DEFERRED_NO_ROOT"},
            "mnms": {"scientific_manifest_status": "DEFERRED", "root_availability": False, "root_logical_identity": "configured_mnms_image_only_root_not_available", "image_only_firewall_result": "DEFERRED_NO_ROOT"},
        },
        "shared_spatial_contract": {
            "version": grid["version"], "target_hw": grid["target_hw"], "whole_fov": grid["whole_fov"],
            "resampling": grid["forward_values"], "grid_sha256": grid["shared_grid_sha256"],
        },
        "consumer_contract": {
            "cuts_primary": "CUTS-2D", "cuts_sensitivity": "CUTS-2.5D",
            "dfc_primary": "DFC-Direct-2D-Default-MinL3", "shared_inventory_required": True,
            "method_specific_normalization": ["FreeMask fit-only/masked", "CUTS cardiac preprocessing", "DFC central percentile/z-score"],
        },
        "firewall": {"schema_status": "PASS_LOCAL", "physical_isolation_status": "DEFERRED_NO_REAL_IMAGE_ONLY_MOUNT"},
        "adapter": {
            "adapter_version": "cardiac_adapter_v1", "input_schema_version": "shared_benchmark.anonymous_partition.v1",
            "specification_status": "FROZEN_PRE_IMPLEMENTATION", "specification_sha256": spec_hash,
            "synthetic_fixture_set_version": fixtures["fixture_set_version"], "synthetic_fixture_set_sha256": fixture_hash,
            "synthetic_fixture_invariant": "validity == (semantic != VOID)",
        },
        "scientific_readiness": "NOT_READY_REAL_DATA_AND_PHYSICAL_GT_ISOLATION_DEFERRED",
        "bound_repository_files": bound_repo,
        "bound_freeze_files": bound_freeze,
        "pre_freeze_sources": {
            "adapter_spec_source_sha256": _sha256_file(SOURCE_SPEC),
            "fixture_source_sha256": _sha256_file(SOURCE_FIXTURES),
            "fixture_generator": "scripts/regenerate_cardiac_benchmark_freeze.py",
        },
    }
    payload_hash = _sha256_bytes(_canonical_bytes(scientific_payload))
    manifest = {
        "freeze_schema": "shared_benchmark.cardiac_benchmark_freeze.v1",
        "freeze_id": f"cardiac-benchmark-v1-{payload_hash[:16]}",
        "scientific_payload_sha256": payload_hash,
        "scientific_payload": scientific_payload,
    }
    _write_json(output / "FREEZE_MANIFEST.json", manifest)
    (output / "FREEZE_REPORT.md").write_text(
        f"# Cardiac benchmark v1 freeze\n\n"
        f"- freeze ID: `{manifest['freeze_id']}`\n"
        f"- scientific payload SHA-256: `{payload_hash}`\n"
        f"- repository commit: `{commit}`\n"
        f"- adapter spec SHA-256: `{spec_hash}`\n"
        f"- synthetic fixture SHA-256: `{fixture_hash}`\n"
        f"- shared grid SHA-256: `{SHARED_GRID_SHA256}`\n\n"
        "This regenerated pre-data freeze is image-only by contract. ACDC and "
        "M&Ms scientific manifests and physical GT isolation remain deferred. "
        "The global fixture invariant is enforced before output.\n",
        encoding="utf-8",
    )
    return {"freeze_id": manifest["freeze_id"], "scientific_payload_sha256": payload_hash, "adapter_spec_sha256": spec_hash, "fixture_sha256": fixture_hash}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    print(json.dumps(regenerate(Path(args.output)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
