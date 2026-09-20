#!/usr/bin/env python
"""Generate the non-overwriting Self-Audit ACDC adapter-v2/v4 freeze.

This is a static provenance operation.  It copies the already frozen v3
image-only manifest and selection receipt; it never opens ACDC images or GT
and never invokes a baseline algorithm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_FREEZE = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v3"
SOURCE_SPEC = REPO_ROOT / "configs" / "adapter_v2_spec_source.json"
SOURCE_FIXTURES = REPO_ROOT / "configs" / "adapter_v1_synthetic_fixtures_source.json"
FREEZE_SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v4"
FREEZE_ID_PREFIX = "cardiac-benchmark-v4"
DEFAULT_OUTPUT = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v4"


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
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _run(*args: str) -> str:
    return subprocess.run(args, cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _repo_binding(relative: str) -> dict[str, str]:
    path = REPO_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "sha256": _sha256_file(path)}


def _grid_contract() -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from shared_benchmark.spatial import grid_hash, load_self_audit_grid_spec

    grid = load_self_audit_grid_spec(REPO_ROOT)
    grid["shared_grid_sha256"] = grid_hash(grid)
    return grid


def _counts(manifest: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    rows = list(manifest.get("records", []))
    for split in ("train", "dev", "test"):
        selected = [row for row in rows if row.get("split") == split]
        result[split] = {
            "patients": len({str(row.get("patient_id")) for row in selected}),
            "volumes": len({str(row.get("volume_id")) for row in selected}),
            "samples": len(selected),
        }
    return result


def _environment_receipt() -> dict[str, Any]:
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        torch_version, cuda_available = "unavailable", False
    return {
        "receipt_schema": "shared_benchmark.freeze_environment_receipt.v1",
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch_version,
        "cuda_available": cuda_available,
        "note": "Static provenance only; no ACDC image or algorithm workload was run.",
    }


def regenerate(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing freeze: {output}")
    parent_manifest = _read_json(PARENT_FREEZE / "FREEZE_MANIFEST.json")
    parent_payload = parent_manifest.get("scientific_payload")
    if parent_manifest.get("freeze_schema") != "shared_benchmark.cardiac_benchmark_freeze.v3" or not isinstance(parent_payload, dict):
        raise ValueError("v4 requires the immutable v3 parent freeze")
    manifest = _read_json(PARENT_FREEZE / "data" / "acdc_shared_manifest.json")
    selection = _read_json(PARENT_FREEZE / "data" / "acdc_selection_receipt.json")
    spec = _read_json(SOURCE_SPEC)
    fixtures = _read_json(SOURCE_FIXTURES)
    grid = _grid_contract()
    counts = _counts(manifest)
    if (
        manifest.get("scientific") is not True
        or manifest.get("split_policy_version") != "self_audit.acdc.patient_split.v1"
        or manifest.get("shared_grid_hash") != grid["shared_grid_sha256"]
        or counts != {
            "train": {"patients": 80, "volumes": 160, "samples": 1526},
            "dev": {"patients": 20, "volumes": 40, "samples": 376},
            "test": {"patients": 0, "volumes": 0, "samples": 0},
        }
    ):
        raise ValueError(f"parent v3 static protocol is not the required Self-Audit cohort: {counts}")
    if selection.get("schema_version") != "self_audit.acdc.frame_selection.v1":
        raise ValueError("parent v3 frame-selection receipt is unsupported")
    if spec.get("adapter_version") != "cardiac_adapter_v2" or spec.get("schema_version") != "shared_benchmark.cardiac_adapter_spec.v2":
        raise ValueError("adapter-v2 source specification is invalid")

    output.mkdir(parents=True)
    for relative in ("configs", "data", "receipts"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    (output / "configs" / "adapter_v2_spec.json").write_bytes(SOURCE_SPEC.read_bytes())
    # The topology fixtures are carried forward byte-for-byte: the v2 change
    # seals trace/metadata and the real central-image representation, not the
    # topology policy or expected semantic maps.
    (output / "configs" / "adapter_v1_synthetic_fixtures.json").write_bytes(SOURCE_FIXTURES.read_bytes())
    _write_json(output / "configs" / "resolved_shared_contract.json", grid)
    for relative in ("acdc_shared_manifest.json", "acdc_selection_receipt.json", "acdc_patient_split_seed42.json"):
        (output / "data" / relative).write_bytes((PARENT_FREEZE / "data" / relative).read_bytes())
    (output / "data" / "README.md").write_text(
        "# Static Self-Audit ACDC protocol\n\n"
        "v4 preserves the v3 cohort exactly: 80 train patients / 20 dev patients, "
        "ED+ES and all acquired Z slices, with no test cohort. This directory "
        "does not contain images, GT, predictions, checkpoints, or evaluation metrics.\n",
        encoding="utf-8",
    )
    _write_json(output / "receipts" / "repository.json", {
        "receipt_schema": "shared_benchmark.freeze_repository_receipt.v1",
        "repository_commit_sha": _run("git", "rev-parse", "HEAD"),
        "branch_at_freeze_attempt": _run("git", "branch", "--show-current"),
        "worktree_clean_at_start": False,
        "note": "Existing worktree changes were preserved; this receipt is not a clean-tree assertion.",
    })
    _write_json(output / "receipts" / "environment.json", _environment_receipt())
    _write_json(output / "receipts" / "firewall.json", {
        "receipt_schema": "shared_benchmark.freeze_firewall_receipt.v1",
        "schema_firewall": "PASS_STATIC",
        "physical_gt_isolation": "INHERITED_PARENT_V3_STATIC_EVIDENCE",
        "runtime_workloads": "NOT_RUN_BY_USER_PROTOCOL",
    })
    _write_json(output / "receipts" / "tests.json", {
        "receipt_schema": "shared_benchmark.freeze_test_receipt.v1",
        "status": "STATIC_VALIDATION_PENDING",
        "runtime_gates": "NOT_RUN",
    })

    bound_paths = [
        "configs/adapter_v2_spec_source.json", "configs/adapter_v1_synthetic_fixtures_source.json",
        "configs/self_audit_full.yaml", "configs/self_audit_annotation.yaml", "configs/self_audit_auditor.yaml", "configs/self_audit_joint.yaml",
        "splits/acdc_patient_split_seed42.json", "src/shared_benchmark/adapter.py", "src/shared_benchmark/region_graph.py",
        "src/shared_benchmark/semantic_contract.py", "src/shared_benchmark/artifacts.py", "src/shared_benchmark/firewall.py",
        "src/shared_benchmark/spatial.py", "src/shared_benchmark/provenance.py", "src/self_audit_maskfree/data/firewall.py",
        "src/shared_benchmark/manifest.py", "src/shared_benchmark/self_audit_protocol.py",
        "scripts/regenerate_cardiac_benchmark_freeze.py", "scripts/regenerate_self_audit_cardiac_benchmark_v4_freeze.py",
        "scripts/validate_cardiac_benchmark_v4_freeze.py", "scripts/run_cuts_scientific.py", "scripts/run_dfc_scientific.py",
        "scripts/build_self_audit_acdc_selection.py", "scripts/prepare_self_audit_shared_benchmark_manifest.py",
        "baseline/CUTS/src/cardiac_benchmark/dataset.py", "baseline/CUTS/src/cardiac_benchmark/train_stage1.py",
        "baseline/CUTS/src/cardiac_benchmark/export_latents.py", "baseline/CUTS/src/cardiac_benchmark/cluster_kmeans.py",
        "baseline/DFC/src/cardiac_benchmark/dataset.py", "baseline/DFC/src/cardiac_benchmark/config.py",
        "baseline/DFC/src/cardiac_benchmark/dfc_runner.py", "baseline/DFC/config/cardiac/dfc_direct_2d_minl3.yaml",
    ]
    bound_freeze_paths = [
        "configs/adapter_v2_spec.json", "configs/adapter_v1_synthetic_fixtures.json", "configs/resolved_shared_contract.json",
        "data/acdc_shared_manifest.json", "data/acdc_selection_receipt.json", "data/acdc_patient_split_seed42.json",
    ]
    payload: dict[str, Any] = {
        "freeze_schema_version": FREEZE_SCHEMA,
        "parent_freeze": {
            "freeze_id": parent_manifest["freeze_id"],
            "scientific_payload_sha256": parent_manifest["scientific_payload_sha256"],
            "adapter_spec_sha256": parent_payload["adapter"]["specification_sha256"],
            "relationship": "v3 raw/checkpoint provenance retained; v4 is required for new semantic artifacts",
        },
        "authoritative_self_audit_policy": parent_payload["authoritative_self_audit_policy"],
        "shared_manifest": {
            "manifest_sha256": manifest["manifest_hash"],
            "manifest_copy_sha256": _sha256_file(output / "data" / "acdc_shared_manifest.json"),
            "patient_counts": {key: counts[key]["patients"] for key in counts},
            "volume_counts": {key: counts[key]["volumes"] for key in counts},
            "sample_counts": {key: counts[key]["samples"] for key in counts},
        },
        "shared_spatial_contract": {
            "version": grid["version"], "target_hw": grid["target_hw"], "whole_fov": grid["whole_fov"],
            "crop": grid["crop"], "resampling": grid["forward_values"], "grid_sha256": grid["shared_grid_sha256"],
            "config_provenance": grid["config_provenance"],
        },
        "adapter": {
            "adapter_version": "cardiac_adapter_v2",
            "input_schema_version": "shared_benchmark.anonymous_partition.v2",
            "output_schema_version": "shared_benchmark.cardiac-semantic.v2",
            "specification_sha256": _sha256_file(output / "configs" / "adapter_v2_spec.json"),
            "central_image_contract": "Self-Audit full-volume 0.5/99.5 percentile clip + population z-score, then central plane whole-FOV bilinear resize 256x256 align_corners=False",
            "intensity_resolution": "disabled",
            "orientation_resolution": "disabled",
            "topology_policy": "unchanged_from_v1",
            "metadata_seal": "complete adapter_metadata plus raw-input recomputation on read",
            "implementation_identity_closure": ["adapter.py", "region_graph.py", "semantic_contract.py", "artifacts.py", "firewall.py", "spatial.py", "provenance.py", "self_audit_maskfree/data/firewall.py"],
            "synthetic_fixture_set_sha256": _sha256_file(output / "configs" / "adapter_v1_synthetic_fixtures.json"),
        },
        "semantic_lineage": {
            "prior_semantic_outputs": "SUPERSEDED_PENDING_REGENERATION_FROM_VERIFIED_RAW",
            "checkpoint_and_raw_contract": "UNCHANGED; no checkpoint, raw partition, cohort, split, frame/slice selection, grid, or baseline optimization schedule was modified",
            "reusable_raw_requirement": "raw artifact must pass its original verifier and match the v4 runner input contract before regeneration",
        },
        "datasets": {"acdc": {"scientific_manifest_status": "PASS_STATIC", "runtime_status": "NOT_RUN"}, "mnms": {"scientific_manifest_status": "DEFERRED", "runtime_status": "NOT_RUN"}},
        "scientific_readiness": "CODE_AND_STATIC_PREFLIGHT_PASS_WAITING_FOR_USER_DECISION",
        "bound_repository_files": [_repo_binding(path) for path in bound_paths],
        "bound_freeze_files": [{"path": path, "sha256": _sha256_file(output / path)} for path in bound_freeze_paths],
        "pre_freeze_sources": {
            "adapter_v2_spec_source_sha256": _sha256_file(SOURCE_SPEC),
            "topology_fixture_source_sha256": _sha256_file(SOURCE_FIXTURES),
            "parent_freeze_manifest_sha256": _sha256_file(PARENT_FREEZE / "FREEZE_MANIFEST.json"),
            "generator": "scripts/regenerate_self_audit_cardiac_benchmark_v4_freeze.py",
        },
    }
    payload_hash = _sha256_bytes(_canonical_bytes(payload))
    freeze = {
        "freeze_schema": FREEZE_SCHEMA,
        "freeze_id": f"{FREEZE_ID_PREFIX}-{payload_hash[:16]}",
        "scientific_payload_sha256": payload_hash,
        "scientific_payload": payload,
    }
    _write_json(output / "FREEZE_MANIFEST.json", freeze)
    (output / "FREEZE_REPORT.md").write_text(
        "# Cardiac benchmark v4 freeze (adapter v2)\n\n"
        f"- freeze ID: `{freeze['freeze_id']}`\n"
        f"- scientific payload SHA-256: `{payload_hash}`\n"
        f"- parent: `{parent_manifest['freeze_id']}`\n"
        f"- ACDC samples: train={counts['train']['samples']} / dev={counts['dev']['samples']} / test=0\n\n"
        "Static code/provenance only. No ACDC generation, training, DFC optimization, PHATE/KMeans, or GT evaluation was run.\n",
        encoding="utf-8",
    )
    return {"freeze_id": freeze["freeze_id"], "scientific_payload_sha256": payload_hash, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(regenerate(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
