#!/usr/bin/env python
"""Create the active v7 STEGO/PiCIE compat-224 adapter-v3 freeze."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v6"
OUTPUT = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v7"
SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v7"
PREFIX = "cardiac-benchmark-v7"

source = str(REPO_ROOT / "src")
if source not in sys.path:
    sys.path.insert(0, source)

from shared_benchmark.manifest import manifest_hash
from shared_benchmark.semantic_contract import (
    ADAPTER_V3_INPUT_SCHEMA_VERSION,
    ADAPTER_V3_OUTPUT_SCHEMA_VERSION,
    ADAPTER_V3_VERSION,
    FROZEN_ADAPTER_V3_SPEC_SHA256,
)
from shared_benchmark.spatial import grid_hash, load_self_audit_compat_224_grid_spec, spatial_transform_for_record


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
        raise ValueError(f"expected object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _binding(relative: str) -> dict[str, str]:
    path = REPO_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "sha256": _hash_file(path)}


def _compat_manifest(parent_manifest: Mapping[str, Any], compat_grid: Mapping[str, Any]) -> dict[str, Any]:
    manifest = copy.deepcopy(dict(parent_manifest))
    manifest["shared_grid"] = copy.deepcopy(dict(compat_grid))
    manifest["shared_grid_hash"] = grid_hash(compat_grid)
    for record in manifest.get("records", []):
        record["shared_grid"] = copy.deepcopy(dict(compat_grid))
        record["spatial_transform"] = spatial_transform_for_record(record, compat_grid)
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def regenerate(output: Path = OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing freeze: {output}")
    parent = _read(PARENT / "FREEZE_MANIFEST.json")
    parent_payload = parent.get("scientific_payload")
    if parent.get("freeze_schema") != "shared_benchmark.cardiac_benchmark_freeze.v6" or not isinstance(parent_payload, dict):
        raise ValueError("v7 requires retained v6 static provenance")
    parent_manifest = _read(PARENT / "data" / "acdc_shared_manifest.json")
    counts = {split: sum(row.get("split") == split for row in parent_manifest.get("records", [])) for split in ("train", "dev", "test")}
    if counts != {"train": 1526, "dev": 376, "test": 0}:
        raise ValueError("v7 requires the Self-Audit ACDC train/dev cohort")
    compat_grid = load_self_audit_compat_224_grid_spec(REPO_ROOT)
    if compat_grid["target_hw"] != [224, 224]:
        raise ValueError("v7 requires the frozen compat-224 grid")
    compat_manifest = _compat_manifest(parent_manifest, compat_grid)

    output.mkdir(parents=True)
    for directory in ("configs", "data", "receipts"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    resolved_grid = dict(compat_grid)
    resolved_grid["shared_grid_sha256"] = grid_hash(compat_grid)
    _write(output / "configs" / "resolved_shared_contract.json", resolved_grid)
    (output / "configs" / "adapter_v3_spec.json").write_bytes((REPO_ROOT / "configs" / "adapter_v3_spec_source.json").read_bytes())
    (output / "configs" / "adapter_v1_synthetic_fixtures.json").write_bytes((PARENT / "configs" / "adapter_v1_synthetic_fixtures.json").read_bytes())
    _write(output / "data" / "acdc_shared_manifest.json", compat_manifest)
    for relative in ("data/acdc_selection_receipt.json", "data/acdc_patient_split_seed42.json"):
        (output / relative).write_bytes((PARENT / relative).read_bytes())
    (output / "data" / "README.md").write_text(
        "# Active Self-Audit ACDC STEGO/PiCIE compat-224 freeze\n\n"
        "v7 retains the authoritative Self-Audit ACDC cohort while publishing the 224x224 compat grid for "
        "STEGO/PiCIE SA224 raw artifacts and the shared adapter-v3 semantic handoff. The freeze is a static "
        "reprojection from v6 plus the new frozen adapter spec; runtime workloads are not executed by this script.\n",
        encoding="utf-8",
    )
    _write(output / "receipts" / "repository.json", {
        "receipt_schema": "shared_benchmark.freeze_repository_receipt.v1",
        "repository_commit_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "worktree_clean_at_start": False,
        "note": "Static re-freeze: derive compat-224 shared manifest from v6 and bind adapter-v3/stego-picie runners; runtime evidence is external.",
    })
    _write(output / "receipts" / "tests.json", {
        "receipt_schema": "shared_benchmark.freeze_test_receipt.v1",
        "status": "STATIC_VALIDATION_PENDING",
        "runtime_gates": "NOT_RUN_BY_FREEZE_SCRIPT",
    })
    _write(output / "receipts" / "firewall.json", {
        "receipt_schema": "shared_benchmark.freeze_firewall_receipt.v1",
        "schema_firewall": "PASS_STATIC",
        "runtime_workloads": "NOT_RUN_BY_FREEZE_SCRIPT",
    })

    frozen_files = [
        "configs/adapter_v3_spec.json",
        "configs/adapter_v1_synthetic_fixtures.json",
        "configs/resolved_shared_contract.json",
        "data/acdc_shared_manifest.json",
        "data/acdc_selection_receipt.json",
        "data/acdc_patient_split_seed42.json",
    ]
    bound = [
        "configs/adapter_v3_spec_source.json",
        "configs/adapter_v1_synthetic_fixtures_source.json",
        "src/shared_benchmark/checkpoint_contract.py",
        "src/shared_benchmark/adapter.py",
        "src/shared_benchmark/region_graph.py",
        "src/shared_benchmark/semantic_contract.py",
        "src/shared_benchmark/artifacts.py",
        "src/shared_benchmark/firewall.py",
        "src/shared_benchmark/spatial.py",
        "src/shared_benchmark/provenance.py",
        "src/shared_benchmark/manifest.py",
        "src/shared_benchmark/self_audit_protocol.py",
        "src/self_audit_maskfree/data/firewall.py",
        "scripts/prepare_shared_benchmark_manifest.py",
        "scripts/run_stego_scientific.py",
        "scripts/run_picie_scientific.py",
        "scripts/regenerate_self_audit_cardiac_benchmark_v6_freeze.py",
        "scripts/validate_cardiac_benchmark_v6_freeze.py",
        "scripts/regenerate_self_audit_cardiac_benchmark_v7_freeze.py",
        "scripts/validate_cardiac_benchmark_v7_freeze.py",
        "baseline/STEGO/src/cardiac_benchmark/config.py",
        "baseline/STEGO/src/cardiac_benchmark/dataset.py",
        "baseline/STEGO/src/cardiac_benchmark/stego_runner.py",
        "baseline/PICIE/src/cardiac_benchmark/config.py",
        "baseline/PICIE/src/cardiac_benchmark/dataset.py",
        "baseline/PICIE/src/cardiac_benchmark/picie_runner.py",
        "baseline/PICIE/picie/__init__.py",
        "baseline/PICIE/picie/backbone.py",
        "baseline/PICIE/picie/fpn.py",
    ]

    payload = {
        "freeze_schema_version": SCHEMA,
        "parent_freeze": {
            "freeze_id": parent["freeze_id"],
            "scientific_payload_sha256": parent["scientific_payload_sha256"],
            "status": "RETAINED_STATIC_PREDECESSOR",
        },
        "lineage": parent_payload["lineage"],
        "authoritative_self_audit_policy": parent_payload["authoritative_self_audit_policy"],
        "shared_manifest": {
            "manifest_sha256": compat_manifest["manifest_hash"],
            "manifest_copy_sha256": _hash_file(output / "data" / "acdc_shared_manifest.json"),
            "patient_counts": parent_payload["shared_manifest"]["patient_counts"],
            "volume_counts": parent_payload["shared_manifest"]["volume_counts"],
            "sample_counts": counts,
        },
        "shared_spatial_contract": {
            "version": compat_grid["version"],
            "target_hw": compat_grid["target_hw"],
            "whole_fov": compat_grid["whole_fov"],
            "crop": compat_grid["crop"],
            "resampling": compat_grid["forward_values"],
            "grid_sha256": grid_hash(compat_grid),
            "config_provenance": compat_grid["config_provenance"],
        },
        "adapter": {
            "adapter_version": ADAPTER_V3_VERSION,
            "input_schema_version": ADAPTER_V3_INPUT_SCHEMA_VERSION,
            "output_schema_version": ADAPTER_V3_OUTPUT_SCHEMA_VERSION,
            "specification_sha256": FROZEN_ADAPTER_V3_SPEC_SHA256,
            "central_image_contract": "Self-Audit full-volume 0.5/99.5 percentile clip + population z-score, then central plane whole-FOV bilinear resize 224x224 align_corners=False on the STEGO/PiCIE compat grid",
            "intensity_resolution": "disabled",
            "orientation_resolution": "disabled",
            "topology_policy": "connected-component many-to-one merge only; no component splitting",
            "metadata_seal": "complete adapter_metadata plus raw-input recomputation on read",
            "implementation_identity_closure": [
                "adapter.py",
                "region_graph.py",
                "semantic_contract.py",
                "artifacts.py",
                "firewall.py",
                "spatial.py",
                "provenance.py",
                "self_audit_maskfree/data/firewall.py",
            ],
            "synthetic_fixture_set_sha256": _hash_file(output / "configs" / "adapter_v1_synthetic_fixtures.json"),
        },
        "semantic_lineage": {
            "prior_semantic_outputs": "NO_COMPAT224_SHARED_STAGE_PUBLISHED_BEFORE_V7",
            "checkpoint_and_raw_contract": "STEGO/PiCIE SA224 raw artifacts bind Self-Audit-normalized ACDC sources on the frozen compat-224 grid; CUTS/DFC v6 raw and semantic contracts remain unchanged",
            "reusable_raw_requirement": "raw artifact must pass its verifier and match the v7 STEGO/PiCIE SA224 runner contract before shared semantic regeneration",
            "semantic_artifact_stage": "semantic-cardiac_adapter_v3",
            "historical_v2_semantic_stage": "semantic-cardiac_adapter_v2",
            "overwrite_policy": "forbidden_by_distinct_stage",
        },
        "runner_binding": {
            "cuts_adapter_default": "benchmark_freezes/cardiac_benchmark_v6/configs/adapter_v2_spec.json",
            "dfc_adapter_default": "benchmark_freezes/cardiac_benchmark_v6/configs/adapter_v2_spec.json",
            "stego_sa224_adapter_default": "benchmark_freezes/cardiac_benchmark_v7/configs/adapter_v3_spec.json",
            "picie_sa224_adapter_default": "benchmark_freezes/cardiac_benchmark_v7/configs/adapter_v3_spec.json",
            "stego_sa224_fair_requires_checkpoint_contract": True,
            "picie_sa224_fair_requires_checkpoint_contract": True,
        },
        "datasets": {
            "acdc": {
                "scientific_manifest_status": "PASS_STATIC",
                "runtime_status": "RUNTIME_EVIDENCE_EXTERNAL_TO_FREEZE_SCRIPT",
            },
            "mnms": parent_payload["datasets"]["mnms"],
        },
        "scientific_readiness": "STATIC_PREFLIGHT_PASS_WAITING_FOR_RUNTIME_DECISION",
        "bound_repository_files": [_binding(path) for path in bound],
        "bound_freeze_files": [{"path": path, "sha256": _hash_file(output / path)} for path in frozen_files],
        "pre_freeze_sources": {
            "parent_v6_freeze_manifest_sha256": _hash_file(PARENT / "FREEZE_MANIFEST.json"),
            "parent_v6_shared_manifest_sha256": _hash_file(PARENT / "data" / "acdc_shared_manifest.json"),
            "generator": "scripts/regenerate_self_audit_cardiac_benchmark_v7_freeze.py",
            "manifest_derivation": "static_grid_reprojection_from_v6_to_self_audit_compat_224",
        },
    }
    digest = _hash_bytes(_canonical(payload))
    document = {"freeze_schema": SCHEMA, "freeze_id": f"{PREFIX}-{digest[:16]}", "scientific_payload_sha256": digest, "scientific_payload": payload}
    _write(output / "FREEZE_MANIFEST.json", document)
    (output / "FREEZE_REPORT.md").write_text(
        "# Active cardiac benchmark v7 freeze (STEGO/PiCIE compat-224 adapter v3)\n\n"
        f"- freeze ID: `{document['freeze_id']}`\n"
        "- Self-Audit ACDC samples: train=1526 / dev=376 / test=0\n"
        "- Shared grid: Self-Audit compat 224x224\n"
        "- Runtime workloads: NOT_RUN_BY_FREEZE_SCRIPT\n",
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
