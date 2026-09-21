#!/usr/bin/env python
"""Create the active v6 adapter-v2 freeze without modifying prior freezes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v5"
OUTPUT = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v6"
SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v6"
PREFIX = "cardiac-benchmark-v6"


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


def regenerate(output: Path = OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing freeze: {output}")
    parent = _read(PARENT / "FREEZE_MANIFEST.json")
    parent_payload = parent.get("scientific_payload")
    if parent.get("freeze_schema") != "shared_benchmark.cardiac_benchmark_freeze.v5" or not isinstance(parent_payload, dict):
        raise ValueError("v6 requires retained v5 static provenance")
    manifest = _read(PARENT / "data" / "acdc_shared_manifest.json")
    counts = {split: sum(row.get("split") == split for row in manifest.get("records", [])) for split in ("train", "dev", "test")}
    if counts != {"train": 1526, "dev": 376, "test": 0}:
        raise ValueError("v6 requires the Self-Audit ACDC train/dev cohort")
    output.mkdir(parents=True)
    for directory in ("configs", "data", "receipts"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    frozen_files = [
        "configs/adapter_v2_spec.json", "configs/adapter_v1_synthetic_fixtures.json", "configs/resolved_shared_contract.json",
        "data/acdc_shared_manifest.json", "data/acdc_selection_receipt.json", "data/acdc_patient_split_seed42.json",
    ]
    for relative in frozen_files:
        (output / relative).write_bytes((PARENT / relative).read_bytes())
    (output / "data" / "README.md").write_text(
        "# Active Self-Audit ACDC adapter-v2 freeze\n\n"
        "v6 retains the v3 cohort and v2 semantic contract. It introduces a versioned semantic artifact directory, "
        "so v2 generation cannot overwrite preserved v1 semantic artifacts. No runtime workload was run.\n",
        encoding="utf-8",
    )
    _write(output / "receipts" / "repository.json", {
        "receipt_schema": "shared_benchmark.freeze_repository_receipt.v1",
        "repository_commit_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "worktree_clean_at_start": False,
        "note": "Static re-freeze: preserve historical semantic directory and bind v2 stage identity; no data access.",
    })
    _write(output / "receipts" / "tests.json", {"receipt_schema": "shared_benchmark.freeze_test_receipt.v1", "status": "STATIC_VALIDATION_PENDING", "runtime_gates": "NOT_RUN"})
    _write(output / "receipts" / "firewall.json", {"receipt_schema": "shared_benchmark.freeze_firewall_receipt.v1", "schema_firewall": "PASS_STATIC", "runtime_workloads": "NOT_RUN_BY_USER_PROTOCOL"})
    bound = [
        "configs/adapter_v2_spec_source.json", "configs/adapter_v1_synthetic_fixtures_source.json",
        "src/shared_benchmark/adapter.py", "src/shared_benchmark/region_graph.py", "src/shared_benchmark/semantic_contract.py",
        "src/shared_benchmark/artifacts.py", "src/shared_benchmark/firewall.py", "src/shared_benchmark/spatial.py", "src/shared_benchmark/provenance.py",
        "src/shared_benchmark/manifest.py", "src/shared_benchmark/self_audit_protocol.py", "src/self_audit_maskfree/data/firewall.py",
        "scripts/regenerate_self_audit_cardiac_benchmark_v4_freeze.py", "scripts/validate_cardiac_benchmark_v4_freeze.py",
        "scripts/regenerate_self_audit_cardiac_benchmark_v5_freeze.py", "scripts/validate_cardiac_benchmark_v5_freeze.py",
        "scripts/regenerate_self_audit_cardiac_benchmark_v6_freeze.py", "scripts/validate_cardiac_benchmark_v6_freeze.py",
        "scripts/run_cuts_scientific.py", "scripts/run_dfc_scientific.py", "scripts/regenerate_cardiac_benchmark_freeze.py",
        "baseline/CUTS/src/cardiac_benchmark/dataset.py", "baseline/CUTS/src/cardiac_benchmark/train_stage1.py",
        "baseline/CUTS/src/cardiac_benchmark/export_latents.py", "baseline/CUTS/src/cardiac_benchmark/cluster_kmeans.py",
        "baseline/DFC/src/cardiac_benchmark/dataset.py", "baseline/DFC/src/cardiac_benchmark/config.py",
        "baseline/DFC/src/cardiac_benchmark/dfc_runner.py", "baseline/DFC/config/cardiac/dfc_direct_2d_minl3.yaml",
    ]
    payload = {
        "freeze_schema_version": SCHEMA,
        "parent_freeze": {"freeze_id": parent["freeze_id"], "scientific_payload_sha256": parent["scientific_payload_sha256"], "status": "RETAINED_STATIC_PREDECESSOR"},
        "lineage": parent_payload["lineage"],
        "authoritative_self_audit_policy": parent_payload["authoritative_self_audit_policy"],
        "shared_manifest": parent_payload["shared_manifest"],
        "shared_spatial_contract": parent_payload["shared_spatial_contract"],
        "adapter": parent_payload["adapter"],
        "semantic_lineage": {**parent_payload["semantic_lineage"], "semantic_artifact_stage": "semantic-cardiac_adapter_v2", "historical_v1_semantic_stage": "semantic", "overwrite_policy": "forbidden_by_distinct_stage"},
        "runner_binding": {"cuts_adapter_default": "benchmark_freezes/cardiac_benchmark_v6/configs/adapter_v2_spec.json", "dfc_adapter_default": "benchmark_freezes/cardiac_benchmark_v6/configs/adapter_v2_spec.json"},
        "datasets": parent_payload["datasets"],
        "scientific_readiness": "CODE_AND_STATIC_PREFLIGHT_PASS_WAITING_FOR_USER_DECISION",
        "bound_repository_files": [_binding(path) for path in bound],
        "bound_freeze_files": [{"path": path, "sha256": _hash_file(output / path)} for path in frozen_files],
        "pre_freeze_sources": {"parent_v5_freeze_manifest_sha256": _hash_file(PARENT / "FREEZE_MANIFEST.json"), "generator": "scripts/regenerate_self_audit_cardiac_benchmark_v6_freeze.py"},
    }
    digest = _hash_bytes(_canonical(payload))
    document = {"freeze_schema": SCHEMA, "freeze_id": f"{PREFIX}-{digest[:16]}", "scientific_payload_sha256": digest, "scientific_payload": payload}
    _write(output / "FREEZE_MANIFEST.json", document)
    (output / "FREEZE_REPORT.md").write_text(
        "# Active cardiac benchmark v6 freeze (adapter v2)\n\n"
        f"- freeze ID: `{document['freeze_id']}`\n"
        "- Self-Audit ACDC samples: train=1526 / dev=376 / test=0\n"
        "- Runtime workloads: NOT_RUN\n", encoding="utf-8",
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
