#!/usr/bin/env python
"""Create the final, non-overwriting adapter-v2 v5 static freeze from v4."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_FREEZE = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v4"
FREEZE_SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v5"
FREEZE_ID_PREFIX = "cardiac-benchmark-v5"
DEFAULT_OUTPUT = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v5"


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
        raise ValueError(f"expected object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _repo_binding(relative: str) -> dict[str, str]:
    path = REPO_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "sha256": _sha256_file(path)}


def _counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    return {split: sum(row.get("split") == split for row in manifest.get("records", [])) for split in ("train", "dev", "test")}


def regenerate(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing freeze: {output}")
    parent = _read_json(PARENT_FREEZE / "FREEZE_MANIFEST.json")
    parent_payload = parent.get("scientific_payload")
    if parent.get("freeze_schema") != "shared_benchmark.cardiac_benchmark_freeze.v4" or not isinstance(parent_payload, dict):
        raise ValueError("v5 requires the retained v4 adapter-v2 static freeze")
    manifest = _read_json(PARENT_FREEZE / "data" / "acdc_shared_manifest.json")
    if _counts(manifest) != {"train": 1526, "dev": 376, "test": 0}:
        raise ValueError("parent manifest is not the Self-Audit v3 cohort")
    output.mkdir(parents=True)
    for directory in ("configs", "data", "receipts"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    copied = [
        "configs/adapter_v2_spec.json", "configs/adapter_v1_synthetic_fixtures.json", "configs/resolved_shared_contract.json",
        "data/acdc_shared_manifest.json", "data/acdc_selection_receipt.json", "data/acdc_patient_split_seed42.json",
    ]
    for relative in copied:
        (output / relative).write_bytes((PARENT_FREEZE / relative).read_bytes())
    (output / "data" / "README.md").write_text(
        "# Active adapter-v2 static freeze\n\n"
        "v5 preserves v4/v3 Self-Audit ACDC data protocol exactly and updates only the bound runner references to adapter v2. "
        "No image, GT, prediction, checkpoint, generation, optimization, or evaluation artifact is stored here.\n",
        encoding="utf-8",
    )
    _write_json(output / "receipts" / "repository.json", {
        "receipt_schema": "shared_benchmark.freeze_repository_receipt.v1",
        "repository_commit_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "worktree_clean_at_start": False,
        "note": "Static re-freeze after runner default changed from historical v1 to v2; no data workload occurred.",
    })
    _write_json(output / "receipts" / "tests.json", {"receipt_schema": "shared_benchmark.freeze_test_receipt.v1", "status": "STATIC_VALIDATION_PENDING", "runtime_gates": "NOT_RUN"})
    _write_json(output / "receipts" / "firewall.json", {"receipt_schema": "shared_benchmark.freeze_firewall_receipt.v1", "schema_firewall": "PASS_STATIC", "runtime_workloads": "NOT_RUN_BY_USER_PROTOCOL"})

    bound_paths = [
        "configs/adapter_v2_spec_source.json", "configs/adapter_v1_synthetic_fixtures_source.json",
        "src/shared_benchmark/adapter.py", "src/shared_benchmark/region_graph.py", "src/shared_benchmark/semantic_contract.py",
        "src/shared_benchmark/artifacts.py", "src/shared_benchmark/firewall.py", "src/shared_benchmark/spatial.py", "src/shared_benchmark/provenance.py",
        "src/shared_benchmark/manifest.py", "src/shared_benchmark/self_audit_protocol.py", "src/self_audit_maskfree/data/firewall.py",
        "scripts/regenerate_self_audit_cardiac_benchmark_v4_freeze.py", "scripts/validate_cardiac_benchmark_v4_freeze.py",
        "scripts/regenerate_self_audit_cardiac_benchmark_v5_freeze.py", "scripts/validate_cardiac_benchmark_v5_freeze.py",
        "scripts/run_cuts_scientific.py", "scripts/run_dfc_scientific.py", "scripts/regenerate_cardiac_benchmark_freeze.py",
        "baseline/CUTS/src/cardiac_benchmark/dataset.py", "baseline/CUTS/src/cardiac_benchmark/train_stage1.py",
        "baseline/CUTS/src/cardiac_benchmark/export_latents.py", "baseline/CUTS/src/cardiac_benchmark/cluster_kmeans.py",
        "baseline/DFC/src/cardiac_benchmark/dataset.py", "baseline/DFC/src/cardiac_benchmark/config.py",
        "baseline/DFC/src/cardiac_benchmark/dfc_runner.py", "baseline/DFC/config/cardiac/dfc_direct_2d_minl3.yaml",
    ]
    payload: dict[str, Any] = {
        "freeze_schema_version": FREEZE_SCHEMA,
        "parent_freeze": {"freeze_id": parent["freeze_id"], "scientific_payload_sha256": parent["scientific_payload_sha256"], "status": "RETAINED_PRE_RUNNER_REFERENCE_FREEZE_NOT_FOR_RUNTIME"},
        "lineage": parent_payload["parent_freeze"],
        "authoritative_self_audit_policy": parent_payload["authoritative_self_audit_policy"],
        "shared_manifest": parent_payload["shared_manifest"],
        "shared_spatial_contract": parent_payload["shared_spatial_contract"],
        "adapter": parent_payload["adapter"],
        "semantic_lineage": parent_payload["semantic_lineage"],
        "runner_binding": {"cuts_adapter_default": "benchmark_freezes/cardiac_benchmark_v5/configs/adapter_v2_spec.json", "dfc_adapter_default": "benchmark_freezes/cardiac_benchmark_v5/configs/adapter_v2_spec.json"},
        "datasets": parent_payload["datasets"],
        "scientific_readiness": "CODE_AND_STATIC_PREFLIGHT_PASS_WAITING_FOR_USER_DECISION",
        "bound_repository_files": [_repo_binding(path) for path in bound_paths],
        "bound_freeze_files": [{"path": path, "sha256": _sha256_file(output / path)} for path in copied],
        "pre_freeze_sources": {"parent_v4_freeze_manifest_sha256": _sha256_file(PARENT_FREEZE / "FREEZE_MANIFEST.json"), "generator": "scripts/regenerate_self_audit_cardiac_benchmark_v5_freeze.py"},
    }
    payload_hash = _sha256_bytes(_canonical_bytes(payload))
    document = {"freeze_schema": FREEZE_SCHEMA, "freeze_id": f"{FREEZE_ID_PREFIX}-{payload_hash[:16]}", "scientific_payload_sha256": payload_hash, "scientific_payload": payload}
    _write_json(output / "FREEZE_MANIFEST.json", document)
    (output / "FREEZE_REPORT.md").write_text(
        "# Active cardiac benchmark v5 freeze (adapter v2)\n\n"
        f"- freeze ID: `{document['freeze_id']}`\n"
        f"- parent static freeze: `{parent['freeze_id']}`\n"
        "- ACDC samples: train=1526 / dev=376 / test=0\n\n"
        "Runtime gates remain NOT_RUN.\n", encoding="utf-8",
    )
    return {"freeze_id": document["freeze_id"], "scientific_payload_sha256": payload_hash, "counts": _counts(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(regenerate(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
