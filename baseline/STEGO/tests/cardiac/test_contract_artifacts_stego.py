from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BASELINE_ROOT.parents[1]
for name in list(sys.modules):
    if name == "cardiac_benchmark" or name.startswith("cardiac_benchmark."):
        sys.modules.pop(name)
for import_path in (BASELINE_ROOT / "src", BASELINE_ROOT, REPO_ROOT, REPO_ROOT / "src"):
    if str(import_path) in sys.path:
        sys.path.remove(str(import_path))
for import_path in (BASELINE_ROOT / "src", BASELINE_ROOT, REPO_ROOT, REPO_ROOT / "src"):
    sys.path.insert(0, str(import_path))

import pytest

from scripts import run_stego_scientific
from shared_benchmark.spatial import load_self_audit_compat_224_grid_spec


def test_stego_runner_exposes_shared_artifact_cli():
    parser = run_stego_scientific.build_parser()
    options = {action.dest for action in parser._actions}
    required = {
        "manifest", "image_root", "output_root", "checkpoint", "split", "device",
        "limit", "sample_list", "config_hash", "apply_adapter", "adapter_spec",
        "semantic_root", "retry_failed", "checkpoint_contract",
    }
    assert required <= options


def test_stego_sa224_runner_allows_semantic_handoff():
    run_stego_scientific._require_profile_semantic_handoff("STEGO-SA224", True)


def test_stego_legacy_runner_rejects_semantic_handoff():
    with pytest.raises(run_stego_scientific.ArtifactError, match="published only for STEGO SA224 profiles"):
        run_stego_scientific._require_profile_semantic_handoff("STEGO-2D", True)


def test_stego_sa224_runner_accepts_compat_contract():
    manifest = {
        "schema_version": "shared_benchmark_manifest.v1",
        "split_policy_version": "self_audit.acdc.patient_split.v1",
        "shared_grid": load_self_audit_compat_224_grid_spec(REPO_ROOT),
    }
    manifest["shared_grid_hash"] = run_stego_scientific.grid_hash(manifest["shared_grid"])
    run_stego_scientific._require_manifest_contract(manifest, "STEGO-SA224")


def test_stego_sa224_runner_rejects_legacy_grid():
    manifest = {
        "schema_version": "shared_benchmark_manifest.v1",
        "split_policy_version": "maskfree150.data.discovery.v2",
        "shared_grid": copy.deepcopy(load_self_audit_compat_224_grid_spec(REPO_ROOT)),
    }
    manifest["shared_grid"]["version"] = run_stego_scientific.SPATIAL_CONTRACT_VERSION
    manifest["shared_grid_hash"] = run_stego_scientific.grid_hash(manifest["shared_grid"])
    with pytest.raises(run_stego_scientific.ArtifactError, match="STEGO-SA224"):
        run_stego_scientific._require_manifest_contract(manifest, "STEGO-SA224")


def test_stego_fair_runner_requires_checkpoint_contract():
    manifest = {
        "dataset": "acdc",
        "split_policy_version": "self_audit.acdc.patient_split.v1",
        "shared_grid_hash": run_stego_scientific.grid_hash(load_self_audit_compat_224_grid_spec(REPO_ROOT)),
    }
    with pytest.raises(run_stego_scientific.ArtifactError, match="requires --checkpoint-contract"):
        run_stego_scientific._require_fair_checkpoint_contract(
            "STEGO-SA224-FAIR",
            None,
            checkpoint_sha256="abc123",
            manifest=manifest,
        )


def test_stego_fair_runner_accepts_matching_checkpoint_contract(tmp_path):
    grid = load_self_audit_compat_224_grid_spec(REPO_ROOT)
    manifest = {
        "dataset": "acdc",
        "split_policy_version": "self_audit.acdc.patient_split.v1",
        "shared_grid_hash": run_stego_scientific.grid_hash(grid),
    }
    contract_path = tmp_path / "stego_checkpoint_contract.json"
    contract_path.write_text(json.dumps({
        "schema_version": "shared_benchmark.checkpoint_contract.v1",
        "baseline_name": "STEGO",
        "baseline_mode": "STEGO-SA224-FAIR",
        "benchmark_tier": "fair",
        "checkpoint_sha256": "abc123",
        "dataset": "acdc",
        "split_policy_version": "self_audit.acdc.patient_split.v1",
        "shared_grid_hash": manifest["shared_grid_hash"],
        "normalization_version": "stego.cardiac.sa224_self_audit_volume_then_central_fixed_affine_v1",
        "training_data_schema": "self_audit.acdc.image_only.v1",
        "training_tags": ["in_domain_acdc", "self_audit_normalized"],
    }), encoding="utf-8")
    run_stego_scientific._require_fair_checkpoint_contract(
        "STEGO-SA224-FAIR",
        contract_path,
        checkpoint_sha256="abc123",
        manifest=manifest,
    )
