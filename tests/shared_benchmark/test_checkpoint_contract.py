from __future__ import annotations

import pytest

from shared_benchmark.checkpoint_contract import CheckpointContractError, validate_checkpoint_contract


def test_checkpoint_contract_accepts_matching_fair_sidecar():
    contract = {
        "schema_version": "shared_benchmark.checkpoint_contract.v1",
        "baseline_name": "STEGO",
        "baseline_mode": "STEGO-SA224-FAIR",
        "benchmark_tier": "fair",
        "checkpoint_sha256": "abc123",
        "dataset": "acdc",
        "split_policy_version": "self_audit.acdc.patient_split.v1",
        "shared_grid_hash": "grid123",
        "normalization_version": "norm123",
        "training_data_schema": "self_audit.acdc.image_only.v1",
        "training_tags": ["in_domain_acdc", "self_audit_normalized"],
    }
    result = validate_checkpoint_contract(
        contract,
        baseline_name="STEGO",
        baseline_mode="STEGO-SA224-FAIR",
        checkpoint_sha256="abc123",
        dataset="acdc",
        split_policy_version="self_audit.acdc.patient_split.v1",
        shared_grid_hash="grid123",
        normalization_version="norm123",
    )
    assert result == contract


def test_checkpoint_contract_requires_in_domain_acdc_tag():
    contract = {
        "schema_version": "shared_benchmark.checkpoint_contract.v1",
        "baseline_name": "PICIE",
        "baseline_mode": "PICIE-SA224-FAIR",
        "benchmark_tier": "fair",
        "checkpoint_sha256": "abc123",
        "dataset": "acdc",
        "split_policy_version": "self_audit.acdc.patient_split.v1",
        "shared_grid_hash": "grid123",
        "normalization_version": "norm123",
        "training_data_schema": "self_audit.acdc.image_only.v1",
        "training_tags": ["self_audit_normalized"],
    }
    with pytest.raises(CheckpointContractError, match="in_domain_acdc"):
        validate_checkpoint_contract(
            contract,
            baseline_name="PICIE",
            baseline_mode="PICIE-SA224-FAIR",
            checkpoint_sha256="abc123",
            dataset="acdc",
            split_policy_version="self_audit.acdc.patient_split.v1",
            shared_grid_hash="grid123",
            normalization_version="norm123",
        )
