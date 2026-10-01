"""Fair-benchmark checkpoint sidecar validation for shared cardiac baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "shared_benchmark.checkpoint_contract.v1"


class CheckpointContractError(ValueError):
    """Raised when a fair-benchmark checkpoint sidecar is absent or inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointContractError(f"cannot read checkpoint contract {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointContractError(f"checkpoint contract must be a JSON object: {path}")
    return value


def load_checkpoint_contract(path: str | Path) -> dict[str, Any]:
    return _load_json(Path(path))


def validate_checkpoint_contract(
    contract: Mapping[str, Any],
    *,
    baseline_name: str,
    baseline_mode: str,
    checkpoint_sha256: str,
    dataset: str,
    split_policy_version: str,
    shared_grid_hash: str,
    normalization_version: str,
    benchmark_tier: str = "fair",
    training_data_schema: str = "self_audit.acdc.image_only.v1",
) -> dict[str, Any]:
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointContractError("checkpoint contract schema_version is invalid")
    expected = {
        "baseline_name": baseline_name,
        "baseline_mode": baseline_mode,
        "benchmark_tier": benchmark_tier,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset": dataset,
        "split_policy_version": split_policy_version,
        "shared_grid_hash": shared_grid_hash,
        "normalization_version": normalization_version,
        "training_data_schema": training_data_schema,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise CheckpointContractError(f"checkpoint contract mismatch: {key}")
    training_tags = contract.get("training_tags")
    if not isinstance(training_tags, list) or not training_tags:
        raise CheckpointContractError("checkpoint contract training_tags are required")
    if not all(isinstance(tag, str) and tag for tag in training_tags):
        raise CheckpointContractError("checkpoint contract training_tags must be non-empty strings")
    if "in_domain_acdc" not in training_tags:
        raise CheckpointContractError("fair checkpoint contract must declare in_domain_acdc")
    return dict(contract)


__all__ = [
    "SCHEMA_VERSION",
    "CheckpointContractError",
    "load_checkpoint_contract",
    "validate_checkpoint_contract",
]

