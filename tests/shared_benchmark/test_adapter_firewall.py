from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shared_benchmark.adapter import adapt_partition
from shared_benchmark.semantic_contract import ADAPTER_VERSION, FROZEN_ADAPTER_SPEC_SHA256, AdapterContractError, array_hash
from shared_benchmark.spatial import build_grid_spec


ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "benchmark_freezes/cardiac_benchmark_v1/configs/adapter_v1_spec.json").read_text())


def _record():
    shape = (4, 4)
    partition = np.zeros(shape, dtype=np.int64)
    image = np.zeros(shape, dtype=np.float32)
    return {
        "sample_id": "firewall-fixture",
        "shared_manifest_sha256": "fixture-manifest-sha256",
        "partition_sha256": array_hash(partition),
        "central_image_sha256": array_hash(image),
        "adapter_version": ADAPTER_VERSION,
        "adapter_config_sha256": FROZEN_ADAPTER_SPEC_SHA256,
        "shared_grid": build_grid_spec(shape, config_provenance={"fixture": "firewall"}),
        "geometry_validity": {"orientation_valid": False},
    }


@pytest.mark.parametrize("field", [
    "gt", "ground_truth", "mask", "mask_path", "label", "label_path", "annotation",
    "dice", "iou", "hausdorff", "oracle", "hungarian", "freemask_semantic_output",
    "freemask_auditor", "candidate_bank", "predictive_evidence", "o_fit", "o_select", "o_verify",
])
def test_forbidden_semantic_evidence_is_rejected(field):
    record = _record()
    record[field] = "forbidden"
    partition = np.zeros((4, 4), dtype=np.int64)
    image = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(AdapterContractError):
        adapt_partition(record, partition, image, adapter_spec=SPEC)


def test_unknown_method_field_is_rejected():
    record = _record()
    record["method"] = "CUTS"
    with pytest.raises(AdapterContractError):
        adapt_partition(record, np.zeros((4, 4), dtype=np.int64), np.zeros((4, 4), dtype=np.float32), adapter_spec=SPEC)


def test_nested_forbidden_semantic_evidence_is_rejected():
    record = _record()
    record["geometry_validity"] = {"orientation_valid": False, "oracle_mapping": {"0": "LV"}}
    with pytest.raises(AdapterContractError):
        adapt_partition(record, np.zeros((4, 4), dtype=np.int64), np.zeros((4, 4), dtype=np.float32), adapter_spec=SPEC)
