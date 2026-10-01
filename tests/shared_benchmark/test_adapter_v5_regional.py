from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared_benchmark.adapter import adapt_partition
from shared_benchmark.adapter_v5 import _retain_nonconflicting_roles
from shared_benchmark.semantic_contract import ADAPTER_V5_VERSION, BG, LV, MYO, RV, FROZEN_ADAPTER_V5_SPEC_SHA256, array_hash


ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "benchmark_freezes/cardiac_benchmark_v9/configs/adapter_v5_spec.json").read_text(encoding="utf-8"))
GRID = json.loads((ROOT / "benchmark_freezes/cardiac_benchmark_v7/configs/resolved_shared_contract.json").read_text(encoding="utf-8"))
GRID.pop("shared_grid_sha256")


def _adapt(partition: np.ndarray):
    image = np.zeros(partition.shape, dtype=np.float32)
    record = {
        "sample_id": "v5-regional-fixture",
        "shared_manifest_sha256": "fixture-manifest-sha256",
        "partition_sha256": array_hash(partition),
        "central_image_sha256": array_hash(image),
        "adapter_version": ADAPTER_V5_VERSION,
        "adapter_config_sha256": FROZEN_ADAPTER_V5_SPEC_SHA256,
        "shared_grid": GRID,
        "geometry_validity": {"orientation_valid": False},
    }
    return adapt_partition(record, partition, image, adapter_spec=SPEC)


def _ring(partition: np.ndarray, *, outer_id: int, inner_id: int, left: int) -> None:
    partition[50:150, left:left + 80] = outer_id
    partition[70:130, left + 20:left + 60] = inner_id


def test_v5_retains_best_local_lvmayo_labels_when_an_alternative_omits_them():
    partition = np.zeros((224, 224), dtype=np.int64)
    _ring(partition, outer_id=2, inner_id=3, left=20)
    _ring(partition, outer_id=4, inner_id=5, left=125)

    result = _adapt(partition)

    # The deterministic tie-break selects the right ring.  The left
    # near-optimal ring omits its components, yet that absence no longer
    # voids the selected right-ring anatomy.
    assert np.all(result.semantic_map[70:130, 145:185] == 3)
    assert np.all(result.semantic_map[50:150, 125:205][partition[50:150, 125:205] == 4] == 2)
    assert result.metadata["solver"]["near_optimal_count"] == 2
    assert result.metadata["solver"]["conflict_policy"] == "regional_role_only"


def test_v5_voids_only_a_component_with_a_contradictory_near_optimal_role():
    assert _retain_nonconflicting_roles(
        {0: BG, 1: MYO, 2: LV, 3: RV},
        [{0: BG, 1: MYO, 2: MYO}],
    ) == {0: BG, 1: MYO, 3: RV}


def test_v5_merges_connected_rv_fragments_that_individually_fail_the_threshold():
    partition = np.zeros((224, 224), dtype=np.int64)
    _ring(partition, outer_id=2, inner_id=3, left=50)
    partition[75:100, 25:50] = 6
    partition[100:125, 25:50] = 7

    result = _adapt(partition)

    assert np.all(result.semantic_map[75:125, 25:50] == 1)
    assert result.metadata["role_reasons"]["RV"] == "resolved"
    assert len(result.metadata["solver"]["rv"]["candidates"][0]["component_indices"]) == 2


def test_v5_is_cluster_id_permutation_invariant_and_repeatable():
    partition = np.zeros((224, 224), dtype=np.int64)
    _ring(partition, outer_id=2, inner_id=3, left=50)
    partition[75:100, 25:50] = 6
    partition[100:125, 25:50] = 7
    first = _adapt(partition)
    relabeled = np.select(
        [partition == 0, partition == 2, partition == 3, partition == 6, partition == 7],
        [91, -4, 83, 16, 48],
        default=71,
    ).astype(np.int64)
    second = _adapt(relabeled)

    np.testing.assert_array_equal(second.semantic_map, first.semantic_map)
    np.testing.assert_array_equal(second.validity_map, first.validity_map)
    assert second.metadata["component_graph_digest"] == first.metadata["component_graph_digest"]
    assert second.metadata["scientific_result_sha256"] == first.metadata["scientific_result_sha256"]
