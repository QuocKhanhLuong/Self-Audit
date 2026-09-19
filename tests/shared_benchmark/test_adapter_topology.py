from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared_benchmark.adapter import adapt_partition
from shared_benchmark.region_graph import build_region_graph
from shared_benchmark.semantic_contract import ADAPTER_VERSION, FROZEN_ADAPTER_SPEC_SHA256, array_hash
from shared_benchmark.spatial import build_grid_spec


ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "benchmark_freezes/cardiac_benchmark_v2/configs/adapter_v1_spec.json").read_text())


def _adapt(partition: np.ndarray):
    partition = partition.astype(np.int64)
    image = np.zeros(partition.shape, dtype=np.float32)
    record = {
        "sample_id": "topology-fixture",
        "shared_manifest_sha256": "fixture-manifest-sha256",
        "partition_sha256": array_hash(partition),
        "central_image_sha256": array_hash(image),
        "adapter_version": ADAPTER_VERSION,
        "adapter_config_sha256": FROZEN_ADAPTER_SPEC_SHA256,
        "shared_grid": build_grid_spec(partition.shape, config_provenance={"fixture": "topology"}),
        "geometry_validity": {"orientation_valid": False},
    }
    return adapt_partition(record, partition, image, adapter_spec=SPEC)


def test_region_graph_uses_four_connectivity_and_four_neighbor_adjacency():
    partition = np.asarray([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.int64)
    graph = build_region_graph(partition)
    assert len(graph.components) == 9
    assert all(component.area == 1 for component in graph.components)
    assert sorted(sum(length for _, length in component.adjacency) for component in graph.components) == [2, 2, 2, 2, 3, 3, 3, 3, 4]


def test_cardinality_is_not_semantic_evidence_and_missing_boundaries_are_void():
    for k in (1, 2, 3, 4, 5, 10):
        width = max(2, k)
        values = list(range(k)) + [k - 1] * (2 * width - k)
        partition = np.asarray(values, dtype=np.int64).reshape(2, width)
        result = _adapt(partition)
        assert result.semantic_map.shape == partition.shape
        if k < 3:
            assert not bool(np.isin(result.semantic_map, [1, 2, 3]).any())


def test_border_tie_is_void_not_first_candidate():
    partition = np.asarray([[1, 2], [1, 2]], dtype=np.int64)
    result = _adapt(partition)
    assert np.all(result.semantic_map == 4)
    assert result.metadata["role_reasons"]["BG"] == "ambiguous_bg"


def test_same_raw_id_nonborder_component_is_not_promoted_to_background():
    partition = np.asarray([
        [7, 7, 7, 7, 7],
        [7, 3, 3, 3, 7],
        [7, 3, 7, 3, 7],
        [7, 3, 3, 3, 7],
        [7, 7, 7, 7, 7],
    ], dtype=np.int64)
    partition[0, 1] = 3
    result = _adapt(partition)
    assert result.semantic_map[0, 0] == 0
    assert result.semantic_map[2, 2] == 4


def test_repeated_execution_is_bit_identical():
    partition = np.asarray([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=np.int64)
    first = _adapt(partition)
    second = _adapt(partition.copy())
    np.testing.assert_array_equal(first.semantic_map, second.semantic_map)
    np.testing.assert_array_equal(first.validity_map, second.validity_map)
    assert first.metadata == second.metadata
