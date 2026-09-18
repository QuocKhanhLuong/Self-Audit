from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared_benchmark.adapter import adapt_partition
from shared_benchmark.semantic_contract import array_hash
from shared_benchmark.spatial import build_grid_spec

from test_adapter_frozen_fixtures import _fixture_input


ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "benchmark_freezes/cardiac_benchmark_v1/configs/adapter_v1_spec.json").read_text())
FIXTURE = json.loads((ROOT / "benchmark_freezes/cardiac_benchmark_v1/configs/adapter_v1_synthetic_fixtures.json").read_text())["fixtures"][0]


def _run(partition: np.ndarray):
    record, _, image = _fixture_input(FIXTURE)
    record["shared_grid"] = build_grid_spec(partition.shape, config_provenance={"fixture": "permutation"})
    record["partition_sha256"] = array_hash(partition)
    return adapt_partition(record, partition, image, adapter_spec=SPEC)


def test_many_raw_id_bijections_preserve_semantic_and_validity_outputs():
    _, partition, _ = _fixture_input(FIXTURE)
    baseline = _run(partition)
    baseline_graph = baseline.metadata["component_graph_digest"]
    baseline_scientific = baseline.metadata["scientific_result_sha256"]
    values = sorted(set(int(value) for value in partition.flat))
    permutations = [values[::-1], [100, -8, 77, 0, 12], [0, 77, -8, 100, 42]]
    for replacement in permutations:
        mapping = dict(zip(values, replacement))
        relabeled = np.vectorize(mapping.__getitem__, otypes=[np.int64])(partition)
        result = _run(relabeled)
        np.testing.assert_array_equal(result.semantic_map, baseline.semantic_map)
        np.testing.assert_array_equal(result.validity_map, baseline.validity_map)
        assert result.metadata["component_graph_digest"] == baseline_graph
        assert result.metadata["semantic_map_sha256"] == baseline.metadata["semantic_map_sha256"]
        assert result.metadata["validity_map_sha256"] == baseline.metadata["validity_map_sha256"]
        assert result.metadata["scientific_result_sha256"] == baseline_scientific
