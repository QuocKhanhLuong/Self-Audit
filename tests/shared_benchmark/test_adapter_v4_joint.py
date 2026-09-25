from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared_benchmark.adapter import adapt_partition
from shared_benchmark.semantic_contract import ADAPTER_V4_VERSION, FROZEN_ADAPTER_V4_SPEC_SHA256, array_hash


ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "benchmark_freezes/cardiac_benchmark_v8/configs/adapter_v4_spec.json").read_text(encoding="utf-8"))
GRID = json.loads((ROOT / "benchmark_freezes/cardiac_benchmark_v7/configs/resolved_shared_contract.json").read_text(encoding="utf-8"))
GRID.pop("shared_grid_sha256")


def _partition(*, gap: bool = False, remote: bool = False, equal_second: bool = False) -> np.ndarray:
    partition = np.zeros((224, 224), dtype=np.int64)
    # Main MYO ring and a two-component LV group.
    partition[50:150, 50:150] = 2
    partition[70:130, 70:100] = 3
    partition[70:130, 100:130] = 4
    # A wide RV has a stronger normalized adjacency score than one-pixel noise.
    partition[75:125, 25:50] = 1
    partition[74, 150] = 8
    if gap:
        # A one-pixel-wide channel crosses the 20-pixel MYO thickness.  It
        # removes strict hole containment but leaves 239/240 LV boundary
        # pixels adjacent to the MYO component.
        partition[50:70, 100] = 0
    if remote:
        partition[10:18, 180:188] = 9
        partition[12:16, 182:186] = 10
    if equal_second:
        partition[50:150, 155:220] = 20
        partition[70:130, 170:205] = 21
    return partition


def _adapt(partition: np.ndarray):
    image = np.zeros(partition.shape, dtype=np.float32)
    record = {
        "sample_id": "v4-joint-fixture",
        "shared_manifest_sha256": "fixture-manifest-sha256",
        "partition_sha256": array_hash(partition),
        "central_image_sha256": array_hash(image),
        "adapter_version": ADAPTER_V4_VERSION,
        "adapter_config_sha256": FROZEN_ADAPTER_V4_SPEC_SHA256,
        "shared_grid": GRID,
        "geometry_validity": {"orientation_valid": False},
    }
    return adapt_partition(record, partition, image, adapter_spec=SPEC)


def test_v4_groups_split_lv_selects_remote_competitor_and_ignores_one_pixel_rv_noise():
    partition = _partition(remote=True)
    result = _adapt(partition)
    assert np.all(result.semantic_map[70:130, 70:130] == 3)
    assert np.all(result.semantic_map[50:150, 50:150][partition[50:150, 50:150] == 2] == 2)
    assert np.all(result.semantic_map[75:125, 25:50] == 1)
    assert result.semantic_map[74, 150] == 4
    assert result.metadata["role_reasons"] == {"BG": "resolved", "LV": "resolved", "MYO": "resolved", "RV": "resolved"}
    assert result.metadata["solver"]["candidate_count"] >= 2


def test_v4_allows_small_myo_gap_without_creating_pixels():
    partition = _partition(gap=True)
    result = _adapt(partition)
    assert np.all(result.semantic_map[50:70, 100] == 0)
    assert np.all(result.semantic_map[70:130, 70:130] == 3)
    assert result.metadata["solver"]["hypotheses"][0]["source"] == "soft_enclosure"


def test_v4_near_equal_lvmayo_hypotheses_leave_only_disputed_foreground_void():
    partition = np.zeros((224, 224), dtype=np.int64)
    for outer_id, inner_id, left in ((2, 3, 20), (4, 5, 125)):
        partition[50:150, left:left + 80] = outer_id
        partition[70:130, left + 20:left + 60] = inner_id
    result = _adapt(partition)
    assert not np.isin(result.semantic_map, [1, 2, 3]).any()
    assert result.metadata["solver"]["near_optimal_count"] == 2
    assert any(reason == "hypothesis_conflict" for reason in result.metadata["void_reasons"].values())


def test_v4_is_invariant_to_raw_id_permutation():
    partition = _partition(gap=True, remote=True)
    baseline = _adapt(partition)
    values = sorted(set(int(value) for value in partition.flat))
    mapping = dict(zip(values, [77, -11, 9, 103, 0, 45, -200, 8, 71, 16, 91]))
    relabeled = np.vectorize(mapping.__getitem__, otypes=[np.int64])(partition)
    result = _adapt(relabeled)
    np.testing.assert_array_equal(result.semantic_map, baseline.semantic_map)
    np.testing.assert_array_equal(result.validity_map, baseline.validity_map)
    assert result.metadata["component_graph_digest"] == baseline.metadata["component_graph_digest"]
    assert result.metadata["scientific_result_sha256"] == baseline.metadata["scientific_result_sha256"]


def test_v4_repeated_execution_is_bit_identical_and_rejects_forbidden_evidence():
    partition = _partition(remote=True)
    first = _adapt(partition)
    second = _adapt(partition.copy())
    np.testing.assert_array_equal(first.semantic_map, second.semantic_map)
    np.testing.assert_array_equal(first.validity_map, second.validity_map)
    assert first.metadata == second.metadata
