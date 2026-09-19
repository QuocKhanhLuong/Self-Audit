from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared_benchmark.adapter import adapt_partition
from shared_benchmark.provenance import sha256_file
from shared_benchmark.semantic_contract import ADAPTER_VERSION, FROZEN_ADAPTER_SPEC_SHA256, array_hash
from shared_benchmark.spatial import build_grid_spec


ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "benchmark_freezes/cardiac_benchmark_v2"
SPEC = json.loads((FREEZE / "configs/adapter_v1_spec.json").read_text())
FIXTURES = json.loads((FREEZE / "configs/adapter_v1_synthetic_fixtures.json").read_text())


def _fixture_input(fixture: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    mapping = fixture["raw_id_by_symbol"]
    partition = np.asarray([[mapping[symbol] for symbol in row] for row in fixture["partition_grid"]], dtype=np.int64)
    shape = partition.shape
    image_spec = fixture["central_image"]
    if image_spec["kind"] == "constant":
        image = np.full(shape, float(image_spec["value"]), dtype=np.float32)
    else:
        image = np.linspace(float(image_spec["start"]), float(image_spec["stop"]), num=partition.size, dtype=np.float32).reshape(shape)
    record = {
        "sample_id": f"fixture:{fixture['id']}",
        "shared_manifest_sha256": "fixture-manifest-sha256",
        "partition_sha256": array_hash(partition),
        "central_image_sha256": array_hash(image),
        "adapter_version": ADAPTER_VERSION,
        "adapter_config_sha256": FROZEN_ADAPTER_SPEC_SHA256,
        "shared_grid": build_grid_spec(shape, config_provenance={"fixture": fixture["id"]}),
        "geometry_validity": fixture["geometry"],
    }
    return record, partition, image


def _expected_arrays(fixture: dict) -> tuple[np.ndarray, np.ndarray]:
    labels = {"B": 0, "R": 1, "M": 2, "L": 3, "V": 4}
    semantic = np.asarray([[labels[symbol] for symbol in row] for row in fixture["expected_semantic_grid"]], dtype=np.uint8)
    validity = np.asarray([[symbol == "1" for symbol in row] for row in fixture["expected_validity_grid"]], dtype=bool)
    return semantic, validity


def test_all_frozen_topology_fixtures_match_exact_maps_and_validity():
    for fixture in FIXTURES["fixtures"]:
        record, partition, image = _fixture_input(fixture)
        result = adapt_partition(record, partition, image, adapter_spec=SPEC)
        expected_semantic, expected_validity = _expected_arrays(fixture)
        np.testing.assert_array_equal(result.semantic_map, expected_semantic, err_msg=fixture["id"])
        np.testing.assert_array_equal(result.validity_map, expected_validity, err_msg=fixture["id"])
        assert result.metadata["intensity_resolution"] == "disabled"
        assert result.metadata["orientation_resolution"] == "disabled"
        for expected_reason in fixture["expected_unresolved_reasons"].values():
            assert expected_reason in set(result.metadata["role_reasons"].values()) | set(result.metadata["void_reasons"].values()), fixture["id"]


def test_frozen_spec_and_fixture_hashes_are_unchanged():
    assert sha256_file(FREEZE / "configs/adapter_v1_spec.json") == "34b1faeb7b40f77c2d4d9789a6e1a342e891fcb8ff8db4edd30957b2ae9d114a"
    payload = json.loads((FREEZE / "FREEZE_MANIFEST.json").read_text())["scientific_payload"]
    assert sha256_file(FREEZE / "configs/adapter_v1_synthetic_fixtures.json") == payload["adapter"]["synthetic_fixture_set_sha256"]
