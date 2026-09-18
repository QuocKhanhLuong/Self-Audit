from __future__ import annotations

import copy
import shutil

import pytest

from self_audit_maskfree.data.discovery import assign_splits, discover_dataset
from shared_benchmark.manifest import SharedManifestError, build_shared_manifest, validate_manifest, validate_scientific_manifest
from shared_benchmark.spatial import build_grid_spec

from helpers import discovered_projection, write_image


def test_projection_preserves_patient_isolation_inventory_context_and_canonical_order(tmp_path):
    for index in range(7):
        write_image(tmp_path, f"patient{index:03d}.npy", shape=(7, 11, 3, 2), seed=index)
    upstream, projection = discovered_projection(tmp_path)
    assert len(upstream["records"]) == 7 * 3 * 2
    assert len(projection["records"]) == len(upstream["records"])
    patient_splits = {}
    for record in projection["records"]:
        assert record["context_indices"] == [
            max(0, record["slice_index"] - 1), record["slice_index"],
            min(record["depth"] - 1, record["slice_index"] + 1),
        ]
        patient_splits.setdefault(record["patient_id"], record["split"])
        assert patient_splits[record["patient_id"]] == record["split"]
    assert [record["sample_id"] for record in projection["records"]] == sorted(
        record["sample_id"] for record in projection["records"]
    )
    first = next(record for record in projection["records"] if record["slice_index"] == 0)
    last = next(record for record in projection["records"] if record["slice_index"] == 2)
    assert first["context_indices"] == [0, 0, 1]
    assert last["context_indices"] == [1, 2, 2]


def test_projection_hash_and_sample_identity_ignore_root_relocation_and_row_order(tmp_path):
    original = tmp_path / "original"
    write_image(original, "patient001.npy", seed=1)
    write_image(original, "patient002.npy", seed=2)
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    first_upstream = discover_dataset(original, "acdc", seed=42, depth_axis=2)
    second_upstream = discover_dataset(relocated, "acdc", seed=42, depth_axis=2)
    grid = build_grid_spec((8, 8), config_provenance={"source": "test"})
    first = build_shared_manifest(first_upstream, grid, fixture=True, scientific=False, local_source_root=original)
    reordered = copy.deepcopy(second_upstream)
    reordered["records"] = list(reversed(reordered["records"]))
    second = build_shared_manifest(reordered, grid, fixture=True, scientific=False, local_source_root=relocated)
    assert first["manifest_hash"] == second["manifest_hash"]
    assert [row["sample_id"] for row in first["records"]] == [row["sample_id"] for row in second["records"]]


def test_official_mnms_membership_and_unassigned_fallback_are_preserved(tmp_path):
    write_image(tmp_path, "Training/case001.npy", seed=1)
    write_image(tmp_path, "Validation/case002.npy", seed=2)
    write_image(tmp_path, "Testing/case003.npy", seed=3)
    upstream = discover_dataset(tmp_path, "mnms", seed=42, depth_axis=2)
    by_patient = {row["patient_id"]: row["split"] for row in upstream["records"]}
    assert by_patient == {"case001": "train", "case002": "dev", "case003": "test"}
    assigned, provenance = assign_splits(
        ["case001", "case002", "case003", "case004"], dataset="mnms", seed=42,
        official={"case001": "train", "case002": "dev", "case003": "test"},
    )
    assert {key: assigned[key] for key in ("case001", "case002", "case003")} == {
        "case001": "train", "case002": "dev", "case003": "test"
    }
    assert assigned["case004"] == "train"
    assert provenance["rule"] == "official_mnms_folder_membership_plus_rank_hashed_unassigned"


def test_gt_shaped_files_do_not_change_inventory_and_fixture_never_validates_scientifically(tmp_path):
    write_image(tmp_path, "patient001.npy", seed=1)
    before, projection = discovered_projection(tmp_path)
    write_image(tmp_path, "patient001_gt.npy", seed=99)
    after = discover_dataset(tmp_path, "acdc", seed=42, depth_axis=2)
    assert before["manifest_id"] == after["manifest_id"]
    with pytest.raises(SharedManifestError, match="fixture manifest"):
        validate_scientific_manifest(projection, image_root=tmp_path)
    relabeled = copy.deepcopy(projection)
    relabeled["fixture"] = False
    relabeled["scientific"] = True
    with pytest.raises(SharedManifestError, match="payload hash mismatch"):
        validate_manifest(relabeled)
    bad = copy.deepcopy(projection)
    bad["records"][0]["mask_path"] = "forbidden"
    with pytest.raises(SharedManifestError, match="forbidden generation field"):
        validate_manifest(bad)
