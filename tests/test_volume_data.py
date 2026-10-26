import hashlib
import json

import numpy as np
import pytest
import torch

from self_audit.volume.data import VolumeTrainingData
from self_audit.volume.fixtures import make_volume_fixture
from self_audit.volume import data as volume_data


def test_sample_batch_shapes_support_and_unknown255(tmp_path):
    paths = make_volume_fixture(tmp_path / "fixture", unknown_fraction=0.08)
    loader = VolumeTrainingData(*paths, synthetic=True)
    assert loader.max_cached_arrays == 8
    out = loader.sample_batch(0, np.random.default_rng(9), 2, labeled=True, slab_depth=3, crop_size=13)
    assert out["image"].shape == (2, 3, 3, 13, 13)
    assert out["labels"].shape == (2, 3, 13, 13)
    assert out["geometric_valid"].dtype == torch.bool
    assert out["label_valid"].dtype == torch.bool
    assert torch.all(out["label_valid"] <= out["geometric_valid"])
    assert any(int(x) == 255 for x in out["labels"].reshape(-1))
    assert all("current_affine_xyz_to_world_mm" in m and "reversible_transform" in m for m in out["metadata"])


def test_unlabeled_path_returns_no_labels_and_never_reads_masks(tmp_path):
    paths = make_volume_fixture(tmp_path / "fixture")
    loader = VolumeTrainingData(*paths, synthetic=True, max_cached_arrays=1)
    out = loader.sample_batch(0, np.random.default_rng(3), 1, labeled=False, slab_depth=1, crop_size=0)
    assert out["labels"] is None and out["label_valid"].shape == out["geometric_valid"].shape
    assert not bool(out["label_valid"].any())
    assert loader.label_reads == set()
    assert len(loader.cache) <= loader.max_cached_arrays
    assert out["image"].shape[:3] == (1, 3, 1)


@pytest.mark.parametrize("capacity", [0, -1, True, False, 1.5, "8", None])
def test_cache_capacity_must_be_a_positive_non_bool_integer(tmp_path, capacity):
    paths = make_volume_fixture(tmp_path / "fixture")
    with pytest.raises(ValueError, match="max_cached_arrays must be a positive integer"):
        VolumeTrainingData(*paths, synthetic=True, max_cached_arrays=capacity)


def test_bounded_cache_loads_more_fixture_arrays_and_evicts_lru_without_closing_views(tmp_path):
    paths = make_volume_fixture(tmp_path / "fixture", n_labeled=2, n_unlabeled=2)
    loader = VolumeTrainingData(*paths, synthetic=True, max_cached_arrays=2)
    first = loader.labeled[0]
    second = loader.labeled[1]
    third = loader.unlabeled[0]
    payloads = [
        (first["image"], first["image_sha256"]),
        (second["image"], second["image_sha256"]),
        (third["image"], third["image_sha256"]),
    ]
    # Retain a view before eviction: the bounded cache must not close its
    # mmap, and the bytes remain readable after a later cache miss.
    first_view = loader._array(*payloads[0])
    second_view = loader._array(*payloads[1])
    assert len(loader.cache) == 2
    assert list(loader.cache) == [str(loader._resolve(payloads[0][0])), str(loader._resolve(payloads[1][0]))]
    assert loader._array(*payloads[0]) is first_view
    assert list(loader.cache) == [str(loader._resolve(payloads[1][0])), str(loader._resolve(payloads[0][0]))]
    loader._array(*payloads[2])
    assert len(loader.cache) == 2
    assert str(loader._resolve(payloads[1][0])) not in loader.cache
    assert np.asarray(first_view).tobytes() == np.asarray(loader._array(*payloads[0])).tobytes()
    # The second view was evicted and remains a valid independent mmap view;
    # reloading the same payload is a cache miss, not a closed-view failure.
    second_reload = loader._array(*payloads[1])
    assert second_reload is not second_view
    assert np.asarray(second_view).tobytes() == np.asarray(second_reload).tobytes()
    assert len(loader.cache) <= loader.max_cached_arrays


def test_sample_bytes_match_for_capacity_one_and_eight(tmp_path):
    paths = make_volume_fixture(tmp_path / "fixture", unknown_fraction=0.08)
    narrow = VolumeTrainingData(*paths, synthetic=True, max_cached_arrays=1)
    wide = VolumeTrainingData(*paths, synthetic=True, max_cached_arrays=8)
    out_narrow = narrow.sample_batch(0, np.random.default_rng(91), 3, labeled=True,
                                     slab_depth=3, crop_size=13)
    out_wide = wide.sample_batch(0, np.random.default_rng(91), 3, labeled=True,
                                slab_depth=3, crop_size=13)
    for key in ("image", "labels", "geometric_valid", "label_valid"):
        assert out_narrow[key].numpy().tobytes() == out_wide[key].numpy().tobytes()
    assert out_narrow["metadata"] == out_wide["metadata"]
    assert len(narrow.cache) <= 1
    assert len(wide.cache) <= 8


def test_temporal_pair_shares_grid_and_adjacent_frame(tmp_path):
    paths = make_volume_fixture(tmp_path / "fixture")
    loader = VolumeTrainingData(*paths, synthetic=True)
    out = loader.sample_temporal_pair(0, np.random.default_rng(4), labeled=False, slab_depth=3, crop_size=9)
    assert out["image"].shape == out["adjacent_image"].shape
    assert out["metadata"]["patient"] == out["adjacent_metadata"]["patient"]
    assert out["metadata"]["frame"] + 1 == out["adjacent_metadata"]["frame"]
    assert out["metadata"]["z_indices"] == out["adjacent_metadata"]["z_indices"]
    assert out["metadata"]["reversible_transform"]["crop"] == out["adjacent_metadata"]["reversible_transform"]["crop"]


def test_missing_geometry_fails_closed(tmp_path):
    manifest, msha, geometry, gsha = make_volume_fixture(tmp_path / "fixture")
    obj = json.loads(geometry.read_text()); del obj["patients"]["synthetic_volume_000"]["affine_xyz_to_world_mm"]
    geometry.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        VolumeTrainingData(manifest, msha, geometry, gsha, synthetic=True)


def test_sidecar_image_hash_is_bound_before_sampling(tmp_path):
    manifest, msha, geometry, gsha = make_volume_fixture(tmp_path / "fixture")
    obj = json.loads(geometry.read_text())
    obj["patients"]["synthetic_volume_000"]["image_sha256"] = "0" * 64
    geometry.write_text(json.dumps(obj, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="image source hash"):
        VolumeTrainingData(manifest, msha, geometry,
                           hashlib.sha256(geometry.read_bytes()).hexdigest(), synthetic=True)


def test_unlabeled_geometry_rejects_nested_mask_before_raw_validation(tmp_path, monkeypatch):
    manifest, msha, geometry, _ = make_volume_fixture(tmp_path / "fixture")
    obj = json.loads(geometry.read_text())
    # The manifest remains strictly image-only; the forbidden reference is
    # injected only into U geometry to exercise the pre-I/O role gate.
    obj["patients"]["synthetic_volume_002"]["source_image"] = {
        "path": "raw_should_not_be_opened.nii.gz", "sha256": "0" * 64,
        "provenance": {"nested_mask_path": "forbidden_mask.nii.gz"},
    }
    obj["patients"]["synthetic_volume_002"]["mask_sources"] = {
        "0": {"source": {"path": "forbidden_mask0.nii.gz", "sha256": "0" * 64}}
    }
    geometry.write_text(json.dumps(obj, sort_keys=True) + "\n")
    calls = []
    monkeypatch.setattr(volume_data, "validate_nifti_correspondence",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(ValueError, match="forbidden mask reference"):
        VolumeTrainingData(manifest, msha, geometry,
                           hashlib.sha256(geometry.read_bytes()).hexdigest(), synthetic=True)
    assert calls == []


def test_real_loader_requires_matching_sidecar_identity_and_raw_provenance(tmp_path):
    manifest, msha, geometry, _ = make_volume_fixture(tmp_path / "fixture")
    manifest_obj = json.loads(manifest.read_text())
    manifest_obj["synthetic"] = False
    manifest.write_text(json.dumps(manifest_obj, sort_keys=True) + "\n")
    msha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    geometry_obj = json.loads(geometry.read_text())
    geometry_obj["source_manifest_sha256"] = msha
    geometry.write_text(json.dumps(geometry_obj, sort_keys=True) + "\n")
    gsha = hashlib.sha256(geometry.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="synthetic.*identity"):
        VolumeTrainingData(manifest, msha, geometry, gsha, synthetic=False)

    # Even a sidecar relabeled as real cannot fall back to synthetic identity
    # geometry: every record needs verified raw source provenance.
    geometry_obj["synthetic"] = False
    geometry.write_text(json.dumps(geometry_obj, sort_keys=True) + "\n")
    gsha = hashlib.sha256(geometry.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="real geometry requires source_image"):
        VolumeTrainingData(manifest, msha, geometry, gsha, synthetic=False)


def test_real_like_nifti_preparation_and_loader_path_is_reachable(tmp_path):
    nib = pytest.importorskip("nibabel")
    from self_audit.model import digest
    from scripts.volume_prepare import build_sidecar

    manifest, _, geometry, _ = make_volume_fixture(tmp_path / "fixture", shape_tzyx=(2, 2, 3, 4))
    manifest_obj = json.loads(manifest.read_text())
    manifest_obj["synthetic"] = False
    physical = {}
    affine = np.diag([1.3, 0.9, 4.2, 1.0])
    for record in manifest_obj["labeled"] + manifest_obj["unlabeled"]:
        pid = record["id"]
        np.save(tmp_path / "fixture" / record["image"], np.zeros((2, 2, 3, 4), dtype=np.float32))
        record["image_sha256"] = digest(tmp_path / "fixture" / record["image"])
        raw_image = np.zeros((4, 3, 2, 2), dtype=np.float32)
        image_path = tmp_path / "fixture" / f"{pid}.nii.gz"
        image_obj = nib.Nifti1Image(raw_image, affine); image_obj.header.set_xyzt_units("mm", "sec")
        nib.save(image_obj, image_path)
        entry = {
            "affine_xyz_to_world_mm": affine.tolist(), "orientation": "RAS",
            "slice_order": "increasing_native_z_index", "temporal_frame_mapping": [0, 1],
            "source_image": {"path": image_path.name, "sha256": digest(image_path)},
        }
        if "masks" in record:
            mask_sources = {}
            for frame, item in record["masks"].items():
                prepared_path = tmp_path / "fixture" / item["path"]
                np.save(prepared_path, np.zeros((2, 3, 4), dtype=np.uint8))
                item["sha256"] = digest(prepared_path)
                raw_mask_path = tmp_path / "fixture" / f"{pid}_mask_{frame}.nii.gz"
                mask_obj = nib.Nifti1Image(np.zeros((4, 3, 2), dtype=np.uint8), affine)
                mask_obj.header.set_xyzt_units("mm", "sec"); nib.save(mask_obj, raw_mask_path)
                mask_sources[frame] = {"source_path": raw_mask_path.name,
                                       "source_sha256": digest(raw_mask_path)}
            entry["mask_sources"] = mask_sources
        physical[pid] = entry
    manifest.write_text(json.dumps(manifest_obj, sort_keys=True) + "\n")
    msha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    sidecar = build_sidecar(manifest, msha, synthetic=False, physical_metadata=physical)
    geometry.write_text(json.dumps(sidecar, sort_keys=True) + "\n")
    loader = VolumeTrainingData(manifest, msha, geometry,
                                hashlib.sha256(geometry.read_bytes()).hexdigest(), synthetic=False)
    out = loader.sample_batch(0, np.random.default_rng(0), 1, labeled=True, slab_depth=1, crop_size=0)
    assert out["image"].shape == (1, 3, 1, 3, 4)


def test_physical_resampling_is_explicit_and_updates_spacing(tmp_path):
    paths = make_volume_fixture(tmp_path / "fixture")
    loader = VolumeTrainingData(*paths, synthetic=True)
    out = loader.sample_batch(0, np.random.default_rng(12), 1, labeled=True, slab_depth=3, crop_size=7,
                              spacing_xyz_mm=(2.6, 1.8, 8.4))
    spacing = out["metadata"][0]["spacing_xyz_mm"]
    assert np.allclose(spacing, (2.6, 1.8, 8.4), atol=1e-4)
    assert out["geometric_valid"].dtype == torch.bool


def test_labeled_sampling_uses_only_declared_sparse_frames(tmp_path):
    manifest, msha, geometry, gsha = make_volume_fixture(tmp_path / "fixture", shape_tzyx=(10, 3, 9, 11))
    manifest_obj = json.loads(manifest.read_text())
    labeled = manifest_obj["labeled"][0]
    labeled["masks"] = {k: labeled["masks"][k] for k in ("0", "5")}
    manifest.write_text(json.dumps(manifest_obj, sort_keys=True) + "\n")
    # The sidecar is hash-pinned to the manifest; update only that bridge.
    geometry_obj = json.loads(geometry.read_text())
    geometry_obj["source_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    geometry.write_text(json.dumps(geometry_obj, sort_keys=True) + "\n")
    loader = VolumeTrainingData(manifest, geometry_obj["source_manifest_sha256"], geometry,
                                hashlib.sha256(geometry.read_bytes()).hexdigest(), synthetic=True)
    # Repeated draws must never select an unavailable frame (e.g. t=8).
    frames = [loader.sample_batch(0, np.random.default_rng(seed), 1, labeled=True,
                                  slab_depth=1, crop_size=0)["metadata"][0]["frame"]
              for seed in range(40)]
    assert set(frames) <= {0, 5}
    with pytest.raises(ValueError, match="frame index"):
        loader._sample_one(loader.labeled[0], np.random.default_rng(1), t=8, center_z=0,
                           labeled=True, slab_depth=1, crop_size=0,
                           spacing_xyz_mm=None, fov_xyz_mm=None, flip=False)


def test_slab_out_of_grid_padding_is_invalid_and_edge_triplet_is_preserved(tmp_path):
    paths = make_volume_fixture(tmp_path / "fixture", shape_tzyx=(2, 3, 7, 9))
    loader = VolumeTrainingData(*paths, synthetic=True)
    sample = loader._sample_one(loader.labeled[0], np.random.default_rng(2), t=0, center_z=0,
                                labeled=True, slab_depth=5, crop_size=0,
                                spacing_xyz_mm=None, fov_xyz_mm=None, flip=False)
    assert sample["metadata"]["z_indices"] == [-2, -1, 0, 1, 2]
    assert not bool(sample["geometric_valid"][:2].any())
    assert torch.all(sample["labels"][:2] == 255)
    assert torch.all(sample["image"][:, :2] == 0)
    image = loader._record_image(loader.labeled[0])[0]
    assert np.array_equal(sample["image"][:, 2].numpy(), image[0, [0, 0, 1]])
    right = loader._sample_one(loader.labeled[0], np.random.default_rng(2), t=0, center_z=2,
                               labeled=True, slab_depth=5, crop_size=0,
                               spacing_xyz_mm=None, fov_xyz_mm=None, flip=False)
    assert right["metadata"]["z_indices"] == [0, 1, 2, 3, 4]
    assert not bool(right["geometric_valid"][3:].any())
    assert torch.all(right["labels"][3:] == 255)
