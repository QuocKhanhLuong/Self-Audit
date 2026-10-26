import json
import hashlib

import numpy as np
import pytest

from self_audit.volume.fixtures import make_volume_fixture
from self_audit.volume.geometry import (
    AXIS_ORDER,
    SCHEMA,
    invert_probabilities,
    load_sidecar,
    sha256_file,
    spacing_from_affine,
    validate_geometry_record,
    voxel_to_world,
    world_to_voxel,
    validate_nifti_correspondence,
)


def test_sidecar_hash_and_affine_roundtrip(tmp_path):
    manifest, msha, geometry, gsha = make_volume_fixture(tmp_path / "fixture")
    sidecar = load_sidecar(geometry, gsha)
    assert sidecar["schema"] == SCHEMA
    assert sidecar["axis_order"] == AXIS_ORDER
    rec = sidecar["patients"]["synthetic_volume_000"]
    assert rec["native_shape_tzyx"] == [2, 5, 17, 19]
    p = np.array([2.5, 4.0, 1.25])
    assert np.allclose(world_to_voxel(rec["affine_xyz_to_world_mm"], voxel_to_world(rec["affine_xyz_to_world_mm"], p)), p)
    assert np.allclose(spacing_from_affine(rec["affine_xyz_to_world_mm"]), [1.3, 0.9, 4.2])


def test_geometry_rejects_missing_or_inconsistent_physical_fields():
    with pytest.raises(ValueError, match="affine"):
        validate_geometry_record({"native_shape_tzyx": [1, 2, 3, 4]})
    a = np.eye(4).tolist()
    with pytest.raises(ValueError, match="spacing"):
        validate_geometry_record({"native_shape_tzyx": [1, 2, 3, 4], "affine_xyz_to_world_mm": a,
                                  "spacing_xyz_mm": [2, 1, 1], "units": "mm"})
    with pytest.raises(ValueError, match="orientation"):
        validate_geometry_record({"native_shape_tzyx": [1, 2, 3, 4], "affine_xyz_to_world_mm": a,
                                  "units": "mm", "slice_order": "increasing_native_z_index",
                                  "temporal_frame_mapping": [0]})


def test_invert_probabilities_is_native_grid_and_conservative(tmp_path):
    *_, geometry, gsha = make_volume_fixture(tmp_path / "fixture")
    # Metadata mirrors the loader's reversible transform, including a slab
    # whose crop is outside the native XY extent because of edge padding.
    metadata = {
        "native_shape_zyx": [3, 5, 7],
        "reversible_transform": {
            "z_indices": [0, 1, 2], "current_shape_zyx": [3, 5, 7],
            "crop": {"top": 0, "left": 0, "height": 5, "width": 7, "pad_lrtb": [0, 0, 0, 0]},
            "flip": True,
        },
    }
    p = np.zeros((4, 3, 5, 7), np.float32); p[2] = 1
    out, support = invert_probabilities(p, metadata)
    assert out.shape == (4, 3, 5, 7)
    assert support.shape == (3, 5, 7) and support.all()
    assert np.allclose(out[2], 1)


def test_fixture_manifest_has_no_u_mask_reference(tmp_path):
    manifest, *_ = make_volume_fixture(tmp_path / "fixture")
    m = json.loads(manifest.read_text())
    assert all(set(r) == {"id", "image", "image_sha256"} for r in m["unlabeled"])


def test_resampling_requires_full_probability_stencil_support_and_normalizes(tmp_path):
    # Native width 5 is represented on a width-3 current grid with spacing X=2.
    # Only current x=1:3 is emitted (crop x=1..2).  Native x=1 and x=3 are
    # half-supported interpolation locations; nearest-neighbour support would
    # incorrectly mark them valid.
    metadata = {
        "native_shape_zyx": [1, 3, 5],
        "reversible_transform": {
            "z_indices": [0], "current_shape_zyx": [1, 3, 3],
            "crop": {"top": 0, "left": 1, "height": 3, "width": 2, "pad_lrtb": [0, 0, 0, 0]},
            "resample": {
                "native_to_current_matrix_zyx": np.diag([1.0, 1.0, 2.0]).tolist(),
                "native_to_current_offset_zyx": [0.0, 0.0, 0.0],
            },
            "geometric_valid": np.ones((1, 3, 2), dtype=bool).tolist(),
        },
    }
    probabilities = np.zeros((2, 1, 3, 2), dtype=np.float32)
    probabilities[0] = 2.0
    probabilities[1] = 1.0
    native, support = invert_probabilities(probabilities, metadata)
    assert support.shape == (1, 3, 5)
    assert not support[0, 1, [0, 1]].any()
    assert support[0, 1, [2, 3, 4]].all()
    assert np.allclose(native[0][support], 2 / 3)
    assert np.allclose(native[1][support], 1 / 3)
    assert np.all(native[:, ~support] == 0)


def test_physical_nifti_frame_bridge_checks_hash_units_orientation_and_voxels(tmp_path):
    nib = pytest.importorskip("nibabel")
    affine = np.diag([1.2, 0.8, 3.5, 1.0])
    raw = np.arange(3 * 4 * 2 * 2, dtype=np.float32).reshape(3, 4, 2, 2)
    image_path = tmp_path / "cine.nii.gz"
    image = nib.Nifti1Image(raw, affine)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, image_path)
    frame = raw[..., 0]
    frame_path = tmp_path / "frame0.nii.gz"
    frame_img = nib.Nifti1Image(frame, affine)
    frame_img.header.set_xyzt_units("mm", "sec")
    nib.save(frame_img, frame_path)
    mask = np.zeros_like(frame, dtype=np.uint8); mask[1, 2, 0] = 2
    mask_path = tmp_path / "mask0.nii.gz"
    mask_img = nib.Nifti1Image(mask, affine)
    mask_img.header.set_xyzt_units("mm", "sec")
    nib.save(mask_img, mask_path)
    rec = validate_nifti_correspondence(
        image_path, image_sha256=sha256_file(image_path), expected_shape_tzyx=(2, 2, 4, 3),
        expected_affine_xyz_to_world_mm=affine, expected_orientation="RAS",
        expected_frame_mapping=[0, 1], masks={"0": {
            "source_path": str(mask_path), "source_sha256": sha256_file(mask_path),
            "frame_path": str(frame_path), "frame_sha256": sha256_file(frame_path),
        }}, base_dir=tmp_path,
    )
    assert rec["units"] == "mm" and rec["frame_mapping"] == [0, 1]
    bad = raw[..., 1].copy(); nib.save(nib.Nifti1Image(bad, affine), frame_path)
    with pytest.raises(ValueError, match="voxel correspondence"):
        validate_nifti_correspondence(
            image_path, image_sha256=sha256_file(image_path), expected_shape_tzyx=(2, 2, 4, 3),
            expected_affine_xyz_to_world_mm=affine, expected_orientation="RAS",
            expected_frame_mapping=[0, 1], masks={"0": {
                "source_path": str(mask_path), "source_sha256": sha256_file(mask_path),
                "frame_path": str(frame_path), "frame_sha256": sha256_file(frame_path),
            }}, base_dir=tmp_path,
        )


def test_physical_nifti_binds_prepared_image_voxels(tmp_path):
    nib = pytest.importorskip("nibabel")
    affine = np.diag([1.2, 0.8, 3.5, 1.0])
    raw = np.arange(3 * 4 * 2 * 2, dtype=np.float32).reshape(3, 4, 2, 2)
    image_path = tmp_path / "cine_prepared.nii.gz"
    image = nib.Nifti1Image(raw, affine)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, image_path)
    lo, hi = np.percentile(raw, [0.5, 99.5])
    prepared = np.clip((raw - lo) / max(float(hi - lo), 1e-6), 0, 1)
    prepared = prepared.transpose(3, 2, 1, 0).astype(np.float32)
    validate_nifti_correspondence(
        image_path, image_sha256=sha256_file(image_path), expected_shape_tzyx=(2, 2, 4, 3),
        expected_affine_xyz_to_world_mm=affine, expected_orientation="RAS",
        expected_frame_mapping=[0, 1], base_dir=tmp_path,
        prepared_image_tzyx=prepared,
    )
    prepared_bad = prepared.copy(); prepared_bad[0, 0, 0, 0] += 0.25
    with pytest.raises(ValueError, match="prepared image/raw cine"):
        validate_nifti_correspondence(
            image_path, image_sha256=sha256_file(image_path), expected_shape_tzyx=(2, 2, 4, 3),
            expected_affine_xyz_to_world_mm=affine, expected_orientation="RAS",
            expected_frame_mapping=[0, 1], base_dir=tmp_path,
            prepared_image_tzyx=prepared_bad,
        )


def test_real_manifest_cannot_use_synthetic_identity_or_unpinned_source(tmp_path):
    manifest, msha, _, _ = make_volume_fixture(tmp_path / "fixture")
    obj = json.loads(manifest.read_text())
    obj["synthetic"] = False
    manifest.write_text(json.dumps(obj, sort_keys=True) + "\n")
    msha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    from scripts.volume_prepare import build_sidecar
    with pytest.raises(ValueError, match="synthetic"):
        build_sidecar(manifest, msha, synthetic=True)
    with pytest.raises(ValueError, match="physical affine"):
        build_sidecar(manifest, msha, synthetic=False)
