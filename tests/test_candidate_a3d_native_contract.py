"""Focused native-volume contracts for Candidate A-3D.

These tests use manufactured arrays only.  They verify the geometry and
measurement protocol; they are not patient efficacy evidence.
"""

import json

import numpy as np
import pytest
import torch

from self_audit.candidate_a3d.bank import BankConfig3D, QueryBank3D
from self_audit.candidate_a3d.data import CandidateA3DData, synthetic_fixture
from self_audit.candidate_a3d.geometry import (
    VolumeGeometry3D,
    canonical_geometry,
    require_full_volume_shape,
)
from self_audit.model import digest
from self_audit.data import write_json


def test_boundary_distance_is_physical_and_excludes_fov_edges():
    bank = QueryBank3D(BankConfig3D(classes=(1,), radius_mm=1.0))
    shape = (3, 3, 3)
    spacing = (2.0, 1.0, 0.5)

    # One central voxel has nearest continuous voxel face 0.25 mm away in x,
    # not one voxel centre away.  The radius is one, so the normalized target
    # is exactly -0.25.
    central = torch.zeros(shape, dtype=torch.bool)
    central[1, 1, 1] = True
    assert bank._signed_distance(central, (1, 1, 1), spacing, 1.0) == pytest.approx(-0.25)
    assert bank._signed_distance(central, (0, 1, 1), spacing, 1.0) == pytest.approx(1.0)

    absent = torch.zeros(shape, dtype=torch.bool)
    full = torch.ones(shape, dtype=torch.bool)
    assert bank._signed_distance(absent, (1, 1, 1), spacing, 1.0) == 1.0
    assert bank._signed_distance(full, (1, 1, 1), spacing, 1.0) == -1.0
    assert bank.boundary_status(absent) == "absent"
    assert bank.boundary_status(full) == "full_occupied_no_boundary"

    labels = torch.ones((1, *shape), dtype=torch.long)
    geometry = {
        "shape": shape,
        "spacing_zyx": spacing,
        "affine_zyx": np.diag((*spacing, 1.0)).tolist(),
        "context": "full_volume",
    }
    sdf = bank.sdf_targets(labels, geometry)
    assert torch.all(sdf[:, 1] == -1.0)
    assert torch.all(sdf[:, 0] == 1.0)

    # A class that touches z=0 and z=2 still has only internal interfaces.
    # FOV planes at coordinate 0 and 6 are never emitted as anatomy.
    touching = np.zeros(shape, dtype=bool)
    touching[:, 1, 1] = True
    faces = bank._boundary_faces(touching, spacing)
    assert faces
    for axis, coordinate, *_ in faces:
        assert 0.0 < coordinate < shape[axis] * spacing[axis]
    assert bank.boundary_status(touching) == "interface"


def test_region_fraction_is_voxel_volume_weighted_and_spacing_independent():
    bank = QueryBank3D(BankConfig3D(classes=(1,), region_grid=(1, 1, 1), region_scales=(1.0,), types=("REGION",)))
    labels = torch.zeros((1, 2, 3, 4), dtype=torch.long)
    labels[:, :, :2, :] = 1
    query = next(iter(bank))
    first = float(bank.targets(labels, (3.0, 2.0, 0.5))[0, query.index])
    second = float(bank.targets(labels, (1.0, 1.0, 1.0))[0, query.index])
    assert first == pytest.approx(2.0 / 3.0)
    assert second == pytest.approx(first)


def test_runtime_geometry_is_exact_json_safe_and_rejects_crop():
    geometry = VolumeGeometry3D(
        shape=(2, 3, 4), spacing_zyx=(2.0, 1.0, 0.5),
        affine_zyx=((2.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                   (0.0, 0.0, 0.5, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    payload = geometry.as_dict()
    assert set(payload) == {"shape", "spacing_zyx", "affine_zyx", "context"}
    json.dumps(payload, allow_nan=False)
    assert canonical_geometry(payload)["context"] == "full_volume"
    with pytest.raises(ValueError, match="native full-volume"):
        require_full_volume_shape(geometry, (1, 3, 4))
    with pytest.raises(ValueError, match="exactly"):
        VolumeGeometry3D.from_mapping({**payload, "crop_origin": (0, 0, 0)})


def test_frame_loader_returns_full_volume_and_rejects_crop_or_ignore(tmp_path):
    fixture = synthetic_fixture(
        tmp_path / "fixture", n_labeled=2, n_unlabeled=1, n_evaluation=1,
        shape_tzyx=(2, 3, 4, 5),
    )
    data = CandidateA3DData(fixture["train"], fixture["train_sha256"], synthetic=True)
    keys = data.labeled_frame_keys()
    assert keys == [(0, 0), (0, 1), (1, 0), (1, 1)]
    image, labels, geometry, metadata = data.load_labeled_frame(keys[0])
    assert tuple(image.shape) == (1, 1, 3, 4, 5)
    assert tuple(labels.shape) == (1, 3, 4, 5)
    assert metadata == {"patient": "train_labeled_000", "frame": 0}
    assert geometry["context"] == "full_volume"
    json.dumps(geometry, allow_nan=False)
    assert not any("masks" in record for record in data.unlabeled)

    # Context metadata is not a trusted opt-in for cropped tensors.
    cropped_manifest = fixture["train"].parent / "cropped.json"
    manifest = json.loads(fixture["train"].read_text())
    patient = manifest["labeled"][0]["id"]
    manifest["geometry"]["by_patient"][patient]["context"] = "cropped"
    write_json(cropped_manifest, manifest)
    with pytest.raises(ValueError, match="full_volume"):
        CandidateA3DData(cropped_manifest, digest(cropped_manifest), synthetic=True)
    with pytest.raises(ValueError, match="full_volume"):
        VolumeGeometry3D.from_cache_geometry(
            manifest["geometry"]["by_patient"][patient], expected_shape=(3, 4, 5)
        )

    # Ignore labels are rejected before a training frame can enter a rollout.
    ignore_manifest = fixture["train"].parent / "ignore.json"
    manifest = json.loads(fixture["train"].read_text())
    mask_record = manifest["labeled"][0]["masks"]["0"]
    mask_path = fixture["train"].parent / mask_record["path"]
    mask = np.load(mask_path, allow_pickle=False)
    mask = np.array(mask, copy=True)
    mask[0, 0, 0] = 255
    np.save(mask_path, mask)
    mask_record["sha256"] = digest(mask_path)
    write_json(ignore_manifest, manifest)
    ignore_data = CandidateA3DData(ignore_manifest, digest(ignore_manifest), synthetic=True)
    with pytest.raises(ValueError, match="ignore=255"):
        ignore_data.labeled_frame_keys()
