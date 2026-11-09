"""Synthetic native-volume bank/packet contracts; no patient evidence."""

import math

import pytest
import torch

from self_audit.candidate_a3d.bank import BankConfig3D, QueryBank3D
from self_audit.candidate_a3d.codec import MeasurementCodec3D


def tiny_bank(*, types=("REGION", "BOUNDARY")):
    return QueryBank3D(BankConfig3D(
        region_grid=(1, 1, 1), boundary_grid=(2, 1, 1),
        region_scales=(1.0,), radius_mm=5.0, types=types,
    ))


def geometry(shape=(3, 4, 5)):
    return {
        "shape": shape,
        "spacing_zyx": (2.0, 1.0, 0.5),
        "affine_zyx": ((2.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                       (0.0, 0.0, 0.5, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "context": "full_volume",
    }


def test_query_ids_descriptors_and_native_footprints_are_reproducible():
    first, second = tiny_bank(), tiny_bank()
    assert [q.query_id for q in first] == [q.query_id for q in second]
    assert first.fingerprint == second.fingerprint
    descriptors = first.descriptors(dtype=torch.float64)
    assert descriptors.shape == (len(first), 9)
    assert torch.isfinite(descriptors).all()
    for q in first:
        if q.kind == "REGION":
            slices = first.region_slices(q, (3, 4, 5))
            assert slices == (slice(0, 3), slice(0, 4), slice(0, 5))
        else:
            point = first.point_index(q, (3, 4, 5))
            assert point[1:] == (2, 2) and point[0] in {1, 2}


def test_targets_use_anisotropic_spacing_and_absent_class_plus_one():
    bank = tiny_bank()
    labels = torch.zeros((1, 3, 4, 5), dtype=torch.long)
    labels[:, 1, 1:3, 1:4] = 1
    targets = bank.targets(labels, (2.0, 1.0, 0.5))
    assert targets.shape == (1, len(bank))
    assert torch.isfinite(targets).all()
    for q in bank:
        if q.kind == "BOUNDARY" and q.class_id == 3:
            assert targets[0, q.index].item() == 1.0
        elif q.kind == "REGION":
            expected = 6 / 60 if q.class_id == 1 else 0.0
            assert math.isclose(float(targets[0, q.index]), expected, abs_tol=1e-6)
    # The point at z=0,y=2,x=2 is inside class 1 and therefore signed negative.
    boundary_class1 = next(q for q in bank if q.kind == "BOUNDARY" and q.class_id == 1 and q.z < 0.75)
    assert targets[0, boundary_class1.index] < 0


def test_invalid_geometry_and_labels_fail_closed():
    bank = tiny_bank()
    labels = torch.zeros((1, 3, 4, 5), dtype=torch.long)
    with pytest.raises(ValueError, match="invalid"):
        bank.targets(labels + 255, (2.0, 1.0, 0.5))
    with pytest.raises(ValueError, match="spacing"):
        bank.targets(labels, (2.0, -1.0, 0.5))


def test_packet_roundtrip_is_float32_geometry_and_budget_exact():
    bank = tiny_bank()
    codec = MeasurementCodec3D(bank, value_bits=5, base_ids=(0,))
    g = geometry()
    ids = (0, 1)
    levels = (17, 4)
    payload = codec.serialize(ids, levels, g["shape"], g["spacing_zyx"], affine_zyx=g["affine_zyx"])
    packet = codec.deserialize(payload)
    assert packet.ids == ids
    assert packet.shape == g["shape"]
    assert packet.spacing_zyx == tuple(float(v) for v in torch.tensor(g["spacing_zyx"], dtype=torch.float32))
    assert codec.packet_bits(payload) == len(payload) * 8 if hasattr(codec, "packet_bits") else True
    restored = codec.serialize(packet.ids, packet.levels, packet.shape, packet.spacing_zyx, affine_zyx=packet.affine_zyx)
    assert restored == payload
    assert codec.cost(ids, shape=g["shape"], spacing=g["spacing_zyx"], affine_zyx=g["affine_zyx"]) == len(payload) * 8
    with pytest.raises(ValueError, match="budget"):
        codec.serialize(ids, levels, g["shape"], g["spacing_zyx"], affine_zyx=g["affine_zyx"], budget_bits=len(payload) * 8 - 1)
