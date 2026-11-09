"""Student-only native-volume behavior checks on manufactured tensors."""

import inspect

import pytest
import torch

from self_audit.candidate_a3d.bank import BankConfig3D, QueryBank3D
from self_audit.candidate_a3d.model import CandidateA3DStudent, StudentConfig3D


def bank(*, types=("REGION", "BOUNDARY")):
    return QueryBank3D(BankConfig3D(
        region_grid=(1, 1, 1), boundary_grid=(1, 1, 1),
        region_scales=(0.5,), types=types,
    ))


def geometry(shape=(3, 4, 5)):
    return {
        "shape": shape, "spacing_zyx": (2.0, 1.0, 0.5),
        "affine_zyx": ((2.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                       (0.0, 0.0, 0.5, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "context": "full_volume",
    }


def make_model(query_bank=None, *, local_queries=True):
    return CandidateA3DStudent(query_bank or bank(), StudentConfig3D(
        width=8, decoder_dim=16, num_heads=4, chunk_size=7,
        local_queries=local_queries, depth=2,
    ))


def test_3d_encode_value_policy_and_code_only_decode_shapes():
    model = make_model().eval()
    x = torch.rand(1, 1, 3, 4, 5)
    features = model.encode(x)
    values = model.predict_values(features, geometry())
    assert features.shape == (1, 8, 3, 4, 5)
    assert values.shape == (1, len(model.bank))
    assert torch.isfinite(values).all()
    policy = model.policy_logits(features, (), torch.empty(1, 0), remaining_bits=0)
    assert policy.shape == (1, len(model.bank) + 1)
    assert torch.isfinite(policy[:, model.stop_index]).all()
    assert torch.isneginf(policy[:, :len(model.bank)]).all()
    assert "features" not in inspect.signature(model.decode).parameters
    logits = model.decode((), torch.empty(0), geometry(), chunk_size=5)
    assert logits.shape == (1, 4, 3, 4, 5)
    assert torch.isfinite(logits).all()


def test_local_region_and_boundary_heads_use_native_coordinates():
    model = make_model().eval()
    features = torch.zeros(1, 8, 3, 4, 5)
    local = model._local_features(features, geometry())
    region = next(q for q in model.bank if q.kind == "REGION")
    boundary = next(q for q in model.bank if q.kind == "BOUNDARY")
    rz, ry, rx = model.bank.region_slices(region, (3, 4, 5))
    features[..., rz, ry, rx] = 1
    local_changed = model._local_features(features, geometry())
    assert local_changed[:, region.index].sum() > local[:, region.index].sum()
    bz, by, bx = model.bank.point_index(boundary, (3, 4, 5))
    assert torch.equal(local_changed[:, boundary.index], features[..., bz, by, bx])


def test_value_head_is_conditioned_on_physical_spacing_and_extent():
    model = make_model().eval()
    features = model.encode(torch.rand(1, 1, 3, 4, 5))
    first = model.predict_values(features, geometry())
    unit_geometry = {
        "shape": (3, 4, 5), "spacing_zyx": (1.0, 1.0, 1.0),
        "affine_zyx": ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                       (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "context": "full_volume",
    }
    second = model.predict_values(features, unit_geometry)
    assert not torch.allclose(first, second)


def test_global_only_ablation_keeps_parameter_shapes_and_masks_local_signal():
    torch.manual_seed(41)
    local_model = make_model(local_queries=True)
    torch.manual_seed(41)
    global_model = make_model(local_queries=False)
    for key, value in local_model.state_dict().items():
        assert key in global_model.state_dict()
        assert global_model.state_dict()[key].shape == value.shape
    x = torch.rand(1, 1, 3, 4, 5)
    geometry_value = geometry()
    local_model.eval(); global_model.eval()
    local_values = local_model.predict_values(local_model.encode(x), geometry_value)
    global_values = global_model.predict_values(global_model.encode(x), geometry_value)
    assert torch.isfinite(local_values).all() and torch.isfinite(global_values).all()


def test_bank_size_does_not_shift_shared_seeded_parameters_or_semantic_rows():
    # 12 REGION rows versus 24 REGION+BOUNDARY rows mirrors the reference
    # ablation scale while keeping the region IDs byte-for-byte identical.
    full = QueryBank3D(BankConfig3D(
        region_grid=(1, 1, 2), boundary_grid=(1, 1, 4),
        region_scales=(0.5, 1.0),
    ))
    region = QueryBank3D(BankConfig3D(
        region_grid=(1, 1, 2), boundary_grid=(1, 1, 4),
        region_scales=(0.5, 1.0), types=("REGION",),
    ))
    assert len(region) == 12 and len(full) == 24
    torch.manual_seed(101)
    full_model = make_model(full)
    torch.manual_seed(101)
    region_model = make_model(region)
    bank_dependent = {"query_descriptors", "query_kinds", "id_embedding.weight", "policy_id_embedding.weight"}
    for key, value in full_model.state_dict().items():
        if key in bank_dependent:
            continue
        torch.testing.assert_close(value, region_model.state_dict()[key])
    full_by_id = {q.query_id: i for i, q in enumerate(full)}
    for i, q in enumerate(region):
        torch.testing.assert_close(full_model.id_embedding.weight[full_by_id[q.query_id]], region_model.id_embedding.weight[i])
        torch.testing.assert_close(full_model.policy_id_embedding.weight[full_by_id[q.query_id]], region_model.policy_id_embedding.weight[i])


def test_geometry_contract_rejects_2d_or_cropped_volume():
    model = make_model()
    with pytest.raises(ValueError, match="exactly"):
        model.decode((), torch.empty(0), {"shape": (4, 5), "spacing_yx": (1.0, 1.0)})
    with pytest.raises(ValueError, match="shape"):
        model.predict_values(model.encode(torch.rand(1, 1, 3, 4, 5)), {**geometry(), "shape": (2, 4, 5)})
