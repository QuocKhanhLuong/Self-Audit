"""Synthetic contract tests for the candidate-A bank and finite-bit codec.

These tests exercise geometry and serialization only.  They are not a patient
segmentation result, a SOTA claim, or evidence for clinical performance.
"""

import math

import pytest
import torch

from self_audit.candidate_a.bank import BankConfig, QueryBank
from self_audit.candidate_a.codec import MeasurementCodec


def tiny_bank(*, types=("REGION", "BOUNDARY")):
    return QueryBank(
        BankConfig(
            classes=(1, 2, 3),
            region_grid=(1, 1),
            boundary_grid=(2, 2),
            region_scales=(1.0,),
            radius_mm=5.0,
            types=types,
        )
    )


def test_bank_is_finite_stable_and_encoder_independent():
    first = tiny_bank()
    second = tiny_bank()
    assert len(first) == 15
    assert [q.query_id for q in first] == [q.query_id for q in second]
    assert first.fingerprint == second.fingerprint
    d = first.descriptors(dtype=torch.float64)
    assert d.shape == (len(first), 8)
    assert torch.isfinite(d).all()
    assert torch.equal(d[:, :2].sum(dim=1), torch.ones(len(first), dtype=d.dtype))


def test_region_and_boundary_targets_use_native_geometry_and_absence_convention():
    bank = tiny_bank()
    # A full-image REGION query makes its proportion exact and easy to audit.
    labels = torch.tensor(
        [[[1, 1, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0]]], dtype=torch.long
    )
    spacing = torch.tensor([[2.0, 1.0]])  # anisotropic dy/dx
    targets = bank.targets(labels, spacing)
    for q in bank:
        if q.kind == "REGION":
            expected = (labels == q.class_id).float().mean()
            torch.testing.assert_close(targets[:, q.index], expected[None])
        elif q.class_id == 3:
            # Class 3 is absent: a boundary query is explicitly outside +1.
            assert torch.all(targets[:, q.index] == 1)
    sdf = bank.sdf_targets(labels, spacing)
    assert sdf.shape == (1, 4, 3, 4)
    assert torch.isfinite(sdf).all()
    # Class 1 at this pixel is inside and has a negative SDF.  Its nearest
    # vertical interface is half a one-mm x-cell away (0.5 / 5 = 0.1).
    assert sdf[0, 1, 0, 1] < 0
    assert math.isclose(float(sdf[0, 1, 0, 1]), -0.1, abs_tol=1e-6)


def test_full_class_and_absent_class_have_finite_signed_distance():
    bank = QueryBank(BankConfig(classes=(1,), region_grid=(1, 1), boundary_grid=(1, 1), region_scales=(1.0,)))
    full = torch.ones((1, 2, 3), dtype=torch.long)
    absent = torch.zeros_like(full)
    s_full = bank.sdf_targets(full, (2.0, 3.0))
    s_absent = bank.sdf_targets(absent, (2.0, 3.0))
    assert torch.isfinite(s_full).all() and torch.isfinite(s_absent).all()
    assert (s_full[:, 1] < 0).all()
    assert (s_absent[:, 1] == 1).all()


def test_invalid_labels_spacing_and_ambiguous_dimensions_fail_closed():
    bank = tiny_bank()
    labels = torch.zeros((1, 2, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="invalid/ignore"):
        bank.targets(labels + 255, (1.0, 1.0))
    with pytest.raises(ValueError, match="spacing"):
        bank.targets(labels, (1.0, -1.0))
    with pytest.raises(ValueError, match="shape"):
        bank.targets(labels[0], (1.0, 1.0))


def test_teacher_values_match_region_and_boundary_protocol():
    bank = tiny_bank()
    labels = torch.zeros((1, 3, 4), dtype=torch.long)
    region_probs = torch.zeros((1, 4, 3, 4), dtype=torch.float32)
    region_probs[:, 1] = 0.25
    region_probs[:, 2] = 0.50
    region_probs[:, 3] = 0.75
    sdf = torch.zeros_like(region_probs)
    sdf[:, 1] = -0.1
    sdf[:, 2] = -0.2
    sdf[:, 3] = 0.3
    values = bank.teacher_values(region_probs, sdf)
    for q in bank:
        expected = region_probs[:, q.class_id].mean() if q.kind == "REGION" else sdf[:, q.class_id, 0, 0]
        expected = expected[None] if expected.ndim == 0 else expected
        torch.testing.assert_close(values[:, q.index], expected)


def test_quantization_levels_roundtrip_and_ste_gradient():
    bank = tiny_bank()
    codec = MeasurementCodec(bank, value_bits=3)
    ids = (0, 1, len(bank) - 1)
    with pytest.raises(ValueError, match="non-finite"):
        codec.quantize_values(torch.tensor([[float("nan"), 0.0, 0.0]]), ids)
    raw = torch.tensor([[0.13, 0.25, 0.91]], requires_grad=True)
    q = codec.quantize_values(raw, ids, ste=True)
    assert q.levels.dtype == torch.int64
    assert torch.allclose(q.values.detach(), codec.dequantize_levels(ids, q.levels[0]))
    q.values.sum().backward()
    assert raw.grad is not None and torch.all(raw.grad == 1)
    assert torch.all((q.levels >= 0) & (q.levels <= 7))


def test_packet_roundtrip_fingerprint_geometry_bits_and_tamper_detection():
    bank = tiny_bank()
    codec = MeasurementCodec(bank, value_bits=5, base_ids=(0,))
    ids = (0, 1, 4)  # R0 ID 0 carries its actual student level too.
    levels = (17, 3, 29)
    payload = codec.serialize(ids, levels, (11, 13), (2.0, 1.25))
    packet = codec.deserialize(payload)
    assert packet.ids == ids
    assert packet.shape == (11, 13)
    assert packet.spacing == (2.0, 1.25)
    assert codec.packet_bits(payload) == len(payload) * 8
    decoded = codec.dequantize(packet)
    assert decoded.shape == (3,)
    assert codec.serialize(packet.ids, packet.levels, packet.shape, packet.spacing) == payload

    tampered = bytearray(payload)
    tampered[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        codec.deserialize(tampered)
    with pytest.raises(ValueError, match="fingerprint"):
        codec.deserialize(payload[:10] + bytes([0]) * (len(payload) - 10))

    # Duplicate IDs are rejected before serialization and on the wire.
    with pytest.raises(ValueError, match="duplicate"):
        codec.serialize((1, 1), (1, 2), (11, 13), (2.0, 1.25))
    with pytest.raises(ValueError, match="R0"):
        codec.serialize((1,), (1,), (11, 13), (2.0, 1.25))


def test_actual_budget_cost_includes_r0_and_geometry_and_incremental_cost():
    bank = tiny_bank()
    codec = MeasurementCodec(bank, value_bits=8, base_ids=(0,))
    base = codec.cost(0, shape=(32, 32), spacing=(2.0, 1.0))
    one = codec.cost((1,), shape=(32, 32), spacing=(2.0, 1.0))
    assert base > 0 and one > base
    assert codec.incremental_cost((), 1, shape=(32, 32), spacing=(2.0, 1.0)) == one - base
    assert codec.incremental_cost((1,), 1) == 0
    assert codec.fits((1,), one, shape=(32, 32), spacing=(2.0, 1.0))
    with pytest.raises(ValueError, match="budget"):
        codec.serialize((0, 1), (128, 2), (32, 32), (2.0, 1.0), budget_bits=base)
