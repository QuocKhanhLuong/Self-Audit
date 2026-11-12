"""Synthetic contract checks for the candidate-A student only."""

import inspect

import torch

from self_audit.candidate_a.bank import BankConfig, QueryBank
from self_audit.candidate_a.codec import MeasurementCodec
from self_audit.candidate_a.model import CandidateAStudent, StudentConfig


def bank():
    return QueryBank(BankConfig(region_grid=(1, 1), boundary_grid=(1, 1), region_scales=(1.0,)))


def model():
    return CandidateAStudent(bank(), StudentConfig(width=16, decoder_dim=16, num_heads=4, chunk_size=5))


def test_values_are_bounded_and_mask_gradient_reaches_encoder_and_value_head():
    torch.manual_seed(7)
    m = model()
    x = torch.rand(1, 3, 7, 9)
    f = m.encode(x)
    values = m.predict_values(f)
    assert values.shape == (1, len(m.bank))
    kinds = m.query_kinds
    assert bool(((values[:, kinds == 0] >= 0) & (values[:, kinds == 0] <= 1)).all())
    assert bool(((values[:, kinds == 1] >= -1) & (values[:, kinds == 1] <= 1)).all())
    ids = (m.query_ids[0], m.query_ids[1])
    logits = m.decode(ids, values[:, :2], {"shape": (7, 9), "spacing_yx": (2.0, 0.7)}, chunk_size=4)
    assert logits.shape == (1, 4, 7, 9)
    logits.mean().backward()
    assert m.encoder[0].weight.grad is not None and m.encoder[0].weight.grad.abs().sum() > 0
    assert m.value_head[0].weight.grad is not None and m.value_head[0].weight.grad.abs().sum() > 0


def test_policy_masks_used_ids_stop_is_valid_from_step_zero():
    m = model()
    f = m.encode(torch.rand(1, 3, 5, 6))
    empty = m.policy_logits(f, (), torch.empty(1, 0), remaining_bits=0)
    assert torch.isfinite(empty[0, m.stop_index])
    assert torch.isneginf(empty[0, : len(m.bank)]).all()
    ids = (m.query_ids[0],)
    logits = m.policy_logits(f, ids, torch.zeros(1, 1), remaining_bits=10)
    assert torch.isneginf(logits[0, 0])
    assert torch.isfinite(logits[0, 1:]).any()


def test_decoder_has_no_feature_argument_and_empty_code_uses_null_token():
    m = model().eval()
    assert "features" not in inspect.signature(m.decode).parameters
    output = m.decode((), torch.empty(0), {"shape": (6, 8), "spacing_yx": (1.5, 2.0)})
    assert output.shape == (1, 4, 6, 8)
    assert torch.isfinite(output).all()


def test_teacher_free_state_load_matches_decoder():
    torch.manual_seed(8)
    m = model().eval()
    ids = (m.query_ids[0], m.query_ids[3])
    values = torch.tensor([[0.25, -0.5]])
    geometry = {"shape": (5, 7), "spacing_yx": (1.0, 1.0)}
    expected = m.decode(ids, values, geometry)
    restored = model().eval()
    restored.load_state_dict(m.state_dict(), strict=True)
    actual = restored.decode(ids, values, geometry)
    torch.testing.assert_close(expected, actual)
    assert all("teacher" not in key.lower() for key in restored.state_dict())


def test_decoder_is_code_only_and_serialized_quantized_code_is_geometry_stable():
    """Packet dequantization must reproduce the exact code-only decoder call."""
    torch.manual_seed(13)
    m = model().eval()
    codec = MeasurementCodec(m.bank, value_bits=5, base_ids=(0,))
    geometry = {"shape": (7, 9), "spacing_yx": (0.123456789, 2.345678901)}
    raw = m.predict_values(m.encode(torch.rand(1, 3, 7, 9)))

    # Exercise both a dense code and a sparse/non-contiguous code.  Values
    # entering decode are exactly the codec's finite-bit dequantized values.
    for ids in (tuple(range(len(m.bank))), (0, 2, 5)):
        quantized = codec.quantize_values(raw[:, list(ids)], ids, ste=False)
        before = m.decode(ids, quantized.values, geometry)
        packet = codec.deserialize(codec.serialize(ids, quantized.levels[0], geometry["shape"], geometry["spacing_yx"]))
        restored_values = codec.dequantize(packet, dtype=before.dtype).unsqueeze(0)
        after = m.decode(packet.ids, restored_values,
                         {"shape": packet.shape, "spacing_yx": packet.spacing})
        assert packet.ids == ids
        torch.testing.assert_close(before, after, atol=0, rtol=0)

        # Calling the encoder with another image cannot affect a code-only
        # decoder (there is no feature/skip path and no stateful normalization).
        _ = m.encode(torch.rand(1, 3, 7, 9))
        again = m.decode(packet.ids, restored_values,
                          {"shape": packet.shape, "spacing_yx": packet.spacing})
        torch.testing.assert_close(after, again, atol=0, rtol=0)


def test_bank_ablation_preserves_common_seeded_student_parameters():
    """Changing bank size must not silently change shared initialization."""
    full = bank()
    region_only = QueryBank(BankConfig(
        region_grid=(1, 1), boundary_grid=(1, 1), region_scales=(1.0,),
        types=("REGION",),
    ))
    boundary_only = QueryBank(BankConfig(
        region_grid=(1, 1), boundary_grid=(1, 1), region_scales=(1.0,),
        types=("BOUNDARY",),
    ))
    assert len(full) == 6 and len(region_only) == 3 and len(boundary_only) == 3
    torch.manual_seed(101)
    full_model = CandidateAStudent(full, StudentConfig(width=8, decoder_dim=8, num_heads=2))
    torch.manual_seed(101)
    region_model = CandidateAStudent(region_only, StudentConfig(width=8, decoder_dim=8, num_heads=2))
    torch.manual_seed(101)
    boundary_model = CandidateAStudent(boundary_only, StudentConfig(width=8, decoder_dim=8, num_heads=2))
    bank_dependent = {"query_descriptors", "query_kinds", "id_embedding.weight", "policy_id_embedding.weight"}
    full_state = full_model.state_dict()
    shared = [key for key in full_state if key not in bank_dependent]
    assert shared
    for subset_model in (region_model, boundary_model):
        subset_state = subset_model.state_dict()
        assert all(key in subset_state for key in shared)
        for key in shared:
            torch.testing.assert_close(full_state[key], subset_state[key], msg=f"seed drift in {key}")

    full_by_id = {query.query_id: index for index, query in enumerate(full)}
    for subset, subset_model in ((region_only, region_model), (boundary_only, boundary_model)):
        subset_by_id = {query.query_id: index for index, query in enumerate(subset)}
        for query_id, subset_index in subset_by_id.items():
            full_index = full_by_id[query_id]
            torch.testing.assert_close(full_model.id_embedding.weight[full_index],
                                       subset_model.id_embedding.weight[subset_index])
            torch.testing.assert_close(full_model.policy_id_embedding.weight[full_index],
                                       subset_model.policy_id_embedding.weight[subset_index])
