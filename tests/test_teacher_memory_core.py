import hashlib
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from self_audit.teacher_memory import (
    CLASS_PAIRS,
    MemoryBank,
    TeacherMemory,
    load_bank,
    permute_endpoints,
    random_feature_control,
    save_bank,
)


def make_bank(*, empty_group=None):
    rng = np.random.default_rng(41)
    endpoints = np.zeros((6, 2, 2, 32), dtype=np.float32)
    confidences = np.zeros((6, 2, 2), dtype=np.float32)
    valid = np.zeros((6, 2), dtype=bool)
    for group in range(6):
        if group == empty_group:
            continue
        for row in range(2):
            vectors = rng.normal(size=(2, 32)).astype(np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            endpoints[group, row] = vectors
            confidences[group, row] = (0.9 + 0.02 * row, 0.95 - 0.02 * row)
            valid[group, row] = True
    metadata = {
        "provenance": {"kind": "manufactured_test", "split": "synthetic"},
        "entry_metadata": [
            [[f"g{g}-n{n}-c", f"g{g}-n{n}-d"] for n in range(2)] for g in range(6)
        ],
    }
    return MemoryBank(endpoints, confidences, valid, metadata)


def manual_pair(bank, q, group, tau=0.2, mode="paired"):
    ep = bank.endpoints[group].to(q)
    conf = bank.confidences[group].to(q)
    valid = bank.valid[group]
    if not bool(valid.any()):
        return q.new_zeros(q.shape[0], *q.shape[-2:])
    score = torch.einsum("bdhw,nkd->bnhwk", q, ep)
    sc, sd = score[..., 0], score[..., 1]
    mask = valid[None, :, None, None]
    if mode == "paired":
        energy = (sc + sd) / (2 * tau) + (conf[:, 0] * conf[:, 1]).log()[None, :, None, None]
        attention = energy.masked_fill(~mask, -torch.inf).softmax(dim=1)
        return (attention * torch.tanh((sc - sd) / tau)).sum(dim=1)
    ac = (sc / (2 * tau) + conf[:, 0].log()[None, :, None, None]).masked_fill(~mask, -torch.inf).softmax(1)
    ad = (sd / (2 * tau) + conf[:, 1].log()[None, :, None, None]).masked_fill(~mask, -torch.inf).softmax(1)
    values = torch.tanh((sc[:, :, None] - sd[:, None, :]) / tau)
    return (ac[:, :, None] * ad[:, None, :] * values).sum((1, 2))


def test_bank_round_trip_is_hash_pinned_and_buffers_are_persistent(tmp_path):
    bank = make_bank()
    path = tmp_path / "memory.npz"
    digest = save_bank(bank, path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_bank(path, expected_sha256=digest)
    assert torch.equal(loaded.endpoints, bank.endpoints)
    assert loaded.metadata == bank.metadata
    assert {"endpoints", "confidences", "valid"} <= set(loaded.state_dict())
    with pytest.raises(ValueError, match="SHA-256"):
        load_bank(path, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("classmap", "class map"),
        ("nonfinite", "finite"),
        ("norm", "unit normalized"),
        ("confidence", "confidence >= 0.8"),
        ("shape", "shape"),
        ("provenance", "provenance"),
    ],
)
def test_bank_constructor_fails_closed(mutation, message):
    bank = make_bank()
    ep = bank.endpoints.numpy().copy()
    conf = bank.confidences.numpy().copy()
    valid = bank.valid.numpy().copy()
    metadata = dict(bank.metadata)
    if mutation == "classmap":
        metadata["class_map"] = ["BG", "LV", "MYO", "RV"]
    elif mutation == "nonfinite":
        ep[0, 0, 0, 0] = np.nan
    elif mutation == "norm":
        ep[0, 0, 0] *= 2
    elif mutation == "confidence":
        conf[0, 0, 0] = 0.5
    elif mutation == "shape":
        ep = ep[:, :-1]
    elif mutation == "provenance":
        metadata["provenance"] = {}
    with pytest.raises(ValueError, match=message):
        MemoryBank(ep, conf, valid, metadata)


def test_paired_a3_matches_exact_formula_gate_and_zero_sum():
    torch.manual_seed(17)
    bank = make_bank(empty_group=5)
    module = TeacherMemory(channels=3, bank=bank, mode="paired", tau=0.2)
    h = torch.randn(2, 3, 4, 5, requires_grad=True)
    z0 = torch.randn(2, 4, 16, 20, requires_grad=True)
    delta = module(h, z0)

    q = F.normalize(F.conv2d(h, module.projection[:, :, None, None]), dim=1)
    p0 = F.interpolate(z0.softmax(1).detach(), size=h.shape[-2:], mode="area")
    expected4 = torch.zeros_like(torch.empty(2, 4, 4, 5))
    for idx, (c, d) in enumerate(CLASS_PAIRS):
        r = manual_pair(bank, q, idx, mode="paired")
        g = 4 * p0[:, c] * p0[:, d]
        value = (0.5 * module.alpha[idx].sigmoid()) * g * r
        expected4[:, c] += value
        expected4[:, d] -= value
    expected = F.interpolate(expected4, size=z0.shape[-2:], mode="bilinear", align_corners=False)
    torch.testing.assert_close(delta, expected)
    torch.testing.assert_close(delta.sum(1), torch.zeros_like(delta[:, 0]), atol=2e-7, rtol=0)
    assert delta.abs().max() <= module.gamma_max + 1e-6
    assert delta.abs().max() <= 0.5 * torch.sigmoid(torch.tensor(-4.0)) + 1e-6
    assert torch.count_nonzero(delta[:, :, :, :]) > 0

    delta.square().mean().backward()
    assert module.projection.grad is not None and module.projection.grad.abs().sum() > 0
    assert module.alpha.grad is not None and module.alpha.grad.abs().sum() > 0
    assert h.grad is not None and h.grad.abs().sum() > 0
    assert z0.grad is None
    assert bank.endpoints.grad is None and bank.confidences.grad is None
    assert not list(bank.parameters())


def test_unpaired_a2_is_independent_marginalization_with_same_trainable_budget():
    torch.manual_seed(23)
    bank = make_bank()
    a3 = TeacherMemory(2, bank, mode="paired")
    a2 = TeacherMemory(2, bank, mode="unpaired")
    a2.projection.data.copy_(a3.projection.data)
    a2.alpha.data.copy_(a3.alpha.data)
    assert sum(p.numel() for p in a2.parameters()) == sum(p.numel() for p in a3.parameters())
    h = torch.randn(1, 2, 3, 4)
    z0 = torch.randn(1, 4, 12, 16)
    q = F.normalize(F.conv2d(h, a2.projection[:, :, None, None]), dim=1)
    actual = a2._pool_unpaired(q, 0)
    expected = manual_pair(bank, q, 0, mode="unpaired")
    torch.testing.assert_close(actual, expected)


def test_empty_groups_padding_mask_and_control_constructors():
    bank = make_bank(empty_group=2)
    paired = TeacherMemory(2, bank)
    q = F.normalize(torch.randn(1, 32, 3, 4), dim=1)
    assert torch.count_nonzero(paired._pool_paired(q, 2)) == 0
    h = torch.randn(1, 2, 3, 4)
    z0 = torch.randn(1, 4, 12, 16)
    valid = torch.ones(1, 12, 16, dtype=torch.bool)
    valid[..., 12:] = False
    delta = paired(h, z0, valid)
    assert torch.count_nonzero(delta[..., 12:]) == 0
    assert torch.count_nonzero(delta[..., :12]) > 0

    shuffled = permute_endpoints(bank, seed=7)
    randomized = random_feature_control(bank, seed=7)
    assert torch.equal(shuffled.endpoints[:, :, 0], bank.endpoints[:, :, 0])
    assert shuffled.metadata["control"]["identity_groups"] == [2]
    for group in (0, 1, 3, 4, 5):
        assert not torch.equal(shuffled.endpoints[group, :, 1], bank.endpoints[group, :, 1])
        for row in range(2):
            source = next(
                i for i in range(2)
                if torch.equal(shuffled.endpoints[group, row, 1], bank.endpoints[group, i, 1])
            )
            assert shuffled.confidences[group, row, 1] == bank.confidences[group, source, 1]
            assert (
                shuffled.metadata["entry_metadata"][group][row][1]
                == bank.metadata["entry_metadata"][group][source][1]
            )
    assert torch.equal(randomized.valid, bank.valid)
    assert torch.equal(randomized.confidences, bank.confidences)
    assert torch.equal(randomized.endpoints[~randomized.valid], torch.zeros_like(randomized.endpoints[~randomized.valid]))
    for group, row in zip(*np.where(randomized.valid.numpy())):
        np.testing.assert_allclose(randomized.endpoints[group, row].norm(dim=-1).numpy(), 1.0, atol=1e-6)


def test_nondivisible_native_size_uses_aligned_zero_padding_and_stride4_pooling():
    torch.manual_seed(103)
    bank = make_bank(empty_group=5)
    module = TeacherMemory(channels=2, bank=bank, mode="paired")
    h = torch.randn(1, 2, 3, 4)
    # Native dimensions are not multiples of four: the decoder feature covers
    # a 12x16 padded footprint while the actual image is 9x13.
    z0 = torch.randn(1, 4, 9, 13)
    actual = module(h, z0)

    probabilities = z0.softmax(1).detach()
    p0_padded = F.pad(probabilities, (0, 3, 0, 3), value=0.0)
    p0_4 = F.avg_pool2d(p0_padded, kernel_size=4, stride=4)
    native_valid = torch.ones(1, 1, 9, 13, dtype=torch.bool)
    valid_padded = F.pad(native_valid, (0, 3, 0, 3), value=False)
    valid4 = F.avg_pool2d(valid_padded.float(), 4, 4).ge(1.0 - 1e-6)

    q = F.normalize(F.conv2d(h, module.projection[:, :, None, None]), dim=1)
    expected4 = h.new_zeros((1, 4, 3, 4))
    for idx, (c, d) in enumerate(CLASS_PAIRS):
        residual = manual_pair(bank, q, idx)
        gate = 4 * p0_4[:, c] * p0_4[:, d] * valid4[:, 0]
        pair_delta = 0.5 * module.alpha[idx].sigmoid() * gate * residual
        expected4[:, c] += pair_delta
        expected4[:, d] -= pair_delta
    expected_padded = F.interpolate(expected4, size=(12, 16), mode="bilinear", align_corners=False)
    expected = expected_padded[..., :9, :13]
    torch.testing.assert_close(actual, expected)

    # The last stride-4 row and column intersect internal padding, so their
    # gates are zero. Direct resize-to-feature-grid would shift/compress them.
    assert torch.count_nonzero(expected4[..., 2, :]) == 0
    assert torch.count_nonzero(expected4[..., :, 3]) == 0
    assert torch.isfinite(actual).all()

