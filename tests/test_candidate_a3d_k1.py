"""Shared K1 on a true [B,C,D,H,W] synthetic rollout."""

import math
import torch
import pytest

from self_audit.candidate_a3d.bank import BankConfig3D, QueryBank3D
from self_audit.candidate_a3d.codec import MeasurementCodec3D
from self_audit.candidate_a3d.k1 import K1Config, STOP_ID, student_rollout
from self_audit.candidate_a3d.model import CandidateA3DStudent, StudentConfig3D
from self_audit.candidate_a.k1 import trial_utilities, policy_cross_entropy


def test_3d_k1_quantized_rollout_loss_and_gradient():
    bank = QueryBank3D(BankConfig3D(
        region_grid=(1, 1, 1), boundary_grid=(1, 1, 1), region_scales=(1.0,),
    ))
    codec = MeasurementCodec3D(bank, value_bits=4, base_ids=(0,))
    model = CandidateA3DStudent(bank, StudentConfig3D(width=8, decoder_dim=16, num_heads=4, chunk_size=9))
    geometry = {
        "shape": (3, 4, 5), "spacing_zyx": (2.0, 1.0, 0.5),
        "affine_zyx": ((2.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                       (0.0, 0.0, 0.5, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "context": "full_volume",
    }
    image = torch.rand(1, 1, 3, 4, 5)
    labels = torch.zeros(1, 3, 4, 5, dtype=torch.long)
    labels[:, 1:, 1:3, 1:4] = 1
    base_bits = codec.cost((0,), shape=geometry["shape"], spacing=geometry["spacing_zyx"], affine_zyx=geometry["affine_zyx"])
    result = student_rollout(
        model, image, bank, codec=codec, geometry=geometry, labels=labels,
        config=K1Config(max_steps=1, candidate_limit=3, teacher_topk=0,
                        random_candidates=1, bit_budget=base_bits + 256,
                        baseline_ids=(0,)),
    )
    assert result.code_bits <= base_bits + 256
    assert len(result.code_ids) == len(set(result.code_ids))
    assert result.final_logits.shape == (1, 4, 3, 4, 5)
    assert result.total_loss is not None and torch.isfinite(result.total_loss)
    result.total_loss.backward()
    assert model.encoder[0].weight.grad is not None
    assert torch.isfinite(model.encoder[0].weight.grad).all()


def test_3d_stop_is_valid_at_step_zero():
    bank = QueryBank3D(BankConfig3D(
        region_grid=(1, 1, 1), boundary_grid=(1, 1, 1), region_scales=(1.0,),
    ))
    codec = MeasurementCodec3D(bank, value_bits=4, base_ids=(0,))
    model = CandidateA3DStudent(bank, StudentConfig3D(width=8, decoder_dim=16, num_heads=4))
    geometry = {
        "shape": (2, 3, 4), "spacing_zyx": (1.0, 1.0, 1.0),
        "affine_zyx": ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                       (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "context": "full_volume",
    }
    bits = codec.cost((0,), shape=geometry["shape"], spacing=geometry["spacing_zyx"], affine_zyx=geometry["affine_zyx"])
    result = student_rollout(
        model, torch.rand(1, 1, 2, 3, 4), bank, codec=codec, geometry=geometry,
        config=K1Config(max_steps=1, candidate_limit=1, bit_budget=bits, baseline_ids=(0,)),
    )
    assert result.traces[0].candidate_ids == (STOP_ID,)
    assert result.code_ids == (0,)


def test_3d_teacher_adapter_receives_public_geometry():
    bank = QueryBank3D(BankConfig3D(
        region_grid=(1, 1, 1), boundary_grid=(1, 1, 1), region_scales=(1.0,),
    ))
    codec = MeasurementCodec3D(bank, value_bits=4, base_ids=(0,))
    model = CandidateA3DStudent(bank, StudentConfig3D(width=8, decoder_dim=16, num_heads=4))
    geometry = {
        "shape": (2, 3, 4), "spacing_zyx": (1.5, 0.8, 0.6),
        "affine_zyx": ((1.5, 0.0, 0.0, 0.0), (0.0, 0.8, 0.0, 0.0),
                       (0.0, 0.0, 0.6, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "context": "full_volume",
    }
    seen = {}

    class Teacher:
        def __call__(self, image, *, geometry):
            seen["geometry"] = geometry
            shape = image.shape[-3:]
            zeros = torch.zeros(1, 4, *shape)
            return {"region_logits": zeros, "boundary_logits": zeros, "sdf": zeros}

    bits = codec.cost((0,), shape=geometry["shape"], spacing=geometry["spacing_zyx"], affine_zyx=geometry["affine_zyx"])
    student_rollout(
        model, torch.rand(1, 1, 2, 3, 4), bank, codec=codec, geometry=geometry,
        teacher=Teacher(), config=K1Config(max_steps=1, candidate_limit=1,
                                           bit_budget=bits, baseline_ids=(0,)),
    )
    assert seen["geometry"] == geometry


def test_trial_sign_cost_and_restricted_policy_match_closed_form_volume_loss():
    bank = QueryBank3D(BankConfig3D(types=("BOUNDARY",), boundary_grid=(1, 1, 1)))
    codec = MeasurementCodec3D(bank, value_bits=8)
    geometry = {"shape": (2, 3, 4), "spacing_zyx": (1.0, 1.0, 1.0),
                "affine_zyx": torch.eye(4).tolist(), "context": "full_volume"}

    class ConstantLogitDecoder:
        def decode(self, ids, values, geometry, chunk_size=None):
            output = torch.zeros(1, 4, *geometry["shape"])
            output[:, 1] = values.sum()
            return output

    values = torch.tensor([[1.0, -1.0, 0.0]], requires_grad=True)
    utility = trial_utilities(ConstantLogitDecoder(), current_ids=(),
        current_values=values[:, :0], candidate_ids=(0, 1, STOP_ID),
        student_values=values, labels=torch.ones(1, 2, 3, 4, dtype=torch.long),
        codec=codec, geometry=geometry, lambda_cost=0.001)

    def analytical_loss(logit):
        probability = math.exp(logit) / (math.exp(logit) + 3)
        dice = (48 * probability + 1e-5) / (24 * probability + 24 + 1e-5)
        return -math.log(probability) + 1 - dice

    # One uint32 ID + one uint8 value adds exactly 40 bits to the packet.
    expected = [analytical_loss(0) - analytical_loss(v) - 0.04 for v in (1, -1)] + [0]
    assert utility.tolist() == pytest.approx(expected, abs=1e-6)
    assert utility[0] > 0 > utility[1] and utility[2] == 0
    assert not utility.requires_grad and values.grad is None
    logits = torch.tensor([[0.1, 0.2, 20.0, 0.3]], requires_grad=True)
    target_input = utility.clone().requires_grad_(True)
    loss, target = policy_cross_entropy(logits, (0, 1, STOP_ID), target_input,
                                        bank_size=3, temperature=0.5)
    restricted = logits[:, [0, 1, 3]].log_softmax(-1)[0]
    assert loss.item() == pytest.approx(-(utility.div(0.5).softmax(0) * restricted).sum().item())
    loss.backward()
    assert not target.requires_grad and target_input.grad is None
    assert logits.grad[0, 2] == 0  # the untried, high-logit query cannot enter this CE
