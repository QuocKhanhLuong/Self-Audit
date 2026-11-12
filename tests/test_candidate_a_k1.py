"""Focused K1 contracts; all fixtures are manufactured CPU tensors."""

import torch
import pytest
from types import MethodType

from self_audit.candidate_a.bank import BankConfig, QueryBank
from self_audit.candidate_a.codec import MeasurementCodec
from self_audit.candidate_a.k1 import (
    K1Config,
    STOP_ID,
    infer_student,
    policy_cross_entropy,
    student_rollout,
    trial_utilities,
)
from self_audit.candidate_a.model import CandidateAStudent, StudentConfig


def tiny_bank():
    return QueryBank(BankConfig(
        classes=(1, 2, 3), region_grid=(1, 1), boundary_grid=(1, 1),
        region_scales=(1.0,), radius_mm=5.0,
    ))


def tiny_student(bank, codec):
    return CandidateAStudent(bank, StudentConfig(width=8, decoder_dim=8, num_heads=2))


def geom():
    return {"shape": (6, 7), "spacing_yx": (2.0, 1.0)}


def test_stop_is_valid_at_step_zero_and_no_query_is_added():
    bank = tiny_bank(); codec = MeasurementCodec(bank, value_bits=4, base_ids=(0,))
    model = tiny_student(bank, codec)
    x = torch.rand(1, 3, 6, 7)
    base_bits = codec.cost((0,), shape=(6, 7), spacing=(2.0, 1.0))
    result = student_rollout(
        model, x, bank, codec=codec, geometry=geom(),
        config=K1Config(max_steps=1, candidate_limit=1, bit_budget=base_bits, baseline_ids=(0,)),
    )
    assert result.traces[0].candidate_ids == (STOP_ID,)
    assert result.traces[0].selected_id == STOP_ID
    assert result.code_ids == (0,)


def test_candidate_mask_happens_before_proposals_and_uses_actual_codec_cost():
    bank = tiny_bank(); codec = MeasurementCodec(bank, value_bits=8, base_ids=(0,))
    model = tiny_student(bank, codec)
    x = torch.rand(1, 3, 6, 7)
    features = model.encode(x)
    values = model.predict_values(features)
    base_bits = codec.cost((0,), shape=(6, 7), spacing=(2.0, 1.0))
    one_extra = codec.incremental_cost((0,), 1, shape=(6, 7), spacing=(2.0, 1.0))
    result = student_rollout(
        model, x, bank, codec=codec, geometry=geom(),
        config=K1Config(max_steps=4, candidate_limit=6, bit_budget=base_bits + one_extra,
                        baseline_ids=(0,), random_candidates=10),
    )
    assert len(result.code_ids) == len(set(result.code_ids))
    assert result.code_bits <= base_bits + one_extra
    for trace in result.traces:
        assert set(trace.state_ids).isdisjoint(set(trace.candidate_ids) - {STOP_ID})
        assert all(
            trace.state_bits + codec.incremental_cost(trace.state_ids, i, shape=(6, 7), spacing=(2.0, 1.0))
            <= base_bits + one_extra
            for i in trace.candidate_ids if i != STOP_ID
        )


def test_action_uses_full_feasible_bank_not_only_utility_trial_candidates():
    bank = tiny_bank(); codec = MeasurementCodec(bank, value_bits=4, base_ids=(0,))
    model = tiny_student(bank, codec)
    x = torch.rand(1, 3, 6, 7)

    def policy(self, features, code_ids, code_values, remaining_bits=None):
        row = torch.full((1, len(bank) + 1), -10.0, device=features.device)
        row[:, 2] = 10.0  # query 2 is deliberately outside the one-query trial set
        row[:, -1] = -10.0
        return row

    model.policy_logits = MethodType(policy, model)
    features = model.encode(x)
    student_values = model.predict_values(features).detach()
    teacher_values = student_values.clone()
    teacher_values[:, 0] += 1.0  # teacher proposal is query 0, not query 2
    result = student_rollout(
        model, x, bank, codec=codec, geometry=geom(), teacher_values=teacher_values,
        config=K1Config(max_steps=1, candidate_limit=2, teacher_topk=1, student_topk=0,
                        random_candidates=0, bit_budget=1024, baseline_ids=(0,)),
    )
    assert 2 not in result.traces[0].candidate_ids
    assert result.traces[0].selected_id == 2


def test_student_only_action_is_independent_of_teacher_proposal_settings():
    bank = tiny_bank(); codec = MeasurementCodec(bank, value_bits=4, base_ids=(0,))
    model = tiny_student(bank, codec).eval()
    x = torch.rand(1, 3, 6, 7)
    broad = K1Config(max_steps=2, candidate_limit=2, teacher_topk=3, student_topk=1,
                     random_candidates=3, bit_budget=1024, baseline_ids=(0,))
    narrow = K1Config(max_steps=2, candidate_limit=2, teacher_topk=0, student_topk=1,
                      random_candidates=0, bit_budget=1024, baseline_ids=(0,))
    first = infer_student(model, x, bank, codec=codec, geometry=geom(), config=broad)
    second = infer_student(model, x, bank, codec=codec, geometry=geom(), config=narrow)
    assert first.code_ids == second.code_ids


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_feasible_action_cannot_enter_rollout(bad_value):
    bank = tiny_bank(); codec = MeasurementCodec(bank, value_bits=4, base_ids=(0,))
    model = tiny_student(bank, codec)

    def policy(self, features, code_ids, code_values, remaining_bits=None):
        row = torch.zeros((1, len(bank) + 1), device=features.device)
        row[:, 1] = bad_value
        return row

    model.policy_logits = MethodType(policy, model)
    with pytest.raises(FloatingPointError):
        student_rollout(model, torch.rand(1, 3, 6, 7), bank, codec=codec, geometry=geom(),
                        config=K1Config(max_steps=1, student_topk=0, teacher_topk=0,
                                        random_candidates=0, bit_budget=1024, baseline_ids=(0,)))


class SpyDecoder:
    def __init__(self, codec):
        self.codec = codec
        self.calls = []

    def decode(self, ids, values, geometry, chunk_size=None):
        self.calls.append((tuple(ids), values.detach().clone()))
        score = values.sum(dim=1)[:, None, None, None]
        out = torch.zeros(values.shape[0], 4, *geometry["shape"])
        out[:, 1:2] = score
        return out


def test_trial_uses_quantized_student_value_and_one_decoder_state():
    bank = tiny_bank(); codec = MeasurementCodec(bank, value_bits=3, base_ids=(0,))
    spy = SpyDecoder(codec)
    student_values = torch.tensor([[0.13, 0.37, -0.41, 0.21, 0.5, -0.2]])[:, :len(bank)]
    current = codec.quantize_values(student_values[:, :1], (0,), ste=False).values
    labels = torch.ones(1, 6, 7, dtype=torch.long)
    utilities = trial_utilities(
        spy, current_ids=(0,), current_values=current, candidate_ids=(1, STOP_ID),
        student_values=student_values, labels=labels, codec=codec,
        geometry=geom(), lambda_cost=0.0,
    )
    expected = codec.quantize_values(student_values[:, 1:2], (1,), ste=False).values[:, 0]
    assert len(spy.calls) == 2  # one base call and one real trial; STOP does not decode
    assert spy.calls[0][0] == (0,)
    assert spy.calls[1][0] == (0, 1)
    torch.testing.assert_close(spy.calls[1][1][:, -1], expected)
    assert utilities[-1].item() == 0.0


def test_policy_target_is_detached_but_policy_logit_keeps_gradient():
    logits = torch.randn(1, len(tiny_bank()) + 1, requires_grad=True)
    utilities = torch.tensor([0.2, -0.4, 0.0], requires_grad=True)
    loss, target = policy_cross_entropy(logits, (0, 1, STOP_ID), utilities,
                                        bank_size=len(tiny_bank()), temperature=0.7)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert utilities.grad is None
    assert not target.requires_grad


def test_student_loss_backpropagates_through_value_and_decoder_not_teacher():
    bank = tiny_bank(); codec = MeasurementCodec(bank, value_bits=4, base_ids=(0,))
    model = tiny_student(bank, codec)
    x = torch.rand(1, 3, 6, 7)
    labels = torch.zeros(1, 6, 7, dtype=torch.long)
    teacher = torch.zeros(1, len(bank), requires_grad=True)
    targets = bank.targets(labels, (2.0, 1.0))
    result = student_rollout(
        model, x, bank, codec=codec, geometry=geom(), labels=labels,
        teacher_values=teacher, query_targets=targets,
        config=K1Config(max_steps=1, candidate_limit=2, bit_budget=1024, baseline_ids=(0,)),
    )
    result.total_loss.backward()
    assert teacher.grad is None
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.value_head.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.decoder_head.parameters())


def test_random_control_disables_policy_loss_and_student_only_inference_has_no_teacher():
    bank = tiny_bank(); codec = MeasurementCodec(bank, value_bits=4, base_ids=(0,))
    model = tiny_student(bank, codec)
    x = torch.rand(1, 3, 6, 7)
    labels = torch.zeros(1, 6, 7, dtype=torch.long)
    result = student_rollout(
        model, x, bank, codec=codec, geometry=geom(), labels=labels,
        config=K1Config(max_steps=1, candidate_limit=3, bit_budget=1024, baseline_ids=(0,),
                        policy_mode="random", policy_enabled=False),
    )
    assert result.policy_loss is not None and float(result.policy_loss.detach()) == 0.0
    assert result.traces[0].target_probs is None
    inference = infer_student(
        model, x, bank, codec=codec, geometry=geom(),
        config=K1Config(max_steps=1, candidate_limit=3, bit_budget=1024, baseline_ids=(0,)),
    )
    assert inference.final_logits.shape == (1, 4, 6, 7)
    assert inference.code_values.shape[1] == inference.query_count


def test_teacher_adapter_consumes_dual_branch_logits_without_geometry_argument():
    bank = tiny_bank(); codec = MeasurementCodec(bank, value_bits=4, base_ids=(0,))
    model = tiny_student(bank, codec).eval()
    x = torch.rand(1, 3, 6, 7)

    class Teacher:
        def __call__(self, image):
            shape = image.shape[-2:]
            return {
                "region_logits": torch.zeros(1, 4, *shape),
                "boundary_logits": torch.zeros(1, 4, *shape),
                "sdf": torch.zeros(1, 4, *shape),
            }

    result = student_rollout(model, x, bank, codec=codec, geometry=geom(),
                             config=K1Config(max_steps=1, bit_budget=1024, baseline_ids=(0,)),
                             teacher=Teacher())
    assert result.final_logits.shape == (1, 4, 6, 7)


def test_teacher_proposal_type_ablation_filters_only_teacher_candidates():
    bank = tiny_bank(); codec = MeasurementCodec(bank, value_bits=4, base_ids=(0,))
    model = tiny_student(bank, codec).eval()
    x = torch.rand(1, 3, 6, 7)
    values = model.predict_values(model.encode(x)).detach()
    teacher_values = values.clone()
    boundary_id = next(q.index for q in bank if q.kind == "BOUNDARY")
    teacher_values[:, boundary_id] += 1.0
    result = student_rollout(
        model, x, bank, codec=codec, geometry=geom(), teacher_values=teacher_values,
        config=K1Config(max_steps=1, candidate_limit=2, teacher_topk=1, student_topk=0,
                        random_candidates=0, teacher_proposal_types=("REGION",), bit_budget=1024,
                        baseline_ids=(0,)),
    )
    assert all(bank.query(i).kind == "REGION" for i in result.traces[0].candidate_ids if i != STOP_ID)


def test_candidate_streams_round_robin_each_teacher_type_student_and_random():
    bank = tiny_bank(); codec = MeasurementCodec(bank, value_bits=4, base_ids=(0,))
    model = tiny_student(bank, codec).eval()
    x = torch.rand(1, 3, 6, 7)
    features = model.encode(x)
    student_values = model.predict_values(features).detach()
    teacher_values = student_values.clone()
    region_id = next(q.index for q in bank if q.kind == "REGION" and q.index != 0)
    boundary_id = next(q.index for q in bank if q.kind == "BOUNDARY")
    region_value = student_values[:, region_id]
    teacher_values[:, region_id] = torch.where(
        region_value < 0.9, region_value + 0.1, region_value - 0.1
    )
    boundary_value = student_values[:, boundary_id]
    teacher_values[:, boundary_id] = torch.where(
        boundary_value >= 0, -torch.ones_like(boundary_value), torch.ones_like(boundary_value)
    )

    def policy(self, features, code_ids, code_values, remaining_bits=None):
        row = torch.full((1, len(bank) + 1), -100.0, device=features.device)
        row[:, 2] = 100.0  # student stream's distinct top query
        return row

    model.policy_logits = MethodType(policy, model)
    result = student_rollout(
        model, x, bank, codec=codec, geometry=geom(), teacher_values=teacher_values,
        config=K1Config(max_steps=1, candidate_limit=5, teacher_topk=1,
                        student_topk=1, random_candidates=1, seed=0,
                        bit_budget=1024, baseline_ids=(0,)),
    )
    non_stop = tuple(i for i in result.traces[0].candidate_ids if i != STOP_ID)
    assert len(non_stop) == 4
    assert set(non_stop) == {region_id, boundary_id, 2, 5}
    assert bank.query(region_id).kind == "REGION"
    assert bank.query(boundary_id).kind == "BOUNDARY"
