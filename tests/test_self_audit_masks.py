from __future__ import annotations

import pytest
import torch
from torch import nn

from src.self_audit.models.auditor import AuditOutput
from src.self_audit.models.self_audit_net import SelfAuditNet
from src.self_audit.training import finetune_joint


class _InitialHead(nn.Module):
    def forward(self, shared: torch.Tensor, *, output_size: tuple[int, int]) -> torch.Tensor:
        return shared.new_zeros((shared.shape[0], 2, *output_size))


class _ScriptedExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(
        self,
        shared_features: torch.Tensor,
        annotation_logits: torch.Tensor,
        *args,
        turn_index: int = 0,
        **kwargs,
    ):
        self.batch_sizes.append(int(shared_features.shape[0]))
        candidate = annotation_logits + float(turn_index + 1)
        return type(
            "ExpertOutput",
            (),
            {
                "candidate_logits": candidate,
                "delta_logits": candidate - annotation_logits,
                "update_gate": annotation_logits.new_zeros((annotation_logits.shape[0], 1, *annotation_logits.shape[-2:])),
                "depth": min(int(turn_index) + 1, 3),
            },
        )()


class _ScriptedAuditor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, H: torch.Tensor, previous: torch.Tensor, candidate: torch.Tensor, *args, **kwargs) -> AuditOutput:
        self.batch_sizes.append(int(H.shape[0]))
        if len(self.batch_sizes) == 1:
            values = H.new_tensor([-1.0, 1.0])[: H.shape[0]]
        else:
            values = H.new_full((H.shape[0],), -1.0)
        delta_q = values[:, None]
        local_logits = H.new_zeros((H.shape[0], 3, *previous.shape[-2:]))
        return AuditOutput(local_logits=local_logits, delta_q=delta_q)


class _ScriptedSelfAuditNet(SelfAuditNet):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.num_classes = 2
        self.max_turns = 3
        self.initial_head = _InitialHead()
        self.annotation_expert = _ScriptedExpert()
        self.auditor = _ScriptedAuditor()

    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"features": images, "shared": images}


def test_infer_tracks_per_sample_active_masks_and_attempt_metrics() -> None:
    model = _ScriptedSelfAuditNet()
    images = torch.zeros(2, 3, 4, 4)

    output = model.infer(images, mode="self_audit", tau_accept=0.0, t_max=3)

    assert model.annotation_expert.batch_sizes == [2, 1]
    assert model.auditor.batch_sizes == [2, 1]
    assert output["accepted_count"].tolist() == [0, 1]
    assert output["halt_turn"].tolist() == [0, 1]
    assert output["num_attempted_turns"].tolist() == [1, 2]
    assert len(output["transition_active_masks"]) == 2
    assert torch.equal(output["transition_active_masks"][0], torch.tensor([True, True]))
    assert torch.equal(output["transition_active_masks"][1], torch.tensor([False, True]))
    assert torch.equal(output["transition_state_masks"][0], torch.tensor([False, True]))
    assert torch.equal(output["transition_state_masks"][1], torch.tensor([False, False]))
    assert torch.equal(
        output["transition_active_mask_tensor"],
        torch.tensor([[True, True], [False, True]]),
    )
    assert torch.equal(output["audits"][1]["active_mask"], torch.tensor([False, True]))
    assert torch.equal(output["audits"][1]["audit_mask"], torch.tensor([False, True]))
    assert torch.equal(output["audits"][1]["state_mask"], torch.tensor([False, False]))
    assert torch.allclose(output["transition_candidates"][1][0], output["transition_previous"][1][0])
    assert torch.allclose(output["logits"][0], output["initial_logits"][0])
    assert not torch.allclose(output["logits"][1], output["initial_logits"][1])


def test_inference_modes_keep_mixed_batch_bookkeeping_explicit() -> None:
    images = torch.zeros(2, 3, 4, 4)

    initial = _ScriptedSelfAuditNet().infer(images, mode="initial_only", t_max=3)
    assert initial["transition_previous"] == []
    assert initial["transition_candidates"] == []
    assert initial["accepted_count"].tolist() == [0, 0]
    assert initial["num_attempted_turns"].tolist() == [0, 0]

    always = _ScriptedSelfAuditNet().infer(images, mode="always_accept_refinement", t_max=2)
    assert len(always["transition_previous"]) == 2
    assert always["accepted_count"].tolist() == [2, 2]
    assert always["halt_turn"].tolist() == [-1, -1]
    assert always["num_attempted_turns"].tolist() == [2, 2]
    assert bool(always["final_active"].all())

    oracle_model = _ScriptedSelfAuditNet()
    with pytest.raises(ValueError, match="requires oracle_target"):
        oracle_model.infer(images, mode="oracle_accept", t_max=1)
    oracle = oracle_model.infer(
        images,
        mode="oracle_accept",
        oracle_target=torch.zeros(2, 4, 4, dtype=torch.long),
        t_max=1,
    )
    assert len(oracle["transition_previous"]) == 1
    assert oracle["accepted_count"].tolist() == [0, 0]
    assert oracle["halt_turn"].tolist() == [0, 0]
    assert oracle["num_attempted_turns"].tolist() == [1, 1]
    assert not bool(oracle["final_active"].any())


def test_select_audit_output_filters_only_active_rows() -> None:
    audit_output = {
        "local_logits": torch.randn(2, 3, 4, 4),
        "delta_q": torch.randn(2, 1),
        "active_mask": torch.tensor([False, True]),
        "state_mask": torch.tensor([False, True]),
    }

    selected = finetune_joint.select_audit_output(audit_output)

    assert selected is not None
    assert selected["local_logits"].shape == (1, 3, 4, 4)
    assert selected["delta_q"].shape == (1, 1)
    assert selected["active_mask"].tolist() == [True]
    assert selected["state_mask"].tolist() == [True]
    assert finetune_joint.select_audit_output(audit_output, torch.tensor([False, False])) is None


class _MixedBatchJointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.annotation = nn.Conv2d(3, 4, 1)
        self.auditor = nn.Linear(1, 1)

    def infer(self, images: torch.Tensor, *, mode: str, tau_accept: float, t_max: int):
        assert mode == "self_audit"
        initial = self.annotation(images)
        candidate = initial + 0.1
        first_q = self.auditor(images.new_ones((images.shape[0], 1)))
        second_q = self.auditor(images.new_ones((images.shape[0], 1)))
        first_local = first_q[:, :, None, None].expand(-1, 3, images.shape[-2], images.shape[-1])
        second_local = second_q[:, :, None, None].expand(-1, 3, images.shape[-2], images.shape[-1])
        return {
            "logits": candidate,
            "transition_previous": [initial, candidate],
            "transition_candidates": [candidate, candidate + 0.1],
            "audits": [
                {
                    "local_logits": first_local,
                    "delta_q": first_q,
                    "active_mask": torch.tensor([True, True]),
                },
                {
                    "local_logits": second_local,
                    "delta_q": second_q,
                    "active_mask": torch.tensor([False, True]),
                },
            ],
        }


def test_phase_c_filters_halted_rows_before_audit_targets(monkeypatch) -> None:
    seen: list[tuple[int, int]] = []

    def fake_audit_loss(output, targets, **kwargs):
        seen.append((int(output["delta_q"].shape[0]), int(targets.local.shape[0])))
        loss = output["delta_q"].square().mean()
        return loss, {"local": loss.detach(), "global": loss.detach(), "loss": loss.detach()}

    monkeypatch.setattr(finetune_joint, "audit_loss", fake_audit_loss)
    model = _MixedBatchJointModel()
    batch = {"image": torch.randn(2, 3, 4, 4), "mask": torch.randint(0, 4, (2, 4, 4))}

    total, details = finetune_joint.compute_joint_losses(model, batch, t_max=2)

    assert seen == [(2, 2), (1, 1)]
    assert details["transition_count"] == 2
    assert details["audit_sample_counts"] == [2, 1]
    total.backward()
    assert model.auditor.weight.grad is not None
    assert model.annotation.weight.grad is not None
