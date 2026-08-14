"""End-to-end locked Self-Audit annotation and transition-audit model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .annotation_expert import AnnotationExpert
from .annotation_head import InitialAnnotationHead
from .auditor import CounterfactualAuditor
from .encoder import ConvNeXtTinyEncoder, build_encoder
from .fpn import LightweightFPN


class SelfAuditNet(nn.Module):
    """2.5-D ConvNeXt/FPN annotation model with thresholded self-audit."""

    def __init__(
        self,
        *,
        encoder_name: str = "convnext_tiny",
        pretrained_encoder: bool = False,
        encoder_allow_fallback: bool = True,
        shared_channels: int = 96,
        num_classes: int = 4,
        window_k: int = 8,
        max_turns: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.max_turns = int(max_turns)
        self.encoder: ConvNeXtTinyEncoder = build_encoder(
            name=encoder_name,
            pretrained=pretrained_encoder,
            in_channels=3,
            allow_fallback=encoder_allow_fallback,
        )
        self.fpn = LightweightFPN(self.encoder.out_channels, out_channels=int(shared_channels))
        self.initial_head = InitialAnnotationHead(int(shared_channels), self.num_classes)
        self.annotation_expert = AnnotationExpert(
            feature_channels=int(shared_channels),
            num_classes=self.num_classes,
            window_k=int(window_k),
            max_turns=self.max_turns,
        )
        self.auditor = CounterfactualAuditor(feature_channels=int(shared_channels), num_classes=self.num_classes)

    def encode(self, images: Tensor) -> dict[str, Any]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Self-Audit input must be [B,3,H,W], got {tuple(images.shape)}")
        features = self.encoder(images)
        shared = self.fpn(features)
        return {"features": features, "shared": shared}

    def forward_annotation(self, images: Tensor, *, turns: int | None = None) -> dict[str, Any]:
        """Phase-A path: produce soft intermediate states without audit decisions."""

        encoded = self.encode(images)
        shared = encoded["shared"]
        initial = self.initial_head(shared, output_size=images.shape[-2:])
        state = initial
        audit_evidence = None
        states: list[Tensor] = []
        expert_outputs = []
        for turn in range(min(int(turns if turns is not None else self.max_turns), self.max_turns)):
            expert_output = self.annotation_expert(
                shared,
                state,
                previous_audit_evidence=audit_evidence,
                turn_index=turn,
                iteration_index=turn,
                return_metadata=False,
            )
            state = expert_output.candidate_logits
            states.append(state)
            expert_outputs.append(expert_output)
        return {
            "initial_logits": initial,
            "a0_logits": initial,
            "A0_logits": initial,
            "logits": state,
            "A_t": state,
            # ``states`` remains refinement-only for compatibility.  The
            # explicit trace is the source of truth for adjacent transitions.
            "state_trace": [initial] + states,
            "all_states": [initial] + states,
            "states": states,
            "refinement_logits": states,
            "expert_outputs": expert_outputs,
            "shared_features": shared,
            "features": encoded["features"],
        }

    def infer(
        self,
        images: Tensor,
        *,
        mode: str = "self_audit",
        tau_accept: float = 0.0,
        threshold: float | None = None,
        t_max: int | None = None,
        oracle_target: Tensor | None = None,
    ) -> dict[str, Any]:
        """Run initial-only, always-accept, or threshold-controlled inference."""

        modes = {"initial_only", "always_accept_refinement", "self_audit", "oracle_accept"}
        if mode not in modes:
            raise ValueError(f"mode must be one of {sorted(modes)}, got {mode!r}")
        if threshold is not None:
            tau_accept = float(threshold)
        cap = self.max_turns if t_max is None else int(t_max)
        if cap < 0:
            raise ValueError("t_max must be non-negative")
        if mode == "oracle_accept" and oracle_target is None:
            raise ValueError("oracle_accept is analysis-only and requires oracle_target explicitly")
        encoded = self.encode(images)
        shared = encoded["shared"]
        initial = self.initial_head(shared, output_size=images.shape[-2:])
        state = initial
        previous_audit = None
        audits: list[dict[str, Tensor]] = []
        candidates: list[Tensor] = []
        accepted_turns: list[int] = []
        halted_turns: list[int] = []
        transition_previous: list[Tensor] = []
        transition_candidates: list[Tensor] = []
        batch_size = int(images.shape[0])
        active = torch.ones(batch_size, dtype=torch.bool, device=images.device)
        accepted_count = torch.zeros(batch_size, dtype=torch.long, device=images.device)
        halt_turn = torch.full((batch_size,), -1, dtype=torch.long, device=images.device)
        num_attempted_turns = torch.zeros(batch_size, dtype=torch.long, device=images.device)
        transition_active_masks: list[Tensor] = []
        transition_state_masks: list[Tensor] = []
        if mode != "initial_only":
            for turn in range(cap):
                if not bool(active.any()):
                    break

                # Run the recurrent expert and Auditor only for rows that are
                # still active.  This keeps a halted row from receiving a new
                # candidate or audit evidence in a mixed batch.
                attempted = active.clone()
                active_indices = attempted.nonzero(as_tuple=False).flatten()
                transition_active_masks.append(attempted)
                num_attempted_turns = num_attempted_turns + attempted.to(torch.long)

                shared_active = shared.index_select(0, active_indices)
                state_active = state.index_select(0, active_indices)
                previous_audit_active = (
                    None
                    if previous_audit is None
                    else previous_audit.index_select(0, active_indices)
                )
                expert_output = self.annotation_expert(
                    shared_active,
                    state_active,
                    previous_audit_evidence=previous_audit_active,
                    turn_index=turn,
                    iteration_index=turn,
                    return_metadata=False,
                )
                candidate_active = expert_output.candidate_logits
                candidate_active_full = self._scatter_rows(state, active_indices, candidate_active)
                candidate = torch.where(
                    attempted.view(-1, 1, 1, 1),
                    candidate_active_full,
                    state,
                )
                transition_previous.append(state)
                transition_candidates.append(candidate)
                previous_probs = state_active.detach().softmax(dim=1)
                candidate_probs = candidate_active.detach().softmax(dim=1)
                entropy_previous = -(
                    previous_probs.clamp_min(1e-8)
                    * previous_probs.clamp_min(1e-8).log()
                ).sum(dim=1, keepdim=True)
                entropy_candidate = -(
                    candidate_probs.clamp_min(1e-8)
                    * candidate_probs.clamp_min(1e-8).log()
                ).sum(dim=1, keepdim=True)
                audit_output = self.auditor(
                    shared_active.detach(),
                    previous_probs.detach(),
                    candidate_probs.detach(),
                    (candidate_probs - previous_probs).detach(),
                    entropy_previous=entropy_previous.detach(),
                    entropy_candidate=entropy_candidate.detach(),
                )
                local_logits_active = self._audit_tensor(audit_output, "local_logits")
                delta_q_active = self._audit_tensor(audit_output, "delta_q")
                delta_q = delta_q_active.detach().reshape(-1)
                if mode == "always_accept_refinement":
                    accepted_active = torch.ones_like(delta_q, dtype=torch.bool)
                elif mode == "oracle_accept":
                    oracle_target_active = oracle_target.index_select(0, active_indices)
                    accepted_active = self._candidate_improves(
                        state_active,
                        candidate_active,
                        oracle_target_active,
                    )
                else:
                    accepted_active = delta_q > float(tau_accept)
                accepted = self._scatter_mask(active, active_indices, accepted_active)

                # The state update and the recurrent audit evidence are both
                # row-masked.  A rejected row keeps its previous state and no
                # local evidence is fed back into a later active row.
                state = torch.where(accepted[:, None, None, None], candidate, state)
                local_logits = self._scatter_rows_from_values(
                    batch_size,
                    active_indices,
                    local_logits_active,
                )
                local_evidence = local_logits.detach().softmax(dim=1)
                previous_audit = torch.where(
                    accepted[:, None, None, None],
                    local_evidence,
                    torch.zeros_like(local_evidence),
                )
                delta_q_full = self._scatter_rows_from_values(
                    batch_size,
                    active_indices,
                    audit_output.delta_q,
                )
                candidates.append(candidate)
                audits.append({
                    "local_logits": local_logits,
                    "delta_q": delta_q_full,
                    "accepted": accepted,
                    "active_mask": attempted,
                    "audit_mask": attempted,
                    "state_mask": accepted,
                })
                transition_state_masks.append(accepted)
                accepted_count = accepted_count + accepted.to(torch.long)
                rejected = attempted & ~accepted
                halt_now = rejected & (halt_turn < 0)
                halt_turn = torch.where(
                    halt_now,
                    torch.full_like(halt_turn, int(turn)),
                    halt_turn,
                )
                if bool(accepted.any()):
                    accepted_turns.append(turn)
                if bool(rejected.any()):
                    halted_turns.append(turn)
                if mode == "self_audit" or mode == "oracle_accept":
                    active = accepted
                # always_accept intentionally runs to the hard cap; the cap is
                # a safety limit, never a requirement for the self-audit path.
        transition_active_tensor = self._stack_masks(
            transition_active_masks,
            batch_size=batch_size,
            device=images.device,
        )
        transition_state_tensor = self._stack_masks(
            transition_state_masks,
            batch_size=batch_size,
            device=images.device,
        )
        return {
            "logits": state,
            "initial_logits": initial,
            "a0_logits": initial,
            "A0_logits": initial,
            "A_t": state,
            "states": candidates,
            "candidates": candidates,
            "transition_previous": transition_previous,
            "transition_candidates": transition_candidates,
            "audits": audits,
            "accepted_turns": accepted_turns,
            "halted_turns": halted_turns,
            "accepted_count": accepted_count,
            "halt_turn": halt_turn,
            "num_attempted_turns": num_attempted_turns,
            "active_mask": active,
            "final_active": active,
            # Per-turn lists retain the explicit [B] alignment contract.
            "active_masks": transition_active_masks,
            "transition_active_masks": transition_active_masks,
            "audit_masks": transition_active_masks,
            "transition_audit_masks": transition_active_masks,
            "state_masks": transition_state_masks,
            "transition_state_masks": transition_state_masks,
            # Stacked forms are convenience views for callers that want a
            # [T,B] tensor without changing the per-turn public contract.
            "active_mask_tensor": transition_active_tensor,
            "transition_active_mask_tensor": transition_active_tensor,
            "state_mask_tensor": transition_state_tensor,
            "transition_state_mask_tensor": transition_state_tensor,
            "shared_features": shared,
            "features": encoded["features"],
        }

    @staticmethod
    def _scatter_rows(reference: Tensor, indices: Tensor, values: Tensor) -> Tensor:
        """Scatter active-row values into a full batch while retaining autograd."""

        return SelfAuditNet._scatter_rows_from_values(reference.shape[0], indices, values)

    @staticmethod
    def _scatter_rows_from_values(batch_size: int, indices: Tensor, values: Tensor) -> Tensor:
        if values.ndim == 0 or values.shape[0] != indices.numel():
            raise ValueError(
                "Cannot scatter active rows: "
                f"values shape={tuple(values.shape)}, indices={indices.numel()}"
            )
        full = values.new_zeros((int(batch_size), *values.shape[1:]))
        return full.index_copy(0, indices, values)

    @staticmethod
    def _scatter_mask(reference: Tensor, indices: Tensor, values: Tensor) -> Tensor:
        if values.ndim != 1 or values.shape[0] != indices.numel():
            raise ValueError(
                "Cannot scatter active mask: "
                f"values shape={tuple(values.shape)}, indices={indices.numel()}"
            )
        full = torch.zeros_like(reference, dtype=torch.bool)
        return full.index_copy(0, indices, values.to(dtype=torch.bool))

    @staticmethod
    def _stack_masks(masks: list[Tensor], *, batch_size: int, device: torch.device) -> Tensor:
        if not masks:
            return torch.empty((0, int(batch_size)), dtype=torch.bool, device=device)
        return torch.stack(masks, dim=0)

    @staticmethod
    def _audit_tensor(output: Any, name: str) -> Tensor:
        if isinstance(output, dict):
            value = output.get(name)
        else:
            value = getattr(output, name, None)
        if not torch.is_tensor(value):
            raise TypeError(f"Auditor output must contain tensor {name!r}")
        return value

    def _candidate_improves(self, previous: Tensor, candidate: Tensor, target: Tensor) -> Tensor:
        previous_labels = previous.detach().argmax(dim=1)
        candidate_labels = candidate.detach().argmax(dim=1)
        target = target.detach().long()
        previous_score = self._dice(previous_labels, target)
        candidate_score = self._dice(candidate_labels, target)
        return candidate_score > previous_score

    @staticmethod
    def _dice(prediction: Tensor, target: Tensor) -> Tensor:
        values = []
        for cls in (1, 2, 3):
            pred = prediction == cls
            true = target == cls
            denominator = pred.flatten(1).sum(1) + true.flatten(1).sum(1)
            score = torch.where(denominator == 0, torch.ones_like(denominator, dtype=torch.float32), 2.0 * (pred & true).flatten(1).sum(1).float() / denominator.clamp_min(1).float())
            values.append(score)
        return torch.stack(values, dim=1).mean(dim=1)

    def forward(self, images: Tensor, **kwargs: Any) -> dict[str, Any]:
        return self.infer(images, **kwargs)


def build_self_audit_net(**kwargs: Any) -> SelfAuditNet:
    return SelfAuditNet(**kwargs)


SelfAudit = SelfAuditNet
