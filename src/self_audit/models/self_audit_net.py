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
        active = torch.ones(images.shape[0], dtype=torch.bool, device=images.device)
        if mode != "initial_only":
            for turn in range(cap):
                if not bool(active.any()):
                    break
                expert_output = self.annotation_expert(
                    shared,
                    state,
                    previous_audit_evidence=previous_audit,
                    turn_index=turn,
                    iteration_index=turn,
                    return_metadata=False,
                )
                candidate = expert_output.candidate_logits
                previous_probs = state.detach().softmax(dim=1)
                candidate_probs = candidate.detach().softmax(dim=1)
                entropy_previous = -(previous_probs.clamp_min(1e-8) * previous_probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
                entropy_candidate = -(candidate_probs.clamp_min(1e-8) * candidate_probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
                audit_output = self.auditor(
                    shared.detach(),
                    previous_probs.detach(),
                    candidate_probs.detach(),
                    (candidate_probs - previous_probs).detach(),
                    entropy_previous=entropy_previous.detach(),
                    entropy_candidate=entropy_candidate.detach(),
                )
                delta_q = audit_output.delta_q.detach().reshape(-1)
                if mode == "always_accept_refinement":
                    accepted = active.clone()
                elif mode == "oracle_accept":
                    accepted = active & self._candidate_improves(state, candidate, oracle_target)
                else:
                    accepted = active & (delta_q > float(tau_accept))
                state = torch.where(accepted[:, None, None, None], candidate, state)
                local_evidence = audit_output.local_logits.detach().softmax(dim=1)
                previous_audit = torch.where(accepted[:, None, None, None], local_evidence, torch.zeros_like(local_evidence))
                candidates.append(candidate)
                audits.append({
                    "local_logits": audit_output.local_logits,
                    "delta_q": audit_output.delta_q,
                    "accepted": accepted,
                })
                if bool(accepted.any()):
                    accepted_turns.append(turn)
                rejected = active & ~accepted
                if bool(rejected.any()):
                    halted_turns.append(turn)
                if mode == "self_audit" or mode == "oracle_accept":
                    active = active & accepted
                # always_accept intentionally runs to the hard cap; the cap is
                # a safety limit, never a requirement for the self-audit path.
        return {
            "logits": state,
            "initial_logits": initial,
            "a0_logits": initial,
            "A0_logits": initial,
            "A_t": state,
            "states": candidates,
            "candidates": candidates,
            "audits": audits,
            "accepted_turns": accepted_turns,
            "halted_turns": halted_turns,
            "shared_features": shared,
            "features": encoded["features"],
        }

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
