"""Non-differentiable threshold-controlled accept/reject gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import torch
from torch import Tensor


@dataclass
class GateDecision:
    state: Tensor
    accepted: Tensor
    halted: Tensor


def threshold_accept(delta_q: Tensor, tau_accept: float = 0.0) -> Tensor:
    """Accept strictly when predicted signed improvement exceeds the threshold."""

    return delta_q.detach().reshape(-1) > float(tau_accept)


def accept_reject(
    previous: Tensor,
    candidate: Tensor,
    delta_q: Tensor,
    *,
    tau_accept: float = 0.0,
    active: Tensor | None = None,
) -> GateDecision:
    accepted = threshold_accept(delta_q, tau_accept)
    if active is not None:
        accepted = accepted & active.to(device=accepted.device, dtype=torch.bool).reshape(-1)
    state = torch.where(accepted.view(-1, *([1] * (candidate.ndim - 1))), candidate, previous)
    halted = (~accepted) if active is None else (active.to(dtype=torch.bool).reshape(-1) & ~accepted)
    return GateDecision(state=state, accepted=accepted, halted=halted)


class ThresholdGate:
    def __init__(self, tau_accept: float = 0.0, t_max: int = 3) -> None:
        if int(t_max) < 0:
            raise ValueError("t_max must be non-negative")
        self.tau_accept = float(tau_accept)
        self.t_max = int(t_max)

    def decide(self, previous: Tensor, candidate: Tensor, delta_q: Tensor, active: Tensor | None = None) -> GateDecision:
        return accept_reject(previous, candidate, delta_q, tau_accept=self.tau_accept, active=active)

    def run(self, initial_state: Tensor, transition: Callable[[Tensor, int], tuple[Tensor, Tensor]]) -> dict[str, Any]:
        state = initial_state
        active = torch.ones(state.shape[0], device=state.device, dtype=torch.bool)
        accepted_turns: list[int] = []
        halted_turns: list[int] = []
        for turn in range(self.t_max):
            if not bool(active.any()):
                break
            candidate, delta_q = transition(state, turn)
            decision = self.decide(state, candidate, delta_q, active=active)
            state = decision.state
            if bool(decision.accepted.any()):
                accepted_turns.append(turn)
            if bool(decision.halted.any()):
                halted_turns.append(turn)
            active = active & decision.accepted
        return {"state": state, "accepted_turns": accepted_turns, "halted_turns": halted_turns, "active": active}


accept_candidate = accept_reject
