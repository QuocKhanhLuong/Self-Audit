"""Checkpoint-capped, per-example uncertainty routing (not an accuracy estimate)."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F

PROFILE_TURNS = {"compact": 0, "balanced": 1, "accurate": 2}

@dataclass(frozen=True)
class RuntimeBudget:
    max_profile: str = "accurate"
    entropy_compact: float = .25
    entropy_accurate: float = .55

    def __post_init__(self):
        if self.max_profile not in PROFILE_TURNS:
            raise ValueError(f"Unknown max_profile {self.max_profile}")
        if not 0 <= self.entropy_compact <= self.entropy_accurate <= 1:
            raise ValueError("bad entropy thresholds")

    @classmethod
    def from_config(cls, config, *, max_profile):
        return cls(max_profile=max_profile, entropy_compact=config["entropy_compact"],
                   entropy_accurate=config["entropy_accurate"])


def sample_entropies(logits):
    if logits.ndim != 4 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
        raise ValueError("expected finite [B,C,H,W] logits")
    logp = F.log_softmax(logits.float(), 1)
    entropy = -(logp.exp() * logp).sum(1) / math.log(logits.shape[1])
    # Maximum pixel entropy is conservative: small uncertain structures are not
    # diluted by confident background. This can spend more compute, not certify anatomy.
    return entropy.flatten(1).amax(1).clamp(0, 1)


def normalized_entropy(logits):
    return float(sample_entropies(logits).max().item())


def choose_profile(a0_logits, budget):
    h = normalized_entropy(a0_logits)
    desired = 0 if h < budget.entropy_compact else (1 if h < budget.entropy_accurate else 2)
    return tuple(PROFILE_TURNS)[min(desired, PROFILE_TURNS[budget.max_profile])]


class AdaptiveRuntime:
    def __init__(self, model, budget=None):
        cap = int(model.trained_profile_cap)
        if cap not in PROFILE_TURNS.values():
            raise ValueError("missing trained profile coverage; load a verified student checkpoint")
        if budget is None:
            low, high = model.runtime_thresholds.tolist()
            budget = RuntimeBudget(tuple(PROFILE_TURNS)[cap], low, high)
        if PROFILE_TURNS[budget.max_profile] > cap:
            raise ValueError("runtime budget exceeds trained checkpoint profile")
        self.model = model
        self.budget = budget
        self.model.eval()

    @torch.no_grad()
    def __call__(self, x, *, return_metadata=False):
        feat = self.model.encode(x)
        a0 = self.model.initial_logits(feat, x.shape[-2:])
        profiles = [choose_profile(a0[i:i+1], self.budget) for i in range(len(x))]
        outputs = [self.model.refine_from_features(feat[i:i+1], a0[i:i+1], profile=p,
                    return_metadata=return_metadata) for i, p in enumerate(profiles)]
        result = dict(outputs[0])
        result.update(a0_logits=a0, final_logits=torch.cat([o["final_logits"] for o in outputs]),
                      profile=profiles[0] if len(set(profiles)) == 1 else "mixed",
                      profiles=tuple(profiles), selection_entropy=normalized_entropy(a0),
                      sample_entropies=sample_entropies(a0).tolist())
        if len(outputs) > 1:
            result["stages"] = tuple(o["stages"] for o in outputs)
            result["window_metadata"] = tuple(o["window_metadata"] for o in outputs)
        return result
