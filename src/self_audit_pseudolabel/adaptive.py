"""Resource + uncertainty controller for the deployment annotator."""
from __future__ import annotations
from dataclasses import dataclass
import math
from torch.nn import functional as F

PROFILE_TURNS={"compact":0,"balanced":1,"accurate":2}

@dataclass(frozen=True)
class RuntimeBudget:
    max_profile:str="accurate"
    entropy_compact:float=.25
    entropy_accurate:float=.55
    def __post_init__(self):
        if self.max_profile not in PROFILE_TURNS: raise ValueError(f"Unknown max_profile {self.max_profile}")
        if not 0<=self.entropy_compact<=self.entropy_accurate<=1: raise ValueError("bad entropy thresholds")

def normalized_entropy(logits):
    p=F.softmax(logits.float(),1); h=-(p*p.clamp_min(1e-8).log()).sum(1)/math.log(logits.shape[1])
    return float(h.mean().item())

def choose_profile(a0_logits,budget):
    h=normalized_entropy(a0_logits); cap=PROFILE_TURNS[budget.max_profile]
    desired=0 if h<budget.entropy_compact else (1 if h<budget.entropy_accurate else 2)
    turns=min(desired,cap); return {0:"compact",1:"balanced",2:"accurate"}[turns]

class AdaptiveRuntime:
    def __init__(self,model,budget=None): self.model=model; self.budget=budget or RuntimeBudget()
    def __call__(self,x,*,return_metadata=False):
        feat=self.model.encode(x); a0=self.model.initial_logits(feat,x.shape[-2:])
        profile=choose_profile(a0,self.budget)
        out=self.model.refine_from_features(feat,a0,profile=profile,return_metadata=return_metadata)
        out["selection_entropy"]=normalized_entropy(a0); return out
