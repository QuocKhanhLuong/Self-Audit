"""Cheap value prediction, with an explicit batch-level audit budget."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


class AuditTrigger(nn.Module):
    def __init__(self, feature_channels=16):
        super().__init__()
        # C pooled features + 4 probabilities + entropy + xy mean/std = C+9.
        self.input_dim=feature_channels+9
        self.value=nn.Sequential(nn.Linear(self.input_dim,32),nn.GELU(),nn.Linear(32,1))

    @staticmethod
    @torch.no_grad()
    def state_features(state):
        p=state.logits.detach().softmax(1)
        entropy=-(p*p.clamp_min(1e-8).log()).sum(1).mean((1,2),keepdim=False)/math.log(4)
        xy=state.coordinates.detach().reshape(len(p),-1,2)
        return torch.cat([state.features.detach().mean((2,3)),p.mean((2,3)),entropy[:,None],
                          xy.mean(1),xy.std(1,unbiased=False)],1).detach()

    def forward(self, features):
        if features.ndim!=2 or features.shape[1]!=self.input_dim:
            raise ValueError('invalid detached trigger features')
        return self.value(features.detach()).squeeze(1)

    @staticmethod
    @torch.no_grad()
    def select(values,budget_fraction=.5):
        if values.ndim!=1 or not torch.isfinite(values).all(): raise ValueError('finite scalar values required')
        if not 0 <= budget_fraction <= 1: raise ValueError('budget fraction outside [0,1]')
        result=torch.zeros_like(values,dtype=torch.bool)
        count=math.floor(len(values)*budget_fraction)
        eligible=torch.where(values>0)[0]
        order=torch.argsort(values[eligible],descending=True,stable=True)
        result[eligible[order[:count]]]=True
        return result


def trigger_loss(trigger,features,values):
    return F.mse_loss(trigger(features.detach()),values.detach())
