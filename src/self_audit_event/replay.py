"""Uniform seeded RF-only replay with explicit actor-step expiry."""
from __future__ import annotations
import random
import torch
from .reference_free_auditor import AuditObservation


class AuditReplay:
    def __init__(self,capacity=256,seed=0):
        if capacity<1: raise ValueError('positive capacity required')
        self.capacity=capacity;self.rows=[];self.rng=random.Random(seed)

    def add(self,observation,trigger_features,values,step):
        if not isinstance(observation,AuditObservation): raise TypeError('AuditObservation required')
        if len(observation.image)!=len(trigger_features) or values.shape!=(len(trigger_features),):
            raise ValueError('replay batch mismatch')
        for i in range(len(values)):
            self.rows.append((observation.image[i:i+1].detach().clone(),observation.logits[i:i+1].detach().clone(),
                              trigger_features[i:i+1].detach().clone(),values[i:i+1].detach().clone(),int(step)))
        self.rows=self.rows[-self.capacity:]

    def sample(self,n,current_step,max_age=8):
        if n<1 or max_age<0: raise ValueError('invalid sample size/age')
        fresh=[r for r in self.rows if 0<=current_step-r[-1]<=max_age]
        if not fresh: return None
        chosen=self.rng.sample(fresh,min(n,len(fresh)))
        return {'observation':AuditObservation(torch.cat([r[0] for r in chosen]),torch.cat([r[1] for r in chosen])),
                'features':torch.cat([r[2] for r in chosen]),'values':torch.cat([r[3] for r in chosen]),
                'steps':torch.tensor([r[4] for r in chosen])}
