"""Reliability-first semantic prototype evolution."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.nn import functional as F

@dataclass
class PrototypeBank:
    num_classes:int
    dim:int
    momentum:float=0.9
    def __post_init__(self):
        self.prototypes=torch.zeros(self.num_classes,self.dim); self.counts=torch.zeros(self.num_classes)
    def to(self,device):
        self.prototypes=self.prototypes.to(device); self.counts=self.counts.to(device); return self
    @torch.no_grad()
    def update(self,features,class_prob,valid,min_confidence:float=0.8):
        if not bool(valid.any()): return
        conf,cls=class_prob.max(-1); use=valid&(conf>=min_confidence)
        for c in range(self.num_classes):
            mask=use&(cls==c)
            if not bool(mask.any()): continue
            vec=F.normalize(features[mask].mean(0),dim=0)
            if self.counts[c]==0: self.prototypes[c]=vec
            else: self.prototypes[c]=F.normalize(self.momentum*self.prototypes[c]+(1-self.momentum)*vec,dim=0)
            self.counts[c]+=float(mask.sum())
    def logits(self,features,scale:float=2.0):
        f=F.normalize(features,dim=-1); p=F.normalize(self.prototypes.to(features.device),dim=-1)
        active=(self.counts.to(features.device)>0).to(features.dtype)
        return torch.einsum("bkd,cd->bkc",f,p)*float(scale)*active[None,None,:]

def accepted_region_mask(class_prob,*,min_prob=.70,min_margin=.20):
    top2=class_prob.topk(2,-1).values
    return (top2[...,0]>=min_prob)&((top2[...,0]-top2[...,1])>=min_margin)
