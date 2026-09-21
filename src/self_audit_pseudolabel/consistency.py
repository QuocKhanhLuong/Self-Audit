"""Post-generation cross-time/cross-slice consistency for pseudo probabilities.

This is a conservative verifier, not a source of semantic truth.
"""
from __future__ import annotations
import torch

def consistency_gate(prob_tzc: torch.Tensor, valid_tz: torch.Tensor,
                     temporal_weight: float=.5, slice_weight: float=.25,
                     min_agreement: float=.60):
    if prob_tzc.ndim!=5: raise ValueError("prob must be [T,Z,C,H,W]")
    expected=(prob_tzc.shape[0],prob_tzc.shape[1],prob_tzc.shape[3],prob_tzc.shape[4])
    if valid_tz.shape!=expected: raise ValueError("valid shape mismatch")
    base=prob_tzc; support=torch.zeros_like(base); weight=torch.ones_like(base[:,:,:1])
    support += temporal_weight*(torch.roll(base,1,0)+torch.roll(base,-1,0)); weight += 2*temporal_weight
    if base.shape[1]>1:
        prev=torch.cat([base[:,:1],base[:,:-1]],1); nxt=torch.cat([base[:,1:],base[:,-1:]],1)
        support += slice_weight*(prev+nxt); weight += 2*slice_weight
    refined=(base+support)/weight
    current=base.argmax(2); neighbour=refined.argmax(2)
    confidence=refined.max(2).values
    new_valid=valid_tz & (current==neighbour) & (confidence>=float(min_agreement))
    return refined,new_valid
