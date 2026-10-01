"""Rejection-only consistency. No periodic wrap or propagation from invalid neighbours."""
from __future__ import annotations
import torch
from torch.nn import functional as F

def consistency_gate(prob_tzc,valid_tz,temporal_weight=.5,slice_weight=0.,min_agreement=.60,
                     flow_prev=None,flow_next=None):
    if prob_tzc.ndim!=5 or valid_tz.shape!=(*prob_tzc.shape[:2],*prob_tzc.shape[-2:]):
        raise ValueError("expected probabilities [T,Z,C,H,W], validity [T,Z,H,W]")
    if valid_tz.dtype!=torch.bool or temporal_weight<0 or slice_weight<0 or not 0<=min_agreement<=1:
        raise ValueError("invalid consistency parameters")
    if slice_weight:
        raise ValueError("cross-slice correspondence is not implemented; use slice_weight=0")
    if (flow_prev is None)!=(flow_next is None): raise ValueError("supply both temporal flow fields")
    tmax,zmax,_,h,w=prob_tzc.shape
    label=prob_tzc.argmax(2); agree=valid_tz.float().clone(); total=valid_tz.float().clone()
    yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=prob_tzc.device),torch.linspace(-1,1,w,device=prob_tzc.device),indexing='ij')
    identity=torch.stack([xx,yy],-1)[None]
    for t in range(tmax):
        for dt,flow in ((-1,flow_prev),(1,flow_next)):
            nt=t+dt
            if not 0<=nt<tmax: continue  # no invented last-to-first acquisition link
            other=prob_tzc[nt]; supported=valid_tz[nt]
            if flow is not None:
                f=F.interpolate(flow[t],(h,w),mode='bilinear',align_corners=False)
                grid=identity+f.permute(0,2,3,1)
                other=F.grid_sample(other,grid,align_corners=True,padding_mode='zeros')
                supported=F.grid_sample(supported[:,None].float(),grid,mode='nearest',align_corners=True)[:,0]>.5
                supported &= (grid.abs()<=1).all(-1)
            weight=supported.float()*temporal_weight
            total[t]+=weight
            agree[t]+=weight*(label[t]==other.argmax(1)).float()
    new_valid=valid_tz & (agree/total.clamp_min(1e-8)>=min_agreement)
    # Do not replace probabilities with neighbour averages that move boundaries.
    return prob_tzc,new_valid
