"""Image-only region evidence for named cardiac pseudo-label seeds."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
from torch.nn import functional as F
BG,RV,MYO,LV=0,1,2,3

@dataclass(frozen=True)
class EvidenceConfig:
    border_weight: float=3.0
    low_motion_bg_weight: float=1.0
    enclosure_weight: float=4.0
    adjacency_weight: float=1.5
    orientation_weight: float=2.0
    boundary_weight: float=0.75
    min_region_pixels: int=4

def _boundary_gradient(image):
    image=image.astype(np.float32,copy=False)
    gy=np.gradient(image,axis=0) if image.shape[0]>1 else np.zeros_like(image)
    gx=np.gradient(image,axis=1) if image.shape[1]>1 else np.zeros_like(image)
    return np.sqrt(gx*gx+gy*gy)

def _shift(mask,dy,dx):
    out=np.zeros_like(mask)
    ys=slice(max(0,dy),mask.shape[0]+min(0,dy)); xs=slice(max(0,dx),mask.shape[1]+min(0,dx))
    sy=slice(max(0,-dy),mask.shape[0]-max(0,dy)); sx=slice(max(0,-dx),mask.shape[1]-max(0,dx))
    out[ys,xs]=mask[sy,sx]; return out

def _adjacent(a,b):
    return any(np.any(_shift(a,dy,dx)&b) for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)))

def _holes(mask):
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(mask)&(~mask)

def build_region_evidence(region_prob,image,motion_features,*,patient_left_axis:Sequence[str|None]|None=None,config=None):
    cfg=config or EvidenceConfig(); b,k,h,w=region_prob.shape
    hard=region_prob.detach().argmax(1).cpu().numpy()
    img=F.interpolate(image.detach(),(h,w),mode="bilinear",align_corners=False)[:,0].cpu().numpy()
    mot=F.interpolate(motion_features.detach().pow(2).mean(1,keepdim=True).sqrt(),(h,w),mode="bilinear",align_corners=False)[:,0].cpu().numpy()
    logits=np.zeros((b,k,4),dtype=np.float32); diags=[]; axes=list(patient_left_axis or [None]*b)
    for bi in range(b):
        labels=hard[bi]; grad=_boundary_gradient(img[bi]); stats=[]
        for rid in range(k):
            mask=labels==rid; n=int(mask.sum())
            if n<cfg.min_region_pixels: stats.append(None); continue
            yy,xx=np.nonzero(mask)
            border=np.zeros_like(mask); border[0]=mask[0]; border[-1]=mask[-1]; border[:,0]|=mask[:,0]; border[:,-1]|=mask[:,-1]
            eroded=mask&_shift(mask,1,0)&_shift(mask,-1,0)&_shift(mask,0,1)&_shift(mask,0,-1); rim=mask&(~eroded)
            stats.append({"mask":mask,"n":n,"cy":float(yy.mean()),"cx":float(xx.mean()),
                          "border":float(border.sum())/max(float(n),1.0),
                          "motion":float(mot[bi][mask].mean()),
                          "boundary":float(grad[rim].mean()) if np.any(rim) else 0.0,
                          "holes":_holes(mask)})
        mv=[s["motion"] for s in stats if s is not None]; mlo=min(mv) if mv else 0.; mhi=max(mv) if mv else 1.; den=max(mhi-mlo,1e-6)
        for rid,s in enumerate(stats):
            if s is None: continue
            mn=(s["motion"]-mlo)/den
            logits[bi,rid,BG]+=cfg.border_weight*s["border"]+cfg.low_motion_bg_weight*(1-mn)
            logits[bi,rid,MYO]+=cfg.boundary_weight*np.tanh(s["boundary"])
        enclosures=[]
        for outer,so in enumerate(stats):
            if so is None or not np.any(so["holes"]): continue
            for inner,si in enumerate(stats):
                if inner==outer or si is None: continue
                inside=float((si["mask"]&so["holes"]).sum())/max(float(si["n"]),1.)
                if inside>=0.80:
                    logits[bi,outer,MYO]+=cfg.enclosure_weight*inside; logits[bi,inner,LV]+=cfg.enclosure_weight*inside
                    enclosures.append((outer,inner,inside))
        if enclosures:
            outer,inner,_=max(enclosures,key=lambda x:x[2])
            candidates=[inner]
            for rid,s in enumerate(stats):
                if s is None or rid in (outer,inner): continue
                if _adjacent(s["mask"],stats[outer]["mask"]):
                    logits[bi,rid,RV]+=cfg.adjacency_weight; candidates.append(rid)
            axis=axes[bi] if bi<len(axes) else None
            if axis in {"+x","-x","+y","-y"} and len(candidates)>=2:
                def proj(rid):
                    val=stats[rid]["cx"] if axis.endswith("x") else stats[rid]["cy"]
                    return val if axis.startswith("+") else -val
                order=sorted(candidates,key=proj,reverse=True)
                logits[bi,order[0],LV]+=cfg.orientation_weight; logits[bi,order[-1],RV]+=cfg.orientation_weight
        diags.append({"enclosures":enclosures})
    return torch.from_numpy(logits).to(region_prob.device,region_prob.dtype),diags
