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
    from scipy.ndimage import label as components
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
                          "holes":_holes(mask), "connected":components(mask)[1]==1})
        mv=[s["motion"] for s in stats if s is not None]; mlo=min(mv) if mv else 0.; mhi=max(mv) if mv else 1.; den=max(mhi-mlo,1e-6)
        for rid,s in enumerate(stats):
            if s is None: continue
            mn=(s["motion"]-mlo)/den
            logits[bi,rid,BG]+=cfg.border_weight*s["border"]+cfg.low_motion_bg_weight*(1-mn)
            if not s["border"] and s["connected"]:
                logits[bi,rid,MYO]+=cfg.boundary_weight*np.tanh(s["boundary"])
        # A semantic vote applies to an entire anonymous region. Disconnected
        # regions cannot be certified by a single component that happens to fit.
        # Exterior regions may surround every organ, but cannot be cardiac walls.
        enclosures=[]
        for outer,so in enumerate(stats):
            if so is None or so["border"] or not so["connected"]: continue
            holes=so["holes"]
            if components(holes)[1]!=1: continue
            for inner,si in enumerate(stats):
                if inner==outer or si is None or si["border"] or not si["connected"]: continue
                if np.any(si["holes"]) or not _adjacent(so["mask"],si["mask"]): continue
                overlap=float((si["mask"]&holes).sum())
                inside=overlap/si["n"]
                coverage=overlap/float(holes.sum())
                if inside>=.80 and coverage>=.80:
                    enclosures.append((outer,inner,min(inside,coverage)))
        # Aggregate physical pair votes symmetrically, bounded by one enclosure
        # weight per class. No max/argmax tie can select an arbitrary region ID.
        votes=np.zeros((k,4),dtype=np.float32)
        axis=axes[bi] if bi<len(axes) else None
        for outer,inner,strength in enclosures:
            pair=np.zeros_like(votes)
            pair[outer,MYO]=cfg.enclosure_weight*strength
            pair[inner,LV]=cfg.enclosure_weight*strength
            candidates=[inner]
            for rid,s in enumerate(stats):
                if s is None or rid in (outer,inner) or s["border"] or not s["connected"]: continue
                if np.any(s["holes"]): continue
                if _adjacent(s["mask"],stats[outer]["mask"]):
                    pair[rid,RV]=cfg.adjacency_weight
                    candidates.append(rid)
            if axis in {"+x","-x","+y","-y"} and len(candidates)>=2:
                def proj(rid):
                    val=stats[rid]["cx"] if axis.endswith("x") else stats[rid]["cy"]
                    return val if axis.startswith("+") else -val
                values=np.asarray([proj(rid) for rid in candidates])
                # A tied physical projection supplies no orientation evidence.
                high=np.flatnonzero(np.isclose(values,values.max(),rtol=0,atol=1e-6))
                low=np.flatnonzero(np.isclose(values,values.min(),rtol=0,atol=1e-6))
                if len(high)==len(low)==1 and high[0]!=low[0]:
                    pair[candidates[int(high[0])],LV]+=cfg.orientation_weight
                    pair[candidates[int(low[0])],RV]+=cfg.orientation_weight
            votes=np.maximum(votes,pair)
        logits[bi]+=votes
        diags.append({"enclosures":enclosures,"patient_left_axis":axis,
                      "rv_support_policy":"cardinal_orientation_required_at_default_gate"})
    return torch.from_numpy(logits).to(region_prob.device,region_prob.dtype),diags
