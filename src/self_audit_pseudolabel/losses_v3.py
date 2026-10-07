"""Image-only objectives. Seed loss uses RAW neural predictions, not target-added logits."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F

@dataclass(frozen=True)
class BootstrapLossConfig:
    """Explicit image-only experimental recipe; absent from the legacy path."""
    spatial_weight: float=.3
    region_reconstruction_weight: float=1.0
    edge_scale: float=.25

    def __post_init__(self):
        for name in ('spatial_weight','region_reconstruction_weight','edge_scale'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'bootstrap {name} must be finite')
            if value<0 or (name=='edge_scale' and value==0):
                raise ValueError(f'invalid bootstrap {name}')
        if self.spatial_weight+self.region_reconstruction_weight==0:
            raise ValueError('bootstrap requires an image-region learning objective')


def _region_image(region_prob,image):
    value=F.interpolate(image.detach(),region_prob.shape[-2:],mode='bilinear',align_corners=False)
    value=value-value.mean((-2,-1),keepdim=True)
    scale=value.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    return value/scale


def spatial_continuity_loss(region_prob,image,*,edge_scale=.25):
    """Edge-aware local total variation, no wrap or supplied region targets.

    Image differences are normalized per slice. Existing information loss and
    region reconstruction oppose trivial merging; TV alone is not sufficient.
    """
    if not math.isfinite(edge_scale) or edge_scale<=0: raise ValueError('edge_scale must be positive')
    value=_region_image(region_prob,image)
    numerator=region_prob.sum()*0.; denominator=region_prob.new_zeros(())
    for dim in (-2,-1):
        if region_prob.shape[dim]<2: continue
        delta_q=region_prob.diff(dim=dim).abs().sum(1,keepdim=True)*.5
        affinity=torch.exp(-.5*(value.diff(dim=dim)/edge_scale).square())
        numerator=numerator+(affinity*delta_q).sum()
        denominator=denominator+affinity.sum()
    return numerator/denominator.clamp_min(1e-8)


def region_reconstruction_loss(region_prob,image):
    """Soft piecewise-constant intensity distortion through the assignments.

    Unlike the appearance decoder, this image-only reconstruction explicitly
    trains q. Means are estimated from the input; no segmentation is supplied.
    """
    value=_region_image(region_prob,image)
    mass=region_prob.sum((-2,-1)).clamp_min(1e-6)
    mean=(region_prob*value).sum((-2,-1))/mass
    return (region_prob*(value-mean[:,:,None,None]).square()).sum(1).mean()

def reconstruction_loss(reconstruction,cur_25d):
    # Ordinary image reconstruction, NOT claimed to be masked reconstruction.
    return F.smooth_l1_loss(reconstruction,cur_25d[:,1:2])

def prototype_information_loss(region_prob,eps=1e-8):
    p=region_prob.clamp_min(eps); cond=-(p*p.log()).sum(1).mean()
    marginal=p.mean(dim=(0,2,3))
    return cond+(marginal*marginal.clamp_min(eps).log()).sum()

def registration_loss(outputs,cur_25d):
    cur=cur_25d[:,1:2]
    photo=.5*(F.smooth_l1_loss(outputs["warped_prev"],cur)+F.smooth_l1_loss(outputs["warped_next"],cur))
    def smooth(flow):
        value=flow.sum()*0.0
        if flow.shape[-1]>1: value=value+(flow[:,:,:,1:]-flow[:,:,:,:-1]).abs().mean()
        if flow.shape[-2]>1: value=value+(flow[:,:,1:,:]-flow[:,:,:-1,:]).abs().mean()
        return value
    return photo,.5*(smooth(outputs["flow_prev"])+smooth(outputs["flow_next"]))

def seed_cross_entropy(semantic_logits,evidence_logits,valid_region):
    if not bool(valid_region.any()): return semantic_logits.sum()*0.0
    target=evidence_logits.detach().argmax(-1)
    return F.cross_entropy(semantic_logits[valid_region],target[valid_region])

def teacher_loss(outputs,cur_25d,evidence_logits,valid_region,*,w_recon=1.,w_proto=.1,w_seed=1.,w_motion=1.,w_motion_smooth=.05):
    photo,smooth=registration_loss(outputs,cur_25d)
    # Stable raw-logit CE keeps gradients even for confidently wrong heads.
    terms={"reconstruction":reconstruction_loss(outputs["reconstruction"],cur_25d),
           "prototype":prototype_information_loss(outputs.get("anonymous_region_prob",outputs["region_prob"])),
           "semantic_seed":seed_cross_entropy(outputs["semantic_logits"],evidence_logits,valid_region),
           "motion_photo":photo,"motion_smooth":smooth}
    terms["total"]=(w_recon*terms["reconstruction"]+w_proto*terms["prototype"]+w_seed*terms["semantic_seed"]
                    +w_motion*photo+w_motion_smooth*smooth)
    return terms
