"""Image-only objectives. Seed loss uses RAW neural predictions, not target-added logits."""
from __future__ import annotations
import torch
from torch.nn import functional as F

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

def seed_cross_entropy(semantic_prob,evidence_logits,valid_region):
    if not bool(valid_region.any()): return semantic_prob.sum()*0.0
    target=evidence_logits.detach().argmax(-1)
    return F.nll_loss(semantic_prob.clamp_min(1e-8).log()[valid_region],target[valid_region])

def teacher_loss(outputs,cur_25d,evidence_logits,valid_region,*,w_recon=1.,w_proto=.1,w_seed=1.,w_motion=1.,w_motion_smooth=.05):
    photo,smooth=registration_loss(outputs,cur_25d)
    # Explicitly recover raw probabilities, even if the caller also returns guided ones.
    raw_prob=outputs["semantic_logits"].softmax(-1)
    terms={"reconstruction":reconstruction_loss(outputs["reconstruction"],cur_25d),
           "prototype":prototype_information_loss(outputs["region_prob"]),
           "semantic_seed":seed_cross_entropy(raw_prob,evidence_logits,valid_region),
           "motion_photo":photo,"motion_smooth":smooth}
    terms["total"]=(w_recon*terms["reconstruction"]+w_proto*terms["prototype"]+w_seed*terms["semantic_seed"]
                    +w_motion*photo+w_motion_smooth*smooth)
    return terms
