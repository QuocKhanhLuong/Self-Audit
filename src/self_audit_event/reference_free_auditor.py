"""Explicit intrinsic image compatibility. Never an estimator of absolute truth.

The fixed energy is invariant to semantic label permutations. Its learned risk
map is a distillation target, not GT-derived error supervision.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .annotation import AnnotationState


@dataclass(frozen=True)
class AuditObservation:
    image: torch.Tensor
    logits: torch.Tensor

    def __post_init__(self):
        if self.image.ndim!=4 or self.image.shape[1]!=3 or not self.image.is_floating_point():
            raise ValueError('floating image [B,3,H,W] required')
        if self.logits.ndim!=4 or self.logits.shape[1]!=4 or not self.logits.is_floating_point():
            raise ValueError('floating four-class logits required; raw masks are forbidden')
        if (len(self.image),*self.image.shape[-2:])!=(len(self.logits),*self.logits.shape[-2:]):
            raise ValueError('image/logit shapes mismatch')

    @classmethod
    def from_state(cls,image,state):
        if not isinstance(state,AnnotationState): raise TypeError('AnnotationState required, not a mask/target')
        return cls(image.detach().clone(),state.logits.detach().clone())


class FrozenImageAnchor(nn.Module):
    def __init__(self):
        super().__init__()
        kernel=torch.tensor([[[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],
                             [[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]])/8
        self.register_buffer('sobel',kernel[:,None])

    def forward(self,image):
        if image.ndim!=4 or image.shape[1]!=3: raise ValueError('2.5D image requires three channels')
        center=image.detach()[:,1:2].float()
        center=(center-center.mean((2,3),keepdim=True))/center.var((2,3),unbiased=False,keepdim=True).clamp_min(1e-4).sqrt()
        padded=F.pad(center,(1,1,1,1),mode='replicate')
        local=F.avg_pool2d(padded,3,stride=1)
        edges=F.conv2d(padded,self.sobel.to(center))
        return torch.cat([center,local,edges],1)


class ReferenceFreeAuditor(nn.Module):
    def __init__(self,num_classes=4,hidden=16):
        super().__init__()
        if num_classes!=4: raise ValueError('fixed four-class contract')
        self.anchor=FrozenImageAnchor()
        self.risk=nn.Sequential(nn.Conv2d(4+num_classes,hidden,3,padding=1),nn.GELU(),
                                nn.Conv2d(hidden,1,1),nn.Softplus())

    def forward(self,observation):
        if not isinstance(observation,AuditObservation): raise TypeError('AuditObservation required')
        z=self.anchor(observation.image.detach())
        p=observation.logits.detach().float().softmax(1)
        return self.risk(torch.cat([z,p],1))

    @torch.no_grad()
    def intrinsic_risk(self,observation):
        if not isinstance(observation,AuditObservation): raise TypeError('AuditObservation required')
        z=self.anchor(observation.image.detach());b,c,h,w=z.shape
        labels=observation.logits.detach().argmax(1)
        groups=F.one_hot(labels,4).permute(0,3,1,2).float().flatten(2)
        means=groups@z.flatten(2).transpose(1,2)/groups.sum(2,keepdim=True).clamp_min(1)
        assigned=means.gather(1,labels.flatten(1)[:,:,None].expand(-1,-1,c)).transpose(1,2).reshape(b,c,h,w)
        variance=z.var((2,3),unbiased=False,keepdim=True)+1e-4
        residual=((z-assigned).square()/variance).mean(1,keepdim=True)
        horizontal=(labels[:,:,1:]!=labels[:,:,:-1]).float()*torch.exp(-(z[:,:,:,1:]-z[:,:,:,:-1]).square().mean(1))
        vertical=(labels[:,1:,:]!=labels[:,:-1,:]).float()*torch.exp(-(z[:,:,1:,:]-z[:,:,:-1,:]).square().mean(1))
        boundary=(F.pad(horizontal,(0,1,0,0))+F.pad(vertical,(0,0,0,1)))/2
        return residual+.05*boundary[:,None]

    @torch.no_grad()
    def quality(self,observation):
        return -self.intrinsic_risk(observation).mean((1,2,3))


def rf_loss(auditor,observation):
    return F.mse_loss(auditor(observation),auditor.intrinsic_risk(observation).detach())
