"""One optional audit-guided read. SKIP is exact identity; no GT arguments."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from .annotation import AnnotationExpert, AnnotationState
from .reference_free_auditor import ReferenceFreeAuditor, AuditObservation
from .audit_trigger import AuditTrigger


@dataclass
class EventOutput:
    initial: AnnotationState
    final_logits: torch.Tensor
    decisions: torch.Tensor
    trigger_features: torch.Tensor
    predicted_values: torch.Tensor
    audit_calls: int
    read_calls: int


def subset(state,indices):
    return AnnotationState(state.features.index_select(0,indices),state.logits.index_select(0,indices),
                           state.coordinates.index_select(0,indices))


class EventAuditModel(nn.Module):
    def __init__(self,channels=16,num_classes=4,k=8):
        super().__init__()
        if num_classes!=4: raise ValueError('diagnostic uses fixed 4-class ACDC contract')
        self.actor=AnnotationExpert(channels,num_classes,k)
        self.auditor=ReferenceFreeAuditor(num_classes=num_classes)
        self.trigger=AuditTrigger(channels)

    def forward(self,image,policy='learned',budget_fraction=.5,generator=None,step=0,period=2):
        if not 0<=budget_fraction<=1 or period<1: raise ValueError('invalid allocation controls')
        state=self.actor.start(image)
        features=self.trigger.state_features(state)
        with torch.no_grad(): values=self.trigger(features)
        b=len(image);decision=torch.zeros(b,dtype=torch.bool,device=image.device)
        count=math.floor(b*budget_fraction)
        if policy=='learned': decision=self.trigger.select(values,budget_fraction)
        elif policy=='always': decision[:]=True
        elif policy=='none': pass
        elif policy=='periodic': decision[:]=int(step)%period==0
        elif policy=='random': decision[torch.randperm(b,generator=generator,device='cpu')[:count].to(image.device)]=True
        elif policy=='entropy':
            with torch.no_grad():
                p=state.logits.detach().softmax(1)
                entropy=-(p*p.clamp_min(1e-8).log()).sum(1).mean((1,2))
                decision[torch.argsort(entropy,descending=True,stable=True)[:count]]=True
        else: raise ValueError('unknown policy '+str(policy))
        ids=torch.where(decision)[0];final=state.logits
        if len(ids):
            selected=subset(state,ids)
            obs=AuditObservation.from_state(image.index_select(0,ids),selected)
            with torch.no_grad(): guidance=self.auditor(obs)
            after=self.actor.refine(selected,guidance.detach())
            final=state.logits.index_copy(0,ids,after.logits)
        return EventOutput(state,final,decision.detach(),features,values.detach(),len(ids),b+len(ids))

    @torch.no_grad()
    def twins(self,image,state=None,indices=None):
        if state is None: state=self.actor.start(image)
        if indices is None: indices=torch.arange(len(image),device=image.device)
        indices=indices.to(image.device,dtype=torch.long)
        selected=subset(state,indices);x=image.index_select(0,indices)
        obs=AuditObservation.from_state(x,selected)
        guidance=self.auditor(obs)
        after=self.actor.refine(selected,guidance)
        delta=self.auditor.quality(AuditObservation.from_state(x,after))-self.auditor.quality(obs)
        return {'observation':obs,'trigger_features':self.trigger.state_features(selected),
                'audit_logits':after.logits.detach(),'skip_logits':selected.logits.detach().clone(),
                'guidance':guidance.detach(),'coordinates':after.coordinates.detach(),
                'intrinsic_delta':delta.detach(),
                'costs':{'audit_calls':len(indices),'read_calls':len(indices),'quality_evaluations':2*len(indices)}}
