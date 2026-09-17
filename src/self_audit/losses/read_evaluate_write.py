"""Training-only REW targets/losses. Inference never imports these targets."""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .annotation import annotation_loss
from ..models.read_evaluate_write import CC, FIX, REGRESS, WW, Outcome


def validate_target(target: Tensor, shape: tuple[int,...]) -> None:
    if tuple(target.shape) != shape or target.dtype != torch.long:
        raise ValueError("target must be int64 [B,H,W], matching predictions")
    if bool(((target < 0) | (target > 3)).any()):
        raise ValueError("unknown target labels; expected BG=0,RV=1,MYO=2,LV=3")


@torch.no_grad()
def transition_target(previous: Tensor, candidate: Tensor, target: Tensor) -> Tensor:
    if previous.shape != candidate.shape or previous.ndim != 4 or previous.shape[1] != 4:
        raise ValueError("expected matching four-class logits")
    validate_target(target,(previous.shape[0],*previous.shape[-2:]))
    pc = previous.detach().argmax(1) == target
    cc = candidate.detach().argmax(1) == target
    result = torch.full_like(target,WW)
    result[pc & cc] = CC
    result[~pc & cc] = FIX
    result[pc & ~cc] = REGRESS
    return result


@torch.no_grad()
def slice_dice(logits: Tensor, target: Tensor) -> Tensor:
    """FG macro slice-proxy; both-empty excluded, all-FG-empty=1 convention.

    Not native 3D quality, calibration, or evidence of external validity.
    """
    validate_target(target,(logits.shape[0],*logits.shape[-2:]))
    pred = logits.detach().argmax(1)
    scores, available = [], []
    for cls in range(1,4):
        p,y = pred == cls, target == cls
        den = p.flatten(1).sum(1)+y.flatten(1).sum(1)
        scores.append(2*(p&y).flatten(1).sum(1)/den.clamp_min(1))
        available.append(den>0)
    s,a = torch.stack(scores,1),torch.stack(available,1)
    return torch.where(a.any(1),(s*a).sum(1)/a.sum(1).clamp_min(1),torch.ones_like(s[:,0]))


def outcome_loss(outcome: Outcome, previous: Tensor, candidate: Tensor, target: Tensor) -> tuple[Tensor,Tensor]:
    truth = transition_target(previous,candidate,target)
    change = slice_dice(candidate,target)-slice_dice(previous,target)
    # No class-reweighting in probability head: reweighting alters class priors.
    # Paired utility supervision below supplies additional local FIX/R signal.
    local = F.cross_entropy(outcome.local_logits.float(),truth)
    return local + F.smooth_l1_loss(outcome.delta_q,change),truth


def _tile_mean(value: Tensor, tile: int) -> Tensor:
    h,w=value.shape[-2:]
    pad=(0,(-w)%tile,0,(-h)%tile)
    v=F.avg_pool2d(F.pad(value,pad),tile,tile)
    count=F.avg_pool2d(F.pad(torch.ones_like(value[:,:1]),pad),tile,tile)
    return v/count


def compute_rew_losses(model, output: dict, target: Tensor, *, proposal_weight: float = .5,
                       audit_weight: float = 1., read_advantage_weight: float = .2,
                       synthetic_weight: float = .25) -> tuple[Tensor,dict]:
    """Always teach actual proposals, even when the detached hard gate rejects.

    Natural all-proposal audit and paired read-advantage targets share the same
    actor state. Advantage gradients update the auditor, NEVER actor features.
    Actor/geometry learn from GT segmentation losses on all proposed reads.
    No claim of optimizing utility through a critic or of a distillation stage.
    """
    initial=output['initial']
    validate_target(target,(initial.shape[0],*initial.shape[-2:]))
    actor=annotation_loss(output['logits'].float(),target)[0]
    actor=actor + .5*annotation_loss(initial.float(),target)[0]
    state_loss,state_truth=outcome_loss(output['state_outcome'],initial,initial,target)
    proposal_terms=[];audit_terms=[];adv_terms=[]
    counts=torch.zeros(4,device=target.device,dtype=torch.long)
    counts+=torch.bincount(state_truth.flatten(),minlength=4)
    choice_counts=torch.zeros(4,device=target.device,dtype=torch.long)
    attempts=0;accepted=0;foreground=0
    for turn in output['turns']:
        ids=turn['ids']; y=target.index_select(0,ids);prev=turn['previous']
        predicted_util=[];true_util=[]
        for candidate,outcome in zip(turn['candidates'],turn['outcomes']):
            proposal_terms.append(annotation_loss(candidate.float(),y)[0])
            term,truth=outcome_loss(outcome,prev,candidate,y)
            audit_terms.append(term);counts+=torch.bincount(truth.flatten(),minlength=4)
            true_util.append(_tile_mean(((truth==FIX).float()-model.harm_weight*(truth==REGRESS).float())[:,None],model.tile_size))
            p=outcome.probabilities
            predicted_util.append(_tile_mean(p[:,FIX:FIX+1]-model.harm_weight*p[:,REGRESS:REGRESS+1],model.tile_size))
        # Local-only read is the matched control, not an oracle input to sampling.
        for j in (1,2):
            adv_terms.append(F.mse_loss(predicted_util[j]-predicted_util[0],true_util[j]-true_util[0]))
        term,_=outcome_loss(turn['assembled_outcome'],prev,turn['assembled'],y)
        audit_terms.append(term)
        attempts+=ids.numel();accepted+=int(turn['accepted'].sum().detach())
        choice_counts+=torch.bincount(turn['choice'].detach().flatten(),minlength=4)
        foreground+=int((turn['assembled'].detach().argmax(1)>0).sum())
    zero=initial.sum()*0.
    proposal=torch.stack(proposal_terms).mean() if proposal_terms else zero
    natural=torch.stack(audit_terms).mean() if audit_terms else state_loss*0
    advantage=torch.stack(adv_terms).mean() if adv_terms else state_loss*0
    synthetic=state_loss*0
    if synthetic_weight:
        # Same-patient, training-only partial repair / label disturbance. Forward
        # inference finished before GT is used. Both directions are measured,
        # never labeled "good" or "bad" merely by the edit name.
        with torch.no_grad():
            labels=initial.detach().argmax(1)
            coarse=torch.rand((labels.shape[0],1,max(1,labels.shape[1]//32),max(1,labels.shape[2]//32)),device=labels.device)
            region=F.interpolate(coarse,size=labels.shape[-2:],mode='nearest')[:,0]
            edited=torch.where(region<.25,target,labels)
            edited=torch.where(region>.75,(labels+1)%4,edited)
            draft=4.*F.one_hot(edited,4).permute(0,3,1,2).float()
        first=model.auditor(output['features'],initial,draft)
        reverse=model.auditor(output['features'],draft,initial)
        synthetic=.5*(outcome_loss(first,initial,draft,target)[0]+outcome_loss(reverse,draft,initial,target)[0])
    actor=actor+proposal_weight*proposal
    audit=state_loss+natural+read_advantage_weight*advantage+synthetic_weight*synthetic
    total=actor+audit_weight*audit
    metrics={"loss":float(total.detach()),"actor_loss":float(actor.detach()),
             "proposal_loss":float(proposal.detach()),"auditor_loss":float(audit.detach()),
             "state_loss":float(state_loss.detach()),"read_advantage_loss":float(advantage.detach()),
             "synthetic_loss":float(synthetic.detach()),"attempts":attempts,"accepted":accepted,
             "choices_pixels":choice_counts.detach().cpu().tolist(),"audit_targets_pixels":counts.detach().cpu().tolist(),
             "candidate_foreground_pixels":foreground}
    # Only tests use separate differentiable terms to assert firewall behavior.
    return total,dict(metrics=metrics,actor_tensor=actor,audit_tensor=audit)
