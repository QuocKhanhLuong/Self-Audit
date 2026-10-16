"""Frozen-evidence losses. Confidence in a source is not semantic correctness."""
from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F


def evidence_distribution(mass: Tensor, support: Tensor, min_support: float=1e-6):
    """Normalize nonnegative class MASS by SUPPORT, never invert uncertain Q.

    mass: B,C,H,W; support: B,H,W; must originate from the same positive
    transport applied to source evidence. Invalid cells return uniform ONLY
    as storage; returned valid=False must zero their semantic weights.
    """
    mass,support=mass.detach(),support.detach()
    if mass.ndim!=4 or support.shape!=mass.shape[:1]+mass.shape[2:] or min_support<=0:
        raise ValueError('evidence shapes or cutoff invalid')
    if (not torch.isfinite(mass).all() or not torch.isfinite(support).all()
        or (mass<0).any() or (support<0).any()
        or not torch.allclose(mass.sum(1),support,atol=1e-6,rtol=1e-5)):
        raise ValueError('mass/support must be nonnegative and consistent')
    valid=support>=min_support
    z=mass/support.clamp_min(min_support)[:,None]
    z=torch.where(valid[:,None],z,torch.full_like(z,1/mass.shape[1]))
    return z,valid


def soft_ce(logits: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    target,weights=target.detach(),weights.detach()
    if target.shape!=logits.shape or weights.shape!=logits.shape[:1]+logits.shape[2:]:
        raise ValueError('aligned NCHW target and NHW weights required; no implicit resize')
    if (not torch.isfinite(target).all() or not torch.isfinite(weights).all()
        or (target<0).any() or (weights<0).any()
        or not torch.allclose(target.sum(1),torch.ones_like(weights),atol=1e-5,rtol=0)):
        raise ValueError('invalid targets/weights')
    numerator=(-(target*logits.log_softmax(1)).sum(1)*weights).sum()
    return numerator/weights.sum().clamp_min(1e-8)


def set_loss(logits: Tensor, admissible: Tensor, weights: Tensor) -> Tensor:
    if admissible.shape!=logits.shape or admissible.dtype!=torch.bool:
        raise ValueError('admissible must be a Boolean NCHW label set')
    if not admissible.any(1).all(): raise ValueError('empty candidate label set')
    if weights.shape!=logits.shape[:1]+logits.shape[2:] or (weights<0).any() or not torch.isfinite(weights).all():
        raise ValueError('invalid weights')
    w=weights.detach()
    log_allowed=torch.logsumexp(logits.masked_fill(~admissible,float('-inf')),1)
    return ((torch.logsumexp(logits,1)-log_allowed)*w).sum()/w.sum().clamp_min(1e-8)


def masked_feature_loss(student: Tensor, teacher: Tensor, masked_valid: Tensor) -> Tensor:
    """Student must have seen a masked view; caller must align valid positions.

    The Boolean mask is structural, not a semantic confidence gate. This
    function cannot itself verify whether the input was actually masked.
    """
    teacher=teacher.detach()
    if student.shape!=teacher.shape or masked_valid.shape!=student.shape[:1]+student.shape[2:]:
        raise ValueError('feature/mask shapes disagree')
    valid=masked_valid.detach().bool() & (teacher.square().sum(1)>1e-10)
    if not valid.any(): return student.sum()*0
    return (1-F.cosine_similarity(student,teacher,dim=1))[valid].mean()
