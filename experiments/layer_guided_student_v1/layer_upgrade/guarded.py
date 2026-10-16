"""Reference class-conditional gradient projection and verified additive step.

The guarantee concerns fixed TRAIN proxy losses, not anatomy or future data.
Do not pass a projected gradient to AdamW and claim the same guarantee.
"""
from __future__ import annotations
from collections.abc import Callable, Sequence
from itertools import combinations
import math
import torch
from torch import Tensor, nn


def project_direction(candidate: Tensor, guards: Tensor, tol: float=1e-9):
    """min_d ||d-candidate||^2/2 subject to guards @ d >= 0.

    Exact active-set enumeration is feasible here for <=8 guard losses. Uses
    float64 for the tiny Gram solve and vectors; intended as a correctness
    reference on a selected parameter block, not a huge-backbone optimizer.
    A conservative zero fallback is disclosed if no numerical KKT point passes.
    """
    if candidate.ndim!=1 or guards.ndim!=2 or guards.shape[1]!=candidate.numel():
        raise ValueError('expected vector candidate and row guard gradients')
    if len(guards)>8 or not torch.isfinite(candidate).all() or not torch.isfinite(guards).all():
        raise ValueError('need <=8 finite guards and finite direction')
    d0=candidate.detach().double(); g=guards.detach().to(d0)
    norms=g.norm(dim=1); g=g[norms>0]; norms=norms[norms>0]
    if len(g)==0:
        return candidate.detach().clone(),dict(status='NO_NONZERO_GUARDS',active=[])
    g=g/norms[:,None]
    scale=max(1.,float(d0.norm())); eps=tol*scale
    if bool((g@d0>=-eps).all()):
        return candidate.detach().clone(),dict(status='ALREADY_FEASIBLE',active=[])
    best=None; best_obj=float('inf'); best_set=[]
    for n in range(1,len(g)+1):
        for ids in combinations(range(len(g)),n):
            a=g[list(ids)]; gram=a@a.T; rhs=-(a@d0)
            lam=torch.linalg.pinv(gram,rtol=1e-12)@rhs
            if float((gram@lam-rhs).norm())>eps or bool((lam < -eps).any()): continue
            d=d0+a.T@lam
            if bool((g@d < -eps).any()): continue
            value=float((d-d0).square().sum())
            if value<best_obj: best,best_obj,best_set=d,value,list(ids)
    if best is None:
        best=torch.zeros_like(d0); status='CONSERVATIVE_ZERO_NUMERICAL_FALLBACK'
    else: status='KKT_VERIFIED'
    result=best.to(candidate)
    # Casting can cause a tiny violation. Finite-step acceptance is separately checked.
    return result,dict(status=status,active=best_set,
                       min_normalized_inner_product=float((g@result.double()).min()),
                       correction_norm=float((best-d0).norm()))


def _flat_grads(loss: Tensor, params: Sequence[nn.Parameter]) -> Tensor:
    if loss.ndim!=0 or not torch.isfinite(loss): raise ValueError('finite scalar loss required')
    if not loss.requires_grad: return torch.cat([torch.zeros_like(p).reshape(-1) for p in params])
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)])


def guarded_sgd_step(params: Sequence[nn.Parameter],
                     objective: Callable[[],Tensor],
                     guard_losses: Callable[[],Sequence[Tensor]], *,
                     step_size: float, max_backtracks: int=16,
                     atol: float=1e-10) -> dict:
    """Apply a tested additive step, or restore every selected parameter.

    Closures MUST be deterministic, side-effect-free, and evaluate frozen
    targets/masks on exactly the same batch. Use eval mode/no dropout. Nonselected
    parameters must remain fixed. No optimizer momentum, decay or state update.
    This function does not populate .grad. Only this selected parameter block
    receives an update. Rejection can legitimately stall progress.
    """
    params=tuple(p for p in params if p.requires_grad)
    if not params or len({id(p) for p in params})!=len(params): raise ValueError('unique nonempty trainable parameters required')
    if not math.isfinite(step_size) or step_size<=0 or max_backtracks<0 or atol<0:
        raise ValueError('invalid step/acceptance settings')
    if len({p.device for p in params})!=1 or len({p.dtype for p in params})!=1:
        raise ValueError('reference implementation needs one device and dtype')
    baseline_objective=objective()
    baseline_guards=list(guard_losses())
    if not baseline_guards: raise ValueError('at least one protected fixed loss required')
    base_values=[float(v.detach()) for v in baseline_guards]
    base_obj=float(baseline_objective.detach())
    candidate=_flat_grads(baseline_objective,params)
    guards=torch.stack([_flat_grads(v,params) for v in baseline_guards])
    direction,projection=project_direction(candidate,guards)
    saved=[p.detach().clone() for p in params]

    def restore():
        with torch.no_grad():
            for p,old in zip(params,saved): p.copy_(old)

    accepted=False; eta=step_size; values=base_values; obj=base_obj
    try:
        for attempt in range(max_backtracks+1):
            with torch.no_grad():
                off=0
                for p,old in zip(params,saved):
                    d=direction[off:off+p.numel()].view_as(p); off+=p.numel()
                    p.copy_(old-eta*d)
                trial=list(guard_losses()); trial_obj=objective()
            if len(trial)!=len(base_values): raise ValueError('guard set changed between evaluations')
            values=[float(v.detach()) for v in trial]; obj=float(trial_obj.detach())
            finite=all(math.isfinite(v) for v in [*values,obj])
            if finite and all(v<=b+atol for v,b in zip(values,base_values)) and obj<=base_obj+atol:
                accepted=True; break
            eta*=.5
        if not accepted: restore(); values=base_values; obj=base_obj; eta=0.
    except BaseException:
        restore(); raise
    return dict(accepted=accepted,step_size=eta,projection=projection,
                guard_before=base_values,guard_after=values,
                objective_before=base_obj,objective_after=obj,
                direction_norm=float(direction.norm()),
                scope='fixed_proxy_losses_selected_parameters_no_optimizer_state',
                tolerance=atol)
