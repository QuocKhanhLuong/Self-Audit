"""A frozen image-energy refinement primitive, not a semantic correctness oracle."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor


@dataclass
class FrozenImageEnergy:
    """E(p)=<unary,p>+lam/2 sum_e a_e||p_i-p_j||²
                    +beta/2 sum_i w_i||p_i-anchor_i||².

    Edges list each undirected edge once (repeated edges are allowed but count
    repeatedly). Unary, edges, weights and anchors are frozen during refinement.
    All terms are summed, not averaged; the Lipschitz constant matches this sum.
    Class meaning must come from justified source anchors/unaries, NOT graph
    smoothness. Missing source evidence should have anchor_weight=0.
    """
    unary: Tensor
    edges: Tensor
    edge_weight: Tensor
    anchor: Tensor
    anchor_weight: Tensor
    lam: float=.2
    beta: float=.5

    def __post_init__(self):
        self.unary=self.unary.detach().double()
        device=self.unary.device
        self.edges=self.edges.detach().to(device=device,dtype=torch.long)
        self.edge_weight=self.edge_weight.detach().to(device=device,dtype=torch.double)
        self.anchor=self.anchor.detach().to(device=device,dtype=torch.double)
        self.anchor_weight=self.anchor_weight.detach().to(device=device,dtype=torch.double)
        n,c=self.unary.shape
        if c<2 or self.edges.ndim!=2 or self.edges.shape[0]!=2 or self.edge_weight.shape!=(self.edges.shape[1],):
            raise ValueError('invalid unary/graph shape')
        if self.anchor.shape!=(n,c) or self.anchor_weight.shape!=(n,): raise ValueError('invalid anchor shape')
        if self.edges.numel() and (self.edges.min()<0 or self.edges.max()>=n): raise ValueError('edge index out of range')
        if self.edges.numel() and (self.edges[0]==self.edges[1]).any(): raise ValueError('self edges must be omitted')
        if (not all(torch.isfinite(t).all() for t in (self.unary,self.edge_weight,self.anchor,self.anchor_weight))
            or (self.edge_weight<0).any() or (self.anchor_weight<0).any() or (self.anchor<0).any()
            or not torch.allclose(self.anchor.sum(-1),torch.ones(n,device=device,dtype=torch.double),atol=1e-6,rtol=0)
            or not math.isfinite(self.lam) or not math.isfinite(self.beta) or min(self.lam,self.beta)<0):
            raise ValueError('invalid frozen energy values')

    def value(self,p:Tensor)->Tensor:
        diff=p[self.edges[0]]-p[self.edges[1]]
        return ((self.unary*p).sum()+self.lam/2*(self.edge_weight[:,None]*diff.square()).sum()
                +self.beta/2*(self.anchor_weight[:,None]*(p-self.anchor).square()).sum())

    def gradient(self,p:Tensor)->Tensor:
        g=self.unary+self.beta*self.anchor_weight[:,None]*(p-self.anchor)
        diff=(p[self.edges[0]]-p[self.edges[1]])*self.edge_weight[:,None]*self.lam
        g=g.clone(); g.index_add_(0,self.edges[0],diff); g.index_add_(0,self.edges[1],-diff)
        return g

    def lipschitz_bound(self)->float:
        degree=torch.zeros(len(self.unary),device=self.unary.device,dtype=torch.double)
        for end in self.edges: degree.index_add_(0,end,self.edge_weight)
        return 2*self.lam*float(degree.max())+self.beta*float(self.anchor_weight.max())


def image_grid_graph(image:Tensor,sigma:float=.2):
    """Four-neighbor graph from a single frozen 2D image, no anatomical labels.

    Similar intensity does not imply same anatomical structure. Edge scale is
    an experimental choice, not a validated setting. No learned flow here.
    """
    if image.ndim!=2 or not torch.isfinite(image).all() or not math.isfinite(sigma) or sigma<=0:
        raise ValueError('finite 2D image and positive sigma required')
    h,w=image.shape; ids=torch.arange(h*w,device=image.device).view(h,w)
    left=torch.cat((ids[:,:-1].reshape(-1),ids[:-1,:].reshape(-1)))
    right=torch.cat((ids[:,1:].reshape(-1),ids[1:,:].reshape(-1)))
    edges=torch.stack((left,right))
    flat=image.detach().reshape(-1).double()
    weights=torch.exp(-(flat[left]-flat[right]).square()/(2*sigma*sigma))
    return edges,weights


@torch.no_grad()
def refine_distribution(logits:Tensor,energy:FrozenImageEnergy,steps:int=3,eta:float|None=None):
    """Entropic mirror-descent proposals, detached from the student graph.

    Analytic descent bound assumes positive p and exact arithmetic. Float64
    checks enforce numerical finiteness and nonincrease, with explicit 1e-10
    energy tolerance. Returns a proposal plus receipt, NOT an authorization
    that its anatomy is correct. Do not recompute the energy inside this loop.
    """
    if logits.shape!=energy.unary.shape or not torch.isfinite(logits).all() or steps<0:
        raise ValueError('invalid refinement input')
    logp=logits.detach().to(energy.unary).log_softmax(-1)
    p=logp.exp()
    if (p<=0).any(): raise ValueError('positive probability support required; logits are too extreme')
    L=energy.lipschitz_bound()
    step=(.5/L if L>0 else 1.) if eta is None else float(eta)
    if not math.isfinite(step) or step<=0 or (L>0 and step>1/L):
        raise ValueError('need 0<eta<=1/L for the stated analytic bound')
    trace=[]
    for k in range(steps):
        before=float(energy.value(p)); next_log=(logp-step*energy.gradient(p)).log_softmax(-1)
        q=next_log.exp()
        if not torch.isfinite(q).all() or (q<=0).any(): raise FloatingPointError('mirror step lost positive support')
        after=float(energy.value(q)); kl=float((q*(next_log-logp)).sum())
        bound=(1/step-L)*kl
        if after>before+1e-10: raise FloatingPointError('frozen image energy increased numerically')
        trace.append(dict(step=k+1,energy_before=before,energy_after=after,
                          kl_new_to_old=kl,guaranteed_drop_bound=bound,
                          inequality_slack=before-after-bound))
        p,logp=q,next_log
    return p,dict(L_bound=L,eta=step,trace=trace,
                  guarantee='frozen_image_proxy_energy_only_not_Dice',
                  student_parameters_updated=False,detached=True)
