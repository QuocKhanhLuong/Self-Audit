import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from self_audit.control import (solve_allocation, scalar_allocation, audit_weights, risks,
                                 make_candidates, controlled_head_step, response_moments)


def test_joint_counterexample():
    V = np.array([[.5, .15], [-.07, .02]])
    r = np.array([.8, .1]); A = np.einsum('gi,gj->gij', V, V)
    sol = solve_allocation(A, r[:, None]*V, .5*r*r, ridge=.001, source_ids=['a','b'])
    np.testing.assert_allclose(sol.alpha, [2/7, 1], atol=2e-6)
    assert np.all(sol.local_risks <= .5*r*r+1e-9)
    independent = .5*(r-V@np.array([0.,1.]))**2
    assert sol.local_risks.mean() < independent.mean()


@pytest.mark.parametrize('seed', range(8))
def test_random_feasibility(seed):
    rng=np.random.default_rng(seed); v=rng.normal(size=(3,12,4)); r=rng.normal(size=(3,12))
    A=np.einsum('gnm,gnk->gmk',v,v); b=np.einsum('gn,gnm->gm',r,v); e=.5*(r*r).sum(1)
    sol=solve_allocation(A,b,e,budget=0)
    assert np.all(sol.alpha>=0) and np.all(sol.alpha<=1)
    assert np.all(sol.local_risks<=e+1e-9)
    assert sol.objective<=e.mean()+1e-9


def test_scalar_formula_matches_numerical():
    a=np.array([.25,.0025]); b=np.array([.4,.001]); e=np.ones(2)
    x=scalar_allocation(a,b,ridge=.01)
    sol=solve_allocation(a[:,None,None],b[:,None],e,ridge=.01)
    assert abs(sol.alpha[0]-x)<1e-7
    assert scalar_allocation([1],[ -1], budget=0)==0


@pytest.mark.parametrize('ids',[['x','x'],['x']])
def test_duplicate_rejected(ids):
    with pytest.raises(ValueError):
        solve_allocation(np.stack([np.eye(2)]*3),np.ones((3,2)),np.ones(3),source_ids=ids)


def test_invalid_grams_and_budget():
    with pytest.raises(ValueError): solve_allocation(-np.ones((1,1,1)),[[1]],[1])
    with pytest.raises(ValueError): solve_allocation(np.ones((1,1,1)),[[1]],[1],budget=-1)
    with pytest.raises(ValueError): scalar_allocation([0],[1])


def test_no_candidates():
    sol=solve_allocation(np.empty((3,0,0)),np.empty((3,0)),np.ones(3))
    assert sol.status=='NO_CANDIDATES'


def test_weights_include_absent_class_negatives_and_ignore():
    y=torch.tensor([[[0,1],[255,0]]])
    w,t,pos=audit_weights(y)
    torch.testing.assert_close(w.flatten(1).sum(1),torch.ones(3,dtype=torch.float64))
    assert w[...,1,0].sum()==0 and pos==[[1],[0],[0]]
    p=torch.full((1,4,2,2),.25)
    before=risks(p,w,t)
    p[:,2]=.8
    assert risks(p,w,t)[1]>before[1]


def fixture():
    h=nn.Conv2d(4,4,1);nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
    f=torch.eye(4).T.reshape(1,4,2,2);y=torch.tensor([[[0,1],[2,3]]])
    q=F.one_hot(y,4).permute(0,3,1,2).float()
    return h,f,y,q


def test_candidate_sum_equals_global_pseudo_gradient():
    h,f,y,q=fixture();ds,ids,_=make_candidates(h,f,q,.7,.2)
    g=torch.autograd.grad(F.cross_entropy(h(f),y),tuple(h.parameters()))
    for k in range(2):torch.testing.assert_close(sum(d[k] for d in ds),-.2*g[k])
    assert len(ids)==4


def test_analytic_head_jvp_matches_autograd():
    h,f,y,q=fixture();ds,ids,_=make_candidates(h,f,q,.7,.2)
    dw,db=ds[0];p=h(f).softmax(1);dz=F.conv2d(f,dw,db)
    dp=p*(dz-(p*dz).sum(1,keepdim=True))
    fn=lambda w,b:F.conv2d(f,w,b).softmax(1)
    _,true=torch.autograd.functional.jvp(fn,(h.weight,h.bias),(dw,db))
    torch.testing.assert_close(dp,true)


@pytest.mark.parametrize('method',['joint_guard','independent_guard','joint_no_class'])
def test_finite_step_guard(method):
    h,f,y,q=fixture();ds,ids,_=make_candidates(h,f,q,.7,4.)
    report=controlled_head_step(h,f,y,ds,ids,method=method)
    if method != 'independent_guard': assert report['scale']>0
    assert np.all(np.array(report['after'])<=np.array(report['before'])+1e-8)
    assert report['optimizer_steps_in_transaction']==0 and report['ema_updates_in_transaction']==0


def test_rejection_is_exact_and_no_optimizer_state_changes():
    h,f,y,q=fixture();opt=torch.optim.AdamW(h.parameters())
    F.cross_entropy(h(f),y).backward();opt.step()
    before=[p.detach().clone() for p in h.parameters()]
    state={k:v.clone() for k,v in opt.state[h.weight].items() if isinstance(v,torch.Tensor)}
    ds=[(torch.zeros_like(h.weight), torch.tensor([1.,0.,0.,0.]))]
    r=controlled_head_step(h,f,y,ds,['harmful_BG_shift'],method='joint_guard')
    assert r['scale']==0
    for p,b in zip(h.parameters(),before):torch.testing.assert_close(p,b,rtol=0,atol=0)
    for k,v in state.items():torch.testing.assert_close(opt.state[h.weight][k],v,rtol=0,atol=0)
