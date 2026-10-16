import math
import pytest
import torch
from torch import nn
from layer_upgrade.layers import *
from layer_upgrade.losses import *
from layer_upgrade.guarded import *
from layer_upgrade.refine import *
from layer_upgrade.model import LayerGuidedStudent

torch.set_num_threads(2)

@pytest.mark.parametrize('factor',[1,2,4,8])
def test_projection_identities(factor):
    g=torch.Generator().manual_seed(17)
    x=torch.randn(2,4,16,24,generator=g,dtype=torch.double)
    h=zero_mean_detail(x,factor)
    p=block_lift(block_mean(x,factor),factor)
    assert torch.allclose(block_mean(h,factor),torch.zeros_like(block_mean(h,factor)),atol=1e-14,rtol=0)
    assert torch.allclose(p+h,x,atol=1e-14,rtol=0)
    assert torch.allclose(zero_mean_detail(h,factor),h,atol=1e-14,rtol=0)
    assert abs(float((p*h).sum()))<1e-10

@pytest.mark.parametrize('factor',[2,4])
def test_coarse_guidance_has_no_direct_detail_gradient(factor):
    a=torch.randn(1,4,4,5,dtype=torch.double,requires_grad=True)
    r=torch.randn(1,4,4*factor,5*factor,dtype=torch.double,requires_grad=True)
    q=torch.randn_like(a).softmax(1)
    l=compose_logits(a,r,factor)
    loss=soft_ce(block_mean(l,factor),q,torch.ones_like(q[:,0]))
    ga,gr=torch.autograd.grad(loss,(a,r))
    assert gr.abs().max()<1e-13
    assert ga.norm()>0


def test_softmax_and_pooling_do_not_commute():
    # Protected object is the coarse logit, NOT mean native class probability.
    a=torch.tensor([[[[1.]],[[0.]]]],dtype=torch.double)
    r=torch.tensor([[[[4.,-4.],[4.,-4.]],[[0.,0.],[0.,0.]]]],dtype=torch.double)
    l=compose_logits(a,r,2)
    assert torch.allclose(block_mean(l,2),a)
    assert not torch.allclose(block_mean(l.softmax(1),2),a.softmax(1))


def test_projection_rejects_implicit_padding():
    with pytest.raises(ValueError): block_mean(torch.ones(1,1,7,8),4)

@pytest.mark.parametrize('seed',[0,1,2])
def test_input_initially_equals_center_rgb(seed):
    torch.manual_seed(seed); m=CenterDifferenceInput().double()
    x=torch.rand(2,3,13,17,dtype=torch.double)
    ref=(x[:,1:2].expand(-1,3,-1,-1)-m.mean)/m.std
    assert torch.equal(m(x),ref)
    (m(x).square().mean()).backward()
    assert m.context.weight.grad.norm()>0


def test_equal_slices_context_is_always_zero():
    m=CenterDifferenceInput().double(); nn.init.normal_(m.context.weight)
    c=torch.rand(1,1,8,8,dtype=torch.double); x=c.repeat(1,3,1,1)
    assert torch.equal(m(x),(x-m.mean)/m.std)

@pytest.mark.parametrize('shape',[(1,1,4,4),(1,4,4,4)])
def test_input_rejects_wrong_channels(shape):
    with pytest.raises(ValueError): CenterDifferenceInput()(torch.ones(shape))


def test_q_z_ce_decomposition_and_ce_kl_gradient_equivalence():
    torch.manual_seed(1)
    z=torch.rand(2,4,5,7,dtype=torch.double);z=z/z.sum(1,keepdim=True)
    h=torch.rand(2,1,5,7,dtype=torch.double);u=torch.ones_like(z)/4
    q=((1-h)*u+h*z).requires_grad_()
    logits=torch.randn_like(q,requires_grad=True);w=torch.rand_like(q[:,0],requires_grad=True)
    ce=soft_ce(logits,q,w)
    lp=logits.log_softmax(1)
    expanded=(-(((1-h)*u+h*z)*lp).sum(1)*w.detach()).sum()/w.detach().sum()
    assert torch.allclose(ce,expanded,atol=1e-12)
    entropy=(-(q.detach()*q.detach().log()).sum(1)*w.detach()).sum()/w.detach().sum()
    kl=ce-entropy
    assert torch.equal(torch.autograd.grad(ce,logits,retain_graph=True)[0],torch.autograd.grad(kl,logits,retain_graph=True)[0])
    ce.backward();assert q.grad is None and w.grad is None


def test_ignorance_is_storage_not_target():
    m=torch.zeros(1,4,2,2);h=torch.zeros(1,2,2)
    z,valid=evidence_distribution(m,h)
    logits=torch.randn_like(z,requires_grad=True)
    loss=soft_ce(logits,z,valid.float());loss.backward()
    assert loss==0 and logits.grad.abs().sum()==0


def test_evidence_transport_normalization():
    mass=torch.tensor([[[[0.,1.]],[[0.,2.]],[[0.,0.]],[[0.,1.]]]])
    z,valid=evidence_distribution(mass,mass.sum(1))
    assert not valid[0,0,0] and valid[0,0,1]
    assert torch.allclose(z[0,:,0,1],torch.tensor([.25,.5,0.,.25]))


def test_label_set_limits():
    logits=torch.randn(1,4,3,3,dtype=torch.double,requires_grad=True);w=torch.ones_like(logits[:,0])
    allset=torch.ones_like(logits,dtype=torch.bool)
    loss=set_loss(logits,allset,w)
    assert loss==0
    one=torch.zeros_like(allset);one[:,2]=True
    target=one.double()
    assert torch.allclose(set_loss(logits,one,w),soft_ce(logits,target,w),atol=1e-12)
    with pytest.raises(ValueError):set_loss(logits,torch.zeros_like(one),w)


def test_masked_features_detach_teacher():
    s=torch.randn(1,8,4,4,requires_grad=True);t=torch.randn_like(s,requires_grad=True)
    masked_feature_loss(s,t,torch.ones(1,4,4,dtype=torch.bool)).backward()
    assert t.grad is None and s.grad.norm()>0

@pytest.mark.parametrize('seed',range(8))
def test_gradient_projection_feasible_and_kkt(seed):
    torch.manual_seed(seed)
    g=torch.randn(4,11,dtype=torch.double);d=torch.randn(11,dtype=torch.double)
    got,receipt=project_direction(d,g)
    assert (g@got).min()>-1e-8
    assert ((got-d)**2).sum()<=d.square().sum()+1e-8 # zero is feasible
    assert receipt['status']!='CONSERVATIVE_ZERO_NUMERICAL_FALLBACK'


def test_opposed_guards_can_stall():
    d,receipt=project_direction(torch.tensor([2.],dtype=torch.double),torch.tensor([[1.],[-1.]],dtype=torch.double))
    assert d.abs().max()<1e-12


def test_projected_direction_is_not_safe_after_arbitrary_preconditioning():
    guard=torch.tensor([1.,-1.]);d=torch.tensor([1.,1.])
    assert guard@d==0
    assert guard@(torch.tensor([1.,2.])*d)<0


def test_finite_guarded_step_checks_actual_losses():
    p=nn.Parameter(torch.tensor([0.,0.],dtype=torch.double))
    def guards():return [0.5*(p[0]+1)**2, 0.5*(p[0]-p[1]-1)**2]
    def obj():return guards()[0]
    r=guarded_sgd_step([p],obj,guards,step_size=.2,atol=1e-12)
    assert r['accepted'] and p.norm()>0
    assert all(a<=b+1e-12 for a,b in zip(r['guard_after'],r['guard_before']))
    assert p.grad is None


def test_failed_guarded_step_restores_weights():
    # A zero first derivative does not prevent a finite increase in the guard.
    p=nn.Parameter(torch.tensor([0.],dtype=torch.double))
    def guards():return [(p**2).sum()]
    def obj():return (p-1).square().sum()
    r=guarded_sgd_step([p],obj,guards,step_size=1,max_backtracks=2,atol=0)
    assert not r['accepted'] and p.item()==0


def energy_fixture(seed=0):
    torch.manual_seed(seed)
    image=torch.rand(4,5,dtype=torch.double)
    edges,w=image_grid_graph(image)
    u=torch.randn(20,4,dtype=torch.double)
    a=torch.rand_like(u);a=a/a.sum(-1,keepdim=True)
    return FrozenImageEnergy(u,edges,w,a,torch.rand(20,dtype=torch.double),lam=.4,beta=.6)


def test_energy_gradient_matches_autograd():
    e=energy_fixture();p=torch.randn(20,4,dtype=torch.double).softmax(-1).requires_grad_()
    grad=torch.autograd.grad(e.value(p),p)[0]
    assert torch.allclose(grad,e.gradient(p),atol=1e-12,rtol=0)

@pytest.mark.parametrize('seed',range(5))
def test_refiner_descent_and_simplex(seed):
    e=energy_fixture(seed);logits=torch.randn(20,4,dtype=torch.double,requires_grad=True)
    p,r=refine_distribution(logits,e,steps=8)
    assert not p.requires_grad and logits.grad is None
    assert torch.allclose(p.sum(-1),torch.ones(20,dtype=torch.double),atol=1e-12)
    assert p.min()>0
    for row in r['trace']:
        assert row['energy_after']<=row['energy_before']+1e-10
        assert row['inequality_slack']>=-1e-10


def test_refiner_rejects_unproved_step():
    e=energy_fixture()
    with pytest.raises(ValueError):refine_distribution(torch.zeros(20,4),e,eta=2/e.lipschitz_bound())


def test_zero_evidence_does_not_invent_anatomical_information():
    image=torch.zeros(2,2,dtype=torch.double);edges,w=image_grid_graph(image)
    a=torch.full((4,4),.25,dtype=torch.double)
    e=FrozenImageEnergy(torch.zeros_like(a),edges,w,a,torch.zeros(4),lam=1,beta=0)
    p,r=refine_distribution(torch.zeros_like(a),e,steps=3)
    assert torch.equal(p,a)


class TinyEncoder(nn.Module):
    channels=(8,16,32,64)
    def __init__(self):
        super().__init__();cs=(3,*self.channels)
        self.layers=nn.ModuleList([nn.Conv2d(cs[i],cs[i+1],4 if i==0 else 2,stride=4 if i==0 else 2) for i in range(4)])
    def forward(self,x):
        out=[]
        for layer in self.layers:x=layer(x);out.append(x)
        return out

@pytest.mark.parametrize('hw',[(32,32),(47,53),(64,96)])
def test_complete_decoder_native_shapes_and_exact_coarse_identity(hw):
    m=LayerGuidedStudent(TinyEncoder(),decoder_channels=(16,24,32,40))
    out=m(torch.rand(1,3,*hw))
    assert out['logits'].shape==(1,4,*hw)
    assert torch.allclose(block_mean(out['logits_padded']),out['coarse_logits'],atol=1e-6,rtol=1e-5)
    out['logits'].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

@pytest.mark.parametrize('bound',[.1,.3,1.])
def test_bounded_detail_and_large_margin_argmax_invariance(bound):
    a=torch.tensor([0.,0.,0.,5*bound],dtype=torch.double).view(1,4,1,1).expand(1,4,3,4)
    raw=torch.randn(1,4,12,16,dtype=torch.double)*100
    detail=zero_mean_detail(bound*raw.tanh(),4)
    assert detail.abs().max()<=2*bound+1e-12
    result=compose_logits(a,bound*raw.tanh(),4)
    assert torch.equal(result.argmax(1),torch.full((1,12,16),3))


def test_energy_certificate_does_not_establish_class_correctness():
    # A deliberately wrong, class-specific unary lowers proxy energy while
    # increasing probability of class 1. The externally chosen truth class 0
    # is not available to the operator and is used only for this counterexample.
    edges=torch.empty(2,0,dtype=torch.long)
    unary=torch.tensor([[0.,-5.,0.,0.]],dtype=torch.double)
    e=FrozenImageEnergy(unary,edges,torch.empty(0),torch.full((1,4),.25),torch.zeros(1),lam=0,beta=0)
    p,r=refine_distribution(torch.tensor([[3.,0.,0.,0.]]),e,steps=1)
    assert p.argmax(1).item()==1
    assert r['trace'][0]['energy_after']<r['trace'][0]['energy_before']


def test_positive_image_margin_can_correct_wrong_anchors():
    # Two different true classes linked by a graph edge, both anchors wrong.
    # Truth only evaluates this manufactured mathematical construction.
    p0=torch.tensor([[.05,.8,.1,.05],[.7,.2,.05,.05]],dtype=torch.double)
    unary=torch.tensor([[-12.,0.,0.,0.],[0.,-12.,0.,0.]],dtype=torch.double)
    anchors=torch.tensor([[0.,1.,0.,0.],[1.,0.,0.,0.]],dtype=torch.double)
    e=FrozenImageEnergy(unary,torch.tensor([[0],[1]]),torch.ones(1),anchors,torch.ones(2),lam=.2,beta=.3)
    eta=.5/e.lipschitz_bound()
    q,receipt=refine_distribution(p0.log(),e,steps=1,eta=eta)
    threshold=.7; required=math.log(3*threshold/(1-threshold))
    worst_penalty=2*.2+2*.3
    for i,y in enumerate((0,1)):
        for c in range(4):
            if c==y:continue
            lower=math.log(float(p0[i,y]/p0[i,c]))+eta*(12-worst_penalty)
            assert lower>required
        assert q[i,y]>=threshold
