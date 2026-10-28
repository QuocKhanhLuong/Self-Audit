"""Independent integration witnesses; these do not establish patient efficacy."""
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from self_audit.compact_control import build_class_candidates, apply_compact_update


class StatefulStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1)
        self.norm = nn.BatchNorm2d(4)
        self.drop = nn.Dropout(.4)
        self.register_buffer('calls', torch.zeros(()))

    def forward(self, x, valid_mask=None):
        self.calls.add_(1)
        random.random()
        np.random.random()
        return self.drop(self.norm(self.encoder(x)))


def test_candidate_generation_preserves_post_supervised_execution_state():
    torch.manual_seed(31)
    model = StatefulStudent().train()
    model.norm.eval()  # Mixed modes must not be collapsed by model.train(flag).
    x = torch.rand(2, 3, 3, 5)
    q = torch.rand(2, 4, 3, 5).softmax(1)
    valid = torch.ones(2, 3, 5, dtype=torch.bool)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    state = copy.deepcopy(model.state_dict())
    grads = {n:p.grad.clone() for n,p in model.named_parameters()}
    modes = {n:m.training for n,m in model.named_modules()}
    trng, prng, nrng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    result = build_class_candidates(model, x, q, valid, lr=.1)
    assert result['directions']
    assert all(torch.equal(state[n],t) for n,t in model.state_dict().items())
    assert all(torch.equal(grads[n],p.grad) for n,p in model.named_parameters())
    assert modes == {n:m.training for n,m in model.named_modules()}
    assert torch.equal(trng,torch.get_rng_state()) and prng==random.getstate()
    now=np.random.get_state()
    assert nrng[0]==now[0] and np.array_equal(nrng[1],now[1]) and nrng[2:]==now[2:]


def test_active_norm_limit_preserves_common_scaled_plain_displacement():
    torch.manual_seed(32)
    model=nn.Conv2d(3,4,1)
    x=torch.rand(2,3,3,4)
    labels=torch.arange(24).reshape(2,3,4)%4
    q=torch.nn.functional.one_hot(labels,4).permute(0,3,1,2).float()
    valid=torch.ones_like(labels,dtype=torch.bool)
    raw=build_class_candidates(model,x,q,valid,lr=1.)
    plain={n:sum(d[n] for d in raw['directions']) for n,_ in model.named_parameters()}
    norm=sum(v.double().square().sum() for v in plain.values()).sqrt().item()
    cap=norm/20
    clipped=build_class_candidates(model,x,q,valid,lr=1.,max_norm=cap)
    for n,_ in model.named_parameters():
        actual=sum(d[n] for d in clipped['directions'])
        torch.testing.assert_close(actual,plain[n]*(cap/norm),rtol=2e-5,atol=1e-7)


class PixelStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=nn.Parameter(torch.tensor([[[[0.,2.]],[[2.,0.]],[[0.,0.]],[[0.,0.]]]]))

    def forward(self,x,valid_mask=None):
        return self.logits.expand(x.shape[0],-1,-1,-1)


def audit_factory(labels=None):
    y=torch.tensor([[[1,0]]]) if labels is None else labels
    def batches():
        yield {'x':torch.zeros(1,3,1,2),'y':y,'valid':y!=255,
               'patient':'train_L','frame':0,'slices':[0]}
    return batches


def directions(model):
    d=torch.zeros_like(model.logits)
    d[:,0,0,0]=5.
    return {'directions':[{'logits':d}], 'source_ids':['pseudo_class_0'], 'denominator':2}


@pytest.mark.parametrize('key,value', [('eps',float('nan')),('absent_fp_budget',float('nan')),
                                      ('tolerance',float('nan')),('max_trials',1.5),
                                      ('typo_guard_option',0)])
def test_guard_rejects_invalid_policy_before_any_commit(key,value):
    model=PixelStudent()
    state=copy.deepcopy(model.state_dict())
    with pytest.raises((ValueError,TypeError)):
        apply_compact_update(model,directions(model),method='scalar_guard',
                             audit_batches=audit_factory(),config={key:value})
    assert all(torch.equal(state[n],p) for n,p in model.state_dict().items())


def test_no_audit_voxel_cannot_vacuously_accept_u():
    model=PixelStudent()
    before=model.logits.detach().clone()
    with pytest.raises(ValueError):
        apply_compact_update(model,directions(model),method='joint_guard',
                            audit_batches=audit_factory(torch.full((1,1,2),255)))
    assert torch.equal(before,model.logits)


def test_audit_cannot_change_between_reference_and_trial():
    model=PixelStudent()
    before=model.logits.detach().clone()
    calls=[]
    def changing():
        calls.append(1)
        y=torch.tensor([[[1,0]]]) if len(calls)==1 else torch.tensor([[[0,0]]])
        yield from audit_factory(y)()
    with pytest.raises((ValueError,RuntimeError)):
        apply_compact_update(model,directions(model),method='joint_guard',audit_batches=changing)
    assert torch.equal(before,model.logits)


def test_scalar_custom_proposal_cannot_be_a_vector_policy():
    model=PixelStudent()
    d=directions(model)
    d['directions'].append({k:v*.5 for k,v in d['directions'][0].items()})
    d['source_ids'].append('pseudo_class_1')
    with pytest.raises(ValueError):
        apply_compact_update(model,d,method='scalar_guard',audit_batches=audit_factory(),
                             config={'trial_alphas':[[0.,1.]]})


def test_duplicate_candidate_source_not_an_extra_degree_of_freedom():
    model=PixelStudent()
    d=directions(model)
    d['directions']*=2
    d['source_ids']*=2
    with pytest.raises(ValueError):
        apply_compact_update(model,d,method='joint_guard',audit_batches=audit_factory())


def test_zero_direction_is_not_reject_all():
    model=PixelStudent()
    d=directions(model)
    d['directions'][0]['logits'].zero_()
    receipt=apply_compact_update(model,d,method='scalar_guard',audit_batches=audit_factory())
    assert receipt['status']=='zero_displacement'
    assert not receipt['accepted']


def test_explicit_zero_coefficient_is_zero_displacement():
    model=PixelStudent()
    before=model.logits.detach().clone()
    receipt=apply_compact_update(model,directions(model),method='scalar_guard',
                                audit_batches=audit_factory(),config={'trial_alphas':[[0.]]})
    assert receipt['status']=='zero_displacement'
    assert not receipt['accepted']
    assert torch.equal(before,model.logits)


@pytest.mark.parametrize('change',['image','late_label','late_valid'])
def test_complete_audit_stream_is_pinned_not_a_prefix(change):
    model=nn.Conv2d(3,4,1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([2.,0.,0.,0.]))
    candidates={'directions':[{n:torch.ones_like(p)*.01 for n,p in model.named_parameters()}],
                'source_ids':['pseudo_class_0'],'denominator':1000}
    calls=[]
    def changing():
        calls.append(1)
        x=torch.zeros(1,3,1,1000)
        y=torch.zeros(1,1,1000,dtype=torch.long)
        valid=torch.ones_like(y,dtype=torch.bool)
        if len(calls)>1:
            if change=='image': x[0,0,0,-1]=.25
            elif change=='late_label': y[0,0,-1]=1
            else: valid[0,0,-1]=False
        yield {'x':x,'y':y,'valid':valid,'patient':'train_L','frame':0,'slice':0}
    before=copy.deepcopy(model.state_dict())
    with pytest.raises((ValueError,RuntimeError)):
        apply_compact_update(model,candidates,method='joint_guard',audit_batches=changing)
    assert all(torch.equal(before[n],v) for n,v in model.state_dict().items())
