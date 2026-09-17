"""Scientific contracts for the isolated event-audit diagnostic baseline."""
import inspect
import numpy as np
import pytest
import torch
from self_audit_event.annotation import AnnotationExpert, AnnotationState
from self_audit_event.reference_free_auditor import AuditObservation, ReferenceFreeAuditor, rf_loss
from self_audit_event.audit_trigger import AuditTrigger, trigger_loss
from self_audit_event.model import EventAuditModel
from self_audit_event.replay import AuditReplay
from self_audit_event.losses import segmentation_loss
from self_audit_event.evaluation import class_dice, oracle_delta, binary_ranking, transition_metrics, convergence_metrics
from self_audit_event.training import EventTrainer, TrainConfig, initial_state, evaluate
from self_audit_event.data import PhantomDataset


@pytest.fixture(autouse=True)
def deterministic_fixture():
    torch.set_num_threads(2)
    torch.manual_seed(12)


def fixture(batch=4):
    data=PhantomDataset(batch,32,5)
    return torch.stack([r['image'] for r in data]),torch.stack([r['mask'] for r in data])


def test_no_gt_forward_or_rf_loss_arguments():
    forbidden={'gt','target','mask','error_map','correctness','labels'}
    for method in (ReferenceFreeAuditor.forward, ReferenceFreeAuditor.quality,
                   ReferenceFreeAuditor.intrinsic_risk, AuditTrigger.forward,
                   EventAuditModel.forward, EventAuditModel.twins, rf_loss, trigger_loss):
        assert not (set(inspect.signature(method).parameters)&forbidden)
        assert all(p.kind != p.VAR_KEYWORD for p in inspect.signature(method).parameters.values())


def test_observation_rejects_raw_target_and_detaches_clones():
    x,y=fixture();actor=AnnotationExpert();state=actor.start(x)
    with pytest.raises((TypeError,ValueError,AttributeError)):
        AuditObservation.from_state(x,y)
    bad=AnnotationState(state.features,y,state.coordinates)
    with pytest.raises((TypeError,ValueError)):
        AuditObservation.from_state(x,bad)
    obs=AuditObservation.from_state(x,state)
    assert not obs.logits.requires_grad and obs.logits.data_ptr()!=state.logits.data_ptr()
    assert not obs.image.requires_grad and obs.image.data_ptr()!=x.data_ptr()


def test_gt_permutation_cannot_change_rf_targets_or_trigger():
    x,y=fixture();model=EventAuditModel();state=model.actor.start(x)
    obs=AuditObservation.from_state(x,state)
    before=(model.auditor.intrinsic_risk(obs),rf_loss(model.auditor,obs),model.trigger(model.trigger.state_features(state)))
    # GT remains in actor loss/evaluation only; changing it cannot alter audit observations.
    segmentation_loss(state.logits,(y+1)%4)
    after=(model.auditor.intrinsic_risk(obs),rf_loss(model.auditor,obs),model.trigger(model.trigger.state_features(state)))
    for a,b in zip(before,after): assert torch.equal(a,b)


def test_both_gradient_firewalls_and_frozen_anchor():
    x,y=fixture();model=EventAuditModel();anchor={k:v.clone() for k,v in model.auditor.anchor.state_dict().items()}
    out=model(x,policy='always');segmentation_loss(out.final_logits,y).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.actor.parameters())
    assert all(p.grad is None for p in model.auditor.parameters())
    assert all(p.grad is None for p in model.trigger.parameters())
    model.zero_grad(set_to_none=True)
    obs=AuditObservation.from_state(x,initial_state(out));rf_loss(model.auditor,obs).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.auditor.parameters())
    assert all(p.grad is None for p in model.actor.parameters())
    assert all(not p.requires_grad for p in model.auditor.anchor.parameters())
    for k,v in anchor.items(): assert torch.equal(v,model.auditor.anchor.state_dict()[k])


def test_skip_exact_and_does_not_invoke_auditor():
    x,_=fixture();model=EventAuditModel()
    called=[];handle=model.auditor.register_forward_hook(lambda *a:called.append(1))
    out=model(x,policy='none');handle.remove()
    assert torch.equal(out.final_logits,initial_state(out).logits)
    assert out.audit_calls==0 and out.read_calls==len(x) and not called
    assert not out.decisions.any()


def test_always_and_learned_controlled_actions_budget():
    x,_=fixture();model=EventAuditModel()
    out=model(x,policy='always');assert out.decisions.all() and out.audit_calls==len(x)
    selected=AuditTrigger.select(torch.tensor([-.1,.2,.3,-.4]),budget_fraction=.5)
    assert selected.tolist()==[False,True,True,False]
    assert not AuditTrigger.select(torch.ones(4)*-.1,.5).any()
    assert int(AuditTrigger.select(torch.ones(4),.5).sum())==2
    assert not AuditTrigger.select(torch.ones(1),.5).any()  # explicit floor quota contract
    assert AuditTrigger.select(torch.ones(4),.5).tolist()==[True,True,False,False]
    trigger=AuditTrigger()
    with torch.no_grad():
        for p in trigger.parameters(): p.zero_()
        trigger.value[0].weight[0,0]=1
        trigger.value[2].weight[0,0]=1
        trigger.value[2].bias[0]=-.2
    features=torch.zeros(4,trigger.input_dim);features[:,0]=torch.tensor([-1.,1.,2.,-2.])
    assert trigger.select(trigger(features),.5).tolist()==[False,True,True,False]


def test_guidance_support_output_causal_path_and_gradients():
    x,y=fixture();actor=AnnotationExpert();state=actor.start(x)
    g0=torch.zeros_like(state.logits[:,:1]);g1=torch.ones_like(g0,requires_grad=True)
    zero=actor.refine(state,g0);guided=actor.refine(state,g1)
    assert not torch.allclose(zero.coordinates,guided.coordinates,atol=1e-7,rtol=0)
    assert not torch.allclose(zero.logits,guided.logits,atol=1e-7,rtol=0)
    fixed=actor.refine(state,g1,coordinate_override=zero.coordinates)
    assert torch.allclose(fixed.logits,zero.logits,atol=1e-6,rtol=1e-6)
    segmentation_loss(guided.logits,y).backward()
    assert g1.grad is None  # detached guidance; controller parameters still learn
    grads=[p.grad for n,p in actor.named_parameters() if 'generator' in n and p.grad is not None]
    assert grads and sum(float(g.abs().sum()) for g in grads)>0


def test_twins_identical_start_no_mutation_or_cross_sample_history():
    x,_=fixture();model=EventAuditModel();state=model.actor.start(x)
    saved=[v.detach().clone() for v in (state.features,state.logits,state.coordinates)]
    twin=model.twins(x,state=state)
    assert torch.equal(twin['skip_logits'],state.logits)
    assert not twin['audit_logits'].requires_grad and not twin['intrinsic_delta'].requires_grad
    for v,s in zip((state.features,state.logits,state.coordinates),saved): assert torch.equal(v,s)
    single=model(x[:1],policy='always').final_logits
    model(x[1:],policy='always')
    assert torch.allclose(single,model(x[:1],policy='always').final_logits,atol=1e-6)
    assert torch.allclose(single,model(x,policy='always').final_logits[:1],atol=2e-5)


def test_replay_seed_cloning_expiration_and_no_gt_fields():
    x,_=fixture();m=EventAuditModel();s=m.actor.start(x)
    o=AuditObservation.from_state(x,s);features=m.trigger.state_features(s);values=torch.arange(4.).requires_grad_()
    a,b=AuditReplay(seed=2),AuditReplay(seed=2)
    for replay in (a,b): replay.add(o,features,values,step=3)
    first=a.sample(3,current_step=4);second=b.sample(3,current_step=4)
    assert set(first)=={'observation','features','values','steps'}
    assert torch.equal(first['values'],second['values']) and not first['values'].requires_grad
    assert a.sample(4,current_step=50,max_age=8) is None
    with torch.no_grad(): o.logits.zero_();values.add_(100)
    assert (b.sample(4,current_step=4)['values']<100).all()


def test_oracle_detached_and_class_permutation_counterexample():
    x,y=fixture();logits=torch.nn.functional.one_hot(y,4).permute(0,3,1,2).float()*20-10
    logits.requires_grad_();permuted=logits[:,[0,2,3,1]]
    delta=oracle_delta(logits,permuted,y)
    assert not delta.requires_grad and (delta < -.8).all()
    auditor=ReferenceFreeAuditor()
    # Evaluation-only oracle construction, never RF training data.
    a,b=AuditObservation(x,logits.detach()),AuditObservation(x,permuted.detach())
    assert torch.allclose(auditor.quality(a),auditor.quality(b),atol=1e-6)


def test_trigger_loss_is_not_actor_loss_and_metadata_hidden():
    x,_=fixture();m=EventAuditModel();s=m.actor.start(x)
    values=torch.tensor([-.1,.2,-.3,.4],requires_grad=True)
    loss=trigger_loss(m.trigger,m.trigger.state_features(s),values);loss.backward()
    assert values.grad is None and all(p.grad is None for p in m.actor.parameters())
    assert all(p.grad is None for p in m.auditor.parameters())
    with pytest.raises(TypeError): AuditObservation(x,s.logits,corruption_id=1)
    # Hiding metadata does not rule out visible corruption artifacts; no such claim.


def test_slow_updates_separate_from_invocation_and_never_explores():
    x,y=fixture();t=EventTrainer(TrainConfig(batch_size=4,slow_every=2,exploration_fraction=1))
    t.step(x,y,policy='learned',step=0)
    assert t.counters['rf_updates']==0 and t.counters['probe_audit_calls']==4
    t.step(x,y,policy='learned',step=1)
    assert t.counters['actor_updates']==2 and t.counters['rf_updates']==1 and t.counters['trigger_updates']==1
    b=EventTrainer(TrainConfig(batch_size=4,exploration_fraction=1))
    b.step(x,y,policy='none',step=0)
    assert b.counters['probe_audit_calls']==0 and b.counters['rf_updates']==0


def test_metric_ties_empty_and_censoring():
    assert binary_ranking([0,1],[1,1])['auroc']==.5
    assert binary_ranking([0,1],[1,1])['auprc']==.5
    assert binary_ranking([0,1],[0,1])['auroc']==1
    assert transition_metrics([],[],[])['n']==0
    c=convergence_metrics([dict(updates=1,examples_seen=8,wall_seconds=1,final_dice=.1),
                           dict(updates=2,examples_seen=16,wall_seconds=3,final_dice=.3)])
    assert c['thresholds']['0.5']['right_censored']
    assert np.isclose(c['area_under_curve']['updates']['normalized'],.2)


def test_evaluation_uses_complete_cases_and_patient_units():
    data=PhantomDataset(4,32,6);model=EventAuditModel()
    result=evaluate(model,data,TrainConfig(batch_size=4),policy='none')
    assert result['patient_count']==4 and result['volume_count']==4
    assert result['a0_dice']==result['final_dice'] and result['harm_patient_rate']==0
    assert len(result['patients'])==4


def test_periodic_evaluation_advances_eligible_events():
    data=PhantomDataset(8,32,6);model=EventAuditModel()
    result=evaluate(model,data,TrainConfig(batch_size=4,period=2),policy='periodic',include_twins=False)
    assert result['costs']['policy_audit_calls']==4
    assert result['audit_fraction']==.5


def test_trainer_gt_permutation_leaves_preupdate_rf_experience_unchanged():
    x,y=fixture()
    cfg=TrainConfig(batch_size=4,slow_every=1,exploration_fraction=1,seed=7)
    a,b=EventTrainer(cfg),EventTrainer(cfg)
    a.step(x,y,policy='learned',step=0)
    b.step(x,(y+1)%4,policy='learned',step=0)
    ra,rb=a.replay.sample(4,0),b.replay.sample(4,0)
    assert torch.equal(ra['values'],rb['values'])
    assert torch.equal(ra['features'],rb['features'])
    assert torch.equal(ra['observation'].logits,rb['observation'].logits)
    for p,q in zip(a.model.auditor.parameters(),b.model.auditor.parameters()): assert torch.equal(p,q)
    for p,q in zip(a.model.trigger.parameters(),b.model.trigger.parameters()): assert torch.equal(p,q)
    assert any(not torch.equal(p,q) for p,q in zip(a.model.actor.parameters(),b.model.actor.parameters()))
