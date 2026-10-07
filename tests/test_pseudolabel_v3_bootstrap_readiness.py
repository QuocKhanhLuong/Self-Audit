"""Synthetic candidate-support controls; no claim of clinical segmentation quality."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from self_audit_pseudolabel.bootstrap_v3 import (
    BootstrapConfig, BootstrapReadiness, assess_bootstrap_candidates,
)
from self_audit_pseudolabel.evidence import build_region_evidence
from self_audit_pseudolabel.evolution import accepted_region_mask


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)


def fixture(pid='p1',t=1,z=0,kind='ring',seed=1):
    y,x=torch.meshgrid(torch.arange(40),torch.arange(40),indexing='ij')
    r=((y-20)**2+(x-22)**2).float().sqrt()
    labels=torch.zeros(40,40,dtype=torch.long)
    labels[(r>=7)&(r<=10)]=1
    labels[r<7]=2
    labels[(((y-20)**2+(x-8)**2)<=16)&(labels==0)]=3
    if kind=='open': labels[19:22,22:34]=0
    if kind=='disconnected': labels[2:5,2:5]=1
    q=torch.nn.functional.one_hot(labels,4).permute(2,0,1)[None].float()
    image=torch.zeros(1,3,40,40)
    image[:,:,labels==1]=.2
    image[:,:,labels==2]=1.
    image[:,:,labels==3]=.9
    if kind=='constant': image.zero_()
    if kind=='noise': image=torch.randn(image.shape,generator=torch.Generator().manual_seed(seed))
    batch={'cur':image,'prev':image.clone(),'nxt':image.clone(),
           'patient_id':[pid],'t':torch.tensor([t]),'z':torch.tensor([z]),'num_frames':torch.tensor([3]),
           'patient_left_axis':['+x']}
    if kind=='noise':
        batch['prev']=torch.randn(image.shape,generator=torch.Generator().manual_seed(seed+1000))
        batch['nxt']=torch.randn(image.shape,generator=torch.Generator().manual_seed(seed+2000))
    raw,diag=build_region_evidence(q,image[:,1:2],torch.zeros(1,2,40,40),patient_left_axis=['+x'])
    valid=accepted_region_mask(raw.softmax(-1))
    return batch,q,raw,valid,diag


def support(f,config=None):
    return assess_bootstrap_candidates(*f,config=config)


def test_true_image_ring_support_and_sign_invariance():
    f=fixture()
    s=support(f)
    assert s.sample_eligible.tolist()==[True]
    assert s.region_eligible[0,1:3].tolist()==[True,True]
    assert s.diagnostics[0]['trusted_pairs']==1
    b,q,raw,valid,diag=f
    reversed_batch={**b,**{key:-b[key] for key in ('cur','prev','nxt')}}
    reverse=assess_bootstrap_candidates(reversed_batch,q,raw,valid,diag)
    assert torch.equal(s.region_eligible,reverse.region_eligible)


@pytest.mark.parametrize('kind',['constant','open','disconnected'])
def test_negative_controls_cannot_supply_semantic_training(kind):
    s=support(fixture(kind=kind))
    assert not s.sample_eligible.any()
    assert not s.region_eligible.any()  # BG is also blocked
    assert s.masks==[None]


def test_independent_noise_control_over_many_seeds():
    for seed in range(50):
        assert not support(fixture(kind='noise',seed=seed)).sample_eligible.any()


def test_temporal_corrob_requires_true_adjacent_images_and_metadata():
    b,q,raw,valid,diag=fixture(t=0)
    b['nxt'].zero_()  # duplicated prev at t=0 must not count
    assert not assess_bootstrap_candidates(b,q,raw,valid,diag).sample_eligible.any()
    b.pop('num_frames')
    assert not assess_bootstrap_candidates(b,q,raw,valid,diag).sample_eligible.any()


def test_raw_acceptance_and_role_identity_are_required():
    b,q,raw,valid,diag=fixture()
    valid[:,1]=False
    assert not assess_bootstrap_candidates(b,q,raw,valid,diag).sample_eligible.any()
    valid[:,1]=True
    raw[:,1]=torch.tensor([0.,0.,0.,10.])
    assert not assess_bootstrap_candidates(b,q,raw,valid,diag).sample_eligible.any()


def test_channel_permutation_keeps_physical_support():
    b,q,raw,valid,diag=fixture()
    original=assess_bootstrap_candidates(b,q,raw,valid,diag)
    perm=torch.tensor([2,0,3,1])
    changed=q[:,perm]
    ev,ds=build_region_evidence(changed,b['cur'][:,1:2],torch.zeros(1,2,40,40),patient_left_axis=['+x'])
    out=assess_bootstrap_candidates(b,changed,ev,accepted_region_mask(ev.softmax(-1)),ds)
    assert torch.equal(out.region_eligible,original.region_eligible[:,perm])
    assert torch.equal(out.masks[0],original.masks[0])


def _advance(tracker,pid='p1',z=0,kind='ring'):
    f=fixture(pid=pid,z=z,kind=kind)
    return tracker.observe(f[0],support(f,tracker.config))


def test_repeated_one_sample_or_one_patient_never_unlocks():
    tracker=BootstrapReadiness(train_patient_ids=['p1','p2'])
    for _ in range(12): _advance(tracker)
    assert not tracker.ready and tracker.qualified_samples==1
    for z in range(4):
        _advance(tracker,z=z);_advance(tracker,z=z)
    assert not tracker.ready and tracker.qualified_patients==1
    assert not _advance(tracker,pid='p2')
    assert _advance(tracker,pid='p2')


def test_successful_update_floor_is_separate_from_image_support():
    cfg=replace(BootstrapConfig(),min_train_samples=2,min_successful_updates=10)
    tracker=BootstrapReadiness(cfg,['p1','p2'])
    for _ in range(2):
        _advance(tracker,'p1');_advance(tracker,'p2')
    assert tracker.qualified_samples==2 and not tracker.ready
    for _ in range(6): _advance(tracker)
    assert tracker.ready


def test_instability_and_latest_negative_reset_consecutive_support():
    cfg=replace(BootstrapConfig(),min_train_samples=2,min_successful_updates=0)
    tracker=BootstrapReadiness(cfg,['p1','p2'])
    f=fixture(); s=support(f,cfg)
    tracker.observe(f[0],s)
    changed=copy.deepcopy(s)
    changed.masks[0]=torch.roll(changed.masks[0],8,-1)
    tracker.observe(f[0],changed)
    assert tracker.qualified_samples==0
    tracker.observe(f[0],changed)
    assert tracker.qualified_samples==1
    _advance(tracker,kind='constant')
    assert tracker.qualified_samples==0


def test_assessment_does_not_advance_readiness_and_validation_cannot_mutate():
    tracker=BootstrapReadiness(train_patient_ids=['p1','p2'])
    before=tracker.state_dict()
    for _ in range(3): support(fixture())
    assert tracker.state_dict()==before
    f=fixture(pid='validation')
    with pytest.raises(ValueError,match='TRAIN'):
        tracker.observe(f[0],support(f))
    assert tracker.state_dict()==before


def test_checkpoint_round_trip_preserves_latch_and_masks(tmp_path):
    cfg=replace(BootstrapConfig(),min_train_samples=2,min_successful_updates=0,max_history_samples=2)
    tracker=BootstrapReadiness(cfg,['p1','p2'])
    for _ in range(2):
        _advance(tracker,'p1');_advance(tracker,'p2')
    assert tracker.ready
    # Live samples can be evicted after activation; activation proof survives.
    for z in (1,2,3): _advance(tracker,'p1',z)
    path=tmp_path/'readiness.pt';torch.save(tracker.state_dict(),path)
    restored=BootstrapReadiness(cfg,['p1','p2'])
    restored.load_state_dict(torch.load(path,weights_only=True))
    assert restored.ready and restored.successful_updates==tracker.successful_updates
    assert len(restored.state_dict()['history'])==2
    assert restored.activation==tracker.activation
    for a,b in zip(restored.state_dict()['history'],tracker.state_dict()['history']):
        assert a['key']==b['key'] and torch.equal(a['mask'],b['mask'])
    negative=support(fixture(kind='constant'),cfg)
    assert not negative.region_eligible.any()  # readiness never bypasses local gate


@pytest.mark.parametrize('mutation',[
    lambda s:s.update(ready=True),
    lambda s:s.update(successful_updates=-1),
    lambda s:s.update(train_patient_ids=['p1','validation']),
    lambda s:s['config'].update(min_mask_iou=.1),
    lambda s:s['history'][0].update(observations=1000),
    lambda s:s['history'][0].update(mask=torch.zeros(2,2,2)),
])
def test_malformed_checkpoint_is_rejected_atomically(mutation):
    tracker=BootstrapReadiness(train_patient_ids=['p1','p2'])
    _advance(tracker)
    state=tracker.state_dict();mutation(state)
    target=BootstrapReadiness(train_patient_ids=['p1','p2'])
    with pytest.raises(ValueError):target.load_state_dict(state)
    assert target.successful_updates==0 and not target.ready


@pytest.mark.parametrize('kw',[{'min_train_patients':1},{'min_train_samples':1},
    {'min_observations_per_sample':1},{'min_mask_iou':float('nan')},
    {'min_edge_enrichment':1.},{'max_history_samples':2},{'min_temporal_neighbors':0}])
def test_invalid_configuration_fails_closed(kw):
    with pytest.raises(ValueError):BootstrapConfig(**kw)


def test_pre_optimizer_identity_validation_is_pure_and_rejects_wrong_cohort():
    tracker=BootstrapReadiness(train_patient_ids=['p1','p2'])
    good=fixture()[0]
    assert tracker.validate_training_batch(good)==(('p1',1,0),)
    for bad in ({**good,'patient_id':['validation']},{**good,'t':torch.tensor([True])},
                {**good,'z':torch.tensor([-1])},{**good,'patient_id':[]},
                {**good,'t':0}):
        with pytest.raises(ValueError):tracker.validate_training_batch(bad)
    duplicate={key:(value*2 if isinstance(value,list) else torch.cat([value,value]))
               for key,value in good.items()}
    with pytest.raises(ValueError,match='duplicate'):tracker.validate_training_batch(duplicate)
    assert tracker.successful_updates==0 and not tracker.ready and not tracker.state_dict()['history']


def test_identity_anchor_reservoir_survives_full_epoch_and_revisits():
    cfg=replace(BootstrapConfig(),max_history_samples=8,min_successful_updates=0)
    tracker=BootstrapReadiness(cfg,['p1','p2'])
    f=fixture(); s=support(f,cfg)
    candidates=[(pid,z) for z in range(150) for pid in ('p1','p2')]
    for pid,z in candidates:
        batch={**f[0],'patient_id':[pid],'z':torch.tensor([z])}
        tracker.observe(batch,s)
        assert len(tracker.state_dict()['history'])<=8
    anchors=[tuple(row['key']) for row in tracker.state_dict()['history']]
    assert len(anchors)==8 and {key[0] for key in anchors}=={'p1','p2'}
    # Revisit all earlier eligible identities, including hundreds of intervening
    # nonanchors. The deterministic anchors survive and gain a second observation.
    for pid,z in reversed(candidates):
        tracker.observe({**f[0],'patient_id':[pid],'z':torch.tensor([z])},s)
    assert tracker.ready and tracker.qualified_samples==8 and tracker.qualified_patients==2
    assert [tuple(row['key']) for row in tracker.state_dict()['history']]==anchors


def test_anchor_selection_is_permutation_deterministic_and_patient_balanced():
    cfg=replace(BootstrapConfig(),max_history_samples=6,min_successful_updates=0)
    ids=['p1','p2','p3']
    candidates=[(pid,z) for z in range(60) for pid in ids]
    f=fixture();s=support(f,cfg)
    histories=[]
    for order in (candidates,list(reversed(candidates)),
                  [candidates[i] for i in np.random.default_rng(42).permutation(len(candidates))]):
        tracker=BootstrapReadiness(cfg,ids)
        for pid,z in order:
            tracker.observe({**f[0],'patient_id':[pid],'z':torch.tensor([z])},s)
        keys=[tuple(row['key']) for row in tracker.state_dict()['history']]
        assert len(keys)==6 and {key[0] for key in keys}==set(ids)
        histories.append(keys)
    assert histories[0]==histories[1]==histories[2]


def test_anchor_reservoir_with_more_patients_than_budget_is_deterministic():
    cfg=replace(BootstrapConfig(),max_history_samples=4,min_successful_updates=0)
    ids=[f'p{i}' for i in range(20)]
    f=fixture();s=support(f,cfg)
    results=[]
    for order in (ids,ids[::-1]):
        tracker=BootstrapReadiness(cfg,ids)
        for pid in order:
            tracker.observe({**f[0],'patient_id':[pid]},s)
        results.append([row['key'] for row in tracker.state_dict()['history']])
    assert results[0]==results[1] and len(results[0])==4


def test_old_lru_state_and_changed_selection_policy_are_rejected():
    tracker=BootstrapReadiness(train_patient_ids=['p1','p2'])
    for change in ({'version':1},{'selection_policy':'lru'}):
        state=tracker.state_dict();state.update(change)
        with pytest.raises(ValueError):tracker.load_state_dict(state)


def test_uncertain_wall_seam_cannot_certify_a_closed_prototype():
    b,q,raw,valid,diag=fixture()
    labels=q.argmax(1)[0]
    y,x=torch.meshgrid(torch.arange(40),torch.arange(40),indexing='ij')
    seam=(y==20)&(x>=22)&(labels==1)
    assert seam.any()
    uncertain=q*.95+(1-q)*(.05/3)
    uncertain[0,:,seam]=.49/3
    uncertain[0,1,seam]=.51
    assert torch.equal(uncertain.argmax(1),q.argmax(1))
    # Existing hard topology still proposes the ring. Readiness must reject it
    # rather than using pixels that the confidence-aware exporter would withhold.
    ev,ds=build_region_evidence(uncertain,b['cur'][:,1:2],torch.zeros(1,2,40,40),patient_left_axis=['+x'])
    result=assess_bootstrap_candidates(b,uncertain,ev,accepted_region_mask(ev.softmax(-1)),ds)
    assert not result.sample_eligible.any() and not result.region_eligible.any()
    assert result.masks==[None]
    assert result.diagnostics[0]['pairs'][0]['reason']=='unsupported_wall_or_cavity_pixels'


def test_onehot_components_need_original_confidence_and_reject_partial_region():
    b,q,raw,valid,diag=fixture()
    confidence=torch.full_like(q[:,0],.95)
    wall_pixel=(q[0,1]>0).nonzero()[0]
    confidence[0,wall_pixel[0],wall_pixel[1]]=.51
    assert support((b,q,raw,valid,diag)).sample_eligible.any()
    result=assess_bootstrap_candidates(b,q,raw,valid,diag,assignment_confidence=confidence)
    assert not result.sample_eligible.any() and result.masks==[None]
    confidence.fill_(.95)
    result=assess_bootstrap_candidates(b,q,raw,valid,diag,assignment_confidence=confidence)
    assert result.sample_eligible.any()
    assert not (result.masks[0] & (confidence[0]<BootstrapConfig().min_assignment_probability)).any()


def test_uncertain_cavity_pixel_rejects_entire_pair():
    b,q,raw,valid,diag=fixture()
    confidence=torch.ones_like(q[:,0]); confidence[0,20,22]=.69
    result=assess_bootstrap_candidates(b,q,raw,valid,diag,assignment_confidence=confidence)
    assert not result.region_eligible.any() and result.masks==[None]


@pytest.mark.parametrize('confidence',[
    torch.ones(1,1,40,40),torch.ones(1,40,40,dtype=torch.bool),
    torch.full((1,40,40),float('nan')),torch.full((1,40,40),1.01),torch.full((1,40,40),-.1),
])
def test_invalid_assignment_confidence_is_rejected(confidence):
    with pytest.raises(ValueError,match='assignment_confidence'):
        assess_bootstrap_candidates(*fixture(),assignment_confidence=confidence)


@pytest.mark.parametrize('threshold',[.5,.1,1.01,True,float('nan')])
def test_invalid_assignment_confidence_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):BootstrapConfig(min_assignment_probability=threshold)
