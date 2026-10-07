"""Experimental bootstrap contracts. Manufactured inputs are not clinical evidence."""
from __future__ import annotations
import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from self_audit_pseudolabel.bootstrap_v3 import BootstrapConfig
from self_audit_pseudolabel.losses_v3 import BootstrapLossConfig,spatial_continuity_loss,region_reconstruction_loss
from self_audit_pseudolabel.system_v3 import CinePseudoTeacher
from self_audit_pseudolabel.trainer_v3 import ProgressiveTeacherTrainer,ProgressiveConfig
from self_audit_pseudolabel.pipeline_v3 import load_config,bootstrap_recipe
from test_pseudolabel_v3_checkup import ring
from test_pseudolabel_v3_regressions import _nifti_fixture

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1);torch.manual_seed(42)


def model(): return CinePseudoTeacher(width=8,appearance_dim=16,motion_dim=8,fused_dim=24,k=4)


def batch(cur):
    n=cur.shape[0]
    return {'prev':cur.clone(),'cur':cur,'nxt':cur.clone(),
            'patient_id':['a','b'][:n],'t':torch.ones(n,dtype=torch.long),
            'z':torch.zeros(n,dtype=torch.long),'num_frames':torch.full((n,),3),
            'patient_left_axis':['+x']*n}


def state_equal(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    return a==b


def test_bootstrap_losses_finite_gradients_permutation_and_degenerate_shapes():
    for shape in ((1,4,1,1),(2,4,5,7)):
        logits=torch.randn(*shape,requires_grad=True);q=logits.softmax(1)
        image=torch.randn(shape[0],1,shape[-2]*4,shape[-1]*4)
        terms=[spatial_continuity_loss(q,image),region_reconstruction_loss(q,image)]
        shuffled=q[:,[2,0,3,1]]
        assert torch.allclose(terms[0],spatial_continuity_loss(shuffled,image))
        assert torch.allclose(terms[1],region_reconstruction_loss(shuffled,image))
        sum(terms).backward()
        assert torch.isfinite(logits.grad).all()
    constant=torch.ones(1,1,8,8)
    q=torch.softmax(torch.randn(1,4,4,4,requires_grad=True),1)
    assert region_reconstruction_loss(q,constant).item()==0
    collapsed=torch.zeros_like(q);collapsed[:,0]=1
    assert torch.isfinite(region_reconstruction_loss(collapsed,torch.randn_like(constant)))


def test_continuity_penalizes_fragmentation_and_preserves_image_edges():
    y,x=torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
    smooth=(x>=4).long();noise=(x+y)%2
    onehot=lambda m:F.one_hot(m,2).permute(2,0,1)[None].float()
    flat=torch.zeros(1,1,8,8);image=smooth[None,None].float()
    assert spatial_continuity_loss(onehot(smooth),flat)<spatial_continuity_loss(onehot(noise),flat)
    assert spatial_continuity_loss(onehot(smooth),image)<spatial_continuity_loss(onehot(smooth),flat)
    assert region_reconstruction_loss(onehot(smooth),image)==0
    assert region_reconstruction_loss(onehot(noise),image)>0


def test_warmup_freezes_semantic_head_and_bank_even_with_adamw_decay():
    net=model();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.9)
    tr=ProgressiveTeacherTrainer(net,opt,bootstrap_config=BootstrapConfig(),train_patient_ids=['a','b'])
    before=copy.deepcopy(net.semantic.state_dict());regions=net.regions.p.detach().clone()
    b=batch(torch.randn(2,3,32,32))
    for _ in range(3):
        loss,accepted,_=tr.train_batch(b,collect_metrics=True)
        assert loss['semantic_seed']==0 and accepted==0
        assert 'spatial_continuity' in loss and 'region_reconstruction' in loss
    assert state_equal(before,net.semantic.state_dict())
    assert not torch.equal(regions,net.regions.p)
    assert not tr.bank.counts.any() and not tr.bank.prototypes.any()
    assert not tr.bootstrap.ready and tr.bootstrap.successful_updates==3
    state=tr.bootstrap.state_dict();output,_,_=tr.infer_batch(b)
    assert not output['valid'].any() and (output['pseudo_label']==255).all()
    assert state_equal(state,tr.bootstrap.state_dict())


def test_held_out_batch_rejected_before_any_optimizer_or_parameter_change():
    net=model();tr=ProgressiveTeacherTrainer(net,torch.optim.AdamW(net.parameters()),
        bootstrap_config=BootstrapConfig(),train_patient_ids=['a','b'])
    b=batch(torch.randn(2,3,32,32));b['patient_id'][1]='validation'
    before=copy.deepcopy(net.state_dict())
    with pytest.raises(ValueError,match='TRAIN'): tr.train_batch(b)
    assert state_equal(before,net.state_dict()) and tr.bootstrap.successful_updates==0
    assert not tr.optimizer.state


def controlled_region_trainer(monkeypatch):
    # Resolver/state-machine fixture only: anonymous regions are supplied, not
    # learned. Actual learning comparisons are reported separately.
    labels=ring();labels[0,:]=3  # a supported anonymous border region, not RV
    q=F.one_hot(labels,4).permute(2,0,1)[None].float()
    net=model()
    monkeypatch.setattr(net.regions,'forward',lambda fused:q.to(fused).expand(fused.shape[0],-1,-1,-1)+net.regions.p.sum()*0)
    image=torch.full((32,32),.1);image[labels==1]=.35;image[labels==2]=1.
    cur=F.interpolate(image[None,None],(128,128),mode='nearest').repeat(2,3,1,1)
    tr=ProgressiveTeacherTrainer(net,torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.5),
        bootstrap_config=BootstrapConfig(min_successful_updates=3,min_train_samples=2),train_patient_ids=['a','b'])
    return net,tr,batch(cur)


def test_only_repeated_train_support_enables_semantics_and_unsupported_batch_still_freezes(monkeypatch):
    net,tr,b=controlled_region_trainer(monkeypatch)
    initial=copy.deepcopy(net.semantic.state_dict())
    for step in range(3):
        loss,accepted,_=tr.train_batch(b,collect_metrics=True)
        assert accepted==0 and loss['semantic_seed']==0
        assert state_equal(initial,net.semantic.state_dict()) and not tr.bank.counts.any()
    assert tr.bootstrap.ready
    loss,accepted,_=tr.train_batch(b)
    assert accepted>0 and loss['semantic_seed']>0 and tr.bank.counts[2:].sum()>0
    assert not state_equal(initial,net.semantic.state_dict())
    state=tr.bootstrap.state_dict();counts=tr.bank.counts.clone()
    eval_batch=dict(b,patient_id=['held_out_a','held_out_b'])
    prediction,_,_=tr.infer_batch(eval_batch)
    assert ((prediction['pseudo_label']>=1)&(prediction['pseudo_label']<=3)).any()
    assert state_equal(state,tr.bootstrap.state_dict()) and torch.equal(counts,tr.bank.counts)
    negative=batch(torch.zeros_like(b['cur']))
    before=copy.deepcopy(net.semantic.state_dict());counts=tr.bank.counts.clone()
    loss,accepted,_=tr.train_batch(negative)
    assert accepted==0 and loss['semantic_seed']==0
    assert state_equal(before,net.semantic.state_dict()) and torch.equal(counts,tr.bank.counts)
    output,_,_=tr.infer_batch(negative)
    assert (output['pseudo_label']==255).all()


def test_prototype_veto_cannot_leave_a_background_only_semantic_update(monkeypatch):
    net,tr,b=controlled_region_trainer(monkeypatch)
    for _ in range(3):tr.train_batch(b)
    assert tr.bootstrap.ready
    def veto(features,scale):
        out=features.new_zeros((*features.shape[:2],4));out[...,0]=100
        return out
    monkeypatch.setattr(tr.bank,'logits',veto)
    before=copy.deepcopy(net.semantic.state_dict())
    loss,accepted,_=tr.train_batch(b)
    assert tr.last_seed_metrics['raw_seed_bg_regions']>0
    assert tr.last_seed_metrics['bootstrap_candidate_lv_regions']>0
    assert accepted==0 and loss['semantic_seed']==0 and not tr.bank.counts.any()
    assert state_equal(before,net.semantic.state_dict())


def test_explicit_disabled_recipe_preserves_legacy_updates():
    cfg=load_config(ROOT/'configs/pseudolabel_v3.json')
    assert bootstrap_recipe(cfg)==(None,None,None)
    off=copy.deepcopy(cfg);off['bootstrap']={'enabled':False}
    assert bootstrap_recipe(off)==(None,None,None)
    a=model();b=copy.deepcopy(a)
    ta=ProgressiveTeacherTrainer(a,torch.optim.AdamW(a.parameters()))
    tb=ProgressiveTeacherTrainer(b,torch.optim.AdamW(b.parameters()),bootstrap_config=None)
    sample=batch(torch.randn(2,3,32,32))
    for _ in range(2):
        assert ta.train_batch(sample)[:2]==tb.train_batch(sample)[:2]
        assert state_equal(a.state_dict(),b.state_dict())


@pytest.mark.parametrize('change',[{'enabled':1},{'enabled':False,'losses':{}},
    {'enabled':True,'readiness':{},'losses':{},'regions':{'mode':'force_masks'}}])
def test_ambiguous_bootstrap_config_rejected(change):
    cfg=load_config(ROOT/'configs/pseudolabel_v3.json');cfg['bootstrap']=change
    with pytest.raises((ValueError,TypeError)):bootstrap_recipe(cfg)


def test_native_warmup_freeze_checkpoint_reload_is_no_gt_and_no_state_update(tmp_path,monkeypatch):
    import nibabel as nib
    from self_audit_pseudolabel.pipeline_v3 import teacher_main,student_main
    from self_audit_pseudolabel.inference_v3 import teacher_export_main
    from self_audit_pseudolabel.freeze import verify_frozen
    data,split=_nifti_fixture(tmp_path)
    load=nib.load
    def guard(path,*args,**kwargs):
        assert '_gt' not in str(path),'training/export read reference labels'
        return load(path,*args,**kwargs)
    monkeypatch.setattr(nib,'load',guard)
    out=tmp_path/'teacher'
    result=teacher_main(['--dataset','acdc','--root',str(data),'--split-manifest',str(split),
        '--config',str(ROOT/'configs/pseudolabel_v3_bootstrap_experimental.json'),
        '--out',str(out),'--epochs','1','--max-train-batches','2','--threads','1','--no-progress'])
    payload=verify_frozen(out/'FROZEN.json',expected_role='teacher_freeze')
    assert result['bootstrap']['stage']=='anonymous_warmup' and result['valid_foreground_pixels']==0
    checkpoint=torch.load(out/'teacher.pt',weights_only=True)
    assert checkpoint['bootstrap_state']['successful_updates']==2 and not checkpoint['bootstrap_state']['ready']
    def forbidden(*args,**kwargs):pytest.fail('inference advanced readiness')
    monkeypatch.setattr('self_audit_pseudolabel.bootstrap_v3.BootstrapReadiness.observe',forbidden)
    exported=tmp_path/'reexport'
    teacher_export_main(['--source-freeze',str(out/'FROZEN.json'),'--root',str(data),
                        '--out',str(exported),'--split','val','--threads','1','--no-progress'])
    verify_frozen(exported/'FROZEN.json',expected_role='teacher_freeze')
    for entry in payload['entries']:
        if entry['split']!='val':continue
        with np.load(out/entry['path']) as a,np.load(exported/entry['path']) as b:
            for key in ('pseudo_label','valid','soft_label'): assert np.array_equal(a[key],b[key])
            assert (b['pseudo_label']==255).all()
    assert state_equal(checkpoint,torch.load(out/'teacher.pt',weights_only=True))
    with pytest.raises(ValueError,match='NO_FOREGROUND_SEEDS'):
        student_main(['--dataset','acdc','--root',str(data),'--manifest',str(out/'FROZEN.json'),
                      '--out',str(tmp_path/'student.pt'),'--threads','1','--no-progress'])
    assert not (tmp_path/'student.pt').exists()
