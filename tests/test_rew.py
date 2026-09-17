"""REW correctness/gradient/runner checks, CPU synthetic unless marked CUDA."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from self_audit.models.read_evaluate_write import (
    ReadEvaluateWriteNet, SharedReadWriter, OutcomeAuditor, Outcome, select_tiles, CC,FIX,REGRESS,WW)
from self_audit.losses.read_evaluate_write import transition_target, outcome_loss, compute_rew_losses, slice_dice
from self_audit.training.rew_runner import REWConfig, _synthetic_records, resolve_data, evaluate, _resources, atomic_json


def network(checkpoint=False, **kwargs):
    torch.manual_seed(73)
    return ReadEvaluateWriteNet(channels=16,encoder=nn.Conv2d(3,16,4,stride=4),fpn=nn.Identity(),
                               checkpoint_reads=checkpoint,**kwargs)


def data():
    g=torch.Generator().manual_seed(87)
    return torch.randn(2,3,32,32,generator=g),torch.randint(4,(2,32,32),generator=g)


def logits(labels):return 4*F.one_hot(labels,4).permute(0,3,1,2).float()


def has_grad(module):return any(p.grad is not None and bool((p.grad != 0).any()) for p in module.parameters())


def test_all_64_transition_semantics():
    combinations=torch.cartesian_prod(torch.arange(4),torch.arange(4),torch.arange(4))
    a,b,y=[combinations[:,i].reshape(1,8,8) for i in range(3)]
    result=transition_target(logits(a),logits(b),y)
    assert torch.bincount(result.flatten(),minlength=4).tolist()==[4,12,12,36]
    assert torch.equal((result==FIX).int()-(result==REGRESS).int(),(b==y).int()-(a==y).int())
    reverse=transition_target(logits(b),logits(a),y)
    assert torch.equal(reverse,torch.tensor([CC,REGRESS,FIX,WW])[result])


def test_same_wrong_different_label_is_WW():
    z=torch.zeros(1,2,2,dtype=torch.long)
    assert (transition_target(logits(z),logits(z+1),z+2)==WW).all()


def test_identity_has_state_gradient_and_zero_quality_change():
    auditor=OutcomeAuditor(16)
    feat=torch.randn(2,16,8,8,requires_grad=True)
    pred=torch.randn(2,4,32,32,requires_grad=True)
    _,target=data();out=auditor.state(feat,pred)
    assert torch.equal(out.delta_q,torch.zeros(2))
    assert torch.equal(out.probabilities[:,[FIX,REGRESS]],torch.zeros_like(out.probabilities[:,[FIX,REGRESS]]))
    term,truth=outcome_loss(out,pred,pred,target);term.backward()
    assert term>0 and has_grad(auditor)
    assert feat.grad is None and pred.grad is None
    assert set(truth.unique().tolist()).issubset({CC,WW})


def fake_outcome(shape,fix,regress):
    b,c,h,w=shape
    p=torch.zeros(b,4,h,w);p[:,FIX]=fix;p[:,REGRESS]=regress;p[:,WW]=1-fix-regress
    return Outcome(p.clamp_min(1e-10).log(),p,torch.ones(b))


@pytest.mark.parametrize('shape',[(2,4,32,32),(1,4,35,21)])
@pytest.mark.parametrize('fix,regress',[(0.,0.),(.2,.1),(.01,.2)])
def test_KEEP_is_bitwise_identity_for_ties_and_bad_reads(shape,fix,regress):
    a=torch.randn(shape);c=[torch.randn(shape) for _ in range(3)]
    o=[fake_outcome(shape,fix,regress) for _ in c]
    result,choice=select_tiles(a,c,o,16,2.,0.)
    assert torch.equal(result,a) and (choice==0).all()


def test_selected_logits_are_exact_candidate_not_average_and_partial_tiles():
    shape=(1,4,35,21);a=torch.randn(shape);cs=[torch.randn(shape) for _ in range(3)]
    os=[fake_outcome(shape,.1,0),fake_outcome(shape,.7,0),fake_outcome(shape,.2,0)]
    result,choice=select_tiles(a,cs,os,16,2.,0.)
    assert torch.equal(result,cs[1]) and (choice==2).all()


def test_tilewise_mosaic_chooses_actual_local_winner():
    a=torch.zeros(1,4,32,32);cs=[a+1,a+2,a+3]
    os=[fake_outcome(a.shape,0,0) for _ in cs]
    os[0].probabilities[:,FIX,:,:16]=.8;os[1].probabilities[:,FIX,:,16:]=.9
    result,choice=select_tiles(a,cs,os,16,2.,0)
    assert (result[:,:,:,:16]==1).all() and (result[:,:,:,16:]==2).all()


@pytest.mark.parametrize('hw',[(8,8),(7,13),(1,5),(5,1)])
def test_read_geometry_bounds_and_cardinality(hw):
    writer=SharedReadWriter(16)
    z=torch.randn(2,16,*hw);e=torch.randn(2,4,*hw)
    grids=writer.coordinates(z,e)
    assert all(g.shape==(2,*hw,8,2) for g in grids)
    for g in grids[1:]:
        probes=g[...,4:,:]
        assert probes[...,0].min()>=-1e-5 and probes[...,0].max()<=hw[1]-1+1e-5
        assert probes[...,1].min()>=-1e-5 and probes[...,1].max()<=hw[0]-1+1e-5
    assert torch.equal(grids[0][...,:4,:],grids[1][...,:4,:])


def test_reject_all_still_teaches_geometry_and_auditor():
    m=network(tau_accept=1e6);x,y=data();out=m(x)
    assert len(out['turns'])==1 and not out['turns'][0]['accepted'].any()
    assert torch.equal(out['logits'],out['initial'])
    loss,parts=compute_rew_losses(m,out,y);loss.backward()
    assert has_grad(m.annotation_expert.geometry) and has_grad(m.annotation_expert.message)
    assert has_grad(m.auditor) and has_grad(m.encoder)
    assert parts['metrics']['proposal_loss']>0 and parts['metrics']['auditor_loss']>0


def test_audit_loss_cannot_teach_actor():
    m=network();x,y=data();out=m(x);_,parts=compute_rew_losses(m,out,y)
    parts['audit_tensor'].backward()
    assert has_grad(m.auditor)
    assert not has_grad(m.encoder) and not has_grad(m.initial_head) and not has_grad(m.annotation_expert)


def test_actor_loss_cannot_teach_auditor():
    m=network();x,y=data();out=m(x);_,parts=compute_rew_losses(m,out,y)
    parts['actor_tensor'].backward()
    assert has_grad(m.annotation_expert) and not has_grad(m.auditor)


def test_GT_changes_targets_not_prediction_or_choice():
    m=network().eval();x,y=data()
    with torch.no_grad():out=m(x)
    before=out['logits'].clone();choices=[t['choice'].clone() for t in out['turns']]
    compute_rew_losses(m,out,y)
    compute_rew_losses(m,out,(y+1)%4)
    assert torch.equal(out['logits'],before)
    for a,b in zip(choices,out['turns']):assert torch.equal(a,b['choice'])
    with pytest.raises(TypeError):m(x,ground_truth=y)


def test_checkpointed_read_matches_output_and_gradients():
    a=network(False,tau_accept=1e6);b=network(True,tau_accept=1e6);b.load_state_dict(a.state_dict())
    x,y=data()
    outputs=[]
    for m in (a,b):
        torch.manual_seed(32);out=m(x);loss,_=compute_rew_losses(m,out,y);loss.backward();outputs.append(out['logits'])
    torch.testing.assert_close(*outputs,atol=1e-6,rtol=1e-6)
    for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert na==nb
        if pa.grad is None:assert pb.grad is None
        else:torch.testing.assert_close(pa.grad,pb.grad,atol=1e-6,rtol=1e-5)


def test_assembled_candidate_is_reaudited_and_gate_can_reject_it(monkeypatch):
    m=network(max_turns=1);x,_=data();calls=[]
    class Controlled(nn.Module):
        def state(self,f,a):return fake_outcome(a.shape,0,0)
        def forward(self,f,a,c):
            calls.append(c.detach().clone());o=fake_outcome(c.shape,.9,0)
            # All three shadows look good, final assembled output is vetoed.
            o.delta_q=torch.full((a.shape[0],),1. if len(calls)<=3 else -1.)
            return o
    m.auditor=Controlled()
    out=m(x)
    assert len(calls)==4
    assert torch.equal(calls[-1],out['turns'][0]['assembled'])
    assert torch.equal(out['logits'],out['initial'])


def test_accept_then_keep_halts_and_feedback_is_real(monkeypatch):
    m=network(max_turns=3);x,_=data();n=0
    class Controlled(nn.Module):
        def state(self,f,a):return fake_outcome(a.shape,0,0)
        def forward(self,f,a,c):
            nonlocal n;n+=1
            o=fake_outcome(c.shape,.9 if n<=4 else 0,0)
            o.delta_q=torch.ones(a.shape[0])
            return o
    def candidate(f,a,feedback,turn):
        if turn==1:assert (feedback[:,3]==1).all()
        changed=a.clone();changed[:,0]=a.amax(1)+5
        return [changed,changed,changed],()
    m.auditor=Controlled();monkeypatch.setattr(m.annotation_expert,'forward',candidate)
    out=m(x)
    assert len(out['turns'])==2
    assert out['turns'][0]['accepted'].all() and not out['turns'][1]['accepted'].any()


@pytest.mark.parametrize('updates',[dict(batch_size=0),dict(num_workers=16),dict(amp='auto'),dict(epochs=True),
                                     dict(proposal_weight=0),dict(harm_weight=float('nan'))])
def test_config_fail_closed(updates):
    with pytest.raises(ValueError):REWConfig(**updates)


def test_invalid_GT_not_clamped():
    x,y=data();m=network();out=m(x)
    with pytest.raises(ValueError):compute_rew_losses(m,out,y+4)


def test_supervised_split_reserves_official_test_and_unknowns(tmp_path):
    _synthetic_records(tmp_path/'training');_synthetic_records(tmp_path/'testing')
    # Give official test cases independent patient names.
    for p in (tmp_path/'testing').rglob('*.npy'):p.rename(p.with_name(p.name.replace('patient','subject')))
    sets,identity=resolve_data(tmp_path,REWConfig(image_size=32,num_workers=0),split_policy='train80val20',manifest=None)
    train={r.patient_id for r in sets['train'].records};val={r.patient_id for r in sets['val'].records}
    test={r['patient_id'] for r in identity['splits']['test']}
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert all('mask_sha256' not in r for r in identity['splits']['test'])


def test_partial_validation_does_not_report_volume_Dice(tmp_path):
    _synthetic_records(tmp_path/'data')
    cfg=REWConfig(image_size=32,batch_size=1,num_workers=0,amp='off')
    ds,_=resolve_data(tmp_path/'data',cfg,split_policy='train80val20',manifest=None)
    result=evaluate(network(),ds['val'],cfg,torch.device('cpu'),max_batches=1)
    assert not result['complete'] and result['final_dice'] is None


def test_full_validation_has_patient_metrics(tmp_path):
    _synthetic_records(tmp_path/'data')
    cfg=REWConfig(image_size=32,batch_size=1,num_workers=0,amp='off')
    ds,_=resolve_data(tmp_path/'data',cfg,split_policy='train80val20',manifest=None)
    result=evaluate(network(),ds['val'],cfg,torch.device('cpu'))
    assert result['complete'] and result['patients']==2
    assert 0<=result['final_dice']<=1


def test_cli_bounded_run_and_occupied_output(tmp_path):
    root=Path(__file__).resolve().parents[1]
    cmd=[sys.executable,'scripts/train_rew.py','--config','configs/self_audit_rew_16gb.yaml',
         '--synthetic','--device','cpu','--output',str(tmp_path/'run'),'--max-batches','3','--max-val-batches','1']
    result=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stderr
    status=json.loads((tmp_path/'run/run_status.json').read_text())
    assert status['status']=='completed_bounded' and status['optimizer_steps']==2
    receipt=tmp_path/'run/run_status.json';original=receipt.read_bytes()
    result=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=45)
    assert result.returncode!=0 and 'occupied output' in result.stderr
    assert receipt.read_bytes()==original
    ck=torch.load(tmp_path/'run/bounded.pt',weights_only=False)
    assert not ck['resumable']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='no CUDA device in this environment')
def test_cuda_real_shape_bf16():
    m=network(True).cuda();x=torch.randn(1,3,256,256,device='cuda');y=torch.randint(4,(1,256,256),device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=m(x);loss,_=compute_rew_losses(m,out,y)
    loss.backward()
    assert torch.isfinite(loss) and has_grad(m.annotation_expert.geometry)


def test_patient_manifest_preserves_old_membership_with_native_frame_names(tmp_path):
    _synthetic_records(tmp_path/'data')
    manifest=tmp_path/'split.json'
    atomic_json(manifest,dict(train_patients=['patient000','patient002','patient004'],
                             val_patients=['patient001','patient003','patient005']))
    ds,identity=resolve_data(tmp_path/'data',REWConfig(image_size=32),split_policy='preserve',manifest=None,patient_manifest=manifest)
    assert {r.patient_id for r in ds['train'].records}=={'patient000','patient002','patient004'}
    assert all('_frame01' in r.case_id for r in ds['val'].records)
    assert identity['patient_manifest_sha256'] is not None
    atomic_json(manifest,dict(train_patients=['patient000','patient001'],val_patients=['patient001']))
    with pytest.raises(ValueError,match='overlapping'):resolve_data(tmp_path/'data',REWConfig(),split_policy='preserve',manifest=None,patient_manifest=manifest)


def test_runner_epoch_resume_matches_uninterrupted(tmp_path,monkeypatch):
    """Real runner/preprocessor, explicit small injected test network; no CUDA claim."""
    from self_audit.training import rew_runner as runner
    import yaml
    _synthetic_records(tmp_path/'data')
    cfg=REWConfig(image_size=32,channels=16,batch_size=2,accumulation_steps=3,num_workers=0,amp='off',
                  cpu_threads=1,epochs=2,warmup_epochs=0,pretrained=False,checkpoint_reads=False)
    config=tmp_path/'cfg.yaml';config.write_text(yaml.safe_dump(cfg.__dict__))
    monkeypatch.setattr(runner,'_model',lambda cfg,synthetic:network(max_turns=1))
    def args(name):return runner.build_parser().parse_args(['--config',str(config),'--data-root',str(tmp_path/'data'),
                       '--split-policy','train80val20','--device','cpu','--output',str(tmp_path/name)])
    complete=runner.run(args('full'))
    assert complete['status']=='completed' and complete['epochs_completed']==2
    atomic=runner._atomic
    class StopAfterEpoch(Exception):pass
    def interrupted(path,write):
        atomic(path,write)
        if path.name=='last.pt':raise StopAfterEpoch()
    monkeypatch.setattr(runner,'_atomic',interrupted)
    with pytest.raises(StopAfterEpoch):runner.run(args('resume'))
    monkeypatch.setattr(runner,'_atomic',atomic)
    resume=args('resume');resume.resume=tmp_path/'resume/last.pt'
    runner.run(resume)
    a=torch.load(tmp_path/'full/last.pt',weights_only=False)
    b=torch.load(tmp_path/'resume/last.pt',weights_only=False)
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    assert a['updates']==b['updates']==4 # accumulation=3, including final short group
    assert len((tmp_path/'resume/epoch_metrics.jsonl').read_text().splitlines())==2


@pytest.mark.parametrize('workers',[0,2])
def test_raw_nifti_and_real_dataloader_workers(tmp_path,workers):
    nib=pytest.importorskip('nibabel')
    import numpy as np
    from self_audit.training.rew_runner import _loader
    root=tmp_path/'raw';root.mkdir()
    for i in range(6):
        folder=root/f'patient{i:03d}';folder.mkdir()
        image=np.random.default_rng(i).normal(size=(32,32,2)).astype(np.float32)
        mask=np.zeros((32,32,2),np.int16);mask[8:24,8:24]=2;mask[12:20,12:20]=3
        nib.save(nib.Nifti1Image(image,np.eye(4)),folder/f'patient{i:03d}_frame01.nii.gz')
        nib.save(nib.Nifti1Image(mask,np.eye(4)),folder/f'patient{i:03d}_frame01_gt.nii.gz')
    cfg=REWConfig(image_size=32,batch_size=2,num_workers=workers,amp='off')
    ds,_=resolve_data(root,cfg,split_policy='train80val20',manifest=None)
    batches=list(_loader(ds['train'],cfg,0,True))
    assert sum(b['image'].shape[0] for b in batches)==len(ds['train'])
    assert all(b['image'].shape[1:]==(3,32,32) for b in batches)
    assert all(b['mask'].dtype==torch.long for b in batches)
