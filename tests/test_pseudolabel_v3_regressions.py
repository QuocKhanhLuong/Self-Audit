"""Regression and synthetic integration tests, not evidence of real cardiac accuracy."""
import importlib.util,json,math,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from self_audit_pseudolabel.system_v3 import CinePseudoTeacher,pseudo_supervision_loss
from self_audit_pseudolabel.losses_v3 import teacher_loss,registration_loss
from self_audit_pseudolabel.trainer_v3 import ProgressiveTeacherTrainer
from self_audit_pseudolabel.data_v3 import read_patient_splits,PatientBatchSampler,is_reference_path
from self_audit_pseudolabel.freeze import export_pseudo_npz,write_freeze_manifest,verify_frozen
from self_audit_pseudolabel.consistency import consistency_gate

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1); torch.manual_seed(42)

def teacher():
    return CinePseudoTeacher(width=8,appearance_dim=16,motion_dim=8,fused_dim=24,k=4)
def ctx(n=16): return torch.rand(1,3,n,n)

def test_unknown_255_is_excluded_before_ce_and_has_zero_gradient():
    x=torch.zeros(1,4,4,4,requires_grad=True); y=torch.full((1,4,4),255,dtype=torch.long); v=torch.zeros_like(y,dtype=torch.bool)
    y[:,1,1]=3; v[:,1,1]=True
    loss=pseudo_supervision_loss({'a0_logits':x,'final_logits':x},y,v); loss.backward()
    assert torch.isfinite(loss) and loss.item()>0
    assert torch.count_nonzero(x.grad.permute(0,2,3,1)[~v])==0

def test_all_unknown_has_zero_loss_and_zero_gradient():
    x=torch.randn(1,4,3,3,requires_grad=True); y=torch.full((1,3,3),255,dtype=torch.long)
    loss=pseudo_supervision_loss({'a0_logits':x,'final_logits':x},y,torch.zeros_like(y,dtype=torch.bool));loss.backward()
    assert loss.item()==0 and x.grad.abs().sum()==0

def test_unknown_cannot_be_marked_valid():
    x=torch.zeros(1,4,2,2)
    with pytest.raises(ValueError,match='semantic class'):
        pseudo_supervision_loss({'a0_logits':x,'final_logits':x},torch.full((1,2,2),255),torch.ones(1,2,2,dtype=torch.bool))

def test_no_evidence_forces_abstention_even_with_confident_head():
    model=teacher(); x=ctx()
    with torch.no_grad():
        model.semantic[-1].weight.zero_();model.semantic[-1].bias.copy_(torch.tensor([0.,0.,0.,20.]))
        out=model(x,x,x)
    assert not out['valid'].any() and torch.all(out['pseudo_label']==255)

def test_target_evidence_is_not_added_to_logits_being_supervised():
    model=teacher();x=ctx()
    with torch.no_grad():model.semantic[-1].weight.zero_();model.semantic[-1].bias.zero_()
    evidence=torch.zeros(1,4,4);evidence[:,:,3]=20
    out=model(x,x,x,evidence_logits=evidence)
    losses=teacher_loss(out,x,evidence,torch.ones(1,4,dtype=torch.bool))
    assert losses['semantic_seed'].item()==pytest.approx(math.log(4),abs=1e-6)
    losses['semantic_seed'].backward()
    assert model.semantic[-1].bias.grad.abs().sum()>0

def test_conflicting_regions_cannot_make_a_confident_dense_label():
    model=teacher();base={'semantic_logits':torch.zeros(1,2,4),'region_prob':torch.full((1,2,2,2),.5)}
    ev=torch.tensor([[[0.,20.,0.,0.],[0.,0.,0.,20.]]])
    out=model.decode_evidence(base,(8,8),evidence_logits=ev)
    assert out['region_valid'].all() and not out['valid'].any()

def test_teacher_train_and_infer_encode_once_each():
    model=teacher();trainer=ProgressiveTeacherTrainer(model,torch.optim.AdamW(model.parameters(),lr=1e-3))
    x=ctx();batch={'prev':x,'cur':x,'nxt':x,'patient_left_axis':['-x']};calls=[]
    hook=model.appearance.register_forward_hook(lambda *args:calls.append(1))
    terms,_,_=trainer.train_batch(batch)
    assert len(calls)==1 and all(math.isfinite(v) for v in terms.values())
    counts=trainer.bank.counts.clone();trainer.infer_batch(batch)
    hook.remove();assert len(calls)==2 and torch.equal(counts,trainer.bank.counts)

def test_registration_singleton_smoothness_is_finite():
    x=ctx(4);out=teacher()(x,x,x)
    photo,smooth=registration_loss(out,x)
    assert torch.isfinite(photo+smooth)

def test_reference_filenames_are_blocked_before_header_read():
    for name in ['p_gt.nii.gz','p_gt_4d.nii.gz','p_seg.nii','scribbles/a.nii','masks/a.nii']:
        assert is_reference_path(name)
    assert not is_reference_path('patient001/patient001_4d.nii.gz')

def test_split_leakage_rejected(tmp_path):
    path=tmp_path/'split.json';path.write_text(json.dumps({'train_patients':['patient001'],'val_patients':['patient001']}))
    with pytest.raises(ValueError,match='leakage'):read_patient_splits(path)

def test_existing_patient_split_schema_and_case_aliases(tmp_path):
    path=tmp_path/'split.json';path.write_text(json.dumps({'train':['patient001_ED','patient001_ES'],'val':['patient002_ED']}))
    splits=read_patient_splits(path);assert splits['train']==['patient001'] and splits['val']==['patient002']

def test_sampler_never_mixes_patient_shapes():
    keys=['a']*5+['b']*3;sampler=PatientBatchSampler(keys,2)
    batches=list(sampler);assert sorted(i for b in batches for i in b)==list(range(8))
    assert all(len({keys[i] for i in b})==1 for b in batches)

def _freeze(tmp_path):
    root=tmp_path/'freeze';root.mkdir();label=np.zeros((3,4),np.uint8);valid=np.ones_like(label)
    soft=np.eye(4,dtype=np.float32)[label].transpose(2,0,1)
    sha=export_pseudo_npz(root/'one.npz',pseudo_label=label,valid=valid,soft_label=soft,
                         metadata={'patient_id':'p','t':0,'z':0})
    e={'path':'one.npz','sha256':sha,'patient_id':'p','t':0,'z':0}
    write_freeze_manifest(root,[e],{})
    return root

def test_freeze_detects_changed_payload(tmp_path):
    root=_freeze(tmp_path);verify_frozen(root/'FROZEN.json')
    with (root/'one.npz').open('ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError,match='changed'):verify_frozen(root/'FROZEN.json')

def test_freeze_detects_manifest_metadata_change(tmp_path):
    root=_freeze(tmp_path);path=root/'FROZEN.json';p=json.loads(path.read_text());p['entries'][0]['t']=100;path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='digest'):verify_frozen(path)

def test_freeze_is_write_once(tmp_path):
    root=_freeze(tmp_path)
    with pytest.raises(FileExistsError):write_freeze_manifest(root,[],{})

def test_reject_invalid_probabilities_at_export(tmp_path):
    with pytest.raises(ValueError,match='probabilities'):
        export_pseudo_npz(tmp_path/'x.npz',pseudo_label=np.zeros((2,2)),valid=np.ones((2,2)),soft_label=np.zeros((4,2,2)),metadata={})

def test_evaluator_import_does_not_import_torch():
    code="import sys;from self_audit_pseudolabel.freeze import verify_frozen;assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)

def test_consistency_never_promotes_invalid_pixels():
    p=torch.softmax(torch.randn(3,2,4,5,5),2);v=torch.zeros(3,2,5,5,dtype=torch.bool)
    q,u=consistency_gate(p,v);assert torch.equal(q,p) and not u.any()

def test_consistency_uses_motion_correspondence():
    p=torch.zeros(2,1,4,4,6);p[:,:,0]=1
    p[0,0,0,:,1]=0;p[0,0,3,:,1]=1;p[1,0,0,:,2]=0;p[1,0,3,:,2]=1
    v=torch.ones(2,1,4,6,dtype=torch.bool)
    _,unaligned=consistency_gate(p,v,temporal_weight=2,min_agreement=.6)
    fp=torch.zeros(2,1,2,4,6);fn=fp.clone();fp[1,:,0]=-2/5;fn[0,:,0]=2/5
    _,aligned=consistency_gate(p,v,temporal_weight=2,min_agreement=.6,flow_prev=fp,flow_next=fn)
    assert not unaligned[1,0,:,2].any() and aligned[1,0,:,2].all()

def test_consistency_has_no_implicit_cycle_wrap():
    p=torch.zeros(3,1,4,2,2);p[0,:,1]=1;p[1,:,1]=1;p[2,:,2]=1
    _,v=consistency_gate(p,torch.ones(3,1,2,2,dtype=torch.bool),temporal_weight=2,min_agreement=.8)
    assert v[0].all()  # last frame must NOT influence first frame

def _nifti_fixture(tmp_path):
    nib=pytest.importorskip('nibabel');data=tmp_path/'data';data.mkdir()
    rng=np.random.default_rng(17)
    for pid,h in [('patient001',16),('patient002',20)]:
        p=data/pid;p.mkdir();array=rng.random((h,16,2,3),dtype=np.float32)
        nib.save(nib.Nifti1Image(array,np.diag([1.,1.,5.,1.])),p/f'{pid}_4d.nii.gz')
        (p/'Info.cfg').write_text('ED: 1\nES: 3\n')
        for frame in (1,3):
            truth=np.zeros((h,16,2),np.uint8);truth[3:10,3:10,:]=3
            nib.save(nib.Nifti1Image(truth,np.diag([1.,1.,5.,1.])),p/f'{pid}_frame{frame:02d}_gt.nii.gz')
    split=tmp_path/'split.json';split.write_text(json.dumps({'train_patients':['patient001'],'val_patients':['patient002']}))
    return data,split

def test_native_slice_axis_not_changed_by_canonicalization(tmp_path):
    nib=pytest.importorskip('nibabel')
    from self_audit_pseudolabel.data_v3 import CineRecord,CineSliceDataset
    a=np.arange(4*5*2*3,dtype=np.float32).reshape(4,5,2,3)
    affine=np.array([[0,0,5,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]],float)
    path=tmp_path/'study.nii.gz';nib.save(nib.Nifti1Image(a,affine),path)
    ds=CineSliceDataset([CineRecord('acdc','p',path)])
    assert len(ds)==6 and ds[0]['cur'].shape==(3,5,4) and ds[0]['patient_left_axis']==''

def test_mnms_4d_reference_excluded_before_nib_load(tmp_path,monkeypatch):
    nib=pytest.importorskip('nibabel')
    from self_audit_pseudolabel.data_v3 import discover_mnms_full_cine
    image=np.zeros((5,6,2,3),np.float32)
    nib.save(nib.Nifti1Image(image,np.eye(4)),tmp_path/'ABC_sa.nii.gz')
    nib.save(nib.Nifti1Image(image,np.eye(4)),tmp_path/'ABC_sa_gt.nii.gz')
    load=nib.load
    def guard(path,*a,**k):
        assert '_gt' not in str(path)
        return load(path,*a,**k)
    monkeypatch.setattr(nib,'load',guard)
    records=discover_mnms_full_cine(tmp_path);assert len(records)==1 and records[0].patient_id=='ABC'

def test_split_teacher_export_evaluator_end_to_end(tmp_path,monkeypatch):
    data,split=_nifti_fixture(tmp_path)
    from self_audit_pseudolabel.pipeline_v3 import teacher_main,FrozenPseudoDataset
    import nibabel as nib
    load=nib.load
    def guard(path,*a,**k):
        assert '_gt' not in str(path),'method attempted a reference read'
        return load(path,*a,**k)
    out=tmp_path/'teacher'
    with monkeypatch.context() as m:
        m.setattr(nib,'load',guard)
        teacher_main(['--dataset','acdc','--root',str(data),'--split-manifest',str(split),
            '--out',str(out),'--max-train-batches','2','--threads','1','--batch-size','2'])
    manifest=verify_frozen(out/'FROZEN.json')
    assert manifest['config']['producer_patient_ids']==['patient001'] and len(manifest['entries'])==12
    ds=FrozenPseudoDataset(data,'acdc',out/'FROZEN.json')
    assert len(ds)==6 and {r[2] for r in ds.rows}=={'patient001'}
    result=tmp_path/'scores'/'eval.json'
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_pseudolabel_frozen.py'),
        '--run',str(out),'--references',str(data),'--out',str(result)],check=True,capture_output=True,text=True)
    scores=json.loads(result.read_text());assert scores['patients']==1 and scores['scored_slices']==4
    assert scores['patient_rows'][0]['patient']=='patient002'

def test_student_refuses_background_only_freeze(tmp_path):
    data,split=_nifti_fixture(tmp_path)
    from self_audit_pseudolabel.pipeline_v3 import inventory,discover,student_main
    splits=read_patient_splits(split);records=inventory(discover('acdc',data),splits)
    run=tmp_path/'freeze';run.mkdir();entries=[]
    for r in records:
        for t in range(3):
            for z in range(2):
                label=np.zeros((r['shape'][1],r['shape'][0]),np.uint8);soft=np.eye(4)[label].transpose(2,0,1)
                path=f"{r['patient_id']}_{t}_{z}.npz";meta={'patient_id':r['patient_id'],'t':t,'z':z}
                sha=export_pseudo_npz(run/path,pseudo_label=label,valid=np.ones_like(label),soft_label=soft,metadata=meta)
                entries.append({**meta,'path':path,'sha256':sha,'split':r['split'],'valid_foreground':0})
    cfg={'dataset':'acdc','image_records':records,'export_records':records,'split_patients':splits,'producer_patient_ids':splits['train']}
    write_freeze_manifest(run,entries,cfg)
    with pytest.raises(ValueError,match='NO_FOREGROUND_SEEDS'):
        student_main(['--dataset','acdc','--root',str(data),'--manifest',str(run/'FROZEN.json'),'--out',str(tmp_path/'student.pt')])
    assert not (tmp_path/'student.pt').exists()

def test_student_dynamic_window_pass_counts_and_gradients():
    if importlib.util.find_spec('self_audit') is None:pytest.skip('canonical source exercised in repository CI')
    from self_audit_pseudolabel.system_v3 import AdaptiveAnnotationStudent
    model=AdaptiveAnnotationStudent(width=32,window_k=4);x=ctx();enc=[];dw=[]
    h1=model.encoder.register_forward_hook(lambda *a:enc.append(1))
    h2=model.refiner.refinement_block.register_forward_hook(lambda *a:dw.append(1))
    for profile,count in [('compact',0),('balanced',1),('accurate',3)]:
        enc.clear();dw.clear();out=model(x,profile=profile)
        assert len(enc)==1 and len(dw)==count
        if profile=='compact': assert torch.equal(out['a0_logits'],out['final_logits'])
    out['final_logits'].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.refiner.parameters())
    h1.remove();h2.remove()

def test_evaluator_rejects_tampering_before_reference_discovery(tmp_path):
    root=_freeze(tmp_path)
    with (root/'one.npz').open('ab') as f:f.write(b'changed')
    cp=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_pseudolabel_frozen.py'),
        '--run',str(root),'--references',str(tmp_path/'nonexistent_references'),'--out',str(tmp_path/'result.json')],capture_output=True,text=True)
    assert cp.returncode!=0 and 'frozen output changed' in cp.stderr
    assert not (tmp_path/'result.json').exists()

def test_mixed_validity_student_training_end_to_end(tmp_path):
    data,split=_nifti_fixture(tmp_path)
    if importlib.util.find_spec('self_audit') is None:pytest.skip('canonical source exercised in repository CI')
    from self_audit_pseudolabel.pipeline_v3 import inventory,discover,student_main
    splits=read_patient_splits(split);records=inventory(discover('acdc',data),splits)
    run=tmp_path/'toy_freeze';run.mkdir();entries=[]
    # Artificial partial labels for a SOFTWARE test; not generated medical evidence.
    for r in records:
        for t in range(3):
            for z in range(2):
                y,x=r['shape'][1],r['shape'][0]
                valid=np.zeros((y,x),np.uint8);valid[3:10,3:10]=1
                label=np.full((y,x),255,np.uint8);label[valid.astype(bool)]=3
                soft=np.zeros((4,y,x),np.float32);soft[0]=1;soft[0,3:10,3:10]=0;soft[3,3:10,3:10]=1
                name=f"{r['patient_id']}_{t}_{z}.npz";meta={'patient_id':r['patient_id'],'t':t,'z':z}
                sha=export_pseudo_npz(run/name,pseudo_label=label,valid=valid,soft_label=soft,metadata=meta)
                entries.append({**meta,'path':name,'sha256':sha,'split':r['split'],'valid_foreground':49})
    cfg={'dataset':'acdc','image_records':records,'export_records':records,'split_patients':splits,
         'producer_patient_ids':splits['train'],'resolved_config':{'deployment':{'student_width':32,'window_k':4}}}
    write_freeze_manifest(run,entries,cfg)
    out=tmp_path/'student.pt'
    student_main(['--dataset','acdc','--root',str(data),'--manifest',str(run/'FROZEN.json'),'--out',str(out),
                  '--profile','balanced','--max-train-batches','2','--threads','1','--batch-size','2'])
    result=json.loads(out.with_suffix('.json').read_text())
    assert out.exists() and result['optimizer_steps']==2 and result['trained_patient_ids']==['patient001']
