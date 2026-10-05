"""05-Oct checkup regressions. All image/training evidence here is synthetic CPU only."""
import importlib.util
import itertools
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from self_audit_pseudolabel.evidence import build_region_evidence
from self_audit_pseudolabel.evolution import accepted_region_mask
from self_audit_pseudolabel.losses_v3 import seed_cross_entropy
from self_audit_pseudolabel.adaptive import AdaptiveRuntime, RuntimeBudget, choose_profile
from self_audit_pseudolabel.freeze import (export_pseudo_npz, verify_frozen, _digest, sha256_file)
from self_audit_pseudolabel.system_v3 import AdaptiveAnnotationStudent, pseudo_supervision_loss

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1); torch.manual_seed(42)


def evidence(labels,k=None,axis='+x'):
    q=torch.nn.functional.one_hot(labels.long(),k or int(labels.max())+1).permute(2,0,1)[None].float()
    z=torch.zeros(1,1,*labels.shape)
    return build_region_evidence(q,z,z,patient_left_axis=[axis])[0]


def ring(rv=False):
    y,x=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
    r=((y-16)**2+(x-16)**2).float().sqrt()
    labels=torch.zeros(32,32,dtype=torch.long)
    labels[(r>=7)&(r<=9)]=1; labels[r<7]=2
    if rv: labels[(((y-16)**2+(x-5)**2)<=16)&(labels==0)]=3
    return labels


def test_ring_semantics_and_validity_not_just_positive_logits():
    ev=evidence(ring(),4); accepted=accepted_region_mask(ev.softmax(-1),min_prob=.7,min_margin=.2)
    assert ev[0,:3].argmax(-1).tolist()==[0,2,3]
    assert accepted[0,1:3].tolist()==[True,True]
    assert ev[0,0,2]==0 and ev[0,1,3]==0
    assert not accepted[0,3]


@pytest.mark.parametrize('kind',['plain','rv','nested','disconnected','border','two_cavities'])
def test_region_id_permutation_equivariance(kind):
    labels=ring(rv=kind=='rv')
    if kind=='disconnected': labels[2:5,2:5]=1
    if kind=='border': labels=torch.roll(labels,16,1)
    if kind=='nested':
        y,x=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
        r=((y-16)**2+(x-16)**2).float().sqrt();labels[(r>=11)&(r<=13)]=3
    if kind=='two_cavities':
        labels=torch.zeros(32,40,dtype=torch.long)
        y,x=torch.meshgrid(torch.arange(32),torch.arange(40),indexing='ij')
        for cx,wall,cavity in [(10,1,2),(29,3,4)]:
            r=((y-16)**2+(x-cx)**2).float().sqrt()
            labels[(r>=3)&(r<=5)]=wall;labels[r<3]=cavity
    k=max(4,int(labels.max())+1); base=evidence(labels,k)
    for perm in itertools.permutations(range(k)):
        changed=evidence(torch.tensor(perm)[labels],k)
        assert torch.equal(changed[:,list(perm)],base)
    if kind in {'disconnected','border'}:
        assert not accepted_region_mask(base.softmax(-1),min_prob=.7,min_margin=.2)[0,1]


def test_no_orientation_abstains_from_rv_at_default_gate():
    ev=evidence(ring(rv=True),axis='')
    accepted=accepted_region_mask(ev.softmax(-1),min_prob=.7,min_margin=.2)
    assert not accepted[0,3] and ev[0,3].softmax(-1)[1]<.7
    oriented=evidence(ring(rv=True))
    assert oriented[0,3].argmax()==1
    assert accepted_region_mask(oriented.softmax(-1),min_prob=.7,min_margin=.2)[0,3]


@pytest.mark.parametrize('scale',[100.,-100.])
def test_seed_ce_saturated_logits_have_correct_gradient(scale):
    logits=torch.tensor([[[scale,0.,0.,-scale]]],requires_grad=True)
    target=3 if scale>0 else 0
    ev=torch.zeros_like(logits,requires_grad=True)
    with torch.no_grad(): ev[0,0,target]=10
    loss=seed_cross_entropy(logits,ev,torch.ones(1,1,dtype=torch.bool));loss.backward()
    assert torch.isfinite(loss) and loss.item()==pytest.approx(200)
    assert logits.grad[0,0,target]<-.99
    assert logits.grad[0,0,3-target]>.99 and ev.grad is None


@pytest.mark.parametrize('profile,rows',[('compact',0),('balanced',1),('accurate',2)])
def test_profile_embedding_gradient_coverage(profile,rows):
    model=AdaptiveAnnotationStudent(width=32,window_k=4)
    x=torch.randn(1,3,17,19);target=torch.full((1,17,19),3,dtype=torch.long)
    out=model(x,profile=profile)
    pseudo_supervision_loss(out,target,torch.ones_like(target,dtype=torch.bool)).backward()
    embeddings=[p for name,p in model.named_parameters() if ('turn_embedding' in name or 'iteration_embedding' in name)]
    assert len(embeddings)==2
    for param in embeddings:
        if rows==0: assert param.grad is None
        else:
            assert (param.grad[:rows].abs().sum(1)>0).all()
            assert (param.grad[rows:].abs().sum(1)==0).all()


@pytest.mark.parametrize('profile',['compact','balanced'])
def test_state_dict_and_runtime_enforce_profile_cap(profile):
    model=AdaptiveAnnotationStudent(width=32,window_k=4)
    with pytest.raises(ValueError,match='missing trained'): AdaptiveRuntime(model)
    model.mark_profile_trained(profile)
    copy=AdaptiveAnnotationStudent(width=32,window_k=4);copy.load_state_dict(model.state_dict())
    runtime=AdaptiveRuntime(copy)
    assert runtime.budget.max_profile==profile
    with pytest.raises(ValueError,match='exceeds'): AdaptiveRuntime(copy,RuntimeBudget())
    with pytest.raises(ValueError,match='exceeds'): copy(torch.rand(1,3,16,16),profile='accurate')


def test_entropy_routing_not_diluted_by_background_or_batch():
    certain=torch.zeros(1,4,9,11);certain[:,0]=30
    uncertain=certain.clone();uncertain[:,:,4,5]=0
    budget=RuntimeBudget()
    assert choose_profile(certain,budget)=='compact'
    assert choose_profile(uncertain,budget)=='accurate'
    from self_audit_pseudolabel.adaptive import sample_entropies
    assert sample_entropies(torch.cat([certain]*10+[uncertain])).tolist()[-1]==pytest.approx(1)
    # Config thresholds, not a second hidden set of defaults, control selection.
    logits=torch.tensor([3.,0.,0.,0.]).reshape(1,4,1,1)
    assert choose_profile(logits,RuntimeBudget.from_config({'entropy_compact':.9,'entropy_accurate':.95},max_profile='accurate'))=='compact'
    assert choose_profile(logits,RuntimeBudget.from_config({'entropy_compact':.01,'entropy_accurate':.02},max_profile='accurate'))=='accurate'


@pytest.mark.parametrize('name',['gt/ABC_sa.nii.gz','segmentation/ABC_sa.nii.gz','ABC_sa_segmentation.nii.gz',
                                  'ABC-sa-mask.nii.gz','reference/ABC_sa.nii.gz'])
def test_reference_path_rejected_before_header(tmp_path,monkeypatch,name):
    import nibabel as nib
    from self_audit_pseudolabel.data_v3 import discover_mnms_full_cine
    path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    monkeypatch.setattr(nib,'load',lambda *_:pytest.fail('reference header opened'))
    with pytest.raises(FileNotFoundError): discover_mnms_full_cine(tmp_path)


def test_unknown_mnms_layout_fails_before_header(tmp_path,monkeypatch):
    import nibabel as nib
    from self_audit_pseudolabel.data_v3 import discover_mnms_full_cine
    (tmp_path/'unknown.nii.gz').touch()
    monkeypatch.setattr(nib,'load',lambda *_:pytest.fail('unknown header opened'))
    with pytest.raises(ValueError,match='unrecognized image layout'): discover_mnms_full_cine(tmp_path)


@pytest.fixture
def frozen(tmp_path):
    import nibabel as nib
    from self_audit_pseudolabel.pipeline_v3 import discover,inventory
    from pseudolabel_v3_fixtures import seal_test_teacher
    data=tmp_path/'images';data.mkdir()
    splits={'train':['patient001'],'val':['patient002'],'test':['patient003']}
    for pid in ['patient001','patient002','patient003']:
        folder=data/pid;folder.mkdir()
        a=np.random.default_rng(2).random((17,19,2,3),dtype=np.float32)
        affine=np.diag([1.2,1.3,5.,1.]);affine[:3,3]=[7,8,9]
        image=nib.Nifti1Image(a,affine);image.header.set_xyzt_units('mm','sec');image.header.set_zooms((1.2,1.3,5.,.04));nib.save(image,folder/f'{pid}_4d.nii.gz')
        (folder/'Info.cfg').write_text('ED: 1\nES: 3\n')
        for frame in (1,3):
            truth=np.zeros((17,19,2),np.uint8);truth[2:8,3:9]=3
            im=nib.Nifti1Image(truth,affine);im.header.set_xyzt_units('mm');nib.save(im,folder/f'{pid}_frame{frame:02d}_gt.nii.gz')
    records=inventory(discover('acdc',data),splits)
    run=tmp_path/'freeze';run.mkdir();entries=[]
    for record in records:
        for t in range(3):
            for z in range(2):
                label=np.zeros((19,17),np.uint8);label[3:9,2:8]=3
                soft=np.eye(4,dtype=np.float32)[label].transpose(2,0,1)
                meta={'patient_id':record['patient_id'],'t':t,'z':z};name=f"{meta['patient_id']}_{t}_{z}.npz"
                digest=export_pseudo_npz(run/name,pseudo_label=label,valid=np.ones_like(label),soft_label=soft,metadata=meta)
                entries.append({**meta,'path':name,'sha256':digest,'split':record['split']})
    cfg={'dataset':'acdc','image_records':records,'export_records':records,'split_patients':splits,'producer_patient_ids':splits['train']}
    seal_test_teacher(run,entries,cfg)
    return data,run


def resign(run,payload):
    body=dict(payload);body.pop('manifest_id',None);body['manifest_id']=_digest(body)
    (run/'FROZEN.json').write_text(json.dumps(body))


def evaluator():
    spec=importlib.util.spec_from_file_location('checkup_evaluator',ROOT/'scripts/evaluate_pseudolabel_frozen.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.main


def references(data):
    return [{'patient_id':'patient002','phase':phase,'t':t,'path':str(data/'patient002'/f'patient002_frame{frame:02d}_gt.nii.gz'),
             'label_map':{'0':0,'1':1,'2':2,'3':3}} for phase,t,frame in [('ED',0,1),('ES',2,3)]]


@pytest.mark.parametrize('field,value',[('t',-.1),('t',.9),('t',True),('t','0'),('t',3),
    ('reference_frame_index',-1),('reference_frame_index',1.9),('reference_frame_index',False),('reference_frame_index','1')])
def test_invalid_phase_indices_fail_before_reference_load(frozen,tmp_path,monkeypatch,field,value):
    import nibabel as nib
    data,run=frozen;refs=references(data);refs[0][field]=value
    path=tmp_path/'references.json';path.write_text(json.dumps({'references':refs}))
    monkeypatch.setattr(nib,'load',lambda *_:pytest.fail('invalid phase opened a reference'))
    with pytest.raises(ValueError,match='integer'):
        evaluator()(['--run',str(run),'--reference-manifest',str(path),'--out',str(tmp_path/'scores.json')])


@pytest.mark.parametrize('case',['missing_phase','unit_mismatch','unknown_unit','out_of_range_4d'])
def test_reference_protocol_and_geometry_fail_closed(frozen,tmp_path,case):
    import nibabel as nib
    data,run=frozen;refs=references(data)
    if case=='missing_phase': refs=refs[:1]
    else:
        path=Path(refs[0]['path']);im=nib.load(path);array=np.asarray(im.dataobj)
        if case=='out_of_range_4d': array=np.stack([array,array],-1);refs[0]['reference_frame_index']=2
        changed=nib.Nifti1Image(array,im.affine)
        changed.header.set_xyzt_units({'unit_mismatch':'meter','unknown_unit':'unknown'}.get(case,'mm'))
        nib.save(changed,path)
    path=tmp_path/'refs.json';path.write_text(json.dumps({'references':refs}))
    with pytest.raises(ValueError): evaluator()(['--run',str(run),'--reference-manifest',str(path),'--out',str(tmp_path/'scores.json')])
    assert not (tmp_path/'scores.json').exists()


@pytest.mark.parametrize('split,kind',[('train','in_sample'),('val','held_out_development'),('test','held_out_test')])
def test_evaluator_split_phase_and_reference_provenance(frozen,tmp_path,split,kind):
    data,run=frozen
    result=evaluator()(['--run',str(run),'--references',str(data),'--split',split,'--out',str(tmp_path/'scores.json')])
    assert result['evaluation_kind']==kind and len(result['frame_rows'])==2
    assert result['phase_metrics']['ED']['lv']==1 and result['phase_metrics']['ES']['lv']==1
    assert all(len(r['sha256'])==64 for r in result['reference_rows'])
    assert result['test_untouched']=='NOT_CERTIFIED_BY_CODE'


@pytest.mark.parametrize('mutation',['missing_artifact','missing_inventory','stats','split','shape','locked_split'])
def test_self_consistent_but_semantically_invalid_freeze_rejected(frozen,mutation):
    _,run=frozen;p=json.loads((run/'FROZEN.json').read_text())
    if mutation=='missing_artifact':p['artifacts'].pop('teacher.pt')
    if mutation=='missing_inventory':p['config'].pop('export_records')
    if mutation=='stats':p['entries'][0]['valid_foreground']+=1
    if mutation=='split':p['entries'][0]['split']='test'
    if mutation=='shape':
        p['config']['image_records'][0]['shape'][0]+=1;p['config']['export_records'][0]['shape'][0]+=1
    if mutation=='locked_split':p['config']['split_patients']['train'].append('phantom')
    resign(run,p)
    with pytest.raises(ValueError):verify_frozen(run/'FROZEN.json',expected_role='teacher_freeze')


def test_bounded_background_only_updates_write_no_checkpoint(frozen,tmp_path,monkeypatch):
    data,run=frozen
    from self_audit_pseudolabel.pipeline_v3 import student_main
    payload=json.loads((run/'FROZEN.json').read_text())
    entry=next(e for e in payload['entries'] if e['split']=='train')
    path=run/entry['path']
    with np.load(path) as data_row:
        shape=data_row['pseudo_label'].shape;meta=json.loads(str(data_row['metadata_json']))
    label=np.zeros(shape,np.uint8);soft=np.eye(4,dtype=np.float32)[label].transpose(2,0,1)
    path.unlink()
    entry['sha256']=export_pseudo_npz(path,pseudo_label=label,valid=np.ones(shape,np.uint8),soft_label=soft,metadata=meta)
    entry.update(valid_foreground=0,class_pixels=[label.size,0,0,0],valid_fraction=1.)
    resign(run,payload)
    from self_audit_pseudolabel.data_v3 import PatientBatchSampler
    monkeypatch.setattr(PatientBatchSampler,'__iter__',lambda self:iter([[i] for i in range(6)]))
    out=tmp_path/'student.pt'
    with pytest.raises(ValueError,match='NO_OBSERVED_FOREGROUND'):
        student_main(['--dataset','acdc','--root',str(data),'--manifest',str(run/'FROZEN.json'),'--out',str(out),
                      '--max-train-batches','1','--profile','compact','--threads','1'])
    assert not out.exists()


@pytest.mark.parametrize('profile',['compact','balanced'])
def test_student_checkpoint_native_test_export_and_evaluation(frozen,tmp_path,profile):
    data,run=frozen
    from self_audit_pseudolabel.pipeline_v3 import student_main
    from self_audit_pseudolabel.checkpoint import load_student_checkpoint
    from self_audit_pseudolabel.inference_v3 import student_inference_main
    checkpoint=tmp_path/'student.pt'
    result=student_main(['--dataset','acdc','--root',str(data),'--manifest',str(run/'FROZEN.json'),'--out',str(checkpoint),
                        '--max-train-batches','1','--profile',profile,'--threads','1'])
    assert result['coverage']['class_pixels'][3]>0
    assert result['coverage']['missing_foreground_classes']==[1,2]
    model,budget,payload=load_student_checkpoint(checkpoint)
    assert budget.max_profile==profile and payload['checkpoint_sha256']==sha256_file(checkpoint)
    with pytest.raises(ValueError,match='exceeds'):load_student_checkpoint(checkpoint,max_profile='accurate')
    prediction=tmp_path/'predictions'
    before=sha256_file(checkpoint)
    exported=student_inference_main(['--source-freeze',str(run/'FROZEN.json'),'--root',str(data),
        '--checkpoint',str(checkpoint),'--out',str(prediction),'--split','test','--threads','1'])
    assert sha256_file(checkpoint)==before
    assert {e['patient_id'] for e in exported['entries']}=={'patient003'}
    import nibabel as nib
    native=nib.load(prediction/'native/patient003.nii.gz')
    source=nib.load(data/'patient003/patient003_4d.nii.gz')
    assert native.shape==source.shape and np.allclose(native.affine,source.affine)
    assert native.header.get_xyzt_units()==('mm','sec')
    assert np.allclose(native.header.get_zooms(),source.header.get_zooms())
    arr=np.asarray(native.dataobj)
    for e in exported['entries']:
        with np.load(prediction/e['path']) as p:
            assert np.array_equal(arr[:,:,e['z'],e['t']].T,p['pseudo_label'])
    scores=evaluator()(['--run',str(prediction),'--references',str(data),'--split','test','--out',str(tmp_path/'scores.json')])
    assert scores['artifact_role']=='student_predictions' and len(scores['frame_rows'])==2
    # Dependency identity is enforced even when tensor shapes remain compatible.
    bad=torch.load(checkpoint,weights_only=True);bad['source_sha256']['self_audit/models/dynamic_window.py']='0'*64
    torch.save(bad,tmp_path/'changed.pt')
    with pytest.raises(ValueError,match='source/dependency'):load_student_checkpoint(tmp_path/'changed.pt')


def test_temporal_default_policy_matrix():
    from self_audit_pseudolabel.consistency import consistency_gate
    p=torch.zeros(3,1,4,3,3);p[0,:,1]=1;p[1,:,2]=1;p[2,:,1]=1
    v=torch.ones(3,1,3,3,dtype=torch.bool)
    _,accepted=consistency_gate(p,v,temporal_weight=.5,min_agreement=.6)
    assert accepted[0].all() and accepted[2].all() and not accepted[1].any()
    v[2]=False
    _,accepted=consistency_gate(p,v,temporal_weight=.5,min_agreement=.6)
    assert accepted[1].all() and not accepted[2].any()


def test_patient_sampler_retains_cache_locality():
    from self_audit_pseudolabel.data_v3 import PatientBatchSampler
    keys=['a']*9+['b']*7+['c']*8
    batches=list(PatientBatchSampler(keys,2))
    order=[keys[b[0]] for b in batches]
    assert sum(a!=b for a,b in zip(order,order[1:]))==2
    assert sorted(i for b in batches for i in b)==list(range(len(keys)))


def test_config_epoch_default_and_cli_override(tmp_path):
    import argparse
    from self_audit_pseudolabel.pipeline_v3 import load_config,_common,_validate
    config=json.loads((ROOT/'configs/pseudolabel_v3.json').read_text())
    config['training']['bounded_default_epochs']=3
    config['deployment'].update(entropy_compact=.6,entropy_accurate=.9)
    path=tmp_path/'config.json';path.write_text(json.dumps(config));cfg=load_config(path)
    parser=argparse.ArgumentParser();_common(parser)
    base=['--dataset','acdc','--root','images','--out','unused']
    args=parser.parse_args(base);_validate(args,cfg);assert args.epochs==3
    args=parser.parse_args(base+['--epochs','2']);_validate(args,cfg);assert args.epochs==2


def test_runtime_routing_is_per_sample_and_encodes_once(monkeypatch):
    model=AdaptiveAnnotationStudent(width=32,window_k=4);model.mark_profile_trained('accurate')
    runtime=AdaptiveRuntime(model)
    certain=torch.zeros(1,4,16,16);certain[:,0]=30
    uncertain=torch.zeros_like(certain)
    logits=torch.cat([certain,uncertain])
    monkeypatch.setattr(model,'initial_logits',lambda feat,hw:logits[:len(feat)])
    calls=[];hook=model.encoder.register_forward_hook(lambda *args:calls.append(1))
    result=runtime(torch.rand(2,3,16,16));hook.remove()
    assert len(calls)==1 and result['profiles']==('compact','accurate') and result['profile']=='mixed'
    assert result['final_logits'].shape==(2,4,16,16)


def test_phase_macro_and_phase_pooled_estimands_differ(frozen,tmp_path):
    import nibabel as nib
    data,run=frozen
    path=data/'patient002/patient002_frame03_gt.nii.gz';image=nib.load(path)
    truth=np.asarray(image.dataobj).copy();truth[:,:,:]=3
    modified=nib.Nifti1Image(truth,image.affine);modified.header.set_xyzt_units('mm');nib.save(modified,path)
    result=evaluator()(['--run',str(run),'--references',str(data),'--out',str(tmp_path/'scores.json')])
    assert result['phase_macro']['lv']!=pytest.approx(result['lv'])
    ed=result['phase_metrics']['ED']['lv'];es=result['phase_metrics']['ES']['lv']
    assert result['phase_macro']['lv']==pytest.approx((ed+es)/2)
    assert result['lv']==pytest.approx(144/(108+17*19))


def test_teacher_test_export_has_no_optimizer_or_reference_reads(frozen,tmp_path,monkeypatch):
    import nibabel as nib
    from self_audit_pseudolabel.pipeline_v3 import teacher_main
    from self_audit_pseudolabel.inference_v3 import teacher_export_main
    data,_=frozen
    split=tmp_path/'split.json';split.write_text(json.dumps({'train':['patient001'],'val':['patient002'],'test':['patient003']}))
    source=tmp_path/'teacher'
    load=nib.load
    def guard(path,*a,**kw):
        assert '_gt' not in str(path)
        return load(path,*a,**kw)
    monkeypatch.setattr(nib,'load',guard)
    teacher_main(['--dataset','acdc','--root',str(data),'--split-manifest',str(split),'--out',str(source),
                  '--max-train-batches','1','--threads','1'])
    before=sha256_file(source/'teacher.pt')
    monkeypatch.setattr(torch.optim.AdamW,'step',lambda *a,**kw:pytest.fail('test export retrained teacher'))
    result=teacher_export_main(['--source-freeze',str(source/'FROZEN.json'),'--root',str(data),
                               '--out',str(tmp_path/'test_export'),'--split','test','--threads','1'])
    assert {e['patient_id'] for e in result['entries']}=={'patient003'}
    assert sha256_file(source/'teacher.pt')==before
    assert result['config']['inference_only'] is True


def test_streamed_temporal_gate_matches_full_volume(frozen,tmp_path):
    from self_audit_pseudolabel.native_export import export_native
    from self_audit_pseudolabel.consistency import consistency_gate
    from self_audit_pseudolabel.pipeline_v3 import discover,inventory
    data,run=frozen;record=[r for r in discover('acdc',data) if r.patient_id=='patient002'][0]
    splits=json.loads((run/'FROZEN.json').read_text())['config']['split_patients']
    info=inventory([record],splits)[0]
    soft=torch.softmax(torch.randn(3,2,4,19,17),2);valid=torch.rand(3,2,19,17)>.15
    fp=torch.randn(3,2,2,5,5)*.1;fn=torch.randn_like(fp)*.1
    cfg={'enabled':True,'slice_weight':0,'temporal_weight':2.,'min_agreement':.6}
    expected_soft,expected_valid=consistency_gate(soft,valid,temporal_weight=2.,min_agreement=.6,flow_prev=fp,flow_next=fn)
    def predict(b):
        t,z=int(b['t'][0]),int(b['z'][0])
        return {'soft_label':soft[t,z][None],'valid':valid[t,z][None],'flow_prev':fp[t,z][None],'flow_next':fn[t,z][None]}
    out=tmp_path/'streamed';out.mkdir()
    entries,_,report=export_native([record],{record.patient_id:info},out,predict,cfg)
    for e in entries:
        with np.load(out/e['path']) as p:
            assert np.array_equal(p['valid'],expected_valid[e['t'],e['z']].numpy())
            assert np.array_equal(p['soft_label'],expected_soft[e['t'],e['z']].numpy())
    assert report['patient_rows'][0]['cache_misses']==1
    assert not list(out.glob('.cine-*'))


@pytest.mark.parametrize('mutation',['threshold','profile','environment','manifest','preprocessing'])
def test_checkpoint_contract_mismatch_rejected(tmp_path,mutation):
    from self_audit_pseudolabel.checkpoint import save_student_checkpoint,load_student_checkpoint
    model=AdaptiveAnnotationStudent(width=32,window_k=4)
    config=json.loads((ROOT/'configs/pseudolabel_v3.json').read_text())['deployment'];config['window_k']=4
    coverage={'class_pixels':[2,2,2,2],'patient_ids':['p'],'bounded_training':True}
    path=tmp_path/'model.pt'
    save_student_checkpoint(path,model,args={'profile':'balanced'},model_config=config,manifest_id='original',coverage=coverage)
    payload=torch.load(path,weights_only=True)
    if mutation=='threshold':payload['model_config']['entropy_compact']=.1
    if mutation=='profile':payload['profile_trained']='accurate'
    if mutation=='environment':payload['environment']['torch']='other'
    if mutation=='preprocessing':payload['preprocessing']={}
    torch.save(payload,path)
    with pytest.raises(ValueError):load_student_checkpoint(path,manifest_id='other' if mutation=='manifest' else 'original')


def test_observed_patient_coverage_excludes_unseen_declared_patients(frozen,tmp_path,monkeypatch):
    from pseudolabel_v3_fixtures import seal_test_teacher
    from self_audit_pseudolabel.pipeline_v3 import student_main
    from self_audit_pseudolabel.data_v3 import PatientBatchSampler
    data,run=frozen;payload=json.loads((run/'FROZEN.json').read_text());cfg=payload['config']
    cfg['split_patients']={'train':['patient001','patient002'],'val':[],'test':['patient003']}
    cfg['producer_patient_ids']=cfg['split_patients']['train']
    for record in cfg['image_records']+cfg['export_records']:
        if record['patient_id']=='patient002':record['split']='train'
    for entry in payload['entries']:
        if entry['patient_id']=='patient002':entry['split']='train'
    (run/'FROZEN.json').unlink();seal_test_teacher(run,payload['entries'],cfg)
    monkeypatch.setattr(PatientBatchSampler,'__iter__',lambda self:iter([[i] for i in range(12)]))
    result=student_main(['--dataset','acdc','--root',str(data),'--manifest',str(run/'FROZEN.json'),
        '--out',str(tmp_path/'student.pt'),'--profile','compact','--threads','1','--max-train-batches','1'])
    assert result['declared_train_patient_ids']==['patient001','patient002']
    assert result['trained_patient_ids']==['patient001']


def test_source_identity_checks_actually_imported_dependencies(tmp_path,monkeypatch):
    from self_audit_pseudolabel.checkpoint import source_identity
    from self_audit.models import dynamic_window
    monkeypatch.setattr(dynamic_window,'__file__',str(tmp_path/'other_dynamic_window.py'))
    with pytest.raises(ValueError,match='imported student dependency'):source_identity()


def test_checkpoint_refuses_source_drift_during_training(tmp_path):
    from self_audit_pseudolabel.checkpoint import save_student_checkpoint
    path=tmp_path/'student.pt';model=AdaptiveAnnotationStudent(width=32,window_k=4)
    with pytest.raises(ValueError,match='source changed during training'):
        save_student_checkpoint(path,model,args={'profile':'compact'},model_config={},manifest_id='test',
                                coverage={'class_pixels':[0,1,0,0]},source_at_start={'old':'source'})
    assert not path.exists()
