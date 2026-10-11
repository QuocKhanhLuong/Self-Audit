"""Full-path contracts. Native NIfTI tests require genuine nibabel (never a stub).

Synthetic files/provider are clearly labeled fixtures; they are not ACDC results.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from self_audit_guidance.io import (write_json,read_json,read_splits,digest,read_cine,
                                    freeze,verify,write_native_labels)
from self_audit_guidance.full import derive_split,stage_command,existing_stage,run_full
from self_audit_guidance.evaluate import score_volume,aggregate_scores,resolve_references,evaluate
from self_audit_guidance.preflight import probe_student,run_preflight,inventory
from self_audit_guidance.core import GuidanceConfig
from self_audit_guidance.pipeline import prepare,export
from self_audit_guidance.student import train_student


def test_unknown_is_error_not_background_or_discarded():
    ref=np.zeros((4,4,2),np.uint8);ref[:2]=3
    m=score_volume(np.full_like(ref,255),ref)
    assert m['classes']['LV']['dice']==0 and m['classes']['LV']['fn']==16
    assert m['coverage']==0 and m['accuracy_all_pixels']==0
    assert m['classes']['RV']['dice'] is None
    assert m['classes']['LV']['surface_status']=='MISSING_PREDICTION'


def test_exact_segmentation_dice_and_hd95():
    ref=np.zeros((5,7,2),np.uint8);ref[1:3,1:4]=2;ref[3:,2:5]=3
    m=score_volume(ref,ref,np.diag([2,3,4,1]),'mm')
    assert m['classes']['MYO']['dice']==1
    assert m['classes']['MYO']['hd95_mm']==0
    assert m['classes']['RV']['dice'] is None


def test_hd95_uses_native_affine_spacing():
    ref=np.zeros((5,5,5),np.uint8);ref[1,2,2]=3
    pred=np.zeros_like(ref);pred[2,2,2]=3
    m=score_volume(pred,ref,np.diag([2.5,1,1,1]),'mm')
    assert m['classes']['LV']['hd95_mm']==2.5


def test_patient_macro_not_frame_weighted():
    r=np.full((2,2,2),3,np.uint8)
    good=score_volume(r,r);bad=score_volume(np.full_like(r,255),r)
    rows=[dict(patient_id='p1',metrics=good)]*3+[dict(patient_id='p2',metrics=bad)]
    assert aggregate_scores(rows)['patient_macro_dice']['LV']==.5


@pytest.mark.parametrize('labels',[4,255,-1,np.nan])
def test_invalid_reference_class_refused(labels):
    with pytest.raises(ValueError):score_volume(np.zeros((2,2,2)),np.full((2,2,2),labels))


def test_derive_split_preserves_train_and_source(tmp_path):
    p=tmp_path/'source.json';write_json(p,dict(train=['p1','p2'],val=['p3','p4','p5','p6']))
    before=digest(p);a=derive_split(p,tmp_path/'a.json');b=derive_split(p,tmp_path/'b.json')
    assert a==b and digest(p)==before and a['train']==['p1','p2']
    assert len(a['val'])==2 and len(a['test'])==2 and not set(a['val'])&set(a['test'])
    assert 'previously_used_validation' in a['provenance']['test_status']


def test_existing_test_cannot_be_silently_repartitioned(tmp_path):
    p=tmp_path/'source.json';write_json(p,dict(train=['p1'],val=['p2'],test=['p3']))
    with pytest.raises(ValueError,match='already'):derive_split(p,tmp_path/'new.json')


def test_ambiguous_within_split_ids_refused(tmp_path):
    p=tmp_path/'source.json';write_json(p,dict(train=['p1_ED','p1_ES']))
    with pytest.raises(ValueError,match='duplicate'):read_splits(p)


def test_stage_command_does_not_mix_evaluator_and_trainer():
    cmd=stage_command('train',bank='/b',out='/o',epochs=2)
    assert '--reference-root' not in cmd and 'train' in cmd
    assert stage_command('evaluate',predictions='/p',out='/e')[1].endswith('evaluate_stable_guidance.py')


def test_incomplete_stage_never_resumed_as_success(tmp_path):
    p=tmp_path/'student';p.mkdir();(p/'last.pt').write_bytes(b'partial')
    with pytest.raises(RuntimeError,match='INCOMPLETE_STAGE'):existing_stage(p,'student',True)


def test_feature_backward_preflight_is_real_torch_and_checkpoint_io():
    rng=np.random.default_rng(4)
    cine=rng.normal(size=(2,2,16,20)).astype(np.float32)
    features=rng.normal(size=(2,2,8,8,4)).astype(np.float32)
    torch.set_num_threads(1)
    r=probe_student(cine,features,'cpu',3,8,steps=2)
    assert r['encoder_updated'] and r['checkpoint_roundtrip']
    assert all(x>0 for x in r['gradient_norm']) and r['semantic_quality_tested'] is False


def test_preflight_records_blocker_not_mock_pass(tmp_path,monkeypatch):
    from self_audit_guidance import preflight
    def fail():raise RuntimeError('controlled missing dependency')
    monkeypatch.setattr(preflight,'check_dependencies',fail)
    p=tmp_path/'PREFLIGHT.json'
    r=run_preflight('/not-mounted','acdc','/not-present',p,device='cpu')
    assert r['status']=='BLOCKED' and r['failed_stage']=='dependencies'
    assert 'controlled missing' in read_json(p)['error']
    assert r['gt_read'] is False


def test_full_refuses_empty_test_before_training(tmp_path):
    split=tmp_path/'split.json';cfg=tmp_path/'cfg.json'
    write_json(split,dict(train=['p1'],val=['p2']));write_json(cfg,GuidanceConfig().dictionary())
    with pytest.raises(ValueError,match='nonempty'):run_full('/data',split,tmp_path/'run',cfg,device='cpu')
    assert not (tmp_path/'run').exists()


NIB_AVAILABLE=importlib.util.find_spec('nibabel') is not None
requires_nib=pytest.mark.skipif(not NIB_AVAILABLE,reason='genuine nibabel not installed; no fake native-IO pass')


def native_fixture(tmp_path):
    import nibabel as nib
    root=tmp_path/'raw';root.mkdir()
    rng=np.random.default_rng(42)
    ids=['patient001','patient002','patient003','patient004']
    affine=np.array([[0,-1.7,0,14],[1.3,0,0,-8],[0,0,5,3],[0,0,0,1]],float)
    for pid in ids:
        d=root/pid;d.mkdir()
        image=rng.normal(size=(20,16,2,3)).astype(np.float32)
        nii=nib.Nifti1Image(image,affine);nii.header.set_xyzt_units('mm','sec')
        nib.save(nii,d/f'{pid}_4d.nii.gz')
        for t in (1,3):
            mask=np.zeros(image.shape[:3],np.uint8);mask[6:12,5:10,:]=3
            n=nib.Nifti1Image(mask,affine);n.header.set_xyzt_units('mm')
            nib.save(n,d/f'{pid}_frame{t:02d}_gt.nii.gz')
    split=tmp_path/'split.json';write_json(split,dict(train=ids[:2],val=ids[2:3],test=ids[3:]))
    return root,split,affine


@requires_nib
def test_real_nifti_reader_shape_affine_and_full_cine(tmp_path):
    root,split,affine=native_fixture(tmp_path)
    im,meta=read_cine(root/'patient001/patient001_4d.nii.gz')
    assert im.shape==(3,2,16,20) and np.allclose(meta['affine'],affine)
    out=tmp_path/'native.nii.gz';labels=np.full(im.shape,255,np.uint8)
    write_native_labels(out,labels,meta)
    import nibabel as nib
    check=nib.load(out)
    assert check.shape==(20,16,2,3) and np.allclose(check.affine,affine)
    assert np.all(np.asarray(check.dataobj)==255)


@requires_nib
def test_native_inventory_all_splits_no_annotation_read(tmp_path,monkeypatch):
    root,split,_=native_fixture(tmp_path)
    import nibabel as nib
    real=nib.load;opened=[]
    def guarded(path,*a,**kw):
        opened.append(str(path));assert '_gt' not in str(path)
        return real(path,*a,**kw)
    monkeypatch.setattr(nib,'load',guarded)
    cfg=GuidanceConfig(grid=8,feature_dim=4,clusters=(2,),temporal=False)
    r=inventory(root,'acdc',split,cfg)
    assert r['split_counts']=={'train':2,'val':1,'test':1}
    assert len(opened)==4


@requires_nib
def test_native_prepare_train_export_evaluate_roundtrip(tmp_path):
    root,split,affine=native_fixture(tmp_path)
    from test_self_audit_stable_guidance import FrozenSyntheticProvider
    cfg=tmp_path/'cfg.json';write_json(cfg,GuidanceConfig(grid=8,feature_dim=4,clusters=(2,),temporal=False).dictionary())
    bank=tmp_path/'bank'
    # Real native reader, explicit synthetic feature provider: not real pretrained accuracy.
    prepare(root,'acdc',split,bank,cfg,patients=0,device='cpu',teacher=FrozenSyntheticProvider())
    ck=train_student(bank,tmp_path/'student',epochs=1,batch_size=2,device='cpu',width=8,threads=1)
    assert ck['train_patients']==['patient001','patient002']
    for s in ('val','test'):
        out=tmp_path/f'pred_{s}'
        export(tmp_path/'student',root,'acdc',split,s,out,'cpu',2)
        m=verify(out,'student_predictions')
        assert not set(m['patients'])&set(ck['train_patients'])
        evaluated=evaluate(out,tmp_path/f'eval_{s}',reference_root=root)
        assert evaluated['summary']['patient_count']==1 and evaluated['summary']['labeled_frames']==2
        if ck['status']=='REPRESENTATION_ONLY':
            assert evaluated['summary']['patient_macro_dice']['LV']==0
            assert evaluated['summary']['coverage_patient_macro']==0


@requires_nib
def test_annotation_geometry_mismatch_refused(tmp_path):
    root,split,affine=native_fixture(tmp_path)
    pid='patient003';im,meta=read_cine(root/pid/f'{pid}_4d.nii.gz')
    out=tmp_path/'pred';out.mkdir();np.savez_compressed(out/f'{pid}.npz',labels=np.zeros(im.shape,np.uint8))
    freeze(out,dict(role='student_predictions',split='val',patients=[pid],images={pid:meta},
                    train_patients=['patient001'],student_status='TEST_FIXTURE'))
    import nibabel as nib
    path=root/pid/f'{pid}_frame01_gt.nii.gz';n=nib.load(path)
    changed=affine.copy();changed[0,3]+=10
    nib.save(nib.Nifti1Image(np.asarray(n.dataobj),changed),path)
    with pytest.raises(ValueError,match='geometry'):evaluate(out,tmp_path/'eval',reference_root=root)


@requires_nib
def test_corrupt_native_file_fails_before_training(tmp_path):
    path=tmp_path/'broken_4d.nii.gz';path.write_bytes(b'not-a-nifti')
    with pytest.raises(Exception):read_cine(path)


@requires_nib
def test_nan_native_input_fails(tmp_path):
    import nibabel as nib
    a=np.zeros((8,8,2,3),np.float32);a[0,0,0,0]=np.nan
    path=tmp_path/'bad_4d.nii';nib.save(nib.Nifti1Image(a,np.eye(4)),path)
    with pytest.raises(ValueError,match='nonfinite'):read_cine(path)


def test_orchestrator_freezes_both_cohorts_before_evaluator(tmp_path,monkeypatch):
    """Subprocess-order contract only; no pretended NIfTI/teacher execution."""
    from self_audit_guidance import full
    split=tmp_path/'split.json';cfg=tmp_path/'cfg.json';out=tmp_path/'run'
    splits=dict(train=['p1'],val=['p2'],test=['p3']);write_json(split,splits)
    write_json(cfg,GuidanceConfig().dictionary());sequence=[]
    images={p:dict(image_sha256='fixture-image-hash-'+p) for p in ('p1','p2','p3')}
    def value(cmd,key):return Path(cmd[cmd.index('--'+key)+1])
    def fake_stage(name,cmd,root):
        sequence.append(name);dest=value(cmd,'out')
        if name=='preflight':
            write_json(dest,dict(status='PASS_EXECUTION_ONLY_QUALITY_UNVERIFIED',
                checks=dict(native_cine_inventory=dict(images=images))))
            return 0
        dest.mkdir()
        if name=='prepare':
            freeze(dest,dict(role='guidance_bank',patients=['p1'],image_records={'p1':images['p1']},status='CONTRACT_FIXTURE'))
        elif name=='train':
            freeze(dest,dict(role='student',bank_manifest_sha256=digest(out/'bank/FROZEN.json'),status='REPRESENTATION_ONLY'))
        elif name.startswith('export_'):
            cohort=name.split('_')[1];ids=splits[cohort]
            freeze(dest,dict(role='student_predictions',patients=ids,images={p:images[p] for p in ids},
                student_manifest_sha256=digest(out/'student/FROZEN.json')))
        else:
            assert (out/'predictions_val/FROZEN.json').is_file()
            assert (out/'predictions_test/FROZEN.json').is_file()
            assert (out/'PREDICTIONS_LOCKED.json').is_file()
            cohort=name.split('_')[1];write_json(dest/'METRICS.json',dict(summary={'contract_fixture':True}))
            freeze(dest,dict(role='independent_evaluation',prediction_manifest_sha256=digest(out/f'predictions_{cohort}/FROZEN.json')))
        return 2 if name in ('prepare','train') else 0
    monkeypatch.setattr(full,'run_stage',fake_stage)
    result=run_full(tmp_path/'raw',split,out,cfg,epochs=1,device='cpu')
    assert sequence==['preflight','prepare','train','export_val','export_test','evaluate_val','evaluate_test']
    assert result['status']=='COMPLETED_SCORED_NO_SEMANTIC_SUPPORT'
    with pytest.raises(ValueError,match='resume recipe'):
        run_full(tmp_path/'raw',split,out,cfg,epochs=2,device='cpu',resume=True)


def test_blocked_real_preflight_never_proceeds_to_bank(tmp_path,monkeypatch):
    from self_audit_guidance import full
    split=tmp_path/'split.json';cfg=tmp_path/'cfg.json'
    write_json(split,dict(train=['p1'],val=['p2'],test=['p3']));write_json(cfg,GuidanceConfig().dictionary())
    sequence=[]
    def fail(name,cmd,out):
        sequence.append(name);raise RuntimeError('CUDA not available in this contract fixture')
    monkeypatch.setattr(full,'run_stage',fail)
    with pytest.raises(RuntimeError,match='CUDA'):
        run_full(tmp_path/'raw',split,tmp_path/'run',cfg,device='cpu')
    assert sequence==['preflight']
    assert read_json(tmp_path/'run/RUN_STATUS.json')['failed_stage']=='preflight'
    assert not (tmp_path/'run/bank').exists()


def test_official_acdc_ids_from_image_only_names_and_no_resplit(tmp_path):
    from self_audit_guidance.full import external_acdc_patients
    test_root=tmp_path/'testing';test_root.mkdir()
    for pid in ('patient101','patient102'):
        folder=test_root/pid;folder.mkdir()
        (folder/(pid+'_4d.nii.gz')).write_bytes(b'raw-name-only-contract')
        # A labelled-looking filename may coexist but must not be opened.
        (folder/(pid+'_frame01_gt.nii.gz')).write_bytes(b'NEVER_READ')
    assert external_acdc_patients(test_root)==['patient101','patient102']
    (test_root/'patient103').mkdir()
    with pytest.raises(ValueError,match='expected one raw full-cine image'):
        external_acdc_patients(test_root)


def test_official_acdc_pipeline_uses_original_80_20_split_and_scores_val_only(tmp_path,monkeypatch):
    """Orchestrator contract: external TEST has no GT; only VAL evaluator runs."""
    from self_audit_guidance import full
    source=tmp_path/'split.json';cfg=tmp_path/'cfg.json';out=tmp_path/'run'
    original=dict(train=['patient001','patient002'],val=['patient003','patient004'])
    write_json(source,original);original_hash=digest(source)
    write_json(cfg,GuidanceConfig().dictionary())
    test_root=tmp_path/'testing';test_root.mkdir()
    for pid in ('patient101','patient102'):
        folder=test_root/pid;folder.mkdir()
        (folder/(pid+'_4d.nii.gz')).write_bytes(b'FAKE_NOT_READ')
    split_expected=dict(**original,test=['patient101','patient102'])
    images={pid:dict(image_sha256='made-up-hash-'+pid) for pid in sum((split_expected[k] for k in ('train','val','test')),[])}
    stages=[]
    def value(cmd,key):return Path(cmd[cmd.index('--'+key)+1])
    def fake_stage(name,cmd,_):
        stages.append(name);dest=value(cmd,'out')
        if name=='preflight':
            assert value(cmd,'test-root')==test_root
            assert read_splits(value(cmd,'split-manifest'))==split_expected
            write_json(dest,dict(status='PASS_EXECUTION_ONLY_QUALITY_UNVERIFIED',
                                 checks=dict(native_cine_inventory=dict(images=images))))
            return 0
        dest.mkdir()
        if name=='prepare':
            assert value(cmd,'root')==tmp_path/'training'
            assert read_splits(value(cmd,'split-manifest'))==split_expected
            freeze(dest,dict(role='guidance_bank',patients=original['train'],
                image_records={p:images[p] for p in original['train']},status='CONTRACT_FIXTURE'))
        elif name=='train':
            freeze(dest,dict(role='student',bank_manifest_sha256=digest(out/'bank/FROZEN.json'),
                status='REPRESENTATION_ONLY'))
        elif name.startswith('export_'):
            s=name.split('_')[1]
            assert value(cmd,'root')==(test_root if s=='test' else tmp_path/'training')
            assert read_splits(value(cmd,'split-manifest'))==split_expected
            freeze(dest,dict(role='student_predictions',patients=split_expected[s],
                images={p:images[p] for p in split_expected[s]},
                student_manifest_sha256=digest(out/'student/FROZEN.json')))
        elif name=='evaluate_val':
            assert (out/'predictions_val/FROZEN.json').is_file()
            assert (out/'predictions_test/FROZEN.json').is_file()
            assert (out/'PREDICTIONS_LOCKED.json').is_file()
            assert value(cmd,'reference-root')==tmp_path/'training'
            write_json(dest/'METRICS.json',dict(summary={'val_only':True}))
            freeze(dest,dict(role='independent_evaluation',
                prediction_manifest_sha256=digest(out/'predictions_val/FROZEN.json')))
        else:raise AssertionError('GT leakage into official testing: '+name)
        return 2 if name in ('prepare','train') else 0
    monkeypatch.setattr(full,'run_stage',fake_stage)
    result=run_full(tmp_path/'training',source,out,cfg,device='cpu',epochs=1,
                    test_root=test_root,evaluation='acdc')
    assert stages==['preflight','prepare','train','export_val','export_test','evaluate_val']
    assert result['status']=='COMPLETED_VAL_SCORED_OFFICIAL_TEST_UNSCORED_NO_SEMANTIC_SUPPORT'
    assert result['split_counts']=={'train':2,'val':2,'test':2}
    assert result['stages']['evaluate_test']=='NOT_SCORED_OFFICIAL_ACDC_TEST_GT_UNAVAILABLE'
    assert digest(source)==original_hash and read_splits(source)['test']==[]
    assert read_splits(out/'EFFECTIVE_SPLIT.json')==split_expected
    assert not (out/'evaluation_test').exists()
    assert (out/'PREDICTIONS_LOCKED.json').exists()
    with pytest.raises(ValueError,match='resume recipe'):
        run_full(tmp_path/'training',source,out,cfg,device='cpu',epochs=2,
                 test_root=test_root,evaluation='acdc',resume=True)


def test_official_test_duplicate_patient_unsafe_and_missing_root(tmp_path):
    from self_audit_guidance.full import external_acdc_patients
    with pytest.raises(FileNotFoundError):external_acdc_patients(tmp_path/'missing')
    source=tmp_path/'split.json';cfg=tmp_path/'cfg.json';testing=tmp_path/'testing'
    write_json(source,dict(train=['patient001'],val=['patient002']))
    write_json(cfg,GuidanceConfig().dictionary())
    testing.mkdir();d=testing/'patient001';d.mkdir();(d/'patient001_4d.nii.gz').write_bytes(b'fake')
    with pytest.raises(ValueError,match='leakage'):
        run_full(tmp_path/'training',source,tmp_path/'run',cfg,device='cpu',test_root=testing)
    assert not (tmp_path/'run').exists()


@requires_nib
def test_genuine_native_inventory_with_external_unlabeled_test(tmp_path,monkeypatch):
    import nibabel as nib
    base,_,_=native_fixture(tmp_path)
    testing=tmp_path/'testing';testing.mkdir()
    original=nib.load(base/'patient004/patient004_4d.nii.gz')
    path=testing/'patient101';path.mkdir()
    nib.save(original,path/'patient101_4d.nii.gz')
    p=tmp_path/'official.json';write_json(p,dict(train=['patient001','patient002'],val=['patient003'],test=['patient101']))
    real=nib.load;opened=[]
    def forbid_gt(path,*args,**kwargs):
        assert '_gt' not in str(path)
        opened.append(str(path))
        return real(path,*args,**kwargs)
    monkeypatch.setattr(nib,'load',forbid_gt)
    cfg=GuidanceConfig(grid=8,feature_dim=4,clusters=(2,),temporal=False)
    result=inventory(base,'acdc',p,cfg,batch_size=2,test_root=testing)
    assert result['split_counts']=={'train':2,'val':1,'test':1}
    assert len(opened)==4
    assert result['images']['patient101']['raw_image']==str(path/'patient101_4d.nii.gz')
