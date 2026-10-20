import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from self_audit.data import TrainingData, synthetic_fixture, label_subset, write_json
from self_audit.model import build_model, digest
from self_audit.losses import supervised_loss
from self_audit.engine import DEFAULT, train, load_checkpoint, ema_update, validate_config
from self_audit.decoder_control import decoder_transaction


def recipe(method='joint_guard',scope='head'):
    c=copy.deepcopy(DEFAULT);c.update(method=method,update_scope=scope,steps=5,warmup_steps=2,
                                      pseudo_threshold=.25,ema_decay=.9,body_lr=.001,head_lr=.03,save_every=100)
    c['model']['backbone']='tiny_test'
    return c


def test_native_detail_receives_real_gradient():
    torch.manual_seed(0);model,_=build_model(recipe()['model'],synthetic=True)
    x=torch.rand(1,3,31,35); y=torch.randint(0,4,(1,31,35))
    out=model(x);assert out.shape==(1,4,31,35)
    supervised_loss(out,y).backward()
    assert model.detail[0].weight.grad.norm()>0 and model.encoder.stages[0].weight.grad.norm()>0


def test_ema_exactly_matches_declared_mix():
    model,_=build_model(recipe()['model'],synthetic=True); teacher=copy.deepcopy(model)
    old={n:p.detach().clone() for n,p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():p.add_(.1)
    ema_update(teacher,model,.9)
    for n,p in teacher.named_parameters():torch.testing.assert_close(p,old[n]+.01)


def test_budget_nested_and_deterministic():
    ids=[f'patient{i:03d}' for i in range(1,81)]
    assert set(label_subset(ids,4,42))<set(label_subset(ids,8,42))<set(label_subset(ids,16,42))
    assert label_subset(ids,8,42)==label_subset(ids[::-1],8,42)


def test_no_u_masks_and_overlap(tmp_path):
    m,sha=synthetic_fixture(tmp_path/'data')
    data=TrainingData(m,sha,synthetic=True)
    for i in range(3):data.sample_unlabeled(i,np.random.default_rng(0))
    assert data.label_reads==set()
    obj=json.loads(m.read_text()); obj['unlabeled'][0]['mask']='hidden.npy';write_json(m,obj)
    with pytest.raises(ValueError,match='image only'):TrainingData(m,digest(m),synthetic=True)
    del obj['unlabeled'][0]['mask'];obj['evaluation_ids']=[obj['labeled'][0]['id']];write_json(m,obj)
    with pytest.raises(ValueError,match='overlap'):TrainingData(m,digest(m),synthetic=True)


def test_pretrained_fails_closed():
    with pytest.raises(ValueError,match='prohibited'):build_model(recipe()['model'],synthetic=False)
    with pytest.raises(ValueError):validate_config({'schema':'historical_Q'})


@pytest.mark.parametrize('method',['supervised','supervised_extra','ema_full','ema_block','sign',
                                   'independent_guard','joint_no_class','joint_no_guard','joint_guard'])
def test_end_to_end_each_arm(tmp_path,method):
    m,sha=synthetic_fixture(tmp_path/'data');out=tmp_path/'run'
    report=train(recipe(method),m,sha,out,device='cpu',synthetic=True)
    assert report['ema_updates']==5 and report['quality']=='NOT_EVALUATED'
    assert report['label_reads_outside_budget']==[]
    rows=[json.loads(x) for x in (out/'steps.jsonl').read_text().splitlines()]
    assert all(r['supervised']['patient']!=r['audit']['patient'] for r in rows)
    assert len(report['seen_supervised'])==3 and len(report['seen_unlabeled'])==3
    model,meta=load_checkpoint(out/'final',report['final_manifest_sha256'])
    assert meta['prediction_model']=='student'
    assert torch.isfinite(model(torch.rand(1,3,32,32))).all()
    for r in rows:
        if r.get('control',{}).get('guarded'):
            assert np.all(np.array(r['control']['after'])<=np.array(r['control']['before'])+1e-8)


def test_decoder_jvp_transaction(tmp_path):
    m,sha=synthetic_fixture(tmp_path/'data')
    report=train(recipe('joint_guard','decoder'),m,sha,tmp_path/'run',device='cpu',synthetic=True)
    rows=[json.loads(s) for s in (tmp_path/'run/steps.jsonl').read_text().splitlines()]
    assert any(r.get('control',{}).get('jvp_count',0)>0 for r in rows)
    for r in rows:
        if r.get('control',{}).get('guarded'):
            assert np.all(np.array(r['control']['after'])<=np.array(r['control']['before'])+1e-8)


def test_rng_and_sampling_match_across_arms(tmp_path):
    m,sha=synthetic_fixture(tmp_path/'data')
    for method in ['ema_block','joint_guard']:
        train(recipe(method),m,sha,tmp_path/method,device='cpu',synthetic=True)
    rows=[[json.loads(s) for s in (tmp_path/v/'steps.jsonl').read_text().splitlines()] for v in ['ema_block','joint_guard']]
    for x,y in zip(*rows):
        for key in ['supervised','audit','unlabeled']:assert x[key]==y[key]


def test_annotation_leak_fails(tmp_path):
    m,sha=synthetic_fixture(tmp_path/'data');obj=json.loads(m.read_text())
    obj['label_budget']=2;write_json(m,obj)
    with pytest.raises(ValueError,match='budget'):TrainingData(m,digest(m),synthetic=True)


def test_native_patient_batches_do_not_read_u_labels(tmp_path):
    m, sha = synthetic_fixture(tmp_path/'data')
    data = TrainingData(m, sha, synthetic=True)
    x, meta = data.sample_batch(0, np.random.default_rng(1), 4, labeled=False)
    assert x.shape == (4, 3, 32, 32) and data.label_reads == set()
    assert len(meta['samples']) == 4
    assert {s['patient'] for s in meta['samples']} == {data.uids[0]}
    x, y, meta = data.sample_batch(1, np.random.default_rng(1), 4, labeled=True)
    assert x.shape == (4, 3, 32, 32) and y.shape == (4, 32, 32)
    assert data.label_reads == {data.lids[1]}
    old_rng, new_rng = np.random.default_rng(8), np.random.default_rng(8)
    old = data.sample_labeled(0, old_rng)
    new = data.sample_batch(0, new_rng, 1, labeled=True)
    torch.testing.assert_close(old[0], new[0]); torch.testing.assert_close(old[1], new[1])
    assert old[2] == new[2] and old_rng.bit_generator.state == new_rng.bit_generator.state


@pytest.mark.parametrize('method', ['supervised', 'supervised_extra', 'ema_full', 'ema_block',
                                   'sign', 'independent_guard', 'joint_no_class', 'joint_no_guard', 'joint_guard'])
def test_matched_native_batch_transactions(tmp_path, method):
    m, sha = synthetic_fixture(tmp_path/'data')
    c = recipe(method, 'decoder'); c.update(batch_size=2, crop_size=40, steps=2, warmup_steps=0)
    report = train(c, m, sha, tmp_path/'run', device='cpu', synthetic=True)
    assert report['ema_updates'] == 2 and report['label_reads_outside_budget'] == []
    rows = [json.loads(s) for s in (tmp_path/'run/steps.jsonl').read_text().splitlines()]
    for row in rows:
        assert len(row['audit']['samples']) == 2
        assert row['crop_size'] == 40 and row['unlabeled_valid_pixels'] == 2048
        if 'proposal' in row:
            assert row['proposal']['accepted_pixels'] <= 2048
        if 'accepted_pixels' in row:
            assert row['accepted_pixels'] <= 2048
        assert row['supervised']['patient'] != row['audit']['patient']
        if row.get('control', {}).get('guarded'):
            assert np.all(np.array(row['control']['after']) <= np.array(row['control']['before']) + 1e-8)
    control = tmp_path/'reference'
    c['method'] = 'supervised'
    train(c, m, sha, control, device='cpu', synthetic=True)
    reference = [json.loads(s) for s in (control/'steps.jsonl').read_text().splitlines()]
    for row, ref in zip(rows, reference):
        for role in ['supervised', 'audit', 'unlabeled']:
            assert row[role] == ref[role]


def test_legacy_batch_recipe_and_invalid_batch_sizes():
    c = recipe(); del c['batch_size']
    assert validate_config(c)['batch_size'] == 1
    for invalid in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match='batch_size'):
            validate_config({**c, 'batch_size': invalid})


def test_crop_alignment_geometry_only_and_padding_ignore():
    from self_audit.data import crop_native
    y=(torch.arange(35).reshape(1,5,7)%4).long()
    x=(y[:,None].float()/3).repeat(1,3,1,1)
    a,labels,valid,meta=crop_native(x,y,np.random.default_rng(7),9)
    assert a.shape==(1,3,9,9) and labels.shape==valid.shape==(1,9,9)
    assert int(valid.sum())==35 and (labels[~valid]==255).all()
    torch.testing.assert_close(a[:,1][valid]*3,labels[valid].float())
    b,other,valid2,meta2=crop_native(x,torch.zeros_like(y),np.random.default_rng(7),9)
    assert meta==meta2 and torch.equal(a,b) and torch.equal(valid,valid2)
    xlarge=torch.rand(1,3,40,50);yl=torch.randint(0,4,(1,40,50))
    c,cl,cv,cm=crop_native(xlarge,yl,np.random.default_rng(4),16)
    assert torch.equal(c,xlarge[...,cm['top']:cm['top']+16,cm['left']:cm['left']+16])
    assert cv.all()
    config=recipe();config.pop('crop_size')
    assert validate_config(config)['crop_size']==0
    for value in [-1,True,2.5]:
        with pytest.raises(ValueError,match='crop_size'):
            validate_config({**config,'crop_size':value})
