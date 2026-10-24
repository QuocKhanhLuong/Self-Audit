"""Storage/parsing and mask-access parity; manufactured data are not efficacy."""
import copy
import gzip
import json
import os
from pathlib import Path
import numpy as np
import pytest

from self_audit.data import synthetic_fixture, write_json
from self_audit.engine import DEFAULT, train
from self_audit.evaluation import evaluate
from self_audit.model import digest
from scripts.evaluate_frozen_labels import evaluate_labels


@pytest.fixture
def frozen_inputs(tmp_path):
    manifest,sha=synthetic_fixture(tmp_path/'train')
    c=copy.deepcopy(DEFAULT);c.update(method='supervised',steps=1,warmup_steps=1,save_every=10)
    c['model']['backbone']='tiny_test'
    report=train(c,manifest,sha,tmp_path/'run',device='cpu',synthetic=True)
    ep,_=synthetic_fixture(tmp_path/'eval_data',seed=17)
    records=json.loads(ep.read_text())['labeled']
    for r in records:
        ip=ep.parent/r['image'];x=np.load(ip);np.save(ip,np.tile(x,(2,2,1,1)))
        r['id']='evaluation_'+r['id'];r['image_sha256']=digest(ip)
        mask=r['masks']['0'];mp=ep.parent/mask['path'];np.save(mp,np.tile(np.load(mp),(2,1,1)))
        mask['sha256']=digest(mp);r['masks']={'0':mask,'1':dict(mask)}
    write_json(ep,dict(schema='self_audit.data.v2',role='external_frozen',synthetic=True,patients=records))
    return tmp_path/'run/final',report['final_manifest_sha256'],ep


@pytest.mark.parametrize('compress',[False,True])
def test_labels_match_probability_evaluator_and_freeze_before_masks(tmp_path,frozen_inputs,monkeypatch,compress):
    cp,sha,ep=frozen_inputs
    dense=evaluate(cp,sha,ep,digest(ep),tmp_path/'dense','cpu')
    mask_paths={str((ep.parent/m['path']).resolve()) for r in json.loads(ep.read_text())['patients'] for m in r['masks'].values()}
    original=np.load;seen=[];out=tmp_path/'labels'
    def checked_load(path,*args,**kwargs):
        if isinstance(path,(str,os.PathLike)) and str(Path(path).resolve()) in mask_paths:
            assert (out/'PREDICTIONS_FROZEN.json').exists();seen.append(str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',checked_load)
    compact=evaluate_labels(cp,sha,ep,digest(ep),out,'cpu',batch_size=2,compress_predictions=compress)
    assert seen
    assert compact['frames']==dense['frames']
    assert compact['macro_patient']==dense['macro_patient']
    assert compact['patients']==dense['patients']
    for record in json.loads(ep.read_text())['patients']:
        pid=record['id'];probs=np.load(tmp_path/'dense/predictions'/f'{pid}.npy')
        if compress:
            with gzip.open(out/'predictions'/f'{pid}.npy.gz','rb') as stream:labels=np.load(stream,allow_pickle=False)
        else:labels=np.load(out/'predictions'/f'{pid}.npy')
        assert labels.dtype==np.uint8 and np.array_equal(labels,probs.argmax(2))
    assert compact['all_class_patient_macro'] is not None


def test_labels_reject_train_patient_id(tmp_path,frozen_inputs):
    cp,sha,ep=frozen_inputs
    m=json.loads(ep.read_text());m['patients'][0]['id']='synthetic_000';write_json(ep,m)
    with pytest.raises(ValueError,match='overlap'):
        evaluate_labels(cp,sha,ep,digest(ep),tmp_path/'forbidden','cpu')
