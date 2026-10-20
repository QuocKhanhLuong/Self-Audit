import json
import numpy as np
import pytest
from self_audit.evaluation import confusion_matrix,class_metrics,evaluate
from self_audit.data import synthetic_fixture,write_json
from self_audit.model import digest
from self_audit.engine import train
from tests.test_pipeline import recipe


def test_unknowns_are_missed_foreground_and_empty_is_not_perfect():
    y=np.array([0,1,2,3]);p=np.array([255]*4)
    cm=confusion_matrix(y,p)
    assert np.all(cm[:,4]==1)
    assert all(r['dice']==0 for r in class_metrics(cm))
    absent=class_metrics(confusion_matrix(np.array([0,0]),np.array([2,2])))
    assert absent[2]['dice'] is None and absent[2]['FP']==2


def test_full_freeze_then_score(tmp_path):
    m,sha=synthetic_fixture(tmp_path/'train_data')
    report=train(recipe(),m,sha,tmp_path/'run',device='cpu',synthetic=True)
    em,_=synthetic_fixture(tmp_path/'eval_data',seed=70)
    obj=json.loads(em.read_text());records=obj['labeled']
    for j,r in enumerate(records):r['id']=f'unseen_{j}'
    ep=tmp_path/'eval_data/development.json'
    write_json(ep,{'schema':'self_audit.data.v2','role':'development','synthetic':True,'patients':records})
    result=evaluate(tmp_path/'run/final',report['final_manifest_sha256'],ep,digest(ep),tmp_path/'eval','cpu',.7)
    assert (tmp_path/'eval/PREDICTIONS_FROZEN.json').exists()
    assert result['synthetic'] and len(result['frames'])==3
    assert len(result['pooled_confusion'])==4


def test_mask_not_treated_as_background():
    cm=confusion_matrix(np.array([255,1]),np.array([2,1]))
    assert cm.sum()==1 and cm[1,1]==1
