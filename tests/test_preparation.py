import json,sys,types
from pathlib import Path
import numpy as np
import pytest
from prepare_acdc import read_patient_split,prepare


def test_existing_split_variants_agree(tmp_path):
    p=tmp_path/'split.json';p.write_text(json.dumps({'train_patients':['patient001','patient002','patient003'],
          'val_patients':['patient004'],'splits':{'train':{'patients':['patient003','patient001','patient002']}}}))
    assert read_patient_split(p)['test']==[]
    x=json.loads(p.read_text());x['splits']['train']['patients']=['patient008'];p.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='conflicting'):read_patient_split(p)


def test_preparation_opens_only_budgeted_masks_with_mock_reader(tmp_path,monkeypatch):
    root=tmp_path/'raw';root.mkdir();registry={};reads=[]
    class Image:
        def __init__(self,a):self.dataobj=a;self.affine=np.eye(4);self.shape=a.shape
    rng=np.random.default_rng(2)
    for i in range(1,5):
        pid=f'patient{i:03d}';d=root/pid;d.mkdir();a=rng.integers(0,100,(5,6,2,2),dtype=np.int16)
        for name,value in [(f'{pid}_4d.nii',a),(f'{pid}_frame01.nii',a[...,0]),
                           (f'{pid}_frame01_gt.nii',rng.integers(0,4,(5,6,2),dtype=np.int16))]:
            p=d/name;p.write_bytes(value.tobytes());registry[str(p)]=Image(value)
    def load(path):reads.append(str(path));return registry[str(path)]
    monkeypatch.setitem(sys.modules,'nibabel',types.SimpleNamespace(load=load))
    split=tmp_path/'split.json';split.write_text(json.dumps({'train':['patient001','patient002','patient003'],
                                                          'val':['patient004'],'test':[]}))
    out=tmp_path/'prepared';prepare(root,split,out,2,42,False)
    m=json.loads((out/'train.json').read_text());lids={r['id'] for r in m['labeled']}
    mask_ids={Path(p).parent.name for p in reads if '_gt' in p}
    assert mask_ids==lids and len(lids)==2
    assert not any('patient004' in p for p in reads)
    assert all(set(r)=={'id','image','image_sha256'} for r in m['unlabeled'])
