"""Independent integration regressions for cached-teacher provenance."""
import copy
import json
import numpy as np
import pytest
import torch
from self_audit.compact_data import CompactTrainingData, synthetic_compact_fixture
from self_audit.compact_teacher import FixedTeacherCache, build_teacher_cache
from self_audit.model import digest


def cache_fixture(tmp_path):
    path, sha = synthetic_compact_fixture(tmp_path/'data', n_labeled=1,n_unlabeled=1,size=16,num_slices=2)
    data=CompactTrainingData(path,sha,synthetic=True)
    identity={'teacher_kind':'synthetic','inference_state':'student','label_budget':1,
              'labeled_ids':data.lids,'manifest_sha256':'a'*64,'payload_sha256':'b'*64}
    model=torch.nn.Conv2d(3,4,1)
    receipt=build_teacher_cache(model,data,tmp_path/'cache',identity,batch_size=2)
    return data,identity,receipt


def test_cache_hash_is_not_confused_with_teacher_checkpoint_hash(tmp_path):
    data,identity,r=cache_fixture(tmp_path)
    assert r['cache_manifest_sha256'] != identity['manifest_sha256']
    with FixedTeacherCache(r['cache_manifest_path'],r['cache_manifest_sha256'],data,identity) as cache:
        probabilities,valid=cache.sample_batch({'patient':data.uids[0],'frame':0,'slice':0})
    assert probabilities.shape==(1,4,16,16) and valid.all()


def test_configured_teacher_identity_cannot_be_ignored(tmp_path):
    data,identity,r=cache_fixture(tmp_path)
    identity['inference_state']='teacher_ema'
    with pytest.raises(ValueError,match='differs from the configured teacher'):
        FixedTeacherCache(r['cache_manifest_path'],r['cache_manifest_sha256'],data,identity)


def test_real_cache_generation_requires_a_physically_verified_model(tmp_path):
    data,identity,r=cache_fixture(tmp_path)
    data.manifest['synthetic']=False  # Negative fixture; this is not a real patient.
    identity.update(teacher_kind='canonical',source_sha256='c'*64,training_method='ema_full',
                    quality_receipt={'path':'missing','sha256':'d'*64})
    with pytest.raises(ValueError,match='physically verified teacher loader'):
        build_teacher_cache(torch.nn.Conv2d(3,4,1),data,tmp_path/'rejected',identity)
    assert not (tmp_path/'rejected').exists()


def test_cache_claimed_shape_cannot_hide_wrong_payload_geometry(tmp_path):
    data,identity,r=cache_fixture(tmp_path)
    manifest_path=tmp_path/'cache'/'CACHE_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text())
    entry=manifest['patients'][data.uids[0]]
    array_path=manifest_path.parent/entry['file']
    np.save(array_path,np.ones((2,2,4,1,1),dtype=np.float32)/4)
    entry['file_sha256']=digest(array_path)  # Claimed geometry remains original.
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='payload geometry/dtype'):
        FixedTeacherCache(manifest_path,digest(manifest_path),data,identity)
