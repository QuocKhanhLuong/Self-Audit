import json
import nibabel as nib
import numpy as np
import pytest
from self_audit.compact_artifacts import digest
from self_audit.compact_geometry import attach_geometry


def geometry_fixture(tmp_path):
    raw=np.random.default_rng(45).normal(size=(7,9,2,3)).astype(np.float32)
    obj=nib.Nifti1Image(raw,np.diag([1.3,1.6,8.,1.]))
    obj.header.set_xyzt_units('mm','sec')
    src=tmp_path/'cine.nii.gz'
    nib.save(obj,src)
    lo,hi=np.percentile(raw,[.5,99.5])
    prepared=np.clip((raw-lo)/(hi-lo),0,1).transpose(3,2,1,0).astype(np.float32)
    image=tmp_path/'cine.npy'
    np.save(image,prepared)
    m=tmp_path/'eval.json'
    m.write_text(json.dumps({'schema':'self_audit.data.v2','role':'development','synthetic':True,
                            'patients':[{'id':'p','image':image.name,'image_sha256':digest(image),
                                         'masks':{'0':{'path':'DOES_NOT_EXIST.npy','sha256':'a'*64}}}]}))
    s=tmp_path/'sources.json'
    s.write_text(json.dumps({'schema':'self_audit.compact_geometry_sources.v1',
                            'patients':{'p':{'path':src.name,'sha256':digest(src)}}}))
    return m,s,image


def test_geometry_attaches_without_opening_reference_masks(tmp_path):
    m,s,image=geometry_fixture(tmp_path)
    old=digest(m)
    result=attach_geometry(m,old,s,digest(s),tmp_path/'derived'/'eval.json')
    assert result['reference_masks_opened'] is False and digest(m)==old
    derived=json.loads((tmp_path/'derived'/'eval.json').read_text())
    assert derived['patients'][0]['image_sha256']==digest(image)
    np.testing.assert_allclose(derived['patients'][0]['affine'],np.diag([1.3,1.6,8.,1.]),rtol=1e-6)


def test_geometry_requires_voxel_correspondence_not_shape_alone(tmp_path):
    m,s,image=geometry_fixture(tmp_path)
    x=np.load(image)
    np.save(image,1-x)
    manifest=json.loads(m.read_text())
    manifest['patients'][0]['image_sha256']=digest(image)
    m.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='voxel correspondence'):
        attach_geometry(m,digest(m),s,digest(s),tmp_path/'out.json')
