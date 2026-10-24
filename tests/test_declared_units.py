"""Image metadata adjudication contracts, not patient accuracy evidence."""
import nibabel as nib
import numpy as np
import pytest
from scripts.declare_acdc_spatial_units import declared_copy
from self_audit.model import digest


def test_unit_copy_preserves_scaled_voxels_affine_and_source(tmp_path):
    source=tmp_path/'cine.nii'; dest=tmp_path/'derived.nii.gz'
    data=np.arange(120,dtype=np.int16).reshape(5,4,3,2)
    image=nib.Nifti1Image(data,np.diag([1.5,1.5,10,1]));image.header.set_xyzt_units('unknown','sec')
    image.header.set_slope_inter(2.0,7.0);nib.save(image,source)
    original=digest(source); receipt=declared_copy(source,dest,original)
    assert receipt['changed_uncompressed_header_bytes']==[123]
    assert digest(source)==original
    a,b=nib.load(source),nib.load(dest)
    assert np.array_equal(np.asarray(a.dataobj),np.asarray(b.dataobj))
    assert np.array_equal(a.affine,b.affine)
    assert b.header.get_xyzt_units()==('mm','sec')


def test_unit_copy_rejects_hash_or_known_units(tmp_path):
    source=tmp_path/'cine.nii';dest=tmp_path/'derived.nii.gz'
    image=nib.Nifti1Image(np.zeros((2,2,2,1),dtype=np.float32),np.eye(4))
    image.header.set_xyzt_units('mm');nib.save(image,source)
    with pytest.raises(ValueError,match='hash mismatch'):declared_copy(source,dest,'0'*64)
    with pytest.raises(ValueError,match='unknown'):declared_copy(source,dest,digest(source))
