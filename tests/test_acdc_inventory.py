"""Manufactured inventory contract; no patient efficacy evidence."""
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from scripts.verify_acdc_download import verify


def test_duplicate_cannot_cross_active_roles_and_reviewed_exclusions_are_explicit(tmp_path):
    root=tmp_path/'raw'
    for i in range(1,101):
        pid=f'patient{i:03d}';folder=root/'training'/pid;folder.mkdir(parents=True)
        (folder/'Info.cfg').write_text('not consumed by inventory')
        a=np.full((2,3,1,2),74 if i==76 else i,dtype=np.int16)
        nib.save(nib.Nifti1Image(a,np.eye(4)),folder/(pid+'_4d.nii'))
        for t in range(2):nib.save(nib.Nifti1Image(a[...,t],np.eye(4)),folder/(pid+f'_frame{t+1:02d}.nii'))
    for i in range(101,151):(root/'testing'/f'patient{i:03d}').mkdir(parents=True)
    repo=Path(__file__).resolve().parents[1]
    original=verify(root,repo/'splits/acdc_patient_split_seed42.json')
    assert original['status']=='FAIL' and any('duplicate cine voxels' in s for s in original['errors'])
    reviewed=repo/'splits/acdc_reviewed_locked_train80_val18.json'
    result=verify(root,reviewed)
    assert result['status']=='PASS' and result['mask_arrays_opened']==0
    assert result['quarantined_duplicates']
    assert next(r for r in result['patients'] if r['patient']=='patient076')['role']=='QUARANTINED'
    # Unexplained missing IDs do not become implicit exclusions.
    o=json.loads(reviewed.read_text());del o['excluded_patients']
    bad=tmp_path/'bad.json';bad.write_text(json.dumps(o))
    assert verify(root,bad)['status']=='FAIL'
