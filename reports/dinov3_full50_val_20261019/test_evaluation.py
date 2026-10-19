"""Metric contract checks using manufactured arrays, not patient evidence."""
import importlib.util,json
from pathlib import Path
import numpy as np
import nibabel as nib
spec=importlib.util.spec_from_file_location('evaluate',Path(__file__).with_name('evaluate.py'))
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
def test_unknown_is_miss_and_auroc_uses_class_score(tmp_path):
 e.OUT=tmp_path
 (tmp_path/'predictions').mkdir()
 # All four classes present, correct argmax but every probability below gate.
 gt=np.array([0,1,2,3],dtype=np.uint8).reshape(2,2,1)
 p=np.full((2,1,4,2,2),.15,dtype=np.float32)
 for x in range(2):
  for y in range(2):p[:,:,gt[x,y,0],y,x]=.55
 np.save(tmp_path/'predictions/patient001.npy',p)
 for f in [1,2]:nib.save(nib.Nifti1Image(gt,np.eye(4)),tmp_path/f'patient001_frame{f:02}_gt.nii')
 r=dict(min_probability=.7,min_margin=.2,scope='synthetic only',provenance='synthetic only',images={'patient001':{'path':str(tmp_path/'patient001_4d.nii')}})
 e.write(tmp_path/'RECIPE.json',r)
 e.write(tmp_path/'PREDICTIONS_FROZEN.json',dict(recipe_sha256=e.sha(tmp_path/'RECIPE.json'),allowed_classes=[True]*4,inventory={'patient001':dict(sha256=e.sha(tmp_path/'predictions/patient001.npy'),image={'affine':np.eye(4).tolist()})}))
 e.score();m=json.loads((tmp_path/'METRICS.json').read_text())
 assert m['mean_foreground_coverage']==0
 for v in m['macro_per_frame'].values():
  assert v['raw_dice']==1 and v['gated_dice']==0
  assert v['auroc']==1 and v['ap']==1
 assert np.array(m['confusion']['gated'])[:,4].tolist()==[2]*4
 assert sum(m['confidence_histogram']['counts'])==8

def test_frame_bridge_accepts_affine_difference_but_rejects_voxel_mismatch(tmp_path):
 import sys,pytest
 sys.path.insert(0,str(Path(__file__).parent))
 import score_v2
 data=np.arange(8,dtype=np.float32).reshape(2,2,2)
 affine=np.eye(4);affine[0,3]=100
 cine=tmp_path/'patient001_4d.nii';frame=tmp_path/'patient001_frame01.nii';gt=tmp_path/'patient001_frame01_gt.nii'
 nib.save(nib.Nifti1Image(np.stack([data,data],axis=-1),affine),cine)
 nib.save(nib.Nifti1Image(data,np.eye(4)),frame)
 nib.save(nib.Nifti1Image(np.zeros_like(data),np.eye(4)),gt)
 rec=dict(path=str(cine),sha256=e.sha(cine))
 score_v2.verify_grid(gt,nib.load(gt),0,rec)
 nib.save(nib.Nifti1Image(data+1,np.eye(4)),frame)
 with pytest.raises(AssertionError):score_v2.verify_grid(gt,nib.load(gt),0,rec)
