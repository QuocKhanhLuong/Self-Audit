"""Scorer v2: exact frame-image bridge for ACDC affine inconsistencies.
Frozen predictions and probability/margin thresholds are unchanged.
"""
from evaluate import OUT,sha,write
from pathlib import Path
import json,re
import numpy as np
GRID=[]
def verify_grid(ref,gtnii,idx,image_record):
 import nibabel as nib
 frame=Path(str(ref).replace('_gt.nii','.nii'))
 cine=Path(image_record['path']);fh=sha(frame);ch=sha(cine)
 assert ch==image_record['sha256']
 im=nib.load(frame);ci=nib.load(cine)
 assert gtnii.shape==im.shape==ci.shape[:3]
 assert np.allclose(gtnii.affine,im.affine)
 assert np.array_equal(np.asarray(im.dataobj),np.asarray(ci.dataobj[...,idx]))
 assert sha(frame)==fh and sha(cine)==ch
 GRID.append(dict(reference=str(ref),frame=str(frame),frame_sha256=fh,cine_sha256=ch,reference_frame_affine_match=True,frame_cine_exact_voxel_equality=True,cine_reference_affine_match=bool(np.allclose(ci.affine,gtnii.affine))))
def score():
 import nibabel as nib
 from sklearn.metrics import roc_auc_score,average_precision_score
 r=json.loads((OUT/'RECIPE.json').read_text());f=json.loads((OUT/'PREDICTIONS_FROZEN.json').read_text());assert sha(OUT/'RECIPE.json')==f['recipe_sha256']
 rows=[];references={};hist=np.zeros(20,dtype=np.int64);conf={k:np.zeros((4,5),dtype=np.int64) for k in ('raw','gated')}
 for pid,rec in f['inventory'].items():
  path=OUT/'predictions'/ (pid+'.npy');assert sha(path)==rec['sha256'];allp=np.load(path,mmap_mode='r');d=Path(r['images'][pid]['path']).parent
  refs=sorted(d.glob(pid+'_frame*_gt.nii*'));assert len(refs)==2
  for ref in refs:
   before=sha(ref);gtnii=nib.load(ref);gt=np.asarray(gtnii.dataobj).transpose(2,1,0);assert np.isin(gt,[0,1,2,3,255]).all()
   idx=int(re.search(r'_frame(\d+)_gt',ref.name)[1])-1;verify_grid(ref,gtnii,idx,r['images'][pid]);p=np.asarray(allp[idx]);assert gt.shape==p.shape[:1]+p.shape[2:]
   raw=p.argmax(1);top=np.sort(p,axis=1);c=top[:,-1];margin=c-top[:,-2];accepted=(c>=r['min_probability'])&(margin>=r['min_margin'])&np.asarray(f['allowed_classes'])[raw];gated=np.where(accepted,raw,255)
   valid=gt!=255;fg=valid&(gt>0);hist+=np.histogram(c[valid],bins=np.linspace(0,1,21))[0]
   row=dict(patient=pid,frame=idx+1,coverage=float(accepted[valid].mean()),foreground_coverage=float(accepted[fg].mean()) if fg.any() else None,predicted_fg_fraction=float(((raw>0)&valid).sum()/valid.sum()),metrics={})
   for mode,lab in [('raw',raw),('gated',gated)]:
    col=np.where(lab==255,4,lab);cm=np.bincount((gt[valid]*5+col[valid]).astype(int),minlength=20).reshape(4,5);conf[mode]+=cm;row[mode+'_confusion']=cm.tolist()
   for k,name in enumerate(['BG','RV','MYO','LV']):
    target=gt==k;vals={}
    for mode,lab in [('raw',raw),('gated',gated)]:
     a=(lab==k)&valid;b=target&valid;den=a.sum()+b.sum();vals[mode+'_dice']=float(2*(a&b).sum()/den) if den else None
    a=(gated==k)&valid;b=target&valid&accepted;den=a.sum()+b.sum();vals['selective_dice']=float(2*(a&b).sum()/den) if den else None
    y=target[valid];s=p[:,k][valid];vals['auroc']=float(roc_auc_score(y,s)) if y.any() and not y.all() else None;vals['ap']=float(average_precision_score(y,s)) if y.any() else None
    row['metrics'][name]=vals
   assert sha(ref)==before;references[str(ref)]=before;rows.append(row)
 macro={name:{metric:float(np.mean([row['metrics'][name][metric] for row in rows if row['metrics'][name][metric] is not None])) for metric in rows[0]['metrics'][name] if any(row['metrics'][name][metric] is not None for row in rows)} for name in ['BG','RV','MYO','LV']}
 write(OUT/'METRICS.json',dict(scope=r['scope'],macro_per_frame=macro,confusion={k:v.tolist() for k,v in conf.items()},confidence_histogram=dict(edges=np.linspace(0,1,21).tolist(),counts=hist.tolist()),mean_foreground_coverage=float(np.mean([x['foreground_coverage'] for x in rows if x['foreground_coverage'] is not None])),mean_coverage=float(np.mean([x['coverage'] for x in rows])),frames=rows,references=references,provenance=r['provenance']))
 write(OUT/'STATUS.json',dict(status='VAL_SCORED_SOURCE_COMPARISON_BLOCKED',patients=len(f['inventory']),frames=len(rows),h06='NOT_RUN_SOURCE_NOT_QUALIFIED'))
 print(json.dumps(macro),flush=True)

if __name__=='__main__':
 score()
 write(OUT/'GEOMETRY_VERIFIED.json',GRID)
