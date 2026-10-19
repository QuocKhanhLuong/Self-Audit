from pathlib import Path
import json,hashlib,numpy as np
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
bank=root/'runs/stable_guidance_main513ddda_acdc_4070_20261015_210312/run/bank'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
m=json.loads((bank/'FROZEN.json').read_text());assert sha(bank/'FROZEN.json')=='f5a7b51328cc877ebdc4e70cbf7f90bebfe6da672597082beb5ae270f3e98665'
rows=[]
for pid in m['patients']:
 p=bank/(pid+'.npz');assert sha(p)==m['files'][p.name]
 with np.load(p,allow_pickle=False) as f:
  q=f['q'];r=f['r0'];lab=f['labels'];rel=f['reliability'];assert np.isfinite(q).all() and np.isfinite(r).all()
  mask=r.sum(-1)>0
  rows.append(dict(patient=pid,cells=int(lab.size),r0_supported=int(mask.sum()),r0_argmax_on_support=np.bincount(r.argmax(-1)[mask],minlength=4).tolist(),r0_class_mass=r.sum(axis=(0,1,2,3),dtype=np.float64).tolist(),q_class_mass=q.sum(axis=(0,1,2,3),dtype=np.float64).tolist(),q_argmax=np.bincount(q.argmax(-1).ravel(),minlength=4).tolist(),gated_class_counts=[int((lab==k).sum()) for k in range(4)],unknown=int((lab==255).sum()),mean_reliability=float(rel.mean())))
 print('audited',pid,flush=True)
g=json.loads((out/'SOURCE_GATE.json').read_text());g['evidence']='All 80 TRAIN NPZ payload hashes verified; read only R0, Q, reliability, labels; no patient reference read.';g['patients_diagnostics']=rows
g['totals']={key:np.array([r[key] for r in rows]).sum(axis=0).tolist() for key in ['cells','r0_supported','r0_argmax_on_support','r0_class_mass','q_class_mass','q_argmax','gated_class_counts','unknown']}
(out/'SOURCE_GATE.json').write_text(json.dumps(g,indent=2)+'\n')
