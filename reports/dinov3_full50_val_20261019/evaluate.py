"""Frozen, image-only full-cine prediction followed by separate development scoring."""
from pathlib import Path
import hashlib,json,sys,os,subprocess,time,re
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments/layer_guided_student_v1'),str(ROOT/'src')]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def gpu_free():
 rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 return not any(UUID in row for row in rows.splitlines())
UUID=''
def prepare():
 run=ROOT/'runs/astra_dinov3_c4973553_4070_20261016_214836'
 ck=run/'full50/final';m=json.loads((ck/'MANIFEST.json').read_text());rep=json.loads((run/'full50/REPORT.json').read_text())
 assert sha(ck/'MANIFEST.json')==rep['final_manifest_sha256']
 assert sha(ck/'model.pt')==m['payload_sha256']
 from training_v1.config import source_hashes
 assert source_hashes()==m['metadata']['source_hashes']
 from self_audit_guidance.io import discover
 images=discover(ROOT/'data/ACDC/training','acdc',m['metadata']['lineage']['locked_splits']['val'])
 approval=run/'receipts/BACKBONE_APPROVAL.json'
 assert sha(approval)=='0c5d3301c86d8a6d3732be96bed1328e5debfa778eb1c51c7c5ebfbd27efb94d'
 recipe=dict(schema='dinov3.frozen_development_eval.v1',checkpoint=str(ck),manifest_sha256=sha(ck/'MANIFEST.json'),payload_sha256=sha(ck/'model.pt'),approval=str(approval),approval_sha256=sha(approval),script_sha256=sha(__file__),min_probability=.7,min_margin=.2,batch_size=1,precision='float32_no_autocast',input='full_cine_read_cine_then_normalize_cine',gpu='RTX 4070 only; wait for no compute processes',unknown=255,images={pid:dict(path=str(p),sha256=sha(p)) for pid,p in images.items()},metrics='per-frame class Dice, macro over frames; gated full-domain UNKNOWN counts as FN; additional selective Dice; one-v-rest pixel AUROC/AP per frame; confusion rows GT BG/RV/MYO/LV columns BG/RV/MYO/LV/UNKNOWN; foreground coverage on GT foreground',scope='exposed VAL development; not unseen confirmation',provenance='Mirror weights hash-pinned and runtime-verified; official direct download returned 403; upstream full-digest lineage not independently verified; no full Meta certification.',teacher_comparison='No matched VAL R0/Q cache located. Do not substitute another bank or regenerate with GT.',h06='BLOCKED_PENDING_SOURCE_QUALIFICATION')
 write(OUT/'RECIPE.json',recipe)
 print('Frozen recipe',sha(OUT/'RECIPE.json'),flush=True)
def predict():
 global UUID
 import torch
 from self_audit_guidance.io import read_cine
 from integration.geometry import normalize_cine,slice_contexts
 from training_v1.checkpoint import load_checkpoint
 r=json.loads((OUT/'RECIPE.json').read_text());assert sha(__file__)==r['script_sha256']
 devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)
 matches=[line.split(',')[0].strip() for line in devices.splitlines() if line.split(',')[1].strip()=='NVIDIA GeForce RTX 4070'];assert len(matches)==1
 UUID=matches[0]
 while not gpu_free():
  write(OUT/'STATUS.json',dict(status='WAITING_FOR_4070_NO_COMPUTE',time=time.time()));time.sleep(30)
 os.environ['CUDA_VISIBLE_DEVICES']=UUID
 assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 4070'
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.85,0)
 assert sha(r['approval'])==r['approval_sha256']
 model,meta=load_checkpoint(r['checkpoint'],r['manifest_sha256'],json.loads(Path(r['approval']).read_text()),device='cuda')
 allowed=torch.tensor(meta['lineage']['allowed_classes'],device='cuda')
 pred=OUT/'predictions';pred.mkdir(exist_ok=False);inventory={};diag={}
 for pid,record in r['images'].items():
  assert sha(record['path'])==record['sha256'];cine,imeta=read_cine(record['path']);x,_=normalize_cine(torch.from_numpy(cine),record['sha256']);t,z,h,w=x.shape
  probs=np.lib.format.open_memmap(pred/(pid+'.npy'),mode='w+',dtype='float32',shape=(t,z,4,h,w));maxdiff=0.;detail=0.
  with torch.inference_mode():
   for ti in range(t):
    for zi in range(z):
     o=model(slice_contexts(x,[(ti,zi)]).cuda());p=o['logits'].softmax(1)
     lifted=o['coarse_logits'].repeat_interleave(4,-2).repeat_interleave(4,-1)[...,:h,:w]
     maxdiff=max(maxdiff,float((o['logits']-lifted).abs().max()));detail=max(detail,float(o['raw_detail'].abs().max()))
     probs[ti,zi]=p[0].cpu().numpy()
  probs.flush();del probs
  inventory[pid]=dict(sha256=sha(pred/(pid+'.npy')),image=imeta)
  diag[pid]=dict(native_minus_lifted_coarse_max_abs=maxdiff,raw_detail_max_abs=detail)
  write(OUT/'STATUS.json',dict(status='PREDICTING',completed=list(inventory),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
  print('predicted',pid,diag[pid],flush=True)
 write(OUT/'PREDICTIONS_FROZEN.json',dict(recipe_sha256=sha(OUT/'RECIPE.json'),inventory=inventory,diagnostics=diag,allowed_classes=meta['lineage']['allowed_classes'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
def score():
 import nibabel as nib
 from sklearn.metrics import roc_auc_score,average_precision_score
 r=json.loads((OUT/'RECIPE.json').read_text());f=json.loads((OUT/'PREDICTIONS_FROZEN.json').read_text());assert sha(OUT/'RECIPE.json')==f['recipe_sha256']
 rows=[];references={};hist=np.zeros(20,dtype=np.int64);conf={k:np.zeros((4,5),dtype=np.int64) for k in ('raw','gated')}
 for pid,rec in f['inventory'].items():
  path=OUT/'predictions'/ (pid+'.npy');assert sha(path)==rec['sha256'];allp=np.load(path,mmap_mode='r');d=Path(r['images'][pid]['path']).parent
  refs=sorted(d.glob(pid+'_frame*_gt.nii*'));assert len(refs)==2
  for ref in refs:
   before=sha(ref);gtnii=nib.load(ref);gt=np.asarray(gtnii.dataobj).transpose(2,1,0);assert np.isin(gt,[0,1,2,3,255]).all();assert np.allclose(gtnii.affine,rec['image']['affine'])
   idx=int(re.search(r'_frame(\d+)_gt',ref.name)[1])-1;p=np.asarray(allp[idx]);assert gt.shape==p.shape[:1]+p.shape[2:]
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
 {'prepare':prepare,'predict':predict,'score':score}[sys.argv[1]]()
