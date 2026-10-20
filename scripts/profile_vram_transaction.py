#!/usr/bin/env python3
"""Real full-transaction VRAM probe; fresh optimizer, no quality/resume claim.

Loads a pinned native-v2 checkpoint only to exercise realistic EMA candidates.
The actual new training runs still initialize from the original pretrained model.
"""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from self_audit.data import TrainingData
from self_audit.model import build_model,digest
from self_audit.engine import supervised_step,ema_update,source_digest
from self_audit.decoder_control import decoder_transaction
from self_audit.losses import strong_intensity
from torch.nn import functional as F

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--crop-size',type=int,default=0);p.add_argument('--batch',type=int,required=True);p.add_argument('--method',choices=['joint_guard','ema_full'],default='joint_guard');p.add_argument('--unlabeled',default='patient057');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():p.error('new output required')
root=Path('/workspace/self-audit-v2');manifest=root/'cache/L8_S42/train.json'
data=TrainingData(manifest,digest(manifest));rng=np.random.default_rng(42)
li=data.lids.index('patient026');ai=data.lids.index('patient011');ui=data.uids.index(a.unlabeled)
xs,ys,sm=data.sample_batch(li,rng,a.batch,labeled=True,crop_size=a.crop_size);xa,ya,am=data.sample_batch(ai,rng,a.batch,labeled=True,crop_size=a.crop_size);xu,um,uvalid=data.sample_batch(ui,rng,a.batch,labeled=False,crop_size=a.crop_size,return_valid=True)
checkpoint=root/'runs/L8_S42_supervised_B16_10000/step_001000'
cm=json.loads((checkpoint/'MANIFEST.json').read_text());assert digest(checkpoint/'weights.pt')==cm['payload_sha256']
torch.manual_seed(42);torch.set_num_threads(2)
model,provenance=build_model(json.loads((root/'BACKBONE.json').read_text()));teacher=copy.deepcopy(model)
payload=torch.load(checkpoint/'weights.pt',map_location='cpu',weights_only=True)
model.load_state_dict(payload['student'],strict=True);teacher.load_state_dict(payload['teacher'],strict=True);del payload
model=model.cuda();teacher=teacher.cuda().eval().requires_grad_(False)
body=[v for k,v in model.named_parameters() if not k.startswith('head.')]
opts=[torch.optim.AdamW(body,lr=3e-5,weight_decay=1e-4),torch.optim.SGD(model.head.parameters(),lr=.003)]
xs,ys,xa,ya,xu,uvalid=[v.cuda() for v in (xs,ys,xa,ya,xu,uvalid)];strong=strong_intensity(xu,rng)
result={'diagnostic_only':True,'fresh_optimizer_not_resume':True,'method':a.method,'batch_size':a.batch,'crop_size':a.crop_size,'supervised_patient':sm['patient'],'audit_patient':am['patient'],'unlabeled_patient':um['patient'],'U_shape':list(xu.shape),'L_shape':list(xs.shape),'audit_shape':list(xa.shape),'training_source_sha256':source_digest(),'checkpoint_manifest_sha256':digest(checkpoint/'MANIFEST.json'),'backbone':provenance,'status':'STARTED'}
torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
try:
 loss,gnorm=supervised_step(model,opts,xs,ys);model.eval()
 with torch.no_grad():q=teacher(xu).softmax(1)
 result.update(supervised_loss=loss,gradient_norm=gnorm,teacher_max_confidence=float(q.max()),teacher_argmax_classes=q.argmax(1).unique().cpu().tolist())
 if a.method=='joint_guard':
  proposal,control=decoder_transaction(model,strong,q,xa,ya,method='joint_guard',threshold=.7,step_size=.001,ridge=1e-6,budget=0.,backtracks=8,tolerance=1e-8,valid_mask=uvalid)
  result.update(proposal=proposal,control=control)
 else:
  conf,py=q.max(1);valid=(conf>=.7)&uvalid;result['accepted_pixels']=int(valid.sum())
  if valid.any():
   for opt in opts:opt.zero_grad(set_to_none=True)
   uloss=(F.cross_entropy(model(strong),py,reduction='none')*valid).sum()/valid.sum();uloss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
   for opt in opts:opt.step()
 ema_update(teacher,model,.99);torch.cuda.synchronize();result['status']='PASS'
except torch.cuda.OutOfMemoryError as e:result.update(status='OOM',error=str(e))
finally:
 result.update(seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),device_total_bytes=torch.cuda.get_device_properties(0).total_memory)
 a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
raise SystemExit(0 if result['status']=='PASS' else 2)
