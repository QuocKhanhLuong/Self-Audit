"""Replay actual hard-CE candidates; finite-guard/Dice counterexample, CPU only."""
from pathlib import Path
import sys,json,copy,hashlib,subprocess
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from self_audit.control import make_candidates,controlled_head_step
from self_audit.evaluation import confusion_matrix,class_metrics
from self_audit.engine import DEFAULT,source_digest

torch.set_num_threads(1)
base=Path(__file__).parents[1];ev=base/'evidence'
search=json.loads((ev/'F_search_result.json').read_text());raw=search['found']
def t(x,dtype=torch.float64):return torch.tensor(x,dtype=dtype)
def arr(x):return x.detach().cpu().numpy().tolist()
def head_from(w,b):
 h=torch.nn.Conv2d(w.shape[1],4,1,bias=True).to(dtype=w.dtype)
 with torch.no_grad():h.weight.copy_(w);h.bias.copy_(b)
 return h
# Remove only the two rejected U pixels: denominator, every direction, and logits stay identical.
w=t(raw['weight']);b=t(raw['bias']);uf=t(raw['unlabeled_features']);af=t(raw['audit_features']);labels=t(raw['labels'],torch.int64)
accepted=t(raw['teacher_probs']).max(1).values[0,0]>=raw['threshold']
variants=[('exact_accepted_U_only',w,b,uf[:,:,:,accepted],af,float(raw['step_size']))]
# One predefined simplification, no rounding-level search.
r=lambda x:torch.round(x*1000)/1000
rw,rb,ru,ra=r(w),r(b),r(uf[:,:,:,accepted]),r(af);reta=round(float(raw['step_size']),3)
variants.append(('rounded3_accepted_U_only',rw,rb,ru,ra,reta))
# Exact homogeneous-kernel lifting to reproduce the same logit path at default eta.
eta=float(DEFAULT['pseudo_lr']);ratio=reta/eta;k=ratio**.5;c=(ratio-1)**.5
wu=torch.cat([rw/k,torch.zeros((4,1,1,1),dtype=torch.float64)],1)
u=torch.cat([k*ru,torch.full((1,1,1,ru.shape[-1]),c,dtype=torch.float64)],1)
a=torch.cat([k*ra,torch.full((1,1,1,ra.shape[-1]),c,dtype=torch.float64)],1)
variants.append(('default_lr_lifted_rounded3',wu,rb,u,a,eta))
# Two fixed dtype stress checks; both retained, not a new search.
variants.append(('rounded3_float32',rw.float(),rb.float(),ru.float(),ra.float(),reta))
variants.append(('default_lr_lifted_float32',wu.float(),rb.float(),u.float(),a.float(),eta))
results={};effects={};full_ds=None
for name,vw,vb,vu,va,step in variants:
 model=head_from(vw,vb);teacher=copy.deepcopy(model).eval().requires_grad_(False)
 with torch.no_grad():q=teacher(vu).softmax(1);p0=model(va).softmax(1)
 ds,ids,proposal=make_candidates(model,vu,q,DEFAULT['pseudo_threshold'],step)
 assert ids==['pseudo_class_1','pseudo_class_2'] and proposal['accepted_pixels']==2 and len(set(ids))==2
 # Independently rederive ordinary hard-CE gradient with common accepted denominator.
 conf,py=q.max(1);valid=conf>=DEFAULT['pseudo_threshold']
 loss=(F.cross_entropy(model(vu),py,reduction='none')*valid).sum()/valid.sum()
 grads=torch.autograd.grad(loss,tuple(model.parameters()))
 aggregate=tuple(sum(d[j] for d in ds) for j in range(2))
 grad_error=max(float((aggregate[j]+step*grads[j]).abs().max()) for j in range(2))
 assert grad_error<(1e-11 if vw.dtype==torch.float64 else 1e-5)
 def metrics(h):
  pred=h(va).argmax(1).detach().numpy();cm=confusion_matrix(labels.numpy(),pred)
  return {'predictions':pred.tolist(),'rv':class_metrics(cm)[1],'classes':class_metrics(cm),'probabilities':arr(h(va).softmax(1))}
 before=metrics(model);joint=copy.deepcopy(model);scalar=copy.deepcopy(model);plain=copy.deepcopy(model)
 settings=dict(budget=DEFAULT['risk_budget'],tolerance=DEFAULT['guard_tolerance'],backtracks=DEFAULT['backtracks'])
 jr=controlled_head_step(joint,va,labels,ds,ids,method='joint_guard',ridge=DEFAULT['ridge'],**settings)
 # Exact alpha=a*ones restriction: canonical scalar specialization + same finite guard, ridge=m*lambda.
 sr=controlled_head_step(scalar,va,labels,[aggregate],['aggregate_same_ce'],method='independent_guard',ridge=len(ds)*DEFAULT['ridge'],**settings)
 pr=controlled_head_step(plain,va,labels,ds,ids,method='ema_head',ridge=DEFAULT['ridge'],**settings)
 jm,sm,pm=metrics(joint),metrics(scalar),metrics(plain)
 assert jr['scale']>0 and all(x<y-1e-8 for x,y in zip(jr['after'],jr['before']))
 assert jm['rv']['dice']<before['rv']['dice']
 assert jr['positive_pixels']==[[2],[1],[1]]
 # Retain all probabilities, source directions and candidate-generator validation.
 effects[name]=torch.stack([F.conv2d(va,dw,db) for dw,db in ds])
 results[name]={'dtype':str(vw.dtype),'feature_dim':vw.shape[1],'weight':arr(vw),'bias':arr(vb),'U_features':arr(vu),'audit_features':arr(va),'labels':arr(labels),'teacher_probs':arr(q),'max_abs_feature':float(max(vu.abs().max(),va.abs().max())),'step_size':step,'settings':{**settings,'threshold':DEFAULT['pseudo_threshold'],'ridge':DEFAULT['ridge']},'source_ids':ids,'proposal':proposal,'directions':[[arr(v) for v in d] for d in ds],'aggregate_gradient_max_error':grad_error,'before':before,'joint':{'report':jr,'metrics':jm},'aggregate_scalar':{'report':sr,'metrics':sm,'ridge':len(ds)*DEFAULT['ridge']},'plain':{'report':pr,'metrics':pm}}
# Raw-to-reduced U candidates are exactly preserved apart from roundoff.
full=head_from(w,b);ds_full,ids_full,_=make_candidates(full,uf,t(raw['teacher_probs']),raw['threshold'],raw['step_size'])
reduced=results['exact_accepted_U_only']['directions']
reduction_error=max(float((ds_full[j][k]-t(reduced[j][k])).abs().max()) for j in range(2) for k in range(2))
lift_error=float((effects['rounded3_accepted_U_only']-effects['default_lr_lifted_rounded3']).abs().max())
assert reduction_error<1e-12 and lift_error<1e-11
result={'kind':'ACTUAL_CE_GENERATOR_AND_ACTUAL_FINITE_GUARD_SYNTHETIC_CPU','parent_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'canonical_sha':'637f38ced217fe1db2347a87924695533056d8f0','source_sha256':source_digest(),'torch':torch.__version__,'numpy':np.__version__,'discovery_trials':search['actual_trials'],'postprocessing_variants':5,'postprocessing_selection':'all five reported; predetermined rounding3, exact removal/lifting, and two fixed float32 checks; no optimization/search','rejected_U_removal_max_candidate_error':reduction_error,'default_lr_lifting_max_audit_logit_effect_error':lift_error,'limitations':['Affine features/head/audit labels independently manufactured; not patient data or native backbone features.','Teacher q is produced by an identical finite affine teacher copy, not an observed EMA training trajectory.','First success in a selected bounded search is existence evidence only, not prevalence.','Default pseudo_lr lifting uses large finite features; no activation-distribution realism claim.'],'variants':results}
(ev/'F_realizable_replay.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':'PASS','variants':{name:{'eta':v['step_size'],'feature_dim':v['feature_dim'],'max_feature':v['max_abs_feature'],'before_dice':v['before']['rv']['dice'],'joint_dice':v['joint']['metrics']['rv']['dice'],'joint_before_risk':v['joint']['report']['before'],'joint_after_risk':v['joint']['report']['after'],'joint_alpha':v['joint']['report']['alpha'],'joint_scale':v['joint']['report']['scale'],'scalar_dice':v['aggregate_scalar']['metrics']['rv']['dice'],'scalar_alpha':v['aggregate_scalar']['report']['alpha'],'scalar_scale':v['aggregate_scalar']['report']['scale'],'plain_dice':v['plain']['metrics']['rv']['dice']} for name,v in results.items()},'reduction_error':reduction_error,'lift_error':lift_error},indent=2))
