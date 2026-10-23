"""Bounded first-success CE-generator search; synthetic CPU only."""
from pathlib import Path
import sys,json,copy,hashlib,subprocess,time
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from self_audit.control import make_candidates,controlled_head_step,audit_weights,risks,response_moments,scalar_allocation
from self_audit.evaluation import confusion_matrix,class_metrics

torch.set_num_threads(1)
SEED=20261023
MAX_TRIALS=2000
rng=np.random.default_rng(SEED)
base=Path(__file__).parents[1];evidence=base/'evidence';evidence.mkdir(parents=True,exist_ok=True)
logpath=evidence/'F_search_trials.jsonl'
counts={};found=None;start=time.monotonic()
def tensor(a): return torch.tensor(a,dtype=torch.float64)
def array(t): return t.detach().cpu().numpy().tolist()
def rv_dice(head,audit,labels):
 return class_metrics(confusion_matrix(labels.numpy(),head(audit).argmax(1).detach().numpy()))[1]['dice']
with logpath.open('w') as log:
 for trial in range(1,MAX_TRIALS+1):
  # Fixed trial recipe: affine feature dimension2, U4 pixels, audit6 pixels.
  w=tensor(rng.normal(0,1,(4,2,1,1))); b=tensor(rng.normal(0,.25,4))
  uf=tensor(rng.normal(0,2,(1,2,1,4))); af=tensor(rng.normal(0,1,(1,2,1,6)))
  labels=torch.tensor([[[0,1,1,2,3,0]]])
  step_size=float(10**rng.uniform(-1,1))
  head=torch.nn.Conv2d(2,4,1,bias=True).double()
  with torch.no_grad(): head.weight.copy_(w);head.bias.copy_(b)
  # A finite affine teacher copy realizes q exactly; no EMA trajectory asserted.
  teacher=copy.deepcopy(head).eval().requires_grad_(False)
  with torch.no_grad(): q=teacher(uf).softmax(1)
  before=rv_dice(head,af,labels)
  directions,ids,info=make_candidates(head,uf,q,.7,step_size)
  row={'trial':trial,'rv_before':before,'source_ids':ids,'accepted_pixels':info['accepted_pixels'],'step_size':step_size}
  if len(ids)<2:
   row['outcome']='LESS_THAN_TWO_SOURCES'
  elif before==0:
   row['outcome']='NO_INITIAL_RV_TRUE_POSITIVE'
  else:
   report=controlled_head_step(head,af,labels,directions,ids,method='joint_guard',ridge=1e-6,budget=0.,backtracks=8,tolerance=1e-8)
   after=rv_dice(head,af,labels)
   lower=bool(np.all(np.asarray(report['after'])<np.asarray(report['before'])-1e-10))
   success=bool(report['scale']>0 and lower and after<before)
   row.update(rv_after=after,guard_report=report,all_three_risks_strictly_lower=lower,outcome='SUCCESS' if success else 'NOT_COUNTEREXAMPLE')
   if success:
    found={'trial':trial,'weight':array(w),'bias':array(b),'unlabeled_features':array(uf),'audit_features':array(af),'labels':array(labels),'teacher_probs':array(q),'step_size':step_size,'threshold':.7,'ridge':1e-6,'budget':0.,'tolerance':1e-8,'backtracks':8,'proposal':info,'source_ids':ids,'directions':[list(map(array,d)) for d in directions],'rv_dice_before':before,'rv_dice_after':after,'guard_report':report}
  counts[row['outcome']]=counts.get(row['outcome'],0)+1
  log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
  if found is not None:break
result={'kind':'MANUFACTURED_CPU_FIRST_SUCCESS_SEARCH','seed':SEED,'max_trials':MAX_TRIALS,'actual_trials':trial,'selection':'first qualifying trial; no prevalence estimate','fixed_shape':{'feature_dim':2,'U_pixels':4,'audit_pixels':6,'audit_positive_counts':{'RV':2,'MYO':1,'LV':1}},'q_recipe':'softmax of identical finite affine teacher copy at manufactured reference; no actual EMA history','counts':counts,'found':found,'elapsed_seconds':time.monotonic()-start}
(evidence/'F_search_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'actual_trials':trial,'counts':counts,'found':None if found is None else {'trial':trial,'source_ids':found['source_ids'],'before':found['rv_dice_before'],'after':found['rv_dice_after'],'risk_before':found['guard_report']['before'],'risk_after':found['guard_report']['after'],'alpha':found['guard_report']['alpha'],'scale':found['guard_report']['scale']},'elapsed_seconds':result['elapsed_seconds']},indent=2))
