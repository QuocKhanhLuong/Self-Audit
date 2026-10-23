"""Manufactured adversarial probes. No patient data or training."""
from pathlib import Path
import json, sys, hashlib, subprocess
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from self_audit.control import controlled_head_step
from self_audit.evaluation import class_metrics, confusion_matrix

torch.set_num_threads(1)
# Native affine head with one-hot audit features, deliberately constructed displacement.
p0=torch.tensor([[.44,.46,.05,.05],[.85,.05,.05,.05],[.8,.1,.05,.05],[.8,.1,.05,.05]],dtype=torch.float64).T
p1=torch.tensor([[.50,.42,.04,.04],[.52,.40,.04,.04],[.84,.08,.04,.04],[.84,.08,.04,.04]],dtype=torch.float64).T
features=torch.eye(4,dtype=torch.float64).reshape(1,4,1,4)
y=torch.tensor([[[1,1,0,0]]])
head=torch.nn.Conv2d(4,4,1,bias=True).double()
with torch.no_grad():
 head.weight.copy_(p0.log().reshape(4,4,1,1)); head.bias.zero_()
dw=(p1.log()-p0.log()).reshape(4,4,1,1)
def dice():
 pred=head(features).argmax(1).detach().numpy()
 return class_metrics(confusion_matrix(y.numpy(),pred))[1]['dice']
before=dice()
report=controlled_head_step(head,features,y,[(dw,torch.zeros(4,dtype=torch.float64))],['constructed_direction'],method='joint_guard',ridge=1e-6,budget=0.,tolerance=1e-12)
assert report['scale']==1.0 and report['alpha']==[1.0]
assert all(a < b for a,b in zip(report['after'],report['before']))
assert before==2/3 and dice()==0.0
result={'kind':'SYNTHETIC_CPU_ONLY','guard_and_dice':{'rv_dice_before':before,'rv_dice_after':dice(),'guard_report':report}}
# Reference-absent frames have null Dice even if all pixels are false positives.
cm_present=confusion_matrix(np.array([1,1]),np.array([1,1]))
cm_absent=confusion_matrix(np.array([0,0]),np.array([1,1]))
a=class_metrics(cm_present)[1]; b=class_metrics(cm_absent)[1]
vals=[r['dice'] for r in [a,b] if r['dice'] is not None]
assert a['dice']==1.0 and b['dice'] is None and b['FP']==2
result['empty_class_aggregation']={'present_frame':a,'absent_frame':b,'reported_mean_available_frame_dice':float(np.mean(vals)),'pooled_dice':class_metrics(cm_present+cm_absent)[1]['dice']}
# Exact deterministic patient-index schedules from both main and experiment source.
schedules={}
for nl in [4,8,16]:
 nu=80-nl
 pairs={i:set() for i in range(nu)}
 for step in range(10000): pairs[step%nu].add(step%nl)
 schedules[str(nl)]={'unlabeled_count':nu,'supervised_partners_per_unlabeled':sorted(set(map(len,pairs.values()))),'never_eligible_audit_partner_per_unlabeled':1,'possible_partner_count':nl}
assert all(r['supervised_partners_per_unlabeled']==[1] for r in schedules.values())
result['patient_pairing']=schedules
# Pin source independently from immutable Git objects, not copied evidence files.
sha='8960b6baac4c6600553d900c303f192c2fcfa4f0'
def git(path): return subprocess.check_output(['git','show',sha+':'+path])
files=subprocess.check_output(['git','ls-tree','-r','--name-only',sha,'self_audit'],text=True).splitlines()
files=sorted(p for p in files if p.count('/')==1 and p.endswith('.py'))+['train.py']
source=''.join(p.split('/')[-1]+':'+hashlib.sha256(git(p)).hexdigest()+'\n' for p in files)
result['source_sha256']=hashlib.sha256(source.encode()).hexdigest()
assert result['source_sha256']=='6e12885e44b725c132a01437dc6179cbf524b94c51dd5a5f5254701eb78cf442'
# Reaggregate committed patient CSV only: verification of summaries, not payload replay.
import csv,io
rows=list(csv.DictReader(io.StringIO(git('results/PATIENT_METRICS.csv').decode())))
metrics={}
for method in ['supervised','supervised_extra','ema_full','ema_block']:
 subset=[r for r in rows if r['run']=='L8_S42_'+method and r['class_name']!='BG']
 metrics[method]={c:float(np.mean([float(r['dice_mean_frames']) for r in subset if r['class_name']==c])) for c in ['RV','MYO','LV']}
 metrics[method]['foreground_macro']=float(np.mean(list(metrics[method].values())))
result['committed_patient_csv_reaggregation']=metrics
pooled={}
for method in ['supervised','supervised_extra','ema_full','ema_block']:
 pooled[method]={}
 for c in ['RV','MYO','LV']:
  rs=[r for r in rows if r['run']=='L8_S42_'+method and r['class_name']==c]
  vals=[2*int(r['TP'])/(2*int(r['TP'])+int(r['FP'])+int(r['FN'])) for r in rs]
  pooled[method][c]=sum(vals)/len(vals)
 pooled[method]['derived_fg_mean']=sum(pooled[method].values())/3
out=Path(__file__).parents[1]/'evidence'/'F_probe_results.json';out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
(out.parent/'F_pooled_patient_sensitivity.json').write_text(json.dumps({'kind':'DERIVED_FROM_COMMITTED_CONFUSIONS_NOT_PRIMARY_ENDPOINT','results':pooled},indent=2)+'\n')
print(json.dumps(result,indent=2))
