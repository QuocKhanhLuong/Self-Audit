from pathlib import Path
import sys,os,json,hashlib,shutil,subprocess,datetime
import numpy as np
import torch
repo=Path('/root/Self-Audit-fuse-batch2-20261113');sys.path.insert(0,str(repo));os.chdir(repo)
from self_audit.candidate_fuse.config import config_sha256,validate_config
from self_audit.candidate_fuse.engine import _atomic_torch_save,_atomic_json
b=Path('/root/mount_data/runs/fuse-20261113')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
services=subprocess.run(['supervisorctl','status','fuse-campaign','fuse-watchdog'],capture_output=True,text=True).stdout
assert services.count('STOPPED')==2,services
source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert source_sha=='d9dfdf80966cf7fdf480c972b098c5cf0d64fb99'
cp=b/'resume.pt';old_sha=sha(cp);assert old_sha==json.loads((b/'resume.sha256.json').read_text())['sha256']
c=torch.load(cp,map_location='cpu',weights_only=False);assert c['stage']=='T' and c['update']==1100
old_identity=dict(c['identity']);old_cfg=json.loads((b/'recipe.json').read_text());assert 'teacher_batch_size' not in old_cfg['training']
new_cfg=json.loads(json.dumps(old_cfg));new_cfg['training']['teacher_batch_size']=2;validate_config(new_cfg,real_run=True)
new_identity={**old_identity,'config_sha256':config_sha256(new_cfg)}
meta=json.loads((b/'teacher.pt.meta.json').read_text());assert meta['identity']==old_identity and meta['sha256']==sha(b/'teacher.pt')
state=json.loads((b/'STATE.json').read_text());assert state['checkpoint_sha256']==old_sha
campaign=json.loads((b/'RUN_STATE.json').read_text());assert campaign['config_sha256']==sha(b/'recipe.json') and campaign['active']=='T' and not campaign['completed']
assert campaign['test_ledger']['status']=='NOT_RUN'
r=c['metrics_receipt'];prefix=(b/'metrics.jsonl').read_bytes()[:r['offset']];assert hashlib.sha256(prefix).hexdigest()==r['sha256']
vals=[json.loads(x) for x in prefix.splitlines() if json.loads(x).get('type')=='validation']
assert max(x['modes']['teacher']['fg_dice'] for x in vals)==c['best_t']
assert (b/'teacher.pt').stat().st_mtime<=cp.stat().st_mtime
archive=b/'batch1_archive';archive.mkdir(exist_ok=False)
files=['resume.pt','resume.sha256.json','STATE.json','RUN_STATE.json','recipe.json','teacher.pt','teacher.pt.meta.json','metrics.jsonl','EPOCH_BUDGET_LOCK.json']
old_files={}
for name in files:
 src=b/name;old_files[name]=sha(src)
 if name in ['resume.pt','teacher.pt']:os.link(src,archive/name)
 else:shutil.copy2(src,archive/name)
# Hash all checkpoint training payload excluding only the explicitly changed identity.
def fingerprint(value):
 h=hashlib.sha256()
 def visit(x):
  h.update(type(x).__name__.encode())
  if isinstance(x,torch.Tensor):
   h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=repr):h.update(repr(k).encode());visit(x[k])
  elif isinstance(x,(tuple,list)):
   for v in x:visit(v)
  else:h.update(repr(x).encode())
 visit({k:v for k,v in value.items() if k!='identity'});return h.hexdigest()
before=fingerprint(c);c['identity']=new_identity;new_sha=_atomic_torch_save(c,cp)
reloaded=torch.load(cp,map_location='cpu',weights_only=False);assert fingerprint(reloaded)==before
_atomic_json(b/'recipe.json',new_cfg)
meta['identity']=new_identity;_atomic_json(b/'teacher.pt.meta.json',meta)
_atomic_json(b/'resume.sha256.json',{'path':'resume.pt','sha256':new_sha})
state['checkpoint_sha256']=new_sha;state['status']='PAUSED';_atomic_json(b/'STATE.json',state)
campaign['config_sha256']=sha(b/'recipe.json');campaign['status']='PAUSED';_atomic_json(b/'RUN_STATE.json',campaign)
receipt={'schema':'self_audit.candidate_fuse.batch_migration.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'MIGRATED','source_commit':source_sha,'user_authorization':'Increase batch to optimize VRAM; preserve native data and epoch counts','old_identity':old_identity,'new_identity':new_identity,'old_checkpoint_sha256':old_sha,'new_checkpoint_sha256':new_sha,'preserved_training_payload_sha256':before,'weights_optimizer_rng_sampler_metrics_unchanged':True,'teacher_batch_size':2,'joint_pairs_per_step':1,'teacher_examples_seen':c['update'],'teacher_optimizer_steps_before':c['update'],'teacher_epochs':50,'joint_epochs':100,'archive':str(archive),'archived_files':old_files,'best_teacher_score_before':c['best_t'],'remaining_teacher_frames':8000-c['update'],'expected_new_optimizer_steps_for_full_teacher_epochs':80,'notes':'Partial resumed epoch may have singleton frames. T update counts labeled frame exposures; optimizer_steps is separate. J update semantics unchanged. Old weights and all source files preserved.'}
_atomic_json(b/'BATCH_MIGRATION_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2))
