"""Execution-only coordinator: every learner call remains canonical train.py."""
import sys,json,time,threading,argparse,os
from pathlib import Path
ROOT=Path('/workspace/Self-Audit');sys.path.insert(0,str(ROOT))
from scripts import run_automatic_pilot as pilot
from self_audit.model import digest
R=ROOT/'reports/automatic_pilot_pro4000';D=ROOT/'data/prepared/ACDC_L8_S42/train.json'
original_load=pilot.load_arm_config

def load_config(arm,config_dir=None):
 c=original_load(arm,config_dir);c.update(threads=4,max_extract_batch=4,max_controller_workspace_bytes=8*1024**3,save_every=1000,batch_size=int(os.environ.get("PILOT_BATCH_SIZE","1")),full_ema_microbatch_size=int(os.environ.get("PILOT_MICROBATCH_SIZE","0")))
 return c
pilot.load_arm_config=load_config
stop=threading.Event();monitor_start=time.time()
def monitor():
 last={}
 while not stop.wait(10):
  try:
   import subprocess
   gpu=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
   ram=Path('/sys/fs/cgroup/memory.current').read_text().strip()
   with (R/'utilization.csv').open('a') as f:f.write(f'{gpu},{ram}\n')
   latest=sorted((p for p in (ROOT/'runs/pro4000').rglob('steps.jsonl') if p.stat().st_mtime >= monitor_start),key=lambda p:p.stat().st_mtime,reverse=True)
   if latest:
    p=latest[0]
    with p.open('rb') as f:
     f.seek(max(0,p.stat().st_size-100000));lines=f.read().splitlines()
    if lines:
     x=json.loads(lines[-1]);key=(str(p),x['step'])
     if last.get('key')!=key:
      print('TRAIN_PROGRESS',json.dumps({'run':str(p.parent),'step':x['step'],'step_seconds':x['step_seconds'],'elapsed_seconds':x['elapsed_seconds'],'status':x.get('status'),'supervised_loss':x.get('supervised_loss'),'gpu':gpu,'ram_bytes':ram}),flush=True);last['key']=key
   else:print('STARTUP',gpu,'RAM',ram,flush=True)
  except Exception as e:print('monitor:',type(e).__name__,str(e),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['profile_warm','profile_matrix','profile_candidates','profile_batch','warm_start','equal_update','equal_time']);ap.add_argument('--steps',type=int);ap.add_argument('--time-budget',type=float,default=600);ap.add_argument('--suffix',default='');ap.add_argument('--arms',default='all');ap.add_argument('--pseudo-threshold',type=float);a=ap.parse_args()
 stage=a.stage;out=ROOT/'runs/pro4000'/(stage+a.suffix);model=json.loads((R/'model.json').read_text());s=threading.Thread(target=monitor,daemon=True);s.start();start=time.monotonic()
 try:
  if stage in ('profile_warm','warm_start'):
   out.mkdir(parents=True,exist_ok=False)
   path,sha,receipt=pilot.run_warm_start_pipeline(out,D,digest(D),support_patients=[r['id'] for r in json.loads(D.read_text())['labeled']],device='cuda',synthetic=False,allow_full_run=stage=='warm_start',steps=a.steps or (2 if stage=='profile_warm' else 512),model_config=model)
   report={'stage':stage,'path':str(path),'manifest_sha256':sha,'receipt':receipt}
  else:
   warmstage='profile_warm' if stage in ('profile_matrix','profile_candidates') else 'warm_start';warm=json.loads((R/os.environ.get('PILOT_WARM_START_RECEIPT',warmstage+'.json')).read_text())
   report=pilot.run_experiment_matrix(pilot.FIXED_ARMS if a.arms=='all' else a.arms.split(','),D,digest(D),out,comparison='equal_time' if stage=='equal_time' else 'equal_update',time_budget=a.time_budget if stage=='equal_time' else None,steps=a.steps or (2 if stage.startswith('profile_') else 128),device='cuda',synthetic=False,allow_full_run=not stage.startswith('profile_'),warm_start_checkpoint=warm['path'],warm_start_sha256=warm['manifest_sha256'],model_config=model,pseudo_threshold=a.pseudo_threshold,max_steps=100000)
   # Reuse the measured receipt for the one shared warm-start; this is not
   # an estimate and does not resume any timed arm trajectory.
   report['warm_start']['receipt']=warm['receipt']
   for item in report['arms'].values():
    item['accounting']=pilot.compute_resource_accounting(item['receipt'],warm_start_receipt=warm['receipt'],evaluation_receipt=None)
   (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  (R/(stage+a.suffix+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  print('PHASE_COMPLETE',stage,'wall_seconds',time.monotonic()-start,flush=True)
 finally:stop.set();s.join(timeout=1)
