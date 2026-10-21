#!/usr/bin/env python3
"""Finish an already-running EMA-block, score, check and push; never train."""
import argparse,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from self_audit.model import digest
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--trainer-pid',type=int,required=True);a=p.parse_args()
repo=Path(__file__).resolve().parents[1];root=a.root.resolve();results=repo/'results';run=root/'runs/L8_S42_ema_block_C256_B52_10000';ev=root/'evaluation/L8_S42_ema_block_C256_B52_10000'
status_path=root/'FINALIZATION_STATUS.json'
def status(stage,**extra):
 o={'stage':stage,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra};tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(o,indent=2)+'\n');tmp.replace(status_path);print(json.dumps(o),flush=True)
def cmd(args,**kwargs):
 return subprocess.run(args,cwd=repo,check=True,**kwargs)
def running():
 try:return Path(f'/proc/{a.trainer_pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False
try:
 status('WAITING_EMA_BLOCK',trainer_pid=a.trainer_pid,remaining_training_cancelled=True,test_inference_cancelled=True)
 while not (run/'REPORT.json').is_file():
  if (run/'FAILURE.json').is_file() or not running():raise RuntimeError('EMA-block ended without successful REPORT; no restart attempted')
  time.sleep(20)
 while running():time.sleep(2)
 r=json.loads((run/'REPORT.json').read_text());sha=digest(run/'final/MANIFEST.json')
 assert r['status']=='COMPLETED' and r['steps']==10000 and not r['synthetic'] and sha==r['final_manifest_sha256']
 assert not r['label_reads_outside_budget']
 dev=Path('/workspace/self-audit-v2/cache/L8_S42/development.json')
 status('EVALUATING_FROZEN_EMA_BLOCK',checkpoint_manifest_sha256=sha)
 if not (ev/'METRICS.json').exists():
  if ev.exists():raise RuntimeError('Incomplete evaluation directory exists; preserve it, do not approximate resume')
  cmd([sys.executable,'evaluate.py','--checkpoint',str(run/'final'),'--checkpoint-sha256',sha,'--data',str(dev),'--data-sha256',digest(dev),'--out',str(ev),'--device','cuda'])
 recipe=json.loads((ev/'RECIPE.json').read_text());assert recipe['checkpoint_sha256']==sha
 status('EXPORTING_FINAL_REPORTS')
 figures=results/'figures/L8_S42_ema_block_C256_B52_10000'
 if not (figures/'PROVENANCE.json').exists():
  if figures.exists():raise RuntimeError('Incomplete figure export exists; preserve it')
  cmd([sys.executable,'scripts/render_acdc_results.py','--evaluation',str(ev),'--data',str(dev),'--out',str(figures),'--train-run',str(run)])
 cmd([sys.executable,'scripts/collect_acdc_results.py','--root',str(root),'--results',str(results),'--suffix','_C256_B52_10000'])
 cmd([sys.executable,'scripts/report_completed_acdc.py','--root',str(root),'--results',str(results)])
 cmd([sys.executable,'scripts/plot_acdc_comparison.py','--results',str(results)])
 status('RUNNING_REQUIRED_CHECKS')
 checks={}
 check_env=dict(os.environ);check_env['CUDA_VISIBLE_DEVICES']=''
 for name,args in [('check_repo',[sys.executable,'scripts/check_repo.py']),('pytest',[sys.executable,'-m','pytest','-q'])]:
  c=subprocess.run(args,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=check_env)
  (root/'logs'/('final_'+name+'.log')).write_text(c.stdout)
  checks[name]={'returncode':c.returncode,'CUDA_VISIBLE_DEVICES':'','output_tail':c.stdout[-2500:]}
  (results/'logs/FINAL_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
  if c.returncode:raise RuntimeError(name+' failed; final push withheld')
 branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip();assert branch=='experiments/acdc-v2-vram24'
 files=[results/'REPORT.md',results/'configs/MATRIX.json']+list(results.glob('*.csv'))+list((results/'logs').glob('*.json'))
 for directory in (results/'figures').iterdir():
  if directory.is_dir():
   for name in ['confusion.png','confusion.pdf','patient_dice.png','patient_dice.pdf','training_cost_and_acceptance.png','training_cost_and_acceptance.pdf','PROVENANCE.json']:
    if (directory/name).is_file():files.append(directory/name)
 for prefix in ['comparison_val_dice','accuracy_cost','acceptance_comparison']:
  files += [results/'figures'/(prefix+'.'+ext) for ext in ['png','pdf']]
 status('PUSHING_FINAL_RESULTS',branch=branch)
 cmd(['git','add','--']+[str(f.relative_to(repo)) for f in files])
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=repo).returncode:
  cmd(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Record final EMA-block VAL results and stop remaining experiments at user request'])
 for attempt in range(3):
  c=subprocess.run(['git','-c','http.version=HTTP/1.1','-c','http.postBuffer=52428800','push','origin',branch],cwd=repo)
  if not c.returncode:break
  if attempt<2:time.sleep(30)
 else:raise RuntimeError('Final results committed locally but Git push failed after three attempts')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 final={'status':'COMPLETED_AND_PUSHED','commit':commit,'branch':branch,'training_finished':True,'test_inference_run':False,'remaining_matrix_runs':'CANCELLED_BY_USER'}
 (root/'FINALIZED.json').write_text(json.dumps(final,indent=2)+'\n');status('COMPLETED_AND_PUSHED',**final)
except BaseException as exc:
 status('FINALIZATION_FAILED',error=repr(exc));traceback.print_exc();raise
