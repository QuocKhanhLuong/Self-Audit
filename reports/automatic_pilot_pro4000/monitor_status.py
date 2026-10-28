import json,subprocess
from pathlib import Path
ROOT=Path('/workspace/Self-Audit');R=ROOT/'reports/automatic_pilot_pro4000';ready=[];counts={};latest=None
for regime in ['equal_update_B128_M40','equal_time_B128_M40']:
 ps=sorted((ROOT/'runs/pro4000'/regime).glob('*/REPORT.json'));counts[regime]=len(ps)
 for p in ps:
  r=json.loads(p.read_text())
  if r.get('status')=='COMPLETED':ready.append({'kind':'training','key':regime+'_'+p.parent.name,'path':str(p)})
 for p in (ROOT/'runs/pro4000'/regime).glob('*/steps.jsonl'):
  if p.stat().st_size and (latest is None or p.stat().st_mtime>latest[0]):
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-100000));rs=f.read().splitlines()
   if rs:
    x=json.loads(rs[-1]);latest=(p.stat().st_mtime,{'regime':regime,'arm':p.parent.name,'step':x['step'],'step_seconds':x['step_seconds']})
for regime in ['equal_update_B128_M40','equal_time_B128_M40','equal_time_B128_M40_MnM_verified_VAL']:
 p=R/f'decision_{regime}.json'
 if p.exists():ready.append({'kind':'decision','key':regime,'path':str(p)})
 log=R/f'evaluation_progress_{regime}.json'
 if log.exists():counts['evaluation_'+regime]=len(json.loads(log.read_text())['completed_arms'])
smoke=R/'mnm_loading_smoke.json'
if smoke.exists() and json.loads(smoke.read_text()).get('status')=='PASS':ready.append({'kind':'dataset','key':'MnM_verified_preparation','path':str(smoke)})
log=R/'optimized128-full-training-live.log';txt=log.read_text()[-6000:] if log.exists() else ''
running=subprocess.run(['pgrep','-f','^bash reports/automatic_pilot_pro4000/execute_B128.sh$'],capture_output=True).returncode==0
complete='ALL_FEASIBLE_B128_EXPERIMENTS_COMPLETE' in txt
published=json.loads((R/'PUBLICATIONS.json').read_text()) if (R/'PUBLICATIONS.json').exists() else {}
ready=[a for a in ready if a['kind']+'_'+a['key'] not in published]
print(json.dumps({'ready':ready,'counts':counts,'latest':latest[1] if latest and counts.get('equal_time_B128_M40',0)<10 else None,'running':running,'complete':complete,'failed':not running and not complete,'tail':txt[-2000:] if not running else ''}))
