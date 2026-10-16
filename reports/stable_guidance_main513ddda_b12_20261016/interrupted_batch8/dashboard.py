"""Read-only progress for this fixed main checkout."""
from pathlib import Path
import ast
import datetime
import json
import sys
JOB=Path(__file__).resolve().parent
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
m=read(JOB/'LAUNCH_MANIFEST.json');s=read(JOB/'STATUS.json');w=read(JOB/'tracking/WANDB_RUN.json')
print(datetime.datetime.now().astimezone().strftime('%d/%m %H:%M:%S %Z'))
print('SELF-AUDIT STABLE GUIDANCE | RTX 4070 | main '+m['git_commit'][:7])
print('Budget:',m['epochs'],'epochs | batch',m['batch_size'],'| seed',m['seed'])
print('Architecture:',m['architecture'])
print('Geometry:',m['orientation_policy'])
print('Status:',s.get('status'),'| stage:',w.get('stage',s.get('stage')))
print('W&B:',w.get('url','initializing'))
if len(sys.argv)>1 and sys.argv[1]=='metrics':
    final=read(JOB/'tracking/FINAL_METRICS.json')
    if final:
        for key in ['val/dice','val/iou','val/precision','val/recall','val/coverage']:
            print(key,final['metrics'].get(key))
    else:print('Final metrics follow 50 epochs -> VAL + TEST freezes -> VAL evaluation.')
    print('AUROC: unavailable in this hard-label pipeline.')
    print('Precision: undefined values excluded, with defined-patient counts.')
else:
    event=w.get('last_event',{});print('Latest telemetry:',event)
    for pf in sorted((JOB/'run').glob('PREFLIGHT_*.json'))[-1:]:
        p=read(pf);print('Preflight:',p.get('status'))
        inventory=p.get('checks',{}).get('native_cine_inventory',{})
        if inventory:print('Split:',inventory.get('split_counts'),'| steps/epoch:',inventory.get('train_steps_per_epoch'))
    rows=[]
    if (JOB/'launcher.log').exists():
        for line in (JOB/'launcher.log').read_text().splitlines():
            if line.startswith('[STUDENT] '):rows.append(ast.literal_eval(line.partition('] ')[2]))
    print('Completed epochs:',len(rows),'/',m['epochs'])
    for r in rows[-5:]:print('Epoch',r['epoch'],'feature',round(r['feature_loss'],5),'semantic',round(r['semantic_loss'],5))
    print('Telemetry errors:',w.get('telemetry_errors',0))
if s.get('error'):print('ERROR:',s['error'])
