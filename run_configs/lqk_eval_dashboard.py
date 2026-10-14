import datetime,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
JOB=Path((ROOT/'runs/LATEST_GUIDANCE_EVAL.txt').read_text().strip())
def read(path):return json.loads(path.read_text()) if path.exists() else {}
s=read(JOB/'STATUS.json')
print(datetime.datetime.now().astimezone().strftime('%d/%m %H:%M:%S %Z'))
print('Q/Z PILOT EVALUATION | RTX 4070 | locked VAL18 / 36 frames')
print('Status:',s.get('status','STARTING'),'| stage:',s.get('stage','initialization'))
print('Exports:',s.get('completed_exports',0),'/ 54 |',s.get('current_model',''),s.get('current_patient',''))
print('W&B:',s.get('wandb_url','starting'))
r=read(JOB/'scored/MAIN_METRICS.json')
if r:
    print('Model      Dice      IoU     Prec   Recall    AUROC Coverage (%)')
    for arm in ['Baseline','Q','Z']:
        m=r['results'][arm]['gated']['overall']
        print(f'{arm:8s} '+' '.join(f'{100*m[k]:8.3f}' for k in ['dice','iou','precision','recall','auroc','coverage']))
    print('Precision zero_division=0; AUROC uses ungated probabilities.')
if s.get('error'):print(s['error'])
