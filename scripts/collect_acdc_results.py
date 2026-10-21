#!/usr/bin/env python3
"""Collect actual receipts only; optional watcher never starts training."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from self_audit.model import digest
from self_audit.evaluation import class_metrics


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def write_csv(path, fields, rows):
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    temp.replace(path)


def collect(root, results, suffix='_B16_10000'):
    jobs=read(results/'configs/MATRIX.json')['jobs']
    policy=read(root/'EXECUTION_POLICY.json') or {}
    statuses=[];ablation=[];patient_rows=[];groups={}
    for job in jobs:
        name=job['name'];cfg=read(results/'configs'/(name+'.json'))
        run=root/'runs'/(name+suffix);evaluation=root/'evaluation'/(name+suffix)
        report=read(run/'REPORT.json');failure=read(run/'FAILURE.json');startup=read(run/'STARTUP.json')
        recipe=read(evaluation/'RECIPE.json');metrics=read(evaluation/'METRICS.json')
        status='NOT_STARTED';reason='priority queue covers L8 subset seed42 first; other configurations pending'
        if failure:status='FAILED';reason=failure['error']
        elif report:status='TRAINED_NOT_SCORED';reason='frozen evaluation pending'
        elif startup:status='RUNNING';reason='training final checkpoint not yet complete'
        elif job['label_budget']==8 and job['label_seed']==42:status='QUEUED';reason='await prior queue stage and successful pilots/data verification'
        if policy.get('cancel_all_unstarted') and status in {'NOT_STARTED','QUEUED'}:
            status='NOT_RUN_USER_STOP';reason='user requested stop after EMA-block; no further training scheduled'
        if report and metrics:
            manifest=run/'final/MANIFEST.json'
            assert digest(manifest)==report['final_manifest_sha256']==recipe['checkpoint_sha256']
            assert not metrics['synthetic'] and not report['synthetic']
            status='SCORED_EXPOSED_VAL';reason='fixed final step; development-exposed VAL18, not confirmation'
            groups.setdefault((cfg['method'],job['label_budget']),[]).append(metrics)
            for pid in sorted(metrics['patients']):
                cm=sum((np.asarray(r['raw_confusion']) for r in metrics['frames'] if r['patient']==pid),start=np.zeros((4,5),dtype=np.int64))
                for c in class_metrics(cm):
                    patient_rows.append(dict(run=name,batch_size=cfg['batch_size'],crop_size=cfg.get('crop_size',0),subset_seed=job['label_seed'],patient=pid,class_name=c['class'],dice_mean_frames=metrics['patients'][pid][c['class']],iou_pooled_frames=c['iou'],precision=c['precision'],recall=c['recall'],reference_present=c['reference_present'],TP=c['TP'],FP=c['FP'],FN=c['FN'],bg_to_myo_fp=int(cm[0,2]),hd95_mm=None,confusion_4x5=json.dumps(cm.tolist())))
        statuses.append(dict(run=name,label_budget=job['label_budget'],subset_seed=job['label_seed'],method=cfg['method'],batch_size=cfg['batch_size'],crop_size=cfg.get('crop_size',0),status=status,reason=reason))
        if report:
            rows=[json.loads(s) for s in (run/'steps.jsonl').read_text().splitlines()]
            cm=read(run/'final/MANIFEST.json')
            trials=[t for r in rows for t in r.get('control',{}).get('trials',[])]
            ablation.append(dict(run=name,method=cfg['method'],scope=cfg['update_scope'],label_budget=job['label_budget'],subset_seed=job['label_seed'],training_seed=cfg['seed'],batch_size=cfg['batch_size'],crop_size=cfg.get('crop_size',0),steps=report['steps'],status=status,checkpoint_sha256=report['final_manifest_sha256'],weights_sha256=cm['backbone']['weights_sha256'],data_sha256=cm['data_sha256'],source_sha256=cm['source_sha256'],wall_gpu_hours=report['elapsed_seconds']/3600,peak_vram_bytes=max(r.get('cuda_peak_allocated',0) or 0 for r in rows),accepted_pseudo_transactions=report['pseudo_accepted'],accepted_trials=sum(t['accepted'] for t in trials),rejected_trials=sum(not t['accepted'] for t in trials)))
    main=[]
    for (method,budget),values in groups.items():
        row={'method':method,'label_budget':budget,'batch_size':cfg['batch_size'],'crop_size':cfg.get('crop_size',0),'completed_subset_seeds':len(values),'training_seed':42}
        for c in ['RV','MYO','LV']:
            vals=[v['macro_patient'][c]['dice'] for v in values if v['macro_patient'][c]['dice'] is not None]
            row[c+'_dice_mean']=float(np.mean(vals)) if vals else None
            row[c+'_dice_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        main.append(row)
    write_csv(results/'EXPERIMENT_STATUS.csv',list(statuses[0]),statuses)
    write_csv(results/'RESULTS_MAIN.csv',['method','label_budget','batch_size','crop_size','completed_subset_seeds','training_seed','RV_dice_mean','RV_dice_std','MYO_dice_mean','MYO_dice_std','LV_dice_mean','LV_dice_std'],main)
    write_csv(results/'RESULTS_ABLATION.csv',['run','method','scope','label_budget','subset_seed','training_seed','batch_size','crop_size','steps','status','checkpoint_sha256','weights_sha256','data_sha256','source_sha256','wall_gpu_hours','peak_vram_bytes','accepted_pseudo_transactions','accepted_trials','rejected_trials'],ablation)
    write_csv(results/'PATIENT_METRICS.csv',['run','batch_size','crop_size','subset_seed','patient','class_name','dice_mean_frames','iou_pooled_frames','precision','recall','reference_present','TP','FP','FN','bg_to_myo_fp','hd95_mm','confusion_4x5'],patient_rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--suffix',default='_B16_10000');p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        collect(a.root,a.results,a.suffix)
        if not a.watch or (a.root/'FINALIZED.json').exists():break
        time.sleep(30)
