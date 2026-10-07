#!/usr/bin/env python3
"""Bounded TRAIN-only bootstrap pilot. No reference evaluation or student training."""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[1]
for path in (REPO_ROOT,REPO_ROOT/'src'):
    if str(path) not in sys.path: sys.path.insert(0,str(path))
from self_audit_pseudolabel.pipeline_v3 import load_config,bootstrap_recipe,discover
from self_audit_pseudolabel.data_v3 import read_patient_splits,select_records,inspect_record
from self_audit_pseudolabel.freeze import verify_frozen,sha256_file
from scripts.run_full_pipeline_v3 import run_stage


def choose_patients(splits,count,seed):
    if type(count) is not int or count<2 or count>len(splits['train']):
        raise ValueError('pilot needs at least two patients, all from the locked TRAIN split')
    rank=lambda pid:hashlib.sha256(f'{seed}:{pid}'.encode()).hexdigest()
    return sorted(sorted(splits['train'],key=rank)[:count])


def semantic_optimization_enabled(cfg):
    values=(cfg['loss_weights']['semantic_seed'],cfg['training']['lr'])
    return all(not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in values)


def support_report(teacher):
    teacher=Path(teacher)
    payload=verify_frozen(teacher/'FROZEN.json',expected_role='teacher_freeze')
    if any(e['split']!='train' for e in payload['entries']):
        raise ValueError('pilot must export only TRAIN predictions')
    cfg=payload['config'];bc,_,_=bootstrap_recipe(cfg['resolved_config'])
    if bc is None: raise ValueError('pilot requires enabled image-only bootstrap')
    history=json.loads((teacher/'train_metrics.json').read_text())
    per_patient={pid:[0]*4 for pid in cfg['producer_patient_ids']}
    for entry in payload['entries']:
        per_patient[entry['patient_id']]=[a+b for a,b in zip(per_patient[entry['patient_id']],entry['class_pixels'])]
    counts=[sum(row[c] for row in per_patient.values()) for c in range(4)]
    positive=[pid for pid,row in per_patient.items() if sum(row[1:])>0]
    semantic_updates=sum(sum(row.get(f'accepted_{c}_regions',0) for c in ('rv','myo','lv'))>0 for row in history)
    state=cfg.get('bootstrap_summary',{})
    gates={'bootstrap_activated':state.get('ready') is True and state.get('stage')=='semantic',
           'foreground_optimizer_updates':semantic_updates>0 and semantic_optimization_enabled(cfg['resolved_config']),
           'foreground_on_multiple_train_patients':len(positive)>=bc.min_train_patients,
           'rv_myo_lv_observed':all(counts[c]>0 for c in (1,2,3))}
    return {'schema':'self_audit.bootstrap_pilot.v1',
            'status':'SUPPORT_OBSERVED_NOT_QUALITY_CERTIFIED' if all(gates.values()) else 'STOP_INSUFFICIENT_SUPPORT',
            'manifest_id':payload['manifest_id'],'selected_train_patients':list(per_patient),
            'bootstrap':state,'gates':gates,'optimizer_steps':len(history),
            'foreground_optimizer_steps':semantic_updates,'class_order':['BG','RV','MYO','LV'],
            'class_pixels':counts,'patient_class_pixels':per_patient,
            'foreground_patients':positive,'missing_foreground_classes':[c for c in (1,2,3) if counts[c]==0],
            'evaluation_performed':False,'quality_status':'NOT_EVALUATED'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['acdc','mnms'],default='acdc')
    p.add_argument('--root',required=True);p.add_argument('--split-manifest',required=True)
    p.add_argument('--config',default=str(REPO_ROOT/'configs/pseudolabel_v3_bootstrap_experimental.json'))
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--patients',type=int,default=4);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--log-every',type=int,default=25)
    p.add_argument('--max-planned-updates',type=int,default=5000)
    p.add_argument('--no-progress',action='store_true');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    config_sha=sha256_file(args.config);cfg=load_config(args.config);bc,_,_=bootstrap_recipe(cfg)
    if bc is None: raise ValueError('pilot requires an explicitly enabled bootstrap configuration')
    if not semantic_optimization_enabled(cfg): raise ValueError('pilot requires positive semantic seed weight and learning rate')
    if args.epochs<bc.min_observations_per_sample or args.patients<bc.min_train_patients:
        raise ValueError('pilot budget cannot supply the configured repeated multi-patient readiness evidence')
    if min(args.batch_size,args.threads,args.max_planned_updates)<1 or args.log_every<0:
        raise ValueError('invalid pilot budget')
    split_sha=sha256_file(args.split_manifest);splits=read_patient_splits(args.split_manifest)
    patients=choose_patients(splits,args.patients,args.seed)
    subset={'train':patients,'val':[],'test':[],
            'pilot':{'source_split_sha256':split_sha,'source_train_count':len(splits['train']),
                     'selection':'sha256_seed_patient_identity','seed':args.seed}}
    records=select_records(discover(args.dataset,args.root),subset,'train')
    samples={r.patient_id:math.prod(inspect_record(r)[0].shape[2:]) for r in records}
    planned=args.epochs*sum(math.ceil(n/args.batch_size) for n in samples.values())
    if planned>args.max_planned_updates:
        raise ValueError(f'planned {planned} updates exceed pilot ceiling {args.max_planned_updates}; no training started')
    if sha256_file(args.split_manifest)!=split_sha: raise ValueError('source split changed during planning')
    if sha256_file(args.config)!=config_sha: raise ValueError('source configuration changed during planning')
    out=Path(args.out).resolve()
    if out.exists(): raise FileExistsError(out)
    print(f'[PILOT] train_patients={patients} full_samples_per_patient={samples} planned_updates={planned}',flush=True)
    if args.dry_run:
        print('[PILOT] dry run: no training, reference access, or output writes',flush=True)
        return 0
    out.mkdir(parents=True)
    split=out/'pilot_split.json';split.write_text(json.dumps(subset,indent=2))
    config=out/'pilot_config.json';config.write_text(json.dumps(cfg,indent=2))
    command=[sys.executable,str(REPO_ROOT/'scripts/train_pseudolabel_v3.py'),'--dataset',args.dataset,
             '--root',str(Path(args.root).resolve()),'--split-manifest',str(split),'--config',str(config),
             '--export-split','train','--out',str(out/'teacher'),'--device',args.device,'--epochs',str(args.epochs),
             '--batch-size',str(args.batch_size),'--threads',str(args.threads),'--seed',str(args.seed),
             '--log-every',str(args.log_every)]
    if args.no_progress:command.append('--no-progress')
    env=dict(os.environ);env['PYTHONPATH']=str(REPO_ROOT/'src')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    with (out/'pilot.log').open('x') as log:
        run_stage('teacher',command,env,log)
    result=support_report(out/'teacher')
    result.update(planned_optimizer_steps=planned,source_split_sha256=split_sha,source_config_sha256=config_sha)
    (out/'PILOT_REPORT.json').write_text(json.dumps(result,indent=2))
    print(f"[PILOT] {result['status']} class_pixels={result['class_pixels']} gates={result['gates']} report={out/'PILOT_REPORT.json'}",flush=True)
    return 0 if all(result['gates'].values()) else 2


if __name__=='__main__':raise SystemExit(main())
