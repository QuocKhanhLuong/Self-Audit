#!/usr/bin/env python3
"""Run bounded CPU falsification, never a production/full training job."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import yaml
from self_audit_event.data import PhantomDataset, load_acdc
from self_audit_event.training import TrainConfig, EventTrainer, collate_rows, batch_indices, evaluate
from self_audit_event.reference_free_auditor import AuditObservation, rf_loss
from self_audit_event.audit_trigger import trigger_loss
from self_audit_event.evaluation import convergence_metrics, json_safe


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(json_safe(value),indent=2,allow_nan=False));tmp.replace(path)


@torch.no_grad()
def audit_distillation_diagnostic(model, dataset, count=16):
    """Validation-only intrinsic target fit; no segmentation labels or optimizer."""
    errors=[];teacher=[];predicted=[]
    for start in range(0,min(count,len(dataset)),4):
        x=torch.stack([dataset[i]['image'] for i in range(start,min(start+4,count,len(dataset)))])
        obs=AuditObservation.from_state(x,model.actor.start(x))
        target=model.auditor.intrinsic_risk(obs);risk=model.auditor(obs)
        errors.extend(((risk-target)**2).mean((1,2,3)).tolist())
        teacher.extend((-target.mean((1,2,3))).tolist());predicted.extend((-risk.mean((1,2,3))).tolist())
    correct,total=0,0
    for i in range(len(teacher)):
        for j in range(i):
            if teacher[i]!=teacher[j]:
                total+=1;correct+=int((teacher[i]-teacher[j])*(predicted[i]-predicted[j])>0)
    return {'mse':float(np.mean(errors)),'teacher_ranking_accuracy':correct/total if total else None,
            'pairs':total,'meaning':'Fit to intrinsic teacher only, NOT ranking by true Dice.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/self_audit_event_acdc.yaml')
    parser.add_argument('--data-root')
    parser.add_argument('--patient-manifest')
    parser.add_argument('--output',required=True)
    parser.add_argument('--phantom',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    if cfg.get('device') != 'cpu': raise ValueError('This bounded harness is CPU-only; GPU requires a separate free-device receipt.')
    steps,warmup=int(cfg['steps']),int(cfg['warmup_steps'])
    if not 1 <= steps <= 256 or not 0 <= warmup <= 128 or len(cfg['seeds'])>3:
        raise ValueError('bounded-run limits exceeded')
    if int(cfg.get('rf_warmup_steps',0))>64 or int(cfg.get('trigger_warmup_steps',0))>64:
        raise ValueError('bounded RF warmup limits exceeded')
    torch.set_num_threads(int(cfg.get('cpu_threads',4)))
    output=Path(args.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Refusing to overwrite an existing experiment')
    output.mkdir(parents=True,exist_ok=True)
    if args.phantom:
        datasets={'train':PhantomDataset(64,cfg['image_size'],0),'val':PhantomDataset(24,cfg['image_size'],100)}
        identity={'source':'SYNTHETIC_PHANTOM','image_size':cfg['image_size'],'independent_patient_count':0}
    else:
        if not args.data_root or not args.patient_manifest: parser.error('real ACDC requires data root AND unchanged patient manifest')
        datasets,identity=load_acdc(args.data_root,args.patient_manifest,image_size=cfg['image_size'],
                                  validation_patient_limit=cfg.get('validation_patient_limit'))
    root=Path(__file__).resolve().parents[1]
    paths=list((root/'src/self_audit_event').glob('*.py'))+[Path(__file__),Path(args.config),root/'src/self_audit/models/dynamic_window.py']
    save(output/'provenance.json',{'config':cfg,'data':identity,'python':sys.version,'torch':torch.__version__,
        'platform':platform.platform(),'device':'cpu','cuda_used':False,'physical_batch':cfg['training']['batch_size'],
        'accumulation':1,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    results=[]
    for seed in cfg['seeds']:
        config=TrainConfig(seed=seed,**cfg['training'])
        base=EventTrainer(config)
        stream=batch_indices(len(datasets['train']),warmup+steps+128,config.batch_size,seed+200)
        t=time.perf_counter()
        for idx,indices in enumerate(stream[:warmup]):
            b=collate_rows([datasets['train'][i] for i in indices]);base.step(b['image'],b['mask'],policy='none',step=idx)
        actor_setup_seconds=time.perf_counter()-t
        actor_setup_counts=copy.deepcopy(base.counters)
        rf_before=audit_distillation_diagnostic(base.model,datasets['val'])
        # Label tensors are never loaded into RF/trigger warmup functions.
        t=time.perf_counter()
        for idx in range(int(cfg.get('rf_warmup_steps',0))):
            x=torch.stack([datasets['train'][i]['image'] for i in stream[warmup+idx]])
            with torch.no_grad(): state=base.model.actor.start(x)
            obs=AuditObservation.from_state(x,state)
            base.rf_optimizer.zero_grad(set_to_none=True)
            rf_loss(base.model.auditor,obs).backward();base.rf_optimizer.step()
        rf_setup_seconds=time.perf_counter()-t
        rf_after=audit_distillation_diagnostic(base.model,datasets['val'])
        t=time.perf_counter()
        warm_values=[]
        for idx in range(int(cfg.get('trigger_warmup_steps',0))):
            x=torch.stack([datasets['train'][i]['image'] for i in stream[warmup+idx]])
            twin=base.model.twins(x)
            values=twin['intrinsic_delta']-config.cost;warm_values.extend(values.tolist())
            base.trigger_optimizer.zero_grad(set_to_none=True)
            trigger_loss(base.model.trigger,twin['trigger_features'],values).backward();base.trigger_optimizer.step()
        trigger_setup_seconds=time.perf_counter()-t
        setup={'actor_seconds':actor_setup_seconds,'rf_seconds':rf_setup_seconds,'trigger_seconds':trigger_setup_seconds,
               'actor_updates':warmup,'actor_examples':warmup*config.batch_size,
               'rf_updates':cfg.get('rf_warmup_steps',0),'trigger_updates':cfg.get('trigger_warmup_steps',0),
               'rf_examples':cfg.get('rf_warmup_steps',0)*config.batch_size,
               'trigger_probe_examples':cfg.get('trigger_warmup_steps',0)*config.batch_size,
               'rf_distillation_before':rf_before,'rf_distillation_after':rf_after,
               'trigger_warmup_net_values':warm_values,
               'note':'RF/trigger setup uses only training images/model predictions; cost included only for policies using that module.'}
        save(output/f'seed{seed}_setup.json',setup)
        frozen={name:copy.deepcopy(getattr(base,name).state_dict()) for name in ('model','actor_optimizer','rf_optimizer','trigger_optimizer')}
        for policy in cfg['policies']:
            trainer=EventTrainer(config)
            for name,value in frozen.items(): getattr(trainer,name).load_state_dict(value)
            train_logs=[];curve=[];t=time.perf_counter()
            setup_seconds=actor_setup_seconds+(rf_setup_seconds if policy not in ('none','unguided') else 0)+(trigger_setup_seconds if policy=='learned' else 0)
            base_eval=evaluate(trainer.model,datasets['val'],config,policy=policy,step=warmup,include_twins=False)
            curve.append(dict(updates=warmup,examples_seen=warmup*config.batch_size,wall_seconds=setup_seconds+time.perf_counter()-t,
                              **base_eval))
            for idx,indices in enumerate(stream[warmup:warmup+steps]):
                data_start=time.perf_counter()
                b=collate_rows([datasets['train'][i] for i in indices]);data_seconds=time.perf_counter()-data_start
                row=trainer.step(b['image'],b['mask'],policy=policy,step=warmup+idx)
                row.update(update=warmup+idx+1,data_seconds=data_seconds);train_logs.append(row)
                if (idx+1)%cfg['eval_every']==0 or idx+1==steps:
                    ev=evaluate(trainer.model,datasets['val'],config,policy=policy,step=warmup+idx+1,include_twins=False)
                    curve.append(dict(updates=warmup+idx+1,examples_seen=(warmup+idx+1)*config.batch_size,
                                      wall_seconds=setup_seconds+time.perf_counter()-t,**ev))
                    print(json.dumps({'seed':seed,'policy':policy,'step':idx+1,'dice':ev['final_dice'],'audit_rate':ev['audit_fraction']}),flush=True)
            training_total_seconds=setup_seconds+time.perf_counter()-t
            final=evaluate(trainer.model,datasets['val'],config,policy=policy,step=warmup+steps,include_twins=True)
            result={'seed':seed,'policy':policy,'config':vars(config),'setup':setup,'training':train_logs,'curve':curve,
                    'convergence':convergence_metrics(curve,tuple(cfg.get('dice_thresholds',[.5,.7,.8]))),
                    'counters':trainer.counters,'final':final,'training_total_seconds':training_total_seconds,
                    'peak_process_host_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'rss_note':'Process-wide cumulative high water mark; macOS bytes, Linux KiB. Not VRAM or isolated-policy peak.',
                    'wall_note':'Includes setup used by this policy, data loading, training and scheduled validation. Final twin diagnostics excluded and separately timed.'}
            save(output/f'seed{seed}_{policy}.json',result)
            torch.save({'state_dict':trainer.model.state_dict(),'config':vars(config)},output/f'seed{seed}_{policy}.pt')
            results.append({'seed':seed,'policy':policy,'dice':final['final_dice'],'a0':final['a0_dice'],
                            'harm':final['harm_patient_rate'],'audit_rate':final['audit_fraction'],
                            'seconds':training_total_seconds,'correlation':final['patient_transition_metrics']})
            save(output/'summary.json',results)
    print(json.dumps(json_safe(results),allow_nan=False),flush=True)


if __name__=='__main__': main()
