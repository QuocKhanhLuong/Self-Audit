#!/usr/bin/env python3
"""Fail-closed sequential server experiment; no resume or background trainer fork."""
import csv, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from self_audit.engine import source_digest
from self_audit.model import digest
from self_audit.data import write_json

ROOT=Path('/root/self-audit-pilot-artifacts')
PY=sys.executable
PINNED=source_digest()

def status(stage,**kw):
    obj=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256=PINNED,**kw)
    write_json(ROOT/'STATUS.json',obj)
    print(json.dumps(obj),flush=True)

def run(cmd,log,allowed=(0,),env=None):
    assert source_digest()==PINNED,'runtime source changed'
    print('COMMAND '+json.dumps(list(map(str,cmd))),flush=True)
    with (ROOT/log).open('x') as f:
        p=subprocess.run(list(map(str,cmd)),cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=env)
    if p.returncode not in allowed:raise RuntimeError(f'{log}: exit {p.returncode}')
    return p.returncode

def recipe(arm,name,batch,threads,steps=10000,warmup=1000):
    c=json.loads((REPO/f'configs/window_pilot_{arm}.json').read_text())
    c.update(steps=steps,warmup_steps=warmup,batch_size=batch,threads=threads)
    c['model']=json.loads((ROOT/'BACKBONE.json').read_text())
    dest=ROOT/'configs'/f'{name}.json';write_json(dest,c)
    return dest

def train(config,out,log):
    return run([PY,'train.py','--device','cuda','--allow-full-run','--config',config,
        '--data',ROOT/'data/cache/L8_S42/train.json','--data-sha256',digest(ROOT/'data/cache/L8_S42/train.json'),
        '--out',out],log)

def bench(batch,threads,tag):
    out=ROOT/'benchmarks'/f'{tag}.json'
    run([PY,'scripts/benchmark_window_cuda.py','--checkpoint',ROOT/'runs/preflight/final',
        '--data',ROOT/'data/cache/L8_S42/train.json','--batch',batch,'--threads',threads,
        '--repeats',5,'--out',out],f'benchmarks/{tag}.log',allowed=(0,2))
    return json.loads(out.read_text())

def main():
    for p in ('configs','benchmarks','runs','evaluation'): (ROOT/p).mkdir(exist_ok=True)
    gate=json.loads((ROOT/'hypergradient-cuda.json').read_text())
    assert gate['status']=='PASS' and gate['source_sha256']==PINNED and gate['device']=='cuda'
    status('WAITING_FOR_RCLONE_COPY')
    # PID belongs exclusively to this experiment's rclone copy, captured at launch.
    pid=int((ROOT/'download.pid').read_text())
    while Path(f'/proc/{pid}').exists():time.sleep(5)
    status('CHECKING_TRANSPORT')
    run(['rclone','check','KhanhDrive:Dataset/ACDC/training',ROOT/'data/raw/ACDC/training',
        '--one-way','--checkers','8','--contimeout','15s','--timeout','60s','--retries','3',
        '--combined',ROOT/'ACDC_RCLONE_CHECK.txt'],'check-acdc.log')
    lines=(ROOT/'ACDC_RCLONE_CHECK.txt').read_text().splitlines()
    assert len(lines)>=700 and all(x.startswith('= ') for x in lines)
    status('PREPARING_TRAIN_L8_AND_SEPARATE_DEVELOPMENT')
    run([PY,'prepare_acdc.py','--root',ROOT/'data/raw/ACDC/training',
         '--split',REPO/'splits/acdc_reviewed_locked_train80_val18.json','--labels','8','--seed','42',
         '--evaluation','--out',ROOT/'data/cache/L8_S42'],'prepare.log')
    prep=json.loads((ROOT/'data/cache/L8_S42/PREPARATION.json').read_text())
    assert prep['unlabeled_mask_reads']==0
    # TrainingData validates all normalized images through mmap at startup, warming
    # the OS cache. No per-worker cine copies; comfortably within the 129GB budget.
    status('REAL_BACKBONE_PREFLIGHT')
    cfg=recipe('meta','preflight',1,2,steps=1032,warmup=1000)
    train(cfg,ROOT/'runs/preflight','preflight.log')
    rep=json.loads((ROOT/'runs/preflight/REPORT.json').read_text())
    rows=[json.loads(x) for x in (ROOT/'runs/preflight/steps.jsonl').read_text().splitlines()]
    updates=[r['meta_window'] for r in rows if r.get('meta_window',{}).get('geometry_update_norm',0)>0]
    assert rep['geometry_updates']>0 and updates and rep['label_reads_outside_budget']==[]
    assert all(r['supervised']['patient']!=r['audit']['patient'] for r in rows)
    write_json(ROOT/'PREFLIGHT.json',dict(status='PASS',source_sha256=PINNED,report=rep,actual_nonzero_meta_updates=updates))
    status('BENCHMARK_FULL_META_TRANSACTIONS')
    results=[]
    batch=1
    while True:
        result=bench(batch,2,f'b{batch}_t2');results.append(result)
        write_json(ROOT/'BENCHMARK.json',results)
        if result['status']!='PASS':
            assert result['status']=='OOM_WITH_2GIB_MARGIN',result
            break
        batch*=2
        if batch>256:raise RuntimeError('no capacity limit established by batch256')
    passing=[r for r in results if r['status']=='PASS']
    assert passing
    initial=max(passing,key=lambda x:x['samples_per_second'])
    for threads in (4,8):
        result=bench(initial['batch_size'],threads,f'b{initial["batch_size"]}_t{threads}')
        results.append(result);write_json(ROOT/'BENCHMARK.json',results)
    best=max((r for r in results if r['status']=='PASS'),key=lambda x:x['samples_per_second'])
    with (ROOT/'BENCHMARK.csv').open('w') as f:
        fields=['batch_size','threads','status','samples_per_second','median_seconds','peak_allocated_bytes','peak_reserved_bytes']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
    write_json(ROOT/'SELECTED_RECIPE.json',dict(batch_size=best['batch_size'],threads=best['threads'],crop_size=256,
        selection='maximum measured samples_per_second across passing full meta transaction probes',
        margin_bytes=2*2**30,cache_policy='mmap + warmed OS page cache; no worker copies',benchmark=best))
    summaries=[];reference_schedule=None
    for arm in ('baseline','fixed','meta'):
        cfg=recipe(arm,f'{arm}_10000',best['batch_size'],best['threads'])
        status('TRAINING',arm=arm,batch=best['batch_size'],threads=best['threads'])
        out=ROOT/'runs'/arm
        train(cfg,out,f'train-{arm}.log')
        report=json.loads((out/'REPORT.json').read_text());assert report['steps']==10000 and report['status']=='COMPLETED'
        rows=[json.loads(x) for x in (out/'steps.jsonl').read_text().splitlines()]
        schedule=hashlib.sha256()
        for row in rows:
            assert row['supervised']['patient']!=row['audit']['patient']
            schedule.update(json.dumps({k:row[k] for k in ('supervised','audit','unlabeled')},sort_keys=True).encode())
        sched=schedule.hexdigest()
        if reference_schedule is None:reference_schedule=sched
        assert sched==reference_schedule,'unmatched sample schedules'
        assert report['label_reads_outside_budget']==[]
        if arm=='meta':assert report['geometry_updates']>0
        status('FROZEN_DEVELOPMENT_EVALUATION',arm=arm)
        dev=ROOT/'data/cache/L8_S42/development.json'
        env=dict(os.environ,OMP_NUM_THREADS=str(best['threads']),MKL_NUM_THREADS=str(best['threads']))
        run([PY,'evaluate.py','--checkpoint',out/'final','--checkpoint-sha256',report['final_manifest_sha256'],
            '--data',dev,'--data-sha256',digest(dev),'--out',ROOT/'evaluation'/arm,'--device','cuda'],f'eval-{arm}.log',env=env)
        metrics=json.loads((ROOT/'evaluation'/arm/'METRICS.json').read_text())
        cm=json.loads((out/'final/MANIFEST.json').read_text())
        summary=dict(arm=arm,steps=report['steps'],warmup_steps=1000,batch_size=best['batch_size'],crop_size=256,
            seed=42,label_budget=8,scope='development-exposed VAL18; not confirmation',
            macro_patient=metrics['macro_patient'],
            foreground_patient_macro=float(np.mean([np.mean([p[k] for k in ('RV','MYO','LV') if p[k] is not None]) for p in metrics['patients'].values()])),
            elapsed_seconds=report['elapsed_seconds'],peak_allocated_bytes=max(r['cuda_peak_allocated'] for r in rows),
            geometry_updates=report['geometry_updates'],source_sha256=cm['source_sha256'],
            weights_sha256=cm['backbone']['weights_sha256'],data_sha256=cm['data_sha256'],
            final_manifest_sha256=report['final_manifest_sha256'],payload_sha256=cm['payload_sha256'],
            schedule_sha256=sched,metrics_sha256=digest(ROOT/'evaluation'/arm/'METRICS.json'))
        summaries.append(summary);write_json(ROOT/'RESULTS.json',summaries)
        with (ROOT/'RESULTS.csv').open('w') as f:
            fields=['arm','RV','MYO','LV','patient_macro','hours','peak_vram_GiB']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for s in summaries:w.writerow(dict(arm=s['arm'],**{k:s['macro_patient'][k]['dice'] for k in ('RV','MYO','LV')},
                patient_macro=s['foreground_patient_macro'],hours=s['elapsed_seconds']/3600,peak_vram_GiB=s['peak_allocated_bytes']/2**30))
    status('FINAL_CHECKS')
    run([PY,'scripts/check_repo.py'],'final-check-repo.log')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='');run([PY,'-m','pytest','-q'],'final-pytest.log',env=env)
    status('COMPLETED',results=str(ROOT/'RESULTS.json'))

if __name__=='__main__':
    import numpy as np
    try:main()
    except BaseException as e:status('FAILED',error=repr(e));raise
