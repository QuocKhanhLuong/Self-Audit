#!/usr/bin/env python3
"""Bounded manufactured CLI smoke; never a patient accuracy/resource experiment."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from self_audit.compact_artifacts import digest
from self_audit.compact_data import synthetic_compact_fixture
from self_audit.engine import source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    logs = out/'commands'
    logs.mkdir()
    commands = []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def run(name, *argv, allowed=(0,)):
        command = [sys.executable, *map(str, argv)]
        started = time.perf_counter()
        proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
        (logs/f'{name}.stdout').write_text(proc.stdout)
        (logs/f'{name}.stderr').write_text(proc.stderr)
        commands.append({'name':name,'argv':command,'exit_code':proc.returncode,
                         'seconds':time.perf_counter()-started})
        (out/'COMMANDS.json').write_text(json.dumps(commands, indent=2)+'\n')
        if proc.returncode not in allowed:
            raise RuntimeError(f'{name} exited {proc.returncode}; see {logs/name}.stderr\n{proc.stderr[-3000:]}')

    train, train_sha = synthetic_compact_fixture(out/'train_data', seed=217,
                                                n_labeled=1,n_unlabeled=2,size=16,
                                                num_frames=2,num_slices=2)
    run('cache','scripts/build_compact_teacher_cache.py','--data-manifest',train,
        '--data-sha256',train_sha,'--output-dir',out/'cache','--synthetic-teacher',
        '--device',args.device,'--batch-size',2)
    cache_path=out/'cache'/'CACHE_MANIFEST.json'
    cache=json.loads(cache_path.read_text())
    teacher=cache['teacher_identity']
    train_meta=json.loads(train.read_text())
    ids=[r['id'] for r in train_meta['labeled']]
    configs={}
    configs_cache_sha = digest(cache_path)
    for method in ('supervised','plain_kd','scalar_guard','joint_guard'):
        c=json.loads((ROOT/'configs/compact'/f'{method}.json').read_text())
        c.update(steps=2,warmup_steps=1,seed=217,batch_size=1,labeled_batch_size=1,
                 unlabeled_batch_size=1,crop_size=16,pseudo_threshold=0.,max_trials=2)
        c['model']={'name':'CompactStudent','widths':[4,8],'in_channels':3,'num_classes':4}
        c['data']={'manifest_sha256':train_sha,'budget_k':1,'labeled_ids':ids}
        c['teacher']={'cache_path':str(cache_path),'cache_sha256':digest(cache_path),
                      'teacher_identity':teacher,'source_training_cost':'SYNTHETIC_NOT_PATIENT_EVIDENCE',
                      'lineage':{'budget_k':1,'exact_labeled_ids':ids}}
        config_path=out/f'{method}.json'
        config_path.write_text(json.dumps(c,indent=2)+'\n')
        configs[method]={'path':str(config_path),'sha256':digest(config_path)}
        run(method,'train.py','--config',config_path,'--data',train,'--data-sha256',train_sha,
            '--out',out/method,'--device',args.device,'--synthetic')

    checkpoint=out/'plain_kd'/'final'
    cp_sha=digest(checkpoint/'MANIFEST.json')
    export=out/'export'
    run('export','scripts/export_compact_student.py','--checkpoint',checkpoint,
        '--checkpoint-sha256',cp_sha,'--out',export)
    export_sha=digest(export/'MANIFEST.json')
    eval_dir=out/'eval_data'
    eval_dir.mkdir()
    rng=np.random.default_rng(218)
    patients=[]
    for j,(h,w) in enumerate(((17,19),(19,21))):
        pid=f'synthetic_eval_{j}'
        cine=rng.random((2,2,h,w),dtype=np.float32)
        image=eval_dir/f'{pid}.npy'
        np.save(image,cine)
        mask=eval_dir/f'{pid}_frame0.npy'
        np.save(mask,(cine[0]*4).astype(np.uint8))
        affine=np.diag([1.25,1.5,7.5,1.])
        affine[0,3]=j*10+3
        patients.append({'id':pid,'domain':'manufactured','image':image.name,'image_sha256':digest(image),
                         'axes':['T','Z','Y','X'],'affine':affine.tolist(),
                         'masks':{'0':{'path':mask.name,'sha256':digest(mask)}}})
    eval_manifest=eval_dir/'manifest.json'
    eval_manifest.write_text(json.dumps({'schema':'self_audit.native_manifest.v1','role':'development',
                                        'synthetic':True,'patients':patients},indent=2)+'\n')
    eval_sha=digest(eval_manifest)
    common=['--export',export,'--export-sha256',export_sha,'--manifest',eval_manifest,
            '--manifest-sha256',eval_sha,'--device',args.device,'--batch-size',2]
    run('infer','scripts/infer_compact_student.py',*common,'--out',out/'inference')
    run('evaluate','scripts/evaluate_compact_student.py',*common,'--train-manifest',train,
        '--train-manifest-sha256',train_sha,'--out',out/'evaluation')
    benchmark=out/'benchmark.json'
    run('benchmark','scripts/benchmark_compact_student.py','--compact-export',export,
        '--compact-export-sha256',export_sha,'--manifest',eval_manifest,'--manifest-sha256',eval_sha,
        '--out',benchmark,'--device',args.device,'--batch-size',2,'--warmup',1,'--repeats',1)
    quality=out/'evaluation'/'METRICS.json'
    run('assess','scripts/assess_compact_deployment.py','--export',export,'--export-sha256',export_sha,
        '--quality',quality,'--quality-sha256',digest(quality),'--benchmark',benchmark,
        '--benchmark-sha256',digest(benchmark),'--out',out/'ACCEPTANCE.json',allowed=(2,))
    assessment=json.loads((out/'ACCEPTANCE.json').read_text())
    if assessment['status']!='BLOCKED':
        raise AssertionError('Manufactured smoke must not pass patient deployment gates')
    receipts={m:json.loads((out/m/'REPORT.json').read_text()) for m in configs}
    initial_hashes = {r['initialization']['initial_state_sha256'] for r in receipts.values()}
    if len(initial_hashes) != 1:
        raise AssertionError('compact arm initializations differ')
    step_logs = {m:[json.loads(line) for line in (out/m/'TRAIN_LOG.jsonl').read_text().splitlines()] for m in configs}
    schedules = {tuple(row['data_schedule_sha256'] for row in rows) for rows in step_logs.values()}
    if len(schedules) != 1:
        raise AssertionError('compact arm L/U schedules differ')
    for rows in step_logs.values():
        if not all(value > 0 for value in rows[0]['warm_start']['weight_changes'].values()):
            raise AssertionError('a compact module was not actually updated')
    for method in ('plain_kd','scalar_guard','joint_guard'):
        last = step_logs[method][-1]
        if last['candidate_receipt']['denominator'] <= 0:
            raise AssertionError('fixed teacher signal did not reach the student')
        if last['trial_count'] != len(last['transaction']['trials']) or last['trial_count'] > 2:
            raise AssertionError('incomplete transaction receipt or exceeded trial cap')
    if receipts['plain_kd']['counts']['pseudo_accepted'] < 1:
        raise AssertionError('plain distillation produced no nonzero update')
    if digest(cache_path) != configs_cache_sha:
        raise AssertionError('fixed teacher cache changed during student training')
    result={'schema':'self_audit.compact.software_smoke.v1','status':'VERIFIED','synthetic':True,
            'patient_quality':'NOT_RUN','patient_gpu_resources':'NOT_RUN',
            'source_sha256':source_digest(),'train_manifest_sha256':train_sha,'evaluation_manifest_sha256':eval_sha,
            'teacher_cache_manifest_sha256':digest(cache_path),'teacher_identity':teacher,
            'configurations':configs,'checkpoint_manifest_sha256':cp_sha,'export_manifest_sha256':export_sha,
            'export_payload_sha256':digest(export/'weights.pt'),'commands':commands,
            'training_receipts':receipts,'transaction_logs':step_logs,'matched_initialization_and_schedules':True,'benchmark':json.loads(benchmark.read_text()),
            'acceptance_status':assessment['status']}
    (out/'SMOKE_REPORT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'VERIFIED_SOFTWARE_ONLY','receipt':str(out/'SMOKE_REPORT.json')},indent=2))


if __name__=='__main__':
    main()
