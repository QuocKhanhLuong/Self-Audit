#!/usr/bin/env python3
"""Verify and package completed real experiment evidence without binary assets.

Replay Dice from frozen uint8 predictions and hash-pinned evaluation references;
no inference, training, tuning or checkpoint selection takes place here.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from self_audit.engine import source_digest
from self_audit.evaluation import confusion_matrix, class_metrics
from self_audit.model import digest


def read(path):return json.loads(path.read_text())
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
def csvwrite(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def package(root,out):
    assert read(root/'STATUS.json')['stage']=='COMPLETED'
    out.mkdir(parents=True,exist_ok=False)
    expected=read(root/'PROTOCOL.json')['source_sha256']
    assert source_digest()==expected
    training=read(root/'TRAINING_RESULTS.json');assert len(training)==7
    evaluations=read(root/'RESULTS.json');assert len(evaluations)==21
    for name in ('PROTOCOL.json','STATUS.json','RESULTS.json','RESULTS.csv','TRAINING_RESULTS.json',
                 'FROZEN_FINAL_CHECKPOINTS.json','SELECTED_RECIPE.json','PERFORMANCE_OVERRIDE.json',
                 'BENCHMARK.json','CPU_THREAD_AUDIT.json','PARALLEL_BENCHMARK.json',
                 'OFFLINE_STATUS.json','DENSE_CAPACITY_B24.json','RESOURCE_USAGE.csv',
                 'check-repo-final.log','pytest-final.log'):
        copy(root/name,out/name)
    for source,target in (('runtime-optimization/RESULTS.json','RUNTIME_BENCHMARK.json'),
                          ('a5-parallel-optimization/BENCHMARK.json','A5_BENCHMARK.json'),
                          ('a5-parallel-optimization/SELECTED.json','A5_SELECTED.json'),
                          ('teacher-bank/BUILD_RECEIPT.json','TEACHER_BANK_RECEIPT.json'),
                          ('student-bank/BUILD_RECEIPT.json','A5_BANK_RECEIPT.json')):
        copy(root/source,out/target)
    data=Path('/root/self-audit-pilot-artifacts/data/cache')
    manifests={'VAL18':data/'L8_S42/development.json','TEST50':data/'TEST50_verified/test.json'}
    assets={};checkpoint_hashes=[];compute=[];all_schedule=[];baseline_config=None
    resource=[]
    for row in csv.reader((root/'RESOURCE_USAGE.csv').open()):
        if len(row)>=5:resource.append((datetime.fromisoformat(row[0].replace('Z','+00:00')).timestamp(),int(row[1]),int(row[3]),int(row[4])))
    for item in training:
        arm=item['arm'];run=root/'runs'/arm;config=read(run/'CONFIG.json');startup=read(run/'STARTUP.json')
        assert config['steps']==10000 and config['warmup_steps']==1000 and config['seed']==42
        assert config['batch_size']==24 and config['crop_size']==256 and config['threads']==8
        normalized={k:v for k,v in config.items() if k!='teacher_memory'}
        if baseline_config is None:baseline_config=normalized
        assert normalized==baseline_config,'non-memory scientific settings differ'
        assert startup['source_sha256']==expected and startup['device']=='cuda' and startup['synthetic'] is False
        assert item['report']['status']=='COMPLETED' and item['report']['steps']==10000
        assert item['report']['supervised_updates']==item['report']['ema_updates']==10000
        assert item['report']['pseudo_accepted']==9000 and not item['report']['label_reads_outside_budget']
        h=hashlib.sha256();peak=0;compact=[];count=0
        for line in (run/'steps.jsonl').open():
            row=json.loads(line);count+=1
            assert row['step']==count and row['supervised']['patient']!=row['audit']['patient']
            assert math.isfinite(row['supervised_loss']) and math.isfinite(row['gradient_norm'])
            h.update(json.dumps({k:row[k] for k in ('supervised','audit','unlabeled')},sort_keys=True).encode())
            peak=max(peak,row['cuda_peak_allocated'])
            if count==1 or count%100==0:
                compact.append({k:row.get(k) for k in ('step','elapsed_seconds','supervised_loss','gradient_norm',
                                                      'pseudo_loss','pseudo_gradient_norm','warmup','cuda_peak_allocated')})
        assert count==10000 and h.hexdigest()==item['schedule_sha256']
        all_schedule.append(h.hexdigest())
        target=out/'logs'/f'{arm}_steps_every100.jsonl';target.parent.mkdir(exist_ok=True)
        target.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in compact))
        copy(run/'CONFIG.json',out/'configs'/f'{arm}.json')
        copy(run/'STARTUP.json',out/'training'/f'{arm}_STARTUP.json')
        copy(run/'REPORT.json',out/'training'/f'{arm}_REPORT.json')
        copy(root/f'train-{arm}.log',out/'logs'/f'train-{arm}.log')
        for step in ('step_005000','final'):
            cp=run/step;meta=read(cp/'MANIFEST.json');manifest_sha=digest(cp/'MANIFEST.json')
            assert meta['source_sha256']==expected and meta['config']==config
            assert meta['step']==(5000 if step=='step_005000' else 10000)
            assert digest(cp/'weights.pt')==meta['payload_sha256'],'checkpoint payload mismatch'
            if step=='final':assert manifest_sha==item['checkpoint_manifest_sha256']
            copy(cp/'MANIFEST.json',out/'checkpoints'/f'{arm}_{step}_MANIFEST.json')
            checkpoint_hashes.append(dict(arm=arm,checkpoint=step,manifest_sha256=manifest_sha,
                weights_sha256=meta['payload_sha256'],source_sha256=expected,data_sha256=meta['data_sha256'],
                server_path=str(cp)))
        for key in ('model',):
            model=config[key];p=Path(model['weights'])
            if str(p) not in assets:
                assert digest(p)==model['weights_sha256']
                assets[str(p)]=model['weights_sha256']
        tm=config['teacher_memory']
        if tm['bank_path'] and tm['bank_path'] not in assets:
            assert digest(tm['bank_path'])==tm['bank_sha256'];assets[tm['bank_path']]=tm['bank_sha256']
        if tm['cache_path']:
            assert digest(tm['cache_path'])==tm['cache_sha256'];assets[tm['cache_path']]=tm['cache_sha256']
        begin=(run/'STARTUP.json').stat().st_mtime;end=(run/'REPORT.json').stat().st_mtime
        samples=[row for row in resource if begin<=row[0]<=end]
        compute.append(dict(arm=arm,training_seconds=item['report']['elapsed_seconds'],
            training_hours=item['report']['elapsed_seconds']/3600,peak_torch_allocated_GiB=peak/2**30,
            sampled_peak_device_used_GiB=max(row[2] for row in samples)/1024,
            sampled_peak_cgroup_RAM_GiB=max(row[1] for row in samples)/2**30,
            resource_samples=len(samples),cuda_step_log_sha256=digest(run/'steps.jsonl')))
        print('Verified training and checkpoints '+arm,flush=True)
    assert len(set(all_schedule))==1
    csvwrite(out/'CHECKPOINT_HASHES.csv',checkpoint_hashes)
    csvwrite(out/'COMPUTE.csv',compute)
    verified=[];prediction_count=0;frame_count=0
    for item in evaluations:
        arm,split,step=item['arm'],item['split'],item['checkpoint']
        original=Path(item['metrics_path']);folder=original.parent
        assert digest(original)==item['metrics_sha256']
        m=read(original);recipe=read(folder/'RECIPE.json');frozen=read(folder/'PREDICTIONS_FROZEN.json')
        assert digest(folder/'RECIPE.json')==frozen['recipe_sha256']
        manifest=manifests[split];assert digest(manifest)==recipe['data_sha256']
        assert digest(root/'runs'/arm/step/'MANIFEST.json')==recipe['checkpoint_sha256']
        dm=read(manifest);records=dm['patients'];inventory={p['id']:p for p in frozen['files']}
        assert len(inventory)==len(records)==(18 if split=='VAL18' else 50)
        rows={(f['patient'],f['frame']):f for f in m['frames']};patients={};total=np.zeros((4,5),np.int64)
        for record in records:
            pid=record['id'];p=folder/'predictions'/inventory[pid]['file']
            assert digest(p)==inventory[pid]['sha256']
            with gzip.open(p,'rb') as stream:pred=np.load(stream,allow_pickle=False)
            assert pred.dtype==np.uint8
            values={name:[] for name in ('BG','RV','MYO','LV')}
            for frame,reference in record['masks'].items():
                path=Path(reference['path']);path=path if path.is_absolute() else manifest.parent/path
                assert digest(path)==reference['sha256']
                y=np.load(path,allow_pickle=False);cm=confusion_matrix(y,pred[int(frame)])
                prior=rows[(pid,int(frame))];assert cm.tolist()==prior['raw_confusion']
                raw=class_metrics(cm);assert raw==prior['raw'];total+=cm
                for j,name in enumerate(values):
                    if raw[j]['dice'] is not None:values[name].append(raw[j]['dice'])
                frame_count+=1
            patients[pid]={k:float(np.mean(v)) if v else None for k,v in values.items()}
            prediction_count+=1
        assert patients==m['patients'] and total.tolist()==m['pooled_confusion']
        for classes,key in ((('BG','RV','MYO','LV'),'all_class_patient_macro'),(('RV','MYO','LV'),'foreground_patient_macro')):
            score=float(np.mean([np.mean([p[c] for c in classes if p[c] is not None]) for p in patients.values()]))
            assert abs(score-m[key])<1e-14 and m[key]==item[key]
        for name in ('METRICS.json','RECIPE.json','PREDICTIONS_FROZEN.json'):
            copy(folder/name,out/'evaluation'/folder.name/name)
        copy(root/f'eval-{arm}-{split}-{step}.log',out/'logs'/f'eval-{arm}-{split}-{step}.log')
        verified.append(dict(arm=arm,split=split,checkpoint=step,metrics_sha256=digest(original),
            recipe_sha256=digest(folder/'RECIPE.json'),frozen_predictions_receipt_sha256=digest(folder/'PREDICTIONS_FROZEN.json'),
            patients=len(patients),frames=len(rows),elapsed_seconds=m['elapsed_seconds']))
        print('Replayed frozen Dice '+folder.name,flush=True)
    runtime=read(root/'runtime-optimization/RESULTS.json')
    csvwrite(out/'RUNTIME_BENCHMARK.csv',[dict(arm=r['arm'],batch=r['batch'],threads=r['threads'],
        cached_inputs=r['cached_inputs'],prefetch_inputs=r['prefetch_inputs'],status=r['status'],
        seconds_per_step=r.get('seconds_per_step'),samples_per_second=r.get('samples_per_second'),
        peak_reserved_GiB=(r.get('peak_reserved_bytes') or 0)/2**30) for r in runtime])
    offline=[]
    for kind,name in (('teacher','PARALLEL_BENCHMARK.json'),('A5','A5_BENCHMARK.json')):
        for r in read(out/name):
            offline.append(dict(kind=kind,workers=r['workers'],status=r['status'],
                timepoints_per_second=r.get('timepoints_per_second'),processing_seconds=r.get('processing_seconds'),
                peak_device_used_GiB=(r.get('peak_device_used_bytes') or 0)/2**30,
                peak_cgroup_RAM_GiB=(r.get('peak_cgroup_ram_bytes') or 0)/2**30))
    csvwrite(out/'OFFLINE_BENCHMARK.csv',offline)
    write(out/'VERIFIED_EVIDENCE.json',dict(status='PASS',verified_utc=datetime.now(timezone.utc).isoformat(),
        canonical_source_sha256=expected,shared_schedule_sha256=all_schedule[0],
        training_steps=70000,checkpoints_hashed=len(checkpoint_hashes),
        frozen_prediction_files_hashed_and_replayed=prediction_count,evaluation_frames_replayed=frame_count,
        assets_hashed=assets,evaluations=verified,
        retained_on_server_only='patient arrays, frozen prediction arrays, checkpoint and bank tensor payloads',
        replay='CPU confusion matrices and patient aggregation replayed from all frozen predictions and hash-checked refs; no new inference'))
    write(out/'FILE_HASHES.json',{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();package(a.root.resolve(),a.out.resolve())
