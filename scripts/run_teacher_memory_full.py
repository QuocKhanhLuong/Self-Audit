#!/usr/bin/env python3
"""Sequential A0-A6 run, complete banks first, matched schedule, frozen scoring.

All assets are local and pinned. No downloads, exact resume, early stopping,
TEST-based selection, concurrent GPU jobs, dataset deletion or AMP occurs.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from self_audit.engine import source_digest
from self_audit.model import digest


def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


class Experiment:
    def __init__(self,root,steps,warmup):
        self.root=root.resolve();self.steps=steps;self.warmup=warmup;self.source=source_digest()
        self.data=Path('/root/self-audit-pilot-artifacts/data/cache/L8_S42/train.json')
        self.model=Path('/root/self-audit-pilot-artifacts/BACKBONE.json')
        self.assets=Path('/root/self-audit-teacher-memory-assets')
        self.shm=Path('/dev/shm/self-audit-memory-full-20261024')
        self.env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        self.env.pop('PYTHONPATH',None)
        for name in ('configs','profiles','runs','evaluation','reports'): (self.root/name).mkdir(exist_ok=True)

    def status(self,stage,**kw):
        obj=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256=self.source,**kw)
        write(self.root/'STATUS.json',obj);print(json.dumps(obj),flush=True)

    def command(self,cmd,log,*,env=None,allow_failure=False):
        assert source_digest()==self.source,'runtime source changed'
        print('COMMAND '+json.dumps(list(map(str,cmd))),flush=True)
        with (self.root/log).open('x') as f:
            p=subprocess.Popen(list(map(str,cmd)),cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=env or self.env)
            write(self.root/'CURRENT_COMMAND.json',dict(pid=p.pid,command=list(map(str,cmd)),log=str(self.root/log)))
            rc=p.wait()
        if rc and not allow_failure:raise RuntimeError(f'{log}: exit {rc}')
        return rc

    def idle_gpu(self):
        while True:
            p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            if not any(s.strip().isdigit() for s in p.stdout.splitlines()):return
            self.status('WAITING_FOR_IDLE_GPU');time.sleep(15)

    def recipe(self,arm,batch,threads,bank=None,cache=None):
        c=json.loads((REPO/f'configs/teacher_memory_pilot/{arm}.json').read_text())
        c.update(steps=self.steps,warmup_steps=self.warmup,seed=42,batch_size=batch,crop_size=256,
                 threads=threads,save_every=max(1,self.steps//2))
        c['model']=json.loads(self.model.read_text())
        if arm in ('A2','A3','A4','A5','A6'):
            c['teacher_memory'].update(bank_path=str(bank),bank_sha256=digest(bank))
        if arm=='A1':c['teacher_memory'].update(cache_path=str(cache),cache_sha256=digest(cache))
        return c

    def profile(self,batch,threads,bank,cache,tag,dense=False):
        self.idle_gpu();out=self.root/'profiles'/tag
        wrapper="import sys,runpy,torch; total=torch.cuda.get_device_properties(0).total_memory; torch.cuda.set_per_process_memory_fraction((total-int(2.5*2**30))/total); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        cmd=[sys.executable,'-c',wrapper,'scripts/profile_teacher_memory.py','--out',out,
             '--model-config',self.model,'--device','cuda','--warmup-updates','1','--measured-updates','3']
        if dense:
            cmd+=['--manufactured-native-cost','--native-batch-size',batch,'--native-crop-size','256','--arms','A2,A3']
        else:
            # Explicit cost-only threshold override forces the complete pseudo path
            # from fresh initialization. Long-run recipes keep threshold0.7.
            cmd+=['--manifest',self.data,'--manifest-sha256',digest(self.data),
                  '--config-dir',self.root/'profile-configs',
                  '--arms','A0,A1,A2,A3','--batches',str(batch),'--pseudo-threshold','0',
                  '--bank',f'A2={bank}={digest(bank)}','--bank',f'A3={bank}={digest(bank)}',
                  '--cache',cache,'--cache-sha256',digest(cache)]
        rc=self.command(cmd,f'profile-{tag}.log',allow_failure=True)
        record=json.loads((out/'PROFILE.json').read_text())
        if rc:
            log=(self.root/f'profile-{tag}.log').read_text()
            assert (record.get('failure',{}).get('cuda_oom') or record.get('status')=='FAILED_OOM'
                    or 'CUDA out of memory' in log or '"status": "FAILED_OOM"' in log),record
            return dict(batch=batch,threads=threads,status='OOM',receipt=str(out/'PROFILE.json'))
        if dense:
            runs=record['runs'];peak=max(r['cuda_peak_reserved_bytes_including_model_and_teacher'] for r in runs)
            return dict(batch=batch,threads=threads,status='PASS',peak_reserved_bytes=peak,receipt=str(out/'PROFILE.json'))
        by={r['arm']:r for r in record['runs']}
        assert set(by)=={'A0','A1','A2','A3'}
        for arm in ('A2','A3'):
            gradients=by[arm]['memory_gradients_after_pseudo_backward']
            assert gradients and any((g.get('projection_grad_norm') or 0)>0
                and (g.get('alpha_gamma_grad_norm') or 0)>0 for g in gradients),'no nonzero real memory gradient'
        # A3 represents the four paired retrieval variants' common operator cost.
        seconds={a:by[a]['counts_and_throughput']['measured_seconds_from_engine_rows']/3 for a in by}
        suite_seconds=seconds['A0']+seconds['A1']+seconds['A2']+4*seconds['A3']
        peak=max(r['cuda_peak_reserved_bytes_including_warmup'] for r in by.values())
        return dict(batch=batch,threads=threads,status='PASS',
            suite_samples_per_second=batch*7/suite_seconds,
            per_arm_seconds=seconds,peak_reserved_bytes=peak,receipt=str(out/'PROFILE.json'))

    def choose_batch(self,bank,cache):
        d=self.root/'profile-configs';d.mkdir()
        for arm in ('A0','A1','A2','A3'):
            c=self.recipe(arm,1,2,bank,cache);write(d/f'{arm}.json',c)
        records=[]
        for batch in (8,16,24,32,40,44,48,52,64):
            self.status('BENCHMARKING_REAL_FULL_TRANSACTIONS',batch=batch)
            rec=self.profile(batch,2,bank,cache,f'b{batch}_t2');records.append(rec)
            write(self.root/'BENCHMARK.json',records)
            if rec['status']=='OOM':break
        candidates=sorted((r for r in records if r['status']=='PASS'),key=lambda r:r['suite_samples_per_second'],reverse=True)
        assert candidates,'no safe real-data profile'
        best=None
        for candidate in candidates:
            stress=self.profile(candidate['batch'],2,bank,cache,f'dense-capacity-b{candidate["batch"]}',dense=True)
            write(self.root/f'DENSE_CAPACITY_B{candidate["batch"]}.json',stress)
            if stress['status']=='PASS':best=candidate;break
        assert best,'no safe full-capacity memory probe'
        for threads in (4,8):
            for arm in ('A0','A1','A2','A3'):
                c=self.recipe(arm,best['batch'],threads,bank,cache);write(d/f'{arm}.json',c)
            rec=self.profile(best['batch'],threads,bank,cache,f'b{best["batch"]}_t{threads}')
            records.append(rec)
            if rec['status']=='PASS' and rec['suite_samples_per_second']>best['suite_samples_per_second']:best=rec
        write(self.root/'BENCHMARK.json',records)
        write(self.root/'SELECTED_RECIPE.json',dict(**best,crop=256,dtype='FP32; no AMP',
            selection='highest measured full-suite throughput estimate across real A0/A1/A2/A3; dense A2/A3 capacity guard',
            actual_training_pseudo_threshold=0.7,profile_pseudo_threshold=0.0,
            safe_memory_policy='2GiB spare plus 512MiB context allowance'))
        return best['batch'],best['threads']

    def run_training(self,arm,c):
        self.idle_gpu();self.status('TRAINING',arm=arm,batch=c['batch_size'],steps=c['steps'])
        assert shutil.disk_usage(self.root).free>3*2**30,'disk reserve below 3GiB'
        cfg=self.root/'configs'/f'{arm}.json';write(cfg,c)
        out=self.root/'runs'/arm
        self.command([sys.executable,'train.py','--device','cuda','--allow-full-run',
            '--config',cfg,'--model-config',self.model,'--data',self.data,
            '--data-sha256',digest(self.data),'--out',out],f'train-{arm}.log')
        report=json.loads((out/'REPORT.json').read_text())
        metadata=json.loads((out/'final/MANIFEST.json').read_text())
        assert source_digest()==self.source==metadata['source_sha256'],'runtime source mismatch'
        assert metadata['data_sha256']==digest(self.data)
        assert report['status']=='COMPLETED' and report['steps']==self.steps
        assert report['ema_updates']==self.steps and report['label_reads_outside_budget']==[]
        h=hashlib.sha256();peak=0
        with (out/'steps.jsonl').open() as f:
            for line in f:
                row=json.loads(line);assert row['supervised']['patient']!=row['audit']['patient']
                h.update(json.dumps({k:row[k] for k in ('supervised','audit','unlabeled')},sort_keys=True).encode())
                peak=max(peak,row.get('cuda_peak_allocated') or 0)
        return dict(arm=arm,report=report,schedule_sha256=h.hexdigest(),peak_allocated_bytes=peak,
                    checkpoint_manifest_sha256=digest(out/'final/MANIFEST.json'))

    def evaluate(self,arm,split,step='final'):
        self.idle_gpu();self.status('EVALUATING_FROZEN_CHECKPOINT',arm=arm,split=split,checkpoint=step)
        data=Path('/root/self-audit-pilot-artifacts/data/cache')/('L8_S42/development.json' if split=='VAL18' else 'TEST50_verified/test.json')
        cp=self.root/'runs'/arm/step;out=self.root/'evaluation'/f'{arm}_{split}_{step}'
        self.command([sys.executable,'scripts/evaluate_frozen_labels.py','--checkpoint',cp,
            '--checkpoint-sha256',digest(cp/'MANIFEST.json'),'--data',data,'--data-sha256',digest(data),
            '--out',out,'--device','cuda','--batch-size','8','--threads','2','--compress-predictions'],f'eval-{arm}-{split}-{step}.log')
        m=json.loads((out/'METRICS.json').read_text())
        return dict(arm=arm,split=split,checkpoint=step,metrics_path=str(out/'METRICS.json'),
            metrics_sha256=digest(out/'METRICS.json'),macro_patient=m['macro_patient'],
            all_class_patient_macro=m['all_class_patient_macro'],foreground_patient_macro=m['foreground_patient_macro'])

    def main(self):
        plan=dict(schema='self_audit.teacher_memory_full.v1',source_sha256=self.source,
            base_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            arms=[f'A{i}' for i in range(7)],steps=self.steps,warmup_steps=self.warmup,seed=42,label_budget=8,crop_size=256,
            matched_inputs_and_sampling=True,train_manifest_sha256=digest(self.data),
            a5_snapshot_rule=f'new matched A0 final step{self.steps}; fixed before any new VAL/TEST outcomes',
            validation='frozen midpoint and final checkpoints; no early stopping or tuning',
            test='all seven preselected final student checkpoints, only after every training arm completes; reused/exposed Drive TEST50',
            primary_reporting='Dice BG/RV/MYO/LV and patient macro over all4 classes; foreground macro retained separately',
            storage='new lossless gzip argmax volumes; checkpoint midpoint+final; no existing data deleted',
            geometry='explicit pinned dataset-level mm declaration for missing raw NIfTI unit tags; metadata-only derived copies',
            cache='complete FP32 external targets in shared RAM; derived/rebuildable, lost on container restart',
            teacher_subject_overlap='UNKNOWN; M&Ms/UKB lineage does not certify ACDC-disjoint subjects')
        write(self.root/'PROTOCOL.json',plan)
        self.status('WAITING_FOR_COMPLETE_TEACHER_BANK')
        while True:
            try:state=json.loads((self.root/'OFFLINE_STATUS.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):time.sleep(5);continue
            if state['stage']=='FAILED':raise RuntimeError('offline full-bank extraction failed')
            if state['stage']=='COMPLETED':break
            time.sleep(15)
        bank=Path(state['bank']);cache=Path(state['cache'])
        assert digest(bank)==state['bank_sha256'] and digest(cache)==state['cache_sha256']
        receipt=json.loads((self.root/'teacher-bank/BUILD_RECEIPT.json').read_text())
        assert receipt['coverage_status']=='FULL_TRAIN' and receipt['coverage']['timepoints_used']==2148
        b,t=self.choose_batch(bank,cache)
        training=[];schedule=None
        for arm in ('A0','A1','A2','A3','A4','A5','A6'):
            active_bank=bank
            if arm=='A5':
                self.idle_gpu();self.status('BUILDING_FULL_A5_FROM_PRESELECTED_A0_FINAL')
                native=self.root/'native-train-full-declared.json';cp=self.root/'runs/A0/final';dest=self.root/'student-bank'
                self.command([sys.executable,'scripts/build_teacher_memory.py','--mode','a5',
                    '--native-manifest',native,'--native-manifest-sha256',digest(native),
                    '--student-checkpoint',cp,'--student-checkpoint-manifest-sha256',digest(cp/'MANIFEST.json'),
                    '--device','cuda','--out',dest],'build-a5-full.log')
                receipt=json.loads((dest/'BUILD_RECEIPT.json').read_text())
                assert receipt['coverage_status']=='FULL_TRAIN' and receipt['mask_paths_read']==0
                active_bank=dest/'fixed_bank.npz'
            c=self.recipe(arm,b,t,active_bank,cache)
            result=self.run_training(arm,c)
            if schedule is None:schedule=result['schedule_sha256']
            assert result['schedule_sha256']==schedule,'patient/frame/slice/crop sampling mismatch'
            training.append(result);write(self.root/'TRAINING_RESULTS.json',training)
        # Freeze every checkpoint choice before any new evaluation is exposed.
        write(self.root/'FROZEN_FINAL_CHECKPOINTS.json',training)
        results=[]
        for arm in ('A0','A1','A2','A3','A4','A5','A6'):
            midpoint=self.root/'runs'/arm/f'step_{self.steps//2:06d}'
            if midpoint.exists():results.append(self.evaluate(arm,'VAL18',midpoint.name))
            results.append(self.evaluate(arm,'VAL18'))
            results.append(self.evaluate(arm,'TEST50'))
            write(self.root/'RESULTS.json',results)
            with (self.root/'RESULTS.csv').open('w') as f:
                fields=['arm','split','checkpoint','BG','RV','MYO','LV','all_class_patient_macro','foreground_patient_macro']
                w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
                for r in results:w.writerow(dict(arm=r['arm'],split=r['split'],checkpoint=r['checkpoint'],
                    **{k:r['macro_patient'][k]['dice'] for k in ('BG','RV','MYO','LV')},
                    all_class_patient_macro=r['all_class_patient_macro'],foreground_patient_macro=r['foreground_patient_macro']))
        self.command([sys.executable,'scripts/check_repo.py'],'check-repo-final.log')
        env=dict(self.env,CUDA_VISIBLE_DEVICES='');self.command([sys.executable,'-m','pytest','-q'],'pytest-final.log',env=env)
        self.status('COMPLETED',results=str(self.root/'RESULTS.csv'),all_seven_arms=True,
            git_publication='Final metrics ready for checked publication; no raw images/weights are added automatically')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--warmup-steps',type=int,default=1000);args=p.parse_args()
    if args.steps<2 or not 0<=args.warmup_steps<args.steps:p.error('invalid full-run budget')
    exp=Experiment(args.root,args.steps,args.warmup_steps)
    # Keep the lock after completion/failure: exact resume is unsupported and a
    # repeated launch must never overwrite the running experiment's journal.
    with (exp.root/'RUN_LOCK.json').open('x') as lock:
        json.dump(dict(pid=os.getpid(),steps=args.steps,warmup_steps=args.warmup_steps,
                       source_sha256=exp.source),lock,indent=2)
    try:exp.main()
    except BaseException as exc:exp.status('FAILED',error=repr(exc));raise
