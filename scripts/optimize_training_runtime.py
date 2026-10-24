#!/usr/bin/env python3
"""Measure then apply wall-time improvements at the next completed-arm boundary."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_teacher_memory_full import write
from scripts.continue_teacher_memory_full import inspect_completed
from self_audit.engine import source_digest
from self_audit.model import digest


def done(pid):
    try: return Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z'
    except FileNotFoundError: return True


def choose_runtime(rows):
    """Batch changes increase work at fixed steps and are diagnostic only."""
    baseline=next(r for r in rows if r['arm']=='A2' and r['batch']==24 and r['threads']==8
                  and not r['cached_inputs'] and r['status']=='PASS')
    cache8=next((r for r in rows if r['arm']=='A2' and r['batch']==24 and r['threads']==8
                 and r['cached_inputs'] and not r.get('prefetch_inputs') and r['status']=='PASS'),None)
    prefetch8=next((r for r in rows if r['arm']=='A2' and r['batch']==24 and r['threads']==8
                    and r.get('prefetch_inputs') and r['status']=='PASS'),None)
    fields=('student_sha256','teacher_sha256','optimizers_sha256','final_rng')
    parity=bool(cache8 and all(baseline['state'][k]==cache8['state'][k] for k in fields))
    prefetch_parity=bool(prefetch8 and all(baseline['state'][k]==prefetch8['state'][k] for k in fields))
    candidates=[r for r in rows if r['arm']=='A2' and r['batch']==24 and r['status']=='PASS'
                and (prefetch_parity if r.get('prefetch_inputs') else (not r['cached_inputs'] or parity))]
    best=min(candidates,key=lambda r:r['seconds_per_step'])
    # Tiny, noisy differences should not change an existing matched run.
    if best['seconds_per_step'] > .97*baseline['seconds_per_step']: best=baseline
    return dict(status='MEASURED_AND_VERIFIED',batch=24,threads=best['threads'],
                cached_inputs=best['cached_inputs'],cache_state_bitwise_parity=parity,
                prefetch_inputs=best.get('prefetch_inputs',False),prefetch_state_bitwise_parity=prefetch_parity,
                baseline_seconds_per_step=baseline['seconds_per_step'],
                selected_seconds_per_step=best['seconds_per_step'],
                estimated_remaining_training_seconds_saved=50000*(baseline['seconds_per_step']-best['seconds_per_step']),
                selection='minimum seconds/step at original batch24; require >=3% improvement; no change to steps or sample exposure')


def main(root):
    out=root/'runtime-optimization';out.mkdir(exist_ok=False)
    parent=int((root/'full-pipeline.pid').read_text())
    child=json.loads((root/'CURRENT_COMMAND.json').read_text())['pid']
    expected=source_digest();handed_off=False;paused=False
    runtime_sources={name:digest(ROOT/'scripts'/name) for name in
                     ('runtime_performance.py','prefetch_runtime.py','profile_runtime_performance.py',
                      'continue_teacher_memory_full.py')}
    def interrupted(signum, frame):
        raise SystemExit(f'interrupted by signal {signum}; original queue retained')
    signal.signal(signal.SIGTERM, interrupted)
    def status(stage,**kwargs):
        obj=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs)
        write(out/'STATUS.json',obj);print(json.dumps(obj),flush=True)
    try:
        assert b'run_teacher_memory_full.py' in Path(f'/proc/{parent}/cmdline').read_bytes()
        assert b'/runs/A1' in Path(f'/proc/{child}/cmdline').read_bytes()
        os.kill(parent,signal.SIGSTOP);paused=True
        status('WAITING_FOR_A1',current_training_untouched=True,parent_pid=parent,child_pid=child,
               target='minimum remaining wall time; matched batch24',thread_candidates=[1,2,4,8,16,24,32,64])
        while not done(child):time.sleep(5)
        # Validate completed science BEFORE taking ownership of the rest of the queue.
        completed=[inspect_completed(root,a,10000,expected) for a in ('A0','A1')]
        assert completed[0]['schedule_sha256']==completed[1]['schedule_sha256']
        while any(line.strip().isdigit() for line in subprocess.check_output(
                ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()):time.sleep(5)
        cpu_receipt=json.loads((root/'REAL_INPUT_CACHE_PREFETCH_BENCHMARK.json').read_text())
        assert cpu_receipt['status']=='PASS'
        rows=[]
        cases=[('A2',24,8,False,False),('A2',24,8,True,False),('A2',24,8,True,True)]+[
            ('A2',24,t,True,True) for t in (1,2,4,16,24,32,64)]+[('A2',48,8,False,False)]
        def probe(arm,batch,threads,cached,prefetch,tag):
            assert source_digest()==expected
            assert all(digest(ROOT/'scripts'/name)==sha for name,sha in runtime_sources.items()), 'runtime helper changed'
            ram=int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text())
            assert ram<120*2**30,'RAM budget reached; keep original queue'
            status('BENCHMARKING',arm=arm,batch=batch,threads=threads,cached_inputs=cached,prefetch_inputs=prefetch)
            c=json.loads((root/'configs/A1.json').read_text())
            c.update(batch_size=batch,threads=threads,steps=40,warmup_steps=0,pseudo_threshold=0.,save_every=1000)
            if arm!='A1':
                bank=root/'teacher-bank/fixed_bank.npz'
                c['teacher_memory'].update(arm=arm,bank_path=str(bank),bank_sha256=digest(bank),cache_path=None,cache_sha256=None)
            cfg=out/f'{tag}.json';write(cfg,c);resultdir=out/tag
            cmd=[sys.executable,'scripts/profile_runtime_performance.py']
            if prefetch:cmd+=['--prefetch-inputs']
            elif cached:cmd+=['--cached-inputs']
            cmd+=['--config',str(cfg),'--data','/root/self-audit-pilot-artifacts/data/cache/L8_S42/train.json',
                '--data-sha256','e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4',
                '--out',str(resultdir),'--device','cuda']
            env=dict(os.environ,OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),PYTHONUNBUFFERED='1');env.pop('PYTHONPATH',None)
            started=time.monotonic()
            with (root/f'runtime-probe-{tag}.log').open('x') as log:
                try:rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=100).returncode
                except subprocess.TimeoutExpired:rc=124
            row=dict(arm=arm,batch=batch,threads=threads,cached_inputs=cached,prefetch_inputs=prefetch,status='PASS' if rc==0 else 'FAILED',
                     returncode=rc,probe_wall_seconds=time.monotonic()-started,cost_only=True,probe_threshold=0.,training_threshold=.7)
            if rc==0:
                data=[json.loads(s) for s in (resultdir/'steps.jsonl').read_text().splitlines()]
                assert len(data)==40
                seconds=data[-1]['elapsed_seconds']-data[9]['elapsed_seconds']
                row.update(measured_steps=30,seconds_per_step=seconds/30,samples_per_second=batch*30/seconds,
                           state=json.loads((resultdir/'RUNTIME_PROBE.json').read_text()))
                row['peak_reserved_bytes']=row['state']['peak_reserved_bytes']
            rows.append(row);write(out/'RESULTS.json',rows)
            return row
        for arm,batch,threads,cached,prefetch in cases:
            probe(arm,batch,threads,cached,prefetch,f'{arm}_b{batch}_t{threads}_cache{int(cached)}_prefetch{int(prefetch)}')
        choice=choose_runtime(rows)
        # Verify paired retrieval also benefits; a cost-only check never looks at VAL/TEST.
        base3=probe('A3',24,8,False,False,'A3_baseline')
        best3=probe('A3',24,choice['threads'],choice['cached_inputs'],choice['prefetch_inputs'],'A3_selected')
        if base3['status']!='PASS' or best3['status']!='PASS' or best3['seconds_per_step']>base3['seconds_per_step']*1.03:
            status('RETAIN_ORIGINAL_RUNTIME',rejected_choice=choice)
            baseline=next(r for r in rows if r['arm']=='A2' and r['batch']==24 and r['threads']==8 and not r['cached_inputs'])
            choice.update(threads=8,cached_inputs=False,prefetch_inputs=False,
                selected_seconds_per_step=baseline['seconds_per_step'],estimated_remaining_training_seconds_saved=0.)
        choice.update(canonical_source_sha256=expected,results_sha256=digest(out/'RESULTS.json'),
                      input_parity_sha256=digest(root/'REAL_INPUT_CACHE_PREFETCH_BENCHMARK.json'),
                      runtime_launcher_sha256=digest(ROOT/'scripts/runtime_performance.py'),
                      prefetch_launcher_sha256=digest(ROOT/'scripts/prefetch_runtime.py'),
                      completed_arms_preserved=['A0','A1'])
        write(root/'PERFORMANCE_OVERRIDE.json',choice)
        # A ready child waits at a gate while the old parent is stopped. Keep the
        # old queue available as fallback until startup validation has succeeded.
        write(out/'HANDOFF.json',dict(old_parent_pid=parent,completed_arms=completed,
              reason='apply measured runtime-only optimization; no partial training/checkpoint resume'))
        gate=out/'HANDOFF_GATE'
        with (root/'full-pipeline.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u','scripts/continue_teacher_memory_full.py','--root',str(root),
                                '--start-gate',str(gate)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS=str(choice['threads']),MKL_NUM_THREADS=str(choice['threads'])))
        deadline=time.monotonic()+30
        while not gate.with_suffix('.ready').exists():
            if p.poll() is not None: raise RuntimeError('continuation startup failed; original queue retained')
            if time.monotonic()>deadline:
                p.terminate();p.wait();raise RuntimeError('continuation readiness timeout; original queue retained')
            time.sleep(.2)
        os.kill(parent,signal.SIGKILL);paused=False
        gate.write_text('start after completed A1\n')
        handed_off=True
        (root/'full-pipeline.pid').write_text(str(p.pid)+'\n')
        status('APPLIED_AT_COMPLETED_ARM_BOUNDARY',new_parent_pid=p.pid,choice=choice)
    except BaseException as exc:
        status('FAILED',error=repr(exc));raise
    finally:
        if paused and not handed_off and Path(f'/proc/{parent}').exists():os.kill(parent,signal.SIGCONT)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root.resolve())
