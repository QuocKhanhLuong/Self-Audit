import json,os,subprocess,sys,time,shutil
from pathlib import Path

R=Path('/root/self-audit-memory-full-20261024');REPO=Path('/root/Self-Audit-memory-full-20261024')
O=Path('/dev/shm/self-audit-memory-full-20261024/parallel-optimization')
N=R/'native-train-full-declared.json';ASSET=Path('/root/self-audit-teacher-memory-assets')
sys.path.insert(0,str(REPO))
from self_audit.model import digest

def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(temp,path)
def status(stage,**kw):
    d=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    write(R/'OFFLINE_STATUS.json',d);print(json.dumps(d),flush=True)
def cmd(workers,out,bounded):
    c=[sys.executable,'scripts/build_teacher_memory_parallel.py','--native-manifest',str(N),
       '--native-manifest-sha256',digest(N),'--teacher-source',str(ASSET/'CineMA'),
       '--teacher-config',str(ASSET/'config.yaml'),'--teacher-weights',str(ASSET/'mnms_sax_0.safetensors'),
       '--device','cuda','--workers',str(workers),'--ram-limit-gib','120','--out',str(out)]
    if bounded:c+=['--max-patients','12','--max-timepoints','2']
    else:c+=['--write-a1-cache']
    return c
def run(c,log):
    print('COMMAND '+json.dumps(c),flush=True)
    env=dict(os.environ,PYTHONPATH=str(ASSET/'deps'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with log.open('x') as f:
        p=subprocess.Popen(c,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=env)
        write(R/'CURRENT_OFFLINE_COMMAND.json',dict(pid=p.pid,command=c,log=str(log)))
        return p.wait()
def main():
    O.mkdir(parents=True,exist_ok=False);records=[]
    for workers in (2,4,8,12):
        status('BENCHMARKING_PARALLEL_EXTRACTION',workers=workers,inputs='12 largest-geometry TRAIN cines x first2 timepoints; cost only')
        out=O/f'benchmark_w{workers}';log=R/f'parallel-benchmark-w{workers}.log'
        rc=run(cmd(workers,out,True),log)
        if rc:
            record=dict(workers=workers,status='FAILED_RESOURCE_OR_RUNTIME',log=str(log))
        else:
            record=json.loads((out/'BUILD_RECEIPT.json').read_text());record['status']='BENCHMARKED_NOT_PATIENT_QUALITY'
            shutil.copyfile(out/'BUILD_RECEIPT.json',R/f'PARALLEL_BENCHMARK_W{workers}.json')
        records.append(record);write(R/'PARALLEL_BENCHMARK.json',records)
    passing=[r for r in records if r['status']=='BENCHMARKED_NOT_PATIENT_QUALITY']
    if not passing:raise RuntimeError('no safe parallel profile')
    best=max(passing,key=lambda x:x['timepoints_per_second']);workers=best['workers']
    write(R/'SELECTED_OFFLINE_PARALLELISM.json',dict(workers=workers,timepoints_per_second=best['timepoints_per_second'],
        tile_batch_size=1,selection='highest measured steady-state real-volume throughput with >=2GiB GPU margin and <120GiB cgroup RAM',
        benchmark_inputs='12 largest TRAIN cines, 2 timepoints each; no masks accessed',full_bank_complete=False))
    status('BUILDING_FULL_TRAIN_PARALLEL',workers=workers,patients=80,timepoints=2148,
        benchmark_timepoints_per_second=best['timepoints_per_second'],serial_attempt_preserved=True)
    out=O/'full'
    if run(cmd(workers,out,False),R/'parallel-full-build.log'):
        raise RuntimeError('parallel full extraction failed; partial outputs preserved')
    receipt=json.loads((out/'BUILD_RECEIPT.json').read_text())
    assert receipt['coverage_status']=='FULL_TRAIN' and receipt['coverage']['timepoints_used']==2148
    assert receipt['mask_paths_read']==0 and receipt['evaluation_ids_used'] is False
    dest=R/'teacher-bank';dest.mkdir(exist_ok=False)
    for name in ('fixed_bank.npz','BUILD_RECEIPT.json'):shutil.copyfile(out/name,dest/name)
    cache=Path(receipt['a1_cache'])
    status('COMPLETED',bank=str(dest/'fixed_bank.npz'),bank_sha256=digest(dest/'fixed_bank.npz'),
        cache=str(cache),cache_sha256=digest(cache),workers=workers,
        full_coverage=receipt['coverage'],serial_attempt_preserved=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:status('FAILED',error=repr(e));raise
