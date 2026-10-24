#!/usr/bin/env python3
"""Benchmark canonical A5 patient parallelism, then build the complete TRAIN bank."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_teacher_memory_full import write
from self_audit.engine import source_digest
from self_audit.model import digest


def main(root):
    out=root/'a5-parallel-optimization';out.mkdir(exist_ok=False)
    native=root/'native-train-full-declared.json';checkpoint=root/'runs/A0/final'
    expected=source_digest()
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1');env.pop('PYTHONPATH',None)
    def run(workers, target, bounded):
        assert source_digest()==expected
        cmd=[sys.executable,'scripts/build_teacher_memory_parallel.py','--mode','a5',
             '--native-manifest',str(native),'--native-manifest-sha256',digest(native),
             '--student-checkpoint',str(checkpoint),'--student-checkpoint-manifest-sha256',digest(checkpoint/'MANIFEST.json'),
             '--device','cuda','--workers',str(workers),'--ram-limit-gib','120','--out',str(target)]
        if bounded:cmd+=['--max-patients','16','--max-timepoints','2']
        with (root/f'a5-parallel-{target.name}.log').open('x') as log:
            rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc: return dict(status='FAILED_RESOURCE_OR_RUNTIME',workers=workers,returncode=rc)
        row=json.loads((target/'BUILD_RECEIPT.json').read_text());row['status']='PASS'
        return row
    rows=[]
    for workers in (1,2,4,8,12,16):
        row=run(workers,out/f'benchmark_w{workers}',True)
        rows.append(row);write(out/'BENCHMARK.json',rows)
        # A resource failure defines the safe limit; keep its negative receipt.
        if row['status']!='PASS':break
    passing=[r for r in rows if r['status']=='PASS']
    assert passing,'no safe A5 extraction benchmark'
    assert passing[0]['workers']==1,'serial reference required'
    best=max(passing,key=lambda r:r['timepoints_per_second'])
    write(out/'SELECTED.json',dict(workers=best['workers'],timepoints_per_second=best['timepoints_per_second'],
          selection='highest real TRAIN cost throughput, fixed slice batch1, bitwise bank/entry parity with serial reference',
          canonical_source_sha256=expected))
    full=run(best['workers'],root/'student-bank',False)
    assert full['status']=='PASS' and full['coverage_status']=='FULL_TRAIN'
    assert full['coverage']['timepoints_used']==2148 and full['mask_paths_read']==0
    write(out/'COMPLETED.json',full)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root.resolve())
