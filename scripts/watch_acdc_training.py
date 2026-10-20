#!/usr/bin/env python3
"""Compact live display of real step receipts; never changes a job."""
import argparse
import json
import time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);a=p.parse_args();last=None
while True:
    logs=list(a.runs.glob('*/steps.jsonl'))
    if logs:
        f=max(logs,key=lambda p:p.stat().st_mtime_ns)
        try:
            with f.open('rb') as h:
                h.seek(max(0,f.stat().st_size-65536));line=h.read().splitlines()[-1]
            r=json.loads(line);key=(str(f),r['step'])
            if key!=last:
                c=r.get('control',{});status=c.get('status','WARMUP' if r['warmup'] else 'SUPERVISED')
                print(f"{f.parent.name} | step {r['step']:5d} | loss {r['supervised_loss']:.4f} | grad {r['gradient_norm']:.3f} | VRAM {r['cuda_peak_allocated']/1024**3:.2f} GiB | elapsed {r['elapsed_seconds']:.1f}s | {status} scale={c.get('scale',0):.3g}",flush=True)
                last=key
        except (ValueError,IndexError,FileNotFoundError):pass
    time.sleep(1)
