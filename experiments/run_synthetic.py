#!/usr/bin/env python3
"""Exercise the complete data/train/EMA/allocation/checkpoint pipeline on toys.

These are execution diagnostics, not evidence of patient generalization. The
explicit tiny model and low confidence threshold must not become patient defaults.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,copy,json
from self_audit.engine import DEFAULT,train
from self_audit.data import synthetic_fixture,write_json


def run(out,steps=24):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    m,sha=synthetic_fixture(out/'data')
    records=[]
    for scope,methods in [('head',['supervised','supervised_extra','ema_full','ema_block','sign',
                               'independent_guard','joint_no_class','joint_no_guard','joint_guard']),
                          ('decoder',['ema_block','joint_guard'])]:
        for method in methods:
            c=copy.deepcopy(DEFAULT);c.update(method=method,update_scope=scope,steps=steps,warmup_steps=min(8,steps),
                    pseudo_threshold=.25,ema_decay=.9,body_lr=.001,head_lr=.03,pseudo_lr=.01)
            c['model']['backbone']='tiny_test';folder=out/f'{scope}_{method}'
            report=train(c,m,sha,folder,device='cpu',synthetic=True)
            rows=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
            risks=[r['control']['realized_max_regression'] for r in rows if 'control' in r and 'realized_max_regression' in r['control']]
            records.append({'scope':scope,'method':method,'report':report,
                            'max_observed_audit_regression':max(risks) if risks else None,
                            'patient_quality':'NOT_MEASURED'})
    write_json(out/'SUMMARY.json',{'synthetic':True,'evidence':'software and manufactured diagnostics only','arms':records})
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=24)
    a=p.parse_args();records=run(a.out,a.steps)
    for r in records:print(r['scope'],r['method'],r['report']['pseudo_accepted'],r['max_observed_audit_regression'])
