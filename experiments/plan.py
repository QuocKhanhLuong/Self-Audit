#!/usr/bin/env python3
"""Write matched experiment recipes and commands. Never launch training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,copy,json,shlex
from self_audit.engine import DEFAULT
from self_audit.data import write_json


def plan(out,steps=10000):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);jobs=[]
    arms=['supervised','supervised_extra','ema_full','ema_block','sign','independent_guard','joint_no_class','joint_no_guard','joint_guard']
    for labels in [4,8,16]:
        for label_seed in [42,123,2026]:
            for method in arms:
                name=f'L{labels}_S{label_seed}_{method}';c=copy.deepcopy(DEFAULT)
                c.update(method=method,update_scope='decoder',steps=steps,warmup_steps=min(1000,steps),seed=42)
                cf=out/(name+'.json');write_json(cf,c)
                jobs.append({'name':name,'label_budget':labels,'label_seed':label_seed,'training_seed':42,
                  'config':str(cf),'train_manifest_pattern':f'CACHE/L{labels}_S{label_seed}/train.json',
                  'command_template':f'python train.py --config {shlex.quote(str(cf))} --model-config BACKBONE.json --data CACHE/L{labels}_S{label_seed}/train.json --data-sha256 VERIFIED_SHA --out RUNS/{name} --device cuda --allow-full-run'})
    write_json(out/'MATRIX.json',{'jobs':jobs,'status':'PLANNED_NOT_RUN','patient_metrics':'NONE',
       'protocol':'nested patient label subsets; audit labels included; fixed seed; no VAL selection',
       'required_external_baselines':['BCP official matched protocol','HCO exact algorithm after full-text review; sign control is NOT HCO'],
       'additional_ablation':'head versus decoder scope; same initialization and pseudo grouping',
       'compute_controls':'supervised_extra is label-use control, not an exact wall-time match; report runtime/JVP/trial overhead and learning curves'})
    return jobs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=10000)
    a=p.parse_args();print(len(plan(a.out,a.steps)),'planned jobs; none launched')
