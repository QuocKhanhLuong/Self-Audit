#!/usr/bin/env python3
"""Manufactured theory evidence. Not medical segmentation results."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import torch
from torch import nn
from self_audit.control import solve_allocation, scalar_allocation, controlled_head_step, make_candidates
from self_audit.data import write_json


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    V=np.array([[.5,.15],[-.07,.02]]);r=np.array([.8,.1])
    A=np.einsum('gi,gj->gij',V,V);b=r[:,None]*V;e=.5*r*r
    result=solve_allocation(A,b,e,ridge=.001,source_ids=['a','b'])
    cases={}
    for name,alpha in [('zero',[0,0]),('all',[1,1]),('independent',[0,1]),('joint',result.alpha)]:
        cases[name]={'alpha':np.asarray(alpha).tolist(),'risk':(.5*(r-V@alpha)**2).tolist()}
    randomized=[]
    for seed in range(50):
        rng=np.random.default_rng(seed);v=rng.normal(0,.15,(3,16,4));rr=rng.normal(0,.3,(3,16))
        aa=np.einsum('gnm,gnk->gmk',v,v);bb=np.einsum('gn,gnm->gm',rr,v);ee=.5*(rr*rr).sum(1)
        sol=solve_allocation(aa,bb,ee,ridge=.001,source_ids=list(range(4)))
        randomized.append({'seed':seed,'status':sol.status,'max_local_regression':float((sol.local_risks-ee).max()),
                           'baseline_mean':float(ee.mean()),'new_mean':float(sol.local_risks.mean()),'alpha':sol.alpha.tolist()})
    h=nn.Conv2d(4,4,1);nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
    f=torch.eye(4).reshape(1,4,2,2);y=torch.tensor([[[0,1],[2,3]]]);q=torch.nn.functional.one_hot(y,4).permute(0,3,1,2).float()
    d,ids,_=make_candidates(h,f,q,.7,4.)
    grouped={}
    for method in ['independent_guard','joint_guard']:
        hh=nn.Conv2d(4,4,1);hh.load_state_dict(h.state_dict())
        grouped[method]=controlled_head_step(hh,f,y,d,ids,method=method)
    report={'scope':'manufactured finite vectors/tensors only; no patient or pretrained evidence',
            'joint_compensation':cases,'tensor_shared_head_compensation':grouped,'randomized_local_problems':randomized,
            'limitations':['Zero audit regression is not unseen-patient improvement.',
              'A wrong or unrepresentative audit label can favor a wrong update.',
              'Class budgets can reject all unlabeled steps; reject-all is not efficacy.',
              'SLSQP feasibility is checked; no exact optimizer certificate is claimed.']}
    write_json(out/'RESULTS.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    a=p.parse_args();r=run(a.out);print(r['joint_compensation'])
