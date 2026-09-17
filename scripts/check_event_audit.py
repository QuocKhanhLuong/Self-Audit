#!/usr/bin/env python3
"""Frozen-state causal interventions and reference-free semantic red teams."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from self_audit.models.dynamic_window import DynamicWindowAttention
from self_audit_event.model import EventAuditModel
from self_audit_event.reference_free_auditor import AuditObservation
from self_audit_event.data import PhantomDataset, load_acdc
from self_audit_event.training import collate_rows
from self_audit_event.training import batch_indices
from self_audit_event.audit_trigger import trigger_loss
from self_audit_event.evaluation import oracle_delta, transition_metrics, json_safe


@contextmanager
def intervene_source(reader, kind):
    def replace(module,args,value):
        if kind=='zero': return torch.zeros_like(value)
        if kind=='swap': return value.roll(1,0)
        raise ValueError(kind)
    hooks=[reader.key.register_forward_hook(replace),reader.value.register_forward_hook(replace)]
    try: yield
    finally:
        for hook in hooks: hook.remove()


@torch.no_grad()
def run(model,dataset,batch_size=8):
    model.eval()
    reader=next(m for m in model.actor.modules() if isinstance(m,DynamicWindowAttention))
    rows=[];permutations=[]
    for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_rows):
        x,y=batch['image'],batch['mask'];state=model.actor.start(x)
        obs=AuditObservation.from_state(x,state);guide=model.auditor(obs)
        base=model.actor.refine(state,guide)
        variants={
            'guidance_removed':model.actor.refine(state,torch.zeros_like(guide)),
            'guidance_swapped':model.actor.refine(state,guide.roll(1,0)),
            'geometry_shuffled':model.actor.refine(state,guide,coordinate_override=base.coordinates.roll(1,0)),
            'same_coordinates_different_guidance':model.actor.refine(state,torch.zeros_like(guide),coordinate_override=base.coordinates),
        }
        for kind in ('zero','swap'):
            with intervene_source(reader,kind):
                variants['source_'+kind]=model.actor.refine(state,guide,coordinate_override=base.coordinates)
        shuffled_obs=AuditObservation(x.roll(1,0),state.logits.detach())
        qswap=model.auditor(shuffled_obs)
        variants['auditor_image_swapped']=model.actor.refine(state,qswap)
        q0=model.auditor.quality(obs);qb=model.auditor.quality(AuditObservation.from_state(x,base))
        true_base=oracle_delta(state.logits,base.logits,y)
        entropy=-(state.logits.softmax(1)*state.logits.log_softmax(1)).sum(1).mean((1,2))
        predicted=model.trigger(model.trigger.state_features(state))
        learned=model.trigger.select(predicted,.5)
        # Random/entropy use the learned realized cardinality; the oracle may skip negative actions.
        count=int(learned.sum());random=torch.zeros(len(x),dtype=torch.bool)
        rng=torch.Generator().manual_seed(300+len(rows));random[torch.randperm(len(x),generator=rng)[:count]]=True
        ent=torch.zeros_like(random);ent[torch.argsort(entropy,descending=True,stable=True)[:count]]=True
        oracle=torch.zeros_like(random)
        eligible=torch.where(torch.nan_to_num(true_base,nan=-float('inf'))>0)[0]
        oracle[eligible[torch.argsort(true_base[eligible],descending=True,stable=True)[:count]]]=True
        changes={}
        for name,variant in variants.items():
            changes[name]={'logit_mae':(variant.logits-base.logits).abs().mean((1,2,3)),
                'hard_change_fraction':(variant.logits.argmax(1)!=base.logits.argmax(1)).float().mean((1,2)),
                'coordinate_mae':(variant.coordinates-base.coordinates).abs().flatten(1).mean(1),
                'delta_dice_vs_guided':oracle_delta(base.logits,variant.logits,y)}
        # GT below is used ONLY to construct an evaluation oracle counterexample.
        correct=torch.nn.functional.one_hot(y,4).permute(0,3,1,2).float()*20-10
        wrong=correct[:,[0,2,3,1]]
        equal=model.auditor.quality(AuditObservation(x,correct))-model.auditor.quality(AuditObservation(x,wrong))
        bad=oracle_delta(correct,wrong,y)
        for j in range(len(x)):
            row={'patient_id':batch['patient_id'][j],'case_id':batch['case_id'][j],
                 'slice_idx':batch['slice_idx'][j],'rf_delta':float(qb[j]-q0[j]),'true_delta':float(true_base[j]),
                 'predicted_value':float(predicted[j]),'entropy':float(entropy[j]),
                 'actions':{k:bool(z[j]) for k,z in [('learned',learned),('matched_random',random),('matched_entropy',ent),('offline_matched_oracle',oracle)]},
                 'interventions':{k:{key:float(v[j]) for key,v in val.items()} for k,val in changes.items()}}
            rows.append(row);permutations.append({'patient_id':batch['patient_id'][j],'quality_difference':float(equal[j]),'true_delta':float(bad[j])})
    summary=transition_metrics([r['rf_delta'] for r in rows],[r['true_delta'] for r in rows],[r['predicted_value'] for r in rows])
    policies={}
    for name in rows[0]['actions']:
        policies[name]=transition_metrics([r['rf_delta'] for r in rows],[r['true_delta'] for r in rows],
                                           decisions=[r['actions'][name] for r in rows])
    return json_safe({'rows':rows,'state_level_metrics':summary,'matched_frozen_policy_metrics':policies,
        'intervention_means':{name:{key:float(np.nanmean([r['interventions'][name][key] for r in rows]))
             for key in rows[0]['interventions'][name]} for name in rows[0]['interventions']},
        'semantic_permutation':{'max_abs_quality_difference':max(abs(r['quality_difference']) for r in permutations),
                               'mean_true_delta':float(np.nanmean([r['true_delta'] for r in permutations]))},
        'note':'Frozen slice-state diagnostics, descriptive only: no independent-slice confidence intervals. Source hooks alter K/V only; query, geometry and local writer path stay fixed. Oracle uses GT offline only. No labels train any score here.'})


def fit_frozen_trigger(model,dataset,cost=.001):
    """Diagnostic: can the same value head fit fresh targets at one frozen version?

    Fixed 256 random training-image states and 256 optimizer updates. No target
    mask is passed to this routine's scoring or optimization path. This is not
    a convergence result or an adopted new training recipe.
    """
    features=[];values=[]
    for indices in batch_indices(len(dataset),32,8,407):
        x=torch.stack([dataset[i]['image'] for i in indices])
        twin=model.twins(x)
        features.append(twin['trigger_features']);values.append(twin['intrinsic_delta']-cost)
    f,v=torch.cat(features),torch.cat(values)
    with torch.no_grad(): before=float(trigger_loss(model.trigger,f,v))
    optimizer=torch.optim.Adam(model.trigger.parameters(),lr=.001)
    for _ in range(256):
        optimizer.zero_grad(set_to_none=True);loss=trigger_loss(model.trigger,f,v)
        loss.backward();optimizer.step()
    with torch.no_grad(): after=float(trigger_loss(model.trigger,f,v))
    return {'updates':256,'training_states':256,'mse_before':before,'mse_after':after,
            'positive_targets':int((v>0).sum()),'negative_targets':int((v<0).sum()),
            'gt_used_for_fit':False,'actor_auditor_frozen':True,'cost':cost}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data-root');p.add_argument('--patient-manifest');p.add_argument('--image-size',type=int,default=64);p.add_argument('--phantom',action='store_true')
    p.add_argument('--fit-frozen-trigger',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=EventAuditModel(channels=state['config']['channels'],k=state['config']['k']);model.load_state_dict(state['state_dict'])
    if args.phantom:
        data=PhantomDataset(24,args.image_size,100);train=PhantomDataset(64,args.image_size,0)
    else:
        if not args.data_root or not args.patient_manifest: p.error('ACDC root and patient manifest required')
        datasets=load_acdc(args.data_root,args.patient_manifest,image_size=args.image_size)[0]
        data,train=datasets['val'],datasets['train']
    fit=fit_frozen_trigger(model,train,cost=state['config']['cost']) if args.fit_frozen_trigger else None
    result=run(model,data,batch_size=state['config']['batch_size'])
    result['frozen_fit_diagnostic']=fit
    path=Path(args.output)
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},allow_nan=False))


if __name__=='__main__': main()
