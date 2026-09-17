"""Bounded CPU research harness. GT enters only the actor's segmentation loss.

Audit invocation and RF/trigger optimizer updates have separate counters. Replay
targets are measured before the actor update, from twin paths sharing A0.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import EventAuditModel
from .reference_free_auditor import AuditObservation, rf_loss
from .audit_trigger import trigger_loss
from .replay import AuditReplay
from .losses import segmentation_loss
from .evaluation import class_dice, transition_metrics, json_safe


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    channels: int = 16
    k: int = 8
    batch_size: int = 8
    actor_lr: float = .001
    rf_lr: float = .001
    trigger_lr: float = .001
    budget_fraction: float = .5
    exploration_fraction: float = .25
    cost: float = .001  # intrinsic proxy units, NOT seconds or Dice
    slow_every: int = 4
    replay_capacity: int = 256
    replay_max_age: int = 8
    period: int = 2


def initial_state(output):
    return output.initial if hasattr(output, 'initial') else output.state


def forward_policy(model, image, policy, config, generator, step):
    if policy != 'unguided':
        return model(image, policy=policy, budget_fraction=config.budget_fraction,
                     generator=generator, step=step, period=config.period)
    state = model.actor.start(image)
    following = model.actor.refine(state, torch.zeros_like(state.logits[:, :1]))
    features = model.trigger.state_features(state)
    return SimpleNamespace(initial=state, final_logits=following.logits,
                           decisions=torch.ones(len(image), dtype=torch.bool, device=image.device),
                           predicted_values=model.trigger(features).detach(), trigger_features=features,
                           audit_calls=0, read_calls=2*len(image))


class EventTrainer:
    def __init__(self, config=TrainConfig(), model=None):
        self.config = config
        if config.slow_every < 1 or not 0 <= config.exploration_fraction <= 1:
            raise ValueError('invalid optimization/exploration schedule')
        torch.manual_seed(config.seed)
        self.model = model or EventAuditModel(channels=config.channels, k=config.k)
        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=config.actor_lr)
        self.rf_optimizer = torch.optim.Adam(self.model.auditor.parameters(), lr=config.rf_lr)
        self.trigger_optimizer = torch.optim.Adam(self.model.trigger.parameters(), lr=config.trigger_lr)
        self.replay = AuditReplay(capacity=config.replay_capacity, seed=config.seed+19)
        self.action_generator = torch.Generator().manual_seed(config.seed+29)
        self.probe_generator = torch.Generator().manual_seed(config.seed+39)
        self.counters = dict(actor_updates=0, rf_updates=0, trigger_updates=0, examples_seen=0,
                             policy_audit_calls=0, probe_audit_calls=0, policy_read_calls=0,
                             probe_read_calls=0, quality_evaluations=0, rf_train_examples=0)

    def step(self, image, target, *, policy='learned', step=0):
        """Only target-consuming operation before evaluation is segmentation_loss."""
        self.model.train()
        begin = time.perf_counter()
        for opt in (self.actor_optimizer, self.rf_optimizer, self.trigger_optimizer):
            opt.zero_grad(set_to_none=True)
        output = forward_policy(self.model, image, policy, self.config, self.action_generator, step)
        n_probe, delta = 0, []
        # B0 and the extra-read control have no hidden audit training expense.
        if policy not in ('none', 'unguided'):
            indices = torch.where(torch.rand(len(image), generator=self.probe_generator)
                                  < self.config.exploration_fraction)[0]
            if len(indices):
                twin = self.model.twins(image, state=initial_state(output), indices=indices)
                value = twin['intrinsic_delta'] - self.config.cost
                self.replay.add(twin['observation'], twin['trigger_features'], value, step=step)
                n_probe = len(indices)
                delta = twin['intrinsic_delta'].detach().cpu().tolist()
        loss = segmentation_loss(output.final_logits, target)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite actor loss')
        loss.backward()
        actor_grad = sum(float(p.grad.detach().square().sum()) for p in self.model.actor.parameters() if p.grad is not None)**.5
        window_grad = sum(float(p.grad.detach().square().sum()) for n,p in self.model.actor.named_parameters()
                          if 'generator' in n and p.grad is not None)**.5
        if any(p.grad is not None for p in self.model.auditor.parameters()):
            raise AssertionError('actor loss crossed auditor firewall')
        if any(p.grad is not None for p in self.model.trigger.parameters()):
            raise AssertionError('actor loss crossed trigger firewall')
        self.actor_optimizer.step()
        actor_seconds = time.perf_counter()-begin
        slow_begin = time.perf_counter()
        rf_value, trigger_value = None, None
        if policy not in ('none','unguided') and (step+1) % self.config.slow_every == 0:
            replay = self.replay.sample(self.config.batch_size, current_step=step,
                                        max_age=self.config.replay_max_age)
            if replay is not None:
                self.actor_optimizer.zero_grad(set_to_none=True)
                qloss = rf_loss(self.model.auditor, replay['observation'])
                if not torch.isfinite(qloss): raise FloatingPointError('nonfinite RF loss')
                qloss.backward()
                self.rf_optimizer.step()
                self.counters['rf_updates'] += 1
                self.counters['rf_train_examples'] += len(replay['values'])
                rf_value = float(qloss.detach())
                if policy == 'learned':
                    vloss = trigger_loss(self.model.trigger, replay['features'], replay['values'])
                    if not torch.isfinite(vloss): raise FloatingPointError('nonfinite trigger loss')
                    vloss.backward()
                    self.trigger_optimizer.step()
                    self.counters['trigger_updates'] += 1
                    trigger_value = float(vloss.detach())
                if any(p.grad is not None for p in self.model.actor.parameters()):
                    raise AssertionError('RF/trigger loss crossed actor firewall')
        self.counters['actor_updates'] += 1
        self.counters['examples_seen'] += len(image)
        self.counters['policy_audit_calls'] += int(output.audit_calls)
        self.counters['policy_read_calls'] += int(output.read_calls)
        self.counters['probe_audit_calls'] += n_probe
        self.counters['probe_read_calls'] += n_probe
        self.counters['quality_evaluations'] += 2*n_probe
        return dict(loss=float(loss.detach()), rf_loss=rf_value, trigger_loss=trigger_value,
                    actor_grad_norm=actor_grad, window_grad_norm=window_grad,
                    selected=int(output.decisions.sum()), probes=n_probe, rf_deltas=delta,
                    actor_and_probe_seconds=actor_seconds, slow_update_seconds=time.perf_counter()-slow_begin)


def collate_rows(rows):
    """No optional metadata enters any model API."""
    return {'image':torch.stack([r['image'] for r in rows]),
            'mask':torch.stack([r['mask'] for r in rows]),
            'patient_id':[r['patient_id'] for r in rows],
            'case_id':[r['case_id'] for r in rows],
            'slice_idx':[int(r['slice_idx']) for r in rows],
            'num_slices':[int(r.get('num_slices',1)) for r in rows]}


def batch_indices(length, steps, batch_size, seed):
    """A single predetermined sample stream shared by every policy."""
    if length < batch_size: raise ValueError('dataset smaller than physical batch')
    rng, result = np.random.default_rng(seed), []
    while len(result) < steps*batch_size:
        result.extend(rng.permutation(length).tolist())
    return [result[i*batch_size:(i+1)*batch_size] for i in range(steps)]


@torch.no_grad()
def evaluate(model, dataset, config, *, policy='learned', step=0, include_twins=True):
    """Complete-volume confusion counts, then patient aggregation; no slice CI."""
    model.eval()
    start = time.perf_counter()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_rows)
    rng = torch.Generator().manual_seed(config.seed+101)
    cases, rows, totals = {}, [], dict(policy_audit_calls=0, read_calls=0, probe_audit_calls=0)
    for batch_number,batch in enumerate(loader):
        x,y = batch['image'], batch['mask']
        output = forward_policy(model,x,policy,config,rng,step+batch_number)
        state = initial_state(output)
        a0, final = state.logits.argmax(1), output.final_logits.argmax(1)
        twin = model.twins(x,state=state) if include_twins else None
        audit = twin['audit_logits'].argmax(1) if twin is not None else a0
        totals['policy_audit_calls'] += int(output.audit_calls)
        totals['read_calls'] += int(output.read_calls)
        totals['probe_audit_calls'] += len(x) if twin is not None else 0
        dq = twin['intrinsic_delta'].cpu().tolist() if twin is not None else [None]*len(x)
        for j in range(len(x)):
            cid,pid = batch['case_id'][j],batch['patient_id'][j]
            row=cases.setdefault(cid,{'patient_id':pid,'slices':set(),'expected':batch['num_slices'][j],
                                     'counts':{name:np.zeros((3,2),dtype=np.int64) for name in ('a0','final','audit')},
                                     'q':[],'v':[],'z':[]})
            if batch['slice_idx'][j] in row['slices']: raise ValueError('duplicate validation slice')
            row['slices'].add(batch['slice_idx'][j])
            for name,pred in [('a0',a0[j]),('final',final[j]),('audit',audit[j])]:
                for c in range(1,4):
                    p,t=pred==c,y[j]==c
                    row['counts'][name][c-1] += [int((p&t).sum()),int(p.sum()+t.sum())]
            if dq[j] is not None: row['q'].append(dq[j])
            row['v'].append(float(output.predicted_values[j]));row['z'].append(bool(output.decisions[j]))
    patients={}
    for cid,row in cases.items():
        if row['slices'] != set(range(row['expected'])): raise ValueError('incomplete validation volume '+cid)
        entry={'case_id':cid,'q':float(np.mean(row['q'])) if row['q'] else None,
               'value':float(np.mean(row['v'])),'audit_fraction':float(np.mean(row['z']))}
        for name,count in row['counts'].items():
            entry[name]=np.divide(2*count[:,0],count[:,1],out=np.full(3,np.nan),where=count[:,1]>0)
        patients.setdefault(row['patient_id'],[]).append(entry)
    for pid,entries in sorted(patients.items()):
        row={'patient_id':pid,'case_count':len(entries),
             'rf_delta':float(np.mean([e['q'] for e in entries])) if include_twins else None,
             'predicted_value':float(np.mean([e['value'] for e in entries])),
             'audit_fraction':float(np.mean([e['audit_fraction'] for e in entries]))}
        for name in ('a0','final','audit'):
            row[name+'_class_dice']=np.nanmean([e[name] for e in entries],axis=0).tolist()
            row[name+'_dice']=float(np.nanmean(row[name+'_class_dice']))
        row['true_delta']=row['audit_dice']-row['a0_dice']
        row['retained_delta']=row['final_dice']-row['a0_dice']
        rows.append(row)
    result={'patient_count':len(rows),'volume_count':len(cases),'patients':rows,
            'a0_dice':float(np.mean([r['a0_dice'] for r in rows])),
            'final_dice':float(np.mean([r['final_dice'] for r in rows])),
            'class_dice':np.mean([r['final_class_dice'] for r in rows],axis=0).tolist(),
            'harm_patient_rate':float(np.mean([r['retained_delta'] < 0 for r in rows])),
            'audit_fraction':float(np.mean([r['audit_fraction'] for r in rows])),
            'costs':totals,'evaluation_seconds':time.perf_counter()-start,
            'metric_note':'Patient mean of per-volume foreground Dice on resized complete volumes; no slice CI.'}
    if include_twins:
        result['patient_transition_metrics']=transition_metrics([r['rf_delta'] for r in rows],
            [r['true_delta'] for r in rows],[r['predicted_value'] for r in rows])
        result['patient_transition_metrics']['aggregation_note']='Patient-average RF contrast vs patient volume Dice contrast; trigger actions occur on slices, not patients.'
    return json_safe(result)
