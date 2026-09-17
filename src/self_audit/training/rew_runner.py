"""Small explicit supervised REW runner. No legacy completion-150 gates.

Uses the existing ACDC preprocessing and patient identity utilities. New REW
checkpoints only; resumes are at COMPLETED epoch boundaries. Bounded probes
never masquerade as full training or as a qualified memory/performance gate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import yaml
from tqdm import tqdm

from ..data.acdc import ACDCDataset, discover_acdc_records, resolve_effective_acdc_splits
from ..data.common import patient_level_split, validate_patient_split
from ..models.read_evaluate_write import ReadEvaluateWriteNet
from ..losses.read_evaluate_write import compute_rew_losses


@dataclass(frozen=True)
class REWConfig:
    schema: str = 'rew_v1'
    seed: int = 42
    image_size: int = 256
    channels: int = 96
    max_turns: int = 3
    tile_size: int = 32
    harm_weight: float = 2.
    min_gain: float = .01
    tau_accept: float = 0.
    checkpoint_reads: bool = True
    pretrained: bool = True
    batch_size: int = 1
    accumulation_steps: int = 8
    num_workers: int = 2
    cache_volumes_per_worker: int = 1
    cpu_threads: int = 4
    amp: str = 'bfloat16'
    epochs: int = 50
    warmup_epochs: int = 5
    encoder_lr: float = 3e-5
    actor_lr: float = 3e-4
    auditor_lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 3.
    proposal_weight: float = .5
    audit_weight: float = 1.
    read_advantage_weight: float = .2
    synthetic_weight: float = .25
    validation_every: int = 1

    def __post_init__(self):
        if self.schema != 'rew_v1': raise ValueError('not a REW v1 config')
        for name in ('seed','image_size','channels','max_turns','tile_size','batch_size','accumulation_steps',
                     'num_workers','cache_volumes_per_worker','cpu_threads','epochs','warmup_epochs','validation_every'):
            value=getattr(self,name)
            if type(value) is not int or value < (0 if name in ('seed','num_workers','warmup_epochs') else 1):
                raise ValueError(f'invalid integer {name}={value!r}')
        if self.max_turns>3 or self.num_workers>8 or self.channels<16 or self.image_size<32:
            raise ValueError('max_turns<=3, workers<=8, channels>=16, image_size>=32 required')
        if self.amp not in ('off','bfloat16','float16'): raise ValueError('invalid amp')
        for name in ('harm_weight','min_gain','tau_accept','encoder_lr','actor_lr','auditor_lr','weight_decay',
                     'grad_clip','proposal_weight','audit_weight','read_advantage_weight','synthetic_weight'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'invalid number {name}')
            if name != 'tau_accept' and value<0: raise ValueError(f'negative {name}')
        if min(self.encoder_lr,self.actor_lr,self.auditor_lr,self.grad_clip)<=0:
            raise ValueError('LR and clipping must be positive')
        if self.proposal_weight<=0 or self.audit_weight<=0:
            raise ValueError('REW requires proposal and auditor training from the start')
        if type(self.checkpoint_reads) is not bool or type(self.pretrained) is not bool:
            raise ValueError('checkpoint_reads/pretrained must be boolean')


def load_config(path: Path) -> REWConfig:
    data=yaml.safe_load(path.read_text())
    if not isinstance(data,dict): raise ValueError('config must be a mapping')
    extra=set(data)-{f.name for f in fields(REWConfig)}
    if extra: raise ValueError(f'unknown REW config fields: {sorted(extra)}')
    return REWConfig(**data)


def atomic_json(path: Path, value) -> None:
    _atomic(path, lambda f: f.write(json.dumps(value,indent=2,allow_nan=False).encode()))


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temp.open('wb') as f:
            write(f);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        temp.unlink(missing_ok=True)


def _append(path: Path, value) -> None:
    with path.open('a') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n');f.flush()


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def source_identity() -> dict:
    root=Path(__file__).resolve().parents[1]
    paths=sorted(root.rglob('*.py'))
    return {p.relative_to(root).as_posix():sha_file(p) for p in paths}


def _worker_init(worker_id: int) -> None:
    torch.set_num_threads(1)
    seed=torch.initial_seed()%2**32
    np.random.seed(seed);random.seed(seed)


def _loader(dataset,cfg: REWConfig,epoch:int,shuffle:bool):
    kwargs=dict(batch_size=cfg.batch_size,shuffle=shuffle,num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available(),worker_init_fn=_worker_init,
                generator=torch.Generator().manual_seed(cfg.seed+epoch),drop_last=False)
    if cfg.num_workers:
        kwargs.update(prefetch_factor=1,persistent_workers=False,multiprocessing_context='spawn')
    return DataLoader(dataset,**kwargs)


def _synthetic_records(root: Path):
    root.mkdir(parents=True,exist_ok=True)
    (root/'volumes').mkdir(exist_ok=True);(root/'masks').mkdir(exist_ok=True)
    rng=np.random.default_rng(91)
    for i in range(6):
        yy,xx=np.mgrid[:32,:32]
        labels=np.zeros((32,32,2),np.int64)
        r=(xx-16)**2+(yy-16)**2
        labels[r<100]=2;labels[r<45]=3;labels[(xx-7)**2+(yy-17)**2<16]=1
        image=labels.astype(np.float32)+rng.normal(0,.1,labels.shape).astype(np.float32)
        np.save(root/'volumes'/f'patient{i:03d}_frame01.npy',image)
        np.save(root/'masks'/f'patient{i:03d}_frame01.npy',labels)


def resolve_data(root: Path,cfg:REWConfig,*,split_policy:str,manifest:Path|None,patient_manifest:Path|None=None):
    records=discover_acdc_records(root)
    if patient_manifest is not None:
        payload=json.loads(patient_manifest.read_text())
        owners={};declared=set()
        for name in ('train','val','test'):
            patients=payload.get(name+'_patients',[])
            if not isinstance(patients,list) or (name in ('train','val') and not patients):
                raise ValueError('patient manifest needs nonempty train_patients and val_patients')
            for patient in patients:
                if not isinstance(patient,str) or patient in owners:
                    raise ValueError('invalid/duplicate/overlapping patient manifest')
                owners[patient]=name;declared.add(patient)
        found={r.patient_id for r in records}
        if declared-found:raise ValueError(f'manifest patients missing from paired data: {sorted(declared-found)[:5]}')
        splits={'train':[],'val':[],'test':[]}
        for record in records:
            name=owners.get(record.patient_id)
            if record.split=='test':
                if name in ('train','val'):raise ValueError('official test patient assigned to train/val')
                name='test'
            if name is None:raise ValueError(f'unassigned paired patient: {record.patient_id}')
            if record.split=='val' and name!='val':raise ValueError('official validation membership conflict')
            splits[name].append(record)
        strategy='existing_patient_manifest;native_frame_ids_retained_without_ED_ES_renaming'
    elif manifest is not None:
        # Training pool may legitimately be subdivided by a patient manifest.
        # Never overwrite explicit validation/test membership.
        relaxed=[replace(r,split='') if r.split=='train' else r for r in records]
        splits=resolve_effective_acdc_splits(relaxed,split_manifest=manifest,seed=cfg.seed).splits
        strategy='explicit_manifest'
    elif split_policy=='preserve':
        splits=resolve_effective_acdc_splits(records,seed=cfg.seed,train_fraction=.8,val_fraction=.1).splits
        strategy='canonical_resolver_80_10_10_or_explicit'
    else:
        tagged={r.split for r in records}
        if 'val' in tagged:
            splits=resolve_effective_acdc_splits(records,seed=cfg.seed).splits
            strategy='existing_validation_preserved'
        else:
            if '' in tagged and len(tagged)>1: raise ValueError('mixed tagged/untagged data; supply a manifest')
            pool=[r for r in records if r.split in ('','train')]
            if len({r.patient_id for r in pool})<3: raise ValueError('need >=3 training patients for an 80/20 holdout')
            members=patient_level_split([r.case_id for r in pool],train_fraction=.8,val_fraction=.1,seed=cfg.seed)
            # Canonical helper keeps two held-out groups; combine them for the
            # explicitly requested development holdout, never official test.
            members['val']=sorted(members['val']+members.pop('test'))
            splits={s:[r for r in pool if r.case_id in set(ids)] for s,ids in members.items()}
            splits['test']=[r for r in records if r.split=='test']
            strategy='explicit_requested_training_pool_80_20;official_test_reserved'
    if not splits.get('train') or not splits.get('val'): raise ValueError('nonempty patient-disjoint train/val required')
    validate_patient_split({s:[r.case_id for r in rs] for s,rs in splits.items()})
    inventory={s:[dict(case_id=r.case_id,patient_id=r.patient_id,image=str(r.image_path.resolve()),
                       mask=str(r.mask_path.resolve())) for r in rs] for s,rs in splits.items()}
    # Hash only training/development content. Test images/masks are not opened
    # by this runner (their filenames only are inventoried/reserved).
    for split in ('train','val'):
        for row in tqdm(inventory[split],desc=f'Hash {split}',unit='volume'):
            row['image_sha256']=sha_file(Path(row['image']))
            row['mask_sha256']=sha_file(Path(row['mask']))
    datasets={s:ACDCDataset(records=splits[s],image_size=cfg.image_size,augment=(s=='train'),
                           foreground_only=False,depth_axis=2,max_cache=cfg.cache_volumes_per_worker)
              for s in ('train','val')}
    return datasets,dict(strategy=strategy,splits=inventory,
        preprocessing='existing VolumeSliceDataset: 0.5/99.5 volume percentiles,zscore,2.5D,depth_axis2',
        validation_grid='resized_3d_per_volume_then_patient_macro',
        test_evaluated=False,patient_manifest_sha256=sha_file(patient_manifest) if patient_manifest else None)


def _model(cfg:REWConfig,synthetic:bool):
    kwargs=dict(channels=cfg.channels,max_turns=cfg.max_turns,tile_size=cfg.tile_size,
                harm_weight=cfg.harm_weight,min_gain=cfg.min_gain,tau_accept=cfg.tau_accept,
                checkpoint_reads=cfg.checkpoint_reads,pretrained=cfg.pretrained)
    if synthetic:
        # Explicit synthetic fixture, never a silent backbone replacement.
        kwargs.update(encoder=nn.Sequential(nn.Conv2d(3,cfg.channels,4,stride=4),nn.GELU()),fpn=nn.Identity())
    return ReadEvaluateWriteNet(**kwargs)


def _amp(cfg,device):
    return torch.autocast(device.type,dtype=torch.bfloat16 if cfg.amp=='bfloat16' else torch.float16,
                          enabled=cfg.amp!='off')


def _resources(device):
    result={}
    try:
        import psutil
        p=psutil.Process();ps=[p]+p.children(recursive=True)
        rss=0
        for q in ps:
            try:rss+=q.memory_info().rss
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        result['rss_sum_process_tree_gib']=rss/2**30
    except (ImportError,OSError): result['rss_sum_process_tree_gib']=None
    if device.type=='cuda':
        result.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                      gpu_name=torch.cuda.get_device_name(device))
    else:result.update(peak_allocated_gib=None,peak_reserved_gib=None)
    return result


def _gradient_norm(parameters):
    values=[p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.


def _rng():
    return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state):
    torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])


@torch.no_grad()
def evaluate(model,dataset,cfg,device,max_batches=None):
    model.eval();stats={};seen=set();limited=False
    loader=_loader(dataset,cfg,0,False)
    for n,batch in enumerate(tqdm(loader,desc='Validation',unit='batch')):
        if max_batches is not None and n>=max_batches:limited=True;break
        with _amp(cfg,device): out=model(batch['image'].to(device,non_blocking=True))
        targets=batch['mask']
        for arm in ('initial','logits'):
            pred=out[arm].argmax(1).cpu()
            for i,case in enumerate(batch['case_id']):
                key=(str(batch['patient_id'][i]),str(case),arm)
                row=stats.setdefault(key,np.zeros((3,3),dtype=np.int64))
                for j,c in enumerate((1,2,3)):
                    p,y=pred[i]==c,targets[i]==c
                    row[j]+=np.array([int((p&y).sum()),int(p.sum()),int(y.sum())])
        for i,case in enumerate(batch['case_id']):
            key=(str(case),int(batch['slice_idx'][i]))
            if key in seen:raise ValueError('duplicate validation unit')
            seen.add(key)
    complete=not limited and len(seen)==len(dataset)
    if not complete:
        return dict(complete=False,scored_slices=len(seen),expected_slices=len(dataset),
                    initial_dice=None,final_dice=None,reason='bounded validation; no partial-volume Dice reported')
    per_patient={}
    for (pid,case,arm),row in stats.items():
        den=row[:,1]+row[:,2]
        scores=np.divide(2*row[:,0],den,out=np.full(3,np.nan),where=den>0)
        per_patient.setdefault((pid,arm),[]).append(scores)
    patient_rows={}
    for (pid,arm),rows in per_patient.items():
        array=np.stack(rows);counts=np.isfinite(array).sum(0)
        cls=np.divide(np.nansum(array,0),counts,out=np.full(3,np.nan),where=counts>0)
        score=float(np.nanmean(cls)) if np.isfinite(cls).any() else None
        patient_rows.setdefault(pid,{})[arm]=score
    vals={arm:[r[arm] for r in patient_rows.values() if r.get(arm) is not None] for arm in ('initial','logits')}
    paired=[(r['initial'],r['logits']) for r in patient_rows.values() if all(r.get(a) is not None for a in vals)]
    return dict(complete=True,scored_slices=len(seen),patients=len(patient_rows),
        initial_dice=float(np.mean(vals['initial'])) if vals['initial'] else None,
        final_dice=float(np.mean(vals['logits'])) if vals['logits'] else None,
        harm_patients=sum(b<a for a,b in paired),paired_patients=len(paired),patient_rows=patient_rows,
        metric='resized-grid 3D class Dice per case, mean per patient/class then foreground macro; both-empty excluded',
        clinical_or_native_surface_claim=False)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    manifests=p.add_mutually_exclusive_group()
    manifests.add_argument('--split-manifest',type=Path)
    manifests.add_argument('--patient-manifest',type=Path,help='Preserve train_patients/val_patients across raw frame vs ED/ES filename conventions')
    p.add_argument('--split-policy',choices=('preserve','train80val20'),default='preserve')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--max-batches',type=int,help='bounded probe cap across this invocation; NOT epochs')
    p.add_argument('--max-val-batches',type=int)
    p.add_argument('--synthetic',action='store_true',help='explicit CPU fixture; never represents ConvNeXt/real data')
    p.add_argument('--resume',type=Path,help='trusted REW last.pt from a completed epoch, same output/config/source/data')
    return p


def run(args):
    job_started=time.perf_counter()
    cfg=load_config(args.config)
    if args.synthetic and args.data_root is not None:raise ValueError('synthetic and real root are mutually exclusive')
    if not args.synthetic and args.data_root is None:raise ValueError('--data-root is required')
    for name in ('max_batches','max_val_batches'):
        v=getattr(args,name)
        if v is not None and v<1:raise ValueError(f'{name} must be positive')
    if args.max_val_batches and not args.max_batches:raise ValueError('bounded validation only permitted in bounded probes')
    if args.synthetic:
        cfg=replace(cfg,pretrained=False,image_size=32,channels=16,num_workers=0,amp='off',
                    batch_size=2,accumulation_steps=2,max_turns=2,epochs=2,warmup_epochs=0)
        if args.device!='cpu':raise ValueError('synthetic fixture requires explicit --device cpu')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    if device.type=='cpu' and cfg.amp!='off':raise ValueError('CPU real runs require amp: off')
    if device.type=='cuda' and cfg.amp=='bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU does not support BF16; explicitly select float16 in a new run config')
    output=args.output.resolve()
    if args.resume is None:
        if output.exists() and any(output.iterdir()):raise ValueError('occupied output; select a fresh --output')
    else:
        if args.resume.resolve()!=output/'last.pt':raise ValueError('resume must be this output/last.pt')
        if args.max_batches:raise ValueError('bounded probe cannot resume a full run')
    output.mkdir(parents=True,exist_ok=True)
    root=args.data_root
    if args.synthetic:root=output/'synthetic';_synthetic_records(root)
    torch.set_num_threads(cfg.cpu_threads)
    torch.manual_seed(cfg.seed);np.random.seed(cfg.seed);random.seed(cfg.seed)
    datasets,data_id=resolve_data(root,cfg,split_policy=('train80val20' if args.synthetic else args.split_policy),manifest=args.split_manifest,patient_manifest=getattr(args,'patient_manifest',None))
    print(f"Split: {data_id['strategy']} | train={len(datasets['train'])} slices, val={len(datasets['val'])} slices",flush=True)
    print(f"REW: batch={cfg.batch_size}, accumulation={cfg.accumulation_steps}, epochs={cfg.epochs}, workers={cfg.num_workers}, AMP={cfg.amp}",flush=True)
    identity=dict(config=asdict(cfg),data=data_id,source=source_identity(),synthetic=args.synthetic,
        device=device.type,torch_version=str(torch.__version__),numpy_version=str(np.__version__),
        cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        dependencies={name:importlib.metadata.version(name) for name in ('PyYAML','numpy')})
    # Full identity is retained in each checkpoint; no old-source exact resume.
    start=0;checkpoint_data=None
    if args.resume:
        checkpoint_data=torch.load(args.resume,map_location='cpu',weights_only=False)
        if checkpoint_data.get('identity')!=identity or not checkpoint_data.get('resumable'):
            raise ValueError('checkpoint/source/config/data mismatch or non-resumable probe')
        start=checkpoint_data['epoch']+1
        if start>=cfg.epochs:raise ValueError('run already complete')
    if checkpoint_data:
        for name,offset in checkpoint_data['log_offsets'].items():
            log=output/name
            if not log.exists() or log.stat().st_size<offset:raise ValueError('checkpointed log is missing/truncated: '+name)
        for name,offset in checkpoint_data['log_offsets'].items():
            with (output/name).open('r+b') as f:f.truncate(offset)
    atomic_json(output/'requested_config.json',asdict(cfg))
    atomic_json(output/'identity.json',identity)
    model=_model(cfg,args.synthetic).to(device)
    groups=[dict(params=list(model.encoder.parameters()),lr=cfg.encoder_lr),
        dict(params=list(model.fpn.parameters())+list(model.initial_head.parameters())+list(model.annotation_expert.parameters()),lr=cfg.actor_lr),
        dict(params=list(model.auditor.parameters()),lr=cfg.auditor_lr)]
    optimizer=torch.optim.AdamW(groups,weight_decay=cfg.weight_decay,betas=(.9,.999),eps=1e-8)
    scaler=torch.amp.GradScaler(device.type,enabled=(cfg.amp=='float16'))
    updates=0
    if checkpoint_data:
        model.load_state_dict(checkpoint_data['model'],strict=True);optimizer.load_state_dict(checkpoint_data['optimizer'])
        scaler.load_state_dict(checkpoint_data['scaler']);updates=checkpoint_data['updates'];_restore_rng(checkpoint_data['rng'])
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    atomic_json(output/'run_status.json',dict(status='running',epochs_completed=start,synthetic=args.synthetic,invocation_id=getattr(args,'invocation_id',None)))
    total_batches=0;started=time.perf_counter();peak_rss=0.;bounded=False;last_validation=None
    batches_per_epoch=math.ceil(len(datasets['train'])/cfg.batch_size)
    steps_per_epoch=math.ceil(batches_per_epoch/cfg.accumulation_steps)
    total_updates=steps_per_epoch*cfg.epochs
    for epoch in range(start,cfg.epochs):
        model.train();optimizer.zero_grad(set_to_none=True);epoch_start=time.perf_counter()
        loader=_loader(datasets['train'],cfg,epoch,True)
        remaining=None if args.max_batches is None else args.max_batches-total_batches
        limit=len(loader) if remaining is None else min(len(loader),remaining)
        bar=tqdm(loader,total=limit,desc=f'REW {epoch+1}/{cfg.epochs}',unit='batch')
        samples=0;last_batch_end=time.perf_counter()
        for index,batch in enumerate(bar):
            if index>=limit:break
            group_start=(index//cfg.accumulation_steps)*cfg.accumulation_steps
            group_end=min(group_start+cfg.accumulation_steps,limit)
            group_samples=min(group_end*cfg.batch_size,len(datasets['train']))-group_start*cfg.batch_size
            x=batch['image'].to(device,non_blocking=True);y=batch['mask'].to(device,non_blocking=True)
            if device.type=='cuda':torch.cuda.synchronize(device)
            tick=last_batch_end;compute_started=time.perf_counter()
            with _amp(cfg,device):
                out=model(x)
                loss,details=compute_rew_losses(model,out,y,proposal_weight=cfg.proposal_weight,audit_weight=cfg.audit_weight,
                    read_advantage_weight=cfg.read_advantage_weight,synthetic_weight=cfg.synthetic_weight)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss; no optimizer step')
            scaler.scale(loss*x.shape[0]/group_samples).backward()
            metrics=details['metrics']
            if index+1==group_end:
                scaler.unscale_(optimizer)
                grads=[p for p in model.parameters() if p.grad is not None]
                if any(not bool(torch.isfinite(p.grad).all()) for p in grads):
                    raise FloatingPointError('nonfinite gradients; refusing partial/NaN update')
                metrics['geometry_grad_norm']=_gradient_norm(model.annotation_expert.geometry.parameters())
                metrics['auditor_grad_norm']=_gradient_norm(model.auditor.parameters())
                torch.nn.utils.clip_grad_norm_(grads,cfg.grad_clip,error_if_nonfinite=True)
                warm=cfg.warmup_epochs*steps_per_epoch
                factor=(updates+1)/warm if warm and updates<warm else .5*(1+math.cos(math.pi*(updates-warm)/max(1,total_updates-warm)))
                for group,lr in zip(optimizer.param_groups,(cfg.encoder_lr,cfg.actor_lr,cfg.auditor_lr)):group['lr']=lr*factor
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);updates+=1
            if device.type=='cuda':torch.cuda.synchronize(device)
            metrics.update(epoch=epoch+1,batch=index+1,seconds=time.perf_counter()-tick,compute_seconds=time.perf_counter()-compute_started,optimizer_steps=updates)
            metrics.update(_resources(device))
            peak_rss=max(peak_rss,metrics.get('rss_sum_process_tree_gib') or 0)
            _append(output/'train_metrics.jsonl',metrics)
            bar.set_postfix(L=f"{metrics['loss']:.3f}",audit=f"{metrics['auditor_loss']:.3f}",
                            accept=f"{metrics['accepted']}/{metrics['attempts']}")
            samples+=x.shape[0];total_batches+=1
            del out,details,loss
            last_batch_end=time.perf_counter()
        # Workers are released before validation to avoid two simultaneous pools.
        del bar,loader
        bounded=args.max_batches is not None and total_batches>=args.max_batches
        epoch_complete=samples==len(datasets['train'])
        if bounded or (epoch+1)%cfg.validation_every==0 or epoch+1==cfg.epochs:
            last_validation=evaluate(model,datasets['val'],cfg,device,args.max_val_batches)
            _append(output/'validation.jsonl',dict(epoch=epoch+1,**last_validation))
        receipt=dict(epoch=epoch+1,train_epoch_complete=epoch_complete,samples=samples,
                     seconds=time.perf_counter()-epoch_start,optimizer_steps=updates,validation=last_validation)
        _append(output/'epoch_metrics.jsonl',receipt)
        log_offsets={}
        for name in ('train_metrics.jsonl','validation.jsonl','epoch_metrics.jsonl'):
            log=output/name
            if log.exists():
                with log.open('ab') as f:f.flush();os.fsync(f.fileno());log_offsets[name]=f.tell()
        payload=dict(log_offsets=log_offsets,schema='rew_checkpoint_v1',identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
                     scaler=scaler.state_dict(),epoch=epoch,updates=updates,rng=_rng(),resumable=epoch_complete and not bounded)
        name='bounded.pt' if bounded else 'last.pt'
        _atomic(output/name,lambda f:torch.save(payload,f))
        if bounded:break
    status=dict(status='completed_bounded' if bounded else 'completed',
        epochs_completed=(epoch+1 if epoch_complete else epoch),configured_epochs=cfg.epochs,
        batches_this_invocation=total_batches,optimizer_steps=updates,
        wall_seconds=time.perf_counter()-started,full_invocation_seconds=time.perf_counter()-job_started,
        setup_seconds=started-job_started,synthetic=args.synthetic,
        validation=last_validation,peak_sampled_process_tree_rss_gib=peak_rss or None,
        numerical_equivalence_to_legacy=False,memory_fit_for_full_run_not_certified=True,**_resources(device))
    atomic_json(output/'run_status.json',status)
    print(json.dumps(status,indent=2,allow_nan=False))
    return status


def main(argv=None):
    args=build_parser().parse_args(argv)
    args.invocation_id=uuid.uuid4().hex
    try:run(args)
    except (Exception,KeyboardInterrupt) as exc:
        # Never write over checkpoints/identity on a failed run. stderr and a
        # side receipt distinguish failure from successful bounded completion.
        receipt=args.output.resolve()/'run_status.json'
        if receipt.exists():
            current=json.loads(receipt.read_text())
            if current.get('invocation_id')==args.invocation_id:
                current.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                atomic_json(receipt,current)
        print(f'REW failed: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise
    return 0
