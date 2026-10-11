"""Real dependency/checkpoint/NIfTI/device preflight; no test-provider fallback.

A PASS certifies bounded execution, not anatomy accuracy or whole-cohort VRAM.
Reference masks and GT metadata are never opened by this module.
"""
from __future__ import annotations
import argparse
import gc
import importlib
import importlib.metadata
import math
import platform
import shutil
import tempfile
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .core import GuidanceConfig, make_bank
from .io import read_splits, discover, read_cine, read_json, write_json, digest
from .student import GuidanceStudent, guidance_loss, contexts
from .teacher import FrozenCineMA, validate_device, WEIGHT_SHA256, HF_REVISION


def environment_report(device='cuda'):
    result=dict(python=platform.python_version(),platform=platform.platform(),device_requested=device,
                packages={},cuda_available=torch.cuda.is_available(),torch_cuda=torch.version.cuda)
    for package in ('torch','torchvision','numpy','scipy','nibabel','cinema','monai','timm',
                    'huggingface_hub','safetensors','omegaconf'):
        try: result['packages'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: result['packages'][package]=None
    if torch.cuda.is_available():
        result['cuda_devices']=[dict(index=i,name=torch.cuda.get_device_name(i),
            capability=list(torch.cuda.get_device_capability(i)),
            total_memory_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]
    return result


def check_dependencies():
    failures={}
    for name in ('nibabel','cinema.mae.mae','safetensors.torch','huggingface_hub'):
        try: importlib.import_module(name)
        except Exception as exc: failures[name]=f'{type(exc).__name__}: {exc}'
    if failures: raise RuntimeError(f'REAL_DEPENDENCY_IMPORT_FAILED: {failures}')


def inventory(root,dataset,split_manifest,cfg,batch_size=8):
    """Read all native voxel arrays, once, to detect corrupt/NaN cine before a run."""
    splits=read_splits(split_manifest)
    if any(not splits[s] for s in ('train','val','test')):
        raise ValueError('FULL_PIPELINE_REQUIRES_NONEMPTY_TRAIN_VAL_TEST: do not reuse validation as test')
    all_ids=[p for s in ('train','val','test') for p in splits[s]]
    records=discover(root,dataset,all_ids); details={}; hashes={}; cache_bytes=0
    for pid,path in records.items():
        cine,meta=read_cine(path)
        partition=next(s for s in splits if pid in splits[s])
        if meta['image_sha256'] in hashes:
            raise ValueError(f'duplicate image bytes for {pid} and {hashes[meta["image_sha256"]]}; check patient split')
        hashes[meta['image_sha256']]=pid
        nodes=int(cine.shape[0]*cine.shape[1]*cfg.grid**2)
        if partition=='train' and nodes>cfg.max_nodes:
            raise ValueError(f'{pid}: {nodes} graph nodes exceed configured {cfg.max_nodes}; no silent grid reduction')
        # Rough uncompressed payload, not an exact disk/RAM promise.
        estimate=int(cine.nbytes+nodes*(cfg.feature_dim*2+4*4*2+4+1))
        if partition=='train': cache_bytes+=estimate
        details[pid]={**meta,'split':partition,'samples':int(cine.shape[0]*cine.shape[1]),
                      'graph_nodes':nodes,'uncompressed_bank_bytes_estimate':estimate}
        print(f'[PREFLIGHT/DATA] {pid} split={partition} shape={meta["shape_xyzt"]} nodes={nodes}',flush=True)
        del cine
    train_samples=sum(details[p]['samples'] for p in splits['train'])
    steps=sum(math.ceil(details[p]['samples']/batch_size) for p in splits['train'])
    return dict(splits=splits,images=details,source_split_sha256=digest(split_manifest),
                split_counts={s:len(v) for s,v in splits.items()},train_samples=train_samples,
                train_steps_per_epoch=steps,uncompressed_bank_bytes_estimate=cache_bytes)


def probe_student(cine,features,device='cpu',batch_size=8,width=32,lr=3e-4,steps=2):
    """Bounded real backward/reload using feature-only targets, not invented anatomy.

    Fabricated tensors may be passed by unit tests, never by the production CLI.
    Repeated samples fill the requested memory batch; this is not training data.
    """
    d=validate_device(device)
    if type(batch_size) is not int or batch_size<1 or type(steps) is not int or steps<1:
        raise ValueError('invalid probe budget')
    feature_dim=features.shape[-1]
    model=GuidanceStudent(feature_dim,width).to(d).train()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    available=list(np.ndindex(cine.shape[:2]))
    indices=np.array([available[i%len(available)] for i in range(batch_size)])
    t,z=indices[:,0],indices[:,1]
    x=torch.from_numpy(contexts(cine,indices)).to(d)
    target=torch.from_numpy(features[t,z].astype(np.float32)).permute(0,3,1,2).to(d)
    q=torch.full((len(indices),4,*target.shape[-2:]),.25,device=d)
    reliability=torch.zeros((len(indices),*target.shape[-2:]),device=d)
    original=model.enc0[0].weight.detach().clone(); loss_values=[]; grad_values=[]
    if d.type=='cuda': torch.cuda.reset_peak_memory_stats(d)
    start=time.perf_counter()
    for _ in range(steps):
        out=model(x); fl,sl,active=guidance_loss(out,target,q,reliability,False)
        if active or sl.item()!=0 or not torch.isfinite(fl):
            raise RuntimeError('invalid feature-only preflight target')
        opt.zero_grad(set_to_none=True); fl.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(norm) or norm<=0:
            raise RuntimeError('NO_FINITE_NONZERO_STUDENT_GRADIENT')
        for p in model.semantic_parameters(): p.grad=None
        opt.step();loss_values.append(float(fl.detach()));grad_values.append(float(norm))
    if torch.equal(original,model.enc0[0].weight.detach()):
        raise RuntimeError('student encoder did not update')
    model.eval()
    with torch.inference_mode(): before=model(x)['features'].detach().cpu()
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'probe.pt';torch.save(model.state_dict(),path)
        other=GuidanceStudent(feature_dim,width)
        other.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        other=other.eval().to(d)
        with torch.inference_mode(): after=other(x)['features'].detach().cpu()
    if not torch.allclose(before,after,atol=1e-6,rtol=1e-5):
        raise RuntimeError('student checkpoint roundtrip mismatch')
    if d.type=='cuda': torch.cuda.synchronize(d)
    return dict(executed_steps=steps,batch_size=batch_size,loss=loss_values,gradient_norm=grad_values,
                encoder_updated=True,checkpoint_roundtrip=True,elapsed_seconds=time.perf_counter()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(d) if d.type=='cuda' else None,
                sample_repetitions_for_memory_test=max(0,batch_size-len(available)),
                semantic_quality_tested=False)


def run_preflight(root,dataset,split_manifest,out,config=None,device='cuda',batch_size=8,
                  tile_batch=1,width=32,checkpoint=None,teacher_config=None,offline=False,threads=4):
    report_path=Path(out)
    if report_path.exists(): raise FileExistsError(report_path)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report=dict(schema='self_audit.real_preflight.v1',status='RUNNING',checks={},
                environment=environment_report(device),gt_read=False,
                scope='run_in_the_environment_invoking_this_command',
                quality_evaluated=False)
    start=time.perf_counter(); stage='arguments'
    def record(name,value):
        report['checks'][name]=value;write_json(report_path,report)
    try:
        cfg=GuidanceConfig(**(read_json(config) if config else {}))
        if min(batch_size,tile_batch,threads)<1: raise ValueError('positive batch/thread budgets required')
        d=validate_device(device);torch.set_num_threads(threads);torch.manual_seed(42)
        stage='dependencies';check_dependencies();record(stage,{'passed':True})
        stage='native_cine_inventory';data=inventory(root,dataset,split_manifest,cfg,batch_size);record(stage,data)
        free=shutil.disk_usage(report_path.parent).free
        record('storage',dict(free_bytes=free,uncompressed_bank_bytes_estimate=data['uncompressed_bank_bytes_estimate']))
        if free<data['uncompressed_bank_bytes_estimate']*1.25:
            raise RuntimeError('INSUFFICIENT_FREE_DISK_FOR_ESTIMATED_BANK_PLUS_MARGIN')
        stage='real_teacher'
        teacher=FrozenCineMA(device,checkpoint,teacher_config,offline=offline)
        # A deterministic TRAIN sample with maximal in-plane area, not best anatomy.
        pid=max(data['splits']['train'],key=lambda p:(np.prod(data['images'][p]['shape_xyzt'][:2]),p))
        cine,meta=read_cine(data['images'][pid]['raw_image'])
        sample=cine[:2].copy()  # all Z slices, two genuine frames; no label-based crop
        if d.type=='cuda': torch.cuda.reset_peak_memory_stats(d)
        f=teacher.extract(sample,grid=cfg.grid,feature_dim=cfg.feature_dim,tile_batch=tile_batch)
        g=teacher.extract(sample[:1],grid=cfg.grid,feature_dim=cfg.feature_dim,tile_batch=tile_batch)
        diff=float(np.max(np.abs(f[:1].astype(float)-g.astype(float))))
        if diff>2e-3 or not np.isfinite(f).all() or not (np.linalg.norm(f.astype(float),axis=-1)>0).all():
            raise RuntimeError('unstable or empty frozen-teacher features')
        record(stage,dict(passed=True,provenance=teacher.provenance,patient_id=pid,
                          input_shape=list(sample.shape),feature_shape=list(f.shape),repeat_max_abs_error=diff,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(d) if d.type=='cuda' else None))
        del teacher;gc.collect()
        if d.type=='cuda': torch.cuda.empty_cache()
        stage='bank_solve'
        small=F.interpolate(torch.from_numpy(sample).flatten(0,1)[:,None],(cfg.grid,cfg.grid),mode='bilinear',align_corners=False)
        bank,bank_report=make_bank(small[:,0].reshape(*sample.shape[:2],cfg.grid,cfg.grid).numpy(),f,meta['patient_left_axis'],cfg)
        record(stage,bank_report)
        stage='student_backward_reload'
        record(stage,probe_student(sample,f,device,batch_size,width))
        report['status']='PASS_EXECUTION_ONLY_QUALITY_UNVERIFIED'
    except Exception as exc:
        report.update(status='BLOCKED',failed_stage=stage,error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc())
    report['elapsed_seconds']=time.perf_counter()-start
    report['expected_teacher_sha256']=WEIGHT_SHA256;report['expected_model_revision']=HF_REVISION
    write_json(report_path,report)
    print('[PREFLIGHT]',report['status'],'report=',str(report_path),flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--dataset',choices=['acdc','mnms'],default='acdc')
    p.add_argument('--split-manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--config')
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--tile-batch',type=int,default=1);p.add_argument('--width',type=int,default=32)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--checkpoint');p.add_argument('--teacher-config')
    p.add_argument('--offline',action='store_true')
    result=run_preflight(**vars(p.parse_args(argv)))
    return 0 if result['status'].startswith('PASS_') else 1
