#!/usr/bin/env python3
"""Actual full meta transactions on pinned TRAIN data; never patient-quality evidence.

Starts from a fresh-run adaptive pilot checkpoint, not an older segmentation model.
Includes L update, EMA target, second-order virtual AdamW, geometry update, real
pseudo update and one EMA update. Threshold and every model parameter stay intact.
"""
import argparse, copy, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from self_audit.data import TrainingData, write_json
from self_audit.model import build_model, digest
from self_audit.engine import source_digest, supervised_step, full_pseudo_step, ema_update
from self_audit.losses import aligned_flip, strong_intensity
from self_audit.meta_window import geometry_named_parameters, meta_geometry_gradients, _present_foreground_dice


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('new output required')
    total=torch.cuda.get_device_properties(0).total_memory
    # Enforce the requested 2 GiB margin even for probes that approach OOM.
    torch.cuda.set_per_process_memory_fraction((total-2*2**30)/total)
    torch.set_num_threads(a.threads); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    cm=json.loads((a.checkpoint/'MANIFEST.json').read_text()); c=cm['config']
    assert cm['source_sha256']==source_digest()
    assert digest(a.checkpoint/'weights.pt')==cm['payload_sha256']
    assert c['method']=='ema_full_meta_window' and c['crop_size']==256
    data=TrainingData(a.data,digest(a.data));rng=np.random.default_rng(42)
    model,prov=build_model(c['model'],window=c['window']);teacher=copy.deepcopy(model)
    payload=torch.load(a.checkpoint/'weights.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(payload['student']);teacher.load_state_dict(payload['teacher'])
    model=model.cuda();teacher=teacher.cuda().eval().requires_grad_(False)
    geo=dict(geometry_named_parameters(model))
    body=[v for n,v in model.named_parameters() if not n.startswith('head.') and n not in geo]
    opts=[torch.optim.AdamW(body,lr=c['body_lr'],weight_decay=c['weight_decay'],foreach=False),
          torch.optim.SGD(model.head.parameters(),lr=c['head_lr'],foreach=False)]
    gopt=torch.optim.SGD(geo.values(),lr=c['meta_window']['lr'])
    for opt,state in zip(opts+[gopt],payload['optimizers'],strict=True):opt.load_state_dict(state)
    del payload
    rows=[]; result=dict(status='RUNNING',batch_size=a.batch,threads=a.threads,crop_size=256,
        dtype='float32; no AMP',source_sha256=source_digest(),backbone=prov,
        checkpoint_manifest_sha256=digest(a.checkpoint/'MANIFEST.json'),data_sha256=digest(a.data),
        device_total_bytes=total,margin_bytes=2*2**30,transactions=rows,
        throughput_definition='batch_size / full_transaction_seconds; three sampled role batches per transaction')
    try:
        # First full transaction warms kernels; remaining transactions are measured.
        for step in range(a.repeats+1):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            li=step%len(data.lids);ai=(li+1+int(rng.integers(len(data.lids)-1)))%len(data.lids)
            xs,ys,sm,sv=data.sample_batch(li,rng,a.batch,labeled=True,crop_size=256,return_valid=True)
            xa,ya,am,av=data.sample_batch(ai,rng,a.batch,labeled=True,crop_size=256,return_valid=True)
            xu,um,uv=data.sample_batch(step%len(data.uids),rng,a.batch,labeled=False,crop_size=256,return_valid=True)
            assert sm['patient']!=am['patient']
            xs,ys,sv=aligned_flip(xs,ys,rng,sv)
            data_seconds=time.perf_counter()-start
            xs,ys,sv,xa,ya,av,xu,uv=[v.cuda() for v in (xs,ys,sv,xa,ya,av,xu,uv)]
            strong=strong_intensity(xu,rng)
            sl,sg=supervised_step(model,opts,xs,ys,valid_mask=sv,freeze_geometry=True)
            model.eval()
            with torch.no_grad():q=teacher(xu,valid_mask=uv).softmax(1)
            conf,py=q.max(1);valid=(conf>=c['pseudo_threshold'])&uv
            assert valid.any(),'empty pseudo support is not a full meta-step benchmark'
            grads,meta=meta_geometry_gradients(model,opts,strong,py.detach(),valid.detach(),xa,ya,input_valid=uv,audit_valid=av,clip_norm=5.)
            assert grads and meta['geometry_grad_norm']>0,'no nonzero meta-gradient'
            gopt.zero_grad(set_to_none=True);before={n:v.detach().clone() for n,v in geo.items()}
            for n,v in geo.items():v.grad=grads[n].detach().clone()
            gn=torch.nn.utils.clip_grad_norm_(list(geo.values()),c['meta_window']['clip_norm'],error_if_nonfinite=True)
            gopt.step();gopt.zero_grad(set_to_none=True)
            update=float(torch.sqrt(sum((v.detach()-before[n]).square().sum() for n,v in geo.items())))
            assert np.isfinite(update) and update>0,'zero/nonfinite geometry update'
            pl,pg=full_pseudo_step(model,opts,strong,py.detach(),valid,input_valid=uv,freeze_geometry=True)
            with torch.no_grad():committed,_,_=_present_foreground_dice(model(xa,valid_mask=av),ya,av)
            ema_update(teacher,model,c['ema_decay']);torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            rows.append(dict(warmup=step==0,seconds=elapsed,data_loading_seconds=data_seconds,supervised_loss=sl,pseudo_loss=pl,
                geometry_gradient_norm=float(gn),geometry_update_norm=update,meta=meta,
                actual_dice_loss_reduction=meta['reference_outer_dice_loss']-float(committed),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                supervised=sm,audit=am,unlabeled=um))
            print(json.dumps(dict(batch=a.batch,iteration=step,seconds=elapsed,geometry_update_norm=update,peak_reserved_bytes=rows[-1]['peak_reserved_bytes'])),flush=True)
            del grads,before,q,conf,py,valid,strong
        measured=[r for r in rows if not r['warmup']]
        result.update(status='PASS',samples_per_second=a.batch*len(measured)/sum(r['seconds'] for r in measured),
            median_seconds=float(np.median([r['seconds'] for r in measured])))
    except torch.cuda.OutOfMemoryError as e:result.update(status='OOM_WITH_2GIB_MARGIN',error=str(e))
    except Exception as e:result.update(status='FAILED',error=repr(e));raise
    finally:
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if rows:
            result['peak_allocated_bytes']=max(result['peak_allocated_bytes'],max(r['peak_allocated_bytes'] for r in rows))
            result['peak_reserved_bytes']=max(result['peak_reserved_bytes'],max(r['peak_reserved_bytes'] for r in rows))
        result['label_reads_outside_budget']=sorted(data.label_reads-set(data.lids))
        write_json(a.out,result)
    return 0 if result['status']=='PASS' else 2

if __name__=='__main__':raise SystemExit(main())
