#!/usr/bin/env python3
"""Parallel image-only patient shards, using canonical inference and selection.

Each worker keeps tile batch1 to preserve teacher arithmetic. Shards retain the
first8 items of their patient-round-robin stream per group. Every global first8
item is in that local prefix; merging/reselecting loses no eligible global item.
Endpoint tensors are copied directly, without a second normalization.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from self_audit.memory_bank import (load_native_manifest,build_memory_bank,patient_balanced_select,
    _rank_candidate,write_a1_cache,sha256_file)
from self_audit.teacher_memory import MemoryBank,load_bank,save_bank,CLASS_PAIRS


def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def merge_selected_banks(banks,provenance):
    pools={}
    for shard,bank in enumerate(banks):
        for group,pair in enumerate(CLASS_PAIRS):
            for slot in range(bank.entries_per_group):
                if not bool(bank.valid[group,slot]):continue
                c,d=bank.metadata['entry_metadata'][group][slot]
                if c['patient']!=d['patient']:raise ValueError('endpoint patient mismatch')
                item=dict(patient=c['patient'],timepoint=c['timepoint'],class_pair=list(pair),
                    coord_c_xyz=c['coord_xyz'],coord_d_xyz=d['coord_xyz'],image_sha256=c['image_sha256'],
                    location=(shard,group,slot))
                item['_rank']=_rank_candidate(item)
                pools.setdefault(c['patient'],{}).setdefault(tuple(pair),[]).append(item)
    selected=patient_balanced_select(pools)
    first=banks[0]
    ep=torch.zeros_like(first.endpoints);conf=torch.zeros_like(first.confidences);valid=torch.zeros_like(first.valid)
    metadata=copy.deepcopy(first.metadata);metadata['provenance']=provenance
    entries=[[[{},{}] for _ in range(first.entries_per_group)] for _ in CLASS_PAIRS]
    for group,pair in enumerate(CLASS_PAIRS):
        for slot,item in enumerate(selected[tuple(pair)]):
            shard,g,n=item['location'];source=banks[shard]
            ep[group,slot].copy_(source.endpoints[g,n]);conf[group,slot].copy_(source.confidences[g,n]);valid[group,slot]=True
            entries[group][slot]=copy.deepcopy(source.metadata['entry_metadata'][g][n])
    metadata['entry_metadata']=entries
    return MemoryBank(ep,conf,valid,metadata)


def worker(args):
    from self_audit.cinema_teacher import CineMATeacher
    out=args.out;out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    if args.device.startswith('cuda'):
        total=torch.cuda.get_device_properties(0).total_memory
        allowed=(total-2*2**30-args.workers*384*2**20)/args.workers
        if allowed<=0:raise ValueError('invalid aggregate GPU memory budget')
        torch.cuda.set_per_process_memory_fraction(allowed/total)
    started=time.perf_counter()
    adapter=CineMATeacher(args.teacher_config,args.teacher_weights,source_root=args.teacher_source,device=args.device)
    write(out/'READY.json',dict(pid=os.getpid(),model_load_seconds=time.perf_counter()-started))
    while not args.go.exists():time.sleep(.1)
    start=time.perf_counter()
    result=build_memory_bank(args.native_manifest,args.native_manifest_sha256,mode='teacher',adapter=adapter,
        device=args.device,max_timepoints=args.max_timepoints,
        collect_a1_cache=args.write_a1_cache,a1_cache_dir=out/'external_pseudo' if args.write_a1_cache else None)
    payload=result['bank'];bank=MemoryBank(payload['endpoints'],payload['confidences'],payload['valid'],payload['metadata'])
    bank_sha=save_bank(bank,out/'fixed_bank.npz')
    write(out/'RESULT.json',dict(status='SHARD_COMPLETED_NOT_FULL_BANK',coverage=result['coverage'],
        bank_sha256=bank_sha,a1_records=result['a1_records'],seconds=time.perf_counter()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device.startswith('cuda') else None))


def coordinator(args):
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    source=load_native_manifest(args.native_manifest,args.native_manifest_sha256)
    records=source['patients']
    if args.max_patients:
        # Cost diagnostics select by image geometry only, never labels or Dice.
        records=sorted(records,key=lambda r:math.prod(r['shape_xyzt'][:3]),reverse=True)[:args.max_patients]
    groups=[[] for _ in range(min(args.workers,len(records)))];costs=[0]*len(groups)
    for record in sorted(records,key=lambda r:math.prod(r['shape_xyzt']),reverse=True):
        i=min(range(len(groups)),key=costs.__getitem__);groups[i].append(record)
        costs[i]+=math.prod(record['shape_xyzt'][:3])*(min(record['shape_xyzt'][3],args.max_timepoints) if args.max_timepoints else record['shape_xyzt'][3])
    go=out/'GO';children=[];loaded=time.perf_counter();peak_gpu=0;peak_ram=0
    device_total=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())*2**20
    def monitor():
        nonlocal peak_gpu,peak_ram
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
        peak_gpu=max(peak_gpu,int(raw)*2**20)
        if peak_gpu>device_total-2*2**30:raise MemoryError('GPU spare memory below 2GiB')
        peak_ram=max(peak_ram,int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text()))
        if peak_ram>args.ram_limit_gib*2**30:raise MemoryError('declared RAM limit exceeded')
    try:
        for i,group in enumerate(groups):
            shard=copy.deepcopy(source);shard['patients']=sorted(group,key=lambda r:r['id'])
            shard['coverage'].update(train_patients_selected=len(group),complete=len(group)==80,bounded=len(group)!=80)
            manifest=out/f'shard_{i}.json';write(manifest,shard)
            target=out/f'worker_{i}'
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--out',str(target),
                '--native-manifest',str(manifest),'--native-manifest-sha256',sha256_file(manifest),
                '--teacher-source',str(args.teacher_source),'--teacher-config',str(args.teacher_config),
                '--teacher-weights',str(args.teacher_weights),'--device',args.device,'--workers',str(len(groups)),
                '--go',str(go)]
            if args.max_timepoints:cmd+=['--max-timepoints',str(args.max_timepoints)]
            if args.write_a1_cache:cmd+=['--write-a1-cache']
            log=(out/f'worker_{i}.log').open('x');p=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT);log.close();children.append(p)
        while not all((out/f'worker_{i}/READY.json').exists() for i in range(len(groups))):
            if any(p.poll() is not None for p in children):raise RuntimeError('worker failed during model load')
            monitor();time.sleep(.5)
        startup=time.perf_counter()-loaded;start=time.perf_counter();go.touch()
        while any(p.poll() is None for p in children):
            monitor();time.sleep(.5)
        if any(p.returncode for p in children):raise RuntimeError('worker failed; partial outputs preserved')
        seconds=time.perf_counter()-start
        results=[json.loads((out/f'worker_{i}/RESULT.json').read_text()) for i in range(len(groups))]
        banks=[load_bank(out/f'worker_{i}/fixed_bank.npz',r['bank_sha256']) for i,r in enumerate(results)]
        used=sum(r['coverage']['timepoints_used'] for r in results)
        full=len(records)==len(source['patients']) and used==sum(r['shape_xyzt'][3] for r in source['patients'])
        coverage=dict(train_patients_total=80,train_patients_in_manifest=len(records),train_patients_used=len(records),
            timepoints_total=sum(r['shape_xyzt'][3] for r in source['patients']),timepoints_used=used,
            complete=full,bounded=not full)
        provenance=copy.deepcopy(banks[0].metadata['provenance'])
        for key in ('image_hashes','native_image_hashes','native_geometry_by_patient',
                    'teacher_lineage_by_patient_timepoint','teacher_geometry_by_patient_timepoint'):
            merged={}
            for bank in banks:
                values=bank.metadata['provenance'][key]
                if set(merged)&set(values):raise ValueError('duplicate shard source IDs')
                merged.update(values)
            provenance[key]=merged
        provenance.update(native_manifest_sha256=args.native_manifest_sha256,coverage=coverage,
            parallel_execution=dict(workers=len(groups),tile_batch_size=1,selection='canonical global patient-balanced prefix merge; no vector renormalization'))
        bank=merge_selected_banks(banks,provenance);bank_sha=save_bank(bank,out/'fixed_bank.npz')
        parity=None
        if not full:
            for prior in sorted(out.parent.glob('benchmark_w*/BUILD_RECEIPT.json')):
                if prior.parent==out:continue
                previous=json.loads(prior.read_text());reference=load_bank(prior.parent/'fixed_bank.npz',previous['bank_sha256'])
                same=(torch.equal(reference.endpoints,bank.endpoints)
                    and torch.equal(reference.confidences,bank.confidences)
                    and torch.equal(reference.valid,bank.valid)
                    and reference.metadata['entry_metadata']==bank.metadata['entry_metadata'])
                if not same:raise RuntimeError('parallel-count benchmark changed selected bank tensors/entries')
                parity=dict(reference=str(prior),selected_bank_bitwise_equal=True)
                break
        cache=None
        if args.write_a1_cache:
            if not full:raise ValueError('partial shards cannot certify a full A1 cache')
            destination=out/'external_pseudo';destination.mkdir();items=[]
            for result in results:
                for item in result['a1_records']:
                    entry=copy.deepcopy(item)
                    for key in ('probabilities_path','valid_support_path'):
                        source_path=Path(item[key]);target=destination/source_path.name;os.link(source_path,target);entry[key]=str(target.resolve())
                    items.append(entry)
            payload=dict(bank=dict(metadata=bank.metadata),coverage=coverage,a1_records=items,manifest=source)
            cache=write_a1_cache(destination,payload,bank_sha256=bank_sha)
        receipt=dict(schema='self_audit.memory_bank_build.v1',status='BUILT',mode='external_teacher_image_only_parallel',
            bank=str((out/'fixed_bank.npz').resolve()),bank_sha256=bank_sha,native_manifest_sha256=args.native_manifest_sha256,
            coverage=coverage,coverage_status='FULL_TRAIN' if full else 'BOUNDED_NOT_FULL_BANK',
            a1_cache=str(cache) if cache else None,mask_paths_read=0,evaluation_ids_used=False,workers=len(groups),
            tile_batch_size=1,startup_seconds=startup,processing_seconds=seconds,
            timepoints_per_second=used/seconds,peak_device_used_bytes=peak_gpu,peak_cgroup_ram_bytes=peak_ram,
            parallel_count_parity=parity)
        write(out/'BUILD_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
        return receipt
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:p.wait()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','native-manifest','teacher-source','teacher-config','teacher-weights'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--native-manifest-sha256',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--ram-limit-gib',type=float,default=120)
    p.add_argument('--max-patients',type=int);p.add_argument('--max-timepoints',type=int)
    p.add_argument('--write-a1-cache',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--go',type=Path)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.workers<1:raise ValueError('positive worker count required')
    if args.max_patients is not None and args.max_patients<1:raise ValueError('positive patient bound required')
    if args.max_timepoints is not None and args.max_timepoints<1:raise ValueError('positive timepoint bound required')
    worker(args) if args.worker else coordinator(args)
