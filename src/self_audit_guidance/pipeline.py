"""Opt-in stages: prepare fixed guidance, train a student, export held-out predictions.

No evaluation/reference module is imported. No mutation of v3/CQA artifacts.
"""
from __future__ import annotations
import argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .core import GuidanceConfig,make_bank
from .io import read_splits,read_cine,discover,digest,write_json,read_json,freeze,verify
from .teacher import FrozenCineMA,validate_device
from .student import train_student,load_student,predict_cine


def choose_patients(splits,count,seed):
    ids=splits['train']
    if type(count) is not int or count<0 or count>len(ids):raise ValueError('invalid TRAIN patient count')
    key=lambda p:hashlib.sha256(f'{seed}:{p}'.encode()).hexdigest()
    return sorted(sorted(ids,key=key)[:count]) if count else ids


def prepare(root,dataset,split_manifest,out,config=None,patients=4,seed=42,device='cuda',
            tile_batch=1,checkpoint=None,teacher_config=None,offline=False,teacher=None,reader=None):
    """teacher/reader injection is for explicit contract tests, never a CLI fallback."""
    cfg=GuidanceConfig(**(read_json(config) if config else {}));validate_device(device)
    splits=read_splits(split_manifest);split_sha=digest(split_manifest)
    ids=choose_patients(splits,patients,seed);records=discover(root,dataset,ids)
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    # Resolve dependencies/checkpoint BEFORE creating an output that looks like a completed bank.
    teacher=teacher if teacher is not None else FrozenCineMA(device,checkpoint,teacher_config,offline=offline)
    if not isinstance(teacher.provenance,dict) or not teacher.provenance.get('frozen'):raise ValueError('frozen teacher provenance required')
    reader=reader or read_cine
    out.mkdir(parents=True,exist_ok=False);reports={};inputs={}
    write_json(out/'config.json',cfg.dictionary());write_json(out/'split.json',splits)
    for pid,path in records.items():
        print(f'[GUIDANCE] patient={pid} raw={path.name}',flush=True)
        cine,meta=reader(path)
        if cine.ndim!=4 or cine.size<1 or not np.isfinite(cine).all():raise ValueError('invalid image-only reader output')
        if cine.shape[0]*cine.shape[1]*cfg.grid**2>cfg.max_nodes:raise ValueError('planned graph exceeds max_nodes')
        features=teacher.extract(cine,grid=cfg.grid,feature_dim=cfg.feature_dim,tile_batch=tile_batch)
        expected=(*cine.shape[:2],cfg.grid,cfg.grid,cfg.feature_dim)
        if features.shape!=expected or not np.isfinite(features).all():raise ValueError('teacher feature shape/finiteness mismatch')
        small=F.interpolate(torch.from_numpy(cine.copy()).flatten(0,1)[:,None],(cfg.grid,cfg.grid),mode='bilinear',align_corners=False)
        small=small[:,0].reshape(*cine.shape[:2],cfg.grid,cfg.grid).numpy()
        bank,report=make_bank(small,features,meta['patient_left_axis'],cfg)
        np.savez_compressed(out/(pid+'.npz'),image=cine.astype(np.float32),features=features.astype(np.float16),**bank)
        reports[pid]=report;inputs[pid]=meta
        print(f'[BANK] {pid} {report["status"]} classes={report["class_pixels"]} residual={report["solver"]["residual"]:.3g}',flush=True)
    if digest(split_manifest)!=split_sha:raise ValueError('split changed during preparation')
    for pid,path in records.items():
        if digest(path)!=inputs[pid]['image_sha256']:raise ValueError('raw image changed during preparation')
    class_patients=[sum(reports[pid]['class_pixels'][c]>0 for pid in ids) for c in range(4)]
    ready=all(n>=2 for n in class_patients[1:])
    report=dict(status='SUPPORT_NOT_ACCURACY' if ready else 'INSUFFICIENT_SEMANTIC_SUPPORT_FEATURE_BANK_READY',
                class_patients=class_patients,patient_reports=reports,teacher=teacher.provenance,
                quality_evaluated=False,gt_read=False)
    write_json(out/'REPORT.json',report)
    freeze(out,dict(role='guidance_bank',split='train',patients=ids,locked_splits=splits,
        source_split_sha256=split_sha,config=cfg.dictionary(),teacher=teacher.provenance,
        image_records=inputs,patient_reports=reports,status=report['status'],seed=seed,
        source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        supervision='no_manual_annotations_in_target_pipeline_declared_anatomy_priors_upstream_SSL_lineage_pending',
        evidence_policy='fixed_teacher_fixed_graph_fixed_anchor_no_student_feedback'))
    print('[GUIDANCE]',report['status'],flush=True)
    return report


def export(student,root,dataset,split_manifest,split,out,device='cuda',batch_size=8):
    if split not in ('val','test'):raise ValueError('held-out export must use val/test; never TRAIN')
    d=validate_device(device);model,ck=load_student(student,d)
    splits=read_splits(split_manifest)
    if splits!=ck['locked_splits']:raise ValueError('student and export patient split mismatch')
    ids=splits[split]
    if not ids:raise ValueError('empty export split')
    if set(ids)&set(ck['train_patients']):raise ValueError('patient leakage in export')
    records=discover(root,dataset,ids);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    inventory={};counts={}
    for pid,path in records.items():
        cine,meta=read_cine(path);labels=predict_cine(model,ck,cine,str(d),batch_size)
        # Native TZYX stored with exact XYZ T shape and affine in the frozen manifest.
        np.savez_compressed(out/(pid+'.npz'),labels=labels)
        inventory[pid]=meta;counts[pid]=[int((labels==c).sum()) for c in range(4)]
        print(f'[EXPORT] {pid} class_pixels={counts[pid]}',flush=True)
    freeze(out,dict(role='student_predictions',split=split,patients=ids,images=inventory,class_pixels=counts,
        student_manifest_sha256=digest(Path(student)/'FROZEN.json'),student_status=ck['status'],
        labels_axis_order='TZYX',unknown=255,quality_evaluated=False,gt_read=False))
    return ck['status']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='stage',required=True)
    a=sub.add_parser('prepare');a.add_argument('--root',required=True);a.add_argument('--dataset',choices=['acdc','mnms'],default='acdc')
    a.add_argument('--split-manifest',required=True);a.add_argument('--config');a.add_argument('--out',required=True)
    a.add_argument('--patients',type=int,default=4);a.add_argument('--seed',type=int,default=42);a.add_argument('--device',default='cuda')
    a.add_argument('--tile-batch',type=int,default=1);a.add_argument('--checkpoint');a.add_argument('--teacher-config');a.add_argument('--offline',action='store_true')
    a=sub.add_parser('train');a.add_argument('--bank',required=True);a.add_argument('--out',required=True)
    a.add_argument('--epochs',type=int,default=10);a.add_argument('--batch-size',type=int,default=8);a.add_argument('--device',default='cuda')
    a.add_argument('--width',type=int,default=32);a.add_argument('--lr',type=float,default=3e-4);a.add_argument('--seed',type=int,default=42);a.add_argument('--threads',type=int,default=4)
    a=sub.add_parser('export');a.add_argument('--student',required=True);a.add_argument('--root',required=True)
    a.add_argument('--dataset',choices=['acdc','mnms'],default='acdc');a.add_argument('--split-manifest',required=True)
    a.add_argument('--split',choices=['val','test'],required=True);a.add_argument('--out',required=True)
    a.add_argument('--device',default='cuda');a.add_argument('--batch-size',type=int,default=8)
    a=sub.add_parser('inspect');a.add_argument('--bank',required=True)
    args=vars(p.parse_args(argv));stage=args.pop('stage')
    if stage=='prepare':
        result=prepare(**args)
        return 0 if result['status']=='SUPPORT_NOT_ACCURACY' else 2
    if stage=='train':
        result=train_student(**args)
        return 0 if result['status']=='SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED' else 2
    if stage=='export':
        result=export(**args)
        return 0 if result=='SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED' else 2
    m=verify(args['bank'],'guidance_bank');print(m['status']);print(read_json(Path(args['bank'])/'REPORT.json'))
    return 0
