"""Runnable, split-isolated research workflows. Scientific usefulness is NOT inferred from completion."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .data_v3 import (discover_acdc_full_cine,discover_mnms_full_cine,CineSliceDataset,
                      PatientBatchSampler,read_patient_splits,select_records,inspect_record)
from .system_v3 import CinePseudoTeacher,AdaptiveAnnotationStudent,pseudo_supervision_loss,UNKNOWN
from .trainer_v3 import ProgressiveTeacherTrainer,ProgressiveConfig
from .freeze import sha256_file,export_pseudo_npz,write_freeze_manifest,verify_frozen,safe_path
from .consistency import consistency_gate

def discover(dataset,root):
    return discover_acdc_full_cine(root) if dataset=='acdc' else discover_mnms_full_cine(root)

def move(batch,device):
    return {k:(v.to(device).float() if k in ('prev','cur','nxt') else v) for k,v in batch.items()}

def inventory(records,splits):
    rows=[]
    for r in records:
        im,g=inspect_record(r)
        rows.append({'patient_id':r.patient_id,'dataset':r.dataset,'path':str(r.image_path.resolve()),
            'image_sha256':sha256_file(r.image_path),'shape':list(im.shape),'affine':g.affine.tolist(),
            'ed_index':r.ed_index,'es_index':r.es_index,'axis_order':'native_XYZT','spatial_unit':im.header.get_xyzt_units()[0],
            'split':next(s for s,ids in splits.items() if r.patient_id in ids)})
    return rows

def load_config(path):
    cfg=json.loads(Path(path).read_text())
    if cfg.get('schema_version')!=3: raise ValueError('use reviewed schema_version 3 config')
    required={'schema_version','teacher','reliability','loss_weights','consistency','training','deployment'}
    if set(cfg)!=required: raise ValueError('unexpected configuration sections')
    if cfg['consistency']['slice_weight']!=0: raise ValueError('unregistered cross-slice voting is disabled')
    if cfg['deployment']['profiles']!={'compact':0,'balanced':1,'accurate':2}: raise ValueError('unsupported profile contract')
    return cfg

def _common(parser):
    parser.add_argument('--dataset',choices=['acdc','mnms'],required=True)
    parser.add_argument('--root',required=True); parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cpu'); parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--epochs',type=int,default=1); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=4); parser.add_argument('--max-train-batches',type=int,default=0)

def _validate(args):
    if args.epochs<1 or args.batch_size<1 or args.threads<1 or args.max_train_batches<0: raise ValueError('invalid run budget')
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; no silent fallback')

def _loader(ds,batch_size,seed):
    keys=[ds.records[ri].patient_id for ri,_,_ in ds.index]
    return DataLoader(ds,batch_sampler=PatientBatchSampler(keys,batch_size,seed),num_workers=0)

def _flush(rows,out_root,record,cfg):
    tmax,zmax=record['shape'][3],record['shape'][2]
    if len(rows)!=tmax*zmax: raise ValueError('incomplete patient export')
    c,h,w=rows[0]['soft'].shape
    soft=torch.stack([r['soft'] for r in rows]).reshape(tmax,zmax,c,h,w)
    valid=torch.stack([r['valid'] for r in rows]).reshape(tmax,zmax,h,w)
    if cfg['enabled']:
        fp=torch.stack([r['fp'] for r in rows]).reshape(tmax,zmax,2,*rows[0]['fp'].shape[-2:])
        fn=torch.stack([r['fn'] for r in rows]).reshape_as(fp)
        soft,valid=consistency_gate(soft,valid,temporal_weight=cfg['temporal_weight'],slice_weight=0,
                min_agreement=cfg['min_agreement'],flow_prev=fp,flow_next=fn)
    labels=torch.where(valid,soft.argmax(2),torch.full_like(valid,UNKNOWN,dtype=torch.long))
    entries=[]
    for row in rows:
        t,z=row['t'],row['z']; rel=Path('pseudo')/record['patient_id']/f't{t:03d}_z{z:03d}.npz'
        meta={'patient_id':record['patient_id'],'t':t,'z':z,'dataset':record['dataset'],'split':record['split'],
              'image_sha256':record['image_sha256'],'affine':record['affine'],'axis_order':'native_XYZT'}
        sha=export_pseudo_npz(out_root/rel,pseudo_label=labels[t,z].numpy(),valid=valid[t,z].numpy(),
                              soft_label=soft[t,z].numpy(),metadata=meta)
        entries.append({'path':str(rel),'sha256':sha,'patient_id':record['patient_id'],'t':t,'z':z,
                        'split':record['split'],'valid_fraction':float(valid[t,z].float().mean()),
                        'valid_foreground':int((valid[t,z]&(labels[t,z]>0)).sum())})
    return entries

def teacher_main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); _common(ap)
    ap.add_argument('--split-manifest',required=True)
    ap.add_argument('--export-split',choices=['train','val','train,val'],default='train,val')
    ap.add_argument('--config',default=str(Path(__file__).resolve().parents[2]/'configs/pseudolabel_v3.json'))
    args=ap.parse_args(argv); _validate(args); cfg=load_config(args.config)
    records=discover(args.dataset,args.root); splits=read_patient_splits(args.split_manifest)
    train_records=select_records(records,splits,'train')
    export_records=[]
    for split in args.export_split.split(','): export_records+=select_records(records,splits,split)
    if not export_records: raise ValueError('empty export split')
    train_ds=CineSliceDataset(train_records); export_ds=CineSliceDataset(export_records)
    # Fingerprint before training and re-check before freezing to detect changed inputs.
    record_inventory=inventory(list({r.patient_id:r for r in train_records+export_records}.values()),splits)
    by_id={r['patient_id']:r for r in record_inventory}
    kw=dict(cfg['teacher']); kw['k']=kw.pop('anonymous_prototypes'); maxdisp=kw.pop('registration_max_displacement')
    teacher=CinePseudoTeacher(**kw).to(args.device); teacher.motion.max_disp=float(maxdisp)
    weights=cfg['loss_weights']; rel=cfg['reliability']
    tc=ProgressiveConfig(min_prob=rel['min_probability'],min_margin=rel['min_margin'],
        prototype_min_confidence=rel['prototype_min_confidence'],w_recon=weights['reconstruction'],
        w_proto=weights['anonymous_prototype'],w_seed=weights['semantic_seed'],
        w_motion=weights['motion_photometric'],w_motion_smooth=weights['motion_smoothness'])
    opt=torch.optim.AdamW(teacher.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    trainer=ProgressiveTeacherTrainer(teacher,opt,tc)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False); shutil.copyfile(args.split_manifest,out/'split.json')
    history=[]; loader=_loader(train_ds,args.batch_size,args.seed)
    for epoch in range(args.epochs):
        for step,b in enumerate(loader):
            if args.max_train_batches and step>=args.max_train_batches: break
            losses,accepted,_=trainer.train_batch(move(b,args.device))
            history.append({'epoch':epoch,'step':step,'accepted_regions':accepted,**losses})
    (out/'train_metrics.json').write_text(json.dumps(history,indent=2))
    torch.save({'model':teacher.state_dict(),'prototype_bank':trainer.bank.prototypes.cpu(),
        'prototype_counts':trainer.bank.counts.cpu(),'config':cfg,'producer_patient_ids':splits['train']},out/'teacher.pt')
    entries=[]; rows=[]; current=None
    for b in DataLoader(export_ds,batch_size=1,shuffle=False,num_workers=0):
        pid=b['patient_id'][0]
        if current is not None and pid!=current:
            entries+=_flush(rows,out,by_id[current],cfg['consistency']); rows=[]
        current=pid; pred,_,_=trainer.infer_batch(move(b,args.device))
        rows.append({'t':int(b['t'][0]),'z':int(b['z'][0]),'soft':pred['soft_label'][0].cpu(),
            'valid':pred['valid'][0].cpu(),'fp':pred['flow_prev'][0].cpu(),'fn':pred['flow_next'][0].cpu()})
    entries+=_flush(rows,out,by_id[current],cfg['consistency'])
    for r in record_inventory:
        if sha256_file(r['path'])!=r['image_sha256']: raise ValueError('image changed during run')
    package=Path(__file__).parent
    sources={p.name:sha256_file(p) for p in package.glob('*.py')}
    run_config={'dataset':args.dataset,'manual_mask_input':False,'scribble_input':False,
        'args':vars(args),'resolved_config':cfg,'trainer_config':asdict(tc),'source_sha256':sources,
        'split_patients':splits,'producer_patient_ids':splits['train'],
        'image_records':record_inventory,'export_records':[by_id[r.patient_id] for r in export_records],
        'supervision':'image_only_with_handwritten_priors','bounded_training':bool(args.max_train_batches),
        'teacher_ready':'NOT_EVALUATED'}
    payload=write_freeze_manifest(out,entries,run_config); verify_frozen(out/'FROZEN.json')
    result={'status':'generation_complete','optimizer_steps':len(history),'exports':len(entries),
        'manifest_id':payload['manifest_id'],'valid_foreground_pixels':sum(e['valid_foreground'] for e in entries),
        'teacher_ready':'NOT_EVALUATED','out':str(out)}
    (out/'run_summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

class FrozenPseudoDataset(Dataset):
    def __init__(self,root,dataset,manifest):
        self.payload=verify_frozen(manifest); config=self.payload['config']
        if config.get('dataset')!=dataset: raise ValueError('dataset mismatch')
        splits=config.get('split_patients',{})
        if set(config.get('producer_patient_ids',[]))!=set(splits.get('train',[])):
            raise ValueError('teacher training cohort is not the locked training split')
        if set(splits['train'])&set(splits.get('val',[])+splits.get('test',[])): raise ValueError('patient leakage')
        records=select_records(discover(dataset,root),splits,'train')
        source={r['patient_id']:r for r in config['image_records']}
        for r in records:
            if sha256_file(r.image_path)!=source[r.patient_id]['image_sha256']: raise ValueError('student image differs from frozen source')
        self.base=CineSliceDataset(records); self.freeze_root=Path(manifest).parent; self.rows=[]
        lookup={(self.base.records[ri].patient_id,t,z):i for i,(ri,t,z) in enumerate(self.base.index)}
        for e in self.payload['entries']:
            if e['split']!='train': continue  # development pseudo-labels NEVER train the student
            key=(e['patient_id'],e['t'],e['z'])
            if key not in lookup: raise ValueError('invalid training pseudo identity')
            self.rows.append((lookup[key],safe_path(self.freeze_root,e['path']),e['patient_id']))
        if not self.rows: raise ValueError('freeze contains no training pseudo-labels; export train,val')
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        idx,path,_=self.rows[i]
        with np.load(path,allow_pickle=False) as p:
            return {'image':self.base[idx]['cur'],'target':torch.from_numpy(p['pseudo_label'].astype(np.int64)),
                    'valid':torch.from_numpy(p['valid'].astype(bool))}

def student_main(argv=None):
    ap=argparse.ArgumentParser(description='Student from frozen TRAINING pseudo-labels only'); _common(ap)
    ap.add_argument('--manifest',required=True); ap.add_argument('--profile',choices=['compact','balanced','accurate'],default='balanced')
    ap.add_argument('--lr',type=float,default=1e-3)
    args=ap.parse_args(argv); _validate(args)
    out=Path(args.out)
    if out.exists() or out.with_suffix('.json').exists(): raise FileExistsError(out)
    ds=FrozenPseudoDataset(args.root,args.dataset,args.manifest)
    # Do not manufacture a successful training run from zero foreground supervision.
    fg=sum(e.get('valid_foreground',0) for e in ds.payload['entries'] if e['split']=='train')
    if fg==0: raise ValueError('NO_FOREGROUND_SEEDS: teacher not ready; student training refused')
    model_cfg=ds.payload['config']['resolved_config']['deployment']
    model=AdaptiveAnnotationStudent(width=model_cfg['student_width'],window_k=model_cfg['window_k']).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sampler=PatientBatchSampler([r[2] for r in ds.rows],args.batch_size,args.seed)
    hist=[]; skipped=0
    for epoch in range(args.epochs):
        model.train()
        for step,b in enumerate(DataLoader(ds,batch_sampler=sampler,num_workers=0)):
            if args.max_train_batches and step>=args.max_train_batches: break
            if not bool(b['valid'].any()): skipped+=1; continue  # no AdamW drift/momentum update
            pred=model(b['image'].to(args.device),profile=args.profile)
            loss=pseudo_supervision_loss(pred,b['target'].to(args.device),b['valid'].to(args.device))
            if not bool(torch.isfinite(loss)): raise FloatingPointError('non-finite student loss')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            hist.append({'epoch':epoch,'step':step,'loss':float(loss.detach().cpu())})
    if not hist: raise ValueError('NO_VALID_UPDATES: no checkpoint written')
    verify_frozen(args.manifest)
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'args':vars(args),'model_config':model_cfg,
                'manifest_id':ds.payload['manifest_id'],'profile_trained':args.profile},out)
    result={'status':'training_complete','optimizer_steps':len(hist),'skipped_empty_batches':skipped,
            'trained_patient_ids':ds.payload['config']['split_patients']['train'],'history':hist}
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))
