"""Runnable, split-isolated research workflows. Scientific usefulness is NOT inferred from completion."""
from __future__ import annotations
import argparse,json,shutil,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .data_v3 import (discover_acdc_full_cine,discover_mnms_full_cine,CineSliceDataset,
                      PatientBatchSampler,read_patient_splits,select_records,inspect_record)
from .system_v3 import CinePseudoTeacher,AdaptiveAnnotationStudent,pseudo_supervision_loss
from .trainer_v3 import ProgressiveTeacherTrainer,ProgressiveConfig
from .freeze import sha256_file,write_freeze_manifest,verify_frozen,safe_path

def discover(dataset,root):
    return discover_acdc_full_cine(root) if dataset=='acdc' else discover_mnms_full_cine(root)

def _log(tag,message):
    print(f"[{tag}] {message}",flush=True)

def _mean(rows,key):
    vals=[float(r[key]) for r in rows if key in r]
    return float(np.mean(vals)) if vals else float("nan")

def _peak_vram_gb(device):
    dev=torch.device(device)
    if dev.type!="cuda" or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(dev)/(1024**3))

def _reset_peak_vram(device):
    dev=torch.device(device)
    if dev.type=="cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)

def _parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def move(batch,device):
    return {k:(v.to(device).float() if k in ('prev','cur','nxt') else v) for k,v in batch.items()}

def inventory(records,splits):
    rows=[]
    for r in records:
        im,g=inspect_record(r)
        rows.append({'patient_id':r.patient_id,'dataset':r.dataset,'path':str(r.image_path.resolve()),
            'image_sha256':sha256_file(r.image_path),'shape':list(im.shape),'affine':g.affine.tolist(),
            'ed_index':r.ed_index,'es_index':r.es_index,'axis_order':'native_XYZT','spatial_unit':im.header.get_xyzt_units()[0],
            'temporal_unit':im.header.get_xyzt_units()[1],'zooms':[float(v) for v in im.header.get_zooms()],
            'split':next(s for s,ids in splits.items() if r.patient_id in ids)})
    return rows

def load_config(path):
    cfg=json.loads(Path(path).read_text())
    if cfg.get('schema_version')!=3: raise ValueError('use reviewed schema_version 3 config')
    required={'schema_version','teacher','reliability','loss_weights','consistency','training','deployment'}
    if set(cfg)!=required: raise ValueError('unexpected configuration sections')
    if cfg['consistency']['slice_weight']!=0: raise ValueError('unregistered cross-slice voting is disabled')
    if cfg['deployment']['profiles']!={'compact':0,'balanced':1,'accurate':2}: raise ValueError('unsupported profile contract')
    from .adaptive import RuntimeBudget
    RuntimeBudget.from_config(cfg["deployment"],max_profile="accurate")
    if type(cfg["training"]["bounded_default_epochs"]) is not int or cfg["training"]["bounded_default_epochs"]<1:
        raise ValueError("configured default epochs must be a positive integer")
    return cfg

def _common(parser):
    parser.add_argument('--dataset',choices=['acdc','mnms'],required=True)
    parser.add_argument('--root',required=True); parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cpu'); parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--epochs',type=int,default=None); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=4); parser.add_argument('--max-train-batches',type=int,default=0)

def _validate(args,cfg):
    if args.epochs is None: args.epochs=cfg["training"]["bounded_default_epochs"]
    if type(args.epochs) is not int or args.epochs<1 or args.batch_size<1 or args.threads<1 or args.max_train_batches<0: raise ValueError('invalid run budget')
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; no silent fallback')

def _loader(ds,batch_size,seed):
    keys=[ds.records[ri].patient_id for ri,_,_ in ds.index]
    return DataLoader(ds,batch_sampler=PatientBatchSampler(keys,batch_size,seed),num_workers=0)

def teacher_main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); _common(ap)
    ap.add_argument('--split-manifest',required=True)
    ap.add_argument('--export-split',choices=['train','val','train,val'],default='train,val')
    ap.add_argument('--config',default=str(Path(__file__).resolve().parents[2]/'configs/pseudolabel_v3.json'))
    ap.add_argument('--log-patients-every',type=int,default=10)
    args=ap.parse_args(argv); cfg=load_config(args.config); _validate(args,cfg)
    if args.log_patients_every<1: raise ValueError('log-patients-every must be positive')
    _log("RUN",f"teacher dataset={args.dataset} device={args.device} epochs={args.epochs} batch={args.batch_size} threads={args.threads}")
    _log("LOAD",f"discover root={Path(args.root).resolve()}")
    records=discover(args.dataset,args.root); splits=read_patient_splits(args.split_manifest)
    train_records=select_records(records,splits,'train')
    export_records=[]
    for split in args.export_split.split(','): export_records+=select_records(records,splits,split)
    if not export_records: raise ValueError('empty export split')
    _log("LOAD",f"patients discovered={len(records)} train={len(train_records)} val={len(splits.get('val',[]))} test={len(splits.get('test',[]))}")
    train_ds=CineSliceDataset(train_records); export_ds=CineSliceDataset(export_records)
    _log("LOAD",f"samples train={len(train_ds)} export={len(export_ds)} export_split={args.export_split}")
    # Fingerprint before training and re-check before freezing to detect changed inputs.
    _log("LOAD","fingerprinting native NIfTI inputs")
    record_inventory=inventory(list({r.patient_id:r for r in train_records+export_records}.values()),splits)
    by_id={r['patient_id']:r for r in record_inventory}
    _log("LOAD",f"inventory ready patients={len(record_inventory)}")
    kw=dict(cfg['teacher']); kw['k']=kw.pop('anonymous_prototypes'); maxdisp=kw.pop('registration_max_displacement')
    teacher=CinePseudoTeacher(**kw).to(args.device); teacher.motion.max_disp=float(maxdisp)
    _log("MODEL",f"teacher params={_parameter_count(teacher):,} prototypes={kw['k']} max_disp={maxdisp}")
    weights=cfg['loss_weights']; rel=cfg['reliability']
    tc=ProgressiveConfig(min_prob=rel['min_probability'],min_margin=rel['min_margin'],
        prototype_min_confidence=rel['prototype_min_confidence'],w_recon=weights['reconstruction'],
        w_proto=weights['anonymous_prototype'],w_seed=weights['semantic_seed'],
        w_motion=weights['motion_photometric'],w_motion_smooth=weights['motion_smoothness'])
    opt=torch.optim.AdamW(teacher.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    trainer=ProgressiveTeacherTrainer(teacher,opt,tc)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False); shutil.copyfile(args.split_manifest,out/'split.json')
    history=[]; loader=_loader(train_ds,args.batch_size,args.seed)
    _log("TRAIN",f"start batches_per_epoch={len(loader)} bounded={bool(args.max_train_batches)}")
    for epoch in range(args.epochs):
        started=time.perf_counter(); epoch_rows=[]; accepted_total=0
        _reset_peak_vram(args.device)
        for step,b in enumerate(loader):
            if args.max_train_batches and step>=args.max_train_batches: break
            losses,accepted,_=trainer.train_batch(move(b,args.device))
            row={'epoch':epoch,'step':step,'accepted_regions':accepted,**losses}
            history.append(row); epoch_rows.append(row); accepted_total+=int(accepted)
        elapsed=time.perf_counter()-started
        _log("EPOCH",
            f"{epoch+1}/{args.epochs} steps={len(epoch_rows)} "
            f"loss={_mean(epoch_rows,'total'):.4f} seed={_mean(epoch_rows,'semantic_seed'):.4f} "
            f"proto={_mean(epoch_rows,'prototype'):.4f} recon={_mean(epoch_rows,'reconstruction'):.4f} "
            f"motion={_mean(epoch_rows,'motion_photo'):.4f}/{_mean(epoch_rows,'motion_smooth'):.4f} "
            f"accepted={accepted_total} time={elapsed:.1f}s peak_vram={_peak_vram_gb(args.device):.2f}GB")
    (out/'train_metrics.json').write_text(json.dumps(history,indent=2))
    torch.save({'model':teacher.state_dict(),'prototype_bank':trainer.bank.prototypes.cpu(),
        'prototype_counts':trainer.bank.counts.cpu(),'config':cfg,'producer_patient_ids':splits['train']},out/'teacher.pt')
    _log("CHECKPOINT",f"saved {out/'teacher.pt'} prototype_counts={[int(x) for x in trainer.bank.counts.cpu().tolist()]}")
    from .native_export import export_native
    entries,volumes,export_report=export_native(export_records,by_id,out,
        lambda b:trainer.infer_batch(move(b,args.device))[0],cfg['consistency'],log_every=args.log_patients_every)
    (out/'export_report.json').write_text(json.dumps(export_report,indent=2))
    for r in record_inventory:
        if sha256_file(r['path'])!=r['image_sha256']: raise ValueError('image changed during run')
    package=Path(__file__).parent
    sources={p.name:sha256_file(p) for p in package.glob('*.py')}
    run_config={'artifact_role':'teacher_freeze','dataset':args.dataset,'manual_mask_input':False,'scribble_input':False,
        'args':vars(args),'resolved_config':cfg,'trainer_config':asdict(tc),'source_sha256':sources,
        'split_patients':splits,'producer_patient_ids':splits['train'],
        'image_records':record_inventory,'export_records':[by_id[r.patient_id] for r in export_records],
        'supervision':'image_only_with_handwritten_priors','bounded_training':bool(args.max_train_batches),
        'teacher_ready':'NOT_EVALUATED'}
    payload=write_freeze_manifest(out,entries,run_config,extra_artifacts=[*volumes,'export_report.json']); verify_frozen(out/'FROZEN.json')
    result={'status':'generation_complete','optimizer_steps':len(history),'exports':len(entries),
        'manifest_id':payload['manifest_id'],'valid_foreground_pixels':sum(e['valid_foreground'] for e in entries),
        'teacher_ready':'NOT_EVALUATED','out':str(out)}
    (out/'run_summary.json').write_text(json.dumps(result,indent=2))
    _log("DONE",f"teacher steps={len(history)} exports={len(entries)} valid_fg={result['valid_foreground_pixels']} manifest={payload['manifest_id'][:12]} out={out}")
    return result

class FrozenPseudoDataset(Dataset):
    def __init__(self,root,dataset,manifest):
        self.payload=verify_frozen(manifest,expected_role='teacher_freeze'); config=self.payload['config']
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
                    'valid':torch.from_numpy(p['valid'].astype(bool)),
                    'patient_id':self.rows[i][2],'t':self.base.index[idx][1],'z':self.base.index[idx][2]}

def student_main(argv=None):
    ap=argparse.ArgumentParser(description='Student from frozen TRAINING pseudo-labels only'); _common(ap)
    ap.add_argument('--manifest',required=True); ap.add_argument('--profile',choices=['compact','balanced','accurate'],default='balanced')
    ap.add_argument('--lr',type=float,default=1e-3)
    args=ap.parse_args(argv)
    payload=verify_frozen(args.manifest,expected_role='teacher_freeze')
    cfg=payload['config']['resolved_config']; _validate(args,cfg)
    out=Path(args.out)
    if out.exists() or out.with_suffix('.json').exists(): raise FileExistsError(out)
    _log("RUN",f"student dataset={args.dataset} device={args.device} epochs={args.epochs} batch={args.batch_size} profile={args.profile}")
    ds=FrozenPseudoDataset(args.root,args.dataset,args.manifest)
    # Do not manufacture a successful training run from zero foreground supervision.
    fg=sum(e.get('valid_foreground',0) for e in ds.payload['entries'] if e['split']=='train')
    _log("LOAD",f"frozen train samples={len(ds)} valid_foreground_pixels={fg} manifest={ds.payload['manifest_id'][:12]}")
    if fg==0: raise ValueError('NO_FOREGROUND_SEEDS: teacher not ready; student training refused')
    model_cfg=ds.payload['config']['resolved_config']['deployment']
    model=AdaptiveAnnotationStudent(width=model_cfg['student_width'],window_k=model_cfg['window_k']).to(args.device)
    _log("MODEL",f"student params={_parameter_count(model):,} width={model_cfg['student_width']} window_k={model_cfg['window_k']}")
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sampler=PatientBatchSampler([r[2] for r in ds.rows],args.batch_size,args.seed)
    hist=[]; skipped=0; class_pixels=[0]*4; patient_pixels={}; phase_pixels={}
    for epoch in range(args.epochs):
        model.train(); epoch_rows=[]; epoch_skipped=0; started=time.perf_counter(); _reset_peak_vram(args.device)
        for step,b in enumerate(DataLoader(ds,batch_sampler=sampler,num_workers=0)):
            if args.max_train_batches and step>=args.max_train_batches: break
            if not bool(b['valid'].any()): skipped+=1; epoch_skipped+=1; continue  # no AdamW drift/momentum update
            pred=model(b['image'].to(args.device),profile=args.profile)
            loss=pseudo_supervision_loss(pred,b['target'].to(args.device),b['valid'].to(args.device))
            if not bool(torch.isfinite(loss)): raise FloatingPointError('non-finite student loss')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            for i,pid in enumerate(b['patient_id']):
                counts=torch.bincount(b['target'][i][b['valid'][i]],minlength=4).tolist()
                if not sum(counts): continue
                class_pixels=[a+c for a,c in zip(class_pixels,counts)]
                patient_pixels[pid]=[a+c for a,c in zip(patient_pixels.get(pid,[0]*4),counts)]
                key=f"{pid}:t{int(b['t'][i])}"
                phase_pixels[key]=[a+c for a,c in zip(phase_pixels.get(key,[0]*4),counts)]
            row={'epoch':epoch,'step':step,'loss':float(loss.detach().cpu())}
            hist.append(row); epoch_rows.append(row)
        _log("EPOCH",f"{epoch+1}/{args.epochs} steps={len(epoch_rows)} skipped={epoch_skipped} "
             f"loss={_mean(epoch_rows,'loss'):.4f} time={time.perf_counter()-started:.1f}s "
             f"peak_vram={_peak_vram_gb(args.device):.2f}GB")
    if not hist: raise ValueError('NO_VALID_UPDATES: no checkpoint written')
    if sum(class_pixels[1:])==0:
        raise ValueError('NO_OBSERVED_FOREGROUND: bounded updates saw only background; no checkpoint written')
    verify_frozen(args.manifest,expected_role='teacher_freeze')
    for record in ds.base.records:
        source=next(r for r in ds.payload['config']['image_records'] if r['patient_id']==record.patient_id)
        if sha256_file(record.image_path)!=source['image_sha256']: raise ValueError('image changed during student training')
    coverage={'class_pixels':class_pixels,'patient_ids':sorted(patient_pixels),
              'patient_class_pixels':patient_pixels,'patient_frame_class_pixels':phase_pixels,
              'missing_foreground_classes':[c for c in (1,2,3) if class_pixels[c]==0],
              'support_status':'MISSING_CLASSES' if any(class_pixels[c]==0 for c in (1,2,3)) else 'ALL_CLASSES_OBSERVED',
              'quality_status':'NOT_EVALUATED','bounded_training':bool(args.max_train_batches)}
    from .checkpoint import save_student_checkpoint
    out.parent.mkdir(parents=True,exist_ok=True)
    digest=save_student_checkpoint(out,model,args=vars(args),model_config=model_cfg,
                                   manifest_id=ds.payload['manifest_id'],coverage=coverage)
    result={'status':'training_complete','optimizer_steps':len(hist),'skipped_empty_batches':skipped,
            'declared_train_patient_ids':ds.payload['config']['split_patients']['train'],
            'trained_patient_ids':sorted(patient_pixels),'coverage':coverage,'checkpoint_sha256':digest,'history':hist}
    out.with_suffix('.json').write_text(json.dumps(result,indent=2))
    _log("DONE",f"student steps={len(hist)} skipped={skipped} checkpoint={out}")
    return result
