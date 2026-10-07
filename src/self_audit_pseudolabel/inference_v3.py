"""Frozen-checkpoint native inference. No optimizer, fit, reference, or threshold search."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import torch
from .freeze import verify_frozen, sha256_file, write_freeze_manifest
from .data_v3 import select_records
from .pipeline_v3 import discover, inventory, move
from .native_export import export_native


def _arguments(argv, student):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-freeze',required=True,help='Verified teacher FROZEN.json')
    parser.add_argument('--root',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--split',choices=['train','val','test'],default='val')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--no-progress',action='store_true',help='disable native export progress bars')
    if student:
        parser.add_argument('--checkpoint',required=True)
        parser.add_argument('--profile',choices=['adaptive','compact','balanced','accurate'],default='adaptive')
    args = parser.parse_args(argv)
    if args.threads<1: raise ValueError('threads must be positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent fallback')
    torch.set_num_threads(args.threads)
    return args


def _inputs(args):
    source = Path(args.source_freeze)
    payload = verify_frozen(source,expected_role='teacher_freeze')
    cfg = payload['config']; splits = cfg['split_patients']
    records = select_records(discover(cfg['dataset'],args.root),splits,args.split)
    if not records: raise ValueError('empty requested split')
    current = inventory(records,splits)
    old = {r['patient_id']:r for r in cfg['image_records']}
    for row in current:
        previous = old.get(row['patient_id'])
        if previous and any(row[k]!=previous[k] for k in ('image_sha256','shape','affine','spatial_unit','ed_index','es_index')):
            raise ValueError('inference image differs from frozen source')
        old[row['patient_id']] = row
    return source,payload,records,current,list(old.values())


def _seal(args, source, payload, records, current, images, predict, *, student, checkpoint_meta=None):
    cfg = payload['config']; out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source,out/'source_freeze.json')
    shutil.copyfile(source.parent/'split.json',out/'split.json')
    if student:
        shutil.copyfile(args.checkpoint,out/'student.pt')
        if sha256_file(out/'student.pt')!=checkpoint_meta['checkpoint_sha256']:
            raise ValueError('checkpoint changed before inference freeze')
        consistency = {'enabled':False,'slice_weight':0}
    else:
        for name in ('teacher.pt','train_metrics.json'):
            shutil.copyfile(source.parent/name,out/name)
        consistency = cfg['resolved_config']['consistency']
    entries,volumes,report = export_native(records,{r['patient_id']:r for r in current},out,predict,consistency,progress=not args.no_progress)
    for row in current:
        if sha256_file(row['path'])!=row['image_sha256']: raise ValueError('image changed during inference')
    verify_frozen(source,expected_role='teacher_freeze')
    (out/'export_report.json').write_text(json.dumps(report,indent=2))
    run_cfg = {**cfg,'artifact_role':'student_predictions' if student else 'teacher_freeze',
        'image_records':images,'export_records':current,'source_manifest_id':payload['manifest_id'],
        'inference_only':True,'test_untouched':'NOT_CERTIFIED_BY_CODE','inference_args':vars(args)}
    if student:
        run_cfg.update(checkpoint_sha256=checkpoint_meta['checkpoint_sha256'],
                       source_sha256=checkpoint_meta['source_sha256'],
                       profile_trained=checkpoint_meta['profile_trained'],
                       observed_training_coverage=checkpoint_meta['coverage'],
                       prediction_policy='dense_argmax_no_abstention',
                       bounded_training=checkpoint_meta['coverage']['bounded_training'])
    write_freeze_manifest(out,entries,run_cfg,extra_artifacts=[*volumes,'export_report.json'])
    result = verify_frozen(out/'FROZEN.json',expected_role=run_cfg['artifact_role'])
    print(f"[DONE] frozen {run_cfg['artifact_role']} split={args.split} manifest={result['manifest_id']}",flush=True)
    return result


def student_inference_main(argv=None):
    from .checkpoint import load_student_checkpoint
    from .adaptive import AdaptiveRuntime
    args = _arguments(argv,student=True)
    source,payload,records,current,images = _inputs(args)
    fixed = None if args.profile=='adaptive' else args.profile
    model,budget,meta = load_student_checkpoint(args.checkpoint,device=args.device,max_profile=fixed,
                                               manifest_id=payload['manifest_id'])
    runtime = AdaptiveRuntime(model,budget)
    @torch.no_grad()
    def predict(batch):
        x = batch['cur'].to(args.device)
        output = runtime(x) if fixed is None else model(x,profile=fixed)
        soft = output['final_logits'].softmax(1)
        return {'soft_label':soft,'valid':torch.ones_like(soft[:,0],dtype=torch.bool)}
    return _seal(args,source,payload,records,current,images,predict,student=True,checkpoint_meta=meta)


def teacher_export_main(argv=None):
    from .system_v3 import CinePseudoTeacher
    from .trainer_v3 import ProgressiveTeacherTrainer, ProgressiveConfig
    args = _arguments(argv,student=False)
    source,payload,records,current,images = _inputs(args)
    cfg = payload['config']
    actual = {p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
    if cfg['source_sha256']!=actual: raise ValueError('teacher source identity mismatch')
    checkpoint = torch.load(source.parent/'teacher.pt',map_location='cpu',weights_only=True)
    if checkpoint['config']!=cfg['resolved_config'] or checkpoint['producer_patient_ids']!=cfg['producer_patient_ids']:
        raise ValueError('teacher checkpoint/config mismatch')
    kw = dict(checkpoint['config']['teacher']); kw['k']=kw.pop('anonymous_prototypes')
    maxdisp = kw.pop('registration_max_displacement')
    teacher = CinePseudoTeacher(**kw).to(args.device)
    teacher.motion.max_disp = float(maxdisp)
    teacher.load_state_dict(checkpoint['model'],strict=True)
    trainer = ProgressiveTeacherTrainer(teacher,None,ProgressiveConfig(**cfg['trainer_config']))
    trainer.bank.prototypes.copy_(checkpoint['prototype_bank'])
    trainer.bank.counts.copy_(checkpoint['prototype_counts'])
    return _seal(args,source,payload,records,current,images,
                 lambda batch:trainer.infer_batch(move(batch,args.device))[0],student=False)
