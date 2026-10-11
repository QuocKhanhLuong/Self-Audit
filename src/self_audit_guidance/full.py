"""Full split-isolated guidance pipeline with mandatory real preflight.

Fixed epoch/checkpoint policy. Validation and test references are visible only
in separate evaluator processes, AFTER both prediction sets have been frozen.
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from .io import read_json,write_json,read_splits,digest,verify,discover
from .teacher import WEIGHT_SHA256,HF_REVISION
from .core import GuidanceConfig

REPO=Path(__file__).resolve().parents[2]


def derive_split(source,out,seed=42):
    """Preserve TRAIN; split legacy validation by identity, never by image/GT score."""
    import hashlib
    splits=read_splits(source)
    if splits['test']:
        raise ValueError('source already has a test partition; use it directly')
    if len(splits['val'])<2:
        raise ValueError('need at least two held-out patients to split validation/test')
    order=sorted(splits['val'],key=lambda p:hashlib.sha256(f'{seed}:{p}'.encode()).hexdigest())
    n=len(order)//2
    result=dict(train=splits['train'],val=sorted(order[:n]),test=sorted(order[n:]),
                provenance=dict(source_path=str(Path(source).resolve()),source_sha256=digest(source),seed=seed,
                    policy='preserve_train_split_legacy_validation_by_sha256_identity',
                    test_status='exploratory_internal_holdout_from_previously_used_validation_not_pristine_or_hidden_test'))
    out=Path(out)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);write_json(out,result)
    print('[SPLIT]',{s:len(result[s]) for s in ('train','val','test')},flush=True)
    return result

def external_acdc_patients(test_root):
    """Discover official ACDC TEST IDs from image-only directory/file names.

    No GT/Info.cfg/ED/ES inspection; fail if a study is missing its 4-D cine.
    """
    root=Path(test_root).resolve()
    if not root.is_dir():raise FileNotFoundError(f'official ACDC test directory not found: {root}')
    candidates=sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r'patient[0-9]+',p.name))
    if not candidates:raise ValueError(f'no ACDC patient directories in test root: {root}')
    ids=[p.name for p in candidates]
    discover(root,'acdc',ids)  # only raw *4d NIfTI file names; never open masks
    return ids


def stage_command(stage,**kwargs):
    paths={'preflight':REPO/'scripts/preflight_stable_guidance.py',
           'evaluate':REPO/'scripts/evaluate_stable_guidance.py'}
    cmd=[sys.executable,str(paths.get(stage,REPO/'scripts/run_stable_guidance.py'))]
    if stage not in paths:cmd.append(stage)
    for key,value in kwargs.items():
        if value is None or value is False: continue
        cmd.append('--'+key.replace('_','-'))
        if value is not True:cmd.append(str(value))
    return cmd


def run_stage(name,command,out):
    """Synchronous subprocess; logs are outside immutable stage artifacts."""
    path=Path(out)/'logs';path.mkdir(exist_ok=True)
    log=path/f'{name}_{time.time_ns()}.log'
    env=dict(os.environ);env['PYTHONPATH']=str(REPO/'src')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONUNBUFFERED']='1'
    print('[STAGE]',name,' '.join(command),flush=True)
    with log.open('x') as f:
        with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env) as process:
            for line in process.stdout:
                print(line,end='',flush=True);f.write(line);f.flush()
            code=process.wait()
    if code not in (0,2):raise RuntimeError(f'{name} failed with exit {code}; see {log}')
    return code


def existing_stage(path,role,resume):
    path=Path(path)
    if not path.exists():return None
    if not resume:raise FileExistsError(path)
    if not (path/'FROZEN.json').is_file():
        raise RuntimeError(f'INCOMPLETE_STAGE {path}: will not overwrite or treat as completed; use a fresh run or recover explicitly')
    return verify(path,role)


def run_full(root,split_manifest,out,config,*,dataset='acdc',epochs=50,batch_size=8,
             tile_batch=1,width=32,lr=3e-4,device='cuda',threads=4,seed=42,
             checkpoint=None,teacher_config=None,offline=False,resume=False,preflight_only=False,
             evaluation='acdc',reference_root=None,reference_manifest=None,test_root=None):
    if min(epochs,batch_size,tile_batch,threads)<1:raise ValueError('positive execution budgets required')
    cfg=GuidanceConfig(**read_json(config));splits=read_splits(split_manifest)
    # The official ACDC testing cohort is separate from the local 80/20 TRAIN/VAL
    # split. Only directory and raw cine filenames define its test identities.
    official_test=test_root is not None
    if official_test:
        if dataset!='acdc':raise ValueError('--test-root currently supports official ACDC only')
        if Path(test_root).resolve()==Path(root).resolve():
            raise ValueError('--test-root must be separate from the TRAIN/VAL root')
        test_ids=external_acdc_patients(test_root)
        if splits['test'] and splits['test']!=test_ids:
            raise ValueError('provided TEST split disagrees with official testing directory')
        if set(test_ids)&(set(splits['train'])|set(splits['val'])):
            raise ValueError('patient leakage: official test overlaps TRAIN or VAL')
        splits['test']=test_ids
    if any(not splits[s] for s in ('train','val','test')):
        raise ValueError('full pipeline requires disjoint nonempty train/val/test; use --test-root for a separate official ACDC/testing directory')
    if evaluation not in ('acdc','manifest','none'):raise ValueError('invalid evaluation mode')
    if dataset!='acdc' and evaluation=='acdc':raise ValueError('M&Ms requires an explicit reference manifest or evaluation=none')
    if evaluation=='manifest' and not reference_manifest:raise ValueError('reference manifest required')
    if evaluation=='none' and (reference_root or reference_manifest):raise ValueError('unused reference inputs refused')
    root=str(Path(root).resolve());config=str(Path(config).resolve());split_manifest=str(Path(split_manifest).resolve())
    test_root=str(Path(test_root).resolve()) if official_test else None
    checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None
    teacher_config=str(Path(teacher_config).resolve()) if teacher_config else None
    reference_root=str(Path(reference_root or root).resolve()) if evaluation=='acdc' else None
    reference_manifest=str(Path(reference_manifest).resolve()) if reference_manifest else None
    # Paths only: no reference file is opened/fingerprinted before the frozen predictions.
    recipe=dict(schema='self_audit.full_guidance_recipe.v1',root=root,dataset=dataset,
        split_manifest=split_manifest,split_sha256=digest(split_manifest),splits=splits,
        official_acdc_test_unlabeled=official_test,test_root=test_root,
        config_sha256=digest(config),config=cfg.dictionary(),expected_teacher_sha256=WEIGHT_SHA256,
        expected_model_revision=HF_REVISION,epochs=epochs,batch_size=batch_size,tile_batch=tile_batch,
        width=width,lr=lr,seed=seed,device=device,threads=threads,
        checkpoint=checkpoint,teacher_config=teacher_config,
        teacher_config_sha256=digest(teacher_config) if teacher_config else None,
        checkpoint_selection='fixed_last_epoch_no_reference_feedback',
        evaluation=evaluation,reference_root=reference_root,reference_manifest=reference_manifest,
        source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    out=Path(out).resolve()
    if out.exists():
        if not resume:raise FileExistsError(out)
        if read_json(out/'RECIPE.json')!=__import__('json').loads(__import__('json').dumps(recipe)):
            raise ValueError('resume recipe/source/split mismatch; do not reinterpret old artifacts')
    else:
        out.mkdir(parents=True,exist_ok=False);write_json(out/'RECIPE.json',recipe)
    # Every stage uses the SAME effective patient split. The user's original
    # 80/20 manifest is never rewritten or repartitioned.
    effective_split_manifest=split_manifest
    if official_test:
        effective=out/'EFFECTIVE_SPLIT.json'
        payload=dict(train=splits['train'],val=splits['val'],test=splits['test'],
                     provenance=dict(source_split=split_manifest,source_sha256=digest(split_manifest),
                        test_root=test_root,test_status='official_acdc_unlabeled_export_only',
                        identity_source='raw_full_cine_filenames_only'))
        if effective.exists():
            if read_json(effective)!=payload:raise ValueError('effective split changed across resume')
        else:write_json(effective,payload)
        effective_split_manifest=str(effective)
    status=dict(schema='self_audit.full_guidance_run.v1',status='RUNNING',
                split_counts={s:len(v) for s,v in splits.items()},stages={},gt_used_for_training=False,
                gt_used_for_checkpoint_selection=False,quality_certified=False,
                official_test_unlabeled=official_test,test_root=test_root)
    def save():write_json(out/'RUN_STATUS.json',status)
    save()
    current='preflight'
    try:
        preflight_path=out/f'PREFLIGHT_{time.time_ns()}.json'
        run_stage('preflight',stage_command('preflight',root=root,dataset=dataset,
            split_manifest=effective_split_manifest,test_root=test_root,out=preflight_path,config=config,device=device,
            batch_size=batch_size,tile_batch=tile_batch,width=width,threads=threads,
            checkpoint=checkpoint,teacher_config=teacher_config,offline=offline),out)
        pf=read_json(preflight_path)
        if pf['status']!='PASS_EXECUTION_ONLY_QUALITY_UNVERIFIED':raise RuntimeError('real preflight did not pass')
        status['stages']['preflight']=str(preflight_path);save()
        if preflight_only:
            status['status']='PREFLIGHT_PASSED_NO_FULL_TRAINING';save();return status
        current='prepare';bank=out/'bank'
        bm=existing_stage(bank,'guidance_bank',resume)
        if bm is None:
            run_stage('prepare',stage_command('prepare',root=root,dataset=dataset,
                split_manifest=effective_split_manifest,config=config,out=bank,patients=0,seed=seed,device=device,
                tile_batch=tile_batch,checkpoint=checkpoint,teacher_config=teacher_config,offline=offline),out)
            bm=verify(bank,'guidance_bank')
        if bm['patients']!=splits['train']:raise ValueError('full run did not use ALL TRAIN patients')
        # Binding input identity across resume, preflight, feature bank and export.
        for pid,meta in bm['image_records'].items():
            if meta['image_sha256']!=pf['checks']['native_cine_inventory']['images'][pid]['image_sha256']:
                raise ValueError('bank input changed since preparation')
        status['stages']['bank']=bm['status'];save()
        current='train';student=out/'student'
        sm=existing_stage(student,'student',resume)
        if sm is None:
            run_stage('train',stage_command('train',bank=bank,out=student,epochs=epochs,
                batch_size=batch_size,device=device,width=width,lr=lr,seed=seed,threads=threads),out)
            sm=verify(student,'student')
        if sm['bank_manifest_sha256']!=digest(bank/'FROZEN.json'):raise ValueError('student bank lineage mismatch')
        status['stages']['student']=sm['status'];save()
        # Freeze BOTH cohorts before opening ANY reference mask.
        prediction_hashes={}
        for split in ('val','test'):
            current=f'export_{split}';pred=out/f'predictions_{split}'
            pm=existing_stage(pred,'student_predictions',resume)
            if pm is None:
                image_root=test_root if split=='test' and official_test else root
                run_stage(current,stage_command('export',student=student,root=image_root,dataset=dataset,
                    split_manifest=effective_split_manifest,split=split,out=pred,device=device,batch_size=batch_size),out)
                pm=verify(pred,'student_predictions')
            if pm['patients']!=splits[split] or pm['student_manifest_sha256']!=digest(student/'FROZEN.json'):
                raise ValueError('export cohort/checkpoint lineage mismatch')
            for pid,meta in pm['images'].items():
                if meta['image_sha256']!=pf['checks']['native_cine_inventory']['images'][pid]['image_sha256']:
                    raise ValueError('export image changed since preflight')
            prediction_hashes[split]=digest(pred/'FROZEN.json')
            status['stages'][current]='FROZEN';save()
        lock=dict(predictions=prediction_hashes,student_manifest_sha256=digest(student/'FROZEN.json'),
                  recipe_sha256=digest(out/'RECIPE.json'),selection='fixed_last_epoch')
        lock_path=out/'PREDICTIONS_LOCKED.json'
        if lock_path.exists():
            if read_json(lock_path)!=lock:raise ValueError('frozen prediction-pair receipt changed')
        else:write_json(lock_path,lock)
        if official_test:
            status['stages']['evaluate_test']='NOT_SCORED_OFFICIAL_ACDC_TEST_GT_UNAVAILABLE';save()
        if evaluation!='none':
            for split in (('val',) if official_test else ('val','test')):
                current=f'evaluate_{split}';dest=out/f'evaluation_{split}'
                ev=existing_stage(dest,'independent_evaluation',resume)
                if ev is None:
                    run_stage(current,stage_command('evaluate',predictions=out/f'predictions_{split}',out=dest,
                        reference_root=reference_root,reference_manifest=reference_manifest),out)
                    ev=verify(dest,'independent_evaluation')
                if ev['prediction_manifest_sha256']!=prediction_hashes[split]:raise ValueError('evaluation lineage mismatch')
                status['stages'][current]=read_json(dest/'METRICS.json')['summary'];save()
        status['status']=('COMPLETED_VAL_SCORED_OFFICIAL_TEST_UNSCORED' if official_test and evaluation!='none'
                          else 'COMPLETED_UNSCORED' if evaluation=='none' else 'COMPLETED_SCORED')
        if sm['status']=='REPRESENTATION_ONLY':status['status']+='_NO_SEMANTIC_SUPPORT'
        elif sm['status'].startswith('PARTIAL'):status['status']+='_PARTIAL_SEMANTIC_SUPPORT'
        save();print('[FULL]',status['status'],flush=True);return status
    except Exception as exc:
        status.update(status='BLOCKED',failed_stage=current,error=f'{type(exc).__name__}: {exc}')
        save();raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--split-manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--config',default=str(REPO/'configs/stable_guidance_experimental.json'))
    p.add_argument('--dataset',choices=['acdc','mnms'],default='acdc');p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--tile-batch',type=int,default=1)
    p.add_argument('--width',type=int,default=32);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--threads',type=int,default=4);p.add_argument('--device',default='cuda')
    p.add_argument('--checkpoint');p.add_argument('--teacher-config');p.add_argument('--offline',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--evaluation',choices=['acdc','manifest','none'],default='acdc')
    p.add_argument('--reference-root');p.add_argument('--reference-manifest')
    p.add_argument('--test-root',help='separate unlabeled official ACDC/testing directory; preserve existing TRAIN/VAL split')
    r=run_full(**vars(p.parse_args(argv)))
    return 2 if 'NO_SEMANTIC' in r['status'] or 'PARTIAL_SEMANTIC' in r['status'] else 0
