"""Support diagnostics are synthetic software evidence, never quality certification."""
from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
import torch

from self_audit_pseudolabel.system_v3 import CinePseudoTeacher
from self_audit_pseudolabel.trainer_v3 import ProgressiveTeacherTrainer
from test_pseudolabel_v3_checkup import ring, frozen
from test_pseudolabel_v3_regressions import _nifti_fixture


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(42)


def _fixed_teacher(monkeypatch, labels, semantic_bias=0.):
    model=CinePseudoTeacher(width=8,appearance_dim=16,motion_dim=8,fused_dim=24,k=4)
    q=torch.nn.functional.one_hot(labels,4).permute(2,0,1)[None].float()
    logits=torch.zeros(1,4,4); logits[:,:,0]=semantic_bias
    zero=torch.zeros(1,1,*labels.shape)
    base={'region_prob':q,'semantic_logits':logits,'region_features':torch.zeros(1,4,model.region_dim),
          'flow_prev':zero.repeat(1,2,1,1),'flow_next':zero.repeat(1,2,1,1)}
    monkeypatch.setattr(model,'forward',lambda *args:base)
    trainer=ProgressiveTeacherTrainer(model,torch.optim.AdamW(model.parameters(),lr=.001))
    cur=zero.repeat(1,3,1,1)
    batch={'prev':cur,'cur':cur,'nxt':cur,'patient_left_axis':['+x']}
    return trainer,batch


def test_diagnostics_distinguish_missing_topology_from_neural_rejection(monkeypatch):
    trainer,batch=_fixed_teacher(monkeypatch,ring(rv=True),semantic_bias=100.)
    output,_,_=trainer.infer_batch(batch)
    diag=output['seed_diagnostics']
    assert diag['enclosure_pairs']>0
    assert sum(diag[f'raw_seed_{c}_regions'] for c in ('rv','myo','lv'))>0
    assert sum(diag[f'accepted_{c}_regions'] for c in ('rv','myo','lv'))>0
    assert sum(diag[f'decoded_{c}_regions'] for c in ('rv','myo','lv'))==0
    assert not ((output['pseudo_label']>=1)&(output['pseudo_label']<=3)).any()


def test_no_enclosure_cannot_be_reported_as_foreground_seed(monkeypatch):
    trainer,batch=_fixed_teacher(monkeypatch,torch.zeros(32,32,dtype=torch.long))
    output,_,_=trainer.infer_batch(batch)
    diag=output['seed_diagnostics']
    assert diag['enclosure_pairs']==0
    for stage in ('raw_seed','accepted','decoded'):
        assert sum(diag[f'{stage}_{c}_regions'] for c in ('rv','myo','lv'))==0
    assert not ((output['pseudo_label']>=1)&(output['pseudo_label']<=3)).any()


def test_validation_foreground_cannot_satisfy_training_support(frozen,tmp_path):
    import numpy as np
    from self_audit_pseudolabel.freeze import verify_frozen,export_pseudo_npz,_digest
    from self_audit_pseudolabel.pipeline_v3 import student_main
    data,run=frozen
    payload=verify_frozen(run/'FROZEN.json',expected_role='teacher_freeze')
    for entry in payload['entries']:
        if entry['split']!='train': continue
        path=run/entry['path']
        with np.load(path,allow_pickle=False) as arrays:
            label=np.zeros_like(arrays['pseudo_label'])
            metadata=json.loads(str(arrays['metadata_json']))
        valid=np.ones_like(label)
        soft=np.eye(4,dtype=np.float32)[label].transpose(2,0,1)
        path.unlink()
        entry.update(sha256=export_pseudo_npz(path,pseudo_label=label,valid=valid,soft_label=soft,metadata=metadata),
                     class_pixels=[label.size,0,0,0],valid_foreground=0,valid_fraction=1.)
    payload.pop('manifest_id');payload['manifest_id']=_digest(payload)
    (run/'FROZEN.json').write_text(json.dumps(payload))
    verify_frozen(run/'FROZEN.json',expected_role='teacher_freeze')
    assert sum(e['valid_foreground'] for e in payload['entries'])>0
    with pytest.raises(ValueError,match='NO_FOREGROUND_SEEDS: frozen TRAIN split'):
        student_main(['--dataset','acdc','--root',str(data),'--manifest',str(run/'FROZEN.json'),
                      '--out',str(tmp_path/'student.pt'),'--threads','1','--no-progress'])
    assert not (tmp_path/'student.pt').exists()


def test_teacher_reports_complete_support_even_without_sampled_logging(tmp_path,monkeypatch,capsys):
    import nibabel as nib
    from self_audit_pseudolabel.pipeline_v3 import teacher_main
    from self_audit_pseudolabel.freeze import verify_frozen
    data,split=_nifti_fixture(tmp_path)
    load=nib.load
    def guard(path,*args,**kwargs):
        assert '_gt' not in str(path), 'teacher attempted to open reference labels'
        return load(path,*args,**kwargs)
    monkeypatch.setattr(nib,'load',guard)
    out=tmp_path/'teacher'
    summary=teacher_main(['--dataset','acdc','--root',str(data),'--split-manifest',str(split),
                         '--out',str(out),'--epochs','1','--max-train-batches','2',
                         '--threads','1','--log-every','0','--no-progress'])
    payload=verify_frozen(out/'FROZEN.json',expected_role='teacher_freeze')
    history=json.loads((out/'train_metrics.json').read_text())
    for row in history:
        assert sum(row[f'accepted_{c}_regions'] for c in ('bg','rv','myo','lv'))==row['accepted_regions']
        for c in ('bg','rv','myo','lv'):
            assert row[f'raw_seed_{c}_regions']>=row[f'accepted_{c}_regions']
    report=json.loads((out/'export_report.json').read_text())
    for row in report['patient_rows']:
        assert 'raw_seed_lv_regions' in row['seed_region_counts']
        assert all(a>=b for a,b in zip(row['pre_consistency_class_pixels'],row['class_pixels']))
        assert row['consistency_rejected_class_pixels']==[
            a-b for a,b in zip(row['pre_consistency_class_pixels'],row['class_pixels'])]
    for split,stats in summary['split_support'].items():
        entries=[e for e in payload['entries'] if e['split']==split]
        assert stats['valid_foreground_pixels']==sum(e['valid_foreground'] for e in entries)
        assert stats['class_pixels']==[sum(e['class_pixels'][c] for e in entries) for c in range(4)]
        assert stats['support_status']==('FOREGROUND_OBSERVED' if stats['valid_foreground_pixels'] else 'NO_FOREGROUND_SEEDS')
    assert summary['teacher_ready']=='NOT_EVALUATED'
    assert not (out/'train_metrics.jsonl').exists()
    assert '[SEEDS]' in capsys.readouterr().out


@pytest.mark.parametrize('failed_stage',['pseudo-eval','student'])
def test_failed_pipeline_preserves_completed_evaluation_and_failed_wandb(tmp_path,monkeypatch,failed_stage):
    from scripts import run_full_pipeline_v3 as runner
    fake=MagicMock(); monkeypatch.setitem(sys.modules,'wandb',fake)
    out=tmp_path/'run'; launched=[]
    def stage(name,cmd,env,fh,tracker=None):
        launched.append(name)
        if name==failed_stage:
            raise subprocess.CalledProcessError(1,cmd)
        if name=='teacher':
            (out/'teacher').mkdir()
            (out/'teacher'/'run_summary.json').write_text(json.dumps({'valid_foreground_pixels':0}))
        elif name=='pseudo-eval':
            (out/'evaluation_val.json').write_text(json.dumps({'foreground_mean':0.,'lv':0.,'known_fraction':0.}))
        return .1
    monkeypatch.setattr(runner,'run_stage',stage)
    with pytest.raises(subprocess.CalledProcessError):
        runner.main(['--dataset','acdc','--root',str(tmp_path),'--split-manifest',str(tmp_path/'split.json'),
                     '--out',str(out),'--device','cpu','--train-student','--wandb','--wandb-mode','offline'])
    summary=json.loads((out/'PIPELINE_SUMMARY.json').read_text())
    assert summary['status']=='failed' and summary['failed_stage']==failed_stage
    assert summary['return_code']==1
    assert summary['completed_stages']==(['teacher','pseudo-eval'] if failed_stage=='student' else ['teacher'])
    assert summary['student'] is None and summary['student_predictions'] is None
    assert summary['student_evaluation'] is None
    if failed_stage=='student':
        assert summary['evaluation']==str(out/'evaluation_val.json')
        assert summary['pseudo_metrics']['foreground_mean']==0.
    else:
        assert summary['evaluation'] is None
    assert 'student-inference' not in launched
    fake.finish.assert_called_once_with(exit_code=1)
