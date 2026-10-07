"""Synthetic telemetry regressions, never evidence of segmentation quality."""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from self_audit_pseudolabel.progress_v3 import (
    TrainingProgress, coverage_metrics, EVENT_PREFIX, EVENT_SCHEMA,
)
from self_audit_pseudolabel.system_v3 import CinePseudoTeacher
from self_audit_pseudolabel.trainer_v3 import ProgressiveTeacherTrainer
from test_pseudolabel_v3_checkup import frozen
from test_pseudolabel_v3_regressions import _nifti_fixture


@pytest.fixture(autouse=True)
def single_cpu_thread():
    torch.set_num_threads(1)


def read_events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_jsonl_flush_cadence_crash_retention_and_bounded_eta(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr('self_audit_pseudolabel.progress_v3.time.perf_counter', lambda: now[0])
    path = tmp_path / 'metrics.jsonl'
    with pytest.raises(RuntimeError, match='training interrupted'):
        with TrainingProgress(stage='student',path=path,epochs=2,batches_per_epoch=3,
                              log_every=50,enabled=False) as progress:
            progress.start_epoch(0)
            for step in range(3):
                now[0] += 2.0
                progress.update(step,batch_samples=2,metrics={'skipped_empty':True,'optimizer_steps':0})
                if step == 0:
                    row = read_events(path)[0]['metrics']  # visible before close/stage end
                    assert row['epoch_eta_seconds'] == 4.0
                    assert row['training_eta_seconds'] == 10.0
                    assert row['samples_per_second'] == 1.0
            rows = read_events(path)
            assert [r['metrics']['batch'] for r in rows] == [1,3]
            assert rows[-1]['metrics']['epoch_eta_seconds'] == 0.0
            assert rows[-1]['metrics']['training_eta_seconds'] == 6.0
            raise RuntimeError('training interrupted')
    assert len(read_events(path)) == 2


def test_cadence_crosses_epochs_without_duplicate_last_batch(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    with TrainingProgress(stage='teacher',path=path,epochs=2,batches_per_epoch=4,
                          log_every=2,enabled=False) as progress:
        for epoch in range(2):
            progress.start_epoch(epoch)
            for step in range(4):
                progress.update(step,batch_samples=1,metrics={'total':0.1})
    rows = [r['metrics'] for r in read_events(path)]
    assert [r['global_batch'] for r in rows] == [1,2,4,5,6,8]
    assert rows[-1]['training_eta_seconds'] == 0.0


def test_disabled_logging_and_invalid_interval(tmp_path):
    path = tmp_path / 'off.jsonl'
    with TrainingProgress(stage='teacher',path=path,epochs=1,batches_per_epoch=1,
                          log_every=0,enabled=False) as progress:
        progress.start_epoch(0)
        progress.update(0,batch_samples=1,metrics={'total':1.0})
    assert not path.exists()
    with pytest.raises(ValueError,match='nonnegative'):
        TrainingProgress(stage='teacher',path=path,epochs=1,batches_per_epoch=1,log_every=-1)


def test_coverage_counts_valid_pixels_not_region_counts_or_accuracy():
    target = torch.tensor([[[0,1,2],[3,255,3]]])
    valid = target != 255
    metrics = coverage_metrics(target,valid,prefix='target')
    assert metrics['target_valid_pixels'] == 5
    assert metrics['target_foreground_pixels'] == 4
    assert metrics['target_valid_fraction'] == pytest.approx(5/6)
    assert metrics['target_lv_fraction'] == pytest.approx(2/6)
    assert metrics['target_bg_pixels'] == 1
    assert not any('accuracy' in key or 'dice' in key for key in metrics)
    empty = coverage_metrics(torch.full_like(target,255),torch.zeros_like(valid),prefix='target')
    assert empty['target_foreground_pixels'] == empty['target_valid_fraction'] == 0


def test_teacher_logging_preserves_exact_updates_rng_and_single_forward():
    torch.manual_seed(17)
    model = CinePseudoTeacher(width=8,appearance_dim=16,motion_dim=8,fused_dim=24,k=4)
    other = copy.deepcopy(model)
    a = ProgressiveTeacherTrainer(model,torch.optim.AdamW(model.parameters(),lr=1e-3))
    b = ProgressiveTeacherTrainer(other,torch.optim.AdamW(other.parameters(),lr=1e-3))
    cur = torch.rand(1,3,16,16)
    batch = {'prev':cur.roll(-1,-1),'cur':cur,'nxt':cur.roll(1,-1),'patient_left_axis':['+x']}
    calls = []
    hook = other.register_forward_hook(lambda *_: calls.append(1))
    for _ in range(3):
        before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        python_before = random.getstate()
        plain = a.train_batch(batch,collect_metrics=False)
        logged = b.train_batch(batch,collect_metrics=True)
        assert plain == logged
        assert torch.equal(before,torch.get_rng_state())
        assert numpy_before[0] == np.random.get_state()[0]
        assert np.array_equal(numpy_before[1],np.random.get_state()[1])
        assert python_before == random.getstate()
        for key,value in model.state_dict().items():
            assert torch.equal(value,other.state_dict()[key]), key
        assert torch.equal(a.bank.prototypes,b.bank.prototypes)
        assert torch.equal(a.bank.counts,b.bank.counts)
    hook.remove()
    assert len(calls) == 3
    assert 'decoded_valid_fraction' in b.last_metrics
    assert sum(b.last_metrics[f'accepted_{c}_regions'] for c in ('bg','rv','myo','lv')) == logged[1]


def test_teacher_pipeline_logging_on_off_identical_weights_exports_and_no_gt(tmp_path,monkeypatch):
    import nibabel as nib
    from self_audit_pseudolabel.pipeline_v3 import teacher_main
    from self_audit_pseudolabel.freeze import verify_frozen
    data,split = _nifti_fixture(tmp_path)
    load = nib.load
    def guard(path,*args,**kwargs):
        assert '_gt' not in str(path)
        return load(path,*args,**kwargs)
    monkeypatch.setattr(nib,'load',guard)
    outputs = []
    for interval in (0,1):
        out = tmp_path / f'teacher-{interval}'
        teacher_main(['--dataset','acdc','--root',str(data),'--split-manifest',str(split),
                      '--out',str(out),'--epochs','1','--max-train-batches','2',
                      '--threads','1','--batch-size','2','--log-every',str(interval),'--no-progress'])
        verify_frozen(out/'FROZEN.json')
        outputs.append(out)
    left,right = [torch.load(p/'teacher.pt',weights_only=True) for p in outputs]
    for key,value in left['model'].items():
        assert torch.equal(value,right['model'][key])
    assert torch.equal(left['prototype_bank'],right['prototype_bank'])
    assert json.loads((outputs[0]/'train_metrics.json').read_text()) == json.loads((outputs[1]/'train_metrics.json').read_text())
    for path in (outputs[0]/'pseudo').rglob('*.npz'):
        with np.load(path) as a,np.load(outputs[1]/path.relative_to(outputs[0])) as b:
            for key in ('pseudo_label','valid','soft_label'):
                assert np.array_equal(a[key],b[key])
    assert not (outputs[0]/'train_metrics.jsonl').exists()
    rows = read_events(outputs[1]/'train_metrics.jsonl')
    assert len(rows) == 2 and rows[-1]['metrics']['epoch_eta_seconds'] == 0


def test_student_logging_on_off_identical_weights_skips_and_no_gt(frozen,tmp_path,monkeypatch):
    import nibabel as nib
    from self_audit_pseudolabel.pipeline_v3 import student_main,FrozenPseudoDataset
    data,run = frozen
    load = nib.load
    get = FrozenPseudoDataset.__getitem__
    def guard(path,*args,**kwargs):
        assert '_gt' not in str(path)
        return load(path,*args,**kwargs)
    def one_empty(self,index):
        row = get(self,index)
        if index == 0:
            row['valid'].zero_()
            row['target'].fill_(255)
        return row
    monkeypatch.setattr(nib,'load',guard)
    monkeypatch.setattr(FrozenPseudoDataset,'__getitem__',one_empty)
    outputs = []
    results = []
    rng_states = []
    for interval in (0,1):
        path = tmp_path / f'student-{interval}.pt'
        result = student_main(['--dataset','acdc','--root',str(data),'--manifest',str(run/'FROZEN.json'),
                               '--out',str(path),'--epochs','2','--threads','1','--profile','balanced',
                               '--log-every',str(interval),'--no-progress'])
        outputs.append(path)
        results.append(result)
        rng_states.append(torch.get_rng_state().clone())
    assert torch.equal(*rng_states)
    left,right = [torch.load(p,weights_only=True) for p in outputs]
    for key,value in left['model'].items():
        assert torch.equal(value,right['model'][key]),key
    assert results[0]['history'] == results[1]['history']
    assert results[0]['coverage'] == results[1]['coverage']
    rows = read_events(outputs[1].with_suffix('.metrics.jsonl'))
    assert len(rows) == 12
    assert sum(r['metrics']['skipped_empty'] for r in rows) == 2
    assert rows[-1]['metrics']['optimizer_steps'] == 10
    assert rows[-1]['metrics']['training_eta_seconds'] == 0
    assert all('loss' not in r['metrics'] for r in rows if r['metrics']['skipped_empty'])


def test_subprocess_stream_is_live_before_exit_and_preserves_tqdm_cr(tmp_path,capsys):
    from scripts.run_full_pipeline_v3 import run_stage
    ack = tmp_path / 'ack'
    event = EVENT_PREFIX + json.dumps({'schema':EVENT_SCHEMA,'stage':'teacher',
                                     'metrics':{'global_batch':1,'total':0.25,'label':'café'}},ensure_ascii=False) + '\n'
    # The child will fail if telemetry is only delivered after subprocess exit.
    code = f'''import os,sys,time
assert os.environ['PYTHONUNBUFFERED']=='1'
sys.stderr.write('\\rteacher 1/1: 1/2 ETA 1s\\n');sys.stderr.flush()
raw={event!r}.encode('utf-8')
split=raw.index('é'.encode('utf-8'))+1
for chunk in (raw[:3],raw[3:split],raw[split:]):
    sys.stdout.buffer.write(chunk);sys.stdout.flush();time.sleep(.02)
start=time.monotonic()
while not os.path.exists({str(ack)!r}):
    if time.monotonic()-start>5: raise RuntimeError('not live')
    time.sleep(.01)
print('ACK received before exit')
'''
    tracker = MagicMock()
    tracker.log.side_effect = lambda payload: ack.write_text(json.dumps(payload))
    log = io.StringIO()
    run_stage('teacher',[sys.executable,'-c',code],os.environ.copy(),log,tracker=tracker)
    tracker.log.assert_called_once()
    assert json.loads(ack.read_text())['teacher']['total'] == .25
    output = capsys.readouterr().out
    assert '\rteacher 1/1' in output and 'ACK received before exit' in output
    assert 'teacher 1/1' in log.getvalue()


def test_stream_rejects_wrong_stage_and_malformed_records():
    from scripts.run_full_pipeline_v3 import _stream_event
    tracker = MagicMock()
    for line in ('ordinary output',EVENT_PREFIX+'broken',EVENT_PREFIX+'[]',
                 EVENT_PREFIX+json.dumps({'schema':EVENT_SCHEMA,'stage':'student','metrics':{}})):
        _stream_event(line,'teacher',tracker)
    tracker.log.assert_not_called()


def test_wandb_summary_does_not_replay_live_history(tmp_path,monkeypatch):
    from self_audit_pseudolabel.wandb_v3 import WandbV3Tracker
    fake = MagicMock()
    monkeypatch.setitem(sys.modules,'wandb',fake)
    tracker = WandbV3Tracker(enabled=True,mode='offline',project='test',entity=None,
                           run_name='test',run_dir=tmp_path,config={})
    tracker.log({'student':{'global_batch':1,'loss':0.5}})
    path = tmp_path/'student.json'
    path.write_text(json.dumps({'optimizer_steps':1,'history':[{'loss':0.5}]}))
    tracker.log_json('student_summary',path,include_history=False)
    tracker.finish()
    fake.init.assert_called_once()
    assert fake.log.call_count == 2
    assert fake.log.call_args_list[1].args[0] == {'student_summary/optimizer_steps':1}


def test_cli_propagates_logging_and_progress_flags(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable,str(root/'scripts/run_full_pipeline_v3.py'),
        '--dataset','acdc','--root',str(tmp_path),'--split-manifest',str(tmp_path/'split.json'),
        '--out',str(tmp_path/'dry'),'--device','cpu','--train-student','--dry-run',
        '--log-every','7','--no-progress'],capture_output=True,text=True,check=True)
    commands = [line for line in result.stdout.splitlines() if line.startswith('[DRYRUN]')]
    assert len(commands) == 5
    assert all('--log-every 7' in commands[i] for i in (0,2))
    assert all('--no-progress' in commands[i] for i in (0,2,3))


def test_orchestrator_streams_without_replaying_final_histories(tmp_path,monkeypatch):
    from scripts import run_full_pipeline_v3 as runner
    fake = MagicMock()
    monkeypatch.setitem(sys.modules,'wandb',fake)
    out = tmp_path/'pipeline'
    def stage(name,cmd,env,fh,tracker=None):
        if name == 'teacher':
            folder = out/'teacher';folder.mkdir()
            (folder/'run_summary.json').write_text(json.dumps({'optimizer_steps':1}))
            (folder/'train_metrics.json').write_text(json.dumps([{'total':999}]))
            runner._stream_event(EVENT_PREFIX+json.dumps({'schema':EVENT_SCHEMA,'stage':name,
                'metrics':{'global_batch':1,'total':.25}}),name,tracker)
        elif name == 'student':
            (out/'student_balanced.json').write_text(json.dumps({'optimizer_steps':1,'history':[{'loss':999}]}))
            runner._stream_event(EVENT_PREFIX+json.dumps({'schema':EVENT_SCHEMA,'stage':name,
                'metrics':{'global_batch':1,'loss':.5}}),name,tracker)
        elif name in ('pseudo-eval','student-eval'):
            Path(cmd[cmd.index('--out')+1]).write_text(json.dumps({'foreground_mean':.1}))
        return .01
    monkeypatch.setattr(runner,'run_stage',stage)
    runner.main(['--dataset','acdc','--root',str(tmp_path),'--split-manifest',str(tmp_path/'split.json'),
                 '--out',str(out),'--device','cpu','--train-student','--wandb','--wandb-mode','offline'])
    fake.init.assert_called_once()
    fake.finish.assert_called_once()
    payloads = [call.args[0] for call in fake.log.call_args_list]
    assert [p['teacher/total'] for p in payloads if 'teacher/total' in p] == [.25]
    assert [p['student/loss'] for p in payloads if 'student/loss' in p] == [.5]
    assert not any(value == 999 for p in payloads for value in p.values())


def test_student_refuses_stale_metrics_even_when_logging_is_disabled(frozen,tmp_path):
    from self_audit_pseudolabel.pipeline_v3 import student_main
    data,run = frozen
    out = tmp_path/'retry.pt'
    metrics = out.with_suffix('.metrics.jsonl')
    metrics.write_text('old interrupted run\n')
    with pytest.raises(FileExistsError):
        student_main(['--dataset','acdc','--root',str(data),'--manifest',str(run/'FROZEN.json'),
                      '--out',str(out),'--log-every','0','--no-progress'])
    assert metrics.read_text() == 'old interrupted run\n'
    assert not out.exists()
