#!/usr/bin/env python3
"""Continue at a completed-arm boundary; never resume a partial checkpoint.

Only CPU threading and an input-identical TRAIN metadata cache may differ.
Completed arms are verified and retained. Sampling and all science settings stay
fixed. Separate runtime launch hashes accompany cached training checkpoints.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_teacher_memory_full import Experiment, write
from self_audit.engine import source_digest
from self_audit.model import digest


def inspect_completed(root, arm, steps, expected_source):
    out = root/'runs'/arm
    report = json.loads((out/'REPORT.json').read_text())
    metadata = json.loads((out/'final/MANIFEST.json').read_text())
    config = json.loads((out/'CONFIG.json').read_text())
    assert report['status'] == 'COMPLETED' and report['steps'] == steps
    assert report['ema_updates'] == steps and not report['label_reads_outside_budget']
    assert metadata['source_sha256'] == expected_source == source_digest()
    assert digest(out/'final/MANIFEST.json') == report['final_manifest_sha256']
    assert digest(out/'final/weights.pt') == metadata['payload_sha256']
    assert config['batch_size'] == 24 and config['crop_size'] == 256
    assert config['seed'] == 42 and config['warmup_steps'] == 1000
    assert metadata['data_sha256'] == 'e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4'
    h = hashlib.sha256(); peak = 0; count = 0
    for line in (out/'steps.jsonl').read_text().splitlines():
        row = json.loads(line); count += 1
        assert row['supervised']['patient'] != row['audit']['patient']
        h.update(json.dumps({k:row[k] for k in ('supervised','audit','unlabeled')}, sort_keys=True).encode())
        peak = max(peak, row.get('cuda_peak_allocated') or 0)
    assert count == steps
    result = dict(arm=arm, report=report, schedule_sha256=h.hexdigest(), peak_allocated_bytes=peak,
                  checkpoint_manifest_sha256=digest(out/'final/MANIFEST.json'))
    runtime = out/'RUNTIME_PERFORMANCE.json'
    if runtime.exists(): result['runtime_performance_sha256'] = digest(runtime)
    return result


class ContinuedExperiment(Experiment):
    def __init__(self, root, override):
        super().__init__(root, 10000, 1000)
        self.override = override
        self.env.update(OMP_NUM_THREADS=str(override['threads']), MKL_NUM_THREADS=str(override['threads']))

    def command(self, cmd, log, **kwargs):
        cmd = list(cmd)
        if len(cmd)>1 and str(cmd[1]) == 'train.py' and self.override['cached_inputs']:
            cmd[1] = 'scripts/runtime_performance.py'
            if self.override.get('prefetch_inputs'): cmd.insert(2, '--prefetch-inputs')
        return super().command(cmd, log, **kwargs)

    def main(self):
        plan = json.loads((self.root/'PROTOCOL.json').read_text())
        assert plan['source_sha256'] == self.source
        assert plan['steps'] == 10000 and plan['warmup_steps'] == 1000
        assert self.override['batch'] == 24
        state = json.loads((self.root/'OFFLINE_STATUS.json').read_text())
        assert state['stage'] == 'COMPLETED'
        bank, cache = Path(state['bank']), Path(state['cache'])
        assert digest(bank) == state['bank_sha256'] and digest(cache) == state['cache_sha256']
        training = []; schedule = None
        for arm in ('A0','A1','A2','A3','A4','A5','A6'):
            if (self.root/'runs'/arm/'REPORT.json').exists():
                result = inspect_completed(self.root, arm, self.steps, self.source)
            else:
                assert arm not in ('A0','A1'), 'A0 and A1 must finish before handoff'
                assert not (self.root/'runs'/arm).exists(), 'partial arm cannot be resumed'
                active_bank = bank
                if arm == 'A5':
                    self.idle_gpu(); self.status('BUILDING_FULL_A5_FROM_PRESELECTED_A0_FINAL')
                    native=self.root/'native-train-full-declared.json';cp=self.root/'runs/A0/final';dest=self.root/'student-bank'
                    self.command([sys.executable,'scripts/optimize_a5_extraction.py','--root',self.root],'build-a5-full.log')
                    receipt=json.loads((dest/'BUILD_RECEIPT.json').read_text())
                    assert receipt['coverage_status']=='FULL_TRAIN' and receipt['mask_paths_read']==0
                    active_bank=dest/'fixed_bank.npz'
                c = self.recipe(arm, 24, self.override['threads'], active_bank, cache)
                self.run_training(arm, c)
                result = inspect_completed(self.root, arm, self.steps, self.source)
            if schedule is None: schedule = result['schedule_sha256']
            assert result['schedule_sha256'] == schedule, 'sampling mismatch'
            training.append(result); write(self.root/'TRAINING_RESULTS.json', training)
        write(self.root/'FROZEN_FINAL_CHECKPOINTS.json', training)
        results=[]
        for arm in ('A0','A1','A2','A3','A4','A5','A6'):
            midpoint=self.root/'runs'/arm/f'step_{self.steps//2:06d}'
            if midpoint.exists(): results.append(self.evaluate(arm,'VAL18',midpoint.name))
            results.append(self.evaluate(arm,'VAL18'))
            results.append(self.evaluate(arm,'TEST50'))
            write(self.root/'RESULTS.json',results)
            with (self.root/'RESULTS.csv').open('w') as f:
                fields=['arm','split','checkpoint','BG','RV','MYO','LV','all_class_patient_macro','foreground_patient_macro']
                writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
                for r in results: writer.writerow(dict(arm=r['arm'],split=r['split'],checkpoint=r['checkpoint'],
                    **{k:r['macro_patient'][k]['dice'] for k in ('BG','RV','MYO','LV')},
                    all_class_patient_macro=r['all_class_patient_macro'],foreground_patient_macro=r['foreground_patient_macro']))
        self.command([sys.executable,'scripts/check_repo.py'],'check-repo-final.log')
        self.command([sys.executable,'-m','pytest','-q'],'pytest-final.log',env=dict(self.env,CUDA_VISIBLE_DEVICES=''))
        self.status('COMPLETED', results=str(self.root/'RESULTS.csv'),all_seven_arms=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--start-gate',type=Path)
    args=p.parse_args();root=args.root.resolve()
    override=json.loads((root/'PERFORMANCE_OVERRIDE.json').read_text())
    exp=ContinuedExperiment(root,override)
    assert override['status']=='MEASURED_AND_VERIFIED'
    assert override['canonical_source_sha256']==exp.source
    assert digest(ROOT/'scripts/runtime_performance.py')==override['runtime_launcher_sha256']
    if override.get('prefetch_inputs'):
        assert digest(ROOT/'scripts/prefetch_runtime.py')==override['prefetch_launcher_sha256']
    if args.start_gate:
        # The old queue remains stopped and alive until this child proves ready.
        verified=[inspect_completed(root,a,10000,exp.source) for a in ('A0','A1')]
        assert verified[0]['schedule_sha256']==verified[1]['schedule_sha256']
        write(args.start_gate.with_suffix('.ready'),dict(pid=os.getpid(),status='READY'))
        deadline=time.monotonic()+60
        while not args.start_gate.exists():
            if time.monotonic()>deadline: raise RuntimeError('handoff gate timeout; no training started')
            time.sleep(.2)
    write(root/'CONTINUATION_PID.json',dict(pid=os.getpid(),source_sha256=exp.source,
        override_sha256=digest(root/'PERFORMANCE_OVERRIDE.json'),launcher_sha256=digest(Path(__file__))))
    try: exp.main()
    except BaseException as exc: exp.status('FAILED',error=repr(exc));raise
