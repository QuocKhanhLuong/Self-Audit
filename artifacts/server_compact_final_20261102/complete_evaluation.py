"""Finish already-authorized VAL/TEST evaluation without altering old backup receipts."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

RUNTIME = Path('/workspace/Self-Audit-kd-local-20261102')
BASE = Path('/workspace/selfaudit-20261101')
OUT = BASE / 'final_eval_after_git_20261102'
sys.path.insert(0, str(RUNTIME))
from self_audit.server_runner import (
    atomic_json, sha256_file, canonical_sha256, _read_validation_log, _select_best,
    evaluate_checkpoint, _write_best_artifacts, _write_supplemental_evaluation_report,
)
from self_audit.compact_inference import run_two_phase_evaluation

RUNS = {
    'S0': BASE / 'runs/s0_acdc_l8_b32_seed42',
    'S1': BASE / 'kd_local_20261102/arms/S1/run',
    'S2': BASE / 'kd_local_20261102/arms/S2/run',
}
FINAL_PINS = {
    'S0': 'b9263aa8b4fcbc9511d40d5b377f7b2d1044bbf2c1c2ba942c2020a618c50d39',
    'S1': '31d3350f2e0df0949f755fcef35c29755fe1b3f20c2c7421cd6a5250bf991dad',
    'S2': '2431742cd74730eeb95359c29b7c28a255602428a8e26d215c2ebb23b269ec5d',
}
RUNTIME_HEAD = '543c968504e0ddda8a14813a11b790b0a0f266a5'
STATE = OUT / 'STATUS.json'


def event(status, **extra):
    data = {'status': status, 'updated_epoch': time.time(), 'pid': os.getpid(),
            'training': 'NOT_RUN_EVALUATION_ONLY', 'drive_backup': 'UPLOAD_PENDING', **extra}
    atomic_json(STATE, data)
    print(json.dumps(data, sort_keys=True), flush=True)


def preflight():
    head = subprocess.check_output(['git', '-C', str(RUNTIME), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == RUNTIME_HEAD
    plan = {'schema': 'self_audit.completed_run_evaluation.v1', 'runtime_git_head': head,
            'script_sha256': sha256_file(__file__), 'new_training_steps': 0,
            'authorization': 'User authorized VAL/TEST50 and deferred Drive backup: ko can lo backup drive, xong r push 1 the cx duoc',
            'selection_rule': 'highest VAL class_first_macro4; exact ties keep earliest step',
            'eval_batch_size': 8, 'normalization': 'prepared',
            'runtime_files': {}, 'arms': {}, 'old_receipt_pins': {}}
    files = ['self_audit/compact_inference.py', 'self_audit/evaluation.py',
             'self_audit/compact_model.py', 'self_audit/compact_artifacts.py', 'self_audit/server_runner.py']
    for relative in files:
        plan['runtime_files'][relative] = sha256_file(RUNTIME / relative)
    for arm, run in RUNS.items():
        lock = json.loads((run / 'RUN_LOCK.json').read_text())
        final = json.loads((run / 'FULL_RESUME_CHECKPOINT.json').read_text())
        cp = Path(final['checkpoint_path'])
        assert final['manifest_sha256'] == FINAL_PINS[arm] == sha256_file(cp / 'MANIFEST.json')
        assert final['payload_sha256'] == sha256_file(cp / 'weights.pt')
        assert final['step'] == 10000 == lock['protocol']['steps']
        assert final['config_sha256'] == canonical_sha256(lock['config'])
        assert lock['protocol']['eval_batch_size'] == 8
        assert lock['protocol']['best_rule'] == plan['selection_rule']
        assert json.loads((run / 'RUN_STATUS.json').read_text())['training_complete'] is True
        entries = _read_validation_log(run)
        assert sorted(int(row['step']) for row in entries) == list(range(1000, 10000, 1000))
        for role in ('train', 'val', 'test'):
            ref = lock['inputs'][role]
            assert sha256_file(ref['path']) == ref['sha256']
        for relative in files[:4]:
            expected = lock['source']['files'][relative]
            actual = plan['runtime_files'][relative]
            if expected != actual:
                assert arm == 'S0' and relative == 'self_audit/compact_inference.py'
                old = subprocess.check_output(['git', '-C', str(RUNTIME), 'show',
                    lock['source']['git_head'] + ':' + relative], text=True)
                assert hashlib.sha256(old.encode()).hexdigest() == expected
                current = (RUNTIME / relative).read_text()
                old_line = '    metrics["evaluation_source_manifest_sha256"] = manifest_data.get("geometry_provenance", {}).get("parent_manifest_sha256", actual_manifest_sha)\n'
                start = current.index('    geometry_provenance = manifest_data.get("geometry_provenance")\n')
                end = current.index('    metrics["geometry_provenance"] =', start)
                assert old.replace(old_line, current[start:end], 1) == current
                plan['s0_metadata_only_compatibility'] = {
                    'old_sha256': expected, 'runtime_sha256': actual,
                    'difference': 'Handle string geometry_provenance when writing evaluation_source_manifest_sha256; numerical prediction/metric code unchanged',
                }
        for name in ('RUN_LOCK.json', 'RUN_STATUS.json', 'BEST.json', 'BEST_SELECTION.json',
                     'VALIDATION_LOG.jsonl', 'FULL_RESUME_CHECKPOINT.json', 'best.pt',
                     'BACKUP_STATUS.json', 'BACKUP_STATUS_core.json'):
            path = run / name
            if path.is_file():
                plan['old_receipt_pins'][str(path)] = sha256_file(path)
        plan['arms'][arm] = {'run_root': str(run), 'lock_sha256': sha256_file(run / 'RUN_LOCK.json'),
                             'final': final, 'previous_selection': _select_best(entries)}
    assert shutil.disk_usage(BASE).free >= 10 * 1024 ** 3
    atomic_json(OUT / 'PLAN.json', plan)
    event('READY', plan_sha256=sha256_file(OUT / 'PLAN.json'))
    return plan


def run(expected_plan_sha):
    assert sha256_file(OUT / 'PLAN.json') == expected_plan_sha
    plan = json.loads((OUT / 'PLAN.json').read_text())
    assert plan['script_sha256'] == sha256_file(__file__)
    for relative, expected in plan['runtime_files'].items():
        assert sha256_file(RUNTIME / relative) == expected
    for path, expected in plan['old_receipt_pins'].items():
        assert sha256_file(path) == expected
    assert json.loads(STATE.read_text())['status'] == 'READY', 'No duplicate run or silent restart'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    results = {}
    for arm, spec in plan['arms'].items():
        old = Path(spec['run_root'])
        target = OUT / arm
        target.mkdir(exist_ok=False)
        lock = json.loads((old / 'RUN_LOCK.json').read_text())
        assert sha256_file(old / 'RUN_LOCK.json') == spec['lock_sha256']
        cp = Path(spec['final']['checkpoint_path'])
        for name in ('RUN_LOCK.json', 'LOCKED_CONFIG.json', 'PROVENANCE.json', 'VALIDATION_LOG.jsonl'):
            shutil.copyfile(old / name, target / name)
        event('RUNNING', arm=arm, phase='final_10000_VAL', results=results)
        evaluate_checkpoint(cp, FINAL_PINS[arm], lock=lock, run_root=target, device='cuda')
        entries = _read_validation_log(target)
        assert sorted(int(row['step']) for row in entries) == list(range(1000, 10001, 1000))
        best = _write_best_artifacts(target, lock, cp, FINAL_PINS[arm])
        frozen = Path(best['predictions_frozen_path'])
        assert sha256_file(frozen) == best['predictions_frozen_sha256']
        frozen_data = json.loads(frozen.read_text())
        assert frozen_data['coverage']['complete'] is True
        assert frozen_data['export_manifest_sha256'] == best['export_manifest_sha256']
        assert frozen_data['evaluation_manifest_sha256'] == lock['inputs']['val']['sha256']
        # Persist selection before the TEST evaluator may open any TEST references.
        atomic_json(target / 'SELECTION_LOCK_BEFORE_TEST.json', {
            'best': best, 'validation_checkpoint_count': len(entries),
            'selection_complete': True, 'test_used_for_selection': False,
            'selected_val_predictions_location': str(frozen),
            'prior_val_artifacts_remain_in_original_run': True,
        })
        event('RUNNING', arm=arm, phase='selected_best_TEST50', results=results)
        test = lock['inputs']['test']
        metrics = run_two_phase_evaluation(
            export_path=target / 'best_export', export_sha256=best['export_manifest_sha256'],
            manifest_path=test['path'], manifest_sha256=test['sha256'],
            train_manifest_path=lock['inputs']['train']['path'],
            train_manifest_sha256=lock['inputs']['train']['sha256'],
            out_dir=target / 'test50', device='cuda', batch_size=8, normalization='prepared',
        )
        assert metrics['coverage']['complete'] is True
        _write_supplemental_evaluation_report(target, lock, best, test_metrics=metrics)
        results[arm] = {
            'status': 'COMPLETED', 'best_step': best['step'], 'best_pt_sha256': best['best_pt_sha256'],
            'val_macro4': best['val_macro4'], 'val_foreground': best['foreground_macro'],
            'test_macro4': metrics['class_first_macro4'],
            'test_foreground': metrics['class_first_foreground_macro'],
            'test_metrics_sha256': sha256_file(target / 'test50/METRICS.json'),
            'test_frozen_sha256': sha256_file(target / 'test50/PREDICTIONS_FROZEN.json'),
            'output': str(target), 'drive_backup': 'UPLOAD_PENDING',
        }
        atomic_json(target / 'RESULT.json', results[arm])
        print(json.dumps(results[arm], sort_keys=True), flush=True)
    for path, expected in plan['old_receipt_pins'].items():
        assert sha256_file(path) == expected, 'An original receipt changed during isolated evaluation'
    event('COMPLETED', results=results, old_receipts_unchanged=True, git_publication='PENDING')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'execution.lock').open('w') as gate:
        fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.preflight:
                assert not (OUT / 'PLAN.json').exists()
                preflight()
            else:
                assert args.plan_sha256
                run(args.plan_sha256)
        except Exception as exc:
            traceback.print_exc()
            event('BLOCKED', error_type=type(exc).__name__, error=str(exc))
            raise
