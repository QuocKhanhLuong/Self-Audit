"""Run the pinned Astra handoff after explicit backbone approval and GPU admission."""
from pathlib import Path
import fcntl
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback

JOB = Path(__file__).resolve().parent
PLAN = json.loads((JOB / 'PLAN.json').read_text())
SOURCE = Path(PLAN['source'])
EXPERIMENT = SOURCE / 'experiments/layer_guided_student_v1'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=PLAN['gpu_uuid'], PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8',
           MPLCONFIGDIR=str(JOB / 'mplconfig'),
           PYTHONPATH=os.pathsep.join([str(EXPERIMENT), str(SOURCE / 'src'), str(JOB / 'python_deps')]))
STATE = dict(status='STARTING', stage='admission', worker_pid=os.getpid(),
             git_commit=PLAN['git_commit'], training_started=False,
             official_backbone_loaded=False, pilot_completed=False, reload_passed=False,
             full_train_completed=False, quality='NOT_EVALUATED')
CHILD = None


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def status(**values):
    STATE.update(values, updated_at=time.time())
    write(JOB / 'STATUS.json', STATE)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_check():
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == PLAN['git_commit']
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    assert digest(Path(PLAN['bank']) / 'FROZEN.json') == PLAN['bank_manifest_sha256']


def wait_gpu(next_stage):
    while True:
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True)
        occupants = [r.strip() for r in rows.splitlines() if r.startswith(PLAN['gpu_uuid'])]
        free = int(subprocess.check_output(['nvidia-smi', '-i', PLAN['gpu_uuid'], '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if not occupants and free >= 10000:
            status(status='RUNNING', stage=next_stage, gpu_occupants=[], gpu_free_mib=free)
            return
        status(status='QUEUED', stage='waiting_for_4070', next_stage=next_stage,
               gpu_occupants=occupants, gpu_free_mib=free)
        time.sleep(20)


def stop(signum, _frame):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signum)
    raise KeyboardInterrupt


def run_stage(name, command, tracker=None):
    global CHILD
    source_check()
    status(status='RUNNING', stage=name)
    write(JOB / 'receipts' / f'{name}-command.json', command)
    print('[STAGE]', name, flush=True)
    started = time.time()
    with (JOB / 'receipts' / f'{name}-console.log').open('w', buffering=1) as log:
        CHILD = subprocess.Popen(command, cwd=SOURCE, env=ENV, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        status(child_pid=CHILD.pid)
        for line in CHILD.stdout:
            log.write(line)
            if line.startswith('[ASTRA_STARTUP] '):
                startup = json.loads(line.split('] ', 1)[1])
                assert startup['class'] == 'LayerGuidedStudent'
                assert startup['architecture'] == 'layer_guided_convnext_tiny_v1'
                assert startup['backbone'] == 'dinov3_convnext_tiny' and startup['pretrained_loaded']
                assert startup['backbone_weights_sha256'] == APPROVAL['weights_sha256']
                assert startup['device'] == 'cuda' and startup['device_name'] == PLAN['gpu_name']
                status(training_started=True, parameters=startup['parameters'])
                write(JOB / 'receipts' / f'{name}-startup-observed.json', startup)
                print(line, end='', flush=True)
            elif line.startswith('[ASTRA_METRIC] '):
                row = json.loads(line.split('] ', 1)[1])
                for value in row.values():
                    if isinstance(value, float) and not math.isfinite(value):
                        raise ValueError('nonfinite training metric')
                if row['step'] == 1 or row['step'] % 25 == 0 or name == 'pilot32':
                    status(last_metric=row)
                    print(line, end='', flush=True)
                    if tracker:
                        values = {f'train/{key}': value for key, value in row.items() if value is not None}
                        values['train/global_step'] = row['step']
                        try:
                            tracker.log(values)
                        except Exception as exc:
                            status(telemetry_errors=STATE.get('telemetry_errors', 0) + 1,
                                   last_telemetry_error=type(exc).__name__)
            elif not line.startswith('[ASTRA_COMPLETE] '):
                print(line, end='', flush=True)
        code = CHILD.wait()
    CHILD = None
    status(child_pid=None)
    write(JOB / 'receipts' / f'{name}-exit.json', dict(exit_code=code, elapsed_seconds=time.time()-started))
    if code:
        raise RuntimeError(f'{name} failed with exit code {code}; full run prohibited')


def tracked_training(name, recipe, output):
    import wandb
    tracking = JOB / 'tracking' / name
    tracking.mkdir(parents=True, exist_ok=False)
    config = json.loads(recipe.read_text())
    run = wandb.init(project='self-audit-astra-student',
                     entity='nthne-hanoi-university-of-science-and-technology',
                     name=JOB.name + '-' + name, group=JOB.name, job_type=name,
                     mode='online', dir=str(tracking),
                     config={**config, 'git_commit': PLAN['git_commit'],
                             'handoff_commit': PLAN['handoff_commit'],
                             'bank_manifest_sha256': PLAN['bank_manifest_sha256'],
                             'backbone_weights_sha256': APPROVAL['weights_sha256'],
                             'approval_sha256': APPROVAL_SHA,
                             'quality_evaluation': 'NOT_RUN', 'native_detail': 'NOT_TRAINED_NATIVE_DETAIL'},
                     tags=['astra', 'dinov3-convnext-tiny', '4070', 'frozen-historical-q'])
    run.define_metric('train/global_step')
    run.define_metric('train/*', step_metric='train/global_step')
    write(tracking / 'WANDB_RUN.json', dict(id=run.id, url=run.url, status='running'))
    status(wandb_url=run.url)
    command = [PLAN['python'], '-u', str(SOURCE / 'scripts/train_astra_student.py'),
               '--bank', PLAN['bank'], '--bank-manifest-sha256', PLAN['bank_manifest_sha256'],
               '--approval', str(APPROVAL_PATH), '--approval-sha256', APPROVAL_SHA,
               '--recipe', str(recipe), '--out', str(output), '--device', 'cuda']
    try:
        run_stage(name, command, run)
        report = json.loads((output / 'REPORT.json').read_text())
        assert report['receipt_status'] == 'SUCCESS'
        assert report['semantic_updates'] > 0
        run.summary.update({f'final/{k}': report[k] for k in ['status', 'optimizer_steps', 'semantic_updates', 'epochs_completed', 'elapsed_seconds', 'peak_allocated_bytes']})
        run.summary['quality_status'] = 'NOT_EVALUATED'
        write(tracking / 'WANDB_RUN.json', dict(id=run.id, url=run.url, status='finished'))
        run.finish(exit_code=0)
        return report
    except BaseException:
        write(tracking / 'WANDB_RUN.json', dict(id=run.id, url=run.url, status='failed'))
        run.finish(exit_code=1)
        raise


def main():
    global APPROVAL, APPROVAL_PATH, APPROVAL_SHA
    with (JOB / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, stop)
        status(status='PREPARING')
        source_check()
        parity = json.loads((JOB / 'receipts/software-final/PARITY_REPORT.json').read_text())
        assert parity['status'] == 'PASS_CPU_SOFTWARE_ONLY'
        assert parity['checkout_commit'] == PLAN['git_commit']
        assert parity['tests'] == dict(tests=125, failures=0, errors=0, skipped=0)
        pin = json.loads((JOB / 'receipts/APPROVAL_PIN.json').read_text())
        APPROVAL_PATH = Path(pin['path'])
        APPROVAL_SHA = pin['sha256']
        assert digest(APPROVAL_PATH) == APPROVAL_SHA
        APPROVAL = json.loads(APPROVAL_PATH.read_text())
        assert APPROVAL['approved'] is True and pin['user_review_record']
        wait_gpu('official_backbone_preflight')
        run_stage('official_backbone_preflight', [PLAN['python'], '-m', 'integration.cli',
                  '--recipe', str(EXPERIMENT / 'recipe.json'), '--approval', str(APPROVAL_PATH),
                  '--expected-approval-sha256', APPROVAL_SHA, '--out', PLAN['preflight_out'], '--device', 'cuda'])
        preflight = json.loads((Path(PLAN['preflight_out']) / 'PREFLIGHT.json').read_text())
        assert preflight['device'] == 'cuda' and preflight['optimizer_steps'] == 0
        assert preflight['backbone']['verified_weights_sha256'] == APPROVAL['weights_sha256']
        status(official_backbone_loaded=True)
        wait_gpu('pilot32')
        report = tracked_training('pilot32', EXPERIMENT / 'training_v1/pilot.json', Path(PLAN['pilot_out']))
        rows = [json.loads(line) for line in (Path(PLAN['pilot_out']) / 'metrics.jsonl').read_text().splitlines()]
        assert report['status'] == 'COMPLETED_MAX_STEPS' and report['optimizer_steps'] == len(rows) == 32
        assert report['metrics_rows'] == 32 and report['semantic_updates'] == sum(r['semantic_update'] for r in rows)
        write(JOB / 'receipts/PILOT_GATE.json', dict(status='PASS', optimizer_steps=32,
              semantic_updates=report['semantic_updates'], final_manifest_sha256=report['final_manifest_sha256']))
        status(pilot_completed=True)
        wait_gpu('reload')
        run_stage('reload', [PLAN['python'], str(JOB / 'reload_checkpoint.py'), PLAN['pilot_out'],
                            str(APPROVAL_PATH), APPROVAL_SHA, str(JOB / 'receipts/PILOT_GATE.json')])
        status(reload_passed=True)
        bank = json.loads((JOB / 'receipts/BANK_VERIFIED.json').read_text())
        delta = [b['elapsed_seconds']-a['elapsed_seconds'] for a,b in zip(rows[1:],rows[2:])]
        payload_bytes = sum(p.stat().st_size for p in (Path(PLAN['pilot_out']) / 'final').iterdir() if p.is_file())
        required = int(payload_bytes * 51 * 1.25 + 4 * 2**30)
        free = os.statvfs(JOB).f_bavail * os.statvfs(JOB).f_frsize
        budget = dict(epochs=50, batch_size=1, contexts_per_epoch=bank['contexts_per_epoch'],
                      optimizer_updates=bank['planned_full_updates'],
                      pilot_median_step_seconds=statistics.median(delta),
                      naive_training_hours=statistics.median(delta)*bank['planned_full_updates']/3600,
                      estimate_scope='One-patient pilot only; excludes other shapes, epoch IO, hashing and checkpoint writes.',
                      estimated_disk_required_bytes=required, free_disk_bytes=free,
                      checkpoint_bytes_observed=payload_bytes, full_initialization='fresh_from_same_pretrained_weights')
        write(JOB / 'receipts/FULL_BUDGET.json', budget)
        assert free >= required, 'insufficient disk for retained full50 checkpoints; no run deleted'
        wait_gpu('full50')
        report = tracked_training('full50', EXPERIMENT / 'training_v1/full50.json', Path(PLAN['full_out']))
        assert report['status'] == 'COMPLETED_EPOCH_BUDGET' and report['epochs_completed'] == 50
        assert report['optimizer_steps'] == bank['planned_full_updates']
        status(status='FINISHED', stage='complete', full_train_completed=True, final_report=report['status'])
        (JOB / 'FINAL_EXIT_CODE').write_text('0\n')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if CHILD is not None and CHILD.poll() is None:
            os.killpg(CHILD.pid, signal.SIGTERM)
            try:
                CHILD.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(CHILD.pid, signal.SIGKILL)
                CHILD.wait()
        (JOB / 'failure.log').write_text(traceback.format_exc())
        status(status='FAILED', error=repr(exc), child_pid=None)
        (JOB / 'FINAL_EXIT_CODE').write_text('1\n')
        raise
