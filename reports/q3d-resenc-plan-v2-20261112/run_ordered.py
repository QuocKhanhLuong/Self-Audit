"""One supervised, fail-closed launch of the user-authorized Q3D run order."""
import datetime
import hashlib
import json
import subprocess
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path('/root/Self-Audit')
BASE = Path('/root/mount_data/runs/q3d-resenc-pixdim-20261112')
PLAN = BASE / 'campaign-v2/PLAN.json'
RECEIPTS = Path('/root/mount_data/receipts/q3d-pixdim-20261112')
sys.path.insert(0, str(ROOT))
from scripts.q3d_campaign import execute_campaign, validate_campaign_plan

lock = threading.Lock()
stop = threading.Event()


def log(message):
    line = datetime.datetime.now(datetime.timezone.utc).isoformat() + ' ' + message
    with lock:
        print(line, flush=True)
        with (BASE / 'launch.log').open('a') as handle:
            handle.write(line + '\n')


def state(status, **fields):
    payload = {'status': status, 'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'code_commit': sys.argv[1], 'plan_sha256': sys.argv[2], **fields}
    target = BASE / 'QUEUE_STATUS.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(target)


def stream_metrics(plan):
    offsets = {}
    paths = {Path(job['output']) / 'metrics.jsonl': job['id'] for job in plan['stage_jobs']}
    while not stop.wait(2):
        for path, name in paths.items():
            if not path.exists():
                continue
            with path.open() as handle:
                handle.seek(offsets.get(path, 0))
                for line in handle:
                    if line.strip():
                        log('METRIC ' + name + ' ' + line.strip())
                offsets[path] = handle.tell()


def runner(command, **kwargs):
    log('TRAIN_COMMAND ' + json.dumps(command))
    result = subprocess.run(command, **kwargs)
    if result.returncode:
        log('TRAIN_STDERR ' + (result.stderr or ''))
        log('TRAIN_STDOUT ' + (result.stdout or ''))
    return result


def train(setting, k, through='J'):
    state('RUNNING', setting=setting, k=k, through_stage=through)
    log(f'CAMPAIGN_START setting={setting} k={k} through={through}')
    result = execute_campaign(PLAN, execute=True, setting=setting, k=(k,),
                              through_stage=through, device='cuda', runner=runner)
    log('CAMPAIGN_RESULT ' + json.dumps(result))
    expected = 'PARTIAL' if through == 'P' else 'COMPLETED'
    if result['status'] != expected:
        raise RuntimeError('unexpected campaign completion status')


def main():
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if actual != sys.argv[1] or hashlib.sha256(PLAN.read_bytes()).hexdigest() != sys.argv[2]:
        raise RuntimeError('launch source or plan SHA changed')
    plan = validate_campaign_plan(PLAN)
    if any(job['readiness'] != 'READY' for job in plan['stage_jobs']):
        raise RuntimeError('plan contains an unready stage')
    t_profile = json.loads((RECEIPTS / 'GPU_PROFILE_T.json').read_text())
    experiment = next(e for e in plan['experiments'] if e['setting'] == 'v1' and e['k'] == 8)
    if (t_profile['status'] != 'COMPLETED'
            or t_profile['stage_results']['T']['measured_updates'] != 20
            or t_profile['config_sha256'] != experiment['config_sha256']):
        raise RuntimeError('T profile did not pass for the production recipe bytes')
    threading.Thread(target=stream_metrics, args=(plan,), daemon=True).start()
    train('v1', 8, 'P')
    runtime = json.loads((PLAN.parent / 'RUN_STATE.json').read_text())
    artifacts = runtime['experiments'][experiment['id']]['stage_artifacts']
    args = json.loads((BASE / 'profile_args.json').read_text())
    for flag, value in [('--stage', 'J'), ('--config', experiment['config_path']),
                        ('--report', str(RECEIPTS / 'GPU_PROFILE_J.json'))]:
        args[args.index(flag) + 1] = value
    for stage in ('T', 'Q', 'P'):
        args += ['--stage-artifact', stage + '=' + artifacts[stage]['path'],
                 '--stage-artifact-sha', stage + '=' + artifacts[stage]['sha256']]
    state('J_PROFILE_RUNNING', setting='v1', k=8)
    log('J_PROFILE_START: 20 updates with verified production T/Q/P artifacts')
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    log('J_PROFILE_STDOUT ' + result.stdout)
    if result.returncode:
        log('J_PROFILE_STDERR ' + result.stderr)
        raise RuntimeError('J profile failed; campaign queue stopped')
    profile = json.loads((RECEIPTS / 'GPU_PROFILE_J.json').read_text())
    if profile['status'] != 'COMPLETED' or profile['stage_results']['J']['measured_updates'] != 20:
        raise RuntimeError('J profile did not complete exactly 20 updates')
    for setting, k in [('v1', 8), ('v1', 5), ('v1', 10), ('v1u', 8), ('c1', 8), ('c0', 8)]:
        train(setting, k)
    state('COMPLETED')
    log('QUEUE_COMPLETED')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        state('FAILED', error=str(exc))
        log(traceback.format_exc())
        raise
    finally:
        stop.set()
