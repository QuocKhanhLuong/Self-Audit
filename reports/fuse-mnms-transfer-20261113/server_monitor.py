"""One-run operational helpers; never retry training or modify a live recipe."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
A = Path('/root/mount_data/runs/fuse-acdc-k8-20261113')
B = Path('/root/mount_data/runs/fuse-mnms-transfer-20261113')
ARCHIVE = Path('/root/mount_data/receipts/q3d-drive-archive-20261113-parts')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def service(name):
    proc = subprocess.run(['supervisorctl', 'status', name], capture_output=True,
                          text=True, timeout=20)
    words = proc.stdout.split()
    return words[1] if len(words) > 1 else 'UNKNOWN'


def archive_wait():
    deadline = time.monotonic() + 3 * 3600
    while time.monotonic() < deadline:
        if (ARCHIVE / 'DELETED.json').exists():
            return 0
        state = read(ARCHIVE / 'STATUS.json')
        if state.get('status') == 'FAILED':
            raise RuntimeError('Archive failed; server source remains untouched')
        if state.get('status') == 'VERIFIED' and service('fuse-q3d-archive-parts') == 'EXITED':
            return subprocess.run([sys.executable, str(Path(__file__).with_name('archive_cleanup.py')),
                                   '--execute'], check=False).returncode
        print(json.dumps({'cleanup': 'WAITING_FOR_VERIFIED_ARCHIVE',
                          'archive': state.get('status'), 'part': state.get('part')}, sort_keys=True), flush=True)
        time.sleep(30)
    raise RuntimeError('Archive cleanup wait expired; no source deletion')


def watchdog_wait():
    while not (B / 'campaign/RUN_STATE.json').exists():
        state = read(B / 'TRANSFER_QUEUE_STATE.json')
        if state.get('status') in {'FAILED', 'BLOCKED', 'COMPLETED'}:
            print(json.dumps(state, sort_keys=True), flush=True)
            return 3
        time.sleep(30)
    os.execv(sys.executable, [sys.executable, '-u', str(ROOT / 'scripts/fuse_watchdog.py'),
                             '--run-dir', str(B / 'campaign'), '--interval-seconds', '420'])


def live():
    while not (B / 'campaign/metrics.jsonl').exists():
        print(json.dumps({'B': read(B / 'TRANSFER_QUEUE_STATE.json'),
                          'A_progress': read(A / 'STATE.json'),
                          'archive': read(ARCHIVE / 'STATUS.json')}, sort_keys=True), flush=True)
        time.sleep(30)
    os.execv(sys.executable, [sys.executable, '-u', str(ROOT / 'scripts/fuse_watch_metrics.py'),
                             '--run-dir', str(B / 'campaign')])


def alarm():
    sys.path.insert(0, str(ROOT))
    from scripts.fuse_watchdog import inspect_run
    queue = read(B / 'TRANSFER_QUEUE_STATE.json')
    archive = read(ARCHIVE / 'STATUS.json')
    issues = []
    status = queue.get('status', 'MISSING')
    if status in {'FAILED', 'BLOCKED', 'MISSING', 'COMPLETED'}:
        issues.append('queue_' + status)
    elif service('fuse-mnms-transfer') != 'RUNNING':
        issues.append('queue_service_not_running')
    cleanup = service('fuse-q3d-cleanup')
    campaign = inspect_run(B / 'campaign') if (B / 'campaign/RUN_STATE.json').exists() else None
    if campaign:
        issues.extend(campaign['alerts'])
    print(json.dumps({'queue': queue, 'campaign': campaign, 'archive': archive.get('status'),
                      'cleanup': cleanup, 'issues': issues,
                      'observed_utc': dt.datetime.now(dt.timezone.utc).isoformat()}, sort_keys=True))
    return 0 if issues else 1


def archive_alarm():
    state = read(ARCHIVE / 'STATUS.json')
    deleted = (ARCHIVE / 'DELETED.json').exists()
    cleanup = service('fuse-q3d-cleanup')
    failed = state.get('status') == 'FAILED' or (ARCHIVE / 'DELETE_FAILED.json').exists()
    unhealthy = cleanup not in {'RUNNING', 'STARTING'} and not deleted
    print(json.dumps({'archive': state.get('status'), 'deleted': deleted,
                      'cleanup': cleanup, 'failed': failed, 'unhealthy': unhealthy}))
    return 0 if deleted or failed or unhealthy else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['archive-wait', 'watchdog-wait', 'live', 'alarm', 'archive-alarm'])
    mode = parser.parse_args().mode
    raise SystemExit({'archive-wait': archive_wait, 'watchdog-wait': watchdog_wait,
                      'live': live, 'alarm': alarm, 'archive-alarm': archive_alarm}[mode]())
