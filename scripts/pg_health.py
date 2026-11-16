#!/usr/bin/env python3
"""Bounded PG health precheck; SSH failures wake the repair agent too."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

DEFAULT_RUN = '/root/mount_data/runs/pg-k8-20261116'


def inspect_run(run: Path, *, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    state_path = run / 'QUEUE_STATE.json'
    if not state_path.is_file():
        return {'status': 'MISSING_QUEUE', 'needs_attention': True}
    state = json.loads(state_path.read_text())
    status = state.get('status')
    stage = state.get('active_stage')
    if status != 'RUNNING':
        return {'status': status, 'stage': stage, 'needs_attention': True}
    record = state.get('stages', {}).get(stage, {})
    log = Path(record.get('log_path', run / 'logs' / f'{stage}.log'))
    # S1 scans TRAIN for cosine at epoch boundaries; allow that bounded work
    # to produce sparse console lines, but never infer progress from GPU use.
    evidence = [p for p in (log, run / 'S1/diag.jsonl', run / 'metrics.jsonl') if p.is_file()]
    last_write = max((p.stat().st_mtime for p in evidence), default=state_path.stat().st_mtime)
    age = max(0, now - last_write)
    return {'status': 'STALE' if age > 1200 else 'RUNNING', 'stage': stage,
            'seconds_since_progress_file': age, 'needs_attention': age > 1200}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path(DEFAULT_RUN))
    p.add_argument('--server', action='store_true')
    p.add_argument('--ssh-host', default='vast-gpu')
    p.add_argument('--remote-script', default='/root/Self-Audit-pg-20261116/scripts/pg_health.py')
    args = p.parse_args(argv)
    if args.server:
        try:
            result = inspect_run(args.run)
            process = subprocess.run(['supervisorctl', 'status', 'pg-campaign'],
                                     capture_output=True, text=True, timeout=10)
            result['process'] = process.stdout.strip()
            if result.get('status') == 'RUNNING' and (process.returncode or ' RUNNING ' not in process.stdout):
                result.update(status='PROCESS_STOPPED', needs_attention=True)
        except Exception as exc:
            result = {'status': 'MONITOR_ERROR', 'needs_attention': True,
                      'error': f'{type(exc).__name__}: {exc}'}
        print(json.dumps(result, sort_keys=True))
        return 0 if result['needs_attention'] else 3
    remote = shlex.join(['/venv/main/bin/python', args.remote_script, '--server', '--run', str(args.run)])
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                 args.ssh_host, remote], capture_output=True, text=True, timeout=45)
        # Exit 3 is the one known healthy signal. Every transport/parse/process
        # error must wake the agent, rather than be silently skipped by Orca.
        lines = result.stdout.strip().splitlines()
        payload = json.loads(lines[-1]) if lines else {}
        if result.returncode == 3 and payload.get('status') == 'RUNNING' and payload.get('needs_attention') is False:
            print(json.dumps(payload))
            return 3
        print(json.dumps({'returncode': result.returncode, 'health': payload,
                          'stderr': result.stderr[-1500:]}))
    except Exception as exc:
        print(json.dumps({'status': 'SSH_OR_MONITOR_ERROR', 'error': str(exc)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
