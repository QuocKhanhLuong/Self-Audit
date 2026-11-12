"""Bounded SSH precheck for the overnight Q3D agent alarm.

Exit 0 requests an agent inspection; exit 1 skips a healthy scheduled check.
This command never starts/stops training or changes a remote file.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess


REMOTE_CHECK = r'''
import datetime,json,subprocess,sys
from pathlib import Path
base=Path(sys.argv[1])
queue=json.loads((base/'QUEUE_STATUS.json').read_text())
path=base/'MONITOR_STATUS.json'
monitor=json.loads(path.read_text()) if path.exists() else {}
result=subprocess.run(['supervisorctl','status','q3d-campaign'],capture_output=True,text=True,timeout=15)
running=' RUNNING ' in (' '+result.stdout.strip()+' ')
stamp=monitor.get('updated_utc') or monitor.get('timestamp_utc')
stale=True
if stamp:
    at=datetime.datetime.fromisoformat(stamp.replace('Z','+00:00'))
    stale=(datetime.datetime.now(datetime.timezone.utc)-at).total_seconds()>900
attention=(queue.get('status') in {'FAILED','COMPLETED'} or not running
           or bool(monitor.get('attention_required')) or stale)
print(json.dumps({'attention_required':attention,'queue_status':queue.get('status'),
                 'supervisor':result.stdout.strip(),'monitor_status':monitor.get('status'),
                 'monitor_stale':stale}))
sys.exit(0 if attention else 1)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='vast-gpu')
    parser.add_argument('--base', default='/root/mount_data/runs/q3d-resenc-pixdim-20261112')
    args = parser.parse_args()
    command = shlex.join(['/venv/main/bin/python', '-c', REMOTE_CHECK, args.base])
    try:
        result = subprocess.run(
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host, command],
            capture_output=True, text=True, timeout=40,
        )
        print(result.stdout.strip())
        try:
            observed = json.loads(result.stdout.strip().splitlines()[-1])
            attention = observed['attention_required']
            valid = type(attention) is bool and result.returncode == (0 if attention else 1)
        except (ValueError, IndexError, KeyError, TypeError):
            valid = False
        if not valid:
            print(json.dumps({'attention_required': True, 'error': 'SSH check failed',
                              'returncode': result.returncode}))
            return 0
        return result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({'attention_required': True, 'error': str(exc)}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
