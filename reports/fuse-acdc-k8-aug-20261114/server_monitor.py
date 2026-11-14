"""Read-only A-aug alarm precheck: 1=healthy/progressing, 0=needs coordinator."""
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path('/root/Self-Audit-fuse-aug-20261114')
RUN = Path('/root/mount_data/runs/fuse-acdc-k8-aug-20261114')
sys.path.insert(0, str(ROOT))


def main():
    from scripts.fuse_watchdog import inspect_run, atomic_write
    status = inspect_run(RUN, stale_seconds=1800)
    try:
        flow = json.loads((RUN / 'AUG_FLOW.json').read_text())
    except (OSError, ValueError):
        flow = {'status': 'MISSING'}
    status['flow'] = flow
    status['observed_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    status['publish_ready'] = (RUN / 'PUBLISH_READY.json').is_file()
    atomic_write(RUN / 'MONITOR_STATUS.json', status)
    print(json.dumps(status, sort_keys=True))
    # No lifecycle mutation or automatic retry. Preserve all prior run states.
    terminal = flow.get('status') in {'FAILED', 'COMPLETED', 'MISSING'}
    return 0 if terminal or status['alerts'] or status['publish_ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
