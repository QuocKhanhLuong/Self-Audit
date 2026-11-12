#!/usr/bin/env python3
"""Read-only live queue/GPU view, including stages without epoch metrics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        queue = read_json(args.base / 'QUEUE_STATUS.json')
        monitor = read_json(args.base / 'MONITOR_STATUS.json')
        timestamp = datetime.now(timezone.utc)
        if not args.once:
            print('\033[2J\033[H', end='')
        print(timestamp.isoformat(), flush=True)
        print(f"Q3D EXPLORATORY | {queue.get('status', 'UNKNOWN')} | "
              f"{queue.get('setting', '?')} K={queue.get('k', '?')} | "
              f"stage={queue.get('current_stage', '?')}")
        print('Stage status recorded:', queue.get('updated_utc', 'UNKNOWN'))
        print('Job:', queue.get('current_job', 'UNKNOWN'))
        print('Watchdog:', monitor.get('status', 'UNKNOWN'), '| checked:', monitor.get('updated_utc', 'UNKNOWN'))
        if queue.get('current_stage') == 'P':
            print('P: computing physiology cache across U cine frames; no optimizer epochs.')
        print('VAL RV/MYO/LV and training metrics appear in the other pane when emitted.')
        if queue.get('error'):
            print('Queue error:', queue['error'])
        try:
            gpu = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=4,
            )
            print('GPU:', gpu.stdout.strip() if gpu.returncode == 0 else 'UNKNOWN')
        except (OSError, subprocess.TimeoutExpired):
            print('GPU: UNKNOWN')
        print('Observed status only; GPU utilization is not a completed-update counter.', flush=True)
        if args.once:
            return
        time.sleep(5)


if __name__ == '__main__':
    main()
