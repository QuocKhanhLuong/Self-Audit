#!/usr/bin/env python3
"""Run the authorized A-aug campaign once, then materialize its result report."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--old-run', type=Path, required=True)
    parser.add_argument('--phase0', type=Path, required=True)
    args = parser.parse_args(argv)
    from self_audit.candidate_fuse.config import load_config
    cfg = load_config(args.config, real_run=True)
    if cfg.get('augmentation') != {'policy': 'inplane_strong_v1'}:
        raise ValueError('A-aug requires its locked augmentation policy')
    old_cfg = json.loads((args.old_run / 'recipe.json').read_text())
    raw_cfg = json.loads(args.config.read_text())
    unchanged = dict(raw_cfg)
    unchanged.pop('augmentation')
    if unchanged != old_cfg:
        raise ValueError('A-aug may differ from old A only by augmentation')
    old_ledger = json.loads((args.old_run / 'TEST_ONCE/TEST_TRANSACTION.json').read_text())
    if old_ledger.get('status') != 'COMMITTED':
        raise ValueError('old A must have its committed TEST receipt')
    if args.out.resolve() == args.old_run.resolve():
        raise ValueError('new run must not overwrite old A')
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = args.out / 'AUG_FLOW.json'
    if receipt.exists() or (args.out / 'RUN_STATE.json').exists():
        raise ValueError('A-aug was already attempted; zero automatic retries')
    if not args.phase0.is_file():
        raise ValueError('Phase0 report file must exist (scores do not gate launch)')
    state = {'schema': 'self_audit.fuse.a_aug.flow.v1', 'status': 'RUNNING',
             'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'test_opening': 2, 'test_label': 'TEST lần 2, sau khi TEST đã mở',
             'automatic_retry_limit': 0, 'old_run': str(args.old_run.resolve()),
             'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest()}
    atomic_json(receipt, state)
    start = time.monotonic()
    try:
        subprocess.run([sys.executable, 'scripts/fuse_campaign.py', '--config',
                        str(args.config.resolve()), '--out', str(args.out.resolve()),
                        '--device', 'cuda', '--execute'], cwd=ROOT, check=True)
        campaign = json.loads((args.out / 'RUN_STATE.json').read_text())
        if campaign.get('status') != 'COMPLETED' or campaign.get('test_ledger', {}).get('status') != 'COMMITTED':
            raise RuntimeError('campaign did not commit its complete TEST transaction')
        state.update(status='REPORTING', campaign_seconds=time.monotonic() - start)
        atomic_json(receipt, state)
        subprocess.run([sys.executable, 'scripts/fuse_aug_report.py', '--old-run',
                        str(args.old_run.resolve()), '--new-run', str(args.out.resolve()),
                        '--phase0', str(args.phase0.resolve()), '--out',
                        str((args.out / 'report').resolve())], cwd=ROOT, check=True)
        state.update(status='COMPLETED', completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     total_seconds=time.monotonic() - start, report=str((args.out / 'report').resolve()))
        atomic_json(receipt, state)
        atomic_json(args.out / 'PUBLISH_READY.json', state)
        print('A-aug COMPLETE: report ready for publication; TEST opening 2', flush=True)
        return 0
    except Exception as exc:
        state.update(status='FAILED', error=f'{type(exc).__name__}: {exc}',
                     elapsed_seconds=time.monotonic() - start,
                     stopped_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(receipt, state)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
