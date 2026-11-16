#!/usr/bin/env python3
"""Download a pinned Drive dataset folder and verify every local byte by MD5."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

FOLDERS = {'ACDC': '1EelzBVjIoDQ4uzt0_2JzmF_PuUHsD93e',
           'MNMS': '1DpW8ucYE17Tj8iMlsAev_yM6TaZZmWYj'}


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def safe_relative(value: str) -> Path:
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in value:
        raise ValueError(f'unsafe source path: {value!r}')
    return Path(*p.parts)


def validate_inventory(rows: list[dict]) -> None:
    if not rows:
        raise ValueError('empty source inventory')
    seen = set()
    for row in rows:
        name = str(safe_relative(row['Path']))
        if name in seen or row.get('IsDir') or row['Size'] <= 0:
            raise ValueError(f'duplicate or invalid source file: {name}')
        seen.add(name)
        digest = row.get('Hashes', {}).get('md5', '')
        if len(digest) != 32 or any(x not in '0123456789abcdefABCDEF' for x in digest):
            raise ValueError(f'missing source MD5: {name}')


def verify_file(path: Path, source: dict) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != source['Size']:
        raise ValueError(f'file missing, symlink, or size mismatch: {path}')
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest().lower() != source['Hashes']['md5'].lower():
        raise ValueError(f'Drive/local MD5 mismatch: {path}')
    if source.get('Hashes', {}).get('sha256') and sha.hexdigest() != source['Hashes']['sha256'].lower():
        raise ValueError(f'Drive/local SHA256 mismatch: {path}')
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                safe_relative(info.filename)
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f'ZIP CRC failed: {path}: {bad}')
    return {'path': str(path), 'relative_path': source['Path'], 'drive_id': source.get('ID'),
            'bytes': path.stat().st_size, 'md5': md5.hexdigest(), 'sha256': sha.hexdigest()}


def range_download(url: str, part: Path, size: int) -> None:
    """Bound each HTTP transfer so a trickling connection cannot stall a run."""
    chunk = part.with_name(part.name + '.chunk')
    with part.open('wb') as output:
        for start in range(0, size, 8 * 1024**2):
            end = min(size, start + 8 * 1024**2) - 1
            response = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                '--connect-timeout', '15', '--max-time', '40', '--speed-time', '15',
                '--speed-limit', '131072', '--retry', '2', '--range', f'{start}-{end}',
                '--output', str(chunk), '--write-out', '%{http_code}',
                url + f'&range_request={start}-{end}'], capture_output=True, text=True, check=True)
            if response.stdout != '206' or chunk.stat().st_size != end - start + 1:
                raise ValueError('Drive range response did not match requested byte interval')
            with chunk.open('rb') as data:
                shutil.copyfileobj(data, output)
            output.flush()
    chunk.unlink()


def direct_file(target: Path, source: dict, *, attempts: int = 3, ranged: bool = False) -> None:
    """Download a public Drive file by its inventoried ID, then atomically seal it."""
    path = target / safe_relative(source['Path'])
    if not path.resolve().is_relative_to(target.resolve()):
        raise ValueError('destination escaped dataset root')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink():
            raise ValueError('destination is a symlink')
        try:
            verify_file(path, source)
            return
        except ValueError:
            # rclone can leave a preallocated final filename on interruption.
            # Preserve those bytes, then fetch a verified replacement.
            quarantine = path.with_name(path.name + f'.interrupted-{time.time_ns()}')
            os.replace(path, quarantine)
            print(f'Preserved incomplete transfer: {quarantine}', flush=True)
    file_id = source.get('ID', '')
    if not file_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in file_id):
        raise ValueError('missing or unsafe inventoried Drive ID')
    part = path.with_name(path.name + ('.pg-range' if ranged else '.pg-part'))
    url = 'https://drive.usercontent.google.com/download?id=' + file_id + '&export=download&confirm=t'
    for attempt in range(attempts):
        try:
            # Write only a private partial file; an HTML quota page can never be
            # promoted to a dataset file because source size and hashes must match.
            if ranged:
                range_download(url, part, source['Size'])
            else:
                with urllib.request.urlopen(url, timeout=90) as response, part.open('wb') as out:
                    if 'text/html' in response.headers.get('Content-Type', '').lower():
                        raise ValueError('Drive returned HTML rather than dataset bytes')
                    shutil.copyfileobj(response, out, length=1024 * 1024)
            verify_file(part, source)
            os.replace(part, path)
            return
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def direct_copy(target: Path, rows: list[dict], transfers: int, *, ranged: bool = False) -> None:
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=transfers) as pool:
        pending = {pool.submit(direct_file, target, row, ranged=ranged): row for row in rows}
        total_bytes = 0
        errors = []
        for count, future in enumerate(as_completed(pending), 1):
            try:
                future.result()
                total_bytes += pending[future]['Size']
            except Exception as exc:
                errors.append(f"{pending[future]['Path']}: {type(exc).__name__}: {exc}")
                print('Download failure: ' + errors[-1], flush=True)
            if count % 10 == 0 or count == len(rows):
                print(f'Direct Drive: {count}/{len(rows)} verified files, '
                      f'{total_bytes / 1024**3:.3f} GiB, {time.monotonic() - start:.1f}s', flush=True)
        if errors:
            raise RuntimeError(f'{len(errors)} downloads failed: ' + '; '.join(errors[:5]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(FOLDERS), required=True)
    parser.add_argument('--root', type=Path, default=Path('/root/dataset'))
    parser.add_argument('--remote', default='KhanhDrive:')
    parser.add_argument('--transfers', type=int, default=4)
    parser.add_argument('--transport', choices=('rclone', 'direct', 'ranged'), default='rclone')
    args = parser.parse_args(argv)
    if args.transfers < 1:
        parser.error('transfers must be positive')
    receipts = args.root / 'receipts'
    receipts.mkdir(parents=True, exist_ok=True)
    lock = (receipts / f'{args.dataset}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    target = args.root / args.dataset / ('raw' if args.dataset == 'ACDC' else 'archives')
    target.mkdir(parents=True, exist_ok=True)
    receipt = receipts / f'{args.dataset}_DOWNLOAD.json'
    inventory_path = receipts / f'{args.dataset}_SOURCE.json'
    state = {'schema': 'self_audit.dataset_download.v1', 'dataset': args.dataset,
             'source_folder_id': FOLDERS[args.dataset], 'target': str(target),
             'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'INVENTORY'}
    atomic_json(receipt, state)
    start = time.monotonic()
    try:
        if inventory_path.exists():
            rows = json.loads(inventory_path.read_text())
        else:
            raw = subprocess.check_output(['rclone', 'lsjson', args.remote,
                '--drive-root-folder-id', FOLDERS[args.dataset], '--recursive', '--files-only', '--hash'], text=True)
            rows = sorted(json.loads(raw), key=lambda x: x['Path'])
            validate_inventory(rows)
            atomic_json(inventory_path, rows)
        validate_inventory(rows)
        total = sum(x['Size'] for x in rows)
        missing = sum(x['Size'] for x in rows if not (target / safe_relative(x['Path'])).exists())
        if shutil.disk_usage(target).free < missing + 10 * 1024**3:
            raise RuntimeError('insufficient storage for dataset and 10 GiB reserve')
        state.update(status='DOWNLOADING', files=len(rows), bytes=total,
                     source_inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest())
        atomic_json(receipt, state)
        print(f'{args.dataset}: download {len(rows)} files, {total / 1024**3:.2f} GiB into {target}', flush=True)
        if args.transport in {'direct', 'ranged'}:
            direct_copy(target, rows, args.transfers, ranged=args.transport == 'ranged')
        else:
            subprocess.run(['rclone', 'copy', args.remote, str(target), '--drive-root-folder-id',
                FOLDERS[args.dataset], '--checksum', '--immutable', '--transfers', str(args.transfers),
                '--checkers', '8', '--retries', '3', '--low-level-retries', '10',
                '--stats', '15s', '--stats-one-line', '--log-level', 'INFO'], check=True)
        state['status'] = 'VERIFYING'
        atomic_json(receipt, state)
        checked = []
        for i, source in enumerate(rows, 1):
            path = target / safe_relative(source['Path'])
            if not path.resolve().is_relative_to(target.resolve()):
                raise ValueError('destination escaped dataset root')
            checked.append(verify_file(path, source))
            if i % 25 == 0 or i == len(rows):
                print(f'{args.dataset}: verified {i}/{len(rows)} files (Drive MD5 + SHA256; ZIP CRC when applicable)', flush=True)
        atomic_json(receipts / f'{args.dataset}_FILES.json', checked)
        state.update(status='DOWNLOADED_VERIFIED', elapsed_seconds=time.monotonic() - start,
                     completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     files_receipt=str(receipts / f'{args.dataset}_FILES.json'))
        atomic_json(receipt, state)
        print(f'{args.dataset}: DOWNLOADED_VERIFIED; geometry/preparation is a separate step', flush=True)
        return 0
    except Exception as exc:
        state.update(status='FAILED', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.monotonic() - start)
        atomic_json(receipt, state)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
