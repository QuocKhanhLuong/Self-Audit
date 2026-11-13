"""One-off, user-authorized cleanup of the verified stopped Q3D archive.

Default is a read-only plan. This script never removes datasets or FUSE runs.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

SOURCE = Path('/root/mount_data/runs/q3d-resenc-pixdim-20261112')
RECEIPTS = Path('/root/mount_data/receipts/q3d-drive-archive-20261113-parts')
DEST = 'KhanhDrive:Q3D-archive-20261113'
FLAGS = ['--drive-root-folder-id', '1KLYuulOT6riQjeF1oarmeBDmwcsD3aEX',
         '--tpslimit', '1', '--tpslimit-burst', '1', '--retries', '3']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    target = RECEIPTS / name
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, target)


def inventory():
    rows = []
    for path in sorted(SOURCE.rglob('*')):
        if path.is_symlink():
            raise RuntimeError('Unexpected source symlink')
        if path.is_file():
            stat = path.stat()
            rows.append({'path': str(path.relative_to(SOURCE)), 'bytes': stat.st_size,
                         'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino})
    return rows


def stopped():
    for service in ('q3d-campaign', 'q3d-watchdog', 'fuse-q3d-archive-parts'):
        state = subprocess.run(['supervisorctl', 'status', service],
                               text=True, capture_output=True).stdout.split()
        if len(state) < 2 or state[1] not in ('STOPPED', 'EXITED'):
            raise RuntimeError(f'Service is not stopped: {service} {state}')
    prefix = str(SOURCE) + '/'
    for folder in Path('/proc').glob('[0-9]*/fd'):
        try:
            descriptors = list(folder.iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                if os.readlink(descriptor).startswith(prefix):
                    raise RuntimeError(f'Open source file: {descriptor}')
            except (FileNotFoundError, PermissionError):
                pass


def plan():
    if SOURCE.is_symlink() or not SOURCE.is_dir() or SOURCE.resolve() != SOURCE:
        raise RuntimeError('Exact historical source directory is required')
    stopped()
    record = json.loads((RECEIPTS / 'VERIFIED.json').read_text())
    status = json.loads((RECEIPTS / 'STATUS.json').read_text())
    before = json.loads((RECEIPTS / 'INVENTORY.json').read_text())
    if record.get('status') != 'VERIFIED' or status.get('status') != 'VERIFIED':
        raise RuntimeError('Archive has not completed verification')
    if (record.get('source') != str(SOURCE) or before.get('source') != str(SOURCE)
            or record.get('destination') != DEST + '/parts'
            or record.get('source_inventory_unchanged') is not True):
        raise RuntimeError('Archive identity mismatch')
    parts = record['parts']
    if not parts or sum(p['bytes'] for p in parts) != record['archive_bytes']:
        raise RuntimeError('Incomplete archive part inventory')
    for index, part in enumerate(parts):
        name = f'q3d-resenc-pixdim-20261112.tar.part{index:03d}'
        if (part['name'] != name or part['remote'] != DEST + '/parts/' + name
                or part['bytes'] <= 0 or len(part['sha256']) != 64
                or part['sha256'] != part['remote_sha256']):
            raise RuntimeError('Archive part SHA verification mismatch')
    if inventory() != before['files'] or len(before['files']) != record['file_count']:
        raise RuntimeError('Historical source inventory changed')
    remote_record = subprocess.run(
        ['rclone', 'cat', DEST + '/receipts-parts/VERIFIED.json', *FLAGS],
        check=True, capture_output=True).stdout
    if hashlib.sha256(remote_record).hexdigest() != sha(RECEIPTS / 'VERIFIED.json'):
        raise RuntimeError('Drive verification receipt differs from local receipt')
    listing = json.loads(subprocess.run(
        ['rclone', 'lsjson', DEST + '/parts', *FLAGS],
        check=True, capture_output=True, text=True).stdout)
    files = [row for row in listing if not row['IsDir']]
    if len(files) != len(parts) or len({row['Name'] for row in files}) != len(files):
        raise RuntimeError('Drive part names are duplicated or incomplete')
    sizes = {row['Name']: row['Size'] for row in files}
    if sizes != {part['name']: part['bytes'] for part in parts}:
        raise RuntimeError('Drive parts inventory differs from verified inventory')
    return {'status': 'VERIFIED_CLEANUP_PLAN', 'source': str(SOURCE),
            'archive_sha256': record['archive_sha256'],
            'archive_bytes': record['archive_bytes'], 'parts': len(parts),
            'file_count': record['file_count'],
            'source_bytes': sum(row['bytes'] for row in before['files']),
            'verification_receipt_sha256': sha(RECEIPTS / 'VERIFIED.json'),
            'inventory_sha256': sha(RECEIPTS / 'INVENTORY.json'),
            'drive_folder': 'https://drive.google.com/drive/folders/1-EPd8W7iBvnn2RFxGm-LvG4CuSFL26Gk',
            'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    receipt = plan()
    if args.execute:
        write('PREDELETE.json', receipt)
        stopped()
        if inventory() != json.loads((RECEIPTS / 'INVENTORY.json').read_text())['files']:
            raise RuntimeError('Source changed before deletion')
        free_before = shutil.disk_usage(SOURCE.parent).free
        try:
            shutil.rmtree(SOURCE)
        except Exception as error:
            write('DELETE_FAILED.json', {**receipt, 'status': 'DELETE_FAILED',
                  'error': str(error), 'remaining_inventory': inventory(),
                  'failed_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
            raise
        if SOURCE.exists():
            raise RuntimeError('Source still exists after cleanup')
        receipt.update(status='DELETED_VERIFIED_SERVER_COPY',
                       deleted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                       free_bytes_before=free_before,
                       free_bytes_after=shutil.disk_usage(SOURCE.parent).free)
        write('DELETED.json', receipt)
        subprocess.run(['rclone', 'copyto', str(RECEIPTS / 'DELETED.json'),
                        DEST + '/receipts-parts/DELETED.json', *FLAGS], check=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
