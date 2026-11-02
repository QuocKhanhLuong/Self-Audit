"""Read actual GitHub best.pt blobs at an immutable commit; emit SHA256 receipt.

Requires an already authenticated gh CLI. Never prints credentials or downloads
patient data. Usage: python verify_github.py FULL_COMMIT_SHA
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = 'QuocKhanhLuong/Self-Audit'


def main():
    commit = sys.argv[1]
    assert re.fullmatch(r'[0-9a-f]{40}', commit)
    publication = json.loads((ROOT / 'PUBLICATION.json').read_text())

    def verify(arm):
        relative = f'artifacts/server_compact_final_20261102/{arm}/best.pt'
        endpoint = f'repos/{REPO}/contents/{relative}?ref={commit}'
        meta = json.loads(subprocess.check_output(['gh', 'api', endpoint]))
        blob_sha = subprocess.check_output(['git', 'rev-parse', f'{commit}:{relative}'], text=True).strip()
        assert meta['sha'] == blob_sha and meta['type'] == 'file'
        blob_endpoint = f'repos/{REPO}/git/blobs/{blob_sha}'
        payload = subprocess.check_output(['gh', 'api', '-H', 'Accept: application/vnd.github.raw+json', blob_endpoint])
        expected = publication['arms'][arm]
        assert len(payload) == meta['size'] == expected['best_pt_bytes']
        assert not payload.startswith(b'version https://git-lfs.github.com/spec')
        actual = hashlib.sha256(payload).hexdigest()
        assert actual == expected['best_pt_sha256']
        git_sha = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
        assert git_sha == blob_sha
        return arm, {'status': 'VERIFIED_FULL_PAYLOAD', 'path': relative, 'git_blob_sha1': blob_sha, 'bytes': len(payload), 'sha256': actual, 'contents_endpoint': endpoint, 'blob_endpoint': blob_endpoint, 'lfs_pointer': False}

    with ThreadPoolExecutor(max_workers=3) as pool:
        arms = dict(pool.map(verify, ('S0', 'S1', 'S2')))
    print(json.dumps({'status': 'PASS', 'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'commit': commit, 'repository': REPO, 'method': 'GitHub contents API resolves commit/path; raw Git blob API returns every byte; full SHA256 and Git SHA1 recomputed', 'drive_backup': 'UPLOAD_PENDING', 'drive_remote_verified': False, 'arms': arms}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
