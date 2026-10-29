import time
from datetime import datetime, timezone
from pathlib import Path

root = Path('/dev/shm/compact_teacher_b8')
err = Path('/workspace/compact-acdc-l8/20261028/b8_recovery/cache.stderr.log')
print('ACDC L8 | B8 fixed teacher cache | image-only U72 | Ctrl-C exits viewer only', flush=True)
while True:
    now = datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
    manifest = root / 'CACHE_MANIFEST.json'
    if manifest.exists():
        print(f'{now} CACHE COMPLETE | {manifest}', flush=True)
        break
    if err.exists() and err.stat().st_size:
        print(f'{now} CACHE ERROR | {err}', flush=True)
        break
    files = list(root.glob('patient*_probs.npy')) if root.exists() else []
    size_gb = sum(p.stat().st_size for p in files) / 1e9
    print(f'{now} U volumes {len(files)}/72 | allocated files {size_gb:.2f} GB', flush=True)
    time.sleep(5)
