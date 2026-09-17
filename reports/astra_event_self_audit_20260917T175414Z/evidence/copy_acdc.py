import subprocess, shlex, tarfile, json, time, hashlib
from pathlib import Path
start=time.monotonic()
out=Path('/tmp/astra_event_acdc_training'); out.mkdir(exist_ok=True)
remote = """from pathlib import Path
import tarfile,sys,json
root=Path('/root/Self-Audit/data/ACDC')
manifest=Path('/root/Self-Audit/splits/acdc_patient_split_seed42.json')
m=json.loads(manifest.read_text())
ids=set(m['train_patients'])|set(m['val_patients'])
assert len(ids)==100
paths=[]
for pid in sorted(ids):
    d=root/'training'/pid
    assert d.is_dir(), d
    masks=sorted(d.glob('*_gt.nii'))+sorted(d.glob('*_gt.nii.gz'))
    assert len(masks)==2, (pid,len(masks))
    for mask in masks:
        img=Path(str(mask).replace('_gt.nii','.nii'))
        assert img.exists(), img
        paths.extend([img,mask])
    if (d/'Info.cfg').exists(): paths.append(d/'Info.cfg')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
    for p in paths: t.add(p,arcname=str(p.relative_to(root)),recursive=False)
    t.add(manifest,arcname='acdc_patient_split_seed42.json',recursive=False)
    citation=root/'MANDATORY_CITATION.md'
    if citation.exists(): t.add(citation,arcname='MANDATORY_CITATION.md',recursive=False)
"""
archive=out/'training_pairs.tar.gz'
with archive.open('wb') as f:
    r=subprocess.run(['ssh','-o','BatchMode=yes','vast-gpu','python3 -c '+shlex.quote(remote)],stdout=f,stderr=subprocess.PIPE)
if r.returncode: raise RuntimeError(r.stderr.decode())
with tarfile.open(archive) as t:
    for item in t:
        if item.issym() or item.islnk() or Path(item.name).is_absolute() or '..' in Path(item.name).parts: raise RuntimeError('Unsafe archive entry')
        t.extract(item,out,filter='data')
receipt={'source':'vast-gpu:/root/Self-Audit/data/ACDC/training','destination':str(out),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-start,'test_data_copied':False,'files':len(list(out.rglob('*.nii'))),'manifest_sha256':hashlib.sha256((out/'acdc_patient_split_seed42.json').read_bytes()).hexdigest()}
Path('/Users/alvinluong/Self-Audit-event/reports/astra_event_self_audit_20260917T175414Z/evidence/acdc_copy_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
print((out/'MANDATORY_CITATION.md').read_text())
