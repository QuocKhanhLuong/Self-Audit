#!/usr/bin/env python3
"""Upload a verified checkpoint ZIP using an existing rclone Drive remote; no sharing changes."""
import argparse
from pathlib import Path
import datetime,hashlib,json,subprocess,time,traceback
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path('/workspace/self-audit-v2'))
parser.add_argument('--drive-root',default='KhanhDrive:Self-Audit-v2-artifacts',help='Destination on an already configured Google Drive remote')
args=parser.parse_args();root=args.root.resolve()
info=json.loads((root/'exports/LATEST_EXPORT.json').read_text());out=Path(info['directory']);name=info['name'];archive=Path(info['zip'])
remote=args.drive_root.rstrip('/')+'/'+name
log=root/'exports/rclone-final4-upload.log';common=['--tpslimit','1','--contimeout','20s','--timeout','120s','--retries','6','--retries-sleep','30s','--low-level-retries','3']
def run(args,**kwargs):
 for attempt in range(20):
  try:
   return subprocess.run(['rclone']+args+common,check=True,**kwargs)
  except subprocess.CalledProcessError:
   if attempt==19:raise
   print('rclone command failed; retrying in 30s; attempt '+str(attempt+1),flush=True);time.sleep(30)
def stage(value,**more):
 p=root/'vram24/FINALIZATION_STATUS.json';p.parent.mkdir(parents=True,exist_ok=True);old=json.loads(p.read_text()) if p.exists() else {};old.update(stage=value,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**more);temp=p.with_suffix('.upload.tmp');temp.write_text(json.dumps(old,indent=2)+'\n');temp.replace(p);print(value,flush=True)
try:
 stage('PREPARING_DRIVE_ARCHIVE',archive=str(archive))
 if not (out/'EXPORT_RECEIPT.json').exists():raise FileNotFoundError('Package and verify the archive before upload')
 receipt=json.loads((out/'EXPORT_RECEIPT.json').read_text());assert receipt['zip_crc_check']=='PASS'
 h=hashlib.sha256()
 with archive.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 assert h.hexdigest()==receipt['sha256'], 'Archive changed after packaging'
 selected=[archive.name,archive.name+'.sha256','ARCHIVE_MANIFEST.json','README_RESTORE.txt']
 filters=out/'upload-filter.txt';filters.write_text(''.join('+ /'+p+'\n' for p in selected)+'- **\n')
 stage('UPLOADING_CHECKPOINT_ARCHIVE',drive_path=remote,archive_sha256=receipt['sha256'])
 run(['copy',str(out),remote,'--immutable','--filter-from',str(filters),'--transfers','1','--checkers','1','--drive-chunk-size','64M','--stats','15s','--log-level','INFO','--log-file',str(log)])
 stage('VERIFYING_DRIVE_ARCHIVE',drive_path=remote)
 run(['check',str(out),remote,'--one-way','--filter-from',str(filters),'--checkers','1','--combined',str(out/'rclone-check.txt'),'--log-file',str(log),'--log-level','INFO'])
 lines=(out/'rclone-check.txt').read_text().splitlines();assert len(lines)==4 and all(p.startswith('= ') for p in lines),lines
 listing=run(['lsjson',remote,'--hash'],stdout=subprocess.PIPE,text=True)
 entries=json.loads(listing.stdout);z=next(x for x in entries if x['Name']==archive.name)
 hashes={k.lower():v for k,v in z.get('Hashes',{}).items()};assert hashes.get('md5')==receipt['md5'];assert z['Size']==receipt['size_bytes']
 receipt.update(status='UPLOADED_AND_VERIFIED',drive_directory=remote,drive_archive=remote+'/'+archive.name,remote_md5=hashes['md5'],remote_size_bytes=z['Size'],remote_id=z.get('ID'),private_drive_view_url=('https://drive.google.com/file/d/'+z['ID']+'/view') if z.get('ID') else None,remote_checksum_comparison='4/4 matched',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (out/'EXPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 run(['copyto',str(out/'EXPORT_RECEIPT.json'),remote+'/EXPORT_RECEIPT.json','--immutable','--log-file',str(log),'--log-level','INFO'])
 stage('CHECKPOINT_ARCHIVE_UPLOADED_AND_VERIFIED',drive_path=receipt['drive_archive'],drive_view_url=receipt['private_drive_view_url'],archive_sha256=receipt['sha256'])
 print(json.dumps(receipt,indent=2),flush=True)
except BaseException as e:
 stage('CHECKPOINT_UPLOAD_FAILED',error=repr(e),drive_path=remote);traceback.print_exc();raise
