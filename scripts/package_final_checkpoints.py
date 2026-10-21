#!/usr/bin/env python3
"""Package the four completed ACDC final checkpoints; never include patient arrays."""
import argparse
from pathlib import Path
import datetime,hashlib,importlib.metadata as md,json,platform,subprocess,zipfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--root',type=Path,default=Path('/workspace/self-audit-v2'),help='Existing artifact root containing vram24/runs, weights and external/dinov3')
args=parser.parse_args();repo=args.repo.resolve();root=args.root.resolve()
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
name=f'Self-Audit-v2_L8-S42_C256-B52_final4_{stamp}'
exports=root/'exports';exports.mkdir(parents=True,exist_ok=True);out=exports/name;out.mkdir()
archive=out/(name+'.zip')
info={'name':name,'directory':str(out),'zip':str(archive)}
(exports/'LATEST_EXPORT.json').write_text(json.dumps(info,indent=2)+'\n')
source=root/'external/dinov3'
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()=='6876159a11b4df116f30f667f8c9888617df0751'
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
subprocess.run(['git','-C',str(source),'bundle','create',str(out/'dinov3-official-source.bundle'),'--all'],check=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
source_files=sorted((repo/'self_audit').glob('*.py'))+[repo/'train.py']
source_sha=hashlib.sha256(''.join(f'{p.name}:{digest(p)}\n' for p in source_files).encode()).hexdigest()
files=[];checkpoints=[]
for method in ['supervised','supervised_extra','ema_full','ema_block']:
 run=root/'vram24/runs'/f'L8_S42_{method}_C256_B52_10000';report=json.loads((run/'REPORT.json').read_text());manifest=run/'final/MANIFEST.json';m=json.loads(manifest.read_text())
 assert report['status']=='COMPLETED' and report['steps']==10000 and not report['synthetic']
 assert digest(manifest)==report['final_manifest_sha256']
 assert m['source_sha256']==source_sha, 'Current source differs from the checkpoint; use its pinned worktree'
 assert digest(run/'final/weights.pt')==m['payload_sha256']
 for p in sorted((run/'final').iterdir()):
  assert p.name in ['weights.pt','MANIFEST.json']
  files.append((p,f'checkpoints/{run.name}/final/{p.name}'))
 for fname in ['CONFIG.json','STARTUP.json','REPORT.json']:
  files.append((run/fname,f'checkpoints/{run.name}/{fname}'))
 ev=root/'vram24/evaluation'/run.name
 for fname in ['METRICS.json','RECIPE.json','PREDICTIONS_FROZEN.json']:
  files.append((ev/fname,f'evaluation_receipts/{run.name}/{fname}'))
 checkpoints.append({'method':method,'run':run.name,'steps':report['steps'],'manifest_sha256':report['final_manifest_sha256'],'weights_payload_sha256':m['payload_sha256'],'source_sha256':m['source_sha256'],'training_data_sha256':m['data_sha256'],'exact_resume_supported':False})
p=root/'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth';assert digest(p)=='21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54';files.append((p,'pretrained/'+p.name))
files.append((out/'dinov3-official-source.bundle','source/dinov3-official-source.bundle'))
# Current v2 source only, not historical trainers/checkpoints or repository history.
source_paths=['train.py','evaluate.py','prepare_acdc.py','ACTIVE_PIPELINE.json','requirements.txt','README.md','docs/METHOD.md','docs/RUNBOOK.md','docs/EXPERIMENTS.md','splits/acdc_reviewed_locked_train80_val18.json']
source_paths += [str(p.relative_to(repo)) for p in sorted((repo/'self_audit').glob('*.py'))]
for rel in source_paths:files.append((repo/rel,'source/Self-Audit/'+rel))
for rel in ['REPORT.md','RESULTS_MAIN.csv','RESULTS_ABLATION.csv','PATIENT_METRICS.csv','EXPERIMENT_STATUS.csv','DATA_PREPARATION.csv']:
 files.append((repo/'results'/rel,'reports/'+rel))
versions={p:md.version(p) for p in ['torch','numpy','scipy','nibabel','torchmetrics','omegaconf','matplotlib']}
manifest={'schema':'self_audit.final_checkpoint_export.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'source_worktree':str(repo),'python':platform.python_version(),'packages':versions,'official_dino_source_commit':'6876159a11b4df116f30f667f8c9888617df0751','pretrained_weight_provenance':'user-authorized mirror; full SHA matches mirror LFS and official short hash; independent full upstream certification unavailable','checkpoints':checkpoints,'excluded':'all patient images/masks/cines, full frozen probabilities, credentials, raw logs, intermediate checkpoints, incomplete/native runs and pilot checkpoints','files':[]}
for path,rel in files:
 assert path.is_file()
 manifest['files'].append({'path':rel,'size_bytes':path.stat().st_size,'sha256':digest(path)})
(out/'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
readme='''Self-Audit v2: four completed L8/S42 crop256/batch52 final checkpoints

Contains supervised, supervised_extra, ema_full and ema_block, all step10000.
Each final/weights.pt retains student, EMA teacher, optimizer states and Torch RNG;
final/MANIFEST.json retains config, label IDs, NumPy RNG and frozen payload hashes.
Exact resume is NOT implemented. Native and crop experiments are distinct factors.
Pretrained DINOv3 ConvNeXt-Tiny weights and pinned official source bundle included.
No raw patient data, masks, cines, full prediction arrays or credentials included.
Intermediate checkpoints and pilots are excluded; they remain on the local host.

Verify ZIP CRC using unzip -t. Check ARCHIVE_MANIFEST.json SHA256 per file.
Verify the external archive .sha256 receipt before extraction.

Reload: source/Self-Audit contains the executed current-v2 source files. Install
its requirements and pinned official DINO source dependencies. To recreate the
DINO source repo: git clone source/dinov3-official-source.bundle dinov3; then
checkout commit 6876159a11b4df116f30f667f8c9888617df0751.
The original hashed manifests retain absolute source/weight paths. For unchanged
load_checkpoint() verification, recreate these paths on a Linux host/container:
/workspace/self-audit-v2/external/dinov3
/workspace/self-audit-v2/weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
Do not silently edit hashed manifests or claim approximate/exact resume. The
original source worktree /root/Self-Audit-vram24 is preserved on the current host.
The source SHA256 in each checkpoint must match the unpacked v2 Python source.
Use the final manifest SHA printed in ARCHIVE_MANIFEST.json when loading/scoring.

Scores are one-seed exposed VAL18 development results, not clinical evidence.
No full Self-Audit guard experiment, TEST Dice, HD95 or multi-seed result exists.
'''
(out/'README_RESTORE.txt').write_text(readme)
partial=archive.with_suffix('.zip.partial')
assert not archive.exists() and not partial.exists()
with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
 for path,rel in files:
  z.write(path,name+'/'+rel);print('PACKED',rel,flush=True)
 z.write(out/'ARCHIVE_MANIFEST.json',name+'/ARCHIVE_MANIFEST.json');z.write(out/'README_RESTORE.txt',name+'/README_RESTORE.txt')
with zipfile.ZipFile(partial) as z:
 bad=z.testzip();assert bad is None,bad
partial.rename(archive)
h=hashlib.sha256();m=hashlib.md5()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b);m.update(b)
sha=h.hexdigest();md5=m.hexdigest();(out/(archive.name+'.sha256')).write_text(sha+'  '+archive.name+'\n')
receipt={'archive':str(archive),'size_bytes':archive.stat().st_size,'sha256':sha,'md5':md5,'zip_crc_check':'PASS','file_count':len(files)+2,'checkpoint_methods':[r['method'] for r in checkpoints],'status':'PACKAGED_NOT_UPLOADED'}
(out/'EXPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
