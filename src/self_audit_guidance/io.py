"""Image-only discovery and immutable, hash-verified stage artifacts."""
from __future__ import annotations
from pathlib import Path
import hashlib, json, re
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_image(path):
    p=Path(path)
    banned=r'(^|[_ .-])(gt|ground_truth|masks?|labels?|segs?|segmentations?|scribbles?|references?)([_ .-]|$)'
    if any(re.search(banned,part.lower()) for part in p.parts):
        raise ValueError(f'reference-like path refused before reading: {p}')
    return p


def read_splits(path):
    raw=read_json(path);out={};seen=set()
    for name,aliases in {'train':('train','training'),'val':('val','validation','dev'),'test':('test','testing')}.items():
        candidates=[raw[key] for key in [*(a+'_patients' for a in aliases),*aliases] if key in raw]
        if len(candidates)>1 and any(c!=candidates[0] for c in candidates[1:]):raise ValueError('ambiguous split aliases')
        ids=candidates[0] if candidates else []
        if not isinstance(ids,list) or any(not isinstance(x,str) for x in ids):raise ValueError('patient split lists required')
        ids=[re.sub(r'_(ED|ES)$','',x) for x in ids]
        if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',x) or '..' in x for x in ids):raise ValueError('unsafe patient ID')
        ids=sorted(set(ids))
        if seen.intersection(ids):raise ValueError('patient leakage between splits')
        seen.update(ids);out[name]=ids
    if not out['train']:raise ValueError('empty TRAIN split')
    return out


def discover(root,dataset,patient_ids):
    root=safe_image(Path(root).resolve())
    candidates={pid:[] for pid in patient_ids}
    # No Info.cfg, ED/ES selection, label headers, bounding boxes or disease metadata.
    if dataset=='acdc':
        dirs=[root] if root.name in candidates else [root/pid for pid in patient_ids]
        for d in dirs:
            if d.name not in candidates:continue
            for pattern in ('*4d.nii','*4d.nii.gz'):
                for p in d.glob(pattern):candidates[d.name].append(safe_image(p))
    elif dataset=='mnms':
        for p in root.rglob('*.nii*'):
            if not p.is_file():continue
            m=re.fullmatch(r'([A-Za-z0-9.-]+)_(?:sa|4d)\.nii(?:\.gz)?',p.name)
            if m and m[1] in candidates:candidates[m[1]].append(safe_image(p))
    else:raise ValueError('dataset must be acdc or mnms')
    for pid,paths in candidates.items():
        if len(paths)!=1:raise ValueError(f'{pid}: expected one raw full-cine image, found {len(paths)}')
    return {pid:paths[0] for pid,paths in candidates.items()}


def read_cine(path):
    path=safe_image(Path(path).resolve())
    try:import nibabel as nib
    except ImportError as e:raise RuntimeError('nibabel is required for raw NIfTI; no preprocessed fallback') from e
    before=digest(path);nii=nib.load(str(path))
    if len(nii.shape)!=4 or min(nii.shape)<1 or nii.shape[3]<2:raise ValueError('full native XYZ T cine required')
    affine=np.asarray(nii.affine,dtype=np.float64)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3]))<1e-12:raise ValueError('invalid native affine')
    a=np.asarray(nii.dataobj,dtype=np.float32)
    if not np.isfinite(a).all():raise ValueError('nonfinite cine')
    if digest(path)!=before:raise ValueError('input changed while reading')
    lo,hi=np.percentile(a,[1,99]);a=np.clip(a,lo,hi)
    mean,std=float(a.mean()),float(a.std());a=(a-mean)/max(std,1e-6)
    spacing=np.linalg.norm(affine[:3,:3],axis=0)
    projection=-affine[0,:2]/spacing[:2];axis=int(np.argmax(np.abs(projection)))
    left=(('+' if projection[axis]>0 else '-')+('x' if axis==0 else 'y')) if abs(projection[axis])>=.9 else ''
    return a.transpose(3,2,1,0).copy(),dict(image_sha256=before,shape_xyzt=list(nii.shape),
        affine=affine.tolist(),spacing_xyz=spacing.tolist(),patient_left_axis=left,
        spatial_unit=nii.header.get_xyzt_units()[0],normalization=dict(lo=float(lo),hi=float(hi),mean=mean,std=std),
        preprocessing='native_TZYX_image_percentiles_no_ROI_no_ED_ES',raw_image=str(path))


def freeze(folder,manifest):
    folder=Path(folder)
    files={p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='FROZEN.json'}
    payload={**manifest,'schema':'self_audit.stable_guidance.v1','files':files}
    if (folder/'FROZEN.json').exists():raise FileExistsError('frozen artifact cannot be replaced')
    write_json(folder/'FROZEN.json',payload)
    return payload


def verify(folder,role=None):
    folder=Path(folder);m=read_json(folder/'FROZEN.json')
    if m.get('schema')!='self_audit.stable_guidance.v1' or (role and m.get('role')!=role):raise ValueError('artifact contract mismatch')
    if not isinstance(m.get('files'),dict):raise ValueError('missing file manifest')
    actual={p.name for p in folder.iterdir() if p.is_file() and p.name!='FROZEN.json'}
    if actual!=set(m['files']):raise ValueError('artifact files differ from frozen inventory')
    for name,expected in m['files'].items():
        p=folder/name
        if Path(name).name!=name or p.is_symlink() or not p.is_file() or digest(p)!=expected:raise ValueError(f'artifact hash mismatch: {name}')
    return m


def load_bank_patient(folder,pid):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',pid) or '..' in pid:raise ValueError('unsafe patient ID')
    with np.load(Path(folder)/(pid+'.npz'),allow_pickle=False) as f:
        if set(f.files)!={'image','features','q','r0','reliability','labels'}:raise ValueError('unexpected bank fields')
        d={key:f[key] for key in f.files}
    im,q,feat=d['image'],d['q'],d['features']
    if im.ndim!=4 or feat.ndim!=5 or feat.shape[:2]!=im.shape[:2] or q.shape!=(*feat.shape[:-1],4):raise ValueError('bank grid mismatch')
    if d['r0'].shape!=q.shape or d['reliability'].shape!=q.shape[:-1] or d['labels'].shape!=q.shape[:-1]:raise ValueError('bank field shape mismatch')
    if any(not np.isfinite(v).all() for v in d.values()):raise ValueError('nonfinite bank')
    if not np.allclose(q.sum(-1),1,atol=1e-5) or np.any(q<0):raise ValueError('invalid bank probabilities')
    if np.any((d['reliability']<0)|(d['reliability']>1)):raise ValueError('invalid reliability')
    if np.any(~np.isin(d['labels'],[0,1,2,3,255])):raise ValueError('invalid bank class')
    return d
