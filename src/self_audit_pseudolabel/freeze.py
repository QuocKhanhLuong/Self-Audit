"""Immutable prediction artifacts and pre-reference integrity verification (no torch import)."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def _digest(body):
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def safe_path(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts: raise ValueError("artifact path escapes run")
    root=Path(root).resolve(); dest=(root/p).resolve()
    if root not in dest.parents: raise ValueError("artifact path escapes run")
    return dest

def validate_arrays(label,valid,soft):
    if label.ndim!=2 or valid.shape!=label.shape or soft.shape!=(4,*label.shape): raise ValueError("pseudo shape mismatch")
    if not np.isfinite(soft).all() or np.any(soft<0) or not np.allclose(soft.sum(0),1,atol=2e-5):
        raise ValueError("invalid soft-label probabilities")
    if not set(np.unique(valid)).issubset({0,1}): raise ValueError("validity is not binary")
    use=valid.astype(bool)
    if not np.isin(label[use],[0,1,2,3]).all() or not np.all(label[~use]==255):
        raise ValueError("UNKNOWN/validity mismatch")
    if not np.array_equal(label[use],soft.argmax(0)[use]): raise ValueError("hard/soft-label disagreement")

def export_pseudo_npz(path,*,pseudo_label,valid,soft_label,metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    label=np.asarray(pseudo_label); valid=np.asarray(valid); soft=np.asarray(soft_label)
    validate_arrays(label,valid,soft)
    with path.open('xb') as f:
        np.savez_compressed(f,pseudo_label=label.astype(np.uint8),valid=valid.astype(np.uint8),
            soft_label=soft.astype(np.float32),metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
    return sha256_file(path)

def write_freeze_manifest(root,entries,config,*,extra_artifacts=()):
    root=Path(root); artifacts={}
    for name in ('teacher.pt','student.pt','train_metrics.json','split.json','source_freeze.json',*extra_artifacts):
        if (root/name).exists(): artifacts[name]=sha256_file(root/name)
    payload={"schema_version":3,"config":config,"entries":entries,"artifacts":artifacts}
    payload['manifest_id']=_digest(payload)
    with (root/'FROZEN.json').open('x',encoding='utf-8') as f: json.dump(payload,f,indent=2,allow_nan=False)
    return payload

def verify_frozen(manifest,*,expected_role=None):
    """Verify ALL files/keys before a consumer opens any reference or trains."""
    manifest=Path(manifest); root=manifest.parent
    payload=json.loads(manifest.read_text()); body=dict(payload); claimed=body.pop('manifest_id',None)
    if claimed!=_digest(body): raise ValueError("freeze manifest digest mismatch")
    if payload.get('schema_version')!=3: raise ValueError("legacy freeze requires regeneration, not silent trust")
    entries=payload.get('entries',[])
    if not entries: raise ValueError("empty frozen cohort")
    paths=set(); keys=set(); samples={}
    for e in entries:
        strict_index(e['t'],'t'); strict_index(e['z'],'z')
        key=(e['patient_id'],e['t'],e['z'])
        if key in keys or e['path'] in paths: raise ValueError("duplicate frozen sample/path")
        keys.add(key); paths.add(e['path'])
        path=safe_path(root,e['path'])
        if sha256_file(path)!=e['sha256']: raise ValueError(f"frozen output changed: {path}")
        with np.load(path,allow_pickle=False) as p:
            validate_arrays(p['pseudo_label'],p['valid'],p['soft_label'])
            meta=json.loads(str(p['metadata_json']))
            strict_index(meta.get('t'),'metadata t'); strict_index(meta.get('z'),'metadata z')
            if any(meta.get(k)!=e[k] for k in ('patient_id','t','z')): raise ValueError("sample identity mismatch")
            label=p['pseudo_label']; valid=p['valid'].astype(bool)
            counts=[int((valid&(label==c)).sum()) for c in range(4)]
            for field,value in [('valid_foreground',sum(counts[1:])),('class_pixels',counts)]:
                if field in e and e[field]!=value: raise ValueError(f"frozen {field} disagrees with arrays")
            if 'valid_fraction' in e and not np.isclose(e['valid_fraction'],valid.mean(),rtol=0,atol=1e-7):
                raise ValueError("frozen valid_fraction disagrees with arrays")
            samples[key]=(meta,label.shape,counts)
    for name,sha in payload.get('artifacts',{}).items():
        if sha256_file(safe_path(root,name))!=sha: raise ValueError(f"frozen artifact changed: {name}")
    config=payload['config']
    if 'export_records' in config:
        for record in config['export_records']:
            shape=record.get('shape',[])
            if len(shape)!=4 or any(type(n) is not int or n<1 for n in shape): raise ValueError('invalid export shape')
        expected={(r['patient_id'],t,z) for r in config['export_records'] for t in range(r['shape'][3]) for z in range(r['shape'][2])}
        if keys!=expected: raise ValueError("incomplete frozen volume cohort")
    role=config.get('artifact_role','generic')
    accepted=(expected_role,) if isinstance(expected_role,str) else expected_role
    if accepted is not None and role not in accepted:
        raise ValueError('required role-specific freeze missing; regenerate teacher/student artifact')
    if role in {'teacher_freeze','student_predictions'}:
        _validate_role(payload,samples,root)
    elif role!='generic':
        raise ValueError('unknown frozen artifact role')
    return payload


def strict_index(value,name,upper=None):
    """JSON indices are integer values, never bools, strings, or truncated floats."""
    if type(value) is not int or value<0 or (upper is not None and value>=upper):
        raise ValueError(f'{name} must be a nonnegative integer within the declared axis')
    return value


def _validate_role(payload,samples,root):
    cfg=payload['config']; role=cfg['artifact_role']
    required={'dataset','split_patients','producer_patient_ids','image_records','export_records',
              'resolved_config','source_sha256','manual_mask_input','scribble_input'}
    if not required.issubset(cfg) or not cfg['source_sha256'] or not cfg['resolved_config']:
        raise ValueError('incomplete role-specific freeze configuration')
    if cfg['manual_mask_input'] is not False or cfg['scribble_input'] is not False:
        raise ValueError('image-only freeze contract violated')
    artifact_names={'teacher.pt','split.json','train_metrics.json'} if role=='teacher_freeze' else {'student.pt','split.json','source_freeze.json'}
    if not artifact_names.issubset(payload.get('artifacts',{})):
        raise ValueError('missing required role-specific artifacts')
    splits=cfg['split_patients']
    if set(splits)!={'train','val','test'} or not splits['train']:
        raise ValueError('invalid locked split schema')
    owners={}
    for split,ids in splits.items():
        if not isinstance(ids,list) or len(set(ids))!=len(ids): raise ValueError('invalid locked patients')
        for pid in ids:
            if not isinstance(pid,str) or pid in owners: raise ValueError('patient leakage/duplicate')
            owners[pid]=split
    # Reconcile the copied locked split without importing data_v3/torch.
    raw=json.loads((root/'split.json').read_text())
    import re
    for split,aliases in {'train':['train','training'],'val':['val','validation','dev'],'test':['test','testing']}.items():
        values=next((raw[k] for k in [a+'_patients' for a in aliases]+aliases if k in raw),[])
        if not isinstance(values,list) or any(not isinstance(v,str) for v in values): raise ValueError('invalid split artifact')
        ids=sorted(set(re.sub(r'_(?:ED|ES)$','',v,flags=re.I) for v in values))
        if sorted(splits[split])!=ids: raise ValueError('split artifact/config mismatch')
    if sorted(cfg['producer_patient_ids'])!=sorted(splits['train']):
        raise ValueError('producer cohort differs from locked train split')
    images={}
    for record in cfg['image_records']:
        pid=record['patient_id']
        if pid in images or owners.get(pid)!=record.get('split'): raise ValueError('invalid image inventory split')
        if record.get('dataset')!=cfg['dataset'] or record.get('axis_order')!='native_XYZT':
            raise ValueError('invalid image inventory contract')
        shape=record['shape']
        if len(shape)!=4 or any(type(n) is not int or n<1 for n in shape) or shape[3]<2:
            raise ValueError('invalid native shape')
        aff=np.asarray(record['affine'])
        if aff.shape!=(4,4) or not np.isfinite(aff).all() or abs(np.linalg.det(aff[:3,:3]))<1e-12:
            raise ValueError('invalid inventory affine')
        if record.get('spatial_unit') not in {'mm','meter','micron','unknown'}: raise ValueError('invalid spatial unit')
        if not re.fullmatch('[0-9a-f]{64}',record.get('image_sha256','')): raise ValueError('missing image identity')
        images[pid]=record
    if not set(splits['train']).issubset(images): raise ValueError('missing producer image inventory')
    exports={}
    for record in cfg['export_records']:
        pid=record['patient_id']
        if pid in exports or images.get(pid)!=record: raise ValueError('export/image inventory mismatch')
        exports[pid]=record
    for split in {r['split'] for r in exports.values()}:
        if {p for p,r in exports.items() if r['split']==split}!=set(splits[split]):
            raise ValueError('incomplete locked export split')
    for e in payload['entries']:
        key=(e['patient_id'],e['t'],e['z']); record=exports[e['patient_id']]
        meta,shape,counts=samples[key]
        if shape!=(record['shape'][1],record['shape'][0]): raise ValueError('native slice shape mismatch')
        if not {'valid_foreground','valid_fraction','class_pixels'}.issubset(e): raise ValueError('missing frozen pixel statistics')
        if e.get('split')!=record['split']: raise ValueError('entry split mismatch')
        for field in ('split','dataset','image_sha256','affine','axis_order','spatial_unit'):
            if meta.get(field)!=record[field]: raise ValueError(f'sample {field} mismatch')
    if role=='student_predictions' or 'source_freeze.json' in payload['artifacts']:
        source=json.loads((root/'source_freeze.json').read_text())
        body=dict(source); claimed=body.pop('manifest_id',None)
        if claimed!=_digest(body) or claimed!=cfg.get('source_manifest_id'):
            raise ValueError('student source freeze identity mismatch')
        if source['config']['split_patients']!=splits:
            raise ValueError('student split differs from source freeze')
        if role=='student_predictions' and cfg.get('checkpoint_sha256')!=payload['artifacts']['student.pt']:
            raise ValueError('student checkpoint identity mismatch')
