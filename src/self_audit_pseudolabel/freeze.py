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

def write_freeze_manifest(root,entries,config):
    root=Path(root); artifacts={}
    for name in ('teacher.pt','train_metrics.json','split.json'):
        if (root/name).exists(): artifacts[name]=sha256_file(root/name)
    payload={"schema_version":3,"config":config,"entries":entries,"artifacts":artifacts}
    payload['manifest_id']=_digest(payload)
    with (root/'FROZEN.json').open('x',encoding='utf-8') as f: json.dump(payload,f,indent=2,allow_nan=False)
    return payload

def verify_frozen(manifest):
    """Verify ALL files/keys before a consumer opens any reference or trains."""
    manifest=Path(manifest); root=manifest.parent
    payload=json.loads(manifest.read_text()); body=dict(payload); claimed=body.pop('manifest_id',None)
    if claimed!=_digest(body): raise ValueError("freeze manifest digest mismatch")
    if payload.get('schema_version')!=3: raise ValueError("legacy freeze requires regeneration, not silent trust")
    entries=payload.get('entries',[])
    if not entries: raise ValueError("empty frozen cohort")
    paths=set(); keys=set()
    for e in entries:
        key=(e['patient_id'],e['t'],e['z'])
        if key in keys or e['path'] in paths: raise ValueError("duplicate frozen sample/path")
        keys.add(key); paths.add(e['path'])
        path=safe_path(root,e['path'])
        if sha256_file(path)!=e['sha256']: raise ValueError(f"frozen output changed: {path}")
        with np.load(path,allow_pickle=False) as p:
            validate_arrays(p['pseudo_label'],p['valid'],p['soft_label'])
            meta=json.loads(str(p['metadata_json']))
            if any(meta.get(k)!=e[k] for k in ('patient_id','t','z')): raise ValueError("sample identity mismatch")
    for name,sha in payload.get('artifacts',{}).items():
        if sha256_file(safe_path(root,name))!=sha: raise ValueError(f"frozen artifact changed: {name}")
    config=payload['config']
    if 'export_records' in config:
        expected={(r['patient_id'],t,z) for r in config['export_records'] for t in range(r['shape'][3]) for z in range(r['shape'][2])}
        if keys!=expected: raise ValueError("incomplete frozen volume cohort")
    return payload
