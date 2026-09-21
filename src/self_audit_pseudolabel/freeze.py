"""Freeze/export helpers for pseudo-label evidence."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def export_pseudo_npz(path,*,pseudo_label,valid,soft_label,metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,pseudo_label=np.asarray(pseudo_label,dtype=np.uint8),
                       valid=np.asarray(valid,dtype=np.uint8),soft_label=np.asarray(soft_label,dtype=np.float32),
                       metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
    return sha256_file(path)

def write_freeze_manifest(root,entries,config):
    root=Path(root); payload={"config":config,"entries":entries}
    body=json.dumps(payload,sort_keys=True,separators=(",",":")); payload["manifest_id"]=hashlib.sha256(body.encode()).hexdigest()
    out=root/"FROZEN.json"; out.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8"); return payload
