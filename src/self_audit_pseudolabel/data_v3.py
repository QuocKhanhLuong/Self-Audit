"""Image-only full-cine data contract for Self-Audit v3.

No segmentation mask path is accepted by this module. ACDC Info.cfg is used
only for ED/ES frame numbers.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
import re
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass(frozen=True)
class CineRecord:
    dataset: str
    patient_id: str
    image_path: Path
    ed_index: int | None = None
    es_index: int | None = None

@dataclass(frozen=True)
class CineGeometry:
    affine: np.ndarray
    spacing_xyz: tuple[float, float, float]
    patient_left_axis: str | None
    canonical_ras: bool

def parse_acdc_info(path: str | Path):
    ed = es = None
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        m = re.match(r"\s*(ED|ES)\s*:\s*(\d+)\s*$", line, re.I)
        if m:
            if m.group(1).upper() == "ED": ed = int(m.group(2))
            else: es = int(m.group(2))
    return ed, es

def discover_acdc_full_cine(root: str | Path):
    root = Path(root)
    records=[]
    for patient in sorted(p for p in root.rglob("patient*") if p.is_dir()):
        cand=sorted(patient.glob("*4d.nii"))+sorted(patient.glob("*4d.nii.gz"))
        if not cand: continue
        if len(cand)!=1: raise ValueError(f"Expected one 4-D cine for {patient}, found {len(cand)}")
        info=patient/"Info.cfg"
        ed,es=parse_acdc_info(info) if info.exists() else (None,None)
        records.append(CineRecord("acdc",patient.name,cand[0],ed,es))
    if not records: raise FileNotFoundError(f"No ACDC full-cine *4d.nii[.gz] files under {root}")
    return records

def discover_mnms_full_cine(root: str | Path):
    root=Path(root); blocked={"label","labels","mask","masks","seg","segs","segmentations"}; records=[]
    try: import nibabel as nib
    except ImportError as exc: raise ImportError("M&Ms NIfTI discovery requires nibabel") from exc
    for path in sorted(root.rglob("*.nii"))+sorted(root.rglob("*.nii.gz")):
        if any(part.lower() in blocked for part in path.parts): continue
        shape=nib.load(str(path)).shape
        if len(shape)==4 and int(shape[-1])>1:
            records.append(CineRecord("mnms",path.name.split(".nii")[0],path))
    if not records: raise FileNotFoundError(f"No image-only 4-D M&Ms cine under {root}")
    return records

def load_cine(record: CineRecord):
    try: import nibabel as nib
    except ImportError as exc: raise ImportError("Full-cine loading requires nibabel") from exc
    image=nib.as_closest_canonical(nib.load(str(record.image_path)))
    arr=np.asarray(image.dataobj,dtype=np.float32)
    if arr.ndim!=4: raise ValueError(f"Expected 4-D cine, got {arr.shape}")
    cine=np.transpose(arr,(3,2,1,0))
    cine=np.nan_to_num(cine,nan=0.0,posinf=0.0,neginf=0.0)
    lo,hi=np.percentile(cine,[1.,99.]); cine=np.clip(cine,lo,hi)
    mean,std=float(cine.mean()),float(cine.std())
    cine=((cine-mean)/max(std,1e-6)).astype(np.float32,copy=False)
    zooms=image.header.get_zooms()
    geom=CineGeometry(np.asarray(image.affine,dtype=np.float64),(float(zooms[0]),float(zooms[1]),float(zooms[2])),"-x",True)
    return cine,geom

def triplet_z(frame_zhw: np.ndarray,z:int):
    n=int(frame_zhw.shape[0]); idx=[max(0,z-1),z,min(n-1,z+1)]
    return np.stack([frame_zhw[i] for i in idx],axis=0).astype(np.float32,copy=False)

class CineSliceDataset(Dataset):
    def __init__(self,records,cache_records:int=2):
        self.records=list(records); self.cache_records=max(int(cache_records),1)
        self._cache=OrderedDict(); self.index=[]
        try: import nibabel as nib
        except ImportError as exc: raise ImportError("CineSliceDataset requires nibabel") from exc
        for ri,record in enumerate(self.records):
            shape=nib.load(str(record.image_path)).shape
            if len(shape)!=4: raise ValueError(f"Expected 4-D cine at {record.image_path}, got {shape}")
            z_count,t_count=int(shape[2]),int(shape[3])
            for t in range(t_count):
                for z in range(z_count): self.index.append((ri,t,z))
    def _get_record(self,ri):
        if ri in self._cache:
            value=self._cache.pop(ri); self._cache[ri]=value; return value
        value=load_cine(self.records[ri]); self._cache[ri]=value
        while len(self._cache)>self.cache_records: self._cache.popitem(last=False)
        return value
    def __len__(self): return len(self.index)
    def __getitem__(self,item):
        ri,t,z=self.index[item]; cine,geom=self._get_record(ri)
        tp=max(0,t-1); tn=min(cine.shape[0]-1,t+1); rec=self.records[ri]
        return {"prev":torch.from_numpy(triplet_z(cine[tp],z)),"cur":torch.from_numpy(triplet_z(cine[t],z)),
                "nxt":torch.from_numpy(triplet_z(cine[tn],z)),"patient_id":rec.patient_id,"dataset":rec.dataset,
                "t":int(t),"z":int(z),"num_frames":int(cine.shape[0]),"num_slices":int(cine.shape[1]),
                "ed_index":-1 if rec.ed_index is None else int(rec.ed_index),
                "es_index":-1 if rec.es_index is None else int(rec.es_index),
                "patient_left_axis":geom.patient_left_axis}
