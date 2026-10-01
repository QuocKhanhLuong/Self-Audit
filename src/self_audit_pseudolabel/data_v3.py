"""Image-only native cine grids. No reference discovery and no implicit spatial reorientation."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import json,re
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler

@dataclass(frozen=True)
class CineRecord:
    dataset: str
    patient_id: str
    image_path: Path
    ed_index: int | None=None  # ACDC metadata, 1-based; not segmentation supervision.
    es_index: int | None=None

@dataclass(frozen=True)
class CineGeometry:
    affine: np.ndarray
    spacing_xyz: tuple[float,float,float]
    patient_left_axis: str | None
    canonical_ras: bool

def is_reference_path(path):
    p=Path(path)
    blocked={"label","labels","mask","masks","seg","segs","segmentations","scribbles","references"}
    stem=p.name.lower().split(".nii")[0]
    return any(part.lower() in blocked for part in p.parts) or bool(re.search(r"(?:^|_)(gt|mask|label|seg|scribble)(?:_|$)",stem))

def parse_acdc_info(path):
    vals={}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m=re.fullmatch(r"\s*(ED|ES)\s*:\s*(\d+)\s*",line,re.I)
        if m: vals[m[1].upper()]=int(m[2])
    return vals.get("ED"),vals.get("ES")

def _unique(records):
    if not records: raise FileNotFoundError("No image-only full-cine NIfTI found; ED/ES-only files are insufficient")
    ids=[r.patient_id for r in records]
    if len(set(ids)) != len(ids): raise ValueError("duplicate patient cine; use one unambiguous study per patient")
    return records

def discover_acdc_full_cine(root):
    root=Path(root); records=[]
    dirs=([root] if root.name.startswith("patient") else [])+sorted(p for p in root.rglob("patient*") if p.is_dir())
    for patient in dirs:
        candidates=sorted(patient.glob("*4d.nii"))+sorted(patient.glob("*4d.nii.gz"))
        candidates=[p for p in candidates if not is_reference_path(p)]
        if not candidates: continue
        if len(candidates)!=1: raise ValueError(f"ambiguous full cine in {patient}")
        info=patient/"Info.cfg"; ed,es=parse_acdc_info(info) if info.exists() else (None,None)
        records.append(CineRecord("acdc",patient.name,candidates[0],ed,es))
    return _unique(records)

def discover_mnms_full_cine(root):
    import nibabel as nib
    records=[]; root=Path(root)
    for path in sorted(root.rglob("*.nii"))+sorted(root.rglob("*.nii.gz")):
        if is_reference_path(path): continue  # BEFORE even reading the NIfTI header.
        shape=nib.load(str(path)).shape
        if len(shape)==4 and shape[-1]>1:
            stem=path.name.split(".nii")[0]
            patient=re.sub(r"(?:_sa|_4d)$","",stem,flags=re.I)
            records.append(CineRecord("mnms",patient,path))
    return _unique(records)

def inspect_record(record):
    import nibabel as nib
    if is_reference_path(record.image_path): raise ValueError("reference path cannot be an image input")
    im=nib.load(str(record.image_path))
    if len(im.shape)!=4 or min(im.shape)<1 or im.shape[-1]<2:
        raise ValueError(f"expected native [X,Y,Z,T] full cine: {record.image_path}")
    affine=np.asarray(im.affine,dtype=np.float64)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3]))<1e-12:
        raise ValueError("invalid native affine")
    spacing=np.linalg.norm(affine[:3,:3],axis=0)
    # RAS world X increases toward patient right. Only expose a cardinal
    # in-plane approximation when well aligned; oblique/through-plane is UNKNOWN.
    projection=-affine[0,:2]/spacing[:2]; axis=int(np.argmax(np.abs(projection)))
    left=(('+' if projection[axis]>0 else '-')+('x' if axis==0 else 'y')) if abs(projection[axis])>=.9 else ''
    return im,CineGeometry(affine,tuple(float(v) for v in spacing),left,False)

def load_cine(record):
    im,geom=inspect_record(record)
    arr=np.asarray(im.dataobj,dtype=np.float32)
    if not np.isfinite(arr).all(): raise ValueError(f"non-finite image: {record.image_path}")
    # Preserve acquisition slice axis Z, even when affine axes are permuted.
    cine=np.transpose(arr,(3,2,1,0)); lo,hi=np.percentile(cine,[1.,99.])
    cine=np.clip(cine,lo,hi); mean,std=float(cine.mean()),float(cine.std())
    return ((cine-mean)/max(std,1e-6)).astype(np.float32),geom

def triplet_z(frame_zhw,z):
    if frame_zhw.ndim!=3 or not 0<=z<frame_zhw.shape[0]: raise ValueError("invalid slice index")
    return np.stack([frame_zhw[max(0,z-1)],frame_zhw[z],frame_zhw[min(frame_zhw.shape[0]-1,z+1)]]).astype(np.float32)

def read_patient_splits(path):
    payload=json.loads(Path(path).read_text()); splits={}; owners={}
    for split,aliases in {"train":["train","training"],"val":["val","validation","dev"],"test":["test","testing"]}.items():
        values=None
        for key in [a+"_patients" for a in aliases]+aliases:
            if key in payload:
                values=payload[key]; break
        if values is None: values=[]
        if not isinstance(values,list) or any(not isinstance(v,str) for v in values):
            raise ValueError("split manifest requires patient-ID lists")
        ids=sorted(set(re.sub(r"_(?:ED|ES)$","",v,flags=re.I) for v in values))
        for pid in ids:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+",pid) or pid in {'.','..'}: raise ValueError("invalid patient ID")
            if pid in owners: raise ValueError(f"patient leakage: {pid} in {owners[pid]} and {split}")
            owners[pid]=split
        splits[split]=ids
    if not splits["train"]: raise ValueError("empty training patient split")
    return splits

def select_records(records,splits,split):
    wanted=set(splits[split]); lookup={r.patient_id:r for r in _unique(records)}
    missing=wanted-set(lookup)
    if missing: raise ValueError(f"full cine missing for {split} patients: {sorted(missing)}")
    return [lookup[p] for p in sorted(wanted)]

class CineSliceDataset(Dataset):
    def __init__(self,records,cache_records=2):
        self.records=list(records); self.cache_records=max(int(cache_records),1); self._cache=OrderedDict(); self.index=[]
        for ri,r in enumerate(_unique(self.records)):
            im,_=inspect_record(r)
            for t in range(im.shape[3]):
                for z in range(im.shape[2]): self.index.append((ri,t,z))
    def _get_record(self,ri):
        if ri in self._cache: value=self._cache.pop(ri)
        else: value=load_cine(self.records[ri])
        self._cache[ri]=value
        while len(self._cache)>self.cache_records: self._cache.popitem(last=False)
        return value
    def __len__(self): return len(self.index)
    def __getitem__(self,item):
        ri,t,z=self.index[item]; cine,geom=self._get_record(ri); rec=self.records[ri]
        return {"prev":torch.from_numpy(triplet_z(cine[max(0,t-1)],z)),"cur":torch.from_numpy(triplet_z(cine[t],z)),
                "nxt":torch.from_numpy(triplet_z(cine[min(cine.shape[0]-1,t+1)],z)),
                "patient_id":rec.patient_id,"dataset":rec.dataset,"t":t,"z":z,
                "num_frames":int(cine.shape[0]),"num_slices":int(cine.shape[1]),
                "ed_index":-1 if rec.ed_index is None else rec.ed_index,
                "es_index":-1 if rec.es_index is None else rec.es_index,
                "patient_left_axis":geom.patient_left_axis or ''}

class PatientBatchSampler(Sampler):
    """Batches stay within one patient/native shape. No distortion or padded evidence."""
    def __init__(self,patient_keys,batch_size,seed=42):
        if batch_size<1: raise ValueError("batch size must be positive")
        self.groups={}; self.batch_size=batch_size; self.seed=seed; self.epoch=0
        for i,p in enumerate(patient_keys): self.groups.setdefault(p,[]).append(i)
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch); batches=[]
        for group in self.groups.values():
            ids=rng.permutation(group).tolist()
            batches.extend(ids[i:i+self.batch_size] for i in range(0,len(ids),self.batch_size))
        rng.shuffle(batches); self.epoch+=1
        yield from batches
    def __len__(self): return sum((len(g)+self.batch_size-1)//self.batch_size for g in self.groups.values())
