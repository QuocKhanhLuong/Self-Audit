"""Disk-backed cine export; temporal rejection holds at most three slices in RAM."""
from __future__ import annotations
from contextlib import ExitStack
import resource
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data_v3 import CineSliceDataset
from .freeze import export_pseudo_npz
from .consistency import consistency_gate
from .progress_v3 import progress_bar


def export_native(records, inventory, root, predict, consistency, *, log_every=10, progress=True):
    import nibabel as nib
    root = Path(root)
    entries = []; volumes = []; patient_reports = []
    if consistency.get('slice_weight',0)!=0:
        raise ValueError('streaming export requires rejection-only temporal correspondence')
    if log_every<1: raise ValueError("log_every must be positive")
    for patient_index,record in enumerate(records):
        info = inventory[record.patient_id]; x,y,zmax,tmax = info['shape']
        ds = CineSliceDataset([record],cache_records=1)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='.cine-',dir=root) as scratch, ExitStack() as cleanup:
            bar=progress_bar(total=2*len(ds),desc=f'export {patient_index+1}/{len(records)} predict/write',
                             enabled=progress,unit='slice-pass')
            if bar is not None: cleanup.callback(bar.close)
            arrays = {}
            seen = set()
            for b in DataLoader(ds,batch_size=1,shuffle=False,num_workers=0):
                t,z = int(b['t'][0]),int(b['z'][0])
                if (t,z) in seen: raise ValueError('duplicate patient export sample')
                seen.add((t,z))
                prediction = predict(b)
                fields = {'soft':prediction['soft_label'][0], 'valid':prediction['valid'][0]}
                if consistency['enabled']:
                    fields.update(fp=prediction['flow_prev'][0],fn=prediction['flow_next'][0])
                for name,tensor in fields.items():
                    value = tensor.detach().cpu().numpy()
                    if name not in arrays:
                        arrays[name] = np.lib.format.open_memmap(Path(scratch)/f'{name}.npy', mode='w+',
                            dtype=value.dtype,shape=(tmax,zmax,*value.shape))
                    arrays[name][t,z] = value
                if bar is not None: bar.update(1)
            if seen!={(t,z) for t in range(tmax) for z in range(zmax)}:
                raise ValueError('incomplete patient export')
            native = np.lib.format.open_memmap(Path(scratch)/'native.npy',mode='w+',
                                              dtype=np.uint8,shape=(x,y,zmax,tmax))
            counts = [0]*4; frame_counts = {}
            for t in range(tmax):
                frame_counts[str(t)] = [0]*4
                for z in range(zmax):
                    lo,hi = max(0,t-1),min(tmax,t+2)
                    soft = torch.from_numpy(np.array(arrays['soft'][lo:hi,z:z+1]))
                    valid = torch.from_numpy(np.array(arrays['valid'][lo:hi,z:z+1])).bool()
                    if consistency['enabled']:
                        fp = torch.from_numpy(np.array(arrays['fp'][lo:hi,z:z+1]))
                        fn = torch.from_numpy(np.array(arrays['fn'][lo:hi,z:z+1]))
                        soft,valid = consistency_gate(soft,valid,temporal_weight=consistency['temporal_weight'],
                            slice_weight=0,min_agreement=consistency['min_agreement'],flow_prev=fp,flow_next=fn)
                    prob = soft[t-lo,0].numpy(); use = valid[t-lo,0].numpy()
                    label = np.where(use,prob.argmax(0),255).astype(np.uint8)
                    meta = {k:info[k] for k in ('patient_id','dataset','split','image_sha256','affine','axis_order','spatial_unit')}
                    meta.update(t=t,z=z)
                    path = Path('pseudo')/record.patient_id/f't{t:03d}_z{z:03d}.npz'
                    digest = export_pseudo_npz(root/path,pseudo_label=label,valid=use,soft_label=prob,metadata=meta)
                    pixels = [int((use&(label==c)).sum()) for c in range(4)]
                    entries.append({'path':str(path),'sha256':digest,'patient_id':record.patient_id,
                        't':t,'z':z,'split':info['split'],'valid_foreground':sum(pixels[1:]),
                        'valid_fraction':float(use.mean()),'class_pixels':pixels})
                    counts = [a+b for a,b in zip(counts,pixels)]
                    frame_counts[str(t)] = [a+b for a,b in zip(frame_counts[str(t)],pixels)]
                    native[:,:,z,t] = label.T
                    if bar is not None: bar.update(1)
            if bar is not None: bar.set_description(f'export {patient_index+1}/{len(records)} save NIfTI')
            path = Path('native')/f'{record.patient_id}.nii.gz'
            (root/path).parent.mkdir(parents=True,exist_ok=True)
            image = nib.Nifti1Image(native,np.asarray(info['affine']))
            image.header.set_xyzt_units(info['spatial_unit'],info.get('temporal_unit','unknown'))
            if 'zooms' in info: image.header.set_zooms(info['zooms'])
            nib.save(image,root/path)
            volumes.append(str(path))
            staged_bytes = sum(p.stat().st_size for p in Path(scratch).iterdir())
            patient_reports.append({'patient_id':record.patient_id,'split':info['split'],
                'patient_left_axis':ds[0]['patient_left_axis'],'class_pixels':counts,
                'frame_class_pixels':frame_counts,
                'phase_class_pixels':{phase:frame_counts.get(str(info[field]-1)) for phase,field in [('ED','ed_index'),('ES','es_index')] if type(info.get(field)) is int},
                'missing_foreground_classes':[c for c in (1,2,3) if counts[c]==0],
                'support_status':'MISSING_CLASSES' if any(counts[c]==0 for c in (1,2,3)) else 'ALL_CLASSES_OBSERVED',
                'quality_status':'NOT_EVALUATED','cache_hits':ds.cache_hits,'cache_misses':ds.cache_misses,
                'cine_load_seconds':ds.load_seconds,'export_seconds':time.perf_counter()-started,
                'staging_bytes':staged_bytes})
            del image
            for array in [*arrays.values(),native]:
                array.flush(); array._mmap.close()
        if (patient_index+1)%log_every==0 or patient_index+1==len(records):
            print(f'[EXPORT] patients={patient_index+1}/{len(records)} patient={record.patient_id} class_pixels={counts}',flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {'patient_rows':patient_reports,'peak_process_host_rss_bytes':int(rss if sys.platform=='darwin' else rss*1024),
              'temporal_buffer_policy':'at_most_three_slices','quality_status':'NOT_EVALUATED'}
    return entries,volumes,report
