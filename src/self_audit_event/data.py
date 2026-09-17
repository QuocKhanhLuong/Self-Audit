"""Explicit phantom fixtures and patient-preserving ACDC adapters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class PhantomDataset(Dataset):
    """Synthetic mechanism fixture. Never labeled ACDC or clinical evidence."""
    def __init__(self, count=64, image_size=32, seed=0):
        rng = np.random.default_rng(seed)
        self.rows = []
        yy, xx = np.mgrid[-1:1:complex(image_size), -1:1:complex(image_size)]
        for idx in range(count):
            cx, cy = rng.uniform(-.15, .15, 2)
            r = ((xx-cx)/rng.uniform(.8,1.1))**2 + ((yy-cy)/rng.uniform(.8,1.1))**2
            y = np.zeros((image_size,image_size), np.int64)
            y[r < .38] = 2
            y[r < .18] = 3
            y[(xx-cx+.47)**2+(yy-cy)**2 < .06] = 1
            levels = np.array([0., .65, .32, .95])
            x = levels[y] + .08*rng.standard_normal(y.shape) + .08*xx
            image = np.stack([x+.02*rng.standard_normal(y.shape), x, x+.02*rng.standard_normal(y.shape)]).astype('float32')
            self.rows.append({'image':torch.from_numpy(image), 'mask':torch.from_numpy(y),
                              'patient_id':f'phantom_{seed}_{idx:04d}', 'case_id':f'phantom_{seed}_{idx:04d}', 'slice_idx':0})

    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def load_acdc(root, manifest, *, image_size=128, validation_patient_limit=None):
    """Mandatory existing patient manifest. No fallback split or test-label reads."""
    from self_audit.data.acdc import ACDCDataset, discover_acdc_records
    root, manifest = Path(root).resolve(), Path(manifest).resolve()
    payload = json.loads(manifest.read_text())
    owners = {}
    for split in ('train','val','test'):
        patients = payload.get(split+'_patients', [])
        if not isinstance(patients,list) or (split != 'test' and not patients):
            raise ValueError('Existing train_patients/val_patients manifest required')
        for pid in patients:
            if not isinstance(pid,str) or pid in owners:
                raise ValueError('invalid or overlapping patient split')
            owners[pid] = split
    records = discover_acdc_records(root)
    found = {r.patient_id for r in records}
    if set(owners)-found: raise ValueError('manifest patients absent from paired data')
    groups = {'train':[], 'val':[], 'test':[]}
    for record in records:
        split = owners.get(record.patient_id)
        if record.split == 'test':
            if split in ('train','val'): raise ValueError('official test leakage')
            split = 'test'
        if split is None: raise ValueError('unassigned patient '+record.patient_id)
        if record.split == 'val' and split != 'val': raise ValueError('official validation conflict')
        groups[split].append(record)
    all_validation = sorted({r.patient_id for r in groups['val']})
    if validation_patient_limit is not None:
        if validation_patient_limit < 1: raise ValueError('invalid bounded validation limit')
        keep = set(all_validation[:validation_patient_limit])
        groups['val'] = [r for r in groups['val'] if r.patient_id in keep]
    datasets = {s:ACDCDataset(records=groups[s], image_size=image_size, augment=False,
                             foreground_only=False, depth_axis=2, max_cache=2) for s in ('train','val')}
    inventory = {}
    for s in ('train','val'):
        inventory[s] = []
        for record in groups[s]:
            row = {'patient_id':record.patient_id, 'case_id':record.case_id}
            for kind,p in [('image',record.image_path),('mask',record.mask_path)]:
                p=Path(p); digest=hashlib.sha256()
                with p.open('rb') as stream:
                    for block in iter(lambda:stream.read(2**20),b''): digest.update(block)
                row[kind] = str(p.resolve());row[kind+'_sha256'] = digest.hexdigest()
            inventory[s].append(row)
    identity = {'source':'REAL_ACDC', 'root':str(root),'manifest':str(manifest),
                'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'image_size':image_size,'depth_axis':2,'augment':False,
                'validation_patients':sorted({r.patient_id for r in groups['val']}),
                'all_manifest_validation_patients':all_validation,
                'bounded_validation_subset':validation_patient_limit is not None,
                'test_images_or_masks_opened':False,'inventory':inventory}
    return datasets, identity
