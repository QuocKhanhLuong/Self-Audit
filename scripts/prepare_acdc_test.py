#!/usr/bin/env python3
"""Prepare an explicitly authorized frozen TEST evaluation, never training data.

The Drive testing copy and reference provenance are not independently certified
as an official challenge-server evaluation. Reject exact content overlap with
active reviewed TRAIN/VAL cines, report matches with explicitly quarantined IDs,
and verify every reference's frame/cine voxel bridge.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
from self_audit.data import write_json
from self_audit.model import digest


def unique(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f'expected one input, got {len(paths)}')
    return paths[0]


def voxel_hash(raw):
    return hashlib.sha256(str(raw.shape).encode() + np.asarray(raw, dtype='<f4').tobytes()).hexdigest()


def prepare(root, comparison_root, split_path, protocol_path, out):
    root, comparison_root, out = root.resolve(), comparison_root.resolve(), out.resolve()
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out)
                           for p in (root, comparison_root)):
        raise ValueError('new separate TEST cache required')
    split = json.loads(split_path.read_text())
    ids = sorted(split['test'])
    if ids != [f'patient{i:03d}' for i in range(101, 151)]:
        raise ValueError('locked TEST50 IDs required')
    if set(ids) & set(split['train'] + split['val']):
        raise ValueError('TEST/TRAIN/VAL ID overlap')
    observed = sorted(p.name for p in root.glob('patient*') if p.is_dir())
    if observed != ids:
        raise ValueError('raw TEST directory differs from locked TEST50')
    protocol = json.loads(protocol_path.read_text())
    if set(protocol['checkpoints']) != {'baseline', 'fixed', 'meta'}:
        raise ValueError('all three final checkpoints must be pinned before TEST preparation')
    # Compare images only; no TRAIN/U/VAL mask arrays are read here.
    prior = {}
    active_ids = set(split['train'] + split['val'])
    quarantined_ids = set(split.get('excluded_patients', []))
    if active_ids & quarantined_ids or set(ids) & quarantined_ids:
        raise ValueError('overlapping active/quarantined split IDs')
    comparison_ids = sorted(p.name for p in comparison_root.glob('patient*') if p.is_dir())
    if set(comparison_ids) != active_ids | quarantined_ids:
        raise ValueError('comparison directory differs from reviewed active/quarantined IDs')
    for pid in comparison_ids:
        path = unique((comparison_root / pid).glob(f'{pid}_4d.nii*'))
        prior.setdefault(voxel_hash(np.asarray(nib.load(path).dataobj)), []).append(pid)
    out.mkdir(parents=True)
    records, sources, seen, quarantined_matches = [], [], set(), []
    for pid in ids:
        folder = root / pid
        cine = unique(folder.glob(f'{pid}_4d.nii*'))
        ni = nib.load(cine); raw = np.asarray(ni.dataobj)
        if raw.ndim != 4 or not np.isfinite(raw).all() or not np.isfinite(ni.affine).all():
            raise ValueError(f'{pid}: invalid image')
        vh = voxel_hash(raw)
        matches = prior.get(vh, [])
        if any(pid in active_ids for pid in matches) or vh in seen:
            raise ValueError(f'{pid}: exact duplicate cine content across patients/roles: {prior.get(vh)}')
        if matches:
            # These IDs were explicitly excluded in the locked split before
            # this TEST request. Keep all TEST patients; never silently drop one.
            quarantined_matches.append(dict(test_id=pid, excluded_raw_ids=matches))
        seen.add(vh)
        lo, hi = np.percentile(raw, [.5, 99.5])
        normalized = np.clip((raw-lo)/max(float(hi-lo), 1e-6), 0, 1).transpose(3, 2, 1, 0).astype(np.float32)
        ip = out / f'{pid}.npy'; np.save(ip, normalized)
        record = dict(id=pid, image=ip.name, image_sha256=digest(ip), masks={})
        source = dict(patient=pid, cine_sha256=digest(cine), cine_voxel_sha256=vh,
                      shape_xyzt=list(raw.shape), masks=[])
        # Explicitly authorized TEST references, isolated from the training API.
        mask_paths = sorted(folder.glob(f'{pid}_frame*_gt.nii*'))
        if len(mask_paths) != 2:
            raise ValueError(f'{pid}: expected two reference masks')
        for mp in mask_paths:
            frame_path = Path(str(mp).replace('_gt.nii', '.nii'))
            frame = nib.load(frame_path); mask = nib.load(mp)
            t = int(mp.name.split('_frame')[1].split('_gt')[0]) - 1
            if (mask.shape != frame.shape or mask.shape != raw.shape[:3]
                    or not np.allclose(mask.affine, frame.affine)
                    or not 0 <= t < raw.shape[3]
                    or not np.array_equal(np.asarray(frame.dataobj), raw[..., t])):
                raise ValueError(f'{pid}: reference/frame/cine geometry or voxel bridge mismatch')
            y = np.asarray(mask.dataobj)
            if not np.isin(y, [0, 1, 2, 3, 255]).all():
                raise ValueError(f'{pid}: unknown reference class')
            op = out / f'{pid}_t{t:03d}_mask.npy'; np.save(op, y.transpose(2, 1, 0).astype(np.uint8))
            record['masks'][str(t)] = dict(path=op.name, sha256=digest(op))
            source['masks'].append(dict(frame=t, source_sha256=digest(mp), frame_image_sha256=digest(frame_path)))
        records.append(record); sources.append(source)
        print(json.dumps(dict(patient=pid, status='verified', shape=source['shape_xyzt'])), flush=True)
    manifest = dict(schema='self_audit.data.v2', role='external_frozen', synthetic=False,
                    scope='Drive_TEST50; reference provenance not independently certified as official challenge GT',
                    patients=records)
    write_json(out / 'test.json', manifest)
    receipt = dict(status='PASS', patients=50, reference_frames=100,
                   normalization='per_cine_image_only_percentiles_0.5_99.5; full native grid',
                   test_mask_patient_ids_read=ids, train_u_val_mask_reads=0,
                   exact_cine_overlap_with_active_train_val=0,
                   quarantined_raw_cine_matches=quarantined_matches,
                   raw_train_val_cines_compared=sum(map(len, prior.values())),
                   source_split_sha256=digest(split_path), pinned_protocol_sha256=digest(protocol_path),
                   manifest_sha256=digest(out / 'test.json'), sources=sources,
                   warning='Drive copy; independent official label provenance and challenge-server certification unavailable')
    write_json(out / 'PREPARATION.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'comparison-root', 'split', 'protocol', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.root, a.comparison_root, a.split, a.protocol, a.out)
