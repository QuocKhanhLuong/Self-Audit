#!/usr/bin/env python3
"""Create image-only derived NIfTIs under an explicit, hash-pinned unit declaration.

Unknown units are never guessed by the canonical manifest reader. This separate
step records a dataset-level adjudication, patches only NIfTI-1 byte123 in a new
gzip copy, and checks unchanged payload bytes, affine and spacing. Raw data stay
untouched. Known-unit cines are hard-linked, not rewritten.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
from self_audit.model import digest


def open_image(path):
    return gzip.open(path, 'rb') if path.name.endswith('.gz') else path.open('rb')


def declared_copy(source, destination, expected_sha):
    if digest(source) != expected_sha or destination.exists():
        raise ValueError('source hash mismatch or existing derived output')
    image = nib.load(source)
    units = image.header.get_xyzt_units()[0]
    if units != 'unknown':
        raise ValueError('only an explicitly unknown spatial unit may be declared')
    with open_image(source) as inp, destination.open('wb') as raw_out:
        header = inp.read(348)
        if len(header) != 348 or header[344:348] != b'n+1\0' or header[123] & 7:
            raise ValueError('expected single-file NIfTI-1 with unknown spatial units')
        updated = bytearray(header); updated[123] = (updated[123] & 248) | 2
        before = hashlib.sha256()
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw_out, compresslevel=1, mtime=0) as out:
            out.write(updated)
            for block in iter(lambda: inp.read(2**20), b''):
                before.update(block); out.write(block)
    after = hashlib.sha256()
    with gzip.open(destination, 'rb') as inp:
        assert inp.read(348) == bytes(updated)
        for block in iter(lambda: inp.read(2**20), b''):after.update(block)
    if before.hexdigest() != after.hexdigest():
        raise ValueError('derived payload differs')
    derived = nib.load(destination)
    if (derived.shape != image.shape or not np.array_equal(derived.affine, image.affine)
            or derived.header.get_zooms() != image.header.get_zooms()
            or derived.header.get_xyzt_units()[0] != 'mm' or digest(source) != expected_sha):
        raise ValueError('derived geometry or raw source changed')
    return dict(source_sha256=expected_sha, derived_sha256=digest(destination),
                original_spatial_units=units, declared_spatial_units='mm',
                changed_uncompressed_header_bytes=[123], unchanged_payload_sha256=before.hexdigest())


def run(args):
    if digest(args.declaration) != args.declaration_sha256 or args.out.exists():
        raise ValueError('declaration hash mismatch or existing output')
    declaration = json.loads(args.declaration.read_text())
    if (declaration.get('schema') != 'self_audit.acdc_units_declaration.v1'
            or declaration.get('spatial_units_for_unknown') != 'mm'
            or not declaration.get('evidence_urls')):
        raise ValueError('explicit dataset adjudication and evidence required')
    train_path=Path(declaration['train_manifest'])
    if digest(train_path)!=declaration['train_manifest_sha256']:
        raise ValueError('TRAIN manifest hash mismatch')
    train=json.loads(train_path.read_text())
    ids=[r['id'] for r in train['labeled']+train['unlabeled']]
    declared_ids=[r['id'] for r in declaration['patients']]
    if (train.get('role')!='training' or train.get('synthetic') is not False
            or set(ids)!=set(declared_ids) or len(declared_ids)!=len(set(declared_ids))
            or any(not re.fullmatch(r'patient\d{3}',pid) for pid in declared_ids)):
        raise ValueError('exact TRAIN-only patient membership required')
    args.out.mkdir(parents=True)
    receipts = []
    for entry in declaration['patients']:
        pid = entry['id']; source = Path(entry['path'])
        if source.name not in (f'{pid}_4d.nii', f'{pid}_4d.nii.gz') or digest(source) != entry['sha256']:
            raise ValueError('only pinned exact image cines are eligible')
        folder = args.out / pid; folder.mkdir()
        image = nib.load(source); units = image.header.get_xyzt_units()[0]
        if units == 'unknown':
            destination = folder / f'{pid}_4d.nii.gz'
            rec = declared_copy(source, destination, entry['sha256'])
        elif units in ('mm', 'meter', 'micron'):
            destination = folder / source.name; os.link(source, destination)
            rec = dict(source_sha256=entry['sha256'], derived_sha256=digest(destination),
                       original_spatial_units=units, declared_spatial_units=units,
                       changed_uncompressed_header_bytes=[])
        else:
            raise ValueError(f'unsupported units {units}')
        receipts.append(dict(patient=pid, source=str(source), derived=str(destination), **rec))
        print(json.dumps(dict(patient=pid, units=units, status='VERIFIED')), flush=True)
    report = dict(status='PASS', declaration_sha256=args.declaration_sha256,
                  raw_files_modified=0, mask_reads=0, patients=receipts,
                  scientific_status='Explicit dataset-level mm adjudication; original missing-unit metadata retained in provenance')
    (args.out / 'UNIT_DECLARATION_RECEIPT.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--declaration', type=Path, required=True)
    p.add_argument('--declaration-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
