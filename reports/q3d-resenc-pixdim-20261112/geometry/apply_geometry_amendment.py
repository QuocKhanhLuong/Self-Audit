"""Apply the user-approved geometry-only amendment to unchanged restored views."""
import copy
import hashlib
import json
from pathlib import Path

from self_audit.candidate_q3d.geometry_repair import repair_manifest_geometry

ROOT = Path('/root/mount_data/prepared/ACDC')
OUT = Path('/root/mount_data/receipts/q3d-pixdim-20261112')
EXPECTED = {
    'k5/train.json': '08e52386ca3c4194bb13cc3b70db38783c01c97f56ef62276bea9637139e7a4e',
    'k8/train.json': '175e99e455f35f61453dcb6711b32d6682571ec082374740f9dbfa44576a5944',
    'k10/train.json': '4982655478334405a538c5ced3fe5093c4305c0c9ba7c758a6f54e20247a1cb5',
    'k5/development.json': 'ba46934afe8ea5c47dd798019bee233c97c98320b358e2d40165e87a651dc7df',
    'k8/development.json': 'ba46934afe8ea5c47dd798019bee233c97c98320b358e2d40165e87a651dc7df',
    'k10/development.json': 'ba46934afe8ea5c47dd798019bee233c97c98320b358e2d40165e87a651dc7df',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    mapping = json.loads((OUT / 'cine-mapping.json').read_text())
    declarations = json.loads((OUT / 'unit-declarations.json').read_text())
    results = []
    for relative, source_sha in EXPECTED.items():
        source = ROOT / relative
        assert sha(source) == source_sha, f'source changed: {relative}'
        original = json.loads(source.read_text())
        ids = list(original['geometry']['by_patient'])
        stem = relative.replace('/', '-').removesuffix('.json')
        map_path = OUT / (stem + '.cine-map.json')
        new_json(map_path, {pid: mapping[pid] for pid in ids})
        output = source.with_name(source.stem + '.q3d-pixdim.json')
        report = repair_manifest_geometry(
            source, source_sha, {pid: mapping[pid] for pid in ids},
            axis_map='XYZ->ZYX', cine_mapping_source=str(map_path),
            cine_mapping_sha256=sha(map_path),
            cine_mapping_provenance='Exact patient ID and original CINE filename in restored ACDC raw training/testing; original Drive byte integrity audited separately; no voxel bridge is claimed here.',
            report_json_path=OUT / (stem + '.repair.json'),
            report_csv_path=OUT / (stem + '.spacing.csv'),
            output_manifest_path=output,
            unit_declarations_by_patient={pid: declarations[pid] for pid in ids if pid in declarations},
            audit_only=False, geometry_policy='acdc_pixdim_native_v1',
        )
        assert report['status'] == 'success', report['errors']
        repaired = json.loads(output.read_text())
        assert {k:v for k,v in original.items() if k != 'geometry'} == {k:v for k,v in repaired.items() if k != 'geometry'}
        results.append({'source': str(source), 'source_sha256': source_sha, 'output': str(output), 'output_sha256': sha(output), 'patient_count': len(ids), 'report': str(OUT / (stem + '.repair.json')), 'report_sha256': sha(OUT / (stem + '.repair.json')), 'non_geometry_unchanged': True})
        print(json.dumps(results[-1]), flush=True)
    # TRAIN80 is the full existing TRAIN cohort; this alias preserves its K10 label visibility.
    reference = ROOT / 'k10/train.q3d-pixdim.json'
    train80 = ROOT / 'k10/train80.q3d-pixdim.json'
    with train80.open('xb') as stream:
        stream.write(reference.read_bytes())
    new_json(OUT / 'MANIFEST_AMENDMENT.json', {
        'status': 'PASS', 'geometry_policy': 'acdc_pixdim_native_v1', 'manifests': results,
        'train80_reference': {'path': str(train80), 'sha256': sha(train80), 'same_bytes_as': str(reference), 'label_visibility': 'K10; this is a TRAIN80 cohort reference, not 80 labeled patients'},
        'split_selector_voxels_labels': 'UNCHANGED_FROM_RESTORED_SOURCE',
        'prior_restoration_changes': 'See separate RESTORATION_LINEAGE.json; this amendment does not attribute pre-existing path/order/metadata changes to geometry.',
        'a3d_caveat': 'Historical A-3D k8 runs used the old manifest geometry; SDF/mm-dependent results may be affected. No baseline rerun.',
    })


if __name__ == '__main__':
    main()
