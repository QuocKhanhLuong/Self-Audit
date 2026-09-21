"""Preregistered dimensionless seed falsification (new v2 before any GT evaluation); image-only, no C0/C1 labels/weights.

This intentionally simple diagnostic tests whether explicit named image anchors
are precise enough to justify a learner. It is NOT a trained v3 teacher.
All constants and all arms are frozen before the separate evaluator opens GT.
"""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
import scipy
import skimage

CONFIG = {
    'experiment_id': 'astra-v3-r0-dimensionless-seeds-20260920-v2',
    'seed': 17, 'patients_per_split': 4,
    'selection': 'lowest sha256(astra-v3-r0- + patient_id) independently in original train/dev',
    'normalization_percentiles': [1, 99], 'gaussian_fov_fraction': 0.004,
    'bright_quantile': 70, 'bright_floor': 0.45,
    'lv_radius_fov_fraction': [0.02, 0.12], 'lv_max_eccentricity': 0.85,
    'lv_min_solidity': 0.75, 'lv_min_contrast': 0.08,
    'lv_max_normalized_distance_from_fov_center': 0.35,
    'winner_ratio': 1.2, 'seed_erosion_fov_fraction': 0.006,
    'myo_annulus_fov_fraction': [0.006, 0.018], 'myo_intensity_floor': 0.10,
    'rv_radius_fov_fraction': [0.015, 0.12], 'rv_max_center_distance_fov_fraction': 0.23,
    'orientation': 'NOT RUN: local sform disagrees with header spacing and units unknown',
    'topology_min_dark_annulus_fraction': 0.70,
    'boundary_min_contrast': 0.16,
    'cross_slice_tolerance_fov_fraction': 0.012, 'augmentation_gamma': [0.9, 1.1],
    'arms': ['appearance_anchor', 'plus_topology', 'plus_boundary',
             'plus_cross_slice', 'plus_augmentation', 'all_available', 'unknown_control'],
    'unavailable_cues': ['motion', 'full_cycle_temporal_consistency', 'native_orientation'],
    'primary_dice': 'mean_patient(mean_phase(mean_RV_MYO_LV(native_3D_Dice))); UNKNOWN counts FN',
    'gate': {'per_class_precision_min': 0.95, 'per_class_gt_coverage_min': 0.05,
             'patient_fraction_with_all_three_classes_min': 0.75},
    'selection_after_evaluation': 'none; one frozen all_available candidate, other arms diagnostic',
    'geometry_policy': 'dimensionless short-axis stored-grid FOV fractions; no mm or trustworthy native orientation claim; z is stored axis 2',
    'historical_provenance': 'ED/ES local copy originally selected by paired GT presence; development informed; not pristine no-GT acquisition',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_volume(volume, affine, spacing, gamma=1.0):
    image = np.asarray(volume, dtype=np.float32)
    lo, hi = np.percentile(image[np.isfinite(image)], CONFIG['normalization_percentiles'])
    image = np.clip((image - lo) / max(float(hi - lo), 1e-6), 0, 1) ** gamma
    shape = image.shape
    base = np.full(shape, 255, np.uint8)
    oriented = base.copy()
    topological = base.copy()
    boundary = base.copy()
    details = []
    for z in range(shape[2]):
        x = ndi.gaussian_filter(image[:, :, z], sigma=np.asarray([CONFIG['gaussian_fov_fraction']]*2)/spacing[:2])
        # Dark border-connected image background only. No anatomy is used to crop.
        dark = x < 0.03
        lab, _ = ndi.label(dark)
        edge_ids = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        bg = np.isin(lab, edge_ids[edge_ids != 0])
        for a in (base, oriented, topological, boundary):
            a[:, :, z][bg] = 0
        positive = x[x > .05]
        if not positive.size:
            details.append({'z': z, 'reason': 'no_positive_signal'})
            continue
        threshold = max(CONFIG['bright_floor'], float(np.percentile(positive, CONFIG['bright_quantile'])))
        lab, _ = ndi.label(x >= threshold)
        candidates = []
        props = regionprops(lab)
        for prop in props:
            radius = np.sqrt(prop.area * np.prod(spacing[:2]) / np.pi)
            center = np.asarray(prop.centroid)
            normalized_distance = np.linalg.norm((center / np.asarray(shape[:2])) - .5)
            if not (CONFIG['lv_radius_fov_fraction'][0] <= radius <= CONFIG['lv_radius_fov_fraction'][1]):
                continue
            if prop.eccentricity > CONFIG['lv_max_eccentricity'] or prop.solidity < CONFIG['lv_min_solidity']:
                continue
            if normalized_distance > CONFIG['lv_max_normalized_distance_from_fov_center']:
                continue
            component = lab == prop.label
            outside = ndi.distance_transform_edt(~component, sampling=spacing[:2])
            ring = (outside >= CONFIG['myo_annulus_fov_fraction'][0]) & (outside <= CONFIG['myo_annulus_fov_fraction'][1])
            if not ring.any():
                continue
            mean_inner = float(x[component].mean())
            contrast = mean_inner - float(x[ring].mean())
            if contrast < CONFIG['lv_min_contrast']:
                continue
            score = contrast * prop.solidity * (1 - .5 * prop.eccentricity)
            candidates.append((score, prop.label, component, ring, center, mean_inner, contrast))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if not candidates or (len(candidates) > 1 and candidates[0][0] < CONFIG['winner_ratio']*candidates[1][0]):
            details.append({'z': z, 'reason': 'no_unique_compact_bright_pool'})
            continue
        score, label, lv, ring, center, mean_inner, contrast = candidates[0]
        lv_seed = ndi.distance_transform_edt(lv, sampling=spacing[:2]) >= CONFIG['seed_erosion_fov_fraction']
        myo_seed = ring & (x > CONFIG['myo_intensity_floor']) & (x < mean_inner - CONFIG['lv_min_contrast'])
        topo_ok = float(myo_seed.sum() / max(int(ring.sum()), 1)) >= CONFIG['topology_min_dark_annulus_fraction']
        boundary_ok = contrast >= CONFIG['boundary_min_contrast']
        rv_candidates = []
        for prop in props:
            radius = np.sqrt(prop.area*np.prod(spacing[:2])/np.pi)
            distance = float(np.linalg.norm((np.asarray(prop.centroid)-center)*spacing[:2]))
            if prop.label == label or not CONFIG['rv_radius_fov_fraction'][0] <= radius <= CONFIG['rv_radius_fov_fraction'][1]:
                continue
            if not 0.02 <= distance <= CONFIG['rv_max_center_distance_fov_fraction']:
                continue
            rv_candidates.append((distance, prop.label, np.asarray(prop.centroid)))
        rv_candidates.sort(key=lambda item: (item[0], item[1]))
        rv_seed = np.zeros_like(lv)
        orientation_ok = False
        if rv_candidates:
            _, rv_label, rv_center = rv_candidates[0]
            rv_seed = ndi.distance_transform_edt(lab == rv_label, sampling=spacing[:2]) >= CONFIG['seed_erosion_fov_fraction']
            lv_world = nib.affines.apply_affine(affine, [*center, z])
            rv_world = nib.affines.apply_affine(affine, [*rv_center, z])
            # NIfTI world coordinates use RAS: increasing x is patient-right.
            orientation_ok = False  # not evaluated: native orientation provenance is insufficient
        for a, accept in [(base, True), (oriented, orientation_ok), (topological, topo_ok), (boundary, boundary_ok)]:
            if accept:
                a[:, :, z][lv_seed] = 3
                a[:, :, z][myo_seed] = 2
                a[:, :, z][rv_seed] = 1
        details.append({'z': z, 'contrast': contrast, 'orientation_ok': bool(orientation_ok),
                        'topology_ok': bool(topo_ok), 'boundary_ok': bool(boundary_ok)})
    return base, oriented, topological, boundary, details


def cross_slice(seeds, spacing):
    out = seeds.copy()
    for z in range(seeds.shape[2]):
        neighbors = [j for j in [z-1, z+1] if 0 <= j < seeds.shape[2]]
        for c in (1, 2, 3):
            supports = []
            for j in neighbors:
                present = seeds[:, :, j] == c
                support = ndi.distance_transform_edt(~present, sampling=spacing[:2]) <= CONFIG['cross_slice_tolerance_fov_fraction'] if present.any() else present
                supports.append(support)
            stable = np.logical_and.reduce(supports) if supports else np.zeros(seeds.shape[:2], bool)
            out[:, :, z][(seeds[:, :, z] == c) & ~stable] = 255
    return out


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(exist_ok=False)
    (output/'images').mkdir()
    (output/'predictions').mkdir()
    script = Path(__file__).resolve()
    raw = Path('/tmp/astra_event_acdc_training/training')
    split_path = Path('/tmp/astra_event_acdc_training/acdc_patient_split_seed42.json')
    split = json.loads(split_path.read_text())
    assert not set(split['train_patients']) & set(split['val_patients'])
    observed_opens = []

    def audit_open(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(args[0]).resolve()
        if raw == path or raw in path.parents or output/'images' in path.parents:
            if '_gt' in path.name.lower() or path.name.lower() == 'info.cfg':
                raise PermissionError('GT or annotation sidecar access forbidden in seed generator')
            observed_opens.append(str(path))

    sys.addaudithook(audit_open)
    cohort = []
    for role, key in [('train', 'train_patients'), ('dev', 'val_patients')]:
        selected = sorted(split[key], key=lambda x: hashlib.sha256(('astra-v3-r0-'+x).encode()).hexdigest())[:CONFIG['patients_per_split']]
        cohort.extend((patient, role) for patient in selected)
    (output/'config.json').write_text(json.dumps(CONFIG, indent=2)+'\n')
    records, reads = [], []
    for patient, role in cohort:
        images = sorted(p for p in (raw/patient).iterdir() if p.name.endswith(('.nii', '.nii.gz')) and '_gt' not in p.name.lower())
        if len(images) != 2:
            raise ValueError('Expected this diagnostic cohort to have exactly two ED/ES exports; full cine is unsupported')
        for source in images:
            # Explicit image-only allowlist, copied before loading. No GT existence query.
            assert '_gt' not in source.name.lower()
            target = output/'images'/source.name
            shutil.copyfile(source, target)
            nii = nib.load(target)
            if len(nii.shape) != 3 or abs(np.linalg.det(nii.affine[:3, :3])) < 1e-8:
                raise ValueError('invalid native geometry')
            header_spacing = np.asarray(nii.header.get_zooms()[:3])
            spacing = np.ones(3) / min(nii.shape[:2])  # explicit dimensionless grid contract
            vol = np.asarray(nii.dataobj, dtype=np.float32)
            if not np.isfinite(vol).all():
                raise ValueError('nonfinite image')
            base, orient, topo, bound, detail = seed_volume(vol, nii.affine, spacing)
            zstable = cross_slice(base, spacing)
            aug = base.copy()
            for gamma in CONFIG['augmentation_gamma']:
                augmented = seed_volume(vol, nii.affine, spacing, gamma=gamma)[0]
                aug[base != augmented] = 255
            combined = base.copy()
            for a in [topo, bound, zstable, aug]:
                combined[a != base] = 255
            arms = dict(zip(CONFIG['arms'], [base, topo, bound, zstable, aug, combined, np.full_like(base, 255)]))
            prediction = output/'predictions'/(source.name.replace('.nii.gz','').replace('.nii','')+'.npz')
            np.savez_compressed(prediction, **arms, affine=nii.affine)
            record = {'patient_id': patient, 'split': role, 'source_image': str(source),
                      'image_sha256': sha(target), 'shape': list(nii.shape), 'header_spacing_unverified_units': header_spacing.tolist(), 'units': list(nii.header.get_xyzt_units()), 'model_grid_spacing': spacing.tolist(),
                      'affine': nii.affine.tolist(), 'orientation': list(nib.aff2axcodes(nii.affine)),
                      'prediction': str(prediction), 'prediction_sha256': sha(prediction),
                      'slice_diagnostics': detail,
                      'coverage': {name: float((a != 255).mean()) for name, a in arms.items()}}
            records.append(record)
            reads.append({'path': str(source), 'purpose': 'image_copy', 'sha256': record['image_sha256']})
            print(patient,role,source.name,'all_available coverage',record['coverage']['all_available'],flush=True)
    receipt = {'schema': 'astra-v3-r0-frozen-v1', 'experiment_id': CONFIG['experiment_id'],
               'frozen_utc': datetime.now(timezone.utc).isoformat(), 'base_commit': 'd4f503b7107eca4e8ac85f37a5dd01ec50479666',
               'config_path': str(output/'config.json'), 'config_sha256': sha(output/'config.json'),
               'script_path': str(script), 'script_sha256': sha(script),
               'split_path': str(split_path), 'split_sha256': sha(split_path), 'records': records,
               'source_reads': reads, 'gt_opened_by_generator': False,
               'observed_data_opens': sorted(set(observed_opens)),
               'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'skimage': skimage.__version__, 'nibabel': nib.__version__},
               'cue_availability': {'motion': 'NOT RUN: no full cine', 'full_cycle': 'NOT RUN: no full cine', 'orientation': 'NOT RUN: insufficient native geometry provenance'},
               'semantic_source': 'explicit bright compact pool, surrounding darker annulus, adjacent bright pool; dimensionless image priors; not learned or verified'}
    (output/'FROZEN.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('FROZEN',output/'FROZEN.json',sha(output/'FROZEN.json'))


if __name__ == '__main__':
    main()
