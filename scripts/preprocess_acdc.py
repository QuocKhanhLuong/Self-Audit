"""
Preprocess ACDC dataset: Cardiac MRI → .npy volumes
Classes: 0=BG, 1=RV, 2=MYO, 3=LV

Usage:
    python scripts/preprocess_acdc.py --input data/ACDC/training --output preprocessed_data/ACDC/training
"""

import os
import argparse
import configparser
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
from skimage.transform import resize


def normalize_zscore(image):
    """Z-score normalization with outlier clipping."""
    p05 = np.percentile(image, 0.5)
    p995 = np.percentile(image, 99.5)
    image = np.clip(image, p05, p995)
    mean, std = np.mean(image), np.std(image)
    return (image - mean) / std if std > 0 else image - mean


def validate_image_mask_geometry(img_nii, mask_nii, case_name: str = "") -> None:
    """Verify that image and mask NIfTI objects share identical shape and affine."""
    if img_nii.shape != mask_nii.shape:
        raise ValueError(
            f"Image/mask shape mismatch for {case_name}: {img_nii.shape} vs {mask_nii.shape}"
        )
    if not np.allclose(img_nii.affine, mask_nii.affine, atol=1e-3):
        raise ValueError(
            f"Image/mask affine mismatch for {case_name}: image affine != mask affine"
        )


def preprocess_patient(patient_path, target_size=(224, 224)):
    """Process one ACDC patient: load ED and ES frames with spacing and geometry info."""
    patient_folder = os.path.basename(patient_path)
    info_cfg_path = os.path.join(patient_path, 'Info.cfg')

    if not os.path.exists(info_cfg_path):
        return []

    try:
        parser = configparser.ConfigParser()
        with open(info_cfg_path, 'r') as f:
            parser.read_string('[DEFAULT]\n' + f.read())
        ed_frame = int(parser['DEFAULT']['ED'])
        es_frame = int(parser['DEFAULT']['ES'])
    except Exception as e:
        print(f"  Error: {patient_folder}: {e}")
        return []

    results = []

    for frame_num, frame_name in [(ed_frame, 'ED'), (es_frame, 'ES')]:
        img_filename = f'{patient_folder}_frame{frame_num:02d}.nii.gz'
        mask_filename = f'{patient_folder}_frame{frame_num:02d}_gt.nii.gz'

        img_path = os.path.join(patient_path, img_filename)
        mask_path = os.path.join(patient_path, mask_filename)

        if not os.path.exists(img_path):
            img_path = img_path.replace('.nii.gz', '.nii')
            mask_path = mask_path.replace('.nii.gz', '.nii')

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        validate_image_mask_geometry(img_nii, mask_nii, f"{patient_folder} frame {frame_num:02d} ({frame_name})")

        try:
            img_data = normalize_zscore(img_nii.get_fdata())
            mask_data = mask_nii.get_fdata()

            orig_shape = img_data.shape
            orig_spacing = img_nii.header.get_zooms()  # (sx, sy, sz) in mm
            num_slices = img_data.shape[2]

            eff_spacing_y = float(orig_spacing[0]) * orig_shape[0] / target_size[0]
            eff_spacing_x = float(orig_spacing[1]) * orig_shape[1] / target_size[1]
            eff_spacing_z = float(orig_spacing[2])

            orig_affine = [list(float(v) for v in row) for row in img_nii.affine]
            try:
                orientation = list(str(c) for c in nib.aff2axcodes(img_nii.affine))
            except Exception:
                orientation = None

            resize_chain = [
                {
                    "step": "in_plane_resize",
                    "input_shape": [int(orig_shape[0]), int(orig_shape[1])],
                    "output_shape": [int(target_size[0]), int(target_size[1])],
                    "input_axis_order": "HW",
                    "output_axis_order": "HW",
                    "mode_image": "bilinear",
                    "mode_mask": "nearest",
                    "anti_aliasing_image": True,
                    "anti_aliasing_mask": False,
                }
            ]

            resized_img = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
            resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)

            for i in range(num_slices):
                resized_img[:, :, i] = resize(img_data[:, :, i], target_size, order=1, preserve_range=True, anti_aliasing=True, mode='reflect')
                resized_mask[:, :, i] = resize(mask_data[:, :, i], target_size, order=0, preserve_range=True, anti_aliasing=False, mode='reflect').astype(np.uint8)

            spacing_info = {
                'schema_version': 2,
                'orig_shape': [int(s) for s in orig_shape],
                'orig_shape_order': 'HWZ',
                'orig_spacing': [float(s) for s in orig_spacing],
                'orig_spacing_order': 'XYZ',
                'orig_affine': orig_affine,
                'orig_orientation': orientation,
                'effective_spacing': [eff_spacing_z, eff_spacing_y, eff_spacing_x],
                'effective_spacing_order': 'ZYX',
                'resize_chain': resize_chain,
            }

            results.append((resized_img, resized_mask, f"{patient_folder}_{frame_name}", spacing_info))
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            print(f"  Error: {patient_folder} frame {frame_num}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Preprocess ACDC dataset')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--no-skip', action='store_true')
    args = parser.parse_args()

    target_size = (args.size, args.size)
    os.makedirs(args.output, exist_ok=True)
    volumes_dir = os.path.join(args.output, 'volumes')
    masks_dir = os.path.join(args.output, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    metadata_path = os.path.join(args.output, 'metadata.json')

    existing_metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
        except Exception as exc:
            raise ValueError(f"Existing metadata in {metadata_path} is malformed: {exc}") from exc
        if not isinstance(existing_metadata, dict):
            raise ValueError(f"Existing metadata in {metadata_path} must be a JSON object, got {type(existing_metadata)}")

    existing_volume_info = existing_metadata.get('volume_info', {}) if isinstance(existing_metadata, dict) else {}
    if not isinstance(existing_volume_info, dict):
        raise ValueError(f"Existing metadata volume_info in {metadata_path} must be a dictionary")
    existing_target_size = existing_metadata.get('target_size') if isinstance(existing_metadata, dict) else None

    # Global preflight validation: reject target size changes when existing output directory contains retained arrays
    if existing_target_size is not None and list(existing_target_size) != list(target_size):
        if existing_volume_info or (os.path.exists(volumes_dir) and any(os.scandir(volumes_dir))):
            raise ValueError(
                f"Existing output directory declares target_size {existing_target_size}, "
                f"which differs from requested size {target_size}. "
                "Refusing to mix conflicting geometries or rewrite target_size for retained old arrays."
            )

    # Validate all existing retained pairs and their grid shapes on disk BEFORE any writes
    for retained_id, r_info in existing_volume_info.items():
        r_vol = os.path.join(volumes_dir, f"{retained_id}.npy")
        r_mask = os.path.join(masks_dir, f"{retained_id}.npy")
        if not os.path.exists(r_vol) or not os.path.exists(r_mask):
            raise ValueError(
                f"Retained metadata entry {retained_id} has missing array file on disk (vol={os.path.exists(r_vol)}, mask={os.path.exists(r_mask)}). "
                "Safe merging is ambiguous."
            )
        try:
            r_vol_arr = np.load(r_vol, mmap_mode='r')
            r_mask_arr = np.load(r_mask, mmap_mode='r')
            expected_size = tuple(existing_target_size) if existing_target_size is not None else target_size
            if tuple(r_vol_arr.shape[:2]) != expected_size:
                raise ValueError(
                    f"Retained volume {r_vol} shape {r_vol_arr.shape[:2]} does not match expected size {expected_size}."
                )
            if tuple(r_mask_arr.shape[:2]) != expected_size:
                raise ValueError(
                    f"Retained mask {r_mask} shape {r_mask_arr.shape[:2]} does not match expected size {expected_size}."
                )
            if r_vol_arr.shape != r_mask_arr.shape:
                raise ValueError(
                    f"Shape mismatch between retained volume and mask on disk for {retained_id}: "
                    f"{r_vol_arr.shape} vs {r_mask_arr.shape}."
                )
        except Exception as err:
            if isinstance(err, ValueError):
                raise
            raise ValueError(f"Could not verify retained array geometry for {retained_id}: {err}") from err

    patient_folders = sorted([
        os.path.join(args.input, d) for d in os.listdir(args.input)
        if os.path.isdir(os.path.join(args.input, d)) and d.startswith('patient')
    ])

    print(f"ACDC Preprocessing: {len(patient_folders)} patients")

    merged_volume_info = dict(existing_volume_info)
    processed, skipped = 0, 0

    for patient_path in tqdm(patient_folders, desc="ACDC"):
        for volume, mask, volume_id, spacing_info in preprocess_patient(patient_path, target_size):
            vol_path = os.path.join(volumes_dir, f'{volume_id}.npy')
            mask_path = os.path.join(masks_dir, f'{volume_id}.npy')
            vol_exists = os.path.exists(vol_path)
            mask_exists = os.path.exists(mask_path)

            if not args.no_skip:
                if vol_exists != mask_exists:
                    raise ValueError(
                        f"Partial pair detected for {volume_id}: volume exists={vol_exists}, "
                        f"mask exists={mask_exists}. Refusing to silently overwrite without --no-skip."
                    )
                if vol_exists and mask_exists:
                    try:
                        vol_on_disk = np.load(vol_path, mmap_mode='r')
                        mask_on_disk = np.load(mask_path, mmap_mode='r')
                        if tuple(vol_on_disk.shape[:2]) != target_size:
                            raise ValueError(
                                f"Target size mismatch for existing volume {vol_path}: "
                                f"file shape is {vol_on_disk.shape[:2]}, requested {target_size}."
                            )
                        if tuple(mask_on_disk.shape[:2]) != target_size:
                            raise ValueError(
                                f"Target size mismatch for existing mask {mask_path}: "
                                f"file shape is {mask_on_disk.shape[:2]}, requested {target_size}."
                            )
                        if vol_on_disk.shape != mask_on_disk.shape:
                            raise ValueError(
                                f"Shape mismatch between existing volume and mask on disk for {volume_id}: "
                                f"{vol_on_disk.shape} vs {mask_on_disk.shape}."
                            )
                    except Exception as err:
                        if isinstance(err, ValueError):
                            raise
                        raise ValueError(f"Could not verify existing volume/mask for {volume_id}: {err}") from err

                    if volume_id in existing_volume_info:
                        # Preserve historical metadata untouched; do NOT stamp fresh provenance
                        merged_volume_info[volume_id] = existing_volume_info[volume_id]
                        skipped += 1
                        continue
                    else:
                        raise ValueError(
                            f"Existing output file {vol_path} has no record in existing metadata.json; "
                            "cannot verify provenance without recomputing. Use --no-skip to reprocess."
                        )

            np.save(vol_path, volume)
            np.save(mask_path, mask)
            merged_volume_info[volume_id] = {
                'num_slices': int(mask.shape[2]),
                **spacing_info,
            }
            processed += 1

    if processed == 0 and skipped > 0 and os.path.exists(metadata_path):
        print(f"Done! All {skipped} volumes already exist with verified metadata. Existing metadata preserved untouched.")
        return

    metadata = {
        'schema_version': 2,
        'dataset': 'ACDC',
        'num_classes': 4,
        'class_names': ['Background', 'RV', 'MYO', 'LV'],
        'target_size': list(target_size),
        'total_volumes': len(merged_volume_info),
        'volume_info': merged_volume_info,
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Done! Processed: {processed}, Skipped: {skipped}")


if __name__ == '__main__':
    main()
