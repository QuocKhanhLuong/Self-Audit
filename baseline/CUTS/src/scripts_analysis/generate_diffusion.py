import argparse
import json
import os
import sys
import warnings
from glob import glob
from typing import Tuple

import numpy as np
import yaml
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.diffusion_condensation import diffusion_condensation_catch, diffusion_condensation_msphate
from utils.parse import parse_settings
from utils.artifact_contract import file_sha256, json_scalar, validate_hierarchy

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1


def generate_diffusion(
        shape: Tuple[int],
        latent: np.array,
        knn: int = 100,
        num_workers: int = 1,
        random_seed: int = 0,
        use_msphate: bool = True) -> Tuple[float, np.array, np.array]:

    H, W, C = shape
    assert latent.shape == (H * W, C)

    if use_msphate:
        labels_pred, granularities = diffusion_condensation_msphate(
            X=latent, knn=knn, num_workers=num_workers, random_seed=random_seed)
    else:
        labels_pred, granularities = diffusion_condensation_catch(
            X=latent, knn=knn, num_workers=num_workers, random_seed=random_seed)

    return labels_pred, granularities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('-o',
                        '--overwrite',
                        action='store_true',
                        help='If true, overwrite previously computed results.')
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)

    files_folder = '%s/%s' % (config.output_save_path, 'numpy_files')
    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))
    if not np_files_path:
        raise ValueError('No latent NPZ inputs found')

    save_path_numpy = '%s/%s' % (config.output_save_path,
                                 'numpy_files_seg_diffusion')
    os.makedirs(save_path_numpy, exist_ok=True)

    for image_idx in tqdm(range(len(np_files_path))):
        save_path = '%s/%s' % (save_path_numpy,
                               os.path.basename(np_files_path[image_idx]))

        diffusion_seed = getattr(config, 'diffusion_random_seed', 0)
        provenance = {
            'source_sha256': file_sha256(np_files_path[image_idx]),
            'config_sha256': file_sha256(args.config),
            'diffusion_random_seed': diffusion_seed,
            'knn': 100, 'backend': 'multiscale_phate',
        }
        extra = {}
        with np.load(np_files_path[image_idx], allow_pickle=False) as numpy_array:
            image = numpy_array['image']
            recon = numpy_array['recon']
            latent = numpy_array['latent']
            if not getattr(config, 'image_only', False) and 'label' in numpy_array:
                extra['label'] = numpy_array['label']
            for key in ('sample_metadata', 'provenance'):
                if key in numpy_array:
                    extra[key] = numpy_array[key]
            if getattr(config, 'image_only', False):
                metadata = json.loads(str(extra.get('sample_metadata', '{}')))
                if not metadata.get('image_only') or 'provenance' not in extra:
                    raise ValueError('GT-free diffusion requires image-only latent provenance')

        image = (image + 1) / 2
        recon = (recon + 1) / 2

        H, W = image.shape[:2]
        C = latent.shape[-1]

        if os.path.exists(save_path) and not args.overwrite:
            with np.load(save_path, allow_pickle=False) as previous:
                if ('diffusion_provenance' not in previous or
                        json.loads(str(previous['diffusion_provenance'])) != provenance):
                    raise ValueError('Existing diffusion output has different or missing provenance')
                validate_hierarchy(previous['labels_diffusion'], previous['granularities_diffusion'], H * W)
            continue

        labels_pred, granularities = generate_diffusion(
            (H, W, C), latent, num_workers=1, random_seed=diffusion_seed)
        validate_hierarchy(labels_pred, granularities, H * W)

        with open(save_path, 'wb' if args.overwrite else 'xb') as f:
            np.savez(
                f,
                image=image,
                recon=recon,
                latent=latent,
                labels_diffusion=labels_pred,
                granularities_diffusion=granularities,
                diffusion_provenance=json_scalar(provenance),
                **extra,
            )

    print('All diffusion results generated.')

    # Somehow the code may hang at this point...
    # Force exit.
    os._exit(0)
