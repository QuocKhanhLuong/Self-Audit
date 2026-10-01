"""ACDC loader in the same style as the official `brain_tumor_nifti.py`
(the closest official CUTS template: raw NIfTI in, cv2 cubic/nearest resize +
crop_or_pad, min-max rescale to [-1,1]). Deliberately independent of
`shared_benchmark`/`cardiac_benchmark`: no shared manifest, no shared grid
contract, no patient-disjoint split -- this is the "protocol like the
original" counterpart to `cardiac_benchmark.dataset.ImageOnlyCardiacDataset`,
meant to be run side by side with it for comparison (see
reports/cuts_acdc_original_style_protocol_proposal_20260927.md).

Two necessary, documented deviations from a literal copy of
`brain_tumor_nifti.py` (both because ACDC differs structurally from the
original datasets, not stylistic choices):
  1. Each ACDC NIfTI is a 3D volume [H,W,Z], not a pre-extracted single 2D
     slice like `brain_tumor_nifti.py` assumes -- so `__init__` enumerates
     (patient, frame, z) slices instead of one file per sample.
  2. Labels are kept as native ACDC integers 0-3 (BG/RV/MYO/LV), not forced
     to binary via `label != np.unique(label)[0]` -- every official CUTS
     dataset is binary-only; this is the minimal necessary multi-class
     extension, not a paper-derived choice.
"""
from glob import glob
from pathlib import Path
from typing import Tuple

import cv2
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


def _parse_ed_es(info_cfg_path: Path) -> Tuple[int, int]:
    ed = es = None
    for line in info_cfg_path.read_text().splitlines():
        if line.startswith('ED:'):
            ed = int(line.split(':')[1].strip())
        elif line.startswith('ES:'):
            es = int(line.split(':')[1].strip())
    if ed is None or es is None:
        raise ValueError(f'{info_cfg_path} missing ED/ES')
    return ed, es


def crop_or_pad(in_image: np.array,
                in_shape: Tuple[int],
                out_shape: Tuple[int],
                pad_value: float = 0) -> np.array:
    """Verbatim copy of the official `crop_or_pad` (brain_tumor(_nifti).py)."""
    assert len(in_shape) == len(out_shape)
    D = len(in_shape)

    out_shape_min = [
        int(np.floor((out_shape[i] - in_shape[i]) /
                     2)) if out_shape[i] >= in_shape[i] else 0
        for i in range(D)
    ]
    out_shape_max = [
        out_shape_min[i] + in_shape[i] if out_shape[i] >= in_shape[i] else None
        for i in range(D)
    ]

    in_shape_min = [
        0 if out_shape[i] >= in_shape[i] else int(
            np.floor((in_shape[i] - out_shape[i]) / 2)) for i in range(D)
    ]
    in_shape_max = [
        None if out_shape[i] >= in_shape[i] else in_shape_min[i] + out_shape[i]
        for i in range(D)
    ]

    in_slicer = tuple(
        [slice(i, j) for (i, j) in zip(in_shape_min, in_shape_max)])
    out_slicer = tuple(
        [slice(i, j) for (i, j) in zip(out_shape_min, out_shape_max)])

    out_image = np.ones(out_shape) * pad_value
    out_image[out_slicer] = in_image[in_slicer]

    return out_image


class ACDCOriginalStyle(Dataset):
    """CUTS-original-style ACDC loader: no shared_benchmark, no manifest, no
    patient-disjoint split. Enumerates every (patient, ED/ES frame, z-slice)
    directly from the native `data/ACDC/training/` layout, exactly the way
    each official CUTS dataset parses its own raw file layout independently.
    """

    def __init__(self,
                 base_path: str = 'data/ACDC/training/',
                 out_shape: Tuple[int, int] = (224, 224)):
        self.base_path = base_path
        self.out_shape = out_shape
        self.samples: list[Tuple[str, int, int]] = []  # (patient_dir, frame, z)

        for patient_dir in sorted(glob('%s/patient*' % base_path)):
            patient_dir = Path(patient_dir)
            info_cfg = patient_dir / 'Info.cfg'
            if not info_cfg.exists():
                continue
            patient_id = patient_dir.name
            for frame in sorted(set(_parse_ed_es(info_cfg))):
                gt_path = patient_dir / f'{patient_id}_frame{frame:02d}_gt.nii'
                if not gt_path.exists():
                    continue
                depth = nib.load(str(gt_path)).shape[2]
                for z in range(depth):
                    self.samples.append((str(patient_dir), frame, z))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        patient_dir, frame, z = self.samples[idx]
        patient_id = Path(patient_dir).name
        image_path = f'{patient_dir}/{patient_id}_frame{frame:02d}.nii'
        label_path = f'{patient_dir}/{patient_id}_frame{frame:02d}_gt.nii'

        image = nib.load(image_path).get_fdata()[:, :, z]
        label = nib.load(label_path).get_fdata()[:, :, z]

        assert image.shape == label.shape
        assert len(image.shape) == 2

        resize_factor = np.array(self.out_shape) / image.shape[:2]
        dsize = np.int16(resize_factor.min() * np.float16(image.shape[:2]))
        image = cv2.resize(src=image, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(src=label, dsize=dsize, interpolation=cv2.INTER_NEAREST)

        # DEVIATION from brain_tumor_nifti.py: keep native 0-3 integer
        # classes (BG/RV/MYO/LV). Every official CUTS dataset asserts binary
        # labels here instead -- ACDC has no binary-label precedent in CUTS.
        label = np.round(label).astype(np.int64)
        assert set(np.unique(label).tolist()) <= {0, 1, 2, 3}

        image = crop_or_pad(image, image.shape, self.out_shape)
        # crop_or_pad (official implementation) always returns float64, even
        # for an int input -- re-cast so GT semantic labels stay integer.
        label = crop_or_pad(label, label.shape, self.out_shape, pad_value=0).astype(np.int64)

        # Official min-max rescale to [-1,1] (verbatim from brain_tumor_nifti.py).
        epsilon = 1e-12
        image = 2 * (image - image.min() + epsilon) / (image.max() - image.min() + epsilon) - 1

        assert image.shape == self.out_shape
        assert label.shape == self.out_shape
        image = image[None, :, :]
        label = label[None, :, :]

        return image, label

    def num_image_channel(self) -> int:
        return 1

    def num_classes(self) -> int:
        return 4
