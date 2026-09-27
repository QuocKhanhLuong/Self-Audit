"""ACDC loader following the data-path conventions of upstream CUTS.

This is intentionally separate from ``cardiac_benchmark.dataset``.  It is an
ACDC port for upstream ``main.py``: it enumerates native ED/ES 2-D slices,
rescales a slice while retaining its aspect ratio, centre-crops/pads it to the
declared canvas, and min--max normalizes that slice to ``[-1, 1]``.

The upstream CUTS datasets return labels even though the unsupervised training
loss does not consume them.  This loader follows that interface for the
original-style protocol.  Consequently, it is not an image-only loader: GT is
used to verify that an ED/ES annotation exists and is returned for the
upstream evaluation scripts.  The shared cardiac benchmark must continue to
use its own image-only dataset and manifest.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

try:  # Keep existing CUTS configurations importable without the ACDC extra.
    import nibabel as nib
except ImportError:  # pragma: no cover - exercised only in a missing optional dep environment.
    nib = None


def _parse_ed_es(info_cfg_path: Path) -> Tuple[int, int]:
    """Read the ED and ES frame numbers from a native ACDC ``Info.cfg``."""
    ed = es = None
    for line in info_cfg_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ED:"):
            ed = int(line.split(":", 1)[1].strip())
        elif line.startswith("ES:"):
            es = int(line.split(":", 1)[1].strip())
    if ed is None or es is None:
        raise ValueError(f"{info_cfg_path} must declare both ED and ES")
    return ed, es


def crop_or_pad(in_image: np.ndarray, in_shape: Tuple[int, int],
                out_shape: Tuple[int, int], pad_value: float = 0) -> np.ndarray:
    """Verbatim spatial rule from ``datasets/brain_tumor.py``."""
    assert len(in_shape) == len(out_shape)
    dimensions = len(in_shape)
    out_shape_min = [
        int(np.floor((out_shape[i] - in_shape[i]) / 2)) if out_shape[i] >= in_shape[i] else 0
        for i in range(dimensions)
    ]
    out_shape_max = [
        out_shape_min[i] + in_shape[i] if out_shape[i] >= in_shape[i] else None
        for i in range(dimensions)
    ]
    in_shape_min = [
        0 if out_shape[i] >= in_shape[i] else int(np.floor((in_shape[i] - out_shape[i]) / 2))
        for i in range(dimensions)
    ]
    in_shape_max = [
        None if out_shape[i] >= in_shape[i] else in_shape_min[i] + out_shape[i]
        for i in range(dimensions)
    ]
    in_slicer = tuple(slice(i, j) for i, j in zip(in_shape_min, in_shape_max))
    out_slicer = tuple(slice(i, j) for i, j in zip(out_shape_min, out_shape_max))
    out_image = np.ones(out_shape) * pad_value
    out_image[out_slicer] = in_image[in_slicer]
    return out_image


class ACDCOriginalStyle(Dataset):
    """Native ACDC ED/ES-slice dataset for the upstream CUTS entrypoint."""

    def __init__(self, base_path: str, out_shape: Tuple[int, int] = (224, 224)):
        if nib is None:
            raise ImportError("ACDCOriginalStyle requires nibabel; install the CUTS ACDC dependencies.")
        self.base_path = str(base_path)
        self.out_shape = tuple(int(value) for value in out_shape)
        if len(self.out_shape) != 2 or min(self.out_shape) <= 0:
            raise ValueError(f"out_shape must be two positive values, got {out_shape!r}")

        self.samples: List[Tuple[str, int, int]] = []
        for patient_dir_text in sorted(glob(str(Path(base_path) / "patient*"))):
            patient_dir = Path(patient_dir_text)
            info_cfg = patient_dir / "Info.cfg"
            if not info_cfg.is_file():
                continue
            patient_id = patient_dir.name
            for frame in sorted(set(_parse_ed_es(info_cfg))):
                image_path = patient_dir / f"{patient_id}_frame{frame:02d}.nii"
                label_path = patient_dir / f"{patient_id}_frame{frame:02d}_gt.nii"
                if not image_path.is_file() or not label_path.is_file():
                    continue
                image_shape = nib.load(str(image_path)).shape
                label_shape = nib.load(str(label_path)).shape
                if len(image_shape) != 3 or image_shape != label_shape:
                    raise ValueError(f"ACDC image/GT shape mismatch for {image_path}")
                for z_index in range(image_shape[2]):
                    self.samples.append((str(patient_dir), int(frame), int(z_index)))
        if not self.samples:
            raise ValueError(f"no annotated ACDC ED/ES slices found under {base_path!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        patient_dir, frame, z_index = self.samples[idx]
        patient_id = Path(patient_dir).name
        image_path = Path(patient_dir) / f"{patient_id}_frame{frame:02d}.nii"
        label_path = Path(patient_dir) / f"{patient_id}_frame{frame:02d}_gt.nii"
        image = nib.load(str(image_path)).get_fdata()[:, :, z_index]
        label = nib.load(str(label_path)).get_fdata()[:, :, z_index]
        if image.shape != label.shape or image.ndim != 2:
            raise ValueError(f"invalid two-dimensional ACDC sample {image_path}, z={z_index}")

        resize_factor = np.asarray(self.out_shape, dtype=np.float64) / image.shape[:2]
        resized_hw = np.maximum(1, np.floor(resize_factor.min() * np.asarray(image.shape[:2])).astype(np.int64))
        dsize = (int(resized_hw[1]), int(resized_hw[0]))  # OpenCV takes (width, height).
        image = cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, dsize=dsize, interpolation=cv2.INTER_NEAREST)
        label = np.rint(label).astype(np.int64)
        if not set(np.unique(label).tolist()) <= {0, 1, 2, 3}:
            raise ValueError(f"ACDC label has values outside BG/RV/MYO/LV for {label_path}")

        image = crop_or_pad(image, image.shape, self.out_shape)
        label = crop_or_pad(label, label.shape, self.out_shape, pad_value=0).astype(np.int64)
        epsilon = 1e-12
        image = 2 * (image - image.min() + epsilon) / (image.max() - image.min() + epsilon) - 1
        return image[None, :, :].astype(np.float32), label[None, :, :]

    def num_image_channel(self) -> int:
        return 1

    def num_classes(self) -> int:
        return 4
