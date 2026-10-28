"""Compact training data loader with opt-in K=1 support and audit frame streaming.

Owned by Worker teacher/data per COMPACT_IMPLEMENTATION_CONTRACT.md:
  - CompactTrainingData: Inherits from and remains 100% compatible with TrainingData methods.
  - Opt-in K=1 support: Reuses single audit patient without creating duplicate fake patients.
  - Legacy loader K>=2 behavior unchanged.
  - Repeatable audit_batches streaming whole labeled frames for compact_control audit trials.
  - Synthetic fixture creation for reproducible unit and integration tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .data import TrainingData, crop_native, write_json
from .model import digest

__all__ = [
    "CompactTrainingData",
    "build_compact_training_data",
    "create_compact_audit_stream",
    "synthetic_compact_fixture",
]


class CompactTrainingData(TrainingData):
    """Training data provider for compact student distillation and audit.

    Compatible with all TrainingData methods, with opt-in support for K=1
    label budget where the supervised patient is legitimately reused for audit
    without fabricating duplicate fake patient records.
    """

    def __init__(
        self,
        manifest: Union[str, Path],
        expected_sha256: str,
        *,
        synthetic: bool = False,
        allow_k1: bool = True,
        max_array_handles: int = 8,
    ):
        if type(max_array_handles) is not int or max_array_handles < 1:
            raise ValueError("max_array_handles must be a positive integer")
        self.max_array_handles = max_array_handles
        super().__init__(manifest, expected_sha256, synthetic=synthetic, allow_k1=allow_k1)

        # Enforce that all patient IDs are strictly distinct (no duplicate fake patients)
        all_ids = self.lids + self.uids
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("duplicate patient IDs detected; fake patient duplication is prohibited")

        # In K=1 mode, exactly 1 labeled patient is present
        if len(self.lids) == 1 and not allow_k1:
            raise ValueError("K=1 labeled patient budget requires allow_k1=True")

    def array(self, path, sha):
        """Bound retained image/mask mmaps; live caller views survive eviction."""
        key = str(self.resolve(path))
        value = super().array(path, sha)
        self.cache.pop(key)
        self.cache[key] = value
        while len(self.cache) > self.max_array_handles:
            # Do not forcibly close mmap storage that an active audit generator
            # still references. Dropping the cache owner releases inactive maps.
            self.cache.pop(next(iter(self.cache)))
        return value

    def close(self):
        self.cache.clear()

    def sample_indices(self, step: int, rng: np.random.Generator) -> Tuple[int, int, int]:
        """Return (li, ai, ui) indices for a training step.

        For K=1 (single labeled patient):
            li = 0
            ai = 0 (reused audit patient, no duplicate fake patient)
            ui = step % len(self.uids)
        For K >= 2:
            Preserves exact TrainingData behavior with disjoint li and ai.
        """
        return super().sample_indices(step, rng)

    def stream_whole_labeled_frames(
        self,
        patient_index: int,
        *,
        device: str = "cpu",
    ) -> Iterable[Dict[str, Any]]:
        """Stream whole labeled frames and all slices for a given labeled patient.

        Yields dictionaries with:
          - x: (1, 3, H, W) Tensor (3-slice context: z-1, z, z+1)
          - y: (1, H, W) Tensor (ground truth segmentation mask)
          - valid: (1, H, W) boolean Tensor
          - frame: int (time frame index)
          - slice: int (z slice index)
          - patient: str (patient ID)
        """
        if not 0 <= patient_index < len(self.labeled):
            raise IndexError(f"patient_index {patient_index} out of range [0, {len(self.labeled)})")

        r = self.labeled[patient_index]
        cine = self.array(r["image"], r["image_sha256"])
        self.label_reads.add(r["id"])

        for frame_str in sorted(r["masks"], key=int):
            t = int(frame_str)
            rec = r["masks"][frame_str]
            mask = self.array(rec["path"], rec["sha256"])
            if mask.shape != cine.shape[1:]:
                raise ValueError(f"mask shape {mask.shape} != cine spatial shape {cine.shape[1:]}")

            z_dim = cine.shape[1]
            for z in range(z_dim):
                x = self.context(cine, t, z).to(device)  # (1, 3, H, W)
                y = torch.from_numpy(np.array(mask[z], dtype=np.int64, copy=True))[None].to(device)  # (1, H, W)
                valid = torch.ones_like(y, dtype=torch.bool)
                yield {
                    "x": x,
                    "y": y,
                    "valid": valid,
                    "frame": t,
                    "slice": z,
                    "patient": r["id"],
                }

    def create_audit_batches(
        self,
        patient_index: Optional[int] = None,
        *,
        device: str = "cpu",
    ) -> Callable[[], Iterable[Dict[str, Any]]]:
        """Create a repeatable callable yielding audit batches for compact_control.

        Returns a zero-argument callable that produces a fresh generator streaming
        whole labeled frames for the requested patient (or all labeled patients if None).
        """
        def _generator() -> Iterable[Dict[str, Any]]:
            if patient_index is not None:
                indices = [patient_index]
            else:
                indices = list(range(len(self.labeled)))

            for p_idx in indices:
                yield from self.stream_whole_labeled_frames(p_idx, device=device)

        return _generator


def build_compact_training_data(
    manifest_path: Union[str, Path],
    expected_sha256: str,
    *,
    synthetic: bool = False,
    allow_k1: bool = True,
) -> CompactTrainingData:
    """Instantiate and validate a CompactTrainingData loader."""
    return CompactTrainingData(
        manifest=manifest_path,
        expected_sha256=expected_sha256,
        synthetic=synthetic,
        allow_k1=allow_k1,
    )


def create_compact_audit_stream(
    data: CompactTrainingData,
    patient_index: int,
    *,
    device: str = "cpu",
) -> Callable[[], Iterable[Dict[str, Any]]]:
    """Helper to create audit_batches callable for apply_compact_update."""
    return data.create_audit_batches(patient_index=patient_index, device=device)


def synthetic_compact_fixture(
    folder: Union[str, Path],
    *,
    seed: int = 42,
    n_labeled: int = 1,
    n_unlabeled: int = 2,
    size: int = 32,
    num_frames: int = 2,
    num_slices: int = 4,
) -> Tuple[Path, str]:
    """Create a synthetic compact training fixture with manufactured images and masks.

    Manufactured software test fixture only; never evidence of cardiac accuracy.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[:size, :size]
    labeled: List[Dict[str, Any]] = []
    unlabeled: List[Dict[str, Any]] = []

    total_patients = n_labeled + n_unlabeled
    for j in range(total_patients):
        pid = f"synthetic_p{j:03d}"
        cine = np.zeros((num_frames, num_slices, size, size), dtype=np.float32)

        masks_dict: Dict[str, Dict[str, str]] = {}
        is_labeled = j < n_labeled

        for t in range(num_frames):
            for z in range(num_slices):
                cy = size * (0.5 + rng.uniform(-0.05, 0.05))
                cx = size * (0.5 + rng.uniform(-0.05, 0.05))
                radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

                mask_2d = np.zeros((size, size), dtype=np.uint8)
                mask_2d[radius < size * 0.35] = 1  # RV
                mask_2d[radius < size * 0.25] = 2  # MYO
                mask_2d[radius < size * 0.15] = 3  # LV

                noise = rng.normal(0, 0.02, (size, size)).astype(np.float32)
                intensity = np.choose(mask_2d, [0.1, 0.4, 0.6, 0.9]).astype(np.float32)
                cine[t, z] = np.clip(intensity + noise, 0.0, 1.0)

            if is_labeled and t == 0:
                # Save mask volume for frame 0
                mask_vol = np.zeros((num_slices, size, size), dtype=np.uint8)
                for z in range(num_slices):
                    cy = size * 0.5
                    cx = size * 0.5
                    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                    mask_vol[z][radius < size * 0.35] = 1
                    mask_vol[z][radius < size * 0.25] = 2
                    mask_vol[z][radius < size * 0.15] = 3

                mask_path = folder / f"{pid}_frame{t}_mask.npy"
                np.save(mask_path, mask_vol)
                masks_dict[str(t)] = {
                    "path": mask_path.name,
                    "sha256": digest(mask_path),
                }

        image_path = folder / f"{pid}_cine.npy"
        np.save(image_path, cine)
        rec = {
            "id": pid,
            "image": image_path.name,
            "image_sha256": digest(image_path),
        }
        if is_labeled:
            rec["masks"] = masks_dict
            labeled.append(rec)
        else:
            unlabeled.append(rec)

    manifest_data = {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": True,
        "label_budget": n_labeled,
        "labeled": labeled,
        "unlabeled": unlabeled,
        "evaluation_ids": [],
    }
    manifest_path = folder / "train.json"
    write_json(manifest_path, manifest_data)
    manifest_sha = digest(manifest_path)
    return manifest_path, manifest_sha
