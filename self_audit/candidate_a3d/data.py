"""Full-volume A-3D view over the existing hash-pinned cache loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data import CLASS_MAPPING, GEOMETRY_MAP_SCHEMA, GEOMETRY_SCHEMA, TrainingData, digest, write_json
from .geometry import VolumeGeometry3D


class CandidateA3DData:
    """Strict full-volume loader for annotated TRAIN frames.

    The cache remains ``[T,D,H,W]``.  A sample is copied to ``image`` shape
    ``[1,1,D,H,W]`` and ``labels`` shape ``[1,D,H,W]``.  No crop, resample,
    interpolation, or hidden patch coordinate is performed.
    """

    def __init__(self, manifest, expected_sha256, *, synthetic: bool = False):
        self.base = TrainingData(manifest, expected_sha256, synthetic=synthetic)
        self.path = self.base.path
        self.manifest = self.base.manifest
        self.labeled = self.base.labeled
        self.unlabeled = self.base.unlabeled
        self.lids = self.base.lids
        self.uids = self.base.uids
        self.label_reads = self.base.label_reads
        self.geometry: dict[str, VolumeGeometry3D] = {}
        for record in self.labeled + self.unlabeled:
            patient = record["id"]
            raw_geometry = self.base.geometry_for(patient, required=True)
            image = self.base.array(record["image"], record["image_sha256"])
            self.geometry[patient] = VolumeGeometry3D.from_cache_geometry(
                raw_geometry, expected_shape=image.shape[1:]
            )

    def geometry_for(self, patient_id: str) -> dict[str, Any]:
        try:
            return self.geometry[patient_id].as_dict()
        except KeyError as exc:
            raise ValueError(f"patient {patient_id!r} has no full-volume geometry") from exc

    def labeled_frame_keys(self) -> list[tuple[int, int]]:
        """Return compact ``(record_index, frame)`` references only for TRAIN GT."""
        keys: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.labeled):
            image = self.base.array(record["image"], record["image_sha256"])
            geometry = self.geometry[record["id"]]
            for frame_key in sorted(record["masks"], key=int):
                frame = int(frame_key)
                mask_record = record["masks"][frame_key]
                mask = self.base.array(mask_record["path"], mask_record["sha256"])
                if mask.shape != geometry.shape or mask.shape != image.shape[1:]:
                    raise ValueError("A-3D annotated mask must cover the native full volume")
                if not np.isin(mask, [0, 1, 2, 3]).all():
                    raise ValueError("A-3D dense TRAIN masks must not contain ignore=255")
                keys.append((record_index, frame))
        if not keys:
            raise ValueError("TRAIN manifest has no annotated full-volume frames")
        return keys

    def load_labeled_frame(self, key: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
        """Materialize one full-volume frame and return geometry separately."""
        if len(key) != 2:
            raise ValueError("A-3D volume key must be (record_index, frame)")
        record_index, frame = (int(value) for value in key)
        if not 0 <= record_index < len(self.labeled):
            raise IndexError("A-3D record index is outside TRAIN")
        record = self.labeled[record_index]
        frame_key = str(frame)
        if frame_key not in record["masks"]:
            raise IndexError("A-3D frame is not annotated")
        image = self.base.array(record["image"], record["image_sha256"])
        mask_record = record["masks"][frame_key]
        mask = self.base.array(mask_record["path"], mask_record["sha256"])
        geometry = self.geometry[record["id"]]
        if not 0 <= frame < image.shape[0] or image.shape[1:] != geometry.shape or mask.shape != geometry.shape:
            raise ValueError("A-3D sample is not a native full-volume frame")
        if not np.isin(mask, [0, 1, 2, 3]).all():
            raise ValueError("A-3D dense TRAIN masks must not contain ignore=255")
        self.label_reads.add(record["id"])
        x = torch.from_numpy(np.array(image[frame], dtype=np.float32, copy=True))[None, None]
        y = torch.from_numpy(np.array(mask, dtype=np.int64, copy=True))[None]
        if not torch.isfinite(x).all() or bool((x < 0).any()) or bool((x > 1).any()):
            raise ValueError("A-3D image volume must be finite and normalized to [0,1]")
        runtime_geometry = geometry.as_dict()
        metadata = {"patient": record["id"], "frame": frame}
        return x, y, runtime_geometry, metadata

def _synthetic_geometry(shape_zyx: tuple[int, int, int], spacing=(2.5, 0.7, 1.3)) -> dict[str, Any]:
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag(spacing)
    return {
        "schema": GEOMETRY_SCHEMA,
        "array_axes": ["T", "Z", "Y", "X"],
        "spatial_axes": ["Z", "Y", "X"],
        "shape_zyx": list(shape_zyx),
        "spacing_zyx_mm": list(spacing),
        "spacing_yx_mm": list(spacing[1:]),
        "affine_zyx": affine.tolist(),
        "class_mapping": dict(CLASS_MAPPING),
    }


def synthetic_fixture(
    folder, *, seed: int = 11, n_labeled: int = 3, n_unlabeled: int = 2,
    n_evaluation: int = 2, shape_tzyx: tuple[int, int, int, int] = (2, 4, 8, 9),
):
    """Create a small full-volume fixture with the same API as Candidate A.

    The fixture is manufactured software evidence only.  It intentionally
    stores native ``[T,D,H,W]`` arrays and full masks; no crop or resampling is
    used.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    if len(shape_tzyx) != 4 or any(type(value) is not int or value < 1 for value in shape_tzyx):
        raise ValueError("shape_tzyx must contain positive integers")
    frame_count, depth, height, width = shape_tzyx
    geometry = _synthetic_geometry((depth, height, width))
    rng = np.random.default_rng(seed)

    def make_patient(patient: str, with_masks: bool) -> dict[str, Any]:
        zz, yy, xx = np.mgrid[:depth, :height, :width]
        center_y = height * (0.45 + rng.uniform(-0.04, 0.04))
        center_x = width * (0.52 + rng.uniform(-0.04, 0.04))
        center_z = depth * 0.5
        radius = np.sqrt(((yy - center_y) / max(height, 1)) ** 2 + ((xx - center_x) / max(width, 1)) ** 2)
        mask = np.zeros((depth, height, width), dtype=np.uint8)
        mask[radius < 0.30] = 2
        mask[radius < 0.18] = 3
        mask[((yy - center_y) ** 2 + (xx - (center_x - width * 0.27)) ** 2 +
              (zz - center_z) ** 2) < (min(height, width, depth) * 0.12) ** 2] = 1
        image = np.empty((frame_count, depth, height, width), dtype=np.float32)
        levels = np.asarray([0.10, 0.80, 0.45, 0.90], dtype=np.float32)
        for frame in range(frame_count):
            image[frame] = np.clip(levels[mask] + rng.normal(0.0, 0.02, mask.shape), 0.0, 1.0)
        image_path = folder / f"{patient}.npy"
        np.save(image_path, image)
        record = {"id": patient, "image": image_path.name, "image_sha256": digest(image_path)}
        if with_masks:
            masks = {}
            for frame in range(frame_count):
                mask_path = folder / f"{patient}_t{frame:03d}_mask.npy"
                np.save(mask_path, mask)
                masks[str(frame)] = {"path": mask_path.name, "sha256": digest(mask_path)}
            record["masks"] = masks
        return record

    labeled = [make_patient(f"train_labeled_{index:03d}", True) for index in range(n_labeled)]
    unlabeled = [make_patient(f"train_unlabeled_{index:03d}", False) for index in range(n_unlabeled)]
    evaluation = [make_patient(f"evaluation_{index:03d}", True) for index in range(n_evaluation)]
    train_ids = [record["id"] for record in labeled + unlabeled]
    eval_ids = [record["id"] for record in evaluation]
    train_manifest = folder / "train.json"
    write_json(train_manifest, {
        "schema": "self_audit.data.v2", "role": "training", "synthetic": True,
        "dataset": "synthetic", "label_budget": n_labeled,
        "labeled": labeled, "unlabeled": unlabeled, "evaluation_ids": eval_ids,
        "class_mapping": dict(CLASS_MAPPING),
        "geometry": {"schema": GEOMETRY_MAP_SCHEMA,
                     "by_patient": {patient: dict(geometry) for patient in train_ids}},
    })
    evaluation_manifest = folder / "development.json"
    write_json(evaluation_manifest, {
        "schema": "self_audit.data.v2", "role": "development", "synthetic": True,
        "dataset": "synthetic", "scope": "candidate_a3d_synthetic",
        "patients": evaluation, "class_mapping": dict(CLASS_MAPPING),
        "geometry": {"schema": GEOMETRY_MAP_SCHEMA,
                     "by_patient": {patient: dict(geometry) for patient in eval_ids}},
    })
    return {
        "train": train_manifest, "train_sha256": digest(train_manifest),
        "evaluation": evaluation_manifest, "evaluation_sha256": digest(evaluation_manifest),
        "train_ids": train_ids, "evaluation_ids": eval_ids,
    }


synthetic_candidate_a3d_fixture = synthetic_fixture


__all__ = [
    "CandidateA3DData", "synthetic_fixture",
    "synthetic_candidate_a3d_fixture",
]
