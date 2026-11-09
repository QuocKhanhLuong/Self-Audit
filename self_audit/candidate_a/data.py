"""Candidate-A data boundary: existing sampling plus explicit native geometry.

This wrapper intentionally does not augment or resample images.  Teacher and
student therefore receive the same cached ``T,Z,Y,X`` slice context; geometry
is metadata only and never inferred as unit spacing.
"""
from pathlib import Path

import numpy as np

from ..data import (
    CLASS_MAPPING,
    GEOMETRY_MAP_SCHEMA,
    GEOMETRY_SCHEMA,
    TrainingData,
    digest,
    write_json,
)


class CandidateAData:
    """Strict geometry-bearing view over the canonical ``TrainingData`` API.

    ``sample_labeled`` returns ``(x, y, spacing_yx_mm, patient_id)`` where
    ``x`` is ``[1,3,H,W]`` and ``y`` is ``[1,H,W]``.  The base sampler remains
    responsible for deterministic frame/slice selection and label accounting.
    ``sample_unlabeled`` returns ``(x, spacing_yx_mm, patient_id)``.
    """

    def __init__(self, manifest, expected_sha256, *, synthetic=False):
        self.base = TrainingData(manifest, expected_sha256, synthetic=synthetic)
        self.path = self.base.path
        self.manifest = self.base.manifest
        self.labeled = self.base.labeled
        self.unlabeled = self.base.unlabeled
        self.lids = self.base.lids
        self.uids = self.base.uids
        self.label_reads = self.base.label_reads
        self.geometry = self.base.geometry
        missing = [pid for pid in self.lids + self.uids if pid not in self.geometry]
        if missing:
            raise ValueError(
                "Candidate A requires validated native geometry for every TRAIN patient: "
                + ",".join(missing)
            )

    def geometry_for(self, patient_id):
        return self.base.geometry_for(patient_id, required=True)

    def spacing_yx_for(self, patient_id):
        return self.base.spacing_yx_for(patient_id, required=True)

    def sample_labeled(self, index, rng, *, foreground_probability=.5):
        x, y, metadata = self.base.sample_labeled(
            index, rng, foreground_probability=foreground_probability
        )
        patient_id = metadata["patient"]
        return x, y, self.spacing_yx_for(patient_id), patient_id

    def sample_unlabeled(self, index, rng):
        x, metadata = self.base.sample_unlabeled(index, rng)
        patient_id = metadata["patient"]
        return x, self.spacing_yx_for(patient_id), patient_id


def _synthetic_geometry(shape_zyx, spacing_zyx=(2.5, 0.7, 1.3)):
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag(spacing)
    return {
        "schema": GEOMETRY_SCHEMA,
        "array_axes": ["T", "Z", "Y", "X"],
        "spatial_axes": ["Z", "Y", "X"],
        "shape_zyx": [int(x) for x in shape_zyx],
        "spacing_zyx_mm": [float(x) for x in spacing],
        "spacing_yx_mm": [float(x) for x in spacing[1:]],
        "affine_zyx": affine.tolist(),
        "class_mapping": dict(CLASS_MAPPING),
    }


def synthetic_candidate_a_fixture(
    folder,
    *,
    seed=11,
    n_labeled=3,
    n_unlabeled=2,
    n_evaluation=2,
    shape_tzyx=(2, 3, 12, 13),
):
    """Create tiny finite train/evaluation manifests with anisotropic geometry.

    The result is manufactured-only and follows ``self_audit.data.v2``.  It
    returns paths and hashes for both manifests so callers can pass them to the
    normal hash-pinned loaders without weakening provenance checks.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    if len(shape_tzyx) != 4 or any(type(x) is not int or x < 1 for x in shape_tzyx):
        raise ValueError("shape_tzyx must contain positive integers")
    t_count, z_count, height, width = shape_tzyx
    geometry = _synthetic_geometry(shape_tzyx[1:])
    rng = np.random.default_rng(seed)

    def make_patient(pid, with_masks):
        yy, xx = np.mgrid[:height, :width]
        cy = height * (0.45 + rng.uniform(-0.04, 0.04))
        cx = width * (0.52 + rng.uniform(-0.04, 0.04))
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        base = np.zeros((height, width), dtype=np.uint8)
        base[radius < min(height, width) * 0.30] = 2
        base[radius < min(height, width) * 0.18] = 3
        base[(yy - cy) ** 2 + (xx - (cx - width * 0.27)) ** 2 < (min(height, width) * 0.12) ** 2] = 1
        mask = np.stack([np.roll(base, shift=k % 2, axis=0) for k in range(z_count)], axis=0)
        image = np.empty((t_count, z_count, height, width), dtype=np.float32)
        levels = np.asarray([0.10, 0.80, 0.45, 0.90], dtype=np.float32)
        for t in range(t_count):
            image[t] = np.clip(
                levels[mask] + rng.normal(0.0, 0.02, mask.shape), 0.0, 1.0
            )
        image_path = folder / f"{pid}.npy"
        np.save(image_path, image)
        record = {"id": pid, "image": image_path.name, "image_sha256": digest(image_path)}
        if with_masks:
            masks = {}
            for t in range(t_count):
                mask_path = folder / f"{pid}_t{t:03d}_mask.npy"
                np.save(mask_path, mask)
                masks[str(t)] = {"path": mask_path.name, "sha256": digest(mask_path)}
            record["masks"] = masks
        return record

    labeled = [make_patient(f"train_labeled_{i:03d}", True) for i in range(n_labeled)]
    unlabeled = [make_patient(f"train_unlabeled_{i:03d}", False) for i in range(n_unlabeled)]
    evaluation = [make_patient(f"evaluation_{i:03d}", True) for i in range(n_evaluation)]
    train_ids = [r["id"] for r in labeled + unlabeled]
    eval_ids = [r["id"] for r in evaluation]
    train_manifest = folder / "train.json"
    write_json(
        train_manifest,
        {
            "schema": "self_audit.data.v2",
            "role": "training",
            "synthetic": True,
            "label_budget": n_labeled,
            "labeled": labeled,
            "unlabeled": unlabeled,
            "evaluation_ids": eval_ids,
            "class_mapping": dict(CLASS_MAPPING),
            "geometry": {
                "schema": GEOMETRY_MAP_SCHEMA,
                "by_patient": {pid: dict(geometry) for pid in train_ids},
            },
        },
    )
    evaluation_manifest = folder / "development.json"
    write_json(
        evaluation_manifest,
        {
            "schema": "self_audit.data.v2",
            "role": "development",
            "synthetic": True,
            "scope": "candidate_a_synthetic",
            "patients": evaluation,
            "class_mapping": dict(CLASS_MAPPING),
            "geometry": {
                "schema": GEOMETRY_MAP_SCHEMA,
                "by_patient": {pid: dict(geometry) for pid in eval_ids},
            },
        },
    )
    return {
        "train": train_manifest,
        "train_sha256": digest(train_manifest),
        "evaluation": evaluation_manifest,
        "evaluation_sha256": digest(evaluation_manifest),
        "train_ids": train_ids,
        "evaluation_ids": eval_ids,
    }
