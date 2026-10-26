"""Small manufactured native-volume fixtures for geometry/data tests only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from self_audit.model import digest

from .geometry import AXIS_ORDER, SCHEMA, sha256_file, validate_sidecar


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def make_volume_fixture(
    folder: str | Path,
    *,
    seed: int = 11,
    n_labeled: int = 2,
    n_unlabeled: int = 2,
    shape_tzyx: tuple[int, int, int, int] = (2, 5, 17, 19),
    affine_xyz_to_world_mm: Any | None = None,
    unknown_fraction: float = 0.0,
) -> tuple[Path, str, Path, str]:
    """Create a deterministic normalized TRAIN manifest and geometry sidecar.

    Return order is ``(manifest_path, manifest_sha256, geometry_path,
    geometry_sha256)``.  Every generated object is explicitly marked
    ``synthetic`` and must never be used as patient/GPU evidence.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if len(shape_tzyx) != 4 or any(int(x) < 1 for x in shape_tzyx):
        raise ValueError("shape_tzyx must be four positive dimensions")
    if not 0 <= unknown_fraction < 1:
        raise ValueError("unknown_fraction must be in [0,1)")
    n_labeled, n_unlabeled = int(n_labeled), int(n_unlabeled)
    if n_labeled < 1 or n_unlabeled < 1:
        raise ValueError("fixture needs at least one labeled and one unlabeled patient")
    rng = np.random.default_rng(seed)
    tdim, zdim, h, w = (int(x) for x in shape_tzyx)
    if affine_xyz_to_world_mm is None:
        affine = np.eye(4, dtype=float)
        affine[0, 0], affine[1, 1], affine[2, 2] = 1.3, 0.9, 4.2
    else:
        affine = np.asarray(affine_xyz_to_world_mm, dtype=float)
    if affine.shape != (4, 4):
        raise ValueError("affine_xyz_to_world_mm must be 4x4")
    labeled, unlabeled, patients = [], [], {}
    yy, xx = np.mgrid[:h, :w]
    for j in range(n_labeled + n_unlabeled):
        pid = f"synthetic_volume_{j:03d}"
        image = rng.random(shape_tzyx, dtype=np.float32)
        # A moving, asymmetric three-class phantom is useful for coupled crop
        # and temporal tests, but remains manufactured data.
        mask = np.zeros(shape_tzyx, dtype=np.uint8)
        for t in range(tdim):
            cy = h * (0.48 + 0.05 * np.sin(t + j)); cx = w * (0.47 + 0.04 * np.cos(t + j))
            r = ((yy - cy) / max(1, h * 0.27)) ** 2 + ((xx - cx) / max(1, w * 0.30)) ** 2
            plane = np.zeros((h, w), dtype=np.uint8)
            plane[r < 1.0] = 2
            plane[r < 0.55] = 3
            plane[((yy - cy) ** 2 + (xx - (cx - w * 0.32)) ** 2) < (min(h, w) * 0.11) ** 2] = 1
            mask[t] = plane[None]
        if unknown_fraction:
            unknown = rng.random(mask.shape) < unknown_fraction
            mask[unknown] = 255
        image_path = folder / f"{pid}.npy"; np.save(image_path, image)
        rec = {"id": pid, "image": image_path.name, "image_sha256": digest(image_path)}
        if j < n_labeled:
            rec["masks"] = {}
            for t in range(tdim):
                mask_path = folder / f"{pid}_mask_t{t:03d}.npy"; np.save(mask_path, mask[t])
                rec["masks"][str(t)] = {"path": mask_path.name, "sha256": digest(mask_path)}
            labeled.append(rec)
        else:
            unlabeled.append(rec)
        patients[pid] = {
            "native_shape_tzyx": list(shape_tzyx),
            "affine_xyz_to_world_mm": affine.tolist(),
            "axis_order": AXIS_ORDER, "units": "mm",
            "orientation": "fixture_affine_direction_cosines",
            "slice_order": "increasing_native_z_index",
            "temporal_frame_mapping": list(range(tdim)),
            "image_sha256": rec["image_sha256"],
        }
    manifest_path = folder / "train.json"
    _write_json(manifest_path, {
        "schema": "self_audit.data.v2", "role": "training", "synthetic": True,
        "label_budget": n_labeled, "labeled": labeled, "unlabeled": unlabeled,
        "evaluation_ids": [],
    })
    manifest_sha = sha256_file(manifest_path)
    geometry_path = folder / "geometry.json"
    sidecar = {
        "schema": SCHEMA, "axis_order": AXIS_ORDER, "units": "mm",
        "synthetic": True, "source_manifest": manifest_path.name,
        "source_manifest_sha256": manifest_sha, "patients": patients,
    }
    sidecar = validate_sidecar(sidecar)
    _write_json(geometry_path, sidecar)
    return manifest_path, manifest_sha, geometry_path, sha256_file(geometry_path)


__all__ = ["make_volume_fixture"]
