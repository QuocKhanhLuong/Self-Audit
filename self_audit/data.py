"""Separate labeled/unlabeled access. U masks are not listed or opened."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .model import digest


CLASS_MAPPING = {
    "0": "background",
    "1": "RV",
    "2": "MYO",
    "3": "LV",
    "255": "ignore",
}
GEOMETRY_SCHEMA = "self_audit.geometry.native_zyx.v1"
GEOMETRY_MAP_SCHEMA = "self_audit.geometry_map.v1"


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def label_subset(patient_ids, count, seed):
    if len(set(patient_ids)) != len(patient_ids) or not 2 <= count < len(patient_ids):
        raise ValueError("need unique TRAIN IDs and 2 <= label budget < TRAIN patients")
    return sorted(sorted(patient_ids, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())[:count])


def _as_float_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def validate_geometry(geometry, *, shape_zyx=None):
    """Validate native cache geometry; never invent spacing for old manifests.

    ``affine_zyx`` maps cache voxel coordinates ``(z, y, x, 1)`` to native
    world coordinates.  The linear axes must be orthogonal because the
    candidate-A distance target only supports a native orthogonal grid.
    """
    if not isinstance(geometry, dict) or geometry.get("schema") != GEOMETRY_SCHEMA:
        raise ValueError("invalid native ZYX geometry schema")
    if geometry.get("array_axes") != ["T", "Z", "Y", "X"]:
        raise ValueError("geometry array_axes must be T,Z,Y,X")
    if geometry.get("spatial_axes") != ["Z", "Y", "X"]:
        raise ValueError("geometry spatial_axes must be Z,Y,X")
    shape = geometry.get("shape_zyx")
    if not isinstance(shape, list) or len(shape) != 3 or any(type(x) is not int or x < 1 for x in shape):
        raise ValueError("geometry shape_zyx must contain positive integers")
    if shape_zyx is not None and tuple(shape) != tuple(int(x) for x in shape_zyx):
        raise ValueError("geometry shape_zyx does not match cached image")
    spacing = _as_float_array(geometry.get("spacing_zyx_mm"), (3,), "spacing_zyx_mm")
    if (spacing <= 0).any():
        raise ValueError("spacing_zyx_mm must be positive")
    spacing_yx = _as_float_array(geometry.get("spacing_yx_mm"), (2,), "spacing_yx_mm")
    if not np.allclose(spacing_yx, spacing[1:], rtol=1e-6, atol=1e-6):
        raise ValueError("spacing_yx_mm must equal spacing_zyx_mm[1:]")
    affine = _as_float_array(geometry.get("affine_zyx"), (4, 4), "affine_zyx")
    if not np.allclose(affine[3], [0., 0., 0., 1.], rtol=0., atol=1e-6):
        raise ValueError("affine_zyx must be homogeneous")
    linear = affine[:3, :3]
    lengths = np.linalg.norm(linear, axis=0)
    if not np.allclose(lengths, spacing, rtol=1e-5, atol=1e-6):
        raise ValueError("affine_zyx axis lengths disagree with spacing")
    if (lengths <= 0).any():
        raise ValueError("affine_zyx has a degenerate axis")
    directions = linear / lengths
    if not np.allclose(directions.T @ directions, np.eye(3), rtol=1e-5, atol=1e-5):
        raise ValueError("sheared native geometry is unsupported")
    mapping = geometry.get("class_mapping")
    if mapping != CLASS_MAPPING:
        raise ValueError("geometry class_mapping must be the explicit native ontology")
    # Return a JSON-safe normalized copy, so callers never retain mutable input.
    normalized = dict(geometry)
    normalized["shape_zyx"] = [int(x) for x in shape]
    normalized["spacing_zyx_mm"] = [float(x) for x in spacing]
    normalized["spacing_yx_mm"] = [float(x) for x in spacing_yx]
    normalized["affine_zyx"] = affine.tolist()
    normalized["class_mapping"] = dict(CLASS_MAPPING)
    return normalized


def geometry_map(manifest):
    """Return validated patient geometry from an optional manifest section."""
    section = manifest.get("geometry")
    if section is None:
        return {}
    if not isinstance(section, dict) or section.get("schema") != GEOMETRY_MAP_SCHEMA:
        raise ValueError("invalid geometry map schema")
    records = section.get("by_patient")
    if not isinstance(records, dict):
        raise ValueError("geometry map must contain by_patient")
    return {str(pid): validate_geometry(value) for pid, value in records.items()}


class TrainingData:
    def __init__(self, manifest, expected_sha256, *, synthetic=False):
        self.path = Path(manifest).resolve()
        if digest(self.path) != expected_sha256:
            raise ValueError("training manifest hash mismatch")
        self.manifest = json.loads(self.path.read_text())
        m = self.manifest
        if m.get("schema") != "self_audit.data.v2" or m.get("role") != "training":
            raise ValueError("not a v2 training manifest")
        if bool(m.get("synthetic", False)) != synthetic:
            raise ValueError("synthetic/patient identity mismatch")
        self.labeled, self.unlabeled = m["labeled"], m["unlabeled"]
        self.lids, self.uids = [r["id"] for r in self.labeled], [r["id"] for r in self.unlabeled]
        if len(self.lids) < 2 or not self.uids or len(set(self.lids + self.uids)) != len(self.lids + self.uids):
            raise ValueError("nonempty disjoint patient sets and at least two labeled patients required")
        if set(self.lids + self.uids) & set(m.get("evaluation_ids", [])):
            raise ValueError("TRAIN/evaluation patient overlap")
        if m.get("label_budget") != len(self.lids):
            raise ValueError("audit labels must be counted in the label budget")
        if "class_mapping" in m and m["class_mapping"] != CLASS_MAPPING:
            raise ValueError("manifest class_mapping must be the explicit native ontology")
        self.geometry = geometry_map(m)
        valid_ids = set(self.lids + self.uids)
        if not set(self.geometry).issubset(valid_ids):
            raise ValueError("geometry map contains an unknown TRAIN patient")
        for r in self.unlabeled:
            if set(r) != {"id", "image", "image_sha256"}:
                raise ValueError("unlabeled record must contain image only, never reference paths")
        for r in self.labeled:
            if set(r) != {"id", "image", "image_sha256", "masks"} or not r["masks"]:
                raise ValueError("invalid labeled record")
        self.cache, self.label_reads = {}, set()
        hashes = [r["image_sha256"] for r in self.labeled + self.unlabeled]
        if len(set(hashes)) != len(hashes):
            raise ValueError("duplicate TRAIN image content across patient IDs")
        for r in self.labeled + self.unlabeled:
            a = self.array(r["image"], r["image_sha256"])
            if a.ndim != 4 or min(a.shape) < 1 or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
                raise ValueError("image must be finite normalized TZYX")
            if r["id"] in self.geometry:
                self.geometry[r["id"]] = validate_geometry(self.geometry[r["id"]], shape_zyx=a.shape[1:])
            if "masks" in r:
                for t, mask in r["masks"].items():
                    if not 0 <= int(t) < a.shape[0] or set(mask) != {"path", "sha256"}:
                        raise ValueError("invalid labeled frame")
                    # Hash access is allowed only for the budgeted labeled pool.
                    if digest(self.resolve(mask["path"])) != mask["sha256"]:
                        raise ValueError("labeled mask hash mismatch")

    def resolve(self, path):
        p = Path(path)
        return p if p.is_absolute() else self.path.parent / p

    def array(self, path, sha):
        resolved = self.resolve(path)
        key = str(resolved)
        if key not in self.cache:
            if digest(resolved) != sha:
                raise ValueError("data payload hash mismatch")
            self.cache[key] = np.load(resolved, mmap_mode="r", allow_pickle=False)
        return self.cache[key]

    def geometry_for(self, patient_id, *, required=False):
        geometry = self.geometry.get(patient_id)
        if geometry is None and required:
            raise ValueError(f"patient {patient_id!r} has no trustworthy native geometry")
        return None if geometry is None else dict(geometry)

    def spacing_yx_for(self, patient_id, *, required=False):
        geometry = self.geometry_for(patient_id, required=required)
        if geometry is None:
            return None
        return tuple(float(x) for x in geometry["spacing_yx_mm"])

    @staticmethod
    def context(cine, t, z):
        inds = [max(0, z - 1), z, min(cine.shape[1] - 1, z + 1)]
        return torch.from_numpy(np.array(cine[t, inds], dtype=np.float32, copy=True))[None]

    def sample_labeled(self, index, rng, *, foreground_probability=.5):
        r = self.labeled[index]
        cine = self.array(r["image"], r["image_sha256"])
        frame = sorted(r["masks"], key=int)[int(rng.integers(len(r["masks"])))]
        rec = r["masks"][frame]
        mask = self.array(rec["path"], rec["sha256"])
        if mask.shape != cine.shape[1:] or not np.isin(mask, [0, 1, 2, 3, 255]).all():
            raise ValueError("mask shape/ontology mismatch")
        self.label_reads.add(r["id"])
        foreground = np.flatnonzero(((mask > 0) & (mask < 4)).reshape(len(mask), -1).any(1))
        z = int(rng.choice(foreground)) if len(foreground) and rng.random() < foreground_probability else int(rng.integers(len(mask)))
        x = self.context(cine, int(frame), z)
        y = torch.from_numpy(np.array(mask[z], dtype=np.int64, copy=True))[None]
        return x, y, {"patient": r["id"], "frame": int(frame), "slice": z}

    def sample_labeled_with_geometry(self, index, rng, *, foreground_probability=.5, require_geometry=False):
        """Existing sample plus validated geometry; sample_labeled stays unchanged."""
        x, y, meta = self.sample_labeled(index, rng, foreground_probability=foreground_probability)
        return x, y, self.geometry_for(meta["patient"], required=require_geometry), meta

    def sample_unlabeled(self, index, rng):
        r = self.unlabeled[index]
        cine = self.array(r["image"], r["image_sha256"])
        t, z = int(rng.integers(len(cine))), int(rng.integers(cine.shape[1]))
        return self.context(cine, t, z), {"patient": r["id"], "frame": t, "slice": z}

    def sample_unlabeled_with_geometry(self, index, rng, *, require_geometry=False):
        x, meta = self.sample_unlabeled(index, rng)
        return x, self.geometry_for(meta["patient"], required=require_geometry), meta


def synthetic_fixture(folder, seed=11, n_labeled=3, n_unlabeled=3, size=32):
    """Manufactured objects only. Never evidence of cardiac accuracy."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    labeled, unlabeled = [], []
    for j in range(n_labeled + n_unlabeled):
        cy, cx = size * (.5 + rng.uniform(-.08, .08)), size * (.5 + rng.uniform(-.08, .08))
        radius = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        y = np.zeros((size, size), np.uint8)
        y[radius < size * .27] = 2
        y[radius < size * .18] = 3
        y[(yy - cy)**2 + (xx - (cx - size * .29))**2 < (size * .12)**2] = 1
        image = np.clip(np.choose(y, [.1, .8, .45, .9]) + rng.normal(0, .04, y.shape), 0, 1).astype(np.float32)[None, None]
        pid = f"synthetic_{j:03d}"; ip = folder / f"{pid}.npy"; np.save(ip, image)
        r = {"id": pid, "image": ip.name, "image_sha256": digest(ip)}
        if j < n_labeled:
            mp = folder / f"{pid}_mask.npy"; np.save(mp, y[None])
            r["masks"] = {"0": {"path": mp.name, "sha256": digest(mp)}}; labeled.append(r)
        else:
            unlabeled.append(r)
    manifest = folder / "train.json"
    write_json(manifest, {"schema": "self_audit.data.v2", "role": "training", "synthetic": True,
                         "label_budget": n_labeled, "labeled": labeled, "unlabeled": unlabeled, "evaluation_ids": []})
    return manifest, digest(manifest)
