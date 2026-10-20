"""Separate labeled/unlabeled access. U masks are not listed or opened."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .model import digest


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def label_subset(patient_ids, count, seed):
    if len(set(patient_ids)) != len(patient_ids) or not 2 <= count < len(patient_ids):
        raise ValueError("need unique TRAIN IDs and 2 <= label budget < TRAIN patients")
    return sorted(sorted(patient_ids, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())[:count])


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

    def sample_unlabeled(self, index, rng):
        r = self.unlabeled[index]
        cine = self.array(r["image"], r["image_sha256"])
        t, z = int(rng.integers(len(cine))), int(rng.integers(cine.shape[1]))
        return self.context(cine, t, z), {"patient": r["id"], "frame": t, "slice": z}

    def sample_batch(self, index, rng, batch_size, *, labeled):
        """Native slices from one patient: no resizing, padding, or U mask access.

        A transaction uses different supervised/audit patients. For batch>1 the
        audit measure gives equal mass to sampled slices of that audit patient.
        This is a declared batch protocol, not the single-slice experiment.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("positive integer batch_size required")
        sampler = self.sample_labeled if labeled else self.sample_unlabeled
        samples = [sampler(index, rng) for _ in range(batch_size)]
        images = torch.cat([s[0] for s in samples])
        metadata = dict(samples[0][-1])
        if batch_size > 1:
            metadata["samples"] = [s[-1] for s in samples]
        if labeled:
            return images, torch.cat([s[1] for s in samples]), metadata
        return images, metadata


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
