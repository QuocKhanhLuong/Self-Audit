"""Raw-image-only preparation, file-read firewall and native-grid reconstruction."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReadFirewall:
    """Audit actual opens while active. Method input mirror contains images only.

    This does not by itself prove historical preprocessing cleanliness; raw image
    hashes/header and the exact preprocessing transform are recorded separately.
    """
    def __init__(self):
        self.active = False
        self.reads: set[str] = set()
        self.denied: list[str] = []
        sys.addaudithook(self._audit)

    def _audit(self, event, args):
        if not self.active or event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = str(args[0])
        mode = args[1]
        write_only = isinstance(mode, str) and ("w" in mode or "a" in mode) and "+" not in mode
        if write_only:
            return
        # Input contract has no ground-truth files, regardless of file extension.
        name = Path(path).name.lower()
        source_code = name.endswith((".py", ".pyc", ".so", ".dylib"))
        if not source_code and re.search(r"(^|[_\-.])(gt|mask|masks|reference|references|label_true)([_\-.]|$)", name):
            self.denied.append(path)
            raise PermissionError("Reference input forbidden in method process: " + path)
        self.reads.add(path)

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *exc):
        self.active = False

    def save(self, path: Path):
        path.write_text(json.dumps({"reads": sorted(self.reads), "denied": self.denied}, indent=2))


def prepare_images(raw: Path, split_path: Path, out: Path, size: int = 224) -> dict:
    """Full FOV, no label reads; all available frame images and every z slice."""
    split = json.loads(split_path.read_text())
    train = set(split["train_patients"])
    dev = set(split["val_patients"])
    if train & dev:
        raise ValueError("Patient split overlap")
    out.mkdir(parents=True, exist_ok=False)
    mirror = out / "images_only"
    mirror.mkdir()
    arrays, supports, units, volumes = [], [], [], []
    seen = set()
    # Full cine is not locally available; explicitly frame image names only.
    for source in sorted(raw.rglob("*.nii*")):
        if not re.fullmatch(r"patient\d+_frame\d+\.nii(?:\.gz)?", source.name):
            continue
        patient = source.name.split("_")[0]
        volume_id = source.name.split(".nii")[0]
        if patient not in train | dev:
            raise ValueError("Unassigned patient: " + patient)
        destination = mirror / source.name
        # Copy compressed raw IMAGE bytes, no paired reference or info-file reads.
        destination.write_bytes(source.read_bytes())
        image = nib.load(str(destination))
        volume = image.get_fdata(dtype=np.float32)
        if volume.ndim != 3 or not np.isfinite(volume).all():
            raise ValueError("Expected finite raw 3D frame: " + source.name)
        lo, hi = np.percentile(volume, [1, 99])
        volume = np.clip((volume - lo) / max(float(hi - lo), 1e-6), 0, 1)
        h, w, d = volume.shape
        scale = size / max(h, w)
        rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
        top, left = (size - rh) // 2, (size - rw) // 2
        resized = F.interpolate(torch.from_numpy(volume.transpose(2, 0, 1).copy())[:, None],
                                (rh, rw), mode="bilinear", align_corners=False)[:, 0].numpy()
        first = len(arrays)
        for z in range(d):
            x = np.zeros((size, size), dtype=np.float32)
            support = np.zeros((size, size), dtype=bool)
            x[top:top + rh, left:left + rw] = resized[z]
            support[top:top + rh, left:left + rw] = True
            arrays.append(x)
            supports.append(support)
            units.append({"patient": patient, "volume": volume_id, "z": z,
                          "split": "train" if patient in train else "dev",
                          "box": [top, left, rh, rw], "index": len(units)})
        seen.add(patient)
        volumes.append({"id": volume_id, "patient": patient, "filename": source.name,
                        "raw_image_sha256": sha256(source), "mirror_sha256": sha256(destination),
                        "native_shape": [h, w, d], "affine": image.affine.tolist(),
                        "zooms": list(map(float, image.header.get_zooms())),
                        "first": first, "count": d, "box": [top, left, rh, rw],
                        "scale": scale, "normalization_percentiles": [float(lo), float(hi)]})
    if seen != train | dev:
        raise ValueError("Missing patients: " + str(sorted((train | dev) - seen)))
    np.save(out / "images.npy", np.stack(arrays))
    np.save(out / "supports.npy", np.stack(supports))
    (out / "patient_split.json").write_bytes(split_path.read_bytes())
    manifest = {"schema": "nogt.v1", "size": size, "units": units, "volumes": volumes,
                "split_sha256": sha256(split_path), "train_patients": sorted(train),
                "development_patients": sorted(dev), "preprocessing": "raw image percentiles 1/99; full FOV resize/pad; every slice",
                "manual_masks_used": False, "selection": "all available ED/ES frame images; source release selects ED/ES",
                "test_status": "development labels used in earlier research; not untouched test"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def crop(array: np.ndarray, box: list[int]) -> np.ndarray:
    top, left, h, w = box
    return array[top:top + h, left:left + w]


def native_volume(padded: np.ndarray, volume: dict) -> np.ndarray:
    top, left, h, w = volume["box"]
    native_h, native_w, _ = volume["native_shape"]
    x = padded[:, top:top + h, left:left + w]
    result = F.interpolate(torch.from_numpy(x.astype(np.float32))[:, None],
                           (native_h, native_w), mode="nearest")[:, 0].numpy()
    return result.transpose(1, 2, 0).astype(np.uint8)
