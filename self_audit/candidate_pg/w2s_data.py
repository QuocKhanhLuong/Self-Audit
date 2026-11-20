"""W2S-Match VAL20 view: the receipt-validated VAL18 loader plus two documented duplicate patients.

``FuseData`` and the PG materialization validator hard-code VAL18, so the two extra patients
(patient055, patient076) are served by a thin wrapper that delegates everything else to the
VAL18 loader.  The extra manifest is hash-pinned by ``config["data"]["w2s_val20"]`` and every
image and mask is checked against its recorded SHA256 before use.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from self_audit.candidate_fuse.data import Frame, _geometry


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Val20Data:
    """VAL18 loader plus the extra duplicate patients; selection and monitoring see 20 patients."""

    def __init__(self, base: Any, manifest_path: Path, expected_sha: str) -> None:
        if _sha(manifest_path) != expected_sha:
            raise ValueError("W2S VAL20 manifest SHA256 mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta = manifest.get("w2s_val20") or {}
        added = list(meta.get("added_patients", ()))
        if not added or meta.get("base_manifest_sha256") != getattr(base, "manifest_sha256", None):
            raise ValueError("W2S VAL20 manifest is not an extension of the opened VAL18 manifest")
        records = {r["id"]: r for r in manifest["patients"]}
        if set(base.patients) & set(added) or set(base.patients) | set(added) != set(records):
            raise ValueError("W2S VAL20 patient identity differs from VAL18 + added patients")
        self._base, self._frames, self._patients = base, {}, tuple(sorted(records))
        self._extra_patients = tuple(sorted(added))
        for pid in self._extra_patients:
            record = records[pid]
            image_path = Path(record["image"])
            if _sha(image_path) != record["image_sha256"]:
                raise ValueError(f"W2S VAL20 image hash mismatch for {pid}")
            cine = np.load(image_path, mmap_mode="r", allow_pickle=False)
            geometry = _geometry(manifest["geometry"]["by_patient"][pid])
            for frame in sorted(int(v) for v in record["phases"].values()):
                entry = record["masks"][str(frame)]
                mask_path = Path(entry["path"])
                if _sha(mask_path) != entry["sha256"]:
                    raise ValueError(f"W2S VAL20 mask hash mismatch for {pid} frame {frame}")
                target = torch.from_numpy(np.array(np.load(mask_path, allow_pickle=False), dtype=np.int64, copy=True))[None]
                image = torch.from_numpy(np.array(cine[frame], dtype=np.float32, copy=True))[None, None]
                if tuple(image.shape[-3:]) != tuple(target.shape[-3:]):
                    raise ValueError(f"W2S VAL20 image/mask shape mismatch for {pid}")
                self._frames[(pid, frame)] = Frame(image, target, dict(geometry), pid, frame)

    def __getattr__(self, name: str) -> Any:  # everything not overridden comes from the VAL18 loader
        return getattr(self._base, name)

    @property
    def patients(self) -> tuple[str, ...]:
        return self._patients

    @property
    def extra_patients(self) -> tuple[str, ...]:
        return self._extra_patients

    def labeled_keys(self) -> list[tuple[str, int]]:
        return list(self._base.labeled_keys()) + sorted(self._frames)

    def load(self, key: tuple[str, int], *, labeled: bool) -> Frame:
        if key in self._frames:
            if not labeled:
                raise ValueError("evaluation access is limited to labeled ED/ES targets")
            frame = self._frames[key]
            return Frame(frame.image.clone(), frame.target.clone(), dict(frame.geometry), frame.patient_id, frame.frame_index)
        return self._base.load(key, labeled=labeled)


def with_val20(bundle: Any, config: Mapping[str, Any]) -> Any:
    """Return ``bundle`` with ``val`` replaced by the VAL20 view when ``data.w2s_val20`` is configured."""

    ref = (config.get("data") or {}).get("w2s_val20")
    if not ref:
        return bundle
    view = Val20Data(bundle.val, Path(ref["path"]), ref["sha256"])
    return dataclasses.replace(bundle, val=view)


__all__ = ["Val20Data", "with_val20"]
