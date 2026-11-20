"""SSL4MIS-split data views for the PG engine: L7 / U63 (ED+ES only) / val10 / test20.

``Ssl4misData`` subclasses ``FuseData`` only so ``PGSampler``/``PGData`` accept it; it does not run the FUSE
80/18/50-specific constructor.  Every image and mask is checked against its SHA256, shapes and label range.
Training objects can only ever load L7 and U63 frames: val10 and test20 live in separate objects.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from self_audit.candidate_fuse.data import Frame, FrameKey, FuseData, _CLASSES, _geometry

from .data import PGData

SCHEMA = "self_audit.ssl4mis.data.v1"
LOCK_SCHEMA = "self_audit.ssl4mis_split_lock.v1"
RECEIPT_SCHEMA = "self_audit.ssl4mis.materialization.v1"
EXPECTED = {"labeled": 7, "unlabeled": 63, "val": 10, "test": 20}
_STAGES = (0, 1)  # stored frame index: 0 = ED, 1 = ES


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Ssl4misData(FuseData):
    def __init__(self, manifest_path: str | Path, expected_sha256: str | None, *, role: str) -> None:  # noqa: D107 (no super().__init__)
        if role not in {"train", "val", "test"}:
            raise ValueError("role must be train, val or test")
        self.path = Path(manifest_path).resolve()
        self.manifest_sha256 = _sha(self.path)
        if expected_sha256 is not None and self.manifest_sha256 != expected_sha256:
            raise ValueError("SSL4MIS manifest SHA256 mismatch")
        self.manifest = m = json.loads(self.path.read_text(encoding="utf-8"))
        if m.get("schema") != SCHEMA or m.get("role") != role or m.get("synthetic") is not False or m.get("class_mapping") != _CLASSES:
            raise ValueError("SSL4MIS manifest schema, role, synthetic flag or class mapping mismatch")
        self.role, self.synthetic = role, False
        self._schema = SCHEMA
        self._images: dict[str, np.ndarray] = {}
        self._geometry: dict[str, dict[str, Any]] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._labeled_by_id: dict[str, dict[str, Any]] = {}
        if role == "train":
            labeled, unlabeled = list(m["labeled"]), list(m["unlabeled"])
            self._labeled_ids = {r["id"] for r in labeled}
            u_ids = [r["id"] for r in unlabeled]
            if len(self._labeled_ids) != EXPECTED["labeled"] or len(u_ids) != EXPECTED["unlabeled"] or len(set(u_ids)) != len(u_ids):
                raise ValueError("SSL4MIS TRAIN must hold exactly L7 and U63")
            if self._labeled_ids & set(u_ids) or set(m["patients"]) != set(u_ids):
                raise ValueError("SSL4MIS TRAIN L and U must be disjoint and patients must equal U63")
            self._patients = tuple(sorted(u_ids))
            for rec in labeled:
                if set(rec) != {"id", "image", "image_sha256", "phases", "masks"}:
                    raise ValueError("labeled record has invalid fields")
                self._check(rec, labeled=True)
                self._labeled_by_id[rec["id"]] = rec
            for rec in unlabeled:
                if set(rec) != {"id", "image", "image_sha256"}:
                    raise ValueError("U records must be image-only")
                self._check(rec, labeled=False)
                self._records[rec["id"]] = rec
        else:
            records = list(m["patients"])
            if len(records) != EXPECTED[role] or len({r["id"] for r in records}) != len(records):
                raise ValueError(f"SSL4MIS {role} must hold exactly {EXPECTED[role]} unique patients")
            self._patients = tuple(sorted(r["id"] for r in records))
            self._labeled_ids = set()
            for rec in records:
                if set(rec) != {"id", "image", "image_sha256", "phases", "masks"}:
                    raise ValueError("evaluation record has invalid fields")
                self._check(rec, labeled=True)
                self._records[rec["id"]] = rec

    def _check(self, rec: Mapping[str, Any], *, labeled: bool) -> None:
        pid = rec["id"]
        image_path = Path(rec["image"])
        if _sha(image_path) != rec["image_sha256"]:
            raise ValueError(f"image SHA256 mismatch for {pid}")
        cine = np.load(image_path, mmap_mode="r", allow_pickle=False)
        geometry = _geometry(self.manifest["geometry"]["by_patient"][pid])
        if cine.ndim != 4 or cine.shape[0] != 2 or tuple(cine.shape[1:]) != tuple(geometry["shape"]):
            raise ValueError(f"{pid}: expected [2,D,H,W] ED/ES stack matching the native geometry")
        if not np.isfinite(cine).all() or cine.min() < 0 or cine.max() > 1:
            raise ValueError(f"{pid}: image must be finite and normalized to [0,1]")
        self._images[pid], self._geometry[pid] = cine, geometry
        if labeled:
            if rec["phases"] != {"ED": 0, "ES": 1} or set(rec["masks"]) != {"0", "1"}:
                raise ValueError(f"{pid}: labeled record must declare exactly the ED/ES targets")
            for frame, mask in rec["masks"].items():
                mask_path = Path(mask["path"])
                if _sha(mask_path) != mask["sha256"]:
                    raise ValueError(f"mask SHA256 mismatch for {pid} frame {frame}")
                labels = np.load(mask_path, mmap_mode="r", allow_pickle=False)
                if tuple(labels.shape) != tuple(geometry["shape"]) or not np.isin(labels, (0, 1, 2, 3)).all():
                    raise ValueError(f"{pid}: mask must cover native geometry with labels 0..3")

    @property
    def patients(self) -> tuple[str, ...]:
        return self._patients

    def labeled_keys(self) -> list[FrameKey]:
        if self.role == "train":
            return [(pid, t) for pid in sorted(self._labeled_by_id) for t in _STAGES]
        return [(pid, t) for pid in self._patients for t in _STAGES]

    def unlabeled_keys(self) -> list[FrameKey]:
        if self.role != "train":
            raise ValueError("unlabeled_keys are available only for TRAIN")
        return [(pid, t) for pid in self._patients for t in _STAGES]

    def load(self, key: FrameKey, *, labeled: bool) -> Frame:
        if not isinstance(key, tuple) or len(key) != 2 or key[1] not in _STAGES:
            raise ValueError("invalid SSL4MIS frame key")
        pid, t = key
        if self.role == "train":
            if labeled:
                if pid not in self._labeled_by_id:
                    raise ValueError("labeled load is restricted to L7")
                record = self._labeled_by_id[pid]
            else:
                if pid not in self._records:
                    raise ValueError("U load is restricted to U63")
                record = self._records[pid]
        else:
            if not labeled or pid not in self._records:
                raise ValueError("evaluation access is limited to labeled val/test ED/ES targets")
            record = self._records[pid]
        image = torch.from_numpy(np.array(self._images[pid][t], dtype=np.float32, copy=True))[None, None]
        target = None
        if labeled:
            entry = record["masks"][str(t)]
            target = torch.from_numpy(np.array(np.load(entry["path"], allow_pickle=False), dtype=np.int64, copy=True))[None]
        return Frame(image, target, dict(self._geometry[pid]), pid, t)


def open_ssl4mis_data(config: Mapping[str, Any]) -> PGData:
    data = config["data"]
    ref = data["ssl4mis"]
    lock_path, receipt_path = Path(ref["lock"]["path"]), Path(ref["receipt"]["path"])
    if _sha(lock_path) != ref["lock"]["sha256"] or _sha(receipt_path) != ref["receipt"]["sha256"]:
        raise ValueError("SSL4MIS lock or preparation receipt SHA256 mismatch")
    lock = json.loads(lock_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if lock.get("schema") != LOCK_SCHEMA or receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != "READY":
        raise ValueError("SSL4MIS lock or receipt is not READY")
    if receipt["lock"]["sha256"] != ref["lock"]["sha256"]:
        raise ValueError("preparation receipt was made from a different split lock")
    ids = receipt["split_ids"]
    if (ids["train_l"], ids["train_u"], ids["val"], ids["test"]) != (lock["labeled"]["patients"], lock["unlabeled"]["patients"],
                                                                     lock["val"]["patients"], lock["test"]["patients"]):
        raise ValueError("preparation receipt patient sets differ from the split lock")
    views = {}
    for role in ("train", "val", "test"):
        pinned = data["manifests"][role]
        if pinned != receipt["manifests"][role]:
            raise ValueError(f"{role} manifest in the config is not the receipt-pinned materialization")
        views[role] = Ssl4misData(pinned["path"], pinned["sha256"], role=role)
    return PGData(train=views["train"], val=views["val"], test=views["test"])


__all__ = ["Ssl4misData", "open_ssl4mis_data"]
