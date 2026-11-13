"""Candidate FUSE data contract and native frame loading.

Schema ``self_audit.candidate_fuse.data.v1``: a train manifest contains
``labeled`` and ``unlabeled`` records with identical TRAIN patient IDs. Each
record carries ``id``, an absolute image path, and SHA256; labeled records also
carry exactly the annotated ED/ES frame map (decimal frame keys to absolute
mask path, SHA256, source NIfTI SHA256, and exact array transpose). TRAIN
``phases`` maps ED/ES to zero-based image frame indices. U records are
image-only. Development/test manifests use role-specific ``patients`` records
with phase maps and two ED/ES masks. All manifests pin the fixed patient split,
per-file source/generated SHA256, class mapping BG/RV/MYO/LV, and canonical
native geometry ``shape`` (D,H,W), ``spacing_zyx`` mm, ``affine_zyx``, and
``context=full_volume`` plus source geometry provenance. No crop/resampling.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from ..candidate_a3d.geometry import VolumeGeometry3D

FrameKey = tuple[str, int]
_ROLES = {"train", "development", "test"}
_CLASSES = {"0": "BG", "1": "RV", "2": "MYO", "3": "LV"}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _safe_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _geometry(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"shape", "spacing_zyx", "affine_zyx", "context"}
    if not isinstance(value, Mapping) or not required.issubset(value) or value.get("context") != "full_volume":
        raise ValueError("geometry must declare native full_volume shape, spacing and affine")
    g = VolumeGeometry3D.from_mapping({k: value[k] for k in required})
    return g.as_dict()


@dataclass(frozen=True)
class Frame:
    image: torch.Tensor
    target: torch.Tensor | None
    geometry: dict[str, Any]
    patient_id: str
    frame_index: int


class FuseData:
    """Hash-pinned, role-checked native volumes with explicit L/U APIs."""

    def __init__(self, manifest_path, expected_sha256=None, *, role, synthetic=False):
        if role not in _ROLES:
            raise ValueError("role must be train, development, or test")
        self.path = Path(manifest_path).resolve()
        self.manifest_sha256 = _sha(self.path)
        if expected_sha256 is not None and self.manifest_sha256 != _safe_hash(expected_sha256, "manifest SHA256"):
            raise ValueError("manifest SHA256 mismatch")
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        m = self.manifest
        schema = m.get("schema")
        if schema not in {"self_audit.candidate_fuse.data.v1", "self_audit.candidate_fuse.data.v2"} or m.get("role") != role:
            raise ValueError("FUSE manifest schema or role mismatch")
        if schema == "self_audit.candidate_fuse.data.v2" and role != "train":
            raise ValueError("FUSE data.v2 is a TRAIN-only semi-supervised manifest")
        if type(m.get("synthetic")) is not bool or m["synthetic"] != bool(synthetic):
            raise ValueError("synthetic/patient identity mismatch")
        if m.get("class_mapping") != _CLASSES:
            raise ValueError("FUSE class mapping must be BG/RV/MYO/LV")
        self.role, self.synthetic = role, bool(synthetic)
        self._schema = schema
        self._images: dict[str, np.ndarray] = {}
        self._geometry: dict[str, dict[str, Any]] = {}
        self._records: dict[str, dict[str, Any]] = {}
        if role == "train":
            self._labeled = list(m.get("labeled", []))
            self._unlabeled = list(m.get("unlabeled", []))
            self._patients = tuple(str(x) for x in m.get("patients", []))
            l_ids = [str(r.get("id")) for r in self._labeled]
            u_ids = [str(r.get("id")) for r in self._unlabeled]
            if schema == "self_audit.candidate_fuse.data.v1":
                if len(l_ids) != 80 or set(l_ids) != set(u_ids) or len(set(l_ids)) != 80:
                    if not synthetic:
                        raise ValueError("real TRAIN requires the same exact 80 patients in L and U")
                    if not l_ids or set(l_ids) != set(u_ids) or len(set(l_ids)) != len(l_ids):
                        raise ValueError("synthetic TRAIN L/U patient IDs must match and be unique")
                self._labeled_ids = set(l_ids)
            else:
                self._labeled_ids = self._validate_labeled_contract(m, l_ids)
                if not self._labeled_ids or not set(l_ids).issubset(set(u_ids)) or len(set(l_ids)) != len(l_ids):
                    raise ValueError("semi-supervised TRAIN requires unique L patients contained in U")
                if not synthetic and len(self._labeled_ids) != 8:
                    raise ValueError("real semi-supervised TRAIN requires the locked L8 budget")
            if len(self._patients) != len(u_ids) or len(set(u_ids)) != len(u_ids):
                raise ValueError("TRAIN patient inventory must equal unique U patient IDs")
            if set(self._patients) != set(u_ids):
                raise ValueError("TRAIN patient inventory must equal U patient IDs")
            for r in self._labeled:
                if set(r) != {"id", "image", "image_sha256", "phases", "masks"}:
                    raise ValueError("labeled record has invalid fields")
            for r in self._unlabeled:
                if set(r) != {"id", "image", "image_sha256"}:
                    raise ValueError("U records must be image-only and contain no mask references")
            self._labeled_by_id = {r["id"]: r for r in self._labeled}
            self._unlabeled_by_id = {r["id"]: r for r in self._unlabeled}
            if schema == "self_audit.candidate_fuse.data.v1":
                for pid in self._patients:
                    lrec = self._labeled_by_id[pid]
                    urec = self._unlabeled_by_id[pid]
                    if lrec["image_sha256"] != urec["image_sha256"] or lrec["image"] != urec["image"]:
                        raise ValueError("L and U must reference identical cine bytes per TRAIN patient")
                    self._check_record(pid, lrec, require_phases=True)
                    self._check_record(pid, urec, require_phases=False)
                    self._check_lu_frames(pid)
            else:
                for pid in self._patients:
                    urec = self._unlabeled_by_id[pid]
                    if pid in self._labeled_by_id:
                        lrec = self._labeled_by_id[pid]
                        if lrec["image_sha256"] != urec["image_sha256"] or lrec["image"] != urec["image"]:
                            raise ValueError("L and U must reference identical cine bytes per TRAIN patient")
                        self._check_record(pid, lrec, require_phases=True)
                        self._check_lu_frames(pid)
                    else:
                        self._check_record(pid, urec, require_phases=False)
        else:
            self._records_list = list(m.get("patients", []))
            ids = [str(r.get("id")) for r in self._records_list]
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("evaluation manifest requires unique patients")
            self._patients = tuple(ids)
            for r in self._records_list:
                if set(r) != {"id", "image", "image_sha256", "phases", "masks"}:
                    raise ValueError("evaluation record must contain image and phase masks")
                self._check_record(str(r["id"]), r, require_phases=True)
                if set(r["masks"]) != {str(v) for v in r["phases"].values()}:
                    raise ValueError("evaluation masks must be exactly ED and ES")
                self._records[str(r["id"])] = r
        split = m.get("split", {})
        if not isinstance(split, Mapping):
            raise ValueError("manifest split identity must be a mapping")
        if (set(split.get("train", [])) & set(split.get("development", []))
                or set(split.get("train", [])) & set(split.get("test", []))
                or set(split.get("development", [])) & set(split.get("test", []))):
            raise ValueError("fixed patient splits overlap")
        expected_sizes = {"train": 80, "development": 18, "test": 50}
        for split_role, expected_size in expected_sizes.items():
            ids = split.get(split_role)
            if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
                raise ValueError(f"locked {split_role} split must contain unique patient IDs")
            if not synthetic and len(ids) != expected_size:
                raise ValueError(f"real locked {split_role} split must contain exactly {expected_size} patients")
        if role == "train" and set(self._patients) != set(split.get("train", [])):
            raise ValueError("TRAIN records differ from locked TRAIN split")
        if role == "development" and set(self._patients) != set(split.get("development", [])):
            raise ValueError("development records differ from locked development split")
        if role == "test" and set(self._patients) != set(split.get("test", [])):
            raise ValueError("test records differ from locked TEST split")

    @property
    def patients(self):
        return self._patients

    def _validate_labeled_contract(self, manifest, labeled_ids):
        """Verify v2's locked L identity and source lineage before payload checks."""
        if not isinstance(manifest.get("dataset"), str) or not manifest["dataset"]:
            raise ValueError("semi-supervised TRAIN must declare its dataset")
        contract = manifest.get("labeled_contract")
        fields = {"patient_ids", "patient_ids_sha256", "source_manifest", "source_manifest_sha256"}
        if not isinstance(contract, Mapping) or set(contract) != fields:
            raise ValueError("v2 labeled_contract has invalid fields")
        ids = contract["patient_ids"]
        if (not isinstance(ids, list) or any(not isinstance(pid, str) or not pid for pid in ids)
                or ids != sorted(set(ids))):
            raise ValueError("locked L patient IDs must be sorted and unique")
        expected_ids_sha = hashlib.sha256(json.dumps(ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        if _safe_hash(contract["patient_ids_sha256"], "locked L patient IDs SHA256") != expected_ids_sha:
            raise ValueError("locked L patient IDs SHA256 mismatch")
        if sorted(labeled_ids) != ids:
            raise ValueError("TRAIN L records differ from locked patient IDs")
        source_path_raw = contract["source_manifest"]
        if not isinstance(source_path_raw, str) or not source_path_raw:
            raise ValueError("locked L source manifest path is required")
        source_path = Path(source_path_raw)
        if not source_path.is_absolute():
            source_path = (self.path.parent / source_path).resolve()
        source_sha = _safe_hash(contract["source_manifest_sha256"], "locked L source manifest SHA256")
        if _sha(source_path) != source_sha:
            raise ValueError("locked L source manifest SHA256 mismatch")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if (source.get("schema") != "self_audit.data.v2" or source.get("role") != "training"
                or source.get("synthetic") is not self.synthetic or source.get("dataset") != manifest["dataset"]):
            raise ValueError("locked L source manifest identity mismatch")
        source_labeled = source.get("labeled")
        if not isinstance(source_labeled, list):
            raise ValueError("locked L source manifest has no labeled records")
        source_ids = [r.get("id") if isinstance(r, Mapping) else None for r in source_labeled]
        if (any(not isinstance(pid, str) or not pid for pid in source_ids)
                or sorted(source_ids) != ids or len(set(source_ids)) != len(source_ids)):
            raise ValueError("locked L source IDs differ from the locked contract")
        if source.get("label_budget") != len(source_labeled):
            raise ValueError("locked L source label_budget differs from its records")
        by_source = {}
        for src in source_labeled:
            if set(src) != {"id", "image", "image_sha256", "masks"}:
                raise ValueError("locked L source record keys differ from the reviewed interface")
            _safe_hash(src["image_sha256"], "locked L source image SHA256")
            if not isinstance(src["image"], str) or not src["image"] or not isinstance(src["masks"], Mapping):
                raise ValueError("locked L source record is malformed")
            for frame, mask in src["masks"].items():
                if not isinstance(frame, str) or not isinstance(mask, Mapping) or set(mask) != {"path", "sha256"}:
                    raise ValueError("locked L source mask keys differ from the reviewed interface")
                if not isinstance(mask["path"], str) or not mask["path"]:
                    raise ValueError("locked L source mask path is malformed")
                _safe_hash(mask["sha256"], "locked L source mask SHA256")
            by_source[src["id"]] = src
        for record in manifest.get("labeled", []):
            src = by_source.get(record.get("id"))
            if src is None or record.get("image_sha256") != src.get("image_sha256"):
                raise ValueError("TRAIN L image identity differs from locked source")
            src_masks = src.get("masks")
            masks = record.get("masks")
            if not isinstance(src_masks, Mapping) or not isinstance(masks, Mapping) or set(src_masks) != set(masks):
                raise ValueError("TRAIN L mask inventory differs from locked source")
            for frame in masks:
                src_mask = src_masks[frame]
                if not isinstance(src_mask, Mapping) or masks[frame].get("sha256") != src_mask.get("sha256"):
                    raise ValueError("TRAIN L mask identity differs from locked source")
        return set(ids)

    def _resolve(self, raw):
        path = Path(raw)
        return path if path.is_absolute() else (self.path.parent / path).resolve()

    def _check_record(self, pid, record, *, require_phases):
        if not isinstance(record.get("id"), str) or not record["id"]:
            raise ValueError("patient ID must be a nonempty string")
        _safe_hash(record.get("image_sha256"), "image SHA256")
        ip = self._resolve(record["image"])
        if _sha(ip) != record["image_sha256"]:
            raise ValueError(f"image SHA256 mismatch for {pid}")
        cine = np.load(ip, mmap_mode="r", allow_pickle=False)
        if cine.ndim != 4 or cine.shape[0] < 1 or not np.isfinite(cine).all() or cine.min() < 0 or cine.max() > 1:
            raise ValueError("FUSE image must be finite normalized [T,D,H,W]")
        geometry = _geometry(self.manifest["geometry"]["by_patient"][pid])
        if tuple(cine.shape[1:]) != tuple(geometry["shape"]):
            raise ValueError("native cine shape differs from declared geometry")
        self._images[pid], self._geometry[pid] = cine, geometry
        if require_phases:
            phases = record.get("phases")
            if not isinstance(phases, Mapping) or set(phases) != {"ED", "ES"}:
                raise ValueError("record must declare distinct ED and ES frame indices")
            if any(type(v) is not int or not 0 <= v < cine.shape[0] for v in phases.values()) or phases["ED"] == phases["ES"]:
                raise ValueError("ED/ES indices must be distinct valid zero-based cine frames")
            masks = record.get("masks")
            if not isinstance(masks, Mapping) or set(masks) != {str(v) for v in phases.values()}:
                raise ValueError("only exact ED and ES targets may be declared")
            for frame, mask_rec in masks.items():
                self._check_mask(pid, int(frame), mask_rec, geometry)
        self._records[pid] = record

    def _check_mask(self, pid, frame, mask_rec, geometry):
        if not isinstance(mask_rec, Mapping) or not {"path", "sha256"}.issubset(mask_rec):
            raise ValueError("mask record must include path and generated SHA256")
        _safe_hash(mask_rec["sha256"], "mask SHA256")
        path = self._resolve(mask_rec["path"])
        if _sha(path) != mask_rec["sha256"]:
            raise ValueError(f"mask SHA256 mismatch for {pid} frame {frame}")
        mask = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(mask.shape) != tuple(geometry["shape"]) or not np.isin(mask, [0, 1, 2, 3]).all():
            raise ValueError("mask must cover native geometry and use labels 0..3")

    def _check_lu_frames(self, pid):
        l = self._labeled_by_id[pid]
        ed_es = set(l["phases"].values())
        if set(int(k) for k in l["masks"]) != ed_es:
            raise ValueError("L masks must be exactly annotated ED/ES frames")
        if len(ed_es) != 2 or len(self._images[pid]) < 3:
            raise ValueError("TRAIN cine requires two ED/ES frames and at least one non-ED/ES frame")

    def labeled_keys(self) -> list[FrameKey]:
        if self.role != "train":
            return [(pid, frame) for pid in self._patients
                    for frame in sorted(self._records[pid]["phases"].values())]
        return [(pid, frame) for pid in self._patients if pid in self._labeled_by_id
                for frame in sorted(self._labeled_by_id[pid]["phases"].values())]

    def unlabeled_keys(self) -> list[FrameKey]:
        if self.role != "train":
            raise ValueError("unlabeled_keys are available only for TRAIN")
        return [(pid, t) for pid in self._patients for t in range(self._images[pid].shape[0])
                if pid not in self._labeled_by_id
                or t not in set(self._labeled_by_id[pid]["phases"].values())]

    def load(self, key: FrameKey, *, labeled: bool) -> Frame:
        if not isinstance(key, tuple) or len(key) != 2 or key[0] not in self._images or type(key[1]) is not int:
            raise ValueError("invalid FUSE frame key")
        pid, frame = key
        cine = self._images[pid]
        if not 0 <= frame < cine.shape[0]:
            raise IndexError("frame index outside cine")
        if self.role != "train":
            if not labeled or frame not in set(self._records[pid]["phases"].values()):
                raise ValueError("evaluation access is limited to labeled ED/ES targets")
            record = self._records[pid]
            entry = record["masks"][str(frame)]
            target = torch.from_numpy(np.array(np.load(self._resolve(entry["path"]), allow_pickle=False), dtype=np.int64, copy=True))[None]
            image = torch.from_numpy(np.array(cine[frame], dtype=np.float32, copy=True))[None, None]
            return Frame(image, target, dict(self._geometry[pid]), pid, frame)
        expected = (set(self._labeled_by_id[pid]["phases"].values())
                    if pid in self._labeled_by_id else set())
        if labeled and frame not in expected:
            raise ValueError("labeled load is restricted to ED/ES keys")
        if not labeled and frame in expected:
            raise ValueError("U load excludes annotated ED/ES frames")
        if not labeled and self.role != "train":
            raise ValueError("image-only U loads are available only for TRAIN")
        x = torch.from_numpy(np.array(cine[frame], dtype=np.float32, copy=True))[None, None]
        target = None
        if labeled:
            entry = self._labeled_by_id[pid]["masks"][str(frame)]
            target = torch.from_numpy(np.array(np.load(self._resolve(entry["path"]), allow_pickle=False), dtype=np.int64, copy=True))[None]
        return Frame(x, target, dict(self._geometry[pid]), pid, frame)


class FuseSampler:
    """Checkpointable seeded permutation streams for teacher and joint epochs."""

    def __init__(self, data: FuseData, *, seed=42):
        if data.role != "train":
            raise ValueError("FuseSampler requires TRAIN data")
        self.data, self.seed = data, int(seed)
        self.py_rng = random.Random(self.seed)
        self.epoch_rng = random.Random(self.seed ^ 0x5F05E)
        self.lkeys, self.ukeys = data.labeled_keys(), data.unlabeled_keys()
        self._stream = []
        self._cursor = 0
        self._l_stream = []
        self._l_cursor = 0
        self._u_patients = []
        self._u_cursor = 0
        self._new_teacher_epoch()
        self._new_patient_epoch()

    @property
    def teacher_epoch_size(self):
        return len(self.lkeys)

    @property
    def joint_epoch_size(self):
        return len(self.data.patients)

    def _new_teacher_epoch(self):
        self._stream = list(self.lkeys); self.py_rng.shuffle(self._stream); self._cursor = 0

    def _new_patient_epoch(self):
        self._u_patients = list(self.data.patients); self.epoch_rng.shuffle(self._u_patients); self._u_cursor = 0

    def next_labeled(self):
        if self._cursor == len(self._stream): self._new_teacher_epoch()
        key = self._stream[self._cursor]; self._cursor += 1
        return key

    def next_labeled_batch(self, batch_size=2):
        """Consume up to two same-patient L frames from this epoch's stream."""
        if type(batch_size) is not int or batch_size not in (1, 2):
            raise ValueError("teacher batch size must be integer 1 or 2")
        first = self.next_labeled()
        if batch_size == 1:
            return [first]
        # Do not wrap the permutation: epoch membership is a strict boundary.
        match = next((i for i in range(self._cursor, len(self._stream))
                      if self._stream[i][0] == first[0]), None)
        if match is None:
            return [first]
        key = self._stream.pop(match)
        self._stream.insert(self._cursor, key)
        self._cursor += 1
        return [first, key]

    def next_pair(self):
        if self._u_cursor == len(self._u_patients): self._new_patient_epoch()
        pid = self._u_patients[self._u_cursor]; self._u_cursor += 1
        if self._l_cursor == len(self._l_stream):
            self._l_stream = list(self.lkeys); self.py_rng.shuffle(self._l_stream); self._l_cursor = 0
        lkey = self._l_stream[self._l_cursor]; self._l_cursor += 1
        u_candidates = [key for key in self.ukeys if key[0] == pid]
        ukey = u_candidates[self.epoch_rng.randrange(len(u_candidates))]
        return lkey, ukey

    def state_dict(self):
        return {"seed": self.seed, "py_rng": self.py_rng.getstate(), "epoch_rng": self.epoch_rng.getstate(),
                "teacher_stream": self._stream, "teacher_cursor": self._cursor,
                "l_stream": self._l_stream, "l_cursor": self._l_cursor,
                "u_patients": self._u_patients, "u_cursor": self._u_cursor}

    def load_state_dict(self, state):
        if int(state["seed"]) != self.seed: raise ValueError("sampler seed mismatch")
        self.py_rng.setstate(state["py_rng"]); self.epoch_rng.setstate(state["epoch_rng"])
        for name, expected in (("teacher_stream", set(self.lkeys)),
                               ("u_patients", set(self.data.patients))):
            values = state[name]
            normalized = [tuple(v) if name != "u_patients" else v for v in values]
            if len(normalized) != len(expected) or set(normalized) != expected:
                raise ValueError("sampler checkpoint permutation differs from dataset")
        l_values = state["l_stream"]
        l_cursor = int(state["l_cursor"])
        if l_values:
            normalized_l = [tuple(v) for v in l_values]
            if len(normalized_l) != len(self.lkeys) or set(normalized_l) != set(self.lkeys):
                raise ValueError("sampler checkpoint permutation differs from dataset")
        elif l_cursor != 0 or int(state["u_cursor"]) != 0:
            raise ValueError("empty initial joint stream requires labeled and patient cursors zero")
        self._stream = [tuple(v) for v in state["teacher_stream"]]
        self._cursor = int(state["teacher_cursor"])
        self._l_stream = [tuple(v) for v in state["l_stream"]]
        self._l_cursor = l_cursor
        self._u_patients = list(state["u_patients"]); self._u_cursor = int(state["u_cursor"])
        if not 0 <= self._cursor <= len(self._stream) or not 0 <= self._l_cursor <= len(self._l_stream) or not 0 <= self._u_cursor <= len(self._u_patients):
            raise ValueError("invalid sampler cursor")
        # The outstanding patient permutation must match the current epoch;
        # stream contents are unique permutations, not just matching sets.
        if (len(self._stream) != len(set(self._stream)) or len(self._l_stream) != len(set(self._l_stream))
                or len(self._u_patients) != len(set(self._u_patients))):
            raise ValueError("sampler checkpoint contains duplicate permutation entries")


def write_synthetic_manifests(out, *, seed=42):
    """Materialize small TRAIN/DEV/TEST native fixtures; software evidence only."""
    root = Path(out).resolve()
    if root.exists(): raise FileExistsError(f"synthetic output already exists: {root}")
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    d, h, w, nframes = 3, 8, 9, 4
    geometry = {"shape": [d, h, w], "spacing_zyx": [2.5, .7, 1.3],
                "affine_zyx": np.diag([2.5, .7, 1.3, 1.]).tolist(), "context": "full_volume",
                "provenance": {"synthetic": True, "seed": seed}}
    outputs = {}
    for role, patients in (("train", ["train001", "train002"]), ("development", ["dev001"]), ("test", ["test001"])):
        records, lrecords, urecords = [], [], []
        for pid in patients:
            cine = rng.random((nframes, d, h, w), dtype=np.float32)
            ip = root / f"{pid}.npy"; np.save(ip, cine); ih = _sha(ip)
            phases = {"ED": 0, "ES": 2}
            masks = {}
            for frame in phases.values():
                mask = np.zeros((d, h, w), dtype=np.uint8); mask[:, 2:6, 2:7] = 2; mask[:, 3:5, 3:6] = 3
                mp = root / f"{pid}_t{frame:03d}_mask.npy"; np.save(mp, mask)
                masks[str(frame)] = {"path": str(mp), "sha256": _sha(mp), "source_sha256": "synthetic", "transpose": [2, 1, 0]}
            rec = {"id": pid, "image": str(ip), "image_sha256": ih, "phases": phases, "masks": masks}
            if role == "train":
                lrecords.append(rec); urecords.append({"id": pid, "image": str(ip), "image_sha256": ih})
            else: records.append(rec)
        manifest = {"schema": "self_audit.candidate_fuse.data.v1", "role": role, "synthetic": True,
                    "dataset": "synthetic", "class_mapping": _CLASSES,
                    "split": {"train": ["train001", "train002"], "development": ["dev001"], "test": ["test001"]},
                    "geometry": {"by_patient": {pid: geometry for pid in patients}}}
        if role == "train":
            manifest.update(patients=patients, labeled=lrecords, unlabeled=urecords)
        else: manifest["patients"] = records
        path = root / f"{role}.json"; path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        outputs[role] = {"path": str(path), "sha256": _sha(path)}
    return outputs


__all__ = ["Frame", "FrameKey", "FuseData", "FuseSampler", "write_synthetic_manifests"]
