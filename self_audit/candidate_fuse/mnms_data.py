"""Native M&Ms data.v3 reader.

The manifest is metadata only. NIfTI voxels are decoded lazily by ``load``;
images are normalized with the shared preparation transform and held in a
small LRU cache. Source containers and members are checked before decoding.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import struct
import os
import zipfile
from typing import Any, Mapping

import numpy as np
import torch

from ..preparation import normalize_cine, spatial_to_zyx
from .data import Frame, FrameKey, _geometry, _safe_hash, _sha

_SCHEMA = "self_audit.candidate_fuse.data.v3"
_CLASS_MAP = {"0": 0, "1": 3, "2": 2, "3": 1}
_EXPECTED = {"train": 175, "development": 34, "test": 136}
_DEFAULT_CACHE_BYTES = 32 * 1024**3
_MAX_CACHE_BYTES = 1 << 50


def _cache_limit_from_env() -> int:
    raw = os.environ.get("FUSE_MNMS_CACHE_BYTES")
    if raw is None:
        return _DEFAULT_CACHE_BYTES
    try:
        value = int(raw, 10)
    except (TypeError, ValueError) as exc:
        raise ValueError("FUSE_MNMS_CACHE_BYTES must be a nonnegative integer") from exc
    if value < 0 or value > _MAX_CACHE_BYTES:
        raise ValueError("FUSE_MNMS_CACHE_BYTES must be between 0 and 2^50 bytes")
    return value


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_path(value: Mapping[str, Any]) -> Path:
    raw = value.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("source path must be a non-empty string")
    return Path(raw).expanduser().resolve()


def _compressed_member(path: Path, member: str) -> bytes:
    """Return ZIP member's compressed bytes without decoding its NIfTI payload."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        offset = info.header_offset
        compressed_size = info.compress_size
    with path.open("rb") as stream:
        stream.seek(offset)
        header = stream.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError("invalid ZIP local member header")
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        stream.seek(name_len + extra_len, 1)
        payload = stream.read(compressed_size)
        if len(payload) != compressed_size:
            raise ValueError("truncated ZIP member")
        return payload


class MNMSFuseData:
    """FuseData-compatible, lazy reader for role-frozen M&Ms v3 manifests."""

    def __init__(self, manifest_path, expected_sha256=None, *, role, synthetic=False,
                 image_cache_bytes=None):
        if role not in {"train", "development", "test"}:
            raise ValueError("role must be train, development, or test")
        if image_cache_bytes is None:
            image_cache_bytes = _cache_limit_from_env()
        if type(image_cache_bytes) is not int or image_cache_bytes < 0 or image_cache_bytes > _MAX_CACHE_BYTES:
            raise ValueError("image_cache_bytes must be a nonnegative integer no greater than 2^50")
        self.path = Path(manifest_path).resolve()
        self.manifest_sha256 = _sha(self.path)
        if expected_sha256 is not None and self.manifest_sha256 != _safe_hash(expected_sha256, "manifest SHA256"):
            raise ValueError("manifest SHA256 mismatch")
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        m = self.manifest
        if (m.get("schema") != _SCHEMA or m.get("role") != role
                or m.get("dataset_release") != "original_2020_open345"):
            raise ValueError("M&Ms v3 schema, role, or release mismatch")
        base_fields = {"schema", "role", "synthetic", "dataset_release", "class_mapping", "split", "patients",
                       "geometry_receipt", "l8_lock", "mask_mapping_receipt",
                       "source_inventory", "source_inventory_sha256"}
        expected_fields = base_fields | ({"labeled", "unlabeled"} if role == "train" else set())
        if set(m) != expected_fields:
            raise ValueError("M&Ms v3 manifest has unexpected or missing fields")
        if type(m.get("synthetic")) is not bool or m["synthetic"] != bool(synthetic):
            raise ValueError("synthetic/patient identity mismatch")
        if m.get("class_mapping") != _CLASS_MAP:
            raise ValueError("M&Ms class map must be explicit source-label to canonical-label mapping")
        self.role, self.synthetic = role, bool(synthetic)
        self._cache_limit = image_cache_bytes
        self._cache_bytes = 0
        self._container_hashes: dict[tuple[Path, int, int, int], str] = {}
        self._images: OrderedDict[str, np.ndarray] = OrderedDict()
        self._geometry: dict[str, dict[str, Any]] = {}
        self._raw_affines: dict[str, np.ndarray] = {}
        self._records: dict[str, dict[str, Any]] = {}

        split = m.get("split")
        if not isinstance(split, Mapping) or set(split) != {"train", "development", "test"}:
            raise ValueError("v3 manifest must carry the complete official split")
        ids_by_role = {k: split[k] for k in split}
        for split_role, ids in ids_by_role.items():
            if not isinstance(ids, list) or len(ids) != len(set(ids)) or not all(isinstance(x, str) for x in ids):
                raise ValueError("official split IDs must be unique strings")
            if not synthetic and len(ids) != _EXPECTED[split_role]:
                raise ValueError(f"official {split_role} split must contain exactly {_EXPECTED[split_role]} IDs")
        if any(set(ids_by_role[a]) & set(ids_by_role[b]) for a, b in (("train", "development"), ("train", "test"), ("development", "test"))):
            raise ValueError("official M&Ms splits overlap")
        self._split = ids_by_role
        raw_patients = m.get("patients", [])
        if role == "train":
            self._patients = tuple(str(x) for x in raw_patients)
        else:
            self._patients = tuple(str(r.get("id")) for r in raw_patients if isinstance(r, Mapping))
        if not self._patients or len(set(self._patients)) != len(self._patients):
            raise ValueError("role manifest requires unique patients")
        if set(self._patients) != set(ids_by_role[role]):
            raise ValueError("manifest patients differ from official role membership")

        self._validate_receipt(m.get("geometry_receipt"), "geometry receipt")
        self._validate_receipt(m.get("l8_lock"), "L8 lock")
        self._validate_receipt(m.get("mask_mapping_receipt"), "mask mapping receipt")
        if not synthetic:
            self._validate_receipt(m.get("source_inventory"), "source inventory")
        l8 = self._load_l8_lock(m["l8_lock"])
        self._l8_ids = set(l8)
        self._geometry_rows = self._load_geometry_receipt(m["geometry_receipt"])
        self._mask_rows = self._load_mask_mapping(m["mask_mapping_receipt"])
        self._source_rows = self._load_source_inventory(m.get("source_inventory"))

        if role == "train":
            self._labeled = list(m.get("labeled", []))
            self._unlabeled = list(m.get("unlabeled", []))
            l_ids = [str(r.get("id")) for r in self._labeled]
            u_ids = [str(r.get("id")) for r in self._unlabeled]
            if set(u_ids) != set(self._patients) or len(u_ids) != len(self._patients):
                raise ValueError("TRAIN U inventory must contain all official TRAIN patients")
            if set(l_ids) != self._l8_ids or len(l_ids) != len(self._l8_ids):
                raise ValueError("TRAIN mask records must be exactly the locked L8")
            self._labeled_by_id = {r["id"]: r for r in self._labeled}
            self._unlabeled_by_id = {r["id"]: r for r in self._unlabeled}
            for record in self._unlabeled:
                if set(record) != {"id", "image", "frame_count", "phases"}:
                    raise ValueError("U records must contain image metadata only")
                self._validate_image_record(record)
                if "masks" in record or "mask" in record:
                    raise ValueError("U records cannot contain mask references")
            for record in self._labeled:
                if set(record) != {"id", "image", "frame_count", "phases", "masks"}:
                    raise ValueError("L records must contain image and explicit ED/ES masks")
                self._validate_image_record(record)
                self._validate_phases(record)
                masks = record["masks"]
                if not isinstance(masks, Mapping) or set(masks) != {str(v) for v in record["phases"].values()}:
                    raise ValueError("L masks must reference exactly the locked patient's ED and ES frames")
                for frame, source in masks.items():
                    self._validate_source(source, f"mask {record['id']} frame {frame}")
                    mapped = {str(x["frame"]): x for x in self._mask_rows[record["id"]]["masks"]}
                    if frame not in mapped or Path(source["member"]).name != Path(mapped[frame]["mask_path"]).name:
                        raise ValueError(f"L8 mask reference differs from pinned mapping for {record['id']} frame {frame}")
                    if not zipfile.is_zipfile(_source_path(source)) and source["member_sha256"] != mapped[frame]["mask_sha256"]:
                        raise ValueError(f"L8 mask SHA differs from pinned mapping for {record['id']} frame {frame}")
                self._records[record["id"]] = record
        else:
            self._records = {r.get("id"): r for r in m.get("patients", []) if isinstance(r, Mapping)}
            if set(self._records) != set(self._patients):
                raise ValueError("evaluation patient records differ from role membership")
            for record in self._records.values():
                if set(record) != {"id", "image", "frame_count", "phases", "masks"}:
                    raise ValueError("evaluation records require image and phase-mask metadata")
                self._validate_image_record(record)
                self._validate_phases(record)
                masks = record["masks"]
                if not isinstance(masks, Mapping) or set(masks) != {str(v) for v in record["phases"].values()}:
                    raise ValueError("evaluation records require only ED/ES masks")
                for frame, source in masks.items():
                    self._validate_source(source, f"mask {record['id']} frame {frame}")

    @property
    def patients(self):
        return self._patients

    def geometry_for(self, patient_id: str) -> dict[str, Any]:
        """Return adjudicated native geometry without opening image voxels."""
        if patient_id not in self._geometry:
            raise KeyError(patient_id)
        return dict(self._geometry[patient_id])

    def frame_count_for(self, patient_id: str) -> int:
        """Return manifest-pinned cine length without opening image voxels."""
        if patient_id not in self._patients:
            raise KeyError(patient_id)
        record = self._unlabeled_by_id[patient_id] if self.role == "train" else self._records[patient_id]
        return record["frame_count"]

    def _validate_receipt(self, rec, label):
        if not isinstance(rec, Mapping) or set(rec) != {"path", "sha256"}:
            raise ValueError(f"{label} reference must contain path and SHA256")
        path = self._resolve_manifest_path(rec["path"])
        if _sha(path) != _safe_hash(rec["sha256"], f"{label} SHA256"):
            raise ValueError(f"{label} SHA256 mismatch")

    def _resolve_manifest_path(self, value):
        path = Path(value)
        return path if path.is_absolute() else (self.path.parent / path).resolve()

    def _load_geometry_receipt(self, rec):
        receipt = json.loads(self._resolve_manifest_path(rec["path"]).read_text(encoding="utf-8"))
        if receipt.get("schema") != "self_audit.fuse_mnms_geometry_receipt.v1":
            raise ValueError("unexpected adjudicated geometry receipt schema")
        if receipt.get("source_identities_sha256") != self._l8_lock_doc.get("source_identities_sha256"):
            raise ValueError("geometry receipt source identity digest differs from L8 lock")
        rows = {r.get("patient_id"): r for r in receipt.get("records", [])
                if r.get("status") == "GEOMETRY_READY"}
        return rows

    def _load_l8_lock(self, rec):
        lock = json.loads(self._resolve_manifest_path(rec["path"]).read_text(encoding="utf-8"))
        self._l8_lock_doc = lock
        if lock.get("schema") != "self_audit.fuse_mnms_target_l8_lock.v1" or lock.get("seed") != 42:
            raise ValueError("unexpected target L8 lock schema or seed")
        ids = lock.get("selected_ids")
        if (not isinstance(ids, list) or len(ids) != 8 or ids != sorted(set(ids))
                or _safe_hash(lock.get("selected_ids_sha256"), "L8 IDs SHA256") !=
                hashlib.sha256(json.dumps(ids, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()):
            raise ValueError("target L8 IDs or hash invalid")
        if lock.get("mask_allowlist") != ids or lock.get("mask_reads_outside_l8") != 0:
            raise ValueError("target L8 allowlist mismatch")
        if lock.get("geometry_manifest_sha256") != self.manifest["geometry_receipt"]["sha256"]:
            raise ValueError("runtime geometry receipt SHA differs from pinned L8 lock")
        expected_splits = {"train": lock.get("patient_ids_by_split", {}).get("train"),
                           "development": lock.get("patient_ids_by_split", {}).get("val"),
                           "test": lock.get("patient_ids_by_split", {}).get("test")}
        if any(not isinstance(ids, list) or sorted(ids) != sorted(self._split[split])
               for split, ids in expected_splits.items()):
            raise ValueError("official split membership differs from pinned L8 lock")
        if lock.get("source_identities_sha256") != self.manifest.get("source_inventory_sha256"):
            raise ValueError("source inventory identity digest differs from pinned L8 lock")
        if not self.synthetic and _sha(self._resolve_manifest_path(rec["path"])) != "ab239da264645bf82e246b8db6bb55dc5f3314260ef2dcccadc6c9e49bd579f4":
            raise ValueError("real M&Ms v3 must use the pinned target L8 lock")
        return ids

    def _load_mask_mapping(self, rec):
        receipt = json.loads(self._resolve_manifest_path(rec["path"]).read_text(encoding="utf-8"))
        if (receipt.get("schema") != "self_audit.fuse_mnms_l8_mask_mapping_receipt.v1"
                or receipt.get("dataset_release") != "original_2020_open345"
                or receipt.get("source_to_canonical_class_map") != _CLASS_MAP
                or receipt.get("mask_reads_outside_l8") != 0):
            raise ValueError("invalid pinned L8 mask mapping receipt")
        rows = {r.get("patient_id"): r for r in receipt.get("records", [])}
        if set(rows) != self._l8_ids or set(receipt.get("selected_ids", [])) != self._l8_ids:
            raise ValueError("mask mapping receipt differs from locked L8 IDs")
        return rows

    def _load_source_inventory(self, rec):
        if rec is None:
            return {}
        snapshot = json.loads(self._resolve_manifest_path(rec["path"]).read_text(encoding="utf-8"))
        if snapshot.get("schema") != "self_audit.fuse_mnms_open345_snapshot.v1":
            raise ValueError("unexpected source inventory schema")
        if snapshot.get("source_identities_sha256") != self.manifest.get("source_inventory_sha256"):
            raise ValueError("source inventory identity digest mismatch")
        if snapshot.get("source_identities_sha256") != self._l8_lock_doc.get("source_identities_sha256"):
            raise ValueError("source inventory identity digest differs from L8 lock")
        rows = {r.get("patient_id"): r for r in snapshot.get("records", [])}
        all_ids = set().union(*(set(v) for v in self._split.values()))
        if set(rows) != all_ids:
            raise ValueError("source inventory does not match complete pinned split membership")
        for split, ids in (("train", self._split["train"]), ("val", self._split["development"]), ("test", self._split["test"])):
            if {pid for pid, row in rows.items() if row.get("split") == split} != set(ids):
                raise ValueError("source inventory split IDs differ from pinned L8 lock")
        return rows

    def _validate_source(self, source, label):
        fields = {"path", "member", "container_sha256", "member_sha256", "member_size", "crc32"}
        if not isinstance(source, Mapping) or set(source) != fields:
            raise ValueError(f"{label} source identity has invalid fields")
        _safe_hash(source["container_sha256"], f"{label} container SHA256")
        if source["member_sha256"] is None:
            if self.role != "test":
                raise ValueError(f"{label} member SHA256 may be null only for TEST ZIP refs")
        else:
            _safe_hash(source["member_sha256"], f"{label} member SHA256")
        if type(source["member_size"]) is not int or source["member_size"] < 1:
            raise ValueError(f"{label} member size invalid")
        if not isinstance(source["crc32"], str) or len(source["crc32"]) != 8:
            raise ValueError(f"{label} CRC32 invalid")
        path = _source_path(source)
        if path.is_file() and not zipfile.is_zipfile(path):
            if source["member_sha256"] is None:
                raise ValueError(f"{label} null member SHA requires TEST ZIP source")
            if path.stat().st_size != source["member_size"]:
                raise ValueError(f"{label} source file size mismatch")
            return
        if not path.is_file() or not zipfile.is_zipfile(path) or not isinstance(source["member"], str):
            raise ValueError(f"{label} source container/member is unavailable")
        with zipfile.ZipFile(path) as archive:
            try:
                info = archive.getinfo(source["member"])
            except KeyError as exc:
                raise ValueError(f"{label} archive member missing") from exc
            if info.file_size != source["member_size"] or f"{info.CRC:08x}" != source["crc32"].lower():
                raise ValueError(f"{label} archive member size/CRC mismatch")

    def _validate_image_record(self, record):
        pid = record.get("id")
        if not isinstance(pid, str) or not pid:
            raise ValueError("patient ID must be a nonempty string")
        if type(record.get("frame_count")) is not int or record["frame_count"] < 3:
            raise ValueError("cine frame_count must be at least three")
        self._validate_source(record.get("image"), f"image {pid}")
        row = self._geometry_rows.get(pid)
        if row is None:
            raise ValueError(f"no adjudicated native geometry for {pid}")
        # Real geometry receipts adjudicate geometry by patient ID and do not
        # carry source paths. Their source path/member/hash is pinned by the
        # mandatory source snapshot below. Synthetic receipts may include a
        # path, in which case retain the direct cross-check as well.
        if row.get("path") is None and not self._source_rows:
            raise ValueError(f"pathless geometry row requires a pinned source inventory for {pid}")
        if row.get("path") is not None and record["image"].get("member") != row["path"]:
            raise ValueError(f"image path differs from adjudicated source inventory for {pid}")
        if self._source_rows:
            inventory = self._source_rows.get(pid)
            expected_split = {"train": "train", "development": "val", "test": "test"}[self.role]
            if inventory is None or inventory.get("split") != expected_split:
                raise ValueError(f"source inventory role differs for {pid}")
            identity = inventory.get("source_identity", {})
            expected_member = identity.get("member", inventory.get("path"))
            source = record["image"]
            if (source.get("member") != expected_member
                    or source.get("container_sha256") != identity.get("source_sha256")):
                raise ValueError(f"image source identity differs from pinned inventory for {pid}")
            if identity.get("crc32") and source.get("crc32") != identity["crc32"]:
                raise ValueError(f"image source CRC differs from pinned inventory for {pid}")
        self._geometry[pid] = _geometry(row["geometry"])
        phases = record.get("phases")
        if phases is not None:
            self._validate_phases(record)

    def _validate_phases(self, record):
        phases = record.get("phases")
        if not isinstance(phases, Mapping) or set(phases) != {"ED", "ES"}:
            raise ValueError("phase metadata must declare ED and ES")
        if any(type(v) is not int or not 0 <= v < record["frame_count"] for v in phases.values()) or phases["ED"] == phases["ES"]:
            raise ValueError("ED/ES phases must be distinct valid frame indices")

    def _read_member(self, source):
        path = _source_path(source)
        st = path.stat()
        fingerprint = (path, st.st_size, st.st_mtime_ns, getattr(st, "st_ino", 0))
        if zipfile.is_zipfile(path):
            container_sha = self._container_hashes.get(fingerprint)
            if container_sha is None:
                container_sha = _sha(path)
                if container_sha != source["container_sha256"]:
                    raise ValueError("source container SHA256 mismatch before NIfTI decode")
                self._container_hashes[fingerprint] = container_sha
            if source["member_sha256"] is not None:
                compressed = _compressed_member(path, source["member"])
                if _digest_bytes(compressed) != source["member_sha256"]:
                    raise ValueError("source compressed member SHA256 mismatch before NIfTI decode")
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo(source["member"])
                if info.file_size != source["member_size"] or f"{info.CRC:08x}" != source["crc32"].lower():
                    raise ValueError("source member size/CRC mismatch before NIfTI decode")
                raw = archive.read(source["member"])
        else:
            raw = path.read_bytes()
            digest = _digest_bytes(raw)
            if source["member_sha256"] is None or digest != source["container_sha256"] or digest != source["member_sha256"]:
                raise ValueError("source file member SHA256/size mismatch before NIfTI decode")
            if len(raw) != source["member_size"]:
                raise ValueError("source file member size mismatch before NIfTI decode")
            if source["crc32"].lower() != f"{_crc_path(path):08x}":
                raise ValueError("source file member CRC mismatch before NIfTI decode")
            self._container_hashes[fingerprint] = digest
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, getattr(after, "st_ino", 0)) != fingerprint[1:]:
            raise ValueError("source container changed during member verification")
        return raw

    def verify_sources(self) -> dict[str, Any]:
        """Stream-check unique source containers without reading ZIP member payloads."""
        sources = []
        records = self._labeled if self.role == "train" else self._records.values()
        for record in records:
            sources.append(record["image"])
            if self.role != "train":
                sources.extend(record["masks"].values())
        if self.role == "train":
            sources.extend(record["image"] for record in self._unlabeled)
            sources.extend(source for record in self._labeled for source in record["masks"].values())
        checked: dict[Path, tuple[tuple[int, int, int], str]] = {}
        for source in sources:
            path = _source_path(source)
            st = path.stat()
            fingerprint = (st.st_size, st.st_mtime_ns, getattr(st, "st_ino", 0))
            prior = checked.get(path)
            if prior is None:
                digest = _sha(path)
                if digest != source["container_sha256"]:
                    raise ValueError("source container SHA256 mismatch before TEST ledger")
                checked[path] = (fingerprint, digest)
            elif prior[0] != fingerprint:
                raise ValueError("source container changed during metadata verification")
            if zipfile.is_zipfile(path):
                if not isinstance(source.get("member"), str):
                    raise ValueError("ZIP source member path is missing")
                with zipfile.ZipFile(path) as archive:
                    try:
                        info = archive.getinfo(source["member"])
                    except KeyError as exc:
                        raise ValueError("source archive member missing during metadata verification") from exc
                if info.file_size != source["member_size"] or f"{info.CRC:08x}" != source["crc32"].lower():
                    raise ValueError("source member size/CRC mismatch during metadata verification")
        for path, (fingerprint, _) in checked.items():
            st = path.stat()
            if (st.st_size, st.st_mtime_ns, getattr(st, "st_ino", 0)) != fingerprint:
                raise ValueError("source container changed during metadata verification")
        return {"schema": "self_audit.candidate_fuse.source_verification.v1",
                "role": self.role,
                "sources": [{"path": str(path), "sha256": digest,
                             "bytes": fingerprint[0]}
                            for path, (fingerprint, digest) in sorted(checked.items(),
                                                                      key=lambda item: str(item[0]))]}

    def verify_container_sources(self) -> dict[str, Any]:
        """Compatibility name for :meth:`verify_sources`."""
        return self.verify_sources()

    def _image(self, pid):
        if pid in self._images:
            self._images.move_to_end(pid)
            return self._images[pid]
        record = self._unlabeled_by_id[pid] if self.role == "train" else self._records[pid]
        raw = self._read_member(record["image"])
        nii = _load_nifti(raw)
        self._raw_affines[pid] = np.asarray(nii.affine, dtype=np.float64).copy()
        source_xyz = tuple(int(x) for x in nii.shape[:3])
        if len(nii.shape) != 4 or int(nii.shape[3]) != record["frame_count"]:
            raise ValueError("native image shape/frame count mismatch")
        geometry = self._geometry[pid]
        if tuple(reversed(source_xyz)) != tuple(geometry["shape"]):
            raise ValueError("native NIfTI shape differs from adjudicated geometry")
        cine = normalize_cine(np.asarray(nii.dataobj), "XYZT")
        if self._cache_limit and cine.nbytes <= self._cache_limit:
            self._images[pid] = cine
            self._cache_bytes += cine.nbytes
            while self._cache_bytes > self._cache_limit:
                _, evicted = self._images.popitem(last=False)
                self._cache_bytes -= evicted.nbytes
        return cine

    def labeled_keys(self) -> list[FrameKey]:
        if self.role == "train":
            return [(pid, f) for pid in self._patients if pid in self._labeled_by_id
                    for f in sorted(self._labeled_by_id[pid]["phases"].values())]
        return [(pid, f) for pid in self._patients for f in sorted(self._records[pid]["phases"].values())]

    def unlabeled_keys(self) -> list[FrameKey]:
        if self.role != "train":
            raise ValueError("unlabeled_keys are available only for TRAIN")
        result = []
        for pid in self._patients:
            ed_es = set(self._labeled_by_id[pid]["phases"].values()) if pid in self._labeled_by_id else set()
            result.extend((pid, t) for t in range(self._unlabeled_by_id[pid]["frame_count"]) if t not in ed_es)
        return result

    def load(self, key: FrameKey, *, labeled: bool) -> Frame:
        if not isinstance(key, tuple) or len(key) != 2 or key[0] not in self._patients or type(key[1]) is not int:
            raise ValueError("invalid FUSE frame key")
        pid, frame = key
        if self.role == "train":
            if labeled:
                if pid not in self._labeled_by_id or frame not in self._labeled_by_id[pid]["phases"].values():
                    raise ValueError("labeled access is restricted to locked L8 ED/ES masks")
                record = self._labeled_by_id[pid]
            else:
                if frame not in {k[1] for k in self.unlabeled_keys() if k[0] == pid}:
                    raise ValueError("U access cannot include locked ED/ES frames")
                record = self._unlabeled_by_id[pid]
        else:
            if not labeled or frame not in self._records[pid]["phases"].values():
                raise ValueError("evaluation access is limited to labeled ED/ES targets")
            record = self._records[pid]
        cine = self._image(pid)
        image = torch.from_numpy(np.array(cine[frame], dtype=np.float32, copy=True))[None, None]
        target = None
        if labeled:
            source = record["masks"][str(frame)]
            raw = self._read_member(source)
            nii = _load_nifti(raw)
            if not np.allclose(np.asarray(nii.affine, dtype=np.float64), self._raw_affines[pid], rtol=0., atol=1e-5):
                raise ValueError("mask native affine differs from its cine image")
            data = np.asarray(nii.dataobj)
            if data.ndim == 4:
                if frame >= data.shape[3]:
                    raise ValueError("mask frame is outside native source volume")
                data = data[..., frame]
            if data.ndim != 3 or tuple(reversed(data.shape)) != tuple(self._geometry[pid]["shape"]):
                raise ValueError("native mask geometry mismatch")
            mask = spatial_to_zyx(data, "XYZ")
            lut = np.asarray([0, 3, 2, 1], dtype=np.uint8)
            if not np.issubdtype(mask.dtype, np.number) or np.issubdtype(mask.dtype, np.complexfloating):
                raise ValueError("source mask labels must be real numeric values")
            if not np.isfinite(mask).all():
                raise ValueError("source mask contains non-finite labels")
            if not np.equal(mask, np.floor(mask)).all():
                raise ValueError("source mask contains fractional labels")
            if not np.isin(mask, [0, 1, 2, 3]).all():
                raise ValueError("source mask contains labels outside explicit class map")
            # The range check above proves conversion is safe and exact for the
            # supported float32/float64 NIfTI storage commonly used for labels.
            indices = mask.astype(np.uint8, copy=False)
            target = torch.from_numpy(lut[indices].astype(np.int64, copy=False))[None]
        return Frame(image, target, dict(self._geometry[pid]), pid, frame)


def _crc_path(path: Path) -> int:
    import zlib
    value = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value = zlib.crc32(block, value)
    return value & 0xFFFFFFFF


def _load_nifti(raw: bytes):
    import gzip
    import nibabel as nib
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return nib.Nifti1Image.from_bytes(raw)


def _write_synthetic_nifti(path: Path, *, rng, mask: bool, affine: np.ndarray) -> None:
    import nibabel as nib
    if mask:
        x, y, z, t = np.indices((9, 8, 3, 4))
        data = np.zeros((9, 8, 3, 4), dtype=np.uint8)
        data[(x >= 1) & (x < 4) & (y >= 1) & (y < 6)] = 1
        data[(x >= 3) & (x < 6) & (y >= 2) & (y < 6)] = 2
        data[(x >= 5) & (x < 8) & (y >= 1) & (y < 5)] = 3
    else:
        data = rng.normal(size=(9, 8, 3, 4)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _synthetic_file_source(path: Path, member: str) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "member": member,
            "container_sha256": _digest_bytes(raw), "member_sha256": _digest_bytes(raw),
            "member_size": len(raw), "crc32": f"{_crc_path(path):08x}"}


def _synthetic_zip_source(path: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    return {"path": str(path.resolve()), "member": member,
            "container_sha256": _sha(path), "member_sha256": None,
            "member_size": info.file_size, "crc32": f"{info.CRC:08x}"}


def write_synthetic_mnms_manifests(out, *, seed=42):
    """Write tiny native NIfTI/ZIP v3 fixtures for software checks only.

    Produces eight L TRAIN patients, two image-only TRAIN patients, and two
    patients each for development and TEST. It uses the same locked receipt
    cross-links, label map, lazy TEST ZIP source contract, and return shape as
    ``write_synthetic_manifests``; no real patient data is involved.
    """
    root = Path(out).resolve()
    if root.exists():
        raise FileExistsError(f"synthetic output already exists: {root}")
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    affine_xyz = np.diag([1.2, 0.8, 2.5, 1.0])
    geometry = {"shape": [3, 8, 9], "spacing_zyx": [2.5, 0.8, 1.2],
                "affine_zyx": np.diag([2.5, 0.8, 1.2, 1.0]).tolist(), "context": "full_volume"}
    l_ids = [f"mnmsL{i:02d}" for i in range(8)]
    u_ids = [f"mnmsU{i:02d}" for i in range(2)]
    dev_ids = [f"mnmsD{i:02d}" for i in range(2)]
    test_ids = [f"mnmsT{i:02d}" for i in range(2)]
    split = {"train": l_ids + u_ids, "development": dev_ids, "test": test_ids}
    patient_rows: dict[str, dict[str, Any]] = {}
    snapshot_rows = []
    geometry_rows = []
    native_root = root / "native"
    test_zip = root / "test_sources.zip"
    test_members: list[tuple[str, Path, str, str]] = []

    for role, ids, source_split in (("train", split["train"], "train"),
                                    ("development", dev_ids, "val"),
                                    ("test", test_ids, "test")):
        for pid in ids:
            folder = {"train": "Training/Labeled", "development": "Validation", "test": "Testing"}[role]
            image_member = f"{folder}/{pid}/{pid}_sa.nii.gz"
            mask_member = f"{folder}/{pid}/{pid}_sa_gt.nii.gz"
            img_tmp = native_root / image_member
            mask_tmp = native_root / mask_member
            _write_synthetic_nifti(img_tmp, rng=rng, mask=False, affine=affine_xyz)
            _write_synthetic_nifti(mask_tmp, rng=rng, mask=True, affine=affine_xyz)
            geometry_rows.append({"patient_id": pid, "path": image_member, "split": source_split,
                                  "status": "GEOMETRY_READY", "geometry": geometry})
            row = {"id": pid, "frame_count": 4, "phases": {"ED": 0, "ES": 2}}
            if role == "test":
                test_members.append((image_member, img_tmp, pid, "image"))
                test_members.append((mask_member, mask_tmp, pid, "mask"))
            else:
                row["image"] = _synthetic_file_source(img_tmp, image_member)
                row["mask_source"] = _synthetic_file_source(mask_tmp, mask_member)
            patient_rows[pid] = row

    with zipfile.ZipFile(test_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, path, _, _ in test_members:
            archive.write(path, member)
    test_archive_sha = _sha(test_zip)

    for member, _, pid, kind in test_members:
        # Record an image row with archive/member identity; TEST member payload
        # SHA stays null and preparation never reads the member body.
        if kind == "image":
            patient_rows[pid]["image"] = _synthetic_zip_source(test_zip, member)
        else:
            patient_rows[pid]["mask_source"] = _synthetic_zip_source(test_zip, member)

    for role, ids in (("train", split["train"]), ("development", dev_ids), ("test", test_ids)):
        split_name = {"train": "train", "development": "val", "test": "test"}[role]
        for pid in ids:
            row = patient_rows[pid]
            image_ref = row["image"]
            identity = {"patient_id": pid, "split": split_name,
                        "path": image_ref["member"], "source_sha256": image_ref["container_sha256"]}
            if role == "test":
                identity.update(member=image_ref["member"], crc32=image_ref["crc32"])
            snapshot_rows.append({"patient_id": pid, "path": image_ref["member"], "split": split_name,
                                  "shape": [9, 8, 3, 4], "source_identity": identity})

    source_ids = [row["source_identity"] for row in snapshot_rows]
    source_ids_sha = _digest_bytes(json.dumps(source_ids, separators=(",", ":"), sort_keys=True).encode())
    geometry_path = root / "geometry_manifest.json"
    geometry_path.write_text(json.dumps({"schema": "self_audit.fuse_mnms_geometry_receipt.v1",
                                        "dataset_release": "original_2020_open345",
                                        "source_identities_sha256": source_ids_sha,
                                        "records": geometry_rows}, sort_keys=True, indent=2) + "\n")
    geometry_sha = _sha(geometry_path)
    ids_sha = _digest_bytes(json.dumps(l_ids, separators=(",", ":"), ensure_ascii=False).encode())
    lock_path = root / "L8_LOCK.json"
    lock_doc = {"schema": "self_audit.fuse_mnms_target_l8_lock.v1", "seed": 42, "k": 8,
                "selected_ids": l_ids, "selected_ids_sha256": ids_sha, "mask_allowlist": l_ids,
                "mask_reads_outside_l8": 0, "geometry_manifest_sha256": geometry_sha,
                "source_identities_sha256": source_ids_sha,
                "patient_ids_by_split": {"train": split["train"], "val": dev_ids, "test": test_ids}}
    lock_path.write_text(json.dumps(lock_doc, sort_keys=True, indent=2) + "\n")
    mapping_path = root / "MASK_MAPPING_RECEIPT.json"
    mapping_rows = []
    for pid in l_ids:
        mask_ref = patient_rows[pid]["mask_source"]
        mapping_rows.append({"patient_id": pid, "masks": [
            {"phase": phase, "frame": frame, "mask_path": mask_ref["member"],
             "mask_sha256": mask_ref["member_sha256"]}
            for phase, frame in (("ED", 0), ("ES", 2))]})
    mapping_path.write_text(json.dumps({"schema": "self_audit.fuse_mnms_l8_mask_mapping_receipt.v1",
                                        "dataset_release": "original_2020_open345", "selected_ids": l_ids,
                                        "l8_lock_sha256": _sha(lock_path), "mask_reads_outside_l8": 0,
                                        "source_to_canonical_class_map": _CLASS_MAP,
                                        "records": mapping_rows}, sort_keys=True, indent=2) + "\n")
    snapshot_path = root / "source_snapshot.json"
    snapshot_path.write_text(json.dumps({"schema": "self_audit.fuse_mnms_open345_snapshot.v1",
                                        "source_identities_sha256": source_ids_sha,
                                        "records": snapshot_rows}, sort_keys=True, indent=2) + "\n")
    out = {}
    for role in ("train", "development", "test"):
        ids = split[role]
        common = {"schema": _SCHEMA, "role": role, "synthetic": True,
                  "dataset_release": "original_2020_open345", "class_mapping": _CLASS_MAP,
                  "split": split, "geometry_receipt": {"path": str(geometry_path), "sha256": geometry_sha},
                  "l8_lock": {"path": str(lock_path), "sha256": _sha(lock_path)},
                  "mask_mapping_receipt": {"path": str(mapping_path), "sha256": _sha(mapping_path)},
                  "source_inventory": {"path": str(snapshot_path), "sha256": _sha(snapshot_path)},
                  "source_inventory_sha256": source_ids_sha}
        if role == "train":
            labeled = []
            unlabeled = []
            for pid in ids:
                row = patient_rows[pid]
                image_ref = row["image"]
                unlabeled.append({"id": pid, "image": image_ref, "frame_count": 4, "phases": row["phases"]})
                if pid in l_ids:
                    labeled.append({"id": pid, "image": image_ref, "frame_count": 4,
                                    "phases": row["phases"],
                                    "masks": {"0": row["mask_source"], "2": row["mask_source"]}})
            common.update(patients=ids, labeled=labeled, unlabeled=unlabeled)
        else:
            common["patients"] = [{"id": pid, "image": patient_rows[pid]["image"],
                                   "frame_count": 4, "phases": patient_rows[pid]["phases"],
                                   "masks": {"0": patient_rows[pid]["mask_source"],
                                             "2": patient_rows[pid]["mask_source"]}} for pid in ids]
        path = root / f"{role}.json"
        path.write_text(json.dumps(common, sort_keys=True, indent=2) + "\n")
        out[role] = {"path": str(path), "sha256": _sha(path)}
    return out


__all__ = ["MNMSFuseData", "write_synthetic_mnms_manifests"]
