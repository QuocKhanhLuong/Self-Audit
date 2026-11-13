#!/usr/bin/env python3
"""Lock the M&Ms TRAIN L8 IDs, then validate only those eight ED/ES masks.

This tool is a metadata/sample check. It does not prepare voxel arrays or
launch training. VAL and TEST masks are never opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from self_audit.data import label_subset
from self_audit.candidate_a3d.geometry import canonical_geometry


_RELEASE = "original_2020_open345"
_GEOMETRY_SCHEMA = "self_audit.fuse_mnms_geometry_receipt.v1"
_SUMMARY_SCHEMA = "self_audit.fuse_mnms_open345_adjudication.v1"
_CLASS_MAP = {0: 0, 1: 3, 2: 2, 3: 1}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_INDEX_COUNTS = {"train": 175, "val": 34}
_RELEASE_COUNTS = {"train": 175, "val": 34, "test": 136}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _compact_ids_sha(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _partition(value: str) -> tuple[str, str | None]:
    key = (value or "").strip().lower().replace("\\", "/")
    if key in {"train", "training"}:
        return "train", None
    if key in {"training/labeled", "training/labeled/"}:
        return "train", "labeled"
    if key in {"training/unlabeled", "training/unlabeled/"}:
        return "train", "unlabeled"
    if key in {"val", "valid", "validation"}:
        return "val", None
    if key in {"test", "testing"}:
        return "test", None
    raise ValueError(f"unknown source_partition {value!r}")


def _safe_relative(raw: str, field: str) -> Path:
    posix = PurePosixPath(raw.replace("\\", "/"))
    if not raw or posix.is_absolute() or ".." in posix.parts or "." in posix.parts:
        raise ValueError(f"{field} must be a safe relative path")
    return Path(*posix.parts)


def _read_index(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"patient_id", "split", "image", "ed_frame", "es_frame", "source_partition"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"index must include {sorted(required)}")
        rows = list(reader)
    result = []
    seen = set()
    for raw in rows:
        pid = (raw.get("patient_id") or "").strip()
        split = (raw.get("split") or "").strip().lower()
        part_split, partition_subtype = _partition(raw.get("source_partition", ""))
        if not pid or pid in seen:
            raise ValueError(f"missing or duplicate patient ID {pid!r} in index")
        if split not in _INDEX_COUNTS or split != part_split:
            raise ValueError(f"index split/source_partition mismatch for {pid}")
        image = _safe_relative((raw.get("image") or "").strip(), "image path")
        phases = {}
        for key in ("ed_frame", "es_frame"):
            value = (raw.get(key) or "").strip()
            if not value:
                phases[key] = None
            elif value.isdecimal():
                phases[key] = int(value)
            else:
                raise ValueError(f"{key} must be a nonnegative integer for {pid}")
        if (phases["ed_frame"] is not None and phases["es_frame"] is not None
                and phases["ed_frame"] == phases["es_frame"]):
            raise ValueError(f"ED and ES frame indices must differ for {pid}")
        both_known = phases["ed_frame"] is not None and phases["es_frame"] is not None
        both_blank = phases["ed_frame"] is None and phases["es_frame"] is None
        if (partition_subtype == "unlabeled" and not (both_known or both_blank)) or (
                partition_subtype != "unlabeled" and not both_known):
            raise ValueError(f"only Training/Unlabeled rows may omit both ED/ES indices ({pid})")
        masks = {}
        for phase in ("ed", "es"):
            raw_mask = (raw.get(f"mask_{phase}") or raw.get("mask") or "").strip()
            masks[phase] = _safe_relative(raw_mask, f"{phase.upper()} mask path") if raw_mask else None
        seen.add(pid)
        result.append({"patient_id": pid, "split": split, "source_partition": raw["source_partition"].strip(),
                       "partition_subtype": partition_subtype, "image": image,
                       "ed_frame": phases["ed_frame"], "es_frame": phases["es_frame"], "masks": masks})
    counts = {split: sum(row["split"] == split for row in result) for split in _INDEX_COUNTS}
    if counts != _INDEX_COUNTS:
        raise ValueError(f"source CSV must contain TRAIN/VAL only with counts {_INDEX_COUNTS}; got {counts}")
    return result


def _load_geometry_evidence(index_path: Path, geometry_report_dir: Path, rows: list[dict[str, Any]]):
    summary_path = geometry_report_dir / "SUMMARY.json"
    manifest_path = geometry_report_dir / "geometry_manifest.json"
    patients_path = geometry_report_dir / "per_patient.json"
    snapshot_path = geometry_report_dir / "source_snapshot.json"
    if not all(path.is_file() for path in (summary_path, manifest_path, patients_path, snapshot_path)):
        raise ValueError("geometry report must contain SUMMARY.json, geometry_manifest.json, per_patient.json and source_snapshot.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if summary.get("schema") != _SUMMARY_SCHEMA or summary.get("status") != "GEOMETRY_READY":
        raise ValueError("geometry summary must be a successful open345 adjudication")
    if summary.get("dataset_release") != _RELEASE:
        raise ValueError("geometry summary dataset_release mismatch")
    if manifest.get("schema") != _GEOMETRY_SCHEMA or manifest.get("dataset_release") != _RELEASE:
        raise ValueError("geometry manifest schema/release mismatch")
    if _sha256(manifest_path) != summary.get("geometry_manifest_sha256"):
        raise ValueError("geometry manifest SHA256 does not match geometry summary")
    if _sha256(patients_path) != summary.get("per_patient_sha256"):
        raise ValueError("per-patient geometry SHA256 does not match geometry summary")
    per_patient = json.loads(patients_path.read_text(encoding="utf-8"))
    index_sha = _sha256(index_path)
    if summary.get("artifact_sha256", {}).get("index") != index_sha:
        raise ValueError("geometry summary index SHA256 does not match the supplied index")
    if snapshot.get("schema") != "self_audit.fuse_mnms_open345_snapshot.v1" or snapshot.get("dataset_release") != _RELEASE:
        raise ValueError("source snapshot schema/release mismatch")
    if _sha256(snapshot_path) != summary.get("snapshot_sha256"):
        raise ValueError("source snapshot SHA256 does not match geometry summary")
    if snapshot.get("artifact_sha256", {}).get("index") != index_sha:
        raise ValueError("source snapshot index SHA256 does not match the supplied index")
    source_identities = snapshot.get("source_identities")
    if not isinstance(source_identities, list):
        raise ValueError("source snapshot source_identities must be a list")
    canonical_ids = json.dumps(source_identities, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if hashlib.sha256(canonical_ids).hexdigest() != snapshot.get("source_identities_sha256"):
        raise ValueError("source snapshot source identity SHA256 is invalid")
    source_sha = summary.get("source_identities_sha256")
    if not isinstance(source_sha, str) or not _SHA.fullmatch(source_sha):
        raise ValueError("geometry summary source_identities_sha256 is invalid")
    if snapshot.get("source_identities_sha256") != source_sha or manifest.get("source_identities_sha256") != source_sha:
        raise ValueError("geometry manifest source identity SHA mismatch")
    if summary.get("snapshot_sha256") is None or not _SHA.fullmatch(str(summary["snapshot_sha256"])):
        raise ValueError("geometry summary does not carry a valid snapshot SHA256")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("geometry manifest records must be a list")
    by_id = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("geometry manifest contains a malformed record")
        pid = record.get("patient_id")
        if not isinstance(pid, str) or not pid or pid in by_id:
            raise ValueError("geometry manifest has missing or duplicate patient IDs")
        if record.get("status") != "GEOMETRY_READY" or not isinstance(record.get("geometry"), dict):
            raise ValueError(f"geometry is not ready for indexed patient {pid}")
        by_id[pid] = record
    expected = {row["patient_id"]: row["split"] for row in rows}
    snapshot_records = snapshot.get("records")
    if not isinstance(snapshot_records, list):
        raise ValueError("source snapshot records must be a list")
    snapshot_by_id = {record.get("patient_id"): record for record in snapshot_records if isinstance(record, dict)}
    if len(snapshot_by_id) != len(snapshot_records) or len(snapshot_records) != sum(_RELEASE_COUNTS.values()):
        raise ValueError("source snapshot has duplicate identities or wrong open345 case count")
    snapshot_counts = {split: sum(record.get("split") == split for record in snapshot_records)
                       for split in _RELEASE_COUNTS}
    if snapshot_counts != _RELEASE_COUNTS:
        raise ValueError(f"source snapshot split counts must be {_RELEASE_COUNTS}")
    snapshot_splits = {pid: record.get("split") for pid, record in snapshot_by_id.items()}
    expected = snapshot_splits
    if set(by_id) != set(expected):
        raise ValueError("geometry manifest patient IDs differ from the supplied index")
    if any(by_id[pid].get("split") != split for pid, split in expected.items()):
        raise ValueError("geometry manifest split IDs differ from the supplied index")
    if not isinstance(per_patient, list):
        raise ValueError("per_patient.json must contain a list")
    patient_by_id = {record.get("patient_id"): record for record in per_patient if isinstance(record, dict)}
    if len(patient_by_id) != len(per_patient) or set(patient_by_id) != set(by_id):
        raise ValueError("per-patient report IDs differ from geometry manifest")
    for pid, record in by_id.items():
        patient_record = patient_by_id[pid]
        if patient_record.get("split") != record.get("split") or patient_record.get("status") != "GEOMETRY_READY":
            raise ValueError(f"per-patient geometry status/split mismatch for {pid}")
        if patient_record.get("geometry") != record.get("geometry"):
            raise ValueError(f"per-patient canonical geometry differs from manifest for {pid}")
    indexed = {row["patient_id"]: row for row in rows}
    for pid, row in indexed.items():
        snap = snapshot_by_id[pid]
        if snap.get("split") != row["split"] or snap.get("path") != row["image"].as_posix():
            raise ValueError(f"source snapshot identity differs from index for {pid}")
    identity_by_id = {identity.get("patient_id"): identity for identity in source_identities if isinstance(identity, dict)}
    if len(identity_by_id) != len(source_identities) or set(identity_by_id) != set(snapshot_by_id):
        raise ValueError("source snapshot identities do not bind all snapshot records")
    for pid, snap in snapshot_by_id.items():
        if snap.get("source_identity") != identity_by_id[pid]:
            raise ValueError(f"source snapshot record identity mismatch for {pid}")
        try:
            header_bytes = bytes.fromhex(snap["header_hex"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"source snapshot header encoding invalid for {pid}") from exc
        header_sha = snap.get("header_sha256")
        if len(header_bytes) != 348 or not isinstance(header_sha, str) or not _SHA.fullmatch(header_sha):
            raise ValueError(f"source snapshot header identity invalid for {pid}")
        if hashlib.sha256(header_bytes).hexdigest() != header_sha or identity_by_id[pid].get("header_sha256") != header_sha:
            raise ValueError(f"source snapshot header SHA mismatch for {pid}")
        shape = snap.get("shape")
        if not isinstance(shape, list) or len(shape) not in (3, 4):
            raise ValueError(f"source snapshot shape invalid for {pid}")
        metadata = snap.get("metadata", {})
        is_unlabeled_train = snap.get("split") == "train" and snap.get("path", "").lower().startswith("training/unlabeled/")
        if is_unlabeled_train:
            if metadata.get("ED") is not None or metadata.get("ES") is not None or metadata.get("phase_status") != "UNANNOTATED":
                raise ValueError(f"unlabeled TRAIN snapshot phases must remain unknown for {pid}")
        elif len(shape) == 4 and any(not isinstance(metadata.get(key), int) or
                                     not 0 <= metadata[key] < shape[3] for key in ("ED", "ES")):
            raise ValueError(f"source snapshot ED/ES index invalid for {pid}")
        elif metadata.get("ED") == metadata.get("ES"):
            raise ValueError(f"annotated source snapshot ED/ES indices must differ for {pid}")
    for pid, row in indexed.items():
        snap = snapshot_by_id[pid]
        if ((row["ed_frame"] is not None and snap["metadata"].get("ED") != row["ed_frame"])
                or (row["es_frame"] is not None and snap["metadata"].get("ES") != row["es_frame"])):
            raise ValueError(f"source snapshot phases differ from index for {pid}")
    return summary, manifest, by_id, index_sha, source_sha, identity_by_id


def _eligible_train(rows: list[dict[str, Any]]) -> list[str]:
    eligible = []
    for row in rows:
        if row["split"] != "train":
            continue
        is_labeled_partition = row["partition_subtype"] == "labeled" or (
            row["partition_subtype"] is None and "labeled" in row["source_partition"].lower()
        )
        if is_labeled_partition and row["masks"]["ed"] is not None and row["masks"]["es"] is not None:
            if row["ed_frame"] is None or row["es_frame"] is None:
                raise ValueError(f"eligible labeled TRAIN patient {row['patient_id']} lacks ED/ES indices")
            eligible.append(row["patient_id"])
    if len(eligible) != len(set(eligible)) or len(eligible) < 8:
        raise ValueError("eligible labeled TRAIN IDs must be unique and contain at least eight patients")
    return sorted(eligible)


def _read_nifti_header(path: Path):
    import nibabel as nib
    try:
        path.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    # nib.load reads NIfTI metadata; voxel access occurs only through an
    # explicitly selected dataobj frame in the allowlist reader below.
    return nib.load(str(path))


class _L8MaskReader:
    """Mask payload reader that rejects non-L8 access before touching disk."""

    def __init__(self, selected_ids: set[str], raw_root: Path):
        self.selected_ids = frozenset(selected_ids)
        self.raw_root = raw_root
        self.outside_reads = 0
        self.opened_patient_ids: set[str] = set()

    def read_pair(self, row: dict[str, Any], *, image_header: Any) -> list[dict[str, Any]]:
        pid = row["patient_id"]
        if pid not in self.selected_ids:
            self.outside_reads += 1
            raise PermissionError(f"mask allowlist denies patient {pid}")
        self.opened_patient_ids.add(pid)
        import nibabel as nib
        shape = tuple(int(v) for v in image_header.shape)
        spatial_shape = shape[:3]
        image_affine, image_code = image_header.header.get_sform(coded=True)
        if not int(image_code):
            image_affine, image_code = image_header.header.get_qform(coded=True)
        if not int(image_code):
            raise ValueError(f"selected image {pid} has no coded affine")
        outputs = []
        for phase in ("ed", "es"):
            relative = row["masks"][phase]
            expected = Path("Training") / "Labeled" / pid / f"{pid}_sa_gt.nii.gz"
            if relative != expected:
                raise ValueError(f"{pid} mask path must be bound to {expected.as_posix()}")
            mask_path = (self.raw_root / relative).resolve(strict=True)
            if not mask_path.is_relative_to(self.raw_root) or mask_path.relative_to(self.raw_root) != expected:
                raise ValueError(f"{pid} mask path resolves outside its patient-bound source path")
            mask_sha = _sha256(mask_path)
            mask_img = nib.load(str(mask_path))
            mask_shape = tuple(int(v) for v in mask_img.shape)
            if mask_shape not in (spatial_shape, shape):
                raise ValueError(f"{pid} {phase.upper()} mask shape {mask_shape} disagrees with image {shape}")
            mask_affine, mask_code = mask_img.header.get_sform(coded=True)
            if not int(mask_code):
                mask_affine, mask_code = mask_img.header.get_qform(coded=True)
            if not int(mask_code) or not np.allclose(mask_affine, image_affine, rtol=1e-5, atol=1e-5):
                raise ValueError(f"{pid} {phase.upper()} mask raw affine disagrees with image header")
            frame = row[f"{phase}_frame"]
            if frame is None or (len(mask_shape) == 4 and frame >= mask_shape[3]):
                raise ValueError(f"{pid} {phase.upper()} frame index is invalid for mask")
            raw = np.asanyarray(mask_img.dataobj if len(mask_shape) == 3 else mask_img.dataobj[..., frame])
            if raw.shape != spatial_shape:
                raise ValueError(f"{pid} {phase.upper()} selected mask frame shape mismatch")
            if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
                raise ValueError(f"{pid} {phase.upper()} mask values must be finite integers")
            source_counts = {str(label): int(np.count_nonzero(raw == label)) for label in range(4)}
            unique = set(np.unique(raw).tolist())
            if not unique.issubset(set(_CLASS_MAP)):
                raise ValueError(f"{pid} {phase.upper()} mask has labels outside 0,1,2,3: {sorted(unique-set(_CLASS_MAP))}")
            canonical = np.zeros(raw.shape, dtype=np.uint8)
            for source_label, target_label in _CLASS_MAP.items():
                canonical[raw == source_label] = target_label
            canonical_counts = {str(label): int(np.count_nonzero(canonical == label)) for label in range(4)}
            if canonical_counts["0"] != source_counts["0"] or canonical_counts["2"] != source_counts["2"]:
                raise AssertionError("BG/MYO counts changed during the explicit class remap")
            if canonical_counts["1"] != source_counts["3"] or canonical_counts["3"] != source_counts["1"]:
                raise AssertionError("LV/RV counts did not swap under the explicit class remap")
            outputs.append({"phase": phase.upper(), "frame": int(frame), "mask_path": str(mask_path.resolve()),
                            "mask_sha256": mask_sha, "source_class_counts": source_counts,
                            "canonical_class_counts": canonical_counts,
                            "spatial_shape_matches_image": True, "raw_affine_matches_image": True})
        return outputs


def _make_lock(index_sha: str, geometry_manifest_sha: str, source_ids_sha: str,
               selected_ids: list[str], ids_by_split: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "schema": "self_audit.fuse_mnms_target_l8_lock.v1",
        "dataset_release": _RELEASE,
        "selected_ids": selected_ids,
        "selected_ids_sha256": _compact_ids_sha(selected_ids),
        "selector_sha256_by_patient": {
            pid: hashlib.sha256(f"42:{pid}".encode()).hexdigest() for pid in selected_ids
        },
        "seed": 42,
        "k": 8,
        "index_sha256": index_sha,
        "geometry_manifest_sha256": geometry_manifest_sha,
        "source_identities_sha256": source_ids_sha,
        "patient_ids_by_split": ids_by_split,
        "mask_allowlist": selected_ids,
        "mask_reads_outside_l8": 0,
        "prepared_manifests": "NOT_RUN",
        "training_data_arrays": "NOT_RUN",
        "B_engine": "NOT_RUN",
    }


def lock_l8(index_path: str | Path, raw_root: str | Path, geometry_report_dir: str | Path,
            out_dir: str | Path) -> dict[str, Any]:
    index_path, raw_root = Path(index_path).resolve(), Path(raw_root).resolve()
    report_dir, out_dir = Path(geometry_report_dir).resolve(), Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"L8 output directory already exists: {out_dir}")
    if out_dir == raw_root or out_dir.is_relative_to(raw_root):
        raise ValueError("L8 receipt output must be outside the immutable raw-data root")
    if out_dir == report_dir or out_dir.is_relative_to(report_dir):
        raise ValueError("L8 receipt output must be outside the geometry evidence directory")
    if out_dir == index_path or index_path.is_relative_to(out_dir):
        raise ValueError("L8 receipt output must not contain or replace the source index")
    rows = _read_index(index_path)
    _summary, _manifest, geometry_by_id, index_sha, source_ids_sha, source_by_id = _load_geometry_evidence(
        index_path, report_dir, rows
    )
    # Require geometry for every indexed patient, including VAL and TEST, before
    # any mask path is opened.
    for pid, record in geometry_by_id.items():
        try:
            canonical_geometry(record["geometry"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"canonical geometry contract invalid for {pid}: {exc}") from exc
    eligible = _eligible_train(rows)
    selected_ids = sorted(label_subset(eligible, 8, 42))
    selected_set = set(selected_ids)
    by_id = {row["patient_id"]: row for row in rows}
    if any(by_id[pid]["split"] != "train" for pid in selected_ids):
        raise AssertionError("selector returned an ID outside TRAIN")

    # Persist the immutable cohort lock before opening any mask headers or
    # voxel frames. The directory is exclusive and never overwritten.
    out_dir.mkdir(parents=True, exist_ok=False)
    geometry_manifest_sha = _sha256(report_dir / "geometry_manifest.json")
    ids_by_split = {split: sorted(pid for pid, rec in geometry_by_id.items() if rec["split"] == split)
                    for split in ("train", "val", "test")}
    lock = _make_lock(index_sha, geometry_manifest_sha, source_ids_sha, selected_ids, ids_by_split)
    lock_path = out_dir / "L8_LOCK.json"
    with lock_path.open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    reader = _L8MaskReader(selected_set, raw_root)
    receipts = []
    for pid in selected_ids:
        row = by_id[pid]
        image_path = (raw_root / row["image"]).resolve(strict=True)
        if not image_path.is_relative_to(raw_root):
            raise ValueError(f"selected image path for {pid} resolves outside raw root")
        image_sha = _sha256(image_path)
        if image_sha != source_by_id[pid].get("source_sha256"):
            raise ValueError(f"selected image SHA256 does not match source snapshot for {pid}")
        image_header = _read_nifti_header(image_path)
        image_shape = tuple(int(v) for v in image_header.shape)
        if len(image_shape) not in (3, 4) or any(size < 2 for size in image_shape[:3]):
            raise ValueError(f"selected image {pid} has invalid 3D/4D shape")
        expected_zyx = tuple(int(v) for v in geometry_by_id[pid]["geometry"]["shape"])
        if expected_zyx != tuple(reversed(image_shape[:3])):
            raise ValueError(f"selected image {pid} shape differs from adjudicated native geometry")
        for phase in ("ed", "es"):
            frame = row[f"{phase}_frame"]
            if frame is None or frame >= (image_shape[3] if len(image_shape) == 4 else 1):
                raise ValueError(f"selected image {pid} {phase.upper()} frame index is invalid")
        mask_rows = reader.read_pair(row, image_header=image_header)
        receipts.append({"patient_id": pid, "image_path": str(image_path),
                         "image_sha256": image_sha, "image_shape": image_shape,
                         "masks": mask_rows})
    if reader.outside_reads != 0:
        raise AssertionError("non-L8 mask read attempt was observed")
    receipt = {
        "schema": "self_audit.fuse_mnms_l8_mask_mapping_receipt.v1",
        "dataset_release": _RELEASE,
        "l8_lock_sha256": _sha256(lock_path),
        "selected_ids": selected_ids,
        "mask_reads_outside_l8": reader.outside_reads,
        "mask_patient_ids_read": sorted(reader.opened_patient_ids),
        "source_to_canonical_class_map": {str(k): v for k, v in _CLASS_MAP.items()},
        "records": receipts,
        "prepared_manifests": "NOT_RUN",
        "training_data_arrays": "NOT_RUN",
        "B_engine": "NOT_RUN",
    }
    receipt_path = out_dir / "MASK_MAPPING_RECEIPT.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return {"lock": lock, "receipt": receipt, "out_dir": str(out_dir)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--geometry-report-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = lock_l8(args.index, args.raw_root, args.geometry_report_dir, args.out)
    print(json.dumps({"out_dir": result["out_dir"], "selected_ids": result["lock"]["selected_ids"],
                      "mask_reads_outside_l8": result["receipt"]["mask_reads_outside_l8"],
                      "prepared_manifests": "NOT_RUN", "B_engine": "NOT_RUN"}, sort_keys=True))


if __name__ == "__main__":
    main()
