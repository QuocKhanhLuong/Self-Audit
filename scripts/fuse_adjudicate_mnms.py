#!/usr/bin/env python3
"""Snapshot and adjudicate original M&Ms open345 image headers only.

The snapshot operation never materializes voxel arrays or opens segmentation
members; it hashes raw compressed source bytes. Adjudication writes a separate
metadata receipt and never edits source images or headers.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "self_audit.fuse_mnms_open345_snapshot.v1"
RELEASE = "original_2020_open345"
SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 175, "val": 34, "test": 136}
from self_audit.candidate_fuse.geometry import UNITS_EVIDENCE_URL, UNITS_EVIDENCE_SHA256
_SHA = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    p = PurePosixPath(value.replace("\\", "/"))
    if not value or p.is_absolute() or ".." in p.parts or "." in p.parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return p


def _read_header(raw: bytes):
    """Parse exactly a single-file NIfTI header, without data proxy access."""
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise ValueError("nibabel is required for M&Ms header inspection") from exc
    if len(raw) < 352:
        raise ValueError("NIfTI file is shorter than a single-file header")
    # Header bytes are decompressed only through byte 348. This never asks
    # nibabel for image data or consumes the compressed voxel payload.
    header = nib.Nifti1Header.from_fileobj(io.BytesIO(raw[:348]))
    if int(header["sizeof_hdr"]) != 348:
        raise ValueError("unsupported NIfTI header size")
    return header, raw[:348]


def _header_from_gzip(stream):
    """Read a gzip NIfTI header and stop before voxel payload decoding."""
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise ValueError("nibabel is required for M&Ms header inspection") from exc
    with gzip.GzipFile(fileobj=stream, mode="rb") as gz:
        # Gzip's buffered reader may request more compressed bytes than the
        # caller asked to decompress. The NIfTI reader itself is capped here
        # at 348 output bytes, the header only.
        raw_header = gz.read(348)
    if len(raw_header) != 348:
        raise ValueError("compressed NIfTI member has a truncated header")
    header = nib.Nifti1Header.from_fileobj(io.BytesIO(raw_header))
    if int(header["sizeof_hdr"]) != 348:
        raise ValueError("unsupported NIfTI header size")
    return header, raw_header


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _metadata(path: Path) -> dict[str, dict[str, str]]:
    fields, rows = _csv_rows(path)
    required = {"External code", "ED", "ES", "VendorName", "Vendor", "Centre"}
    if not required.issubset(fields):
        raise ValueError(f"metadata CSV must include {sorted(required)}")
    result = {}
    for row in rows:
        pid = (row.get("External code") or "").strip()
        if not pid or pid in result:
            raise ValueError(f"missing or duplicate metadata patient ID: {pid!r}")
        result[pid] = {key: (row.get(key) or "").strip() for key in required - {"External code"}}
    if len(result) != 345:
        raise ValueError(f"metadata must contain exactly 345 unique IDs, got {len(result)}")
    return result


def _frame(value: str, *, patient_id: str, name: str) -> int | None:
    value = (value or "").strip()
    if not value or value.lower() in {"na", "nan", "null", "none", "-1"}:
        return None
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"invalid {name} frame for {patient_id}: {value!r}") from exc
    if number < 0:
        raise ValueError(f"invalid {name} frame for {patient_id}: {number}")
    return number


def _partition(value: str) -> str:
    key = (value or "").strip().lower()
    mapping = {"train": "train", "training": "train", "training/labeled": "train", "training/unlabeled": "train",
               "val": "val", "valid": "val", "validation": "val", "test": "test", "testing": "test"}
    if key not in mapping:
        raise ValueError(f"unknown source_partition: {value!r}")
    return mapping[key]


def _validate_layout(split: str, rel: str) -> None:
    p = _safe_relative(rel)
    parts = tuple(part.lower() for part in p.parts)
    if not parts or p.suffix.lower() not in {".nii", ".gz"}:
        raise ValueError(f"index image must name a NIfTI file: {rel}")
    if split == "train" and not (len(parts) >= 3 and parts[0] == "training" and parts[1] in {"labeled", "unlabeled"}):
        raise ValueError(f"TRAIN path must be under Training/Labeled or Training/Unlabeled: {rel}")
    if split == "val" and not (len(parts) >= 2 and parts[0] == "validation"):
        raise ValueError(f"VAL path must be under Validation: {rel}")
    if split == "test" and not (len(parts) >= 3 and parts[0] == "testing"):
        raise ValueError(f"TEST paths come only from the Testing ZIP: {rel}")


def _read_index(path: Path, raw_root: Path) -> list[dict[str, Any]]:
    fields, rows = _csv_rows(path)
    required = {"patient_id", "split", "image", "ed_frame", "es_frame", "source_partition"}
    if not required.issubset(fields):
        raise ValueError(f"index must include {sorted(required)}")
    result, ids, paths = [], set(), set()
    for row in rows:
        pid = (row.get("patient_id") or "").strip()
        split = (row.get("split") or "").strip().lower()
        rel = (row.get("image") or "").strip()
        if not pid or split not in SPLITS or not rel:
            raise ValueError(f"incomplete index row: {row}")
        if pid in ids:
            raise ValueError(f"duplicate index patient ID: {pid}")
        safe = _safe_relative(rel)
        canonical_rel = safe.as_posix()
        if (split, canonical_rel) in paths:
            raise ValueError(f"duplicate index image path: {canonical_rel}")
        ids.add(pid); paths.add((split, canonical_rel))
        partition = _partition(row.get("source_partition", ""))
        if partition != split:
            raise ValueError(f"source_partition disagrees with split for {pid}: {partition} != {split}")
        _validate_layout(split, canonical_rel)
        partition_text = (row.get("source_partition") or "").strip().lower()
        if split == "train" and partition_text in {"training/labeled", "training/unlabeled"}:
            if PurePosixPath(canonical_rel).parts[1].lower() != partition_text.split("/")[1]:
                raise ValueError(f"source_partition path disagrees with image path for {pid}")
        if split in {"train", "val"}:
            if PurePosixPath(canonical_rel).parent.name != pid or PurePosixPath(canonical_rel).name != f"{pid}_sa.nii.gz":
                raise ValueError(f"image path does not bind to patient ID {pid}: {canonical_rel}")
        result.append({"patient_id": pid, "split": split, "path": canonical_rel,
                       "ed_frame": _frame(row.get("ed_frame", ""), patient_id=pid, name="ED"),
                       "es_frame": _frame(row.get("es_frame", ""), patient_id=pid, name="ES"),
                       "source_partition": (row.get("source_partition") or "").strip()})
    return result


def snapshot(index: Path, raw_root: Path, test_archive: Path, metadata_path: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"snapshot output already exists: {out}")
    metadata = _metadata(metadata_path)
    records = _read_index(index, raw_root)
    index_ids = {row["patient_id"] for row in records}
    if not index_ids.issubset(set(metadata)):
        raise ValueError(f"index contains IDs absent from metadata: {sorted(index_ids-set(metadata))[:8]}")
    counts = {split: sum(row["split"] == split for row in records) for split in SPLITS}
    if counts["test"] != 0:
        raise ValueError("TEST rows must be inventoried from the official Testing ZIP, not the local index")
    if counts["train"] != EXPECTED_COUNTS["train"] or counts["val"] != EXPECTED_COUNTS["val"]:
        raise ValueError("index must contain exactly the 175 TRAIN and 34 VAL rows")
    archive_sha = sha256_file(test_archive)
    index_sha = sha256_file(index)
    metadata_sha = sha256_file(metadata_path)
    zip_records: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(test_archive) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("TEST ZIP contains duplicate member names")
        image_members = []
        all_names = set()
        for name in names:
            p = _safe_relative(name)
            if p.parts[0].lower() != "testing" or len(p.parts) < 3:
                raise ValueError(f"unexpected TEST ZIP member path: {name}")
            zip_records[p.as_posix()] = archive.getinfo(name)
            all_names.add(p.as_posix())
            if p.name.lower().endswith("_sa.nii.gz"):
                image_members.append(p.as_posix())
        if len(names) != 272:
            raise ValueError(f"TEST ZIP must contain exactly 272 image/GT members, got {len(names)}")
        if len(image_members) != 136 or len(set(image_members)) != 136:
            raise ValueError(f"TEST ZIP must contain 136 unique _sa.nii.gz images, got {len(image_members)}")
        gt_members = {name for name in all_names if PurePosixPath(name).name.lower().endswith("_gt.nii.gz")}
        if len(gt_members) != 136:
            raise ValueError(f"TEST ZIP must contain exactly 136 companion _gt.nii.gz members, got {len(gt_members)}")
        test_ids = {PurePosixPath(name).parent.name for name in image_members}
        if len(test_ids) != 136 or not test_ids.issubset(metadata):
            raise ValueError("TEST image member IDs are duplicated or absent from metadata")
        for image_member in image_members:
            pid = PurePosixPath(image_member).parent.name
            if PurePosixPath(image_member).name != f"{pid}_sa.nii.gz":
                raise ValueError(f"TEST image path does not bind to patient ID {pid}: {image_member}")
        if index_ids | test_ids != set(metadata) or index_ids & test_ids:
            raise ValueError("index and TEST archive IDs must be disjoint and exactly cover metadata IDs")
        for image_member in image_members:
            gt_member = image_member.replace("_sa.nii.gz", "_sa_gt.nii.gz")
            if gt_member not in all_names:
                raise ValueError(f"TEST companion GT member is absent: {gt_member}")
            pid = PurePosixPath(image_member).parent.name
            records.append({"patient_id": pid, "split": "test", "path": image_member,
                            "ed_frame": None, "es_frame": None})
        counts = {split: sum(row["split"] == split for row in records) for split in SPLITS}
        test_rows = [r for r in records if r["split"] == "test"]
        expected_members = {r["path"] for r in test_rows}
        if expected_members != set(image_members):
            raise ValueError("TEST image inventory does not match ZIP image members")
        if counts != EXPECTED_COUNTS:
            raise ValueError(f"combined archive partition counts must be {EXPECTED_COUNTS}, got {counts}")

        identities = []
        output_records = []
        for row in sorted(records, key=lambda r: (r["split"], r["patient_id"])):
            md = metadata[row["patient_id"]]
            is_unlabeled_train = (row["split"] == "train"
                                  and row.get("source_partition", "").strip().lower() == "training/unlabeled")
            source_ed, source_es = md["ED"], md["ES"]
            if is_unlabeled_train:
                if row["ed_frame"] is not None or row["es_frame"] is not None:
                    raise ValueError(f"unlabeled TRAIN index phases must remain blank for {row['patient_id']}")
                # These source CSV zeros are an unannotated sentinel, not phase indices.
                md_ed = md_es = None
                phase_status = "UNANNOTATED"
            else:
                md_ed = _frame(source_ed, patient_id=row["patient_id"], name="metadata ED")
                md_es = _frame(source_es, patient_id=row["patient_id"], name="metadata ES")
                if md_ed is None or md_es is None:
                    raise ValueError(f"known ED and ES metadata frames are required for {row['patient_id']}")
                phase_status = "KNOWN"
            for name, indexed, official in (("ED", row["ed_frame"], md_ed), ("ES", row["es_frame"], md_es)):
                if indexed is not None and official is not None and indexed != official:
                    raise ValueError(f"index {name} frame disagrees with metadata for {row['patient_id']}")
            if not is_unlabeled_train and md_ed == md_es:
                raise ValueError(f"ED and ES metadata frames are equal for {row['patient_id']}")

            if row["split"] == "test":
                info = zip_records[row["path"]]
                with archive.open(info, "r") as member:
                    header, raw_header = _header_from_gzip(member)
                source_record = {"patient_id": row["patient_id"], "split": row["split"], "path": row["path"],
                                 "source_sha256": archive_sha, "member": row["path"], "crc32": f"{info.CRC:08x}",
                                 "header_sha256": hashlib.sha256(raw_header).hexdigest()}
            else:
                image_path = raw_root / row["path"]
                if not image_path.is_file():
                    raise ValueError(f"missing source image: {image_path}")
                source_sha = sha256_file(image_path)
                # gzip streams are consumed only for 348 header bytes; raw NIfTI is sliced.
                if image_path.name.lower().endswith(".nii.gz"):
                    with image_path.open("rb") as stream:
                        header, raw_header = _header_from_gzip(stream)
                else:
                    with image_path.open("rb") as stream:
                        raw = stream.read(352)
                    header, raw_header = _read_header(raw)
                source_record = {"patient_id": row["patient_id"], "split": row["split"], "path": row["path"],
                                 "source_sha256": source_sha, "header_sha256": hashlib.sha256(raw_header).hexdigest()}
            shape = tuple(int(v) for v in header.get_data_shape())
            if len(shape) not in (3, 4):
                raise ValueError(f"unsupported NIfTI shape for {row['patient_id']}: {shape}")
            for name, frame in (("ED", md_ed), ("ES", md_es)):
                if frame is not None and frame >= (shape[3] if len(shape) == 4 else 1):
                    raise ValueError(f"metadata {name} frame outside image shape for {row['patient_id']}")
            identity = source_record
            identities.append(identity)
            output_records.append({**row, "shape": list(shape), "header_hex": raw_header.hex(),
                                   "header_sha256": source_record["header_sha256"],
                                   "metadata": {"source_ED": source_ed, "source_ES": source_es,
                                                "ED": md_ed, "ES": md_es, "phase_status": phase_status,
                                                "VendorName": md["VendorName"],
                                                "Vendor": md["Vendor"], "Centre": md["Centre"]},
                                   "source_identity": identity})

    identities.sort(key=lambda r: (r["split"], r["patient_id"]))
    result = {"schema": SCHEMA, "dataset_release": RELEASE,
              "artifact_sha256": {"index": index_sha, "metadata": metadata_sha, "test_archive": archive_sha},
              "split_counts": counts, "source_identities": identities,
              "source_identities_sha256": _json_hash(identities), "records": output_records}
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents overwriting a concurrent or existing snapshot.
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def verify_snapshot(snapshot_data: dict[str, Any]) -> list[dict[str, Any]]:
    if snapshot_data.get("schema") != SCHEMA or snapshot_data.get("dataset_release") != RELEASE:
        raise ValueError("unsupported M&Ms snapshot schema or release")
    records = snapshot_data.get("records")
    identities = snapshot_data.get("source_identities")
    if not isinstance(records, list) or not isinstance(identities, list):
        raise ValueError("snapshot records/source_identities must be lists")
    identity_by_id = {r.get("patient_id"): r for r in identities if isinstance(r, dict)}
    if len(identity_by_id) != len(identities) or len(records) != 345:
        raise ValueError("snapshot has duplicate identities or is not the complete 345-case release")
    split_counts = {split: sum(r.get("split") == split for r in records) for split in SPLITS}
    if split_counts != EXPECTED_COUNTS or snapshot_data.get("split_counts") != EXPECTED_COUNTS:
        raise ValueError(f"snapshot partition counts must be {EXPECTED_COUNTS}")
    if _json_hash(identities) != snapshot_data.get("source_identities_sha256"):
        raise ValueError("source_identities_sha256 does not match snapshot identities")
    seen = set()
    for row in records:
        pid = row.get("patient_id")
        if not pid or pid in seen or pid not in identity_by_id:
            raise ValueError(f"snapshot has missing/duplicate/unbound record {pid!r}")
        seen.add(pid)
        identity = row.get("source_identity")
        if identity != identity_by_id[pid]:
            raise ValueError(f"source identity record tampered for {pid}")
        if row.get("split") != identity.get("split") or row.get("path") != identity.get("path"):
            raise ValueError(f"snapshot split/path mismatch for {pid}")
        try:
            header_bytes = bytes.fromhex(row["header_hex"])
        except Exception as exc:
            raise ValueError(f"invalid header_hex for {pid}") from exc
        if len(header_bytes) != 348 or hashlib.sha256(header_bytes).hexdigest() != row.get("header_sha256") or row.get("header_sha256") != identity.get("header_sha256"):
            raise ValueError(f"header integrity failure for {pid}")
    if seen != set(identity_by_id):
        raise ValueError("snapshot records do not cover all source identities")
    return records


def _raw_findings(header) -> dict[str, Any]:
    # Preserve historical strict audit findings from the original header; this
    # operates in-memory and cannot access a voxel array.
    raw_unit = header.get_xyzt_units()[0]
    pixdim = np.asarray(header["pixdim"], dtype=float)[1:4]
    affine, code = header.get_sform(coded=True)
    if not int(code):
        affine, code = header.get_qform(coded=True)
    findings = []
    norms = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0) if int(code) else np.zeros(3)
    dirs = np.asarray(affine)[:3, :3] / np.where(norms > 0, norms, np.nan) if int(code) else np.full((3, 3), np.nan)
    gram = dirs.T @ dirs
    if not int(code): findings.append("missing_coded_affine")
    if not np.isfinite(pixdim).all() or np.any(pixdim <= 0): findings.append("invalid_pixdim")
    if not np.isfinite(gram).all() or not np.allclose(gram, np.eye(3), atol=1e-4, rtol=0): findings.append("nonorthogonal_affine")
    if not np.isfinite(norms).all() or np.any(norms <= 0): findings.append("invalid_affine_spacing")
    if not np.allclose(norms, pixdim, rtol=1e-5, atol=1e-5): findings.append("affine_pixdim_disagreement")
    if raw_unit not in {"mm", "meter", "micron"}: findings.append("unknown_spatial_units")
    qform, qcode = header.get_qform(coded=True)
    sform, scode = header.get_sform(coded=True)
    def finite(value):
        if isinstance(value, list): return [finite(v) for v in value]
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        return value
    return {"status": "INVALID" if findings else "VALID", "findings": findings,
            "raw_unit": raw_unit, "pixdim_xyz": finite(pixdim.tolist()), "raw_affine_xyz": finite(np.asarray(affine).tolist()) if int(code) else None,
            "qform_code": int(qcode), "qform_xyz": finite(np.asarray(qform).tolist()) if int(qcode) else None,
            "sform_code": int(scode), "sform_xyz": finite(np.asarray(sform).tolist()) if int(scode) else None}


def _policy_check(policy: dict[str, Any], snapshot_data: dict[str, Any], units_evidence: Path) -> None:
    required = {"schema", "name", "dataset_release", "units_evidence_url", "units_evidence_sha256", "source_identities_sha256", "reason"}
    if set(policy) != required:
        raise ValueError("policy must contain exactly the locked M&Ms geometry policy fields")
    if policy["schema"] != "self_audit.mnms_geometry_policy.v1" or policy["name"] != "mnms_open345_mm_polar_v1" or policy["dataset_release"] != RELEASE:
        raise ValueError("unsupported M&Ms open345 policy identity")
    for key in ("units_evidence_sha256", "source_identities_sha256"):
        if not isinstance(policy[key], str) or not _SHA.fullmatch(policy[key]):
            raise ValueError(f"invalid {key}")
    if policy["source_identities_sha256"] != snapshot_data["source_identities_sha256"]:
        raise ValueError("policy source_identities_sha256 does not bind this snapshot")
    if not isinstance(policy["units_evidence_url"], str) or not policy["units_evidence_url"].startswith("https://"):
        raise ValueError("units_evidence_url must be https")
    if policy["units_evidence_url"] != UNITS_EVIDENCE_URL:
        raise ValueError("units evidence URL is not the locked acquisition source")
    if policy["units_evidence_sha256"] != UNITS_EVIDENCE_SHA256:
        raise ValueError("units evidence SHA256 is not the locked acquisition source")
    if not isinstance(policy["reason"], str) or not policy["reason"].strip():
        raise ValueError("policy reason must be nonempty")
    if sha256_file(units_evidence) != policy["units_evidence_sha256"]:
        raise ValueError("units evidence SHA256 does not match policy")


def adjudicate(snapshot_path: Path, policy_path: Path, units_evidence: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"adjudication output directory already exists: {out_dir}")
    snapshot_data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    records = verify_snapshot(snapshot_data)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    _policy_check(policy, snapshot_data, units_evidence)
    from self_audit.candidate_fuse.geometry import adjudicate_geometry

    results = []
    for row in records:
        result = {"patient_id": row["patient_id"], "split": row["split"], "path": row["path"],
                  "shape": row["shape"], "metadata": row["metadata"], "source_identity": row["source_identity"]}
        try:
            raw_header = bytes.fromhex(row["header_hex"])
            import nibabel as nib
            header = nib.Nifti1Header.from_fileobj(io.BytesIO(raw_header))
            raw = _raw_findings(header)
            result["raw_findings"] = raw
            geometry = adjudicate_geometry(header, policy=policy)
            result.update(status="GEOMETRY_READY", geometry=geometry["geometry"],
                          policy_name=geometry["policy_name"], raw_unit=geometry["raw_unit"],
                          effective_unit=geometry["effective_unit"],
                          unit_interpretation=geometry["unit_interpretation"],
                          affine_source=geometry["affine_source"],
                          original_affine_xyz=geometry["original_affine_xyz"],
                          effective_affine_xyz=geometry["effective_affine_xyz"],
                          gram_error=geometry["gram_error"],
                          max_corner_displacement_mm=geometry["max_corner_displacement_mm"])
        except Exception as exc:
            result.update(status="BLOCKED", blocker=f"{type(exc).__name__}: {exc}")
            result.setdefault("raw_findings", {"status": "UNAVAILABLE", "findings": ["header_parse_or_raw_audit_failed"]})
        results.append(result)

    counts = {split: sum(row["split"] == split for row in results) for split in SPLITS}
    blocked_val_test = [r["patient_id"] for r in results if r["split"] in {"val", "test"} and r["status"] != "GEOMETRY_READY"]
    blocked_train = [r["patient_id"] for r in results if r["split"] == "train" and r["status"] != "GEOMETRY_READY"]
    raw_unknown = sum("unknown_spatial_units" in r["raw_findings"]["findings"] for r in results)
    corrected = sum(r["status"] == "GEOMETRY_READY" and r["raw_findings"]["status"] != "VALID" for r in results)
    max_delta = max((r.get("max_corner_displacement_mm", 0.0) for r in results if r["status"] == "GEOMETRY_READY"), default=0.0)
    sha_map = snapshot_data["artifact_sha256"]
    all_canonical_pass = len(results) == 345 and all(r["status"] == "GEOMETRY_READY" for r in results)
    status = "GEOMETRY_READY" if all_canonical_pass else "BLOCKED"
    summary = {"schema": "self_audit.fuse_mnms_open345_adjudication.v1", "status": status,
               "dataset_release": RELEASE, "policy": policy, "policy_sha256": sha256_file(policy_path),
               "snapshot_sha256": sha256_file(snapshot_path), "source_identities_sha256": snapshot_data["source_identities_sha256"],
               "artifact_sha256": sha_map, "split_counts": counts, "geometry_ready_counts": {s: sum(r["split"] == s and r["status"] == "GEOMETRY_READY" for r in results) for s in SPLITS},
               "raw_unknown_unit_count": raw_unknown, "corrected_count": corrected,
               "max_corner_displacement_mm": max_delta,
               "blocked_patient_ids_by_split": {s: [r["patient_id"] for r in results if r["split"] == s and r["status"] != "GEOMETRY_READY"] for s in SPLITS},
               "blockers": (["invalid_VAL_TEST_geometry_stops_Track_B_and_B0"] if blocked_val_test else []) + (["TRAIN_geometry_failures=" + ",".join(blocked_train)] if blocked_train else []),
               "target_L8_lock": "NOT_RUN", "prepared_manifests": "NOT_RUN", "B_training": "NOT_RUN", "mask_sample_checks": "NOT_RUN",
               "scope": "header_only_no_voxel_or_mask_payloads; metadata geometry receipt, not a training manifest"}
    out_dir.mkdir(parents=True)
    (out_dir / "source_snapshot.json").write_bytes(snapshot_path.read_bytes())
    (out_dir / "per_patient.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    fields = sorted({key for row in results for key in row})
    with (out_dir / "per_patient.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in results)
    (out_dir / "geometry_manifest.json").write_text(json.dumps({"schema": "self_audit.fuse_mnms_geometry_receipt.v1", "dataset_release": RELEASE, "policy_sha256": summary["policy_sha256"], "source_identities_sha256": summary["source_identities_sha256"], "records": [{"patient_id": r["patient_id"], "split": r["split"], "geometry": r.get("geometry"), "status": r["status"]} for r in results]}, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    summary["per_patient_sha256"] = sha256_file(out_dir / "per_patient.json")
    summary["geometry_manifest_sha256"] = sha256_file(out_dir / "geometry_manifest.json")
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--index", type=Path, required=True); snap.add_argument("--raw-root", type=Path, required=True)
    snap.add_argument("--test-archive", type=Path, required=True); snap.add_argument("--metadata", type=Path, required=True)
    snap.add_argument("--out", type=Path, required=True)
    adj = sub.add_parser("adjudicate")
    adj.add_argument("--snapshot", type=Path, required=True); adj.add_argument("--policy", type=Path, required=True)
    adj.add_argument("--units-evidence", type=Path, required=True); adj.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        result = snapshot(args.index, args.raw_root, args.test_archive, args.metadata, args.out)
    else:
        result = adjudicate(args.snapshot, args.policy, args.units_evidence, args.out)
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "source_identities"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
