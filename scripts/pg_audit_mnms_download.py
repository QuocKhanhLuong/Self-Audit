#!/usr/bin/env python3
"""Read-only integrity and NIfTI header audit for the three M&Ms archives.

ZIP members are streamed for CRC integrity; image members are decompressed only
through the NIfTI header. No archive member is extracted and no voxel or mask
array is read. Geometry is applied only from a matching committed FUSE receipt
and the SHA-pinned acquisition publication.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import zipfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PUBLISHED_SOURCE = ROOT / "reports/fuse-mnms-geometry-20261113"
EXPECTED_ARCHIVES = {
    "Testing-003.zip": "test",
    "Training-001.zip": "train",
    "M&M-20251118T142205Z-1-002.zip": "val",
}
EXPECTED_COUNTS = {"train": 175, "val": 34, "test": 136}
CLASS_MAPPING = {"0": 0, "1": 3, "2": 2, "3": 1}
GEOMETRY_AFFINE_ATOL = 1e-10
PAPER_SHA256 = "c3584856b3933cdeb440d3d2215f1bcdc6d90a7bbe5803bee615e7a823b4c7fd"
PAPER_URL = "https://diposit.ub.edu/server/api/core/bitstreams/f2f7dc43-becc-4554-910d-360ed7fd339c/content"
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if not name or path.is_absolute() or ".." in path.parts or "." in path.parts or not path.parts:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    if any(not SAFE.fullmatch(part) for part in path.parts):
        raise ValueError(f"unexpected ZIP member path characters: {name!r}")
    return path


def _header(stream) -> tuple[Any, bytes]:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("nibabel is required for M&Ms NIfTI header audit") from exc
    raw = stream.read(348)
    if len(raw) != 348:
        raise ValueError("NIfTI header shorter than 348 bytes")
    header = nib.Nifti1Header.from_fileobj(io.BytesIO(raw))
    if int(header["sizeof_hdr"]) != 348:
        raise ValueError("unsupported NIfTI header size")
    return header, raw


def _archive_inventory(path: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    digest = sha256_file(path)
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        files = [i for i in infos if not i.is_dir()]
        # The user's Drive export wraps the official Validation.zip and its
        # metadata CSV. Unwrap only these two observed, exact member names;
        # preserve the original container and bind both levels by SHA256.
        nested = 'M&M/Validation.zip'
        metadata = 'M&M/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv'
        if split == 'val' and {i.filename for i in files} == {nested, metadata}:
            with tempfile.TemporaryDirectory(prefix='pg-mnms-header-audit-') as temporary:
                inner = Path(temporary) / 'Validation.zip'
                with archive.open(nested) as src, inner.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
                metadata_sha = hashlib.sha256(archive.read(metadata)).hexdigest()
                result, rows = _archive_inventory(inner, split)
                inner_sha = result['sha256']
                for row in rows:
                    row.update(archive=path.name, archive_sha256=digest,
                               nested_member=nested, nested_archive_sha256=inner_sha)
                result.update(path=str(path), sha256=digest, nested_member=nested,
                              nested_archive_sha256=inner_sha, metadata_sha256=metadata_sha)
                return result, rows
        names = [_safe_member(i.filename).as_posix() for i in files]
        if len(names) != len(set(names)):
            raise ValueError(f"{path.name}: duplicate normalized member paths")
        images = []
        for info, name in zip(files, names):
            member = PurePosixPath(name)
            low = tuple(x.lower() for x in member.parts)
            expected_root = {"train": "training", "val": "validation", "test": "testing"}[split]
            if not low or low[0] != expected_root or len(member.parts) < 3:
                raise ValueError(f"{path.name}: unexpected {split} member: {name}")
            if member.name.lower().endswith("_sa.nii.gz") and not member.name.lower().endswith("_sa_gt.nii.gz"):
                images.append((info, name))
        # Reading every member forces zlib/CRC verification without writing files.
        for info, name in zip(files, names):
            with archive.open(info) as member_stream:
                while member_stream.read(4 * 1024 * 1024):
                    pass
        seen_ids: set[str] = set()
        for info, name in images:
            rel = PurePosixPath(name)
            pid = rel.parent.name
            if not re.fullmatch(r"[A-Z0-9]{6}", pid) or pid in seen_ids or rel.name != f"{pid}_sa.nii.gz":
                raise ValueError(f"{path.name}: invalid or duplicate patient image path: {name}")
            seen_ids.add(pid)
            with archive.open(info) as compressed:
                with gzip.GzipFile(fileobj=compressed, mode="rb") as decoded:
                    header, raw_header = _header(decoded)
            shape = [int(v) for v in header.get_data_shape()]
            if len(shape) not in (3, 4):
                raise ValueError(f"{name}: unsupported image dimensionality {shape}")
            rows.append({
                "patient_id": pid, "split": split, "archive": path.name, "member": name,
                "archive_sha256": digest, "crc32": f"{info.CRC:08x}",
                "file_size": info.file_size, "compressed_size": info.compress_size,
                "header_sha256": hashlib.sha256(raw_header).hexdigest(),
                "shape": shape, "raw_header": _raw_findings(header),
            })
        expected = EXPECTED_COUNTS[split]
        if len(rows) != expected:
            raise ValueError(f"{path.name}: expected {expected} image members, found {len(rows)}")
        if split != "train" and len(files) != 2 * expected:
            raise ValueError(f"{path.name}: expected image plus GT member pairs ({2 * expected}), found {len(files)}")
        if split == "train" and len(files) != 2 * expected:
            raise ValueError(f"{path.name}: expected 175 images plus 175 masks, found {len(files)} members")
    return {"path": str(path), "sha256": digest, "member_count": len(files),
            "image_count": len(rows), "integrity": "CRC_VERIFIED"}, rows


def _raw_findings(header) -> dict[str, Any]:
    raw_unit = header.get_xyzt_units()[0]
    pixdim = np.asarray(header["pixdim"], dtype=float)[1:4]
    sform, scode = header.get_sform(coded=True)
    qform, qcode = header.get_qform(coded=True)
    affine, code = (sform, scode) if int(scode) else (qform, qcode)
    findings = []
    if not int(code):
        findings.append("missing_coded_affine")
        norms = np.zeros(3)
        gram = np.full((3, 3), np.nan)
    else:
        affine = np.asarray(affine, dtype=float)
        norms = np.linalg.norm(affine[:3, :3], axis=0)
        dirs = affine[:3, :3] / np.where(norms > 0, norms, np.nan)
        gram = dirs.T @ dirs
        if not np.isfinite(affine).all() or abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-12:
            findings.append("invalid_or_singular_affine")
        if not np.isfinite(gram).all() or not np.allclose(gram, np.eye(3), atol=1e-4, rtol=0):
            findings.append("nonorthogonal_affine")
    if not np.isfinite(pixdim).all() or np.any(pixdim <= 0):
        findings.append("invalid_pixdim")
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        findings.append("invalid_affine_spacing")
    if not np.allclose(norms, pixdim, rtol=1e-5, atol=1e-5):
        findings.append("affine_pixdim_disagreement")
    if raw_unit not in {"mm", "meter", "micron"}:
        findings.append("unknown_spatial_units")
    return {"status": "VALID" if not findings else "INVALID", "findings": findings,
            "raw_unit": raw_unit, "pixdim_xyz": _finite(pixdim.tolist()),
            "selected_affine_xyz": _finite(np.asarray(affine).tolist()) if int(code) else None,
            "sform_code": int(scode), "qform_code": int(qcode)}


def _finite(value):
    if isinstance(value, list):
        return [_finite(x) for x in value]
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _compare_geometry(receipt_geometry: dict[str, Any], recomputed_geometry: dict[str, Any]) -> dict[str, Any]:
    """Compare stable geometry identity strictly and polar affine numerically.

    NumPy/LAPACK SVD implementations can differ in the last few bits; the
    committed 345-patient receipt was observed to differ by at most 1.1e-14.
    The 1e-10 affine tolerance is therefore a generous numerical-only bound,
    while shape and pixdim-derived spacing remain exact. The policy's separate
    0.05 mm corner-displacement acceptance check remains enforced by
    ``adjudicate_geometry`` before this comparison.
    """
    errors = []
    if receipt_geometry.get("shape") != list(recomputed_geometry.get("shape", [])):
        errors.append({"field": "shape", "receipt": receipt_geometry.get("shape"),
                       "recomputed": list(recomputed_geometry.get("shape", []))})
    receipt_spacing = receipt_geometry.get("spacing_zyx")
    recomputed_spacing = list(recomputed_geometry.get("spacing_zyx", []))
    if receipt_spacing != recomputed_spacing:
        errors.append({"field": "spacing_zyx", "receipt": receipt_spacing,
                       "recomputed": recomputed_spacing})
    try:
        receipt_affine = np.asarray(receipt_geometry.get("affine_zyx"), dtype=np.float64)
        recomputed_affine = np.asarray(recomputed_geometry.get("affine_zyx"), dtype=np.float64)
        if receipt_affine.shape != (4, 4) or recomputed_affine.shape != (4, 4):
            raise ValueError("affine_zyx must be 4x4")
        if not np.isfinite(receipt_affine).all() or not np.isfinite(recomputed_affine).all():
            raise ValueError("affine_zyx contains nonfinite values")
        affine_max_abs_diff = float(np.max(np.abs(receipt_affine - recomputed_affine)))
        if affine_max_abs_diff > GEOMETRY_AFFINE_ATOL:
            errors.append({"field": "affine_zyx", "max_abs_diff": affine_max_abs_diff,
                           "atol": GEOMETRY_AFFINE_ATOL})
    except (TypeError, ValueError) as exc:
        affine_max_abs_diff = None
        errors.append({"field": "affine_zyx", "error": f"{type(exc).__name__}: {exc}"})
    if receipt_geometry.get("context") != recomputed_geometry.get("context"):
        errors.append({"field": "context", "receipt": receipt_geometry.get("context"),
                       "recomputed": recomputed_geometry.get("context")})
    return {"matches": not errors, "errors": errors,
            "affine_max_abs_diff": affine_max_abs_diff,
            "affine_atol": GEOMETRY_AFFINE_ATOL}


def audit_archives(archives: list[Path], out: Path, *, source_document: Path | None = None,
                   source_snapshot: Path | None = None, geometry_summary: Path | None = None) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"output exists; refusing overwrite: {out}")
    by_name = {p.name: p for p in archives}
    if set(by_name) != set(EXPECTED_ARCHIVES):
        raise ValueError(f"--archives must name exactly: {', '.join(EXPECTED_ARCHIVES)}")
    source_document = source_document or PUBLISHED_SOURCE / "SOURCES.json"
    source_snapshot = source_snapshot or PUBLISHED_SOURCE / "adjudicated/source_snapshot.json"
    geometry_summary = geometry_summary or PUBLISHED_SOURCE / "adjudicated/SUMMARY.json"
    document_status = "UNAVAILABLE"
    if source_document.is_file():
        if sha256_file(source_document) != PAPER_SHA256:
            raise ValueError(f"--source-document SHA256 mismatch; expected {PAPER_SHA256}")
        document_status = "PINNED_PUBLICATION_SHA_VERIFIED"

    archive_rows, patients = [], []
    for name, split in EXPECTED_ARCHIVES.items():
        result, rows = _archive_inventory(by_name[name], split)
        archive_rows.append(result)
        patients.extend(rows)
    counts = {split: sum(row["split"] == split for row in patients) for split in EXPECTED_COUNTS}
    ids = [row["patient_id"] for row in patients]
    if len(ids) != len(set(ids)):
        raise ValueError("patient IDs overlap across archive partitions")
    snapshot = json.loads(source_snapshot.read_text()) if source_snapshot.is_file() else None
    identity_match = False
    if snapshot is not None:
        source_by_id = {r["patient_id"]: r for r in snapshot.get("records", [])}
        identity_match = len(source_by_id) == 345 and all(
            (s := source_by_id.get(r["patient_id"])) is not None
            and s.get("split") == r["split"] and s.get("path") == r["member"]
            and s.get("header_sha256") == r["header_sha256"] for r in patients)
    summary = json.loads(geometry_summary.read_text()) if geometry_summary.is_file() else None
    geometry_errors = []
    geometry_comparisons = {"matched": 0, "max_affine_abs_diff": 0.0,
                            "affine_atol": GEOMETRY_AFFINE_ATOL}
    geometry_preconditions = {
        "source_identity_match": identity_match,
        "summary_status_ready": bool(summary and summary.get("status") == "GEOMETRY_READY"),
        "summary_identity_hash_match": bool(summary and snapshot and summary.get("source_identities_sha256") == snapshot.get("source_identities_sha256")),
        "summary_split_counts_match": bool(summary and summary.get("split_counts") == EXPECTED_COUNTS),
        "publication_sha_verified": document_status == "PINNED_PUBLICATION_SHA_VERIFIED",
    }
    for check, passed in geometry_preconditions.items():
        if not passed:
            geometry_errors.append({"check": check, "error": "precondition failed"})
    geometry_match = all(geometry_preconditions.values())
    if geometry_match:
        try:
            from self_audit.candidate_fuse.geometry import adjudicate_geometry
            import nibabel as nib
        except Exception as exc:
            geometry_errors.append({"check": "geometry_engine_import",
                                    "error": f"{type(exc).__name__}: {exc}"})
            geometry_match = False
    if geometry_match:
        receipt_path = geometry_summary.parent / "per_patient.json"
        try:
            receipt_rows = json.loads(receipt_path.read_text())
        except Exception as exc:
            receipt_rows = []
            geometry_errors.append({"check": "per_patient_receipt_read", "path": str(receipt_path),
                                    "error": f"{type(exc).__name__}: {exc}"})
            geometry_match = False
        receipt_by_id = {r.get("patient_id"): r for r in receipt_rows if isinstance(r, dict)}
        if set(receipt_by_id) != set(ids):
            geometry_errors.append({"check": "per_patient_identity", "expected_count": len(ids),
                                    "actual_count": len(receipt_by_id),
                                    "missing_ids": sorted(set(ids) - set(receipt_by_id)),
                                    "extra_ids": sorted(set(receipt_by_id) - set(ids))})
            geometry_match = False
        snapshot_by_id = {r.get("patient_id"): r for r in snapshot.get("records", []) if isinstance(r, dict)}
        for row in patients:
            if not geometry_match and not receipt_by_id:
                break
            pid = row["patient_id"]
            rec = receipt_by_id.get(pid)
            source_row = snapshot_by_id.get(pid)
            if rec is None or source_row is None:
                geometry_errors.append({"patient_id": pid, "check": "patient_identity", "error": "missing matching source or geometry row"})
                geometry_match = False
                continue
            try:
                header_bytes = bytes.fromhex(source_row["header_hex"])
                geometry = adjudicate_geometry(nib.Nifti1Header.from_fileobj(io.BytesIO(header_bytes)),
                                               policy=summary["policy"])
            except Exception as exc:
                geometry_errors.append({"patient_id": pid, "check": "geometry_recompute",
                                        "error": f"{type(exc).__name__}: {exc}"})
                geometry_match = False
                continue
            comparison = _compare_geometry(rec.get("geometry", {}), geometry["geometry"])
            geometry_comparisons["max_affine_abs_diff"] = max(
                geometry_comparisons["max_affine_abs_diff"], comparison["affine_max_abs_diff"] or 0.0)
            if (rec.get("status") != "GEOMETRY_READY"
                    or rec.get("source_identity", {}).get("header_sha256") != row["header_sha256"]
                    or not comparison["matches"]):
                geometry_errors.append({"patient_id": pid, "check": "receipt_geometry_match",
                                        "receipt_status": rec.get("status"),
                                        "source_header_sha256_match": rec.get("source_identity", {}).get("header_sha256") == row["header_sha256"],
                                        "comparison": comparison})
                geometry_match = False
            else:
                geometry_comparisons["matched"] += 1
                row["adjudicated_geometry"] = geometry["geometry"]
                row["geometry_status"] = "GEOMETRY_READY"
        if geometry_match and CLASS_MAPPING != {"0": 0, "1": 3, "2": 2, "3": 1}:
            geometry_errors.append({"check": "canonical_class_mapping", "error": "M&Ms class mapping differs from reviewed FUSE implementation"})
            geometry_match = False
        if not geometry_match:
            for row in patients:
                row.pop("adjudicated_geometry", None)
                row.pop("geometry_status", None)
    status = "GEOMETRY_READY" if geometry_match else "RAW_VERIFIED"
    out.mkdir(parents=True)
    report = {
        "schema": "self_audit.pg_mnms_archive_audit.v1", "status": status,
        "training_ready": False, "training": "NOT_RUN", "scope": "archive_integrity_and_image_headers_only_no_extraction_no_voxel_or_mask_array_reads",
        "archive_integrity": "PASS_ALL_THREE" if len(archive_rows) == 3 else "INCOMPLETE",
        "archives": archive_rows, "split_counts": counts,
        "class_mapping_source_to_canonical": CLASS_MAPPING if geometry_match else None,
        "class_mapping_status": "PINNED_ORGANIZER_EVIDENCE" if geometry_match else "UNVERIFIED_PENDING_MATCHING_RELEASE_RECEIPT",
        "patient_records": patients,
        "raw_header_findings": {"invalid_count": sum(r["raw_header"]["status"] != "VALID" for r in patients),
                                 "unknown_unit_count": sum("unknown_spatial_units" in r["raw_header"]["findings"] for r in patients),
                                 "by_split": {s: sum(r["split"] == s and r["raw_header"]["status"] != "VALID" for r in patients) for s in EXPECTED_COUNTS}},
        "geometry_policy": {"status": "APPLIED_MATCHING_RECEIPT" if geometry_match else "NOT_APPLIED",
                             "policy": "mnms_open345_mm_polar_v1" if geometry_match else None,
                             "source_document_status": document_status,
                             "source_document_sha256": PAPER_SHA256 if geometry_match else None,
                             "source_inventory_match": identity_match,
                             "geometry_receipt_match": geometry_match,
                             "geometry_validation_preconditions": geometry_preconditions,
                             "geometry_comparison": geometry_comparisons,
                             "geometry_validation_errors": geometry_errors,
                             "raw_header_findings_retained": True,
                             "interpretation": "dataset-level mm evidence; raw unit tags are never relabeled"},
        "mask_caveat": "Never cast source masks to uint8 before confirming dtype/scaling; non-unit scl_slope or fractional values can change labels. No mask payload was read.",
        "acdc_queue_effect": "independent; geometry blockers do not change ACDC queue status",
    }
    (out / "MNMS_AUDIT.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, nargs=3, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-document", type=Path, help="pinned publication file; SHA256 must match the FUSE policy")
    parser.add_argument("--source-snapshot", type=Path, help="committed original-M&Ms source snapshot")
    parser.add_argument("--geometry-summary", type=Path, help="committed FUSE geometry SUMMARY.json")
    args = parser.parse_args()
    print(json.dumps(audit_archives(args.archives, args.out, source_document=args.source_document,
                                    source_snapshot=args.source_snapshot, geometry_summary=args.geometry_summary), indent=2))


if __name__ == "__main__":
    main()
