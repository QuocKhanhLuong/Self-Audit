#!/usr/bin/env python3
"""Header-only audit for the original M&Ms 2020 images.

No voxel payloads, masks, or labels are read. Affines are reported as found;
the audit never repairs or waives missing/unknown geometry.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "val", "test")
MIN_VALID_TRAIN = 8


def _finite_json(value: Any) -> Any:
    """Convert numpy values recursively and encode nonfinite floats as null."""
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    return value


def audit_header(patient_id: str, split: str, image_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"patient_id": patient_id, "split": split, "image_path": str(image_path)}
    if not image_path.is_file():
        row.update(status="INVALID", findings=["missing_image_header"])
        return row
    try:
        import nibabel as nib
        image = nib.load(str(image_path))
        header = image.header
        affine = np.asarray(image.affine, dtype=np.float64)
        pixdim = np.asarray(header["pixdim"], dtype=np.float64)[1:4]
        zooms = np.asarray(header.get_zooms()[:3], dtype=np.float64)
        linear = affine[:3, :3]
        column_norms = np.linalg.norm(linear, axis=0)
        unit_dirs = linear / np.where(column_norms > 0, column_norms, np.nan)
        gram = unit_dirs.T @ unit_dirs
        findings: list[str] = []
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            findings.append("invalid_affine")
        if not np.isfinite(pixdim).all() or np.any(pixdim <= 0):
            findings.append("invalid_pixdim")
        gram_finite = bool(np.isfinite(gram).all())
        if not gram_finite or not np.allclose(gram, np.eye(3), atol=1e-4, rtol=0):
            findings.append("nonorthogonal_affine")
        if not np.isfinite(column_norms).all() or np.any(column_norms <= 0):
            findings.append("invalid_affine_spacing")
        affine_pixdim_agree = bool(
            np.isfinite(column_norms).all() and np.isfinite(pixdim).all()
            and np.allclose(column_norms, pixdim, rtol=1e-5, atol=1e-5)
        )
        if not affine_pixdim_agree:
            findings.append("affine_pixdim_disagreement")
        spatial_unit, time_unit = header.get_xyzt_units()
        if spatial_unit not in {"mm", "meter", "micron"}:
            findings.append("unknown_spatial_units")
        determinant = float(np.linalg.det(linear))
        if not math.isfinite(determinant) or abs(determinant) <= 1e-12:
            findings.append("singular_affine")
        row.update(
            status="INVALID" if findings else "VALID",
            findings=findings,
            shape=list(image.shape),
            pixdim_xyz=_finite_json(pixdim),
            header_zooms_xyz=_finite_json(zooms),
            affine_column_norms_xyz=_finite_json(column_norms),
            affine_direction_gram=_finite_json(gram),
            affine_pixdim_agree=affine_pixdim_agree,
            spatial_unit=spatial_unit,
            time_unit=time_unit,
            affine=_finite_json(affine),
            sform_code=int(header["sform_code"]),
            qform_code=int(header["qform_code"]),
            determinant=_finite_json(determinant),
        )
    except Exception as exc:
        row.update(status="INVALID", findings=["missing_or_unreadable_header"], error=f"{type(exc).__name__}: {exc}")
    return row


def audit_index(index_path: Path, raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with index_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"patient_id", "split", "image"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"index must include columns {sorted(required)}")
        for source in reader:
            patient_id = (source.get("patient_id") or "").strip()
            split = (source.get("split") or "").strip().lower()
            rel = (source.get("image") or "").strip()
            if not patient_id or split not in SPLITS or not rel:
                raise ValueError(f"index row has missing patient_id/image or unknown split: {source}")
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ValueError(f"index image path must stay inside raw root: {rel}")
            if any(row["patient_id"] == patient_id for row in rows):
                raise ValueError(f"duplicate patient ID in index: {patient_id}")
            if any(row["split"] == split and row["image_path"] == str(raw_root / rel) for row in rows):
                raise ValueError(f"duplicate image path in split {split}: {rel}")
            # Resolve explicit index paths only. Never discover patients from filenames.
            rows.append(audit_header(patient_id, split, raw_root / rel))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seen_ids: set[str] = set()
    for row in rows:
        pid, split = str(row.get("patient_id", "")).strip(), row.get("split")
        if not pid or pid in seen_ids:
            raise ValueError(f"missing or duplicate patient ID in audit rows: {pid!r}")
        if split not in SPLITS:
            raise ValueError(f"unknown split in audit rows: {split!r}")
        seen_ids.add(pid)
    invalid = {split: [r["patient_id"] for r in rows if r["split"] == split and r["status"] != "VALID"]
               for split in SPLITS}
    split_counts = {s: sum(r["split"] == s for r in rows) for s in SPLITS}
    valid_train_ids = [r["patient_id"] for r in rows if r["split"] == "train" and r["status"] == "VALID"]
    blockers: list[str] = []
    if not split_counts["test"]:
        blockers.append("official_test_split_not_present_in_extracted_index_or_data")
    if len(valid_train_ids) < MIN_VALID_TRAIN:
        blockers.append(f"fewer_than_{MIN_VALID_TRAIN}_valid_train_patients")
    if invalid["val"]:
        blockers.append("invalid_validation_geometry_stops_Track_B_and_B0")
    if invalid["test"]:
        blockers.append("invalid_test_geometry_stops_Track_B_and_B0")
    for split, ids in invalid.items():
        if split in {"val", "test"} and ids:
            blockers.append(f"invalid_{split}_patient_ids=" + ",".join(ids))
    blockers.extend((
        "official_train_val_test_split_evidence_unverified",
        "class_mapping_BG_RV_MYO_LV_unverified",
    ))
    exclusions = {"train": invalid["train"], "reason": "invalid_header_or_geometry; excluded without repair"}
    return {
        "schema": "self_audit.fuse_mnms_header_audit.v1",
        "status": "BLOCKED" if blockers else "AUDITED",
        "scope": "header_only_no_voxel_or_mask_payloads",
        "split_counts": split_counts,
        "invalid_patient_ids_by_split": invalid,
        "train_exclusions": exclusions,
        "valid_train_patient_ids": valid_train_ids,
        "blockers": blockers,
        "official_split_evidence": "UNVERIFIED: local index is staging metadata, not authoritative official split evidence",
        "class_mapping_evidence": "UNVERIFIED: authoritative label-value ontology evidence is absent",
    }


def write_audit(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    (out_dir / "per_patient.json").write_text(json.dumps(_finite_json(rows), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = sorted({k for row in rows for k in row})
    with (out_dir / "per_patient.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: json.dumps(_finite_json(v), allow_nan=False) if isinstance(v, (list, dict)) else _finite_json(v)
                          for k, v in row.items()} for row in rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = audit_index(args.index, args.raw_root)
    print(json.dumps(write_audit(rows, args.out), indent=2))


if __name__ == "__main__":
    main()
