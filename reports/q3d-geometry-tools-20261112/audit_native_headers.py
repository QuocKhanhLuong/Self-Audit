#!/usr/bin/env python3
"""Read-only Phase A diagnostic: NIfTI headers and manifest JSON, never voxel/GT arrays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--historical-k8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    historical_sha = sha(args.historical_k8)
    if historical_sha != "e524c7531079996232fe71cdfe982a46a5950d49d91a8203eee396b0e67434a3":
        raise ValueError("historical K8 manifest SHA mismatch")
    historical = json.loads(args.historical_k8.read_text())
    manifests = {k: json.loads((args.prepared / f"k{k}/train.json").read_text()) for k in (5, 8, 10)}
    development_path = args.prepared / "k8/development.json"
    development = json.loads(development_path.read_text())
    ids = lambda m: sorted(r["id"] for pool in ("labeled", "unlabeled") for r in m[pool])
    train_ids = ids(manifests[8])
    dev_ids = sorted(r["id"] for r in development["patients"])
    test_ids = sorted(p.name for p in (args.raw / "testing").iterdir() if p.is_dir() and p.name.startswith("patient"))
    if train_ids != ids(historical) or any(ids(m) != train_ids for m in manifests.values()):
        raise ValueError("restored TRAIN cohort differs from historical cohort")
    if (len(train_ids), len(dev_ids), len(test_ids)) != (80, 18, 50):
        raise ValueError("unexpected cohort counts")
    if set(train_ids) & set(dev_ids) or set(train_ids + dev_ids) & set(test_ids):
        raise ValueError("cohort overlap")
    labeled = {k: sorted(r["id"] for r in m["labeled"]) for k, m in manifests.items()}
    if labeled[8] != sorted(r["id"] for r in historical["labeled"]):
        raise ValueError("restored L8 differs from historical L8")
    rows = []
    for cohort, patients, manifest in (("TRAIN80", train_ids, manifests[8]), ("DEV18", dev_ids, development), ("TEST50", test_ids, None)):
        for patient in patients:
            paths = list((args.raw / ("testing" if cohort == "TEST50" else "training") / patient).glob("*4d.nii*"))
            row = {"cohort": cohort, "patient_id": patient, "errors": []}
            geometry = manifest["geometry"]["by_patient"].get(patient) if manifest else None
            row["manifest_spacing_zyx_mm"] = geometry.get("spacing_zyx_mm") if geometry else None
            row["manifest_affine_zyx"] = geometry.get("affine_zyx") if geometry else None
            row["manifest_geometry_status"] = "DECLARED" if geometry else "NOT_RUN_NO_TEST_MANIFEST"
            if len(paths) != 1:
                row["errors"].append("expected exactly one original CINE header")
                rows.append(row)
                continue
            path = paths[0]
            image = nib.load(str(path))
            header = image.header
            zooms = np.asarray(header.get_zooms()[:3], dtype=float)
            affine = np.asarray(image.affine, dtype=float)
            lengths = np.linalg.norm(affine[:3, :3], axis=0)
            row.update({
                "source_path": str(path), "source_file_sha256": sha(path),
                "source_header_sha256": hashlib.sha256(header.binaryblock).hexdigest(),
                "source_shape_xyzt": list(image.shape), "header_spacing_zyx_native": zooms[::-1].tolist(),
                "header_spatial_unit": header.get_xyzt_units()[0],
                "sform_code": int(header["sform_code"]), "qform_code": int(header["qform_code"]),
                "sform_xyz": header.get_sform().tolist(), "qform_xyz": header.get_qform().tolist(),
                "nibabel_selected_affine_xyz": affine.tolist(),
                "selected_affine_axis_lengths_zyx_native": lengths[::-1].tolist(),
                "pixdim_affine_agree": bool(np.allclose(lengths, zooms, rtol=1e-5, atol=1e-6)),
            })
            if not np.isfinite(zooms).all() or (zooms <= 0).any():
                row["errors"].append("nonfinite/nonpositive pixdim")
            if np.allclose(zooms, 1, rtol=0, atol=1e-6):
                row["errors"].append("unit pixdim placeholder")
            if not row["pixdim_affine_agree"]:
                row["errors"].append("pixdim conflicts with nibabel-selected affine")
            if np.allclose(lengths, 1, rtol=0, atol=1e-6):
                row["errors"].append("selected affine unit-spacing placeholder")
            if not row["sform_code"] and not row["qform_code"]:
                row["errors"].append("no coded authoritative affine; nibabel fallback is not certified")
            if row["header_spatial_unit"] == "unknown":
                row["errors"].append("spatial unit unspecified; no override applied in this audit")
            rows.append(row)
            if len(rows) % 20 == 0:
                print(f"audited_headers={len(rows)}/148", flush=True)
    result = {
        "schema": "self_audit.q3d.phase_a_original_header_audit.v1", "status": "BLOCKED_HEADER_GEOMETRY" if any(r["errors"] for r in rows) else "PASS",
        "evidence_status": "READ_ONLY_ORIGINAL_HEADER_AUDIT", "nibabel_version": nib.__version__,
        "scope": "headers and manifest metadata only; no voxel/GT array access, raw-header edit, geometry substitution or training",
        "historical_k8_sha256": historical_sha,
        "manifest_sha256": {**{f"k{k}": sha(args.prepared / f"k{k}/train.json") for k in manifests}, "development": sha(development_path)},
        "labeled_patient_ids": labeled, "nested_l5_l8_l10": set(labeled[5]) < set(labeled[8]) < set(labeled[10]),
        "test_manifest_status": "NOT_RUN_NO_TEST_MANIFEST", "rows": rows,
        "cohorts": {cohort: {"patients": sum(r["cohort"] == cohort for r in rows), "pixdim_affine_conflicts": sum(r["cohort"] == cohort and r.get("pixdim_affine_agree") is False for r in rows), "patients_with_errors": sum(r["cohort"] == cohort and bool(r["errors"]) for r in rows)} for cohort in ("TRAIN80", "DEV18", "TEST50")},
        "downstream": {phase: "NOT_RUN" for phase in ("corrected_manifests", "teacher_stride_phase", "L8_physiology", "PLAN_v2", "GPU_profile", "campaign")},
    }
    args.out.mkdir(parents=True)
    (args.out / "header-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    (args.out / "header-audit.csv").write_text(buffer.getvalue())
    print(json.dumps({key: result[key] for key in ("status", "cohorts", "nested_l5_l8_l10")}, indent=2))


if __name__ == "__main__":
    main()
