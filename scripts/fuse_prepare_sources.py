#!/usr/bin/env python3
"""Prepare fixed FUSE source split and missing native TEST cine cache.

This is an offline CPU utility. It reads source manifests and raw cine files,
never TEST masks. TEST image normalization matches ``prepare_acdc.py`` and
native geometry is recovered with the approved ``acdc_pixdim_native_v1``
geometry repair policy. Output is a new directory; source artifacts are read
only. Production CLI locks TRAIN80/VAL18/TEST50 and official TEST IDs 101-150.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.candidate_q3d.geometry_repair import (  # noqa: E402
    ACDC_PIXDIM_NATIVE_V1,
    _patient_geometry,
    validate_geometry_record,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _ids(records, name: str) -> list[str]:
    if not isinstance(records, list):
        raise ValueError(f"{name} records must be a list")
    ids = [record.get("id") for record in records if isinstance(record, dict)]
    if len(ids) != len(records) or any(not isinstance(pid, str) or not pid for pid in ids):
        raise ValueError(f"{name} contains invalid patient records")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} contains duplicate patient IDs")
    return ids


def _source_ids(train_manifest: dict, development_manifest: dict):
    if train_manifest.get("synthetic") is not False or train_manifest.get("dataset", "").lower() != "acdc":
        raise ValueError("source TRAIN must be a real ACDC manifest")
    if development_manifest.get("synthetic") is not False or development_manifest.get("dataset", "").lower() != "acdc":
        raise ValueError("source development must be a real ACDC manifest")
    if train_manifest.get("role") != "training":
        raise ValueError("source TRAIN manifest role must be training")
    if development_manifest.get("role") != "development":
        raise ValueError("source development manifest role must be development")
    labeled = _ids(train_manifest.get("labeled"), "TRAIN labeled")
    unlabeled = _ids(train_manifest.get("unlabeled"), "TRAIN unlabeled")
    if set(labeled) & set(unlabeled) or set(labeled) | set(unlabeled) != set(train_manifest.get("geometry", {}).get("by_patient", {})):
        raise ValueError("TRAIN L/U must be disjoint and jointly equal the source geometry patient IDs")
    train_ids = sorted(labeled + unlabeled)
    development_ids = sorted(_ids(development_manifest.get("patients"), "development"))
    return train_ids, development_ids


def _raw_patient_dirs(raw_root: Path, role: str) -> dict[str, Path]:
    root = raw_root / role
    if not root.is_dir():
        raise ValueError(f"raw {role} directory is missing: {root}")
    result = {}
    for path in root.iterdir():
        if path.is_dir() and re.fullmatch(r"patient\d{3}", path.name):
            result[path.name] = path
    return result


def _normalize_cine(cine_path: Path) -> np.ndarray:
    import nibabel as nib

    image = nib.load(str(cine_path), mmap=False)
    raw = np.asarray(image.dataobj)
    if raw.ndim != 4 or any(size < 1 for size in raw.shape) or not np.isfinite(raw).all():
        raise ValueError(f"raw cine must be finite native XYZT: {cine_path}")
    lo, hi = np.percentile(raw, [0.5, 99.5])
    normalized = np.clip((raw - lo) / max(float(hi - lo), 1e-6), 0, 1)
    result = np.ascontiguousarray(normalized.transpose(3, 2, 1, 0), dtype=np.float32)
    if not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise ValueError(f"normalized cine is invalid: {cine_path}")
    return result


def _test_ids(raw_testing: dict[str, Path], expected_test_ids: list[str] | None) -> list[str]:
    ids = expected_test_ids if expected_test_ids is not None else [f"patient{i:03d}" for i in range(101, 151)]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"patient\d{3}", pid) for pid in ids):
        raise ValueError("expected TEST patient IDs must be unique patientNNN IDs")
    missing = sorted(set(ids) - set(raw_testing))
    if missing:
        raise ValueError(f"raw testing is missing locked TEST patient directories: {missing}")
    return sorted(ids)


def prepare_sources(
    source_train: str | Path,
    source_development: str | Path,
    raw_root: str | Path,
    out: str | Path,
    *,
    _expected_counts: tuple[int, int, int] = (80, 18, 50),
    _expected_test_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Write fixed split and TEST image/geometry source manifest to a new dir.

    Underscored cohort overrides exist solely for tiny CPU fixture tests; the
    CLI never exposes them and always uses the locked 80/18/50 cohort.
    """
    train_path, development_path = Path(source_train).resolve(), Path(source_development).resolve()
    raw_root, out = Path(raw_root).resolve(), Path(out).resolve()
    if out.exists() or out in {train_path, development_path} or out.is_relative_to(raw_root):
        raise ValueError("output must be a new directory outside source/raw roots")
    train_manifest = json.loads(train_path.read_text(encoding="utf-8"))
    development_manifest = json.loads(development_path.read_text(encoding="utf-8"))
    train_ids, development_ids = _source_ids(train_manifest, development_manifest)
    raw_training = _raw_patient_dirs(raw_root, "training")
    raw_testing = _raw_patient_dirs(raw_root, "testing")
    test_ids = _test_ids(raw_testing, _expected_test_ids)
    if (len(train_ids), len(development_ids), len(test_ids)) != _expected_counts:
        raise ValueError(f"source cohort counts must be {_expected_counts}, got {(len(train_ids), len(development_ids), len(test_ids))}")
    if set(train_ids) & set(development_ids) or set(train_ids) & set(test_ids) or set(development_ids) & set(test_ids):
        raise ValueError("TRAIN/development/TEST patient IDs overlap")
    missing_train = sorted(set(train_ids) - set(raw_training))
    missing_development = sorted(set(development_ids) - set(raw_training))
    if missing_train or missing_development:
        raise ValueError(f"raw training is missing source IDs: TRAIN={missing_train}, development={missing_development}")

    # Preflight all test cines, phase metadata, and source geometry before
    # creating output. No TEST masks are discovered or opened.
    import nibabel as nib

    prepared = []
    geometry_by_patient = {}
    audit_rows = []
    raw_hashes = {}
    for pid in test_ids:
        folder = raw_testing[pid]
        cine_candidates = sorted(folder.glob(f"{pid}_4d.nii*"))
        if len(cine_candidates) != 1:
            raise ValueError(f"expected one raw TEST cine for {pid}, got {len(cine_candidates)}")
        cine_path = cine_candidates[0]
        nib_image = nib.load(str(cine_path), mmap=False)
        shape = tuple(int(v) for v in nib_image.shape)
        if len(shape) != 4 or any(v <= 0 for v in shape):
            raise ValueError(f"TEST cine must have positive XYZT shape for {pid}")
        shape_zyx = [shape[2], shape[1], shape[0]]
        spatial_unit = nib_image.header.get_xyzt_units()[0]
        declarations = {}
        if spatial_unit not in {"mm", "meter", "micron"}:
            declarations[pid] = {
                "unit": "mm",
                "provenance": "User-authorized ACDC 2026-11-12 pixdim amendment: unknown spatial units declared millimetres for acdc_pixdim_native_v1.",
            }
        geometry, row = _patient_geometry(
            pid, cine_path, {"shape_zyx": shape_zyx}, declarations, ACDC_PIXDIM_NATIVE_V1,
        )
        if row["errors"] or not geometry:
            raise ValueError(f"TEST native geometry rejected for {pid}: {row['errors']}")
        geometry = validate_geometry_record(geometry)
        cine_hash = _sha(cine_path)
        raw_hashes[pid] = {"cine_path": str(cine_path), "cine_sha256": cine_hash,
                           "source_header_sha256": row["source_header_sha256"]}
        prepared.append((pid, cine_path, cine_hash, row))
        geometry_by_patient[pid] = geometry
        audit_rows.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.building-", dir=out.parent) as temp_dir:
        stage = Path(temp_dir)
        records = []
        for pid, cine_path, cine_hash, row in prepared:
            image = _normalize_cine(cine_path)
            if tuple(image.shape[1:]) != tuple(geometry_by_patient[pid]["shape_zyx"]):
                raise ValueError(f"TEST cine and repaired geometry differ for {pid}")
            staged_array = stage / f"{pid}.npy"
            np.save(staged_array, image, allow_pickle=False)
            records.append({"id": pid, "image": str((out / staged_array.name).resolve()),
                            "image_sha256": _sha(staged_array),
                            "source_cine_path": str(cine_path), "source_cine_sha256": cine_hash})
        test_manifest = {
            "schema": "self_audit.data.v2", "role": "test", "synthetic": False,
            "dataset": "acdc", "patients": records,
            "class_mapping": {"0": "background", "1": "RV", "2": "MYO", "3": "LV", "255": "ignore"},
            "geometry": {"schema": "self_audit.geometry_map.v1", "by_patient": geometry_by_patient},
            "source_geometry_policy": ACDC_PIXDIM_NATIVE_V1,
            "source_geometry_audit": {row["patient_id"]: row for row in audit_rows},
        }
        test_manifest_path = stage / "test.source.json"
        _write_json(test_manifest_path, test_manifest)

        split = {
            "schema": "self_audit.candidate_fuse.split.v1",
            "train": train_ids, "development": development_ids, "test": test_ids,
            "source_receipts": {
                "train_manifest": {"path": str(train_path), "sha256": _sha(train_path)},
                "development_manifest": {"path": str(development_path), "sha256": _sha(development_path)},
                "raw_test_cines": raw_hashes,
            },
        }
        split_path = stage / "split.json"
        _write_json(split_path, split)
        receipt = {
            "schema": "self_audit.candidate_fuse.source_preparation.v1",
            "outputs": {
                "split": {"path": str(out / "split.json"), "sha256": _sha(split_path)},
                "test_source_manifest": {"path": str(out / "test.source.json"), "sha256": _sha(test_manifest_path)},
            },
            "source_manifests": {
                "train": {"path": str(train_path), "sha256": _sha(train_path)},
                "development": {"path": str(development_path), "sha256": _sha(development_path)},
            },
            "counts": {"train": len(train_ids), "development": len(development_ids), "test": len(test_ids)},
            "test_mask_reads": 0,
            "normalization": "global cine percentiles 0.5/99.5, clip to [0,1], transpose XYZT to TZYX, float32",
            "geometry_policy": ACDC_PIXDIM_NATIVE_V1,
            "geometry_audit_rows": audit_rows,
        }
        _write_json(stage / "SOURCE_RECEIPT.json", receipt)
        split_sha = _sha(split_path)
        test_manifest_sha = _sha(test_manifest_path)
        if out.exists():
            raise FileExistsError(f"output appeared during preparation: {out}")
        stage.rename(out)
    return {"split": {"path": str(out / "split.json"), "sha256": split_sha},
            "test": {"path": str(out / "test.source.json"), "sha256": test_manifest_sha},
            "receipt": str(out / "SOURCE_RECEIPT.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--source-development", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_sources(args.source_train, args.source_development,
                                    args.raw_root, args.out), indent=2))


if __name__ == "__main__":
    main()
