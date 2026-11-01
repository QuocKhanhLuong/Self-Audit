#!/usr/bin/env python3
"""Prepare a split-pinned ACDC or M&Ms reference-evaluation manifest.

This adapter only normalizes selected image cines and prepares already
authorized reference frames. It never trains, infers, scores metrics, or opens
TRAIN/U payloads. Input split and (for M&Ms) ED/ES metadata bytes must be
explicitly hash-pinned by the caller.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_acdc import normalize_cine
from self_audit.shared_data import (
    KNOWN_QUARANTINED_MNM_IDS,
    RAW_MNM_MAPPING,
    validate_and_remap_mask,
)


MANIFEST_SCHEMA = "self_audit.data.v2"
RECEIPT_SCHEMA = "self_audit.server_evaluation_preparation.v1"
NORMALIZATION = "per-cine image-only .5/99.5 percentiles; clip [0,1]; native XYZT to TZYX float32"
RAW_MNM_MASK_PATTERN = "{id}_sa_gt.nii[.gz]"
RAW_MNM_IMAGE_PATTERN = "{id}_sa.nii[.gz]"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ACDC_ID_RE = re.compile(r"^patient\d{3}$")
_SAFE_MNM_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_RAW_MNM_MAPPING_JSON = {str(key): int(value) for key, value in RAW_MNM_MAPPING.items()}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_pin(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an explicit 64-character SHA-256")
    return value.lower()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_array(path: Path, array: np.ndarray) -> str:
    with path.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    stored = np.load(path, mmap_mode="r", allow_pickle=False)
    if stored.shape != array.shape or stored.dtype != array.dtype:
        raise RuntimeError(f"array write verification failed: {path}")
    return sha256_file(path)


def _read_pinned_split(path: Path, expected_sha256: str, dataset: str) -> tuple[dict[str, list[str]], str]:
    expected = _sha_pin(expected_sha256, "split JSON SHA-256")
    split_path = path.resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"split JSON not found: {split_path}")
    actual = sha256_file(split_path)
    if actual != expected:
        raise ValueError(f"split JSON SHA-256 mismatch: expected {expected}, got {actual}")
    try:
        raw = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("split JSON must be valid UTF-8 JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("split JSON must contain an object")
    declared_dataset = raw.get("dataset")
    if declared_dataset is not None:
        normalized = str(declared_dataset).strip().lower().replace("&", "").replace(" ", "")
        expected_names = {"acdc"} if dataset == "acdc" else {"mnm", "mnms", "mms", "mandms"}
        if normalized not in expected_names:
            raise ValueError(f"split dataset {declared_dataset!r} does not match requested {dataset}")

    roles: dict[str, list[str]] = {}
    for role in ("train", "val", "test"):
        variants: list[list[Any]] = []
        for key in (role, role + "_patients"):
            if isinstance(raw.get(key), list):
                variants.append(raw[key])
        for group_key in ("splits", "locked_splits"):
            group = raw.get(group_key)
            value = group.get(role) if isinstance(group, Mapping) else None
            if isinstance(value, Mapping):
                value = value.get("patients")
            if isinstance(value, list):
                variants.append(value)
        if not variants:
            raise ValueError(f"split JSON requires an explicit {role.upper()} patient list")
        normalized_variants: list[list[str]] = []
        for values in variants:
            if any(not isinstance(pid, str) or not pid.strip() for pid in values):
                raise ValueError(f"split {role} IDs must be nonempty strings")
            ids = [pid.strip() for pid in values]
            if len(ids) != len(set(ids)):
                raise ValueError(f"split {role} contains duplicate IDs")
            for pid in ids:
                if not _SAFE_MNM_ID_RE.fullmatch(pid):
                    raise ValueError(f"unsafe split patient ID: {pid!r}")
                if dataset == "acdc" and not _ACDC_ID_RE.fullmatch(pid):
                    raise ValueError(f"invalid ACDC patient ID: {pid!r}")
            normalized_variants.append(ids)
        if any(set(values) != set(normalized_variants[0]) for values in normalized_variants[1:]):
            raise ValueError(f"conflicting split representations for {role}")
        roles[role] = normalized_variants[0]

    all_ids = roles["train"] + roles["val"] + roles["test"]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("TRAIN/VAL/TEST split IDs overlap")
    return roles, actual


def _read_mnm_metadata(path: Path, expected_sha256: str) -> tuple[dict[str, dict[str, int]], str]:
    expected = _sha_pin(expected_sha256, "M&Ms metadata CSV SHA-256")
    csv_path = path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"M&Ms metadata CSV not found: {csv_path}")
    actual = sha256_file(csv_path)
    if actual != expected:
        raise ValueError(f"M&Ms metadata CSV SHA-256 mismatch: expected {expected}, got {actual}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = [str(name).strip() if name is not None else "" for name in (reader.fieldnames or [])]
        if len(headers) != len(set(headers)) or not {"External code", "ED", "ES"}.issubset(headers):
            raise ValueError("M&Ms metadata CSV requires unique External code, ED, and ES columns")
        reader.fieldnames = headers
        result: dict[str, dict[str, int]] = {}
        for row_number, row in enumerate(reader, start=2):
            external_code = str(row.get("External code") or "").strip()
            if not _SAFE_MNM_ID_RE.fullmatch(external_code):
                raise ValueError(f"invalid External code at CSV row {row_number}: {external_code!r}")
            if external_code in result:
                raise ValueError(f"duplicate External code in M&Ms metadata CSV: {external_code}")
            frame_indices: dict[str, int] = {}
            for key in ("ED", "ES"):
                value = str(row.get(key) or "").strip()
                if not re.fullmatch(r"\d+", value):
                    raise ValueError(f"M&Ms {key} must be a zero-based integer for {external_code}")
                frame_indices[key.lower()] = int(value)
            if frame_indices["ed"] == frame_indices["es"]:
                raise ValueError(f"M&Ms ED and ES must identify distinct zero-based frames for {external_code}")
            result[external_code] = frame_indices
    return result, actual


def _resolve_output_path(path: Path, protected_paths: list[Path]) -> Path:
    output = path.resolve()
    if output.exists():
        raise ValueError(f"output directory must be new: {output}")
    for protected in protected_paths:
        source = protected.resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"output directory overlaps an input source: {source}")
    return output


def _find_exact_source(root: Path, names: list[str], description: str) -> Path:
    """Find exactly one named source file, without enumerating other patient IDs."""
    root_resolved = root.resolve()
    matches: dict[str, Path] = {}
    for name in names:
        for found in root_resolved.rglob(name):
            if not found.is_file():
                continue
            resolved = found.resolve()
            if not resolved.is_relative_to(root_resolved):
                raise ValueError(f"{description} resolves outside declared raw root: {resolved}")
            matches[str(resolved)] = resolved
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {description} under {root_resolved}, found {len(matches)}")
    return next(iter(matches.values()))


def _require_source_beneath(path: Path, root: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{description} resolves outside declared raw root: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _validate_cine(raw_xyzt: np.ndarray, affine: Any, patient_id: str) -> np.ndarray:
    if raw_xyzt.ndim != 4 or any(int(size) < 1 for size in raw_xyzt.shape):
        raise ValueError(f"cine for {patient_id} must have nonempty native XYZT dimensions")
    if raw_xyzt.dtype.kind not in "fiu" or not np.isfinite(raw_xyzt).all():
        raise ValueError(f"cine for {patient_id} must be finite numeric XYZT data")
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not math.isfinite(float(np.linalg.det(matrix))) or abs(float(np.linalg.det(matrix))) < 1e-8:
        raise ValueError(f"cine for {patient_id} must have a finite invertible native affine")
    return matrix


def _save_image(stage: Path, patient_id: str, raw_xyzt: np.ndarray) -> tuple[str, str, list[int]]:
    normalized = normalize_cine(raw_xyzt)
    if normalized.dtype != np.float32 or normalized.ndim != 4 or not np.isfinite(normalized).all():
        raise ValueError(f"canonical normalization failed for {patient_id}")
    image_name = f"{patient_id}.npy"
    image_sha = _write_array(stage / image_name, normalized)
    return image_name, image_sha, [int(size) for size in normalized.shape]


def _prepare_acdc_patient(
    *, raw_root: Path, stage: Path, patient_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    import nibabel as nib

    cine_path = _find_exact_source(raw_root, [f"{patient_id}_4d.nii", f"{patient_id}_4d.nii.gz"], f"ACDC cine for {patient_id}")
    cine_sha = sha256_file(cine_path)
    cine_image = nib.load(str(cine_path))
    raw_xyzt = np.asarray(cine_image.dataobj)
    cine_affine = _validate_cine(raw_xyzt, cine_image.affine, patient_id)
    image_name, image_sha, shape_tzyx = _save_image(stage, patient_id, raw_xyzt)

    mask_paths = sorted(cine_path.parent.glob(f"{patient_id}_frame*_gt.nii*"))
    if not mask_paths:
        raise ValueError(f"ACDC evaluation patient {patient_id} has no reference frame masks")
    masks: dict[str, dict[str, str]] = {}
    geometry_rows: list[dict[str, Any]] = []
    raw_sources = [{"kind": "cine", "patient_id": patient_id, "path": str(cine_path), "sha256": cine_sha}]
    seen_frames: set[int] = set()
    for mask_path in mask_paths:
        mask_path = _require_source_beneath(mask_path, raw_root, f"ACDC reference mask for {patient_id}")
        match = re.fullmatch(rf"{re.escape(patient_id)}_frame(\d+)_gt\.(nii(?:\.gz)?)", mask_path.name)
        if match is None:
            raise ValueError(f"malformed ACDC reference mask filename: {mask_path.name}")
        frame_index = int(match.group(1)) - 1
        if frame_index < 0 or frame_index >= raw_xyzt.shape[3]:
            raise ValueError(f"ACDC reference frame index is outside cine for {patient_id}: {frame_index}")
        if frame_index in seen_frames:
            raise ValueError(f"duplicate ACDC reference frame index for {patient_id}: {frame_index}")
        seen_frames.add(frame_index)
        frame_path = _require_source_beneath(
            mask_path.with_name(mask_path.name.replace("_gt.nii", ".nii", 1)),
            raw_root,
            f"ACDC frame image for {patient_id} frame {frame_index}",
        )

        mask_sha, frame_sha = sha256_file(mask_path), sha256_file(frame_path)
        mask_image, frame_image = nib.load(str(mask_path)), nib.load(str(frame_path))
        if mask_image.shape != frame_image.shape or not np.allclose(mask_image.affine, frame_image.affine):
            raise ValueError("reference/frame geometry mismatch")
        frame_voxels = np.asarray(frame_image.dataobj)
        if not np.array_equal(frame_voxels, raw_xyzt[..., frame_index]):
            raise ValueError("frame/cine exact voxel bridge failed")
        labels = np.asarray(mask_image.dataobj)
        if labels.ndim != 3 or tuple(int(x) for x in labels.shape) != tuple(int(x) for x in raw_xyzt.shape[:3]):
            raise ValueError(f"ACDC reference mask must be a native XYZ volume for {patient_id} frame {frame_index}")
        if labels.dtype.kind not in "fiu" or not np.isfinite(labels).all() or not np.isin(labels, [0, 1, 2, 3, 255]).all():
            raise ValueError(f"unknown or nonfinite annotation class for {patient_id} frame {frame_index}")
        output_mask = labels.transpose(2, 1, 0).astype(np.uint8)
        output_mask_name = f"{patient_id}_t{frame_index:03d}_mask.npy"
        output_mask_sha = _write_array(stage / output_mask_name, output_mask)
        masks[str(frame_index)] = {"path": output_mask_name, "sha256": output_mask_sha}

        frame_affine = np.asarray(frame_image.affine, dtype=np.float64)
        mask_affine = np.asarray(mask_image.affine, dtype=np.float64)
        geometry_rows.append({
            "patient_id": patient_id,
            "frame_index": frame_index,
            "frame_equals_cine_frame": True,
            "frame_and_reference_affines_match": True,
            "frame_affine_matches_cine": bool(np.allclose(frame_affine, cine_affine)),
            "reference_affine_matches_cine": bool(np.allclose(mask_affine, cine_affine)),
            "frame_affine": frame_affine.tolist(),
            "reference_affine": mask_affine.tolist(),
            "cine_affine": cine_affine.tolist(),
            "frame_affine_delta_from_cine": (frame_affine - cine_affine).tolist(),
            "frame_source_path": str(frame_path),
            "frame_source_sha256": frame_sha,
            "reference_source_path": str(mask_path),
            "reference_source_sha256": mask_sha,
        })
        raw_sources.extend([
            {"kind": "cine_frame", "patient_id": patient_id, "frame_index": frame_index, "path": str(frame_path), "sha256": frame_sha},
            {"kind": "reference_mask", "patient_id": patient_id, "frame_index": frame_index, "path": str(mask_path), "sha256": mask_sha},
        ])

    record = {
        "id": patient_id,
        "domain": "acdc",
        "image": image_name,
        "image_sha256": image_sha,
        "axes": ["T", "Z", "Y", "X"],
        "shape_tzyx": shape_tzyx,
        "affine": cine_affine.tolist(),
        "geometry_source_image": {"path": str(cine_path), "sha256": cine_sha},
        "masks": masks,
        "frame_geometry_bridge": [
            {
                "frame_index": int(row["frame_index"]),
                "frame_affine_matches_cine": row["frame_affine_matches_cine"],
                "reference_affine_matches_cine": row["reference_affine_matches_cine"],
                "frame_affine_delta_from_cine": row["frame_affine_delta_from_cine"],
            }
            for row in geometry_rows
        ],
    }
    return record, geometry_rows, raw_sources


def _prepare_mnm_patient(
    *, raw_root: Path, stage: Path, patient_id: str, ed_es: Mapping[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    import nibabel as nib

    image_path = _find_exact_source(raw_root, [f"{patient_id}_sa.nii", f"{patient_id}_sa.nii.gz"], f"M&Ms short-axis cine for {patient_id}")
    mask_path = _find_exact_source(raw_root, [f"{patient_id}_sa_gt.nii", f"{patient_id}_sa_gt.nii.gz"], f"M&Ms short-axis reference mask for {patient_id}")
    image_sha, mask_sha = sha256_file(image_path), sha256_file(mask_path)
    image, mask_image = nib.load(str(image_path)), nib.load(str(mask_path))
    raw_xyzt = np.asarray(image.dataobj)
    cine_affine = _validate_cine(raw_xyzt, image.affine, patient_id)
    if mask_image.ndim != 4 or tuple(int(x) for x in mask_image.shape[:3]) != tuple(int(x) for x in raw_xyzt.shape[:3]):
        raise ValueError(f"M&Ms reference mask must be native XYZT for {patient_id}")
    if not np.allclose(np.asarray(mask_image.affine), cine_affine, atol=1e-5, rtol=1e-6):
        raise ValueError(f"M&Ms reference mask affine differs from its cine for {patient_id}")

    image_name, output_image_sha, shape_tzyx = _save_image(stage, patient_id, raw_xyzt)
    raw_mask = np.asarray(mask_image.dataobj)
    if raw_mask.dtype.kind not in "fiu" or not np.isfinite(raw_mask).all():
        raise ValueError(f"M&Ms reference mask must be finite numeric data for {patient_id}")
    masks: dict[str, dict[str, str]] = {}
    geometry_rows: list[dict[str, Any]] = []
    for phase in ("ed", "es"):
        frame_index = int(ed_es[phase])
        if not 0 <= frame_index < raw_xyzt.shape[3] or not 0 <= frame_index < raw_mask.shape[3]:
            raise ValueError(f"M&Ms zero-based {phase.upper()} frame is outside image/mask cine for {patient_id}: {frame_index}")
        frame_mask = raw_mask[..., frame_index]
        rounded = np.rint(frame_mask)
        roundoff = float(np.max(np.abs(frame_mask - rounded))) if frame_mask.size else 0.0
        if not roundoff < 1e-4:
            raise ValueError(f"M&Ms reference mask has scaling roundoff >=1e-4 for {patient_id} frame {frame_index}: {roundoff}")
        integer_mask = rounded.astype(np.int64, copy=False)
        canonical = validate_and_remap_mask(
            integer_mask,
            "mnm",
            "raw_mnm",
            provenance_receipt={"mapping": "raw_mnm"},
        )
        output_mask = canonical.transpose(2, 1, 0).astype(np.uint8, copy=False)
        output_mask_name = f"{patient_id}_t{frame_index:03d}_mask.npy"
        output_mask_sha = _write_array(stage / output_mask_name, output_mask)
        masks[str(frame_index)] = {"path": output_mask_name, "sha256": output_mask_sha}
        geometry_rows.append({
            "patient_id": patient_id,
            "phase": phase.upper(),
            "frame_index": frame_index,
            "mask_affine_matches_cine": True,
            "mask_affine": np.asarray(mask_image.affine, dtype=np.float64).tolist(),
            "cine_affine": cine_affine.tolist(),
            "max_integer_roundoff": roundoff,
            "source_mask_path": str(mask_path),
            "source_mask_sha256": mask_sha,
            "mapping_applied_once": _RAW_MNM_MAPPING_JSON,
        })

    record = {
        "id": patient_id,
        "domain": "mnm",
        "image": image_name,
        "image_sha256": output_image_sha,
        "axes": ["T", "Z", "Y", "X"],
        "shape_tzyx": shape_tzyx,
        "affine": cine_affine.tolist(),
        "geometry_source_image": {"path": str(image_path), "sha256": image_sha},
        "masks": masks,
        "metadata_frames_zero_based": {"ed": int(ed_es["ed"]), "es": int(ed_es["es"])},
    }
    raw_sources = [
        {"kind": "cine", "patient_id": patient_id, "path": str(image_path), "sha256": image_sha},
        {"kind": "reference_mask", "patient_id": patient_id, "path": str(mask_path), "sha256": mask_sha},
    ]
    return record, geometry_rows, raw_sources


def prepare_server_evaluation(
    *,
    dataset: str,
    raw_root: str | Path,
    split_json: str | Path,
    split_sha256: str,
    role: str,
    out_dir: str | Path,
    metadata_csv: str | Path | None = None,
    metadata_csv_sha256: str | None = None,
    synthetic: bool,
) -> dict[str, Any]:
    """Prepare one explicit development or test cohort without TRAIN payload reads."""
    dataset = str(dataset).strip().lower()
    role = str(role).strip().lower()
    if dataset not in {"acdc", "mnm"}:
        raise ValueError("dataset must be exactly 'acdc' or 'mnm'")
    if role not in {"development", "test"}:
        raise ValueError("role must be exactly 'development' or 'test'")
    if type(synthetic) is not bool:
        raise ValueError("synthetic must be explicitly boolean")
    root = Path(raw_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"raw root not found: {root}")
    split_path = Path(split_json).resolve()
    split, actual_split_sha = _read_pinned_split(split_path, split_sha256, dataset)
    selected_split_key = "val" if role == "development" else "test"
    requested_ids = list(split[selected_split_key])
    if not requested_ids:
        raise ValueError(f"pinned split contains no explicit {selected_split_key.upper()} IDs")

    metadata_path: Path | None = None
    metadata: dict[str, dict[str, int]] = {}
    actual_metadata_sha: str | None = None
    if dataset == "mnm":
        if metadata_csv is None or metadata_csv_sha256 is None:
            raise ValueError("M&Ms preparation requires both metadata CSV path and SHA-256")
        metadata_path = Path(metadata_csv).resolve()
        metadata, actual_metadata_sha = _read_mnm_metadata(metadata_path, metadata_csv_sha256)
    elif metadata_csv is not None or metadata_csv_sha256 is not None:
        raise ValueError("ACDC preparation does not accept M&Ms metadata arguments")

    quarantined = sorted(pid for pid in requested_ids if dataset == "mnm" and pid in KNOWN_QUARANTINED_MNM_IDS)
    cohort_ids = [pid for pid in requested_ids if pid not in set(quarantined)]
    if not cohort_ids:
        raise ValueError("authorized cohort is empty after the pinned M&Ms quarantine")
    if dataset == "mnm":
        missing = sorted(set(cohort_ids) - metadata.keys())
        if missing:
            raise ValueError(f"M&Ms ED/ES metadata is missing selected cohort IDs: {missing}")

    protected_paths = [root, split_path]
    if metadata_path is not None:
        protected_paths.append(metadata_path)
    output = _resolve_output_path(Path(out_dir), protected_paths)
    output.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": dataset,
        "role": role,
        "raw_root": str(root),
        "split_json": str(split_path),
        "split_sha256": actual_split_sha,
        "metadata_csv": str(metadata_path) if metadata_path else None,
        "metadata_csv_sha256": actual_metadata_sha,
        "cohort_ids": cohort_ids,
        "quarantined_ids_excluded": quarantined,
        "output_dir": str(output),
        "synthetic": synthetic,
        "normalization": NORMALIZATION,
        "mapping": _RAW_MNM_MAPPING_JSON if dataset == "mnm" else "canonical_identity",
        "metadata_frame_index_base": 0 if dataset == "mnm" else None,
    }
    source_paths = [ROOT / "prepare_acdc.py", Path(__file__).resolve()]
    if dataset == "mnm":
        source_paths.append(ROOT / "self_audit" / "shared_data.py")
    source_code_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in source_paths
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        patients: list[dict[str, Any]] = []
        geometry_checks: list[dict[str, Any]] = []
        raw_sources: list[dict[str, str]] = []
        for patient_id in cohort_ids:
            if dataset == "acdc":
                patient, patient_geometry, sources = _prepare_acdc_patient(raw_root=root, stage=staging, patient_id=patient_id)
            else:
                patient, patient_geometry, sources = _prepare_mnm_patient(
                    raw_root=root, stage=staging, patient_id=patient_id, ed_es=metadata[patient_id]
                )
            patients.append(patient)
            geometry_checks.extend(patient_geometry)
            raw_sources.extend(sources)

        manifest_name = f"{role}.json"
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "role": role,
            "synthetic": synthetic,
            "dataset": dataset,
            "domain": dataset,
            "scope": "exposed_development" if role == "development" else "test_reference_preparation_only",
            "source_split": {"path": str(split_path), "sha256": actual_split_sha, "selected_key": selected_split_key},
            "cohort_ids": list(cohort_ids),
            "quarantined_ids_excluded": quarantined,
            "normalization": "prepared",
            "preprocessing": {
                "source_axes": "native XYZT",
                "prepared_axes": "TZYX",
                "image_normalization": NORMALIZATION,
                "spatial_resizing": "none",
            },
            "patients": patients,
        }
        manifest_sha = _write_json(staging / manifest_name, manifest)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "PASS",
            "purpose": "reference_preparation_only; no inference, metrics, or training",
            "dataset": dataset,
            "role": role,
            "synthetic": synthetic,
            "configuration": config,
            "configuration_sha256": _json_sha256(config),
            "source_code_sha256": source_code_hashes,
            "source_split": {"path": str(split_path), "sha256": actual_split_sha},
            "metadata_csv": {"path": str(metadata_path), "sha256": actual_metadata_sha, "frame_index_base": 0} if metadata_path else None,
            "cohort": {
                "source_split_key": selected_split_key,
                "source_ids": requested_ids,
                "prepared_ids": cohort_ids,
                "prepared_ids_sha256": _json_sha256(cohort_ids),
                "quarantined_ids_excluded": quarantined,
                "prepared_patient_count": len(patients),
            },
            "preprocessing": {
                "normalization": NORMALIZATION,
                "axes": "native XYZT to TZYX",
                "cine_affine": "retained from native cine NIfTI without coordinate reorientation",
                "mask_mapping": {"source": "raw_mnm", "mapping_applied_once": config["mapping"]} if dataset == "mnm" else "ACDC canonical labels; identity",
            },
            "geometry_checks": geometry_checks,
            "raw_source_files": raw_sources,
            "access": {
                "authorized_role_cine_files_opened": len(patients),
                "authorized_role_reference_mask_files_opened": sum(len(patient["masks"]) for patient in patients),
                "train_reads": 0,
                "train_payload_files_opened": 0,
                "train_mask_files_discovered": 0,
                "unlabeled_payload_files_opened": 0,
                "unlabeled_mask_reads": 0,
                "unlabeled_mask_files_discovered": 0,
                "inference_runs": 0,
                "metrics_computed": 0,
                "training_runs": 0,
            },
            "manifest": {"path": manifest_name, "sha256": manifest_sha},
            "output_dir": str(output),
        }
        _write_json(staging / "PREPARATION.json", receipt)
        if output.exists():
            raise ValueError(f"output directory appeared during preparation: {output}")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "manifest_path": str(output / manifest_name),
        "manifest_sha256": manifest_sha,
        "receipt_path": str(output / "PREPARATION.json"),
        "output_dir": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("acdc", "mnm"))
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--role", required=True, choices=("development", "test"))
    parser.add_argument("--out", required=True, help="new output directory; must not already exist")
    parser.add_argument("--metadata-csv", help="required for M&Ms; exact ED/ES source CSV")
    parser.add_argument("--metadata-csv-sha256", help="required for M&Ms; SHA-256 of exact CSV bytes")
    args = parser.parse_args()
    result = prepare_server_evaluation(
        dataset=args.dataset,
        raw_root=args.raw_root,
        split_json=args.split_json,
        split_sha256=args.split_sha256,
        role=args.role,
        out_dir=args.out,
        metadata_csv=args.metadata_csv,
        metadata_csv_sha256=args.metadata_csv_sha256,
        synthetic=False,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
