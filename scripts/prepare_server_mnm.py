#!/usr/bin/env python3
"""Prepare one hash-pinned ACDC-L4 + M&Ms-L4 compact TRAIN manifest.

This is a metadata-driven adapter for the existing self_audit.data.v2 and
shared_data contracts. It does not infer a split by scanning directories. The
M&Ms source manifest must enumerate the official TRAIN-L candidate cines and
TRAIN-U cines; TRAIN-U records are image-only. Only the four selected M&Ms
TRAIN-L masks are opened and remapped. Evaluation manifests are supplied as a
separate generic name -> manifest/SHA map and are read image-only here.

Manufactured fixtures may call prepare_joint_data(..., synthetic=True). The
public CLI always prepares real-source manifests and never starts training.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.data import write_json
from self_audit.model import digest
from self_audit.shared_data import (
    DEFAULT_SEED,
    KNOWN_EXCLUDED_ACDC_IDS,
    KNOWN_QUARANTINED_MNM_IDS,
    RAW_MNM_MAPPING,
    build_shared_manifest,
    filter_clean_patients,
    is_quarantined_patient,
    namespace_patient_id,
    select_shared_train_labels,
    validate_and_remap_mask,
    validate_no_image_overlap,
    validate_shared_manifest,
)


MNM_SOURCE_SCHEMA = "self_audit.mnm.raw_source.v1"
EVALUATION_MAPS_SCHEMA = "self_audit.server_evaluation_maps.v1"
PREPARATION_SCHEMA = "self_audit.server_joint_preparation.v1"
DEFAULT_PRIOR_RECEIPT = ROOT / "reports/baseline_recovery/MNM_PREPARATION_SOURCE.json"
DEFAULT_PRIOR_RECEIPT_SHA256 = "82f6a05b38fa8e69607b0fd6fcecf0062809353dfb47cc8107cbbfb865bf030b"
EXPECTED_RAW_MAPPING = {str(k): int(v) for k, v in RAW_MNM_MAPPING.items()}
SAFE_MAP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
EVAL_SCHEMAS = {
    "self_audit.data.v2",
    "self_audit.evaluation.v1",
    "self_audit.evaluation.v2",
    "self_audit.compact.data.v1",
    "self_audit.native_manifest.v1",
}
EVAL_ROLES = {"development", "external_frozen", "test", "evaluation"}


def _sha_pin(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be a pinned 64-character SHA-256")
    return value.lower()


def _load_json_pin(path: Path | str, expected_sha256: str, name: str) -> tuple[Path, dict[str, Any]]:
    p = Path(path).resolve()
    expected = _sha_pin(expected_sha256, f"{name} SHA-256")
    if not p.is_file():
        raise FileNotFoundError(f"{name} not found: {p}")
    actual = digest(p)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    try:
        value = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be valid JSON: {p}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return p, value


def _resolve_beneath(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"{name} path must be a nonempty string")
    p = Path(relative)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        raise ValueError(f"{name} must be a safe path relative to the declared M&Ms root")
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} resolves outside the declared M&Ms root: {resolved}")
    return resolved


def _resolve_manifest_path(parent: Path, path_value: Any, name: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{name} path must be a nonempty string")
    p = Path(path_value)
    resolved = (p if p.is_absolute() else parent / p).resolve()
    if ".." in p.parts:
        raise ValueError(f"{name} path may not contain parent traversal")
    return resolved


def _raw_ids(value: Any, role: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(pid, str) or not pid.strip() for pid in value):
        raise ValueError(f"M&Ms split {role} must be a list of patient IDs")
    ids = [pid.strip() for pid in value]
    if len(ids) != len(set(ids)):
        raise ValueError(f"M&Ms split {role} contains duplicate patient IDs")
    if any(":" in pid or any(part in pid for part in ("..", "/", "\\", "\0", "\n", "\r", "\t")) for pid in ids):
        raise ValueError(f"M&Ms split {role} contains an unsafe or already-namespaced ID")
    return ids


def read_split_metadata(path: Path, expected_sha256: str) -> tuple[dict[str, list[str]], str]:
    """Read the same explicit train/val/test JSON split forms as prepare_acdc.py."""
    expected = _sha_pin(expected_sha256, "M&Ms split metadata SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"pinned M&Ms split metadata not found: {path}")
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"M&Ms split metadata SHA-256 mismatch: expected {expected}, got {actual}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("pinned M&Ms split metadata must be JSON") from exc
    if not isinstance(obj, dict):
        raise ValueError("pinned M&Ms split metadata must contain an object")
    if obj.get("dataset") not in (None, "M&Ms"):
        raise ValueError("pinned split metadata does not identify the official M&Ms dataset")

    result: dict[str, list[str]] = {}
    for role in ("train", "val", "test"):
        variants = []
        for key in (role, role + "_patients"):
            if isinstance(obj.get(key), list):
                variants.append(obj[key])
        for key in ("splits", "locked_splits"):
            group = obj.get(key, {}).get(role) if isinstance(obj.get(key), dict) else None
            if isinstance(group, dict):
                group = group.get("patients")
            if isinstance(group, list):
                variants.append(group)
        if not variants:
            if role == "train":
                raise ValueError("pinned M&Ms split metadata has no explicit TRAIN patient list")
            result[role] = []
            continue
        normalized = [_raw_ids(items, role) for items in variants]
        if any(set(items) != set(normalized[0]) for items in normalized[1:]):
            raise ValueError(f"conflicting M&Ms split representations for {role}")
        result[role] = normalized[0]

    all_ids = result["train"] + result["val"] + result["test"]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("pinned M&Ms TRAIN/VAL/TEST split IDs overlap")
    return result, actual


def validate_prior_mnm_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify the repository's exact prior MnM ontology/quarantine receipt."""
    receipt_path, report = _load_json_pin(path, expected_sha256, "prior M&Ms receipt")
    if expected_sha256.lower() != DEFAULT_PRIOR_RECEIPT_SHA256:
        raise ValueError("this server preparation pins the reviewed repository M&Ms receipt SHA")
    if report.get("kind") != "dataset":
        raise ValueError("prior M&Ms receipt is not a dataset preparation report")
    prior = report.get("receipt")
    if not isinstance(prior, dict) or prior.get("status") != "PASS":
        raise ValueError("prior M&Ms receipt does not declare status PASS")
    mapping = prior.get("mapping")
    if not isinstance(mapping, dict) or {str(k): int(v) for k, v in mapping.items()} != EXPECTED_RAW_MAPPING:
        raise ValueError("prior M&Ms receipt does not pin the required raw-to-canonical mapping")
    quarantined = set(prior.get("quarantined_image_duplicate_ids", []))
    if quarantined != set(KNOWN_QUARANTINED_MNM_IDS):
        raise ValueError("prior M&Ms receipt quarantine differs from the repository's reviewed list")
    if prior.get("unlabeled_mask_reads") != 0:
        raise ValueError("prior M&Ms receipt reports an unlabeled mask read")
    return {
        "path": str(receipt_path),
        "sha256": expected_sha256.lower(),
        "prior_server_artifact": report.get("server_artifact"),
        "prior_server_artifact_sha256": report.get("artifact_sha256"),
        "prior_train_manifest": prior.get("train_manifest"),
        "prior_train_manifest_sha256": prior.get("sha256"),
        "mapping": EXPECTED_RAW_MAPPING,
        "quarantined_ids": sorted(quarantined),
        "limitation": report.get("limitation"),
        "current_payload_verified": False,
    }


def _validate_raw_source_manifest(raw: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if raw.get("schema") != MNM_SOURCE_SCHEMA or raw.get("dataset") != "M&Ms":
        raise ValueError(f"M&Ms raw source must declare schema {MNM_SOURCE_SCHEMA!r} and dataset 'M&Ms'")
    if type(raw.get("synthetic")) is not bool:
        raise ValueError("M&Ms raw source must declare an explicit boolean synthetic flag")
    split_ref = raw.get("source_split")
    if not isinstance(split_ref, dict) or set(split_ref) != {"path", "sha256"}:
        raise ValueError("M&Ms raw source must pin source_split as {path, sha256}")
    _sha_pin(split_ref.get("sha256"), "M&Ms source split SHA-256")
    labeled = raw.get("train_labeled_candidates")
    unlabeled = raw.get("train_unlabeled")
    if not isinstance(labeled, list) or not isinstance(unlabeled, list):
        raise ValueError("M&Ms raw source must list train_labeled_candidates and train_unlabeled")

    def validate_records(records: list[Any], role: str) -> list[dict[str, Any]]:
        output = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"M&Ms {role} records must be objects")
            expected = {"id", "image", "image_sha256"} if role == "unlabeled" else {"id", "image", "image_sha256", "masks"}
            if set(record) != expected:
                raise ValueError(f"M&Ms {role} record must contain exactly {sorted(expected)}")
            patient_id = record.get("id")
            if not isinstance(patient_id, str) or not patient_id.strip() or ":" in patient_id:
                raise ValueError(f"M&Ms {role} record ID must be an unnamespaced string")
            _sha_pin(record.get("image_sha256"), f"M&Ms image SHA-256 for {patient_id}")
            if role == "unlabeled":
                output.append(dict(record))
                continue
            masks = record.get("masks")
            if not isinstance(masks, dict) or not masks:
                raise ValueError(f"M&Ms labeled candidate {patient_id} must list mask frames")
            clean_masks = {}
            for frame, mask_record in masks.items():
                if not isinstance(frame, str) or not frame.isdigit() or str(int(frame)) != frame:
                    raise ValueError(f"M&Ms mask frame key for {patient_id} must be a zero-based integer string")
                if not isinstance(mask_record, dict) or set(mask_record) not in (
                    {"path", "sha256"}, {"path", "sha256", "source_frame"}
                ):
                    raise ValueError(f"M&Ms mask frame {frame} for {patient_id} must contain path, sha256, and optional source_frame")
                _sha_pin(mask_record.get("sha256"), f"M&Ms mask SHA-256 for {patient_id} frame {frame}")
                if "source_frame" in mask_record and (type(mask_record["source_frame"]) is not int or mask_record["source_frame"] < 0):
                    raise ValueError(f"M&Ms source_frame for {patient_id} frame {frame} must be a nonnegative integer")
                clean_masks[frame] = dict(mask_record)
            output.append({**record, "masks": clean_masks})
        ids = [r["id"] for r in output]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate M&Ms patient IDs in {role} records")
        return output

    labeled_records = validate_records(labeled, "labeled")
    unlabeled_records = validate_records(unlabeled, "unlabeled")
    all_ids = [r["id"] for r in labeled_records + unlabeled_records]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("M&Ms TRAIN labeled/unlabeled source IDs overlap")
    return labeled_records, unlabeled_records, dict(split_ref)


def _load_evaluation_maps(path: Path, expected_sha256: str, *, synthetic: bool) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    index_path, index = _load_json_pin(path, expected_sha256, "evaluation maps index")
    if index.get("schema") != EVALUATION_MAPS_SCHEMA:
        raise ValueError(f"evaluation maps index schema must be {EVALUATION_MAPS_SCHEMA}")
    maps = index.get("maps")
    if not isinstance(maps, dict) or not maps:
        raise ValueError("evaluation maps index must contain a nonempty 'maps' object")

    prepared_index: dict[str, Any] = {"schema": EVALUATION_MAPS_SCHEMA, "maps": {}}
    eval_records: list[dict[str, Any]] = []
    eval_ids: set[str] = set()
    for name in sorted(maps):
        if not isinstance(name, str) or SAFE_MAP_NAME.fullmatch(name) is None:
            raise ValueError(f"invalid evaluation map name {name!r}")
        entry = maps[name]
        if not isinstance(entry, dict) or set(entry) != {"domain", "manifest", "sha256"}:
            raise ValueError(f"evaluation map {name} must contain exactly domain, manifest, and sha256")
        domain = entry.get("domain")
        if domain not in ("acdc", "mnm"):
            raise ValueError(f"evaluation map {name} domain must be 'acdc' or 'mnm'")
        manifest_path = _resolve_manifest_path(index_path.parent, entry.get("manifest"), f"evaluation map {name}")
        manifest_sha = _sha_pin(entry.get("sha256"), f"evaluation map {name} manifest SHA-256")
        m_path, manifest = _load_json_pin(manifest_path, manifest_sha, f"evaluation map {name} manifest")
        if manifest.get("schema") not in EVAL_SCHEMAS or manifest.get("role") not in EVAL_ROLES:
            raise ValueError(f"evaluation map {name} has unsupported schema/role")
        if type(manifest.get("synthetic")) is not bool or manifest["synthetic"] != synthetic:
            raise ValueError(f"evaluation map {name} synthetic flag does not match the preparation identity")
        patients = manifest.get("patients")
        if not isinstance(patients, list) or not patients:
            raise ValueError(f"evaluation map {name} must have a nonempty patients list")

        mapped_patient_ids: list[str] = []
        for row in patients:
            if not isinstance(row, dict):
                raise ValueError(f"evaluation map {name} patient rows must be objects")
            original_id = row.get("id")
            if not isinstance(original_id, str) or not original_id.strip():
                raise ValueError(f"evaluation map {name} patient ID must be a nonempty string")
            patient_id = namespace_patient_id(domain, original_id)
            declared_domain = row.get("domain")
            if declared_domain is not None and declared_domain != domain:
                raise ValueError(f"evaluation map {name} patient {original_id} domain conflicts with map metadata")
            if is_quarantined_patient(patient_id):
                raise ValueError(f"quarantined patient {patient_id} appears in evaluation map {name}")
            image_path = _resolve_manifest_path(m_path.parent, row.get("image"), f"evaluation image {patient_id}")
            image_sha = _sha_pin(row.get("image_sha256"), f"evaluation image SHA-256 for {patient_id}")
            mapped_patient_ids.append(patient_id)
            eval_records.append({"id": patient_id, "image": str(image_path), "image_sha256": image_sha})
        if len(mapped_patient_ids) != len(set(mapped_patient_ids)):
            raise ValueError(f"duplicate patient IDs in evaluation map {name}")
        overlap = eval_ids & set(mapped_patient_ids)
        if overlap:
            raise ValueError(f"patients occur in multiple evaluation maps: {sorted(overlap)}")
        eval_ids.update(mapped_patient_ids)
        prepared_index["maps"][name] = {
            "domain": domain,
            "manifest_path": str(m_path),
            "manifest_sha256": manifest_sha,
            "role": manifest["role"],
            "patient_ids": mapped_patient_ids,
        }
    return {
        "path": str(index_path),
        "sha256": expected_sha256.lower(),
        "prepared_index": prepared_index,
    }, eval_records, eval_ids


def _load_raw_nifti(path: Path, expected_sha256: str, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    actual_sha = digest(path)
    expected_sha = _sha_pin(expected_sha256, f"{name} SHA-256")
    if actual_sha != expected_sha:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required to prepare raw M&Ms NIfTI inputs") from exc
    return nib.load(str(path)), actual_sha


def normalize_mnm_cine(raw_xyzt: np.ndarray) -> np.ndarray:
    """Apply the shared image-only per-cine 0.5/99.5 normalization to native TZYX."""
    raw = np.asarray(raw_xyzt)
    if raw.ndim != 4 or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"raw M&Ms cine must be finite numeric XYZT, got {raw.shape}/{raw.dtype}")
    lo, hi = np.percentile(raw, [0.5, 99.5])
    normalized = np.clip((raw - lo) / max(float(hi - lo), 1e-6), 0, 1)
    result = normalized.transpose(3, 2, 1, 0).astype(np.float32)
    if result.ndim != 4 or not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise ValueError("normalized M&Ms cine is not finite TZYX in [0,1]")
    return result


def _validate_prepared_cine(path: Path, expected_sha256: str, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    actual = digest(path)
    expected = _sha_pin(expected_sha256, f"{name} SHA-256")
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"{name} must be a prepared NPY cine") from exc
    if array.ndim != 4 or array.dtype.kind != "f" or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError(f"{name} must be finite normalized TZYX in [0,1]")
    return array


def _validate_acdc_manifest(path: Path, expected_sha256: str, *, synthetic: bool) -> tuple[Path, dict[str, Any], str]:
    manifest_path, manifest = _load_json_pin(path, expected_sha256, "ACDC TRAIN manifest")
    if manifest.get("schema") != "self_audit.data.v2" or manifest.get("role") != "training":
        raise ValueError("ACDC source must be self_audit.data.v2 role training")
    if type(manifest.get("synthetic")) is not bool or manifest["synthetic"] != synthetic:
        raise ValueError("ACDC source synthetic flag does not match preparation identity")
    if manifest.get("shared") is True:
        raise ValueError("ACDC source must be a single-domain candidate manifest, not an existing shared manifest")
    labeled, unlabeled = manifest.get("labeled"), manifest.get("unlabeled")
    if not isinstance(labeled, list) or not labeled or not isinstance(unlabeled, list) or not unlabeled:
        raise ValueError("ACDC source must contain nonempty labeled and unlabeled TRAIN records")
    ids = []
    for row in labeled:
        if not isinstance(row, dict) or set(row) != {"id", "image", "image_sha256", "masks"} or not row["masks"]:
            raise ValueError("ACDC labeled records must have id, image, image_sha256, and masks")
        ids.append(row["id"])
        _sha_pin(row["image_sha256"], f"ACDC image SHA-256 for {row['id']}")
    for row in unlabeled:
        if not isinstance(row, dict) or set(row) != {"id", "image", "image_sha256"}:
            raise ValueError("ACDC unlabeled records must be image-only")
        ids.append(row["id"])
        _sha_pin(row["image_sha256"], f"ACDC image SHA-256 for {row['id']}")
    if len(ids) != len(set(ids)):
        raise ValueError("ACDC TRAIN patient IDs are duplicated")
    if manifest.get("label_budget") != len(labeled):
        raise ValueError("ACDC source label_budget does not match its labeled records")
    eval_ids = manifest.get("evaluation_ids", [])
    if not isinstance(eval_ids, list) or any(not isinstance(pid, str) for pid in eval_ids):
        raise ValueError("ACDC source evaluation_ids must be a list of patient IDs")
    return manifest_path, manifest, digest(manifest_path)


def _resolved_acdc_record(row: Mapping[str, Any], base: Path, *, with_masks: bool) -> dict[str, Any]:
    record = {"id": row["id"], "image": str(_resolve_manifest_path(base, row["image"], f"ACDC image {row['id']}")),
              "image_sha256": row["image_sha256"]}
    if with_masks:
        if not isinstance(row.get("masks"), dict) or not row["masks"]:
            raise ValueError(f"selected ACDC patient {row['id']} has no reference masks")
        masks = {}
        for frame, mask in row["masks"].items():
            if not isinstance(frame, str) or not frame.isdigit() or not isinstance(mask, dict) or set(mask) != {"path", "sha256"}:
                raise ValueError(f"invalid selected ACDC mask metadata for {row['id']} frame {frame}")
            masks[frame] = {"path": str(_resolve_manifest_path(base, mask["path"], f"ACDC mask {row['id']} frame {frame}")),
                            "sha256": _sha_pin(mask["sha256"], f"ACDC mask SHA-256 for {row['id']} frame {frame}")}
        record["masks"] = masks
    return record


def _validate_acdc_selected_masks(record: Mapping[str, Any], cine: np.ndarray) -> None:
    for frame, mask_record in record["masks"].items():
        if not 0 <= int(frame) < cine.shape[0]:
            raise ValueError(f"selected ACDC mask frame {frame} is outside cine {record['id']}")
        mask_path = Path(mask_record["path"])
        if not mask_path.is_file() or digest(mask_path) != mask_record["sha256"]:
            raise ValueError(f"selected ACDC mask hash mismatch for {record['id']} frame {frame}")
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if mask.shape != cine.shape[1:] or mask.dtype.kind not in "uib" or not np.isin(mask, [0, 1, 2, 3, 255]).all():
            raise ValueError(f"selected ACDC mask shape/ontology mismatch for {record['id']} frame {frame}")


def _validate_source_split_binding(raw_manifest: Mapping[str, Any], source_root: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    split_ref = raw_manifest["source_split"]
    split_path = _resolve_beneath(source_root, split_ref["path"], "M&Ms source split metadata")
    split, split_sha = read_split_metadata(split_path, split_ref["sha256"])
    return split, {"path": str(split_path), "sha256": split_sha}


def _raw_record_paths(record: Mapping[str, Any], source_root: Path, *, labeled: bool) -> tuple[Path, list[tuple[str, Path, str, int | None]]]:
    image_path = _resolve_beneath(source_root, record["image"], f"M&Ms image {record['id']}")
    masks: list[tuple[str, Path, str, int | None]] = []
    if labeled:
        for frame, meta in record["masks"].items():
            masks.append((frame, _resolve_beneath(source_root, meta["path"], f"M&Ms mask {record['id']} frame {frame}"),
                          meta["sha256"], meta.get("source_frame")))
    return image_path, masks


def _read_and_canonicalize_mask(path: Path, expected_sha256: str, *, cine_shape_xyz: tuple[int, int, int],
                                cine_affine: np.ndarray, patient_id: str, frame: str, source_frame: int | None,
                                out_path: Path) -> dict[str, Any]:
    image, source_sha = _load_raw_nifti(path, expected_sha256, f"M&Ms TRAIN-L mask {patient_id} frame {frame}")
    raw = np.asarray(image.dataobj)
    if raw.ndim == 4:
        if source_frame is None:
            raise ValueError(f"M&Ms 4D mask {patient_id} frame {frame} requires an explicit source_frame")
        if not 0 <= source_frame < raw.shape[3]:
            raise ValueError(f"M&Ms source_frame {source_frame} is outside mask cine for {patient_id} frame {frame}")
        raw = raw[..., source_frame]
    elif source_frame is not None:
        raise ValueError(f"M&Ms 3D mask {patient_id} frame {frame} cannot declare source_frame")
    if raw.ndim != 3 or tuple(raw.shape) != cine_shape_xyz:
        raise ValueError(f"M&Ms TRAIN-L mask {patient_id} frame {frame} must resolve to a native XYZ volume")
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"M&Ms TRAIN-L mask {patient_id} frame {frame} must be numeric")
    if not np.isfinite(raw).all():
        raise ValueError(f"M&Ms TRAIN-L mask {patient_id} frame {frame} contains nonfinite values")
    if not np.allclose(np.asarray(image.affine), cine_affine, atol=1e-5, rtol=1e-6):
        raise ValueError(f"M&Ms TRAIN-L mask {patient_id} frame {frame} affine differs from its cine")
    rounded = np.rint(raw)
    roundoff = float(np.max(np.abs(raw - rounded))) if raw.size else 0.0
    if roundoff > 1e-4:
        raise ValueError(f"M&Ms TRAIN-L mask {patient_id} frame {frame} has unexplained scaling roundoff {roundoff}")
    integer_mask = rounded.astype(np.int64, copy=False)
    canonical = validate_and_remap_mask(
        integer_mask,
        "mnm",
        "raw_mnm",
        provenance_receipt={"mapping": "raw_mnm"},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, canonical.transpose(2, 1, 0).astype(np.uint8, copy=False))
    return {
        "path": str(out_path.resolve()),
        "sha256": digest(out_path),
        "source_path": str(path.resolve()),
        "source_sha256": source_sha,
        "max_integer_roundoff": roundoff,
        "mapping": EXPECTED_RAW_MAPPING,
    }


def prepare_joint_data(
    *,
    acdc_manifest_path: Path | str,
    acdc_manifest_sha256: str,
    mnm_root: Path | str,
    mnm_source_manifest_path: Path | str,
    mnm_source_manifest_sha256: str,
    evaluation_maps_path: Path | str,
    evaluation_maps_sha256: str,
    out_dir: Path | str,
    prior_receipt_path: Path | str = DEFAULT_PRIOR_RECEIPT,
    prior_receipt_sha256: str = DEFAULT_PRIOR_RECEIPT_SHA256,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Prepare canonical shared TRAIN data from pinned metadata and raw M&Ms.

    The seed and budgets are deliberately fixed at ACDC4 + M&Ms4, total L8,
    seed 42. All selection is complete before any M&Ms mask is opened.
    """
    if type(synthetic) is not bool:
        raise ValueError("synthetic must be an explicit bool")
    seed = DEFAULT_SEED
    acdc_path, acdc_manifest, _ = _validate_acdc_manifest(Path(acdc_manifest_path), acdc_manifest_sha256, synthetic=synthetic)
    source_root = Path(mnm_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"M&Ms raw root not found: {source_root}")
    mnm_manifest_path_resolved, raw_manifest = _load_json_pin(mnm_source_manifest_path, mnm_source_manifest_sha256, "M&Ms raw source manifest")
    mnm_labeled_source, mnm_unlabeled_source, _ = _validate_raw_source_manifest(raw_manifest)
    if raw_manifest.get("synthetic") != synthetic:
        raise ValueError("M&Ms source synthetic flag does not match preparation identity")

    split, split_info = _validate_source_split_binding(raw_manifest, source_root)
    train_source_ids = [r["id"] for r in mnm_labeled_source + mnm_unlabeled_source]
    if set(train_source_ids) != set(split["train"]) or len(train_source_ids) != len(split["train"]):
        raise ValueError("M&Ms source records do not exactly cover the pinned TRAIN split; no split inference is allowed")

    prior_receipt = validate_prior_mnm_receipt(Path(prior_receipt_path), prior_receipt_sha256)
    map_receipt, eval_records, eval_ids = _load_evaluation_maps(
        Path(evaluation_maps_path), evaluation_maps_sha256, synthetic=synthetic
    )

    # Resolve the fixed L4+L4 subset only from metadata. Quarantined IDs are
    # filtered by shared_data before any image or mask payload is opened.
    acdc_l_clean, acdc_l_quarantined = filter_clean_patients([r["id"] for r in acdc_manifest["labeled"]], "acdc")
    acdc_u_clean, acdc_u_quarantined = filter_clean_patients([r["id"] for r in acdc_manifest["unlabeled"]], "acdc")
    mnm_l_clean, mnm_l_quarantined = filter_clean_patients([r["id"] for r in mnm_labeled_source], "mnm")
    mnm_u_clean, mnm_u_quarantined = filter_clean_patients([r["id"] for r in mnm_unlabeled_source], "mnm")
    if len(acdc_l_clean) <= 4 or len(mnm_l_clean) <= 4:
        raise ValueError("each clean TRAIN-L candidate pool must contain more than four patients for deterministic L4 selection")
    selected_acdc, selected_mnm = select_shared_train_labels(
        [r["id"] for r in acdc_manifest["labeled"]],
        [r["id"] for r in mnm_labeled_source],
        acdc_budget=4,
        mnm_budget=4,
        seed=seed,
    )
    selected_acdc_ids, selected_mnm_ids = set(selected_acdc), set(selected_mnm)

    acdc_l_by_id = {r["id"]: r for r in acdc_manifest["labeled"]}
    acdc_u_by_id = {r["id"]: r for r in acdc_manifest["unlabeled"]}
    mnm_l_by_id = {r["id"]: r for r in mnm_labeled_source}
    mnm_u_by_id = {r["id"]: r for r in mnm_unlabeled_source}
    acdc_l_records = [_resolved_acdc_record(acdc_l_by_id[pid], acdc_path.parent, with_masks=True)
                      for pid in acdc_l_clean if pid in selected_acdc_ids]
    acdc_demoted = [_resolved_acdc_record(acdc_l_by_id[pid], acdc_path.parent, with_masks=False)
                    for pid in acdc_l_clean if pid not in selected_acdc_ids]
    acdc_u_records = [_resolved_acdc_record(acdc_u_by_id[pid], acdc_path.parent, with_masks=False) for pid in acdc_u_clean]
    acdc_u_records.extend(acdc_demoted)

    out = Path(out_dir).resolve()
    if out.exists():
        raise ValueError(f"output directory must be brand new: {out}")
    protected = [source_root, acdc_path.parent, mnm_manifest_path_resolved.parent,
                 Path(evaluation_maps_path).resolve().parent, Path(prior_receipt_path).resolve().parent]
    protected.extend(Path(entry["manifest_path"]).parent for entry in map_receipt["prepared_index"]["maps"].values())
    for root in protected:
        if out == root or out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError(f"output directory overlaps protected input path: {root}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    (out / "masks").mkdir()

    try:
        # Prepare every eligible TRAIN cine using image-only normalization.
        # M&Ms U records have no mask metadata and no mask path resolution.
        mnm_image_records: dict[str, dict[str, Any]] = {}
        mnm_affines: dict[str, np.ndarray] = {}
        mnm_shapes_xyz: dict[str, tuple[int, int, int]] = {}
        raw_image_receipts: dict[str, dict[str, Any]] = {}
        mnm_train_source = [r for r in mnm_labeled_source if r["id"] in mnm_l_clean]
        mnm_train_source.extend(r for r in mnm_unlabeled_source if r["id"] in mnm_u_clean)
        for row in mnm_train_source:
            raw_image_path, _ = _raw_record_paths(row, source_root, labeled=False)
            image, source_sha = _load_raw_nifti(raw_image_path, row["image_sha256"], f"M&Ms TRAIN image {row['id']}")
            raw_xyzt = np.asarray(image.dataobj)
            cine_tzyx = normalize_mnm_cine(raw_xyzt)
            image_out = out / "images" / f"mnm_{row['id']}.npy"
            np.save(image_out, cine_tzyx)
            mnm_image_records[row["id"]] = {
                "id": row["id"],
                "image": str(image_out.resolve()),
                "image_sha256": digest(image_out),
            }
            mnm_affines[row["id"]] = np.asarray(image.affine, dtype=np.float64)
            mnm_shapes_xyz[row["id"]] = tuple(int(v) for v in raw_xyzt.shape[:3])
            raw_image_receipts[row["id"]] = {
                "source_path": str(raw_image_path),
                "source_sha256": source_sha,
                "prepared_path": str(image_out.resolve()),
                "prepared_sha256": digest(image_out),
                "shape_tzyx": list(cine_tzyx.shape),
            }

        mnm_l_records = [dict(mnm_image_records[pid]) for pid in sorted(selected_mnm_ids)]
        mnm_u_records: list[dict[str, Any]] = []
        for pid in sorted(mnm_image_records):
            if pid not in selected_mnm_ids:
                # Strip all candidate annotation paths by constructing a fresh
                # image-only record. No mask source is resolved, hashed or read.
                mnm_u_records.append({key: mnm_image_records[pid][key] for key in ("id", "image", "image_sha256")})

        # Validate selected ACDC arrays and all training/evaluation image
        # payloads before any ACDC mask or evaluation mask is accessed.
        acdc_cine_by_id: dict[str, np.ndarray] = {}
        for row in acdc_l_records + acdc_u_records:
            acdc_cine_by_id[row["id"]] = _validate_prepared_cine(Path(row["image"]), row["image_sha256"], f"ACDC TRAIN image {row['id']}")
        for row in eval_records:
            _validate_prepared_cine(Path(row["image"]), row["image_sha256"], f"evaluation image {row['id']}")

        training_records = acdc_l_records + acdc_u_records + mnm_l_records + mnm_u_records
        required_eval_ids = {
            namespace_patient_id("acdc", pid) for pid in acdc_manifest.get("evaluation_ids", [])
        }
        required_eval_ids.update(namespace_patient_id("mnm", pid) for pid in split["val"] + split["test"])
        missing_eval = required_eval_ids - eval_ids
        if missing_eval:
            raise ValueError(f"generic evaluation maps do not cover all declared source evaluation IDs: {sorted(missing_eval)}")
        train_ids = {namespace_patient_id("acdc", r["id"]) for r in acdc_l_records + acdc_u_records}
        train_ids.update(namespace_patient_id("mnm", r["id"]) for r in mnm_l_records + mnm_u_records)
        if train_ids & eval_ids:
            raise ValueError(f"TRAIN/evaluation ID overlap: {sorted(train_ids & eval_ids)}")
        validate_no_image_overlap(training_records, eval_records=eval_records, evaluation_ids=sorted(eval_ids))

        # Label payloads are touched only after the complete TRAIN/evaluation
        # image overlap preflight succeeds. ACDC masks are checked separately;
        # M&Ms remapping happens once into new canonical arrays.
        for row in acdc_l_records:
            _validate_acdc_selected_masks(row, acdc_cine_by_id[row["id"]])

        mnm_l_by_id = {r["id"]: r for r in mnm_labeled_source}
        mask_transformations: dict[str, Any] = {}
        for record in mnm_l_records:
            pid = record["id"]
            source_record = mnm_l_by_id[pid]
            _, mask_paths = _raw_record_paths(source_record, source_root, labeled=True)
            mask_by_frame = {frame: (path, sha, source_frame) for frame, path, sha, source_frame in mask_paths}
            prepared_masks: dict[str, Any] = {}
            for frame in source_record["masks"]:
                mask_path, mask_sha, source_frame = mask_by_frame[frame]
                out_mask = out / "masks" / f"mnm_{pid}_t{int(frame):03d}.npy"
                receipt = _read_and_canonicalize_mask(
                    mask_path,
                    mask_sha,
                    cine_shape_xyz=mnm_shapes_xyz[pid],
                    cine_affine=mnm_affines[pid],
                    patient_id=pid,
                    frame=frame,
                    source_frame=source_frame,
                    out_path=out_mask,
                )
                prepared_masks[frame] = {"path": receipt["path"], "sha256": receipt["sha256"]}
                mask_transformations[f"mnm:{pid}:{frame}"] = receipt
            if not prepared_masks:
                raise ValueError(f"selected M&Ms TRAIN patient {pid} has no prepared masks")
            record["masks"] = prepared_masks

        manifest = build_shared_manifest(
            acdc_labeled=acdc_l_records,
            acdc_unlabeled=acdc_u_records,
            mnm_labeled=mnm_l_records,
            mnm_unlabeled=mnm_u_records,
            evaluation_ids=sorted(eval_ids),
            acdc_budget=4,
            mnm_budget=4,
            seed=seed,
            synthetic=synthetic,
            candidate_pool_mode="restricted_manifest_labeled",
            source_manifests={
                "acdc": {"path": str(acdc_path), "sha256": acdc_manifest_sha256.lower()},
                "mnm": {"path": str(mnm_manifest_path_resolved), "sha256": mnm_source_manifest_sha256.lower()},
            },
            mapping_provenance={
                "acdc": "canonical; permits no remap",
                "mnm": "canonical after raw_mnm remapped once (0->0, 1->3, 2->2, 3->1, 255->255)",
                "mnm_source_split": split_info,
                "mnm_prior_receipt": prior_receipt,
                "mnm_mask_transformations": mask_transformations,
            },
        )
        train_path = out / "train.json"
        write_json(train_path, manifest)
        train_sha = digest(train_path)
        validate_shared_manifest(train_path)

        map_index = copy.deepcopy(map_receipt["prepared_index"])
        map_index_path = out / "EVALUATION_MAPS.json"
        write_json(map_index_path, map_index)
        map_index_sha = digest(map_index_path)

        prep = {
            "schema": PREPARATION_SCHEMA,
            "status": "PASS",
            "evidence_scope": "manufactured_software_fixture" if synthetic else "source_preparation_only",
            "train_manifest": str(train_path),
            "train_manifest_sha256": train_sha,
            "evaluation_maps": str(map_index_path),
            "evaluation_maps_sha256": map_index_sha,
            "source_inputs": {
                "acdc_train_manifest": {"path": str(acdc_path), "sha256": acdc_manifest_sha256.lower()},
                "mnm_raw_source_manifest": {"path": str(mnm_manifest_path_resolved), "sha256": mnm_source_manifest_sha256.lower()},
                "mnm_source_split": split_info,
                "mnm_prior_receipt": prior_receipt,
                "evaluation_maps_index": map_receipt,
            },
            "budget": {"acdc": 4, "mnm": 4, "total": 8},
            "seed": seed,
            "selected_labeled_ids": {
                "acdc": sorted(namespace_patient_id("acdc", pid) for pid in selected_acdc_ids),
                "mnm": sorted(namespace_patient_id("mnm", pid) for pid in selected_mnm_ids),
            },
            "selection_pool": {
                "acdc": "eligible labeled IDs in the hash-pinned prepared source manifest",
                "mnm": "eligible TRAIN-L candidate IDs in the hash-pinned raw source manifest, checked against pinned official TRAIN split IDs",
                "mode": "restricted_manifest_labeled",
            },
            "quarantined_ids_excluded": {
                "acdc": sorted(namespace_patient_id("acdc", pid) for pid in set(acdc_l_quarantined + acdc_u_quarantined)),
                "mnm": sorted(namespace_patient_id("mnm", pid) for pid in set(mnm_l_quarantined + mnm_u_quarantined)),
            },
            "mnm_images": raw_image_receipts,
            "mnm_mask_transformations": mask_transformations,
            "preprocessing": {
                "normalization": "per-cine image-only .5/99.5 percentiles",
                "array_shape_order": "native TZYX",
                "float_range": "[0, 1]",
                "spatial_resizing": "none",
            },
            "mask_mapping": {
                "source_ontology": "raw_mnm",
                "canonical_ontology": "BG0/RV1/MYO2/LV3/ignore255",
                "mapping_applied_once": EXPECTED_RAW_MAPPING,
            },
            "mask_access": {
                "selected_train_mask_files_opened": sum(len(r["masks"]) for r in mnm_l_records),
                "unlabeled_mask_reads": 0,
                "unlabeled_output_records_have_mask_paths": False,
                "evaluation_mask_files_opened": 0,
            },
            "source_train_ids": {
                "mnm_train_count": len(split["train"]),
                "mnm_train_ids_sha256": digest_bytes("\n".join(sorted(split["train"]))),
                "mnm_evaluation_ids": sorted(namespace_patient_id("mnm", pid) for pid in split["val"] + split["test"]),
            },
            "acdc_quarantine_source_ids": sorted(KNOWN_EXCLUDED_ACDC_IDS),
            "mnm_quarantine_source_ids": sorted(KNOWN_QUARANTINED_MNM_IDS),
        }
        prep_path = out / "PREPARATION.json"
        write_json(prep_path, prep)
        return {"train_manifest": str(train_path), "train_manifest_sha256": train_sha,
                "evaluation_maps": str(map_index_path), "evaluation_maps_sha256": map_index_sha,
                "preparation": str(prep_path), "selected_labeled_ids": prep["selected_labeled_ids"]}
    except Exception as exc:
        failure = {"schema": PREPARATION_SCHEMA, "status": "FAILED", "error": repr(exc),
                   "source_split": split_info, "unlabeled_mask_reads": 0}
        try:
            write_json(out / "FAILURE.json", failure)
        except Exception:
            pass
        raise


def digest_bytes(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acdc-manifest", type=Path, required=True)
    parser.add_argument("--acdc-manifest-sha256", required=True)
    parser.add_argument("--mnm-root", type=Path, required=True,
                        help="Existing extracted official M&Ms dataset root; no download or directory scan is performed")
    parser.add_argument("--mnm-source-manifest", type=Path, required=True,
                        help=f"Hash-pinned {MNM_SOURCE_SCHEMA} JSON enumerating raw TRAIN cines and TRAIN-L mask references")
    parser.add_argument("--mnm-source-manifest-sha256", required=True)
    parser.add_argument("--evaluation-maps", type=Path, required=True,
                        help=f"Hash-pinned {EVALUATION_MAPS_SCHEMA} index with generic ACDC/MnM native manifests")
    parser.add_argument("--evaluation-maps-sha256", required=True)
    parser.add_argument("--prior-mnm-receipt", type=Path, default=DEFAULT_PRIOR_RECEIPT)
    parser.add_argument("--prior-mnm-receipt-sha256", default=DEFAULT_PRIOR_RECEIPT_SHA256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_joint_data(
        acdc_manifest_path=args.acdc_manifest,
        acdc_manifest_sha256=args.acdc_manifest_sha256,
        mnm_root=args.mnm_root,
        mnm_source_manifest_path=args.mnm_source_manifest,
        mnm_source_manifest_sha256=args.mnm_source_manifest_sha256,
        evaluation_maps_path=args.evaluation_maps,
        evaluation_maps_sha256=args.evaluation_maps_sha256,
        out_dir=args.out,
        prior_receipt_path=args.prior_mnm_receipt,
        prior_receipt_sha256=args.prior_mnm_receipt_sha256,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
