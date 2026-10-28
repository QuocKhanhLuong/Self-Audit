"""Compact student inference, historical evaluation, and isolated benchmarking.

Owned by Worker inference per COMPACT_IMPLEMENTATION_CONTRACT.md:
  - Imports load_compact_export directly from self_audit.compact_artifacts (no duplicate models or loaders).
  - Lazy imports historical metric functions (confusion_matrix, class_metrics) from self_audit.evaluation.
  - Strict SHA-256 mandatory pins on all manifests, images, masks, export artifacts, and TRAIN data.
  - Strict validation: role, schema, synthetic parity, class mapping [0,1,2,3,255], axes, finite invertible affine,
    safe unique IDs, in-bounds frame indices, non-overlapping output paths.
  - Explicit normalization contract ('raw' 0.5/99.5 percentiles vs 'prepared' strict [0, 1]).
  - Native streaming masks with pixel-boolean valid_mask support, scoring with ignored padding (255).
  - Mandatory TRAIN provenance verification: ID disjointness and physical byte/voxel content overlap checks on both
    L (labeled) and U (unlabeled) canonical records BEFORE mask reads; binds export lineage data_sha256.
  - Mandatory mask shape == prediction frame shape (no broadcast).
  - Two-phase frozen evaluation: all predictions frozen to PREDICTIONS_FROZEN.json before any reference mask is opened.
  - Exact historical class-first patient macro metrics:
      frame -> patient/class -> classwise patientmeans -> meanclass Macro4 & FG.
      Omits Dice when GT is absent even with false positives.
      Reports extra patient-first aggregate and absent-class false positive tracking with fixed valid count denominator.
  - Benchmark rewrite: streams ALL manifest cases without preloading all contexts; actual E2E includes
    load + preprocess + context + H2D + forward + argmax + D2H + live disk write; separate forward-only timing with CUDA sync;
    cold load measured separately; throughput definitions (volumes/sec, slices/sec); psutil active RAM polling during inference;
    reference uses deployed weights only (no training optimizer payload); missing baseline assets strictly NOT_RUN.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import psutil
import torch
from torch import Tensor, nn

from .compact_artifacts import digest, load_compact_export


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SAFE_ID_RE = re.compile(r"^(?:(?:acdc|mnm):)?[A-Za-z0-9_-]+$")

CLASS_NAMES: Tuple[str, ...] = ("BG", "RV", "MYO", "LV")
FOREGROUND_CLASS_NAMES: Tuple[str, ...] = ("RV", "MYO", "LV")
VALID_CLASSES = frozenset({0, 1, 2, 3, 255})

VALID_EVAL_SCHEMAS = frozenset({
    "self_audit.data.v2",
    "self_audit.evaluation.v1",
    "self_audit.evaluation.v2",
    "self_audit.compact.data.v1",
    "self_audit.native_manifest.v1",
})
VALID_EVAL_ROLES = frozenset({"development", "external_frozen", "test", "evaluation"})
TRAIN_ROLES = frozenset({"train", "training"})


def require_sha256(value: Any, name: str) -> str:
    """Validate that value is a non-null 64-character hexadecimal SHA-256 string."""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a mandatory 64-hex SHA-256 pin, got {value!r}")
    return value.lower()


def write_json(path: Union[str, Path], data: Any) -> str:
    """Write formatted JSON with trailing newline and return its SHA-256."""
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    target = Path(path)
    target.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _records_from_manifest(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return records from a native evaluation or canonical L/U manifest."""
    if "patients" in data:
        records = data["patients"]
        if not isinstance(records, list):
            raise ValueError("manifest patients must be a list")
        return records

    lower_present = "labeled" in data or "unlabeled" in data
    upper_present = "L" in data or "U" in data
    if lower_present and upper_present:
        raise ValueError("manifest must use exactly one L/U record spelling")
    if lower_present:
        labeled, unlabeled = data.get("labeled"), data.get("unlabeled")
    elif upper_present:
        labeled, unlabeled = data.get("L"), data.get("U")
    else:
        return []
    if not isinstance(labeled, list) or not isinstance(unlabeled, list):
        raise ValueError("TRAIN manifest must contain both labeled/unlabeled or L/U record lists")
    return list(labeled) + list(unlabeled)


def _canonical_train_records(data: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read canonical TRAIN L/U records and cross-check any parallel ID lists."""
    lower_present = "labeled" in data or "unlabeled" in data
    upper_present = "L" in data or "U" in data
    if lower_present and upper_present:
        raise ValueError("TRAIN manifest must use exactly one L/U record spelling")
    if lower_present:
        labeled, unlabeled = data.get("labeled"), data.get("unlabeled")
    elif upper_present:
        labeled, unlabeled = data.get("L"), data.get("U")
    else:
        raise ValueError("TRAIN manifest must contain canonical labeled/unlabeled or L/U records")
    if not isinstance(labeled, list) or not isinstance(unlabeled, list) or not labeled or not unlabeled:
        raise ValueError("TRAIN L/U records must both be nonempty lists")
    all_records = list(labeled) + list(unlabeled)
    ids: List[str] = []
    for record in all_records:
        if not isinstance(record, Mapping):
            raise ValueError("TRAIN L/U records must be objects")
        pid = record.get("id")
        if not isinstance(pid, str) or not _SAFE_ID_RE.fullmatch(pid):
            raise ValueError(f"Invalid or unsafe TRAIN patient ID: {pid!r}")
        ids.append(pid)
        image = record.get("image") or record.get("path")
        if not isinstance(image, str) or not image:
            raise ValueError(f"TRAIN patient '{pid}' is missing its image path")
        require_sha256(record.get("image_sha256") or record.get("sha256"), f"TRAIN patient '{pid}' image_sha256")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate patient IDs in TRAIN L/U records: {ids}")

    lids = [record["id"] for record in labeled]
    uids = [record["id"] for record in unlabeled]
    for name, declared, actual in (("lids", data.get("lids"), lids), ("uids", data.get("uids"), uids)):
        if declared is not None and (not isinstance(declared, list) or declared != actual):
            raise ValueError(f"TRAIN manifest {name} does not exactly match its record list")
    return list(labeled), list(unlabeled)


def _voxel_fingerprint(volume: np.ndarray) -> str:
    """Hash decoded cine values canonically, without one full-volume copy."""
    array = np.asarray(volume)
    if array.ndim != 4 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"cine voxel content must be a numeric TZYX array, got {array.shape}/{array.dtype}")
    hasher = hashlib.sha256()
    hasher.update(json.dumps({"shape": list(array.shape), "dtype": "<f8"}, sort_keys=True).encode("utf-8"))
    for ti in range(array.shape[0]):
        frame = np.array(array[ti], dtype="<f8", order="C", copy=True)
        frame[frame == 0] = 0.0
        hasher.update(memoryview(np.ascontiguousarray(frame).view(np.uint8)))
    return hasher.hexdigest()


def _load_raw_tzyx(path: Path, patient_id: str) -> np.ndarray:
    """Decode an image as TZYX for provenance hashing without requiring geometry metadata."""
    if path.name.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        raw = np.asarray(nib.load(str(path)).dataobj)
        if raw.ndim == 4:
            volume = raw.transpose(3, 2, 1, 0)
        elif raw.ndim == 3:
            volume = raw.transpose(2, 1, 0)[None]
        else:
            raise ValueError(f"Unexpected NIfTI dimensions {raw.ndim} for TRAIN patient '{patient_id}'")
    elif path.suffix == ".npy":
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw.ndim == 4:
            volume = raw
        elif raw.ndim == 3:
            volume = raw[None]
        else:
            raise ValueError(f"Unexpected NPY dimensions {raw.ndim} for TRAIN patient '{patient_id}'")
    else:
        raise ValueError(f"Unsupported TRAIN cine format for patient '{patient_id}': {path.name}")
    volume = np.asarray(volume)
    if not np.isfinite(volume).all():
        raise ValueError(f"TRAIN cine contains non-finite values for patient '{patient_id}'")
    return volume


def write_prediction_file(path: Union[str, Path], prediction: np.ndarray) -> Dict[str, Any]:
    """Write and verify the same native uint8 NPY volume format used by infer CLI."""
    target = Path(path)
    array = np.asarray(prediction)
    if array.ndim != 4 or array.dtype != np.uint8:
        raise ValueError(f"prediction output must be uint8 TZYX, got {array.shape}/{array.dtype}")
    if not np.isin(array, [0, 1, 2, 3, 255]).all():
        raise ValueError("prediction output contains an invalid class value")
    with target.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    actual_sha = digest(target)
    stored = np.load(target, mmap_mode="r", allow_pickle=False)
    if stored.shape != array.shape or stored.dtype != np.uint8:
        raise RuntimeError(f"prediction write verification failed for {target}")
    return {"file": target.name, "sha256": actual_sha, "shape_tzyx": list(stored.shape), "dtype": str(stored.dtype)}


def validate_export_preprocessing(export_meta: Mapping[str, Any], normalization: str) -> Dict[str, Any]:
    """Require inference to use the exact preprocessing recipe pinned by the export."""
    from .compact_artifacts import _canonical_preprocessing_spec

    declared = export_meta.get("preprocessing")
    expected = _canonical_preprocessing_spec()
    if declared != expected:
        raise ValueError("compact export preprocessing differs from the canonical runtime preprocessing contract")
    if normalization not in expected["normalization_recipes"]:
        raise ValueError(f"unsupported export normalization mode {normalization!r}")
    return copy.deepcopy(expected["normalization_recipes"][normalization])


def _get_metric_funcs():
    """Lazily import canonical evaluation metrics from self_audit.evaluation."""
    from .evaluation import class_metrics, confusion_matrix
    return confusion_matrix, class_metrics


# ==============================================================================
# Normalization Contract & Image Preprocessing
# ==============================================================================

def normalize_image(
    image: np.ndarray,
    mode: str = "prepared",
) -> np.ndarray:
    """Apply strict image normalization contract.

    Args:
        image: Cine image array (any shape, typically TZYX).
        mode:
          - 'raw': Per-cine 0.5% and 99.5% percentile intensity scaling to [0, 1].
          - 'prepared': Validates that values are already in [0, 1] without renormalizing.

    Returns:
        np.float32 array with values in [0.0, 1.0].
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"expected numpy.ndarray, got {type(image).__name__}")
    if not np.isfinite(image).all():
        raise ValueError("Image contains non-finite (NaN or Inf) values")

    if mode == "raw":
        lo, hi = np.percentile(image, [0.5, 99.5])
        denom = max(float(hi - lo), 1e-6)
        return np.clip((image.astype(np.float32) - np.float32(lo)) / np.float32(denom), 0.0, 1.0).astype(np.float32)
    elif mode == "prepared":
        min_val = float(image.min())
        max_val = float(image.max())
        if min_val < -1e-5 or max_val > 1.0 + 1e-5:
            raise ValueError(
                f"Prepared mode requires values in [0, 1]; got min={min_val}, max={max_val}"
            )
        return np.clip(image.astype(np.float32), 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization mode '{mode}'; expected 'raw' or 'prepared'")


def extract_slice_context(cine: np.ndarray, t: int, z: int) -> Tensor:
    """Extract 3-channel 2.5D context for slice z at time t.

    Matching TrainingData.context: slices [max(0, z-1), z, min(Z-1, z+1)].
    Returns Tensor of shape (1, 3, H, W) dtype float32.
    """
    Z = cine.shape[1]
    inds = [max(0, z - 1), z, min(Z - 1, z + 1)]
    chunk = np.array(cine[t, inds], dtype=np.float32, copy=True)
    return torch.from_numpy(chunk)[None]


# ==============================================================================
# Native Manifest & Cine Loading with Strict Validation
# ==============================================================================

def validate_affine_matrix(affine: Any, patient_id: str) -> np.ndarray:
    """Validate that affine is a real, finite, invertible 4x4 matrix."""
    if affine is None:
        raise ValueError(
            f"Missing real affine metadata for patient '{patient_id}'; "
            "do not invent artificial affines."
        )
    aff = np.asarray(affine, dtype=np.float64)
    if aff.shape != (4, 4):
        raise ValueError(f"Patient '{patient_id}' affine must be a 4x4 matrix, got shape {aff.shape}")
    if not np.isfinite(aff).all():
        raise ValueError(f"Patient '{patient_id}' affine contains non-finite values (NaN or Inf)")

    det = float(np.linalg.det(aff))
    if not np.isfinite(det) or abs(det) < 1e-8:
        raise ValueError(f"Patient '{patient_id}' affine matrix is singular or non-invertible (det={det})")
    return aff


def load_native_manifest(
    manifest_path: Union[str, Path],
    expected_sha: str,
    *,
    expected_roles: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Load and validate native manifest JSON with mandatory SHA and schema checks."""
    clean_expected_sha = require_sha256(expected_sha, "manifest_sha256")
    p = Path(manifest_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Manifest not found: {p}")

    actual_sha = digest(p)
    if actual_sha != clean_expected_sha:
        raise ValueError(f"Manifest hash mismatch: expected {clean_expected_sha}, got {actual_sha}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Manifest root must be a JSON object")
    schema = data.get("schema")
    if schema not in VALID_EVAL_SCHEMAS:
        raise ValueError(f"Unrecognized manifest schema '{schema}', expected one of {sorted(VALID_EVAL_SCHEMAS)}")

    if expected_roles is not None:
        role = data.get("role")
        if role not in expected_roles:
            raise ValueError(f"Manifest role '{role}' not in permitted roles {list(expected_roles)}")

    synthetic = data.get("synthetic")
    if type(synthetic) is not bool:
        raise ValueError("Manifest synthetic flag must be explicitly boolean")

    patients = _records_from_manifest(data)
    if not patients:
        raise ValueError("Manifest contains no patients or canonical TRAIN records")

    pids = []
    for r in patients:
        if not isinstance(r, Mapping):
            raise ValueError("Manifest patient records must be JSON objects")
        pid = r.get("id")
        if not pid or not isinstance(pid, str) or not _SAFE_ID_RE.fullmatch(pid):
            raise ValueError(f"Invalid or unsafe patient ID: {pid!r}")
        pids.append(pid)

    if len(set(pids)) != len(pids):
        raise ValueError(f"Duplicate patient IDs detected in manifest: {pids}")

    return data, actual_sha


def load_patient_cine(
    patient_record: Dict[str, Any],
    manifest_dir: Path,
    normalization: str = "prepared",
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Load patient cine array, pixel valid_mask, real affine, and axis metadata.

    Supports .nii, .nii.gz, and .npy files.
    Preserves real affine metadata (does not invent).

    Returns:
        (cine_tzyx, valid_mask_tzyx, metadata_dict)
    """
    pid = patient_record.get("id")
    if not pid or not isinstance(pid, str) or not _SAFE_ID_RE.fullmatch(pid):
        raise ValueError(f"Invalid or unsafe patient ID: {pid!r}")

    raw_path_str = patient_record.get("image") or patient_record.get("path")
    if not raw_path_str:
        raise ValueError(f"Missing image path in patient record: {pid}")

    img_path = Path(raw_path_str)
    if not img_path.is_absolute():
        img_path = (manifest_dir / img_path).resolve()

    if not img_path.is_file():
        raise FileNotFoundError(f"Cine image file not found: {img_path}")

    expected_sha = require_sha256(
        patient_record.get("image_sha256") or patient_record.get("sha256"),
        f"patient '{pid}' image_sha256",
    )
    actual_sha = digest(img_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"Patient '{pid}' image hash mismatch: expected {expected_sha}, got {actual_sha}"
        )

    affine: Optional[np.ndarray] = None
    header_meta: Dict[str, Any] = {}

    if img_path.name.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        ni = nib.load(str(img_path))
        raw = np.asarray(ni.dataobj)
        affine = validate_affine_matrix(ni.affine, pid)
        header_meta["spatial_units"] = str(ni.header.get_xyzt_units())
        if raw.ndim == 4:
            cine_raw = raw.transpose(3, 2, 1, 0)
        elif raw.ndim == 3:
            cine_raw = raw.transpose(2, 1, 0)[None]
        else:
            raise ValueError(f"Unexpected NIfTI dimensions {raw.ndim} in {img_path}")
    elif img_path.suffix == ".npy":
        raw = np.load(img_path, mmap_mode="r", allow_pickle=False)
        if raw.ndim == 4:
            cine_raw = np.asarray(raw)
        elif raw.ndim == 3:
            cine_raw = np.asarray(raw)[None]
        else:
            raise ValueError(f"Unexpected NPY dimensions {raw.ndim} in {img_path}")

        affine = validate_affine_matrix(patient_record.get("affine"), pid)
    else:
        raise ValueError(f"Unsupported cine format: {img_path.name}")

    if cine_raw.ndim != 4:
        raise ValueError(f"Cine must have 4 dimensions (T, Z, H, W), got {cine_raw.shape}")

    t, z, h, w = cine_raw.shape

    declared_axes = patient_record.get("axes")
    valid_axis_orders = (
        ["time", "slice", "height", "width"],
        ["time", "slice", "y", "x"],
        ["T", "Z", "Y", "X"],
        ["T", "Z", "H", "W"],
    )
    if declared_axes not in valid_axis_orders:
        raise ValueError(f"Patient '{pid}' must declare native TZYX axes; got {declared_axes!r}")

    raw_voxel_sha = _voxel_fingerprint(cine_raw)

    cine_norm = normalize_image(cine_raw, mode=normalization)

    # Optional valid_mask loading and validation (pixel-level boolean array)
    valid_mask: Optional[np.ndarray] = None
    vm_spec = patient_record.get("valid_mask")
    if vm_spec is not None:
        if isinstance(vm_spec, Mapping):
            vm_path_str = vm_spec.get("path")
            vm_p = Path(vm_path_str)
            if not vm_p.is_absolute():
                vm_p = (manifest_dir / vm_p).resolve()
            if not vm_p.is_file():
                raise FileNotFoundError(f"valid_mask file not found: {vm_p}")
            expected_vm_sha = require_sha256(vm_spec.get("sha256"), f"patient '{pid}' valid_mask sha256")
            actual_vm_sha = digest(vm_p)
            if actual_vm_sha != expected_vm_sha:
                raise ValueError(f"valid_mask hash mismatch for patient '{pid}'")
            vm_raw = np.load(vm_p, allow_pickle=False)
        elif isinstance(vm_spec, np.ndarray):
            vm_raw = vm_spec
        else:
            raise ValueError(f"Unexpected valid_mask specification type: {type(vm_spec)}")

        if not np.issubdtype(vm_raw.dtype, np.bool_):
            raise ValueError(f"valid_mask must have boolean dtype, got {vm_raw.dtype}")

        if vm_raw.shape == (t, z, h, w):
            valid_mask = vm_raw
        elif vm_raw.shape == (z, h, w):
            valid_mask = np.broadcast_to(vm_raw, (t, z, h, w))
        elif vm_raw.shape == (h, w):
            valid_mask = np.broadcast_to(vm_raw, (t, z, h, w))
        else:
            raise ValueError(
                f"valid_mask shape {vm_raw.shape} incompatible with cine shape {(t, z, h, w)}"
            )

    masks_dict = patient_record.get("masks", {})
    if not isinstance(masks_dict, Mapping):
        raise ValueError(f"Patient '{pid}' masks must be an object keyed by native frame index")
    seen_ti = set()
    for ti_str, mrec in masks_dict.items():
        try:
            ti = int(ti_str)
        except (TypeError, ValueError):
            raise ValueError(f"Non-integer frame index '{ti_str}' for patient '{pid}'")
        if str(ti) != str(ti_str):
            raise ValueError(f"Non-canonical frame index '{ti_str}' for patient '{pid}'")
        if ti < 0 or ti >= t:
            raise ValueError(f"Frame index {ti} out of bounds for patient '{pid}' with T={t}")
        if ti in seen_ti:
            raise ValueError(f"Duplicate frame index {ti} for patient '{pid}'")
        seen_ti.add(ti)
        if not isinstance(mrec, Mapping) or not isinstance(mrec.get("path"), str) or not mrec.get("path"):
            raise ValueError(f"Patient '{pid}' frame {ti} mask record requires a path")
        require_sha256(mrec.get("sha256"), f"patient '{pid}' frame {ti} mask sha256")

    meta = {
        "id": pid,
        "shape_tzyx": [t, z, h, w],
        "affine": affine.tolist(),
        "axes": declared_axes,
        "image_file": str(img_path),
        "image_sha256": actual_sha,
        "raw_voxel_sha256": raw_voxel_sha,
        "normalization": normalization,
        "has_valid_mask": valid_mask is not None,
        **header_meta,
    }
    return cine_norm, valid_mask, meta


# ==============================================================================
# Streaming Inference (No Full Probability Cache)
# ==============================================================================

def infer_cine_stream(
    model: nn.Module,
    cine: np.ndarray,
    batch_size: int = 16,
    device: str = "cpu",
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run streaming inference over a cine array.

    Generates uint8 class prediction mask array of shape (T, Z, H, W).
    Does NOT store full float32 probability volume to disk/memory.
    Invalid / padding pixels (where valid_mask is False) are set to 255.

    Args:
        model: CompactStudent returning B4HW logits.
        cine: Normalized float32 cine array of shape (T, Z, H, W).
        batch_size: Mini-batch size of 2.5D slices.
        device: 'cpu' or 'cuda'.
        valid_mask: Optional boolean validity mask of shape (T, Z, H, W).

    Returns:
        np.uint8 prediction array with values in {0, 1, 2, 3, 255}.
    """
    t, z, h, w = cine.shape
    pred = np.zeros((t, z, h, w), dtype=np.uint8)

    indices = [(ti, zi) for ti in range(t) for zi in range(z)]
    model.eval()

    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            chunk = indices[offset: offset + batch_size]
            contexts = [extract_slice_context(cine, ti, zi) for ti, zi in chunk]
            batch_x = torch.cat(contexts, dim=0).to(device)

            batch_valid: Optional[Tensor] = None
            if valid_mask is not None:
                v_slices = [
                    torch.from_numpy(valid_mask[ti, zi]).bool()
                    for ti, zi in chunk
                ]
                batch_valid = torch.stack(v_slices, dim=0).to(device)

            logits = model(batch_x, valid_mask=batch_valid)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Model produced non-finite logits during inference")

            chunk_preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

            for idx, ((ti, zi), p_slice) in enumerate(zip(chunk, chunk_preds)):
                if valid_mask is not None:
                    p_slice = np.where(valid_mask[ti, zi], p_slice, 255).astype(np.uint8)
                pred[ti, zi] = p_slice

    return pred


# ==============================================================================
# Historical Class-First Patient Macro Metrics
# ==============================================================================

def compute_historical_metrics(
    frame_rows: List[Dict[str, Any]],
    patient_ids: Sequence[str],
) -> Dict[str, Any]:
    """Compute exact historical class-first patient macro metrics.

    Contract:
      1. Per-frame class Dice: 2*TP / (GT + Pred) if GT > 0 else None.
         Omits Dice when GT is absent even with FP.
      2. Per-patient per-class Dice: mean of present frames for that patient/class.
         If class never present in patient, value is None.
      3. Class macro: mean of patient values across patients where class is present.
      4. Historical macro4: mean of the 4 class macro dice values [BG, RV, MYO, LV].
         Historical foreground macro: mean of 3 foreground class macro dice [RV, MYO, LV].
      5. Extra patient-first aggregate (equal on full cohort, distinct when class absent):
         Average patient foreground mean across patients.
      6. Absent class FP tracking: fixed valid-voxel count denominator.
    """
    patient_metrics: Dict[str, Dict[str, Any]] = {}
    failure_cases: List[Dict[str, Any]] = []
    total_valid_voxels = 0

    for pid in patient_ids:
        p_frames = [r for r in frame_rows if r["patient"] == pid]
        p_classes: Dict[str, Optional[float]] = {}

        for j, c_name in enumerate(CLASS_NAMES):
            vals = [
                f["raw"][j]["dice"]
                for f in p_frames
                if f["raw"][j]["dice"] is not None
            ]
            p_classes[c_name] = float(np.mean(vals)) if vals else None

        avail_fg = [p_classes[c] for c in FOREGROUND_CLASS_NAMES if p_classes[c] is not None]
        patient_fg = float(np.mean(avail_fg)) if avail_fg else None

        avail_all = [p_classes[c] for c in CLASS_NAMES if p_classes[c] is not None]
        patient_all = float(np.mean(avail_all)) if avail_all else None

        p_valid = sum(f.get("valid_voxels", 0) for f in p_frames)
        total_valid_voxels += p_valid

        p_record = {
            **p_classes,
            "foreground_mean": patient_fg,
            "all_class_mean": patient_all,
            "valid_voxels": p_valid,
            "frame_count": len(p_frames),
        }
        patient_metrics[pid] = p_record

        if patient_fg is None:
            failure_cases.append({
                "patient": pid,
                "reason": "absent_or_undefined_foreground",
                "scores": p_classes,
            })
        elif patient_fg < 0.50 or any(p_classes[c] == 0.0 for c in FOREGROUND_CLASS_NAMES if p_classes[c] is not None):
            failure_cases.append({
                "patient": pid,
                "reason": "collapsed_or_low_foreground",
                "foreground_mean": patient_fg,
                "scores": p_classes,
            })

    # Class-first patient macro (Historical exact)
    class_macros: Dict[str, Dict[str, Any]] = {}
    for c_name in CLASS_NAMES:
        present_patients = [
            p[c_name]
            for p in patient_metrics.values()
            if p[c_name] is not None
        ]
        class_macros[c_name] = {
            "dice": float(np.mean(present_patients)) if present_patients else None,
            "present_patients": len(present_patients),
        }

    all_class_macro_vals = [
        class_macros[c]["dice"]
        for c in CLASS_NAMES
        if class_macros[c]["dice"] is not None
    ]
    macro4_class_mean = float(np.mean(all_class_macro_vals)) if all_class_macro_vals else None

    fg_class_macro_vals = [
        class_macros[c]["dice"]
        for c in FOREGROUND_CLASS_NAMES
        if class_macros[c]["dice"] is not None
    ]
    foreground_class_mean = float(np.mean(fg_class_macro_vals)) if fg_class_macro_vals else None

    # Extra Patient-First Aggregates (equal on full cohort, distinct when absent)
    patient_fg_dices = [
        p["foreground_mean"]
        for p in patient_metrics.values()
        if p["foreground_mean"] is not None
    ]
    foreground_patient_macro = float(np.mean(patient_fg_dices)) if patient_fg_dices else None

    patient_all_dices = [
        p["all_class_mean"]
        for p in patient_metrics.values()
        if p["all_class_mean"] is not None
    ]
    all_class_patient_macro = float(np.mean(patient_all_dices)) if patient_all_dices else None

    # Absent class false positive tracking across all frames
    absent_fp_tracking: Dict[str, Dict[str, Any]] = {}
    for j, c_name in enumerate(CLASS_NAMES):
        absent_frames = [
            f for f in frame_rows
            if not f["raw"][j]["reference_present"]
        ]
        absent_valid_voxels = sum(f.get("valid_voxels", 0) for f in absent_frames)
        absent_fps = sum(f["raw"][j]["prediction_count"] for f in absent_frames)
        rate = float(absent_fps / absent_valid_voxels) if absent_valid_voxels > 0 else None
        absent_fp_tracking[c_name] = {
            "absent_frame_count": len(absent_frames),
            "absent_valid_voxels": absent_valid_voxels,
            "absent_false_positive_voxels": absent_fps,
            "absent_fp_rate": rate,
        }

    return {
        "macro_patient": class_macros,
        "class_first_macro4": macro4_class_mean,
        "class_first_foreground_macro": foreground_class_mean,
        "patient_first_all_class_macro": all_class_patient_macro,
        "patient_first_foreground_macro": foreground_patient_macro,
        "patients": patient_metrics,
        "absent_class_tracking": absent_fp_tracking,
        "failure_cases": failure_cases,
        "total_valid_voxels": total_valid_voxels,
    }


# ==============================================================================
# Two-Phase Frozen Evaluation Pipeline
# ==============================================================================

def run_two_phase_evaluation(
    export_path: Union[str, Path],
    export_sha256: str,
    manifest_path: Union[str, Path],
    manifest_sha256: str,
    out_dir: Union[str, Path],
    *,
    train_manifest_path: Union[str, Path],
    train_manifest_sha256: str,
    device: str = "cpu",
    batch_size: int = 16,
    normalization: str = "prepared",
) -> Dict[str, Any]:
    """Execute strict two-phase frozen evaluation.

    MANDATORY PRE-CONDITIONS (Checked BEFORE any side effects):
      - output directory must not exist and must not overlap with manifest directory.
      - train_manifest_path and train_manifest_sha256 are mandatory.
      - export manifest and payload SHA-256 verified.
      - evaluation manifest verified against expected SHA-256.
      - synthetic flag matches export synthetic flag.
      - binds export lineage data_sha256 to train manifest SHA.
      - zero ID overlap between canonical TRAIN (labeled + unlabeled) and EVAL.
      - zero voxel content / physical byte overlap between TRAIN (L+U) and EVAL cines.
      - Phase 1: freezes and hashes all predictions before Phase 2 opens reference masks.
      - Reference mask shape must exactly match prediction frame shape (no broadcast).
    """
    clean_export_sha = require_sha256(export_sha256, "export_sha256")
    clean_manifest_sha = require_sha256(manifest_sha256, "manifest_sha256")
    clean_train_sha = require_sha256(train_manifest_sha256, "train_manifest_sha256")

    out = Path(out_dir).resolve()
    if out.exists():
        raise ValueError(f"Output directory already exists: {out}")

    manifest_p = Path(manifest_path).resolve()
    manifest_dir = manifest_p.parent
    if out == manifest_dir or out.is_relative_to(manifest_dir) or manifest_dir.is_relative_to(out):
        raise ValueError(f"Output directory cannot overlap with evaluation manifest directory: {out}")

    confusion_matrix, class_metrics = _get_metric_funcs()

    # 1. Load the exported artifact and bind its preprocessing and exact TRAIN source.
    model, export_meta = load_compact_export(export_path, clean_export_sha, device=device)
    preprocessing_recipe = validate_export_preprocessing(export_meta, normalization)
    lineage = export_meta.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("compact export lineage is mandatory")
    lineage_train_sha = require_sha256(lineage.get("data_sha256"), "export.lineage.data_sha256")
    if clean_train_sha != lineage_train_sha:
        raise ValueError(
            f"TRAIN manifest hash mismatch against export lineage: "
            f"provided {clean_train_sha} != lineage {lineage_train_sha}"
        )

    # 2. Load evaluation and canonical TRAIN manifests with exact pinned bytes.
    manifest_data, actual_manifest_sha = load_native_manifest(
        manifest_path,
        clean_manifest_sha,
        expected_roles=list(VALID_EVAL_ROLES),
    )
    patient_records = manifest_data.get("patients")
    if not isinstance(patient_records, list) or not patient_records:
        raise ValueError("Evaluation manifest must contain a nonempty patients list")
    train_data, actual_train_sha = load_native_manifest(train_manifest_path, clean_train_sha)
    if train_data.get("schema") != "self_audit.data.v2" or train_data.get("role") not in TRAIN_ROLES:
        raise ValueError("TRAIN manifest must use self_audit.data.v2 with role training or train")
    if actual_train_sha != lineage_train_sha:
        raise ValueError("verified TRAIN manifest bytes do not match export lineage data_sha256")
    train_labeled, train_unlabeled = _canonical_train_records(train_data)
    train_records = train_labeled + train_unlabeled
    train_lids = [r["id"] for r in train_labeled]
    train_uids = [r["id"] for r in train_unlabeled]
    if sorted(lineage.get("lids", [])) != sorted(train_lids) or sorted(lineage.get("uids", [])) != sorted(train_uids):
        raise ValueError("canonical TRAIN L/U patient IDs do not match export lineage")

    if type(export_meta.get("synthetic")) is not bool:
        raise ValueError("export synthetic flag must be explicitly boolean")
    if type(manifest_data.get("synthetic")) is not bool:
        raise ValueError("evaluation manifest synthetic flag must be explicitly boolean")
    export_synthetic = export_meta["synthetic"]
    manifest_synthetic = manifest_data["synthetic"]
    if export_synthetic != manifest_synthetic:
        raise ValueError(
            f"synthetic parity mismatch: export={export_synthetic} vs manifest={manifest_synthetic}"
        )

    train_dir = Path(train_manifest_path).resolve().parent
    eval_ids = [p["id"] for p in patient_records]
    train_ids = {r["id"] for r in train_records}
    overlap = train_ids & set(eval_ids)
    if overlap:
        raise ValueError(f"Evaluation patient ID overlap with TRAIN: {sorted(overlap)}")

    # Require complete L/U image provenance and check both byte and decoded voxel overlap.
    train_voxel_shas = set()
    train_file_shas = set()
    for tr in train_records:
        pid = tr["id"]
        tr_img = tr.get("image") or tr.get("path")
        tip = Path(tr_img)
        if not tip.is_absolute():
            tip = (train_dir / tip).resolve()
        if not tip.is_file():
            raise FileNotFoundError(f"TRAIN image missing for patient '{pid}': {tip}")
        declared_sha = require_sha256(tr.get("image_sha256") or tr.get("sha256"), f"TRAIN patient '{pid}' image_sha256")
        actual_sha = digest(tip)
        if actual_sha != declared_sha:
            raise ValueError(f"TRAIN image hash mismatch for patient '{pid}'")
        train_file_shas.add(actual_sha)
        train_voxel_shas.add(_voxel_fingerprint(_load_raw_tzyx(tip, pid)))

    # Validate all native image metadata and content overlap before creating output or opening GT.
    manifest_dir = Path(manifest_path).resolve().parent
    mask_frames_by_patient: Dict[str, List[int]] = {}
    datasets_by_patient: Dict[str, str] = {}
    case_metadata: Dict[str, Dict[str, Any]] = {}
    for prec in patient_records:
        pid = prec["id"]
        masks = prec.get("masks")
        if not isinstance(masks, Mapping) or not masks:
            raise ValueError(f"Evaluation patient '{pid}' requires at least one reference frame")
        frames = sorted(int(frame) for frame in masks)
        if len(frames) != len(set(frames)):
            raise ValueError(f"Duplicate evaluation frame index for patient '{pid}'")
        for frame, mask_rec in masks.items():
            if not isinstance(mask_rec, Mapping) or not isinstance(mask_rec.get("path"), str) or not mask_rec.get("path"):
                raise ValueError(f"Evaluation patient '{pid}' frame {frame} requires a mask path")
            require_sha256(mask_rec.get("sha256"), f"patient '{pid}' frame {frame} mask sha256")
        cine, _, cine_meta = load_patient_cine(prec, manifest_dir, normalization=normalization)
        if cine_meta["image_sha256"] in train_file_shas:
            raise ValueError(f"Evaluation image file hash overlap with TRAIN: patient {pid}")
        if cine_meta["raw_voxel_sha256"] in train_voxel_shas:
            raise ValueError(f"Evaluation voxel content overlap with TRAIN: patient {pid}")
        if any(frame < 0 or frame >= cine.shape[0] for frame in frames):
            raise ValueError(f"Evaluation frame index out of bounds for patient '{pid}'")
        mask_frames_by_patient[pid] = frames
        dataset = prec.get("domain") or prec.get("dataset") or manifest_data.get("domain") or manifest_data.get("dataset") or "unspecified"
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError(f"Evaluation patient '{pid}' has invalid dataset/domain metadata")
        datasets_by_patient[pid] = dataset
        case_metadata[pid] = {
            "shape_tzyx": cine_meta["shape_tzyx"],
            "affine": cine_meta["affine"],
            "axes": cine_meta["axes"],
            "image_sha256": cine_meta["image_sha256"],
            "dataset": dataset,
        }
        del cine

    for protected in (manifest_dir, train_dir, Path(export_path).resolve()):
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError(f"Output directory cannot overlap with input artifact/data directory: {protected}")

    out.mkdir(parents=True)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True)

    recipe = {
        "export_manifest_sha256": clean_export_sha,
        "export_payload_sha256": export_meta.get("payload_sha256"),
        "export_synthetic": export_synthetic,
        "evaluation_manifest_sha256": actual_manifest_sha,
        "train_manifest_sha256": actual_train_sha,
        "normalization": normalization,
        "preprocessing_recipe": preprocessing_recipe,
        "device": device,
        "batch_size": batch_size,
        "synthetic": manifest_synthetic,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    actual_recipe_sha = write_json(out / "RECIPE.json", recipe)

    # -------------------------------------------------------------------------
    # PHASE 1: GENERATE AND FREEZE PREDICTIONS
    # NO REFERENCE MASKS OPENED DURING THIS PHASE
    # -------------------------------------------------------------------------
    frozen_inventory = []
    patient_records = manifest_data["patients"]

    for prec in patient_records:
        pid = prec["id"]
        cine, valid_mask, cine_meta = load_patient_cine(prec, manifest_dir, normalization=normalization)
        if (
            case_metadata[pid]["shape_tzyx"] != cine_meta["shape_tzyx"]
            or case_metadata[pid]["axes"] != cine_meta["axes"]
            or case_metadata[pid]["image_sha256"] != cine_meta["image_sha256"]
            or not np.array_equal(np.asarray(case_metadata[pid]["affine"]), np.asarray(cine_meta["affine"]))
        ):
            raise ValueError(f"Evaluation geometry/content changed after preflight for patient '{pid}'")

        pred = infer_cine_stream(
            model=model,
            cine=cine,
            batch_size=batch_size,
            device=device,
            valid_mask=valid_mask,
        )

        pred_filename = f"{pid}_pred.npy"
        pred_path = (pred_dir / pred_filename).resolve()
        if not pred_path.is_relative_to(pred_dir.resolve()):
            raise ValueError(f"Prediction path escapes pred_dir: {pred_path}")

        written = write_prediction_file(pred_path, pred)
        pred_sha = written["sha256"]

        frozen_inventory.append({
            "id": pid,
            "prediction_file": pred_filename,
            "prediction_sha256": pred_sha,
            "shape_tzyx": written["shape_tzyx"],
            "affine": cine_meta["affine"],
            "axes": cine_meta["axes"],
            "dataset": datasets_by_patient[pid],
            "image_sha256": cine_meta["image_sha256"],
            "has_valid_mask": cine_meta["has_valid_mask"],
            "expected_frame_indices": mask_frames_by_patient[pid],
        })

    expected_patient_ids = [p["id"] for p in patient_records]
    frozen_patient_ids = [p["id"] for p in frozen_inventory]
    if len(frozen_patient_ids) != len(set(frozen_patient_ids)) or frozen_patient_ids != expected_patient_ids:
        raise ValueError("Frozen prediction patient coverage is duplicate or incomplete")

    frozen_receipt = {
        "recipe_sha256": actual_recipe_sha,
        "export_manifest_sha256": clean_export_sha,
        "export_payload_sha256": export_meta.get("payload_sha256"),
        "evaluation_manifest_sha256": actual_manifest_sha,
        "synthetic": manifest_synthetic,
        "coverage": {
            "expected_patients": expected_patient_ids,
            "predicted_patients": frozen_patient_ids,
            "complete": True,
        },
        "predictions": frozen_inventory,
    }
    actual_frozen_sha = write_json(out / "PREDICTIONS_FROZEN.json", frozen_receipt)

    # -------------------------------------------------------------------------
    # PHASE 2: SCORE FROZEN PREDICTIONS AGAINST REFERENCE MASKS
    # REFERENCE MASKS ONLY OPENED HERE AFTER INTEGRITY VERIFICATION
    # -------------------------------------------------------------------------
    for item in frozen_inventory:
        p_path = pred_dir / item["prediction_file"]
        if not p_path.is_file():
            raise FileNotFoundError(f"Missing frozen prediction file: {p_path}")
        if digest(p_path) != item["prediction_sha256"]:
            raise ValueError(f"Tampered frozen prediction file: {p_path}")

    frame_rows: List[Dict[str, Any]] = []
    total_confusion = np.zeros((4, 5), dtype=np.int64)
    scored_frames: Dict[str, List[int]] = {pid: [] for pid in expected_patient_ids}

    if len(frozen_inventory) != len(patient_records):
        raise ValueError("Frozen prediction patient count does not match evaluation manifest")
    for prec, item in zip(patient_records, frozen_inventory):
        pid = prec["id"]
        if item["id"] != pid:
            raise ValueError("Frozen prediction order differs from evaluation manifest")
        pred_path = pred_dir / item["prediction_file"]
        pred_arr = np.load(pred_path, allow_pickle=False)
        if list(pred_arr.shape) != item["shape_tzyx"]:
            raise ValueError(f"Frozen prediction shape mismatch for patient '{pid}'")

        _, valid_mask, _ = load_patient_cine(prec, manifest_dir, normalization=normalization)

        masks_dict = prec.get("masks", {})
        for ti_str, mrec in masks_dict.items():
            ti = int(ti_str)
            mask_rel = mrec.get("path")
            mask_path = Path(mask_rel)
            if not mask_path.is_absolute():
                mask_path = (manifest_dir / mask_path).resolve()

            if not mask_path.is_file():
                raise FileNotFoundError(f"Reference mask not found: {mask_path}")

            expected_msha = require_sha256(mrec.get("sha256"), f"patient '{pid}' frame {ti} mask sha256")
            actual_msha = digest(mask_path)
            if actual_msha != expected_msha:
                raise ValueError(
                    f"Reference mask hash mismatch for {pid} frame {ti}: "
                    f"expected {expected_msha}, got {actual_msha}"
                )

            y = np.load(mask_path, allow_pickle=False)
            if not np.issubdtype(y.dtype, np.integer):
                raise ValueError(f"Reference mask must have integer dtype, got {y.dtype}")
            if not np.isin(y, [0, 1, 2, 3, 255]).all():
                raise ValueError(f"Reference mask contains invalid values outside [0, 1, 2, 3, 255]")

            pred_slice = pred_arr[ti]

            # Mandatory mask shape == pred frame shape (no broadcast)
            if y.shape != pred_slice.shape:
                raise ValueError(
                    f"Shape mismatch: reference mask shape {y.shape} != prediction frame shape {pred_slice.shape}"
                )

            if valid_mask is not None:
                vm_slice = valid_mask[ti]
                y_eval = np.where(vm_slice, y, 255).astype(y.dtype)
                pred_eval = np.where(vm_slice, pred_slice, 255).astype(pred_slice.dtype)
            else:
                y_eval = y
                pred_eval = pred_slice

            cm = confusion_matrix(y_eval, pred_eval)
            total_confusion += cm
            c_metrics = class_metrics(cm)
            valid_vox = int(cm.sum())

            frame_rows.append({
                "patient": pid,
                "frame": ti,
                "dataset": datasets_by_patient[pid],
                "raw": c_metrics,
                "confusion": cm.tolist(),
                "valid_voxels": valid_vox,
            })
            scored_frames[pid].append(ti)

    if any(sorted(scored_frames[pid]) != mask_frames_by_patient[pid] for pid in expected_patient_ids):
        raise ValueError("Evaluation reference frame coverage is duplicate or incomplete")
    if set(scored_frames) != set(expected_patient_ids) or any(not frames for frames in scored_frames.values()):
        raise ValueError("Evaluation patient scoring coverage is duplicate or incomplete")

    patient_ids = expected_patient_ids
    global_metrics = compute_historical_metrics(frame_rows, patient_ids)
    dataset_reports: Dict[str, Any] = {}
    for dataset in sorted(set(datasets_by_patient.values())):
        dataset_patient_ids = [pid for pid in patient_ids if datasets_by_patient[pid] == dataset]
        dataset_rows = [row for row in frame_rows if datasets_by_patient[row["patient"]] == dataset]
        dataset_reports[dataset] = compute_historical_metrics(dataset_rows, dataset_patient_ids)
        dataset_reports[dataset]["synthetic"] = manifest_synthetic

    metrics = dict(global_metrics)
    metrics["synthetic"] = manifest_synthetic
    metrics["status"] = "COMPLETED"
    metrics["global"] = dict(global_metrics)
    metrics["datasets"] = dataset_reports
    metrics["coverage"] = {
        "expected_patients": patient_ids,
        "predicted_patients": frozen_patient_ids,
        "scored_patients": sorted(scored_frames),
        "expected_frames": {pid: mask_frames_by_patient[pid] for pid in patient_ids},
        "scored_frames": {pid: sorted(scored_frames[pid]) for pid in patient_ids},
        "complete": True,
    }
    metrics["preprocessing_recipe"] = preprocessing_recipe
    metrics["pooled_confusion"] = total_confusion.tolist()
    metrics["recipe_sha256"] = actual_recipe_sha
    metrics["frozen_predictions_sha256"] = actual_frozen_sha
    metrics["export_manifest_sha256"] = clean_export_sha
    metrics["export_payload_sha256"] = export_meta.get("payload_sha256")
    metrics["evaluation_manifest_sha256"] = actual_manifest_sha
    metrics["evaluation_source_manifest_sha256"] = manifest_data.get("geometry_provenance", {}).get("parent_manifest_sha256", actual_manifest_sha)
    metrics["geometry_provenance"] = manifest_data.get("geometry_provenance")
    metrics["train_manifest_sha256"] = actual_train_sha
    metrics["frames"] = frame_rows

    class_counts = {}
    for j, c_name in enumerate(CLASS_NAMES):
        tp = int(total_confusion[j, j])
        gt = int(total_confusion[j].sum())
        pred_c = int(total_confusion[:, j].sum())
        class_counts[c_name] = {
            "TP": tp,
            "FP": pred_c - tp,
            "FN": gt - tp,
            "reference_count": gt,
            "prediction_count": pred_c,
        }
    metrics["class_counts"] = class_counts

    write_json(out / "METRICS.json", metrics)
    return metrics


# ==============================================================================
# Isolated Benchmark Engine (Subprocess Profiling)
# ==============================================================================

def _prepare_historical_reference_bundle(
    checkpoint_dir: Union[str, Path],
    expected_manifest_sha: str,
    historical_source_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    inference_state: str,
) -> Tuple[Path, str, Dict[str, Any]]:
    """Verify the pinned historical reference and write an inference-only selected-state bundle."""
    from .compact_teacher import (
        HISTORICAL_BASELINE_PINS,
        HISTORICAL_INFERENCE_STATE,
        HISTORICAL_METHOD,
        HISTORICAL_SOURCE_CLOSURE,
        HISTORICAL_SOURCE_COMMIT,
        load_teacher_for_cache,
        verify_historical_source,
    )

    if inference_state != HISTORICAL_INFERENCE_STATE:
        raise ValueError(
            f"historical reference pins inference_state={HISTORICAL_INFERENCE_STATE!r}; got {inference_state!r}"
        )
    base = Path(checkpoint_dir).resolve()
    manifest_path = base / "CHECKPOINT_MANIFEST.json"
    if not manifest_path.is_file():
        manifest_path = base / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"historical reference manifest missing in {base}")
    clean_sha = require_sha256(expected_manifest_sha, "baseline_checkpoint_sha256")
    actual_manifest_sha = digest(manifest_path)
    if actual_manifest_sha != clean_sha or actual_manifest_sha != HISTORICAL_BASELINE_PINS["manifest_sha256"]:
        raise ValueError("historical reference manifest SHA does not match the exact pinned baseline")

    source_receipt = verify_historical_source(historical_source_path)
    if source_receipt.get("source_commit") != HISTORICAL_SOURCE_COMMIT or source_receipt.get("source_closure") != HISTORICAL_SOURCE_CLOSURE:
        raise ValueError("historical source receipt differs from locked reference identity")
    identity = dict(HISTORICAL_BASELINE_PINS)
    identity["teacher_kind"] = "historical_baseline"
    model = load_teacher_for_cache(
        checkpoint_dir=base,
        teacher_identity=identity,
        device="cpu",
        historical_source_path=historical_source_path,
        synthetic=False,
    )
    model.eval()
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    for name, tensor in state_dict.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite historical reference tensor: {name}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest_data.get("config")
    if not isinstance(config, Mapping) or not isinstance(config.get("model"), Mapping):
        raise ValueError("historical reference manifest lacks a model configuration")
    if config.get("method") != HISTORICAL_METHOD:
        raise ValueError("historical reference method differs from locked full EMA method")

    bundle = Path(out_dir).resolve()
    if bundle.exists():
        raise ValueError(f"historical reference bundle destination already exists: {bundle}")
    bundle.mkdir(parents=True)
    weights_path = bundle / "weights.pt"
    torch.save(state_dict, weights_path)
    del state_dict
    del model
    gc.collect()
    weights_sha = digest(weights_path)
    deployed_manifest = {
        "schema": "self_audit.compact.historical_reference.v1",
        "model_class": "NativeStudent",
        "model_config": copy.deepcopy(dict(config["model"])),
        "window": copy.deepcopy(config.get("window")),
        "synthetic": False,
        "inference_state": HISTORICAL_INFERENCE_STATE,
        "training_method": HISTORICAL_METHOD,
        "weights": "weights.pt",
        "weights_sha256": weights_sha,
        "source_manifest_sha256": actual_manifest_sha,
        "source_payload_sha256": HISTORICAL_BASELINE_PINS["payload_sha256"],
        "source_sha256": HISTORICAL_SOURCE_CLOSURE,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "source_verified": True,
    }
    deploy_sha = write_json(bundle / "MANIFEST.json", deployed_manifest)
    identity_receipt = {
        "status": "VERIFIED",
        "source_manifest_sha256": actual_manifest_sha,
        "source_payload_sha256": HISTORICAL_BASELINE_PINS["payload_sha256"],
        "source_sha256": HISTORICAL_SOURCE_CLOSURE,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "inference_state": HISTORICAL_INFERENCE_STATE,
        "training_method": HISTORICAL_METHOD,
        "deploy_manifest_sha256": deploy_sha,
        "deploy_payload_sha256": weights_sha,
        "verified": True,
    }
    return bundle, deploy_sha, identity_receipt

def run_isolated_benchmark(
    compact_export_path: Union[str, Path],
    compact_export_sha: str,
    manifest_path: Union[str, Path],
    manifest_sha: str,
    out_path: Union[str, Path],
    *,
    baseline_checkpoint_path: Optional[Union[str, Path]] = None,
    baseline_checkpoint_sha: Optional[str] = None,
    baseline_state: str = "student",
    historical_source_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    batch_size: int = 16,
    warmup: int = 2,
    repeats: int = 5,
    normalization: str = "prepared",
) -> Dict[str, Any]:
    """Execute fair isolated-process benchmarking comparing compact student and baseline.

    Conditions are strictly matched:
      - Streams ALL manifest cases (no preloading all contexts).
      - Actual E2E includes load + preprocess + context extraction + H2D + forward + argmax + D2H + live disk write.
      - Separate forward-only timing with CUDA synchronization.
      - Cold load timed separately.
      - Active host RAM sampled continuously during inference with polling (including child processes).
      - Baseline reference prepares deployed weights-only BEFORE timing process to prevent optimizer memory contamination.
      - Reports tensor param bytes and deployed file bytes separately.
      - Missing or unverified baseline assets strictly recorded as NOT_RUN.
    """
    clean_compact_sha = require_sha256(compact_export_sha, "compact_export_sha")
    clean_manifest_sha = require_sha256(manifest_sha, "manifest_sha")

    out_file = Path(out_path).resolve()
    if out_file.exists():
        raise ValueError(f"Benchmark output path already exists: {out_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    scratch_dir = out_file.parent / f"_bench_scratch_{int(time.time())}"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Profile compact exported student in isolated subprocess
        compact_sub_result = _profile_in_subprocess(
            model_type="compact_export",
            model_path=str(Path(compact_export_path).resolve()),
            expected_sha=clean_compact_sha,
            manifest_path=str(Path(manifest_path).resolve()),
            manifest_sha=clean_manifest_sha,
            scratch_dir=str(scratch_dir / "compact"),
            device=device,
            batch_size=batch_size,
            warmup=warmup,
            repeats=repeats,
            normalization=normalization,
        )

        # 2. Verify and prepare a standalone historical weights-only bundle before timing.
        baseline_sub_result: Dict[str, Any]
        reference_identity: Dict[str, Any] = {"status": "NOT_RUN", "verified": False}
        if baseline_checkpoint_path is not None and baseline_checkpoint_sha is not None and historical_source_path is not None:
            base_p = Path(baseline_checkpoint_path).resolve()
            clean_base_sha = require_sha256(baseline_checkpoint_sha, "baseline_checkpoint_sha")
            try:
                reference_bundle, deploy_sha, reference_identity = _prepare_historical_reference_bundle(
                    base_p,
                    clean_base_sha,
                    historical_source_path,
                    scratch_dir / "historical_reference",
                    inference_state=baseline_state,
                )
                baseline_sub_result = _profile_in_subprocess(
                    model_type="historical_reference",
                    model_path=str(reference_bundle),
                    expected_sha=deploy_sha,
                    manifest_path=str(Path(manifest_path).resolve()),
                    manifest_sha=clean_manifest_sha,
                    scratch_dir=str(scratch_dir / "baseline"),
                    device=device,
                    batch_size=batch_size,
                    warmup=warmup,
                    repeats=repeats,
                    normalization=normalization,
                    historical_source_path=str(Path(historical_source_path).resolve()),
                )
                baseline_sub_result["source_manifest_sha256"] = reference_identity["source_manifest_sha256"]
                baseline_sub_result["source_payload_sha256"] = reference_identity["source_payload_sha256"]
                baseline_sub_result["inference_state"] = reference_identity["inference_state"]
            except Exception as e:
                baseline_sub_result = {
                    "status": "NOT_RUN",
                    "reason": f"Historical baseline verification or profiling failed: {e}",
                    "checkpoint_path": str(base_p),
                }
                reference_identity = {"status": "NOT_RUN", "verified": False, "reason": str(e)}
        elif baseline_checkpoint_path is not None or baseline_checkpoint_sha is not None:
            reason = "Historical baseline requires checkpoint path, manifest SHA-256, and explicit verified --historical-source"
            baseline_sub_result = {
                "status": "NOT_RUN",
                "reason": reason,
                "checkpoint_path": str(baseline_checkpoint_path) if baseline_checkpoint_path is not None else None,
            }
            reference_identity = {"status": "NOT_RUN", "verified": False, "reason": reason}
        else:
            baseline_sub_result = {
                "status": "NOT_RUN",
                "reason": "No validated baseline checkpoint provided; missing real assets stays NOT_RUN",
            }

        if compact_sub_result.get("status") == "COMPLETED" and baseline_sub_result.get("status") == "COMPLETED":
            compact_sig = compact_sub_result.get("workload", {}).get("case_hashes")
            baseline_sig = baseline_sub_result.get("workload", {}).get("case_hashes")
            if compact_sig != baseline_sig:
                baseline_sub_result = {
                    "status": "NOT_RUN",
                    "reason": "Compact and historical reference case identity/shape receipts differ",
                }
                reference_identity = {
                    **reference_identity,
                    "status": "NOT_RUN",
                    "verified": False,
                    "reason": "inference workload did not match",
                }

        comparison = {}
        if compact_sub_result.get("status") == "COMPLETED" and baseline_sub_result.get("status") == "COMPLETED":
            base_steady = baseline_sub_result["latency"]["steady"]["model_only_median_seconds_per_volume"]
            comp_steady = compact_sub_result["latency"]["steady"]["model_only_median_seconds_per_volume"]
            base_ram = baseline_sub_result["memory"]["peak_inference_process_tree_ram_mb"]
            comp_ram = compact_sub_result["memory"]["peak_inference_process_tree_ram_mb"]
            base_bytes = baseline_sub_result["model_size"]["deployed_file_bytes"]
            comp_bytes = compact_sub_result["model_size"]["deployed_file_bytes"]

            comparison = {
                "model_only_speedup": float(base_steady / comp_steady) if comp_steady > 0 else None,
                "peak_inference_ram_reduction_factor": float(base_ram / comp_ram) if comp_ram > 0 else None,
                "deployed_bytes_reduction_factor": float(base_bytes / comp_bytes) if comp_bytes > 0 else None,
            }

        full_report = {
            "schema": "self_audit.compact.benchmark.v1",
            "benchmark_conditions": {
                "device": device,
                "batch_size": batch_size,
                "warmup_runs": warmup,
                "steady_repeats": repeats,
                "normalization": normalization,
                "precision": "float32",
                "volume_definition": "full_native_4d_cine_tzyx",
                "io_cache_policy": "streaming_native_no_full_prob_cache; live_verified_npy_write_per_volume; OS_cache_not_flushed",
                "io_policy": "streaming native; no full probability cache; verified uint8 NPY writes; OS filesystem cache retained",
                "evaluation_manifest_sha256": clean_manifest_sha,
                "synthetic": compact_sub_result.get("synthetic"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "environment": {
                    "os": platform.system(),
                    "python_version": sys.version.split()[0],
                    "torch_version": torch.__version__,
                    "cpu_count": os.cpu_count(),
                },
            },
            "compact_student": compact_sub_result,
            "baseline_student": baseline_sub_result,
            "reference_identity": reference_identity,
            "comparison": comparison,
        }

        write_json(out_file, full_report)
        return full_report
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def _profile_in_subprocess(
    model_type: str,
    model_path: str,
    expected_sha: str,
    manifest_path: str,
    manifest_sha: str,
    scratch_dir: str,
    device: str,
    batch_size: int,
    warmup: int,
    repeats: int,
    normalization: str,
    historical_source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Launch isolated worker script in subprocess to measure process RAM cleanly."""
    cmd = [
        sys.executable,
        "-m",
        "self_audit.compact_inference",
        "--subprocess-worker",
        "--model-type", model_type,
        "--model-path", model_path,
        "--expected-sha", expected_sha,
        "--manifest-path", manifest_path,
        "--manifest-sha", manifest_sha,
        "--scratch-dir", scratch_dir,
        "--device", device,
        "--batch-size", str(batch_size),
        "--warmup", str(warmup),
        "--repeats", str(repeats),
        "--normalization", normalization,
    ]
    if historical_source_path:
        cmd.extend(["--historical-source", historical_source_path])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Subprocess profiling failed (exit {proc.returncode}):\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse subprocess JSON output: {e}\nSTDOUT:\n{proc.stdout}")


def _worker_benchmark_target(args: argparse.Namespace) -> None:
    """Internal entrypoint executed inside isolated benchmark subprocess.

    Streams ALL manifest cases without preloading all contexts.
    Performs live disk writes during E2E.
    Active background thread polls host peak RAM during inference.
    """
    clean_expected_sha = require_sha256(args.expected_sha, "expected_sha")
    clean_manifest_sha = require_sha256(args.manifest_sha, "manifest_sha")
    device = args.device

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")

    scratch_p = Path(args.scratch_dir).resolve()
    scratch_p.mkdir(parents=True, exist_ok=True)

    t0_load = time.perf_counter()
    artifact_manifest_sha = clean_expected_sha
    artifact_payload_sha: Optional[str] = None
    reference_meta: Optional[Dict[str, Any]] = None
    synthetic: Optional[bool] = None
    if args.model_type == "compact_export":
        model, meta = load_compact_export(args.model_path, clean_expected_sha, device=device)
        validate_export_preprocessing(meta, args.normalization)
        w_file = Path(args.model_path) / meta.get("weights", "weights.pt")
        deployed_file_bytes = w_file.stat().st_size
        artifact_payload_sha = meta.get("payload_sha256")
        synthetic = meta.get("synthetic")
    elif args.model_type == "historical_reference":
        from .compact_teacher import (
            HISTORICAL_BASELINE_PINS,
            HISTORICAL_INFERENCE_STATE,
            HISTORICAL_METHOD,
            HISTORICAL_SOURCE_CLOSURE,
            HISTORICAL_SOURCE_COMMIT,
            load_historical_teacher_model,
            verify_historical_source,
        )

        bundle = Path(args.model_path).resolve()
        manifest_file = bundle / "MANIFEST.json"
        if not manifest_file.is_file() or digest(manifest_file) != clean_expected_sha:
            raise ValueError("standalone historical reference manifest pin mismatch")
        ref = json.loads(manifest_file.read_text(encoding="utf-8"))
        if ref.get("schema") != "self_audit.compact.historical_reference.v1":
            raise ValueError("unsupported standalone historical reference schema")
        if ref.get("source_verified") is not True or ref.get("source_manifest_sha256") != HISTORICAL_BASELINE_PINS["manifest_sha256"]:
            raise ValueError("standalone historical reference is not bound to the pinned source manifest")
        if ref.get("source_payload_sha256") != HISTORICAL_BASELINE_PINS["payload_sha256"]:
            raise ValueError("standalone historical reference is not bound to the pinned source payload")
        if ref.get("source_sha256") != HISTORICAL_SOURCE_CLOSURE or ref.get("source_commit") != HISTORICAL_SOURCE_COMMIT:
            raise ValueError("standalone historical reference source closure mismatch")
        if ref.get("inference_state") != HISTORICAL_INFERENCE_STATE or ref.get("training_method") != HISTORICAL_METHOD:
            raise ValueError("standalone historical reference state/method mismatch")
        if ref.get("synthetic") is not False:
            raise ValueError("historical baseline reference must be real, non-synthetic")
        source_receipt = verify_historical_source(args.historical_source)
        if source_receipt.get("source_closure") != HISTORICAL_SOURCE_CLOSURE:
            raise ValueError("historical source closure changed after parent verification")
        model = load_historical_teacher_model(
            args.historical_source,
            ref.get("model_config"),
            synthetic=False,
            window=ref.get("window"),
        )
        w_file = bundle / ref.get("weights", "weights.pt")
        if not w_file.is_file() or digest(w_file) != ref.get("weights_sha256"):
            raise ValueError("standalone historical weights-only file hash mismatch")
        state = torch.load(w_file, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError("standalone historical weights file must contain a state mapping")
        model.load_state_dict(state, strict=True)
        del state
        artifact_payload_sha = ref["weights_sha256"]
        deployed_file_bytes = w_file.stat().st_size
        synthetic = False
        reference_meta = ref
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    load_time = time.perf_counter() - t0_load
    tensor_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    manifest_data, actual_m_sha = load_native_manifest(
        args.manifest_path,
        clean_manifest_sha,
        expected_roles=list(VALID_EVAL_ROLES),
    )
    manifest_dir = Path(args.manifest_path).resolve().parent
    patient_records = manifest_data.get("patients", [])
    if not patient_records:
        raise ValueError("Manifest contains no patients")
    if type(manifest_data.get("synthetic")) is not bool:
        raise ValueError("benchmark manifest synthetic flag must be explicitly boolean")
    if synthetic is not None and synthetic != manifest_data["synthetic"]:
        raise ValueError("benchmark model and evaluation synthetic flags differ")
    synthetic = manifest_data["synthetic"]

    def _sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    # Poll process-tree RAM throughout cold, warmup, and measured inference.
    stop_sampler = threading.Event()
    peak_sampled_bytes = [0]
    proc = psutil.Process()

    def _sample_ram_once() -> None:
        total_rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if total_rss > peak_sampled_bytes[0]:
            peak_sampled_bytes[0] = total_rss

    def _sample_ram() -> None:
        while not stop_sampler.is_set():
            try:
                _sample_ram_once()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.005)

    sampler_thread = threading.Thread(target=_sample_ram, daemon=True)
    sampler_thread.start()
    _sample_ram_once()

    def _run_volume(prec: Dict[str, Any], case_idx: int) -> Tuple[float, float, int, Dict[str, Any]]:
        """Run a complete native cine with verified NPY output in the E2E interval."""
        _sample_ram_once()
        _sync()
        t_e2e_start = time.perf_counter()
        cine, valid_mask, cine_meta = load_patient_cine(prec, manifest_dir, normalization=args.normalization)
        t, z, h, w = cine.shape
        if prec.get("shape") is not None and list(prec["shape"]) != [t, z, h, w]:
            raise ValueError(f"Declared shape differs from actual cine for patient '{prec['id']}'")
        num_slices = t * z
        indices = [(ti, zi) for ti in range(t) for zi in range(z)]
        if len(indices) != num_slices:
            raise RuntimeError(f"Incomplete native slice coverage for patient '{prec['id']}'")
        pred = np.zeros((t, z, h, w), dtype=np.uint8)
        forward_dur = 0.0

        for offset in range(0, len(indices), args.batch_size):
            chunk = indices[offset: offset + args.batch_size]
            contexts = [extract_slice_context(cine, ti, zi) for ti, zi in chunk]
            batch_x = torch.cat(contexts, dim=0).to(device)
            batch_valid: Optional[Tensor] = None
            if valid_mask is not None:
                v_slices = [torch.from_numpy(valid_mask[ti, zi]).bool() for ti, zi in chunk]
                batch_valid = torch.stack(v_slices, dim=0).to(device)
            _sync()
            t_fwd_start = time.perf_counter()
            with torch.inference_mode():
                logits = model(batch_x, valid_mask=batch_valid)
            _sync()
            forward_dur += time.perf_counter() - t_fwd_start
            if logits.shape != (len(chunk), 4, h, w) or not torch.isfinite(logits).all():
                raise ValueError(f"Invalid logits shape or value for patient '{prec['id']}'")
            chunk_preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            for (ti, zi), p_slice in zip(chunk, chunk_preds):
                if valid_mask is not None:
                    p_slice = np.where(valid_mask[ti, zi], p_slice, 255).astype(np.uint8)
                pred[ti, zi] = p_slice

        pred_out_path = scratch_p / f"{prec['id']}_case_{case_idx}.npy"
        written = write_prediction_file(pred_out_path, pred)
        if written["shape_tzyx"] != [t, z, h, w] or not pred_out_path.is_file():
            raise RuntimeError(f"Live prediction output write/shape verification failed: {pred_out_path}")
        _sync()
        e2e_dur = time.perf_counter() - t_e2e_start
        case = {
            "id": prec["id"],
            "dataset": prec.get("domain") or prec.get("dataset") or manifest_data.get("domain") or manifest_data.get("dataset") or "unspecified",
            "image_sha256": cine_meta["image_sha256"],
            "shape_tzyx": cine_meta["shape_tzyx"],
            "axes": cine_meta["axes"],
            "affine": cine_meta["affine"],
            "slice_count": num_slices,
            "prediction_file": written["file"],
            "prediction_sha256": written["sha256"],
        }
        _sample_ram_once()
        return e2e_dur, forward_dur, num_slices, case

    try:
        cold_e2e_times: List[float] = []
        cold_fwd_times: List[float] = []
        cold_case_receipts: List[Dict[str, Any]] = []
        total_manifest_slices = 0
        for idx, prec in enumerate(patient_records):
            e2e_d, fwd_d, n_sl, case = _run_volume(prec, case_idx=idx)
            cold_e2e_times.append(e2e_d)
            cold_fwd_times.append(fwd_d)
            cold_case_receipts.append(case)
            total_manifest_slices += n_sl
        expected_ids = [p["id"] for p in patient_records]
        actual_ids = [p["id"] for p in cold_case_receipts]
        if len(actual_ids) != len(set(actual_ids)) or actual_ids != expected_ids:
            raise ValueError("Benchmark inference case coverage is duplicate or incomplete")

        for _ in range(args.warmup):
            for idx, prec in enumerate(patient_records):
                _run_volume(prec, case_idx=idx)

        steady_volume_e2e_times: List[float] = []
        steady_volume_fwd_times: List[float] = []
        steady_started = time.perf_counter()
        for _ in range(args.repeats):
            for idx, prec in enumerate(patient_records):
                e2e_d, fwd_d, _, _ = _run_volume(prec, case_idx=idx)
                steady_volume_e2e_times.append(e2e_d)
                steady_volume_fwd_times.append(fwd_d)
        _sync()
        total_steady_wall_seconds = time.perf_counter() - steady_started
    finally:
        _sample_ram_once()
        stop_sampler.set()
        sampler_thread.join(timeout=1.0)

    peak_sampled_mb = peak_sampled_bytes[0] / (1024 * 1024)
    if peak_sampled_mb <= 0:
        raise RuntimeError("host RAM sampler did not record a positive process-tree peak")
    cuda_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.startswith("cuda") else None
    cuda_res_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024) if device.startswith("cuda") else None
    median_fwd = float(np.median(steady_volume_fwd_times))
    p95_fwd = float(np.percentile(steady_volume_fwd_times, 95))
    median_e2e = float(np.median(steady_volume_e2e_times))
    p95_e2e = float(np.percentile(steady_volume_e2e_times, 95))
    total_steady_volumes = len(steady_volume_e2e_times)
    total_steady_slices = total_manifest_slices * args.repeats
    vol_per_sec = float(total_steady_volumes / total_steady_wall_seconds) if total_steady_wall_seconds > 0 else 0.0
    slices_per_sec = float(total_steady_slices / total_steady_wall_seconds) if total_steady_wall_seconds > 0 else 0.0

    case_hashes = [
        {key: case[key] for key in ("id", "dataset", "image_sha256", "shape_tzyx", "axes", "affine")}
        for case in cold_case_receipts
    ]
    workload = {
        "total_cases": len(patient_records),
        "total_slices_per_cohort": total_manifest_slices,
        "batch_size": args.batch_size,
        "case_hashes": case_hashes,
        "case_shapes": [{"id": case["id"], "shape_tzyx": case["shape_tzyx"]} for case in cold_case_receipts],
        "patient_coverage_complete": actual_ids == expected_ids,
    }
    result = {
        "status": "COMPLETED",
        "model_type": args.model_type,
        "model_path": args.model_path,
        "artifact_manifest_sha256": artifact_manifest_sha,
        "artifact_payload_sha256": artifact_payload_sha,
        "evaluation_manifest_sha256": actual_m_sha,
        "synthetic": synthetic,
        "device": device,
        "precision": "float32",
        "volume_definition": "full_native_4d_cine_tzyx; one complete patient cine per volume",
        "io_policy": "streaming native; no full probability cache; verified uint8 NPY writes; OS filesystem cache retained",
        "io_cache_policy": "streaming native; no full probability cache; verified NPY volume writes; OS cache not flushed",
        "reference": reference_meta,
        "workload": workload,
        "inference_outputs": cold_case_receipts,
        "model_size": {
            "total_parameters": param_count,
            "trainable_parameters": trainable_params,
            "tensor_param_bytes": tensor_param_bytes,
            "deployed_file_bytes": deployed_file_bytes,
            "deployed_weights_bytes": deployed_file_bytes,
            "load_time_seconds": load_time,
        },
        "latency": {
            "cold": {
                "load_seconds": load_time,
                "model_only_median_seconds_per_volume": float(np.median(cold_fwd_times)),
                "e2e_median_seconds_per_volume": float(np.median(cold_e2e_times)),
            },
            "steady": {
                "repeats": args.repeats,
                "model_only_median_seconds_per_volume": median_fwd,
                "model_only_p95_seconds_per_volume": p95_fwd,
                "model_only_min_seconds_per_volume": float(np.min(steady_volume_fwd_times)),
                "model_only_max_seconds_per_volume": float(np.max(steady_volume_fwd_times)),
                "e2e_median_seconds_per_volume": median_e2e,
                "e2e_p95_seconds_per_volume": p95_e2e,
                "total_wall_seconds": total_steady_wall_seconds,
                "total_volumes": total_steady_volumes,
                "total_slices": total_steady_slices,
                "volumes_per_second": vol_per_sec,
                "slices_per_second": slices_per_sec,
            },
        },
        "memory": {
            "peak_inference_process_tree_ram_mb": peak_sampled_mb,
            "ram_poll_interval_ms": 5,
            "cuda_peak_allocated_mb": cuda_alloc_mb,
            "cuda_peak_reserved_mb": cuda_res_mb,
        },
    }

    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


# ==============================================================================
# CLI Entrypoint for Subprocess Worker
# ==============================================================================

if __name__ == "__main__":
    if "--subprocess-worker" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--subprocess-worker", action="store_true")
        parser.add_argument("--model-type", required=True)
        parser.add_argument("--model-path", required=True)
        parser.add_argument("--expected-sha", required=True)
        parser.add_argument("--manifest-path", required=True)
        parser.add_argument("--manifest-sha", required=True)
        parser.add_argument("--scratch-dir", required=True)
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--warmup", type=int, default=2)
        parser.add_argument("--repeats", type=int, default=5)
        parser.add_argument("--normalization", default="prepared")
        parser.add_argument("--historical-source", default=None)
        args = parser.parse_args()
        _worker_benchmark_target(args)
