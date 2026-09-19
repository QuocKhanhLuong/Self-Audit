"""Image-only source manifest for the original Self-Audit ACDC protocol.

The benchmark builder used to import ``self_audit_maskfree`` and therefore
silently inherited its 70/15/15-plus-official-test policy.  This module is a
separate, small protocol adapter for the checked-in Self-Audit source:

* the ACDC *training* cohort only (100 patients / 200 ED+ES volumes);
* the checked-in patient split (80 train / 20 validation patients);
* ED and ES frame selection from the source-side selection receipt; and
* every acquired Z slice with endpoint-replicated 2.5-D context.

The selection receipt is produced from ``Info.cfg`` before an image-only mount
is created.  This builder never opens Info.cfg, annotations, or a mask path;
inside the bwrap namespace it sees only selected image files and this receipt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np

from .provenance import sha256_file, sha256_json


PROTOCOL_SCHEMA_VERSION = "self_audit.acdc.image_only.v1"
SPLIT_POLICY_VERSION = "self_audit.acdc.patient_split.v1"
SELECTION_SCHEMA_VERSION = "self_audit.acdc.frame_selection.v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _nifti_geometry(path: Path, *, depth_axis: int) -> dict[str, Any]:
    image = nib.load(str(path))
    header = image.header
    affine = np.asarray(image.affine, dtype=np.float64)
    if len(image.shape) != 3:
        raise ValueError(f"Self-Audit selected ACDC image must be rank-3, got {path}: {image.shape}")
    if depth_axis not in (0, 1, 2):
        raise ValueError(f"depth_axis must be 0, 1 or 2, got {depth_axis}")
    if any(int(size) <= 0 for size in image.shape):
        raise ValueError(f"selected image has non-positive shape: {path}: {image.shape}")
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)) or abs(float(np.linalg.det(affine[:3, :3]))) <= 0.0:
        raise ValueError(f"selected image has invalid affine: {path}")
    orientation = [str(code) for code in nib.aff2axcodes(affine)]
    zooms = [float(value) for value in header.get_zooms()[:3]]
    units_raw = header.get_xyzt_units()
    units = [None if value is None else str(value).lower() for value in units_raw]
    spatial_unit = str(units[0]).lower() if units and units[0] else "unknown"
    unit_factor = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "meter": 1000.0}.get(spatial_unit)
    spacing_valid = bool(unit_factor is not None and all(np.isfinite(value) and value > 0.0 for value in zooms))
    return {
        "source_format": "nifti",
        "shape": [int(value) for value in image.shape],
        "dtype": str(np.dtype(header.get_data_dtype())),
        "native_affine": affine.tolist(),
        "orientation": orientation,
        "zooms": zooms,
        "xyzt_units": units,
        "spatial_unit": spatial_unit,
        "spacing_mm": [float(value * unit_factor) for value in zooms]
        if spacing_valid and unit_factor is not None
        else None,
        "spacing_valid": spacing_valid,
        "native_geometry": "available",
        "native_geometry_reason": None,
        "native_grid_export": True,
        "export_grid": "native",
        "depth_axis": int(depth_axis),
    }


def _frame_binding(source_hash: str, frame_index: int, frame_label: str) -> str:
    """Bind the selected frame identity without pretending to be a pixel hash."""
    digest = hashlib.sha256()
    digest.update(f"{source_hash}|frame={int(frame_index)}|label={frame_label}".encode("utf-8"))
    return digest.hexdigest()


def _selection_rows(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("unsupported Self-Audit frame-selection receipt")
    rows = selection.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Self-Audit frame-selection receipt has no records")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("frame-selection rows must be objects")
        required = {"patient_id", "split", "relative_path", "frame_index", "frame_label"}
        missing = required - set(raw)
        if missing:
            raise ValueError(f"frame-selection row missing {sorted(missing)}")
        patient = str(raw["patient_id"])
        split = str(raw["split"])
        if split not in {"train", "dev"}:
            raise ValueError(f"Self-Audit source has no test cohort; invalid split {split!r}")
        relative = str(raw["relative_path"]).replace("\\", "/")
        frame_index = int(raw["frame_index"])
        frame_label = str(raw["frame_label"])
        if frame_label not in {"ED", "ES"} or frame_index <= 0:
            raise ValueError("Self-Audit frame receipt must contain positive ED/ES frame indices")
        if not relative.startswith(f"training/{patient}/"):
            raise ValueError(f"selected image is outside the declared patient cohort: {relative}")
        if not Path(relative).name.startswith(f"{patient}_frame"):
            raise ValueError(f"selected image filename does not bind patient identity: {relative}")
        key = (patient, relative)
        if key in seen:
            raise ValueError(f"duplicate selected frame: {key}")
        seen.add(key)
        result.append(
            {
                "patient_id": patient,
                "split": split,
                "relative_path": relative,
                "frame_index": frame_index,
                "frame_label": frame_label,
            }
        )
    return sorted(result, key=lambda row: (row["split"], row["patient_id"], row["relative_path"]))


def discover_self_audit_acdc(
    image_root: str | Path,
    selection_path: str | Path,
    *,
    seed: int = 42,
    depth_axis: int = 2,
) -> dict[str, Any]:
    """Build a source manifest from selected images under an image-only root."""
    root = Path(image_root).resolve()
    selection_file = Path(selection_path)
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if not isinstance(selection, Mapping):
        raise ValueError("Self-Audit selection receipt must be a JSON object")
    rows = _selection_rows(selection)
    if int(selection.get("seed", -1)) != int(seed):
        raise ValueError("selection receipt seed does not match the frozen seed 42")
    records: list[dict[str, Any]] = []
    patients: set[str] = set()
    split_patients: dict[str, set[str]] = {"train": set(), "dev": set()}
    for row in rows:
        path = (root / row["relative_path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"selected image escapes image-only root: {row['relative_path']}") from exc
        if not path.is_file() or path.name.lower().endswith(("_gt.nii", "_gt.nii.gz")):
            raise ValueError(f"selected image is missing or annotation-shaped: {path}")
        source_hash = sha256_file(path)
        geometry = _nifti_geometry(path, depth_axis=depth_axis)
        shape = [int(value) for value in geometry["shape"]]
        depth = int(shape[depth_axis])
        native_hw = [int(shape[index]) for index in range(3) if index != depth_axis]
        patient = str(row["patient_id"])
        split = str(row["split"])
        patients.add(patient)
        split_patients[split].add(patient)
        volume_id = f"acdc:self_audit:{patient}:frame-{int(row['frame_index']):04d}"
        frame_binding = _frame_binding(source_hash, int(row["frame_index"]), str(row["frame_label"]))
        for slice_index in range(depth):
            records.append(
                {
                    "volume_id": volume_id,
                    "unit_id": f"{volume_id}:z{slice_index:04d}",
                    "study_id": f"acdc:{patient}",
                    "patient_id": patient,
                    "dataset": "acdc",
                    "split": split,
                    "path": str(path),
                    "source_path": str(path),
                    "relative_path": row["relative_path"],
                    "source_format": "nifti",
                    "shape": shape,
                    # The shared projector keeps both the stored NIfTI shape
                    # and a nested geometry object.  Keep these fields
                    # explicit so source records can be validated without
                    # consulting any annotation-side metadata.
                    "stored_shape": shape,
                    "native_shape": shape,
                    "dtype": geometry["dtype"],
                    "native_hw": native_hw,
                    "depth": depth,
                    "num_slices": depth,
                    "depth_axis": int(depth_axis),
                    "frame_axis": None,
                    "slice_index": int(slice_index),
                    "frame_index": int(row["frame_index"]),
                    "frame_time": None,
                    "frame_selection_rule": "Self-Audit preprocess_acdc.py Info.cfg ED/ES",
                    "native_geometry": geometry["native_geometry"],
                    "native_geometry_reason": geometry["native_geometry_reason"],
                    "native_affine": geometry["native_affine"],
                    "orientation": geometry["orientation"],
                    "zooms": geometry["zooms"],
                    "xyzt_units": geometry["xyzt_units"],
                    "spatial_unit": geometry["spatial_unit"],
                    "spacing_mm": geometry["spacing_mm"],
                    "spacing_reason": None if geometry["spacing_valid"] else "source header lacks recognised spatial units",
                    "spacing_valid": bool(geometry["spacing_valid"]),
                    "study_grid_compatibility": {
                        "status": "single_source_geometry",
                        "no_resampling": True,
                    },
                    "geometry": geometry,
                    "export_grid": geometry["export_grid"],
                    "native_grid_export": bool(geometry["native_grid_export"]),
                    "frames": {
                        "is_multi_frame": False,
                        "frame_axis": None,
                        "n_frames": 1,
                        "frame_indices": [int(row["frame_index"])],
                    },
                    "cine_eligible": False,
                    "cine_reason": "Self-Audit protocol selects ED/ES rank-3 frames",
                    "acquisition_id": volume_id,
                    "source_fingerprint": source_hash,
                    "source_hash": source_hash,
                    "frame_fingerprint": frame_binding,
                    "duplicate_of": None,
                    "cohort_provenance": {
                        "source_cohort": "ACDC/training",
                        "official_folder": "training",
                        "split_manifest": "splits/acdc_patient_split_seed42.json",
                        "split_rule": "checked_in_patient_manifest_seed42",
                        "split_identity": "patient_id",
                        "frame_selection_rule": "Info.cfg ED/ES",
                        "mask_inputs_used": False,
                    },
                }
            )
    if split_patients["train"] & split_patients["dev"]:
        raise ValueError("Self-Audit selection receipt has patient leakage")
    source_manifest = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "dataset": "acdc",
        "seed": int(seed),
        "depth_axis": int(depth_axis),
        "split_provenance": {
            "policy_version": SPLIT_POLICY_VERSION,
            "rule": "checked_in_splits/acdc_patient_split_seed42.json",
            "seed": int(seed),
            "cohort": "ACDC/training only; no independent test split",
            "train_patients": len(split_patients["train"]),
            "dev_patients": len(split_patients["dev"]),
            "test_patients": 0,
            "patient_level_disjoint": True,
            "split_identity": "patient_id",
            "selection_inputs": ["checked_in_split_manifest", "Info.cfg ED/ES receipt"],
            "frame_selection": "ED and ES only",
            "content_fingerprint_used": False,
        },
        "discovery_contract": {
            "version": "self_audit.acdc.image_only.discovery.v1",
            "source_hash": "sha256_selected_image_file_bytes",
            "frame_binding": "sha256(source_hash,frame_index,frame_label)",
            "mask_files_read": 0,
            "annotation_sidecars_read": 0,
            "no_resampling": True,
        },
        "generation_membership": "selected_ED_ES_frames_x_all_Z_slices",
        "selection_receipt_sha256": sha256_file(selection_file),
        "records": sorted(records, key=lambda row: (row["patient_id"], row["volume_id"], row["slice_index"])),
    }
    source_manifest["manifest_id"] = sha256_json(
        {
            "schema_version": source_manifest["schema_version"],
            "seed": source_manifest["seed"],
            "split_provenance": source_manifest["split_provenance"],
            "selection_receipt_sha256": source_manifest["selection_receipt_sha256"],
            "records": [
                {
                    "patient_id": row["patient_id"],
                    "split": row["split"],
                    "relative_path": row["relative_path"],
                    "frame_index": row["frame_index"],
                    "slice_index": row["slice_index"],
                    "source_hash": row["source_hash"],
                }
                for row in source_manifest["records"]
            ],
        }
    )[:32]
    return source_manifest
