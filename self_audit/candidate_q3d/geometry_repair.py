"""Header-only, fail-closed repair of cached Q3D manifest geometry.

The repair reads only NIfTI headers and file bytes needed for SHA-256. It never
accesses ``dataobj``, voxel arrays, or segmentation masks. Source manifest
records and all fields other than the top-level ``geometry`` map are preserved
byte-for-byte at the JSON value level.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..data import (
    CLASS_MAPPING,
    GEOMETRY_MAP_SCHEMA,
    GEOMETRY_SCHEMA,
    validate_geometry as _validate_cache_geometry,
)
from ..preparation import (
    SPATIAL_UNIT_SCALE_MM,
    affine_in_mm,
    native_geometry_zyx,
    spatial_unit,
)


REPAIR_SCHEMA = "self_audit.q3d.geometry_repair.v1"
RECEIPT_SCHEMA = "self_audit.q3d.geometry_repair_receipt.v1"
_AXIS_MAP = "XYZ->ZYX"
_CSV_FIELDS = (
    "patient_id",
    "cine_path",
    "source_file_sha256",
    "source_header_sha256",
    "manifest_shape_zyx",
    "manifest_spacing_zyx_mm",
    "manifest_affine_zyx",
    "source_shape_xyzt",
    "header_zooms_xyz_native",
    "header_zooms_xyz_mm",
    "sform_code",
    "sform_affine_xyz",
    "qform_code",
    "qform_affine_xyz",
    "selected_affine_xyz",
    "original_spatial_unit",
    "resolved_spatial_unit",
    "unit_resolution",
    "unit_provenance",
    "cine_mapping_source",
    "cine_mapping_sha256",
    "cine_mapping_provenance",
    "voxel_bridge_status",
    "repaired_spacing_zyx_mm",
    "repaired_affine_zyx",
    "errors",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(_json_safe(value), indent=2, sort_keys=False, allow_nan=False) + "\n").encode("utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(_json_safe(value), separators=(",", ":"), allow_nan=False)


def _header_forms(header: Any) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    forms: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name in ("sform", "qform"):
        coded_getter = getattr(header, f"get_{name}", None)
        if not callable(coded_getter):
            errors.append(f"NIfTI header does not expose {name}")
            forms[name] = {"code": None, "affine": None}
            continue
        try:
            coded_matrix, raw_code = coded_getter(coded=True)
            raw_matrix = coded_getter(coded=False)
            code = int(raw_code)
            matrix = None if raw_matrix is None else np.asarray(raw_matrix, dtype=np.float64)
            if matrix is not None and matrix.shape != (4, 4):
                errors.append(f"NIfTI {name} affine must be 4x4")
                matrix = None
            if code > 0 and coded_matrix is None:
                errors.append(f"NIfTI {name} code is set but affine is missing")
            if code > 0 and coded_matrix is not None:
                coded_array = np.asarray(coded_matrix, dtype=np.float64)
                if coded_array.shape != (4, 4):
                    errors.append(f"NIfTI {name} affine must be 4x4")
                elif matrix is None:
                    matrix = coded_array
        except Exception as exc:
            errors.append(f"cannot read NIfTI {name}: {exc}")
            code, matrix = None, None
        forms[name] = {"code": code, "affine": matrix}

    sform = forms["sform"]
    qform = forms["qform"]
    selected: np.ndarray | None = None
    if sform["code"] and sform["code"] > 0 and sform["affine"] is not None:
        selected = sform["affine"]
        selected_name = "sform"
    elif qform["code"] and qform["code"] > 0 and qform["affine"] is not None:
        selected = qform["affine"]
        selected_name = "qform"
    else:
        selected_name = None
        errors.append("NIfTI has no coded authoritative sform or qform geometry")
    if (
        sform["code"] and sform["code"] > 0 and sform["affine"] is not None
        and qform["code"] and qform["code"] > 0 and qform["affine"] is not None
        and not np.allclose(
        sform["affine"], qform["affine"], rtol=1e-5, atol=1e-5
        )
    ):
        errors.append("coded sform and qform disagree")
    selected_info = None if selected is None else {
        "name": selected_name, "affine": np.asarray(selected, dtype=np.float64)
    }
    return forms, selected_info, errors


def _resolve_units(
    patient_id: str,
    original_unit: Any,
    declarations: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, str | None]:
    raw = str(original_unit or "").strip().lower()
    if raw in SPATIAL_UNIT_SCALE_MM:
        if patient_id in declarations:
            raise ValueError(
                f"unit declaration for {patient_id} is only allowed when header units are unknown"
            )
        return spatial_unit(None, raw), "nifti_header", None

    declaration = declarations.get(patient_id)
    if not isinstance(declaration, Mapping):
        raise ValueError(
            f"unknown spatial unit {original_unit!r}; explicit unit declaration with provenance required"
        )
    unit_value = declaration.get("unit")
    provenance = declaration.get("provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(f"unit declaration for {patient_id} requires non-empty provenance")
    unit = spatial_unit(None, str(unit_value or ""))
    return unit, "explicit_declaration", provenance.strip()


def _records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ("labeled", "unlabeled", "patients"):
        value = manifest.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ValueError(f"manifest {key} must be a list")
        for record in value:
            if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
                raise ValueError(f"manifest {key} contains a record without a non-empty string id")
            records.append(record)
    if not records:
        raise ValueError("manifest has no patient records in labeled, unlabeled, or patients")
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("manifest patient record IDs must be unique")
    return records


def _manifest_geometry(manifest: Mapping[str, Any]) -> dict[str, Any]:
    section = manifest.get("geometry")
    if not isinstance(section, dict) or section.get("schema") != GEOMETRY_MAP_SCHEMA:
        raise ValueError("source manifest must contain the native geometry map")
    by_patient = section.get("by_patient")
    if not isinstance(by_patient, dict):
        raise ValueError("source manifest geometry must contain by_patient")
    return by_patient


def _source_shape(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, dict):
        return None
    shape = value.get("shape_zyx")
    if not isinstance(shape, list) or len(shape) != 3:
        return None
    if any(type(item) is not int or item <= 0 for item in shape):
        return None
    return tuple(shape)


def _patient_geometry(
    patient_id: str,
    cine_path: Path,
    manifest_geometry: Any,
    unit_declarations: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one source header and return its replacement geometry and audit row."""
    row: dict[str, Any] = {
        "patient_id": patient_id,
        "cine_path": str(cine_path),
        "manifest_shape_zyx": None,
        "manifest_spacing_zyx_mm": None,
        "manifest_affine_zyx": None,
        "source_file_sha256": None,
        "source_header_sha256": None,
        "source_shape_xyzt": None,
        "header_zooms_xyz_native": None,
        "header_zooms_xyz_mm": None,
        "sform_code": None,
        "sform_affine_xyz": None,
        "qform_code": None,
        "qform_affine_xyz": None,
        "selected_affine_xyz": None,
        "original_spatial_unit": None,
        "resolved_spatial_unit": None,
        "unit_resolution": None,
        "unit_provenance": None,
        "cine_mapping_source": None,
        "cine_mapping_sha256": None,
        "cine_mapping_provenance": None,
        "voxel_bridge_status": "NOT_VERIFIED",
        "repaired_spacing_zyx_mm": None,
        "repaired_affine_zyx": None,
        "errors": [],
    }
    if isinstance(manifest_geometry, dict):
        row["manifest_shape_zyx"] = manifest_geometry.get("shape_zyx")
        row["manifest_spacing_zyx_mm"] = manifest_geometry.get("spacing_zyx_mm")
        row["manifest_affine_zyx"] = manifest_geometry.get("affine_zyx")
    expected_shape = _source_shape(manifest_geometry)
    if expected_shape is None:
        row["errors"].append("source manifest geometry must declare positive integer shape_zyx")

    issues = row["errors"]
    try:
        row["source_file_sha256"] = _sha256_file(cine_path)
    except Exception as exc:
        issues.append(f"cannot hash source CINE file: {exc}")
        return {}, row

    try:
        import nibabel as nib  # Optional preparation dependency; no voxel reads are performed.

        image = nib.load(str(cine_path), mmap=False)
    except Exception as exc:
        issues.append(f"cannot read source NIfTI header: {exc}")
        return {}, row

    header = getattr(image, "header", None)
    if header is None:
        issues.append("NIfTI header is missing")
        return {}, row
    binary_header = getattr(header, "binaryblock", None)
    if binary_header is None:
        issues.append("NIfTI header bytes are unavailable")
    else:
        row["source_header_sha256"] = hashlib.sha256(bytes(binary_header)).hexdigest()

    try:
        source_shape = tuple(int(item) for item in getattr(image, "shape", ()))
        row["source_shape_xyzt"] = list(source_shape)
    except Exception as exc:
        source_shape = ()
        issues.append(f"cannot read source NIfTI shape from header: {exc}")
    if len(source_shape) != 4 or any(item <= 0 for item in source_shape):
        issues.append("source CINE NIfTI must have positive native XYZT shape")
    elif expected_shape is not None and tuple(reversed(source_shape[:3])) != expected_shape:
        issues.append(
            f"source XYZ shape {source_shape[:3]} does not match manifest ZYX shape {expected_shape}"
        )

    try:
        zooms = np.asarray(header.get_zooms()[:3], dtype=np.float64)
        row["header_zooms_xyz_native"] = zooms.tolist()
    except Exception as exc:
        zooms = np.asarray([], dtype=np.float64)
        issues.append(f"cannot read NIfTI header zooms: {exc}")
    if zooms.shape != (3,) or not np.isfinite(zooms).all() or (zooms <= 0).any():
        issues.append("NIfTI header zooms must be finite and positive")
    elif np.allclose(zooms, np.ones(3), rtol=0.0, atol=1e-6):
        issues.append("NIfTI header spacing [1,1,1] is not accepted as authoritative geometry")

    try:
        units = header.get_xyzt_units()
        original_unit = units[0] if isinstance(units, (tuple, list)) and units else None
        row["original_spatial_unit"] = original_unit
    except Exception as exc:
        original_unit = None
        issues.append(f"cannot read NIfTI header spatial units: {exc}")
    resolved_unit: str | None = None
    try:
        resolved_unit, unit_resolution, unit_provenance = _resolve_units(
            patient_id, original_unit, unit_declarations
        )
        row["resolved_spatial_unit"] = resolved_unit
        row["unit_resolution"] = unit_resolution
        row["unit_provenance"] = unit_provenance
        if zooms.shape == (3,) and np.isfinite(zooms).all():
            row["header_zooms_xyz_mm"] = (zooms * SPATIAL_UNIT_SCALE_MM[resolved_unit]).tolist()
    except Exception as exc:
        issues.append(str(exc))

    forms, selected, form_errors = _header_forms(header)
    issues.extend(form_errors)
    for name in ("sform", "qform"):
        row[f"{name}_code"] = forms[name]["code"]
        matrix = forms[name]["affine"]
        row[f"{name}_affine_xyz"] = None if matrix is None else matrix.tolist()

    affine = None if selected is None else selected["affine"]
    if affine is not None:
        row["selected_affine_xyz"] = affine.tolist()
        if not np.isfinite(affine).all():
            issues.append("authoritative NIfTI affine must be finite")
        elif not np.allclose(affine[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-6):
            issues.append("authoritative NIfTI affine must be homogeneous")
        elif np.allclose(affine, np.eye(4), rtol=0.0, atol=1e-8):
            issues.append("identity affine is not accepted as authoritative geometry")
        else:
            linear = affine[:3, :3]
            lengths = np.linalg.norm(linear, axis=0)
            if not np.isfinite(lengths).all() or (lengths <= 0).any():
                issues.append("authoritative NIfTI affine has a degenerate spatial axis")
            else:
                if zooms.shape == (3,) and np.isfinite(zooms).all() and not np.allclose(
                    lengths, zooms, rtol=1e-5, atol=1e-6
                ):
                    issues.append("NIfTI header zooms disagree with authoritative affine column lengths")
                directions = linear / lengths
                if not np.allclose(directions.T @ directions, np.eye(3), rtol=1e-5, atol=1e-5):
                    issues.append("sheared authoritative NIfTI affine is unsupported")

    if expected_shape is None:
        issues.append("manifest spatial shape is required for geometry repair")
    if issues or affine is None or resolved_unit is None or expected_shape is None:
        return {}, row
    try:
        geometry = native_geometry_zyx(
            affine_in_mm(affine, resolved_unit), expected_shape, source_axis_order="XYZ"
        )
        geometry = validate_geometry_record(geometry)
    except Exception as exc:
        issues.append(str(exc))
        return {}, row
    row["repaired_spacing_zyx_mm"] = geometry["spacing_zyx_mm"]
    row["repaired_affine_zyx"] = geometry["affine_zyx"]
    return geometry, row


def validate_audit_geometry(row: Mapping[str, Any], geometry: Mapping[str, Any]) -> None:
    """Bind a serialized header audit to the full corrected affine, without I/O.

    Original NIfTI file hashes and prepared array hashes describe different
    payloads. This check preserves that boundary; it validates geometry only.
    """
    def matrix(key: str) -> np.ndarray:
        try:
            value = np.asarray(row.get(key), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"audit {key} must be a finite affine") from exc
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError(f"audit {key} must be a finite 4x4 affine")
        if not np.allclose(value[3], [0, 0, 0, 1], rtol=0, atol=1e-6):
            raise ValueError(f"audit {key} must be homogeneous")
        return value

    forms = {}
    for name in ("sform", "qform"):
        code = row.get(f"{name}_code")
        if type(code) is not int or code < 0:
            raise ValueError(f"audit {name} code must be declared")
        if code > 0:
            forms[name] = matrix(f"{name}_affine_xyz")
    if not forms:
        raise ValueError("audit has no coded authoritative affine")
    if len(forms) == 2 and not np.allclose(forms["sform"], forms["qform"], rtol=1e-5, atol=1e-5):
        raise ValueError("audit coded sform and qform disagree")
    selected = matrix("selected_affine_xyz")
    authoritative = forms.get("sform", forms.get("qform"))
    if not np.allclose(selected, authoritative, rtol=1e-5, atol=1e-6):
        raise ValueError("audit selected affine differs from coded authoritative affine")
    zooms = np.asarray(row.get("header_zooms_xyz_native"), dtype=np.float64)
    if zooms.shape != (3,) or not np.isfinite(zooms).all() or (zooms <= 0).any():
        raise ValueError("audit native header zooms must be finite and positive")
    if not np.allclose(np.linalg.norm(selected[:3, :3], axis=0), zooms, rtol=1e-5, atol=1e-6):
        raise ValueError("audit header zooms disagree with authoritative affine")
    unit = row.get("resolved_spatial_unit")
    if unit not in SPATIAL_UNIT_SCALE_MM:
        raise ValueError("audit resolved spatial unit is unsupported")
    converted_zooms = np.asarray(row.get("header_zooms_xyz_mm"), dtype=np.float64)
    if converted_zooms.shape != (3,) or not np.allclose(
        converted_zooms, zooms * SPATIAL_UNIT_SCALE_MM[unit], rtol=1e-5, atol=1e-6,
    ):
        raise ValueError("audit converted header zooms disagree with declared unit")
    normalized = validate_geometry_record(geometry)
    expected = native_geometry_zyx(
        affine_in_mm(selected, unit), normalized["shape_zyx"], source_axis_order="XYZ",
    )
    if not np.allclose(expected["affine_zyx"], normalized["affine_zyx"], rtol=1e-5, atol=1e-6):
        raise ValueError("audit authoritative affine differs from corrected XYZ-to-ZYX geometry")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("xb")  # An existing path is never owned or removed here.
    try:
        with stream as opened:
            opened.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_outputs(outputs: list[tuple[Path, bytes]]) -> None:
    """Publish only fully serialized new outputs; roll back this batch on I/O failure."""
    written: list[Path] = []
    try:
        for path, payload in outputs:
            _write_new(path, payload)
            written.append(path)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            field: " | ".join(row.get(field, [])) if field == "errors" else _json_cell(row.get(field))
            for field in _CSV_FIELDS
        })
    return stream.getvalue().encode("utf-8")


def repair_manifest_geometry(
    source_manifest_path: str | Path,
    source_manifest_sha256: str,
    cine_paths_by_patient: Mapping[str, str | Path],
    *,
    axis_map: str,
    cine_mapping_source: str,
    cine_mapping_sha256: str,
    cine_mapping_provenance: str,
    report_json_path: str | Path,
    report_csv_path: str | Path,
    output_manifest_path: str | Path | None = None,
    unit_declarations_by_patient: Mapping[str, Mapping[str, Any]] | None = None,
    audit_only: bool = True,
) -> dict[str, Any]:
    """Audit or build a new manifest using source CINE header geometry.

    The default pass only writes JSON and CSV diagnostics. A rejected batch
    never writes a corrected manifest. Output files must all be new paths.
    """
    source_path = Path(source_manifest_path)
    output_path = None if output_manifest_path is None else Path(output_manifest_path)
    json_path = Path(report_json_path)
    csv_path = Path(report_csv_path)
    if not audit_only and output_path is None:
        raise ValueError("output_manifest_path is required when audit_only is false")
    declared_outputs = [json_path.resolve(), csv_path.resolve()]
    if output_path is not None:
        declared_outputs.append(output_path.resolve())
    if len(set(declared_outputs)) != len(declared_outputs):
        raise ValueError("manifest and audit report paths must be distinct")
    if any(path.exists() for path in declared_outputs):
        raise FileExistsError("geometry repair outputs must be new paths; refusing overwrite")
    if axis_map != _AXIS_MAP:
        raise ValueError(f"axis_map must be explicitly declared as {_AXIS_MAP!r}")

    declarations = unit_declarations_by_patient or {}
    report: dict[str, Any] = {
        "schema": REPAIR_SCHEMA,
        "status": "rejected",
        "source_manifest_path": str(source_path),
        "source_manifest_sha256_expected": source_manifest_sha256,
        "axis_map": axis_map,
        "audit_only": bool(audit_only),
        "cine_mapping": {
            "source": cine_mapping_source,
            "sha256": cine_mapping_sha256,
            "provenance": cine_mapping_provenance,
            "voxel_bridge_status": "NOT_VERIFIED",
            "shape_match_is_not_voxel_equivalence": True,
        },
        "unit_declarations_by_patient": copy.deepcopy(dict(declarations)),
        "audit_rows": [],
        "errors": [],
        "receipt": None,
    }
    records: list[dict[str, Any]] = []
    source_manifest: dict[str, Any] | None = None
    actual_source_sha: str | None = None
    if not isinstance(source_manifest_sha256, str) or len(source_manifest_sha256) != 64:
        report["errors"].append("source_manifest_sha256 must be an explicit 64-character SHA-256")
    elif any(character not in "0123456789abcdefABCDEF" for character in source_manifest_sha256):
        report["errors"].append("source_manifest_sha256 must be hexadecimal")
    elif not source_path.is_file():
        report["errors"].append("source manifest file is missing")
    else:
        try:
            actual_source_sha = _sha256_file(source_path)
            report["source_manifest_sha256_actual"] = actual_source_sha
            if actual_source_sha.lower() != source_manifest_sha256.lower():
                report["errors"].append("source manifest SHA-256 mismatch")
            else:
                parsed = json.loads(source_path.read_text(encoding="utf-8"))
                if not isinstance(parsed, dict):
                    raise ValueError("source manifest JSON must contain an object")
                source_manifest = parsed
                # Non-geometry values are immutable protocol metadata. Do not
                # sanitize nonfinite values into different metadata on output.
                json.dumps({key: value for key, value in parsed.items() if key != "geometry"}, allow_nan=False)
                records = _records(source_manifest)
        except Exception as exc:
            report["errors"].append(str(exc))

    if not isinstance(cine_mapping_source, str) or not cine_mapping_source.strip():
        report["errors"].append("cine_mapping_source must be a declared non-empty source")
    if (
        not isinstance(cine_mapping_sha256, str)
        or len(cine_mapping_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in cine_mapping_sha256)
    ):
        report["errors"].append("cine_mapping_sha256 must be an explicit 64-character SHA-256")
    if not isinstance(cine_mapping_provenance, str) or not cine_mapping_provenance.strip():
        report["errors"].append("cine_mapping_provenance must be declared")

    replacement_geometry: dict[str, Any] = {}
    if source_manifest is not None:
        ids = [record["id"] for record in records]
        try:
            old_geometry = _manifest_geometry(source_manifest)
        except Exception as exc:
            old_geometry = {}
            report["errors"].append(str(exc))
        cine_paths = {str(pid): Path(path) for pid, path in cine_paths_by_patient.items()}
        missing_paths = sorted(set(ids) - set(cine_paths))
        extra_paths = sorted(set(cine_paths) - set(ids))
        if missing_paths:
            report["errors"].append(f"missing original CINE paths for patients: {missing_paths}")
        if extra_paths:
            report["errors"].append(f"CINE path map contains unknown patients: {extra_paths}")
        if set(old_geometry) != set(ids):
            report["errors"].append("source geometry patient IDs must exactly match manifest record IDs")
        if set(declarations) - set(ids):
            report["errors"].append(
                f"unit declarations contain unknown patients: {sorted(set(declarations) - set(ids))}"
            )

        for patient_id in ids:
            cine_path = cine_paths.get(patient_id, Path(""))
            geometry, row = _patient_geometry(
                patient_id,
                cine_path,
                old_geometry.get(patient_id),
                declarations,
            )
            row["cine_mapping_source"] = cine_mapping_source
            row["cine_mapping_sha256"] = cine_mapping_sha256
            row["cine_mapping_provenance"] = cine_mapping_provenance
            report["audit_rows"].append(row)
            if row["errors"]:
                report["errors"].extend(
                    {"patient_id": patient_id, "error": error} for error in row["errors"]
                )
            elif geometry:
                replacement_geometry[patient_id] = geometry

        for patient_id in extra_paths:
            report["audit_rows"].append({
                "patient_id": patient_id,
                "cine_path": str(cine_paths[patient_id]),
                "errors": ["CINE path map contains unknown patient ID"],
            })

    geometry_valid = (
        source_manifest is not None
        and not report["errors"]
        and len(replacement_geometry) == len(records)
    )
    if geometry_valid and not audit_only:
        repaired_manifest = copy.deepcopy(source_manifest)
        repaired_manifest["geometry"] = {
            "schema": GEOMETRY_MAP_SCHEMA,
            "by_patient": replacement_geometry,
        }
        repaired_bytes = _json_bytes(repaired_manifest)
        repaired_sha = hashlib.sha256(repaired_bytes).hexdigest()
        report["status"] = "success"
        report["receipt"] = {
            "schema": RECEIPT_SCHEMA,
            "status": "success",
            "geometry_only": True,
            "source_manifest_sha256": actual_source_sha,
            "repaired_manifest_sha256": repaired_sha,
            "patient_ids": [record["id"] for record in records],
            "axis_map": axis_map,
            "cine_mapping_source": cine_mapping_source,
            "cine_mapping_sha256": cine_mapping_sha256,
            "cine_mapping_provenance": cine_mapping_provenance,
            "voxel_bridge_status": "NOT_VERIFIED",
        }
        report_bytes = _json_bytes(report)
        csv_bytes = _csv_bytes(report["audit_rows"])
        _write_outputs([(json_path, report_bytes), (csv_path, csv_bytes), (output_path, repaired_bytes)])
    elif geometry_valid:
        report["status"] = "audit_only_pass"
        report["receipt"] = {
            "schema": RECEIPT_SCHEMA,
            "status": "audit_only",
            "geometry_only": True,
            "source_manifest_sha256": actual_source_sha,
            "patient_ids": [record["id"] for record in records],
            "axis_map": axis_map,
            "cine_mapping_source": cine_mapping_source,
            "cine_mapping_sha256": cine_mapping_sha256,
            "cine_mapping_provenance": cine_mapping_provenance,
            "voxel_bridge_status": "NOT_VERIFIED",
        }
        report_bytes = _json_bytes(report)
        _write_outputs([(json_path, report_bytes), (csv_path, _csv_bytes(report["audit_rows"]))])
    else:
        report_bytes = _json_bytes(report)
        _write_outputs([(json_path, report_bytes), (csv_path, _csv_bytes(report["audit_rows"]))])
    return report


def validate_geometry_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one cached native geometry record before data access.

    This is the lightweight runtime guard for callers that already possess a
    manifest. It does not open files or require a raw NIfTI header.
    """
    if not isinstance(record, Mapping):
        raise ValueError("geometry record must be a mapping")
    normalized = _validate_cache_geometry(dict(record))
    spacing = np.asarray(normalized["spacing_zyx_mm"], dtype=np.float64)
    affine = np.asarray(normalized["affine_zyx"], dtype=np.float64)
    xyz_to_zyx = np.eye(4, dtype=np.float64)
    xyz_to_zyx[:3, :3] = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    if np.allclose(affine, np.eye(4), rtol=0.0, atol=1e-8):
        raise ValueError("identity affine_zyx is not accepted as authoritative")
    if np.allclose(affine, xyz_to_zyx, rtol=0.0, atol=1e-8):
        raise ValueError("native XYZ identity affine reordered to ZYX is not authoritative")
    if np.allclose(spacing, np.ones(3), rtol=0.0, atol=1e-6):
        raise ValueError("geometry spacing [1,1,1] is not accepted as authoritative")
    if normalized["class_mapping"] != CLASS_MAPPING:
        raise ValueError("geometry class_mapping must be the explicit native ontology")
    return normalized


def validate_geometry_repair_receipt(
    source_manifest: Mapping[str, Any],
    repaired_manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
    repaired_manifest_sha256: str,
) -> dict[str, Any]:
    """Purely validate a repair receipt and a geometry-only manifest change.

    Hashes are supplied by the caller after hashing the exact source/output
    file bytes, so this helper performs no filesystem access.
    """
    if not isinstance(receipt, Mapping):
        raise ValueError("geometry repair receipt must be a mapping")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != "success":
        raise ValueError("invalid or unsuccessful geometry repair receipt")
    if receipt.get("geometry_only") is not True:
        raise ValueError("receipt must declare a geometry-only change")
    if receipt.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("receipt source manifest SHA-256 mismatch")
    if receipt.get("repaired_manifest_sha256") != repaired_manifest_sha256:
        raise ValueError("receipt repaired manifest SHA-256 mismatch")
    for name, value in (
        ("source manifest", source_manifest_sha256),
        ("repaired manifest", repaired_manifest_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise ValueError(f"{name} hash must be an explicit SHA-256 value")

    source_non_geometry = {key: value for key, value in source_manifest.items() if key != "geometry"}
    repaired_non_geometry = {key: value for key, value in repaired_manifest.items() if key != "geometry"}
    if source_non_geometry != repaired_non_geometry:
        raise ValueError("non-geometry manifest fields changed")

    source_geometry = _manifest_geometry(source_manifest)
    repaired_geometry = _manifest_geometry(repaired_manifest)
    if set(source_geometry) != set(repaired_geometry):
        raise ValueError("geometry repair changed patient IDs")
    source_record_ids = {record["id"] for record in _records(source_manifest)}
    if set(source_geometry) != source_record_ids:
        raise ValueError("source geometry IDs do not match manifest patient records")
    receipt_patient_ids = receipt.get("patient_ids")
    if not isinstance(receipt_patient_ids, list) or any(not isinstance(pid, str) for pid in receipt_patient_ids):
        raise ValueError("receipt patient_ids must be a list of strings")
    if sorted(receipt_patient_ids) != sorted(source_geometry):
        raise ValueError("receipt patient IDs do not match manifest geometry IDs")
    for patient_id in source_geometry:
        old_shape = _source_shape(source_geometry[patient_id])
        new = repaired_geometry[patient_id]
        if old_shape is None or _source_shape(new) != old_shape:
            raise ValueError(f"geometry repair changed or omitted native shape for {patient_id}")
        if new.get("schema") != GEOMETRY_SCHEMA:
            raise ValueError(f"repaired geometry for {patient_id} has invalid schema")
        if new.get("array_axes") != ["T", "Z", "Y", "X"] or new.get("spatial_axes") != ["Z", "Y", "X"]:
            raise ValueError(f"repaired geometry for {patient_id} has invalid axis declaration")
        validate_geometry_record(new)

    if receipt.get("axis_map") != _AXIS_MAP:
        raise ValueError("receipt axis map mismatch")
    if receipt.get("voxel_bridge_status") != "NOT_VERIFIED":
        raise ValueError("receipt must retain the unresolved voxel bridge status")
    for key in ("cine_mapping_source", "cine_mapping_sha256", "cine_mapping_provenance"):
        if not isinstance(receipt.get(key), str) or not receipt[key].strip():
            raise ValueError(f"receipt is missing declared {key}")
    mapping_sha = receipt["cine_mapping_sha256"]
    if len(mapping_sha) != 64 or any(character not in "0123456789abcdefABCDEF" for character in mapping_sha):
        raise ValueError("receipt CINE mapping hash must be a SHA-256 value")

    return {
        "valid": True,
        "source_manifest_sha256": source_manifest_sha256,
        "repaired_manifest_sha256": repaired_manifest_sha256,
        "patient_count": len(source_geometry),
        "geometry_only": True,
    }
