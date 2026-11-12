"""Fail-closed helpers for the real-data Q3D Phase D GPU profile.

The profile runs the production Q3D engine with its full recipe budgets and
passes ``update_limit=20`` to bound each measured stage.  This module does not
provide a proxy model, crop, synthetic fallback, or alternate loss path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from .campaign_runtime import (
    EXPLORATORY_WAIVER_CONTENT_HASH,
    EXPLORATORY_WAIVER_FLAG,
    exploratory_marker,
    exploratory_waiver_policy,
    waiver_policy_sha256,
)

PROFILE_UPDATE_LIMIT = 20
PRODUCTION_STAGE_BUDGETS = {"T": 8000, "Q": 1000, "P": 1, "J": 8000}
_REQUIRED_STAGE_ARTIFACTS = ("T", "Q", "P")
_PROFILE_STAGE_SELECTIONS = (("T",), ("J",), ("T", "J"))


class ProfilePrerequisiteError(ValueError):
    """A profile prerequisite is missing or does not match the locked recipe."""


def _validate_measured_stages(measured_stages: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(measured_stages, tuple) or measured_stages not in _PROFILE_STAGE_SELECTIONS:
        raise ProfilePrerequisiteError(
            "measured_stages must be one of ('T',), ('J',), or ('T', 'J')"
        )
    return measured_stages


@dataclass(frozen=True)
class PatientVolume:
    patient_id: str
    shape_zyx: tuple[int, int, int]
    voxel_count: int
    pool: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "shape_zyx": list(self.shape_zyx),
            "voxel_count_per_frame": self.voxel_count,
            "pool": self.pool,
        }


@dataclass(frozen=True)
class StageArtifact:
    stage: str
    manifest_path: Path
    manifest_sha256: str
    exploratory: bool = False
    exploratory_quality_waiver: dict[str, Any] | None = field(default=None, compare=False, hash=False)
    exploratory_policy_sha256: str | None = None

    def engine_binding(self) -> dict[str, str]:
        return {
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
        }

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
        }
        if self.exploratory:
            record["exploratory"] = True
            record["exploratory_quality_waiver"] = self.exploratory_quality_waiver
            record[EXPLORATORY_WAIVER_CONTENT_HASH] = self.exploratory_policy_sha256
        return record


@dataclass(frozen=True)
class ProfileInputs:
    config: dict[str, Any]
    config_path: Path
    config_sha256: str
    train_manifest_path: Path
    train_manifest_sha256: str
    development_manifest_path: Path
    development_manifest_sha256: str
    largest_l: PatientVolume
    largest_u: PatientVolume
    largest_train: PatientVolume
    stage_artifacts: dict[str, StageArtifact]
    labeled_patient_ids: tuple[str, ...]
    unlabeled_patient_ids: tuple[str, ...]
    train_geometry_evidence: dict[str, Any] = field(default_factory=dict)
    development_geometry_evidence: dict[str, Any] = field(default_factory=dict)
    phase_evidence: dict[str, Any] = field(default_factory=dict)
    geometry_policy: str = "strict"
    exploratory_quality_waiver: dict[str, Any] | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfilePrerequisiteError(f"{label} is missing or invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProfilePrerequisiteError(f"{label} must contain a JSON object: {path}")
    return value


def _manifest_record(config: Mapping[str, Any], name: str) -> tuple[Path, str]:
    raw = config.get("dataset", {}).get("manifests", {}).get(name)
    if not isinstance(raw, Mapping):
        raise ProfilePrerequisiteError(f"Q3D config has no {name} manifest record")
    path_value, expected = raw.get("path"), raw.get("sha256")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ProfilePrerequisiteError(f"Q3D {name} manifest path is not pinned")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ProfilePrerequisiteError(f"Q3D {name} manifest SHA-256 is not pinned")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ProfilePrerequisiteError(f"Q3D {name} manifest does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ProfilePrerequisiteError(f"Q3D {name} manifest SHA-256 mismatch: {path}")
    return path, actual


def _pinned_json(path_value: str | Path, expected_sha256: str, *, label: str) -> tuple[Path, str, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ProfilePrerequisiteError(f"{label} is missing: {path}")
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)):
        raise ProfilePrerequisiteError(f"{label} SHA-256 must be explicit lowercase hex")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ProfilePrerequisiteError(f"{label} SHA-256 mismatch: {path}")
    return path, actual, _read_json_object(path, label=label)


def _geometry_records(manifest: Mapping[str, Any], *, label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    patients = manifest.get("patients")
    if not isinstance(patients, list) or not patients or any(not isinstance(row, Mapping) for row in patients):
        raise ProfilePrerequisiteError(f"{label} must expose its patient records")
    ids = [row.get("id") for row in patients]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ProfilePrerequisiteError(f"{label} patient IDs must be unique nonempty strings")
    geometry = manifest.get("geometry")
    from ..data import GEOMETRY_MAP_SCHEMA

    if (not isinstance(geometry, Mapping) or geometry.get("schema") != GEOMETRY_MAP_SCHEMA
            or not isinstance(geometry.get("by_patient"), Mapping)):
        raise ProfilePrerequisiteError(f"{label} must declare a native per-patient geometry map")
    by_patient = geometry["by_patient"]
    if set(by_patient) != set(ids):
        raise ProfilePrerequisiteError(f"{label} native geometry inventory differs from patient IDs")
    return [dict(row) for row in patients], dict(by_patient)


def _audit_geometry_policy(container: Mapping[str, Any], *, label: str) -> str:
    policy = container.get("geometry_policy", "strict")
    if not isinstance(policy, str) or policy not in {"strict", "acdc_pixdim_native_v1"}:
        raise ProfilePrerequisiteError(f"{label} has an unsupported geometry_policy")
    return policy


def _validate_repair_audit_rows(
    source_manifest: Mapping[str, Any],
    repaired_manifest: Mapping[str, Any],
    repair_report: Mapping[str, Any],
    *,
    label: str,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    repaired_manifest_sha256: str,
) -> dict[str, Any]:
    """Check shared repair evidence against both exact manifests, metadata only."""
    from .geometry_repair import (
        REPAIR_SCHEMA,
        validate_audit_geometry,
        validate_geometry_record,
        validate_geometry_repair_receipt,
    )

    if repair_report.get("schema") != REPAIR_SCHEMA or repair_report.get("status") != "success":
        raise ProfilePrerequisiteError(f"{label} repair report must be a successful geometry_repair.v1 report")
    if repair_report.get("audit_only") is not False or repair_report.get("errors") != []:
        raise ProfilePrerequisiteError(f"{label} repair report must be a successful non-audit-only repair with no errors")
    receipt = repair_report.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ProfilePrerequisiteError(f"{label} repair report has no validated repair receipt")
    receipt_policy = _audit_geometry_policy(receipt, label=f"{label} repair receipt")
    report_policy = _audit_geometry_policy(repair_report, label=f"{label} repair report")
    if receipt_policy != report_policy:
        raise ProfilePrerequisiteError(f"{label} geometry_policy differs between repair receipt and report")
    source_path = repair_report.get("source_manifest_path")
    if (not isinstance(source_path, str)
            or Path(source_path).expanduser().resolve() != source_manifest_path.resolve()):
        raise ProfilePrerequisiteError(f"{label} repair report source path is not bound")
    source_sha = source_manifest_sha256
    repaired_sha = repaired_manifest_sha256
    if (repair_report.get("source_manifest_sha256_expected") != source_sha
            or repair_report.get("source_manifest_sha256_actual") != source_sha):
        raise ProfilePrerequisiteError(f"{label} repair report source SHA-256 lineage is invalid")
    try:
        receipt_validation = validate_geometry_repair_receipt(
            source_manifest,
            repaired_manifest,
            receipt,
            source_manifest_sha256=source_sha,
            repaired_manifest_sha256=repaired_sha,
            geometry_policy=receipt_policy,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfilePrerequisiteError(f"{label} geometry repair receipt is invalid: {exc}") from exc

    source_rows, source_geometry = _geometry_records(source_manifest, label=f"source {label}")
    repaired_rows, repaired_geometry = _geometry_records(repaired_manifest, label=f"corrected {label}")
    source_by_id = {row["id"]: row for row in source_rows}
    repaired_by_id = {row["id"]: row for row in repaired_rows}
    if list(source_by_id) != list(repaired_by_id):
        raise ProfilePrerequisiteError(f"{label} repair changed patient inventory/order")
    audit_rows = repair_report.get("audit_rows")
    if not isinstance(audit_rows, list):
        raise ProfilePrerequisiteError(f"{label} repair report must contain audited patient rows")
    audit_by_id: dict[str, Mapping[str, Any]] = {}
    for row in audit_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("patient_id"), str):
            raise ProfilePrerequisiteError(f"{label} repair report contains an invalid audit row")
        if row["patient_id"] in audit_by_id:
            raise ProfilePrerequisiteError(f"{label} repair report has duplicate audit rows")
        audit_by_id[row["patient_id"]] = row
    if set(audit_by_id) != set(source_by_id):
        raise ProfilePrerequisiteError(f"{label} audit rows must exactly cover its patient IDs")
    for patient_id in source_by_id:
        row = audit_by_id[patient_id]
        row_policy = _audit_geometry_policy(row, label=f"{label} audit row {patient_id}")
        if row_policy != receipt_policy:
            raise ProfilePrerequisiteError(
                f"{label} geometry_policy differs between repair receipt and audit row {patient_id}"
            )
        old = source_geometry[patient_id]
        new = repaired_geometry[patient_id]
        if row.get("errors") != []:
            raise ProfilePrerequisiteError(f"{label} audit row {patient_id} did not pass")
        if row.get("voxel_bridge_status") != "NOT_VERIFIED":
            raise ProfilePrerequisiteError(f"{label} audit row {patient_id} must retain voxel bridge NOT_VERIFIED")
        for mapping_key in ("cine_mapping_source", "cine_mapping_sha256", "cine_mapping_provenance"):
            if row.get(mapping_key) != receipt.get(mapping_key):
                raise ProfilePrerequisiteError(f"{label} audit row {patient_id} differs from receipt at {mapping_key}")
        for hash_key in ("source_file_sha256", "source_header_sha256"):
            value = row.get(hash_key)
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
                raise ProfilePrerequisiteError(f"{label} audit row {patient_id} has invalid {hash_key}")
        for unit_key in ("resolved_spatial_unit", "unit_resolution"):
            if not isinstance(row.get(unit_key), str) or not row[unit_key].strip():
                raise ProfilePrerequisiteError(f"{label} audit row {patient_id} lacks {unit_key}")
        if row["unit_resolution"] not in {"explicit_declaration", "nifti_header"}:
            raise ProfilePrerequisiteError(f"{label} audit row {patient_id} has unknown unit resolution")
        if row["unit_resolution"] == "explicit_declaration" and (
            not isinstance(row.get("unit_provenance"), str) or not row["unit_provenance"].strip()
        ):
            raise ProfilePrerequisiteError(f"{label} audit row {patient_id} lacks explicit unit provenance")
        if row.get("manifest_shape_zyx") != old.get("shape_zyx"):
            raise ProfilePrerequisiteError(f"{label} audit row {patient_id} source shape differs from manifest")
        if row.get("manifest_spacing_zyx_mm") != old.get("spacing_zyx_mm"):
            raise ProfilePrerequisiteError(f"{label} audit row {patient_id} source spacing differs from manifest")
        if row.get("manifest_affine_zyx") != old.get("affine_zyx"):
            raise ProfilePrerequisiteError(f"{label} audit row {patient_id} source affine differs from manifest")
        if row.get("repaired_spacing_zyx_mm") != new.get("spacing_zyx_mm"):
            raise ProfilePrerequisiteError(f"{label} audit row {patient_id} spacing differs from corrected manifest")
        if row.get("repaired_affine_zyx") != new.get("affine_zyx"):
            raise ProfilePrerequisiteError(f"{label} audit row {patient_id} affine differs from corrected manifest")
        source_shape = row.get("source_shape_xyzt")
        if (not isinstance(source_shape, list) or len(source_shape) != 4
                or any(type(value) is not int or value <= 0 for value in source_shape)
                or list(reversed(source_shape[:3])) != old.get("shape_zyx")):
            raise ProfilePrerequisiteError(f"{label} audit row {patient_id} source header shape is invalid")
        header_zooms = row.get("header_zooms_xyz_mm")
        try:
            converted = [float(value) for value in reversed(header_zooms)]
            spacing = [float(value) for value in new["spacing_zyx_mm"]]
        except (TypeError, ValueError, KeyError):
            raise ProfilePrerequisiteError(f"{label} audit row {patient_id} header zooms are invalid") from None
        if len(converted) != 3 or any(not math.isfinite(value) or value <= 0 for value in converted) or any(
            not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6) for a, b in zip(converted, spacing)
        ):
            raise ProfilePrerequisiteError(f"{label} audit row {patient_id} header zooms differ from corrected geometry")
        try:
            validate_geometry_record(new)
            validate_audit_geometry(row, new, geometry_policy=receipt_policy)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProfilePrerequisiteError(
                f"{label} corrected geometry/header audit for {patient_id} is invalid: {exc}"
            ) from exc
    return {
        "status": "PASS",
        "source_manifest_sha256": source_sha,
        "corrected_manifest_sha256": repaired_sha,
        "repair_receipt_validation": receipt_validation,
        "patient_count": len(source_by_id),
        "audit_rows_passed": len(audit_by_id),
        "geometry_policy": receipt_policy,
        "arrays_or_headers_read_by_profiler": False,
    }


def _validate_geometry_reprobe(
    phase_receipt_path: str | Path,
    phase_receipt_sha256: str,
    proposal: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[Path, str, dict[str, Any]]:
    """Bind the canonical ResEnc phase report to repaired TRAIN geometry and config."""
    path, digest, payload = _pinned_json(
        phase_receipt_path, phase_receipt_sha256, label="corrected-geometry phase receipt",
    )
    if payload.get("schema") != "self_audit.q3d_resenc.erf_probe.v1":
        raise ProfilePrerequisiteError("phase evidence must use the canonical Q3D ResEnc ERF receipt schema")
    if payload.get("evidence_class") != "SOFTWARE_ONLY_SYNTHETIC_ARCHITECTURE_PROBE":
        raise ProfilePrerequisiteError("phase receipt must declare its synthetic architecture evidence class")
    geometry_reprobe = payload.get("geometry_reprobe")
    if not isinstance(geometry_reprobe, Mapping):
        raise ProfilePrerequisiteError(
            "old Phase-0 receipt is insufficient; require geometry_reprobe bound to corrected TRAIN"
        )
    if geometry_reprobe.get("status") != "PROBE_COMPLETED_NOT_A_GEOMETRY_LOCK":
        raise ProfilePrerequisiteError("phase geometry probe has an unexpected status")
    if geometry_reprobe.get("manifest_lineage") != proposal.get("manifest_lineage"):
        raise ProfilePrerequisiteError("phase receipt TRAIN manifest/repair-report lineage differs from recomputed proposal")
    if geometry_reprobe.get("proposal_geometry") != proposal.get("geometry"):
        raise ProfilePrerequisiteError("phase receipt geometry proposal differs from recomputed corrected TRAIN geometry")

    from .config import NNFOUNDATION_ARCHITECTURE
    from scripts.q3d_reprobe_geometry import RESENC_ARCHITECTURE

    teacher = config.get("model", {}).get("teacher", {})
    phase_model = payload.get("model", {})
    assets = geometry_reprobe.get("pinned_probe_assets", {})
    pretrained = teacher.get("pretrained", {})
    if (not isinstance(phase_model, Mapping)
            or teacher.get("architecture") != NNFOUNDATION_ARCHITECTURE
            or not isinstance(assets, Mapping)
            or phase_model.get("architecture") != RESENC_ARCHITECTURE
            or phase_model.get("checkpoint_sha256") != pretrained.get("weights_sha256")
            or phase_model.get("checkpoint_sha256") != assets.get("checkpoint_sha256")
            or phase_model.get("registry_sha256") != assets.get("registry_sha256")):
        raise ProfilePrerequisiteError("phase receipt model/checkpoint identity differs from pinned real teacher provenance")

    stride_schedule = proposal.get("geometry", {}).get("planner_output", {}).get("stage_strides_zyx")
    try:
        from scripts.q3d_reprobe_geometry import canonical_probe_parameters, phase_proposal_from_probe_result

        expected_contract = canonical_probe_parameters(stride_schedule)
        expected_phase = phase_proposal_from_probe_result(payload)
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        raise ProfilePrerequisiteError(f"canonical geometry phase receipt validation failed: {exc}") from exc
    if geometry_reprobe.get("probe_contract") != expected_contract:
        raise ProfilePrerequisiteError("phase receipt probe contract differs from corrected TRAIN topology")
    if geometry_reprobe.get("phase_proposal") != expected_phase:
        raise ProfilePrerequisiteError("phase receipt proposal differs from the canonical signed-mean rule")

    phase_measurement = teacher.get("phase_measurement", {})
    if phase_measurement.get("receipt_sha256") != digest:
        raise ProfilePrerequisiteError(
            "Phase D NOT_RUN: config still pins a different phase receipt; after Phase A/B passes, root must register "
            "this corrected-geometry phase receipt and its hash in the real-run config validator"
        )
    configured_receipt = phase_measurement.get("receipt_path")
    if not isinstance(configured_receipt, str) or not configured_receipt.strip():
        raise ProfilePrerequisiteError("config teacher.phase_measurement must pin the corrected geometry phase receipt path")
    configured_path = Path(configured_receipt).expanduser()
    if not configured_path.is_absolute():
        configured_path = Path(__file__).resolve().parents[2] / configured_path
    if configured_path.resolve() != path:
        raise ProfilePrerequisiteError(
            "Phase D NOT_RUN: config phase receipt path is not the corrected-geometry receipt; root must lock it "
            "after Phase A/B passes"
        )

    feature_source = teacher.get("feature_source")
    configured_offsets = teacher.get("phase_offsets", {}).get(feature_source)
    if expected_phase.get("status") == "MAIN_PHASE_ZERO_CORRECTION_PROPOSED":
        expected_offsets = expected_phase.get("proposed_correction_zyx_voxels")
    elif expected_phase.get("status") == "MEASURED_OFFSET_REVIEW_REQUIRED":
        expected_offsets = expected_phase.get("signed_mean_offset_zyx_voxels")
    else:
        raise ProfilePrerequisiteError("phase receipt has an unsupported phase decision")
    try:
        offset_matches = (
            isinstance(configured_offsets, list)
            and isinstance(expected_offsets, list)
            and len(configured_offsets) == len(expected_offsets) == 3
            and all(math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-7)
                    for actual, expected in zip(configured_offsets, expected_offsets))
        )
    except (TypeError, ValueError):
        offset_matches = False
    if not offset_matches:
        raise ProfilePrerequisiteError(
            "Phase D NOT_RUN: teacher phase offsets must be locked to the zero-correction proposal or explicitly "
            "measured mean after Phase A/B passes"
        )
    return path, digest, {
        "status": expected_phase.get("status"),
        "path": str(path),
        "sha256": digest,
        "manifest_lineage": geometry_reprobe["manifest_lineage"],
        "phase_proposal": expected_phase,
        "evidence_class": payload["evidence_class"],
    }


def _records_by_pool(manifest: Mapping[str, Any], pool: str) -> list[dict[str, Any]]:
    records = manifest.get(pool)
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise ProfilePrerequisiteError(f"ACDC TRAIN {pool} must be a list of patient records")
    ids = [item.get("id") for item in records]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ProfilePrerequisiteError(f"ACDC TRAIN {pool} patient IDs must be unique nonempty strings")
    return records


def _geometry_for_patient(geometry_by_patient: Mapping[str, Any], patient_id: str) -> tuple[int, int, int]:
    raw = geometry_by_patient.get(patient_id)
    if not isinstance(raw, Mapping):
        raise ProfilePrerequisiteError(f"native geometry is missing for TRAIN patient {patient_id!r}")
    try:
        # The shared Q3D validator rejects unit spacing, identity affines,
        # missing units and invalid axis declarations.  It reads metadata only.
        from .geometry_repair import validate_geometry_record

        geometry = validate_geometry_record(raw)
    except (TypeError, ValueError, KeyError) as exc:
        raise ProfilePrerequisiteError(
            f"invalid native full-volume geometry for TRAIN patient {patient_id!r}: {exc}"
        ) from exc
    return tuple(int(value) for value in geometry["shape_zyx"])


def _largest_pool_patient(
    records: list[dict[str, Any]], geometry_by_patient: Mapping[str, Any], pool: str,
) -> PatientVolume:
    if not records:
        raise ProfilePrerequisiteError(f"ACDC TRAIN {pool} has no patients")
    candidates: list[PatientVolume] = []
    for record in records:
        patient_id = str(record["id"])
        shape = _geometry_for_patient(geometry_by_patient, patient_id)
        candidates.append(PatientVolume(patient_id, shape, math.prod(shape), pool))
    # Stable ID tie-break makes the selection reproducible from manifest bytes.
    return sorted(candidates, key=lambda item: (-item.voxel_count, item.patient_id))[0]


def largest_train_patients(train_manifest: Mapping[str, Any]) -> tuple[PatientVolume, PatientVolume, PatientVolume]:
    """Select the largest native volume in L and U from manifest metadata only."""
    if (train_manifest.get("schema") != "self_audit.data.v2"
            or train_manifest.get("role") != "training"
            or train_manifest.get("synthetic") is not False
            or train_manifest.get("dataset") != "acdc"):
        raise ProfilePrerequisiteError("profile requires a real ACDC self_audit.data.v2 TRAIN manifest")
    labeled = _records_by_pool(train_manifest, "labeled")
    unlabeled = _records_by_pool(train_manifest, "unlabeled")
    label_budget = train_manifest.get("label_budget")
    if type(label_budget) is not int or label_budget != 8 or len(labeled) != 8:
        raise ProfilePrerequisiteError("profile requires the corrected ACDC K8 labeled TRAIN manifest")
    labeled_ids = {record["id"] for record in labeled}
    unlabeled_ids = {record["id"] for record in unlabeled}
    if labeled_ids & unlabeled_ids or not unlabeled_ids:
        raise ProfilePrerequisiteError("ACDC K8 L/U pools must be disjoint and U must be nonempty")
    for record in unlabeled:
        if set(record) != {"id", "image", "image_sha256"}:
            raise ProfilePrerequisiteError(
                "ACDC U records must be image-only; masks or reference paths are forbidden"
            )
    for record in labeled:
        if not isinstance(record.get("masks"), Mapping) or not record["masks"]:
            raise ProfilePrerequisiteError("each ACDC L patient must declare TRAIN masks")
    from ..data import GEOMETRY_MAP_SCHEMA

    geometry = train_manifest.get("geometry")
    if (not isinstance(geometry, Mapping) or geometry.get("schema") != GEOMETRY_MAP_SCHEMA
            or not isinstance(geometry.get("by_patient"), Mapping)):
        raise ProfilePrerequisiteError("ACDC TRAIN manifest must declare per-patient native geometry")
    geometry_by_patient = geometry["by_patient"]
    expected_ids = labeled_ids | unlabeled_ids
    if set(geometry_by_patient) != expected_ids:
        raise ProfilePrerequisiteError("ACDC TRAIN native geometry inventory differs from the L/U cohort")
    largest_l = _largest_pool_patient(labeled, geometry_by_patient, "L")
    largest_u = _largest_pool_patient(unlabeled, geometry_by_patient, "U")
    largest_train = sorted(
        (largest_l, largest_u), key=lambda item: (-item.voxel_count, item.patient_id),
    )[0]
    return largest_l, largest_u, largest_train


def _waiver_policy(reason: Any) -> dict[str, Any] | None:
    try:
        return exploratory_waiver_policy(reason)
    except ValueError as exc:
        raise ProfilePrerequisiteError(str(exc)) from exc


def _require_artifact_waivers(
    artifacts: Mapping[str, StageArtifact], policy: Mapping[str, Any] | None,
) -> None:
    waived = sorted(stage for stage, artifact in artifacts.items() if artifact.exploratory)
    if waived and policy is None:
        raise ProfilePrerequisiteError(
            f"stage artifacts {waived} are exploratory (quality gates waived); profiling them requires "
            f"the same explicit {EXPLORATORY_WAIVER_FLAG} REASON"
        )
    if policy is not None:
        expected = waiver_policy_sha256(policy)
        mismatched = sorted(
            stage for stage in waived
            if artifacts[stage].exploratory_policy_sha256 not in (None, expected)
            or (artifacts[stage].exploratory_quality_waiver is not None
                and exploratory_marker(artifacts[stage].as_dict())["policy_sha256"] != expected)
        )
        if mismatched:
            raise ProfilePrerequisiteError(
                f"stage artifacts {mismatched} were waived with a different {EXPLORATORY_WAIVER_FLAG} REASON"
            )


def parse_stage_artifact_bindings(
    paths: list[str], hashes: list[str],
    *, exploratory_quality_waiver: str | None = None,
) -> dict[str, StageArtifact]:
    """Parse and hash-check explicit T/Q/P stage-manifest bindings.

    Exploratory (quality-gate-waived) artifacts are accepted only with an
    explicit waiver reason, and their label is carried into the report.
    """
    policy = _waiver_policy(exploratory_quality_waiver)
    def parse(items: list[str], label: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for item in items:
            stage, separator, value = item.partition("=")
            if (not separator or stage not in _REQUIRED_STAGE_ARTIFACTS or not value
                    or stage in parsed):
                raise ProfilePrerequisiteError(
                    f"{label} requires unique T=..., Q=..., and P=... bindings"
                )
            parsed[stage] = value
        if set(parsed) != set(_REQUIRED_STAGE_ARTIFACTS):
            raise ProfilePrerequisiteError(
                f"J profiling requires supplied SHA-pinned T/Q/P stage artifacts; missing {sorted(set(_REQUIRED_STAGE_ARTIFACTS) - set(parsed))}"
            )
        return parsed

    path_values = parse(paths, "--stage-artifact")
    hash_values = parse(hashes, "--stage-artifact-sha")
    result: dict[str, StageArtifact] = {}
    for stage in _REQUIRED_STAGE_ARTIFACTS:
        expected = hash_values[stage]
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ProfilePrerequisiteError(f"{stage} stage artifact SHA-256 must be lowercase hex")
        path = Path(path_values[stage]).expanduser().resolve()
        if not path.is_file():
            raise ProfilePrerequisiteError(f"{stage} stage artifact manifest is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ProfilePrerequisiteError(f"{stage} stage artifact manifest SHA-256 mismatch: {path}")
        document = _read_json_object(path, label=f"{stage} stage artifact manifest")
        if document.get("stage") != stage:
            raise ProfilePrerequisiteError(
                f"stage artifact binding {stage} points to manifest stage {document.get('stage')!r}"
            )
        try:
            marker = exploratory_marker(document)
        except ValueError as exc:
            raise ProfilePrerequisiteError(f"{stage} stage artifact: {exc}") from exc
        result[stage] = StageArtifact(
            stage, path, actual,
            exploratory=marker is not None,
            exploratory_quality_waiver=marker["policy"] if marker is not None else None,
            exploratory_policy_sha256=marker["policy_sha256"] if marker is not None else None,
        )
    _require_artifact_waivers(result, policy)
    return result


def _check_development_manifest(payload: Mapping[str, Any]) -> None:
    if (payload.get("schema") != "self_audit.data.v2"
            or payload.get("role") != "development"
            or payload.get("synthetic") is not False
            or payload.get("dataset") != "acdc"):
        raise ProfilePrerequisiteError("profile requires a real ACDC DEV18 manifest")
    patients = payload.get("patients")
    if not isinstance(patients, list) or len(patients) != 18:
        raise ProfilePrerequisiteError("ACDC DEV18 manifest must expose exactly 18 development patients")
    ids = [item.get("id") for item in patients if isinstance(item, Mapping)]
    if len(ids) != len(patients) or any(not isinstance(item, str) or not item for item in ids):
        raise ProfilePrerequisiteError("ACDC DEV18 patient records are invalid")
    if len(set(ids)) != len(ids):
        raise ProfilePrerequisiteError("ACDC DEV18 patient IDs must be unique")
    geometry = payload.get("geometry")
    from ..data import GEOMETRY_MAP_SCHEMA

    if (not isinstance(geometry, Mapping) or geometry.get("schema") != GEOMETRY_MAP_SCHEMA
            or not isinstance(geometry.get("by_patient"), Mapping)):
        raise ProfilePrerequisiteError("ACDC DEV18 manifest must declare native per-patient geometry")
    by_patient = geometry["by_patient"]
    if set(by_patient) != set(ids):
        raise ProfilePrerequisiteError("ACDC DEV18 native geometry inventory differs from patients")
    for patient_id in ids:
        _geometry_for_patient(by_patient, patient_id)


def _phase_lock_error(exc: ValueError) -> ProfilePrerequisiteError | None:
    detail = str(exc).lower()
    if "phase" in detail and ("receipt" in detail or "measurement" in detail or "offset" in detail):
        return ProfilePrerequisiteError(
            "Phase D NOT_RUN: the current real-run config validator does not yet register the corrected-geometry "
            "phase lock; Phase A/B must pass and root must lock the new receipt and offsets before profiling. "
            f"Current validator: {exc}"
        )
    return None


def load_profile_inputs(
    config_path: str | Path,
    stage_artifacts: Mapping[str, StageArtifact],
    *,
    measured_stages: tuple[str, ...] = ("T", "J"),
    source_train_manifest_path: str | Path,
    source_train_manifest_sha256: str,
    train_repair_report_path: str | Path,
    train_repair_report_sha256: str,
    source_development_manifest_path: str | Path,
    source_development_manifest_sha256: str,
    development_repair_report_path: str | Path,
    development_repair_report_sha256: str,
    phase_receipt_path: str | Path,
    phase_receipt_sha256: str,
    exploratory_quality_waiver: str | None = None,
) -> ProfileInputs:
    """Validate real manifests, independent repair evidence, phase and teacher provenance."""
    measured_stages = _validate_measured_stages(measured_stages)
    waiver_policy = _waiver_policy(exploratory_quality_waiver)
    _require_artifact_waivers(stage_artifacts, waiver_policy)
    if "J" in measured_stages:
        if set(stage_artifacts) != set(_REQUIRED_STAGE_ARTIFACTS):
            raise ProfilePrerequisiteError(
                "J profiling requires exact verified T/Q/P stage artifact bindings; Q gate and P cache are mandatory"
            )
    elif stage_artifacts:
        raise ProfilePrerequisiteError("T-only profiling does not accept unused T/Q/P artifact bindings")
    from .config import load_config, validate_real_teacher_provenance

    source = Path(config_path).expanduser().resolve()
    if not source.is_file():
        raise ProfilePrerequisiteError(f"Q3D profiling config is missing: {source}")
    try:
        config = load_config(source, real_run=True)
    except ValueError as exc:
        phase_error = _phase_lock_error(exc)
        if phase_error is not None:
            raise phase_error from exc
        raise
    if config.get("arm") != "v1":
        raise ProfilePrerequisiteError("Phase D J profile requires arm=v1")
    if config.get("protocol", {}).get("variant") != "A" or config.get("protocol", {}).get("k") != 8:
        raise ProfilePrerequisiteError("Phase D profile requires ACDC protocol A with K=8")
    if (config.get("seed") != 42 or config.get("protocol", {}).get("seed") != 42
            or config.get("checkpoint", {}).get("namespace") != "q3d/acdc/protocol-a/k8/seed-42"):
        raise ProfilePrerequisiteError("Phase D profile requires the locked seed-42 ACDC K8 V1 namespace")
    if config.get("model", {}).get("student", {}).get("residual2", {}).get("enabled") is not False:
        raise ProfilePrerequisiteError("Phase D profile requires the V1 recipe with residual2 disabled")
    if config.get("dataset", {}).get("name") != "acdc":
        raise ProfilePrerequisiteError("Phase D profile supports the locked ACDC campaign only")
    runtime = config.get("runtime", {})
    if runtime.get("synthetic") is not False or runtime.get("synthetic_gate_override") is not False:
        raise ProfilePrerequisiteError("real GPU profiling forbids synthetic configs and gate overrides")
    if runtime.get("smoke_update_overrides") != {"T": None, "Q": None, "P": None, "J": None}:
        raise ProfilePrerequisiteError("real GPU profiling requires unchanged production stage budgets")
    updates = config.get("training", {}).get("stage_updates", {})
    if updates != PRODUCTION_STAGE_BUDGETS:
        raise ProfilePrerequisiteError(
            f"recipe stage budgets must remain {PRODUCTION_STAGE_BUDGETS}; update_limit separately caps profiling"
        )
    manifests = config.get("dataset", {}).get("manifests", {})
    if manifests.get("teacher_train", {}).get("path") is not None:
        raise ProfilePrerequisiteError("protocol A T must use the same K8 L pool; separate teacher_train is forbidden")
    train_path, train_sha = _manifest_record(config, "train")
    dev_path, dev_sha = _manifest_record(config, "development")
    train_manifest = _read_json_object(train_path, label="ACDC K8 TRAIN manifest")
    dev_manifest = _read_json_object(dev_path, label="ACDC DEV18 manifest")

    source_train_path, source_train_sha, source_train = _pinned_json(
        source_train_manifest_path, source_train_manifest_sha256, label="source ACDC K8 TRAIN manifest",
    )
    train_report_path, train_report_sha, train_repair_report = _pinned_json(
        train_repair_report_path, train_repair_report_sha256, label="ACDC K8 TRAIN geometry repair report",
    )
    try:
        from .geometry_planning import prepare_geometry_proposal

        train_proposal = prepare_geometry_proposal(
            source_manifest_path=source_train_path,
            repaired_manifest_path=train_path,
            repair_report_path=train_report_path,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ProfilePrerequisiteError(f"ACDC TRAIN repair/proposal evidence failed validation: {exc}") from exc
    lineage = train_proposal.get("manifest_lineage", {})
    if (train_proposal.get("status") != "PROPOSAL_ONLY_NOT_LOCKED"
            or train_proposal.get("evidence_class") != "SOFTWARE_ONLY_METADATA_PROPOSAL"
            or lineage.get("source_manifest_sha256") != source_train_sha
            or lineage.get("repaired_manifest_sha256") != train_sha
            or lineage.get("repair_report_sha256") != train_report_sha):
        raise ProfilePrerequisiteError("ACDC TRAIN geometry proposal is not bound to the pinned source, correction, and report")
    # Keep the repair report bytes tied to the same proposal lineage even if a
    # caller supplies another report with equivalent normalized JSON.
    if train_repair_report.get("schema") is None:
        raise ProfilePrerequisiteError("ACDC TRAIN geometry repair report has no declared schema")

    source_dev_path, source_dev_sha, source_dev = _pinned_json(
        source_development_manifest_path, source_development_manifest_sha256,
        label="source ACDC DEV18 manifest",
    )
    dev_report_path, dev_report_sha, dev_repair_report = _pinned_json(
        development_repair_report_path, development_repair_report_sha256,
        label="ACDC DEV18 geometry repair report",
    )
    _check_development_manifest(dev_manifest)
    if (source_dev.get("schema") != "self_audit.data.v2"
            or source_dev.get("role") != "development"
            or source_dev.get("synthetic") is not False
            or source_dev.get("dataset") != "acdc"):
        raise ProfilePrerequisiteError("source ACDC DEV18 manifest identity is invalid")
    source_dev_rows, _ = _geometry_records(source_dev, label="source ACDC DEV18 manifest")
    if len(source_dev_rows) != 18 or len(dev_manifest.get("patients", [])) != 18:
        raise ProfilePrerequisiteError("source and corrected ACDC DEV manifests must each contain exactly 18 patients")
    dev_validation = _validate_repair_audit_rows(
        source_dev,
        dev_manifest,
        dev_repair_report,
        label="ACDC DEV18",
        source_manifest_path=source_dev_path,
        source_manifest_sha256=source_dev_sha,
        repaired_manifest_sha256=dev_sha,
    )
    train_geometry_policy = _audit_geometry_policy(train_proposal, label="ACDC TRAIN geometry proposal")
    if train_geometry_policy != dev_validation["geometry_policy"]:
        raise ProfilePrerequisiteError("ACDC TRAIN and DEV geometry_policy differ")
    if dev_report_sha != sha256_file(dev_report_path):
        raise ProfilePrerequisiteError("ACDC DEV18 repair report SHA-256 changed during validation")

    largest_l, largest_u, largest_train = largest_train_patients(train_manifest)
    train_ids = {item["id"] for pool in ("labeled", "unlabeled") for item in train_manifest[pool]}
    dev_ids = {item["id"] for item in dev_manifest["patients"]}
    if train_ids & dev_ids:
        raise ProfilePrerequisiteError("ACDC TRAIN and DEV18 patient IDs overlap")
    evaluation_ids = train_manifest.get("evaluation_ids")
    if not isinstance(evaluation_ids, list) or not dev_ids.issubset(set(evaluation_ids)):
        raise ProfilePrerequisiteError("ACDC TRAIN manifest does not bind the DEV18 IDs as evaluation patients")
    phase_path, phase_sha, phase_summary = _validate_geometry_reprobe(
        phase_receipt_path, phase_receipt_sha256, train_proposal, config,
    )
    # This verifies pinned ResEnc source, pretrained-file SHA, overlap
    # provenance, and the currently configured phase receipt. It reads no
    # patient arrays or headers.
    try:
        validate_real_teacher_provenance(config)
    except ValueError as exc:
        phase_error = _phase_lock_error(exc)
        if phase_error is not None:
            raise phase_error from exc
        raise
    return ProfileInputs(
        config=config,
        config_path=source,
        config_sha256=sha256_file(source),
        train_manifest_path=train_path,
        train_manifest_sha256=train_sha,
        development_manifest_path=dev_path,
        development_manifest_sha256=dev_sha,
        largest_l=largest_l,
        largest_u=largest_u,
        largest_train=largest_train,
        stage_artifacts=dict(stage_artifacts),
        labeled_patient_ids=tuple(sorted(item["id"] for item in train_manifest["labeled"])),
        unlabeled_patient_ids=tuple(sorted(item["id"] for item in train_manifest["unlabeled"])),
        train_geometry_evidence={
            "source_manifest": {"path": str(source_train_path), "sha256": source_train_sha},
            "corrected_manifest": {"path": str(train_path), "sha256": train_sha},
            "repair_report": {"path": str(train_report_path), "sha256": train_report_sha},
            "proposal_status": train_proposal["status"],
            "proposal_manifest_lineage": lineage,
            "geometry_policy": train_geometry_policy,
        },
        development_geometry_evidence={
            "source_manifest": {"path": str(source_dev_path), "sha256": source_dev_sha},
            "corrected_manifest": {"path": str(dev_path), "sha256": dev_sha},
            "repair_report": {"path": str(dev_report_path), "sha256": dev_report_sha},
            **dev_validation,
        },
        phase_evidence=phase_summary,
        geometry_policy=train_geometry_policy,
        exploratory_quality_waiver=waiver_policy,
    )


class UpdateMeasurement:
    """Synchronize CUDA and measure complete engine updates through optimizer.step."""

    def __init__(
        self, device: str, *, cuda_api: Any, clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.device = device
        self.cuda_api = cuda_api
        self.clock = clock
        self.records: list[dict[str, Any]] = []
        self.active_started: float | None = None
        self.failed_update: dict[str, Any] | None = None
        self.selection_trace: list[dict[str, str | None]] = []
        self.optimizer_parameter_dtypes: set[str] = set()

    def observe_optimizer_dtypes(self, optimizer: Any) -> None:
        for group in getattr(optimizer, "param_groups", ()):
            for parameter in group.get("params", ()):
                dtype = getattr(parameter, "dtype", None)
                if dtype is not None:
                    self.optimizer_parameter_dtypes.add(str(dtype))

    def begin_update(self) -> None:
        if self.active_started is not None:
            raise RuntimeError("Q3D profile observed overlapping optimizer updates")
        self.cuda_api.synchronize(self.device)
        self.cuda_api.reset_peak_memory_stats(self.device)
        self.active_started = self.clock()

    def _peak_memory(self) -> tuple[int | None, int | None]:
        try:
            allocated = int(self.cuda_api.max_memory_allocated(self.device))
            reserved = int(self.cuda_api.max_memory_reserved(self.device))
            return allocated, reserved
        except Exception:
            return None, None

    def finish_update(self) -> None:
        if self.active_started is None:
            raise RuntimeError("Q3D optimizer.step occurred outside a measured profile update")
        self.cuda_api.synchronize(self.device)
        elapsed = max(0.0, self.clock() - self.active_started)
        allocated, reserved = self._peak_memory()
        self.records.append({
            "update": len(self.records) + 1,
            "wall_seconds": elapsed,
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
        })
        self.active_started = None

    def fail_active_update(self) -> None:
        if self.active_started is None:
            return
        try:
            self.cuda_api.synchronize(self.device)
        except Exception:
            pass
        elapsed = max(0.0, self.clock() - self.active_started)
        allocated, reserved = self._peak_memory()
        self.failed_update = {
            "update": len(self.records) + 1,
            "seconds_until_failure": elapsed,
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
        }
        self.active_started = None

    def summary(self) -> dict[str, Any]:
        times = [float(item["wall_seconds"]) for item in self.records]
        allocated = [item["peak_allocated_bytes"] for item in self.records if item["peak_allocated_bytes"] is not None]
        reserved = [item["peak_reserved_bytes"] for item in self.records if item["peak_reserved_bytes"] is not None]
        return {
            "measured_updates": len(self.records),
            "updates": list(self.records),
            "mean_seconds_per_update": statistics.fmean(times) if times else None,
            "median_seconds_per_update": statistics.median(times) if times else None,
            "peak_allocated_bytes": max(allocated) if allocated else None,
            "peak_reserved_bytes": max(reserved) if reserved else None,
            "failed_update": self.failed_update,
            "selection_trace": list(self.selection_trace),
            "optimizer_parameter_dtypes": sorted(self.optimizer_parameter_dtypes),
        }


def _engine_profile_hooks(
    engine: Any,
    stage: str,
    measurement: UpdateMeasurement,
    largest_l_id: str,
    largest_u_id: str,
    optimizer_type: Any,
):
    """Install narrow stage-specific selectors and optimizer timing wrappers."""
    from contextlib import contextmanager

    @contextmanager
    def installed():
        original_step = optimizer_type.step
        original_teacher_choice = engine._teacher_frame_choice
        original_sampler_factory = engine._make_q3d_sampler

        def timed_step(optimizer, *args, **kwargs):
            measurement.observe_optimizer_dtypes(optimizer)
            value = original_step(optimizer, *args, **kwargs)
            measurement.finish_update()
            return value

        optimizer_type.step = timed_step

        if stage == "T":
            def largest_l_teacher_frame(data, rng):
                measurement.begin_update()
                # Consume the exact engine RNG selection on every update.
                # Only update 1 is replaced, leaving the subsequent 19 picks
                # identical to the production sampler's stream.
                original_key = original_teacher_choice(data, rng)
                if measurement.selection_trace:
                    patient = str(data.teacher_patient_for_key(original_key))
                    measurement.selection_trace.append({
                        "labeled_patient": patient, "unlabeled_patient": None,
                    })
                    return original_key
                candidates = [
                    key for key in data.teacher_frame_keys()
                    if data.teacher_patient_for_key(key) == largest_l_id
                ]
                if not candidates:
                    raise ProfilePrerequisiteError(
                        f"largest L patient {largest_l_id!r} has no T-eligible annotated frame"
                    )
                key = sorted(candidates)[0]
                measurement.selection_trace.append({
                    "labeled_patient": largest_l_id, "unlabeled_patient": None,
                })
                return key

            engine._teacher_frame_choice = largest_l_teacher_frame

        elif stage == "J":
            def largest_stream_sampler(data, seed, state, *, require_unlabeled=True):
                sampler = original_sampler_factory(
                    data, seed, state, require_unlabeled=require_unlabeled,
                )
                if not require_unlabeled:
                    raise ProfilePrerequisiteError("V1 J must retain its paired image-only U stream")
                if sampler.updates != 0 or sampler._unlabeled_cursor != 0:
                    raise ProfilePrerequisiteError("GPU profile requires a fresh J sampler, not a resumed run")
                u_patients = list(sampler._unlabeled_permutation)
                if largest_u_id not in u_patients:
                    raise ProfilePrerequisiteError(
                        f"largest U patient {largest_u_id!r} is absent from the image-only J stream"
                    )
                u_start = u_patients.index(largest_u_id)
                sampler._unlabeled_permutation = u_patients[u_start:] + u_patients[:u_start]
                labeled_keys = [
                    key for key in data.labeled_frame_keys()
                    if data.labeled_patient_for_key(key) == largest_l_id
                ]
                if not labeled_keys:
                    raise ProfilePrerequisiteError(
                        f"largest L patient {largest_l_id!r} has no J-eligible annotated frame"
                    )
                selected_l_key = sorted(labeled_keys)[0]
                original_sample_update = sampler.sample_update

                def sample_largest_stream_update():
                    measurement.begin_update()
                    record = original_sample_update()
                    if not measurement.selection_trace and record.get("unlabeled_patient") != largest_u_id:
                        raise ProfilePrerequisiteError(
                            "largest U patient was not first in the measured J image-only stream"
                        )
                    if not measurement.selection_trace:
                        # The original random L draw is consumed first; only
                        # the first measured update uses the largest L case.
                        record["labeled_key"] = selected_l_key
                        record["labeled_patient"] = largest_l_id
                    measurement.selection_trace.append({
                        "labeled_patient": str(record["labeled_patient"]),
                        "unlabeled_patient": str(record["unlabeled_patient"]),
                    })
                    return record

                sampler.sample_update = sample_largest_stream_update
                return sampler

            engine._make_q3d_sampler = largest_stream_sampler
        else:
            optimizer_type.step = original_step
            raise ValueError(f"unsupported GPU profile stage {stage!r}")
        try:
            yield
        finally:
            optimizer_type.step = original_step
            engine._teacher_frame_choice = original_teacher_choice
            engine._make_q3d_sampler = original_sampler_factory

    return installed()


def run_engine_stage(
    inputs: ProfileInputs,
    stage: str,
    output_dir: Path,
    measurement: UpdateMeasurement,
    *,
    device: str,
    engine_module: Any | None = None,
    optimizer_type: Any | None = None,
) -> Mapping[str, Any]:
    """Execute one real stage through the canonical Q3D engine/model/loss path."""
    import torch

    if engine_module is None:
        from . import engine as engine_module
    if optimizer_type is None:
        optimizer_type = torch.optim.AdamW
    artifacts = None
    if stage == "J":
        artifacts = {
            key: value.engine_binding() for key, value in inputs.stage_artifacts.items()
        }
        if set(artifacts) != set(_REQUIRED_STAGE_ARTIFACTS):
            raise ProfilePrerequisiteError(
                "J profiling requires verified supplied T/Q/P artifacts; Q tokenizer gate and P cache are mandatory"
            )
    elif stage != "T":
        raise ValueError(f"unsupported GPU profile stage {stage!r}")
    waiver = inputs.exploratory_quality_waiver
    if stage == "J":
        _require_artifact_waivers(inputs.stage_artifacts, waiver)
    # Forward the reason only on explicit opt-in so the strict call is unchanged.
    waiver_kwargs = {"exploratory_quality_waiver": waiver["reason"]} if waiver is not None else {}
    with _engine_profile_hooks(
        engine_module, stage, measurement, inputs.largest_l.patient_id,
        inputs.largest_u.patient_id, optimizer_type,
    ):
        try:
            result = engine_module.train(
                inputs.config,
                output_dir,
                device=device,
                resume_from=None,
                smoke=False,
                update_limit=PROFILE_UPDATE_LIMIT,
                synthetic_gate_override=False,
                stage=stage,
                stage_artifacts=artifacts,
                **waiver_kwargs,
            )
        except Exception as exc:
            try:
                setattr(exc, "q3d_profile_metrics", _read_metric_records(output_dir / "metrics.jsonl"))
            except Exception:
                pass
            raise
    return {**dict(result), "profile_metrics": _read_metric_records(output_dir / "metrics.jsonl")}


def _read_metric_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Q3D metric log line {line_number} is invalid JSON") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _atomic_write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False, default=str) + "\n"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # link() publishes without overwriting a report created concurrently.
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _is_oom(exc: BaseException, torch_module: Any) -> bool:
    oom_type = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", ())
    return (isinstance(oom_type, type) and isinstance(exc, oom_type)) or "out of memory" in str(exc).lower()


def _stage_report(
    stage: str,
    measurement: UpdateMeasurement,
    *,
    status: str,
    failure: str | None = None,
    metric_records: list[dict[str, Any]] | None = None,
    measured_phase: str | None = None,
) -> dict[str, Any]:
    budget = PRODUCTION_STAGE_BUDGETS[stage]
    result = {
        "stage": stage,
        "status": status,
        "full_recipe_updates": budget,
        "requested_profile_updates": PROFILE_UPDATE_LIMIT,
        **measurement.summary(),
        "engine_metric_records": list(metric_records or []),
    }
    if measured_phase is not None:
        result["measured_phase"] = measured_phase
    if failure is not None:
        result["failure"] = failure
    return result


def measured_campaign_estimate(
    stage_results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    measured_stages: tuple[str, ...] = ("T", "J"),
) -> dict[str, Any]:
    """Project measured portions only; never invent Q/P/T-b/evaluation timings."""
    measured_stages = _validate_measured_stages(measured_stages)
    t_result = stage_results.get("T", {})
    j_result = stage_results.get("J", {})
    t_updates = t_result.get("updates", [])
    j_updates = j_result.get("updates", [])
    t_times = [float(item["wall_seconds"]) for item in t_updates]
    j_times = [float(item["wall_seconds"]) for item in j_updates]
    t_mean = statistics.fmean(t_times) if t_times else None
    j_mean = statistics.fmean(j_times) if j_times else None
    training = config.get("training", {})
    losses = config.get("losses", {})
    t_a_budget = int(math.ceil(
        PRODUCTION_STAGE_BUDGETS["T"] * float(training.get("teacher_warmup_fraction", 0.0))
    ))
    j_lr_warmup_budget = int(math.ceil(
        PRODUCTION_STAGE_BUDGETS["J"] * float(training.get("warmup_fraction", 0.0))
    ))
    j_codekd_ramp_budget = int(math.ceil(
        PRODUCTION_STAGE_BUDGETS["J"] * float(losses.get("codekd_ramp_fraction", 0.0))
    ))
    t_a_projection = t_mean * t_a_budget if t_mean is not None else None
    j_projection = j_mean * PRODUCTION_STAGE_BUDGETS["J"] if j_mean is not None else None
    observed = sum(t_times) + sum(j_times)
    projected = sum(value for value in (t_a_projection, j_projection) if value is not None)
    return {
        "status": "PARTIAL_PROJECTION" if t_times or j_times else "NOT_RUN",
        "observed_profile_update_seconds": observed,
        "measured_portion_projections_seconds": {
            f"T_a_warmup_{t_a_budget}_updates_from_measured_T_a_rate": t_a_projection,
            "J_v1_8000_updates_from_first_20_J_rate": j_projection,
            "partial_projected_subtotal_seconds": projected if t_times or j_times else None,
        },
        "measured_rates_seconds_per_update": {"T_a": t_mean, "J_initial_ramp": j_mean},
        "coverage": {
            "selected_measured_stages": list(measured_stages),
            "stage_status": {
                stage: stage_results.get(stage, {}).get("status", "NOT_RUN")
                for stage in ("T", "Q", "P", "J")
            },
            "T_a_full_budget_updates": t_a_budget,
            "T_a_measured_updates": len(t_times),
            "T_b_main_teacher_updates": PRODUCTION_STAGE_BUDGETS["T"] - t_a_budget,
            "T_b_main_teacher": "NOT_RUN",
            "J_lr_warmup_updates": j_lr_warmup_budget,
            "J_codekd_ramp_updates": j_codekd_ramp_budget,
            "J_measured_phase": "learning_rate_warmup_and_codekd_ramp",
            "J_measured_updates": len(j_times),
            "Q": "NOT_RUN",
            "P": "NOT_RUN",
            "evaluation": "NOT_RUN",
        },
        "full_campaign_seconds": None,
        "interpretation": (
            "Partial measured-rate projection only, not a mathematical lower bound. Only selected measured "
            "portions are projected; unmeasured T-b, Q, P, and evaluation remain NOT_RUN, so no full campaign "
            "duration is reported."
        ),
    }


def _gpu_metadata(torch_module: Any, device: str) -> dict[str, Any]:
    selected = torch_module.device(device)
    properties = torch_module.cuda.get_device_properties(selected)
    return {
        "device": str(selected),
        "model": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "autocast_dtype": str(torch_module.bfloat16).replace("torch.", ""),
        "optimizer_parameter_dtype_source": "observed from live optimizer parameter groups during each measured update",
        "timing_method": "perf_counter; synchronize before update and after optimizer.step",
    }


def _run_profile_core(
    *,
    config_path: str | Path,
    artifact_paths: list[str],
    artifact_hashes: list[str],
    report_path: str | Path,
    scratch_root: str | Path,
    device: str = "cuda:0",
    repo_root: str | Path | None = None,
    inputs: ProfileInputs | None = None,
    stage_runner: Callable[..., Mapping[str, Any]] | None = None,
    cuda_api: Any | None = None,
    torch_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
    evidence_inputs: Mapping[str, Any] | None = None,
    software_only: bool = False,
    measured_stages: tuple[str, ...] = ("T", "J"),
    exploratory_quality_waiver: str | None = None,
) -> dict[str, Any]:
    """Shared execution core; software test mode cannot create GPU evidence."""
    import torch

    measured_stages = _validate_measured_stages(measured_stages)
    waiver_policy = _waiver_policy(exploratory_quality_waiver)
    if not device.startswith("cuda"):
        raise ProfilePrerequisiteError("Q3D Phase D profiling requires a CUDA device")
    torch_module = torch_module or torch
    cuda_api = cuda_api or torch_module.cuda
    report = Path(report_path).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    repository = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    if _within(report, repository):
        raise ProfilePrerequisiteError("profile report must be outside the repository and main run paths")
    if _within(scratch, repository):
        raise ProfilePrerequisiteError("profile scratch root must be outside the repository and main run paths")
    if report.exists():
        raise ProfilePrerequisiteError(f"refusing to overwrite existing profile report: {report}")

    report_doc: dict[str, Any] = {
        "schema": "self_audit.candidate_q3d.phase_d_profile.v1",
        "status": "NOT_RUN",
        "evidence_status": "SOFTWARE_ONLY" if software_only else "NOT_RUN",
        "execution_mode": "SOFTWARE_ONLY_TEST_HARNESS" if software_only else "REAL_CUDA_PROFILE",
        "selected_stages": list(measured_stages),
        "profile_update_limit": PROFILE_UPDATE_LIMIT,
        "production_stage_budgets": dict(PRODUCTION_STAGE_BUDGETS),
        "profile_device": device,
        "stage_results": {
            stage: {
                "stage": stage,
                "status": "NOT_RUN",
                "reason": (
                    "stage was selected but profiling did not start"
                    if stage in measured_stages
                    else "stage was not selected for this profile"
                    if stage in {"T", "J"}
                    else "the profile tool does not execute this production stage"
                ),
            }
            for stage in ("T", "Q", "P", "J")
        },
        "temporary_output_cleanup_policy": "persist report, then remove only this invocation's owned temporary tree",
    }
    if waiver_policy is not None:
        # evidence_status keeps its normal meaning; the explicit marker says
        # gate measurements stayed NO_GO and the profile is exploratory.
        report_doc["exploratory"] = True
        report_doc["exploratory_quality_waiver"] = waiver_policy
    temporary_root: Path | None = None
    try:
        if inputs is None:
            if "J" in measured_stages:
                parsed = parse_stage_artifact_bindings(
                    artifact_paths, artifact_hashes,
                    exploratory_quality_waiver=exploratory_quality_waiver,
                )
            else:
                if artifact_paths or artifact_hashes:
                    raise ProfilePrerequisiteError(
                        "T-only profiling does not accept unused T/Q/P artifact bindings"
                    )
                parsed = {}
            inputs = load_profile_inputs(
                config_path, parsed, measured_stages=measured_stages, **dict(evidence_inputs or {}),
                exploratory_quality_waiver=exploratory_quality_waiver,
            )
        elif "J" in measured_stages and set(inputs.stage_artifacts) != set(_REQUIRED_STAGE_ARTIFACTS):
            raise ProfilePrerequisiteError(
                "J profiling requires exact verified T/Q/P stage artifact bindings; Q gate and P cache are mandatory"
            )
        elif "J" not in measured_stages and inputs.stage_artifacts:
            raise ProfilePrerequisiteError("T-only profiling does not accept unused T/Q/P artifact bindings")
        if waiver_policy is not None and inputs.exploratory_quality_waiver is None:
            inputs = replace(inputs, exploratory_quality_waiver=waiver_policy)
        elif waiver_policy is None and inputs.exploratory_quality_waiver is not None:
            raise ProfilePrerequisiteError(
                f"profile inputs carry an exploratory waiver; pass {EXPLORATORY_WAIVER_FLAG} REASON explicitly"
            )
        elif inputs.exploratory_quality_waiver != waiver_policy:
            raise ProfilePrerequisiteError("profile inputs carry a different exploratory waiver REASON")
        _require_artifact_waivers(inputs.stage_artifacts, inputs.exploratory_quality_waiver)
        report_doc.update({
            "config_path": str(inputs.config_path),
            "config_sha256": inputs.config_sha256,
            "train_manifest": {
                "path": str(inputs.train_manifest_path),
                "sha256": inputs.train_manifest_sha256,
            },
            "development_manifest": {
                "path": str(inputs.development_manifest_path),
                "sha256": inputs.development_manifest_sha256,
            },
            "stage_artifact_inputs": {
                stage: artifact.as_dict() for stage, artifact in inputs.stage_artifacts.items()
            },
            "train_geometry_evidence": inputs.train_geometry_evidence,
            "development_geometry_evidence": inputs.development_geometry_evidence,
            "geometry_policy": inputs.geometry_policy,
            "phase_evidence": inputs.phase_evidence,
            "selected_patients": {
                "largest_train_by_native_voxel_count": inputs.largest_train.as_dict(),
                "largest_L_for_T_and_J": inputs.largest_l.as_dict(),
                "largest_U_for_J_image_only": inputs.largest_u.as_dict(),
                "T_U_supervision": "forbidden; protocol A teacher samples only L",
            },
        })
        if not cuda_api.is_available():
            raise ProfilePrerequisiteError("CUDA is unavailable; GPU profile NOT_RUN")
        if software_only:
            report_doc["evidence_status"] = "SOFTWARE_ONLY"
        else:
            report_doc["gpu"] = _gpu_metadata(torch_module, device)
            report_doc["evidence_status"] = "GPU_PROFILE"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(tempfile.mkdtemp(prefix="q3d-profile-", dir=scratch)).resolve()
        if temporary_root.parent != scratch:
            raise ProfilePrerequisiteError("temporary profile output escaped the declared scratch root")
        if _within(report, temporary_root):
            raise ProfilePrerequisiteError("profile report must be outside its temporary outputs")
        if stage_runner is None:
            stage_runner = run_engine_stage

        for stage in measured_stages:
            measurement = UpdateMeasurement(device, cuda_api=cuda_api, clock=clock)
            stage_output = temporary_root / stage
            metric_records: list[dict[str, Any]] = []
            measured_phase: str | None = None
            try:
                result = stage_runner(
                    inputs, stage, stage_output, measurement,
                    device=device,
                )
                raw_metrics = result.get("profile_metrics", [])
                metric_records = raw_metrics if isinstance(raw_metrics, list) else []
                if measurement.active_started is not None:
                    measurement.fail_active_update()
                    raise RuntimeError("engine returned with an unfinished optimizer update")
                if result.get("runtime_synthetic") is not False:
                    raise ProfilePrerequisiteError("engine reported synthetic runtime during profiling")
                result_marker = exploratory_marker(result)
                if result_marker is not None and waiver_policy is None:
                    raise ProfilePrerequisiteError(
                        "engine reported an exploratory result without an explicit waiver"
                    )
                if result_marker is not None and result_marker["policy_sha256"] not in (
                    None, waiver_policy_sha256(waiver_policy),
                ):
                    raise ProfilePrerequisiteError("engine reported a different exploratory waiver REASON")
                if len(measurement.records) != PROFILE_UPDATE_LIMIT:
                    raise RuntimeError(
                        f"engine completed {len(measurement.records)} profiled optimizer updates; expected {PROFILE_UPDATE_LIMIT}"
                    )
                if not software_only and not measurement.optimizer_parameter_dtypes:
                    raise RuntimeError("profile did not observe live optimizer parameter dtypes")
                if stage == "T":
                    if (len(measurement.selection_trace) != PROFILE_UPDATE_LIMIT
                            or measurement.selection_trace[0].get("labeled_patient") != inputs.largest_l.patient_id
                            or any(row.get("unlabeled_patient") is not None for row in measurement.selection_trace)
                            or any(row.get("labeled_patient") not in inputs.labeled_patient_ids
                                   for row in measurement.selection_trace)):
                        raise RuntimeError("T profile did not select largest L first and remain L-only")
                    training = inputs.config.get("training", {})
                    t_a_budget = int(math.ceil(
                        PRODUCTION_STAGE_BUDGETS["T"] * float(training.get("teacher_warmup_fraction", 0.0))
                    ))
                    t_metrics = [
                        row for row in metric_records
                        if row.get("stage") == "T" and row.get("event") is None
                    ]
                    expected_updates = list(range(1, PROFILE_UPDATE_LIMIT + 1))
                    if ([row.get("update") for row in t_metrics] != expected_updates
                            or any(row.get("phase") != "T-a" for row in t_metrics)
                            or PROFILE_UPDATE_LIMIT > t_a_budget):
                        raise RuntimeError(
                            "T metric phase does not verify the measured updates as T-a warmup"
                        )
                    measured_phase = "T-a"
                else:
                    if (len(measurement.selection_trace) != PROFILE_UPDATE_LIMIT
                            or measurement.selection_trace[0].get("labeled_patient") != inputs.largest_l.patient_id
                            or measurement.selection_trace[0].get("unlabeled_patient") != inputs.largest_u.patient_id
                            or any(row.get("labeled_patient") not in inputs.labeled_patient_ids
                                   for row in measurement.selection_trace)
                            or any(row.get("unlabeled_patient") not in inputs.unlabeled_patient_ids
                                   for row in measurement.selection_trace)):
                        raise RuntimeError(
                            "J profile must use largest L/U first, then remain within the declared L/U pools"
                        )
                report_doc["stage_results"][stage] = _stage_report(
                    stage, measurement,
                    status="SOFTWARE_ONLY_COMPLETED" if software_only else "COMPLETED",
                    metric_records=metric_records, measured_phase=measured_phase,
                )
                if result_marker is not None:
                    report_doc["stage_results"][stage]["exploratory"] = True
                    report_doc["stage_results"][stage]["exploratory_quality_waiver"] = (
                        result_marker["policy"] or waiver_policy
                    )
            except Exception as exc:
                measurement.fail_active_update()
                exc_metrics = getattr(exc, "q3d_profile_metrics", None)
                if isinstance(exc_metrics, list):
                    metric_records = exc_metrics
                failure_status = "OOM" if _is_oom(exc, torch_module) else "FAILED"
                report_doc["stage_results"][stage] = _stage_report(
                    stage, measurement,
                    status=f"SOFTWARE_ONLY_{failure_status}" if software_only else failure_status,
                    failure=f"{type(exc).__name__}: {exc}", metric_records=metric_records,
                )
                report_doc["failure"] = {
                    "stage": stage,
                    "status": failure_status,
                    "message": f"{type(exc).__name__}: {exc}",
                    "retry_attempted": False,
                    "architecture_or_batch_changes": "none",
                }
                break

        complete = all(
            report_doc["stage_results"].get(stage, {}).get("status")
            == ("SOFTWARE_ONLY_COMPLETED" if software_only else "COMPLETED")
            for stage in measured_stages
        )
        if software_only:
            report_doc["status"] = "SOFTWARE_ONLY" if complete else "SOFTWARE_ONLY_FAILURE"
        else:
            report_doc["status"] = "COMPLETED" if complete else "FAILED"
        report_doc["campaign_estimate"] = measured_campaign_estimate(
            report_doc["stage_results"], inputs.config, measured_stages,
        )
    except Exception as exc:
        if software_only:
            report_doc["status"] = "SOFTWARE_ONLY_FAILURE"
        else:
            report_doc["status"] = "NOT_RUN" if not report_doc["stage_results"] else "FAILED"
        report_doc["failure"] = {
            "stage": "preflight",
            "status": "PREREQUISITE_FAILED",
            "message": f"{type(exc).__name__}: {exc}",
            "retry_attempted": False,
        }
        report_doc["campaign_estimate"] = measured_campaign_estimate(
            report_doc["stage_results"], inputs.config if inputs is not None else {}, measured_stages,
        )
    finally:
        # The durable report is published before only this invocation's fresh
        # temporary output tree is removed.
        try:
            _atomic_write_report(report, report_doc)
        finally:
            if temporary_root is not None:
                shutil.rmtree(temporary_root)
    return report_doc


def run_profile(
    *,
    config_path: str | Path,
    artifact_paths: list[str],
    artifact_hashes: list[str],
    source_train_manifest_path: str | Path,
    source_train_manifest_sha256: str,
    train_repair_report_path: str | Path,
    train_repair_report_sha256: str,
    source_development_manifest_path: str | Path,
    source_development_manifest_sha256: str,
    development_repair_report_path: str | Path,
    development_repair_report_sha256: str,
    phase_receipt_path: str | Path,
    phase_receipt_sha256: str,
    report_path: str | Path,
    scratch_root: str | Path,
    device: str = "cuda:0",
    allow_real_profile: bool = False,
    measured_stages: tuple[str, ...] = ("T", "J"),
    exploratory_quality_waiver: str | None = None,
) -> dict[str, Any]:
    """Run the canonical engine on real data; all software dependencies are fixed internally."""
    measured_stages = _validate_measured_stages(measured_stages)
    if not allow_real_profile:
        raise ProfilePrerequisiteError("pass --allow-real-profile to execute this CUDA-only real-data command")
    if not device.startswith("cuda"):
        raise ProfilePrerequisiteError("Q3D Phase D profiling requires a CUDA device")
    evidence_inputs = {
        "source_train_manifest_path": source_train_manifest_path,
        "source_train_manifest_sha256": source_train_manifest_sha256,
        "train_repair_report_path": train_repair_report_path,
        "train_repair_report_sha256": train_repair_report_sha256,
        "source_development_manifest_path": source_development_manifest_path,
        "source_development_manifest_sha256": source_development_manifest_sha256,
        "development_repair_report_path": development_repair_report_path,
        "development_repair_report_sha256": development_repair_report_sha256,
        "phase_receipt_path": phase_receipt_path,
        "phase_receipt_sha256": phase_receipt_sha256,
    }
    return _run_profile_core(
        config_path=config_path,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        report_path=report_path,
        scratch_root=scratch_root,
        device=device,
        stage_runner=run_engine_stage,
        evidence_inputs=evidence_inputs,
        software_only=False,
        measured_stages=measured_stages,
        exploratory_quality_waiver=exploratory_quality_waiver,
    )


def run_profile_software_test(
    *,
    inputs: ProfileInputs,
    stage_runner: Callable[..., Mapping[str, Any]],
    report_path: str | Path,
    scratch_root: str | Path,
    cuda_api: Any,
    torch_module: Any,
    device: str = "cuda:0",
    repo_root: str | Path | None = None,
    clock: Callable[[], float] = time.perf_counter,
    measured_stages: tuple[str, ...] = ("T", "J"),
    exploratory_quality_waiver: str | None = None,
) -> dict[str, Any]:
    """Explicit SOFTWARE_ONLY harness for synthetic unit tests; never emits GPU_PROFILE."""
    measured_stages = _validate_measured_stages(measured_stages)
    if stage_runner is run_engine_stage:
        raise ProfilePrerequisiteError("SOFTWARE_ONLY test harness refuses the canonical real engine runner")
    return _run_profile_core(
        config_path="<software-test-fixture>",
        artifact_paths=[],
        artifact_hashes=[],
        report_path=report_path,
        scratch_root=scratch_root,
        device=device,
        repo_root=repo_root,
        inputs=inputs,
        stage_runner=stage_runner,
        cuda_api=cuda_api,
        torch_module=torch_module,
        clock=clock,
        software_only=True,
        measured_stages=measured_stages,
        exploratory_quality_waiver=exploratory_quality_waiver,
    )


__all__ = [
    "PROFILE_UPDATE_LIMIT",
    "PRODUCTION_STAGE_BUDGETS",
    "PatientVolume",
    "ProfileInputs",
    "ProfilePrerequisiteError",
    "StageArtifact",
    "UpdateMeasurement",
    "largest_train_patients",
    "load_profile_inputs",
    "measured_campaign_estimate",
    "parse_stage_artifact_bindings",
    "run_engine_stage",
    "run_profile",
    "sha256_file",
]
