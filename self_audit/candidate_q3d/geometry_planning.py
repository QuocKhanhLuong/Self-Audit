"""Native-geometry topology proposal helpers for Q3D.

The topology rule below is a small, checked port of nnUNet's pinned
``get_pool_and_conv_props`` implementation.  It is intentionally called on the
componentwise median TRAIN spacing and unchanged componentwise median native
shape.  It does not implement nnU-Net preprocessing, target spacing, cropping,
or resampling.
"""

from __future__ import annotations

import math
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


NNUNET_REPOSITORY = "MIC-DKFZ/nnUNet"
NNUNET_COMMIT = "c04f7ec8bbffaf3ebc3b26aa2c89042e8a8e368b"
NNUNET_TOPOLOGY_PATH = (
    "nnunetv2/experiment_planning/experiment_planners/network_topology.py"
)
NNUNET_TOPOLOGY_GIT_BLOB_SHA1 = "6922f7b5dbbe5aca552acb6f0439c9b65dbf93bf"
NNUNET_TOPOLOGY_SHA256 = "2140c9c6b20273bbf3052a72459954543d93790cc5269e0d8fbaddcfbd30aa8f"
NNUNET_RESENC_PLANNER_PATH = (
    "nnunetv2/experiment_planning/experiment_planners/residual_unets/"
    "residual_encoder_unet_planners.py"
)
NNUNET_RESENC_PLANNER_GIT_BLOB_SHA1 = "34eb06535b329f0b79eaa42e8b01215e6d8387c4"
NNUNET_RESENC_PLANNER_SHA256 = "f2de43f8bc21a3bc8d7527380ff18015404a7a95561f7762a401cc87f740ddff"
NNUNET_MIN_FEATURE_MAP_SIZE = 4
NNUNET_MAX_NUMPOOL_ARGUMENT = 999999

CANONICAL_PROBE_CONTRACT: dict[str, Any] = {
    "historical_probe": "reports/q3d-resenc-phase0-20261112/erf_probe.py",
    "historical_probe_sha256": "f074df03edd2828e6a63d7416fa7421b5ea2ea7f17196b8c02e71208b93d2636",
    "samples": 16,
    "seed_start": 42,
    "seed_end_inclusive": 57,
    "shape_zyx": [33, 97, 97],
    "stage_index": 3,
    "channel_aggregation": "signed_mean_all_stage_channels",
    "padding_mode": "replicate",
    "decision_threshold_voxels": 2.0,
    "threads": 4,
    "device": "cpu",
    "patient_data": "NOT_USED",
}


def _numeric_triplet(values: Sequence[Any], *, name: str, integer: bool) -> tuple[float, ...] | tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly three ZYX values")
    parsed: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} values must be numeric") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name} values must be finite and positive")
        if integer and not number.is_integer():
            raise ValueError(f"{name} values must be integer voxel counts")
        parsed.append(number)
    if integer:
        return tuple(int(value) for value in parsed)
    return tuple(parsed)


def get_pool_and_conv_props(
    spacing: Sequence[float],
    patch_size: Sequence[float],
    min_feature_map_size: int = NNUNET_MIN_FEATURE_MAP_SIZE,
    max_numpool: int = NNUNET_MAX_NUMPOOL_ARGUMENT,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Return topology from the pinned nnU-Net helper for native metadata.

    The returned five-tuple matches the pinned source function's structure:
    pool counts, pool kernels, convolution kernels, divisibility-padded
    diagnostic patch, and per-axis divisibility.  ``patch_size`` is passed into
    the algorithm without resampling or target-spacing transformation.  For a
    median shape with half-voxel values, its values remain fractional until
    the pinned padding helper's final integer cast, matching upstream.
    """

    try:
        spacing_dim = len(spacing)
        patch_dim = len(patch_size)
    except TypeError as exc:
        raise ValueError("spacing and patch_size must be sized sequences") from exc
    if spacing_dim == 0 or spacing_dim != patch_dim:
        raise ValueError("spacing and patch_size must have the same nonzero dimensionality")
    if isinstance(min_feature_map_size, bool) or int(min_feature_map_size) != min_feature_map_size:
        raise ValueError("min_feature_map_size must be an integer")
    if int(min_feature_map_size) < 1:
        raise ValueError("min_feature_map_size must be positive")
    if isinstance(max_numpool, bool) or int(max_numpool) != max_numpool or int(max_numpool) < 0:
        raise ValueError("max_numpool must be a nonnegative integer")

    current_spacing = [float(value) for value in spacing]
    current_size = [float(value) for value in patch_size]
    if any(not math.isfinite(value) or value <= 0 for value in current_spacing):
        raise ValueError("spacing values must be finite and positive")
    if any(not math.isfinite(value) or value <= 0 for value in current_size):
        raise ValueError("patch_size values must be finite and positive")

    dim = len(current_spacing)
    pool_op_kernel_sizes: list[list[int]] = [[1] * dim]
    conv_kernel_sizes: list[list[int]] = []
    num_pool_per_axis = [0] * dim
    kernel_size = [1] * dim

    while True:
        # Match the pinned source: first enforce the minimum bottleneck edge,
        # then retain axes whose spacing is within a factor of two.
        valid_axes_for_pool = [
            index
            for index in range(dim)
            if current_size[index] >= 2 * int(min_feature_map_size)
        ]
        if not valid_axes_for_pool:
            break

        min_spacing_of_valid = min(current_spacing[index] for index in valid_axes_for_pool)
        valid_axes_for_pool = [
            index
            for index in valid_axes_for_pool
            if current_spacing[index] / min_spacing_of_valid < 2
        ]
        valid_axes_for_pool = [
            index
            for index in valid_axes_for_pool
            if num_pool_per_axis[index] < int(max_numpool)
        ]

        if len(valid_axes_for_pool) == 1:
            if current_size[valid_axes_for_pool[0]] < 3 * int(min_feature_map_size):
                break
        if not valid_axes_for_pool:
            break

        for axis in range(dim):
            if kernel_size[axis] != 3 and current_spacing[axis] / min(current_spacing) < 2:
                kernel_size[axis] = 3

        other_axes = [axis for axis in range(dim) if axis not in valid_axes_for_pool]
        pool_kernel_sizes = [0] * dim
        for axis in valid_axes_for_pool:
            pool_kernel_sizes[axis] = 2
            num_pool_per_axis[axis] += 1
            current_spacing[axis] *= 2
            current_size[axis] = math.ceil(current_size[axis] / 2)
        for axis in other_axes:
            pool_kernel_sizes[axis] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(list(kernel_size))

    divisibility = [2**count for count in num_pool_per_axis]
    # Equivalent to the pinned source's pad_shape: round each axis upward to a
    # multiple, preserving an already divisible value, then cast to int.
    padded_patch = []
    for size, multiple in zip(patch_size, divisibility):
        value = float(size)
        rounded = value + multiple - value % multiple
        if value % multiple == 0:
            rounded -= multiple
        padded_patch.append(int(rounded))

    conv_kernel_sizes.append([3] * dim)
    return (
        tuple(num_pool_per_axis),
        tuple(tuple(kernel) for kernel in pool_op_kernel_sizes),
        tuple(tuple(kernel) for kernel in conv_kernel_sizes),
        tuple(padded_patch),
        tuple(divisibility),
    )


def _componentwise_median(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not rows:
        raise ValueError("at least one TRAIN geometry row is required")
    width = len(rows[0])
    if width != 3 or any(len(row) != width for row in rows):
        raise ValueError("TRAIN geometry rows must be three-dimensional ZYX triplets")
    return tuple(float(statistics.median(row[axis] for row in rows)) for axis in range(width))


def summarize_train_geometry(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only explicit native TRAIN geometry records.

    No image, mask, NIfTI header, or metadata outside these supplied records is
    accessed by this function.
    """

    shapes: list[tuple[int, ...]] = []
    spacings: list[tuple[float, ...]] = []
    for index, record in enumerate(rows):
        if not isinstance(record, dict):
            raise ValueError(f"TRAIN geometry row {index} must be an object")
        shapes.append(_numeric_triplet(record.get("shape_zyx"), name="shape_zyx", integer=True))
        spacings.append(_numeric_triplet(record.get("spacing_zyx_mm"), name="spacing_zyx_mm", integer=False))

    median_shape = _componentwise_median(shapes)
    median_spacing = _componentwise_median(spacings)
    per_patient_anisotropy = [max(spacing) / min(spacing) for spacing in spacings]
    per_patient_z_to_inplane = [
        spacing[0] / math.sqrt(spacing[1] * spacing[2]) for spacing in spacings
    ]
    topology = get_pool_and_conv_props(
        median_spacing,
        median_shape,
        min_feature_map_size=NNUNET_MIN_FEATURE_MAP_SIZE,
        max_numpool=NNUNET_MAX_NUMPOOL_ARGUMENT,
    )
    pool_count, pool_kernels, conv_kernels, padded_shape, divisibility = topology
    return {
        "train_patient_count": len(shapes),
        "median_spacing_zyx_mm": list(median_spacing),
        "median_z_spacing_mm": median_spacing[0],
        "median_native_shape_zyx_voxels": list(median_shape),
        "anisotropy": {
            "definition": "median across TRAIN patients of max(spacing_zyx_mm) / min(spacing_zyx_mm); each ratio is computed before taking the median",
            "median_per_patient_max_over_min": float(statistics.median(per_patient_anisotropy)),
            "definition_z_to_inplane": "median across TRAIN patients of spacing_z / sqrt(spacing_y * spacing_x)",
            "median_per_patient_z_over_geometric_mean_yx": float(statistics.median(per_patient_z_to_inplane)),
            "ratio_of_componentwise_median_spacing_max_over_min": max(median_spacing) / min(median_spacing),
        },
        "planner_inputs": {
            "spacing_zyx_mm": list(median_spacing),
            "patch_size_zyx_voxels_unchanged": list(median_shape),
            "min_feature_map_size": NNUNET_MIN_FEATURE_MAP_SIZE,
            "max_numpool_argument": NNUNET_MAX_NUMPOOL_ARGUMENT,
            "resampling": "none",
            "target_spacing": "none",
            "planner_conv_kernels_applied_to_teacher": False,
            "teacher_kernel_contract": "retain ResEnc-L checkpoint 3x3x3 kernels; only planner-derived strides are proposed",
        },
        "planner_output": {
            "effective_pool_count_zyx": list(pool_count),
            "pool_op_kernel_sizes_zyx": [list(value) for value in pool_kernels],
            "planner_conv_kernel_sizes_diagnostic_only_zyx": [list(value) for value in conv_kernels],
            "teacher_kernel_sizes_preserved_zyx": [[3, 3, 3] for _ in pool_kernels],
            "shape_must_be_divisible_by_zyx": list(divisibility),
            "padded_median_patch_shape_diagnostic_zyx": list(padded_shape),
            "stage_count": len(pool_kernels),
            "stage_strides_zyx": [list(value) for value in pool_kernels],
        },
    }


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without interpreting the file contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {resolved}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return resolved, value, digest


def _manifest_geometry_records(manifest: dict[str, Any], *, label: str) -> tuple[list[str], dict[str, Any]]:
    if manifest.get("schema") != "self_audit.data.v2" or manifest.get("role") != "training":
        raise ValueError(f"{label} must be a self_audit.data.v2 TRAIN manifest")
    if str(manifest.get("dataset", "")).lower() != "acdc":
        raise ValueError(f"{label} must declare dataset ACDC")
    if manifest.get("synthetic") is not False:
        raise ValueError(f"{label} must be a real TRAIN manifest")
    labeled, unlabeled = manifest.get("labeled"), manifest.get("unlabeled")
    if not isinstance(labeled, list) or not isinstance(unlabeled, list):
        raise ValueError(f"{label} must contain labeled and unlabeled TRAIN records")
    ids: list[str] = []
    for record in labeled + unlabeled:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            raise ValueError(f"{label} contains an invalid TRAIN patient record")
        ids.append(record["id"])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{label} must have a nonempty, unique TRAIN patient list")
    if set(ids) & set(map(str, manifest.get("evaluation_ids", []))):
        raise ValueError(f"{label} TRAIN IDs overlap evaluation IDs")
    section = manifest.get("geometry")
    if not isinstance(section, dict) or section.get("schema") != "self_audit.geometry_map.v1":
        raise ValueError(f"{label} geometry map schema is invalid")
    by_patient = section.get("by_patient")
    if not isinstance(by_patient, dict) or set(by_patient) != set(ids):
        raise ValueError(f"{label} geometry map must exactly cover TRAIN patient IDs")
    return ids, by_patient


def prepare_geometry_proposal(
    *,
    source_manifest_path: str | Path,
    repaired_manifest_path: str | Path,
    repair_report_path: str | Path,
) -> dict[str, Any]:
    """Validate shared repair lineage and derive a non-lock native proposal.

    The shared repair module owns receipt and geometry-record validation.  This
    function only reads JSON metadata, hashes source bytes, verifies complete
    TRAIN geometry coverage, and applies the pinned topology port.
    """

    source_path, source_manifest, source_sha = _read_json_object(
        source_manifest_path, label="source TRAIN manifest"
    )
    repaired_path, repaired_manifest, repaired_sha = _read_json_object(
        repaired_manifest_path, label="repaired TRAIN manifest"
    )
    report_path, repair_report, report_sha = _read_json_object(
        repair_report_path, label="geometry repair report"
    )
    from .geometry_repair import REPAIR_SCHEMA

    if repair_report.get("schema") != REPAIR_SCHEMA or repair_report.get("status") != "success":
        raise ValueError("geometry repair report must be a successful geometry_repair.v1 report")
    if repair_report.get("audit_only") is not False:
        raise ValueError("audit-only repair reports cannot authorize a corrected geometry proposal")
    if repair_report.get("errors") != []:
        raise ValueError("geometry repair report must have no errors")
    receipt = repair_report.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("successful geometry repair report must contain its receipt")
    audit_rows = repair_report.get("audit_rows")
    if not isinstance(audit_rows, list):
        raise ValueError("geometry repair report must contain audited patient rows")
    mapping = repair_report.get("cine_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("geometry repair report must declare its CINE mapping provenance")

    from .geometry_repair import (
        validate_audit_geometry,
        validate_geometry_record,
        validate_geometry_repair_receipt,
    )

    receipt_validation = validate_geometry_repair_receipt(
        source_manifest,
        repaired_manifest,
        receipt,
        source_manifest_sha256=source_sha,
        repaired_manifest_sha256=repaired_sha,
    )
    train_ids, by_patient = _manifest_geometry_records(
        repaired_manifest, label="repaired TRAIN manifest"
    )
    source_ids, source_by_patient = _manifest_geometry_records(
        source_manifest, label="source TRAIN manifest"
    )
    if source_ids != train_ids:
        raise ValueError("source and repaired TRAIN patient order differs")

    if repair_report.get("source_manifest_sha256_expected") != source_sha:
        raise ValueError("repair report expected source SHA-256 differs from source manifest bytes")
    if repair_report.get("source_manifest_sha256_actual") != source_sha:
        raise ValueError("repair report actual source SHA-256 differs from source manifest bytes")
    report_source_path = repair_report.get("source_manifest_path")
    if not isinstance(report_source_path, str) or Path(report_source_path).expanduser().resolve() != source_path:
        raise ValueError("repair report source path does not identify the supplied source manifest")
    if receipt.get("source_manifest_sha256") != source_sha:
        raise ValueError("repair receipt source manifest SHA-256 differs from source bytes")
    if receipt.get("repaired_manifest_sha256") != repaired_sha:
        raise ValueError("repair receipt repaired manifest SHA-256 differs from repaired bytes")
    if repair_report.get("axis_map") != receipt.get("axis_map"):
        raise ValueError("repair report axis map differs from its validated receipt")

    expected_mapping = {
        "source": receipt.get("cine_mapping_source"),
        "sha256": receipt.get("cine_mapping_sha256"),
        "provenance": receipt.get("cine_mapping_provenance"),
        "voxel_bridge_status": receipt.get("voxel_bridge_status"),
    }
    if any(mapping.get(key) != value for key, value in expected_mapping.items()):
        raise ValueError("repair report CINE mapping fields differ from its validated receipt")
    if mapping.get("shape_match_is_not_voxel_equivalence") is not True:
        raise ValueError("repair report must preserve the unresolved voxel-equivalence boundary")
    mapping_sha = receipt.get("cine_mapping_sha256")
    if not isinstance(mapping_sha, str) or len(mapping_sha) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in mapping_sha
    ):
        raise ValueError("repair receipt CINE mapping SHA-256 is invalid")

    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        if not isinstance(row, dict) or not isinstance(row.get("patient_id"), str):
            raise ValueError("repair report audit rows must identify each patient")
        patient_id = row["patient_id"]
        if patient_id in rows_by_id:
            raise ValueError(f"repair report has duplicate audit row for {patient_id}")
        rows_by_id[patient_id] = row
    if set(rows_by_id) != set(train_ids):
        raise ValueError("repair report audit rows must exactly cover TRAIN patient IDs")
    for patient_id in train_ids:
        row = rows_by_id[patient_id]
        if row.get("errors") != []:
            raise ValueError(f"repair report audit row for {patient_id} contains errors")
        if row.get("cine_mapping_source") != receipt.get("cine_mapping_source"):
            raise ValueError(f"audit row mapping source differs from receipt for {patient_id}")
        if row.get("cine_mapping_sha256") != receipt.get("cine_mapping_sha256"):
            raise ValueError(f"audit row mapping SHA-256 differs from receipt for {patient_id}")
        if row.get("cine_mapping_provenance") != receipt.get("cine_mapping_provenance"):
            raise ValueError(f"audit row mapping provenance differs from receipt for {patient_id}")
        if row.get("voxel_bridge_status") != "NOT_VERIFIED":
            raise ValueError(f"audit row must preserve NOT_VERIFIED voxel bridge for {patient_id}")
        for hash_key in ("source_file_sha256", "source_header_sha256"):
            value = row.get(hash_key)
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in value
            ):
                raise ValueError(f"audit row for {patient_id} has invalid {hash_key}")
        for unit_key in ("resolved_spatial_unit", "unit_resolution"):
            value = row.get(unit_key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"audit row for {patient_id} lacks {unit_key}")
        resolution = row["unit_resolution"]
        if resolution == "explicit_declaration":
            provenance = row.get("unit_provenance")
            if not isinstance(provenance, str) or not provenance.strip():
                raise ValueError(f"audit row for {patient_id} lacks explicit unit provenance")
        elif resolution != "nifti_header":
            raise ValueError(f"audit row for {patient_id} has unknown unit resolution")
        old_shape = source_by_patient[patient_id].get("shape_zyx")
        old_spacing = source_by_patient[patient_id].get("spacing_zyx_mm")
        old_affine = source_by_patient[patient_id].get("affine_zyx")
        new_geometry = by_patient[patient_id]
        if row.get("manifest_shape_zyx") != old_shape:
            raise ValueError(f"audited source shape differs from source manifest for {patient_id}")
        if row.get("manifest_spacing_zyx_mm") != old_spacing:
            raise ValueError(f"audited source spacing differs from source manifest for {patient_id}")
        if row.get("manifest_affine_zyx") != old_affine:
            raise ValueError(f"audited source affine differs from source manifest for {patient_id}")
        if row.get("repaired_spacing_zyx_mm") != new_geometry.get("spacing_zyx_mm"):
            raise ValueError(f"audited spacing differs from repaired manifest for {patient_id}")
        if row.get("repaired_affine_zyx") != new_geometry.get("affine_zyx"):
            raise ValueError(f"audited affine differs from repaired manifest for {patient_id}")
        source_shape_xyzt = row.get("source_shape_xyzt")
        if not isinstance(source_shape_xyzt, list) or len(source_shape_xyzt) != 4 \
                or any(type(value) is not int or value <= 0 for value in source_shape_xyzt) \
                or list(reversed(source_shape_xyzt[:3])) != old_shape:
            raise ValueError(f"audited native header shape differs from source manifest for {patient_id}")
        zooms_xyz_mm = row.get("header_zooms_xyz_mm")
        if not isinstance(zooms_xyz_mm, list) or len(zooms_xyz_mm) != 3:
            raise ValueError(f"audit row for {patient_id} lacks three converted header zooms")
        try:
            reversed_zooms = [float(value) for value in reversed(zooms_xyz_mm)]
            repaired_spacing = [float(value) for value in new_geometry["spacing_zyx_mm"]]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"audit row for {patient_id} has invalid header zooms") from exc
        if any(not math.isfinite(value) or value <= 0 for value in reversed_zooms) \
                or any(not math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-6)
                       for left, right in zip(reversed_zooms, repaired_spacing)):
            raise ValueError(f"header zooms disagree with repaired spacing for {patient_id}")
        validate_audit_geometry(row, new_geometry)

    geometry_rows: list[dict[str, Any]] = []
    for patient_id in train_ids:
        normalized = validate_geometry_record(by_patient[patient_id])
        geometry_rows.append(
            {
                "shape_zyx": normalized["shape_zyx"],
                "spacing_zyx_mm": normalized["spacing_zyx_mm"],
            }
        )
    summary = summarize_train_geometry(geometry_rows)
    if summary["planner_output"]["stage_count"] != 6:
        raise ValueError(
            "pinned native topology proposes "
            f"{summary['planner_output']['stage_count']} stages; Q3D ResEnc-L requires six"
        )

    return {
        "schema": "self_audit.q3d_resenc.geometry_reprobe_proposal.v1",
        "status": "PROPOSAL_ONLY_NOT_LOCKED",
        "evidence_class": "SOFTWARE_ONLY_METADATA_PROPOSAL",
        "manifest_lineage": {
            "source_manifest_path": str(source_path),
            "source_manifest_sha256": source_sha,
            "repaired_manifest_path": str(repaired_path),
            "repaired_manifest_sha256": repaired_sha,
            "repair_report_path": str(report_path),
            "repair_report_sha256": report_sha,
            "repair_receipt_schema": receipt.get("schema"),
            "repair_receipt_status": receipt.get("status"),
            "repair_receipt_validation": receipt_validation,
            "train_patient_count": len(train_ids),
            "geometry_source": "repaired_manifest.geometry.by_patient",
            "arrays_or_headers_read_by_this_command": False,
        },
        "geometry": summary,
        "topology_source": {
            "repository": NNUNET_REPOSITORY,
            "commit": NNUNET_COMMIT,
            "path": NNUNET_TOPOLOGY_PATH,
            "git_blob_sha1": NNUNET_TOPOLOGY_GIT_BLOB_SHA1,
            "sha256": NNUNET_TOPOLOGY_SHA256,
            "algorithm": "get_pool_and_conv_props",
            "port_matches_pinned_source_golden_vectors": True,
        },
        "resenc_planner_source": {
            "repository": NNUNET_REPOSITORY,
            "commit": NNUNET_COMMIT,
            "path": NNUNET_RESENC_PLANNER_PATH,
            "git_blob_sha1": NNUNET_RESENC_PLANNER_GIT_BLOB_SHA1,
            "sha256": NNUNET_RESENC_PLANNER_SHA256,
            "native_protocol_call": "direct median TRAIN spacing and unchanged median native shape into get_pool_and_conv_props; omit target-spacing and patch-size preprocessing",
        },
        "script_source_hashes": {
            "self_audit/candidate_q3d/geometry_planning.py": sha256_file(Path(__file__)),
        },
        "probe_contract": dict(CANONICAL_PROBE_CONTRACT),
        "limitations": [
            "This is a software-only topology proposal and does not lock config or checkpoint identity.",
            "The median diagnostic is not a per-patient topology guarantee.",
            "The nnU-Net planner conv kernels are diagnostic only; the Q3D teacher retains its 3x3x3 checkpoint kernel contract.",
            "No ERF checkpoint probe, patient arrays, raw headers, GPU run, training, or campaign was performed by this command.",
            "A successful geometry repair receipt retains voxel_bridge_status=NOT_VERIFIED; shape equality alone does not establish voxel equivalence.",
        ],
    }
