#!/usr/bin/env python3
"""Read-only Q-3D P0 diagnostics; unavailable evidence stays NOT_RUN.

This command inspects one validated nested recipe, local source/weight paths,
the declared train manifest and model source. It never downloads assets,
modifies a manifest, trains a model, or reads DEV/TEST labels.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESENC_HF_REVISION = "f443c09319988e833b3d9589bf95909e4443d7b2"
RESENC_DNA_VERSION = "0.4.4"
RESENC_CHECKPOINT_SHA256 = "ac262d3e8c226c79f9567fddc34d38e011284730ddf2294b5c6c319ce2f22bbd"
_RESENC_CHECKPOINT_ENV = os.environ.get("Q3D_RESENC_CHECKPOINT")
RESENC_CHECKPOINT = Path(_RESENC_CHECKPOINT_ENV).expanduser() if _RESENC_CHECKPOINT_ENV else None
RESENC_INSPECTION = Path(os.environ.get(
    "Q3D_RESENC_INSPECTION",
    str(ROOT / "reports/q3d-resenc-phase0-20261112/checkpoint_inspection.json"),
))
RESENC_ERF_RECEIPT = Path(os.environ.get(
    "Q3D_RESENC_ERF_RECEIPT",
    str(ROOT / "reports/q3d-resenc-pixdim-20261112/erf_probe.json"),
))
RESENC_ARCHITECTURE = "ResEncL"
RESENC_ERF_SAMPLES = 16
RESENC_ERF_THRESHOLD_VOXELS = 2.0
RESENC_ERF_LOCKED_OFFSET = (0.0, 0.0, 0.0)
RESENC_WEIGHTS_ENV = "Q3D_NNFOUNDATION_WEIGHTS"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: Any, *, base: Path = ROOT) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _resolve_teacher_weights_path(value: Any) -> Path | None:
    """Resolve the same explicit runtime binding used by the Q-3D loader."""

    if isinstance(value, str) and value.strip() == "${Q3D_NNFOUNDATION_WEIGHTS}":
        bound = os.environ.get(RESENC_WEIGHTS_ENV)
        if not bound:
            return None
        value = bound
    return _resolve_path(value)


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        location = (len(ordered) - 1) * q
        lower = math.floor(location)
        upper = math.ceil(location)
        if lower == upper:
            return ordered[lower]
        fraction = location - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return {
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": float(statistics.median(ordered)),
        "p75": percentile(0.75),
        "max": ordered[-1],
    }


def _erf_protocol_error(
    *, samples: int, threshold: float,
    locked_offset: tuple[float, float, float],
) -> str | None:
    """Reject diagnostic overrides that would invalidate the locked Phase-0 receipt."""

    if int(samples) != RESENC_ERF_SAMPLES:
        return f"ERF protocol is locked to {RESENC_ERF_SAMPLES} samples"
    if float(threshold) != RESENC_ERF_THRESHOLD_VOXELS:
        return f"ERF protocol is locked to threshold {RESENC_ERF_THRESHOLD_VOXELS:g} voxels"
    if tuple(float(value) for value in locked_offset) != RESENC_ERF_LOCKED_OFFSET:
        return "ERF protocol is locked to native phase offset (0, 0, 0)"
    return None


def _geometry_fields(value: Any) -> tuple[tuple[int, int, int] | None, tuple[float, float, float] | None]:
    if not isinstance(value, dict):
        return None, None
    nested = value.get("geometry")
    if isinstance(nested, dict):
        value = nested
    shape = value.get("shape_zyx", value.get("shape"))
    spacing = value.get("spacing_zyx_mm", value.get("spacing_zyx"))
    try:
        shape_tuple = tuple(int(item) for item in shape)
        spacing_tuple = tuple(float(item) for item in spacing)
    except (TypeError, ValueError):
        return None, None
    if len(shape_tuple) != 3 or any(item <= 0 for item in shape_tuple):
        shape_tuple = None
    if len(spacing_tuple) != 3 or any(not math.isfinite(item) or item <= 0 for item in spacing_tuple):
        spacing_tuple = None
    return shape_tuple, spacing_tuple


def _manifest_diagnostics(
    dataset_name: str,
    manifest_spec: Any,
    *,
    config_path: Path,
    student_phase: Any = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    if not isinstance(manifest_spec, dict):
        return {"status": "NOT_RUN", "reason": "no TRAIN manifest is declared"}
    raw_path = manifest_spec.get("path")
    raw_sha = manifest_spec.get("sha256")
    if raw_path is None or raw_sha is None:
        return {
            "status": "NOT_RUN",
            "reason": "TRAIN manifest path or SHA-256 is not declared",
            "configured_status": manifest_spec.get("status", "NOT_RUN"),
        }
    manifest_path = _resolve_path(raw_path, base=config_path.parent)
    if manifest_path is None or not manifest_path.is_file():
        return {
            "status": "NOT_RUN",
            "reason": "declared TRAIN manifest is not present on disk",
            "path": str(manifest_path) if manifest_path else str(raw_path),
        }
    actual_sha = _sha256(manifest_path)
    if not isinstance(raw_sha, str) or actual_sha.lower() != raw_sha.lower():
        return {
            "status": "HASH_MISMATCH",
            "reason": "TRAIN manifest bytes do not match the configured SHA-256",
            "path": str(manifest_path),
            "expected_sha256": raw_sha,
            "actual_sha256": actual_sha,
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "INVALID_MANIFEST", "path": str(manifest_path), "reason": str(exc)}
    if not isinstance(manifest, dict):
        return {"status": "INVALID_MANIFEST", "path": str(manifest_path), "reason": "root is not an object"}
    if manifest.get("role") not in (None, "training", "train"):
        return {
            "status": "NOT_RUN",
            "path": str(manifest_path),
            "reason": "manifest role is not TRAIN; no DEV/TEST geometry or masks are opened",
        }
    if manifest.get("dataset") not in (None, dataset_name):
        return {
            "status": "INVALID_MANIFEST",
            "path": str(manifest_path),
            "reason": f"manifest dataset {manifest.get('dataset')!r} differs from {dataset_name!r}",
        }

    from self_audit.candidate_q3d.data import validate_acdc_geometry

    try:
        # No GT loader runs after a failed geometry or dataset identity check.
        validate_acdc_geometry(manifest, synthetic=synthetic, dataset_name=dataset_name)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": "INVALID_GEOMETRY",
            "path": str(manifest_path),
            "manifest_sha256": actual_sha,
            "reason": str(exc),
            "original_header_audit": {"status": "NOT_RUN"},
        }

    geometry = manifest.get("geometry")
    by_patient = geometry.get("by_patient") if isinstance(geometry, dict) else None
    if not isinstance(by_patient, dict) or not by_patient:
        return {
            "status": "NOT_RUN",
            "path": str(manifest_path),
            "manifest_sha256": actual_sha,
            "reason": "TRAIN manifest has no per-patient native geometry map",
        }
    shapes: list[tuple[int, int, int]] = []
    spacings: list[tuple[float, float, float]] = []
    missing_geometry: list[str] = []
    for patient, patient_geometry in by_patient.items():
        shape, spacing = _geometry_fields(patient_geometry)
        if shape is None or spacing is None:
            missing_geometry.append(str(patient))
            continue
        shapes.append(shape)
        spacings.append(spacing)
    if missing_geometry:
        return {
            "status": "NOT_RUN",
            "path": str(manifest_path),
            "manifest_sha256": actual_sha,
            "reason": "one or more TRAIN patients lack native shape or millimetre spacing",
            "missing_geometry_patients": missing_geometry,
            "patients_with_geometry": len(shapes),
        }

    return {
        "status": "READ_ONLY_MANIFEST_DIAGNOSTICS",
        "path": str(manifest_path),
        "manifest_sha256": actual_sha,
        "patient_count": len(shapes),
        "native_depth_D": _quantiles([float(shape[0]) for shape in shapes]),
        "spacing_zyx_mm": {
            axis: _quantiles([spacing[index] for spacing in spacings])
            for index, axis in enumerate(("z", "y", "x"))
        },
        "r_sv_r_m_on_labeled_train_ground_truth": _ground_truth_physiology(
            manifest_path, actual_sha,
        ),
        "labeled_train_stage4_gt_coverage": _ground_truth_stage4_coverage(
            manifest_path, actual_sha, student_phase,
        ),
    }


def _ground_truth_physiology(manifest_path: Path, manifest_sha: str) -> dict[str, Any]:
    """Measure r_SV/r_M from TRAIN labels only; invalid patients stay explicit."""

    try:
        from self_audit.candidate_a3d.data import CandidateA3DData

        data = CandidateA3DData(manifest_path, manifest_sha, synthetic=False)
        frame_volumes: dict[str, list[dict[str, float | int]]] = {}
        for key in data.labeled_frame_keys():
            _image, labels, geometry, metadata = data.load_labeled_frame(key)
            patient = str(metadata["patient"])
            voxel_ml = math.prod(float(value) for value in geometry["spacing_zyx"]) / 1000.0
            counts = torch_bincount(labels)
            frame_volumes.setdefault(patient, []).append({
                "frame": int(metadata["frame"]),
                "RV": counts[1] * voxel_ml,
                "MYO": counts[2] * voxel_ml,
                "LV": counts[3] * voxel_ml,
            })
    except Exception as exc:  # report missing/corrupt data, never manufacture a passing gate
        return {
            "status": "NOT_RUN",
            "reason": f"TRAIN GT could not be read with the strict A-3D loader: {type(exc).__name__}: {exc}",
        }

    rows: list[dict[str, float | str]] = []
    invalid: dict[str, int] = {"nonpositive_mean_stroke_volume": 0, "zero_mean_myo_volume": 0}
    for patient, frames in sorted(frame_volumes.items()):
        if len(frames) < 2:
            continue
        ed = max(frames, key=lambda row: float(row["LV"]))
        es = min(frames, key=lambda row: float(row["LV"]))
        sv_lv = float(ed["LV"]) - float(es["LV"])
        sv_rv = float(ed["RV"]) - float(es["RV"])
        mean_sv = (sv_lv + sv_rv) / 2.0
        mean_myo = (float(ed["MYO"]) + float(es["MYO"])) / 2.0
        if mean_sv <= 0.0:
            invalid["nonpositive_mean_stroke_volume"] += 1
            continue
        if mean_myo <= 0.0:
            invalid["zero_mean_myo_volume"] += 1
            continue
        rows.append({
            "patient": patient,
            "r_sv": abs(sv_lv - sv_rv) / mean_sv,
            "r_m": abs(float(ed["MYO"]) - float(es["MYO"])) / mean_myo,
        })
    return {
        "status": "MEASURED_TRAIN_GROUND_TRUTH" if rows else "NOT_RUN",
        "patients_with_two_or_more_labeled_frames": len(rows) + sum(invalid.values()),
        "patients_with_valid_r_sv_and_r_m": len(rows),
        "r_sv": _quantiles([float(row["r_sv"]) for row in rows]),
        "r_m": _quantiles([float(row["r_m"]) for row in rows]),
        "invalid_patient_counts": invalid,
        "patient_values": rows,
        "interpretation": "descriptive TRAIN-label diagnostics only; not a clinical gate",
    }


def torch_bincount(labels: Any) -> list[int]:
    # This function is intentionally tiny so the preflight needs no extra array
    # package beyond the repository's required PyTorch dependency.
    import torch

    return [int(value) for value in torch.bincount(labels.reshape(-1).long(), minlength=4).tolist()]


def _phase_json(phase: Any) -> dict[str, list[int | float]]:
    """Serialize the geometry API phase without rounding half-voxel offsets."""

    if not isinstance(phase, (tuple, list)) or len(phase) != 2:
        raise ValueError("grid phase must be a (step_zyx, offset_zyx) pair")
    step, offset = phase
    if not isinstance(step, (tuple, list)) or not isinstance(offset, (tuple, list)) \
            or len(step) != 3 or len(offset) != 3:
        raise ValueError("grid phase step and offset must each have three axes")

    def axis(values: Any) -> list[int | float]:
        output: list[int | float] = []
        for raw in values:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("grid phase contains a non-finite value")
            output.append(int(value) if value.is_integer() else value)
        return output

    return {"step_zyx": axis(step), "offset_zyx": axis(offset)}


def _phase_matches(first: Any, second: Any) -> bool:
    left = _phase_json(first)
    right = _phase_json(second)
    return left == right


def _phase_tuple_from_json(value: Any) -> tuple[tuple[int, ...], tuple[float, ...]] | None:
    if not isinstance(value, dict) or not isinstance(value.get("step_zyx"), list) \
            or not isinstance(value.get("offset_zyx"), list):
        return None
    if len(value["step_zyx"]) != 3 or len(value["offset_zyx"]) != 3:
        return None
    return (
        tuple(int(item) for item in value["step_zyx"]),
        tuple(float(item) for item in value["offset_zyx"]),
    )


def _conv3d_inventory(module: Any, *, name: str, phase: Any) -> dict[str, Any]:
    import torch

    if not isinstance(module, torch.nn.Conv3d):
        raise TypeError(f"{name} must be an actual Conv3d module")
    return {
        "module": type(module).__name__,
        "kernel_zyx": [int(value) for value in module.kernel_size],
        "stride_zyx": [int(value) for value in module.stride],
        "padding_zyx": [int(value) for value in module.padding],
        "dilation_zyx": [int(value) for value in module.dilation],
        "groups": int(module.groups),
        "in_channels": int(module.in_channels),
        "out_channels": int(module.out_channels),
        "phase_native": _phase_json(phase),
    }


def _resenc_api() -> Any:
    """Load the repository-owned ResEnc teacher API, never an old adapter."""

    from self_audit.candidate_q3d import teacher_resenc

    return teacher_resenc


def _resenc_erf_api() -> Any | None:
    """Load the standalone empirical ERF probe when it is available.

    Keeping the probe in its own module makes it reusable by the trainer and
    prevents the preflight from silently growing a second implementation.  A
    missing module is reported by the caller as a failed/optional diagnostic;
    it is never treated as a successful measurement.
    """

    try:
        from self_audit.candidate_q3d import erf
    except (ImportError, ModuleNotFoundError):
        return None
    return erf


def _erf_geometry_defaults(teacher_config: dict[str, Any]) -> dict[str, Any]:
    """Return a center-aligned odd native shape and its stage-3 cell."""

    raw_shape = teacher_config.get("erf_input_shape") or teacher_config.get("erf_shape") \
        or (33, 97, 97)
    try:
        shape = tuple(int(value) for value in raw_shape)
    except (TypeError, ValueError):
        shape = (33, 97, 97)
    if len(shape) != 3 or any(value <= 0 or value % 2 == 0 for value in shape):
        shape = (33, 97, 97)
    center = tuple((value - 1) // 2 for value in shape)
    raw_strides = teacher_config.get("strides") or teacher_config.get("encoder_strides") \
        or ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))
    try:
        source_step = [1, 1, 1]
        for stride in tuple(raw_strides)[:4]:
            values = stride if isinstance(stride, (tuple, list)) else (stride,) * 3
            for axis, value in enumerate(values):
                source_step[axis] *= int(value)
        source_cell = tuple(value // step for value, step in zip(center, source_step))
    except (TypeError, ValueError, IndexError):
        source_step = [1, 1, 1]
        source_cell = None
    return {
        "input_shape": shape,
        "input_center": center,
        "source_step": tuple(source_step),
        "source_cell": source_cell,
    }


def _invoke_api_callable(function: Any, *, config: dict[str, Any], checkpoint: Path,
                         checkpoint_sha256: str,
                         diagnostic_options: dict[str, Any] | None = None,
                         teacher: Any = None) -> Any:
    """Call a ResEnc API function using only parameters it explicitly declares."""

    if not callable(function):
        raise TypeError("ResEnc teacher API entrypoint is not callable")
    signature = inspect.signature(function)
    teacher_config = config.get("model", {}).get("teacher", {})
    if not isinstance(teacher_config, dict):
        teacher_config = {}
    erf_defaults = _erf_geometry_defaults(teacher_config)
    erf_shape = erf_defaults["input_shape"]
    input_center = erf_defaults["input_center"]
    source_cell = erf_defaults["source_cell"]
    values = {
        "config": config,
        "teacher_config": teacher_config,
        "teacher": teacher,
        "model": teacher,
        "checkpoint": checkpoint,
        "path": checkpoint,
        "checkpoint_path": checkpoint,
        "weights_path": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "sha256": checkpoint_sha256,
        "weights_sha256": checkpoint_sha256,
        "strict": True,
        "strict_encoder": True,
        "device": "cpu",
        "strides": (
            teacher_config.get("strides") or teacher_config.get("encoder_strides")
        ),
        "phase_offsets": teacher_config.get("phase_offsets"),
        "input_shape": erf_shape,
        "native_shape": erf_shape,
        "shape": erf_shape,
        "center": input_center,
        "input_center": input_center,
        "center_voxel": input_center,
        "cell_index": teacher_config.get("erf_cell_index") or source_cell,
        "source_cell": teacher_config.get("erf_cell_index") or source_cell,
        "center_cell": teacher_config.get("erf_cell_index") or source_cell,
        "aggregation": "mean_signed_offset",
        "mean_signed_offset": True,
        "threshold_voxels": 2.0,
        "samples": 16,
        "seed": 42,
    }
    if diagnostic_options:
        values.update(diagnostic_options)
    positional: list[Any] = []
    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if parameter.name in values:
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                positional.append(values[parameter.name])
            else:
                kwargs[parameter.name] = values[parameter.name]
        elif parameter.default is inspect.Parameter.empty:
            raise TypeError(f"unsupported required ResEnc API argument: {parameter.name}")
    return function(*positional, **kwargs)


def _resenc_phase_inventory(encoder: Any, *, threshold_voxels: float = 2.0) -> dict[str, Any] | None:
    """Measure ResEnc main and residual skip lattices from live modules.

    ResEnc uses an average-pool residual skip on the first block of each
    downsampling stage.  That pool has a half-cell offset, so reporting only
    the main convolution path would hide a real lattice mismatch.  Return
    ``None`` when the encoder does not expose the expected nnssl module
    topology; callers can then retain an API-provided diagnostic without
    manufacturing one from constants.
    """

    import torch
    from self_audit.candidate_q3d.alignment import grid_phase

    def conv_from(module: Any) -> torch.nn.Conv3d | None:
        if isinstance(module, torch.nn.Conv3d):
            return module
        direct = getattr(module, "conv", None)
        if isinstance(direct, torch.nn.Conv3d):
            return direct
        convs = getattr(module, "convs", None)
        if convs is not None:
            for child in convs:
                found = conv_from(child)
                if found is not None:
                    return found
        children = getattr(module, "children", None)
        if callable(children):
            for child in children():
                found = conv_from(child)
                if found is not None:
                    return found
        return None

    stem = conv_from(getattr(encoder, "stem", None))
    stages = getattr(encoder, "stages", None)
    if stem is None or stages is None:
        return None
    main_chain: list[Any] = [stem]
    residual_skip_chain: list[Any] = [stem]
    records: list[dict[str, Any]] = []
    try:
        for stage_index, stage in enumerate(stages):
            blocks = getattr(stage, "blocks")
            for block_index, block in enumerate(blocks):
                conv1 = conv_from(getattr(block, "conv1", None))
                conv2 = conv_from(getattr(block, "conv2", None))
                if conv1 is None or conv2 is None:
                    return None
                main_chain.extend((conv1, conv2))
                if block_index == 0 and stage_index > 0:
                    skip = getattr(block, "skip", None)
                    if skip is None:
                        return None
                    pool = next(
                        (child for child in skip.modules()
                         if isinstance(child, torch.nn.AvgPool3d)), None,
                    )
                    skip_conv = conv_from(skip)
                    if pool is None:
                        return None
                    residual_skip_chain.append(pool)
                    if skip_conv is not None:
                        residual_skip_chain.append(skip_conv)
            if stage_index > 0 and len(residual_skip_chain) > 1:
                records.append({
                    "stage": int(stage_index),
                    "main": _phase_json(grid_phase(main_chain)),
                    "residual_skip": _phase_json(grid_phase(residual_skip_chain)),
                })
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    if not records:
        return None
    source = next((item for item in records if item["stage"] == 3), records[-1])
    skip_offset = source["residual_skip"]["offset_zyx"]
    max_offset = max(abs(float(value)) for value in skip_offset)
    threshold = float(threshold_voxels)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("phase threshold must be finite and nonnegative")
    return {
        "status": "THEORETICAL_SKIP_OFFSET_REPORTED",
        "convention": "main_path",
        "source": "live_ResEncL_encoder_modules",
        "feature_source": "encoder_stride8" if source["stage"] == 3 else f"stage_{source['stage']}",
        "threshold_voxels": threshold,
        "max_abs_residual_skip_offset_voxels": max_offset,
        "stages": records,
        "main_path": source["main"],
        "residual_skip_path": source["residual_skip"],
    }


def _build_resenc_teacher(config: dict[str, Any], *, checkpoint: Path,
                          checkpoint_sha256: str) -> tuple[Any, dict[str, Any]]:
    """Build/load the ResEnc-L teacher through the worker-owned strict API."""

    api = _resenc_api()
    loader = getattr(api, "load_nnfoundation_encoder", None)
    teacher_type = getattr(api, "Q3DResEncTeacher", None)
    if not callable(loader) or not callable(teacher_type):
        raise AttributeError(
            "teacher_resenc must expose load_nnfoundation_encoder and Q3DResEncTeacher"
        )
    loaded = _invoke_api_callable(
        loader, config=config, checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2 or not isinstance(loaded[1], dict):
        raise TypeError("strict ResEnc loader must return (encoder, provenance)")
    encoder, provenance = loaded
    teacher_cfg = config.get("model", {}).get("teacher", {})
    strides = teacher_cfg.get("strides") or teacher_cfg.get("encoder_strides")
    phase_offsets = teacher_cfg.get("phase_offsets")
    input_channels = int(config.get("model", {}).get("student", {}).get("input_channels", 1))
    teacher = teacher_type(
        encoder=encoder,
        strides=strides,
        phase_offsets=phase_offsets,
        input_channels=input_channels,
        synthetic=False,
    )
    return teacher, dict(provenance)


def _resenc_network_inventory(teacher: Any, *, api: Any, config: dict[str, Any],
                              checkpoint_inventory: dict[str, Any],
                              diagnostic_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect actual ResEnc modules plus CPU phase/ERF probes when exposed."""

    import torch

    encoder = getattr(teacher, "encoder", teacher)
    modules = [
        {"name": name, "module": type(module).__name__,
         "kernel_zyx": [int(value) for value in module.kernel_size],
         "stride_zyx": [int(value) for value in module.stride],
         "padding_zyx": [int(value) for value in module.padding]}
        for name, module in encoder.named_modules()
        if isinstance(module, torch.nn.Conv3d)
    ]
    if not modules:
        raise ValueError("ResEnc teacher encoder exposes no Conv3d modules")
    result: dict[str, Any] = {
        "status": "ACTUAL_RESENC_MODULES_INSPECTED",
        "architecture": RESENC_ARCHITECTURE,
        "encoder_conv_count": len(modules),
        "encoder_convs": modules,
        "checkpoint_encoder_inventory": checkpoint_inventory,
        "erf_request": dict(diagnostic_options or {}),
    }
    cumulative_stride = getattr(teacher, "cumulative_stride", None)
    feature_phases = getattr(teacher, "feature_phases", None)
    if cumulative_stride is not None:
        result["geometry"] = {
            "status": "MEASURED_CPU",
            "native_grid": "declared_native_no_resampling",
            "cumulative_encoder_stride_zyx": [int(value) for value in cumulative_stride],
            "input_padding": "high_side_replicate_no_phase_shift",
        }
    phase_inventory = _resenc_phase_inventory(
        encoder,
        threshold_voxels=float((diagnostic_options or {}).get("phase_threshold_voxels", 2.0)),
    )
    if phase_inventory is not None:
        result["phase"] = phase_inventory
    elif isinstance(feature_phases, dict):
        result["phase"] = {
            "status": "DECLARED_PHASE_ONLY",
            "source": "declared_teacher_feature_phases",
            "feature_phases": {
                str(name): _phase_json(value) for name, value in feature_phases.items()
            },
            "locked_offset": list((diagnostic_options or {}).get("locked_offset", (0.0, 0.0, 0.0))),
        }
    diagnostic_calls = {
        "geometry": ("geometry_diagnostics", "resenc_geometry_diagnostics", "geometry_inventory"),
        "phase": ("phase_probe", "resenc_phase_probe", "phase_diagnostics"),
        "erf": ("gradient_erf_probe", "empirical_erf_probe", "erf_probe",
                "erf_diagnostics", "resenc_erf_diagnostics", "effective_receptive_field"),
    }
    erf_api = _resenc_erf_api()
    for field, names in diagnostic_calls.items():
        value = None
        reason = None
        run_erf = bool((diagnostic_options or {}).get("run_erf", False))
        if field == "erf" and run_erf:
            # A real run must execute the worker-owned gradient implementation
            # (run_resenc_erf). A placeholder NOT_RUN result is never accepted
            # as completion of --run-erf.
            namespaces = tuple(namespace for namespace in (erf_api, api) if namespace is not None)
            for namespace in namespaces:
                for name in (("run_resenc_erf",) + names[:3]):
                    candidate = getattr(namespace, name, None)
                    if not callable(candidate):
                        continue
                    try:
                        value = _invoke_api_callable(
                            candidate, config=config, checkpoint=Path(checkpoint_inventory["path"]),
                            checkpoint_sha256=str(checkpoint_inventory["sha256"]),
                            diagnostic_options=diagnostic_options, teacher=teacher,
                        )
                        if isinstance(value, dict) and value.get("status") not in {
                            "NOT_RUN", "NOT_RUN_OPTIONAL", "FAILED",
                        }:
                            break
                        value = None
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                if value is not None:
                    break
            if value is None:
                result["erf"] = {
                    "status": "FAILED",
                    "reason": reason or "worker-owned ResEnc ERF API is unavailable",
                    "required": "--run-erf must execute at least 16 gradient samples",
                }
                continue
        names_to_try = names[3:] if field == "erf" and not run_erf else names
        for name in names_to_try:
            if field == "erf" and run_erf and value is not None:
                break
            candidate = getattr(api, name, None)
            if not callable(candidate):
                continue
            try:
                value = _invoke_api_callable(
                    candidate, config=config, checkpoint=Path(checkpoint_inventory["path"]),
                    checkpoint_sha256=str(checkpoint_inventory["sha256"]),
                    diagnostic_options=diagnostic_options, teacher=teacher,
                )
                break
            except Exception as exc:  # diagnostics are reported, never fabricated
                reason = f"{type(exc).__name__}: {exc}"
        if value is None:
            if field == "erf" and not run_erf:
                result[field] = {
                    "status": "NOT_RUN_OPTIONAL",
                    "reason": "ERF probe is opt-in; use --run-erf for the CPU gradient diagnostic",
                }
            elif field in result:
                continue
            else:
                result[field] = {"status": "NOT_RUN", "reason": reason or "ResEnc API did not expose this probe"}
        else:
            result[field] = value if isinstance(value, dict) else {"status": "MEASURED", "value": value}
    phase = result.get("phase")
    erf = result.get("erf")
    if isinstance(phase, dict) and isinstance(erf, dict):
        if erf.get("status") in {"PASS", "PASS_MAIN_PHASE"}:
            phase["status"] = "EMPIRICAL_MAIN_PHASE_SELECTED"
            phase["empirical_decision"] = "USE_NOMINAL_MAIN_PHASE"
        elif erf.get("status") == "MEASURED_OFFSET_REQUIRED":
            phase["status"] = "EMPIRICAL_MEASURED_PHASE_SELECTED"
            phase["empirical_decision"] = "USE_MEASURED_SIGNED_MEAN_OFFSET"
            phase["selected_phase"] = erf.get("selected_phase")
    return result


def _resenc_checkpoint_inventory(checkpoint: Path | None, expected_sha256: str,
                                 inspection_path: Path | None = None) -> dict[str, Any]:
    """Verify the pinned Phase-0 checkpoint and its strict encoder inventory."""

    result: dict[str, Any] = {
        "path": str(checkpoint) if checkpoint is not None else None,
        "sha256": expected_sha256,
        "status": "NOT_RUN",
        "hf_revision": RESENC_HF_REVISION,
        "dna_version": RESENC_DNA_VERSION,
    }
    if checkpoint is None or not checkpoint.is_file():
        result.update({"status": "NOT_RUN_CHECKPOINT_MISSING", "reason": "checkpoint file is absent"})
        # A committed Phase-0 inspection receipt is useful evidence of the
        # exact bytes inspected elsewhere, but it is never promoted to a
        # local checkpoint hash verification or a loadable teacher.
        if inspection_path is None or not inspection_path.is_file():
            return result
        try:
            inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return result
        if isinstance(inspection, dict):
            expected_receipt = {
                "status": "PASS_STRICT_ENCODER",
                "hf_revision": RESENC_HF_REVISION,
                "weights_sha256": expected_sha256,
                "encoder_key_count": 448,
                "missing_keys": [],
                "unexpected_keys": [],
            }
            adaptation_plan = inspection.get("nnssl_adaptation_plan")
            architecture_plans = adaptation_plan.get("architecture_plans") \
                if isinstance(adaptation_plan, dict) else None
            architecture = architecture_plans.get("arch_class_name") \
                if isinstance(architecture_plans, dict) else None
            if all(inspection.get(key) == value for key, value in expected_receipt.items()) \
                    and architecture == RESENC_ARCHITECTURE:
                result["strict_encoder"] = {
                    "status": "PASS_RECEIPT_ONLY",
                    "architecture": RESENC_ARCHITECTURE,
                    "encoder_key_count": 448,
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "inspection_path": str(inspection_path),
                }
                result["receipt_only"] = True
            elif all(inspection.get(key) == value for key, value in expected_receipt.items()):
                result["strict_encoder"] = {
                    "status": "FAIL",
                    "architecture": architecture,
                    "reason": "strict inspection receipt architecture is not ResEncL",
                    "inspection_path": str(inspection_path),
                }
        return result
    actual = _sha256(checkpoint)
    result["actual_sha256"] = actual
    if actual.lower() != expected_sha256.lower():
        result.update({"status": "CHECKPOINT_SHA_MISMATCH", "reason": "checkpoint bytes differ from the pinned Phase-0 digest"})
        return result
    result["status"] = "HASH_VERIFIED_FILE_PRESENT"
    if inspection_path is None or not inspection_path.is_file():
        result["strict_encoder"] = {"status": "NOT_RUN", "reason": "strict checkpoint inspection receipt is absent"}
        return result
    try:
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["strict_encoder"] = {"status": "INVALID_INSPECTION", "reason": str(exc)}
        return result
    if not isinstance(inspection, dict):
        result["strict_encoder"] = {
            "status": "INVALID_INSPECTION",
            "reason": "strict checkpoint inspection receipt root is not an object",
        }
        return result
    required = {
        "status": "PASS_STRICT_ENCODER",
        "hf_revision": RESENC_HF_REVISION,
        "weights_sha256": expected_sha256,
        "encoder_key_count": 448,
        "missing_keys": [],
        "unexpected_keys": [],
    }
    mismatches = {
        key: {"expected": expected, "actual": inspection.get(key)}
        for key, expected in required.items() if inspection.get(key) != expected
    }
    adaptation_plan = inspection.get("nnssl_adaptation_plan")
    architecture_plans = adaptation_plan.get("architecture_plans") \
        if isinstance(adaptation_plan, dict) else None
    architecture = architecture_plans.get("arch_class_name") \
        if isinstance(architecture_plans, dict) else None
    if architecture != RESENC_ARCHITECTURE:
        mismatches["arch_class_name"] = {"expected": RESENC_ARCHITECTURE, "actual": architecture}
    result["strict_encoder"] = {
        "status": "PASS" if not mismatches else "FAIL",
        "architecture": architecture,
        "encoder_key_count": inspection.get("encoder_key_count"),
        "missing_keys": inspection.get("missing_keys"),
        "unexpected_keys": inspection.get("unexpected_keys"),
        "mismatches": mismatches,
        "inspection_path": str(inspection_path),
    }
    if mismatches:
        result["status"] = "STRICT_ENCODER_INVENTORY_FAILED"
    return result


def _resenc_receipt_network_inventory(
    receipt_path: Path,
    *,
    expected_sha256: str,
    diagnostic_options: dict[str, Any],
) -> dict[str, Any]:
    """Validate the committed Phase-0 ERF receipt without a local checkpoint."""

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Phase-0 ERF receipt is unavailable: {exc}") from exc
    if not isinstance(receipt, dict) or receipt.get("schema") != "self_audit.q3d_resenc.erf_probe.v1":
        raise ValueError("Phase-0 ERF receipt schema is invalid")
    model = receipt.get("model")
    compute = receipt.get("compute")
    decision = receipt.get("decision_rule")
    summary = receipt.get("summary")
    if not all(isinstance(value, dict) for value in (model, compute, decision, summary)):
        raise ValueError("Phase-0 ERF receipt is missing model, compute, decision or summary")
    if receipt.get("status") != "PASS":
        raise ValueError(f"Phase-0 ERF receipt status is {receipt.get('status')!r}")
    if model.get("architecture") != RESENC_ARCHITECTURE:
        raise ValueError("Phase-0 ERF receipt architecture is not ResEncL")
    if model.get("checkpoint_sha256") != expected_sha256:
        raise ValueError("Phase-0 ERF receipt checkpoint SHA-256 differs from the configured pin")
    try:
        from self_audit.candidate_q3d.geometry_planning import CANONICAL_PROBE_CONTRACT
        from self_audit.candidate_q3d.config import NNFOUNDATION_DEFAULT_ENCODER_STRIDES
    except ImportError as exc:
        raise ValueError("canonical geometry-planning ERF contract is unavailable") from exc
    canonical = CANONICAL_PROBE_CONTRACT
    canonical_schedule = [
        [int(axis) for axis in stage] for stage in NNFOUNDATION_DEFAULT_ENCODER_STRIDES
    ]
    probe_contract = receipt.get("geometry_reprobe", {}).get("probe_contract") \
        if isinstance(receipt.get("geometry_reprobe"), dict) else None
    if not isinstance(probe_contract, dict):
        raise ValueError("Phase-0 ERF receipt is missing its geometry-planning probe contract")
    canonical_fields = (
        "samples", "seed_start", "seed_end_inclusive", "shape_zyx", "stage_index",
        "channel_aggregation", "padding_mode", "decision_threshold_voxels",
    )
    if any(field not in canonical for field in canonical_fields):
        raise ValueError("canonical geometry-planning ERF contract is incomplete")
    for field in canonical_fields:
        if probe_contract.get(field) != canonical[field]:
            raise ValueError(f"Phase-0 ERF receipt probe contract {field} differs from geometry planning")
    if probe_contract.get("stride_schedule_zyx") != canonical_schedule:
        raise ValueError("Phase-0 ERF receipt probe contract stride schedule differs from Q3D config")

    samples = compute.get("samples")
    threshold = float(decision.get("threshold_voxels", 0.0))
    signed_mean_offset = summary.get("mean_centroid_offset_voxels")
    mean_offset = summary.get("absolute_mean_centroid_offset_voxels")
    stride = model.get("cumulative_stride_zyx")
    full_stride = model.get("full_teacher_cumulative_stride_zyx")
    shape = compute.get("shape_zyx")
    schedule = model.get("stride_schedule_zyx")
    expected_schedule = diagnostic_options.get("expected_stride_schedule_zyx")
    receipt_samples = receipt.get("samples")
    expected_seeds = list(range(
        int(canonical["seed_start"]), int(canonical["seed_end_inclusive"]) + 1,
    ))
    if not isinstance(receipt_samples, list):
        raise ValueError("Phase-0 ERF receipt does not contain its canonical sample rows")
    actual_seeds = [row.get("seed") if isinstance(row, dict) else None for row in receipt_samples]
    if samples != canonical["samples"] or len(receipt_samples) != canonical["samples"] \
            or actual_seeds != expected_seeds:
        raise ValueError("Phase-0 ERF receipt must contain the exact canonical 16 seeds 42-57")
    if compute.get("seed_start") != canonical["seed_start"]:
        raise ValueError("Phase-0 ERF receipt seed start differs from the canonical contract")
    if not math.isfinite(threshold) or threshold <= 0 \
            or not isinstance(mean_offset, list) or len(mean_offset) != 3 \
            or not isinstance(signed_mean_offset, list) or len(signed_mean_offset) != 3 \
            or not isinstance(stride, list) or len(stride) != 3 \
            or not isinstance(full_stride, list) or len(full_stride) != 3 \
            or not isinstance(shape, list) or len(shape) != 3 \
            or not isinstance(schedule, list) or len(schedule) != 6 \
            or not all(math.isfinite(float(value)) for value in mean_offset + signed_mean_offset):
        raise ValueError("Phase-0 ERF receipt does not satisfy the locked sample/geometry contract")
    if compute.get("padding_mode") != "replicate":
        raise ValueError("Phase-0 ERF receipt must use full-teacher replicate padding")
    if shape != canonical["shape_zyx"]:
        raise ValueError("Phase-0 ERF receipt input shape differs from the canonical contract")
    if any(
        not isinstance(row, dict) or row.get("input_shape_zyx") != canonical["shape_zyx"]
        for row in receipt_samples
    ):
        raise ValueError("Phase-0 ERF sample rows do not use the canonical input shape")
    if any(int(value) <= 0 or int(value) % 2 == 0 for value in shape):
        raise ValueError("Phase-0 ERF receipt input shape must be positive and odd")
    if any(int(value) <= 0 for value in stride + full_stride):
        raise ValueError("Phase-0 ERF receipt strides must be positive")
    if any(
        not isinstance(value, list) or len(value) != 3
        or any(int(axis) <= 0 for axis in value)
        for value in schedule
    ):
        raise ValueError("Phase-0 ERF receipt has an invalid stride schedule")
    if schedule != canonical_schedule:
        raise ValueError("Phase-0 ERF receipt stride schedule differs from Q3D config")
    if expected_schedule is not None and schedule != expected_schedule:
        raise ValueError("Phase-0 ERF receipt stride schedule differs from the configured teacher")
    cumulative_from_schedule = [
        math.prod(int(stage[axis]) for stage in schedule[: int(canonical["stage_index"]) + 1])
        for axis in range(3)
    ]
    full_cumulative_from_schedule = [
        math.prod(int(stage[axis]) for stage in schedule)
        for axis in range(3)
    ]
    if stride != cumulative_from_schedule or full_stride != full_cumulative_from_schedule:
        raise ValueError("Phase-0 ERF receipt cumulative strides disagree with its actual schedule")
    if compute.get("random_zscore_input") is not True or compute.get("patient_data") != "NOT_USED":
        raise ValueError("Phase-0 ERF receipt does not prove random Z-score, patient-free inputs")
    if int(model.get("stage_index", -1)) != canonical["stage_index"] \
            or model.get("stage_name") != f"encoder.stages[{canonical['stage_index']}]":
        raise ValueError("Phase-0 ERF receipt does not target encoder stage 3")
    if threshold != float(diagnostic_options.get("erf_threshold", threshold)):
        raise ValueError("Phase-0 ERF receipt threshold differs from the requested diagnostic threshold")
    if threshold != float(canonical["decision_threshold_voxels"]):
        raise ValueError("Phase-0 ERF receipt threshold differs from geometry planning")
    if any(
        not math.isclose(abs(float(signed)), float(absolute), rel_tol=0.0, abs_tol=1e-6)
        for signed, absolute in zip(signed_mean_offset, mean_offset)
    ):
        raise ValueError("Phase-0 ERF receipt signed and absolute mean offsets disagree")
    if any(abs(float(value)) > threshold for value in signed_mean_offset):
        raise ValueError("Phase-0 ERF receipt mean offset does not pass its declared threshold")
    return {
        "status": "PHASE0_RECEIPT_ONLY",
        "architecture": RESENC_ARCHITECTURE,
        "source": "repo_relative_phase0_erf_receipt",
        "checkpoint_encoder_inventory": {
            "status": "PASS_RECEIPT_ONLY",
            "checkpoint_sha256": expected_sha256,
            "receipt_path": str(receipt_path),
        },
        "geometry": {
            "status": "RECEIPT_VERIFIED",
            "native_grid": "declared_native_no_resampling",
            "cumulative_encoder_stride_zyx": [int(value) for value in stride],
            "full_teacher_cumulative_stride_zyx": [int(value) for value in full_stride],
            "input_padding": "high_side_replicate_no_phase_shift",
        },
        "phase": {
            "status": "EMPIRICAL_MAIN_PHASE_SELECTED",
            "source": "repo_relative_phase0_erf_receipt",
            "empirical_decision": "USE_NOMINAL_MAIN_PHASE",
            "selected_phase": {
                "step_zyx": [int(value) for value in stride],
                "offset_zyx": [0.0, 0.0, 0.0],
            },
            "erf_summary": {
                "samples": samples,
                "shape_zyx": [int(value) for value in shape],
                "signed_mean_centroid_offset_voxels": [float(value) for value in signed_mean_offset],
                "absolute_mean_centroid_offset_voxels": [float(value) for value in mean_offset],
                "threshold_voxels": threshold,
            },
        },
        "erf": {
            "status": "PASS_RECEIPT_ONLY",
            "receipt_path": str(receipt_path),
            "samples": samples,
            "summary": summary,
            "decision_rule": decision,
        },
        "erf_request": dict(diagnostic_options),
    }


def _stage4_coverage_record(
    class_counts: list[int], *, frame_count: int, patient_count: int,
) -> dict[str, Any]:
    total_cells = sum(int(value) for value in class_counts)
    if len(class_counts) != 4 or total_cells <= 0:
        return {
            "status": "NOT_RUN",
            "reason": "no valid L-labeled S4 cells were pooled",
            "label_scope": "L only",
        }
    class_names = ("BG", "RV", "MYO", "LV")
    fractions = [int(value) / total_cells for value in class_counts]
    myo_fraction = fractions[2]
    myo_open = myo_fraction < 0.02
    return {
        "status": "MEASURED_TRAIN_L_GROUND_TRUTH",
        "label_scope": "L only; no U, DEV or TEST labels",
        "class_order": list(class_names),
        "total_s4_cells": total_cells,
        "labeled_frame_count": int(frame_count),
        "labeled_patient_count": int(patient_count),
        "class_counts": {name: int(value) for name, value in zip(class_names, class_counts)},
        "class_fractions": {name: float(value) for name, value in zip(class_names, fractions)},
        "tie_break_class_id": 0,
        "tie_break_rule": "argmax chooses the lowest class index",
        "myo_review": {
            "status": "OPEN" if myo_open else "CLEAR",
            "threshold_fraction": 0.02,
            "observed_fraction": float(myo_fraction),
            "threshold_changed": False,
            "reason": "MYO support is below 2%; review class coverage and balancing before P2"
            if myo_open else None,
        },
    }


def _gt_probabilities_to_counts(probabilities: Any) -> list[int]:
    import torch

    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 5 \
            or probabilities.shape[0] != 1 or probabilities.shape[1] != 4:
        raise ValueError("pooled L GT probabilities must have shape [1,4,D,h,w]")
    if not torch.isfinite(probabilities).all() or bool((probabilities < 0).any()):
        raise ValueError("pooled L GT probabilities must be finite and nonnegative")
    sums = probabilities.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
        raise ValueError("pooled L GT class fractions must sum to one for every S4 cell")
    winners = probabilities.argmax(dim=1)
    return torch_bincount(winners)


def _ground_truth_stage4_coverage(
    manifest_path: Path, manifest_sha: str, student_phase: Any,
) -> dict[str, Any]:
    """Measure phase-correct argmax class fractions from labeled TRAIN L only."""

    if student_phase is None:
        return {
            "status": "NOT_RUN",
            "reason": "verified student S4 phase is unavailable",
            "label_scope": "L only",
        }
    try:
        import torch
        from torch.nn import functional as F

        from self_audit.candidate_a3d.data import CandidateA3DData
        from self_audit.candidate_q3d.alignment import pool_native_cells

        data = CandidateA3DData(manifest_path, manifest_sha, synthetic=False)
        class_counts = [0, 0, 0, 0]
        frame_count = 0
        patients: set[str] = set()
        for key in data.labeled_frame_keys():
            _image, labels, _geometry, metadata = data.load_labeled_frame(key)
            labels = torch.as_tensor(labels).long()
            if labels.ndim != 4 or labels.shape[0] != 1:
                raise ValueError("L mask must have native [1,D,H,W] shape")
            labels = labels[0]
            if bool((labels < 0).any()) or bool((labels > 3).any()):
                raise ValueError("L mask contains class IDs outside BG/RV/MYO/LV")
            native_shape = tuple(int(value) for value in labels.shape)
            one_hot = F.one_hot(labels, num_classes=4).permute(3, 0, 1, 2).unsqueeze(0)
            class_fractions = pool_native_cells(
                one_hot.to(dtype=torch.float32), native_shape, student_phase,
            )
            frame_counts = _gt_probabilities_to_counts(class_fractions)
            class_counts = [left + right for left, right in zip(class_counts, frame_counts)]
            frame_count += 1
            patients.add(str(metadata["patient"]))
        return _stage4_coverage_record(
            class_counts, frame_count=frame_count, patient_count=len(patients),
        )
    except Exception as exc:
        return {
            "status": "NOT_RUN",
            "reason": f"TRAIN L GT S4 coverage could not be measured: {type(exc).__name__}: {exc}",
            "label_scope": "L only; no U, DEV or TEST labels",
        }


def _teacher_diagnostics(
    config: dict[str, Any],
    *,
    erf_samples: int = RESENC_ERF_SAMPLES,
    erf_threshold: float = RESENC_ERF_THRESHOLD_VOXELS,
    locked_offset: tuple[float, float, float] = RESENC_ERF_LOCKED_OFFSET,
    run_erf: bool = False,
) -> dict[str, Any]:
    """Run read-only ResEnc Phase-0 checkpoint, geometry and ERF diagnostics."""

    teacher_cfg = config.get("model", {}).get("teacher", {})
    if not isinstance(teacher_cfg, dict):
        teacher_cfg = {}
    pretrained = teacher_cfg.get("pretrained")
    pretrained = pretrained if isinstance(pretrained, dict) else {}
    weights_value = (
        pretrained.get("weights_path")
        or pretrained.get("checkpoint_path")
        or teacher_cfg.get("weights_path")
        or teacher_cfg.get("checkpoint_path")
    )
    weights_sha = (
        pretrained.get("weights_sha256")
        or pretrained.get("checkpoint_sha256")
        or teacher_cfg.get("weights_sha256")
        or teacher_cfg.get("checkpoint_sha256")
        or RESENC_CHECKPOINT_SHA256
    )
    configured_weights_path = _resolve_teacher_weights_path(weights_value)
    if configured_weights_path is not None and configured_weights_path.is_file():
        weights_path = configured_weights_path
    elif RESENC_CHECKPOINT is not None and RESENC_CHECKPOINT.expanduser().is_file():
        weights_path = RESENC_CHECKPOINT.expanduser().resolve()
    else:
        weights_path = configured_weights_path
    checkpoint_inventory = _resenc_checkpoint_inventory(
        weights_path,
        str(weights_sha),
        RESENC_INSPECTION.resolve(),
    )
    result: dict[str, Any] = {
        "requested_backbone": teacher_cfg.get("architecture"),
        "requested_teacher": "nnFoundationCNN_ResEncL_Phase0",
        "expected_architecture": RESENC_ARCHITECTURE,
        "hf_revision": RESENC_HF_REVISION,
        "dna_version": RESENC_DNA_VERSION,
        "source_status": "NOT_RUN",
        "source_reason": None,
        "checkpoint": checkpoint_inventory,
        "weights": {
            "path": str(weights_path) if weights_path is not None else None,
            "sha256": weights_sha,
            "status": checkpoint_inventory.get("status"),
            "pretraining_dataset": pretrained.get("pretraining_dataset", "OpenMind/nnFoundationCNN"),
            "overlap_acdc": pretrained.get("overlap_acdc"),
            "overlap_mnms": pretrained.get("overlap_mnms"),
            "load_scope": "encoder_only_strict",
        },
        "resenc_source_inventory": {
            "status": "NOT_RUN",
            "architecture": RESENC_ARCHITECTURE,
            "hf_revision": RESENC_HF_REVISION,
            "dna_version": RESENC_DNA_VERSION,
            "checkpoint_sha256": weights_sha,
        },
    }
    protocol_error = _erf_protocol_error(
        samples=erf_samples, threshold=erf_threshold, locked_offset=locked_offset,
    )
    if protocol_error is not None:
        result["source_reason"] = protocol_error
        result["resenc_source_inventory"]["reason"] = protocol_error
        return result
    if teacher_cfg.get("architecture") not in {
        "resenc_l", "resenc_l_3d", "nnfoundation_cnn_resenc_l",
        "nnfoundation_resenc_l", RESENC_ARCHITECTURE,
    }:
        result["source_reason"] = "selected arm does not declare the ResEnc-L teacher"
        result["resenc_source_inventory"]["reason"] = result["source_reason"]
        return result
    checkpoint_ready = checkpoint_inventory.get("status") == "HASH_VERIFIED_FILE_PRESENT" \
        and checkpoint_inventory.get("strict_encoder", {}).get("status") == "PASS"
    diagnostic_options = {
        "samples": int(erf_samples),
        "erf_samples": int(erf_samples),
        "seed": 42,
        "threshold": float(erf_threshold),
        "erf_threshold": float(erf_threshold),
        "phase_threshold_voxels": 2.0,
        "locked_offset": tuple(float(value) for value in locked_offset),
        "run_erf": bool(run_erf),
        "aggregation": "mean_signed_offset",
        "offset_rule": "mean_signed_offset",
        "expected_stride_schedule_zyx": teacher_cfg.get("strides") or teacher_cfg.get("encoder_strides"),
        **_erf_geometry_defaults(teacher_cfg),
    }
    if not checkpoint_ready:
        if not run_erf:
            try:
                receipt_network = _resenc_receipt_network_inventory(
                    RESENC_ERF_RECEIPT.resolve(),
                    expected_sha256=str(weights_sha),
                    diagnostic_options=diagnostic_options,
                )
            except Exception as exc:
                receipt_network = None
                result["resenc_source_inventory"]["receipt_reason"] = str(exc)
            if receipt_network is not None:
                result["resenc_source_inventory"].update({
                    "status": "PHASE0_RECEIPT_VERIFIED",
                    "network": receipt_network,
                })
                result["source_status"] = "PHASE0_RECEIPT_ONLY"
                result["source_reason"] = "local checkpoint is absent; committed Phase-0 receipt is report-only"
                return result
        result["source_reason"] = "Phase-0 checkpoint hash or strict encoder inventory is not verified"
        result["resenc_source_inventory"]["reason"] = result["source_reason"]
        return result
    try:
        teacher, provenance = _build_resenc_teacher(
            config, checkpoint=weights_path,
            checkpoint_sha256=str(weights_sha),
        )
        api = _resenc_api()
        network_inventory = _resenc_network_inventory(
            teacher, api=api, config=config,
            checkpoint_inventory=checkpoint_inventory,
            diagnostic_options=diagnostic_options,
        )
    except Exception as exc:
        reason = f"strict ResEnc teacher load/probe failed closed: {type(exc).__name__}: {exc}"
        result["source_reason"] = reason
        result["resenc_source_inventory"]["reason"] = reason
        return result
    result["resenc_source_inventory"].update({
        "status": "PHASE0_CHECKPOINT_AND_ACTUAL_MODULES_INSPECTED",
        "provenance": provenance,
        "network": network_inventory,
    })
    result.update({
        "source_status": "ACTUAL_RESENC_MODULES_INSPECTED",
        "source_reason": None,
        "actual_module": type(teacher).__name__,
        "weight_quality": provenance.get("quality") if isinstance(provenance, dict) else None,
    })
    return result


def _student_parameter_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    student_cfg = config.get("model", {}).get("student", {})
    if student_cfg.get("architecture") != "unetrpp_lite_3d":
        return {
            "status": "NOT_RUN",
            "reason": f"P0 parameter probe supports unetrpp_lite_3d; received {student_cfg.get('architecture')!r}",
        }
    try:
        from self_audit.candidate_q3d.student import CandidateQ3DStudent
        from self_audit.candidate_q3d.alignment import student_grid_phase

        model = CandidateQ3DStudent(config=config)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        maximum = int(student_cfg.get("max_parameters", 10_000_000))
        try:
            phase = _phase_json(student_grid_phase(model))
            phase_status = "PASS"
            phase_reason = None
        except Exception as exc:
            phase = None
            phase_status = "NOT_RUN"
            phase_reason = f"student S4 phase probe failed: {type(exc).__name__}: {exc}"
        return {
            "status": "PASS" if total <= maximum else "NO_GO",
            "parameter_count": total,
            "trainable_parameter_count": trainable,
            "maximum_parameters": maximum,
            "within_budget": total <= maximum,
            "stage4_grid_phase_status": phase_status,
            "stage4_grid_phase": phase,
            "stage4_grid_phase_reason": phase_reason,
            "source": "CandidateQ3DStudent instantiated from the validated nested config; no forward run",
        }
    except Exception as exc:
        return {"status": "NOT_RUN", "reason": f"student parameter probe failed: {type(exc).__name__}: {exc}"}


def _near_far_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    from self_audit.candidate_q3d.losses import balanced_near_far_sdf_loss, q3d_teacher_loss
    from self_audit.candidate_a3d.teacher import teacher_loss as a3d_teacher_loss

    q3d_teacher_source = inspect.getsource(q3d_teacher_loss)
    a3d_teacher_source = inspect.getsource(a3d_teacher_loss)
    return {
        "status": "IMPLEMENTED_IN_Q3D_SOURCE" if (
            "balanced_near_far_sdf_loss" in q3d_teacher_source
            and callable(balanced_near_far_sdf_loss)
        ) else "NOT_RUN",
        "q3d_teacher_loss": {
            "source": inspect.getsourcefile(q3d_teacher_loss),
            "uses_balanced_near_far": "balanced_near_far_sdf_loss" in q3d_teacher_source,
            "radius_mm": config.get("losses", {}).get("sdf_radius_mm"),
            "near_mm": config.get("losses", {}).get("sdf_near_boundary_mm"),
            "alpha": config.get("losses", {}).get("sdf_near_far_alpha"),
        },
        "legacy_a3d_teacher_loss_uses_near_far": "balanced_near_far" in a3d_teacher_source,
    }


def build_report(
    config_path: Path,
    *,
    erf_samples: int = RESENC_ERF_SAMPLES,
    erf_threshold: float = RESENC_ERF_THRESHOLD_VOXELS,
    locked_offset: tuple[float, float, float] = RESENC_ERF_LOCKED_OFFSET,
    run_erf: bool = False,
) -> dict[str, Any]:
    from self_audit.candidate_q3d.config import load_config

    config_path = config_path.resolve()
    raw_sha = _sha256(config_path) if config_path.is_file() else None
    try:
        config = load_config(config_path)
    except Exception as exc:
        try:
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw_config, dict):
                raw_config = {}
        except (OSError, json.JSONDecodeError):
            raw_config = {}
        return {
            "schema": "self_audit.candidate_q3d.p0_preflight.v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "INVALID_CONFIG",
            "report_status": "READ_ONLY_DIAGNOSTICS_COMPLETE",
            "config_path": str(config_path),
            "config_sha256": raw_sha,
            "config_validation": {"status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"},
            "teacher_feature": _teacher_diagnostics(
                raw_config,
                erf_samples=erf_samples,
                erf_threshold=erf_threshold,
                locked_offset=locked_offset,
                run_erf=run_erf,
            ) if raw_config else {
                "source_status": "NOT_RUN", "source_reason": "config JSON is unavailable or invalid",
            },
            "near_far_sdf": _near_far_diagnostics(raw_config),
            "datasets": {
                "acdc": {
                    "status": "NOT_RUN", "reason": "Q-3D recipe failed validation",
                    "labeled_train_stage4_gt_coverage": {
                        "status": "NOT_RUN", "reason": "Q-3D recipe failed validation",
                        "label_scope": "L only; no U, DEV or TEST labels",
                    },
                },
                "mnms": {
                    "status": "NOT_RUN", "reason": "Q-3D recipe failed validation",
                    "labeled_train_stage4_gt_coverage": {
                        "status": "NOT_RUN", "reason": "Q-3D recipe failed validation",
                        "label_scope": "L only; no U, DEV or TEST labels",
                    },
                },
            },
            "student_parameters": _student_parameter_diagnostics(raw_config) if raw_config else {
                "status": "NOT_RUN", "reason": "config JSON is unavailable or invalid",
            },
            "gpu_diagnostics": {"status": "NOT_RUN"},
            "clinical_gates": {"status": "NOT_RUN"},
        }

    dataset_config = config.get("dataset", {})
    current_name = str(dataset_config.get("name", "unknown")).lower()
    manifests = dataset_config.get("manifests", {})
    train_manifest = manifests.get("train") if isinstance(manifests, dict) else None
    teacher_feature = _teacher_diagnostics(
        config,
        erf_samples=erf_samples,
        erf_threshold=erf_threshold,
        locked_offset=locked_offset,
        run_erf=run_erf,
    )
    student_parameters = _student_parameter_diagnostics(config)
    student_phase = None
    if student_parameters.get("stage4_grid_phase_status") == "PASS":
        student_phase = _phase_tuple_from_json(student_parameters.get("stage4_grid_phase"))
    datasets: dict[str, Any] = {}
    for name in ("acdc", "mnms"):
        if name == current_name:
            spec = train_manifest
        else:
            spec = None
        datasets[name] = _manifest_diagnostics(
            name, spec, config_path=config_path, student_phase=student_phase,
            synthetic=config.get("runtime", {}).get("synthetic", False),
        )
        if "labeled_train_stage4_gt_coverage" not in datasets[name]:
            datasets[name]["labeled_train_stage4_gt_coverage"] = {
                "status": "NOT_RUN",
                "reason": datasets[name].get("reason", "TRAIN manifest diagnostics are unavailable"),
                "label_scope": "L only; no U, DEV or TEST labels",
            }
    readiness_reasons = []
    if teacher_feature.get("resenc_source_inventory", {}).get("status") \
            != "PHASE0_CHECKPOINT_AND_ACTUAL_MODULES_INSPECTED":
        readiness_reasons.append("Phase-0 ResEnc checkpoint/module/phase inventory is NOT_RUN")
    if teacher_feature.get("source_status") != "ACTUAL_RESENC_MODULES_INSPECTED":
        readiness_reasons.append("Q3D config does not load the strict ResEnc-L teacher API")
    if teacher_feature.get("checkpoint", {}).get("status") != "HASH_VERIFIED_FILE_PRESENT":
        readiness_reasons.append("hash-pinned Phase-0 ResEnc teacher checkpoint is unavailable")
    network_diagnostics = teacher_feature.get("resenc_source_inventory", {}).get("network", {})
    phase_status = network_diagnostics.get("phase", {}).get("status")
    erf_status = network_diagnostics.get("erf", {}).get("status")
    if erf_status in {None, "NOT_RUN", "NOT_RUN_OPTIONAL"}:
        readiness_reasons.append("empirical ResEnc ERF phase probe is NOT_RUN")
    elif erf_status == "FAILED":
        readiness_reasons.append("empirical ResEnc ERF phase probe failed")
    elif erf_status not in {
        "PASS", "PASS_MAIN_PHASE", "MEASURED_OFFSET_REQUIRED", "PASS_RECEIPT_ONLY",
    }:
        readiness_reasons.append("empirical ResEnc ERF result is not verified")
    elif phase_status not in {
        "EMPIRICAL_MAIN_PHASE_SELECTED", "EMPIRICAL_MEASURED_PHASE_SELECTED",
    }:
        readiness_reasons.append("empirical ResEnc ERF did not select a phase contract")
    if any(value.get("status") != "READ_ONLY_MANIFEST_DIAGNOSTICS" for value in datasets.values()):
        readiness_reasons.append("one or more dataset TRAIN geometry/GT diagnostics are NOT_RUN")
    if student_parameters.get("stage4_grid_phase_status") != "PASS":
        readiness_reasons.append("verified student S4 grid phase is NOT_RUN")
    for value in datasets.values():
        coverage = value.get("labeled_train_stage4_gt_coverage", {})
        if coverage.get("status") != "MEASURED_TRAIN_L_GROUND_TRUTH":
            readiness_reasons.append("one or more dataset L-only S4 GT class coverage diagnostics are NOT_RUN")
            break
        if coverage.get("myo_review", {}).get("status") == "OPEN":
            readiness_reasons.append("MYO S4 coverage is OPEN for review before P2")
            break
    if student_parameters.get("status") != "PASS":
        readiness_reasons.append("student parameter-budget diagnostic did not pass")

    return {
        "schema": "self_audit.candidate_q3d.p0_preflight.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "NOT_READY" if readiness_reasons else "SOFTWARE_PREFLIGHT_COMPLETE",
        "report_status": "READ_ONLY_DIAGNOSTICS_COMPLETE",
        "readiness_reasons": readiness_reasons,
        "evidence_boundary": "software and declared TRAIN manifest diagnostics only; no training, GPU, DEV/TEST or clinical gate",
        "config_path": str(config_path),
        "config_sha256": raw_sha,
        "config_validation": {
            "status": "PASS",
            "schema": config.get("schema"),
            "arm": config.get("arm"),
            "dataset": current_name,
            "protocol": config.get("protocol", {}).get("variant"),
            "k": config.get("protocol", {}).get("k"),
        },
        "teacher_feature": teacher_feature,
        "near_far_sdf": _near_far_diagnostics(config),
        "datasets": datasets,
        "student_parameters": student_parameters,
        "gpu_diagnostics": {"status": "NOT_RUN"},
        "clinical_gates": {
            "status": "NOT_RUN",
            "reason": "P0 reports geometry and TRAIN-GT residual diagnostics only; it does not certify clinical validity",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/candidate_q3d/v1.json",
        help="validated Q-3D JSON recipe (default: V1 ACDC K=5)",
    )
    parser.add_argument(
        "--output", default="reports/q3d-implementation-20261112/p0_preflight.json",
        help="JSON diagnostics output path",
    )
    parser.add_argument(
        "--erf-samples", type=int, default=RESENC_ERF_SAMPLES,
        help="locked CPU ERF probe sample count",
    )
    parser.add_argument(
        "--erf-threshold", type=float, default=RESENC_ERF_THRESHOLD_VOXELS,
        help="locked ERF threshold; only 2.0 voxels is accepted",
    )
    parser.add_argument(
        "--locked-offset", nargs=3, type=float, default=RESENC_ERF_LOCKED_OFFSET, metavar=("Z", "Y", "X"),
        help="locked native phase offset; only 0 0 0 is accepted",
    )
    parser.add_argument(
        "--run-erf", action="store_true",
        help="run the CPU gradient ERF probe; never performs training",
    )
    args = parser.parse_args(argv)
    config_path = _resolve_path(args.config)
    output_path = _resolve_path(args.output)
    if config_path is None or output_path is None:
        parser.error("config and output must be non-empty paths")
    if args.erf_samples < 1 or not math.isfinite(args.erf_threshold) or args.erf_threshold <= 0:
        parser.error("ERF samples must be positive and threshold must be finite and positive")
    protocol_error = _erf_protocol_error(
        samples=args.erf_samples,
        threshold=args.erf_threshold,
        locked_offset=tuple(args.locked_offset),
    )
    if protocol_error is not None:
        parser.error(protocol_error)
    report = build_report(
        config_path,
        erf_samples=args.erf_samples,
        erf_threshold=args.erf_threshold,
        locked_offset=tuple(args.locked_offset),
        run_erf=args.run_erf,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output_path)}, sort_keys=True))
    return 0 if report.get("report_status") == "READ_ONLY_DIAGNOSTICS_COMPLETE" \
        or report["status"] == "INVALID_CONFIG" else 1


if __name__ == "__main__":
    raise SystemExit(main())
