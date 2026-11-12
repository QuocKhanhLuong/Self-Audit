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
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
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


def _teacher_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    teacher_cfg = config.get("model", {}).get("teacher", {})
    pretrained = teacher_cfg.get("pretrained") if isinstance(teacher_cfg, dict) else None
    pretrained = pretrained if isinstance(pretrained, dict) else {}
    source_repo_value = teacher_cfg.get("source_repo", pretrained.get("source_repo"))
    source_commit = teacher_cfg.get("source_commit", pretrained.get("source_commit"))
    weights_value = pretrained.get("weights_path", teacher_cfg.get("weights"))
    weights_sha = pretrained.get("weights_sha256", teacher_cfg.get("weights_sha256"))
    weights_path = _resolve_path(weights_value)
    weight_status = "NOT_RUN"
    if weights_path is None:
        weight_status = "NOT_RUN_NO_WEIGHTS_PATH_OR_SHA"
    elif not weights_path.is_file():
        weight_status = "NOT_RUN_WEIGHTS_FILE_MISSING"
    elif not isinstance(weights_sha, str) or len(weights_sha) != 64:
        weight_status = "NOT_RUN_WEIGHTS_SHA_MISSING"
    elif _sha256(weights_path).lower() != weights_sha.lower():
        weight_status = "WEIGHTS_SHA_MISMATCH"
    else:
        weight_status = "HASH_VERIFIED_FILE_PRESENT"

    source_name = "stride8" if teacher_cfg.get("feature_stride") == 8 else (
        "stride4" if teacher_cfg.get("feature_stride") == 4 else (
            "stride16" if teacher_cfg.get("feature_stride") == 16 else "stride8"
        )
    )
    result: dict[str, Any] = {
        "requested_backbone": teacher_cfg.get("architecture"),
        "requested_source": source_name,
        "expected_module_by_adapter_contract": {
            "stride4": "dec_block_2",
            "stride8": "dec_block_3",
            "stride16": "bottleneck",
        },
        "actual_module": None,
        "feature_channels": None,
        "source_status": "NOT_RUN",
        "source_reason": None,
        "weights": {
            "path": str(weights_path) if weights_path else None,
            "sha256": weights_sha,
            "status": weight_status,
            "pretraining_dataset": pretrained.get("pretraining_dataset"),
            "overlap_acdc": pretrained.get("overlap_acdc"),
            "overlap_mnms": pretrained.get("overlap_mnms"),
        },
    }
    if teacher_cfg.get("architecture") != "mednext_l_k3":
        result["source_reason"] = "selected arm has no MedNeXt teacher"
        return result
    source_repo = _resolve_path(source_repo_value)
    if source_repo is None or source_commit is None:
        result["source_reason"] = "local MedNeXt source checkout and pinned commit are not declared"
        return result
    if not source_repo.is_dir():
        result["source_reason"] = "declared local MedNeXt source checkout is not present"
        result["source_repo"] = str(source_repo)
        return result

    # Architecture-only construction is permitted for module/channel inventory;
    # the result remains explicitly untrained if a verified compatible weight
    # file is unavailable. No forward, data, download or gate is run here.
    from self_audit.candidate_a3d.teacher import build_teacher3d
    from self_audit.candidate_q3d.teacher_features import teacher_feature_source

    build_config = {
        "backbone": "mednext_l_k3",
        "source_repo": str(source_repo),
        "source_commit": str(source_commit),
        "weights": str(weights_path) if weights_path and weights_path.is_file() else None,
        "weights_sha256": weights_sha if weights_path and weights_path.is_file() else None,
        "input_channels": int(config.get("model", {}).get("student", {}).get("input_channels", 1)),
    }
    try:
        teacher, provenance = build_teacher3d(build_config, synthetic=False)
        details = teacher_feature_source(teacher, source_name)
    except Exception as exc:
        result["source_status"] = "NOT_RUN"
        result["source_reason"] = f"local module inventory failed closed: {type(exc).__name__}: {exc}"
        return result
    result.update({
        "source_status": "ACTUAL_LOCAL_MODULE_INSPECTED",
        "source_reason": None,
        "source_repo": str(source_repo),
        "source_commit": provenance.get("backbone", {}).get("source_commit"),
        "actual_module": details["module_name"],
        "feature_channels": details["channels"],
        "features_api": details["features_api"],
        "weight_quality": provenance.get("quality"),
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

        model = CandidateQ3DStudent(config=config)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        maximum = int(student_cfg.get("max_parameters", 10_000_000))
        return {
            "status": "PASS" if total <= maximum else "NO_GO",
            "parameter_count": total,
            "trainable_parameter_count": trainable,
            "maximum_parameters": maximum,
            "within_budget": total <= maximum,
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


def build_report(config_path: Path) -> dict[str, Any]:
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
            "teacher_feature": _teacher_diagnostics(raw_config) if raw_config else {
                "source_status": "NOT_RUN", "source_reason": "config JSON is unavailable or invalid",
            },
            "near_far_sdf": _near_far_diagnostics(raw_config),
            "datasets": {
                "acdc": {"status": "NOT_RUN", "reason": "Q-3D recipe failed validation"},
                "mnms": {"status": "NOT_RUN", "reason": "Q-3D recipe failed validation"},
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
    datasets: dict[str, Any] = {}
    for name in ("acdc", "mnms"):
        if name == current_name:
            spec = train_manifest
        else:
            spec = None
        datasets[name] = _manifest_diagnostics(name, spec, config_path=config_path)
    teacher_feature = _teacher_diagnostics(config)
    student_parameters = _student_parameter_diagnostics(config)
    readiness_reasons = []
    if teacher_feature.get("source_status") != "ACTUAL_LOCAL_MODULE_INSPECTED":
        readiness_reasons.append("actual MedNeXt feature module/channel inventory is NOT_RUN")
    if teacher_feature.get("weights", {}).get("status") != "HASH_VERIFIED_FILE_PRESENT":
        readiness_reasons.append("compatible hash-pinned teacher weights are unavailable")
    if any(value.get("status") != "READ_ONLY_MANIFEST_DIAGNOSTICS" for value in datasets.values()):
        readiness_reasons.append("one or more dataset TRAIN geometry/GT diagnostics are NOT_RUN")
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
    args = parser.parse_args(argv)
    config_path = _resolve_path(args.config)
    output_path = _resolve_path(args.output)
    if config_path is None or output_path is None:
        parser.error("config and output must be non-empty paths")
    report = build_report(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output_path)}, sort_keys=True))
    return 0 if report.get("report_status") == "READ_ONLY_DIAGNOSTICS_COMPLETE" \
        or report["status"] == "INVALID_CONFIG" else 1


if __name__ == "__main__":
    raise SystemExit(main())
