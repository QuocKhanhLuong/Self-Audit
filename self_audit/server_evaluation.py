"""Domain-wise development evaluation for the shared server compact student.

The development selector is deliberately locked to the equal mean of the ACDC
and M&Ms historical class-first patient Macro4 values.  Both datasets are
assembled into one namespaced native manifest and passed through the canonical
two-phase compact evaluator, so every prediction is frozen before any
development mask is opened.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from .compact_artifacts import digest
from .compact_inference import (
    CLASS_NAMES,
    VALID_EVAL_ROLES,
    load_native_manifest,
    require_sha256,
    run_two_phase_evaluation,
    write_json,
)
from .shared_data import namespace_patient_id


DOMAIN_NAMES: Tuple[str, str] = ("acdc", "mnm")
EVALUATION_MAPS_SCHEMA = "self_audit.server_evaluation_maps.v1"
DOMAIN_SELECTION_RULE = (
    "selection_score = arithmetic mean of ACDC and M&Ms historical class-first patient Macro4; "
    "never pool patients across domains"
)


def shared_selection_summary(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize an already frozen two-domain evaluation without reading GT."""
    datasets = metrics.get("datasets", {})
    if set(datasets) != set(DOMAIN_NAMES) or metrics.get("coverage", {}).get("complete") is not True:
        raise ValueError("shared selection requires complete acdc and mnm evaluation")
    macro, foreground, reports = {}, {}, {}
    for domain in DOMAIN_NAMES:
        row = datasets[domain]
        for key, dest in (("class_first_macro4", macro), ("class_first_foreground_macro", foreground)):
            value = row.get(key)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("shared selection metric is missing/nonfinite")
            dest[domain] = float(value)
        reports[domain] = {
            **row,
            "macro_patient": _domain_classwise_metrics(row["macro_patient"], metrics["frames"], domain),
        }
    return {"macro4": sum(macro.values()) / 2,
            "foreground_macro": sum(foreground.values()) / 2,
            "selection_metric": "equal_domain_class_first_macro4", "domain_results": reports}
_SAFE_MAP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _read_pinned_json(path: Union[str, Path], expected_sha256: str, name: str) -> Tuple[Path, Dict[str, Any], str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    expected = require_sha256(expected_sha256, f"{name} SHA-256")
    actual = digest(resolved)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{name} root must be a JSON object")
    return resolved, data, actual


def _manifest_pin(entry: Mapping[str, Any], index_dir: Path, map_name: str) -> Tuple[Path, str, str | None, Sequence[str] | None]:
    """Resolve either source-index pins or the prepared-index pin spelling."""
    if set(entry) == {"domain", "manifest", "sha256"}:
        raw_path, raw_sha = entry["manifest"], entry["sha256"]
        role = None
        patient_ids = None
    elif set(entry) == {"domain", "manifest_path", "manifest_sha256", "role", "patient_ids"}:
        raw_path, raw_sha = entry["manifest_path"], entry["manifest_sha256"]
        role = entry["role"]
        patient_ids = entry["patient_ids"]
    else:
        raise ValueError(
            f"evaluation map {map_name} must use exact source or prepared path/SHA pin fields"
        )
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"evaluation map {map_name} manifest path must be a nonempty string")
    manifest_path = Path(raw_path)
    if not manifest_path.is_absolute():
        manifest_path = index_dir / manifest_path
    return manifest_path.resolve(), require_sha256(raw_sha, f"evaluation map {map_name} manifest SHA-256"), role, patient_ids


def _resolve_record_path(raw_path: Any, manifest_dir: Path, label: str) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{label} must be a nonempty path string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_dir / path
    return str(path.resolve())


def _validate_shared_l4_l4_train(train_path: Path, expected_sha256: str) -> Tuple[Dict[str, Any], str]:
    """Fail closed unless TRAIN is the namespaced shared 4+4 L8 protocol."""
    train_data, actual_sha = load_native_manifest(
        train_path,
        expected_sha256,
        expected_roles=("training", "train"),
    )
    if train_data.get("schema") != "self_audit.data.v2" or train_data.get("role") != "training":
        raise ValueError("joint compact evaluation requires the canonical self_audit.data.v2 training manifest")
    if train_data.get("shared") is not True or train_data.get("domains") != ["acdc", "mnm"]:
        raise ValueError("joint compact evaluation requires the canonical shared ACDC+M&Ms TRAIN manifest")
    label_budget = train_data.get("label_budget")
    if type(label_budget) is not int or label_budget != 8:
        raise ValueError("joint compact TRAIN label_budget must be exactly 8 (ACDC L4 + M&Ms L4)")
    budgets = train_data.get("domain_budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(DOMAIN_NAMES):
        raise ValueError("joint compact TRAIN domain_budgets must have exactly acdc and mnm keys")
    if any(type(budgets[name]) is not int or budgets[name] != 4 for name in DOMAIN_NAMES):
        raise ValueError("joint compact TRAIN domain_budgets must be exactly ACDC4 + M&Ms4")
    labeled = train_data.get("labeled")
    unlabeled = train_data.get("unlabeled")
    if not isinstance(labeled, list) or len(labeled) != 8 or not isinstance(unlabeled, list):
        raise ValueError("joint compact TRAIN must declare exactly eight labeled records and an unlabeled list")
    domain_map = train_data.get("domain_map")
    if not isinstance(domain_map, dict):
        raise ValueError("joint compact TRAIN domain_map is required")
    counts = {name: 0 for name in DOMAIN_NAMES}
    all_records = labeled + unlabeled
    ids = []
    for record in all_records:
        if not isinstance(record, Mapping):
            raise ValueError("joint compact TRAIN records must be objects")
        patient_id = record.get("id")
        if not isinstance(patient_id, str) or ":" not in patient_id:
            raise ValueError("joint compact TRAIN IDs must already use the canonical domain:id namespace")
        domain, _ = patient_id.split(":", 1)
        if domain not in DOMAIN_NAMES or domain_map.get(patient_id) != domain:
            raise ValueError(f"joint compact TRAIN ID/domain_map mismatch for {patient_id!r}")
        ids.append(patient_id)
    if len(ids) != len(set(ids)):
        raise ValueError("joint compact TRAIN contains duplicate labeled/unlabeled IDs")
    if set(domain_map) != set(ids):
        raise ValueError("joint compact TRAIN domain_map must exactly cover labeled and unlabeled IDs")
    for record in labeled:
        counts[domain_map[record["id"]]] += 1
    if counts != {"acdc": 4, "mnm": 4}:
        raise ValueError("joint compact TRAIN labeled IDs must be exactly four per domain")
    return train_data, actual_sha


def _normalized_domain_manifest(
    domain: str,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> Tuple[list[Dict[str, Any]], list[str]]:
    if manifest.get("role") != "development":
        raise ValueError(f"{domain} manifest must have role='development'")
    for field in ("domain", "dataset"):
        declared = manifest.get(field)
        if declared is not None and declared != domain:
            raise ValueError(f"{domain} development manifest has conflicting top-level {field}={declared!r}")
    records = manifest.get("patients")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{domain} development manifest must have a nonempty patients list")

    normalized: list[Dict[str, Any]] = []
    patient_ids: list[str] = []
    for source_record in records:
        if not isinstance(source_record, Mapping):
            raise ValueError(f"{domain} development patient records must be JSON objects")
        record = copy.deepcopy(dict(source_record))
        original_id = record.get("id")
        if not isinstance(original_id, str) or not original_id.strip():
            raise ValueError(f"{domain} development patient ID must be a nonempty string")
        for field in ("domain", "dataset"):
            declared = record.get(field)
            if declared is not None and declared != domain:
                raise ValueError(f"{domain} patient {original_id!r} has conflicting {field}={declared!r}")
        patient_id = namespace_patient_id(domain, original_id)
        record["id"] = patient_id
        record["domain"] = domain

        image_key = "image" if record.get("image") is not None else "path"
        record[image_key] = _resolve_record_path(record.get(image_key), manifest_path.parent, f"{domain} patient {patient_id} image")
        masks = record.get("masks")
        if not isinstance(masks, Mapping) or not masks:
            raise ValueError(f"{domain} patient {patient_id} must declare at least one reference mask")
        normalized_masks: Dict[str, Any] = {}
        for frame, mask_record in masks.items():
            if not isinstance(mask_record, Mapping):
                raise ValueError(f"{domain} patient {patient_id} frame {frame} mask record must be an object")
            updated_mask = copy.deepcopy(dict(mask_record))
            updated_mask["path"] = _resolve_record_path(
                updated_mask.get("path"), manifest_path.parent, f"{domain} patient {patient_id} frame {frame} mask"
            )
            normalized_masks[str(frame)] = updated_mask
        record["masks"] = normalized_masks

        valid_mask = record.get("valid_mask")
        if isinstance(valid_mask, Mapping):
            updated_valid_mask = copy.deepcopy(dict(valid_mask))
            updated_valid_mask["path"] = _resolve_record_path(
                updated_valid_mask.get("path"), manifest_path.parent, f"{domain} patient {patient_id} valid_mask"
            )
            record["valid_mask"] = updated_valid_mask

        patient_ids.append(patient_id)
        normalized.append(record)
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError(f"{domain} development manifest contains duplicate namespaced patient IDs")
    return normalized, patient_ids


def _domain_coverage(metrics: Mapping[str, Any], domain_ids: Sequence[str]) -> Dict[str, Any]:
    coverage = metrics.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("canonical evaluator did not return patient/frame coverage")
    expected_frames = coverage.get("expected_frames")
    scored_frames = coverage.get("scored_frames")
    predicted_all = coverage.get("predicted_patients")
    scored_all = coverage.get("scored_patients")
    if not all(isinstance(value, Mapping) for value in (expected_frames, scored_frames)):
        raise ValueError("canonical evaluator returned malformed frame coverage")
    if not isinstance(predicted_all, list) or not isinstance(scored_all, list):
        raise ValueError("canonical evaluator returned malformed patient coverage")
    domain_set = set(domain_ids)
    predicted = sorted(domain_set & set(predicted_all))
    scored = sorted(domain_set & set(scored_all))
    expected_frames_domain = {pid: expected_frames[pid] for pid in domain_ids if pid in expected_frames}
    scored_frames_domain = {pid: scored_frames[pid] for pid in domain_ids if pid in scored_frames}
    expected_frame_count = sum(len(frames) for frames in expected_frames_domain.values())
    scored_frame_count = sum(len(frames) for frames in scored_frames_domain.values())
    complete = (
        len(expected_frames_domain) == len(domain_ids)
        and predicted == sorted(domain_ids)
        and scored == sorted(domain_ids)
        and expected_frames_domain == scored_frames_domain
    )
    return {
        "expected_patients": list(domain_ids),
        "predicted_patients": predicted,
        "scored_patients": scored,
        "expected_patient_count": len(domain_ids),
        "predicted_patient_count": len(predicted),
        "scored_patient_count": len(scored),
        "expected_frames": expected_frames_domain,
        "scored_frames": scored_frames_domain,
        "expected_frame_count": expected_frame_count,
        "scored_frame_count": scored_frame_count,
        "complete": complete,
    }


def _domain_classwise_metrics(
    dice_macros: Mapping[str, Any],
    frame_rows: Any,
    domain: str,
) -> Dict[str, Dict[str, Any]]:
    """Report patient-mean Dice, precision, and recall for each canonical class."""
    per_patient: Dict[str, Dict[str, Dict[str, list[float]]]] = {}
    if not isinstance(frame_rows, list):
        raise ValueError("canonical evaluator did not return frame rows")
    for row in frame_rows:
        if not isinstance(row, Mapping) or row.get("dataset") != domain:
            continue
        patient_id = row.get("patient")
        raw = row.get("raw")
        if not isinstance(patient_id, str) or not isinstance(raw, list) or len(raw) != len(CLASS_NAMES):
            raise ValueError(f"canonical evaluator returned malformed {domain} frame metrics")
        patient_values = per_patient.setdefault(
            patient_id,
            {name: {metric: [] for metric in ("precision", "recall")} for name in CLASS_NAMES},
        )
        for index, class_name in enumerate(CLASS_NAMES):
            class_values = raw[index]
            if not isinstance(class_values, Mapping):
                raise ValueError(f"canonical evaluator returned malformed {domain}/{class_name} frame metrics")
            for metric in ("precision", "recall"):
                value = class_values.get(metric)
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"canonical evaluator returned invalid {domain}/{class_name}/{metric}")
                patient_values[class_name][metric].append(float(value))

    result: Dict[str, Dict[str, Any]] = {}
    for class_name in CLASS_NAMES:
        dice_record = dice_macros.get(class_name, {})
        if not isinstance(dice_record, Mapping):
            raise ValueError(f"canonical evaluator returned malformed {domain}/{class_name} Dice macro")
        class_result: Dict[str, Any] = {
            "dice": dice_record.get("dice"),
            "dice_patients": dice_record.get("present_patients", 0),
        }
        for metric in ("precision", "recall"):
            patient_means = [
                sum(values[class_name][metric]) / len(values[class_name][metric])
                for values in per_patient.values()
                if values[class_name][metric]
            ]
            class_result[metric] = sum(patient_means) / len(patient_means) if patient_means else None
            class_result[f"{metric}_patients"] = len(patient_means)
        result[class_name] = class_result
    return result


def evaluate_shared_domains(
    *,
    export_path: Union[str, Path],
    export_sha256: str,
    train_manifest_path: Union[str, Path],
    train_manifest_sha256: str,
    evaluation_maps_path: Union[str, Path],
    evaluation_maps_sha256: str,
    out_dir: Union[str, Path],
    device: str = "cpu",
    batch_size: int = 16,
    normalization: str = "prepared",
) -> Dict[str, Any]:
    """Evaluate one raw compact export on pinned ACDC and M&Ms development data.

    ``evaluation_maps_path`` must pin a ``self_audit.server_evaluation_maps.v1``
    index containing exactly one role=development manifest for each canonical
    domain. Other roles, including final test manifests, are not scored here.
    The returned selection score is the equal mean of the two domain Macro4
    values; it never pools patients across domains.
    """
    clean_export_sha = require_sha256(export_sha256, "export_sha256")
    clean_train_sha = require_sha256(train_manifest_sha256, "train_manifest_sha256")
    clean_maps_sha = require_sha256(evaluation_maps_sha256, "evaluation_maps_sha256")
    export_root = Path(export_path).resolve()
    train_path = Path(train_manifest_path).resolve()
    train_data, actual_train_sha = _validate_shared_l4_l4_train(train_path, clean_train_sha)
    if actual_train_sha != clean_train_sha:
        raise ValueError("joint TRAIN manifest changed after pin validation")

    index_path, index, actual_maps_sha = _read_pinned_json(
        evaluation_maps_path, clean_maps_sha, "evaluation maps index"
    )
    if index.get("schema") != EVALUATION_MAPS_SCHEMA:
        raise ValueError(f"evaluation maps index schema must be {EVALUATION_MAPS_SCHEMA}")
    maps = index.get("maps")
    if not isinstance(maps, Mapping) or not maps:
        raise ValueError("evaluation maps index must contain a nonempty maps object")

    selected: Dict[str, Dict[str, Any]] = {}
    for map_name in sorted(maps):
        if not isinstance(map_name, str) or _SAFE_MAP_NAME.fullmatch(map_name) is None:
            raise ValueError(f"invalid evaluation map name {map_name!r}")
        entry = maps[map_name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"evaluation map {map_name} must be an object")
        domain = entry.get("domain")
        if domain not in DOMAIN_NAMES:
            raise ValueError(f"evaluation map {map_name} domain must be exactly 'acdc' or 'mnm'")
        manifest_path, manifest_sha, index_role, indexed_patient_ids = _manifest_pin(entry, index_path.parent, map_name)
        if index_role is not None and index_role not in VALID_EVAL_ROLES:
            raise ValueError(f"evaluation map {map_name} has unsupported indexed role {index_role!r}")
        manifest, actual_manifest_sha = load_native_manifest(manifest_path, manifest_sha)
        role = manifest.get("role")
        if role not in VALID_EVAL_ROLES:
            raise ValueError(f"evaluation map {map_name} has unsupported manifest role {role!r}")
        if index_role is not None and role != index_role:
            raise ValueError(f"evaluation map {map_name} indexed role differs from its pinned manifest")
        if role != "development":
            continue
        if type(manifest.get("synthetic")) is not bool or manifest["synthetic"] != train_data.get("synthetic"):
            raise ValueError(f"evaluation map {map_name} synthetic identity differs from shared TRAIN")
        records, patient_ids = _normalized_domain_manifest(domain, manifest_path, manifest)
        if indexed_patient_ids is not None:
            if not isinstance(indexed_patient_ids, list) or indexed_patient_ids != patient_ids:
                raise ValueError(f"evaluation map {map_name} patient_ids do not match its pinned manifest")
        if domain in selected:
            raise ValueError(f"evaluation index has more than one development manifest for domain {domain!r}")
        selected[domain] = {
            "map_name": map_name,
            "manifest_path": manifest_path,
            "manifest_sha256": actual_manifest_sha,
            "patients": records,
            "patient_ids": patient_ids,
        }
    if set(selected) != set(DOMAIN_NAMES):
        raise ValueError("evaluation maps index must pin exactly one development manifest for both acdc and mnm")

    all_ids = [pid for domain in DOMAIN_NAMES for pid in selected[domain]["patient_ids"]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("ACDC and M&Ms development manifests contain duplicate namespaced IDs")
    synthetic = train_data.get("synthetic")
    composite = {
        "schema": "self_audit.native_manifest.v1",
        "role": "development",
        "synthetic": synthetic,
        "dataset": "acdc_mnm_shared",
        "scope": "exposed_development",
        "source_manifest_index_sha256": actual_maps_sha,
        "source_manifests": {
            domain: {
                "map_name": selected[domain]["map_name"],
                "path": str(selected[domain]["manifest_path"]),
                "sha256": selected[domain]["manifest_sha256"],
            }
            for domain in DOMAIN_NAMES
        },
        "patients": [record for domain in DOMAIN_NAMES for record in selected[domain]["patients"]],
    }

    out_root = Path(out_dir).resolve()
    if out_root.exists():
        raise ValueError(f"Output directory already exists: {out_root}")
    if not out_root.name:
        raise ValueError("Output directory must name a new directory")
    manifest_root = out_root.with_name(f"{out_root.name}_manifests")
    selection_lock_path = out_root.with_name(f"{out_root.name}_SELECTION_LOCK.json")
    if manifest_root.exists() or selection_lock_path.exists():
        raise ValueError("domain evaluation manifest or selection-lock output already exists")

    protected_dirs = [
        export_root,
        train_path.parent,
        index_path.parent,
        *(selected[domain]["manifest_path"].parent for domain in DOMAIN_NAMES),
    ]
    for protected in protected_dirs:
        if out_root == protected or out_root.is_relative_to(protected) or protected.is_relative_to(out_root):
            raise ValueError(f"Output directory cannot overlap an input artifact/data directory: {protected}")
        if manifest_root == protected or manifest_root.is_relative_to(protected) or protected.is_relative_to(manifest_root):
            raise ValueError(f"Namespaced manifest directory cannot overlap an input artifact/data directory: {protected}")

    manifest_root.mkdir(parents=True, exist_ok=False)
    composite_path = manifest_root / "shared_development_manifest.json"
    composite_sha = write_json(composite_path, composite)

    # The fixed source-level rule is recorded before the canonical evaluator
    # runs its prediction-freeze phase and opens any reference masks.
    selection_lock_sha = write_json(selection_lock_path, {
        "schema": "self_audit.server_domain_selection_lock.v1",
        "locked_before_scoring": True,
        "rule": DOMAIN_SELECTION_RULE,
        "selection_metric": "equal_domain_class_first_macro4",
        "domains": list(DOMAIN_NAMES),
        "train_manifest_sha256": clean_train_sha,
        "evaluation_maps_sha256": actual_maps_sha,
        "development_manifests": {
            domain: {
                "map_name": selected[domain]["map_name"],
                "manifest_sha256": selected[domain]["manifest_sha256"],
            }
            for domain in DOMAIN_NAMES
        },
        "export_manifest_sha256": clean_export_sha,
    })

    metrics = run_two_phase_evaluation(
        export_path=export_root,
        export_sha256=clean_export_sha,
        manifest_path=composite_path,
        manifest_sha256=composite_sha,
        out_dir=out_root,
        train_manifest_path=train_path,
        train_manifest_sha256=clean_train_sha,
        device=device,
        batch_size=batch_size,
        normalization=normalization,
    )
    if metrics.get("status") != "COMPLETED":
        raise ValueError("canonical compact evaluator did not complete both-domain development scoring")
    if metrics.get("export_manifest_sha256") != clean_export_sha:
        raise ValueError("canonical evaluator metrics do not bind the exact shared compact export")
    if metrics.get("train_manifest_sha256") != clean_train_sha:
        raise ValueError("canonical evaluator metrics do not bind the exact shared TRAIN manifest")
    if metrics.get("evaluation_manifest_sha256") != composite_sha:
        raise ValueError("canonical evaluator metrics do not bind the namespaced development manifest")
    datasets = metrics.get("datasets")
    if not isinstance(datasets, Mapping) or set(DOMAIN_NAMES) - set(datasets):
        raise ValueError("canonical compact evaluator did not report both development domains")

    domain_results: Dict[str, Any] = {}
    domain_macro4: Dict[str, float] = {}
    for domain in DOMAIN_NAMES:
        report = datasets[domain]
        macro4 = report.get("class_first_macro4") if isinstance(report, Mapping) else None
        if isinstance(macro4, bool) or not isinstance(macro4, (int, float)) or not math.isfinite(float(macro4)):
            raise ValueError(f"{domain} development did not return a finite historical class-first Macro4")
        domain_macro4[domain] = float(macro4)
        ids = selected[domain]["patient_ids"]
        coverage = _domain_coverage(metrics, ids)
        if not coverage["complete"]:
            raise ValueError(f"{domain} development patient/frame coverage is incomplete")
        classwise = report.get("macro_patient", {})
        if not isinstance(classwise, Mapping):
            raise ValueError(f"{domain} development returned malformed class-wise patient metrics")
        domain_results[domain] = {
            "status": "COMPLETED",
            "class_first_macro4": float(macro4),
            "class_first_foreground_macro": report.get("class_first_foreground_macro"),
            "macro_patient": _domain_classwise_metrics(classwise, metrics.get("frames", []), domain),
            "class_counts": _domain_class_counts(metrics.get("frames", []), domain),
            "patient_coverage": coverage,
            "failure_cases": report.get("failure_cases", []),
            "total_valid_voxels": report.get("total_valid_voxels"),
        }

    selection_score = (domain_macro4["acdc"] + domain_macro4["mnm"]) / 2.0
    if not math.isfinite(selection_score):
        raise ValueError("equal-domain Macro4 selection score is non-finite")
    metrics_path = out_root / "METRICS.json"
    frozen_path = out_root / "PREDICTIONS_FROZEN.json"
    report = {
        "schema": "self_audit.server_domain_evaluation.v1",
        "status": "COMPLETED",
        "selection_rule": DOMAIN_SELECTION_RULE,
        "selection_score": selection_score,
        "selection_metric": "equal_domain_class_first_macro4",
        "domain_macro4": domain_macro4,
        "training_protocol": {
            "identity": "joint_supervised_acdc_l4_mnm_l4",
            "initialization_provenance": "owned by the locked server runner protocol; not inferred by this evaluator",
            "label_budget": train_data["label_budget"],
            "domain_budgets": {name: train_data["domain_budgets"][name] for name in DOMAIN_NAMES},
            "train_manifest_path": str(train_path),
            "train_manifest_sha256": actual_train_sha,
        },
        "evaluation_maps_path": str(index_path),
        "evaluation_maps_sha256": actual_maps_sha,
        "development_manifests": {
            domain: {
                "map_name": selected[domain]["map_name"],
                "path": str(selected[domain]["manifest_path"]),
                "sha256": selected[domain]["manifest_sha256"],
                "namespaced_patient_ids": selected[domain]["patient_ids"],
            }
            for domain in DOMAIN_NAMES
        },
        "namespaced_manifest_path": str(composite_path),
        "namespaced_manifest_sha256": composite_sha,
        "selection_lock_path": str(selection_lock_path),
        "selection_lock_sha256": selection_lock_sha,
        "export_manifest_sha256": metrics.get("export_manifest_sha256"),
        "export_payload_sha256": metrics.get("export_payload_sha256"),
        "prediction_freeze": {
            "path": str(frozen_path),
            "sha256": metrics.get("frozen_predictions_sha256"),
            "precedes_scoring": True,
            "coverage": metrics.get("coverage"),
        },
        "domain_results": domain_results,
        "historical_metrics_path": str(metrics_path),
        "historical_metrics_sha256": digest(metrics_path),
        "domain_report_path": str(out_root / "DOMAIN_EVALUATION.json"),
    }
    write_json(out_root / "DOMAIN_EVALUATION.json", report)
    return report


def _domain_class_counts(frame_rows: Any, domain: str) -> Dict[str, Dict[str, int]]:
    """Return per-domain pooled confusion counts as a descriptive supplement."""
    import numpy as np

    confusion = np.zeros((4, 5), dtype=np.int64)
    if not isinstance(frame_rows, list):
        raise ValueError("canonical evaluator did not return frame rows")
    for row in frame_rows:
        if not isinstance(row, Mapping) or row.get("dataset") != domain:
            continue
        cm = np.asarray(row.get("confusion"))
        if cm.shape != (4, 5) or not np.issubdtype(cm.dtype, np.integer):
            raise ValueError(f"canonical evaluator returned malformed confusion counts for {domain}")
        confusion += cm.astype(np.int64, copy=False)
    result: Dict[str, Dict[str, int]] = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        reference_count = int(confusion[index].sum())
        prediction_count = int(confusion[:, index].sum())
        result[name] = {
            "TP": tp,
            "FP": prediction_count - tp,
            "FN": reference_count - tp,
            "reference_count": reference_count,
            "prediction_count": prediction_count,
        }
    return result


__all__ = ["DOMAIN_NAMES", "DOMAIN_SELECTION_RULE", "evaluate_shared_domains"]
