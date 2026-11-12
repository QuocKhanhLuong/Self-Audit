#!/usr/bin/env python3
"""Plan and explicitly execute the Q-3D T/Q/P/J campaign.

Planning is read-only with respect to patient data and creates only a
hash-pinned campaign plan plus per-job recipe copies. Execution requires the
explicit ``--execute`` switch; real teacher runs also require validated,
hash-pinned manifests and pretrained-weight provenance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from self_audit.candidate_q3d.config import (
    ARMS,
    BASELINE_ARMS,
    DEFAULT_SEED,
    K8_LOCKED_DEV_MANIFEST_SHA256,
    K8_LOCKED_TRAIN_MANIFEST_SHA256,
    K_VALUES,
    PRIMARY_ARMS,
    SECONDARY_ARMS,
    arm_settings,
    canonical_config,
    load_config,
    resolve_stage_accounting,
    resolve_stage_updates,
    stages_for_arm,
    validate_budget_update_equality,
    validate_config,
    validate_real_teacher_provenance,
)
from self_audit.candidate_q3d.campaign_runtime import select_campaign_experiments
from self_audit.candidate_q3d.lineage import load_external_lineage, validate_external_view

PLAN_SCHEMA = "self_audit.candidate_q3d.campaign.v1"
STATE_SCHEMA = "self_audit.candidate_q3d.campaign_state.v1"
SHARED_DAG_SCHEMA = "self_audit.candidate_q3d.shared_stage_dag.v1"
DEFAULT_REPORT_DIR = ROOT / "reports/q3d-implementation-20261112/campaign"
CAMPAIGN_K_ORDER = (5, 10, 2)
SUPPORTED_K_ORDER = (8, 5, 10, 2)
PRIMARY_SETTINGS = PRIMARY_ARMS
SECONDARY_SETTINGS = SECONDARY_ARMS + ("v1_residual2",)
LEGACY_SECONDARY_SETTINGS = SECONDARY_SETTINGS
DEFAULT_REDUCED_RUNS = (
    "v1@8", "v1@5", "v1@10", "v1u@8", "c1@8", "c0@8",
)
_TEACHER_SUPERVISED = "teacher_supervised"
_EXPLICIT_RUN_SETTINGS = PRIMARY_SETTINGS + SECONDARY_SETTINGS
CAMPAIGN_K_VALUES = tuple(sorted(set(K_VALUES) | {8}))
SYNTHETIC_CONFIDENT_TEACHER_ARMS = {"c1", "v1", "v1u", "v0", "b1", "e1", "o1", "s1"}
BASELINE_LAUNCHES = {
    "mean_teacher": {
        "runner": "self_audit.candidate_q3d.engine.train",
        "entrypoint": "train.py",
        "method": "candidate_q3d",
        "arm": "mean_teacher",
        "status": "ENGINE_ARM",
    },
    "a3d_teacher_free": {
        "runner": "self_audit.candidate_q3d.baselines.run_a3d_teacher_free",
        "entrypoint": "train.py",
        "method": "candidate_q3d",
        "arm": "a3d_teacher_free",
        "template": "configs/candidate_a3d/teacher_free_acdc_k8.json",
        "update_match": "requires exact J budget from K-manifest labeled-frame count",
        "status": "IMPLEMENTED",
    },
    "teacher_supervised": {
        "runner": "self_audit.candidate_q3d.baselines.run_teacher_supervised",
        "entrypoint": "train.py",
        "method": "candidate_q3d",
        "arm": "teacher_supervised",
        "status": "IMPLEMENTED",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _manifest_ref(path_value: str | Path | None) -> dict[str, Any]:
    if path_value is None:
        return {"path": None, "sha256": None, "status": "NOT_RUN"}
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"manifest file does not exist: {path}")
    return {"path": str(path), "sha256": _sha256(path), "status": "READY"}


def _parse_manifest_flags(values: list[str], *, cross: bool = False) -> dict[str, dict[str, str]]:
    parsed: dict[str, dict[str, str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"manifest mapping must be KEY=PATH: {value}")
        key, path = value.split("=", 1)
        if not path:
            raise ValueError("manifest path cannot be empty")
        if cross:
            if key not in {"acdc", "mnms"}:
                raise ValueError("cross-evaluation targets must be named acdc or mnms")
            parsed.setdefault(key, {})["test"] = path
        else:
            if ":" not in key:
                raise ValueError("manifest key must be DATASET:ROLE or DATASET:K:ROLE")
            parts = key.split(":")
            if len(parts) == 2:
                dataset, role = parts
                manifest_key = role
            elif len(parts) == 3:
                dataset, k_text, role = parts
                try:
                    k = int(k_text)
                except ValueError as exc:
                    raise ValueError(f"manifest K must be one of {CAMPAIGN_K_VALUES}") from exc
                if k not in CAMPAIGN_K_VALUES:
                    raise ValueError(f"manifest K must be one of {CAMPAIGN_K_VALUES}")
                manifest_key = f"k{k}:{role}"
            else:
                raise ValueError("manifest key must be DATASET:ROLE or DATASET:K:ROLE")
            if dataset not in {"acdc", "mnms"} or role not in {"train", "teacher_train", "development", "test"}:
                raise ValueError("manifest key must be acdc|mnms:ROLE or acdc|mnms:K:ROLE")
            parsed.setdefault(dataset, {})[manifest_key] = path
    return parsed


def _normalize_run_specs(values: tuple[Any, ...] | list[Any]) -> tuple[tuple[str, int], ...]:
    """Parse an explicit ``setting@K`` catalogue and reject duplicates."""
    if not values:
        raise ValueError("--runs requires at least one SETTING@K pair")
    parsed: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for raw in values:
        if isinstance(raw, str):
            token = raw.strip()
            if token.count("@") != 1:
                raise ValueError(f"run must use SETTING@K syntax: {raw!r}")
            setting, k_text = (part.strip() for part in token.split("@", 1))
            try:
                k = int(k_text)
            except ValueError as exc:
                raise ValueError(f"run K must be an integer in {CAMPAIGN_K_VALUES}: {raw!r}") from exc
        elif isinstance(raw, (tuple, list)) and len(raw) == 2:
            setting, k = raw
        else:
            raise ValueError(f"run must use SETTING@K syntax: {raw!r}")
        if not isinstance(setting, str) or setting not in _EXPLICIT_RUN_SETTINGS:
            raise ValueError(f"unsupported run setting {setting!r}; choose a Q-3D student setting")
        if type(k) is not int or k not in CAMPAIGN_K_VALUES:
            raise ValueError(f"unsupported run K {k!r}; choose one of {CAMPAIGN_K_VALUES}")
        pair = (setting, k)
        if pair in seen:
            raise ValueError(f"duplicate run pair: {setting}@{k}")
        seen.add(pair)
        parsed.append(pair)
    return tuple(parsed)


def _set_manifest_inputs(config: dict[str, Any], manifest_inputs: Mapping[str, Mapping[str, str]],
                         cross_inputs: Mapping[str, Mapping[str, str]],
                         *, enforce_k8_lock: bool = False) -> None:
    dataset_name = config["dataset"]["name"]
    k = config["protocol"]["k"]
    roles = manifest_inputs.get(dataset_name, {})
    for role in ("train", "teacher_train", "development", "test"):
        specific_key = f"k{k}:{role}"
        has_k_specific_train = any(key.startswith("k") and key.endswith(":train") for key in roles)
        path_value = roles.get(specific_key)
        if path_value is None and not (role == "train" and has_k_specific_train):
            path_value = roles.get(role)
        if path_value is None:
            continue
        reference = _manifest_ref(path_value)
        if enforce_k8_lock and dataset_name == "acdc" and k == 8:
            expected_sha = {"train": K8_LOCKED_TRAIN_MANIFEST_SHA256,
                            "development": K8_LOCKED_DEV_MANIFEST_SHA256}.get(role)
            if expected_sha is not None and reference["sha256"] != expected_sha:
                raise ValueError(f"ACDC K=8 {role.upper()} manifest SHA-256 does not match the locked campaign input")
        if enforce_k8_lock and role == "train" and specific_key not in roles and path_value == roles.get("train"):
            payload = _read_json(Path(reference["path"]))
            declared_budget = payload.get("label_budget")
            if type(declared_budget) is int and declared_budget != k:
                raise ValueError(
                    f"global TRAIN manifest label_budget={declared_budget} cannot serve K={k}; "
                    "provide DATASET:K:train explicitly"
                )
        config["dataset"]["manifests"][role] = reference
    known = {item["dataset"]: item for item in config["dataset"]["cross_evaluation"][:2]}
    for target, roles in cross_inputs.items():
        if "test" in roles:
            known[target]["manifest"] = _manifest_ref(roles["test"])
    if config["dataset"]["manifests"]["test"]["status"] == "READY":
        known[dataset_name]["manifest"] = copy.deepcopy(config["dataset"]["manifests"]["test"])


def _validate_nested_train_lineage(
    manifest_inputs: Mapping[str, Mapping[str, str]],
    *, datasets: tuple[str, ...], selected_ks: tuple[int, ...], seed: int,
    lineage_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed when K-specific selectors do not document one nested cohort."""
    audit: dict[str, Any] = {
        "status": "PASS",
        "schema_loader_status": "READY",
        "records": [],
    }
    for dataset in datasets:
        external = (load_external_lineage(lineage_reference, dataset=dataset, seed=seed)
                    if lineage_reference is not None else None)
        roles = manifest_inputs.get(dataset, {})
        selected_by_k: dict[int, set[str]] = {}
        source_hashes: set[str] = set()
        for k in sorted(selected_ks):
            path_value = roles.get(f"k{k}:train")
            if path_value is None:
                if audit["status"] == "PASS":
                    audit["status"] = "PARTIAL_NOT_RUN"
                audit["records"].append({
                    "dataset": dataset,
                    "k": k,
                    "path": None,
                    "schema": None,
                    "role": "training",
                    "status": "NOT_RUN",
                    "reason": "K-specific TRAIN manifest was not supplied; nesting was not inferred",
                })
                continue
            path = Path(path_value).expanduser().resolve()
            if dataset == "acdc" and k == 8:
                if _manifest_ref(path)["sha256"] != K8_LOCKED_TRAIN_MANIFEST_SHA256:
                    raise ValueError("ACDC K=8 TRAIN manifest SHA-256 does not match the locked campaign input")
            payload = _read_json(path)
            schema = payload.get("schema")
            role = payload.get("role")
            record = {"dataset": dataset, "k": k, "path": str(path), "schema": schema, "role": role}
            audit["records"].append(record)
            if schema != "self_audit.data.v2" or role != "training":
                audit["status"] = "NOT_RUN"
                audit["schema_loader_status"] = "NOT_RUN"
                record["status"] = "NOT_RUN"
                record["reason"] = (
                    "manifest schema/role is not accepted by the Q3D data loader; "
                    "locked bytes were retained without transformation"
                )
                continue
            record["status"] = "READY"
            lineage = payload.get("fewshot")
            if external is not None:
                verified = validate_external_view(payload, k=k, context=external)
                record["external_lineage_evidence"] = verified["external_evidence"]
                record["selector_validation"] = verified["validation"]
                if not isinstance(lineage, Mapping):
                    lineage = verified
            if not isinstance(lineage, Mapping):
                # The locked K=8 artifact is immutable campaign evidence. If an
                # older copy lacks lineage metadata, retain its exact bytes and
                # report the cohort relationship as unknown rather than
                # synthesising metadata or silently selecting patients again.
                if (dataset == "acdc" and k == 8
                        and _manifest_ref(path)["sha256"] == K8_LOCKED_TRAIN_MANIFEST_SHA256):
                    audit["status"] = "PARTIAL_NOT_RUN" if audit["status"] == "PASS" else audit["status"]
                    record["status"] = "NOT_RUN"
                    record["reason"] = (
                        "locked K=8 manifest has no few-shot lineage metadata; "
                        "nested cohort relation was not inferred"
                    )
                    continue
                raise ValueError(f"K={k} TRAIN manifest is missing few-shot lineage metadata")
            if payload.get("label_budget") != k or lineage.get("k") != k or lineage.get("seed") != seed:
                raise ValueError(f"K={k} TRAIN manifest label budget/seed does not match the selector")
            selected = lineage.get("selected_ids")
            if (not isinstance(selected, list) or len(selected) != k or len(selected) != len(set(selected))
                    or any(not isinstance(item, str) or not item for item in selected)):
                raise ValueError(f"K={k} TRAIN manifest has invalid selected_ids lineage")
            labeled = payload.get("labeled")
            labeled_ids = [item.get("id") for item in labeled
                           if isinstance(item, Mapping) and isinstance(item.get("id"), str)] \
                if isinstance(labeled, list) else []
            if (len(labeled_ids) != len(labeled or [])
                    or set(labeled_ids) != set(selected)):
                raise ValueError(
                    f"K={k} TRAIN manifest labeled IDs do not match few-shot selected_ids lineage"
                )
            selected_by_k[k] = set(selected)
            source_sha = lineage.get("source_sha256")
            if not isinstance(source_sha, str) or len(source_sha) != 64:
                raise ValueError(f"K={k} TRAIN manifest has no source manifest SHA-256 lineage")
            source_hashes.add(source_sha)
        if len(source_hashes) > 1:
            raise ValueError(f"{dataset} K-specific TRAIN manifests do not share one source manifest")
        ordered = sorted(selected_by_k)
        for lower, upper in zip(ordered, ordered[1:]):
            if not selected_by_k[lower].issubset(selected_by_k[upper]):
                raise ValueError(f"{dataset} few-shot TRAIN cohorts are not nested: K={lower} is not subset of K={upper}")
    return audit


def _validate_reduced_development_pins(
    manifest_inputs: Mapping[str, Mapping[str, str]], *, datasets: tuple[str, ...],
    protocols: tuple[str, ...], selected_ks: tuple[int, ...], seed: int,
) -> None:
    """Keep the reduced ACDC K budgets on one existing VAL manifest.

    The K=8 development hash is the only locked VAL byte available to the
    campaign.  K=5/K=10 may reuse that exact file, but no new VAL hash or
    patient selection is invented here.  Missing inputs remain NOT_RUN.
    """

    if seed != DEFAULT_SEED or "acdc" not in datasets or "A" not in protocols:
        return
    roles = manifest_inputs.get("acdc", {})
    for k in selected_ks:
        if k not in {5, 8, 10}:
            continue
        path_value = roles.get(f"k{k}:development") or roles.get("development")
        if path_value is None:
            continue
        reference = _manifest_ref(path_value)
        if reference["sha256"] != K8_LOCKED_DEV_MANIFEST_SHA256:
            raise ValueError(
                f"ACDC reduced K={k} VALIDATION must reuse the locked K=8 manifest "
                "SHA-256; no alternative VAL bytes may be introduced"
            )


def _set_teacher_provenance(config: dict[str, Any], provenance: Mapping[str, Any] | None) -> None:
    source = config["model"]["teacher"]["pretrained"]
    if source is None or provenance is None:
        return
    source.update(copy.deepcopy(dict(provenance)))
    source["class_mapping"] = list(config["dataset"]["class_mapping"])
    if all(source.get(key) is not None for key in (
        "weights_path", "weights_sha256", "weights_source_url", "pretraining_dataset",
        "overlap_acdc", "overlap_mnms",
    )):
        source["provenance_status"] = "READY"


def _read_training_manifest(config: Mapping[str, Any]) -> dict[str, Any] | None:
    record = config["dataset"]["manifests"]["train"]
    if record["status"] != "READY":
        return None
    path = Path(record["path"])
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"training manifest SHA-256 changed: {path}")
    manifest = _read_json(path)
    if manifest.get("schema") != "self_audit.data.v2" or manifest.get("role") != "training":
        raise ValueError("Q-3D training input must be a self_audit.data.v2 training manifest")
    if manifest.get("dataset") != config["dataset"]["name"]:
        raise ValueError("training manifest dataset differs from the config")
    if manifest.get("label_budget") != config["protocol"]["k"]:
        raise ValueError("student TRAIN manifest label_budget must equal configured K")
    return manifest


def _manifest_loader_block_reason(
    record: Mapping[str, Any], *, expected_role: str, expected_k: int | None = None,
    lineage_context: Mapping[str, Any] | None = None,
) -> str | None:
    """Report schema/loader incompatibility without rewriting a pinned file."""

    if record.get("status") != "READY":
        return None
    path_value = record.get("path")
    if not isinstance(path_value, str):
        return "manifest path is not a readable file"
    try:
        manifest = _read_json(Path(path_value))
    except (OSError, ValueError) as exc:
        return f"manifest cannot be read by the Q3D loader: {exc}"
    if manifest.get("schema") != "self_audit.data.v2" or manifest.get("role") != expected_role:
        return (
            "manifest schema/role is not accepted by the Q3D data loader "
            f"(schema={manifest.get('schema')!r}, role={manifest.get('role')!r}, "
            f"expected_schema='self_audit.data.v2', expected_role={expected_role!r}); "
            "pinned bytes were retained without transformation"
        )
    if expected_role == "training":
        if expected_k is not None and manifest.get("label_budget") != expected_k:
            return (
                "training manifest label_budget does not match configured K "
                f"(label_budget={manifest.get('label_budget')!r}, K={expected_k})"
            )
        lineage = manifest.get("fewshot")
        if lineage_context is not None:
            try:
                verified = validate_external_view(manifest, k=expected_k, context=lineage_context)
            except ValueError as exc:
                return f"external few-shot lineage is invalid: {exc}"
            if not isinstance(lineage, Mapping):
                lineage = verified
        if not isinstance(lineage, Mapping):
            return "training manifest is missing few-shot lineage metadata"
        selected = lineage.get("selected_ids")
        labeled = manifest.get("labeled")
        labeled_ids = [item.get("id") for item in labeled
                       if isinstance(item, Mapping) and isinstance(item.get("id"), str)] \
            if isinstance(labeled, list) else []
        if (not isinstance(selected, list) or not selected
                or len(selected) != len(set(selected))
                or any(not isinstance(item, str) or not item for item in selected)
                or (expected_k is not None and len(selected) != expected_k)
                or len(labeled_ids) != len(labeled or [])
                or set(labeled_ids) != set(selected)):
            return "training manifest labeled IDs do not match few-shot selected_ids lineage"
    return None


def _a3d_labeled_frame_count(manifest: Mapping[str, Any]) -> int:
    labeled = manifest.get("labeled")
    if not isinstance(labeled, list) or not labeled:
        raise ValueError("A-3D teacher_free needs a nonempty labeled TRAIN list")
    count = 0
    for record in labeled:
        masks = record.get("masks") if isinstance(record, Mapping) else None
        if not isinstance(masks, list) or not masks:
            raise ValueError("A-3D labeled patient record must declare annotated frame masks")
        count += len(masks)
    return count


def _setting_recipe(setting: str, *, dataset: str, protocol: str, k: int, seed: int,
                    smoke: bool, smoke_updates: int | None) -> dict[str, Any]:
    arm = "v1" if setting == "v1_residual2" else setting
    source_stride = 16 if setting == "s1" else 8
    config = canonical_config(
        arm, dataset=dataset, protocol=protocol, k=k, seed=seed,
        source_stride=source_stride,
        residual2=setting == "v1_residual2",
    )
    if smoke:
        config["runtime"]["synthetic"] = True
        config["runtime"]["device"] = "cpu"
        if config["tokenizer"]["enabled"]:
            config["runtime"]["smoke_initialization_volumes"] = min(199, max(8, k + 2))
        config["runtime"]["synthetic_gate_override"] = True
        if smoke_updates is not None:
            config["runtime"]["smoke_update_overrides"] = {
                "T": smoke_updates, "Q": smoke_updates, "P": 1, "J": smoke_updates,
            }
            if setting == "a3d_teacher_free":
                # The synthetic A-3D fixture exposes two annotated frames per K patient.
                config["runtime"]["smoke_update_overrides"]["J"] = 2 * k
    return config


def _experiment_id(dataset: str, protocol: str, k: int, seed: int, setting: str) -> str:
    return f"{dataset}_{protocol.lower()}_k{k}_seed{seed}_{setting}"


def _read_a3d_template(*, dataset: str, seed: int, train_manifest: Mapping[str, Any] | None,
                       target_updates: int, synthetic_frame_count: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    template_path = ROOT / "configs/candidate_a3d/teacher_free_acdc_k8.json"
    a3d = _read_json(template_path)
    a3d["dataset"]["name"] = dataset
    a3d["seed"] = seed
    frame_count = None
    update_match = False
    reason = "training manifest not configured"
    if synthetic_frame_count is not None:
        frame_count = synthetic_frame_count
        if target_updates % frame_count:
            reason = f"synthetic target {target_updates} updates is not divisible by {frame_count} labeled frames"
        else:
            a3d["epochs"] = target_updates // frame_count
            update_match = a3d["epochs"] * frame_count == target_updates
            reason = "synthetic update match" if update_match else "synthetic update mismatch"
    if train_manifest is not None:
        frame_count = _a3d_labeled_frame_count(train_manifest)
        if target_updates % frame_count:
            reason = f"target {target_updates} updates is not divisible by {frame_count} labeled frames"
        else:
            a3d["epochs"] = target_updates // frame_count
            update_match = a3d["epochs"] * frame_count == target_updates
            reason = "exact update match" if update_match else "update mismatch"
    launch = {
        **copy.deepcopy(BASELINE_LAUNCHES["a3d_teacher_free"]),
        "template_path": str(template_path),
        "resolved_epochs": a3d["epochs"],
        "labeled_frame_count": frame_count,
        "target_student_updates": target_updates,
        "realized_student_updates": (a3d["epochs"] * frame_count) if frame_count is not None else None,
        "update_match": update_match,
        "update_match_reason": reason,
    }
    return a3d, launch


def _job_readiness(config: Mapping[str, Any], *, synthetic: bool, stage: str,
                   a3d_update_match: bool | None = None,
                   lineage_reference: Mapping[str, Any] | None = None) -> tuple[str, str | None]:
    if synthetic:
        if config["arm"] == "a3d_teacher_free" and a3d_update_match is not True:
            return "NOT_RUN", "A-3D teacher_free requires an exact synthetic J budget"
        return "SMOKE_READY", None
    teacher = config["model"]["teacher"]
    lineage_context = (load_external_lineage(
        lineage_reference, dataset=config["dataset"]["name"], seed=config["seed"],
    ) if lineage_reference is not None else None)
    for role in ("train", "development"):
        if config["dataset"]["manifests"][role]["status"] != "READY":
            return "NOT_RUN", f"{role} manifest path and SHA-256 are not configured"
        loader_reason = _manifest_loader_block_reason(
            config["dataset"]["manifests"][role],
            expected_role="training" if role == "train" else "development",
            expected_k=int(config["protocol"]["k"]) if role == "train" else None,
            lineage_context=lineage_context if role == "train" else None,
        )
        if loader_reason is not None:
            return "NOT_RUN", f"{role} manifest loader contract is not ready: {loader_reason}"
    if (stage == "T" and config["protocol"]["variant"] == "B"
            and config["arm"] != _TEACHER_SUPERVISED
            and config["dataset"]["manifests"]["teacher_train"]["status"] != "READY"):
        return "NOT_RUN", "Protocol B teacher training requires a separate all-TRAIN manifest"
    if config["arm"] == "a3d_teacher_free" and a3d_update_match is not True:
        return "NOT_RUN", "A-3D teacher_free labeled-frame count cannot match the declared J budget exactly"
    if stage in {"T", "Q", "P", "J"} and config["model"]["teacher"].get("architecture") not in {
        None, "none", "ema_student", "candidate_a3d",
    }:
        try:
            validate_real_teacher_provenance(config)
        except (OSError, TypeError, ValueError) as exc:
            return "NOT_RUN", f"pretrained teacher provenance is not ready: {exc}"
    return "READY", None


def _add_experiment(plan: dict[str, Any], *, setting: str, dataset: str, protocol: str, k: int,
                    seed: int, out_root: Path, manifest_inputs: Mapping[str, Mapping[str, str]],
                    cross_inputs: Mapping[str, Mapping[str, str]], teacher_provenance: Mapping[str, Any] | None,
                    synthetic: bool, smoke_updates: int | None,
                    enforce_k8_lock: bool = False) -> None:
    experiment_id = _experiment_id(dataset, protocol, k, seed, setting)
    config = _setting_recipe(setting, dataset=dataset, protocol=protocol, k=k, seed=seed,
                             smoke=synthetic, smoke_updates=smoke_updates)
    _set_manifest_inputs(config, manifest_inputs, cross_inputs, enforce_k8_lock=enforce_k8_lock)
    _set_teacher_provenance(config, teacher_provenance)
    config = validate_config(config)
    config_path = out_root / "recipes" / f"{experiment_id}.json"
    _write_json(config_path, config)
    config_sha = _sha256(config_path)
    settings = arm_settings(config["arm"])
    stage_sequence = list(stages_for_arm(config["arm"]))
    stage_accounting = resolve_stage_accounting(config, smoke=synthetic)
    target_updates = stage_accounting["stage_updates"]["J"]
    a3d_template_path = None
    a3d_launch = None
    a3d_update_match = None
    if setting == "a3d_teacher_free":
        train_manifest = _read_training_manifest(config)
        _, a3d_launch = _read_a3d_template(
            dataset=dataset, seed=seed, train_manifest=train_manifest, target_updates=target_updates,
            synthetic_frame_count=(2 * k if synthetic and train_manifest is None else None),
        )
        a3d_update_match = a3d_launch["update_match"]
    experiment = {
        "id": experiment_id,
        "dataset": dataset,
        "protocol": protocol,
        "k": k,
        "seed": seed,
        "setting": setting,
        "arm": config["arm"],
        "group": ("primary" if setting in PRIMARY_SETTINGS else
                  "secondary" if setting in SECONDARY_SETTINGS else "baseline"),
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "candidate_a3d_config_path": None,
        "stages": stage_sequence,
        "student_joint_updates": target_updates,
        "student_joint_update_unit": config["training"]["stage_update_units"]["J"],
        "student_joint_optimizer_updates": stage_accounting["optimizer_updates"]["J"],
        "student_joint_stream_microbatches": stage_accounting["stream_microbatches"]["J"],
        "output": str((out_root / "runs" / experiment_id).resolve()),
        "interface": settings["interface"],
        "teacher_signal": settings["teacher_signal"],
        "physiology_weighting": settings["physiology"],
        "launch_mapping": a3d_launch or copy.deepcopy(BASELINE_LAUNCHES.get(setting)),
    }
    plan["experiments"].append(experiment)
    previous_job = None
    for stage in stage_sequence:
        stage_job_id = f"{experiment_id}.{stage.lower()}"
        dependencies = [previous_job] if previous_job else []
        updates = stage_accounting["stage_updates"][stage]
        if setting == "a3d_teacher_free" and synthetic and stage == "J":
            updates = target_updates
        optimizer_updates = stage_accounting["optimizer_updates"][stage]
        stream_microbatches = stage_accounting["stream_microbatches"][stage]
        if setting == "a3d_teacher_free" and synthetic and stage == "J":
            optimizer_updates = updates
            stream_microbatches = 2 * updates
        manifest_role = "teacher_train" if stage == "T" and protocol == "B" else "train"
        train_ref = config["dataset"]["manifests"][manifest_role]
        if stage == "T" and protocol == "A":
            train_ref = config["dataset"]["manifests"]["train"]
        readiness, reason = _job_readiness(config, synthetic=synthetic, stage=stage,
                                           a3d_update_match=a3d_update_match,
                                           lineage_reference=plan.get("external_lineage_reference"))
        stage_job = {
            "id": stage_job_id,
            "experiment_id": experiment_id,
            "setting": setting,
            "arm": config["arm"],
            "dataset": dataset,
            "protocol": protocol,
            "k": k,
            "seed": seed,
            "group": experiment["group"],
            "method": "candidate_q3d",
            "stage": stage,
            "mode": config["training"]["stage_modes"][stage],
            "updates": None if stage == "P" else updates,
            "update_unit": config["training"]["stage_update_units"][stage],
            "optimizer_updates": optimizer_updates,
            "stream_microbatches": stream_microbatches,
            "forward_passes": 1 if stage == "P" else None,
            "budget_unit": config["training"]["stage_update_units"][stage],
            "dependencies": dependencies,
            "train_manifest": copy.deepcopy(train_ref),
            "development_manifest": copy.deepcopy(config["dataset"]["manifests"]["development"]),
            "config_path": str(config_path.resolve()),
            "config_sha256": config_sha,
            "candidate_a3d_config_path": None,
            "output": str((out_root / "runs" / experiment_id).resolve()),
            "readiness": readiness,
            "reason": reason,
            "status": "NOT_RUN",
        }
        if setting in BASELINE_ARMS:
            stage_job["launch_mapping"] = copy.deepcopy(a3d_launch or BASELINE_LAUNCHES[setting])
        plan["stage_jobs"].append(stage_job)
        previous_job = stage_job_id


def _stage_artifact_identity(experiment: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": experiment["dataset"],
        "protocol": experiment["protocol"],
        "k": experiment["k"],
        "seed": experiment["seed"],
    }


def _artifact_source(config: Mapping[str, Any], setting: str) -> tuple[str, int, str]:
    teacher = config["model"]["teacher"]
    source = teacher.get("feature_source")
    stride = teacher.get("feature_stride")
    if not isinstance(source, str) or type(stride) is not int:
        raise ValueError(f"{setting} must declare a teacher feature source and stride before sharing")
    source_variant = "o1_online" if setting == "o1" else "frozen_teacher"
    return source, stride, source_variant


def _stage_source_config(
    stage: str,
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    source: str | None = None,
    stride: int | None = None,
    source_variant: str | None = None,
) -> dict[str, Any]:
    from self_audit.candidate_q3d.stages import stage_source_config

    contract = stage_source_config(config, stage)
    if contract.get("identity") != dict(identity):
        raise ValueError(f"{stage} stage source contract identity differs from its plan")
    if stage in {"Q", "P"} and (
        contract.get("teacher_feature_source") != source
        or contract.get("teacher_feature_stride") != stride
        or contract.get("teacher_variant") != source_variant
    ):
        raise ValueError(f"{stage} stage source contract differs from its feature-source key")
    return contract


def _identity_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(identity[key] for key in ("dataset", "protocol", "k", "seed"))


def _identity_slug(identity: Mapping[str, Any]) -> str:
    return f"{identity['dataset']}_{str(identity['protocol']).lower()}_k{identity['k']}_seed{identity['seed']}"


def _source_slug(source: str, stride: int, source_variant: str) -> str:
    value = f"{source}_s{stride}_{source_variant}"
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _shared_producer_id(
    stage: str,
    identity: Mapping[str, Any],
    source_key: tuple[str, int, str] | None = None,
) -> str:
    suffix = "teacher" if stage == "T" else _source_slug(*source_key) if source_key else ""
    return f"shared.{_identity_slug(identity)}.{suffix}.{stage.lower()}"


def _pending_manifest_ref(job: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = job["artifact_output"]["manifest_path"]
    return {
        "stage": job["stage"],
        "producer_job_id": job["id"],
        "path": manifest_path,
        "sha256": None,
        "status": "NOT_RUN",
    }


def _shared_stage_dag(plan: dict[str, Any], out_root: Path) -> None:
    """Replace repeated arm-local T/Q/P work with content-addressed producers."""

    old_jobs = list(plan["stage_jobs"])
    old_by_experiment_stage = {
        (job["experiment_id"], job["stage"]): job for job in old_jobs
    }
    configs = {experiment["id"]: load_config(experiment["config_path"]) for experiment in plan["experiments"]}
    identities = {
        experiment["id"]: _stage_artifact_identity(experiment, configs[experiment["id"]])
        for experiment in plan["experiments"]
    }
    groups: dict[tuple[str, tuple[Any, ...], tuple[str, int, str] | None], list[dict[str, Any]]] = {}
    teacher_supervised_consumers: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    source_by_experiment: dict[str, tuple[str, int, str]] = {}

    for experiment in plan["experiments"]:
        config = configs[experiment["id"]]
        identity = identities[experiment["id"]]
        id_key = _identity_key(identity)
        if experiment["setting"] == _TEACHER_SUPERVISED:
            teacher_supervised_consumers.setdefault(id_key, []).append(experiment)
            continue
        teacher_architecture = config["model"]["teacher"].get("architecture")
        if teacher_architecture in {None, "none", "ema_student", "candidate_a3d"}:
            continue
        source_key = _artifact_source(config, experiment["setting"])
        source_by_experiment[experiment["id"]] = source_key
        for stage in experiment["stages"]:
            if stage == "T":
                groups.setdefault((stage, id_key, None), []).append(experiment)
            elif stage in {"Q", "P"}:
                groups.setdefault((stage, id_key, source_key), []).append(experiment)

    def producer_id(stage: str, experiment: Mapping[str, Any]) -> str:
        identity = identities[experiment["id"]]
        source_key = source_by_experiment.get(experiment["id"]) if stage in {"Q", "P"} else None
        return _shared_producer_id(stage, identity, source_key)

    producer_ids: dict[tuple[str, tuple[Any, ...], tuple[str, int, str] | None], str] = {
        key: _shared_producer_id(key[0], identities[members[0]["id"]], key[2])
        for key, members in groups.items()
    }
    producers: dict[str, dict[str, Any]] = {}
    producer_for_experiment: dict[str, dict[str, str]] = {item["id"]: {} for item in plan["experiments"]}
    source_contract_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def source_contract(stage: str, experiment: Mapping[str, Any]) -> dict[str, Any]:
        key = (experiment["id"], stage)
        if key not in source_contract_cache:
            config = configs[experiment["id"]]
            source_key = source_by_experiment.get(experiment["id"]) if stage in {"Q", "P"} else None
            source, stride, variant = source_key if source_key else (None, None, None)
            source_contract_cache[key] = _stage_source_config(
                stage, config, identities[experiment["id"]],
                source=source, stride=stride, source_variant=variant,
            )
        return source_contract_cache[key]

    def preferred_anchor(members: list[dict[str, Any]]) -> dict[str, Any]:
        priority = {"v1": 0, "c1": 1, "v1u": 2}
        return sorted(members, key=lambda item: (priority.get(item["setting"], 10), item["setting"]))[0]

    for (stage, id_key, source_key), members in sorted(
        groups.items(),
        key=lambda item: (
            ("T", "Q", "P").index(item[0][0]),
            item[0][1],
            item[0][2] or ("", 0, ""),
        ),
    ):
        anchor = preferred_anchor(members)
        identity = identities[anchor["id"]]
        config = configs[anchor["id"]]
        source, stride, source_variant = source_key if source_key else (None, None, None)
        source_config = source_contract(stage, anchor)
        signatures = [source_contract(stage, member) for member in members]
        if any(signature != source_config for signature in signatures):
            raise ValueError(f"cannot share {stage} across incompatible stage source configurations")

        job_id = producer_ids[(stage, id_key, source_key)]
        original = old_by_experiment_stage[(anchor["id"], stage)]
        output = (out_root / "runs" / "shared" / job_id).resolve()
        manifest_path = output / f"{stage}.artifact.json"
        dependencies: list[str] = []
        input_stages: list[str] = []
        if stage == "Q":
            dependencies.append(producer_ids[("T", id_key, None)])
            input_stages.append("T")
        elif stage == "P":
            dependencies.append(producer_ids[("T", id_key, None)])
            input_stages.append("T")
            q_key = ("Q", id_key, source_key)
            if q_key in producer_ids:
                dependencies.append(producer_ids[q_key])
                input_stages.append("Q")
        readiness, reason = _job_readiness(
            config, synthetic=bool(plan["selection"]["synthetic"]), stage=stage,
            lineage_reference=plan.get("external_lineage_reference"),
        )
        stage_job = copy.deepcopy(original)
        stage_job.update({
            "id": job_id,
            "experiment_id": anchor["id"],
            "experiment_ids": [member["id"] for member in members],
            "consumer_settings": [member["setting"] for member in members],
            "setting": anchor["setting"],
            "group": "shared_stage_producer",
            "kind": "shared_stage_producer",
            "stage": stage,
            "dependencies": dependencies,
            "stage_artifact_inputs": {},
            "output": str(output),
            "readiness": readiness,
            "reason": reason,
            "status": "NOT_RUN",
            "artifact_identity": identity,
            "source_config": source_config,
            "source_key": None if source_key is None else {
                "feature_source": source,
                "feature_stride": stride,
                "teacher_variant": source_variant,
            },
            "artifact_output": {
                "stage": stage,
                "manifest_path": str(manifest_path),
                "manifest_sha256": None,
                "status": "NOT_RUN",
            },
        })
        producers[job_id] = stage_job
        for input_stage in input_stages:
            upstream_id = producer_ids[(input_stage, id_key, None if input_stage == "T" else source_key)]
            stage_job["stage_artifact_inputs"][input_stage] = _pending_manifest_ref(producers[upstream_id])
        for member in members:
            producer_for_experiment[member["id"]][stage] = job_id

        if stage == "T":
            for baseline in teacher_supervised_consumers.get(id_key, []):
                producer_for_experiment[baseline["id"]]["T"] = job_id
                stage_job["experiment_ids"].append(baseline["id"])
                stage_job["consumer_settings"].append(baseline["setting"])

    # A single shared T stage is the only supervised teacher training run per
    # dataset/protocol/K/seed. The named supervised baseline reuses its region
    # head; it does not schedule a second teacher training job.
    for baseline in plan["experiments"]:
        if baseline["setting"] != _TEACHER_SUPERVISED:
            continue
        t_job_id = producer_for_experiment[baseline["id"]].get("T")
        if t_job_id is None:
            raise ValueError(f"teacher-supervised baseline has no shared T producer: {baseline['id']}")
        baseline["baseline_reuse"] = {
            "kind": "shared_q3d_teacher_region_head",
            "source_stage_job_id": t_job_id,
            "independent_teacher_training": False,
            "independent_baseline": False,
            "status": "NOT_RUN",
            "scope": "Shared Q3D T region head reused for baseline export/evaluation; not an independent teacher training result.",
            "protocol_b_note": (
                "Shared T uses the all-TRAIN teacher view under Protocol B and does not establish "
                "the independent K-labeled teacher baseline."
            ),
        }
        baseline["launch_mapping"] = {
            "runner": "shared Q3D T artifact export/evaluation",
            "entrypoint": "evaluate.py",
            "method": "candidate_q3d",
            "arm": _TEACHER_SUPERVISED,
            "source_stage_job_id": t_job_id,
            "status": "REUSE_SHARED_T_NOT_RUN",
        }

    joint_jobs = []
    for experiment in plan["experiments"]:
        experiment_id = experiment["id"]
        config = configs[experiment_id]
        if "J" in experiment["stages"]:
            job = copy.deepcopy(old_by_experiment_stage[(experiment_id, "J")])
            input_stages = [stage for stage in ("T", "Q", "P") if stage in experiment["stages"]]
            optional_inputs: list[str] = []
            if experiment["setting"] == "c1":
                # C1 has no tokenizer module, but pins the shared Q manifest
                # as optional provenance for the P cache it consumes.
                input_stages.append("Q")
                optional_inputs.append("Q")
            input_refs = {}
            for stage in input_stages:
                upstream_id = producer_for_experiment[experiment_id].get(stage)
                if upstream_id is None and stage == "Q" and experiment["setting"] == "c1":
                    source_key = source_by_experiment[experiment_id]
                    upstream_id = producer_ids.get(("Q", _identity_key(identities[experiment_id]), source_key))
                if upstream_id is None:
                    raise ValueError(f"{experiment_id} lacks a shared {stage} producer")
                if stage in optional_inputs:
                    producer_for_experiment[experiment_id][stage] = upstream_id
                reference = _pending_manifest_ref(producers[upstream_id])
                if stage in optional_inputs:
                    reference["role"] = "optional_source_metadata"
                input_refs[stage] = reference
            job.update({
                "kind": "student_joint",
                "experiment_ids": [experiment_id],
                "dependencies": [ref["producer_job_id"] for ref in input_refs.values()],
                "stage_artifact_inputs": input_refs,
                "optional_stage_artifact_inputs": optional_inputs,
                "artifact_identity": identities[experiment_id],
                "status": "NOT_RUN",
            })
            joint_jobs.append(job)
            producer_for_experiment[experiment_id]["J"] = job["id"]
        experiment["stage_artifact_producers"] = dict(producer_for_experiment[experiment_id])
        experiment["stage_job_ids"] = [
            producer_for_experiment[experiment_id][stage] for stage in experiment["stages"]
        ]
        experiment["expected_stage_artifacts"] = {
            stage: _pending_manifest_ref(producers[producer_for_experiment[experiment_id][stage]])
            for stage in ("T", "Q", "P")
            if stage in producer_for_experiment[experiment_id]
        }

    all_jobs = list(producers.values()) + joint_jobs
    group_order = {name: index for index, name in enumerate(plan["ordering"]["group_order"])}
    dataset_order = {name: index for index, name in enumerate(plan["ordering"]["dataset_order"])}
    k_order = {value: index for index, value in enumerate(plan["ordering"]["k_order"])}
    stage_order = {name: index for index, name in enumerate(("T", "Q", "P", "J"))}
    all_jobs.sort(key=lambda job: (
        stage_order[job["stage"]],
        dataset_order[job["dataset"]],
        job["protocol"],
        k_order[job["k"]],
        job.get("source_key", {}).get("feature_source", "") if isinstance(job.get("source_key"), Mapping) else "",
        job.get("source_key", {}).get("teacher_variant", "") if isinstance(job.get("source_key"), Mapping) else "",
        group_order.get(job.get("group"), len(group_order)),
        job["id"],
    ))
    plan["stage_jobs"] = all_jobs

    by_setting_key = {
        (experiment["dataset"], experiment["protocol"], experiment["k"], experiment["seed"], experiment["setting"]): experiment
        for experiment in plan["experiments"]
    }
    shared_pairs = []
    source_ablation_refs = []
    for key, v1 in list(by_setting_key.items()):
        if key[-1] != "v1":
            continue
        identity_tuple = key[:-1]
        v1u = by_setting_key.get((*identity_tuple, "v1u"))
        if v1u is None:
            continue
        shared_pairs.append({
            "identity": {"dataset": key[0], "protocol": key[1], "k": key[2], "seed": key[3]},
            "arms": ["v1", "v1u"],
            "producer_job_ids": {stage: v1["stage_artifact_producers"][stage] for stage in ("T", "Q", "P")},
            "consumer_job_ids": {"v1": v1["stage_artifact_producers"]["J"], "v1u": v1u["stage_artifact_producers"]["J"]},
            "manifest_sha256": {stage: None for stage in ("T", "Q", "P")},
            "status": "NOT_RUN",
        })
        for variant in ("o1", "s1"):
            other = by_setting_key.get((*identity_tuple, variant))
            if other is None:
                continue
            source_ablation_refs.append({
                "identity": {"dataset": key[0], "protocol": key[1], "k": key[2], "seed": key[3]},
                "setting": variant,
                "shared_T_producer_job_id": v1["stage_artifact_producers"]["T"],
                "distinct_Q_producer_job_id": other["stage_artifact_producers"]["Q"],
                "distinct_P_producer_job_id": other["stage_artifact_producers"]["P"],
                "feature_source": configs[other["id"]]["model"]["teacher"]["feature_source"],
                "feature_stride": configs[other["id"]]["model"]["teacher"]["feature_stride"],
                "status": "NOT_RUN",
            })

    shared_consumer_references = []
    for key, v1 in list(by_setting_key.items()):
        if key[-1] != "v1":
            continue
        identity_tuple = key[:-1]
        canonical_refs = {
            stage: v1["stage_artifact_producers"][stage] for stage in ("T", "Q", "P")
        }
        consumer_job_ids = {}
        for setting in ("c1", "v1", "v1u", "e1"):
            consumer = by_setting_key.get((*identity_tuple, setting))
            if consumer is None:
                continue
            refs = consumer["stage_artifact_producers"]
            if any(refs.get(stage) != canonical_refs[stage] for stage in ("T", "Q", "P")):
                raise ValueError("C1/V1/V1-u/E1 must consume the same shared T/Q/P producers")
            consumer_job_ids[setting] = refs["J"]
        if consumer_job_ids:
            shared_consumer_references.append({
                "identity": {"dataset": key[0], "protocol": key[1], "k": key[2], "seed": key[3]},
                "producer_job_ids": canonical_refs,
                "consumer_job_ids": consumer_job_ids,
                "consumer_settings": list(consumer_job_ids),
                "status": "NOT_RUN",
            })

    stage_counts = {stage: sum(job["stage"] == stage for job in all_jobs) for stage in ("T", "Q", "P", "J")}
    producer_counts = {stage: sum(job.get("kind") == "shared_stage_producer" and job["stage"] == stage
                                  for job in all_jobs) for stage in ("T", "Q", "P")}
    plan["shared_stage_dag"] = {
        "schema": SHARED_DAG_SCHEMA,
        "status": "PLAN_ONLY_NOT_RUN",
        "stage_job_counts": stage_counts,
        "shared_producer_counts": producer_counts,
        "expected_v1_v1u_shared_references": shared_pairs,
        "expected_v1_v1u_shared_reference_count": len(shared_pairs),
        "shared_consumer_references": shared_consumer_references,
        "source_ablation_references": source_ablation_refs,
        "teacher_supervised_reuse": "Shared Q3D T region-head export/evaluation; no separate T training job.",
        "hash_policy": "Manifest paths are fixed at planning time; manifest SHA-256 values stay null until producer completion.",
    }


def _add_cross_evaluation_records(plan: dict[str, Any], *, datasets: list[str],
                                  cross_inputs: Mapping[str, Mapping[str, str]],
                                  manifest_inputs: Mapping[str, Mapping[str, str]]) -> None:
    targets = []
    for target in ("acdc", "mnms"):
        target_test = (cross_inputs.get(target, {}).get("test")
                       or manifest_inputs.get(target, {}).get("test"))
        ref = _manifest_ref(target_test)
        targets.append({"dataset": target, "manifest": ref})
    unknown = [
        {"dataset": None, "manifest": _manifest_ref(None), "status": "NOT_RUN",
         "reason": "dataset identity not specified in Q-3D spec"}
        for _ in range(2)
    ]
    for source in datasets:
        for experiment in plan["experiments"]:
            if experiment["dataset"] != source or experiment["group"] != "primary":
                continue
            final_job = f"{experiment['id']}.{experiment['stages'][-1].lower()}"
            for target in targets:
                status = "READY" if target["manifest"]["status"] == "READY" else "NOT_RUN"
                plan["cross_evaluation"].append({
                    "source_experiment": experiment["id"],
                    "source_dataset": source,
                    "target_dataset": target["dataset"],
                    "depends_on": final_job,
                    "manifest": copy.deepcopy(target["manifest"]),
                    "status": status,
                })
        for item in unknown:
            plan["cross_evaluation"].append(copy.deepcopy(item))


def create_campaign_plan(*, out: str | Path, datasets: tuple[str, ...] = ("acdc",),
                         protocols: tuple[str, ...] = ("A",),
                         ks: tuple[int, ...] = CAMPAIGN_K_ORDER,
                         seed: int = DEFAULT_SEED,
                         manifest_inputs: Mapping[str, Mapping[str, str]] | None = None,
                         cross_inputs: Mapping[str, Mapping[str, str]] | None = None,
                         teacher_provenance: Mapping[str, Any] | None = None,
                         include_secondary: bool = True, include_baselines: bool = True,
                         synthetic: bool = False, smoke_updates: int | None = None,
                         share_stages: bool = True,
                         runs: tuple[Any, ...] | list[Any] | None = None,
                         lineage_reference: Mapping[str, Any] | None = None) -> Path:
    """Build a hash-pinned campaign plan and per-experiment nested recipes."""
    if not datasets or any(name not in {"acdc", "mnms"} for name in datasets):
        raise ValueError("datasets must explicitly select acdc and/or mnms")
    if len(set(datasets)) != len(datasets):
        raise ValueError("datasets cannot contain duplicates")
    if not protocols or any(name not in {"A", "B"} for name in protocols):
        raise ValueError("protocols must be A and/or B")
    if len(set(protocols)) != len(protocols):
        raise ValueError("protocols cannot contain duplicates")
    if not ks or any(type(k) is not int or k not in CAMPAIGN_K_VALUES for k in ks):
        raise ValueError(f"K values must be selected from {CAMPAIGN_K_VALUES}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative")
    if smoke_updates is not None and (type(smoke_updates) is not int or smoke_updates < 1):
        raise ValueError("smoke_updates must be a positive integer")
    if smoke_updates is not None and not synthetic:
        raise ValueError("smoke update overrides require --synthetic")
    if synthetic and smoke_updates is None:
        smoke_updates = 1
    explicit_runs = None if runs is None else _normalize_run_specs(runs)
    if lineage_reference is not None and explicit_runs is None:
        raise ValueError("external lineage evidence requires an explicit run catalogue")
    if explicit_runs is not None:
        if not include_secondary or not include_baselines or not share_stages:
            raise ValueError(
                "explicit runs cannot be combined with --primary-only, --without-baselines, "
                "or --independent-stages; use the run catalogue to select jobs"
            )
        selected_run_ks = tuple(dict.fromkeys(k for _, k in explicit_runs))
        if ks != CAMPAIGN_K_ORDER and set(selected_run_ks) != set(ks):
            raise ValueError(
                "explicit runs and --k disagree; omit --k or pass exactly the K values used by runs"
            )
        ks = selected_run_ks
    manifest_inputs = copy.deepcopy(dict(manifest_inputs or {}))
    cross_inputs = copy.deepcopy(dict(cross_inputs or {}))
    nested_lineage_audit: dict[str, Any] | None = None
    if explicit_runs is not None:
        _validate_reduced_development_pins(
            manifest_inputs, datasets=datasets, protocols=protocols,
            selected_ks=tuple(dict.fromkeys(k for _, k in explicit_runs)), seed=seed,
        )
        nested_lineage_audit = _validate_nested_train_lineage(
            manifest_inputs, datasets=datasets,
            selected_ks=tuple(dict.fromkeys(k for _, k in explicit_runs)), seed=seed,
            lineage_reference=lineage_reference,
        )
    teacher_provenance = copy.deepcopy(dict(teacher_provenance)) if teacher_provenance is not None else None
    out_root = Path(out).expanduser().resolve()
    if out_root.exists():
        if any(out_root.iterdir()):
            raise FileExistsError(f"campaign output directory is not empty: {out_root}")
    else:
        out_root.mkdir(parents=True, exist_ok=False)
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "status": "PLAN_ONLY_NOT_RUN",
        "selection": {
            "datasets": list(datasets),
            "dataset_order": [name for name in ("acdc", "mnms") if name in datasets],
            "protocols": list(protocols),
            "k_order": list(
                tuple(dict.fromkeys(k for _, k in explicit_runs))
                if explicit_runs is not None
                else (k for k in SUPPORTED_K_ORDER if k in ks)
            ),
            "seed": seed,
            "synthetic": synthetic,
            "smoke_updates": smoke_updates,
            "shared_stage_dag": bool(share_stages),
            "primary_settings": list(
                dict.fromkeys(setting for setting, _ in explicit_runs)
                if explicit_runs is not None else PRIMARY_SETTINGS
            ),
            "secondary_settings": [] if explicit_runs is not None else list(
                SECONDARY_SETTINGS if share_stages else LEGACY_SECONDARY_SETTINGS
            ) if include_secondary else [],
            "baseline_settings": (
                [_TEACHER_SUPERVISED]
                if explicit_runs is not None and 8 in {k for _, k in explicit_runs}
                else [] if explicit_runs is not None
                else list(BASELINE_ARMS) if include_baselines else []
            ),
        },
        "experiments": [],
        "stage_jobs": [],
        "cross_evaluation": [],
        "baseline_launches": copy.deepcopy(BASELINE_LAUNCHES),
        "evaluation_jobs": [],
        "external_references": [],
        "evidence": {
            "patient_training": "NOT_RUN",
            "gpu_validation": "NOT_RUN",
            "real_data_metrics": "NOT_RUN",
            "novelty_claim": "NOT_CLAIMED",
        },
    }
    if lineage_reference is not None:
        plan["external_lineage_reference"] = copy.deepcopy(dict(lineage_reference))
    settings_by_group = [("primary", PRIMARY_SETTINGS)]
    if include_secondary:
        settings_by_group.append(("secondary", SECONDARY_SETTINGS if share_stages else LEGACY_SECONDARY_SETTINGS))
    if include_baselines:
        settings_by_group.append(("baseline", BASELINE_ARMS))
    ordered_ks = list(
        tuple(dict.fromkeys(k for _, k in explicit_runs))
        if explicit_runs is not None
        else (k for k in SUPPORTED_K_ORDER if k in ks)
    )
    if len(ordered_ks) != len(set(ks)):
        raise ValueError("K values contain duplicates or unsupported ordering entries")
    if explicit_runs is None:
        for dataset in ("acdc", "mnms"):
            if dataset not in datasets:
                continue
            for group, settings in settings_by_group:
                for protocol in protocols:
                    for k in ordered_ks:
                        for setting in settings:
                            _add_experiment(
                                plan, setting=setting, dataset=dataset, protocol=protocol, k=k, seed=seed,
                                out_root=out_root, manifest_inputs=manifest_inputs, cross_inputs=cross_inputs,
                                teacher_provenance=teacher_provenance, synthetic=synthetic,
                                smoke_updates=smoke_updates,
                                enforce_k8_lock=(
                                    not synthetic and dataset == "acdc" and protocol == "A"
                                    and k == 8 and seed == DEFAULT_SEED
                                ),
                            )
    else:
        settings_by_k: dict[int, list[str]] = {}
        for setting, k in explicit_runs:
            settings_by_k.setdefault(k, []).append(setting)
        for dataset in ("acdc", "mnms"):
            if dataset not in datasets:
                continue
            for protocol in protocols:
                for setting, k in explicit_runs:
                    _add_experiment(
                        plan, setting=setting, dataset=dataset, protocol=protocol, k=k, seed=seed,
                        out_root=out_root, manifest_inputs=manifest_inputs, cross_inputs=cross_inputs,
                        teacher_provenance=teacher_provenance, synthetic=synthetic,
                        smoke_updates=smoke_updates,
                        enforce_k8_lock=(
                            not synthetic and dataset == "acdc" and protocol == "A"
                            and k == 8 and seed == DEFAULT_SEED
                        ),
                    )
        if 8 in settings_by_k:
            for dataset in ("acdc", "mnms"):
                if dataset not in datasets:
                    continue
                for protocol in protocols:
                    _add_experiment(
                        plan, setting=_TEACHER_SUPERVISED, dataset=dataset, protocol=protocol, k=8,
                        seed=seed, out_root=out_root, manifest_inputs=manifest_inputs,
                        cross_inputs=cross_inputs, teacher_provenance=teacher_provenance,
                        synthetic=synthetic, smoke_updates=smoke_updates,
                        enforce_k8_lock=(
                            not synthetic and dataset == "acdc" and protocol == "A" and seed == DEFAULT_SEED
                        ),
                    )
                    plan["external_references"].append({
                        "id": f"{dataset}_{protocol.lower()}_k8_seed{seed}_a3d_teacher_free",
                        "dataset": dataset,
                        "protocol": protocol,
                        "k": 8,
                        "seed": seed,
                        "setting": "a3d_teacher_free",
                        "status": "NOT_RUN",
                        "reason": (
                            "reference only; no A-3D run is scheduled unless an actual equivalent "
                            "execution is separately proven"
                        ),
                        "launch_mapping": {
                            **copy.deepcopy(BASELINE_LAUNCHES["a3d_teacher_free"]),
                            "status": "REFERENCE_ONLY_NOT_RUN",
                        },
                    })
    if explicit_runs is not None:
        plan["selection"].update({
            "runs": [list(pair) for pair in explicit_runs],
            "locked_k8_train_manifest_sha256": K8_LOCKED_TRAIN_MANIFEST_SHA256,
            "locked_k8_development_manifest_sha256": K8_LOCKED_DEV_MANIFEST_SHA256,
        })
        k8_roles = manifest_inputs.get("acdc", {})
        k8_train_provided = bool(k8_roles.get("k8:train") or k8_roles.get("train"))
        selected_k_values = tuple(dict.fromkeys(k for _, k in explicit_runs))
        development_manifest_pins = []
        for selected_k in selected_k_values:
            development_path = k8_roles.get(f"k{selected_k}:development") or k8_roles.get("development")
            development_manifest_pins.append({
                "k": selected_k,
                **(_manifest_ref(development_path) if development_path is not None else {
                    "path": None, "sha256": None, "status": "NOT_RUN",
                    "reason": "development manifest was not supplied",
                }),
            })
        if not k8_train_provided:
            lineage_status = "K8_LOCKED_NOT_PROVIDED"
        elif nested_lineage_audit is None or nested_lineage_audit.get("schema_loader_status") != "READY":
            lineage_status = "K8_LOCKED_NOT_RUN"
        elif (nested_lineage_audit.get("status") != "PASS"
              or sum(bool(k8_roles.get(f"k{k}:train")) for k in selected_k_values)
              != len(selected_k_values)):
            lineage_status = "K8_LOCKED_PARTIAL_NOT_RUN"
        else:
            lineage_status = "K8_LOCKED_READY"
        plan["lineage_audit"] = {
            "schema": "q3d.campaign.lineage_audit.v1",
            "status": lineage_status,
            "seed": seed,
            "dataset": "acdc",
            "expected_k8_train_sha256": K8_LOCKED_TRAIN_MANIFEST_SHA256,
            "expected_k8_development_sha256": K8_LOCKED_DEV_MANIFEST_SHA256,
            "nested_k_specific_train_manifests": sum(
                bool(k8_roles.get(f"k{k}:train")) for k in selected_k_values
            ),
            "nested_validation": (
                "PASS"
                if (nested_lineage_audit is not None
                    and nested_lineage_audit.get("status") == "PASS"
                    and sum(bool(k8_roles.get(f"k{k}:train")) for k in selected_k_values) == len(selected_k_values))
                else "PARTIAL_NOT_RUN"
            ),
            "schema_loader_status": (
                nested_lineage_audit.get("schema_loader_status", "NOT_RUN")
                if nested_lineage_audit is not None else "NOT_RUN"
            ),
            "manifest_contracts": (
                nested_lineage_audit.get("records", [])
                if nested_lineage_audit is not None else []
            ),
            "development_manifest_pins": development_manifest_pins,
            "note": "K-specific manifests are never inferred from K=8; absent K manifests remain NOT_RUN",
        }
    _add_cross_evaluation_records(plan, datasets=list(datasets), cross_inputs=cross_inputs,
                                  manifest_inputs=manifest_inputs)
    plan["ordering"] = {
        "stage_order": ["T", "Q", "P", "J"],
        "group_order": [group for group, _ in settings_by_group],
        "dataset_order": plan["selection"]["dataset_order"],
        "k_order": plan["selection"]["k_order"],
        "secondary_after_primary": True,
    }
    if share_stages:
        _shared_stage_dag(plan, out_root)
    if explicit_runs is not None:
        for experiment in plan["experiments"]:
            if experiment["setting"] != _TEACHER_SUPERVISED:
                continue
            source_stage_job_id = experiment["stage_artifact_producers"].get("T")
            if source_stage_job_id is None:
                raise ValueError("teacher-supervised evaluation is missing its shared T producer")
            plan["evaluation_jobs"].append({
                "id": f"{experiment['id']}.eval_teacher_head",
                "dataset": experiment["dataset"],
                "protocol": experiment["protocol"],
                "k": experiment["k"],
                "seed": experiment["seed"],
                "setting": _TEACHER_SUPERVISED,
                "stage": "T",
                "head": "region",
                "student_joint_updates": 0,
                "source_stage_job_id": source_stage_job_id,
                "independent_baseline": False,
                "status": "NOT_RUN",
                "reason": "evaluation-only teacher-head reference; no duplicate T producer or J job",
            })
    plan_path = out_root / "PLAN.json"
    _write_json(plan_path, plan)
    validate_campaign_plan(plan_path)
    return plan_path


def _validate_shared_stage_dag(plan: Mapping[str, Any]) -> None:
    dag = plan.get("shared_stage_dag")
    if not isinstance(dag, Mapping) or dag.get("schema") != SHARED_DAG_SCHEMA:
        raise ValueError("shared campaign is missing its Q-3D stage DAG contract")
    if dag.get("status") != "PLAN_ONLY_NOT_RUN":
        raise ValueError("shared campaign plans must remain PLAN_ONLY_NOT_RUN")
    jobs = plan["stage_jobs"]
    job_by_id = {job["id"]: job for job in jobs}
    stage_rank = {stage: index for index, stage in enumerate(("T", "Q", "P", "J"))}
    for job in jobs:
        stage = job.get("stage")
        if stage not in stage_rank:
            raise ValueError(f"unknown shared Q-3D campaign stage in {job.get('id')}")
        dependencies = job.get("dependencies")
        if not isinstance(dependencies, list) or len(dependencies) != len(set(dependencies)):
            raise ValueError(f"invalid dependency list for {job['id']}")
        for dependency_id in dependencies:
            upstream = job_by_id.get(dependency_id)
            if upstream is None:
                raise ValueError(f"missing DAG producer {dependency_id} for {job['id']}")
            if stage_rank[upstream["stage"]] >= stage_rank[stage]:
                raise ValueError(f"stage dependency order is invalid for {job['id']}")
        if job.get("status") != "NOT_RUN":
            raise ValueError("campaign stage jobs must remain NOT_RUN in a plan")

        inputs = job.get("stage_artifact_inputs", {})
        if not isinstance(inputs, Mapping):
            raise ValueError(f"stage_artifact_inputs is malformed for {job['id']}")
        for input_stage, reference in inputs.items():
            if input_stage not in {"T", "Q", "P"} or not isinstance(reference, Mapping):
                raise ValueError(f"invalid stage artifact input for {job['id']}")
            producer_id = reference.get("producer_job_id")
            producer = job_by_id.get(producer_id)
            if producer is None or producer.get("stage") != input_stage:
                raise ValueError(f"{job['id']} does not point at a {input_stage} producer")
            output = producer.get("artifact_output")
            if not isinstance(output, Mapping) or reference.get("path") != output.get("manifest_path"):
                raise ValueError(f"{job['id']} stage input path differs from its producer output")
            if reference.get("sha256") is not None or reference.get("status") != "NOT_RUN":
                raise ValueError("unrun plan references must not invent stage manifest hashes")
            if producer_id not in dependencies:
                raise ValueError(f"{job['id']} is missing a direct dependency on {producer_id}")

        if job.get("kind") == "shared_stage_producer":
            if stage not in {"T", "Q", "P"}:
                raise ValueError("shared stage producers may only produce T/Q/P")
            output = job.get("artifact_output")
            if not isinstance(output, Mapping) or output.get("stage") != stage:
                raise ValueError(f"{job['id']} has no stage manifest output record")
            expected_path = str((Path(job["output"]) / f"{stage}.artifact.json").resolve())
            if output.get("manifest_path") != expected_path:
                raise ValueError(f"{job['id']} stage manifest path differs from the output convention")
            if output.get("manifest_sha256") is not None or output.get("status") != "NOT_RUN":
                raise ValueError("producer manifest hashes stay null until a stage completes")
            identity = job.get("artifact_identity")
            if not isinstance(identity, Mapping) or _identity_key(identity) != (
                job["dataset"], job["protocol"], job["k"], job["seed"],
            ):
                raise ValueError(f"{job['id']} has a mismatched data/protocol/K/seed identity")
            if stage == "T" and inputs:
                raise ValueError("T producer cannot consume earlier stage artifacts")
            if stage == "Q" and set(inputs) != {"T"}:
                raise ValueError("Q producer must consume exactly the shared T manifest")
            if stage == "P" and ("T" not in inputs or not set(inputs).issubset({"T", "Q"})):
                raise ValueError("P producer must consume T and may consume Q")
            if stage in {"Q", "P"}:
                source_key = job.get("source_key")
                if not isinstance(source_key, Mapping):
                    raise ValueError(f"{stage} producer has no feature-source identity")
                source_config = job.get("source_config")
                if (not isinstance(source_config, Mapping)
                        or source_config.get("teacher_feature_source") != source_key.get("feature_source")
                        or source_config.get("teacher_feature_stride") != source_key.get("feature_stride")
                        or source_config.get("teacher_variant") != source_key.get("teacher_variant")):
                    raise ValueError(f"{stage} producer source_config does not pin its feature source")
        elif stage == "J":
            if job.get("kind") != "student_joint":
                raise ValueError(f"J job {job['id']} must be a student_joint consumer")
            if "T" in inputs or "Q" in inputs or "P" in inputs:
                for input_stage in inputs:
                    if input_stage not in {"T", "Q", "P"}:
                        raise ValueError("J may consume only pinned T/Q/P manifests")
            if job.get("artifact_output") is not None:
                raise ValueError("J consumes shared artifacts but does not produce a T/Q/P manifest")
        else:
            raise ValueError(f"unrecognized shared DAG job kind for {job['id']}")

    counts = {stage: sum(job["stage"] == stage for job in jobs) for stage in stage_rank}
    producer_counts = {
        stage: sum(job.get("kind") == "shared_stage_producer" and job["stage"] == stage for job in jobs)
        for stage in ("T", "Q", "P")
    }
    if dag.get("stage_job_counts") != counts or dag.get("shared_producer_counts") != producer_counts:
        raise ValueError("shared campaign DAG stage counts are stale")
    if producer_counts["T"] != len({
        _identity_key(job["artifact_identity"])
        for job in jobs if job.get("kind") == "shared_stage_producer" and job["stage"] == "T"
    }):
        raise ValueError("shared campaign has duplicate T producers for a data/protocol/K/seed identity")
    if any(job.get("setting") == _TEACHER_SUPERVISED and job["stage"] == "T" for job in jobs):
        raise ValueError("teacher-supervised baseline must reuse shared T instead of retraining it")

    experiments = plan["experiments"]
    experiment_by_setting = {
        (item["dataset"], item["protocol"], item["k"], item["seed"], item["setting"]): item
        for item in experiments
    }
    for experiment in experiments:
        expected_ids = experiment.get("stage_artifact_producers", {})
        if not isinstance(expected_ids, Mapping):
            raise ValueError(f"{experiment['id']} has no producer-reference map")
        stage_ids = experiment.get("stage_job_ids", [])
        if stage_ids != [expected_ids.get(stage) for stage in experiment["stages"]]:
            raise ValueError(f"{experiment['id']} stage_job_ids differ from producer references")
        for job_id in stage_ids:
            if job_id not in job_by_id:
                raise ValueError(f"{experiment['id']} references a missing shared stage job")
        expected_artifacts = {
            stage for stage in experiment["stages"] if stage in {"T", "Q", "P"}
        }
        if experiment["setting"] == "c1" and "Q" in expected_ids:
            expected_artifacts.add("Q")
        actual_artifacts = set(experiment.get("expected_stage_artifacts", {}))
        if expected_artifacts != actual_artifacts:
            raise ValueError(f"{experiment['id']} expected artifact refs differ from its stages")
        for stage, reference in experiment.get("expected_stage_artifacts", {}).items():
            if reference.get("producer_job_id") != expected_ids.get(stage) or reference.get("sha256") is not None:
                raise ValueError(f"{experiment['id']} has a stale or fabricated {stage} reference")
        if experiment["setting"] == _TEACHER_SUPERVISED:
            reuse = experiment.get("baseline_reuse")
            if not isinstance(reuse, Mapping) or reuse.get("source_stage_job_id") != expected_ids.get("T"):
                raise ValueError("teacher-supervised baseline must explicitly reuse the shared T artifact")
        if "J" in experiment["stages"]:
            joint_id = expected_ids["J"]
            joint = job_by_id[joint_id]
            expected_inputs = {
                stage: expected_ids[stage] for stage in ("T", "Q", "P") if stage in experiment["stages"]
            }
            optional_inputs = ["Q"] if experiment["setting"] == "c1" else []
            for stage in optional_inputs:
                q_ref = joint["stage_artifact_inputs"].get("Q", {})
                q_job_id = q_ref.get("producer_job_id")
                q_producer = job_by_id.get(q_job_id)
                p_producer = job_by_id.get(expected_ids.get("P"))
                if (q_producer is None or q_producer.get("stage") != "Q"
                        or q_ref.get("role") != "optional_source_metadata"
                        or p_producer is None
                        or q_producer.get("artifact_identity") != p_producer.get("artifact_identity")
                        or q_producer.get("source_key") != p_producer.get("source_key")):
                    raise ValueError(f"{experiment['id']} is missing optional Q source metadata")
                expected_inputs["Q"] = q_job_id
            actual_inputs = {
                stage: ref.get("producer_job_id")
                for stage, ref in joint["stage_artifact_inputs"].items()
            }
            if actual_inputs != expected_inputs:
                raise ValueError(f"{experiment['id']} J stage does not consume its pinned stage artifacts")
            if joint["stage"] != "J":
                raise ValueError("J consumer must select only the joint stage, never retrain T or Q")

    for key, v1 in experiment_by_setting.items():
        if key[-1] != "v1":
            continue
        identity = key[:-1]
        v1u = experiment_by_setting.get((*identity, "v1u"))
        if v1u is None:
            continue
        for stage in ("T", "Q", "P"):
            if v1["stage_artifact_producers"].get(stage) != v1u["stage_artifact_producers"].get(stage):
                raise ValueError(f"V1 and V1-u do not share the same {stage} producer")
        t_id = v1["stage_artifact_producers"]["T"]
        for variant in ("o1", "s1"):
            other = experiment_by_setting.get((*identity, variant))
            if other is None:
                continue
            if other["stage_artifact_producers"].get("T") != t_id:
                raise ValueError(f"{variant} must share T with V1")
            for stage in ("Q", "P"):
                if other["stage_artifact_producers"].get(stage) == v1["stage_artifact_producers"].get(stage):
                    raise ValueError(f"{variant} must use a distinct {stage} source artifact")
        residual2 = experiment_by_setting.get((*identity, "v1_residual2"))
        if residual2 is not None:
            for stage in ("T", "Q", "P"):
                if residual2["stage_artifact_producers"].get(stage) != v1["stage_artifact_producers"].get(stage):
                    raise ValueError("V1 residual2 must reuse the main V1 T/Q/P source")

    expected_pairs = dag.get("expected_v1_v1u_shared_references")
    if not isinstance(expected_pairs, list):
        raise ValueError("shared campaign DAG must report expected V1/V1-u shared refs")
    for pair in expected_pairs:
        refs = pair.get("producer_job_ids")
        if not isinstance(refs, Mapping) or set(refs) != {"T", "Q", "P"}:
            raise ValueError("V1/V1-u shared-ref record must pin T/Q/P producer IDs")
        if set(pair.get("manifest_sha256", {})) != {"T", "Q", "P"} or any(
            value is not None for value in pair["manifest_sha256"].values()
        ):
            raise ValueError("planned V1/V1-u SHA values must remain null until stages complete")
    if dag.get("expected_v1_v1u_shared_reference_count") != len(expected_pairs):
        raise ValueError("shared V1/V1-u expected-reference count is stale")
    for record in dag.get("shared_consumer_references", []):
        producer_ids = record.get("producer_job_ids")
        consumer_ids = record.get("consumer_job_ids")
        if not isinstance(producer_ids, Mapping) or set(producer_ids) != {"T", "Q", "P"}:
            raise ValueError("shared C1/V1/V1-u/E1 record must pin T/Q/P producer IDs")
        if not isinstance(consumer_ids, Mapping):
            raise ValueError("shared C1/V1/V1-u/E1 record must list its J consumers")
        identity = record.get("identity", {})
        for setting, consumer_id in consumer_ids.items():
            experiment = experiment_by_setting.get((
                identity.get("dataset"), identity.get("protocol"),
                identity.get("k"), identity.get("seed"), setting,
            ))
            if experiment is None:
                raise ValueError(f"shared consumer reference names missing setting {setting}")
            if experiment["stage_artifact_producers"].get("J") != consumer_id:
                raise ValueError(f"shared consumer J reference differs for {setting}")
            if any(experiment["stage_artifact_producers"].get(stage) != producer_ids[stage]
                   for stage in ("T", "Q", "P")):
                raise ValueError(f"{setting} does not consume the shared T/Q/P producers")


def validate_campaign_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path).expanduser().resolve()
    plan = _read_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "PLAN_ONLY_NOT_RUN":
        raise ValueError("unsupported or already mutated Q-3D campaign plan")
    selection = plan.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("campaign selection is missing")
    experiments = plan.get("experiments")
    jobs = plan.get("stage_jobs")
    if not isinstance(experiments, list) or not isinstance(jobs, list) or not experiments or not jobs:
        raise ValueError("campaign must contain experiments and ordered stage jobs")
    experiment_by_id = {item["id"]: item for item in experiments}
    if len(experiment_by_id) != len(experiments):
        raise ValueError("experiment IDs must be unique")
    job_by_id = {item["id"]: item for item in jobs}
    if len(job_by_id) != len(jobs):
        raise ValueError("stage job IDs must be unique")
    group_rank = {name: index for index, name in enumerate(plan["ordering"]["group_order"])}
    dataset_rank = {name: index for index, name in enumerate(plan["ordering"]["dataset_order"])}
    k_rank = {value: index for index, value in enumerate(plan["ordering"]["k_order"])}
    protocol_rank = {name: index for index, name in enumerate(selection["protocols"])}
    settings_by_group = {
        "primary": list(selection["primary_settings"]),
        "secondary": list(selection["secondary_settings"]),
        "baseline": list(selection["baseline_settings"]),
    }
    setting_rank = {group: {name: index for index, name in enumerate(values)}
                     for group, values in settings_by_group.items()}
    if selection.get("runs") is not None:
        declared_runs = _normalize_run_specs(list(selection["runs"]))
        expected_order = [
            (dataset, protocol, setting, k)
            for dataset in plan["ordering"]["dataset_order"]
            for protocol in selection["protocols"] for setting, k in declared_runs
        ]
        if any(k == 8 for _, k in declared_runs):
            expected_order += [(dataset, protocol, _TEACHER_SUPERVISED, 8)
                               for dataset in plan["ordering"]["dataset_order"]
                               for protocol in selection["protocols"]]
        actual_order = [(item["dataset"], item["protocol"], item["setting"], item["k"])
                        for item in experiments]
        if actual_order != expected_order:
            raise ValueError("campaign ordering must preserve the explicit run catalogue")
    else:
        exp_keys = [(dataset_rank[item["dataset"]], group_rank[item["group"]],
                     protocol_rank[item["protocol"]], k_rank[item["k"]],
                     setting_rank[item["group"]][item["setting"]]) for item in experiments]
        if exp_keys != sorted(exp_keys):
            raise ValueError("campaign ordering must put ACDC first, selected K values in declared order, and secondaries after primaries")
    for experiment in experiments:
        config_path = Path(experiment["config_path"])
        if not config_path.is_file() or _sha256(config_path) != experiment["config_sha256"]:
            raise ValueError(f"campaign recipe missing or changed: {config_path}")
        config = load_config(config_path)
        if plan.get("external_lineage_reference") is not None:
            context = load_external_lineage(
                plan["external_lineage_reference"], dataset=config["dataset"]["name"],
                seed=config["seed"],
            )
            reason = _manifest_loader_block_reason(
                config["dataset"]["manifests"]["train"], expected_role="training",
                expected_k=config["protocol"]["k"], lineage_context=context,
            )
            if reason is not None:
                raise ValueError(reason)
        if (config["arm"] != experiment["arm"] or config["dataset"]["name"] != experiment["dataset"]
                or config["protocol"]["variant"] != experiment["protocol"]
                or config["protocol"]["k"] != experiment["k"]):
            raise ValueError(f"experiment identity differs from its config: {experiment['id']}")
        if experiment["group"] in {"primary", "secondary", "baseline"}:
            stage_ids = [job_id for job_id in experiment.get("stage_job_ids", [])]
            if not stage_ids:
                stage_ids = [f"{experiment['id']}.{name.lower()}" for name in experiment["stages"]]
            stage_records = [job_by_id.get(job_id) for job_id in stage_ids]
            if any(record is None for record in stage_records):
                raise ValueError(f"experiment stage references are incomplete: {experiment['id']}")
            actual_stages = [record["stage"] for record in stage_records]
            if actual_stages != experiment["stages"]:
                raise ValueError(f"stage dependency order differs for {experiment['id']}")
            if not selection.get("shared_stage_dag", False):
                previous = None
                for record in stage_records:
                    expected_dependencies = [previous] if previous else []
                    if record["dependencies"] != expected_dependencies:
                        raise ValueError(f"stage dependency chain is invalid for {record['id']}")
                    stage_rank = {name: index for index, name in enumerate(("T", "Q", "P", "J"))}
                    if previous and stage_rank[job_by_id[previous]["stage"]] >= stage_rank[record["stage"]]:
                        raise ValueError("stage order must be T then Q then P then J")
                    previous = record["id"]
    explicit_runs = selection.get("runs")
    if explicit_runs is not None:
        parsed_runs = _normalize_run_specs(list(explicit_runs))
        primary_pairs = [
            (item["setting"], item["k"])
            for item in experiments if item["group"] == "primary"
        ]
        expected_pairs = list(parsed_runs) * len(selection["datasets"]) * len(selection["protocols"])
        if len(primary_pairs) != len(expected_pairs) or set(primary_pairs) != set(expected_pairs):
            raise ValueError("explicit run catalogue does not match the planned primary experiments")
        if selection.get("secondary_settings"):
            raise ValueError("explicit run plans cannot contain secondary settings")
        expected_baselines = [_TEACHER_SUPERVISED] if 8 in {k for _, k in parsed_runs} else []
        if selection.get("baseline_settings") != expected_baselines:
            raise ValueError("explicit run baseline selection is stale")
        evaluation_jobs = plan.get("evaluation_jobs", [])
        expected_evaluations = len(selection["datasets"]) * len(selection["protocols"])
        if 8 not in {k for _, k in parsed_runs}:
            expected_evaluations = 0
        if len(evaluation_jobs) != expected_evaluations:
            raise ValueError("explicit run plan has an incomplete teacher-head evaluation catalogue")
        for evaluation in evaluation_jobs:
            if (evaluation.get("stage"), evaluation.get("head"), evaluation.get("student_joint_updates")) != (
                "T", "region", 0
            ):
                raise ValueError("teacher-head evaluation must be T/region with zero J updates")
            if evaluation.get("independent_baseline") is not False or evaluation.get("status") != "NOT_RUN":
                raise ValueError("teacher-head evaluation must remain a non-independent NOT_RUN reference")
            source = job_by_id.get(evaluation.get("source_stage_job_id"))
            if source is None or source.get("stage") != "T" or source.get("kind") != "shared_stage_producer":
                raise ValueError("teacher-head evaluation must reference a shared T producer")
            identity = (source["dataset"], source["protocol"], source["k"], source["seed"])
            if (evaluation.get("dataset"), evaluation.get("protocol"), evaluation.get("k"), evaluation.get("seed")) != identity:
                raise ValueError("teacher-head evaluation identity differs from its T producer")
        for reference in plan.get("external_references", []):
            if reference.get("setting") != "a3d_teacher_free" or reference.get("status") != "NOT_RUN":
                raise ValueError("A-3D reference must remain an external NOT_RUN record")
            if reference.get("launch_mapping", {}).get("status") != "REFERENCE_ONLY_NOT_RUN":
                raise ValueError("A-3D reference launch mapping must remain reference-only")
            if any(item["setting"] == "a3d_teacher_free" for item in experiments):
                raise ValueError("A-3D reference-only setting must not create an experiment job")
    comparisons: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for experiment in experiments:
        if experiment["setting"] in PRIMARY_ARMS + SECONDARY_ARMS:
            key = (experiment["dataset"], experiment["protocol"], experiment["k"], experiment["seed"])
            comparisons.setdefault(key, []).append(load_config(experiment["config_path"]))
    for configs in comparisons.values():
        validate_budget_update_equality(configs, smoke=bool(selection["synthetic"]))
    if selection.get("shared_stage_dag", False):
        _validate_shared_stage_dag(plan)
    return plan


def _rewrite_synthetic_manifest(path: Path, *, dataset: str) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("schema") != "self_audit.data.v2":
        raise ValueError("synthetic fixture manifest schema mismatch")
    manifest["dataset"] = dataset
    _write_json(path, manifest)
    return {"path": str(path.resolve()), "sha256": _sha256(path), "status": "READY"}


def _synthetic_protocol_b_student_view(teacher_manifest: Mapping[str, Any], *, k: int) -> dict[str, Any]:
    """Derive K-labeled plus image-only U from the exact all-labeled B cohort."""
    labeled = teacher_manifest.get("labeled")
    unlabeled = teacher_manifest.get("unlabeled")
    if not isinstance(labeled, list) or len(labeled) <= k or unlabeled != []:
        raise ValueError("Protocol B synthetic teacher view must label K+U TRAIN patients with no U pool")
    if any(not isinstance(record, Mapping) or not record.get("masks") for record in labeled):
        raise ValueError("Protocol B synthetic teacher view must carry masks for every TRAIN patient")
    student = copy.deepcopy(dict(teacher_manifest))
    student["label_budget"] = k
    student["labeled"] = copy.deepcopy(labeled[:k])
    student["unlabeled"] = [
        {key: value for key, value in record.items() if key != "masks"}
        for record in labeled[k:]
    ]
    if any("masks" in record for record in student["unlabeled"]):
        raise ValueError("Protocol B student U records must not retain teacher masks")
    return student


def _prepare_synthetic_run(plan_path: Path, experiment: Mapping[str, Any],
                           config: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Create an isolated manufactured fixture and a run-time config copy."""
    if experiment["protocol"] == "B" and experiment["dataset"] not in {"acdc", "mnms"}:
        raise ValueError("synthetic Protocol B requires a named supported dataset")
    from self_audit.candidate_a3d.data import synthetic_fixture

    fixture_root = plan_path.parent / "synthetic_fixtures" / experiment["id"]
    if fixture_root.exists():
        raise FileExistsError(f"synthetic fixture already exists: {fixture_root}")
    fixture_root.parent.mkdir(parents=True, exist_ok=True)
    from scripts.smoke_q3d import _manufacture_cine_frames
    dataset_name = experiment["dataset"]
    if experiment["protocol"] == "B":
        cohort_fixture = synthetic_fixture(
            fixture_root / "cohort", seed=19, n_labeled=experiment["k"] + 2, n_unlabeled=0,
            n_evaluation=1, shape_tzyx=(2, 8, 32, 32),
        )
        teacher_train = _rewrite_synthetic_manifest(Path(cohort_fixture["train"]), dataset=dataset_name)
        _manufacture_cine_frames(Path(teacher_train["path"]))
        teacher_train = _manifest_ref(teacher_train["path"])
        teacher_manifest = _read_json(Path(teacher_train["path"]))
        student_manifest = _synthetic_protocol_b_student_view(teacher_manifest, k=experiment["k"])
        student_manifest_path = Path(teacher_train["path"]).with_name("student_train.json")
        _write_json(student_manifest_path, student_manifest)
        student_train = _manifest_ref(student_manifest_path)
        development = _rewrite_synthetic_manifest(Path(cohort_fixture["evaluation"]), dataset=dataset_name)
        _manufacture_cine_frames(Path(development["path"]))
        development = _manifest_ref(development["path"])
        config["dataset"]["manifests"]["teacher_train"] = teacher_train
    else:
        student_fixture = synthetic_fixture(
            fixture_root / "student", seed=19, n_labeled=experiment["k"], n_unlabeled=2,
            n_evaluation=1, shape_tzyx=(2, 8, 32, 32),
        )
        student_train = _rewrite_synthetic_manifest(Path(student_fixture["train"]), dataset=dataset_name)
        _manufacture_cine_frames(Path(student_train["path"]))
        student_train = _manifest_ref(student_train["path"])
        development = _rewrite_synthetic_manifest(Path(student_fixture["evaluation"]), dataset=dataset_name)
        _manufacture_cine_frames(Path(development["path"]))
        development = _manifest_ref(development["path"])
        config["dataset"]["manifests"]["teacher_train"] = copy.deepcopy(student_train)
    config["dataset"]["manifests"]["train"] = student_train
    config["dataset"]["manifests"]["development"] = development
    config["dataset"]["manifests"]["test"] = {"path": None, "sha256": None, "status": "NOT_RUN"}
    validate_config(config)
    config_path = plan_path.parent / "runtime_configs" / f"{experiment['id']}.json"
    _write_json(config_path, config)
    train_manifest = _read_json(Path(student_train["path"]))
    development_manifest = _read_json(Path(development["path"]))
    return config_path, train_manifest, {"train": student_train, "development": development}


def _job_command(experiment: Mapping[str, Any], *, config_path: Path,
                 train_ref: Mapping[str, Any], dev_ref: Mapping[str, Any],
                 device: str, synthetic: bool, stage: str | None = None,
                 stage_artifacts: Mapping[str, str] | None = None) -> list[str]:
    command = [sys.executable, str(ROOT / "train.py"), "--config", str(config_path),
               "--out", experiment["output"], "--device", device]
    if stage is not None:
        command.extend(["--q3d-stage", stage])
    for artifact_stage, manifest_path in sorted((stage_artifacts or {}).items()):
        command.extend(["--q3d-stage-artifact", f"{artifact_stage}={manifest_path}"])
    if synthetic:
        command.append("--synthetic")
        if experiment["arm"] in SYNTHETIC_CONFIDENT_TEACHER_ARMS:
            command.append("--synthetic-teacher-fixture")
    else:
        command.append("--allow-full-run")
    return command


def _read_state(state_path: Path, plan: Mapping[str, Any], plan_sha: str) -> dict[str, Any]:
    if state_path.exists():
        state = _read_json(state_path)
        if state.get("schema") != STATE_SCHEMA or state.get("plan_sha256") != plan_sha:
            raise ValueError("campaign run state does not match this immutable plan")
        return state
    return {
        "schema": STATE_SCHEMA,
        "plan_sha256": plan_sha,
        "jobs": {job["id"]: {"status": "NOT_RUN", "exit_code": None} for job in plan["stage_jobs"]},
        "experiments": {experiment["id"]: {"status": "NOT_RUN", "exit_code": None}
                        for experiment in plan["experiments"]},
    }


def execute_campaign(path: str | Path, *, execute: bool = False, setting: str = "all",
                     k: tuple[int, ...] | None = None, through_stage: str = "J",
                     device: str = "cuda",
                     runner: Any = subprocess.run) -> dict[str, Any]:
    """Validate or explicitly execute selected settings; the trainer owns stage transitions."""
    plan_path = Path(path).expanduser().resolve()
    plan = validate_campaign_plan(plan_path)
    if setting != "all" and setting not in ARMS and setting not in SECONDARY_SETTINGS:
        raise ValueError(f"unknown setting: {setting}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if through_stage not in {"T", "Q", "P", "J"}:
        raise ValueError("through-stage must be one of T, Q, P, J")
    if through_stage != "J" and not plan["selection"].get("shared_stage_dag"):
        raise ValueError("partial stage execution requires a shared-stage campaign plan")
    selected, selected_ks = select_campaign_experiments(
        plan["experiments"], setting=setting, k=k,
    )
    if not execute:
        return {"status": "PLAN_VALIDATED_NOT_RUN", "plan": str(plan_path),
                "selected_setting": setting,
                "selected_k": selected_ks,
                "through_stage": through_stage,
                "patient_training": "NOT_RUN", "jobs": len(plan["stage_jobs"])}
    if plan["selection"].get("shared_stage_dag"):
        from self_audit.candidate_q3d.campaign_runtime import execute_shared_campaign

        return execute_shared_campaign(
            plan_path, plan, setting=setting, k=k, through_stage=through_stage,
            device=device, runner=runner,
            api=sys.modules[__name__],
        )
    synthetic = bool(plan["selection"]["synthetic"])
    if not synthetic and device == "cpu":
        raise ValueError("real campaign execution requires explicit CUDA; CPU is reserved for synthetic smoke")
    plan_sha = _sha256(plan_path)
    state_path = plan_path.parent / "RUN_STATE.json"
    state = _read_state(state_path, plan, plan_sha)
    jobs_by_experiment: dict[str, list[dict[str, Any]]] = {}
    for job in plan["stage_jobs"]:
        jobs_by_experiment.setdefault(job["experiment_id"], []).append(job)
    # Validate the complete selection before creating run outputs or invoking
    # any trainer, so an unavailable baseline cannot leave an ``all`` campaign
    # half launched after earlier settings have already consumed resources.
    for experiment in selected:
        experiment_jobs = jobs_by_experiment[experiment["id"]]
        if state["experiments"][experiment["id"]].get("status") != "NOT_RUN":
            raise ValueError(f"setting already has a run-state record; no implicit resume: {experiment['id']}")
        for job in experiment_jobs:
            required_readiness = "SMOKE_READY" if synthetic else "READY"
            if job["readiness"] != required_readiness:
                raise ValueError(f"execution blocked for {job['id']}: {job['reason']}")
    completed = []
    for experiment in selected:
        experiment_jobs = jobs_by_experiment[experiment["id"]]
        train_ref = experiment_jobs[0]["train_manifest"]
        dev_ref = experiment_jobs[0]["development_manifest"]
        run_config = Path(experiment["config_path"])
        if synthetic:
            raw_config = load_config(run_config)
            run_config, _, fixture_refs = _prepare_synthetic_run(plan_path, experiment, raw_config)
            train_ref, dev_ref = fixture_refs["train"], fixture_refs["development"]
        elif experiment_jobs[0]["method"] == "candidate_q3d":
            load_config(experiment["config_path"], real_run=True)
        command = _job_command(experiment, config_path=run_config, train_ref=train_ref,
                               dev_ref=dev_ref, device=device, synthetic=synthetic)
        output = Path(experiment["output"])
        output.mkdir(parents=True, exist_ok=False)
        state["experiments"][experiment["id"]] = {"status": "RUNNING", "command": command, "exit_code": None}
        for job in experiment_jobs:
            state["jobs"][job["id"]] = {"status": "RUNNING", "command": command, "exit_code": None}
        _write_json(state_path, state)
        try:
            result = runner(command, cwd=ROOT, check=False, capture_output=True, text=True)
            code = getattr(result, "returncode", result if isinstance(result, int) else 0)
        except Exception as exc:
            state["experiments"][experiment["id"]] = {"status": "FAILED", "command": command,
                                                       "error": str(exc), "exit_code": None}
            for job in experiment_jobs:
                state["jobs"][job["id"]] = {"status": "FAILED", "command": command,
                                             "error": str(exc), "exit_code": None}
            _write_json(state_path, state)
            raise
        if code != 0:
            state["experiments"][experiment["id"]] = {"status": "FAILED", "command": command, "exit_code": code}
            for job in experiment_jobs:
                state["jobs"][job["id"]] = {"status": "FAILED", "command": command, "exit_code": code}
            _write_json(state_path, state)
            raise RuntimeError(f"campaign setting failed ({code}): {experiment['id']}")
        stdout = getattr(result, "stdout", "") or ""
        try:
            receipt = json.loads(stdout) if stdout.strip() else {}
        except json.JSONDecodeError:
            receipt = {}
        reported = receipt.get("completed_stages", []) if isinstance(receipt, Mapping) else []
        if isinstance(receipt, Mapping) and receipt.get("status") in {"COMPLETED", "COMPLETED_SOFTWARE_ONLY"}:
            if experiment["arm"] == "a3d_teacher_free":
                reported = reported or ["J"]
            elif experiment["arm"] == _TEACHER_SUPERVISED:
                reported = reported or ["T"]
        expected_stages = experiment["stages"]
        completed_stages = [stage_name for stage_name in expected_stages if stage_name in reported]
        state["experiments"][experiment["id"]] = {
            "status": "COMPLETED" if completed_stages == expected_stages else "PARTIAL",
            "command": command,
            "exit_code": 0,
            "completed_stages": completed_stages,
            "stdout_tail": stdout[-4000:],
        }
        for job in experiment_jobs:
            stage_status = "COMPLETED" if job["stage"] in completed_stages else "NOT_COMPLETED"
            state["jobs"][job["id"]] = {
                "status": stage_status, "command": command, "exit_code": 0,
                "completed_by": "single trainer invocation",
            }
        _write_json(state_path, state)
        if completed_stages != expected_stages:
            raise RuntimeError(f"trainer did not report all planned stages for {experiment['id']}")
        completed.append(experiment["id"])
    return {"status": "COMPLETED", "plan": str(plan_path), "run_state": str(state_path),
            "completed_settings": completed, "selected_k": selected_ks,
            "through_stage": through_stage,
            "patient_training": "SYNTHETIC_ONLY" if synthetic else "EXECUTED"}


def _parse_teacher_provenance(args: argparse.Namespace) -> dict[str, Any] | None:
    if not any((args.teacher_weights, args.teacher_weights_sha256, args.teacher_weights_source_url,
                args.pretraining_dataset, args.overlap_acdc is not None, args.overlap_mnms is not None)):
        return None
    if not all((args.teacher_weights, args.teacher_weights_sha256, args.teacher_weights_source_url,
                args.pretraining_dataset, args.overlap_acdc is not None, args.overlap_mnms is not None)):
        raise ValueError(
            "teacher provenance requires weights path/hash/source URL, pretraining dataset, and both overlap declarations"
        )
    return {
        "weights_path": str(Path(args.teacher_weights).expanduser().resolve()) if args.teacher_weights else None,
        "weights_sha256": args.teacher_weights_sha256,
        "weights_source_url": args.teacher_weights_source_url,
        "pretraining_dataset": args.pretraining_dataset,
        "overlap_acdc": args.overlap_acdc,
        "overlap_mnms": args.overlap_mnms,
        "class_mapping": ["BG", "RV", "MYO", "LV"],
        "provenance_status": "READY",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="write a NOT_RUN campaign plan")
    plan_parser.add_argument("--out", default=str(DEFAULT_REPORT_DIR))
    plan_parser.add_argument("--dataset", choices=("acdc", "mnms", "both"), default="acdc")
    plan_parser.add_argument("--protocol", choices=("A", "B", "both"), default="A")
    plan_parser.add_argument("--k", nargs="+", type=int, choices=CAMPAIGN_K_VALUES, default=list(CAMPAIGN_K_ORDER))
    plan_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    plan_parser.add_argument("--lineage-reference")
    plan_parser.add_argument("--lineage-reference-sha256")
    plan_parser.add_argument("--lineage-source")
    plan_parser.add_argument("--lineage-source-sha256")
    plan_parser.add_argument(
        "--manifest", action="append", default=[], metavar="DATASET[:K]:ROLE=PATH",
        help="pin a manifest globally or for one K (for example acdc:8:train=PATH)",
    )
    plan_parser.add_argument("--cross-eval-manifest", action="append", default=[], metavar="DATASET=PATH")
    plan_parser.add_argument("--teacher-weights")
    plan_parser.add_argument("--teacher-weights-sha256")
    plan_parser.add_argument("--teacher-weights-source-url")
    plan_parser.add_argument("--pretraining-dataset")
    plan_parser.add_argument("--overlap-acdc", choices=("yes", "no"))
    plan_parser.add_argument("--overlap-mnms", choices=("yes", "no"))
    plan_parser.add_argument("--primary-only", action="store_true")
    plan_parser.add_argument("--without-baselines", action="store_true")
    plan_parser.add_argument(
        "--runs", nargs="+", metavar="SETTING@K",
        help="explicit six-run catalogue; bare ACDC/A/seed-42 uses the same catalogue",
    )
    plan_parser.add_argument(
        "--legacy-catalogue", action="store_true",
        help="retain the historical full catalogue when --runs is omitted",
    )
    plan_parser.add_argument("--independent-stages", action="store_true",
                             help="retain the legacy per-arm T/Q/P chains instead of shared artifacts")
    plan_parser.add_argument("--synthetic", action="store_true")
    plan_parser.add_argument("--smoke-updates", type=int)
    run_parser = commands.add_parser("run", help="validate a plan or explicitly execute it")
    run_parser.add_argument("--plan", required=True)
    run_parser.add_argument("--execute", action="store_true")
    run_parser.add_argument("--setting", default="all", choices=("all", *ARMS, *SECONDARY_SETTINGS))
    run_parser.add_argument(
        "--k", nargs="+", type=int, choices=CAMPAIGN_K_VALUES, metavar="K",
        help="select one or more K values; omit to include every K in the selected setting",
    )
    run_parser.add_argument(
        "--through-stage", choices=("T", "Q", "P", "J"), default="J",
        help="execute selected shared stages up to this stage (default: J)",
    )
    run_parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    if args.command == "plan":
        lineage_values = (args.lineage_reference, args.lineage_reference_sha256,
                          args.lineage_source, args.lineage_source_sha256)
        if any(lineage_values) and not all(lineage_values):
            parser.error("external lineage requires both reference/source paths and SHA-256 pins")
        lineage_reference = ({
            "reference_path": args.lineage_reference,
            "reference_sha256": args.lineage_reference_sha256,
            "source_path": args.lineage_source,
            "source_sha256": args.lineage_source_sha256,
        } if all(lineage_values) else None)
        if args.runs and (
            args.primary_only or args.without_baselines or args.independent_stages or args.legacy_catalogue
        ):
            parser.error(
                "--runs cannot be combined with --primary-only, --without-baselines, "
                "--independent-stages, or --legacy-catalogue"
            )
        legacy_selection_flags = (
            args.primary_only or args.without_baselines or args.independent_stages
            or args.legacy_catalogue or tuple(args.k) != CAMPAIGN_K_ORDER
        )
        run_values = args.runs
        if run_values is None and not legacy_selection_flags:
            run_values = list(DEFAULT_REDUCED_RUNS)
        dataset_names = ("acdc", "mnms") if args.dataset == "both" else (args.dataset,)
        protocol_names = ("A", "B") if args.protocol == "both" else (args.protocol,)
        manifest_inputs = _parse_manifest_flags(args.manifest)
        cross_inputs = _parse_manifest_flags(args.cross_eval_manifest, cross=True)
        if args.synthetic and (manifest_inputs or cross_inputs):
            parser.error("synthetic smoke plans cannot point at patient manifests")
        if args.overlap_acdc is not None and args.overlap_mnms is not None:
            args.overlap_acdc = args.overlap_acdc == "yes"
            args.overlap_mnms = args.overlap_mnms == "yes"
        else:
            args.overlap_acdc = args.overlap_mnms = None
        plan_path = create_campaign_plan(
            out=args.out, datasets=dataset_names, protocols=protocol_names, ks=tuple(args.k), seed=args.seed,
            manifest_inputs=manifest_inputs, cross_inputs=cross_inputs,
            teacher_provenance=_parse_teacher_provenance(args),
            include_secondary=not args.primary_only, include_baselines=not args.without_baselines,
            synthetic=args.synthetic, smoke_updates=args.smoke_updates,
            share_stages=not args.independent_stages,
            runs=tuple(run_values) if run_values else None,
            lineage_reference=lineage_reference,
        )
        print(json.dumps({"status": "NOT_RUN", "plan": str(plan_path), "sha256": _sha256(plan_path)}, indent=2))
    else:
        result = execute_campaign(
            args.plan, execute=args.execute, setting=args.setting,
            k=tuple(args.k) if args.k is not None else None,
            through_stage=args.through_stage, device=args.device,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
