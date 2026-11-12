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
)

PLAN_SCHEMA = "self_audit.candidate_q3d.campaign.v1"
STATE_SCHEMA = "self_audit.candidate_q3d.campaign_state.v1"
DEFAULT_REPORT_DIR = ROOT / "reports/q3d-implementation-20261112/campaign"
CAMPAIGN_K_ORDER = (5, 10, 2)
PRIMARY_SETTINGS = PRIMARY_ARMS
SECONDARY_SETTINGS = SECONDARY_ARMS + ("v1_stride4", "v1_residual2")
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
    "mednext_supervised": {
        "runner": "self_audit.candidate_q3d.baselines.run_mednext_supervised",
        "entrypoint": "train.py",
        "method": "candidate_q3d",
        "arm": "mednext_supervised",
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
                raise ValueError("manifest key must be DATASET:ROLE")
            dataset, role = key.split(":", 1)
            if dataset not in {"acdc", "mnms"} or role not in {"train", "teacher_train", "development", "test"}:
                raise ValueError("manifest key must be acdc|mnms:train|teacher_train|development|test")
            parsed.setdefault(dataset, {})[role] = path
    return parsed


def _set_manifest_inputs(config: dict[str, Any], manifest_inputs: Mapping[str, Mapping[str, str]],
                         cross_inputs: Mapping[str, Mapping[str, str]]) -> None:
    dataset_name = config["dataset"]["name"]
    for role, path_value in manifest_inputs.get(dataset_name, {}).items():
        config["dataset"]["manifests"][role] = _manifest_ref(path_value)
    known = {item["dataset"]: item for item in config["dataset"]["cross_evaluation"][:2]}
    for target, roles in cross_inputs.items():
        if "test" in roles:
            known[target]["manifest"] = _manifest_ref(roles["test"])
    if config["dataset"]["manifests"]["test"]["status"] == "READY":
        known[dataset_name]["manifest"] = copy.deepcopy(config["dataset"]["manifests"]["test"])


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


def _set_teacher_source(config: dict[str, Any], *, source_repo: str | None,
                        source_commit: str | None) -> None:
    teacher = config["model"]["teacher"]
    if teacher["architecture"] != "mednext_l_k3":
        return
    if source_repo is not None:
        teacher["source_repo"] = str(Path(source_repo).expanduser().resolve())
    if source_commit is not None:
        teacher["source_commit"] = source_commit


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
                    smoke: bool, smoke_updates: int | None,
                    teacher_source_repo: str | None = None,
                    teacher_source_commit: str | None = None) -> dict[str, Any]:
    arm = "v1" if setting in {"v1_stride4", "v1_residual2"} else setting
    source_stride = 4 if setting == "v1_stride4" else (16 if setting == "s1" else 8)
    config = canonical_config(
        arm, dataset=dataset, protocol=protocol, k=k, seed=seed,
        source_stride=source_stride, residual2=setting == "v1_residual2",
    )
    _set_teacher_source(config, source_repo=teacher_source_repo, source_commit=teacher_source_commit)
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
                   a3d_update_match: bool | None = None) -> tuple[str, str | None]:
    if synthetic:
        if config["arm"] == "a3d_teacher_free" and a3d_update_match is not True:
            return "NOT_RUN", "A-3D teacher_free requires an exact synthetic J budget"
        return "SMOKE_READY", None
    for role in ("train", "development"):
        if config["dataset"]["manifests"][role]["status"] != "READY":
            return "NOT_RUN", f"{role} manifest path and SHA-256 are not configured"
    if (stage == "T" and config["protocol"]["variant"] == "B"
            and config["arm"] != "mednext_supervised"
            and config["dataset"]["manifests"]["teacher_train"]["status"] != "READY"):
        return "NOT_RUN", "Protocol B teacher training requires a separate all-TRAIN manifest"
    if config["arm"] == "a3d_teacher_free" and a3d_update_match is not True:
        return "NOT_RUN", "A-3D teacher_free labeled-frame count cannot match the declared J budget exactly"
    if stage in {"T", "Q", "P", "J"} and config["arm"] in {
        "c1", "v1", "v1u", "v0", "b1", "e1", "o1", "s1", "mednext_supervised",
    }:
        source = config["model"]["teacher"]["pretrained"]
        if not isinstance(source, Mapping) or source.get("provenance_status") != "READY":
            return "NOT_RUN", "pretrained teacher path/hash/dataset/overlap provenance is not configured"
    return "READY", None


def _add_experiment(plan: dict[str, Any], *, setting: str, dataset: str, protocol: str, k: int,
                    seed: int, out_root: Path, manifest_inputs: Mapping[str, Mapping[str, str]],
                    cross_inputs: Mapping[str, Mapping[str, str]], teacher_provenance: Mapping[str, Any] | None,
                    synthetic: bool, smoke_updates: int | None,
                    teacher_source_repo: str | None, teacher_source_commit: str | None) -> None:
    experiment_id = _experiment_id(dataset, protocol, k, seed, setting)
    config = _setting_recipe(setting, dataset=dataset, protocol=protocol, k=k, seed=seed,
                             smoke=synthetic, smoke_updates=smoke_updates,
                             teacher_source_repo=teacher_source_repo,
                             teacher_source_commit=teacher_source_commit)
    _set_manifest_inputs(config, manifest_inputs, cross_inputs)
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
                                           a3d_update_match=a3d_update_match)
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
                         teacher_source_repo: str | None = None,
                         teacher_source_commit: str | None = None,
                         include_secondary: bool = True, include_baselines: bool = True,
                         synthetic: bool = False, smoke_updates: int | None = None) -> Path:
    """Build a hash-pinned campaign plan and per-experiment nested recipes."""
    if not datasets or any(name not in {"acdc", "mnms"} for name in datasets):
        raise ValueError("datasets must explicitly select acdc and/or mnms")
    if len(set(datasets)) != len(datasets):
        raise ValueError("datasets cannot contain duplicates")
    if not protocols or any(name not in {"A", "B"} for name in protocols):
        raise ValueError("protocols must be A and/or B")
    if len(set(protocols)) != len(protocols):
        raise ValueError("protocols cannot contain duplicates")
    if not ks or any(type(k) is not int or k not in K_VALUES for k in ks):
        raise ValueError("K values must be selected from 2, 5, and 10")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative")
    if smoke_updates is not None and (type(smoke_updates) is not int or smoke_updates < 1):
        raise ValueError("smoke_updates must be a positive integer")
    if smoke_updates is not None and not synthetic:
        raise ValueError("smoke update overrides require --synthetic")
    if synthetic and smoke_updates is None:
        smoke_updates = 1
    manifest_inputs = copy.deepcopy(dict(manifest_inputs or {}))
    cross_inputs = copy.deepcopy(dict(cross_inputs or {}))
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
            "k_order": [k for k in CAMPAIGN_K_ORDER if k in ks],
            "seed": seed,
            "synthetic": synthetic,
            "smoke_updates": smoke_updates,
            "primary_settings": list(PRIMARY_SETTINGS),
            "secondary_settings": list(SECONDARY_SETTINGS) if include_secondary else [],
            "baseline_settings": list(BASELINE_ARMS) if include_baselines else [],
        },
        "experiments": [],
        "stage_jobs": [],
        "cross_evaluation": [],
        "baseline_launches": copy.deepcopy(BASELINE_LAUNCHES),
        "evidence": {
            "patient_training": "NOT_RUN",
            "gpu_validation": "NOT_RUN",
            "real_data_metrics": "NOT_RUN",
            "novelty_claim": "NOT_CLAIMED",
        },
    }
    settings_by_group = [("primary", PRIMARY_SETTINGS)]
    if include_secondary:
        settings_by_group.append(("secondary", SECONDARY_SETTINGS))
    if include_baselines:
        settings_by_group.append(("baseline", BASELINE_ARMS))
    ordered_ks = [k for k in CAMPAIGN_K_ORDER if k in ks]
    if len(ordered_ks) != len(set(ks)):
        raise ValueError("K values contain duplicates or unsupported ordering entries")
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
                            smoke_updates=smoke_updates, teacher_source_repo=teacher_source_repo,
                            teacher_source_commit=teacher_source_commit,
                        )
    _add_cross_evaluation_records(plan, datasets=list(datasets), cross_inputs=cross_inputs,
                                  manifest_inputs=manifest_inputs)
    plan["ordering"] = {
        "stage_order": ["T", "Q", "P", "J"],
        "group_order": [group for group, _ in settings_by_group],
        "dataset_order": plan["selection"]["dataset_order"],
        "k_order": plan["selection"]["k_order"],
        "secondary_after_primary": True,
    }
    plan_path = out_root / "PLAN.json"
    _write_json(plan_path, plan)
    validate_campaign_plan(plan_path)
    return plan_path


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
    exp_keys = [(dataset_rank[item["dataset"]], group_rank[item["group"]],
                 protocol_rank[item["protocol"]], k_rank[item["k"]],
                 setting_rank[item["group"]][item["setting"]]) for item in experiments]
    if exp_keys != sorted(exp_keys):
        raise ValueError("campaign ordering must put ACDC first, K=5/10/2, and secondaries after primaries")
    for experiment in experiments:
        config_path = Path(experiment["config_path"])
        if not config_path.is_file() or _sha256(config_path) != experiment["config_sha256"]:
            raise ValueError(f"campaign recipe missing or changed: {config_path}")
        config = load_config(config_path)
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
            previous = None
            for record in stage_records:
                expected_dependencies = [previous] if previous else []
                if record["dependencies"] != expected_dependencies:
                    raise ValueError(f"stage dependency chain is invalid for {record['id']}")
                stage_rank = {name: index for index, name in enumerate(("T", "Q", "P", "J"))}
                if previous and stage_rank[job_by_id[previous]["stage"]] >= stage_rank[record["stage"]]:
                    raise ValueError("stage order must be T then Q then P then J")
                previous = record["id"]
    comparisons: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for experiment in experiments:
        if experiment["setting"] in PRIMARY_ARMS + SECONDARY_ARMS:
            key = (experiment["dataset"], experiment["protocol"], experiment["k"], experiment["seed"])
            comparisons.setdefault(key, []).append(load_config(experiment["config_path"]))
    for configs in comparisons.values():
        validate_budget_update_equality(configs, smoke=bool(selection["synthetic"]))
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
                 device: str, synthetic: bool) -> list[str]:
    command = [sys.executable, str(ROOT / "train.py"), "--config", str(config_path),
               "--out", experiment["output"], "--device", device]
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
                     device: str = "cuda",
                     runner: Any = subprocess.run) -> dict[str, Any]:
    """Validate or explicitly execute selected settings; the trainer owns stage transitions."""
    plan_path = Path(path).expanduser().resolve()
    plan = validate_campaign_plan(plan_path)
    if setting != "all" and setting not in ARMS and setting not in SECONDARY_SETTINGS:
        raise ValueError(f"unknown setting: {setting}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if not execute:
        return {"status": "PLAN_VALIDATED_NOT_RUN", "plan": str(plan_path),
                "selected_setting": setting,
                "patient_training": "NOT_RUN", "jobs": len(plan["stage_jobs"])}
    synthetic = bool(plan["selection"]["synthetic"])
    if not synthetic and device == "cpu":
        raise ValueError("real campaign execution requires explicit CUDA; CPU is reserved for synthetic smoke")
    plan_sha = _sha256(plan_path)
    state_path = plan_path.parent / "RUN_STATE.json"
    state = _read_state(state_path, plan, plan_sha)
    jobs_by_experiment: dict[str, list[dict[str, Any]]] = {}
    for job in plan["stage_jobs"]:
        jobs_by_experiment.setdefault(job["experiment_id"], []).append(job)
    selected = [experiment for experiment in plan["experiments"]
                if setting == "all" or experiment["setting"] == setting]
    if not selected:
        raise ValueError("selection did not match any campaign settings")
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
            elif experiment["arm"] == "mednext_supervised":
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
            "completed_settings": completed, "patient_training": "SYNTHETIC_ONLY" if synthetic else "EXECUTED"}


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
    plan_parser.add_argument("--k", nargs="+", type=int, choices=K_VALUES, default=list(CAMPAIGN_K_ORDER))
    plan_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    plan_parser.add_argument("--manifest", action="append", default=[], metavar="DATASET:ROLE=PATH")
    plan_parser.add_argument("--cross-eval-manifest", action="append", default=[], metavar="DATASET=PATH")
    plan_parser.add_argument("--teacher-weights")
    plan_parser.add_argument("--teacher-weights-sha256")
    plan_parser.add_argument("--teacher-weights-source-url")
    plan_parser.add_argument("--teacher-source-repo")
    plan_parser.add_argument("--teacher-source-commit")
    plan_parser.add_argument("--pretraining-dataset")
    plan_parser.add_argument("--overlap-acdc", choices=("yes", "no"))
    plan_parser.add_argument("--overlap-mnms", choices=("yes", "no"))
    plan_parser.add_argument("--primary-only", action="store_true")
    plan_parser.add_argument("--without-baselines", action="store_true")
    plan_parser.add_argument("--synthetic", action="store_true")
    plan_parser.add_argument("--smoke-updates", type=int)
    run_parser = commands.add_parser("run", help="validate a plan or explicitly execute it")
    run_parser.add_argument("--plan", required=True)
    run_parser.add_argument("--execute", action="store_true")
    run_parser.add_argument("--setting", default="all", choices=("all", *ARMS, *SECONDARY_SETTINGS))
    run_parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    if args.command == "plan":
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
            teacher_source_repo=args.teacher_source_repo,
            teacher_source_commit=args.teacher_source_commit,
            include_secondary=not args.primary_only, include_baselines=not args.without_baselines,
            synthetic=args.synthetic, smoke_updates=args.smoke_updates,
        )
        print(json.dumps({"status": "NOT_RUN", "plan": str(plan_path), "sha256": _sha256(plan_path)}, indent=2))
    else:
        result = execute_campaign(args.plan, execute=args.execute, setting=args.setting,
                                  device=args.device)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
