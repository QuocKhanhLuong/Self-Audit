"""Strict configuration contract for the research-only Q-3D candidate.

The JSON recipes intentionally preserve unknown dataset and pretrained-weight
provenance as NOT_RUN.  A plan can therefore be inspected without claiming
that a real-data run is ready; execution validates every required path and
hash before it launches a teacher stage.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

SCHEMA = "self_audit.candidate_q3d.v1"
METHOD = "candidate_q3d"
PRIMARY_ARMS = ("c0", "c1", "v1", "v1u")
SECONDARY_ARMS = ("v0", "b1", "e1", "o1", "s1")
BASELINE_ARMS = ("mean_teacher", "a3d_teacher_free", "mednext_supervised")
ARMS = PRIMARY_ARMS + SECONDARY_ARMS + BASELINE_ARMS
K_VALUES = (2, 5, 10)
DEFAULT_SEED = 42
DEFAULT_STAGE_UPDATES = {"T": 8000, "Q": 1000, "P": 1, "J": 8000}
UPSTREAM_MEDNEXT_COMMIT = "0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba"
DEFAULT_MEDNEXT_SOURCE_REPO: str | None = None
_EXPECTED_CLASSES = ["BG", "RV", "MYO", "LV"]
_FEATURE_CHANNELS = {4: 128, 8: 256, 16: 512}
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OFFICIAL_WEIGHT_HOSTS = {"github.com", "raw.githubusercontent.com", "zenodo.org", "huggingface.co"}

_ROOT_KEYS = {
    "schema", "method", "arm", "seed", "dataset", "protocol", "model",
    "training", "tokenizer", "physiology", "losses", "augmentation",
    "checkpoint", "runtime", "baselines",
}
_MANIFEST_KEYS = {"path", "sha256", "status"}
_BASELINE_CATALOG = {
    "mean_teacher": {
        "entrypoint": "train.py",
        "method": METHOD,
        "arm": "mean_teacher",
        "runner": "self_audit.candidate_q3d.engine.train",
        "status": "ENGINE_ARM",
        "budget_stage": "J",
        "evaluation_name": "mean_teacher_student",
    },
    "a3d_teacher_free": {
        "entrypoint": "train.py",
        "method": METHOD,
        "arm": "a3d_teacher_free",
        "runner": "self_audit.candidate_q3d.baselines.run_a3d_teacher_free",
        "config_template": "configs/candidate_a3d/teacher_free_acdc_k8.json",
        "status": "IMPLEMENTED",
        "budget_stage": "J",
        "evaluation_name": "a3d_teacher_free",
        "resume_scope": "trainer_output_state",
    },
    "mednext_supervised": {
        "entrypoint": "train.py",
        "method": METHOD,
        "arm": "mednext_supervised",
        "runner": "self_audit.candidate_q3d.baselines.run_mednext_supervised",
        "status": "IMPLEMENTED",
        "budget_stage": "T",
        "evaluation_name": "mednext_supervised",
        "resume_from": None,
        "resume_scope": "latest_checkpoint_in_output",
    },
}

_ARM_SETTINGS: dict[str, dict[str, Any]] = {
    "c0": {"interface": "continuous", "teacher_signal": "none", "physiology": False,
            "tokenizer": False, "stages": ("J",)},
    "c1": {"interface": "continuous", "teacher_signal": "feature_kd", "physiology": True,
            "tokenizer": False, "stages": ("T", "P", "J")},
    "v1": {"interface": "vq", "teacher_signal": "codekd", "physiology": True,
            "tokenizer": True, "stages": ("T", "Q", "P", "J")},
    "v1u": {"interface": "vq", "teacher_signal": "codekd", "physiology": False,
            "tokenizer": True, "stages": ("T", "Q", "J")},
    "v0": {"interface": "vq", "teacher_signal": "none", "physiology": False,
            "tokenizer": True, "stages": ("T", "Q", "J")},
    "b1": {"interface": "vq", "teacher_signal": "codekd", "physiology": True,
            "tokenizer": True, "stages": ("T", "Q", "P", "J")},
    "e1": {"interface": "vq", "teacher_signal": "codekd", "physiology": True,
            "tokenizer": True, "stages": ("T", "Q", "P", "J")},
    "o1": {"interface": "vq", "teacher_signal": "codekd_online_teacher", "physiology": True,
            "tokenizer": True, "stages": ("T", "Q", "P", "J")},
    "s1": {"interface": "vq", "teacher_signal": "codekd", "physiology": True,
            "tokenizer": True, "stages": ("T", "Q", "P", "J")},
    "mean_teacher": {"interface": "continuous", "teacher_signal": "student_ema", "physiology": False,
                     "tokenizer": False, "stages": ("J",)},
    "a3d_teacher_free": {"interface": "candidate_a3d", "teacher_signal": "none", "physiology": False,
                         "tokenizer": False, "stages": ("J",)},
    "mednext_supervised": {"interface": "mednext_supervised", "teacher_signal": "none",
                           "physiology": False, "tokenizer": False, "stages": ("T",)},
}


def _manifest() -> dict[str, Any]:
    return {"path": None, "sha256": None, "status": "NOT_RUN"}


def _joint_stream_count(arm: str) -> int:
    return 1 if arm in {"c0", "a3d_teacher_free", "mednext_supervised"} else 2


def _joint_update_unit(arm: str) -> str:
    return ("L_only_updates" if arm in {"c0", "a3d_teacher_free", "mednext_supervised"}
            else "paired_L_U_updates")


def _stage_accounting(
    stage_updates: Mapping[str, int], arm: str,
) -> tuple[dict[str, int], dict[str, int | None]]:
    """Derive explicit optimizer and stream-microbatch counts from stage units.

    T and Q are optimizer-update budgets, P is a pass budget, and J counts
    student updates. Paired L/U arms contribute two stream microbatches per J
    update; C0 and supervised baselines are L-only. Every J pair/update is one
    optimizer step, with accumulation measured in L/U stream microbatches.
    """
    return (
        {"T": int(stage_updates["T"]), "Q": int(stage_updates["Q"]),
         "P": 0, "J": int(stage_updates["J"])},
        {"T": int(stage_updates["T"]), "Q": int(stage_updates["Q"]),
         "P": None, "J": _joint_stream_count(arm) * int(stage_updates["J"])},
    )


def _base_config(arm: str, *, dataset: str = "acdc", protocol: str = "A",
                 k: int = 5, seed: int = DEFAULT_SEED,
                 source_stride: int | None = None, residual2: bool = False) -> dict[str, Any]:
    if dataset not in {"acdc", "mnms"}:
        raise ValueError("dataset must be acdc or mnms; unnamed CMR datasets stay NOT_RUN")
    if protocol not in {"A", "B"}:
        raise ValueError("protocol must be A or B")
    source = dataset
    target = "mnms" if dataset == "acdc" else "acdc"
    cache_namespace = f"q3d/{dataset}/protocol-{protocol.lower()}/k{k}/seed-{seed}"
    settings = _ARM_SETTINGS[arm]
    teacher_used = arm in {"c1", "v1", "v1u", "v0", "b1", "e1", "o1", "s1", "mednext_supervised"}
    source_stride = (16 if arm == "s1" else 8) if source_stride is None else source_stride
    teacher_online = arm == "o1"
    code_only = arm == "b1"
    adaptive_epa = arm == "e1"
    teacher_architecture = (
        "mednext_l_k3" if teacher_used else
        "ema_student" if arm == "mean_teacher" else
        "candidate_a3d" if arm == "a3d_teacher_free" else "none"
    )
    student_architecture = (
        "mednext_l_k3_supervised" if arm == "mednext_supervised" else
        "candidate_a3d" if arm == "a3d_teacher_free" else "unetrpp_lite_3d"
    )
    if student_architecture == "candidate_a3d":
        decoder_inputs = ["native_volume"]
    elif settings["interface"] == "vq":
        decoder_inputs = ["codes"] if code_only else ["F4", "codes"]
    else:
        decoder_inputs = ["F4"] if student_architecture == "unetrpp_lite_3d" else ["native_volume"]
    setting_namespace = (
        "v1_stride4" if arm == "v1" and source_stride == 4 else
        "v1_residual2" if arm == "v1" and residual2 else arm
    )
    codekd_enabled = arm in {"v1", "v1u", "b1", "e1", "o1", "s1"}
    w_mode = "uniform" if arm == "v1u" else ("physiology" if settings["physiology"] else "none")
    stage_updates = copy.deepcopy(DEFAULT_STAGE_UPDATES)
    stage_optimizer_updates, stage_stream_microbatches = _stage_accounting(stage_updates, arm)
    joint_stream_count = _joint_stream_count(arm)
    return {
        "schema": SCHEMA,
        "method": METHOD,
        "arm": arm,
        "seed": seed,
        "dataset": {
            "name": source,
            "release": "original_2020" if source == "mnms" else "acdc",
            "class_mapping": ["BG", "RV", "MYO", "LV"],
            "manifests": {
                "train": _manifest(),
                "teacher_train": _manifest(),
                "development": _manifest(),
                "test": _manifest(),
            },
            "cross_evaluation": [
                {"dataset": source, "manifest": _manifest()},
                {"dataset": target, "manifest": _manifest()},
                {"dataset": None, "manifest": _manifest(), "status": "NOT_RUN"},
                {"dataset": None, "manifest": _manifest(), "status": "NOT_RUN"},
            ],
            "deferred_cmr_datasets": [
                {"name": None, "status": "NOT_RUN", "reason": "dataset identity not specified in Q-3D spec"},
                {"name": None, "status": "NOT_RUN", "reason": "dataset identity not specified in Q-3D spec"},
            ],
        },
        "protocol": {
            "variant": protocol,
            "k": k,
            "teacher_label_budget": "k" if protocol == "A" else "all_train",
            "student_label_budget": "k",
            "unlabeled_source": "remaining_train_image_only",
            "seed": seed,
            "cache_isolation_key": cache_namespace,
        },
        "model": {
            "student": {
                "architecture": student_architecture,
                "input_shape": "[B,1,D,H,W]",
                "input_channels": 1,
                "classes": ["BG", "RV", "MYO", "LV"],
                "stage_channels": [16, 32, 64, 128, 256],
                "z_stride": 1,
                "epa_heads": 4,
                "epa_support": "adaptive" if adaptive_epa else "uniform_fixed",
                "epa_pool_zyx": [4, 4, 4],
                "residual2": {"enabled": residual2, "zero_init": True},
                "decoder_inputs": decoder_inputs,
                "code_gate_init": 0.0,
                "dense_skip": not code_only,
                "sdf_head_train_only": True,
                "max_parameters": 10000000,
                "native_grid": True,
                "crop": False,
                "resample": False,
            },
            "teacher": {
                "architecture": teacher_architecture,
                "heads": ["region", "boundary_sdf"] if teacher_architecture == "mednext_l_k3" else [],
                "feature_stride": source_stride if teacher_architecture == "mednext_l_k3" else None,
                "feature_channels": _FEATURE_CHANNELS[source_stride] if teacher_architecture == "mednext_l_k3" else None,
                "source_repo": DEFAULT_MEDNEXT_SOURCE_REPO if teacher_architecture == "mednext_l_k3" else None,
                "source_commit": UPSTREAM_MEDNEXT_COMMIT if teacher_architecture == "mednext_l_k3" else None,
                "freeze_joint": not teacher_online,
                "online_joint": teacher_online,
                "shared_trunk": teacher_architecture == "mednext_l_k3",
                "pretrained": ({
                    "weights_path": None,
                    "weights_sha256": None,
                    "weights_source_url": None,
                    "pretraining_dataset": None,
                    "overlap_acdc": None,
                    "overlap_mnms": None,
                    "class_mapping": list(_EXPECTED_CLASSES),
                    "provenance_status": "NOT_FOUND_OFFICIAL",
                } if teacher_architecture == "mednext_l_k3" else None),
            },
        },
        "training": {
            "stages": ["T", "Q", "P", "J"],
            "stage_modes": {"T": "updates", "Q": "updates", "P": "single_forward_pass", "J": "updates"},
            "stage_updates": stage_updates,
            "stage_update_units": {
                "T": "optimizer_updates",
                "Q": "optimizer_updates_after_separate_init_pass",
                "P": "full_U_patient_cine_passes",
                "J": _joint_update_unit(arm),
            },
            "stage_optimizer_updates": stage_optimizer_updates,
            "stage_stream_microbatches": stage_stream_microbatches,
            "gradient_accumulation_unit": "stream_microbatches_per_optimizer_step",
            "teacher_warmup_fraction": 0.05,
            "teacher_boundary_lr": 1.0e-3,
            "teacher_encoder_lr": 1.0e-4,
            "teacher_head_lr": 1.0e-3,
            "teacher_joint_lr_multiplier": 0.1,
            "student_optimizer": "AdamW",
            "student_lr": 3.0e-4,
            "weight_decay": 1.0e-4,
            "warmup_fraction": 0.05,
            "gradient_clip_norm": 1.0,
            "gradient_accumulation": joint_stream_count,
            "batch_per_stream": 1,
            "teacher_frozen_joint": not teacher_online,
            "checkpoint_selection": "development_patient_macro_fg_dice",
            "checkpoint_evaluation_interval": 400,
            "checkpoint_evaluation_unit": _joint_update_unit(arm),
        },
        "tokenizer": {
            "enabled": bool(settings["tokenizer"]),
            "initialization_volumes": 200,
            "codebook_size": 256,
            "dimension": 64,
            "adapter_hidden": 128,
            "temperature": 0.1,
            "commitment_beta": 0.25,
            "anatomy_loss_weight": 0.5,
            "ema_decay": 0.99,
            "dead_code_fraction": 0.001,
            "usage_window_updates": 200,
            "minimum_perplexity": 32.0,
            "maximum_dead_code_fraction": 0.25,
            "minimum_class_codes": 4,
            "source_feature": {
                4: "stride4_decoder_block",
                8: "stride8_decoder_block_after_bottleneck",
                16: "bottleneck_stride16",
            }[source_stride],
            "codebook_frozen_after_stage": "Q",
        },
        "physiology": {
            "enabled": bool(settings["physiology"]),
            "stroke_balance": True,
            "certainty_threshold": 0.7,
            "tau_stroke_volume": 0.15,
            "tau_myocardium": 0.10,
            "minimum_lv_ed_ml": 10.0,
            "ef_range_percent": [5.0, 90.0],
            "cache_refresh_joint_epochs": 10 if teacher_online else None,
            "use_ed_es_metadata": False,
        },
        "losses": {
            "labeled_supervised_coefficient": 1.0,
            "supervised_divide_by_stream_count": False,
            "l_u_auxiliary_reduction": "mean",
            "region_ce_dice": 1.0,
            "boundary_ce_dice": 1.0,
            "balanced_sdf": 1.0,
            "sdf_radius_mm": 5.0,
            "sdf_near_boundary_mm": 2.5,
            "sdf_near_far_alpha": 0.5,
            "continuous_feature_kd": 1.0 if arm == "c1" else 0.0,
            "codekd": 1.0 if codekd_enabled else 0.0,
            "codekd_ramp_fraction": 0.10,
            "commitment_beta": 0.25,
            "teacher_confidence_weighting": True,
            "w_mode": w_mode,
            "uniform_w": 1.0 if w_mode == "uniform" else None,
            "codekd_class_balance": True,
            "background_class_equal_weight": True,
            "skip_zero_weight_classes": True,
        },
        "augmentation": {
            "flip_y_probability": 0.5,
            "flip_x_probability": 0.5,
            "rotate90_probability": 0.5,
            "rotate90_requires_equal_spacing": True,
            "spatial_transform_shared": True,
            "recompute_sdf_after_spatial_transform": True,
            "intensity_probability_labeled": 0.5,
            "intensity_probability_unlabeled_student": 1.0,
            "gamma_range": [0.7, 1.5],
            "voxel_noise_std": 0.03,
            "unlabeled_teacher_intensity_augmentation": False,
            "preserve_native_geometry": True,
        },
        "checkpoint": {
            "namespace": cache_namespace,
            "isolate_dataset": True,
            "isolate_protocol": True,
            "isolate_k": True,
            "isolate_seed": True,
            "stage_outputs": {name: f"{cache_namespace}/{setting_namespace}/{name.lower()}" for name in ("T", "Q", "P", "J")},
            "resume_scope": "stage_checkpoint_only",
        },
        "runtime": {
            "synthetic": False,
            "device": "cuda",
            "smoke_update_overrides": {"T": None, "Q": None, "P": None, "J": None},
            "smoke_initialization_volumes": None,
            "synthetic_gate_override": False,
            "allow_unverified_teacher": False,
        },
        "baselines": copy.deepcopy(_BASELINE_CATALOG),
    }


def canonical_config(arm: str, *, dataset: str = "acdc", protocol: str = "A",
                     k: int = 5, seed: int = DEFAULT_SEED, source_stride: int | None = None,
                     residual2: bool = False) -> dict[str, Any]:
    """Build a fresh canonical nested config for a declared arm and setting."""
    if arm not in ARMS:
        raise ValueError(f"unknown Q-3D arm: {arm}")
    if k not in K_VALUES:
        raise ValueError(f"k must be one of {K_VALUES}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return validate_config(_base_config(arm, dataset=dataset, protocol=protocol, k=k, seed=seed,
                                        source_stride=source_stride, residual2=residual2))


def _exact_mapping(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        missing = sorted(keys - set(value)) if isinstance(value, Mapping) else sorted(keys)
        extra = sorted(set(value) - keys) if isinstance(value, Mapping) else []
        raise ValueError(f"{name} must have exact fields (missing={missing}, extra={extra})")
    return value


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> None:
    floor = 0 if allow_zero else 1
    if type(value) is not int or value < floor:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")


def _finite_number(value: Any, name: str, *, minimum: float | None = None,
                   maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    if minimum is not None and float(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and float(value) > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def _validate_official_weight_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("weights_source_url must be an explicit HTTPS URL")
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").lower()
    official_host = any(host == root or host.endswith("." + root) for root in _OFFICIAL_WEIGHT_HOSTS)
    if (parsed.scheme != "https" or not official_host or parsed.username or parsed.password):
        raise ValueError("weights_source_url must use official GitHub, Zenodo, or Hugging Face HTTPS hosting")
    return value.strip()


def _verify_mednext_source_checkout(source_repo: str, source_commit: str) -> Path:
    path = Path(source_repo).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"verified MedNeXt source checkout is missing: {path}")
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    if head.returncode != 0 or head.stdout.strip() != source_commit:
        actual = head.stdout.strip() if head.returncode == 0 else "unavailable"
        raise ValueError(f"MedNeXt source checkout commit mismatch: expected {source_commit}, got {actual}")
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=False, capture_output=True, text=True,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError("MedNeXt source checkout must be clean")
    return path


def _validate_runtime(value: Any) -> dict[str, Any]:
    runtime = dict(_exact_mapping(value, {
        "synthetic", "device", "smoke_update_overrides", "smoke_initialization_volumes",
        "synthetic_gate_override", "allow_unverified_teacher",
    }, "runtime"))
    if type(runtime["synthetic"]) is not bool or type(runtime["allow_unverified_teacher"]) is not bool:
        raise ValueError("runtime synthetic and provenance override fields must be boolean")
    if runtime["allow_unverified_teacher"]:
        raise ValueError("unverified teacher bypass is forbidden")
    if runtime["device"] not in {"cpu", "cuda"}:
        raise ValueError("runtime.device must be cpu or cuda")
    if type(runtime["synthetic_gate_override"]) is not bool:
        raise ValueError("runtime.synthetic_gate_override must be boolean")
    overrides = _exact_mapping(runtime["smoke_update_overrides"], {"T", "Q", "P", "J"},
                               "runtime.smoke_update_overrides")
    runtime["smoke_update_overrides"] = dict(overrides)
    for stage, count in overrides.items():
        if count is not None:
            _positive_int(count, f"runtime.smoke_update_overrides.{stage}")
    init_count = runtime["smoke_initialization_volumes"]
    if init_count is not None:
        _positive_int(init_count, "runtime.smoke_initialization_volumes")
        if init_count > 200:
            raise ValueError("synthetic initialization override cannot exceed the pinned 200-volume design")
    has_smoke_override = (
        any(count is not None for count in overrides.values())
        or init_count is not None
        or runtime["synthetic_gate_override"]
    )
    if has_smoke_override and not runtime["synthetic"]:
        raise ValueError("synthetic update/init/gate overrides cannot affect a real run")
    return runtime


def _validate_manifest(value: Any, name: str, *, required: bool = False) -> dict[str, Any]:
    record = dict(_exact_mapping(value, _MANIFEST_KEYS, name))
    if record["status"] not in {"READY", "NOT_RUN"}:
        raise ValueError(f"{name}.status must be READY or NOT_RUN")
    path, digest = record["path"], record["sha256"]
    if (path is None) != (digest is None):
        raise ValueError(f"{name} path and sha256 must be supplied together")
    if path is not None:
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{name}.path must be a non-empty string")
        if not isinstance(digest, str) or not _HEX_SHA256.fullmatch(digest):
            raise ValueError(f"{name}.sha256 must be lowercase SHA-256 hex")
    if required and record["status"] != "READY":
        raise ValueError(f"{name} must be READY for a real run")
    if record["status"] == "READY" and path is None:
        raise ValueError(f"{name} READY status requires path and sha256")
    return record


def _validate_dataset(value: Any, *, real_run: bool) -> dict[str, Any]:
    keys = {"name", "release", "class_mapping", "manifests", "cross_evaluation", "deferred_cmr_datasets"}
    dataset = dict(_exact_mapping(value, keys, "dataset"))
    if dataset["name"] not in {"acdc", "mnms"}:
        raise ValueError("dataset.name must be acdc or original mnms")
    expected_release = "original_2020" if dataset["name"] == "mnms" else "acdc"
    if dataset["release"] != expected_release:
        raise ValueError("dataset.release must distinguish original M&Ms 2020 from M&Ms-2")
    if dataset["class_mapping"] != ["BG", "RV", "MYO", "LV"]:
        raise ValueError("Q-3D class mapping must be BG/RV/MYO/LV")
    manifests = _exact_mapping(dataset["manifests"], {"train", "teacher_train", "development", "test"}, "dataset.manifests")
    normalized_manifests = {}
    for name, record in manifests.items():
        normalized_manifests[name] = _validate_manifest(record, f"dataset.manifests.{name}",
                                                        required=real_run and name in {"train", "development"})
    cross = dataset["cross_evaluation"]
    if not isinstance(cross, list) or len(cross) != 4:
        raise ValueError("cross_evaluation must declare ACDC, M&Ms, and two unnamed NOT_RUN slots")
    seen = set()
    normalized_cross = []
    for index, item in enumerate(cross):
        item_keys = {"dataset", "manifest"} if index < 2 else {"dataset", "manifest", "status"}
        entry = dict(_exact_mapping(item, item_keys, f"cross_evaluation[{index}]"))
        name = entry["dataset"]
        if index < 2:
            if name not in {"acdc", "mnms"} or name in seen:
                raise ValueError("known cross-evaluation targets must explicitly name ACDC and M&Ms once")
            seen.add(name)
            entry["manifest"] = _validate_manifest(entry["manifest"], f"cross_evaluation.{name}")
        else:
            if name is not None or entry["status"] != "NOT_RUN":
                raise ValueError("the two unnamed CMR targets must remain NOT_RUN; do not invent names")
            entry["manifest"] = _validate_manifest(entry["manifest"], f"cross_evaluation.unknown[{index}]")
        normalized_cross.append(entry)
    if seen != {"acdc", "mnms"}:
        raise ValueError("cross-evaluation manifests must explicitly enumerate ACDC and original M&Ms")
    deferred = dataset["deferred_cmr_datasets"]
    if (not isinstance(deferred, list) or len(deferred) != 2
            or any(not isinstance(item, Mapping) or set(item) != {"name", "status", "reason"}
                   or item.get("name") is not None or item.get("status") != "NOT_RUN"
                   for item in deferred)):
        raise ValueError("exactly two unnamed CMR datasets must be marked NOT_RUN")
    dataset["manifests"] = normalized_manifests
    dataset["cross_evaluation"] = normalized_cross
    dataset["deferred_cmr_datasets"] = [dict(item) for item in deferred]
    return dataset


def _validate_model(value: Any, *, arm: str) -> dict[str, Any]:
    model = dict(_exact_mapping(value, {"student", "teacher"}, "model"))
    student_keys = {
        "architecture", "input_shape", "input_channels", "classes", "stage_channels", "z_stride",
        "epa_heads", "epa_support", "epa_pool_zyx", "residual2", "decoder_inputs", "code_gate_init",
        "dense_skip", "sdf_head_train_only", "max_parameters", "native_grid", "crop", "resample",
    }
    student = dict(_exact_mapping(model["student"], student_keys, "model.student"))
    if student["architecture"] not in {"unetrpp_lite_3d", "mednext_l_k3_supervised", "candidate_a3d"}:
        raise ValueError("unsupported Q-3D student/baseline architecture")
    if student["input_shape"] != "[B,1,D,H,W]" or student["input_channels"] != 1:
        raise ValueError("Q-3D accepts native [B,1,D,H,W]")
    if student["classes"] != ["BG", "RV", "MYO", "LV"]:
        raise ValueError("student class mapping mismatch")
    expected_student_arch = (
        "mednext_l_k3_supervised" if arm == "mednext_supervised" else
        "candidate_a3d" if arm == "a3d_teacher_free" else "unetrpp_lite_3d"
    )
    if student["architecture"] != expected_student_arch:
        raise ValueError(f"{arm} requires student architecture {expected_student_arch}")
    expected_decoder_inputs = (
        ["native_volume"] if arm in {"a3d_teacher_free", "mednext_supervised"} else
        ["codes"] if arm == "b1" else
        ["F4", "codes"] if _ARM_SETTINGS[arm]["interface"] == "vq" else ["F4"]
    )
    if student["decoder_inputs"] != expected_decoder_inputs:
        raise ValueError("model.student.decoder_inputs must match the selected interface ablation")
    if student["architecture"] == "unetrpp_lite_3d":
        if student["stage_channels"] != [16, 32, 64, 128, 256] or student["z_stride"] != 1:
            raise ValueError("UNETR++-lite stage channels and z stride are pinned by the spec")
        if student["epa_heads"] != 4 or student["epa_pool_zyx"] != [4, 4, 4]:
            raise ValueError("EPA head count and adaptive support are pinned by the spec")
        if student["epa_support"] not in {"uniform_fixed", "adaptive"}:
            raise ValueError("EPA support must be uniform_fixed or the E1 adaptive ablation")
        if student["epa_support"] == "adaptive" and arm != "e1":
            raise ValueError("adaptive EPA support is reserved for arm E1")
        residual = _exact_mapping(student["residual2"], {"enabled", "zero_init"}, "model.student.residual2")
        if type(residual["enabled"]) is not bool or type(residual["zero_init"]) is not bool:
            raise ValueError("residual2 flags must be boolean")
        if residual["enabled"] and not residual["zero_init"]:
            raise ValueError("optional residual2 contribution must be zero-initialized")
        if not isinstance(student["decoder_inputs"], list) or not student["decoder_inputs"]:
            raise ValueError("decoder_inputs must be non-empty")
        expected_inputs = ["codes"] if arm == "b1" else ["F4", "codes"]
        if arm in SECONDARY_ARMS or arm in {"v1", "v1u"}:
            if student["decoder_inputs"] != expected_inputs:
                raise ValueError("decoder inputs must match the declared ablation")
        for boolean_field in ("dense_skip", "sdf_head_train_only", "native_grid", "crop", "resample"):
            if type(student[boolean_field]) is not bool:
                raise ValueError(f"model.student.{boolean_field} must be boolean")
        if student["crop"] or student["resample"] or not student["native_grid"]:
            raise ValueError("Q-3D reference path is native, uncropped, and not resampled")
        if student["dense_skip"] != (arm != "b1"):
            raise ValueError("only B1 removes the dense skip path")
        _positive_int(student["max_parameters"], "model.student.max_parameters")
        _finite_number(student["code_gate_init"], "model.student.code_gate_init")
        if student["code_gate_init"] != 0.0:
            raise ValueError("the code injection gate must start at zero")
    teacher_keys = {
        "architecture", "heads", "feature_stride", "feature_channels", "source_repo",
        "source_commit", "freeze_joint", "online_joint", "shared_trunk", "pretrained",
    }
    teacher = dict(_exact_mapping(model["teacher"], teacher_keys, "model.teacher"))
    expected_teacher_arch = (
        "mednext_l_k3" if arm in {"c1", "v1", "v1u", "v0", "b1", "e1", "o1", "s1", "mednext_supervised"} else
        "ema_student" if arm == "mean_teacher" else
        "candidate_a3d" if arm == "a3d_teacher_free" else "none"
    )
    if teacher["architecture"] != expected_teacher_arch:
        raise ValueError(f"{arm} requires teacher architecture {expected_teacher_arch}")
    if teacher["architecture"] not in {"mednext_l_k3", "ema_student", "candidate_a3d", "none"}:
        raise ValueError("unsupported teacher architecture")
    if teacher["architecture"] == "mednext_l_k3":
        if teacher["heads"] != ["region", "boundary_sdf"] or type(teacher["shared_trunk"]) is not bool:
            raise ValueError("MedNeXt teacher must declare two heads on one shared trunk")
        if teacher["feature_stride"] not in {4, 8, 16}:
            raise ValueError("teacher feature_stride must be 4, 8, or 16")
        if teacher["feature_channels"] != _FEATURE_CHANNELS[teacher["feature_stride"]]:
            raise ValueError("teacher feature_channels must match the selected verified stride adapter")
        if teacher["source_repo"] is not None and (
            not isinstance(teacher["source_repo"], str) or not teacher["source_repo"].strip()
        ):
            raise ValueError("MedNeXt teacher source_repo must be a non-empty path or null placeholder")
        if teacher["source_commit"] != UPSTREAM_MEDNEXT_COMMIT:
            raise ValueError(f"MedNeXt source_commit must be the verified upstream commit {UPSTREAM_MEDNEXT_COMMIT}")
        if teacher["feature_stride"] == 16 and arm != "s1":
            raise ValueError("stride-16 teacher source is reserved for S1")
        if teacher["feature_stride"] == 4 and arm not in {"v1", "c1"}:
            raise ValueError("stride-4 source is an explicitly named C1/V1 source ablation")
        if type(teacher["freeze_joint"]) is not bool or type(teacher["online_joint"]) is not bool:
            raise ValueError("teacher joint mode flags must be boolean")
        if teacher["online_joint"] != (arm == "o1") or teacher["freeze_joint"] == teacher["online_joint"]:
            raise ValueError("only O1 updates the teacher online; all other teacher arms freeze it")
        pretrained = _exact_mapping(teacher["pretrained"], {
            "weights_path", "weights_sha256", "weights_source_url", "pretraining_dataset",
            "overlap_acdc", "overlap_mnms", "class_mapping", "provenance_status",
        }, "model.teacher.pretrained")
        if pretrained["provenance_status"] not in {"READY", "NOT_RUN", "NOT_FOUND_OFFICIAL"}:
            raise ValueError("teacher provenance_status must be READY, NOT_RUN, or NOT_FOUND_OFFICIAL")
        if pretrained["class_mapping"] != _EXPECTED_CLASSES:
            raise ValueError(f"teacher pretrained class_mapping must be {_EXPECTED_CLASSES}")
        source_url = pretrained["weights_source_url"]
        if source_url is not None:
            _validate_official_weight_url(source_url)
        if pretrained["provenance_status"] == "READY":
            if not isinstance(pretrained["weights_path"], str) or not pretrained["weights_path"].strip():
                raise ValueError("READY pretrained provenance requires weights_path")
            if not isinstance(pretrained["weights_sha256"], str) or not _HEX_SHA256.fullmatch(pretrained["weights_sha256"]):
                raise ValueError("READY pretrained provenance requires a lowercase weights_sha256")
            if source_url is None:
                raise ValueError("READY pretrained provenance requires an official HTTPS weights_source_url")
            if not isinstance(pretrained["pretraining_dataset"], str) or not pretrained["pretraining_dataset"].strip():
                raise ValueError("READY pretrained provenance requires pretraining_dataset")
            for field in ("overlap_acdc", "overlap_mnms"):
                if type(pretrained[field]) is not bool:
                    raise ValueError(f"READY pretrained provenance requires explicit {field}")
        elif any(pretrained[field] is not None for field in (
            "weights_path", "weights_sha256", "weights_source_url", "pretraining_dataset",
            "overlap_acdc", "overlap_mnms",
        )):
            raise ValueError("non-READY pretrained provenance must keep unavailable weights and overlap facts null")
    else:
        if (teacher["feature_stride"] is not None or teacher["feature_channels"] is not None
                or teacher["source_repo"] is not None or teacher["source_commit"] is not None
                or teacher["pretrained"] is not None):
            raise ValueError("non-MedNeXt baselines cannot carry MedNeXt feature or checkpoint provenance")
    return {"student": student, "teacher": teacher}


def _validate_training(value: Any, *, arm: str, runtime: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "stages", "stage_modes", "stage_updates", "stage_update_units", "stage_optimizer_updates",
        "stage_stream_microbatches", "gradient_accumulation_unit", "teacher_warmup_fraction", "teacher_boundary_lr",
        "teacher_encoder_lr", "teacher_head_lr", "teacher_joint_lr_multiplier", "student_optimizer",
        "student_lr", "weight_decay", "warmup_fraction", "gradient_clip_norm", "gradient_accumulation",
        "batch_per_stream", "teacher_frozen_joint", "checkpoint_selection", "checkpoint_evaluation_interval",
        "checkpoint_evaluation_unit",
    }
    training = dict(_exact_mapping(value, keys, "training"))
    if training["stages"] != ["T", "Q", "P", "J"]:
        raise ValueError("training.stages must preserve canonical dependency order T,Q,P,J")
    if set(training["stage_modes"]) != {"T", "Q", "P", "J"}:
        raise ValueError("training.stage_modes must cover T,Q,P,J")
    if training["stage_modes"] != {"T": "updates", "Q": "updates", "P": "single_forward_pass", "J": "updates"}:
        raise ValueError("Q-3D stage modes must match the declared stage protocol")
    updates = _exact_mapping(training["stage_updates"], {"T", "Q", "P", "J"}, "training.stage_updates")
    for stage, count in updates.items():
        _positive_int(count, f"training.stage_updates.{stage}")
    expected_units = {
        "T": "optimizer_updates",
        "Q": "optimizer_updates_after_separate_init_pass",
        "P": "full_U_patient_cine_passes",
        "J": _joint_update_unit(arm),
    }
    units = _exact_mapping(training["stage_update_units"], set(expected_units), "training.stage_update_units")
    if dict(units) != expected_units:
        raise ValueError("training.stage_update_units must declare T/Q optimizer updates, separate Q init, P U-pass, and paired J units")
    optimizer_updates, stream_microbatches = _stage_accounting(updates, arm)
    observed_optimizer = _exact_mapping(training["stage_optimizer_updates"], set(optimizer_updates),
                                        "training.stage_optimizer_updates")
    if dict(observed_optimizer) != optimizer_updates:
        raise ValueError("training.stage_optimizer_updates must count one optimizer step per T/Q update and per J pair")
    observed_microbatches = _exact_mapping(training["stage_stream_microbatches"], set(stream_microbatches),
                                           "training.stage_stream_microbatches")
    if dict(observed_microbatches) != stream_microbatches:
        raise ValueError("training.stage_stream_microbatches must count two L/U streams per paired J update")
    for field in ("teacher_warmup_fraction", "warmup_fraction"):
        _finite_number(training[field], f"training.{field}", minimum=0.0, maximum=1.0)
    for field in ("teacher_boundary_lr", "teacher_encoder_lr", "teacher_head_lr", "teacher_joint_lr_multiplier",
                  "student_lr", "weight_decay", "gradient_clip_norm"):
        _finite_number(training[field], f"training.{field}", minimum=0.0)
    if training["student_optimizer"] != "AdamW":
        raise ValueError("student optimizer is pinned to AdamW")
    _positive_int(training["gradient_accumulation"], "training.gradient_accumulation")
    if (training["gradient_accumulation"] != _joint_stream_count(arm)
            or training["gradient_accumulation_unit"] != "stream_microbatches_per_optimizer_step"):
        raise ValueError("J stream accumulation must match the selected L-only or paired L/U update unit")
    _positive_int(training["batch_per_stream"], "training.batch_per_stream")
    _positive_int(training["checkpoint_evaluation_interval"], "training.checkpoint_evaluation_interval")
    if training["checkpoint_selection"] != "development_patient_macro_fg_dice":
        raise ValueError("checkpoint selection must use the locked read-only development metric")
    if training["checkpoint_evaluation_unit"] != _joint_update_unit(arm):
        raise ValueError("development evaluation interval must use the declared J update unit")
    if type(training["teacher_frozen_joint"]) is not bool:
        raise ValueError("training.teacher_frozen_joint must be boolean")
    if training["teacher_frozen_joint"] != (arm != "o1"):
        raise ValueError("only O1 may update the teacher in the joint stage")
    overrides = runtime["smoke_update_overrides"]
    for stage, value in overrides.items():
        if value is not None:
            _positive_int(value, f"runtime.smoke_update_overrides.{stage}")
    if any(value is not None for value in overrides.values()) and not runtime["synthetic"]:
        raise ValueError("reduced update budgets are allowed only for explicit synthetic smoke")
    return training


def validate_real_teacher_provenance(config: Mapping[str, Any]) -> None:
    """Require and verify every pretrained MedNeXt fact needed by a real run."""
    arm = config.get("arm")
    if arm not in {"c1", "v1", "v1u", "v0", "b1", "e1", "o1", "s1", "mednext_supervised"}:
        return
    teacher = config["model"]["teacher"]
    if teacher.get("architecture") != "mednext_l_k3":
        raise ValueError("real teacher stage requires MedNeXt-L k3 provenance")
    source_repo = teacher.get("source_repo")
    source_commit = teacher.get("source_commit")
    if not isinstance(source_repo, str) or source_commit != UPSTREAM_MEDNEXT_COMMIT:
        raise ValueError("real teacher requires the pinned official MedNeXt source_repo/source_commit")
    _verify_mednext_source_checkout(source_repo, source_commit)
    source = teacher.get("pretrained")
    if not isinstance(source, Mapping):
        raise ValueError("real teacher stage requires pretrained weights provenance")
    if source.get("provenance_status") != "READY":
        raise ValueError(f"real teacher weights are unavailable: {source.get('provenance_status')}")
    path_value = source.get("weights_path")
    digest = source.get("weights_sha256")
    _validate_official_weight_url(source.get("weights_source_url"))
    pretraining_dataset = source.get("pretraining_dataset")
    overlap_acdc, overlap_mnms = source.get("overlap_acdc"), source.get("overlap_mnms")
    if source.get("class_mapping") != _EXPECTED_CLASSES:
        raise ValueError(f"real teacher class_mapping must be {_EXPECTED_CLASSES}; classifier reinitialization is forbidden")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("real teacher stage requires an explicit pretrained weights path")
    if not isinstance(digest, str) or not _HEX_SHA256.fullmatch(digest):
        raise ValueError("real teacher stage requires a lowercase weights SHA-256")
    if not isinstance(pretraining_dataset, str) or not pretraining_dataset.strip():
        raise ValueError("real teacher stage requires the pretrained dataset declaration")
    if type(overlap_acdc) is not bool or type(overlap_mnms) is not bool:
        raise ValueError("real teacher stage requires explicit true/false overlap declarations for ACDC and M&Ms")
    weight_path = Path(path_value).expanduser()
    if not weight_path.is_file():
        raise ValueError(f"pretrained weight file does not exist: {weight_path}")
    hasher = hashlib.sha256()
    with weight_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != digest:
        raise ValueError("pretrained weight file SHA-256 does not match its declaration")


def stages_for_arm(arm: str) -> tuple[str, ...]:
    """Return the ordered stages required by one declared arm."""
    if arm not in _ARM_SETTINGS:
        raise ValueError(f"unknown Q-3D arm: {arm}")
    return tuple(_ARM_SETTINGS[arm]["stages"])


def arm_settings(arm: str) -> dict[str, Any]:
    """Return a defensive copy of the public comparison factors for an arm."""
    if arm not in _ARM_SETTINGS:
        raise ValueError(f"unknown Q-3D arm: {arm}")
    return copy.deepcopy(_ARM_SETTINGS[arm])


def validate_config(config: Mapping[str, Any], *, real_run: bool = False) -> dict[str, Any]:
    """Validate an exact nested Q-3D config and return a defensive copy."""
    raw = copy.deepcopy(dict(config)) if isinstance(config, Mapping) else None
    if raw is None:
        raise TypeError("config must be a mapping")
    _exact_mapping(raw, _ROOT_KEYS, "config")
    if raw["schema"] != SCHEMA or raw["method"] != METHOD:
        raise ValueError("Q-3D schema/method mismatch")
    arm = raw["arm"]
    if arm not in ARMS:
        raise ValueError(f"unknown Q-3D arm: {arm}")
    runtime = _validate_runtime(raw["runtime"])
    _positive_int(raw["seed"], "seed", allow_zero=True)
    dataset = _validate_dataset(raw["dataset"], real_run=real_run)
    protocol_keys = {"variant", "k", "teacher_label_budget", "student_label_budget", "unlabeled_source", "seed", "cache_isolation_key"}
    protocol = dict(_exact_mapping(raw["protocol"], protocol_keys, "protocol"))
    if protocol["variant"] not in {"A", "B"} or type(protocol["k"]) is not int or protocol["k"] not in K_VALUES:
        raise ValueError("protocol must use A/B and K in {2,5,10}")
    if protocol["seed"] != raw["seed"] or protocol["student_label_budget"] != "k":
        raise ValueError("protocol seed and student labeled budget must match the config")
    expected_teacher_budget = "k" if protocol["variant"] == "A" else "all_train"
    if protocol["teacher_label_budget"] != expected_teacher_budget:
        raise ValueError("Protocol A uses K teacher labels; Protocol B uses all TRAIN labels")
    if protocol["unlabeled_source"] != "remaining_train_image_only":
        raise ValueError("U must be the remaining image-only TRAIN patients")
    if protocol["variant"] == "A" and protocol["teacher_label_budget"] != "k":
        raise ValueError("Protocol A teacher must share the K-labeled student cohort")
    expected_namespace = f"q3d/{dataset['name']}/protocol-{protocol['variant'].lower()}/k{protocol['k']}/seed-{raw['seed']}"
    if protocol["cache_isolation_key"] != expected_namespace:
        raise ValueError("protocol cache namespace must isolate dataset/protocol/K/seed")
    model = _validate_model(raw["model"], arm=arm)
    training = _validate_training(raw["training"], arm=arm, runtime=raw["runtime"])
    tokenizer = dict(_exact_mapping(raw["tokenizer"], {
        "enabled", "initialization_volumes", "codebook_size", "dimension", "adapter_hidden", "temperature", "commitment_beta",
        "anatomy_loss_weight", "ema_decay", "dead_code_fraction", "usage_window_updates", "minimum_perplexity",
        "maximum_dead_code_fraction", "minimum_class_codes", "source_feature", "codebook_frozen_after_stage",
    }, "tokenizer"))
    if type(tokenizer["enabled"]) is not bool or tokenizer["enabled"] != _ARM_SETTINGS[arm]["tokenizer"]:
        raise ValueError("tokenizer.enabled must match the selected arm")
    _positive_int(tokenizer["initialization_volumes"], "tokenizer.initialization_volumes")
    if tokenizer["initialization_volumes"] != 200:
        raise ValueError("tokenizer initialization is a separate 200-volume pass, not Q optimizer updates")
    if runtime["smoke_initialization_volumes"] is not None and not tokenizer["enabled"]:
        raise ValueError("synthetic tokenizer initialization override requires an enabled tokenizer")
    if (tokenizer["codebook_size"], tokenizer["dimension"], tokenizer["adapter_hidden"]) != (256, 64, 128):
        raise ValueError("tokenizer dimensions are pinned to 256x64 with a 128-wide adapter")
    for field in ("temperature", "commitment_beta", "anatomy_loss_weight", "ema_decay", "dead_code_fraction",
                  "minimum_perplexity", "maximum_dead_code_fraction"):
        _finite_number(tokenizer[field], f"tokenizer.{field}", minimum=0.0)
    if tokenizer["temperature"] != 0.1 or tokenizer["ema_decay"] != 0.99:
        raise ValueError("tokenizer temperature and EMA decay are pinned by the spec")
    _positive_int(tokenizer["usage_window_updates"], "tokenizer.usage_window_updates")
    _positive_int(tokenizer["minimum_class_codes"], "tokenizer.minimum_class_codes")
    if tokenizer["minimum_class_codes"] != 4 or tokenizer["codebook_frozen_after_stage"] != "Q":
        raise ValueError("tokenizer gate/freeze contract mismatch")
    source_stride = model["teacher"]["feature_stride"]
    expected_source_feature = {
        4: "stride4_decoder_block",
        8: "stride8_decoder_block_after_bottleneck",
        16: "bottleneck_stride16",
    }.get(source_stride, "stride8_decoder_block_after_bottleneck")
    if tokenizer["source_feature"] != expected_source_feature:
        raise ValueError("tokenizer source feature must agree with the teacher stride declaration")
    physiology = dict(_exact_mapping(raw["physiology"], {
        "enabled", "stroke_balance", "certainty_threshold", "tau_stroke_volume", "tau_myocardium",
        "minimum_lv_ed_ml", "ef_range_percent", "cache_refresh_joint_epochs", "use_ed_es_metadata",
    }, "physiology"))
    if type(physiology["enabled"]) is not bool or physiology["enabled"] != _ARM_SETTINGS[arm]["physiology"]:
        raise ValueError("physiology.enabled must match the selected arm")
    for field in ("stroke_balance", "use_ed_es_metadata"):
        if type(physiology[field]) is not bool:
            raise ValueError(f"physiology.{field} must be boolean")
    if physiology["use_ed_es_metadata"]:
        raise ValueError("physiology ED/ES selection must be inferred from predicted LV volume")
    for field in ("certainty_threshold", "tau_stroke_volume", "tau_myocardium", "minimum_lv_ed_ml"):
        _finite_number(physiology[field], f"physiology.{field}", minimum=0.0)
    ef = physiology["ef_range_percent"]
    if not isinstance(ef, list) or ef != [5.0, 90.0]:
        raise ValueError("physiology EF guard is pinned to [5,90] percent")
    if arm == "o1":
        if physiology["cache_refresh_joint_epochs"] != 10:
            raise ValueError("O1 refreshes physiology weights every 10 joint epochs")
    elif physiology["cache_refresh_joint_epochs"] is not None:
        raise ValueError("only O1 refreshes the physiology cache during joint training")
    losses = dict(_exact_mapping(raw["losses"], {
        "labeled_supervised_coefficient", "supervised_divide_by_stream_count", "l_u_auxiliary_reduction",
        "region_ce_dice", "boundary_ce_dice", "balanced_sdf", "sdf_radius_mm", "sdf_near_boundary_mm",
        "sdf_near_far_alpha", "continuous_feature_kd", "codekd", "codekd_ramp_fraction", "commitment_beta",
        "teacher_confidence_weighting", "w_mode", "uniform_w", "codekd_class_balance",
        "background_class_equal_weight", "skip_zero_weight_classes",
    }, "losses"))
    if losses["labeled_supervised_coefficient"] != 1.0 or losses["supervised_divide_by_stream_count"] is not False:
        raise ValueError("labeled supervision stays at coefficient 1 and is not divided by stream count")
    if losses["l_u_auxiliary_reduction"] != "mean":
        raise ValueError("paired L/U KD and commitment terms must be averaged as separate stream terms")
    expected_w_mode = "uniform" if arm == "v1u" else ("physiology" if _ARM_SETTINGS[arm]["physiology"] else "none")
    if losses["w_mode"] != expected_w_mode:
        raise ValueError("losses.w_mode must match the selected arm")
    for field in ("region_ce_dice", "boundary_ce_dice", "balanced_sdf", "sdf_radius_mm",
                  "sdf_near_boundary_mm", "sdf_near_far_alpha", "continuous_feature_kd", "codekd",
                  "codekd_ramp_fraction", "commitment_beta"):
        _finite_number(losses[field], f"losses.{field}", minimum=0.0)
    if (losses["region_ce_dice"], losses["boundary_ce_dice"], losses["balanced_sdf"],
            losses["sdf_radius_mm"], losses["sdf_near_boundary_mm"], losses["sdf_near_far_alpha"],
            losses["codekd_ramp_fraction"], losses["commitment_beta"]) != (1.0, 1.0, 1.0, 5.0, 2.5, 0.5, 0.10, 0.25):
        raise ValueError("teacher/student loss constants differ from the locked Q-3D spec")
    expected_feature_kd = 1.0 if arm == "c1" else 0.0
    expected_codekd = 1.0 if arm in {"v1", "v1u", "b1", "e1", "o1", "s1"} else 0.0
    if losses["continuous_feature_kd"] != expected_feature_kd or losses["codekd"] != expected_codekd:
        raise ValueError("teacher signal weights must match the selected ablation")
    if losses["uniform_w"] != (1.0 if arm == "v1u" else None):
        raise ValueError("V1-u uses unit weights and other arms do not")
    for field in ("teacher_confidence_weighting", "codekd_class_balance", "background_class_equal_weight", "skip_zero_weight_classes"):
        if type(losses[field]) is not bool:
            raise ValueError(f"losses.{field} must be boolean")
    if not all(losses[field] for field in ("codekd_class_balance", "background_class_equal_weight", "skip_zero_weight_classes")):
        raise ValueError("codeKD must balance every class and skip only zero-weight classes")
    if losses["codekd"] > 0 and losses["codekd_class_balance"] is not True:
        raise ValueError("codeKD class balancing is mandatory")
    augmentation = dict(_exact_mapping(raw["augmentation"], {
        "flip_y_probability", "flip_x_probability", "rotate90_probability", "rotate90_requires_equal_spacing",
        "spatial_transform_shared", "recompute_sdf_after_spatial_transform", "intensity_probability_labeled",
        "intensity_probability_unlabeled_student", "gamma_range", "voxel_noise_std",
        "unlabeled_teacher_intensity_augmentation", "preserve_native_geometry",
    }, "augmentation"))
    for field in ("flip_y_probability", "flip_x_probability", "rotate90_probability", "intensity_probability_labeled",
                  "intensity_probability_unlabeled_student"):
        _finite_number(augmentation[field], f"augmentation.{field}", minimum=0.0, maximum=1.0)
    if augmentation["gamma_range"] != [0.7, 1.5] or augmentation["voxel_noise_std"] != 0.03:
        raise ValueError("intensity augmentation is pinned to gamma [0.7,1.5] and noise std 0.03")
    if augmentation["intensity_probability_unlabeled_student"] != 1.0:
        raise ValueError("the U strong student view must use the declared intensity perturbation on every update")
    for field in ("rotate90_requires_equal_spacing", "spatial_transform_shared", "recompute_sdf_after_spatial_transform",
                  "unlabeled_teacher_intensity_augmentation", "preserve_native_geometry"):
        if type(augmentation[field]) is not bool:
            raise ValueError(f"augmentation.{field} must be boolean")
    if (not augmentation["rotate90_requires_equal_spacing"] or not augmentation["spatial_transform_shared"]
            or not augmentation["recompute_sdf_after_spatial_transform"] or augmentation["unlabeled_teacher_intensity_augmentation"]
            or not augmentation["preserve_native_geometry"]):
        raise ValueError("augmentation must preserve shared native geometry and keep the U teacher view unmodified")
    checkpoint = dict(_exact_mapping(raw["checkpoint"], {
        "namespace", "isolate_dataset", "isolate_protocol", "isolate_k", "isolate_seed", "stage_outputs", "resume_scope",
    }, "checkpoint"))
    if checkpoint["namespace"] != expected_namespace or checkpoint["namespace"] != protocol["cache_isolation_key"]:
        raise ValueError("checkpoint namespace must match the protocol isolation key")
    for field in ("isolate_dataset", "isolate_protocol", "isolate_k", "isolate_seed"):
        if type(checkpoint[field]) is not bool or not checkpoint[field]:
            raise ValueError(f"checkpoint.{field} must be true")
    teacher_stride = model["teacher"]["feature_stride"]
    residual2_enabled = model["student"]["residual2"]["enabled"]
    setting_namespace = (
        "v1_stride4" if arm == "v1" and teacher_stride == 4 else
        "v1_residual2" if arm == "v1" and residual2_enabled else arm
    )
    expected_outputs = {name: f"{expected_namespace}/{setting_namespace}/{name.lower()}" for name in ("T", "Q", "P", "J")}
    if checkpoint["stage_outputs"] != expected_outputs or checkpoint["resume_scope"] != "stage_checkpoint_only":
        raise ValueError("checkpoint stage paths/resume scope must preserve cache isolation")
    if not isinstance(raw["baselines"], Mapping) or set(raw["baselines"]) != set(BASELINE_ARMS):
        raise ValueError("baselines must enumerate Mean Teacher, A-3D teacher_free, and MedNeXt supervised")
    for name, expected in _BASELINE_CATALOG.items():
        if dict(raw["baselines"][name]) != expected:
            raise ValueError(f"baseline launch mapping for {name} differs from the declared implementation")
    if real_run:
        validate_real_teacher_provenance(raw)
        # Real executions need pinned source manifests; development is used
        # read-only for checkpoint selection and training uses only TRAIN.
        for part in ("train", "development"):
            record = dataset["manifests"][part]
            path = Path(record["path"]).expanduser()
            if not path.is_file():
                raise ValueError(f"real run manifest does not exist: {path}")
            with path.open("rb") as handle:
                manifest_sha = hashlib.file_digest(handle, "sha256").hexdigest()
            if manifest_sha != record["sha256"]:
                raise ValueError(f"real run manifest SHA-256 mismatch: {path}")
        teacher_train = dataset["manifests"]["teacher_train"]
        if protocol["variant"] == "B" and teacher_train["status"] != "READY":
            raise ValueError("Protocol B requires an explicit all-TRAIN teacher manifest")
        if teacher_train["status"] == "READY":
            teacher_path = Path(teacher_train["path"]).expanduser()
            if not teacher_path.is_file():
                raise ValueError(f"teacher TRAIN manifest does not exist: {teacher_path}")
            with teacher_path.open("rb") as handle:
                teacher_sha = hashlib.file_digest(handle, "sha256").hexdigest()
            if teacher_sha != teacher_train["sha256"]:
                raise ValueError(f"teacher TRAIN manifest SHA-256 mismatch: {teacher_path}")
    validated = copy.deepcopy(raw)
    validated["dataset"] = dataset
    validated["protocol"] = protocol
    validated["model"] = model
    validated["training"] = training
    validated["tokenizer"] = tokenizer
    validated["physiology"] = physiology
    validated["losses"] = losses
    validated["augmentation"] = augmentation
    validated["checkpoint"] = checkpoint
    validated["runtime"] = runtime
    if runtime["synthetic"]:
        for stage, count in runtime["smoke_update_overrides"].items():
            if count is not None:
                validated["training"]["stage_updates"][stage] = count
        optimizer_updates, stream_microbatches = _stage_accounting(validated["training"]["stage_updates"], arm)
        validated["training"]["stage_optimizer_updates"] = optimizer_updates
        validated["training"]["stage_stream_microbatches"] = stream_microbatches
    return validated


def resolve_stage_updates(config: Mapping[str, Any], *, smoke: bool = False) -> dict[str, int]:
    """Resolve full design budgets or explicit synthetic-only smoke overrides."""
    validated = validate_config(config)
    if smoke and not validated["runtime"]["synthetic"]:
        raise ValueError("smoke budget reduction requires runtime.synthetic=true")
    resolved = dict(validated["training"]["stage_updates"])
    overrides = validated["runtime"]["smoke_update_overrides"]
    if smoke:
        for stage, count in overrides.items():
            if count is not None:
                resolved[stage] = count
    elif any(count is not None for count in overrides.values()):
        raise ValueError("smoke update overrides cannot affect a real run")
    return resolved


def resolve_stage_accounting(config: Mapping[str, Any], *, smoke: bool = False) -> dict[str, dict[str, int | None]]:
    """Return update, optimizer-step and stream-microbatch counts explicitly."""
    updates = resolve_stage_updates(config, smoke=smoke)
    optimizer_updates, stream_microbatches = _stage_accounting(updates, str(config["arm"]))
    return {
        "stage_updates": updates,
        "optimizer_updates": optimizer_updates,
        "stream_microbatches": stream_microbatches,
    }


def validate_budget_update_equality(configs: list[Mapping[str, Any]], *, smoke: bool = False) -> dict[str, int]:
    """Ensure all supplied comparable arms keep the same student update budget."""
    if not configs:
        raise ValueError("at least one config is required")
    validated = [validate_config(config) for config in configs]
    identity = {(c["dataset"]["name"], c["protocol"]["variant"], c["protocol"]["k"], c["seed"]) for c in validated}
    if len(identity) != 1:
        raise ValueError("budget equality is checked within one dataset/protocol/K/seed setting")
    updates = [resolve_stage_updates(c, smoke=smoke) for c in validated]
    joint = {item["J"] for item in updates}
    if len(joint) != 1:
        raise ValueError("compared arms must use the same joint student update budget")
    teacher_updates = {updates[index]["T"] for index, c in enumerate(validated)
                       if "T" in _ARM_SETTINGS[c["arm"]]["stages"]}
    if len(teacher_updates) > 1:
        raise ValueError("teacher-updating arms must use the same T update budget")
    tokenizer_updates = {updates[index]["Q"] for index, c in enumerate(validated)
                         if "Q" in _ARM_SETTINGS[c["arm"]]["stages"]}
    if len(tokenizer_updates) > 1:
        raise ValueError("tokenizer arms must use the same Q update budget")
    return updates[0]


def load_config(path: str | Path, *, real_run: bool = False) -> dict[str, Any]:
    """Read and validate one JSON recipe."""
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read Q-3D config {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Q-3D config JSON must contain an object")
    normalized = copy.deepcopy(dict(value))

    def resolve_local(value: Any) -> Any:
        if not isinstance(value, str) or not value.strip():
            return value
        local = Path(value).expanduser()
        return str((local if local.is_absolute() else config_path.parent / local).resolve())

    dataset = normalized.get("dataset")
    if isinstance(dataset, dict):
        manifests = dataset.get("manifests")
        if isinstance(manifests, dict):
            for record in manifests.values():
                if isinstance(record, dict) and record.get("path") is not None:
                    record["path"] = resolve_local(record["path"])
        for record in dataset.get("cross_evaluation", []):
            if isinstance(record, dict) and isinstance(record.get("manifest"), dict):
                manifest = record["manifest"]
                if manifest.get("path") is not None:
                    manifest["path"] = resolve_local(manifest["path"])

    teacher = normalized.get("model", {}).get("teacher")
    if isinstance(teacher, dict):
        if isinstance(teacher.get("source_repo"), str):
            teacher["source_repo"] = resolve_local(teacher["source_repo"])
        pretrained = teacher.get("pretrained")
        if isinstance(pretrained, dict) and pretrained.get("weights_path") is not None:
            pretrained["weights_path"] = resolve_local(pretrained["weights_path"])
    return validate_config(normalized, real_run=real_run)


def main(argv: list[str] | None = None) -> None:
    """Print canonical config or validate a JSON file for campaign tooling."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--dataset", choices=("acdc", "mnms"), default="acdc")
    parser.add_argument("--protocol", choices=("A", "B"), default="A")
    parser.add_argument("--k", type=int, choices=K_VALUES, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--real-run", action="store_true")
    args = parser.parse_args(argv)
    if args.config:
        config = load_config(args.config, real_run=args.real_run)
    elif args.arm:
        config = canonical_config(args.arm, dataset=args.dataset, protocol=args.protocol, k=args.k, seed=args.seed)
    else:
        parser.error("provide --config or --arm")
    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
