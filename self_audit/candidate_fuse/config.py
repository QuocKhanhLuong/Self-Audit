"""Locked configuration contract for Candidate FUSE."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "self_audit.candidate_fuse.v1"
METHOD = "candidate_fuse"
_ROOT_KEYS = {"schema", "method", "seed", "data", "model", "training", "evaluation", "runtime"}
_DATA_KEYS = {"train_manifest", "train_manifest_sha256", "val_manifest", "val_manifest_sha256", "test_manifest", "test_manifest_sha256"}
_TRAINING_DEFAULTS = {
    "teacher_epochs": 50, "joint_epochs": 100,
    "teacher_lr_encoder": 1e-4, "teacher_lr_head": 1e-3,
    "student_lr": 3e-4, "weight_decay": 1e-4,
    "warmup_fraction": 0.05, "gradient_clip_norm": 1.0,
    "unlabeled_ramp_fraction": 0.1, "mimic_weight": 0.1,
    "max_teacher_dropout": 0.5,
}


def _teacher_recipe() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "configs/candidate_q3d/v1.json"
    return copy.deepcopy(json.loads(path.read_text())["model"]["teacher"])


def default_config(*, synthetic: bool = False) -> dict[str, Any]:
    channels = [4, 8, 12, 16, 32] if synthetic else [32, 64, 128, 256, 512]
    config = {
        "schema": SCHEMA, "method": METHOD, "seed": 42,
        "data": {key: None for key in _DATA_KEYS},
        "model": {"teacher": _teacher_recipe(), "student": {"channels": channels}},
        "training": copy.deepcopy(_TRAINING_DEFAULTS),
        "evaluation": {"every_epochs": 1, "deploy_margin": 0.01, "test_once": True},
        "runtime": {"synthetic": bool(synthetic), "amp_dtype": "bfloat16", "deterministic": True, "num_threads": 1 if synthetic else 1},
    }
    if synthetic:
        config["training"].update(teacher_epochs=1, joint_epochs=1)
        config["runtime"]["amp_dtype"] = "float32"
    return config


def normalize_config(mapping: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        raise TypeError("FUSE config must be a mapping")
    result = copy.deepcopy(dict(mapping))
    defaults = default_config(synthetic=bool(result.get("runtime", {}).get("synthetic", False)))
    for key in ("training", "evaluation", "runtime"):
        if isinstance(result.get(key), dict):
            for name, value in defaults[key].items():
                result[key].setdefault(name, copy.deepcopy(value))
    return validate_config(result)


def validate_config(config: Mapping[str, Any], *, real_run: bool = False) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    if set(value) != _ROOT_KEYS:
        raise ValueError(f"FUSE config keys must be exactly {sorted(_ROOT_KEYS)}")
    if value.get("schema") != SCHEMA or value.get("method") != METHOD:
        raise ValueError("unsupported FUSE schema or method")
    if type(value.get("seed")) is not int or value["seed"] != 42:
        raise ValueError("FUSE seed is locked to 42")
    data = value.get("data")
    if not isinstance(data, dict) or set(data) != _DATA_KEYS:
        raise ValueError("FUSE data must declare the six train/val/test manifest fields")
    for name in ("train", "val", "test"):
        path, sha = data[f"{name}_manifest"], data[f"{name}_manifest_sha256"]
        if (path is None) != (sha is None):
            raise ValueError(f"{name} manifest path and SHA must both be set or null")
        if sha is not None and (not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError(f"invalid {name} manifest SHA-256")
        if real_run and (path is None or not Path(path).is_file()):
            raise ValueError(f"real FUSE execution requires an existing pinned {name} manifest")
        if real_run:
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual != sha:
                raise ValueError(f"real FUSE {name} manifest SHA-256 mismatch")
    model = value.get("model")
    if not isinstance(model, dict) or set(model) != {"teacher", "student"}:
        raise ValueError("FUSE model requires teacher and student")
    if model["teacher"] != _teacher_recipe():
        raise ValueError("FUSE teacher recipe must match the pinned Q3D recipe by value")
    synthetic = value.get("runtime", {}).get("synthetic") is True
    expected_channels = [4, 8, 12, 16, 32] if synthetic else [32, 64, 128, 256, 512]
    if model["student"] != {"channels": expected_channels}:
        raise ValueError(f"FUSE student channels are locked to {expected_channels}")
    training = value.get("training")
    if not isinstance(training, dict) or set(training) != set(_TRAINING_DEFAULTS):
        raise ValueError("FUSE training fields do not match the locked interface")
    defaults = default_config(synthetic=synthetic)["training"]
    epoch_fields = {"teacher_epochs", "joint_epochs"}
    expected_epochs = {"teacher_epochs": 1, "joint_epochs": 1} if synthetic else {
        "teacher_epochs": 50, "joint_epochs": 100,
    }
    for name, expected in expected_epochs.items():
        actual = training[name]
        if type(actual) is not int or actual != expected:
            description = "synthetic FUSE defaults use one teacher and joint epoch" if synthetic else "real FUSE defaults use 50 teacher and 100 joint epochs"
            raise ValueError(description)
    for name, expected in defaults.items():
        if name in epoch_fields:
            continue
        actual = training[name]
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"FUSE training scalar {name} must be a finite number")
        if not math.isfinite(actual):
            raise ValueError(f"FUSE training scalar {name} must be a finite number")
        if actual != expected:
            raise ValueError(f"FUSE training scalar {name} is locked to {expected}")
    for section, expected in (("evaluation", {"every_epochs": 1, "deploy_margin": 0.01, "test_once": True}),
                              ("runtime", {"synthetic", "amp_dtype", "deterministic", "num_threads"})):
        current = value.get(section)
        if not isinstance(current, dict) or (set(current) != set(expected) if isinstance(expected, set) else current != expected):
            raise ValueError(f"invalid FUSE {section} contract")
    if value["runtime"]["amp_dtype"] not in {"bfloat16", "float32"} or type(value["runtime"]["deterministic"]) is not bool or type(value["runtime"]["num_threads"]) is not int or value["runtime"]["num_threads"] < 1:
        raise ValueError("invalid FUSE runtime settings")
    return value


def load_config(path: str | Path, *, real_run: bool = False) -> dict[str, Any]:
    return normalize_config(json.loads(Path(path).read_text())) if not real_run else validate_config(json.loads(Path(path).read_text()), real_run=True)


def config_sha256(mapping: Mapping[str, Any]) -> str:
    normalized = normalize_config(mapping)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()
