"""Locked configuration contract for Candidate FUSE."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "self_audit.candidate_fuse.v1"
SEMI_SCHEMA = "self_audit.candidate_fuse.v2"
TRANSFER_SCHEMA = "self_audit.candidate_fuse.v3"
METHOD = "candidate_fuse"
_ROOT_KEYS = {"schema", "method", "seed", "data", "model", "training", "evaluation", "runtime"}
_DATA_KEYS = {"train_manifest", "train_manifest_sha256", "val_manifest", "val_manifest_sha256", "test_manifest", "test_manifest_sha256"}
_TRAINING_DEFAULTS = {
    "teacher_epochs": 50, "joint_epochs": 100, "teacher_batch_size": 1,
    "teacher_lr_encoder": 1e-4, "teacher_lr_head": 1e-3,
    "student_lr": 3e-4, "weight_decay": 1e-4,
    "warmup_fraction": 0.05, "gradient_clip_norm": 1.0,
    "unlabeled_ramp_fraction": 0.1, "mimic_weight": 0.1,
    "max_teacher_dropout": 0.5,
}
_TRANSFER_SOURCE_SHA256 = "bf2de8287a3fb6a5df7c490b0e081d39e58911a99cfa081243015a2c6ca894ec"
_TRANSFER_SOURCE_METADATA_SHA256 = "5c2791fb0a45f353b98885765d9b546df3efba95c88150f647cf0a84b27d4752"
_TRANSFER_SOURCE_IDENTITY = {
    "config_sha256": "73e7fdd1bf5376a550dc8c7bf24d850f186df3ef5efe297c0fcc06322bd253d7",
    "train_manifest_sha256": "3b30c0006218b2aa348730b1667ca2a78fe47742af509871d9a860bb9a8d5b9b",
    "val_manifest_sha256": "e3421779441afcab8f8feadbd293bedad6c00877730eb3c7151fa6a15667e88b",
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


def default_transfer_config(*, synthetic: bool = False) -> dict[str, Any]:
    """Return the Track B v3 template with its immutable ACDC source pins."""
    config = default_config(synthetic=synthetic)
    config["schema"] = TRANSFER_SCHEMA
    config["experiment"] = {
        "track": "B", "dataset": "MNMS",
        "source_teacher": {
            "path": "/root/mount_data/runs/fuse-20261113/teacher.pt",
            "sha256": _TRANSFER_SOURCE_SHA256,
            "metadata_sha256": _TRANSFER_SOURCE_METADATA_SHA256,
            "identity": copy.deepcopy(_TRANSFER_SOURCE_IDENTITY),
        },
    }
    config["training"].update(
        teacher_epochs=1 if synthetic else 250,
        joint_epochs=1 if synthetic else 100,
        teacher_batch_size=2,
    )
    return config


def normalize_config(mapping: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        raise TypeError("FUSE config must be a mapping")
    result = copy.deepcopy(dict(mapping))
    defaults = default_config(synthetic=bool(result.get("runtime", {}).get("synthetic", False)))
    semi = result.get("schema") == SEMI_SCHEMA
    transfer = result.get("schema") == TRANSFER_SCHEMA
    if transfer:
        transfer_defaults = default_transfer_config(synthetic=bool(result.get("runtime", {}).get("synthetic", False)))
        if set(result) != _ROOT_KEYS | {"experiment"}:
            raise ValueError("FUSE v3 requires exactly the locked Track B experiment object")
        result["training"] = copy.deepcopy(result.get("training", {}))
        for name in ("teacher_epochs", "joint_epochs", "teacher_batch_size"):
            result["training"].setdefault(name, transfer_defaults["training"][name])
    if semi:
        if set(result) != _ROOT_KEYS | {"experiment"} or result.get("experiment") != {"track": "A", "dataset": "ACDC"}:
            raise ValueError("FUSE v2 requires exactly experiment={track:A,dataset:ACDC}")
        result["training"] = copy.deepcopy(result.get("training", {}))
        result["training"].setdefault("teacher_epochs", 1 if result.get("runtime", {}).get("synthetic") else 500)
    for key in ("training", "evaluation", "runtime"):
        if isinstance(result.get(key), dict):
            for name, value in defaults[key].items():
                result[key].setdefault(name, copy.deepcopy(value))
    return validate_config(result)


def validate_config(config: Mapping[str, Any], *, real_run: bool = False) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    semi = value.get("schema") == SEMI_SCHEMA
    transfer = value.get("schema") == TRANSFER_SCHEMA
    expected_root = _ROOT_KEYS | ({"experiment"} if (semi or transfer) else set())
    if set(value) != expected_root:
        raise ValueError(f"FUSE config keys must be exactly {sorted(expected_root)}")
    if value.get("schema") not in {SCHEMA, SEMI_SCHEMA, TRANSFER_SCHEMA} or value.get("method") != METHOD:
        raise ValueError("unsupported FUSE schema or method")
    if semi and value.get("experiment") != {"track": "A", "dataset": "ACDC"}:
        raise ValueError("FUSE v2 experiment must be exactly track A, dataset ACDC")
    if transfer:
        experiment = value.get("experiment")
        if not isinstance(experiment, Mapping) or set(experiment) != {"track", "dataset", "source_teacher"} or experiment.get("track") != "B" or experiment.get("dataset") != "MNMS":
            raise ValueError("FUSE v3 experiment must be Track B on MNMS with source_teacher")
        source = experiment.get("source_teacher")
        if not isinstance(source, Mapping) or set(source) != {"path", "sha256", "metadata_sha256", "identity"}:
            raise ValueError("FUSE v3 source_teacher has invalid fields")
        synthetic = value.get("runtime", {}).get("synthetic") is True
        for field in ("sha256", "metadata_sha256"):
            digest = source.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"FUSE v3 source teacher {field} must be lowercase SHA256")
        identity = source.get("identity")
        identity_fields = {"config_sha256", "train_manifest_sha256", "val_manifest_sha256"}
        if not isinstance(identity, Mapping) or set(identity) != identity_fields or any(
            not isinstance(identity.get(field), str) or len(identity[field]) != 64
            or any(c not in "0123456789abcdef" for c in identity[field])
            for field in identity_fields
        ):
            raise ValueError("FUSE v3 source teacher identity must pin config/train/val SHA256")
        if not synthetic and (source.get("sha256") != _TRANSFER_SOURCE_SHA256
                              or source.get("metadata_sha256") != _TRANSFER_SOURCE_METADATA_SHA256
                              or identity != _TRANSFER_SOURCE_IDENTITY):
            raise ValueError("FUSE v3 source teacher pins differ from the selected ACDC80 source")
        source_path = source.get("path")
        if source_path is not None and (not isinstance(source_path, str) or not source_path.strip()):
            raise ValueError("FUSE v3 source teacher path must be a nonempty path or null template")
        if real_run and (source_path is None or not Path(source_path).is_file()):
            raise ValueError("real FUSE Track B requires the existing pinned selected source teacher")
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
    expected_epochs = ({"teacher_epochs": 1, "joint_epochs": 1} if synthetic else {
        "teacher_epochs": 250 if transfer else 500 if semi else 50, "joint_epochs": 100,
    })
    for name, expected in expected_epochs.items():
        actual = training[name]
        if type(actual) is not int or actual != expected:
            description = "synthetic FUSE defaults use one teacher and joint epoch" if synthetic else "real FUSE defaults use 50 teacher and 100 joint epochs"
            raise ValueError(description)
    batch_size = training["teacher_batch_size"]
    if type(batch_size) is not int or batch_size not in (1, 2) or (transfer and batch_size != 2):
        raise ValueError("FUSE teacher_batch_size must be integer 1 or 2")
    for name, expected in defaults.items():
        if name in epoch_fields or name == "teacher_batch_size":
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
