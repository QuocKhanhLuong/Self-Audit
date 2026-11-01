"""Fail-closed runtime for the frozen gradient-bridge diagnostic.

This module coordinates pinned model loading, TRAIN-only projector fitting,
shared A/B protocol locks, image-only streaming prediction, and authorized
scoring.  It owns no model architecture or metric definitions: those are
provided by ``diagnostic_models`` and ``gradient_bridge_metrics``.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA = "self_audit.gradient_bridge.runtime.v1"
LOCK_SCHEMA = "self_audit.gradient_bridge.lock.v1"
PREDICTION_SCHEMA = "self_audit.gradient_bridge.predictions.v1"
SCORE_SCHEMA = "self_audit.gradient_bridge.score.v1"
REPORT_SCHEMA = "self_audit.gradient_bridge.report.v1"
ROOT = Path(__file__).resolve().parents[1]
ARMS = ("D0", "D1", "D2", "D3", "D4", "D5")
CLASS_NAMES = ("BG", "RV", "MYO", "LV")
IGNORE_INDEX = 255
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MODEL_FIELDS = {"path", "manifest_sha256", "payload_sha256", "runtime_root", "runtime_source_sha256"}
_MANIFEST_FIELDS = {"path", "sha256"}
_HISTORICAL_PINS = {
    "teacher_manifest_sha256": "ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa",
    "teacher_payload_sha256": "0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6",
    "student_manifest_sha256": "197dd4367877c51bc21b91d172f25442549ef9a3d0e422d28118bb07308067de",
    "student_payload_sha256": "32a35c1fc3b83d57d083de4ed17ab80bda6ba38a8ebc50133e5aede17ab00380",
    "train_manifest_sha256": "e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4",
    "val_manifest_sha256": "cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50",
}
_HISTORICAL_KD = {
    "temperature": 1.0,
    "coefficient": 1.0,
    "confidence": 0.7,
    "class_weights": None,
    "denominator": "valid_pixel_count",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _read_object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    _canonical(value)
    return value


def _write_once(path: str | Path, value: Any) -> Path:
    """Atomically publish JSON without replacing evidence from an earlier run."""
    from .diagnostic_protocol import write_once

    target = Path(path)
    write_once(target, value)
    return target


def _resolve(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Read the strict B-runtime schema and return it with its source path."""
    config_path = Path(path).resolve()
    config = _read_object(config_path)
    validate_config(config)
    return config, config_path


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "profile", "synthetic_fixture", "device", "threads", "batch_size",
        "normalization", "teacher", "student", "train_manifest", "val_manifest",
        "parity", "fit_p", "kd", "directions", "gate",
    }
    if not isinstance(config, Mapping) or set(config) != required:
        raise ValueError(f"config keys must be exactly {sorted(required)}")
    if config["schema"] != SCHEMA:
        raise ValueError(f"unsupported gradient bridge config schema {config.get('schema')!r}")
    if config["profile"] not in {"historical", "exploratory", "fixture"}:
        raise ValueError("profile must be historical, exploratory, or fixture")
    if type(config["synthetic_fixture"]) is not bool:
        raise ValueError("synthetic_fixture must be explicitly boolean")
    if (config["profile"] == "fixture") != config["synthetic_fixture"]:
        raise ValueError("fixture profile and synthetic_fixture must agree")
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    for key in ("threads", "batch_size"):
        value = config[key]
        if type(value) is not int or value < (2 if key == "batch_size" else 1):
            raise ValueError(f"{key} must be an integer >= {2 if key == 'batch_size' else 1}")
    if config["normalization"] != "prepared":
        raise ValueError("gradient bridge requires the matched prepared [0,1] native preprocessing")

    for role in ("teacher", "student"):
        spec = config[role]
        if not isinstance(spec, Mapping) or set(spec) != _MODEL_FIELDS:
            raise ValueError(f"{role} must contain exactly {sorted(_MODEL_FIELDS)}")
        for key in ("manifest_sha256", "payload_sha256", "runtime_source_sha256"):
            _require_sha(spec[key], f"{role}.{key}")
        for key in ("path", "runtime_root"):
            if not isinstance(spec[key], str) or not spec[key].strip():
                raise ValueError(f"{role}.{key} must be a nonempty path")

    for role in ("train_manifest", "val_manifest"):
        spec = config[role]
        if not isinstance(spec, Mapping) or set(spec) != _MANIFEST_FIELDS:
            raise ValueError(f"{role} must contain exactly path and sha256")
        if not isinstance(spec["path"], str) or not spec["path"].strip():
            raise ValueError(f"{role}.path must be a nonempty path")
        _require_sha(spec["sha256"], f"{role}.sha256")

    parity = config["parity"]
    if not isinstance(parity, Mapping) or set(parity) != {"logit_atol", "logit_rtol"}:
        raise ValueError("parity must contain logit_atol and logit_rtol")
    for key, value in parity.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"parity.{key} must be finite and nonnegative")

    fit = config["fit_p"]
    fit_fields = {"seed", "max_source_ids", "max_train_patients", "max_pixels", "ridge", "fit_loss", "alignment_mode", "output_file"}
    if not isinstance(fit, Mapping) or set(fit) != fit_fields:
        raise ValueError(f"fit_p must contain exactly {sorted(fit_fields)}")
    for key in ("seed", "max_source_ids", "max_train_patients", "max_pixels"):
        value = fit[key]
        minimum = 0 if key == "seed" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"fit_p.{key} must be an integer >= {minimum}")
    if fit["max_train_patients"] < 2 or fit["max_source_ids"] < fit["max_train_patients"]:
        raise ValueError("fit-P budget must cover at least one source from each of two TRAIN patients")
    if isinstance(fit["ridge"], bool) or not isinstance(fit["ridge"], (int, float)) or not math.isfinite(float(fit["ridge"])) or fit["ridge"] <= 0:
        raise ValueError("fit_p.ridge must be positive finite")
    if fit["fit_loss"] != "patient_balanced_pre_refine0_feature_mse":
        raise ValueError("fit-P objective is locked to patient-balanced pre-refine0 feature MSE")
    if fit["alignment_mode"] != "bilinear_align_corners_false_then_1x1_affine":
        raise ValueError("fit-P spatial alignment is locked to bilinear align_corners=False then 1x1 affine")
    name = fit["output_file"]
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("fit_p.output_file must be a single safe filename")

    kd = config["kd"]
    kd_fields = {"temperature", "coefficient", "confidence", "class_weights", "denominator"}
    if not isinstance(kd, Mapping) or set(kd) != kd_fields:
        raise ValueError(f"kd must contain exactly {sorted(kd_fields)}")
    for key in ("temperature", "coefficient"):
        value = kd[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"kd.{key} must be positive finite")
    confidence = kd["confidence"]
    if confidence is None:
        raise ValueError("kd.confidence must be explicitly selected; use 0 for no threshold")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0 <= confidence <= 1:
        raise ValueError("kd.confidence must be an explicit finite threshold in [0,1]")
    class_weights = kd["class_weights"]
    if class_weights is not None:
        if (not isinstance(class_weights, list) or len(class_weights) != 4
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or x < 0 for x in class_weights)
                or not any(float(x) > 0 for x in class_weights)):
            raise ValueError("kd.class_weights must be null or four finite nonnegative class weights with positive mass")
    if kd["denominator"] not in {"valid_pixel_count", "valid_weight_sum"}:
        raise ValueError("kd.denominator must be valid_pixel_count or valid_weight_sum")
    if class_weights is None and kd["denominator"] != "valid_pixel_count":
        raise ValueError("unweighted historical KD denominator must be valid_pixel_count")
    if config["profile"] == "historical":
        for key, expected in _HISTORICAL_KD.items():
            actual = kd[key]
            if key in {"temperature", "coefficient", "confidence"}:
                actual = float(actual)
            if actual != expected:
                raise ValueError(f"historical B8/wide plain-KD protocol requires kd.{key}={expected!r}")

    directions = config["directions"]
    direction_fields = {"rho_primary", "rho_sensitivity", "norm_epsilon", "shuffle_seed"}
    if not isinstance(directions, Mapping) or set(directions) != direction_fields:
        raise ValueError(f"directions must contain exactly {sorted(direction_fields)}")
    for key in ("rho_primary", "norm_epsilon"):
        value = directions[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"directions.{key} must be finite and nonnegative")
    if directions["norm_epsilon"] <= 0:
        raise ValueError("directions.norm_epsilon must be positive")
    if type(directions["shuffle_seed"]) is not int or directions["shuffle_seed"] < 0:
        raise ValueError("directions.shuffle_seed must be a nonnegative integer")
    sensitivity = directions["rho_sensitivity"]
    if (not isinstance(sensitivity, list) or not sensitivity
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or x < 0 for x in sensitivity)
            or len({float(x) for x in sensitivity}) != len(sensitivity)
            or float(directions["rho_primary"]) not in {float(x) for x in sensitivity}):
        raise ValueError("rho_sensitivity must be a unique nonempty list containing rho_primary")

    # Use the same validator as the eventual scorer; missing thresholds fail
    # before a B lock can be written.
    from .gradient_bridge_metrics import _validate_gate_config

    _validate_gate_config(config["gate"])
    return dict(config)


def _config_paths(config: Mapping[str, Any], origin: Path) -> dict[str, Any]:
    base = origin.parent
    paths: dict[str, Any] = {
        "train_manifest": _resolve(base, config["train_manifest"]["path"], "train_manifest.path"),
        "val_manifest": _resolve(base, config["val_manifest"]["path"], "val_manifest.path"),
        "teacher_path": _resolve(base, config["teacher"]["path"], "teacher.path"),
        "teacher_runtime": _resolve(base, config["teacher"]["runtime_root"], "teacher.runtime_root"),
        "student_path": _resolve(base, config["student"]["path"], "student.path"),
        "student_runtime": _resolve(base, config["student"]["runtime_root"], "student.runtime_root"),
    }
    return paths


def _runtime_source_files(root: Path) -> list[Path]:
    package = root / "self_audit"
    train = root / "train.py"
    if not package.is_dir() or not train.is_file():
        raise FileNotFoundError(f"incomplete pinned model runtime: {root}")
    files = sorted(package.glob("*.py")) + [train]
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"incomplete pinned model source closure: {root}")
    return files


def runtime_source_digest(root: str | Path) -> str:
    root = Path(root).resolve()
    data = "".join(f"{path.name}:{sha256_file(path)}\n" for path in _runtime_source_files(root))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _artifact_static(spec: Mapping[str, Any], role: str, origin: Path) -> tuple[Path, dict[str, Any], Path]:
    base = origin.parent
    folder = _resolve(base, spec["path"], f"{role}.path")
    manifest_path = folder / "MANIFEST.json"
    payload_path = folder / "weights.pt"
    expected_manifest = _require_sha(spec["manifest_sha256"], f"{role}.manifest_sha256")
    expected_payload = _require_sha(spec["payload_sha256"], f"{role}.payload_sha256")
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest:
        raise ValueError(f"{role} model manifest is missing or has a SHA-256 mismatch")
    manifest = _read_object(manifest_path)
    if manifest.get("payload_sha256") != expected_payload:
        raise ValueError(f"{role} manifest does not bind the configured payload bytes")
    if not payload_path.is_file() or sha256_file(payload_path) != expected_payload:
        raise ValueError(f"{role} model payload is missing or has a SHA-256 mismatch")
    runtime_root = _resolve(base, spec["runtime_root"], f"{role}.runtime_root")
    observed_source = runtime_source_digest(runtime_root)
    if observed_source != spec["runtime_source_sha256"]:
        raise ValueError(f"{role} runtime source closure differs from the locked source hash")
    return folder, manifest, payload_path


def _load_student(spec: Mapping[str, Any], origin: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    folder, manifest, payload_path = _artifact_static(spec, "student", origin)
    runtime_root = _resolve(origin.parent, spec["runtime_root"], "student.runtime_root")
    if runtime_root == ROOT:
        from .compact_artifacts import load_compact_export
        from .engine import source_digest

        loader = load_compact_export
        observed_source = source_digest()
    else:
        # Historical compact exports have their own source closure and model
        # class. Import and load them under an isolated alias; never rewrite
        # their manifest identity or deserialize their state into a guessed
        # current architecture.
        from .compact_teacher import import_historical_package

        alias = "_gradient_bridge_compact_" + spec["runtime_source_sha256"][:16]
        import_historical_package(runtime_root, pkg_alias=alias)
        artifacts = importlib.import_module(f"{alias}.compact_artifacts")
        engine = importlib.import_module(f"{alias}.engine")
        loader = getattr(artifacts, "load_compact_export", None)
        source_fn = getattr(engine, "source_digest", None)
        if loader is None or source_fn is None:
            raise ValueError("pinned compact runtime must expose its strict export loader and source_digest")
        observed_source = source_fn()
    if observed_source != spec["runtime_source_sha256"]:
        raise ValueError("compact runtime source_digest differs from the pinned source bytes")
    model, metadata = loader(folder, spec["manifest_sha256"], device=device)
    if sha256_file(payload_path) != metadata.get("payload_sha256"):
        raise ValueError("student payload bytes changed during strict compact loading")
    return model.eval(), metadata


def _load_teacher(spec: Mapping[str, Any], origin: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a teacher through its own source alias and strict pinned engine."""
    folder, manifest, payload_path = _artifact_static(spec, "teacher", origin)
    runtime_root = _resolve(origin.parent, spec["runtime_root"], "teacher.runtime_root")
    if runtime_root == ROOT:
        from . import engine
    else:
        from .compact_teacher import import_historical_package

        alias = "_gradient_bridge_historical_" + spec["runtime_source_sha256"][:16]
        import_historical_package(runtime_root, pkg_alias=alias)
        engine = importlib.import_module(f"{alias}.engine")
    source_fn = getattr(engine, "source_digest", None)
    loader = getattr(engine, "load_checkpoint", None)
    if source_fn is None or loader is None:
        raise ValueError("pinned teacher runtime must expose source_digest and strict load_checkpoint")
    if source_fn() != spec["runtime_source_sha256"]:
        raise ValueError("historical alias runtime source_digest differs from the pinned source bytes")
    teacher, metadata = loader(folder, spec["manifest_sha256"], device=device)
    if sha256_file(payload_path) != spec["payload_sha256"] or metadata.get("payload_sha256") != spec["payload_sha256"]:
        raise ValueError("teacher payload bytes changed during strict historical loading")
    return teacher.eval(), metadata


def _load_model_pair(config: Mapping[str, Any], origin: Path, device: str) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any], dict[str, Any]]:
    teacher, teacher_meta = _load_teacher(config["teacher"], origin, device)
    student, student_meta = _load_student(config["student"], origin, device)
    for role, model in (("teacher", teacher), ("student", student)):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(module.training for module in model.modules()):
            raise ValueError(f"{role} model did not remain in eval mode")
    return teacher, student, teacher_meta, student_meta


def _manifest(config: Mapping[str, Any], origin: Path, role: str) -> tuple[dict[str, Any], Path]:
    from .compact_inference import load_native_manifest

    spec = config[role]
    path = _resolve(origin.parent, spec["path"], f"{role}.path")
    expected = _require_sha(spec["sha256"], f"{role}.sha256")
    if role == "train_manifest":
        roles = ["training", "train"]
    else:
        roles = ["development"]
    data, actual = load_native_manifest(path, expected, expected_roles=roles)
    if role == "train_manifest" and data.get("schema") != "self_audit.data.v2":
        raise ValueError("fit-P requires the canonical self_audit.data.v2 TRAIN manifest")
    if role == "val_manifest" and data.get("role") != "development":
        raise ValueError("gradient bridge scoring is limited to the locked development cohort")
    if type(data.get("synthetic")) is not bool:
        raise ValueError(f"{role} synthetic identity must be an explicit boolean")
    return data, path


def _canonical_train(data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from .compact_inference import _canonical_train_records

    labeled, unlabeled = _canonical_train_records(data)
    for record in unlabeled:
        if any(key in record for key in ("masks", "mask", "ground_truth", "labels")):
            raise ValueError(f"TRAIN-U record {record.get('id')} must not carry a mask/label field")
    return list(labeled), list(unlabeled)


def _train_image_path(record: Mapping[str, Any], manifest_dir: Path) -> Path:
    raw = record.get("prepared_image_path") or record.get("image") or record.get("path")
    return _resolve(manifest_dir, raw, f"TRAIN image {record.get('id')}")


def _image_file_hash(record: Mapping[str, Any], path: Path, name: str) -> str:
    expected = _require_sha(record.get("image_sha256") or record.get("sha256"), f"{name}.image_sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} image hash differs from its manifest")
    return actual


def _validate_lineage(
    config: Mapping[str, Any], train_data: Mapping[str, Any], train_sha: str,
    val_data: Mapping[str, Any], teacher_meta: Mapping[str, Any], student_meta: Mapping[str, Any],
) -> dict[str, Any]:
    profile = config["profile"]
    train_labeled, train_unlabeled = _canonical_train(train_data)
    train_lids = [str(row["id"]) for row in train_labeled]
    train_uids = [str(row["id"]) for row in train_unlabeled]
    if len(set(train_lids + train_uids)) != len(train_lids) + len(train_uids):
        raise ValueError("canonical TRAIN L/U source IDs must be unique")
    train_ids = set(train_lids + train_uids)
    val_records = val_data.get("patients")
    if not isinstance(val_records, list) or not val_records:
        raise ValueError("development manifest requires nonempty patient records")
    val_ids = [str(row.get("id")) for row in val_records if isinstance(row, Mapping)]
    if len(val_ids) != len(val_records) or len(val_ids) != len(set(val_ids)):
        raise ValueError("development manifest patient IDs must be unique")
    if train_ids & set(val_ids):
        raise ValueError(f"TRAIN/development patient ID overlap: {sorted(train_ids & set(val_ids))}")

    teacher_synthetic = teacher_meta.get("synthetic")
    student_synthetic = student_meta.get("synthetic")
    if type(teacher_synthetic) is not bool or type(student_synthetic) is not bool:
        raise ValueError("teacher and student manifests must explicitly record synthetic identity")
    if teacher_synthetic != student_synthetic or teacher_synthetic != train_data["synthetic"] or student_synthetic != val_data["synthetic"]:
        raise ValueError("teacher, student, TRAIN and development synthetic identities must agree")
    if profile == "fixture" and (not teacher_synthetic or not config["synthetic_fixture"]):
        raise ValueError("fixture profile requires explicitly synthetic teacher/student/data")
    if profile != "fixture" and (teacher_synthetic or config["synthetic_fixture"]):
        raise ValueError("non-fixture profile cannot contain synthetic artifacts or data")

    lineage = student_meta.get("lineage")
    identity = lineage.get("teacher_identity") if isinstance(lineage, Mapping) else None
    if profile != "fixture":
        if not isinstance(lineage, Mapping) or not isinstance(identity, Mapping):
            raise ValueError("student export must preserve teacher identity and TRAIN lineage")
        teacher_spec, student_spec = config["teacher"], config["student"]
        checks = {
            "payload_sha256": teacher_spec["payload_sha256"],
            "manifest_sha256": teacher_spec["manifest_sha256"],
            "inference_state": "student",
            "train_manifest_sha256": train_sha,
        }
        for key, expected in checks.items():
            if identity.get(key) != expected:
                raise ValueError(f"student lineage teacher_identity.{key} does not match the loaded teacher")
        if lineage.get("data_sha256") != train_sha:
            raise ValueError("compact student lineage is not bound to the configured TRAIN manifest")
        lineage_lids, lineage_uids = lineage.get("lids"), lineage.get("uids")
        if (not isinstance(lineage_lids, list) or not isinstance(lineage_uids, list)
                or len(set(lineage_lids)) != len(lineage_lids) or len(set(lineage_uids)) != len(lineage_uids)
                or set(lineage_lids) != set(train_lids) or set(lineage_uids) != set(train_uids)):
            raise ValueError("student L/U lineage does not match canonical TRAIN patient IDs")
        teacher_lids = identity.get("labeled_ids")
        if (not isinstance(teacher_lids, list) or len(set(teacher_lids)) != len(teacher_lids)
                or set(teacher_lids) != set(train_lids)):
            raise ValueError("teacher labeled IDs differ from the compact student's TRAIN-L identities")
        if lineage.get("label_budget") != len(train_lids) or identity.get("label_budget") != len(train_lids):
            raise ValueError("teacher/student label budgets must equal the declared TRAIN-L union")

    if profile == "historical":
        specs = {
            "teacher_manifest_sha256": config["teacher"]["manifest_sha256"],
            "teacher_payload_sha256": config["teacher"]["payload_sha256"],
            "student_manifest_sha256": config["student"]["manifest_sha256"],
            "student_payload_sha256": config["student"]["payload_sha256"],
            "train_manifest_sha256": train_sha,
            "val_manifest_sha256": config["val_manifest"]["sha256"],
        }
        for key, expected in specs.items():
            if expected != _HISTORICAL_PINS[key]:
                raise ValueError(f"historical profile requires exact pinned {key}")
        if teacher_meta.get("source_sha256") != config["teacher"]["runtime_source_sha256"]:
            raise ValueError("historical teacher manifest does not bind its pinned runtime source")
        if lineage.get("label_budget") != 8 or len(train_lids) != 8:
            raise ValueError("historical gradient bridge pair requires the locked L8 TRAIN budget")
        split_ids = lineage.get("lids", []) + lineage.get("uids", [])
        if set(split_ids) != train_ids or set(lineage.get("lids", [])) & set(lineage.get("uids", [])):
            raise ValueError("historical compact L/U IDs must partition the locked TRAIN manifest")

    return {
        "profile": profile,
        "synthetic": bool(teacher_synthetic),
        "train_patient_ids": sorted(train_ids),
        "train_labeled_ids": sorted(train_lids),
        "train_unlabeled_ids": sorted(train_uids),
        "evaluation_patient_ids": val_ids,
        "evaluation_frame_ids": {str(row["id"]): sorted(str(k) for k in row.get("masks", {})) for row in val_records},
        "teacher_state": "raw_student",
        "student_state": "raw_student",
        "lineage_verified": profile != "fixture",
    }


def _verify_image_overlap(train_data: Mapping[str, Any], train_manifest_path: Path, val_data: Mapping[str, Any], val_manifest_path: Path) -> dict[str, Any]:
    from .compact_inference import _load_raw_tzyx, _voxel_fingerprint, load_patient_cine

    labeled, unlabeled = _canonical_train(train_data)
    train_records = labeled + unlabeled
    train_files: set[str] = set()
    train_voxels: set[str] = set()
    checked_train = []
    for record in train_records:
        pid = record["id"]
        image_path = _train_image_path(record, train_manifest_path.parent)
        if not image_path.is_file():
            raise FileNotFoundError(f"TRAIN image missing for {pid}: {image_path}")
        image_sha = _image_file_hash(record, image_path, f"TRAIN patient {pid}")
        volume = _load_raw_tzyx(image_path, str(pid))
        fingerprint = _voxel_fingerprint(volume)
        train_files.add(image_sha)
        train_voxels.add(fingerprint)
        checked_train.append({"patient_id": str(pid), "image_sha256": image_sha, "voxel_sha256": fingerprint})

    checked_eval = []
    for record in val_data["patients"]:
        pid = str(record["id"])
        cine, _, meta = load_patient_cine(record, val_manifest_path.parent, normalization="prepared")
        if meta["image_sha256"] in train_files or meta["raw_voxel_sha256"] in train_voxels:
            raise ValueError(f"development image content overlaps TRAIN for patient {pid}")
        checked_eval.append({"patient_id": pid, "image_sha256": meta["image_sha256"], "voxel_sha256": meta["raw_voxel_sha256"], "shape_tzyx": list(cine.shape)})
    return {"train_images": checked_train, "development_images": checked_eval, "overlap": False}


def _check_device(config: Mapping[str, Any]) -> None:
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(config["threads"])


def preflight(config: Mapping[str, Any], config_path: str | Path, round_dir: str | Path, *, load_weights: bool = True) -> dict[str, Any]:
    """Verify model/source/data identity without opening reference-mask payloads."""
    validate_config(config)
    origin = Path(config_path).resolve()
    round_path = Path(round_dir).resolve()
    if (round_path / "VAL_EXPOSURE.json").exists():
        raise ValueError("this round already exposed labels; B cannot be started or relocked")
    _check_device(config)
    train_data, train_path = _manifest(config, origin, "train_manifest")
    val_data, val_path = _manifest(config, origin, "val_manifest")
    if (train_data["synthetic"] is not config["synthetic_fixture"]
            or val_data["synthetic"] is not config["synthetic_fixture"]):
        raise ValueError("config fixture flag must match TRAIN and development manifest synthetic fields")
    teacher_folder, teacher_meta, _ = _artifact_static(config["teacher"], "teacher", origin)
    student_folder, student_meta, _ = _artifact_static(config["student"], "student", origin)
    lineage = _validate_lineage(
        config, train_data, config["train_manifest"]["sha256"], val_data, teacher_meta, student_meta
    )
    image_checks = _verify_image_overlap(train_data, train_path, val_data, val_path)
    if load_weights:
        teacher, student, loaded_teacher_meta, loaded_student_meta = _load_model_pair(config, origin, "cpu")
        if loaded_teacher_meta.get("payload_sha256") != config["teacher"]["payload_sha256"]:
            raise ValueError("strict teacher loader returned unexpected payload metadata")
        if loaded_student_meta.get("payload_sha256") != config["student"]["payload_sha256"]:
            raise ValueError("strict compact loader returned unexpected payload metadata")
        del teacher, student
    return {
        "schema": "self_audit.gradient_bridge.preflight.v1",
        "status": "READY_FOR_FIXTURE" if config["profile"] == "fixture" else "READY_FOR_LOCK",
        "profile": config["profile"],
        "patient_inference_run": False,
        "ground_truth_payloads_opened": False,
        "strict_weights_loaded": bool(load_weights),
        "teacher": {"path": str(teacher_folder), "manifest_sha256": config["teacher"]["manifest_sha256"], "payload_sha256": config["teacher"]["payload_sha256"], "runtime_source_sha256": config["teacher"]["runtime_source_sha256"]},
        "student": {"path": str(student_folder), "manifest_sha256": config["student"]["manifest_sha256"], "payload_sha256": config["student"]["payload_sha256"], "runtime_source_sha256": config["student"]["runtime_source_sha256"]},
        "train_manifest_sha256": config["train_manifest"]["sha256"],
        "val_manifest_sha256": config["val_manifest"]["sha256"],
        "lineage": lineage,
        "image_overlap_check": image_checks,
        "limits": "Metadata, checkpoint bytes, and image bytes were checked; no patient prediction or reference-mask payload access occurred.",
    }


def _config_sha(config_path: str | Path, expected: str | None = None) -> str:
    actual = sha256_file(config_path)
    if expected is not None and _require_sha(expected, "config_sha256") != actual:
        raise ValueError("runtime config bytes differ from the supplied SHA-256")
    return actual


def _preflight_path(round_dir: str | Path) -> Path:
    return Path(round_dir).resolve() / "PREFLIGHT_B.json"


def preflight_stage(config_path: str | Path, round_dir: str | Path, *, expected_config_sha256: str | None = None) -> dict[str, Any]:
    """Recheck B inputs without opening reference-mask payloads and bind receipt."""
    config, origin = load_config(config_path)
    config_sha = _config_sha(origin, expected_config_sha256)
    round_path = Path(round_dir).resolve()
    if not round_path.is_dir():
        raise FileNotFoundError("run Diagnostic A preflight first; the shared round directory must exist")
    _assert_round_disjoint(round_path, origin, config, origin)
    report = preflight(config, origin, round_path, load_weights=True)
    record = {
        "schema": "self_audit.gradient_bridge.preflight_record.v1",
        "config_path": str(origin),
        "config_sha256": config_sha,
        "report": report,
    }
    _write_once(_preflight_path(round_path), record)
    return report


def _read_preflight(round_dir: str | Path, config_path: str | Path, config_sha: str) -> dict[str, Any]:
    path = _preflight_path(round_dir)
    record = _read_object(path)
    if (record.get("schema") != "self_audit.gradient_bridge.preflight_record.v1"
            or record.get("config_sha256") != config_sha
            or Path(record.get("config_path", "")).resolve() != Path(config_path).resolve()):
        raise ValueError("B preflight receipt/config identity mismatch")
    if record.get("report", {}).get("status") not in {"READY_FOR_FIXTURE", "READY_FOR_LOCK"}:
        raise ValueError("B preflight did not complete successfully")
    return record


def _assert_round_disjoint(round_dir: Path, config_path: Path, config: Mapping[str, Any], origin: Path) -> None:
    protected = [config_path.resolve(), origin.parent.resolve()]
    paths = _config_paths(config, origin)
    protected.extend(paths[name] for name in ("train_manifest", "val_manifest", "teacher_path", "student_path", "teacher_runtime", "student_runtime"))
    for manifest_key in ("train_manifest", "val_manifest"):
        data, manifest_path = _manifest(config, origin, manifest_key)
        base = manifest_path.parent
        if manifest_key == "train_manifest":
            labeled, unlabeled = _canonical_train(data)
            for record in labeled + unlabeled:
                protected.append(_train_image_path(record, base))
        else:
            for record in data["patients"]:
                protected.append(_resolve(base, record.get("image") or record.get("path"), f"patient {record['id']} image"))
                for mask in record.get("masks", {}).values():
                    protected.append(_resolve(base, mask.get("path"), f"patient {record['id']} mask"))
    if any(round_dir == item or round_dir.is_relative_to(item) or item.is_relative_to(round_dir) for item in protected):
        raise ValueError("shared round directory overlaps runtime config or input assets")


def _execution_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    import platform
    from .compact_runtime import deterministic_runtime

    torch.set_num_threads(config["threads"])
    # Capture the effective context used by predict, including CUDA deterministic
    # bilinear VJPs, rather than the caller's unrelated initial flags.
    with deterministic_runtime(config["device"] == "cuda"):
        flags = {
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": config["device"],
        "threads": config["threads"],
        "batch_size": config["batch_size"],
        "normalization": config["normalization"],
        "dtype": "float32",
        **flags,
        "cuda_version": torch.version.cuda,
        "feature_tap": "compact.encoder[-1]",
        "feature_support": "false_pad_native_validity_then_full_cell_area_support",
        "student_conditioning": "detached_original_encoder_skips",
        "bridge_conditioning": "P_then_teacher_refine0_upsample_fuse_zero_detail_head; no_teacher_skips",
        "class_order": list(CLASS_NAMES),
        "weights_updated": False,
    }


def _model_identity(spec: Mapping[str, Any]) -> dict[str, str]:
    return {key: str(spec[key]) for key in ("manifest_sha256", "payload_sha256", "runtime_source_sha256")}


def _assert_a_b_match(a_protocol: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Require Diagnostic A and B to share the exact pair, cohort, and preprocessing."""
    a_config = a_protocol.get("config")
    if not isinstance(a_config, Mapping):
        raise ValueError("A lock protocol.config is missing")
    for role in ("teacher", "student"):
        actual = a_config.get(role)
        if not isinstance(actual, Mapping):
            raise ValueError(f"A locked config is missing {role} identity")
        for key, value in _model_identity(config[role]).items():
            if actual.get(key) != value:
                raise ValueError(f"A/B {role}.{key} identity mismatch")
    a_manifest = a_config.get("val_manifest") or a_config.get("evaluation_manifest")
    if not isinstance(a_manifest, Mapping) or a_manifest.get("sha256") != config["val_manifest"]["sha256"]:
        raise ValueError("A/B evaluation manifest identity mismatch")
    # Older historical A configs predate the explicit fixture flag and are
    # necessarily real-data configs. Treat the absent field as False while
    # still requiring an exact match when either side is synthetic.
    a_synthetic = a_config.get("synthetic_fixture", False)
    if a_synthetic is not config["synthetic_fixture"]:
        raise ValueError("A/B synthetic cohort identity mismatch")
    a_normalization = a_config.get("normalization", "prepared")
    if a_normalization != config["normalization"]:
        raise ValueError("A/B image preprocessing mismatch")
    if a_config.get("parity") != config.get("parity"):
        raise ValueError("A/B native parity config mismatch")
    if a_config.get("device") != config.get("device"):
        raise ValueError("A/B locked device mismatch")
    a_execution = a_protocol.get("execution")
    if isinstance(a_execution, Mapping) and a_execution.get("device") not in (None, config["device"]):
        raise ValueError("A/B locked device mismatch")
    return {
        "teacher": _model_identity(config["teacher"]),
        "student": _model_identity(config["student"]),
        "evaluation_manifest_sha256": config["val_manifest"]["sha256"],
        "normalization": config["normalization"],
        "synthetic_fixture": config["synthetic_fixture"],
        "parity": config["parity"],
        "device": config["device"],
    }


def _assets_for_lock(config: Mapping[str, Any], origin: Path, config_sha: str, alignment_path: Path | None = None) -> dict[str, dict[str, str]]:
    assets: dict[str, dict[str, str]] = {}

    def add(name: str, path: Path, expected: str) -> None:
        path = path.resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"locked asset is missing or changed: {name}")
        assets[name] = {"path": str(path), "sha256": expected}

    add("B_config", origin, config_sha)
    for role in ("train_manifest", "val_manifest"):
        path = _resolve(origin.parent, config[role]["path"], f"{role}.path")
        add(role, path, config[role]["sha256"])
        data = _read_object(path)
        if role == "train_manifest":
            labeled, unlabeled = _canonical_train(data)
            records = labeled + unlabeled
            for record in records:
                image = _train_image_path(record, path.parent)
                add(f"TRAIN_image:{record['id']}", image, record.get("image_sha256") or record.get("sha256"))
        else:
            for record in data["patients"]:
                image = _resolve(path.parent, record.get("image") or record.get("path"), f"VAL image {record['id']}")
                add(f"VAL_image:{record['id']}", image, record.get("image_sha256") or record.get("sha256"))
    for role in ("teacher", "student"):
        spec = config[role]
        folder = _resolve(origin.parent, spec["path"], f"{role}.path")
        add(f"{role}_manifest", folder / "MANIFEST.json", spec["manifest_sha256"])
        add(f"{role}_payload", folder / "weights.pt", spec["payload_sha256"])
        runtime = _resolve(origin.parent, spec["runtime_root"], f"{role}.runtime_root")
        for index, source in enumerate(_runtime_source_files(runtime)):
            add(f"{role}_runtime_file_{index:03d}", source, sha256_file(source))
    if alignment_path is not None:
        add("fit_P", alignment_path, sha256_file(alignment_path))
    return assets


def _alignment_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "student_sha256": config["student"]["payload_sha256"],
        "teacher_sha256": config["teacher"]["payload_sha256"],
        "train_manifest_sha256": config["train_manifest"]["sha256"],
        "student_manifest_sha256": config["student"]["manifest_sha256"],
        "teacher_manifest_sha256": config["teacher"]["manifest_sha256"],
        "student_runtime_source_sha256": config["student"]["runtime_source_sha256"],
        "teacher_runtime_source_sha256": config["teacher"]["runtime_source_sha256"],
    }


def _alignment_path(config: Mapping[str, Any], round_dir: Path) -> Path:
    return round_dir / config["fit_p"]["output_file"]


def _load_alignment(config: Mapping[str, Any], round_dir: Path):
    from .bridge_alignment import load_bridge_alignment

    path = _alignment_path(config, round_dir)
    if not path.is_file():
        raise FileNotFoundError("fit-P alignment artifact is missing; lock-B never creates or fits P")
    return load_bridge_alignment(path, expected_identities=_alignment_identities(config))


def _seeded_patient_records(records: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Order TRAIN patients deterministically from the declared fit seed."""
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        pid = str(record["id"])
        if pid in unique:
            raise ValueError(f"duplicate TRAIN patient record: {pid}")
        unique[pid] = dict(record)
    return sorted(unique.values(), key=lambda row: (hashlib.sha256(f"{seed}\0{row['id']}".encode()).hexdigest(), row["id"]))


def _fit_valid_support(valid_hw: torch.Tensor, input_hw: tuple[int, int], target_hw: tuple[int, int]) -> torch.Tensor:
    import torch.nn.functional as F

    height, width = input_hw
    padded_h = height + (-height) % 32
    padded_w = width + (-width) % 32
    if target_hw != (padded_h // 4, padded_w // 4):
        raise ValueError("teacher pre-refine0 tap has an unexpected native stride/geometry")
    padded = F.pad(valid_hw[None, None].to(dtype=torch.float32), (0, padded_w - width, 0, padded_h - height), value=0)
    return F.interpolate(padded, size=target_hw, mode="area").ge(1.0 - 1e-6)[:, 0]


def _iter_fit_samples(config: Mapping[str, Any], origin: Path, train_data: Mapping[str, Any], teacher: torch.nn.Module, student: torch.nn.Module):
    from .bridge_alignment import BridgeFitSample
    from .compact_inference import _load_raw_tzyx, normalize_image
    from .diagnostic_models import CompactStudentTap, extract_teacher_pre_refine0

    labeled, unlabeled = _canonical_train(train_data)
    ordered_patients = _seeded_patient_records(labeled + unlabeled, config["fit_p"]["seed"])
    selected = ordered_patients[:config["fit_p"]["max_train_patients"]]
    tap = CompactStudentTap(student)
    device = torch.device(config["device"])
    train_manifest_path = _resolve(origin.parent, config["train_manifest"]["path"], "train_manifest.path")

    def patient_samples(record: Mapping[str, Any]):
        pid = str(record["id"])
        image_path = _train_image_path(record, train_manifest_path.parent)
        _image_file_hash(record, image_path, f"TRAIN patient {pid}")
        raw = _load_raw_tzyx(image_path, pid)
        for t in range(raw.shape[0]):
            for z in range(raw.shape[1]):
                source_id = f"{pid}:t{t}:z{z}"
                indices = (max(0, z - 1), z, min(raw.shape[1] - 1, z + 1))
                image_context = np.stack([raw[t, index] for index in indices], axis=0)
                image_context = normalize_image(image_context, mode=config["normalization"])
                x = torch.from_numpy(np.ascontiguousarray(image_context))[None].to(device=device, dtype=torch.float32)
                valid = torch.ones((1, x.shape[-2], x.shape[-1]), dtype=torch.bool, device=device)
                with torch.no_grad():
                    context = tap.extract(x, valid)
                    target = extract_teacher_pre_refine0(teacher, x)
                    support = _fit_valid_support(valid[0], tuple(x.shape[-2:]), tuple(target.shape[-2:]))
                yield BridgeFitSample(
                    student_h=context.h.detach(), teacher_pre_refine0=target.detach(),
                    valid_mask=support, patient_id=pid, source_id=source_id, split="TRAIN",
                )

    per_patient = [patient_samples(record) for record in selected]
    for group in itertools.zip_longest(*per_patient):
        for sample in group:
            if sample is not None:
                yield sample


def fit_p_stage(config_path: str | Path, round_dir: str | Path, *, expected_config_sha256: str | None = None) -> dict[str, Any]:
    """Fit deterministic P from bounded TRAIN image-only activation statistics."""
    import time
    started = time.perf_counter()
    config, origin = load_config(config_path)
    config_sha = _config_sha(origin, expected_config_sha256)
    round_path = Path(round_dir).resolve()
    _read_preflight(round_path, origin, config_sha)
    if (round_path / "protocol-B.lock.json").exists() or (round_path / "VAL_EXPOSURE.json").exists():
        raise ValueError("fit-P must precede lock-B and all VAL exposure")
    output_path = _alignment_path(config, round_path)
    if output_path.exists():
        raise FileExistsError("fit-P is append-once; existing P is never overwritten")
    train_data, _ = _manifest(config, origin, "train_manifest")
    _canonical_train(train_data)  # Reject U labels before any image/model access.
    _check_device(config)
    teacher, student, _, _ = _load_model_pair(config, origin, config["device"])
    from .bridge_alignment import BridgeAlignmentConfig, fit_bridge_alignment

    fit_config = config["fit_p"]
    alignment = fit_bridge_alignment(
        _iter_fit_samples(config, origin, train_data, teacher, student),
        BridgeAlignmentConfig(
            max_source_ids=fit_config["max_source_ids"],
            max_train_patients=fit_config["max_train_patients"],
            max_pixels=fit_config["max_pixels"],
            ridge=float(fit_config["ridge"]),
            seed=fit_config["seed"],
        ),
        _alignment_identities(config), output_path,
    )
    if config["device"] == "cuda":
        torch.cuda.synchronize()
    record = {
        "schema": "self_audit.gradient_bridge.fit_p.v1",
        "status": "FIT_ON_TRAIN_IMAGES_ONLY",
        "config_sha256": config_sha,
        "artifact_path": str(output_path.resolve()),
        "artifact_sha256": sha256_file(output_path),
        "artifact_identity": alignment.artifact_sha256,
        "metadata": alignment.metadata,
        "elapsed_seconds_including_load_and_fit": time.perf_counter() - started,
        "ground_truth_payloads_opened": False,
    }
    _write_once(round_path / "FIT_P.json", record)
    return record


def lock_b_stage(config_path: str | Path, round_dir: str | Path, *, expected_config_sha256: str | None = None) -> dict[str, Any]:
    """Bind B to the same frozen A pair/cohort and the completed TRAIN fit-P."""
    from .diagnostic_protocol import create_lock, read_lock, verify_frozen

    config, origin = load_config(config_path)
    config_sha = _config_sha(origin, expected_config_sha256)
    round_path = Path(round_dir).resolve()
    preflight_record = _read_preflight(round_path, origin, config_sha)
    a_lock = read_lock(round_path, "A")
    a_frozen = verify_frozen(round_path, "A")
    paired = _assert_a_b_match(a_lock["protocol"], config)
    alignment = _load_alignment(config, round_path)
    fit_record = _read_object(round_path / "FIT_P.json")
    if fit_record.get("artifact_sha256") != sha256_file(_alignment_path(config, round_path)):
        raise ValueError("fit-P receipt does not bind the current alignment artifact")
    if fit_record.get("config_sha256") != config_sha or fit_record.get("artifact_identity") != alignment.artifact_sha256:
        raise ValueError("fit-P receipt/config/alignment identity mismatch")
    current = preflight(config, origin, round_path, load_weights=True)
    if current != preflight_record["report"]:
        raise ValueError("B preflight assets changed before lock-B")
    execution = _execution_snapshot(config)
    protocol = {
        "config": config,
        "config_sha256": config_sha,
        "execution": execution,
        "paired_A_B_identity": paired,
        "A_lock_sha256": a_lock["record_sha256"],
        "A_freeze_sha256": a_frozen["record_sha256"],
        "alignment_artifact_sha256": alignment.artifact_sha256,
        "alignment_file_sha256": sha256_file(_alignment_path(config, round_path)),
        "lineage": preflight_record["report"]["lineage"],
    }
    assets = _assets_for_lock(config, origin, config_sha, _alignment_path(config, round_path))
    return create_lock(round_path, "B", protocol, assets)


def _prediction_dir(round_dir: Path) -> Path:
    return round_dir / "B" / "predictions"


def _prediction_name(patient_id: str, frame_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", patient_id).strip("._-")[:48] or "patient"
    suffix = hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{suffix}_t{frame_id}_logits.npy"


def _feature_support(context: Any) -> torch.Tensor:
    """Map center-pixel validity to bottleneck cells, marking padded tails false."""
    import torch.nn.functional as F

    h = context.h
    valid = context.valid_mask
    if valid is None:
        return torch.ones((h.shape[0], *h.shape[-2:]), dtype=torch.bool, device=h.device)
    if valid.dtype is not torch.bool or valid.ndim != 3 or valid.shape[0] != h.shape[0]:
        raise ValueError("tap context valid mask must be boolean [B,H,W]")
    height, width = context.input_hw
    padded_h, padded_w = context.padded_hw
    if valid.shape[-2:] != (height, width) or padded_h < height or padded_w < width:
        raise ValueError("tap context valid mask and padded native geometry disagree")
    padded = F.pad(valid[:, None].to(device=h.device, dtype=torch.float32),
                   (0, padded_w - width, 0, padded_h - height), value=0.0)
    support = F.interpolate(padded, size=h.shape[-2:], mode="area")[:, 0]
    return support.ge(1.0 - 1e-6)


def _bridge_dependence_probe(config: Mapping[str, Any], origin: Path, val_data: Mapping[str, Any], teacher: torch.nn.Module, student: torch.nn.Module, bridge: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    from .compact_inference import extract_slice_context, load_patient_cine
    from .diagnostic_models import CompactStudentTap, check_bridge_h_dependence
    from .gradient_bridge import KDLossContext

    records = val_data["patients"]
    tap = CompactStudentTap(student)
    by_geometry: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for record in records:
        # Image metadata is read through the canonical native loader. Mask paths
        # remain metadata and no reference payload is opened or hashed here.
        cine, _, _ = load_patient_cine(record, _resolve(origin.parent, config["val_manifest"]["path"], "val_manifest.path").parent, normalization=config["normalization"])
        by_geometry.setdefault(tuple(int(x) for x in cine.shape[-2:]), []).append(record)
        del cine
    pair = next((rows[:2] for rows in by_geometry.values() if len(rows) >= 2), None)
    if pair is None:
        return {
            "status": "NOT_AVAILABLE_NO_COMPATIBLE_CROSSPATIENT_SHAPE",
            "patient_ids": [],
            "keep_vs_zero_mean_abs": None,
            "keep_vs_shuffle_mean_abs": None,
            "h_vjp_norm": None,
        }
    features, targets, valids, input_hw, pids = [], [], [], None, []
    manifest_dir = _resolve(origin.parent, config["val_manifest"]["path"], "val_manifest.path").parent
    for record in pair:
        cine, valid_volume, _ = load_patient_cine(record, manifest_dir, normalization=config["normalization"])
        frame = 0
        x = extract_slice_context(cine, frame, 0).to(device)
        valid = np.ones((1, *x.shape[-2:]), dtype=bool) if valid_volume is None else valid_volume[frame, 0][None]
        valid_tensor = torch.from_numpy(np.ascontiguousarray(valid)).to(device=device, dtype=torch.bool)
        with torch.no_grad():
            context = tap.extract(x, valid_tensor)
            target = torch.softmax(teacher(x).float(), dim=1).detach()
        features.append(context.h)
        targets.append(target)
        valids.append(valid_tensor)
        input_hw = context.input_hw
        pids.append(str(record["id"]))
        del cine
    if input_hw is None or features[0].shape != features[1].shape:
        return {
            "status": "NOT_AVAILABLE_INCOMPATIBLE_FEATURE_GEOMETRY",
            "patient_ids": pids,
            "keep_vs_zero_mean_abs": None,
            "keep_vs_shuffle_mean_abs": None,
            "h_vjp_norm": None,
        }
    result = check_bridge_h_dependence(
        bridge, torch.cat(features, dim=0), input_hw, pids, config["directions"]["shuffle_seed"]
    )
    base_h = torch.cat(features, dim=0).detach()
    target_batch = torch.cat(targets, dim=0)
    valid_batch = torch.cat(valids, dim=0)

    def objective_probe(probe_h: torch.Tensor) -> tuple[list[float], list[float]]:
        logits = bridge(probe_h, input_hw)
        losses = []
        for index in range(len(pids)):
            context = KDLossContext(
                target_batch[index:index + 1], valid_batch[index:index + 1],
                temperature=float(config["kd"]["temperature"]),
                coefficient=float(config["kd"]["coefficient"]),
                confidence=float(config["kd"]["confidence"]),
                class_weights=config["kd"]["class_weights"],
                denominator=config["kd"]["denominator"],
            )
            losses.append(context(logits[index:index + 1]))
        loss_vector = torch.stack(losses)
        gradient = torch.autograd.grad(loss_vector.sum(), probe_h, retain_graph=False, create_graph=False)[0]
        norms = torch.linalg.vector_norm(gradient.double().flatten(1), dim=1)
        return [float(value) for value in loss_vector.detach().cpu().tolist()], [float(value) for value in norms.detach().cpu().tolist()]

    keep_h = base_h.clone().requires_grad_(True)
    keep_kd, keep_grad = objective_probe(keep_h)
    zero_h = torch.zeros_like(base_h).requires_grad_(True)
    zero_kd, zero_grad = objective_probe(zero_h)
    donor = torch.as_tensor(result.donor_indices, dtype=torch.long, device=base_h.device)
    shuffled_h = base_h.index_select(0, donor).detach().clone().requires_grad_(True)
    shuffle_kd, shuffle_grad = objective_probe(shuffled_h)
    keep_zero = result.keep_vs_zero_mean_abs.detach().cpu().numpy()
    keep_shuffle = result.keep_vs_shuffle_mean_abs.detach().cpu().numpy()
    vjp = result.h_vjp_norm.detach().cpu().numpy()
    epsilon = float(config["directions"]["norm_epsilon"])
    active = bool(np.any(keep_zero > epsilon) and np.any(keep_shuffle > epsilon) and np.any(vjp > epsilon))
    return {
        "status": "ACTIVE_H_DEPENDENCE" if active else "INVALID_NO_MEASURABLE_H_DEPENDENCE",
        "patient_ids": pids,
        "keep_vs_zero_mean_abs": keep_zero.astype(float).tolist(),
        "keep_vs_shuffle_mean_abs": keep_shuffle.astype(float).tolist(),
        "h_vjp_norm": vjp.astype(float).tolist(),
        "keep_kd": keep_kd,
        "zero_kd": zero_kd,
        "crosspatient_shuffle_kd": shuffle_kd,
        "keep_kd_h_gradient_norm": keep_grad,
        "zero_kd_h_gradient_norm": zero_grad,
        "crosspatient_shuffle_kd_h_gradient_norm": shuffle_grad,
        "donor_indices": list(result.donor_indices),
        "threshold": epsilon,
        "selection": "first_two_manifest_patients_with_equal_native_HW; t0,z0; image_only",
        "conditioning": "student_h_only; zero_teacher_detail; no_teacher_skips",
        "kd_scope": "per_patient_probe_denominator",
        "cost_counts": {"teacher_full_forwards": 2, "student_encoder_forwards": 2,
                        "bridge_forwards": 6, "backward_calls": 4},
    }


def _predict_stage_impl(config_path: str | Path, round_dir: str | Path, *, expected_config_sha256: str | None = None) -> dict[str, Any]:
    """Generate and freeze every D0-D5/rho logit shard using images only."""
    from .compact_inference import extract_slice_context, load_native_manifest, load_patient_cine
    from .diagnostic_models import CompactStudentTap, TeacherSuffixBridge
    from .diagnostic_protocol import freeze_predictions, read_lock, verify_frozen
    from .gradient_bridge import KDLossContext, compute_gradient_arms

    config, origin = load_config(config_path)
    config_sha = _config_sha(origin, expected_config_sha256)
    round_path = Path(round_dir).resolve()
    preflight_record = _read_preflight(round_path, origin, config_sha)
    lock = read_lock(round_path, "B")
    if lock["protocol"]["execution"] != _execution_snapshot(config):
        raise ValueError("B effective execution environment changed after lock")
    a_lock = read_lock(round_path, "A")
    a_frozen = verify_frozen(round_path, "A")
    _assert_a_b_match(a_lock["protocol"], config)
    if (lock["protocol"].get("config_sha256") != config_sha
            or lock["protocol"].get("A_lock_sha256") != a_lock["record_sha256"]
            or lock["protocol"].get("A_freeze_sha256") != a_frozen["record_sha256"]):
        raise ValueError("B lock is not bound to the current A lock and frozen prediction inventory")
    if lock["protocol"].get("alignment_file_sha256") != sha256_file(_alignment_path(config, round_path)):
        raise ValueError("fit-P changed after lock-B")
    current_preflight = preflight(config, origin, round_path, load_weights=True)
    if current_preflight != preflight_record["report"]:
        raise ValueError("B image/model/preprocessing identity changed after preflight")
    val_data, val_manifest_path = _manifest(config, origin, "val_manifest")
    alignment = _load_alignment(config, round_path)
    _check_device(config)
    device = torch.device(config["device"])
    import time

    started = time.perf_counter()
    reference_forward_seconds = 0.0
    gradient_arm_seconds = 0.0
    teacher, student, teacher_meta, student_meta = _load_model_pair(config, origin, config["device"])
    tap = CompactStudentTap(student)
    bridge = TeacherSuffixBridge(teacher, alignment, expected_identities=_alignment_identities(config)).to(device).eval()
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dependence_started = time.perf_counter()
    dependence = _bridge_dependence_probe(config, origin, val_data, teacher, student, bridge, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dependence_seconds = time.perf_counter() - dependence_started
    output_dir = _prediction_dir(round_path)
    if output_dir.exists():
        raise FileExistsError("B prediction output already exists; prediction freezes are append-once")
    output_dir.mkdir(parents=True)
    rho_values = [float(value) for value in config["directions"]["rho_sensitivity"]]
    primary_rho = float(config["directions"]["rho_primary"])
    primary_index = rho_values.index(primary_rho)
    diagnostics: list[dict[str, Any]] = []
    batch_diagnostics: list[dict[str, Any]] = []
    reference_batches = 0
    shard_inventory: list[dict[str, Any]] = []
    parity = {"slices": 0, "mismatches": 0, "max_abs_error": 0.0}
    frame_count = 0
    sample_count = 0
    manifest_dir = val_manifest_path.parent
    class_names = ("D0", "D1", "D2", "D3", "D4", "D5")

    for patient in val_data["patients"]:
        pid = str(patient["id"])
        cine, valid_volume, cine_meta = load_patient_cine(patient, manifest_dir, normalization=config["normalization"])
        t_count, z_count, height, width = (int(value) for value in cine.shape)
        frame_ids = sorted(int(value) for value in patient["masks"])
        for frame_id in frame_ids:
            filename = _prediction_name(pid, str(frame_id))
            shard_path = output_dir / filename
            shard = np.lib.format.open_memmap(
                shard_path, mode="w+", dtype=np.float32,
                shape=(len(rho_values), len(class_names), 4, z_count, height, width),
            )
            fallback_by_rho = [{arm: False for arm in class_names} for _ in rho_values]
            for z_start in range(0, z_count, config["batch_size"]):
                z_indices = list(range(z_start, min(z_count, z_start + config["batch_size"])))
                x = torch.cat([extract_slice_context(cine, frame_id, zi) for zi in z_indices], dim=0).to(device)
                if valid_volume is None:
                    valid_np = np.ones((len(z_indices), height, width), dtype=bool)
                else:
                    valid_np = np.ascontiguousarray(valid_volume[frame_id, z_indices])
                valid = torch.from_numpy(valid_np).to(device=device, dtype=torch.bool)
                # Tap extraction freezes the native decoder skips. The baseline
                # forward is independently evaluated for exact D0 parity.
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                reference_started = time.perf_counter()
                context = tap.extract(x, valid)
                with torch.no_grad():
                    original_logits = student(x)
                    teacher_logits = teacher(x)
                    q_teacher = torch.softmax(teacher_logits.float(), dim=1).detach()
                    d0_logits = tap.decode(context)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                reference_forward_seconds += time.perf_counter() - reference_started
                reference_batches += 1
                if original_logits.shape != d0_logits.shape:
                    raise ValueError("D0 tap and exported compact logits have different native geometry")
                error = float((original_logits - d0_logits).abs().max().item())
                parity["slices"] += len(z_indices)
                parity["max_abs_error"] = max(parity["max_abs_error"], error)
                if not torch.allclose(original_logits, d0_logits, atol=float(config["parity"]["logit_atol"]), rtol=float(config["parity"]["logit_rtol"])):
                    parity["mismatches"] += len(z_indices)
                    raise ValueError(f"D0 tap parity failed for {pid} frame {frame_id} z={z_indices[0]}")
                feature_mask = _feature_support(context)
                kd_context = KDLossContext(
                    q_teacher, valid,
                    temperature=float(config["kd"]["temperature"]),
                    coefficient=float(config["kd"]["coefficient"]),
                    confidence=config["kd"]["confidence"],
                    class_weights=config["kd"]["class_weights"],
                    denominator=config["kd"]["denominator"],
                )
                student_logits_fn = lambda h, feature_context=context: tap.decode(feature_context, h)
                bridge_logits_fn = lambda h, input_hw=context.input_hw: bridge(h, input_hw)
                sample_ids = [f"{pid}:t{frame_id}:z{zi}" for zi in z_indices]
                for rho_index, rho in enumerate(rho_values):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    arm_started = time.perf_counter()
                    arms = compute_gradient_arms(
                        context.h, student_logits_fn, bridge_logits_fn, kd_context,
                        feature_mask=feature_mask,
                        rho=rho,
                        seed=int(config["directions"]["shuffle_seed"]),
                        sample_ids=sample_ids,
                        min_gradient_norm=float(config["directions"]["norm_epsilon"]),
                        min_feature_norm=float(config["directions"]["norm_epsilon"]),
                        frozen_modules=(tap.model, bridge),
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    batch_elapsed = time.perf_counter() - arm_started
                    gradient_arm_seconds += batch_elapsed
                    batch_id = f"{pid}:t{frame_id}:z{z_start}:rho{rho_index}"
                    batch_diagnostics.append({
                        "batch_id": batch_id, "sample_ids": sample_ids, "rho": rho,
                        "scope": "one_batch_common_denominator_not_per_sample",
                        "kd_denominator": float(arms.denominator),
                        "accepted_pixels": int(arms.accepted_pixels),
                        "per_arm_kd": {arm: {head: float(value.item()) for head, value in losses.items()}
                                       for arm, losses in arms.per_arm_kd_losses.items()},
                        "forward_counts": dict(arms.forward_counts),
                        "backward_counts": dict(arms.backward_counts),
                        "elapsed_seconds": batch_elapsed,
                    })
                    for arm_index, arm in enumerate(class_names):
                        logits = arms.logits[arm].detach().to(device="cpu", dtype=torch.float32).numpy()
                        if logits.shape != (len(z_indices), 4, height, width) or not np.isfinite(logits).all():
                            raise ValueError(f"{arm} prediction has invalid class order, shape, or finite values")
                        for batch_index, zi in enumerate(z_indices):
                            shard[rho_index, arm_index, :, zi] = logits[batch_index]
                    for batch_index, zi in enumerate(z_indices):
                        status = {arm: arms.statuses.get(arm, ("unknown",) * len(z_indices))[batch_index] for arm in class_names}
                        d1 = status["D1"] == "active"
                        d2 = status["D2"] == "active"
                        group = "both_active" if d1 and d2 else ("self_only" if d1 else ("bridge_only" if d2 else "neither"))
                        cosine = None
                        if d1 and d2:
                            g1 = (arms.gradients["D1"][batch_index] * feature_mask[batch_index][None]).double().flatten()
                            g2 = (arms.gradients["D2"][batch_index] * feature_mask[batch_index][None]).double().flatten()
                            norm_product = torch.linalg.vector_norm(g1) * torch.linalg.vector_norm(g2)
                            if float(norm_product.item()) > 0:
                                cosine = float((torch.dot(g1, g2) / norm_product).item())
                        diag = {
                            "patient_id": pid, "frame_id": str(frame_id), "z": int(zi),
                            "rho": rho, "rho_is_primary": rho_index == primary_index,
                            "gradient_group": group, "statuses": status,
                            "batch_id": batch_id,
                            "D3_shuffle_mode": arms.shuffle_modes["D3"][batch_index],
                            "D1_D2_cosine_if_both_active": cosine,
                            "gradient_norms": {arm: float(arms.gradient_norms[arm][batch_index].item()) for arm in arms.gradient_norms},
                            "displacement_norms": {arm: float(arms.displacement_norms[arm][batch_index].item()) for arm in arms.displacement_norms},
                        }
                        diagnostics.append(diag)
                        for arm in class_names:
                            fallback_by_rho[rho_index][arm] |= status[arm] != "active" and arm != "D0"
                        sample_count += 1
            shard.flush()
            del shard
            # Ensure the NPY header and every streamed slice are visible before freeze.
            with shard_path.open("rb") as stream:
                os.fsync(stream.fileno())
            shard_inventory.append({
                "patient_id": pid, "frame_id": str(frame_id),
                "path": str(shard_path.resolve()), "shape": [len(rho_values), len(class_names), 4, z_count, height, width],
                "rho_values": rho_values, "primary_rho_index": primary_index,
                "image_sha256": cine_meta["image_sha256"],
                "fallback_by_rho": fallback_by_rho,
            })
            frame_count += 1
        del cine

    if parity["slices"] <= 0 or parity["mismatches"] != 0:
        raise ValueError("D0 parity is unavailable or mismatched")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    try:
        import resource
        raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        cpu_peak_rss_bytes = raw_rss if sys.platform == "darwin" else raw_rss * 1024
    except (ImportError, AttributeError, OSError):
        cpu_peak_rss_bytes = None
    cost = {
        "reference_forward_seconds": reference_forward_seconds,
        "gradient_arm_forward_backward_decode_seconds": gradient_arm_seconds,
        "h_dependence_seconds": dependence_seconds,
        "end_to_end_elapsed_seconds": elapsed,
        "native_slices_per_second": (parity["slices"] / elapsed) if elapsed > 0 else None,
        "gradient_arm_batches": len(batch_diagnostics),
        "reference_batches": reference_batches,
        "reference_forward_counts": {"student_full": reference_batches, "student_encoder_tap": reference_batches,
                                     "student_decoder_parity": reference_batches, "teacher_full": reference_batches},
        "gradient_forward_counts": {key: sum(row["forward_counts"].get(key, 0) for row in batch_diagnostics)
                                    for key in sorted({k for row in batch_diagnostics for k in row["forward_counts"]})},
        "gradient_backward_counts": {key: sum(row["backward_counts"].get(key, 0) for row in batch_diagnostics)
                                     for key in sorted({k for row in batch_diagnostics for k in row["backward_counts"]})},
        "scope": "predict including model load, image IO, parity, bridge dependence and shards; excludes preflight/fit/score",
        "cpu_peak_rss_bytes": cpu_peak_rss_bytes,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        "gpu_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    prediction_receipt = {
        "schema": PREDICTION_SCHEMA,
        "stage": "B",
        "status": "PREDICTIONS_READY_FOR_SHARED_SCORE",
        "config_sha256": config_sha,
        "B_lock_sha256": lock["record_sha256"],
        "A_lock_sha256": a_lock["record_sha256"],
        "A_freeze_sha256": a_frozen["record_sha256"],
        "teacher_payload_sha256": config["teacher"]["payload_sha256"],
        "student_payload_sha256": config["student"]["payload_sha256"],
        "teacher_loaded_payload_sha256": teacher_meta.get("payload_sha256"),
        "student_loaded_payload_sha256": student_meta.get("payload_sha256"),
        "val_manifest_sha256": config["val_manifest"]["sha256"],
        "normalization": config["normalization"],
        "rho_values": rho_values,
        "primary_rho": primary_rho,
        "arms": list(class_names),
        "coverage": {"expected_patients": [str(row["id"]) for row in val_data["patients"]],
                     "predicted_patients": sorted({item["patient_id"] for item in shard_inventory}),
                     "frames": frame_count, "slice_samples": sample_count},
        "D0_parity": parity,
        "h_dependence": dependence,
        "cost": cost,
        "gradient_diagnostics": diagnostics,
        "batch_diagnostics": batch_diagnostics,
        "shards": shard_inventory,
        "ground_truth_payloads_opened": False,
        "synthetic_fixture": config["synthetic_fixture"],
    }
    receipt_path = round_path / "B" / "PREDICTION_B.json"
    _write_once(receipt_path, prediction_receipt)
    files = [item["path"] for item in shard_inventory] + [str(receipt_path.resolve())]
    frozen = freeze_predictions(
        round_path, "B", files,
        {
            "config_sha256": config_sha,
            "prediction_receipt_sha256": sha256_file(receipt_path),
            "A_freeze_sha256": a_frozen["record_sha256"],
            "ground_truth_opened_before_freeze": False,
            "patients": prediction_receipt["coverage"]["predicted_patients"],
            "frames": frame_count,
            "shards": len(shard_inventory),
        },
    )
    return {
        "schema": PREDICTION_SCHEMA,
        "status": "PREDICTIONS_FROZEN",
        "prediction_receipt": str(receipt_path.resolve()),
        "prediction_receipt_sha256": sha256_file(receipt_path),
        "freeze_sha256": frozen["record_sha256"],
        "shards": len(shard_inventory),
        "frames": frame_count,
        "D0_parity": parity,
        "h_dependence": dependence,
        "cost": cost,
    }


def predict_stage(config_path: str | Path, round_dir: str | Path, *, expected_config_sha256: str | None = None) -> dict[str, Any]:
    config, _ = load_config(config_path)
    from .compact_runtime import deterministic_runtime

    with deterministic_runtime(config["device"] == "cuda"):
        return _predict_stage_impl(config_path, round_dir, expected_config_sha256=expected_config_sha256)


def _verify_prediction_inventory(config: Mapping[str, Any], origin: Path, round_path: Path, lock: Mapping[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    from .diagnostic_protocol import read_lock, verify_frozen
    from .compact_inference import load_native_manifest

    config_sha = _config_sha(origin)
    a_lock = read_lock(round_path, "A")
    a_frozen = verify_frozen(round_path, "A")
    b_lock = read_lock(round_path, "B")
    _assert_a_b_match(a_lock["protocol"], config)
    if (lock["record_sha256"] != b_lock["record_sha256"]
            or b_lock["protocol"].get("config_sha256") != config_sha
            or b_lock["protocol"].get("A_lock_sha256") != a_lock["record_sha256"]
            or b_lock["protocol"].get("A_freeze_sha256") != a_frozen["record_sha256"]):
        raise ValueError("B prediction lock is not bound to the current A lock/freeze/config")
    manifest_path = _resolve(origin.parent, config["val_manifest"]["path"], "val_manifest.path")
    val_data, _ = load_native_manifest(manifest_path, config["val_manifest"]["sha256"], expected_roles=["development"])
    receipt_path = round_path / "B" / "PREDICTION_B.json"
    receipt = _read_object(receipt_path)
    if (receipt.get("schema") != PREDICTION_SCHEMA or receipt.get("stage") != "B"
            or receipt.get("config_sha256") != config_sha
            or receipt.get("B_lock_sha256") != b_lock["record_sha256"]
            or receipt.get("A_lock_sha256") != a_lock["record_sha256"]
            or receipt.get("A_freeze_sha256") != a_frozen["record_sha256"]
            or receipt.get("val_manifest_sha256") != config["val_manifest"]["sha256"]
            or receipt.get("ground_truth_payloads_opened") is not False):
        raise ValueError("B prediction receipt does not match the locked image-only run")
    expected_rhos = [float(value) for value in config["directions"]["rho_sensitivity"]]
    expected_patients = [str(item["id"]) for item in val_data["patients"]]
    expected_frames = {(str(item["id"]), str(frame)) for item in val_data["patients"] for frame in item["masks"]}
    from .compact_inference import load_patient_cine
    manifest_dir = manifest_path.parent
    shape_by_patient = {}
    for record in val_data["patients"]:
        cine, _, _ = load_patient_cine(record, manifest_dir, normalization=config["normalization"])
        shape_by_patient[str(record["id"])] = [int(value) for value in cine.shape[1:]]
        del cine
    if receipt.get("rho_values") != expected_rhos or receipt.get("primary_rho") != float(config["directions"]["rho_primary"]):
        raise ValueError("prediction rho inventory differs from the locked config")
    if receipt.get("coverage", {}).get("expected_patients") != expected_patients:
        raise ValueError("prediction patient inventory differs from the locked cohort")
    shards: dict[tuple[str, str], dict[str, Any]] = {}
    frozen = verify_frozen(round_path, "B")
    frozen_paths = {item["path"] for item in frozen["files"]}
    expected_files = {str(receipt_path.resolve())}
    output_root = _prediction_dir(round_path).resolve()
    image_sha = {str(item["id"]): item["image_sha256"] for item in val_data["patients"]}
    for item in receipt.get("shards", []):
        if not isinstance(item, Mapping):
            raise ValueError("prediction shard inventory entry must be an object")
        pid, frame = str(item.get("patient_id")), str(item.get("frame_id"))
        key = (pid, frame)
        if key in shards or key not in expected_frames:
            raise ValueError("prediction shard inventory contains duplicate or unexpected patient/frame")
        path = Path(item.get("path", "")).resolve()
        if not path.is_relative_to(output_root) or not path.is_file():
            raise ValueError("prediction shard path escapes the locked output directory")
        expected_shape = item.get("shape")
        if (not isinstance(expected_shape, list) or len(expected_shape) != 6
                or expected_shape[:2] != [len(expected_rhos), len(ARMS)]):
            raise ValueError("prediction shard shape metadata is malformed")
        if item.get("image_sha256") != image_sha[pid]:
            raise ValueError("prediction shard image identity differs from the evaluation manifest")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != expected_shape or array.dtype != np.float32:
            raise ValueError("frozen logit shard shape/dtype mismatch")
        if expected_shape[2] != 4 or expected_shape[3:] != shape_by_patient[pid]:
            raise ValueError("frozen logit shard class metadata is malformed")
        if item.get("rho_values") != expected_rhos:
            raise ValueError("shard sensitivity rho list differs from the locked config")
        for rho_index in range(array.shape[0]):
            for z_start in range(0, array.shape[3], max(1, config["batch_size"])):
                chunk = np.asarray(array[rho_index, :, :, z_start:z_start + config["batch_size"]])
                if not np.isfinite(chunk).all():
                    raise ValueError("frozen logit shard contains NaN or Inf")
        shards[key] = dict(item)
        expected_files.add(str(path))
    if set(shards) != expected_frames:
        raise ValueError("frozen prediction coverage is incomplete for the locked patient/frame cohort")
    if set(receipt.get("coverage", {}).get("predicted_patients", [])) != set(expected_patients):
        raise ValueError("prediction receipt patient coverage is incomplete")
    if frozen_paths != expected_files:
        raise ValueError("shared B freeze contains missing or untracked prediction outputs")
    if receipt.get("D0_parity", {}).get("mismatches") != 0 or receipt.get("D0_parity", {}).get("slices", 0) <= 0:
        raise ValueError("frozen B predictions lack successful D0 parity")
    return receipt, shards, val_data


def _load_gt_mask(mask_record: Mapping[str, Any], manifest_dir: Path, patient_id: str, frame_id: str) -> np.ndarray:
    path = _resolve(manifest_dir, mask_record.get("path"), f"GT mask {patient_id}/{frame_id}")
    if not path.is_file():
        raise FileNotFoundError(f"reference mask is missing for {patient_id}/{frame_id}")
    expected = _require_sha(mask_record.get("sha256"), f"GT mask {patient_id}/{frame_id} SHA-256")
    if sha256_file(path) != expected:
        raise ValueError(f"reference mask hash mismatch for {patient_id}/{frame_id}")
    if path.suffix != ".npy":
        raise ValueError("the compact native evaluator supports NPY reference masks only")
    truth = np.load(path, mmap_mode="r", allow_pickle=False)
    if not np.issubdtype(truth.dtype, np.integer):
        raise ValueError(f"reference mask must have integer dtype for {patient_id}/{frame_id}")
    return truth


def _write_csv_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty diagnostic CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _class_rows(patient_rows: Sequence[Mapping[str, Any]], rho: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for patient in patient_rows:
        for arm in ARMS:
            details = patient["arms"][arm]
            for class_name in CLASS_NAMES:
                counts = details["class_counts"][class_name]
                absent = details["absent_class_fp"][class_name]
                rows.append({
                    "patient_id": patient["patient_id"],
                    "rho": rho,
                    "arm": arm,
                    "class_name": class_name,
                    "dice": details["class_dice"][class_name],
                    "reference_count": counts["reference_count"],
                    "prediction_count": counts["prediction_count"],
                    "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
                    "recall": counts["tp"] / counts["reference_count"] if counts["reference_count"] else None,
                    "precision": counts["tp"] / counts["prediction_count"] if counts["prediction_count"] else None,
                    "absent_class_fp": absent["false_positive_voxels"],
                    "absent_valid_voxels": absent["absent_valid_voxels"],
                    "absent_class_fp_rate": absent["rate"],
                    "fallback_any": details["fallback_any"],
                })
    return rows


def score_stage(config_path: str | Path, round_dir: str | Path, *, expected_config_sha256: str | None = None) -> dict[str, Any]:
    """Authorize the shared A/B round before opening any VAL mask payload."""
    from .compact_inference import load_patient_cine
    from .diagnostic_protocol import authorize_score, read_lock, verify_frozen
    from .gradient_bridge_metrics import GradientBridgeFrameAccumulator, aggregate_patient_rows, evaluate_paired_gate, summarize_patient_rows

    config, origin = load_config(config_path)
    config_sha = _config_sha(origin, expected_config_sha256)
    round_path = Path(round_dir).resolve()
    preflight_record = _read_preflight(round_path, origin, config_sha)
    lock = read_lock(round_path, "B")
    fit_record = _read_object(round_path / "FIT_P.json")
    alignment_path = _alignment_path(config, round_path)
    alignment = _load_alignment(config, round_path)
    if (fit_record.get("config_sha256") != config_sha
            or fit_record.get("artifact_sha256") != sha256_file(alignment_path)
            or fit_record.get("artifact_identity") != alignment.artifact_sha256
            or lock["protocol"].get("alignment_file_sha256") != sha256_file(alignment_path)):
        raise ValueError("scoring P artifact/fit receipt differs from the B lock")
    # This validates the complete frozen inventory and scans logits for finite
    # values. No GT paths are opened or hashed before the shared authorization.
    receipt, shards, val_data = _verify_prediction_inventory(config, origin, round_path, lock)
    frozen_a = verify_frozen(round_path, "A")
    if receipt["A_freeze_sha256"] != frozen_a["record_sha256"]:
        raise ValueError("B prediction receipt refers to another A freeze")
    # read_lock/verify_frozen above revalidate the complete locked asset and
    # output sets. Do not rerun preflight here: the authorized A score may
    # already have written VAL_EXPOSURE for this same B round.
    exposure = authorize_score(round_path, "B")

    manifest_path = _resolve(origin.parent, config["val_manifest"]["path"], "val_manifest.path")
    rhos = [float(value) for value in config["directions"]["rho_sensitivity"]]
    rho_scores: dict[str, Any] = {}
    all_frames_by_rho: dict[str, list[dict[str, Any]]] = {str(rho): [] for rho in rhos}
    for patient in val_data["patients"]:
        pid = str(patient["id"])
        cine, valid_volume, _ = load_patient_cine(patient, manifest_path.parent, normalization=config["normalization"])
        z_count, height, width = (int(value) for value in cine.shape[1:])
        for frame_text, mask_record in sorted(patient["masks"].items(), key=lambda pair: int(pair[0])):
            frame_id = str(frame_text)
            gt = _load_gt_mask(mask_record, manifest_path.parent, pid, frame_id)
            if tuple(gt.shape) != (z_count, height, width):
                raise ValueError(f"GT shape {gt.shape} does not match native image frame {(z_count, height, width)}")
            key = (pid, frame_id)
            item = shards[key]
            shard = np.load(item["path"], mmap_mode="r", allow_pickle=False)
            valid = None if valid_volume is None else valid_volume[int(frame_id)]
            for rho_index, rho in enumerate(rhos):
                fallback = item["fallback_by_rho"][rho_index]
                accumulator = GradientBridgeFrameAccumulator(
                    pid, frame_id, frame_shape=(z_count, height, width), fallback_by_arm=fallback,
                )
                for z_start in range(0, z_count, config["batch_size"]):
                    z_end = min(z_count, z_start + config["batch_size"])
                    truth_chunk = np.asarray(gt[z_start:z_end])
                    if truth_chunk.size and not np.isin(truth_chunk, [0, 1, 2, 3, IGNORE_INDEX]).all():
                        raise ValueError(f"reference mask has invalid class values for {pid}/{frame_id}")
                    logits_by_arm = {
                        arm: np.asarray(shard[rho_index, arm_index, :, z_start:z_end], dtype=np.float32)
                        for arm_index, arm in enumerate(ARMS)
                    }
                    valid_chunk = None if valid is None else np.ascontiguousarray(valid[z_start:z_end])
                    accumulator.update_chunk(
                        z_start,
                        truth_chunk,
                        logits_by_arm,
                        valid_chunk,
                    )
                all_frames_by_rho[str(rho)].append(accumulator.finalize())
            del shard, gt
        del cine

    primary_rho = float(config["directions"]["rho_primary"])
    mechanism = receipt.get("h_dependence", {})
    integrity_errors = []
    if mechanism.get("status") != "ACTIVE_H_DEPENDENCE":
        integrity_errors.append(f"bridge h-dependence status: {mechanism.get('status', 'missing')}")
    for rho in rhos:
        key = str(rho)
        patient_rows = aggregate_patient_rows(all_frames_by_rho[key])
        gate = evaluate_paired_gate(
            patient_rows, config["gate"],
            integrity_errors=integrity_errors if rho == primary_rho else (),
        ) if rho == primary_rho else None
        rho_scores[key] = {
            "rho": rho,
            "is_primary": rho == primary_rho,
            "frame_records": all_frames_by_rho[key],
            "patient_rows": patient_rows,
            "summary": summarize_patient_rows(patient_rows),
            "gate": gate,
        }

    primary_gate = rho_scores[str(primary_rho)]["gate"]
    if config["synthetic_fixture"]:
        status = "FIXTURE_ONLY"
        interpretation = "Synthetic fixture software path only; metrics are excluded from patient reports and scientific decisions."
    else:
        status = primary_gate["status"]
        interpretation = "Descriptive development-set diagnostic; this result is not confirmation or evidence of clinical utility."
    result = {
        "schema": SCORE_SCHEMA,
        "stage": "B",
        "status": status,
        "synthetic_fixture": config["synthetic_fixture"],
        "interpretation": interpretation,
        "config_sha256": config_sha,
        "B_lock_sha256": lock["record_sha256"],
        "A_freeze_sha256": frozen_a["record_sha256"],
        "B_prediction_freeze_sha256": verify_frozen(round_path, "B")["record_sha256"],
        "exposure_ledger_sha256": sha256_file(round_path / "VAL_EXPOSURE.json"),
        "teacher_manifest_sha256": config["teacher"]["manifest_sha256"],
        "teacher_payload_sha256": config["teacher"]["payload_sha256"],
        "student_manifest_sha256": config["student"]["manifest_sha256"],
        "student_payload_sha256": config["student"]["payload_sha256"],
        "teacher_runtime_source_sha256": config["teacher"]["runtime_source_sha256"],
        "student_runtime_source_sha256": config["student"]["runtime_source_sha256"],
        "train_manifest_sha256": config["train_manifest"]["sha256"],
        "val_manifest_sha256": config["val_manifest"]["sha256"],
        "alignment_artifact_sha256": alignment.artifact_sha256,
        "rho_primary": primary_rho,
        "rho_sensitivity": rhos,
        "arms": list(ARMS),
        "execution": lock["protocol"]["execution"],
        "prediction_cost": receipt.get("cost"),
        "fit_P_elapsed_seconds": fit_record.get("elapsed_seconds_including_load_and_fit"),
        "ground_truth_opened_after_authorization": True,
        "prediction_receipt_sha256": sha256_file(round_path / "B" / "PREDICTION_B.json"),
        "mechanism_check": mechanism,
        "exposure_ledger": exposure,
    }
    stage_dir = round_path / "B"
    if config["synthetic_fixture"]:
        result["synthetic_metrics_not_for_patient_reporting"] = True
        result["fixture_gate_status"] = primary_gate["status"]
        result["fixture_rho_summaries"] = {
            key: {"rho": value["rho"], "is_primary": value["is_primary"], "summary": value["summary"]}
            for key, value in rho_scores.items()
        }
    else:
        result["rho_scores"] = rho_scores
        primary_rows = rho_scores[str(primary_rho)]["patient_rows"]
        _write_once(stage_dir / "FRAME_RECORDS_B.json", all_frames_by_rho[str(primary_rho)])
        _write_once(stage_dir / "PATIENT_ROWS_B.json", primary_rows)
        _write_csv_once(stage_dir / "PATIENT_CLASS_DICE_B.csv", _class_rows(primary_rows, primary_rho))
        result["patient_csv"] = str((stage_dir / "PATIENT_CLASS_DICE_B.csv").resolve())
    _write_once(stage_dir / "SCORES_B.json", result)
    return result


def report_stage(round_dir: str | Path) -> dict[str, Any]:
    """Write a compact round report from the frozen A and B score receipts."""
    round_path = Path(round_dir).resolve()
    from .diagnostic_protocol import read_lock, verify_frozen

    a_lock = read_lock(round_path, "A")
    a_frozen = verify_frozen(round_path, "A")
    a_result_path = round_path / "A" / "RESULT.json"
    a_result = _read_object(a_result_path) if a_result_path.is_file() else None
    b_lock_path = round_path / "protocol-B.lock.json"
    b_lock = read_lock(round_path, "B") if b_lock_path.is_file() else None
    b_frozen_path = round_path / "predictions-B.frozen.json"
    b_frozen = verify_frozen(round_path, "B") if b_frozen_path.is_file() else None
    b_result_path = round_path / "B" / "SCORES_B.json"
    b_result = _read_object(b_result_path) if b_result_path.is_file() else None
    synthetic = bool(a_lock["protocol"].get("config", {}).get("synthetic_fixture", False))
    for result in (a_result, b_result):
        if result is not None and bool(result.get("synthetic_fixture", False)) != synthetic:
            raise ValueError("A/B score fixture identity disagrees with the A lock")
    if synthetic:
        overall = "FIXTURE_ONLY"
    elif b_result is None:
        ledger_path = round_path / "VAL_EXPOSURE.json"
        ledger = _read_object(ledger_path) if ledger_path.exists() else {}
        overall = "BLOCKED" if ledger.get("B_blocked_reason") else "NOT_RUN"
    else:
        # A is reported independently. An A-negative or reproduction mismatch
        # is not a gate on B's separately specified mechanism comparison.
        overall = b_result["status"]
    report = {
        "schema": REPORT_SCHEMA,
        "status": overall,
        "round_dir": str(round_path),
        "synthetic_fixture": synthetic,
        "stages": {
            "A": {"lock_sha256": a_lock["record_sha256"], "freeze_sha256": a_frozen["record_sha256"],
                  "score_status": None if a_result is None else a_result.get("status")},
            "B": {"lock_sha256": None if b_lock is None else b_lock["record_sha256"],
                  "freeze_sha256": None if b_frozen is None else b_frozen["record_sha256"],
                  "score_status": None if b_result is None else b_result.get("status")},
        },
        "A_result": a_result,
        "B_result": b_result,
        "patient_results_included": bool(not synthetic and (a_result is not None or b_result is not None)),
        "limitations": [
            "The development cohort is exposed descriptive evidence, not confirmation.",
            "Synthetic fixture metrics are never emitted as patient Dice tables.",
            "Patient, CUDA, clinical, and production success must be reported only from actually completed runs.",
        ],
    }
    report_path = round_path / "REPORT.json"
    markdown_path = round_path / "REPORT.md"
    _write_once(report_path, report)
    gate_status = "NOT_RUN" if b_result is None else b_result.get("status", "UNKNOWN")
    text = "\n".join([
        "# Context and Gradient Bridge Diagnostic",
        "",
        f"- Status: {overall}",
        f"- Diagnostic A score: {(a_result or {}).get('status', 'NOT_RUN')}",
        f"- Diagnostic B score: {gate_status}",
        f"- A lock SHA-256: {a_lock['record_sha256']}",
        f"- A prediction freeze SHA-256: {a_frozen['record_sha256']}",
        f"- B lock SHA-256: {(b_lock or {}).get('record_sha256', 'NOT_RUN')}",
        f"- B prediction freeze SHA-256: {(b_frozen or {}).get('record_sha256', 'NOT_RUN')}",
        f"- Synthetic fixture: {synthetic}",
        "- Context results: [A/RESULT.json](A/RESULT.json)",
        "- Gradient results and paired bootstrap/LOO: [B/SCORES_B.json](B/SCORES_B.json)",
        "",
        "## Evidence limits",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
    ])
    if markdown_path.exists():
        raise FileExistsError("REPORT.md is append-once")
    markdown_path.write_text(text, encoding="utf-8")
    return report
