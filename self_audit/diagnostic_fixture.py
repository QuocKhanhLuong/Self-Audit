"""Deterministic, synthetic-only inputs for the context and bridge diagnostics.

The fixture uses the repository's real NativeStudent(tiny_test) and
CompactStudent architectures.  It writes ordinary, hash-bound checkpoint and
export artifacts, then relies on the production strict loaders to reconstruct
both models.  No real profile or historical asset gate is relaxed here.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
_DEV_IDS = ("fixture_dev_0", "fixture_dev_1")
_TRAIN_L_IDS = ("fixture_train_l0",)
_TRAIN_U_IDS = ("fixture_train_u0",)
_IMAGE_SHAPE = (1, 3, 32, 32)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _save_array(path: Path, value: np.ndarray) -> str:
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    return _sha256(path)


def _fixture_image(patient_index: int) -> np.ndarray:
    yy, xx = np.indices((32, 32), dtype=np.float32)
    slices = [((xx * 3.0 + yy * 5.0 + patient_index * 7.0 + z * 3.0) % 32.0) / 31.0
              for z in range(_IMAGE_SHAPE[1])]
    return np.stack(slices).astype(np.float32, copy=False)[None]


def _fixture_mask(patient_index: int) -> np.ndarray:
    yy, xx = np.indices((32, 32), dtype=np.int16)
    return np.stack([((xx // 8 + yy // 8 + patient_index + z) % 4).astype(np.uint8)
                     for z in range(_IMAGE_SHAPE[1])])


def _write_data_manifests(root: Path) -> tuple[Path, str, Path, str]:
    train_dir = root / "train"
    val_dir = root / "development"
    train_dir.mkdir()
    val_dir.mkdir()

    labeled_image = _fixture_image(10)
    labeled_image_path = train_dir / "fixture_train_l0.npy"
    labeled_image_sha = _save_array(labeled_image_path, labeled_image)
    labeled_mask_path = train_dir / "fixture_train_l0_mask.npy"
    labeled_mask_sha = _save_array(labeled_mask_path, _fixture_mask(10))
    labeled = {
        "id": _TRAIN_L_IDS[0],
        "image": labeled_image_path.name,
        "image_sha256": labeled_image_sha,
        "masks": {"0": {"path": labeled_mask_path.name, "sha256": labeled_mask_sha}},
    }

    unlabeled_image_path = train_dir / "fixture_train_u0.npy"
    unlabeled_image_sha = _save_array(unlabeled_image_path, _fixture_image(11))
    unlabeled = {
        "id": _TRAIN_U_IDS[0],
        "image": unlabeled_image_path.name,
        "image_sha256": unlabeled_image_sha,
    }
    train_manifest_path = train_dir / "train.json"
    _write_json(train_manifest_path, {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": True,
        "label_budget": 1,
        "labeled": [labeled],
        "unlabeled": [unlabeled],
        "evaluation_ids": [],
    })
    train_sha = _sha256(train_manifest_path)

    patients = []
    for index, patient_id in enumerate(_DEV_IDS):
        image_path = val_dir / f"{patient_id}.npy"
        image_sha = _save_array(image_path, _fixture_image(index))
        mask_path = val_dir / f"{patient_id}_mask.npy"
        mask_sha = _save_array(mask_path, _fixture_mask(index))
        patients.append({
            "id": patient_id,
            "image": image_path.name,
            "image_sha256": image_sha,
            "axes": ["time", "slice", "height", "width"],
            "affine": np.eye(4, dtype=np.float64).tolist(),
            "masks": {"0": {"path": mask_path.name, "sha256": mask_sha}},
        })
    val_manifest_path = val_dir / "development.json"
    _write_json(val_manifest_path, {
        "schema": "self_audit.data.v2",
        "role": "development",
        "synthetic": True,
        "patients": patients,
    })
    return train_manifest_path, train_sha, val_manifest_path, _sha256(val_manifest_path)


def _write_model_artifacts(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    from .compact_artifacts import (
        _canonical_preprocessing_spec,
        load_compact_export,
        runtime_model_file_sha256,
    )
    from .compact_model import build_compact_model, validate_architecture
    from .engine import load_checkpoint, source_digest
    from .model import build_model

    source_sha = source_digest()
    model_specs: dict[str, dict[str, Any]] = {}
    teacher_dir = root / "teacher"
    student_dir = root / "student"
    teacher_dir.mkdir()
    student_dir.mkdir()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(18112026)
        teacher, teacher_provenance = build_model({"backbone": "tiny_test"}, synthetic=True)
        if teacher_provenance.get("synthetic") is not True:
            raise ValueError("tiny_test teacher construction did not report synthetic identity")
        teacher_weights = teacher_dir / "weights.pt"
        torch.save({"student": {key: value.detach().cpu() for key, value in teacher.state_dict().items()}}, teacher_weights)
        teacher_payload_sha = _sha256(teacher_weights)
        teacher_manifest_path = teacher_dir / "MANIFEST.json"
        _write_json(teacher_manifest_path, {
            "schema": "self_audit.checkpoint.v2",
            "model_class": "NativeStudent",
            "config": {"model": {"backbone": "tiny_test"}},
            "source_sha256": source_sha,
            "payload_sha256": teacher_payload_sha,
            "synthetic": True,
            "quality": "NOT_EVALUATED",
            "prediction_model": "student",
            "fixture_only": True,
            "training_status": "UNTRAINED_RANDOM_INITIALIZATION",
        })
        teacher_manifest_sha = _sha256(teacher_manifest_path)
        loaded_teacher, teacher_meta = load_checkpoint(teacher_dir, teacher_manifest_sha, device="cpu")
        if type(teacher_meta.get("synthetic")) is not bool or teacher_meta["synthetic"] is not True:
            raise ValueError("strict NativeStudent loader did not preserve synthetic=true")
        if loaded_teacher.__class__.__name__ != "NativeStudent" or loaded_teacher.encoder.__class__.__name__ != "TinyEncoder":
            raise ValueError("fixture teacher did not load as the actual tiny NativeStudent architecture")

        torch.manual_seed(18112027)
        architecture = validate_architecture({
            "name": "CompactStudent",
            "widths": [16, 32, 64, 128],
            "in_channels": 3,
            "num_classes": 4,
        })
        student = build_compact_model(architecture)
        student_weights = student_dir / "weights.pt"
        student_state = {key: value.detach().cpu() for key, value in student.state_dict().items()}
        torch.save({"student": student_state}, student_weights)
        student_payload_sha = _sha256(student_weights)
        student_manifest_path = student_dir / "MANIFEST.json"
        parameter_count = sum(parameter.numel() for parameter in student.parameters())
        weights_only_bytes = sum(parameter.numel() * parameter.element_size() for parameter in student.parameters())
        _write_json(student_manifest_path, {
            "schema": "self_audit.compact.export.v1",
            "model_class": "CompactStudent",
            "architecture": architecture,
            "config": {},
            "preprocessing": _canonical_preprocessing_spec(),
            "class_map": {
                "0": "background",
                "1": "right_ventricle",
                "2": "myocardium",
                "3": "left_ventricle",
            },
            "native_axes": {
                "slice_batch_tensor": ["B", "C", "H", "W"],
                "patient_volume_cine": ["T", "Z", "Y", "X"],
            },
            "lineage": {
                # The fixture networks are initialized for software-path
                # validation only; they are not trained on these images.
                "lids": [],
                "uids": [],
                "label_budget": 0,
                "data_sha256": hashlib.sha256(b"synthetic-untrained-fixture").hexdigest(),
                "synthetic": True,
                "teacher_identity": None,
                "training_status": "UNTRAINED_RANDOM_INITIALIZATION",
            },
            "synthetic": True,
            "source_sha256": source_sha,
            "training_source_sha256": source_sha,
            "model_file_sha256": runtime_model_file_sha256(),
            "payload_sha256": student_payload_sha,
            "weights_only_bytes": weights_only_bytes,
            "serialization_bytes": student_weights.stat().st_size,
            "parameters_total": parameter_count,
            "parameters_trainable": sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad),
            "teacher_included": False,
            "external_teacher_required": False,
            "teacher_memory_included": False,
            "dino_required": False,
            "pretrained_weights_required": False,
        })
        student_manifest_sha = _sha256(student_manifest_path)
        loaded_student, student_meta = load_compact_export(student_dir, student_manifest_sha, device="cpu")
        if student_meta.get("synthetic") is not True:
            raise ValueError("strict CompactStudent loader did not preserve synthetic=true")
        if loaded_student.__class__.__name__ != "CompactStudent":
            raise ValueError("fixture student did not load as the actual CompactStudent architecture")

    common = {
        "runtime_root": str(ROOT.resolve()),
        "runtime_source_sha256": source_sha,
    }
    model_specs["teacher"] = {
        **common,
        "path": str(teacher_dir.resolve()),
        "manifest_sha256": teacher_manifest_sha,
        "payload_sha256": teacher_payload_sha,
    }
    model_specs["student"] = {
        **common,
        "path": str(student_dir.resolve()),
        "manifest_sha256": student_manifest_sha,
        "payload_sha256": student_payload_sha,
    }
    return model_specs["teacher"], model_specs["student"], source_sha


def build_diagnostic_fixture(root: str | Path) -> dict[str, Any]:
    """Write a bounded shared A/B synthetic fixture in a new directory.

    The fixture contains one TRAIN-L and one TRAIN-U patient for deterministic
    fit-P, plus two same-geometry development patients for cross-patient tests.
    It returns both complete configs so A and B consume identical images,
    checkpoints, manifests, and protocol identities.
    """
    fixture_root = Path(root).resolve()
    if fixture_root.exists():
        raise FileExistsError(f"diagnostic fixture directory must be new: {fixture_root}")
    fixture_root.mkdir(parents=True)
    train_path, train_sha, val_path, val_sha = _write_data_manifests(fixture_root)
    teacher_spec, student_spec, source_sha = _write_model_artifacts(fixture_root)
    common = {
        "teacher": teacher_spec,
        "student": student_spec,
        "train_manifest": {"path": str(train_path.resolve()), "sha256": train_sha},
        "val_manifest": {"path": str(val_path.resolve()), "sha256": val_sha},
    }

    context_config_path = fixture_root / "context_config.json"
    context_config = {
        "schema": "self_audit.context_diagnostic.v1",
        "synthetic_fixture": True,
        "lambdas": [0.0, 0.5, 1.0],
        "device": "cpu",
        "threads": 1,
        "parity": {"logit_atol": 1e-6, "logit_rtol": 1e-5},
        "val_manifest": common["val_manifest"],
        "teacher": teacher_spec,
        "student": student_spec,
    }
    _write_json(context_config_path, context_config)

    bridge_config_path = fixture_root / "gradient_bridge_config.json"
    bridge_config = {
        "schema": "self_audit.gradient_bridge.runtime.v1",
        "profile": "fixture",
        "synthetic_fixture": True,
        "device": "cpu",
        "threads": 1,
        "batch_size": 2,
        "normalization": "prepared",
        **common,
        "parity": {"logit_atol": 1e-6, "logit_rtol": 1e-5},
        "fit_p": {
            "seed": 41,
            "max_source_ids": 2,
            "max_train_patients": 2,
            "max_pixels": 2048,
            "ridge": 1e-6,
            "fit_loss": "patient_balanced_pre_refine0_feature_mse",
            "alignment_mode": "bilinear_align_corners_false_then_1x1_affine",
            "output_file": "P_ALIGNMENT.json",
        },
        "kd": {
            "temperature": 1.0,
            "coefficient": 1.0,
            "confidence": 0.0,
            "class_weights": None,
            "denominator": "valid_pixel_count",
        },
        "directions": {
            "rho_primary": 0.1,
            "rho_sensitivity": [0.1],
            "norm_epsilon": 1e-8,
            "shuffle_seed": 43,
        },
        "gate": {
            "schema": "self_audit.gradient_bridge_gate.v1",
            "minimum_fg_absolute_delta": 0.0,
            "bootstrap": {"n_resamples": 20, "seed": 47},
            "confidence_rule": {
                "method": "paired_percentile",
                "level": 0.95,
                "minimum_defined_fraction": 1.0,
                "acceptance": "lower_bound_strictly_above_minimum",
            },
            "loo_rule": {"method": "all_deltas_at_least_minimum", "require_all_defined": True},
        },
    }
    _write_json(bridge_config_path, bridge_config)

    # Exercise both schema validators at fixture construction time. Runtime
    # prediction still goes through each CLI's pinned strict loader path.
    from scripts.context_prediction_worker import _validate_config as validate_context_config
    from .gradient_bridge_runtime import validate_config as validate_bridge_config

    validate_context_config(context_config)
    validate_bridge_config(bridge_config)
    return {
        "root": str(fixture_root),
        "context_config_path": str(context_config_path.resolve()),
        "context_config_sha256": _sha256(context_config_path),
        "gradient_bridge_config_path": str(bridge_config_path.resolve()),
        "gradient_bridge_config_sha256": _sha256(bridge_config_path),
        "train_manifest_path": str(train_path.resolve()),
        "train_manifest_sha256": train_sha,
        "val_manifest_path": str(val_path.resolve()),
        "val_manifest_sha256": val_sha,
        "teacher_spec": copy.deepcopy(teacher_spec),
        "student_spec": copy.deepcopy(student_spec),
        "model_specs": {"teacher": copy.deepcopy(teacher_spec), "student": copy.deepcopy(student_spec)},
        "runtime_root": str(ROOT.resolve()),
        "runtime_source_sha256": source_sha,
        "train_patient_ids": [*_TRAIN_L_IDS, *_TRAIN_U_IDS],
        "development_patient_ids": list(_DEV_IDS),
        "context_config": context_config,
        "gradient_bridge_config": bridge_config,
    }


def run_fixture_smoke(round_dir: str | Path) -> dict[str, Any]:
    """Exercise actual CLI processes in the prescribed order, using no patient assets."""
    import subprocess
    import sys
    import time

    round_path = Path(round_dir).resolve()
    asset_root = round_path.with_name(round_path.name + "_fixture_inputs")
    log_path = round_path.with_name(round_path.name + "_cli_smoke.json")
    if round_path.exists() or asset_root.exists() or log_path.exists():
        raise FileExistsError("fixture smoke requires a new round and sibling asset/log paths")
    if round_path.is_relative_to(ROOT) or ROOT.is_relative_to(round_path):
        raise ValueError("fixture round must be outside the source checkout")
    round_path.parent.mkdir(parents=True, exist_ok=True)
    fixture = build_diagnostic_fixture(asset_root)
    commands = []
    for entrypoint, stage, config_key, sha_key in (
        ("run_context_diagnostic.py", "preflight", "context_config_path", "context_config_sha256"),
        ("run_context_diagnostic.py", "lock", "context_config_path", "context_config_sha256"),
        ("run_context_diagnostic.py", "predict", "context_config_path", "context_config_sha256"),
        ("run_gradient_bridge_diagnostic.py", "preflight", "gradient_bridge_config_path", "gradient_bridge_config_sha256"),
        ("run_gradient_bridge_diagnostic.py", "fit-p", "gradient_bridge_config_path", "gradient_bridge_config_sha256"),
        ("run_gradient_bridge_diagnostic.py", "lock-b", "gradient_bridge_config_path", "gradient_bridge_config_sha256"),
        ("run_gradient_bridge_diagnostic.py", "predict", "gradient_bridge_config_path", "gradient_bridge_config_sha256"),
        ("run_context_diagnostic.py", "score", "context_config_path", "context_config_sha256"),
        ("run_gradient_bridge_diagnostic.py", "score", "gradient_bridge_config_path", "gradient_bridge_config_sha256"),
    ):
        argv = [sys.executable, str(ROOT / "scripts" / entrypoint), stage,
                "--config", fixture[config_key], "--config-sha256", fixture[sha_key],
                "--round-dir", str(round_path)]
        if entrypoint == "run_context_diagnostic.py" and stage != "preflight":
            argv += ["--stage", "A"]
        commands.append(argv)
    commands.append([sys.executable, str(ROOT / "scripts/run_gradient_bridge_diagnostic.py"),
                     "report", "--round-dir", str(round_path)])
    records = []
    for argv in commands:
        started = time.perf_counter()
        completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=120)
        records.append({"argv": argv, "returncode": completed.returncode,
                        "elapsed_seconds": time.perf_counter() - started,
                        "stdout": completed.stdout, "stderr": completed.stderr})
        _write_json(log_path, {"synthetic_fixture": True, "fixture": fixture, "commands": records})
        if completed.returncode:
            raise RuntimeError(f"fixture command failed; inspect {log_path}: {completed.stdout} {completed.stderr}")
    report = json.loads((round_path / "REPORT.json").read_text())
    if report["status"] != "FIXTURE_ONLY" or report["patient_results_included"]:
        raise ValueError("fixture report must remain explicitly synthetic")
    return {"status": "FIXTURE_ONLY", "patient_experiment_run": False,
            "round_dir": str(round_path), "commands_passed": len(records),
            "cli_log": str(log_path), "cli_log_sha256": _sha256(log_path),
            "report_sha256": _sha256(round_path / "REPORT.json"),
            "fixture": fixture}
