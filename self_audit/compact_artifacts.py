"""Artifact persistence and standalone export for CompactStudent.

Provides hash-pinned checkpointing and strict standalone export.
Standalone exports require only package + torch/numpy; no source weights,
teacher, cache, support banks or DINO dependencies.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
import torch
from torch import nn

from .compact_model import CompactStudent, build_compact_model, validate_architecture

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _require_sha256(value: Any, name: str) -> str:
    """Validate that value is a non-null 64-character hexadecimal SHA-256 string."""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a mandatory 64-hex SHA-256 pin, got {value!r}")
    return value.lower()


def digest(path: str | Path) -> str:
    """Compute SHA-256 digest of a file in 1MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runtime_model_file_sha256() -> str:
    """Compute SHA-256 digest of the runtime model definition file (compact_model.py)."""
    model_file = Path(__file__).resolve().parent / "compact_model.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"runtime model file not found: {model_file}")
    return digest(model_file)


def compact_training_source_sha256() -> str:
    """Compute deterministic SHA-256 digest of training source files using canonical engine if available."""
    try:
        from .engine import source_digest as _canonical_source_digest
        return _canonical_source_digest()
    except (ImportError, AttributeError):
        root = Path(__file__).resolve().parent.parent
        files = sorted((root / "self_audit").glob("*.py")) + [root / "train.py"]
        return hashlib.sha256("".join(f"{p.name}:{digest(p)}\n" for p in files if p.is_file()).encode()).hexdigest()


# Backwards compatibility alias
source_digest = compact_training_source_sha256


def _extract_strict_data_provenance(data: Any) -> dict[str, Any]:
    """Strictly extract lids, uids, manifest, and data_sha256 ensuring disjoint unique IDs."""
    if data is None:
        raise ValueError("data provenance is mandatory for compact checkpoints; data cannot be None")

    if isinstance(data, Mapping):
        raw_lids = data.get("lids")
        raw_uids = data.get("uids")
        raw_manifest = data.get("manifest")
        path = str(data.get("path")) if data.get("path") is not None else None
        data_sha = data.get("data_sha256")
        data_synthetic = data.get("synthetic")
    else:
        raw_lids = getattr(data, "lids", None)
        raw_uids = getattr(data, "uids", None)
        raw_manifest = getattr(data, "manifest", None)
        path_attr = getattr(data, "path", None)
        path = str(path_attr) if path_attr is not None else None
        data_sha = getattr(data, "data_sha256", None)
        data_synthetic = getattr(data, "synthetic", None)

    if not isinstance(raw_lids, (list, tuple)) or not all(isinstance(x, str) for x in raw_lids):
        raise ValueError(f"data provenance lids must be a list of strings, got {raw_lids!r}")
    if not isinstance(raw_uids, (list, tuple)) or not all(isinstance(x, str) for x in raw_uids):
        raise ValueError(f"data provenance uids must be a list of strings, got {raw_uids!r}")

    # Enforce unique IDs and strict disjointness between labeled and unlabeled
    lids_set = set(raw_lids)
    uids_set = set(raw_uids)
    if len(lids_set) != len(raw_lids):
        raise ValueError(f"duplicate labeled IDs in data provenance: {raw_lids}")
    if len(uids_set) != len(raw_uids):
        raise ValueError(f"duplicate unlabeled IDs in data provenance: {raw_uids}")
    overlap = lids_set & uids_set
    if overlap:
        raise ValueError(f"labeled and unlabeled IDs must be strictly disjoint; overlapping: {sorted(overlap)}")

    manifest = dict(raw_manifest) if isinstance(raw_manifest, Mapping) else {}

    # Synthetic determination: allow explicit verified false if real legacy manifest lacks 'synthetic'
    if "synthetic" in manifest:
        if type(manifest["synthetic"]) is not bool:
            raise ValueError("manifest synthetic must be a boolean")
        synthetic = manifest["synthetic"]
    elif data_synthetic is not None and type(data_synthetic) is bool:
        synthetic = data_synthetic
    else:
        # Legacy canonical manifests omit this field; their validated loader
        # defines omission as real data. An explicit malformed flag never does.
        synthetic = False

    if data_sha is None and path is not None and Path(path).is_file():
        data_sha = digest(path)

    if data_sha is None:
        raise ValueError("data provenance data_sha256 is mandatory and could not be inferred")
    data_sha = _require_sha256(data_sha, "data_provenance.data_sha256")

    return {
        "lids": list(raw_lids),
        "uids": list(raw_uids),
        "label_budget": len(raw_lids),
        "synthetic": synthetic,
        "path": path,
        "data_sha256": data_sha,
        "manifest": manifest,
    }


def _verify_all_tensors_finite(state_dict: Mapping[str, torch.Tensor], context: str) -> None:
    """Verify that every tensor in state_dict contains only finite values."""
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"non-finite values (NaN or Inf) found in {context}: {name}")


def _clean_config_for_export(config: Mapping[str, Any]) -> dict[str, Any]:
    """Clean training asset paths and references to prevent runtime path dereferences."""
    clean = copy.deepcopy(dict(config))
    forbidden_keys = {
        "weights", "weights_sha256", "source_repo", "source_commit",
        "cache_path", "cache_sha256", "bank_path", "bank_sha256",
        "teacher_path", "teacher_sha256", "data", "data_sha256",
        "teacher_memory", "teacher_memory_included",
    }
    for k in list(clean.keys()):
        if k in forbidden_keys:
            clean[k] = None
        elif isinstance(clean[k], Mapping):
            sub = dict(clean[k])
            for sk in list(sub.keys()):
                if sk in forbidden_keys:
                    sub[sk] = None
            clean[k] = sub
    return clean


def _canonical_preprocessing_spec() -> dict[str, Any]:
    """Exact, non-ambiguous image preprocessing and normalization recipes matching inference worker."""
    return {
        "input_channels": 3,
        "context_slices": [-1, 0, 1],
        "context_boundary_mode": "clamp",
        "normalization_recipes": {
            "prepared": {
                "mode": "prepared",
                "value_range": [0.0, 1.0],
                "tolerance": 1e-5,
                "renormalize": False,
                "description": "Validates that values are already in [0, 1] without renormalizing.",
            },
            "raw": {
                "mode": "raw",
                "percentiles": [0.5, 99.5],
                "percentile_scope": "per_cine",
                "axes": ["T", "Z", "Y", "X"],
                "target_range": [0.0, 1.0],
                "epsilon": 1e-6,
                "renormalize": True,
                "description": "Per-cine 0.5% and 99.5% percentile intensity scaling to [0, 1].",
            },
        },
        "default_normalization": "prepared",
    }


def _validate_export_metadata_consistency(manifest: Mapping[str, Any]) -> None:
    """Validate consistency of export preprocessing, class_map, native_axes, and lineage."""
    # 1. Preprocessing validation
    prep = manifest.get("preprocessing")
    if not isinstance(prep, Mapping):
        raise ValueError("export manifest missing valid 'preprocessing' section")
    if prep.get("input_channels") != 3:
        raise ValueError(f"export preprocessing input_channels must be 3, got {prep.get('input_channels')}")
    if prep.get("context_slices") != [-1, 0, 1]:
        raise ValueError(f"export preprocessing context_slices must be [-1, 0, 1], got {prep.get('context_slices')}")
    recipes = prep.get("normalization_recipes")
    if not isinstance(recipes, Mapping) or "prepared" not in recipes or "raw" not in recipes:
        raise ValueError("export preprocessing missing exact 'prepared' and 'raw' normalization recipes")
    if recipes["prepared"].get("value_range") != [0.0, 1.0]:
        raise ValueError("prepared normalization recipe must specify value_range [0.0, 1.0]")
    if recipes["raw"].get("percentiles") != [0.5, 99.5]:
        raise ValueError("raw normalization recipe must specify percentiles [0.5, 99.5]")

    # 2. Class map validation
    cmap = manifest.get("class_map")
    if not isinstance(cmap, Mapping):
        raise ValueError("export manifest missing valid 'class_map'")
    expected_classes = {
        "0": "background",
        "1": "right_ventricle",
        "2": "myocardium",
        "3": "left_ventricle",
    }
    for k, v in expected_classes.items():
        if cmap.get(k) != v and cmap.get(int(k)) != v:
            raise ValueError(f"export class_map mismatch for class {k}: expected {v}")

    # 3. Native axes validation
    axes = manifest.get("native_axes")
    if not isinstance(axes, Mapping):
        raise ValueError("export manifest missing valid 'native_axes'")
    if axes.get("slice_batch_tensor") != ["B", "C", "H", "W"]:
        raise ValueError("native_axes slice_batch_tensor must be ['B', 'C', 'H', 'W']")
    if axes.get("patient_volume_cine") != ["T", "Z", "Y", "X"]:
        raise ValueError("native_axes patient_volume_cine must be ['T', 'Z', 'Y', 'X']")

    # 4. Lineage validation
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("export manifest missing valid 'lineage'")
    lids = lineage.get("lids")
    uids = lineage.get("uids")
    if not isinstance(lids, list) or not isinstance(uids, list):
        raise ValueError("lineage lids and uids must be lists")
    if len(set(lids)) != len(lids):
        raise ValueError("lineage contains duplicate lids")
    if len(set(uids)) != len(uids):
        raise ValueError("lineage contains duplicate uids")
    if set(lids) & set(uids):
        raise ValueError("lineage lids and uids must be strictly disjoint")
    if lineage.get("label_budget") != len(lids):
        raise ValueError("lineage label_budget does not match len(lids)")
    _require_sha256(lineage.get("data_sha256"), "lineage.data_sha256")
    if type(manifest.get("synthetic")) is not bool or lineage.get("synthetic") is not manifest["synthetic"]:
        raise ValueError("export synthetic lineage mismatch")


def save_compact_checkpoint(
    model: nn.Module,
    optimizer: Any,
    config: Mapping[str, Any],
    data: Any,
    out: str | Path,
    *,
    step: int,
    teacher_identity: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Save full training checkpoint with model, optimizer, config, and data provenance.

    Returns the SHA-256 hex string of the written MANIFEST.json.
    """
    if type(step) is not int or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")

    arch = getattr(model, "architecture", None)
    if arch is None or not isinstance(arch, Mapping):
        raise ValueError("model must possess an explicit architecture attribute")
    validated_arch = validate_architecture(arch)

    data_provenance = _extract_strict_data_provenance(data)

    out_dir = Path(out).resolve()
    if out_dir.exists():
        raise ValueError(f"checkpoint destination already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    weights_path = out_dir / "weights.pt"
    manifest_path = out_dir / "MANIFEST.json"

    student_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    _verify_all_tensors_finite(student_state, "checkpoint model parameters")

    opt_state = None
    if optimizer is not None:
        if hasattr(optimizer, "state_dict"):
            opt_state = optimizer.state_dict()
        elif isinstance(optimizer, (list, tuple)):
            opt_state = [opt.state_dict() if hasattr(opt, "state_dict") else opt for opt in optimizer]
        elif isinstance(optimizer, Mapping):
            opt_state = dict(optimizer)

    payload = {
        "student": student_state,
        "optimizer": opt_state,
        "step": step,
        "teacher_identity": dict(teacher_identity) if teacher_identity is not None else None,
        "extra": dict(extra) if extra is not None else {},
    }
    torch.save(payload, weights_path)

    weights_only_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    serialization_bytes = os.path.getsize(weights_path)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    manifest = {
        "schema": "self_audit.compact.checkpoint.v1",
        "model_class": "CompactStudent",
        "architecture": validated_arch,
        "config": copy.deepcopy(dict(config)),
        "step": step,
        "data_provenance": data_provenance,
        "teacher_identity": copy.deepcopy(dict(teacher_identity)) if teacher_identity is not None else None,
        "model_file_sha256": runtime_model_file_sha256(),
        "training_source_sha256": compact_training_source_sha256(),
        "source_sha256": compact_training_source_sha256(),
        "payload_sha256": digest(weights_path),
        "weights_only_bytes": weights_only_bytes,
        "serialization_bytes": serialization_bytes,
        "parameters_total": total_params,
        "parameters_trainable": trainable_params,
        "teacher_included": False,
        "dino_required": False,
        "extra": copy.deepcopy(dict(extra)) if extra is not None else {},
    }

    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return digest(manifest_path)


def load_compact_checkpoint(
    path: str | Path,
    expected_sha: str,
    device: str | torch.device = "cpu",
) -> tuple[CompactStudent, dict[str, Any]]:
    """Load CompactStudent and metadata from a hash-pinned checkpoint.

    expected_sha is mandatory and must be a 64-hex SHA-256 pin.
    Verifies runtime model file SHA BEFORE constructing or restoring model.
    Returns (model, metadata).
    """
    clean_expected_sha = _require_sha256(expected_sha, "expected_sha")

    folder = Path(path).resolve()
    manifest_path = folder / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest not found: {manifest_path}")

    actual_manifest_sha = digest(manifest_path)
    if actual_manifest_sha != clean_expected_sha:
        raise ValueError(
            f"checkpoint manifest hash mismatch: expected {clean_expected_sha}, got {actual_manifest_sha}"
        )

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "self_audit.compact.checkpoint.v1":
        raise ValueError(f"unsupported checkpoint schema: {manifest.get('schema')}")

    # Verify model file SHA BEFORE model reconstruction/export
    expected_model_sha = _require_sha256(manifest.get("model_file_sha256"), "manifest.model_file_sha256")
    actual_model_sha = runtime_model_file_sha256()
    if actual_model_sha != expected_model_sha:
        raise ValueError(
            f"checkpoint model file SHA mismatch: expected {expected_model_sha}, got {actual_model_sha}. "
            "Model architecture code has been modified since checkpoint was saved."
        )

    weights_path = folder / "weights.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f"checkpoint payload weights.pt not found: {weights_path}")

    actual_payload_sha = digest(weights_path)
    expected_payload_sha = _require_sha256(manifest.get("payload_sha256"), "manifest.payload_sha256")
    if actual_payload_sha != expected_payload_sha:
        raise ValueError(
            f"checkpoint payload hash mismatch: expected {expected_payload_sha}, got {actual_payload_sha}"
        )

    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or "student" not in payload:
        raise ValueError("checkpoint payload does not contain student state dict")

    if manifest.get("step") != payload.get("step"):
        raise ValueError(
            f"checkpoint step mismatch between manifest ({manifest.get('step')}) and payload ({payload.get('step')})"
        )

    _verify_all_tensors_finite(payload["student"], "loaded checkpoint parameters")

    arch = validate_architecture(manifest.get("architecture", {}))
    model = build_compact_model(arch)
    model.load_state_dict(payload["student"], strict=True)
    model.to(device)

    metadata = dict(manifest)
    metadata["manifest_sha256"] = actual_manifest_sha
    if "optimizer" in payload:
        metadata["optimizer_state_available"] = payload["optimizer"] is not None

    return model, metadata


def export_compact_student(
    checkpoint: str | Path,
    expected_sha: str,
    out: str | Path,
) -> str:
    """Export a standalone, teacher/cache/DINO-independent student artifact.

    expected_sha is mandatory and must be a 64-hex SHA-256 pin.
    Returns the SHA-256 hex string of the written export MANIFEST.json.
    """
    clean_expected_sha = _require_sha256(expected_sha, "expected_sha")

    checkpoint_dir = Path(checkpoint).resolve()
    out_dir = Path(out).resolve()

    if out_dir.exists():
        raise ValueError(f"export destination already exists: {out_dir}")
    if out_dir == checkpoint_dir or out_dir.is_relative_to(checkpoint_dir) or checkpoint_dir.is_relative_to(out_dir):
        raise ValueError("export destination cannot overlap with source checkpoint")

    model, checkpoint_meta = load_compact_checkpoint(checkpoint_dir, expected_sha=clean_expected_sha, device="cpu")
    out_dir.mkdir(parents=True)

    weights_path = out_dir / "weights.pt"
    manifest_path = out_dir / "MANIFEST.json"

    # Strictly student-only state dict with all-finite verification
    student_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    _verify_all_tensors_finite(student_state, "export student parameters")
    torch.save({"student": student_state}, weights_path)

    weights_only_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    serialization_bytes = os.path.getsize(weights_path)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    clean_cfg = _clean_config_for_export(checkpoint_meta.get("config", {}))

    export_manifest = {
        "schema": "self_audit.compact.export.v1",
        "model_class": "CompactStudent",
        "architecture": validate_architecture(checkpoint_meta["architecture"]),
        "config": clean_cfg,
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
            "lids": checkpoint_meta["data_provenance"]["lids"],
            "uids": checkpoint_meta["data_provenance"]["uids"],
            "label_budget": checkpoint_meta["data_provenance"]["label_budget"],
            "data_sha256": checkpoint_meta["data_provenance"]["data_sha256"],
            "synthetic": checkpoint_meta["data_provenance"]["synthetic"],
            "teacher_identity": checkpoint_meta.get("teacher_identity"),
        },
        "synthetic": checkpoint_meta["data_provenance"]["synthetic"],
        "source_checkpoint_manifest_sha256": clean_expected_sha,
        "source_checkpoint_payload_sha256": checkpoint_meta.get("payload_sha256"),
        "model_file_sha256": runtime_model_file_sha256(),
        "training_source_sha256": checkpoint_meta.get("training_source_sha256", compact_training_source_sha256()),
        "source_sha256": compact_training_source_sha256(),
        "payload_sha256": digest(weights_path),
        "weights_only_bytes": weights_only_bytes,
        "serialization_bytes": serialization_bytes,
        "parameters_total": total_params,
        "parameters_trainable": trainable_params,
        "teacher_included": False,
        "external_teacher_required": False,
        "teacher_memory_included": False,
        "dino_required": False,
        "pretrained_weights_required": False,
    }

    manifest_path.write_text(json.dumps(export_manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return digest(manifest_path)


def load_compact_export(
    path: str | Path,
    expected_sha: str,
    device: str | torch.device = "cpu",
) -> tuple[CompactStudent, dict[str, Any]]:
    """Load a standalone CompactStudent export strictly without teacher or training assets.

    expected_sha is mandatory and must be a 64-hex SHA-256 pin.
    Re-verifies runtime model file integrity and validates consistency of preprocessing,
    class_map, native_axes, and lineage.
    Returns (model, metadata).
    """
    clean_expected_sha = _require_sha256(expected_sha, "expected_sha")

    folder = Path(path).resolve()
    manifest_path = folder / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"export manifest not found: {manifest_path}")

    actual_manifest_sha = digest(manifest_path)
    if actual_manifest_sha != clean_expected_sha:
        raise ValueError(
            f"export manifest hash mismatch: expected {clean_expected_sha}, got {actual_manifest_sha}"
        )

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "self_audit.compact.export.v1":
        raise ValueError(f"unsupported export schema: {manifest.get('schema')}")

    # Strict rejection of forbidden teacher or external dependencies
    if manifest.get("teacher_included") is not False:
        raise ValueError("forbidden teacher state in standalone export manifest")
    if manifest.get("external_teacher_required") is not False:
        raise ValueError("forbidden external teacher dependency in standalone export manifest")
    if manifest.get("teacher_memory_included") is not False:
        raise ValueError("forbidden teacher memory flag in standalone export manifest")
    if manifest.get("dino_required") is not False or manifest.get("pretrained_weights_required") is not False:
        raise ValueError("forbidden pretrained/DINO dependency in standalone export manifest")
    if "teacher_memory" in manifest.get("config", {}) and manifest["config"]["teacher_memory"] is not None:
        raise ValueError("forbidden non-null teacher_memory in export config")

    # Mandatory runtime model file SHA verification
    expected_model_sha = _require_sha256(manifest.get("model_file_sha256"), "manifest.model_file_sha256")
    actual_model_sha = runtime_model_file_sha256()
    if actual_model_sha != expected_model_sha:
        raise ValueError(
            f"runtime model file SHA mismatch: expected {expected_model_sha}, got {actual_model_sha}. "
            "Runtime compact_model.py has been tampered or modified."
        )

    # Validate preprocessing, class_map, native_axes, and lineage consistency
    _validate_export_metadata_consistency(manifest)

    weights_path = folder / "weights.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f"export payload weights.pt not found: {weights_path}")

    actual_payload_sha = digest(weights_path)
    expected_payload_sha = _require_sha256(manifest.get("payload_sha256"), "manifest.payload_sha256")
    if actual_payload_sha != expected_payload_sha:
        raise ValueError(
            f"export payload hash mismatch: expected {expected_payload_sha}, got {actual_payload_sha}"
        )

    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload.keys()) != {"student"}:
        raise ValueError("standalone export payload must strictly contain only 'student' state")

    student_state = payload["student"]
    if not isinstance(student_state, Mapping):
        raise ValueError("payload['student'] must be a mapping of parameters")

    for k in student_state.keys():
        if any(prefix in k for prefix in ("teacher", "teacher_memory", "window", "memory_bank")):
            raise ValueError(f"forbidden state dict parameter key in export: {k}")

    _verify_all_tensors_finite(student_state, "loaded export student parameters")

    arch = validate_architecture(manifest.get("architecture", {}))
    model = build_compact_model(arch)
    model.load_state_dict(student_state, strict=True)
    model.to(device).eval()

    # Recompute and verify weights-only bytes and parameters count
    recomputed_weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    if manifest.get("weights_only_bytes") != recomputed_weights_bytes:
        raise ValueError(
            f"manifest weights_only_bytes ({manifest.get('weights_only_bytes')}) differs from recomputed ({recomputed_weights_bytes})"
        )

    recomputed_total_params = sum(p.numel() for p in model.parameters())
    if manifest.get("parameters_total") != recomputed_total_params:
        raise ValueError(
            f"manifest parameters_total ({manifest.get('parameters_total')}) differs from recomputed ({recomputed_total_params})"
        )

    metadata = dict(manifest)
    metadata["manifest_sha256"] = actual_manifest_sha
    return model, metadata


__all__ = [
    "compact_training_source_sha256",
    "digest",
    "export_compact_student",
    "load_compact_checkpoint",
    "load_compact_export",
    "runtime_model_file_sha256",
    "save_compact_checkpoint",
    "source_digest",
]
