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
import random
import re
from typing import Any, Mapping
import numpy as np
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


def _serialize_rng_state(rng_state: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Serialize RNG state into weights_only-safe pure Python and Tensor types."""
    if rng_state is None:
        return None
    res = dict(rng_state)
    if "np_legacy_rng" in res and res["np_legacy_rng"] is not None:
        s = res["np_legacy_rng"]
        if isinstance(s, (list, tuple)) and len(s) == 5:
            keys = s[1]
            if hasattr(keys, "tolist"):
                keys = keys.tolist()
            res["np_legacy_rng"] = [s[0], list(keys), int(s[2]), int(s[3]), float(s[4])]
    if "py_rng" in res and res["py_rng"] is not None:
        res["py_rng"] = list(res["py_rng"])
    return res


def save_compact_checkpoint(
    model: nn.Module,
    optimizer: Any,
    config: Mapping[str, Any],
    data: Any,
    out: str | Path,
    *,
    step: int,
    teacher_identity: Mapping[str, Any] | None = None,
    rng_state: Mapping[str, Any] | None = None,
    counters: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
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
        "config": copy.deepcopy(dict(config)),
        "teacher_identity": dict(teacher_identity) if teacher_identity is not None else None,
        "rng": _serialize_rng_state(rng_state),
        "counters": dict(counters) if counters is not None else None,
        "protocol": dict(protocol) if protocol is not None else None,
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
        "exact_resume_supported": rng_state is not None,
        "rng_saved": rng_state is not None,
        "counters": copy.deepcopy(dict(counters)) if counters is not None else None,
        "protocol": copy.deepcopy(dict(protocol)) if protocol is not None else None,
        "initial_state_sha256": (protocol.get("initial_state_sha256") if protocol else None),
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
    if "rng" in payload:
        metadata["rng_state_available"] = payload["rng"] is not None

    return model, metadata


def load_exact_compact_checkpoint(
    path: str | Path,
    expected_sha: str,
    device: str | torch.device = "cpu",
    expected_config: Mapping[str, Any] | None = None,
) -> tuple[CompactStudent, dict[str, Any], dict[str, Any]]:
    """Load CompactStudent, full payload, and metadata for exact training resume.

    Enforces mandatory 64-hex SHA-256 pin, runtime model file integrity,
    payload hash validation, all-finite parameters, presence of RNG states,
    and immutability of protocol and total step budget.

    Returns:
        (model, payload, metadata)
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

    # 1. Source training hash equality
    expected_source_sha = _require_sha256(
        manifest.get("training_source_sha256") or manifest.get("source_sha256"),
        "manifest.training_source_sha256",
    )
    actual_source_sha = compact_training_source_sha256()
    if actual_source_sha != expected_source_sha:
        raise ValueError(
            f"checkpoint training source SHA mismatch: expected {expected_source_sha}, got {actual_source_sha}. "
            "Training source code has been modified since checkpoint was saved."
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

    # Contract: Exact resume without required RNG metadata must fail clearly.
    rng_state = payload.get("rng")
    if rng_state is None:
        raise ValueError(
            f"checkpoint at {folder} lacks mandatory exact RNG state metadata; "
            "legacy checkpoints cannot be used for exact training resume"
        )
    for required_key in ("torch_cpu_rng", "py_rng", "np_legacy_rng", "data_rng_state", "probe_rng_state"):
        if required_key not in rng_state or rng_state[required_key] is None:
            raise ValueError(
                f"checkpoint RNG state missing mandatory '{required_key}' for exact resume"
            )

    is_cuda_device = (
        (isinstance(device, str) and device.startswith("cuda"))
        or (isinstance(device, torch.device) and device.type == "cuda")
    )
    if is_cuda_device:
        if "torch_cuda_rng" not in rng_state or rng_state["torch_cuda_rng"] is None:
            raise ValueError("checkpoint lacks mandatory 'torch_cuda_rng' for exact CUDA resume")
        cuda_rng = rng_state["torch_cuda_rng"]
        if not isinstance(cuda_rng, (list, tuple)) or len(cuda_rng) == 0:
            raise TypeError("torch_cuda_rng must be a non-empty list or tuple for exact CUDA resume")

    # Mandatory optimizer state in payload
    if payload.get("optimizer") is None:
        raise ValueError("checkpoint payload missing mandatory optimizer state for exact resume")

    # Mandatory config in payload and manifest
    if "config" not in payload or payload["config"] is None:
        raise ValueError("checkpoint payload missing mandatory persisted config")
    if manifest.get("config") != payload.get("config"):
        raise ValueError("manifest config does not match payload config")

    # Mandatory protocol and counters in payload and manifest
    if payload.get("protocol") is None or manifest.get("protocol") is None:
        raise ValueError("checkpoint missing mandatory protocol metadata for exact resume")
    if payload.get("counters") is None or manifest.get("counters") is None:
        raise ValueError("checkpoint missing mandatory counters metadata for exact resume")

    # Compare manifest protocol/counters/teacher to hashed payload
    if manifest.get("protocol") != payload.get("protocol"):
        raise ValueError("manifest protocol does not match payload protocol")
    if manifest.get("counters") != payload.get("counters"):
        raise ValueError("manifest counters do not match payload counters")
    if manifest.get("teacher_identity") != payload.get("teacher_identity"):
        raise ValueError("manifest teacher_identity does not match payload teacher_identity")

    counters = payload["counters"]
    if not isinstance(counters, Mapping):
        raise TypeError("checkpoint counters must be a mapping")

    required_counter_keys = (
        "completed_steps",
        "counts",
        "slices_sampled",
        "slices_optimized",
        "optimizer_calls",
        "cumulative_training_walltime",
        "total_cache_time",
        "seen_l", "seen_a", "seen_u", "seen_probe",
        "audit_totals", "audit_unique_frames",
        "total_u_cache_time", "total_probe_cache_time", "total_probe_sample_time",
        "warm_start_logs",
    )
    for ck in required_counter_keys:
        if ck not in counters or counters[ck] is None:
            raise ValueError(f"checkpoint counters missing mandatory field '{ck}' for exact resume")
    for ck in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        if ck not in counters:
            raise ValueError(f"checkpoint counters missing mandatory field '{ck}' for exact resume")
        value = counters[ck]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"checkpoint counters '{ck}' must be non-negative bytes or null")
    for ck in ("cumulative_training_walltime", "total_cache_time", "total_u_cache_time",
               "total_probe_cache_time", "total_probe_sample_time"):
        value = counters[ck]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"checkpoint counters '{ck}' must be finite and non-negative")
    for ck in ("seen_l", "seen_a", "seen_u", "seen_probe", "audit_unique_frames", "warm_start_logs"):
        if not isinstance(counters[ck], list):
            raise ValueError(f"checkpoint counters '{ck}' must be a list")
    if not isinstance(counters['audit_totals'], Mapping):
        raise ValueError("checkpoint counters 'audit_totals' must be a mapping")

    step_val = int(payload["step"])
    if int(counters["completed_steps"]) != step_val:
        raise ValueError(
            f"counters completed_steps ({counters['completed_steps']}) mismatch checkpoint step ({step_val})"
        )

    if step_val > 0:
        cum_wall = counters.get("cumulative_training_walltime")
        if not isinstance(cum_wall, (int, float)) or cum_wall < 0:
            raise ValueError("cumulative_training_walltime must be a non-negative float")
        for map_key in ("counts", "slices_sampled", "slices_optimized", "optimizer_calls"):
            m = counters.get(map_key)
            if not isinstance(m, Mapping) or len(m) == 0:
                raise ValueError(f"checkpoint counters field '{map_key}' cannot be empty for step {step_val}")

    # Protocol requirements: initial_state_sha256 must be persisted 64-hex
    proto = payload["protocol"]
    init_state_sha = proto.get("initial_state_sha256") or manifest.get("initial_state_sha256")
    if not init_state_sha:
        raise ValueError("checkpoint missing persisted initial_state_sha256 lineage; fallback hashing prohibited")
    _require_sha256(init_state_sha, "protocol.initial_state_sha256")

    step_val = int(payload["step"])
    if step_val > 0:
        if "log_bytes" not in proto or "log_sha256" not in proto:
            raise ValueError(
                f"checkpoint protocol missing mandatory 'log_bytes' and 'log_sha256' for exact resume at step {step_val}"
            )
        _require_sha256(proto["log_sha256"], "protocol.log_sha256")
        if type(proto["log_bytes"]) is not int or proto["log_bytes"] < 0:
            raise ValueError("protocol.log_bytes must be a non-negative integer")

    # Immutability validation against current configuration
    if expected_config is not None:
        from .compact_training import validate_compact_config

        checkpoint_cfg = manifest.get("config", {})

        norm_expected = validate_compact_config(expected_config, allow_full_run=True)
        norm_ckpt = validate_compact_config(checkpoint_cfg, allow_full_run=True)

        # 1. Total step budget cannot be silently changed
        ckpt_steps = norm_ckpt.get("steps")
        req_steps = norm_expected.get("steps")
        if ckpt_steps is not None and req_steps is not None and req_steps != ckpt_steps:
            raise ValueError(
                f"cannot silently change total step budget on resume: "
                f"checkpoint declared steps={ckpt_steps}, requested steps={req_steps}"
            )

        # 2. Immutable method
        ckpt_method = norm_ckpt.get("method")
        req_method = norm_expected.get("method")
        if ckpt_method is not None and req_method is not None and req_method != ckpt_method:
            raise ValueError(
                f"protocol mismatch on resume: checkpoint method={ckpt_method}, requested method={req_method}"
            )

        # 3. Seed immutability
        ckpt_seed = norm_ckpt.get("seed")
        req_seed = norm_expected.get("seed")
        if ckpt_seed is not None and req_seed is not None and req_seed != ckpt_seed:
            raise ValueError(
                f"protocol mismatch on resume: checkpoint seed={ckpt_seed}, requested seed={req_seed}"
            )

        # 4. Probe seed immutability
        ckpt_probe_seed = norm_ckpt.get("probe_seed")
        req_probe_seed = norm_expected.get("probe_seed")
        if ckpt_probe_seed is not None and req_probe_seed is not None and req_probe_seed != ckpt_probe_seed:
            raise ValueError(
                f"protocol mismatch on resume: checkpoint probe_seed={ckpt_probe_seed}, requested probe_seed={req_probe_seed}"
            )

        # 5. Selector immutability
        ckpt_selector = norm_ckpt.get("selector", "original")
        req_selector = norm_expected.get("selector", "original")
        if ckpt_selector != req_selector:
            raise ValueError(
                f"protocol mismatch on resume: checkpoint selector={ckpt_selector}, requested selector={req_selector}"
            )

        # Exclude ONLY stop_after_step execution boundary
        norm_expected_diff = dict(norm_expected)
        norm_ckpt_diff = dict(norm_ckpt)
        norm_expected_diff.pop("stop_after_step", None)
        norm_ckpt_diff.pop("stop_after_step", None)
        if norm_expected.get("data") is None:
            norm_expected_diff.pop("data", None)
            norm_ckpt_diff.pop("data", None)

        differing_keys = []
        all_keys = set(norm_expected_diff.keys()) | set(norm_ckpt_diff.keys())
        for k in sorted(all_keys):
            if norm_expected_diff.get(k) != norm_ckpt_diff.get(k):
                differing_keys.append(k)
        if differing_keys:
            raise ValueError(
                f"immutable configuration mismatch on resume for keys: {differing_keys}; "
                f"checkpoint had {[norm_ckpt_diff.get(k) for k in differing_keys]}, "
                f"requested {[norm_expected_diff.get(k) for k in differing_keys]}"
            )

        # Probe state validation against expected configuration
        probe_state = rng_state.get("probe_rng_state")
        if isinstance(probe_state, Mapping):
            if "seed" in probe_state and norm_expected.get("probe_seed") is not None and probe_state["seed"] != norm_expected["probe_seed"]:
                raise ValueError(
                    f"probe sampler seed mismatch: state seed={probe_state.get('seed')!r}, expected={norm_expected['probe_seed']!r}"
                )
            if "batch_size" in probe_state and norm_expected.get("probe_batch_size") is not None and probe_state["batch_size"] != norm_expected["probe_batch_size"]:
                raise ValueError(
                    f"probe sampler batch_size mismatch: state batch_size={probe_state.get('batch_size')!r}, expected={norm_expected['probe_batch_size']!r}"
                )
            if "crop_size" in probe_state and norm_expected.get("crop_size") is not None and probe_state["crop_size"] != norm_expected["crop_size"]:
                raise ValueError(
                    f"probe sampler crop_size mismatch: state crop_size={probe_state.get('crop_size')!r}, expected={norm_expected['crop_size']!r}"
                )

        # Data provenance immutability
        ckpt_prov = manifest.get("data_provenance", {})
        if "data" in expected_config and isinstance(expected_config["data"], Mapping):
            req_data_sha = expected_config["data"].get("manifest_sha256")
            if req_data_sha and ckpt_prov.get("data_sha256") != req_data_sha:
                raise ValueError(
                    f"data provenance mismatch on resume: checkpoint data_sha256={ckpt_prov.get('data_sha256')}, requested={req_data_sha}"
                )
            if expected_config["data"].get("budget_k") is not None and ckpt_prov.get("label_budget") != expected_config["data"]["budget_k"]:
                raise ValueError(
                    f"data provenance label_budget mismatch on resume: checkpoint label_budget={ckpt_prov.get('label_budget')}, requested={expected_config['data']['budget_k']}"
                )
            if expected_config["data"].get("labeled_ids") is not None and sorted(ckpt_prov.get("lids", [])) != sorted(expected_config["data"]["labeled_ids"]):
                raise ValueError(
                    f"data provenance labeled_ids mismatch on resume: checkpoint lids={ckpt_prov.get('lids')}, requested={expected_config['data']['labeled_ids']}"
                )
            if proto.get("train_manifest_sha256") and req_data_sha and proto["train_manifest_sha256"] != req_data_sha:
                raise ValueError(
                    f"protocol train_manifest_sha256 mismatch on resume: protocol={proto['train_manifest_sha256']}, requested={req_data_sha}"
                )

    # Reconstruct model and load state strictly
    arch = validate_architecture(manifest.get("architecture", {}))
    model = build_compact_model(arch)
    model.load_state_dict(payload["student"], strict=True)
    model.to(device)

    metadata = dict(manifest)
    metadata["manifest_sha256"] = actual_manifest_sha
    metadata["payload_sha256"] = actual_payload_sha
    metadata["rng_state_available"] = True
    metadata["optimizer_state_available"] = payload.get("optimizer") is not None
    metadata["initial_state_sha256"] = init_state_sha

    return model, dict(payload), metadata


def restore_compact_rng(
    rng_dict: Mapping[str, Any],
    data_rng: np.random.Generator,
    probe_sampler: Any = None,
    device: str | torch.device = "cpu",
) -> None:
    """Restore all global PyTorch/Python/NumPy RNGs and generator states after model load.

    Validates all fields strictly and never swallows Python exceptions.
    """
    if not isinstance(rng_dict, Mapping):
        raise TypeError("rng_dict must be a mapping of RNG states")

    # 1. PyTorch CPU RNG
    if "torch_cpu_rng" not in rng_dict or rng_dict["torch_cpu_rng"] is None:
        raise ValueError("rng_dict missing mandatory 'torch_cpu_rng'")
    cpu_rng = rng_dict["torch_cpu_rng"]
    if isinstance(cpu_rng, torch.Tensor):
        torch.set_rng_state(cpu_rng.cpu())
    elif isinstance(cpu_rng, (list, bytes, bytearray)):
        torch.set_rng_state(torch.tensor(cpu_rng, dtype=torch.uint8))
    else:
        raise TypeError(f"invalid type for torch_cpu_rng: {type(cpu_rng)}")

    # 2. PyTorch CUDA RNG
    is_cuda_device = (
        (isinstance(device, str) and device.startswith("cuda"))
        or (isinstance(device, torch.device) and device.type == "cuda")
    )
    if is_cuda_device:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({device}) but CUDA is not available")
        if "torch_cuda_rng" not in rng_dict or rng_dict["torch_cuda_rng"] is None:
            raise ValueError("checkpoint lacks mandatory 'torch_cuda_rng' for exact CUDA resume")
        cuda_rng = rng_dict["torch_cuda_rng"]
        if not isinstance(cuda_rng, (list, tuple)) or len(cuda_rng) == 0:
            raise TypeError(f"torch_cuda_rng must be a non-empty list or tuple, got {type(cuda_rng)}")
        if len(cuda_rng) != torch.cuda.device_count():
            raise ValueError(
                f"CUDA device count mismatch on resume: checkpoint recorded {len(cuda_rng)} devices, "
                f"current environment has {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([
            t.cpu() if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.uint8)
            for t in cuda_rng
        ])
    elif "torch_cuda_rng" in rng_dict and rng_dict["torch_cuda_rng"] is not None:
        if torch.cuda.is_available():
            cuda_rng = rng_dict["torch_cuda_rng"]
            if isinstance(cuda_rng, (list, tuple)) and len(cuda_rng) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([
                    t.cpu() if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.uint8)
                    for t in cuda_rng
                ])

    # 3. Python Random RNG (DO NOT SWALLOW EXCEPTIONS)
    if "py_rng" not in rng_dict or rng_dict["py_rng"] is None:
        raise ValueError("rng_dict missing mandatory 'py_rng'")
    py_rng = rng_dict["py_rng"]
    if not isinstance(py_rng, (list, tuple)):
        raise TypeError(f"py_rng must be a list or tuple, got {type(py_rng)}")
    random.setstate(tuple(py_rng))

    # 4. NumPy Legacy RNG
    if "np_legacy_rng" not in rng_dict or rng_dict["np_legacy_rng"] is None:
        raise ValueError("rng_dict missing mandatory 'np_legacy_rng'")
    raw_leg = rng_dict["np_legacy_rng"]
    if not isinstance(raw_leg, (list, tuple)) or len(raw_leg) != 5:
        raise ValueError("np_legacy_rng must be a 5-element tuple or list")
    arr = np.array(raw_leg[1], dtype=np.uint32)
    np.random.set_state((str(raw_leg[0]), arr, int(raw_leg[2]), int(raw_leg[3]), float(raw_leg[4])))

    # 5. Data Generator RNG
    if "data_rng_state" not in rng_dict or rng_dict["data_rng_state"] is None:
        raise ValueError("rng_dict missing mandatory 'data_rng_state'")
    if not isinstance(rng_dict["data_rng_state"], Mapping):
        raise TypeError("data_rng_state must be a mapping")
    data_rng.bit_generator.state = copy.deepcopy(rng_dict["data_rng_state"])

    # 6. Probe Generator RNG
    if probe_sampler is not None:
        if "probe_rng_state" not in rng_dict or rng_dict["probe_rng_state"] is None:
            raise ValueError("rng_dict missing mandatory 'probe_rng_state' for probe sampler")
        probe_sampler.load_state(rng_dict["probe_rng_state"])



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
    "load_exact_compact_checkpoint",
    "restore_compact_rng",
    "runtime_model_file_sha256",
    "save_compact_checkpoint",
    "source_digest",
]
