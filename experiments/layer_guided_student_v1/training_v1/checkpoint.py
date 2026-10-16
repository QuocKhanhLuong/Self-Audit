"""Strict checkpoints and image-only inference for the patient-training pilot.

This module deliberately has a different artifact schema from
``integration.artifacts``.  The latter is the representation-only integration
scaffold and remains loadable only through its original API.  Checkpoints
written here contain an optimizer snapshot for auditability, but this version
does not implement exact training resume (RNG state and sampler state are not
restored).  Reload therefore returns an inference-only model.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from integration import artifacts
from integration.geometry import normalize_cine, require_digest, slice_contexts

from . import config as training_config

SCHEMA = "layer_guided.training_checkpoint.v1"
PREPROCESSING = "layer_guided.per_cine_minmax.v1"
GEOMETRY = "native_pad32_coarse4_feature16_pixel_edges"
ONTOLOGY = ("BG", "RV", "MYO", "LV")
UNKNOWN = 255
STATUS_INITIALIZED = "INITIALIZED_ONLY"
STATUS_REPRESENTATION = "REPRESENTATION_ONLY"
STATUS_PSEUDO = "PSEUDO_SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED"
EVIDENCE_SCOPES = {
    "training": "FROZEN_HISTORICAL_Q_RELIABILITY_PSEUDO",
    "scope": "THIS_TRAINING_LOOP",
    "quality": "NOT_QUALITY_CERTIFIED",
    "gt_used_for_training": False,
    "upstream_lineage": "INHERITED_FROM_PINNED_BANK_AND_BACKBONE_RECEIPTS",
    "quality_certified": False,
}


def _json_copy(value: Any) -> Any:
    """Copy a metadata value while rejecting non-finite JSON values."""

    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _cpu_copy(value: Any) -> Any:
    """Detach tensors recursively so a checkpoint cannot alias live state."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return copy.deepcopy(value)


def _assert_finite(value: Any, *, where: str) -> None:
    """Reject non-finite tensors anywhere in the serialized training state."""

    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not torch.isfinite(value).all():
                raise ValueError(f"nonfinite value in {where}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, where=f"{where}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, where=f"{where}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite value in {where}")


def _require_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validated_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("training config mapping required")
    return training_config.validate_recipe(dict(config))


def _constructor_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    return training_config.constructor_recipe(_validated_config(config))


def _current_source_hashes() -> dict[str, str]:
    hashes = training_config.source_hashes()
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("training source hash inventory is empty")
    if any(
        not isinstance(path, str) or not isinstance(digest, str)
        for path, digest in hashes.items()
    ):
        raise ValueError("training source hash inventory is malformed")
    for digest in hashes.values():
        require_digest(digest)
    return dict(sorted(hashes.items()))


def _constructor_identity(
    model: torch.nn.Module, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify that ``model`` came from the approved integration constructor."""

    identity = getattr(model, "integration_identity", None)
    expected_recipe = _constructor_recipe(config)
    expected_sources = artifacts.source_hashes()
    if not isinstance(identity, Mapping):
        raise ValueError("model must carry approved constructor identity")
    if identity.get("recipe") != expected_recipe:
        raise ValueError("model constructor recipe differs from training config")
    if identity.get("sources") != expected_sources:
        raise ValueError("model integration sources differ from current source hashes")
    if not isinstance(identity.get("backbone"), Mapping):
        raise ValueError("approved backbone provenance missing from model identity")
    return {
        "recipe": _json_copy(identity["recipe"]),
        "backbone": _json_copy(identity["backbone"]),
        "sources": _json_copy(identity["sources"]),
    }


def _validate_training_lineage(lineage: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(lineage, Mapping):
        raise ValueError("training lineage mapping required")
    normalized = _json_copy(dict(lineage))
    # The old validator owns the split, teacher, bank and train-patient rules.
    # Call it unchanged so the old schema and its semantics remain authoritative.
    artifacts.validate_lineage(normalized)
    required = {
        "bank_manifest_sha256",
        "teacher",
        "locked_splits",
        "train_patients",
        "optimizer_steps",
        "semantic_updates",
        "allowed_classes",
    }
    missing = required.difference(normalized)
    if missing:
        raise ValueError(f"training lineage missing fields: {sorted(missing)}")
    _require_nonnegative_int(normalized["semantic_updates"], "semantic_updates")
    if normalized["semantic_updates"] > normalized["optimizer_steps"]:
        raise ValueError("semantic_updates cannot exceed optimizer_steps")
    allowed = normalized["allowed_classes"]
    if (
        not isinstance(allowed, list)
        or len(allowed) != len(ONTOLOGY)
        or any(type(value) is not bool for value in allowed)
    ):
        raise ValueError("allowed_classes must be four explicit booleans")
    return normalized


def _status(lineage: Mapping[str, Any]) -> str:
    steps = lineage["optimizer_steps"]
    updates = lineage["semantic_updates"]
    if steps == 0:
        if updates:
            raise ValueError(
                "semantic_updates cannot be positive before an optimizer step"
            )
        return STATUS_INITIALIZED
    if updates > 0:
        return STATUS_PSEUDO
    return STATUS_REPRESENTATION


def _validate_training_state(
    training_state: Mapping[str, Any], *, optimizer_steps: int
) -> dict[str, Any]:
    if not isinstance(training_state, Mapping):
        raise ValueError("training_state mapping required")
    state = _cpu_copy(dict(training_state))
    if "epoch" not in state or "step" not in state:
        raise ValueError("training_state must contain epoch and step")
    _require_nonnegative_int(state["epoch"], "training_state.epoch")
    step = _require_nonnegative_int(state["step"], "training_state.step")
    if step != optimizer_steps:
        raise ValueError("training_state.step must equal lineage.optimizer_steps")
    # Exact resume is intentionally not advertised.  Accepting these fields
    # would create a misleading impression that the sampler/RNG can be resumed.
    if "rng_states" in state or "sampler_state" in state or "resume" in state:
        raise ValueError("exact resume is not implemented; save inference-only state")
    _assert_finite(state, where="training_state")
    return state


def _state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = _cpu_copy(model.state_dict())
    if not isinstance(state, dict) or not state:
        raise ValueError("model state dictionary is empty")
    if any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("model state dictionary must contain named tensors")
    _assert_finite(state, where="model_state")
    return state


def _optimizer_state(optimizer: Any) -> dict[str, Any]:
    if optimizer is None or not hasattr(optimizer, "state_dict"):
        raise ValueError("optimizer with state_dict is required")
    state = _cpu_copy(optimizer.state_dict())
    if not isinstance(state, dict):
        raise ValueError("optimizer state dictionary required")
    _assert_finite(state, where="optimizer_state")
    return state


def _metadata(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    lineage: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    validated = _validated_config(config)
    identity = _constructor_identity(model, validated)
    status = _status(lineage)
    metadata = {
        "schema": SCHEMA,
        "config": validated,
        "constructor_recipe": identity["recipe"],
        "backbone": identity["backbone"],
        "integration_identity": identity,
        "source_hashes": _current_source_hashes(),
        "lineage": _json_copy(dict(lineage)),
        "status": status,
        "preprocessing": PREPROCESSING,
        "geometry": GEOMETRY,
        "ontology": list(ONTOLOGY),
        "evidence_scopes": EVIDENCE_SCOPES,
        "reload": {
            "mode": "inference_only",
            "resume_supported": False,
            "optimizer_state": "audit_only",
        },
        "training_progress": {
            "epoch": training_state["epoch"],
            "step": training_state["step"],
        },
    }
    return _json_copy(metadata)


def _validate_metadata(
    metadata: Mapping[str, Any], *, current_sources: bool = True
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping) or metadata.get("schema") != SCHEMA:
        raise ValueError("incompatible training checkpoint schema")
    result = _json_copy(dict(metadata))
    config = _validated_config(result.get("config"))
    if result.get("config") != config:
        raise ValueError("checkpoint config is not canonical")
    constructor_recipe = _constructor_recipe(config)
    if result.get("constructor_recipe") != constructor_recipe:
        raise ValueError("checkpoint constructor recipe mismatch")
    if result.get("ontology") != list(ONTOLOGY):
        raise ValueError("checkpoint ontology mismatch")
    if (
        result.get("preprocessing") != PREPROCESSING
        or result.get("geometry") != GEOMETRY
    ):
        raise ValueError("checkpoint preprocessing/geometry mismatch")
    lineage = _validate_training_lineage(result.get("lineage"))
    if result.get("lineage") != lineage:
        raise ValueError("checkpoint lineage is not canonical")
    expected_status = _status(lineage)
    if result.get("status") != expected_status:
        raise ValueError("checkpoint status does not match actual training counts")
    if result.get("evidence_scopes") != EVIDENCE_SCOPES:
        raise ValueError("checkpoint evidence scope mismatch")
    if result.get("reload") != {
        "mode": "inference_only",
        "resume_supported": False,
        "optimizer_state": "audit_only",
    }:
        raise ValueError("checkpoint reload contract mismatch")
    progress = result.get("training_progress")
    if (
        not isinstance(progress, Mapping)
        or set(progress) != {"epoch", "step"}
        or type(progress["epoch"]) is not int
        or type(progress["step"]) is not int
        or progress["epoch"] < 0
        or progress["step"] != lineage["optimizer_steps"]
    ):
        raise ValueError("checkpoint training progress mismatch")
    if current_sources and result.get("source_hashes") != _current_source_hashes():
        raise ValueError("checkpoint source hashes differ from current training source")
    require_digest(result.get("backbone", {}).get("weights_sha256"))
    if result.get("integration_identity", {}).get("recipe") != constructor_recipe:
        raise ValueError("checkpoint integration recipe mismatch")
    if (
        result.get("integration_identity", {}).get("sources")
        != artifacts.source_hashes()
    ):
        raise ValueError("checkpoint integration sources mismatch")
    if result.get("integration_identity", {}).get("backbone") != result.get("backbone"):
        raise ValueError("checkpoint backbone identity mismatch")
    return result


def _inventory(path: Path) -> dict[str, dict[str, Any]]:
    return {
        "model.pt": {
            "sha256": artifacts.sha256(path),
            "bytes": path.stat().st_size,
        }
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Any,
    out: str | Path,
    config: Mapping[str, Any],
    lineage: Mapping[str, Any],
    *,
    training_state: Mapping[str, Any],
) -> str:
    """Write a fresh inference-only training artifact and return its manifest SHA."""

    validated_config = _validated_config(config)
    validated_lineage = _validate_training_lineage(lineage)
    validated_training_state = _validate_training_state(
        training_state, optimizer_steps=validated_lineage["optimizer_steps"]
    )
    metadata = _metadata(
        model, validated_config, validated_lineage, validated_training_state
    )
    constructor_recipe = metadata["constructor_recipe"]
    state = _state_dict(model)
    artifacts.validate_fixed_buffers(state, constructor_recipe)
    optimizer_state = _optimizer_state(optimizer)
    payload = {
        "schema": SCHEMA,
        "metadata": metadata,
        "model_state": state,
        "optimizer_state": optimizer_state,
        "training_state": validated_training_state,
    }
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=False)
    payload_path = destination / "model.pt"
    try:
        torch.save(payload, payload_path)
        inventory = _inventory(payload_path)
        manifest = {
            "schema": SCHEMA,
            "metadata": metadata,
            "config_sha256": artifacts.canonical_hash(validated_config),
            "payload_sha256": inventory["model.pt"]["sha256"],
            "payload_bytes": inventory["model.pt"]["bytes"],
            "inventory": inventory,
        }
        (destination / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    except Exception:
        # Keep failed artifacts inspectable but never present a partial artifact
        # as a successful checkpoint.  The caller can remove this fresh folder.
        raise
    return artifacts.sha256(destination / "MANIFEST.json")


def _validate_manifest(
    manifest: Mapping[str, Any], payload_path: Path
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping) or manifest.get("schema") != SCHEMA:
        raise ValueError("incompatible training checkpoint manifest")
    metadata = _validate_metadata(manifest.get("metadata"))
    expected_payload = {
        "sha256": artifacts.sha256(payload_path),
        "bytes": payload_path.stat().st_size,
    }
    if manifest.get("payload_sha256") != expected_payload["sha256"]:
        raise ValueError("checkpoint payload digest mismatch")
    if manifest.get("payload_bytes") != expected_payload["bytes"]:
        raise ValueError("checkpoint payload size mismatch")
    if manifest.get("inventory") != {"model.pt": expected_payload}:
        raise ValueError("checkpoint inventory mismatch")
    if manifest.get("config_sha256") != artifacts.canonical_hash(metadata["config"]):
        raise ValueError("checkpoint config digest mismatch")
    return metadata


def load_checkpoint(
    folder: str | Path,
    expected_manifest_sha256: str,
    approval: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load only the new schema with strict source and constructor checks."""

    destination = Path(folder)
    require_digest(expected_manifest_sha256)
    manifest_path = destination / "MANIFEST.json"
    payload_path = destination / "model.pt"
    if not manifest_path.is_file() or not payload_path.is_file():
        raise ValueError(
            "incompatible artifact: new training checkpoint inventory required"
        )
    entries = list(destination.iterdir())
    if {entry.name for entry in entries} != {"MANIFEST.json", "model.pt"} or any(
        entry.is_symlink() for entry in entries
    ):
        raise ValueError("unexpected checkpoint inventory")
    if artifacts.sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("checkpoint manifest digest mismatch")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid checkpoint manifest") from exc
    metadata = _validate_manifest(manifest, payload_path)
    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("invalid checkpoint payload") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "metadata",
        "model_state",
        "optimizer_state",
        "training_state",
    }:
        raise ValueError("checkpoint payload schema mismatch")
    if payload.get("schema") != SCHEMA or payload.get("metadata") != metadata:
        raise ValueError("checkpoint payload metadata mismatch")
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint model state missing")
    _assert_finite(model_state, where="model_state")
    artifacts.validate_fixed_buffers(model_state, metadata["constructor_recipe"])
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint optimizer state missing")
    _assert_finite(optimizer_state, where="optimizer_state")
    training_state = _validate_training_state(
        payload.get("training_state"),
        optimizer_steps=metadata["lineage"]["optimizer_steps"],
    )
    if training_state != payload.get("training_state"):
        raise ValueError("checkpoint training state is not canonical")
    if metadata["training_progress"] != {
        "epoch": training_state["epoch"],
        "step": training_state["step"],
    }:
        raise ValueError("checkpoint training progress does not match payload")
    model, provenance = artifacts.construct_student(
        metadata["constructor_recipe"], approval, device
    )
    if provenance != metadata["backbone"]:
        raise ValueError("checkpoint backbone provenance mismatch")
    identity = getattr(model, "integration_identity", None)
    if (
        not isinstance(identity, Mapping)
        or identity.get("recipe") != metadata["constructor_recipe"]
    ):
        raise ValueError("reconstructed model constructor identity mismatch")
    if identity.get("sources") != artifacts.source_hashes():
        raise ValueError("reconstructed model integration source mismatch")
    try:
        model.load_state_dict(dict(model_state), strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            "checkpoint model state does not match approved student"
        ) from exc
    _assert_finite(model.state_dict(), where="loaded_model_state")
    return model.eval(), metadata


def _validate_prediction_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    checked = _validate_metadata(metadata)
    if checked["status"] not in {STATUS_PSEUDO}:
        raise ValueError(
            "representation-only or initialized checkpoint cannot produce named labels"
        )
    return checked


def _prediction_device(device: str | torch.device) -> torch.device:
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("prediction device must be CPU or CUDA")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA prediction device is unavailable")
    return selected


def predict_cine(
    model: torch.nn.Module,
    metadata: Mapping[str, Any],
    cine_tzyx: Any,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    min_probability: float = 0.7,
    min_margin: float = 0.2,
    *,
    image_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run image-only native inference with a separate dense and gated export.

    ``dense_labels`` is the four-class argmax.  ``gated_labels`` uses the
    declared probability/margin gate and the checkpoint's allowed-class
    policy; rejected pixels are ``UNKNOWN``.  No teacher, bank, or training
    state is consulted or modified by this function.

    Input must already have the historical image preprocessing used in the
    bank (1st/99th percentile clipping and z-score). For a raw NIfTI cine,
    call ``self_audit_guidance.io.read_cine`` first, then pass its TZYX array.
    This function applies the final per-cine min/max transform and slice context.
    """

    checked = _validate_prediction_metadata(metadata)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive integer batch_size required")
    for name, value in (
        ("min_probability", min_probability),
        ("min_margin", min_margin),
    ):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be finite in [0,1]")
    require_digest(image_sha256)
    try:
        cine = torch.as_tensor(cine_tzyx)
    except Exception as exc:
        raise ValueError("cine_tzyx must be a finite TZYX array") from exc
    normalized, receipt = normalize_cine(cine, image_sha256)
    if normalized.ndim != 4:
        raise ValueError("cine_tzyx must have TZYX shape")
    t_count, z_count, height, width = map(int, normalized.shape)
    indices = [(t, z) for t in range(t_count) for z in range(z_count)]
    selected = _prediction_device(device)
    model = model.to(selected)
    was_training = model.training
    model.eval()
    dense_chunks: list[torch.Tensor] = []
    gated_chunks: list[torch.Tensor] = []
    probability_chunks: list[torch.Tensor] = []
    entropy_chunks: list[torch.Tensor] = []
    allowed = torch.tensor(
        checked["lineage"]["allowed_classes"], dtype=torch.bool, device=selected
    )
    try:
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                images = slice_contexts(
                    normalized, indices[start : start + batch_size]
                ).to(selected)
                outputs = model(images)
                if not isinstance(outputs, Mapping) or "logits" not in outputs:
                    raise ValueError("student forward must return native logits")
                logits = outputs["logits"]
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.ndim != 4
                    or logits.shape[0] != images.shape[0]
                    or logits.shape[1] != len(ONTOLOGY)
                    or tuple(logits.shape[-2:]) != (height, width)
                    or not torch.isfinite(logits).all()
                ):
                    raise ValueError(
                        "student logits must be finite Bx4 native image shapes"
                    )
                probabilities = logits.softmax(dim=1)
                top = probabilities.topk(k=2, dim=1).values
                max_probability = top[:, 0]
                margin = top[:, 0] - top[:, 1]
                entropy = -(
                    probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
                    * probabilities.clamp_min(
                        torch.finfo(probabilities.dtype).tiny
                    ).log()
                ).sum(1)
                dense = probabilities.argmax(dim=1).to(torch.uint8)
                accepted = (max_probability >= float(min_probability)) & (
                    margin >= float(min_margin)
                )
                accepted &= allowed[dense.long()]
                gated = torch.where(accepted, dense, torch.full_like(dense, UNKNOWN))
                dense_chunks.append(dense.cpu())
                gated_chunks.append(gated.cpu())
                probability_chunks.append(max_probability.cpu())
                entropy_chunks.append(entropy.cpu())
    finally:
        model.train(was_training)
    shape = (t_count, z_count, height, width)
    dense_labels = torch.cat(dense_chunks, dim=0).reshape(shape).numpy()
    gated_labels = torch.cat(gated_chunks, dim=0).reshape(shape).numpy()
    max_probability = torch.cat(probability_chunks, dim=0).reshape(shape).numpy()
    entropy = torch.cat(entropy_chunks, dim=0).reshape(shape).numpy()
    receipt = {
        **receipt,
        "context_order": "previous_center_next_z",
        "model_schema": SCHEMA,
        "prediction_shape_tzyx": list(shape),
        "batch_size": batch_size,
        "thresholds": {
            "min_probability": float(min_probability),
            "min_margin": float(min_margin),
        },
        "allowed_classes": list(checked["lineage"]["allowed_classes"]),
        "dense_labels": "four_class_argmax",
        "gated_unknown": UNKNOWN,
        "teacher_forward": False,
        "bank_or_training_gate_mutated": False,
    }
    return dense_labels, gated_labels, max_probability, entropy, receipt
