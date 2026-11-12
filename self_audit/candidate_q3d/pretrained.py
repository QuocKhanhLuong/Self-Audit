"""Fail-closed loading for Q-3D MedNeXt public pretrained weights.

The A-3D factory is used only to construct its verified native MedNeXt
architecture. Q-3D loads a separately hashed public payload into that model so
the old A-3D TRAIN-checkpoint provenance contract is neither reused nor
fabricated. The boundary/SDF branch remains at its random initialization.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from torch import Tensor, nn

from ..candidate_a3d.teacher import (
    CLASS_NAMES,
    UPSTREAM_MAIN_SHA,
    DualBranchTeacher3D,
    MedNeXtL3DAdapter,
    build_teacher3d,
)

_EXPECTED_CLASSES = tuple(CLASS_NAMES)
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METADATA_FIELDS = (
    "pretraining_dataset",
    "overlap_acdc",
    "overlap_mnms",
    "class_mapping",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Q-3D {field} must be a mapping")
    return value


def _class_names(value: Any, field: str, *, allow_other: bool) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        try:
            value = [value[str(index)] if str(index) in value else value[index] for index in range(len(value))]
        except (KeyError, TypeError):
            raise ValueError(f"Q-3D {field} must be an ordered class mapping") from None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Q-3D {field} must declare an ordered class mapping")
    names = tuple(str(item).strip().upper() for item in value)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"Q-3D {field} must contain unique non-empty class names")
    if not allow_other and names != _EXPECTED_CLASSES:
        raise ValueError(f"Q-3D {field} must be {list(_EXPECTED_CLASSES)}")
    return names


def _metadata_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for container_key in ("metadata", "provenance"):
        value = payload.get(container_key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"Q-3D checkpoint {container_key} must be a mapping")
        for key in _METADATA_FIELDS:
            if key in value:
                if key in metadata and metadata[key] != value[key]:
                    raise ValueError(f"Q-3D checkpoint has conflicting metadata for {key}")
                metadata[key] = value[key]
    for key in _METADATA_FIELDS:
        if key in payload:
            if key in metadata and metadata[key] != payload[key]:
                raise ValueError(f"Q-3D checkpoint has conflicting metadata for {key}")
            metadata[key] = payload[key]
    return metadata


def _extract_state(payload: Any) -> tuple[Mapping[str, Tensor], dict[str, Any]]:
    """Accept a raw tensor state dict or a conventional safe checkpoint wrapper."""

    if not isinstance(payload, Mapping):
        raise ValueError("Q-3D checkpoint must be a mapping")
    metadata = _metadata_from_payload(payload)
    state_keys = [key for key in ("state_dict", "model_state_dict") if key in payload]
    if len(state_keys) > 1:
        raise ValueError("Q-3D checkpoint has ambiguous state_dict and model_state_dict fields")
    if state_keys:
        state = payload[state_keys[0]]
    elif payload and all(isinstance(value, Tensor) for value in payload.values()):
        # The official upstream export may be a bare state dict. Provenance is
        # still required from the explicit Q-3D config in that case.
        state = payload
    else:
        raise ValueError("Q-3D checkpoint must contain state_dict or model_state_dict tensors")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Q-3D checkpoint state dictionary is empty or invalid")
    if any(not isinstance(key, str) or not isinstance(value, Tensor) for key, value in state.items()):
        raise ValueError("Q-3D checkpoint state dictionary must map string keys to tensors")
    if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Q-3D checkpoint contains non-finite floating-point weights")
    return state, metadata


def _merge_declared_metadata(
    pretrained: Mapping[str, Any], payload_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in _METADATA_FIELDS:
        declared = pretrained.get(field)
        embedded = payload_metadata.get(field)
        if field == "class_mapping":
            declared_normalized = (
                _class_names(declared, "configured class_mapping", allow_other=True)
                if declared is not None else None
            )
            embedded_normalized = (
                _class_names(embedded, "checkpoint class_mapping", allow_other=True)
                if embedded is not None else None
            )
            if declared_normalized is not None and embedded_normalized is not None \
                    and declared_normalized != embedded_normalized:
                raise ValueError(f"Q-3D configured {field} conflicts with checkpoint metadata")
            value = declared_normalized if declared_normalized is not None else embedded_normalized
        else:
            if declared is not None and embedded is not None and declared != embedded:
                raise ValueError(f"Q-3D configured {field} conflicts with checkpoint metadata")
            value = declared if declared is not None else embedded
        if value is None:
            raise ValueError(f"Q-3D pretrained provenance is missing {field}")
        result[field] = value

    dataset = result["pretraining_dataset"]
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("Q-3D pretraining_dataset must be a non-empty declared name")
    for field in ("overlap_acdc", "overlap_mnms"):
        if type(result[field]) is not bool:
            raise ValueError(f"Q-3D {field} must be explicitly true or false")
    result["class_mapping"] = list(_class_names(result["class_mapping"], "checkpoint class_mapping", allow_other=True))
    return result


def _source_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Q-3D public checkpoint requires an explicit weights_source_url")
    parsed = urlparse(value.strip())
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Q-3D weights_source_url must be a public HTTPS URL")
    return value.strip()


def _normalize_state_keys(state: Mapping[str, Tensor], expected: Mapping[str, Tensor]) -> dict[str, Tensor]:
    keys = set(state)
    expected_keys = set(expected)
    if keys == expected_keys or keys & expected_keys:
        return dict(state)
    for prefix in ("module.", "model."):
        if keys and all(key.startswith(prefix) for key in keys):
            stripped = {key[len(prefix):]: value for key, value in state.items()}
            if len(stripped) != len(state):
                raise ValueError("Q-3D checkpoint has duplicate keys after prefix normalization")
            if set(stripped) & expected_keys:
                return stripped
    return dict(state)


def _load_exact_mednext_state(
    model: nn.Module,
    state: Mapping[str, Tensor],
    source_classes: tuple[str, ...],
) -> dict[str, Any]:
    if source_classes != _EXPECTED_CLASSES:
        raise ValueError(
            f"Q-3D checkpoint class_mapping must exactly match {list(_EXPECTED_CLASSES)}; "
            "classifier reinitialization is unsupported"
        )
    expected = model.state_dict()
    normalized = _normalize_state_keys(state, expected)
    expected_keys = set(expected)
    supplied_keys = set(normalized)
    unexpected = sorted(supplied_keys - expected_keys)
    missing = sorted(expected_keys - supplied_keys)
    shape_mismatches = {
        key: (tuple(normalized[key].shape), tuple(expected[key].shape))
        for key in sorted(supplied_keys & expected_keys)
        if tuple(normalized[key].shape) != tuple(expected[key].shape)
    }
    if unexpected:
        raise ValueError(f"Q-3D checkpoint has unexpected MedNeXt keys: {unexpected[:8]}")
    if missing:
        raise ValueError(f"Q-3D checkpoint is missing required MedNeXt keys: {missing[:8]}")
    if shape_mismatches:
        raise ValueError(f"Q-3D checkpoint has incompatible MedNeXt tensor shapes: {shape_mismatches}")
    model.load_state_dict(normalized, strict=True)
    return {
        "loaded_keys": sorted(f"encoder.model.{key}" for key in normalized),
        "missing_keys": [],
        "reinitialized_keys": [],
        "unexpected_keys": [],
        "checkpoint_unused_keys": [],
        "classifier_reinitialized": False,
        "source_class_mapping": list(source_classes),
        "target_class_mapping": list(_EXPECTED_CLASSES),
    }


def _construct_mednext_teacher(teacher_config: Mapping[str, Any]) -> tuple[DualBranchTeacher3D, dict[str, Any]]:
    """Construct only; this call deliberately passes weights=None."""

    teacher, provenance = build_teacher3d(
        {
            "backbone": "mednext_l_k3",
            "input_channels": 1,
            "source_repo": teacher_config["source_repo"],
            "source_commit": teacher_config["source_commit"],
            "weights": None,
            "weights_sha256": None,
        },
        synthetic=False,
    )
    if not isinstance(teacher, DualBranchTeacher3D):
        raise TypeError("Q-3D architecture factory did not return DualBranchTeacher3D")
    if not isinstance(teacher.encoder, MedNeXtL3DAdapter):
        raise TypeError("Q-3D real teacher requires the pinned MedNeXt-L k3 adapter")
    if getattr(teacher.encoder.model, "_candidate_a_native_v1", False) is not True:
        raise ValueError("Q-3D MedNeXt source does not expose the verified native v1 sequence")
    if provenance.get("backbone", {}).get("source_commit") != UPSTREAM_MAIN_SHA:
        raise ValueError("Q-3D source factory returned an unverified MedNeXt commit")
    return teacher, provenance


def _synthetic_teacher(config: Mapping[str, Any]) -> tuple[DualBranchTeacher3D, dict[str, Any]]:
    model = _require_mapping(config.get("model"), "model")
    teacher_cfg = _require_mapping(model.get("teacher"), "model.teacher")
    if teacher_cfg.get("architecture") != "mednext_l_k3":
        raise ValueError("Q-3D synthetic smoke must preserve the declared teacher track")
    teacher, base = build_teacher3d(
        {
            "backbone": "tiny_test3d",
            "input_channels": 1,
            "feature_channels": [8, 16, 24, 32],
        },
        synthetic=True,
    )
    random_keys = sorted(teacher.state_dict())
    inventory = {
        "loaded_keys": [],
        "missing_keys": [],
        "reinitialized_keys": random_keys,
        "unexpected_keys": [],
        "checkpoint_unused_keys": [],
        "source": "synthetic_random_fixture",
        "evidence_scope": "SOFTWARE_ONLY",
    }
    provenance = {
        **base,
        "quality": "SYNTHETIC_RANDOM_INIT",
        "synthetic": True,
        "accuracy_claim": "NOT_MEASURED",
        "weight_inventory": inventory,
    }
    teacher.provenance = copy.deepcopy(provenance)
    return teacher, provenance


def build_q3d_teacher(
    config: Mapping[str, Any], synthetic: bool = False
) -> tuple[DualBranchTeacher3D, dict[str, Any]]:
    """Build the Q-3D teacher and return a provenance-bound state inventory.

    Real runs require a clean source checkout pinned to the verified MedNeXt
    upstream commit, a local checkpoint and matching SHA-256, a declared public
    source URL, pretraining dataset, overlap flags and the exact checkpoint
    class mapping. The only random real-run parameters are the boundary/SDF path.
    No path in this loader downloads or falls back to random real-run weights.
    """

    if type(synthetic) is not bool:
        raise ValueError("Q-3D synthetic selector must be a boolean")
    if not isinstance(config, Mapping):
        raise ValueError("Q-3D teacher config must be a nested mapping")
    if synthetic:
        return _synthetic_teacher(config)

    model = _require_mapping(config.get("model"), "model")
    teacher_cfg = _require_mapping(model.get("teacher"), "model.teacher")
    if teacher_cfg.get("architecture") != "mednext_l_k3":
        raise ValueError("real Q-3D teacher requires architecture mednext_l_k3")
    pretrained = _require_mapping(teacher_cfg.get("pretrained"), "model.teacher.pretrained")

    source_repo = teacher_cfg.get("source_repo")
    source_commit = teacher_cfg.get("source_commit")
    if not isinstance(source_repo, str) or not source_repo.strip():
        raise ValueError("real Q-3D teacher requires source_repo")
    if source_commit != UPSTREAM_MAIN_SHA:
        raise ValueError(f"real Q-3D teacher source_commit must be {UPSTREAM_MAIN_SHA}")

    weights_value = pretrained.get("weights_path")
    expected_hash = pretrained.get("weights_sha256")
    if not isinstance(weights_value, str) or not weights_value.strip():
        raise ValueError("real Q-3D teacher requires a local weights_path")
    if not isinstance(expected_hash, str) or not _HEX_SHA256.fullmatch(expected_hash.lower()):
        raise ValueError("real Q-3D teacher requires a 64-character weights_sha256")
    source_url = _source_url(pretrained.get("weights_source_url"))
    path = Path(weights_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Q-3D MedNeXt checkpoint not found: {path}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash.lower():
        raise ValueError("Q-3D MedNeXt checkpoint SHA256 mismatch")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Q-3D checkpoint could not be loaded with weights_only=True") from exc
    state, embedded_metadata = _extract_state(payload)
    declared_metadata = _merge_declared_metadata(pretrained, embedded_metadata)
    source_classes = tuple(declared_metadata["class_mapping"])
    if "classifier_reinitialization" in pretrained:
        raise ValueError(
            "Q-3D classifier_reinitialization is unsupported; encoder and region head must load"
        )

    teacher, construction_provenance = _construct_mednext_teacher(teacher_cfg)
    inventory = _load_exact_mednext_state(
        teacher.encoder.model,
        state,
        source_classes,
    )
    boundary_keys = sorted(key for key in teacher.state_dict() if key.startswith("boundary_path."))
    inventory["missing_keys"] = sorted(set(inventory["missing_keys"]) | set(boundary_keys))
    inventory["reinitialized_keys"] = sorted(set(inventory["reinitialized_keys"]) | set(boundary_keys))
    inventory["boundary_random_keys"] = boundary_keys
    inventory["source"] = "public_checkpoint_local_hash_verified"
    inventory["evidence_scope"] = "PROVENANCE_VERIFIED_SOFTWARE_LOAD_ONLY"

    backbone_meta = copy.deepcopy(getattr(teacher, "backbone_meta", {}))
    backbone_meta.update(
        {
            "weights_status": "PUBLIC_PRETRAINED_LOCAL_HASH_VERIFIED",
            "weights_path": str(path),
            "weights_sha256": actual_hash,
            "weights_source_url": source_url,
            "pretraining_data": copy.deepcopy(declared_metadata),
            "region_head_pretrained": True,
            "random_components": ["boundary_path"],
        }
    )
    teacher.backbone_meta = backbone_meta
    teacher.encoder.weights_sha256 = actual_hash
    teacher.encoder.weights_path = str(path)
    teacher.encoder.pretraining_data = copy.deepcopy(declared_metadata)
    provenance = copy.deepcopy(construction_provenance)
    provenance.update(
        {
            "backbone": backbone_meta,
            "pretraining_data": copy.deepcopy(declared_metadata),
            "quality": "PUBLIC_PRETRAINED_LOCAL_HASH_VERIFIED",
            "synthetic": False,
            "accuracy_claim": "NOT_MEASURED",
            "checkpoint": {
                "weights_path": str(path),
                "weights_sha256": actual_hash,
                "weights_source_url": source_url,
                "source_repo": str(Path(source_repo).expanduser().resolve()),
                "source_commit": source_commit,
            },
            "weight_inventory": inventory,
        }
    )
    teacher.provenance = copy.deepcopy(provenance)
    return teacher, provenance


__all__ = ["build_q3d_teacher"]
