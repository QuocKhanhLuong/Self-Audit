"""Fail-closed loading for the hash-pinned nnFoundationCNN ResEnc-L teacher.

The public checkpoint contains an encoder only. Q-3D constructs a fresh
dual-head teacher around that encoder and deliberately never imports decoder
or segmentation-head tensors from the checkpoint. Real construction has no
random or reduced-architecture fallback; synthetic smoke uses the same
``Q3DResEncTeacher`` class with its explicit six-stage small fixture.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from torch import nn

from .teacher_resenc import (
    Q3DResEncTeacher,
    build_res_enc_l,
    load_nnfoundation_encoder,
)

_EXPECTED_CLASSES = ("BG", "RV", "MYO", "LV")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HF_REPO = "MIC-DKFZ/nnFoundationCNN"
_HF_REVISION = "f443c09319988e833b3d9589bf95909e4443d7b2"
_HF_RESOLVE_URL = (
    f"https://huggingface.co/{_HF_REPO}/resolve/{_HF_REVISION}/checkpoint_final.pth"
)
_PRETRAINING_DATASET = "nnFoundation (~2.1M CT/MR/PET, 125 sources; arXiv 2609.26924 Table D7)"
_OVERLAP_NOTE = "According to Table D7; not independently verified."
_LICENSE = "CC-BY-SA-4.0"
_WEIGHTS_SHA256 = "ac262d3e8c226c79f9567fddc34d38e011284730ddf2294b5c6c319ce2f22bbd"
_WEIGHTS_ENV = "Q3D_NNFOUNDATION_WEIGHTS"
_PHASE_RECEIPT_PATH = "reports/q3d-resenc-phase0-20261112/erf_probe.json"
_PHASE_RECEIPT_SHA256 = "2f05888b45ff3bfbf19f90bd91abbd4f0b2ab193b37375c5d875950117d2a14e"
_DNA_VERSION = "0.4.4"
_NNSSL_COMMIT = "3d774e942256bd70e5e264c6b94fa38e25f7ed74"
_NNUNET_COMMIT = "c04f7ec8bbffaf3ebc3b26aa2c89042e8a8e368b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_weights_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("real nnFoundation teacher requires weights_path")
    text = value.strip()
    if text == "${Q3D_NNFOUNDATION_WEIGHTS}":
        bound = os.environ.get(_WEIGHTS_ENV)
        if not bound:
            raise ValueError(
                f"weights_path uses ${{Q3D_NNFOUNDATION_WEIGHTS}}; set {_WEIGHTS_ENV} for a real run"
            )
        text = bound
    return Path(text).expanduser().resolve()


def _resolve_phase_receipt_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("nnFoundation phase receipt path must be declared")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Q-3D {field} must be a mapping")
    return value


def _source_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Q-3D nnFoundation checkpoint requires an explicit HF resolve URL")
    url = value.strip()
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.hostname != "huggingface.co"
            or parsed.username or parsed.password or url != _HF_RESOLVE_URL):
        raise ValueError("Q-3D nnFoundation weights_source_url must pin the exact HF revision")
    return url


def _declared_provenance(pretrained: Mapping[str, Any]) -> dict[str, Any]:
    """Validate fields that cannot be recovered from the bare HF payload."""

    expected = {
        "weights_path", "weights_sha256", "weights_source_url", "pretraining_dataset",
        "overlap_acdc", "overlap_mnms", "overlap_note", "license",
        "class_mapping", "class_mapping_scope", "provenance_status",
    }
    if set(pretrained) != expected:
        missing = sorted(expected - set(pretrained))
        extra = sorted(set(pretrained) - expected)
        raise ValueError(f"model.teacher.pretrained has missing={missing}, extra={extra}")
    if pretrained.get("provenance_status") != "READY":
        raise ValueError("real nnFoundation teacher requires provenance_status=READY")
    path = pretrained.get("weights_path")
    digest = pretrained.get("weights_sha256")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("real nnFoundation teacher requires weights_path")
    if not isinstance(digest, str) or not _HEX_SHA256.fullmatch(digest):
        raise ValueError("real nnFoundation teacher requires a lowercase 64-character weights_sha256")
    if digest != _WEIGHTS_SHA256:
        raise ValueError("real nnFoundation teacher weights_sha256 is not the Phase-0 checkpoint hash")
    _source_url(pretrained.get("weights_source_url"))
    if pretrained.get("pretraining_dataset") != _PRETRAINING_DATASET:
        raise ValueError("nnFoundation pretraining_dataset differs from the Table D7 declaration")
    if pretrained.get("overlap_acdc") is not False or pretrained.get("overlap_mnms") is not False:
        raise ValueError("nnFoundation provenance must declare overlap_acdc=overlap_mnms=false")
    if pretrained.get("overlap_note") != _OVERLAP_NOTE:
        raise ValueError("nnFoundation overlap_note must state that Table D7 was not independently verified")
    if pretrained.get("license") != _LICENSE:
        raise ValueError("nnFoundation weights require license CC-BY-SA-4.0")
    if list(pretrained.get("class_mapping", ())) != list(_EXPECTED_CLASSES):
        raise ValueError("nnFoundation downstream class_mapping must be BG/RV/MYO/LV")
    if pretrained.get("class_mapping_scope") != "downstream_segmentation_contract":
        raise ValueError("nnFoundation class_mapping is a downstream segmentation contract, not pretraining ontology")
    return copy.deepcopy(dict(pretrained))


def _teacher_strides(teacher_config: Mapping[str, Any]) -> list[list[int]]:
    value = teacher_config.get("encoder_strides")
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError("nnFoundation teacher must declare six encoder_strides")
    parsed: list[list[int]] = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError(f"encoder_strides[{index}] must be a z,y,x triple")
        if any(type(axis) is not int or axis <= 0 for axis in item):
            raise ValueError(f"encoder_strides[{index}] must contain positive integers")
        parsed.append(list(item))
    return parsed


def _require_dependency_pins(teacher_config: Mapping[str, Any]) -> None:
    pins = teacher_config.get("dependency_pins")
    expected = {
        "dynamic_network_architectures": _DNA_VERSION,
        "nnssl_commit": _NNSSL_COMMIT,
        "nnunet_commit": _NNUNET_COMMIT,
    }
    if not isinstance(pins, Mapping) or dict(pins) != expected:
        raise ValueError("nnFoundation teacher dependency pins do not match Phase 0")
    try:
        installed = importlib.metadata.version("dynamic-network-architectures")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("dynamic-network-architectures==0.4.4 is required for ResEnc-L") from exc
    if installed != _DNA_VERSION:
        raise RuntimeError(
            f"dynamic-network-architectures=={_DNA_VERSION} is required; found {installed}"
        )


def _require_phase_measurement(teacher_config: Mapping[str, Any]) -> dict[str, Any]:
    """Require the locked stage-3 ERF receipt without inventing stage-4 data."""

    phase = teacher_config.get("phase_measurement")
    if not isinstance(phase, Mapping):
        raise ValueError("nnFoundation teacher requires the locked ERF phase measurement")
    if phase.get("receipt_path") != _PHASE_RECEIPT_PATH:
        raise ValueError("nnFoundation phase receipt path is not the locked Phase-0 artifact")
    if phase.get("receipt_sha256") != _PHASE_RECEIPT_SHA256:
        raise ValueError("nnFoundation phase receipt SHA-256 is not the locked Phase-0 artifact")
    if (phase.get("status") != "PASS"
            or phase.get("method") != "gradient_erf_mean_centroid"
            or phase.get("samples") != 16
            or phase.get("threshold_voxels") != 2.0):
        raise ValueError("nnFoundation phase measurement must be the passing gradient-ERF receipt")
    main = phase.get("encoder_stride8")
    s1 = phase.get("encoder_stride16")
    if (not isinstance(main, Mapping) or main.get("status") != "MEASURED_WITHIN_THRESHOLD"
            or main.get("offset_zyx") != [0.0, 0.0, 0.0]
            or not isinstance(s1, Mapping)
            or s1.get("status") != "NOT_MEASURED"
            or s1.get("offset_zyx") != [0.0, 0.0, 0.0]
            or s1.get("policy") != "theoretical_zero_reference_not_inherited_from_encoder_stride8"):
        raise ValueError(
            "nnFoundation phase contract must lock nominal stage-3 phase and explicitly mark stage-4 unmeasured"
        )
    if teacher_config.get("phase_offsets") != {
        "encoder_stride8": [0.0, 0.0, 0.0],
        "encoder_stride16": [0.0, 0.0, 0.0],
    }:
        raise ValueError("nnFoundation phase_offsets must match the locked nominal phase contract")
    receipt_path = _resolve_phase_receipt_path(phase["receipt_path"])
    if not receipt_path.is_file():
        raise FileNotFoundError(f"nnFoundation ERF phase receipt not found: {receipt_path}")
    if _sha256(receipt_path) != _PHASE_RECEIPT_SHA256:
        raise ValueError("nnFoundation ERF phase receipt SHA-256 mismatch")
    return copy.deepcopy(dict(phase))


def _seeded_context(config: Mapping[str, Any]):
    seed = config.get("seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("Q-3D teacher seed must be a nonnegative integer")
    context = torch.random.fork_rng(devices=[])
    context.__enter__()
    torch.manual_seed(seed)
    return context


def _attach_channel_alias(encoder: nn.Module) -> None:
    """Keep the engine accessor while exposing the ResEnc widths."""

    channels = getattr(encoder, "output_channels", None)
    if channels is None:
        raise ValueError("nnFoundation encoder must expose output_channels")
    values = tuple(int(item) for item in channels)
    if values not in {(32, 64, 128, 256, 320, 320), (4, 8, 12, 16, 20, 20)}:
        raise ValueError("unexpected ResEnc-L encoder channel declaration")
    if not hasattr(encoder, "channels"):
        encoder.channels = values


def _construct_resenc_teacher(
    teacher_config: Mapping[str, Any], *, config: Mapping[str, Any], synthetic: bool = False,
) -> tuple[Q3DResEncTeacher, dict[str, Any]]:
    """Build a fresh ResEnc-L dual-head teacher with deterministic heads."""

    _require_dependency_pins(teacher_config)
    strides = _teacher_strides(teacher_config)
    context = _seeded_context(config)
    try:
        if synthetic:
            teacher = Q3DResEncTeacher(
                strides=strides,
                phase_offsets=teacher_config.get("phase_offsets"),
                synthetic=True,
                num_classes=len(_EXPECTED_CLASSES),
            )
        else:
            encoder_wrapper = build_res_enc_l(strides)
            encoder = getattr(encoder_wrapper, "encoder", encoder_wrapper)
            _attach_channel_alias(encoder)
            teacher = Q3DResEncTeacher(
                encoder=encoder,
                phase_offsets=teacher_config.get("phase_offsets"),
                num_classes=len(_EXPECTED_CLASSES),
            )
    finally:
        context.__exit__(None, None, None)
    _attach_channel_alias(teacher.encoder)
    provenance = {
        "architecture": "nnfoundation_resenc_l",
        "nnssl_commit": _NNSSL_COMMIT,
        "nnunet_commit": _NNUNET_COMMIT,
        "dynamic_network_architectures": _DNA_VERSION,
        "synthetic": bool(synthetic),
        "heads": ["region", "boundary_sdf"],
        "random_components": ["region_decoder_decoder_and_head", "boundary_path"],
        "phase_measurement": copy.deepcopy(teacher_config.get("phase_measurement")),
    }
    return teacher, provenance


def _synthetic_teacher(config: Mapping[str, Any]) -> tuple[Q3DResEncTeacher, dict[str, Any]]:
    model = _require_mapping(config.get("model"), "model")
    teacher_cfg = _require_mapping(model.get("teacher"), "model.teacher")
    if teacher_cfg.get("architecture") != "nnfoundation_resenc_l":
        raise ValueError("Q-3D synthetic smoke must preserve the nnFoundation teacher track")
    teacher, provenance = _construct_resenc_teacher(teacher_cfg, config=config, synthetic=True)
    inventory = {
        "loaded_keys": [],
        "missing_keys": [],
        "unexpected_keys": [],
        "reinitialized_keys": sorted(teacher.state_dict()),
        "source": "synthetic_resenc_fixture",
        "evidence_scope": "SOFTWARE_ONLY",
    }
    provenance.update({
        "quality": "SYNTHETIC_RANDOM_INIT",
        "accuracy_claim": "NOT_MEASURED",
        "weight_inventory": inventory,
    })
    teacher.provenance = copy.deepcopy(provenance)
    return teacher, provenance


def build_q3d_teacher(
    config: Mapping[str, Any], synthetic: bool = False,
) -> tuple[Q3DResEncTeacher, dict[str, Any]]:
    """Build the Q-3D teacher; real construction is strictly provenance-bound."""

    if type(synthetic) is not bool or not isinstance(config, Mapping):
        raise ValueError("Q-3D teacher requires a mapping config and boolean synthetic selector")
    if synthetic:
        return _synthetic_teacher(config)

    model = _require_mapping(config.get("model"), "model")
    teacher_config = _require_mapping(model.get("teacher"), "model.teacher")
    if teacher_config.get("architecture") != "nnfoundation_resenc_l":
        raise ValueError("real Q-3D teacher requires architecture nnfoundation_resenc_l")
    _require_dependency_pins(teacher_config)
    phase_measurement = _require_phase_measurement(teacher_config)
    pretrained = _declared_provenance(
        _require_mapping(teacher_config.get("pretrained"), "model.teacher.pretrained")
    )
    path = _resolve_weights_path(pretrained["weights_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Q-3D nnFoundation checkpoint not found: {path}")
    actual_sha = _sha256(path)
    if actual_sha != pretrained["weights_sha256"]:
        raise ValueError("Q-3D nnFoundation checkpoint SHA-256 mismatch")

    # The architecture worker performs weights_only=True, adaptation-plan
    # validation, strict encoder key/shape checks, and decoder-key exclusion.
    context = _seeded_context(config)
    try:
        encoder, load_meta = load_nnfoundation_encoder(
            path,
            pretrained["weights_sha256"],
            strides=_teacher_strides(teacher_config),
            input_channels=1,
        )
        _attach_channel_alias(encoder)
        teacher = Q3DResEncTeacher(
            encoder=encoder,
            phase_offsets=teacher_config.get("phase_offsets"),
            num_classes=len(_EXPECTED_CLASSES),
        )
    finally:
        context.__exit__(None, None, None)

    encoder_keys = sorted(f"encoder.{key}" for key in encoder.state_dict())
    boundary_keys = sorted(key for key in teacher.state_dict() if key.startswith("boundary_path."))
    region_keys = sorted(key for key in teacher.state_dict() if key.startswith("region_decoder."))
    # ``UNetDecoder`` keeps the encoder as a registered module.  Its
    # ``region_decoder.encoder.*`` state-dict entries are aliases of the
    # strictly loaded ``encoder.*`` tensors, not random decoder weights.
    loaded_alias_keys = sorted(
        key for key in region_keys if key.startswith("region_decoder.encoder.")
    )
    region_decoder_random_keys = sorted(
        key for key in region_keys if not key.startswith("region_decoder.encoder.")
    )
    inventory = {
        "loaded_keys": encoder_keys,
        "loaded_key_count": len(encoder_keys),
        "loaded_alias_keys": loaded_alias_keys,
        "loaded_alias_key_count": len(loaded_alias_keys),
        "missing_keys": [],
        "unexpected_keys": [],
        "reinitialized_keys": sorted(set(boundary_keys + region_decoder_random_keys)),
        "random_components": ["region_decoder_decoder_and_head", "boundary_path"],
        "boundary_random_keys": boundary_keys,
        "region_decoder_random_keys": region_decoder_random_keys,
        "source": "nnfoundation_public_checkpoint_local_hash_verified",
        "evidence_scope": "PROVENANCE_VERIFIED_SOFTWARE_LOAD_ONLY",
    }
    backbone_meta = copy.deepcopy(getattr(teacher, "backbone_meta", {}))
    backbone_meta.update({
        "architecture": "nnfoundation_resenc_l",
        "weights_status": "PUBLIC_PRETRAINED_LOCAL_HASH_VERIFIED",
        "weights_path": str(path),
        "weights_sha256": actual_sha,
        "weights_source_url": pretrained["weights_source_url"],
        "hf_revision": _HF_REVISION,
        "nnssl_commit": _NNSSL_COMMIT,
        "nnunet_commit": _NNUNET_COMMIT,
        "dynamic_network_architectures": _DNA_VERSION,
        "phase_measurement": copy.deepcopy(phase_measurement),
        "pretraining_data": copy.deepcopy(pretrained),
        "strict_encoder_load": True,
        "encoder_load": copy.deepcopy(load_meta),
        "random_components": ["region_decoder_decoder_and_head", "boundary_path"],
    })
    teacher.backbone_meta = backbone_meta
    teacher.encoder.weights_sha256 = actual_sha
    teacher.encoder.weights_path = str(path)
    teacher.encoder.pretraining_data = copy.deepcopy(pretrained)
    provenance = {
        "architecture": "nnfoundation_resenc_l",
        "nnssl_commit": _NNSSL_COMMIT,
        "nnunet_commit": _NNUNET_COMMIT,
        "dynamic_network_architectures": _DNA_VERSION,
        "synthetic": False,
        "accuracy_claim": "NOT_MEASURED",
        "backbone": backbone_meta,
        "pretraining_data": copy.deepcopy(pretrained),
        "checkpoint": {
            "weights_path": str(path),
            "weights_sha256": actual_sha,
            "weights_source_url": pretrained["weights_source_url"],
            "hf_revision": _HF_REVISION,
        },
        "weight_inventory": inventory,
    }
    teacher.provenance = copy.deepcopy(provenance)
    return teacher, provenance


__all__ = ["build_q3d_teacher"]
