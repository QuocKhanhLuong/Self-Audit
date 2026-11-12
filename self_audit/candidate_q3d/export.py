"""Student-only Q-3D export and cold-load validation.

The artifact is a minimal tensor state plus the inference constructor fields.
It does not serialize optimizer, teacher, tokenizer-training, or cache state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from self_audit.data import CLASS_MAPPING
from . import CHECKPOINT_SCHEMA
from .quality import (
    WAIVER_EVENTS_KEY,
    WAIVER_STATE_KEY,
    quality_waiver,
    validate_quality_waiver,
)

EXPORT_SCHEMA = "q3d.student_export.v1"
ENGINE_RESUME_SCHEMAS = {CHECKPOINT_SCHEMA}
_FORBIDDEN_STATE_PARTS = {"teacher", "sdf", "cache", "optimizer", "tokenizer_teacher", "physiology"}
_WAIVER_FLAG = "--q3d-exploratory-quality-waiver"
_EXPLORATORY_OVERRIDE_STATUS = "EXPLORATORY_QUALITY_WAIVER"
_EXPLORATORY_METADATA_KEYS = ("exploratory", WAIVER_STATE_KEY, "measured_quality_gates")


def _supplied_waiver(reason: str | None) -> dict[str, Any] | None:
    try:
        return quality_waiver(reason)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{_WAIVER_FLAG} requires a nonempty REASON string") from exc


def _json_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Q-3D {label} is not finite JSON metadata") from exc


def _gate_summary(gate: Any, label: str) -> Any:
    """Copy one measured gate verbatim, minus per-patient weight tables."""
    if gate is None:
        return None
    if not isinstance(gate, Mapping):
        raise ValueError(f"Q-3D checkpoint {label} must be a mapping")
    return _json_copy({key: value for key, value in gate.items() if key != "patient_weights"}, label)


def _checkpoint_exploratory(payload: Mapping[str, Any],
                            supplied: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return truthful exploratory export metadata, or ``None`` for a normal checkpoint.

    A waived checkpoint exports only with the exact explicit policy; measured
    gates are copied verbatim (never rewritten to PASS) and a synthetic
    override can never masquerade as, or be combined with, a waiver.
    """
    try:
        policy = validate_quality_waiver(payload.get(WAIVER_STATE_KEY))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Q-3D checkpoint exploratory waiver is malformed: {exc}") from exc
    marker = payload.get("exploratory")
    gate = payload.get("gate")
    override = payload.get("gate_override")
    exploratory_override = (
        isinstance(override, Mapping) and override.get("status") == _EXPLORATORY_OVERRIDE_STATUS
    )
    tokenizer_artifact = payload.get("tokenizer_artifact")
    metadata_policy = (tokenizer_artifact.get(WAIVER_STATE_KEY)
                       if isinstance(tokenizer_artifact, Mapping) else None)
    if policy is None:
        if (marker not in (None, False) or exploratory_override or metadata_policy is not None
                or payload.get("exploratory_tokenizer_waiver_used") is True):
            raise ValueError("Q-3D checkpoint declares an exploratory waiver without its policy")
        if supplied is not None:
            raise ValueError(f"{_WAIVER_FLAG} was supplied but the checkpoint is not exploratory")
        return None
    if marker not in (None, True):
        raise ValueError("Q-3D checkpoint carries a waiver but is marked exploratory=false")
    if supplied is None:
        raise RuntimeError(
            "Q-3D checkpoint was trained under an exploratory quality waiver; export requires "
            f"the same explicit {_WAIVER_FLAG} REASON"
        )
    if dict(supplied) != policy:
        raise RuntimeError("Q-3D checkpoint exploratory waiver differs from the supplied REASON")
    if payload.get("synthetic_gate_override_used") is True or (
        isinstance(override, Mapping) and override.get("status") == "SYNTHETIC_TEST_OVERRIDE"
    ):
        raise ValueError("Q-3D checkpoint cannot combine the synthetic override and a waiver")
    if isinstance(gate, Mapping) and gate.get("passed") is False:
        if isinstance(tokenizer_artifact, Mapping) and tokenizer_artifact.get("frozen") is True:
            if (
                not exploratory_override
                or override.get("original_gate") != dict(gate)
                or override.get("waiver") != policy
                or metadata_policy != policy
                or tokenizer_artifact.get("exploratory") is not True
                or tokenizer_artifact.get("gate") != dict(gate)
                or tokenizer_artifact.get("initialized") is not True
            ):
                raise ValueError(
                    "Q-3D waived checkpoint tokenizer metadata does not record the measured "
                    "failed gate and the exact waiver"
                )
    elif isinstance(gate, Mapping) and gate.get("passed") is True:
        if exploratory_override or metadata_policy is not None or gate.get("status") not in (None, "PASS"):
            raise ValueError("Q-3D checkpoint tokenizer gate claims PASS inconsistently with a waiver")
    gates = payload.get("physiology_support_gates", [])
    if not isinstance(gates, (list, tuple)):
        raise ValueError("Q-3D checkpoint physiology_support_gates must be a list")
    measured = {
        "tokenizer_gate": _gate_summary(gate, "tokenizer gate"),
        "physiology_support_gate": _gate_summary(
            payload.get("physiology_support_gate"), "physiology support gate"),
        "physiology_support_gates": [
            _gate_summary(item, "physiology support gate") for item in gates
        ],
        "kd_support": _json_copy({
            "j_kd_supported_branches": payload.get("j_kd_supported_branches"),
            "exploratory_zero_distillation": payload.get("exploratory_zero_distillation"),
        }, "KD support"),
    }
    return {
        "exploratory": True,
        WAIVER_STATE_KEY: copy.deepcopy(policy),
        "measured_quality_gates": measured,
        "events": _json_copy(list(payload.get(WAIVER_EVENTS_KEY, []) or []), "waiver events"),
    }


def _artifact_exploratory(metadata: Mapping[str, Any],
                          supplied: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate a deploy artifact's exploratory declaration against the explicit policy."""
    try:
        policy = validate_quality_waiver(metadata.get(WAIVER_STATE_KEY))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Q-3D artifact exploratory waiver is malformed: {exc}") from exc
    marker = metadata.get("exploratory")
    if policy is None:
        if marker not in (None, False) or "measured_quality_gates" in metadata:
            raise ValueError("Q-3D artifact declares exploratory metadata without its waiver policy")
        if supplied is not None:
            raise ValueError(f"{_WAIVER_FLAG} was supplied but the artifact is not exploratory")
        return None
    if marker is not True or not isinstance(metadata.get("measured_quality_gates"), Mapping):
        raise ValueError("Q-3D exploratory artifact lacks exploratory=True or its measured gates")
    if supplied is None:
        raise RuntimeError(
            "Q-3D artifact is exploratory (quality gates waived); loading it requires the same "
            f"explicit {_WAIVER_FLAG} REASON"
        )
    if dict(supplied) != policy:
        raise RuntimeError("Q-3D artifact exploratory waiver differs from the supplied REASON")
    return policy


def _sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_object_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes(order="C")).hexdigest()


def _valid_extra_state(value: Any) -> bool:
    valid = (
        isinstance(value, Mapping)
        and value.get("schema") == "self_audit.q3d.codebook.v1"
        and set(value) == {"schema", "version", "sha256", "size", "dimension", "initialized", "frozen"}
        and (value.get("version") is None or type(value.get("version")) in {str, int})
        and (value.get("sha256") is None or isinstance(value.get("sha256"), str))
        and type(value.get("size")) is int and value.get("size") >= 2
        and type(value.get("dimension")) is int and value.get("dimension") >= 1
        and type(value.get("initialized")) is bool
        and type(value.get("frozen")) is bool
    )
    if not valid:
        return False
    version, checksum = value["version"], value["sha256"]
    if value["frozen"] is True:
        return (
            value["initialized"] is True
            and type(version) in {str, int} and version not in {"", 0}
            and isinstance(checksum, str) and len(checksum) == 64
            and all(char in "0123456789abcdef" for char in checksum.lower())
    )
    return value["initialized"] is False and version is None and checksum is None


def _is_state_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str)
        and (torch.is_tensor(item) or (key == "_extra_state" and _valid_extra_state(item)))
        for key, item in value.items()
    )


def _source_student_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = ("best_student_state", "student_state", "student_state_dict", "student",
                  "state_dict", "model_state_dict")
    raw = next((payload[key] for key in candidates if _is_state_dict(payload.get(key))), None)
    if raw is None and _is_state_dict(payload):
        raw = payload
    if raw is None:
        raise ValueError("Q-3D checkpoint must contain a recognized student state mapping")
    return raw


def _normalized_state_key(original_key: str) -> str:
    key = original_key
    while key.startswith("module."):
        key = key[len("module."):]
    key = key.removeprefix("student.")
    return key


def _continuous_head_channels(raw_state: Mapping[str, Any]) -> int | None:
    weight = next((value for original_key, value in raw_state.items()
                   if isinstance(original_key, str)
                   and _normalized_state_key(original_key) == "continuous_feature_head.weight"), None)
    if weight is None:
        return None
    if not torch.is_tensor(weight) or weight.ndim != 5 or int(weight.shape[0]) < 1:
        raise ValueError("Q-3D train-only C1 head has an invalid checkpoint shape")
    return int(weight.shape[0])


def _extract_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = _source_student_state(payload)
    result: dict[str, Any] = {}
    for original_key, tensor in raw.items():
        key = _normalized_state_key(str(original_key))
        if key == "_extra_state":
            if not _valid_extra_state(tensor):
                raise ValueError("Q-3D student extra state has an unsupported codebook schema")
            result[key] = dict(tensor)
            continue
        if key.startswith("sdf_head."):
            continue
        if key.startswith("continuous_feature_head."):
            continue
        parts = {part.lower() for part in key.split(".")}
        if parts & _FORBIDDEN_STATE_PARTS:
            raise ValueError(f"forbidden non-deploy state key in student export: {original_key}")
        if "sdf" in key.lower():
            raise ValueError(f"unrecognized train-only SDF state key: {original_key}")
        if key in result:
            raise ValueError(f"duplicate student state key after prefix normalization: {key}")
        if not torch.is_tensor(tensor):
            raise ValueError(f"non-tensor Q-3D student state is not an allowed extra state: {original_key}")
        result[key] = tensor.detach().cpu().contiguous()
    if not result:
        raise ValueError("Q-3D student state is empty after removing the SDF head")
    return result


def _mapping_at(value: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, Mapping) else {}


def _declared_arm(config: Mapping[str, Any], student: Mapping[str, Any]) -> str:
    value = config.get("arm", student.get("arm", ""))
    if not isinstance(value, str) or not value:
        raise ValueError("Q-3D checkpoint config must name its training arm")
    return value.strip().lower().replace("_", "-")


def _student_arm(config: Mapping[str, Any], student: Mapping[str, Any]) -> str:
    value = student.get("arm", config.get("arm", ""))
    if not isinstance(value, str) or not value:
        raise ValueError("Q-3D checkpoint config must name its student arm")
    source = value.strip().lower().replace("_", "-")
    aliases = {
        "c0": "C0", "mean-teacher": "C0", "c1": "C1",
        "v0": "V0", "v1": "V1", "v1u": "V1-U", "v1-u": "V1-U",
        "b1": "B1", "e1": "E1", "o1": "O1", "s1": "S1",
        # The residual source ablation retains the V1 student API; its altered
        # student settings are already carried in model.student.
        "v1-residual2": "V1",
    }
    try:
        return aliases[source]
    except KeyError as exc:
        raise ValueError(f"unsupported student-only Q-3D export arm: {source}") from exc


def _minimal_student_config(config: Mapping[str, Any], state: Mapping[str, Any], *,
                            continuous_head_channels: int | None = None) -> dict[str, Any]:
    """Keep only model construction values; discard data, training and teacher metadata."""
    model_root = _mapping_at(config, "model")
    student_src = model_root.get("student", config.get("student", {}))
    if not isinstance(student_src, Mapping):
        raise TypeError("Q-3D config must contain a nested model.student mapping")
    allowed = {
        "architecture", "input_shape", "input_channels", "classes", "stage_channels", "z_stride",
        "epa_heads", "epa_support", "epa_pool_zyx", "epa_geometric_fraction", "residual2",
        "decoder_inputs", "code_gate_init", "dense_skip", "sdf_head_train_only", "max_parameters",
        "native_grid", "crop", "resample", "channels", "encoder_channels", "num_classes",
        "in_channels", "arm", "attention", "vq", "residual_vq", "residual_level",
        "sdf_head", "sdf_head_enabled", "support_count",
    }
    student = {key: copy.deepcopy(value) for key, value in student_src.items() if key in allowed}
    source_arm = _declared_arm(config, student_src)
    arm = _student_arm(config, student_src)
    student["arm"] = arm
    # Resolve the training-only output as absent during model construction.
    student["sdf_head_enabled"] = False
    student["sdf_head"] = {"enabled": False}
    tokenizer = config.get("tokenizer", {})
    if not isinstance(tokenizer, Mapping):
        tokenizer = {}
    vq = copy.deepcopy(dict(student.get("vq", {}))) if isinstance(student.get("vq", {}), Mapping) else {}
    codebook_tensor = state.get("C0")
    extra_state = state.get("_extra_state")
    if not _valid_extra_state(extra_state):
        raise ValueError("Q-3D student checkpoint is missing the canonical codebook metadata")
    vq.setdefault("size", int(extra_state["size"]))
    vq.setdefault("dim", int(extra_state["dimension"]))
    if torch.is_tensor(codebook_tensor):
        codebook_shape = tuple(int(value) for value in codebook_tensor.shape)
        if codebook_shape and all(codebook_shape):
            if len(codebook_shape) != 2:
                raise ValueError("Q-3D C0 state must be [K,d]")
            vq.setdefault("size", codebook_shape[0])
            vq.setdefault("dim", codebook_shape[1])
            vq.setdefault("tau", tokenizer.get("tau", tokenizer.get("temperature", 0.1)))
    token = {key: copy.deepcopy(tokenizer[key]) for key in
             ("codebook_size", "dimension", "temperature", "size", "dim", "tau", "embedding_dim")
             if key in tokenizer}
    if vq:
        student["vq"] = vq
    if arm == "C1" and (type(continuous_head_channels) is not int or continuous_head_channels <= 0):
        raise ValueError("C1 export requires the saved train-only continuous head to establish its shape")
    return {
        "arm": arm, "source_arm": source_arm,
        "tokenizer": token, "model": {"student": student},
    }


def _construct_deploy_model(config: Mapping[str, Any], state: Mapping[str, Any], *,
                            device: str | torch.device = "cpu",
                            dtype: torch.dtype | None = None) -> torch.nn.Module:
    from self_audit.candidate_q3d.student import CandidateQ3DStudent

    extra_state = state.get("_extra_state")
    if not _valid_extra_state(extra_state):
        raise ValueError("Q-3D deploy state is missing canonical codebook metadata")
    constructor_config = copy.deepcopy(dict(config))
    if config.get("arm") == "C1":
        # C1's only train-only structural addition is its continuous feature
        # alignment head. The deploy graph is the otherwise identical C0
        # student; no teacher channel count or teacher module is needed.
        constructor_config["arm"] = "C0"
        constructor_config["model"]["student"]["arm"] = "C0"
    model = CandidateQ3DStudent(constructor_config)
    if dtype is not None:
        model.to(dtype=dtype)
    metadata = {key: extra_state[key] for key in
                ("version", "sha256", "size", "dimension", "initialized", "frozen")}
    if model.use_vq:
        codebook = state.get("C0")
        if not torch.is_tensor(codebook) or codebook.ndim != 2:
            raise ValueError("Q-3D VQ deploy state is missing its C0 matrix")
        model.install_codebook({"codes": codebook, **metadata})
    elif extra_state["frozen"] or extra_state["initialized"]:
        raise ValueError("continuous Q-3D student cannot carry an initialized VQ codebook")
    model.sdf_head = None
    model.continuous_feature_head = None
    try:
        model.load_state_dict(dict(state), strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Q-3D deploy state mismatch: {exc}") from exc
    if model.codebook_metadata() != metadata:
        raise ValueError("Q-3D deploy codebook metadata changed during strict state loading")
    model.eval()
    model.to(device)
    return model


def _assert_teacher_free_imports(module_names: Sequence[str] | None = None) -> None:
    names = sys.modules if module_names is None else module_names
    forbidden = [name for name in names if (
        name in {
            "self_audit.candidate_q3d.teacher",
            "self_audit.candidate_q3d.teacher_resenc",
            "self_audit.candidate_a3d.teacher",
        }
        or name.endswith((
            ".candidate_q3d.teacher",
            ".candidate_q3d.teacher_resenc",
            ".candidate_a3d.teacher",
        ))
        or name == "dynamic_network_architectures"
        or name.startswith("dynamic_network_architectures.")
        or name == "nnssl"
        or name.startswith("nnssl.")
    )]
    if forbidden:
        raise RuntimeError(f"student-only cold load imported teacher modules: {sorted(forbidden)}")


def _load_payload(path: str | Path, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Q-3D checkpoint is not a file: {checkpoint}")
    actual_sha = _sha256_file(checkpoint)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError("Q-3D source checkpoint SHA-256 mismatch")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Q-3D checkpoint must be safe for weights_only loading") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("Q-3D checkpoint payload must be a mapping")
    return dict(payload), actual_sha


def _cost_fields(cost_report: Mapping[str, Any] | None, parameter_count: int) -> dict[str, Any]:
    report = cost_report if isinstance(cost_report, Mapping) else {}
    fields = {
        "parameters": {"status": "COMPUTED", "value": parameter_count, "unit": "parameters"},
        "flops": report.get("flops", {"status": "NOT_RUN", "reason": "no profile report supplied"}),
        "peak_vram_mb": report.get("peak_vram_mb", {"status": "NOT_RUN", "reason": "no profile report supplied"}),
        "gpu_latency_ms_per_volume": report.get("gpu_latency_ms_per_volume", {
            "status": "NOT_RUN", "reason": "GPU profile not supplied"}),
        "cpu_latency_ms_per_volume": report.get("cpu_latency_ms_per_volume", {
            "status": "NOT_RUN", "reason": "CPU profile not supplied"}),
    }
    return fields


def export_checkpoint(checkpoint_path: str | Path, output_path: str | Path, *,
                      expected_checkpoint_sha256: str | None = None,
                      config_path: str | Path | None = None,
                      code_commit: str | None = None,
                      cost_report: Mapping[str, Any] | None = None,
                      exploratory_quality_waiver: str | None = None,
                      ) -> tuple[Path, dict[str, Any], torch.nn.Module]:
    """Build a deploy-only artifact from a frozen Q-3D training checkpoint."""
    supplied_waiver = _supplied_waiver(exploratory_quality_waiver)
    payload, checkpoint_sha = _load_payload(checkpoint_path, expected_checkpoint_sha256)
    if payload.get("schema") in {
        "self_audit.candidate_q3d.resume.v1", "self_audit.candidate_q3d.resume.v2",
    }:
        version = payload["schema"].rsplit(".", 1)[-1]
        raise ValueError(f"Q-3D resume-{version} checkpoints are retired; export requires the current resume-v3 schema")
    raw_config = payload.get("config", payload.get("configuration"))
    config_sha256 = None
    engine_config_sha256 = payload.get("config_sha256")
    config_recipe = None
    if config_path is not None:
        config_file = Path(config_path).resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"Q-3D source config is not a file: {config_file}")
        config_sha256 = _sha256_file(config_file)
        config_recipe = config_file.stem
        try:
            loaded_config = json.loads(config_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Q-3D source config must be valid JSON") from exc
        if not isinstance(loaded_config, Mapping):
            raise TypeError("Q-3D source config must be a JSON object")
        if payload.get("schema") in ENGINE_RESUME_SCHEMAS:
            from .config import validate_config

            if not isinstance(engine_config_sha256, str) or len(engine_config_sha256) != 64:
                raise ValueError("Q-3D engine checkpoint is missing its canonical config SHA-256")
            try:
                normalized_config = validate_config(loaded_config, real_run=False)
                normalized_sha256 = _canonical_object_sha256(normalized_config)
                normalized_payload_config = (
                    validate_config(raw_config, real_run=False)
                    if isinstance(raw_config, Mapping) else None
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("Q-3D engine source config is not a valid declared recipe") from exc
            if normalized_sha256 != engine_config_sha256:
                raise ValueError("Q-3D source config canonical SHA-256 differs from the training checkpoint")
            if normalized_payload_config is not None and normalized_payload_config != normalized_config:
                raise ValueError("Q-3D checkpoint config differs from the supplied source config")
        elif raw_config is not None and dict(raw_config) != dict(loaded_config):
            raise ValueError("Q-3D checkpoint config differs from the supplied source config")
        raw_config = loaded_config
    if not isinstance(raw_config, Mapping):
        raise TypeError("Q-3D checkpoint must include config or be paired with --config")
    if payload.get("schema") in ENGINE_RESUME_SCHEMAS:
        if config_path is None:
            raise ValueError("Q-3D engine resume checkpoint export requires the exact --config recipe")
        if expected_checkpoint_sha256 is None:
            raise ValueError("Q-3D engine resume checkpoint export requires its exact source SHA-256")
        checkpoint_file = Path(checkpoint_path).resolve()
        sidecar = checkpoint_file.with_suffix(checkpoint_file.suffix + ".sha256")
        if not sidecar.is_file():
            raise ValueError("Q-3D engine resume checkpoint is missing its SHA-256 sidecar")
        sidecar_parts = sidecar.read_text(encoding="ascii").split()
        if (len(sidecar_parts) != 2 or sidecar_parts[0] != checkpoint_sha
                or sidecar_parts[1] != checkpoint_file.name):
            raise ValueError("Q-3D engine resume checkpoint sidecar SHA-256 mismatch")
    exploratory = _checkpoint_exploratory(payload, supplied_waiver)
    state = _extract_state(payload)
    source_state = _source_student_state(payload)
    head_channels = _continuous_head_channels(source_state)
    config = _minimal_student_config(raw_config, state, continuous_head_channels=head_channels)
    if config["arm"] == "C1":
        # C1's adapter feeds only the saved continuous alignment head; neither
        # value participates in eval-mode logits, so both training-only paths
        # are absent from the student-only deploy graph.
        state = {key: value for key, value in state.items()
                 if not key.startswith("adapter.")}
    source_codebook = state["_extra_state"]
    tokenizer_artifact = payload.get("tokenizer_artifact")
    if source_codebook["frozen"] and isinstance(tokenizer_artifact, Mapping):
        for field in ("version", "sha256", "size", "dimension", "initialized", "frozen"):
            if tokenizer_artifact.get(field) != source_codebook.get(field):
                raise ValueError(f"Q-3D tokenizer artifact disagrees with student metadata: {field}")
    deploy_model = _construct_deploy_model(config, state)
    deploy_state = {
        key: (dict(value) if key == "_extra_state" else value.detach().cpu().contiguous())
        for key, value in deploy_model.state_dict().items()
    }
    if any("sdf" in key.lower() or "teacher" in key.lower() or "cache" in key.lower()
           for key in deploy_state):
        raise ValueError("deploy model still contains a train-only or teacher state key")
    if any("continuous_feature_head" in key.lower() for key in deploy_state):
        raise ValueError("deploy model still contains the train-only C1 feature head")
    arm = config["arm"]
    codebook_meta = deploy_model.codebook_metadata()
    if deploy_model.use_vq:
        if not codebook_meta["initialized"] or not codebook_meta["frozen"]:
            raise ValueError("Q-3D VQ export requires frozen C0 version provenance")
        c0 = {"status": "COMPUTED", **codebook_meta}
    else:
        c0 = {"status": "NOT_APPLICABLE_CONTINUOUS_ARM", **codebook_meta}
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model_parameters = sum(parameter.numel() for parameter in deploy_model.parameters())
    artifact = {
        "schema": EXPORT_SCHEMA,
        "config": config,
        "state_dict": deploy_state,
        "metadata": {
            "arm": arm,
            "deploy_student_arm": deploy_model.arm,
            "source_arm": config["source_arm"],
            "source_config_name": config_recipe,
            "class_mapping": dict(CLASS_MAPPING),
            "input_contract": "[1,1,D,H,W] float32 in [0,1] with native full_volume geometry",
            "output_contract": "[1,4,D,H,W] raw segmentation logits on native grid",
            "codebook_c0": c0,
            "source_checkpoint_sha256": checkpoint_sha,
            "source_config_sha256": config_sha256,
            "engine_config_sha256": engine_config_sha256,
            "deploy_parameter_count": int(model_parameters),
        },
    }
    if exploratory is not None:
        artifact["metadata"].update({
            key: copy.deepcopy(exploratory[key]) for key in _EXPLORATORY_METADATA_KEYS
        })
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(artifact, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    artifact_sha = _sha256_file(output)
    if cost_report is not None and cost_report.get("artifact_sha256") != artifact_sha:
        raise ValueError("Q-3D measured cost report is not bound to these exact artifact bytes")
    if code_commit is None:
        repo = Path(__file__).resolve().parents[2]
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                   check=False, capture_output=True, text=True)
        code_commit = completed.stdout.strip() if completed.returncode == 0 else "NOT_RECORDED"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                               check=False, capture_output=True, text=True)
        code_dirty = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    else:
        code_dirty = None
    manifest = {
        "schema": "q3d.export_manifest.v1",
        "artifact": output.name,
        "artifact_sha256": artifact_sha,
        "artifact_bytes": output.stat().st_size,
        "source_checkpoint_sha256": checkpoint_sha,
        "source_config_sha256": config_sha256,
        "engine_config_sha256": engine_config_sha256,
        "code_commit": code_commit or "NOT_RECORDED",
        "code_dirty_at_export": code_dirty,
        "arm": arm,
        "deploy_student_arm": deploy_model.arm,
        "source_arm": config["source_arm"],
        "source_config_name": config_recipe,
        "class_mapping": dict(CLASS_MAPPING),
        "parameters": {"status": "COMPUTED", "value": int(model_parameters)},
        "C0": c0,
        "cost": _cost_fields(cost_report, int(model_parameters)),
        "cold_load": {"status": "NOT_RUN", "reason": "three native DEV volumes were not supplied"},
        "input_contract": {
            "shape": "[1,1,D,H,W]",
            "intensity": "finite float in [0,1]",
            "geometry": ["shape", "spacing_zyx", "affine_zyx", "context=full_volume"],
        },
        "output_contract": "raw 4-class logits on the native full-volume grid",
        "software_only": True,
    }
    if exploratory is not None:
        manifest.update(copy.deepcopy(exploratory))
        manifest["exploratory_quality_waiver_events"] = manifest.pop("events")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    _write_json_atomic(manifest_path, manifest)
    return output, manifest, deploy_model


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def load_export(artifact_path: str | Path, *, expected_sha256: str | None = None,
                device: str | torch.device = "cpu",
                dtype: torch.dtype | None = None,
                exploratory_quality_waiver: str | None = None,
                ) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Cold-load a student-only artifact; no teacher factory is available here.

    Exploratory (quality-waived) artifacts load only with the exact explicit
    waiver REASON; default loads reject them.
    """
    supplied_waiver = _supplied_waiver(exploratory_quality_waiver)
    path = Path(artifact_path).resolve()
    artifact_sha = _sha256_file(path)
    if expected_sha256 is not None and artifact_sha != expected_sha256:
        raise ValueError("Q-3D exported artifact SHA-256 mismatch")
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Q-3D artifact is not safe weights-only data") from exc
    if not isinstance(artifact, Mapping) or set(artifact) != {"schema", "config", "state_dict", "metadata"}:
        raise ValueError("Q-3D artifact must contain exactly schema/config/state_dict/metadata")
    if artifact.get("schema") != EXPORT_SCHEMA:
        raise ValueError("unsupported Q-3D export schema")
    if not _is_state_dict(artifact["state_dict"]):
        raise ValueError("Q-3D deploy state must contain tensors and canonical codebook extra state only")
    if any("sdf" in key.lower() or "teacher" in key.lower() or "cache" in key.lower()
           for key in artifact["state_dict"]):
        raise ValueError("Q-3D deploy artifact contains a forbidden state key")
    if any("continuous_feature_head" in key.lower() for key in artifact["state_dict"]):
        raise ValueError("Q-3D deploy artifact contains the train-only C1 feature head")
    if not isinstance(artifact["metadata"], Mapping):
        raise ValueError("Q-3D artifact metadata must be a mapping")
    _artifact_exploratory(artifact["metadata"], supplied_waiver)
    modules_before = set(sys.modules)
    model = _construct_deploy_model(
        artifact["config"], artifact["state_dict"], device=device, dtype=dtype
    )
    _assert_teacher_free_imports(set(sys.modules) - modules_before)
    metadata = dict(artifact["metadata"])
    metadata["artifact_sha256"] = artifact_sha
    return model, metadata


def _read_verification_volume(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    from self_audit.candidate_q3d.evaluation import validate_native_image

    with np.load(path, allow_pickle=False) as data:
        required = {"image", "shape_zyx", "spacing_zyx", "affine_zyx", "context"}
        if required - set(data.files):
            raise ValueError(f"verification input is missing geometry fields: {sorted(required - set(data.files))}")
        context = str(np.asarray(data["context"]).item())
        geometry = {"shape": tuple(int(v) for v in data["shape_zyx"].tolist()),
                    "spacing_zyx": data["spacing_zyx"].tolist(),
                    "affine_zyx": data["affine_zyx"].tolist(), "context": context}
        image = np.array(data["image"], copy=True)
    return validate_native_image(image, geometry)


def _runtime_dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    candidate = getattr(torch, name, None)
    if not isinstance(candidate, torch.dtype) or not candidate.is_floating_point:
        raise ValueError(f"unsupported floating-point dtype for Q-3D cold load: {dtype}")
    return name


def _runtime_dtype(name: str) -> torch.dtype:
    candidate = getattr(torch, name, None)
    if not isinstance(candidate, torch.dtype) or not candidate.is_floating_point:
        raise ValueError(f"unsupported floating-point dtype for Q-3D cold load: {name}")
    return candidate


def _cold_load_runtime(model: torch.nn.Module, images: Sequence[torch.Tensor]) -> dict[str, Any]:
    parameter_dtype = next(
        (parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()), None
    )
    if parameter_dtype is None:
        raise ValueError("Q-3D reference model has no floating-point parameter dtype")
    return {
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "default_dtype": _runtime_dtype_name(torch.get_default_dtype()),
        "model_dtype": _runtime_dtype_name(parameter_dtype),
        "input_dtypes": [_runtime_dtype_name(image.dtype) for image in images],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _configure_cold_load_runtime(runtime: Mapping[str, Any]) -> None:
    """Restore inference settings before the deploy model is constructed."""
    torch.set_num_threads(int(runtime["num_threads"]))
    torch.set_num_interop_threads(int(runtime["num_interop_threads"]))
    torch.set_default_dtype(_runtime_dtype(str(runtime["default_dtype"])))
    torch.use_deterministic_algorithms(
        bool(runtime["deterministic_algorithms"]),
        warn_only=bool(runtime["deterministic_warn_only"]),
    )
    torch.set_float32_matmul_precision(str(runtime["float32_matmul_precision"]))
    torch.backends.cudnn.deterministic = bool(runtime["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(runtime["cudnn_benchmark"])
    torch.backends.cudnn.allow_tf32 = bool(runtime["cudnn_allow_tf32"])
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime["cuda_matmul_allow_tf32"])


def _logits_sha256(logits: torch.Tensor) -> str:
    value = logits.detach().contiguous().cpu().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def verify_cold_load(model: torch.nn.Module, artifact_path: str | Path,
                     volume_paths: Sequence[str | Path], *, device: str = "cpu",
                     exploratory_quality_waiver: str | None = None) -> dict[str, Any]:
    """Run a fresh interpreter on exactly three declared native volumes and compare logits bitwise."""
    _supplied_waiver(exploratory_quality_waiver)
    if len(volume_paths) != 3:
        return {"status": "NOT_RUN_REQUIRES_EXACTLY_THREE_VOLUMES", "volumes": len(volume_paths)}
    artifact = Path(artifact_path).resolve()
    expected_hashes: list[str] = []
    model = model.to(device).eval()
    images: list[torch.Tensor] = []
    for volume_path in volume_paths:
        image, geometry = _read_verification_volume(volume_path)
        images.append(image)
        with torch.inference_mode():
            logits = model.infer_logits(image.to(device), geometry=geometry)
        if not torch.is_tensor(logits) or not torch.isfinite(logits).all():
            raise ValueError("Q-3D reference model returned invalid verification logits")
        expected_hashes.append(_logits_sha256(logits))
    runtime = _cold_load_runtime(model, images)
    code = r'''import json, os, sys
artifact_path, device, runtime_json, waiver_json, *volume_paths = sys.argv[1:]
runtime = json.loads(runtime_json)
waiver_reason = json.loads(waiver_json)
if runtime["cublas_workspace_config"] is None:
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
else:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(runtime["cublas_workspace_config"])
import torch
from self_audit.candidate_q3d.export import (
    _configure_cold_load_runtime, _logits_sha256, _read_verification_volume,
    _runtime_dtype, load_export,
)
_configure_cold_load_runtime(runtime)

model, metadata = load_export(artifact_path, device=device,
                              dtype=_runtime_dtype(runtime["model_dtype"]),
                              exploratory_quality_waiver=waiver_reason)
actual_input_dtypes = []
with torch.inference_mode():
    output = []
    for volume_path, input_dtype_name in zip(volume_paths, runtime["input_dtypes"], strict=True):
        image, geometry = _read_verification_volume(volume_path)
        image = image.to(device=device, dtype=_runtime_dtype(input_dtype_name))
        actual_input_dtypes.append(str(image.dtype).removeprefix("torch."))
        logits = model.infer_logits(image, geometry=geometry)
        output.append(_logits_sha256(logits))
forbidden = [name for name in sys.modules if (
    name in {
        "self_audit.candidate_q3d.teacher",
        "self_audit.candidate_q3d.teacher_resenc",
        "self_audit.candidate_a3d.teacher",
    }
    or name.endswith((
        ".candidate_q3d.teacher",
        ".candidate_q3d.teacher_resenc",
        ".candidate_a3d.teacher",
    ))
    or name == "dynamic_network_architectures"
    or name.startswith("dynamic_network_architectures.")
    or name == "nnssl"
    or name.startswith("nnssl.")
)]
if forbidden:
    raise RuntimeError("teacher module imported in cold-load process: " + repr(forbidden))
print(json.dumps({"hashes": output, "teacher_modules": forbidden,
                  "artifact_sha256": metadata["artifact_sha256"],
                  "exploratory": metadata.get("exploratory") is True,
                  "runtime": {"num_threads": torch.get_num_threads(),
                              "num_interop_threads": torch.get_num_interop_threads(),
                              "default_dtype": str(torch.get_default_dtype()).removeprefix("torch."),
                              "model_dtype": str(next(
                                  parameter.dtype for parameter in model.parameters()
                                  if parameter.is_floating_point()
                              )).removeprefix("torch."),
                              "input_dtypes": actual_input_dtypes,
                              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                              "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                              "cudnn_deterministic": torch.backends.cudnn.deterministic,
                              "cudnn_benchmark": torch.backends.cudnn.benchmark,
                              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                              "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                              "float32_matmul_precision": torch.get_float32_matmul_precision(),
                              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}}))
'''
    root = Path(__file__).resolve().parents[2]
    child_env = os.environ.copy()
    if runtime["cublas_workspace_config"] is None:
        child_env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        child_env["CUBLAS_WORKSPACE_CONFIG"] = str(runtime["cublas_workspace_config"])
    completed = subprocess.run(
        [sys.executable, "-c", code, str(artifact), device, json.dumps(runtime, sort_keys=True),
         json.dumps(exploratory_quality_waiver),
         *[str(Path(p).resolve()) for p in volume_paths]],
        cwd=root, env=child_env, check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Q-3D cold-load subprocess failed: {completed.stderr.strip()}")
    try:
        cold = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Q-3D cold-load subprocess returned malformed JSON") from exc
    expected_child_runtime = runtime
    if (cold.get("teacher_modules") or cold.get("hashes") != expected_hashes
            or cold.get("runtime") != expected_child_runtime):
        raise RuntimeError("Q-3D cold-load logits differ bitwise or imported a teacher module")
    result = {"status": "PASSED_BITWISE_THREE_NATIVE_VOLUMES",
              "volumes": 3, "logit_sha256": expected_hashes,
              "cold_load_artifact_sha256": cold.get("artifact_sha256"),
              "teacher_modules_imported": []}
    if cold.get("exploratory") is True:
        result["exploratory"] = True
    return result


__all__ = ["EXPORT_SCHEMA", "export_checkpoint", "load_export", "verify_cold_load"]
