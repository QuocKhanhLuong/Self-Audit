"""Immutable, provenance-checked Q-3D T/Q/P stage artifacts.

Stage artifacts deliberately contain only the teacher, tokenizer, and sealed
physiology cache needed by a later arm. Optimizer, student, sampler, and RNG
state stay local to a run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from . import CHECKPOINT_SCHEMA


_STAGES = ("T", "Q", "P")
_ARTIFACT_STATE_KEYS = {
    "schema",
    "data_identity",
    "completed_stages",
    "completed_updates_total",
    "teacher_state",
    "best_teacher_state",
    "best_teacher_score",
    "best_teacher_dev_per_class",
    "best_teacher_dev_update",
    "teacher_dev_history",
    "teacher_dev_fg_dice",
    "teacher_dev_per_class",
    "teacher_selected_dev_update",
    "teacher_terminal_dev_fg_dice",
    "teacher_terminal_dev_per_class",
    "teacher_dev_curve_still_improving",
    "dev_curve_still_improving",
    "metrics",
    "teacher_best_sha256",
    "tokenizer_state",
    "tokenizer_artifact",
    "gate",
    "gate_override",
    "synthetic_gate_override_used",
    "synthetic_gate_override_requested",
    "physiology_rows",
    "physiology_cache",
    "physiology_support_gate",
    "stage_artifacts",
    "stage_artifact_hashes",
    "shared_codebook_sha256",
    "shared_codebook_metadata",
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Q-3D stage artifact requires {name} to be a mapping")
    return value


def _submapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Q-3D stage artifact config is missing {name}")
    return value


def _manifest_digest(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    dataset = _submapping(config.get("dataset"), "dataset")
    manifests = dataset.get("manifests", {})
    manifests = manifests if isinstance(manifests, Mapping) else {}
    raw = manifests.get(name)
    if not isinstance(raw, Mapping):
        return {"status": "MISSING", "sha256": None}
    return {"status": raw.get("status"), "sha256": raw.get("sha256")}


def stage_identity(config: Mapping[str, Any], data_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the four-field campaign identity, checking actual data separately."""

    dataset = _submapping(config.get("dataset"), "dataset")
    protocol = _submapping(config.get("protocol"), "protocol")
    runtime = _submapping(config.get("runtime"), "runtime")
    actual = _mapping(data_identity, "data identity")
    dataset_name = dataset.get("name")
    variant = protocol.get("variant")
    k = protocol.get("k")
    seed = config.get("seed")
    if (dataset_name not in {"acdc", "mnms"} or variant not in {"A", "B"}
            or type(k) is not int or k < 1 or type(seed) is not int or seed < 0):
        raise ValueError("Q-3D stage identity has invalid dataset/protocol/K/seed")
    if protocol.get("seed", seed) != seed:
        raise ValueError("Q-3D config root and protocol seeds differ")
    if actual.get("protocol") != variant:
        raise ValueError("Q-3D stage data identity protocol differs from config")
    student_view = actual.get("student_view")
    if not isinstance(student_view, Mapping):
        raise ValueError("Q-3D stage data identity lacks a student manifest view")
    data_dataset = student_view.get("dataset")
    teacher_view = actual.get("teacher_view")
    if not isinstance(teacher_view, Mapping):
        raise ValueError("Q-3D stage data identity lacks a teacher manifest view")
    # Synthetic fixtures deliberately name their manifest dataset ``synthetic``
    # even when they exercise an ACDC/M&Ms recipe.  Permit that one mismatch
    # only in explicit synthetic mode, while sealing both actual views below.
    synthetic_fixture = (
        runtime.get("synthetic") is True
        and data_dataset == "synthetic"
        and teacher_view.get("dataset") == "synthetic"
    )
    if data_dataset != dataset_name and not synthetic_fixture:
        raise ValueError("Q-3D stage data identity dataset differs from config")
    if teacher_view.get("dataset") != data_dataset:
        raise ValueError("Q-3D stage teacher and student dataset identities differ")
    train_sha = _manifest_digest(config, "train").get("sha256")
    if train_sha is not None and student_view.get("manifest_sha256") != train_sha:
        raise ValueError("Q-3D stage data identity train manifest SHA differs from config")
    teacher_manifest_name = "teacher_train" if variant == "B" else "train"
    teacher_sha = _manifest_digest(config, teacher_manifest_name).get("sha256")
    if teacher_sha is not None and teacher_view.get("manifest_sha256") != teacher_sha:
        raise ValueError("Q-3D stage data identity teacher manifest SHA differs from config")
    return {
        "dataset": dataset_name,
        "protocol": variant,
        "k": k,
        "seed": seed,
    }


def _source_data_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    protocol = _submapping(config.get("protocol"), "protocol")
    runtime = _submapping(config.get("runtime"), "runtime")
    return {
        "train_manifest_sha256": _manifest_digest(config, "train").get("sha256"),
        "teacher_train_manifest_sha256": _manifest_digest(
            config, "teacher_train" if protocol.get("variant") == "B" else "train",
        ).get("sha256"),
        "development_manifest_sha256": _manifest_digest(config, "development").get("sha256"),
        "runtime_synthetic": runtime.get("synthetic"),
    }


def _stage_family(config: Mapping[str, Any], stage: str) -> str:
    """Keep scientifically separate Q/P source families from accidental reuse."""

    if stage == "T":
        return "q3d_shared_teacher_v1"
    arm = str(config.get("arm", "")).lower()
    return "q3d_o1_online_tokenizer_v1" if arm == "o1" else "q3d_frozen_teacher_tokenizer_v1"


def _measured_student_phase(config: Mapping[str, Any]) -> dict[str, Any]:
    from .alignment import student_grid_phase
    from .student import CandidateQ3DStudent

    # Source-config generation also happens after checkpoint RNG restoration.
    # Build the temporary CPU module without advancing the training RNG stream.
    with torch.random.fork_rng(devices=[]):
        student = CandidateQ3DStudent(config=config)
        step, offset = student_grid_phase(student)
    return {
        "step_zyx": [int(value) for value in step],
        "offset_zyx": [float(value) for value in offset],
    }


def _sampling_phase(config: Mapping[str, Any], teacher: nn.Module) -> dict[str, Any]:
    from .teacher_features import teacher_feature_phase

    model = _submapping(config.get("model"), "model")
    teacher_config = _submapping(model.get("teacher"), "model.teacher")
    source = teacher_config.get("feature_source")
    student_phase = getattr(teacher, "_q3d_student_phase", None)
    if student_phase is None:
        raise ValueError("Q-3D sampling phase is missing the live student grid phase")
    teacher_phase = teacher_feature_phase(
        teacher, str(source), synthetic=bool(config["runtime"]["synthetic"]),
    )

    def as_json(phase):
        step, offset = phase
        return {
            "step_zyx": [int(value) for value in step],
            "offset_zyx": [float(value) for value in offset],
        }

    measured_student = as_json(student_phase)
    if measured_student != _measured_student_phase(config):
        raise ValueError("Q-3D live teacher/student phase differs from the configured student modules")
    return {
        "teacher_feature_source": source,
        "teacher_phase": as_json(teacher_phase),
        "student_phase": measured_student,
    }


def _teacher_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    model = _submapping(config.get("model"), "model")
    teacher = _submapping(model.get("teacher"), "model.teacher")
    keys = (
        "architecture", "heads", "source_repo", "source_commit", "pretrained",
        "encoder_strides", "phase_offsets", "encoder_channels", "encoder_blocks",
        "dependency_pins",
    )
    return {key: copy.deepcopy(teacher.get(key)) for key in keys}


def _select(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: copy.deepcopy(mapping[key]) for key in keys if key in mapping}


def stage_source_config(config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Build an arm-independent source contract for T, Q, or P."""

    if stage not in _STAGES:
        raise ValueError("Q-3D stage source config only supports T, Q, or P")
    model = _submapping(config.get("model"), "model")
    teacher = _submapping(model.get("teacher"), "model.teacher")
    student = _submapping(model.get("student"), "model.student")
    runtime = _submapping(config.get("runtime"), "runtime")
    training = _submapping(config.get("training"), "training")
    losses = _submapping(config.get("losses"), "losses")
    tokenizer = _submapping(config.get("tokenizer"), "tokenizer")
    physiology = _submapping(config.get("physiology"), "physiology")
    from .config import resolve_stage_updates

    resolved_updates = resolve_stage_updates(
        config, smoke=runtime.get("synthetic") is True,
    )

    identity = {
        "dataset": config["dataset"]["name"],
        "protocol": config["protocol"]["variant"],
        "k": config["protocol"]["k"],
        "seed": config["seed"],
    }
    base: dict[str, Any] = {
        "stage": stage,
        "identity": identity,
        "data_identity": _source_data_identity(config),
    }
    if stage == "T":
        teacher_contract = _teacher_provenance(config)
        teacher_contract["model"] = _select(teacher, ("architecture", "heads"))
        base.update({
            "teacher": teacher_contract,
            "update_budget": resolved_updates.get("T"),
            "teacher_training": _select(training, (
                "teacher_warmup_fraction", "teacher_boundary_lr", "teacher_encoder_lr",
                "teacher_head_lr", "weight_decay", "gradient_clip_norm",
                "warmup_fraction", "checkpoint_selection", "checkpoint_evaluation_interval",
            )),
            "losses": _select(losses, (
                "region_ce_dice", "boundary_ce_dice", "balanced_sdf", "sdf_radius_mm",
                "sdf_near_boundary_mm", "sdf_near_far_alpha",
            )),
            "augmentation": copy.deepcopy(config.get("augmentation")),
        })
    elif stage == "Q":
        teacher_keys = (
            "architecture", "feature_stride", "feature_source", "feature_channels",
            "source_repo", "source_commit", "pretrained", "encoder_strides", "phase_offsets",
            "encoder_channels", "encoder_blocks", "dependency_pins",
        )
        # EPA attention/decoder choices do not alter the teacher feature grid
        # consumed by Q.  Bind only dimensions and the actual Q-stage phase.
        student_keys = (
            "architecture", "input_shape", "input_channels", "stage_channels", "z_stride",
            "native_grid", "crop", "resample",
        )
        tokenizer_contract = {key: copy.deepcopy(value) for key, value in tokenizer.items() if key != "enabled"}
        tokenizer_contract["synthetic_initialization_volumes"] = (
            copy.deepcopy(runtime.get("smoke_initialization_volumes"))
            if runtime.get("synthetic") is True else None
        )
        base.update({
            "source_family": _stage_family(config, stage),
            "teacher_feature_source": teacher.get("feature_source"),
            "teacher_feature_stride": copy.deepcopy(teacher.get("feature_stride")),
            "teacher_variant": "o1_online" if str(config.get("arm", "")).lower() == "o1" else "frozen_teacher",
            "teacher_source": _select(teacher, teacher_keys),
            "student_network_phase": {
                "network": _select(student, student_keys),
                "student_s4_grid_phase": _measured_student_phase(config),
                "teacher_feature_source": teacher.get("feature_source"),
                "teacher_feature_stride": teacher.get("feature_stride"),
                "teacher_feature_channels": teacher.get("feature_channels"),
                "contract": "q3d_native_index_phase_v1",
            },
            "tokenizer": tokenizer_contract,
            "update_budget": resolved_updates.get("Q"),
            "optimizer": _select(training, ("weight_decay", "gradient_clip_norm")),
            "losses": _select(losses, ("commitment_beta",)),
            "augmentation": copy.deepcopy(config.get("augmentation")),
        })
    else:
        physiology_contract = {
            key: copy.deepcopy(value)
            for key, value in physiology.items()
            if key != "cache_refresh_joint_epochs"
        }
        base.update({
            "source_family": _stage_family(config, stage),
            "teacher_feature_source": teacher.get("feature_source"),
            "teacher_feature_stride": copy.deepcopy(teacher.get("feature_stride")),
            "teacher_variant": "o1_online" if str(config.get("arm", "")).lower() == "o1" else "frozen_teacher",
            "teacher_source": _select(teacher, (
                "architecture", "heads", "source_repo", "source_commit", "pretrained",
                "feature_source", "feature_stride", "feature_channels", "encoder_strides",
                "phase_offsets", "encoder_channels", "encoder_blocks", "dependency_pins",
            )),
            "physiology": physiology_contract,
            "update_budget": resolved_updates.get("P"),
            "cache_training": {
                "stage_mode": copy.deepcopy(training.get("stage_modes", {}).get("P")),
                "stage_update_unit": copy.deepcopy(training.get("stage_update_units", {}).get("P")),
            },
        })
    return base


def _manifest_path(reference: Any, stage: str) -> Path:
    if isinstance(reference, (str, Path)):
        raw = reference
    elif isinstance(reference, Mapping):
        raw = None
        for key in ("manifest_path", "manifest", "path"):
            candidate = reference.get(key)
            if isinstance(candidate, (str, Path)):
                raw = candidate
                break
    else:
        raw = None
    if raw is None:
        raise ValueError(f"stage artifact {stage} must identify its manifest path")
    path = Path(raw).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"stage artifact {stage} manifest is not a file: {path}")
    if isinstance(reference, Mapping):
        expected_sha = reference.get("manifest_sha256", reference.get("sha256"))
        if expected_sha is not None:
            if not isinstance(expected_sha, str) or _file_sha256(path) != expected_sha:
                raise ValueError(f"stage artifact {stage} manifest file SHA-256 mismatch")
    return path


def _manifest_document(loaded: Any) -> Mapping[str, Any]:
    if not isinstance(loaded, Mapping):
        raise ValueError("stage manifest loader returned a non-mapping document")
    nested = loaded.get("manifest")
    if isinstance(nested, Mapping):
        return nested
    return loaded


def _manifest_field(document: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in document:
            return document[key]
    for parent_key in ("checkpoint", "artifact", "files"):
        parent = document.get(parent_key)
        if isinstance(parent, Mapping):
            for key in keys:
                if key in parent:
                    return parent[key]
            entry = parent.get("checkpoint")
            if isinstance(entry, Mapping):
                for key in keys:
                    if key in entry:
                        return entry[key]
    return None


def _checkpoint_details(document: Mapping[str, Any], manifest_path: Path, loaded: Any) -> tuple[Path, str]:
    path_value = _manifest_field(document, (
        "checkpoint_path", "checkpoint", "artifact_path", "path",
    ))
    sha_value = _manifest_field(document, (
        "checkpoint_sha256", "checkpoint_file_sha256", "sha256", "file_sha256",
    ))
    content_hashes = document.get("content_hashes")
    if isinstance(content_hashes, Mapping):
        if sha_value is None:
            sha_value = content_hashes.get("checkpoint_sha256")
        if path_value is None:
            path_value = content_hashes.get("checkpoint_path")
    if isinstance(path_value, Mapping):
        sha_value = sha_value or path_value.get("sha256")
        path_value = path_value.get("path")
    if path_value is None and isinstance(loaded, Mapping):
        path_value = loaded.get("checkpoint_path")
    if not isinstance(path_value, (str, Path)):
        raise ValueError("stage manifest has no checkpoint path")
    checkpoint = Path(path_value).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = manifest_path.parent / checkpoint
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_file() or not isinstance(sha_value, str):
        raise ValueError("stage manifest checkpoint path or SHA-256 is malformed")
    actual_sha = _file_sha256(checkpoint)
    if actual_sha != sha_value:
        raise ValueError(f"Q-3D {document.get('stage')} stage checkpoint SHA-256 mismatch")
    return checkpoint, actual_sha


def _load_payload(checkpoint_path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch builds
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Q-3D stage checkpoint must contain a mapping")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Q-3D stage artifact checkpoint schema mismatch")
    return dict(payload)


def _codebook_sha256(tokenizer_state: Mapping[str, Any]) -> str:
    codebook = tokenizer_state.get("C0")
    if not isinstance(codebook, Tensor) or not torch.is_floating_point(codebook):
        raise ValueError("Q-stage artifact lacks the frozen C0 tensor")
    payload = codebook.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _artifact_ref(
    stage: str,
    manifest_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "identity": copy.deepcopy(dict(identity)),
    }


def _dependency_manifest_refs(state: Mapping[str, Any], dependencies: tuple[str, ...]) -> dict[str, dict[str, str]]:
    from .artifacts import stage_manifest_file_ref

    result: dict[str, dict[str, str]] = {}
    stored = state.get("stage_artifacts", {})
    if not isinstance(stored, Mapping):
        raise ValueError("Q-3D stage state has no verified dependency manifest references")
    for stage in dependencies:
        ref = stored.get(stage)
        if isinstance(ref, Mapping):
            manifest_path = ref.get("manifest_path")
            manifest_sha = ref.get("manifest_sha256")
        else:
            manifest_path = ref
            manifest_sha = None
        if not isinstance(manifest_path, str):
            raise ValueError(f"Q-3D {stage} dependency manifest reference is missing")
        path = Path(manifest_path).expanduser().resolve(strict=True)
        manifest_ref = stage_manifest_file_ref(path)
        actual_sha = manifest_ref["sha256"]
        if manifest_sha is not None and manifest_sha != actual_sha:
            raise ValueError(f"Q-3D {stage} dependency manifest SHA-256 changed")
        result[stage] = manifest_ref
    return result


def _loaded_stage(
    stage: str,
    reference: Any,
    *,
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    data_identity: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any], dict[str, Any], dict[str, Any]]:
    from .artifacts import load_stage_manifest
    from .config import arm_settings

    manifest_path = _manifest_path(reference, stage)
    source_config = stage_source_config(config, stage)
    if stage == "Q" and not arm_settings(str(config.get("arm", ""))).get("tokenizer"):
        # C1 consumes Q's frozen metadata and cache lineage but has no Q
        # producer settings of its own. Match the whole producer contract while
        # accepting its explicitly recorded synthetic initialization count.
        loaded = load_stage_manifest(manifest_path, identity=identity)
        producer_source = loaded.get("source_config") if isinstance(loaded, Mapping) else None
        if not isinstance(producer_source, Mapping):
            raise ValueError("shared Q stage manifest lacks its pinned source config")
        expected_source = copy.deepcopy(source_config)
        producer_contract = copy.deepcopy(dict(producer_source))
        expected_contract = expected_source
        producer_tokenizer = producer_contract.get("tokenizer")
        expected_tokenizer = expected_contract.get("tokenizer")
        if not isinstance(producer_tokenizer, Mapping) or not isinstance(expected_tokenizer, Mapping):
            raise ValueError("shared Q source config lacks tokenizer provenance")
        producer_tokenizer = dict(producer_tokenizer)
        expected_tokenizer = dict(expected_tokenizer)
        init_count = producer_tokenizer.pop("synthetic_initialization_volumes", None)
        expected_tokenizer.pop("synthetic_initialization_volumes", None)
        producer_contract["tokenizer"] = producer_tokenizer
        expected_contract["tokenizer"] = expected_tokenizer
        if producer_contract != expected_contract:
            raise ValueError("shared Q stage source config differs from the tokenizerless consumer")
        runtime = config.get("runtime", {})
        if runtime.get("synthetic") is not True:
            raise ValueError("tokenizerless Q metadata sharing is restricted to the same synthetic mode")
        if init_count is not None and (type(init_count) is not int or not 1 <= init_count <= 200):
            raise ValueError("shared Q source has an invalid synthetic initialization volume override")
    else:
        loaded = load_stage_manifest(
            manifest_path, identity=identity, source_config=source_config,
        )
    document = _manifest_document(loaded)
    manifest_stage = document.get("stage")
    if manifest_stage != stage:
        raise ValueError(f"stage manifest mismatch: expected {stage}, got {manifest_stage!r}")
    checkpoint_path, checkpoint_sha = _checkpoint_details(document, manifest_path, loaded)
    content_hashes = document.get("content_hashes", {})
    if not isinstance(content_hashes, Mapping):
        raise ValueError(f"{stage} stage manifest content_hashes must be a mapping")
    expected_data_identity_sha = _canonical_sha256(data_identity)
    if content_hashes.get("data_identity_sha256") != expected_data_identity_sha:
        raise ValueError(f"{stage} stage artifact data identity hash differs")
    payload = _load_payload(checkpoint_path)
    if payload.get("stage_artifact_stage") != stage:
        raise ValueError(f"{stage} stage checkpoint does not identify its completed stage")
    completed = payload.get("completed_stages")
    if not isinstance(completed, list) or stage not in completed:
        raise ValueError(f"{stage} stage checkpoint is not complete")
    ref = _artifact_ref(stage, manifest_path, checkpoint_path, checkpoint_sha, identity)
    return checkpoint_path, content_hashes, payload, ref


def _validate_tokenizer_gate(
    config: Mapping[str, Any], state: Mapping[str, Any], *, explicit_override_requested: bool,
) -> None:
    gate = state.get("gate")
    if not isinstance(gate, Mapping) or type(gate.get("passed")) is not bool:
        raise ValueError("Q stage artifact is missing its measured tokenizer gate")
    if gate["passed"]:
        return
    runtime = config.get("runtime", {})
    override = state.get("gate_override")
    allowed = bool(
        isinstance(runtime, Mapping)
        and runtime.get("synthetic") is True
        and explicit_override_requested
        and state.get("synthetic_gate_override_used") is True
        and isinstance(override, Mapping)
        and override.get("status") == "SYNTHETIC_TEST_OVERRIDE"
        and override.get("continuation_allowed") is True
    )
    if not allowed:
        raise RuntimeError("Q-3D tokenizer gate failed; shared Q artifact cannot enter J")


def _validate_physiology_gate(state: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    gate = state.get("physiology_support_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("P stage artifact is missing its measured physiology support gate")
    if gate.get("passed") is True:
        return
    runtime = config.get("runtime", {})
    if not (
        isinstance(runtime, Mapping)
        and runtime.get("synthetic") is True
        and state.get("synthetic_gate_override_requested") is True
        and gate.get("override_used") is True
    ):
        raise RuntimeError("Q-3D physiology gate failed; shared P artifact cannot enter J")


def _state_mapping_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash a serialized module state with the engine's state-hash contract."""

    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, Tensor):
            tensor = value.detach().cpu().reshape(-1).clone(memory_format=torch.contiguous_format)
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(_canonical_json_bytes(list(value.shape)))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(value, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(value, key=lambda item: str(item)):
                update(str(key))
                update(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(b"tuple\0" if isinstance(value, tuple) else b"list\0")
            digest.update(_canonical_json_bytes(len(value)))
            for item in value:
                update(item)
        elif value is None or isinstance(value, (str, bool, int, float)):
            digest.update(b"json\0")
            digest.update(_canonical_json_bytes(value))
        else:
            raise ValueError(f"unsupported Q-3D state value: {type(value).__name__}")

    for name, value in sorted(state_dict.items()):
        digest.update(str(name).encode("utf-8"))
        update(value)
    return digest.hexdigest()


def _validate_dependency_refs(
    loaded: Mapping[str, tuple[Path, Mapping[str, Any], dict[str, Any], dict[str, Any]]],
) -> None:
    for stage, (_path, _hashes, _payload, reference) in loaded.items():
        manifest_doc = json.loads(Path(reference["manifest_path"]).read_text(encoding="utf-8"))
        dependencies = manifest_doc.get("dependencies")
        if not isinstance(dependencies, Mapping):
            raise ValueError(f"{stage} stage manifest dependency map is malformed")
        if stage == "T" and dependencies:
            raise ValueError("T stage manifest cannot have dependencies")
        if stage == "Q" and set(dependencies) != {"T"}:
            raise ValueError("Q stage manifest must depend on T")
        if stage == "P" and ("T" not in dependencies or not set(dependencies).issubset({"T", "Q"})):
            raise ValueError("P stage manifest must depend on T and optionally Q")
        for dependency_stage, dependency in dependencies.items():
            parent_ref = loaded.get(dependency_stage)
            if parent_ref is None or not isinstance(dependency, Mapping):
                raise ValueError(f"{stage} stage manifest has an unavailable {dependency_stage} dependency")
            expected = parent_ref[3]
            if (dependency.get("path"), dependency.get("sha256")) != (
                expected["manifest_path"], expected["manifest_sha256"],
            ):
                raise ValueError(f"{stage} stage manifest does not pin the supplied {dependency_stage} artifact")


def _assert_fresh_state(state: Mapping[str, Any]) -> None:
    existing_refs = state.get("stage_artifacts")
    if isinstance(existing_refs, Mapping) and existing_refs:
        return
    progressed = bool(
        state.get("saved_at") is not None
        or state.get("student_state") is not None
        or state.get("completed_stages")
        or int(state.get("j_optimizer_updates", 0) or 0) > 0
    )
    if progressed:
        raise ValueError("stage artifacts can only be imported into a fresh Q-3D run")


def _record_artifact_ref(state: dict[str, Any], reference: Mapping[str, Any]) -> None:
    stage = str(reference["stage"])
    refs = state.setdefault("stage_artifacts", {})
    hashes = state.setdefault("stage_artifact_hashes", {})
    existing = refs.get(stage) if isinstance(refs, Mapping) else None
    if isinstance(existing, Mapping) and (
        existing.get("manifest_path") != reference.get("manifest_path")
        or existing.get("manifest_sha256") != reference.get("manifest_sha256")
    ):
        raise ValueError(f"Q-3D {stage} state already references a different immutable artifact")
    refs[stage] = dict(reference)
    hashes[stage] = {
        "manifest_sha256": reference["manifest_sha256"],
        "checkpoint_sha256": reference["checkpoint_sha256"],
    }


def _unused_recovery_path(path: Path) -> Path:
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}.recovery-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _current_codebook_sha(
    state: Mapping[str, Any], tokenizer: nn.Module | None,
) -> tuple[str | None, Mapping[str, Any] | None]:
    if tokenizer is not None:
        tokenizer_state = tokenizer.state_dict()
        metadata = tokenizer.artifact_metadata()
        digest = _codebook_sha256(tokenizer_state)
        if metadata.get("sha256") != digest or metadata.get("frozen") is not True:
            raise ValueError("live Q-3D tokenizer is not the frozen shared C0 artifact")
        return digest, metadata
    metadata = state.get("shared_codebook_metadata") or state.get("tokenizer_artifact")
    digest = state.get("shared_codebook_sha256")
    if digest is None and isinstance(metadata, Mapping):
        digest = metadata.get("sha256")
    if digest is None:
        return None, metadata if isinstance(metadata, Mapping) else None
    if not isinstance(digest, str) or not isinstance(metadata, Mapping) or metadata.get("sha256") != digest:
        raise ValueError("shared Q-3D C0 metadata is missing or inconsistent")
    tokenizer_state = state.get("tokenizer_state")
    if isinstance(tokenizer_state, Mapping) and _codebook_sha256(tokenizer_state) != digest:
        raise ValueError("shared Q-3D C0 state differs from its metadata SHA-256")
    return digest, metadata


def _adopt_existing_export(
    config: Mapping[str, Any],
    data: Any,
    state: dict[str, Any],
    teacher: nn.Module,
    tokenizer: nn.Module | None,
    stage: str,
    *,
    identity: Mapping[str, Any],
    identity_data: Mapping[str, Any],
    source_config: Mapping[str, Any],
    manifest_path: Path,
    dependencies: Mapping[str, Mapping[str, str]],
    teacher_sha: str,
    codebook_sha: str | None,
) -> dict[str, Any]:
    from .artifacts import load_stage_manifest
    from . import engine

    checkpoint_path, hashes, payload, reference = _loaded_stage(
        stage, manifest_path, config=config, identity=identity,
        data_identity=identity_data,
    )
    document = load_stage_manifest(
        manifest_path, identity=identity, source_config=source_config,
    )
    if document.get("dependencies") != dict(dependencies):
        raise ValueError(f"existing {stage} artifact has different verified dependencies")
    if hashes.get("teacher_state_sha256") != teacher_sha or engine._state_sha256(teacher) != teacher_sha:
        raise ValueError(f"existing {stage} artifact teacher identity differs from live T")
    if hashes.get("checkpoint_sha256") not in {None, _file_sha256(checkpoint_path)}:
        raise ValueError(f"existing {stage} manifest checkpoint hash differs from its content hashes")

    if stage == "T":
        selected = payload.get("best_teacher_state")
        if not isinstance(selected, Mapping):
            selected = payload.get("teacher_state")
        if not isinstance(selected, Mapping) or _state_mapping_sha256(selected) != teacher_sha:
            raise ValueError("existing T artifact selected teacher state differs from live T")
        if state.get("teacher_best_sha256") != teacher_sha:
            raise ValueError("current completed T state differs from its existing artifact")
    elif stage == "Q":
        _validate_tokenizer_gate(
            config, payload,
            explicit_override_requested=bool(state.get("synthetic_gate_override_requested", False)),
        )
        tokenizer_state = payload.get("tokenizer_state")
        metadata = payload.get("tokenizer_artifact")
        if not isinstance(tokenizer_state, Mapping) or not isinstance(metadata, Mapping):
            raise ValueError("existing Q artifact lacks frozen tokenizer state and metadata")
        actual_codebook_sha = _codebook_sha256(tokenizer_state)
        if actual_codebook_sha != codebook_sha or hashes.get("codebook_sha256") != actual_codebook_sha:
            raise ValueError("existing Q artifact C0 differs from live shared metadata")
        expected_phase = _sampling_phase(config, teacher)
        sampling_phase = payload.get("sampling_phase")
        if (
            not isinstance(sampling_phase, Mapping)
            or dict(sampling_phase) != expected_phase
            or hashes.get("sampling_phase_sha256") != _canonical_sha256(expected_phase)
        ):
            raise ValueError("existing Q artifact sampling phase differs from live modules")
        if metadata.get("sha256") != actual_codebook_sha or metadata.get("frozen") is not True:
            raise ValueError("existing Q artifact C0 metadata is not frozen or SHA-pinned")
        if tokenizer is not None:
            live_tokenizer_state = tokenizer.state_dict()
            if _codebook_sha256(live_tokenizer_state) != actual_codebook_sha:
                raise ValueError("live Q-3D tokenizer differs from existing Q artifact")
            if _state_mapping_sha256(live_tokenizer_state) != _state_mapping_sha256(tokenizer_state):
                raise ValueError("live Q-3D tokenizer state differs from existing Q artifact")
    else:
        _validate_physiology_gate(payload, config)
        cache_state = payload.get("physiology_cache")
        rows = payload.get("physiology_rows")
        if not isinstance(cache_state, Mapping) or not isinstance(rows, Mapping):
            raise ValueError("existing P artifact lacks its sealed physiology cache and rows")
        cache_ref = cache_state.get("path")
        if not isinstance(cache_ref, str):
            raise ValueError("existing P artifact has no sealed physiology cache path")
        cache_file = Path(cache_ref).expanduser()
        if not cache_file.is_absolute():
            cache_file = checkpoint_path.parent / cache_file
        cache_file = cache_file.resolve(strict=True)
        file_sha = _file_sha256(cache_file)
        if cache_state.get("file_sha256") != file_sha or hashes.get("cache_file_sha256") != file_sha:
            raise ValueError("existing P artifact cache file SHA differs from its manifest")
        if hashes.get("cache_sha256") != cache_state.get("sha256"):
            raise ValueError("existing P artifact canonical cache SHA differs from its manifest")
        if cache_state.get("source_identity") != data.cache_source_identity():
            raise ValueError("existing P artifact cache belongs to a different TRAIN U cohort")
        if cache_state.get("teacher_sha256") != teacher_sha:
            raise ValueError("existing P artifact cache teacher SHA differs from live T")
        if cache_state.get("codebook_sha256") != codebook_sha:
            raise ValueError("existing P artifact cache C0 SHA differs from live/shared Q")
        if codebook_sha is not None and hashes.get("codebook_sha256") != codebook_sha:
            raise ValueError("existing P manifest codebook SHA differs from live/shared Q")
        live_cache = state.get("physiology_cache")
        if not isinstance(live_cache, Mapping) or cache_state.get("sha256") != live_cache.get("sha256"):
            raise ValueError("existing P artifact cache differs from the completed P run state")
        from .physiology import load_physiology_cache, summarize_physiology_cohort

        cache_document = load_physiology_cache(
            cache_file,
            teacher_sha256=teacher_sha,
            codebook_sha256=codebook_sha,
            protocol=str(config["protocol"]["variant"]),
            config=cache_state.get("config") if isinstance(cache_state.get("config"), Mapping) else None,
            source_identity=data.cache_source_identity(),
        )
        if cache_document.get("sha256") != cache_state.get("sha256"):
            raise ValueError("existing P artifact cache canonical SHA is invalid")
        cache_payload = cache_document.get("payload")
        if not isinstance(cache_payload, Mapping):
            raise ValueError("existing P artifact cache payload is malformed")
        cache_rows = {row["id"]: row for row in cache_payload.get("patients", [])}
        if dict(rows) != cache_rows or dict(rows) != dict(state.get("physiology_rows", {})):
            raise ValueError("existing P artifact rows differ from the exact sealed cache")
        summary = summarize_physiology_cohort([dict(rows[key]) for key in sorted(rows)])
        if cache_state.get("cohort_weighting") != summary or cache_payload.get("cohort_weighting") != summary:
            raise ValueError("existing P artifact cohort summary differs from exact rows")

    _record_artifact_ref(state, reference)
    return reference


def import_stage_artifacts(
    config: Mapping[str, Any],
    data: Any,
    state: dict[str, Any],
    teacher: nn.Module | None,
    tokenizer: nn.Module | None,
    output_dir: str | Path,
    artifacts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate and import shared T/Q/P artifacts without crossing run state."""

    if artifacts is None:
        return state
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("stage_artifacts must be a nonempty T/Q/P mapping")
    if set(artifacts) - set(_STAGES):
        raise ValueError("stage_artifacts may contain only T, Q, and P")
    from . import engine

    identity_data = engine._data_identity_without_labels(data)
    identity = stage_identity(config, identity_data)
    # Validate every supplied manifest first. No model or mutable state is
    # changed until the whole bundle, including its dependency hashes, passes.
    loaded: dict[str, tuple[Path, Mapping[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for stage in _STAGES:
        if stage in artifacts:
            loaded[stage] = _loaded_stage(
                stage, artifacts[stage], config=config, identity=identity,
                data_identity=identity_data,
            )
    if "Q" in loaded and "T" not in loaded:
        raise ValueError("Q stage artifact import requires its verified T artifact")
    if "P" in loaded and "T" not in loaded:
        raise ValueError("P stage artifact import requires its verified T artifact")
    if "P" in loaded:
        from .config import arm_settings

        if arm_settings(str(config.get("arm", ""))).get("tokenizer") and "Q" not in loaded:
            raise ValueError("VQ physiology P artifact import requires the matching Q artifact")

    _validate_dependency_refs(loaded)
    if not loaded:
        return state

    # The T checkpoint carries the VAL-selected teacher weights. Later stages
    # must be bitwise tied to that exact teacher snapshot.
    teacher_state: Mapping[str, Any] | None = None
    teacher_sha: str | None = None
    if "T" in loaded:
        _t_path, t_hashes, t_payload, _t_ref = loaded["T"]
        selected = t_payload.get("best_teacher_state")
        if not isinstance(selected, Mapping):
            selected = t_payload.get("teacher_state")
        if not isinstance(selected, Mapping):
            raise ValueError("T stage checkpoint lacks its selected teacher state")
        state_sha = t_payload.get("teacher_best_sha256")
        if not isinstance(state_sha, str):
            raise ValueError("T stage checkpoint lacks teacher_best_sha256")
        if t_hashes.get("teacher_state_sha256") != state_sha or _state_mapping_sha256(selected) != state_sha:
            raise ValueError("T manifest teacher state hash differs from its checkpoint")
        teacher_sha = state_sha
        teacher_state = copy.deepcopy(dict(selected))

    codebook_sha: str | None = None
    tokenizer_payload: Mapping[str, Any] | None = None
    if "Q" in loaded:
        _q_path, q_hashes, q_payload, _q_ref = loaded["Q"]
        _validate_tokenizer_gate(
            config, q_payload,
            explicit_override_requested=bool(state.get("synthetic_gate_override_requested", False)),
        )
        if not isinstance(q_payload.get("tokenizer_state"), Mapping):
            raise ValueError("Q stage artifact lacks tokenizer_state")
        tokenizer_payload = q_payload["tokenizer_state"]
        codebook_sha = _codebook_sha256(tokenizer_payload)
        artifact = q_payload.get("tokenizer_artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError("Q stage artifact lacks frozen tokenizer metadata")
        if artifact.get("sha256") != codebook_sha or artifact.get("frozen") is not True \
                or artifact.get("initialized") is not True:
            raise ValueError("Q stage C0 metadata does not match its frozen tokenizer state")
        if q_hashes.get("codebook_sha256") != codebook_sha:
            raise ValueError("Q manifest codebook SHA-256 differs from its tokenizer state")
        sampling_phase = q_payload.get("sampling_phase")
        expected_sampling_phase = _sampling_phase(config, teacher) if teacher is not None else None
        if (
            not isinstance(sampling_phase, Mapping)
            or expected_sampling_phase is None
            or dict(sampling_phase) != expected_sampling_phase
            or q_hashes.get("sampling_phase_sha256") != _canonical_sha256(expected_sampling_phase)
        ):
            raise ValueError("Q manifest sampling phase differs from the loaded teacher/student modules")
        if teacher_sha is not None and q_hashes.get("teacher_state_sha256") != teacher_sha:
            raise ValueError("Q stage artifact was built from a different teacher snapshot")

    if "P" in loaded:
        p_path, p_hashes, p_payload, _p_ref = loaded["P"]
        _validate_physiology_gate(p_payload, config)
        cache_state = p_payload.get("physiology_cache")
        rows = p_payload.get("physiology_rows")
        if not isinstance(cache_state, Mapping) or not isinstance(rows, Mapping):
            raise ValueError("P stage artifact lacks its sealed physiology cache and rows")
        cache_ref = cache_state.get("path")
        if not isinstance(cache_ref, str):
            raise ValueError("P stage physiology cache has no sealed path")
        cache_candidate = Path(cache_ref).expanduser()
        cache_file = cache_candidate if cache_candidate.is_absolute() else p_path.parent / cache_candidate
        cache_file = cache_file.resolve()
        if not cache_file.is_file():
            raise ValueError("P stage artifact physiology cache file is missing")
        cache_file_sha = _file_sha256(cache_file)
        if p_hashes.get("cache_file_sha256") != cache_file_sha:
            raise ValueError("P stage artifact physiology cache file SHA-256 mismatch")
        if cache_state.get("file_sha256") != cache_file_sha:
            raise ValueError("P checkpoint physiology cache file SHA-256 differs")
        cache_config = cache_state.get("config")
        source_identity = data.cache_source_identity()
        if cache_state.get("source_identity") != source_identity:
            raise ValueError("P stage physiology cache source cohort differs from current TRAIN U")
        expected_teacher_sha = teacher_sha or cache_state.get("teacher_sha256")
        if not isinstance(expected_teacher_sha, str) or cache_state.get("teacher_sha256") != expected_teacher_sha:
            raise ValueError("P stage physiology cache teacher identity differs from T")
        expected_codebook_sha = codebook_sha or cache_state.get("codebook_sha256")
        if expected_codebook_sha is not None and "Q" not in loaded:
            raise ValueError("P cache with a teacher codebook requires the matching Q metadata artifact")
        if cache_state.get("codebook_sha256") != expected_codebook_sha:
            raise ValueError("P stage physiology cache codebook identity differs from Q")
        if p_hashes.get("teacher_state_sha256") != expected_teacher_sha:
            raise ValueError("P manifest teacher state hash differs from its physiology cache")
        if codebook_sha is not None and p_hashes.get("codebook_sha256") != codebook_sha:
            raise ValueError("P manifest codebook SHA-256 differs from Q")
        if p_hashes.get("cache_sha256") != cache_state.get("sha256"):
            raise ValueError("P manifest canonical cache SHA-256 differs from checkpoint")
        if expected_codebook_sha is not None:
            manifest_doc = json.loads(Path(_p_ref["manifest_path"]).read_text(encoding="utf-8"))
            if "Q" not in manifest_doc.get("dependencies", {}):
                raise ValueError("P manifest with shared C0 must pin its Q dependency")
        from .physiology import load_physiology_cache, summarize_physiology_cohort

        document = load_physiology_cache(
            cache_file,
            teacher_sha256=expected_teacher_sha,
            codebook_sha256=expected_codebook_sha,
            protocol=str(config["protocol"]["variant"]),
            config=cache_config if isinstance(cache_config, Mapping) else None,
            source_identity=source_identity,
        )
        if document.get("sha256") != cache_state.get("sha256"):
            raise ValueError("P physiology cache canonical SHA-256 differs from checkpoint")
        payload = document.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("P physiology cache payload is malformed")
        cache_rows = {row["id"]: row for row in payload.get("patients", [])}
        if dict(rows) != cache_rows:
            raise ValueError("P physiology cache rows differ from the sealed cache payload")
        summary = summarize_physiology_cohort([dict(rows[key]) for key in sorted(rows)])
        if cache_state.get("cohort_weighting") != summary or payload.get("cohort_weighting") != summary:
            raise ValueError("P physiology cache cohort summary differs from sealed patient rows")

    existing_refs = state.get("stage_artifacts")
    if isinstance(existing_refs, Mapping) and existing_refs:
        candidate_refs = {stage: item[3] for stage, item in loaded.items()}
        if not set(candidate_refs).issubset(existing_refs):
            raise ValueError("resumed Q-3D state is missing a supplied stage artifact reference")
        stored_hashes = state.get("stage_artifact_hashes", {})
        for stage, candidate in candidate_refs.items():
            stored = existing_refs[stage]
            stored_path = stored.get("manifest_path") if isinstance(stored, Mapping) else stored
            stored_sha = stored.get("manifest_sha256") if isinstance(stored, Mapping) else None
            hash_entry = stored_hashes.get(stage) if isinstance(stored_hashes, Mapping) else None
            if isinstance(hash_entry, Mapping):
                stored_sha = hash_entry.get("manifest_sha256", stored_sha)
            elif isinstance(hash_entry, str):
                stored_sha = hash_entry
            if stored_path != candidate["manifest_path"] or stored_sha != candidate["manifest_sha256"]:
                raise ValueError("resumed Q-3D state stage artifact references differ from supplied refs")
        # A resumed J validates external refs and sealed cache bytes, but never
        # replaces its evolved model, optimizer, RNG, metrics, or cache state.
        return state

    if teacher is None:
        raise ValueError("Q-3D stage artifact import requires a teacher module")

    # Checkpoint states are transferred by allowlist only. No source student,
    # optimizer, EMA, sampler, metric history, or RNG state crosses arms.
    _assert_fresh_state(state)
    imported_stages: list[str] = []
    if "T" in loaded:
        _t_path, _t_hashes, t_payload, _t_ref = loaded["T"]
        state["teacher_state"] = copy.deepcopy(dict(teacher_state or {}))
        state["best_teacher_state"] = copy.deepcopy(dict(teacher_state or {}))
        state["teacher_best_sha256"] = teacher_sha
        state["best_teacher_score"] = t_payload.get("best_teacher_score")
        for key in (
            "best_teacher_dev_per_class", "best_teacher_dev_update",
            "teacher_dev_history", "teacher_dev_fg_dice", "teacher_dev_per_class",
            "teacher_selected_dev_update", "teacher_terminal_dev_fg_dice",
            "teacher_terminal_dev_per_class", "teacher_dev_curve_still_improving",
            "dev_curve_still_improving",
        ):
            if key in t_payload:
                state[key] = copy.deepcopy(t_payload[key])
        state["completed_updates_total"] = int(t_payload.get("completed_updates_total", 0) or 0)
        imported_stages.append("T")
    if "Q" in loaded:
        _q_path, _q_hashes, q_payload, _q_ref = loaded["Q"]
        if tokenizer is not None:
            state["tokenizer_state"] = copy.deepcopy(dict(tokenizer_payload or {}))
        state["tokenizer_artifact"] = copy.deepcopy(dict(q_payload["tokenizer_artifact"]))
        state["gate"] = copy.deepcopy(dict(q_payload["gate"]))
        state["gate_override"] = copy.deepcopy(q_payload.get("gate_override"))
        state["synthetic_gate_override_used"] = bool(q_payload.get("synthetic_gate_override_used", False))
        state["shared_codebook_sha256"] = codebook_sha
        state["shared_codebook_metadata"] = copy.deepcopy(dict(q_payload["tokenizer_artifact"]))
        state["completed_updates_total"] = max(
            int(state.get("completed_updates_total", 0) or 0),
            int(q_payload.get("completed_updates_total", 0) or 0),
        )
        imported_stages.append("Q")
    if "P" in loaded:
        p_path, _p_hashes, p_payload, _p_ref = loaded["P"]
        cache_state = copy.deepcopy(dict(p_payload["physiology_cache"]))
        cache_ref = Path(str(cache_state["path"]))
        if not cache_ref.is_absolute():
            cache_ref = p_path.parent / cache_ref
        cache_state["path"] = str(cache_ref.resolve())
        cache_state["file_sha256"] = _file_sha256(cache_state["path"])
        state["physiology_cache"] = cache_state
        state["physiology_rows"] = copy.deepcopy(dict(p_payload["physiology_rows"]))
        # This raw summary is stage independent. The runtime omega receipt is
        # reconstructed for the destination arm below.
        state["physiology_cache"]["cohort_weighting"] = copy.deepcopy(
            p_payload["physiology_cache"]["cohort_weighting"]
        )
        state["physiology_support_gate"] = copy.deepcopy(p_payload["physiology_support_gate"])
        state["physiology_cohort_weighting"] = None
        helper = engine._cohort_weighting_metadata
        weighting = helper(
            config, data, state,
            context="Q-3D shared stage artifact import",
            synthetic_override=bool(state.get("synthetic_gate_override_requested", False)),
        )
        state["physiology_cohort_weighting"] = weighting
        weighting = state["physiology_cohort_weighting"]
        if isinstance(weighting, Mapping) and not weighting.get("gate_passed") \
                and not weighting.get("override_used"):
            raise RuntimeError("Q-3D physiology cohort gate failed; imported P cannot enter J")
        imported_stages.append("P")

    # Load shared tensors only after every manifest and cache has validated.
    if "T" in loaded:
        teacher.load_state_dict(teacher_state, strict=True)
        if engine._state_sha256(teacher) != teacher_sha:
            raise ValueError("loaded Q-3D teacher differs from the selected T snapshot")
        state["teacher_state"] = copy.deepcopy(dict(teacher_state or {}))
        state["best_teacher_state"] = copy.deepcopy(dict(teacher_state or {}))
        state["teacher_best_sha256"] = teacher_sha
    if "Q" in loaded and tokenizer is not None:
        tokenizer.load_state_dict(tokenizer_payload, strict=True)
        actual_metadata = tokenizer.artifact_metadata()
        if actual_metadata.get("sha256") != codebook_sha or not actual_metadata.get("frozen"):
            raise ValueError("loaded Q-3D tokenizer is not the shared frozen C0 artifact")

    refs = {stage: loaded[stage][3] for stage in loaded}
    state["stage_artifacts"] = refs
    state["stage_artifact_hashes"] = {
        stage: {
            "manifest_sha256": ref["manifest_sha256"],
            "checkpoint_sha256": ref["checkpoint_sha256"],
        }
        for stage, ref in refs.items()
    }
    previous = list(state.get("completed_stages", []))
    state["completed_stages"] = [stage for stage in _STAGES if stage in set(previous).union(imported_stages)]
    state["stage"] = None
    state["stage_cursor"] = 0
    return state


def export_stage_artifact(
    config: Mapping[str, Any],
    data: Any,
    state: dict[str, Any],
    teacher: nn.Module | None,
    tokenizer: nn.Module | None,
    output_dir: str | Path,
    stage: str,
) -> dict[str, Any]:
    """Write one completed T/Q/P checkpoint and immutable manifest."""

    if stage not in _STAGES:
        raise ValueError("Q-3D stage artifact export only supports T, Q, or P")
    if stage not in state.get("completed_stages", []):
        raise ValueError(f"cannot export incomplete Q-3D stage {stage}")
    if teacher is None:
        raise ValueError("Q-3D shared stage artifact requires a teacher module")
    from . import engine
    from .artifacts import load_stage_manifest, write_stage_manifest

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity_data = engine._data_identity_without_labels(data)
    identity = stage_identity(config, identity_data)
    source_config = stage_source_config(config, stage)
    artifact_dir = output / "stage_artifacts" / stage
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / f"{stage}.checkpoint.pt"
    manifest_path = output / f"{stage}.artifact.json"
    teacher_sha = engine._state_sha256(teacher)
    codebook_sha: str | None = None
    tokenizer_meta: Mapping[str, Any] | None = None
    if stage in {"Q", "P"}:
        codebook_sha, tokenizer_meta = _current_codebook_sha(state, tokenizer)

    dependency_stages: tuple[str, ...]
    if stage == "T":
        dependency_stages = ()
    elif stage == "Q":
        dependency_stages = ("T",)
    else:
        cache_state = state.get("physiology_cache")
        cache_codebook_sha = cache_state.get("codebook_sha256") if isinstance(cache_state, Mapping) else None
        if cache_codebook_sha != codebook_sha:
            raise ValueError("P cache C0 SHA differs from live/shared Q metadata")
        dependency_stages = ("T", "Q") if codebook_sha is not None else ("T",)
    dependencies = _dependency_manifest_refs(state, dependency_stages)

    if manifest_path.exists():
        return _adopt_existing_export(
            config, data, state, teacher, tokenizer, stage,
            identity=identity, identity_data=identity_data, source_config=source_config,
            manifest_path=manifest_path, dependencies=dependencies,
            teacher_sha=teacher_sha, codebook_sha=codebook_sha,
        )

    # A completed checkpoint or cache may have been written immediately
    # before a crash. Preserve that orphan and publish a new immutable recovery
    # file rather than replacing bytes that could have escaped elsewhere.
    if checkpoint_path.exists():
        checkpoint_path = _unused_recovery_path(checkpoint_path)

    snapshot = copy.deepcopy(state)
    # Keep the stage outputs and immutable provenance only. In particular,
    # mutable arm-specific training state is never carried into a shared file.
    snapshot = {key: snapshot[key] for key in _ARTIFACT_STATE_KEYS if key in snapshot}
    snapshot["stage_artifact_stage"] = stage
    snapshot["stage"] = None
    snapshot["stage_cursor"] = 0
    snapshot["completed_stages"] = [name for name in _STAGES if name in state.get("completed_stages", [])]
    snapshot["data_identity_sha256"] = _canonical_sha256(identity_data)
    snapshot["config_sha256"] = _canonical_sha256(source_config)

    if stage == "T":
        # A T artifact is a teacher-only hand-off.  A full run may already
        # have instantiated a tokenizer, so explicitly remove Q/P state and
        # retain only the T metric history needed to verify VAL selection.
        snapshot.pop("tokenizer_state", None)
        snapshot.pop("tokenizer_artifact", None)
        snapshot.pop("gate", None)
        snapshot.pop("gate_override", None)
        snapshot.pop("synthetic_gate_override_used", None)
        snapshot.pop("synthetic_gate_override_requested", None)
        snapshot.pop("physiology_rows", None)
        snapshot.pop("physiology_cache", None)
        snapshot.pop("physiology_support_gate", None)
        snapshot.pop("shared_codebook_sha256", None)
        snapshot.pop("shared_codebook_metadata", None)
        snapshot.pop("stage_artifacts", None)
        snapshot.pop("stage_artifact_hashes", None)
        metrics = snapshot.get("metrics")
        if isinstance(metrics, list):
            snapshot["metrics"] = [
                copy.deepcopy(row) for row in metrics
                if isinstance(row, Mapping) and row.get("stage") == "T"
            ]
        else:
            snapshot["metrics"] = []
        selected = snapshot.get("best_teacher_state")
        if not isinstance(selected, Mapping):
            selected = snapshot.get("teacher_state")
        if not isinstance(selected, Mapping):
            raise ValueError("completed T stage has no selected teacher state")
        if state.get("teacher_best_sha256") != teacher_sha:
            raise ValueError("T stage teacher weights differ from the selected checkpoint")
        snapshot["teacher_state"] = copy.deepcopy(dict(selected))
        snapshot["best_teacher_state"] = copy.deepcopy(dict(selected))
    else:
        expected_teacher_sha = state.get("teacher_best_sha256")
        if expected_teacher_sha is not None and expected_teacher_sha != teacher_sha:
            raise ValueError(f"{stage} stage teacher differs from imported selected T")

    content_hashes: dict[str, str] = {
        "data_identity_sha256": _canonical_sha256(identity_data),
        "teacher_state_sha256": teacher_sha,
    }
    if stage == "Q":
        if tokenizer is not None:
            tokenizer_state = copy.deepcopy(tokenizer.state_dict())
        else:
            tokenizer_state = snapshot.get("tokenizer_state")
            tokenizer_meta = snapshot.get("shared_codebook_metadata") or snapshot.get("tokenizer_artifact")
        if not isinstance(tokenizer_state, Mapping) or not isinstance(tokenizer_meta, Mapping):
            raise ValueError(f"{stage} stage artifact lacks shared Q tokenizer metadata")
        if _codebook_sha256(tokenizer_state) != codebook_sha:
            raise ValueError(f"{stage} stage artifact C0 state differs from its shared metadata")
        if tokenizer_meta.get("sha256") != codebook_sha or tokenizer_meta.get("frozen") is not True:
            raise ValueError(f"{stage} stage artifact C0 is not the frozen Q codebook")
        snapshot["tokenizer_state"] = copy.deepcopy(dict(tokenizer_state))
        snapshot["tokenizer_artifact"] = copy.deepcopy(dict(tokenizer_meta))
        snapshot["shared_codebook_sha256"] = codebook_sha
        content_hashes["codebook_sha256"] = codebook_sha
        sampling_phase = _sampling_phase(config, teacher)
        snapshot["sampling_phase"] = copy.deepcopy(sampling_phase)
        content_hashes["sampling_phase_sha256"] = _canonical_sha256(sampling_phase)
        if stage == "Q":
            _validate_tokenizer_gate(
                config, snapshot,
                explicit_override_requested=bool(snapshot.get("synthetic_gate_override_requested", False)),
            )
    elif stage == "P":
        if codebook_sha is not None:
            if not isinstance(tokenizer_meta, Mapping) or tokenizer_meta.get("sha256") != codebook_sha:
                raise ValueError("P stage shared C0 metadata does not match the physiology cache")
            snapshot["shared_codebook_sha256"] = codebook_sha
            snapshot["shared_codebook_metadata"] = copy.deepcopy(dict(tokenizer_meta))
            content_hashes["codebook_sha256"] = codebook_sha

    if stage == "P":
        _validate_physiology_gate(snapshot, config)
        cache_state = snapshot.get("physiology_cache")
        rows = snapshot.get("physiology_rows")
        if not isinstance(cache_state, Mapping) or not isinstance(rows, Mapping):
            raise ValueError("completed P stage has no sealed physiology cache")
        source_path = Path(str(cache_state.get("path", ""))).expanduser()
        if not source_path.is_absolute():
            source_path = output / source_path
        source_path = source_path.resolve(strict=True)
        if not source_path.is_file():
            raise ValueError("completed P stage physiology cache file is missing")
        cache_file = artifact_dir / "P.physiology_cache.json"
        if cache_file.exists():
            cache_file = _unused_recovery_path(cache_file)
        fd, temporary_name = tempfile.mkstemp(
            prefix=cache_file.name + ".", suffix=".tmp", dir=str(cache_file.parent),
        )
        try:
            with source_path.open("rb") as source_stream, os.fdopen(fd, "wb") as target_stream:
                shutil.copyfileobj(source_stream, target_stream)
                target_stream.flush()
                os.fsync(target_stream.fileno())
            os.replace(temporary_name, cache_file)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        from .physiology import load_physiology_cache, summarize_physiology_cohort

        expected_source = data.cache_source_identity()
        cache_document = load_physiology_cache(
            cache_file,
            teacher_sha256=teacher_sha,
            codebook_sha256=content_hashes.get("codebook_sha256"),
            protocol=str(config["protocol"]["variant"]),
            config=cache_state.get("config") if isinstance(cache_state.get("config"), Mapping) else None,
            source_identity=expected_source,
        )
        if cache_document.get("sha256") != cache_state.get("sha256"):
            raise ValueError("completed P cache canonical SHA-256 differs from checkpoint")
        cache_payload = cache_document.get("payload")
        if not isinstance(cache_payload, Mapping):
            raise ValueError("completed P cache payload is malformed")
        cache_rows = {row["id"]: row for row in cache_payload.get("patients", [])}
        if dict(rows) != cache_rows:
            raise ValueError("completed P cache rows differ from its sealed cache file")
        summary = summarize_physiology_cohort([dict(rows[key]) for key in sorted(rows)])
        if cache_payload.get("cohort_weighting") != summary or cache_state.get("cohort_weighting") != summary:
            raise ValueError("completed P cache cohort summary differs from its exact rows")
        snapshot["physiology_rows"] = copy.deepcopy(dict(rows))
        snapshot["physiology_cache"] = copy.deepcopy(dict(cache_state))
        snapshot["physiology_cache"]["path"] = "P.physiology_cache.json"
        snapshot["physiology_cache"]["file_sha256"] = _file_sha256(cache_file)
        snapshot["physiology_cohort_weighting"] = None
        if cache_state.get("codebook_sha256") != codebook_sha:
            raise ValueError("P cache codebook SHA differs from shared Q metadata")
        content_hashes["cache_sha256"] = str(cache_state["sha256"])
        content_hashes["cache_file_sha256"] = _file_sha256(cache_file)

    engine._atomic_torch_save(snapshot, checkpoint_path)
    checkpoint_sha = _file_sha256(checkpoint_path)
    content_hashes["checkpoint_sha256"] = checkpoint_sha
    result = write_stage_manifest(
        manifest_path,
        stage=stage,
        checkpoint_path=checkpoint_path,
        identity=identity,
        source_config=source_config,
        dependencies=dependencies,
        content_hashes=content_hashes,
    )
    # Re-read to ensure the immutable bytes and all dependencies validate
    # before publishing the reference into the live run state.
    verified = load_stage_manifest(
        manifest_path, identity=identity, source_config=source_config,
    )
    document = _manifest_document(verified if isinstance(verified, Mapping) else result)
    actual_checkpoint, actual_sha = _checkpoint_details(document, manifest_path, verified)
    ref = _artifact_ref(stage, manifest_path, actual_checkpoint, actual_sha, identity)
    _record_artifact_ref(state, ref)
    return ref


__all__ = [
    "export_stage_artifact",
    "import_stage_artifacts",
    "stage_identity",
    "stage_source_config",
]
