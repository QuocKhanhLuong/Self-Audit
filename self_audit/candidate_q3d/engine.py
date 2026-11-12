"""Stage-wise Q-3D training engine.

The runner is deliberately separate from both Candidate A and A-3D.  It
reuses only the existing native A-3D data/teacher primitives and the Q-3D
student, tokenizer, alignment, augmentation, and physiology contracts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from . import CHECKPOINT_SCHEMA
from .alignment import align_teacher_to_student4
from .augmentation import augment_labeled, augment_unlabeled
from .data import Q3DData, Q3DJointSampler
from .losses import (
    codekd_loss,
    cohort_omega,
    commitment_loss,
    feature_kd_loss,
    mean_teacher_consistency_loss,
    q3d_teacher_loss,
    student_supervised_loss,
    weighted_class_balanced_reduce,
)


_CLASS_NAMES = ("BG", "RV", "MYO", "LV")
STATE_SCHEMA = CHECKPOINT_SCHEMA
_VQ_ARMS = {"v1", "v1u", "v0", "b1", "e1", "o1", "s1"}
_TEACHER_ARMS = {"c1", *_VQ_ARMS, "teacher_supervised"}
_STAGE_ORDER = ("T", "Q", "P", "J")
_MEAN_TEACHER_DECAY = 0.99
_STUDENT_CODE_COLLAPSE_WINDOW = 200


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, Tensor):
            shape = list(value.shape)
            tensor = value.detach().cpu().reshape(-1).clone(memory_format=torch.contiguous_format)
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(_canonical_json(shape))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(value, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(value, key=lambda item: str(item)):
                update(str(key))
                update(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(b"tuple\0" if isinstance(value, tuple) else b"list\0")
            digest.update(_canonical_json(len(value)))
            for item in value:
                update(item)
        elif value is None or isinstance(value, (str, bool, int, float)):
            digest.update(b"json\0")
            digest.update(_canonical_json(value))
        else:
            raise TypeError(f"unsupported non-tensor model state value: {type(value).__name__}")

    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        update(value)
    return digest.hexdigest()


def _seed_all(seed: int) -> tuple[random.Random, torch.Generator]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    spatial_rng = random.Random(seed + 11)
    intensity_rng = torch.Generator(device="cpu").manual_seed(seed + 29)
    return spatial_rng, intensity_rng


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if isinstance(numpy_state, Mapping):
        np.random.set_state((
            str(numpy_state["generator"]),
            numpy_state["keys"].detach().cpu().numpy().astype(np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        ))
    else:
        np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(value: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temp_path = stream.name
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    digest = _file_sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temp: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", dir=sidecar.parent,
            prefix=f".{sidecar.name}.", delete=False,
        ) as stream:
            sidecar_temp = stream.name
            stream.write(f"{digest}  {path.name}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(sidecar_temp, sidecar)
    finally:
        if sidecar_temp and os.path.exists(sidecar_temp):
            os.unlink(sidecar_temp)
    return digest


def _load_checkpoint(path: str | Path, *, config_sha256: str, data_identity: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Q-3D resume checkpoint does not exist: {target}")
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError("Q-3D resume checkpoint is missing its SHA-256 sidecar")
    sidecar_parts = sidecar.read_text(encoding="ascii").split()
    if len(sidecar_parts) != 2 or sidecar_parts[1] != target.name:
        raise ValueError("Q-3D checkpoint SHA-256 sidecar is malformed")
    if _file_sha256(target) != sidecar_parts[0]:
        raise ValueError("Q-3D checkpoint SHA-256 mismatch")
    payload = torch.load(target, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != STATE_SCHEMA:
        raise ValueError(
            "Q-3D resume checkpoint schema mismatch: expected resume.v3; "
            "resume.v2 uses the retired U sampling stream and cannot be resumed"
        )
    if payload.get("config_sha256") != config_sha256:
        raise ValueError("Q-3D resume config hash differs from checkpoint")
    if payload.get("data_identity") != dict(data_identity):
        raise ValueError("Q-3D resume checkpoint belongs to another TRAIN protocol/cohort")
    return dict(payload)


def _jsonl_append(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _autocast(device: torch.device):
    # CUDA training uses the specified bf16 path. CPU synthetic checks remain
    # fp32 because CPU bfloat16 kernels vary by the installed PyTorch build.
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _channels_last_3d(module: nn.Module) -> nn.Module:
    return module.to(memory_format=torch.channels_last_3d)


def _load_data(config: Mapping[str, Any], supplied: Any | None) -> Q3DData:
    from .data import validate_acdc_geometry, validate_manifest_geometry_file

    synthetic = config["runtime"]["synthetic"]
    dataset_name = config["dataset"].get("name")
    if supplied is not None:
        if isinstance(supplied, Q3DData):
            data = supplied
            # Supplied views may have been materialized by their caller; the
            # engine still checks their metadata before any further access.
            validate_acdc_geometry(
                data.student_data.manifest, synthetic=synthetic, dataset_name=dataset_name,
            )
            validate_acdc_geometry(
                data.teacher_data.manifest, synthetic=synthetic, dataset_name=dataset_name,
            )
        else:
            raise TypeError("data must be a Q3DData instance so protocol views remain explicit")
    else:
        from ..candidate_a3d.data import CandidateA3DData

        dataset = config["dataset"]
        manifests = dataset["manifests"]
        train_manifest = manifests["train"]
        validate_manifest_geometry_file(
            train_manifest["path"], train_manifest["sha256"],
            synthetic=synthetic, dataset_name=dataset_name,
        )
        protocol = config["protocol"]["variant"]
        teacher_manifest = None
        if protocol == "B":
            teacher_manifest = manifests.get("teacher_train")
            if not isinstance(teacher_manifest, Mapping):
                raise ValueError("protocol B requires a separately hash-pinned teacher_train manifest")
            validate_manifest_geometry_file(
                teacher_manifest["path"], teacher_manifest["sha256"],
                synthetic=synthetic, dataset_name=dataset_name,
            )
        # Validate both protocol views before constructing either loader.
        student_data = CandidateA3DData(
            train_manifest["path"], train_manifest["sha256"], synthetic=synthetic,
        )
        teacher_data = None
        if teacher_manifest is not None:
            teacher_data = CandidateA3DData(
                teacher_manifest["path"], teacher_manifest["sha256"], synthetic=synthetic,
            )
        data = Q3DData(student_data, protocol=protocol, teacher_data=teacher_data)
    if data.protocol != config["protocol"]["variant"]:
        raise ValueError("Q3DData protocol differs from the nested config")
    if len(data.labeled_patients) != config["protocol"]["k"]:
        raise ValueError("Q3DData student label budget differs from protocol K")
    if data.protocol == "B" and len(data.teacher_patients) != len(data.train_patients):
        raise ValueError("protocol B teacher must have labels for every TRAIN patient")
    if data.protocol == "A" and set(data.teacher_patients) != set(data.labeled_patients):
        raise ValueError("protocol A teacher must use only the same K labeled TRAIN patients")
    return data


def _data_identity_without_labels(data: Q3DData) -> dict[str, Any]:
    """Bind a run to its image cohorts and manifest bytes without reading masks."""

    student = data.student_data
    teacher = data.teacher_data

    def view_identity(view) -> dict[str, Any]:
        records = list(view.labeled) + list(view.unlabeled)
        return {
            "manifest_sha256": Q3DData._manifest_sha256(view),
            "dataset": view.manifest.get("dataset"),
            "patients": sorted(record["id"] for record in records),
            "images": sorted(
                [{"id": record["id"], "sha256": record["image_sha256"]}
                 for record in records],
                key=lambda item: item["id"],
            ),
        }

    return {
        "protocol": data.protocol,
        "train_patients": list(data.train_patients),
        "student_labeled_patients": sorted(data.labeled_patients),
        "student_unlabeled_patients": sorted(data.unlabeled_patients),
        "teacher_labeled_patients": sorted(data.teacher_patients),
        "student_view": view_identity(student),
        "teacher_view": view_identity(teacher),
    }


def _labeled_image_frame_keys(data: Q3DData) -> list[tuple[int, int]]:
    """Enumerate L cine frames from image arrays without touching GT masks."""

    view = data.student_data
    keys: list[tuple[int, int]] = []
    for record_index, record in enumerate(view.labeled):
        cine = view.base.array(record["image"], record["image_sha256"])
        geometry = view.geometry_for(record["id"])
        if cine.ndim != 4 or tuple(cine.shape[1:]) != tuple(geometry["shape"]):
            raise ValueError("Q-stage labeled cine must match native full-volume geometry")
        keys.extend((record_index, frame) for frame in range(cine.shape[0]))
    if not keys:
        raise ValueError("Q stage has no labeled-patient image frames")
    return keys


def _student_arm(arm: str) -> str:
    return {
        "c0": "C0", "mean_teacher": "C0", "c1": "C1",
        "v1": "V1", "v1u": "V1-U", "v0": "V0", "b1": "B1",
        "e1": "V1", "o1": "V1", "s1": "V1",
    }.get(arm, "")


def _teacher_feature_channels(config: Mapping[str, Any], teacher: nn.Module | None) -> int | None:
    # Prefer the live module's declared lattice width.  Synthetic ResEnc
    # fixtures intentionally shrink [32,64,128,256,320,320] to
    # [4,8,12,16,20,20], while the recipe keeps the real architecture's
    # channel contract for hashing and readiness checks.
    encoder = getattr(teacher, "encoder", None) if teacher is not None else None
    if isinstance(teacher, nn.Module):
        direct = getattr(teacher, "feature_channels", None)
        if isinstance(direct, int) and direct > 0:
            return direct
        encoder_channels = getattr(teacher, "encoder_channels", None)
        stride = config["model"]["teacher"].get("feature_stride")
        index = {4: 2, 8: 3, 16: 4}.get(stride)
        if isinstance(encoder_channels, (list, tuple)) and index is not None and index < len(encoder_channels):
            value = int(encoder_channels[index])
            if value > 0:
                return value
    if isinstance(encoder, nn.Module):
        direct = getattr(encoder, "feature_channels", None)
        if isinstance(direct, int) and direct > 0:
            return direct
    declared = config["model"]["teacher"].get("feature_channels")
    if declared is not None:
        return int(declared)
    if not isinstance(encoder, nn.Module):
        return None
    channels = getattr(encoder, "channels", None)
    if not isinstance(channels, (list, tuple)):
        return None
    stride = config["model"]["teacher"].get("feature_stride")
    index = {4: 2, 8: 3, 16: 4}.get(stride)
    if index is None or index >= len(channels):
        return None
    value = int(channels[index])
    return value if value > 0 else None


def _student_config(config: Mapping[str, Any], teacher: nn.Module | None = None) -> dict[str, Any]:
    raw = config["model"]["student"]
    if raw["architecture"] != "unetrpp_lite_3d":
        raise ValueError(f"Q-3D engine cannot construct student architecture {raw['architecture']!r}")
    arm = _student_arm(str(config["arm"]))
    if not arm:
        raise ValueError(f"Q-3D arm {config['arm']!r} does not use UNETR++-lite")
    teacher_feature_channels = _teacher_feature_channels(config, teacher)
    if arm == "C1" and teacher_feature_channels is None:
        raise ValueError("C1 requires model.teacher.feature_channels for exact feature matching")
    if arm == "C1" and bool(config["runtime"]["synthetic"]) and teacher is not None:
        if teacher_feature_channels is None:
            raise ValueError("synthetic C1 teacher cannot declare its captured feature channels")
    return {
        "student": {
            "arm": arm,
            "channels": list(raw["stage_channels"]),
            "num_classes": len(raw["classes"]),
            "in_channels": raw["input_channels"],
            "vq": {
                "size": config["tokenizer"]["codebook_size"],
                "dim": config["tokenizer"]["dimension"],
                "tau": config["tokenizer"]["temperature"],
            },
            "attention": {"support": "adaptive" if raw["epa_support"] == "adaptive" else "uniform"},
            "residual_vq": {
                "enabled": raw["residual2"]["enabled"],
                "size": config["tokenizer"]["codebook_size"],
            },
        },
        "teacher": {"feature_channels": teacher_feature_channels},
    }


def _tokenizer_config(config: Mapping[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    raw = config["tokenizer"]
    initialization_volumes = int(raw["initialization_volumes"])
    smoke_init_override = config["runtime"].get("smoke_initialization_volumes")
    if smoke and bool(config["runtime"]["synthetic"]) and smoke_init_override is not None:
        initialization_volumes = int(smoke_init_override)
    return {
        "tokenizer": {
            "codebook_size": raw["codebook_size"],
            "embedding_dim": raw["dimension"],
            "hidden_channels": raw["adapter_hidden"],
            "temperature": raw["temperature"],
            "commitment_beta": raw["commitment_beta"],
            "reconstruction_weight": 1.0,
            "anatomy_weight": raw["anatomy_loss_weight"],
            "ema_decay": raw["ema_decay"],
            "initialization_volumes": initialization_volumes,
            "initialization_points_per_volume": 1024,
            "history_window": raw["usage_window_updates"],
            "dead_code_threshold": raw["dead_code_fraction"],
            "min_perplexity": raw["minimum_perplexity"],
            "max_dead_code_fraction": raw["maximum_dead_code_fraction"],
            "min_class_codes": raw["minimum_class_codes"],
            "seed": config["seed"],
        },
    }


def _build_teacher(config: Mapping[str, Any], *, synthetic: bool) -> tuple[nn.Module, dict[str, Any]]:
    from .pretrained import build_q3d_teacher

    # The builder owns the strict ResEnc-L architecture/checkpoint contract;
    # the engine only consumes its declared dual-head teacher interface.
    return build_q3d_teacher(config, synthetic=synthetic)


def _build_tokenizer(
    config: Mapping[str, Any], teacher: nn.Module, device: torch.device, *, smoke: bool = False,
) -> nn.Module | None:
    if not config["tokenizer"]["enabled"]:
        return None
    from .tokenizer import TeacherTokenizer

    channels = _teacher_feature_channels(config, teacher)
    if channels is None:
        raise ValueError("Q-3D tokenizer requires a verified teacher feature channel declaration")
    tokenizer = TeacherTokenizer(channels, _tokenizer_config(config, smoke=smoke))
    return tokenizer.to(device)


def _build_models(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    teacher: nn.Module | None,
    student: nn.Module | None,
    tokenizer: nn.Module | None,
    smoke: bool = False,
) -> tuple[nn.Module | None, nn.Module, nn.Module | None, dict[str, Any]]:
    arm = str(config["arm"])
    provenance: dict[str, Any] = {"teacher": None}
    if teacher is None and arm in _TEACHER_ARMS:
        teacher, teacher_provenance = _build_teacher(config, synthetic=bool(config["runtime"]["synthetic"]))
        provenance["teacher"] = teacher_provenance
    elif teacher is not None:
        if not isinstance(teacher, nn.Module):
            raise TypeError("teacher must be a torch module")
        provenance["teacher"] = copy.deepcopy(getattr(teacher, "provenance", {"injected": True}))

    if student is None:
        from .student import CandidateQ3DStudent

        student = CandidateQ3DStudent(_student_config(config, teacher))
    if not isinstance(student, nn.Module):
        raise TypeError("student must be a torch module")

    if tokenizer is None and config["tokenizer"]["enabled"]:
        if teacher is None:
            raise ValueError("tokenizer stage requires a Q-3D teacher")
        tokenizer = _build_tokenizer(config, teacher, device, smoke=smoke)
    elif tokenizer is not None and not isinstance(tokenizer, nn.Module):
        raise TypeError("tokenizer must be a torch module")

    if teacher is not None:
        teacher.to(device)
        _channels_last_3d(teacher)
    student.to(device)
    _channels_last_3d(student)
    if tokenizer is not None:
        tokenizer.to(device)
        _channels_last_3d(tokenizer)
    if teacher is not None:
        from .alignment import student_grid_phase

        if arm != "teacher_supervised":
            teacher._q3d_student_phase = student_grid_phase(student)
    return teacher, student, tokenizer, provenance


def _teacher_stage(teacher: nn.Module, stage: str, *, freeze: bool) -> None:
    """Apply the dual-head teacher train/eval contract without baseline imports."""

    aliases = {"teacher_pretrain": "pretrain", "teacher_warmup": "warmup"}
    stage = aliases.get(stage, stage)
    setter = getattr(teacher, "set_teacher_stage", None)
    if not callable(setter):
        setter = getattr(teacher, "set_stage", None)
    if callable(setter):
        try:
            setter(stage, freeze=freeze)
        except TypeError:
            setter(stage)
        return
    if stage not in {"pretrain", "warmup", "joint"}:
        raise ValueError("teacher stage must be pretrain, warmup or joint")
    teacher.requires_grad_(not freeze)
    if freeze:
        teacher.eval()
        return
    teacher.train()
    if stage == "warmup":
        encoder = getattr(teacher, "encoder", None)
        if isinstance(encoder, nn.Module):
            encoder.requires_grad_(False)
            encoder.eval()


def _sample_parts(data: Q3DData, key: Any, *, teacher: bool = False) -> tuple[Tensor, Tensor, dict[str, Any], dict[str, Any]]:
    image, labels, geometry, metadata = (
        data.load_teacher_labeled_frame(key) if teacher else data.load_labeled_frame(key)
    )
    image = torch.as_tensor(image, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    geometry = dict(geometry)
    if image.ndim != 5 or tuple(image.shape[:2]) != (1, 1):
        raise ValueError("Q-3D labeled input must be native [1,1,D,H,W]")
    if labels.ndim != 4 or tuple(labels.shape) != (1, *image.shape[-3:]):
        raise ValueError("Q-3D labeled mask must be native [1,D,H,W]")
    from ..candidate_a3d.geometry import canonical_geometry

    geometry = canonical_geometry(geometry)
    if tuple(image.shape[-3:]) != tuple(geometry["shape"]):
        raise ValueError("Q-3D labeled tensor differs from declared native geometry")
    if not torch.isfinite(image).all() or bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError("Q-3D labeled image must be finite and in [0,1]")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    return image, labels, geometry, metadata


def _load_u(data: Q3DData, key: Any) -> tuple[Tensor, dict[str, Any], dict[str, Any]]:
    image, geometry, metadata = data.load_unlabeled_frame(key)
    image = torch.as_tensor(image, dtype=torch.float32)
    from ..candidate_a3d.geometry import canonical_geometry

    geometry = canonical_geometry(geometry)
    if image.ndim != 5 or tuple(image.shape[:2]) != (1, 1) \
            or tuple(image.shape[-3:]) != tuple(geometry["shape"]):
        raise ValueError("Q-3D U sample must be one native full-volume frame")
    if not torch.isfinite(image).all() or bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError("Q-3D U image must be finite and normalized to [0,1]")
    metadata = dict(metadata)
    if set(metadata) != {"patient", "frame"}:
        raise ValueError("Q-3D U metadata may contain only patient and frame identifiers")
    if metadata["patient"] != data.unlabeled_patient_for_key(key):
        raise ValueError("Q-3D U patient does not match its image-only frame key")
    return image, geometry, metadata


def _sdf_targets(labels: Tensor, geometry: Mapping[str, Any], *, radius_mm: float) -> Tensor:
    from ..candidate_a3d.bank import BankConfig3D, QueryBank3D

    bank = QueryBank3D(BankConfig3D(radius_mm=radius_mm))
    return bank.sdf_targets(labels, geometry["spacing_zyx"])


def _pooled_teacher_targets(
    region_logits: Tensor, *, target_hw: tuple[int, int], student_phase,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pool native voxels around actual student centers, excluding all pad."""
    from .alignment import pool_teacher_targets

    result = pool_teacher_targets(region_logits, region_logits.shape[-3:], student_phase)
    if tuple(result[0].shape[-2:]) != tuple(target_hw):
        raise ValueError("teacher targets do not match the student phase lattice")
    return result


def _align_stride16(feature: Tensor, native_shape: Sequence[int], *, teacher_phase, student_phase) -> Tensor:
    """S1 uses the same native-center map as every other teacher source."""
    return align_teacher_to_student4(feature, native_shape, teacher_phase, student_phase)


def _teacher_outputs(
    teacher: nn.Module,
    image: Tensor,
    geometry: Mapping[str, Any],
    *,
    with_feature: bool,
    source: str,
    synthetic: bool,
) -> dict[str, Tensor]:
    if not with_feature:
        with torch.no_grad():
            output = teacher(image, geometry=geometry)
        if not isinstance(output, Mapping) or set(output) != {"region_logits", "boundary_logits", "sdf"}:
            raise ValueError("Q-3D teacher output keys differ from the frozen dual-branch contract")
        return dict(output)

    from .teacher_features import teacher_feature_phase

    student_phase = getattr(teacher, "_q3d_student_phase", None)
    if student_phase is None:
        raise ValueError("Q3D teacher feature extraction requires the actual student grid phase")
    phases = {
        "teacher_phase": teacher_feature_phase(teacher, source, synthetic=synthetic),
        "student_phase": student_phase,
    }
    from .teacher_features import forward_teacher_with_features

    return {**forward_teacher_with_features(teacher, image, geometry=geometry, source=source), **phases}


def _features_to_student_grid(
    feature: Tensor, native_shape: Sequence[int], source: str, *, teacher_phase, student_phase,
) -> Tensor:
    if source == "encoder_stride16":
        return _align_stride16(feature, native_shape, teacher_phase=teacher_phase, student_phase=student_phase)
    return align_teacher_to_student4(feature, native_shape, teacher_phase, student_phase)


def _aligned_teacher_output(outputs: Mapping[str, Any], native_shape: Sequence[int], source: str) -> Tensor:
    return _features_to_student_grid(
        outputs["feature"], native_shape, source,
        teacher_phase=outputs["teacher_phase"], student_phase=outputs["student_phase"],
    )


def _teacher_feature_source(config: Mapping[str, Any]) -> str:
    return str(config["model"]["teacher"]["feature_source"])


def _make_optimizer(parameters, *, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not params:
        raise ValueError("Q-3D optimizer received no trainable parameters")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _schedule_lr(optimizer: torch.optim.Optimizer, *, step: int, total: int,
                 base_lrs: Sequence[float], warmup_fraction: float) -> None:
    warmup = max(1, int(math.ceil(total * warmup_fraction))) if warmup_fraction > 0 else 0
    if warmup and step < warmup:
        factor = float(step + 1) / float(warmup)
    else:
        decay_steps = max(1, total - warmup)
        progress = min(1.0, max(0.0, (step - warmup + 1) / decay_steps))
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(base_lr) * factor


def _make_teacher_optimizer(teacher: nn.Module, config: Mapping[str, Any], *, warmup: bool) -> torch.optim.Optimizer:
    training = config["training"]
    decay = float(training["weight_decay"])
    if warmup:
        return _make_optimizer(
            _teacher_head_parameters(teacher),
            lr=float(training["teacher_head_lr"]),
            weight_decay=decay,
        )
    return torch.optim.AdamW(
        _teacher_parameter_groups(teacher, config, multiplier=1.0), weight_decay=decay,
    )


def _unique_parameters(parameters: Any) -> list[nn.Parameter]:
    seen: set[int] = set()
    result: list[nn.Parameter] = []
    for parameter in parameters:
        if not isinstance(parameter, nn.Parameter) or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.requires_grad:
            result.append(parameter)
    return result


def _teacher_head_parameters(teacher: nn.Module) -> list[nn.Parameter]:
    """Return decoder/region/boundary parameters without registered aliases."""

    encoder = getattr(teacher, "encoder", None)
    encoder_ids = {
        id(parameter) for parameter in encoder.parameters()
    } if isinstance(encoder, nn.Module) else set()
    preferred = (
        "decoder", "region_path", "region_decoder", "region_head",
        "segmentation_head", "boundary_path", "boundary_head",
    )
    selected: list[nn.Parameter] = []
    for root in preferred:
        module = getattr(teacher, root, None)
        if isinstance(module, nn.Module):
            selected.extend(module.parameters())
    # If the new wrapper uses a differently named decoder, include every
    # top-level child other than the shared encoder.  Decoder.encoder aliases
    # are removed by the encoder-id exclusion below.
    if not selected:
        for root, module in teacher.named_children():
            if root != "encoder":
                selected.extend(module.parameters())
    return [parameter for parameter in _unique_parameters(selected)
            if id(parameter) not in encoder_ids]


def _teacher_parameter_groups(
    teacher: nn.Module,
    config: Mapping[str, Any],
    *,
    multiplier: float,
) -> list[dict[str, Any]]:
    """Build disjoint encoder and decoder/head groups for the ResEnc teacher.

    Some ResEnc decoder implementations register a reference to the shared
    encoder (``decoder.encoder``). Group membership is therefore by parameter
    identity, never by name, so no tensor is optimized twice.
    """
    training = config["training"]
    encoder = getattr(teacher, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("Q-3D teacher must expose an encoder module")
    trunk = _unique_parameters(encoder.parameters())
    trunk_ids = {id(parameter) for parameter in trunk}
    all_parameters = _unique_parameters(teacher.parameters())
    head = [parameter for parameter in all_parameters if id(parameter) not in trunk_ids]
    groups: list[dict[str, Any]] = []
    for parameters, base_lr, name in (
        (trunk, float(training["teacher_encoder_lr"]), "teacher_trunk"),
        (head, float(training["teacher_head_lr"]), "teacher_heads"),
    ):
        if parameters:
            lr = base_lr * float(multiplier)
            groups.append({"params": parameters, "lr": lr, "initial_lr": lr, "name": name})
    if not groups:
        raise ValueError("Q-3D teacher optimizer has no trainable parameters")
    return groups


def _optimizer_states(state: dict[str, Any], key: str, optimizer: torch.optim.Optimizer) -> None:
    if key in state.get("optimizer_states", {}):
        optimizer.load_state_dict(state["optimizer_states"][key])


def _save_optimizer_state(state: dict[str, Any], key: str, optimizer: torch.optim.Optimizer) -> None:
    state.setdefault("optimizer_states", {})[key] = copy.deepcopy(optimizer.state_dict())


def _teacher_loss_targets(labels: Tensor, geometry: Mapping[str, Any], config: Mapping[str, Any]) -> Tensor:
    return _sdf_targets(labels, geometry, radius_mm=float(config["losses"]["sdf_radius_mm"]))


def _code_interface_diagnostics(model: nn.Module, outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Measure the declared gated code contribution on an evaluated volume."""
    gate = getattr(model, "vq_gate", None)
    projector = getattr(model, "quantized_projector", None)
    f4, codes = outputs.get("f4"), outputs.get("quantized")
    if gate is None or projector is None or not isinstance(f4, Tensor) or not isinstance(codes, Tensor):
        return {"status": "NOT_APPLICABLE", "vq_gate": None, "code_to_f4_norm_ratio": None}
    with torch.no_grad():
        base_norm = float(torch.linalg.vector_norm(f4.float()))
        contribution_norm = float(torch.linalg.vector_norm((gate * projector(codes)).float()))
    ratio = contribution_norm / base_norm if base_norm > 0 else None
    return {
        "status": "MEASURED" if ratio is not None else "ZERO_F4_NORM",
        "vq_gate": float(gate.detach()), "code_to_f4_norm_ratio": ratio,
        "gated_code_norm": contribution_norm, "f4_norm": base_norm,
        "interface_claim_supported": ratio is not None and ratio >= 0.05,
        "decoder_mode": "code_only" if getattr(model, "code_only", False) else "dense_plus_code",
    }


def _development_fg_dice(
    model: nn.Module,
    manifest: Mapping[str, Any],
    *,
    sha256: str,
    train_ids: Sequence[str],
    synthetic: bool,
    output_kind: str,
    device: torch.device,
) -> dict[str, Any]:
    """Read VAL masks only for the locked patient-macro checkpoint selector."""

    from ..candidate_a3d.evaluation import _validated_manifest, _volume_input
    from ..candidate_a3d.geometry import VolumeGeometry3D

    if output_kind not in {"student", "teacher"}:
        raise ValueError("development output_kind must be student or teacher")
    path, _raw_manifest, patients, patient_ids, geometries, resolve = _validated_manifest(
        manifest["path"], sha256, train_ids=train_ids, synthetic=synthetic,
    )
    from .evaluation import aggregate_patient_class_dice, frame_class_dice

    per_patient: dict[str, float] = {}
    per_patient_class: dict[str, dict[str, float]] = {}
    interface_rows: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for record, patient_id in zip(patients, patient_ids):
                image = np.load(resolve(record["image"]), mmap_mode="r", allow_pickle=False)
                geometry = VolumeGeometry3D.from_cache_geometry(
                    geometries[patient_id], expected_shape=image.shape[1:],
                ).as_dict()
                masks = record.get("masks")
                if not isinstance(masks, Mapping) or not masks:
                    raise ValueError(f"VAL patient {patient_id!r} has no frozen annotated frames")
                frame_dice_rows: list[dict[str, Any]] = []
                for frame_key, mask_record in sorted(masks.items(), key=lambda row: int(row[0])):
                    required_mask_keys = {"path", "sha256"}
                    allowed_mask_keys = required_mask_keys | {"phase"}
                    if (not isinstance(mask_record, Mapping)
                            or not required_mask_keys.issubset(mask_record)
                            or not set(mask_record).issubset(allowed_mask_keys)):
                        raise ValueError("VAL mask records require path and sha256 with optional phase")
                    if "phase" in mask_record:
                        phase = mask_record["phase"]
                        if phase not in ("ED", "ES"):
                            raise ValueError("VAL mask phase must be ED or ES")
                        phases = record.get("phases")
                        try:
                            phase_frame = int(phases[phase]) if isinstance(phases, Mapping) else None
                            mask_frame = int(frame_key)
                        except (KeyError, TypeError, ValueError):
                            phase_frame = None
                            mask_frame = None
                        if phase_frame is None or mask_frame is None or phase_frame != mask_frame:
                            raise ValueError(f"VAL mask phase {phase} does not match its declared frame")
                    mask_path = resolve(mask_record["path"])
                    if _file_sha256(mask_path) != mask_record["sha256"]:
                        raise ValueError(f"VAL mask hash mismatch for {patient_id} frame {frame_key}")
                    target = np.load(mask_path, allow_pickle=False)
                    if target.shape != image.shape[1:] or not np.isin(target, (0, 1, 2, 3)).all():
                        raise ValueError("VAL mask must be a full native BG/RV/MYO/LV volume")
                    x = _volume_input(image, int(frame_key), device)
                    if output_kind == "teacher":
                        outputs = model(x, geometry=geometry)
                        logits = outputs["region_logits"]
                    else:
                        outputs = model(x, geometry=geometry)
                        logits = outputs["logits"]
                        interface_rows.append(_code_interface_diagnostics(model, outputs))
                    prediction = logits.argmax(1)[0].cpu().numpy()
                    frame_dice_rows.append(frame_class_dice(target, prediction))
                patient_aggregate = aggregate_patient_class_dice(frame_dice_rows)
                patient_dice = patient_aggregate["fg_dice"]
                if patient_dice is not None:
                    per_patient[patient_id] = float(patient_dice)
                    per_patient_class[patient_id] = {
                        name: float(value["dice"])
                        for name, value in patient_aggregate["dice_by_class"].items()
                        if value["dice"] is not None
                    }
    finally:
        model.train(was_training)
    if not per_patient:
        raise ValueError("VAL checkpoint selection has no patient with foreground support")
    patient_scores = per_patient
    per_class = {
        name: float(np.mean([
            patient_values[name] for patient_values in per_patient_class.values()
            if name in patient_values
        ]))
        if any(name in patient_values for patient_values in per_patient_class.values()) else None
        for name in _CLASS_NAMES[1:]
    }
    return {
        "metric": "development_patient_macro_fg_dice",
        "score": float(np.mean(list(patient_scores.values()))),
        "patients": patient_scores,
        "per_class": per_class,
        "patient_per_class": per_patient_class,
        "patient_count": len(patient_scores),
        "code_interface": {
            "frames": interface_rows,
            "vq_gate": next((row["vq_gate"] for row in interface_rows if row["vq_gate"] is not None), None),
            "mean_code_to_f4_norm_ratio": (
                float(np.mean([row["code_to_f4_norm_ratio"] for row in interface_rows
                               if row["code_to_f4_norm_ratio"] is not None]))
                if any(row["code_to_f4_norm_ratio"] is not None for row in interface_rows) else None
            ),
        },
    }


def _clone_trainable_snapshot(module: nn.Module | None) -> dict[str, Any] | None:
    if module is None:
        return None
    snapshot: dict[str, Any] = {}
    for name, value in module.state_dict().items():
        snapshot[name] = value.detach().cpu().clone() if isinstance(value, Tensor) else copy.deepcopy(value)
    return snapshot


def _same_state_value(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return isinstance(left, Tensor) and isinstance(right, Tensor) and torch.equal(
            left.detach().cpu(), right.detach().cpu(),
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping) and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_same_state_value(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, type(right)) and isinstance(right, type(left))
            and len(left) == len(right)
            and all(_same_state_value(a, b) for a, b in zip(left, right))
        )
    return left == right


def _assert_snapshot_unchanged(module: nn.Module | None, snapshot: Mapping[str, Any] | None, name: str) -> None:
    if module is None or snapshot is None:
        return
    state = module.state_dict()
    for key, before in snapshot.items():
        if key not in state or not _same_state_value(state[key], before):
            raise RuntimeError(f"frozen Q-3D {name} state changed during joint training: {key}")


def _teacher_curve_status(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize whether the last three actual T VAL scores strictly rise."""

    points: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        update = item.get("update")
        score = item.get("score")
        if type(update) is not int or update < 1:
            continue
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score_value):
            continue
        points.append({"update": update, "score": score_value})
    points.sort(key=lambda item: item["update"])
    last = points[-3:]
    if len(last) < 3:
        return {"status": "INSUFFICIENT", "value": None, "points": copy.deepcopy(last)}
    scores = [item["score"] for item in last]
    return {
        "status": "MEASURED",
        "value": bool(scores[0] < scores[1] < scores[2]),
        "points": copy.deepcopy(last),
    }


def _initial_run_state(config_sha256: str, data_identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "config_sha256": config_sha256,
        "data_identity": copy.deepcopy(dict(data_identity)),
        "stage": None,
        "stage_cursor": 0,
        "completed_stages": [],
        "completed_updates_total": 0,
        "optimizer_states": {},
        "sampler_state": None,
        "teacher_state": None,
        "student_state": None,
        "tokenizer_state": None,
        "ema_state": None,
        "best_teacher_state": None,
        "best_teacher_score": None,
        "best_teacher_dev_per_class": None,
        "best_teacher_dev_update": None,
        "teacher_dev_history": [],
        "teacher_dev_fg_dice": None,
        "teacher_dev_per_class": None,
        "teacher_selected_dev_update": None,
        "teacher_terminal_dev_fg_dice": None,
        "teacher_terminal_dev_per_class": None,
        "teacher_dev_curve_still_improving": {
            "status": "INSUFFICIENT", "value": None, "points": [],
        },
        "dev_curve_still_improving": {
            "status": "INSUFFICIENT", "value": None, "points": [],
        },
        "best_student_state": None,
        "best_student_score": None,
        "physiology_rows": None,
        # Frozen U-cohort weighting is a first-class resume identity.  The
        # rows alone are insufficient because omega is normalized over the
        # complete U cohort, not over each minibatch.
        "physiology_cohort_weighting": None,
        "physiology_weighting_logs": [],
        "gate": None,
        "metrics": [],
        "spatial_rng_state": None,
        "intensity_rng_state": None,
        "stage_rng_state": None,
        "j_u_patient_cycle": None,
        "rng": None,
        "saved_at": None,
    }


def _capture_run_state(
    state: dict[str, Any],
    *,
    teacher: nn.Module | None,
    student: nn.Module,
    tokenizer: nn.Module | None,
    ema_model: nn.Module | None,
    spatial_rng: random.Random,
    intensity_generator: torch.Generator,
    stage_rng: random.Random,
    sampler: Q3DJointSampler | None,
) -> None:
    state["teacher_state"] = copy.deepcopy(teacher.state_dict()) if teacher is not None else None
    state["student_state"] = copy.deepcopy(student.state_dict())
    state["tokenizer_state"] = copy.deepcopy(tokenizer.state_dict()) if tokenizer is not None else None
    state["tokenizer_initialization_points"] = (
        [value.detach().cpu().clone() for value in tokenizer._initialization_points]
        if tokenizer is not None and hasattr(tokenizer, "_initialization_points") else []
    )
    state["ema_state"] = copy.deepcopy(ema_model.state_dict()) if ema_model is not None else None
    state["spatial_rng_state"] = spatial_rng.getstate()
    state["intensity_rng_state"] = intensity_generator.get_state()
    state["stage_rng_state"] = stage_rng.getstate()
    state["sampler_state"] = sampler.state_dict() if sampler is not None else None
    state["rng"] = _rng_state()
    state["saved_at"] = time.time()


def _restore_run_state(
    state: Mapping[str, Any],
    *,
    teacher: nn.Module | None,
    student: nn.Module,
    tokenizer: nn.Module | None,
    ema_model: nn.Module | None,
    spatial_rng: random.Random,
    intensity_generator: torch.Generator,
    stage_rng: random.Random,
) -> None:
    if teacher is not None and state.get("teacher_state") is not None:
        teacher.load_state_dict(state["teacher_state"], strict=True)
    if state.get("student_state") is not None:
        student.load_state_dict(state["student_state"], strict=True)
    if tokenizer is not None and state.get("tokenizer_state") is not None:
        tokenizer.load_state_dict(state["tokenizer_state"], strict=True)
    if ema_model is not None and state.get("ema_state") is not None:
        ema_model.load_state_dict(state["ema_state"], strict=True)
    if state.get("spatial_rng_state") is not None:
        spatial_rng.setstate(state["spatial_rng_state"])
    if state.get("intensity_rng_state") is not None:
        intensity_generator.set_state(state["intensity_rng_state"])
    if state.get("stage_rng_state") is not None:
        stage_rng.setstate(state["stage_rng_state"])
    if state.get("rng") is not None:
        _restore_rng(state["rng"])


def _save_run_checkpoint(
    state: dict[str, Any],
    path: Path,
    *,
    teacher: nn.Module | None,
    student: nn.Module,
    tokenizer: nn.Module | None,
    ema_model: nn.Module | None,
    spatial_rng: random.Random,
    intensity_generator: torch.Generator,
    stage_rng: random.Random,
    sampler: Q3DJointSampler | None,
) -> str:
    _capture_run_state(
        state, teacher=teacher, student=student, tokenizer=tokenizer, ema_model=ema_model,
        spatial_rng=spatial_rng, intensity_generator=intensity_generator, stage_rng=stage_rng,
        sampler=sampler,
    )
    return _atomic_torch_save(state, path)


def _intensity_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config["augmentation"]
    return {"intensity_gamma": raw["gamma_range"], "intensity_noise_std": raw["voxel_noise_std"]}


def _teacher_frame_choice(data: Q3DData, rng: random.Random) -> tuple[int, int]:
    keys = data.teacher_frame_keys()
    if not keys:
        raise ValueError("Q-3D teacher stage has no labeled TRAIN frames")
    return rng.choice(keys)


def _make_q3d_sampler(
    data: Q3DData,
    seed: int,
    state: dict[str, Any],
    *,
    require_unlabeled: bool = True,
) -> Q3DJointSampler:
    sampler = Q3DJointSampler(data, seed=seed, require_unlabeled=require_unlabeled)
    if state.get("sampler_state") is not None:
        sampler.load_state_dict(state["sampler_state"])
    return sampler


def _stage_directory(root: Path, config: Mapping[str, Any], stage: str) -> Path:
    declared = Path(config["checkpoint"]["stage_outputs"][stage])
    return root / "stages" / declared.name


def _checkpoint_every(config: Mapping[str, Any], *, smoke: bool, has_update_limit: bool) -> int:
    if smoke or has_update_limit:
        return 1
    return int(config["training"]["checkpoint_evaluation_interval"])


def _record_metric(state: dict[str, Any], metrics_path: Path, record: Mapping[str, Any]) -> None:
    value = dict(record)
    state.setdefault("metrics", []).append(value)
    _jsonl_append(metrics_path, value)


def _observational_development(model, manifest, *, data, config, device, output_kind, evaluator):
    """Extra VAL observations must not perturb training RNG or mixed module modes."""
    rng = _rng_state()
    modes = [(module, module.training) for module in model.modules()]
    try:
        if evaluator is not None:
            return dict(evaluator(model, manifest, output_kind, device))
        return _development_fg_dice(
            model, manifest, sha256=manifest["sha256"], train_ids=data.train_patients,
            synthetic=bool(config["runtime"]["synthetic"]), output_kind=output_kind, device=device,
        )
    finally:
        for module, training in modes:
            module.training = training
        _restore_rng(rng)


def _record_epoch_development(state, output_dir, *, stage, update, epoch, epoch_definition,
                              epoch_updates, dev, selection_point):
    score = float(dev["score"])
    if not math.isfinite(score):
        raise FloatingPointError("epoch VAL score is nonfinite")
    _record_metric(state, output_dir / "metrics.jsonl", {
        "stage": stage, "event": "development_epoch", "update": update,
        "epoch": epoch, "epoch_definition": epoch_definition, "epoch_updates": epoch_updates,
        "score": score, "per_class": copy.deepcopy(dev.get("per_class")),
        "patient_count": int(dev.get("patient_count", len(dev.get("patients", {})))),
        "checkpoint_selection_point": selection_point,
        "code_interface": dev.get("code_interface", {"status": "NOT_RUN_CUSTOM_EVALUATOR"}),
    })


def _log_teacher_development(
    config: Mapping[str, Any], *, update: int, update_budget: int,
    train_loss: float, dev: Mapping[str, Any],
) -> None:
    """Print sparse teacher VAL diagnostics without corrupting JSON stdout."""

    def shown(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "NA"
        return f"{numeric:.6f}" if math.isfinite(numeric) else "NA"

    dataset = config.get("dataset", {}).get("name", "unknown")
    protocol = config.get("protocol", {}).get("variant", "?")
    k = config.get("protocol", {}).get("k", "?")
    seed = config.get("seed", "?")
    per_class = dev.get("per_class")
    if not isinstance(per_class, Mapping):
        per_class = {}
    print(
        f"[Q3D {dataset} protocol={protocol} k={k} seed={seed}] "
        f"T update={update}/{update_budget} | train_loss={shown(train_loss)} | "
        f"VAL patient_macro_fg_dice={shown(dev.get('score'))} | "
        f"RV={shown(per_class.get('RV'))} "
        f"MYO={shown(per_class.get('MYO'))} "
        f"LV={shown(per_class.get('LV'))}",
        file=sys.stderr,
        flush=True,
    )


def _run_teacher_stage(
    config: Mapping[str, Any],
    *,
    data: Q3DData,
    teacher: nn.Module,
    student: nn.Module,
    tokenizer: nn.Module | None,
    ema_model: nn.Module | None,
    state: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    spatial_rng: random.Random,
    intensity_generator: torch.Generator,
    stage_rng: random.Random,
    sampler: Q3DJointSampler | None,
    smoke: bool,
    update_limit_active: bool,
    dev_evaluator=None,
    update_budget: int,
) -> int:
    stage = "T"
    if update_budget <= 0:
        raise ValueError("Q-3D stage T update budget must be positive when active")
    cursor = int(state["stage_cursor"] if state.get("stage") == stage else 0)
    if state.get("stage") != stage:
        state["stage"] = stage
        state["stage_cursor"] = 0
        cursor = 0
    state.setdefault("best_teacher_dev_per_class", None)
    state.setdefault("best_teacher_dev_update", None)
    state.setdefault("teacher_dev_history", [])
    state.setdefault("teacher_dev_fg_dice", None)
    state.setdefault("teacher_dev_per_class", None)
    state.setdefault("teacher_selected_dev_update", None)
    state.setdefault("teacher_terminal_dev_fg_dice", None)
    state.setdefault("teacher_terminal_dev_per_class", None)
    state.setdefault("teacher_dev_curve_still_improving", {
        "status": "INSUFFICIENT", "value": None, "points": [],
    })
    state.setdefault("dev_curve_still_improving", copy.deepcopy(state["teacher_dev_curve_still_improving"]))
    warmup_updates = int(math.ceil(update_budget * float(config["training"]["teacher_warmup_fraction"])))
    warmup_updates = min(update_budget, max(0, warmup_updates))
    optimizer: torch.optim.Optimizer | None = None
    optimizer_key: str | None = None
    phase: str | None = None
    base_lrs: list[float] = []
    checkpoint_every = _checkpoint_every(config, smoke=smoke, has_update_limit=update_limit_active)
    interval = int(config["training"]["checkpoint_evaluation_interval"])
    dev_manifest = config["dataset"]["manifests"]["development"]
    if not isinstance(dev_manifest, Mapping) or not dev_manifest.get("path") or not dev_manifest.get("sha256"):
        raise ValueError("active Q-3D teacher stage requires a pinned VAL manifest for checkpoint selection")
    epoch_updates = 2 * int(config["protocol"]["k"])
    while cursor < update_budget:
        current_phase = "T-a" if cursor < warmup_updates else "T-b"
        if current_phase != phase:
            phase = current_phase
            _teacher_stage(teacher, "warmup" if phase == "T-a" else "pretrain", freeze=False)
            optimizer = _make_teacher_optimizer(teacher, config, warmup=(phase == "T-a"))
            optimizer_key = "teacher_Ta" if phase == "T-a" else "teacher_Tb"
            base_lrs = [float(group.get("initial_lr", group["lr"])) for group in optimizer.param_groups]
            _optimizer_states(state, optimizer_key, optimizer)
        assert optimizer is not None and optimizer_key is not None
        if phase == "T-b":
            _schedule_lr(
                optimizer,
                step=cursor - warmup_updates,
                total=max(1, update_budget - warmup_updates),
                base_lrs=base_lrs,
                warmup_fraction=float(config["training"]["warmup_fraction"]),
            )
        key = _teacher_frame_choice(data, stage_rng)
        image, labels, geometry, _metadata = _sample_parts(data, key, teacher=True)
        image, labels, geometry = augment_labeled(
            image, labels, geometry, rng=spatial_rng,
            intensity_config=_intensity_config(config), intensity_generator=intensity_generator,
            flip_probability=float(config["augmentation"]["flip_y_probability"]),
            rotation_probability=float(config["augmentation"]["rotate90_probability"]),
        )
        image = image.to(device=device, memory_format=torch.channels_last_3d)
        labels = labels.to(device=device)
        sdf_targets = _teacher_loss_targets(labels, geometry, config).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            outputs = teacher(image, geometry=geometry)
            loss, parts = q3d_teacher_loss(
                outputs, labels, sdf_targets,
                lambda_distance=float(config["losses"]["balanced_sdf"]),
                radius_mm=float(config["losses"]["sdf_radius_mm"]),
                near_mm=float(config["losses"]["sdf_near_boundary_mm"]),
                alpha=float(config["losses"]["sdf_near_far_alpha"]),
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Q-3D teacher stage produced nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]],
            float(config["training"]["gradient_clip_norm"]),
        )
        optimizer.step()
        cursor += 1
        state["stage_cursor"] = cursor
        state["completed_updates_total"] += 1
        record = {
            "stage": stage,
            "phase": phase,
            "update": cursor,
            "logical_updates": cursor,
            "optimizer_updates": cursor,
            "loss": float(loss.detach().float().item()),
            "region_loss": float(parts["region"].float().item()),
            "boundary_loss": float(parts["boundary"].float().item()),
            "sdf_near_far_loss": float(parts["sdf_near_far"].float().item()),
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        }
        _record_metric(state, output_dir / "metrics.jsonl", record)

        epoch_complete = cursor % epoch_updates == 0
        evaluate_now = cursor % interval == 0 or cursor == update_budget
        if evaluate_now:
            if dev_evaluator is None:
                dev = _development_fg_dice(
                    teacher, dev_manifest, sha256=dev_manifest["sha256"],
                    train_ids=data.train_patients, synthetic=bool(config["runtime"]["synthetic"]),
                    output_kind="teacher", device=device,
                )
            else:
                dev = dict(dev_evaluator(teacher, dev_manifest, "teacher", device))
            score = float(dev["score"])
            if not math.isfinite(score):
                raise FloatingPointError("VAL checkpoint-selection score is nonfinite")
            per_class = dev.get("per_class")
            if isinstance(per_class, Mapping):
                normalized_per_class: dict[str, float | None] = {}
                for name, value in per_class.items():
                    if name not in _CLASS_NAMES[1:]:
                        continue
                    try:
                        value_float = float(value) if value is not None else None
                    except (TypeError, ValueError):
                        value_float = None
                    normalized_per_class[name] = (
                        value_float if value_float is not None and math.isfinite(value_float) else None
                    )
                per_class = normalized_per_class
            else:
                per_class = None
            history = []
            for item in state.get("teacher_dev_history", []):
                if not isinstance(item, Mapping) or item.get("update") == cursor:
                    continue
                if type(item.get("update")) is not int or item["update"] < 1:
                    continue
                try:
                    old_score = float(item.get("score"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(old_score):
                    history.append({"update": item["update"], "score": old_score})
            history.append({"update": cursor, "score": score})
            history.sort(key=lambda item: int(item["update"]))
            state["teacher_dev_history"] = history
            state["teacher_terminal_dev_fg_dice"] = score
            state["teacher_terminal_dev_per_class"] = copy.deepcopy(per_class)
            state["teacher_dev_curve_still_improving"] = _teacher_curve_status(history)
            state["dev_curve_still_improving"] = copy.deepcopy(state["teacher_dev_curve_still_improving"])
            if state.get("best_teacher_score") is None or score > float(state["best_teacher_score"]):
                state["best_teacher_score"] = score
                state["best_teacher_state"] = copy.deepcopy(teacher.state_dict())
                state["best_teacher_dev_per_class"] = copy.deepcopy(per_class)
                state["best_teacher_dev_update"] = cursor
            _record_metric(state, output_dir / "metrics.jsonl", {
                "stage": stage, "event": "development_selection", "update": cursor,
                "epoch_logged": epoch_complete,
                "score": score, "best_score": float(state["best_teacher_score"]),
                "teacher_dev_fg_dice": score,
                "teacher_dev_per_class": copy.deepcopy(per_class),
                "train_loss": float(record["loss"]),
                "dev_curve_still_improving": copy.deepcopy(state["teacher_dev_curve_still_improving"]),
                "patient_count": int(dev.get("patient_count", len(dev.get("patients", {})))),
                "code_interface": dev.get("code_interface", {"status": "NOT_RUN_CUSTOM_EVALUATOR"}),
            })
            _log_teacher_development(
                config, update=cursor, update_budget=update_budget,
                train_loss=float(record["loss"]), dev=dev,
            )
        if epoch_complete:
            if not evaluate_now:
                dev = _observational_development(
                    teacher, dev_manifest, data=data, config=config, device=device,
                    output_kind="teacher", evaluator=dev_evaluator,
                )
            _record_epoch_development(
                state, output_dir, stage=stage, update=cursor, epoch=cursor // epoch_updates,
                epoch_definition="2K_random_labeled_frame_draws", epoch_updates=epoch_updates,
                dev=dev, selection_point=evaluate_now,
            )
        if cursor % checkpoint_every == 0 or cursor == update_budget:
            _save_optimizer_state(state, optimizer_key, optimizer)
            _capture_run_state(
                state, teacher=teacher, student=student, tokenizer=tokenizer,
                ema_model=ema_model, spatial_rng=spatial_rng, intensity_generator=intensity_generator,
                stage_rng=stage_rng, sampler=sampler,
            )
            _atomic_torch_save(state, _stage_directory(output_dir, config, stage) / "latest.pt")
            _atomic_torch_save(state, output_dir / "latest.pt")
        if (update_limit_active and cursor < update_budget
                and state["completed_updates_total"] >= int(state["requested_update_limit"])):
            return cursor

    if state.get("best_teacher_state") is None:
        raise RuntimeError("Q-3D teacher stage completed without a VAL-selected checkpoint")
    state["teacher_dev_fg_dice"] = (
        float(state["best_teacher_score"])
        if state.get("best_teacher_score") is not None else None
    )
    state["teacher_dev_per_class"] = copy.deepcopy(state.get("best_teacher_dev_per_class"))
    state["teacher_selected_dev_update"] = state.get("best_teacher_dev_update")
    state["teacher_dev_curve_still_improving"] = _teacher_curve_status(
        state.get("teacher_dev_history", [])
    )
    state["dev_curve_still_improving"] = copy.deepcopy(state["teacher_dev_curve_still_improving"])
    teacher.load_state_dict(state["best_teacher_state"], strict=True)
    state["completed_stages"].append(stage)
    state["stage"] = None
    state["stage_cursor"] = 0
    state["teacher_best_sha256"] = _state_sha256(teacher)
    _capture_run_state(
        state, teacher=teacher, student=student, tokenizer=tokenizer,
        ema_model=ema_model, spatial_rng=spatial_rng, intensity_generator=intensity_generator,
        stage_rng=stage_rng, sampler=sampler,
    )
    _atomic_torch_save(state, _stage_directory(output_dir, config, stage) / "latest.pt")
    _atomic_torch_save(state, output_dir / "latest.pt")
    return cursor


def _load_labeled_image_only(data: Q3DData, key: Any) -> tuple[Tensor, dict[str, Any], dict[str, Any]]:
    """Load a labeled patient's image for Q without opening its mask file."""

    loader = getattr(data, "load_labeled_image_frame", None)
    if callable(loader):
        image, geometry, metadata = loader(key)
    else:
        # Older Q3DData snapshots do not yet expose the dedicated method. Read
        # only the image record through the native cache loader; never delegate
        # to load_labeled_frame, which would materialize the mask.
        student_data = getattr(data, "student_data", None)
        if student_data is None or not hasattr(student_data, "base"):
            raise RuntimeError("Q3DData cannot provide an image-only labeled frame for Q")
        record_index, frame = data._checked_key(key, len(student_data.labeled), "labeled")
        record = student_data.labeled[record_index]
        image_cine = student_data.base.array(record["image"], record["image_sha256"])
        geometry = student_data.geometry_for(record["id"])
        if image_cine.ndim != 4 or not 0 <= frame < image_cine.shape[0]:
            raise ValueError("Q-stage labeled image frame is outside its native cine")
        image = torch.from_numpy(np.array(image_cine[frame], dtype=np.float32, copy=True))[None, None]
        metadata = {"patient": record["id"], "frame": frame}
    image = torch.as_tensor(image, dtype=torch.float32)
    from ..candidate_a3d.geometry import canonical_geometry

    geometry = canonical_geometry(geometry)
    if image.ndim != 5 or tuple(image.shape[:2]) != (1, 1) \
            or tuple(image.shape[-3:]) != tuple(geometry["shape"]):
        raise ValueError("Q-stage labeled image must cover one native full-volume frame")
    if not torch.isfinite(image).all() or bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError("Q-stage labeled image must be finite and normalized to [0,1]")
    if not isinstance(metadata, Mapping) or set(metadata) != {"patient", "frame"}:
        raise ValueError("Q-stage labeled metadata must contain only patient and frame")
    return image, geometry, dict(metadata)


def _tokenizer_forward(
    teacher: nn.Module,
    tokenizer: nn.Module,
    image: Tensor,
    geometry: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor]:
    source = _teacher_feature_source(config)
    teacher_outputs = _teacher_outputs(
        teacher, image, geometry, with_feature=True, source=source,
        synthetic=bool(config["runtime"]["synthetic"]),
    )
    aligned = _aligned_teacher_output(teacher_outputs, image.shape[-3:], source)
    probs, _classes, _confidence = _pooled_teacher_targets(
        teacher_outputs["region_logits"], target_hw=tuple(aligned.shape[-2:]),
        student_phase=teacher_outputs["student_phase"],
    )
    result = tokenizer(aligned, class_probs=probs)
    return teacher_outputs, result


def _tokenizer_usage_report(
    tokenizer: nn.Module,
    *,
    fixed_codes: Tensor | None = None,
    fixed_probs: Tensor | None = None,
) -> dict[str, Any]:
    if fixed_codes is not None and fixed_probs is not None:
        return tokenizer.metrics(
            codes=fixed_codes.detach(), class_probs=fixed_probs.detach(),
            snapshot_id="q3d_fixed_train_frame", cohort_id="q3d_fixed_train_frame",
        )
    return tokenizer.metrics()


def _freeze_tokenizer_for_synthetic_override(
    tokenizer: nn.Module, actual_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze a fixture tokenizer without rewriting a failed gate as PASS."""

    if not bool(getattr(tokenizer, "codebook_initialized", False)):
        raise RuntimeError("synthetic tokenizer override cannot freeze an uninitialized codebook")
    if actual_gate.get("status") == "PASS" or actual_gate.get("passed") is True:
        raise ValueError("synthetic override is only valid when the actual tokenizer gate failed")
    if not hasattr(tokenizer, "is_frozen"):
        raise TypeError("synthetic tokenizer override requires the canonical TeacherTokenizer API")
    tokenizer.is_frozen.fill_(True)
    if hasattr(tokenizer, "reconstructor"):
        tokenizer.reconstructor = None
    if hasattr(tokenizer, "anatomy_head"):
        tokenizer.anatomy_head = None
    tokenizer.requires_grad_(False)
    tokenizer.eval()
    metadata = tokenizer.artifact_metadata()
    return {
        **metadata,
        "gate": copy.deepcopy(dict(actual_gate)),
        "synthetic_gate_override": True,
        "evidence_scope": "SOFTWARE_ONLY_SYNTHETIC",
    }


def _run_tokenizer_stage(
    config: Mapping[str, Any],
    *,
    data: Q3DData,
    teacher: nn.Module,
    student: nn.Module,
    tokenizer: nn.Module,
    ema_model: nn.Module | None,
    state: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    spatial_rng: random.Random,
    intensity_generator: torch.Generator,
    stage_rng: random.Random,
    sampler: Q3DJointSampler | None,
    smoke: bool,
    update_limit_active: bool,
    synthetic_gate_override: bool,
    update_budget: int,
) -> int:
    stage = "Q"
    if update_budget <= 0:
        raise ValueError("Q-3D stage Q update budget must be positive when active")
    if state.get("stage") != stage:
        state["stage"] = stage
        state["stage_cursor"] = 0
    cursor = int(state["stage_cursor"])
    _teacher_stage(teacher, "joint", freeze=True)
    tokenizer.train(True)
    train_cfg = config["training"]
    labeled_keys = _labeled_image_frame_keys(data)
    unlabeled_keys = data.unlabeled_frame_keys()
    if not labeled_keys or not unlabeled_keys:
        raise ValueError("Q stage requires both labeled L and image-only TRAIN U frames")
    source = _teacher_feature_source(config)
    checkpoint_every = _checkpoint_every(config, smoke=smoke, has_update_limit=update_limit_active)

    # The 200-volume k-means++ initialization is a no-GT prepass, separate
    # from the declared Q optimizer-update budget. It uses L images through
    # the image-only loader and U images through the image-only data view.
    init_sources = [("L", key) for key in labeled_keys] + [("U", key) for key in unlabeled_keys]
    init_cursor = int(state.get("q_initialization_volumes", 0))
    while not tokenizer.codebook_initialized:
        if init_cursor >= int(tokenizer.initialization_volumes):
            raise RuntimeError("Q-3D tokenizer has an invalid initialization volume count")
        source_kind, key = stage_rng.choice(init_sources)
        if source_kind == "L":
            image, geometry, _metadata = _load_labeled_image_only(data, key)
        else:
            image, geometry, _metadata = _load_u(data, key)
        image = image.to(device=device, memory_format=torch.channels_last_3d)
        with torch.no_grad(), _autocast(device):
            _teacher_outputs_for_init, initialized = _tokenizer_forward(
                teacher, tokenizer, image, geometry, config=config, device=device,
            )
        if not initialized.get("initialized") and int(tokenizer.initialization_volume_count.item()) <= init_cursor:
            raise RuntimeError("Q-3D tokenizer initialization did not consume an image-only volume")
        init_cursor = int(tokenizer.initialization_volume_count.item())
        state["q_initialization_volumes"] = init_cursor
        if init_cursor % 50 == 0 or tokenizer.codebook_initialized:
            _record_metric(state, output_dir / "metrics.jsonl", {
                "stage": stage, "event": "kmeans_initialization", "volumes": init_cursor,
                "initialized": tokenizer.codebook_initialized,
            })
        if init_cursor % checkpoint_every == 0 or tokenizer.codebook_initialized:
            _capture_run_state(
                state, teacher=teacher, student=student, tokenizer=tokenizer, ema_model=ema_model,
                spatial_rng=spatial_rng, intensity_generator=intensity_generator, stage_rng=stage_rng,
                sampler=sampler,
            )
            _atomic_torch_save(state, _stage_directory(output_dir, config, stage) / "latest.pt")
            _atomic_torch_save(state, output_dir / "latest.pt")

    optimizer = _make_optimizer(
        [*tokenizer.adapter.parameters(), *tokenizer.reconstructor.parameters(), *tokenizer.anatomy_head.parameters()],
        lr=float(train_cfg["student_lr"]), weight_decay=float(train_cfg["weight_decay"]),
    )
    _optimizer_states(state, "tokenizer_Q", optimizer)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    fixed_key = unlabeled_keys[0]
    while cursor < update_budget:
        use_unlabeled = cursor % 2 == 0
        if use_unlabeled:
            key = stage_rng.choice(unlabeled_keys)
            image, geometry, _metadata = _load_u(data, key)
        else:
            key = stage_rng.choice(labeled_keys)
            image, geometry, _metadata = _load_labeled_image_only(data, key)
        image = image.to(device=device, memory_format=torch.channels_last_3d)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), _autocast(device):
            teacher_outputs, tokenizer_output = _tokenizer_forward(
                teacher, tokenizer, image, geometry, config=config, device=device,
            )
        if tokenizer_output.get("initialized"):
            loss = tokenizer_output.get("loss")
            if not isinstance(loss, Tensor) or loss.ndim != 0:
                raise RuntimeError("initialized tokenizer did not return a scalar Q-stage loss")
            # Teacher targets are detached, while the tokenizer's A_T,
            # reconstructor and anatomy head retain gradients.
            with _autocast(device):
                tokenizer_output = tokenizer(
                    _aligned_teacher_output(teacher_outputs, image.shape[-3:], source),
                    class_probs=_pooled_teacher_targets(
                        teacher_outputs["region_logits"],
                        target_hw=tuple(_aligned_teacher_output(teacher_outputs, image.shape[-3:], source).shape[-2:]),
                        student_phase=teacher_outputs["student_phase"],
                    )[0],
                )
                loss = tokenizer_output["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Q-stage tokenizer loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                float(train_cfg["gradient_clip_norm"]),
            )
            optimizer.step()
            optimizer_update = True
            tokenizer.update_codebook(tokenizer_output["u"], tokenizer_output["codes"])
            loss_value = float(loss.detach().float().item())
            codes = tokenizer_output["codes"].detach().cpu()
            class_probs = _pooled_teacher_targets(
                teacher_outputs["region_logits"], target_hw=tuple(codes.shape[-2:]),
                student_phase=teacher_outputs["student_phase"],
            )[0].detach().cpu()
            usage = tokenizer.metrics(
                codes=codes, class_probs=class_probs,
                cohort_id=f"q_train_update:{cursor + 1}",
            )
        else:
            optimizer_update = False
            loss_value = None
            usage = tokenizer.metrics()
            # Initialization consumes image-only teacher features but performs
            # no optimizer or EMA-codebook update until k-means++ is complete.

        cursor += 1
        state["stage_cursor"] = cursor
        state["completed_updates_total"] += 1
        state["q_optimizer_updates"] = int(state.get("q_optimizer_updates", 0)) + int(optimizer_update)
        record = {
            "stage": stage,
            "update": cursor,
            "logical_updates": cursor,
            "optimizer_updates": int(state["q_optimizer_updates"]),
            "loss": loss_value,
            "initialized": bool(tokenizer.codebook_initialized),
            "initialization_volumes": int(tokenizer.initialization_volume_count.item()),
            "ema_updates": int(tokenizer.ema_updates.item()),
        }
        _record_metric(state, output_dir / "metrics.jsonl", record)

        if cursor % 50 == 0 or cursor == update_budget:
            fixed_image, fixed_geometry, _fixed_metadata = _load_u(data, fixed_key)
            fixed_image = fixed_image.to(device=device, memory_format=torch.channels_last_3d)
            with torch.no_grad():
                fixed_teacher, fixed_tokenizer = _tokenizer_forward(
                    teacher, tokenizer, fixed_image, fixed_geometry, config=config, device=device,
                )
                fixed_codes = fixed_tokenizer.get("codes")
                if fixed_codes is not None:
                    fixed_probs = _pooled_teacher_targets(
                        fixed_teacher["region_logits"], target_hw=tuple(fixed_codes.shape[-2:]),
                        student_phase=fixed_teacher["student_phase"],
                    )[0]
                    usage = _tokenizer_usage_report(
                        tokenizer, fixed_codes=fixed_codes, fixed_probs=fixed_probs,
                    )
                else:
                    usage = {"initialized": False, "churn": None}
            _record_metric(state, output_dir / "metrics.jsonl", {
                "stage": stage, "event": "codebook_diagnostics", "update": cursor,
                **usage,
            })
        if cursor % checkpoint_every == 0 or cursor == update_budget:
            _save_optimizer_state(state, "tokenizer_Q", optimizer)
            _capture_run_state(
                state, teacher=teacher, student=student, tokenizer=tokenizer, ema_model=ema_model,
                spatial_rng=spatial_rng, intensity_generator=intensity_generator, stage_rng=stage_rng,
                sampler=sampler,
            )
            _atomic_torch_save(state, _stage_directory(output_dir, config, stage) / "latest.pt")
            _atomic_torch_save(state, output_dir / "latest.pt")
        if (update_limit_active and cursor < update_budget
                and state["completed_updates_total"] >= int(state["requested_update_limit"])):
            return cursor

    if not tokenizer.codebook_initialized:
        state["gate"] = {"status": "NOT_READY", "passed": False, "missing": ["initialized_codebook"]}
        raise RuntimeError("Q-3D tokenizer did not reach k-means++ initialization within its declared Q budget")
    metrics = _tokenizer_usage_report(tokenizer)
    gate = tokenizer.check_gate(metrics)
    state["gate"] = gate
    _record_metric(state, output_dir / "metrics.jsonl", {
        "stage": stage, "event": "pre_joint_gate", **gate,
    })
    state["synthetic_gate_override_used"] = False
    if not gate.get("passed") and synthetic_gate_override:
        if not bool(config["runtime"]["synthetic"]):
            raise ValueError("Q-3D synthetic gate override cannot be used for a real run")
        state["synthetic_gate_override_used"] = True
        state["gate_override"] = {
            "status": "SYNTHETIC_TEST_OVERRIDE",
            "continuation_allowed": True,
            "evidence_scope": "SOFTWARE_ONLY_SYNTHETIC",
            "original_gate": gate,
        }
        _record_metric(state, output_dir / "metrics.jsonl", {
            "stage": stage, "event": "synthetic_gate_override", "original_gate": gate,
        })
    if not gate.get("passed") and not state["synthetic_gate_override_used"]:
        _capture_run_state(
            state, teacher=teacher, student=student, tokenizer=tokenizer, ema_model=ema_model,
            spatial_rng=spatial_rng, intensity_generator=intensity_generator, stage_rng=stage_rng,
            sampler=sampler,
        )
        _atomic_torch_save(state, _stage_directory(output_dir, config, stage) / "latest.pt")
        _atomic_torch_save(state, output_dir / "latest.pt")
        raise RuntimeError(f"Q-3D tokenizer gate failed; joint stage stopped: {gate}")
    if state.get("synthetic_gate_override_used"):
        freeze_result = _freeze_tokenizer_for_synthetic_override(tokenizer, gate)
    else:
        freeze_result = tokenizer.freeze()
    if not tokenizer.codebook_frozen:
        raise RuntimeError("Q-3D tokenizer failed to freeze after passing its gate")
    if getattr(student, "use_vq", False):
        student.install_codebook({"codes": tokenizer.C0.detach(), **tokenizer.artifact_metadata()})
    state["tokenizer_artifact"] = freeze_result
    state["completed_stages"].append(stage)
    state["stage"] = None
    state["stage_cursor"] = 0
    _capture_run_state(
        state, teacher=teacher, student=student, tokenizer=tokenizer, ema_model=ema_model,
        spatial_rng=spatial_rng, intensity_generator=intensity_generator, stage_rng=stage_rng,
        sampler=sampler,
    )
    _atomic_torch_save(state, _stage_directory(output_dir, config, stage) / "latest.pt")
    _atomic_torch_save(state, output_dir / "latest.pt")
    return cursor


def _run_physiology_stage(
    config: Mapping[str, Any],
    *,
    data: Q3DData,
    teacher: nn.Module,
    tokenizer: nn.Module | None,
    student: nn.Module,
    ema_model: nn.Module | None,
    state: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    spatial_rng: random.Random,
    intensity_generator: torch.Generator,
    stage_rng: random.Random,
    sampler: Q3DJointSampler | None,
) -> dict[str, Any]:
    from .physiology import physiology_weight_for_patient, save_physiology_cache

    _teacher_stage(teacher, "joint", freeze=True)
    teacher_sha = _state_sha256(teacher)
    if tokenizer is None:
        codebook_sha = None
    else:
        codebook_sha = str(tokenizer.codebook_sha256)
    physiology = config["physiology"]
    physiology_config = {
        "tau_sv": physiology["tau_stroke_volume"],
        "tau_m": physiology["tau_myocardium"],
        "stroke_balance": physiology["stroke_balance"],
        "has_rv": True,
        "minimum_lv_ed_ml": physiology["minimum_lv_ed_ml"],
        "ef_range_percent": physiology["ef_range_percent"],
    }
    rows = []
    for patient_id in data.unlabeled_patients:
        row = physiology_weight_for_patient(
            teacher, data, patient_id, teacher_sha256=teacher_sha,
            codebook_sha256=codebook_sha, protocol=config["protocol"]["variant"],
            config=physiology_config, device=device,
        )
        rows.append(row)
    state["physiology_rows"] = {row["id"]: row for row in rows}
    cache_path = output_dir / "physiology_cache.json"
    source_identity = data.cache_source_identity()
    checksum = save_physiology_cache(
        cache_path, rows, teacher_sha256=teacher_sha, codebook_sha256=codebook_sha,
        protocol=config["protocol"]["variant"], refreshed_epoch=0,
        config=physiology_config, source_identity=source_identity,
    )
    from .physiology import summarize_physiology_cohort

    cohort_summary = summarize_physiology_cohort(rows)
    state["physiology_cache"] = {"path": str(cache_path), "sha256": checksum,
                                  "teacher_sha256": teacher_sha, "codebook_sha256": codebook_sha,
                                  "source_identity": source_identity, "config": physiology_config,
                                  "cohort_weighting": cohort_summary}
    physiology_gate = _require_positive_u_physiology_support(
        config, data, state["physiology_rows"], state,
        context="P physiology cache", synthetic_override=bool(
            state.get("synthetic_gate_override_requested", False)
        ),
    )
    if physiology_gate is not None:
        _record_metric(state, output_dir / "metrics.jsonl", {
            "stage": "P", "event": "physiology_support_gate", **physiology_gate,
        })
        if not physiology_gate["passed"] and not physiology_gate["override_used"]:
            raise RuntimeError(
                "P physiology cache has zero support for every TRAIN U patient; "
                "physiology-weighted KD cannot enter J"
            )
    cohort_metadata = state.get("physiology_cohort_weighting")
    _record_metric(state, output_dir / "metrics.jsonl", {
        "stage": "P", "event": "physiology_cache", "patient_count": len(rows),
        "teacher_sha256": teacher_sha, "codebook_sha256": codebook_sha,
        "cache_path": str(cache_path), "cache_sha256": checksum,
        "weights": {row["id"]: row["w_pat"] for row in rows},
        "cohort_weighting": cohort_metadata,
    })
    state["completed_stages"].append("P")
    state["stage"] = None
    state["stage_cursor"] = 0
    _capture_run_state(
        state, teacher=teacher, student=student, tokenizer=tokenizer, ema_model=ema_model,
        spatial_rng=spatial_rng, intensity_generator=intensity_generator, stage_rng=stage_rng,
        sampler=sampler,
    )
    _atomic_torch_save(state, _stage_directory(output_dir, config, "P") / "latest.pt")
    _atomic_torch_save(state, output_dir / "latest.pt")
    return state["physiology_rows"]


def _cohort_policy_arm(config: Mapping[str, Any]) -> str | None:
    """Return the U-cohort weighting policy for the active arm.

    V1-U emits unit patient multipliers but shares V1/C1 eligibility.  The
    explicit physiology flag and loss mode keep direct synthetic stage probes
    that disable weighting from requiring a cache accidentally.
    """

    arm = str(config.get("arm", ""))
    losses = config.get("losses", {})
    physiology = config.get("physiology", {})
    mode = losses.get("w_mode") if isinstance(losses, Mapping) else None
    enabled = bool(physiology.get("enabled", False)) if isinstance(physiology, Mapping) else False
    if arm == "v1u":
        return "V1-U" if enabled and mode == "uniform" else None
    if not (enabled and mode == "physiology"):
        return None
    if arm == "c1":
        return "C1"
    if arm in {"v1", "b1", "e1", "o1", "s1"}:
        return "V1"
    return None


def _physiology_rows_for_u(data: Q3DData, state: Mapping[str, Any]) -> dict[str, float]:
    """Extract exactly one finite cached w_pat for every image-only U patient."""

    rows = state.get("physiology_rows")
    if not isinstance(rows, Mapping):
        raise RuntimeError("Q-3D physiology cache has no frozen U patient rows")
    expected = tuple(data.unlabeled_patients)
    if not expected:
        raise RuntimeError("Q-3D physiology weighting requires a nonempty TRAIN U cohort")
    if set(rows) != set(expected):
        raise RuntimeError("Q-3D physiology cache U cohort differs from the current TRAIN U cohort")
    values: dict[str, float] = {}
    for patient_id in sorted(expected):
        row = rows.get(patient_id)
        if not isinstance(row, Mapping) or "w_pat" not in row:
            raise RuntimeError(f"Q-3D physiology cache is missing w_pat for U patient {patient_id!r}")
        value = float(row["w_pat"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"Q-3D physiology cache has invalid w_pat for U patient {patient_id!r}")
        values[patient_id] = value
    return values


def _cohort_weighting_metadata(
    config: Mapping[str, Any],
    data: Q3DData,
    state: dict[str, Any],
    *,
    context: str,
    synthetic_override: bool,
    require_existing: bool = False,
) -> dict[str, Any] | None:
    """Build/verify the frozen full-U omega map and eligibility receipt."""

    policy_arm = _cohort_policy_arm(config)
    if policy_arm is None:
        return None
    w_pat = _physiology_rows_for_u(data, state)
    raw = cohort_omega(w_pat, "L")
    gate_error: str | None = None
    try:
        computed = cohort_omega(w_pat, policy_arm)
        gate_passed = True
    except ValueError as exc:
        computed = raw
        gate_passed = False
        gate_error = str(exc)

    all_zero = raw["mean_w_pat"] <= 0.0 or raw["ess"] <= 0.0
    runtime = config.get("runtime", {})
    override_allowed = bool(
        synthetic_override
        and isinstance(runtime, Mapping)
        and runtime.get("synthetic") is True
        and state.get("synthetic_gate_override_requested") is True
    )
    override_used = bool(override_allowed and all_zero and not gate_passed)
    if gate_passed:
        emitted = {str(patient): float(value) for patient, value in computed["omega"].items()}
        mean_omega = float(computed["mean_omega"])
        max_omega = float(computed["max_omega"])
        zero_fraction = float(computed["zero_fraction"])
        override_policy = None
    elif policy_arm == "V1-U" and override_used:
        emitted = {patient: 1.0 for patient in w_pat}
        mean_omega, max_omega, zero_fraction = 1.0, 1.0, 0.0
        override_policy = "synthetic_v1u_unit_omega"
    else:
        # Never replace an invalid physiology signal with a guessed prior.
        # Weighted smoke arms keep their KD graph alive with omega=0.
        emitted = {patient: 0.0 for patient in w_pat}
        mean_omega, max_omega, zero_fraction = 0.0, 0.0, 1.0
        override_policy = "synthetic_weighted_zero_omega" if override_used else "gate_failed_zero_omega"

    positive_count = sum(value > 0.0 for value in w_pat.values())
    status = "PASS_POSITIVE_U_SUPPORT" if gate_passed else (
        "NO_GO_ALL_U_WEIGHTS_ZERO" if all_zero else "NO_GO_U_ESS_GATE"
    )
    gate = {
        "context": context,
        "status": status,
        "passed": gate_passed,
        "patient_weights": dict(w_pat),
        "positive_patient_count": positive_count,
        "patient_count": len(w_pat),
        "mean_w_pat": float(raw["mean_w_pat"]),
        "ess": float(raw["ess"]),
        "ess_fraction": float(raw["ess_fraction"]),
        "override_used": override_used,
        "override_scope": "synthetic_smoke_only" if override_used else None,
        "gate_error": gate_error,
    }
    metadata = {
        "schema": "self_audit.candidate_q3d.runtime_cohort_weighting.v1",
        "source": "U",
        "source_cohort": "U",
        "arm": policy_arm,
        "patient_ids": sorted(w_pat),
        "w_pat": dict(w_pat),
        "omega": emitted,
        "mean_w_pat": float(raw["mean_w_pat"]),
        "ess": float(raw["ess"]),
        "ess_fraction": float(raw["ess_fraction"]),
        "mean_omega": mean_omega,
        "max_omega": max_omega,
        "zero_fraction": zero_fraction,
        "gate": gate,
        "gate_passed": gate_passed,
        "override_used": override_used,
        "override_policy": override_policy,
    }
    existing = state.get("physiology_cohort_weighting")
    if require_existing:
        if not isinstance(existing, Mapping):
            raise RuntimeError(
                "Q-3D resume has an old physiology cache without frozen cohort_weighting metadata"
            )
        for field in (
            "schema", "source", "source_cohort", "arm", "patient_ids", "w_pat", "omega",
            "mean_w_pat", "ess", "ess_fraction", "mean_omega", "max_omega", "zero_fraction",
            "gate_passed", "override_used", "override_policy",
        ):
            if existing.get(field) != metadata.get(field):
                raise RuntimeError(
                    f"Q-3D resume physiology cohort_weighting is stale or inconsistent ({field})"
                )
        cache_metadata = state.get("physiology_cache")
        if not isinstance(cache_metadata, Mapping):
            raise RuntimeError("Q-3D resume has no physiology cache identity")
        cached_summary = cache_metadata.get("cohort_weighting")
        if not isinstance(cached_summary, Mapping):
            raise RuntimeError("Q-3D resume cache is missing cohort_weighting metadata")
        expected_summary = {
            "schema": "self_audit.candidate_q3d.cohort_weighting.v1",
            "patient_ids": sorted(w_pat),
            "count": len(w_pat),
            "mean_w_pat": float(raw["mean_w_pat"]),
            "ess": float(raw["ess"]),
            "ess_fraction": float(raw["ess_fraction"]),
            "gate_passed": bool(gate_passed),
        }
        if dict(cached_summary) != expected_summary:
            raise RuntimeError("Q-3D resume cache and checkpoint cohort_weighting metadata differ")
    state["physiology_cohort_weighting"] = copy.deepcopy(metadata)
    state["physiology_support_gate"] = copy.deepcopy(gate)
    return metadata


def _w_pat(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    patient_id: str,
    *,
    labeled: bool,
) -> float:
    if labeled:
        return 1.0
    rows = state.get("physiology_rows")
    if not isinstance(rows, Mapping) or patient_id not in rows:
        raise ValueError(f"no frozen physiology weight exists for U patient {patient_id!r}")
    value = float(rows[patient_id]["w_pat"])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("cached Q-3D w_pat must be finite and in [0,1]")
    return value


def _new_patient_omega(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    patient_id: str,
    *,
    labeled: bool,
) -> float:
    """Return the frozen full-U patient multiplier for one KD frame."""

    if labeled:
        return 1.0
    policy_arm = _cohort_policy_arm(config)
    if policy_arm is None:
        return 1.0
    metadata = state.get("physiology_cohort_weighting")
    if not isinstance(metadata, Mapping):
        # Direct unit probes may call this helper before the stage gate.  Use
        # the complete rows map if available, while real resume paths are
        # still rejected by _validate_resumed_physiology_cache.
        rows = state.get("physiology_rows")
        if not isinstance(rows, Mapping) or not rows:
            raise RuntimeError("Q-3D U KD has no frozen cohort omega metadata")
        source = {
            str(patient): float(row["w_pat"])
            for patient, row in rows.items()
            if isinstance(row, Mapping) and "w_pat" in row
        }
        if set(source) != set(rows):
            raise RuntimeError("Q-3D U KD has malformed physiology rows")
        try:
            computed = cohort_omega(source, policy_arm)
        except ValueError as exc:
            raise RuntimeError("Q-3D U KD cohort omega gate failed") from exc
        metadata = {"omega": computed["omega"]}
    omega = metadata.get("omega")
    if not isinstance(omega, Mapping) or patient_id not in omega:
        raise RuntimeError(f"Q-3D U KD has no frozen omega for patient {patient_id!r}")
    value = float(omega[patient_id])
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"Q-3D U KD has invalid frozen omega for patient {patient_id!r}")
    return value


def _require_positive_u_physiology_support(
    config: Mapping[str, Any],
    data: Q3DData,
    rows: Mapping[str, Any],
    state: dict[str, Any],
    *,
    context: str,
    synthetic_override: bool = False,
    require_existing: bool = False,
) -> dict[str, Any] | None:
    """Fail closed when a weighted arm has no eligible full-U cohort."""
    arm = str(config["arm"])
    has_kd = (arm == "c1" and float(config["losses"].get("continuous_feature_kd", 0.0)) > 0.0) or (
        arm in {"v1", "v1u", "b1", "e1", "o1", "s1"}
        and float(config["losses"].get("codekd", 0.0)) > 0.0
    )
    if _cohort_policy_arm(config) is None or not has_kd:
        return None
    if not isinstance(state.get("physiology_rows"), Mapping) and isinstance(rows, Mapping):
        state["physiology_rows"] = copy.deepcopy(dict(rows))
    metadata = _cohort_weighting_metadata(
        config, data, state, context=context,
        synthetic_override=synthetic_override,
        require_existing=require_existing,
    )
    assert metadata is not None
    gate = copy.deepcopy(metadata["gate"])
    state.setdefault("physiology_support_gates", []).append(gate)
    if gate["override_used"]:
        state["synthetic_gate_override_used"] = True
    return gate


def _cell_weights(
    config: Mapping[str, Any],
    confidence: Tensor,
    patient_id: str,
    state: Mapping[str, Any],
    *,
    labeled: bool,
) -> Tensor:
    """Return confidence-only cell weights; patient omega is separate."""

    if config["losses"]["teacher_confidence_weighting"]:
        threshold = float(config["physiology"]["certainty_threshold"])
        cell = confidence * (confidence >= threshold).to(confidence.dtype)
    else:
        cell = torch.ones_like(confidence)
    return cell.detach()


def _distillation_or_zero(
    loss_function,
    teacher_target: Tensor,
    student_target: Tensor,
    classes: Tensor,
    weights: Tensor,
    *,
    omega: float = 1.0,
) -> tuple[Tensor, dict[str, Any]]:
    """Reduce per-cell KD on each frame, then apply its patient multiplier."""

    per_cell = loss_function(teacher_target.detach(), student_target)
    if per_cell.ndim != 4 or classes.shape != per_cell.shape or weights.shape != per_cell.shape:
        raise ValueError("KD maps, classes and confidence must share [B,D,h,w]")
    frame_losses, frames = [], []
    for loss_map, class_map, confidence in zip(per_cell, classes.detach(), weights.detach()):
        reduced, stats = weighted_class_balanced_reduce(loss_map, class_map, confidence, omega)
        frame_losses.append(reduced)
        frames.append(stats)
    if not frame_losses:
        raise ValueError("KD batch must contain at least one frame")
    loss = torch.stack(frame_losses).mean().to(per_cell.dtype)
    active = sorted({c for frame in frames for c in frame["active_classids"]})
    return loss, {
        "frames": frames, "used_classes": active,
        "zero_weight_classes": sorted(set(range(4)) - set(active)),
        "omega": float(omega),
        "base_loss": sum(frame["base_loss"] for frame in frames) / len(frames),
    }


def _kd_stats_has_support(stats: Mapping[str, Any]) -> bool:
    """Check reducer support without treating a graph-zero loss as skipped."""

    frames = stats.get("frames")
    if isinstance(frames, Sequence):
        return any(bool(frame.get("active_classids")) for frame in frames if isinstance(frame, Mapping))
    return bool(stats.get("used_classes"))


def _update_u_patient_cycle(
    state: dict[str, Any],
    patient_id: str,
    cohort: Sequence[str],
    omega: float,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Accumulate the weights actually consumed in one sampler permutation.

    The observations survive interrupted resume. O1 can change cache snapshots
    within a permutation; keep those observations rather than restarting a
    synthetic coverage cycle when its cache changes.
    """

    expected = set(str(value) for value in cohort)
    if not expected or len(expected) != len(cohort):
        raise ValueError("U patient cycle requires a nonempty unique source cohort")
    if patient_id not in expected or not math.isfinite(omega) or omega < 0:
        raise ValueError("U patient cycle has an invalid patient or omega")
    current = state.get("j_u_patient_cycle")
    if current is None:
        current = {"observations": [], "completed_cycles": 0}
    if not isinstance(current, Mapping) or not isinstance(current.get("observations"), list):
        raise ValueError("U patient cycle checkpoint lacks v3 observations")
    observations = copy.deepcopy(current["observations"])
    completed = current.get("completed_cycles")
    if type(completed) is not int or completed < 0:
        raise ValueError("U patient cycle completed count is invalid")
    seen = [row["patient"] for row in observations]
    if len(set(seen)) != len(seen) or not set(seen).issubset(expected) or patient_id in seen:
        raise ValueError("U permutation contains a repeated or unknown patient")
    cycle_complete = len(observations) + 1 == len(expected)
    if (sample.get("unlabeled_cycle") != completed + 1
            or sample.get("unlabeled_cycle_position") != len(observations) + 1
            or sample.get("unlabeled_cycle_complete") is not cycle_complete):
        raise ValueError("U patient cycle checkpoint differs from sampler permutation")
    observations.append({
        "patient": patient_id, "omega": float(omega),
        "snapshot": (state.get("physiology_cache") or {}).get("sha256"),
    })
    values = [row["omega"] for row in observations]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("U patient cycle checkpoint contains invalid omega observations")
    snapshots = list(dict.fromkeys(row["snapshot"] for row in observations))
    result = {
        "seen_patient_count": len(observations),
        "cohort_patient_count": len(expected),
        "completed_cycles": completed + int(cycle_complete),
        "cycle_complete": cycle_complete,
        "snapshot": snapshots[0] if len(snapshots) == 1 else None,
        "snapshots": snapshots,
        "realized_mean_omega": math.fsum(values) / len(values),
        "realized_max_omega": max(values),
        "realized_zero_fraction": sum(value == 0 for value in values) / len(values),
    }
    state["j_u_patient_cycle"] = {
        "observations": [] if cycle_complete else observations,
        "completed_cycles": result["completed_cycles"],
    }
    return result


def _joint_objective_scales(
    *,
    has_unlabeled_stream: bool,
    has_distillation: bool,
    has_commitment: bool,
) -> dict[str, float | int | str]:
    """Define logical-pair accumulation and per-term stream averaging."""
    microbatches = 1 + int(has_unlabeled_stream)
    return {
        "logical_update_unit": "L_U_pair" if has_unlabeled_stream else "L_only",
        "accumulation_unit": "stream_microbatch",
        "accumulation_microbatches": microbatches,
        "optimizer_step_per_logical_update": 1,
        "supervised_l_coefficient": 1.0,
        "kd_stream_divisor": float(microbatches if has_distillation else 1),
        "commitment_stream_divisor": float(microbatches if has_commitment else 1),
    }


def _teacher_assignments(
    teacher: nn.Module,
    tokenizer: nn.Module,
    image: Tensor,
    geometry: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    with torch.no_grad(), _autocast(device):
        outputs, tokenized = _tokenizer_forward(
            teacher, tokenizer, image, geometry, config=config, device=device,
        )
    if not tokenized.get("initialized") or tokenized.get("pi") is None:
        raise RuntimeError("joint codeKD requires the frozen tokenizer codebook from stage Q")
    probs, classes, confidence = _pooled_teacher_targets(
        outputs["region_logits"], target_hw=tuple(tokenized["pi"].shape[-2:]),
        student_phase=outputs["student_phase"],
    )
    return outputs, {
        "pi": tokenized["pi"].detach(),
        "codes": tokenized["codes"].detach(),
        "class_probs": probs.detach(),
        "classes": classes.detach(),
        "confidence": confidence.detach(),
        "feature": _aligned_teacher_output(outputs, image.shape[-3:], _teacher_feature_source(config)).detach(),
    }


def _teacher_outputs_for_joint(
    teacher: nn.Module,
    image: Tensor,
    geometry: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    need_feature: bool,
    device: torch.device,
) -> dict[str, Tensor]:
    with torch.no_grad(), _autocast(device):
        return _teacher_outputs(
            teacher, image, geometry, with_feature=need_feature,
            source=_teacher_feature_source(config),
            synthetic=bool(config["runtime"]["synthetic"]),
        )


def _mean_teacher_update(ema_model: nn.Module, student: nn.Module, decay: float) -> None:
    with torch.no_grad():
        ema_state = ema_model.state_dict()
        online_state = student.state_dict()
        non_tensor_state: dict[str, Any] = {}
        for name, target in ema_state.items():
            source = online_state[name]
            if not isinstance(target, Tensor) or not isinstance(source, Tensor):
                if not isinstance(target, Tensor) and not isinstance(source, Tensor):
                    non_tensor_state[name] = copy.deepcopy(source)
                continue
            # Frozen tokenizer C0 is an artifact shared by the online and EMA
            # students, not a parameter to exponentially average.
            if name == "C0" or name.endswith(".C0"):
                target.copy_(source.detach())
            elif target.is_floating_point():
                target.mul_(decay).add_(source.detach(), alpha=1.0 - decay)
            else:
                target.copy_(source.detach())
        if non_tensor_state:
            # Module extra_state carries the codebook version/SHA identity.
            # Loading only those entries leaves EMA parameters untouched.
            ema_model.load_state_dict(non_tensor_state, strict=False)


def _student_code_histogram(output: Mapping[str, Any], codebook_size: int) -> list[int]:
    codes = output.get("codes")
    if not isinstance(codes, Tensor) or codes.numel() == 0:
        raise ValueError("VQ student output must contain nonempty discrete codes")
    flat = codes.detach().reshape(-1)
    if flat.is_floating_point() and not bool(torch.equal(flat, flat.round())):
        raise ValueError("VQ student code IDs must be integral")
    ids = flat.to(dtype=torch.long)
    if int(ids.min().item()) < 0 or int(ids.max().item()) >= codebook_size:
        raise ValueError("VQ student emitted a code ID outside the frozen codebook")
    return torch.bincount(ids.cpu(), minlength=codebook_size).tolist()


def _update_student_code_usage(
    state: dict[str, Any],
    histograms: Sequence[Sequence[int]],
    *,
    codebook_size: int,
    ramp_complete: bool,
) -> dict[str, Any]:
    if not histograms:
        raise ValueError("student VQ usage requires at least one stream histogram per J update")
    update_hist = np.zeros(codebook_size, dtype=np.int64)
    for histogram in histograms:
        values = np.asarray(histogram, dtype=np.int64)
        if values.shape != (codebook_size,) or (values < 0).any():
            raise ValueError("student VQ usage histogram has invalid shape or negative counts")
        update_hist += values
    if int(update_hist.sum()) <= 0:
        raise ValueError("student VQ usage histogram contains no assigned codes")

    usage_state = state.setdefault("j_student_code_usage", {
        "schema": "self_audit.candidate_q3d.student_code_usage.v1",
        "window_size": _STUDENT_CODE_COLLAPSE_WINDOW,
        "histograms": [],
        "updates_seen": 0,
    })
    if (not isinstance(usage_state, dict)
            or usage_state.get("schema") != "self_audit.candidate_q3d.student_code_usage.v1"
            or usage_state.get("window_size") != _STUDENT_CODE_COLLAPSE_WINDOW):
        raise ValueError("student VQ usage resume state schema/window mismatch")
    history = list(usage_state.get("histograms", []))
    history.append(update_hist.tolist())
    history = history[-_STUDENT_CODE_COLLAPSE_WINDOW:]
    usage_state["histograms"] = history
    usage_state["updates_seen"] = int(usage_state.get("updates_seen", 0)) + 1

    window_hist = np.asarray(history, dtype=np.float64).sum(axis=0)
    total = float(window_hist.sum())
    probabilities = window_hist[window_hist > 0] / total
    perplexity = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    active_codes = int(np.count_nonzero(window_hist))
    complete_window = len(history) == _STUDENT_CODE_COLLAPSE_WINDOW
    evaluated = complete_window and ramp_complete
    collapse = bool(evaluated and active_codes == 1 and math.isclose(perplexity, 1.0, rel_tol=0.0, abs_tol=1e-12))
    status = "COLLAPSE_FAIL" if collapse else ("NOT_COLLAPSED" if evaluated else "NOT_EVALUATED")
    return {
        "status": status,
        "collapse_detected": collapse,
        "evaluated": evaluated,
        "ramp_complete": bool(ramp_complete),
        "window_updates": len(history),
        "window_required": _STUDENT_CODE_COLLAPSE_WINDOW,
        "active_codes": active_codes,
        "perplexity": perplexity,
        "updates_seen": int(usage_state["updates_seen"]),
    }


def _make_o1_teacher_optimizer(teacher: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    multiplier = float(training["teacher_joint_lr_multiplier"])
    return torch.optim.AdamW(
        _teacher_parameter_groups(teacher, config, multiplier=multiplier),
        weight_decay=float(training["weight_decay"]),
    )


def _refresh_o1_physiology_and_churn(
    config: Mapping[str, Any],
    *,
    data: Q3DData,
    teacher: nn.Module,
    tokenizer: nn.Module,
    state: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    epoch: int,
) -> dict[str, float | None]:
    from .physiology import physiology_weight_for_patient, save_physiology_cache

    teacher_sha = _state_sha256(teacher)
    codebook_sha = tokenizer.codebook_sha256
    physiology = config["physiology"]
    physiology_config = {
        "tau_sv": physiology["tau_stroke_volume"],
        "tau_m": physiology["tau_myocardium"],
        "stroke_balance": physiology["stroke_balance"],
        "has_rv": True,
        "minimum_lv_ed_ml": physiology["minimum_lv_ed_ml"],
        "ef_range_percent": physiology["ef_range_percent"],
    }
    rows = []
    churns: list[float] = []
    patient_churn: dict[str, float | None] = {}
    prior_snapshots = dict(state.get("o1_code_snapshots", {}))
    current_snapshots: dict[str, Tensor] = {}
    teacher.eval()
    teacher.requires_grad_(False)
    for patient_id in data.unlabeled_patients:
        row = physiology_weight_for_patient(
            teacher, data, patient_id, teacher_sha256=teacher_sha,
            codebook_sha256=codebook_sha, protocol=config["protocol"]["variant"],
            config=physiology_config, device=device,
        )
        rows.append(row)
        patient_keys = [key for key in data.unlabeled_frame_keys()
                        if data.unlabeled_patient_for_key(key) == patient_id]
        if not patient_keys:
            raise ValueError(f"O1 churn check has no U frames for {patient_id}")
        image, geometry, _metadata = _load_u(data, patient_keys[0])
        image = image.to(device=device, memory_format=torch.channels_last_3d)
        _outputs, assignments = _teacher_assignments(
            teacher, tokenizer, image, geometry, config, device=device,
        )
        current_codes = assignments["codes"].detach().cpu().long()
        previous_codes = prior_snapshots.get(patient_id)
        churn: float | None = None
        if isinstance(previous_codes, Tensor) and previous_codes.shape == current_codes.shape:
            churn = float((previous_codes != current_codes).float().mean().item())
            churns.append(churn)
        patient_churn[patient_id] = churn
        current_snapshots[patient_id] = current_codes
    state["physiology_rows"] = {row["id"]: row for row in rows}
    physiology_gate = _require_positive_u_physiology_support(
        config, data, state["physiology_rows"], state,
        context="O1 physiology cache refresh",
        synthetic_override=bool(state.get("synthetic_gate_override_requested", False)),
    )
    if physiology_gate is not None:
        _record_metric(state, output_dir / "metrics.jsonl", {
            "stage": "J", "event": "physiology_support_gate_refresh", **physiology_gate,
        })
        if not physiology_gate["passed"] and not physiology_gate["override_used"]:
            raise RuntimeError("O1 physiology cache refresh failed the frozen U cohort gate")
    cache_path = output_dir / "physiology_cache.json"
    source_identity = data.cache_source_identity()
    checksum = save_physiology_cache(
        cache_path, rows, teacher_sha256=teacher_sha, codebook_sha256=codebook_sha,
        protocol=config["protocol"]["variant"], refreshed_epoch=epoch,
        config=physiology_config, source_identity=source_identity,
    )
    from .physiology import summarize_physiology_cohort

    cohort_summary = summarize_physiology_cohort(rows)
    cohort_metadata = state.get("physiology_cohort_weighting")
    state["o1_code_snapshots"] = current_snapshots
    state["physiology_cache"] = {"path": str(cache_path), "sha256": checksum,
                                  "teacher_sha256": teacher_sha, "codebook_sha256": codebook_sha,
                                  "refreshed_epoch": epoch, "source_identity": source_identity,
                                  "config": physiology_config, "cohort_weighting": cohort_summary}
    teacher.requires_grad_(True)
    return {"mean_assignment_churn": float(np.mean(churns)) if churns else None,
            "patient_churn": patient_churn,
            "cohort_weighting": cohort_metadata,
            "cache_sha256": checksum}


def _run_joint_stage(
    config: Mapping[str, Any],
    *,
    data: Q3DData,
    teacher: nn.Module | None,
    student: nn.Module,
    tokenizer: nn.Module | None,
    ema_model: nn.Module | None,
    state: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    spatial_rng: random.Random,
    intensity_generator: torch.Generator,
    stage_rng: random.Random,
    sampler: Q3DJointSampler,
    smoke: bool,
    update_limit_active: bool,
    dev_evaluator=None,
    update_budget: int,
) -> int:
    stage = "J"
    if update_budget <= 0:
        raise ValueError("Q-3D stage J update budget must be positive")
    if state.get("stage") != stage:
        state["stage"] = stage
        state["stage_cursor"] = 0
    cursor = int(state["stage_cursor"])
    if state.get("sampler_state") is not None:
        sampler.load_state_dict(state["sampler_state"])

    arm = str(config["arm"])
    is_mean_teacher = arm == "mean_teacher"
    is_o1 = arm == "o1"
    wants_codekd = arm in {"v1", "v1u", "b1", "e1", "o1", "s1"} \
        and float(config["losses"]["codekd"]) > 0.0
    wants_feature_kd = arm == "c1" and float(config["losses"]["continuous_feature_kd"]) > 0.0
    wants_distillation = wants_codekd or wants_feature_kd
    # VQ commitment is also trained on the U stream in V0, even though V0
    # removes the teacher codeKD factor.
    wants_u = is_mean_teacher or wants_distillation or arm in _VQ_ARMS
    physiology_gate = _require_positive_u_physiology_support(
        config, data, state.get("physiology_rows") or {}, state,
        context="J physiology cache reload",
        synthetic_override=bool(state.get("synthetic_gate_override_requested", False)),
        require_existing=(cursor > 0 or isinstance(state.get("physiology_cache"), Mapping)),
    )
    if physiology_gate is not None:
        _record_metric(state, output_dir / "metrics.jsonl", {
            "stage": "J", "event": "physiology_support_gate", **physiology_gate,
        })
        weighting = state.get("physiology_cohort_weighting")
        if isinstance(weighting, Mapping):
            _record_metric(state, output_dir / "metrics.jsonl", {
                "stage": "J", "event": "physiology_weighting",
                "source_cohort": weighting.get("source_cohort"),
                "arm": weighting.get("arm"),
                "mean_w_pat": weighting.get("mean_w_pat"),
                "ess": weighting.get("ess"),
                "ess_fraction": weighting.get("ess_fraction"),
                "mean_omega": weighting.get("mean_omega"),
                "max_omega": weighting.get("max_omega"),
                "zero_fraction": weighting.get("zero_fraction"),
                "gate_passed": weighting.get("gate_passed"),
                "override_used": weighting.get("override_used"),
                "override_policy": weighting.get("override_policy"),
            })
        if not physiology_gate["passed"] and not physiology_gate["override_used"]:
            raise RuntimeError(
                "J physiology cache reload has zero support for every TRAIN U patient; "
                "physiology-weighted KD cannot enter J"
            )
    if sampler.require_unlabeled != wants_u:
        raise ValueError("Q-3D J sampler L/U mode does not match the active arm objective")
    if wants_codekd and (teacher is None or tokenizer is None or not tokenizer.codebook_frozen):
        raise ValueError("codeKD stage J requires frozen teacher tokenizer outputs from stage Q")
    if wants_feature_kd and teacher is None:
        raise ValueError("C1 continuous feature KD requires the frozen ResEnc teacher")
    if is_mean_teacher and ema_model is None:
        raise RuntimeError("Mean Teacher baseline is missing its student EMA model")

    if wants_codekd:
        assert tokenizer is not None
    if tokenizer is not None and getattr(student, "use_vq", False):
        expected_artifact = {"codes": tokenizer.C0.detach(), **tokenizer.artifact_metadata()}
        installed = student.install_codebook(expected_artifact)
        if (installed.get("sha256") != tokenizer.codebook_sha256
                or installed.get("version") != tokenizer.artifact_metadata().get("version")
                or installed.get("frozen") is not True):
            raise ValueError("student C0 installation does not match the frozen tokenizer artifact")
        if ema_model is not None and getattr(ema_model, "use_vq", False):
            ema_artifact = ema_model.install_codebook(expected_artifact)
            if ema_artifact.get("sha256") != tokenizer.codebook_sha256:
                raise ValueError("EMA student codebook differs from frozen tokenizer C0")
    if teacher is not None and not is_o1:
        _teacher_stage(teacher, "joint", freeze=True)
    if tokenizer is not None and not tokenizer.codebook_frozen:
        raise RuntimeError("stage J requires tokenizer A_T and C0 frozen after Q")
    if ema_model is not None:
        ema_model.to(device).eval().requires_grad_(False)
    student.train(True)

    optimizer = _make_optimizer(
        student.parameters(), lr=float(config["training"]["student_lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    _optimizer_states(state, "student_J", optimizer)
    base_lrs = [float(config["training"]["student_lr"]) for _ in optimizer.param_groups]
    teacher_optimizer = None
    if is_o1:
        assert teacher is not None
        _teacher_stage(teacher, "pretrain", freeze=False)
        teacher_optimizer = _make_o1_teacher_optimizer(teacher, config)
        _optimizer_states(state, "teacher_O1", teacher_optimizer)
    frozen_teacher = _clone_trainable_snapshot(teacher) if teacher is not None and not is_o1 else None
    frozen_tokenizer = _clone_trainable_snapshot(tokenizer)
    frozen_student_codebook = student.C0.detach().cpu().clone() if getattr(student, "use_vq", False) else None
    interval = int(config["training"]["checkpoint_evaluation_interval"])
    checkpoint_every = _checkpoint_every(config, smoke=smoke, has_update_limit=update_limit_active)
    dev_manifest = config["dataset"]["manifests"]["development"]
    if not isinstance(dev_manifest, Mapping) or not dev_manifest.get("path") or not dev_manifest.get("sha256"):
        raise ValueError("active Q-3D student stage requires a pinned VAL manifest for checkpoint selection")
    learning = config["losses"]
    ramp_updates = max(1, int(math.ceil(update_budget * float(learning["codekd_ramp_fraction"]))))
    sdf_lambda = float(learning["balanced_sdf"])
    intensity_config = _intensity_config(config)
    labeled_keys = data.labeled_frame_keys()
    unlabeled_keys = data.unlabeled_frame_keys() if wants_u else []
    if not labeled_keys or (wants_u and not unlabeled_keys):
        raise ValueError("Q-3D joint stage lacks the TRAIN streams required by this arm")
    updates_per_epoch = max(len(labeled_keys), len(data.train_patients), 1)
    refresh_every = int(config["physiology"]["cache_refresh_joint_epochs"] or 0) * updates_per_epoch
    while cursor < update_budget:
        update_record = sampler.sample_update()
        l_image, l_labels, l_geometry, l_metadata = _sample_parts(
            data, update_record["labeled_key"], teacher=False,
        )
        l_image, l_labels, l_geometry = augment_labeled(
            l_image, l_labels, l_geometry, rng=spatial_rng,
            intensity_config=intensity_config, intensity_generator=intensity_generator,
            flip_probability=float(config["augmentation"]["flip_y_probability"]),
            rotation_probability=float(config["augmentation"]["rotate90_probability"]),
        )
        l_image = l_image.to(device=device, memory_format=torch.channels_last_3d)
        l_labels = l_labels.to(device=device)
        l_sdf_target = _teacher_loss_targets(l_labels, l_geometry, config).to(device=device)
        l_patient = str(l_metadata["patient"])

        u_image = u_geometry = u_metadata = None
        u_student_image = None
        u_cycle_record: dict[str, Any] | None = None
        u_cycle_weighting: dict[str, Any] | None = None
        if wants_u:
            u_image, u_geometry, u_metadata = _load_u(data, update_record["unlabeled_key"])
            weak_u, strong_u, u_geometry = augment_unlabeled(
                u_image, u_geometry, rng=spatial_rng,
                intensity_config=intensity_config, intensity_generator=intensity_generator,
                flip_probability=float(config["augmentation"]["flip_y_probability"]),
                rotation_probability=float(config["augmentation"]["rotate90_probability"]),
            )
            # The shared augmentation primitive always constructs a strong
            # candidate. The frozen config pins the use probability at 0.5.
            if stage_rng.random() >= float(config["augmentation"]["intensity_probability_unlabeled_student"]):
                strong_u = weak_u
            u_image = weak_u.to(device=device, memory_format=torch.channels_last_3d)
            u_student_image = strong_u.to(device=device, memory_format=torch.channels_last_3d)

        optimizer.zero_grad(set_to_none=True)
        optimizer_steps_before = int(state.get("j_optimizer_updates", 0))
        update_losses: dict[str, float] = {}

        # O1 trains its teacher online on the same labeled frame, at 0.1 of
        # the stage-T learning rates, before creating detached targets.
        if is_o1:
            assert teacher is not None and teacher_optimizer is not None
            _teacher_stage(teacher, "pretrain", freeze=False)
            teacher.train()
            teacher_optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                online_teacher_outputs = teacher(l_image, geometry=l_geometry)
                online_teacher_loss, _online_parts = q3d_teacher_loss(
                    online_teacher_outputs, l_labels, l_sdf_target,
                    lambda_distance=float(learning["balanced_sdf"]),
                    radius_mm=float(learning["sdf_radius_mm"]),
                    near_mm=float(learning["sdf_near_boundary_mm"]),
                    alpha=float(learning["sdf_near_far_alpha"]),
                )
            online_teacher_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in teacher_optimizer.param_groups for parameter in group["params"]],
                float(config["training"]["gradient_clip_norm"]),
            )
            teacher_optimizer.step()
            update_losses["teacher_online"] = float(online_teacher_loss.detach().float().item())
            _teacher_stage(teacher, "joint", freeze=True)

        objective_scales = _joint_objective_scales(
            has_unlabeled_stream=wants_u,
            has_distillation=wants_distillation,
            has_commitment=arm in _VQ_ARMS,
        )
        stream_count = int(objective_scales["accumulation_microbatches"])
        kd_stream_divisor = float(objective_scales["kd_stream_divisor"])
        commitment_stream_divisor = float(objective_scales["commitment_stream_divisor"])
        ramp = min(1.0, cursor / float(ramp_updates))
        kd_supported_branches = 0
        kd_zero_weight_branches: list[str] = []
        student_code_histograms: list[list[int]] = []

        l_teacher_outputs: dict[str, Tensor] | None = None
        l_assignments: dict[str, Tensor] | None = None
        if wants_distillation:
            assert teacher is not None
            if wants_codekd:
                assert tokenizer is not None
                l_teacher_outputs, l_assignments = _teacher_assignments(
                    teacher, tokenizer, l_image, l_geometry, config, device=device,
                )
            else:
                l_teacher_outputs = _teacher_outputs_for_joint(
                    teacher, l_image, l_geometry, config, need_feature=True, device=device,
                )

        with _autocast(device):
            l_output = student(l_image, geometry=l_geometry)
            if arm in _VQ_ARMS:
                student_code_histograms.append(
                    _student_code_histogram(l_output, int(student.C0.shape[0]))
                )
            l_supervised, l_parts = student_supervised_loss(
                l_output["logits"], l_output.get("sdf"), l_labels, l_sdf_target,
                lambda_distance=sdf_lambda,
                radius_mm=float(learning["sdf_radius_mm"]),
                near_mm=float(learning["sdf_near_boundary_mm"]),
                alpha=float(learning["sdf_near_far_alpha"]),
            )
            l_loss = l_supervised
            l_kd_value = 0.0
            l_omega = _new_patient_omega(config, state, l_patient, labeled=True)
            if wants_codekd:
                assert l_assignments is not None and l_output.get("pi") is not None
                l_weights = _cell_weights(
                    config, l_assignments["confidence"], l_patient, state, labeled=True,
                )
                kd, l_kd_stats = _distillation_or_zero(
                    codekd_loss, l_assignments["pi"], l_output["pi"],
                    l_assignments["classes"], l_weights, omega=l_omega,
                )
                if _kd_stats_has_support(l_kd_stats):
                    kd_supported_branches += 1
                else:
                    kd_zero_weight_branches.append("L")
                l_kd_value = float(kd.detach().float().item())
                update_losses["labeled_kd_stats"] = l_kd_stats
                update_losses["labeled_kd_omega"] = l_omega
                l_loss = l_loss + ramp * float(learning["codekd"]) * kd / kd_stream_divisor
            elif wants_feature_kd:
                assert l_teacher_outputs is not None
                l_features = _aligned_teacher_output(
                    l_teacher_outputs, l_image.shape[-3:], _teacher_feature_source(config),
                )
                student_feature = l_output.get("feature_kd")
                if not isinstance(student_feature, Tensor):
                    raise ValueError("C1 requires the train-only continuous_feature_head output")
                probs, cls, confidence = _pooled_teacher_targets(
                    l_teacher_outputs["region_logits"], target_hw=tuple(student_feature.shape[-2:]),
                    student_phase=l_teacher_outputs["student_phase"],
                )
                if l_features.shape[1] != student_feature.shape[1]:
                    raise ValueError(
                        "C1 requires matching teacher/student continuous feature channels; "
                        "no unverified projection is inserted"
                    )
                weights = _cell_weights(config, confidence, l_patient, state, labeled=True)
                kd, _kd_stats = _distillation_or_zero(
                    feature_kd_loss, l_features, student_feature, cls, weights, omega=l_omega,
                )
                if _kd_stats_has_support(_kd_stats):
                    kd_supported_branches += 1
                else:
                    kd_zero_weight_branches.append("L")
                l_kd_value = float(kd.detach().float().item())
                update_losses["labeled_kd_stats"] = _kd_stats
                update_losses["labeled_kd_omega"] = l_omega
                l_loss = l_loss + ramp * float(learning["continuous_feature_kd"]) * kd / kd_stream_divisor
            if arm in _VQ_ARMS:
                if l_output.get("u") is None or l_output.get("quantized") is None:
                    raise ValueError("VQ student arm did not return u and quantized values")
                l_commitment = commitment_loss(l_output["u"], l_output["quantized"])
                l_loss = l_loss + float(learning["commitment_beta"]) * l_commitment / commitment_stream_divisor
            else:
                l_commitment = l_output["logits"].new_zeros(())

        if not torch.isfinite(l_loss):
            raise FloatingPointError("Q-3D labeled joint loss is nonfinite")
        l_loss.backward()
        update_losses.update({
            "labeled_total": float(l_loss.detach().float().item()),
            "labeled_segmentation": float(l_parts["segmentation"].float().item()),
            "labeled_sdf_near_far": float(l_parts["sdf_near_far"].float().item()),
            "labeled_distillation": l_kd_value,
            "labeled_commitment": float(l_commitment.detach().float().item()),
        })
        del l_output, l_loss, l_supervised, l_teacher_outputs, l_assignments
        u_loss_value = 0.0
        if wants_u:
            assert u_image is not None and u_student_image is not None and u_geometry is not None
            u_patient = str(u_metadata["patient"])
            u_omega = _new_patient_omega(config, state, u_patient, labeled=False)
            if _cohort_policy_arm(config) is not None:
                u_cycle_record = _update_u_patient_cycle(
                    state, u_patient, data.unlabeled_patients, u_omega, update_record,
                )
                if u_cycle_record["cycle_complete"]:
                    # O1 may refresh after this update. The completed cycle
                    # must report the weights its frames actually consumed.
                    u_cycle_weighting = copy.deepcopy(state.get("physiology_cohort_weighting"))
                    if (not is_o1 and u_cycle_weighting["gate_passed"]
                            and not math.isclose(u_cycle_record["realized_mean_omega"], 1.0,
                                                 rel_tol=0.0, abs_tol=1e-12)):
                        raise RuntimeError("complete frozen U permutation must have realized_mean_omega = 1")
            if is_mean_teacher:
                assert ema_model is not None
                ema_model.eval()
                with torch.no_grad(), _autocast(device):
                    ema_output = ema_model(u_image, geometry=u_geometry)
                with _autocast(device):
                    u_output = student(u_student_image, geometry=u_geometry)
                    u_consistency = mean_teacher_consistency_loss(u_output["logits"], ema_output["logits"])
                    u_loss = ramp * u_consistency
                u_loss_value = float(u_loss.detach().float().item())
                update_losses["unlabeled_mean_teacher"] = u_loss_value
            elif wants_distillation:
                assert teacher is not None
                if wants_codekd:
                    assert tokenizer is not None
                    _u_teacher_outputs, u_assignments = _teacher_assignments(
                        teacher, tokenizer, u_image, u_geometry, config, device=device,
                    )
                else:
                    u_teacher_outputs = _teacher_outputs_for_joint(
                        teacher, u_image, u_geometry, config, need_feature=True, device=device,
                    )
                    u_assignments = None
                with _autocast(device):
                    u_output = student(u_student_image, geometry=u_geometry)
                    if arm in _VQ_ARMS:
                        student_code_histograms.append(
                            _student_code_histogram(u_output, int(student.C0.shape[0]))
                        )
                    if wants_codekd:
                        assert u_assignments is not None and u_output.get("pi") is not None
                        u_weights = _cell_weights(
                            config, u_assignments["confidence"], u_patient, state,
                            labeled=False,
                        )
                        u_kd, u_kd_stats = _distillation_or_zero(
                            codekd_loss, u_assignments["pi"], u_output["pi"],
                            u_assignments["classes"], u_weights, omega=u_omega,
                        )
                        if _kd_stats_has_support(u_kd_stats):
                            kd_supported_branches += 1
                        else:
                            kd_zero_weight_branches.append("U")
                    else:
                        u_feature = _aligned_teacher_output(
                            u_teacher_outputs, u_student_image.shape[-3:],
                            _teacher_feature_source(config),
                        )
                        u_student_feature = u_output.get("feature_kd")
                        if not isinstance(u_student_feature, Tensor):
                            raise ValueError("C1 U requires the train-only continuous_feature_head output")
                        _probs, u_classes, u_confidence = _pooled_teacher_targets(
                            u_teacher_outputs["region_logits"], target_hw=tuple(u_student_feature.shape[-2:]),
                            student_phase=u_teacher_outputs["student_phase"],
                        )
                        if u_feature.shape[1] != u_student_feature.shape[1]:
                            raise ValueError("C1 teacher/student continuous feature channels do not match")
                        u_weights = _cell_weights(
                            config, u_confidence, u_patient, state, labeled=False,
                        )
                        u_kd, _u_stats = _distillation_or_zero(
                            feature_kd_loss, u_feature, u_student_feature, u_classes, u_weights,
                            omega=u_omega,
                        )
                        if _kd_stats_has_support(_u_stats):
                            kd_supported_branches += 1
                        else:
                            kd_zero_weight_branches.append("U")
                        u_kd_stats = _u_stats
                    u_loss = ramp * float(learning["codekd"] if wants_codekd else learning["continuous_feature_kd"]) * u_kd / kd_stream_divisor
                    u_commitment = u_output["logits"].new_zeros(())
                    if arm in _VQ_ARMS:
                        u_commitment = commitment_loss(u_output["u"], u_output["quantized"])
                        u_loss = u_loss + float(learning["commitment_beta"]) * u_commitment / commitment_stream_divisor
                    update_losses["unlabeled_kd_stats"] = u_kd_stats
                if not torch.isfinite(u_loss):
                    raise FloatingPointError("Q-3D unlabeled joint loss is nonfinite")
                u_loss_value = float(u_loss.detach().float().item())
                # Keep the raw weighted KD term separate from the total U
                # objective (which also contains ramp/divisor and commitment).
                update_losses["unlabeled_distillation"] = float(u_kd.detach().float().item())
                update_losses["unlabeled_total"] = u_loss_value
                update_losses["unlabeled_commitment"] = float(u_commitment.detach().float().item())
                update_losses["unlabeled_kd_omega"] = u_omega
            elif arm in _VQ_ARMS:
                # V0 has no teacher KD term but retains S4 commitment on both
                # L and U, as specified by the joint loss table.
                with _autocast(device):
                    u_output = student(u_student_image, geometry=u_geometry)
                    if arm in _VQ_ARMS:
                        student_code_histograms.append(
                            _student_code_histogram(u_output, int(student.C0.shape[0]))
                        )
                    if u_output.get("u") is None or u_output.get("quantized") is None:
                        raise ValueError("V0 student did not return its native VQ adapter output")
                    u_commitment = commitment_loss(u_output["u"], u_output["quantized"])
                    u_loss = float(learning["commitment_beta"]) * u_commitment / commitment_stream_divisor
                if not torch.isfinite(u_loss):
                    raise FloatingPointError("Q-3D V0 U commitment is nonfinite")
                u_loss_value = float(u_loss.detach().float().item())
                update_losses["unlabeled_commitment"] = u_loss_value
                update_losses["unlabeled_total"] = u_loss_value
            else:
                raise RuntimeError("Q-3D U stream was requested without a named unlabeled objective")
            # Even a zero-support reducer result is graph-connected to the
            # student output; always traverse backward so optimizer/update
            # accounting remains truthful and hooks observe the branch.
            u_loss.backward()
            del u_output, u_loss

        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]],
            float(config["training"]["gradient_clip_norm"]),
        )
        _schedule_lr(
            optimizer, step=cursor, total=update_budget, base_lrs=base_lrs,
            warmup_fraction=float(config["training"]["warmup_fraction"]),
        )
        optimizer.step()
        if ema_model is not None:
            # Keep the student EMA companion for every arm. Mean Teacher uses
            # it as the U target; the other ablations retain the same helper
            # state without introducing an extra objective.
            _mean_teacher_update(ema_model, student, _MEAN_TEACHER_DECAY)
        _assert_snapshot_unchanged(teacher if not is_o1 else None, frozen_teacher, "teacher")
        _assert_snapshot_unchanged(tokenizer, frozen_tokenizer, "teacher tokenizer")
        if frozen_student_codebook is not None and not torch.equal(student.C0.detach().cpu(), frozen_student_codebook):
            raise RuntimeError("Q-3D student codebook buffer changed during joint training")

        cursor += 1
        state["stage_cursor"] = cursor
        state["completed_updates_total"] += 1
        state["j_optimizer_updates"] = optimizer_steps_before + 1
        state["j_logical_updates"] = int(state.get("j_logical_updates", 0)) + 1
        state["j_stream_microbatches"] = int(state.get("j_stream_microbatches", 0)) + stream_count
        if wants_u:
            state["j_logical_pairs"] = int(state.get("j_logical_pairs", 0)) + 1
        state["j_accumulation_state"] = {
            "unit": "stream_microbatch",
            "microbatches_per_optimizer_step": stream_count,
            "microbatches_accumulated": 0,
            "logical_update_unit": "L_U_pair" if wants_u else "L_only",
            "logical_updates": int(state["j_logical_updates"]),
            "optimizer_updates": int(state["j_optimizer_updates"]),
            "stream_microbatches": int(state["j_stream_microbatches"]),
        }
        student_code_usage = None
        if student_code_histograms:
            student_code_usage = _update_student_code_usage(
                state, student_code_histograms,
                codebook_size=int(student.C0.shape[0]),
                ramp_complete=cursor >= ramp_updates,
            )
            state["j_student_code_usage_last"] = student_code_usage
        student_code_collapse = bool(student_code_usage and student_code_usage["collapse_detected"])
        if wants_distillation:
            state["j_kd_supported_branches"] = int(state.get("j_kd_supported_branches", 0)) + kd_supported_branches
            state["j_kd_zero_weight_branches"] = int(state.get("j_kd_zero_weight_branches", 0)) + len(kd_zero_weight_branches)
        if is_o1 and refresh_every and cursor % refresh_every == 0:
            assert teacher is not None and tokenizer is not None
            teacher.requires_grad_(False)
            teacher.eval()
            churn = _refresh_o1_physiology_and_churn(
                config, data=data, teacher=teacher, tokenizer=tokenizer,
                state=state, output_dir=output_dir, device=device, epoch=cursor // updates_per_epoch,
            )
            update_losses.update({f"o1_{name}": value for name, value in churn.items()})
            _teacher_stage(teacher, "pretrain", freeze=False)
        record = {
            "stage": stage,
            "arm": arm,
            "update": cursor,
            "logical_updates": cursor,
            "optimizer_updates": int(state["j_optimizer_updates"]),
            "streams_per_update": stream_count,
            "logical_update_unit": "L_U_pair" if wants_u else "L_only",
            "accumulation_unit": "stream_microbatch",
            "accumulation_microbatches": stream_count,
            "optimizer_step_per_logical_update": 1,
            "supervised_l_coefficient": 1.0,
            "kd_stream_divisor": kd_stream_divisor,
            "commitment_stream_divisor": commitment_stream_divisor,
            "distillation_ramp": ramp,
            "kd_supported_branches": kd_supported_branches,
            "kd_zero_weight_branches": kd_zero_weight_branches,
            "losses": update_losses,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "sampler": update_record,
        }
        if u_cycle_record is not None:
            record["u_patient_cycle"] = u_cycle_record
            record["u_patient_omega"] = u_omega
        if student_code_usage is not None and (
            cursor % 50 == 0 or cursor == update_budget or student_code_collapse
        ):
            record["student_code_usage"] = student_code_usage
        _record_metric(state, output_dir / "metrics.jsonl", record)
        if u_cycle_record is not None and u_cycle_record["cycle_complete"]:
            weighting = u_cycle_weighting
            if isinstance(weighting, Mapping):
                _record_metric(state, output_dir / "metrics.jsonl", {
                    "stage": stage, "event": "physiology_weighting_full_u_cycle",
                    "cycle": u_cycle_record["completed_cycles"],
                    "snapshot": u_cycle_record.get("snapshot"),
                    "snapshots": u_cycle_record["snapshots"],
                    "patient_count": u_cycle_record["cohort_patient_count"],
                    "realized_mean_omega": u_cycle_record["realized_mean_omega"],
                    "realized_max_omega": u_cycle_record["realized_max_omega"],
                    "realized_zero_fraction": u_cycle_record["realized_zero_fraction"],
                    "mean_omega": weighting.get("mean_omega"),
                    "max_omega": weighting.get("max_omega"),
                    "zero_fraction": weighting.get("zero_fraction"),
                    "ess": weighting.get("ess"),
                    "ess_fraction": weighting.get("ess_fraction"),
                    "source_cohort": weighting.get("source_cohort"),
                })
        state["sampler_state"] = sampler.state_dict()

        if student_code_collapse:
            _save_optimizer_state(state, "student_J", optimizer)
            if teacher_optimizer is not None:
                _save_optimizer_state(state, "teacher_O1", teacher_optimizer)
            _capture_run_state(
                state, teacher=teacher, student=student, tokenizer=tokenizer, ema_model=ema_model,
                spatial_rng=spatial_rng, intensity_generator=intensity_generator, stage_rng=stage_rng,
                sampler=sampler,
            )
            _atomic_torch_save(state, _stage_directory(output_dir, config, stage) / "latest.pt")
            _atomic_torch_save(state, output_dir / "latest.pt")
            raise RuntimeError(
                "Q-3D student VQ collapse: rolling post-ramp 200-update window used one code "
                "(perplexity 1.0)"
            )

        epoch_updates = len(data.unlabeled_patients) if wants_u else 2 * int(config["protocol"]["k"])
        epoch_complete = (bool(update_record["unlabeled_cycle_complete"]) if wants_u
                          else cursor % epoch_updates == 0)
        evaluate_now = cursor % interval == 0 or cursor == update_budget
        if evaluate_now:
            if dev_evaluator is None:
                dev = _development_fg_dice(
                    student, dev_manifest, sha256=dev_manifest["sha256"],
                    train_ids=data.train_patients, synthetic=bool(config["runtime"]["synthetic"]),
                    output_kind="student", device=device,
                )
            else:
                dev = dict(dev_evaluator(student, dev_manifest, "student", device))
            score = float(dev["score"])
            if not math.isfinite(score):
                raise FloatingPointError("Q-3D student VAL selection score is nonfinite")
            if state.get("best_student_score") is None or score > float(state["best_student_score"]):
                state["best_student_score"] = score
                state["best_student_state"] = copy.deepcopy(student.state_dict())
            _record_metric(state, output_dir / "metrics.jsonl", {
                "stage": stage, "event": "development_selection", "update": cursor,
                "epoch_logged": epoch_complete,
                "score": score, "best_score": float(state["best_student_score"]),
                "per_class": copy.deepcopy(dev.get("per_class")),
                "patient_count": int(dev.get("patient_count", len(dev.get("patients", {})))),
                "code_interface": dev.get("code_interface", {"status": "NOT_RUN_CUSTOM_EVALUATOR"}),
            })
        if epoch_complete:
            if not evaluate_now:
                dev = _observational_development(
                    student, dev_manifest, data=data, config=config, device=device,
                    output_kind="student", evaluator=dev_evaluator,
                )
            _record_epoch_development(
                state, output_dir, stage=stage, update=cursor,
                epoch=(int(update_record["unlabeled_cycle"]) if wants_u else cursor // epoch_updates),
                epoch_definition="complete_U_patient_permutation" if wants_u else "2K_L_only_draws",
                epoch_updates=epoch_updates, dev=dev, selection_point=evaluate_now,
            )
        if cursor % checkpoint_every == 0 or cursor == update_budget:
            _save_optimizer_state(state, "student_J", optimizer)
            if teacher_optimizer is not None:
                _save_optimizer_state(state, "teacher_O1", teacher_optimizer)
            _capture_run_state(
                state, teacher=teacher, student=student, tokenizer=tokenizer, ema_model=ema_model,
                spatial_rng=spatial_rng, intensity_generator=intensity_generator, stage_rng=stage_rng,
                sampler=sampler,
            )
            _atomic_torch_save(state, _stage_directory(output_dir, config, stage) / "latest.pt")
            _atomic_torch_save(state, output_dir / "latest.pt")
        if (update_limit_active and cursor < update_budget
                and state["completed_updates_total"] >= int(state["requested_update_limit"])):
            return cursor

    if wants_distillation and int(state.get("j_kd_supported_branches", 0)) == 0:
        raise RuntimeError("Q-3D global KD collapse: no labeled or unlabeled branch had positive distillation support")
    if state.get("best_student_state") is None:
        raise RuntimeError("Q-3D joint stage completed without a VAL-selected student checkpoint")
    student.load_state_dict(state["best_student_state"], strict=True)
    if is_o1 and teacher is not None:
        # The deployed Student has no teacher dependency; retaining the final
        # online teacher only in the research checkpoint preserves churn audit.
        teacher.requires_grad_(False)
        teacher.eval()
    state["completed_stages"].append(stage)
    state["stage"] = None
    state["stage_cursor"] = 0
    _capture_run_state(
        state, teacher=teacher, student=student, tokenizer=tokenizer, ema_model=ema_model,
        spatial_rng=spatial_rng, intensity_generator=intensity_generator, stage_rng=stage_rng,
        sampler=sampler,
    )
    _atomic_torch_save(state, _stage_directory(output_dir, config, stage) / "latest.pt")
    _atomic_torch_save(state, output_dir / "latest.pt")
    return cursor


def _validate_resumed_physiology_cache(
    config: Mapping[str, Any],
    data: Q3DData,
    state: dict[str, Any],
    *,
    output_dir: Path,
    teacher: nn.Module | None,
    tokenizer: nn.Module | None,
) -> None:
    """Reject stale checkpoints and validate the sealed frozen-U weighting."""

    if _cohort_policy_arm(config) is None:
        return
    completed = set(state.get("completed_stages", []))
    if "P" not in completed and state.get("stage") != "J" and "J" not in completed:
        return
    cache_state = state.get("physiology_cache")
    if not isinstance(cache_state, Mapping):
        raise RuntimeError("Q-3D resume requires a physiology cache identity with cohort_weighting metadata")
    cache_path = Path(str(cache_state.get("path", ""))).expanduser()
    if not cache_path.is_absolute():
        cache_path = output_dir / cache_path
    if not cache_path.is_file():
        raise RuntimeError(f"Q-3D resume physiology cache is missing: {cache_path}")
    expected_cache_sha = cache_state.get("sha256")
    if not isinstance(expected_cache_sha, str):
        raise RuntimeError("Q-3D resume physiology cache identity is missing its canonical SHA-256")
    from .physiology import load_physiology_cache

    recorded_teacher_sha = cache_state.get("teacher_sha256")
    if not isinstance(recorded_teacher_sha, str):
        raise RuntimeError("Q-3D resume physiology cache identity is missing its teacher SHA-256")
    expected_codebook_sha = (tokenizer.codebook_sha256 if tokenizer is not None
                             else state.get("shared_codebook_sha256"))
    cache_config = cache_state.get("config")
    try:
        # ``save_physiology_cache`` returns the SHA of the canonical payload,
        # not the SHA of the pretty-printed envelope bytes on disk.  Let the
        # loader verify the complete cache identity and then compare its
        # canonical document SHA with the checkpoint receipt.
        document = load_physiology_cache(
            cache_path,
            teacher_sha256=recorded_teacher_sha,
            codebook_sha256=expected_codebook_sha,
            protocol=str(config["protocol"]["variant"]),
            config=cache_config if isinstance(cache_config, Mapping) else None,
            source_identity=data.cache_source_identity(),
        )
    except (OSError, TypeError, ValueError) as exc:
        if "cohort_weighting" in str(exc):
            raise RuntimeError(
                "Q-3D resume found an old physiology cache without cohort_weighting metadata"
            ) from exc
        raise RuntimeError("Q-3D resume physiology cache failed full identity validation") from exc
    if document.get("sha256") != expected_cache_sha:
        raise RuntimeError("Q-3D resume physiology cache canonical SHA-256 differs from checkpoint")
    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        raise RuntimeError("Q-3D resume physiology cache envelope is malformed")
    cached_summary = payload.get("cohort_weighting")
    if not isinstance(cached_summary, Mapping):
        raise RuntimeError(
            "Q-3D resume found an old physiology cache without cohort_weighting metadata"
        )
    rows = state.get("physiology_rows")
    if not isinstance(rows, Mapping):
        raise RuntimeError("Q-3D resume physiology cache has no frozen patient rows")
    cached_rows = {row["id"]: row for row in payload["patients"]}
    if dict(rows) != cached_rows:
        # Mean/ESS alone cannot detect weights swapped between two patients.
        raise RuntimeError("Q-3D resume patient rows differ from the sealed physiology cache")
    from .physiology import summarize_physiology_cohort

    expected_summary = summarize_physiology_cohort([dict(rows[key]) for key in sorted(rows)])
    if dict(cached_summary) != expected_summary:
        raise RuntimeError("Q-3D resume physiology cache cohort_weighting is stale")
    if cache_state.get("cohort_weighting") != dict(cached_summary):
        raise RuntimeError("Q-3D resume checkpoint/cache cohort_weighting metadata differ")
    # A checkpoint from before runtime cohort weighting was persisted cannot
    # be resumed merely because its cache file happens to contain the newer
    # raw summary.  The runtime omega map is part of the J contract and must
    # be present in the checkpoint itself.
    if not isinstance(state.get("physiology_cohort_weighting"), Mapping):
        raise RuntimeError(
            "Q-3D resume checkpoint is missing frozen physiology cohort_weighting metadata"
        )
    _cohort_weighting_metadata(
        config, data, state, context="Q-3D resume physiology cache",
        synthetic_override=bool(state.get("synthetic_gate_override_requested", False)),
        require_existing=True,
    )
    if str(config.get("arm")) != "o1":
        if teacher is None or payload.get("teacher_sha256") != _state_sha256(teacher):
            raise RuntimeError("Q-3D resume physiology cache teacher snapshot differs")
    if payload.get("codebook_sha256") != expected_codebook_sha:
        raise RuntimeError("Q-3D resume physiology cache codebook snapshot differs")
    if payload.get("source_identity") != data.cache_source_identity():
        raise RuntimeError("Q-3D resume physiology cache source cohort differs")


def run_training(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    data: Q3DData | None = None,
    teacher: nn.Module | None = None,
    student: nn.Module | None = None,
    tokenizer: nn.Module | None = None,
    ema_model: nn.Module | None = None,
    resume_from: str | Path | None = None,
    smoke: bool | None = None,
    update_limit: int | None = None,
    max_updates: int | None = None,
    synthetic_gate_override: bool = False,
    dev_evaluator=None,
    stage: str | None = None,
    stage_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute the configured Q-3D stages with hash-bound exact resume.

    Real execution remains fail-closed on manifest, teacher-source, weight,
    and provenance validation. Synthetic runs are explicitly labeled and
    should use bounded overrides (or ``update_limit``) for software checks.
    """

    from .config import resolve_stage_updates, stages_for_arm, validate_config

    synthetic = bool(config.get("runtime", {}).get("synthetic", False)) if isinstance(config, Mapping) else False
    normalized = validate_config(config, real_run=not synthetic)
    if type(synthetic_gate_override) is not bool:
        raise TypeError("synthetic_gate_override must be a boolean")
    arm = str(normalized["arm"])
    supported = {"c0", "c1", "v1", "v1u", "v0", "b1", "e1", "o1", "s1", "mean_teacher"}
    if arm not in supported:
        raise NotImplementedError(
            f"Q-3D engine has no Q3D student implementation for baseline arm {arm!r}; "
            "A-3D teacher_free remains on the Candidate A-3D trainer"
        )
    if update_limit is not None and max_updates is not None:
        raise ValueError("supply only one of update_limit or max_updates")
    if max_updates is not None:
        update_limit = max_updates
    if update_limit is not None and (type(update_limit) is not int or update_limit <= 0):
        raise ValueError("update_limit must be a positive logical-update count")
    if smoke is None:
        smoke = synthetic
    if smoke and not synthetic:
        raise ValueError("synthetic smoke is unavailable for real patient configs")
    synthetic_gate_override = bool(
        synthetic_gate_override or normalized["runtime"]["synthetic_gate_override"]
    )
    if synthetic_gate_override and (not synthetic or not smoke):
        raise ValueError("synthetic gate override requires both runtime.synthetic=true and smoke=true")
    overrides = normalized["runtime"]["smoke_update_overrides"]
    if synthetic and not any(value is not None for value in overrides.values()) and update_limit is None:
        raise ValueError("synthetic execution requires explicit bounded stage overrides or update_limit")
    stage_updates = resolve_stage_updates(normalized, smoke=bool(smoke))
    active_stages = stages_for_arm(arm)
    if stage is not None:
        if stage not in active_stages:
            raise ValueError(f"stage {stage!r} is not active for arm {arm!r}")
        active_stages = (stage,)

    target_device = torch.device(device if device is not None else normalized["runtime"]["device"])
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("Q-3D runtime supports CPU or CUDA only")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Q-3D config requested CUDA but CUDA is unavailable")

    # Deterministic kernels and explicit RNG streams are part of the resume
    # contract. The checkpoint restores all streams after module construction.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    seed = int(normalized["seed"])
    spatial_rng, intensity_generator = _seed_all(seed)
    stage_rng = random.Random(seed + 47)
    q3d_data = _load_data(normalized, data)
    data_identity = _data_identity_without_labels(q3d_data)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    root_checkpoint = output / "latest.pt"
    if resume_from is None and root_checkpoint.is_file():
        # A root checkpoint is the canonical interrupted-run pointer. Loading
        # it is safe only after the config and data identities are verified.
        resume_from = root_checkpoint
    if resume_from is None and any(output.iterdir()):
        raise FileExistsError("Q-3D output directory is nonempty and has no resumable latest.pt")

    teacher, student, tokenizer, provenance = _build_models(
        normalized, device=target_device, teacher=teacher, student=student, tokenizer=tokenizer,
        smoke=bool(smoke),
    )
    assert student is not None
    if ema_model is None:
        ema_model = copy.deepcopy(student)
    if not isinstance(ema_model, nn.Module):
        raise TypeError("ema_model must be a torch module")
    ema_model.to(target_device).eval().requires_grad_(False)
    sampler: Q3DJointSampler | None = None

    config_sha256 = _object_sha256(normalized)
    if resume_from is None:
        state = _initial_run_state(config_sha256, data_identity)
    else:
        state = _load_checkpoint(
            resume_from, config_sha256=config_sha256, data_identity=data_identity,
        )
        _restore_run_state(
            state, teacher=teacher, student=student, tokenizer=tokenizer,
            ema_model=ema_model, spatial_rng=spatial_rng,
            intensity_generator=intensity_generator, stage_rng=stage_rng,
        )
        if state.get("sampler_state") is not None:
            sampler = _make_q3d_sampler(
                q3d_data, seed, state, require_unlabeled=(arm != "c0"),
            )
        # The codebook's initialization sample list is Python-side state, not
        # a tensor buffer, and must survive interruption before update 200.
        if tokenizer is not None and hasattr(tokenizer, "_initialization_points"):
            tokenizer._initialization_points = [
                value.detach().cpu().clone()
                for value in state.get("tokenizer_initialization_points", [])
            ]

    state["synthetic_gate_override_requested"] = bool(synthetic_gate_override)
    # A stage checkpoint can reach disk before its immutable artifact receipt.
    # Recover that publication boundary before proceeding to a dependent stage.
    # The exporter verifies/adopts an existing manifest and safely regenerates
    # only unpublished files after an interrupted export.
    if resume_from is not None:
        from .stages import export_stage_artifact

        for completed_stage in ("T", "Q", "P"):
            if (completed_stage in state.get("completed_stages", [])
                    and completed_stage not in state.get("stage_artifacts", {})):
                export_stage_artifact(
                    normalized, q3d_data, state, teacher, tokenizer, output, completed_stage,
                )
    if stage_artifacts:
        from .stages import import_stage_artifacts

        import_stage_artifacts(
            normalized, q3d_data, state, teacher, tokenizer, output, stage_artifacts,
        )
    if stage is not None:
        required = [item for item in stages_for_arm(arm)
                    if _STAGE_ORDER.index(item) < _STAGE_ORDER.index(stage)]
        missing = set(required) - set(state.get("completed_stages", []))
        if missing:
            raise ValueError(f"stage {stage} requires verified artifacts for {sorted(missing)}")
    if resume_from is not None:
        _validate_resumed_physiology_cache(
            normalized, q3d_data, state, output_dir=output,
            teacher=teacher, tokenizer=tokenizer,
        )

    if update_limit is not None:
        state["requested_update_limit"] = int(state["completed_updates_total"]) + update_limit
    else:
        state.pop("requested_update_limit", None)
    update_limit_active = update_limit is not None
    if update_limit_active:
        state["requested_update_limit"] = int(state["requested_update_limit"])

    for stage in active_stages:
        if stage in state.get("completed_stages", []):
            continue
        if update_limit_active and int(state["completed_updates_total"]) >= state["requested_update_limit"]:
            break
        if stage == "T":
            if teacher is None:
                raise RuntimeError("active Q-3D T stage has no verified teacher")
            _run_teacher_stage(
                normalized, data=q3d_data, teacher=teacher, student=student,
                tokenizer=tokenizer, ema_model=ema_model, state=state, output_dir=output,
                device=target_device, spatial_rng=spatial_rng,
                intensity_generator=intensity_generator, stage_rng=stage_rng,
                sampler=sampler, smoke=bool(smoke), update_limit_active=update_limit_active,
                dev_evaluator=dev_evaluator, update_budget=stage_updates["T"],
            )
        elif stage == "Q":
            if teacher is None or tokenizer is None:
                raise RuntimeError("active Q stage requires teacher and tokenizer modules")
            _run_tokenizer_stage(
                normalized, data=q3d_data, teacher=teacher, student=student,
                tokenizer=tokenizer, ema_model=ema_model, state=state, output_dir=output,
                device=target_device, spatial_rng=spatial_rng,
                intensity_generator=intensity_generator, stage_rng=stage_rng,
                sampler=sampler, smoke=bool(smoke), update_limit_active=update_limit_active,
                synthetic_gate_override=synthetic_gate_override,
                update_budget=stage_updates["Q"],
            )
        elif stage == "P":
            if teacher is None:
                raise RuntimeError("active P stage has no teacher")
            _run_physiology_stage(
                normalized, data=q3d_data, teacher=teacher, tokenizer=tokenizer,
                student=student, ema_model=ema_model, state=state, output_dir=output,
                device=target_device, spatial_rng=spatial_rng,
                intensity_generator=intensity_generator, stage_rng=stage_rng,
                sampler=sampler,
            )
        elif stage == "J":
            # Validate newly imported/local P artifacts as strictly as resumed
            # ones; entering J must bind the live teacher, C0 and U cohort.
            _validate_resumed_physiology_cache(
                normalized, q3d_data, state, output_dir=output,
                teacher=teacher, tokenizer=tokenizer,
            )
            if sampler is None:
                sampler = _make_q3d_sampler(
                    q3d_data, seed, state, require_unlabeled=(arm != "c0"),
                )
            _run_joint_stage(
                normalized, data=q3d_data, teacher=teacher, student=student,
                tokenizer=tokenizer, ema_model=ema_model, state=state, output_dir=output,
                device=target_device, spatial_rng=spatial_rng,
                intensity_generator=intensity_generator, stage_rng=stage_rng,
                sampler=sampler, smoke=bool(smoke), update_limit_active=update_limit_active,
                dev_evaluator=dev_evaluator, update_budget=stage_updates["J"],
            )
        else:
            raise RuntimeError(f"unknown Q-3D stage {stage!r}")
        if stage in {"T", "Q", "P"} and stage in state.get("completed_stages", []):
            from .stages import export_stage_artifact

            export_stage_artifact(
                normalized, q3d_data, state, teacher, tokenizer, output, stage,
            )

    # Even an update-limited stop gets a durable root pointer with every RNG,
    # optimizer, sampler, module, and tokenizer-init state needed to resume.
    _capture_run_state(
        state, teacher=teacher, student=student, tokenizer=tokenizer, ema_model=ema_model,
        spatial_rng=spatial_rng, intensity_generator=intensity_generator,
        stage_rng=stage_rng, sampler=sampler,
    )
    checkpoint_sha256 = _atomic_torch_save(state, root_checkpoint)
    complete = all(stage in state.get("completed_stages", []) for stage in active_stages)
    tokenizer_metadata = tokenizer.artifact_metadata() if tokenizer is not None else None
    return {
        "schema": "self_audit.candidate_q3d.training_result.v1",
        "status": "COMPLETED" if complete else "PAUSED",
        "evidence_status": "SOFTWARE_ONLY_SYNTHETIC" if synthetic else "TRAIN_EXECUTION",
        "runtime_synthetic": synthetic,
        "arm": arm,
        "active_stages": list(active_stages),
        "completed_stages": list(state.get("completed_stages", [])),
        "completed_updates": int(state.get("completed_updates_total", 0)),
        "joint_optimizer_updates": int(state.get("j_optimizer_updates", 0)),
        "config_sha256": config_sha256,
        "data_identity": data_identity,
        "teacher_provenance": provenance.get("teacher"),
        "teacher_state_sha256": _state_sha256(teacher) if teacher is not None else None,
        "teacher_dev_fg_dice": state.get("teacher_dev_fg_dice"),
        "teacher_dev_per_class": copy.deepcopy(state.get("teacher_dev_per_class")),
        "teacher_selected_dev_update": state.get("teacher_selected_dev_update"),
        "teacher_terminal_dev_fg_dice": state.get("teacher_terminal_dev_fg_dice"),
        "teacher_terminal_dev_per_class": copy.deepcopy(state.get("teacher_terminal_dev_per_class")),
        "dev_curve_still_improving": copy.deepcopy(state.get("teacher_dev_curve_still_improving")),
        "student_state_sha256": _state_sha256(student),
        "tokenizer": tokenizer_metadata,
        "tokenizer_gate": state.get("gate"),
        "synthetic_gate_override_used": bool(state.get("synthetic_gate_override_used", False)),
        "physiology_support_gate": state.get("physiology_support_gate"),
        "physiology_support_gates": list(state.get("physiology_support_gates", [])),
        "physiology_cohort_weighting": state.get("physiology_cohort_weighting"),
        "u_patient_cycle": state.get("j_u_patient_cycle"),
        "checkpoint": str(root_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "output_dir": str(output),
        "stage_artifacts": copy.deepcopy(state.get("stage_artifacts", {})),
        "stage_artifact_hashes": copy.deepcopy(state.get("stage_artifact_hashes", {})),
        "gpu_evidence": "NOT_RUN" if target_device.type == "cpu" else "RUNTIME_REQUESTED",
        "real_data_evidence": "NOT_RUN" if synthetic else "CONFIG_VALIDATED",
    }


def train(config: Mapping[str, Any], output_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Canonical trainer name used by ``train.py`` and the Q-3D smoke CLI."""

    return run_training(config, output_dir, **kwargs)


__all__ = ["run_training", "train"]
