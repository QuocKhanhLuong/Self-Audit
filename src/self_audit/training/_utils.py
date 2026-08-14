"""Small shared helpers for the three explicit training phases."""

from __future__ import annotations

import contextlib
import math
import os
import random
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Keep the model constructor boundary explicit.  Training and data settings
# live beside ``model`` in the YAML files and must not accidentally become
# constructor kwargs (``image_size`` was previously forwarded this way).
MODEL_ARCHITECTURE_KEYS = frozenset(
    {
        "encoder_name",
        "pretrained_encoder",
        "encoder_allow_fallback",
        "shared_channels",
        "num_classes",
        "window_k",
        "max_turns",
    }
)
MODEL_RUNTIME_KEYS = frozenset({"tau_accept", "threshold", "t_max"})
MODEL_CONFIG_KEYS = MODEL_ARCHITECTURE_KEYS | MODEL_RUNTIME_KEYS
CHECKPOINT_FORMAT_VERSION = 1


def _require_integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value}")
    return value


def _require_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _require_positive_real(value: Any, name: str) -> float:
    result = _require_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return result


def _require_nonnegative_real(value: Any, name: str) -> float:
    result = _require_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return result


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value


def validate_image_size(value: Any, *, default: int = 256) -> int | tuple[int, int]:
    """Validate the dataset raster size without making it a model setting."""

    value = default if value is None else value
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"image_size must be an integer or a pair, got {value!r}")
        return (
            _require_integer(value[0], "image_size[0]", minimum=1),
            _require_integer(value[1], "image_size[1]", minimum=1),
        )
    return _require_integer(value, "image_size", minimum=1)


def validate_depth_axis(value: Any, *, name: str = "depth_axis") -> int | None:
    """Validate an optional NumPy/PyTorch depth axis without guessing."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in (0, 1, 2):
        raise ValueError(f"{name} must be 0, 1, 2, or null, got {value!r}")
    return int(value)


def filter_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return only constructor kwargs from a model config.

    ``tau_accept``/``threshold``/``t_max`` are accepted as known inference
    settings for compatibility with the existing joint YAML, but are
    intentionally validated and discarded here.  Dataset ``image_size`` is
    never a model kwarg; putting it under ``model`` is an explicit error so a
    misspelled or misplaced size cannot silently change the data contract.
    """

    if not isinstance(config, Mapping):
        raise ValueError(f"model config must be a mapping, got {type(config).__name__}")
    unknown = set(config) - MODEL_CONFIG_KEYS
    if "image_size" in config:
        raise ValueError("image_size belongs to the dataset config, not model")
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"Unsupported model config key(s): {names}")

    if "encoder_name" in config and not isinstance(config["encoder_name"], str):
        raise ValueError("encoder_name must be a string")
    for key in ("pretrained_encoder", "encoder_allow_fallback"):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f"{key} must be a boolean")
    for key in ("shared_channels", "num_classes", "window_k", "max_turns"):
        if key in config:
            minimum = 2 if key == "num_classes" else 1
            _require_integer(config[key], key, minimum=minimum)
    for key in ("tau_accept", "threshold"):
        if key in config:
            _require_real(config[key], key)
    if "t_max" in config:
        _require_integer(config["t_max"], "t_max", minimum=0)

    return {key: config[key] for key in MODEL_ARCHITECTURE_KEYS if key in config}


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("PyYAML is required to load Self-Audit configs") from exc
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with open(path, encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain a mapping, got {type(value).__name__}")
    return value


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python/NumPy/PyTorch and optionally request deterministic kernels."""

    seed = _require_integer(seed, "seed", minimum=0)
    _require_bool(deterministic, "deterministic")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def resolve_device(requested: str | None) -> torch.device:
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if str(requested).startswith("mps") and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def build_model_from_config(config: dict[str, Any], device: torch.device) -> nn.Module:
    from self_audit.models.self_audit_net import build_self_audit_net

    raw_model_cfg = config.get("model", {})
    if raw_model_cfg is None:
        raw_model_cfg = {}
    model_cfg = filter_model_config(raw_model_cfg)
    model_cfg.setdefault("num_classes", config.get("num_classes", 4))
    model_cfg.setdefault("encoder_name", "convnext_tiny")
    model_cfg.setdefault("pretrained_encoder", False)
    model_cfg.setdefault("encoder_allow_fallback", True)
    model_cfg.setdefault("shared_channels", 96)
    model_cfg.setdefault("window_k", 8)
    model_cfg.setdefault("max_turns", 3)
    if not isinstance(model_cfg["encoder_name"], str) or not model_cfg["encoder_name"].strip():
        raise ValueError("model.encoder_name must be a non-empty string")
    for key in ("shared_channels", "num_classes", "window_k", "max_turns"):
        value = model_cfg[key]
        minimum = 2 if key == "num_classes" else 1
        _require_integer(value, f"model.{key}", minimum=minimum)
    for key in ("pretrained_encoder", "encoder_allow_fallback"):
        if not isinstance(model_cfg[key], bool):
            raise ValueError(f"model.{key} must be a boolean, got {model_cfg[key]!r}")
    model = build_self_audit_net(**model_cfg)
    return model.to(device)


def build_patient_dataset(
    config: dict[str, Any],
    *,
    split: str,
    train: bool,
) -> torch.utils.data.Dataset:
    from self_audit.data.acdc import ACDCDataset

    data_root = config.get("data_root", "preprocessed_data/ACDC")
    kwargs: dict[str, Any] = {
        "data_root": data_root,
        "split": split,
        "image_size": validate_image_size(config.get("image_size")),
        "augment": bool(train and config.get("augment", True)),
    }
    for key in ("split_manifest", "seed", "depth_axis", "expected_slices", "max_cache"):
        if key in config:
            kwargs[key] = config[key]
    return ACDCDataset(**kwargs)


def build_data_loader(
    dataset: Dataset,
    config: Mapping[str, Any],
    *,
    device: torch.device,
    train: bool,
    batch_size: int | None = None,
) -> DataLoader:
    """Build a reliable DataLoader without invalid worker-only arguments."""

    if not isinstance(config, Mapping):
        raise ValueError("DataLoader config must be a mapping")
    workers = _require_integer(config.get("num_workers", 0), "num_workers", minimum=0)
    size = _require_integer(batch_size if batch_size is not None else config.get("batch_size", 1), "batch_size", minimum=1)
    pin_memory = bool(config.get("pin_memory", device.type == "cuda"))
    kwargs: dict[str, Any] = {
        "batch_size": size,
        "shuffle": bool(train),
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", False))
        if "prefetch_factor" in config:
            kwargs["prefetch_factor"] = _require_integer(config["prefetch_factor"], "prefetch_factor", minimum=1)
    return DataLoader(dataset, **kwargs)


def validate_dataset_splits(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate patient-disjoint splits before a training entrypoint starts.

    An explicitly configured manifest is authoritative: missing files, extra
    discovered cases, missing requested splits, or patient overlap fail
    immediately.  A deterministic patient split is used only when no manifest
    was configured.
    """

    if not isinstance(config, Mapping):
        raise ValueError("training config must be a mapping")
    dataset_name = str(config.get("dataset", "acdc")).lower()
    if dataset_name not in {"acdc", "mnms"}:
        raise ValueError(f"Unsupported dataset {dataset_name!r}")
    if dataset_name != "acdc":
        return {"dataset": dataset_name, "validated": False, "reason": "external dataset"}
    from self_audit.data.acdc import discover_acdc_records, resolve_acdc_records
    from self_audit.data.common import (
        load_array,
        patient_id_from_case_id,
        patient_level_split,
        to_depth_first,
        validate_patient_split,
    )

    data_root = config.get("data_root", "preprocessed_data/ACDC")
    records = discover_acdc_records(data_root)
    discovered = {record.case_id for record in records}
    manifest_value = config.get("split_manifest")
    split_names = {
        "train": str(config.get("train_split", "train")),
        "val": str(config.get("val_split", "val")),
    }
    if manifest_value is not None:
        manifest_path = Path(str(manifest_value))
        from self_audit.data.common import read_split_manifest

        manifest = read_split_manifest(manifest_path)
        manifest_cases = set().union(*(set(values) for values in manifest.values()))
        if manifest_cases != discovered:
            raise ValueError(
                f"Configured split manifest does not match discovered cases: "
                f"missing={sorted(manifest_cases - discovered)[:5]}, extra={sorted(discovered - manifest_cases)[:5]}"
            )
        selected_by_split: dict[str, list[str]] = {}
        for name, requested in split_names.items():
            normalized = {"training": "train", "validation": "val", "testing": "test"}.get(requested.lower(), requested.lower())
            if normalized not in manifest:
                raise ValueError(f"Configured split manifest has no {normalized!r} split: {manifest_path}")
            selected_by_split[name] = list(manifest[normalized])
        requested_test = str(config.get("test_split", "test"))
        normalized_test = {"training": "train", "validation": "val", "testing": "test"}.get(requested_test.lower(), requested_test.lower())
        if normalized_test in manifest:
            selected_by_split["test"] = list(manifest[normalized_test])
    else:
        fallback = patient_level_split(sorted(discovered), seed=int(config.get("seed", 42)))
        selected_by_split = {
            name: fallback[{"train": "train", "val": "val"}[name]]
            for name in split_names
        }
        selected_by_split["test"] = fallback["test"]
    validate_patient_split(selected_by_split)
    record_by_case = {record.case_id: record for record in records}
    configured_depth_axis = validate_depth_axis(config.get("depth_axis"))
    slice_counts: dict[str, int] = {}
    for name, case_ids in selected_by_split.items():
        total_slices = 0
        for case_id in case_ids:
            record = record_by_case.get(case_id)
            if record is None:
                raise ValueError(f"Split references undiscovered ACDC case {case_id!r}")
            volume, _ = load_array(record.image_path)
            axis = configured_depth_axis
            if axis is None and record.source_format == "nifti":
                axis = 2
            total_slices += int(to_depth_first(volume, depth_axis=axis).shape[0])
        slice_counts[name] = total_slices
    patient_counts = {
        name: len({patient_id_from_case_id(case_id) for case_id in case_ids})
        for name, case_ids in selected_by_split.items()
    }
    return {
        "dataset": dataset_name,
        "validated": True,
        "cases": {name: len(case_ids) for name, case_ids in selected_by_split.items()},
        "patients": patient_counts,
        "slices": slice_counts,
        "test_available": "test" in selected_by_split,
        "records": len(records),
    }


def _state_list(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        if value.ndim == 5:
            return list(value)
        return [value]
    return [item for item in list(value) if torch.is_tensor(item)]


def extract_annotation_states(output: Any, *, include_initial: bool = False) -> list[torch.Tensor]:
    """Extract refinement states, optionally prefixed with the explicit A0 state.

    ``states`` remains the historical refinement-only alias.  Callers that
    need the inference trajectory should use ``include_initial=True`` or
    :func:`extract_annotation_trajectory` so A0 is not accidentally skipped.
    """

    initial: torch.Tensor | None = None
    if isinstance(output, dict):
        initial = extract_initial_logits(output)
        for key in ("states", "refinement_logits", "intermediate_logits"):
            states = output.get(key)
            if states is not None:
                result = _state_list(states)
                if include_initial and initial is not None:
                    return [initial] + [state for state in result if state is not initial]
                return result
        for key in ("logits", "annotation_logits", "output"):
            value = output.get(key)
            if torch.is_tensor(value):
                result = [value]
                if include_initial and initial is not None and initial is not value:
                    return [initial, value]
                return result
    if torch.is_tensor(output):
        return [output]
    raise TypeError("Could not find annotation logits in model output")


def extract_annotation_trajectory(output: Any) -> list[torch.Tensor]:
    """Return ``[A0, A1, ..., A_T]`` without hard-coding the number of turns."""

    if isinstance(output, dict):
        for key in ("state_trace", "all_states", "annotation_trajectory"):
            value = output.get(key)
            if value is not None:
                states = _state_list(value)
                if states:
                    return states
    return extract_annotation_states(output, include_initial=True)


def adjacent_annotation_pairs(states: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build exactly the adjacent transitions represented by a state trace."""

    if len(states) < 2:
        return []
    return list(zip(states[:-1], states[1:]))


def extract_initial_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("initial_logits", "a0_logits", "logits", "annotation_logits", "output"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    if torch.is_tensor(output):
        return output
    raise TypeError("Could not find initial annotation logits in model output")


def encoder_head_optimizer(
    model: nn.Module,
    *,
    lr: float = 3e-4,
    encoder_lr: float = 3e-5,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Create separate parameter groups with a lower LR for pretrained features."""

    encoder_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    encoder = getattr(model, "encoder", None)
    encoder_ids = {id(p) for p in encoder.parameters()} if encoder is not None else set()
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (encoder_params if id(parameter) in encoder_ids else head_params).append(parameter)
    lr = _require_positive_real(lr, "lr")
    encoder_lr = _require_positive_real(encoder_lr, "encoder_lr")
    weight_decay = _require_nonnegative_real(weight_decay, "weight_decay")
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def build_adamw_optimizer(
    parameters: Iterable[nn.Parameter],
    *,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Build a validated AdamW optimizer for a single parameter group."""

    values = list(parameters)
    if not values:
        raise ValueError("AdamW requires at least one trainable parameter")
    if any(not isinstance(parameter, nn.Parameter) for parameter in values):
        raise TypeError("AdamW parameters must be torch.nn.Parameter instances")
    if any(not parameter.requires_grad for parameter in values):
        raise ValueError("AdamW parameter groups must contain only trainable parameters")
    lr = _require_positive_real(lr, "lr")
    weight_decay = _require_nonnegative_real(weight_decay, "weight_decay")
    if len(betas) != 2 or any(not 0.0 <= float(beta) < 1.0 for beta in betas):
        raise ValueError(f"betas must contain two values in [0, 1), got {betas!r}")
    eps = _require_positive_real(eps, "eps")
    return torch.optim.AdamW(values, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


def validate_accumulation_steps(value: Any) -> int:
    return _require_integer(value, "accumulation_steps", minimum=1)


def optimizer_steps_per_epoch(num_batches: int, accumulation_steps: int) -> int:
    num_batches = _require_integer(num_batches, "num_batches", minimum=0)
    accumulation_steps = validate_accumulation_steps(accumulation_steps)
    return int(math.ceil(num_batches / accumulation_steps)) if num_batches else 0


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Schedule optimizer updates with linear warmup followed by cosine decay."""

    total_steps = _require_integer(total_steps, "total_steps", minimum=1)
    warmup_steps = _require_integer(warmup_steps, "warmup_steps", minimum=0)
    if warmup_steps > total_steps:
        raise ValueError("warmup_steps cannot exceed total_steps")
    min_lr_ratio = _require_real(min_lr_ratio, "min_lr_ratio")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(float(step) / float(warmup_steps), 1e-8)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((float(step) - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def build_training_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    *,
    num_batches: int,
    epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Build the configured update-based scheduler for a training loop."""

    if not isinstance(config, Mapping):
        raise ValueError("training scheduler config must be a mapping")
    name = str(config.get("scheduler", "cosine")).strip().lower()
    if name in {"", "none", "constant", "off"}:
        return None
    if name not in {"cosine", "warmup_cosine", "cosine_with_warmup"}:
        raise ValueError(f"Unsupported scheduler {name!r}; use cosine or none")
    epochs = _require_integer(epochs, "epochs", minimum=1)
    accumulation_steps = validate_accumulation_steps(config.get("accumulation_steps", 1))
    updates_per_epoch = optimizer_steps_per_epoch(num_batches, accumulation_steps)
    if updates_per_epoch == 0:
        raise ValueError("Cannot build a scheduler for an empty DataLoader")
    total_steps = updates_per_epoch * epochs
    if "warmup_steps" in config:
        warmup_steps = _require_integer(config["warmup_steps"], "warmup_steps", minimum=0)
    elif "warmup_epochs" in config:
        warmup_epochs = _require_nonnegative_real(config["warmup_epochs"], "warmup_epochs")
        warmup_steps = int(math.ceil(warmup_epochs * updates_per_epoch))
    else:
        warmup_ratio = _require_nonnegative_real(config.get("warmup_ratio", 0.0), "warmup_ratio")
        if warmup_ratio > 1.0:
            raise ValueError("warmup_ratio must be between 0 and 1")
        warmup_steps = int(math.ceil(total_steps * warmup_ratio))
    # Short smoke runs may have fewer updates than the production warmup
    # setting.  Clamp the schedule rather than crashing before the first step.
    warmup_steps = min(warmup_steps, total_steps)
    return build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=float(config.get("min_lr_ratio", 0.0)),
    )


def resolve_amp(config: Mapping[str, Any], device: torch.device) -> tuple[bool, torch.dtype]:
    """Resolve AMP policy, keeping CPU defaults deterministic and safe."""

    if not isinstance(config, Mapping):
        raise ValueError("AMP config must be a mapping")
    raw_enabled = config.get("amp", config.get("mixed_precision", False))
    if isinstance(raw_enabled, str):
        normalized = raw_enabled.strip().lower()
        if normalized == "auto":
            enabled = device.type == "cuda"
        elif normalized in {"true", "1", "yes", "on"}:
            enabled = True
        elif normalized in {"false", "0", "no", "off"}:
            enabled = False
        else:
            raise ValueError(f"Unsupported amp setting: {raw_enabled!r}")
    else:
        enabled = _require_bool(raw_enabled, "amp")
    if device.type not in {"cuda", "cpu"}:
        if enabled:
            raise ValueError(f"AMP is supported only on CUDA/CPU, got device {device}")
        return False, torch.float32
    dtype_name = str(config.get("amp_dtype", "float16" if device.type == "cuda" else "bfloat16")).lower()
    dtype = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported amp_dtype {dtype_name!r}")
    if device.type == "cpu" and enabled and dtype is torch.float16:
        raise ValueError("CPU AMP requires amp_dtype=bfloat16")
    return enabled, dtype


def build_grad_scaler(*, enabled: bool, device: torch.device, dtype: torch.dtype) -> Any:
    """Create a version-compatible scaler; bfloat16 does not need scaling."""

    scale_enabled = bool(enabled and device.type == "cuda" and dtype is torch.float16)
    try:
        return torch.amp.GradScaler("cuda", enabled=scale_enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older PyTorch
        return torch.cuda.amp.GradScaler(enabled=scale_enabled)


def autocast_context(*, enabled: bool, device: torch.device, dtype: torch.dtype) -> Iterator[None]:
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def is_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value.detach()).all().item())
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError):
        return False


def _gradient_norm(model: nn.Module) -> float:
    gradients = [parameter.grad.detach().float() for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        return 0.0
    total = torch.stack([gradient.norm(2).pow(2) for gradient in gradients]).sum().sqrt()
    return float(total)


def finalize_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    pending_batches: int,
    accumulation_steps: int,
    grad_clip: float | None,
) -> tuple[bool, float]:
    """Unscale, normalize a partial accumulation, guard, and step once."""

    pending_batches = _require_integer(pending_batches, "pending_batches", minimum=1)
    accumulation_steps = validate_accumulation_steps(accumulation_steps)
    if pending_batches > accumulation_steps:
        raise ValueError("pending_batches cannot exceed accumulation_steps")
    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer)
    if pending_batches < accumulation_steps:
        correction = float(accumulation_steps) / float(pending_batches)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
    if grad_clip is not None:
        grad_clip = _require_positive_real(grad_clip, "grad_clip")
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
    else:
        norm = _gradient_norm(model)
    gradients_finite = all(
        is_finite(parameter.grad)
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    step_ok = math.isfinite(norm) and gradients_finite
    if step_ok:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
        if scheduler is not None:
            scheduler.step()
    elif scaler is not None and scaler.is_enabled():
        # Record the non-finite gradients so GradScaler reduces its scale; the
        # optimizer itself will skip the update.
        scaler.step(optimizer)
    if scaler is not None:
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return step_ok, norm


def _finite_tree(value: Any, name: str) -> None:
    if torch.is_tensor(value) and not is_finite(value):
        raise FloatingPointError(f"Non-finite tensor in {name}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{name}[{index}]")


def _cpu_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            result[str(key)] = value.detach().cpu().clone()
        elif isinstance(value, Mapping):
            result[str(key)] = _cpu_state_dict(value)
        else:
            result[str(key)] = value
    return result


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return state


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    optimizer_step: int = 0,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a portable, finite, weights-only-loadable checkpoint."""

    path = Path(path)
    epoch = _require_integer(epoch, "epoch", minimum=0)
    global_step = _require_integer(global_step, "global_step", minimum=0)
    optimizer_step = _require_integer(optimizer_step, "optimizer_step", minimum=0)
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    if extra is not None and not isinstance(extra, Mapping):
        raise ValueError("checkpoint extra must be a mapping")
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": _cpu_state_dict(model.state_dict()),
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "rng_state": _rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if config is not None:
        payload["config"] = dict(config)
    if extra is not None:
        reserved = set(payload).intersection(extra)
        if reserved:
            raise ValueError(f"Checkpoint extra uses reserved key(s): {sorted(reserved)}")
        payload.update(dict(extra))
    _finite_tree(payload["model"], "model")
    for key in ("optimizer", "scheduler", "scaler"):
        if key in payload:
            _finite_tree(payload[key], key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _extract_model_state(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping, got {type(payload).__name__}")
    if "model" in payload:
        model_state = payload["model"]
    elif "state_dict" in payload:
        model_state = payload["state_dict"]
    elif payload and all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()):
        model_state = payload
    else:
        raise ValueError("Checkpoint contains no model/state_dict tensor mapping")
    if not isinstance(model_state, Mapping) or not all(isinstance(key, str) for key in model_state):
        raise ValueError("Checkpoint model state must be a string-keyed mapping")
    return dict(model_state), dict(payload)


def _checkpoint_counter(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    return _require_integer(value, f"checkpoint.{key}", minimum=0)


def _model_architecture_signature(model: nn.Module) -> dict[str, int | str | bool]:
    expert = getattr(model, "annotation_expert", None)
    window = getattr(expert, "refinement_block", None)
    encoder = getattr(model, "encoder", None)
    return {
        "num_classes": int(getattr(model, "num_classes", getattr(expert, "num_classes", -1))),
        "shared_channels": int(getattr(getattr(model, "fpn", None), "out_channels", -1)),
        "window_k": int(getattr(window, "k", -1)),
        "encoder_name": str(getattr(encoder, "name", "")),
    }


def _validate_checkpoint_architecture(payload: Mapping[str, Any], model: nn.Module) -> None:
    config = payload.get("config")
    if not isinstance(config, Mapping):
        return
    configured = config.get("model", config)
    if not isinstance(configured, Mapping):
        return
    actual = _model_architecture_signature(model)
    mismatches: list[str] = []
    for key in ("num_classes", "shared_channels", "window_k", "encoder_name"):
        if key in configured and actual[key] not in {-1, ""} and str(configured[key]) != str(actual[key]):
            mismatches.append(f"{key}: checkpoint={configured[key]!r}, model={actual[key]!r}")
    if mismatches:
        raise ValueError(
            "Incompatible checkpoint architecture; refusing to load: "
            + "; ".join(mismatches)
        )


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint with safe deserialization and optional state restore."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ValueError(f"Unable to safely load checkpoint {path}: {exc}") from exc
    model_state, normalized = _extract_model_state(payload)
    _finite_tree(model_state, "model")
    for key in ("epoch", "global_step", "optimizer_step"):
        _checkpoint_counter(normalized, key)
    if model is not None:
        _validate_checkpoint_architecture(normalized, model)
        model.load_state_dict(model_state, strict=strict)
    if optimizer is not None and "optimizer" in normalized:
        optimizer.load_state_dict(normalized["optimizer"])
    if scheduler is not None and "scheduler" in normalized:
        scheduler.load_state_dict(normalized["scheduler"])
    if scaler is not None and "scaler" in normalized:
        scaler.load_state_dict(normalized["scaler"])
    if restore_rng and isinstance(normalized.get("rng_state"), Mapping):
        rng = normalized["rng_state"]
        if torch.is_tensor(rng.get("torch")):
            torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and isinstance(rng.get("cuda"), (list, tuple)):
            torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
    normalized["model"] = model_state
    return normalized


def checkpoint_progress(payload: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return completed epoch, global batch step, and optimizer step."""

    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return (
        _checkpoint_counter(payload, "epoch"),
        _checkpoint_counter(payload, "global_step"),
        _checkpoint_counter(payload, "optimizer_step"),
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
