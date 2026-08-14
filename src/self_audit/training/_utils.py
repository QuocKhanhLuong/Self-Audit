"""Small shared helpers for the three explicit training phases."""

from __future__ import annotations

import random
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


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


def validate_image_size(value: Any, *, default: int = 256) -> int:
    """Validate the dataset raster size without making it a model setting."""

    return _require_integer(default if value is None else value, "image_size", minimum=1)


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
    with open(path, encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain a mapping, got {type(value).__name__}")
    return value


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested: str | None) -> torch.device:
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def build_model_from_config(config: dict[str, Any], device: torch.device) -> nn.Module:
    from self_audit.models.self_audit_net import build_self_audit_net

    raw_model_cfg = config.get("model", {})
    if raw_model_cfg is None:
        raw_model_cfg = {}
    model_cfg = filter_model_config(raw_model_cfg)
    model_cfg.setdefault("num_classes", int(config.get("num_classes", 4)))
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
    kwargs = {
        "data_root": data_root,
        "split": split,
        "image_size": validate_image_size(config.get("image_size")),
        "augment": bool(train and config.get("augment", True)),
    }
    return ACDCDataset(**kwargs)


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
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": float(encoder_lr)})
    if head_params:
        groups.append({"params": head_params, "lr": float(lr)})
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.AdamW(groups, lr=float(lr), weight_decay=float(weight_decay))


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
