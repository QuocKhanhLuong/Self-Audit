"""Small shared helpers for the three explicit training phases."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


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

    model_cfg = dict(config.get("model", {})) if isinstance(config.get("model"), dict) else {}
    model_cfg.setdefault("num_classes", 4)
    model_cfg.setdefault("image_size", int(config.get("image_size", 256)))
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
        "image_size": int(config.get("image_size", 256)),
        "augment": bool(train and config.get("augment", True)),
    }
    return ACDCDataset(**kwargs)


def extract_annotation_states(output: Any) -> list[torch.Tensor]:
    if isinstance(output, dict):
        for key in ("states", "refinement_logits", "intermediate_logits"):
            states = output.get(key)
            if states is not None:
                if torch.is_tensor(states):
                    return [states]
                return list(states)
        for key in ("logits", "annotation_logits", "output"):
            value = output.get(key)
            if torch.is_tensor(value):
                return [value]
    if torch.is_tensor(output):
        return [output]
    raise TypeError("Could not find annotation logits in model output")


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
