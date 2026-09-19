"""PiCIE raw-partition inference for the shared cardiac benchmark."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared_benchmark.artifacts import GeneratedSample

from .config import PICIEConfig


def _add_picie_src_to_path() -> None:
    picie_root = str(Path(__file__).resolve().parents[2])
    if picie_root not in sys.path:
        sys.path.insert(0, picie_root)


def seed_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip common DataParallel/module prefixes without importing legacy eval code."""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def build_classifier(config: PICIEConfig) -> nn.Module:
    return nn.Conv2d(config.in_dim, config.K_test, kernel_size=1, stride=1, padding=0, bias=True)


def _checkpoint_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("PiCIE checkpoint must be a mapping")
    return payload


def load_picie_model(config: PICIEConfig, *, device: str | torch.device = "cpu") -> tuple[nn.Module, nn.Module]:
    """Load PiCIE FPN and cluster head on CPU, then move them once to ``device``."""
    if not config.checkpoint_path:
        raise ValueError("PiCIE scientific inference requires checkpoint_path")
    _add_picie_src_to_path()
    from modules import fpn

    args = SimpleNamespace(arch=config.arch, pretrain=config.pretrain, in_dim=config.in_dim)
    model = fpn.PanopticFPN(args)
    classifier = build_classifier(config)
    payload = _checkpoint_payload(config.checkpoint_path)
    model_state = payload.get("model_state_dict") or payload.get("state_dict") or payload.get("model")
    if model_state is None:
        raise ValueError("PiCIE checkpoint missing model state")
    model.load_state_dict(normalize_state_dict_keys(model_state), strict=False)
    classifier_state = payload.get("classifier_state_dict") or payload.get("classifier1_state_dict") or payload.get("cluster_head_state_dict")
    if classifier_state is not None:
        classifier.load_state_dict(normalize_state_dict_keys(classifier_state), strict=False)
    runtime_device = torch.device(device)
    model = model.to(runtime_device).eval()
    classifier = classifier.to(runtime_device).eval()
    return model, classifier


def run_inference(
    model: nn.Module,
    classifier: nn.Module,
    image: torch.Tensor,
    *,
    target_hw: tuple[int, int],
    config: PICIEConfig,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    runtime_device = torch.device(device)
    image_tensor = image.unsqueeze(0).to(runtime_device)
    with torch.no_grad():
        feats = model(image_tensor)
        if config.metric_test == "cosine":
            feats = F.normalize(feats, dim=1, p=2)
        logits = classifier(feats)
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
        pred = logits.argmax(1).squeeze(0)
    return pred.cpu().numpy().astype(np.int32, copy=False)


def generate_sample(
    model: nn.Module,
    classifier: nn.Module,
    sample: Any,
    *,
    config: PICIEConfig,
    device: str | torch.device = "cpu",
    checkpoint_sha256: str,
) -> GeneratedSample:
    target_hw = tuple(int(v) for v in sample.provenance["target_shape"])
    partition = run_inference(model, classifier, sample.image, target_hw=target_hw, config=config, device=device)
    metadata = {
        "architecture": config.arch,
        "in_dim": config.in_dim,
        "K_train": config.K_train,
        "K_test": config.K_test,
        "metric_test": config.metric_test,
        "pretrain": config.pretrain,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_identity": checkpoint_sha256,
        "normalization": sample.provenance["normalization"],
        "profile": sample.provenance["profile"],
        "input_channels": sample.provenance["input_channels"],
    }
    return GeneratedSample(
        partition=partition,
        central_image=np.asarray(sample.image[1].detach().cpu().numpy()),
        baseline_metadata=metadata,
        execution_receipt={"requested_device": str(device)},
    )
