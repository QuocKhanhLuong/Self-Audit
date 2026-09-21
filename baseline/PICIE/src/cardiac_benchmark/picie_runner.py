"""PiCIE raw-partition inference for the shared cardiac benchmark."""
from __future__ import annotations

import importlib
import random
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared_benchmark.artifacts import GeneratedSample
from shared_benchmark.region_graph import build_region_graph

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


class _CheckpointCompatStub(nn.Module):
    """Minimal nn.Module placeholder for legacy pickled objects not used by the runner."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - defensive only
        raise RuntimeError("legacy checkpoint stub is not executable")


_ATTRIBUTE_ERROR_RE = re.compile(r"Can't get attribute '([^']+)' on <module '([^']+)'")


def _ensure_module(module_name: str) -> types.ModuleType:
    if module_name in sys.modules:
        module = sys.modules[module_name]
        if isinstance(module, types.ModuleType):
            return module
    if "." in module_name:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = _ensure_module(parent_name)
        module = types.ModuleType(module_name)
        setattr(parent, child_name, module)
    else:
        module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    return module


def _install_stub_attribute(module_name: str, attr_name: str) -> None:
    module = _ensure_module(module_name)
    if not hasattr(module, attr_name):
        setattr(module, attr_name, type(attr_name, (_CheckpointCompatStub,), {}))


def _install_checkpoint_compat_symbols() -> None:
    modules_pkg = importlib.import_module("modules")
    for name in (
        "LambdaLayer",
        "DinoFeaturizer",
        "ResizeAndClassify",
        "ClusterLookup",
        "FeaturePyramidNet",
        "DoubleConv",
        "ContrastiveCorrelationLoss",
        "Decoder",
        "NetWithActivations",
        "ContrastiveCRFLoss",
    ):
        if not hasattr(modules_pkg, name):
            setattr(modules_pkg, name, type(name, (_CheckpointCompatStub,), {}))


def _checkpoint_payload(path: str | Path) -> dict[str, Any]:
    _install_checkpoint_compat_symbols()
    last_error: Exception | None = None
    for _ in range(12):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            break
        except ModuleNotFoundError as exc:
            if not exc.name:
                raise
            _ensure_module(str(exc.name))
            last_error = exc
        except AttributeError as exc:
            match = _ATTRIBUTE_ERROR_RE.search(str(exc))
            if match is None:
                raise
            attr_name, module_name = match.groups()
            _install_stub_attribute(module_name, attr_name)
            last_error = exc
    else:
        raise ValueError(f"PiCIE checkpoint compatibility resolution failed: {last_error}") from last_error
    if not isinstance(payload, dict):
        raise ValueError("PiCIE checkpoint must be a mapping")
    return payload


def _maybe_state_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    state_dict = getattr(value, "state_dict", None)
    if callable(state_dict):
        raw = state_dict()
        if isinstance(raw, Mapping):
            return dict(raw)
    return None


def _compatible_classifier_state(value: Any, classifier: nn.Module) -> dict[str, Any] | None:
    state = _maybe_state_dict(value)
    if state is None:
        return None
    cleaned = normalize_state_dict_keys(state)
    weight = cleaned.get("weight")
    bias = cleaned.get("bias")
    target = classifier.state_dict()
    if weight is None or bias is None:
        return None
    if tuple(weight.shape) != tuple(target["weight"].shape) or tuple(bias.shape) != tuple(target["bias"].shape):
        return None
    return cleaned


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
    model_state = (
        _maybe_state_dict(payload.get("model_state_dict"))
        or _maybe_state_dict(payload.get("state_dict"))
        or _maybe_state_dict(payload.get("model"))
    )
    if model_state is None:
        raise ValueError("PiCIE checkpoint missing model state")
    model.load_state_dict(normalize_state_dict_keys(model_state), strict=False)
    classifier_state = (
        _compatible_classifier_state(payload.get("classifier_state_dict"), classifier)
        or _compatible_classifier_state(payload.get("classifier1_state_dict"), classifier)
        or _compatible_classifier_state(payload.get("cluster_head_state_dict"), classifier)
        or _compatible_classifier_state(payload.get("linear_probe"), classifier)
        or _compatible_classifier_state(payload.get("cluster_probe"), classifier)
    )
    if classifier_state is not None:
        classifier.load_state_dict(classifier_state, strict=False)
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


def topology_summary(partition: np.ndarray) -> dict[str, Any]:
    graph = build_region_graph(partition)
    non_border = [component for component in graph.components if not component.border_contact]
    enclosure_pairs = graph.enclosure_pairs()
    largest = max((component.area_fraction for component in graph.components), default=0.0)
    return {
        "component_count": len(graph.components),
        "border_component_count": sum(component.border_contact for component in graph.components),
        "non_border_component_count": len(non_border),
        "enclosure_pair_count": len(enclosure_pairs),
        "largest_component_fraction": float(largest),
        "graph_digest": graph.digest,
    }


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
        "benchmark_tier": sample.provenance.get("benchmark_tier", "compat"),
        "input_channels": sample.provenance["input_channels"],
        "raw_topology": topology_summary(partition),
    }
    return GeneratedSample(
        partition=partition,
        central_image=np.asarray(sample.image[1].detach().cpu().numpy()),
        baseline_metadata=metadata,
        execution_receipt={"requested_device": str(device)},
    )
