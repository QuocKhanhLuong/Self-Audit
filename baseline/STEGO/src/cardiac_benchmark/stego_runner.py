"""STEGO raw-partition inference for the shared cardiac benchmark."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from shared_benchmark.artifacts import GeneratedSample
from shared_benchmark.region_graph import build_region_graph

from .config import STEGOConfig


def _add_stego_src_to_path() -> None:
    stego_src = str(Path(__file__).resolve().parents[1])
    if stego_src not in sys.path:
        sys.path.insert(0, stego_src)


def seed_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_stego_model(config: STEGOConfig, *, device: str | torch.device = "cpu") -> Any:
    """Load the STEGO DINO featurizer on CPU, then move it once to ``device``."""
    if not config.checkpoint_path:
        raise ValueError("STEGO scientific inference requires checkpoint_path")
    _add_stego_src_to_path()
    from types import SimpleNamespace
    from modules import DinoFeaturizer

    cfg = SimpleNamespace(
        model_type=config.model_type,
        dino_patch_size=config.dino_patch_size,
        dino_feat_type=config.dino_feat_type,
        pretrained_weights=config.checkpoint_path,
        projection_type=config.projection_type,
        dropout=config.dropout,
    )
    model = DinoFeaturizer(config.dim, cfg)
    model = model.to(torch.device(device))
    model.eval()
    return model


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


def run_inference(
    model: Any,
    image: torch.Tensor,
    *,
    target_hw: tuple[int, int] | None = None,
    resolution: int | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Forward pass to a 2-D anonymous partition on the shared target grid."""
    if target_hw is None:
        if resolution is None:
            raise ValueError("target_hw or resolution is required")
        target_hw = (int(resolution), int(resolution))
    runtime_device = torch.device(device)
    image_tensor = image.unsqueeze(0).to(runtime_device)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=runtime_device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=runtime_device).view(1, 3, 1, 1)
    image_norm = (image_tensor / 255.0 - imagenet_mean) / imagenet_std
    with torch.no_grad():
        _, code = model(image_norm)
    partition = code.argmax(1, keepdim=True).float()
    partition = F.interpolate(partition, size=target_hw, mode="nearest").squeeze(0).squeeze(0).long()
    return partition.cpu().numpy().astype(np.int32, copy=False)


def generate_sample(
    model: Any,
    sample: Any,
    *,
    config: STEGOConfig,
    device: str | torch.device = "cpu",
    checkpoint_sha256: str,
) -> GeneratedSample:
    target_hw = tuple(int(v) for v in sample.provenance["target_shape"])
    partition = run_inference(model, sample.image, target_hw=target_hw, device=device)
    metadata = {
        "architecture": config.model_type,
        "dino_patch_size": config.dino_patch_size,
        "dino_feat_type": config.dino_feat_type,
        "projection_type": config.projection_type,
        "dim": config.dim,
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
        central_image=np.asarray(sample.image[0].detach().cpu().numpy()),
        baseline_metadata=metadata,
        execution_receipt={"requested_device": str(device)},
    )
