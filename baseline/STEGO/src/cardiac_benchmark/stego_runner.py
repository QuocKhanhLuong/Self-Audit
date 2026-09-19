"""STEGO inference wrapper for cardiac benchmark.

Loads DINO backbone (frozen) + STEGO heads from checkpoint,
runs forward pass → anonymous cluster assignment int[H,W].
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import STEGOConfig
from .provenance import sha256_array, checkpoint_provenance, environment_identity, rng_contract


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


def load_stego_model(config: STEGOConfig, *, device: str = "cpu") -> Any:
    """Load STEGO model (DINO backbone + projection heads) from checkpoint."""
    _add_stego_src_to_path()
    from types import SimpleNamespace
    from modules import DinoFeaturizer

    cfg = SimpleNamespace(
        model_type=config.model_type,
        dino_patch_size=config.dino_patch_size,
        dino_feat_type=config.dino_feat_type,
        pretrained_weights=config.checkpoint_path if config.checkpoint_path else None,
        projection_type=config.projection_type,
        dropout=config.dropout,
    )
    model = DinoFeaturizer(config.dim, cfg)
    model = model.to(device)
    model.eval()
    return model


def run_inference(
    model: Any,
    image: torch.Tensor,
    *,
    resolution: int = 224,
    device: str = "cpu",
) -> np.ndarray:
    """Forward pass → anonymous cluster assignment int[H,W].

    Args:
        model: DinoFeaturizer instance
        image: float tensor [3, H, W] (already normalized via Fixed Affine)
        resolution: output resolution for the partition
        device: compute device

    Returns:
        int32 array [resolution, resolution] — anonymous partition IDs
    """
    image_tensor = image.unsqueeze(0).to(device)
    if image_tensor.shape[-2] != resolution or image_tensor.shape[-1] != resolution:
        image_tensor = F.interpolate(image_tensor, size=(resolution, resolution), mode="nearest")
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    image_norm = (image_tensor / 255.0 - imagenet_mean) / imagenet_std

    with torch.no_grad():
        _, code = model(image_norm)

    partition = code.argmax(1).squeeze(0)
    partition = F.interpolate(
        partition.float().unsqueeze(0).unsqueeze(0),
        size=(resolution, resolution),
        mode="nearest",
    ).squeeze().long()

    return partition.cpu().numpy().astype(np.int32)


def run_stego_on_sample(
    model: Any,
    sample: Any,
    *,
    config: STEGOConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run STEGO inference on a single ImageOnlySample.

    Returns dict with partition array, provenance, and status.
    """
    start = time.perf_counter()

    partition = run_inference(
        model, sample.image, resolution=config.resolution, device=device,
    )

    elapsed = time.perf_counter() - start
    return {
        "sample_id": sample.provenance["sample_id"],
        "patient_id": sample.provenance["patient_id"],
        "split": sample.provenance["split"],
        "partition": partition,
        "partition_shape": list(partition.shape),
        "partition_hash": sha256_array(partition),
        "n_clusters": int(partition.max() + 1),
        "elapsed_seconds": elapsed,
        "status": "success",
    }


def build_run_metadata(config: STEGOConfig) -> dict[str, Any]:
    """Build metadata for a scientific run."""
    return {
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "checkpoint_provenance": checkpoint_provenance(config.checkpoint_path) if config.checkpoint_path else {},
        "rng_contract": rng_contract(),
        "environment": environment_identity(),
    }
