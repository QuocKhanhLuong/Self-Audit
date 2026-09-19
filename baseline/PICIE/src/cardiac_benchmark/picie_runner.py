"""PICIE inference wrapper for cardiac benchmark.

Loads ResNet-18 backbone + FPN from checkpoint, runs forward pass
with classifier, returns anonymous cluster assignment int[H,W].
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PICIEConfig
from .provenance import sha256_array, checkpoint_provenance, environment_identity, rng_contract


def _add_picie_src_to_path() -> None:
    picie_src = str(Path(__file__).resolve().parents[2])
    if picie_src not in sys.path:
        sys.path.insert(0, picie_src)


def seed_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_picie_model(config: PICIEConfig, *, device: str = "cpu") -> tuple[nn.Module, nn.Module]:
    """Load PICIE model and classifier from checkpoint."""
    _add_picie_src_to_path()
    from modules import fpn
    from utils import initialize_classifier, freeze_all

    class Args:
        pass

    args = Args()
    args.arch = config.arch
    args.pretrain = config.pretrain
    args.in_dim = config.in_dim
    args.K_train = config.K_train
    args.device = torch.device(device)

    model = fpn.PanopticFPN(args)
    # The checkpoint was likely saved with DataParallel ("module." prefix)
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Strip "module." if model is not DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    classifier = initialize_classifier(args)
    classifier_sd = checkpoint.get('classifier1_state_dict', None)
    if classifier_sd is not None:
        # initialize_classifier creates a DataParallel model
        # So we keep the module. prefix or just load it directly
        classifier.load_state_dict(classifier_sd)

    classifier = classifier.to(device)
    classifier.eval()

    return model, classifier


def run_inference(
    model: nn.Module,
    classifier: nn.Module,
    image: torch.Tensor,
    *,
    config: PICIEConfig,
    device: str = "cpu",
) -> np.ndarray:
    """Forward pass → anonymous cluster assignment int[H,W]."""

    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        feats = model(image_tensor)
        if config.metric_test == 'cosine':
            feats = F.normalize(feats, dim=1, p=2)

        probs = classifier(feats)
        probs = F.interpolate(probs, (config.resolution, config.resolution),
                              mode='bilinear', align_corners=False)
        preds = probs.topk(1, dim=1)[1].squeeze(1).squeeze(0).cpu().numpy()

    return preds.astype(np.int32)


def run_picie_on_sample(
    model: nn.Module,
    classifier: nn.Module,
    sample: Any,
    *,
    config: PICIEConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run PICIE inference on a single ImageOnlySample."""
    start = time.perf_counter()

    partition = run_inference(
        model, classifier, sample.image, config=config, device=device,
    )

    elapsed = time.perf_counter() - start
    return {
        "sample_id": sample.provenance["sample_id"],
        "patient_id": sample.provenance["patient_id"],
        "split": sample.provenance["split"],
        "partition": partition,
        "partition_shape": list(partition.shape),
        "partition_hash": sha256_array(partition),
        "n_clusters": int(partition.max() + 1) if partition.size > 0 else 0,
        "elapsed_seconds": elapsed,
        "status": "success",
    }


def build_run_metadata(config: PICIEConfig) -> dict[str, Any]:
    """Build metadata for a scientific run."""
    return {
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "checkpoint_provenance": checkpoint_provenance(config.checkpoint_path) if config.checkpoint_path else {},
        "rng_contract": rng_contract(),
        "environment": environment_identity(),
    }
