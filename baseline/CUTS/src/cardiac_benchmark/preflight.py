"""Bounded synthetic P0 performance checks; never a scientific run."""

from __future__ import annotations

import os
import time
import tracemalloc
from typing import Any

import numpy as np
import torch

from .cluster_kmeans import cluster_latent
from .provenance import sha256_array
from .train_stage1 import Stage1Config, build_model, build_optimization, seed_primary


def _rss_bytes() -> int | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return None


def synthetic_preflight(*, profile: str = "CUTS-2D", target_hw: tuple[int, int] = (32, 32), batches: tuple[int, ...] = (1, 2, 4, 8, 16)) -> dict[str, Any]:
    """Measure bounded synthetic steps and one PHATE/K10 run at the same grid."""
    seed_primary(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Stage1Config(profile=profile, dataset="synthetic", manifest_path="unused", max_epochs=1, batch_size=1)
    rows = []
    channels = 1 if profile == "CUTS-2D" else 3
    for batch in batches:
        model = build_model(config).to(device)
        optimizer, _, recon_loss, contrastive_loss = build_optimization(model, config)
        image = torch.rand(batch, channels, *target_hw, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        _, real, recon, anchor, positive = model(image)
        forward = time.perf_counter()
        reconstruction = recon_loss(real, recon)
        contrastive = contrastive_loss(anchor, positive)
        loss = 0.001 * contrastive + 0.999 * reconstruction
        optimizer.zero_grad(); loss.backward()
        backward = time.perf_counter()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        finished = time.perf_counter()
        rows.append({"batch_size": batch, "forward_seconds": forward - start, "backward_seconds": backward - forward,
                     "optimizer_seconds": finished - backward, "images_per_second": batch / (finished - start),
                     "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                     "loss": float(loss.detach().cpu())})
    model = build_model(config, inference=True).to(device).eval()
    image = torch.rand(1, channels, *target_hw, device=device)
    with torch.no_grad():
        export_start = time.perf_counter(); latent = model(image).detach().cpu().numpy()[0]; export_elapsed = time.perf_counter() - export_start
    before_rss = _rss_bytes()
    clustering = cluster_latent(latent, num_workers=1)
    after_rss = _rss_bytes()
    # Raw arrays are intentionally exported by the partition stage, not embedded
    # in a JSON preflight receipt.
    clustering.pop("raw_cluster_map", None)
    return {"schema_version": "cuts.cardiac.p0.preflight.v1", "fixture_only": True, "profile": profile,
            "target_hw": list(target_hw), "device": str(device), "batches": rows,
            "latent_export_seconds": export_elapsed, "latent_export_images_per_second": 1 / export_elapsed,
            "latent_bytes": int(latent.nbytes), "latent_hash": sha256_array(latent),
            "phate": clustering, "cpu_rss_before": before_rss, "cpu_rss_after": after_rss,
            "cpu_rss_delta": None if before_rss is None or after_rss is None else after_rss - before_rss,
            "artifact_bytes_estimate": int(latent.nbytes + (target_hw[0] * target_hw[1] * np.dtype(np.int64).itemsize)),
            "feasible_batch": max(row["batch_size"] for row in rows)}
