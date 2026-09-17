"""Isolated, label-free reproduction of CUTS PHATE + K=10 clustering."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import phate

from .manifest import require_scientific_manifest
from .provenance import sha256_array, write_json

CLUSTERING_SEED = 1
CLUSTERING_RETRY_SEED = 2


def phate_kmeans(latent_flat: np.ndarray, *, random_seed: int, num_workers: int = 1) -> np.ndarray:
    """Exact numerical calls from helper_generate_kmeans.phate_clustering."""
    operator = phate.PHATE(n_components=3, knn=100, n_landmark=500, t=2,
                           verbose=False, random_state=random_seed, n_jobs=num_workers)
    operator.fit_transform(latent_flat)
    return phate.cluster.kmeans(operator, n_clusters=10, random_state=random_seed)


def cluster_latent(latent: np.ndarray, *, num_workers: int = 1,
                   clustering_fn: Callable[..., np.ndarray] = phate_kmeans) -> dict[str, Any]:
    if latent.ndim != 3:
        raise ValueError("latent must be [L,H,W]")
    channels, height, width = latent.shape
    flat = np.transpose(latent, (1, 2, 0)).reshape(height * width, channels)
    exceptions: list[str] = []
    start = time.perf_counter()
    retry = False
    actual_seed: int | None = None
    clusters: np.ndarray | None = None
    for seed in (CLUSTERING_SEED, CLUSTERING_RETRY_SEED):
        try:
            clusters = clustering_fn(flat, random_seed=seed, num_workers=num_workers)
            actual_seed = seed
            break
        except Exception as exc:  # Official CUTS catches broadly for SVD failures.
            exceptions.append(f"{type(exc).__name__}: {exc}")
            if seed == CLUSTERING_SEED:
                retry = True
    runtime = time.perf_counter() - start
    if clusters is None:
        return {"status": "failure", "clustering_seed": CLUSTERING_SEED,
                "clustering_retry_seed": CLUSTERING_RETRY_SEED, "actual_clustering_seed_used": None,
                "retry_occurred": retry, "exceptions": exceptions, "runtime_seconds": runtime}
    raw = np.asarray(clusters).reshape(height, width).astype(np.int64, copy=False)
    return {"status": "success", "raw_cluster_map": raw, "clustering_seed": CLUSTERING_SEED,
            "clustering_retry_seed": CLUSTERING_RETRY_SEED, "actual_clustering_seed_used": actual_seed,
            "retry_occurred": retry, "exceptions": exceptions, "runtime_seconds": runtime,
            "latent_hash": sha256_array(latent), "raw_partition_hash": sha256_array(raw),
            "phate_version": phate.__version__, "num_workers": num_workers}


def export_raw_partition(latent_metadata: dict[str, Any], output_dir: str | Path, *, num_workers: int = 1,
                         scientific_run: bool = False, manifest_path: str | Path | None = None) -> dict[str, Any]:
    if scientific_run:
        if manifest_path is None:
            raise ValueError("scientific raw clustering requires a frozen manifest path")
        manifest = require_scientific_manifest(manifest_path)
        if latent_metadata["manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError("latent/manifest identity mismatch")
    latent = np.load(latent_metadata["latent_path"], allow_pickle=False)
    result = cluster_latent(latent, num_workers=num_workers)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = str(latent_metadata["sample_id"]).replace(":", "_")
    if result["status"] == "success":
        raw_path = target / f"{stem}.npy"
        np.save(raw_path, result.pop("raw_cluster_map"), allow_pickle=False)
        result["raw_partition_path"] = str(raw_path)
    result.update({"schema_version": "cuts.cardiac.p0.raw-partition.v1", "sample_id": latent_metadata["sample_id"],
                   "dataset": latent_metadata["dataset"], "profile": latent_metadata["profile"],
                   "manifest_hash": latent_metadata["manifest_hash"], "checkpoint_hash": latent_metadata["checkpoint_hash"],
                   "source_cuts_sha": latent_metadata["source_cuts_sha"],
                   "source_freemask_reference_sha": latent_metadata["source_freemask_reference_sha"],
                   "config_hash": latent_metadata["config_hash"], "environment_hash": latent_metadata["environment_hash"]})
    result["image_checksum"] = latent_metadata["provenance"]["image_checksum"]
    result["metadata_hash"] = write_json(target / f"{stem}.json", result)
    return result
