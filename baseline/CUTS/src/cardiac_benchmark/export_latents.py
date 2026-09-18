"""Label-free frozen-encoder latent export for P0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ImageOnlyCardiacDataset, collate_image_only
from .manifest import load_manifest, require_scientific_manifest
from .provenance import sha256_array, sha256_file, write_json
from .train_stage1 import Stage1Config, load_checkpoint_for_export


def export_latents(config: Stage1Config, checkpoint_path: str | Path, output_dir: str | Path, *, split: str, device: str = "cpu") -> list[dict[str, Any]]:
    manifest = require_scientific_manifest(config.manifest_path, image_root=config.image_root) if config.scientific_run else load_manifest(config.manifest_path, check_paths=True)
    model = load_checkpoint_for_export(config, checkpoint_path, device=device)
    dataset = ImageOnlyCardiacDataset(manifest, split=split, profile=config.profile, target_hw=config.target_hw, source_root=config.image_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_image_only)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    exports: list[dict[str, Any]] = []
    with torch.no_grad():
        for image, provenance in loader:
            latent = model(image.float().to(device)).detach().cpu().numpy()[0].astype(np.float32, copy=False)
            record = provenance[0]
            path = output / f"{record['sample_id'].replace(':', '_')}.npy"
            np.save(path, latent, allow_pickle=False)
            metadata = {
                "schema_version": "cuts.cardiac.p0.latent.v1", "sample_id": record["sample_id"],
                "dataset": record["dataset"], "profile": config.profile, "split": record["split"],
                "manifest_hash": manifest["manifest_hash"], "checkpoint_hash": checkpoint_hash,
                "benchmark_seed": 42, "latent_path": str(path), "latent_hash": sha256_array(latent),
                "source_cuts_sha": checkpoint_payload["source_cuts_sha"],
                "source_freemask_reference_sha": checkpoint_payload["source_freemask_reference_sha"],
                "config_hash": checkpoint_payload["config_hash"], "environment_hash": checkpoint_payload["environment_hash"],
                "shape": list(latent.shape), "provenance": record,
            }
            metadata_path = path.with_suffix(".json")
            metadata["metadata_hash"] = write_json(metadata_path, metadata)
            exports.append(metadata)
    return exports
