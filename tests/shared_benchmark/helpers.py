from __future__ import annotations

from pathlib import Path

import numpy as np

from self_audit_maskfree.data.discovery import discover_dataset
from shared_benchmark.manifest import build_shared_manifest
from shared_benchmark.spatial import build_grid_spec


def write_image(root: Path, relative: str, *, shape: tuple[int, ...] = (7, 11, 3), seed: int = 0) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.save(path, rng.normal(size=shape).astype(np.float32), allow_pickle=False)
    return path


def discovered_projection(root: Path, *, dataset: str = "acdc", target_hw: tuple[int, int] = (8, 8)) -> tuple[dict, dict]:
    upstream = discover_dataset(root, dataset, seed=42, protocol="auto", depth_axis=2)
    projection = build_shared_manifest(
        upstream,
        build_grid_spec(target_hw, config_provenance={"source": "shared-test"}),
        fixture=True,
        scientific=False,
        local_source_root=root,
    )
    return upstream, projection
