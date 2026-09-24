"""STEGO cardiac benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class STEGOConfig:
    profile: str = "STEGO-2D"
    dataset: str = "acdc"
    manifest_path: str = ""
    model_type: str = "vit_base"
    arch: str = "dino"
    dino_patch_size: int = 8
    dino_feat_type: str = "feat"
    resolution: int = 224
    dim: int = 70
    projection_type: str = "nonlinear"
    n_classes: int = 4
    extra_clusters: int = 0
    checkpoint_path: str = ""
    knn_split: str = "train"
    batch_size: int = 16
    num_workers: int = 0
    benchmark_seed: int = 42
    scientific_run: bool = False
    dropout: bool = False
    continuous: bool = True

    def validate(self) -> None:
        if self.profile not in {"STEGO-2D", "STEGO-SA224", "STEGO-SA224-FAIR"}:
            raise ValueError("unsupported STEGO cardiac profile")
        if self.benchmark_seed != 42:
            raise ValueError("benchmark_seed must be 42")
        if self.resolution != 224:
            raise ValueError("STEGO cardiac resolution must be 224")
        if self.model_type != "vit_base":
            raise ValueError("STEGO cardiac requires vit_base")
        if self.knn_split != "train":
            raise ValueError("KNN must use train split only (no data leakage)")
        if self.scientific_run and not self.checkpoint_path:
            raise ValueError("scientific_run requires checkpoint_path")

    def config_hash(self) -> str:
        from .provenance import sha256_json
        return sha256_json({
            "profile": self.profile,
            "model_type": self.model_type,
            "arch": self.arch,
            "dino_patch_size": self.dino_patch_size,
            "resolution": self.resolution,
            "dim": self.dim,
            "projection_type": self.projection_type,
            "n_classes": self.n_classes,
            "extra_clusters": self.extra_clusters,
            "knn_split": self.knn_split,
            "benchmark_seed": self.benchmark_seed,
            "continuous": self.continuous,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "dataset": self.dataset,
            "model_type": self.model_type,
            "arch": self.arch,
            "dino_patch_size": self.dino_patch_size,
            "resolution": self.resolution,
            "dim": self.dim,
            "projection_type": self.projection_type,
            "n_classes": self.n_classes,
            "extra_clusters": self.extra_clusters,
            "checkpoint_path": self.checkpoint_path,
            "knn_split": self.knn_split,
            "benchmark_seed": self.benchmark_seed,
            "scientific_run": self.scientific_run,
            "continuous": self.continuous,
        }
