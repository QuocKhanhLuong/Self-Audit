"""PICIE cardiac benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PICIEConfig:
    profile: str = "PICIE-2D"
    dataset: str = "acdc"
    manifest_path: str = ""
    arch: str = "resnet18"
    in_dim: int = 128
    resolution: int = 224
    K_train: int = 4
    K_test: int = 4
    pretrain: bool = False
    metric_test: str = "cosine"
    checkpoint_path: str = ""
    batch_size: int = 64
    num_workers: int = 4
    benchmark_seed: int = 42
    scientific_run: bool = False

    def validate(self) -> None:
        if self.benchmark_seed != 42:
            raise ValueError("benchmark_seed must be 42")
        if self.resolution != 224:
            raise ValueError("PICIE cardiac resolution must be 224")
        if self.arch != "resnet18":
            raise ValueError("PICIE cardiac requires resnet18")
        if self.pretrain:
            raise ValueError("PICIE cardiac requires pretrain=False")
        if self.K_train != 4 or self.K_test != 4:
            raise ValueError("PICIE cardiac requires K_train=4 and K_test=4")
        if self.scientific_run and not self.checkpoint_path:
            raise ValueError("scientific_run requires checkpoint_path")

    def config_hash(self) -> str:
        from .provenance import sha256_json
        return sha256_json({
            "profile": self.profile,
            "arch": self.arch,
            "in_dim": self.in_dim,
            "resolution": self.resolution,
            "K_train": self.K_train,
            "K_test": self.K_test,
            "pretrain": self.pretrain,
            "metric_test": self.metric_test,
            "benchmark_seed": self.benchmark_seed,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "dataset": self.dataset,
            "arch": self.arch,
            "in_dim": self.in_dim,
            "resolution": self.resolution,
            "K_train": self.K_train,
            "K_test": self.K_test,
            "pretrain": self.pretrain,
            "metric_test": self.metric_test,
            "checkpoint_path": self.checkpoint_path,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "benchmark_seed": self.benchmark_seed,
            "scientific_run": self.scientific_run,
        }
