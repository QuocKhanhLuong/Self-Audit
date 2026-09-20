"""Thin train/dev-only wrapper preserving the official CUTS Stage-1 algebra."""

from __future__ import annotations

import json
import random
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils.extend import ExtendedDataset
from model import CUTSEncoder
from utils.losses import NTXentLoss
from utils.scheduler import LinearWarmupCosineAnnealingLR

from .dataset import ImageOnlyCardiacDataset, collate_image_only
from .manifest import load_manifest, require_scientific_manifest
from shared_benchmark.manifest import validate_scientific_manifest
from .provenance import current_cuts_sha, environment_identity, rng_contract, sha256_file, sha256_json
from shared_benchmark.spatial import SELF_AUDIT_NORMALIZATION_VERSION


CHECKPOINT_SCHEMA_VERSION = "cuts.cardiac.stage1.checkpoint.v2"


@dataclass(frozen=True)
class Stage1Config:
    profile: str
    dataset: str
    manifest_path: str
    target_hw: tuple[int, int] | None = None
    scientific_run: bool = False
    batch_size: int = 16
    num_workers: int = 0
    max_epochs: int = 200
    num_kernels: int = 16
    sampled_patches_per_image: int = 8
    patch_size: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_contrastive_loss: float = 0.001
    benchmark_seed: int = 42
    image_root: str | None = None

    def validate(self) -> None:
        if self.benchmark_seed != 42:
            raise ValueError("primary benchmark_seed must be 42")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.scientific_run and self.max_epochs != 200:
            raise ValueError("scientific Stage-1 requires the frozen 200 epoch recipe")
        if self.lambda_contrastive_loss != 0.001:
            raise ValueError("primary loss mixture must retain lambda=0.001")


def scientific_config_payload(config: Stage1Config) -> dict[str, Any]:
    """Canonical behavior-affecting Stage-1 identity, independent of mount paths."""
    return {
        "profile": config.profile,
        "dataset": config.dataset,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "max_epochs": config.max_epochs,
        "num_kernels": config.num_kernels,
        "sampled_patches_per_image": config.sampled_patches_per_image,
        "patch_size": config.patch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "lambda_contrastive_loss": config.lambda_contrastive_loss,
        "benchmark_seed": config.benchmark_seed,
        "input_normalization": SELF_AUDIT_NORMALIZATION_VERSION,
    }


def scientific_config_hash(config: Stage1Config) -> str:
    return sha256_json(scientific_config_payload(config))


def seed_primary(seed: int = 42) -> None:
    if seed != 42:
        raise ValueError("primary training seed must be 42")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def loader_seed_policy() -> str:
    return "torch.Generator.manual_seed(42); worker_seed=torch.initial_seed()%2**32"


def _capture_rng_state() -> dict[str, Any]:
    """Capture every RNG stream needed to continue at an epoch boundary."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Keep the previous completed-epoch checkpoint if a write is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _checkpoint_path_for_resume(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_last{checkpoint_path.suffix}")


def _validate_resume_payload(
    payload: Mapping[str, Any], config: Stage1Config, manifest: Mapping[str, Any],
) -> None:
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("resume checkpoint schema is not epoch-resumable")
    if payload.get("checkpoint_kind") != "last":
        raise ValueError("resume requires checkpoint_last.pt, not the final export checkpoint")
    if payload.get("dataset", "").lower() != config.dataset.lower() or payload.get("profile") != config.profile:
        raise ValueError("resume checkpoint dataset/profile identity mismatch")
    if payload.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("resume checkpoint manifest identity mismatch")
    source_manifest = manifest.get("source_manifest", {})
    if payload.get("source_manifest_logical_sha") != source_manifest.get("logical_sha256"):
        raise ValueError("resume checkpoint source-manifest identity mismatch")
    if payload.get("shared_grid_hash") != manifest.get("shared_grid_hash"):
        raise ValueError("resume checkpoint shared-grid identity mismatch")
    if payload.get("cuts_mode") != config.profile:
        raise ValueError("resume checkpoint CUTS mode identity mismatch")
    if payload.get("config_hash") != scientific_config_hash(config):
        raise ValueError("resume checkpoint config identity mismatch")
    if "optimizer_state_dict" not in payload or "scheduler_state_dict" not in payload:
        raise ValueError("resume checkpoint lacks optimizer/scheduler state")
    if "rng_state" not in payload or "loader_generator_state" not in payload:
        raise ValueError("resume checkpoint lacks RNG/loader state")


def _checkpoint_payload(
    *, config: Stage1Config, manifest: Mapping[str, Any], manifest_provenance: Mapping[str, Any],
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any,
    epoch: int, train_metrics: Mapping[str, float], dev_metrics: Mapping[str, float],
    train_loader: DataLoader, environment: Mapping[str, Any], source_cuts_sha: str,
    checkpoint_kind: str,
) -> dict[str, Any]:
    loader_generator = getattr(train_loader, "generator", None)
    loader_state = loader_generator.get_state() if loader_generator is not None else None
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "state_dict": model.state_dict(),
        "dataset": config.dataset,
        "profile": config.profile,
        "manifest_hash": manifest["manifest_hash"],
        "config": asdict(config),
        "scientific_config": scientific_config_payload(config),
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "config_hash": scientific_config_hash(config),
        "train_metrics": dict(train_metrics),
        "dev_metrics": dict(dev_metrics),
        "checkpoint_selection_policy": "final_epoch",
        "rng": rng_contract(loader_seed_policy=loader_seed_policy()),
        "rng_state": _capture_rng_state(),
        "loader_generator_state": loader_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "source_cuts_sha": source_cuts_sha,
        "repository_commit_sha": source_cuts_sha,
        **dict(manifest_provenance),
        "environment": dict(environment),
        "environment_hash": sha256_json(environment),
        "input_normalization": SELF_AUDIT_NORMALIZATION_VERSION,
    }
    return payload


def build_loaders(config: Stage1Config) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    config.validate()
    if config.scientific_run:
        manifest = require_scientific_manifest(config.manifest_path, image_root=config.image_root)
    elif config.image_root is not None:
        # The limited timing probe still consumes the scientific v3 manifest,
        # but may use max_epochs=1.  Validate hashes against the explicitly
        # mounted image-only root rather than the host-local receipt path.
        manifest = load_manifest(config.manifest_path, check_paths=False)
        validate_scientific_manifest(manifest, image_root=config.image_root)
    else:
        manifest = load_manifest(config.manifest_path, check_paths=True)
    if manifest["dataset"].lower() != config.dataset.lower():
        raise ValueError("checkpoint/run dataset identity differs from manifest")
    train_ds = ImageOnlyCardiacDataset(manifest, split="train", profile=config.profile, target_hw=config.target_hw, source_root=config.image_root)
    dev_ds = ImageOnlyCardiacDataset(manifest, split="dev", profile=config.profile, target_hw=config.target_hw, source_root=config.image_root)
    # Preserve CUTS's five-batch extension for training only.
    train_source = ExtendedDataset(train_ds, max(len(train_ds), config.batch_size * 5))
    generator = torch.Generator().manual_seed(config.benchmark_seed)
    common = {"batch_size": config.batch_size, "num_workers": config.num_workers, "collate_fn": collate_image_only,
              "worker_init_fn": seed_worker, "persistent_workers": config.num_workers > 0}
    train_loader = DataLoader(train_source, shuffle=True, generator=generator, **common)
    dev_loader = DataLoader(dev_ds, shuffle=False, **common)
    return train_loader, dev_loader, manifest


def build_model(config: Stage1Config, *, inference: bool = False) -> CUTSEncoder:
    channels = 1 if config.profile == "CUTS-2D" else 3
    return CUTSEncoder(in_channels=channels, num_kernels=config.num_kernels, random_seed=42,
                       sampled_patches_per_image=config.sampled_patches_per_image,
                       patch_size=config.patch_size, inference=inference)


def build_optimization(model: CUTSEncoder, config: Stage1Config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=10,
                                                warmup_start_lr=1e-3 * config.learning_rate,
                                                max_epochs=config.max_epochs)
    return optimizer, scheduler, torch.nn.MSELoss(), NTXentLoss()


def _losses(model: CUTSEncoder, image: torch.Tensor, recon_loss, contrastive_loss, config: Stage1Config):
    _, patch_real, patch_recon, z_anchor, z_positive = model(image)
    reconstruction = recon_loss(patch_real, patch_recon)
    contrastive = contrastive_loss(z_anchor, z_positive)
    total = config.lambda_contrastive_loss * contrastive + (1 - config.lambda_contrastive_loss) * reconstruction
    return reconstruction, contrastive, total


def train_epoch(model: CUTSEncoder, loader: DataLoader, optimizer, recon_loss, contrastive_loss, config: Stage1Config, device: torch.device) -> dict[str, float]:
    model.train()
    totals = {"reconstruction": 0.0, "contrastive": 0.0, "total": 0.0, "samples": 0}
    for image, _ in loader:
        image = image.float().to(device)
        reconstruction, contrastive, total = _losses(model, image, recon_loss, contrastive_loss, config)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        batch = image.shape[0]
        totals["reconstruction"] += reconstruction.item() * batch
        totals["contrastive"] += contrastive.item() * batch
        totals["total"] += total.item() * batch
        totals["samples"] += batch
    return {key: value / totals["samples"] for key, value in totals.items() if key != "samples"}


def validate_epoch(model: CUTSEncoder, loader: DataLoader, recon_loss, contrastive_loss, config: Stage1Config, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"reconstruction": 0.0, "contrastive": 0.0, "total": 0.0, "samples": 0}
    with torch.no_grad():
        for image, _ in loader:
            image = image.float().to(device)
            reconstruction, contrastive, total = _losses(model, image, recon_loss, contrastive_loss, config)
            batch = image.shape[0]
            totals["reconstruction"] += reconstruction.item() * batch
            totals["contrastive"] += contrastive.item() * batch
            totals["total"] += total.item() * batch
            totals["samples"] += batch
    return {key: value / totals["samples"] for key, value in totals.items() if key != "samples"}


def train_stage1(
    config: Stage1Config, checkpoint_path: str | Path, *, device: str | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train train patients with epoch-boundary resume and a fixed final epoch."""
    seed_primary(config.benchmark_seed)
    train_loader, dev_loader, manifest = build_loaders(config)
    source_manifest = manifest.get("source_manifest")
    if not isinstance(source_manifest, dict) or not source_manifest.get("logical_sha256"):
        raise ValueError("shared manifest lacks source_manifest.logical_sha256")
    if not manifest.get("shared_grid_hash"):
        raise ValueError("shared manifest lacks shared_grid_hash")
    split_provenance = source_manifest.get("split_provenance", {})
    manifest_provenance = {
        "source_manifest_logical_sha": str(source_manifest["logical_sha256"]),
        "shared_grid_hash": str(manifest["shared_grid_hash"]),
        "split_identity": {
            "split_seed": int(manifest["split_seed"]),
            "split_policy_version": str(manifest["split_policy_version"]),
            "source_split_identity": split_provenance.get("split_identity"),
            "training_split": "train",
            "validation_split": "dev",
        },
        "cuts_mode": config.profile,
        "benchmark_seed": int(config.benchmark_seed),
        "input_normalization": SELF_AUDIT_NORMALIZATION_VERSION,
    }
    runtime_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(runtime_device)
    optimizer, scheduler, recon_loss, contrastive_loss = build_optimization(model, config)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = _checkpoint_path_for_resume(checkpoint_path)
    environment = environment_identity()
    source_cuts_sha = current_cuts_sha()
    start_epoch = 0
    resumed_from: str | None = None
    train_metrics: dict[str, float] = {}
    dev_metrics: dict[str, float] = {}

    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        payload = torch.load(resume_path, map_location=runtime_device, weights_only=False)
        _validate_resume_payload(payload, config, manifest)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        _restore_rng_state(payload["rng_state"])
        loader_generator = getattr(train_loader, "generator", None)
        if loader_generator is None:
            raise ValueError("train loader has no generator for deterministic resume")
        loader_generator.set_state(payload["loader_generator_state"])
        start_epoch = int(payload["next_epoch"])
        if start_epoch > config.max_epochs:
            raise ValueError("resume checkpoint is beyond the configured max_epochs")
        train_metrics = {str(key): float(value) for key, value in payload.get("train_metrics", {}).items()}
        dev_metrics = {str(key): float(value) for key, value in payload.get("dev_metrics", {}).items()}
        resumed_from = str(resume_path)

    result: dict[str, Any] = {
        "checkpoint_selection_policy": "final_epoch",
        "resumed_from": resumed_from,
        "last_checkpoint": str(last_checkpoint_path),
    }
    for epoch in range(start_epoch, config.max_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, recon_loss, contrastive_loss, config, runtime_device)
        scheduler.step()
        dev_metrics = validate_epoch(model, dev_loader, recon_loss, contrastive_loss, config, runtime_device)
        result.update({"epoch": epoch, "train_metrics": train_metrics, "dev_metrics": dev_metrics})

        last_payload = _checkpoint_payload(
            config=config, manifest=manifest, manifest_provenance=manifest_provenance,
            model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
            train_metrics=train_metrics, dev_metrics=dev_metrics, train_loader=train_loader,
            environment=environment, source_cuts_sha=source_cuts_sha, checkpoint_kind="last",
        )
        _atomic_torch_save(last_payload, last_checkpoint_path)
        print(json.dumps({
            "event": "epoch_complete",
            "epoch": epoch + 1,
            "total_epochs": config.max_epochs,
            "checkpoint": str(last_checkpoint_path),
            "train_total": train_metrics.get("total"),
            "dev_total": dev_metrics.get("total"),
        }, sort_keys=True), flush=True)

    if not train_metrics or not dev_metrics:
        raise ValueError("training produced no completed epoch; no final checkpoint can be written")
    final_payload = _checkpoint_payload(
        config=config, manifest=manifest, manifest_provenance=manifest_provenance,
        model=model, optimizer=optimizer, scheduler=scheduler, epoch=config.max_epochs - 1,
        train_metrics=train_metrics, dev_metrics=dev_metrics, train_loader=train_loader,
        environment=environment, source_cuts_sha=source_cuts_sha, checkpoint_kind="final",
    )
    _atomic_torch_save(final_payload, checkpoint_path)
    result.update({"epoch": config.max_epochs - 1, "train_metrics": train_metrics, "dev_metrics": dev_metrics})
    result.update({"checkpoint": str(checkpoint_path), "checkpoint_hash": sha256_file(checkpoint_path),
                   "manifest_hash": manifest["manifest_hash"], "source_manifest_logical_sha": manifest_provenance["source_manifest_logical_sha"],
                   "shared_grid_hash": manifest_provenance["shared_grid_hash"], "cuts_mode": config.profile,
                   "checkpoint_selection_policy": "final_epoch",
                   "device": str(runtime_device), "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION})
    return result


def load_checkpoint_for_export(config: Stage1Config, checkpoint_path: str | Path, *, device: str = "cpu") -> CUTSEncoder:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload["dataset"].lower() != config.dataset.lower() or payload["profile"] != config.profile:
        raise ValueError("dataset/profile checkpoint identity mismatch")
    manifest = require_scientific_manifest(config.manifest_path, image_root=config.image_root) if config.scientific_run else load_manifest(config.manifest_path, check_paths=True)
    if payload["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("checkpoint manifest identity mismatch")
    source_manifest = manifest.get("source_manifest", {})
    if payload.get("source_manifest_logical_sha") != source_manifest.get("logical_sha256"):
        raise ValueError("checkpoint source-manifest identity mismatch")
    if payload.get("shared_grid_hash") != manifest.get("shared_grid_hash"):
        raise ValueError("checkpoint shared-grid identity mismatch")
    if payload.get("cuts_mode") != config.profile:
        raise ValueError("checkpoint CUTS mode identity mismatch")
    if config.scientific_run:
        if payload.get("checkpoint_selection_policy") != "final_epoch":
            raise ValueError("scientific CUTS checkpoint must use final_epoch selection")
        if payload.get("checkpoint_kind", "final") != "final":
            raise ValueError("scientific CUTS export requires the completed final checkpoint")
        if payload.get("config_hash") != scientific_config_hash(config):
            raise ValueError("scientific CUTS checkpoint config identity mismatch")
        if payload.get("input_normalization") != SELF_AUDIT_NORMALIZATION_VERSION:
            raise ValueError("scientific CUTS checkpoint normalization identity mismatch")
    model = build_model(config, inference=True).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
