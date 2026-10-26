#!/usr/bin/env python3
"""Bounded synchronized synthetic throughput profile for a volume arm.

The profile reports warmup-excluded timing and effective exposure in volumes,
slices and valid voxels per second.  GPU runs are explicitly capped at twenty
minutes; no data or checkpoint path is opened.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:  # direct script execution
    from volume_preflight import (bounded_steps, load_config, load_registry, make_receipt,
                                  synthetic_runtime_config, validate_config, validate_runtime_bounds)
except ImportError:  # package-style import from repository root
    from scripts.volume_preflight import (bounded_steps, load_config, load_registry, make_receipt,
                                          synthetic_runtime_config, validate_config, validate_runtime_bounds)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build(config: Mapping[str, Any], *, synthetic: bool = True):
    try:
        from self_audit.volume.models import build_volume_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("self_audit.volume.models.build_volume_model is unavailable") from exc
    built = build_volume_model(config, synthetic=synthetic)
    if not isinstance(built, tuple) or len(built) != 2:
        raise RuntimeError("build_volume_model must return (module, provenance)")
    return built


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: (value.to(device) if hasattr(value, "to") else value) for key, value in batch.items()}


def run_profile(config_path: Path, *, iterations: int = 10, warmup: int = 2, device_name: str = "cpu", max_minutes: float = 20.0,
                registry_path: Path | None = None, crop_size: int | list[int] | tuple[int, int] = 64,
                microbatch: int | None = None) -> dict[str, Any]:
    if not isinstance(max_minutes, (int, float)) or isinstance(max_minutes, bool) or not math.isfinite(float(max_minutes)) or not 0 < float(max_minutes) <= 20:
        raise ValueError("max_minutes must be finite and in (0,20]")
    config = load_config(config_path)
    validate_config(config)
    registry, entries = load_registry(registry_path)
    config_id = config_path.stem
    entry = entries.get(config_id)
    if entry is None:
        raise ValueError(f"config filename stem {config_id!r} is not in the native-volume registry")
    if bool(entry.get("historical_reference", False)):
        return make_receipt("profile", config_path, config, status="HISTORICAL_REFERENCE_ONLY",
                            historical_reference=True, historical_config_path=entry.get("historical_a0_source"),
                            historical_config_sha256=entry.get("historical_a0_sha256"), coverage="provenance_only",
                            synthetic=False)
    if iterations < 1 or iterations > 50:
        raise ValueError("iterations must be in [1,50]")
    if warmup < 0 or warmup > 20:
        raise ValueError("warmup must be in [0,20]")
    validate_runtime_bounds(config, max_minutes=max_minutes)
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for volume profile") from exc
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable; GPU profile is NOT_RUN")
    device = torch.device(device_name)
    from self_audit.volume.config import validate_volume_config
    from self_audit.volume.data import VolumeTrainingData
    from self_audit.volume.engine import VolumeTrainer
    from self_audit.volume.fixtures import make_volume_fixture
    runtime_config, runtime_overrides = synthetic_runtime_config(
        config, steps=min(8, max(1, iterations + warmup)), crop_size=crop_size, microbatch=microbatch
    )
    validate_runtime_bounds(runtime_config, max_minutes=max_minutes)
    runtime_config = validate_volume_config(runtime_config, allow_full_run=False)
    method = str(runtime_config["training"]["method"])
    if method in {"ema_full", "ema_full_c01"}:
        runtime_overrides["training.pseudo_threshold"] = {
            "from": runtime_config["training"].get("pseudo_threshold"), "to": 0.0,
            "reason": "bounded synthetic U fixture"
        }
        runtime_overrides["training.pseudo_delay"] = {
            "from": runtime_config["training"].get("pseudo_delay"), "to": 0,
            "reason": "bounded synthetic U fixture"
        }
        runtime_config["training"]["pseudo_threshold"] = 0.0
        runtime_config["training"]["pseudo_delay"] = 0
    torch.manual_seed(int(runtime_config["training"].get("seed", 0)))
    torch.set_num_threads(int(runtime_config["training"].get("threads", 1)))
    start = time.perf_counter()
    model, provenance = _build(runtime_config)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("configured builder did not return a torch module")
    model = model.to(device).eval()
    if time.perf_counter() - start > max_minutes * 60.0:
        raise TimeoutError(f"profile allocation exceeded {max_minutes:.2f} minute cap")
    configured_microbatch = int(runtime_config["training"].get("microbatch", 1))
    accumulation = int(runtime_config["training"].get("accumulation", 1))
    b = configured_microbatch * accumulation
    z = int(runtime_config["geometry"]["slab_depth"])
    crop = runtime_config["geometry"]["crop_size"]
    if isinstance(crop, int):
        h = w = crop
    else:
        h, w = (int(x) for x in crop)
    if z > 16 or h > 256 or w > 256:
        raise ValueError("profile shape exceeds bounded synthetic limit (Z<=16,H/W<=256)")
    max_seconds = max_minutes * 60.0
    with tempfile.TemporaryDirectory(prefix="native_volume_profile_") as fixture_dir:
        manifest, manifest_sha, geometry, geometry_sha = make_volume_fixture(fixture_dir, shape_tzyx=(2, max(5, z), 17, 19))
        data = VolumeTrainingData(manifest, manifest_sha, geometry, geometry_sha, synthetic=True)
        rng = np.random.default_rng(int(runtime_config["training"]["seed"]))
        region_manager = None
        if runtime_config["regions"].get("mode") != "none" and float(runtime_config["regions"].get("weight", 0.0)) > 0:
            from self_audit.volume.regions import RegionManager
            region_manager = RegionManager(runtime_config["regions"])
        trainer = VolumeTrainer(model, runtime_config, region_manager=region_manager)

        def make_batches(step: int) -> tuple[dict[str, Any], dict[str, Any]]:
            lb = data.sample_batch(step % len(data.labeled), rng, b, labeled=True, slab_depth=z, crop_size=crop,
                                   spacing_xyz_mm=runtime_config["geometry"]["spacing_xyz_mm"],
                                   fov_xyz_mm=runtime_config["geometry"]["fov_xyz_mm"], flip=bool(runtime_config["geometry"]["flip"]))
            ub = data.sample_batch(step % len(data.unlabeled), rng, b, labeled=False, slab_depth=z, crop_size=crop,
                                   spacing_xyz_mm=runtime_config["geometry"]["spacing_xyz_mm"],
                                   fov_xyz_mm=runtime_config["geometry"]["fov_xyz_mm"], flip=bool(runtime_config["geometry"]["flip"]))
            if runtime_config["temporal"].get("mode") != "none" and float(runtime_config["temporal"].get("weight", 0.0)) > 0:
                pairs = [data.sample_temporal_pair(step % len(data.unlabeled), rng, labeled=False, slab_depth=z, crop_size=crop,
                                                   spacing_xyz_mm=runtime_config["geometry"]["spacing_xyz_mm"],
                                                   fov_xyz_mm=runtime_config["geometry"]["fov_xyz_mm"], flip=bool(runtime_config["geometry"]["flip"])) for _ in range(b)]
                ub["image"] = torch.stack([pair["image"] for pair in pairs])
                ub["adjacent_image"] = torch.stack([pair["adjacent_image"] for pair in pairs])
                ub["geometric_valid"] = torch.stack([pair["geometric_valid"] for pair in pairs])
                ub["adjacent_geometric_valid"] = torch.stack([pair["adjacent_geometric_valid"] for pair in pairs])
                ub["metadata"] = [pair["metadata"] for pair in pairs]
            return _move_batch(lb, device), _move_batch(ub, device)

        for index in range(warmup):
            if time.perf_counter() - start > max_seconds:
                raise TimeoutError(f"profile warmup exceeded {max_minutes:.2f} minute cap")
            lb, ub = make_batches(index)
            extra_batch = None
            if method == "supervised_extra":
                extra_lb, _ = make_batches(index + 1)
                extra_batch = extra_lb
            trainer.transaction(lb, None if method in {"supervised", "supervised_extra"} else ub,
                                iteration=index, extra_labeled_batch=extra_batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            cuda_start = torch.cuda.Event(enable_timing=True); cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        updates_before_timed = (trainer.supervised_updates, trainer.pseudo_updates, trainer.ema_updates,
                                trainer.supervised_extra_updates, trainer.optimizer_updates)
        timed_start = time.perf_counter()
        volume_count = slice_count = valid_voxel_count = 0
        labeled_geometric_voxels = labeled_valid_voxels = 0
        unlabeled_geometric_voxels = unlabeled_valid_voxels = 0
        records: list[dict[str, Any]] = []
        for index in range(iterations):
            if time.perf_counter() - start > max_seconds:
                raise TimeoutError(f"profile exceeded {max_minutes:.2f} minute cap")
            lb, ub = make_batches(warmup + index)
            extra_batch = None
            if method == "supervised_extra":
                extra_lb, _ = make_batches(warmup + index + 1)
                extra_batch = extra_lb
            record = trainer.transaction(lb, None if method in {"supervised", "supervised_extra"} else ub,
                                         iteration=warmup + index, extra_labeled_batch=extra_batch)
            records.append(record)
            volume_count += int(lb["image"].shape[0])
            valid_voxel_count += int(lb["geometric_valid"].sum().item())
            labeled_geometric_voxels += int(lb["geometric_valid"].sum().item())
            labeled_valid_voxels += int(lb["label_valid"].sum().item())
            slice_count += int(lb["image"].shape[0] * lb["image"].shape[2])
            if method not in {"supervised", "supervised_extra"}:
                volume_count += int(ub["image"].shape[0])
                valid_voxel_count += int(ub["geometric_valid"].sum().item())
                unlabeled_geometric_voxels += int(ub["geometric_valid"].sum().item())
                # U label_valid is intentionally all false; retain explicit count.
                unlabeled_valid_voxels += int(ub["label_valid"].sum().item())
                slice_count += int(ub["image"].shape[0] * ub["image"].shape[2])
            if method == "supervised_extra" and extra_batch is not None:
                volume_count += int(extra_batch["image"].shape[0]); valid_voxel_count += int(extra_batch["geometric_valid"].sum().item()); slice_count += int(extra_batch["image"].shape[0] * extra_batch["image"].shape[2])
                labeled_geometric_voxels += int(extra_batch["geometric_valid"].sum().item())
                labeled_valid_voxels += int(extra_batch["label_valid"].sum().item())
        if device.type == "cuda":
            cuda_end.record(); torch.cuda.synchronize(device); cuda_ms = float(cuda_start.elapsed_time(cuda_end))
        else:
            cuda_ms = None
        elapsed = time.perf_counter() - timed_start
        total = time.perf_counter() - start
    if total > max_seconds:
        raise TimeoutError(f"profile exceeded {max_minutes:.2f} minute cap")
    if elapsed <= 0 or not records:
        raise RuntimeError("profile output/timing is invalid")
    samples = float(volume_count)
    valid_voxels = float(valid_voxel_count)
    receipt = make_receipt(
        "profile",
        config_path,
        config,
        iterations=iterations,
        warmup_excluded=warmup,
        elapsed_seconds=elapsed,
        total_seconds=total,
        device=str(device),
        microbatch=configured_microbatch,
        logical_batch_size=b,
        accumulation=accumulation,
        slab_depth=z,
        crop_size=[h, w],
        volumes_per_second=samples / elapsed,
        examples_per_second=samples / elapsed,
        slices_per_second=float(slice_count) / elapsed,
        valid_voxels_per_second=valid_voxels / elapsed,
        model_parameters=sum(p.numel() for p in model.parameters()),
        model_provenance=provenance if isinstance(provenance, Mapping) else str(provenance),
        precision=str(runtime_config["training"].get("precision", "fp32")),
        end_to_end_trainer=True,
        effective_batch_volumes=volume_count / max(1, iterations),
        effective_batch_slices=slice_count / max(1, iterations),
        effective_batch_valid_voxels=valid_voxel_count / max(1, iterations),
        supervised_updates=trainer.supervised_updates,
        supervised_extra_updates=trainer.supervised_extra_updates,
        pseudo_updates=trainer.pseudo_updates,
        ema_updates=trainer.ema_updates,
        measured_supervised_updates=trainer.supervised_updates - updates_before_timed[0],
        measured_pseudo_updates=trainer.pseudo_updates - updates_before_timed[1],
        measured_ema_updates=trainer.ema_updates - updates_before_timed[2],
        measured_extra_updates=trainer.supervised_extra_updates - updates_before_timed[3],
        optimizer_updates=trainer.optimizer_updates,
        measured_optimizer_updates=trainer.optimizer_updates - updates_before_timed[4],
        cuda_elapsed_ms=cuda_ms,
        runtime_overrides=runtime_overrides,
        fixture_manifest_sha256=manifest_sha,
        fixture_geometry_sha256=geometry_sha,
        labeled_label_reads=sorted(data.label_reads),
        exposure={"labeled_geometric_voxels": labeled_geometric_voxels,
                  "labeled_valid_voxels": labeled_valid_voxels,
                  "unlabeled_geometric_voxels": unlabeled_geometric_voxels,
                  "unlabeled_valid_voxels": unlabeled_valid_voxels},
        u_records_mask_free=True,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        vram_total_bytes=None,
        free_vram_bytes=None,
        vram_headroom_bytes=None,
        gpu_profile="NOT_RUN" if device.type != "cuda" else "VERIFIED_SYNTHETIC_ONLY",
    )
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        free_vram, total_vram = (int(value) for value in torch.cuda.mem_get_info(device))
        receipt.update(
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            vram_total_bytes=total_vram,
            free_vram_bytes=free_vram,
            vram_headroom_bytes=free_vram,
        )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-minutes", type=float, default=20.0)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=None,
                        help="explicit synthetic logical microbatch override")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = run_profile(args.config, iterations=args.iterations, warmup=args.warmup,
                              device_name=args.device, max_minutes=args.max_minutes,
                              registry_path=args.registry, crop_size=args.crop_size,
                              microbatch=args.microbatch)
    except Exception as exc:
        receipt = {"schema": "self_audit.volume.tooling_receipt.v1", "command": "profile", "status": "FAIL", "error": str(exc), "real_train_smoke": "NOT_RUN", "gpu": "NOT_RUN"}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
