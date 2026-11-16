#!/usr/bin/env python3
"""Bounded real TRAIN-only CUDA probe for PG S1 resource planning.

The probe does not run a campaign or evaluation loop. Each thread-count case
starts with seed 42, a fresh canonical PG student, and the existing CineMA
source/preprocessing/alignment/cosine primitives. It reads no VAL/TEST data.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THREAD_COUNTS = (2, 4, 8)
WARMUPS = 3
MEASURED_STEPS = 5


def choose_largest_frame(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Choose largest native frame by voxel count with stable identity ties."""
    if not records:
        raise ValueError("at least one TRAIN frame is required")
    checked = []
    for record in records:
        shape = tuple(int(v) for v in record["shape"])
        if len(shape) != 3 or any(v <= 0 for v in shape):
            raise ValueError("native shape must be a positive D,H,W triple")
        if record.get("split") != "train":
            raise ValueError("resource probe accepts TRAIN records only")
        checked.append((math.prod(shape), str(record["patient_id"]), int(record["frame_index"]), record))
    return min(checked, key=lambda row: (-row[0], row[1], row[2]))[3]


def benchmark_case(
    step: Callable[[], None], *, device: Any, warmups: int = WARMUPS,
    measured_steps: int = MEASURED_STEPS,
) -> dict[str, Any]:
    """Run bounded updates and report synchronized timings/peak allocation."""
    import torch

    if warmups < 0 or measured_steps <= 0:
        raise ValueError("warmups must be nonnegative and measured_steps positive")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real S1 resource profiling requires an available CUDA device")
    torch.cuda.reset_peak_memory_stats(dev)
    samples = run_bounded_steps(step, synchronize=lambda: torch.cuda.synchronize(dev),
                                warmups=warmups, measured_steps=measured_steps)
    return {
        "warmup_steps": warmups,
        "measured_steps": measured_steps,
        "seconds_per_step": samples,
        "mean_seconds_per_step": sum(samples) / len(samples),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
    }


def run_bounded_steps(
    step: Callable[[], None], *, synchronize: Callable[[], None],
    warmups: int = WARMUPS, measured_steps: int = MEASURED_STEPS,
    clock: Callable[[], float] = time.perf_counter,
) -> list[float]:
    """Run the exact warmup/measurement schedule with an injectable clock."""
    if warmups < 0 or measured_steps <= 0:
        raise ValueError("warmups must be nonnegative and measured_steps positive")
    synchronize()
    for _ in range(warmups):
        step()
    synchronize()
    samples = []
    for _ in range(measured_steps):
        started = clock()
        step()
        synchronize()
        samples.append(clock() - started)
    return samples


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    from self_audit.candidate_pg.config import load_config
    from self_audit.candidate_pg.data import _manifest_ref, _resolve_path
    from self_audit.candidate_pg.engine import _LazyTrainImageSequence
    from self_audit.candidate_pg import cinema_init as cine
    from self_audit.candidate_pg.runtime import configure_runtime

    config = load_config(args.config)
    if config.get("runtime", {}).get("synthetic"):
        raise ValueError("resource profile requires the locked real PG config")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; resource profile was not run")
    runtime_receipt = configure_runtime(device, config["runtime"]["cublas_workspace_config"])
    train_path, train_sha = _manifest_ref(config, "train")
    if not train_path or not train_sha:
        raise ValueError("hash-pinned TRAIN manifest path and SHA are required")

    # Open only the TRAIN manifest. FuseData validates native image identity;
    # all listed frame candidates below are image-only U frames, no masks.
    from self_audit.candidate_fuse.data import FuseData
    train = FuseData(_resolve_path(train_path), train_sha, role="train", synthetic=False)
    keys = train.unlabeled_keys()
    candidates = [
        {"patient_id": pid, "frame_index": frame, "shape": train._images[pid].shape[1:], "split": "train"}
        for pid, frame in keys
    ]
    chosen = choose_largest_frame(candidates)
    # Reuse the campaign's image-only TRAIN sequence so any explicit
    # source-only CineMA crop metadata remains identical to S1. The student
    # still receives the original native image and geometry.
    sequence = _LazyTrainImageSequence(train, train.labeled_keys(), keys)
    unlabeled_index = keys.index((chosen["patient_id"], chosen["frame_index"]))
    item = sequence[len(train.labeled_keys()) + unlabeled_index]
    native = torch.as_tensor(item["image"], dtype=torch.float32).squeeze(0).squeeze(0).contiguous()
    geometry = item["geometry"]

    # Provisioning pins the official cache first. Use the canonical resolver;
    # its allow_download=False branch explicitly rejects real Hub resolution.
    source = cine.resolve_cinema_source(config["cinema_init"])
    cinema_encoder = cine._load_official_encoder(source, device).eval()
    image, _, prepared = cine._prepare_record({"image": native, "geometry": geometry, "split": "train"}, synthetic=False)
    cinema_features = cine._extract_cinema(cinema_encoder, prepared, source.architecture)
    if not any(value is not None for value in cinema_features):
        raise RuntimeError("CineMA exposed neither supported S1 tap")

    cases = []
    for threads in THREAD_COUNTS:
        torch.set_num_threads(threads)
        torch.manual_seed(42)
        np.random.seed(42)
        student = cine._build_student(config, synthetic=False, student_encoder=None, factory=None,
                                      architecture=source.architecture).to(device)
        student.train()
        trainable = cine._freeze_student_decoder(student)
        projectors = torch.nn.ModuleDict()
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config["cinema_init"]["optimizer"]["lr"]),
            weight_decay=float(config["cinema_init"]["optimizer"]["weight_decay"]),
        )
        # Match S1's lazy projector creation and optimizer parameter groups.
        with torch.no_grad():
            s3, s4 = cine._student_forward(student, image[None, None].to(device), geometry)
            for name, value, target, stride, phase in (
                ("conv_1_8", cinema_features[0], s3, (1, 8, 8), cine._student_phase(student, 3, (1, 8, 8))),
                ("tokens_1_16", cinema_features[1], s4, (1, 16, 16), cine._student_phase(student, 4, (1, 16, 16))),
            ):
                if value is None:
                    continue
                grid = cine._feature_to_grid(value, feature_name=name, architecture=source.architecture)
                sampled, _ = cine.backproject_feature(grid.to(device), prepared, target.shape[-3:],
                    source_stride_zyx=stride, target_stride_zyx=phase[0], target_offset_zyx=phase[1])
                projectors[name] = cine._make_projector(target.shape[1], sampled.shape[1], device)
                optimizer.add_param_group({"params": projectors[name].parameters()})

        def step() -> None:
            student.train()
            step_image, step_geometry, step_prepared = cine._prepare_record(item, synthetic=False)
            step_features = cine._extract_cinema(cinema_encoder, step_prepared, source.architecture)
            s3, s4 = cine._student_forward(student, step_image[None, None].to(device), step_geometry)
            losses = []
            for name, value, target, stride, phase in (
                ("conv_1_8", step_features[0], s3, (1, 8, 8), cine._student_phase(student, 3, (1, 8, 8))),
                ("tokens_1_16", step_features[1], s4, (1, 16, 16), cine._student_phase(student, 4, (1, 16, 16))),
            ):
                if value is None:
                    continue
                grid = cine._feature_to_grid(value, feature_name=name, architecture=source.architecture)
                sampled, valid = cine.backproject_feature(grid.to(device), step_prepared, target.shape[-3:],
                    source_stride_zyx=stride, target_stride_zyx=phase[0], target_offset_zyx=phase[1])
                losses.append(cine.cosine_loss_valid(projectors[name](target), sampled.detach(), valid))
            if not losses:
                raise RuntimeError("no valid CineMA S1 cosine terms")
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite S1 resource probe loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        measured = benchmark_case(step, device=device)
        cases.append({"torch_threads": threads, **measured})
        del student, projectors, optimizer
        torch.cuda.empty_cache()

    recommendation = min(cases, key=lambda item: item["mean_seconds_per_step"])
    return {
        "schema": "self_audit.pg_resource_profile.v1",
        "status": "MEASURED",
        "scope": "S1 real TRAIN image-only frame; no evaluation loops",
        "seed": 42,
        "dtype": "float32 (current canonical S1 path)",
        "device": {"name": torch.cuda.get_device_name(device), "cuda": torch.version.cuda,
                   "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
                   "runtime": runtime_receipt},
        "source": {"train_manifest_sha256": train.manifest_sha256,
                   "cinema_config_sha256": source.config_sha256,
                   "cinema_weights_sha256": source.weights_sha256},
        "frame": {"patient_id": chosen["patient_id"], "frame_index": chosen["frame_index"],
                  "shape_zyx": list(chosen["shape"]), "voxels": math.prod(chosen["shape"]),
                  "split": "train", "target_read": False},
        "cases": cases,
        "recommendation": {"torch_threads": recommendation["torch_threads"],
                           "mean_seconds_per_step": recommendation["mean_seconds_per_step"],
                           "peak_allocated_bytes": recommendation["peak_allocated_bytes"]},
        "campaign_note": "The S1 campaign must reseed and initialize a fresh student at seed 42.",
        "not_measured": ["S2", "S3", "arms", "optimized batch size"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/candidate_pg/pg_k8_base.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True, help="JSON result path")
    args = parser.parse_args(argv)
    result = _run(args)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out": str(path.resolve()),
                      "recommended_threads": result["recommendation"]["torch_threads"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
