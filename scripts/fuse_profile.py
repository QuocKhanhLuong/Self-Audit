#!/usr/bin/env python3
"""Measure 20 full-volume T updates and 20 paired J updates on CUDA.

The largest TRAIN L and U native volumes are explicitly included. All model
state is held in a temporary scratch directory and discarded after PROFILE.json
has been written; this script never writes a production checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _native_voxels(frame):
    shape = tuple(int(v) for v in frame.geometry["shape"])
    return shape[0] * shape[1] * shape[2]


def _autocast(torch, device, config):
    if device.type == "cuda" and config["runtime"]["amp_dtype"] == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", enabled=False)


def _select_profile_frames(data, count: int = 20):
    """Select largest native L frames and a sampler J stream with largest U."""
    from self_audit.candidate_fuse.data import FuseSampler
    labeled = []
    allowed_l = set(data.labeled_keys())
    if not allowed_l:
        raise ValueError("real profile requires at least one authorized TRAIN L frame")
    for key in sorted(allowed_l):
        frame = data.load(key, labeled=True)
        labeled.append((-_native_voxels(frame), key, tuple(frame.geometry["shape"])))
    labeled.sort(key=lambda row: (row[0], row[1]))
    # Repeat the fixed, authorized L inventory in deterministic ranked order
    # when K8 has fewer distinct frames than the 20-update profile window.
    l_frames = [labeled[i % len(labeled)] for i in range(count)]
    l_frames = [(key, shape) for _, key, shape in l_frames]
    if any(key not in allowed_l for key, _shape in l_frames):
        raise RuntimeError("profile selected a non-L key for a labeled read")

    first_u_by_patient = {}
    all_u_keys = data.unlabeled_keys()
    for key in all_u_keys:
        first_u_by_patient.setdefault(key[0], key)
    u_rows = []
    for key in first_u_by_patient.values():
        frame = data.load(key, labeled=False)
        shape = tuple(frame.geometry["shape"])
        u_rows.append((-_native_voxels(frame), key, shape))
    if len({key[0] for _, key, _ in u_rows}) < count:
        raise ValueError("real profile requires at least 20 TRAIN patients with U frames")
    # Every cine frame for a patient shares the manifest native geometry, so
    # one image-only frame per patient suffices to rank U without redundant reads.
    largest_u = min(u_rows, key=lambda row: (row[0], row[1]))[1]
    sampler = FuseSampler(data, seed=42)
    pairs = [sampler.next_pair() for _ in range(count)]
    patient_matches = [i for i, (_lkey, ukey) in enumerate(pairs) if ukey[0] == largest_u[0]]
    if patient_matches:
        i = patient_matches[0]
        if i != 0:
            pairs[0], pairs[i] = (pairs[0][0], pairs[i][1]), (pairs[i][0], pairs[0][1])
        pairs[0] = (pairs[0][0], largest_u)
    else:
        pairs[0] = (pairs[0][0], largest_u)
    if any(lkey not in allowed_l for lkey, _ukey in pairs):
        raise RuntimeError("profile paired a non-L key for a labeled read")
    return l_frames, pairs, largest_u


def run_profile(config: Path, out: Path, *, device: str = "cuda") -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8", ":16:8"):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    from self_audit.candidate_fuse.config import config_sha256, load_config
    from self_audit.candidate_fuse.data import FuseData
    from self_audit.candidate_fuse.engine import joint_loss
    from self_audit.candidate_fuse.student import FuseStudent
    from self_audit.candidate_fuse.teacher import (
        build_teacher, configure_teacher_step, teacher_loss,
        teacher_parameter_groups,
    )

    cfg = load_config(config, real_run=True)
    if cfg["runtime"]["synthetic"]:
        raise ValueError("FUSE hardware profile requires real native TRAIN manifests")
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FUSE profile requires an available CUDA device")
    if target_device.index is not None and target_device.index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device index is unavailable")
    output = Path(out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    production_markers = ("RUN_STATE.json", "STATE.json", "resume.pt", "teacher.pt", "joint_best.pt")
    if any((output / marker).exists() for marker in production_markers):
        raise ValueError("profile output must be separate from an existing production campaign directory")
    receipt_path = output / "PROFILE.json"
    if receipt_path.exists():
        raise FileExistsError(f"profile receipt already exists: {receipt_path}")

    data = FuseData(cfg["data"]["train_manifest"],
                    expected_sha256=cfg["data"]["train_manifest_sha256"], role="train")
    l_frames, pairs, largest_u_key = _select_profile_frames(data, 20)
    authorized_l_keys = set(data.labeled_keys())
    if any(key not in authorized_l_keys for key, _shape in l_frames) or any(
            lkey not in authorized_l_keys for lkey, _ukey in pairs):
        raise RuntimeError("profile L reads must be restricted to the manifest authorized frame set")
    largest_l = max(l_frames, key=lambda item: item[1][0] * item[1][1] * item[1][2])
    largest_u_frame = data.load(largest_u_key, labeled=False)
    if largest_l[0] not in {item[0] for item in l_frames} or largest_u_key not in {item[1] for item in pairs}:
        raise RuntimeError("largest native L/U frame inclusion invariant failed")

    torch.set_num_threads(cfg["runtime"]["num_threads"])
    deterministic = bool(cfg["runtime"]["deterministic"])
    torch.use_deterministic_algorithms(deterministic, warn_only=(deterministic and target_device.type == "cuda"))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    py_rng = random.Random(cfg["seed"])
    keep_generator = torch.Generator(device=target_device).manual_seed(cfg["seed"])
    teacher, teacher_provenance = build_teacher(cfg, synthetic=False)
    teacher.to(target_device)
    teacher_opt = torch.optim.AdamW(
        teacher_parameter_groups(teacher, cfg),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    student = FuseStudent(cfg, teacher_channels=teacher.feature_channels,
                          teacher_phases=teacher.feature_phases).to(target_device)
    student_opt = torch.optim.AdamW(student.parameters(), lr=float(cfg["training"]["student_lr"]),
                                    weight_decay=float(cfg["training"]["weight_decay"]))
    from self_audit.candidate_fuse import engine as fuse_engine
    fuse_engine._joint_py_rng = random.Random(cfg["seed"])
    t_times: list[float] = []
    j_times: list[float] = []
    t_losses: list[float] = []
    j_losses: list[float] = []

    # Twenty profile-local updates proportionally traverse T-a then T-b,
    # matching the locked 5%/95% schedule without modifying production state.
    teacher.train(True)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    for index, (key, _shape) in enumerate(l_frames):
        _sync(torch, target_device)
        started = time.perf_counter()
        frame = data.load(key, labeled=True)
        image = frame.image.to(target_device, dtype=torch.float32)
        target = frame.target.to(target_device, dtype=torch.long)
        update = index
        configure_teacher_step(teacher, teacher_opt, update, 20, cfg)
        teacher_opt.zero_grad(set_to_none=True)
        with _autocast(torch, target_device, cfg):
            outputs = teacher(image, frame.geometry, return_features=False)
            loss = teacher_loss(outputs, target, frame.geometry)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite T profile loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), float(cfg["training"]["gradient_clip_norm"]))
        teacher_opt.step()
        _sync(torch, target_device)
        t_times.append(time.perf_counter() - started)
        t_losses.append(float(loss.detach().float()))
    t_peak = int(torch.cuda.max_memory_allocated(target_device))
    del frame, image, target, outputs, loss
    for parameter in teacher.parameters():
        parameter.grad = None
    teacher.freeze_for_joint()
    teacher_opt.state.clear()
    del teacher_opt
    gc.collect()
    torch.cuda.empty_cache()

    # J profile uses the sampler's one-patient-per-update stream and includes
    # the largest native U frame; L and U retain independent native geometry.
    student.train(True)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    total_j_updates = len(data.patients) * int(cfg["training"]["joint_epochs"])
    total_t_updates = len(data.labeled_keys()) * int(cfg["training"]["teacher_epochs"])
    if total_t_updates != 8000 or total_j_updates != 8000:
        raise ValueError("real profile projection requires locked 8,000-update T and J budgets")
    warmup = max(1, int(np.ceil(float(cfg["training"]["warmup_fraction"]) * total_j_updates)))
    for update, (lkey, ukey) in enumerate(pairs):
        _sync(torch, target_device)
        started = time.perf_counter()
        labeled = data.load(lkey, labeled=True)
        unlabeled = data.load(ukey, labeled=False)
        for group in student_opt.param_groups:
            if update < warmup:
                scale = (update + 1) / warmup
            else:
                q = min(1.0, (update - warmup) / max(1, total_j_updates - warmup))
                scale = 0.5 * (1.0 + np.cos(np.pi * q))
            group["lr"] = float(cfg["training"]["student_lr"]) * scale
        student_opt.zero_grad(set_to_none=True)
        with _autocast(torch, target_device, cfg):
            loss, stats = joint_loss(student, teacher, labeled, unlabeled, update=update,
                                     total_updates=total_j_updates, config=cfg,
                                     generator=keep_generator)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite J profile loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg["training"]["gradient_clip_norm"]))
        student_opt.step()
        _sync(torch, target_device)
        j_times.append(time.perf_counter() - started)
        j_losses.append(float(stats["loss"]))
    j_peak = int(torch.cuda.max_memory_allocated(target_device))

    if len(t_times) != 20 or len(j_times) != 20:
        raise RuntimeError("profile must complete exactly 20 T and 20 J optimizer updates")
    t_mean, j_mean = float(np.mean(t_times)), float(np.mean(j_times))
    t_projected, j_projected = t_mean * total_t_updates, j_mean * total_j_updates
    receipt = {
        "schema": "self_audit.candidate_fuse.profile.v1",
        "status": "PROFILE_ONLY",
        "config_sha256": config_sha256(cfg),
        "train_manifest_sha256": data.manifest_sha256,
        "device": torch.cuda.get_device_name(target_device),
        "dtype": cfg["runtime"]["amp_dtype"],
        "T": {"optimizer_updates": len(t_times), "labeled_exposures": len(t_times),
              "physical_batch_size": 1,
              "batch_comparability": "profile executes one frame per optimizer step; semi-K8 production teacher uses batch2 where paired frames are available",
              "distinct_labeled_frames": len({key for key, _shape in l_frames}),
              "profiled_L_keys": [list(key) for key, _shape in l_frames],
              "unique_profiled_L_keys": [list(key) for key in sorted({key for key, _shape in l_frames})],
              "updates": len(t_times), "budget_updates": total_t_updates,
              "phase_schedule": "profile-local T-a/T-b proportional schedule",
              "update_indices": list(range(20)),
              "seconds_per_update": t_times, "mean_seconds_per_update": t_mean,
              "peak_vram_bytes": t_peak, "mean_loss": float(np.mean(t_losses)),
              "largest_L_key_included": list(largest_l[0]), "largest_L_shape": list(largest_l[1])},
        "J": {"optimizer_updates": len(j_times), "paired_exposures": len(j_times),
              "physical_batch_size": 1,
              "updates": len(j_times), "budget_updates": total_j_updates,
              "seconds_per_update": j_times, "mean_seconds_per_update": j_mean,
              "peak_vram_bytes": j_peak, "mean_loss": float(np.mean(j_losses)),
              "largest_U_key_included": list(largest_u_key), "largest_U_shape": list(largest_u_frame.geometry["shape"]),
              "largest_U_search_scope": "all permitted image-only TRAIN U frame keys",
              "profiled_U_keys": [list(pair[1]) for pair in pairs]},
        "peak_vram_bytes": max(t_peak, j_peak),
        "projected_full_T_seconds": t_projected,
        "projected_full_J_seconds": j_projected,
        "projected_full_TJ_seconds": t_projected + j_projected,
        "projection_basis": "mean measured full-volume data-load plus optimizer-step time multiplied by locked update count; estimate excludes production checkpoint and VAL overhead",
        "profile_evidence": "measured software/GPU profile on listed native volumes; extrapolated runtime is an estimate, not validated campaign performance",
        "locked_default_joint_epochs": int(cfg["training"]["joint_epochs"]),
        "scratch_checkpoint_policy": "no checkpoint files created; model state discarded after receipt",
        "teacher_provenance": teacher_provenance,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    receipt_path.with_suffix(".sha256").write_text(checksum + "\n", encoding="ascii")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = run_profile(args.config, args.out, device=args.device)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
