#!/usr/bin/env python3
"""Profile matched 20-step A legacy/A-aug T(batch2) and J update windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sync(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_bytes(torch, device) -> int | None:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None


def _geometry_for(data, patient_id: str) -> dict[str, Any]:
    getter = getattr(data, "geometry_for", None)
    if callable(getter):
        return getter(patient_id)
    by_patient = data.manifest.get("geometry", {}).get("by_patient", {})
    if patient_id in by_patient:
        return by_patient[patient_id]
    raise ValueError(f"TRAIN manifest lacks native geometry for {patient_id}")


def _largest_keys(data):
    """Pick largest L/U patients from metadata once per patient, without image reads."""
    lkeys, ukeys = data.labeled_keys(), data.unlabeled_keys()
    if not lkeys or not ukeys:
        raise ValueError("profile requires nonempty locked L and image-only U inventories")
    l_by_patient = {}
    for key in lkeys:
        l_by_patient.setdefault(key[0], key)
    u_by_patient = {}
    for key in ukeys:
        u_by_patient.setdefault(key[0], key)
    def volume(item):
        shape = tuple(int(v) for v in _geometry_for(data, item[0])["shape"])
        return math.prod(shape)
    largest_l = max(l_by_patient.values(), key=lambda key: (volume(key), key[0]))
    largest_u = max(u_by_patient.values(), key=lambda key: (volume(key), key[0]))
    pair = [key for key in lkeys if key[0] == largest_l[0]]
    if len(pair) != 2:
        raise ValueError("largest L patient must have exactly two authorized ED/ES frames")
    return largest_l, pair, largest_u


def _install_augmentation_timers(augmentation, timings: dict[str, float]):
    """Measure augmentation call wall time without changing its RNG or outputs."""
    if augmentation is None:
        return
    for name in ("teacher_view", "joint_views"):
        original = getattr(augmentation, name)
        def timed(*args, _fn=original, **kwargs):
            started = time.perf_counter()
            try:
                return _fn(*args, **kwargs)
            finally:
                timings[name] = timings.get(name, 0.0) + time.perf_counter() - started
        setattr(augmentation, name, timed)


def run(config_path: Path, output: Path, *, device: str = "cuda", legacy: bool = False,
        synthetic: bool = False) -> dict:
    """Run 20 T and 20 J updates; synthetic mode exercises the same loop on CPU."""
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8", ":16:8"):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from self_audit.candidate_fuse.config import load_config
    from self_audit.candidate_fuse.data import FuseSampler, open_fuse_data
    from self_audit.candidate_fuse.augmentation import StrongInplaneAugment
    from self_audit.candidate_fuse.engine import (
        _autocast, _teacher_batch_loss, joint_loss, prepare_teacher_batch,
    )
    from self_audit.candidate_fuse.student import FuseStudent
    from self_audit.candidate_fuse.teacher import (
        build_teacher, configure_teacher_step, teacher_parameter_groups,
    )

    cfg = load_config(config_path, real_run=not synthetic)
    if cfg.get("experiment", {}).get("track") != "A" or (not legacy and "augmentation" not in cfg):
        raise ValueError("profile requires A-aug config; --legacy accepts the old A recipe")
    if legacy and "augmentation" in cfg:
        raise ValueError("--legacy requires the original A config without augmentation")
    if bool(cfg["runtime"]["synthetic"]) != bool(synthetic):
        raise ValueError("synthetic profile flag must match the config runtime identity")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profile requested without an available CUDA device")
    if not synthetic and target.type != "cuda":
        raise RuntimeError("real A-aug hardware profile requires CUDA")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; preserve all existing files")

    # Set execution policy before opening the data or constructing models.
    torch.set_num_threads(int(cfg["runtime"]["num_threads"]))
    torch.use_deterministic_algorithms(bool(cfg["runtime"]["deterministic"]), warn_only=target.type == "cuda")
    deterministic = bool(cfg["runtime"]["deterministic"])
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(42)
    output.mkdir(parents=True, exist_ok=True)
    data = open_fuse_data(cfg["data"]["train_manifest"], cfg["data"]["train_manifest_sha256"],
                          role="train", synthetic=synthetic)
    sampler = FuseSampler(data, seed=42)
    augmentation = None if legacy else StrongInplaneAugment(seed=42)
    augmentation_times: dict[str, float] = {}
    _install_augmentation_timers(augmentation, augmentation_times)
    scratch = Path(tempfile.mkdtemp(prefix="fuse-a-aug-profile-", dir=str(output)))
    try:
        teacher, _ = build_teacher(cfg, synthetic=synthetic)
        teacher.to(target)
        teacher_optimizer = torch.optim.AdamW(
            teacher_parameter_groups(teacher, cfg),
            weight_decay=float(cfg["training"]["weight_decay"]))
        largest_l, largest_pair, largest_u = _largest_keys(data)

        # Same exposure positions and same sampled records in --legacy and A-aug.
        t_schedule = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360,
                      400, 800, 1600, 2400, 3200, 4000, 5000, 6000, 7000, 7980]
        t_seconds, t_prep_seconds, t_losses, t_batch_sizes, t_keys = [], [], [], [], []
        teacher_py_rng = random.Random(42)
        teacher.train(True)
        _reset_peak(torch, target)
        for step in range(20):
            _sync(torch, target)
            started = time.perf_counter()
            # The largest native volume must exercise both frozen-encoder T-a
            # and encoder-backprop T-b; sample that patient's pair at each.
            keys = largest_pair if step in (0, 10) else sampler.next_labeled_batch(2)
            if len(keys) != 2 or keys[0][0] != keys[1][0]:
                raise RuntimeError("T profile requires actual same-patient physical batch2 updates")
            frames = [data.load(key, labeled=True) for key in keys]
            t_keys.append([list(key) for key in keys])
            prep_started = time.perf_counter()
            image, labels, geometry = prepare_teacher_batch(
                frames, target, py_rng=teacher_py_rng, augmentation=augmentation)
            t_prep_seconds.append(time.perf_counter() - prep_started)
            if image.shape[0] != 2 or labels.shape[0] != 2:
                raise RuntimeError("prepare_teacher_batch did not return physical batch2")
            teacher_optimizer.zero_grad(set_to_none=True)
            configure_teacher_step(teacher, teacher_optimizer, t_schedule[step], 8000, cfg)
            with _autocast(target, cfg):
                outputs = teacher(image, geometry, return_features=False)
                loss = _teacher_batch_loss(outputs, labels, geometry)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(),
                float(cfg["training"]["gradient_clip_norm"]))
            teacher_optimizer.step()
            _sync(torch, target)
            t_seconds.append(time.perf_counter() - started)
            t_losses.append(float(loss.detach().float()))
            t_batch_sizes.append(int(labels.shape[0]))
        t_peak = _peak_bytes(torch, target)

        teacher.freeze_for_joint()
        teacher_optimizer.state.clear(); del teacher_optimizer
        student = FuseStudent(cfg, teacher_channels=teacher.feature_channels,
                              teacher_phases=teacher.feature_phases).to(target)
        student_optimizer = torch.optim.AdamW(student.parameters(),
            lr=float(cfg["training"]["student_lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]))
        total_updates = len(data.patients) * int(cfg["training"]["joint_epochs"])
        if not synthetic and total_updates != 8000:
            raise ValueError("real A profile requires the locked 8,000 J optimizer updates")
        if total_updates < 1:
            raise ValueError("joint update budget must be positive")
        generator = torch.Generator(device=target).manual_seed(42)
        from self_audit.candidate_fuse import engine as engine_module
        engine_module._joint_py_rng = random.Random(42)
        student.train(True)
        warmup = max(1, int(math.ceil(float(cfg["training"]["warmup_fraction"]) * total_updates)))
        j_seconds, j_losses, j_pairs = [], [], []
        _reset_peak(torch, target)
        for step in range(20):
            _sync(torch, target)
            started = time.perf_counter()
            lkey, ukey = sampler.next_pair()
            if step == 0:
                lkey, ukey = largest_l, largest_u
            j_pairs.append({"labeled": list(lkey), "unlabeled": list(ukey)})
            labeled = data.load(lkey, labeled=True)
            unlabeled = data.load(ukey, labeled=False)
            j_update = step if not synthetic else step % total_updates
            student_optimizer.zero_grad(set_to_none=True)
            if j_update < warmup:
                scale = (j_update + 1) / warmup
            else:
                q = min(1.0, (j_update - warmup) / max(1, total_updates - warmup))
                scale = .5 * (1 + math.cos(math.pi * q))
            for group in student_optimizer.param_groups:
                group["lr"] = float(cfg["training"]["student_lr"]) * scale
            with _autocast(target, cfg):
                loss, stats = joint_loss(student, teacher, labeled, unlabeled,
                    update=j_update, total_updates=total_updates, config=cfg,
                    generator=generator, augmentation=augmentation)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(),
                float(cfg["training"]["gradient_clip_norm"]))
            student_optimizer.step()
            _sync(torch, target)
            j_seconds.append(time.perf_counter() - started)
            j_losses.append(float(stats["loss"]))

        result = {
            "schema": "self_audit.candidate_fuse.profile.v1",
            "status": "PROFILE_ONLY",
            "evidence_status": "SOFTWARE_ONLY" if synthetic else "GPU_PROFILE_ONLY",
            "profile_variant": "legacy_A" if legacy else "A_aug",
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "train_manifest_sha256": data.manifest_sha256,
            "device": str(target) if synthetic else torch.cuda.get_device_name(target),
            "dtype": "float32" if synthetic else cfg["runtime"]["amp_dtype"],
            "seed": 42,
            "T": {"optimizer_steps": 20, "physical_batch_size": 2,
                  "labeled_frame_exposures": sum(t_batch_sizes),
                  "schedule_examples_seen": t_schedule, "T_a_updates": 10, "T_b_updates": 10,
                  "schedule_phases":["T-a" if value < 400 else "T-b" for value in t_schedule],
                  "batch_sizes": t_batch_sizes, "keys": t_keys, "mean_loss": float(np.mean(t_losses)),
                  "seconds_per_update_including_load_prepare_backward_step": float(np.mean(t_seconds)),
                  "augmentation_prepare_seconds_per_update": float(np.mean(t_prep_seconds)),
                  "peak_vram_bytes": t_peak, "largest_native_l_patient": largest_l[0]},
            "J": {"optimizer_steps": 20, "paired_updates": 20,
                  "mean_loss": float(np.mean(j_losses)),
                  "pairs": j_pairs,
                  "seconds_per_update_including_load_prepare_backward_step": float(np.mean(j_seconds)),
                  "augmentation_call_seconds": augmentation_times,
                  "peak_vram_bytes": _peak_bytes(torch, target),
                  "largest_native_l_key": largest_l, "largest_native_u_key": largest_u},
            "timing_scope": "wall time includes per-update data loading, augmentation/preparation, forward, backward, optimizer step, and CUDA sync; excludes setup, VAL, checkpoints, and export",
            "scratch_checkpoint_policy": "no production checkpoint writes; only this invocation scratch removed",
        }
        receipt = output / ("PROFILE_TJ_LEGACY.json" if legacy else "PROFILE_TJ.json")
        if receipt.exists():
            raise FileExistsError(receipt)
        receipt.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return result
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--legacy", action="store_true", help="disable augmentation for matched old A control")
    parser.add_argument("--synthetic", action="store_true", help="run CPU software-only loop on fixture data")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config, args.out, device=args.device,
                         legacy=args.legacy, synthetic=args.synthetic), indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
