#!/usr/bin/env python3
"""Profile bounded full-volume FUSE updates in an isolated scratch directory.

Real CUDA execution is explicit and belongs to the coordinator. Track B writes
its PROFILE-only teacher into scratch, measures 20 T and 20 J optimizer steps,
and preserves PROFILE.json before cleaning only that script-owned scratch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _geometry_for(data, pid):
    if hasattr(data, "geometry_for"):
        return data.geometry_for(pid)
    if hasattr(data, "manifest"):
        return data.manifest["geometry"]["by_patient"][pid]
    # Compatibility for the small historical selection fixtures.
    labeled_key = next((key for key in data.labeled_keys() if key[0] == pid), None)
    if labeled_key is not None:
        return data.load(labeled_key, labeled=True).geometry
    key = next(key for key in data.unlabeled_keys() if key[0] == pid)
    return data.load(key, labeled=False).geometry


def _select_profile_frames(data, count: int = 20):
    """Return deterministic L/U selection for CPU inventory tests."""
    from self_audit.candidate_fuse.data import FuseSampler
    allowed_l = set(data.labeled_keys())
    if not allowed_l:
        raise ValueError("real profile requires at least one authorized TRAIN L frame")
    l_rows = []
    for key in sorted(allowed_l):
        shape = tuple(int(v) for v in _geometry_for(data, key[0])["shape"])
        l_rows.append((-(shape[0]*shape[1]*shape[2]), key, shape))
    l_rows.sort(key=lambda row: (row[0], row[1]))
    labeled = [(l_rows[i % len(l_rows)][1], l_rows[i % len(l_rows)][2]) for i in range(count)]
    first_u = {}
    for key in data.unlabeled_keys():
        first_u.setdefault(key[0], key)
    u_rows = []
    for key in first_u.values():
        shape = tuple(int(v) for v in _geometry_for(data, key[0])["shape"])
        u_rows.append((-(shape[0]*shape[1]*shape[2]), key, shape))
    if len(u_rows) < count:
        raise ValueError(f"profile requires at least {count} TRAIN patients with U frames")
    largest_u = min(u_rows, key=lambda row: (row[0], row[1]))[1]
    sampler = FuseSampler(data, seed=42)
    pairs = [sampler.next_pair() for _ in range(count)]
    matches = [i for i, (_lkey, ukey) in enumerate(pairs) if ukey[0] == largest_u[0]]
    if matches:
        i = matches[0]
        pairs[0], pairs[i] = (pairs[0][0], pairs[i][1]), (pairs[i][0], pairs[0][1])
    else:
        pairs[0] = (pairs[0][0], largest_u)
    if any(lkey not in allowed_l for lkey, _ in pairs):
        raise RuntimeError("profile selected a non-L key for a labeled read")
    return labeled, pairs, largest_u


def run_profile(config: Path, out: Path, *, device: str = "cuda", execute: bool = False) -> dict[str, Any]:
    from self_audit.candidate_fuse.config import load_config
    from self_audit.candidate_fuse.data import open_fuse_data
    from self_audit.candidate_fuse.evaluation import evaluate_modes

    raw = json.loads(Path(config).read_text(encoding="utf-8"))
    is_b = raw.get("schema") == "self_audit.candidate_fuse.v3"
    if is_b and raw.get("experiment", {}).get("track") != "B":
        raise ValueError("v3 profile requires Track B")
    cfg = load_config(config, real_run=True)
    if cfg["runtime"]["synthetic"]:
        raise ValueError("synthetic data belongs to the CPU smoke; profile must use real manifests")
    if not execute:
        return {"status": "PLAN_ONLY", "track": "B" if is_b else "A/v1-v2",
                "device": device, "updates": {"T_optimizer_steps": 20, "J_optimizer_steps": 20},
                "output": str(Path(out).resolve()), "production_checkpoint_write": False}
    output = Path(out).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("profile output must be a new empty directory; preserve all existing run files")
    if output.parent == Path(config).resolve().parent:
        raise ValueError("profile output must be separate from the locked config directory")
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8", ":16:8"):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FUSE hardware profile requires an available CUDA device")
    data = open_fuse_data(cfg["data"]["train_manifest"], cfg["data"]["train_manifest_sha256"],
                          role="train", synthetic=False)
    from self_audit.candidate_fuse.engine import joint_loss, _autocast, _frame_tensor, _unwrap_teacher
    from self_audit.candidate_fuse.student import FuseStudent
    from self_audit.candidate_fuse.teacher import (build_teacher, teacher_parameter_groups,
                                                   teacher_loss)
    import numpy as np
    torch.manual_seed(42); np.random.seed(42); random.seed(42); torch.set_num_threads(cfg["runtime"]["num_threads"])
    torch.use_deterministic_algorithms(bool(cfg["runtime"]["deterministic"]), warn_only=True)
    if len(data.labeled_keys()) < 2:
        raise ValueError("FUSE profile requires at least two authorized L frames")
    output.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(tempfile.mkdtemp(prefix="fuse-b-profile-", dir=str(output.parent)))
    try:
        t_metrics, j_metrics, t_times, j_times = _measure_tj(torch, np, cfg, data, target, scratch_root,
                                                               build_teacher, teacher_parameter_groups,
                                                               teacher_loss, FuseStudent, joint_loss,
                                                               _autocast, _frame_tensor, _unwrap_teacher,
                                                               evaluate_modes=evaluate_modes, track_b=is_b)
        receipt = {
            "schema": "self_audit.candidate_fuse.profile.v1", "status": "PROFILE_ONLY",
            "track": "B" if is_b else "A/v1-v2", "evidence_status": "GPU_PROFILE_ONLY",
            "config_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest(),
            "train_manifest_sha256": data.manifest_sha256,
            "device": torch.cuda.get_device_name(target), "dtype": cfg["runtime"]["amp_dtype"],
            "T": {"optimizer_steps": 20, "labeled_frame_exposures": 40 if is_b else 20,
                  "physical_batch_size": 2 if is_b else 1,
                  "metrics": t_metrics, "seconds_per_optimizer_step": t_times},
            "J": {"optimizer_steps": 20, "paired_exposures": 20,
                  "physical_batch_size": 1, "metrics": j_metrics, "seconds_per_optimizer_step": j_times},
            "largest_native_labeled_shape": _largest_shape(data, labeled=True),
            "largest_native_unlabeled_shape": _largest_shape(data, labeled=False),
            "locked_joint_epochs": int(cfg["training"]["joint_epochs"]),
            "locked_joint_optimizer_steps": len(data.patients) * int(cfg["training"]["joint_epochs"]),
            "scratch_checkpoint_policy": "no checkpoint files; in-memory optimizer/model state discarded after receipt",
            "scratch_path": str(scratch_root),
            "projection_basis": "Measured 20 update windows; no ETA recommendation without coordinator review",
        }
        if any((output / marker).exists() for marker in ("RUN_STATE.json", "STATE.json", "resume.pt", "teacher.pt", "joint_best.pt")):
            raise ValueError("profile output must be separate from production campaign files")
        receipt_path = output / "PROFILE.json"
        if receipt_path.exists():
            raise FileExistsError(f"profile receipt already exists: {receipt_path}")
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        receipt["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        receipt_path.with_suffix(".sha256").write_text(receipt["receipt_sha256"] + "\n", encoding="ascii")
        return receipt
    finally:
        # Only this invocation's unique profile scratch is eligible for cleanup.
        shutil.rmtree(scratch_root, ignore_errors=True)


def _measure_tj(torch, np, cfg, data, device, scratch_root, build_teacher,
                teacher_parameter_groups, teacher_loss, FuseStudent, joint_loss,
                autocast, frame_tensor, unwrap_teacher, *, evaluate_modes=None, track_b=True):
    from self_audit.candidate_fuse.data import FuseSampler
    from self_audit.candidate_fuse.data import open_fuse_data
    from self_audit.candidate_fuse.teacher import configure_teacher_step
    val_data = None
    teacher, provenance = build_teacher(cfg, synthetic=False)
    teacher.to(device)
    source = cfg.get("experiment", {}).get("source_teacher") if track_b else None
    if track_b:
        val_data = open_fuse_data(cfg["data"]["val_manifest"], cfg["data"]["val_manifest_sha256"],
                                  role="development", synthetic=False)
    source_path = Path(source["path"]) if source else None
    if track_b:
      if hashlib.sha256(source_path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("B profile source teacher SHA differs from the locked pin")
      meta_path = Path(str(source_path) + ".meta.json")
      if not meta_path.is_file() or hashlib.sha256(meta_path.read_bytes()).hexdigest() != source["metadata_sha256"]:
        raise ValueError("B profile source teacher metadata SHA differs from the locked pin")
      source_payload = torch.load(source_path, map_location=device, weights_only=False)
      state = source_payload.get("model") if isinstance(source_payload, dict) else None
      if not isinstance(state, dict):
        raise ValueError("B source teacher artifact lacks strict model state")
      teacher.load_state_dict(state, strict=True)
      source_meta = json.loads(meta_path.read_text(encoding="utf-8"))
      if source_meta.get("status") != "SELECTED" or source_meta.get("identity") != source["identity"]:
        raise ValueError("B profile source teacher must be SELECTED, not PROFILE")
      for parameter in teacher.parameters(): parameter.requires_grad_(False)
      teacher.eval()
      if evaluate_modes is None:
        from self_audit.candidate_fuse.evaluation import evaluate_modes
      zero_shot_metrics = evaluate_modes(teacher, None, val_data, device=device, modes=("teacher",))
      for parameter in teacher.parameters(): parameter.requires_grad_(True)
    else:
      zero_shot_metrics = None
    teacher.train()
    teacher_opt = torch.optim.AdamW(teacher_parameter_groups(teacher, cfg),
                                    weight_decay=float(cfg["training"]["weight_decay"]))
    sampler = FuseSampler(data, seed=42)
    lkeys = data.labeled_keys()
    largest_l_key = max(lkeys, key=lambda key: _volume(_geometry_for(data, key[0])["shape"]))
    largest_l_patient = largest_l_key[0]
    largest_l_pair = [key for key in lkeys if key[0] == largest_l_patient]
    if len(largest_l_pair) != 2:
        raise ValueError("largest native L patient must have exactly its two authorized ED/ES frames")
    largest_u_key = max(data.unlabeled_keys(),
                        key=lambda key: _volume(_geometry_for(data, key[0])["shape"]))
    # B adaptation uses 2,000 T optimizer steps, same-patient pairs, and T-b LR.
    t_seconds, t_loss_values = [], []
    sampled_l = []
    teacher.train()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(20):
        sampled = sampler.next_labeled_batch(2)
        keys = (largest_l_pair if step == 0 else sampled) if track_b else [largest_l_pair[0] if step == 0 else sampler.next_labeled_batch(1)[0]]
        sampled_l.extend(keys)
        frames = [data.load(key, labeled=True) for key in keys]
        if track_b and (len(frames) != 2 or frames[0].patient_id != frames[1].patient_id):
            raise RuntimeError("B T profile must use same-patient native physical batch2")
        xs, ys, geometry = [], [], None
        for frame in frames:
            image, target_mask, frame_geometry = frame_tensor(frame, device)
            if geometry is None:
                geometry = frame_geometry
            elif any(frame_geometry[k] != geometry[k] for k in ("shape", "spacing_zyx", "affine_zyx", "context")):
                raise ValueError("same-patient T batch native geometry mismatch")
            xs.append(image); ys.append(target_mask)
        image, target = torch.cat(xs), torch.cat(ys)
        configure_teacher_step(teacher, teacher_opt, step, 2000 if track_b else 8000, cfg)
        teacher_opt.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with autocast(device, cfg):
            outputs = teacher(image, geometry, return_features=False)
            if track_b:
                loss = sum(teacher_loss({key: value[i:i+1] if torch.is_tensor(value) and value.shape[:1] == (2,) else value
                                         for key, value in outputs.items()}, target[i:i+1], geometry)
                           for i in range(2)) / 2
            else:
                loss = teacher_loss(outputs, target, geometry)
        loss.backward(); torch.nn.utils.clip_grad_norm_(teacher.parameters(), float(cfg["training"]["gradient_clip_norm"]))
        teacher_opt.step()
        if device.type == "cuda": torch.cuda.synchronize(device)
        t_seconds.append(time.perf_counter()-started); t_loss_values.append(float(loss.detach().float()))
    teacher.freeze_for_joint(); teacher_opt.state.clear(); del teacher_opt
    t_peak = int(torch.cuda.max_memory_allocated(device))
    student = FuseStudent(cfg, teacher_channels=teacher.feature_channels,
                          teacher_phases=teacher.feature_phases).to(device)
    student_opt = torch.optim.AdamW(student.parameters(), lr=float(cfg["training"]["student_lr"]),
                                    weight_decay=float(cfg["training"]["weight_decay"]))
    total_updates = len(data.patients) * int(cfg["training"]["joint_epochs"])
    if track_b and total_updates != 17500:
        raise ValueError("B profile requires the locked 17,500 J update projection")
    if not track_b and total_updates != 8000:
        raise ValueError("v1/v2 profile requires the locked 8,000 J update projection")
    generator = torch.Generator(device=device).manual_seed(42)
    from self_audit.candidate_fuse import engine as engine_module
    engine_module._joint_py_rng = random.Random(42)
    j_seconds, j_loss_values = [], []
    sampled_u = []
    student.train()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(20):
        lkey, ukey = sampler.next_pair()
        if step == 0:
            ukey = largest_u_key
        sampled_u.append(ukey)
        if lkey not in set(lkeys):
            raise RuntimeError("profile J selected an unauthorized L frame")
        labeled, unlabeled = data.load(lkey, labeled=True), data.load(ukey, labeled=False)
        warmup = max(1, int(np.ceil(.05 * total_updates)))
        scale = (step + 1)/warmup if step < warmup else .5*(1+np.cos(np.pi*min(1., (step-warmup)/max(1,total_updates-warmup))))
        for group in student_opt.param_groups: group["lr"] = float(cfg["training"]["student_lr"])*scale
        student_opt.zero_grad(set_to_none=True); started = time.perf_counter()
        with autocast(device, cfg):
            loss, stats = joint_loss(student, teacher, labeled, unlabeled, update=step,
                                     total_updates=total_updates, config=cfg, generator=generator)
        loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg["training"]["gradient_clip_norm"]))
        student_opt.step()
        if device.type == "cuda": torch.cuda.synchronize(device)
        j_seconds.append(time.perf_counter()-started); j_loss_values.append(float(stats["loss"]))
    if largest_l_patient not in {key[0] for key in sampled_l} or largest_u_key not in sampled_u:
        raise RuntimeError("largest native L/U volumes were not included in timed profile updates")
    return ({"mean_loss": float(np.mean(t_loss_values)), "peak_vram_bytes": t_peak,
             "teacher_provenance": provenance, "source_teacher_sha256": source["sha256"] if source else None,
             "largest_l_keys_in_profile": largest_l_pair, "largest_u_key_in_profile": largest_u_key,
             "source_zero_shot_val": zero_shot_metrics["teacher"] if zero_shot_metrics else None},
            {"mean_loss": float(np.mean(j_loss_values)), "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))},
            t_seconds, j_seconds)


def _volume(shape):
    return int(shape[0]) * int(shape[1]) * int(shape[2])


def _largest_shape(data, *, labeled: bool) -> list[int]:
    keys = data.labeled_keys() if labeled else data.unlabeled_keys()
    best = None
    for key in keys:
        shape = [int(v) for v in _geometry_for(data, key[0])["shape"]]
        volume = shape[0] * shape[1] * shape[2]
        if best is None or volume > best[0]:
            best = (volume, shape)
    if best is None:
        raise ValueError("native profile inventory is empty")
    return best[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true", help="run the explicit GPU profile")
    args = parser.parse_args(argv)
    result = run_profile(args.config, args.out, device=args.device, execute=args.execute)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] in {"PLAN_ONLY", "PROFILE_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
