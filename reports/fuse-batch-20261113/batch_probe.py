#!/usr/bin/env python3
"""Capacity/throughput probe for FUSE T-b and full-fusion J at batch 1 vs 2.

Local diagnostic only. Uses the real hash-pinned TRAIN manifest, original native
arrays/geometry, writes no checkpoints, and makes no accuracy/campaign claim.
"""
from __future__ import annotations
import argparse, gc, json, os, random, statistics, sys, time, traceback
from pathlib import Path


def clean(torch):
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def sync(torch, device): torch.cuda.synchronize(device)


def amp(torch, device, cfg):
    if device.type == "cuda" and cfg["runtime"]["amp_dtype"] == "bfloat16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", enabled=False)


def batched_frames(data, keys, labeled):
    frames = [data.load(key, labeled=labeled) for key in keys]
    import torch
    geometry = frames[0].geometry
    if any(f.patient_id != frames[0].patient_id or f.geometry != geometry for f in frames):
        raise ValueError("batch must contain same-patient frames with identical native geometry")
    image = torch.cat([f.image for f in frames], dim=0)
    target = torch.cat([f.target for f in frames], dim=0) if labeled else None
    return image, target, geometry


def per_frame_supervised(supervised_loss, logits, target):
    import torch
    return torch.stack([supervised_loss(logits[i:i+1], target[i:i+1])
                        for i in range(logits.shape[0])]).mean()


def run_case(torch, cfg, data, device, lkeys, ukeys, batch_size, stage):
    """One case per scope: all tensors, graphs, models, and optimizer refs die here."""
    from self_audit.candidate_fuse.student import FuseStudent
    from self_audit.candidate_fuse.teacher import build_teacher, configure_teacher_step, teacher_loss, teacher_parameter_groups
    from self_audit.candidate_fuse.engine import _soft_ce, _lambda_u
    from self_audit.losses import supervised_loss
    teacher = student = optimizer = None
    case = {"batch_size":batch_size, "stage":stage, "status":"PENDING",
            "frame_keys_labeled":[list(k) for k in lkeys[:batch_size]],
            "frame_keys_unlabeled":[list(k) for k in ukeys[:batch_size]] if stage.startswith("J") else [],
            "warmup_updates":2, "timed_updates":5}
    try:
        teacher, provenance = build_teacher(cfg, synthetic=False)
        teacher.to(device)
        if stage == "T-b":
            groups = teacher_parameter_groups(teacher, cfg)
            for group in groups:
                group["lr"] = cfg["training"]["teacher_lr_encoder"] if group["name"] == "encoder" else cfg["training"]["teacher_lr_head"]
            optimizer = torch.optim.AdamW(groups, weight_decay=cfg["training"]["weight_decay"])
            teacher.train(True); teacher.model.encoder.train(True)
        else:
            student = FuseStudent(cfg, teacher_channels=teacher.feature_channels,
                                  teacher_phases=teacher.feature_phases).to(device)
            teacher.freeze_for_joint(); student.train(True)
            optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["training"]["student_lr"],
                                          weight_decay=cfg["training"]["weight_decay"])
        image_l, target_l, geom_l = batched_frames(data, lkeys[:batch_size], True)
        image_l, target_l = image_l.to(device), target_l.to(device)
        if stage == "J-full-fusion":
            image_u, _, geom_u = batched_frames(data, ukeys[:batch_size], False)
            image_u = image_u.to(device)
            keep_l = torch.ones((batch_size,1,1,1,1), device=device)
            keep_u = torch.ones_like(keep_l)
        times = []
        for step in range(7):
            if step == 2: torch.cuda.reset_peak_memory_stats(device)
            sync(torch, device); started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            update = 400 if stage == "T-b" else 4000
            if stage == "T-b": configure_teacher_step(teacher, optimizer, update, 8000, cfg)
            with amp(torch, device, cfg):
                if stage == "T-b":
                    outputs = teacher(image_l, geom_l, return_features=False)
                    # Slice batch-shaped tensor outputs only; retain any scalar/metadata fields.
                    per = []
                    for i in range(batch_size):
                        one = {k:(v[i:i+1] if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == batch_size else v)
                               for k,v in outputs.items()}
                        per.append(teacher_loss(one, target_l[i:i+1], geom_l))
                    loss = torch.stack(per).mean()
                else:
                    with torch.no_grad():
                        out_l_t = teacher(image_l, geom_l, return_features=True)
                        out_u_t = teacher(image_u, geom_u, return_features=True)
                    out_l = student(image_l, geom_l, teacher_features=out_l_t["features"], keep_mask=keep_l)
                    out_u = student(image_u, geom_u, teacher_features=out_u_t["features"], keep_mask=keep_u)
                    sup = per_frame_supervised(supervised_loss, out_l["logits"], target_l)
                    soft = _soft_ce(out_u["logits"], out_u_t["region_logits"])
                    mimic = out_l["mimic_loss"] + out_u["mimic_loss"]
                    loss = sup + _lambda_u(update, 8000, cfg["training"]["unlabeled_ramp_fraction"])*soft + cfg["training"]["mimic_weight"]*mimic
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite benchmark loss")
            loss.backward()
            params = teacher.parameters() if stage == "T-b" else student.parameters()
            torch.nn.utils.clip_grad_norm_(params, float(cfg["training"]["gradient_clip_norm"]))
            sync(torch, device); optimizer.step(); sync(torch, device)
            if step >= 2: times.append(time.perf_counter()-started)
        seconds = statistics.mean(times)
        rate_key = "seconds_per_frame" if stage == "T-b" else "seconds_per_pair"
        case.update(status="PASS", seconds_per_update=seconds, **{rate_key:seconds/batch_size},
                    mean_updates_per_second=1.0/seconds,
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                    peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                    teacher_provenance=provenance)
    except torch.cuda.OutOfMemoryError as exc:
        case.update(status="FAILED_OOM", error=str(exc),
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                    peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                    memory_summary=torch.cuda.memory_summary(device=device, abbreviated=True))
    except Exception as exc:
        case.update(status="FAILED", error=repr(exc), traceback=traceback.format_exc())
    finally:
        if optimizer is not None: optimizer.state.clear()
        del optimizer, student, teacher
        clean(torch)
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path.cwd()), help="FUSE repository root (default: current directory)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    if not (repo/"self_audit/candidate_fuse").is_dir(): raise ValueError(f"--repo is not a FUSE checkout: {repo}")
    sys.path.insert(0, str(repo))
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8", ":16:8"):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from self_audit.candidate_fuse.config import load_config
    from self_audit.candidate_fuse.data import FuseData
    config_path = Path(args.config); config_path = config_path if config_path.is_absolute() else repo/config_path
    cfg = load_config(config_path, real_run=True)
    if cfg["runtime"]["synthetic"]: raise ValueError("probe accepts only real FUSE configs")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; no benchmark run")
    device = torch.device("cuda")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    manifest_path = Path(cfg["data"]["train_manifest"])
    if not manifest_path.is_absolute(): manifest_path = repo/manifest_path
    data = FuseData(manifest_path, expected_sha256=cfg["data"]["train_manifest_sha256"], role="train")
    patient_shapes = {}
    for pid in data.patients:
        key = next(k for k in data.labeled_keys() if k[0] == pid)
        shape = tuple(int(v) for v in data.load(key, labeled=True).geometry["shape"])
        patient_shapes[pid] = (shape[0]*shape[1]*shape[2], shape)
    patient = max(patient_shapes, key=lambda p:(patient_shapes[p][0],p))
    lkeys = sorted(k for k in data.labeled_keys() if k[0] == patient)
    ukeys = sorted(k for k in data.unlabeled_keys() if k[0] == patient)
    if len(lkeys) != 2 or len(ukeys) < 2: raise ValueError("largest TRAIN patient needs ED/ES and >=2 U frames")
    geometry = data.load(lkeys[0], labeled=True).geometry
    native_shape = tuple(geometry["shape"])
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    receipt = {"schema":"self_audit.candidate_fuse.batch_probe.v1", "status":"PROBE_ONLY",
               "purpose":"same-patient capacity/throughput; no accuracy claim", "repo":str(repo),
               "patient_id":patient, "native_shape_DHW":native_shape, "native_voxels":patient_shapes[patient][0],
               "geometry":geometry, "cuda_device":torch.cuda.get_device_name(device),
               "dtype":cfg["runtime"]["amp_dtype"], "cases":[]}
    for batch_size in (1,2):
        for stage in ("T-b","J-full-fusion"):
            print(f"Starting B{batch_size} {stage} for {patient} ({native_shape})", flush=True)
            case = run_case(torch,cfg,data,device,lkeys,ukeys,batch_size,stage)
            receipt["cases"].append(case)
            # run_case has returned, so its tensors/graphs/param groups are now
            # out of scope; collect and release cached blocks before next model.
            clean(torch)
            print(f"Finished B{batch_size} {stage}: {case['status']}", flush=True)
    path=out/"BATCH_PROBE.json"
    path.write_text(json.dumps(receipt,indent=2,sort_keys=True,default=str)+"\n")
    print(path, flush=True)

if __name__ == "__main__": main()
