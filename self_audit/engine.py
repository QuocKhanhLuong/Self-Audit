"""One coherent pipeline: labeled reference -> pseudo transaction -> EMA once."""
from pathlib import Path
from contextlib import nullcontext
import copy
import hashlib
import json
import os
import random
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import SCHEMA
from .model import build_model, digest
from .data import TrainingData, write_json
from .losses import supervised_loss, strong_intensity, aligned_flip, validate_supervised_loss_config
from .lr_schedule import (validate_lr_scheduler_config, validate_scheduler_state_dict,
                          TrainPlateauScheduler)
from .control import make_candidates, controlled_head_step
from .decoder_control import decoder_transaction
from .window import WINDOW_DEFAULT, validate_window_config
from .meta_window import (geometry_named_parameters, frozen_geometry, meta_geometry_gradients,
                          _present_foreground_dice)

METHODS = {"supervised", "supervised_extra", "ema_full", "ema_full_meta_window", "ema_block", "sign",
           "independent_guard", "joint_no_class", "joint_no_guard", "joint_guard"}
OPTIONAL_FIELDS = {"supervised_loss", "lr_scheduler"}
DEFAULT = {"schema": SCHEMA, "method": "joint_guard", "update_scope": "head", "steps": 32, "warmup_steps": 8,
           "seed": 42, "body_lr": 3e-5, "head_lr": 3e-3, "pseudo_lr": 1e-3,
           "weight_decay": 1e-4, "ema_decay": .99, "pseudo_threshold": .7,
           "ridge": 1e-6, "risk_budget": 0., "guard_tolerance": 1e-8,
           "backtracks": 8, "save_every": 1000, "threads": 2, "batch_size": 1, "crop_size": 0,
           "window": copy.deepcopy(WINDOW_DEFAULT), "meta_window": {"lr": 1e-3, "clip_norm": 1.},
           "model": {"backbone": "dinov3_convnext_tiny", "source_repo": "",
                     "source_commit": "", "weights": "", "weights_sha256": ""}}


def validate_config(c):
    # Existing pinned v2 recipes retain their original single-slice behavior.
    c = copy.deepcopy(c)
    if "batch_size" not in c:
        c["batch_size"] = 1
    if "crop_size" not in c:
        c["crop_size"] = 0
    # Window experiments are opt-in. Old recipes keep the identical no-window path.
    c.setdefault("window", copy.deepcopy(WINDOW_DEFAULT))
    c.setdefault("meta_window", copy.deepcopy(DEFAULT["meta_window"]))
    extra = set(c) - set(DEFAULT)
    if extra - OPTIONAL_FIELDS or set(DEFAULT) - set(c) or c.get("schema") != SCHEMA or c.get("method") not in METHODS:
        raise ValueError("not an exact v2 recipe; legacy configs and unknown fields are rejected")
    if c["update_scope"] not in {"head", "decoder"}:
        raise ValueError("update_scope must be head or decoder")
    for k in ("steps", "warmup_steps", "seed", "backtracks", "save_every", "threads", "batch_size", "crop_size"):
        if type(c[k]) is not int or c[k] < (1 if k in {"steps", "backtracks", "save_every", "threads", "batch_size"} else 0):
            raise ValueError(f"invalid {k}")
    if c["warmup_steps"] > c["steps"]:
        raise ValueError("warmup exceeds total budget")
    for k in ("body_lr", "head_lr", "pseudo_lr", "weight_decay", "ema_decay", "pseudo_threshold", "ridge", "risk_budget", "guard_tolerance"):
        if isinstance(c[k], bool) or not isinstance(c[k], (float, int)) or not np.isfinite(c[k]) or c[k] < 0:
            raise ValueError(f"invalid {k}")
    if min(c["body_lr"], c["head_lr"], c["ridge"]) <= 0 or not 0 <= c["ema_decay"] < 1 or c["pseudo_threshold"] > 1:
        raise ValueError("invalid optimization range")
    if set(c["model"]) != set(DEFAULT["model"]):
        raise ValueError("model identity/provenance fields required")
    c["window"] = validate_window_config(c["window"])
    mw = c["meta_window"]
    if not isinstance(mw, dict) or set(mw) != {"lr", "clip_norm"}:
        raise ValueError("exact meta_window lr/clip_norm required")
    for k, v in mw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
            raise ValueError(f"invalid meta_window {k}")
    if c["method"] == "ema_full_meta_window" and c["window"]["mode"] != "adaptive":
        raise ValueError("ema_full_meta_window requires adaptive geometry")
    if c["window"]["mode"] != "none" and c["method"] not in {
            "ema_full", "ema_full_meta_window", "supervised", "supervised_extra"}:
        raise ValueError("window pilot supports full-EMA/supervised arms only, not guarded transactions")
    if "supervised_loss" in c:
        c["supervised_loss"] = validate_supervised_loss_config(c["supervised_loss"])
        if c["supervised_loss"] is not None and c["supervised_loss"].get("name") == "focal_dice":
            if c["method"] != "ema_full" or c["window"]["mode"] != "none":
                raise ValueError("supervised_loss focal_dice requires method ema_full with window mode none")
    if "lr_scheduler" in c:
        c["lr_scheduler"] = validate_lr_scheduler_config(c["lr_scheduler"])
        if c["lr_scheduler"] is not None:
            if c["method"] != "ema_full" or c["window"]["mode"] != "none":
                raise ValueError("lr_scheduler train_plateau requires method ema_full with window mode none")
    return copy.deepcopy(c)


def source_digest():
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "self_audit").glob("*.py")) + [root / "train.py"]
    return hashlib.sha256("".join(f"{p.name}:{digest(p)}\n" for p in files).encode()).hexdigest()


def ema_update(teacher, student, decay):
    with torch.no_grad():
        for t, s in zip(teacher.parameters(), student.parameters(), strict=True):
            t.mul_(decay).add_(s, alpha=1-decay)
        for t, s in zip(teacher.buffers(), student.buffers(), strict=True):
            t.copy_(s)


def supervised_step(model, optimizers, x, y, *, valid_mask=None, freeze_geometry=False, loss_config=None):
    model.train()
    model.zero_grad(set_to_none=True)
    with frozen_geometry(model) if freeze_geometry else nullcontext():
        loss = supervised_loss(model(x, valid_mask=valid_mask), y, config=loss_config)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite supervised loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                             5., error_if_nonfinite=True)
        for opt in optimizers:
            opt.step()
    return float(loss.detach()), float(norm)


def full_pseudo_step(model, optimizers, strong, py, accepted, *, input_valid=None, freeze_geometry=False):
    """One ordinary committed full-model pseudo step; no target or EMA refresh."""
    model.eval()
    model.zero_grad(set_to_none=True)
    with frozen_geometry(model) if freeze_geometry else nullcontext():
        loss = (F.cross_entropy(model(strong, valid_mask=input_valid), py, reduction="none") * accepted).sum() / accepted.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite pseudo loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                             5., error_if_nonfinite=True)
        for opt in optimizers:
            opt.step()
    return float(loss.detach()), float(norm)


def save_checkpoint(folder, model, teacher, optimizers, config, data, step, provenance, rng, *,
                    counts=None, seen=None, elapsed_seconds=0.,
                    lr_scheduler_state=None, lr_scheduler_meta=None):
    folder = Path(folder).resolve()
    if folder.exists():
        raise FileExistsError(f"checkpoint already exists: {folder}")
    tmp_folder = folder.parent / f".tmp_{folder.name}_{os.getpid()}_{time.time_ns()}"
    tmp_folder.mkdir(parents=True, exist_ok=False)
    try:
        payload = tmp_folder / "weights.pt"
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        torch.save({"student": model.state_dict(), "teacher": teacher.state_dict(),
                    "optimizers": [o.state_dict() for o in optimizers],
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": cuda_rng,
                    "py_rng": random.getstate(),
                    "model_training": model.training,
                    "teacher_training": teacher.training,
                    "lr_scheduler": lr_scheduler_state,
                    "scaler": None}, payload)
        np_global = np.random.get_state()
        metadata = {"schema": "self_audit.checkpoint.v2", "model_class": "NativeStudent",
                    "config": config, "step": step, "data_sha256": digest(data.path),
                    "labeled_ids": data.lids, "unlabeled_ids": data.uids,
                    "label_budget": len(data.lids), "audit_label_reads": sorted(data.label_reads),
                    "backbone": provenance, "source_sha256": source_digest(), "payload_sha256": digest(payload),
                    "synthetic": bool(data.manifest["synthetic"]), "quality": "NOT_EVALUATED",
                    "prediction_model": "student", "resume": "SUPPORTED" if counts is not None and seen is not None else "NOT_SUPPORTED",
                    "elapsed_seconds": float(elapsed_seconds),
                    "runtime": {"device_type": next(model.parameters()).device.type,
                                "cuda_devices": len(cuda_rng),
                                "torch_version": torch.__version__,
                                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
                    "numpy_rng": rng.bit_generator.state,
                    "numpy_global_rng": [np_global[0], np_global[1].tolist(), np_global[2], np_global[3], np_global[4]],
                    "seen_supervised": sorted(seen[0]) if seen else [],
                    "seen_audit": sorted(seen[1]) if seen else [],
                    "seen_unlabeled": sorted(seen[2]) if seen else [],
                    "counts": dict(counts) if counts else {},
                    "lr_scheduler": lr_scheduler_meta,
                    "scaler": None}
        write_json(tmp_folder / "MANIFEST.json", metadata)
        for durable in (payload, tmp_folder / "MANIFEST.json"):
            with durable.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(tmp_folder, folder)
        return digest(folder / "MANIFEST.json")
    except Exception:
        if tmp_folder.exists():
            shutil.rmtree(tmp_folder)
        raise


def load_checkpoint(folder, expected_sha256, device="cpu"):
    folder = Path(folder); mp = folder / "MANIFEST.json"
    if digest(mp) != expected_sha256:
        raise ValueError("checkpoint manifest hash mismatch")
    m = json.loads(mp.read_text())
    if m.get("schema") != "self_audit.checkpoint.v2" or digest(folder / "weights.pt") != m["payload_sha256"]:
        raise ValueError("invalid v2 checkpoint/payload")
    if source_digest() != m["source_sha256"]:
        raise ValueError("executed source differs from checkpoint; use its pinned worktree")
    model, _ = build_model(m["config"]["model"], synthetic=m["synthetic"], window=m["config"].get("window"))
    payload = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["student"], strict=True)
    return model.to(device).eval(), m


def train(config, manifest, manifest_sha256, out, *, device="cuda", synthetic=False,
          resume=None, resume_sha256=None, stop_after_step=None):
    if bool(resume) != bool(resume_sha256):
        raise ValueError("--resume and --resume-sha256 must be provided together")
    c = validate_config(config)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent device fallback")
    data = TrainingData(manifest, manifest_sha256, synthetic=synthetic)
    out = Path(out).resolve()
    if out == data.path.parent or data.path.parent.is_relative_to(out) or out.is_relative_to(data.path.parent):
        raise ValueError("output must be a new directory outside source data")
    if synthetic and c["model"]["backbone"] != "tiny_test":
        raise ValueError("synthetic smoke uses declared tiny_test, not a pretrained claim")
    if stop_after_step is not None and stop_after_step <= 0:
        raise ValueError("stop_after_step must be positive")

    if resume:
        resume_path = Path(resume).resolve()
        if resume_path.parent != out:
            raise ValueError("resume checkpoint must reside within the specified --out directory")
        if not out.is_dir():
            raise ValueError(f"output directory {out} does not exist for resume")
        mp = resume_path / "MANIFEST.json"
        if not mp.is_file():
            raise ValueError(f"manifest missing in resume checkpoint {resume_path}")
        if digest(mp) != resume_sha256:
            raise ValueError("resume manifest hash mismatch")
        m = json.loads(mp.read_text())
        if m.get("schema") != "self_audit.checkpoint.v2":
            raise ValueError("invalid checkpoint schema")
        if digest(resume_path / "weights.pt") != m.get("payload_sha256"):
            raise ValueError("checkpoint payload hash mismatch")
        if m.get("resume") != "SUPPORTED":
            raise ValueError(f"checkpoint resume status is {m.get('resume')}; resume NOT_SUPPORTED")
        if source_digest() != m.get("source_sha256"):
            raise ValueError("executed source differs from checkpoint; use its pinned worktree")
        if digest(data.path) != m.get("data_sha256"):
            raise ValueError("checkpoint data sha256 mismatch")
        if data.lids != m.get("labeled_ids") or data.uids != m.get("unlabeled_ids"):
            raise ValueError("checkpoint data labeled/unlabeled patient IDs mismatch")
        if bool(data.manifest["synthetic"]) != bool(m.get("synthetic")):
            raise ValueError("checkpoint synthetic flag mismatch")
        if c != m.get("config"):
            raise ValueError("recipe config does not match checkpoint config")
        required = {"numpy_rng", "numpy_global_rng", "counts", "seen_supervised", "seen_audit",
                    "seen_unlabeled", "audit_label_reads", "elapsed_seconds", "runtime"}
        if not required <= m.keys():
            raise ValueError("missing required resume metadata")
        if not {"supervised_updates", "pseudo_accepted", "ema_updates", "geometry_updates"} <= m["counts"].keys():
            raise ValueError("missing required resume counters")
        payload = torch.load(resume_path / "weights.pt", map_location="cpu", weights_only=True)
        required_payload = {"student", "teacher", "optimizers", "torch_rng", "cuda_rng", "py_rng",
                            "lr_scheduler", "scaler", "model_training", "teacher_training"}
        if not required_payload <= payload.keys():
            raise ValueError("missing required resume payload state")
        if device.startswith("cuda") and len(payload["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("missing/mismatched CUDA RNG states")
        runtime = m["runtime"]
        if (runtime["device_type"] != torch.device(device).type
                or runtime["torch_version"] != torch.__version__
                or runtime["cudnn_benchmark"] != torch.backends.cudnn.benchmark
                or runtime["cudnn_deterministic"] != torch.backends.cudnn.deterministic
                or runtime["deterministic_algorithms"] != torch.are_deterministic_algorithms_enabled()):
            raise ValueError("resume runtime settings differ")
        if c.get("lr_scheduler") is None:
            if m.get("lr_scheduler") is not None or payload.get("lr_scheduler") is not None:
                raise ValueError("checkpoint contains lr_scheduler state but config does not enable it")
        else:
            if m.get("lr_scheduler") is None or payload.get("lr_scheduler") is None:
                raise ValueError("checkpoint missing lr_scheduler state required by config")
            sched_state = payload["lr_scheduler"]
            validate_scheduler_state_dict(
                sched_state,
                config=c["lr_scheduler"],
                step=m["step"],
                warmup_steps=c["warmup_steps"],
                initial_body_lr=c["body_lr"],
                initial_head_lr=c["head_lr"],
                payload_optimizers=payload["optimizers"],
            )
            m_sched = m["lr_scheduler"]
            if not isinstance(m_sched, dict):
                raise ValueError("invalid lr_scheduler manifest metadata")
            if (m_sched.get("num_updates") != sched_state["num_updates"]
                    or m_sched.get("metric_count") != sched_state["metric_count"]
                    or abs(m_sched.get("metric_sum", 0.0) - sched_state["metric_sum"]) > 1e-6):
                raise ValueError("lr_scheduler manifest vs payload mismatch")
        start_step = m["step"]
        if type(start_step) is not int or start_step < 1 or m["counts"]["ema_updates"] != start_step:
            raise ValueError("invalid resume step/counters")
        if (out / "final").exists() or any(int(p.name.split("_")[1]) > start_step for p in out.glob("step_[0-9]*") if p.is_dir()):
            raise ValueError("newer checkpoint exists; refuse trajectory rollback")
        if start_step >= c["steps"]:
            raise ValueError(f"checkpoint step {start_step} already reached or exceeded configured steps {c['steps']}")
        if stop_after_step is not None and stop_after_step <= start_step:
            raise ValueError(f"stop_after_step {stop_after_step} must be greater than resume step {start_step}")
    else:
        if out.exists():
            raise ValueError("output must be a new directory outside source data")
        start_step = 0

    torch.manual_seed(c["seed"]); torch.set_num_threads(c["threads"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c["seed"])
    rng = np.random.default_rng(c["seed"])
    model, provenance = build_model(c["model"], synthetic=synthetic, window=c["window"])
    if resume and provenance != m.get("backbone"):
        raise ValueError("backbone provenance mismatch")
    model = model.to(device)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    # No head momentum to be contaminated by a rejected pseudo transaction.
    is_meta = c["method"] == "ema_full_meta_window"
    window_enabled = c["window"]["mode"] != "none"
    geometry = dict(geometry_named_parameters(model))
    body = [p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("head.") and not (is_meta and n in geometry)]
    # The meta virtual optimizer implements the plain, non-fused update.
    opts = [torch.optim.AdamW(body, lr=c["body_lr"], weight_decay=c["weight_decay"], foreach=False if is_meta else None),
            torch.optim.SGD(model.head.parameters(), lr=c["head_lr"], momentum=0., weight_decay=0., foreach=False if is_meta else None)]
    geometry_opt = torch.optim.SGD(geometry.values(), lr=c["meta_window"]["lr"], momentum=0., weight_decay=0.) if is_meta else None
    saved_opts = opts + ([geometry_opt] if geometry_opt is not None else [])
    scheduler = TrainPlateauScheduler(opts, c["lr_scheduler"], c["body_lr"], c["head_lr"]) if c.get("lr_scheduler") else None

    if not resume:
        out.mkdir(parents=True)
        write_json(out / "CONFIG.json", c)
        write_json(out / "STARTUP.json", {"schema": SCHEMA, "method": c["method"], "source_sha256": source_digest(),
                    "parameters": sum(p.numel() for p in model.parameters()), "device": device,
                    "backbone": provenance, "synthetic": synthetic, "label_budget": len(data.lids),
                    "supervision": "dense TRAIN subset; audit included; never no-GT",
                    "pseudo_update_scope": "full_except_meta_geometry" if is_meta else ("full" if c["method"] == "ema_full" else c["update_scope"]),
                    "window": c["window"], "meta_window": c["meta_window"] if is_meta else None,
                    "geometry_scope": "meta_only_after_frozen_warmup" if is_meta else c["window"]["mode"],
                    "outer_loss": "batch_present_foreground_soft_Dice; direct_geometry_detached" if is_meta else None,
                    "batch_size": c["batch_size"], "batch_protocol": "one_patient_per_role; native_grid; equal_slice_audit_mass",
                    "crop_size": c["crop_size"], "crop_policy": "image_geometry_random; edge_pad; ignore_padding_in_all_losses; no_resize",
                    "native_supervised_loss": c["supervised_loss"]["name"] if c.get("supervised_loss") else "CE+present_foreground_Dice",
                    "supervised_loss": c.get("supervised_loss"),
                    "lr_scheduler": c.get("lr_scheduler"),
                    "prediction_model": "student"})
        counts = {"supervised_updates": 0, "pseudo_accepted": 0, "ema_updates": 0, "geometry_updates": 0}
        seen_l, seen_u, seen_a = set(), set(), set()
        log = (out / "steps.jsonl").open("x")
    else:
        model.load_state_dict(payload["student"], strict=True)
        teacher.load_state_dict(payload["teacher"], strict=True)
        teacher.to(device).eval().requires_grad_(False)
        for opt, s in zip(saved_opts, payload["optimizers"], strict=True):
            opt.load_state_dict(s)
        if scheduler is not None:
            scheduler.load_state_dict(payload["lr_scheduler"])
        torch.set_rng_state(payload["torch_rng"])
        if torch.cuda.is_available() and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["py_rng"])
        model.train(payload["model_training"])
        teacher.train(payload["teacher_training"])
        rng.bit_generator.state = m["numpy_rng"]
        ng = m["numpy_global_rng"]
        np.random.set_state((ng[0], np.array(ng[1], dtype=np.uint32), ng[2], ng[3], ng[4]))
        data.label_reads.update(m["audit_label_reads"])
        counts = dict(m["counts"])
        seen_l = set(m["seen_supervised"])
        seen_a = set(m["seen_audit"])
        seen_u = set(m["seen_unlabeled"])
        recovery_tag = str(time.time_ns())
        if (out / "FAILURE.json").exists():
            os.replace(out / "FAILURE.json", out / f"FAILURE.before_resume_{recovery_tag}.json")
        steps_file = out / "steps.jsonl"
        original_log = steps_file.read_bytes()
        lines = original_log.splitlines(keepends=True)
        if len(lines) < start_step:
            raise ValueError("checkpoint log prefix missing")
        for expected_step, line in enumerate(lines[:start_step], 1):
            if json.loads(line).get("step") != expected_step:
                raise ValueError("checkpoint log prefix is not contiguous")
        if len(lines) > start_step:
            (out / f"steps.before_resume_{recovery_tag}.jsonl").write_bytes(original_log)
        temp_log = out / f".steps_{recovery_tag}.tmp"
        with temp_log.open("wb") as stream:
            stream.write(b"".join(lines[:start_step]))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_log, steps_file)
        log = steps_file.open("a")

    elapsed_before = float(m["elapsed_seconds"]) if resume else 0.
    start = time.monotonic() - elapsed_before
    try:
        for step in range(start_step, c["steps"]):
            # Round robin coverage; a separate RNG controls only data/augmentations.
            li, ui = step % len(data.lids), step % len(data.uids)
            ai = (li + 1 + int(rng.integers(len(data.lids) - 1))) % len(data.lids)
            xs, ys, sm, svalid = data.sample_batch(li, rng, c["batch_size"], labeled=True, crop_size=c["crop_size"], return_valid=True)
            xa, ya, am, avalid = data.sample_batch(ai, rng, c["batch_size"], labeled=True, crop_size=c["crop_size"], return_valid=True)
            xu, um, uvalid = data.sample_batch(ui, rng, c["batch_size"], labeled=False, crop_size=c["crop_size"], return_valid=True)
            assert sm["patient"] != am["patient"]
            xs, ys, svalid = aligned_flip(xs, ys, rng, svalid)
            xs, ys, xa, ya, xu = (a.to(device) for a in (xs, ys, xa, ya, xu))
            uvalid = uvalid.to(device)
            svalid, avalid = svalid.to(device), avalid.to(device)
            strong = strong_intensity(xu, rng)  # Generated for every arm, preserving data RNG schedule.
            seen_l.add(sm["patient"]); seen_a.add(am["patient"]); seen_u.add(um["patient"])
            freeze_geo = is_meta or (c["window"]["mode"] == "adaptive" and step < c["warmup_steps"])
            used_body_lr = float(opts[0].param_groups[0]["lr"])
            used_head_lr = float(opts[1].param_groups[0]["lr"])
            sloss, grad_norm = supervised_step(model, opts, xs, ys,
                                               valid_mask=svalid if window_enabled else None,
                                               freeze_geometry=freeze_geo,
                                               loss_config=c.get("supervised_loss"))
            counts["supervised_updates"] += 1
            row = {"step": step+1, "supervised": sm, "audit": am, "unlabeled": um,
                   "batch_size": c["batch_size"], "crop_size": c["crop_size"], "unlabeled_valid_pixels": int(uvalid.sum()),
                   "supervised_loss": sloss, "gradient_norm": grad_norm, "warmup": step < c["warmup_steps"]}
            if step >= c["warmup_steps"] and c["method"] != "supervised":
                if c["method"] == "supervised_extra":
                    row["extra_supervised_loss"], _ = supervised_step(model, opts, xa, ya,
                                                                      valid_mask=avalid if window_enabled else None,
                                                                      loss_config=c.get("supervised_loss"))
                    counts["supervised_updates"] += 1
                else:
                    model.eval()
                    with torch.no_grad():
                        q = teacher(xu, valid_mask=uvalid if window_enabled else None).softmax(1).detach()
                    if c["method"] in {"ema_full", "ema_full_meta_window"}:
                        conf, py = q.max(1); valid = (conf >= c["pseudo_threshold"]) & uvalid
                        row["accepted_pixels"] = int(valid.sum())
                        if valid.any():
                            if is_meta:
                                grads, meta = meta_geometry_gradients(model, opts, strong, py.detach(), valid.detach(), xa, ya,
                                                                     input_valid=uvalid, audit_valid=avalid, clip_norm=5.)
                                row["meta_window"] = meta
                                if grads:
                                    geometry_opt.zero_grad(set_to_none=True)
                                    before = {n: p.detach().clone() for n, p in geometry.items()}
                                    for n, p in geometry.items():
                                        p.grad = grads[n].detach().clone()
                                    gn = torch.nn.utils.clip_grad_norm_(list(geometry.values()), c["meta_window"]["clip_norm"], error_if_nonfinite=True)
                                    geometry_opt.step()
                                    geometry_opt.zero_grad(set_to_none=True)
                                    meta["geometry_gradient_norm"] = float(gn)
                                    meta["geometry_update_norm"] = float(torch.sqrt(sum((p.detach()-before[n]).square().sum() for n,p in geometry.items())))
                                    counts["geometry_updates"] += 1
                            row["pseudo_loss"], row["pseudo_gradient_norm"] = full_pseudo_step(
                                model, opts, strong, py.detach(), valid, input_valid=uvalid if window_enabled else None,
                                freeze_geometry=is_meta)
                            if is_meta and "reference_outer_dice_loss" in row["meta_window"]:
                                # A diagnostic on TRAIN optimization data, never an acceptance
                                # guarantee: new geometry + real update can differ from the trial.
                                with torch.no_grad():
                                    committed, _, _ = _present_foreground_dice(model(xa, valid_mask=avalid), ya, avalid)
                                if not torch.isfinite(committed):
                                    raise FloatingPointError("nonfinite committed TRAIN audit Dice")
                                row["meta_window"]["committed_outer_dice_loss"] = float(committed)
                                row["meta_window"]["actual_dice_loss_reduction"] = row["meta_window"]["reference_outer_dice_loss"] - float(committed)
                            counts["pseudo_accepted"] += 1
                        elif is_meta:
                            row["meta_window"] = {"status": "SKIPPED_EMPTY_PSEUDO_SUPPORT"}
                    elif c["update_scope"] == "decoder":
                        row["proposal"], row["control"] = decoder_transaction(model, strong, q, xa, ya,
                            method=c["method"], threshold=c["pseudo_threshold"], step_size=c["pseudo_lr"],
                            ridge=c["ridge"], budget=c["risk_budget"], backtracks=c["backtracks"], tolerance=c["guard_tolerance"], valid_mask=uvalid)
                        counts["pseudo_accepted"] += int(row["control"]["scale"] > 0)
                    else:
                        with torch.no_grad():
                            uf, af = model.features(strong).detach(), model.features(xa).detach()
                        directions, ids, info = make_candidates(model.head, uf, q, c["pseudo_threshold"], c["pseudo_lr"], valid_mask=uvalid)
                        row["proposal"] = info
                        row["control"] = controlled_head_step(model.head, af, ya, directions, ids,
                            method="ema_head" if c["method"] == "ema_block" else c["method"], ridge=c["ridge"], budget=c["risk_budget"],
                            backtracks=c["backtracks"], tolerance=c["guard_tolerance"])
                        counts["pseudo_accepted"] += int(row["control"]["scale"] > 0)
            # Exactly once, and only after the final committed student state.
            ema_update(teacher, model, c["ema_decay"]); counts["ema_updates"] += 1
            scheduler_stepped = False
            monitored_metric = None
            lr_reduced = False
            if scheduler is not None:
                scheduler.observe(sloss, is_warmup=(step < c["warmup_steps"]))
                scheduler_stepped, monitored_metric, lr_reduced = scheduler.step_if_interval_complete()

            next_body_lr = float(opts[0].param_groups[0]["lr"])
            next_head_lr = float(opts[1].param_groups[0]["lr"])
            if scheduler is not None:
                row["body_lr"] = used_body_lr
                row["head_lr"] = used_head_lr
                row["next_body_lr"] = next_body_lr
                row["next_head_lr"] = next_head_lr
                row["monitored_metric"] = monitored_metric
                row["scheduler_updated"] = scheduler_stepped
                row["lr_reduced"] = lr_reduced
                row["lr_scheduler"] = {
                    "used_body_lr": used_body_lr,
                    "used_head_lr": used_head_lr,
                    "next_body_lr": next_body_lr,
                    "next_head_lr": next_head_lr,
                    "monitored_metric": monitored_metric,
                    "updated": scheduler_stepped,
                    "lr_reduced": lr_reduced,
                    "accumulated_count": scheduler.metric_count,
                    "accumulated_sum": scheduler.metric_sum,
                    "num_updates": scheduler.num_updates,
                }
            row["elapsed_seconds"] = time.monotonic() - start
            row["cuda_peak_allocated"] = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
            log.write(json.dumps(row, allow_nan=False)+"\n"); log.flush()
            sched_state = scheduler.state_dict() if scheduler is not None else None
            sched_meta = scheduler.manifest_metadata() if scheduler is not None else None
            saved_sha = None
            if (step+1) % c["save_every"] == 0 and step+1 < c["steps"]:
                saved_sha = save_checkpoint(out / f"step_{step+1:06d}", model, teacher, saved_opts, c, data, step+1, provenance, rng, counts=counts, seen=(seen_l, seen_a, seen_u), elapsed_seconds=time.monotonic()-start, lr_scheduler_state=sched_state, lr_scheduler_meta=sched_meta)
            if stop_after_step is not None and step + 1 == stop_after_step:
                if saved_sha is None:
                    saved_sha = save_checkpoint(out / f"step_{step+1:06d}", model, teacher, saved_opts, c, data, step+1, provenance, rng, counts=counts, seen=(seen_l, seen_a, seen_u), elapsed_seconds=time.monotonic()-start, lr_scheduler_state=sched_state, lr_scheduler_meta=sched_meta)
                report = {"schema": SCHEMA, "status": "INTERRUPTED", "synthetic": synthetic, "quality": "NOT_EVALUATED",
                          "step": step+1, "steps": c["steps"], **counts, "elapsed_seconds": time.monotonic()-start,
                          "seen_supervised": sorted(seen_l), "seen_audit": sorted(seen_a), "seen_unlabeled": sorted(seen_u),
                          "checkpoint_manifest_sha256": saved_sha, "label_budget": len(data.lids),
                          "label_reads_outside_budget": sorted(data.label_reads-set(data.lids))}
                if scheduler is not None:
                    report["scheduler_updates"] = scheduler.num_updates
                write_json(out / "REPORT.json", report)
                return report
        sched_state = scheduler.state_dict() if scheduler is not None else None
        sched_meta = scheduler.manifest_metadata() if scheduler is not None else None
        sha = save_checkpoint(out / "final", model, teacher, saved_opts, c, data, c["steps"], provenance, rng, counts=counts, seen=(seen_l, seen_a, seen_u), elapsed_seconds=time.monotonic()-start, lr_scheduler_state=sched_state, lr_scheduler_meta=sched_meta)
        report = {"schema": SCHEMA, "status": "COMPLETED", "synthetic": synthetic, "quality": "NOT_EVALUATED",
                  "steps": c["steps"], **counts, "elapsed_seconds": time.monotonic()-start,
                  "seen_supervised": sorted(seen_l), "seen_audit": sorted(seen_a), "seen_unlabeled": sorted(seen_u),
                  "final_manifest_sha256": sha, "label_budget": len(data.lids),
                  "label_reads_outside_budget": sorted(data.label_reads-set(data.lids)),
                  "guard_scope": "none; meta-Dice is not a finite-step Dice guarantee" if is_meta else "extra pseudo step vs supervised-only state on current TRAIN audit slice only"}
        if scheduler is not None:
            report["scheduler_updates"] = scheduler.num_updates
        write_json(out / "REPORT.json", report)
        return report
    except Exception as exc:
        write_json(out / "FAILURE.json", {"status": "FAILED", "error": repr(exc), **counts})
        raise
    finally:
        log.close()
