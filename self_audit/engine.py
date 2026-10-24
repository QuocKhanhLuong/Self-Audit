"""One coherent pipeline: labeled reference -> pseudo transaction -> EMA once."""
from pathlib import Path
from contextlib import nullcontext
import copy
import hashlib
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import SCHEMA
from .model import build_model, digest
from .data import TrainingData, write_json
from .losses import supervised_loss, strong_intensity, aligned_flip
from .control import make_candidates, controlled_head_step
from .decoder_control import decoder_transaction
from .window import WINDOW_DEFAULT, validate_window_config
from .meta_window import (geometry_named_parameters, frozen_geometry, meta_geometry_gradients,
                          _present_foreground_dice)
from .teacher_memory import MemoryBank, load_bank, permute_endpoints, random_feature_control
from .external_pseudo import ExternalPseudoCache

METHODS = {"supervised", "supervised_extra", "ema_full", "ema_full_meta_window", "ema_block", "sign",
           "independent_guard", "joint_no_class", "joint_no_guard", "joint_guard"}
DEFAULT = {"schema": SCHEMA, "method": "joint_guard", "update_scope": "head", "steps": 32, "warmup_steps": 8,
           "seed": 42, "body_lr": 3e-5, "head_lr": 3e-3, "pseudo_lr": 1e-3,
           "weight_decay": 1e-4, "ema_decay": .99, "pseudo_threshold": .7,
           "ridge": 1e-6, "risk_budget": 0., "guard_tolerance": 1e-8,
           "backtracks": 8, "save_every": 1000, "threads": 2, "batch_size": 1, "crop_size": 0,
           "window": copy.deepcopy(WINDOW_DEFAULT), "meta_window": {"lr": 1e-3, "clip_norm": 1.},
           "teacher_memory": {"arm": "A0", "bank_path": None, "bank_sha256": None,
                              "cache_path": None, "cache_sha256": None, "tau": .2,
                              "external_weight": .25},
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
    c.setdefault("teacher_memory", copy.deepcopy(DEFAULT["teacher_memory"]))
    if set(c) != set(DEFAULT) or c.get("schema") != SCHEMA or c.get("method") not in METHODS:
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
    tm = c["teacher_memory"]
    if not isinstance(tm, dict) or set(tm) != set(DEFAULT["teacher_memory"]):
        raise ValueError("exact teacher_memory arm/assets/tau/weight fields required")
    if tm["arm"] not in {"A0", "A1", "A2", "A3", "A4", "A5", "A6"}:
        raise ValueError("unknown teacher-memory pilot arm")
    for key in ("tau", "external_weight"):
        value = tm[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"teacher_memory.{key} must be finite and positive")
    arm = tm["arm"]
    if arm != "A0" and (c["method"] != "ema_full" or c["window"]["mode"] != "fixed"):
        raise ValueError("teacher-memory pilot arms require fixed window and ema_full")
    needs_bank, needs_cache = arm in {"A2", "A3", "A4", "A5", "A6"}, arm == "A1"
    if needs_bank:
        path, expected = tm["bank_path"], tm["bank_sha256"]
        if not isinstance(path, str) or not path or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("memory arms require an explicit bank_path and SHA-256")
        if any(ch not in "0123456789abcdefABCDEF" for ch in expected) or not Path(path).is_file() or digest(path) != expected.lower():
            raise ValueError("memory bank file/hash is missing or mismatched")
    elif tm["bank_path"] is not None or tm["bank_sha256"] is not None:
        raise ValueError("this teacher-memory arm must not declare a bank")
    if needs_cache:
        path, expected = tm["cache_path"], tm["cache_sha256"]
        if not isinstance(path, str) or not path or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("A1 requires an explicit external pseudo cache and SHA-256")
        if any(ch not in "0123456789abcdefABCDEF" for ch in expected) or not Path(path).is_file() or digest(path) != expected.lower():
            raise ValueError("external pseudo cache file/hash is missing or mismatched")
    elif tm["cache_path"] is not None or tm["cache_sha256"] is not None:
        raise ValueError("this teacher-memory arm must not declare an external pseudo cache")
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
    return copy.deepcopy(c)


def source_digest():
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "self_audit").glob("*.py")) + [root / "train.py"]
    return hashlib.sha256("".join(f"{p.name}:{digest(p)}\n" for p in files).encode()).hexdigest()


def effective_bank_digest(bank):
    """Stable digest of transformed/in-memory bank arrays and metadata."""
    h = hashlib.sha256()
    for name in ("endpoints", "confidences", "valid"):
        value = getattr(bank, name).detach().cpu().contiguous().numpy()
        h.update(name.encode("ascii")); h.update(str(value.shape).encode("ascii"))
        h.update(value.dtype.str.encode("ascii")); h.update(value.tobytes())
    h.update(json.dumps(bank.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    return h.hexdigest()


def ema_update(teacher, student, decay):
    with torch.no_grad():
        for t, s in zip(teacher.parameters(), student.parameters(), strict=True):
            t.mul_(decay).add_(s, alpha=1-decay)
        for t, s in zip(teacher.buffers(), student.buffers(), strict=True):
            t.copy_(s)


def supervised_step(model, optimizers, x, y, *, valid_mask=None, freeze_geometry=False):
    model.train()
    model.zero_grad(set_to_none=True)
    with frozen_geometry(model) if freeze_geometry else nullcontext():
        loss = supervised_loss(model(x, valid_mask=valid_mask), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite supervised loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                             5., error_if_nonfinite=True)
        for opt in optimizers:
            opt.step()
    return float(loss.detach()), float(norm)


def full_pseudo_step(model, optimizers, strong, py, accepted, *, input_valid=None, freeze_geometry=False,
                     external_probabilities=None, external_accepted=None, external_weight=0.0):
    """One ordinary committed full-model pseudo step; no target or EMA refresh."""
    model.eval()
    model.zero_grad(set_to_none=True)
    with frozen_geometry(model) if freeze_geometry else nullcontext():
        logits = model(strong, valid_mask=input_valid)
        if accepted.any():
            loss = (F.cross_entropy(logits, py, reduction="none") * accepted).sum() / accepted.sum()
        else:
            loss = logits.sum() * 0.0
        if external_probabilities is not None:
            if (external_probabilities.shape != logits.shape or external_probabilities.device != logits.device
                    or not torch.is_floating_point(external_probabilities)):
                raise ValueError("external probabilities must match native student logits")
            if (external_accepted is None or external_accepted.shape != accepted.shape
                    or external_accepted.dtype is not torch.bool or external_accepted.device != logits.device):
                raise ValueError("external pseudo support must match EMA pseudo support")
            external_ce = -(external_probabilities * logits.log_softmax(1)).sum(1)
            if external_accepted.any():
                ext_loss = (external_ce * external_accepted).sum() / external_accepted.sum()
                loss = loss + float(external_weight) * ext_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite pseudo loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                             5., error_if_nonfinite=True)
        for opt in optimizers:
            opt.step()
    return float(loss.detach()), float(norm)


def save_checkpoint(folder, model, teacher, optimizers, config, data, step, provenance, rng):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    payload = folder / "weights.pt"
    torch.save({"student": model.state_dict(), "teacher": teacher.state_dict(),
                "optimizers": [o.state_dict() for o in optimizers],
                "torch_rng": torch.get_rng_state()}, payload)
    memory = model.teacher_memory
    memory_record = None if memory is None else {
        "arm": config["teacher_memory"]["arm"], "mode": memory.mode, "tau": memory.tau,
        "source_bank_sha256": config["teacher_memory"]["bank_sha256"],
        "effective_bank_sha256": effective_bank_digest(memory.bank),
        "external_pseudo_cache_sha256": config["teacher_memory"]["cache_sha256"],
    }
    metadata = {"schema": "self_audit.checkpoint.v2", "model_class": "NativeStudent",
                "config": config, "step": step, "data_sha256": digest(data.path),
                "labeled_ids": data.lids, "unlabeled_ids": data.uids,
                "label_budget": len(data.lids), "audit_label_reads": sorted(data.label_reads),
                "backbone": provenance, "source_sha256": source_digest(), "payload_sha256": digest(payload),
                "synthetic": bool(data.manifest["synthetic"]), "quality": "NOT_EVALUATED",
                "prediction_model": "student", "resume": "NOT_SUPPORTED",
                "numpy_rng": rng.bit_generator.state, "teacher_memory": memory_record}
    write_json(folder / "MANIFEST.json", metadata)
    return digest(folder / "MANIFEST.json")


def _build_student_from_state(config, synthetic, state):
    tm = config.get("teacher_memory", DEFAULT["teacher_memory"])
    bank = None
    if tm.get("arm") in {"A2", "A3", "A4", "A5", "A6"}:
        prefix = "teacher_memory.bank."
        required = {prefix + "endpoints", prefix + "confidences", prefix + "valid"}
        if not required.issubset(state):
            raise ValueError("checkpoint memory arm is missing embedded fixed-bank buffers")
        bank = MemoryBank(state[prefix + "endpoints"].numpy(), state[prefix + "confidences"].numpy(),
                          state[prefix + "valid"].numpy(), {"provenance": {"source": "checkpoint_embedded"}})
    mode = "unpaired" if tm.get("arm") == "A2" else "paired"
    model, _ = build_model(config["model"], synthetic=synthetic,
                           window=config.get("window"), memory_bank=bank,
                           memory_mode=mode, memory_tau=tm.get("tau", .2))
    model.load_state_dict(state, strict=True)
    return model


def load_checkpoint(folder, expected_sha256, device="cpu"):
    folder = Path(folder); mp = folder / "MANIFEST.json"
    if digest(mp) != expected_sha256:
        raise ValueError("checkpoint manifest hash mismatch")
    m = json.loads(mp.read_text())
    if m.get("schema") != "self_audit.checkpoint.v2" or digest(folder / "weights.pt") != m["payload_sha256"]:
        raise ValueError("invalid v2 checkpoint/payload")
    if source_digest() != m["source_sha256"]:
        raise ValueError("executed source differs from checkpoint; use its pinned worktree")
    payload = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    model = _build_student_from_state(m["config"], m["synthetic"], payload["student"])
    return model.to(device).eval(), m


def load_exported_student(folder, expected_sha256, device="cpu"):
    """Load a model-only export; no teacher, bank file, cache, or pickle needed."""
    folder = Path(folder); mp = folder / "MANIFEST.json"
    if digest(mp) != expected_sha256:
        raise ValueError("student export manifest hash mismatch")
    metadata = json.loads(mp.read_text())
    if metadata.get("schema") != "self_audit.student_export.v1" or digest(folder / "weights.pt") != metadata.get("payload_sha256"):
        raise ValueError("invalid student export/payload")
    if source_digest() != metadata.get("source_sha256"):
        raise ValueError("executed source differs from student export")
    payload = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    model = _build_student_from_state(metadata["config"], metadata["synthetic"], payload["student"])
    return model.to(device).eval(), metadata


def train(config, manifest, manifest_sha256, out, *, device="cuda", synthetic=False):
    c = validate_config(config)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent device fallback")
    data = TrainingData(manifest, manifest_sha256, synthetic=synthetic)
    tm = c["teacher_memory"]
    external_cache = ExternalPseudoCache(tm["cache_path"], tm["cache_sha256"], data) if tm["arm"] == "A1" else None
    out = Path(out).resolve()
    if out.exists() or out == data.path.parent or data.path.parent.is_relative_to(out) or out.is_relative_to(data.path.parent):
        raise ValueError("output must be a new directory outside source data")
    if synthetic and c["model"]["backbone"] != "tiny_test":
        raise ValueError("synthetic smoke uses declared tiny_test, not a pretrained claim")
    torch.manual_seed(c["seed"]); torch.set_num_threads(c["threads"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c["seed"])
    rng = np.random.default_rng(c["seed"])
    memory_bank = None
    if tm["arm"] in {"A2", "A3", "A4", "A5", "A6"}:
        memory_bank = load_bank(tm["bank_path"], tm["bank_sha256"])
        if tm["arm"] == "A4":
            memory_bank = permute_endpoints(memory_bank, seed=c["seed"])
        elif tm["arm"] == "A6":
            memory_bank = random_feature_control(memory_bank, seed=c["seed"])
        if tm["arm"] == "A5":
            lineage = memory_bank.metadata["provenance"]
            payload_sha = lineage.get("student_checkpoint_payload_sha256")
            if (lineage.get("mode") != "A5_frozen_student_only"
                    or not isinstance(payload_sha, str) or len(payload_sha) != 64
                    or lineage.get("teacher_weights_sha256") is not None):
                raise ValueError("A5 requires a frozen-student-only bank with no teacher-derived selection")
    memory_mode = "unpaired" if tm["arm"] == "A2" else "paired"
    model, provenance = build_model(c["model"], synthetic=synthetic, window=c["window"],
                                    memory_bank=memory_bank, memory_mode=memory_mode, memory_tau=tm["tau"])
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
    out.mkdir(parents=True)
    write_json(out / "CONFIG.json", c)
    write_json(out / "STARTUP.json", {"schema": SCHEMA, "method": c["method"], "source_sha256": source_digest(),
                "parameters": sum(p.numel() for p in model.parameters()), "device": device,
                "backbone": provenance, "synthetic": synthetic, "label_budget": len(data.lids),
                "supervision": "dense TRAIN subset; audit included; never no-GT",
                "pseudo_update_scope": "full_except_meta_geometry" if is_meta else ("full" if c["method"] == "ema_full" else c["update_scope"]),
                "window": c["window"], "meta_window": c["meta_window"] if is_meta else None,
                "teacher_memory_arm": tm["arm"], "teacher_memory_bank_sha256": tm["bank_sha256"],
                "teacher_memory_effective_bank_sha256": effective_bank_digest(memory_bank) if memory_bank is not None else None,
                "external_pseudo_cache_sha256": tm["cache_sha256"] if external_cache is not None else None,
                "geometry_scope": "meta_only_after_frozen_warmup" if is_meta else c["window"]["mode"],
                "outer_loss": "batch_present_foreground_soft_Dice; direct_geometry_detached" if is_meta else None,
                "batch_size": c["batch_size"], "batch_protocol": "one_patient_per_role; native_grid; equal_slice_audit_mass",
                "crop_size": c["crop_size"], "crop_policy": "image_geometry_random; edge_pad; ignore_padding_in_all_losses; no_resize",
                "native_supervised_loss": "CE+present_foreground_Dice", "prediction_model": "student"})
    start = time.monotonic(); counts = {"supervised_updates": 0, "pseudo_accepted": 0, "ema_updates": 0, "geometry_updates": 0}
    seen_l, seen_u, seen_a = set(), set(), set()
    log = (out / "steps.jsonl").open("x")
    try:
        for step in range(c["steps"]):
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
            sloss, grad_norm = supervised_step(model, opts, xs, ys,
                                               valid_mask=svalid if window_enabled else None,
                                               freeze_geometry=freeze_geo)
            counts["supervised_updates"] += 1
            row = {"step": step+1, "supervised": sm, "audit": am, "unlabeled": um,
                   "batch_size": c["batch_size"], "crop_size": c["crop_size"], "unlabeled_valid_pixels": int(uvalid.sum()),
                   "supervised_loss": sloss, "gradient_norm": grad_norm, "warmup": step < c["warmup_steps"]}
            if step >= c["warmup_steps"] and c["method"] != "supervised":
                if c["method"] == "supervised_extra":
                    row["extra_supervised_loss"], _ = supervised_step(model, opts, xa, ya,
                                                                      valid_mask=avalid if window_enabled else None)
                    counts["supervised_updates"] += 1
                else:
                    model.eval()
                    with torch.no_grad():
                        q = teacher(xu, valid_mask=uvalid if window_enabled else None).softmax(1).detach()
                    if c["method"] in {"ema_full", "ema_full_meta_window"}:
                        conf, py = q.max(1); valid = (conf >= c["pseudo_threshold"]) & uvalid
                        ext_probs = ext_valid = None
                        if external_cache is not None:
                            ext_probs, ext_support = external_cache.sample_batch(um, device=device)
                            ext_conf = ext_probs.max(1).values
                            ext_valid = (ext_conf >= c["pseudo_threshold"]) & uvalid & ext_support
                        row["accepted_pixels"] = int(valid.sum())
                        if ext_valid is not None:
                            row["external_accepted_pixels"] = int(ext_valid.sum())
                            row["external_loss_pixels"] = int(ext_valid.sum()) if valid.any() else 0
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
                                freeze_geometry=is_meta, external_probabilities=ext_probs,
                                external_accepted=ext_valid,
                                external_weight=tm["external_weight"] if ext_probs is not None else 0.0)
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
            row["elapsed_seconds"] = time.monotonic() - start
            row["cuda_peak_allocated"] = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
            log.write(json.dumps(row, allow_nan=False)+"\n"); log.flush()
            if (step+1) % c["save_every"] == 0 and step+1 < c["steps"]:
                save_checkpoint(out / f"step_{step+1:06d}", model, teacher, saved_opts, c, data, step+1, provenance, rng)
        sha = save_checkpoint(out / "final", model, teacher, saved_opts, c, data, c["steps"], provenance, rng)
        report = {"schema": SCHEMA, "status": "COMPLETED", "synthetic": synthetic, "quality": "NOT_EVALUATED",
                  "steps": c["steps"], **counts, "elapsed_seconds": time.monotonic()-start,
                  "seen_supervised": sorted(seen_l), "seen_audit": sorted(seen_a), "seen_unlabeled": sorted(seen_u),
                  "final_manifest_sha256": sha, "label_budget": len(data.lids),
                  "teacher_memory_arm": tm["arm"], "teacher_memory_source_bank_sha256": tm["bank_sha256"],
                  "teacher_memory_effective_bank_sha256": effective_bank_digest(memory_bank) if memory_bank is not None else None,
                  "label_reads_outside_budget": sorted(data.label_reads-set(data.lids)),
                  "guard_scope": "none; meta-Dice is not a finite-step Dice guarantee" if is_meta else "extra pseudo step vs supervised-only state on current TRAIN audit slice only"}
        write_json(out / "REPORT.json", report)
        return report
    except Exception as exc:
        write_json(out / "FAILURE.json", {"status": "FAILED", "error": repr(exc), **counts})
        raise
    finally:
        log.close()
