"""One coherent pipeline: labeled reference -> pseudo transaction -> EMA once."""
from pathlib import Path
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

METHODS = {"supervised", "supervised_extra", "ema_full", "ema_block", "sign",
           "independent_guard", "joint_no_class", "joint_no_guard", "joint_guard"}
DEFAULT = {"schema": SCHEMA, "method": "joint_guard", "update_scope": "head", "steps": 32, "warmup_steps": 8,
           "seed": 42, "body_lr": 3e-5, "head_lr": 3e-3, "pseudo_lr": 1e-3,
           "weight_decay": 1e-4, "ema_decay": .99, "pseudo_threshold": .7,
           "ridge": 1e-6, "risk_budget": 0., "guard_tolerance": 1e-8,
           "backtracks": 8, "save_every": 1000, "threads": 2, "batch_size": 1,
           "model": {"backbone": "dinov3_convnext_tiny", "source_repo": "",
                     "source_commit": "", "weights": "", "weights_sha256": ""}}


def validate_config(c):
    # Existing pinned v2 recipes retain their original single-slice behavior.
    c = copy.deepcopy(c)
    if "batch_size" not in c:
        c["batch_size"] = 1
    if set(c) != set(DEFAULT) or c.get("schema") != SCHEMA or c.get("method") not in METHODS:
        raise ValueError("not an exact v2 recipe; legacy configs and unknown fields are rejected")
    if c["update_scope"] not in {"head", "decoder"}:
        raise ValueError("update_scope must be head or decoder")
    for k in ("steps", "warmup_steps", "seed", "backtracks", "save_every", "threads", "batch_size"):
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


def supervised_step(model, optimizers, x, y):
    model.train()
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    loss = supervised_loss(model(x), y)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite supervised loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    for opt in optimizers:
        opt.step()
    return float(loss.detach()), float(norm)


def save_checkpoint(folder, model, teacher, optimizers, config, data, step, provenance, rng):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    payload = folder / "weights.pt"
    torch.save({"student": model.state_dict(), "teacher": teacher.state_dict(),
                "optimizers": [o.state_dict() for o in optimizers],
                "torch_rng": torch.get_rng_state()}, payload)
    metadata = {"schema": "self_audit.checkpoint.v2", "model_class": "NativeStudent",
                "config": config, "step": step, "data_sha256": digest(data.path),
                "labeled_ids": data.lids, "unlabeled_ids": data.uids,
                "label_budget": len(data.lids), "audit_label_reads": sorted(data.label_reads),
                "backbone": provenance, "source_sha256": source_digest(), "payload_sha256": digest(payload),
                "synthetic": bool(data.manifest["synthetic"]), "quality": "NOT_EVALUATED",
                "prediction_model": "student", "resume": "NOT_SUPPORTED",
                "numpy_rng": rng.bit_generator.state}
    write_json(folder / "MANIFEST.json", metadata)
    return digest(folder / "MANIFEST.json")


def load_checkpoint(folder, expected_sha256, device="cpu"):
    folder = Path(folder); mp = folder / "MANIFEST.json"
    if digest(mp) != expected_sha256:
        raise ValueError("checkpoint manifest hash mismatch")
    m = json.loads(mp.read_text())
    if m.get("schema") != "self_audit.checkpoint.v2" or digest(folder / "weights.pt") != m["payload_sha256"]:
        raise ValueError("invalid v2 checkpoint/payload")
    if source_digest() != m["source_sha256"]:
        raise ValueError("executed source differs from checkpoint; use its pinned worktree")
    model, _ = build_model(m["config"]["model"], synthetic=m["synthetic"])
    payload = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["student"], strict=True)
    return model.to(device).eval(), m


def train(config, manifest, manifest_sha256, out, *, device="cuda", synthetic=False):
    c = validate_config(config)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent device fallback")
    data = TrainingData(manifest, manifest_sha256, synthetic=synthetic)
    out = Path(out).resolve()
    if out.exists() or out == data.path.parent or data.path.parent.is_relative_to(out) or out.is_relative_to(data.path.parent):
        raise ValueError("output must be a new directory outside source data")
    if synthetic and c["model"]["backbone"] != "tiny_test":
        raise ValueError("synthetic smoke uses declared tiny_test, not a pretrained claim")
    torch.manual_seed(c["seed"]); torch.set_num_threads(c["threads"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c["seed"])
    rng = np.random.default_rng(c["seed"])
    model, provenance = build_model(c["model"], synthetic=synthetic)
    model = model.to(device)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    # No head momentum to be contaminated by a rejected pseudo transaction.
    body = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    opts = [torch.optim.AdamW(body, lr=c["body_lr"], weight_decay=c["weight_decay"]),
            torch.optim.SGD(model.head.parameters(), lr=c["head_lr"], momentum=0., weight_decay=0.)]
    out.mkdir(parents=True)
    write_json(out / "CONFIG.json", c)
    write_json(out / "STARTUP.json", {"schema": SCHEMA, "method": c["method"], "source_sha256": source_digest(),
                "parameters": sum(p.numel() for p in model.parameters()), "device": device,
                "backbone": provenance, "synthetic": synthetic, "label_budget": len(data.lids),
                "supervision": "dense TRAIN subset; audit included; never no-GT",
                "pseudo_update_scope": "full" if c["method"] == "ema_full" else c["update_scope"],
                "batch_size": c["batch_size"], "batch_protocol": "one_patient_per_role; native_grid; equal_slice_audit_mass",
                "native_supervised_loss": "CE+present_foreground_Dice", "prediction_model": "student"})
    start = time.monotonic(); counts = {"supervised_updates": 0, "pseudo_accepted": 0, "ema_updates": 0}
    seen_l, seen_u, seen_a = set(), set(), set()
    log = (out / "steps.jsonl").open("x")
    try:
        for step in range(c["steps"]):
            # Round robin coverage; a separate RNG controls only data/augmentations.
            li, ui = step % len(data.lids), step % len(data.uids)
            ai = (li + 1 + int(rng.integers(len(data.lids) - 1))) % len(data.lids)
            xs, ys, sm = data.sample_batch(li, rng, c["batch_size"], labeled=True)
            xa, ya, am = data.sample_batch(ai, rng, c["batch_size"], labeled=True)
            xu, um = data.sample_batch(ui, rng, c["batch_size"], labeled=False)
            assert sm["patient"] != am["patient"]
            xs, ys = aligned_flip(xs, ys, rng)
            xs, ys, xa, ya, xu = (a.to(device) for a in (xs, ys, xa, ya, xu))
            strong = strong_intensity(xu, rng)  # Generated for every arm, preserving data RNG schedule.
            seen_l.add(sm["patient"]); seen_a.add(am["patient"]); seen_u.add(um["patient"])
            sloss, grad_norm = supervised_step(model, opts, xs, ys)
            counts["supervised_updates"] += 1
            row = {"step": step+1, "supervised": sm, "audit": am, "unlabeled": um,
                   "supervised_loss": sloss, "gradient_norm": grad_norm, "warmup": step < c["warmup_steps"]}
            if step >= c["warmup_steps"] and c["method"] != "supervised":
                if c["method"] == "supervised_extra":
                    row["extra_supervised_loss"], _ = supervised_step(model, opts, xa, ya)
                    counts["supervised_updates"] += 1
                else:
                    model.eval()
                    with torch.no_grad():
                        q = teacher(xu).softmax(1).detach()
                    if c["method"] == "ema_full":
                        conf, py = q.max(1); valid = conf >= c["pseudo_threshold"]
                        row["accepted_pixels"] = int(valid.sum())
                        if valid.any():
                            for opt in opts: opt.zero_grad(set_to_none=True)
                            loss = (F.cross_entropy(model(strong), py, reduction="none") * valid).sum() / valid.sum()
                            if not torch.isfinite(loss): raise FloatingPointError("nonfinite pseudo loss")
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                            for opt in opts: opt.step()
                            counts["pseudo_accepted"] += 1
                    elif c["update_scope"] == "decoder":
                        row["proposal"], row["control"] = decoder_transaction(model, strong, q, xa, ya,
                            method=c["method"], threshold=c["pseudo_threshold"], step_size=c["pseudo_lr"],
                            ridge=c["ridge"], budget=c["risk_budget"], backtracks=c["backtracks"], tolerance=c["guard_tolerance"])
                        counts["pseudo_accepted"] += int(row["control"]["scale"] > 0)
                    else:
                        with torch.no_grad():
                            uf, af = model.features(strong).detach(), model.features(xa).detach()
                        directions, ids, info = make_candidates(model.head, uf, q, c["pseudo_threshold"], c["pseudo_lr"])
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
                save_checkpoint(out / f"step_{step+1:06d}", model, teacher, opts, c, data, step+1, provenance, rng)
        sha = save_checkpoint(out / "final", model, teacher, opts, c, data, c["steps"], provenance, rng)
        report = {"schema": SCHEMA, "status": "COMPLETED", "synthetic": synthetic, "quality": "NOT_EVALUATED",
                  "steps": c["steps"], **counts, "elapsed_seconds": time.monotonic()-start,
                  "seen_supervised": sorted(seen_l), "seen_audit": sorted(seen_a), "seen_unlabeled": sorted(seen_u),
                  "final_manifest_sha256": sha, "label_budget": len(data.lids),
                  "label_reads_outside_budget": sorted(data.label_reads-set(data.lids)),
                  "guard_scope": "extra pseudo step vs supervised-only state on current TRAIN audit slice only"}
        write_json(out / "REPORT.json", report)
        return report
    except Exception as exc:
        write_json(out / "FAILURE.json", {"status": "FAILED", "error": repr(exc), **counts})
        raise
    finally:
        log.close()
