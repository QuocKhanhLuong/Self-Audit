"""Opt-in fixed-step Q versus Z technical continuation on common TRAIN support.

Both arms start at the original checkpoint with fresh, equal AdamW optimizers.
This is not an original-optimizer resume, a reproduction of the untouched full-Q
baseline, or evidence of anatomy accuracy. No export, evaluator, or raw-image
reader is called. The prepared bank must validate against the ORIGINAL bank.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

MAX_STEPS = 128
# Fixed numerical guard, never fitted to VAL. Apply to each batch AND patient.
EXCLUDED_WEIGHT_RTOL = 1e-6
CLASSES = ("BG", "RV", "MYO", "LV")
PLAN_SCHEMA = "guidance_signal.paired_pilot.v1"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def state_hash(state):
    """Canonical sorted key/dtype/shape/raw-CPU-byte hash; not torch.save bytes."""
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                            separators=(",", ":")).encode()
        result.update(len(header).to_bytes(8, "big")); result.update(header)
        result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def original_policy(ck, recipe):
    """The plan pins the exact unchanged choices, including inference vetoes."""
    return dict(context="original_z_minus_1_z_z_plus_1_clamped",
                target_change="Q_to_Z_on_common_valid_h_only",
                activation="original_Q_argmax_foreground_and_positive_original_weight",
                permissions="original_Q_hard_labels_at_least_two_train_patients",
                allowed_classes=ck["allowed_classes"], config=ck["config"],
                width=ck["width"], feature_dim=ck["feature_dim"],
                lr=recipe["lr"], optimizer="fresh_AdamW_not_original_optimizer_resume",
                weight_decay=1e-4, betas=[0.9, 0.999], eps=1e-8,
                clip_grad_norm=5.0, feature_loss="original_frozen_feature_cosine",
                deterministic_algorithms=True, deterministic_warn_only=False,
                cudnn_deterministic=True, cudnn_benchmark=False,
                cublas_workspace_config=CUBLAS_WORKSPACE_CONFIG,
                denominator="original_full_weight_sum_clamp_min_1e-8",
                excluded_weight_rtol=EXCLUDED_WEIGHT_RTOL,
                excluded_weight_guard="excluded_weight_divided_by_max_total_weight_1e-8")


def validate_plan(plan, bank):
    required = {"schema", "patient_ids", "steps", "seed", "batch_size",
                "checkpoint_sha256", "model_state_sha256", "source_sha256",
                "bank_manifest_sha256", "signal_manifest_sha256", "recipe_sha256", "policy"}
    if set(plan) != required or plan["schema"] != PLAN_SCHEMA:
        raise ValueError("plan must contain exactly the pinned pilot v1 fields; no epochs")
    for key in ("steps", "batch_size", "seed"):
        if type(plan[key]) is not int:
            raise ValueError(f"plan {key} must be an integer")
    if not 1 <= plan["steps"] <= MAX_STEPS or plan["batch_size"] < 1 or plan["seed"] < 0:
        raise ValueError("pilot requires 1..128 fixed steps, positive batch_size and nonnegative seed")
    ids = plan["patient_ids"]
    if not isinstance(ids, list) or not ids or any(not isinstance(p, str) for p in ids):
        raise ValueError("selected TRAIN patient_ids required")
    splits = bank["locked_splits"]
    if (bank.get("split") != "train" or len(ids) != len(set(ids))
            or not set(ids) <= set(bank["patients"])
            or not set(bank["patients"]) <= set(splits["train"])
            or set(ids) & (set(splits.get("val", [])) | set(splits.get("test", [])))):
        raise ValueError("pilot permits disjoint selected TRAIN patients only; VAL/TEST forbidden")
    for key in ("checkpoint_sha256", "model_state_sha256", "bank_manifest_sha256",
                "signal_manifest_sha256", "recipe_sha256"):
        if (not isinstance(plan[key], str) or len(plan[key]) != 64
                or any(c not in "0123456789abcdef" for c in plan[key])):
            raise ValueError(f"invalid pinned {key}")


def make_schedule(patient_ids, shapes, steps, batch_size, seed):
    """Original NumPy patient/TZ shuffle, stopped after exactly N updates."""
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError("fixed steps must be within 1..128")
    rng = np.random.default_rng(seed)
    schedule = []
    while len(schedule) < steps:
        for pid in rng.permutation(patient_ids):
            indices = np.array(list(np.ndindex(shapes[str(pid)])), dtype=np.int64)
            if not len(indices):
                raise ValueError("empty patient TZ grid")
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                schedule.append(dict(patient_id=str(pid), indices_tz=indices[start:start+batch_size].tolist()))
                if len(schedule) == steps:
                    return schedule


def check_excluded_mass(weights, valid):
    weights = weights.detach().float()
    valid = valid.detach().bool()
    total = float(weights.double().sum())
    excluded = float(weights[~valid].double().sum())
    return _check_mass_values(total, excluded)


def _check_mass_values(total, excluded):
    objective_fraction = excluded / max(total, 1e-8)
    if objective_fraction > EXCLUDED_WEIGHT_RTOL:
        raise ValueError("substantial original weight lies outside valid h support")
    return dict(original_weight_sum=total, excluded_weight_mass=excluded,
                excluded_weight_fraction=excluded / total if total else 0.0,
                excluded_objective_weight_fraction=objective_fraction,
                excluded_weight_rtol=EXCLUDED_WEIGHT_RTOL)


def semantic_terms(logits, q, z, weights, valid, semantic_enabled=True):
    """Exact original denominator; index valid pixels BEFORE consuming Z.

    CE, H and forward KL are sums over all classes and shared valid pixels,
    weighted by the original detached reliability * Q-argmax permission,
    divided by the original full weight sum (including excluded pixels).
    Batch activation is determined by original Q before the common mask.
    Inactive CE/H/KL are zero; target mass/contrast remain descriptive.
    """
    q, z, w, valid = q.detach().float(), z.detach().float(), weights.detach().float(), valid.detach().bool()
    if q.ndim != 4 or q.shape[1] != 4 or z.shape != q.shape or w.shape != q.shape[:1]+q.shape[2:] or valid.shape != w.shape:
        raise ValueError("mismatched paired guidance shapes")
    if (not torch.isfinite(q).all() or not torch.isfinite(w).all() or (q < 0).any()
            or (w < 0).any() or (w > 1).any()
            or not torch.allclose(q.sum(1), torch.ones_like(w), atol=2e-4, rtol=0)):
        raise ValueError("invalid original guidance")
    mass = check_excluded_mass(w, valid)
    target_z = z.permute(0, 2, 3, 1)[valid]
    if (not torch.isfinite(target_z).all() or (target_z < 0).any()
            or not torch.allclose(target_z.sum(-1), torch.ones_like(w[valid]), atol=2e-4, rtol=0)):
        raise ValueError("Z must be finite probability vectors on valid support")
    lp = F.interpolate(logits.float(), q.shape[-2:], mode="bilinear", align_corners=False).log_softmax(1)
    active = bool(semantic_enabled and ((q.argmax(1) != 0) & (w > 0)).any())
    denominator = w.sum().clamp_min(1e-8)
    zero = lp.sum() * 0
    results = {}
    for name, target, mask in (("full_Q", q, torch.ones_like(valid)), ("Q", q, valid), ("Z", z, valid)):
        target = target.permute(0, 2, 3, 1)[mask]
        logp = lp.permute(0, 2, 3, 1)[mask]
        weight = w[mask, None]
        ce_class = -(weight * target * logp).sum(0) / denominator
        entropy = -(weight * torch.special.xlogy(target, target)).sum() / denominator
        kl = (weight * (torch.special.xlogy(target, target) - target * logp)).sum() / denominator
        original_winner = q.permute(0, 2, 3, 1)[mask].argmax(-1)
        cell_ce = -(target * logp).sum(-1)
        cell_h = -torch.special.xlogy(target, target).sum(-1)
        strata = {}
        for cls, label in enumerate(CLASSES):
            members = original_winner == cls
            class_weights = weight[:, 0][members]
            raw_weight = class_weights.sum()
            class_denominator = raw_weight.clamp_min(1e-8)
            numerators = {"ce": (class_weights * cell_ce[members]).sum(),
                          "entropy": (class_weights * cell_h[members]).sum(),
                          "kl": (class_weights * (cell_ce - cell_h)[members]).sum()}
            strata[label] = dict(pixels=int(members.sum()), weight_sum=raw_weight,
                                 weighted_denominator=class_denominator,
                                 status="MEASURED_TEACHER_FIDELITY_ONLY" if raw_weight > 0 else "NO_WEIGHTED_Q_ARGMAX_SUPPORT",
                                 **{key: (value/class_denominator if active else zero) if raw_weight > 0 else None
                                    for key, value in numerators.items()},
                                 contribution_original_denominator={key: value/denominator if active else zero
                                                                    for key, value in numerators.items()})
        results[name] = dict(ce=ce_class.sum() if active else zero,
                             entropy=entropy if active else zero, kl=kl if active else zero,
                             target_entropy_before_activation=entropy,
                             per_class_ce=ce_class if active else ce_class * 0,
                             per_class_target_mass=(weight * target).sum(0),
                             RV_above_uniform_positive_mass=(weight[:, 0] * (target[:, 1] - .25).clamp_min(0)).sum(),
                             positive_RV_contrast_mass=(weight[:, 0] * (target[:, 1] - target[:, [0, 2, 3]].max(-1).values).clamp_min(0)).sum(),
                             positive_RV_contrast_definition="sum_w_max_target_RV_minus_max_other_classes_0",
                             RV_above_uniform_definition="sum_w_max_target_RV_minus_uniform_0.25_0",
                             strata_by_original_Q_argmax=strata,
                             strata_definition="original_Q_winner; conditional_class_weight_mean_including_original_batch_activation")
    return dict(active=active, denominator=denominator, support_pixels=int(valid.sum()),
                positive_weight_pixels=int((w > 0).sum()), **mass, **results)


def scalar(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: scalar(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [scalar(item) for item in value]
    return value


def prediction_diagnostics(logits, q, z, valid, ck):
    """TRAIN prediction/teacher fidelity only; no reference anatomy or export."""
    coarse = F.interpolate(logits.float(), q.shape[-2:], mode="bilinear", align_corners=False).softmax(1)
    result = {}
    # Nearest transport is explicitly a teacher-grid attribution diagnostic.
    z_argmax = torch.full_like(valid, 255, dtype=torch.long)
    z_argmax[valid] = z.permute(0, 2, 3, 1)[valid].argmax(-1)
    for name, probability in (("common_grid", coarse), ("native_grid", logits.float().softmax(1))):
        size = probability.shape[-2:]
        support = F.interpolate(valid[:, None].float(), size, mode="nearest")[:, 0].bool()
        q_label = F.interpolate(q.argmax(1)[:, None].float(), size, mode="nearest")[:, 0].long()
        z_label = F.interpolate(z_argmax[:, None].float(), size, mode="nearest")[:, 0].long()
        winner = probability.argmax(1)
        top = probability.topk(2, 1).values
        pass_probability = top[:, 0] >= ck["config"]["min_probability"]
        pass_margin = top[:, 0] - top[:, 1] >= ck["config"]["min_margin"]
        pass_permission = torch.tensor(ck["allowed_classes"], device=winner.device)[winner]
        pass_status = ck["status"] != "REPRESENTATION_ONLY"
        accepted = pass_probability & pass_margin & pass_permission & pass_status
        rows = []
        for cls, label in enumerate(CLASSES):
            predicted = winner == cls
            rows.append(dict(class_name=label, argmax=int(predicted.sum()),
                             common_support_argmax=int((predicted & support).sum()),
                             accepted=int((predicted & accepted).sum()),
                             vetoed=int((predicted & ~accepted).sum()),
                             probability_veto=int((predicted & ~pass_probability).sum()),
                             margin_veto=int((predicted & ~pass_margin).sum()),
                             permission_veto=int((predicted & ~pass_permission).sum()),
                             status_veto=int(predicted.sum()) if not pass_status else 0,
                             common_Q_argmax=int(((q_label == cls) & support).sum()),
                             common_Z_argmax=int(((z_label == cls) & support).sum()),
                             common_Q_agreement=int((predicted & (q_label == cls) & support).sum()),
                             common_Z_agreement=int((predicted & (z_label == cls) & support).sum())))
        result[name] = dict(pixels=winner.numel(), support_pixels=int(support.sum()), classes=rows,
                            Q_fidelity=float((winner[support] == q_label[support]).float().mean()) if support.any() else None,
                            Z_fidelity=float((winner[support] == z_label[support]).float().mean()) if support.any() else None)
    result["interpretation"] = "TRAIN_teacher_argmax_fidelity_not_anatomy; native_reference_is_nearest_bank_grid"
    return result


def gradient_norms(model, feature, semantic):
    """Fixed-batch diagnostic, no .grad accumulation and no optimizer update."""
    params = [p for p in model.parameters() if p.requires_grad]
    encoder_ids = {id(p) for block in (model.enc0, model.enc1, model.enc2) for p in block.parameters()}
    result = dict(weights_updated=False, scope="before_clipping")
    vectors = []
    for label, loss in (("feature", feature), ("semantic", semantic), ("total", feature + semantic)):
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        all_sq = sum(float(g.detach().double().square().sum()) for g in grads if g is not None)
        enc = torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                         for p, g in zip(params, grads) if id(p) in encoder_ids])
        result[label + "_all_parameter_norm"] = all_sq ** .5
        result[label + "_encoder_norm"] = float(enc.norm())
        vectors.append(enc)
    a, b = vectors[:2]
    result["feature_semantic_encoder_cosine"] = float(a.dot(b)/(a.norm()*b.norm())) if a.norm() > 0 and b.norm() > 0 else None
    return result


def _dependencies():
    from .runtime import load_runtime
    from .prepare import load_signal_bank, load_signal_patient
    return load_runtime(), load_signal_bank, load_signal_patient


def fresh_destination(path, bank, signals, checkpoint):
    """Apply the sibling's frozen-ancestor guard to plans and pilot outputs."""
    from .prepare import guard_fresh_output
    if Path(path).exists() or Path(path).is_symlink():
        raise FileExistsError("pilot destination must be FRESH")
    return guard_fresh_output(path, bank, signals, Path(checkpoint).resolve().parent.parent)


def validate_inputs(bank_path, signal_path, checkpoint_path, expected_sha, plan, rt, load_signals):
    """Reject identities before constructing an optimizer or reading patient arrays."""
    io = rt.io
    bank = io.verify(bank_path, "guidance_bank")
    validate_plan(plan, bank)
    if checkpoint_path.name != "last.pt":
        raise ValueError("original student/last.pt checkpoint required")
    if io.digest(checkpoint_path) != expected_sha or plan["checkpoint_sha256"] != expected_sha:
        raise ValueError("wrong original checkpoint hash")
    student = io.verify(checkpoint_path.parent, "student")
    recipe_path = checkpoint_path.parent.parent / "RECIPE.json"
    recipe = io.read_json(recipe_path)
    if (io.digest(bank_path / "FROZEN.json") != plan["bank_manifest_sha256"]
            or io.digest(recipe_path) != plan["recipe_sha256"]):
        raise ValueError("bank or original recipe hash mismatch")
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state_hash(ck["model"]) != plan["model_state_sha256"]:
        raise ValueError("original model state hash mismatch")
    counts = [sum(bank["patient_reports"][p]["class_pixels"][c] > 0 for p in bank["patients"]) for c in range(4)]
    if (ck["locked_splits"] != bank["locked_splits"] or ck["train_patients"] != bank["patients"]
            or ck["config"] != bank["config"] or recipe["config"] != bank["config"]
            or ck["class_patients"] != counts or ck["allowed_classes"] != [n >= 2 for n in counts]
            or ck["bank_manifest_sha256"] != plan["bank_manifest_sha256"]
            or student["bank_manifest_sha256"] != plan["bank_manifest_sha256"]
            or ck["status"] != student["status"]):
        raise ValueError("original checkpoint/bank/config/permission binding mismatch")
    if (bank["source_sha256"] != plan["source_sha256"] or student["source_sha256"] != plan["source_sha256"]
            or recipe["source_sha256"] != plan["source_sha256"]
            or rt.source_sha256 != plan["source_sha256"]):
        raise ValueError("original source hash mismatch")
    if (recipe["width"] != ck["width"] or recipe["batch_size"] != plan["batch_size"]
            or ck["feature_dim"] != bank["config"]["feature_dim"]
            or not np.isfinite(recipe["lr"]) or recipe["lr"] <= 0
            or plan["policy"] != original_policy(ck, recipe)):
        raise ValueError("original architecture/batch size/lr/policy mismatch")
    signals = load_signals(signal_path, bank_path)
    if io.digest(signal_path / "FROZEN.json") != plan["signal_manifest_sha256"]:
        raise ValueError("diagnostic signal bank hash mismatch")
    if not set(plan["patient_ids"]) <= set(signals["patients"]):
        raise ValueError("selected patient missing from validated diagnostic bank")
    if (signals["recipe_sha256"] != plan["recipe_sha256"]
            or signals["source_sha256"] != plan["source_sha256"]
            or signals["config"] != bank["config"] or signals["teacher"] != bank["teacher"]):
        raise ValueError("diagnostic signal recipe/source/config/teacher binding mismatch")
    return bank, signals, recipe, ck


def create_plan(bank, signals, checkpoint, expected_checkpoint_sha, patient_ids, steps=64, seed=42):
    """Read/validate original artifacts and create pins without any optimizer."""
    bank, signals, checkpoint = [Path(p).resolve() for p in (bank, signals, checkpoint)]
    rt, load_signals, _ = _dependencies()
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError("fixed steps must be within 1..128")
    original = rt.io.verify(bank, "guidance_bank")
    if rt.io.digest(checkpoint) != expected_checkpoint_sha:
        raise ValueError("wrong original checkpoint hash")
    rt.io.verify(checkpoint.parent, "student")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    recipe_path = checkpoint.parent.parent / "RECIPE.json"
    recipe = rt.io.read_json(recipe_path)
    plan = dict(schema=PLAN_SCHEMA, patient_ids=patient_ids, steps=steps, seed=seed,
                batch_size=recipe["batch_size"], checkpoint_sha256=expected_checkpoint_sha,
                model_state_sha256=state_hash(ck["model"]), source_sha256=original["source_sha256"],
                bank_manifest_sha256=rt.io.digest(bank/"FROZEN.json"),
                signal_manifest_sha256=rt.io.digest(signals/"FROZEN.json"),
                recipe_sha256=rt.io.digest(recipe_path), policy=original_policy(ck, recipe))
    validate_inputs(bank, signals, checkpoint, expected_checkpoint_sha, plan, rt, load_signals)
    return plan


def configure_determinism(seed):
    """Set before any CUDA work; unsupported deterministic operations fail."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG) != CUBLAS_WORKSPACE_CONFIG:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG differs from pinned deterministic policy")
    if torch.cuda.is_initialized() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        raise RuntimeError("CUDA was initialized before the deterministic workspace policy")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    return dict(torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                scope="deterministic_policy_not_cross_device_bitwise_equivalence")


def run_pilot(bank, signals, checkpoint, expected_checkpoint_sha, plan, output, device="cpu", *, expected_plan_sha):
    """Run only when called explicitly; devices are cpu or explicitly cuda."""
    started = time.perf_counter()
    if json_hash(plan) != expected_plan_sha:
        raise ValueError("plan SHA mismatch: use the independently recorded canonical JSON hash")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or explicit cuda")
    fresh_destination(output, bank, signals, checkpoint)
    bank, signals, checkpoint, output = [Path(p).resolve() for p in (bank, signals, checkpoint, output)]
    rt, load_signals, load_patient = _dependencies()
    original, signal_manifest, recipe, ck = validate_inputs(bank, signals, checkpoint, expected_checkpoint_sha, plan, rt, load_signals)
    determinism = configure_determinism(plan["seed"])
    d = torch.device(device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("explicit CUDA requested but unavailable")
    torch.set_num_threads(recipe.get("threads", 4))
    torch.manual_seed(plan["seed"])
    shapes, mass_receipts, frame_mass = {}, {}, {}
    allowed = np.asarray(ck["allowed_classes"], dtype=bool)
    for pid in plan["patient_ids"]:
        data = rt.io.load_bank_patient(bank, pid)
        signal = load_patient(signals, pid)
        if signal["valid"].shape != data["q"].shape[:-1] or signal["Z"].shape != data["q"].shape:
            raise ValueError("signal/original grid mismatch")
        weights = data["reliability"] * allowed[data["q"].argmax(-1)]
        mass_receipts[pid] = check_excluded_mass(torch.from_numpy(weights), torch.from_numpy(signal["valid"]))
        frame_mass[pid] = (weights.sum(axis=(-1, -2), dtype=np.float64),
                           np.where(signal["valid"], 0, weights).sum(axis=(-1, -2), dtype=np.float64))
        shapes[pid] = data["image"].shape[:2]
    schedule = make_schedule(plan["patient_ids"], shapes, plan["steps"], plan["batch_size"], plan["seed"])
    schedule_sha = json_hash(schedule)
    # Do not discover a bad late batch after earlier updates have already run.
    schedule_mass = []
    for item in schedule:
        ti, zi = np.asarray(item["indices_tz"]).T
        total, excluded = frame_mass[item["patient_id"]]
        schedule_mass.append(_check_mass_values(float(total[ti, zi].sum()), float(excluded[ti, zi].sum())))
    # This is the actual fixed TRAIN batch used for no-update probes in BOTH arms.
    fixed = schedule[0]
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    rt.io.write_json(output / "PLAN.json", plan)
    rt.io.write_json(output / "SCHEDULE.json", schedule)
    arms = {}
    semantic_enabled = any(ck["allowed_classes"][1:])
    for arm in ("Q", "Z"):
        torch.manual_seed(plan["seed"])
        model = rt.student.GuidanceStudent(ck["feature_dim"], ck["width"]).to(d)
        model.load_state_dict(copy.deepcopy(ck["model"]), strict=True)
        model.train()
        initial_hash = state_hash(model.state_dict())
        if initial_hash != plan["model_state_sha256"]:
            raise ValueError("arm initialization changed checkpoint weights")
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=1e-4,
                                      betas=(.9, .999), eps=1e-8)
        if optimizer.state:
            raise ValueError("optimizer must start fresh")
        cache_pid = None

        def batch(item):
            nonlocal cache_pid, data, signal
            pid = item["patient_id"]
            if cache_pid != pid:
                data = rt.io.load_bank_patient(bank, pid)
                signal = load_patient(signals, pid)
                cache_pid = pid
            indices = np.asarray(item["indices_tz"])
            ti, zi = indices.T
            tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(d)
            x = tensor(rt.student.contexts(data["image"], indices))
            teacher = tensor(data["features"][ti, zi].astype(np.float32)).permute(0, 3, 1, 2)
            q = tensor(data["q"][ti, zi]).permute(0, 3, 1, 2)
            z = tensor(signal["Z"][ti, zi]).permute(0, 3, 1, 2)
            valid = tensor(signal["valid"][ti, zi]).bool()
            weight = tensor(data["reliability"][ti, zi]) * torch.tensor(ck["allowed_classes"], device=d)[q.argmax(1)]
            return x, teacher, q, z, weight, valid

        def forward(item, with_gradients=False):
            x, teacher, q, z, weight, valid = batch(item)
            out = model(x)
            feature, full_q, active = rt.student.guidance_loss(out, teacher, q, weight, semantic_enabled)
            terms = semantic_terms(out["logits"], q, z, weight, valid, semantic_enabled)
            if active != terms["active"] or not torch.allclose(full_q, terms["full_Q"]["ce"], atol=1e-6, rtol=0):
                raise ValueError("unmodified original Q activation/loss reproduction failed")
            objective = terms[arm]["ce"]
            record = dict(**item, active=active, feature_loss=float(feature.detach()), semantic=scalar(terms),
                          original_full_Q_total_loss=float((feature + full_q).detach()),
                          original_weight_sha256=state_hash({"weight": weight}),
                          common_support_sha256=state_hash({"valid": valid}),
                          context_sha256=state_hash({"x": x}),
                          frozen_features_sha256=state_hash({"teacher": teacher}))
            with torch.no_grad():
                record["prediction_diagnostics"] = prediction_diagnostics(out["logits"], q, z, valid, ck)
            if with_gradients:
                record["gradient_norms"] = gradient_norms(model, feature, objective)
            return feature + objective, record

        probe_hash = state_hash(model.state_dict())
        probe_loss, before = forward(fixed, with_gradients=True)
        del probe_loss
        if state_hash(model.state_dict()) != probe_hash or any(p.grad is not None for p in model.parameters()):
            raise ValueError("fixed-batch probe changed weights or accumulated gradients")
        arm_started = time.perf_counter()
        if d.type == "cuda":
            torch.cuda.synchronize(d); torch.cuda.reset_peak_memory_stats(d)
        rows = []
        for number, item in enumerate(schedule, 1):
            loss, record = forward(item)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite paired objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not record["active"]:
                for parameter in model.semantic_parameters():
                    parameter.grad = None
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            record.update(optimizer_step=number, preclip_total_gradient_norm=float(norm))
            rows.append(record)
        optimizer.zero_grad(set_to_none=True)
        final_hash = state_hash(model.state_dict())
        probe_loss, after = forward(fixed, with_gradients=True)
        del probe_loss
        if state_hash(model.state_dict()) != final_hash or any(p.grad is not None for p in model.parameters()):
            raise ValueError("final fixed-batch probe changed weights or gradients")
        if d.type == "cuda":
            torch.cuda.synchronize(d)
        actual_schedule_sha = json_hash([{key: row[key] for key in ("patient_id", "indices_tz")} for row in rows])
        if actual_schedule_sha != schedule_sha:
            raise ValueError("executed ordered patient/TZ schedule hash mismatch")
        arm_out = output / arm
        arm_out.mkdir(mode=0o700)
        payload = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                       status="PRIVATE_TECHNICAL_PILOT_NOT_EXPORTABLE_NOT_ANATOMY_VALIDATED",
                       arm=arm, original_checkpoint_sha256=expected_checkpoint_sha,
                       initial_model_state_sha256=initial_hash, final_model_state_sha256=final_hash,
                       ordered_patient_tz_schedule_sha256=actual_schedule_sha, optimizer_steps=len(rows),
                       plan_sha256=json_hash(plan), policy=plan["policy"])
        torch.save(payload, arm_out / "private_last.pt")
        (arm_out / "private_last.pt").chmod(0o600)
        summary = dict(arm=arm, initial_model_state_sha256=initial_hash, final_model_state_sha256=final_hash,
                       ordered_patient_tz_schedule_sha256=actual_schedule_sha, optimizer_steps=len(rows),
                       fresh_optimizer_initial_state_entries=0, optimizer="fresh_AdamW_not_resume",
                       active_fraction=sum(row["active"] for row in rows)/len(rows),
                       reduction="equal_batch_mean_including_original_Q_inactive_zeros",
                       mean_semantic={key: sum(row["semantic"][arm][key] for row in rows)/len(rows)
                                      for key in ("ce", "entropy", "kl")},
                       elapsed_seconds=time.perf_counter()-arm_started,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(d) if d.type == "cuda" else None,
                       peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(d) if d.type == "cuda" else None,
                       fixed_train_batch_no_updates=dict(before=before, after=after), steps=rows)
        rt.io.write_json(arm_out / "TRACE.json", summary)
        rt.io.freeze(arm_out, dict(role="private_technical_pilot", arm=arm,
                                  original_checkpoint_sha256=expected_checkpoint_sha, plan_sha256=json_hash(plan)))
        arms[arm] = summary
        del loss, optimizer, model, forward, batch
    invariants = ("patient_id", "indices_tz", "active", "original_weight_sha256", "common_support_sha256",
                  "context_sha256", "frozen_features_sha256")
    for qrow, zrow in zip(arms["Q"]["steps"], arms["Z"]["steps"]):
        if any(qrow[key] != zrow[key] for key in invariants):
            raise ValueError("paired schedule/weights/activation/context/features mismatch")
    # Verify originals and diagnostic reproduction again before certifying success.
    validate_inputs(bank, signals, checkpoint, expected_checkpoint_sha, plan, rt, load_signals)
    report = dict(status="TECHNICAL_PAIRED_TRAIN_PILOT_ONLY", anatomical_claim=False,
                  reference_masks_read=False, test_arrays_read=False, export_or_evaluation_run=False,
                  original_optimizer_resumed=False, compared="matched_common_support_Q_vs_Z",
                  untouched_full_Q_loss="logged_separately_in_each_step_and_fixed_batch_probe",
                  initial_model_state_sha256=plan["model_state_sha256"], ordered_patient_tz_schedule_sha256=schedule_sha,
                  steps_per_arm=plan["steps"], selected_train_patients=plan["patient_ids"],
                  full_Q_control_limit="common-support Q control differs from published full-Q baseline on excluded support",
                  excluded_weight=mass_receipts, scheduled_batch_excluded_weight_preflight=schedule_mass, device=device,
                  determinism=determinism,
                  elapsed_seconds=time.perf_counter()-started,
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                  process_peak_rss_scope="whole_process_high_water_not_arm_increment",
                  arm_manifests={arm: rt.io.digest(output/arm/"FROZEN.json") for arm in arms},
                  pilot_source_sha256=rt.io.digest(Path(__file__)),
                  executed_source_sha256=rt.source_sha256,
                  paired_invariants_checked=list(invariants))
    rt.io.write_json(output / "REPORT.json", report)
    rt.io.freeze(output, dict(role="paired_technical_pilot", plan_sha256=json_hash(plan),
                             arm_manifests=report["arm_manifests"]))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("bank", "signals", "checkpoint", "plan"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--expected-plan-sha", help="canonical JSON SHA256 emitted by --write-plan-only; mandatory for run")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--write-plan-only", action="store_true",
                        help="validate artifacts and create a FRESH --plan file; no optimizer")
    parser.add_argument("--patient-ids", nargs="+")
    parser.add_argument("--steps", type=int, default=64, help="plan creation only; maximum 128")
    parser.add_argument("--seed", type=int, default=42, help="plan creation only")
    args = parser.parse_args(argv)
    if args.write_plan_only:
        if not args.patient_ids or args.output is not None or args.device != "cpu" or args.expected_plan_sha is not None:
            parser.error("plan creation requires --patient-ids, no --output, and cpu")
        fresh_destination(args.plan, args.bank, args.signals, args.checkpoint)
        plan = create_plan(args.bank, args.signals, args.checkpoint, args.expected_checkpoint_sha,
                           args.patient_ids, args.steps, args.seed)
        with args.plan.open("x") as stream:
            json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps(dict(status="PLAN_WRITTEN_NO_OPTIMIZER", plan=str(args.plan), plan_sha256=json_hash(plan))))
        return 0
    if args.output is None or not args.expected_plan_sha or args.patient_ids is not None or args.steps != 64 or args.seed != 42:
        parser.error("run requires --output, fixed --plan and --expected-plan-sha; selection/budget flags are for --write-plan-only")
    from .prepare import guard_fresh_output
    guard_fresh_output(args.output, args.plan)
    plan = json.loads(args.plan.read_text())
    report = run_pilot(args.bank, args.signals, args.checkpoint, args.expected_checkpoint_sha,
                       plan, args.output, args.device, expected_plan_sha=args.expected_plan_sha)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
