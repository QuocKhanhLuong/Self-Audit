"""Baseline recovery helpers: optimizer ownership, module telemetry, and warmup noncollapse gate."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from torch import nn

CANONICAL_MODULES = ("encoder", "decoder", "detail", "fuse", "head")


class WarmupCollapseError(RuntimeError):
    """Raised when the warmup noncollapse gate fails closed on fixed TRAIN-L probe."""
    def __init__(self, message: str, *, evidence: dict | None = None):
        super().__init__(message)
        self.evidence = evidence or {}


def classify_parameter(name: str) -> str:
    """Classify parameter name into its canonical module grouping."""
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith(("lateral.", "up.", "refine.")):
        return "decoder"
    if name.startswith("detail."):
        return "detail"
    if name.startswith("fuse."):
        return "fuse"
    if name.startswith("head."):
        return "head"
    if name.startswith("input_adapter."):
        return "input_adapter"
    if name.startswith("window."):
        return "window"
    if name.startswith("teacher_memory."):
        return "teacher_memory"
    return "other"


def verify_optimizer_ownership(model: nn.Module, optimizers: list[torch.optim.Optimizer]) -> dict:
    """Verify exact parameter ownership between AdamW (body) and SGD (head).

    Ensures:
    1. Exactly 2 optimizers: Optimizer 0 is AdamW (body), Optimizer 1 is SGD (head).
    2. Detects duplicate parameters within param groups and across optimizers.
    3. Every trainable parameter in `model` is owned by exactly one optimizer.
    4. Optimizer 0 (AdamW) owns encoder, decoder, detail, fuse, input_adapter.
    5. Optimizer 1 (SGD) owns head only.
    6. Exact names, numels, LR, weight decay recorded.
    7. All 5 canonical modules have nonzero parameter counts.
    """
    if len(optimizers) != 2:
        raise ValueError(f"Exact 2 optimizers (body AdamW, head SGD) required, got {len(optimizers)}")

    if not isinstance(optimizers[0], torch.optim.AdamW):
        raise ValueError(f"Optimizer 0 must be AdamW, got {type(optimizers[0]).__name__}")
    if not isinstance(optimizers[1], torch.optim.SGD):
        raise ValueError(f"Optimizer 1 must be SGD, got {type(optimizers[1]).__name__}")

    trainable_params = {p: n for n, p in model.named_parameters() if p.requires_grad}

    # Require ALL canonical module parameters to be trainable (no partial or full freezing)
    frozen_canonical = []
    for n, p in model.named_parameters():
        mod = classify_parameter(n)
        if mod in CANONICAL_MODULES and not p.requires_grad:
            frozen_canonical.append(n)
    if frozen_canonical:
        raise ValueError(
            f"Require ALL canonical module parameters to be trainable and owned; "
            f"found frozen/partially-frozen canonical parameters: {sorted(frozen_canonical)}"
        )

    # Detect duplicates within each optimizer and across optimizers
    seen_opt0: set[int] = set()
    for g_idx, group in enumerate(optimizers[0].param_groups):
        for p in group["params"]:
            pid = id(p)
            if pid in seen_opt0:
                pname = trainable_params.get(p, "unknown")
                raise ValueError(f"Duplicate parameter detected in AdamW: {pname}")
            seen_opt0.add(pid)

    seen_opt1: set[int] = set()
    for g_idx, group in enumerate(optimizers[1].param_groups):
        for p in group["params"]:
            pid = id(p)
            if pid in seen_opt1:
                pname = trainable_params.get(p, "unknown")
                raise ValueError(f"Duplicate parameter detected in SGD: {pname}")
            if pid in seen_opt0:
                pname = trainable_params.get(p, "unknown")
                raise ValueError(f"Parameter shared across optimizers (AdamW and SGD): {pname}")
            seen_opt1.add(pid)

    opt0_params = {p for group in optimizers[0].param_groups for p in group["params"]}
    opt1_params = {p for group in optimizers[1].param_groups for p in group["params"]}
    all_opt_params = opt0_params | opt1_params

    # Check completeness
    missing = set(trainable_params.keys()) - all_opt_params
    if missing:
        missing_names = sorted(trainable_params[p] for p in missing)
        raise ValueError(f"Trainable parameters missing from optimizers: {missing_names}")

    extra = all_opt_params - set(trainable_params.keys())
    if extra:
        raise ValueError(f"Optimizers contain {len(extra)} non-trainable or foreign parameters")

    module_counts = {m: 0 for m in CANONICAL_MODULES}
    module_numel = {m: 0 for m in CANONICAL_MODULES}
    module_names: dict[str, list[str]] = {m: [] for m in CANONICAL_MODULES}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        mod = classify_parameter(n)
        if mod in module_counts:
            module_counts[mod] += 1
            module_numel[mod] += p.numel()
            module_names[mod].append(n)

        if mod == "head":
            if p not in opt1_params:
                raise ValueError(f"Head parameter {n} must be owned by SGD (opt 1)")
            if p in opt0_params:
                raise ValueError(f"Head parameter {n} must not be in AdamW (opt 0)")
        elif mod in ("encoder", "decoder", "detail", "fuse", "input_adapter", "window", "teacher_memory"):
            if p not in opt0_params:
                raise ValueError(f"Body parameter {n} must be owned by AdamW (opt 0)")
            if p in opt1_params:
                raise ValueError(f"Body parameter {n} must not be in SGD (opt 1)")

    for m in CANONICAL_MODULES:
        if module_counts[m] == 0:
            raise ValueError(f"Canonical module '{m}' has 0 trainable parameters")

    group_records = []
    for opt_idx, opt in enumerate(optimizers):
        opt_name = "AdamW" if opt_idx == 0 else "SGD"
        for g_idx, group in enumerate(opt.param_groups):
            p_names = [trainable_params[p] for p in group["params"] if p in trainable_params]
            group_records.append({
                "optimizer": opt_name,
                "group_index": g_idx,
                "lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameter_count": len(group["params"]),
                "total_numel": sum(p.numel() for p in group["params"]),
                "parameter_names": sorted(p_names),
            })

    return {
        "verified": True,
        "module_param_counts": module_counts,
        "module_numel": module_numel,
        "module_parameter_names": {m: sorted(names) for m, names in module_names.items()},
        "param_groups": group_records,
        "body_param_count": len(opt0_params),
        "head_param_count": len(opt1_params),
        "total_trainable": len(trainable_params),
    }


def snapshot_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot current model parameter tensors for delta tracking."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def compute_module_telemetry(model: nn.Module, before_snapshot: dict[str, torch.Tensor] | None = None) -> dict:
    """Compute gradient norms and weight update deltas for canonical modules."""
    grad_norms = {}
    deltas = {}

    for mod in CANONICAL_MODULES:
        params = [p for n, p in model.named_parameters() if p.requires_grad and classify_parameter(n) == mod]
        grads = [p.grad.detach() for p in params if p.grad is not None]
        if grads:
            gnorm = float(torch.sqrt(sum(g.square().sum() for g in grads)))
        else:
            gnorm = 0.0
        grad_norms[mod] = gnorm

        if before_snapshot is not None:
            diffs = [(p.detach() - before_snapshot[n]) for n, p in model.named_parameters()
                     if p.requires_grad and classify_parameter(n) == mod and n in before_snapshot]
            if diffs:
                dnorm = float(torch.sqrt(sum(d.square().sum() for d in diffs)))
            else:
                dnorm = 0.0
            deltas[mod] = dnorm

    all_grads_valid = all(np.isfinite(gn) and gn > 0 for gn in grad_norms.values())
    result = {
        "gradient_norms": grad_norms,
        "all_gradients_nonzero": all_grads_valid,
    }
    if before_snapshot is not None:
        all_deltas_valid = all(np.isfinite(dn) and dn > 0 for dn in deltas.values())
        result["update_deltas"] = deltas
        result["all_deltas_nonzero"] = all_deltas_valid

    return result


def build_warmup_probe(data, crop_size: int = 0, seed: int = 42) -> dict:
    """Build a fixed, deterministic probe from budgeted TRAIN-L patients only.

    Retains individual slice samples to support heterogeneous native resolutions.
    """
    from .data import crop_native
    rng = np.random.default_rng(seed)
    probe_samples = []

    for idx, pid in enumerate(data.lids):
        x, y, meta = data.sample_labeled(idx, rng, foreground_probability=1.0)
        x_c, y_c, v_c, crop_info = crop_native(x, y, rng, crop_size)
        meta_entry = dict(meta)
        if crop_info is not None:
            meta_entry["crop"] = crop_info
        probe_samples.append({
            "image": x_c,  # [1, 3, H, W]
            "mask": y_c,   # [1, H, W]
            "valid": v_c,  # [1, H, W]
            "metadata": meta_entry,
            "patient_id": pid,
        })

    return {
        "samples": probe_samples,
        "patient_ids": list(data.lids),
    }


def _validate_min_class_counts(min_class_counts: Any) -> dict[str, int]:
    """Strict validation for min_class_counts: exact keys 1, 2, 3 as positive ints.

    Rejects mixed types and duplicate integer/string alias representations.
    """
    if not isinstance(min_class_counts, dict):
        raise ValueError("min_class_counts must be a dictionary")
    keys = list(min_class_counts.keys())
    if set(keys) == {"1", "2", "3"}:
        pass
    elif set(keys) == {1, 2, 3}:
        pass
    else:
        stringified = [str(k) for k in keys]
        if len(stringified) != len(set(stringified)):
            raise ValueError(f"min_class_counts contains duplicate integer/string alias keys: {keys}")
        raise ValueError(f"min_class_counts keys must be exactly {{'1', '2', '3'}} (or {{1, 2, 3}}), got {keys}")

    norm_counts = {}
    for c_str in ("1", "2", "3"):
        val = min_class_counts[c_str] if c_str in min_class_counts else min_class_counts[int(c_str)]
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise ValueError(f"min_class_counts['{c_str}'] must be a positive integer, got {val}")
        norm_counts[c_str] = int(val)
    return norm_counts


def evaluate_warmup_gate(model: nn.Module, probe: dict | list, device: str = "cpu",
                         min_class_counts: dict[str, int] | None = None) -> dict:
    """Evaluate noncollapse gate on fixed TRAIN-L probe before pseudo training.

    Intersects valid_mask AND label != 255.
    Rejects empty evaluation support and nonfinite logits.
    Restores model mode and RNG states in try/finally even upon failure.
    Streaming execution accommodates heterogeneous native probe shapes.
    Fails closed if any foreground class (RV=1, MYO=2, LV=3) count is below threshold.
    """
    if min_class_counts is None:
        min_class_counts = {"1": 1, "2": 1, "3": 1}
    validated_min_counts = _validate_min_class_counts(min_class_counts)

    submodule_modes = {m: m.training for m in model.modules()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if (device.startswith("cuda") and torch.cuda.is_available()) else None

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_valid_pixels = 0

    if isinstance(probe, list):
        items = probe
        patient_ids = [it.get("patient_id") for it in items if it.get("patient_id") is not None]
    elif isinstance(probe, dict):
        patient_ids = probe.get("patient_ids", [])
        if "samples" in probe:
            items = probe["samples"]
        elif "images" in probe:
            imgs = probe["images"]
            masks = probe.get("masks") if probe.get("masks") is not None else probe.get("labels")
            valids = probe.get("valid")
            items = []
            for b in range(len(imgs)):
                items.append({
                    "image": imgs[b:b+1],
                    "mask": masks[b:b+1] if masks is not None else None,
                    "valid": valids[b:b+1] if valids is not None else None,
                    "patient_id": patient_ids[b] if b < len(patient_ids) else f"p{b}",
                })
        else:
            raise ValueError("Probe dictionary contains neither 'samples' nor 'images'")
    else:
        raise ValueError(f"Invalid probe type: {type(probe)}")

    try:
        model.eval()
        for item in items:
            x = item["image"].to(device)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            v = item.get("valid")
            if v is not None and v.ndim == 2:
                v = v.unsqueeze(0)
            y = item.get("mask") if item.get("mask") is not None else item.get("labels")
            if y is not None and y.ndim == 2:
                y = y.unsqueeze(0)

            v_dev = v.to(device) if v is not None else None

            with torch.no_grad():
                logits = model(x, valid_mask=v_dev)

            if not torch.isfinite(logits).all():
                raise WarmupCollapseError(
                    "Warmup noncollapse gate failed closed: nonfinite logits produced during probe evaluation",
                    evidence={
                        "passed": False,
                        "reason": "nonfinite_logits",
                        "patient_id": item.get("patient_id"),
                        "total_valid_pixels": total_valid_pixels,
                    }
                )

            # Spatial intersection: valid_mask AND label != 255
            b_sz, _, h_sz, w_sz = logits.shape
            eval_mask = torch.ones((b_sz, h_sz, w_sz), dtype=torch.bool, device=logits.device)
            if v is not None:
                eval_mask = eval_mask & v.to(device=logits.device, dtype=torch.bool)
            if y is not None:
                eval_mask = eval_mask & (y.to(device=logits.device) != 255)

            slice_valid_pixels = int(eval_mask.sum().item())
            total_valid_pixels += slice_valid_pixels

            preds = logits.argmax(dim=1)
            valid_preds = preds[eval_mask]
            for c in range(4):
                counts[c] += int((valid_preds == c).sum().item())

    finally:
        for m, mode in submodule_modes.items():
            m.training = mode
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    if total_valid_pixels == 0:
        raise WarmupCollapseError(
            "Warmup noncollapse gate failed closed: evaluation support is completely empty (all pixels are invalid padding or ignore label 255)",
            evidence={
                "passed": False,
                "reason": "empty_support",
                "class_counts": counts,
                "total_valid_pixels": 0,
            }
        )

    failed_classes = {}
    for c_str, min_c in validated_min_counts.items():
        c_int = int(c_str)
        actual = counts.get(c_int, 0)
        if actual < min_c:
            failed_classes[c_str] = {"actual": actual, "min_required": min_c}

    evidence = {
        "passed": len(failed_classes) == 0,
        "class_counts": counts,
        "class_names": {"0": "BG", "1": "RV", "2": "MYO", "3": "LV"},
        "min_class_counts": validated_min_counts,
        "probe_patients": patient_ids,
        "total_valid_probe_pixels": total_valid_pixels,
        "total_probe_pixels": total_valid_pixels,
        "failed_classes": failed_classes,
    }

    if not evidence["passed"]:
        raise WarmupCollapseError(
            f"Warmup noncollapse gate failed closed: classes {failed_classes} did not meet minimum prediction counts in counts {counts}",
            evidence=evidence
        )

    return evidence


ALLOWED_BR_KEYS = {
    "enabled",
    "purpose",
    "supervised_batch_size",
    "unlabeled_batch_size",
    "record_module_telemetry",
    "baseline_certificate_path",
    "baseline_certificate_sha256",
    "warmup_gate",
}

ALLOWED_WG_KEYS = {
    "enabled",
    "min_class_counts",
}


def validate_baseline_recovery_config(cfg: dict | None) -> dict | None:
    """Validate baseline_recovery configuration section."""
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError("baseline_recovery must be a dictionary")

    unknown_keys = set(cfg.keys()) - ALLOWED_BR_KEYS
    if unknown_keys:
        raise ValueError(f"baseline_recovery contains unknown keys: {sorted(unknown_keys)}")

    c = copy.deepcopy(cfg)
    c.setdefault("enabled", True)
    c.setdefault("purpose", "training")
    c.setdefault("supervised_batch_size", 8)
    c.setdefault("unlabeled_batch_size", 8)
    c.setdefault("record_module_telemetry", True)
    c.setdefault("baseline_certificate_path", None)
    c.setdefault("baseline_certificate_sha256", None)
    c.setdefault("warmup_gate", {
        "enabled": True,
        "min_class_counts": {"1": 1, "2": 1, "3": 1},
    })

    if not isinstance(c["enabled"], bool):
        raise ValueError("baseline_recovery.enabled must be a boolean")
    if c["purpose"] not in {"training", "profile"}:
        raise ValueError(f"baseline_recovery.purpose must be 'training' or 'profile', got {c['purpose']}")
    for k in ("supervised_batch_size", "unlabeled_batch_size"):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError(f"baseline_recovery.{k} must be a positive integer")
    if not isinstance(c["record_module_telemetry"], bool):
        raise ValueError("baseline_recovery.record_module_telemetry must be a boolean")

    for k in ("baseline_certificate_path", "baseline_certificate_sha256"):
        val = c[k]
        if val is not None and (not isinstance(val, str) or not val):
            raise ValueError(f"baseline_recovery.{k} must be a non-empty string or null")

    wg = c["warmup_gate"]
    if not isinstance(wg, dict):
        raise ValueError("baseline_recovery.warmup_gate must be a dictionary")
    unknown_wg = set(wg.keys()) - ALLOWED_WG_KEYS
    if unknown_wg:
        raise ValueError(f"baseline_recovery.warmup_gate contains unknown keys: {sorted(unknown_wg)}")
    wg.setdefault("enabled", True)
    wg.setdefault("min_class_counts", {"1": 1, "2": 1, "3": 1})
    if not isinstance(wg["enabled"], bool):
        raise ValueError("baseline_recovery.warmup_gate.enabled must be a boolean")

    wg["min_class_counts"] = _validate_min_class_counts(wg["min_class_counts"])

    return c
