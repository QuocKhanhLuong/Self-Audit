"""Phase B: freeze annotation and train the transition auditor."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.audit.counterfactual import CounterfactualGenerator
from self_audit.audit.targets import build_transition_targets
from self_audit.evaluation.metrics import transition_audit_metrics
from self_audit.losses.audit import audit_loss
from self_audit.training._utils import (
    autocast_context,
    build_adamw_optimizer,
    build_data_loader,
    build_grad_scaler,
    build_model_from_config,
    build_patient_dataset,
    build_training_scheduler,
    checkpoint_progress,
    extract_annotation_trajectory,
    finalize_optimizer_step,
    is_finite,
    load_checkpoint,
    load_config,
    move_batch,
    resolve_amp,
    resolve_device,
    save_checkpoint,
    seed_everything,
    validate_accumulation_steps,
    validate_dataset_splits,
)


def freeze_annotation_network(model: torch.nn.Module) -> None:
    """Freeze all annotation-side parameters while leaving the auditor trainable."""

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("auditor")
    if hasattr(model, "encoder"):
        model.encoder.eval()
    if hasattr(model, "fpn"):
        model.fpn.eval()
    if hasattr(model, "initial_head"):
        model.initial_head.eval()
    if hasattr(model, "annotation_expert"):
        model.annotation_expert.eval()


def _feature_for_audit(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode"):
        encoded = model.encode(images)
        if isinstance(encoded, tuple):
            return encoded[-1]
        if isinstance(encoded, dict):
            return encoded.get("shared", encoded.get("features"))
        return encoded
    return images


def _state_probabilities(state: torch.Tensor) -> torch.Tensor:
    """Convert a state to probabilities without misclassifying arbitrary logits."""

    if bool((state.detach().min() >= 0).item()) and bool((state.detach().max() <= 1).item()):
        sums = state.detach().sum(dim=1, keepdim=True)
        if bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)):
            return state / sums.clamp_min(1e-8)
    return state.softmax(dim=1)


def build_auditor_transitions(
    output: Any,
    ground_truth: torch.Tensor,
    generator: CounterfactualGenerator,
) -> list[dict[str, Any]]:
    """Build adjacent on-policy pairs plus one synthetic pair around every A_t."""

    trajectory = extract_annotation_trajectory(output)
    transitions: list[dict[str, Any]] = []
    for turn, (previous_state, candidate_state) in enumerate(zip(trajectory[:-1], trajectory[1:])):
        transitions.append(
            {
                "previous": _state_probabilities(previous_state).detach(),
                "candidate": _state_probabilities(candidate_state).detach(),
                "turn_index": turn,
                "provenance": "on_policy",
                "valid_mask": torch.ones(previous_state.shape[0], dtype=torch.bool, device=previous_state.device),
            }
        )
    for turn, state in enumerate(trajectory):
        synthetic = generator.generate(_state_probabilities(state).detach(), ground_truth, kind=None)
        valid_mask = synthetic.valid_mask
        if valid_mask is None:
            valid_mask = torch.full(
                (synthetic.previous_probs.shape[0],),
                synthetic.valid,
                dtype=torch.bool,
                device=synthetic.previous_probs.device,
            )
        if bool(valid_mask.any()):
            transitions.append(
                {
                    "previous": synthetic.previous_probs,
                    "candidate": synthetic.candidate_probs,
                    "turn_index": turn,
                    "provenance": f"synthetic:{synthetic.kind}:{synthetic.operation}",
                    "valid_mask": valid_mask,
                    "sample": synthetic,
                }
            )
    return transitions


def local_class_weights(target: torch.Tensor, *, num_classes: int = 3, max_weight: float = 5.0) -> torch.Tensor:
    """Compute bounded inverse-frequency weights without exploding absent classes."""

    values = target.reshape(-1).long()
    counts = torch.bincount(values, minlength=int(num_classes)).float()
    present = counts > 0
    weights = torch.ones(int(num_classes), device=target.device, dtype=torch.float32)
    if bool(present.any()):
        total = counts[present].sum()
        n_present = present.sum().float()
        weights[present] = (total / (n_present * counts[present])).clamp(0.25, float(max_weight))
    return weights


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    values = probs.clamp_min(1e-8)
    return -(values * values.log()).sum(dim=1, keepdim=True)


def _auditor_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    generator: CounterfactualGenerator,
    *,
    neutral_margin: float = 0.005,
    local_weighting: bool = True,
    audit_margin: float = 0.05,
    collect: bool = False,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    timings = {"annotation_forward_ms": 0.0, "counterfactual_ms": 0.0, "auditor_ms": 0.0}
    start = time.perf_counter()
    with torch.no_grad():
        output = model.forward_annotation(batch["image"]) if hasattr(model, "forward_annotation") else model(batch["image"])
        features = _feature_for_audit(model, batch["image"]).detach()
    timings["annotation_forward_ms"] = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    with torch.no_grad():
        transitions = build_auditor_transitions(output, batch["mask"], generator)
    timings["counterfactual_ms"] = (time.perf_counter() - start) * 1000.0
    losses: list[torch.Tensor] = []
    local_predictions: list[torch.Tensor] = []
    local_targets: list[torch.Tensor] = []
    delta_predictions: list[torch.Tensor] = []
    delta_targets: list[torch.Tensor] = []
    local_counts = torch.zeros(3, dtype=torch.long, device=batch["mask"].device)
    start = time.perf_counter()
    for transition in transitions:
        valid_mask = transition["valid_mask"].to(device=batch["image"].device, dtype=torch.bool)
        if not bool(valid_mask.any()):
            continue
        previous = transition["previous"][valid_mask]
        candidate = transition["candidate"][valid_mask]
        target = batch["mask"][valid_mask]
        targets = build_transition_targets(previous, candidate, target)
        weights = local_class_weights(targets.local, num_classes=3) if local_weighting else None
        with torch.no_grad():
            entropy_previous = _entropy(previous)
            entropy_candidate = _entropy(candidate)
        with autocast_context(enabled=amp_enabled, device=batch["image"].device, dtype=amp_dtype):
            audit_output = model.auditor(
                features[valid_mask],
                previous,
                candidate,
                candidate - previous,
                entropy_previous=entropy_previous,
                entropy_candidate=entropy_candidate,
            )
        loss, _ = audit_loss(
            audit_output,
            targets,
            margin=audit_margin,
            neutral_margin=neutral_margin,
            local_class_weights=weights,
        )
        losses.append(loss)
        local_counts += torch.bincount(targets.local.reshape(-1), minlength=3).to(local_counts.device)
        if collect:
            local_predictions.append(audit_output.local_logits.detach())
            local_targets.append(targets.local.detach())
            delta_predictions.append(audit_output.delta_q.detach().reshape(-1))
            delta_targets.append(targets.delta_dice.detach().reshape(-1))
    timings["auditor_ms"] = (time.perf_counter() - start) * 1000.0
    if not losses:
        return None, {"transitions": 0, "timings": timings, "local_counts": local_counts, "transition_data": None}
    loss = torch.stack(losses).mean()
    data = {
        "transitions": len(losses),
        "timings": timings,
        "local_counts": local_counts,
        "transition_data": (
            torch.cat(local_predictions) if local_predictions else None,
            torch.cat(local_targets) if local_targets else None,
            torch.cat(delta_predictions) if delta_predictions else None,
            torch.cat(delta_targets) if delta_targets else None,
        ),
    }
    return loss, data


def train_auditor_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    generator: CounterfactualGenerator | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    gradient_accumulation_steps: int = 1,
    epoch: int = 0,
    max_steps: int | None = None,
    neutral_margin: float = 0.005,
    local_weighting: bool = True,
    audit_margin: float = 0.05,
) -> dict[str, float]:
    if not hasattr(model, "auditor"):
        raise AttributeError("Phase B requires model.auditor")
    model.eval()
    model.auditor.train()
    generator = generator or CounterfactualGenerator()
    accumulation_steps = validate_accumulation_steps(gradient_accumulation_steps)
    if scaler is None:
        scaler = build_grad_scaler(enabled=amp_enabled, device=device, dtype=amp_dtype)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    steps = 0
    batches = 0
    transition_count = 0
    pending = 0
    optimizer_steps = 0
    timing_totals = {"annotation_forward_ms": 0.0, "counterfactual_ms": 0.0, "auditor_ms": 0.0}
    local_counts = torch.zeros(3, dtype=torch.long)
    for batch_index, raw_batch in enumerate(loader):
        if max_steps is not None and optimizer_steps >= int(max_steps):
            break
        batch = move_batch(raw_batch, device)
        loss, details = _auditor_batch(
            model,
            batch,
            generator,
            neutral_margin=neutral_margin,
            local_weighting=local_weighting,
            audit_margin=audit_margin,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        if loss is None:
            continue
        if not is_finite(loss):
            raise FloatingPointError(f"Non-finite Phase-B loss at epoch={epoch} step={batch_index}: {float(loss.detach())!r}")
        scaled = loss / float(accumulation_steps)
        if scaler.is_enabled():
            with autocast_context(enabled=False, device=device, dtype=amp_dtype):
                scaler.scale(scaled).backward()
        else:
            scaled.backward()
        pending += 1
        batches += 1
        steps += 1
        transition_count += int(details["transitions"])
        local_counts += details["local_counts"].cpu()
        for key, value in details["timings"].items():
            timing_totals[key] += float(value)
        total += float(loss.detach())
        if pending == accumulation_steps:
            step_ok, _ = finalize_optimizer_step(
                model,
                optimizer,
                scaler,
                scheduler=scheduler,
                pending_batches=pending,
                accumulation_steps=accumulation_steps,
                grad_clip=3.0,
            )
            if not step_ok:
                raise FloatingPointError(f"Non-finite Phase-B gradients at epoch={epoch} step={batch_index}")
            pending = 0
            optimizer_steps += 1
    if pending:
        step_ok, _ = finalize_optimizer_step(
            model,
            optimizer,
            scaler,
            scheduler=scheduler,
            pending_batches=pending,
            accumulation_steps=accumulation_steps,
            grad_clip=3.0,
        )
        if not step_ok:
            raise FloatingPointError(f"Non-finite Phase-B gradients at epoch={epoch} final_partial_step")
        optimizer_steps += 1
    result = {
        "loss": total / max(steps, 1),
        "transitions": float(transition_count),
        "batches": float(batches),
        "optimizer_steps": float(optimizer_steps),
        "annotation_forward_ms": timing_totals["annotation_forward_ms"] / max(batches, 1),
        "counterfactual_ms": timing_totals["counterfactual_ms"] / max(batches, 1),
        "auditor_ms": timing_totals["auditor_ms"] / max(batches, 1),
        "local_fix_count": float(local_counts[0]),
        "local_unchanged_count": float(local_counts[1]),
        "local_regress_count": float(local_counts[2]),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }
    if result["counterfactual_ms"] > max(result["annotation_forward_ms"], result["auditor_ms"]) * 2.0 and result["counterfactual_ms"] > 100.0:
        result["counterfactual_warning"] = 1.0
        print("WARNING: counterfactual generation dominates Phase-B batch time", file=sys.stderr)
    return result


@torch.no_grad()
def validate_auditor_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    generator: CounterfactualGenerator | None = None,
    neutral_margin: float = 0.005,
    local_weighting: bool = True,
    audit_margin: float = 0.05,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    generator = generator or CounterfactualGenerator()
    losses: list[float] = []
    transition_parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    count = 0
    timings = {"annotation_forward_ms": 0.0, "counterfactual_ms": 0.0, "auditor_ms": 0.0}
    local_counts = torch.zeros(3, dtype=torch.long)
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = move_batch(raw_batch, device)
        loss, details = _auditor_batch(
            model,
            batch,
            generator,
            neutral_margin=neutral_margin,
            local_weighting=local_weighting,
            audit_margin=audit_margin,
            collect=True,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        if loss is None:
            continue
        losses.append(float(loss.detach()))
        count += int(details["transitions"])
        local_counts += details["local_counts"].cpu()
        for key, value in details["timings"].items():
            timings[key] += float(value)
        if details["transition_data"] is not None:
            transition_parts.append(details["transition_data"])
    if transition_parts:
        local_pred = torch.cat([item[0] for item in transition_parts])
        local_target = torch.cat([item[1] for item in transition_parts])
        delta_pred = torch.cat([item[2] for item in transition_parts])
        delta_target = torch.cat([item[3] for item in transition_parts])
        metrics = transition_audit_metrics(local_pred, local_target, delta_pred, delta_target)
    else:
        metrics = {
            "improve_regress_accuracy": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "correlation_delta_q_delta_dice": float("nan"),
            "local_fix_f1": float("nan"),
            "local_regress_f1": float("nan"),
        }
    auroc = float(metrics["auroc"])
    primary = auroc if is_finite(auroc) else float(metrics["improve_regress_accuracy"])
    return {
        "audit_loss": sum(losses) / max(len(losses), 1),
        "loss": sum(losses) / max(len(losses), 1),
        "transitions": float(count),
        "primary_metric": primary,
        "local_fix_count": float(local_counts[0]),
        "local_unchanged_count": float(local_counts[1]),
        "local_regress_count": float(local_counts[2]),
        "annotation_forward_ms": timings["annotation_forward_ms"] / max(len(losses), 1),
        "counterfactual_ms": timings["counterfactual_ms"] / max(len(losses), 1),
        "auditor_ms": timings["auditor_ms"] / max(len(losses), 1),
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Audit Phase B auditor training")
    parser.add_argument("--config", default="configs/self_audit_auditor.yaml")
    parser.add_argument("--annotation_checkpoint", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.image_size is not None:
        config["image_size"] = args.image_size
    device = resolve_device(args.device or config.get("device"))
    seed_everything(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", False)))
    split_stats = validate_dataset_splits(config)
    print(f"split_stats={split_stats}")
    model = build_model_from_config(config, device)
    if args.resume:
        load_checkpoint(args.resume, model=model, map_location=device)
    elif args.annotation_checkpoint:
        load_checkpoint(args.annotation_checkpoint, model=model, map_location=device)
    else:
        raise ValueError("Phase B requires --annotation_checkpoint or --resume")
    freeze_annotation_network(model)
    train_dataset = build_patient_dataset(config, split=str(config.get("train_split", "train")), train=False)
    val_dataset = build_patient_dataset(config, split=str(config.get("val_split", "val")), train=False)
    train_loader = build_data_loader(train_dataset, config, device=device, train=True, batch_size=args.batch_size)
    val_loader = build_data_loader(val_dataset, config, device=device, train=False, batch_size=args.batch_size)
    auditor_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = build_adamw_optimizer(auditor_parameters, lr=float(config.get("auditor_lr", 3e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    epochs = int(args.epochs or config.get("epochs", 20))
    accumulation_steps = validate_accumulation_steps(config.get("gradient_accumulation_steps", 1))
    scheduler_config = dict(config)
    scheduler_config["accumulation_steps"] = accumulation_steps
    scheduler = build_training_scheduler(optimizer, scheduler_config, num_batches=len(train_loader), epochs=epochs)
    amp_enabled, amp_dtype = resolve_amp(config, device)
    scaler = build_grad_scaler(enabled=amp_enabled, device=device, dtype=amp_dtype)
    cf_config = config.get("counterfactual", {})
    if not isinstance(cf_config, dict):
        raise ValueError("counterfactual config must be a mapping")
    generator = CounterfactualGenerator(
        epsilon_neutral=float(cf_config.get("epsilon_neutral", 0.02)),
        neutral_max_retries=int(cf_config.get("neutral_max_retries", 8)),
        num_classes=int(config.get("num_classes", 4)),
    )
    best_metric = float("-inf")
    start_epoch = 0
    if args.resume:
        payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location=device)
        start_epoch, _, _ = checkpoint_progress(payload)
        best_metric = float(payload.get("best_metric", float("-inf")))
    output_dir = Path(args.output or config.get("output", "weights/self_audit/phase_b_auditor.pt")).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        train_stats = train_auditor_epoch(
            model, train_loader, optimizer, device, generator=generator, scheduler=scheduler,
            scaler=scaler, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
            gradient_accumulation_steps=accumulation_steps, epoch=epoch, max_steps=args.max_steps,
            neutral_margin=float(cf_config.get("neutral_margin", 0.005)),
            local_weighting=cf_config.get("local_class_weighting", True) != "none",
        )
        validation = validate_auditor_epoch(
            model, val_loader, device, generator=generator,
            neutral_margin=float(cf_config.get("neutral_margin", 0.005)),
            local_weighting=cf_config.get("local_class_weighting", True) != "none",
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            max_batches=args.max_val_batches,
        )
        metric = float(validation["primary_metric"])
        if not is_finite(metric):
            metric = -float("inf")
        is_best = metric > best_metric
        if is_best:
            best_metric = metric
        save_checkpoint(output_dir / "last.pt", model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch + 1, config=config, extra={"best_metric": best_metric, "phase": "auditor"})
        if is_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch + 1, config=config, extra={"best_metric": metric, "phase": "auditor"})
        print(
            f"epoch={epoch + 1:03d} lr={train_stats['lr']:.3e} loss={train_stats['loss']:.5f} "
            f"val_loss={validation['audit_loss']:.5f} AUROC={validation['auroc']:.4f} "
            f"AUPRC={validation['auprc']:.4f} FIX_F1={validation['local_fix_f1']:.4f} "
            f"REGRESS_F1={validation['local_regress_f1']:.4f} corr={validation['correlation_delta_q_delta_dice']:.4f} "
            f"global_acc={validation['improve_regress_accuracy']:.4f} "
            f"transitions={train_stats['transitions']:.0f} cf_ms={train_stats['counterfactual_ms']:.1f} "
            f"local_counts={int(validation['local_fix_count'])}/{int(validation['local_unchanged_count'])}/{int(validation['local_regress_count'])}"
        )
        if args.max_steps is not None:
            break
    print(f"saved_last={output_dir / 'last.pt'} saved_best={output_dir / 'best.pt'}")


if __name__ == "__main__":  # pragma: no cover
    main()
