"""Phase A: supervised annotation training without the audit decision loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.evaluation.metrics import per_class_dice
from self_audit.losses.annotation import annotation_loss
from self_audit.training._utils import (
    autocast_context,
    build_data_loader,
    build_grad_scaler,
    build_model_from_config,
    build_patient_dataset,
    build_training_scheduler,
    checkpoint_progress,
    encoder_head_optimizer,
    extract_annotation_states,
    extract_initial_logits,
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


DEFAULT_STAGE_WEIGHTS = (0.5, 0.7, 0.8, 1.0)


def resolve_stage_weights(
    num_states: int,
    configured: Iterable[float] | None = None,
) -> list[float]:
    """Resolve deterministic Phase-A weights for the A0-to-final trajectory."""

    count = int(num_states)
    if count < 1:
        raise ValueError("num_states must be positive")
    if configured is not None:
        values = [float(value) for value in configured]
        if len(values) == count:
            weights = values
        else:
            weights = list(DEFAULT_STAGE_WEIGHTS[:count])
            if count > len(DEFAULT_STAGE_WEIGHTS):
                weights.extend([1.0] * (count - len(DEFAULT_STAGE_WEIGHTS)))
    else:
        weights = list(DEFAULT_STAGE_WEIGHTS[:count])
        if count > len(DEFAULT_STAGE_WEIGHTS):
            weights.extend([1.0] * (count - len(DEFAULT_STAGE_WEIGHTS)))
    if any(not is_finite(value) or value < 0.0 for value in weights):
        raise ValueError(f"stage weights must be finite and non-negative, got {weights!r}")
    if not any(value > 0.0 for value in weights):
        raise ValueError("at least one stage weight must be positive")
    return weights


def phase_a_loss(
    output: Any,
    target: torch.Tensor,
    intermediate_weight: float | None = None,
    *,
    stage_weights: Iterable[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise A0 and every soft refinement state."""

    initial = extract_initial_logits(output)
    states = extract_annotation_states(output)
    logits = [initial] + [state for state in states if state is not initial]
    if intermediate_weight is not None:
        weights = [1.0] + [float(intermediate_weight) for _ in logits[1:]]
    else:
        weights = resolve_stage_weights(len(logits), stage_weights)
    losses = [annotation_loss(value, target) for value in logits]
    total = sum(weight * value[0] for weight, value in zip(weights, losses)) / max(sum(weights), 1e-8)
    parts = {f"state_{idx}": float(value[0].detach()) for idx, value in enumerate(losses)}
    parts["loss"] = float(total.detach())
    parts["weight_sum"] = float(sum(weights))
    return total, parts


def _finite_loss_or_raise(loss: torch.Tensor, *, epoch: int, step: int, parts: dict[str, Any]) -> None:
    if not is_finite(loss):
        raise FloatingPointError(
            f"Non-finite Phase-A loss at epoch={epoch} step={step}: "
            f"loss={loss.detach().item()!r} components={parts}"
        )


def train_annotation_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip: float | None = 3.0,
    intermediate_weight: float | None = None,
    stage_weights: Iterable[float] | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    gradient_accumulation_steps: int = 1,
    epoch: int = 0,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.train()
    accumulation_steps = validate_accumulation_steps(gradient_accumulation_steps)
    if max_steps is not None and int(max_steps) < 1:
        raise ValueError("max_steps must be positive when provided")
    if scaler is None:
        scaler = build_grad_scaler(enabled=amp_enabled, device=device, dtype=amp_dtype)
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    count = 0
    pending = 0
    optimizer_steps = 0
    batches = 0
    for batch_index, raw_batch in enumerate(loader):
        if max_steps is not None and optimizer_steps >= int(max_steps):
            break
        batch = move_batch(raw_batch, device)
        with autocast_context(enabled=amp_enabled, device=device, dtype=amp_dtype):
            output = model.forward_annotation(batch["image"]) if hasattr(model, "forward_annotation") else model(batch["image"])
            loss, parts = phase_a_loss(output, batch["mask"], intermediate_weight, stage_weights=stage_weights)
        _finite_loss_or_raise(loss, epoch=epoch, step=batch_index, parts=parts)
        scaled_loss = loss / float(accumulation_steps)
        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        pending += 1
        batches += 1
        batch_size = int(batch["image"].shape[0])
        running += float(loss.detach()) * batch_size
        count += batch_size
        if pending == accumulation_steps:
            step_ok, _ = finalize_optimizer_step(
                model,
                optimizer,
                scaler,
                scheduler=scheduler,
                pending_batches=pending,
                accumulation_steps=accumulation_steps,
                grad_clip=grad_clip,
            )
            if not step_ok:
                raise FloatingPointError(f"Non-finite Phase-A gradients at epoch={epoch} step={batch_index} components={parts}")
            optimizer_steps += 1
            pending = 0
    if pending:
        step_ok, _ = finalize_optimizer_step(
            model,
            optimizer,
            scaler,
            scheduler=scheduler,
            pending_batches=pending,
            accumulation_steps=accumulation_steps,
            grad_clip=grad_clip,
        )
        if not step_ok:
            raise FloatingPointError(f"Non-finite Phase-A gradients at epoch={epoch} final_partial_step")
        optimizer_steps += 1
    return {
        "loss": running / max(count, 1),
        "batches": float(batches),
        "optimizer_steps": float(optimizer_steps),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


@torch.no_grad()
def validate_annotation_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    stage_weights: Iterable[float] | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    dice_sums: dict[int, float] = {}
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = move_batch(raw_batch, device)
        output = model.forward_annotation(batch["image"]) if hasattr(model, "forward_annotation") else model(batch["image"])
        loss, _ = phase_a_loss(output, batch["mask"], stage_weights=stage_weights)
        if not is_finite(loss):
            raise FloatingPointError(f"Non-finite Phase-A validation loss: {loss.detach().item()!r}")
        final_logits = output.get("logits") if isinstance(output, dict) else output
        if not torch.is_tensor(final_logits):
            final_logits = extract_initial_logits(output)
        scores = per_class_dice(final_logits.argmax(dim=1), batch["mask"], num_classes=int(final_logits.shape[1]))
        batch_size = int(batch["image"].shape[0])
        total_loss += float(loss.detach()) * batch_size
        count += batch_size
        for cls, value in scores.items():
            dice_sums[cls] = dice_sums.get(cls, 0.0) + float(value) * batch_size
    result = {"val_loss": total_loss / max(count, 1)}
    for cls, value in sorted(dice_sums.items()):
        result[f"val_dice_class_{cls}"] = value / max(count, 1)
    result["val_macro_foreground_dice"] = sum(dice_sums.values()) / max(count * max(len(dice_sums), 1), 1)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Audit Phase A annotation training")
    parser.add_argument("--config", default="configs/self_audit_annotation.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--no_pretrained", action="store_true", help="Disable external ConvNeXt weights for a local smoke run")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Limit validation batches for a short smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.image_size is not None:
        config["image_size"] = args.image_size
    if args.no_pretrained:
        config["model"] = dict(config.get("model", {}))
        config["model"]["pretrained_encoder"] = False
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    device = resolve_device(args.device or config.get("device"))
    seed_everything(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", False)))
    split_stats = validate_dataset_splits(config)
    print(f"split_stats={split_stats}")
    model = build_model_from_config(config, device)
    train_dataset = build_patient_dataset(config, split=str(config.get("train_split", "train")), train=True)
    val_dataset = build_patient_dataset(config, split=str(config.get("val_split", "val")), train=False)
    train_loader = build_data_loader(train_dataset, config, device=device, train=True, batch_size=args.batch_size)
    val_loader = build_data_loader(val_dataset, config, device=device, train=False, batch_size=args.batch_size)
    optimizer = encoder_head_optimizer(
        model,
        lr=float(config.get("lr", 3e-4)),
        encoder_lr=float(config.get("encoder_lr", 3e-5)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(args.epochs or config.get("epochs", 100))
    accumulation_steps = validate_accumulation_steps(config.get("gradient_accumulation_steps", 1))
    scheduler_config = dict(config)
    scheduler_config["accumulation_steps"] = accumulation_steps
    scheduler = build_training_scheduler(optimizer, scheduler_config, num_batches=len(train_loader), epochs=epochs)
    amp_enabled, amp_dtype = resolve_amp(config, device)
    scaler = build_grad_scaler(enabled=amp_enabled, device=device, dtype=amp_dtype)
    start_epoch = 0
    best_metric = float("-inf")
    if args.resume:
        payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location=device)
        start_epoch, _, _ = checkpoint_progress(payload)
        best_metric = float(payload.get("best_metric", float("-inf")))
        print(f"resumed={args.resume} epoch={start_epoch} best_metric={best_metric:.5f}")
    stage_weights = config.get("stage_weights")
    output_dir = Path(args.output or config.get("output", "weights/self_audit/phase_a_annotation.pt")).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        stats = train_annotation_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip=config.get("grad_clip", 3.0),
            stage_weights=stage_weights,
            scheduler=scheduler,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            gradient_accumulation_steps=accumulation_steps,
            epoch=epoch,
            max_steps=args.max_steps,
        )
        validation = validate_annotation_epoch(
            model,
            val_loader,
            device,
            stage_weights=stage_weights,
            max_batches=args.max_val_batches,
        )
        metric = float(validation["val_macro_foreground_dice"])
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            config=config,
            extra={"best_metric": max(best_metric, metric), "phase": "annotation"},
        )
        if metric > best_metric:
            best_metric = metric
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                config=config,
                extra={"best_metric": best_metric, "phase": "annotation"},
            )
        print(
            f"epoch={epoch + 1:03d} lr={stats['lr']:.3e} train_loss={stats['loss']:.5f} "
            f"val_loss={validation['val_loss']:.5f} "
            f"val_dice_RV={validation.get('val_dice_class_1', float('nan')):.4f} "
            f"val_dice_MYO={validation.get('val_dice_class_2', float('nan')):.4f} "
            f"val_dice_LV={validation.get('val_dice_class_3', float('nan')):.4f} "
            f"macro={metric:.4f}"
        )
        if args.max_steps is not None:
            break
    print(f"saved_last={output_dir / 'last.pt'} saved_best={output_dir / 'best.pt'}")


if __name__ == "__main__":  # pragma: no cover
    main()
