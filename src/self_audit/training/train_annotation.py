"""Phase A: supervised annotation training without the audit decision loop.

**Metric space.**  Every Dice reported by this module is
:data:`~self_audit.audit.semantics.METRIC_SPACE_SLICE_PROXY` -- 2-D per-slice
foreground macro Dice on the resized network grid, averaged over slices.  It
is a *training and monitoring proxy*, not a paper metric: quotable numbers are
per-volume and live in :mod:`self_audit.evaluation.volume_inference`.  Phase A
and Phase C now compute this proxy through the same helper
(:func:`~self_audit.evaluation.metrics.slice_proxy_dice`), so their headline
numbers are directly comparable and neither is batch-size dependent.
"""

from __future__ import annotations

import argparse
import math
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

from tqdm import tqdm

from self_audit.audit.semantics import (
    METRIC_SPACE_SLICE_PROXY,
    resolve_empty_policy,
)
from self_audit.evaluation.metrics import per_class_dice, slice_proxy_dice
from self_audit.losses.annotation import annotation_loss
from self_audit.training._utils import (
    add_wandb_and_tqdm_args,
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
    print_model_parameter_summary,
    resolve_amp,
    resolve_device,
    save_checkpoint,
    seed_everything,
    setup_wandb_logger,
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
    total_epochs: int = 1,
    max_steps: int | None = None,
    disable_tqdm: bool = False,
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
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch + 1:03d}/{total_epochs:03d} [Train A]",
        disable=disable_tqdm,
        leave=False,
    )
    for batch_index, raw_batch in enumerate(pbar):
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
        pbar.set_postfix({
            "loss": f"{float(loss.detach()):.4f}",
            "avg_loss": f"{running / max(count, 1):.4f}",
            "lr": f"{float(optimizer.param_groups[0]['lr']):.2e}",
        })
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


def _accumulate_finite(
    sums: dict[Any, float],
    counts: dict[Any, int],
    key: Any,
    value: float,
) -> bool:
    """Add ``value`` to ``sums[key]`` only when it is finite; return whether it was.

    The canonical empty-class policy is ``"exclude"``, so a per-class Dice may
    legitimately be ``nan``: that class was empty in *both* prediction and
    target on that slice and therefore carries no information.  A plain running
    sum lets a single such slice poison an entire epoch's ``val_dice_class_*``
    and ``val_macro_foreground_dice``, so every key keeps its own count of the
    contributions that were actually finite and the mean divides by that count.
    A key with zero finite contributions must report ``nan`` -- never ``0.0``.
    """

    sums.setdefault(key, 0.0)
    counts.setdefault(key, 0)
    if not math.isfinite(float(value)):
        return False
    sums[key] += float(value)
    counts[key] += 1
    return True


def _finite_mean(total: float, count: int) -> float:
    return float(total) / float(count) if count else float("nan")


@torch.no_grad()
def validate_annotation_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    stage_weights: Iterable[float] | None = None,
    max_batches: int | None = None,
    epoch: int = 0,
    total_epochs: int = 1,
    disable_tqdm: bool = False,
    empty_policy: str | None = None,
) -> dict[str, Any]:
    """Validate Phase A and report the shared slice-proxy Dice.

    **Metric space:** :data:`~self_audit.audit.semantics.METRIC_SPACE_SLICE_PROXY`.
    This is a training/monitoring proxy on the resized network grid, *not* a
    paper metric.

    ``val_macro_foreground_dice`` is the mean over validation slices of
    :func:`~self_audit.evaluation.metrics.slice_proxy_dice` -- the *same*
    helper Phase C uses -- so the two phases' headline numbers are comparable
    and neither depends on how the slices were split into batches.  The
    previous implementation called ``per_class_dice`` on a whole ``[B,H,W]``
    block, which pools confusion counts over the batch (a micro-average) and
    therefore produced a different number for the same predictions at a
    different ``batch_size``.

    ``val_dice_class_{c}`` is the mean over the slices where class ``c`` was
    *not* excluded.  Because different classes are excluded on different
    slices, the macro is deliberately **not** the mean of the per-class means:
    the macro averages per-slice macros, which is the quantity Phase C
    reports.  A class excluded on every validation slice reports ``nan``.
    Each class also carries ``val_dice_class_{c}_slice_count`` so a caller can
    see how much support a value has.
    """

    policy = resolve_empty_policy(empty_policy)
    model.eval()
    total_loss = 0.0
    count = 0
    dice_sums: dict[int, float] = {}
    dice_counts: dict[int, int] = {}
    macro_sums: dict[str, float] = {}
    macro_counts: dict[str, int] = {}
    excluded_slices = 0
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch + 1:03d}/{total_epochs:03d} [Val A]",
        disable=disable_tqdm,
        leave=False,
    )
    for batch_index, raw_batch in enumerate(pbar):
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
        num_classes = int(final_logits.shape[1])
        target = batch["mask"]
        # One shared helper for the headline macro (identical to Phase C) ...
        per_slice_macro = slice_proxy_dice(
            final_logits,
            target,
            num_classes=num_classes,
            empty_policy=policy,
        )
        # ... and a per-sample per-class pass for the class breakdown, which
        # slice_proxy_dice does not expose.  Both go through per_class_dice,
        # so the two are consistent by construction.
        labels = final_logits.argmax(dim=1)
        for index in range(int(labels.shape[0])):
            per_class = per_class_dice(
                labels[index],
                target[index],
                num_classes=num_classes,
                include_background=False,
                empty_policy=policy,
            )
            for cls, value in per_class.items():
                _accumulate_finite(dice_sums, dice_counts, cls, float(value))
            if not _accumulate_finite(macro_sums, macro_counts, "macro", float(per_slice_macro[index])):
                excluded_slices += 1
        batch_size = int(batch["image"].shape[0])
        total_loss += float(loss.detach()) * batch_size
        count += batch_size
        current_macro = _finite_mean(macro_sums.get("macro", 0.0), macro_counts.get("macro", 0))
        pbar.set_postfix({"val_loss": f"{total_loss / max(count, 1):.4f}", "macro_dice": f"{current_macro:.4f}"})
    result: dict[str, Any] = {
        "val_loss": total_loss / max(count, 1),
        "val_metric_space": METRIC_SPACE_SLICE_PROXY,
        "val_empty_class_policy": policy,
        "val_slice_count": float(count),
        "val_excluded_empty_slice_count": float(excluded_slices),
    }
    for cls in sorted(dice_sums):
        result[f"val_dice_class_{cls}"] = _finite_mean(dice_sums[cls], dice_counts.get(cls, 0))
        result[f"val_dice_class_{cls}_slice_count"] = float(dice_counts.get(cls, 0))
    result["val_macro_foreground_dice"] = _finite_mean(
        macro_sums.get("macro", 0.0), macro_counts.get("macro", 0)
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Audit Phase A annotation training")
    parser.add_argument("--config", default="configs/self_audit_annotation.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_manifest", default=None, help="Override split manifest JSON path")
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
    add_wandb_and_tqdm_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.split_manifest is not None:
        config["split_manifest"] = args.split_manifest
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
    print_model_parameter_summary(model, title="Phase A: Annotation Model Parameters")
    epochs = int(args.epochs or config.get("epochs", 100))
    accumulation_steps = validate_accumulation_steps(config.get("gradient_accumulation_steps", 1))
    scheduler_config = dict(config)
    scheduler_config["accumulation_steps"] = accumulation_steps
    scheduler = build_training_scheduler(optimizer, scheduler_config, num_batches=len(train_loader), epochs=epochs)
    amp_enabled, amp_dtype = resolve_amp(config, device)
    scaler = build_grad_scaler(enabled=amp_enabled, device=device, dtype=amp_dtype)
    wandb_logger = setup_wandb_logger(args, config, phase="annotation")
    start_epoch = 0
    best_metric = float("-inf")
    if args.resume:
        payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location=device)
        start_epoch, _, _ = checkpoint_progress(payload)
        best_metric = float(payload.get("best_metric", float("-inf")))
        print(f"resumed={args.resume} epoch={start_epoch} best_metric={best_metric:.5f}")
    stage_weights = config.get("stage_weights")
    output_target = Path(args.output or config.get("output", "weights/self_audit/phase_a_annotation.pt"))
    output_dir = output_target.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
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
                total_epochs=epochs,
                max_steps=args.max_steps,
                disable_tqdm=args.no_tqdm,
            )
            validation = validate_annotation_epoch(
                model,
                val_loader,
                device,
                stage_weights=stage_weights,
                max_batches=args.max_val_batches,
                epoch=epoch,
                total_epochs=epochs,
                disable_tqdm=args.no_tqdm,
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
                save_checkpoint(
                    output_target,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    config=config,
                    extra={"best_metric": best_metric, "phase": "annotation"},
                )
            log_payload = {
                "epoch": epoch + 1,
                "train/loss": stats["loss"],
                "train/lr": stats["lr"],
                "val/loss": validation["val_loss"],
                "val/macro_foreground_dice": metric,
                "val/dice_RV": validation.get("val_dice_class_1", float("nan")),
                "val/dice_MYO": validation.get("val_dice_class_2", float("nan")),
                "val/dice_LV": validation.get("val_dice_class_3", float("nan")),
                "best_macro_dice": best_metric,
            }
            wandb_logger.log(log_payload, step=epoch + 1)
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

        if args.visualize:
            try:
                from self_audit.evaluation.visualizer import plot_phase_a_samples, save_figure
                model.eval()
                sample_images, sample_masks, sample_inits, sample_finals, sample_cids = [], [], [], [], []
                with torch.no_grad():
                    for v_batch in val_loader:
                        v_batch = move_batch(v_batch, device)
                        out = model.forward_annotation(v_batch["image"]) if hasattr(model, "forward_annotation") else model(v_batch["image"])
                        init_log = out.get("initial_logits", out.get("a0_logits", out.get("logits")))
                        fin_log = out.get("logits")
                        sample_images.extend([img.cpu() for img in v_batch["image"]])
                        sample_masks.extend([m.cpu() for m in v_batch["mask"]])
                        sample_inits.extend([l.argmax(dim=0).cpu() for l in init_log])
                        sample_finals.extend([l.argmax(dim=0).cpu() for l in fin_log])
                        sample_cids.extend(v_batch.get("case_id", [f"Case_{len(sample_cids)+1}"]))
                        if len(sample_images) >= max(int(args.vis_samples), 1):
                            break
                if sample_images:
                    fig = plot_phase_a_samples(
                        sample_images[:args.vis_samples],
                        sample_masks[:args.vis_samples],
                        sample_inits[:args.vis_samples],
                        sample_finals[:args.vis_samples],
                        case_ids=sample_cids[:args.vis_samples],
                        title=f"Phase A Validation Samples (Epoch {epoch+1})",
                    )
                    vis_path = Path(args.vis_dir) / "phase_a_val_samples.png"
                    saved_img = save_figure(fig, vis_path)
                    wandb_logger.log_images({"val/phase_a_samples": saved_img}, step=epoch + 1)
                    print(f"visualizations_saved={saved_img}")
            except Exception as vis_err:
                print(f"[Visualizer Warning] Failed to export Phase A visualization: {vis_err}")
    finally:
        wandb_logger.finish()
    print(f"saved_last={output_dir / 'last.pt'} saved_best={output_dir / 'best.pt'} saved_target={output_target}")


if __name__ == "__main__":  # pragma: no cover
    main()
