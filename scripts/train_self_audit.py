#!/usr/bin/env python3
"""Single-process A -> B -> C Self-Audit training pipeline.

Unlike scripts/run_full_pipeline.sh, this orchestrator keeps one model instance
alive across all three phases. Phase boundaries still use separate optimizers
and objectives, but weights are not reconstructed/reloaded between phases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.audit.counterfactual import CounterfactualGenerator
from self_audit.evaluation.audit_decomposition import (
    evaluate_annotation_headroom,
    evaluate_audit_decomposition,
)
from self_audit.evaluation.threshold import select_threshold, sweep_thresholds
from self_audit.training._utils import (
    WandbLogger,
    build_adamw_optimizer,
    build_data_loader,
    build_grad_scaler,
    build_model_from_config,
    build_patient_dataset,
    build_training_scheduler,
    encoder_head_optimizer,
    load_config,
    print_model_parameter_summary,
    resolve_amp,
    resolve_device,
    save_checkpoint,
    seed_everything,
    validate_accumulation_steps,
    validate_dataset_splits,
)
from self_audit.training.finetune_joint import (
    collect_validation_transition_cache,
    finetune_joint_epoch,
    validate_phase_c,
)
from self_audit.training.train_annotation import (
    train_annotation_epoch,
    validate_annotation_epoch,
)
from self_audit.training.train_auditor import (
    freeze_annotation_network,
    train_auditor_epoch,
    validate_auditor_epoch,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Self-Audit A->B->C in one Python process and one model instance"
    )
    parser.add_argument("--config_a", default="configs/self_audit_annotation.yaml")
    parser.add_argument("--config_b", default="configs/self_audit_auditor.yaml")
    parser.add_argument("--config_c", default="configs/self_audit_joint.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epochs_a", type=int, default=None)
    parser.add_argument("--epochs_b", type=int, default=None)
    parser.add_argument("--epochs_c", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--output_dir", default="weights/self_audit_full")
    parser.add_argument("--report_dir", default="reports/self_audit_full")
    parser.add_argument("--neutral_margin", type=float, default=None)
    parser.add_argument("--tau_accept", type=float, default=None)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--skip_calibration", action="store_true")
    parser.add_argument("--threshold_min", type=float, default=-0.02)
    parser.add_argument("--threshold_max", type=float, default=0.02)
    parser.add_argument("--threshold_steps", type=int, default=81)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_mode", default="offline")
    parser.add_argument("--wandb_project", default="self-audit")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default="full-pipeline")
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args()


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = dict(config)
    if args.data_root is not None:
        result["data_root"] = args.data_root
    if args.split_manifest is not None:
        result["split_manifest"] = args.split_manifest
    if args.num_workers is not None:
        result["num_workers"] = int(args.num_workers)
    if args.device is not None:
        result["device"] = args.device
    return result


def _architecture_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    model = dict(config.get("model", {}))
    return (
        model.get("encoder_name", "convnext_tiny"),
        int(model.get("shared_channels", 96)),
        int(model.get("num_classes", config.get("num_classes", 4))),
        int(model.get("window_k", 8)),
        int(model.get("max_turns", 3)),
    )


def _assert_compatible_configs(*configs: dict[str, Any]) -> None:
    signatures = [_architecture_signature(config) for config in configs]
    if len(set(signatures)) != 1:
        raise ValueError(f"A/B/C model architectures differ: {signatures}")


def _set_trainable(model: torch.nn.Module, *, annotation: bool, auditor: bool) -> None:
    for name, parameter in model.named_parameters():
        is_auditor = name.startswith("auditor")
        parameter.requires_grad = bool(auditor if is_auditor else annotation)


def _make_loaders(
    config: dict[str, Any],
    device: torch.device,
    *,
    train_augment: bool,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dataset = build_patient_dataset(
        config,
        split=str(config.get("train_split", "train")),
        train=train_augment,
    )
    val_dataset = build_patient_dataset(
        config,
        split=str(config.get("val_split", "val")),
        train=False,
    )
    return (
        build_data_loader(train_dataset, config, device=device, train=True),
        build_data_loader(val_dataset, config, device=device, train=False),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _log(logger: WandbLogger, phase: str, epoch: int, metrics: dict[str, Any]) -> None:
    payload = {"pipeline/phase": phase, "pipeline/epoch": int(epoch)}
    payload.update(metrics)
    logger.log(payload)


def _scheduler_and_amp(
    model: torch.nn.Module,
    config: dict[str, Any],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    encoder_lr: float,
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler | None,
    Any,
    bool,
    torch.dtype,
    int,
]:
    optimizer = encoder_head_optimizer(
        model,
        lr=float(lr),
        encoder_lr=float(encoder_lr),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    accumulation = validate_accumulation_steps(config.get("gradient_accumulation_steps", 1))
    scheduler_config = dict(config)
    scheduler_config["accumulation_steps"] = accumulation
    scheduler = build_training_scheduler(
        optimizer,
        scheduler_config,
        num_batches=len(loader),
        epochs=int(epochs),
    )
    amp_enabled, amp_dtype = resolve_amp(config, device)
    scaler = build_grad_scaler(enabled=amp_enabled, device=device, dtype=amp_dtype)
    return optimizer, scheduler, scaler, amp_enabled, amp_dtype, accumulation


def main() -> None:
    args = _parse_args()
    config_a = _apply_overrides(load_config(args.config_a), args)
    config_b = _apply_overrides(load_config(args.config_b), args)
    config_c = _apply_overrides(load_config(args.config_c), args)
    _assert_compatible_configs(config_a, config_b, config_c)

    for name, config in (("A", config_a), ("B", config_b), ("C", config_c)):
        stats = validate_dataset_splits(config)
        print(f"phase_{name.lower()}_split_stats={stats}")

    device = resolve_device(args.device or config_a.get("device"))
    seed_everything(
        int(config_a.get("seed", 42)),
        deterministic=bool(config_a.get("deterministic", False)),
    )

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(config_a, device)
    print_model_parameter_summary(model, title="Full Pipeline: Single Shared Model")

    logger = WandbLogger(
        enabled=bool(args.wandb),
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        config={"phase_a": config_a, "phase_b": config_b, "phase_c": config_c},
        mode=args.wandb_mode,
    )

    report: dict[str, Any] = {
        "protocol": {
            "single_process": True,
            "single_model_instance": True,
            "gt_policy": (
                "GT supervises annotation/auditor targets during training and is used "
                "after inference for validation decomposition; deployable self_audit "
                "decisions receive no GT."
            ),
        },
        "phase_a": [],
        "phase_b": [],
        "phase_c": [],
    }

    print("\n=== PHASE A: annotation bootstrap ===")
    _set_trainable(model, annotation=True, auditor=False)
    train_a, val_a = _make_loaders(config_a, device, train_augment=True)
    epochs_a = int(args.epochs_a or config_a.get("epochs", 100))
    opt_a, sched_a, scaler_a, amp_a, dtype_a, accum_a = _scheduler_and_amp(
        model,
        config_a,
        train_a,
        device,
        epochs=epochs_a,
        lr=float(config_a.get("lr", 3e-4)),
        encoder_lr=float(config_a.get("encoder_lr", 3e-5)),
    )
    best_a = -float("inf")
    stage_weights = config_a.get("stage_weights")
    for epoch in range(epochs_a):
        train_stats = train_annotation_epoch(
            model,
            train_a,
            opt_a,
            device,
            grad_clip=config_a.get("grad_clip", 3.0),
            stage_weights=stage_weights,
            scheduler=sched_a,
            scaler=scaler_a,
            amp_enabled=amp_a,
            amp_dtype=dtype_a,
            gradient_accumulation_steps=accum_a,
            epoch=epoch,
            total_epochs=epochs_a,
            max_steps=args.max_steps,
            disable_tqdm=args.no_tqdm,
        )
        val_stats = validate_annotation_epoch(
            model,
            val_a,
            device,
            stage_weights=stage_weights,
            max_batches=args.max_val_batches,
            epoch=epoch,
            total_epochs=epochs_a,
            disable_tqdm=args.no_tqdm,
        )
        headroom = evaluate_annotation_headroom(
            model,
            val_a,
            device,
            neutral_margin=float(
                args.neutral_margin
                if args.neutral_margin is not None
                else config_b.get("counterfactual", {}).get("neutral_margin", 0.005)
            ),
            max_batches=args.max_val_batches,
            disable_tqdm=args.no_tqdm,
        )
        row = {"epoch": epoch + 1, **train_stats, **val_stats, **headroom}
        report["phase_a"].append(row)
        _log(logger, "A", epoch + 1, {f"phase_a/{k}": v for k, v in train_stats.items()})
        _log(logger, "A", epoch + 1, val_stats)
        _log(logger, "A", epoch + 1, headroom)
        metric = float(val_stats["val_macro_foreground_dice"])
        save_checkpoint(
            output_dir / "phase_a_last.pt",
            model,
            optimizer=opt_a,
            scheduler=sched_a,
            scaler=scaler_a,
            epoch=epoch + 1,
            config=config_a,
            extra={"best_metric": max(best_a, metric), "phase": "annotation"},
        )
        if metric > best_a:
            best_a = metric
            save_checkpoint(
                output_dir / "phase_a_best.pt",
                model,
                optimizer=opt_a,
                scheduler=sched_a,
                scaler=scaler_a,
                epoch=epoch + 1,
                config=config_a,
                extra={"best_metric": best_a, "phase": "annotation"},
            )
        print(
            f"A epoch={epoch+1:03d} val_dice={metric:.4f} "
            f"A0={headroom.get('phase_a/a0_dice', float('nan')):.4f} "
            f"refine_gain={headroom.get('phase_a/total_refinement_gain', float('nan')):+.5f} "
            f"headroom={headroom.get('phase_a/mean_stage_headroom', float('nan')):.5f}"
        )
        if args.max_steps is not None:
            break

    print("\n=== PHASE B: counterfactual auditor ===")
    freeze_annotation_network(model)
    train_b, val_b = _make_loaders(config_b, device, train_augment=False)
    epochs_b = int(args.epochs_b or config_b.get("epochs", 20))
    auditor_parameters = [p for p in model.parameters() if p.requires_grad]
    opt_b = build_adamw_optimizer(
        auditor_parameters,
        lr=float(config_b.get("auditor_lr", 3e-4)),
        weight_decay=float(config_b.get("weight_decay", 1e-4)),
    )
    accum_b = validate_accumulation_steps(config_b.get("gradient_accumulation_steps", 1))
    sched_cfg_b = dict(config_b)
    sched_cfg_b["accumulation_steps"] = accum_b
    sched_b = build_training_scheduler(opt_b, sched_cfg_b, num_batches=len(train_b), epochs=epochs_b)
    amp_b, dtype_b = resolve_amp(config_b, device)
    scaler_b = build_grad_scaler(enabled=amp_b, device=device, dtype=dtype_b)
    cf_cfg = dict(config_b.get("counterfactual", {}))
    neutral_margin = float(
        args.neutral_margin
        if args.neutral_margin is not None
        else cf_cfg.get("neutral_margin", 0.005)
    )
    generator = CounterfactualGenerator(
        epsilon_neutral=float(cf_cfg.get("epsilon_neutral", 0.02)),
        neutral_max_retries=int(cf_cfg.get("neutral_max_retries", 8)),
        num_classes=int(config_b.get("num_classes", 4)),
    )
    best_b = -float("inf")
    for epoch in range(epochs_b):
        train_stats = train_auditor_epoch(
            model,
            train_b,
            opt_b,
            device,
            generator=generator,
            scheduler=sched_b,
            scaler=scaler_b,
            amp_enabled=amp_b,
            amp_dtype=dtype_b,
            gradient_accumulation_steps=accum_b,
            epoch=epoch,
            total_epochs=epochs_b,
            max_steps=args.max_steps,
            neutral_margin=neutral_margin,
            local_weighting=cf_cfg.get("local_class_weighting", True) != "none",
            disable_tqdm=args.no_tqdm,
        )
        val_stats = validate_auditor_epoch(
            model,
            val_b,
            device,
            generator=generator,
            neutral_margin=neutral_margin,
            local_weighting=cf_cfg.get("local_class_weighting", True) != "none",
            amp_enabled=amp_b,
            amp_dtype=dtype_b,
            max_batches=args.max_val_batches,
            epoch=epoch,
            total_epochs=epochs_b,
            disable_tqdm=args.no_tqdm,
        )
        row = {"epoch": epoch + 1, **train_stats, **val_stats}
        report["phase_b"].append(row)
        _log(logger, "B", epoch + 1, {f"phase_b/{k}": v for k, v in row.items() if k != "epoch"})
        metric = float(val_stats["primary_metric"])
        save_checkpoint(
            output_dir / "phase_b_last.pt",
            model,
            optimizer=opt_b,
            scheduler=sched_b,
            scaler=scaler_b,
            epoch=epoch + 1,
            config=config_b,
            extra={"best_metric": max(best_b, metric), "phase": "auditor"},
        )
        if np.isfinite(metric) and metric > best_b:
            best_b = metric
            save_checkpoint(
                output_dir / "phase_b_best.pt",
                model,
                optimizer=opt_b,
                scheduler=sched_b,
                scaler=scaler_b,
                epoch=epoch + 1,
                config=config_b,
                extra={"best_metric": best_b, "phase": "auditor"},
            )
        print(
            f"B epoch={epoch+1:03d} AUROC={val_stats['auroc']:.4f} "
            f"corr={val_stats['correlation_delta_q_delta_dice']:.4f} "
            f"FIX_F1={val_stats['local_fix_f1']:.4f} "
            f"REGRESS_F1={val_stats['local_regress_f1']:.4f}"
        )
        if args.max_steps is not None:
            break

    print("\n=== PHASE C: joint low-LR self-audit ===")
    _set_trainable(model, annotation=True, auditor=True)
    train_c, val_c = _make_loaders(config_c, device, train_augment=True)
    epochs_c = int(args.epochs_c or config_c.get("epochs", 10))
    opt_c, sched_c, scaler_c, amp_c, dtype_c, accum_c = _scheduler_and_amp(
        model,
        config_c,
        train_c,
        device,
        epochs=epochs_c,
        lr=float(config_c.get("joint_lr", 1e-5)),
        encoder_lr=float(config_c.get("joint_encoder_lr", 1e-6)),
    )
    audit_cfg = dict(config_c.get("audit", {}))
    tau_accept = float(
        args.tau_accept if args.tau_accept is not None else audit_cfg.get("tau_accept", 0.0)
    )
    t_max = int(
        args.t_max
        if args.t_max is not None
        else audit_cfg.get("t_max", config_c.get("model", {}).get("max_turns", 3))
    )
    neutral_margin_c = float(
        args.neutral_margin
        if args.neutral_margin is not None
        else audit_cfg.get("neutral_margin", neutral_margin)
    )
    lambda_audit = float(config_c.get("lambda_audit", 1.0))
    best_c = -float("inf")
    for epoch in range(epochs_c):
        train_stats = finetune_joint_epoch(
            model,
            train_c,
            opt_c,
            device,
            tau_accept=tau_accept,
            t_max=t_max,
            lambda_audit=lambda_audit,
            neutral_margin=neutral_margin_c,
            local_weighting=audit_cfg.get("local_class_weighting", True) != "none",
            scheduler=sched_c,
            scaler=scaler_c,
            amp_enabled=amp_c,
            amp_dtype=dtype_c,
            gradient_accumulation_steps=accum_c,
            grad_clip=config_c.get("grad_clip", 3.0),
            epoch=epoch,
            total_epochs=epochs_c,
            max_steps=args.max_steps,
            disable_tqdm=args.no_tqdm,
        )
        val_stats = validate_phase_c(
            model,
            val_c,
            device,
            tau_accept=tau_accept,
            t_max=t_max,
            max_batches=args.max_val_batches,
            epoch=epoch,
            total_epochs=epochs_c,
            disable_tqdm=args.no_tqdm,
        )
        decomposition = evaluate_audit_decomposition(
            model,
            val_c,
            device,
            tau_accept=tau_accept,
            t_max=t_max,
            neutral_margin=neutral_margin_c,
            max_batches=args.max_val_batches,
            disable_tqdm=args.no_tqdm,
        )
        row = {"epoch": epoch + 1, **train_stats, **val_stats, **decomposition}
        report["phase_c"].append(row)
        _log(logger, "C", epoch + 1, {f"phase_c/{k}": v for k, v in train_stats.items()})
        _log(logger, "C", epoch + 1, val_stats)
        _log(logger, "C", epoch + 1, decomposition)

        metric = float(val_stats["final_foreground_macro_dice"])
        save_checkpoint(
            output_dir / "phase_c_last.pt",
            model,
            optimizer=opt_c,
            scheduler=sched_c,
            scaler=scaler_c,
            epoch=epoch + 1,
            config=config_c,
            extra={"best_metric": max(best_c, metric), "phase": "joint"},
        )
        if np.isfinite(metric) and metric > best_c:
            best_c = metric
            save_checkpoint(
                output_dir / "phase_c_best.pt",
                model,
                optimizer=opt_c,
                scheduler=sched_c,
                scaler=scaler_c,
                epoch=epoch + 1,
                config=config_c,
                extra={"best_metric": best_c, "phase": "joint"},
            )
        print(
            f"C epoch={epoch+1:03d} initial={decomposition.get('modes/initial_dice', float('nan')):.4f} "
            f"always={decomposition.get('modes/always_accept_dice', float('nan')):.4f} "
            f"self={decomposition.get('modes/self_audit_dice', float('nan')):.4f} "
            f"oracle={decomposition.get('modes/oracle_dice', float('nan')):.4f} "
            f"audit_rescue={decomposition.get('modes/audit_rescue_vs_always', float('nan')):+.5f} "
            f"oracle_headroom={decomposition.get('modes/oracle_headroom', float('nan')):+.5f} "
            f"GT_firewall={int(decomposition.get('gt_firewall/passed', 0.0))}"
        )
        if args.max_steps is not None:
            break

    if not args.skip_calibration:
        print("\n=== CALIBRATION: validation transitions only ===")
        cache = collect_validation_transition_cache(
            model,
            val_c,
            device,
            t_max=t_max,
            disable_tqdm=args.no_tqdm,
        )
        thresholds = np.linspace(
            float(args.threshold_min),
            float(args.threshold_max),
            int(args.threshold_steps),
        )
        rows = sweep_thresholds(cache, thresholds)
        best_threshold = select_threshold(rows)
        report["calibration"] = {
            "best": best_threshold,
            "threshold_min": float(args.threshold_min),
            "threshold_max": float(args.threshold_max),
            "threshold_steps": int(args.threshold_steps),
        }
        torch.save(cache, report_dir / "validation_transitions.pt")
        _save_report(report_dir / "calibration.json", report["calibration"])
        print(
            f"tau={best_threshold['tau_accept']:+.5f} "
            f"final={best_threshold['final_macro_dice']:.4f} "
            f"net_gain={best_threshold['net_dice_gain']:+.5f}"
        )

    _save_report(report_dir / "pipeline_report.json", report)
    logger.finish()
    print(f"\nSaved checkpoints: {output_dir}")
    print(f"Saved report: {report_dir / 'pipeline_report.json'}")


if __name__ == "__main__":
    main()
