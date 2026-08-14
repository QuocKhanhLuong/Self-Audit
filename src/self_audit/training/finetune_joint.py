"""Phase C: threshold-controlled joint fine-tuning with separated gradients."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.audit.targets import build_transition_targets, multiclass_dice
from self_audit.evaluation.metrics import acceptance_metrics, transition_audit_metrics
from self_audit.losses.annotation import annotation_loss
from self_audit.losses.audit import audit_loss
from self_audit.training._utils import (
    autocast_context,
    build_data_loader,
    build_grad_scaler,
    build_model_from_config,
    build_patient_dataset,
    build_training_scheduler,
    checkpoint_progress,
    encoder_head_optimizer,
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


def _zero_audit_term(model: torch.nn.Module, reference: torch.Tensor) -> torch.Tensor:
    """Keep the total differentiable when a hard cap produces no transition."""

    auditor = getattr(model, "auditor", None)
    if auditor is None:
        return reference.new_zeros(())
    return sum((parameter.sum() * 0.0 for parameter in auditor.parameters()), reference.new_zeros(()))


def _local_class_weights(
    target: torch.Tensor,
    *,
    num_classes: int = 3,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Return bounded inverse-frequency weights for local audit targets."""

    values = target.reshape(-1).long()
    counts = torch.bincount(values, minlength=int(num_classes)).float()
    present = counts > 0
    weights = torch.ones(int(num_classes), device=target.device, dtype=torch.float32)
    if bool(present.any()):
        total = counts[present].sum()
        n_present = present.sum().float()
        weights[present] = (total / (n_present * counts[present])).clamp(0.25, float(max_weight))
    return weights


def _mask_tensor(value: Any, *, device: torch.device, batch_size: int) -> torch.Tensor:
    mask = value if torch.is_tensor(value) else torch.as_tensor(value, device=device)
    mask = mask.to(device=device, dtype=torch.bool).reshape(-1)
    if mask.numel() != int(batch_size):
        raise ValueError(f"Transition mask must have {batch_size} rows, got {mask.numel()}")
    return mask


def select_audit_output(
    audit_output: Any,
    active_mask: torch.Tensor | Iterable[bool] | None = None,
) -> Any | None:
    """Select active rows from a batch-aligned Auditor output.

    The model stores full ``[B,...]`` audit tensors for every transition so
    that transition alignment is preserved.  Phase C calls this helper before
    computing GT-derived targets.  If no explicit mask is supplied, the
    helper reads ``active_mask``/``audit_mask`` from a mapping output.
    """

    if audit_output is None:
        return None
    if active_mask is None and isinstance(audit_output, Mapping):
        active_mask = audit_output.get("active_mask", audit_output.get("audit_mask"))
    if active_mask is None:
        return audit_output

    if torch.is_tensor(active_mask):
        mask = active_mask.to(dtype=torch.bool).reshape(-1)
    else:
        mask = torch.as_tensor(list(active_mask), dtype=torch.bool).reshape(-1)
    if not bool(mask.any()):
        return None
    indices = mask.nonzero(as_tuple=False).flatten()

    def select(value: Any) -> Any:
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == mask.numel():
            return value.index_select(0, indices.to(device=value.device))
        return value

    if isinstance(audit_output, Mapping):
        return {key: select(value) for key, value in audit_output.items()}
    if torch.is_tensor(audit_output):
        return select(audit_output)
    # AuditOutput is a small dataclass in the model namespace.  Returning a
    # mapping keeps this helper independent of that class and is accepted by
    # audit_loss's dict/object compatibility layer.
    local_logits = getattr(audit_output, "local_logits", None)
    delta_q = getattr(audit_output, "delta_q", None)
    if torch.is_tensor(local_logits) or torch.is_tensor(delta_q):
        return {
            "local_logits": select(local_logits),
            "delta_q": select(delta_q),
        }
    return audit_output


def _transition_value_at(value: Any, index: int) -> Any | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.ndim == 1:
            return value if index == 0 else None
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return None


def _transition_active_mask(
    output: Mapping[str, Any],
    audit_output: Any,
    index: int,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    masks = output.get("transition_active_masks", output.get("active_masks"))
    value = _transition_value_at(masks, index)
    if value is None and isinstance(audit_output, Mapping):
        value = audit_output.get("active_mask", audit_output.get("audit_mask"))
    if value is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    return _mask_tensor(value, device=device, batch_size=batch_size)


def _select_batch_rows(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        raise ValueError("Transition tensors must have a batch dimension")
    if value.shape[0] == mask.numel():
        return value.index_select(0, mask.nonzero(as_tuple=False).flatten().to(value.device))
    if value.shape[0] == int(mask.sum().item()):
        return value
    raise ValueError(
        f"Transition batch rows {value.shape[0]} do not match mask {mask.numel()} "
        f"or active rows {int(mask.sum().item())}"
    )


def compute_joint_losses(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    lambda_audit: float = 1.0,
    neutral_margin: float = 0.005,
    local_weighting: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute annotation and Auditor losses with an explicit gradient split.

    The inference gate is discrete and receives detached ``DeltaQ``.  The
    annotation term sees only the retained final state.  The audit term uses
    detached transition inputs and therefore can update only Auditor weights.
    """

    if hasattr(model, "infer"):
        output = model.infer(
            batch["image"],
            mode="self_audit",
            tau_accept=float(tau_accept),
            t_max=int(t_max),
        )
    elif hasattr(model, "forward_joint"):  # pragma: no cover - compatibility fallback
        output = model.forward_joint(batch["image"])
    else:  # pragma: no cover
        output = model(batch["image"])

    if isinstance(output, dict):
        final_logits = output.get("logits")
        if not torch.is_tensor(final_logits):
            final_logits = extract_initial_logits(output)
    else:
        final_logits = extract_initial_logits(output)
    annotation_term = annotation_loss(final_logits, batch["mask"])[0]

    audit_terms: list[torch.Tensor] = []
    audit_parts: list[dict[str, torch.Tensor]] = []
    audit_sample_counts: list[int] = []
    audit_transition_masks: list[torch.Tensor] = []
    if isinstance(output, dict):
        previous_states = output.get("transition_previous", [])
        candidate_states = output.get("transition_candidates", output.get("candidates", []))
        audit_outputs = output.get("audits", [])
        batch_size = int(batch["image"].shape[0])
        for index, (previous, candidate, audit_output) in enumerate(
            zip(previous_states, candidate_states, audit_outputs)
        ):
            active_mask = _transition_active_mask(
                output,
                audit_output,
                index,
                batch_size=batch_size,
                device=batch["image"].device,
            )
            if not bool(active_mask.any()):
                continue
            selected_audit = select_audit_output(audit_output, active_mask)
            if selected_audit is None:
                continue
            # Targets and Auditor inputs are explicitly detached.  The Auditor
            # also defensively detaches its inputs internally.
            previous_audit = _select_batch_rows(previous.detach(), active_mask)
            candidate_audit = _select_batch_rows(candidate.detach(), active_mask)
            ground_truth = _select_batch_rows(batch["mask"], active_mask)
            targets = build_transition_targets(previous_audit, candidate_audit, ground_truth)
            weights = _local_class_weights(targets.local) if local_weighting else None
            audit_term, parts = audit_loss(
                selected_audit,
                targets,
                neutral_margin=float(neutral_margin),
                local_class_weights=weights,
            )
            audit_terms.append(audit_term)
            audit_parts.append(parts)
            audit_sample_counts.append(int(active_mask.sum().item()))
            audit_transition_masks.append(active_mask.detach())
    audit_term = torch.stack(audit_terms).mean() if audit_terms else _zero_audit_term(model, annotation_term)
    total = annotation_term + float(lambda_audit) * audit_term
    details: dict[str, Any] = {
        "annotation_loss": annotation_term.detach(),
        "audit_loss": audit_term.detach(),
        "total_loss": total.detach(),
        # Differentiable references are intentionally exposed for phase-level
        # gradient tests and diagnostics; callers should backpropagate only
        # the total or the explicitly selected term.
        "annotation_loss_tensor": annotation_term,
        "audit_loss_tensor": audit_term,
        "total_loss_tensor": total,
        "audit_parts": audit_parts,
        "transition_count": len(audit_terms),
        "audit_sample_counts": audit_sample_counts,
        "audit_transition_masks": audit_transition_masks,
        "output": output,
    }
    return total, details


def _audit_output_tensor(output: Any, *names: str) -> torch.Tensor | None:
    for name in names:
        value = output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    return None


def _foreground_dice_per_sample(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return multiclass_dice(logits, target, num_classes=int(logits.shape[1]))


@torch.no_grad()
def validate_phase_c(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Validate Phase C behavior with GT used only after deployable inference.

    The returned annotation metrics compare initial ``A0`` and retained final
    states.  Transition targets are constructed only in this validation
    helper, after ``model.infer(..., mode="self_audit")`` has completed.
    """

    was_training = model.training
    model.eval()
    initial_scores: list[torch.Tensor] = []
    final_scores: list[torch.Tensor] = []
    attempted_counts: list[torch.Tensor] = []
    accepted_counts: list[torch.Tensor] = []
    accepted_values: list[torch.Tensor] = []
    actual_deltas: list[torch.Tensor] = []
    audit_local_predictions: list[torch.Tensor] = []
    audit_local_targets: list[torch.Tensor] = []
    audit_delta_predictions: list[torch.Tensor] = []
    audit_delta_targets: list[torch.Tensor] = []

    try:
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_batch(raw_batch, device)
            output = model.infer(
                batch["image"],
                mode="self_audit",
                tau_accept=float(tau_accept),
                t_max=int(t_max),
            )
            if not isinstance(output, Mapping):
                raise TypeError("Phase-C validation requires a mapping inference output")
            initial_logits = output.get("initial_logits")
            if not torch.is_tensor(initial_logits):
                initial_logits = extract_initial_logits(output)
            final_logits = output.get("logits")
            if not torch.is_tensor(final_logits):
                final_logits = initial_logits
            ground_truth = batch["mask"]
            initial_scores.append(_foreground_dice_per_sample(initial_logits, ground_truth).detach())
            final_scores.append(_foreground_dice_per_sample(final_logits, ground_truth).detach())

            batch_size = int(batch["image"].shape[0])
            attempted_fallback = torch.zeros(batch_size, dtype=torch.long, device=device)
            accepted_fallback = torch.zeros(batch_size, dtype=torch.long, device=device)
            previous_states = output.get("transition_previous", [])
            candidate_states = output.get("transition_candidates", output.get("candidates", []))
            audit_outputs = output.get("audits", [])
            for index, (previous, candidate, audit_output) in enumerate(
                zip(previous_states, candidate_states, audit_outputs)
            ):
                active_mask = _transition_active_mask(
                    output,
                    audit_output,
                    index,
                    batch_size=batch_size,
                    device=device,
                )
                attempted_fallback += active_mask.to(torch.long)
                if not bool(active_mask.any()):
                    continue
                state_mask_value = _transition_value_at(
                    output.get("transition_state_masks", output.get("state_masks")),
                    index,
                )
                if state_mask_value is None and isinstance(audit_output, Mapping):
                    state_mask_value = audit_output.get("state_mask", audit_output.get("accepted"))
                accepted_mask = (
                    torch.zeros(batch_size, dtype=torch.bool, device=device)
                    if state_mask_value is None
                    else _mask_tensor(state_mask_value, device=device, batch_size=batch_size)
                )
                accepted_fallback += accepted_mask.to(torch.long)

                previous_selected = _select_batch_rows(previous.detach(), active_mask)
                candidate_selected = _select_batch_rows(candidate.detach(), active_mask)
                target_selected = _select_batch_rows(ground_truth, active_mask)
                targets = build_transition_targets(
                    previous_selected,
                    candidate_selected,
                    target_selected,
                )
                accepted_values.append(_select_batch_rows(accepted_mask, active_mask))
                actual_deltas.append(targets.delta_dice.reshape(-1).detach())

                selected_audit = select_audit_output(audit_output, active_mask)
                local_prediction = _audit_output_tensor(selected_audit, "local_logits", "local")
                delta_prediction = _audit_output_tensor(selected_audit, "delta_q", "global_delta_q", "delta_quality")
                if local_prediction is not None and delta_prediction is not None:
                    audit_local_predictions.append(local_prediction.detach())
                    audit_local_targets.append(targets.local.detach())
                    audit_delta_predictions.append(delta_prediction.detach().reshape(-1))
                    audit_delta_targets.append(targets.delta_dice.detach().reshape(-1))

            attempted_value = output.get("num_attempted_turns")
            accepted_value = output.get("accepted_count")
            attempted_counts.append(
                attempted_value.detach().to(torch.long)
                if torch.is_tensor(attempted_value)
                else attempted_fallback
            )
            accepted_counts.append(
                accepted_value.detach().to(torch.long)
                if torch.is_tensor(accepted_value)
                else accepted_fallback
            )
    finally:
        model.train(was_training)

    if initial_scores:
        initial_mean = float(torch.cat(initial_scores).mean())
        final_mean = float(torch.cat(final_scores).mean())
        mean_attempted = float(torch.cat(attempted_counts).float().mean())
        mean_accepted = float(torch.cat(accepted_counts).float().mean())
    else:
        initial_mean = float("nan")
        final_mean = float("nan")
        mean_attempted = float("nan")
        mean_accepted = float("nan")

    if accepted_values and actual_deltas:
        acceptance = acceptance_metrics(
            torch.cat(accepted_values),
            torch.cat(actual_deltas),
        )
    else:
        acceptance = {
            "harmful_acceptance_rate": 0.0,
            "beneficial_rejection_rate": 0.0,
            "net_dice_gain_after_auditing": 0.0,
        }

    if audit_local_predictions:
        audit_metrics = transition_audit_metrics(
            torch.cat(audit_local_predictions),
            torch.cat(audit_local_targets),
            torch.cat(audit_delta_predictions),
            torch.cat(audit_delta_targets),
        )
    else:
        audit_metrics = {
            "improve_regress_accuracy": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "correlation_delta_q_delta_dice": float("nan"),
            "local_fix_f1": float("nan"),
            "local_regress_f1": float("nan"),
        }

    result: dict[str, Any] = {
        "initial_foreground_macro_dice": initial_mean,
        "final_foreground_macro_dice": final_mean,
        "net_dice_gain": final_mean - initial_mean,
        "net_gain": final_mean - initial_mean,
        "harmful_acceptance_rate": acceptance["harmful_acceptance_rate"],
        "beneficial_rejection_rate": acceptance["beneficial_rejection_rate"],
        "net_dice_gain_after_auditing": acceptance["net_dice_gain_after_auditing"],
        "mean_attempted_turns": mean_attempted,
        "mean_accepted_turns": mean_accepted,
        "per_sample_mean_attempted_turns": mean_attempted,
        "per_sample_mean_accepted_turns": mean_accepted,
        "audit_transition_count": int(sum(int(values.numel()) for values in actual_deltas)),
        "audit_metrics": audit_metrics,
    }
    result.update({f"audit_{key}": value for key, value in audit_metrics.items()})
    return result


# Explicit alias for callers that use the validation stage name.
validate_joint = validate_phase_c


@torch.no_grad()
def collect_validation_transition_cache(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    t_max: int = 3,
) -> dict[str, torch.Tensor]:
    """Cache full validation transitions for threshold calibration.

    ``always_accept_refinement`` is used only to expose the full candidate
    trajectory.  The saved cache contains GT-derived deltas for calibration;
    it is never consumed by deployable inference.
    """

    model.eval()
    initial_values: list[torch.Tensor] = []
    quality_values: list[torch.Tensor] = []
    actual_values: list[torch.Tensor] = []
    active_values: list[torch.Tensor] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model.infer(
            batch["image"],
            mode="always_accept_refinement",
            tau_accept=-float("inf"),
            t_max=int(t_max),
        )
        initial = output.get("initial_logits")
        if not torch.is_tensor(initial):
            initial = extract_initial_logits(output)
        initial_values.append(_foreground_dice_per_sample(initial, batch["mask"]).detach().cpu())
        batch_quality: list[torch.Tensor] = []
        batch_actual: list[torch.Tensor] = []
        batch_active: list[torch.Tensor] = []
        for index, (previous, candidate, audit_output) in enumerate(
            zip(output.get("transition_previous", []), output.get("transition_candidates", []), output.get("audits", []))
        ):
            active_mask = _transition_active_mask(
                output,
                audit_output,
                index,
                batch_size=int(batch["image"].shape[0]),
                device=device,
            )
            delta_q = _audit_output_tensor(audit_output, "delta_q", "global_delta_q", "delta_quality")
            if delta_q is None:
                raise ValueError("Auditor output lacks delta_q for threshold cache")
            targets = build_transition_targets(previous.detach(), candidate.detach(), batch["mask"])
            batch_quality.append(delta_q.detach().reshape(-1).cpu())
            batch_actual.append(targets.delta_dice.detach().reshape(-1).cpu())
            batch_active.append(active_mask.detach().cpu())
        if batch_quality:
            quality_values.append(torch.stack(batch_quality, dim=1))
            actual_values.append(torch.stack(batch_actual, dim=1))
            active_values.append(torch.stack(batch_active, dim=1))
    if not initial_values:
        raise ValueError("Cannot cache threshold transitions from an empty validation loader")
    if not quality_values:
        size = int(torch.cat(initial_values).shape[0])
        empty = torch.empty((size, 0), dtype=torch.float32)
        return {"initial_dice": torch.cat(initial_values), "delta_q": empty, "actual_delta_dice": empty, "active_mask": empty.bool()}
    return {
        "initial_dice": torch.cat(initial_values),
        "delta_q": torch.cat(quality_values),
        "actual_delta_dice": torch.cat(actual_values),
        "active_mask": torch.cat(active_values),
    }


def joint_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    lambda_audit: float = 1.0,
    neutral_margin: float = 0.005,
    local_weighting: bool = True,
) -> torch.Tensor:
    """Backward-compatible scalar Phase-C objective."""

    return compute_joint_losses(
        model,
        batch,
        tau_accept=tau_accept,
        t_max=t_max,
        lambda_audit=lambda_audit,
        neutral_margin=neutral_margin,
        local_weighting=local_weighting,
    )[0]


def finetune_joint_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    lambda_audit: float = 1.0,
    neutral_margin: float = 0.005,
    local_weighting: bool = True,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    gradient_accumulation_steps: int = 1,
    grad_clip: float | None = 3.0,
    epoch: int = 0,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.train()
    accumulation_steps = validate_accumulation_steps(gradient_accumulation_steps)
    if max_steps is not None and int(max_steps) < 1:
        raise ValueError("max_steps must be positive when provided")
    if scaler is None:
        scaler = build_grad_scaler(enabled=amp_enabled, device=device, dtype=amp_dtype)
    running = 0.0
    running_annotation = 0.0
    running_audit = 0.0
    transitions = 0
    steps = 0
    optimizer_steps = 0
    pending = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(loader):
        if max_steps is not None and optimizer_steps >= int(max_steps):
            break
        batch = move_batch(raw_batch, device)
        with autocast_context(enabled=amp_enabled, device=device, dtype=amp_dtype):
            loss, details = compute_joint_losses(
                model,
                batch,
                tau_accept=tau_accept,
                t_max=t_max,
                lambda_audit=lambda_audit,
                neutral_margin=neutral_margin,
                local_weighting=local_weighting,
            )
        if not is_finite(loss):
            raise FloatingPointError(
                f"Non-finite Phase-C loss at epoch={epoch} step={batch_index}: "
                f"loss={float(loss.detach())!r} annotation={float(details['annotation_loss'])!r} "
                f"audit={float(details['audit_loss'])!r}"
            )
        scaled_loss = loss / float(accumulation_steps)
        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        pending += 1
        running += float(loss.detach())
        running_annotation += float(details["annotation_loss"])
        running_audit += float(details["audit_loss"])
        transitions += int(details["transition_count"])
        steps += 1
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
                raise FloatingPointError(f"Non-finite Phase-C gradients at epoch={epoch} step={batch_index}")
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
            grad_clip=grad_clip,
        )
        if not step_ok:
            raise FloatingPointError(f"Non-finite Phase-C gradients at epoch={epoch} final_partial_step")
        optimizer_steps += 1
    divisor = max(steps, 1)
    return {
        "loss": running / divisor,
        "annotation_loss": running_annotation / divisor,
        "audit_loss": running_audit / divisor,
        "transitions": float(transitions),
        "optimizer_steps": float(optimizer_steps),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Audit Phase C joint fine-tuning")
    parser.add_argument("--config", default="configs/self_audit_joint.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lambda_audit", type=float, default=None)
    parser.add_argument("--tau_accept", type=float, default=None)
    parser.add_argument("--t_max", type=int, default=None)
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
    initial_checkpoint = args.resume or args.checkpoint
    if initial_checkpoint is None:
        raise ValueError("Phase C requires --checkpoint or --resume")
    train_dataset = build_patient_dataset(config, split=str(config.get("train_split", "train")), train=True)
    val_dataset = build_patient_dataset(config, split=str(config.get("val_split", "val")), train=False)
    loader = build_data_loader(train_dataset, config, device=device, train=True)
    val_loader = build_data_loader(val_dataset, config, device=device, train=False)
    optimizer = encoder_head_optimizer(
        model,
        lr=float(config.get("joint_lr", 1e-5)),
        encoder_lr=float(config.get("joint_encoder_lr", 1e-6)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(args.epochs or config.get("epochs", 10))
    accumulation_steps = validate_accumulation_steps(config.get("gradient_accumulation_steps", 1))
    scheduler_config = dict(config)
    scheduler_config["accumulation_steps"] = accumulation_steps
    scheduler = build_training_scheduler(optimizer, scheduler_config, num_batches=len(loader), epochs=epochs)
    amp_enabled, amp_dtype = resolve_amp(config, device)
    scaler = build_grad_scaler(enabled=amp_enabled, device=device, dtype=amp_dtype)
    start_epoch = 0
    best_metric = float("-inf")
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch, _, _ = checkpoint_progress(payload)
        best_metric = float(payload.get("best_metric", float("-inf")))
    else:
        load_checkpoint(args.checkpoint, model=model, map_location=device)
    audit_config = config.get("audit", {})
    if not isinstance(audit_config, dict):
        raise ValueError("audit config must be a mapping")
    lambda_audit = float(args.lambda_audit if args.lambda_audit is not None else config.get("lambda_audit", 1.0))
    tau_accept = float(args.tau_accept if args.tau_accept is not None else audit_config.get("tau_accept", 0.0))
    t_max = int(args.t_max if args.t_max is not None else audit_config.get("t_max", config.get("model", {}).get("max_turns", 3)))
    output_dir = Path(config.get("output", "weights/self_audit/phase_c_joint.pt")).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        stats = finetune_joint_epoch(
            model,
            loader,
            optimizer,
            device,
            tau_accept=tau_accept,
            t_max=t_max,
            lambda_audit=lambda_audit,
            neutral_margin=float(audit_config.get("neutral_margin", 0.005)),
            local_weighting=audit_config.get("local_class_weighting", True) != "none",
            scheduler=scheduler,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            gradient_accumulation_steps=accumulation_steps,
            grad_clip=config.get("grad_clip", 3.0),
            epoch=epoch,
            max_steps=args.max_steps,
        )
        validation = validate_phase_c(
            model,
            val_loader,
            device,
            tau_accept=tau_accept,
            t_max=t_max,
            max_batches=args.max_val_batches,
        )
        metric = float(validation["final_foreground_macro_dice"])
        is_best = is_finite(metric) and metric > best_metric
        if is_best:
            best_metric = metric
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            config=config,
            extra={"best_metric": best_metric, "phase": "joint"},
        )
        if is_best:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                config=config,
                extra={"best_metric": best_metric, "phase": "joint"},
            )
        print(
            f"epoch={epoch + 1:03d} lr={stats['lr']:.3e} loss={stats['loss']:.5f} "
            f"annotation={stats['annotation_loss']:.5f} audit={stats['audit_loss']:.5f} "
            f"initial_dice={validation['initial_foreground_macro_dice']:.4f} "
            f"final_dice={validation['final_foreground_macro_dice']:.4f} "
            f"net_gain={validation['net_dice_gain']:.4f} "
            f"harmful_acceptance={validation['harmful_acceptance_rate']:.4f} "
            f"beneficial_rejection={validation['beneficial_rejection_rate']:.4f} "
            f"mean_attempted={validation['mean_attempted_turns']:.3f} "
            f"mean_accepted={validation['mean_accepted_turns']:.3f} "
            f"audit_AUROC={validation['audit_auroc']:.4f} "
            f"audit_FIX_F1={validation['audit_local_fix_f1']:.4f} "
            f"audit_REGRESS_F1={validation['audit_local_regress_f1']:.4f} "
            f"audit_corr={validation['audit_correlation_delta_q_delta_dice']:.4f} "
            f"tau={tau_accept:.4f}"
        )
        if args.max_steps is not None:
            break
    print(f"saved_last={output_dir / 'last.pt'} saved_best={output_dir / 'best.pt'}")


if __name__ == "__main__":  # pragma: no cover
    main()
