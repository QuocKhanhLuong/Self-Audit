"""Phase B: freeze annotation and train the transition auditor."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.audit.counterfactual import CounterfactualGenerator
from self_audit.audit.semantics import check_generation_tolerance, resolve_neutral_margin
from self_audit.audit.targets import build_transition_targets
from self_audit.evaluation.metrics import transition_audit_metrics
from self_audit.losses.audit import audit_loss
from self_audit.training._utils import (
    add_wandb_and_tqdm_args,
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
    print_model_parameter_summary,
    resolve_amp,
    resolve_device,
    save_checkpoint,
    seed_everything,
    setup_wandb_logger,
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


#: Coarse provenance buckets a transition can belong to.  ``"on_policy"`` pairs
#: are adjacent states the annotation network actually produced; ``"synthetic"``
#: pairs are constructed by the counterfactual generator from ground truth and
#: therefore cannot occur at inference time.
ON_POLICY = "on_policy"
SYNTHETIC = "synthetic"
PROVENANCE_KINDS = (ON_POLICY, SYNTHETIC)


def provenance_kind(transition: Mapping[str, Any]) -> str:
    """Return the coarse ``"on_policy"`` / ``"synthetic"`` bucket of a transition.

    Prefers the explicit ``provenance_kind`` field.  Transitions built by older
    code (or by hand in a test) only carry the fine-grained ``provenance``
    string, so fall back to parsing its prefix rather than failing.
    """

    kind = transition.get("provenance_kind")
    if kind is not None:
        name = str(kind)
        if name not in PROVENANCE_KINDS:
            raise ValueError(f"provenance_kind must be one of {PROVENANCE_KINDS}, got {kind!r}")
        return name
    provenance = str(transition.get("provenance", ""))
    return ON_POLICY if provenance == ON_POLICY else SYNTHETIC


def build_auditor_transitions(
    output: Any,
    ground_truth: torch.Tensor,
    generator: CounterfactualGenerator,
    *,
    include_on_policy: bool = True,
    include_synthetic: bool = True,
) -> list[dict[str, Any]]:
    """Build adjacent on-policy pairs plus one synthetic pair around every A_t.

    The defaults reproduce the historical list exactly -- same order, same
    content -- so training behaviour is unchanged.  ``include_on_policy`` /
    ``include_synthetic`` exist so evaluation can build a provenance-restricted
    population without re-deriving the trajectory by hand.

    Every transition carries the fine-grained ``provenance`` string it always
    had plus a coarse ``provenance_kind`` in ``{"on_policy", "synthetic"}`` so
    callers do not have to string-parse.
    """

    trajectory = extract_annotation_trajectory(output)
    transitions: list[dict[str, Any]] = []
    if include_on_policy:
        for turn, (previous_state, candidate_state) in enumerate(zip(trajectory[:-1], trajectory[1:])):
            transitions.append(
                {
                    "previous": _state_probabilities(previous_state).detach(),
                    "candidate": _state_probabilities(candidate_state).detach(),
                    "turn_index": turn,
                    "provenance": "on_policy",
                    "provenance_kind": ON_POLICY,
                    "valid_mask": torch.ones(previous_state.shape[0], dtype=torch.bool, device=previous_state.device),
                }
            )
    if include_synthetic:
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
                        "provenance_kind": SYNTHETIC,
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
    # Collected predictions/targets are bucketed by provenance so validation can
    # report on-policy and synthetic quality separately.  The "combined" bucket
    # is appended in loop order, so it is bit-identical to the single flat list
    # this function used to return.
    buckets: dict[str, list[list[torch.Tensor]]] = {
        name: [[], [], [], []] for name in (*PROVENANCE_KINDS, "combined")
    }
    group_counts: dict[str, int] = {name: 0 for name in PROVENANCE_KINDS}
    local_counts = torch.zeros(3, dtype=torch.long, device=batch["mask"].device)
    start = time.perf_counter()
    for transition in transitions:
        kind = provenance_kind(transition)
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
        group_counts[kind] += 1
        local_counts += torch.bincount(targets.local.reshape(-1), minlength=3).to(local_counts.device)
        if collect:
            row = (
                audit_output.local_logits.detach(),
                targets.local.detach(),
                audit_output.delta_q.detach().reshape(-1),
                targets.delta_dice.detach().reshape(-1),
            )
            for name in (kind, "combined"):
                for slot, value in zip(buckets[name], row):
                    slot.append(value)
    timings["auditor_ms"] = (time.perf_counter() - start) * 1000.0
    if not losses:
        return None, {
            "transitions": 0,
            "transitions_by_provenance": {name: 0 for name in PROVENANCE_KINDS},
            "timings": timings,
            "local_counts": local_counts,
            "transition_data": None,
        }
    # Uniform mean over transition groups -- unchanged, synthetic still carries
    # its historical share of the TRAINING loss.  Only reporting is partitioned.
    loss = torch.stack(losses).mean()
    data = {
        "transitions": len(losses),
        "transitions_by_provenance": dict(group_counts),
        "timings": timings,
        "local_counts": local_counts,
        "transition_data": {
            name: tuple(torch.cat(slot) if slot else None for slot in slots)
            for name, slots in buckets.items()
        },
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
    total_epochs: int = 1,
    max_steps: int | None = None,
    neutral_margin: float = 0.005,
    local_weighting: bool = True,
    audit_margin: float = 0.05,
    disable_tqdm: bool = False,
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
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch + 1:03d}/{total_epochs:03d} [Train B]",
        disable=disable_tqdm,
        leave=False,
    )
    for batch_index, raw_batch in enumerate(pbar):
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
        pbar.set_postfix({
            "loss": f"{float(loss.detach()):.4f}",
            "trans": f"{transition_count}",
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


#: Keys every transition-metric namespace is guaranteed to carry, so a caller
#: reading ``audit/on_policy/auroc`` never has to guard for a missing key when a
#: provenance bucket happens to be empty.
TRANSITION_METRIC_KEYS = (
    "improve_regress_accuracy",
    "auroc",
    "auprc",
    "correlation_delta_q_delta_dice",
    "local_fix_f1",
    "local_regress_f1",
)


def _empty_transition_metrics() -> dict[str, float]:
    return {key: float("nan") for key in TRANSITION_METRIC_KEYS}


def _transition_metrics_for(
    parts: list[tuple[Any, Any, Any, Any]],
    *,
    neutral_margin: float | None,
) -> tuple[dict[str, float], int]:
    """Concatenate one provenance bucket and score it.

    ``transition_audit_metrics`` is gaining a ``neutral_margin`` keyword in a
    parallel change (AGY-1).  Call it defensively until that lands: try the
    keyword, and fall back to the historical positional call if the signature
    does not accept it yet.
    """

    usable = [item for item in parts if item is not None and item[2] is not None]
    if not usable:
        return _empty_transition_metrics(), 0
    local_pred = torch.cat([item[0] for item in usable])
    local_target = torch.cat([item[1] for item in usable])
    delta_pred = torch.cat([item[2] for item in usable])
    delta_target = torch.cat([item[3] for item in usable])
    metrics = transition_audit_metrics(
        local_pred, local_target, delta_pred, delta_target, neutral_margin=neutral_margin
    )
    result = _empty_transition_metrics()
    result.update({str(key): value for key, value in metrics.items()})
    return result, int(delta_pred.numel())


def resolve_primary_metric(on_policy_metrics: Mapping[str, Any]) -> tuple[float, str]:
    """Select the checkpoint metric from the ON-POLICY namespace only.

    Fallback chain, applied in this order and recorded verbatim in
    ``primary_metric_source``:

    1. ``on_policy_auroc``
    2. ``on_policy_improve_regress_accuracy``
    3. ``nan`` with source ``"undefined"``

    It deliberately never falls back to the combined or synthetic value: those
    are 57.1% generator-synthesized transitions that cannot occur at inference,
    so selecting on them optimizes a task the deployed model never faces.
    """

    auroc = float(on_policy_metrics.get("auroc", float("nan")))
    if is_finite(auroc):
        return auroc, "on_policy_auroc"
    accuracy = float(on_policy_metrics.get("improve_regress_accuracy", float("nan")))
    if is_finite(accuracy):
        return accuracy, "on_policy_improve_regress_accuracy"
    return float("nan"), "undefined"


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
    epoch: int = 0,
    total_epochs: int = 1,
    disable_tqdm: bool = False,
) -> dict[str, Any]:
    """Score the validation loader, partitioned by transition provenance.

    Three namespaces are emitted -- ``audit/on_policy/*``, ``audit/synthetic/*``
    and ``audit/combined/*`` -- each carrying ``auroc``, ``auprc``,
    ``improve_regress_accuracy``, ``correlation_delta_q_delta_dice``,
    ``local_fix_f1``, ``local_regress_f1`` and ``transition_count``.

    The flat legacy keys (``auroc``, ``auprc``, ...) still carry the COMBINED
    value so existing callers and W&B history keep working, but ``primary_metric``
    -- the value that selects ``best.pt`` -- is now derived from the ON-POLICY
    namespace alone.  Combined is a 4-synthetic / 3-on-policy mixture whose
    synthetic half is generated from ground truth and cannot occur at inference,
    so selecting on it optimizes a task the deployed model never faces.
    """

    model.eval()
    generator = generator or CounterfactualGenerator()
    losses: list[float] = []
    namespaces = (*PROVENANCE_KINDS, "combined")
    transition_parts: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = {
        name: [] for name in namespaces
    }
    group_counts: dict[str, int] = {name: 0 for name in namespaces}
    count = 0
    timings = {"annotation_forward_ms": 0.0, "counterfactual_ms": 0.0, "auditor_ms": 0.0}
    local_counts = torch.zeros(3, dtype=torch.long)
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch + 1:03d}/{total_epochs:03d} [Val B]",
        disable=disable_tqdm,
        leave=False,
    )
    for batch_index, raw_batch in enumerate(pbar):
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
        for name, item in (details["transition_data"] or {}).items():
            if name in transition_parts:
                transition_parts[name].append(item)
        for name, value in details.get("transitions_by_provenance", {}).items():
            if name in group_counts:
                group_counts[name] += int(value)
                group_counts["combined"] += int(value)
        pbar.set_postfix({"loss": f"{float(loss.detach()):.4f}", "trans": f"{count}"})
    namespace_metrics: dict[str, dict[str, float]] = {}
    namespaced: dict[str, float] = {}
    for name in namespaces:
        scores, rows = _transition_metrics_for(transition_parts[name], neutral_margin=neutral_margin)
        scores["transition_count"] = float(rows)
        if name in group_counts:
            scores["transition_group_count"] = float(group_counts[name])
        namespace_metrics[name] = scores
        for key, value in scores.items():
            namespaced[f"audit/{name}/{key}"] = value
    # Legacy flat keys keep emitting the COMBINED value so existing callers and
    # W&B history are unbroken -- but they no longer drive checkpoint selection.
    metrics = {
        key: value
        for key, value in namespace_metrics["combined"].items()
        if key not in ("transition_count", "transition_group_count")
    }
    primary, primary_source = resolve_primary_metric(namespace_metrics[ON_POLICY])
    return {
        "audit_loss": sum(losses) / max(len(losses), 1),
        "loss": sum(losses) / max(len(losses), 1),
        "transitions": float(count),
        "primary_metric": primary,
        "primary_metric_source": primary_source,
        "local_fix_count": float(local_counts[0]),
        "local_unchanged_count": float(local_counts[1]),
        "local_regress_count": float(local_counts[2]),
        "annotation_forward_ms": timings["annotation_forward_ms"] / max(len(losses), 1),
        "counterfactual_ms": timings["counterfactual_ms"] / max(len(losses), 1),
        "auditor_ms": timings["auditor_ms"] / max(len(losses), 1),
        **namespaced,
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Audit Phase B auditor training")
    parser.add_argument("--config", default="configs/self_audit_auditor.yaml")
    parser.add_argument("--annotation_checkpoint", default=None)
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
    parser.add_argument("--max_val_batches", type=int, default=None)
    add_wandb_and_tqdm_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.split_manifest is not None:
        config["split_manifest"] = args.split_manifest
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
    print_model_parameter_summary(model, title="Phase B: Auditor Parameters (Annotation Frozen)")
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
    wandb_logger = setup_wandb_logger(args, config, phase="auditor")
    cf_config = config.get("counterfactual", {})
    if not isinstance(cf_config, dict):
        raise ValueError("counterfactual config must be a mapping")
    epsilon_neutral = float(cf_config.get("epsilon_neutral", 0.02))
    neutral_margin = resolve_neutral_margin(cf_config.get("neutral_margin", 0.005))
    # Surfaces the epsilon_neutral (generation search tolerance) vs neutral_margin
    # (decision margin) conflict at runtime.  Neither number is changed here.
    check_generation_tolerance(epsilon_neutral, neutral_margin, context="Phase-B CounterfactualGenerator")
    generator = CounterfactualGenerator(
        epsilon_neutral=epsilon_neutral,
        neutral_max_retries=int(cf_config.get("neutral_max_retries", 8)),
        num_classes=int(config.get("num_classes", 4)),
    )
    best_metric = float("-inf")
    start_epoch = 0
    if args.resume:
        payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location=device)
        start_epoch, _, _ = checkpoint_progress(payload)
        best_metric = float(payload.get("best_metric", float("-inf")))
    output_target = Path(args.output or config.get("output", "weights/self_audit/phase_b_auditor.pt"))
    output_dir = output_target.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for epoch in range(start_epoch, epochs):
            train_stats = train_auditor_epoch(
                model, train_loader, optimizer, device, generator=generator, scheduler=scheduler,
                scaler=scaler, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                gradient_accumulation_steps=accumulation_steps, epoch=epoch, total_epochs=epochs, max_steps=args.max_steps,
                neutral_margin=neutral_margin,
                local_weighting=cf_config.get("local_class_weighting", True) != "none",
                disable_tqdm=args.no_tqdm,
            )
            validation = validate_auditor_epoch(
                model, val_loader, device, generator=generator,
                neutral_margin=neutral_margin,
                local_weighting=cf_config.get("local_class_weighting", True) != "none",
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                max_batches=args.max_val_batches,
                epoch=epoch,
                total_epochs=epochs,
                disable_tqdm=args.no_tqdm,
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
                save_checkpoint(output_target, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch + 1, config=config, extra={"best_metric": metric, "phase": "auditor"})
            log_payload = {
                "epoch": epoch + 1,
                "train/loss": train_stats["loss"],
                "train/lr": train_stats["lr"],
                "train/transitions": train_stats["transitions"],
                "val/audit_loss": validation["audit_loss"],
                # Legacy flat keys are the COMBINED (on-policy + synthetic)
                # mixture, kept so existing W&B history stays continuous.  They
                # no longer select checkpoints.
                "val/auroc": validation["auroc"],
                "val/auprc": validation["auprc"],
                "val/local_fix_f1": validation["local_fix_f1"],
                "val/local_regress_f1": validation["local_regress_f1"],
                "val/correlation_delta_q": validation["correlation_delta_q_delta_dice"],
                "val/improve_regress_accuracy": validation["improve_regress_accuracy"],
                "val/primary_metric": validation["primary_metric"],
                "val/primary_metric_source": validation["primary_metric_source"],
                "best_primary_metric": best_metric,
            }
            log_payload.update(
                {f"val/{key}": value for key, value in validation.items() if key.startswith("audit/")}
            )
            wandb_logger.log(log_payload, step=epoch + 1)
            print(
                f"epoch={epoch + 1:03d} lr={train_stats['lr']:.3e} loss={train_stats['loss']:.5f} "
                f"val_loss={validation['audit_loss']:.5f} AUROC={validation['auroc']:.4f} "
                f"AUPRC={validation['auprc']:.4f} FIX_F1={validation['local_fix_f1']:.4f} "
                f"REGRESS_F1={validation['local_regress_f1']:.4f} corr={validation['correlation_delta_q_delta_dice']:.4f} "
                f"global_acc={validation['improve_regress_accuracy']:.4f} "
                f"transitions={train_stats['transitions']:.0f} cf_ms={train_stats['counterfactual_ms']:.1f} "
                f"local_counts={int(validation['local_fix_count'])}/{int(validation['local_unchanged_count'])}/{int(validation['local_regress_count'])}"
            )
            print(
                f"  [on_policy] auroc={validation['audit/on_policy/auroc']:.4f} "
                f"auprc={validation['audit/on_policy/auprc']:.4f} "
                f"acc={validation['audit/on_policy/improve_regress_accuracy']:.4f} "
                f"corr={validation['audit/on_policy/correlation_delta_q_delta_dice']:.4f} "
                f"fix_f1={validation['audit/on_policy/local_fix_f1']:.4f} "
                f"regress_f1={validation['audit/on_policy/local_regress_f1']:.4f} "
                f"n={int(validation['audit/on_policy/transition_count'])}"
            )
            print(
                f"  [synthetic] auroc={validation['audit/synthetic/auroc']:.4f} "
                f"auprc={validation['audit/synthetic/auprc']:.4f} "
                f"acc={validation['audit/synthetic/improve_regress_accuracy']:.4f} "
                f"corr={validation['audit/synthetic/correlation_delta_q_delta_dice']:.4f} "
                f"fix_f1={validation['audit/synthetic/local_fix_f1']:.4f} "
                f"regress_f1={validation['audit/synthetic/local_regress_f1']:.4f} "
                f"n={int(validation['audit/synthetic/transition_count'])}"
            )
            print(
                f"  [combined] auroc={validation['audit/combined/auroc']:.4f} "
                f"n={int(validation['audit/combined/transition_count'])} | "
                f"primary_metric={validation['primary_metric']:.4f} "
                f"source={validation['primary_metric_source']} best={best_metric:.4f}"
            )
            if args.max_steps is not None:
                break

        if args.visualize:
            try:
                from self_audit.evaluation.visualizer import plot_phase_b_transitions, save_figure
                model.eval()
                sample_imgs, sample_msks, sample_prevs, sample_cands = [], [], [], []
                sample_loc_preds, sample_loc_tgts, sample_dqs, sample_dds, sample_cids = [], [], [], [], []
                with torch.no_grad():
                    for v_batch in val_loader:
                        v_batch = move_batch(v_batch, device)
                        out = model.forward_annotation(v_batch["image"]) if hasattr(model, "forward_annotation") else model(v_batch["image"])
                        feats = _feature_for_audit(model, v_batch["image"]).detach()
                        transitions = build_auditor_transitions(out, v_batch["mask"], generator)
                        for trans in transitions:
                            v_mask = trans["valid_mask"].to(device=device, dtype=torch.bool)
                            if not bool(v_mask.any()):
                                continue
                            prev = trans["previous"][v_mask]
                            cand = trans["candidate"][v_mask]
                            tgt = v_batch["mask"][v_mask]
                            targets = build_transition_targets(prev, cand, tgt)
                            aud_out = model.auditor(
                                feats[v_mask], prev, cand, cand - prev,
                                entropy_previous=_entropy(prev), entropy_candidate=_entropy(cand),
                            )
                            for idx in range(prev.shape[0]):
                                sample_imgs.append(v_batch["image"][v_mask][idx].cpu())
                                sample_msks.append(tgt[idx].cpu())
                                sample_prevs.append(prev[idx].argmax(dim=0).cpu())
                                sample_cands.append(cand[idx].argmax(dim=0).cpu())
                                sample_loc_preds.append(aud_out.local_logits[idx].argmax(dim=0).cpu())
                                sample_loc_tgts.append(targets.local[idx].cpu())
                                sample_dqs.append(float(aud_out.delta_q[idx].detach().cpu()))
                                sample_dds.append(float(targets.delta_dice[idx].detach().cpu()))
                                sample_cids.append(f"Transition_{len(sample_imgs)}")
                                if len(sample_imgs) >= max(int(args.vis_samples), 1):
                                    break
                            if len(sample_imgs) >= max(int(args.vis_samples), 1):
                                break
                        if len(sample_imgs) >= max(int(args.vis_samples), 1):
                            break
                if sample_imgs:
                    fig = plot_phase_b_transitions(
                        sample_imgs[:args.vis_samples],
                        sample_msks[:args.vis_samples],
                        sample_prevs[:args.vis_samples],
                        sample_cands[:args.vis_samples],
                        sample_loc_preds[:args.vis_samples],
                        sample_loc_tgts[:args.vis_samples],
                        delta_qs=sample_dqs[:args.vis_samples],
                        delta_dices=sample_dds[:args.vis_samples],
                        case_ids=sample_cids[:args.vis_samples],
                        title=f"Phase B Auditor Verification (Epoch {epoch+1})",
                    )
                    vis_path = Path(args.vis_dir) / "phase_b_val_transitions.png"
                    saved_img = save_figure(fig, vis_path)
                    wandb_logger.log_images({"val/phase_b_transitions": saved_img}, step=epoch + 1)
                    print(f"visualizations_saved={saved_img}")
            except Exception as vis_err:
                print(f"[Visualizer Warning] Failed to export Phase B visualization: {vis_err}")
    finally:
        wandb_logger.finish()
    print(f"saved_last={output_dir / 'last.pt'} saved_best={output_dir / 'best.pt'} saved_target={output_target}")


if __name__ == "__main__":  # pragma: no cover
    main()
