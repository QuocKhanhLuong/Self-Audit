"""Phase C: threshold-controlled joint fine-tuning with separated gradients."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.audit.targets import build_transition_targets
from self_audit.losses.annotation import annotation_loss
from self_audit.losses.audit import audit_loss
from self_audit.training._utils import (
    build_model_from_config,
    build_patient_dataset,
    encoder_head_optimizer,
    extract_initial_logits,
    load_config,
    move_batch,
    resolve_device,
    seed_everything,
)


def _zero_audit_term(model: torch.nn.Module, reference: torch.Tensor) -> torch.Tensor:
    """Keep the total differentiable when a hard cap produces no transition."""

    auditor = getattr(model, "auditor", None)
    if auditor is None:
        return reference.new_zeros(())
    return sum((parameter.sum() * 0.0 for parameter in auditor.parameters()), reference.new_zeros(()))


def compute_joint_losses(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    lambda_audit: float = 1.0,
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
    if isinstance(output, dict):
        previous_states = output.get("transition_previous", [])
        candidate_states = output.get("transition_candidates", output.get("candidates", []))
        audit_outputs = output.get("audits", [])
        for previous, candidate, audit_output in zip(previous_states, candidate_states, audit_outputs):
            # Targets and Auditor inputs are explicitly detached.  The Auditor
            # also defensively detaches its inputs internally.
            previous_audit = previous.detach()
            candidate_audit = candidate.detach()
            targets = build_transition_targets(previous_audit, candidate_audit, batch["mask"])
            audit_term, parts = audit_loss(audit_output, targets)
            audit_terms.append(audit_term)
            audit_parts.append(parts)
    audit_term = torch.stack(audit_terms).mean() if audit_terms else _zero_audit_term(model, annotation_term)
    total = annotation_term + float(lambda_audit) * audit_term
    details: dict[str, Any] = {
        "annotation_loss": annotation_term.detach(),
        "audit_loss": audit_term.detach(),
        "total_loss": total.detach(),
        "audit_parts": audit_parts,
        "transition_count": len(audit_terms),
        "output": output,
    }
    return total, details


def joint_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    lambda_audit: float = 1.0,
) -> torch.Tensor:
    """Backward-compatible scalar Phase-C objective."""

    return compute_joint_losses(
        model,
        batch,
        tau_accept=tau_accept,
        t_max=t_max,
        lambda_audit=lambda_audit,
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
) -> dict[str, float]:
    model.train()
    running = 0.0
    running_annotation = 0.0
    running_audit = 0.0
    transitions = 0
    steps = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, details = compute_joint_losses(
            model,
            batch,
            tau_accept=tau_accept,
            t_max=t_max,
            lambda_audit=lambda_audit,
        )
        loss.backward()
        optimizer.step()
        running += float(loss.detach())
        running_annotation += float(details["annotation_loss"])
        running_audit += float(details["audit_loss"])
        transitions += int(details["transition_count"])
        steps += 1
    divisor = max(steps, 1)
    return {
        "loss": running / divisor,
        "annotation_loss": running_annotation / divisor,
        "audit_loss": running_audit / divisor,
        "transitions": float(transitions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Audit Phase C joint fine-tuning")
    parser.add_argument("--config", default="configs/self_audit_joint.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lambda_audit", type=float, default=None)
    parser.add_argument("--tau_accept", type=float, default=None)
    parser.add_argument("--t_max", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    device = resolve_device(args.device or config.get("device"))
    seed_everything(int(config.get("seed", 42)))
    model = build_model_from_config(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
    dataset = build_patient_dataset(config, split="train", train=True)
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 2)), shuffle=True)
    optimizer = encoder_head_optimizer(
        model,
        lr=float(config.get("joint_lr", 1e-5)),
        encoder_lr=float(config.get("joint_encoder_lr", 1e-6)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    audit_config = config.get("audit", {})
    if not isinstance(audit_config, dict):
        raise ValueError("audit config must be a mapping")
    lambda_audit = float(args.lambda_audit if args.lambda_audit is not None else config.get("lambda_audit", 1.0))
    tau_accept = float(args.tau_accept if args.tau_accept is not None else audit_config.get("tau_accept", 0.0))
    t_max = int(args.t_max if args.t_max is not None else audit_config.get("t_max", config.get("model", {}).get("max_turns", 3)))
    for epoch in range(int(args.epochs or config.get("epochs", 10))):
        stats = finetune_joint_epoch(
            model,
            loader,
            optimizer,
            device,
            tau_accept=tau_accept,
            t_max=t_max,
            lambda_audit=lambda_audit,
        )
        print(
            f"epoch={epoch + 1:03d} loss={stats['loss']:.5f} "
            f"annotation={stats['annotation_loss']:.5f} audit={stats['audit_loss']:.5f} "
            f"transitions={stats['transitions']:.0f}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
