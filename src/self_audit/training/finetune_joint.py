"""Phase C: low-LR joint fine-tuning with a detached audit contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.losses.annotation import annotation_loss
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


def joint_step(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run complete threshold-controlled inference, then supervise annotation.

    The accept/reject decision remains non-differentiable.  Auditor inputs are
    detached inside the auditor, so this phase cannot create an auditor/expert
    collusion path through the audit loss.
    """

    if hasattr(model, "forward_joint"):
        output = model.forward_joint(batch["image"])
    elif hasattr(model, "infer"):
        output = model.infer(batch["image"], mode="self_audit", tau_accept=0.0, t_max=3)
    else:  # pragma: no cover
        output = model(batch["image"])
    logits = extract_initial_logits(output)
    if isinstance(output, dict):
        logits = output.get("logits", logits)
    return annotation_loss(logits, batch["mask"])[0]


def finetune_joint_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running = 0.0
    steps = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = joint_step(model, batch)
        loss.backward()
        optimizer.step()
        running += float(loss.detach())
        steps += 1
    return {"loss": running / max(steps, 1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Audit Phase C joint fine-tuning")
    parser.add_argument("--config", default="configs/self_audit_joint.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
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
    for epoch in range(int(args.epochs or config.get("epochs", 10))):
        stats = finetune_joint_epoch(model, loader, optimizer, device)
        print(f"epoch={epoch + 1:03d} loss={stats['loss']:.5f}")


if __name__ == "__main__":  # pragma: no cover
    main()
