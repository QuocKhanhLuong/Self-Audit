"""Phase A: supervised annotation training without the audit decision loop."""

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

from self_audit.losses.annotation import annotation_loss
from self_audit.training._utils import (
    build_model_from_config,
    build_patient_dataset,
    encoder_head_optimizer,
    extract_annotation_states,
    extract_initial_logits,
    load_config,
    move_batch,
    resolve_device,
    seed_everything,
)


def phase_a_loss(output: Any, target: torch.Tensor, intermediate_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise A0 and every soft refinement state."""

    initial = extract_initial_logits(output)
    states = extract_annotation_states(output)
    logits = [initial] + [state for state in states if state is not initial]
    losses = [annotation_loss(value, target) for value in logits]
    weights = [1.0] + [float(intermediate_weight) for _ in losses[1:]]
    total = sum(weight * value[0] for weight, value in zip(weights, losses)) / max(sum(weights), 1e-8)
    parts = {f"state_{idx}": float(value[0].detach()) for idx, value in enumerate(losses)}
    parts["loss"] = float(total.detach())
    return total, parts


def train_annotation_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip: float | None = 3.0,
    intermediate_weight: float = 1.0,
) -> dict[str, float]:
    model.train()
    running = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if hasattr(model, "forward_annotation"):
            output = model.forward_annotation(batch["image"])
        else:  # pragma: no cover - compatibility fallback
            output = model(batch["image"])
        loss, parts = phase_a_loss(output, batch["mask"], intermediate_weight=intermediate_weight)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        batch_size = int(batch["image"].shape[0])
        running += float(loss.detach()) * batch_size
        count += batch_size
    return {"loss": running / max(count, 1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Audit Phase A annotation training")
    parser.add_argument("--config", default="configs/self_audit_annotation.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.image_size is not None:
        config["image_size"] = args.image_size
    device = resolve_device(args.device or config.get("device"))
    seed_everything(int(config.get("seed", 42)))
    model = build_model_from_config(config, device)
    dataset = build_patient_dataset(config, split="train", train=True)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or config.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
    )
    optimizer = encoder_head_optimizer(
        model,
        lr=float(config.get("lr", 3e-4)),
        encoder_lr=float(config.get("encoder_lr", 3e-5)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(args.epochs or config.get("epochs", 100))
    output = Path(args.output or config.get("output", "weights/self_audit/phase_a.pt"))
    for epoch in range(epochs):
        stats = train_annotation_epoch(
            model,
            loader,
            optimizer,
            device,
            grad_clip=config.get("grad_clip", 3.0),
            intermediate_weight=float(config.get("intermediate_weight", 1.0)),
        )
        print(f"epoch={epoch + 1:03d} loss={stats['loss']:.5f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, output)
    print(f"saved={output}")


if __name__ == "__main__":  # pragma: no cover
    main()
