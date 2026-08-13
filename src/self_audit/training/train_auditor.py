"""Phase B: freeze annotation and train the transition auditor."""

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

from self_audit.audit.counterfactual import CounterfactualGenerator
from self_audit.audit.targets import build_transition_targets
from self_audit.losses.audit import audit_loss
from self_audit.training._utils import (
    build_model_from_config,
    build_patient_dataset,
    extract_annotation_states,
    load_config,
    move_batch,
    resolve_device,
    seed_everything,
)


def freeze_annotation_network(model: torch.nn.Module) -> None:
    """Freeze all annotation-side parameters while leaving the auditor trainable."""

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("auditor")
    if hasattr(model, "encoder"):
        model.encoder.eval()
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


def train_auditor_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    generator: CounterfactualGenerator | None = None,
) -> dict[str, float]:
    """Train on synthetic counterfactuals plus real on-policy transitions."""

    if not hasattr(model, "auditor"):
        raise AttributeError("Phase B requires model.auditor")
    model.eval()
    model.auditor.train()
    generator = generator or CounterfactualGenerator()
    total = 0.0
    steps = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.no_grad():
            output = model.forward_annotation(batch["image"]) if hasattr(model, "forward_annotation") else model(batch["image"])
            states = extract_annotation_states(output)
            previous = states[0].detach()
            candidate = states[-1].detach()
            features = _feature_for_audit(model, batch["image"]).detach()
            previous_probs = previous.softmax(dim=1)
            candidate_probs = candidate.softmax(dim=1)
            synthetic = generator.generate(previous_probs, batch["mask"])

        # The real model transition is always included. Synthetic transitions
        # make the auditor see controlled local fixes/regressions as well.
        transitions = [
            (previous_probs, candidate_probs),
            (synthetic.previous_probs, synthetic.candidate_probs),
        ]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for prev_probs, cand_probs in transitions:
            targets = build_transition_targets(prev_probs, cand_probs, batch["mask"])
            audit_output = model.auditor(
                features,
                prev_probs,
                cand_probs,
                cand_probs - prev_probs,
                entropy_previous=_entropy(prev_probs),
                entropy_candidate=_entropy(cand_probs),
            )
            losses.append(audit_loss(audit_output, targets)[0])
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        steps += 1
    return {"loss": total / max(steps, 1)}


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Audit Phase B auditor training")
    parser.add_argument("--config", default="configs/self_audit_auditor.yaml")
    parser.add_argument("--annotation_checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    device = resolve_device(args.device or config.get("device"))
    seed_everything(int(config.get("seed", 42)))
    model = build_model_from_config(config, device)
    checkpoint = torch.load(args.annotation_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
    freeze_annotation_network(model)
    auditor_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(auditor_parameters, lr=float(config.get("auditor_lr", 3e-4)))
    dataset = build_patient_dataset(config, split="train", train=False)
    loader = DataLoader(dataset, batch_size=int(args.batch_size or config.get("batch_size", 4)), shuffle=True)
    output = Path(args.output or config.get("output", "weights/self_audit/phase_b.pt"))
    for epoch in range(int(args.epochs or config.get("epochs", 20))):
        stats = train_auditor_epoch(model, loader, optimizer, device)
        print(f"epoch={epoch + 1:03d} loss={stats['loss']:.5f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, output)
    print(f"saved={output}")


if __name__ == "__main__":  # pragma: no cover
    main()
