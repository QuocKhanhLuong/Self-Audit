#!/usr/bin/env python3
"""Cache validation transitions for validation-only tau_accept calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.training._utils import (
    build_data_loader,
    build_model_from_config,
    build_patient_dataset,
    load_checkpoint,
    load_config,
    resolve_device,
    seed_everything,
    validate_dataset_splits,
)
from self_audit.training.finetune_joint import collect_validation_transition_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/self_audit_joint.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config.get("device"))
    seed_everything(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", False)))
    stats = validate_dataset_splits(config)
    print(f"split_stats={stats}")
    model = build_model_from_config(config, device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    dataset = build_patient_dataset(config, split=str(config.get("val_split", "val")), train=False)
    loader = build_data_loader(dataset, config, device=device, train=False, batch_size=args.batch_size)
    audit_config = config.get("audit", {})
    t_max = int(audit_config.get("t_max", config.get("model", {}).get("max_turns", 3))) if isinstance(audit_config, dict) else 3
    cache = collect_validation_transition_cache(model, loader, device, t_max=t_max)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)
    print(f"cached_samples={cache['initial_dice'].shape[0]} cached_turns={cache['delta_q'].shape[1]} saved={args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
