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

from self_audit.provenance import build_lineage
from self_audit.training._utils import (
    bind_evaluation_checkpoint,
    build_data_loader,
    build_model_from_config,
    build_patient_dataset,
    load_config,
    resolve_device,
    seed_everything,
    validate_dataset_splits,
    verify_bound_state,
)
from self_audit.serialization import atomic_save_torch
from self_audit.training.unified_config import resolve_downstream_config
from self_audit.training.finetune_joint import collect_validation_transition_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/self_audit_full.yaml",
        help="Path to unified or historical config YAML (default: configs/self_audit_full.yaml)",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data_root", default=None, help="Override data_root path")
    parser.add_argument("--split_manifest", default=None, help="Override split manifest JSON path")
    parser.add_argument("--image_size", type=int, default=None, help="Override image spatial size")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--metric_contract",
        default="foreground_dice_exclude_v1",
        help="Metric contract name for state and transition scoring (default: foreground_dice_exclude_v1)",
    )
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bar")
    args = parser.parse_args()

    resolved = resolve_downstream_config(
        args.config,
        overrides={
            "data_root": args.data_root,
            "split_manifest": args.split_manifest,
            "image_size": args.image_size,
            "device": args.device,
            "batch_size": args.batch_size,
        },
    )
    device = resolve_device(args.device or str(resolved.device))
    seed_everything(resolved.seed, deterministic=resolved.deterministic)
    stats = resolved.validate_splits()
    print(f"split_stats={stats}")
    model = resolved.build_model(device)
    # Evaluation-only bind: strict state load, resolved-config check, no
    # optimizer/scheduler/RNG restore, and a digest that ties the cache below
    # to the weights that produced it.
    binding = bind_evaluation_checkpoint(
        model,
        [("checkpoint", args.checkpoint)],
        map_location=device,
        config=resolved.to_legacy_dict(),
    )
    print(
        f"bound checkpoint={binding.path} state_digest={binding.state_digest} "
        f"producer_git_sha={binding.producer.get('producer_git_sha')}"
    )
    dataset = resolved.build_dataset(split=resolved.val_split, train=False)
    loader = resolved.build_dataloader(dataset, train=False, batch_size=args.batch_size)
    t_max = resolved.rollout_max_turns
    verify_bound_state(model, binding, boundary="validation_transition_cache")
    cache = collect_validation_transition_cache(
        model,
        loader,
        device,
        t_max=t_max,
        disable_tqdm=args.no_tqdm,
        metric_contract=args.metric_contract,
    )
    # The cache is authoritative for metric semantics: metric_space,
    # neutral_margin, contract version and empty-class policy are read out of
    # the measurement itself, and a CLI argument that contradicts it is
    # rejected rather than stamped over it.
    cache["lineage"] = build_lineage(
        binding=binding.as_dict(),
        loader=loader,
        split_name=resolved.val_split,
        cache=cache,
        metric_contract=args.metric_contract,
        t_max=t_max,
        batch_size=getattr(loader, "batch_size", None),
        observed_samples=int(cache["initial_dice"].shape[0]),
    )
    # Last check before anything is persisted: the weights that produced these
    # rows must still be the weights the lineage names.
    verify_bound_state(model, binding, boundary="pre_write_validation_transition_cache")
    atomic_save_torch(cache, args.output)
    print(f"cached_samples={cache['initial_dice'].shape[0]} cached_turns={cache['delta_q'].shape[1]} contract={cache.get('metric_contract')} saved={args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
