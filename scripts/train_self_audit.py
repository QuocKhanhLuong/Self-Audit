#!/usr/bin/env python3
"""Unified Self-Audit training pipeline (Schema Version 1).

Executes the approved unified schedule across 130 epochs in a single process
and shared batch optimization loop, carrying last live weights between intervals
and resetting optimizers at epochs 100 and 120.

For historical A->B->C triple-config execution, see scripts/train_self_audit_legacy.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Re-export legacy helpers for test and lineage backwards-compatibility
from scripts.train_self_audit_legacy import (  # noqa: E402
    SELECTION_BIAS_CAVEAT,
    TAU_PRECEDENCE,
    _diagnostic_at_tau,
    _resolve_tau_accept,
    bind_post_training_checkpoint,
    collect_validation_transition_cache,
    run_post_training_calibration,
    sweep_thresholds,
)
from self_audit.training.unified_config import (  # noqa: E402
    apply_overrides,
    load_unified_config,
)
from self_audit.training.unified_trainer import UnifiedTrainer  # noqa: E402


LEGACY_FLAGS = {
    "--config_a",
    "--config_b",
    "--config_c",
    "--config_annotation",
    "--config_auditor",
    "--config_joint",
    "--epochs_a",
    "--epochs_b",
    "--epochs_c",
    "--start_phase",
}


def _parse_args() -> argparse.Namespace:
    # Reject legacy multi-config flags explicitly to enforce one-config canonical API
    passed_legacy = [
        arg.split("=", 1)[0]
        for arg in sys.argv[1:]
        if arg.split("=", 1)[0] in LEGACY_FLAGS
    ]
    if passed_legacy:
        sys.exit(
            f"Error: Legacy multi-config flags {passed_legacy} are not supported by the canonical one-config runner.\n"
            "For historical A->B->C multi-stage execution, use scripts/train_self_audit_legacy.py."
        )

    parser = argparse.ArgumentParser(
        description="Unified Self-Audit training pipeline (Schema Version 1)"
    )
    parser.add_argument(
        "--config",
        default="configs/self_audit_full.yaml",
        help="Path to Schema Version 1 unified configuration YAML.",
    )

    # Runtime overrides
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Global optimizer step budget across the entire run. Stops bounded smoke without certifying full run.",
    )
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--report_dir", default=None)
    parser.add_argument("--tau_accept", type=float, default=None)
    parser.add_argument("--skip_calibration", action="store_true")
    parser.add_argument("--resume", default=None, help="Path to checkpoint (last.pt) to resume from.")

    # W&B arguments
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=None, help="Enable Weights & Biases logging.")
    parser.add_argument(
        "--no_wandb",
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="Explicitly disable Weights & Biases logging.",
    )
    parser.add_argument("--wandb_mode", default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--no_tqdm", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Unified config file not found: {config_path}")

    config = load_unified_config(config_path)

    overrides: dict[str, Any] = {
        "data_root": args.data_root,
        "split_manifest": args.split_manifest,
        "num_workers": args.num_workers,
        "device": args.device,
        "output_dir": args.output_dir,
        "report_dir": args.report_dir,
        "tau_accept": args.tau_accept,
        "skip_calibration": args.skip_calibration,
        "wandb": args.wandb,
        "wandb_mode": args.wandb_mode,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": args.wandb_run_name,
    }
    config = apply_overrides(config, overrides)

    trainer = UnifiedTrainer(config, disable_tqdm=args.no_tqdm)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.resume_from_checkpoint(args.resume)
        print(f"Resuming unified training from epoch {start_epoch} (checkpoint: {args.resume})")

    report = trainer.train(
        start_epoch=start_epoch,
        max_steps=args.max_steps,
        max_val_batches=args.max_val_batches,
    )

    print(
        f"\nUnified training run finished: completed={report.get('completed')} "
        f"epochs_recorded={len(report.get('epochs', []))}"
    )


if __name__ == "__main__":
    main()
