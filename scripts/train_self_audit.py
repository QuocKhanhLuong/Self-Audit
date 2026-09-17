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
from self_audit.artifact_io import atomic_write_json
from self_audit.provenance import git_source_provenance
from self_audit.training.unified_config import (  # noqa: E402
    apply_overrides,
    load_unified_config,
)
from self_audit.training.unified_trainer import (  # noqa: E402
    UnifiedTrainer,
    compute_config_signature,
)
import uuid


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

def _write_pre_init_failure(
    report_dir: Path | str,
    exc: BaseException,
    *,
    stage: str = "init",
    status: str = "failed",
    config_dict: dict[str, Any] | None = None,
) -> None:
    """Best-effort minimal failure artifact write before trainer is fully initialized."""
    try:
        rdir = Path(report_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        canonical_failure_path = rdir / "failure.json"
        if canonical_failure_path.exists():
            failure_path = rdir / f"failure_attempt_{uuid.uuid4().hex[:8]}.json"
        else:
            failure_path = canonical_failure_path

        try:
            exc_msg = str(exc)
        except Exception:
            exc_msg = f"<unformattable {type(exc).__name__}>"

        cfg_sig = None
        if config_dict is not None:
            try:
                cfg_sig = compute_config_signature(config_dict)
            except Exception:
                pass

        git_sha = None
        try:
            prov = git_source_provenance()
            sha = prov.get("git_sha")
            git_sha = str(sha) if sha and sha != "UNKNOWN" else None
        except Exception:
            pass

        payload = {
            "schema_version": 1,
            "completed": False,
            "status": status,
            "failure_stage": stage,
            "current_epoch": 0,
            "global_epoch": 0,  # Alias for zero-based current global_epoch
            "completed_epochs": 0,
            "global_step": 0,
            "optimizer_step": 0,
            "exception_type": type(exc).__name__,
            "exception_message": exc_msg,
            "last_completed_validation": None,
            "last_checkpoint_committed_epoch": None,
            "run_id": None,
            "config_signature": cfg_sig,
            "config_identity": cfg_sig,  # Alias
            "recipe_signature": cfg_sig,
            "source_signature": None,
            "git_commit": git_sha,
            "git_sha": git_sha,  # Alias
            "producer_source_content_signature": None,
        }
        atomic_write_json(failure_path, payload, indent=2, sort_keys=True)
    except Exception as sec_exc:
        print(f"[lifecycle] Secondary error writing pre-init failure artifact: {sec_exc}", file=sys.stderr)


def main() -> None:
    trainer: UnifiedTrainer | None = None
    resolved_report_dir: Path | None = None

    try:
        args = _parse_args()
        if args.report_dir:
            resolved_report_dir = Path(args.report_dir)

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
        resolved_report_dir = Path(config.logging.report_dir)

        trainer = UnifiedTrainer(config, disable_tqdm=args.no_tqdm)

        start_epoch = 0
        if args.resume:
            trainer.failure_stage = "resume"
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
    except KeyboardInterrupt as exc:
        config_dict = config.to_dict() if "config" in locals() and config is not None else None
        if trainer is not None:
            try:
                trainer.record_failure(exc, failure_stage="interrupted", status="interrupted")
            except Exception as sec_exc:
                print(f"[lifecycle] Secondary error in record_failure during interrupt: {sec_exc}", file=sys.stderr)
        elif resolved_report_dir is not None:
            try:
                _write_pre_init_failure(resolved_report_dir, exc, stage="interrupted", status="interrupted", config_dict=config_dict)
            except Exception as sec_exc:
                print(f"[lifecycle] Secondary error in pre-init failure write: {sec_exc}", file=sys.stderr)
        raise
    except Exception as exc:
        config_dict = config.to_dict() if "config" in locals() and config is not None else None
        if trainer is not None:
            try:
                stage = getattr(trainer, "failure_stage", "init")
                trainer.record_failure(exc, failure_stage=stage, status="failed")
            except Exception as sec_exc:
                print(f"[lifecycle] Secondary error in record_failure: {sec_exc}", file=sys.stderr)
        elif resolved_report_dir is not None:
            try:
                _write_pre_init_failure(resolved_report_dir, exc, stage="init", status="failed", config_dict=config_dict)
            except Exception as sec_exc:
                print(f"[lifecycle] Secondary error in pre-init failure write: {sec_exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
