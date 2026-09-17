#!/usr/bin/env python
"""Launch one mask-free 150-epoch dataset run from a single resolved config.

The config file is the only source of scientific settings. CLI flags may set
operational overrides only (resume, bounded smoke/interruption limits, CPU
permission), and every override goes through the same validated dataclass.

Examples
--------
    python scripts/train_maskfree.py --config configs/maskfree_acdc_150.yaml
    python scripts/train_maskfree.py --config configs/maskfree_acdc_150.yaml --preflight
    python scripts/train_maskfree.py --config configs/maskfree_acdc_150.yaml \
        --resume runs/maskfree150/acdc/<run_id>/checkpoints/last.pt

Epoch validation observes both frozen students on the image-only development
split after every completed epoch.  It is report-only: reference masks belong
to the separate evaluator and never tune or select a checkpoint.  Use
``--no-epoch-validation`` for an explicit operational opt-out, or pass
``--epoch-reference-config`` to forward a child evaluator control-plane path;
the training process never opens that path.  No checkpoint selection is
performed.

Exit codes: 0 completed, 2 partial (bounded or interrupted), 3 failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from self_audit_maskfree.config import ConfigError, RUNTIME_FIELDS, load_config  # noqa: E402
from self_audit_maskfree.progress import TerminalProgress  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_maskfree.py",
        description=(
            "Run the mask-free anatomical label-generation pipeline for one dataset "
            "on one continuous global epoch timeline. No manual masks are read by "
            "this entry point; reference evaluation is a separate CLI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="path to the resolved YAML/JSON config")
    parser.add_argument("--resume", default=None,
                        help="checkpoint to resume exactly; scientific mismatches fail closed")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="bounded smoke only; a run with this set can never be completed")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="interruption test only; a run with this set can never be completed")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="permit CPU execution for bounded checks; not a full-run mode")
    parser.add_argument("--run-id", default=None, help="explicit immutable run identifier")
    parser.add_argument("--device", default=None, help="operational device override, e.g. cuda:0")
    parser.add_argument("--audit-device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--timing-mode", choices=("production", "diagnostic"), default=None)
    parser.add_argument("--logging-mode", choices=("sync", "buffered"), default=None)
    for field in RUNTIME_FIELDS[2:]:
        parser.add_argument("--" + field.replace("_", "-"), type=int, default=None)
    parser.add_argument(
        "--epoch-validation", dest="epoch_validation",
        action=argparse.BooleanOptionalAction, default=None,
        help=(
            "enable/disable report-only Dice observation after each completed epoch "
            "(frozen students on image-only dev; no checkpoint selection; default from config)"
        ),
    )
    parser.add_argument(
        "--epoch-reference-config", default=None,
        help=(
            "optional control-plane path forwarded to the isolated epoch evaluator; "
            "training never opens it and it never selects a checkpoint"
        ),
    )
    parser.add_argument("--preflight", action="store_true",
                        help="bounded fit/score/edit/backward/checkpoint/export gate; never completes a run")
    parser.add_argument("--print-config", action="store_true",
                        help="resolve and print the config, then exit without running")
    parser.add_argument("--progress", choices=("compact", "verbose"), default=None,
                        help="console display; default compact tqdm (or MASKFREE_PROGRESS)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with TerminalProgress(mode=args.progress) as progress:
        return _execute(args, progress)


def _execute(args: argparse.Namespace, progress: TerminalProgress) -> int:
    try:
        with progress.stage("config.load", path=args.config):
            config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 3

    overrides: dict[str, object] = {}
    if args.resume is not None:
        overrides["resume"] = args.resume
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs
    if args.allow_cpu:
        overrides["allow_cpu"] = True
    if args.run_id is not None:
        overrides["run_id"] = args.run_id
    if args.device is not None:
        overrides["device"] = args.device
    for field in ("audit_device",) + RUNTIME_FIELDS:
        if getattr(args, field, None) is not None:
            overrides[field] = getattr(args, field)
    if args.epoch_validation is not None:
        overrides["epoch_validation"] = args.epoch_validation
    if args.epoch_reference_config is not None:
        overrides["epoch_reference_config"] = args.epoch_reference_config
    try:
        config = config.replace(**overrides)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 3

    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0

    with progress.stage("imports.load", modules="torch, numpy, maskfree trainer/components"):
        from self_audit_maskfree.trainer import (
            STATUS_COMPLETED, STATUS_PARTIAL, MaskfreeTrainer, TrainerContractError,
        )
    try:
        with progress.stage("trainer.initialize", dataset=config.dataset, device=config.device):
            trainer = MaskfreeTrainer(config, repo_root=_REPO_ROOT)
    except TrainerContractError as exc:
        print(f"trainer unavailable: {exc}", file=sys.stderr)
        return 3

    progress.attach(trainer.paths.reports / ("preflight_progress.jsonl" if args.preflight else "progress.jsonl"))
    from self_audit_maskfree.resources import resource_snapshot
    resources = resource_snapshot(include_gpu=False)
    cpu = resources.get("cgroup", {})
    print(
        f"[maskfree] model={trainer.device} audit={trainer.audit_device} "
        f"precision={'fp16_amp' if config.amp else 'fp32'}/obs-fp64 "
        f"image={config.image_size} batch={config.batch_size} effective={config.effective_batch} "
        f"cpu_leaf_quota={cpu.get('leaf_cpu', {}).get('quota_cores')} "
        f"cpu_visible_upper_bound={cpu.get('visible_cpu_upper_bound_cores')} "
        f"threads={trainer.runtime_summary()['torch_threads']} candidates_workers={config.candidate_workers} "
        f"candidate_chunk={config.candidate_chunk_size} cache_bytes={config.data_cache_bytes} prefetch={config.prefetch_batches}/{config.prefetch_max_bytes}B "
        f"timing={config.timing_mode} logging={config.logging_mode}", file=sys.stderr)
    progress.update(dataset=config.dataset, run_id=trainer.run_id)
    progress.event("run.start", mode="preflight" if args.preflight else "train",
                   physical_batch=config.batch_size, accumulation=config.accumulation_steps,
                   effective_batch=config.effective_batch, epochs=config.total_epochs,
                   data_root=config.data_root, device=str(trainer.device))
    if args.preflight:
        with progress.stage("preflight", operation="fit/score/edit/backward/checkpoint/export gate"):
            receipt = trainer.preflight()
        progress.event("preflight.result", status=receipt.get("status"), report=str(trainer.paths.gate_receipt))
        if progress.mode == "verbose":
            print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
        return 0 if receipt.get("status") == "pass" else 3

    try:
        report = trainer.run()
    except Exception as exc:  # noqa: BLE001 - surfaced with the failure report path
        progress.event("run.failed", error_type=type(exc).__name__, error=str(exc),
                       failure_report=str(trainer.paths.failure_report))
        print(f"run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"failure report: {trainer.paths.failure_report}", file=sys.stderr)
        return 3

    if progress.mode == "verbose":
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    progress.event("run.result", status=report["status"], epochs_completed=report["epochs_completed"],
                   report=str(trainer.paths.pipeline_report))
    if report["status"] == STATUS_COMPLETED:
        return 0
    if report["status"] == STATUS_PARTIAL:
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
