#!/usr/bin/env python3
"""Single-process A -> B -> C Self-Audit training pipeline.

Unlike scripts/run_full_pipeline.sh, this orchestrator keeps one model instance
alive across all three phases. Phase boundaries still use separate optimizers
and objectives, but weights are not reconstructed/reloaded between phases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.artifact_io import atomic_write_json, json_safe_artifact
from self_audit.serialization import atomic_save_torch
from self_audit.audit.counterfactual import CounterfactualGenerator
from self_audit.evaluation.audit_decomposition import (
    evaluate_annotation_headroom,
    evaluate_audit_decomposition,
)
from self_audit.audit.semantics import METRIC_SPACE_SLICE_PROXY
from self_audit.evaluation.calibration_lineage import (
    COHORT_ROLE_CALIBRATION,
    CohortPolicy,
    build_expected_lineage,
    verify_calibration_lineage,
)
from self_audit.evaluation.contracts import resolve_metric_contract
from self_audit.evaluation.threshold import (
    load_calibration,
    save_calibration,
    select_threshold,
    sweep_thresholds,
)
from self_audit.provenance import (
    CheckpointBinding,
    build_lineage,
    cohort_identity,
)
from self_audit.training._utils import (
    WandbLogger,
    bind_evaluation_checkpoint,
    bind_existing_evaluation_state,
    build_adamw_optimizer,
    build_data_loader,
    build_grad_scaler,
    build_model_from_config,
    build_patient_dataset,
    build_training_scheduler,
    encoder_head_optimizer,
    load_config,
    print_model_parameter_summary,
    resolve_amp,
    resolve_device,
    save_checkpoint,
    seed_everything,
    validate_accumulation_steps,
    validate_dataset_splits,
    verify_bound_state,
)
from self_audit.training.finetune_joint import (
    collect_validation_transition_cache,
    finetune_joint_epoch,
    validate_phase_c,
)
from self_audit.training.train_annotation import (
    train_annotation_epoch,
    validate_annotation_epoch,
)
from self_audit.training.train_auditor import (
    freeze_annotation_network,
    train_auditor_epoch,
    validate_auditor_epoch,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Self-Audit A->B->C in one Python process and one model instance"
    )
    parser.add_argument("--config_a", default="configs/self_audit_annotation.yaml")
    parser.add_argument("--config_b", default="configs/self_audit_auditor.yaml")
    parser.add_argument("--config_c", default="configs/self_audit_joint.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epochs_a", type=int, default=None)
    parser.add_argument("--epochs_b", type=int, default=None)
    parser.add_argument("--epochs_c", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--output_dir", default="weights/self_audit_full")
    parser.add_argument("--report_dir", default="reports/self_audit_full")
    parser.add_argument("--neutral_margin", type=float, default=None)
    parser.add_argument("--tau_accept", type=float, default=None)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--skip_calibration", action="store_true")
    parser.add_argument(
        "--calibration",
        default=None,
        help=(
            "Path of the calibration artifact (default <report_dir>/calibration.json). "
            "Written after calibration; also read at startup when --use_calibrated_tau is set."
        ),
    )
    parser.add_argument(
        "--use_calibrated_tau",
        action="store_true",
        help=(
            "Make the calibrated tau the headline tau. Opt-in only: without it the "
            "calibrated number is still reported, side by side, but never becomes the default."
        ),
    )
    parser.add_argument("--threshold_min", type=float, default=-0.02)
    parser.add_argument("--threshold_max", type=float, default=0.02)
    parser.add_argument("--threshold_steps", type=int, default=81)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_mode", default="offline")
    parser.add_argument("--wandb_project", default="self-audit")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default="full-pipeline")
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args()


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = dict(config)
    if args.data_root is not None:
        result["data_root"] = args.data_root
    if args.split_manifest is not None:
        result["split_manifest"] = args.split_manifest
    if args.num_workers is not None:
        result["num_workers"] = int(args.num_workers)
    if args.device is not None:
        result["device"] = args.device
    return result


def _architecture_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    model = dict(config.get("model", {}))
    return (
        model.get("encoder_name", "convnext_tiny"),
        int(model.get("shared_channels", 96)),
        int(model.get("num_classes", config.get("num_classes", 4))),
        int(model.get("window_k", 8)),
        int(model.get("max_turns", 3)),
    )


def _assert_compatible_configs(*configs: dict[str, Any]) -> None:
    signatures = [_architecture_signature(config) for config in configs]
    if len(set(signatures)) != 1:
        raise ValueError(f"A/B/C model architectures differ: {signatures}")


def _set_trainable(model: torch.nn.Module, *, annotation: bool, auditor: bool) -> None:
    for name, parameter in model.named_parameters():
        is_auditor = name.startswith("auditor")
        parameter.requires_grad = bool(auditor if is_auditor else annotation)


def _make_loaders(
    config: dict[str, Any],
    device: torch.device,
    *,
    train_augment: bool,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dataset = build_patient_dataset(
        config,
        split=str(config.get("train_split", "train")),
        train=train_augment,
    )
    val_dataset = build_patient_dataset(
        config,
        split=str(config.get("val_split", "val")),
        train=False,
    )
    return (
        build_data_loader(train_dataset, config, device=device, train=True),
        build_data_loader(val_dataset, config, device=device, train=False),
    )


_json_safe = json_safe_artifact


def _write_json(path: Path, payload: Any) -> Path:
    return atomic_write_json(path, payload, indent=2)


_save_report = _write_json


def _log(logger: WandbLogger, phase: str, epoch: int, metrics: dict[str, Any]) -> None:
    payload = {"pipeline/phase": phase, "pipeline/epoch": int(epoch)}
    payload.update(metrics)
    logger.log(payload)


#: Plain-words statement of why the calibrated number is not evidence.  It is
#: written verbatim into the pipeline report JSON so the caveat travels with
#: the number instead of living only in a document nobody opens.
SELECTION_BIAS_CAVEAT = (
    "tau_accept was chosen by argmax over the threshold grid on the same validation "
    "split on which the calibrated self_audit_dice below is reported. Selecting and "
    "reporting on one split makes every 'tau_calibrated' number an optimistically "
    "biased diagnostic, not held-out evidence: it contains the selection bias of the "
    "grid search. This project has no independent test split, so no unbiased estimate "
    "of the calibrated threshold's benefit exists. The 'tau_uncalibrated' block is the "
    "one free of this bias, and it is what should be quoted."
)

#: Documented precedence for the tau Phase C trains and validates at.
TAU_PRECEDENCE = (
    "--tau_accept > (--use_calibrated_tau and an existing --calibration artifact) > "
    "config_c['audit']['tau_accept'] > 0.0"
)


def _resolve_tau_accept(
    args: argparse.Namespace,
    audit_cfg: dict[str, Any],
    calibration_path: Path,
    *,
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    config_c: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    metric_contract: str,
    neutral_margin: float,
    t_max: int,
) -> tuple[float, str, dict[str, Any] | None]:
    """Resolve the tau Phase C runs at, and say where it came from.

    Precedence is :data:`TAU_PRECEDENCE`.  A calibrated tau is **never** picked
    up silently: an artifact on disk is only consulted when the caller passed
    ``--use_calibrated_tau``.

    Consulting the artifact means verifying it.  Whenever ``--use_calibrated_tau``
    is set and an artifact exists, its lineage is checked against this run's
    starting weights, protocol and cohort before Phase C uses any tau -- and
    that check runs even when ``--tau_accept`` overrides the value, so an
    explicit CLI tau cannot be used to slip an unverified artifact into the
    run.  A legacy artifact without lineage hard-fails here; it may still be
    read with ``inspect_legacy_calibration``, but never applied.
    """

    verification: dict[str, Any] | None = None
    consulted = bool(args.use_calibrated_tau) and calibration_path.exists()
    if consulted:
        payload = load_calibration(calibration_path)
        verification = _verify_calibration_for_phase_c(
            payload,
            model=model,
            val_loader=val_loader,
            config_c=config_c,
            output_dir=output_dir,
            device=device,
            metric_contract=metric_contract,
            neutral_margin=neutral_margin,
            t_max=t_max,
            artifact_path=calibration_path,
        )
    if args.tau_accept is not None:
        return float(args.tau_accept), "cli:--tau_accept", verification
    if consulted:
        return float(payload["tau_accept"]), f"calibration_artifact:{calibration_path}", verification
    if "tau_accept" in audit_cfg:
        return float(audit_cfg["tau_accept"]), "config_c.audit.tau_accept", verification
    return 0.0, "default:0.0", verification


def _verify_calibration_for_phase_c(
    payload: dict[str, Any],
    *,
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    config_c: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    metric_contract: str,
    neutral_margin: float,
    t_max: int,
    artifact_path: Path,
) -> dict[str, Any]:
    """Hard-fail unless the artifact matches the weights Phase C starts from.

    The starting state is *described*, never overwritten: this pipeline keeps
    one model alive across A -> B -> C, so loading a checkpoint here would
    discard the Phase A/B weights.  The live state must therefore already be
    the checkpoint's; if it is not, the operator is told to resume from it or
    drop ``--use_calibrated_tau`` rather than having the run silently changed.
    """

    binding = bind_existing_evaluation_state(
        model,
        [
            ("best", Path(output_dir) / "phase_c_best.pt"),
            ("last", Path(output_dir) / "phase_c_last.pt"),
        ],
        map_location=device,
        config=config_c,
    )
    expected = build_expected_lineage(
        binding=binding,
        loader=val_loader,
        split_name=str(config_c.get("val_split", "val")),
        metric_contract=metric_contract,
        metric_space=METRIC_SPACE_SLICE_PROXY,
        neutral_margin=float(neutral_margin),
        t_max=int(t_max),
        batch_size=getattr(val_loader, "batch_size", None),
    )
    record = verify_calibration_lineage(
        payload,
        expected,
        cohort_policy=CohortPolicy(role=COHORT_ROLE_CALIBRATION),
        artifact_name=f"calibration artifact {artifact_path}",
    )
    record["boundary"] = "phase_c_starting_state"
    record["starting_state_binding"] = binding.as_dict()
    print(
        f"verified calibration lineage against starting state: checkpoint={binding.path} "
        f"state_digest={binding.state_digest[:16]}... cohort_role={record['cohort']['role']}"
    )
    return record


def bind_post_training_checkpoint(
    model: torch.nn.Module,
    *,
    output_dir: Path,
    device: torch.device,
    config: dict[str, Any],
) -> CheckpointBinding:
    """Bind Phase C's selected checkpoint into ``model`` after training.

    Best is preferred; ``phase_c_last.pt`` is an explicitly declared fallback
    when no best checkpoint was written (for example a run whose validation
    metric was never finite).  With neither file present this raises rather
    than letting the calibration branch measure the live last-epoch weights
    while naming a checkpoint it never loaded.

    Phase C's best-selection criterion is untouched: this only loads what that
    criterion already chose.
    """

    return bind_evaluation_checkpoint(
        model,
        [
            ("best", Path(output_dir) / "phase_c_best.pt"),
            ("last", Path(output_dir) / "phase_c_last.pt"),
        ],
        map_location=device,
        config=config,
    )


def _diagnostic_at_tau(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    tau_accept: float,
    tau_source: str,
    t_max: int,
    neutral_margin: float,
    max_batches: int | None,
    disable_tqdm: bool,
) -> dict[str, Any]:
    """Run the full Phase-C diagnostic at one tau and return a flat summary.

    Every number is metric space
    :data:`~self_audit.audit.semantics.METRIC_SPACE_SLICE_PROXY` -- a 2-D
    per-slice training proxy, not a paper metric.
    """

    validation = validate_phase_c(
        model,
        loader,
        device,
        tau_accept=float(tau_accept),
        t_max=int(t_max),
        max_batches=max_batches,
        disable_tqdm=disable_tqdm,
    )
    decomposition = evaluate_audit_decomposition(
        model,
        loader,
        device,
        tau_accept=float(tau_accept),
        t_max=int(t_max),
        neutral_margin=float(neutral_margin),
        max_batches=max_batches,
        disable_tqdm=disable_tqdm,
    )
    nan = float("nan")
    return {
        "tau_accept": float(tau_accept),
        "tau_source": str(tau_source),
        "metric_space": METRIC_SPACE_SLICE_PROXY,
        "initial_dice": float(decomposition.get("modes/initial_dice", nan)),
        "always_accept_dice": float(decomposition.get("modes/always_accept_dice", nan)),
        "self_audit_dice": float(decomposition.get("modes/self_audit_dice", nan)),
        "oracle_dice": float(decomposition.get("modes/oracle_dice", nan)),
        "audit_rescue_vs_always": float(decomposition.get("modes/audit_rescue_vs_always", nan)),
        "oracle_headroom": float(decomposition.get("modes/oracle_headroom", nan)),
        "harmful_acceptance_rate": float(validation.get("harmful_acceptance_rate", nan)),
        "beneficial_rejection_rate": float(validation.get("beneficial_rejection_rate", nan)),
        "validation": validation,
        "decomposition": decomposition,
    }


def _scheduler_and_amp(
    model: torch.nn.Module,
    config: dict[str, Any],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    encoder_lr: float,
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler | None,
    Any,
    bool,
    torch.dtype,
    int,
]:
    optimizer = encoder_head_optimizer(
        model,
        lr=float(lr),
        encoder_lr=float(encoder_lr),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    accumulation = validate_accumulation_steps(config.get("gradient_accumulation_steps", 1))
    scheduler_config = dict(config)
    scheduler_config["accumulation_steps"] = accumulation
    scheduler = build_training_scheduler(
        optimizer,
        scheduler_config,
        num_batches=len(loader),
        epochs=int(epochs),
    )
    amp_enabled, amp_dtype = resolve_amp(config, device)
    scaler = build_grad_scaler(enabled=amp_enabled, device=device, dtype=amp_dtype)
    return optimizer, scheduler, scaler, amp_enabled, amp_dtype, accumulation


def run_post_training_calibration(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    args: argparse.Namespace,
    config_c: dict[str, Any],
    output_dir: Path,
    report_dir: Path,
    calibration_path: Path,
    report: dict[str, Any],
    tau_accept: float,
    tau_source: str,
    t_max: int,
    neutral_margin_c: float,
    headline_tau: float,
    headline_tau_source: str,
) -> tuple[float, str]:
    """Run the post-training branch: bind, cache, calibrate, diagnose.

    This is the whole Phase-C tail of :func:`main`, extracted so it can be
    exercised end to end on a tiny synthetic fixture without training.  The
    checkpoint binding at the top is what makes every measurement below refer
    to the same, named weights; ``report`` is mutated in place exactly as the
    runner expects and the resolved headline tau is returned.
    """

    print("\n=== CALIBRATION: validation transitions only ===")
    # Bind the selected checkpoint into the live model BEFORE anything is
    # measured.  Everything below -- cache, calibration, both diagnostics
    # -- therefore measures the weights the artifact names.
    binding = bind_post_training_checkpoint(
        model,
        output_dir=output_dir,
        device=device,
        config=config_c,
    )
    report["checkpoint_binding"] = binding.as_dict()
    print(
        f"bound checkpoint={binding.path} role={binding.role} "
        f"fallback={int(binding.fallback_used)} "
        f"state_digest={binding.state_digest[:16]}... "
        f"producer_git_sha={binding.producer.get('producer_git_sha')}"
    )
    verify_bound_state(model, binding, boundary="validation_transition_cache")
    cache = collect_validation_transition_cache(
        model,
        val_loader,
        device,
        t_max=t_max,
        disable_tqdm=args.no_tqdm,
    )
    thresholds = np.linspace(
        float(args.threshold_min),
        float(args.threshold_max),
        int(args.threshold_steps),
    )
    lineage = build_lineage(
        binding=binding.as_dict(),
        loader=val_loader,
        split_name=str(config_c.get("val_split", "val")),
        # The collected cache is authoritative for metric semantics; the two
        # values below are passed as assertions and are rejected if they
        # disagree with the contract the cache was actually collected under.
        cache=cache,
        metric_space=METRIC_SPACE_SLICE_PROXY,
        neutral_margin=neutral_margin_c,
        t_max=t_max,
        max_batches=None,
        batch_size=getattr(val_loader, "batch_size", None),
        observed_samples=int(cache["initial_dice"].shape[0]),
    )
    cache["lineage"] = lineage
    verify_bound_state(model, binding, boundary="calibration_sweep")
    rows = sweep_thresholds(cache, thresholds, neutral_margin=neutral_margin_c)
    best_threshold = select_threshold(rows)
    calibrated_tau = float(best_threshold["tau_accept"])
    # The calibrated artifact names the checkpoint that was actually bound
    # and measured, never a same-named file selected independently.
    checkpoint_for_calibration = binding.path
    calibration_payload = save_calibration(
        calibration_path,
        tau_accept=calibrated_tau,
        neutral_margin=neutral_margin_c,
        source_split=str(config_c.get("val_split", "val")),
        checkpoint_path=checkpoint_for_calibration,
        t_max=t_max,
        threshold_grid=thresholds,
        selected_row=best_threshold,
        metric_space=METRIC_SPACE_SLICE_PROXY,
        lineage=lineage,
        extra={
            "pipeline": "scripts/train_self_audit.py",
            "empty_class_policy": cache.get("empty_class_policy"),
            "excluded_empty_slice_count": cache.get("excluded_empty_slice_count"),
            "phase_c_tau_accept": float(tau_accept),
            "phase_c_tau_source": tau_source,
            "lineage": lineage,
        },
    )
    # Round-trip: the tau the final evaluation runs at is read back off
    # disk, never carried in a live variable, so the artifact is proved to
    # be the thing that is actually consumed.
    reloaded = load_calibration(calibration_path)
    loaded_tau = float(reloaded["tau_accept"])
    if loaded_tau.hex() != calibrated_tau.hex():
        raise ValueError(
            "Calibration round-trip mismatch: selected "
            f"{calibrated_tau.hex()} but loaded {loaded_tau.hex()} from {calibration_path}"
        )
    # The artifact just written is verified the way a deployable consumer will
    # verify it -- against lineage rebuilt from these live runtime objects --
    # before the calibrated diagnostic is allowed to use its tau.
    expected_lineage = build_expected_lineage(
        binding=binding,
        loader=val_loader,
        split_name=str(config_c.get("val_split", "val")),
        metric_contract=cache.get("metric_contract"),
        metric_space=METRIC_SPACE_SLICE_PROXY,
        neutral_margin=neutral_margin_c,
        t_max=t_max,
        batch_size=getattr(val_loader, "batch_size", None),
        observed_samples=int(cache["initial_dice"].shape[0]),
    )
    roundtrip_verification = verify_calibration_lineage(
        reloaded,
        expected_lineage,
        cohort_policy=CohortPolicy(role=COHORT_ROLE_CALIBRATION),
        artifact_name=f"calibration artifact {calibration_path}",
    )
    roundtrip_verification["boundary"] = "post_save_roundtrip"
    report["calibration"] = {
        "best": best_threshold,
        "threshold_min": float(args.threshold_min),
        "threshold_max": float(args.threshold_max),
        "threshold_steps": int(args.threshold_steps),
        "artifact_path": str(calibration_path),
        "artifact": calibration_payload,
        "round_trip_tau_hex": loaded_tau.hex(),
        "lineage_verification": roundtrip_verification,
    }
    # Last check before the cache is persisted: the bound weights must still be
    # the weights its lineage names.
    verify_bound_state(model, binding, boundary="pre_write_validation_transition_cache")
    atomic_save_torch(cache, report_dir / "validation_transitions.pt")
    print(
        f"tau={calibrated_tau:+.5f} "
        f"final={best_threshold['final_macro_dice']:.4f} "
        f"net_gain={best_threshold['net_dice_gain']:+.5f}"
    )

    print("\n=== FINAL DIAGNOSTIC: both taus, same split, same checkpoint ===")
    verify_bound_state(model, binding, boundary="diagnostic_uncalibrated")
    uncalibrated = _diagnostic_at_tau(
        model,
        val_loader,
        device,
        tau_accept=tau_accept,
        tau_source=tau_source,
        t_max=t_max,
        neutral_margin=neutral_margin_c,
        max_batches=args.max_val_batches,
        disable_tqdm=args.no_tqdm,
    )
    verify_bound_state(model, binding, boundary="diagnostic_calibrated")
    calibrated = _diagnostic_at_tau(
        model,
        val_loader,
        device,
        tau_accept=loaded_tau,
        tau_source=f"calibration_artifact:{calibration_path}",
        t_max=t_max,
        neutral_margin=neutral_margin_c,
        max_batches=args.max_val_batches,
        disable_tqdm=args.no_tqdm,
    )
    if bool(args.use_calibrated_tau):
        headline_tau = loaded_tau
        headline_tau_source = f"calibration_artifact:{calibration_path}"
    report["tau"]["calibrated_tau_accept"] = loaded_tau
    report["tau"]["headline_tau_accept"] = float(headline_tau)
    report["tau"]["headline_tau_source"] = headline_tau_source
    report["final_diagnostic"] = {
        "metric_space": METRIC_SPACE_SLICE_PROXY,
        "source_split": str(config_c.get("val_split", "val")),
        "checkpoint_binding": binding.as_dict(),
        # ``--max_val_batches`` truncates the diagnostic; the cohort record
        # says so instead of letting a subset read as the full split.
        "cohort": cohort_identity(
            val_loader,
            split_name=str(config_c.get("val_split", "val")),
            max_batches=args.max_val_batches,
            batch_size=getattr(val_loader, "batch_size", None),
        ),
        "tau_uncalibrated": uncalibrated,
        "tau_calibrated": calibrated,
        "headline_tau_accept": float(headline_tau),
        "headline_tau_source": headline_tau_source,
        "selection_bias_caveat": SELECTION_BIAS_CAVEAT,
    }
    print(
        f"self_audit_dice @ tau={uncalibrated['tau_accept']:+.5f} (uncalibrated, "
        f"source={uncalibrated['tau_source']}) = {uncalibrated['self_audit_dice']:.4f}"
    )
    print(
        f"self_audit_dice @ tau={calibrated['tau_accept']:+.5f} (calibrated on this same "
        f"split -> optimistically biased) = {calibrated['self_audit_dice']:.4f}"
    )
    print(f"headline_tau={headline_tau:+.6f} source={headline_tau_source}")
    print(f"selection_bias_caveat: {SELECTION_BIAS_CAVEAT}")
    return float(headline_tau), str(headline_tau_source)


def main() -> None:
    args = _parse_args()
    config_a = _apply_overrides(load_config(args.config_a), args)
    config_b = _apply_overrides(load_config(args.config_b), args)
    config_c = _apply_overrides(load_config(args.config_c), args)
    _assert_compatible_configs(config_a, config_b, config_c)

    for name, config in (("A", config_a), ("B", config_b), ("C", config_c)):
        stats = validate_dataset_splits(config)
        print(f"phase_{name.lower()}_split_stats={stats}")

    device = resolve_device(args.device or config_a.get("device"))
    seed_everything(
        int(config_a.get("seed", 42)),
        deterministic=bool(config_a.get("deterministic", False)),
    )

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(config_a, device)
    print_model_parameter_summary(model, title="Full Pipeline: Single Shared Model")

    logger = WandbLogger(
        enabled=bool(args.wandb),
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        config={"phase_a": config_a, "phase_b": config_b, "phase_c": config_c},
        mode=args.wandb_mode,
    )

    report: dict[str, Any] = {
        "protocol": {
            "single_process": True,
            "single_model_instance": True,
            "gt_policy": (
                "GT supervises annotation/auditor targets during training and is used "
                "after inference for validation decomposition; deployable self_audit "
                "decisions receive no GT."
            ),
        },
        "phase_a": [],
        "phase_b": [],
        "phase_c": [],
    }

    print("\n=== PHASE A: annotation bootstrap ===")
    _set_trainable(model, annotation=True, auditor=False)
    train_a, val_a = _make_loaders(config_a, device, train_augment=True)
    epochs_a = int(args.epochs_a or config_a.get("epochs", 100))
    opt_a, sched_a, scaler_a, amp_a, dtype_a, accum_a = _scheduler_and_amp(
        model,
        config_a,
        train_a,
        device,
        epochs=epochs_a,
        lr=float(config_a.get("lr", 3e-4)),
        encoder_lr=float(config_a.get("encoder_lr", 3e-5)),
    )
    best_a = -float("inf")
    stage_weights = config_a.get("stage_weights")
    for epoch in range(epochs_a):
        train_stats = train_annotation_epoch(
            model,
            train_a,
            opt_a,
            device,
            grad_clip=config_a.get("grad_clip", 3.0),
            stage_weights=stage_weights,
            scheduler=sched_a,
            scaler=scaler_a,
            amp_enabled=amp_a,
            amp_dtype=dtype_a,
            gradient_accumulation_steps=accum_a,
            epoch=epoch,
            total_epochs=epochs_a,
            max_steps=args.max_steps,
            disable_tqdm=args.no_tqdm,
        )
        val_stats = validate_annotation_epoch(
            model,
            val_a,
            device,
            stage_weights=stage_weights,
            max_batches=args.max_val_batches,
            epoch=epoch,
            total_epochs=epochs_a,
            disable_tqdm=args.no_tqdm,
        )
        headroom = evaluate_annotation_headroom(
            model,
            val_a,
            device,
            neutral_margin=float(
                args.neutral_margin
                if args.neutral_margin is not None
                else config_b.get("counterfactual", {}).get("neutral_margin", 0.005)
            ),
            max_batches=args.max_val_batches,
            disable_tqdm=args.no_tqdm,
        )
        row = {"epoch": epoch + 1, **train_stats, **val_stats, **headroom}
        report["phase_a"].append(row)
        _log(logger, "A", epoch + 1, {f"phase_a/{k}": v for k, v in train_stats.items()})
        _log(logger, "A", epoch + 1, val_stats)
        _log(logger, "A", epoch + 1, headroom)
        metric = float(val_stats["val_macro_foreground_dice"])
        save_checkpoint(
            output_dir / "phase_a_last.pt",
            model,
            optimizer=opt_a,
            scheduler=sched_a,
            scaler=scaler_a,
            epoch=epoch + 1,
            config=config_a,
            extra={"best_metric": max(best_a, metric), "phase": "annotation"},
        )
        if metric > best_a:
            best_a = metric
            save_checkpoint(
                output_dir / "phase_a_best.pt",
                model,
                optimizer=opt_a,
                scheduler=sched_a,
                scaler=scaler_a,
                epoch=epoch + 1,
                config=config_a,
                extra={"best_metric": best_a, "phase": "annotation"},
            )
        print(
            f"A epoch={epoch+1:03d} val_dice={metric:.4f} "
            f"A0={headroom.get('phase_a/a0_dice', float('nan')):.4f} "
            f"refine_gain={headroom.get('phase_a/total_refinement_gain', float('nan')):+.5f} "
            f"headroom={headroom.get('phase_a/mean_stage_headroom', float('nan')):.5f}"
        )
        if args.max_steps is not None:
            break

    print("\n=== PHASE B: counterfactual auditor ===")
    freeze_annotation_network(model)
    train_b, val_b = _make_loaders(config_b, device, train_augment=False)
    epochs_b = int(args.epochs_b or config_b.get("epochs", 20))
    auditor_parameters = [p for p in model.parameters() if p.requires_grad]
    opt_b = build_adamw_optimizer(
        auditor_parameters,
        lr=float(config_b.get("auditor_lr", 3e-4)),
        weight_decay=float(config_b.get("weight_decay", 1e-4)),
    )
    accum_b = validate_accumulation_steps(config_b.get("gradient_accumulation_steps", 1))
    sched_cfg_b = dict(config_b)
    sched_cfg_b["accumulation_steps"] = accum_b
    sched_b = build_training_scheduler(opt_b, sched_cfg_b, num_batches=len(train_b), epochs=epochs_b)
    amp_b, dtype_b = resolve_amp(config_b, device)
    scaler_b = build_grad_scaler(enabled=amp_b, device=device, dtype=dtype_b)
    cf_cfg = dict(config_b.get("counterfactual", {}))
    neutral_margin = float(
        args.neutral_margin
        if args.neutral_margin is not None
        else cf_cfg.get("neutral_margin", 0.005)
    )
    generator = CounterfactualGenerator(
        epsilon_neutral=float(cf_cfg.get("epsilon_neutral", 0.02)),
        neutral_max_retries=int(cf_cfg.get("neutral_max_retries", 8)),
        num_classes=int(config_b.get("num_classes", 4)),
    )
    best_b = -float("inf")
    for epoch in range(epochs_b):
        train_stats = train_auditor_epoch(
            model,
            train_b,
            opt_b,
            device,
            generator=generator,
            scheduler=sched_b,
            scaler=scaler_b,
            amp_enabled=amp_b,
            amp_dtype=dtype_b,
            gradient_accumulation_steps=accum_b,
            epoch=epoch,
            total_epochs=epochs_b,
            max_steps=args.max_steps,
            neutral_margin=neutral_margin,
            local_weighting=cf_cfg.get("local_class_weighting", True) != "none",
            disable_tqdm=args.no_tqdm,
        )
        val_stats = validate_auditor_epoch(
            model,
            val_b,
            device,
            generator=generator,
            neutral_margin=neutral_margin,
            local_weighting=cf_cfg.get("local_class_weighting", True) != "none",
            amp_enabled=amp_b,
            amp_dtype=dtype_b,
            max_batches=args.max_val_batches,
            epoch=epoch,
            total_epochs=epochs_b,
            disable_tqdm=args.no_tqdm,
        )
        row = {"epoch": epoch + 1, **train_stats, **val_stats}
        report["phase_b"].append(row)
        _log(logger, "B", epoch + 1, {f"phase_b/{k}": v for k, v in row.items() if k != "epoch"})
        metric = float(val_stats["primary_metric"])
        save_checkpoint(
            output_dir / "phase_b_last.pt",
            model,
            optimizer=opt_b,
            scheduler=sched_b,
            scaler=scaler_b,
            epoch=epoch + 1,
            config=config_b,
            extra={"best_metric": max(best_b, metric), "phase": "auditor"},
        )
        if np.isfinite(metric) and metric > best_b:
            best_b = metric
            save_checkpoint(
                output_dir / "phase_b_best.pt",
                model,
                optimizer=opt_b,
                scheduler=sched_b,
                scaler=scaler_b,
                epoch=epoch + 1,
                config=config_b,
                extra={"best_metric": best_b, "phase": "auditor"},
            )
        print(
            f"B epoch={epoch+1:03d} AUROC={val_stats['auroc']:.4f} "
            f"corr={val_stats['correlation_delta_q_delta_dice']:.4f} "
            f"FIX_F1={val_stats['local_fix_f1']:.4f} "
            f"REGRESS_F1={val_stats['local_regress_f1']:.4f}"
        )
        if args.max_steps is not None:
            break

    print("\n=== PHASE C: joint low-LR self-audit ===")
    _set_trainable(model, annotation=True, auditor=True)
    train_c, val_c = _make_loaders(config_c, device, train_augment=True)
    epochs_c = int(args.epochs_c or config_c.get("epochs", 10))
    opt_c, sched_c, scaler_c, amp_c, dtype_c, accum_c = _scheduler_and_amp(
        model,
        config_c,
        train_c,
        device,
        epochs=epochs_c,
        lr=float(config_c.get("joint_lr", 1e-5)),
        encoder_lr=float(config_c.get("joint_encoder_lr", 1e-6)),
    )
    audit_cfg = dict(config_c.get("audit", {}))
    calibration_path = Path(args.calibration) if args.calibration else report_dir / "calibration.json"
    # The evaluation protocol is settled before any tau is consulted: verifying
    # a calibration artifact requires knowing the semantics it must match.
    t_max = int(
        args.t_max
        if args.t_max is not None
        else audit_cfg.get("t_max", config_c.get("model", {}).get("max_turns", 3))
    )
    neutral_margin_c = float(
        args.neutral_margin
        if args.neutral_margin is not None
        else audit_cfg.get("neutral_margin", neutral_margin)
    )
    metric_contract_c = resolve_metric_contract(audit_cfg.get("metric_contract")).name
    tau_accept, tau_source, calibration_verification = _resolve_tau_accept(
        args,
        audit_cfg,
        calibration_path,
        model=model,
        val_loader=val_c,
        config_c=config_c,
        output_dir=output_dir,
        device=device,
        metric_contract=metric_contract_c,
        neutral_margin=neutral_margin_c,
        t_max=t_max,
    )
    if calibration_verification is not None:
        report["calibration_artifact_verification"] = calibration_verification
    print(f"tau_accept={tau_accept:+.6f} source={tau_source} precedence={TAU_PRECEDENCE}")
    lambda_audit = float(config_c.get("lambda_audit", 1.0))
    best_c = -float("inf")
    for epoch in range(epochs_c):
        train_stats = finetune_joint_epoch(
            model,
            train_c,
            opt_c,
            device,
            tau_accept=tau_accept,
            t_max=t_max,
            lambda_audit=lambda_audit,
            neutral_margin=neutral_margin_c,
            local_weighting=audit_cfg.get("local_class_weighting", True) != "none",
            scheduler=sched_c,
            scaler=scaler_c,
            amp_enabled=amp_c,
            amp_dtype=dtype_c,
            gradient_accumulation_steps=accum_c,
            grad_clip=config_c.get("grad_clip", 3.0),
            epoch=epoch,
            total_epochs=epochs_c,
            max_steps=args.max_steps,
            disable_tqdm=args.no_tqdm,
        )
        val_stats = validate_phase_c(
            model,
            val_c,
            device,
            tau_accept=tau_accept,
            t_max=t_max,
            max_batches=args.max_val_batches,
            epoch=epoch,
            total_epochs=epochs_c,
            disable_tqdm=args.no_tqdm,
        )
        decomposition = evaluate_audit_decomposition(
            model,
            val_c,
            device,
            tau_accept=tau_accept,
            t_max=t_max,
            neutral_margin=neutral_margin_c,
            max_batches=args.max_val_batches,
            disable_tqdm=args.no_tqdm,
        )
        row = {"epoch": epoch + 1, **train_stats, **val_stats, **decomposition}
        report["phase_c"].append(row)
        _log(logger, "C", epoch + 1, {f"phase_c/{k}": v for k, v in train_stats.items()})
        _log(logger, "C", epoch + 1, val_stats)
        _log(logger, "C", epoch + 1, decomposition)

        metric = float(val_stats["final_foreground_macro_dice"])
        save_checkpoint(
            output_dir / "phase_c_last.pt",
            model,
            optimizer=opt_c,
            scheduler=sched_c,
            scaler=scaler_c,
            epoch=epoch + 1,
            config=config_c,
            extra={"best_metric": max(best_c, metric), "phase": "joint"},
        )
        if np.isfinite(metric) and metric > best_c:
            best_c = metric
            save_checkpoint(
                output_dir / "phase_c_best.pt",
                model,
                optimizer=opt_c,
                scheduler=sched_c,
                scaler=scaler_c,
                epoch=epoch + 1,
                config=config_c,
                extra={"best_metric": best_c, "phase": "joint"},
            )
        print(
            f"C epoch={epoch+1:03d} initial={decomposition.get('modes/initial_dice', float('nan')):.4f} "
            f"always={decomposition.get('modes/always_accept_dice', float('nan')):.4f} "
            f"self={decomposition.get('modes/self_audit_dice', float('nan')):.4f} "
            f"oracle={decomposition.get('modes/oracle_dice', float('nan')):.4f} "
            f"audit_rescue={decomposition.get('modes/audit_rescue_vs_always', float('nan')):+.5f} "
            f"oracle_headroom={decomposition.get('modes/oracle_headroom', float('nan')):+.5f} "
            f"GT_firewall={int(decomposition.get('gt_firewall/passed', 0.0))}"
        )
        if args.max_steps is not None:
            break

    headline_tau = tau_accept
    headline_tau_source = tau_source
    report["tau"] = {
        "precedence": TAU_PRECEDENCE,
        "phase_c_tau_accept": float(tau_accept),
        "phase_c_tau_source": tau_source,
        "calibration_artifact": str(calibration_path),
        "use_calibrated_tau": bool(args.use_calibrated_tau),
        "calibrated_tau_accept": None,
        "headline_tau_accept": float(headline_tau),
        "headline_tau_source": headline_tau_source,
        "selection_bias_caveat": SELECTION_BIAS_CAVEAT,
    }

    if not args.skip_calibration:
        headline_tau, headline_tau_source = run_post_training_calibration(
            model,
            val_c,
            device,
            args=args,
            config_c=config_c,
            output_dir=output_dir,
            report_dir=report_dir,
            calibration_path=calibration_path,
            report=report,
            tau_accept=tau_accept,
            tau_source=tau_source,
            t_max=t_max,
            neutral_margin_c=neutral_margin_c,
            headline_tau=headline_tau,
            headline_tau_source=headline_tau_source,
        )
    _save_report(report_dir / "pipeline_report.json", report)
    logger.finish()
    print(f"\nSaved checkpoints: {output_dir}")
    print(f"Saved report: {report_dir / 'pipeline_report.json'}")


if __name__ == "__main__":
    main()
