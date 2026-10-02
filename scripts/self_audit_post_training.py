"""Post-training calibration and checkpoint-lineage compatibility helpers.

These helpers preserve historical artifact evaluation semantics without exposing
the retired multi-config training CLI. The canonical UnifiedTrainer owns its
separate, strict best-checkpoint calibration path. This is supervised-reference
calibration code, not part of the no-GT v3 teacher/student training path.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from self_audit.serialization import atomic_save_torch
from self_audit.evaluation.audit_decomposition import evaluate_audit_decomposition
from self_audit.audit.semantics import METRIC_SPACE_SLICE_PROXY
from self_audit.evaluation.calibration_lineage import (
    COHORT_ROLE_CALIBRATION, CohortPolicy, build_expected_lineage,
    verify_calibration_lineage,
)
from self_audit.evaluation.threshold import (
    load_calibration, save_calibration, select_threshold, sweep_thresholds,
)
from self_audit.provenance import CheckpointBinding, build_lineage, cohort_identity
from self_audit.training._utils import (
    bind_evaluation_checkpoint, bind_existing_evaluation_state, verify_bound_state,
)
from self_audit.training.finetune_joint import (
    collect_validation_transition_cache, validate_phase_c,
)


SELECTION_BIAS_CAVEAT = (
    "tau_accept was chosen by argmax over the threshold grid on the same validation "
    "split on which the calibrated self_audit_dice below is reported. Selecting and "
    "reporting on one split makes every 'tau_calibrated' number an optimistically "
    "biased diagnostic, not held-out evidence: it contains the selection bias of the "
    "grid search. This project has no independent test split, so no unbiased estimate "
    "of the calibrated threshold's benefit exists. The 'tau_uncalibrated' block is the "
    "one free of this bias, and it is what should be quoted."
)


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
