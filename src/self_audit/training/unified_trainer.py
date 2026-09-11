"""Unified Self-Audit trainer executing the approved single-process schedule.

Features:
- Single global epoch loop across all 130 epochs.
- Single shared batch optimization loop across all intervals.
- Carrying live weights across boundaries, resetting optimizers at epochs 100 and 120.
- Strict trainability management: frozen modules have requires_grad=False, eval() mode,
  and are excluded from optimizer parameter groups.
- best.pt selection restricted strictly to global_epoch >= 120 (Phase C).
- Resumable from last.pt with full RNG, global step, and update counter lineage.
- Bounded smoke runs (max_steps) marked incomplete and barred from calibration.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
from collections.abc import Iterator
from numbers import Integral
from typing import Any, Mapping
import uuid

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from self_audit.artifact_io import atomic_write_json, json_safe_artifact
from self_audit.serialization import atomic_save_torch
from self_audit.audit.counterfactual import CounterfactualGenerator
from self_audit.audit.semantics import METRIC_SPACE_SLICE_PROXY
from self_audit.evaluation.audit_decomposition import (
    evaluate_annotation_headroom,
    evaluate_audit_decomposition,
)
from self_audit.evaluation.calibration_lineage import (
    COHORT_ROLE_CALIBRATION,
    CohortPolicy,
    build_expected_lineage,
    verify_calibration_lineage,
)
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
    git_source_provenance,
    source_content_signature,
    split_membership_descriptor,
    state_digest,
)
from self_audit.training._utils import (
    COMMITTED_BEST_REFERENCE_KEY,
    WandbLogger,
    _restore_rng_state,
    _rng_state,
    autocast_context,
    bind_evaluation_checkpoint,
    build_data_loader,
    build_grad_scaler,
    build_model_from_config,
    build_patient_dataset,
    build_warmup_cosine_scheduler,
    finalize_optimizer_step,
    is_finite,
    load_checkpoint,
    move_batch,
    print_model_parameter_summary,
    resolve_amp,
    resolve_device,
    save_checkpoint,
    seed_everything,
    validate_dataset_splits,
    verify_bound_state,
)
from self_audit.training.checkpoint_commit import (
    PUBLIC_BEST_NAME,
    SELECTED_BEST_DIRNAME,
    SelectionError,
    best_alias_matches_reference,
    make_best_reference,
    publish_best_alias,
    relocate_best_reference,
    resolve_best_reference,
)
from self_audit.training.finetune_joint import (
    ORDINARY_REAL_EVIDENCE_KEY,
    PREDICTED_HISTORY_COUNTER_KEYS,
    accumulate_rollout_counters,
    compute_joint_losses,
    compute_predicted_history_annotation_loss,
    compute_predicted_history_audit_loss,
    collect_validation_transition_cache,
    rollout_counters,
    validate_phase_c,
    zero_rollout_counters,
)
from self_audit.training.schedule import Schedule, ScheduleInterval
from self_audit.training.train_annotation import (
    phase_a_loss,
    validate_annotation_epoch,
)
from self_audit.training.train_auditor import (
    _auditor_batch,
    validate_auditor_epoch,
)
from self_audit.training.unified_config import UnifiedConfig


SELECTION_BIAS_CAVEAT = (
    "tau_accept was chosen by argmax over the threshold grid on the same validation "
    "split on which the calibrated self_audit_dice below is reported. Selecting and "
    "reporting on one split makes every 'tau_calibrated' number an optimistically "
    "biased diagnostic, not held-out evidence: it contains the selection bias of the "
    "grid search. This project has no independent test split, so no unbiased estimate "
    "of the calibrated threshold's benefit exists. The 'tau_uncalibrated' block is the "
    "one free of this bias, and it is what should be quoted."
)


_json_safe = json_safe_artifact


#: The only metric the gated interval may select on.  Other intervals report a
#: ``primary_metric`` that is a semantically different quantity, so it is never
#: substituted for a missing Dice.
SELECTION_METRIC_KEY = "final_foreground_macro_dice"


def _strict_counter(payload: Mapping[str, Any], key: str, *, context: str) -> int:
    """Read a required non-negative integer counter, refusing coercion.

    ``int(payload.get(key, 0))`` would accept a missing key as 0, truncate a
    fractional value and coerce a numeric string -- each of which silently
    invents a resume position.
    """

    if key not in payload:
        raise ValueError(f"Checkpoint missing required counter '{key}': {context}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, Integral)):
        raise ValueError(
            f"Checkpoint counter '{key}' must be an integer, got {type(value).__name__}: {value!r} ({context})"
        )
    value = int(value)
    if value < 0:
        raise ValueError(f"Checkpoint counter '{key}' must be non-negative, got {value} ({context})")
    return value


def compute_config_signature(config_dict: Mapping[str, Any]) -> str:
    """Deterministic SHA256 digest of full resolved execution config (excluding output/report destinations)."""
    cfg_copy = copy.deepcopy(dict(config_dict))
    if "checkpoint" in cfg_copy and isinstance(cfg_copy["checkpoint"], dict):
        cfg_copy["checkpoint"].pop("output_dir", None)
    if "logging" in cfg_copy and isinstance(cfg_copy["logging"], dict):
        cfg_copy["logging"].pop("report_dir", None)
        cfg_copy["logging"].pop("wandb", None)
    serialized = json.dumps(_json_safe(cfg_copy), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compare_execution_configs(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Strictly compare saved execution configuration against current configuration.

    Allows ONLY output and logging destinations (checkpoint.output_dir, logging.report_dir,
    logging.wandb) to differ. Changing LR, schedule, data root, split manifest, class mapping,
    seed, AMP, architecture, etc. raises ValueError.
    """
    # 1. Model num_classes specific check for backwards-compatibility with existing tests
    saved_model = saved.get("model", {}) if isinstance(saved, Mapping) else {}
    curr_model = current.get("model", {}) if isinstance(current, Mapping) else {}
    if "num_classes" in saved_model and "num_classes" in curr_model:
        if saved_model["num_classes"] != curr_model["num_classes"]:
            raise ValueError(
                f"Config mismatch on resume: saved num_classes={saved_model['num_classes']}, "
                f"current={curr_model['num_classes']}"
            )

    # 2. Compare full config excluding output and logging destinations
    saved_copy = copy.deepcopy(dict(saved))
    curr_copy = copy.deepcopy(dict(current))

    for c in (saved_copy, curr_copy):
        if "checkpoint" in c and isinstance(c["checkpoint"], dict):
            c["checkpoint"].pop("output_dir", None)
        if "logging" in c and isinstance(c["logging"], dict):
            c["logging"].pop("report_dir", None)
            c["logging"].pop("wandb", None)

    # Recursive key comparison
    def _diff_keys(d1: Any, d2: Any, prefix: str = "") -> None:
        if isinstance(d1, dict) and isinstance(d2, dict):
            all_keys = set(d1.keys()).union(d2.keys())
            for k in sorted(all_keys):
                pfx = f"{prefix}.{k}" if prefix else k
                if k not in d1:
                    raise ValueError(f"Config mismatch on resume: missing key '{pfx}' in saved config")
                if k not in d2:
                    raise ValueError(f"Config mismatch on resume: missing key '{pfx}' in current config")
                _diff_keys(d1[k], d2[k], pfx)
        elif isinstance(d1, list) and isinstance(d2, list):
            if len(d1) != len(d2):
                raise ValueError(
                    f"Config mismatch on resume at '{prefix}': saved length {len(d1)} != current length {len(d2)}"
                )
            for idx, (v1, v2) in enumerate(zip(d1, d2)):
                _diff_keys(v1, v2, f"{prefix}[{idx}]")
        else:
            if d1 != d2:
                raise ValueError(f"Config mismatch on resume at '{prefix}': saved={d1!r}, current={d2!r}")

    _diff_keys(saved_copy, curr_copy)


def compare_cohort_descriptors(saved_cohort: Mapping[str, Any], current_cohort: Mapping[str, Any]) -> None:
    """Validate cohort identity, train/val membership, and loader state between saved checkpoint and active trainer."""
    if not isinstance(saved_cohort, Mapping) or not saved_cohort:
        raise ValueError("Saved cohort descriptor cannot be empty or non-mapping")
    if not isinstance(current_cohort, Mapping) or not current_cohort:
        raise ValueError("Current cohort descriptor cannot be empty or non-mapping")

    # 1. Path and dataset configuration comparison
    critical_fields = ("dataset_name", "data_root", "split_manifest", "train_split", "val_split", "class_mapping")
    for field in critical_fields:
        if field in saved_cohort and field in current_cohort:
            if saved_cohort[field] != current_cohort[field]:
                raise ValueError(
                    f"Cohort mismatch on resume for '{field}': saved={saved_cohort[field]!r}, "
                    f"current={current_cohort[field]!r}"
                )

    # 2. Strict membership validation for train AND val splits
    for split in ("train", "val"):
        mem_key = f"{split}_membership"
        saved_mem = saved_cohort.get(mem_key)
        current_mem = current_cohort.get(mem_key)

        # Resolve real membership signature
        saved_sig = None
        if isinstance(saved_mem, Mapping) and "membership_signature" in saved_mem:
            saved_sig = saved_mem["membership_signature"]
        elif isinstance(saved_cohort.get(f"{split}_cohort_identity"), Mapping) and "membership_signature" in saved_cohort[f"{split}_cohort_identity"]:
            saved_sig = saved_cohort[f"{split}_cohort_identity"]["membership_signature"]
        elif split == "train" and isinstance(saved_cohort.get("cohort_identity"), Mapping) and "membership_signature" in saved_cohort["cohort_identity"]:
            saved_sig = saved_cohort["cohort_identity"]["membership_signature"]

        if saved_sig is None or not isinstance(saved_sig, str) or not saved_sig.strip():
            raise ValueError(
                f"Saved cohort descriptor missing real membership_signature for {split} split (got {saved_sig!r})"
            )

        curr_sig = None
        if isinstance(current_mem, Mapping) and "membership_signature" in current_mem:
            curr_sig = current_mem["membership_signature"]
        elif isinstance(current_cohort.get(f"{split}_cohort_identity"), Mapping) and "membership_signature" in current_cohort[f"{split}_cohort_identity"]:
            curr_sig = current_cohort[f"{split}_cohort_identity"]["membership_signature"]

        if curr_sig is None or not isinstance(curr_sig, str) or not curr_sig.strip():
            raise ValueError(
                f"Current cohort descriptor missing real membership_signature for {split} split (got {curr_sig!r})"
            )

        # Check patient count and patient IDs (strictly catches same-count changed patient fixture!)
        if isinstance(saved_mem, Mapping) and isinstance(current_mem, Mapping):
            saved_patients = saved_mem.get("patient_ids")
            curr_patients = current_mem.get("patient_ids")
            if saved_patients is not None and curr_patients is not None:
                if saved_patients != curr_patients:
                    diff_added = sorted(set(curr_patients) - set(saved_patients))
                    diff_removed = sorted(set(saved_patients) - set(curr_patients))
                    raise ValueError(
                        f"Cohort membership mismatch on resume for {split} split: patient IDs differ "
                        f"(saved count={len(saved_patients)}, current count={len(curr_patients)}). "
                        f"Removed: {diff_removed}, Added: {diff_added}"
                    )

            # Check case count and case IDs
            saved_cases = saved_mem.get("case_ids")
            curr_cases = current_mem.get("case_ids")
            if saved_cases is not None and curr_cases is not None:
                if saved_cases != curr_cases:
                    raise ValueError(
                        f"Cohort membership mismatch on resume for {split} split: case IDs differ "
                        f"(saved count={len(saved_cases)}, current count={len(curr_cases)})"
                    )

        if saved_sig != curr_sig:
            raise ValueError(
                f"Cohort membership signature mismatch on resume for {split} split: "
                f"saved={saved_sig!r}, current={curr_sig!r}"
            )

    # 3. Loader state / nested sampling identity comparison
    for split in ("train", "val"):
        ident_key = f"{split}_cohort_identity"
        saved_ident = saved_cohort.get(ident_key)
        current_ident = current_cohort.get(ident_key)
        if split == "train" and saved_ident is None and "cohort_identity" in saved_cohort:
            saved_ident = saved_cohort["cohort_identity"]

        if not saved_ident or not isinstance(saved_ident, Mapping):
            raise ValueError(f"Saved cohort descriptor missing required '{ident_key}' metadata.")
        if not current_ident or not isinstance(current_ident, Mapping):
            raise ValueError(f"Current cohort descriptor missing required '{ident_key}' metadata.")

        saved_sampling = saved_ident.get("sampling")
        curr_sampling = current_ident.get("sampling")

        if not saved_sampling or not isinstance(saved_sampling, Mapping):
            raise ValueError(f"Saved '{ident_key}' missing required 'sampling' metadata.")
        if not curr_sampling or not isinstance(curr_sampling, Mapping):
            raise ValueError(f"Current '{ident_key}' missing required 'sampling' metadata.")

        for sk in ("shuffled", "drop_last", "iterates_every_record", "sampler", "batch_sampler"):
            if sk in saved_sampling or sk in curr_sampling:
                val_saved = saved_sampling.get(sk)
                val_curr = curr_sampling.get(sk)
                if val_saved != val_curr:
                    raise ValueError(
                        f"Loader sampling mismatch on resume for {split} split ({sk}): "
                        f"saved={val_saved!r}, current={val_curr!r}"
                    )


class UnifiedTrainer:
    """Single-process unified Self-Audit pipeline runner."""

    def __init__(
        self,
        config: UnifiedConfig,
        *,
        model: nn.Module | None = None,
        device: torch.device | None = None,
        disable_tqdm: bool = False,
    ) -> None:
        self.config = config
        self.device = device or resolve_device(config.experiment.device)
        self.disable_tqdm = disable_tqdm

        seed_everything(config.experiment.seed, deterministic=config.experiment.deterministic)

        if model is None:
            self.model = build_model_from_config(config.to_legacy_model_config(), self.device)
        else:
            self.model = model.to(self.device)

        self.schedule: Schedule = config.training.schedule

        cf_cfg = config.training.counterfactual
        self.generator = CounterfactualGenerator(
            positive_fraction=cf_cfg.mixture[0],
            negative_fraction=cf_cfg.mixture[1],
            hard_neutral_fraction=cf_cfg.mixture[2],
            min_repair_fraction=cf_cfg.repair_strength_min,
            max_repair_fraction=cf_cfg.repair_strength_max,
            epsilon_neutral=cf_cfg.epsilon,
            neutral_max_retries=cf_cfg.retries,
            num_classes=config.model.num_classes,
        )

        self.amp_enabled, self.amp_dtype = resolve_amp(
            {"amp": config.training.amp.enabled, "amp_dtype": config.training.amp.dtype},
            self.device,
        )

        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.scaler: Any = None

        self.current_interval_index: int | None = None
        self.global_epoch: int = 0
        self.global_step: int = 0
        self.optimizer_step: int = 0
        self.best_metric: float = -float("inf")
        self.completed: bool = False
        self.incomplete_reason: str | None = None
        self.incomplete_epoch: bool = False
        self.written_best_path: Path | None = None
        self.run_id: str = f"run_{uuid.uuid4().hex[:12]}"
        self.is_resumed: bool = False
        self.failure_stage: str = "init"
        self.completed_epochs: int = 0
        self.last_completed_validation: dict[str, Any] | None = None
        self.last_checkpoint_committed_epoch: int | None = None
        self._failure_recorded: bool = False
        self._failure_path: Path | None = None
        self.best_checkpoint_hash: str | None = None
        self.best_epoch: int | None = None
        # Immutable selected-best reference, as committed in last.pt.  ``None``
        # means no selection has been committed, which is the only state in
        # which ``best_metric`` may stay at its -inf sentinel.
        self.best_reference: dict[str, Any] | None = None

        # Data loader caches per (batch_size, augment)
        self._loader_cache: dict[tuple[int, bool], tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]] = {}

        self.report: dict[str, Any] = {
            "schema_version": config.schema_version,
            "protocol": {
                "unified_schedule": True,
                "single_process": True,
                "single_model_instance": True,
                "total_epochs": self.schedule.total_epochs,
            },
            "epochs": [],
            "completed": False,
        }

        # External logging is NOT started here.  ``resume_from_checkpoint`` runs
        # after construction and is what recovers the run identity, so creating
        # a live W&B run now would start an orphan run on every resume.  A
        # *disabled* logger is a complete no-op object, so every call site keeps
        # working until :meth:`start_logging` supplies the validated identity.
        wandb_cfg = config.logging.wandb
        self._wandb_settings: dict[str, Any] = {
            "enabled": bool(wandb_cfg.enabled),
            "project": wandb_cfg.project,
            "entity": wandb_cfg.entity,
            "run_name": wandb_cfg.run_name,
            "config": config.to_dict(),
            "mode": wandb_cfg.mode,
        }
        self._wandb_started: bool = False
        self.logger = WandbLogger(**{**self._wandb_settings, "enabled": False})
        self.wandb_identity: dict[str, Any] = dict(self.logger.identity_summary)

    @contextlib.contextmanager
    def _preserving_rng(self) -> Iterator[None]:
        """Run telemetry without letting it perturb the training random streams.

        Checkpoint RNG is captured at save time, before post-commit telemetry.
        An SDK -- or a mocked adapter -- that draws from the global streams
        would otherwise make a resumed run diverge from an uninterrupted one.
        Data augmentation RNG semantics are untouched; only telemetry is fenced.
        """

        state = _rng_state()
        try:
            yield
        finally:
            _restore_rng_state(state)

    def start_logging(self) -> dict[str, Any]:
        """Start external logging once the run identity is validated.

        Called after any resume, so a resumed attempt reuses the recovered
        ``run_id`` instead of creating a fresh backend run.  The wrapper decides
        what it may claim: an explicit id and ``resume`` are forwarded only in
        online mode, and an offline attempt reports its limitation rather than
        pretending the local run continues a backend one.
        """

        if self._wandb_started:
            return self.wandb_identity
        self._wandb_started = True
        if not self._wandb_settings["enabled"]:
            self.wandb_identity = dict(self.logger.identity_summary)
            return self.wandb_identity
        with self._preserving_rng():
            self.logger = WandbLogger(
                **self._wandb_settings,
                run_id=self.run_id,
                resume="allow" if self.is_resumed else None,
            )
        self.wandb_identity = dict(self.logger.identity_summary)
        return self.wandb_identity

    def _record_logger_adapter_error(self, stage: str, exc: BaseException) -> None:
        """Keep an adapter exception as a telemetry error the report can show.

        A logger (or a mocked adapter) that raises out of ``log``/``finish``
        instead of appending its own entry would otherwise leave no trace, and
        the run would report clean telemetry.
        """

        logger = getattr(self, "logger", None)
        if logger is None:
            return
        errors = getattr(logger, "telemetry_errors", None)
        if isinstance(errors, list):
            errors.append(f"{stage}: {exc}")
        try:
            logger.failed_log_count = int(getattr(logger, "failed_log_count", 0)) + 1
        except Exception:  # pragma: no cover - a logger that refuses attributes
            pass
        if getattr(logger, "last_error", None) is None and isinstance(exc, Exception):
            logger.last_error = exc

    def _restored_epoch_history(
        self, payload: Mapping[str, Any], completed_epoch: int, path: Path
    ) -> list[dict[str, Any]]:
        """Restore the committed epoch rows a resumed run must keep reporting.

        The history travels inside the checkpoint, so it is covered by the same
        safe load and provenance as the weights.  It is validated rather than
        trusted: a history that does not describe exactly epochs 1..N is a
        refusal, never a silent truncation to the new attempt's rows.
        """

        if completed_epoch == 0:
            return []
        history = payload.get("epoch_history")
        if history is None:
            raise ValueError(
                f"Checkpoint at epoch {completed_epoch} carries no 'epoch_history': refusing to "
                f"resume into a report that would silently drop the completed epochs ({path})."
            )
        if not isinstance(history, (list, tuple)):
            raise ValueError(
                f"Checkpoint 'epoch_history' must be a list, got {type(history).__name__} ({path})"
            )
        if len(history) != completed_epoch:
            raise ValueError(
                f"Checkpoint 'epoch_history' has {len(history)} row(s) but records "
                f"{completed_epoch} completed epoch(s) ({path}); refusing a truncated history."
            )
        restored: list[dict[str, Any]] = []
        for index, entry in enumerate(history, start=1):
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"Checkpoint 'epoch_history' row {index} must be a mapping, got "
                    f"{type(entry).__name__} ({path})"
                )
            recorded = entry.get("epoch")
            if isinstance(recorded, bool) or not isinstance(recorded, (int, Integral)) or int(recorded) != index:
                raise ValueError(
                    f"Checkpoint 'epoch_history' row {index} records epoch {recorded!r}; the history "
                    f"must describe exactly epochs 1..{completed_epoch} in order ({path})."
                )
            restored.append(dict(entry))

        # The tail row must describe the very commit this checkpoint is: a
        # history whose last row disagrees with the payload counters or with
        # the schedule is not the history of these bytes.
        tail = restored[-1]
        for field, expected in (
            ("global_step", _strict_counter(payload, "global_step", context=str(path))),
            ("optimizer_step", _strict_counter(payload, "optimizer_step", context=str(path))),
            ("completed_epochs", completed_epoch),
            ("last_checkpoint_committed_epoch", completed_epoch),
        ):
            if field not in tail:
                raise ValueError(
                    f"Checkpoint 'epoch_history' tail row records no '{field}' ({path})."
                )
            if tail[field] != expected:
                raise ValueError(
                    f"Checkpoint 'epoch_history' tail row has {field}={tail[field]!r} but the payload "
                    f"records {expected!r} ({path}); the history does not describe this commit."
                )
        if tail.get("checkpoint_committed") is not True:
            raise ValueError(
                f"Checkpoint 'epoch_history' tail row is not marked checkpoint_committed ({path}); "
                "the saved history must describe the commit it is part of."
            )
        expected_interval = self.schedule.get_interval(completed_epoch - 1)
        expected_index = self.schedule.get_interval_index(completed_epoch - 1)
        if tail.get("interval") != expected_interval.name or tail.get("interval_index") != expected_index:
            raise ValueError(
                f"Checkpoint 'epoch_history' tail row reports interval "
                f"{tail.get('interval')!r}[{tail.get('interval_index')!r}] but epoch {completed_epoch} "
                f"belongs to {expected_interval.name!r}[{expected_index}] in the configured schedule ({path})."
            )
        return restored

    def _refresh_wandb_identity(self) -> dict[str, Any]:
        """Re-read the logger's live identity so a report cannot cache a stale one.

        ``finish_status`` in particular changes at finalization, and a cached
        ``not_finished`` next to ``finalization_status="finalized"`` would
        contradict itself.
        """

        logger = getattr(self, "logger", None)
        summary = getattr(logger, "identity_summary", None) if logger is not None else None
        if isinstance(summary, Mapping):
            self.wandb_identity = dict(summary)
        return dict(self.wandb_identity)

    def _finalization_status(self) -> str:
        """Report the logger's actual finish outcome, never an assumed one."""

        logger = getattr(self, "logger", None)
        if logger is None:
            return "no_logger"
        status = getattr(logger, "finish_status", None)
        if status == "ok":
            return "finalized"
        if status == "failed":
            return "finish_failed"
        if status == "not_finished":
            return "not_finalized"
        return str(status) if status is not None else "unknown"

    @property
    def source_signature(self) -> str | None:
        """Obtain current source content signature using existing provenance helpers."""
        try:
            sig = source_content_signature()
            return sig.get("source_content_signature")
        except Exception:
            return None

    @property
    def git_commit(self) -> str | None:
        """Obtain current git commit hash if known."""
        try:
            prov = git_source_provenance()
            sha = prov.get("git_sha")
            return str(sha) if sha and sha != "UNKNOWN" else None
        except Exception:
            return None

    @property
    def telemetry_summary(self) -> dict[str, Any]:
        """Obtain safe portable telemetry summary without raw Exception objects or __dict__."""
        try:
            if not hasattr(self, "logger") or self.logger is None:
                return {
                    "enabled": False,
                    "telemetry_status": "disabled",
                    "failed_log_count": 0,
                    "telemetry_errors": [],
                }
            logger = self.logger
            enabled = bool(getattr(logger, "enabled", False))
            errs = list(getattr(logger, "telemetry_errors", []))
            fail_cnt = int(getattr(logger, "failed_log_count", 0))
            status = "degraded" if errs else ("active" if enabled else "disabled")
            return {
                "enabled": enabled,
                "telemetry_status": status,
                "failed_log_count": fail_cnt,
                "telemetry_errors": [str(e) for e in errs],
                "wandb_identity": dict(
                    getattr(logger, "identity_summary", None)
                    or getattr(self, "wandb_identity", {})
                    or {}
                ),
            }
        except Exception:
            return {
                "enabled": False,
                "telemetry_status": "unknown",
                "failed_log_count": 0,
                "telemetry_errors": [],
            }

    def record_failure(
        self,
        exc: BaseException,
        *,
        failure_stage: str | None = None,
        status: str = "failed",
    ) -> Path | None:
        """Record minimal strict atomic failure artifact on error or interruption.

        Failure cleanup belongs in a finally block independent of payload preparation.
        All secondary errors are reported to stderr, and the original exception must propagate without masking.
        """
        if getattr(self, "_failure_recorded", False):
            return getattr(self, "_failure_path", None)
        self._failure_recorded = True

        failure_path: Path | None = None
        stage = failure_stage or getattr(self, "failure_stage", None) or "unknown"
        current_ep = getattr(self, "global_epoch", 0)
        completed_ep = getattr(self, "completed_epochs", 0)

        # Fallback exception stringification protected
        try:
            exc_msg = str(exc)
        except Exception:
            exc_msg = f"<unformattable {type(exc).__name__}>"

        failure_payload: dict[str, Any] = {
            "schema_version": 1,
            "completed": False,
            "status": status,
            "failure_stage": stage,
            "current_epoch": current_ep,
            "global_epoch": current_ep,  # Alias for zero-based current global_epoch
            "completed_epochs": completed_ep,
            "global_step": getattr(self, "global_step", 0),
            "optimizer_step": getattr(self, "optimizer_step", 0),
            "exception_type": type(exc).__name__,
            "exception_message": exc_msg,
            "last_completed_validation": None,
            "last_checkpoint_committed_epoch": getattr(self, "last_checkpoint_committed_epoch", None),
            "run_id": getattr(self, "run_id", None),
            "config_signature": None,
            "config_identity": None,
            "recipe_signature": None,
            "source_signature": None,
            "git_commit": None,
            "git_sha": None,
            "producer_source_content_signature": None,
        }

        report_dir = Path("reports")
        is_ownership_refusal = False

        try:
            # 1. Safely resolve report_dir
            try:
                cfg = getattr(self, "config", None)
                if cfg is not None:
                    logging_cfg = getattr(cfg, "logging", None)
                    if logging_cfg is not None:
                        rdir_val = getattr(logging_cfg, "report_dir", None)
                        if rdir_val:
                            report_dir = Path(rdir_val)
            except Exception as sec_exc:
                print(f"[lifecycle] Secondary error resolving report_dir: {sec_exc}", file=sys.stderr)

            canonical_failure_path = report_dir / "failure.json"
            if canonical_failure_path.exists():
                failure_path = report_dir / f"failure_attempt_{uuid.uuid4().hex[:8]}.json"
            else:
                failure_path = canonical_failure_path
            self._failure_path = failure_path

            # Safe validation snapshot
            try:
                last_val = getattr(self, "last_completed_validation", None)
                if last_val is not None:
                    failure_payload["last_completed_validation"] = copy.deepcopy(last_val)
            except Exception as sec_exc:
                print(f"[lifecycle] Secondary error snapshotting last_completed_validation: {sec_exc}", file=sys.stderr)

            # Safe config signature / identity (Item 2: catch failures only during failure metadata prep)
            cfg_sig = None
            try:
                if hasattr(self, "config") and self.config is not None:
                    cfg_sig = self.config_signature
            except Exception as sec_exc:
                print(f"[lifecycle] Secondary error computing config_signature: {sec_exc}", file=sys.stderr)
            failure_payload["config_signature"] = cfg_sig
            failure_payload["config_identity"] = cfg_sig
            failure_payload["recipe_signature"] = cfg_sig

            # Safe source signature
            src_sig = None
            try:
                src_sig = getattr(self, "source_signature", None)
            except Exception as sec_exc:
                print(f"[lifecycle] Secondary error computing source_signature: {sec_exc}", file=sys.stderr)
            failure_payload["source_signature"] = src_sig
            failure_payload["producer_source_content_signature"] = src_sig

            # Safe git commit / git sha
            git_sha = None
            try:
                git_sha = getattr(self, "git_commit", None)
            except Exception as sec_exc:
                print(f"[lifecycle] Secondary error resolving git_commit: {sec_exc}", file=sys.stderr)
            failure_payload["git_commit"] = git_sha
            failure_payload["git_sha"] = git_sha

            # Update report with truthful incomplete pipeline metadata
            try:
                rep = getattr(self, "report", None)
                if rep is not None and isinstance(rep, dict):
                    rep["completed"] = False
                    rep["status"] = "failed"
                    rep["incomplete_reason"] = f"failed_at_{stage}"
            except Exception as sec_exc:
                print(f"[lifecycle] Secondary error updating report metadata: {sec_exc}", file=sys.stderr)

            # Atomic write failure artifact
            try:
                report_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_json(failure_path, _json_safe(failure_payload), indent=2, sort_keys=True)
            except Exception as sec_exc:
                print(f"[lifecycle] Secondary error writing failure artifact: {sec_exc}", file=sys.stderr)

            # Update progress report if recorded epochs exist and not ownership refusal
            if not is_ownership_refusal:
                try:
                    rep = getattr(self, "report", None)
                    if rep is not None and isinstance(rep, dict) and rep.get("epochs"):
                        progress_report_path = report_dir / "pipeline_report.json"
                        atomic_write_json(progress_report_path, _json_safe(rep), indent=2, sort_keys=True)
                except Exception as sec_exc:
                    print(f"[lifecycle] Secondary error updating progress report on failure: {sec_exc}", file=sys.stderr)

        except Exception as sec_exc:
            print(f"[lifecycle] Secondary error during failure payload preparation: {sec_exc}", file=sys.stderr)

        finally:
            # Item 1: Failure cleanup belongs in a finally independent of payload preparation!
            if hasattr(self, "logger") and self.logger is not None:
                try:
                    self.logger.set_summary({
                        "completed": False,
                        "status": status,
                        "failure_stage": stage,
                        "completed_epochs": completed_ep,
                        "last_checkpoint_committed_epoch": getattr(self, "last_checkpoint_committed_epoch", None),
                    })
                except Exception as sec_exc:
                    print(f"[lifecycle] Secondary error setting logger summary: {sec_exc}", file=sys.stderr)

                try:
                    exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
                    self.logger.finish(exit_code=exit_code)
                except Exception as sec_exc:
                    print(f"[lifecycle] Secondary error finalizing logger: {sec_exc}", file=sys.stderr)

                # Item 6: Capture telemetry failure status AFTER summary/finish in report & failure payload
                try:
                    tel_summary = self.telemetry_summary
                    rep = getattr(self, "report", None)
                    if rep is not None and isinstance(rep, dict):
                        rep["telemetry"] = tel_summary
                        if not is_ownership_refusal and rep.get("epochs"):
                            progress_report_path = report_dir / "pipeline_report.json"
                            atomic_write_json(progress_report_path, _json_safe(rep), indent=2, sort_keys=True)
                    if failure_path is not None and failure_path.exists():
                        failure_payload["telemetry"] = tel_summary
                        atomic_write_json(failure_path, _json_safe(failure_payload), indent=2, sort_keys=True)
                except Exception as sec_exc:
                    print(f"[lifecycle] Secondary error updating telemetry status after finish: {sec_exc}", file=sys.stderr)

        return failure_path

    @property
    def config_signature(self) -> str:
        """Obtain canonical hash signature of execution configuration."""
        return compute_config_signature(self.config.to_dict())

    def get_cohort_descriptor(self, interval: ScheduleInterval | None = None) -> dict[str, Any]:
        """Compute effective cohort descriptor for train AND val cohorts with full membership and loader state."""
        if interval is None:
            epoch = min(max(0, self.global_epoch), self.schedule.total_epochs - 1)
            interval = self.schedule.get_interval(epoch)

        desc: dict[str, Any] = {
            "dataset_name": self.config.dataset.name,
            "data_root": str(self.config.dataset.data_root),
            "split_manifest": str(self.config.dataset.split_manifest),
            "train_split": self.config.dataset.train_split,
            "val_split": self.config.dataset.val_split,
            "test_split": self.config.dataset.test_split,
            "class_mapping": dict(self.config.dataset.class_mapping),
            "interval_name": interval.name,
            "batch_size": interval.batch_size,
        }

        train_loader, val_loader = self.get_loaders(interval)

        # 1. Train membership and loader state
        train_mem = split_membership_descriptor(train_loader, split_name=self.config.dataset.train_split)
        train_cid = cohort_identity(
            train_loader,
            split_name=self.config.dataset.train_split,
            batch_size=interval.batch_size,
        )
        desc["train_membership"] = train_mem
        desc["train_cohort_identity"] = train_cid
        desc["cohort_identity"] = train_cid

        # 2. Validation membership and loader state
        val_mem = split_membership_descriptor(val_loader, split_name=self.config.dataset.val_split)
        val_cid = cohort_identity(
            val_loader,
            split_name=self.config.dataset.val_split,
            batch_size=getattr(val_loader, "batch_size", interval.batch_size),
        )
        desc["val_membership"] = val_mem
        desc["val_cohort_identity"] = val_cid

        return desc

    def guard_training_outputs(
        self,
        *,
        start_epoch: int = 0,
        max_steps: int | None = None,
        max_val_batches: int | None = None,
    ) -> None:
        """Validate execution parameters and guard existing checkpoints before training modifies any files."""
        self.failure_stage = "init"
        # 1. Validate step/batch bounds
        if max_steps is not None and int(max_steps) <= 0:
            raise ValueError(f"max_steps must be > 0, got {max_steps}")
        if max_val_batches is not None and int(max_val_batches) <= 0:
            raise ValueError(f"max_val_batches must be > 0, got {max_val_batches}")

        # 2. Validate checkpoint flags
        if not self.config.checkpoint.save_last:
            raise ValueError(
                "checkpoint.save_last=False is unsupported in unified training pipeline: "
                "last.pt is required for pipeline continuity and resumability."
            )
        if not self.config.checkpoint.save_best:
            raise ValueError(
                "checkpoint.save_best=False is unsupported in unified training pipeline: "
                "best.pt is required for Phase C model selection and post-training calibration."
            )

        # 3. Guard existing checkpoint artifacts on fresh run
        output_dir = Path(self.config.checkpoint.output_dir)
        if start_epoch == 0 and not self.is_resumed:
            existing_artifacts = [
                output_dir / name
                for name in ("best.pt", "last.pt")
                if (output_dir / name).exists()
            ]
            if existing_artifacts:
                names = [p.name for p in existing_artifacts]
                raise FileExistsError(
                    f"Fresh training run refused: existing checkpoint artifact(s) found in {output_dir}: {names}. "
                    "Pre-existing checkpoints must never be unlinked or overwritten by a fresh run. "
                    "Specify a new output_dir or explicitly resume with --resume."
                )

        # 4. Guard existing report artifact on fresh run
        report_dir = Path(self.config.logging.report_dir)
        if start_epoch == 0 and not self.is_resumed:
            existing_report = report_dir / "pipeline_report.json"
            if existing_report.exists():
                raise FileExistsError(
                    f"Fresh training run refused: existing report artifact found in {report_dir}: pipeline_report.json. "
                    "Pre-existing reports must never be unlinked or overwritten by a fresh run. "
                    "Specify a new report_dir or explicitly resume with --resume."
                )

        # 5. Guard and validate dataset protocol splits if data_root exists
        data_root = Path(self.config.dataset.data_root)
        if data_root.exists():
            legacy_cfg = self.config.to_legacy_dataset_config()
            validation = validate_dataset_splits(legacy_cfg)
            if not validation.get("validated"):
                raise ValueError(f"Dataset split validation failed for {self.config.dataset.name}: {validation}")

    def get_loaders(
        self, interval: ScheduleInterval
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Obtain train and val DataLoaders for an interval (cached)."""
        key = (interval.batch_size, interval.augment)
        if key in self._loader_cache:
            return self._loader_cache[key]

        legacy_cfg = self.config.to_legacy_dataset_config(interval)
        train_ds = build_patient_dataset(legacy_cfg, split=self.config.dataset.train_split, train=interval.augment)
        val_ds = build_patient_dataset(legacy_cfg, split=self.config.dataset.val_split, train=False)

        # The resolved configuration is honoured verbatim: silently forcing
        # persistent_workers off here would contradict the provenance the
        # checkpoint records and would disarm the negative resume guard that
        # refuses an augmenting persistent-worker checkpoint.  The canonical
        # YAMLs already resolve to non-persistent workers; a config that asks
        # for persistent augmenting workers is rejected at resume instead.
        train_loader = build_data_loader(train_ds, legacy_cfg, device=self.device, train=True)
        val_loader = build_data_loader(val_ds, legacy_cfg, device=self.device, train=False)

        self._loader_cache[key] = (train_loader, val_loader)
        return train_loader, val_loader

    def set_module_trainability(self, trainable: str) -> None:
        """Enforce requires_grad AND module train/eval mode per design specification.

        Auditor parameters start with 'auditor'. All other parameters belong to annotation.
        Zero LR or frozen modules must not retain gradients or weight decay.
        """
        for name, param in self.model.named_parameters():
            is_auditor = name.startswith("auditor")
            if trainable == "annotation":
                param.requires_grad = not is_auditor
            elif trainable == "auditor":
                param.requires_grad = is_auditor
            elif trainable == "all":
                param.requires_grad = True
            else:
                raise ValueError(f"Unknown trainable setting: {trainable!r}")

        if trainable == "annotation":
            if hasattr(self.model, "encoder"):
                self.model.encoder.train()
            if hasattr(self.model, "fpn"):
                self.model.fpn.train()
            if hasattr(self.model, "initial_head"):
                self.model.initial_head.train()
            if hasattr(self.model, "annotation_expert"):
                self.model.annotation_expert.train()
            if hasattr(self.model, "auditor"):
                self.model.auditor.eval()
        elif trainable == "auditor":
            if hasattr(self.model, "encoder"):
                self.model.encoder.eval()
            if hasattr(self.model, "fpn"):
                self.model.fpn.eval()
            if hasattr(self.model, "initial_head"):
                self.model.initial_head.eval()
            if hasattr(self.model, "annotation_expert"):
                self.model.annotation_expert.eval()
            if hasattr(self.model, "auditor"):
                self.model.auditor.train()
        elif trainable == "all":
            self.model.train()

    def setup_interval_optimizer_and_scheduler(
        self, interval: ScheduleInterval, num_batches: int
    ) -> None:
        """Reset optimizer, scheduler, and scaler at interval boundary while carrying live model weights."""
        self.set_module_trainability(interval.trainable)

        # Build parameter groups strictly for parameters that are trainable and have LR > 0
        param_groups: list[dict[str, Any]] = []
        weight_decay = self.config.training.optimizer.weight_decay

        if interval.trainable in ("annotation", "all"):
            # Encoder parameters
            encoder_params = [
                p for n, p in self.model.named_parameters()
                if not n.startswith("auditor") and "encoder" in n and p.requires_grad
            ]
            if encoder_params and interval.encoder_lr > 0.0:
                param_groups.append({
                    "params": encoder_params,
                    "lr": float(interval.encoder_lr),
                    "weight_decay": float(weight_decay),
                    "name": "encoder",
                })

            # Non-encoder annotation parameters
            head_params = [
                p for n, p in self.model.named_parameters()
                if not n.startswith("auditor") and "encoder" not in n and p.requires_grad
            ]
            if head_params and interval.annotation_lr > 0.0:
                param_groups.append({
                    "params": head_params,
                    "lr": float(interval.annotation_lr),
                    "weight_decay": float(weight_decay),
                    "name": "annotation_heads",
                })

        if interval.trainable in ("auditor", "all"):
            auditor_params = [
                p for n, p in self.model.named_parameters()
                if n.startswith("auditor") and p.requires_grad
            ]
            if auditor_params and interval.auditor_lr > 0.0:
                param_groups.append({
                    "params": auditor_params,
                    "lr": float(interval.auditor_lr),
                    "weight_decay": float(weight_decay),
                    "name": "auditor",
                })

        if not param_groups:
            raise RuntimeError(f"No trainable parameter groups created for interval '{interval.name}'")

        opt_cfg = self.config.training.optimizer
        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
        )

        updates_per_epoch = self.schedule.steps_per_epoch(interval, num_batches)
        total_interval_steps = max(updates_per_epoch * interval.num_epochs, 1)
        warmup_epochs = self.config.training.lr_curve.warmup_epochs
        warmup_steps = min(int(math.ceil(warmup_epochs * updates_per_epoch)), total_interval_steps)

        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer,
            total_steps=total_interval_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=self.config.training.lr_curve.min_ratio,
        )

        self.scaler = build_grad_scaler(
            enabled=self.amp_enabled,
            device=self.device,
            dtype=self.amp_dtype,
        )

    #: Honest label for the evidence the auxiliary predicted-history rollout
    #: consumes at each stage.  During the annotation bootstrap the Auditor is
    #: untrained, so its local evidence is cold and uncalibrated; once the
    #: Auditor is being optimised the same evidence is non-stationary.  Neither
    #: is calibrated exposure and neither is logged as such.
    EVIDENCE_CALIBRATION_LABELS = {
        "weighted_a0_a3": "cold_untrained_predicted",
        "counterfactual_audit": "nonstationary_training_predicted",
        "retained_final_annotation": "nonstationary_training_predicted",
    }

    def predicted_history_settings(self) -> tuple[bool, float]:
        """Resolve the optional ``training.rollout`` curriculum fields.

        The config schema is owned by the config worker; this reads the fields
        defensively so the trainer keeps its documented default (exposure OFF,
        weight 0.1) on a config revision that does not carry them yet, and
        rejects an out-of-contract value rather than silently coercing it.
        """

        rollout = self.config.training.rollout
        enabled = getattr(rollout, "predicted_history_exposure", False)
        if not isinstance(enabled, bool):
            raise TypeError(
                "training.rollout.predicted_history_exposure must be a bool, got "
                f"{type(enabled).__name__}"
            )
        raw_weight = getattr(rollout, "predicted_history_weight", 0.1)
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise TypeError(
                "training.rollout.predicted_history_weight must be a real number, got "
                f"{type(raw_weight).__name__}"
            )
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"training.rollout.predicted_history_weight must be finite and >= 0, got {weight!r}"
            )
        return enabled, weight

    def invalidate_candidate_c_records(self) -> bool:
        """Drop any Candidate-C replay record held by the model, if supported.

        Replay records are only valid against the exact model snapshot that
        produced them, so they must never span a batch or an optimizer update.
        The hook is optional: the core model may not expose it, in which case
        this reports ``False`` and the caller logs zero invalidations rather
        than implying one happened.
        """

        hook = getattr(self.model, "invalidate_candidate_c_records", None)
        if callable(hook):
            hook()
            return True
        return False

    def _empty_rollout_details(self, interval: ScheduleInterval) -> dict[str, Any]:
        enabled, weight = self.predicted_history_settings()
        return {
            "rollout_counters": zero_rollout_counters(),
            "auxiliary_annotation_loss": 0.0,
            "auxiliary_audit_loss": 0.0,
            "auxiliary_batches": 0,
            "auxiliary_seconds": 0.0,
            "predicted_history_exposure": enabled,
            "predicted_history_weight": weight,
            "evidence_calibration": self.EVIDENCE_CALIBRATION_LABELS.get(
                interval.objective, "unmeasured"
            ),
        }

    def compute_batch_loss(
        self,
        batch: dict[str, torch.Tensor],
        interval: ScheduleInterval,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Compute schedule-driven batch loss based on the active interval objective."""
        objective = interval.objective
        rollout_details = self._empty_rollout_details(interval)
        exposure_enabled = bool(rollout_details["predicted_history_exposure"])
        exposure_weight = float(rollout_details["predicted_history_weight"])

        if objective == "weighted_a0_a3":
            with autocast_context(enabled=self.amp_enabled, device=self.device, dtype=self.amp_dtype):
                output = self.model.forward_annotation(batch["image"]) if hasattr(self.model, "forward_annotation") else self.model(batch["image"])
                loss, parts = phase_a_loss(
                    output,
                    batch["mask"],
                    stage_weights=self.config.training.annotation_loss.stage_weights,
                )
                # The primary zero-history bootstrap objective is unchanged.
                loss = loss * float(interval.annotation_weight)

            # Optional bounded auxiliary exposure to the official accepted
            # predicted-history rollout.  Default OFF; when off nothing here
            # runs and the stage is bit-identical to the baseline schedule.
            if exposure_enabled and exposure_weight > 0.0 and float(interval.annotation_weight) > 0.0:
                started = time.perf_counter()
                with autocast_context(enabled=self.amp_enabled, device=self.device, dtype=self.amp_dtype):
                    auxiliary, counters = compute_predicted_history_annotation_loss(
                        self.model,
                        batch,
                        tau_accept=self.config.training.rollout.tau,
                        t_max=self.config.training.rollout.max_turns,
                    )
                    loss = loss + float(interval.annotation_weight) * exposure_weight * auxiliary
                rollout_details["rollout_counters"] = counters
                rollout_details["auxiliary_annotation_loss"] = float(auxiliary.detach())
                rollout_details["auxiliary_batches"] = 1
                rollout_details["auxiliary_seconds"] = time.perf_counter() - started

            return loss, {
                "loss": float(loss.detach()),
                "annotation_loss": float(loss.detach()),
                "audit_loss": 0.0,
                "parts": parts,
                "objective": objective,
                **rollout_details,
            }

        elif objective == "counterfactual_audit":
            # Parity with Phase B reference runner: frozen annotation runs in FP32 outside autocast,
            # internal autocast is applied inside _auditor_batch strictly around model.auditor.
            loss, details = _auditor_batch(
                self.model,
                batch,
                self.generator,
                neutral_margin=self.config.training.audit_loss.neutral_margin,
                local_weighting=(self.config.training.audit_loss.local_class_weighting != "none"),
                audit_margin=self.config.training.audit_loss.audit_margin,
                amp_enabled=self.amp_enabled,
                amp_dtype=self.amp_dtype,
            )

            # Optional auditor-only exposure to predicted-accepted-history
            # transitions.  The annotator stays frozen: the rollout runs under
            # no_grad and every Auditor input is a detached leaf.
            auxiliary: torch.Tensor | None = None
            if exposure_enabled and exposure_weight > 0.0 and float(interval.audit_weight) > 0.0:
                started = time.perf_counter()
                auxiliary, counters = compute_predicted_history_audit_loss(
                    self.model,
                    batch,
                    tau_accept=self.config.training.rollout.tau,
                    t_max=self.config.training.rollout.max_turns,
                    neutral_margin=self.config.training.audit_loss.neutral_margin,
                    local_weighting=(self.config.training.audit_loss.local_class_weighting != "none"),
                    audit_margin=self.config.training.audit_loss.audit_margin,
                    amp_enabled=self.amp_enabled,
                    amp_dtype=self.amp_dtype,
                )
                rollout_details["rollout_counters"] = counters
                rollout_details["auxiliary_seconds"] = time.perf_counter() - started
                if auxiliary is not None:
                    rollout_details["auxiliary_audit_loss"] = float(auxiliary.detach())
                    rollout_details["auxiliary_batches"] = 1

            if loss is None and auxiliary is None:
                return None, {
                    "loss": 0.0,
                    "annotation_loss": 0.0,
                    "audit_loss": 0.0,
                    "transitions": 0,
                    "objective": objective,
                    **rollout_details,
                }

            total = None if loss is None else loss * float(interval.audit_weight)
            if auxiliary is not None:
                scaled = float(interval.audit_weight) * exposure_weight * auxiliary
                total = scaled if total is None else total + scaled
            return total, {
                "loss": float(total.detach()),
                "annotation_loss": 0.0,
                "audit_loss": float(total.detach()),
                "details": details,
                "objective": objective,
                **rollout_details,
            }

        elif objective == "retained_final_annotation":
            with autocast_context(enabled=self.amp_enabled, device=self.device, dtype=self.amp_dtype):
                loss, details = compute_joint_losses(
                    self.model,
                    batch,
                    tau_accept=self.config.training.rollout.tau,
                    t_max=self.config.training.rollout.max_turns,
                    lambda_audit=interval.audit_weight,
                    neutral_margin=self.config.training.audit_loss.neutral_margin,
                    local_weighting=(self.config.training.audit_loss.local_class_weighting != "none"),
                )
                if interval.annotation_weight != 1.0:
                    loss = (
                        float(interval.annotation_weight) * details["annotation_loss_tensor"]
                        + float(interval.audit_weight) * details["audit_loss_tensor"]
                    )
            # The joint objective already rolls out with accepted predicted
            # history, so the auxiliary rollout is deliberately NOT run here.
            # Its counters are measured from the joint rollout itself.
            rollout_details["rollout_counters"] = rollout_counters(
                details.get("output"),
                model=self.model,
                batch_size=int(batch["image"].shape[0]),
            )
            return loss, {
                "loss": float(loss.detach()),
                "annotation_loss": float(details["annotation_loss"]),
                "audit_loss": float(details["audit_loss"]),
                "objective": objective,
                **rollout_details,
            }

        else:
            raise ValueError(f"Unsupported objective: {objective!r}")

    def train_epoch(
        self,
        interval: ScheduleInterval,
        train_loader: torch.utils.data.DataLoader,
        *,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Execute one epoch using the shared batch optimization loop."""
        # Ensure correct module modes
        self.set_module_trainability(interval.trainable)
        self.incomplete_epoch = False

        accumulation_steps = interval.accumulation_steps
        grad_clip = self.config.training.optimizer.grad_clip

        running_loss = 0.0
        running_ann_loss = 0.0
        running_aud_loss = 0.0
        total_samples = 0
        batches = 0
        pending = 0
        epoch_optimizer_steps = 0

        exposure_enabled, exposure_weight = self.predicted_history_settings()
        rollout_totals = zero_rollout_counters()
        auxiliary_annotation_total = 0.0
        auxiliary_audit_total = 0.0
        auxiliary_batches = 0
        auxiliary_seconds = 0.0
        candidate_c_invalidations = 0
        evidence_calibration = self.EVIDENCE_CALIBRATION_LABELS.get(interval.objective, "unmeasured")

        # A replay record is only valid against the snapshot that produced it.
        if self.invalidate_candidate_c_records():
            candidate_c_invalidations += 1

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.global_epoch + 1:03d}/{self.schedule.total_epochs:03d} [{interval.name}]",
            disable=self.disable_tqdm,
            leave=False,
        )

        for batch_index, raw_batch in enumerate(pbar):
            if max_steps is not None and self.optimizer_step >= int(max_steps):
                self.completed = False
                self.incomplete_reason = "max_steps_reached"
                self.incomplete_epoch = True
                break

            batch = move_batch(raw_batch, self.device)
            # No record may survive into the next batch.
            if self.invalidate_candidate_c_records():
                candidate_c_invalidations += 1
            loss, details = self.compute_batch_loss(batch, interval)

            # Counters describe what the rollout actually observed, so they are
            # accumulated before any batch is skipped for having no loss.
            batch_counters = details.get("rollout_counters")
            if isinstance(batch_counters, Mapping):
                accumulate_rollout_counters(rollout_totals, dict(batch_counters))
            auxiliary_annotation_total += float(details.get("auxiliary_annotation_loss", 0.0))
            auxiliary_audit_total += float(details.get("auxiliary_audit_loss", 0.0))
            auxiliary_batches += int(details.get("auxiliary_batches", 0))
            auxiliary_seconds += float(details.get("auxiliary_seconds", 0.0))

            # Skip batch update if loss is None (e.g. Phase B with zero transitions)
            if loss is None:
                continue

            if not is_finite(loss):
                raise FloatingPointError(
                    f"Non-finite loss ({float(loss.detach()):r}) at global_epoch={self.global_epoch} "
                    f"batch={batch_index} details={details}"
                )

            scaled_loss = loss / float(accumulation_steps)
            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            pending += 1
            batches += 1
            batch_size = int(batch["image"].shape[0])
            total_samples += batch_size

            loss_val = float(loss.detach())
            if interval.objective == "weighted_a0_a3":
                # Reference train_annotation_epoch reduces sample-weighted
                running_loss += loss_val * batch_size
                if "parts" in details and "loss" in details["parts"]:
                    ann_val = float(details["parts"]["loss"])
                elif interval.annotation_weight > 0.0:
                    ann_val = loss_val / float(interval.annotation_weight)
                else:
                    ann_val = float(details.get("annotation_loss", loss_val))
                running_ann_loss += ann_val * batch_size

            elif interval.objective == "counterfactual_audit":
                # Reference train_auditor_epoch reduces unweighted mean over batches
                running_loss += loss_val
                if interval.audit_weight > 0.0:
                    aud_val = loss_val / float(interval.audit_weight)
                else:
                    aud_val = float(details.get("audit_loss", loss_val))
                running_aud_loss += aud_val

            elif interval.objective == "retained_final_annotation":
                # Reference finetune_joint_epoch reduces unweighted components over batches
                running_loss += loss_val
                ann_val = float(details.get("annotation_loss", 0.0))
                aud_val = float(details.get("audit_loss", 0.0))
                running_ann_loss += ann_val
                running_aud_loss += aud_val

            else:
                running_loss += loss_val * batch_size
                running_ann_loss += float(details.get("annotation_loss", 0.0)) * batch_size
                running_aud_loss += float(details.get("audit_loss", 0.0)) * batch_size

            if pending == accumulation_steps:
                step_ok, _ = finalize_optimizer_step(
                    self.model,
                    self.optimizer,
                    self.scaler,
                    scheduler=self.scheduler,
                    pending_batches=pending,
                    accumulation_steps=accumulation_steps,
                    grad_clip=grad_clip,
                )
                if not step_ok:
                    raise FloatingPointError(
                        f"Non-finite gradients at global_epoch={self.global_epoch} step={self.optimizer_step}"
                    )
                self.optimizer_step += 1
                self.global_step += 1
                epoch_optimizer_steps += 1
                pending = 0
                # Weights moved: every replay record is now stale.
                if self.invalidate_candidate_c_records():
                    candidate_c_invalidations += 1

                if max_steps is not None and self.optimizer_step >= int(max_steps):
                    self.completed = False
                    self.incomplete_reason = "max_steps_reached"
                    self.incomplete_epoch = (batch_index + 1 < len(train_loader))
                    break

            avg_running = running_loss / max(total_samples, 1) if interval.objective == "weighted_a0_a3" else running_loss / max(batches, 1)
            pbar.set_postfix({
                "loss": f"{float(loss.detach()):.4f}",
                "avg": f"{avg_running:.4f}",
                "opt_step": self.optimizer_step,
            })

        if pending and not self.incomplete_epoch:
            step_ok, _ = finalize_optimizer_step(
                self.model,
                self.optimizer,
                self.scaler,
                scheduler=self.scheduler,
                pending_batches=pending,
                accumulation_steps=accumulation_steps,
                grad_clip=grad_clip,
            )
            if not step_ok:
                raise FloatingPointError(
                    f"Non-finite gradients at global_epoch={self.global_epoch} final partial step"
                )
            self.optimizer_step += 1
            self.global_step += 1
            epoch_optimizer_steps += 1
            pending = 0
            if self.invalidate_candidate_c_records():
                candidate_c_invalidations += 1

        if interval.objective == "weighted_a0_a3":
            avg_loss = running_loss / max(total_samples, 1)
            avg_ann_loss = running_ann_loss / max(total_samples, 1)
            avg_aud_loss = None
        elif interval.objective == "counterfactual_audit":
            divisor = max(batches, 1)
            avg_loss = running_loss / divisor
            avg_ann_loss = None
            avg_aud_loss = running_aud_loss / divisor
        elif interval.objective == "retained_final_annotation":
            divisor = max(batches, 1)
            avg_loss = running_loss / divisor
            avg_ann_loss = running_ann_loss / divisor
            avg_aud_loss = running_aud_loss / divisor
        else:
            divisor = max(batches, 1)
            avg_loss = running_loss / max(total_samples, 1)
            avg_ann_loss = running_ann_loss / max(total_samples, 1) if running_ann_loss > 0 else None
            avg_aud_loss = running_aud_loss / divisor if running_aud_loss > 0 else None

        lrs = {f"lr_{group.get('name', idx)}": float(group['lr']) for idx, group in enumerate(self.optimizer.param_groups)}
        train_stats: dict[str, Any] = {
            "train_loss": avg_loss,
            "loss": avg_loss,
            "train_batches": batches,
            "epoch_optimizer_steps": epoch_optimizer_steps,
            **lrs,
        }
        if avg_ann_loss is not None:
            train_stats["annotation_loss"] = avg_ann_loss
        if avg_aud_loss is not None:
            train_stats["audit_loss"] = avg_aud_loss

        # Sample-level rollout telemetry, emitted for EVERY stage.  A zero is a
        # measured zero; ``rollout/rollout_batches == 0`` marks a stage that ran
        # no rollout, and ``rollout/candidate_c_diagnostics_available == 0.0``
        # marks Candidate-C counters that were not observable at all.
        for key in PREDICTED_HISTORY_COUNTER_KEYS:
            train_stats[f"rollout/{key}"] = int(rollout_totals[key])
        # Narrower companion to real_evidence_attempts: history created by an
        # ordinary (trainable) annotation transition, not by a restitution or
        # rollback acceptance.  Forward exposure only; it is not a claim that a
        # parameter gradient carried that evidence.
        train_stats[f"rollout/{ORDINARY_REAL_EVIDENCE_KEY}"] = int(
            rollout_totals[ORDINARY_REAL_EVIDENCE_KEY]
        )
        train_stats["rollout/ordinary_accepted_path_from_diagnostics"] = float(
            rollout_totals["ordinary_accepted_path_from_diagnostics"]
        )
        train_stats["rollout/attempted_turn_samples"] = int(rollout_totals["attempted_turn_samples"])
        train_stats["rollout/rollout_samples"] = int(rollout_totals["rollout_samples"])
        train_stats["rollout/rollout_batches"] = int(rollout_totals["rollout_batches"])
        train_stats["rollout/candidate_c_diagnostics_available"] = float(
            rollout_totals["candidate_c_diagnostics_available"]
        )
        train_stats["rollout/predicted_history_exposure"] = 1.0 if exposure_enabled else 0.0
        train_stats["rollout/predicted_history_weight"] = float(exposure_weight)
        train_stats["rollout/evidence_calibration"] = evidence_calibration
        train_stats["rollout/candidate_c_invalidations"] = int(candidate_c_invalidations)
        # Auxiliary rollout cost is reported separately from the primary loss.
        train_stats["rollout/auxiliary_batches"] = int(auxiliary_batches)
        train_stats["rollout/auxiliary_seconds"] = float(auxiliary_seconds)
        train_stats["rollout/auxiliary_annotation_loss"] = (
            auxiliary_annotation_total / auxiliary_batches if auxiliary_batches else 0.0
        )
        train_stats["rollout/auxiliary_audit_loss"] = (
            auxiliary_audit_total / auxiliary_batches if auxiliary_batches else 0.0
        )

        for group in self.optimizer.param_groups:
            gname = group.get("name")
            if gname == "encoder":
                train_stats["encoder_lr"] = float(group["lr"])
            elif gname == "annotation_heads":
                train_stats["annotation_lr"] = float(group["lr"])
            elif gname == "auditor":
                train_stats["auditor_lr"] = float(group["lr"])

        return train_stats

    def validate_epoch(
        self,
        interval: ScheduleInterval,
        val_loader: torch.utils.data.DataLoader,
        *,
        max_val_batches: int | None = None,
    ) -> dict[str, Any]:
        """Run schedule-driven evaluation for the active interval."""
        if interval.objective == "weighted_a0_a3":
            val_stats = validate_annotation_epoch(
                self.model,
                val_loader,
                self.device,
                stage_weights=self.config.training.annotation_loss.stage_weights,
                max_batches=max_val_batches,
                epoch=self.global_epoch,
                total_epochs=self.schedule.total_epochs,
                disable_tqdm=self.disable_tqdm,
            )
            headroom: dict[str, Any] = {}
            if self.config.diagnostics.evaluate_headroom:
                headroom = evaluate_annotation_headroom(
                    self.model,
                    val_loader,
                    self.device,
                    neutral_margin=self.config.training.audit_loss.neutral_margin,
                    max_batches=max_val_batches,
                    disable_tqdm=self.disable_tqdm,
                )
            primary = float(val_stats.get("val_macro_foreground_dice", float("nan")))
            return {**val_stats, **headroom, "primary_metric": primary}

        elif interval.objective == "counterfactual_audit":
            val_stats = validate_auditor_epoch(
                self.model,
                val_loader,
                self.device,
                generator=self.generator,
                neutral_margin=self.config.training.audit_loss.neutral_margin,
                local_weighting=(self.config.training.audit_loss.local_class_weighting != "none"),
                amp_enabled=self.amp_enabled,
                amp_dtype=self.amp_dtype,
                max_batches=max_val_batches,
                epoch=self.global_epoch,
                total_epochs=self.schedule.total_epochs,
                disable_tqdm=self.disable_tqdm,
            )
            primary = float(val_stats.get("primary_metric", float("nan")))
            return {**val_stats, "primary_metric": primary}

        elif interval.objective == "retained_final_annotation":
            val_stats = validate_phase_c(
                self.model,
                val_loader,
                self.device,
                tau_accept=self.config.training.rollout.tau,
                t_max=self.config.training.rollout.max_turns,
                max_batches=max_val_batches,
                epoch=self.global_epoch,
                total_epochs=self.schedule.total_epochs,
                disable_tqdm=self.disable_tqdm,
            )
            decomp: dict[str, Any] = {}
            if self.config.diagnostics.evaluate_decomposition:
                decomp = evaluate_audit_decomposition(
                    self.model,
                    val_loader,
                    self.device,
                    tau_accept=self.config.training.rollout.tau,
                    t_max=self.config.training.rollout.max_turns,
                    neutral_margin=self.config.training.audit_loss.neutral_margin,
                    max_batches=max_val_batches,
                    disable_tqdm=self.disable_tqdm,
                )
            primary = float(val_stats.get("final_foreground_macro_dice", float("nan")))
            return {**val_stats, **decomp, "primary_metric": primary}

        else:
            raise ValueError(f"Unknown interval objective: {interval.objective!r}")

    def _build_wandb_payload(
        self,
        epoch: int,
        interval: ScheduleInterval,
        interval_idx: int,
        train_stats: dict[str, Any],
        val_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Build canonical W&B log payload with monotonic global keys and zero phase_a/b/c prefixes."""
        log_payload: dict[str, Any] = {
            "pipeline/global_epoch": epoch + 1,
            "pipeline/global_step": self.global_step,
            "pipeline/optimizer_step": self.optimizer_step,
            "pipeline/interval_index": interval_idx,
            "pipeline/interval_name": interval.name,
        }

        # Canonical schedule keys
        log_payload["schedule/audit_weight"] = float(interval.audit_weight)
        log_payload["schedule/annotation_weight"] = float(interval.annotation_weight)
        log_payload["schedule/rollout_mode"] = str(interval.rollout)
        if hasattr(interval, "transition_population"):
            log_payload["schedule/transition_population"] = str(interval.transition_population)

        # On-policy fraction: ONLY actual population ratio if observed, NEVER fabricate
        on_policy_count: float | None = None
        total_trans_count: float | None = None
        if "audit/on_policy/transition_count" in val_stats and "audit/combined/transition_count" in val_stats:
            on_policy_count = float(val_stats["audit/on_policy/transition_count"])
            total_trans_count = float(val_stats["audit/combined/transition_count"])
        elif "audit/on_policy/transition_group_count" in val_stats and "audit/combined/transition_group_count" in val_stats:
            on_policy_count = float(val_stats["audit/on_policy/transition_group_count"])
            total_trans_count = float(val_stats["audit/combined/transition_group_count"])
        elif "on_policy_transition_count" in val_stats and "total_transition_count" in val_stats:
            on_policy_count = float(val_stats["on_policy_transition_count"])
            total_trans_count = float(val_stats["total_transition_count"])
        elif "on_policy_fraction" in val_stats:
            val_frac = float(val_stats["on_policy_fraction"])
            if math.isfinite(val_frac):
                log_payload["schedule/on_policy_fraction"] = val_frac
        elif "on_policy_transitions" in train_stats and "total_transitions" in train_stats:
            on_policy_count = float(train_stats["on_policy_transitions"])
            total_trans_count = float(train_stats["total_transitions"])
        elif "on_policy_fraction" in train_stats:
            t_frac = float(train_stats["on_policy_fraction"])
            if math.isfinite(t_frac):
                log_payload["schedule/on_policy_fraction"] = t_frac

        if on_policy_count is not None and total_trans_count is not None and total_trans_count > 0:
            log_payload["schedule/on_policy_fraction"] = float(on_policy_count / total_trans_count)

        # Canonical training loss keys
        total_loss = train_stats.get("train_loss", train_stats.get("loss"))
        if total_loss is not None:
            log_payload["train/total_loss"] = float(total_loss)

        if "annotation_loss" in train_stats and train_stats["annotation_loss"] is not None:
            log_payload["train/annotation_loss"] = float(train_stats["annotation_loss"])

        if "audit_loss" in train_stats and train_stats["audit_loss"] is not None:
            log_payload["train/audit_loss"] = float(train_stats["audit_loss"])

        # Canonical learning rate keys
        lr_map = {
            "encoder": 0.0,
            "annotation": 0.0,
            "auditor": 0.0,
        }
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                name = group.get("name", "")
                lr = float(group.get("lr", 0.0))
                if name == "encoder":
                    lr_map["encoder"] = lr
                elif name == "annotation_heads":
                    lr_map["annotation"] = lr
                elif name == "auditor":
                    lr_map["auditor"] = lr
        else:
            if "encoder_lr" in train_stats:
                lr_map["encoder"] = float(train_stats["encoder_lr"])
            elif "lr_encoder" in train_stats:
                lr_map["encoder"] = float(train_stats["lr_encoder"])
            if "annotation_lr" in train_stats:
                lr_map["annotation"] = float(train_stats["annotation_lr"])
            elif "lr_annotation_heads" in train_stats:
                lr_map["annotation"] = float(train_stats["lr_annotation_heads"])
            elif "lr_annotation" in train_stats:
                lr_map["annotation"] = float(train_stats["lr_annotation"])
            if "auditor_lr" in train_stats:
                lr_map["auditor"] = float(train_stats["auditor_lr"])
            elif "lr_auditor" in train_stats:
                lr_map["auditor"] = float(train_stats["lr_auditor"])

        log_payload["train/encoder_lr"] = lr_map["encoder"]
        log_payload["train/annotation_lr"] = lr_map["annotation"]
        log_payload["train/auditor_lr"] = lr_map["auditor"]
        log_payload["train/lr_encoder"] = lr_map["encoder"]
        log_payload["train/lr_annotation"] = lr_map["annotation"]
        log_payload["train/lr_auditor"] = lr_map["auditor"]

        # Canonical validation keys
        if interval.objective == "weighted_a0_a3":
            # Initial Dice from headroom evaluation (Phase A a0_dice) if measured
            init_dice = None
            for ik in ("phase_a/a0_dice", "a0_dice", "initial_dice", "val/initial_dice"):
                if ik in val_stats and val_stats[ik] is not None:
                    v = float(val_stats[ik])
                    if math.isfinite(v):
                        init_dice = v
                        break
            if init_dice is not None:
                log_payload["val/initial_dice"] = init_dice

            # Candidate gain (refinement gain over a0) if measured
            cand_gain = None
            for gk in ("phase_a/total_refinement_gain", "total_refinement_gain", "candidate_gain", "val/candidate_gain"):
                if gk in val_stats and val_stats[gk] is not None:
                    v = float(val_stats[gk])
                    if math.isfinite(v):
                        cand_gain = v
                        break
            if cand_gain is not None:
                log_payload["val/candidate_gain"] = cand_gain

            # Final Dice
            final_dice = None
            for fk in ("phase_a/final_dice", "val_macro_foreground_dice", "final_dice", "val/final_dice"):
                if fk in val_stats and val_stats[fk] is not None:
                    v = float(val_stats[fk])
                    if math.isfinite(v):
                        final_dice = v
                        break
            if final_dice is not None:
                log_payload["val/final_dice"] = final_dice

            # Headroom and validation diagnostics (clean mapped without phase_a/ prefix)
            for k, v in val_stats.items():
                if k == "phase_a/mean_stage_headroom":
                    log_payload["headroom/mean_stage_headroom"] = float(v)
                elif k == "phase_a/headroom_collapse_ratio":
                    log_payload["headroom/headroom_collapse_ratio"] = float(v)
                elif k == "phase_a/sample_count":
                    log_payload["val/sample_count"] = float(v)
                elif k.startswith("phase_a/stage_"):
                    sub_key = k[len("phase_a/"):]
                    log_payload[f"headroom/{sub_key}"] = float(v) if isinstance(v, (int, float)) else v
                elif k == "val_loss":
                    log_payload["val/loss"] = float(v)
                elif k.startswith("val_dice_"):
                    log_payload[f"val/class_{k[len('val_dice_'):]}"] = float(v)

        elif interval.objective == "counterfactual_audit":
            # Map actual validation outputs for auditor evaluation without losing real metrics
            for k, v in val_stats.items():
                clean_k = k[len("phase_b/"):] if k.startswith("phase_b/") else k
                if clean_k.startswith("audit/"):
                    log_payload[clean_k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                elif clean_k.startswith("on_policy/"):
                    log_payload[f"audit/{clean_k}"] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                elif clean_k.startswith("synthetic/"):
                    log_payload[f"audit/{clean_k}"] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                elif clean_k in ("auroc", "auprc", "improve_regress_accuracy", "correlation_delta_q_delta_dice", "local_fix_f1", "local_regress_f1"):
                    if f"audit/combined/{clean_k}" not in log_payload:
                        log_payload[f"audit/combined/{clean_k}"] = float(v)
                elif clean_k in ("audit_loss", "loss"):
                    log_payload["audit/val_loss"] = float(v)
                    log_payload["val/loss"] = float(v)
                elif clean_k == "primary_metric":
                    log_payload["val/primary_metric"] = float(v)
                elif clean_k == "primary_metric_source":
                    log_payload["val/primary_metric_source"] = str(v)
                elif clean_k in ("transitions", "transition_count"):
                    log_payload["audit/transitions"] = float(v)
                elif clean_k in ("local_fix_count", "local_unchanged_count", "local_regress_count"):
                    log_payload[f"audit/{clean_k}"] = float(v)

        elif interval.objective == "retained_final_annotation":
            init_dice = None
            for ik in ("initial_foreground_macro_dice", "modes/initial_dice", "initial_dice", "val/initial_dice"):
                if ik in val_stats and val_stats[ik] is not None:
                    v = float(val_stats[ik])
                    if math.isfinite(v):
                        init_dice = v
                        break
            if init_dice is not None:
                log_payload["val/initial_dice"] = init_dice

            final_dice = None
            for fk in ("final_foreground_macro_dice", "modes/self_audit_dice", "final_dice", "val/final_dice"):
                if fk in val_stats and val_stats[fk] is not None:
                    v = float(val_stats[fk])
                    if math.isfinite(v):
                        final_dice = v
                        break
            if final_dice is not None:
                log_payload["val/final_dice"] = final_dice

            # Late candidate gain: represents PROPOSER improvement (always-accept or actual candidate metric with explicit name)
            # NEVER silently retain self-audit gain as candidate gain!
            cand_gain = None
            if "modes/candidate_path_gain" in val_stats and val_stats["modes/candidate_path_gain"] is not None:
                v = float(val_stats["modes/candidate_path_gain"])
                if math.isfinite(v):
                    cand_gain = v
            elif "modes/always_accept_dice" in val_stats and init_dice is not None:
                always_d = float(val_stats["modes/always_accept_dice"])
                if math.isfinite(always_d):
                    cand_gain = always_d - init_dice
            elif "candidate_gain" in val_stats and val_stats["candidate_gain"] is not None:
                v = float(val_stats["candidate_gain"])
                if math.isfinite(v):
                    cand_gain = v
            elif "proposer_gain" in val_stats and val_stats["proposer_gain"] is not None:
                v = float(val_stats["proposer_gain"])
                if math.isfinite(v):
                    cand_gain = v
            elif "always_accept_gain" in val_stats and val_stats["always_accept_gain"] is not None:
                v = float(val_stats["always_accept_gain"])
                if math.isfinite(v):
                    cand_gain = v

            if cand_gain is not None:
                log_payload["val/candidate_gain"] = cand_gain

            # Self-audit gain logged with its explicit semantic name
            self_audit_gain = None
            for sk in ("modes/self_audit_gain", "net_dice_gain", "net_gain"):
                if sk in val_stats and val_stats[sk] is not None:
                    v = float(val_stats[sk])
                    if math.isfinite(v):
                        self_audit_gain = v
                        break
            if self_audit_gain is None and final_dice is not None and init_dice is not None:
                self_audit_gain = final_dice - init_dice
            if self_audit_gain is not None:
                log_payload["val/self_audit_gain"] = self_audit_gain
                log_payload["val/net_dice_gain"] = self_audit_gain

            # Proposer always-accept and oracle metrics if evaluated
            if "modes/always_accept_dice" in val_stats:
                log_payload["val/always_accept_dice"] = float(val_stats["modes/always_accept_dice"])
            if "modes/oracle_dice" in val_stats:
                log_payload["val/oracle_dice"] = float(val_stats["modes/oracle_dice"])
            if "modes/audit_rescue_vs_always" in val_stats:
                log_payload["modes/audit_rescue_vs_always"] = float(val_stats["modes/audit_rescue_vs_always"])
            if "modes/headroom_capture_ratio" in val_stats:
                log_payload["modes/headroom_capture_ratio"] = float(val_stats["modes/headroom_capture_ratio"])

            if "harmful_acceptance_rate" in val_stats:
                log_payload["audit/harmful_acceptance_rate"] = float(val_stats["harmful_acceptance_rate"])
            if "beneficial_rejection_rate" in val_stats:
                log_payload["audit/beneficial_rejection_rate"] = float(val_stats["beneficial_rejection_rate"])
            if "net_dice_gain_after_auditing" in val_stats:
                log_payload["audit/net_dice_gain_after_auditing"] = float(val_stats["net_dice_gain_after_auditing"])

            if "audit_metrics" in val_stats and isinstance(val_stats["audit_metrics"], Mapping):
                for ak, av in val_stats["audit_metrics"].items():
                    log_payload[f"audit/combined/{ak}"] = float(av) if isinstance(av, (int, float)) else av

            # Any other modes/ or audit/ metrics (clean mapped without phase_c/ prefix)
            for k, v in val_stats.items():
                clean_k = k[len("phase_c/"):] if k.startswith("phase_c/") else k
                if clean_k.startswith("modes/") and clean_k not in log_payload:
                    log_payload[clean_k] = float(v) if isinstance(v, (int, float)) else v
                elif clean_k.startswith("audit/") and clean_k not in log_payload:
                    log_payload[clean_k] = float(v) if isinstance(v, (int, float)) else v

        # Rollout telemetry is stage-independent: copy the measured counters
        # and their denominators verbatim so the W&B fields, the JSON report row
        # and the trainer's own totals cannot disagree.
        for rollout_key, rollout_value in train_stats.items():
            if not rollout_key.startswith("rollout/") or rollout_key in log_payload:
                continue
            if isinstance(rollout_value, bool):
                log_payload[rollout_key] = float(rollout_value)
            elif isinstance(rollout_value, (int, float)):
                log_payload[rollout_key] = float(rollout_value)
            else:
                log_payload[rollout_key] = rollout_value

        # Strict hygiene: eliminate any accidental phase_a/, phase_b/, phase_c/ prefixes
        cleaned_payload: dict[str, Any] = {}
        for k, v in log_payload.items():
            clean_k = k
            for prefix in ("phase_a/", "phase_b/", "phase_c/"):
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
            cleaned_payload[clean_k] = v

        return cleaned_payload

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Resume training from a saved checkpoint, strictly restoring weights, RNG, and update counters."""
        self.failure_stage = "resume"
        self.is_resumed = True
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # The shared loader is the only read path: it enforces safe
        # deserialization (no unrestricted pickle fallback) and finite
        # model/optimizer state validation, and staging on CPU keeps a resumed
        # GPU run from materializing a second copy of the optimizer/RNG tensors
        # on the device.  RNG is restored separately, last, under step 11.
        payload = load_checkpoint(path, map_location="cpu", restore_rng=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Checkpoint payload must be a mapping, got {type(payload).__name__}")

        # 1. Resumability validation.  Exact resume requires each flag to be
        # *affirmatively* right: an absent flag is unknown, and unknown is not
        # permission.  Nothing below this point mutates the output directory,
        # so every provenance gate runs before any alias repair or relocation.
        for flag, required in (("resumable", True), ("incomplete_epoch", False), ("validation_complete", True)):
            if flag not in payload:
                raise ValueError(
                    f"Cannot resume from {checkpoint_path}: required flag '{flag}' is absent, so the "
                    "checkpoint cannot be certified as an exact resume point."
                )
            value = payload[flag]
            if not isinstance(value, bool) or value is not required:
                raise ValueError(
                    f"Cannot resume from {checkpoint_path}: '{flag}' is {value!r}, expected {required!r}."
                )

        # 2. Strict full configuration comparison (reject changed LR, schedule, seed, AMP, architecture, etc.)
        saved_config = payload.get("config")
        if saved_config is None or not isinstance(saved_config, Mapping):
            raise ValueError(f"Checkpoint missing required configuration metadata: {checkpoint_path}")
        compare_execution_configs(saved_config, self.config.to_dict())

        # 3. Strict source signature verification via existing provenance helpers
        saved_source = payload.get("source_signature")
        if saved_source is None:
            prov = payload.get("provenance")
            if isinstance(prov, Mapping):
                saved_source = prov.get("producer_source_content_signature")
        curr_source = self.source_signature

        def _usable_identity(value: Any) -> bool:
            return isinstance(value, str) and bool(value.strip()) and value != "UNKNOWN"

        # Fail closed: an unavailable or UNKNOWN identity on either side cannot
        # certify that the code is the code that produced the checkpoint, and
        # an uncertifiable resume is refused rather than waved through.
        if not _usable_identity(saved_source):
            raise ValueError(
                f"Cannot resume from {checkpoint_path}: the checkpoint records no usable source "
                f"content signature ({saved_source!r}), so source identity cannot be verified."
            )
        if not _usable_identity(curr_source):
            raise ValueError(
                f"Cannot resume from {checkpoint_path}: the current source content signature is "
                f"unavailable ({curr_source!r}), so source identity cannot be verified."
            )
        if saved_source != curr_source:
            raise ValueError(
                f"Source signature mismatch on resume: checkpoint was produced with source {saved_source}, "
                f"but current source is {curr_source}"
            )

        # Run identity must be present and non-empty: every later provenance
        # gate (the snapshot run_id check, the W&B identity) is defined
        # relative to it.
        saved_run_id = payload.get("run_id")
        if not isinstance(saved_run_id, str) or not saved_run_id.strip():
            raise ValueError(
                f"Cannot resume from {checkpoint_path}: no usable run_id recorded ({saved_run_id!r})."
            )

        # The weights must be the weights the producer stamped.
        saved_provenance = payload.get("provenance")
        if not isinstance(saved_provenance, Mapping):
            raise ValueError(
                f"Cannot resume from {checkpoint_path}: no provenance record, so the model state "
                "cannot be verified against what the producer measured."
            )
        recorded_state_digest = saved_provenance.get("state_digest")
        if recorded_state_digest is None:
            raise ValueError(
                f"Cannot resume from {checkpoint_path}: provenance records no state_digest."
            )
        measured_state_digest = state_digest(payload["model"])
        if str(recorded_state_digest) != measured_state_digest:
            raise ValueError(
                f"Tampered checkpoint {checkpoint_path}: provenance state_digest "
                f"{recorded_state_digest!r} does not match the stored tensors "
                f"({measured_state_digest})."
            )

        completed_epoch = _strict_counter(payload, "epoch", context=str(path))
        self.global_epoch = completed_epoch
        self.completed_epochs = completed_epoch
        self.last_checkpoint_committed_epoch = completed_epoch
        self.global_step = _strict_counter(payload, "global_step", context=str(path))
        self.optimizer_step = _strict_counter(payload, "optimizer_step", context=str(path))

        if completed_epoch < 1 or completed_epoch > self.schedule.total_epochs:
            raise ValueError(
                f"Cannot resume from {checkpoint_path}: recorded epoch {completed_epoch} is outside the "
                f"configured schedule [1, {self.schedule.total_epochs}]."
            )
        if self.optimizer_step > self.global_step:
            raise ValueError(
                f"Cannot resume from {checkpoint_path}: optimizer_step {self.optimizer_step} exceeds "
                f"global_step {self.global_step}."
            )

        # Verified progress history, restored before anything can truncate it.
        self.report["epochs"] = self._restored_epoch_history(payload, completed_epoch, path)

        # ``save_checkpoint`` merges ``extra`` into the payload, so the saved
        # validation sits at the top level; reading payload["extra"] found
        # nothing and silently reset it to None, losing the completed
        # validation a failure report right after resume has to show.
        if "last_completed_validation" in payload:
            saved_validation = payload["last_completed_validation"]
            self.last_completed_validation = (
                copy.deepcopy(dict(saved_validation))
                if isinstance(saved_validation, Mapping)
                else saved_validation
            )

        # 4. Cohort descriptor verification with train AND val membership and loader state
        saved_cohort = payload.get("cohort_descriptor") or payload.get("effective_cohort_descriptor")
        if saved_cohort is None or not isinstance(saved_cohort, Mapping):
            raise ValueError(f"Checkpoint missing required 'cohort_descriptor' metadata: {checkpoint_path}")
        saved_interval = (
            self.schedule.get_interval(completed_epoch - 1)
            if completed_epoch > 0
            else self.schedule.intervals[0]
        )
        current_cohort = self.get_cohort_descriptor(saved_interval)
        compare_cohort_descriptors(saved_cohort, current_cohort)

        # 5. Run identity
        if "run_id" in payload:
            self.run_id = str(payload["run_id"])

        if self.global_epoch < self.schedule.total_epochs:
            next_interval = self.schedule.get_interval(self.global_epoch)
            train_loader, _ = self.get_loaders(next_interval)

            # 6. Persistent DataLoader worker limitation
            is_persistent = (
                getattr(train_loader, "num_workers", 0) > 0
                and getattr(train_loader, "persistent_workers", False)
            )
            if is_persistent and next_interval.augment:
                raise RuntimeError(
                    f"Unsupported exact resume: interval '{next_interval.name}' uses augmenting persistent workers "
                    f"(num_workers={getattr(train_loader, 'num_workers', 0)}, persistent_workers=True). "
                    "DataLoader worker RNG cannot be checkpointed. Set num_workers=0 or persistent_workers=False "
                    "for verified exact resume."
                )

            # 7. Live loader resolution and cardinality check against SAVED completed interval
            saved_cardinality = payload.get("loader_cardinality")
            if saved_cardinality is None:
                raise ValueError(f"Checkpoint missing required 'loader_cardinality' metadata: {checkpoint_path}")

            if completed_epoch > 0:
                saved_loader, _ = self.get_loaders(saved_interval)
                expected_cardinality = len(saved_loader)
                if int(saved_cardinality) != expected_cardinality:
                    raise ValueError(
                        f"Loader cardinality mismatch on resume: saved={saved_cardinality}, "
                        f"current={expected_cardinality}"
                    )

            # 8. Setup optimizer and scheduler for next interval
            next_cardinality = len(train_loader)
            self.setup_interval_optimizer_and_scheduler(next_interval, next_cardinality)
            self.current_interval_index = self.schedule.get_interval_index(self.global_epoch)

            # Strict restoration at ordinary epoch vs reset at boundary (no absent metadata bypass)
            if not self.schedule.is_interval_start(self.global_epoch):
                if "optimizer" not in payload or payload["optimizer"] is None:
                    raise ValueError(
                        f"Checkpoint missing required optimizer state for non-boundary resume at epoch {self.global_epoch}"
                    )
                # Staged on CPU; torch's Optimizer.load_state_dict casts each
                # state tensor onto its parameter's device, so this restores
                # to the live parameter devices without a second GPU copy.
                self.optimizer.load_state_dict(payload["optimizer"])

                if "scheduler" not in payload or payload["scheduler"] is None:
                    raise ValueError(
                        f"Checkpoint missing required scheduler state for non-boundary resume at epoch {self.global_epoch}"
                    )
                self.scheduler.load_state_dict(payload["scheduler"])

                saved_scaler = payload.get("scaler")
                live_scaler_enabled = self.scaler is not None and self.scaler.is_enabled()
                saved_scaler_enabled = isinstance(saved_scaler, Mapping) and bool(saved_scaler)
                if live_scaler_enabled:
                    if saved_scaler is None or "scaler" not in payload:
                        raise ValueError(
                            f"Checkpoint missing required scaler state for non-boundary resume at epoch {self.global_epoch}"
                        )
                    if not saved_scaler_enabled:
                        # A disabled GradScaler serializes to {}, which is
                        # present and not None, so a presence-only check would
                        # hand torch an empty state dict and surface as an
                        # opaque "source state dict is empty" RuntimeError.
                        raise ValueError(
                            f"Cannot exactly resume at epoch {self.global_epoch}: gradient scaling is enabled "
                            f"on this device (amp_dtype={self.amp_dtype}), but the checkpoint carries an empty "
                            "scaler state, which is what a disabled scaler saves. The saved run used a "
                            "different effective AMP mode, so its loss scale cannot be continued. Resume on a "
                            "device/AMP configuration matching the one that produced the checkpoint."
                        )
                    self.scaler.load_state_dict(saved_scaler)
                elif saved_scaler_enabled:
                    # Never silently drop a saved loss scale: the saved run was
                    # scaling and this one is not, so continuation is not exact.
                    raise ValueError(
                        f"Cannot exactly resume at epoch {self.global_epoch}: the checkpoint carries a "
                        f"populated gradient-scaler state, but gradient scaling is disabled here "
                        f"(amp_enabled={self.amp_enabled}, amp_dtype={self.amp_dtype}). The canonical bfloat16 "
                        "configuration disables scaling by design; resume on the AMP configuration that "
                        "produced the checkpoint rather than discarding its saved scale."
                    )

        # 9. Load model weights
        if "model" not in payload:
            raise ValueError(f"Checkpoint missing required 'model' state: {checkpoint_path}")
        self.model.load_state_dict(payload["model"])

        # 10. Best checkpoint verification and output relocation safety:
        # Match same run, config, and saved hash. If relocated, ensure target dir does not
        # clobber unrelated existing files and copy best.pt into target dir to retain best lineage.
        saved_best_metric = payload.get("best_metric")
        saved_best_hash = payload.get("best_checkpoint_hash")
        saved_best_epoch = payload.get("best_epoch")
        saved_reference = payload.get(COMMITTED_BEST_REFERENCE_KEY)
        sibling_best = path.parent / PUBLIC_BEST_NAME

        configured_output_dir = Path(self.config.checkpoint.output_dir).resolve()
        is_relocated = (path.parent.resolve() != configured_output_dir)

        expected_alias_hash = (
            str(saved_reference["sha256"])
            if isinstance(saved_reference, Mapping) and "sha256" in saved_reference
            else saved_best_hash
        )
        if is_relocated and configured_output_dir.exists():
            existing_files = [p for p in configured_output_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
            unrelated = [
                p.name for p in existing_files
                if not (
                    p.name == PUBLIC_BEST_NAME
                    and expected_alias_hash is not None
                    and hashlib.sha256(p.read_bytes()).hexdigest() == expected_alias_hash
                )
            ]
            if unrelated:
                raise FileExistsError(
                    f"Output relocation across resume refused: target directory '{configured_output_dir}' "
                    f"already contains existing file(s) ({unrelated[:5]}) that would be clobbered. "
                    "Specify an empty or clean output_dir to preserve existing files."
                )

        if isinstance(saved_reference, Mapping):
            # Immutable-reference contract.  The snapshot, not the public
            # alias, is the committed selection: it is verified by hash before
            # anything is adopted, its recorded metric/epoch must agree with
            # the counters last.pt carries, and the alias is a republishable
            # view of it.  A crash between the last.pt commit and the alias
            # publish is therefore recoverable here, and only here -- the
            # evaluation path refuses a stale alias rather than repairing it.
            source_dir = path.parent
            try:
                snapshot = resolve_best_reference(saved_reference, source_dir)
            except SelectionError as exc:
                raise ValueError(
                    f"Resume refused: the committed selected-best reference in {path} does not "
                    f"verify against {source_dir} ({exc})."
                ) from exc

            reference_metric = float(saved_reference["metric"])
            reference_epoch = int(saved_reference["epoch"])
            # Every selection counter must be present: a missing one is not a
            # licence to skip the crosscheck, it is metadata that no longer
            # describes the state it claims to.
            for required in ("best_metric", "best_epoch", "best_checkpoint_hash"):
                if payload.get(required) is None:
                    raise ValueError(
                        f"Tampered selection metadata in {path}: {COMMITTED_BEST_REFERENCE_KEY} names a "
                        f"committed selection but '{required}' is missing or null."
                    )
            if not math.isclose(
                float(saved_best_metric), reference_metric, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Tampered selection metadata in {path}: best_metric={saved_best_metric!r} disagrees "
                    f"with the referenced snapshot metric {reference_metric!r}."
                )
            if isinstance(saved_best_epoch, bool) or not isinstance(saved_best_epoch, (int, Integral)):
                raise ValueError(
                    f"Tampered selection metadata in {path}: best_epoch must be an integer, got "
                    f"{type(saved_best_epoch).__name__}: {saved_best_epoch!r}."
                )
            if int(saved_best_epoch) != reference_epoch:
                raise ValueError(
                    f"Tampered selection metadata in {path}: best_epoch={saved_best_epoch!r} disagrees "
                    f"with the referenced snapshot epoch {reference_epoch!r}."
                )
            if str(saved_best_hash) != str(saved_reference["sha256"]):
                raise ValueError(
                    f"Tampered selection metadata in {path}: best_checkpoint_hash={saved_best_hash!r} "
                    f"disagrees with the referenced snapshot sha256 {saved_reference['sha256']!r}."
                )

            if reference_epoch > completed_epoch:
                raise ValueError(
                    f"Inconsistent selection metadata in {path}: the committed best reference names "
                    f"epoch {reference_epoch}, later than the completed epoch {completed_epoch}."
                )

            best_payload = load_checkpoint(snapshot, map_location="cpu", restore_rng=False)
            if not isinstance(best_payload, Mapping):
                raise ValueError(f"Corrupt selected-best snapshot payload: {snapshot}")
            # Run identity and execution config are required on a snapshot the
            # protocol wrote, not merely checked when present.
            if best_payload.get("run_id") is None:
                raise ValueError(f"Selected-best snapshot {snapshot} carries no run_id.")
            if best_payload["run_id"] != self.run_id:
                raise ValueError(
                    f"Selected-best snapshot run_id mismatch: {snapshot} has "
                    f"run_id={best_payload['run_id']!r}, expected {self.run_id!r}"
                )
            snapshot_config = best_payload.get("config")
            if not isinstance(snapshot_config, Mapping):
                raise ValueError(
                    f"Selected-best snapshot {snapshot} carries no execution configuration."
                )
            compare_execution_configs(snapshot_config, self.config.to_dict())

            snapshot_state_epoch = _strict_counter(best_payload, "epoch", context=str(snapshot))
            if snapshot_state_epoch != reference_epoch:
                raise ValueError(
                    f"Tampered selection metadata: {snapshot} records epoch {snapshot_state_epoch} but "
                    f"its reference claims epoch {reference_epoch}."
                )
            # The snapshot's own measured metric must be the metric the
            # reference commits, so a reference cannot be re-pointed at a
            # different epoch's weights.
            snapshot_metric = best_payload.get("best_metric")
            if snapshot_metric is None:
                raise ValueError(
                    f"Selected-best snapshot {snapshot} records no measured best_metric."
                )
            if not math.isclose(float(snapshot_metric), reference_metric, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"Tampered selection metadata: {snapshot} measured best_metric={snapshot_metric!r} "
                    f"but its reference claims {reference_metric!r}."
                )
            snapshot_selection_metric = best_payload.get("selection_metric")
            if snapshot_selection_metric is None:
                raise ValueError(
                    f"Selected-best snapshot {snapshot} records no selection_metric, so the quantity it "
                    "was selected on cannot be verified."
                )
            if str(snapshot_selection_metric) != SELECTION_METRIC_KEY:
                raise ValueError(
                    f"Selected-best snapshot {snapshot} was selected on "
                    f"{snapshot_selection_metric!r}, not the configured {SELECTION_METRIC_KEY!r}."
                )
            # Cross-check against the actual observed validation, not merely a
            # duplicated counter: the Dice the snapshot's own completed
            # validation measured must be the metric the selection claims.
            snapshot_validation = best_payload.get("last_completed_validation")
            if not isinstance(snapshot_validation, Mapping):
                raise ValueError(
                    f"Selected-best snapshot {snapshot} records no completed validation, so its selected "
                    "metric cannot be checked against an observed measurement."
                )
            if SELECTION_METRIC_KEY not in snapshot_validation:
                raise ValueError(
                    f"Selected-best snapshot {snapshot} completed validation records no "
                    f"{SELECTION_METRIC_KEY!r}."
                )
            observed_metric = snapshot_validation[SELECTION_METRIC_KEY]
            if observed_metric is None or not math.isfinite(float(observed_metric)):
                # An undefined observation cannot have produced a selection;
                # undefined stays undefined rather than being read as a value.
                raise ValueError(
                    f"Selected-best snapshot {snapshot} observed {SELECTION_METRIC_KEY}="
                    f"{observed_metric!r}, which is undefined and cannot have improved the best."
                )
            if not math.isclose(float(observed_metric), reference_metric, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"Tampered selection metadata: {snapshot} observed {SELECTION_METRIC_KEY}="
                    f"{float(observed_metric)!r} but its reference claims a selected metric of "
                    f"{reference_metric!r}."
                )
            # Provenance: the digest the producer stamped must equal the digest
            # of the tensors actually stored in the snapshot.
            snapshot_provenance = best_payload.get("provenance")
            if not isinstance(snapshot_provenance, Mapping):
                raise ValueError(f"Selected-best snapshot {snapshot} carries no provenance record.")
            recorded_digest = snapshot_provenance.get("state_digest")
            measured_digest = state_digest(best_payload["model"])
            if recorded_digest is None:
                raise ValueError(
                    f"Selected-best snapshot {snapshot} provenance records no state_digest."
                )
            if str(recorded_digest) != measured_digest:
                raise ValueError(
                    f"Tampered selected-best snapshot {snapshot}: provenance state_digest "
                    f"{recorded_digest!r} does not match the stored tensors ({measured_digest})."
                )

            active_reference = dict(saved_reference)
            if is_relocated:
                configured_output_dir.mkdir(parents=True, exist_ok=True)
                # Relocation moves the verified snapshot itself, atomically and
                # refusing any conflicting or unrelated destination entry; the
                # source directory is left intact.
                try:
                    active_reference = dict(
                        relocate_best_reference(saved_reference, source_dir, configured_output_dir)
                    )
                except SelectionError as exc:
                    raise ValueError(
                        f"Failed to retain selected-best lineage across relocation into "
                        f"{configured_output_dir}: {exc}"
                    ) from exc

            # Repair the public alias from the verified snapshot when an
            # interrupted publication left it stale or missing.
            alias_dir = configured_output_dir if is_relocated else source_dir
            if not best_alias_matches_reference(active_reference, alias_dir):
                print(
                    f"[checkpoint] repairing public best alias in {alias_dir} from committed "
                    f"selected-best epoch {reference_epoch} (sha256 {active_reference['sha256']})"
                )
                publish_best_alias(active_reference, alias_dir)

            self.best_reference = active_reference
            self.written_best_path = alias_dir / PUBLIC_BEST_NAME
            self.best_checkpoint_hash = str(active_reference["sha256"])
            self.best_epoch = reference_epoch
            self.best_metric = reference_metric

        elif saved_reference is None and COMMITTED_BEST_REFERENCE_KEY in payload:
            # Explicit "no selection committed".  A finite metric alongside it
            # would silently suppress every future selection, so it is refused
            # rather than carried forward.
            if saved_best_metric is not None and float(saved_best_metric) != -float("inf"):
                # Only -inf (or an absent value) means "nothing selected yet".
                # A finite value, +inf, or NaN would each suppress or corrupt
                # every future selection comparison.
                raise ValueError(
                    f"Inconsistent selection metadata in {path}: {COMMITTED_BEST_REFERENCE_KEY} is "
                    f"explicitly None but best_metric={saved_best_metric!r} is not the -inf sentinel. "
                    "A missing selection must use the explicit -inf sentinel so later epochs can "
                    "still select."
                )
            self.best_reference = None
            self.written_best_path = None
            self.best_checkpoint_hash = None
            self.best_epoch = None
            self.best_metric = -float("inf")

        elif saved_best_hash is not None:
            # Legacy sibling-hash lineage (checkpoints written before the
            # immutable-reference protocol).  Kept verbatim so historical
            # negative provenance gates still apply; such a checkpoint cannot
            # be repaired from a snapshot, because none was recorded.
            if not sibling_best.exists():
                raise FileNotFoundError(
                    f"Resume failed: required best checkpoint '{sibling_best}' referenced in checkpoint lineage is missing."
                )
            actual_bytes = sibling_best.read_bytes()
            actual_hash = hashlib.sha256(actual_bytes).hexdigest()
            if actual_hash != saved_best_hash:
                raise ValueError(
                    f"Stale or replaced best checkpoint detected: expected SHA256 {saved_best_hash}, "
                    f"got {actual_hash} on {sibling_best}. Never adopt an arbitrary or modified sibling best.pt."
                )
            best_payload = load_checkpoint(sibling_best, map_location="cpu", restore_rng=False)
            if not isinstance(best_payload, Mapping):
                raise ValueError(f"Corrupt best checkpoint payload: {sibling_best}")
            if "run_id" in best_payload and best_payload["run_id"] != self.run_id:
                raise ValueError(
                    f"Best checkpoint run_id mismatch: best.pt has run_id={best_payload['run_id']!r}, "
                    f"expected {self.run_id!r}"
                )
            if "config" in best_payload and isinstance(best_payload["config"], Mapping):
                compare_execution_configs(best_payload["config"], self.config.to_dict())

            if is_relocated:
                target_best = configured_output_dir / PUBLIC_BEST_NAME
                if not target_best.exists():
                    configured_output_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sibling_best, target_best)
                target_hash = hashlib.sha256(target_best.read_bytes()).hexdigest()
                if target_hash != saved_best_hash:
                    raise ValueError(
                        f"Failed to retain best lineage across relocation: hash {target_hash} != {saved_best_hash}"
                    )
                self.written_best_path = target_best
            else:
                self.written_best_path = sibling_best

            self.best_reference = None
            self.best_checkpoint_hash = saved_best_hash
            self.best_epoch = int(saved_best_epoch) if saved_best_epoch is not None else None
            self.best_metric = float(saved_best_metric) if saved_best_metric is not None else float(best_payload.get("best_metric", -float("inf")))
        else:
            self.best_reference = None
            self.written_best_path = None
            self.best_checkpoint_hash = None
            self.best_epoch = None
            if saved_best_metric is not None and float(saved_best_metric) != -float("inf"):
                raise ValueError(
                    f"Inconsistent selection metadata in {path}: best_metric={saved_best_metric!r} is "
                    "not the -inf sentinel but no committed selected-best state is recorded. A missing "
                    "selection must use the explicit -inf sentinel so later epochs can still select."
                )
            self.best_metric = -float("inf")

        self.output_dir = configured_output_dir
        self.is_resumed = True

        # 11. RNG restoration LAST
        if "rng_state" in payload and isinstance(payload["rng_state"], Mapping):
            rng = payload["rng_state"]
            if not all(k in rng for k in ("python", "numpy", "torch")):
                missing = [k for k in ("python", "numpy", "torch") if k not in rng]
                raise ValueError(
                    f"Checkpoint missing required RNG components {missing} (expected python, numpy, torch): {checkpoint_path}"
                )
            # On GPU an exact resume requires the CUDA generator state too:
            # without exact_cuda a checkpoint produced on a CPU host would be
            # accepted and the device stream silently left un-restored.
            _restore_rng_state(rng, exact_cuda=(self.device.type == "cuda"))
        else:
            raise ValueError(f"Checkpoint missing required 'rng_state' metadata: {checkpoint_path}")

        return self.global_epoch

    def train(
        self,
        *,
        start_epoch: int = 0,
        max_steps: int | None = None,
        max_val_batches: int | None = None,
    ) -> dict[str, Any]:
        """Execute the full unified training pipeline across scheduled epochs."""
        try:
            return self._train_impl(
                start_epoch=start_epoch,
                max_steps=max_steps,
                max_val_batches=max_val_batches,
            )
        except KeyboardInterrupt as exc:
            try:
                self.record_failure(exc, failure_stage="interrupted", status="interrupted")
            except Exception as sec_exc:
                print(f"[lifecycle] Secondary error in record_failure during interrupt: {sec_exc}", file=sys.stderr)
            raise
        except Exception as exc:
            try:
                stage = getattr(self, "failure_stage", "train")
                self.record_failure(exc, failure_stage=stage, status="failed")
            except Exception as sec_exc:
                print(f"[lifecycle] Secondary error in record_failure during exception: {sec_exc}", file=sys.stderr)
            raise

    def _train_impl(
        self,
        *,
        start_epoch: int = 0,
        max_steps: int | None = None,
        max_val_batches: int | None = None,
    ) -> dict[str, Any]:
        self.failure_stage = "init"
        self.guard_training_outputs(
            start_epoch=start_epoch,
            max_steps=max_steps,
            max_val_batches=max_val_batches,
        )

        output_dir = Path(self.config.checkpoint.output_dir)
        report_dir = Path(self.config.logging.report_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        print_model_parameter_summary(self.model, title="Unified Self-Audit Model")

        # External logging starts here, not in __init__: any resume has already
        # run, so the run identity is validated and a resumed attempt reuses it
        # instead of opening an orphan backend run.
        self.start_logging()
        self.report["wandb_identity"] = dict(self.wandb_identity)

        self.global_epoch = start_epoch
        if not self.is_resumed:
            self.completed_epochs = start_epoch
            self.last_checkpoint_committed_epoch = None

        for epoch in range(start_epoch, self.schedule.total_epochs):
            self.global_epoch = epoch
            interval = self.schedule.get_interval(epoch)
            interval_idx = self.schedule.get_interval_index(epoch)

            # Check if entering a new interval or reset boundary
            train_loader, val_loader = self.get_loaders(interval)
            if self.current_interval_index != interval_idx:
                self.setup_interval_optimizer_and_scheduler(interval, len(train_loader))
                self.current_interval_index = interval_idx

            # 1. Train epoch
            self.failure_stage = "training"
            train_stats = self.train_epoch(interval, train_loader, max_steps=max_steps)

            # 2. Validate epoch (never fabricate metrics after partial validation)
            self.failure_stage = "validation"
            val_stats = self.validate_epoch(interval, val_loader, max_val_batches=max_val_batches)
            self.last_completed_validation = copy.deepcopy(val_stats)

            row = {
                "epoch": epoch + 1,
                "global_epoch": epoch + 1,
                "interval": interval.name,
                "interval_index": interval_idx,
                "global_step": self.global_step,
                "optimizer_step": self.optimizer_step,
                "validation_complete": True,
                "checkpoint_committed": False,
                "last_checkpoint_committed_epoch": self.last_checkpoint_committed_epoch,
                "completed_epochs": self.completed_epochs,
                "status": "validation_complete",
                **train_stats,
                **val_stats,
            }
            self.report["epochs"].append(row)
            self.report["status"] = "running"
            self.report["completed_epochs"] = self.completed_epochs
            self.report["last_checkpoint_committed_epoch"] = self.last_checkpoint_committed_epoch
            self.report["telemetry"] = self.telemetry_summary

            # 3. Recoverable commit protocol: immutable snapshot -> last.pt -> alias.
            #
            # Per-file atomicity is not a transaction.  Writing the public
            # best.pt first meant a subsequent last.pt failure left the old
            # last.pt naming bytes that had already been overwritten (audit
            # R9).  Here the selected weights go to a unique, never-reused
            # snapshot under selected_best/, last.pt commits a hash-verified
            # reference to it, and only then does the public alias move.  A
            # crash at any point leaves the previous last.pt still referencing
            # its own untouched snapshot, and a crash after last.pt commits is
            # repaired on resume from the reference.  No in-process rollback is
            # relied on, because none can run under SIGKILL.
            self.failure_stage = "checkpoint"
            cohort_desc = self.get_cohort_descriptor(interval)
            resolved_schedule = {
                "interval_name": interval.name,
                "interval_index": interval_idx,
                "rollout_mode": str(interval.rollout),
                "trainable": str(interval.trainable),
                "objective": str(interval.objective),
                "transition_population": str(interval.transition_population),
                "annotation_weight": float(interval.annotation_weight),
                "audit_weight": float(interval.audit_weight),
                "encoder_lr": float(interval.encoder_lr),
                "annotation_lr": float(interval.annotation_lr),
                "auditor_lr": float(interval.auditor_lr),
                "batch_size": int(interval.batch_size),
                "accumulation_steps": int(interval.accumulation_steps),
                "augment": bool(interval.augment),
            }
            row["resolved_schedule"] = dict(resolved_schedule)
            commit_extra_common = {
                "run_id": self.run_id,
                "resolved_schedule": dict(resolved_schedule),
                "config_signature": self.config_signature,
                "recipe_signature": self.config_signature,
                "source_signature": self.source_signature,
                "cohort_descriptor": cohort_desc,
                "effective_cohort_descriptor": cohort_desc,
                "resumable": (not self.incomplete_epoch) and (max_val_batches is None),
                "incomplete_epoch": self.incomplete_epoch or (max_val_batches is not None),
                "validation_complete": (max_val_batches is None),
                "completed_epoch": epoch + 1,
                "active_interval": interval.name,
                "interval_name": interval.name,
                "interval_index": interval_idx,
                "loader_cardinality": len(train_loader),
                "telemetry_status": self.telemetry_summary["telemetry_status"],
                "telemetry_error_count": self.logger.failed_log_count,
                "last_completed_validation": self.last_completed_validation,
                "wandb_identity": dict(self.wandb_identity),
            }

            # Selection happens only in the gated interval, and only on the
            # configured Dice.  ``primary_metric`` is a different quantity in
            # the other intervals, so it is never a substitute -- not when the
            # Dice is absent, and not outside the gate.
            selection_metric_used = SELECTION_METRIC_KEY
            is_gated_interval = interval.rollout == "threshold_gate"
            selection_active = (
                max_val_batches is None
                and epoch >= self.config.checkpoint.best_selection_min_epoch
                and is_gated_interval
            )
            new_reference: dict[str, Any] | None = None
            selection_status = "not_gated"
            if selection_active:
                if SELECTION_METRIC_KEY not in val_stats:
                    raise ValueError(
                        f"Best selection in gated interval '{interval.name}' is defined on "
                        f"'{SELECTION_METRIC_KEY}', but epoch {epoch + 1} validation produced no "
                        f"such metric (available: {sorted(val_stats)}). Refusing to fall back to "
                        "'primary_metric', which is a different quantity in the other intervals."
                    )
                raw_metric = val_stats[SELECTION_METRIC_KEY]
                # An explicitly undefined Dice stays undefined: None is the
                # canonical null, never coerced to a number and never zeroed.
                metric = float("nan") if raw_metric is None else float(raw_metric)
                if not np.isfinite(metric):
                    selection_status = "skipped_nonfinite_metric"
                elif metric > self.best_metric:
                    snapshot_path = (
                        output_dir
                        / SELECTED_BEST_DIRNAME
                        / f"epoch_{epoch + 1}_{uuid.uuid4().hex[:12]}.pt"
                    )
                    save_checkpoint(
                        snapshot_path,
                        self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch + 1,
                        global_step=self.global_step,
                        optimizer_step=self.optimizer_step,
                        config=self.config.to_dict(),
                        extra={
                            **commit_extra_common,
                            "role": "selected_best",
                            "selection_metric": selection_metric_used,
                            "best_metric": metric,
                            "best_epoch": epoch + 1,
                        },
                    )
                    new_reference = dict(
                        make_best_reference(
                            snapshot_path, output_dir, epoch=epoch + 1, metric=metric
                        )
                    )
                    selection_status = "selected"
                else:
                    selection_status = "not_improved"

            # The reference last.pt will commit: the new one, or the one already
            # committed.  Selection metadata is derived FROM the reference, so
            # last.best_metric/best_epoch cannot drift from the snapshot they
            # describe.
            committed_reference = new_reference if new_reference is not None else self.best_reference
            if committed_reference is not None:
                commit_metric: float = float(committed_reference["metric"])
                commit_epoch: int | None = int(committed_reference["epoch"])
                commit_hash: str | None = str(committed_reference["sha256"])
            else:
                commit_metric = self.best_metric
                commit_epoch = None
                commit_hash = None
                if commit_metric != -float("inf"):
                    # A finite (or +inf) best_metric with no verifiable selected
                    # state would silently suppress every future selection.
                    raise ValueError(
                        f"Inconsistent selection state at epoch {epoch + 1}: best_metric="
                        f"{commit_metric!r} with no committed selected-best reference. A missing "
                        "selection must use the explicit -inf sentinel."
                    )

            # The row that goes into the checkpoint must already describe the
            # commit it is part of.  Saving the pre-commit row instead recorded
            # checkpoint_committed=False and completed_epochs=<previous> inside
            # last.pt, and a resume then carried that stale tail into the final
            # report even though the durable report said otherwise.  So the
            # complete committed row is proposed here, written, and adopted in
            # memory only once the write succeeded -- never patched afterwards
            # to claim a state the bytes do not contain.
            proposed_completed = epoch + 1
            committed_row = {
                **row,
                "checkpoint_committed": True,
                "status": "checkpoint_committed",
                "completed_epochs": proposed_completed,
                "last_checkpoint_committed_epoch": proposed_completed,
                "best_selection_status": selection_status,
                "selection_metric": selection_metric_used,
                "selection_metric_value": (
                    None
                    if not selection_active
                    else (None if not np.isfinite(metric) else float(metric))
                ),
                "best_metric": commit_metric,
                "best_epoch": commit_epoch,
                "best_checkpoint_hash": commit_hash,
                # The checkpoint commit and the later required JSON report
                # commit are distinct events, and the checkpoint cannot claim
                # the second one has happened.
                "report_committed": False,
            }
            # The history is normalized with exactly the report's semantics, so
            # the checkpoint and pipeline_report.json describe the same rows --
            # including "an undefined metric is an explicit null, never a
            # number and never zero".  Without this the checkpoint kept NaN
            # where the report wrote null, and a restored row differed from the
            # baseline it was supposed to reproduce.
            proposed_history = _json_safe(
                [dict(entry) for entry in self.report["epochs"][:-1]] + [dict(committed_row)]
            )

            save_checkpoint(
                output_dir / "last.pt",
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch + 1,
                global_step=self.global_step,
                optimizer_step=self.optimizer_step,
                config=self.config.to_dict(),
                extra={
                    **commit_extra_common,
                    "role": "last",
                    "selection_metric": selection_metric_used,
                    "best_selection_status": selection_status,
                    COMMITTED_BEST_REFERENCE_KEY: committed_reference,
                    "best_metric": commit_metric,
                    "best_epoch": commit_epoch,
                    "best_checkpoint_path": PUBLIC_BEST_NAME if committed_reference is not None else None,
                    "best_checkpoint_hash": commit_hash,
                    # Verified progress history: a resumed attempt restores the
                    # rows it already committed instead of reporting only the
                    # epochs of the new attempt.
                    "epoch_history": proposed_history,
                },
            )
            # Adopt the proposed commit now that the bytes are durable.
            row.update(committed_row)
            self.last_checkpoint_committed_epoch = proposed_completed

            # last.pt is now durable, so the live selection state may advance
            # and the public alias may be republished -- from the hash-verified
            # snapshot only.  A failure here leaves last.pt authoritative and
            # the alias stale, which resume repairs.
            if new_reference is not None:
                self.best_reference = new_reference
                self.best_metric = commit_metric
                self.best_epoch = commit_epoch
                self.best_checkpoint_hash = commit_hash
            if committed_reference is not None:
                self.failure_stage = "best_alias"
                publish_best_alias(committed_reference, output_dir)
                self.written_best_path = output_dir / PUBLIC_BEST_NAME

            # 4. Report durability: the atomic report publishes the PROPOSED
            # epoch count, and the live committed counter advances only after
            # that write succeeds.  A post-commit reader (the W&B adapter, or a
            # crash investigator) therefore never sees a durable report that
            # still carries the previous count.
            self.failure_stage = "report"
            previous_completed = self.completed_epochs
            self.report["last_checkpoint_committed_epoch"] = self.last_checkpoint_committed_epoch
            self.report["completed_epochs"] = proposed_completed
            self.report["telemetry"] = self.telemetry_summary
            try:
                atomic_write_json(
                    report_dir / "pipeline_report.json", _json_safe(self.report), indent=2, sort_keys=True
                )
            except BaseException:
                # Restore the proposed metadata: nothing durable advanced.  The
                # checkpoint commit itself stands and stays truthful.
                self.report["completed_epochs"] = previous_completed
                raise
            row["report_committed"] = True
            self.completed_epochs = proposed_completed

            # 5. W&B logging strictly follows the required checkpoint and
            # progress-report commits, and cannot disturb the training streams.
            self.failure_stage = "logging"
            log_payload = self._build_wandb_payload(epoch, interval, interval_idx, train_stats, val_stats)
            with self._preserving_rng():
                self.logger.log(log_payload)

            print(
                f"[{interval.name}] Epoch {epoch+1:03d}/{self.schedule.total_epochs:03d} "
                f"train_loss={train_stats['train_loss']:.4f} primary_metric={val_stats['primary_metric']:.4f} "
                f"opt_step={self.optimizer_step}"
            )

            if max_steps is not None and self.optimizer_step >= int(max_steps):
                self.completed = False
                self.incomplete_reason = "max_steps_reached"
                print(f"Global max_steps ({max_steps}) reached; halting bounded smoke run.")
                break
        else:
            if max_val_batches is not None:
                self.completed = False
                self.incomplete_reason = "max_val_batches_restricted"
            elif self.written_best_path is None or not self.written_best_path.exists():
                self.completed = False
                self.incomplete_reason = "missing_best_checkpoint"
            else:
                self.completed = not self.incomplete_epoch

        if not self.completed:
            self.report["completed"] = False
            self.report["incomplete_reason"] = self.incomplete_reason or "incomplete"

        # Calibration firewall: do not calibrate incomplete runs
        if self.completed and self.config.calibration.enabled:
            print("\n=== COMPLETE RUN: Executing Post-Training Calibration ===")
            self.run_post_training_calibration(output_dir, report_dir)
        elif self.completed and not self.config.calibration.enabled:
            # Calibration disabled: check if diagnostics requested
            self.report["calibration"] = {
                "status": "skipped",
                "reason": "calibration_disabled",
                "artifact_path": None,
                "calibrated_tau": None,
            }
            if self.config.diagnostics.evaluate_headroom or self.config.diagnostics.evaluate_decomposition:
                print("\n=== COMPLETE RUN: Executing Diagnostics with Configured Fixed Tau (Calibration Disabled) ===")
                self.run_independent_diagnostics(output_dir, report_dir)
            else:
                self.report["final_diagnostics"] = {
                    "status": "skipped",
                    "reason": "diagnostics_disabled",
                    "headroom": None,
                    "decomposition": None,
                }
        else:
            if not self.completed:
                print("\n=== SKIPPING CALIBRATION: Run was incomplete (bounded smoke). ===")
            self.report["calibration"] = None
            self.report["final_diagnostics"] = {
                "status": "skipped",
                "reason": "incomplete_run",
                "headroom": None,
                "decomposition": None,
            }

        # Full run completed=true is committed only after best binding, calibration, diagnostics succeed
        self.report["completed"] = self.completed
        self.report["status"] = "succeeded" if self.completed else "incomplete"
        self.report["completed_epochs"] = self.completed_epochs
        self.report["last_checkpoint_committed_epoch"] = self.last_checkpoint_committed_epoch

        # 1. Commit final research report first (telemetry finalization explicitly pending/as-of-commit)
        telemetry_snap = dict(self.telemetry_summary)
        telemetry_snap["finalization_status"] = "pending"
        self.report["telemetry"] = telemetry_snap

        self.failure_stage = "report"
        report_path = report_dir / "pipeline_report.json"
        atomic_write_json(report_path, _json_safe(self.report), indent=2, sort_keys=True)

        # 2. Finalize logger summary + finish(exit_code=0).  Telemetry cannot
        #    perturb the training streams, and an adapter that raises instead of
        #    recording its own error is still recorded here -- a lost telemetry
        #    error must not read as a clean finalization.
        if hasattr(self, "logger") and self.logger is not None:
            with self._preserving_rng():
                try:
                    self.logger.set_summary({
                        "completed": self.completed,
                        "status": "succeeded" if self.completed else "incomplete",
                        "completed_epochs": self.completed_epochs,
                        "last_checkpoint_committed_epoch": self.last_checkpoint_committed_epoch,
                    })
                except Exception as sec_exc:
                    print(f"[lifecycle] Secondary error setting logger summary: {sec_exc}", file=sys.stderr)
                    self._record_logger_adapter_error("set_summary", sec_exc)

                try:
                    self.logger.finish(exit_code=0)
                except Exception as sec_exc:
                    print(f"[lifecycle] Secondary error finishing logger: {sec_exc}", file=sys.stderr)
                    self._record_logger_adapter_error("finish", sec_exc)

        # 3. Optional best-effort atomic telemetry-status refresh.  The status
        #    is whatever the logger actually reports, so a failed finish is
        #    never described as finalized.
        telemetry_final = dict(self.telemetry_summary)
        telemetry_final["finalization_status"] = self._finalization_status()
        telemetry_final["wandb_identity"] = self._refresh_wandb_identity()
        self.report["telemetry"] = telemetry_final

        try:
            atomic_write_json(report_path, _json_safe(self.report), indent=2, sort_keys=True)
        except Exception as sec_exc:
            print(f"[lifecycle] Secondary error refreshing telemetry status in pipeline_report: {sec_exc}", file=sys.stderr)
            if hasattr(self, "logger") and self.logger is not None:
                if hasattr(self.logger, "telemetry_errors"):
                    self.logger.telemetry_errors.append(f"post_finish_telemetry_refresh: {sec_exc}")
            self.report["telemetry"]["refresh_warning"] = str(sec_exc)

        return self.report

    def run_post_training_calibration(self, output_dir: Path, report_dir: Path) -> None:
        """Run post-training calibration binding the selected checkpoint."""
        if not self.completed:
            raise RuntimeError(
                f"Cannot run post-training calibration on incomplete run: reason={self.incomplete_reason or 'incomplete'}"
            )
        if self.written_best_path is None or not self.written_best_path.exists():
            raise RuntimeError(
                "A missing best checkpoint cannot be used for calibration; full validation did not produce a selected best.pt checkpoint during the gated schedule, and silent fallback to last.pt is strictly forbidden."
            )
        legacy_cfg_c = self.config.to_legacy_dataset_config(self.schedule.intervals[-1])

        # Bind selected best checkpoint strictly; silent fallback to last.pt is forbidden
        self.failure_stage = "checkpoint_binding"
        candidates: list[tuple[str, Path]] = [("best", self.written_best_path)]

        binding = bind_evaluation_checkpoint(
            self.model,
            candidates,
            map_location=self.device,
            config=legacy_cfg_c,
        )
        self.report["checkpoint_binding"] = binding.as_dict()
        verify_bound_state(self.model, binding, boundary="validation_transition_cache")

        _, val_loader = self.get_loaders(self.schedule.intervals[-1])

        self.failure_stage = "cache_collection"
        cache = collect_validation_transition_cache(
            self.model,
            val_loader,
            self.device,
            t_max=self.config.training.rollout.max_turns,
            disable_tqdm=self.disable_tqdm,
        )

        cal_cfg = self.config.calibration
        thresholds = np.linspace(
            cal_cfg.threshold_min,
            cal_cfg.threshold_max,
            cal_cfg.threshold_steps,
        )

        neutral_margin = self.config.training.audit_loss.neutral_margin
        lineage = build_lineage(
            binding=binding.as_dict(),
            loader=val_loader,
            split_name=self.config.dataset.val_split,
            cache=cache,
            metric_space=METRIC_SPACE_SLICE_PROXY,
            neutral_margin=neutral_margin,
            t_max=self.config.training.rollout.max_turns,
            max_batches=None,
            batch_size=getattr(val_loader, "batch_size", None),
            observed_samples=int(cache["initial_dice"].shape[0]),
        )
        cache["lineage"] = lineage

        verify_bound_state(self.model, binding, boundary="calibration_sweep")
        self.failure_stage = "calibration_sweep"
        rows = sweep_thresholds(cache, thresholds, neutral_margin=neutral_margin)
        best_threshold = select_threshold(rows)
        calibrated_tau = float(best_threshold["tau_accept"])

        self.failure_stage = "calibration_write"
        calibration_path = report_dir / "calibration.json"
        calibration_payload = save_calibration(
            calibration_path,
            tau_accept=calibrated_tau,
            neutral_margin=neutral_margin,
            source_split=self.config.dataset.val_split,
            checkpoint_path=binding.path,
            t_max=self.config.training.rollout.max_turns,
            threshold_grid=thresholds,
            selected_row=best_threshold,
            metric_space=METRIC_SPACE_SLICE_PROXY,
            lineage=lineage,
            extra={"pipeline": "unified_trainer", "schema_version": 1},
        )

        self.failure_stage = "calibration_lineage"
        # Roundtrip verification
        reloaded = load_calibration(calibration_path)
        expected_lineage = build_expected_lineage(
            binding=binding,
            loader=val_loader,
            split_name=self.config.dataset.val_split,
            metric_contract=cache.get("metric_contract"),
            metric_space=METRIC_SPACE_SLICE_PROXY,
            neutral_margin=neutral_margin,
            t_max=self.config.training.rollout.max_turns,
            batch_size=getattr(val_loader, "batch_size", None),
            observed_samples=int(cache["initial_dice"].shape[0]),
        )
        roundtrip_verification = verify_calibration_lineage(
            reloaded,
            expected_lineage,
            cohort_policy=CohortPolicy(role=COHORT_ROLE_CALIBRATION),
            artifact_name=f"calibration artifact {calibration_path}",
        )

        self.failure_stage = "cache_save"
        atomic_save_torch(cache, report_dir / "validation_transitions.pt")

        self.report["calibration"] = {
            "best": best_threshold,
            "artifact_path": str(calibration_path),
            "calibrated_tau": calibrated_tau,
            "lineage_verification": roundtrip_verification,
            "selection_bias_caveat": SELECTION_BIAS_CAVEAT,
        }

        # Final diagnostics binding the selected best checkpoint
        self.failure_stage = "diagnostics"
        verify_bound_state(self.model, binding, boundary="final_diagnostics")
        final_diagnostics: dict[str, Any] = {}
        if self.config.diagnostics.evaluate_headroom:
            final_diagnostics["headroom"] = evaluate_annotation_headroom(
                self.model,
                val_loader,
                self.device,
                neutral_margin=neutral_margin,
                disable_tqdm=self.disable_tqdm,
            )
        if self.config.diagnostics.evaluate_decomposition:
            final_diagnostics["decomposition"] = evaluate_audit_decomposition(
                self.model,
                val_loader,
                self.device,
                tau_accept=calibrated_tau,
                t_max=self.config.training.rollout.max_turns,
                neutral_margin=neutral_margin,
                disable_tqdm=self.disable_tqdm,
            )
        self.report["final_diagnostics"] = final_diagnostics
        verify_bound_state(self.model, binding, boundary="final_diagnostics_complete")

    def run_independent_diagnostics(self, output_dir: Path, report_dir: Path) -> None:
        """Execute independent diagnostics binding selected best checkpoint with configured fixed tau."""
        if not self.completed:
            raise RuntimeError(
                f"Cannot run post-training diagnostics on incomplete run: reason={self.incomplete_reason or 'incomplete'}"
            )
        if self.written_best_path is None or not self.written_best_path.exists():
            raise RuntimeError(
                "A missing best checkpoint cannot be used for diagnostics; silent fallback to last.pt is strictly forbidden."
            )
        legacy_cfg_c = self.config.to_legacy_dataset_config(self.schedule.intervals[-1])
        self.failure_stage = "checkpoint_binding"
        candidates: list[tuple[str, Path]] = [("best", self.written_best_path)]
        binding = bind_evaluation_checkpoint(
            self.model,
            candidates,
            map_location=self.device,
            config=legacy_cfg_c,
        )
        self.report["checkpoint_binding"] = binding.as_dict()
        verify_bound_state(self.model, binding, boundary="independent_diagnostics")

        _, val_loader = self.get_loaders(self.schedule.intervals[-1])
        neutral_margin = self.config.training.audit_loss.neutral_margin
        fixed_tau = float(getattr(self.config.training.audit_loss, "tau_accept", 0.0))

        self.failure_stage = "diagnostics"
        final_diagnostics: dict[str, Any] = {
            "status": "executed_with_fixed_tau",
            "tau_accept": fixed_tau,
        }
        if self.config.diagnostics.evaluate_headroom:
            final_diagnostics["headroom"] = evaluate_annotation_headroom(
                self.model,
                val_loader,
                self.device,
                neutral_margin=neutral_margin,
                disable_tqdm=self.disable_tqdm,
            )
        if self.config.diagnostics.evaluate_decomposition:
            final_diagnostics["decomposition"] = evaluate_audit_decomposition(
                self.model,
                val_loader,
                self.device,
                tau_accept=fixed_tau,
                t_max=self.config.training.rollout.max_turns,
                neutral_margin=neutral_margin,
                disable_tqdm=self.disable_tqdm,
            )
        self.report["final_diagnostics"] = final_diagnostics
        verify_bound_state(self.model, binding, boundary="final_diagnostics_complete")
