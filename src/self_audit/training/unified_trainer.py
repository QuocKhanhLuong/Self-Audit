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

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any, Mapping
import uuid

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

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
    source_content_signature,
    split_membership_descriptor,
)
from self_audit.training._utils import (
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
    move_batch,
    print_model_parameter_summary,
    resolve_amp,
    resolve_device,
    save_checkpoint,
    seed_everything,
    validate_dataset_splits,
    verify_bound_state,
)
from self_audit.training.finetune_joint import (
    compute_joint_losses,
    collect_validation_transition_cache,
    validate_phase_c,
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
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
        self.best_checkpoint_hash: str | None = None
        self.best_epoch: int | None = None

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

        wandb_cfg = config.logging.wandb
        self.logger = WandbLogger(
            enabled=wandb_cfg.enabled,
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            run_name=wandb_cfg.run_name,
            config=config.to_dict(),
            mode=wandb_cfg.mode,
        )

    @property
    def source_signature(self) -> str | None:
        """Obtain current source content signature using existing provenance helpers."""
        try:
            sig = source_content_signature()
            return sig.get("source_content_signature")
        except Exception:
            return None

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

        # 4. Guard and validate dataset protocol splits if data_root exists
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

    def compute_batch_loss(
        self,
        batch: dict[str, torch.Tensor],
        interval: ScheduleInterval,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Compute schedule-driven batch loss based on the active interval objective."""
        objective = interval.objective

        if objective == "weighted_a0_a3":
            with autocast_context(enabled=self.amp_enabled, device=self.device, dtype=self.amp_dtype):
                output = self.model.forward_annotation(batch["image"]) if hasattr(self.model, "forward_annotation") else self.model(batch["image"])
                loss, parts = phase_a_loss(
                    output,
                    batch["mask"],
                    stage_weights=self.config.training.annotation_loss.stage_weights,
                )
                loss = loss * float(interval.annotation_weight)
            return loss, {
                "loss": float(loss.detach()),
                "annotation_loss": float(loss.detach()),
                "audit_loss": 0.0,
                "parts": parts,
                "objective": objective,
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
            if loss is None:
                return None, {
                    "loss": 0.0,
                    "annotation_loss": 0.0,
                    "audit_loss": 0.0,
                    "transitions": 0,
                    "objective": objective,
                }
            loss = loss * float(interval.audit_weight)
            return loss, {
                "loss": float(loss.detach()),
                "annotation_loss": 0.0,
                "audit_loss": float(loss.detach()),
                "details": details,
                "objective": objective,
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
            return loss, {
                "loss": float(loss.detach()),
                "annotation_loss": float(details["annotation_loss"]),
                "audit_loss": float(details["audit_loss"]),
                "objective": objective,
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
            loss, details = self.compute_batch_loss(batch, interval)

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
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Checkpoint payload must be a mapping, got {type(payload).__name__}")

        # 1. Resumability validation (reject bounded smoke checkpoints and incomplete epochs)
        resumable = payload.get("resumable")
        incomplete_epoch = payload.get("incomplete_epoch")
        if resumable is False or incomplete_epoch is True:
            raise ValueError(
                f"Cannot resume from non-resumable or incomplete checkpoint: {checkpoint_path}"
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
        if saved_source is not None and curr_source is not None and saved_source != "UNKNOWN" and curr_source != "UNKNOWN":
            if saved_source != curr_source:
                raise ValueError(
                    f"Source signature mismatch on resume: checkpoint was produced with source {saved_source}, "
                    f"but current source is {curr_source}"
                )

        completed_epoch = int(payload.get("epoch", 0))
        self.global_epoch = completed_epoch
        self.global_step = int(payload.get("global_step", 0))
        self.optimizer_step = int(payload.get("optimizer_step", 0))

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
                self.optimizer.load_state_dict(payload["optimizer"])

                if "scheduler" not in payload or payload["scheduler"] is None:
                    raise ValueError(
                        f"Checkpoint missing required scheduler state for non-boundary resume at epoch {self.global_epoch}"
                    )
                self.scheduler.load_state_dict(payload["scheduler"])

                if self.scaler is not None and self.scaler.is_enabled():
                    if "scaler" not in payload or payload["scaler"] is None:
                        raise ValueError(
                            f"Checkpoint missing required scaler state for non-boundary resume at epoch {self.global_epoch}"
                        )
                    self.scaler.load_state_dict(payload["scaler"])

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
        sibling_best = path.parent / "best.pt"

        configured_output_dir = Path(self.config.checkpoint.output_dir).resolve()
        is_relocated = (path.parent.resolve() != configured_output_dir)

        if is_relocated and configured_output_dir.exists():
            existing_files = [p for p in configured_output_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
            unrelated = [
                p.name for p in existing_files
                if not (p.name == "best.pt" and saved_best_hash is not None and hashlib.sha256(p.read_bytes()).hexdigest() == saved_best_hash)
            ]
            if unrelated:
                raise FileExistsError(
                    f"Output relocation across resume refused: target directory '{configured_output_dir}' "
                    f"already contains existing file(s) ({unrelated[:5]}) that would be clobbered. "
                    "Specify an empty or clean output_dir to preserve existing files."
                )

        if saved_best_hash is not None:
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
            try:
                best_payload = torch.load(sibling_best, map_location=self.device, weights_only=True)
            except Exception:
                best_payload = torch.load(sibling_best, map_location=self.device, weights_only=False)
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
                target_best = configured_output_dir / "best.pt"
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

            self.best_checkpoint_hash = saved_best_hash
            self.best_epoch = int(saved_best_epoch) if saved_best_epoch is not None else None
            self.best_metric = float(saved_best_metric) if saved_best_metric is not None else float(best_payload.get("best_metric", -float("inf")))
        else:
            self.written_best_path = None
            self.best_checkpoint_hash = None
            self.best_epoch = None
            if saved_best_metric is not None:
                self.best_metric = float(saved_best_metric)
            else:
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
            _restore_rng_state(rng)
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
        # Guard parameters and existing checkpoint artifacts BEFORE any file modification
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

        self.global_epoch = start_epoch

        for epoch in range(start_epoch, self.schedule.total_epochs):
            self.global_epoch = epoch
            interval = self.schedule.get_interval(epoch)
            interval_idx = self.schedule.get_interval_index(epoch)

            # Check if entering a new interval or reset boundary
            train_loader, val_loader = self.get_loaders(interval)
            if self.current_interval_index != interval_idx:
                self.setup_interval_optimizer_and_scheduler(interval, len(train_loader))
                self.current_interval_index = interval_idx

            train_stats = self.train_epoch(interval, train_loader, max_steps=max_steps)
            val_stats = self.validate_epoch(interval, val_loader, max_val_batches=max_val_batches)

            row = {
                "global_epoch": epoch + 1,
                "interval": interval.name,
                "interval_index": interval_idx,
                "global_step": self.global_step,
                "optimizer_step": self.optimizer_step,
                **train_stats,
                **val_stats,
            }
            self.report["epochs"].append(row)

            # W&B Logging with canonical monotonic global keys
            log_payload = self._build_wandb_payload(epoch, interval, interval_idx, train_stats, val_stats)
            self.logger.log(log_payload)

            # Best checkpoint selection ONLY from epochs >= best_selection_min_epoch (120) and ONLY with unrestricted validation
            # Update best tracker and save best.pt BEFORE saving last.pt so last.pt has fresh best_metric
            cohort_desc = self.get_cohort_descriptor(interval)
            if max_val_batches is None and epoch >= self.config.checkpoint.best_selection_min_epoch:
                metric = float(val_stats.get("final_foreground_macro_dice", val_stats.get("primary_metric", -float("inf"))))
                if np.isfinite(metric) and metric > self.best_metric:
                    self.best_metric = metric
                    self.best_epoch = epoch + 1
                    best_path = output_dir / "best.pt"
                    save_checkpoint(
                        best_path,
                        self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch + 1,
                        global_step=self.global_step,
                        optimizer_step=self.optimizer_step,
                        config=self.config.to_dict(),
                        extra={
                            "run_id": self.run_id,
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
                            "best_metric": self.best_metric,
                            "best_epoch": self.best_epoch,
                        },
                    )
                    self.written_best_path = best_path
                    self.best_checkpoint_hash = hashlib.sha256(best_path.read_bytes()).hexdigest()

            # Checkpointing: last.pt at every epoch AFTER best.pt update
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
                    "run_id": self.run_id,
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
                    "best_metric": self.best_metric,
                    "best_epoch": self.best_epoch,
                    "best_checkpoint_path": "best.pt" if self.written_best_path is not None else None,
                    "best_checkpoint_hash": self.best_checkpoint_hash,
                },
            )

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

        self.report["completed"] = self.completed
        if not self.completed:
            self.report["incomplete_reason"] = self.incomplete_reason or "interrupted"

        # Calibration firewall: do not calibrate incomplete runs
        if self.completed and self.config.calibration.enabled:
            print("\n=== COMPLETE RUN: Executing Post-Training Calibration ===")
            self.run_post_training_calibration(output_dir, report_dir)
        else:
            if not self.completed:
                print("\n=== SKIPPING CALIBRATION: Run was incomplete (bounded smoke). ===")
            self.report["calibration"] = None

        report_path = report_dir / "pipeline_report.json"
        report_path.write_text(json.dumps(_json_safe(self.report), indent=2), encoding="utf-8")
        self.logger.finish()
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
        rows = sweep_thresholds(cache, thresholds, neutral_margin=neutral_margin)
        best_threshold = select_threshold(rows)
        calibrated_tau = float(best_threshold["tau_accept"])

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

        torch.save(cache, report_dir / "validation_transitions.pt")

        self.report["calibration"] = {
            "best": best_threshold,
            "artifact_path": str(calibration_path),
            "calibrated_tau": calibrated_tau,
            "lineage_verification": roundtrip_verification,
            "selection_bias_caveat": SELECTION_BIAS_CAVEAT,
        }

        # Final diagnostics binding the selected best checkpoint
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
