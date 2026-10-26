"""Native-volume student/EMA transaction engine.

The engine deliberately exposes one transaction boundary: supervised update,
fixed detached teacher targets, at most one pseudo update, then one EMA update.
No region or temporal branch gets its own optimizer step.
"""

from __future__ import annotations

import copy
import random
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from ..losses import strong_intensity
from .config import config_sha256, validate_volume_config
from .losses import combine_accumulated_losses, pseudo_volume_ce, pseudo_weight_ramp, supervised_volume_loss
from .optim import build_optimizers, build_schedulers, safe_optimizer_step
from .regions import RegionTargets, region_consistency_loss
from .temporal import temporal_consistency


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _require_data_pin(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"volume training requires a non-null 64-hex {name} pin")
    return value.lower()


def _tensor(batch: Mapping[str, Any], key: str, *, required: bool = True) -> Tensor | None:
    value = batch.get(key)
    if value is None and required:
        raise ValueError(f"batch is missing {key}")
    if value is not None and not isinstance(value, Tensor):
        value = torch.as_tensor(value)
    return value


def _validate_image(image: Tensor) -> None:
    if image.ndim != 5 or image.shape[1] != 3:
        raise ValueError("image must have shape B,3,Z,H,W")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("image must be finite floating point")
    if torch.any(image < 0) or torch.any(image > 1):
        raise ValueError("image must be normalized to [0,1]")


def _valid_support(batch: Mapping[str, Any], *, shape: tuple[int, ...], device: torch.device) -> Tensor:
    valid = _tensor(batch, "geometric_valid", required=False)
    if valid is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if valid.shape != shape or valid.dtype is not torch.bool:
        raise ValueError("geometric_valid must be bool B,Z,H,W matching image")
    return valid.to(device=device)


def _label_support(batch: Mapping[str, Any], labels: Tensor, geometric_valid: Tensor) -> Tensor:
    label_valid = _tensor(batch, "label_valid", required=False)
    if label_valid is None:
        label_valid = labels != 255
    if label_valid.shape != labels.shape or label_valid.dtype is not torch.bool:
        raise ValueError("label_valid must be bool B,Z,H,W matching labels")
    return label_valid.to(device=labels.device) & geometric_valid & (labels != 255)


def _slice_batch(batch: Mapping[str, Any], sl: slice, batch_size: int) -> dict[str, Any]:
    """Slice tensor/list batch fields while retaining scalar metadata."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            out[key] = value[sl]
        elif isinstance(value, list) and len(value) == batch_size:
            out[key] = value[sl]
        else:
            out[key] = value
    return out


def _supervised_loss(logits: Tensor, labels: Tensor, geometric_valid: Tensor, label_valid: Tensor, mode: str) -> tuple[Tensor, dict[str, float]]:
    result = supervised_volume_loss(logits, labels, geometric_valid, label_valid, mode=mode, return_metrics=True)
    if isinstance(result, tuple):
        loss, metrics = result
    else:
        loss, metrics = result, {}
    if not isinstance(loss, Tensor):
        raise ValueError("supervised_volume_loss must return a tensor")
    return loss, {str(k): float(v) for k, v in dict(metrics).items()}


def _ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for t, s in zip(teacher.parameters(), student.parameters(), strict=True):
            t.mul_(decay).add_(s, alpha=1.0 - decay)
        for t, s in zip(teacher.buffers(), student.buffers(), strict=True):
            t.copy_(s)


class VolumeTrainer:
    """Run bounded native-volume training transactions.

    ``batch`` and ``unlabeled_batch`` are dictionaries returned by the volume
    data API.  The engine never inspects labels from the latter.
    """

    def __init__(self, model: nn.Module, config: Mapping[str, Any], *, teacher: nn.Module | None = None,
                 optimizers: list[torch.optim.Optimizer] | None = None,
                 schedulers: list[Any] | None = None, scaler: torch.amp.GradScaler | None = None,
                 region_manager: Any | None = None, sampler_state: Any | None = None):
        self.config = validate_volume_config(config, allow_full_run=True)
        self.model = model
        self.device = next(model.parameters()).device
        if self.config["training"]["precision"] == "amp" and self.device.type != "cuda":
            raise RuntimeError("AMP requires CUDA; CPU execution must declare fp32")
        self.teacher = copy.deepcopy(model) if teacher is None else teacher
        self.teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.region_manager = region_manager
        if self.region_manager is None and self.config["regions"].get("mode") != "none":
            from .regions import RegionManager
            self.region_manager = RegionManager(self.config["regions"])
        self.sampler_state = sampler_state
        if optimizers is None:
            optimizers, metadata = build_optimizers(model, self.config["optimization"])
            self.optimizer_metadata = metadata
        else:
            self.optimizer_metadata = []
        self.optimizers = list(optimizers)
        if schedulers is None:
            scheduler_config = dict(self.config["optimization"])
            scheduler_config.update({"total_steps": int(self.config["training"]["steps"]),
                                     "pseudo_delay": int(self.config["training"]["pseudo_delay"])})
            schedulers, scheduler_metadata = build_schedulers(self.optimizers, scheduler_config)
            self.scheduler_metadata = scheduler_metadata
        else:
            self.scheduler_metadata = {}
        self.schedulers = list(schedulers or [])
        self.scaler = scaler
        if self.scaler is None and self.config["training"]["precision"] == "amp" and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler("cuda")
        self.outer_step = 0
        self.supervised_updates = 0
        self.supervised_extra_updates = 0
        self.pseudo_updates = 0
        self.optimizer_updates = 0
        self.ema_updates = 0
        self.skipped_supervised_updates = 0
        self.skipped_pseudo_updates = 0
        self.history: list[dict[str, Any]] = []

    def _autocast(self):
        if self.config["training"]["precision"] == "amp" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def _step_optimizers(self, *, scaler: torch.amp.GradScaler | None = None) -> tuple[float, float, bool]:
        clip = float(self.config["optimization"]["clip_norm"])
        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if not params:
            return 0.0, 0.0, False
        if scaler is None:
            receipt = safe_optimizer_step(self.optimizers, clip_norm=clip, has_support=True)
            return float(receipt["gradient_norm"]), float(receipt["parameter_update_norm"]), bool(receipt["updated"])
        if scaler is not None:
            for optimizer in self.optimizers:
                scaler.unscale_(optimizer)
            if not all(bool(torch.isfinite(p.grad).all().detach()) for p in params if p.grad is not None):
                self._zero_grad()
                scaler.update()
                return 0.0, 0.0, False
        if params:
            norm = torch.nn.utils.clip_grad_norm_(params, clip, error_if_nonfinite=True)
        else:
            norm = torch.tensor(0.0, device=self.device)
        before = [parameter.detach().clone() for parameter in params]
        for optimizer in self.optimizers:
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
        if scaler is not None:
            scaler.update()
        update_norm = torch.sqrt(sum(((parameter.detach() - old).float().pow(2).sum()
                                      for parameter, old in zip(params, before, strict=True)),
                                     torch.zeros((), device=self.device)))
        return float(norm.detach()), float(update_norm.detach()), True

    def _scheduler_step(self, outer_iteration: int) -> None:
        # Scheduler clocks are outer transactions, not microbatch updates.
        for scheduler in self.schedulers:
            scheduler.step(outer_iteration)

    def _forward(self, image: Tensor, valid: Tensor) -> Tensor:
        return self.model(image, valid_mask=valid)

    def _teacher_forward(self, image: Tensor, valid: Tensor) -> Tensor:
        return self.teacher(image, valid_mask=valid)

    def _pseudo_weight(self, iteration: int) -> float:
        return pseudo_weight_ramp(
            iteration,
            base_weight=float(self.config["training"]["pseudo_weight"]),
            ramp_steps=int(self.config["training"]["pseudo_ramp_steps"]),
            pseudo_delay=int(self.config["training"]["pseudo_delay"]),
        )

    def _auxiliary_losses(self, *, pseudo_logits: Tensor, teacher_probabilities: Tensor,
                          unlabeled_batch: Mapping[str, Any], valid: Tensor,
                          temporal_teacher_probabilities: Tensor | None = None) -> tuple[dict[str, tuple[Tensor, int]], dict[str, float], bool]:
        """Compute independent region/temporal objectives and denominators.

        Each objective is returned with its own supported-volume/voxel count;
        callers must not weight a region or temporal term by pseudo CE support.
        """
        components: dict[str, tuple[Tensor, int]] = {}
        metrics: dict[str, float] = {}
        supported = False
        region_cfg = self.config["regions"]
        if self.region_manager is not None and region_cfg.get("mode") != "none" and float(region_cfg.get("weight", 0.0)) > 0:
            targets = self.region_manager.targets(
                unlabeled_batch["image"].to(pseudo_logits.device), valid, unlabeled_batch.get("metadata"),
                probabilities=teacher_probabilities.detach() if region_cfg.get("mode") == "ema" else None,
                features=None, step=self.outer_step,
            )
            region_losses: list[Tensor] = []
            region_valid_voxels = 0
            region_active = 0.0
            for bi in range(pseudo_logits.shape[0]):
                # RegionTargets is a detached value object; slice it so the
                # region denominator is per supported volume, excluding empty
                # volumes rather than diluting the objective by batch size.
                single_targets = RegionTargets(
                    mode=targets.mode,
                    region_labels=targets.region_labels[bi:bi + 1],
                    region_weights=targets.region_weights[bi:bi + 1],
                    target_probabilities=(
                        targets.target_probabilities[bi:bi + 1]
                        if targets.target_probabilities is not None else None
                    ),
                    cache_hash=targets.cache_hash,
                    step=targets.step,
                    num_regions=targets.num_regions[bi:bi + 1],
                    metrics=dict(targets.metrics),
                )
                one_loss, one_metrics = region_consistency_loss(
                    pseudo_logits[bi:bi + 1], single_targets,
                    geometric_valid=valid[bi:bi + 1],
                    loss_type=str(region_cfg.get("loss_type", "mse")),
                    weight=float(region_cfg["weight"]),
                )
                one_voxels = int(one_metrics.get("valid_region_voxels", 0.0))
                region_valid_voxels += one_voxels
                region_active += float(one_metrics.get("active_regions", 0.0))
                if one_voxels > 0:
                    region_losses.append(one_loss)
            if region_losses:
                region_loss = sum(region_losses) / float(len(region_losses))
                components["region"] = (region_loss, len(region_losses))
                supported = True
            else:
                region_loss = pseudo_logits.sum() * 0.0
            metrics.update({"region_loss": float(region_loss.detach()),
                            "active_regions": region_active,
                            "valid_region_voxels": float(region_valid_voxels),
                            "region_supported_volumes": float(len(region_losses))})
        temporal_cfg = self.config["temporal"]
        if temporal_cfg.get("mode") != "none" and float(temporal_cfg.get("weight", 0.0)) > 0:
            if "adjacent_image" not in unlabeled_batch or temporal_teacher_probabilities is None:
                raise ValueError("temporal objective requires adjacent_image and adjacent teacher probabilities")
            current = unlabeled_batch["image"].to(pseudo_logits.device)
            adjacent = unlabeled_batch["adjacent_image"].to(pseudo_logits.device)
            adjacent_valid = torch.as_tensor(
                unlabeled_batch["adjacent_geometric_valid"], device=pseudo_logits.device, dtype=torch.bool
            )
            temporal_losses: list[Tensor] = []
            temporal_counts: list[int] = []
            temporal_metrics: dict[str, float] = {}
            metadata = unlabeled_batch.get("metadata")
            for bi in range(pseudo_logits.shape[0]):
                one_metadata = metadata[bi:bi + 1] if isinstance(metadata, list) else metadata
                one_loss, one_metrics = temporal_consistency(
                    pseudo_logits[bi:bi + 1], temporal_teacher_probabilities[bi:bi + 1].detach(),
                    current[bi:bi + 1], adjacent[bi:bi + 1], valid[bi:bi + 1],
                    adjacent_valid[bi:bi + 1], one_metadata, temporal_cfg,
                )
                one_count = int(one_metrics.get("temporal_valid_voxels", 0))
                if one_count > 0:
                    temporal_losses.append(one_loss)
                    temporal_counts.append(one_count)
                for key in ("temporal_valid_voxels", "temporal_valid_fraction", "mean_displacement_voxels",
                            "mean_displacement_mm", "occlusion_fraction"):
                    value = one_metrics.get(key)
                    if isinstance(value, (int, float)):
                        temporal_metrics[key] = temporal_metrics.get(key, 0.0) + float(value)
            if temporal_losses:
                temporal_loss = sum(loss * (count / float(sum(temporal_counts)))
                                    for loss, count in zip(temporal_losses, temporal_counts, strict=True))
                components["temporal"] = (temporal_loss, int(sum(temporal_counts)))
                supported = True
            else:
                temporal_loss = pseudo_logits.sum() * 0.0
            count_total = int(sum(temporal_counts))
            metrics.update({"temporal_loss": float(temporal_loss.detach()),
                            "temporal_valid_voxels": float(count_total),
                            "temporal_valid_fraction": (float(count_total) / float(max(1, valid.numel()))),
                            "temporal_metric_voxels": float(valid.numel()),
                            "mean_displacement_voxels": temporal_metrics.get("mean_displacement_voxels", 0.0) / max(1, pseudo_logits.shape[0]),
                            "mean_displacement_mm": temporal_metrics.get("mean_displacement_mm", 0.0) / max(1, pseudo_logits.shape[0]),
                            "occlusion_fraction": temporal_metrics.get("occlusion_fraction", 0.0) / max(1, pseudo_logits.shape[0]),
                            "temporal_supported_voxels": float(count_total)})
        return components, metrics, supported

    def _extra_supervised_update(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Run C01's second labeled update before the single EMA update."""
        image = _tensor(batch, "image")
        assert image is not None
        _validate_image(image)
        image = image.to(self.device)
        valid = _valid_support(batch, shape=(image.shape[0], image.shape[2], image.shape[3], image.shape[4]), device=self.device)
        labels = _tensor(batch, "labels")
        assert labels is not None
        labels = labels.to(self.device, dtype=torch.long)
        label_valid = _label_support(batch, labels, valid)
        if not bool(label_valid.any().detach()):
            return {"committed": False, "loss": 0.0, "grad_norm": 0.0,
                    "update_norm": 0.0, "label_voxels": 0}
        accumulation = int(self.config["training"]["accumulation"])
        microbatch = int(self.config["training"]["microbatch"])
        if image.shape[0] != microbatch * accumulation:
            raise ValueError("extra labeled B must equal microbatch*accumulation; config microbatch cannot be ignored")
        chunks = [slice(start, start + microbatch) for start in range(0, image.shape[0], microbatch)]
        self._zero_grad()
        losses: list[Tensor] = []
        counts: list[int] = []
        with self._autocast():
            for chunk in chunks:
                count = int(label_valid[chunk].any(dim=(1, 2, 3)).sum().item())
                if count == 0:
                    continue
                loss, _ = _supervised_loss(self._forward(image[chunk], valid[chunk]), labels[chunk], valid[chunk], label_valid[chunk], self.config["loss"]["mode"])
                losses.append(loss); counts.append(count)
            if not losses:
                return {"committed": False, "loss": 0.0, "grad_norm": 0.0,
                        "update_norm": 0.0, "label_voxels": 0}
            total_loss = combine_accumulated_losses(losses, counts, mode=self.config["loss"]["mode"])
        if not torch.isfinite(total_loss):
            raise FloatingPointError("nonfinite C01 supervised-extra loss")
        if self.scaler is not None and self.device.type == "cuda":
            self.scaler.scale(total_loss).backward()
            grad_norm, update_norm, committed = self._step_optimizers(scaler=self.scaler)
        else:
            total_loss.backward()
            grad_norm, update_norm, committed = self._step_optimizers()
        return {"committed": committed, "loss": float(total_loss.detach()), "grad_norm": grad_norm,
                "update_norm": update_norm,
                "label_voxels": int(label_valid.sum().item())}

    def transaction(self, batch: Mapping[str, Any], unlabeled_batch: Mapping[str, Any] | None = None, *, iteration: int | None = None,
                    extra_labeled_batch: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Execute exactly one supervised/pseudo/EMA transaction."""
        if iteration is None:
            iteration = self.outer_step
        if type(iteration) is not int or iteration < 0:
            raise ValueError("iteration must be a non-negative integer")
        # Set the outer-iteration LR before either labeled or U update.  Both
        # updates in C01/EMA arms therefore observe the same warmup/decay rate.
        self._scheduler_step(self.outer_step)
        image = _tensor(batch, "image")
        assert image is not None
        _validate_image(image)
        image = image.to(self.device)
        valid = _valid_support(batch, shape=(image.shape[0], image.shape[2], image.shape[3], image.shape[4]), device=self.device)
        labels = _tensor(batch, "labels")
        assert labels is not None
        labels = labels.to(self.device, dtype=torch.long)
        if labels.shape != valid.shape:
            raise ValueError("labels must have shape B,Z,H,W")
        label_valid = _label_support(batch, labels, valid)
        if not bool(label_valid.any().detach()):
            self._zero_grad()
            self.skipped_supervised_updates += 1
            self.outer_step += 1
            return {"step": self.outer_step, "skipped": True, "skip_reason": "empty_supervision",
                    "supervised_updates": self.supervised_updates, "pseudo_updates": self.pseudo_updates,
                    "ema_updates": self.ema_updates, "optimizer_updates": self.optimizer_updates, "label_voxels": 0}
        self.model.train()
        self._zero_grad()
        with self._autocast():
            accumulation = int(self.config["training"]["accumulation"])
            microbatch = int(self.config["training"]["microbatch"])
            if image.shape[0] != microbatch * accumulation:
                raise ValueError("batch B must equal microbatch*accumulation; config microbatch cannot be ignored")
            chunks = [(slice(start, start + microbatch)) for start in range(0, image.shape[0], microbatch)]
            losses: list[Tensor] = []
            counts: list[int] = []
            metric_values: dict[str, float] = {}
            for chunk in chunks:
                chunk_support = label_valid[chunk]
                volume_count = int(chunk_support.any(dim=(1, 2, 3)).sum().item())
                if volume_count == 0:
                    continue
                chunk_loss, chunk_metrics = _supervised_loss(
                    self._forward(image[chunk], valid[chunk]), labels[chunk], valid[chunk],
                    label_valid[chunk], self.config["loss"]["mode"],
                )
                losses.append(chunk_loss)
                counts.append(volume_count)
                for key, value in chunk_metrics.items():
                    metric_values[key] = metric_values.get(key, 0.0) + float(value) * volume_count
            if not losses:
                self._zero_grad()
                self.skipped_supervised_updates += 1
                self.outer_step += 1
                return {"step": self.outer_step, "skipped": True, "skip_reason": "empty_supervision",
                        "supervised_updates": self.supervised_updates, "pseudo_updates": self.pseudo_updates,
                        "ema_updates": self.ema_updates, "label_voxels": 0}
            sup_loss = combine_accumulated_losses(losses, counts, mode=self.config["loss"]["mode"])
            metrics = {key: value / float(sum(counts)) for key, value in metric_values.items()}
            metrics["label_voxels"] = float(label_valid.sum().item())
        if not torch.isfinite(sup_loss):
            raise FloatingPointError("nonfinite supervised loss")
        if self.scaler is not None and self.device.type == "cuda":
            self.scaler.scale(sup_loss).backward()
            grad_norm, update_norm, committed = self._step_optimizers(scaler=self.scaler)
        else:
            sup_loss.backward()
            grad_norm, update_norm, committed = self._step_optimizers()
        if not committed:
            # AMP overflow/empty support is a skipped transaction: no pseudo
            # target refresh and no EMA update may occur.
            self.skipped_pseudo_updates += 1
            self.outer_step += 1
            return {"step": self.outer_step, "supervised_loss": float(sup_loss.detach()),
                    "supervised_grad_norm": grad_norm, "supervised_update_norm": update_norm, "skipped": True,
                    "supervised_updates": self.supervised_updates, "pseudo_updates": self.pseudo_updates,
                    "ema_updates": self.ema_updates, "optimizer_updates": self.optimizer_updates}
        self.supervised_updates += 1
        self.optimizer_updates += len(self.optimizers)
        extra_metrics: dict[str, Any] = {}
        if self.config["training"]["method"] == "supervised_extra":
            extra_result = self._extra_supervised_update(extra_labeled_batch or batch)
            if extra_result["committed"]:
                self.supervised_extra_updates += 1
                self.optimizer_updates += len(self.optimizers)
            extra_metrics = {"supervised_extra_loss": extra_result["loss"],
                             "supervised_extra_grad_norm": extra_result["grad_norm"],
                             "supervised_extra_update_norm": extra_result["update_norm"],
                             "extra_label_voxels": extra_result["label_voxels"],
                             "supervised_extra_updates": self.supervised_extra_updates}

        pseudo_metrics: dict[str, Any] = {"pseudo_accepted": 0, "pseudo_accepted_per_class": [0, 0, 0, 0],
                                          "pseudo_weight": 0.0, "teacher_student_disagreement": 0.0}
        method = self.config["training"]["method"]
        if unlabeled_batch is not None and method in {"ema_full", "ema_full_c01"} and iteration >= int(self.config["training"]["pseudo_delay"]):
            u_image = _tensor(unlabeled_batch, "image")
            assert u_image is not None
            _validate_image(u_image)
            u_image = u_image.to(self.device)
            view_rng = np.random.default_rng(int(self.config["training"]["seed"]) + int(iteration))
            student_u_image = (
                strong_intensity(u_image, view_rng)
                if self.config["training"]["pseudo_view"] == "strong_intensity"
                else u_image
            )
            u_valid = _valid_support(unlabeled_batch, shape=(u_image.shape[0], u_image.shape[2], u_image.shape[3], u_image.shape[4]), device=self.device)
            self.teacher.eval()
            with torch.no_grad():
                teacher_logits = self._teacher_forward(u_image, u_valid)
                teacher_prob = teacher_logits.softmax(1)
                confidence, pseudo = teacher_prob.max(1)
                accepted = (confidence >= float(self.config["training"]["pseudo_threshold"])) & u_valid
                adjacent_teacher_prob = None
                if self.config["temporal"].get("mode") != "none":
                    adjacent_image = _tensor(unlabeled_batch, "adjacent_image")
                    assert adjacent_image is not None
                    _validate_image(adjacent_image)
                    adjacent_image = adjacent_image.to(self.device)
                    adjacent_valid = _tensor(unlabeled_batch, "adjacent_geometric_valid")
                    assert adjacent_valid is not None
                    adjacent_valid = adjacent_valid.to(self.device, dtype=torch.bool)
                    adjacent_teacher_prob = self._teacher_forward(adjacent_image, adjacent_valid).softmax(1)
            accepted_count = int(accepted.sum().item())
            accepted_per_class = [int(((accepted) & (pseudo == cls)).sum().item()) for cls in range(4)]
            valid_voxels = int(u_valid.sum().item())
            pseudo_weight = self._pseudo_weight(iteration)
            pseudo_metrics.update({"pseudo_accepted": accepted_count,
                                   "pseudo_valid_voxels": valid_voxels,
                                   "pseudo_coverage": float(accepted_count / max(1, valid_voxels)),
                                   "pseudo_accepted_per_class": accepted_per_class,
                                   "teacher_student_disagreement": 0.0,
                                   "pseudo_weight": pseudo_weight})
            accumulation = int(self.config["training"]["accumulation"])
            microbatch = int(self.config["training"]["microbatch"])
            if u_image.shape[0] != microbatch * accumulation:
                raise ValueError("unlabeled B must equal microbatch*accumulation; config microbatch cannot be ignored")
            u_chunks = [slice(start, start + microbatch) for start in range(0, u_image.shape[0], microbatch)]
            self._zero_grad()
            previous_training = self.model.training
            if self.config["training"]["pseudo_student_mode"] == "eval":
                self.model.eval()
            ce_losses: list[Tensor] = []
            ce_counts: list[int] = []
            region_losses: list[Tensor] = []
            region_counts: list[int] = []
            temporal_losses: list[Tensor] = []
            temporal_counts: list[int] = []
            pseudo_loss_metrics: dict[str, float] = {"accepted_voxels": 0.0}
            region_metric_sums: dict[str, float] = {}
            temporal_metric_sums: dict[str, float] = {}
            aux_supported = False
            disagreement_numer = 0
            disagreement_denom = 0
            for chunk in u_chunks:
                chunk_batch = _slice_batch(unlabeled_batch, chunk, u_image.shape[0])
                with self._autocast():
                    pseudo_logits = self._forward(student_u_image[chunk], u_valid[chunk])
                    chunk_pseudo_loss, chunk_metrics = pseudo_volume_ce(
                        pseudo_logits, pseudo[chunk], u_valid[chunk], accepted[chunk],
                        weight=self._pseudo_weight(iteration), return_metrics=True,
                    )
                    aux_batch = dict(chunk_batch)
                    aux_batch["image"] = student_u_image[chunk]
                    chunk_aux_components, chunk_aux_metrics, chunk_aux_supported = self._auxiliary_losses(
                        pseudo_logits=pseudo_logits,
                        teacher_probabilities=teacher_prob[chunk],
                        temporal_teacher_probabilities=(adjacent_teacher_prob[chunk] if adjacent_teacher_prob is not None else None),
                        unlabeled_batch=aux_batch,
                        valid=u_valid[chunk],
                    )
                chunk_accepted = int(accepted[chunk].sum().item())
                if chunk_accepted:
                    disagreement_numer += int(((pseudo_logits.detach().argmax(1) != pseudo[chunk]) & accepted[chunk]).sum().item())
                    disagreement_denom += chunk_accepted
                chunk_ce_count = int(chunk_metrics.get("accepted_voxels", 0))
                if pseudo_weight > 0.0 and chunk_ce_count > 0:
                    ce_losses.append(chunk_pseudo_loss)
                    ce_counts.append(chunk_ce_count)
                if "region" in chunk_aux_components:
                    region_loss, region_count = chunk_aux_components["region"]
                    region_losses.append(region_loss)
                    region_counts.append(region_count)
                if "temporal" in chunk_aux_components:
                    temporal_loss, temporal_count = chunk_aux_components["temporal"]
                    temporal_losses.append(temporal_loss)
                    temporal_counts.append(temporal_count)
                chunk_count = max(chunk_ce_count, 1 if chunk_aux_supported else 0)
                for key, value in chunk_metrics.items():
                    if isinstance(value, Tensor):
                        value = float(value.detach())
                    pseudo_loss_metrics[key] = pseudo_loss_metrics.get(key, 0.0) + float(value)
                for key, value in chunk_aux_metrics.items():
                    value = float(value)
                    if key.startswith("region") or key in {"active_regions", "valid_region_voxels"}:
                        # These are additive exposure counters.  The per-volume
                        # objective weighting above must never square them when
                        # microbatches are re-partitioned.
                        region_metric_sums[key] = region_metric_sums.get(key, 0.0) + value
                    elif key.startswith("temporal") or key in {"mean_displacement_voxels", "mean_displacement_mm", "occlusion_fraction"}:
                        temporal_metric_sums[key] = temporal_metric_sums.get(key, 0.0) + value
                    elif key == "temporal_metric_voxels":
                        temporal_metric_sums[key] = temporal_metric_sums.get(key, 0.0) + value
                aux_supported = aux_supported or chunk_aux_supported
            components: list[Tensor] = []
            combined_pseudo_ce = u_image.sum() * 0.0
            combined_region = u_image.sum() * 0.0
            combined_temporal = u_image.sum() * 0.0
            if ce_losses:
                combined_pseudo_ce = combine_accumulated_losses(ce_losses, ce_counts, mode="volume_mean")
                components.append(combined_pseudo_ce)
            if region_losses:
                combined_region = combine_accumulated_losses(region_losses, region_counts, mode="volume_mean")
                components.append(combined_region)
            if temporal_losses:
                combined_temporal = combine_accumulated_losses(temporal_losses, temporal_counts, mode="volume_mean")
                components.append(combined_temporal)
            pseudo_supported = bool(components)
            if pseudo_supported:
                combined_pseudo_loss = sum(components)
                pseudo_metrics["teacher_student_disagreement"] = float(disagreement_numer / max(1, disagreement_denom))
                if not torch.isfinite(combined_pseudo_loss):
                    raise FloatingPointError("nonfinite pseudo/auxiliary loss")
                if self.scaler is not None and self.device.type == "cuda":
                    self.scaler.scale(combined_pseudo_loss).backward()
                else:
                    combined_pseudo_loss.backward()
                pseudo_grad_norm, pseudo_update_norm, pseudo_committed = self._step_optimizers(scaler=self.scaler if self.device.type == "cuda" else None)
                if pseudo_committed:
                    self.pseudo_updates += 1
                    self.optimizer_updates += len(self.optimizers)
                pseudo_metrics.update({"pseudo_loss": float(combined_pseudo_loss.detach()),
                                       "pseudo_ce_loss": float(combined_pseudo_ce.detach()),
                                       "pseudo_grad_norm": pseudo_grad_norm,
                                       "pseudo_update_norm": pseudo_update_norm,
                                       "aux_grad_witness": float(pseudo_grad_norm) if aux_supported else 0.0,
                                       "pseudo_accepted_voxels": pseudo_loss_metrics.get("accepted_voxels", 0.0),
                                       "region_loss": float(combined_region.detach()),
                                       "region_active_regions": region_metric_sums.get("active_regions", 0.0),
                                       "region_valid_region_voxels": region_metric_sums.get("valid_region_voxels", 0.0),
                                       "region_supported_volumes": region_metric_sums.get("region_supported_volumes", 0.0),
                                       "temporal_loss": float(combined_temporal.detach()),
                                       "temporal_valid_voxels": temporal_metric_sums.get("temporal_valid_voxels", 0.0),
                                       "temporal_valid_fraction": (temporal_metric_sums.get("temporal_valid_voxels", 0.0)
                                                                    / max(1.0, temporal_metric_sums.get("temporal_metric_voxels", 0.0))),
                                       "mean_displacement_voxels": temporal_metric_sums.get("mean_displacement_voxels", 0.0) / max(1.0, float(len(temporal_losses))),
                                       "mean_displacement_mm": temporal_metric_sums.get("mean_displacement_mm", 0.0) / max(1.0, float(len(temporal_losses))),
                                       "occlusion_fraction": temporal_metric_sums.get("occlusion_fraction", 0.0) / max(1.0, float(len(temporal_losses)))})
            else:
                self.skipped_pseudo_updates += 1
            if previous_training:
                self.model.train()

        # Exactly one EMA update, after all committed student changes.
        _ema_update(self.teacher, self.model, float(self.config["training"]["ema_decay"]))
        self.ema_updates += 1
        self.outer_step += 1
        record = {"step": self.outer_step, "supervised_loss": float(sup_loss.detach()),
                  "supervised_grad_norm": grad_norm, "supervised_update_norm": update_norm,
                  **metrics, **pseudo_metrics,
                  **extra_metrics,
                  "supervised_updates": self.supervised_updates, "pseudo_updates": self.pseudo_updates,
                  "ema_updates": self.ema_updates, "optimizer_updates": self.optimizer_updates}
        self.history.append(record)
        return record

    step = transaction

    def state_dict(self) -> dict[str, Any]:
        return {
            "outer_step": self.outer_step,
            "supervised_updates": self.supervised_updates,
            "supervised_extra_updates": self.supervised_extra_updates,
            "pseudo_updates": self.pseudo_updates,
            "optimizer_updates": self.optimizer_updates,
            "ema_updates": self.ema_updates,
            "skipped_supervised_updates": self.skipped_supervised_updates,
            "skipped_pseudo_updates": self.skipped_pseudo_updates,
            "history": copy.deepcopy(self.history),
            "optimizer_metadata": copy.deepcopy(self.optimizer_metadata),
            "sampler_state": self.sampler_state,
            "region_state": self.region_manager.state_dict() if self.region_manager is not None and hasattr(self.region_manager, "state_dict") else None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {"outer_step", "supervised_updates", "supervised_extra_updates", "pseudo_updates",
                    "optimizer_updates", "ema_updates", "skipped_supervised_updates", "skipped_pseudo_updates",
                    "history", "optimizer_metadata", "sampler_state", "region_state"}
        if not required.issubset(state):
            raise ValueError(f"trainer state missing required fields: {sorted(required - set(state))}")
        for key in ("outer_step", "supervised_updates", "supervised_extra_updates", "pseudo_updates", "optimizer_updates", "ema_updates", "skipped_supervised_updates", "skipped_pseudo_updates"):
            setattr(self, key, int(state[key]))
        self.history = copy.deepcopy(list(state["history"]))
        self.sampler_state = state["sampler_state"]
        if self.region_manager is not None and state["region_state"] is not None and hasattr(self.region_manager, "load_state_dict"):
            self.region_manager.load_state_dict(state["region_state"])


def train_volume(config: Mapping[str, Any], *, out: str | None = None, device: str = "cpu",
                 synthetic: bool = False, steps: int | None = None,
                 manifest: str | None = None, manifest_sha256: str | None = None,
                 geometry_path: str | None = None, geometry_sha256: str | None = None,
                 resume: str | None = None, resume_sha256: str | None = None,
                 allow_full_run: bool = False) -> dict[str, Any]:
    """Run a bounded opt-in volume smoke or a caller-supplied data loop.

    Both fixture and patient execution are delegated to ``VolumeTrainingData``
    with hash-pinned manifest and geometry identities.  There is no random
    B1/no-U shortcut: missing assets fail closed without downloads.
    """
    from .config import validate_volume_config
    if type(synthetic) is not bool:
        raise ValueError("volume training synthetic must be an exact bool")
    # ``training.steps`` is the planned scheduler horizon.  A CLI ``steps``
    # override is only a bounded stop target; it must not rebuild a shorter
    # cosine schedule and then resume under a different objective.
    requested = validate_volume_config(config, allow_full_run=allow_full_run, stop_steps=steps)
    c = copy.deepcopy(requested)
    if steps is not None:
        if isinstance(steps, bool) or steps < 1:
            raise ValueError("steps must be positive")
        target_steps = int(steps)
    else:
        target_steps = int(c["training"]["steps"])
    if target_steps > 8 and not allow_full_run:
        raise ValueError("effective requested steps > 8 requires explicit allow_full_run")
    if target_steps > int(c["training"]["steps"]):
        raise ValueError("stop target exceeds the configured scheduler horizon")
    # Validate the effective stop target before model/data/output construction,
    # while preserving the configured horizon in ``c`` for scheduler state.
    validate_volume_config(c, allow_full_run=allow_full_run, stop_steps=target_steps)
    correction = c.get("correction")
    if isinstance(correction, Mapping) and correction.get("protocol", "none") != "none":
        raise ValueError("correction protocols require the dedicated correction workflow CLI; train_volume will not ignore prompts")
    has_pinned_data = all(isinstance(v, str) and v for v in (manifest, geometry_path))
    if has_pinned_data:
        manifest_sha256 = _require_data_pin(manifest_sha256, "manifest_sha256")
        geometry_sha256 = _require_data_pin(geometry_sha256, "geometry_sha256")
    if not has_pinned_data:
        raise ValueError("volume training requires manifest/manifest_sha256/geometry_path/geometry_sha256; fixture and real paths share the same loader")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent device fallback")
    if c["training"]["precision"] == "amp" and target_device.type != "cuda":
        raise RuntimeError("AMP requires CUDA; CPU execution must declare fp32")
    if resume is not None:
        if not resume_sha256:
            raise ValueError("resume requires resume_sha256")
        from .checkpoint import _validate_manifest, load_volume_checkpoint
        # Preflight checkpoint identity before load_volume_checkpoint restores
        # any global RNG or constructs a trainer.  A mismatched data/config
        # identity therefore fails without mutating the accepted live state.
        preflight_manifest = _validate_manifest(Path(resume), resume_sha256)
        preflight_data = preflight_manifest.get("data_provenance")
        if not isinstance(preflight_data, Mapping):
            raise ValueError("resume checkpoint data provenance is required")
        if type(preflight_data.get("synthetic")) is not bool:
            raise ValueError("resume synthetic/data identity must be an exact bool")
        if preflight_data["synthetic"] is not synthetic:
            raise ValueError("resume synthetic/data identity differs from checkpoint")
        for key, supplied in (("manifest_sha256", manifest_sha256), ("geometry_sha256", geometry_sha256)):
            expected = _require_data_pin(preflight_data.get(key), f"resume {key}")
            if expected != _require_data_pin(supplied, key):
                raise ValueError(f"resume {key} differs from checkpoint data identity")
        resume_config = validate_volume_config(preflight_manifest["config"], allow_full_run=True)
        supplied_compare = copy.deepcopy(requested)
        resume_compare = copy.deepcopy(resume_config)
        # A continuation may extend the target step, but every other recipe
        # field must remain byte-identical to the checkpoint manifest.
        if int(requested["training"]["steps"]) != int(resume_compare["training"]["steps"]):
            raise ValueError("resume must retain the checkpoint scheduler horizon")
        supplied_compare["training"]["steps"] = resume_compare["training"]["steps"]
        if config_sha256(supplied_compare) != preflight_manifest.get("config_sha256"):
            raise ValueError("resume config SHA-256 differs from checkpoint recipe")
        trainer, resume_manifest = load_volume_checkpoint(resume, resume_sha256, device=target_device)
        c = resume_config
        correction = c.get("correction")
        if isinstance(correction, Mapping) and correction.get("protocol", "none") != "none":
            raise ValueError("correction protocols require the dedicated correction workflow CLI; train_volume will not ignore prompts")
        # The preflight and checkpoint loader both require this exact bool;
        # preserve it without coercing malformed provenance.
        synthetic = resume_manifest["data_provenance"]["synthetic"]
        if steps is None:
            target_steps = int(c["training"]["steps"])
        else:
            target_steps = int(steps)
        if target_steps <= trainer.outer_step:
            raise ValueError("resume target steps must exceed checkpoint outer_step")
        if target_steps > 8 and not allow_full_run:
            raise ValueError("resume target steps > 8 requires explicit allow_full_run")
        if target_steps > int(c["training"]["steps"]):
            raise ValueError("resume target exceeds the checkpoint scheduler horizon")
        model = trainer.model
        provenance = {"resume_manifest_sha256": resume_sha256, "resumed_from_step": trainer.outer_step}
    else:
        torch.manual_seed(int(c["training"]["seed"]))
        np.random.seed(int(c["training"]["seed"]))
        random.seed(int(c["training"]["seed"]))
        torch.set_num_threads(int(c["training"]["threads"]))
        from .models import build_volume_model
        model, provenance = build_volume_model(c, synthetic=synthetic)
        model.to(target_device)
        trainer = VolumeTrainer(model, c)
    z = int(c["geometry"]["slab_depth"])
    crop_size = c["geometry"]["crop_size"]
    if isinstance(crop_size, (list, tuple)):
        h, w = (int(crop_size[0]), int(crop_size[1]))
    else:
        h = w = int(crop_size)
    provenance_data: dict[str, Any] = {"synthetic": synthetic}
    data = None
    rng = None
    if resume is not None:
        saved_data = resume_manifest.get("data_provenance")
        if not isinstance(saved_data, Mapping) or type(saved_data.get("synthetic")) is not bool:
            raise ValueError("resume synthetic/data identity must be an exact bool")
        if saved_data["synthetic"] is not synthetic:
            raise ValueError("resume synthetic/data identity differs from checkpoint")
        for key, supplied in (("manifest_sha256", manifest_sha256), ("geometry_sha256", geometry_sha256)):
            expected = _require_data_pin(saved_data.get(key), f"resume {key}")
            if expected != _require_data_pin(supplied, key):
                raise ValueError(f"resume {key} differs from checkpoint data identity")
    from .data import VolumeTrainingData
    data = VolumeTrainingData(manifest, manifest_sha256, geometry_path, geometry_sha256, synthetic=synthetic)
    rng = np.random.default_rng(int(c["training"]["seed"]))
    if trainer.sampler_state and isinstance(trainer.sampler_state, Mapping) and trainer.sampler_state.get("numpy_rng") is not None:
        rng.bit_generator.state = trainer.sampler_state["numpy_rng"]
    provenance_data.update({"manifest_sha256": manifest_sha256, "geometry_sha256": geometry_sha256,
                            "labeled_patient_count": len(data.labeled), "unlabeled_patient_count": len(data.unlabeled)})
    start_step = trainer.outer_step
    for i in range(start_step, target_steps):
        assert data is not None and rng is not None
        crop = c["geometry"]["crop_size"]
        logical_batch = int(c["training"]["microbatch"]) * int(c["training"]["accumulation"])
        lb = data.sample_batch(i % len(data.labeled), rng, logical_batch,
                               labeled=True, slab_depth=int(c["geometry"]["slab_depth"]), crop_size=crop,
                               spacing_xyz_mm=c["geometry"]["spacing_xyz_mm"], fov_xyz_mm=c["geometry"]["fov_xyz_mm"],
                               flip=bool(c["geometry"]["flip"]))
        lb_extra = None
        if c["training"]["method"] == "supervised_extra":
            lb_extra = data.sample_batch((i + 1) % len(data.labeled), rng, logical_batch,
                                         labeled=True, slab_depth=int(c["geometry"]["slab_depth"]), crop_size=crop,
                                         spacing_xyz_mm=c["geometry"]["spacing_xyz_mm"], fov_xyz_mm=c["geometry"]["fov_xyz_mm"],
                                         flip=bool(c["geometry"]["flip"]))
        if c["temporal"].get("mode") != "none":
            pairs = [data.sample_temporal_pair(
                i % len(data.unlabeled), rng, labeled=False,
                slab_depth=int(c["geometry"]["slab_depth"]), crop_size=crop,
                spacing_xyz_mm=c["geometry"]["spacing_xyz_mm"], fov_xyz_mm=c["geometry"]["fov_xyz_mm"],
                flip=bool(c["geometry"]["flip"])) for _ in range(logical_batch)]
            ub = {
                "image": torch.stack([pair["image"] for pair in pairs]),
                "adjacent_image": torch.stack([pair["adjacent_image"] for pair in pairs]),
                "geometric_valid": torch.stack([pair["geometric_valid"] for pair in pairs]),
                "adjacent_geometric_valid": torch.stack([pair["adjacent_geometric_valid"] for pair in pairs]),
                "metadata": [pair["metadata"] for pair in pairs],
                "adjacent_metadata": [pair["adjacent_metadata"] for pair in pairs],
            }
        else:
            ub = data.sample_batch(i % len(data.unlabeled), rng, logical_batch,
                                   labeled=False, slab_depth=int(c["geometry"]["slab_depth"]), crop_size=crop,
                                   spacing_xyz_mm=c["geometry"]["spacing_xyz_mm"], fov_xyz_mm=c["geometry"]["fov_xyz_mm"],
                                   flip=bool(c["geometry"]["flip"]))
        def move(batch: Mapping[str, Any]) -> dict[str, Any]:
            return {key: (value.to(target_device) if isinstance(value, Tensor) else value) for key, value in batch.items()}
        trainer.transaction(move(lb), move(ub), iteration=i,
                            extra_labeled_batch=(move(lb_extra) if lb_extra is not None else None))
        trainer.sampler_state = {"numpy_rng": rng.bit_generator.state, "iteration": i + 1}
        if out is not None and (trainer.outer_step % int(c["training"]["save_every"]) == 0) and trainer.outer_step < target_steps:
            from .checkpoint import save_volume_checkpoint
            intermediate = Path(out).with_name(f"{Path(out).name}.step{trainer.outer_step:06d}")
            save_volume_checkpoint(intermediate, trainer, c, data_provenance=provenance_data, step=trainer.outer_step)
    receipt: dict[str, Any] = {"schema": "self_audit.volume.run.v1", "config": c,
                               "provenance": provenance, "steps": trainer.outer_step,
                               "schedule_steps": int(c["training"]["steps"]),
                               "supervised_updates": trainer.supervised_updates,
                               "supervised_extra_updates": trainer.supervised_extra_updates,
                               "pseudo_updates": trainer.pseudo_updates,
                               "optimizer_updates": trainer.optimizer_updates,
                               "ema_updates": trainer.ema_updates,
                               "quality": "SOFTWARE_SYNTHETIC_ONLY" if synthetic else "NOT_EVALUATED",
                               "data": provenance_data}
    if out is not None:
        from .checkpoint import save_volume_checkpoint
        save_data = provenance_data.copy()
        if not synthetic and data is not None:
            save_data["label_reads"] = sorted(data.label_reads)
        receipt["checkpoint_manifest_sha256"] = save_volume_checkpoint(out, trainer, c, data_provenance=save_data)
    return receipt


__all__ = ["VolumeTrainer", "_ema_update", "train_volume"]
