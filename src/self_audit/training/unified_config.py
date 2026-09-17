"""Strict version 1 unified configuration schema and loader for Self-Audit.

Recursively validates unknown keys, duplicate YAML keys, scalar types,
non-finite values, and schedule coverage across all 8 required sections:
experiment, dataset, model, training, checkpoint, calibration, logging, diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import io
import math
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence
import torch
import yaml

from ._utils import (
    CANDIDATE_C_DEFAULTS,
    CANDIDATE_C_SOLVER_MODES,
    DEFAULT_WINDOW_MODE,
    WINDOW_MODES,
    validate_candidate_c_settings,
    validate_window_mode,
)
from .schedule import Schedule, ScheduleInterval


def _ensure_bool(val: Any, name: str) -> bool:
    if not isinstance(val, bool):
        raise TypeError(f"Field '{name}' must be a boolean, got {type(val).__name__} ({val!r})")
    return val


def _ensure_int(val: Any, name: str, *, min_val: int | None = None, max_val: int | None = None) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise TypeError(f"Field '{name}' must be an integer, got {type(val).__name__} ({val!r})")
    if min_val is not None and val < min_val:
        raise ValueError(f"Field '{name}' must be >= {min_val}, got {val}")
    if max_val is not None and val > max_val:
        raise ValueError(f"Field '{name}' must be <= {max_val}, got {val}")
    return val


def _ensure_float(
    val: Any,
    name: str,
    *,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"Field '{name}' must be a float, got {type(val).__name__} ({val!r})")
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"Field '{name}' must be a finite number, got {fval}")
    if min_val is not None and fval < min_val:
        raise ValueError(f"Field '{name}' must be >= {min_val}, got {fval}")
    if max_val is not None and fval > max_val:
        raise ValueError(f"Field '{name}' must be <= {max_val}, got {fval}")
    return fval


def _ensure_str(val: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(val, str):
        raise TypeError(f"Field '{name}' must be a string, got {type(val).__name__} ({val!r})")
    if not allow_empty and not val.strip():
        raise ValueError(f"Field '{name}' must not be empty")
    return val


def _ensure_dict(val: Any, name: str) -> dict[str, Any]:
    if not isinstance(val, Mapping):
        raise TypeError(f"Field '{name}' must be a mapping/dict, got {type(val).__name__}")
    return dict(val)


def _check_keys(
    data: Mapping[str, Any],
    allowed: set[str],
    section_name: str,
    *,
    required: set[str] | None = None,
) -> None:
    data_keys = set(data.keys())
    unknown = data_keys - allowed
    if unknown:
        raise ValueError(f"Unknown keys in section '{section_name}': {sorted(unknown)}")
    req = allowed if required is None else required
    missing = req - data_keys
    if missing:
        raise ValueError(f"Missing required keys in section '{section_name}': {sorted(missing)}")


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML SafeLoader that strictly rejects duplicate mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate key '{key}' found in YAML configuration at line {key_node.start_mark.line + 1}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    deterministic: bool
    device: str
    protocol: str = "unified_schedule_v1"
    recipe: str = "curriculum_v1"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        allowed = {"name", "seed", "deterministic", "device", "protocol", "recipe"}
        required = {"name", "seed", "deterministic", "device"}
        _check_keys(data, allowed, "experiment", required=required)
        protocol = _ensure_str(data.get("protocol", "unified_schedule_v1"), "experiment.protocol")
        recipe = _ensure_str(data.get("recipe", "curriculum_v1"), "experiment.recipe")
        return cls(
            name=_ensure_str(data["name"], "experiment.name"),
            seed=_ensure_int(data["seed"], "experiment.seed", min_val=0),
            deterministic=_ensure_bool(data["deterministic"], "experiment.deterministic"),
            device=_ensure_str(data["device"], "experiment.device"),
            protocol=protocol,
            recipe=recipe,
        )


@dataclass(frozen=True)
class PreprocessingConfig:
    clipping_min: float
    clipping_max: float
    foreground_only: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreprocessingConfig:
        _check_keys(data, {"clipping_min", "clipping_max", "foreground_only"}, "dataset.preprocessing")
        c_min = _ensure_float(data["clipping_min"], "dataset.preprocessing.clipping_min")
        c_max = _ensure_float(data["clipping_max"], "dataset.preprocessing.clipping_max")
        if c_min >= c_max:
            raise ValueError(f"clipping_min ({c_min}) must be strictly less than clipping_max ({c_max})")
        return cls(
            clipping_min=c_min,
            clipping_max=c_max,
            foreground_only=_ensure_bool(data["foreground_only"], "dataset.preprocessing.foreground_only"),
        )


@dataclass(frozen=True)
class DataLoaderConfig:
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataLoaderConfig:
        _check_keys(data, {"num_workers", "pin_memory", "persistent_workers", "prefetch_factor"}, "dataset.dataloader")
        return cls(
            num_workers=_ensure_int(data["num_workers"], "dataset.dataloader.num_workers", min_val=0),
            pin_memory=_ensure_bool(data["pin_memory"], "dataset.dataloader.pin_memory"),
            persistent_workers=_ensure_bool(data["persistent_workers"], "dataset.dataloader.persistent_workers"),
            prefetch_factor=_ensure_int(data["prefetch_factor"], "dataset.dataloader.prefetch_factor", min_val=1),
        )


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_root: str
    split_manifest: str | None
    train_split: str
    val_split: str
    test_split: str | None
    image_size: int
    depth_axis: int
    num_classes: int
    class_mapping: dict[int, int]
    preprocessing: PreprocessingConfig
    dataloader: DataLoaderConfig
    representation: str = "paired_3d"

    @property
    def has_test(self) -> bool:
        return self.test_split is not None

    @property
    def optional_test(self) -> bool:
        return True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetConfig:
        allowed = {
            "name",
            "data_root",
            "split_manifest",
            "train_split",
            "val_split",
            "test_split",
            "image_size",
            "depth_axis",
            "num_classes",
            "class_mapping",
            "preprocessing",
            "dataloader",
            "representation",
        }
        required = {
            "name",
            "data_root",
            "train_split",
            "val_split",
            "image_size",
            "depth_axis",
            "class_mapping",
            "preprocessing",
            "dataloader",
        }
        _check_keys(data, allowed, "dataset", required=required)

        test_split = data.get("test_split", None)
        if test_split is not None:
            test_split = _ensure_str(test_split, "dataset.test_split")

        manifest_val = data.get("split_manifest", None)
        split_manifest = _ensure_str(manifest_val, "dataset.split_manifest") if manifest_val is not None else None

        representation = _ensure_str(data.get("representation", "paired_3d"), "dataset.representation")
        if representation != "paired_3d":
            raise ValueError(f"dataset.representation must be 'paired_3d', got {representation!r}")

        class_map = data["class_mapping"]
        if not isinstance(class_map, Mapping):
            raise TypeError(f"dataset.class_mapping must be a mapping, got {type(class_map).__name__}")
        validated_map: dict[int, int] = {}
        for k, v in class_map.items():
            k_int = _ensure_int(k, "dataset.class_mapping key", min_val=0)
            v_int = _ensure_int(v, "dataset.class_mapping value", min_val=0)
            validated_map[k_int] = v_int

        derived_num_classes = len(set(validated_map.values())) if validated_map else 4
        if "num_classes" in data:
            explicit_nc = _ensure_int(data["num_classes"], "dataset.num_classes", min_val=1)
            if explicit_nc != derived_num_classes and explicit_nc != 4:
                raise ValueError(
                    f"dataset.num_classes ({explicit_nc}) contradicts four-class contract / class mapping ({derived_num_classes})"
                )
            final_num_classes = explicit_nc
        else:
            final_num_classes = derived_num_classes

        return cls(
            name=_ensure_str(data["name"], "dataset.name"),
            data_root=_ensure_str(data["data_root"], "dataset.data_root"),
            split_manifest=split_manifest,
            train_split=_ensure_str(data["train_split"], "dataset.train_split"),
            val_split=_ensure_str(data["val_split"], "dataset.val_split"),
            test_split=test_split,
            image_size=_ensure_int(data["image_size"], "dataset.image_size", min_val=1),
            depth_axis=_ensure_int(data["depth_axis"], "dataset.depth_axis", min_val=0, max_val=2),
            num_classes=final_num_classes,
            class_mapping=validated_map,
            preprocessing=PreprocessingConfig.from_dict(_ensure_dict(data["preprocessing"], "dataset.preprocessing")),
            dataloader=DataLoaderConfig.from_dict(_ensure_dict(data["dataloader"], "dataset.dataloader")),
            representation=representation,
        )


@dataclass(frozen=True)
class CandidateCSettings:
    """Validated Candidate C solver settings carried by the model section.

    This is the schema-side record only; the model package owns the runtime
    ``CandidateCConfig`` and converts the validated mapping produced here.
    Keeping the conversion on the model side avoids an import cycle between
    the configuration schema and the network.
    """

    rho_feature_pixels: float = float(CANDIDATE_C_DEFAULTS["rho_feature_pixels"])
    lam: float = float(CANDIDATE_C_DEFAULTS["lam"])
    lr: float | None = CANDIDATE_C_DEFAULTS["lr"]
    fix_threshold: float = float(CANDIDATE_C_DEFAULTS["fix_threshold"])
    regress_threshold: float = float(CANDIDATE_C_DEFAULTS["regress_threshold"])
    margin_fraction: float = float(CANDIDATE_C_DEFAULTS["margin_fraction"])
    min_regress_mass: float = float(CANDIDATE_C_DEFAULTS["min_regress_mass"])
    replay_atol: float = float(CANDIDATE_C_DEFAULTS["replay_atol"])
    replay_rtol: float = float(CANDIDATE_C_DEFAULTS["replay_rtol"])
    max_backtracks: int = int(CANDIDATE_C_DEFAULTS["max_backtracks"])

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> CandidateCSettings:
        return cls(**validate_candidate_c_settings(data))

    def to_mapping(self) -> dict[str, Any]:
        """Return the complete validated mapping handed to the model package."""
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    encoder_name: str
    pretrained_encoder: bool
    fallback: bool
    shared_channels: int
    num_classes: int
    window_k: int
    max_turns: int
    # Execution mode.  ``current`` is the historical network; every other value
    # is an explicit opt-in and is never inferred from any other setting.
    window_mode: str = DEFAULT_WINDOW_MODE
    candidate_c: CandidateCSettings = field(default_factory=CandidateCSettings)

    @property
    def encoder_allow_fallback(self) -> bool:
        return self.fallback

    @property
    def runs_candidate_c_solver(self) -> bool:
        return self.window_mode in CANDIDATE_C_SOLVER_MODES

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelConfig:
        allowed = {
            "encoder_name",
            "pretrained_encoder",
            "fallback",
            "shared_channels",
            "num_classes",
            "window_k",
            "max_turns",
            "window_mode",
            "candidate_c",
        }
        required = allowed - {"window_mode", "candidate_c"}
        _check_keys(data, allowed, "model", required=required)
        return cls(
            encoder_name=_ensure_str(data["encoder_name"], "model.encoder_name"),
            pretrained_encoder=_ensure_bool(data["pretrained_encoder"], "model.pretrained_encoder"),
            fallback=_ensure_bool(data["fallback"], "model.fallback"),
            shared_channels=_ensure_int(data["shared_channels"], "model.shared_channels", min_val=1),
            num_classes=_ensure_int(data["num_classes"], "model.num_classes", min_val=1),
            window_k=_ensure_int(data["window_k"], "model.window_k", min_val=1),
            max_turns=_ensure_int(data["max_turns"], "model.max_turns", min_val=1),
            window_mode=validate_window_mode(data.get("window_mode", DEFAULT_WINDOW_MODE)),
            candidate_c=CandidateCSettings.from_dict(data.get("candidate_c")),
        )


@dataclass(frozen=True)
class AmpConfig:
    enabled: bool
    dtype: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AmpConfig:
        _check_keys(data, {"enabled", "dtype"}, "training.amp")
        dtype = _ensure_str(data["dtype"], "training.amp.dtype")
        if dtype not in {"bfloat16", "float16"}:
            raise ValueError(f"training.amp.dtype must be 'bfloat16' or 'float16', got {dtype!r}")
        return cls(
            enabled=_ensure_bool(data["enabled"], "training.amp.enabled"),
            dtype=dtype,
        )


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    grad_clip: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizerConfig:
        _check_keys(data, {"name", "betas", "eps", "weight_decay", "grad_clip"}, "training.optimizer")
        name = _ensure_str(data["name"], "training.optimizer.name")
        if name != "adamw":
            raise ValueError(f"training.optimizer.name must be 'adamw', got {name!r}")
        betas_val = data["betas"]
        if not isinstance(betas_val, (list, tuple)) or len(betas_val) != 2:
            raise TypeError("training.optimizer.betas must be a sequence of 2 floats")
        beta1 = _ensure_float(betas_val[0], "training.optimizer.betas[0]", min_val=0.0, max_val=1.0)
        beta2 = _ensure_float(betas_val[1], "training.optimizer.betas[1]", min_val=0.0, max_val=1.0)
        return cls(
            name=name,
            betas=(beta1, beta2),
            eps=_ensure_float(data["eps"], "training.optimizer.eps", min_val=0.0),
            weight_decay=_ensure_float(data["weight_decay"], "training.optimizer.weight_decay", min_val=0.0),
            grad_clip=_ensure_float(data["grad_clip"], "training.optimizer.grad_clip", min_val=0.0),
        )


@dataclass(frozen=True)
class LRCurveConfig:
    name: str
    warmup_epochs: int
    min_ratio: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LRCurveConfig:
        _check_keys(data, {"name", "warmup_epochs", "min_ratio"}, "training.lr_curve")
        name = _ensure_str(data["name"], "training.lr_curve.name")
        if name != "warmup_cosine":
            raise ValueError(f"training.lr_curve.name must be 'warmup_cosine', got {name!r}")
        return cls(
            name=name,
            warmup_epochs=_ensure_int(data["warmup_epochs"], "training.lr_curve.warmup_epochs", min_val=0),
            min_ratio=_ensure_float(data["min_ratio"], "training.lr_curve.min_ratio", min_val=0.0, max_val=1.0),
        )


@dataclass(frozen=True)
class AnnotationLossConfig:
    stage_weights: tuple[float, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnnotationLossConfig:
        _check_keys(data, {"stage_weights"}, "training.annotation_loss")
        weights = data["stage_weights"]
        if not isinstance(weights, (list, tuple)) or not weights:
            raise TypeError("training.annotation_loss.stage_weights must be a non-empty sequence of floats")
        parsed = tuple(_ensure_float(w, f"training.annotation_loss.stage_weights[{i}]", min_val=0.0) for i, w in enumerate(weights))
        return cls(stage_weights=parsed)


@dataclass(frozen=True)
class AuditLossConfig:
    audit_margin: float
    neutral_margin: float
    local_class_weighting: str
    target_contract: str = "audit_target_legacy_one_v1"

    FIXED_AUDIT_MARGIN: ClassVar[float] = 0.05
    FIXED_NEUTRAL_MARGIN: ClassVar[float] = 0.005

    @property
    def audit_target_contract(self) -> str:
        return self.target_contract

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AuditLossConfig:
        allowed = {
            "audit_margin",
            "neutral_margin",
            "local_class_weighting",
            "target_contract",
            "audit_target_contract",
        }
        required = {"audit_margin", "neutral_margin", "local_class_weighting"}
        _check_keys(data, allowed, "training.audit_loss", required=required)
        weighting = _ensure_str(data["local_class_weighting"], "training.audit_loss.local_class_weighting")
        if weighting not in {"balanced_clamped", "none"}:
            raise ValueError(f"training.audit_loss.local_class_weighting must be 'balanced_clamped' or 'none', got {weighting!r}")

        margin = _ensure_float(data["audit_margin"], "training.audit_loss.audit_margin", min_val=0.0)
        if abs(margin - 0.05) > 1e-7:
            raise ValueError(
                f"training.audit_loss.audit_margin must be 0.05 (joint helper uses fixed margin 0.05; arbitrary margins are unsupported and rejected), got {margin}"
            )

        n_margin = _ensure_float(data["neutral_margin"], "training.audit_loss.neutral_margin", min_val=0.0)
        if abs(n_margin - 0.005) > 1e-7:
            raise ValueError(
                f"training.audit_loss.neutral_margin must be 0.005 (fixed approved neutral margin; arbitrary margins are unsupported and rejected), got {n_margin}"
            )

        target_contract = "audit_target_legacy_one_v1"
        if "target_contract" in data:
            target_contract = _ensure_str(data["target_contract"], "training.audit_loss.target_contract")
        elif "audit_target_contract" in data:
            target_contract = _ensure_str(data["audit_target_contract"], "training.audit_loss.audit_target_contract")

        if target_contract != "audit_target_legacy_one_v1":
            raise ValueError(
                f"training audit target contract must be 'audit_target_legacy_one_v1', got {target_contract!r}"
            )

        return cls(
            audit_margin=margin,
            neutral_margin=n_margin,
            local_class_weighting=weighting,
            target_contract=target_contract,
        )


@dataclass(frozen=True)
class CounterfactualConfig:
    mixture: tuple[float, float, float]
    repair_strength_min: float
    repair_strength_max: float
    epsilon: float
    retries: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CounterfactualConfig:
        _check_keys(data, {"mixture", "repair_strength_min", "repair_strength_max", "epsilon", "retries"}, "training.counterfactual")
        mixture = data["mixture"]
        if not isinstance(mixture, (list, tuple)) or len(mixture) != 3:
            raise TypeError("training.counterfactual.mixture must be a sequence of 3 floats")
        m0 = _ensure_float(mixture[0], "training.counterfactual.mixture[0]", min_val=0.0, max_val=1.0)
        m1 = _ensure_float(mixture[1], "training.counterfactual.mixture[1]", min_val=0.0, max_val=1.0)
        m2 = _ensure_float(mixture[2], "training.counterfactual.mixture[2]", min_val=0.0, max_val=1.0)
        if abs(m0 + m1 + m2 - 1.0) > 1e-5:
            raise ValueError(f"training.counterfactual.mixture elements must sum to 1.0, got {m0 + m1 + m2}")

        r_min = _ensure_float(data["repair_strength_min"], "training.counterfactual.repair_strength_min", min_val=0.0, max_val=1.0)
        r_max = _ensure_float(data["repair_strength_max"], "training.counterfactual.repair_strength_max", min_val=0.0, max_val=1.0)
        if r_min > r_max:
            raise ValueError(f"repair_strength_min ({r_min}) must be <= repair_strength_max ({r_max})")

        return cls(
            mixture=(m0, m1, m2),
            repair_strength_min=r_min,
            repair_strength_max=r_max,
            epsilon=_ensure_float(data["epsilon"], "training.counterfactual.epsilon", min_val=0.0),
            retries=_ensure_int(data["retries"], "training.counterfactual.retries", min_val=1),
        )


@dataclass(frozen=True)
class RolloutConfig:
    tau: float
    max_turns: int
    # Opt-in bootstrap curriculum.  Disabled by default so the primary
    # zero-history annotation objective and the 130-epoch recipe are unchanged.
    predicted_history_exposure: bool = False
    predicted_history_weight: float = 0.1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RolloutConfig:
        allowed = {"tau", "max_turns", "predicted_history_exposure", "predicted_history_weight"}
        _check_keys(data, allowed, "training.rollout", required={"tau", "max_turns"})
        return cls(
            tau=_ensure_float(data["tau"], "training.rollout.tau"),
            max_turns=_ensure_int(data["max_turns"], "training.rollout.max_turns", min_val=1),
            predicted_history_exposure=_ensure_bool(
                data.get("predicted_history_exposure", False),
                "training.rollout.predicted_history_exposure",
            ),
            predicted_history_weight=_ensure_float(
                data.get("predicted_history_weight", 0.1),
                "training.rollout.predicted_history_weight",
                min_val=0.0,
            ),
        )


@dataclass(frozen=True)
class TrainingConfig:
    amp: AmpConfig
    optimizer: OptimizerConfig
    lr_curve: LRCurveConfig
    annotation_loss: AnnotationLossConfig
    audit_loss: AuditLossConfig
    counterfactual: CounterfactualConfig
    rollout: RolloutConfig
    schedule: Schedule

    @property
    def audit_target_contract(self) -> str:
        return self.audit_loss.target_contract

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingConfig:
        _check_keys(
            data,
            {
                "amp",
                "optimizer",
                "lr_curve",
                "annotation_loss",
                "audit_loss",
                "counterfactual",
                "rollout",
                "schedule",
            },
            "training",
        )
        return cls(
            amp=AmpConfig.from_dict(_ensure_dict(data["amp"], "training.amp")),
            optimizer=OptimizerConfig.from_dict(_ensure_dict(data["optimizer"], "training.optimizer")),
            lr_curve=LRCurveConfig.from_dict(_ensure_dict(data["lr_curve"], "training.lr_curve")),
            annotation_loss=AnnotationLossConfig.from_dict(_ensure_dict(data["annotation_loss"], "training.annotation_loss")),
            audit_loss=AuditLossConfig.from_dict(_ensure_dict(data["audit_loss"], "training.audit_loss")),
            counterfactual=CounterfactualConfig.from_dict(_ensure_dict(data["counterfactual"], "training.counterfactual")),
            rollout=RolloutConfig.from_dict(_ensure_dict(data["rollout"], "training.rollout")),
            schedule=Schedule.from_dict(_ensure_dict(data["schedule"], "training.schedule")),
        )


@dataclass(frozen=True)
class CheckpointConfig:
    output_dir: str
    save_best: bool
    save_last: bool
    best_selection_min_epoch: int
    best_metric: str
    best_metric_mode: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckpointConfig:
        _check_keys(
            data,
            {
                "output_dir",
                "save_best",
                "save_last",
                "best_selection_min_epoch",
                "best_metric",
                "best_metric_mode",
            },
            "checkpoint",
        )
        min_epoch = _ensure_int(data["best_selection_min_epoch"], "checkpoint.best_selection_min_epoch", min_val=0)

        metric = _ensure_str(data["best_metric"], "checkpoint.best_metric")
        if metric != "final_foreground_macro_dice":
            raise ValueError(
                f"checkpoint.best_metric must be 'final_foreground_macro_dice' (implemented final metric), got {metric!r}"
            )

        mode = _ensure_str(data["best_metric_mode"], "checkpoint.best_metric_mode")
        if mode != "max":
            raise ValueError(
                f"checkpoint.best_metric_mode must be 'max' (final metric must be maximized), got {mode!r}"
            )

        return cls(
            output_dir=_ensure_str(data["output_dir"], "checkpoint.output_dir"),
            save_best=_ensure_bool(data["save_best"], "checkpoint.save_best"),
            save_last=_ensure_bool(data["save_last"], "checkpoint.save_last"),
            best_selection_min_epoch=min_epoch,
            best_metric=metric,
            best_metric_mode=mode,
        )


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool
    split: str
    threshold_min: float
    threshold_max: float
    threshold_steps: int
    metric_contract: str
    metric_space: str = "slice_proxy"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationConfig:
        allowed = {
            "enabled",
            "split",
            "threshold_min",
            "threshold_max",
            "threshold_steps",
            "metric_contract",
            "metric_space",
        }
        required = {
            "enabled",
            "split",
            "threshold_min",
            "threshold_max",
            "threshold_steps",
            "metric_contract",
        }
        _check_keys(data, allowed, "calibration", required=required)
        t_min = _ensure_float(data["threshold_min"], "calibration.threshold_min")
        t_max = _ensure_float(data["threshold_max"], "calibration.threshold_max")
        if t_min >= t_max:
            raise ValueError(f"calibration.threshold_min ({t_min}) must be < threshold_max ({t_max})")

        contract = _ensure_str(data["metric_contract"], "calibration.metric_contract")
        if contract != "foreground_dice_exclude_v1":
            raise ValueError(f"calibration.metric_contract must be 'foreground_dice_exclude_v1', got {contract!r}")

        metric_space = _ensure_str(data.get("metric_space", "slice_proxy"), "calibration.metric_space")
        if metric_space != "slice_proxy":
            raise ValueError(f"calibration.metric_space must be 'slice_proxy', got {metric_space!r}")

        return cls(
            enabled=_ensure_bool(data["enabled"], "calibration.enabled"),
            split=_ensure_str(data["split"], "calibration.split"),
            threshold_min=t_min,
            threshold_max=t_max,
            threshold_steps=_ensure_int(data["threshold_steps"], "calibration.threshold_steps", min_val=2),
            metric_contract=contract,
            metric_space=metric_space,
        )


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    entity: str | None
    run_name: str
    mode: str
    tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WandbConfig:
        _check_keys(data, {"enabled", "project", "entity", "run_name", "mode", "tags"}, "logging.wandb")
        entity = data["entity"]
        if entity is not None:
            entity = _ensure_str(entity, "logging.wandb.entity")
        tags = data["tags"]
        if not isinstance(tags, (list, tuple)):
            raise TypeError("logging.wandb.tags must be a sequence of strings")
        parsed_tags = tuple(_ensure_str(t, f"logging.wandb.tags[{i}]") for i, t in enumerate(tags))
        return cls(
            enabled=_ensure_bool(data["enabled"], "logging.wandb.enabled"),
            project=_ensure_str(data["project"], "logging.wandb.project"),
            entity=entity,
            run_name=_ensure_str(data["run_name"], "logging.wandb.run_name"),
            mode=_ensure_str(data["mode"], "logging.wandb.mode"),
            tags=parsed_tags,
        )


@dataclass(frozen=True)
class LoggingConfig:
    report_dir: str
    wandb: WandbConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LoggingConfig:
        _check_keys(data, {"report_dir", "wandb"}, "logging")
        return cls(
            report_dir=_ensure_str(data["report_dir"], "logging.report_dir"),
            wandb=WandbConfig.from_dict(_ensure_dict(data["wandb"], "logging.wandb")),
        )


@dataclass(frozen=True)
class DiagnosticsConfig:
    evaluate_headroom: bool
    evaluate_decomposition: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DiagnosticsConfig:
        _check_keys(data, {"evaluate_headroom", "evaluate_decomposition"}, "diagnostics")
        return cls(
            evaluate_headroom=_ensure_bool(data["evaluate_headroom"], "diagnostics.evaluate_headroom"),
            evaluate_decomposition=_ensure_bool(data["evaluate_decomposition"], "diagnostics.evaluate_decomposition"),
        )


@dataclass(frozen=True)
class UnifiedConfig:
    """The root configuration object enforcing strict Version 1 schema."""

    schema_version: int
    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig
    calibration: CalibrationConfig
    logging: LoggingConfig
    diagnostics: DiagnosticsConfig

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported schema_version {self.schema_version}; only version 1 is supported")

    def to_dict(self) -> dict[str, Any]:
        """Convert UnifiedConfig to a clean, serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "experiment": asdict(self.experiment),
            "dataset": asdict(self.dataset),
            "model": asdict(self.model),
            "training": {
                "amp": asdict(self.training.amp),
                "optimizer": asdict(self.training.optimizer),
                "lr_curve": asdict(self.training.lr_curve),
                "annotation_loss": asdict(self.training.annotation_loss),
                "audit_loss": asdict(self.training.audit_loss),
                "counterfactual": asdict(self.training.counterfactual),
                "rollout": asdict(self.training.rollout),
                "schedule": self.training.schedule.to_dict(),
            },
            "checkpoint": asdict(self.checkpoint),
            "calibration": asdict(self.calibration),
            "logging": asdict(self.logging),
            "diagnostics": asdict(self.diagnostics),
        }

    def to_legacy_dataset_config(self, interval: ScheduleInterval | None = None) -> dict[str, Any]:
        """Narrow adapter producing a legacy-compatible dict for dataset/dataloader factories."""
        batch_size = interval.batch_size if interval is not None else 4
        return {
            "dataset": self.dataset.name,
            "data_root": self.dataset.data_root,
            "split_manifest": self.dataset.split_manifest,
            "train_split": self.dataset.train_split,
            "val_split": self.dataset.val_split,
            "test_split": self.dataset.test_split,
            "image_size": self.dataset.image_size,
            "depth_axis": self.dataset.depth_axis,
            "representation": self.dataset.representation,
            "num_classes": self.dataset.num_classes,
            "class_mapping": dict(self.dataset.class_mapping),
            "raw_to_acdc": dict(self.dataset.class_mapping),
            "augment": interval.augment if interval is not None else False,
            "preprocessing": {
                "clipping_min": getattr(self.dataset.preprocessing, "clipping_min", self.dataset.preprocessing.get("clipping_min", 0.5) if isinstance(self.dataset.preprocessing, Mapping) else 0.5),
                "clipping_max": getattr(self.dataset.preprocessing, "clipping_max", self.dataset.preprocessing.get("clipping_max", 99.5) if isinstance(self.dataset.preprocessing, Mapping) else 99.5),
                "foreground_only": getattr(self.dataset.preprocessing, "foreground_only", self.dataset.preprocessing.get("foreground_only", False) if isinstance(self.dataset.preprocessing, Mapping) else False),
            },
            "batch_size": batch_size,
            "num_workers": getattr(self.dataset.dataloader, "num_workers", self.dataset.dataloader.get("num_workers", 0) if isinstance(self.dataset.dataloader, Mapping) else 0),
            "pin_memory": getattr(self.dataset.dataloader, "pin_memory", self.dataset.dataloader.get("pin_memory", True) if isinstance(self.dataset.dataloader, Mapping) else True),
            "persistent_workers": getattr(self.dataset.dataloader, "persistent_workers", self.dataset.dataloader.get("persistent_workers", False) if isinstance(self.dataset.dataloader, Mapping) else False),
            "prefetch_factor": getattr(self.dataset.dataloader, "prefetch_factor", self.dataset.dataloader.get("prefetch_factor", None) if isinstance(self.dataset.dataloader, Mapping) else None),
            "seed": self.experiment.seed,
            "deterministic": self.experiment.deterministic,
            "device": self.experiment.device,
        }

    def to_legacy_model_config(self) -> dict[str, Any]:
        """Narrow adapter producing a legacy-compatible dict for model construction."""
        return {
            "model": {
                "encoder_name": self.model.encoder_name,
                "pretrained_encoder": self.model.pretrained_encoder,
                "encoder_allow_fallback": self.model.fallback,
                "shared_channels": self.model.shared_channels,
                "num_classes": self.model.num_classes,
                "window_k": self.model.window_k,
                "max_turns": self.model.max_turns,
                # The execution mode travels with the model kwargs so the
                # constructed network, the checkpoint lineage, and the config
                # can never disagree about which variant ran.
                "window_mode": self.model.window_mode,
                "candidate_c": self.model.candidate_c.to_mapping(),
            },
            "num_classes": self.model.num_classes,
            "device": self.experiment.device,
            "window_mode": self.model.window_mode,
        }


def parse_unified_config(data: Mapping[str, Any]) -> UnifiedConfig:
    """Parse and strictly validate a dictionary against Schema Version 1."""
    if not isinstance(data, Mapping):
        raise TypeError(f"Config root must be a mapping, got {type(data).__name__}")

    required_sections = {
        "schema_version",
        "experiment",
        "dataset",
        "model",
        "training",
        "checkpoint",
        "calibration",
        "logging",
        "diagnostics",
    }
    _check_keys(data, required_sections, "root")

    version = _ensure_int(data["schema_version"], "schema_version")
    if version != 1:
        raise ValueError(f"Unsupported schema_version {version}; only version 1 is supported")

    dataset_cfg = DatasetConfig.from_dict(_ensure_dict(data["dataset"], "dataset"))
    model_cfg = ModelConfig.from_dict(_ensure_dict(data["model"], "model"))
    if dataset_cfg.name == "acdc" and dataset_cfg.class_mapping != {0: 0, 1: 1, 2: 2, 3: 3}:
        raise ValueError("ACDC class_mapping must preserve the BG/RV/MYO/LV identity mapping")
    if model_cfg.num_classes != dataset_cfg.num_classes:
        raise ValueError(
            f"model.num_classes ({model_cfg.num_classes}) must equal dataset.num_classes ({dataset_cfg.num_classes})"
        )

    calibration_cfg = CalibrationConfig.from_dict(_ensure_dict(data["calibration"], "calibration"))
    if calibration_cfg.split != dataset_cfg.val_split:
        raise ValueError(
            f"calibration.split ('{calibration_cfg.split}') must equal dataset.val_split ('{dataset_cfg.val_split}')"
        )

    training_cfg = TrainingConfig.from_dict(_ensure_dict(data["training"], "training"))
    checkpoint_cfg = CheckpointConfig.from_dict(_ensure_dict(data["checkpoint"], "checkpoint"))

    # Best selection epoch must fall within gated intervals (rollout == "threshold_gate")
    gated_interval = next(
        (i for i in training_cfg.schedule.intervals if i.rollout == "threshold_gate"),
        None,
    )
    if gated_interval is not None:
        if checkpoint_cfg.best_selection_min_epoch < gated_interval.start_epoch:
            raise ValueError(
                f"checkpoint.best_selection_min_epoch must be >= {gated_interval.start_epoch} "
                f"(within gated interval [{gated_interval.start_epoch}, {gated_interval.end_epoch})), "
                f"got {checkpoint_cfg.best_selection_min_epoch}; early selection through configurable cutoff is strictly rejected"
            )
        if checkpoint_cfg.best_selection_min_epoch >= gated_interval.end_epoch:
            raise ValueError(
                f"checkpoint.best_selection_min_epoch must be < {gated_interval.end_epoch} "
                f"(within gated interval [{gated_interval.start_epoch}, {gated_interval.end_epoch})), "
                f"got {checkpoint_cfg.best_selection_min_epoch}; beyond-end selection is strictly rejected"
            )

    return UnifiedConfig(
        schema_version=version,
        experiment=ExperimentConfig.from_dict(_ensure_dict(data["experiment"], "experiment")),
        dataset=dataset_cfg,
        model=model_cfg,
        training=training_cfg,
        checkpoint=checkpoint_cfg,
        calibration=calibration_cfg,
        logging=LoggingConfig.from_dict(_ensure_dict(data["logging"], "logging")),
        diagnostics=DiagnosticsConfig.from_dict(_ensure_dict(data["diagnostics"], "diagnostics")),
    )


def load_unified_config(source: str | Path | io.IOBase) -> UnifiedConfig:
    """Load and validate a YAML configuration from path, string, or stream."""
    if isinstance(source, io.IOBase):
        raw_text = source.read()
    elif isinstance(source, (str, Path)) and Path(source).exists():
        raw_text = Path(source).read_text(encoding="utf-8")
    elif isinstance(source, str):
        raw_text = source
    else:
        raise FileNotFoundError(f"Configuration file not found: {source}")

    parsed = yaml.load(raw_text, Loader=UniqueKeyLoader)
    if not isinstance(parsed, Mapping):
        raise ValueError(f"YAML configuration did not parse to a mapping: {parsed!r}")

    return parse_unified_config(parsed)


def apply_overrides(config: UnifiedConfig, overrides: Mapping[str, Any]) -> UnifiedConfig:
    """Apply runtime CLI overrides cleanly to a UnifiedConfig instance."""
    data = config.to_dict()

    if overrides.get("data_root") is not None:
        data["dataset"]["data_root"] = str(overrides["data_root"])
    if overrides.get("split_manifest") is not None:
        data["dataset"]["split_manifest"] = str(overrides["split_manifest"])
    if overrides.get("num_workers") is not None:
        data["dataset"]["dataloader"]["num_workers"] = int(overrides["num_workers"])
    if overrides.get("device") is not None:
        data["experiment"]["device"] = str(overrides["device"])
    if overrides.get("output_dir") is not None:
        data["checkpoint"]["output_dir"] = str(overrides["output_dir"])
    if overrides.get("report_dir") is not None:
        data["logging"]["report_dir"] = str(overrides["report_dir"])
    if overrides.get("tau_accept") is not None:
        data["training"]["rollout"]["tau"] = float(overrides["tau_accept"])
    if overrides.get("skip_calibration") is not None and overrides["skip_calibration"]:
        data["calibration"]["enabled"] = False
    if overrides.get("wandb") is not None:
        data["logging"]["wandb"]["enabled"] = bool(overrides["wandb"])
    if overrides.get("wandb_mode") is not None:
        data["logging"]["wandb"]["mode"] = str(overrides["wandb_mode"])
    if overrides.get("wandb_project") is not None:
        data["logging"]["wandb"]["project"] = str(overrides["wandb_project"])
    if overrides.get("wandb_entity") is not None:
        data["logging"]["wandb"]["entity"] = overrides["wandb_entity"]
    if overrides.get("image_size") is not None:
        data["dataset"]["image_size"] = int(overrides["image_size"])
    if overrides.get("batch_size") is not None:
        bs = int(overrides["batch_size"])
        for interval in data["training"]["schedule"]["intervals"]:
            interval["batch_size"] = bs
    if overrides.get("wandb_run_name") is not None:
        data["logging"]["wandb"]["run_name"] = str(overrides["wandb_run_name"])

    return parse_unified_config(data)


@dataclass(frozen=True)
class ResolvedExecutionConfig:
    """Resolved downstream execution configuration for models, datasets, loaders, rollout, and metrics."""

    unified_config: UnifiedConfig | None
    flat_config: dict[str, Any]
    is_unified: bool

    @property
    def model_config(self) -> dict[str, Any]:
        if self.unified_config is not None:
            return self.unified_config.to_legacy_model_config()
        if "model" in self.flat_config and isinstance(self.flat_config["model"], Mapping):
            cfg = dict(self.flat_config["model"])
            cfg.setdefault("num_classes", self.flat_config.get("num_classes", 4))
            # A historical flat config carries no execution mode; it is the
            # historical network, stated explicitly rather than left unset.
            cfg["window_mode"] = validate_window_mode(cfg.get("window_mode", DEFAULT_WINDOW_MODE))
            cfg["candidate_c"] = validate_candidate_c_settings(cfg.get("candidate_c"))
            return {
                "model": cfg,
                "num_classes": cfg.get("num_classes", 4),
                "device": self.flat_config.get("device", "cpu"),
                "window_mode": cfg["window_mode"],
            }
        return dict(self.flat_config)

    @property
    def dataset_config(self) -> dict[str, Any]:
        if self.unified_config is not None:
            return self.unified_config.to_legacy_dataset_config()
        return dict(self.flat_config)

    @property
    def dataloader_config(self) -> dict[str, Any]:
        if self.unified_config is not None:
            dl = self.unified_config.dataset.dataloader
            return {
                "num_workers": dl.num_workers,
                "pin_memory": dl.pin_memory,
                "persistent_workers": dl.persistent_workers,
                "prefetch_factor": dl.prefetch_factor,
            }
        return dict(self.flat_config)

    @property
    def device(self) -> torch.device:
        from self_audit.training._utils import resolve_device

        dev_str = (
            self.unified_config.experiment.device
            if self.unified_config is not None
            else self.flat_config.get("device")
        )
        return resolve_device(dev_str)

    @property
    def seed(self) -> int:
        return (
            self.unified_config.experiment.seed
            if self.unified_config is not None
            else int(self.flat_config.get("seed", 42))
        )

    @property
    def deterministic(self) -> bool:
        return (
            self.unified_config.experiment.deterministic
            if self.unified_config is not None
            else bool(self.flat_config.get("deterministic", False))
        )

    @property
    def val_split(self) -> str:
        return (
            self.unified_config.dataset.val_split
            if self.unified_config is not None
            else str(self.flat_config.get("val_split", "val"))
        )

    @property
    def train_split(self) -> str:
        return (
            self.unified_config.dataset.train_split
            if self.unified_config is not None
            else str(self.flat_config.get("train_split", "train"))
        )

    @property
    def test_split(self) -> str:
        return (
            self.unified_config.dataset.test_split
            if self.unified_config is not None
            else str(self.flat_config.get("test_split", "test"))
        )

    @property
    def rollout_tau(self) -> float:
        if self.unified_config is not None:
            return float(self.unified_config.training.rollout.tau)
        audit = self.flat_config.get("audit", {})
        if isinstance(audit, Mapping) and "tau_accept" in audit:
            return float(audit["tau_accept"])
        return float(self.flat_config.get("tau_accept", 0.0))

    @property
    def rollout_max_turns(self) -> int:
        if self.unified_config is not None:
            return int(self.unified_config.training.rollout.max_turns)
        audit = self.flat_config.get("audit", {})
        if isinstance(audit, Mapping) and "t_max" in audit:
            return int(audit["t_max"])
        return int(
            self.flat_config.get(
                "t_max",
                self.flat_config.get("model", {}).get("max_turns", 3)
                if isinstance(self.flat_config.get("model"), Mapping)
                else 3,
            )
        )

    @property
    def rollout_policy(self) -> str:
        if self.unified_config is not None:
            return "self_audit"
        audit = self.flat_config.get("audit", {})
        if isinstance(audit, Mapping) and "rollout_policy" in audit:
            return str(audit["rollout_policy"])
        return str(self.flat_config.get("rollout_policy", "self_audit"))

    @property
    def neutral_margin(self) -> float:
        if self.unified_config is not None:
            return float(self.unified_config.training.audit_loss.neutral_margin)
        audit = self.flat_config.get("audit", {})
        if isinstance(audit, Mapping) and "neutral_margin" in audit:
            return float(audit["neutral_margin"])
        return float(self.flat_config.get("neutral_margin", 0.005))

    @property
    def metric_contract(self) -> str:
        if self.unified_config is not None:
            return str(self.unified_config.calibration.metric_contract)
        return str(self.flat_config.get("metric_contract", "foreground_dice_exclude_v1"))

    @property
    def window_mode(self) -> str:
        """The execution mode this resolved config will actually build."""
        if self.unified_config is not None:
            return self.unified_config.model.window_mode
        model_section = self.flat_config.get("model")
        raw = (
            model_section.get("window_mode", DEFAULT_WINDOW_MODE)
            if isinstance(model_section, Mapping)
            else self.flat_config.get("window_mode", DEFAULT_WINDOW_MODE)
        )
        return validate_window_mode(raw)

    @property
    def candidate_c_settings(self) -> dict[str, Any]:
        """The complete validated Candidate C solver mapping."""
        if self.unified_config is not None:
            return self.unified_config.model.candidate_c.to_mapping()
        model_section = self.flat_config.get("model")
        raw = model_section.get("candidate_c") if isinstance(model_section, Mapping) else None
        return validate_candidate_c_settings(raw)

    @property
    def metric_space(self) -> str:
        if self.unified_config is not None:
            return str(self.unified_config.calibration.metric_space)
        return str(self.flat_config.get("metric_space", "slice_proxy"))

    def to_legacy_dict(self) -> dict[str, Any]:
        """Produce a complete legacy dictionary for legacy consumers and binders."""
        if self.unified_config is not None:
            base = self.unified_config.to_legacy_dataset_config()
            base.update(self.unified_config.to_legacy_model_config())
            base["audit"] = {
                "t_max": self.rollout_max_turns,
                "tau_accept": self.rollout_tau,
                "neutral_margin": self.neutral_margin,
            }
            base["tau_accept"] = self.rollout_tau
            base["t_max"] = self.rollout_max_turns
            base["neutral_margin"] = self.neutral_margin
            base["output_dir"] = self.unified_config.checkpoint.output_dir
            base["report_dir"] = self.unified_config.logging.report_dir
            base["metric_contract"] = self.metric_contract
            base["metric_space"] = self.metric_space
            return base
        return dict(self.flat_config)

    def to_dict(self) -> dict[str, Any]:
        if self.unified_config is not None:
            return self.unified_config.to_dict()
        return dict(self.flat_config)

    def __getitem__(self, key: str) -> Any:
        return self.to_legacy_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_legacy_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.to_legacy_dict()

    def build_model(self, device: torch.device | None = None) -> torch.nn.Module:
        from self_audit.training._utils import build_model_from_config

        target_device = device or self.device
        return build_model_from_config(self.to_legacy_dict(), target_device)

    def build_dataset(self, split: str | None = None, train: bool = False) -> torch.utils.data.Dataset:
        from self_audit.training._utils import build_patient_dataset

        split_name = split or (self.train_split if train else self.val_split)
        return build_patient_dataset(self.to_legacy_dict(), split=split_name, train=train)

    def build_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        train: bool = False,
        batch_size: int | None = None,
    ) -> torch.utils.data.DataLoader:
        from self_audit.training._utils import build_data_loader

        return build_data_loader(
            dataset,
            self.to_legacy_dict(),
            device=self.device,
            train=train,
            batch_size=batch_size,
        )

    def validate_splits(self) -> dict[str, Any]:
        from self_audit.training._utils import validate_dataset_splits

        return validate_dataset_splits(self.to_legacy_dict())


def resolve_downstream_config(
    source: str | Path | Mapping[str, Any] | UnifiedConfig,
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedExecutionConfig:
    """Resolve a unified or historical flat configuration into a unified execution adapter.

    If source has schema_version == 1, it is strictly validated as a UnifiedConfig.
    If source is a legacy flat config, it is explicitly recognized as historical without
    silently fabricating unified schema sections.
    """
    if isinstance(source, UnifiedConfig):
        cfg = source
        if overrides:
            cfg = apply_overrides(cfg, overrides)
        return ResolvedExecutionConfig(unified_config=cfg, flat_config={}, is_unified=True)

    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {p}")
        raw_text = p.read_text(encoding="utf-8")
        parsed = yaml.load(raw_text, Loader=UniqueKeyLoader)
        if not isinstance(parsed, Mapping):
            raise ValueError(f"Configuration did not parse to a mapping: {parsed!r}")
    elif isinstance(source, Mapping):
        parsed = dict(source)
    else:
        raise TypeError(f"Unsupported config source type: {type(source).__name__}")

    # Check if this is a Schema Version 1 UnifiedConfig
    if parsed.get("schema_version") == 1:
        unified = parse_unified_config(parsed)
        if overrides:
            unified = apply_overrides(unified, overrides)
        return ResolvedExecutionConfig(unified_config=unified, flat_config={}, is_unified=True)

    # Historical flat config recognized explicitly for reproducibility
    flat = dict(parsed)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                flat[k] = v
    return ResolvedExecutionConfig(unified_config=None, flat_config=flat, is_unified=False)

