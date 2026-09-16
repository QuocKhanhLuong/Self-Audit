"""Strict resolved configuration for the mask-free 150-epoch timeline.

One config object drives one dataset run. Unknown keys are rejected so a stale
supervised recipe cannot leak options into this pipeline, and there is no
pretrained-weight, reference-mask or legacy-override field by construction.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = "maskfree150.config.v1"

DATASETS = ("acdc", "mnms")
PROTOCOLS = ("auto", "cine_predictive", "spatial_predictive")
WANDB_MODES = ("offline", "online", "disabled")
# The observation/audit path may execute on the trainer's model device, or on
# an explicitly selected backend for a matched reference run. ``auto`` is the
# production setting: it resolves to the trainer model device at construction
# time rather than silently falling back when CUDA is unavailable.
AUDIT_DEVICES = ("auto", "cpu", "cuda")
RUNTIME_FIELDS = (
    "timing_mode", "logging_mode", "log_buffer_bytes", "data_cache_bytes",
    "prefetch_batches", "prefetch_max_bytes", "candidate_workers", "candidate_worker_threads",
)

#: Fields that define the scientific identity of a run. A resume whose config
#: disagrees on any of them is a different experiment and fails closed.
SCIENTIFIC_FIELDS = (
    "dataset",
    "data_root",
    "total_epochs",
    "seed",
    "batch_size",
    "accumulation_steps",
    "image_size",
    "lr",
    "weight_decay",
    "warmup_epochs",
    "width",
    "feature_dim",
    "protocol",
    "depth_axis",
    "amp",
)

#: Non-scientific fields. This classification does not authorize exact-resume
#: migration: trainer identity also locks source, backend and RUNTIME_FIELDS.
#: A runtime tuning trial requires a fresh run unless a migration is supported.
OPERATIONAL_FIELDS = (
    "output_dir",
    "device",
    "audit_device",
    "num_workers",
    "wandb_mode",
    "wandb_project",
    "run_id",
    "max_steps",
    "max_epochs",
    "resume",
    "allow_cpu",
    "epoch_validation",
    "epoch_reference_config",
) + RUNTIME_FIELDS

#: Keys that belonged to the supervised pipelines. Rejected with an explicit
#: message instead of the generic unknown-key error, because silently accepting
#: one of them would be a supervision leak, not a typo.
FORBIDDEN_KEYS = {
    "pretrained_encoder": "mask-free producer is randomly initialised",
    "pretrained": "no external checkpoints in the generating pipeline",
    "checkpoint": "use `resume`; supervised checkpoints cannot be migrated",
    "teacher_checkpoint": "no ground-truth-trained teacher",
    "mask_root": "the generating pipeline never reads manual masks",
    "label_root": "the generating pipeline never reads manual masks",
    "reference_root": "reference masks belong to the isolated evaluator only",
    "gt_root": "reference masks belong to the isolated evaluator only",
    "best_metric": "no Dice-selected checkpoint",
    "select_by_dice": "no Dice-selected checkpoint",
    "threshold": "thresholds are never tuned against references",
}


class ConfigError(ValueError):
    """Raised for any configuration that cannot be accepted as-is."""


@dataclass(frozen=True)
class MaskfreeConfig:
    dataset: str
    data_root: str
    output_dir: str
    total_epochs: int = 150
    seed: int = 42
    batch_size: int = 8
    accumulation_steps: int = 1
    image_size: int = 128
    lr: float = 0.001
    weight_decay: float = 0.0001
    warmup_epochs: int = 5
    width: int = 16
    feature_dim: int = 16
    protocol: str = "auto"
    depth_axis: int = 2
    device: str = "cuda"
    audit_device: str = "auto"
    timing_mode: str = "production"
    logging_mode: str = "buffered"
    log_buffer_bytes: int = 256 * 1024
    data_cache_bytes: int = 64 * 1024 * 1024
    prefetch_batches: int = 0
    prefetch_max_bytes: int = 32 * 1024 * 1024
    candidate_workers: int = 0
    candidate_worker_threads: int = 1
    num_workers: int = 0
    amp: bool = True
    wandb_mode: str = "offline"
    wandb_project: str = "self-audit-maskfree"
    run_id: str | None = None
    max_steps: int | None = None
    max_epochs: int | None = None
    resume: str | None = None
    allow_cpu: bool = False
    # Epoch validation is a report-only observation of frozen student outputs
    # on the image-only development split.  It is deliberately operational:
    # enabling it or forwarding a child evaluator config cannot change the
    # scientific identity of the learned run or select a checkpoint.
    epoch_validation: bool = True
    epoch_reference_config: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    # -- validation -----------------------------------------------------
    def _validate(self) -> None:
        if self.dataset not in DATASETS:
            raise ConfigError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        if self.protocol not in PROTOCOLS:
            raise ConfigError(f"protocol must be one of {PROTOCOLS}, got {self.protocol!r}")
        if self.wandb_mode not in WANDB_MODES:
            raise ConfigError(f"wandb_mode must be one of {WANDB_MODES}, got {self.wandb_mode!r}")
        if self.audit_device not in AUDIT_DEVICES:
            raise ConfigError(
                f"audit_device must be one of {AUDIT_DEVICES}, got {self.audit_device!r}"
            )
        if self.timing_mode not in ("production", "diagnostic"):
            raise ConfigError("timing_mode must be production or diagnostic")
        if self.logging_mode not in ("sync", "buffered"):
            raise ConfigError("logging_mode must be sync or buffered")
        for name in ("log_buffer_bytes", "data_cache_bytes", "prefetch_batches",
                     "prefetch_max_bytes", "candidate_workers", "candidate_worker_threads"):
            value = getattr(self, name)
            minimum = 1 if name in ("log_buffer_bytes", "prefetch_max_bytes", "candidate_worker_threads") else 0
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ConfigError(f"{name} must be an integer >= {minimum}")
        if self.prefetch_batches > 2:
            raise ConfigError("prefetch_batches must be 0, 1 or 2 (bounded input lookahead)")
        if self.candidate_workers not in (0, 2, 4):
            raise ConfigError("candidate_workers must be 0, 2 or 4")
        if self.depth_axis not in (0, 1, 2):
            raise ConfigError("depth_axis must be 0, 1 or 2")
        if not str(self.data_root).strip():
            raise ConfigError("data_root is required")
        if not str(self.output_dir).strip():
            raise ConfigError("output_dir is required")

        for name in ("total_epochs", "batch_size", "accumulation_steps", "image_size",
                     "width", "feature_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ConfigError(f"{name} must be a positive integer, got {value!r}")
        for name in ("lr", "weight_decay"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ConfigError(f"{name} must be a non-negative number, got {value!r}")
        if self.lr <= 0:
            raise ConfigError("lr must be positive")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ConfigError("seed must be a non-negative integer")
        if not isinstance(self.warmup_epochs, int) or self.warmup_epochs < 0:
            raise ConfigError("warmup_epochs must be a non-negative integer")
        if self.warmup_epochs >= self.total_epochs:
            raise ConfigError("warmup_epochs must be smaller than total_epochs")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ConfigError("num_workers must be a non-negative integer")
        if self.image_size % 8 != 0:
            raise ConfigError("image_size must be a multiple of 8 for the block role grid")

        if self.max_steps is not None:
            if not isinstance(self.max_steps, int) or self.max_steps <= 0:
                raise ConfigError("max_steps must be a positive integer when given")
        if self.max_epochs is not None:
            if not isinstance(self.max_epochs, int) or self.max_epochs <= 0:
                raise ConfigError("max_epochs must be a positive integer when given")
            if self.max_epochs > self.total_epochs:
                raise ConfigError("max_epochs cannot exceed total_epochs")
        for name in ("run_id", "resume"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ConfigError(f"{name} must be a string or null")
        for name in ("amp", "allow_cpu", "epoch_validation"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a boolean")
        if self.epoch_reference_config is not None:
            if not isinstance(self.epoch_reference_config, str):
                raise ConfigError("epoch_reference_config must be a string or null")
            if not self.epoch_reference_config.strip():
                raise ConfigError("epoch_reference_config must be a non-empty string when given")
        if self.run_id is not None and (not self.run_id or self.run_id in (".", "..") or
                                       Path(self.run_id).name != self.run_id or "\\" in self.run_id):
            raise ConfigError("run_id must be a nonempty single path component")

    # -- accessors ------------------------------------------------------
    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.accumulation_steps

    @property
    def is_bounded_run(self) -> bool:
        """True when max_steps/max_epochs cap the run below the full timeline."""
        return self.max_steps is not None or self.max_epochs is not None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = CONFIG_SCHEMA_VERSION
        return payload

    def scientific_identity(self) -> dict[str, Any]:
        identity = {name: getattr(self, name) for name in SCIENTIFIC_FIELDS}
        identity["schema_version"] = CONFIG_SCHEMA_VERSION
        return identity

    def replace(self, **overrides: Any) -> "MaskfreeConfig":
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        unknown = set(overrides) - set(payload)
        if unknown:
            raise ConfigError(f"unknown config overrides: {sorted(unknown)}")
        payload.update(overrides)
        return MaskfreeConfig(**payload)


def config_from_dict(payload: dict[str, Any]) -> MaskfreeConfig:
    """Build a config from a plain mapping, rejecting anything unrecognised."""
    if not isinstance(payload, dict):
        raise ConfigError("config payload must be a mapping")
    data = dict(payload)
    version = data.pop("schema_version", CONFIG_SCHEMA_VERSION)
    if version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"unsupported schema_version: {version}")

    forbidden = sorted(set(data) & set(FORBIDDEN_KEYS))
    if forbidden:
        details = "; ".join(f"{key}: {FORBIDDEN_KEYS[key]}" for key in forbidden)
        raise ConfigError(f"forbidden configuration keys ({details})")

    known = {f.name for f in fields(MaskfreeConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"unknown configuration keys: {unknown}")

    missing = [name for name in ("dataset", "data_root", "output_dir") if name not in data]
    if missing:
        raise ConfigError(f"missing required configuration keys: {missing}")
    return MaskfreeConfig(**data)


def load_config(path: str | Path) -> MaskfreeConfig:
    """Load a YAML (or JSON) config file into the strict dataclass."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigError("PyYAML is required to read YAML configs") from exc
        payload = yaml.safe_load(text)
    if payload is None:
        raise ConfigError(f"empty config file: {path}")
    return config_from_dict(payload)


def compare_configs(stored: dict[str, Any], current: MaskfreeConfig) -> dict[str, Any]:
    """Split stored-vs-current config differences into scientific and operational.

    A scientific difference means the checkpoint belongs to another experiment.
    An operational difference is allowed on resume and reported.
    """
    current_payload = current.to_dict()
    scientific: dict[str, Any] = {}
    operational: dict[str, Any] = {}
    for name in SCIENTIFIC_FIELDS:
        if stored.get(name) != current_payload.get(name):
            scientific[name] = {"stored": stored.get(name), "current": current_payload.get(name)}
    for name in OPERATIONAL_FIELDS:
        if stored.get(name) != current_payload.get(name):
            operational[name] = {"stored": stored.get(name), "current": current_payload.get(name)}
    stored_schema = stored.get("schema_version", CONFIG_SCHEMA_VERSION)
    if stored_schema != CONFIG_SCHEMA_VERSION:
        scientific["schema_version"] = {"stored": stored_schema, "current": CONFIG_SCHEMA_VERSION}
    return {"scientific": scientific, "operational": operational}
