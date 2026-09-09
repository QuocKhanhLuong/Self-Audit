"""Explicit unified schedule and interval definitions for Self-Audit.

Enforces strict contiguous coverage [0, total_epochs), boundary detection,
trainability contracts, and per-interval learning rates / objectives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


ALLOWED_TRAINABLE = {"annotation", "auditor", "all"}
ALLOWED_OBJECTIVES = {
    "weighted_a0_a3",
    "counterfactual_audit",
    "retained_final_annotation",
}
ALLOWED_TRANSITION_POPULATION = {
    "none",
    "adjacent_and_synthetic",
    "active_attempted",
}
ALLOWED_ROLLOUT = {
    "propagate_no_audit",
    "annotation_eval",
    "threshold_gate",
}


def _validate_nonnegative_finite(val: Any, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(val).__name__} ({val!r})")
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite, got {fval}")
    if fval < 0.0:
        raise ValueError(f"{name} must be non-negative, got {fval}")
    return fval


def _validate_positive_int(val: Any, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise TypeError(f"{name} must be an integer, got {type(val).__name__} ({val!r})")
    if val < 1:
        raise ValueError(f"{name} must be >= 1, got {val}")
    return val


def _validate_nonnegative_int(val: Any, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise TypeError(f"{name} must be an integer, got {type(val).__name__} ({val!r})")
    if val < 0:
        raise ValueError(f"{name} must be >= 0, got {val}")
    return val


def _validate_bool(val: Any, name: str) -> bool:
    if not isinstance(val, bool):
        raise TypeError(f"{name} must be a boolean, got {type(val).__name__} ({val!r})")
    return val


@dataclass(frozen=True)
class ScheduleInterval:
    """An explicit execution interval in the unified training schedule."""

    start_epoch: int
    end_epoch: int
    name: str
    trainable: str
    encoder_lr: float
    annotation_lr: float
    auditor_lr: float
    annotation_weight: float
    audit_weight: float
    objective: str
    transition_population: str
    rollout: str
    batch_size: int
    accumulation_steps: int
    augment: bool
    reset_optimizer: bool

    def __post_init__(self) -> None:
        start = _validate_nonnegative_int(self.start_epoch, "start_epoch")
        end = _validate_positive_int(self.end_epoch, "end_epoch")
        if start >= end:
            raise ValueError(f"start_epoch ({start}) must be strictly less than end_epoch ({end})")

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

        if self.trainable not in ALLOWED_TRAINABLE:
            raise ValueError(
                f"trainable must be one of {sorted(ALLOWED_TRAINABLE)}, got {self.trainable!r}"
            )

        _validate_nonnegative_finite(self.encoder_lr, "encoder_lr")
        _validate_nonnegative_finite(self.annotation_lr, "annotation_lr")
        _validate_nonnegative_finite(self.auditor_lr, "auditor_lr")
        _validate_nonnegative_finite(self.annotation_weight, "annotation_weight")
        _validate_nonnegative_finite(self.audit_weight, "audit_weight")

        if self.objective not in ALLOWED_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {sorted(ALLOWED_OBJECTIVES)}, got {self.objective!r}"
            )

        if self.transition_population not in ALLOWED_TRANSITION_POPULATION:
            raise ValueError(
                f"transition_population must be one of {sorted(ALLOWED_TRANSITION_POPULATION)}, "
                f"got {self.transition_population!r}"
            )

        if self.rollout not in ALLOWED_ROLLOUT:
            raise ValueError(
                f"rollout must be one of {sorted(ALLOWED_ROLLOUT)}, got {self.rollout!r}"
            )

        _validate_positive_int(self.batch_size, "batch_size")
        _validate_positive_int(self.accumulation_steps, "accumulation_steps")
        _validate_bool(self.augment, "augment")
        _validate_bool(self.reset_optimizer, "reset_optimizer")

        # Validate positive LRs for trainable modules and zero LRs for frozen modules
        # to ensure zero LR trainable gradients do not corrupt grad_clip.
        if self.trainable == "annotation":
            if self.encoder_lr <= 0.0:
                raise ValueError(f"Trainable 'annotation' requires positive encoder_lr > 0, got {self.encoder_lr}")
            if self.annotation_lr <= 0.0:
                raise ValueError(f"Trainable 'annotation' requires positive annotation_lr > 0, got {self.annotation_lr}")
            if self.auditor_lr != 0.0:
                raise ValueError(f"Frozen auditor in 'annotation' requires auditor_lr=0.0, got {self.auditor_lr}")
        elif self.trainable == "auditor":
            if self.encoder_lr != 0.0:
                raise ValueError(f"Frozen encoder in 'auditor' requires encoder_lr=0.0, got {self.encoder_lr}")
            if self.annotation_lr != 0.0:
                raise ValueError(f"Frozen annotation in 'auditor' requires annotation_lr=0.0, got {self.annotation_lr}")
            if self.auditor_lr <= 0.0:
                raise ValueError(f"Trainable 'auditor' requires positive auditor_lr > 0, got {self.auditor_lr}")
        elif self.trainable == "all":
            if self.encoder_lr <= 0.0:
                raise ValueError(f"Trainable 'all' requires positive encoder_lr > 0, got {self.encoder_lr}")
            if self.annotation_lr <= 0.0:
                raise ValueError(f"Trainable 'all' requires positive annotation_lr > 0, got {self.annotation_lr}")
            if self.auditor_lr <= 0.0:
                raise ValueError(f"Trainable 'all' requires positive auditor_lr > 0, got {self.auditor_lr}")

        # Cross-field constraints per approved design specification
        if self.objective == "weighted_a0_a3":
            if self.trainable != "annotation":
                raise ValueError(f"Objective 'weighted_a0_a3' requires trainable='annotation', got {self.trainable!r}")
            if self.transition_population != "none":
                raise ValueError(f"Objective 'weighted_a0_a3' requires transition_population='none', got {self.transition_population!r}")
            if self.rollout != "propagate_no_audit":
                raise ValueError(f"Objective 'weighted_a0_a3' requires rollout='propagate_no_audit', got {self.rollout!r}")
            if self.audit_weight != 0.0:
                raise ValueError(f"Objective 'weighted_a0_a3' requires audit_weight=0.0, got {self.audit_weight}")
            if self.annotation_weight != 1.0:
                raise ValueError(f"Objective 'weighted_a0_a3' requires annotation_weight=1.0 per approved schedule, got {self.annotation_weight}")
            if self.auditor_lr != 0.0:
                raise ValueError(f"Objective 'weighted_a0_a3' requires auditor_lr=0.0, got {self.auditor_lr}")
            if self.reset_optimizer:
                raise ValueError("Objective 'weighted_a0_a3' (initial bootstrap) must have reset_optimizer=False")

        elif self.objective == "counterfactual_audit":
            if self.trainable != "auditor":
                raise ValueError(f"Objective 'counterfactual_audit' requires trainable='auditor', got {self.trainable!r}")
            if self.transition_population != "adjacent_and_synthetic":
                raise ValueError(f"Objective 'counterfactual_audit' requires transition_population='adjacent_and_synthetic', got {self.transition_population!r}")
            if self.rollout != "annotation_eval":
                raise ValueError(f"Objective 'counterfactual_audit' requires rollout='annotation_eval', got {self.rollout!r}")
            if self.annotation_weight != 0.0:
                raise ValueError(f"Objective 'counterfactual_audit' requires annotation_weight=0.0, got {self.annotation_weight}")
            if self.audit_weight != 1.0:
                raise ValueError(f"Objective 'counterfactual_audit' requires audit_weight=1.0 per approved schedule, got {self.audit_weight}")
            if self.encoder_lr != 0.0 or self.annotation_lr != 0.0:
                raise ValueError("Objective 'counterfactual_audit' requires encoder_lr=0.0 and annotation_lr=0.0")
            if not self.reset_optimizer:
                raise ValueError("Objective 'counterfactual_audit' (auditor boundary) requires reset_optimizer=True")

        elif self.objective == "retained_final_annotation":
            if self.trainable != "all":
                raise ValueError(f"Objective 'retained_final_annotation' requires trainable='all', got {self.trainable!r}")
            if self.transition_population != "active_attempted":
                raise ValueError(f"Objective 'retained_final_annotation' requires transition_population='active_attempted', got {self.transition_population!r}")
            if self.rollout != "threshold_gate":
                raise ValueError(f"Objective 'retained_final_annotation' requires rollout='threshold_gate', got {self.rollout!r}")
            if self.annotation_weight != 1.0 or self.audit_weight != 1.0:
                raise ValueError("Objective 'retained_final_annotation' requires annotation_weight=1.0 and audit_weight=1.0")
            if not self.reset_optimizer:
                raise ValueError("Objective 'retained_final_annotation' (joint boundary) requires reset_optimizer=True")

    @property
    def num_epochs(self) -> int:
        return self.end_epoch - self.start_epoch

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScheduleInterval:
        if not isinstance(data, Mapping):
            raise TypeError(f"Interval data must be a mapping, got {type(data).__name__}")
        allowed_keys = {
            "start_epoch",
            "end_epoch",
            "name",
            "trainable",
            "encoder_lr",
            "annotation_lr",
            "auditor_lr",
            "annotation_weight",
            "audit_weight",
            "objective",
            "transition_population",
            "rollout",
            "batch_size",
            "accumulation_steps",
            "augment",
            "reset_optimizer",
        }
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown keys in schedule interval: {sorted(unknown)}")
        missing = allowed_keys - set(data.keys())
        if missing:
            raise ValueError(f"Missing required keys in schedule interval: {sorted(missing)}")

        return cls(
            start_epoch=data["start_epoch"],
            end_epoch=data["end_epoch"],
            name=data["name"],
            trainable=data["trainable"],
            encoder_lr=data["encoder_lr"],
            annotation_lr=data["annotation_lr"],
            auditor_lr=data["auditor_lr"],
            annotation_weight=data["annotation_weight"],
            audit_weight=data["audit_weight"],
            objective=data["objective"],
            transition_population=data["transition_population"],
            rollout=data["rollout"],
            batch_size=data["batch_size"],
            accumulation_steps=data["accumulation_steps"],
            augment=data["augment"],
            reset_optimizer=data["reset_optimizer"],
        )


@dataclass(frozen=True)
class Schedule:
    """The complete, strict unified schedule across all training epochs."""

    total_epochs: int
    intervals: tuple[ScheduleInterval, ...]

    def __post_init__(self) -> None:
        total = _validate_positive_int(self.total_epochs, "total_epochs")
        if not self.intervals:
            raise ValueError("Schedule intervals cannot be empty")

        # Validate contiguous, non-overlapping coverage of [0, total_epochs)
        if self.intervals[0].start_epoch != 0:
            raise ValueError(
                f"First schedule interval must start at epoch 0, got {self.intervals[0].start_epoch}"
            )

        for i in range(len(self.intervals) - 1):
            curr_int = self.intervals[i]
            next_int = self.intervals[i + 1]
            if curr_int.end_epoch != next_int.start_epoch:
                raise ValueError(
                    f"Schedule intervals not contiguous between interval {i} ('{curr_int.name}') "
                    f"ending at {curr_int.end_epoch} and interval {i+1} ('{next_int.name}') "
                    f"starting at {next_int.start_epoch}"
                )

        if self.intervals[-1].end_epoch != total:
            raise ValueError(
                f"Last schedule interval end_epoch ({self.intervals[-1].end_epoch}) "
                f"must equal schedule total_epochs ({total})"
            )

        # Validate objective ordering across intervals according to approved curriculum
        obj_order = {
            "weighted_a0_a3": 0,
            "counterfactual_audit": 1,
            "retained_final_annotation": 2,
        }
        for i in range(len(self.intervals) - 1):
            curr_obj = self.intervals[i].objective
            next_obj = self.intervals[i + 1].objective
            if obj_order[curr_obj] > obj_order[next_obj]:
                raise ValueError(
                    f"Invalid schedule objective ordering: interval {i} ('{self.intervals[i].name}') "
                    f"has objective '{curr_obj}' followed by interval {i+1} ('{self.intervals[i+1].name}') "
                    f"with objective '{next_obj}'; objectives must progress according to curriculum: {list(obj_order.keys())}"
                )

    def get_interval(self, epoch: int) -> ScheduleInterval:
        """Return the ScheduleInterval for a zero-based global epoch."""
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError(f"epoch must be an integer, got {type(epoch).__name__}")
        if epoch < 0 or epoch >= self.total_epochs:
            raise IndexError(f"Epoch {epoch} out of range [0, {self.total_epochs})")

        for interval in self.intervals:
            if interval.start_epoch <= epoch < interval.end_epoch:
                return interval
        raise IndexError(f"Epoch {epoch} did not match any schedule interval")

    def get_interval_index(self, epoch: int) -> int:
        """Return the 0-based interval index for a zero-based global epoch."""
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError(f"epoch must be an integer, got {type(epoch).__name__}")
        if epoch < 0 or epoch >= self.total_epochs:
            raise IndexError(f"Epoch {epoch} out of range [0, {self.total_epochs})")

        for idx, interval in enumerate(self.intervals):
            if interval.start_epoch <= epoch < interval.end_epoch:
                return idx
        raise IndexError(f"Epoch {epoch} did not match any schedule interval")

    def is_interval_start(self, epoch: int) -> bool:
        """Return True if the epoch is the start of any interval."""
        if epoch < 0 or epoch >= self.total_epochs:
            return False
        return any(interval.start_epoch == epoch for interval in self.intervals)

    def is_reset_boundary(self, epoch: int) -> bool:
        """Return True if the epoch starts an interval that specifies reset_optimizer."""
        if epoch < 0 or epoch >= self.total_epochs:
            return False
        interval = self.get_interval(epoch)
        return interval.start_epoch == epoch and interval.reset_optimizer

    def steps_per_epoch(self, interval: ScheduleInterval, num_batches: int) -> int:
        """Calculate optimizer steps per epoch for an interval and loader cardinality."""
        if num_batches <= 0:
            return 0
        return int(math.ceil(num_batches / interval.accumulation_steps))

    def interval_total_steps(self, interval: ScheduleInterval, num_batches: int) -> int:
        """Calculate total optimizer steps in an interval."""
        return self.steps_per_epoch(interval, num_batches) * interval.num_epochs

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_epochs": self.total_epochs,
            "intervals": [interval.to_dict() for interval in self.intervals],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Schedule:
        if not isinstance(data, Mapping):
            raise TypeError(f"Schedule data must be a mapping, got {type(data).__name__}")
        allowed_keys = {"total_epochs", "intervals"}
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown keys in schedule: {sorted(unknown)}")
        missing = allowed_keys - set(data.keys())
        if missing:
            raise ValueError(f"Missing required keys in schedule: {sorted(missing)}")

        raw_intervals = data["intervals"]
        if not isinstance(raw_intervals, Sequence) or isinstance(raw_intervals, (str, bytes)):
            raise TypeError("intervals must be a sequence of mappings")

        intervals = tuple(ScheduleInterval.from_dict(item) for item in raw_intervals)
        return cls(
            total_epochs=data["total_epochs"],
            intervals=intervals,
        )
