"""One trainer, one resolved config, one continuous 150-epoch global timeline.

Every epoch visits each training sampling unit exactly once under a seeded
permutation and performs, for every batch:

1. image-only producer self-supervision on the fitting view;
2. bounded hypothesis-bank generation from *detached* producer features;
3. evidence auditing against the selection observations (from epoch 0);
4. one update of ``student_no_audit`` on the initial hypothesis and one update
   of ``student_audited`` on the audited selection, under identical budgets.

The two students share the bank, the initialisation, the augmentation stream and
the update schedule; only the label source differs. Producer gradients never
come from a student, and student gradients never reach the producer: labels are
detached before the student loss.

Dependencies on the other owners' modules are resolved through
:func:`resolve_components`, which imports the exact interfaces frozen in
``reports/maskfree150/architecture_contract.md``. There is no stub training
path: if an owner's module is absent, construction fails loudly with the list of
missing interfaces rather than training against invented data.
"""
from __future__ import annotations

import math
import time
import traceback
import uuid
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from . import runtime
from .progress import current_progress
from .config import MaskfreeConfig, compare_configs
from .contracts import VERSION as CONTRACT_VERSION
from .contracts import AuditResult, FittingView, Hypothesis, ScoringView, TrainingUnit

TRAINER_SCHEMA_VERSION = "maskfree150.trainer.v1"

#: Fixed provisional budgets from the architecture contract.
AUDIT_ROUNDS = 2
IMPROVEMENT_THRESHOLD = 0.001
LABEL_RAMP_EPOCHS = 20
OBSERVATION_MAX_ITERATIONS = 5
OBSERVATION_VARIANCE_FLOOR = 0.05
OBSERVATION_BETA = 0.01

STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"


class TrainerContractError(RuntimeError):
    """Raised when the run cannot proceed without violating the contract."""


class MissingComponentError(TrainerContractError):
    """Raised when another owner's frozen interface is not importable yet."""


class ResumeIdentityError(TrainerContractError):
    """Raised when a checkpoint belongs to a different experiment."""


# ---------------------------------------------------------------------------
# component resolution
# ---------------------------------------------------------------------------
#: module attribute -> (module suffix, attribute name, owning worker)
COMPONENT_SPECS: dict[str, tuple[str, str, str]] = {
    "discover_dataset": ("data", "discover_dataset", "W3"),
    "dataset_factory": ("data", "ImageOnlyDataset", "W3"),
    "load_verification_unit": ("data", "load_verification_unit", "W3"),
    "load_full_input": ("data", "load_full_input", "W3"),
    "make_models": ("models", "make_models", "W4"),
    "producer_loss": ("losses", "producer_loss", "W4"),
    "student_loss": ("losses", "student_loss", "W4"),
    "generate_bank": ("hypotheses", "generate_bank", "W2"),
    "audit_bank": ("auditor", "audit_bank", "W2"),
    "observation_model": ("observation", "ObservationModel", "W1"),
    "run_bank_experiments": ("experiments", "run_bank_experiments", "W6"),
    "matched_coverage": ("experiments", "matched_coverage", "W6"),
    "verify_frozen_bank": ("experiments", "verify_frozen_bank", "W6"),
    "export_prediction": ("export", "export_prediction", "W7"),
    "freeze_predictions": ("export", "freeze_predictions", "W7"),
    "validate_freeze": ("export", "validate_freeze", "W7"),
}


@dataclass(frozen=True)
class Components:
    """The frozen cross-owner interfaces this trainer calls."""

    discover_dataset: Callable[..., dict[str, Any]]
    dataset_factory: Callable[..., Any]
    load_verification_unit: Callable[..., ScoringView]
    load_full_input: Callable[..., tuple[torch.Tensor, dict[str, Any]]]
    make_models: Callable[..., dict[str, torch.nn.Module]]
    producer_loss: Callable[..., tuple[torch.Tensor, dict[str, Any]]]
    student_loss: Callable[..., tuple[torch.Tensor, dict[str, Any]]]
    generate_bank: Callable[..., list[Hypothesis]]
    audit_bank: Callable[..., AuditResult]
    observation_model: Callable[..., Any]
    run_bank_experiments: Callable[..., dict[str, Any]]
    matched_coverage: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    verify_frozen_bank: Callable[..., dict[str, Any]]
    export_prediction: Callable[..., dict[str, Any]]
    freeze_predictions: Callable[..., dict[str, Any]]
    validate_freeze: Callable[..., None]


def resolve_components(*, required: Sequence[str] | None = None) -> Components:
    """Import the other owners' frozen interfaces.

    Raises :class:`MissingComponentError` listing every interface that is not
    importable, naming the owning worker, so a delayed dependency is reported
    instead of silently replaced.
    """
    import importlib

    names = list(COMPONENT_SPECS) if required is None else list(required)
    resolved: dict[str, Any] = {}
    missing: list[str] = []
    for name in names:
        module_suffix, attribute, owner = COMPONENT_SPECS[name]
        try:
            module = importlib.import_module(f"{__package__}.{module_suffix}")
        except ImportError as exc:
            missing.append(f"{owner}: {__package__}.{module_suffix} ({exc})")
            continue
        if not hasattr(module, attribute):
            missing.append(f"{owner}: {__package__}.{module_suffix}.{attribute} (attribute absent)")
            continue
        resolved[name] = getattr(module, attribute)
    if missing:
        raise MissingComponentError(
            "mask-free trainer dependencies are not available yet:\n  - "
            + "\n  - ".join(missing)
        )
    if required is not None:
        for name in COMPONENT_SPECS:
            resolved.setdefault(name, _unavailable_component(name))
    return Components(**resolved)


def _unavailable_component(name: str) -> Callable[..., Any]:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise MissingComponentError(f"component {name!r} was not resolved for this trainer")

    return _raise


# ---------------------------------------------------------------------------
# accumulators
# ---------------------------------------------------------------------------
@dataclass
class EpochAccumulator:
    """Running sums for one global epoch. Denominators are recorded alongside."""

    values: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, key: str, value: float, count: int = 1) -> None:
        if value is None or not math.isfinite(float(value)):
            return
        self.values[key] = self.values.get(key, 0.0) + float(value) * count
        self.counts[key] = self.counts.get(key, 0) + count

    def mean(self, key: str) -> float | None:
        count = self.counts.get(key, 0)
        if count == 0:
            return None
        return self.values[key] / count

    def as_dict(self) -> dict[str, Any]:
        return {
            key: {"mean": self.mean(key), "sum": self.values[key], "count": self.counts[key]}
            for key in sorted(self.values)
        }


def _stable_unit_seed(seed: int, epoch: int, unit_id: str) -> int:
    digest = runtime.sha256_bytes(f"{seed}:{epoch}:{unit_id}".encode("utf-8"))
    return int(digest[:8], 16)


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------
class MaskfreeTrainer:
    """Executable orchestration of the single 150-epoch mask-free timeline."""

    def __init__(
        self,
        config: MaskfreeConfig,
        *,
        components: Components | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self.config = config
        self.components = components if components is not None else resolve_components()
        self.repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]

        self.device = runtime.resolve_device(config.device, allow_cpu=config.allow_cpu)
        self.run_id = config.run_id or self._default_run_id()
        run_root = Path(config.output_dir) / "runs" / "maskfree150" / config.dataset / self.run_id
        if (run_root / "run_identity.json").exists() and not config.resume:
            raise ResumeIdentityError("occupied run directory requires explicit resume")
        self.paths = runtime.RunPaths(run_root).ensure()

        self.source = runtime.source_identity(self.repo_root)
        self.timing = runtime.TimingAccumulator.empty()
        self.timing.device = self.device

        self.sampling_generator = torch.Generator()
        self.sampling_generator.manual_seed(config.seed * 7919 + 13)

        # populated by _setup
        self.manifest: dict[str, Any] = {}
        self.dataset: Any = None
        self.unit_ids: list[str] = []
        self.models: dict[str, torch.nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.scaler: torch.amp.GradScaler | None = None
        self.observation = self.components.observation_model(
            max_iterations=OBSERVATION_MAX_ITERATIONS,
            variance_floor=OBSERVATION_VARIANCE_FLOOR,
            beta=OBSERVATION_BETA,
        )

        self.global_step = 0
        self.component_steps = {"producer": 0, "student_no_audit": 0, "student_audited": 0}
        self.micro_batches_seen = 0
        self.start_epoch = 0
        self.start_batch = 0
        self.resumed_permutation: list[int] | None = None
        #: (epoch, batch_cursor, permutation) that a resume must continue from.
        self._resume_position: tuple[int, int, list[int]] = (0, 0, [])
        self.last_completed_epoch = -1
        self.history: list[dict[str, Any]] = []
        self.sink: runtime.MetricSink | None = None
        self._dropped_partial_records = 0
        self._setup_done = False
        self._partial_epoch: dict[str, Any] | None = None
        self._active_arms: set[str] = set()
        self._student_weight_sums: dict[str, float] = {}

    # -- identity -------------------------------------------------------
    def _default_run_id(self) -> str:
        digest = runtime.sha256_json(self.config.scientific_identity())[:10]
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        return f"{self.config.dataset}-{stamp}-{digest}-{uuid.uuid4().hex[:8]}"

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": TRAINER_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "run_id": self.run_id,
            "dataset": self.config.dataset,
            "scientific_identity": self.config.scientific_identity(),
            "scientific_hash": runtime.sha256_json(self.config.scientific_identity()),
            "manifest_id": self.manifest.get("manifest_id"),
            "manifest_hash": self._manifest_hash,
            "partition_hash": self._partition_hash,
            "resolved_protocol": self.manifest.get("resolved_protocol"),
            "source_hash": self.source["package"]["combined"],
            "git_commit": self.source["git"].get("commit"),
            "runtime_environment": self.source["environment"],
            "device_identity": runtime.device_identity(self.device),
        }

    @property
    def _manifest_hash(self) -> str:
        return runtime.sha256_json(self.manifest) if self.manifest else ""

    @property
    def _partition_hash(self) -> str:
        if not self.manifest:
            return ""
        payload = {
            "protocol": self.manifest.get("resolved_protocol"),
            "unit_ids": list(self.unit_ids),
            "splits": {
                str(record.get("study_id", record.get("unit_id", index))): record.get("split")
                for index, record in enumerate(self.manifest.get("records", []))
            },
        }
        return runtime.sha256_json(payload)

    def pseudo_label_version(self, epoch: int) -> str:
        return f"{self.run_id}:contract={CONTRACT_VERSION}:epoch={epoch}"

    # -- setup ----------------------------------------------------------
    def setup(self) -> None:
        if self._setup_done:
            return
        config = self.config
        runtime.seed_everything(config.seed)

        with self.timing.stage("data_discovery", root=config.data_root, dataset=config.dataset):
            self.manifest = self.components.discover_dataset(
                config.data_root,
                config.dataset,
                seed=config.seed,
                protocol=config.protocol,
                depth_axis=config.depth_axis,
            )
        if not isinstance(self.manifest, dict) or "records" not in self.manifest:
            raise TrainerContractError("discover_dataset must return a manifest mapping with records")
        unreadable = self.manifest.get("readiness", {}).get("skipped_unreadable", [])
        if unreadable:
            raise TrainerContractError(f"image inventory contains unreadable selected files: {unreadable[:3]}")

        with self.timing.stage("dataset_build", split="train", image_size=config.image_size):
            self.dataset = self.components.dataset_factory(
                self.manifest, split="train", image_size=config.image_size, seed=config.seed
            )
        self.unit_ids = list(self.dataset.unit_ids)
        if len(self.dataset) == 0:
            raise TrainerContractError("training split is empty; refusing to report a trained run")

        self.model_info: dict[str, Any] = {}
        with self.timing.stage("model_build"):
            self.models = self.components.make_models(
                seed=config.seed, width=config.width, feature_dim=config.feature_dim
            )
        built = self.models
        self.models = {}
        for name in ("producer", "student_no_audit", "student_audited"):
            if name not in built:
                raise TrainerContractError(f"make_models must return {name!r}")
            with self.timing.stage("model.to_device", model=name, device=str(self.device)):
                self.models[name] = built[name].to(self.device)
        # Anything else make_models reports (measured parameter counts, version)
        # is provenance, not a trainable component.
        self.model_info = {
            key: value for key, value in built.items() if key not in self.models
        }

        self.optimizers = {
            name: torch.optim.AdamW(
                self.models[name].parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
            for name in ("producer", "student_no_audit", "student_audited")
        }
        use_amp = bool(config.amp and self.device.type == "cuda")
        self.scaler = runtime.make_grad_scaler(self.device, enabled=use_amp)
        self.amp_enabled = use_amp

        with self.timing.stage("run_identity.write", path=str(self.paths.root)):
            self._write_run_identity()
        current_progress().event("setup.ready", units=len(self.dataset), batches=self.batches_per_epoch,
                                 protocol=self.manifest.get("resolved_protocol"),
                                 effective_data_workers=0, amp=use_amp)
        self._setup_done = True

    def _write_run_identity(self) -> None:
        """Bind this output directory to this experiment exactly once."""
        path = self.paths.root / "run_identity.json"
        identity = self.identity()
        if path.is_file():
            if not self.config.resume:
                raise ResumeIdentityError("occupied run directory requires explicit resume")
            stored = json.loads(path.read_text(encoding="utf-8"))
            mismatches = {
                key: {"stored": stored.get(key), "current": identity.get(key)}
                for key in ("run_id", "dataset", "scientific_hash", "manifest_hash", "partition_hash")
                if stored.get(key) != identity.get(key)
            }
            if mismatches:
                raise ResumeIdentityError(
                    f"output directory {self.paths.root} already belongs to another run: {mismatches}"
                )
            return
        runtime.atomic_write_json(path, identity)
        runtime.atomic_write_json(self.paths.root / "resolved_config.json", self.config.to_dict())
        runtime.atomic_write_json(self.paths.root / "source_provenance.json", self.source)
        runtime.atomic_write_json(self.paths.root / "data_manifest.json", self.manifest)
        runtime.atomic_write_json(
            self.paths.root / "supervision_ledger.json",
            {
                "manual_masks_read_by_training": False,
                "pretrained_weights": False,
                "gt_teacher": False,
                "reference_selected_checkpoint": False,
                "reference_reader": "self_audit_maskfree.evaluation.reference (separate process only)",
                "permitted_prior_information": [
                    "enumerated anatomical class definitions (BG/RV/MYO/LV)",
                    "image-only anatomical role rules in ontology.py",
                    "fixed provisional budgets recorded in the architecture contract",
                ],
                "contract_version": CONTRACT_VERSION,
            },
        )

    # -- schedule -------------------------------------------------------
    @property
    def batches_per_epoch(self) -> int:
        return max(1, math.ceil(len(self.dataset) / self.config.batch_size))

    @property
    def steps_per_epoch(self) -> int:
        return max(1, math.ceil(self.batches_per_epoch / self.config.accumulation_steps))

    @property
    def total_optimizer_steps(self) -> int:
        return self.steps_per_epoch * self.config.total_epochs

    @property
    def warmup_steps(self) -> int:
        return self.steps_per_epoch * self.config.warmup_epochs

    def current_lr(self) -> float:
        multiplier = runtime.warmup_cosine_multiplier(
            self.global_step, warmup_steps=self.warmup_steps, total_steps=self.total_optimizer_steps
        )
        return float(self.config.lr * multiplier)

    def _epoch_permutation(self, epoch: int) -> list[int]:
        generator = runtime.epoch_generator(self.config.seed, epoch)
        return torch.randperm(len(self.dataset), generator=generator).tolist()

    def _batches(self, permutation: Sequence[int]) -> list[list[int]]:
        size = self.config.batch_size
        return [list(permutation[i : i + size]) for i in range(0, len(permutation), size)]

    # -- main loop ------------------------------------------------------
    def run(self) -> dict[str, Any]:
        started = time.time()
        try:
            self.setup()
            if self.config.total_epochs == 150 and not self.config.is_bounded_run and self.device.type == "cuda":
                if not self.paths.gate_receipt.is_file():
                    raise TrainerContractError("150-epoch GPU run requires a completed matching preflight")
                gate = json.loads(self.paths.gate_receipt.read_text())
                for key, value in {
                    "status": "pass", "actual_physical_batch": self.config.batch_size,
                    "accumulation_steps": self.config.accumulation_steps,
                    "device": runtime.device_identity(self.device),
                    "scientific_hash": self.identity()["scientific_hash"],
                    "source_hash": self.identity()["source_hash"],
                    "manifest_hash": self.identity()["manifest_hash"],
                }.items():
                    if gate.get(key) != value:
                        raise TrainerContractError(f"preflight gate mismatch: {key}")
            if self.config.resume:
                self._load_checkpoint(self.config.resume)
            with self.timing.stage("metrics.wandb_initialize", mode=self.config.wandb_mode):
                self.sink = runtime.MetricSink(
                    project=self.config.wandb_project,
                    mode=self.config.wandb_mode,
                    run_id=self.run_id,
                    config=self.config.to_dict(),
                    directory=self.paths.root / "wandb",
                )
            current_progress().event("metrics.sink", **self.sink.summary())
            result = self._run_epochs(started)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            failure = {
                "status": STATUS_FAILED,
                "completed": False,
                "run_id": self.run_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "last_completed_epoch": self.last_completed_epoch,
                "epochs_completed": self.last_completed_epoch + 1,
                "elapsed_seconds": time.time() - started,
            }
            runtime.atomic_write_json(self.paths.failure_report, failure)
            if self.sink is not None:
                self.sink.finish()
            raise
        finally:
            if self.sink is not None:
                self.sink.finish()
        return result

    def _run_epochs(self, started: float) -> dict[str, Any]:
        config = self.config
        epoch_limit = config.max_epochs if config.max_epochs is not None else config.total_epochs
        stop_reason: str | None = None
        if config.is_bounded_run:
            stop_reason = "bounded_run"

        runtime.reset_peak_memory(self.device)
        # A saved epoch boundary can precede a interrupted validation job. Its
        # exact model state is still available here, before another update.
        if (config.epoch_validation and self.start_batch == 0 and
                self.last_completed_epoch >= 0 and self.start_epoch == self.last_completed_epoch + 1):
            self._observe_epoch(self.last_completed_epoch, resumed=True)
        for epoch in range(self.start_epoch, epoch_limit):
            if config.max_steps is not None and self.global_step >= config.max_steps:
                stop_reason = "max_steps"
                break
            epoch_started = time.monotonic()
            epoch_record, epoch_complete, permutation = self._train_epoch(epoch)
            self.history.append(epoch_record)
            runtime.append_jsonl(self.paths.epoch_metrics, epoch_record)
            current_progress().event("epoch.train_done", **epoch_record)
            if self.sink is not None and epoch_complete:
                self._log_epoch(epoch, epoch_record)
            if epoch_complete:
                self.last_completed_epoch = epoch
                self.start_batch = 0
                self._resume_position = (epoch + 1, 0, [])
                self._save_checkpoint(
                    epoch=epoch + 1,
                    batch_cursor=0,
                    permutation=[],
                    completed=False,
                    status=STATUS_PARTIAL,
                )
                if config.epoch_validation:
                    self._observe_epoch(epoch, epoch_started=epoch_started)
                else:
                    current_progress().event("epoch.summary", **epoch_record,
                                             total_elapsed_seconds=time.monotonic() - epoch_started)
            else:
                # An epoch stopped mid-way is NOT a completed epoch. The exact
                # sampler permutation and cursor are stored so the continuation
                # consumes the remaining units of THIS epoch, once.
                self._resume_position = (epoch, epoch_record["batch_cursor"], permutation)
                self._save_checkpoint(
                    epoch=epoch,
                    batch_cursor=epoch_record["batch_cursor"],
                    permutation=permutation,
                    completed=False,
                    status=STATUS_PARTIAL,
                )
                current_progress().event("epoch.summary", **epoch_record,
                                         total_elapsed_seconds=time.monotonic() - epoch_started)
            if config.max_steps is not None and self.global_step >= config.max_steps:
                stop_reason = "max_steps"
                break

        reached_final_epoch = self.last_completed_epoch == config.total_epochs - 1
        full_timeline = reached_final_epoch and not config.is_bounded_run

        finalization: dict[str, Any] = {
            "attempted": full_timeline,
            "available": False,
            "reason": None if full_timeline else "bounded or unfinished run: finalization not attempted",
        }
        if full_timeline:
            with self.timing.stage("finalization"):
                finalization = self._finalize()

        status = STATUS_COMPLETED if (full_timeline and config.total_epochs == 150 and finalization.get("available")) else STATUS_PARTIAL
        if full_timeline and not finalization.get("available"):
            status = STATUS_FAILED
            stop_reason = "finalization_failed"

        # Re-save under the final status WITHOUT moving the resume position: a
        # run stopped inside an epoch must keep its exact mid-epoch cursor.
        resume_epoch, resume_cursor, resume_permutation = self._resume_position
        self._save_checkpoint(
            epoch=resume_epoch,
            batch_cursor=resume_cursor,
            permutation=resume_permutation,
            completed=status == STATUS_COMPLETED,
            status=status,
        )

        from .epoch_validation import validation_status
        report = {
            "schema_version": TRAINER_SCHEMA_VERSION,
            "status": status,
            "completed": status == STATUS_COMPLETED,
            "runtime_completion": status == STATUS_COMPLETED,
            "full_150_complete": status == STATUS_COMPLETED,
            "hardware_qualified": status == STATUS_COMPLETED and self.device.type == "cuda",
            "scientific_outcome": finalization.get("scientific_outcome"),
            "stop_reason": stop_reason,
            "run_id": self.run_id,
            "run_root": str(self.paths.root),
            "identity": self.identity(),
            "epochs_completed": self.last_completed_epoch + 1,
            "last_completed_epoch": self.last_completed_epoch,
            "total_epochs": config.total_epochs,
            "bounded_run": config.is_bounded_run,
            "software_timeline": config.total_epochs != 150,
            "global_optimizer_steps": self.global_step,
            "component_steps": dict(self.component_steps),
            "micro_batches": self.micro_batches_seen,
            "batches_per_epoch": self.batches_per_epoch,
            "effective_data_workers": 0,
            "steps_per_epoch": self.steps_per_epoch,
            "device": runtime.device_identity(self.device),
            "gpu_stats": runtime.gpu_stats(self.device),
            "amp_enabled": self.amp_enabled,
            "model_info": self.model_info,
            "physical_batch": config.batch_size,
            "accumulation_steps": config.accumulation_steps,
            "effective_batch": config.effective_batch,
            "timing": self.timing.as_dict(),
            "elapsed_seconds": time.time() - started,
            "finalization": finalization,
            "epoch_validation": validation_status(self.paths.root, enabled=config.epoch_validation,
                                                    epochs=self.last_completed_epoch + 1),
            "wandb": self.sink.summary() if self.sink is not None else None,
            "epoch_metrics_path": str(self.paths.epoch_metrics),
            "history_tail": self.history[-3:],
        }
        runtime.atomic_write_json(self.paths.pipeline_report, report)
        return report

    def _observe_epoch(self, epoch: int, *, resumed: bool = False,
                       epoch_started: float | None = None) -> None:
        from .epoch_validation import observe_epoch

        # Report-only: reference values never enter self.history, which is
        # checkpointed, or any training/schedule/selection decision.
        with self.timing.stage("validation.epoch", epoch=epoch + 1, global_epoch=epoch):
            validation = observe_epoch(self, epoch)
        record = next((row for row in reversed(self.history)
                       if row.get("global_epoch") == epoch and row.get("epoch_complete")), None)
        if record is not None and not (resumed and validation.get("cached")):
            total_seconds = (time.monotonic() - epoch_started if epoch_started is not None else
                             record["epoch_seconds"] + validation.get("elapsed_seconds", 0.0))
            current_progress().event("epoch.summary", **record, epoch=epoch + 1,
                                     total_epochs=self.config.total_epochs, validation=validation,
                                     total_elapsed_seconds=total_seconds)

    def _log_epoch(self, epoch: int, record: dict[str, Any]) -> None:
        assert self.sink is not None
        self.sink.log("producer", {key.removeprefix("producer/"): value for key, value in record["producer"].items()}, step=epoch, commit=False)
        self.sink.log("audit", record["audit"], step=epoch, commit=False)
        self.sink.log("student", record["students"], step=epoch)

    # -- one epoch ------------------------------------------------------
    def _train_epoch(self, epoch: int) -> tuple[dict[str, Any], bool, list[int]]:
        config = self.config
        epoch_start = time.time()

        if self.resumed_permutation is not None and epoch == self.start_epoch:
            permutation = list(self.resumed_permutation)
            self.resumed_permutation = None
        else:
            permutation = self._epoch_permutation(epoch)
        batches = self._batches(permutation)

        accumulator = EpochAccumulator()
        audit_counters = {
            "units": 0,
            "accepted_edits": 0,
            "no_change": 0,
            "semantic_unresolved": 0,
            "search_inconclusive": 0,
            "candidates": 0,
            "distinct_candidates": 0,
            "fitting_steps": 0,
            "prior_violations": 0,
            "score_unavailable": 0,
        }
        class_pixels: dict[Any, int] = {index: 0 for index in range(4)}
        class_pixels["total"] = 0
        ramp = runtime.label_ramp(epoch, ramp_epochs=LABEL_RAMP_EPOCHS)
        lineage_path = self.paths.reports / "label_lineage.jsonl"

        start_batch = self.start_batch if epoch == self.start_epoch else 0
        if start_batch:
            if self._partial_epoch is None or self._partial_epoch["epoch"] != epoch:
                raise ResumeIdentityError("mid-epoch resume lacks accumulator state")
            accumulator = EpochAccumulator(**self._partial_epoch["accumulator"])
            audit_counters = dict(self._partial_epoch["audit_counters"])
            class_pixels = dict(self._partial_epoch["class_pixels"])
        batch_cursor = start_batch
        progress = current_progress()
        progress.update(epoch=epoch + 1, epochs=config.total_epochs, batch=start_batch,
                        batches=len(batches), operation="training", unit_id=None)
        progress.event("epoch.start", resumed_from_batch=start_batch, label_ramp=ramp)
        for batch_index in range(start_batch, len(batches)):
            batch_cursor = batch_index
            indices = batches[batch_index]
            progress.update(batch=batch_index + 1, physical_batch=len(indices),
                            accumulation=config.accumulation_steps, unit_id=None,
                            operation="load training batch")
            batch_started = time.monotonic()
            prior_timing = dict(self.timing.seconds)
            is_group_end = (
                (batch_index + 1) % config.accumulation_steps == 0
                or batch_index == len(batches) - 1
            )
            self._train_batch(
                epoch=epoch,
                batch_index=batch_index,
                indices=indices,
                ramp=ramp,
                accumulator=accumulator,
                audit_counters=audit_counters,
                class_pixels=class_pixels,
                lineage_path=lineage_path,
                is_group_end=is_group_end,
                group_weight=len(indices) / sum(len(b) for b in batches[
                    (batch_index // config.accumulation_steps) * config.accumulation_steps:
                    (batch_index // config.accumulation_steps + 1) * config.accumulation_steps]),
            )
            self.micro_batches_seen += 1
            batch_cursor = batch_index + 1
            if progress.enabled:
                progress.update(operation="batch completed", unit_id=None)
                progress.dashboard(
                    force=batch_cursor == len(batches) or (
                        config.max_steps is not None and self.global_step >= config.max_steps),
                    metric_scope="running_epoch_means", lr=self.current_lr(), label_ramp=ramp,
                    global_optimizer_steps=self.global_step, component_steps=dict(self.component_steps),
                    metrics={key: accumulator.mean(key) for key in sorted(accumulator.values)},
                    metric_counts=dict(accumulator.counts), audit=dict(audit_counters),
                    labelled_pixels=class_pixels["total"],
                    class_occupancy={str(k): class_pixels[k] / class_pixels["total"]
                                     if class_pixels["total"] else None for k in range(4)},
                    batch_seconds=time.monotonic() - batch_started,
                    batch_stage_seconds={k: v - prior_timing.get(k, 0.0)
                                         for k, v in self.timing.seconds.items()},
                    timing_note="inclusive stages; do not sum", gpu_stats=runtime.gpu_stats(self.device),
                    verification_status="locked_until_all_predictions_frozen",
                )
            self._partial_epoch = {
                "epoch": epoch,
                "accumulator": {"values": accumulator.values, "counts": accumulator.counts},
                "audit_counters": audit_counters, "class_pixels": class_pixels,
            }
            if is_group_end and self.global_step % 50 == 0:
                self._save_checkpoint(epoch=epoch, batch_cursor=batch_cursor,
                                      permutation=permutation, completed=False, status=STATUS_PARTIAL)
            if config.max_steps is not None and self.global_step >= config.max_steps:
                break

        units = max(audit_counters["units"], 1)
        record = {
            "schema_version": TRAINER_SCHEMA_VERSION,
            "global_epoch": epoch,
            "dataset": config.dataset,
            "protocol": self.manifest.get("resolved_protocol"),
            "contract_version": CONTRACT_VERSION,
            "pseudo_label_version": self.pseudo_label_version(epoch),
            "label_ramp": ramp,
            "lr": self.current_lr(),
            "global_optimizer_steps": self.global_step,
            "component_steps": dict(self.component_steps),
            "batch_cursor": batch_cursor,
            "epoch_complete": batch_cursor >= len(batches),
            "permutation_hash": runtime.sha256_json(permutation),
            "resumed_from_batch": start_batch,
            "batches": len(batches),
            "units_visited": audit_counters["units"],
            "units_available": len(self.dataset),
            "producer": {
                key: accumulator.mean(key)
                for key in sorted(accumulator.values)
                if key.startswith("producer/")
            },
            "students": {
                key: accumulator.mean(key)
                for key in sorted(accumulator.values)
                if key.startswith("student_")
            },
            "audit": {
                **audit_counters,
                "accepted_edit_rate": audit_counters["accepted_edits"] / units,
                "no_change_rate": audit_counters["no_change"] / units,
                "semantic_unresolved_rate": audit_counters["semantic_unresolved"] / units,
                "search_inconclusive_rate": audit_counters["search_inconclusive"] / units,
                "mean_candidates": audit_counters["candidates"] / units,
                "mean_fitting_steps": audit_counters["fitting_steps"] / units,
                "evidence_improvement_mean": accumulator.mean("audit/evidence_improvement"),
                "select_nll_mean": accumulator.mean("audit/select_nll"),
                "fit_nll_mean": accumulator.mean("audit/fit_nll"),
            },
            "coverage": {
                "student_no_audit": accumulator.mean("coverage/student_no_audit"),
                "student_audited": accumulator.mean("coverage/student_audited"),
                "ignored_fraction_audited": accumulator.mean("coverage/ignored_audited"),
                "labelled_pixels": class_pixels["total"],
                "class_occupancy": {
                    str(index): (class_pixels[index] / class_pixels["total"])
                    if class_pixels["total"]
                    else None
                    for index in range(4)
                },
                "class_disappearance_warning": [
                    str(index) for index in range(4) if class_pixels[index] == 0
                ],
            },
            "detail": accumulator.as_dict(),
            "gpu_stats": runtime.gpu_stats(self.device),
            "timing": self.timing.as_dict(),
            "epoch_seconds": time.time() - epoch_start,
        }
        return record, batch_cursor >= len(batches), permutation

    # -- one batch ------------------------------------------------------
    def _train_batch(
        self,
        *,
        epoch: int,
        batch_index: int,
        indices: Sequence[int],
        ramp: float,
        accumulator: EpochAccumulator,
        audit_counters: dict[str, int],
        class_pixels: dict[Any, int],
        lineage_path: Path,
        is_group_end: bool,
        group_weight: float = 1.0,
    ) -> None:
        config = self.config
        with self.timing.stage("data_load", indices=list(indices), batch_units=len(indices)):
            units: list[TrainingUnit] = [self.dataset[index] for index in indices]
        for unit in units:
            unit.fitting.validate()
            unit.selection.validate()

        with self.timing.stage("batch.to_device", device=str(self.device), batch_units=len(units)):
            context = torch.stack([unit.fitting.context for unit in units]).to(self.device).float()
            fit_support = torch.stack([unit.fitting.support for unit in units]).to(self.device)

        producer = self.models["producer"]
        producer.train()
        with self.timing.stage("producer_step"):
            with torch.autocast(self.device.type, enabled=self.amp_enabled):
                loss, producer_metrics = self.components.producer_loss(
                    producer, context, fit_support, generator=self.sampling_generator
                )
            if not torch.isfinite(loss):
                raise TrainerContractError("nonfinite producer loss")
            self.scaler.scale(loss * group_weight).backward()
            self._active_arms.add("producer")
        for key, value in (producer_metrics or {}).items():
            if key == "loss":  # loss itself is recorded once below
                continue
            number = _scalar(value)
            if number is not None:
                accumulator.add(f"producer/{key}", number)
        accumulator.add("producer/loss", float(loss.detach().cpu().item()))

        # Bank generation consumes DETACHED producer features: no student or
        # auditor gradient ever reaches the representation producer.
        with self.timing.stage("feature_forward"):
            producer.eval()
            with torch.no_grad():
                features = producer(context)["features"].detach()
            producer.train()

        probs_initial: list[torch.Tensor] = []
        probs_audited: list[torch.Tensor] = []
        validity_initial: list[torch.Tensor] = []
        validity_audited: list[torch.Tensor] = []

        for offset, unit in enumerate(units):
            current_progress().update(unit_id=unit.fitting.unit_id, unit_in_batch=offset + 1,
                                      operation="generate hypotheses and audit select evidence")
            audit = self._audit_unit(
                epoch=epoch,
                unit=unit,
                features=features[offset],
                audit_counters=audit_counters,
                accumulator=accumulator,
                lineage_path=lineage_path,
            )
            initial, selected = audit.initial, audit.selected
            probs_initial.append(initial.probabilities.to(self.device).float())
            probs_audited.append(selected.probabilities.to(self.device).float())
            validity_initial.append(initial.validity.to(self.device).float())
            validity_audited.append(audit.validity.to(self.device).float())
            labels = selected.labels.detach().cpu()
            valid_pixels = audit.validity.detach().cpu() > 0
            class_pixels["total"] += int(valid_pixels.sum())
            for index in range(4):
                class_pixels[index] += int(((labels == index) & valid_pixels).sum().item())

        targets = {
            "student_no_audit": (
                torch.stack(probs_initial).detach(),
                torch.stack(validity_initial).detach(),
            ),
            "student_audited": (
                torch.stack(probs_audited).detach(),
                torch.stack(validity_audited).detach(),
            ),
        }
        for arm, (probabilities, validity) in targets.items():
            current_progress().update(operation=arm, unit_id=None, unit_in_batch=None)
            model = self.models[arm]
            model.train()
            with self.timing.stage(f"{arm}_step"):
                with torch.autocast(self.device.type, enabled=self.amp_enabled):
                    logits = model(context)
                    student_value, student_metrics = self.components.student_loss(
                        logits, probabilities, validity
                    )
                if not torch.isfinite(student_value):
                    raise TrainerContractError(f"nonfinite {arm} loss")
                if bool((validity > 0).any()) and ramp > 0:
                    # Accumulate weighted numerators under a bounded scale;
                    # normalize by the group's actual pseudo-label support
                    # after AMP unscale. Invalid microbatches contribute zero.
                    weight = group_weight * float(validity.mean().detach().cpu())
                    self.scaler.scale(student_value * ramp * weight).backward()
                    self._student_weight_sums[arm] = self._student_weight_sums.get(arm, 0.0) + weight
                    self._active_arms.add(arm)
            accumulator.add(f"{arm}/loss", float(student_value.detach().cpu().item()))
            for key, value in (student_metrics or {}).items():
                if key == "loss":  # do not double the observation count
                    continue
                number = _scalar(value)
                if number is not None:
                    accumulator.add(f"{arm}/{key}", number)
            if (student_metrics or {}).get("skipped"):
                accumulator.add(f"{arm}/skipped_batches", 1.0)
            coverage_key = "coverage/student_no_audit" if arm == "student_no_audit" else "coverage/student_audited"
            accumulator.add(coverage_key, float((validity > 0).float().mean().cpu().item()))
        accumulator.add(
            "coverage/ignored_audited",
            float((targets["student_audited"][1] <= 0).float().mean().cpu().item()),
        )

        if is_group_end:
            with self.timing.stage("optimizers.step", active_arms=sorted(self._active_arms)):
                self._optimizer_step(accumulator)

    def _optimizer_step(self, accumulator: EpochAccumulator) -> None:
        lr = self.current_lr()
        # Validate every active component before mutating any parameters.
        for name, optimizer in self.optimizers.items():
            if name not in self._active_arms:
                accumulator.add(f"{name}/skipped_optimizer_groups", 1.0)
                continue
            runtime.set_lr(optimizer, lr)
            self.scaler.unscale_(optimizer)
            if name.startswith("student_"):
                divisor = self._student_weight_sums[name]
                for parameter in self.models[name].parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(divisor)
            norm = runtime.global_grad_norm(self.models[name].parameters())
            if not math.isfinite(norm):
                raise TrainerContractError(f"nonfinite gradient in {name}; no component stepped")
            accumulator.add(f"{name}/grad_norm", norm)
        for name, optimizer in self.optimizers.items():
            if name not in self._active_arms:
                continue
            self.scaler.step(optimizer)
            self.component_steps[name] += 1
        self.scaler.update()
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        self._active_arms.clear()
        self._student_weight_sums.clear()
        accumulator.add("schedule/lr", lr)

    # -- per-unit audit --------------------------------------------------
    def _audit_unit(
        self,
        *,
        epoch: int,
        unit: TrainingUnit,
        features: torch.Tensor,
        audit_counters: dict[str, int],
        accumulator: EpochAccumulator,
        lineage_path: Path,
    ) -> AuditResult:
        seed = _stable_unit_seed(self.config.seed, epoch, unit.fitting.unit_id)
        with self.timing.stage("bank_generation", unit_id=unit.fitting.unit_id):
            bank = self.components.generate_bank(unit.fitting, features=features.detach().cpu(), seed=seed)
        if not bank:
            raise TrainerContractError(
                f"generate_bank returned no candidate for unit {unit.fitting.unit_id}"
            )
        with self.timing.stage("audit", unit_id=unit.fitting.unit_id, candidates=len(bank), rounds=AUDIT_ROUNDS):
            audit = self.components.audit_bank(
                bank,
                unit.fitting,
                unit.selection,
                self.observation,
                rounds=AUDIT_ROUNDS,
                improvement_threshold=IMPROVEMENT_THRESHOLD,
            )

        trace = audit.trace or {}
        audit_counters["units"] += 1
        audit_counters["candidates"] += int(trace.get("candidate_count", len(audit.bank)))
        audit_counters["distinct_candidates"] += len({
            runtime.sha256_bytes(h.labels.detach().cpu().numpy().tobytes()) for h in audit.bank})
        audit_counters["fitting_steps"] += sum(f.fitting_steps for f in audit.fitted)
        audit_counters["prior_violations"] += sum(
            int(bool(value)) for h in audit.bank
            for value in h.metadata.get("ontology_trace", {}).get("prior_violations", {}).values())
        accepted = bool(trace.get("accepted", audit.selected.candidate_id != audit.initial.candidate_id))
        audit_counters["accepted_edits"] += int(accepted)
        audit_counters["no_change"] += int(not accepted)
        audit_counters["semantic_unresolved"] += int(bool(audit.selected.semantic_unresolved))
        audit_counters["search_inconclusive"] += int(bool(trace.get("search_inconclusive", False)))

        improvement = _scalar(trace.get("improvement_nats_per_pixel"))
        if improvement is not None:
            accumulator.add("audit/evidence_improvement", improvement)
        for score in list(audit.scores or []) + [f.fit_score for f in audit.fitted]:
            if not getattr(score, "available", True):
                audit_counters["score_unavailable"] += 1
                continue
            value = _scalar(getattr(score, "normalized_nll", None))
            if value is None:
                continue
            key = "audit/select_nll" if getattr(score, "role", "") == "select" else "audit/fit_nll"
            accumulator.add(key, value, count=score.count)

        runtime.append_jsonl(
            lineage_path,
            {
                "global_epoch": epoch,
                "unit_id": unit.fitting.unit_id,
                "study_id": unit.fitting.study_id,
                "partition_id": unit.fitting.partition_id,
                "pseudo_label_version": self.pseudo_label_version(epoch),
                "initial_candidate_id": audit.initial.candidate_id,
                "selected_candidate_id": audit.selected.candidate_id,
                "accepted_edit": accepted,
                "semantic_unresolved": bool(audit.selected.semantic_unresolved),
                "bank_candidate_ids": [candidate.candidate_id for candidate in audit.bank],
                "valid_fraction": float((audit.validity > 0).float().mean().item()),
            },
        )
        return audit

    # -- checkpointing ---------------------------------------------------
    def _save_checkpoint(
        self,
        *,
        epoch: int,
        batch_cursor: int,
        permutation: Sequence[int],
        completed: bool,
        status: str,
    ) -> Path:
        if completed and self.config.is_bounded_run:
            raise TrainerContractError(
                "a bounded run (max_steps/max_epochs) can never be marked completed"
            )
        if completed and self.last_completed_epoch != self.config.total_epochs - 1:
            raise TrainerContractError(
                "completion requires last_completed_epoch == total_epochs - 1"
            )
        # Checkpoints are only ever taken where no accumulation group is open:
        # both the max_steps stop and the end of an epoch land immediately after
        # an optimizer step. Resume can therefore be called exact without
        # serialising partial gradients.
        pending = [
            f"{model_name}.{parameter_name}"
            for model_name, model in self.models.items()
            for parameter_name, parameter in model.named_parameters()
            if parameter.grad is not None
        ]
        if pending:
            raise TrainerContractError(
                "refusing to write a checkpoint with an open accumulation group; "
                f"{len(pending)} parameters still hold gradients (first: {pending[0]})"
            )
        payload = {
            "schema_version": TRAINER_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "identity": self.identity(),
            "config": self.config.to_dict(),
            "status": status,
            "completed": bool(completed),
            "bounded_run": self.config.is_bounded_run,
            "epoch": int(epoch),
            "batch_cursor": int(batch_cursor),
            "accumulation_boundary": True,
            "pending_accumulation_microbatches": 0,
            "epoch_permutation": list(permutation),
            "last_completed_epoch": int(self.last_completed_epoch),
            "epochs_completed": int(self.last_completed_epoch + 1),
            "global_step": int(self.global_step),
            "component_steps": dict(self.component_steps),
            "micro_batches_seen": int(self.micro_batches_seen),
            "pseudo_label_version": self.pseudo_label_version(max(self.last_completed_epoch, 0)),
            "models": {name: model.state_dict() for name, model in self.models.items()},
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "rng": runtime.capture_rng_state(),
            "sampling_generator": self.sampling_generator.get_state(),
            "history": self.history,
            "partial_epoch": self._partial_epoch if batch_cursor else None,
            "log_offsets": {
                str(p.relative_to(self.paths.root)): p.stat().st_size
                for p in (self.paths.epoch_metrics, self.paths.reports / "label_lineage.jsonl") if p.exists()
            },
            "timing": self.timing.as_dict(),
        }
        runtime.atomic_save_torch(self.paths.last_checkpoint, payload)
        # Component final files are written once before freeze in _finalize.
        return self.paths.last_checkpoint

    def _load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise ResumeIdentityError(f"resume checkpoint not found: {path}")
        with self.timing.stage("checkpoint.load", path=str(path), device=str(self.device)):
            payload = torch.load(path, map_location=self.device, weights_only=False)

        diff = compare_configs(payload.get("config", {}), self.config)
        if diff["scientific"]:
            raise ResumeIdentityError(
                f"resume refused: scientific configuration mismatch {diff['scientific']}"
            )
        stored_identity = payload.get("identity", {})
        current_identity = self.identity()
        for key in ("dataset", "manifest_hash", "partition_hash", "scientific_hash", "source_hash"):
            if stored_identity.get(key) != current_identity.get(key):
                raise ResumeIdentityError(
                    f"resume refused: {key} mismatch "
                    f"(checkpoint={stored_identity.get(key)!r}, current={current_identity.get(key)!r})"
                )
        for key in ("run_id", "runtime_environment", "device_identity"):
            if stored_identity.get(key) != current_identity.get(key):
                raise ResumeIdentityError(f"exact resume refused: {key} mismatch")
        old_root = Path(payload["config"]["output_dir"]).resolve()
        if old_root != Path(self.config.output_dir).resolve():
            raise ResumeIdentityError("exact resume requires the same output workspace")
        if payload.get("completed"):
            raise ResumeIdentityError("refusing to resume a run already marked completed")

        for name, model in self.models.items():
            with self.timing.stage("checkpoint.restore_model", model=name):
                model.load_state_dict(payload["models"][name])
        for name, optimizer in self.optimizers.items():
            with self.timing.stage("checkpoint.restore_optimizer", model=name):
                optimizer.load_state_dict(payload["optimizers"][name])
        if payload.get("scaler") is not None and self.scaler is not None:
            self.scaler.load_state_dict(payload["scaler"])
        runtime.restore_rng_state(payload["rng"])
        self.sampling_generator.set_state(payload["sampling_generator"].cpu().to(torch.uint8))

        self.global_step = int(payload["global_step"])
        self.component_steps = dict(payload["component_steps"])
        self.micro_batches_seen = int(payload.get("micro_batches_seen", 0))
        self.last_completed_epoch = int(payload["last_completed_epoch"])
        stored_history = list(payload.get("history", []))
        self.start_epoch = int(payload["epoch"])
        # A partially executed epoch is re-recorded in full by the continuation,
        # so its truncated record is dropped here instead of double counting.
        self.history = [
            record for record in stored_history
            if int(record.get("global_epoch", -1)) < self.start_epoch
        ]
        self._dropped_partial_records = len(stored_history) - len(self.history)
        self.start_batch = int(payload["batch_cursor"])
        self._partial_epoch = payload.get("partial_epoch")
        saved_timing = payload.get("timing", {})
        self.timing = runtime.TimingAccumulator(
            seconds=dict(saved_timing.get("seconds", {})), calls=dict(saved_timing.get("calls", {})),
            device=self.device)
        # Discard log writes beyond the committed checkpoint transaction.
        for relative, offset in payload.get("log_offsets", {}).items():
            log_path = self.paths.root / relative
            if log_path.exists():
                with log_path.open("r+b") as handle:
                    handle.truncate(offset)
        runtime.atomic_write_text(self.paths.epoch_metrics, "".join(
            json.dumps(row, default=runtime.json_default) + "\n" for row in self.history))
        self._resume_position = (self.start_epoch, self.start_batch,
                                 list(payload.get("epoch_permutation", [])))
        stored_permutation = list(payload.get("epoch_permutation", []))
        self.resumed_permutation = stored_permutation or None
        if stored_permutation:
            expected = self._epoch_permutation(self.start_epoch)
            if expected != stored_permutation:
                # The stored permutation wins: it is what the interrupted epoch
                # actually consumed. The divergence is recorded, not hidden.
                runtime.append_jsonl(
                    self.paths.reports / "resume_notes.jsonl",
                    {
                        "epoch": self.start_epoch,
                        "note": "stored sampler permutation differs from recomputed permutation",
                    },
                )
        runtime.append_jsonl(
            self.paths.reports / "resume_notes.jsonl",
            {
                "resumed_from": str(path),
                "epoch": self.start_epoch,
                "batch_cursor": self.start_batch,
                "global_step": self.global_step,
                "dropped_partial_epoch_records": self._dropped_partial_records,
                "operational_differences": diff["operational"],
            },
        )

    # -- finalization ----------------------------------------------------
    def _annotation_repeat(self, unit: TrainingUnit, audit: AuditResult) -> dict[str, Any]:
        """Preregistered fit-only noise repeat; diagnostic labels never feed training."""
        seed = _stable_unit_seed(self.config.seed, self.config.total_epochs - 1, unit.fitting.unit_id)
        generator = torch.Generator().manual_seed(seed + 73)
        noise = torch.randn(unit.fitting.context.shape, generator=generator) * 0.05
        context = unit.fitting.context + noise * unit.fitting.support.unsqueeze(0)
        fitting = replace(unit.fitting, context=context, image=context[1:2].clone())
        with torch.no_grad():
            features = self.models["producer"](context.unsqueeze(0).to(self.device))["features"][0].cpu()
        repeated = self.components.audit_bank(
            self.components.generate_bank(fitting, features=features, seed=seed),
            fitting, unit.selection, self.observation,
            rounds=AUDIT_ROUNDS, improvement_threshold=IMPROVEMENT_THRESHOLD)

        def compare(left, right, left_valid, right_valid):
            common = (left_valid > 0) & (right_valid > 0)
            count = int(common.sum())
            return {"available": True, "count": left.numel(),
                    "agreement": float((left == right).float().mean()),
                    "common_valid_count": count,
                    "valid_agreement": float((left[common] == right[common]).float().mean()) if count else None}

        metrics = {
            "E1_cuts_inspired_control": compare(audit.initial.labels, repeated.initial.labels,
                audit.initial.validity, repeated.initial.validity),
            "E5_evidence_plus_challenge": compare(audit.selected.labels, repeated.selected.labels,
                audit.validity, repeated.validity)}
        student_labels = {}
        with torch.no_grad():
            for arm in ("student_no_audit", "student_audited"):
                original = self.models[arm](unit.fitting.context.unsqueeze(0).to(self.device))[0].argmax(0).cpu()
                changed = self.models[arm](context.unsqueeze(0).to(self.device))[0].argmax(0).cpu()
                valid = audit.initial.validity if arm == "student_no_audit" else audit.validity
                metrics[arm] = compare(original, changed, valid, valid)
                student_labels[arm] = changed
        return {"metrics": {"available": True, "protocol": "fit_noise_sigma_0.05",
                    "scope": "pre_freeze_image_only", "seed": seed + 73, "methods": metrics,
                    "extra_cost": {"producer_forwards": 1, "student_forwards": 4,
                        "candidate_fits": len(repeated.fitted),
                        "fitter_iterations": sum(f.fitting_steps for f in repeated.fitted)}},
                "audit": repeated, "student_labels": student_labels}

    def _finalize(self) -> dict[str, Any]:
        """Stream frozen fits and predictions to disk before opening verification."""
        from .experiments import ForeignEvidenceCache
        config = self.config
        progress = current_progress()
        progress.update(operation="finalization: prepare frozen checkpoints", unit_id=None,
                        unit_in_batch=None, batch=None, epoch=config.total_epochs)
        try:
            if runtime.package_source_hash(refresh=True)["combined"] != self.source["package"]["combined"]:
                raise TrainerContractError("package sources changed during run; finalization refused")
            final_datasets = {}
            for split in ("train", "dev", "test"):
                dataset = self.components.dataset_factory(
                    self.manifest, split=split, image_size=config.image_size, seed=config.seed)
                if len(dataset):
                    final_datasets[split] = dataset
            if not final_datasets:
                raise TrainerContractError("no finalization units")
            for model in self.models.values():
                model.eval()
            primary_frozen = self.paths.freeze_manifest.exists()
            checkpoint_paths = {}
            for name, model in self.models.items():
                path = self.paths.checkpoints / f"{name}_final.pt"
                if not primary_frozen:
                    runtime.atomic_save_torch(path, {
                        "identity": self.identity(), "state_dict": model.state_dict(),
                        "epoch": config.total_epochs - 1, "component_steps": self.component_steps})
                checkpoint_paths[name] = str(path)
            version = self.pseudo_label_version(config.total_epochs - 1)
            state_paths = {}
            if not primary_frozen:
                foreign_cache = ForeignEvidenceCache(capacity=max(32, len(self.manifest["records"])),
                    dataset=config.dataset, run_id=self.manifest["manifest_id"])
                seen_studies = set()
                # First pass: audit each unit, persist it, seed genuine foreign evidence.
                progress.event("finalization.pass", operation="audit and repeat annotation")
                for split, dataset in final_datasets.items():
                    for index in range(len(dataset)):
                        progress.update(operation="final audit and repeat annotation", split=split,
                                        item=index + 1, items=len(dataset), unit_id=dataset.unit_ids[index])
                        unit = dataset[index]
                        with torch.no_grad():
                            context = unit.fitting.context.unsqueeze(0).to(self.device).float()
                            features = self.models["producer"](context)["features"][0].detach().cpu()
                        with self.timing.stage("finalize.audit", unit_id=unit.fitting.unit_id):
                            audit = self.components.audit_bank(
                                self.components.generate_bank(unit.fitting, features=features,
                                    seed=_stable_unit_seed(config.seed, config.total_epochs - 1, unit.fitting.unit_id)),
                                unit.fitting, unit.selection, self.observation,
                                rounds=AUDIT_ROUNDS, improvement_threshold=IMPROVEMENT_THRESHOLD)
                        token = runtime.sha256_bytes(unit.fitting.unit_id.encode())[:24]
                        path = self.paths.checkpoints / "fitted_states" / f"{token}.pt"
                        with self.timing.stage("annotation_repeat"):
                            repeated_annotation = self._annotation_repeat(unit, audit)
                        runtime.atomic_save_torch(path, {"audit": audit, "repeated_annotation": repeated_annotation})
                        state_paths[unit.fitting.unit_id] = path
                        if unit.fitting.study_id not in seen_studies:
                            foreign_cache.push(unit.fitting.study_id, unit.fitting.unit_id, {
                                fit.hypothesis.candidate_id: score.total
                                for fit, score in zip(audit.fitted, audit.scores)
                                if score.available and score.total is not None},
                                candidate_slots={h.candidate_id: i for i, h in enumerate(audit.bank)})
                            seen_studies.add(unit.fitting.study_id)
                entries, experiment_rows, coverage_rows = [], [], []
                # Second pass: all compared selectors, controls, and both students.
                progress.event("finalization.pass", operation="common-bank controls, students, native exports")
                for split, dataset in final_datasets.items():
                    for index in range(len(dataset)):
                        progress.update(operation="selectors, controls and primary exports", split=split,
                                        item=index + 1, items=len(dataset), unit_id=dataset.unit_ids[index])
                        unit = dataset[index]
                        path = state_paths[unit.fitting.unit_id]
                        with self.timing.stage("fitted_state.load", path=str(path)):
                            cached = torch.load(path, map_location="cpu", weights_only=False)
                        audit = cached["audit"]
                        unit.fitting.metadata.update(epoch=config.total_epochs - 1,
                            checkpoint=checkpoint_paths["producer"])
                        with self.timing.stage("finalize.common_bank"):
                            experiments = self.components.run_bank_experiments(
                                audit, unit.fitting, unit.selection, self.observation,
                                seed=config.seed, foreign_cache=foreign_cache)
                        predictions = dict(experiments["predictions"])
                        fitted_predictions = dict(experiments["fitted_predictions"])
                        student_predictions = {}
                        with torch.no_grad():
                            context = unit.fitting.context.unsqueeze(0).to(self.device).float()
                            for arm in ("student_no_audit", "student_audited"):
                                probabilities = torch.softmax(self.models[arm](context).float(), dim=1)[0].cpu()
                                source = audit.initial if arm == "student_no_audit" else audit.selected
                                validity = audit.initial.validity if arm == "student_no_audit" else audit.validity
                                hypothesis = Hypothesis(
                                    candidate_id=f"{arm}:{unit.fitting.unit_id}",
                                    labels=probabilities.argmax(0).long(), probabilities=probabilities,
                                    validity=validity.detach().cpu(), source=arm,
                                    semantic_unresolved=source.semantic_unresolved,
                                    alternatives=source.alternatives, metadata={"view": "fit_view"})
                                student_predictions[arm] = hypothesis
                                predictions[arm] = hypothesis
                                fitted_predictions[arm] = self.observation.fit(hypothesis, unit.fitting)
                        missing = set(predictions) - set(fitted_predictions)
                        if missing:
                            raise TrainerContractError(f"predictions lack frozen fits: {sorted(missing)}")
                        coverage = {"unit_id": unit.fitting.unit_id, "patient_id": unit.record["patient_id"],
                                    "split": split, "pixels": int(audit.validity.numel()),
                                    "semantic_unresolved": bool(audit.selected.semantic_unresolved),
                                    "draft_initial_valid_foreground": int(((audit.initial.validity > 0) & (audit.initial.labels > 0)).sum()),
                                    "draft_audited_valid_foreground": int(((audit.validity > 0) & (audit.selected.labels > 0)).sum()),
                                    "audited_class_occupancy": {
                                        str(role): int(((audit.validity > 0) & (audit.selected.labels == role)).sum())
                                        for role in range(4)},
                                    "audit_trace": audit.trace}
                        for arm, hypothesis in student_predictions.items():
                            coverage[f"natural_support_{arm}"] = int((hypothesis.validity > 0).sum())
                            coverage[f"valid_foreground_{arm}"] = int(((hypothesis.validity > 0) & (hypothesis.labels > 0)).sum())
                        matched = self.components.matched_coverage(
                            student_predictions["student_no_audit"].validity,
                            student_predictions["student_audited"].validity)
                        coverage["matched_support"] = int(torch.as_tensor(matched[0]).sum())
                        coverage_rows.append(coverage)
                        unique_fits = {fit.hypothesis.candidate_id: fit for fit in fitted_predictions.values()}
                        select_scores = {cid: self.observation.score(fit, unit.selection).total
                                         for cid, fit in unique_fits.items()}
                        state = {"audit": audit, "repeated_annotation": cached["repeated_annotation"],
                                 "fitted": list(unique_fits.values()),
                                 "control_fitted": experiments.get("control_fitted", {}),
                                 "selection_scores": select_scores,
                                 "method_to_candidate": {name: fit.hypothesis.candidate_id for name, fit in fitted_predictions.items()},
                                 "unit_id": unit.fitting.unit_id, "partition_id": unit.fitting.partition_id,
                                 "patient_id": unit.record["patient_id"], "split": split}
                        runtime.atomic_save_torch(path, state)
                        checkpoint_paths[f"fit_state_{path.stem}"] = str(path)
                        for row in experiments["rows"]:
                            experiment_rows.append({**row, "unit_id": unit.fitting.unit_id, "split": split})
                        for name, hypothesis in predictions.items():
                            with self.timing.stage("export.native_prediction", method=name, path=str(self.paths.exports)):
                                entries.append(self.components.export_prediction(
                                    self.paths.exports, record=unit.record, prediction_name=name,
                                    labels=hypothesis.labels, probabilities=hypothesis.probabilities,
                                    validity=hypothesis.validity, alternatives=hypothesis.alternatives,
                                    checkpoint_id=checkpoint_paths.get(name, checkpoint_paths["producer"]), version=version))
                runtime.atomic_write_json(self.paths.reports / "experiment_rows.json", {
                    "rows": experiment_rows, "coverage": coverage_rows,
                    "splits": sorted(final_datasets),
                    "units_per_split": {k: len(v) for k, v in final_datasets.items()}})
                with self.timing.stage("freeze.primary_assemble_hash", predictions=len(entries)):
                    freeze_manifest = self.components.freeze_predictions(
                        self.paths.exports, entries, checkpoint_paths, dataset=config.dataset,
                        protocol=self.manifest["resolved_protocol"], epoch=config.total_epochs - 1,
                        required_unit_ids=sorted(state_paths),
                        nuisance_files={path.stem: str(path) for path in state_paths.values()})
                runtime.atomic_write_json(self.paths.freeze_manifest, freeze_manifest)
                self.components.validate_freeze(freeze_manifest)
            else:
                freeze_manifest = json.loads(self.paths.freeze_manifest.read_text())
                self.components.validate_freeze(freeze_manifest)
                saved = json.loads((self.paths.reports / "experiment_rows.json").read_text())
                experiment_rows, coverage_rows = saved["rows"], saved["coverage"]
                entries = freeze_manifest["predictions"]
                for dataset in final_datasets.values():
                    for unit_id in dataset.unit_ids:
                        token = runtime.sha256_bytes(unit_id.encode())[:24]
                        state_paths[unit_id] = self.paths.checkpoints / "fitted_states" / f"{token}.pt"
            # Full-input deployment happens only after predictive labels/fits freeze.
            deployment_freeze_path = self.paths.exports / "deployment" / "freeze_manifest.json"
            if not deployment_freeze_path.exists():
                progress.event("finalization.pass", operation="full-input deployment exports")
                deployment_entries = []
                for split, dataset in final_datasets.items():
                    for index, unit_id in enumerate(dataset.unit_ids):
                        progress.update(operation="deployment image load and export", unit_id=unit_id,
                                        item=index + 1, items=len(dataset), split=split)
                        full_input, record = self.components.load_full_input(
                            self.manifest, unit_id, image_size=config.image_size)
                        with torch.no_grad():
                            for arm in ("student_no_audit", "student_audited"):
                                probabilities = torch.softmax(self.models[arm](
                                    full_input.unsqueeze(0).to(self.device).float()).float(), dim=1)[0].cpu()
                                deployment_entries.append(self.components.export_prediction(
                                    self.paths.exports / "deployment", record=record,
                                    prediction_name=f"{arm}_full_input", labels=probabilities.argmax(0).long(),
                                    probabilities=probabilities, validity=torch.ones_like(probabilities[0]),
                                    alternatives=[], checkpoint_id=checkpoint_paths[arm], version=version))
                with self.timing.stage("freeze.deployment_assemble_hash", predictions=len(deployment_entries)):
                    deployment_manifest = self.components.freeze_predictions(
                        self.paths.exports / "deployment", deployment_entries,
                        {arm: checkpoint_paths[arm] for arm in ("student_no_audit", "student_audited")},
                        dataset=config.dataset, protocol=self.manifest["resolved_protocol"], epoch=config.total_epochs - 1,
                        required_methods=[f"{arm}_full_input" for arm in ("student_no_audit", "student_audited")],
                        required_unit_ids=sorted(state_paths))
                self.components.validate_freeze(deployment_manifest, require_complete=True)
            else:
                deployment_manifest = json.loads(deployment_freeze_path.read_text())
                self.components.validate_freeze(deployment_manifest, require_complete=True)
                deployment_entries = deployment_manifest["predictions"]
            from .export import verified_freeze_session
            verification = []
            progress.event("finalization.pass", operation="frozen verification", freeze_id=freeze_manifest["freeze_id"])
            # Each unit has an atomic cached result. Resuming reuses that result
            # rather than rescoring O_verify; no quadratic all-unit journal rewrite.
            with verified_freeze_session(freeze_manifest, require_complete=True):
                for index, (unit_id, path) in enumerate(state_paths.items()):
                    progress.update(operation="frozen verification load and score", unit_id=unit_id,
                                    item=index + 1, items=len(state_paths), split=None)
                    with self.timing.stage("verification.load", path=str(path)):
                        state = torch.load(path, map_location="cpu", weights_only=False)
                        view = self.components.load_verification_unit(
                            self.manifest, unit_id, freeze_manifest,
                            image_size=config.image_size, seed=config.seed)
                    fits = {fit.hypothesis.candidate_id: fit for fit in state["fitted"]}
                    method_map = state["method_to_candidate"]
                    with self.timing.stage("verification.score_frozen", unit_id=unit_id):
                        row = self.components.verify_frozen_bank(
                            state["fitted"], view, self.observation, freeze_manifest,
                            selection_scores=state["selection_scores"], control_fitted=state["control_fitted"],
                            method_to_candidate=method_map,
                            student_fitted_by_arm={arm: fits[method_map[arm]]
                                for arm in ("student_no_audit", "student_audited")},
                            seed=config.seed, selected_candidate_id=state["audit"].selected.candidate_id,
                            resume_cached=True,
                            ledger_path=self.paths.reports / "verification_units" / f"{path.stem}.json")
                    verification.append({**row, "unit_id": unit_id, "patient_id": state["patient_id"],
                                         "split": state["split"], "method_to_candidate": method_map,
                                         "repeated_annotation_stability": state["repeated_annotation"]["metrics"]})
            runtime.atomic_write_json(self.paths.reports / "verification.json", {
                "freeze_id": freeze_manifest["freeze_id"], "rows": verification})
            self.components.validate_freeze(deployment_manifest, require_complete=True)
            from .reporting import write_image_only_reports
            with self.timing.stage("reports.image_only_metrics", path=str(self.paths.reports), units=len(verification)):
                image_only_reports = write_image_only_reports(
                    self.paths.root, self.manifest, self.history, verification,
                    experiment_rows, coverage_rows, config.total_epochs - 1,
                    checkpoint_paths["producer"])
            progress.event("reports.ready", paths={key: str(path) for key, path in image_only_reports.items()})
            foreground = sum(row["valid_foreground_student_audited"] for row in coverage_rows)
            if self.paths.failure_report.exists():
                self.paths.failure_report.rename(
                    self.paths.reports / f"recovered_failure_{uuid.uuid4().hex[:12]}.json")
            return {"attempted": True, "available": True,
                    "splits": sorted(final_datasets), "units_per_split": {k: len(v) for k, v in final_datasets.items()},
                    "units": len(state_paths), "exported_predictions": len(entries),
                    "deployment_predictions": len(deployment_entries),
                    "freeze_manifest": str(self.paths.freeze_manifest),
                    "image_only_reports": {key: str(path) for key, path in image_only_reports.items()},
                    "verification_rows": len(verification),
                    "verification_available_rows": sum(row.get("available", True) for row in verification),
                    "extra_dataset_passes": {"audit": 1, "experiments_export": 1, "deployment": 1, "verification": 1},
                    "scientific_outcome": ("label_generation_produced_valid_foreground" if foreground else
                                           "label_generation_objective_failed_zero_useful_coverage")}
        except Exception as exc:
            failure = {"stage": "finalization", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
            runtime.atomic_write_json(self.paths.failure_report, failure)
            progress.event("finalization.failed", error=failure["error"], path=str(self.paths.failure_report))
            return {"attempted": True, "available": False, "reason": failure["error"]}

    # -- preflight -------------------------------------------------------
    def preflight(self) -> dict[str, Any]:
        """Bounded real fit/score/edit-or-rejection/backward, checkpoint, export.

        Never marks completion and never writes ``pipeline_report.json``.
        """
        started = time.time()
        parent_paths = self.paths
        if self._setup_done:
            raise TrainerContractError("preflight requires a fresh trainer")
        self.paths = runtime.RunPaths(parent_paths.root / "preflight_probe").ensure()
        receipt: dict[str, Any] = {
            "status": "fail",
            "run_id": self.run_id,
            "device": runtime.device_identity(self.device),
            "physical_batch": self.config.batch_size,
            "accumulation_steps": self.config.accumulation_steps,
            "effective_batch": self.config.effective_batch,
            "completed": False,
            "marks_completion": False,
        }
        try:
            self.setup()
            receipt.update({key: self.identity()[key] for key in
                            ("scientific_hash", "source_hash", "manifest_hash")})
            runtime.reset_peak_memory(self.device)
            accumulator = EpochAccumulator()
            audit_counters = {
                "units": 0, "accepted_edits": 0, "no_change": 0, "semantic_unresolved": 0,
                "search_inconclusive": 0, "candidates": 0, "distinct_candidates": 0,
                "fitting_steps": 0, "prior_violations": 0, "score_unavailable": 0,
            }
            class_pixels: dict[Any, int] = {index: 0 for index in range(4)}
            class_pixels["total"] = 0
            if len(self.dataset) < self.config.effective_batch:
                raise TrainerContractError("preflight requires one full effective batch of distinct units")
            for micro in range(self.config.accumulation_steps):
                indices = list(range(micro * self.config.batch_size, (micro + 1) * self.config.batch_size))
                self._train_batch(
                    epoch=0, batch_index=micro, indices=indices,
                    ramp=runtime.label_ramp(0, ramp_epochs=LABEL_RAMP_EPOCHS),
                    accumulator=accumulator, audit_counters=audit_counters,
                    class_pixels=class_pixels,
                    lineage_path=self.paths.reports / "preflight_label_lineage.jsonl",
                    is_group_end=micro == self.config.accumulation_steps - 1,
                    group_weight=1.0 / self.config.accumulation_steps,
                )
            checkpoint_path = self._save_checkpoint(
                epoch=0, batch_cursor=self.config.accumulation_steps, permutation=list(range(len(self.dataset))),
                completed=False, status=STATUS_PARTIAL,
            )
            reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            export: dict[str, Any] | None = None
            export_reason: str | None = None
            try:
                unit = self.dataset[indices[0]]
                with torch.no_grad():
                    context = unit.fitting.context.unsqueeze(0).to(self.device).float()
                    logits = self.models["student_audited"](context)
                probabilities = torch.softmax(logits.float(), dim=1)[0].cpu()
                export = self.components.export_prediction(
                    self.paths.exports / "preflight",
                    record=unit.record,
                    prediction_name="preflight_student_audited",
                    labels=probabilities.argmax(dim=0).long(),
                    probabilities=probabilities,
                    validity=torch.ones_like(probabilities[0]),
                    alternatives=[],
                    checkpoint_id=str(checkpoint_path),
                    version=self.pseudo_label_version(0),
                )
            except Exception as exc:  # noqa: BLE001
                export_reason = f"{type(exc).__name__}: {exc}"

            audited_units = audit_counters["units"]
            attempted_edits = audit_counters["candidates"] - audited_units
            receipt.update(
                {
                    "status": "pass",
                    "audited_units": audited_units,
                    "audit_attempts": attempted_edits,
                    "accepted_edits": audit_counters["accepted_edits"],
                    "rejected_edits": audit_counters["no_change"],
                    "zero_attempt_audit": attempted_edits <= 0,
                    "optimizer_steps": self.global_step,
                    "component_steps": dict(self.component_steps),
                    "producer_loss": accumulator.mean("producer/loss"),
                    "student_no_audit_loss": accumulator.mean("student_no_audit/loss"),
                    "student_audited_loss": accumulator.mean("student_audited/loss"),
                    "grad_norms": {
                        name: accumulator.mean(f"{name}/grad_norm") for name in self.models
                    },
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_reload_ok": reloaded.get("global_step") == self.global_step,
                    "actual_physical_batch": len(indices),
                    "actual_accumulation_microbatches": self.config.accumulation_steps,
                    "checkpoint_sha256": runtime.sha256_file(checkpoint_path),
                    "export": export,
                    "export_unavailable_reason": export_reason,
                    "amp_enabled": self.amp_enabled,
                    "gpu_stats": runtime.gpu_stats(self.device),
                    "cpu_only_evidence": self.device.type != "cuda",
                    "elapsed_seconds": time.time() - started,
                    "timing": self.timing.as_dict(),
                }
            )
            if attempted_edits <= 0:
                receipt["status"] = "fail"
                receipt["reason"] = "zero-attempt audit does not test the auditor"
            if not all(self.component_steps.get(name, 0) > 0 for name in self.models):
                receipt["status"] = "fail"
                receipt["reason"] = "not every component completed backward/optimizer: full batch memory gate unverified"
            if export is None:
                receipt["status"] = "fail"
                receipt["reason"] = f"tiny export unavailable: {export_reason}"
        except Exception as exc:  # noqa: BLE001
            receipt["reason"] = f"{type(exc).__name__}: {exc}"
            receipt["traceback"] = traceback.format_exc()
        runtime.write_gate_receipt(self.paths.gate_receipt, receipt)
        runtime.atomic_write_json(self.paths.reports / "preflight_report.json", receipt)
        runtime.write_gate_receipt(parent_paths.gate_receipt, receipt)
        self.paths = parent_paths
        return receipt
