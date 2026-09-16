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

import inspect
import json
import math
import time
import traceback
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from . import auditor as auditor_module
from . import hypotheses as hypotheses_module
from . import observation as observation_module
from . import runtime
from .config import RUNTIME_FIELDS, MaskfreeConfig, compare_configs
from .contracts import VERSION as CONTRACT_VERSION
from .contracts import AuditResult, Hypothesis, ScoringView, TrainingUnit
from .progress import current_progress

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

# W1 keeps observation sufficient statistics and likelihood arithmetic in
# FP64 on whichever execution device the trainer resolves.  This is an
# identity-bearing contract: a checkpoint cannot silently migrate between
# observation backends during an exact resume.
AUDIT_NUMERICAL_BACKEND = "torch.float64"


class TrainerContractError(RuntimeError):
    """Raised when the run cannot proceed without violating the contract."""


class MissingComponentError(TrainerContractError):
    """Raised when another owner's frozen interface is not importable yet."""


class ResumeIdentityError(TrainerContractError):
    """Raised when a checkpoint belongs to a different experiment."""


def _resolve_audit_device(requested: str, model_device: torch.device) -> torch.device:
    """Resolve the observation execution device without implicit CUDA fallback.

    ``runtime.resolve_device`` intentionally permits a CUDA-to-CPU fallback for
    bounded neural-network software checks when ``allow_cpu`` is enabled.  The
    audit path has a stricter contract: an explicit ``audit_device='cuda'`` is a
    request for CUDA arithmetic and must fail closed when CUDA is unavailable.
    ``auto`` follows the already-resolved trainer model device, so CPU smoke
    runs remain portable while production templates stay on CUDA.
    """
    requested = str(requested)
    if requested == "auto":
        return torch.device(model_device)
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "cuda":
        raise TrainerContractError(
            f"audit_device must be 'auto', 'cpu' or 'cuda', got {requested!r}"
        )
    if not torch.cuda.is_available():
        raise TrainerContractError(
            "audit_device='cuda' is unavailable; refusing a silent CPU fallback"
        )
    try:
        # If neural models already target an indexed CUDA device, audit math
        # follows that same device.  A bare explicit CUDA request with CPU
        # models uses the current/default CUDA device as its independent audit
        # placement; no implicit cross-GPU migration is allowed.
        device = model_device if model_device.type == "cuda" else torch.device("cuda")
        # Force validation of a malformed/no-visible-device CUDA runtime before
        # constructing the observation model.  This does not synchronize any
        # kernels and is therefore outside the measured inner audit loop.
        torch.cuda.get_device_properties(device)
    except (RuntimeError, AssertionError, IndexError) as exc:
        raise TrainerContractError(
            f"audit_device='cuda' could not be initialized: {exc}"
        ) from exc
    return device


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
    # Optional canonical W2 batch entry point.  Older/custom Components
    # doubles intentionally omit it and therefore stay on the scalar path.
    audit_banks: Callable[..., list[AuditResult]] | None = None


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
    # ``audit_banks`` is an additive optional interface.  Do not make it part of
    # COMPONENT_SPECS: custom/partial test Components from the scalar contract
    # must continue to construct without it.
    try:
        auditor = importlib.import_module(f"{__package__}.auditor")
        resolved["audit_banks"] = getattr(auditor, "audit_banks", None)
    except ImportError:
        resolved["audit_banks"] = None
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
    digest = runtime.sha256_bytes(f"{seed}:{epoch}:{unit_id}".encode())
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


def _scalar_metrics(values: dict[str, Any]) -> dict[str, float | None]:
    """Copy independent scalar metrics once per device/dtype, preserving values."""
    copied = dict(values)
    groups: dict[tuple[torch.device, torch.dtype], list[tuple[str, torch.Tensor]]] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1 and value.device.type != "cpu":
            groups.setdefault((value.device, value.dtype), []).append((key, value.detach().reshape(())))
    for group in groups.values():
        numbers = torch.stack([value for _, value in group]).cpu().tolist()
        for (key, _), number in zip(group, numbers):
            copied[key] = number
    return {key: _scalar(value) for key, value in copied.items()}


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
        self.audit_device = _resolve_audit_device(config.audit_device, self.device)
        self.audit_numerical_backend = AUDIT_NUMERICAL_BACKEND
        self.run_id = config.run_id or self._default_run_id()
        run_root = Path(config.output_dir) / "runs" / "maskfree150" / config.dataset / self.run_id
        if (run_root / "run_identity.json").exists() and not config.resume:
            raise ResumeIdentityError("occupied run directory requires explicit resume")
        self.paths = runtime.RunPaths(run_root).ensure()

        self.source = runtime.source_identity(self.repo_root)
        self.timing = runtime.TimingAccumulator.empty(mode=config.timing_mode)
        # Only diagnostic timing synchronizes stage boundaries. Production
        # stage values are inclusive host wall time, including enqueue/waits.
        self.timing.device = (
            self.audit_device if self.audit_device.type == "cuda" else self.device
        )

        self.sampling_generator = torch.Generator()
        self.sampling_generator.manual_seed(config.seed * 7919 + 13)

        # populated by _setup
        self.manifest: dict[str, Any] = {}
        self.dataset: Any = None
        self.unit_ids: list[str] = []
        self.models: dict[str, torch.nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.scaler: torch.amp.GradScaler | None = None
        observation_factory = self.components.observation_model
        observation_kwargs = {
            "max_iterations": OBSERVATION_MAX_ITERATIONS,
            "variance_floor": OBSERVATION_VARIANCE_FLOOR,
            "beta": OBSERVATION_BETA,
        }
        # W1's additive execution_device keyword is optional for injected test
        # doubles and older source snapshots.  Inspect the callable rather than
        # catching arbitrary TypeError from inside a real constructor: custom
        # components therefore keep working without masking implementation
        # errors, while the canonical model receives the resolved device.
        try:
            signature = inspect.signature(observation_factory)
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            signature = None
        supports_execution_device = signature is None or "execution_device" in (
            signature.parameters if signature is not None else {}
        ) or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in (signature.parameters.values() if signature is not None else ())
        )
        if self.audit_device.type == "cuda" and not supports_execution_device:
            raise TrainerContractError(
                "resolved CUDA audit device requires an ObservationModel factory "
                "that accepts execution_device"
            )
        if supports_execution_device:
            observation_kwargs["execution_device"] = self.audit_device
        self.observation = observation_factory(**observation_kwargs)

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
        self._lineage_writers: dict[Path, Any] = {}
        self._prefetched_batches = None
        self._prefetch_stats: dict[str, Any] = {}
        self._candidate_executor = None
        self._candidate_stats: dict[str, Any] = {}

    def _close_candidate_executor(self) -> None:
        if self._candidate_executor is not None:
            self._candidate_stats = self._candidate_executor.stats()
            self._candidate_executor.close()
            self._candidate_executor = None

    def _append_lineage(self, path: Path, record: dict[str, Any]) -> None:
        if self.config.logging_mode == "sync":
            runtime.append_jsonl(path, record)
            return
        if path not in self._lineage_writers:
            self._lineage_writers[path] = runtime.JSONLWriter(
                path, max_resident_bytes=self.config.log_buffer_bytes)
        self._lineage_writers[path].append(record)

    def _flush_lineage(self) -> None:
        for writer in self._lineage_writers.values():
            writer.flush()

    def _close_lineage(self) -> None:
        # Attempt every close; do not silently swallow a durability failure.
        errors = []
        for writer in self._lineage_writers.values():
            try:
                writer.close()
            except Exception as exc:
                errors.append(exc)
        self._lineage_writers.clear()
        if errors:
            raise errors[0]

    # -- identity -------------------------------------------------------
    def _default_run_id(self) -> str:
        digest = runtime.sha256_json(self.config.scientific_identity())[:10]
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        return f"{self.config.dataset}-{stamp}-{digest}-{uuid.uuid4().hex[:8]}"

    def identity(self) -> dict[str, Any]:
        audit_identity = self._audit_execution_identity()
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
            "audit_device_requested": self.config.audit_device,
            "audit_device_identity": audit_identity["resolved_device"],
            "audit_numerical_backend": audit_identity["numerical_backend"],
            "audit_execution": audit_identity,
            "runtime_options": self._runtime_options(),
        }

    def _runtime_options(self) -> dict[str, Any]:
        return {name: getattr(self.config, name) for name in RUNTIME_FIELDS}

    def runtime_summary(self) -> dict[str, Any]:
        cache_stats = getattr(self.dataset, "cache_stats", None)
        return {
            **self._runtime_options(),
            "model_device": str(self.device), "audit_device": str(self.audit_device),
            "neural_precision": "fp16_amp" if self.config.amp and self.device.type == "cuda" else "fp32",
            "observation_precision": self.audit_numerical_backend,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "data_worker_threads": int(bool(self.config.prefetch_batches)),
            "physical_batch": self.config.batch_size, "effective_batch": self.config.effective_batch,
            "image_size": self.config.image_size,
            "cache": cache_stats() if callable(cache_stats) else None,
            "prefetch": dict(self._prefetch_stats),
            "candidate_execution": (self._candidate_executor.stats() if self._candidate_executor is not None
                                    else dict(self._candidate_stats)),
            "lineage_writers": {str(path.relative_to(self.paths.root)): dict(writer.stats)
                                for path, writer in self._lineage_writers.items()},
        }

    def _audit_execution_identity(self) -> dict[str, Any]:
        """Return portable placement/backend metadata for reports and resumes."""
        return {
            "requested_device": self.config.audit_device,
            "resolved_device": runtime.device_identity(self.audit_device),
            "placement": self.audit_device.type,
            "numerical_backend": self.audit_numerical_backend,
            "model_device": runtime.device_identity(self.device),
            "execution_device_source": "trainer_model_device" if self.config.audit_device == "auto" else "explicit_config",
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
            dataset_kwargs = dict(split="train", image_size=config.image_size, seed=config.seed)
            if "data_cache_bytes" in inspect.signature(self.components.dataset_factory).parameters:
                dataset_kwargs["data_cache_bytes"] = config.data_cache_bytes
            elif config.data_cache_bytes != 64 * 1024 * 1024:
                raise TrainerContractError("custom dataset does not support data_cache_bytes")
            self.dataset = self.components.dataset_factory(self.manifest, **dataset_kwargs)
            if config.prefetch_batches and not hasattr(self.dataset, "iter_batches"):
                raise TrainerContractError("prefetch requires the canonical deterministic dataset iterator")
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
        if config.candidate_workers:
            if not self._bulk_audit_enabled():
                raise TrainerContractError("candidate workers require canonical audit components")
            from .candidate_execution import CandidatePoolExecutor
            self._candidate_executor = CandidatePoolExecutor(
                workers=config.candidate_workers, worker_threads=config.candidate_worker_threads,
                chunk_size=config.candidate_chunk_size)

        with self.timing.stage("run_identity.write", path=str(self.paths.root)):
            self._write_run_identity()
        from .resources import resource_snapshot
        self.startup_resources = resource_snapshot(
            data_roots=(config.data_root,), report_dir=self.paths.reports, include_gpu=False)
        visible_cpu = self.startup_resources.get("cgroup", {}).get("visible_cpu_upper_bound_cores")
        if visible_cpu is not None and config.candidate_workers * config.candidate_worker_threads > visible_cpu:
            raise TrainerContractError("candidate worker threads exceed visible container CPU upper bound")
        runtime.atomic_write_json(self.paths.reports / "startup_resources.json", self.startup_resources)
        runtime.atomic_write_json(self.paths.reports / "runtime_options.json", self.runtime_summary())
        current_progress().event("setup.ready", units=len(self.dataset), batches=self.batches_per_epoch,
                                 protocol=self.manifest.get("resolved_protocol"),
                                 effective_data_workers=int(bool(config.prefetch_batches)), amp=use_amp,
                                 runtime_options=self.runtime_summary(),
                                 audit_device=self._audit_execution_identity()["resolved_device"],
                                 audit_numerical_backend=self.audit_numerical_backend)
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
                for key in (
                    "run_id", "dataset", "scientific_hash", "manifest_hash", "partition_hash",
                    "audit_device_identity", "audit_numerical_backend",
                    "runtime_options",
                )
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
    def _validate_preflight_gate(self) -> None:
        """Validate the receipt emitted by preflight without rewriting it."""
        if not self.paths.gate_receipt.is_file():
            raise TrainerContractError("150-epoch GPU run requires a completed matching preflight")
        gate = json.loads(self.paths.gate_receipt.read_text())
        identity = self.identity()
        for key, value in {
            "status": "pass", "actual_physical_batch": self.config.batch_size,
            "accumulation_steps": self.config.accumulation_steps,
            "device": runtime.device_identity(self.device),
            "audit_device_identity": runtime.device_identity(self.audit_device),
            "audit_numerical_backend": self.audit_numerical_backend,
            "scientific_hash": identity["scientific_hash"],
            "source_hash": identity["source_hash"],
            "manifest_hash": identity["manifest_hash"],
            "runtime_options": self._runtime_options(),
        }.items():
            if gate.get(key) != value:
                raise TrainerContractError(f"preflight gate mismatch: {key}")

    def run(self) -> dict[str, Any]:
        started = time.time()
        try:
            self.setup()
            if self.config.total_epochs == 150 and not self.config.is_bounded_run and self.device.type == "cuda":
                self._validate_preflight_gate()
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
            try:
                self._close_lineage()
            finally:
                self._close_candidate_executor()
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
            "audit_execution": self._audit_execution_identity(),
            "epochs_completed": self.last_completed_epoch + 1,
            "last_completed_epoch": self.last_completed_epoch,
            "total_epochs": config.total_epochs,
            "bounded_run": config.is_bounded_run,
            "software_timeline": config.total_epochs != 150,
            "global_optimizer_steps": self.global_step,
            "component_steps": dict(self.component_steps),
            "micro_batches": self.micro_batches_seen,
            "batches_per_epoch": self.batches_per_epoch,
            "effective_data_workers": int(bool(config.prefetch_batches)),
            "runtime_options": self.runtime_summary(),
            "steps_per_epoch": self.steps_per_epoch,
            "device": runtime.device_identity(self.device),
            "audit_device": runtime.device_identity(self.audit_device),
            "audit_numerical_backend": self.audit_numerical_backend,
            "gpu_stats": runtime.gpu_stats(self.device),
            "audit_gpu_stats": runtime.gpu_stats(self.audit_device),
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
            # Physical bulk API invocations are kept separate from the
            # per-unit logical budgets above.  A single fit_many/score_many
            # call may carry many candidate items and must be counted once.
            "physical_fit_many_calls": 0,
            "physical_fit_many_items": 0,
            "physical_score_many_calls": 0,
            "physical_score_many_items": 0,
            "physical_prepare_score_calls": 0,
            "physical_score_region_calls": 0,
            "physical_scalar_region_score_calls": 0,
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
            for key in (
                "physical_fit_many_calls", "physical_fit_many_items",
                "physical_score_many_calls", "physical_score_many_items",
                "physical_prepare_score_calls", "physical_score_region_calls",
                "physical_scalar_region_score_calls",
            ):
                audit_counters.setdefault(key, 0)
            class_pixels = dict(self._partial_epoch["class_pixels"])
        batch_cursor = start_batch
        progress = current_progress()
        progress.update(epoch=epoch + 1, epochs=config.total_epochs, batch=start_batch,
                        batches=len(batches), operation="training", unit_id=None)
        progress.event("epoch.start", resumed_from_batch=start_batch, label_ramp=ramp)
        if config.prefetch_batches:
            self._prefetched_batches = self.dataset.iter_batches(
                batches, start_batch=start_batch, prefetch_batches=config.prefetch_batches,
                prefetch_max_bytes=config.prefetch_max_bytes)
        try:
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
                        timing_note="inclusive stages; do not sum",
                        gpu_stats=runtime.gpu_stats(self.device),
                        audit_gpu_stats=runtime.gpu_stats(self.audit_device),
                        audit_execution=self._audit_execution_identity(),
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

        finally:
            if self._prefetched_batches is not None:
                self._prefetched_batches.close()
                self._prefetch_stats = self._prefetched_batches.stats()
                self._prefetched_batches = None

        units = max(audit_counters["units"], 1)
        record = {
            "schema_version": TRAINER_SCHEMA_VERSION,
            "global_epoch": epoch,
            "dataset": config.dataset,
            "protocol": self.manifest.get("resolved_protocol"),
            "contract_version": CONTRACT_VERSION,
            "audit_execution": self._audit_execution_identity(),
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

    def _candidate_features_to_cpu(self, features: torch.Tensor) -> torch.Tensor:
        """Transfer only the channels the canonical candidate generator consumes.

        Custom generators keep the full feature interface.  Slicing is a view
        before the device copy; no extra full-sized CUDA clone is allocated.
        The bank identity already hashes exactly these consumed channels.
        """
        detached = features.detach()
        if self.components.generate_bank is hypotheses_module.generate_bank:
            detached = detached[:, :hypotheses_module.MAX_FEATURE_CHANNELS]
        return detached.cpu()

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
        with self.timing.stage("data_load", indices=list(indices), batch_units=len(indices)):
            units: list[TrainingUnit] = (next(self._prefetched_batches) if self._prefetched_batches is not None
                                         else [self.dataset[index] for index in indices])
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
        producer_numbers = _scalar_metrics({**(producer_metrics or {}), "loss": loss})
        for key, number in producer_numbers.items():
            if key == "loss":  # loss itself is recorded once below
                continue
            if number is not None:
                accumulator.add(f"producer/{key}", number)
        accumulator.add("producer/loss", producer_numbers["loss"])

        # Bank generation consumes DETACHED producer features: no student or
        # auditor gradient ever reaches the representation producer.
        with self.timing.stage("feature_forward"):
            producer.eval()
            with torch.no_grad():
                features = producer(context)["features"].detach()
            producer.train()
        # Hypothesis generation remains a detached CPU-side workload so the
        # candidate IDs/labels and topology stay byte-for-byte deterministic.
        # Stage exactly one feature transfer for the complete physical batch;
        # W1's ObservationModel owns the subsequent fitting/scoring placement
        # on ``self.audit_device`` and its public outputs remain portable.
        # The per-unit path below only indexes this CPU tensor and never calls
        # ``.cpu()`` again.
        with self.timing.stage("features.to_cpu", batch_units=len(units)):
            features_cpu = self._candidate_features_to_cpu(features)
        # Neither bank generation nor the students need the GPU feature map.
        del features

        probs_initial: list[torch.Tensor] = []
        probs_audited: list[torch.Tensor] = []
        validity_initial: list[torch.Tensor] = []
        validity_audited: list[torch.Tensor] = []

        bulk_audits: list[AuditResult] | None = None
        banks: list[list[Hypothesis]] | None = None
        if self._bulk_audit_enabled():
            if self._candidate_executor is not None:
                with self.timing.stage("bank_generation", batch_units=len(units), workers=self.config.candidate_workers):
                    banks = self._candidate_executor.generate_banks(
                        [unit.fitting for unit in units], features=list(features_cpu),
                        seeds=[_stable_unit_seed(self.config.seed, epoch, unit.fitting.unit_id) for unit in units])
                if len(banks) != len(units) or any(not bank for bank in banks):
                    raise TrainerContractError("candidate worker returned an incomplete batch")
            else:
                banks = []
                for offset, unit in enumerate(units):
                    # Keep the scalar path's per-unit progress contract: each
                    # unit announces the same operation before bank generation and
                    # audit work, even though the canonical bulk call follows.
                    current_progress().update(
                        unit_id=unit.fitting.unit_id, unit_in_batch=offset + 1,
                        operation="generate hypotheses and audit select evidence",
                    )
                    banks.append(self._generate_bank(epoch=epoch, unit=unit, features=features_cpu[offset]))
            with self.timing.stage(
                "audit", batch_units=len(units), candidates=sum(len(bank) for bank in banks),
                rounds=AUDIT_ROUNDS, physical_mode="bulk",
            ):
                bulk_audits = self.components.audit_banks(
                    banks,
                    [unit.fitting for unit in units],
                    [unit.selection for unit in units],
                    self.observation,
                    rounds=AUDIT_ROUNDS,
                    improvement_threshold=IMPROVEMENT_THRESHOLD,
                )
            if len(bulk_audits) != len(units):
                raise TrainerContractError(
                    "audit_banks returned a result count different from the physical batch"
                )
            # These are API invocations, not claims about the number of kernels
            # or likelihood evaluations inside the observation owner.
            total_candidates = sum(len(bank) for bank in banks)
            audit_counters["physical_fit_many_calls"] += 1
            audit_counters["physical_fit_many_items"] += total_candidates
            audit_counters["physical_score_many_calls"] += 1
            audit_counters["physical_score_many_items"] += total_candidates

        for offset, unit in enumerate(units):
            if bulk_audits is None:
                current_progress().update(
                    unit_id=unit.fitting.unit_id, unit_in_batch=offset + 1,
                    operation="generate hypotheses and audit select evidence",
                )
                audit = self._audit_unit(
                    epoch=epoch,
                    unit=unit,
                    features=features_cpu[offset],
                    audit_counters=audit_counters,
                    accumulator=accumulator,
                    lineage_path=lineage_path,
                )
            else:
                audit = self._record_audit(
                    epoch=epoch,
                    unit=unit,
                    audit=bulk_audits[offset],
                    audit_counters=audit_counters,
                    accumulator=accumulator,
                    lineage_path=lineage_path,
                )
            initial, selected = audit.initial, audit.selected
            probs_initial.append(initial.probabilities)
            probs_audited.append(selected.probabilities)
            validity_initial.append(initial.validity)
            validity_audited.append(audit.validity)
            labels = selected.labels.detach().cpu()
            valid_pixels = audit.validity.detach().cpu() > 0
            class_pixels["total"] += int(valid_pixels.sum())
            for index in range(4):
                class_pixels[index] += int(((labels == index) & valid_pixels).sum().item())

        targets = {
            "student_no_audit": (
                torch.stack(probs_initial).detach().to(self.device).float(),
                torch.stack(validity_initial).detach().to(self.device).float(),
            ),
            "student_audited": (
                torch.stack(probs_audited).detach().to(self.device).float(),
                torch.stack(validity_audited).detach().to(self.device).float(),
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
            student_numbers = _scalar_metrics({**(student_metrics or {}), "loss": student_value})
            accumulator.add(f"{arm}/loss", student_numbers["loss"])
            for key, number in student_numbers.items():
                if key == "loss":  # do not double the observation count
                    continue
                if number is not None:
                    accumulator.add(f"{arm}/{key}", number)
            if student_numbers.get("skipped"):
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
    def _bulk_audit_enabled(self) -> bool:
        """Return whether this trainer is on the canonical, deterministic path.

        ``Components`` is intentionally injectable for focused tests and for
        delayed workers.  A generic ``hasattr`` check would accidentally batch
        custom callbacks/subclasses that override RNG or scalar semantics, so
        only the exact canonical W2 entry points and W1 model class opt in.
        """
        def canonical_method(name: str) -> bool:
            bound = getattr(self.observation, name, None)
            base = getattr(observation_module.ObservationModel, name, None)
            return callable(bound) and getattr(bound, "__func__", bound) is base

        return bool(
            getattr(self.components, "audit_banks", None) is auditor_module.audit_banks
            and self.components.generate_bank is hypotheses_module.generate_bank
            and self.components.audit_bank is auditor_module.audit_bank
            and type(self.observation) is observation_module.ObservationModel
            and all(
                canonical_method(name)
                for name in (
                    "fit", "score", "fit_many", "score_many", "prepare_score", "score_region"
                )
            )
        )

    def _generate_bank(
        self, *, epoch: int, unit: TrainingUnit, features: torch.Tensor
    ) -> list[Hypothesis]:
        """Generate one unchanged per-unit bank from a detached CPU feature row."""
        seed = _stable_unit_seed(self.config.seed, epoch, unit.fitting.unit_id)
        with self.timing.stage("bank_generation", unit_id=unit.fitting.unit_id):
            bank = self.components.generate_bank(
                unit.fitting, features=features.detach(), seed=seed
            )
        if not bank:
            raise TrainerContractError(
                f"generate_bank returned no candidate for unit {unit.fitting.unit_id}"
            )
        return list(bank)

    def _record_audit(
        self,
        *,
        epoch: int,
        unit: TrainingUnit,
        audit: AuditResult,
        audit_counters: dict[str, int],
        accumulator: EpochAccumulator,
        lineage_path: Path,
    ) -> AuditResult:
        """Accumulate one result and persist its label-lineage row."""
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

        physical = trace.get("physical_work")
        # ``audit_banks`` repeats a batch-scoped API summary on each unit.  Do
        # not sum those shared calls; trainer._train_batch records them once.
        if isinstance(physical, dict) and physical.get("scope") != "batch":
            audit_counters["physical_prepare_score_calls"] += int(
                physical.get("prepare_score_calls", 0)
            )
            audit_counters["physical_score_region_calls"] += int(
                physical.get("score_region_calls", 0)
            )
            audit_counters["physical_scalar_region_score_calls"] += int(
                physical.get("scalar_region_score_calls", 0)
            )

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

        self._append_lineage(
            lineage_path,
            {
                "global_epoch": epoch,
                "unit_id": unit.fitting.unit_id,
                "study_id": unit.fitting.study_id,
                "partition_id": unit.fitting.partition_id,
                "audit_execution": self._audit_execution_identity(),
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
        """Generate, scalar-audit and record one unit (compatibility fallback)."""
        bank = self._generate_bank(epoch=epoch, unit=unit, features=features)
        with self.timing.stage(
            "audit", unit_id=unit.fitting.unit_id, candidates=len(bank), rounds=AUDIT_ROUNDS,
            physical_mode="scalar",
        ):
            audit = self.components.audit_bank(
                bank,
                unit.fitting,
                unit.selection,
                self.observation,
                rounds=AUDIT_ROUNDS,
                improvement_threshold=IMPROVEMENT_THRESHOLD,
            )
        return self._record_audit(
            epoch=epoch, unit=unit, audit=audit,
            audit_counters=audit_counters, accumulator=accumulator,
            lineage_path=lineage_path,
        )

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
        # Journal bytes must be durable BEFORE offsets enter the atomic
        # checkpoint. Bytes beyond those offsets may be discarded/replayed.
        self._flush_lineage()
        runtime.fsync_directory(self.paths.reports)
        payload = {
            "schema_version": TRAINER_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "identity": self.identity(),
            "audit_execution": self._audit_execution_identity(),
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
                for p in (self.paths.epoch_metrics, self.paths.reports / "label_lineage.jsonl",
                          self.paths.reports / "preflight_label_lineage.jsonl") if p.exists()
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
        for key in (
            "run_id", "runtime_environment", "device_identity",
            "audit_device_identity", "audit_numerical_backend",
            "runtime_options",
        ):
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
            # Audit stages may enqueue CUDA work on an explicit device that
            # differs from the neural model device.  Preserve that placement
            # across exact resume; CPU audit timing falls back to model timing
            # for legacy behavior.
            device=(self.audit_device if self.audit_device.type == "cuda" else self.device),
            mode=self.config.timing_mode)
        # Discard log writes beyond the committed checkpoint transaction.
        for relative, offset in payload.get("log_offsets", {}).items():
            log_path = self.paths.root / relative
            if (not log_path.resolve().is_relative_to(self.paths.root.resolve())
                    or not isinstance(offset, int) or isinstance(offset, bool) or offset < 0):
                raise ResumeIdentityError("invalid checkpoint journal offset/path")
            if not log_path.is_file() or log_path.stat().st_size < offset:
                raise ResumeIdentityError(f"checkpointed journal missing or shorter than durable offset: {relative}")
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
                    "audit_execution": self._audit_execution_identity(),
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
            "audit_device": runtime.device_identity(self.audit_device),
            "audit_device_identity": runtime.device_identity(self.audit_device),
            "audit_execution": self._audit_execution_identity(),
            "audit_numerical_backend": self.audit_numerical_backend,
            "physical_batch": self.config.batch_size,
            "runtime_options": self._runtime_options(),
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
            runtime.reset_peak_memory(self.audit_device)
            accumulator = EpochAccumulator()
            audit_counters = {
                "units": 0, "accepted_edits": 0, "no_change": 0, "semantic_unresolved": 0,
                "search_inconclusive": 0, "candidates": 0, "distinct_candidates": 0,
                "fitting_steps": 0, "prior_violations": 0, "score_unavailable": 0,
                "physical_fit_many_calls": 0, "physical_fit_many_items": 0,
                "physical_score_many_calls": 0, "physical_score_many_items": 0,
                "physical_prepare_score_calls": 0, "physical_score_region_calls": 0,
                "physical_scalar_region_score_calls": 0,
            }
            class_pixels: dict[Any, int] = {index: 0 for index in range(4)}
            class_pixels["total"] = 0
            if len(self.dataset) < self.config.effective_batch:
                raise TrainerContractError("preflight requires one full effective batch of distinct units")
            if self.config.prefetch_batches:
                self._prefetched_batches = self.dataset.iter_batches(
                    [list(range(m * self.config.batch_size, (m + 1) * self.config.batch_size))
                     for m in range(self.config.accumulation_steps)],
                    prefetch_batches=self.config.prefetch_batches,
                    prefetch_max_bytes=self.config.prefetch_max_bytes)
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
            if self._prefetched_batches is not None:
                self._prefetched_batches.close()
                self._prefetch_stats = self._prefetched_batches.stats()
                self._prefetched_batches = None
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
                    # Keep the logical audit budget and physical bulk API work
                    # visible in the file-backed preflight receipt.  The
                    # ``physical_*`` counters describe API invocations/items;
                    # they are deliberately separate from logical candidate,
                    # round and regional score counts above.
                    "audit_counters": dict(audit_counters),
                    "physical_audit": {
                        key: value for key, value in audit_counters.items()
                        if key.startswith("physical_")
                    },
                    "export": export,
                    "export_unavailable_reason": export_reason,
                    "amp_enabled": self.amp_enabled,
                    "gpu_stats": runtime.gpu_stats(self.device),
                    "audit_gpu_stats": runtime.gpu_stats(self.audit_device),
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
        finally:
            try:
                if self._prefetched_batches is not None:
                    self._prefetched_batches.close()
                    self._prefetched_batches = None
                self._close_lineage()
            except Exception as exc:
                receipt["status"] = "fail"
                receipt["reason"] = f"lineage durability failure: {exc}"
            finally:
                self._close_candidate_executor()
        runtime.write_gate_receipt(self.paths.gate_receipt, receipt)
        runtime.atomic_write_json(self.paths.reports / "preflight_report.json", receipt)
        runtime.write_gate_receipt(parent_paths.gate_receipt, receipt)
        self.paths = parent_paths
        return receipt
