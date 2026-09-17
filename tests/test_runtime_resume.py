"""Exact-resume regressions for the unified curriculum (Wave 4).

The question these tests answer is narrow and load-bearing: does a run that is
interrupted after a *completed* epoch and resumed from its committed ``last.pt``
end in the byte-identical state an uninterrupted run of the same schedule would
have reached?  "State" here is deliberately the whole live training state --
model weights, optimizer, scheduler, gradient scaler, the Python/NumPy/PyTorch
RNG streams, the step counters, the freeze masks, and the per-group learning
rates -- because a resume that reproduces only the weights silently restarts the
schedule and produces a different model from epoch ``k+1`` onwards.

Design notes:

* One real 6-epoch curriculum (2 annotation-bootstrap, 2 auditor, 2 joint) is
  trained once per module.  Every epoch commits through the production commit
  protocol, and the *actually written* checkpoint tree is snapshotted (by hard
  link, so the immutable ``selected_best/`` snapshots and the ``best.pt`` alias
  keep their exact bytes) after each commit.  Nothing is fabricated: the resume
  inputs are the states the trainer itself wrote.
* Resumes are parameterized over the suffix of that single baseline.  No
  counter is edited by hand and no configuration is extended on resume; only
  the output/report destinations move, which is the one difference
  ``compare_execution_configs`` permits.
* The intervals deliberately differ in batch size (2/3/2), loader cardinality
  (3/2/3), accumulation steps (1/1/2), augmentation, trainability, rollout mode
  and learning rates, so a schedule that restarts rather than continues cannot
  coincidentally match.
* The encoder is forced onto the repository's own dependency-light fallback
  during model construction, for these unit tests only, to keep the snapshot
  tree small.  Canonical pretraining is untouched; the unpatched ConvNeXt CLI
  path is covered by the separate smoke suite.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Iterator

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

import self_audit.training.unified_trainer as unified_trainer_module
from self_audit.models.self_audit_net import SelfAuditNet
from self_audit.provenance import source_content_signature
from self_audit.artifact_io import atomic_write_json as _real_atomic_write_json
from self_audit.training._utils import save_checkpoint as _real_save_checkpoint
from self_audit.training.unified_config import parse_unified_config
from self_audit.training.unified_trainer import COMMITTED_BEST_REFERENCE_KEY, UnifiedTrainer


# ---------------------------------------------------------------------------
# Fixture scale
# ---------------------------------------------------------------------------

TOTAL_EPOCHS = 6
SNAPSHOT_EPOCHS = (1, 2, 3, 4, 5)
IMAGE_SIZE = 32
CASE_COUNT = 6
SEED = 4242

# num_workers for the augmenting interval's train loader.  Exercised for real;
# if the platform's start method cannot carry the dataset the fixture says so
# explicitly rather than quietly degrading to num_workers=0.
AUGMENT_WORKERS = 2


_INTERVALS: list[dict[str, Any]] = [
    {
        "start_epoch": 0,
        "end_epoch": 2,
        "name": "annotation_bootstrap",
        "trainable": "annotation",
        "encoder_lr": 1e-4,
        "annotation_lr": 1e-3,
        "auditor_lr": 0.0,
        "annotation_weight": 1.0,
        "audit_weight": 0.0,
        "objective": "weighted_a0_a3",
        "transition_population": "none",
        "rollout": "propagate_no_audit",
        "batch_size": 2,
        "accumulation_steps": 1,
        "augment": False,
        "reset_optimizer": False,
    },
    {
        "start_epoch": 2,
        "end_epoch": 4,
        "name": "auditor_counterfactual",
        "trainable": "auditor",
        "encoder_lr": 0.0,
        "annotation_lr": 0.0,
        "auditor_lr": 5e-4,
        "annotation_weight": 0.0,
        "audit_weight": 1.0,
        "objective": "counterfactual_audit",
        "transition_population": "adjacent_and_synthetic",
        "rollout": "annotation_eval",
        "batch_size": 3,
        "accumulation_steps": 1,
        "augment": True,
        "reset_optimizer": True,
    },
    {
        "start_epoch": 4,
        "end_epoch": 6,
        "name": "joint_finetune",
        "trainable": "all",
        "encoder_lr": 5e-5,
        "annotation_lr": 5e-4,
        "auditor_lr": 5e-4,
        "annotation_weight": 1.0,
        "audit_weight": 1.0,
        "objective": "retained_final_annotation",
        "transition_population": "active_attempted",
        "rollout": "threshold_gate",
        "batch_size": 2,
        "accumulation_steps": 2,
        "augment": False,
        "reset_optimizer": True,
    },
]


class _SyntheticCurriculumDataset(Dataset):
    """Small deterministic image/mask cohort.

    Defined at module scope so a spawned DataLoader worker can import it.
    When ``augment`` is set, ``__getitem__`` draws from the *worker's* torch RNG,
    which is exactly the stream that a resume must reproduce.
    """

    def __init__(
        self,
        count: int = CASE_COUNT,
        size: int = IMAGE_SIZE,
        *,
        augment: bool = False,
        seed: int = 1234,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(count, 3, size, size, generator=generator)
        self.masks = torch.randint(0, 4, (count, size, size), generator=generator)
        self.case_ids = [f"case_{index // 2:03d}" for index in range(count)]
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = self.images[index]
        if self.augment:
            image = image + 0.01 * torch.randn(image.shape)
        return {
            "image": image,
            "mask": self.masks[index],
            "case_id": self.case_ids[index],
        }


@contextmanager
def _forced_fallback_encoder() -> Iterator[None]:
    """Force ``build_encoder`` onto the repository's dependency-light fallback.

    Only the encoder *construction* is affected, and only inside this block.
    """

    try:
        import timm
    except Exception:  # pragma: no cover - fallback is already the only option
        yield
        return

    original = timm.create_model

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("forced dependency-light fallback encoder for unit resume tests")

    timm.create_model = _refuse
    try:
        yield
    finally:
        timm.create_model = original


def _tiny_net() -> SelfAuditNet:
    torch.manual_seed(99)
    with _forced_fallback_encoder():
        return SelfAuditNet(
            pretrained_encoder=False,
            encoder_allow_fallback=True,
            shared_channels=8,
            window_k=2,
            max_turns=2,
        )


def _config_dict(output_dir: Path, report_dir: Path) -> dict[str, Any]:
    with open("configs/self_audit_full.yaml", "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)
    cfg_dict["experiment"]["device"] = "cpu"
    cfg_dict["experiment"]["seed"] = SEED
    cfg_dict["model"]["pretrained_encoder"] = False
    cfg_dict["model"]["fallback"] = True
    cfg_dict["model"]["shared_channels"] = 8
    cfg_dict["model"]["window_k"] = 2
    cfg_dict["model"]["max_turns"] = 2
    cfg_dict["training"]["amp"]["enabled"] = False
    cfg_dict["training"]["schedule"]["total_epochs"] = TOTAL_EPOCHS
    cfg_dict["training"]["schedule"]["intervals"] = copy.deepcopy(_INTERVALS)
    cfg_dict["checkpoint"]["output_dir"] = str(output_dir)
    cfg_dict["checkpoint"]["best_selection_min_epoch"] = 4
    cfg_dict["logging"]["report_dir"] = str(report_dir)
    cfg_dict["logging"]["wandb"]["enabled"] = False
    cfg_dict["dataset"]["dataloader"]["num_workers"] = 0
    cfg_dict["dataset"]["dataloader"]["pin_memory"] = False
    cfg_dict["dataset"]["dataloader"]["persistent_workers"] = False
    cfg_dict["calibration"]["threshold_steps"] = 5
    return cfg_dict


def _install_loaders(trainer: UnifiedTrainer, *, workers_for_augment: int) -> None:
    """Install the per-interval loaders the schedule will ask for.

    The augmenting interval gets a shuffled, non-persistent, multi-worker loader;
    the others stay in-process.  Cardinalities differ per interval on purpose.
    """

    for interval in trainer.config.training.schedule.intervals:
        workers = workers_for_augment if interval.augment else 0
        train_loader = DataLoader(
            _SyntheticCurriculumDataset(augment=interval.augment, seed=1234),
            batch_size=interval.batch_size,
            shuffle=True,
            num_workers=workers,
            persistent_workers=False,
        )
        val_loader = DataLoader(
            _SyntheticCurriculumDataset(augment=False, seed=77),
            batch_size=interval.batch_size,
            shuffle=False,
            num_workers=0,
        )
        trainer._loader_cache[(interval.batch_size, interval.augment)] = (train_loader, val_loader)


def _build_trainer(root: Path, name: str, *, workers_for_augment: int = AUGMENT_WORKERS) -> UnifiedTrainer:
    output_dir = root / name / "weights"
    report_dir = root / name / "reports"
    config = parse_unified_config(_config_dict(output_dir, report_dir))
    trainer = UnifiedTrainer(config, model=_tiny_net(), device=torch.device("cpu"), disable_tqdm=True)
    _install_loaders(trainer, workers_for_augment=workers_for_augment)
    return trainer


# ---------------------------------------------------------------------------
# State capture
# ---------------------------------------------------------------------------


def _digest(obj: Any) -> str:
    """Order-stable content digest of an arbitrary state-dict-shaped object."""

    hasher = hashlib.sha256()

    def walk(value: Any, path: str) -> None:
        if torch.is_tensor(value):
            hasher.update(path.encode())
            tensor = value.detach().cpu().contiguous()
            hasher.update(str(tensor.dtype).encode())
            hasher.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=repr):
                walk(value[key], f"{path}.{key!r}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        else:
            hasher.update(f"{path}={value!r}".encode())

    walk(obj, "")
    return hasher.hexdigest()


def _rng_fingerprint() -> dict[str, str]:
    numpy_state = np.random.get_state()
    return {
        "python": _digest(random.getstate()),
        "numpy": _digest((str(numpy_state[0]), numpy_state[1].tolist(), int(numpy_state[2]),
                          int(numpy_state[3]), float(numpy_state[4]))),
        "torch": _digest(torch.get_rng_state()),
    }


def _capture(trainer: UnifiedTrainer) -> dict[str, Any]:
    """Snapshot everything a resume must reproduce."""

    optimizer = trainer.optimizer
    return {
        "model": _digest(trainer.model.state_dict()),
        "optimizer": _digest(optimizer.state_dict()) if optimizer is not None else None,
        "scheduler": _digest(trainer.scheduler.state_dict()) if trainer.scheduler is not None else None,
        "scaler": _digest(trainer.scaler.state_dict()) if trainer.scaler is not None else None,
        "rng": _rng_fingerprint(),
        "counters": {
            "global_step": trainer.global_step,
            "optimizer_step": trainer.optimizer_step,
            "completed_epochs": trainer.completed_epochs,
            "global_epoch": trainer.global_epoch,
            "current_interval_index": trainer.current_interval_index,
            "last_checkpoint_committed_epoch": trainer.last_checkpoint_committed_epoch,
        },
        "selection": {
            "best_epoch": trainer.best_epoch,
            "best_metric": repr(float(trainer.best_metric)),
        },
        "lrs": [float(group["lr"]) for group in optimizer.param_groups] if optimizer is not None else None,
        "param_group_names": [group.get("name") for group in optimizer.param_groups] if optimizer is not None else None,
        "freeze_mask": {name: bool(param.requires_grad) for name, param in trainer.model.named_parameters()},
        "config_signature": trainer.config_signature,
    }


def _boundary_warmup_factor(trainer: UnifiedTrainer, interval: Any) -> float:
    """The multiplier a freshly built interval scheduler applies at its first step.

    Derived from the configured curve exactly as
    ``setup_interval_optimizer_and_scheduler`` derives it, so the boundary LR is
    asserted against the schedule the run actually has -- not against the raw
    configured LR, which the warmup never emits at step 0.
    """

    schedule = trainer.config.training.schedule
    num_batches = len(trainer.get_loaders(interval)[0])
    updates_per_epoch = schedule.steps_per_epoch(interval, num_batches)
    total_steps = max(updates_per_epoch * interval.num_epochs, 1)
    curve = trainer.config.training.lr_curve
    warmup_steps = min(int(math.ceil(curve.warmup_epochs * updates_per_epoch)), total_steps)
    if warmup_steps:
        return max(0.0 / float(warmup_steps), 1e-8)
    min_ratio = float(curve.min_ratio)
    return min_ratio + (1.0 - min_ratio) * 1.0


def _link_tree(source: Path, destination: Path) -> None:
    """Copy a committed checkpoint tree by hard link.

    Committed checkpoints are written through an atomic replace, so their inodes
    are never mutated in place; linking preserves the exact committed bytes
    (including the immutable ``selected_best/`` snapshots) at no storage cost.
    """

    destination.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.rglob("*")):
        target = destination / entry.relative_to(source)
        if entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists():
            continue
        try:
            os.link(entry, target)
        except OSError:
            shutil.copy2(entry, target)


# ---------------------------------------------------------------------------
# Baseline: one uninterrupted 6-epoch curriculum
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("exact_resume")
    snapshots = root / "snapshots"
    snapshots.mkdir()

    signature_before = source_content_signature().get("source_content_signature")

    trainer = _build_trainer(root, "baseline")
    output_dir = Path(trainer.config.checkpoint.output_dir)
    report_dir = Path(trainer.config.logging.report_dir)
    per_epoch: dict[int, dict[str, Any]] = {}

    def _snapshotting_save(path: Any, model: Any, **kwargs: Any) -> Path:
        written = Path(_real_save_checkpoint(path, model, **kwargs))
        if written.name == "last.pt":
            epoch = int(kwargs["epoch"])
            _link_tree(output_dir, snapshots / f"epoch_{epoch}")
            per_epoch[epoch] = _capture(trainer)
        return written

    def _snapshotting_report_write(path: Any, obj: Any, **kwargs: Any) -> Any:
        """Delegate to the real atomic writer, then copy the durable bytes.

        The snapshot is a copy of the file the trainer actually committed at
        that epoch -- never a reconstruction of the final report, whose epoch
        list and status would be the wrong ones for a mid-run state.
        """

        result = _real_atomic_write_json(path, obj, **kwargs)
        written = Path(path)
        if written.name == "pipeline_report.json":
            with open(written, "r", encoding="utf-8") as handle:
                committed = json.load(handle)
            epoch = committed.get("last_checkpoint_committed_epoch")
            if isinstance(epoch, int) and epoch in SNAPSHOT_EPOCHS:
                destination = snapshots / f"epoch_{epoch}"
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(written, destination / "pipeline_report.json")
        return result

    unified_trainer_module.save_checkpoint = _snapshotting_save
    unified_trainer_module.atomic_write_json = _snapshotting_report_write
    try:
        report = trainer.train(start_epoch=0)
    finally:
        unified_trainer_module.save_checkpoint = _real_save_checkpoint
        unified_trainer_module.atomic_write_json = _real_atomic_write_json

    signature_after = source_content_signature().get("source_content_signature")

    calibration_path = report_dir / "calibration.json"
    calibration = None
    if calibration_path.exists():
        with open(calibration_path, "r", encoding="utf-8") as handle:
            calibration = json.load(handle)

    return {
        "root": root,
        "snapshots": snapshots,
        "trainer": trainer,
        "report": copy.deepcopy(report),
        "final": _capture(trainer),
        "per_epoch": per_epoch,
        "calibration": calibration,
        "loader_cardinalities": [
            len(trainer.get_loaders(interval)[0])
            for interval in trainer.config.training.schedule.intervals
        ],
        "signature_before": signature_before,
        "signature_after": signature_after,
    }


def _require_stable_source(baseline: dict[str, Any]) -> None:
    """Exact resume is *defined* against an unchanged source tree.

    ``resume_from_checkpoint`` refuses a checkpoint produced by different source
    content, which is correct.  If the working tree is edited while this module
    runs (a concurrent implementation session, say), the premise of the whole
    comparison is gone and a failure here would say nothing about resume.
    """

    if baseline["signature_before"] != baseline["signature_after"]:
        pytest.skip(
            "source tree was modified while the baseline curriculum was training "
            f"({baseline['signature_before']} -> {baseline['signature_after']}); "
            "exact-resume comparison requires a quiescent working tree"
        )


def test_source_tree_was_quiescent_while_the_baseline_trained(baseline: dict[str, Any]) -> None:
    """Fail loudly if the working tree changed under the baseline run.

    ``resume_from_checkpoint`` refuses a checkpoint whose ``source_signature``
    differs from the live source content, and that refusal is correct.  When a
    concurrent session edits ``src/`` or ``configs/`` mid-run the resume
    comparisons skip, because their premise is gone -- so this test exists to
    make that condition visible as a failure rather than a quiet skip.
    """

    assert baseline["signature_before"] == baseline["signature_after"], (
        "source tree was modified while the baseline curriculum was training "
        f"({baseline['signature_before']} -> {baseline['signature_after']}); "
        "the exact-resume comparisons were skipped for this reason"
    )


@pytest.fixture(scope="module")
def resumed_runs(baseline: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Resume the single baseline from each committed epoch 1..5.

    Each resume starts from the *actually committed* checkpoint tree for that
    epoch and inherits that epoch's committed pipeline report, which is what a
    real interrupted run would find on disk.
    """

    _require_stable_source(baseline)

    results: dict[int, dict[str, Any]] = {}
    for epoch in SNAPSHOT_EPOCHS:
        trainer = _build_trainer(baseline["root"], f"resume_{epoch}")
        report_dir = Path(trainer.config.logging.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            baseline["snapshots"] / f"epoch_{epoch}" / "pipeline_report.json",
            report_dir / "pipeline_report.json",
        )

        source_checkpoint = baseline["snapshots"] / f"epoch_{epoch}" / "last.pt"
        start_epoch = trainer.resume_from_checkpoint(source_checkpoint)
        state_at_resume = _capture(trainer)
        report = trainer.train(start_epoch=start_epoch)

        calibration_path = report_dir / "calibration.json"
        calibration = None
        if calibration_path.exists():
            with open(calibration_path, "r", encoding="utf-8") as handle:
                calibration = json.load(handle)

        results[epoch] = {
            "trainer": trainer,
            "start_epoch": start_epoch,
            "state_at_resume": state_at_resume,
            "report": copy.deepcopy(report),
            "final": _capture(trainer),
            "calibration": calibration,
        }
    return results


# ---------------------------------------------------------------------------
# 1. The baseline itself must be the shape the regression claims to test
# ---------------------------------------------------------------------------


def test_baseline_curriculum_is_a_real_completed_three_interval_run(baseline: dict[str, Any]) -> None:
    report = baseline["report"]
    assert report["completed"] is True, report.get("incomplete_reason")
    assert [int(row["epoch"]) for row in report["epochs"]] == list(range(1, TOTAL_EPOCHS + 1))
    assert [row["interval"] for row in report["epochs"]] == [
        "annotation_bootstrap",
        "annotation_bootstrap",
        "auditor_counterfactual",
        "auditor_counterfactual",
        "joint_finetune",
        "joint_finetune",
    ]
    # Varied loader cardinality is what makes a silently restarted schedule visible.
    assert baseline["loader_cardinalities"] == [3, 2, 3]
    assert len(set(baseline["loader_cardinalities"])) > 1
    assert baseline["final"]["counters"]["completed_epochs"] == TOTAL_EPOCHS
    # A selected best must exist, and it must come from the gated joint interval.
    assert baseline["final"]["selection"]["best_epoch"] in (5, 6)


def test_baseline_snapshots_are_legitimate_resumable_completed_epochs(baseline: dict[str, Any]) -> None:
    for epoch in SNAPSHOT_EPOCHS:
        checkpoint = baseline["snapshots"] / f"epoch_{epoch}" / "last.pt"
        assert checkpoint.is_file(), f"missing committed checkpoint for epoch {epoch}"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert int(payload["epoch"]) == epoch
        assert payload["resumable"] is True
        assert payload["incomplete_epoch"] is False
        assert payload["validation_complete"] is True
        assert "rng_state" in payload and set(payload["rng_state"]) >= {"python", "numpy", "torch"}
        assert payload["loader_cardinality"] in (2, 3)

        # W4 commit schema: last.pt names its own role, carries the run's W&B
        # identity, and derives its selection metadata from the committed
        # immutable reference rather than from live trainer fields.
        assert payload["role"] == "last"
        assert isinstance(payload.get("wandb_identity"), dict)
        reference = payload.get(COMMITTED_BEST_REFERENCE_KEY)
        if epoch >= 5:
            assert isinstance(reference, dict), f"epoch {epoch} committed no best reference"
            assert reference["sha256"] == payload["best_checkpoint_hash"]
            assert int(reference["epoch"]) == int(payload["best_epoch"]) == 5
            assert payload["best_metric"] == pytest.approx(float(reference["metric"]))
        else:
            # Selection is gated to the threshold_gate interval from epoch 5, so
            # an earlier checkpoint must claim no selected best at all.
            assert reference is None
            assert payload["best_epoch"] is None
            assert payload["best_checkpoint_hash"] is None


def test_multi_worker_augmenting_loader_is_actually_exercised(baseline: dict[str, Any]) -> None:
    """The augmenting interval must run through a real shuffled multi-worker loader.

    If the platform's start method cannot carry the dataset into workers, that is
    reported as missing coverage rather than papered over.
    """

    trainer = baseline["trainer"]
    augmenting = [i for i in trainer.config.training.schedule.intervals if i.augment]
    assert augmenting, "the curriculum must contain an augmenting interval"
    for interval in augmenting:
        train_loader, _ = trainer.get_loaders(interval)
        assert train_loader.num_workers == AUGMENT_WORKERS, (
            f"MISSING COVERAGE: interval {interval.name!r} did not run with "
            f"num_workers={AUGMENT_WORKERS}"
        )
        assert getattr(train_loader, "persistent_workers", False) is False
        assert train_loader.dataset.augment is True
        # Shuffled: a RandomSampler, not a SequentialSampler.
        assert type(train_loader.sampler).__name__ == "RandomSampler"


# ---------------------------------------------------------------------------
# 2. The core claim: resume reproduces the uninterrupted run exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epoch", SNAPSHOT_EPOCHS)
def test_resume_reaches_the_uninterrupted_final_state_exactly(
    baseline: dict[str, Any], resumed_runs: dict[int, dict[str, Any]], epoch: int
) -> None:
    resumed = resumed_runs[epoch]["final"]
    expected = baseline["final"]

    assert resumed["model"] == expected["model"], f"model weights diverged after resume at epoch {epoch}"
    assert resumed["optimizer"] == expected["optimizer"], f"optimizer state diverged after resume at epoch {epoch}"
    assert resumed["scheduler"] == expected["scheduler"], f"scheduler state diverged after resume at epoch {epoch}"
    assert resumed["scaler"] == expected["scaler"], f"scaler state diverged after resume at epoch {epoch}"
    assert resumed["rng"] == expected["rng"], f"RNG streams diverged after resume at epoch {epoch}"
    assert resumed["counters"] == expected["counters"], f"counters diverged after resume at epoch {epoch}"
    assert resumed["selection"] == expected["selection"], f"selection diverged after resume at epoch {epoch}"
    assert resumed["lrs"] == expected["lrs"], f"learning rates diverged after resume at epoch {epoch}"
    assert resumed["freeze_mask"] == expected["freeze_mask"], f"freeze mask diverged after resume at epoch {epoch}"


@pytest.mark.parametrize("epoch", SNAPSHOT_EPOCHS)
def test_state_at_the_moment_of_resume_matches_the_committed_epoch(
    baseline: dict[str, Any], resumed_runs: dict[int, dict[str, Any]], epoch: int
) -> None:
    """Resuming restores the epoch that was committed, without fabricating counters."""

    resumed = resumed_runs[epoch]
    at_resume = resumed["state_at_resume"]
    committed = baseline["per_epoch"][epoch]

    assert resumed["start_epoch"] == epoch
    assert at_resume["model"] == committed["model"]
    assert at_resume["rng"] == committed["rng"]
    assert at_resume["counters"]["global_step"] == committed["counters"]["global_step"]
    assert at_resume["counters"]["optimizer_step"] == committed["counters"]["optimizer_step"]
    assert at_resume["counters"]["completed_epochs"] == epoch
    assert at_resume["counters"]["last_checkpoint_committed_epoch"] == epoch


@pytest.mark.parametrize("epoch", SNAPSHOT_EPOCHS)
def test_resume_restarts_neither_the_schedule_nor_the_optimizer_state(
    baseline: dict[str, Any], resumed_runs: dict[int, dict[str, Any]], epoch: int
) -> None:
    """Freeze masks, per-group LRs, rollout mode and interval index continue.

    At a non-boundary epoch the optimizer/scheduler/scaler must be *restored*;
    at an interval boundary they must be freshly reset, exactly as the
    uninterrupted run does at that same epoch.  Both are checked against the
    uninterrupted run's own state, so neither behaviour can be asserted loosely.
    """

    trainer = resumed_runs[epoch]["trainer"]
    schedule = trainer.config.training.schedule
    at_resume = resumed_runs[epoch]["state_at_resume"]

    next_interval = schedule.get_interval(epoch)
    assert at_resume["counters"]["current_interval_index"] == schedule.get_interval_index(epoch)
    assert at_resume["param_group_names"] is not None and at_resume["param_group_names"], (
        "resume produced no optimizer parameter groups"
    )

    # Freeze mask must match the interval that is about to run, not interval 0.
    for name, requires_grad in at_resume["freeze_mask"].items():
        is_auditor = name.startswith("auditor")
        if next_interval.trainable == "annotation":
            assert requires_grad is (not is_auditor), name
        elif next_interval.trainable == "auditor":
            assert requires_grad is is_auditor, name
        else:
            assert requires_grad is True, name

    if not schedule.is_interval_start(epoch):
        # Mid-interval: the committed optimizer/scheduler/scaler state is restored.
        committed = baseline["per_epoch"][epoch]
        assert at_resume["optimizer"] == committed["optimizer"]
        assert at_resume["scheduler"] == committed["scheduler"]
        assert at_resume["scaler"] == committed["scaler"]
        assert at_resume["lrs"] == committed["lrs"]
    else:
        # Boundary: the interval declares reset_optimizer, so the resumed run must
        # rebuild the optimizer from this interval's configured LRs rather than
        # carrying the previous interval's state forward.  The absolute LR at the
        # boundary is whatever the fresh warmup schedule dictates, so the check is
        # that every group sits at the same fraction of its own configured LR --
        # a carried-over group would not.
        assert next_interval.reset_optimizer is True
        configured = {
            "encoder": next_interval.encoder_lr,
            "annotation_heads": next_interval.annotation_lr,
            "auditor": next_interval.auditor_lr,
        }
        factor = _boundary_warmup_factor(trainer, next_interval)
        for group_name, lr in zip(at_resume["param_group_names"], at_resume["lrs"]):
            base = configured[group_name]
            assert base > 0.0, group_name
            assert lr == pytest.approx(base * factor, rel=1e-12), group_name

        committed = baseline["per_epoch"][epoch]
        assert at_resume["optimizer"] != committed["optimizer"], (
            "optimizer state was carried across an interval boundary that declares reset_optimizer"
        )


# ---------------------------------------------------------------------------
# 3. Configuration and lineage must not drift across the resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epoch", SNAPSHOT_EPOCHS)
def test_resume_does_not_extend_or_rewrite_the_execution_config(
    baseline: dict[str, Any], resumed_runs: dict[int, dict[str, Any]], epoch: int
) -> None:
    trainer = resumed_runs[epoch]["trainer"]
    assert trainer.config_signature == baseline["trainer"].config_signature
    assert trainer.config.training.schedule.total_epochs == TOTAL_EPOCHS
    assert [i.name for i in trainer.config.training.schedule.intervals] == [
        i.name for i in baseline["trainer"].config.training.schedule.intervals
    ]
    saved_config = torch.load(
        baseline["snapshots"] / f"epoch_{epoch}" / "last.pt", map_location="cpu", weights_only=True
    )["config"]
    unified_trainer_module.compare_execution_configs(saved_config, trainer.config.to_dict())


def test_resume_retains_the_immutable_selected_best_reference(
    baseline: dict[str, Any], resumed_runs: dict[int, dict[str, Any]]
) -> None:
    """A resume after selection must carry the selected-best bytes forward."""

    epoch = 5
    snapshot_dir = baseline["snapshots"] / f"epoch_{epoch}"
    selected = sorted((snapshot_dir / "selected_best").glob("*.pt"))
    assert selected, "epoch 5 committed no immutable selected-best snapshot"
    committed_hash = hashlib.sha256(selected[0].read_bytes()).hexdigest()

    trainer = resumed_runs[epoch]["trainer"]
    output_dir = Path(trainer.config.checkpoint.output_dir)
    relocated = sorted((output_dir / "selected_best").glob("*.pt"))
    assert relocated, "resume did not retain a selected-best snapshot in the new output dir"
    assert committed_hash in {hashlib.sha256(p.read_bytes()).hexdigest() for p in relocated}


# ---------------------------------------------------------------------------
# 4. Calibration and epoch history after a resumed, completed curriculum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epoch", SNAPSHOT_EPOCHS)
def test_resumed_run_completes_and_calibrates_identically(
    baseline: dict[str, Any], resumed_runs: dict[int, dict[str, Any]], epoch: int
) -> None:
    """Real post-training calibration, not a stub, must land on the same tau."""

    resumed = resumed_runs[epoch]
    assert resumed["report"]["completed"] is True, resumed["report"].get("incomplete_reason")
    assert baseline["calibration"] is not None, "baseline produced no calibration artifact"
    assert resumed["calibration"] is not None, f"resume at epoch {epoch} produced no calibration artifact"
    assert resumed["calibration"]["tau_accept"] == pytest.approx(baseline["calibration"]["tau_accept"])
    assert resumed["report"]["calibration"]["calibrated_tau"] == pytest.approx(
        baseline["report"]["calibration"]["calibrated_tau"]
    )


@pytest.mark.parametrize("epoch", SNAPSHOT_EPOCHS)
def test_resumed_final_report_preserves_the_completed_epoch_history(
    baseline: dict[str, Any], resumed_runs: dict[int, dict[str, Any]], epoch: int
) -> None:
    """The finished run's report must describe all six epochs, not just the tail.

    The resume inherits the pipeline report the interrupted run had already
    committed, so epochs 1..k are on disk before training restarts.  A final
    report that lists only epochs k+1..6 has dropped committed research history.
    """

    report = resumed_runs[epoch]["report"]
    assert [int(row["epoch"]) for row in report["epochs"]] == list(range(1, TOTAL_EPOCHS + 1))
    assert report["completed_epochs"] == TOTAL_EPOCHS


# ---------------------------------------------------------------------------
# 5. Bounded smoke checkpoints are never resumable
# ---------------------------------------------------------------------------


def test_max_steps_bounded_checkpoint_is_refused_for_resume(tmp_path: Path) -> None:
    """A ``max_steps``-truncated run writes a non-resumable last.pt."""

    trainer = _build_trainer(tmp_path, "bounded", workers_for_augment=0)
    trainer.train(start_epoch=0, max_steps=1)

    bounded = Path(trainer.config.checkpoint.output_dir) / "last.pt"
    payload = torch.load(bounded, map_location="cpu", weights_only=True)
    assert payload["resumable"] is False or payload["incomplete_epoch"] is True

    resumer = _build_trainer(tmp_path, "bounded_resume", workers_for_augment=0)
    with pytest.raises(ValueError, match=r"'resumable' is False, expected True"):
        resumer.resume_from_checkpoint(bounded)
